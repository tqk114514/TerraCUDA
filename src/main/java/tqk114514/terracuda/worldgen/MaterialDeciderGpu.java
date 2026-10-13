package tqk114514.terracuda.worldgen;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.NoiseSettings;

import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.KernelArguments;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;

/**
 * G2a: the material rules' per-block decision on the device. The aquifer's neighbour search,
 * the barrier pressures and the veinifier run in one kernel over every block of a chunk, on
 * the grid tables and per-block values the CPU still supplies — the milestone shape, where
 * parity is proven before the grid half itself moves (that is G2b).
 *
 * <p>The kernel's decisions are the small enum {@code MaterialRules.DECISION_*} plus the default
 * and air ids; {@link #stateOf(int, BlockState)} maps them back to block states the way
 * {@code applyRules} consumes the CPU rule's answer.
 */
public final class MaterialDeciderGpu implements AutoCloseable {

    public static final String KERNEL_NAME = "terracuda_material_decide";
    private static final int BLOCK_SIZE = 128;

    /** Grows past any chunk's grid: the anchors span at most four cells plus two of margin. */
    private static final int MAX_GRID_XZ = 8;
    private static final int MAX_GRID_Y = 40;

    private final CudaContext context;
    private final NoiseSettings geometry;
    private final int count;
    private final int seaLevel;
    private final long oreSeedLo;
    private final long oreSeedHi;
    private final long aquiferSeedLo;
    private final long aquiferSeedHi;
    private final boolean aquifersEnabled;
    private final boolean oreVeinsEnabled;

    private final CudaContext.DeviceBuffer barrierBlob;
    private final CudaContext.DeviceBuffer barrierOffsets;
    private final int barrierCount;
    private final int barrierRoot;
    private final DensityInterpreter barrierInterpreter;
    private final DensityProgram barrierProgram;

    private final CudaContext.DeviceBuffer veinGapBlob;
    private final CudaContext.DeviceBuffer veinGapOffsets;
    private final int veinGapCount;
    private final int veinGapRoot;
    private final DensityInterpreter veinGapInterpreter;
    private final DensityProgram veinGapProgram;

    private final int programScratch;

    private final CudaContext.DeviceBuffer density;
    private final CudaContext.DeviceBuffer veinToggle;
    private final CudaContext.DeviceBuffer veinRidged;
    private final CudaContext.DeviceBuffer cellLevels;
    private final CudaContext.DeviceBuffer cellTypes;
    private final CudaContext.DeviceBuffer barrierFlat;
    private final CudaContext.DeviceBuffer veinGapFlat;
    private final CudaContext.DeviceBuffer scratch;
    private final CudaContext.DeviceBuffer decisions;
    private final CudaContext.DeviceBuffer fluidFlags;

    private final Arena hostArena = Arena.ofShared();
    private final MemorySegment hostDensity;
    private final MemorySegment hostVeinToggle;
    private final MemorySegment hostVeinRidged;
    private final MemorySegment hostCellLevels;
    private final MemorySegment hostCellTypes;
    private final MemorySegment hostDecisions;
    private final MemorySegment hostFluidFlags;
    private final MemorySegment hostFlatTable;

    private boolean closed;

    public MaterialDeciderGpu(CudaContext context, NoiseSettings geometry, int seaLevel,
            long oreSeedLo, long oreSeedHi, long aquiferSeedLo, long aquiferSeedHi,
            boolean aquifersEnabled, boolean oreVeinsEnabled,
            DensityInterpreter barrierInterpreter, DensityProgram barrierProgram,
            DensityInterpreter veinGapInterpreter, DensityProgram veinGapProgram) {
        this.context = context;
        this.geometry = geometry;
        this.count = 16 * 16 * geometry.height();
        this.seaLevel = seaLevel;
        this.oreSeedLo = oreSeedLo;
        this.oreSeedHi = oreSeedHi;
        this.aquiferSeedLo = aquiferSeedLo;
        this.aquiferSeedHi = aquiferSeedHi;
        this.aquifersEnabled = aquifersEnabled;
        this.oreVeinsEnabled = oreVeinsEnabled;
        this.barrierInterpreter = barrierInterpreter;
        this.barrierProgram = barrierProgram;
        this.veinGapInterpreter = veinGapInterpreter;
        this.veinGapProgram = veinGapProgram;

        DensityProgramImage barrierImage = DensityProgramImage.of(barrierProgram);
        this.barrierBlob = uploadBytes(barrierImage.blob());
        this.barrierOffsets = uploadLongs(barrierImage.offsets());
        this.barrierCount = barrierImage.instructionCount();
        this.barrierRoot = barrierImage.root();

        DensityProgramImage veinGapImage = DensityProgramImage.of(veinGapProgram);
        this.veinGapBlob = uploadBytes(veinGapImage.blob());
        this.veinGapOffsets = uploadLongs(veinGapImage.offsets());
        this.veinGapCount = veinGapImage.instructionCount();
        this.veinGapRoot = veinGapImage.root();

        this.programScratch = Math.max(this.barrierCount, this.veinGapCount);

        this.density = context.allocate((long) this.count * Double.BYTES);
        this.veinToggle = context.allocate((long) this.count * Double.BYTES);
        this.veinRidged = context.allocate((long) this.count * Double.BYTES);
        this.cellLevels = context.allocate((long) MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ * 4L);
        this.cellTypes = context.allocate((long) MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ * 4L);
        this.barrierFlat = context.allocate(25L * Double.BYTES);
        this.veinGapFlat = context.allocate(25L * Double.BYTES);
        this.scratch = context.allocate((long) this.count * this.programScratch * Double.BYTES);
        this.decisions = context.allocate((long) this.count * 4L);
        this.fluidFlags = context.allocate((long) this.count * 4L);

        this.hostDensity = this.hostArena.allocate((long) this.count * Double.BYTES);
        this.hostVeinToggle = this.hostArena.allocate((long) this.count * Double.BYTES);
        this.hostVeinRidged = this.hostArena.allocate((long) this.count * Double.BYTES);
        this.hostCellLevels = this.hostArena.allocate(
                (long) MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ * 4L);
        this.hostCellTypes = this.hostArena.allocate(
                (long) MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ * 4L);
        this.hostDecisions = this.hostArena.allocate((long) this.count * 4L);
        this.hostFluidFlags = this.hostArena.allocate((long) this.count * 4L);
        this.hostFlatTable = this.hostArena.allocate(25L * Double.BYTES);
    }

    /**
     * One chunk's decisions on the device.
     *
     * <p>The density must already carry the beard — the same value {@code applyRules} hands the
     * CPU rules. The grid tables come from {@code rules}, which the caller builds the way
     * {@code applyRules} does; the flat-cache tables for the barrier and vein-gap entries are
     * derived per chunk from the same wrapping decision the CPU path makes.
     *
     * @return the per-block decision ids and the aquifer's fluid-tick flags
     */
    public Decisions decide(int chunkX, int chunkZ, double[] density, double[] veinToggle,
            double[] veinRidged, MaterialRules rules) {
        if (this.closed) {
            throw new IllegalStateException("the decider is closed");
        }
        MaterialRules.GridTables tables = rules.exportGrid();
        if (tables.levels().length > MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ) {
            throw new IllegalStateException("grid of " + tables.levels().length
                    + " cells exceeds the decider's " + MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ);
        }

        copyDoubles(this.hostDensity, density);
        this.context.copyToDevice(this.hostDensity, this.density);
        if (this.oreVeinsEnabled) {
            copyDoubles(this.hostVeinToggle, veinToggle);
            this.context.copyToDevice(this.hostVeinToggle, this.veinToggle);
            copyDoubles(this.hostVeinRidged, veinRidged);
            this.context.copyToDevice(this.hostVeinRidged, this.veinRidged);
        }

        copyInts(this.hostCellLevels, tables.levels());
        this.context.copyToDevice(this.hostCellLevels, this.cellLevels);
        copyInts(this.hostCellTypes, tables.types());
        this.context.copyToDevice(this.hostCellTypes, this.cellTypes);

        // A wrapped entry reads its five-by-five table; an unwrapped one is told there is no
        // table (a null pointer) and evaluates its program instead, so nothing uploads.
        double[] barrierTable = MaterialRules.flatCacheTableOrNull(this.barrierInterpreter,
                this.barrierProgram, chunkX, chunkZ);
        if (barrierTable != null) {
            copyDoubles(this.hostFlatTable, barrierTable);
            this.context.copyToDevice(this.hostFlatTable, this.barrierFlat);
        }
        double[] veinGapTable = MaterialRules.flatCacheTableOrNull(this.veinGapInterpreter,
                this.veinGapProgram, chunkX, chunkZ);
        if (veinGapTable != null) {
            copyDoubles(this.hostFlatTable, veinGapTable);
            this.context.copyToDevice(this.hostFlatTable, this.veinGapFlat);
        }

        int firstQuartX = chunkX * 16 >> 2;
        int firstQuartZ = chunkZ * 16 >> 2;
        long barrierFlatAddress = barrierTable != null ? this.barrierFlat.address() : 0L;
        long veinGapFlatAddress = veinGapTable != null ? this.veinGapFlat.address() : 0L;

        try (KernelArguments arguments = new KernelArguments(40)) {
            arguments.addDevicePointer(this.density.address())
                    .addDevicePointer(this.oreVeinsEnabled ? this.veinToggle.address() : 0L)
                    .addDevicePointer(this.oreVeinsEnabled ? this.veinRidged.address() : 0L)
                    .addDevicePointer(this.cellLevels.address())
                    .addDevicePointer(this.cellTypes.address())
                    .addInt(tables.minGridX()).addInt(tables.minGridY()).addInt(tables.minGridZ())
                    .addInt(tables.gridSizeX()).addInt(tables.gridSizeY()).addInt(tables.gridSizeZ())
                    .addInt(tables.skipSamplingAboveY())
                    .addInt(this.seaLevel)
                    .addDevicePointer(this.barrierBlob.address())
                    .addDevicePointer(this.barrierOffsets.address())
                    .addInt(this.barrierCount).addInt(this.barrierRoot)
                    .addDevicePointer(barrierFlatAddress)
                    .addDevicePointer(this.veinGapBlob.address())
                    .addDevicePointer(this.veinGapOffsets.address())
                    .addInt(this.veinGapCount).addInt(this.veinGapRoot)
                    .addDevicePointer(veinGapFlatAddress)
                    .addLong(this.oreSeedLo).addLong(this.oreSeedHi)
                    .addLong(this.aquiferSeedLo).addLong(this.aquiferSeedHi)
                    .addInt(firstQuartX).addInt(firstQuartZ)
                    .addDevicePointer(this.scratch.address())
                    .addInt(this.programScratch)
                    .addDevicePointer(this.decisions.address())
                    .addDevicePointer(this.fluidFlags.address())
                    .addInt(chunkX * 16).addInt(this.geometry.minY()).addInt(chunkZ * 16)
                    .addInt(this.geometry.height())
                    .addInt(this.aquifersEnabled ? 1 : 0)
                    .addInt(this.oreVeinsEnabled ? 1 : 0)
                    .addInt(this.count);
            int grid = (this.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(KERNEL_NAME, grid, 1, 1, BLOCK_SIZE, 1, 1, arguments);
        }
        this.context.synchronize();

        this.context.copyFromDevice(this.decisions, this.hostDecisions);
        this.context.copyFromDevice(this.fluidFlags, this.hostFluidFlags);
        int[] states = new int[this.count];
        int[] ticks = new int[this.count];
        MemorySegment.copy(this.hostDecisions, ValueLayout.JAVA_INT, 0L, states, 0, this.count);
        MemorySegment.copy(this.hostFluidFlags, ValueLayout.JAVA_INT, 0L, ticks, 0, this.count);
        return new Decisions(states, ticks);
    }

    /** Maps a kernel decision back to the block state the CPU rules would have answered with. */
    public static BlockState stateOf(int decision, BlockState defaultBlock) {
        return switch (decision) {
            case 0 -> defaultBlock;
            case MaterialRules.DECISION_WATER -> Blocks.WATER.defaultBlockState();
            case MaterialRules.DECISION_LAVA -> Blocks.LAVA.defaultBlockState();
            case 3 -> Blocks.COPPER_ORE.defaultBlockState();
            case 4 -> Blocks.RAW_COPPER_BLOCK.defaultBlockState();
            case 5 -> Blocks.GRANITE.defaultBlockState();
            case 6 -> Blocks.DEEPSLATE_IRON_ORE.defaultBlockState();
            case 7 -> Blocks.RAW_IRON_BLOCK.defaultBlockState();
            case 8 -> Blocks.TUFF.defaultBlockState();
            case 9 -> Blocks.AIR.defaultBlockState();
            default -> throw new IllegalStateException("unknown decision id " + decision);
        };
    }

    private CudaContext.DeviceBuffer uploadBytes(byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(bytes.length);
            MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, 0, bytes.length);
            CudaContext.DeviceBuffer buffer = this.context.allocate(bytes.length);
            this.context.copyToDevice(segment, buffer);
            return buffer;
        }
    }

    private CudaContext.DeviceBuffer uploadLongs(long[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocateFrom(ValueLayout.JAVA_LONG, values);
            CudaContext.DeviceBuffer buffer = this.context.allocate(segment.byteSize());
            this.context.copyToDevice(segment, buffer);
            return buffer;
        }
    }

    private void copyDoubles(MemorySegment target, double[] values) {
        MemorySegment.copy(values, 0, target, ValueLayout.JAVA_DOUBLE, 0, values.length);
    }

    private static void copyInts(MemorySegment target, int[] values) {
        MemorySegment.copy(values, 0, target, ValueLayout.JAVA_INT, 0, values.length);
    }

    /** The device's answer for one chunk. */
    public record Decisions(int[] states, int[] fluidTicks) {
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        if (this.context.isClosed()) {
            this.hostArena.close();
            return;
        }
        for (CudaContext.DeviceBuffer buffer : new CudaContext.DeviceBuffer[] {
                this.density, this.veinToggle, this.veinRidged, this.cellLevels, this.cellTypes, this.barrierFlat, this.veinGapFlat,
                this.scratch, this.decisions, this.fluidFlags, this.barrierBlob,
                this.barrierOffsets, this.veinGapBlob, this.veinGapOffsets}) {
            this.context.free(buffer);
        }
        this.hostArena.close();
    }
}
