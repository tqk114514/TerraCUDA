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
 * G2b: the material rules' whole decision on the device. Two kernels a chunk: the grid kernel
 * derives every aquifer cell's fluid status — the thirteen-offset preliminary-surface sampling,
 * the floodedness thresholds, the spread and lava noises, the erosion and depth reads with
 * their flat-cache semantics, and the sampling cutoff — and the decide kernel runs the per-block
 * half on those tables. The CPU's supply is gone: what crosses the host-device line per chunk
 * is the density and the two vein arrays in, and the decisions and fluid-tick flags out.
 *
 * <p>The router entries arrive as lowered images uploaded once. The entries vanilla wraps in a
 * flat cache carry the wrapping as a flag, and both kernels reproduce the wrapper's branch:
 * the chunk's quart columns answered from the y=0 samples, the misses exactly.
 */
public final class MaterialDeciderGpu implements AutoCloseable {

    public static final String K0_TABLE_KERNEL_NAME = "terracuda_material_k0_table";
    public static final String GRID_KERNEL_NAME = "terracuda_material_grid";
    public static final String KERNEL_NAME = "terracuda_material_decide";
    private static final int BLOCK_SIZE = 128;

    /** Grows past any chunk's grid: the anchors span at most four cells plus margin. */
    private static final int MAX_GRID_XZ = 8;
    private static final int MAX_GRID_Y = 40;
    private static final int MAX_GRID_TOTAL = MAX_GRID_XZ * MAX_GRID_Y * MAX_GRID_XZ;

    /** Aquifer's grid geometry: the anchor spacing in x/z and y, and the sampling offsets. */
    private static final int GRID_SPACING_XZ_SHIFT = 4;
    private static final int GRID_SPACING_Y = 12;
    private static final int SAMPLE_OFFSET_XZ = -5;

    /**
     * The K0 column table's worst case, from the offset tables the grid kernel bakes in: the
     * 13-offset loop reaches three 16-block grid steps back and one forward in x (48 and 16
     * blocks), one either way in z, on top of the anchors' 0..9 jitter. In quart indices the
     * reach is −12..+6 (x) and −4..+6 (z) columns past the grid span's own.
     */
    private static final int MAX_TABLE_X = ((((MAX_GRID_XZ - 1) << 4) + 25) >> 2) + 12 + 1;
    private static final int MAX_TABLE_Z = ((((MAX_GRID_XZ - 1) << 4) + 25) >> 2) + 4 + 1;

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
    private final int barrierWrapped;
    private final CudaContext.DeviceBuffer veinGapBlob;
    private final CudaContext.DeviceBuffer veinGapOffsets;
    private final int veinGapCount;
    private final int veinGapRoot;
    private final int veinGapWrapped;
    private final CudaContext.DeviceBuffer k0Blob;
    private final CudaContext.DeviceBuffer k0Offsets;
    private final int k0Count;
    private final int k0Root;
    private final CudaContext.DeviceBuffer floodednessBlob;
    private final CudaContext.DeviceBuffer floodednessOffsets;
    private final int floodednessCount;
    private final int floodednessRoot;
    private final int floodednessWrapped;
    private final CudaContext.DeviceBuffer spreadBlob;
    private final CudaContext.DeviceBuffer spreadOffsets;
    private final int spreadCount;
    private final int spreadRoot;
    private final int spreadWrapped;
    private final CudaContext.DeviceBuffer lavaBlob;
    private final CudaContext.DeviceBuffer lavaOffsets;
    private final int lavaCount;
    private final int lavaRoot;
    private final int lavaWrapped;
    private final CudaContext.DeviceBuffer erosionBlob;
    private final CudaContext.DeviceBuffer erosionOffsets;
    private final int erosionCount;
    private final int erosionRoot;
    private final int erosionWrapped;
    private final CudaContext.DeviceBuffer depthBlob;
    private final CudaContext.DeviceBuffer depthOffsets;
    private final int depthCount;
    private final int depthRoot;
    private final int depthWrapped;

    private final int programScratch;
    private final int gridScratch;

    private final CudaContext.DeviceBuffer density;
    private final CudaContext.DeviceBuffer veinToggle;
    private final CudaContext.DeviceBuffer veinRidged;
    private final CudaContext.DeviceBuffer cellLevels;
    private final CudaContext.DeviceBuffer cellTypes;
    private final CudaContext.DeviceBuffer scratch;
    private final CudaContext.DeviceBuffer gridScratchBuffer;
    private final CudaContext.DeviceBuffer decisions;
    private final CudaContext.DeviceBuffer fluidFlags;
    private final CudaContext.DeviceBuffer gridSkip;
    private final CudaContext.DeviceBuffer cellLocations;
    private final CudaContext.DeviceBuffer k0Table;
    private final CudaContext.DeviceBuffer k0TableScratch;

    private final Arena hostArena = Arena.ofShared();
    private final MemorySegment hostDensity;
    private final MemorySegment hostVeinToggle;
    private final MemorySegment hostVeinRidged;
    private final MemorySegment hostDecisions;
    private final MemorySegment hostFluidFlags;
    private final MemorySegment hostSkip;

    private boolean closed;

    public MaterialDeciderGpu(CudaContext context, NoiseSettings geometry, int seaLevel,
            long oreSeedLo, long oreSeedHi, long aquiferSeedLo, long aquiferSeedHi,
            boolean aquifersEnabled, boolean oreVeinsEnabled, GpuChunkFiller.RulesContext rules) {
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

        this.barrierBlob = uploadBlob(rules.barrierNoise.program());
        this.barrierOffsets = uploadOffsets(rules.barrierNoise.program());
        DensityProgramImage barrierImage = DensityProgramImage.of(rules.barrierNoise.program());
        this.barrierCount = barrierImage.instructionCount();
        this.barrierRoot = barrierImage.root();
        this.barrierWrapped = MaterialRules.isFlatCache(rules.barrierNoise.program()) ? 1 : 0;

        this.veinGapBlob = uploadBlob(rules.veinGap.program());
        this.veinGapOffsets = uploadOffsets(rules.veinGap.program());
        DensityProgramImage veinGapImage = DensityProgramImage.of(rules.veinGap.program());
        this.veinGapCount = veinGapImage.instructionCount();
        this.veinGapRoot = veinGapImage.root();
        this.veinGapWrapped = MaterialRules.isFlatCache(rules.veinGap.program()) ? 1 : 0;

        this.k0Blob = uploadBlob(rules.preliminarySurface.program());
        this.k0Offsets = uploadOffsets(rules.preliminarySurface.program());
        DensityProgramImage k0Image = DensityProgramImage.of(rules.preliminarySurface.program());
        this.k0Count = k0Image.instructionCount();
        this.k0Root = k0Image.root();

        this.floodednessBlob = uploadBlob(rules.floodedness.program());
        this.floodednessOffsets = uploadOffsets(rules.floodedness.program());
        DensityProgramImage floodednessImage = DensityProgramImage.of(rules.floodedness.program());
        this.floodednessCount = floodednessImage.instructionCount();
        this.floodednessRoot = floodednessImage.root();
        this.floodednessWrapped = MaterialRules.isFlatCache(rules.floodedness.program()) ? 1 : 0;

        this.spreadBlob = uploadBlob(rules.spread.program());
        this.spreadOffsets = uploadOffsets(rules.spread.program());
        DensityProgramImage spreadImage = DensityProgramImage.of(rules.spread.program());
        this.spreadCount = spreadImage.instructionCount();
        this.spreadRoot = spreadImage.root();
        this.spreadWrapped = MaterialRules.isFlatCache(rules.spread.program()) ? 1 : 0;

        this.lavaBlob = uploadBlob(rules.lava.program());
        this.lavaOffsets = uploadOffsets(rules.lava.program());
        DensityProgramImage lavaImage = DensityProgramImage.of(rules.lava.program());
        this.lavaCount = lavaImage.instructionCount();
        this.lavaRoot = lavaImage.root();
        this.lavaWrapped = MaterialRules.isFlatCache(rules.lava.program()) ? 1 : 0;

        this.erosionBlob = uploadBlob(rules.erosion.program());
        this.erosionOffsets = uploadOffsets(rules.erosion.program());
        DensityProgramImage erosionImage = DensityProgramImage.of(rules.erosion.program());
        this.erosionCount = erosionImage.instructionCount();
        this.erosionRoot = erosionImage.root();
        this.erosionWrapped = MaterialRules.isFlatCache(rules.erosion.program()) ? 1 : 0;

        this.depthBlob = uploadBlob(rules.depth.program());
        this.depthOffsets = uploadOffsets(rules.depth.program());
        DensityProgramImage depthImage = DensityProgramImage.of(rules.depth.program());
        this.depthCount = depthImage.instructionCount();
        this.depthRoot = depthImage.root();
        this.depthWrapped = MaterialRules.isFlatCache(rules.depth.program()) ? 1 : 0;

        int gridProgramScratch = Math.max(Math.max(this.k0Count, this.floodednessCount),
                Math.max(Math.max(this.spreadCount, this.lavaCount),
                        Math.max(this.erosionCount, this.depthCount)));
        this.gridScratch = gridProgramScratch;
        this.programScratch = Math.max(this.barrierCount, this.veinGapCount);

        this.density = context.allocate((long) this.count * Double.BYTES);
        this.veinToggle = context.allocate((long) this.count * Double.BYTES);
        this.veinRidged = context.allocate((long) this.count * Double.BYTES);
        this.cellLevels = context.allocate((long) MAX_GRID_TOTAL * 4L);
        this.cellTypes = context.allocate((long) MAX_GRID_TOTAL * 4L);
        this.scratch = context.allocate((long) this.count * this.programScratch * Double.BYTES);
        this.gridScratchBuffer =
                context.allocate((long) MAX_GRID_TOTAL * this.gridScratch * Double.BYTES);
        this.decisions = context.allocate((long) this.count * 4L);
        this.fluidFlags = context.allocate((long) this.count * 4L);
        this.gridSkip = context.allocate(256L);
        this.cellLocations = context.allocate((long) MAX_GRID_TOTAL * 3 * 4L);
        this.k0Table = context.allocate((long) MAX_TABLE_X * MAX_TABLE_Z * Double.BYTES);
        this.k0TableScratch = context.allocate(
                (long) MAX_TABLE_X * MAX_TABLE_Z * this.k0Count * Double.BYTES);

        this.hostDensity = this.pinnedHost((long) this.count * Double.BYTES);
        this.hostVeinToggle = this.pinnedHost((long) this.count * Double.BYTES);
        this.hostVeinRidged = this.pinnedHost((long) this.count * Double.BYTES);
        this.hostDecisions = this.pinnedHost((long) this.count * 4L);
        this.hostFluidFlags = this.pinnedHost((long) this.count * 4L);
        // Four bytes a chunk: the driver's staging bounce on a copy this small is noise, and the
        // read that follows is on the critical path regardless.
        this.hostSkip = this.hostArena.allocate(256L);
    }

    /**
     * Page-locked staging, so each per-chunk transfer is one direct DMA instead of the driver's
     * bounce through a staging buffer — the same pageable lesson the stream work paid for in the
     * evaluator, replayed here on the decide path.
     *
     * <p>The 64KB floor is what the evaluator's staging uses: it keeps each segment on its own
     * page-aligned allocation, which is what the host register accepts. The transfers slice the
     * exact size off the head because the copy functions take the segment's whole byte size.
     */
    private MemorySegment pinnedHost(long bytes) {
        MemorySegment segment = this.hostArena.allocate(Math.max(bytes, 1L << 16));
        this.context.pin(segment);
        return segment;
    }

    /**
     * One chunk's decisions from host-resident density: uploads and delegates to
     * {@link #decideDevice}. The beard path uses this — its density was folded on the host.
     *
     * @return the per-block decision ids and the aquifer's fluid-tick flags
     */
    public Decisions decide(int chunkX, int chunkZ, double[] density, double[] veinToggle,
            double[] veinRidged) {
        if (this.closed) {
            throw new IllegalStateException("the decider is closed");
        }
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;

        MemorySegment.copy(density, 0, this.hostDensity, ValueLayout.JAVA_DOUBLE, 0,
                density.length);
        this.context.copyToDevice(
                this.hostDensity.asSlice(0L, (long) density.length * Double.BYTES), this.density);
        if (this.oreVeinsEnabled) {
            MemorySegment.copy(veinToggle, 0, this.hostVeinToggle, ValueLayout.JAVA_DOUBLE, 0,
                    veinToggle.length);
            this.context.copyToDevice(
                    this.hostVeinToggle.asSlice(0L, (long) veinToggle.length * Double.BYTES),
                    this.veinToggle);
            MemorySegment.copy(veinRidged, 0, this.hostVeinRidged, ValueLayout.JAVA_DOUBLE, 0,
                    veinRidged.length);
            this.context.copyToDevice(
                    this.hostVeinRidged.asSlice(0L, (long) veinRidged.length * Double.BYTES),
                    this.veinRidged);
        }
        return decideDevice(chunkX, chunkZ, this.density.address(),
                this.oreVeinsEnabled ? this.veinToggle.address() : 0L,
                this.oreVeinsEnabled ? this.veinRidged.address() : 0L);
    }

    /**
     * One chunk's decisions from device-resident inputs: the caller passes the addresses of
     * buffers that already hold the density and the two vein arrays on the device — in practice
     * the evaluators' block results — and guarantees they stay untouched until this call
     * returns. The device thread is serial, so the next submit on those evaluators is the next
     * chunk, after this call is done. Nothing crosses the host line in either direction until
     * the decisions themselves do.
     *
     * <p>All three kernels queue on the default stream; the caller has already synchronised
     * the producers' streams (the evaluator awaits), so the data is complete before the first
     * launch.
     *
     * @return the per-block decision ids and the aquifer's fluid-tick flags
     */
    public Decisions decideDevice(int chunkX, int chunkZ, long densityAddress,
            long veinToggleAddress, long veinRidgedAddress) {
        if (this.closed) {
            throw new IllegalStateException("the decider is closed");
        }
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;
        int minGridX = (chunkMinBlockX + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT;
        int maxGridX = ((chunkMinBlockX + 15 + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT) + 1;
        int minGridY = Math.floorDiv(this.geometry.minY() + 1, GRID_SPACING_Y) - 1;
        int maxGridY = Math.floorDiv(this.geometry.minY() + this.geometry.height() + 1,
                GRID_SPACING_Y) + 1;
        int minGridZ = (chunkMinBlockZ + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT;
        int maxGridZ = ((chunkMinBlockZ + 15 + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT) + 1;
        int gridSizeX = maxGridX - minGridX + 1;
        int gridSizeY = maxGridY - minGridY + 1;
        int gridSizeZ = maxGridZ - minGridZ + 1;
        int gridTotal = gridSizeX * gridSizeY * gridSizeZ;
        if (gridTotal > MAX_GRID_TOTAL) {
            throw new IllegalStateException("grid of " + gridTotal
                    + " cells exceeds the decider's " + MAX_GRID_TOTAL);
        }

        int firstQuartX = chunkMinBlockX >> 2;
        int firstQuartZ = chunkMinBlockZ >> 2;

        // The K0 column table's bounds: the union of every quart column the grid kernel's
        // 13-offset loop and cutoff walk can ask for, derived once here so the kernel's lookups
        // never miss. The kernel evaluates each column exactly once instead of the per-cell
        // 13-offset re-evaluation the previous shape paid ~5000 times a chunk.
        int tableBaseX = ((minGridX << 4) - 48) >> 2;
        int tableSizeX = ((((maxGridX << 4) + 25) >> 2) - tableBaseX) + 1;
        int tableBaseZ = ((minGridZ << 4) - 16) >> 2;
        int tableSizeZ = ((((maxGridZ << 4) + 25) >> 2) - tableBaseZ) + 1;
        int tableTotal = tableSizeX * tableSizeZ;
        if (tableSizeX > MAX_TABLE_X || tableSizeZ > MAX_TABLE_Z) {
            throw new IllegalStateException("k0 table " + tableSizeX + "x" + tableSizeZ
                    + " exceeds the decider's " + MAX_TABLE_X + "x" + MAX_TABLE_Z);
        }

        try (KernelArguments table = new KernelArguments(10)) {
            table.addDevicePointer(this.k0Blob.address())
                    .addDevicePointer(this.k0Offsets.address())
                    .addInt(this.k0Count).addInt(this.k0Root)
                    .addInt(tableBaseX).addInt(tableBaseZ).addInt(tableSizeX).addInt(tableTotal)
                    .addDevicePointer(this.k0Table.address())
                    .addDevicePointer(this.k0TableScratch.address());
            int tableBlocks = (tableTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(K0_TABLE_KERNEL_NAME, tableBlocks, 1, 1, BLOCK_SIZE, 1, 1, table);
        }

        try (KernelArguments grid = new KernelArguments(47)) {
            grid.addInt(minGridX).addInt(minGridY).addInt(minGridZ)
                    .addInt(gridSizeX).addInt(gridSizeY).addInt(gridSizeZ)
                    .addInt(this.seaLevel)
                    .addLong(this.aquiferSeedLo).addLong(this.aquiferSeedHi)
                    .addInt(firstQuartX).addInt(firstQuartZ)
                    .addDevicePointer(this.k0Table.address())
                    .addInt(tableBaseX).addInt(tableBaseZ).addInt(tableSizeX)
                    .addDevicePointer(this.floodednessBlob.address())
                    .addDevicePointer(this.floodednessOffsets.address())
                    .addInt(this.floodednessCount).addInt(this.floodednessRoot)
                    .addInt(this.floodednessWrapped)
                    .addDevicePointer(this.spreadBlob.address())
                    .addDevicePointer(this.spreadOffsets.address())
                    .addInt(this.spreadCount).addInt(this.spreadRoot).addInt(this.spreadWrapped)
                    .addDevicePointer(this.lavaBlob.address())
                    .addDevicePointer(this.lavaOffsets.address())
                    .addInt(this.lavaCount).addInt(this.lavaRoot).addInt(this.lavaWrapped)
                    .addDevicePointer(this.erosionBlob.address())
                    .addDevicePointer(this.erosionOffsets.address())
                    .addInt(this.erosionCount).addInt(this.erosionRoot)
                    .addInt(this.erosionWrapped)
                    .addDevicePointer(this.depthBlob.address())
                    .addDevicePointer(this.depthOffsets.address())
                    .addInt(this.depthCount).addInt(this.depthRoot).addInt(this.depthWrapped)
                    .addDevicePointer(this.gridScratchBuffer.address()).addInt(this.gridScratch)
                    .addDevicePointer(this.cellLevels.address())
                    .addDevicePointer(this.cellTypes.address())
                    .addDevicePointer(this.gridSkip.address())
                    .addDevicePointer(this.cellLocations.address())
                    .addInt(gridTotal);
            int gridBlocks = (gridTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(GRID_KERNEL_NAME, gridBlocks, 1, 1, BLOCK_SIZE, 1, 1, grid);
        }

        try (KernelArguments arguments = new KernelArguments(39)) {
            arguments.addDevicePointer(densityAddress)
                    .addDevicePointer(veinToggleAddress)
                    .addDevicePointer(veinRidgedAddress)
                    .addDevicePointer(this.cellLevels.address())
                    .addDevicePointer(this.cellTypes.address())
                    .addDevicePointer(this.cellLocations.address())
                    .addInt(minGridX).addInt(minGridY).addInt(minGridZ)
                    .addInt(gridSizeX).addInt(gridSizeY).addInt(gridSizeZ)
                    .addDevicePointer(this.gridSkip.address())
                    .addInt(this.seaLevel)
                    .addDevicePointer(this.barrierBlob.address())
                    .addDevicePointer(this.barrierOffsets.address())
                    .addInt(this.barrierCount).addInt(this.barrierRoot)
                    .addInt(this.barrierWrapped)
                    .addDevicePointer(this.veinGapBlob.address())
                    .addDevicePointer(this.veinGapOffsets.address())
                    .addInt(this.veinGapCount).addInt(this.veinGapRoot)
                    .addInt(this.veinGapWrapped)
                    .addLong(this.oreSeedLo).addLong(this.oreSeedHi)
                    .addInt(firstQuartX).addInt(firstQuartZ)
                    .addDevicePointer(this.scratch.address())
                    .addInt(this.programScratch)
                    .addDevicePointer(this.decisions.address())
                    .addDevicePointer(this.fluidFlags.address())
                    .addInt(chunkMinBlockX).addInt(this.geometry.minY()).addInt(chunkMinBlockZ)
                    .addInt(this.geometry.height())
                    .addInt(this.aquifersEnabled ? 1 : 0)
                    .addInt(this.oreVeinsEnabled ? 1 : 0)
                    .addInt(this.count);
            int grid = (this.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(KERNEL_NAME, grid, 1, 1, BLOCK_SIZE, 1, 1, arguments);
        }
        this.context.synchronize();

        this.context.copyFromDevice(this.decisions,
                this.hostDecisions.asSlice(0L, (long) this.count * 4L));
        this.context.copyFromDevice(this.fluidFlags,
                this.hostFluidFlags.asSlice(0L, (long) this.count * 4L));
        int[] states = new int[this.count];
        int[] ticks = new int[this.count];
        MemorySegment.copy(this.hostDecisions, ValueLayout.JAVA_INT, 0L, states, 0, this.count);
        MemorySegment.copy(this.hostFluidFlags, ValueLayout.JAVA_INT, 0L, ticks, 0, this.count);
        return new Decisions(states, ticks);
    }

    /**
     * The grid kernel's tables for one chunk, downloaded -- the parity tests compare them
     * against the CPU's eager export cell by cell.
     */
    GridView debugGrid(int chunkX, int chunkZ) {
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;
        int minGridX = (chunkMinBlockX + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT;
        int maxGridX = ((chunkMinBlockX + 15 + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT) + 1;
        int minGridY = Math.floorDiv(this.geometry.minY() + 1, GRID_SPACING_Y) - 1;
        int maxGridY = Math.floorDiv(this.geometry.minY() + this.geometry.height() + 1,
                GRID_SPACING_Y) + 1;
        int minGridZ = (chunkMinBlockZ + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT;
        int maxGridZ = ((chunkMinBlockZ + 15 + SAMPLE_OFFSET_XZ) >> GRID_SPACING_XZ_SHIFT) + 1;
        int gridSizeX = maxGridX - minGridX + 1;
        int gridSizeY = maxGridY - minGridY + 1;
        int gridSizeZ = maxGridZ - minGridZ + 1;
        int gridTotal = gridSizeX * gridSizeY * gridSizeZ;
        int firstQuartX = chunkMinBlockX >> 2;
        int firstQuartZ = chunkMinBlockZ >> 2;
        int tableBaseX = ((minGridX << 4) - 48) >> 2;
        int tableSizeX = ((((maxGridX << 4) + 25) >> 2) - tableBaseX) + 1;
        int tableBaseZ = ((minGridZ << 4) - 16) >> 2;
        int tableSizeZ = ((((maxGridZ << 4) + 25) >> 2) - tableBaseZ) + 1;
        int tableTotal = tableSizeX * tableSizeZ;
        try (KernelArguments table = new KernelArguments(10)) {
            table.addDevicePointer(this.k0Blob.address())
                    .addDevicePointer(this.k0Offsets.address())
                    .addInt(this.k0Count).addInt(this.k0Root)
                    .addInt(tableBaseX).addInt(tableBaseZ).addInt(tableSizeX).addInt(tableTotal)
                    .addDevicePointer(this.k0Table.address())
                    .addDevicePointer(this.k0TableScratch.address());
            int tableBlocks = (tableTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(K0_TABLE_KERNEL_NAME, tableBlocks, 1, 1, BLOCK_SIZE, 1, 1, table);
        }
        try (KernelArguments grid = new KernelArguments(47)) {
            grid.addInt(minGridX).addInt(minGridY).addInt(minGridZ)
                    .addInt(gridSizeX).addInt(gridSizeY).addInt(gridSizeZ)
                    .addInt(this.seaLevel)
                    .addLong(this.aquiferSeedLo).addLong(this.aquiferSeedHi)
                    .addInt(firstQuartX).addInt(firstQuartZ)
                    .addDevicePointer(this.k0Table.address())
                    .addInt(tableBaseX).addInt(tableBaseZ).addInt(tableSizeX)
                    .addDevicePointer(this.floodednessBlob.address())
                    .addDevicePointer(this.floodednessOffsets.address())
                    .addInt(this.floodednessCount).addInt(this.floodednessRoot)
                    .addInt(this.floodednessWrapped)
                    .addDevicePointer(this.spreadBlob.address())
                    .addDevicePointer(this.spreadOffsets.address())
                    .addInt(this.spreadCount).addInt(this.spreadRoot).addInt(this.spreadWrapped)
                    .addDevicePointer(this.lavaBlob.address())
                    .addDevicePointer(this.lavaOffsets.address())
                    .addInt(this.lavaCount).addInt(this.lavaRoot).addInt(this.lavaWrapped)
                    .addDevicePointer(this.erosionBlob.address())
                    .addDevicePointer(this.erosionOffsets.address())
                    .addInt(this.erosionCount).addInt(this.erosionRoot)
                    .addInt(this.erosionWrapped)
                    .addDevicePointer(this.depthBlob.address())
                    .addDevicePointer(this.depthOffsets.address())
                    .addInt(this.depthCount).addInt(this.depthRoot).addInt(this.depthWrapped)
                    .addDevicePointer(this.gridScratchBuffer.address()).addInt(this.gridScratch)
                    .addDevicePointer(this.cellLevels.address())
                    .addDevicePointer(this.cellTypes.address())
                    .addDevicePointer(this.gridSkip.address())
                    .addDevicePointer(this.cellLocations.address())
                    .addInt(gridTotal);
            int gridBlocks = (gridTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(GRID_KERNEL_NAME, gridBlocks, 1, 1, BLOCK_SIZE, 1, 1, grid);
        }
        this.context.synchronize();
        this.context.copyFromDevice(this.gridSkip, this.hostSkip);
        int[] levels = new int[gridTotal];
        int[] types = new int[gridTotal];
        this.context.copyFromDevice(this.cellLevels,
                this.hostArena.allocate((long) gridTotal * 4L));
        this.context.copyFromDevice(this.cellTypes,
                this.hostArena.allocate((long) gridTotal * 4L));
        return new GridView(minGridX, minGridY, minGridZ, gridSizeX, gridSizeY, gridSizeZ,
                this.hostSkip.get(ValueLayout.JAVA_INT, 0L));
    }

    /** The grid kernel's bounds and cutoff; the tables stay on the device. */
    record GridView(int minGridX, int minGridY, int minGridZ,
            int gridSizeX, int gridSizeY, int gridSizeZ, int skipSamplingAboveY) {
    }

    /** The device's tables for one grid, for the parity tests. */
    record GridTables(int[] levels, int[] types) {
    }

    /** The device skip buffer, for the parity tests' trace. */
    CudaContext.DeviceBuffer debugSkipBuffer() {
        return this.gridSkip;
    }

    /** The device grid buffers, so the tests can download them. */
    CudaContext.DeviceBuffer debugLevelsBuffer() {
        return this.cellLevels;
    }

    CudaContext.DeviceBuffer debugTypesBuffer() {
        return this.cellTypes;
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

    private CudaContext.DeviceBuffer uploadBlob(DensityProgram program) {
        byte[] bytes = DensityProgramImage.of(program).blob();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(bytes.length);
            MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, 0, bytes.length);
            CudaContext.DeviceBuffer buffer = this.context.allocate(bytes.length);
            this.context.copyToDevice(segment, buffer);
            return buffer;
        }
    }

    private CudaContext.DeviceBuffer uploadOffsets(DensityProgram program) {
        long[] values = DensityProgramImage.of(program).offsets();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocateFrom(ValueLayout.JAVA_LONG, values);
            CudaContext.DeviceBuffer buffer = this.context.allocate(segment.byteSize());
            this.context.copyToDevice(segment, buffer);
            return buffer;
        }
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
        this.context.unpin(this.hostDensity);
        this.context.unpin(this.hostVeinToggle);
        this.context.unpin(this.hostVeinRidged);
        this.context.unpin(this.hostDecisions);
        this.context.unpin(this.hostFluidFlags);
        for (CudaContext.DeviceBuffer buffer : new CudaContext.DeviceBuffer[] {
                this.density, this.veinToggle, this.veinRidged, this.cellLevels, this.cellTypes,
                this.scratch, this.gridScratchBuffer, this.decisions, this.fluidFlags,
                this.gridSkip, this.cellLocations, this.k0Table, this.k0TableScratch,
                this.barrierBlob, this.barrierOffsets, this.veinGapBlob,
                this.veinGapOffsets, this.k0Blob, this.k0Offsets, this.floodednessBlob,
                this.floodednessOffsets, this.spreadBlob, this.spreadOffsets, this.lavaBlob,
                this.lavaOffsets, this.erosionBlob, this.erosionOffsets, this.depthBlob,
                this.depthOffsets}) {
            this.context.free(buffer);
        }
        this.hostArena.close();
    }
}
