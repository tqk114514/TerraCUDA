package tqk114514.terracuda.worldgen;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import tqk114514.terracuda.chunk.ChunkReplay;
import tqk114514.terracuda.chunk.EmittedChunk;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.gpu.DensityEvaluatorGpu;

/**
 * Produces a chunk's blocks: the density comes from the device, the material rules run here, and the
 * result is written into the chunk.
 *
 * <p>This is M3's K3 and K4 joined up, and it is a hybrid on purpose. The design doc puts the material
 * rules in the K3 kernel too, but they are branch- and table-driven rather than floating-point heavy —
 * the FLOP density is in the interpolator corners and the arithmetic above them, and both of those are
 * on the device. Moving the aquifer across would mean a per-thread copy of its grid and caches for
 * very little gain, so it stays here until something measures otherwise.
 *
 * <p>Everything is per world: the lowered programs and the device buffers are built once and reused for
 * every chunk. Not thread-safe — one instance belongs to one thread, which is why
 * {@link GpuWorldgenService} owns it from a single dedicated thread.
 */
public final class GpuChunkFiller implements AutoCloseable {

    private final NoiseGeneratorSettings settings;
    private final NoiseSettings geometry;
    private final RandomState randomState;

    private final DensityInterpreter barrierNoise;
    private final DensityInterpreter floodedness;
    private final DensityInterpreter spread;
    private final DensityInterpreter lava;
    private final DensityInterpreter erosion;
    private final DensityInterpreter depth;
    private final DensityInterpreter veinGap;
    private final DensityInterpreter preliminarySurface;

    private final DensityEvaluatorGpu densityEvaluator;
    private final DensityEvaluatorGpu veinToggleEvaluator;
    private final DensityEvaluatorGpu veinRidgedEvaluator;

    /**
     * The three per-block arrays, reused across chunks.
     *
     * <p>Each is 786 KB and there are three, so filling them fresh every chunk would hand the
     * collector about two and a half megabytes a chunk for nothing. They belong to different
     * evaluators, so all three are live at once and none can share.
     */
    private double[] densities;
    private double[] veinToggleValues;
    private double[] veinRidgedValues;

    private boolean closed;

    private GpuChunkFiller(CudaContext context, RandomState randomState, NoiseGeneratorSettings settings) {
        this.settings = settings;
        this.geometry = settings.noiseSettings();
        this.randomState = randomState;

        NoiseRouter router = randomState.router();
        DensityProgram density = DensityCompiler.lower(router.finalDensity());
        DensityProgram veinToggle = DensityCompiler.lower(router.veinToggle());
        DensityProgram veinRidged = DensityCompiler.lower(router.veinRidged());
        this.barrierNoise = lower(router.barrierNoise());
        this.floodedness = lower(router.fluidLevelFloodednessNoise());
        this.spread = lower(router.fluidLevelSpreadNoise());
        this.lava = lower(router.lavaNoise());
        this.erosion = lower(router.erosion());
        this.depth = lower(router.depth());
        this.veinGap = lower(router.veinGap());
        this.preliminarySurface = lower(router.preliminarySurfaceLevel());

        this.densityEvaluator = DensityEvaluatorGpu.forProgram(context, density, this.geometry);
        this.veinToggleEvaluator = DensityEvaluatorGpu.forProgram(context, veinToggle, this.geometry);
        this.veinRidgedEvaluator = DensityEvaluatorGpu.forProgram(context, veinRidged, this.geometry);
    }

    public static GpuChunkFiller create(CudaContext context, RandomState randomState,
            NoiseGeneratorSettings settings) {
        return new GpuChunkFiller(context, randomState, settings);
    }

    /** One chunk's blocks, indexed {@code (x * 16 + z) * height + (y - minY)}. */
    public EmittedChunk blockIds(int chunkX, int chunkZ) {
        int minY = this.geometry.minY();
        int height = this.geometry.height();
        int count = 16 * 16 * height;

        // The material rules read router entries through interpreters, and a flat cache inside one
        // of those needs to know which chunk's columns it is answering for.
        for (DensityInterpreter interpreter : new DensityInterpreter[] {
                this.barrierNoise, this.floodedness, this.spread, this.lava, this.erosion,
                this.depth, this.veinGap, this.preliminarySurface}) {
            interpreter.setChunk(chunkX * 16, chunkZ * 16);
        }

        if (this.densities == null) {
            this.densities = new double[count];
            this.veinToggleValues = new double[count];
            this.veinRidgedValues = new double[count];
        }
        double[] densities = this.densities;
        double[] veinToggleValues = this.veinToggleValues;
        double[] veinRidgedValues = this.veinRidgedValues;

        // final_density is a cache_all_in_cell marker, so the rules see the lerp3 blend; the vein
        // functions are read straight, so they take the incremental one. The difference is in the last
        // bits and InterpolationOrderTest pins it down.
        this.densityEvaluator.blocksForChunk(this.geometry, chunkX, chunkZ,
                CornerInterpolation.LERP3, densities);
        this.veinToggleEvaluator.blocksForChunk(this.geometry, chunkX, chunkZ,
                CornerInterpolation.INCREMENTAL, veinToggleValues);
        this.veinRidgedEvaluator.blocksForChunk(this.geometry, chunkX, chunkZ,
                CornerInterpolation.INCREMENTAL, veinRidgedValues);

        MaterialRules.RouterNoises noises = new MaterialRules.RouterNoises(
                MaterialRules.entry(this.barrierNoise, this.barrierNoise.program(), chunkX, chunkZ),
                MaterialRules.entry(this.floodedness, this.floodedness.program(), chunkX, chunkZ),
                MaterialRules.entry(this.spread, this.spread.program(), chunkX, chunkZ),
                MaterialRules.entry(this.lava, this.lava.program(), chunkX, chunkZ),
                MaterialRules.entry(this.erosion, this.erosion.program(), chunkX, chunkZ),
                MaterialRules.entry(this.depth, this.depth.program(), chunkX, chunkZ),
                MaterialRules.entry(this.veinGap, this.veinGap.program(), chunkX, chunkZ),
                (x, y, z) -> veinToggleValues[index(x, y, z, chunkX, chunkZ, minY, height)],
                (x, y, z) -> veinRidgedValues[index(x, y, z, chunkX, chunkZ, minY, height)]);

        MaterialRules rules = MaterialRules.forChunk(noises, surfaceLevels(), fluidPicker(),
                VanillaRandomExport.export(this.randomState.aquiferRandom()),
                VanillaRandomExport.export(this.randomState.oreRandom()), chunkX, chunkZ, this.geometry,
                this.settings.isAquifersEnabled(), this.settings.oreVeinsEnabled());

        BlockState defaultBlock = this.settings.defaultBlock();
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;
        int[] ids = new int[count];
        long[] fluidUpdates = new long[(count + 63) >>> 6];

        for (int yLocal = 0; yLocal < height; yLocal++) {
            int y = minY + yLocal;
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    int blockIndex = (x * 16 + z) * height + yLocal;
                    double value = densities[blockIndex];
                    BlockState state = rules.compute(chunkMinBlockX + x, y, chunkMinBlockZ + z, value);
                    if (state == null) {
                        state = defaultBlock;
                    }
                    ids[blockIndex] = Block.getId(state);
                    // doFill's condition, in the same place it evaluates it: inside the branch that
                    // actually writes a block.
                    if (rules.shouldScheduleFluidUpdate() && !state.getFluidState().isEmpty()) {
                        fluidUpdates[blockIndex >>> 6] |= 1L << (blockIndex & 63);
                    }
                }
            }
        }
        return new EmittedChunk(ids, fluidUpdates);
    }

    /**
     * Emits the chunk's blocks and writes them into {@code chunk}.
     *
     * <p>The sections are held for the duration, the way {@code fillFromNoise} holds them around
     * {@code doFill}: the write is not atomic, and another thread may reach the chunk while it is
     * being filled.
     */
    public void fill(ChunkAccess chunk) {
        int minY = this.geometry.minY();
        int height = this.geometry.height();
        int top = chunk.getSectionIndex(minY + height - 1);
        int bottom = chunk.getSectionIndex(minY);

        java.util.List<LevelChunkSection> held = new java.util.ArrayList<>(top - bottom + 1);
        for (int index = top; index >= bottom; index--) {
            LevelChunkSection section = chunk.getSection(index);
            section.acquire();
            held.add(section);
        }
        try {
            EmittedChunk emitted = blockIds(chunk.getPos().x(), chunk.getPos().z());
            ChunkReplay.write(chunk, emitted, minY, height);
        } finally {
            for (LevelChunkSection section : held) {
                section.release();
            }
        }
    }

    private MaterialRules.SurfaceLevels surfaceLevels() {
        DensityInterpreter interpreter = this.preliminarySurface;
        // Vanilla caches the preliminary surface level per column, and it matters: the aquifer asks
        // for the same columns over and over — thirteen sampling offsets times every grid cell — and
        // each miss is a find_top_surface, which walks the density downwards in steps of eight.
        java.util.Map<Long, Integer> cache = new java.util.HashMap<>();
        return new MaterialRules.SurfaceLevels() {
            @Override
            public int preliminarySurfaceLevel(int blockX, int blockZ) {
                // Vanilla quantises to quart positions before looking the column up.
                int quartX = (blockX >> 2) << 2;
                int quartZ = (blockZ >> 2) << 2;
                long key = (long) quartX << 32 | quartZ & 0xFFFFFFFFL;
                Integer cached = cache.get(key);
                if (cached != null) {
                    return cached;
                }
                int level = tqk114514.terracuda.math.VanillaMath.floor(
                        interpreter.evaluate(quartX, 0, quartZ));
                cache.put(key, level);
                return level;
            }

            @Override
            public int maxPreliminarySurfaceLevel(int minBlockX, int minBlockZ, int maxBlockX,
                    int maxBlockZ) {
                int max = Integer.MIN_VALUE;
                for (int blockZ = minBlockZ; blockZ <= maxBlockZ; blockZ += 4) {
                    for (int blockX = minBlockX; blockX <= maxBlockX; blockX += 4) {
                        max = Math.max(max, preliminarySurfaceLevel(blockX, blockZ));
                    }
                }
                return max;
            }
        };
    }

    private Aquifer.FluidPicker fluidPicker() {
        int seaLevel = this.settings.seaLevel();
        return (x, y, z) -> y < Math.min(-54, seaLevel)
                ? new Aquifer.FluidStatus(-54, net.minecraft.world.level.block.Blocks.LAVA.defaultBlockState())
                : new Aquifer.FluidStatus(seaLevel, net.minecraft.world.level.block.Blocks.WATER.defaultBlockState());
    }

    private static int index(int x, int y, int z, int chunkX, int chunkZ, int minY, int height) {
        return ((x - chunkX * 16) * 16 + (z - chunkZ * 16)) * height + (y - minY);
    }

    private static DensityInterpreter lower(DensityFunction function) {
        return new DensityInterpreter(DensityCompiler.lower(function));
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        for (DensityEvaluatorGpu evaluator : new DensityEvaluatorGpu[] {
                this.densityEvaluator, this.veinToggleEvaluator, this.veinRidgedEvaluator}) {
            try {
                evaluator.close();
            } catch (RuntimeException ignored) {
                // Shutting down; a failure here has nowhere useful to go.
            }
        }
    }
}
