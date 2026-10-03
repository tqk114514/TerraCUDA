package tqk114514.terracuda.worldgen;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import tqk114514.terracuda.chunk.ChunkReplay;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.gpu.DensityEvaluatorGpu;
import tqk114514.terracuda.gpu.InterpolatedDensity;

/**
 * Produces a chunk's blocks: the density comes from the device, the material rules run here, and the
 * result is written into the chunk.
 *
 * <p>This is M3's K3 and K4 joined up, and it is a hybrid on purpose. The design doc puts the material
 * rules in the K3 kernel too, but they are branch- and table-driven rather than floating-point heavy —
 * the FLOP density is in the interpolator corners, and that part is on the device. Moving the aquifer
 * across would mean a per-thread copy of its grid and caches for very little gain, so it stays here
 * until something measures otherwise. The split is exactly the one the plan describes for the CPU
 * reference: corners from the device, arithmetic above them wherever it is cheapest.
 *
 * <p>Everything is per world: the lowered programs and the device buffers are built once and reused for
 * every chunk. Not thread-safe — one instance belongs to one thread, which is why
 * {@link GpuWorldgenService} owns it from a single dedicated thread.
 */
public final class GpuChunkFiller implements AutoCloseable {

    private final NoiseGeneratorSettings settings;
    private final NoiseSettings geometry;
    private final NoiseRouter router;
    private final RandomState randomState;

    private final DensityInterpreter density;
    private final DensityInterpreter veinToggle;
    private final DensityInterpreter veinRidged;
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

    private boolean closed;

    private GpuChunkFiller(CudaContext context, RandomState randomState, NoiseGeneratorSettings settings) {
        this.settings = settings;
        this.geometry = settings.noiseSettings();
        this.router = randomState.router();
        this.randomState = randomState;

        this.density = lower(this.router.finalDensity());
        this.veinToggle = lower(this.router.veinToggle());
        this.veinRidged = lower(this.router.veinRidged());
        this.barrierNoise = lower(this.router.barrierNoise());
        this.floodedness = lower(this.router.fluidLevelFloodednessNoise());
        this.spread = lower(this.router.fluidLevelSpreadNoise());
        this.lava = lower(this.router.lavaNoise());
        this.erosion = lower(this.router.erosion());
        this.depth = lower(this.router.depth());
        this.veinGap = lower(this.router.veinGap());
        this.preliminarySurface = lower(this.router.preliminarySurfaceLevel());

        this.densityEvaluator = DensityEvaluatorGpu.forProgram(context, this.density.program(), this.geometry);
        this.veinToggleEvaluator = DensityEvaluatorGpu.forProgram(context, this.veinToggle.program(), this.geometry);
        this.veinRidgedEvaluator = DensityEvaluatorGpu.forProgram(context, this.veinRidged.program(), this.geometry);
    }

    public static GpuChunkFiller create(CudaContext context, RandomState randomState,
            NoiseGeneratorSettings settings) {
        return new GpuChunkFiller(context, randomState, settings);
    }

    /** The block state ids of one chunk, indexed {@code (x * 16 + z) * height + (y - minY)}. */
    public int[] blockIds(int chunkX, int chunkZ) {
        int minY = this.geometry.minY();
        int height = this.geometry.height();

        double[] densities = InterpolatedDensity.forChunk(this.densityEvaluator, this.density,
                this.geometry, chunkX, chunkZ);
        double[] veinToggleValues = InterpolatedDensity.forChunk(this.veinToggleEvaluator,
                this.veinToggle, this.geometry, chunkX, chunkZ, CornerInterpolation.INCREMENTAL);
        double[] veinRidgedValues = InterpolatedDensity.forChunk(this.veinRidgedEvaluator,
                this.veinRidged, this.geometry, chunkX, chunkZ, CornerInterpolation.INCREMENTAL);

        MaterialRules.RouterNoises noises = new MaterialRules.RouterNoises(
                this.barrierNoise, this.floodedness, this.spread, this.lava, this.erosion,
                this.depth, this.veinGap,
                (x, y, z) -> veinToggleValues[index(x, y, z, chunkX, chunkZ, minY, height)],
                (x, y, z) -> veinRidgedValues[index(x, y, z, chunkX, chunkZ, minY, height)]);

        MaterialRules rules = MaterialRules.forChunk(noises, surfaceLevels(), fluidPicker(),
                VanillaRandomExport.export(this.randomState.aquiferRandom()),
                VanillaRandomExport.export(this.randomState.oreRandom()), chunkX, chunkZ, this.geometry);

        int defaultBlock = Block.getId(this.settings.defaultBlock());
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;
        int[] ids = new int[16 * 16 * height];

        for (int yLocal = 0; yLocal < height; yLocal++) {
            int y = minY + yLocal;
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    double value = densities[(x * 16 + z) * height + yLocal];
                    BlockState state = rules.compute(chunkMinBlockX + x, y, chunkMinBlockZ + z, value);
                    ids[(x * 16 + z) * height + yLocal] = state == null ? defaultBlock : Block.getId(state);
                }
            }
        }
        return ids;
    }

    /** Emits the chunk's blocks and writes them into {@code chunk}. */
    public void fill(ChunkAccess chunk) {
        int[] ids = blockIds(chunk.getPos().x(), chunk.getPos().z());
        ChunkReplay.write(chunk, ids, this.geometry.minY(), this.geometry.height());
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
