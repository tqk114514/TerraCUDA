package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import java.util.TreeMap;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;
import tqk114514.terracuda.worldgen.GpuChunkFiller;
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * Where a chunk's time goes. Prints; asserts nothing.
 *
 * <p>Kept as a test because it is the only place the numbers are reproducible, and because three of
 * the four "obvious" bottlenecks this project chased turned out not to be the bottleneck at all. It
 * skips cleanly without a CUDA device, so it costs a machine that cannot run it nothing.
 */
class ChunkPassProfileTest {

    private static final int CHUNK_X = 3;
    private static final int CHUNK_Z = -7;
    private static final int REPEATS = 5;

    @Test
    void profile() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseGeneratorSettings settings = OverworldFixture.generatorSettings();
        NoiseSettings geometry = settings.noiseSettings();

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);

                try (GpuChunkFiller filler = GpuChunkFiller.create(context, randomState, settings)) {
                    filler.blockIds(CHUNK_X, CHUNK_Z);
                    filler.blockIds(CHUNK_X + 1, CHUNK_Z);
                    System.out.printf("PROFILE whole chunk        %7.3f ms%n",
                            time(() -> filler.blockIds(CHUNK_X, CHUNK_Z)));
                }

                describeProgram("final_density", randomState.router().finalDensity());
                describeProgram("vein_toggle", randomState.router().veinToggle());
                describeProgram("vein_ridged", randomState.router().veinRidged());

                measurePasses(context, randomState, geometry);
                measureRouterEntries(randomState);

                for (long bytes : new long[] {786432L, 3932160L}) {
                    CudaContext.DeviceBuffer buffer = context.allocate(bytes);
                    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
                        java.lang.foreign.MemorySegment host = arena.allocate(bytes);
                        double ms = time(() -> context.copyFromDevice(buffer, host));
                        System.out.printf("PROFILE D2H %5.1f MB       %7.3f ms%n",
                                bytes / 1048576.0, ms);
                    }
                    context.free(buffer);
                }
            }
        }
    }

    /** The two images a program is lowered into: the full one, and the reduced per-block one. */
    private static void describeProgram(String name,
            net.minecraft.world.level.levelgen.DensityFunction function) {
        DensityProgram program = DensityCompiler.lower(function);
        DensityProgramImage full = DensityProgramImage.of(program);
        DensityProgramImage blocks = DensityProgramImage.ofBlocks(program);
        System.out.printf("PROFILE %-14s %3d instructions -> %2d above the markers (%d markers)%n",
                name, full.instructionCount(), blocks.instructionCount(), blocks.markerCount());
    }

    /** The two device passes, separately, for each of the three chunk-level programs. */
    private static void measurePasses(CudaContext context, RandomState randomState,
            NoiseSettings geometry) {
        measure(context, "final_density", randomState.router().finalDensity(), geometry,
                CornerInterpolation.LERP3);
        measure(context, "vein_toggle", randomState.router().veinToggle(), geometry,
                CornerInterpolation.INCREMENTAL);
        measure(context, "vein_ridged", randomState.router().veinRidged(), geometry,
                CornerInterpolation.INCREMENTAL);
    }

    private static void measure(CudaContext context, String name,
            net.minecraft.world.level.levelgen.DensityFunction function, NoiseSettings geometry,
            int order) {
        DensityProgram program = DensityCompiler.lower(function);
        try (DensityEvaluatorGpu evaluator =
                     DensityEvaluatorGpu.forProgram(context, program, geometry)) {
            evaluator.blocksForChunk(geometry, CHUNK_X, CHUNK_Z, order);
            System.out.printf("PROFILE %-14s markers+blocks %7.3f ms (marker tables alone %7.3f)%n",
                    name,
                    time(() -> evaluator.blocksForChunk(geometry, CHUNK_X, CHUNK_Z, order)),
                    time(() -> ChunkCornerTables.compute(evaluator, geometry, CHUNK_X, CHUNK_Z)));
        }
    }

    private static void measureRouterEntries(RandomState randomState) {
        var router = randomState.router();
        measurePerBlock(router.barrierNoise(), "barrier_noise");
        measurePerBlock(router.fluidLevelFloodednessNoise(), "floodedness");
        measurePerBlock(router.fluidLevelSpreadNoise(), "spread");
        measurePerBlock(router.lavaNoise(), "lava");
        measurePerBlock(router.erosion(), "erosion");
        measurePerBlock(router.depth(), "depth");
        measurePerBlock(router.veinGap(), "vein_gap");
        describeEntry(router.depth(), "depth");
        describeEntry(router.erosion(), "erosion");
        describeEntry(router.preliminarySurfaceLevel(), "prelim_surface");
    }

    private static void describeEntry(net.minecraft.world.level.levelgen.DensityFunction function,
            String name) {
        DensityProgram program = DensityCompiler.lower(function);
        System.out.printf("PROFILE router %-14s %3d instructions, %d markers %s%n",
                name, program.size(), program.markerCount(), markerHistogram(program));
    }

    private static String markerHistogram(DensityProgram program) {
        TreeMap<String, Integer> histogram = new TreeMap<>();
        for (int kind : program.markerKinds()) {
            histogram.merge(kindName(kind), 1, Integer::sum);
        }
        return histogram.toString();
    }

    private static String kindName(int kind) {
        return switch (kind) {
            case DensityProgram.MARKER_INTERPOLATED -> "interpolated";
            case DensityProgram.MARKER_FLAT_CACHE -> "flat_cache";
            case DensityProgram.MARKER_CACHE_2D -> "cache_2d";
            case DensityProgram.MARKER_CACHE_ONCE -> "cache_once";
            case DensityProgram.MARKER_CACHE_ALL_IN_CELL -> "cache_all_in_cell";
            default -> "marker?" + kind;
        };
    }

    /**
     * The cost of evaluating one router entry at every block of a chunk. The material rules ask for
     * these straight at block resolution, so this is what a missing per-column cache would cost.
     */
    private static void measurePerBlock(net.minecraft.world.level.levelgen.DensityFunction function,
            String name) {
        DensityInterpreter interpreter = new DensityInterpreter(DensityCompiler.lower(function));
        int minY = OverworldFixture.noiseSettings().minY();
        int height = OverworldFixture.noiseSettings().height();
        double ms = time(() -> {
            double sink = 0.0;
            for (int y = minY; y < minY + height; y++) {
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        sink += interpreter.evaluate(CHUNK_X * 16 + x, y, CHUNK_Z * 16 + z);
                    }
                }
            }
            if (sink == 12345.6789) {
                System.out.println(sink);
            }
        });
        System.out.printf("PROFILE per-block %-14s %7.3f ms%n", name, ms);
    }

    private static double time(Runnable body) {
        body.run();
        long best = Long.MAX_VALUE;
        for (int i = 0; i < REPEATS; i++) {
            long start = System.nanoTime();
            body.run();
            best = Math.min(best, System.nanoTime() - start);
        }
        return best / 1.0e6;
    }
}
