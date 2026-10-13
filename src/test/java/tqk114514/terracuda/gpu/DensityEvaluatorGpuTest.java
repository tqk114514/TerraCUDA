package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.NoiseSettings;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * M2, K2: the CUDA interpreter must agree with the CPU reference on the <em>real</em> overworld
 * density function, bit for bit.
 *
 * <p>The CPU reference is already known to match vanilla (see {@code DensityInterpreterTest}), so this
 * closes the chain: vanilla == CPU == GPU. Splines, blended noise, shifted noise and the whole
 * arithmetic DAG all have to be right for it to pass.
 */
class DensityEvaluatorGpuTest {

    private static final int POINT_COUNT = 4096;

    @Test
    void theGpuInterpreterMatchesTheCpuReferenceBitForBit() {
        RandomState randomState = OverworldFixture.randomState();
        DensityProgram program = DensityCompiler.lower(randomState.router().finalDensity());
        DensityInterpreter reference = new DensityInterpreter(program);

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);

                int[] coordinates = blockCoordinates(POINT_COUNT);
                try (DensityEvaluatorGpu gpu = new DensityEvaluatorGpu(context, program, POINT_COUNT)) {
                    assertTrue(gpu.instructionCount() > 0);

                    double[] actual = gpu.evaluate(coordinates);
                    assertEquals(POINT_COUNT, actual.length);

                    int mismatches = 0;
                    String[] firstMismatch = new String[1];
                    for (int i = 0; i < POINT_COUNT; i++) {
                        int x = coordinates[3 * i];
                        int y = coordinates[3 * i + 1];
                        int z = coordinates[3 * i + 2];
                        double expected = reference.evaluate(x, y, z);
                        if (Double.doubleToRawLongBits(expected)
                                != Double.doubleToRawLongBits(actual[i])) {
                            if (mismatches == 0) {
                                firstMismatch[0] = "(" + x + ", " + y + ", " + z + "): cpu " + expected
                                        + " (" + Double.doubleToRawLongBits(expected) + ") vs gpu "
                                        + actual[i] + " (" + Double.doubleToRawLongBits(actual[i]) + ")";
                            }
                            mismatches++;
                        }
                    }
                    int total = mismatches;
                    assertEquals(0, total, () -> total + " of " + POINT_COUNT
                            + " positions differ; first mismatch: " + firstMismatch[0]);
                }
            }
        }
    }

    @Test
    void theImageReportsItsShape() {
        DensityProgram program =
                DensityCompiler.lower(OverworldFixture.randomState().router().finalDensity());
        DensityProgramImage image = DensityProgramImage.of(program);

        assertTrue(image.blob().length > 0);
        assertEquals(DensityProgramImage.OFFSET_COUNT, image.offsets().length);
        assertTrue(image.instructionCount() > 0);
        // The image drops instructions the root cannot reach and re-emits shared subgraphs once.
        assertTrue(image.instructionCount() <= program.size(),
                "image has " + image.instructionCount() + " instructions but the program has "
                        + program.size());
        // Post-order puts the root last.
        assertEquals(image.instructionCount() - 1, image.root());
        assertTrue(image.octaveCount() > 0);
        assertTrue(image.summary().contains("bytes"));

        // Every table must be 8-byte aligned: the kernel reinterprets the blob as 64-bit types.
        for (long offset : image.offsets()) {
            assertEquals(0, offset % 8, "offset " + offset + " is not 8-byte aligned");
        }
    }

    /**
     * K0: {@code preliminary_surface_level} is a {@code find_top_surface}, which re-enters the density
     * subtree at a different y for every probe step. It is the root of that program, which is the one
     * shape the kernel can evaluate soundly.
     */
    @Test
    void thePreliminarySurfaceLevelMatchesTheCpuReferenceBitForBit() {
        DensityProgram program =
                DensityCompiler.lower(OverworldFixture.randomState().router().preliminarySurfaceLevel());
        DensityInterpreter reference = new DensityInterpreter(program);

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);

                int count = 1024;
                int[] coordinates = blockCoordinates(count);
                try (DensityEvaluatorGpu gpu = new DensityEvaluatorGpu(context, program, count)) {
                    double[] actual = gpu.evaluate(coordinates);
                    for (int i = 0; i < count; i++) {
                        double expected = reference.evaluate(coordinates[3 * i],
                                coordinates[3 * i + 1], coordinates[3 * i + 2]);
                        TestParity.assertDoubleIdentical(expected, actual[i],
                                "(" + coordinates[3 * i] + ", " + coordinates[3 * i + 1] + ", "
                                        + coordinates[3 * i + 2] + ")");
                    }
                }
            }
        }
    }

    /**
     * Waves grow the point buffers rather than fail: the one-chunk budget the evaluator was built
     * with is a floor, because a wave's corner pass carries several chunks in one launch.
     */
    @Test
    void anOversizedBatchGrowsToFit() {
        DensityProgram program =
                DensityCompiler.lower(OverworldFixture.randomState().router().finalDensity());
        DensityInterpreter reference = new DensityInterpreter(program);

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                try (DensityEvaluatorGpu gpu = new DensityEvaluatorGpu(context, program, 4)) {
                    // Not a whole number of xyz triples stays a hard error.
                    assertThrows(IllegalArgumentException.class, () -> gpu.evaluate(new int[4]));

                    int count = 5;
                    int[] coordinates = blockCoordinates(count);
                    double[] actual = gpu.evaluate(coordinates);
                    assertTrue(gpu.capacity() >= count, "the buffers should have grown past 4, capacity is " + gpu.capacity());
                    for (int i = 0; i < count; i++) {
                        TestParity.assertDoubleIdentical(
                                reference.evaluate(coordinates[3 * i], coordinates[3 * i + 1],
                                        coordinates[3 * i + 2]), actual[i], "grown batch point " + i);
                    }
                }
            }
        }
    }

    /**
     * G1's contract: the wave form is bit-identical to the per-chunk form. Eight chunks — enough to
     * force the corner launch past its one-chunk budget and exercise the growth — go out as one
     * submission set, and every value must come home exactly as the solo pass produced it.
     */
    @Test
    void theWaveFormMatchesThePerChunkFormBitForBit() {
        RandomState randomState = OverworldFixture.randomState();
        DensityProgram program = DensityCompiler.lower(randomState.router().finalDensity());
        NoiseSettings settings = OverworldFixture.noiseSettings();

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                try (DensityEvaluatorGpu gpu = DensityEvaluatorGpu.forProgram(context, program,
                        settings)) {
                    int[] xs = {0, 1, 2, 3, -1, -2, 5, 7};
                    int[] zs = {0, -1, 2, -3, 4, -5, 6, -7};
                    int waveSize = xs.length;
                    double[][] solo = new double[waveSize][];
                    for (int i = 0; i < waveSize; i++) {
                        solo[i] = gpu.blocksForChunk(settings, xs[i], zs[i],
                                tqk114514.terracuda.density.CornerInterpolation.LERP3);
                    }

                    int count = 16 * 16 * settings.height();
                    double[][] wave = new double[waveSize][];
                    for (int i = 0; i < waveSize; i++) {
                        wave[i] = new double[count];
                    }
                    gpu.submitWave(settings, xs, zs,
                            tqk114514.terracuda.density.CornerInterpolation.LERP3, wave);
                    gpu.awaitWave();

                    int mismatches = 0;
                    String[] first = new String[1];
                    for (int i = 0; i < waveSize; i++) {
                        for (int j = 0; j < count; j++) {
                            if (Double.doubleToRawLongBits(solo[i][j])
                                    != Double.doubleToRawLongBits(wave[i][j])) {
                                if (first[0] == null) {
                                    first[0] = "chunk (" + xs[i] + ", " + zs[i] + ") value " + j
                                            + ": solo " + solo[i][j] + " ("
                                            + Double.doubleToRawLongBits(solo[i][j]) + ") vs wave "
                                            + wave[i][j] + " (" + Double.doubleToRawLongBits(wave[i][j])
                                            + ")";
                                }
                                mismatches++;
                            }
                        }
                    }
                    int total = mismatches;
                    assertEquals(0, total, () -> total + " of " + waveSize * count
                            + " values differ; first mismatch: " + first[0]);
                }
            }
        }
    }

    /** Block positions spread over a chunk-sized neighbourhood, including both signs and deep y. */
    private static int[] blockCoordinates(int count) {
        int[] coordinates = new int[3 * count];
        long state = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < coordinates.length; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            long bits = state >>> 11;
            if (i % 3 == 1) {
                coordinates[i] = (int) (bits % 384) - 64;      // y in [-64, 320)
            } else {
                coordinates[i] = (int) (bits % 1024) - 512;    // x, z in [-512, 512)
            }
        }
        return coordinates;
    }
}
