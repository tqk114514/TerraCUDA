package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
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
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * The per-block kernel's coordinate handling.
 *
 * <p>It takes a linear block index and has to recover three things from it — the block position for
 * the DAG, the cell, and the position within the cell — and getting any of them wrong produces terrain
 * that is subtly shifted rather than obviously broken. The three programs the chunk pass actually
 * runs cannot catch it: everything above their markers is pure arithmetic, so the block position is
 * never read. This test uses a program that is nothing but a 3D noise, which reads all three
 * coordinates, and compares every block against the CPU interpreter.
 */
class BlockPassCoordinatesTest {

    private static final int CHUNK_X = 3;
    private static final int CHUNK_Z = -7;

    @Test
    void everyBlockSeesItsOwnCoordinates() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseSettings settings = OverworldFixture.noiseSettings();

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        DensityInterpreter interpreter =
                new DensityInterpreter(DensityCompiler.lower(randomState.router().barrierNoise()));

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                try (DensityEvaluatorGpu evaluator = DensityEvaluatorGpu.forProgram(context,
                        interpreter.program(), settings)) {
                    double[] gpu = evaluator.blocksForChunk(settings, CHUNK_X, CHUNK_Z,
                            CornerInterpolation.LERP3);

                    int minY = settings.minY();
                    int height = settings.height();
                    int mismatches = 0;
                    String[] first = new String[1];
                    for (int x = 0; x < 16; x++) {
                        for (int z = 0; z < 16; z++) {
                            for (int yLocal = 0; yLocal < height; yLocal++) {
                                int y = minY + yLocal;
                                double expected = interpreter.evaluate(CHUNK_X * 16 + x, y,
                                        CHUNK_Z * 16 + z);
                                double actual = gpu[(x * 16 + z) * height + yLocal];
                                if (Double.doubleToRawLongBits(expected)
                                        != Double.doubleToRawLongBits(actual)) {
                                    if (mismatches == 0) {
                                        first[0] = "at (" + (CHUNK_X * 16 + x) + ", " + y + ", "
                                                + (CHUNK_Z * 16 + z) + ") gpu=" + actual
                                                + " cpu=" + expected;
                                    }
                                    mismatches++;
                                }
                            }
                        }
                    }
                    int total = mismatches;
                    assertEquals(0, total, () -> total + " blocks differ; first: " + first[0]);
                }
            }
        }
    }
}
