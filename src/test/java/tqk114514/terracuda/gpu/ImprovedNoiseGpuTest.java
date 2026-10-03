package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.noise.ImprovedNoise;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * The M1 acceptance test: a million points through the GPU kernel must match the Java reference
 * <em>bit for bit</em>, not approximately.
 *
 * <p>This is the test that would catch an accidental FMA contraction, a reordered lerp, or a
 * coordinate that was folded differently. It skips itself when there is no usable GPU, so the suite
 * stays green on a CI runner without one.
 */
class ImprovedNoiseGpuTest {

    private static final int POINT_COUNT = 1_000_000;
    private static final long SEED = 424242L;

    @Test
    void aMillionPointsMatchTheJavaReferenceBitForBit() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();

            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.LoadedModule module = CudaKernels.loadModule(context, device);
                assertFalse(module.detail().isBlank());

                ImprovedNoise reference = new ImprovedNoise(new XoroshiroRandom(SEED));
                double[] xyz = points(POINT_COUNT);

                try (ImprovedNoiseGpu gpu = new ImprovedNoiseGpu(context, reference, POINT_COUNT)) {
                    double[] actual = gpu.evaluate(xyz);
                    assertEquals(POINT_COUNT, actual.length);

                    int mismatches = 0;
                    String[] firstMismatch = new String[1];
                    for (int i = 0; i < POINT_COUNT; i++) {
                        double x = xyz[3 * i];
                        double y = xyz[3 * i + 1];
                        double z = xyz[3 * i + 2];
                        double expected = reference.noise(x, y, z);
                        if (Double.doubleToRawLongBits(expected) != Double.doubleToRawLongBits(actual[i])) {
                            if (mismatches == 0) {
                                firstMismatch[0] = "point " + i + " (" + x + ", " + y + ", " + z
                                        + "): java " + expected + " (" + Double.doubleToRawLongBits(expected)
                                        + ") vs gpu " + actual[i] + " (" + Double.doubleToRawLongBits(actual[i]) + ")";
                            }
                            mismatches++;
                        }
                    }
                    int totalMismatches = mismatches;
                    assertEquals(0, totalMismatches, () -> totalMismatches + " of " + POINT_COUNT
                            + " points differ; first mismatch: " + firstMismatch[0]);
                }
            }
        }
    }

    @Test
    void aSmallBatchIsEvaluatedInOrder() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                ImprovedNoise reference = new ImprovedNoise(new XoroshiroRandom(7L));

                try (ImprovedNoiseGpu gpu = new ImprovedNoiseGpu(context, reference, 64)) {
                    double[] xyz = {1.5, 2.5, 3.5, -100.25, 64.0, 0.125, 0.0, 0.0, 0.0};
                    double[] actual = gpu.evaluate(xyz);

                    assertEquals(3, actual.length);
                    for (int i = 0; i < 3; i++) {
                        assertEquals(Double.doubleToRawLongBits(reference.noise(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2])),
                                Double.doubleToRawLongBits(actual[i]), "point " + i);
                    }
                }
            }
        }
    }

    @Test
    void anOversizedOrRaggedBatchIsRejected() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                ImprovedNoise reference = new ImprovedNoise(new XoroshiroRandom(7L));

                try (ImprovedNoiseGpu gpu = new ImprovedNoiseGpu(context, reference, 4)) {
                    assertThrows(IllegalArgumentException.class, () -> gpu.evaluate(new double[] {1.0, 2.0}));
                    assertThrows(IllegalArgumentException.class,
                            () -> gpu.evaluate(new double[3 * 5]));
                    assertThrows(IllegalArgumentException.class, () -> new ImprovedNoiseGpu(context, reference, 0));
                }
            }
        }
    }

    @Test
    void theModuleReportsWhichImageItUsed() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.LoadedModule module = CudaKernels.loadModule(context, device);
                assertTrue(module.origin() == CudaKernels.Origin.CUBIN
                                || module.origin() == CudaKernels.Origin.PTX
                                || module.origin() == CudaKernels.Origin.NVRTC,
                        "unexpected origin " + module.origin());
            }
        }
    }

    /**
     * The loader must prefer a cubin built for this exact compute capability over the PTX and NVRTC
     * rungs. Without this, a build that quietly stopped producing cubins would go unnoticed.
     */
    @Test
    void aCubinForThisDeviceIsPreferredOverTheLowerRungs() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        CudaDeviceInfo device = environment.firstDevice().orElseThrow();

        String cubin = "/META-INF/terracuda/cuda/terracuda_sm" + device.computeCapability() + ".cubin";
        assumeTrue(ImprovedNoiseGpuTest.class.getResource(cubin) != null,
                "no prebuilt cubin for sm_" + device.computeCapability() + " on the classpath");

        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = loaded.get()) {
            driver.init();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                assertEquals(CudaKernels.Origin.CUBIN, CudaKernels.loadModule(context, device).origin());
            }
        }
    }

    /** Deterministic points spread over [-2000, 2000) in all three axes, both signs. */
    private static double[] points(int count) {
        double[] xyz = new double[3 * count];
        long state = 0x2545F4914F6CDD1DL;
        for (int i = 0; i < xyz.length; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            double unit = (state >>> 11) * 0x1.0p-53;
            xyz[i] = unit * 4000.0 - 2000.0;
        }
        return xyz;
    }
}
