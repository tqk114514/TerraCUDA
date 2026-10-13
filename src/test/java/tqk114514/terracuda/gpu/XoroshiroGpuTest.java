package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.cuda.KernelArguments;
import tqk114514.terracuda.math.VanillaMath;
import tqk114514.terracuda.random.PositionalRandomFactory;

/**
 * G2a's first primitive: the XOROSHIRO128++ positional source on the device must draw the same
 * first float as the CPU reference, bit for bit, at every position and for every factory seed
 * pair the veinifier can meet.
 *
 * <p>The CPU reference is itself tested against vanilla's own generator (see
 * {@code XoroshiroRandomTest}), so this closes the chain: vanilla == CPU == GPU. The positions
 * include the origin and spread coordinates of both signs, and the seed pairs include one that
 * lands the derived state on all-zero -- the golden/silver substitution vanilla performs on the
 * zero state is observable output, and the port must reproduce it.
 */
class XoroshiroGpuTest {

    private static final int POINT_COUNT = 1024;

    @Test
    void theFirstPositionalDrawMatchesTheCpuReferenceBitForBit() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);

                int[] points = points();
                // Seed pairs: an arbitrary pair, their bitwise complements, and one engineered
                // so the state at the origin is all zero (getSeed(0,0,0) == 0).
                long[][] pairs = {
                    {0x9E3779B97F4A7C15L, 0x6A09E667F3BCC909L},
                    {-4523168431819676477L, 123456789L},
                    {~0x9E3779B97F4A7C15L, ~0x6A09E667F3BCC909L},
                    {0L, 0L},
                    {VanillaMath.getSeed(0, 0, 0), 0L},
                };

                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment pointsSegment = arena.allocateFrom(ValueLayout.JAVA_INT, points);
                    MemorySegment resultsSegment = arena.allocate((long) POINT_COUNT * Float.BYTES);
                    CudaContext.DeviceBuffer devicePoints =
                            context.allocate((long) points.length * Integer.BYTES);
                    CudaContext.DeviceBuffer deviceResults =
                            context.allocate((long) POINT_COUNT * Float.BYTES);
                    context.copyToDevice(pointsSegment, devicePoints);

                    for (long[] pair : pairs) {
                        try (KernelArguments arguments = new KernelArguments(5)) {
                            arguments.addLong(pair[0]).addLong(pair[1])
                                    .addDevicePointer(devicePoints.address())
                                    .addDevicePointer(deviceResults.address())
                                    .addInt(POINT_COUNT);
                            int grid = (POINT_COUNT + 127) / 128;
                            context.launch("terracuda_xoroshiro_float_points", grid, 1, 1,
                                    128, 1, 1, arguments);
                        }
                        context.synchronize();
                        context.copyFromDevice(deviceResults, resultsSegment);

                        int mismatches = 0;
                        String[] first = new String[1];
                        PositionalRandomFactory factory = new PositionalRandomFactory(pair[0], pair[1]);
                        for (int i = 0; i < POINT_COUNT; i++) {
                            float expected = factory.at(points[3 * i], points[3 * i + 1], points[3 * i + 2])
                                    .nextFloat();
                            float actual = resultsSegment.get(ValueLayout.JAVA_FLOAT,
                                    (long) i * Float.BYTES);
                            if (Float.floatToRawIntBits(expected) != Float.floatToRawIntBits(actual)) {
                                if (first[0] == null) {
                                    first[0] = "(" + points[3 * i] + ", " + points[3 * i + 1] + ", "
                                            + points[3 * i + 2] + ") seeds (" + pair[0] + ", "
                                            + pair[1] + "): cpu " + expected
                                            + " (" + Float.floatToRawIntBits(expected) + ") vs gpu "
                                            + actual + " (" + Float.floatToRawIntBits(actual) + ")";
                                }
                                mismatches++;
                            }
                        }
                        int total = mismatches;
                        long lo = pair[0];
                        long hi = pair[1];
                        assertEquals(0, total, () -> total + " of " + POINT_COUNT
                                + " draws differ for seeds (" + lo + ", " + hi
                                + "); first mismatch: " + first[0]);
                    }
                    context.free(devicePoints);
                    context.free(deviceResults);
                }
            }
        }
    }

    /** Positions with both signs, extremes, the origin, and vein-window y values. */
    private static int[] points() {
        int[] coordinates = new int[3 * POINT_COUNT];
        long state = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < POINT_COUNT; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            long bits = state >>> 11;
            int x = (int) (bits % 65536) - 32768;
            long bits2 = Long.rotateLeft(state, 17) >>> 11;
            int z = (int) (bits2 % 65536) - 32768;
            int y = (int) ((bits >> 24) % 100) - 60;
            if (i == 0) {
                x = 0;
                y = 0;
                z = 0;
            }
            coordinates[3 * i] = x;
            coordinates[3 * i + 1] = y;
            coordinates[3 * i + 2] = z;
        }
        return coordinates;
    }
}
