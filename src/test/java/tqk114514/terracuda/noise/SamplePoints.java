package tqk114514.terracuda.noise;

/**
 * Deterministic, well-spread sample coordinates for the noise parity tests.
 *
 * <p>A plain LCG keeps the points reproducible across machines and avoids depending on any random
 * source under test. The range {@code [-1000, 1000)} covers several {@code 2^25} wrap boundaries only
 * for the noise stack, and both signs of every coordinate.
 */
final class SamplePoints {

    static final int COUNT = 1000;

    private SamplePoints() {
    }

    static double coordinate(int index, int salt) {
        long h = (index + 1L) * 6364136223846793005L + (salt + 1L) * 1442695040888963407L;
        return ((h >>> 11) * 0x1.0p-53) * 2000.0 - 1000.0;
    }
}
