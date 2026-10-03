package tqk114514.terracuda.math;

/**
 * The subset of {@code net.minecraft.util.Mth} that the terrain noise pipeline depends on,
 * reproduced verbatim.
 *
 * <p>Every expression here is transcribed character-for-character from the 26.1.2 sources. That is
 * deliberate: the CUDA kernels must produce bit-identical results, and any "cleanup" of these
 * formulas (reordering a multiply, folding constants, switching to {@code Math.fma}) changes the
 * last bits of the density field and therefore the terrain. In particular
 * {@link #getSeed(int, int, int)} relies on {@code x * 3129871} being evaluated in {@code int}
 * arithmetic before being widened.
 */
public final class VanillaMath {

    private VanillaMath() {
    }

    public static int floor(double value) {
        return (int) Math.floor(value);
    }

    public static long lfloor(double value) {
        return (long) Math.floor(value);
    }

    /** {@code start + alpha * (end - start)} — the vanilla lerp shape, not a fused variant. */
    public static double lerp(double alpha, double start, double end) {
        return start + alpha * (end - start);
    }

    public static double lerp2(double alpha1, double alpha2,
            double x00, double x10, double x01, double x11) {
        return lerp(alpha2, lerp(alpha1, x00, x10), lerp(alpha1, x01, x11));
    }

    public static double lerp3(double alpha1, double alpha2, double alpha3,
            double x000, double x100, double x010, double x110,
            double x001, double x101, double x011, double x111) {
        return lerp(alpha3, lerp2(alpha1, alpha2, x000, x100, x010, x110),
                lerp2(alpha1, alpha2, x001, x101, x011, x111));
    }

    public static double smoothstep(double x) {
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
    }

    public static double smoothstepDerivative(double x) {
        return 30.0 * x * x * (x - 1.0) * (x - 1.0);
    }

    public static double clamp(double value, double min, double max) {
        return value < min ? min : Math.min(value, max);
    }

    public static double inverseLerp(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    public static double clampedLerp(double factor, double min, double max) {
        if (factor < 0.0) {
            return min;
        }
        return factor > 1.0 ? max : lerp(factor, min, max);
    }

    public static double clampedMap(double value, double fromMin, double fromMax, double toMin, double toMax) {
        return clampedLerp(inverseLerp(value, fromMin, fromMax), toMin, toMax);
    }

    /** The {@code float} overload; the arithmetic stays in {@code float}, as vanilla's does. */
    public static float lerp(float alpha, float start, float end) {
        return start + alpha * (end - start);
    }

    /**
     * {@code Mth.binarySearch}: the first index whose predicate holds, or {@code to} when none does.
     * The cubic splines subtract one from the result to find the interval start.
     */
    public static int binarySearch(int from, int to, java.util.function.IntPredicate condition) {
        int length = to - from;
        while (length > 0) {
            int half = length / 2;
            int middle = from + half;
            if (condition.test(middle)) {
                length = half;
            } else {
                from = middle + 1;
                length -= half + 1;
            }
        }
        return from;
    }

    /**
     * The positional seed hash used by {@code PositionalRandomFactory.at}.
     *
     * <p>{@code x * 3129871} is intentionally {@code int} multiplication that may overflow before the
     * XOR widens it to {@code long}; this mirrors the vanilla source exactly.
     */
    public static long getSeed(int x, int y, int z) {
        long seed = x * 3129871 ^ z * 116129781L ^ y;
        seed = seed * seed * 42317861L + seed * 11L;
        return seed >> 16;
    }
}
