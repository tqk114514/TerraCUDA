package tqk114514.terracuda.math;

import static org.junit.jupiter.api.Assertions.assertEquals;

import net.minecraft.util.Mth;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;

/**
 * Parity between {@link VanillaMath} and {@code net.minecraft.util.Mth}.
 *
 * <p>These are the leaf primitives the CUDA kernels inline, so a mismatch here would surface as
 * subtly wrong terrain rather than as an obvious error.
 */
class VanillaMathTest {

    private static final int[] EXTREME_INTS = {
            0, 1, -1, 2, -2, 7, -7, 16, -16, 1000, -1000,
            Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MAX_VALUE - 1, Integer.MIN_VALUE + 1
    };

    @Test
    void floorMatchesVanilla() {
        double[] values = {0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.9999, -1.0001, 1.0E7, -1.0E7, 3.3554432E7};
        for (double value : values) {
            assertEquals(Mth.floor(value), VanillaMath.floor(value), "floor(" + value + ")");
        }
    }

    @Test
    void lfloorMatchesVanilla() {
        double[] values = {0.0, -0.0, 0.5, -0.5, 1.0E15, -1.0E15, 3.3554432E7, -3.3554432E7};
        for (double value : values) {
            assertEquals(Mth.lfloor(value), VanillaMath.lfloor(value), "lfloor(" + value + ")");
        }
    }

    @Test
    void lerpFamilyMatchesVanilla() {
        double[] alphas = {0.0, 0.25, 0.5, 0.75, 1.0, -0.3, 1.7, 0.123456789};
        double[] points = {-1.5, -0.0, 0.0, 0.25, 1.0, 3.75, 1.0E9};

        for (double alpha : alphas) {
            for (double start : points) {
                for (double end : points) {
                    TestParity.assertDoubleIdentical(Mth.lerp(alpha, start, end),
                            VanillaMath.lerp(alpha, start, end), "lerp");
                }
            }
        }

        for (double a1 : alphas) {
            for (double a2 : alphas) {
                double x00 = points[0];
                double x10 = points[3];
                double x01 = points[5];
                double x11 = points[2];
                TestParity.assertDoubleIdentical(Mth.lerp2(a1, a2, x00, x10, x01, x11),
                        VanillaMath.lerp2(a1, a2, x00, x10, x01, x11), "lerp2");
                TestParity.assertDoubleIdentical(
                        Mth.lerp3(a1, a2, 0.375, x00, x10, x01, x11, -x00, -x10, -x01, -x11),
                        VanillaMath.lerp3(a1, a2, 0.375, x00, x10, x01, x11, -x00, -x10, -x01, -x11),
                        "lerp3");
            }
        }
    }

    @Test
    void smoothstepMatchesVanilla() {
        for (int i = -100; i <= 200; i++) {
            double x = i / 100.0;
            TestParity.assertDoubleIdentical(Mth.smoothstep(x), VanillaMath.smoothstep(x), "smoothstep(" + x + ")");
            TestParity.assertDoubleIdentical(Mth.smoothstepDerivative(x), VanillaMath.smoothstepDerivative(x),
                    "smoothstepDerivative(" + x + ")");
        }
    }

    @Test
    void getSeedMatchesVanillaOverAGrid() {
        for (int x = -80; x <= 80; x += 7) {
            for (int y = -80; y <= 80; y += 13) {
                for (int z = -80; z <= 80; z += 11) {
                    assertEquals(Mth.getSeed(x, y, z), VanillaMath.getSeed(x, y, z),
                            "getSeed(" + x + ", " + y + ", " + z + ")");
                }
            }
        }
    }

    @Test
    void getSeedMatchesVanillaAtIntegerOverflowBoundaries() {
        for (int x : EXTREME_INTS) {
            for (int y : EXTREME_INTS) {
                for (int z : EXTREME_INTS) {
                    assertEquals(Mth.getSeed(x, y, z), VanillaMath.getSeed(x, y, z),
                            "getSeed(" + x + ", " + y + ", " + z + ")");
                }
            }
        }
    }
}
