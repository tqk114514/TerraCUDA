package tqk114514.terracuda.noise;

import static org.junit.jupiter.api.Assertions.assertThrows;

import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Parity between {@link ImprovedNoise} and {@code net.minecraft.world.level.levelgen.synth.ImprovedNoise}.
 *
 * <p>The vanilla type is referenced fully qualified because our class deliberately shares its simple
 * name — the point of this package is a drop-in, bit-identical reimplementation.
 */
class ImprovedNoiseTest {

    private static final long[] SEEDS = {0L, 1L, 42L, -1L, 987654321L, Long.MIN_VALUE};

    @Test
    void noiseMatchesVanilla() {
        for (long seed : SEEDS) {
            net.minecraft.world.level.levelgen.synth.ImprovedNoise vanilla =
                    new net.minecraft.world.level.levelgen.synth.ImprovedNoise(new XoroshiroRandomSource(seed));
            ImprovedNoise mine = new ImprovedNoise(new XoroshiroRandom(seed));

            for (int i = 0; i < SamplePoints.COUNT; i++) {
                double x = SamplePoints.coordinate(i, 0);
                double y = SamplePoints.coordinate(i, 1);
                double z = SamplePoints.coordinate(i, 2);
                TestParity.assertDoubleIdentical(vanilla.noise(x, y, z), mine.noise(x, y, z),
                        "seed=" + seed + " point=" + i);
            }
        }
    }

    @Test
    void yScaledNoiseMatchesVanilla() {
        // BlendedNoise uses the yScale / yFudge variant; exercise both the fudge and the short-circuit.
        double[][] scales = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.25}, {2.0, 1.5}, {8.0, 3.0}};
        for (long seed : SEEDS) {
            net.minecraft.world.level.levelgen.synth.ImprovedNoise vanilla =
                    new net.minecraft.world.level.levelgen.synth.ImprovedNoise(new XoroshiroRandomSource(seed));
            ImprovedNoise mine = new ImprovedNoise(new XoroshiroRandom(seed));

            for (double[] scale : scales) {
                for (int i = 0; i < SamplePoints.COUNT; i++) {
                    double x = SamplePoints.coordinate(i, 3);
                    double y = SamplePoints.coordinate(i, 4);
                    double z = SamplePoints.coordinate(i, 5);
                    TestParity.assertDoubleIdentical(
                            vanilla.noise(x, y, z, scale[0], scale[1]),
                            mine.noise(x, y, z, scale[0], scale[1]),
                            "seed=" + seed + " yScale=" + scale[0] + " yFudge=" + scale[1] + " point=" + i);
                }
            }
        }
    }

    @Test
    void reconstructionFromAPermutationTableReproducesTheSameNoise() {
        ImprovedNoise original = new ImprovedNoise(new XoroshiroRandom(2024L));
        ImprovedNoise rebuilt = new ImprovedNoise(original.permutation(), original.xo, original.yo, original.zo);

        for (int i = 0; i < SamplePoints.COUNT; i++) {
            double x = SamplePoints.coordinate(i, 6);
            double y = SamplePoints.coordinate(i, 7);
            double z = SamplePoints.coordinate(i, 8);
            TestParity.assertDoubleIdentical(original.noise(x, y, z), rebuilt.noise(x, y, z), "point=" + i);
        }
    }

    @Test
    void permutationIsDefensivelyCopied() {
        ImprovedNoise noise = new ImprovedNoise(new XoroshiroRandom(7L));
        byte[] first = noise.permutation();
        byte[] second = noise.permutation();
        first[0] = (byte) ~first[0];
        org.junit.jupiter.api.Assertions.assertArrayEquals(second, noise.permutation());
    }

    @Test
    void aPermutationTableOfTheWrongSizeIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ImprovedNoise(new byte[255], 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new ImprovedNoise(new byte[257], 0, 0, 0));
    }
}
