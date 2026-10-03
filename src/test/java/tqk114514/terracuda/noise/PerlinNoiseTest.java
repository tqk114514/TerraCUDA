package tqk114514.terracuda.noise;

import java.util.Arrays;
import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Parity between {@link PerlinNoise} and {@code net.minecraft.world.level.levelgen.synth.PerlinNoise}.
 *
 * <p>The octave layouts below are shaped after the real overworld noises: {@code jagged} (16 octaves,
 * the most expensive leaf), the continentalness / cave_cheese families, and a couple of layouts with
 * zero amplitudes, which exercise the "no ImprovedNoise for this octave" branch.
 */
class PerlinNoiseTest {

    private static final long[] SEEDS = {0L, 1L, 42L, -1L, 987654321L};

    /** A noise layout: the first octave index and the per-octave amplitudes. */
    record Layout(int firstOctave, double[] amplitudes) {
    }

    static final Layout[] LAYOUTS = {
            new Layout(-16, ones(16)),
            new Layout(-9, ones(10)),
            new Layout(-8, ones(9)),
            new Layout(-3, new double[] {1.0, 1.0, 1.0}),
            new Layout(0, new double[] {1.0}),
            new Layout(-5, new double[] {1.0, 0.0, 1.0, 0.0, 1.0}),
            new Layout(-2, new double[] {1.0, 0.5, 0.25, 0.125})
    };

    static double[] ones(int count) {
        double[] amplitudes = new double[count];
        Arrays.fill(amplitudes, 1.0);
        return amplitudes;
    }

    @Test
    void getValueMatchesVanilla() {
        for (long seed : SEEDS) {
            for (Layout layout : LAYOUTS) {
                double first = layout.amplitudes()[0];
                double[] rest = Arrays.copyOfRange(layout.amplitudes(), 1, layout.amplitudes().length);
                net.minecraft.world.level.levelgen.synth.PerlinNoise vanilla =
                        net.minecraft.world.level.levelgen.synth.PerlinNoise.create(
                                new XoroshiroRandomSource(seed), layout.firstOctave(), first, rest);
                PerlinNoise mine = new PerlinNoise(new XoroshiroRandom(seed), layout.firstOctave(),
                        layout.amplitudes());

                for (int i = 0; i < SamplePoints.COUNT; i++) {
                    double x = SamplePoints.coordinate(i, 0);
                    double y = SamplePoints.coordinate(i, 1);
                    double z = SamplePoints.coordinate(i, 2);
                    TestParity.assertDoubleIdentical(vanilla.getValue(x, y, z), mine.getValue(x, y, z),
                            "seed=" + seed + " firstOctave=" + layout.firstOctave() + " point=" + i);
                }
            }
        }
    }

    @Test
    void wrapMatchesVanillaAcrossTheWrapBoundary() {
        double period = 3.3554432E7;
        double[] values = {
                0.0, -0.0, 1.0, -1.0, period, -period, period * 0.5, -period * 0.5,
                period * 1.5, -period * 1.5, period * 100.0 + 0.25, -period * 100.0 - 0.25
        };
        for (double value : values) {
            TestParity.assertDoubleIdentical(net.minecraft.world.level.levelgen.synth.PerlinNoise.wrap(value),
                    PerlinNoise.wrap(value), "wrap(" + value + ")");
        }
    }

    @Test
    void amplitudesAreDefensivelyCopied() {
        double[] amplitudes = {1.0, 1.0};
        PerlinNoise noise = new PerlinNoise(new XoroshiroRandom(5L), -1, amplitudes);
        amplitudes[0] = 999.0;

        double[] reported = noise.amplitudes();
        reported[1] = 999.0;

        double[] expected = {1.0, 1.0};
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, noise.amplitudes());
    }
}
