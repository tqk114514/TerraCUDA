package tqk114514.terracuda.noise;

import static org.junit.jupiter.api.Assertions.assertEquals;

import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Parity between {@link NormalNoise} and {@code net.minecraft.world.level.levelgen.synth.NormalNoise},
 * including {@code maxValue()} — the value the aquifer and veinifier thresholds are calibrated against.
 */
class NormalNoiseTest {

    private static final long[] SEEDS = {0L, 1L, 42L, -1L, 987654321L};

    @Test
    void getValueAndMaxValueMatchVanilla() {
        for (long seed : SEEDS) {
            for (PerlinNoiseTest.Layout layout : PerlinNoiseTest.LAYOUTS) {
                double first = layout.amplitudes()[0];
                double[] rest = java.util.Arrays.copyOfRange(layout.amplitudes(), 1, layout.amplitudes().length);

                net.minecraft.world.level.levelgen.synth.NormalNoise vanilla =
                        net.minecraft.world.level.levelgen.synth.NormalNoise.create(
                                new XoroshiroRandomSource(seed),
                                new net.minecraft.world.level.levelgen.synth.NormalNoise.NoiseParameters(
                                        layout.firstOctave(), first, rest));
                NormalNoise mine = NormalNoise.create(new XoroshiroRandom(seed),
                        NoiseParameters.of(layout.firstOctave(), first, rest));

                TestParity.assertDoubleIdentical(vanilla.maxValue(), mine.maxValue(),
                        "maxValue seed=" + seed + " firstOctave=" + layout.firstOctave());

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
    void parametersRoundTrip() {
        NoiseParameters parameters = NoiseParameters.of(-8, 1.0, 1.0, 1.0);
        NormalNoise noise = NormalNoise.create(new XoroshiroRandom(1L), parameters);

        assertEquals(parameters, noise.parameters());
        assertEquals(-8, noise.parameters().firstOctave());
        assertEquals(3, noise.parameters().amplitudes().length);
    }

    @Test
    void noiseParametersUseValueEquality() {
        assertEquals(NoiseParameters.of(-3, 1.0, 0.5), NoiseParameters.of(-3, 1.0, 0.5));
        assertEquals(NoiseParameters.of(-3, 1.0, 0.5).hashCode(), NoiseParameters.of(-3, 1.0, 0.5).hashCode());
        org.junit.jupiter.api.Assertions.assertNotEquals(NoiseParameters.of(-3, 1.0, 0.5),
                NoiseParameters.of(-4, 1.0, 0.5));
    }
}
