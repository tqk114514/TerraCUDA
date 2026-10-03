package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;

/**
 * M0: the export path must reproduce every noise in the <em>real</em> overworld density function,
 * bit for bit.
 *
 * <p>This is the test that gives the GPU port its ground truth. It runs against the actual
 * {@code final_density} DAG built by {@code RandomState}, not a hand-made stand-in, so a noise that
 * the export cannot handle shows up here rather than as wrong terrain later.
 */
class NoiseSnapshotTest {

    @Test
    void everyNoiseInTheOverworldDensityFunctionRoundTripsBitExactly() {
        RandomState randomState = OverworldFixture.randomState();

        Map<NormalNoise, Boolean> distinct = new IdentityHashMap<>();
        randomState.router().finalDensity().mapAll(new DensityFunction.Visitor() {
            @Override
            public DensityFunction apply(DensityFunction input) {
                return input;
            }

            @Override
            public DensityFunction.NoiseHolder visitNoise(DensityFunction.NoiseHolder noise) {
                if (noise.noise() != null) {
                    distinct.put(noise.noise(), Boolean.TRUE);
                }
                return noise;
            }
        });

        assertFalse(distinct.isEmpty(), "the overworld final_density should reference noise");

        int checked = 0;
        for (NormalNoise vanilla : distinct.keySet()) {
            NoiseSnapshot snapshot = NoiseSnapshot.of(vanilla);
            tqk114514.terracuda.noise.NormalNoise mine = snapshot.toNormalNoise();

            assertEquals(vanilla.maxValue(), mine.maxValue(), 0.0,
                    "maxValue differs for " + vanilla.parameters());
            assertEquals(vanilla.parameters().firstOctave(), snapshot.parameters().firstOctave());

            for (int i = 0; i < OverworldFixture.SAMPLE_COUNT; i++) {
                double x = OverworldFixture.coordinate(i, 0);
                double y = OverworldFixture.coordinate(i, 1);
                double z = OverworldFixture.coordinate(i, 2);
                TestParity.assertDoubleIdentical(vanilla.getValue(x, y, z), mine.getValue(x, y, z),
                        "firstOctave=" + vanilla.parameters().firstOctave() + " point=" + i);
            }
            checked++;
        }

        assertTrue(checked > 10, "expected the overworld DAG to use many distinct noises, saw " + checked);
    }

    @Test
    void aSnapshotCarriesThePermutationTablesAndOriginsOfEveryNonZeroOctave() {
        RandomState randomState = OverworldFixture.randomState();

        List<NormalNoise> noises = new ArrayList<>();
        randomState.router().finalDensity().mapAll(new DensityFunction.Visitor() {
            @Override
            public DensityFunction apply(DensityFunction input) {
                return input;
            }

            @Override
            public DensityFunction.NoiseHolder visitNoise(DensityFunction.NoiseHolder noise) {
                if (noise.noise() != null && !noises.contains(noise.noise())) {
                    noises.add(noise.noise());
                }
                return noise;
            }
        });

        NoiseSnapshot snapshot = NoiseSnapshot.of(noises.get(0));
        double[] amplitudes = snapshot.parameters().amplitudes();
        assertEquals(amplitudes.length, snapshot.octaveCount());

        for (int i = 0; i < amplitudes.length; i++) {
            NoiseSnapshot.Octave first = snapshot.firstOctaves()[i];
            NoiseSnapshot.Octave second = snapshot.secondOctaves()[i];
            if (amplitudes[i] == 0.0) {
                assertEquals(null, first, "zero-amplitude octave " + i + " should have no noise level");
                assertEquals(null, second);
            } else {
                assertNotNull(first, "octave " + i + " is missing its first Perlin level");
                assertNotNull(second, "octave " + i + " is missing its second Perlin level");
                assertEquals(256, first.permutation().length);
                assertEquals(256, second.permutation().length);
                assertTrue(first.xo() >= 0.0 && first.xo() < 256.0);
            }
        }
    }

    @Test
    void snapshotsAreImmutable() {
        RandomState randomState = OverworldFixture.randomState();
        NormalNoise vanilla = randomState.getOrCreateNoise(net.minecraft.world.level.levelgen.Noises.TEMPERATURE);

        NoiseSnapshot snapshot = NoiseSnapshot.of(vanilla);
        NoiseSnapshot.Octave octave = snapshot.firstOctaves()[0];
        byte original = octave.permutation()[0];

        octave.permutation()[0] = (byte) ~original;
        assertEquals(original, octave.permutation()[0], "permutation() must hand out a copy");

        double[] amplitudes = snapshot.parameters().amplitudes();
        double originalAmplitude = amplitudes[0];
        amplitudes[0] = 12345.0;
        assertEquals(originalAmplitude, snapshot.parameters().amplitudes()[0], "amplitudes() must hand out a copy");
    }
}
