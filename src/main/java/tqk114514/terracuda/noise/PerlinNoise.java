package tqk114514.terracuda.noise;

import tqk114514.terracuda.math.VanillaMath;
import tqk114514.terracuda.random.PositionalRandomFactory;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Mirrors {@code net.minecraft.world.level.levelgen.synth.PerlinNoise}: a stack of {@link ImprovedNoise}
 * octaves with vanilla's amplitude normalisation and the {@code 2^25} coordinate wrap.
 *
 * <p>Only the "new" initialisation path is implemented, because that is the one
 * {@code NormalNoise.create} uses for every world-gen noise. The legacy path
 * ({@code createLegacyForBlendedNoise}) draws from the source differently and is needed only for
 * {@code BlendedNoise}, which the design doc routes to the CPU fallback for now.
 *
 * <p>The {@code octave_i} names passed to {@code fromHashOf} are part of the output contract: they
 * decide each octave's permutation table, so they must not be renamed.
 */
public final class PerlinNoise {

    /** {@code 2^25}, the coordinate wrap period. */
    private static final double WRAP_PERIOD = 3.3554432E7;

    private final ImprovedNoise[] noiseLevels;
    private final int firstOctave;
    private final double[] amplitudes;
    private final double lowestFreqInputFactor;
    private final double lowestFreqValueFactor;
    private final double maxValue;

    /**
     * @param random      consumed for exactly one {@code forkPositional()} plus one
     *                    {@code fromHashOf} per non-zero amplitude, matching vanilla's draw order
     * @param firstOctave the octave index of {@code amplitudes[0]}
     * @param amplitudes  per-octave amplitudes; a copy is taken
     */
    public PerlinNoise(XoroshiroRandom random, int firstOctave, double[] amplitudes) {
        this.firstOctave = firstOctave;
        this.amplitudes = amplitudes.clone();
        int octaves = this.amplitudes.length;
        int zeroOctaveIndex = -this.firstOctave;
        this.noiseLevels = new ImprovedNoise[octaves];

        PositionalRandomFactory positional = random.forkPositional();
        for (int i = 0; i < octaves; i++) {
            if (this.amplitudes[i] != 0.0) {
                int octave = this.firstOctave + i;
                this.noiseLevels[i] = new ImprovedNoise(positional.fromHashOf("octave_" + octave));
            }
        }

        this.lowestFreqInputFactor = Math.pow(2.0, -zeroOctaveIndex);
        this.lowestFreqValueFactor = Math.pow(2.0, octaves - 1) / (Math.pow(2.0, octaves) - 1.0);
        this.maxValue = this.edgeValue(2.0);
    }

    public double maxValue() {
        return this.maxValue;
    }

    public double getValue(double x, double y, double z) {
        double value = 0.0;
        double factor = this.lowestFreqInputFactor;
        double valueFactor = this.lowestFreqValueFactor;

        for (int i = 0; i < this.noiseLevels.length; i++) {
            ImprovedNoise noise = this.noiseLevels[i];
            if (noise != null) {
                double noiseVal = noise.noise(wrap(x * factor), wrap(y * factor), wrap(z * factor), 0.0, 0.0);
                value += this.amplitudes[i] * noiseVal * valueFactor;
            }
            factor *= 2.0;
            valueFactor /= 2.0;
        }
        return value;
    }

    private double edgeValue(double noiseValue) {
        double value = 0.0;
        double valueFactor = this.lowestFreqValueFactor;

        for (int i = 0; i < this.noiseLevels.length; i++) {
            if (this.noiseLevels[i] != null) {
                value += this.amplitudes[i] * noiseValue * valueFactor;
            }
            valueFactor /= 2.0;
        }
        return value;
    }

    /** Wraps a coordinate into {@code [-2^24, 2^24)}; the {@code lfloor} shape matters for parity. */
    public static double wrap(double x) {
        return x - VanillaMath.lfloor(x / WRAP_PERIOD + 0.5) * WRAP_PERIOD;
    }

    public int firstOctave() {
        return this.firstOctave;
    }

    public double[] amplitudes() {
        return this.amplitudes.clone();
    }

    /** The octave at array index {@code index}, or {@code null} when its amplitude is zero. */
    public ImprovedNoise octave(int index) {
        return this.noiseLevels[index];
    }

    public int octaveCount() {
        return this.noiseLevels.length;
    }
}
