package tqk114514.terracuda.noise;

import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Mirrors {@code net.minecraft.world.level.levelgen.synth.NormalNoise}: two {@link PerlinNoise}
 * instances sampled at slightly different coordinates and averaged, which is what gives the output an
 * approximately normal distribution.
 *
 * <p>Constructing one consumes the random source for exactly two {@code PerlinNoise} builds, in
 * order — {@code first} then {@code second}. Reversing them still produces a valid-looking noise, but
 * not the vanilla one, so the order is part of the contract.
 */
public final class NormalNoise {

    private static final double INPUT_FACTOR = 1.0181268882175227;

    private final double valueFactor;
    private final PerlinNoise first;
    private final PerlinNoise second;
    private final double maxValue;
    private final NoiseParameters parameters;

    public static NormalNoise create(XoroshiroRandom random, NoiseParameters parameters) {
        return new NormalNoise(random, parameters);
    }

    private NormalNoise(XoroshiroRandom random, NoiseParameters parameters) {
        int firstOctave = parameters.firstOctave();
        double[] amplitudes = parameters.amplitudes();
        this.parameters = parameters;
        this.first = new PerlinNoise(random, firstOctave, amplitudes);
        this.second = new PerlinNoise(random, firstOctave, amplitudes);

        int minOctave = Integer.MAX_VALUE;
        int maxOctave = Integer.MIN_VALUE;
        for (int i = 0; i < amplitudes.length; i++) {
            if (amplitudes[i] != 0.0) {
                minOctave = Math.min(minOctave, i);
                maxOctave = Math.max(maxOctave, i);
            }
        }

        this.valueFactor = 0.16666666666666666 / expectedDeviation(maxOctave - minOctave);
        this.maxValue = (this.first.maxValue() + this.second.maxValue()) * this.valueFactor;
    }

    public double maxValue() {
        return this.maxValue;
    }

    private static double expectedDeviation(int octaveSpan) {
        return 0.1 * (1.0 + 1.0 / (octaveSpan + 1));
    }

    public double getValue(double x, double y, double z) {
        double x2 = x * INPUT_FACTOR;
        double y2 = y * INPUT_FACTOR;
        double z2 = z * INPUT_FACTOR;
        return (this.first.getValue(x, y, z) + this.second.getValue(x2, y2, z2)) * this.valueFactor;
    }

    public NoiseParameters parameters() {
        return this.parameters;
    }

    public PerlinNoise first() {
        return this.first;
    }

    public PerlinNoise second() {
        return this.second;
    }
}
