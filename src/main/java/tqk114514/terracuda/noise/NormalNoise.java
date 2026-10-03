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

    /**
     * Rebuilds a noise from octave stacks that were exported out of a live vanilla instance.
     *
     * <p>Used by the device-upload path: the permutation tables and origins are mirrored rather than
     * re-derived, so the reconstruction is identical no matter how the world was seeded.
     */
    public static NormalNoise fromPerlinNoise(NoiseParameters parameters, PerlinNoise first, PerlinNoise second) {
        return new NormalNoise(parameters, first, second);
    }

    private NormalNoise(XoroshiroRandom random, NoiseParameters parameters) {
        // The two PerlinNoise builds must happen in this order: `first` then `second`. Reversing them
        // still yields a plausible noise, but not the vanilla one.
        this(parameters,
                new PerlinNoise(random, parameters.firstOctave(), parameters.amplitudes()),
                new PerlinNoise(random, parameters.firstOctave(), parameters.amplitudes()));
    }

    private NormalNoise(NoiseParameters parameters, PerlinNoise first, PerlinNoise second) {
        double[] amplitudes = parameters.amplitudes();
        this.parameters = parameters;
        this.first = first;
        this.second = second;

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
