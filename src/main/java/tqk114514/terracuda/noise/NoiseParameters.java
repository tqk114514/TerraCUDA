package tqk114514.terracuda.noise;

import java.util.Arrays;

/**
 * Mirrors {@code NormalNoise.NoiseParameters}: the octave layout of a world-gen noise.
 *
 * <p>Implemented as a record with an array component, so {@code equals}/{@code hashCode}/{@code toString}
 * are overridden to be value-based — the generated versions would compare array identity, which makes
 * the type useless as a map key when the design doc's DAG lowering deduplicates noise instances.
 *
 * @param firstOctave the octave index of {@code amplitudes[0]}
 * @param amplitudes  per-octave amplitudes, indexed by {@code octave - firstOctave}
 */
public record NoiseParameters(int firstOctave, double[] amplitudes) {

    public NoiseParameters {
        amplitudes = amplitudes.clone();
    }

    /** Convenience form matching {@code NormalNoise.NoiseParameters(int, double, double...)}. */
    public static NoiseParameters of(int firstOctave, double firstAmplitude, double... rest) {
        double[] amplitudes = new double[rest.length + 1];
        amplitudes[0] = firstAmplitude;
        System.arraycopy(rest, 0, amplitudes, 1, rest.length);
        return new NoiseParameters(firstOctave, amplitudes);
    }

    @Override
    public double[] amplitudes() {
        return this.amplitudes.clone();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        return other instanceof NoiseParameters that
                && this.firstOctave == that.firstOctave
                && Arrays.equals(this.amplitudes, that.amplitudes);
    }

    @Override
    public int hashCode() {
        return 31 * Integer.hashCode(this.firstOctave) + Arrays.hashCode(this.amplitudes);
    }

    @Override
    public String toString() {
        return "NoiseParameters[firstOctave=" + this.firstOctave + ", amplitudes=" + Arrays.toString(this.amplitudes) + "]";
    }
}
