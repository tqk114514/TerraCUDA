package tqk114514.terracuda.noise;

import tqk114514.terracuda.math.VanillaMath;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * Mirrors {@code net.minecraft.world.level.levelgen.synth.ImprovedNoise}: a classic Perlin noise
 * octave with vanilla's 256-entry permutation table, per-octave origin offsets and the
 * {@code SimplexNoise.GRADIENT} corner table.
 *
 * <p>This is the single most-executed leaf in the whole generator ({@code old_blended_noise} alone is
 * three of these evaluated across 16 octaves per sample point), and the class the CUDA kernel has to
 * match bit for bit. The Java copy therefore exists as the oracle for the GPU port, and as the
 * receiver for the exported permutation tables described in the design doc (section 3.3).
 *
 * <p>The permutation table is exposed as a defensive copy so that the noise parameters can be
 * serialised to the device without reflection on private fields.
 */
public final class ImprovedNoise {

    /** {@code SimplexNoise.GRADIENT}, the 16 corner directions. */
    private static final int[][] GRADIENT = {
            {1, 1, 0},
            {-1, 1, 0},
            {1, -1, 0},
            {-1, -1, 0},
            {1, 0, 1},
            {-1, 0, 1},
            {1, 0, -1},
            {-1, 0, -1},
            {0, 1, 1},
            {0, -1, 1},
            {0, 1, -1},
            {0, -1, -1},
            {1, 1, 0},
            {0, -1, 1},
            {-1, 1, 0},
            {0, -1, -1}
    };

    private final byte[] p;
    public final double xo;
    public final double yo;
    public final double zo;

    /** Builds the octave from a random source, consuming draws exactly as vanilla does. */
    public ImprovedNoise(XoroshiroRandom random) {
        this.xo = random.nextDouble() * 256.0;
        this.yo = random.nextDouble() * 256.0;
        this.zo = random.nextDouble() * 256.0;
        this.p = new byte[256];

        for (int i = 0; i < 256; i++) {
            this.p[i] = (byte) i;
        }

        for (int i = 0; i < 256; i++) {
            int offset = random.nextInt(256 - i);
            byte tmp = this.p[i];
            this.p[i] = this.p[i + offset];
            this.p[i + offset] = tmp;
        }
    }

    /**
     * Rebuilds an octave from an exported permutation table and origin, bypassing seed derivation.
     *
     * @param permutation 256 entries; a copy is taken
     */
    public ImprovedNoise(byte[] permutation, double xo, double yo, double zo) {
        if (permutation.length != 256) {
            throw new IllegalArgumentException("permutation table must hold 256 entries, got " + permutation.length);
        }
        this.p = permutation.clone();
        this.xo = xo;
        this.yo = yo;
        this.zo = zo;
    }

    public double noise(double x, double y, double z) {
        return this.noise(x, y, z, 0.0, 0.0);
    }

    /**
     * The y-scaled variant used by {@code BlendedNoise}. {@code yScale} / {@code yFudge} are both
     * zero for ordinary terrain noise, which short-circuits the fudge branch.
     */
    public double noise(double x, double y, double z, double yScale, double yFudge) {
        double sx = x + this.xo;
        double sy = y + this.yo;
        double sz = z + this.zo;
        int xf = VanillaMath.floor(sx);
        int yf = VanillaMath.floor(sy);
        int zf = VanillaMath.floor(sz);
        double xr = sx - xf;
        double yr = sy - yf;
        double zr = sz - zf;
        double yrFudge;
        if (yScale != 0.0) {
            double fudgeLimit;
            if (yFudge >= 0.0 && yFudge < yr) {
                fudgeLimit = yFudge;
            } else {
                fudgeLimit = yr;
            }
            yrFudge = VanillaMath.floor(fudgeLimit / yScale + 1.0E-7F) * yScale;
        } else {
            yrFudge = 0.0;
        }
        return this.sampleAndLerp(xf, yf, zf, xr, yr - yrFudge, zr, yr);
    }

    private static double gradDot(int hash, double x, double y, double z) {
        int[] gradient = GRADIENT[hash & 15];
        return gradient[0] * x + gradient[1] * y + gradient[2] * z;
    }

    private int p(int x) {
        return this.p[x & 0xFF] & 0xFF;
    }

    private double sampleAndLerp(int x, int y, int z, double xr, double yr, double zr, double yrOriginal) {
        int x0 = this.p(x);
        int x1 = this.p(x + 1);
        int xy00 = this.p(x0 + y);
        int xy01 = this.p(x0 + y + 1);
        int xy10 = this.p(x1 + y);
        int xy11 = this.p(x1 + y + 1);
        double d000 = gradDot(this.p(xy00 + z), xr, yr, zr);
        double d100 = gradDot(this.p(xy10 + z), xr - 1.0, yr, zr);
        double d010 = gradDot(this.p(xy01 + z), xr, yr - 1.0, zr);
        double d110 = gradDot(this.p(xy11 + z), xr - 1.0, yr - 1.0, zr);
        double d001 = gradDot(this.p(xy00 + z + 1), xr, yr, zr - 1.0);
        double d101 = gradDot(this.p(xy10 + z + 1), xr - 1.0, yr, zr - 1.0);
        double d011 = gradDot(this.p(xy01 + z + 1), xr, yr - 1.0, zr - 1.0);
        double d111 = gradDot(this.p(xy11 + z + 1), xr - 1.0, yr - 1.0, zr - 1.0);
        double xAlpha = VanillaMath.smoothstep(xr);
        double yAlpha = VanillaMath.smoothstep(yrOriginal);
        double zAlpha = VanillaMath.smoothstep(zr);
        return VanillaMath.lerp3(xAlpha, yAlpha, zAlpha, d000, d100, d010, d110, d001, d101, d011, d111);
    }

    /** A copy of the 256-entry permutation table, for exporting to the device. */
    public byte[] permutation() {
        return this.p.clone();
    }
}
