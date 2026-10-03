package tqk114514.terracuda.random;

/**
 * The random source used by world generation, mirroring
 * {@code net.minecraft.world.level.levelgen.XoroshiroRandomSource}.
 *
 * <p>Only the surface that the noise pipeline touches is reproduced: the integer, boolean and float
 * draws plus the two fork operations. {@code nextGaussian()} is omitted because terrain generation
 * never calls it; adding it later means porting {@code MarsagliaPolarGaussian} as well.
 *
 * <p>Two details deserve a warning because they look like typos and are not:
 * <ul>
 *   <li>{@link #nextDouble()} multiplies by a <em>float</em> literal, so the arithmetic happens in
 *       {@code float} and only the result is widened to {@code double}. This was verified against the
 *       vanilla bytecode ({@code l2f / fmul / f2d}).</li>
 *   <li>{@link #nextInt(int)} multiplies by the bound in {@code long} arithmetic and rejects from the
 *       low 32 bits, which is what makes it identical to vanilla's unbiased sampling.</li>
 * </ul>
 */
public final class XoroshiroRandom {

    private static final float FLOAT_UNIT = 5.9604645E-8F;

    private Xoroshiro128PlusPlus generator;

    public XoroshiroRandom(long seed) {
        this.generator = new Xoroshiro128PlusPlus(RandomSeeds.upgradeSeedTo128bit(seed));
    }

    public XoroshiroRandom(RandomSeeds.Seed128bit seed) {
        this.generator = new Xoroshiro128PlusPlus(seed);
    }

    public XoroshiroRandom(long seedLo, long seedHi) {
        this.generator = new Xoroshiro128PlusPlus(seedLo, seedHi);
    }

    public long nextLong() {
        return this.generator.nextLong();
    }

    public int nextInt() {
        return (int) this.generator.nextLong();
    }

    public int nextInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("Bound must be positive");
        }
        long randomBits = Integer.toUnsignedLong(this.nextInt());
        long multipliedRandomBits = randomBits * bound;
        long fractionalPart = multipliedRandomBits & 4294967295L;
        if (fractionalPart < bound) {
            for (int unbiasedBucketsStartIndex = Integer.remainderUnsigned(~bound + 1, bound);
                    fractionalPart < unbiasedBucketsStartIndex;
                    fractionalPart = multipliedRandomBits & 4294967295L) {
                randomBits = Integer.toUnsignedLong(this.nextInt());
                multipliedRandomBits = randomBits * bound;
            }
        }
        return (int) (multipliedRandomBits >> 32);
    }

    public boolean nextBoolean() {
        return (this.generator.nextLong() & 1L) != 0L;
    }

    public float nextFloat() {
        return (float) this.nextBits(24) * FLOAT_UNIT;
    }

    public double nextDouble() {
        // NOT a typo, and not double arithmetic: vanilla's bytecode is
        //   nextBits(53) -> l2f -> ldc 1.110223E-16f -> fmul -> f2d
        // so the multiplication happens in float and only the result is widened. Using a double
        // literal here would be more "correct" but would produce different terrain.
        return this.nextBits(53) * 1.110223E-16F;
    }

    /** Advances the generator by {@code rounds} draws, used to skip unused octaves. */
    public void consumeCount(int rounds) {
        for (int i = 0; i < rounds; i++) {
            this.generator.nextLong();
        }
    }

    /** Forks a fresh source from the next two draws. */
    public XoroshiroRandom fork() {
        return new XoroshiroRandom(this.generator.nextLong(), this.generator.nextLong());
    }

    /** Forks a positional factory from the next two draws. */
    public PositionalRandomFactory forkPositional() {
        return new PositionalRandomFactory(this.generator.nextLong(), this.generator.nextLong());
    }

    private long nextBits(int bits) {
        return this.generator.nextLong() >>> 64 - bits;
    }
}
