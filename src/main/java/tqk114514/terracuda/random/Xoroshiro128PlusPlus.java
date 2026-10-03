package tqk114514.terracuda.random;

/**
 * The XOROSHIRO128++ generator, mirroring
 * {@code net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus}.
 *
 * <p>State is two {@code long}s. Unlike the reference algorithm, vanilla substitutes the golden and
 * silver ratio constants instead of rejecting an all-zero state; that quirk is part of the observable
 * output and is reproduced here.
 *
 * <p>Not thread-safe, matching the vanilla generator. Each thread that samples noise needs its own
 * instance, which is also the model the CUDA kernels follow (one state per thread).
 */
public final class Xoroshiro128PlusPlus {

    private long seedLo;
    private long seedHi;

    public Xoroshiro128PlusPlus(RandomSeeds.Seed128bit seed) {
        this(seed.seedLo(), seed.seedHi());
    }

    public Xoroshiro128PlusPlus(long seedLo, long seedHi) {
        this.seedLo = seedLo;
        this.seedHi = seedHi;
        if ((this.seedLo | this.seedHi) == 0L) {
            this.seedLo = RandomSeeds.GOLDEN_RATIO_64;
            this.seedHi = RandomSeeds.SILVER_RATIO_64;
        }
    }

    public long nextLong() {
        long s0 = this.seedLo;
        long s1 = this.seedHi;
        long result = Long.rotateLeft(s0 + s1, 17) + s0;
        s1 ^= s0;
        this.seedLo = Long.rotateLeft(s0, 49) ^ s1 ^ s1 << 21;
        this.seedHi = Long.rotateLeft(s1, 28);
        return result;
    }
}
