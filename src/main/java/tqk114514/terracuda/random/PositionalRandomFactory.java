package tqk114514.terracuda.random;

import tqk114514.terracuda.math.VanillaMath;

/**
 * Mirrors {@code XoroshiroRandomSource.XoroshiroPositionalRandomFactory}: derives a deterministic
 * {@link XoroshiroRandom} from a block position, a string name or an explicit seed.
 *
 * <p>This is how every {@code ImprovedNoise} octave in a {@code PerlinNoise} is seeded
 * ({@code fromHashOf("octave_" + octave)}) and how the aquifer and ore vein generators are seeded
 * ({@code at(x, y, z)}), so it sits directly on the hot path of the GPU port.
 */
public final class PositionalRandomFactory {

    private final long seedLo;
    private final long seedHi;

    public PositionalRandomFactory(long seedLo, long seedHi) {
        this.seedLo = seedLo;
        this.seedHi = seedHi;
    }

    /** A source bound to a block position, via {@code Mth.getSeed}. */
    public XoroshiroRandom at(int x, int y, int z) {
        long positionalSeed = VanillaMath.getSeed(x, y, z);
        long randomSeed = positionalSeed ^ this.seedLo;
        return new XoroshiroRandom(randomSeed, this.seedHi);
    }

    /** A source derived from {@code MD5(name)}. */
    public XoroshiroRandom fromHashOf(String name) {
        RandomSeeds.Seed128bit seed = RandomSeeds.seedFromHashOf(name);
        return new XoroshiroRandom(seed.xor(this.seedLo, this.seedHi));
    }

    public XoroshiroRandom fromSeed(long seed) {
        return new XoroshiroRandom(seed ^ this.seedLo, seed ^ this.seedHi);
    }

    public long seedLo() {
        return seedLo;
    }

    public long seedHi() {
        return seedHi;
    }
}
