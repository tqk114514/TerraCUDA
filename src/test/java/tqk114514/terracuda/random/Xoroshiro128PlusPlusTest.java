package tqk114514.terracuda.random;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * Parity between {@link Xoroshiro128PlusPlus} and
 * {@code net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus}.
 *
 * <p>The vanilla type is referenced fully qualified because our class deliberately shares its simple
 * name.
 */
class Xoroshiro128PlusPlusTest {

    private static final long[][] SEED_PAIRS = {
            {1L, 2L},
            {0L, 0L},
            {-1L, -1L},
            {42L, 42L},
            {Long.MIN_VALUE, Long.MAX_VALUE},
            {RandomSeeds.GOLDEN_RATIO_64, RandomSeeds.SILVER_RATIO_64},
            {0x0123456789ABCDEFL, -0x0123456789ABCDEFL}
    };

    @Test
    void nextLongSequenceMatchesVanilla() {
        for (long[] pair : SEED_PAIRS) {
            net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus vanilla =
                    new net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus(pair[0], pair[1]);
            Xoroshiro128PlusPlus mine = new Xoroshiro128PlusPlus(pair[0], pair[1]);
            for (int i = 0; i < 1000; i++) {
                assertEquals(vanilla.nextLong(), mine.nextLong(),
                        "draw " + i + " for seed pair (" + pair[0] + ", " + pair[1] + ")");
            }
        }
    }

    @Test
    void allZeroSeedFallsBackToTheRatioConstants() {
        net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus vanilla =
                new net.minecraft.world.level.levelgen.Xoroshiro128PlusPlus(0L, 0L);
        Xoroshiro128PlusPlus mine = new Xoroshiro128PlusPlus(0L, 0L);
        assertEquals(vanilla.nextLong(), mine.nextLong());
        assertEquals(vanilla.nextLong(), mine.nextLong());
    }

    @Test
    void seed128bitConstructorMatchesTheTwoLongConstructor() {
        RandomSeeds.Seed128bit seed = RandomSeeds.upgradeSeedTo128bit(987654321L);
        Xoroshiro128PlusPlus fromRecord = new Xoroshiro128PlusPlus(seed);
        Xoroshiro128PlusPlus fromLongs = new Xoroshiro128PlusPlus(seed.seedLo(), seed.seedHi());
        for (int i = 0; i < 64; i++) {
            assertEquals(fromLongs.nextLong(), fromRecord.nextLong());
        }
    }
}
