package tqk114514.terracuda.random;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import net.minecraft.world.level.levelgen.RandomSupport;
import org.junit.jupiter.api.Test;

class RandomSeedsTest {

    private static final long[] SEEDS = {
            0L, 1L, -1L, 42L, 123456789L, Long.MIN_VALUE, Long.MAX_VALUE,
            0x0123456789ABCDEFL, -0x0123456789ABCDEFL
    };

    @Test
    void mixStafford13MatchesVanilla() {
        for (long value : SEEDS) {
            assertEquals(RandomSupport.mixStafford13(value), RandomSeeds.mixStafford13(value),
                    "mixStafford13(" + value + ")");
        }
    }

    @Test
    void upgradeSeedTo128bitMatchesVanilla() {
        for (long seed : SEEDS) {
            RandomSupport.Seed128bit expected = RandomSupport.upgradeSeedTo128bit(seed);
            RandomSeeds.Seed128bit actual = RandomSeeds.upgradeSeedTo128bit(seed);
            assertEquals(expected.seedLo(), actual.seedLo(), "seedLo for seed " + seed);
            assertEquals(expected.seedHi(), actual.seedHi(), "seedHi for seed " + seed);
        }
    }

    @Test
    void seedFromHashOfMatchesVanilla() {
        List<String> names = List.of(
                "octave_-16", "octave_-9", "octave_-8", "octave_0", "octave_15",
                "offset", "aquifer", "ore", "noodle", "temperature", "");
        for (String name : names) {
            RandomSupport.Seed128bit expected = RandomSupport.seedFromHashOf(name);
            RandomSeeds.Seed128bit actual = RandomSeeds.seedFromHashOf(name);
            assertEquals(expected.seedLo(), actual.seedLo(), "seedLo for '" + name + "'");
            assertEquals(expected.seedHi(), actual.seedHi(), "seedHi for '" + name + "'");
        }
    }

    @Test
    void xorAndMixedMatchVanilla() {
        RandomSeeds.Seed128bit base = RandomSeeds.upgradeSeedTo128bit(12345L);
        RandomSupport.Seed128bit vanillaBase = RandomSupport.upgradeSeedTo128bit(12345L);

        RandomSeeds.Seed128bit mixed = base.mixed();
        RandomSupport.Seed128bit vanillaMixed = vanillaBase.mixed();
        assertEquals(vanillaMixed.seedLo(), mixed.seedLo());
        assertEquals(vanillaMixed.seedHi(), mixed.seedHi());

        RandomSeeds.Seed128bit xored = base.xor(0xDEADBEEFL, 0xFEEDFACECAFEBEEFL);
        RandomSupport.Seed128bit vanillaXored = vanillaBase.xor(0xDEADBEEFL, 0xFEEDFACECAFEBEEFL);
        assertEquals(vanillaXored.seedLo(), xored.seedLo());
        assertEquals(vanillaXored.seedHi(), xored.seedHi());
    }
}
