package tqk114514.terracuda.random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;

/**
 * Parity between {@link XoroshiroRandom} and {@code XoroshiroRandomSource}.
 *
 * <p>Everything the terrain pipeline draws from a random source is covered here, including the
 * rejection-sampling loop in {@code nextInt(bound)} — the path the aquifer's {@code nextInt(10)} /
 * {@code nextInt(9)} offsets take.
 */
class XoroshiroRandomTest {

    private static final long[] SEEDS = {
            0L, 1L, -1L, 42L, 123456789L, Long.MIN_VALUE, Long.MAX_VALUE, 0x0123456789ABCDEFL
    };

    @Test
    void longIntAndBooleanDrawsMatchVanilla() {
        for (long seed : SEEDS) {
            XoroshiroRandomSource vanilla = new XoroshiroRandomSource(seed);
            XoroshiroRandom mine = new XoroshiroRandom(seed);

            for (int i = 0; i < 500; i++) {
                assertEquals(vanilla.nextLong(), mine.nextLong(), "nextLong seed=" + seed);
            }
            for (int i = 0; i < 500; i++) {
                assertEquals(vanilla.nextInt(), mine.nextInt(), "nextInt seed=" + seed);
            }
            for (int i = 0; i < 500; i++) {
                assertEquals(vanilla.nextBoolean(), mine.nextBoolean(), "nextBoolean seed=" + seed);
            }
        }
    }

    @Test
    void floatAndDoubleDrawsMatchVanilla() {
        for (long seed : SEEDS) {
            XoroshiroRandomSource vanilla = new XoroshiroRandomSource(seed);
            XoroshiroRandom mine = new XoroshiroRandom(seed);

            for (int i = 0; i < 1000; i++) {
                TestParity.assertFloatIdentical(vanilla.nextFloat(), mine.nextFloat(), "nextFloat seed=" + seed);
            }
            for (int i = 0; i < 1000; i++) {
                TestParity.assertDoubleIdentical(vanilla.nextDouble(), mine.nextDouble(), "nextDouble seed=" + seed);
            }
        }
    }

    @Test
    void nextIntWithBoundMatchesVanillaIncludingTheRejectionLoop() {
        int[] bounds = {1, 2, 3, 7, 9, 10, 16, 17, 100, 255, 256, 257, 1000, 65536, Integer.MAX_VALUE};
        for (long seed : SEEDS) {
            for (int bound : bounds) {
                XoroshiroRandomSource vanilla = new XoroshiroRandomSource(seed);
                XoroshiroRandom mine = new XoroshiroRandom(seed);
                for (int i = 0; i < 500; i++) {
                    assertEquals(vanilla.nextInt(bound), mine.nextInt(bound),
                            "nextInt(" + bound + ") draw " + i + " seed=" + seed);
                }
            }
        }
    }

    @Test
    void consumeCountMatchesVanilla() {
        for (long seed : SEEDS) {
            XoroshiroRandomSource vanilla = new XoroshiroRandomSource(seed);
            XoroshiroRandom mine = new XoroshiroRandom(seed);

            vanilla.consumeCount(262);
            mine.consumeCount(262);
            assertEquals(vanilla.nextLong(), mine.nextLong(), "after consumeCount(262) seed=" + seed);
        }
    }

    @Test
    void forkMatchesVanilla() {
        for (long seed : SEEDS) {
            XoroshiroRandomSource vanilla = new XoroshiroRandomSource(seed);
            XoroshiroRandom mine = new XoroshiroRandom(seed);

            assertEquals(vanilla.fork().nextLong(), mine.fork().nextLong(), "fork seed=" + seed);
            // The parent's own stream must keep advancing identically afterwards.
            assertEquals(vanilla.nextLong(), mine.nextLong(), "parent after fork seed=" + seed);
        }
    }

    @Test
    void positionalFactoryMatchesVanilla() {
        for (long seed : SEEDS) {
            var vanillaFactory = new XoroshiroRandomSource(seed).forkPositional();
            PositionalRandomFactory mine = new XoroshiroRandom(seed).forkPositional();

            for (int x = -32; x <= 32; x += 8) {
                for (int y = -64; y <= 320; y += 96) {
                    for (int z = -32; z <= 32; z += 8) {
                        assertEquals(vanillaFactory.at(x, y, z).nextLong(),
                                mine.at(x, y, z).nextLong(),
                                "at(" + x + ", " + y + ", " + z + ") seed=" + seed);
                    }
                }
            }

            for (String name : new String[] {"octave_-16", "octave_0", "offset", "aquifer"}) {
                assertEquals(vanillaFactory.fromHashOf(name).nextLong(),
                        mine.fromHashOf(name).nextLong(),
                        "fromHashOf(" + name + ") seed=" + seed);
            }

            assertEquals(vanillaFactory.fromSeed(seed).nextLong(),
                    mine.fromSeed(seed).nextLong(),
                    "fromSeed seed=" + seed);
        }
    }

    @Test
    void nonPositiveBoundsAreRejected() {
        XoroshiroRandom random = new XoroshiroRandom(1L);
        assertThrows(IllegalArgumentException.class, () -> random.nextInt(0));
        assertThrows(IllegalArgumentException.class, () -> random.nextInt(-7));
    }
}
