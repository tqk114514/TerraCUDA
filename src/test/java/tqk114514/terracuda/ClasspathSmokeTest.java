package tqk114514.terracuda;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import net.minecraft.util.Mth;
import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import org.junit.jupiter.api.Test;

/**
 * Sanity check that the test source set can see both JUnit 5 and the deobfuscated
 * Minecraft / NeoForge classes. Every parity test in this project depends on it.
 */
class ClasspathSmokeTest {

    @Test
    void junitAndVanillaAreBothOnTheTestClasspath() {
        var random = new XoroshiroRandomSource(42L);
        assertDoesNotThrow(random::nextLong);
        assertEquals(0, Mth.floor(0.5));
        assertEquals(-1, Mth.floor(-0.5));
    }
}
