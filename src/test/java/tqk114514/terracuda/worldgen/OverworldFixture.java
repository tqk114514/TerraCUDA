package tqk114514.terracuda.worldgen;

import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;

/**
 * The real overworld worldgen inputs, built once per test JVM.
 *
 * <p>{@code VanillaRegistries.createLookup()} assembles the whole vanilla registry set, which is slow,
 * so the {@link RandomState} is cached. It is immutable once built, so sharing it is safe.
 */
public final class OverworldFixture {

    public static final long SEED = 4242L;
    public static final int SAMPLE_COUNT = 200;

    private static RandomState randomState;

    private OverworldFixture() {
    }

    public static synchronized RandomState randomState() {
        if (randomState == null) {
            Bootstrap.bootStrap();
            HolderLookup.Provider registries = VanillaRegistries.createLookup();
            randomState = RandomState.create(registries, NoiseGeneratorSettings.OVERWORLD, SEED);
        }
        return randomState;
    }

    /** The overworld cell geometry: min y, height, and the noise cell sizes. */
    public static NoiseSettings noiseSettings() {
        Bootstrap.bootStrap();
        HolderLookup.Provider registries = VanillaRegistries.createLookup();
        return registries.lookupOrThrow(Registries.NOISE_SETTINGS)
                .getOrThrow(NoiseGeneratorSettings.OVERWORLD)
                .value()
                .noiseSettings();
    }

    /** Deterministic, well-spread sample coordinates in {@code [-1000, 1000)}. */
    public static double coordinate(int index, int salt) {
        long h = (index + 1L) * 6364136223846793005L + (salt + 1L) * 1442695040888963407L;
        return ((h >>> 11) * 0x1.0p-53) * 2000.0 - 1000.0;
    }
}
