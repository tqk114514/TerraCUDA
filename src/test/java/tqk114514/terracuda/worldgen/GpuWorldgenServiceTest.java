package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.chunk.EmittedChunk;

/**
 * The dispatch side of M3: chunk generation runs on several worker threads, none of which may touch the
 * CUDA context, so the work goes through a service that owns one dedicated GPU thread.
 *
 * <p>Correctness of the emitted ids is covered by {@code GpuChunkFillerTest}; this checks that the
 * hand-off works and that an unavailable device degrades to "no ids" rather than an exception into the
 * generation path.
 */
class GpuWorldgenServiceTest {

    @Test
    void blockIdsComeBackFromTheGpuThread() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseGeneratorSettings settings = OverworldFixture.generatorSettings();
        try (GpuWorldgenService service = GpuWorldgenService.of(randomState, settings)) {
            assumeTrue(service.isReady(), service.unavailableReason());

            Optional<EmittedChunk> emitted = service.blockIds(3, -7);
            assertTrue(emitted.isPresent(), "the service reported ready but produced no ids");
            int[] ids = emitted.get().stateIds();
            assertTrue(ids.length > 0);

            // The ids must be ones vanilla would have chosen, which at minimum means real blocks.
            int air = Block.getId(net.minecraft.world.level.block.Blocks.AIR.defaultBlockState());
            int nonAir = 0;
            for (int id : ids) {
                if (id != air) {
                    nonAir++;
                }
            }
            assertTrue(nonAir > 0, "a chunk should not be entirely air");
        }
    }

    @Test
    void describeIsConsistentWithReadiness() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseGeneratorSettings settings = OverworldFixture.generatorSettings();
        try (GpuWorldgenService service = GpuWorldgenService.of(randomState, settings)) {
            assertFalse(service.describe().isBlank());
            if (service.isReady()) {
                assertTrue(service.unavailableReason().isEmpty());
            } else {
                assertFalse(service.unavailableReason().isBlank());
                assertTrue(service.describe().contains(service.unavailableReason()));
            }
        }
    }

    @Test
    void aClosedServiceProducesNoIds() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseGeneratorSettings settings = OverworldFixture.generatorSettings();
        GpuWorldgenService service = GpuWorldgenService.of(randomState, settings);
        service.close();

        assertFalse(service.isReady());
        assertTrue(service.blockIds(0, 0).isEmpty());
    }
}
