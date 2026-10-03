package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.List;
import java.util.Optional;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.gpu.ChunkCornerTables;

/**
 * The dispatch side of M2: chunk generation runs on several worker threads, none of which may touch
 * the CUDA context, so the work goes through a service that owns one dedicated GPU thread.
 *
 * <p>Bit-exactness of the tables themselves is covered by {@code ChunkCornerTablesTest}; this checks
 * that the hand-off works and that an unavailable device degrades to "no tables" rather than an
 * exception into the generation path.
 */
class GpuWorldgenServiceTest {

    @Test
    void markerTablesComeBackFromTheGpuThread() {
        RandomState randomState = OverworldFixture.randomState();
        try (GpuWorldgenService service = GpuWorldgenService.of(randomState)) {
            assumeTrue(service.isReady(), service.unavailableReason());

            NoiseSettings settings = OverworldFixture.noiseSettings();
            Optional<List<ChunkCornerTables.MarkerGrid>> maybe = service.markerTables(settings, 3, -7);
            assertTrue(maybe.isPresent(), "the service reported ready but produced no tables");

            List<ChunkCornerTables.MarkerGrid> grids = maybe.get();
            assertFalse(grids.isEmpty());
            for (ChunkCornerTables.MarkerGrid grid : grids) {
                assertTrue(grid.pointCount() > 0);
                for (double value : grid.values()) {
                    assertFalse(Double.isNaN(value),
                            "marker kind " + grid.kind() + " produced a NaN");
                }
            }
        }
    }

    @Test
    void describeIsConsistentWithReadiness() {
        RandomState randomState = OverworldFixture.randomState();
        try (GpuWorldgenService service = GpuWorldgenService.of(randomState)) {
            assertFalse(service.describe().isBlank());
            if (service.isReady()) {
                assertTrue(service.unavailableReason().isEmpty());
                assertTrue(service.describe().contains("instructions"));
            } else {
                assertFalse(service.unavailableReason().isBlank());
                assertTrue(service.describe().contains(service.unavailableReason()));
            }
        }
    }

    @Test
    void aClosedServiceProducesNoTables() {
        RandomState randomState = OverworldFixture.randomState();
        GpuWorldgenService service = GpuWorldgenService.of(randomState);
        service.close();

        assertFalse(service.isReady());
        assertTrue(service.markerTables(OverworldFixture.noiseSettings(), 0, 0).isEmpty());
    }
}
