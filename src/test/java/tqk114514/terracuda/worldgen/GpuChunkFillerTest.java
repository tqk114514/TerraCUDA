package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;

/**
 * M3's K4, the emit half: the block ids a chunk is made of.
 *
 * <p>{@code GpuChunkFiller} turns the device's density into a full chunk's worth of block state ids —
 * device for the density, CPU for the material rules, which is the split the design doc describes for
 * the reference and which suits the shape of the work. Those ids are what crosses the boundary into
 * {@code ChunkReplay}, so they are worth pinning down on their own: everything upstream is verified, so
 * a failure here is the emit rather than the terrain.
 *
 * <p><b>What this does not cover.</b> The write-back into {@code LevelChunkSection}s is not unit
 * tested. Building a chunk needs a {@code PalettedContainerFactory}, which needs a {@code RegistryAccess}
 * carrying the datapack biome registry — reachable only through {@code VanillaRegistries}' builder, and
 * pulling that into a test would mean assembling registry plumbing that has nothing to do with the code
 * under test. {@code ChunkReplay} is a thin wrapper over vanilla's own
 * {@code LevelChunkSection.setBlockState(..., false)} and {@code Heightmap.update}, so the risk it
 * carries is small; it will be exercised for real the first time the mixin takes over generation.
 */
class GpuChunkFillerTest {

    private static final int[][] CHUNKS = {{3, -7}, {-40, 17}};

    @Test
    void theEmittedIdsAreTheBlockStatesVanillaChose() {
        RandomState randomState = OverworldFixture.randomState();
        NoiseGeneratorSettings settings = OverworldFixture.generatorSettings();

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                try (GpuChunkFiller filler = GpuChunkFiller.create(context, randomState, settings)) {
                    for (int[] chunk : CHUNKS) {
                        int[] ids = filler.blockIds(chunk[0], chunk[1]);
                        VanillaChunkReference.ChunkBlocks vanilla =
                                VanillaChunkReference.generate(settings, randomState, chunk[0], chunk[1]);

                        assertEquals(vanilla.size(), ids.length);
                        int mismatches = 0;
                        String[] first = new String[1];
                        for (int i = 0; i < ids.length; i++) {
                            if (ids[i] != vanilla.stateIds()[i]) {
                                if (mismatches == 0) {
                                    first[0] = "index " + i + " mine=" + Block.stateById(ids[i])
                                            + " vanilla=" + Block.stateById(vanilla.stateIds()[i]);
                                }
                                mismatches++;
                            }
                        }
                        int total = mismatches;
                        assertEquals(0, total, () -> "chunk (" + chunk[0] + ", " + chunk[1] + "): "
                                + total + " of " + ids.length + " ids differ; first: " + first[0]);
                    }
                }
            }
        }
    }

    @Test
    void theChunksUnderTestActuallyContainWater() {
        // The rules are only meaningfully exercised where they have something to place.
        RandomState randomState = OverworldFixture.randomState();
        int waterId = Block.getId(Blocks.WATER.defaultBlockState());
        int totalWater = 0;
        for (int[] chunk : CHUNKS) {
            for (int id : VanillaChunkReference.generate(randomState, chunk[0], chunk[1]).stateIds()) {
                if (id == waterId) {
                    totalWater++;
                }
            }
        }
        assertTrue(totalWater > 0, "none of the chunks under test contain water");
    }
}
