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
import tqk114514.terracuda.chunk.EmittedChunk;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;

/**
 * M3's K4, the emit half: the blocks a chunk is made of, and the positions it wants a fluid tick for.
 *
 * <p>{@code GpuChunkFiller} turns the device's density into a whole chunk's worth of blocks — device
 * for the density, CPU for the material rules, which is the split the design doc describes for the
 * reference and which suits the shape of the work. Those blocks are what crosses the boundary into
 * {@code ChunkReplay}, so they are worth pinning down on their own: everything upstream is verified,
 * so a failure here is the emit rather than the terrain.
 *
 * <p>The write-back itself is covered by {@code ChunkReplayTest}, which needs no device.
 */
class GpuChunkFillerTest {

    /** Land with water, and a chunk dense with ore veins. */
    private static final int[][] CHUNKS = {{3, -7}, {-40, 17}};

    /**
     * Chunks whose aquifer actually asks for a fluid update.
     *
     * <p>Most chunks do not, and that is the point of listing these separately: a chunk where both
     * sides flag nothing would pass the comparison below without exercising it. These three were found
     * by scanning two dozen, and they flag 155, 67 and 145 blocks respectively.
     */
    private static final int[][] FLUID_CHUNKS = {{44, -8}, {-50, -50}, {30, -30}};

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
                        EmittedChunk emitted = filler.blockIds(chunk[0], chunk[1]);
                        int[] ids = emitted.stateIds();
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

    /**
     * The fluid-update bitmap, exactly.
     *
     * <p>This is what makes aquifer water run into the caves beside it, so a chunk with the right
     * blocks and the wrong bitmap is wrong in a way the block ids cannot show. Vanilla's own aquifer
     * says which positions those are, so this compares against it rather than against a smoke test.
     */
    @Test
    void theFluidUpdateFlagsMatchVanillaWhereThereAreAny() {
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
                    for (int[] chunk : FLUID_CHUNKS) {
                        EmittedChunk emitted = filler.blockIds(chunk[0], chunk[1]);
                        VanillaChunkReference.ChunkBlocks vanilla =
                                VanillaChunkReference.generate(settings, randomState, chunk[0], chunk[1]);

                        int flagged = 0;
                        int mismatches = 0;
                        String[] first = new String[1];
                        for (int i = 0; i < emitted.size(); i++) {
                            boolean mine = emitted.fluidUpdate(i);
                            boolean expected = vanilla.fluidUpdateAt(i);
                            if (expected) {
                                flagged++;
                            }
                            if (mine != expected) {
                                if (mismatches == 0) {
                                    first[0] = "index " + i + " mine=" + mine + " vanilla=" + expected;
                                }
                                mismatches++;
                            }
                        }
                        int total = mismatches;
                        int expectedFlags = flagged;
                        assertTrue(expectedFlags > 0, "chunk (" + chunk[0] + ", " + chunk[1]
                                + ") is in FLUID_CHUNKS but vanilla flagged nothing, so this case "
                                + "would pass without testing anything");
                        assertEquals(0, total, () -> "chunk (" + chunk[0] + ", " + chunk[1] + "): "
                                + total + " of " + expectedFlags
                                + " fluid-update flags differ; first: " + first[0]);
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
