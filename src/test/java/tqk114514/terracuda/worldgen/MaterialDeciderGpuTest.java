package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;

import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.chunk.EmittedChunk;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;

/**
 * G2a's acceptance: the per-block decision kernel must answer with the same block state and the
 * same fluid-tick request as the CPU rules, on real overworld chunks, block for block.
 *
 * <p>The CPU reference is {@link GpuChunkFiller#blockIds} — the full synchronous path whose own
 * parity against vanilla is pinned elsewhere, so this closes the G2a chain: vanilla == CPU ==
 * GPU. The chunks are chosen to cover the aquifer's decisions — open sea for water, depth for
 * lava, coastlines for the barrier band, and the veinifier's y windows ride along in every
 * column.
 */
class MaterialDeciderGpuTest {

    /** Chunks with water, shores and open sea around the origin of the fixture seed. */
    private static final int[][] CHUNKS = {
        {0, 0}, {1, -1}, {-3, 2}, {7, 5}, {-12, 9}, {20, -20},
    };

    @Test
    void theDeciderMatchesTheCpuRulesBlockForBlock() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                RandomState randomState = OverworldFixture.randomState();
                var settings = OverworldFixture.generatorSettings();
                GpuChunkFiller filler = GpuChunkFiller.create(context, randomState, settings);
                GpuChunkFiller.RulesContext rulesContext = filler.createRulesContext();

                var oreRandom = VanillaRandomExport.export(randomState.oreRandom());
                var aquiferRandom = VanillaRandomExport.export(randomState.aquiferRandom());
                try (MaterialDeciderGpu decider = new MaterialDeciderGpu(context,
                        OverworldFixture.noiseSettings(), settings.seaLevel(),
                        oreRandom.seedLo(), oreRandom.seedHi(),
                        aquiferRandom.seedLo(), aquiferRandom.seedHi(),
                        settings.isAquifersEnabled(), settings.oreVeinsEnabled(),
                        rulesContext.barrierNoise, rulesContext.barrierNoise.program(),
                        rulesContext.veinGap, rulesContext.veinGap.program())) {

                    for (int[] chunk : CHUNKS) {
                        int chunkX = chunk[0];
                        int chunkZ = chunk[1];
                        int height = OverworldFixture.noiseSettings().height();

                        // The CPU reference: the full synchronous path, beard-free.
                        EmittedChunk expected = filler.blockIds(chunkX, chunkZ);

                        // The GPU path: the same densities, the same wiring's grid tables.
                        GpuChunkFiller.DensityBuffers buffers = filler.emitDensities(chunkX, chunkZ);
                        MaterialRules rules = filler.materialRulesFor(chunkX, chunkZ, rulesContext,
                                buffers.veinToggle(), buffers.veinRidged());
                        MaterialDeciderGpu.Decisions actual = decider.decide(chunkX, chunkZ,
                                buffers.density(), buffers.veinToggle(), buffers.veinRidged(),
                                rules);
                        filler.releaseBuffers(buffers);

                        int mismatches = 0;
                        String[] first = new String[1];
                        for (int blockIndex = 0; blockIndex < actual.states().length; blockIndex++) {
                            int yLocal = blockIndex % height;
                            int rest = blockIndex / height;
                            int zLocal = rest % 16;
                            int xLocal = rest / 16;
                            // The reference carries registry ids; map back through the registry
                            // rather than comparing raw numbers.
                            var expectedState = net.minecraft.world.level.block.Block
                                    .stateById(expected.stateIds()[blockIndex]);
                            var gpuState = MaterialDeciderGpu.stateOf(
                                    actual.states()[blockIndex], settings.defaultBlock());
                            boolean expectedFlag = (expected.fluidUpdates()[blockIndex >>> 6]
                                    & (1L << (blockIndex & 63))) != 0L;
                            boolean gpuFlag = actual.fluidTicks()[blockIndex] != 0
                                    && !gpuState.getFluidState().isEmpty();

                            if (expectedState != gpuState || expectedFlag != gpuFlag) {
                                if (first[0] == null) {
                                    first[0] = "block (" + (chunkX * 16 + xLocal) + ", "
                                            + (OverworldFixture.noiseSettings().minY() + yLocal) + ", "
                                            + (chunkZ * 16 + zLocal) + "): expected "
                                            + expectedState + " (flag " + expectedFlag + "), got "
                                            + gpuState + " (flag " + gpuFlag + ")";
                                }
                                mismatches++;
                            }
                        }
                        int total = mismatches;
                        assertEquals(0, total, () -> total + " of "
                                + actual.states().length + " blocks differ in chunk (" + chunkX
                                + ", " + chunkZ + "); first mismatch: " + first[0]);
                    }
                }
            }
        }
    }
}
