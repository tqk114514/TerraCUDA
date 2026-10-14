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

    /**
     * Diagnostic companion: the grid kernel's tables against the CPU's eager export, cell by
     * cell, with the failing cell's coordinates spelled out.
     */
    @Test
    void theGridKernelMatchesTheCpuTablesCellByCell() {
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
                        rulesContext)) {

                    int chunkX = 1;
                    int chunkZ = -1;
                    GpuChunkFiller.DensityBuffers buffers = filler.emitDensities(chunkX, chunkZ);
                    MaterialRules rules = filler.materialRulesFor(chunkX, chunkZ, rulesContext,
                            buffers.veinToggle(), buffers.veinRidged());
                    MaterialRules.GridTables expected = rules.exportGrid();


                                + " |" + trace);
                    }

                    MaterialDeciderGpu.GridView view = decider.debugGrid(chunkX, chunkZ);
                    assertEquals(expected.minGridX(), view.minGridX());
                    assertEquals(expected.gridSizeX(), view.gridSizeX());
                    assertEquals(expected.skipSamplingAboveY(), view.skipSamplingAboveY(),
                            "skip differs: cpu " + expected.skipSamplingAboveY()
                                    + " gpu " + view.skipSamplingAboveY());


                    var arena = java.lang.foreign.Arena.ofConfined();
                    var hostLevels = arena.allocate((long) expected.levels().length * 4L);
                    var hostTypes = arena.allocate((long) expected.levels().length * 4L);
                    context.copyFromDevice(decider.debugLevelsBuffer(), hostLevels);
                    context.copyFromDevice(decider.debugTypesBuffer(), hostTypes);

                    int total = expected.levels().length;
                    int mismatches = 0;
                    String[] first = new String[1];
                    for (int index = 0; index < total; index++) {
                        int cpuLevel = expected.levels()[index];
                        int cpuType = expected.types()[index];
                        int gpuLevel = hostLevels.get(java.lang.foreign.ValueLayout.JAVA_INT,
                                (long) index * 4L);
                        int gpuType = hostTypes.get(java.lang.foreign.ValueLayout.JAVA_INT,
                                (long) index * 4L);
                        if (cpuLevel != gpuLevel || cpuType != gpuType) {
                            if (first[0] == null) {
                                first[0] = "cell " + index + " (grid "
                                        + (expected.minGridX() + index % expected.gridSizeX())
                                        + ", " + (expected.minGridY()
                                                + (index / expected.gridSizeX()) / expected.gridSizeZ())
                                        + ", " + (expected.minGridZ()
                                                + (index / expected.gridSizeX()) % expected.gridSizeZ())
                                        + "): cpu level " + cpuLevel + " type " + cpuType
                                        + ", gpu level " + gpuLevel + " type " + gpuType;
                            }
                            mismatches++;
                        }
                    }
                    filler.releaseBuffers(buffers);
                    int totalMismatches = mismatches;
                    assertEquals(0, totalMismatches, () -> totalMismatches + " of " + total
                            + " cells differ; first: " + first[0]);
                }
            }
        }
    }

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
                        rulesContext)) {

                    for (int[] chunk : CHUNKS) {
                        int chunkX = chunk[0];
                        int chunkZ = chunk[1];
                        int height = OverworldFixture.noiseSettings().height();

                        // The CPU reference: the full synchronous path, beard-free.
                        EmittedChunk expected = filler.blockIds(chunkX, chunkZ);

                        // The GPU path: the whole decision on the device, no CPU supply.
                        GpuChunkFiller.DensityBuffers buffers = filler.emitDensities(chunkX, chunkZ);
                        MaterialDeciderGpu.Decisions actual = decider.decide(chunkX, chunkZ,
                                buffers.density(), buffers.veinToggle(), buffers.veinRidged());
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
