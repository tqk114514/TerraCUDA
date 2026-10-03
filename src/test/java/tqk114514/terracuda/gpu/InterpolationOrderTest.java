package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.worldgen.OverworldFixture;
import tqk114514.terracuda.worldgen.VanillaChunkReference;

/**
 * Which trilinear blend order does the material rule actually see?
 *
 * <p>Vanilla has two. {@code NoiseInterpolator.updateForY/X/Z} blends Y, then X, then Z;
 * {@code NoiseInterpolator.compute} takes the {@code Mth.lerp3} path, which blends X, then Y, then Z.
 * They differ in the last bits, and {@code final_density} is a {@code cache_all_in_cell} marker — so
 * the answer depends on whether the cache's fill or the interpolator's own {@code compute} is what
 * feeds the material rules.
 *
 * <p>Reading the source suggests the {@code lerp3} path, because {@code CacheAllInCell.fillAllDirectly}
 * sets {@code fillingCell}. This test does not rely on that reading: it turns the aquifer and the ore
 * veinifier off, which makes vanilla's own block state a direct read-out of the sign of the
 * interpolated density — stone where it is positive, a fluid where it is not — and then checks both
 * orders against 98304 real blocks.
 */
class InterpolationOrderTest {

    private static final int CHUNK_X = 3;
    private static final int CHUNK_Z = -7;

    @Test
    void theMaterialRulesSeeTheLerp3Blend() {
        NoiseGeneratorSettings plain = withoutAquifersAndVeins(OverworldFixture.generatorSettings());
        RandomState randomState = OverworldFixture.randomState();

        VanillaChunkReference.ChunkBlocks vanilla =
                VanillaChunkReference.generate(plain, randomState, CHUNK_X, CHUNK_Z);
        int stone = Block.getId(plain.defaultBlock());

        // Sanity: with the rules off, a block is either the default block or a fluid.
        for (int id : vanilla.stateIds()) {
            assertTrue(id == stone
                            || id == Block.getId(Blocks.WATER.defaultBlockState())
                            || id == Block.getId(Blocks.LAVA.defaultBlockState())
                            || id == Block.getId(Blocks.AIR.defaultBlockState()),
                    "unexpected block id " + id + " with the material rules disabled");
        }

        DensityProgram program = DensityCompiler.lower(randomState.router().finalDensity());
        DensityInterpreter interpreter = new DensityInterpreter(program);
        NoiseSettings settings = plain.noiseSettings();

        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        assumeTrue(loaded.isPresent(), "no CUDA driver library on this machine");

        try (CudaDriver driver = loaded.get()) {
            driver.init();
            CudaDeviceInfo device = environment.firstDevice().orElseThrow();
            try (CudaContext context = CudaContext.create(driver, device.index())) {
                CudaKernels.loadModule(context, device);
                try (DensityEvaluatorGpu evaluator = new DensityEvaluatorGpu(context, program, 2048)) {
                    List<ChunkCornerTables.MarkerGrid> grids =
                            ChunkCornerTables.compute(evaluator, settings, CHUNK_X, CHUNK_Z);
                    List<ChunkCornerTables.MarkerGrid> interpolators = grids.stream()
                            .filter(g -> g.kind() == DensityProgram.MARKER_INTERPOLATED)
                            .toList();
                    assertTrue(interpolators.size() > 0, "expected interpolated markers");

                    int incremental = countMismatches(vanilla, stone, interpreter, settings,
                            interpolators, evaluator.image().imageToOriginal(),
                            CornerInterpolation.INCREMENTAL);
                    int lerp3 = countMismatches(vanilla, stone, interpreter, settings,
                            interpolators, evaluator.image().imageToOriginal(),
                            CornerInterpolation.LERP3);

                    System.out.println("ORDERDIAG incremental=" + incremental + " lerp3=" + lerp3
                            + " of " + vanilla.size());

                    // Exactly one of the two orders reproduces the number the material rules saw, bit
                    // for bit. Anything else means something upstream is wrong.
                    assertEquals(0, lerp3, "the lerp3 order should reproduce vanilla's density exactly");
                    assertTrue(incremental > 0,
                            "if the incremental order also matched, this test would not be deciding anything");
                }
            }
        }
    }

    private static int countMismatches(VanillaChunkReference.ChunkBlocks vanilla, int stone,
            DensityInterpreter interpreter, NoiseSettings settings,
            List<ChunkCornerTables.MarkerGrid> interpolators, int[] imageToOriginal, int order) {
        int cellWidth = settings.getCellWidth();
        int cellHeight = settings.getCellHeight();
        int cornerCountY = ChunkCornerTables.cornerCountY(settings);
        int minY = settings.minY();
        int mismatches = 0;
        Map<Integer, Double> overrides = new HashMap<>();

        for (int y = minY; y < minY + settings.height(); y++) {
            int yLocal = y - minY;
            int cellY = yLocal / cellHeight;
            int yInCell = yLocal % cellHeight;
            for (int xLocal = 0; xLocal < 16; xLocal++) {
                int cellX = xLocal / cellWidth;
                int xInCell = xLocal % cellWidth;
                for (int zLocal = 0; zLocal < 16; zLocal++) {
                    int cellZ = zLocal / cellWidth;
                    int zInCell = zLocal % cellWidth;

                    overrides.clear();
                    for (ChunkCornerTables.MarkerGrid grid : interpolators) {
                        // The corner tables are indexed in image numbering; the interpreter works in
                        // program numbering.
                        overrides.put(imageToOriginal[grid.root()], CornerInterpolation.interpolate(
                                grid.values(), cornerCountY, order, cellX, cellZ, cellY, xInCell, yInCell,
                                zInCell, cellWidth, cellHeight));
                    }

                    double density = interpreter.evaluate(CHUNK_X * 16 + xLocal, y, CHUNK_Z * 16 + zLocal,
                            overrides);
                    if (Double.doubleToRawLongBits(density)
                            != Double.doubleToRawLongBits(vanilla.densityAt(xLocal, y, zLocal))) {
                        mismatches++;
                    }
                }
            }
        }
        return mismatches;
    }

    private static NoiseGeneratorSettings withoutAquifersAndVeins(NoiseGeneratorSettings settings) {
        return new NoiseGeneratorSettings(settings.noiseSettings(), settings.defaultBlock(),
                settings.defaultFluid(), settings.noiseRouter(), settings.surfaceRule(),
                settings.spawnTarget(), settings.seaLevel(), settings.disableMobGeneration(),
                false, false, settings.useLegacyRandomSource());
    }
}
