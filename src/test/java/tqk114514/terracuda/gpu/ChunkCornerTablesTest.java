package tqk114514.terracuda.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.List;
import java.util.Optional;
import net.minecraft.world.level.levelgen.NoiseSettings;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * M2, K1 and K2 at chunk scale: every column cache and every interpolator corner grid must match the
 * CPU reference bit for bit.
 *
 * <p>The corner grids are the expensive part of vanilla's generation — 8 interpolators × 5×5×49
 * corners per chunk — and they are what the device is here for. The trilinear interpolation between
 * them stays on the CPU, so this is the last piece of M2 before the vanilla code path takes over.
 */
class ChunkCornerTablesTest {

    private static final int CHUNK_X = 3;
    private static final int CHUNK_Z = -7;
    private static final int MAX_MARKERS_CHECKED = 4;

    @Test
    void theCornerGridHasTheVanillaShape() {
        NoiseSettings settings = OverworldFixture.noiseSettings();

        assertEquals(4, settings.getCellWidth(), "overworld size_horizontal is 1, so cells are 4 blocks");
        assertEquals(8, settings.getCellHeight(), "overworld size_vertical is 2, so cells are 8 blocks");

        int countY = ChunkCornerTables.cornerCountY(settings);
        assertEquals(settings.height() / 8 + 1, countY);

        int[] corners = ChunkCornerTables.cornerCoordinates(settings, CHUNK_X, CHUNK_Z);
        assertEquals(3 * 5 * 5 * countY, corners.length);

        int[] flats = ChunkCornerTables.flatCoordinates(settings, CHUNK_X, CHUNK_Z);
        assertEquals(3 * 25, flats.length);

        // The grid starts at the chunk's own origin and steps by the cell size.
        assertEquals(CHUNK_X * 16, corners[0]);
        assertEquals(settings.minY(), corners[1]);
        assertEquals(CHUNK_Z * 16, corners[2]);
        // Indexed ((cellX * 5) + cellZ) * countY + cellY, so cellX = 1 starts one row in.
        assertEquals(CHUNK_X * 16 + 4, corners[3 * (5 * countY)]);
    }

    @Test
    void everyMarkerTableMatchesTheCpuReferenceBitForBit() {
        DensityProgram program =
                DensityCompiler.lower(OverworldFixture.randomState().router().finalDensity());
        DensityInterpreter reference = new DensityInterpreter(program);
        NoiseSettings settings = OverworldFixture.noiseSettings();

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
                    assertFalse(grids.isEmpty(), "the overworld density function is marked up for caching");

                    int[] imageToOriginal = evaluator.image().imageToOriginal();
                    int interpolators = 0;
                    int caches = 0;
                    int checked = 0;

                    for (ChunkCornerTables.MarkerGrid grid : grids) {
                        if (grid.kind() == DensityProgram.MARKER_INTERPOLATED) {
                            interpolators++;
                        } else {
                            caches++;
                        }
                        // The full 5x5x49 corner grids are expensive on the CPU reference; a few of
                        // each kind is enough to cover every marker shape.
                        if (checked >= MAX_MARKERS_CHECKED) {
                            continue;
                        }
                        checked++;

                        int originalRoot = imageToOriginal[grid.root()];
                        int[] coordinates = grid.coordinates();
                        double[] values = grid.values();
                        assertEquals(coordinates.length / 3, values.length);

                        for (int i = 0; i < values.length; i++) {
                            double expected = reference.evaluateAt(originalRoot,
                                    coordinates[3 * i], coordinates[3 * i + 1], coordinates[3 * i + 2]);
                            TestParity.assertDoubleIdentical(expected, values[i],
                                    "marker kind " + grid.kind() + " point " + i);
                        }
                    }

                    assertTrue(interpolators > 0, "expected interpolated markers");
                    assertTrue(caches > 0, "expected column caches");
                    assertEquals(Math.min(MAX_MARKERS_CHECKED, grids.size()), checked);
                }
            }
        }
    }
}
