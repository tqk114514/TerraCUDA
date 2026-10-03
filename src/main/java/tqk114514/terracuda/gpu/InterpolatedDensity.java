package tqk114514.terracuda.gpu;

import java.util.List;
import net.minecraft.world.level.levelgen.NoiseSettings;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;

/**
 * The per-block density of one chunk: the device's corner tables blended up to block resolution, with
 * the rest of the DAG evaluated on top.
 *
 * <p>This is K3's density half, and it is the shape the GPU path wants: the expensive sub-graphs are
 * sampled on the coarse grid by the device, and everything above them is cheap enough to run wherever
 * it is needed. Nothing here evaluates an interpolated marker at the block — doing that would give the
 * sharper, wrong answer the design doc warns about.
 */
public final class InterpolatedDensity {

    private InterpolatedDensity() {
    }

    /** Per-block density, indexed {@code (x * 16 + z) * height + (y - minY)}. */
    public static double[] forChunk(DensityEvaluatorGpu evaluator, DensityInterpreter interpreter,
            NoiseSettings settings, int chunkX, int chunkZ) {
        return forChunk(evaluator, interpreter, settings, chunkX, chunkZ, CornerInterpolation.DEFAULT);
    }

    /**
     * @param order {@link CornerInterpolation#LERP3} or {@link CornerInterpolation#INCREMENTAL}; only
     *              {@code LERP3} matches vanilla, and the other exists so a test can prove it
     */
    public static double[] forChunk(DensityEvaluatorGpu evaluator, DensityInterpreter interpreter,
            NoiseSettings settings, int chunkX, int chunkZ, int order) {
        DensityProgram program = interpreter.program();
        int[] imageToOriginal = evaluator.image().imageToOriginal();
        List<ChunkCornerTables.MarkerGrid> interpolators = ChunkCornerTables
                .compute(evaluator, settings, chunkX, chunkZ).stream()
                .filter(grid -> grid.kind() == DensityProgram.MARKER_INTERPOLATED)
                .toList();

        int cellWidth = settings.getCellWidth();
        int cellHeight = settings.getCellHeight();
        int cornerCountY = ChunkCornerTables.cornerCountY(settings);
        int minY = settings.minY();
        int height = settings.height();
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;

        double[] densities = new double[16 * 16 * height];

        // The overridden instructions are the same for every block in the chunk, so the indices are
        // computed once and only the values change per block. Passing arrays rather than a map is what
        // keeps this from allocating 98304 maps.
        int[] overrideIndices = new int[interpolators.size()];
        for (int i = 0; i < interpolators.size(); i++) {
            // The corner tables are indexed in image numbering; the interpreter works in program
            // numbering.
            overrideIndices[i] = imageToOriginal[interpolators.get(i).root()];
        }
        double[] overrideValues = new double[interpolators.size()];

        for (int yLocal = 0; yLocal < height; yLocal++) {
            int y = minY + yLocal;
            int cellY = yLocal / cellHeight;
            int yInCell = yLocal % cellHeight;
            for (int xLocal = 0; xLocal < 16; xLocal++) {
                int cellX = xLocal / cellWidth;
                int xInCell = xLocal % cellWidth;
                for (int zLocal = 0; zLocal < 16; zLocal++) {
                    int cellZ = zLocal / cellWidth;
                    int zInCell = zLocal % cellWidth;

                    for (int i = 0; i < interpolators.size(); i++) {
                        ChunkCornerTables.MarkerGrid grid = interpolators.get(i);
                        overrideValues[i] = CornerInterpolation.interpolate(grid.values(), cornerCountY,
                                order, cellX, cellZ, cellY, xInCell, yInCell, zInCell, cellWidth, cellHeight);
                    }

                    densities[(xLocal * 16 + zLocal) * height + yLocal] = interpreter.evaluate(
                            chunkMinBlockX + xLocal, y, chunkMinBlockZ + zLocal, overrideIndices,
                            overrideValues);
                }
            }
        }
        return densities;
    }
}
