package tqk114514.terracuda.gpu;

import java.util.ArrayList;
import java.util.List;
import net.minecraft.world.level.levelgen.NoiseSettings;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;

/**
 * The per-chunk density tables, computed on the device.
 *
 * <p>This is the shape of M2's K1 and K2. Vanilla's marker system says which sub-graphs get sampled on
 * a coarse grid and interpolated between: {@code flat_cache} and {@code cache_2d} are 2D column
 * caches, {@code interpolated} is the 5×5×49 corner grid whose trilinear interpolation produces the
 * per-block density. Each marker recorded the instruction it wraps, and because the image is emitted
 * in post-order that instruction can be evaluated on its own — so both tables come out of the same
 * kernel with a different root and a different point set.
 *
 * <p>Grid layout, matching the vanilla cell geometry from {@link NoiseSettings}:
 * <ul>
 *   <li>corners: {@code 5 × 5 × (height / cellHeight + 1)} points, indexed
 *       {@code ((cellX * 5) + cellZ) * cornerCountY + cellY}</li>
 *   <li>flat: {@code 5 × 5} points at {@code minY}, indexed {@code (cellX * 5) + cellZ}</li>
 * </ul>
 *
 * <p>Nothing here interpolates: the trilinear step stays on the CPU in the vanilla code path, which is
 * what "GPU for the expensive part, vanilla for the rest" means at this milestone.
 */
public final class ChunkCornerTables {

    /** Cells across a chunk; the main world uses {@code size_horizontal = 1} → four 4-block cells. */
    public static final int CELLS_XZ = 4;
    public static final int CORNERS_XZ = CELLS_XZ + 1;

    /**
     * One marker's grid.
     *
     * @param kind          a {@code DensityProgram.MARKER_*} constant
     * @param root          the image instruction the marker wraps
     * @param cornerCountY  y samples per column (1 for the flat grids)
     * @param coordinates   interleaved {@code x, y, z} block positions
     * @param values        one density value per position
     */
    public record MarkerGrid(int kind, int root, int cornerCountY, int[] coordinates, double[] values) {

        public int pointCount() {
            return this.values.length;
        }
    }

    private ChunkCornerTables() {
    }

    public static int cornerCountY(NoiseSettings settings) {
        return settings.height() / settings.getCellHeight() + 1;
    }

    /** The {@code 5 × 5 × cornerCountY} corner grid of one chunk. */
    public static int[] cornerCoordinates(NoiseSettings settings, int chunkX, int chunkZ) {
        int cellWidth = settings.getCellWidth();
        int cellHeight = settings.getCellHeight();
        int countY = cornerCountY(settings);
        int[] coordinates = new int[3 * CORNERS_XZ * CORNERS_XZ * countY];
        int i = 0;
        for (int cellX = 0; cellX < CORNERS_XZ; cellX++) {
            for (int cellZ = 0; cellZ < CORNERS_XZ; cellZ++) {
                for (int cellY = 0; cellY < countY; cellY++) {
                    coordinates[i++] = chunkX * 16 + cellX * cellWidth;
                    coordinates[i++] = settings.minY() + cellY * cellHeight;
                    coordinates[i++] = chunkZ * 16 + cellZ * cellWidth;
                }
            }
        }
        return coordinates;
    }

    /**
     * The {@code 5 × 5} column grid of one chunk. The y value is arbitrary — these markers are 2D — so
     * {@code minY} is used for determinism.
     */
    public static int[] flatCoordinates(NoiseSettings settings, int chunkX, int chunkZ) {
        int cellWidth = settings.getCellWidth();
        int[] coordinates = new int[3 * CORNERS_XZ * CORNERS_XZ];
        int i = 0;
        for (int cellX = 0; cellX < CORNERS_XZ; cellX++) {
            for (int cellZ = 0; cellZ < CORNERS_XZ; cellZ++) {
                coordinates[i++] = chunkX * 16 + cellX * cellWidth;
                coordinates[i++] = settings.minY();
                coordinates[i++] = chunkZ * 16 + cellZ * cellWidth;
            }
        }
        return coordinates;
    }

    /**
     * Evaluates every column cache and every interpolator corner grid for one chunk.
     *
     * <p>Batches are split to the evaluator's capacity, so a single evaluator can serve any chunk size.
     */
    public static List<MarkerGrid> compute(DensityEvaluatorGpu evaluator, NoiseSettings settings,
            int chunkX, int chunkZ) {
        DensityProgramImage image = evaluator.image();
        int[] kinds = image.markerKinds();
        int[] roots = image.markerRoots();
        int[] corners = cornerCoordinates(settings, chunkX, chunkZ);
        int[] flats = flatCoordinates(settings, chunkX, chunkZ);
        int countY = cornerCountY(settings);

        List<MarkerGrid> grids = new ArrayList<>();
        for (int i = 0; i < kinds.length; i++) {
            boolean isCorner = kinds[i] == DensityProgram.MARKER_INTERPOLATED;
            boolean isFlat = kinds[i] == DensityProgram.MARKER_FLAT_CACHE
                    || kinds[i] == DensityProgram.MARKER_CACHE_2D;
            if (!isCorner && !isFlat) {
                continue;
            }
            int[] coordinates = isCorner ? corners : flats;
            double[] values = evaluateInBatches(evaluator, coordinates, roots[i]);
            grids.add(new MarkerGrid(kinds[i], roots[i], isCorner ? countY : 1, coordinates, values));
        }
        return grids;
    }

    private static double[] evaluateInBatches(DensityEvaluatorGpu evaluator, int[] coordinates,
            int root) {
        int points = coordinates.length / 3;
        double[] values = new double[points];
        int batch = evaluator.capacity();
        for (int start = 0; start < points; start += batch) {
            int size = Math.min(batch, points - start);
            int[] slice = new int[3 * size];
            System.arraycopy(coordinates, 3 * start, slice, 0, 3 * size);
            double[] partial = evaluator.evaluate(slice, root);
            System.arraycopy(partial, 0, values, start, size);
        }
        return values;
    }
}
