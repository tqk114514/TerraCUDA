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
     * The number of points a chunk's marker grids need in total.
     *
     * <p>Callers size the evaluator from this, because the grids are evaluated in one launch: a launch
     * costs about the same whether it carries one point or twenty-five, so the marker grids have to
     * travel together to be worth anything.
     */
    public static int pointBudget(DensityProgramImage image, NoiseSettings settings) {
        int countY = cornerCountY(settings);
        int total = 0;
        for (int kind : image.markerKinds()) {
            if (kind == DensityProgram.MARKER_INTERPOLATED) {
                total += CORNERS_XZ * CORNERS_XZ * countY;
            } else if (kind == DensityProgram.MARKER_FLAT_CACHE || kind == DensityProgram.MARKER_CACHE_2D) {
                total += CORNERS_XZ * CORNERS_XZ;
            }
        }
        return total;
    }

    /**
     * Evaluates every column cache and every interpolator corner grid for one chunk.
     *
     * <p>The grids are concatenated into a single launch, up to the evaluator's capacity. That is not
     * a micro-optimisation: a launch of 1225 points against a 450-step serial chain leaves the device
     * mostly idle, and the measured cost is flat at roughly 750 µs whether the launch carries one
     * point or twenty-five. Eighteen separate launches therefore cost eighteen times the ramp.
     */
    public static List<MarkerGrid> compute(DensityEvaluatorGpu evaluator, NoiseSettings settings,
            int chunkX, int chunkZ) {
        DensityProgramImage image = evaluator.image();
        int[] kinds = image.markerKinds();
        int[] roots = image.markerRoots();
        int[] corners = cornerCoordinates(settings, chunkX, chunkZ);
        int[] flats = flatCoordinates(settings, chunkX, chunkZ);
        int countY = cornerCountY(settings);

        List<Integer> markers = new ArrayList<>();
        List<int[]> coordinates = new ArrayList<>();
        for (int i = 0; i < kinds.length; i++) {
            boolean isCorner = kinds[i] == DensityProgram.MARKER_INTERPOLATED;
            boolean isFlat = kinds[i] == DensityProgram.MARKER_FLAT_CACHE
                    || kinds[i] == DensityProgram.MARKER_CACHE_2D;
            if (isCorner || isFlat) {
                markers.add(i);
                coordinates.add(isCorner ? corners : flats);
            }
        }

        List<MarkerGrid> grids = new ArrayList<>();
        int capacityPoints = evaluator.capacity();
        int at = 0;
        while (at < markers.size()) {
            int points = 0;
            int end = at;
            while (end < markers.size()
                    && points + coordinates.get(end).length / 3 <= capacityPoints) {
                points += coordinates.get(end).length / 3;
                end++;
            }
            if (end == at) {
                throw new IllegalArgumentException("one marker grid needs " + coordinates.get(at).length / 3
                        + " points, more than the evaluator's capacity of " + capacityPoints);
            }

            int[] combined = new int[3 * points];
            int[] combinedRoots = new int[points];
            int offset = 0;
            for (int i = at; i < end; i++) {
                int[] grid = coordinates.get(i);
                System.arraycopy(grid, 0, combined, 3 * offset, grid.length);
                int gridPoints = grid.length / 3;
                java.util.Arrays.fill(combinedRoots, offset, offset + gridPoints, roots[markers.get(i)]);
                offset += gridPoints;
            }

            double[] values = evaluator.evaluate(combined, combinedRoots);

            int cursor = 0;
            for (int i = at; i < end; i++) {
                int[] grid = coordinates.get(i);
                int gridPoints = grid.length / 3;
                double[] slice = new double[gridPoints];
                System.arraycopy(values, cursor, slice, 0, gridPoints);
                int index = markers.get(i);
                grids.add(new MarkerGrid(kinds[index], roots[index],
                        kinds[index] == DensityProgram.MARKER_INTERPOLATED ? countY : 1, grid, slice));
                cursor += gridPoints;
            }
            at = end;
        }
        return grids;
    }
}
