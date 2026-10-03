package tqk114514.terracuda.density;

import tqk114514.terracuda.math.VanillaMath;

/**
 * Turns a marker's corner grid into a per-block value, the way vanilla's {@code NoiseInterpolator}
 * does.
 *
 * <p>The corner grid comes from K2: {@code 5 × 5 × (cellCountY + 1)} samples per interpolated marker,
 * indexed {@code ((cellX * 5) + cellZ) * cornerCountY + cellY}. Between the corners the value is a
 * trilinear blend, and the <em>order</em> of that blend is observable in the last bits.
 *
 * <p>Two orders appear in vanilla, and they are not the same function:
 * <ul>
 *   <li>{@code NoiseInterpolator.updateForY/X/Z} blends <b>Y, then X, then Z</b>, and this is what the
 *       interpolator returns when {@code fillingCell} is false.</li>
 *   <li>{@code NoiseInterpolator.compute} takes the {@code Mth.lerp3} path, which blends <b>X, then
 *       Y, then Z</b>, when {@code fillingCell} is true — i.e. while a {@code cache_all_in_cell}
 *       marker is filling its 128 values.</li>
 * </ul>
 *
 * <p>Which one the material rules see was settled by measurement, not by reading: {@code final_density}
 * is a {@code cache_all_in_cell}, so its fill is what feeds them. Against 98304 blocks of a real
 * chunk, {@link #LERP3} reproduces vanilla's density bit for bit and {@link #INCREMENTAL} differs on
 * 5928 of them. {@link #DEFAULT} is therefore {@code LERP3}; {@code InterpolationOrderTest} pins the
 * finding down.
 */
public final class CornerInterpolation {

    /** Corners along x and z: four cells per chunk plus one. */
    public static final int CORNERS_XZ = 5;

    /** The blend order {@code updateForY/X/Z} produces. */
    public static final int INCREMENTAL = 0;

    /** The blend order {@code Mth.lerp3} produces. */
    public static final int LERP3 = 1;

    /** What the material rules actually consume. */
    public static final int DEFAULT = LERP3;

    private CornerInterpolation() {
    }

    /**
     * Blends one block out of its cell's eight corners.
     *
     * @param corners      the marker's corner values, indexed {@code ((cellX * 5) + cellZ) * countY + cellY}
     * @param cornerCountY samples per column, {@code cellCountY + 1}
     * @param order        {@link #INCREMENTAL} or {@link #LERP3}
     */
    public static double interpolate(double[] corners, int cornerCountY, int order,
            int cellX, int cellZ, int cellY,
            int xInCell, int yInCell, int zInCell, int cellWidth, int cellHeight) {
        double n000 = at(corners, cornerCountY, cellX, cellZ, cellY);
        double n001 = at(corners, cornerCountY, cellX, cellZ + 1, cellY);
        double n100 = at(corners, cornerCountY, cellX + 1, cellZ, cellY);
        double n101 = at(corners, cornerCountY, cellX + 1, cellZ + 1, cellY);
        double n010 = at(corners, cornerCountY, cellX, cellZ, cellY + 1);
        double n011 = at(corners, cornerCountY, cellX, cellZ + 1, cellY + 1);
        double n110 = at(corners, cornerCountY, cellX + 1, cellZ, cellY + 1);
        double n111 = at(corners, cornerCountY, cellX + 1, cellZ + 1, cellY + 1);

        double factorX = (double) xInCell / cellWidth;
        double factorY = (double) yInCell / cellHeight;
        double factorZ = (double) zInCell / cellWidth;

        if (order == LERP3) {
            return VanillaMath.lerp3(factorX, factorY, factorZ,
                    n000, n100, n010, n110, n001, n101, n011, n111);
        }

        double valueXZ00 = VanillaMath.lerp(factorY, n000, n010);
        double valueXZ10 = VanillaMath.lerp(factorY, n100, n110);
        double valueXZ01 = VanillaMath.lerp(factorY, n001, n011);
        double valueXZ11 = VanillaMath.lerp(factorY, n101, n111);
        double valueZ0 = VanillaMath.lerp(factorX, valueXZ00, valueXZ10);
        double valueZ1 = VanillaMath.lerp(factorX, valueXZ01, valueXZ11);
        return VanillaMath.lerp(factorZ, valueZ0, valueZ1);
    }

    private static double at(double[] corners, int cornerCountY, int cellX, int cellZ, int cellY) {
        return corners[(cellX * CORNERS_XZ + cellZ) * cornerCountY + cellY];
    }
}
