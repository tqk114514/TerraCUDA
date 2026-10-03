package tqk114514.terracuda.density;

import tqk114514.terracuda.noise.ImprovedNoise;
import tqk114514.terracuda.noise.NormalNoise;
import tqk114514.terracuda.worldgen.NoiseSnapshot;

/**
 * A density function lowered to a flat instruction DAG.
 *
 * <p>The vanilla density function is a tree of roughly thirty node types, most of them package-private
 * records. This form is what the GPU can actually consume: parallel primitive arrays plus small
 * side tables for the things that do not fit in a scalar (noise octaves, cubic splines, blended
 * noise). It is also what {@link DensityInterpreter} walks, so the CPU reference and the kernel
 * evaluate exactly the same program — which is the point.
 *
 * <p>Encoding, per instruction:
 * <ul>
 *   <li>{@code op} — the opcode</li>
 *   <li>{@code ia..id} — int operands: child instruction indices, or indices into the side tables</li>
 *   <li>{@code da, db} — double immediates</li>
 * </ul>
 *
 * <p>Markers ({@code interpolated}, {@code flat_cache}, …) are <em>not</em> instructions. They are
 * pass-through for a sequential evaluation, and they are kernel boundaries rather than operations for
 * the GPU, so they are recorded separately in {@link #markerKinds()} / {@link #markerRoots()}.
 */
public final class DensityProgram {

    public static final int CONSTANT = 1;
    public static final int Y_CLAMPED_GRADIENT = 2;
    public static final int NOISE = 3;
    public static final int SHIFTED_NOISE = 4;
    public static final int SHIFT = 5;
    public static final int SHIFT_A = 6;
    public static final int SHIFT_B = 7;
    public static final int WEIRD_SCALED_TYPE1 = 8;
    public static final int WEIRD_SCALED_TYPE2 = 9;
    public static final int SPLINE = 10;
    public static final int RANGE_CHOICE = 11;
    public static final int BLEND_DENSITY = 12;
    public static final int BLEND_ALPHA = 13;
    public static final int BLEND_OFFSET = 14;
    public static final int BEARDIFIER = 15;
    public static final int BLENDED_NOISE = 16;
    public static final int ADD = 17;
    public static final int MUL = 18;
    public static final int MIN = 19;
    public static final int MAX = 20;
    public static final int MUL_ADD = 21;
    public static final int ABS = 22;
    public static final int SQUARE = 23;
    public static final int CUBE = 24;
    public static final int HALF_NEGATIVE = 25;
    public static final int QUARTER_NEGATIVE = 26;
    public static final int INVERT = 27;
    public static final int SQUEEZE = 28;
    public static final int CLAMP = 29;
    public static final int FIND_TOP_SURFACE = 30;

    /**
     * A value supplied from outside the program.
     *
     * <p>Only produced by {@code DensityProgramImage.ofBlocks}, which lowers a program with its
     * interpolated markers cut out: those markers become leaves of this kind, and the caller fills
     * them in — with the trilinear blend of the marker's corner grid — before evaluating. The rest of
     * the DAG above them is unchanged, so the chunk-level pass can run on the device without the
     * marker subgraphs, which is the whole point.
     */
    public static final int OVERRIDE = 31;

    public static final int MARKER_INTERPOLATED = 1;
    public static final int MARKER_FLAT_CACHE = 2;
    public static final int MARKER_CACHE_2D = 3;
    public static final int MARKER_CACHE_ONCE = 4;
    public static final int MARKER_CACHE_ALL_IN_CELL = 5;

    /** A {@code NormalNoise}, both as its raw snapshot and as this project's reference implementation. */
    public record NoiseEntry(NoiseSnapshot snapshot, NormalNoise reference) {
    }

    /**
     * One cubic spline node. {@code kind} is 0 for a constant, 1 for a multipoint; a multipoint's
     * values are themselves spline nodes, referenced by {@link #children()}.
     */
    public record SplineNode(int kind, float constant, int coordinate, float[] locations,
            float[] derivatives, int[] children) {

        public static final int CONSTANT = 0;
        public static final int MULTIPOINT = 1;

        public SplineNode {
            locations = locations.clone();
            derivatives = derivatives.clone();
            children = children.clone();
        }

        @Override
        public float[] locations() {
            return this.locations.clone();
        }

        @Override
        public float[] derivatives() {
            return this.derivatives.clone();
        }

        @Override
        public int[] children() {
            return this.children.clone();
        }
    }

    /** The raw octave levels of a {@code BlendedNoise}, in {@code noiseLevels} order. */
    public record BlendedEntry(NoiseSnapshot.BlendedNoiseData data) {
    }

    private final int[] op;
    private final int[] ia;
    private final int[] ib;
    private final int[] ic;
    private final int[] id;
    private final double[] da;
    private final double[] db;
    private final int root;
    private final NoiseEntry[] noises;
    private final SplineNode[] splines;
    private final BlendedEntry[] blended;
    private final int[] markerKinds;
    private final int[] markerRoots;

    DensityProgram(int[] op, int[] ia, int[] ib, int[] ic, int[] id, double[] da, double[] db, int root,
            NoiseEntry[] noises, SplineNode[] splines, BlendedEntry[] blended,
            int[] markerKinds, int[] markerRoots) {
        this.op = op;
        this.ia = ia;
        this.ib = ib;
        this.ic = ic;
        this.id = id;
        this.da = da;
        this.db = db;
        this.root = root;
        this.noises = noises;
        this.splines = splines;
        this.blended = blended;
        this.markerKinds = markerKinds;
        this.markerRoots = markerRoots;
    }

    /** Number of instructions. */
    public int size() {
        return this.op.length;
    }

    /** Index of the instruction to evaluate. */
    public int root() {
        return this.root;
    }

    public int op(int pc) {
        return this.op[pc];
    }

    public int ia(int pc) {
        return this.ia[pc];
    }

    public int ib(int pc) {
        return this.ib[pc];
    }

    public int ic(int pc) {
        return this.ic[pc];
    }

    public int id(int pc) {
        return this.id[pc];
    }

    public double da(int pc) {
        return this.da[pc];
    }

    public double db(int pc) {
        return this.db[pc];
    }

    public NoiseEntry noise(int index) {
        return this.noises[index];
    }

    public int noiseCount() {
        return this.noises.length;
    }

    public SplineNode spline(int index) {
        return this.splines[index];
    }

    public int splineCount() {
        return this.splines.length;
    }

    public BlendedEntry blended(int index) {
        return this.blended[index];
    }

    public int blendedCount() {
        return this.blended.length;
    }

    /** Marker kinds, parallel with {@link #markerRoots()}. */
    public int[] markerKinds() {
        return this.markerKinds.clone();
    }

    /** The instruction index each marker wraps, parallel with {@link #markerKinds()}. */
    public int[] markerRoots() {
        return this.markerRoots.clone();
    }

    public int markerCount() {
        return this.markerKinds.length;
    }

    /** A one-line description for logs. */
    public String summary() {
        return "DensityProgram[" + this.op.length + " instructions, " + this.noises.length + " noises, "
                + this.splines.length + " splines, " + this.blended.length + " blended, "
                + this.markerKinds.length + " markers]";
    }
}
