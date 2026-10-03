package tqk114514.terracuda.density;

import tqk114514.terracuda.noise.ImprovedNoise;
import tqk114514.terracuda.worldgen.NoiseSnapshot;

/**
 * A {@link DensityProgram} flattened into the byte layout the CUDA interpreter reads.
 *
 * <p>Everything the kernel needs lives in one read-only blob, addressed through an offset table. That
 * keeps the kernel signature down to a handful of arguments and removes the chance of an argument
 * order slipping out of sync with the device side.
 *
 * <p>Every table is 8-byte aligned, because the kernel reinterprets the blob as {@code int},
 * {@code float}, {@code double} and {@code long long} pointers.
 *
 * <p>Offsets are indexed by the {@code TC_OFF_*} constants in {@code terracuda.cu}; the two must stay
 * in step, which is why they are spelled out here as well rather than being inferred.
 */
public final class DensityProgramImage {

    public static final int OFF_OPS = 0;
    public static final int OFF_IA = 1;
    public static final int OFF_IB = 2;
    public static final int OFF_IC = 3;
    public static final int OFF_ID = 4;
    public static final int OFF_DA = 5;
    public static final int OFF_DB = 6;
    public static final int OFF_NOISE_FIRST_OCTAVE = 7;
    public static final int OFF_NOISE_OCTAVE_START = 8;
    public static final int OFF_NOISE_OCTAVE_COUNT = 9;
    public static final int OFF_NOISE_VALUE_FACTOR = 10;
    public static final int OFF_NOISE_AMP_START = 11;
    public static final int OFF_OCTAVE_ORIGINS = 12;
    public static final int OFF_OCTAVE_PERMUTATIONS = 13;
    public static final int OFF_OCTAVE_VALID = 14;
    public static final int OFF_AMPLITUDES = 15;
    public static final int OFF_SPLINE_KIND = 16;
    public static final int OFF_SPLINE_CONSTANT = 17;
    public static final int OFF_SPLINE_COORDINATE = 18;
    public static final int OFF_SPLINE_LOC_START = 19;
    public static final int OFF_SPLINE_LOC_COUNT = 20;
    public static final int OFF_SPLINE_CHILD_START = 21;
    public static final int OFF_SPLINE_LOCATIONS = 22;
    public static final int OFF_SPLINE_DERIVATIVES = 23;
    public static final int OFF_SPLINE_CHILDREN = 24;
    public static final int OFF_BLENDED_OCTAVE_START = 25;
    public static final int OFF_BLENDED_XZ_MULT = 26;
    public static final int OFF_BLENDED_Y_MULT = 27;
    public static final int OFF_BLENDED_XZ_FACTOR = 28;
    public static final int OFF_BLENDED_Y_FACTOR = 29;
    public static final int OFF_BLENDED_SMEAR = 30;
    public static final int OFF_BLENDED_MIN_COUNT = 31;
    public static final int OFF_BLENDED_MAIN_COUNT = 32;
    public static final int OFFSET_COUNT = 33;

    private static final int PERMUTATION_BYTES = 256;

    private final byte[] blob;
    private final long[] offsets;
    private final int instructionCount;
    private final int root;
    private final int octaveCount;

    private DensityProgramImage(byte[] blob, long[] offsets, int instructionCount, int root,
            int octaveCount) {
        this.blob = blob;
        this.offsets = offsets;
        this.instructionCount = instructionCount;
        this.root = root;
        this.octaveCount = octaveCount;
    }

    /**
     * Flattens {@code program}.
     *
     * @throws DensityCompiler.UnsupportedDensityFunctionException when the program contains a node the
     *         kernel cannot evaluate — {@code find_top_surface} re-enters the DAG at a different y,
     *         which the flat single-pass evaluation cannot express. That node only appears in
     *         {@code preliminary_surface_level}, never in {@code final_density}.
     */
    public static DensityProgramImage of(DensityProgram program) {
        int size = program.size();
        int[] rawOp = new int[size];
        int[] rawIa = new int[size];
        int[] rawIb = new int[size];
        int[] rawIc = new int[size];
        int[] rawId = new int[size];
        double[] rawDa = new double[size];
        double[] rawDb = new double[size];
        for (int pc = 0; pc < size; pc++) {
            rawOp[pc] = program.op(pc);
            if (rawOp[pc] == DensityProgram.FIND_TOP_SURFACE) {
                throw new DensityCompiler.UnsupportedDensityFunctionException(
                        "the GPU interpreter cannot evaluate find_top_surface at instruction " + pc);
            }
            rawIa[pc] = program.ia(pc);
            rawIb[pc] = program.ib(pc);
            rawIc[pc] = program.ic(pc);
            rawId[pc] = program.id(pc);
            rawDa[pc] = program.da(pc);
            rawDb[pc] = program.db(pc);
        }

        // The compiler's indices do not guarantee that a child is emitted before its parent: a
        // subgraph shared by two parents keeps the index it got the first time. Re-emit the reachable
        // graph in post-order so that every child precedes its parent, which is what lets the kernel
        // evaluate the whole stream in one forward pass.
        int[] order = postOrder(program, rawOp, rawIa, rawIb, rawIc);
        int[] newIndex = new int[size];
        java.util.Arrays.fill(newIndex, -1);
        for (int i = 0; i < order.length; i++) {
            newIndex[order[i]] = i;
        }

        int count = order.length;
        int[] ops = new int[count];
        int[] ia = new int[count];
        int[] ib = new int[count];
        int[] ic = new int[count];
        int[] id = new int[count];
        double[] da = new double[count];
        double[] db = new double[count];
        for (int i = 0; i < count; i++) {
            int pc = order[i];
            ops[i] = rawOp[pc];
            ia[i] = remap(rawIa[pc], ops[i], 0, newIndex);
            ib[i] = remap(rawIb[pc], ops[i], 1, newIndex);
            ic[i] = remap(rawIc[pc], ops[i], 2, newIndex);
            id[i] = rawId[pc];
            da[i] = rawDa[pc];
            db[i] = rawDb[pc];
        }
        int root = newIndex[program.root()];

        // Noise octaves and blended octaves share one global octave table.
        BlobWriter writer = new BlobWriter();
        long[] offsets = new long[OFFSET_COUNT];
        OctaveCollector octaves = new OctaveCollector();

        int noiseCount = program.noiseCount();
        int[] noiseFirstOctave = new int[noiseCount];
        int[] noiseOctaveStart = new int[noiseCount];
        int[] noiseOctaveCount = new int[noiseCount];
        double[] noiseValueFactor = new double[noiseCount];
        int[] noiseAmplitudeStart = new int[noiseCount];

        for (int n = 0; n < noiseCount; n++) {
            DensityProgram.NoiseEntry entry = program.noise(n);
            NoiseSnapshot snapshot = entry.snapshot();
            double[] amplitudes = snapshot.parameters().amplitudes();

            noiseFirstOctave[n] = snapshot.parameters().firstOctave();
            noiseOctaveCount[n] = snapshot.octaveCount();
            noiseValueFactor[n] = valueFactor(amplitudes);
            noiseAmplitudeStart[n] = octaves.addAmplitudes(amplitudes);
            noiseOctaveStart[n] = octaves.add(snapshot.firstOctaves());
            octaves.add(snapshot.secondOctaves());
        }

        int blendedCount = program.blendedCount();
        int[] blendedOctaveStart = new int[blendedCount];
        int[] blendedMinCount = new int[blendedCount];
        int[] blendedMainCount = new int[blendedCount];
        double[] blendedXzMult = new double[blendedCount];
        double[] blendedYMult = new double[blendedCount];
        double[] blendedXzFactor = new double[blendedCount];
        double[] blendedYFactor = new double[blendedCount];
        double[] blendedSmear = new double[blendedCount];

        for (int b = 0; b < blendedCount; b++) {
            NoiseSnapshot.BlendedNoiseData data = program.blended(b).data();
            blendedXzMult[b] = data.xzMultiplier();
            blendedYMult[b] = data.yMultiplier();
            blendedXzFactor[b] = data.xzFactor();
            blendedYFactor[b] = data.yFactor();
            blendedSmear[b] = data.smearScaleMultiplier();
            blendedMinCount[b] = data.minLimit().length;
            blendedMainCount[b] = data.main().length;
            blendedOctaveStart[b] = octaves.addRaw(data.minLimit());
            octaves.addRaw(data.maxLimit());
            octaves.addRaw(data.main());
        }

        offsets[OFF_OPS] = writer.putInts(ops);
        offsets[OFF_IA] = writer.putInts(ia);
        offsets[OFF_IB] = writer.putInts(ib);
        offsets[OFF_IC] = writer.putInts(ic);
        offsets[OFF_ID] = writer.putInts(id);
        offsets[OFF_DA] = writer.putDoubles(da);
        offsets[OFF_DB] = writer.putDoubles(db);
        offsets[OFF_NOISE_FIRST_OCTAVE] = writer.putInts(noiseFirstOctave);
        offsets[OFF_NOISE_OCTAVE_START] = writer.putInts(noiseOctaveStart);
        offsets[OFF_NOISE_OCTAVE_COUNT] = writer.putInts(noiseOctaveCount);
        offsets[OFF_NOISE_VALUE_FACTOR] = writer.putDoubles(noiseValueFactor);
        offsets[OFF_NOISE_AMP_START] = writer.putInts(noiseAmplitudeStart);
        offsets[OFF_OCTAVE_ORIGINS] = writer.putDoubles(octaves.origins());
        offsets[OFF_OCTAVE_PERMUTATIONS] = writer.putBytes(octaves.permutations());
        offsets[OFF_OCTAVE_VALID] = writer.putInts(octaves.valid());
        offsets[OFF_AMPLITUDES] = writer.putDoubles(octaves.amplitudes());

        int splineCount = program.splineCount();
        int[] splineKind = new int[splineCount];
        float[] splineConstant = new float[splineCount];
        int[] splineCoordinate = new int[splineCount];
        int[] splineLocStart = new int[splineCount];
        int[] splineLocCount = new int[splineCount];
        int[] splineChildStart = new int[splineCount];
        FloatList splineLocations = new FloatList();
        FloatList splineDerivatives = new FloatList();
        IntList splineChildren = new IntList();

        for (int s = 0; s < splineCount; s++) {
            DensityProgram.SplineNode node = program.spline(s);
            splineKind[s] = node.kind();
            splineConstant[s] = node.constant();
            splineCoordinate[s] = node.coordinate() < 0 ? -1 : newIndex[node.coordinate()];
            splineLocStart[s] = splineLocations.size();
            splineLocCount[s] = node.locations().length;
            splineLocations.add(node.locations());
            splineDerivatives.add(node.derivatives());
            splineChildStart[s] = splineChildren.size();
            splineChildren.add(node.children());
        }

        offsets[OFF_SPLINE_KIND] = writer.putInts(splineKind);
        offsets[OFF_SPLINE_CONSTANT] = writer.putFloats(splineConstant);
        offsets[OFF_SPLINE_COORDINATE] = writer.putInts(splineCoordinate);
        offsets[OFF_SPLINE_LOC_START] = writer.putInts(splineLocStart);
        offsets[OFF_SPLINE_LOC_COUNT] = writer.putInts(splineLocCount);
        offsets[OFF_SPLINE_CHILD_START] = writer.putInts(splineChildStart);
        offsets[OFF_SPLINE_LOCATIONS] = writer.putFloats(splineLocations.toArray());
        offsets[OFF_SPLINE_DERIVATIVES] = writer.putFloats(splineDerivatives.toArray());
        offsets[OFF_SPLINE_CHILDREN] = writer.putInts(splineChildren.toArray());

        offsets[OFF_BLENDED_OCTAVE_START] = writer.putInts(blendedOctaveStart);
        offsets[OFF_BLENDED_XZ_MULT] = writer.putDoubles(blendedXzMult);
        offsets[OFF_BLENDED_Y_MULT] = writer.putDoubles(blendedYMult);
        offsets[OFF_BLENDED_XZ_FACTOR] = writer.putDoubles(blendedXzFactor);
        offsets[OFF_BLENDED_Y_FACTOR] = writer.putDoubles(blendedYFactor);
        offsets[OFF_BLENDED_SMEAR] = writer.putDoubles(blendedSmear);
        offsets[OFF_BLENDED_MIN_COUNT] = writer.putInts(blendedMinCount);
        offsets[OFF_BLENDED_MAIN_COUNT] = writer.putInts(blendedMainCount);

        return new DensityProgramImage(writer.toArray(), offsets, count, root, octaves.count());
    }

    /**
     * Emits the reachable graph in post-order, so that every instruction appears after all of the
     * instructions it reads. Shared subgraphs are emitted once.
     */
    private static int[] postOrder(DensityProgram program, int[] ops, int[] ia, int[] ib, int[] ic) {
        boolean[] visited = new boolean[ops.length];
        boolean[] splineVisited = new boolean[program.splineCount()];
        IntList order = new IntList();
        visit(program, ops, ia, ib, ic, program.root(), visited, splineVisited, order);
        return order.toArray();
    }

    private static void visit(DensityProgram program, int[] ops, int[] ia, int[] ib, int[] ic, int pc,
            boolean[] visited, boolean[] splineVisited, IntList order) {
        if (visited[pc]) {
            return;
        }
        visited[pc] = true;
        if (isChild(ops[pc], 0)) {
            visit(program, ops, ia, ib, ic, ia[pc], visited, splineVisited, order);
        }
        if (isChild(ops[pc], 1)) {
            visit(program, ops, ia, ib, ic, ib[pc], visited, splineVisited, order);
        }
        if (isChild(ops[pc], 2)) {
            visit(program, ops, ia, ib, ic, ic[pc], visited, splineVisited, order);
        }
        if (ops[pc] == DensityProgram.SPLINE) {
            // A spline reads the instructions computing its coordinates, and those dependencies live
            // in the spline table rather than in the instruction's operands. Nested spline values have
            // coordinates of their own, so the whole spline tree has to be walked.
            visitSplineCoordinates(program, ops, ia, ib, ic, ia[pc], visited, splineVisited, order);
        }
        order.add(pc);
    }

    private static void visitSplineCoordinates(DensityProgram program, int[] ops, int[] ia, int[] ib,
            int[] ic, int splineIndex, boolean[] visited, boolean[] splineVisited, IntList order) {
        if (splineVisited[splineIndex]) {
            return;
        }
        splineVisited[splineIndex] = true;
        DensityProgram.SplineNode node = program.spline(splineIndex);
        if (node.kind() == DensityProgram.SplineNode.CONSTANT) {
            return;
        }
        int coordinate = node.coordinate();
        if (coordinate >= 0) {
            visit(program, ops, ia, ib, ic, coordinate, visited, splineVisited, order);
        }
        for (int child : node.children()) {
            visitSplineCoordinates(program, ops, ia, ib, ic, child, visited, splineVisited, order);
        }
    }

    /** Whether operand {@code slot} of {@code op} is an instruction index rather than a table index. */
    private static boolean isChild(int op, int slot) {
        return switch (slot) {
            case 0 -> switch (op) {
                case DensityProgram.SHIFTED_NOISE, DensityProgram.WEIRD_SCALED_TYPE1,
                        DensityProgram.WEIRD_SCALED_TYPE2, DensityProgram.RANGE_CHOICE,
                        DensityProgram.BLEND_DENSITY, DensityProgram.ADD, DensityProgram.MUL,
                        DensityProgram.MIN, DensityProgram.MAX, DensityProgram.MUL_ADD,
                        DensityProgram.ABS, DensityProgram.SQUARE, DensityProgram.CUBE,
                        DensityProgram.HALF_NEGATIVE, DensityProgram.QUARTER_NEGATIVE,
                        DensityProgram.INVERT, DensityProgram.SQUEEZE, DensityProgram.CLAMP,
                        DensityProgram.FIND_TOP_SURFACE -> true;
                default -> false;
            };
            case 1 -> switch (op) {
                case DensityProgram.SHIFTED_NOISE, DensityProgram.RANGE_CHOICE, DensityProgram.ADD,
                        DensityProgram.MUL, DensityProgram.MIN, DensityProgram.MAX,
                        DensityProgram.FIND_TOP_SURFACE -> true;
                default -> false;
            };
            case 2 -> switch (op) {
                case DensityProgram.SHIFTED_NOISE, DensityProgram.RANGE_CHOICE -> true;
                default -> false;
            };
            default -> false;
        };
    }

    private static int remap(int value, int op, int slot, int[] newIndex) {
        return isChild(op, slot) ? newIndex[value] : value;
    }

    /** {@code NormalNoise}'s {@code valueFactor}, derived from the non-zero amplitude span. */
    static double valueFactor(double[] amplitudes) {
        int minOctave = Integer.MAX_VALUE;
        int maxOctave = Integer.MIN_VALUE;
        for (int i = 0; i < amplitudes.length; i++) {
            if (amplitudes[i] != 0.0) {
                minOctave = Math.min(minOctave, i);
                maxOctave = Math.max(maxOctave, i);
            }
        }
        double span = maxOctave - minOctave;
        double expectedDeviation = 0.1 * (1.0 + 1.0 / (span + 1.0));
        return 0.16666666666666666 / expectedDeviation;
    }

    public byte[] blob() {
        return this.blob.clone();
    }

    public long[] offsets() {
        return this.offsets.clone();
    }

    public int instructionCount() {
        return this.instructionCount;
    }

    public int root() {
        return this.root;
    }

    public int octaveCount() {
        return this.octaveCount;
    }

    public String summary() {
        return "DensityProgramImage[" + this.blob.length + " bytes, " + this.instructionCount
                + " instructions, " + this.octaveCount + " octaves]";
    }

    /** Collects octaves from every noise and blended noise into one global table. */
    private static final class OctaveCollector {
        private final DoubleList origins = new DoubleList();
        private final ByteList permutations = new ByteList();
        private final IntList valid = new IntList();
        private final DoubleList amplitudes = new DoubleList();
        private int count;

        int add(NoiseSnapshot.Octave[] octaves) {
            int start = this.count;
            for (NoiseSnapshot.Octave octave : octaves) {
                addOne(octave);
            }
            return start;
        }

        int addRaw(ImprovedNoise[] levels) {
            int start = this.count;
            for (ImprovedNoise level : levels) {
                if (level == null) {
                    addOne(null);
                } else {
                    addOne(new NoiseSnapshot.Octave(level.xo, level.yo, level.zo, level.permutation()));
                }
            }
            return start;
        }

        private void addOne(NoiseSnapshot.Octave octave) {
            if (octave == null) {
                this.origins.add(0.0, 0.0, 0.0);
                this.permutations.add(new byte[PERMUTATION_BYTES]);
                this.valid.add(0);
            } else {
                this.origins.add(octave.xo(), octave.yo(), octave.zo());
                this.permutations.add(octave.permutation());
                this.valid.add(1);
            }
            this.count++;
        }

        int addAmplitudes(double[] values) {
            int start = this.amplitudes.size();
            this.amplitudes.add(values);
            return start;
        }

        double[] origins() {
            return this.origins.toArray();
        }

        byte[] permutations() {
            return this.permutations.toArray();
        }

        int[] valid() {
            return this.valid.toArray();
        }

        double[] amplitudes() {
            return this.amplitudes.toArray();
        }

        int count() {
            return this.count;
        }
    }

    /** Little-endian, 8-byte-aligned table builder. */
    private static final class BlobWriter {
        private byte[] data = new byte[4096];
        private int size;

        long putInts(int[] values) {
            long offset = align();
            ensure(values.length * Integer.BYTES);
            for (int value : values) {
                writeInt(value);
            }
            return offset;
        }

        long putDoubles(double[] values) {
            long offset = align();
            ensure(values.length * Double.BYTES);
            for (double value : values) {
                writeLong(Double.doubleToRawLongBits(value));
            }
            return offset;
        }

        long putFloats(float[] values) {
            long offset = align();
            ensure(values.length * Float.BYTES);
            for (float value : values) {
                writeInt(Float.floatToRawIntBits(value));
            }
            return offset;
        }

        long putBytes(byte[] values) {
            long offset = align();
            ensure(values.length);
            System.arraycopy(values, 0, this.data, this.size, values.length);
            this.size += values.length;
            return offset;
        }

        private long align() {
            while (this.size % Long.BYTES != 0) {
                ensure(1);
                this.data[this.size++] = 0;
            }
            return this.size;
        }

        private void writeInt(int value) {
            ensure(Integer.BYTES);
            this.data[this.size++] = (byte) value;
            this.data[this.size++] = (byte) (value >>> 8);
            this.data[this.size++] = (byte) (value >>> 16);
            this.data[this.size++] = (byte) (value >>> 24);
        }

        private void writeLong(long value) {
            writeInt((int) value);
            writeInt((int) (value >>> 32));
        }

        private void ensure(int extra) {
            if (this.size + extra <= this.data.length) {
                return;
            }
            int capacity = this.data.length;
            while (capacity < this.size + extra) {
                capacity *= 2;
            }
            this.data = java.util.Arrays.copyOf(this.data, capacity);
        }

        byte[] toArray() {
            return java.util.Arrays.copyOf(this.data, this.size);
        }
    }

    private static final class IntList {
        private int[] data = new int[16];
        private int size;

        void add(int value) {
            if (this.size == this.data.length) {
                this.data = java.util.Arrays.copyOf(this.data, this.size * 2);
            }
            this.data[this.size++] = value;
        }

        void add(int[] values) {
            for (int value : values) {
                add(value);
            }
        }

        int size() {
            return this.size;
        }

        int[] toArray() {
            return java.util.Arrays.copyOf(this.data, this.size);
        }
    }

    private static final class FloatList {
        private float[] data = new float[16];
        private int size;

        void add(float[] values) {
            for (float value : values) {
                if (this.size == this.data.length) {
                    this.data = java.util.Arrays.copyOf(this.data, this.size * 2);
                }
                this.data[this.size++] = value;
            }
        }

        int size() {
            return this.size;
        }

        float[] toArray() {
            return java.util.Arrays.copyOf(this.data, this.size);
        }
    }

    private static final class DoubleList {
        private double[] data = new double[16];
        private int size;

        void add(double... values) {
            for (double value : values) {
                if (this.size == this.data.length) {
                    this.data = java.util.Arrays.copyOf(this.data, this.size * 2);
                }
                this.data[this.size++] = value;
            }
        }

        int size() {
            return this.size;
        }

        double[] toArray() {
            return java.util.Arrays.copyOf(this.data, this.size);
        }
    }

    private static final class ByteList {
        private byte[] data = new byte[256];
        private int size;

        void add(byte[] values) {
            while (this.size + values.length > this.data.length) {
                this.data = java.util.Arrays.copyOf(this.data, this.data.length * 2);
            }
            System.arraycopy(values, 0, this.data, this.size, values.length);
            this.size += values.length;
        }

        byte[] toArray() {
            return java.util.Arrays.copyOf(this.data, this.size);
        }
    }
}
