package tqk114514.terracuda.density;

import java.lang.reflect.Method;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.util.CubicSpline;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import tqk114514.terracuda.worldgen.NoiseSnapshot;

/**
 * Lowers a vanilla {@link DensityFunction} tree into a {@link DensityProgram}.
 *
 * <p>Why reflection: almost every node type in {@code DensityFunctions} is a {@code private} or
 * {@code protected} record, so they cannot be referenced, let alone pattern-matched, from another
 * package. Record components are read through {@link Class#getRecordComponents()}, which is public
 * API and therefore does not depend on guessing at field names. The one non-record that matters,
 * {@code BlendedNoise}, has its fields read by {@link NoiseSnapshot}.
 *
 * <p>Anything not understood raises {@link UnsupportedDensityFunctionException}. That is the intended
 * behaviour, not a failure: the design doc requires an unknown node — from a data pack or another mod
 * — to send the whole world back to vanilla rather than produce wrong terrain.
 *
 * <p>Two of vanilla's evaluation shortcuts are reproduced rather than simplified away. {@code mul}
 * returns {@code 0.0} without evaluating its second argument when the first is zero, and that is
 * observable as the <em>sign of zero</em>; the interpreter keeps it. {@code min}/{@code max} also
 * short-circuit, but there the eager form is provably the same value, so it is not special-cased.
 */
public final class DensityCompiler {

    /** Thrown when the DAG contains a node this compiler does not implement. */
    public static class UnsupportedDensityFunctionException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public UnsupportedDensityFunctionException(String message) {
            super(message);
        }
    }

    private static final Map<Class<?>, Map<String, Method>> ACCESSORS = new HashMap<>();

    private final List<Integer> op = new ArrayList<>();
    private final List<Integer> ia = new ArrayList<>();
    private final List<Integer> ib = new ArrayList<>();
    private final List<Integer> ic = new ArrayList<>();
    private final List<Integer> id = new ArrayList<>();
    private final List<Double> da = new ArrayList<>();
    private final List<Double> db = new ArrayList<>();
    private final List<DensityProgram.NoiseEntry> noises = new ArrayList<>();
    private final List<DensityProgram.SplineNode> splines = new ArrayList<>();
    private final List<DensityProgram.BlendedEntry> blended = new ArrayList<>();
    private final List<Integer> markerKinds = new ArrayList<>();
    private final List<Integer> markerRoots = new ArrayList<>();

    private final Map<DensityFunction, Integer> compiled = new IdentityHashMap<>();
    private final Map<NormalNoise, Integer> noiseIndices = new IdentityHashMap<>();
    private final Map<BlendedNoise, Integer> blendedIndices = new IdentityHashMap<>();

    /** Lowers {@code function}. The result is immutable and safe to cache per world. */
    public static DensityProgram lower(DensityFunction function) {
        DensityCompiler compiler = new DensityCompiler();
        int root = compiler.compile(function);
        return compiler.build(root);
    }

    private DensityProgram build(int root) {
        return new DensityProgram(toIntArray(this.op), toIntArray(this.ia), toIntArray(this.ib),
                toIntArray(this.ic), toIntArray(this.id), toDoubleArray(this.da), toDoubleArray(this.db),
                root, this.noises.toArray(DensityProgram.NoiseEntry[]::new),
                this.splines.toArray(DensityProgram.SplineNode[]::new),
                this.blended.toArray(DensityProgram.BlendedEntry[]::new),
                toIntArray(this.markerKinds), toIntArray(this.markerRoots));
    }

    private int compile(DensityFunction function) {
        Integer existing = this.compiled.get(function);
        if (existing != null) {
            return existing;
        }

        int pc = reserve();
        this.compiled.put(function, pc);

        String kind = function.getClass().getSimpleName();
        switch (kind) {
            case "Constant" -> set(pc, DensityProgram.CONSTANT, 0, 0, 0, 0,
                    (double) component(function, "value"), 0.0);
            case "YClampedGradient" -> set(pc, DensityProgram.Y_CLAMPED_GRADIENT,
                    (int) component(function, "fromY"), (int) component(function, "toY"), 0, 0,
                    (double) component(function, "fromValue"), (double) component(function, "toValue"));
            case "Noise" -> set(pc, DensityProgram.NOISE,
                    noiseIndex((DensityFunction.NoiseHolder) component(function, "noise")), 0, 0, 0,
                    (double) component(function, "xzScale"), (double) component(function, "yScale"));
            case "ShiftedNoise" -> set(pc, DensityProgram.SHIFTED_NOISE,
                    compile((DensityFunction) component(function, "shiftX")),
                    compile((DensityFunction) component(function, "shiftY")),
                    compile((DensityFunction) component(function, "shiftZ")),
                    noiseIndex((DensityFunction.NoiseHolder) component(function, "noise")),
                    (double) component(function, "xzScale"), (double) component(function, "yScale"));
            case "Shift" -> set(pc, DensityProgram.SHIFT,
                    noiseIndex((DensityFunction.NoiseHolder) component(function, "offsetNoise")),
                    0, 0, 0, 0.0, 0.0);
            case "ShiftA" -> set(pc, DensityProgram.SHIFT_A,
                    noiseIndex((DensityFunction.NoiseHolder) component(function, "offsetNoise")),
                    0, 0, 0, 0.0, 0.0);
            case "ShiftB" -> set(pc, DensityProgram.SHIFT_B,
                    noiseIndex((DensityFunction.NoiseHolder) component(function, "offsetNoise")),
                    0, 0, 0, 0.0, 0.0);
            case "WeirdScaledSampler" -> {
                Object rarity = component(function, "rarityValueMapper");
                int opcode = "TYPE2".equals(((Enum<?>) rarity).name())
                        ? DensityProgram.WEIRD_SCALED_TYPE2
                        : DensityProgram.WEIRD_SCALED_TYPE1;
                set(pc, opcode, compile((DensityFunction) component(function, "input")),
                        noiseIndex((DensityFunction.NoiseHolder) component(function, "noise")), 0, 0, 0.0, 0.0);
            }
            case "Spline" -> set(pc, DensityProgram.SPLINE,
                    splineIndex((CubicSpline<?, ?>) component(function, "spline")), 0, 0, 0, 0.0, 0.0);
            case "RangeChoice" -> set(pc, DensityProgram.RANGE_CHOICE,
                    compile((DensityFunction) component(function, "input")),
                    compile((DensityFunction) component(function, "whenInRange")),
                    compile((DensityFunction) component(function, "whenOutOfRange")), 0,
                    (double) component(function, "minInclusive"),
                    (double) component(function, "maxExclusive"));
            case "BlendDensity" -> set(pc, DensityProgram.BLEND_DENSITY,
                    compile((DensityFunction) component(function, "input")), 0, 0, 0, 0.0, 0.0);
            case "BlendAlpha" -> set(pc, DensityProgram.BLEND_ALPHA, 0, 0, 0, 0, 0.0, 0.0);
            case "BlendOffset" -> set(pc, DensityProgram.BLEND_OFFSET, 0, 0, 0, 0, 0.0, 0.0);
            case "BeardifierMarker" -> set(pc, DensityProgram.BEARDIFIER, 0, 0, 0, 0, 0.0, 0.0);
            case "BlendedNoise" -> set(pc, DensityProgram.BLENDED_NOISE,
                    blendedIndex((BlendedNoise) function), 0, 0, 0, 0.0, 0.0);
            case "Ap2" -> {
                String type = ((Enum<?>) component(function, "type")).name();
                int opcode = switch (type) {
                    case "ADD" -> DensityProgram.ADD;
                    case "MUL" -> DensityProgram.MUL;
                    case "MIN" -> DensityProgram.MIN;
                    case "MAX" -> DensityProgram.MAX;
                    default -> throw new UnsupportedDensityFunctionException("unknown Ap2 type " + type);
                };
                set(pc, opcode, compile((DensityFunction) component(function, "argument1")),
                        compile((DensityFunction) component(function, "argument2")), 0, 0, 0.0, 0.0);
            }
            case "MulOrAdd" -> {
                String type = ((Enum<?>) component(function, "specificType")).name();
                int sub = "MUL".equals(type) ? 1 : 0;
                set(pc, DensityProgram.MUL_ADD, compile((DensityFunction) component(function, "input")),
                        sub, 0, 0, (double) component(function, "argument"), 0.0);
            }
            case "Mapped" -> {
                String type = ((Enum<?>) component(function, "type")).name();
                int opcode = switch (type) {
                    case "ABS" -> DensityProgram.ABS;
                    case "SQUARE" -> DensityProgram.SQUARE;
                    case "CUBE" -> DensityProgram.CUBE;
                    case "HALF_NEGATIVE" -> DensityProgram.HALF_NEGATIVE;
                    case "QUARTER_NEGATIVE" -> DensityProgram.QUARTER_NEGATIVE;
                    case "INVERT" -> DensityProgram.INVERT;
                    case "SQUEEZE" -> DensityProgram.SQUEEZE;
                    default -> throw new UnsupportedDensityFunctionException("unknown Mapped type " + type);
                };
                set(pc, opcode, compile((DensityFunction) component(function, "input")), 0, 0, 0, 0.0, 0.0);
            }
            case "Clamp" -> set(pc, DensityProgram.CLAMP,
                    compile((DensityFunction) component(function, "input")), 0, 0, 0,
                    (double) component(function, "minValue"), (double) component(function, "maxValue"));
            case "FindTopSurface" -> set(pc, DensityProgram.FIND_TOP_SURFACE,
                    compile((DensityFunction) component(function, "density")),
                    compile((DensityFunction) component(function, "upperBound")),
                    (int) component(function, "lowerBound"), (int) component(function, "cellHeight"),
                    0.0, 0.0);
            case "Marker" -> {
                // Markers are kernel boundaries, not operations: record them and pass through.
                DensityFunctions.MarkerOrMarked marker = (DensityFunctions.MarkerOrMarked) function;
                int root = compile(marker.wrapped());
                this.compiled.put(function, root);
                // The marker enum type is package-private, so its name is read through Enum<?>.
                this.markerKinds.add(markerKind(((Enum<?>) marker.type()).name()));
                this.markerRoots.add(root);
                return root;
            }
            case "HolderHolder" -> {
                // A cross-file reference (e.g. overworld/sloped_cheese). Lowering inlines it.
                int root = compile(((DensityFunctions.HolderHolder) function).function().value());
                this.compiled.put(function, root);
                return root;
            }
            default -> throw new UnsupportedDensityFunctionException(
                    "unsupported density function: " + function.getClass().getName());
        }
        return pc;
    }

    private static int markerKind(String name) {
        return switch (name) {
            case "Interpolated" -> DensityProgram.MARKER_INTERPOLATED;
            case "FlatCache" -> DensityProgram.MARKER_FLAT_CACHE;
            case "Cache2D" -> DensityProgram.MARKER_CACHE_2D;
            case "CacheOnce" -> DensityProgram.MARKER_CACHE_ONCE;
            case "CacheAllInCell" -> DensityProgram.MARKER_CACHE_ALL_IN_CELL;
            default -> throw new UnsupportedDensityFunctionException("unknown marker type " + name);
        };
    }

    private int noiseIndex(DensityFunction.NoiseHolder holder) {
        NormalNoise noise = holder.noise();
        if (noise == null) {
            throw new UnsupportedDensityFunctionException(
                    "noise holder " + holder.noiseData() + " has not been instantiated");
        }
        Integer existing = this.noiseIndices.get(noise);
        if (existing != null) {
            return existing;
        }
        NoiseSnapshot snapshot = NoiseSnapshot.of(noise);
        int index = this.noises.size();
        this.noises.add(new DensityProgram.NoiseEntry(snapshot, snapshot.toNormalNoise()));
        this.noiseIndices.put(noise, index);
        return index;
    }

    private int blendedIndex(BlendedNoise noise) {
        Integer existing = this.blendedIndices.get(noise);
        if (existing != null) {
            return existing;
        }
        int index = this.blended.size();
        this.blended.add(new DensityProgram.BlendedEntry(NoiseSnapshot.blendedOf(noise)));
        this.blendedIndices.put(noise, index);
        return index;
    }

    private int splineIndex(CubicSpline<?, ?> spline) {
        int index = this.splines.size();
        this.splines.add(null); // reserve so nested values can reference this node's index
        this.splines.set(index, lowerSpline(spline));
        return index;
    }

    @SuppressWarnings("unchecked")
    private DensityProgram.SplineNode lowerSpline(CubicSpline<?, ?> spline) {
        if (spline instanceof CubicSpline.Constant<?, ?> constant) {
            return new DensityProgram.SplineNode(DensityProgram.SplineNode.CONSTANT, constant.value(),
                    -1, EMPTY_FLOATS, EMPTY_FLOATS, EMPTY_INTS);
        }
        if (spline instanceof CubicSpline.Multipoint<?, ?> multipoint) {
            DensityFunctions.Spline.Coordinate coordinate =
                    (DensityFunctions.Spline.Coordinate) multipoint.coordinate();
            int coordinatePc = compile(coordinate.function().value());

            List<CubicSpline<?, ?>> values = (List<CubicSpline<?, ?>>) (List<?>) multipoint.values();
            int[] children = new int[values.size()];
            for (int i = 0; i < values.size(); i++) {
                children[i] = splineIndex(values.get(i));
            }
            return new DensityProgram.SplineNode(DensityProgram.SplineNode.MULTIPOINT, 0.0f, coordinatePc,
                    multipoint.locations(), multipoint.derivatives(), children);
        }
        throw new UnsupportedDensityFunctionException("unsupported cubic spline: " + spline.getClass().getName());
    }

    private static final float[] EMPTY_FLOATS = new float[0];
    private static final int[] EMPTY_INTS = new int[0];

    private int reserve() {
        this.op.add(0);
        this.ia.add(0);
        this.ib.add(0);
        this.ic.add(0);
        this.id.add(0);
        this.da.add(0.0);
        this.db.add(0.0);
        return this.op.size() - 1;
    }

    private void set(int pc, int opcode, int a, int b, int c, int d, double d0, double d1) {
        this.op.set(pc, opcode);
        this.ia.set(pc, a);
        this.ib.set(pc, b);
        this.ic.set(pc, c);
        this.id.set(pc, d);
        this.da.set(pc, d0);
        this.db.set(pc, d1);
    }

    private static Object component(Object record, String name) {
        Method accessor = ACCESSORS.computeIfAbsent(record.getClass(), c -> new HashMap<>())
                .computeIfAbsent(name, n -> {
                    for (RecordComponent component : record.getClass().getRecordComponents()) {
                        if (component.getName().equals(n)) {
                            Method method = component.getAccessor();
                            method.setAccessible(true);
                            return method;
                        }
                    }
                    throw new UnsupportedDensityFunctionException(record.getClass().getSimpleName()
                            + " has no record component '" + n + "'");
                });
        try {
            return accessor.invoke(record);
        } catch (ReflectiveOperationException e) {
            throw new UnsupportedDensityFunctionException("cannot read " + name + " of "
                    + record.getClass().getSimpleName() + ": " + e);
        }
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] array = new int[values.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = values.get(i);
        }
        return array;
    }

    private static double[] toDoubleArray(List<Double> values) {
        double[] array = new double[values.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = values.get(i);
        }
        return array;
    }
}
