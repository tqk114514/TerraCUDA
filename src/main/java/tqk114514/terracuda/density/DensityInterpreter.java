package tqk114514.terracuda.density;

import tqk114514.terracuda.math.VanillaMath;
import tqk114514.terracuda.noise.ImprovedNoise;
import tqk114514.terracuda.noise.PerlinNoise;
import tqk114514.terracuda.worldgen.NoiseSnapshot;

/**
 * The CPU reference evaluator for a {@link DensityProgram}: sequential, and as literal a transcription
 * of the vanilla node semantics as possible.
 *
 * <p>This is the oracle the CUDA interpreter kernel is checked against. It is deliberately dumb — no
 * marker caching, no interpolation, no batching — because its only job is to be obviously right. The
 * per-point memo it does keep is a pure speed measure: it caches by instruction index and is
 * invalidated whenever the evaluation context changes, so it can never change a result.
 *
 * <p>Two things it does <em>not</em> abstract away, because they are observable:
 * <ul>
 *   <li>{@code mul} short-circuits on a zero left operand, which shows up as the sign of zero;</li>
 *   <li>cubic splines are evaluated entirely in {@code float}, then widened, exactly as
 *       {@code CubicSpline.Multipoint.apply} does.</li>
 * </ul>
 *
 * <p>Not thread-safe: it holds per-context scratch state. One instance per thread.
 */
public final class DensityInterpreter {

    private final DensityProgram program;
    private final double[] memo;
    private final int[] stamp;
    private int version;

    private java.util.Map<Integer, Double> overrides = java.util.Map.of();

    private int blockX;
    private int blockY;
    private int blockZ;

    public DensityInterpreter(DensityProgram program) {
        this.program = program;
        this.memo = new double[program.size()];
        this.stamp = new int[program.size()];
    }

    public DensityProgram program() {
        return this.program;
    }

    /** Evaluates the program's root at one block position. */
    public double evaluate(int x, int y, int z) {
        this.overrides = java.util.Map.of();
        setContext(x, y, z);
        return eval(this.program.root());
    }

    /**
     * Evaluates the program with some instructions' values supplied from outside.
     *
     * <p>This is how the interpolated markers are handled: vanilla samples them on a coarse grid and
     * interpolates between the samples, so the density at a block is not what the sub-graph would
     * compute there. Passing the interpolated values in as overrides lets the rest of the DAG be
     * evaluated unchanged, which is exactly the split the GPU path uses — the corners come from the
     * device, the arithmetic above them does not.
     *
     * @param overrides instruction index to value; the program's own computation is skipped for those
     */
    public double evaluate(int x, int y, int z, java.util.Map<Integer, Double> overrides) {
        this.overrides = overrides;
        setContext(x, y, z);
        return eval(this.program.root());
    }

    /**
     * Evaluates an arbitrary instruction at one block position.
     *
     * <p>Used to check the device-side marker tables, which are evaluated at each marker's wrapped
     * instruction rather than at the program root.
     */
    public double evaluateAt(int instruction, int x, int y, int z) {
        setContext(x, y, z);
        return eval(instruction);
    }

    private void setContext(int x, int y, int z) {
        this.blockX = x;
        this.blockY = y;
        this.blockZ = z;
        this.version++;
    }

    private double eval(int pc) {
        if (this.stamp[pc] == this.version) {
            return this.memo[pc];
        }
        Double override = this.overrides.get(pc);
        double value = override != null ? override : compute(pc);
        this.memo[pc] = value;
        this.stamp[pc] = this.version;
        return value;
    }

    private double compute(int pc) {
        switch (this.program.op(pc)) {
            case DensityProgram.CONSTANT:
                return this.program.da(pc);
            case DensityProgram.Y_CLAMPED_GRADIENT:
                return VanillaMath.clampedMap(this.blockY, this.program.ia(pc), this.program.ib(pc),
                        this.program.da(pc), this.program.db(pc));
            case DensityProgram.NOISE:
                return noise(pc).getValue(this.blockX * this.program.da(pc),
                        this.blockY * this.program.db(pc), this.blockZ * this.program.da(pc));
            case DensityProgram.SHIFTED_NOISE: {
                double x = this.blockX * this.program.da(pc) + eval(this.program.ia(pc));
                double y = this.blockY * this.program.db(pc) + eval(this.program.ib(pc));
                double z = this.blockZ * this.program.da(pc) + eval(this.program.ic(pc));
                return this.program.noise(this.program.id(pc)).reference().getValue(x, y, z);
            }
            case DensityProgram.SHIFT:
                return shift(pc, this.blockX, this.blockY, this.blockZ);
            case DensityProgram.SHIFT_A:
                return shift(pc, this.blockX, 0.0, this.blockZ);
            case DensityProgram.SHIFT_B:
                return shift(pc, this.blockZ, this.blockX, 0.0);
            case DensityProgram.WEIRD_SCALED_TYPE1:
            case DensityProgram.WEIRD_SCALED_TYPE2: {
                double input = eval(this.program.ia(pc));
                double rarity = this.program.op(pc) == DensityProgram.WEIRD_SCALED_TYPE1
                        ? spaghettiRarity3D(input)
                        : spaghettiRarity2D(input);
                return rarity * Math.abs(this.program.noise(this.program.ib(pc)).reference()
                        .getValue(this.blockX / rarity, this.blockY / rarity, this.blockZ / rarity));
            }
            case DensityProgram.SPLINE:
                return evalSpline(this.program.ia(pc));
            case DensityProgram.RANGE_CHOICE: {
                double input = eval(this.program.ia(pc));
                return input >= this.program.da(pc) && input < this.program.db(pc)
                        ? eval(this.program.ib(pc))
                        : eval(this.program.ic(pc));
            }
            case DensityProgram.BLEND_DENSITY:
                // Blender.empty().blendDensity returns the input unchanged; a non-empty blender is a
                // legacy-world upgrade case that the design doc routes to the vanilla path entirely.
                return eval(this.program.ia(pc));
            case DensityProgram.BLEND_ALPHA:
                return 1.0;
            case DensityProgram.BLEND_OFFSET:
                return 0.0;
            case DensityProgram.BEARDIFIER:
                return 0.0;
            case DensityProgram.BLENDED_NOISE:
                return blendedNoise(this.program.ia(pc));
            case DensityProgram.ADD:
                return eval(this.program.ia(pc)) + eval(this.program.ib(pc));
            case DensityProgram.MUL: {
                // Vanilla returns 0.0 without evaluating the right operand; the sign of zero is observable.
                double left = eval(this.program.ia(pc));
                return left == 0.0 ? 0.0 : left * eval(this.program.ib(pc));
            }
            case DensityProgram.MIN:
                return Math.min(eval(this.program.ia(pc)), eval(this.program.ib(pc)));
            case DensityProgram.MAX:
                return Math.max(eval(this.program.ia(pc)), eval(this.program.ib(pc)));
            case DensityProgram.MUL_ADD: {
                double input = eval(this.program.ia(pc));
                return this.program.ib(pc) == 1 ? input * this.program.da(pc) : input + this.program.da(pc);
            }
            case DensityProgram.ABS:
                return Math.abs(eval(this.program.ia(pc)));
            case DensityProgram.SQUARE: {
                double input = eval(this.program.ia(pc));
                return input * input;
            }
            case DensityProgram.CUBE: {
                double input = eval(this.program.ia(pc));
                return input * input * input;
            }
            case DensityProgram.HALF_NEGATIVE: {
                double input = eval(this.program.ia(pc));
                return input > 0.0 ? input : input * 0.5;
            }
            case DensityProgram.QUARTER_NEGATIVE: {
                double input = eval(this.program.ia(pc));
                return input > 0.0 ? input : input * 0.25;
            }
            case DensityProgram.INVERT:
                return 1.0 / eval(this.program.ia(pc));
            case DensityProgram.SQUEEZE: {
                double clamped = VanillaMath.clamp(eval(this.program.ia(pc)), -1.0, 1.0);
                return clamped / 2.0 - clamped * clamped * clamped / 24.0;
            }
            case DensityProgram.CLAMP:
                return VanillaMath.clamp(eval(this.program.ia(pc)), this.program.da(pc), this.program.db(pc));
            case DensityProgram.FIND_TOP_SURFACE:
                return findTopSurface(pc);
            default:
                throw new IllegalStateException("unknown opcode " + this.program.op(pc) + " at " + pc);
        }
    }

    private tqk114514.terracuda.noise.NormalNoise noise(int pc) {
        return this.program.noise(this.program.ia(pc)).reference();
    }

    /** {@code ShiftNoise.compute}: the offset noise sampled at a quarter scale and scaled back up. */
    private double shift(int pc, double localX, double localY, double localZ) {
        return this.program.noise(this.program.ia(pc)).reference()
                .getValue(localX * 0.25, localY * 0.25, localZ * 0.25) * 4.0;
    }

    /**
     * {@code FindTopSurface.compute}: probe downwards in {@code cellHeight} steps until the density
     * turns positive. Each probe is a different y, so the memo has to be invalidated around it and
     * restored afterwards — otherwise the enclosing evaluation would keep reading probe results.
     */
    private double findTopSurface(int pc) {
        int densityPc = this.program.ia(pc);
        int lowerBound = this.program.ic(pc);
        int cellHeight = this.program.id(pc);
        int topY = VanillaMath.floor(eval(this.program.ib(pc)) / cellHeight) * cellHeight;
        if (topY <= lowerBound) {
            return lowerBound;
        }

        int savedVersion = this.version;
        int savedX = this.blockX;
        int savedY = this.blockY;
        int savedZ = this.blockZ;
        try {
            for (int probeY = topY; probeY >= lowerBound; probeY -= cellHeight) {
                setContext(savedX, probeY, savedZ);
                if (eval(densityPc) > 0.0) {
                    return probeY;
                }
            }
            return lowerBound;
        } finally {
            // Bumping the version rather than restoring it is the whole point: restoring would leave
            // the probe results stamped with versions the *next* evaluation will hand out, so they
            // would be mistaken for the outer context's values. Bumping invalidates every probe
            // result and re-establishes the outer position; the enclosing evaluation recomputes
            // whatever it had, which is merely slower, never wrong.
            setContext(savedX, savedY, savedZ);
        }
    }

    /**
     * {@code BlendedNoise.compute}. Unlike {@code PerlinNoise.getValue}, this walks the raw octave
     * levels directly and applies no amplitude normalisation, so the snapshot hands over the levels.
     */
    private double blendedNoise(int index) {
        NoiseSnapshot.BlendedNoiseData data = this.program.blended(index).data();
        ImprovedNoise[] main = data.main();
        ImprovedNoise[] minLimit = data.minLimit();
        ImprovedNoise[] maxLimit = data.maxLimit();

        double limitX = this.blockX * data.xzMultiplier();
        double limitY = this.blockY * data.yMultiplier();
        double limitZ = this.blockZ * data.xzMultiplier();
        double mainX = limitX / data.xzFactor();
        double mainY = limitY / data.yFactor();
        double mainZ = limitZ / data.xzFactor();
        double limitSmear = data.yMultiplier() * data.smearScaleMultiplier();
        double mainSmear = limitSmear / data.yFactor();

        double blendMin = 0.0;
        double blendMax = 0.0;
        double mainNoiseValue = 0.0;
        double pow = 1.0;

        for (int i = 0; i < 8; i++) {
            ImprovedNoise level = octave(main, i);
            if (level != null) {
                mainNoiseValue += level.noise(
                        PerlinNoise.wrap(mainX * pow), PerlinNoise.wrap(mainY * pow),
                        PerlinNoise.wrap(mainZ * pow), mainSmear * pow, mainY * pow) / pow;
            }
            pow /= 2.0;
        }

        double factor = (mainNoiseValue / 10.0 + 1.0) / 2.0;
        boolean isMax = factor >= 1.0;
        boolean isMin = factor <= 0.0;
        pow = 1.0;

        for (int i = 0; i < 16; i++) {
            double wx = PerlinNoise.wrap(limitX * pow);
            double wy = PerlinNoise.wrap(limitY * pow);
            double wz = PerlinNoise.wrap(limitZ * pow);
            double yScalePow = limitSmear * pow;
            if (!isMax) {
                ImprovedNoise level = octave(minLimit, i);
                if (level != null) {
                    blendMin += level.noise(wx, wy, wz, yScalePow, limitY * pow) / pow;
                }
            }
            if (!isMin) {
                ImprovedNoise level = octave(maxLimit, i);
                if (level != null) {
                    blendMax += level.noise(wx, wy, wz, yScalePow, limitY * pow) / pow;
                }
            }
            pow /= 2.0;
        }

        return VanillaMath.clampedLerp(factor, blendMin / 512.0, blendMax / 512.0) / 128.0;
    }

    /** {@code PerlinNoise.getOctaveNoise(i)} indexes the level array from the back. */
    private static ImprovedNoise octave(ImprovedNoise[] levels, int i) {
        return levels[levels.length - 1 - i];
    }

    /** {@code CubicSpline.Multipoint.apply} — float arithmetic throughout, widened by the caller. */
    private float evalSpline(int index) {
        DensityProgram.SplineNode node = this.program.spline(index);
        if (node.kind() == DensityProgram.SplineNode.CONSTANT) {
            return node.constant();
        }

        float[] locations = node.locations();
        float[] derivatives = node.derivatives();
        int[] children = node.children();
        float input = (float) eval(node.coordinate());

        int start = VanillaMath.binarySearch(0, locations.length, i -> input < locations[i]) - 1;
        int lastIndex = locations.length - 1;

        if (start < 0) {
            return linearExtend(input, locations, derivatives, 0, evalSpline(children[0]));
        }
        if (start == lastIndex) {
            return linearExtend(input, locations, derivatives, lastIndex, evalSpline(children[lastIndex]));
        }

        float x1 = locations[start];
        float x2 = locations[start + 1];
        float t = (input - x1) / (x2 - x1);
        float y1 = evalSpline(children[start]);
        float y2 = evalSpline(children[start + 1]);
        float d1 = derivatives[start];
        float d2 = derivatives[start + 1];
        float a = d1 * (x2 - x1) - (y2 - y1);
        float b = -d2 * (x2 - x1) + (y2 - y1);
        return VanillaMath.lerp(t, y1, y2) + t * (1.0F - t) * VanillaMath.lerp(t, a, b);
    }

    private static float linearExtend(float input, float[] locations, float[] derivatives, int index,
            float value) {
        float derivative = derivatives[index];
        return derivative == 0.0F ? value : value + derivative * (input - locations[index]);
    }

    /** {@code NoiseRouterData.QuantizedSpaghettiRarity.getSpaghettiRarity3D} (type_1). */
    static double spaghettiRarity3D(double rarity) {
        if (rarity < -0.5) {
            return 0.75;
        }
        if (rarity < 0.0) {
            return 1.0;
        }
        return rarity < 0.5 ? 1.5 : 2.0;
    }

    /** {@code NoiseRouterData.QuantizedSpaghettiRarity.getSphaghettiRarity2D} (type_2). */
    static double spaghettiRarity2D(double rarity) {
        if (rarity < -0.75) {
            return 0.5;
        }
        if (rarity < -0.5) {
            return 0.75;
        }
        if (rarity < 0.5) {
            return 1.0;
        }
        return rarity < 0.75 ? 2.0 : 3.0;
    }
}
