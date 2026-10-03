package tqk114514.terracuda.worldgen;

import java.lang.reflect.Field;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;
import tqk114514.terracuda.noise.NoiseParameters;

/**
 * An immutable snapshot of everything needed to reproduce one vanilla {@link NormalNoise}: its octave
 * layout plus the permutation table and origin of every octave's {@link ImprovedNoise}.
 *
 * <p>Why snapshot at all, rather than re-deriving? Because the seeding path is long — world seed →
 * {@code forkPositional} → {@code fromHashOf("octave_n")} → XORoshiro → permutation shuffle — and any
 * divergence anywhere along it silently produces different terrain. Reading the tables straight off
 * the objects the game already built removes that whole class of bug, and it keeps working when a
 * datapack or another mod changes the noise parameters. This is the design doc's section 2.4
 * decision, and the snapshot is what gets uploaded to the device.
 *
 * <p>The vanilla fields are private, so this uses reflection. The field handles are resolved once,
 * with a clear failure if the game's layout changes — the design doc requires exactly that, so a
 * future Minecraft update degrades to vanilla instead of reading garbage.
 */
public final class NoiseSnapshot {

    /** One Perlin octave: its origin offset and the 256-entry permutation table. */
    public record Octave(double xo, double yo, double zo, byte[] permutation) {

        public Octave {
            permutation = permutation.clone();
        }

        @Override
        public byte[] permutation() {
            return permutation.clone();
        }
    }

    private static final Field NORMAL_NOISE_FIRST = field(NormalNoise.class, "first");
    private static final Field NORMAL_NOISE_SECOND = field(NormalNoise.class, "second");
    private static final Field PERLIN_NOISE_LEVELS = field(PerlinNoise.class, "noiseLevels");
    private static final Field IMPROVED_NOISE_PERMUTATION = field(ImprovedNoise.class, "p");

    private static final Field BLENDED_MIN_LIMIT = field(BlendedNoise.class, "minLimitNoise");
    private static final Field BLENDED_MAX_LIMIT = field(BlendedNoise.class, "maxLimitNoise");
    private static final Field BLENDED_MAIN = field(BlendedNoise.class, "mainNoise");
    private static final Field BLENDED_XZ_MULTIPLIER = field(BlendedNoise.class, "xzMultiplier");
    private static final Field BLENDED_Y_MULTIPLIER = field(BlendedNoise.class, "yMultiplier");
    private static final Field BLENDED_XZ_FACTOR = field(BlendedNoise.class, "xzFactor");
    private static final Field BLENDED_Y_FACTOR = field(BlendedNoise.class, "yFactor");
    private static final Field BLENDED_SMEAR = field(BlendedNoise.class, "smearScaleMultiplier");

    /**
     * Everything {@code BlendedNoise.compute} needs. Unlike {@link NoiseSnapshot}, the octave stacks
     * are consumed raw: {@code compute} calls {@code ImprovedNoise.noise} directly on each level and
     * never applies the per-octave amplitude normalisation that {@code PerlinNoise.getValue} does.
     *
     * @param minLimit  {@code minLimitNoise.noiseLevels}, natural order
     * @param maxLimit  {@code maxLimitNoise.noiseLevels}
     * @param main      {@code mainNoise.noiseLevels}
     */
    public record BlendedNoiseData(
            tqk114514.terracuda.noise.ImprovedNoise[] minLimit,
            tqk114514.terracuda.noise.ImprovedNoise[] maxLimit,
            tqk114514.terracuda.noise.ImprovedNoise[] main,
            double xzMultiplier,
            double yMultiplier,
            double xzFactor,
            double yFactor,
            double smearScaleMultiplier) {
    }

    /** Extracts the raw octave stacks and multipliers from a live {@code BlendedNoise}. */
    public static BlendedNoiseData blendedOf(BlendedNoise noise) {
        return new BlendedNoiseData(
                levelsOf((PerlinNoise) read(BLENDED_MIN_LIMIT, noise)),
                levelsOf((PerlinNoise) read(BLENDED_MAX_LIMIT, noise)),
                levelsOf((PerlinNoise) read(BLENDED_MAIN, noise)),
                (double) read(BLENDED_XZ_MULTIPLIER, noise),
                (double) read(BLENDED_Y_MULTIPLIER, noise),
                (double) read(BLENDED_XZ_FACTOR, noise),
                (double) read(BLENDED_Y_FACTOR, noise),
                (double) read(BLENDED_SMEAR, noise));
    }

    /**
     * Rebuilds the raw octave levels of one {@code PerlinNoise}, in {@code noiseLevels} order.
     * {@code null} entries are zero-amplitude octaves.
     */
    public static tqk114514.terracuda.noise.ImprovedNoise[] levelsOf(PerlinNoise perlin) {
        ImprovedNoise[] levels = (ImprovedNoise[]) read(PERLIN_NOISE_LEVELS, perlin);
        tqk114514.terracuda.noise.ImprovedNoise[] out =
                new tqk114514.terracuda.noise.ImprovedNoise[levels.length];
        for (int i = 0; i < levels.length; i++) {
            if (levels[i] != null) {
                out[i] = new tqk114514.terracuda.noise.ImprovedNoise(
                        (byte[]) read(IMPROVED_NOISE_PERMUTATION, levels[i]),
                        levels[i].xo, levels[i].yo, levels[i].zo);
            }
        }
        return out;
    }

    private final NoiseParameters parameters;
    private final Octave[] firstOctaves;
    private final Octave[] secondOctaves;

    private NoiseSnapshot(NoiseParameters parameters, Octave[] firstOctaves, Octave[] secondOctaves) {
        this.parameters = parameters;
        this.firstOctaves = firstOctaves;
        this.secondOctaves = secondOctaves;
    }

    /** Extracts a snapshot from a live vanilla noise instance. */
    public static NoiseSnapshot of(NormalNoise noise) {
        PerlinNoise first = (PerlinNoise) read(NORMAL_NOISE_FIRST, noise);
        PerlinNoise second = (PerlinNoise) read(NORMAL_NOISE_SECOND, noise);
        return new NoiseSnapshot(new NoiseParameters(
                noise.parameters().firstOctave(),
                noise.parameters().amplitudes().toDoubleArray()),
                octavesOf(first),
                octavesOf(second));
    }

    private static Octave[] octavesOf(PerlinNoise perlin) {
        ImprovedNoise[] levels = (ImprovedNoise[]) read(PERLIN_NOISE_LEVELS, perlin);
        Octave[] octaves = new Octave[levels.length];
        for (int i = 0; i < levels.length; i++) {
            ImprovedNoise level = levels[i];
            if (level != null) {
                octaves[i] = new Octave(level.xo, level.yo, level.zo,
                        (byte[]) read(IMPROVED_NOISE_PERMUTATION, level));
            }
        }
        return octaves;
    }

    /** The octave layout: first octave index and per-octave amplitudes. */
    public NoiseParameters parameters() {
        return this.parameters;
    }

    /** Octaves of the first Perlin stack; {@code null} entries are zero-amplitude octaves. */
    public Octave[] firstOctaves() {
        return this.firstOctaves.clone();
    }

    /** Octaves of the second Perlin stack. */
    public Octave[] secondOctaves() {
        return this.secondOctaves.clone();
    }

    /** Number of octaves in each stack. */
    public int octaveCount() {
        return this.firstOctaves.length;
    }

    /**
     * Rebuilds the noise with this project's own implementation.
     *
     * <p>For the CPU reference path. The GPU path uploads {@link #firstOctaves()} and
     * {@link #secondOctaves()} directly instead.
     */
    public tqk114514.terracuda.noise.NormalNoise toNormalNoise() {
        return tqk114514.terracuda.noise.NormalNoise.fromPerlinNoise(this.parameters,
                toPerlinNoise(this.firstOctaves),
                toPerlinNoise(this.secondOctaves));
    }

    private tqk114514.terracuda.noise.PerlinNoise toPerlinNoise(Octave[] octaves) {
        double[] amplitudes = this.parameters.amplitudes();
        tqk114514.terracuda.noise.ImprovedNoise[] levels = new tqk114514.terracuda.noise.ImprovedNoise[octaves.length];
        for (int i = 0; i < octaves.length; i++) {
            Octave octave = octaves[i];
            if (octave != null) {
                levels[i] = new tqk114514.terracuda.noise.ImprovedNoise(
                        octave.permutation(), octave.xo(), octave.yo(), octave.zo());
            }
        }
        return new tqk114514.terracuda.noise.PerlinNoise(this.parameters.firstOctave(), amplitudes, levels);
    }

    private static Field field(Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException("vanilla layout changed: " + owner.getName()
                    + " has no field '" + name + "'", e);
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot access " + owner.getName() + "." + name
                    + " (is the game module opened to this mod?)", e);
        }
    }

    private static Object read(Field field, Object target) {
        try {
            return field.get(target);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot read " + field, e);
        }
    }
}
