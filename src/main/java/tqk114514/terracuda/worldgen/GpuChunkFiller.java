package tqk114514.terracuda.worldgen;

import java.util.concurrent.LinkedBlockingQueue;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import tqk114514.terracuda.chunk.ChunkReplay;
import tqk114514.terracuda.chunk.EmittedChunk;
import tqk114514.terracuda.config.TerracudaConfig;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.density.CornerInterpolation;
import tqk114514.terracuda.density.DensityCompiler;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.gpu.DensityEvaluatorGpu;

/**
 * Produces a chunk's blocks: the density comes from the device, the material rules run on a CPU, and
 * the result is written into the chunk.
 *
 * <p>This is M3's K3 and K4 joined up, and it is a hybrid on purpose. The design doc puts the material
 * rules in the K3 kernel too, but they are branch- and table-driven rather than floating-point heavy —
 * the FLOP density is in the interpolator corners and the arithmetic above them, and both of those are
 * on the device. Moving the aquifer across would mean a per-thread copy of its grid and caches for
 * very little gain, so it stays here until something measures otherwise.
 *
 * <p>The class is split at that same seam, because the halves measured differently: the rules and the
 * write-back were 6.3 of the 11.6 ms the device thread spent per chunk, all of it CPU work serialised
 * behind the device. In takeover the halves run on different threads — the device thread produces
 * {@link DensityBuffers} and the rules workers consume them, each through its own
 * {@link RulesContext} — while shadow mode and the tests use the synchronous composition,
 * {@link #blockIds}. Ownership keeps each half to its own state: the evaluators are touched only by
 * the device half, each worker's interpreters only by that worker, and the buffer pool is the
 * single crossing point.
 *
 * <p>Everything is per world: the lowered programs and the device buffers are built once and reused
 * for every chunk.
 */
public final class GpuChunkFiller implements AutoCloseable {

    private final NoiseGeneratorSettings settings;
    private final NoiseSettings geometry;
    private final RandomState randomState;

    /** The eight router programs, lowered once and shared by every rules worker's context. */
    private final DensityProgram barrierNoiseProgram;
    private final DensityProgram floodednessProgram;
    private final DensityProgram spreadProgram;
    private final DensityProgram lavaProgram;
    private final DensityProgram erosionProgram;
    private final DensityProgram depthProgram;
    private final DensityProgram veinGapProgram;
    private final DensityProgram preliminarySurfaceProgram;

    /**
     * The context the synchronous composition runs through: shadow mode and the parity tests,
     * single-threaded by construction. Takeover workers each get their own via
     * {@link #createRulesContext()}.
     */
    private final RulesContext inlineRules;

    private final DensityEvaluatorGpu densityEvaluator;
    private final DensityEvaluatorGpu veinToggleEvaluator;
    private final DensityEvaluatorGpu veinRidgedEvaluator;

    /**
     * One chunk's device-produced values, on loan from {@link #bufferPool} until
     * {@link #applyRules} returns them.
     *
     * <p>Each array is 786 KB, so a set is about two and a half megabytes when veins are on; they are
     * pooled because handing the collector a fresh set per chunk measured as page faults that cost
     * more than the copy the arrays exist to hold. The two vein arrays are {@code null} when the
     * world has no ore veins, mirroring the evaluators that would fill them.
     */
    public record DensityBuffers(double[] density, double[] veinToggle, double[] veinRidged) {
    }

    /**
     * Free buffer sets. Unbounded in principle, bounded in practice by the chunks in flight: the
     * device thread borrows a set, the rules worker returns it, and the worker queue in front of it
     * has a fixed capacity that the device thread blocks on rather than outrunning.
     */
    private final LinkedBlockingQueue<DensityBuffers> bufferPool = new LinkedBlockingQueue<>();

    /** Read once: a per-chunk branch on a volatile-free static is not worth the noise. */
    private static final boolean TIMING = TerracudaConfig.timing();

    /**
     * A finer split of the chunk's time: the density program, the two vein programs, the rules, and
     * the write-back. The first two are written by the device thread alone; the last two are added
     * to from as many rules workers as there are, so they are adders rather than plain fields, and
     * all are read across threads only by the periodic timing report, which tolerates a window
     * straddling a reset.
     */
    private volatile long densityNanos;
    private volatile long veinNanos;
    private final java.util.concurrent.atomic.LongAdder rulesNanos = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder replayNanos = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder emittedChunks = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder filledChunks = new java.util.concurrent.atomic.LongAdder();

    /**
     * Whether the material rules will ever ask for a vein value.
     *
     * <p>They will not: {@code MaterialRules.compute} returns null without calling the veinifier when
     * ore veins are off, and the veinifier is the only reader of {@code veinToggle} and
     * {@code veinRidged}. Evaluating both for a chunk that cannot use either was two full-chunk
     * device passes thrown away — a third of the thread's service time, for a dimension whose
     * settings say it has none.
     */
    private final boolean oreVeinsEnabled;

    private boolean closed;

    private GpuChunkFiller(CudaContext context, RandomState randomState, NoiseGeneratorSettings settings) {
        this.settings = settings;
        this.geometry = settings.noiseSettings();
        this.randomState = randomState;

        NoiseRouter router = randomState.router();
        DensityProgram density = DensityCompiler.lower(router.finalDensity());
        DensityProgram veinToggle = DensityCompiler.lower(router.veinToggle());
        DensityProgram veinRidged = DensityCompiler.lower(router.veinRidged());
        this.barrierNoiseProgram = DensityCompiler.lower(router.barrierNoise());
        this.floodednessProgram = DensityCompiler.lower(router.fluidLevelFloodednessNoise());
        this.spreadProgram = DensityCompiler.lower(router.fluidLevelSpreadNoise());
        this.lavaProgram = DensityCompiler.lower(router.lavaNoise());
        this.erosionProgram = DensityCompiler.lower(router.erosion());
        this.depthProgram = DensityCompiler.lower(router.depth());
        this.veinGapProgram = DensityCompiler.lower(router.veinGap());
        this.preliminarySurfaceProgram = DensityCompiler.lower(router.preliminarySurfaceLevel());
        this.inlineRules = createRulesContext();

        this.densityEvaluator = DensityEvaluatorGpu.forProgram(context, density, this.geometry);
        this.oreVeinsEnabled = settings.oreVeinsEnabled();
        this.veinToggleEvaluator = this.oreVeinsEnabled
                ? DensityEvaluatorGpu.forProgram(context, veinToggle, this.geometry)
                : null;
        this.veinRidgedEvaluator = this.oreVeinsEnabled
                ? DensityEvaluatorGpu.forProgram(context, veinRidged, this.geometry)
                : null;
    }

    public static GpuChunkFiller create(CudaContext context, RandomState randomState,
            NoiseGeneratorSettings settings) {
        return new GpuChunkFiller(context, randomState, settings);
    }

    /**
     * The device half of one chunk: every density the material rules will read, in borrowed buffers.
     *
     * <p>Runs on the device thread. The buffers are on loan until handed to {@link #applyRules},
     * which returns them; the synchronous composition does that immediately, the takeover pipeline
     * does it on the rules worker after the chunk crosses the queue.
     *
     * <p>final_density is a cache_all_in_cell marker, so the rules see the lerp3 blend; the vein
     * functions are read straight, so they take the incremental one. The difference is in the last
     * bits and InterpolationOrderTest pins it down.
     */
    public DensityBuffers emitDensities(int chunkX, int chunkZ) {
        int count = 16 * 16 * this.geometry.height();
        DensityBuffers buffers = acquireBuffers(count);

        long began = TIMING ? System.nanoTime() : 0L;
        // Three submissions, then three waits: the three programs are independent and now run on
        // separate streams, so their corner tables, block passes and copies overlap on the device.
        // Submitting one and waiting before the next was the serialisation.
        this.densityEvaluator.submitChunk(this.geometry, chunkX, chunkZ,
                CornerInterpolation.LERP3, buffers.density());
        if (this.oreVeinsEnabled) {
            this.veinToggleEvaluator.submitChunk(this.geometry, chunkX, chunkZ,
                    CornerInterpolation.INCREMENTAL, buffers.veinToggle());
            this.veinRidgedEvaluator.submitChunk(this.geometry, chunkX, chunkZ,
                    CornerInterpolation.INCREMENTAL, buffers.veinRidged());
        }
        long densityAt = TIMING ? System.nanoTime() : 0L;
        this.densityEvaluator.awaitChunk();
        if (this.oreVeinsEnabled) {
            this.veinToggleEvaluator.awaitChunk();
            this.veinRidgedEvaluator.awaitChunk();
        }
        long veinsAt = TIMING ? System.nanoTime() : 0L;
        if (TIMING) {
            this.densityNanos += densityAt - began;
            this.veinNanos += veinsAt - densityAt;
        }
        return buffers;
    }

    /**
     * The CPU half of one chunk: the material rules over values the device already produced, plus
     * the fluid-update bitmap {@code doFill} maintains.
     *
     * <p>Runs on a rules worker in takeover, on the caller in the synchronous composition. The
     * interpreters in {@code context} carry per-chunk state ({@code setChunk} for the flat caches),
     * which is why a context belongs to one worker: two chunks through the same context at once
     * would answer each other's columns.
     */
    public EmittedChunk applyRules(int chunkX, int chunkZ, DensityBuffers buffers, RulesContext context,
            Beardifier beardifier) {
        int minY = this.geometry.minY();
        int height = this.geometry.height();
        long began = TIMING ? System.nanoTime() : 0L;
        // Reused for every block the beard reaches: one mutable context, no per-block allocation.
        BeardContext beardCtx = beardifier != null ? new BeardContext() : null;
        try {
            // The material rules read router entries through interpreters, and a flat cache inside one
            // of those needs to know which chunk's columns it is answering for.
            for (DensityInterpreter interpreter : context.all()) {
                interpreter.setChunk(chunkX * 16, chunkZ * 16);
            }

            double[] densities = buffers.density();
            double[] veinToggleValues = buffers.veinToggle();
            double[] veinRidgedValues = buffers.veinRidged();

            MaterialRules.RouterNoises noises = new MaterialRules.RouterNoises(
                    MaterialRules.entry(context.barrierNoise, context.barrierNoise.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.floodedness, context.floodedness.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.spread, context.spread.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.lava, context.lava.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.erosion, context.erosion.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.depth, context.depth.program(), chunkX, chunkZ),
                    MaterialRules.entry(context.veinGap, context.veinGap.program(), chunkX, chunkZ),
                    (x, y, z) -> veinValue(veinToggleValues, "veinToggle", x, y, z, chunkX, chunkZ, minY, height),
                    (x, y, z) -> veinValue(veinRidgedValues, "veinRidged", x, y, z, chunkX, chunkZ, minY, height));

            MaterialRules rules = MaterialRules.forChunk(noises, surfaceLevels(context), fluidPicker(),
                    VanillaRandomExport.export(this.randomState.aquiferRandom()),
                    VanillaRandomExport.export(this.randomState.oreRandom()), chunkX, chunkZ, this.geometry,
                    this.settings.isAquifersEnabled(), this.settings.oreVeinsEnabled());

            BlockState defaultBlock = this.settings.defaultBlock();
            int chunkMinBlockX = chunkX * 16;
            int chunkMinBlockZ = chunkZ * 16;
            int[] ids = new int[16 * 16 * height];
            long[] fluidUpdates = new long[(ids.length + 63) >>> 6];

            for (int yLocal = 0; yLocal < height; yLocal++) {
                int y = minY + yLocal;
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        int blockIndex = (x * 16 + z) * height + yLocal;
                        double value = densities[blockIndex];
                        if (beardCtx != null) {
                            // The beard is added at full block resolution, above the interpolator —
                            // the same place vanilla's `add(finalDensity, Beardifier)` sits in the
                            // DAG. Most blocks fall outside the structure's bounding box and the
                            // compute returns zero at the cost of one isInside check.
                            beardCtx.x = chunkMinBlockX + x;
                            beardCtx.y = y;
                            beardCtx.z = chunkMinBlockZ + z;
                            value += beardifier.compute(beardCtx);
                        }
                        BlockState state = rules.compute(chunkMinBlockX + x, y,
                                chunkMinBlockZ + z, value);
                        if (state == null) {
                            state = defaultBlock;
                        }
                        ids[blockIndex] = Block.getId(state);
                        // doFill's condition, in the same place it evaluates it: inside the branch that
                        // actually writes a block.
                        if (rules.shouldScheduleFluidUpdate() && !state.getFluidState().isEmpty()) {
                            fluidUpdates[blockIndex >>> 6] |= 1L << (blockIndex & 63);
                        }
                    }
                }
            }
            return new EmittedChunk(ids, fluidUpdates);
        } finally {
            if (TIMING) {
                this.rulesNanos.add(System.nanoTime() - began);
                this.emittedChunks.increment();
            }
            releaseBuffers(buffers);
        }
    }

    /**
     * One chunk's blocks, indexed {@code (x * 16 + z) * height + (y - minY)}.
     *
     * <p>The synchronous composition: both halves on the calling thread, which is what shadow mode
     * and the tests want. Takeover does not come through here — it runs the halves as a pipeline.
     */
    public EmittedChunk blockIds(int chunkX, int chunkZ) {
        return applyRules(chunkX, chunkZ, emitDensities(chunkX, chunkZ), this.inlineRules, null);
    }

    /**
     * Reads a vein value that the caller has already computed for the whole chunk.
     *
     * <p>The null check is not defensive coding: when ore veins are off the arrays are never filled,
     * and a read means the material rules started asking for them without the gate that is supposed
     * to stop them. Failing loudly there is the point — a stale or zero value would produce plausible
     * terrain with the wrong ore in it.
     */
    private static double veinValue(double[] values, String name, int x, int y, int z,
            int chunkX, int chunkZ, int minY, int height) {
        if (values == null) {
            throw new IllegalStateException(name + " was read while ore veins are disabled");
        }
        return values[index(x, y, z, chunkX, chunkZ, minY, height)];
    }

    /**
     * The write-back half of the rules pipeline, on its own thread: turns one chunk's computed
     * block states into written blocks, holding the sections for the duration the way
     * {@code fillFromNoise} holds them around {@code doFill}.
     *
     * <p>Split from {@link #applyRules} because the halves measured differently on the pipeline:
     * the rules computation is 4.8 ms of CPU per chunk and the write is 2.1 ms of memory traffic,
     * and running both on one worker meant the device thread waited 2 ms per chunk on the handoff
     * that the combined 7 ms created. On separate threads the rules workers turn at 4.8 ms
     * (~208/s each) and the write workers at 2.1 ms (~476/s each), and the device thread's
     * queue drains before it fills.
     */
    public void writeChunk(ChunkAccess chunk, EmittedChunk emitted) {
        int minY = this.geometry.minY();
        int height = this.geometry.height();
        int top = chunk.getSectionIndex(minY + height - 1);
        int bottom = chunk.getSectionIndex(minY);

        java.util.List<LevelChunkSection> held = new java.util.ArrayList<>(top - bottom + 1);
        for (int index = top; index >= bottom; index--) {
            LevelChunkSection section = chunk.getSection(index);
            section.acquire();
            held.add(section);
        }
        try {
            long began = TIMING ? System.nanoTime() : 0L;
            ChunkReplay.write(chunk, emitted, this.geometry);
            if (TIMING) {
                this.replayNanos.add(System.nanoTime() - began);
                this.filledChunks.increment();
            }
        } finally {
            for (LevelChunkSection section : held) {
                section.release();
            }
        }
    }

    private DensityBuffers acquireBuffers(int count) {
        DensityBuffers buffers = this.bufferPool.poll();
        if (buffers == null) {
            buffers = new DensityBuffers(new double[count],
                    this.oreVeinsEnabled ? new double[count] : null,
                    this.oreVeinsEnabled ? new double[count] : null);
        }
        return buffers;
    }

    private void releaseBuffers(DensityBuffers buffers) {
        this.bufferPool.offer(buffers);
    }

    /** Nanoseconds in the density program, across every emit since the last {@link #resetTiming()}. */
    public long densityNanos() {
        return this.densityNanos;
    }

    /** Nanoseconds in the two vein programs, or zero when ore veins are off. */
    public long veinNanos() {
        return this.veinNanos;
    }

    /** Nanoseconds in the material rules, added across every rules worker; this half runs on CPUs. */
    public long rulesNanos() {
        return this.rulesNanos.sum();
    }

    /** Nanoseconds writing emitted blocks into chunks; zero in shadow mode, which writes nothing. */
    public long replayNanos() {
        return this.replayNanos.sum();
    }

    /** Blocks emitted since the reset, which is what the segment figures divide by. */
    public long emittedChunks() {
        return this.emittedChunks.sum();
    }

    /** Chunks written since the reset; zero in shadow mode, which never writes a chunk. */
    public long filledChunks() {
        return this.filledChunks.sum();
    }

    public void resetTiming() {
        this.densityNanos = 0L;
        this.veinNanos = 0L;
        this.rulesNanos.reset();
        this.replayNanos.reset();
        this.emittedChunks.reset();
        this.filledChunks.reset();
    }

    private MaterialRules.SurfaceLevels surfaceLevels(RulesContext context) {
        DensityInterpreter interpreter = context.preliminarySurface;
        // Vanilla caches the preliminary surface level per column, and it matters: the aquifer asks
        // for the same columns over and over — thirteen sampling offsets times every grid cell — and
        // each miss is a find_top_surface, which walks the density downwards in steps of eight.
        java.util.Map<Long, Integer> cache = new java.util.HashMap<>();
        return new MaterialRules.SurfaceLevels() {
            @Override
            public int preliminarySurfaceLevel(int blockX, int blockZ) {
                // Vanilla quantises to quart positions before looking the column up.
                int quartX = (blockX >> 2) << 2;
                int quartZ = (blockZ >> 2) << 2;
                long key = (long) quartX << 32 | quartZ & 0xFFFFFFFFL;
                Integer cached = cache.get(key);
                if (cached != null) {
                    return cached;
                }
                int level = tqk114514.terracuda.math.VanillaMath.floor(
                        interpreter.evaluate(quartX, 0, quartZ));
                cache.put(key, level);
                return level;
            }

            @Override
            public int maxPreliminarySurfaceLevel(int minBlockX, int minBlockZ, int maxBlockX,
                    int maxBlockZ) {
                int max = Integer.MIN_VALUE;
                for (int blockZ = minBlockZ; blockZ <= maxBlockZ; blockZ += 4) {
                    for (int blockX = minBlockX; blockX <= maxBlockX; blockX += 4) {
                        max = Math.max(max, preliminarySurfaceLevel(blockX, blockZ));
                    }
                }
                return max;
            }
        };
    }

    private Aquifer.FluidPicker fluidPicker() {
        int seaLevel = this.settings.seaLevel();
        return (x, y, z) -> y < Math.min(-54, seaLevel)
                ? new Aquifer.FluidStatus(-54, net.minecraft.world.level.block.Blocks.LAVA.defaultBlockState())
                : new Aquifer.FluidStatus(seaLevel, net.minecraft.world.level.block.Blocks.WATER.defaultBlockState());
    }

    private static int index(int x, int y, int z, int chunkX, int chunkZ, int minY, int height) {
        return ((x - chunkX * 16) * 16 + (z - chunkZ * 16)) * height + (y - minY);
    }

    /**
     * One rules worker's interpreter set, over programs every context shares.
     *
     * <p>The interpreters carry per-chunk mutable state — {@code setChunk} points the flat caches at
     * the chunk the rules are asking about — so a context belongs to exactly one worker: two chunks
     * through one context at once would answer each other's columns. The programs underneath are
     * immutable and lowered once per world, so a context is eight thin wrappers, cheap to make per
     * worker.
     */
    public static final class RulesContext {
        final DensityInterpreter barrierNoise;
        final DensityInterpreter floodedness;
        final DensityInterpreter spread;
        final DensityInterpreter lava;
        final DensityInterpreter erosion;
        final DensityInterpreter depth;
        final DensityInterpreter veinGap;
        final DensityInterpreter preliminarySurface;

        private RulesContext(DensityInterpreter barrierNoise, DensityInterpreter floodedness,
                DensityInterpreter spread, DensityInterpreter lava, DensityInterpreter erosion,
                DensityInterpreter depth, DensityInterpreter veinGap,
                DensityInterpreter preliminarySurface) {
            this.barrierNoise = barrierNoise;
            this.floodedness = floodedness;
            this.spread = spread;
            this.lava = lava;
            this.erosion = erosion;
            this.depth = depth;
            this.veinGap = veinGap;
            this.preliminarySurface = preliminarySurface;
        }

        DensityInterpreter[] all() {
            return new DensityInterpreter[] { this.barrierNoise, this.floodedness, this.spread,
                    this.lava, this.erosion, this.depth, this.veinGap, this.preliminarySurface };
        }
    }

    /** Builds one rules worker's context over the shared, immutable router programs. */
    public RulesContext createRulesContext() {
        return new RulesContext(
                new DensityInterpreter(this.barrierNoiseProgram),
                new DensityInterpreter(this.floodednessProgram),
                new DensityInterpreter(this.spreadProgram),
                new DensityInterpreter(this.lavaProgram),
                new DensityInterpreter(this.erosionProgram),
                new DensityInterpreter(this.depthProgram),
                new DensityInterpreter(this.veinGapProgram),
                new DensityInterpreter(this.preliminarySurfaceProgram));
    }

    /**
     * The block coordinate context the beard compute reads, mutable and reused per block.
     *
     * <p>One instance per applyRules call that has a beard, zero when it does not. The interface
     * has three int methods; a record would allocate per block, and 98,304 records per chunk near
     * a structure is garbage the rules loop should not make.
     */
    private static final class BeardContext implements DensityFunction.FunctionContext {
        int x;
        int y;
        int z;

        @Override
        public int blockX() {
            return this.x;
        }

        @Override
        public int blockY() {
            return this.y;
        }

        @Override
        public int blockZ() {
            return this.z;
        }
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        for (DensityEvaluatorGpu evaluator : new DensityEvaluatorGpu[] {
                this.densityEvaluator, this.veinToggleEvaluator, this.veinRidgedEvaluator}) {
            if (evaluator == null) {
                continue;
            }
            try {
                evaluator.close();
            } catch (RuntimeException ignored) {
                // Shutting down; a failure here has nowhere useful to go.
            }
        }
    }
}
