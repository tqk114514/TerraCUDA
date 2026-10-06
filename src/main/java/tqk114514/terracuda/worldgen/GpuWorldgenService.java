package tqk114514.terracuda.worldgen;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import tqk114514.terracuda.chunk.EmittedChunk;
import net.minecraft.world.level.levelgen.RandomState;

import tqk114514.terracuda.chunk.EmittedChunk;
import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.CudaDeviceInfo;
import tqk114514.terracuda.cuda.CudaDriver;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.cuda.CudaException;
import tqk114514.terracuda.cuda.CudaKernels;
import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;

/**
 * Owns the GPU side of world generation for one {@link RandomState}.
 *
 * <p>The CUDA context is thread-affine and the device work is serialised anyway, so the device half
 * runs on a single dedicated thread that creates the context, builds the density program and emits
 * chunk densities. The CPU half of a chunk — the material rules and the write-back — runs on a rules
 * worker, handed buffers across a bounded queue: that is the measured 6.3 of the old 11.6 ms of
 * service time, all of it CPU work that used to serialise the device thread behind it. Chunk
 * generation happens on several {@code wgen_fill_noise} dispatcher threads, and none of them may
 * touch the context directly.
 *
 * <p>Two ways in, and the difference matters. {@link #fill} hands back a future and returns at once, so
 * the dispatcher can carry on scheduling while the device works; that is the shape the design doc calls
 * for, and it is what a batch has to be built on. {@link #blockIds} blocks its caller, because it is
 * the diagnostic path and there is nothing to overlap with.
 *
 * <p>Failure is contained by construction: if CUDA is unavailable, the module will not load, or the
 * density function contains something the compiler does not understand, {@link #isReady()} stays
 * false, {@link #unavailableReason()} explains why, and callers simply use vanilla. Nothing here
 * throws into the game's generation path.
 */
public final class GpuWorldgenService implements AutoCloseable {

    private static final int QUEUE_CAPACITY = 64;

    /**
     * Chunks waiting for their rules and write-back, handed from the device thread to the rules
     * worker.
     *
     * <p>Bounded on purpose: when it is full the device thread blocks on the hand-off, which is the
     * backpressure — the producer slows to the consumer instead of converting queue depth into
     * two-and-a-half-megabyte buffer sets. The queue in front of the device thread is what turns
     * overload into a vanilla fallback; this one is inside the pipeline, where waiting is the
     * intended behaviour.
     */
    private static final int REPLAY_CAPACITY = 4;

    /** One service per world. Released on server stop — see {@link #releaseAll()}. */
    private static final Map<RandomState, GpuWorldgenService> SERVICES = new ConcurrentHashMap<>();
    private static final int INIT_TIMEOUT_SECONDS = 30;

    /** A unit of work for the GPU thread, which completes its own future. */
    private interface Job {
        void run(GpuChunkFiller filler);
    }

    /** One chunk handed from the device thread to a rules worker. */
    private record ReplayJob(GpuChunkFiller filler, ChunkAccess chunk,
            GpuChunkFiller.DensityBuffers buffers, CompletableFuture<Void> done,
            Beardifier beardifier) {
    }

    private final BlockingQueue<Job> jobs = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final BlockingQueue<ReplayJob> replayJobs = new ArrayBlockingQueue<>(REPLAY_CAPACITY);
    private final CountDownLatch initialised = new CountDownLatch(1);
    private final Thread thread;
    private Thread[] replayThreads = new Thread[0];

    private volatile boolean ready;
    private volatile String unavailableReason = "not started";
    private volatile boolean closed;

    /** Read once; see the note on the same field in {@link GpuChunkFiller}. */
    private static final boolean TIMING = TerracudaConfig.timing();
    private static final int TIMING_EVERY = 128;

    /**
     * Time actually spent running jobs, and the wall clock it was measured over.
     *
     * <p>The wall clock starts at the first job rather than at startup, so an idle thread waiting for
     * the world to load does not count as idle capacity. That is deliberate: the question is whether
     * the thread is saturated while chunks are arriving, not whether it is busy over the session.
     */
    private long busyNanos;
    private long timedJobs;
    private long windowStartNanos;
    /** Per-job service times of the current window, kept for percentiles rather than a mean alone. */
    private final long[] serviceNanos = new long[TIMING_EVERY];

    private CudaContext context;
    private GpuChunkFiller filler;

    /**
     * The service for {@code randomState}, built on first use.
     *
     * <p>Keyed by {@link RandomState} because that is what a world's generator settings and seed
     * resolve to: the overworld, the nether and the end each get their own, with their own lowered
     * programs and their own device buffers.
     */
    public static GpuWorldgenService forWorld(RandomState randomState,
            NoiseGeneratorSettings settings) {
        return SERVICES.computeIfAbsent(randomState, key -> of(key, settings));
    }

    /**
     * Closes every service and forgets it.
     *
     * <p>Without this the map only ever grows: leaving a world and loading another leaves the first
     * one's CUDA context, its device buffers — eight megabytes and up — and its daemon thread alive
     * for the life of the process. Called from the server-stopping event.
     */
    public static void releaseAll() {
        List<GpuWorldgenService> services = new ArrayList<>(SERVICES.values());
        SERVICES.clear();
        for (GpuWorldgenService service : services) {
            service.close();
        }
    }

    private GpuWorldgenService(RandomState randomState, NoiseGeneratorSettings settings) {
        this.thread = new Thread(() -> run(randomState, settings), "TerraCUDA-GPU");
        this.thread.setDaemon(true);
        this.thread.start();
    }

    /**
     * Returns a service for {@code randomState}, once its GPU thread has finished initialising.
     *
     * <p>A service that cannot use the GPU is still returned, so the reason is available for the log;
     * check {@link #isReady()} before using it.
     */
    public static GpuWorldgenService of(RandomState randomState, NoiseGeneratorSettings settings) {
        GpuWorldgenService service = new GpuWorldgenService(randomState, settings);
        try {
            if (!service.initialised.await(INIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                service.unavailableReason = "initialisation timed out";
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            service.unavailableReason = "interrupted while initialising";
        }
        return service;
    }

    private void run(RandomState randomState, NoiseGeneratorSettings settings) {
        try {
            initialise(randomState, settings);
        } catch (Throwable t) {
            this.unavailableReason = "initialisation failed: " + t;
            this.ready = false;
        } finally {
            this.initialised.countDown();
        }
        try {
            while (!this.closed) {
                Job job = this.jobs.poll(200, TimeUnit.MILLISECONDS);
                if (job != null) {
                    long began = TIMING ? System.nanoTime() : 0L;
                    job.run(this.filler);
                    if (TIMING) {
                        long took = System.nanoTime() - began;
                        this.busyNanos += took;
                        this.serviceNanos[(int) (this.timedJobs % TIMING_EVERY)] = took;
                        if (this.timedJobs == 0L) {
                            this.windowStartNanos = began;
                        }
                        if (++this.timedJobs % TIMING_EVERY == 0) {
                            reportTiming();
                        }
                    }
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (TIMING && this.timedJobs > 0) {
                reportTiming();
            }
            closeQuietly();
        }
    }

    /**
     * One rules worker's loop: computes the material rules for a chunk, writes it, and completes
     * the future — both halves on this thread.
     *
     * <p>The split experiment (separate rules and write threads, with a queue between them)
     * measured 2:22 against the combined 2:07: the queue handoff and the thread switch cost more
     * than the 2 ms of device-thread handoff wait the split was designed to save. On this machine
     * (12 logical / 6 physical cores) the combined worker is the right shape; the split methods
     * ({@code applyRules} and {@code writeChunk}) stay because they are clearer than the
     * monolithic {@code fill} was, and because the write no longer holds sections during the
     * rules computation, which is a real improvement even at the same throughput.
     */
    private void runReplay(GpuChunkFiller.RulesContext context) {
        try {
            while (!this.closed || !this.replayJobs.isEmpty()) {
                ReplayJob job = this.replayJobs.poll(200, TimeUnit.MILLISECONDS);
                if (job == null) {
                    continue;
                }
                try {
                    EmittedChunk emitted = job.filler().applyRules(
                            job.chunk().getPos().x(), job.chunk().getPos().z(),
                            job.buffers(), context, job.beardifier());
                    job.filler().writeChunk(job.chunk(), emitted);
                    job.done().complete(null);
                } catch (Throwable t) {
                    job.done().completeExceptionally(t);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Logs what the GPU thread did over the last {@value #TIMING_EVERY} jobs, then starts a new
     * window.
     *
     * <p>The busy fraction is the one number that settles whether this thread is the throughput
     * limit: a saturated queue shows a fraction near 100%, and a thread waiting on something
     * upstream shows the gap. Reporting per window rather than cumulatively matters because a
     * single average over a whole run hides the difference between "busy throughout" and "busy
     * during the initial burst and idle afterwards".
     */
    private void reportTiming() {
        long wall = System.nanoTime() - this.windowStartNanos;
        if (wall <= 0 || this.timedJobs == 0) {
            return;
        }
        double perSecond = this.timedJobs * 1.0e9 / wall;
        double busyPct = 100.0 * this.busyNanos / wall;
        double serviceMs = this.busyNanos / 1.0e6 / this.timedJobs;

        StringBuilder message = new StringBuilder(128)
                .append("TerraCUDA: GPU thread ").append(String.format("%.1f", perSecond))
                .append(" chunks/s, busy ").append(String.format("%.1f", busyPct))
                .append("%, service ").append(String.format("%.1f", serviceMs)).append(" ms");
        // The mean says how much capacity the thread has; the percentiles say what the mean hides.
        // The first chunk's JIT, collector pauses, and a client sharing the CUDA device with its
        // renderer all live in the tail, and the tail is what a queue feels like.
        int samples = (int) Math.min(this.timedJobs, TIMING_EVERY);
        if (samples > 0) {
            long[] window = new long[samples];
            System.arraycopy(this.serviceNanos, 0, window, 0, samples);
            java.util.Arrays.sort(window);
            message.append(" (p50 ").append(String.format("%.1f", percentile(window, 50) / 1.0e6))
                    .append(", p95 ").append(String.format("%.1f", percentile(window, 95) / 1.0e6))
                    .append(")");
        }
        long emitted = this.filler == null ? 0 : this.filler.emittedChunks();
        if (emitted > 0) {
            // Where the time goes: the density program and the two vein programs are queued on the
            // device thread and overlap on their own streams, the rules loop and the write-back run
            // on the rules worker. Four numbers rather than one because the halves live on different
            // threads and can move independently. Shadow mode shows replay as zero, because it
            // never writes a chunk.
            double millis = 1.0e6 * emitted;
            message.append(" [density ").append(String.format("%.1f", this.filler.densityNanos() / millis))
                    .append(", veins ").append(String.format("%.1f", this.filler.veinNanos() / millis))
                    .append(", rules ").append(String.format("%.1f", this.filler.rulesNanos() / millis))
                    .append(", replay ").append(String.format("%.1f", this.filler.replayNanos() / millis))
                    .append("]");
        }
        message.append(" over ").append(this.timedJobs).append(" chunks");
        TerraCUDA.LOGGER.info(message.toString());

        this.busyNanos = 0L;
        this.timedJobs = 0L;
        this.windowStartNanos = System.nanoTime();
        if (this.filler != null) {
            this.filler.resetTiming();
        }
    }

    /** Nearest-rank percentile of a sorted array: the ceil(p · count)-th smallest sample. */
    private static long percentile(long[] sorted, int hundredths) {
        int rank = Math.min(sorted.length, (sorted.length * hundredths + 99) / 100);
        return sorted[rank - 1];
    }

    private void initialise(RandomState randomState, NoiseGeneratorSettings settings) {
        CudaEnvironment environment = CudaEnvironment.detect();
        if (!environment.available()) {
            this.unavailableReason = environment.reason();
            return;
        }
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        if (loaded.isEmpty()) {
            this.unavailableReason = "the CUDA driver library could not be loaded";
            return;
        }
        CudaDriver driver = loaded.get();
        driver.init();
        CudaDeviceInfo device = environment.firstDevice().orElseThrow();

        this.context = CudaContext.create(driver, device.index());
        CudaKernels.loadModule(this.context, device);

        this.filler = GpuChunkFiller.create(this.context, randomState, settings);
        int workers = Math.max(1, TerracudaConfig.rulesWorkers());
        this.replayThreads = new Thread[workers];
        for (int i = 0; i < workers; i++) {
            GpuChunkFiller.RulesContext context = this.filler.createRulesContext();
            this.replayThreads[i] = new Thread(() -> runReplay(context), "TerraCUDA-replay-" + i);
            this.replayThreads[i].setDaemon(true);
            this.replayThreads[i].start();
        }
        this.ready = true;
        this.unavailableReason = "";
    }

    /** Whether the device path can be used. */
    public boolean isReady() {
        return this.ready && !this.closed;
    }

    /** Why the device path is unavailable; blank when it is ready. */
    public String unavailableReason() {
        return this.unavailableReason;
    }

    /**
     * Emits one chunk's densities on the device thread and hands it to the rules worker.
     *
     * <p>Returns immediately. The future completes once the worker has applied the rules and
     * written the chunk, or completes exceptionally if either half could not do its part — the
     * caller is expected to fall back to vanilla in that case, which is why this reports failure
     * rather than swallowing it.
     *
     * @param beardifier the structure beard for this chunk, or {@code null} when none reaches it;
     *                   added to each block's density on the host, the way vanilla adds it in the
     *                   density DAG above the interpolator
     */
    public CompletableFuture<Void> fill(ChunkAccess chunk, Beardifier beardifier) {
        CompletableFuture<Void> done = new CompletableFuture<>();
        if (!isReady()) {
            done.completeExceptionally(new IllegalStateException(unavailableReason));
            return done;
        }
        Job job = filler -> {
            try {
                GpuChunkFiller.DensityBuffers buffers =
                        filler.emitDensities(chunk.getPos().x(), chunk.getPos().z());
                // A blocking hand-off: when the rules worker is REPLAY_CAPACITY chunks behind, the
                // device thread waits here rather than running ahead in memory.
                this.replayJobs.put(new ReplayJob(filler, chunk, buffers, done, beardifier));
            } catch (Throwable t) {
                done.completeExceptionally(t);
            }
        };
        if (!this.jobs.offer(job)) {
            // The device is behind. Blocking here would stall the dispatcher, and falling back to
            // vanilla is both correct and parallel, so the caller gets to choose.
            done.completeExceptionally(new IllegalStateException("the GPU queue is full"));
        }
        return done;
    }

    /**
     * Emits one chunk's blocks, blocking until they are ready.
     *
     * @return the ids and the fluid-update bitmap, or empty when the device path is unavailable
     */
    public Optional<EmittedChunk> blockIds(int chunkX, int chunkZ) {
        return submit(() -> this.filler.blockIds(chunkX, chunkZ));
    }

    private <T> Optional<T> submit(Callable<T> body) {
        if (!isReady()) {
            return Optional.empty();
        }
        CompletableFuture<T> result = new CompletableFuture<>();
        if (!this.jobs.offer(filler -> {
            try {
                result.complete(body.call());
            } catch (Throwable t) {
                result.completeExceptionally(t);
            }
        })) {
            return Optional.empty();
        }
        try {
            return Optional.ofNullable(result.get());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Optional.empty();
        } catch (ExecutionException e) {
            return Optional.empty();
        }
    }

    private void closeQuietly() {
        try {
            if (this.filler != null) {
                this.filler.close();
            }
        } catch (CudaException ignored) {
            // Shutting down; a failure here has nowhere useful to go.
        }
        try {
            if (this.context != null) {
                this.context.close();
            }
        } catch (CudaException ignored) {
            // As above.
        }
    }

    @Override
    public void close() {
        this.closed = true;
        this.thread.interrupt();
    }

    /** A one-line description for logs. */
    public String describe() {
        if (!isReady()) {
            return "unavailable (" + this.unavailableReason + ")";
        }
        return "ready";
    }
}
