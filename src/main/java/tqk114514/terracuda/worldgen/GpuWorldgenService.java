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
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
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
 * <p>The CUDA context is thread-affine and the device work is serialised anyway, so this runs a single
 * dedicated thread that creates the context, builds the density program and executes submitted jobs.
 * Chunk generation happens on several {@code wgen_fill_noise} dispatcher threads, and none of them may
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

    /** One service per world. Released on server stop — see {@link #releaseAll()}. */
    private static final Map<RandomState, GpuWorldgenService> SERVICES = new ConcurrentHashMap<>();
    private static final int INIT_TIMEOUT_SECONDS = 30;

    /** A unit of work for the GPU thread, which completes its own future. */
    private interface Job {
        void run(GpuChunkFiller filler);
    }

    private final BlockingQueue<Job> jobs = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final CountDownLatch initialised = new CountDownLatch(1);
    private final Thread thread;

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

        int filled = this.filler == null ? 0 : this.filler.filledChunks();
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
        if (filled > 0) {
            // Only takeover fills chunks; shadow mode has no write-back to report.
            message.append(" (emit ").append(String.format("%.1f", this.filler.emitNanos() / 1.0e6 / filled))
                    .append(" + replay ").append(String.format("%.1f", this.filler.replayNanos() / 1.0e6 / filled))
                    .append(")");
        }
        int emitted = this.filler == null ? 0 : this.filler.emittedChunks();
        if (emitted > 0) {
            // Where emit goes: the density program, the two vein programs, and the per-block rules
            // loop. Three numbers rather than one because the first two are device passes that a
            // future change could skip, and the third is CPU work a future change could move.
            double millis = 1.0e6 * emitted;
            message.append(" [density ").append(String.format("%.1f", this.filler.densityNanos() / millis))
                    .append(", veins ").append(String.format("%.1f", this.filler.veinNanos() / millis))
                    .append(", rules ").append(String.format("%.1f", this.filler.rulesNanos() / millis))
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
     * Emits one chunk's blocks and writes them into {@code chunk}, on the GPU thread.
     *
     * <p>Returns immediately. The future completes once the chunk has been written, or completes
     * exceptionally if the device path could not do it — the caller is expected to fall back to
     * vanilla in that case, which is why this reports failure rather than swallowing it.
     */
    public CompletableFuture<Void> fill(ChunkAccess chunk) {
        CompletableFuture<Void> done = new CompletableFuture<>();
        if (!isReady()) {
            done.completeExceptionally(new IllegalStateException(unavailableReason));
            return done;
        }
        Job job = filler -> {
            try {
                filler.fill(chunk);
                done.complete(null);
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
