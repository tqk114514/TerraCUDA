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
                    job.run(this.filler);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            closeQuietly();
        }
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
