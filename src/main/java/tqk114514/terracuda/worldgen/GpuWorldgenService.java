package tqk114514.terracuda.worldgen;

import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;

import tqk114514.terracuda.chunk.ChunkReplay;
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
 * Chunk generation happens on several {@code wgen_fill_noise} worker threads, and none of them may
 * touch the context directly — they submit and wait. That is the seed of the batching dispatcher the
 * design doc calls for; the difference is that this version runs one job at a time.
 *
 * <p>Failure is contained by construction: if CUDA is unavailable, the module will not load, or the
 * density function contains something the compiler does not understand, {@link #isReady()} stays
 * false, {@link #unavailableReason()} explains why, and callers simply use vanilla. Nothing here
 * throws into the game's generation path.
 */
public final class GpuWorldgenService implements AutoCloseable {

    private static final int QUEUE_CAPACITY = 64;
    private static final int INIT_TIMEOUT_SECONDS = 30;

    private final BlockingQueue<Runnable> jobs = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final CountDownLatch initialised = new CountDownLatch(1);
    private final Thread thread;

    private volatile boolean ready;
    private volatile String unavailableReason = "not started";
    private volatile boolean closed;

    private CudaContext context;
    private GpuChunkFiller filler;

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
                Runnable job = this.jobs.poll(200, TimeUnit.MILLISECONDS);
                if (job != null) {
                    job.run();
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
     * Emits one chunk's blocks.
     *
     * @return the ids and the fluid-update bitmap, or empty when the device path is unavailable
     */
    public Optional<EmittedChunk> blockIds(int chunkX, int chunkZ) {
        return submit(() -> this.filler.blockIds(chunkX, chunkZ));
    }

    /** Emits one chunk's blocks and writes them into {@code chunk}. */
    public Optional<EmittedChunk> fill(ChunkAccess chunk, int minY, int height) {
        Optional<EmittedChunk> emitted = blockIds(chunk.getPos().x(), chunk.getPos().z());
        emitted.ifPresent(blocks -> ChunkReplay.write(chunk, blocks, minY, height));
        return emitted;
    }

    private <T> Optional<T> submit(Callable<T> job) {
        if (!isReady()) {
            return Optional.empty();
        }
        FutureTask<T> task = new FutureTask<>(job);
        if (!this.jobs.offer(task)) {
            return Optional.empty();
        }
        try {
            return Optional.ofNullable(task.get());
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
