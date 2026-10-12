package tqk114514.terracuda.mixin;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import net.minecraft.server.level.ChunkTaskDispatcher;
import net.minecraft.server.level.ChunkTaskPriorityQueue;
import net.minecraft.util.Util;
import net.minecraft.util.thread.PriorityConsecutiveExecutor;
import net.minecraft.util.thread.StrictQueue;
import net.minecraft.util.thread.TaskScheduler;

import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;

/**
 * Runs the worldgen dispatcher's per-chunk stage-advancement steps on {@code -Dterracuda.parallelSched=N}
 * parallel lanes instead of vanilla's single {@code ConsecutiveExecutor}.
 *
 * <p>What is serial in vanilla, and why it is the wall. A chunk moves towards its target status
 * one layer at a time: {@code ChunkGenerationTask.runUntilWait} schedules a layer of
 * {@code GenerationChunkHolder.applyStep} calls, returns the layer's pending future, and the
 * continuation is re-submitted when that future completes. Each such step is fast — with the
 * stage bodies offloaded it is only dependency checks and submissions — but every step of every
 * chunk passes through two single-lane executors: the priority bookkeeping executor that owns the
 * {@code ChunkTaskPriorityQueue}, and the {@code ConsecutiveExecutor("worldgen")} that runs the
 * steps themselves, one chunk's batch at a time. Twelve-odd steps per chunk at ~200 chunks/s is
 * ~2,400 lane handoffs a second through a pool shared with the stage work. That is the ripple
 * advancement the performance notes end on: every component below 90% busy, throughput pinned
 * by how fast chunks can advance stages.
 *
 * <p>What this changes. The bookkeeping lane is kept exactly vanilla — {@code submit}, {@code
 * release}, {@code resort} and the queue {@code pop} all stay on the priority executor's single
 * thread, because {@code ChunkTaskPriorityQueue} is a plain fastutil map with no internal
 * synchronisation. Only the popped batches move: instead of handing the chunk's task list to the
 * {@code ConsecutiveExecutor} one item at a time and polling the next chunk only after the whole
 * batch completes, the batch runs as one job on a shared fixed pool, and up to {@code N} batches
 * are in flight before the drain loop stops popping. A completing batch re-enters the drain loop,
 * so the lanes stay full whenever work is queued.
 *
 * <p>Why that is safe. Per chunk, ordering cannot change: a chunk's next continuation is queued
 * only after its current step returned and its layer future completed, so a chunk never has two
 * steps anywhere in the pipeline — the batch just runs the (usually one) queued continuations
 * sequentially, as vanilla does. Across chunks, the state the steps touch is already built for
 * concurrent probing: {@code GenerationChunkHolder}'s status bump is an atomic compare-and-set
 * (the loser gets the winner's future), its futures live in an {@code AtomicReferenceArray},
 * generation refcounts are atomic, the profiler is thread-local, and chunk IO is enqueued through
 * its own worker. Vanilla relies on the single lane for memory consistency inside one task, never
 * across tasks — that is what the atomics are for.
 *
 * <p>Scope. Only the dispatcher wired to the {@code "worldgen"} executor is parallelised; the
 * light dispatcher and the player-ticket throttle keep vanilla's single lane, and so does every
 * dispatcher when the flag is off. Requires {@code -Dterracuda.offload} (the stage bodies must not
 * run on the lanes without the {@code ThreadingDetector} bypass), and it forces FEATURES onto
 * its dedicated single thread via {@link TerracudaConfig#featuresDedicated()} — two adjacent
 * chunks' FEATURES stages must never overlap, and on parallel lanes only the dedicated thread
 * guarantees that.
 */
@Mixin(ChunkTaskDispatcher.class)
public abstract class ChunkTaskDispatcherMixin {

    @Shadow
    @Final
    private TaskScheduler<Runnable> executor;

    @Shadow
    @Final
    private PriorityConsecutiveExecutor dispatcher;

    @Shadow
    protected boolean sleeping;

    @Shadow
    protected abstract void pollTask();

    @Shadow
    protected abstract ChunkTaskPriorityQueue.TasksForChunk popTasks();

    @Shadow
    protected abstract void scheduleForExecution(ChunkTaskPriorityQueue.TasksForChunk tasksForChunk);

    /** Logged once per run, so a run shows the scheduler moved or that it did not. */
    private static final AtomicBoolean REPORTED = new AtomicBoolean();
    private static final AtomicBoolean WARNED_NO_OFFLOAD = new AtomicBoolean();

    @Unique
    private static volatile ExecutorService terracuda$schedulerPool;

    /** Batches this dispatcher currently has on the lanes; the drain loop keeps it under the cap. */
    @Unique
    private final AtomicInteger terracuda$inFlight = new AtomicInteger();

    /** Cached per instance: the name check and the config reads should not sit on the hot path. */
    @Unique
    private Boolean terracuda$parallelReady;

    @Unique
    private boolean terracuda$parallel() {
        Boolean ready = this.terracuda$parallelReady;
        if (ready == null) {
            ready = TerracudaConfig.parallelSched() > 0 && "worldgen".equals(this.executor.name());
            if (ready && !TerracudaConfig.offload()) {
                if (WARNED_NO_OFFLOAD.compareAndSet(false, true)) {
                    TerraCUDA.LOGGER.warn("TerraCUDA: -Dterracuda.parallelSched needs "
                            + "-Dterracuda.offload (the stage bodies must not run on the lanes); "
                            + "staying serial");
                }
                ready = false;
            }
            if (ready && REPORTED.compareAndSet(false, true)) {
                TerraCUDA.LOGGER.info("TerraCUDA: {} parallel stage-scheduler lanes on the "
                        + "worldgen dispatcher ({})", TerracudaConfig.parallelSched(),
                        TerracudaConfig.summary());
            }
            this.terracuda$parallelReady = ready;
        }
        return ready;
    }

    /**
     * Replaces the one-batch-at-a-time poll with a drain loop: keep popping chunk batches onto
     * the lanes until {@code N} are in flight or the queue is empty. Runs on the bookkeeping
     * lane, so the queue need not be synchronised. When the cap stops the loop the queue is left
     * unpopped and {@code sleeping} untouched — every in-flight batch re-enters this loop when
     * it finishes, so the work is picked up then.
     */
    @Inject(method = "pollTask", at = @At("HEAD"), cancellable = true)
    private void terracuda$drainAcrossLanes(CallbackInfo ci) {
        if (!this.terracuda$parallel()) {
            return;
        }
        this.dispatcher.schedule(new StrictQueue.RunnableWithPriority(3, this::terracuda$drain));
        ci.cancel();
    }

    @Unique
    private void terracuda$drain() {
        int lanes = TerracudaConfig.parallelSched();
        while (this.terracuda$inFlight.get() < lanes) {
            ChunkTaskPriorityQueue.TasksForChunk tasks = this.popTasks();
            if (tasks == null) {
                this.sleeping = true;
                return;
            }

            this.scheduleForExecution(tasks);
        }
    }

    /**
     * Runs one chunk's batch on a lane as a single sequential job, then re-enters the drain loop.
     *
     * <p>The vanilla body schedules each task through the worldgen {@code ConsecutiveExecutor} and
     * polls the next chunk only after the last future completes; this submits the batch to the
     * shared pool and returns at once, so the cap in the drain loop — not this method — decides
     * how many chunks advance at once. The {@code finally} is load-bearing: vanilla stalls its
     * whole dispatcher if a step throws (the completing future never fires), while a lane must
     * always release its slot and re-poll, or every later chunk would queue behind one dead step.
     */
    @Inject(method = "scheduleForExecution", at = @At("HEAD"), cancellable = true)
    private void terracuda$runBatchOnLane(ChunkTaskPriorityQueue.TasksForChunk tasksForChunk, CallbackInfo ci) {
        if (!this.terracuda$parallel()) {
            return;
        }

        this.terracuda$inFlight.incrementAndGet();
        terracuda$schedulerPool().execute(() -> {
            try {
                for (Runnable task : tasksForChunk.tasks()) {
                    Util.runNamed(task, "worldgen");
                }
            } finally {
                this.terracuda$inFlight.decrementAndGet();
                this.pollTask();
            }
        });
        ci.cancel();
    }

    /** A batch on a lane is work in progress, even when the queues are momentarily empty. */
    @Inject(method = "hasWork", at = @At("HEAD"), cancellable = true, require = 0)
    private void terracuda$lanesCountAsWork(CallbackInfoReturnable<Boolean> ci) {
        if (this.terracuda$parallel() && this.terracuda$inFlight.get() > 0) {
            ci.setReturnValue(true);
        }
    }

    @Unique
    private static ExecutorService terracuda$schedulerPool() {
        ExecutorService pool = terracuda$schedulerPool;
        if (pool == null) {
            synchronized (ChunkTaskDispatcherMixin.class) {
                pool = terracuda$schedulerPool;
                if (pool == null) {
                    AtomicInteger laneCounter = new AtomicInteger();
                    int lanes = TerracudaConfig.parallelSched();
                    pool = Executors.newFixedThreadPool(lanes, runnable -> {
                        Thread thread = new Thread(runnable, "TerraCUDA-sched-" + laneCounter.getAndIncrement());
                        thread.setDaemon(true);
                        return thread;
                    });
                    terracuda$schedulerPool = pool;
                }
            }
        }
        return pool;
    }
}
