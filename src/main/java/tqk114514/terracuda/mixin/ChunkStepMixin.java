package tqk114514.terracuda.mixin;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;

import net.minecraft.server.level.GenerationChunkHolder;
import net.minecraft.util.StaticCache2D;
import net.minecraft.util.Util;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.chunk.status.ChunkStatusTask;
import net.minecraft.world.level.chunk.status.ChunkStep;
import net.minecraft.world.level.chunk.status.WorldGenContext;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;

/**
 * Moves the write-isolated generation stages off the serial stage dispatcher, behind
 * {@code -Dterracuda.offload}.
 *
 * <p>Only the stages the pyramid itself declares safe move, per
 * {@code docs/26.1.2区块管线深度调研.md}:
 *
 * <ul>
 *   <li><b>Write radius zero</b> (NOISE, SURFACE, CARVERS): the stage writes only its own chunk.
 *       Safety follows from the pyramid's own dependency semantics — a chunk in a radius-zero
 *       stage cannot have a neighbour in FEATURES, because FEATURES requires CARVERS at
 *       radius 1. These move to the shared background executor.</li>
 *   <li><b>Write radius one</b> (FEATURES): the stage writes its 3×3 neighbourhood, so two of
 *       them may never overlap. It moves to a dedicated single thread — the one-thread
 *       serialization is the whole guarantee, and it also measured faster than a gate-policed
 *       pool, whose polling ate the parallelism it bought. The remaining stages stay on the
 *       dispatcher: LIGHT because the light engine's internals have no thread-safety audit,
 *       and the structure/spawn steps because moving them was measured end-to-end neutral
 *       while putting structure-generator shared state on the concurrent path.</li>
 * </ul>
 *
 * <p>The wrap is on the {@code doWork} call rather than a rewrite of {@code apply}, so the
 * status bookkeeping, the profiling hook and the completion chain stay exactly vanilla on the
 * dispatcher; only the body of the stage moves. Off by default, non-required, inert without
 * the flag.
 */
@Mixin(ChunkStep.class)
public abstract class ChunkStepMixin {

    /** Logged once per kind, so a run shows which stages moved or that none did. */
    private static final AtomicBoolean REPORTED_RADIUS_ZERO = new AtomicBoolean();
    private static final AtomicBoolean REPORTED_FEATURES = new AtomicBoolean();

    /**
     * The dedicated FEATURES executor: one thread, deliberately.
     *
     * <p>One thread held 77% busy at the measured ~180 cps of the full offload line and was the
     * last single-threaded stage left in the pipeline; two threads behind a write-area gate
     * measured slower than this one thread, so the thread count is the safety guarantee and
     * not a compromise. A deadlock that killed an earlier attempt was the serial fallback
     * running on the dispatcher against the dedicated thread; this form has no fallback —
     * every FEATURES body runs on the one thread, so there is no cycle and no race.
     */
    private static final ExecutorService FEATURES_EXECUTOR = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "TerraCUDA-features");
        thread.setDaemon(true);
        return thread;
    });

    @WrapOperation(method = "apply",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/world/level/chunk/status/ChunkStatusTask;"
                    + "doWork(Lnet/minecraft/world/level/chunk/status/WorldGenContext;"
                    + "Lnet/minecraft/world/level/chunk/status/ChunkStep;"
                    + "Lnet/minecraft/util/StaticCache2D;"
                    + "Lnet/minecraft/world/level/chunk/ChunkAccess;)"
                    + "Ljava/util/concurrent/CompletableFuture;"),
            require = 0)
    private CompletableFuture<ChunkAccess> terracuda$offloadStages(ChunkStatusTask task,
            WorldGenContext context, ChunkStep step, StaticCache2D<GenerationChunkHolder> cache,
            ChunkAccess chunk, Operation<CompletableFuture<ChunkAccess>> original) {
        if (!TerracudaConfig.offload()) {
            return original.call(task, context, step, cache, chunk);
        }

        int radius = step.blockStateWriteRadius();
        if (radius == 0) {
            if (REPORTED_RADIUS_ZERO.compareAndSet(false, true)) {
                TerraCUDA.LOGGER.info("TerraCUDA: moving radius-0 stages (NOISE, SURFACE, CARVERS) "
                        + "off the serial dispatcher ({})", TerracudaConfig.summary());
            }
            // supplyAsync wraps doWork's own future in another future; thenCompose flattens the two
            // so the caller still sees the stage's completion, exactly as it would synchronously.
            return CompletableFuture.supplyAsync(() -> original.call(task, context, step, cache, chunk),
                            Util.backgroundExecutor())
                    .thenCompose(future -> future);
        }

        if (radius == 1 && step.targetStatus() == ChunkStatus.FEATURES
                && TerracudaConfig.featuresOffload()) {
            if (REPORTED_FEATURES.compareAndSet(false, true)) {
                TerraCUDA.LOGGER.info("TerraCUDA: moving FEATURES to a dedicated thread "
                        + "({})", TerracudaConfig.summary());
            }
            return CompletableFuture.supplyAsync(() -> original.call(task, context, step, cache, chunk),
                            FEATURES_EXECUTOR)
                    .thenCompose(future -> future);
        }

        return original.call(task, context, step, cache, chunk);
    }
}
