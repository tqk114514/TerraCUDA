package tqk114514.terracuda.mixin;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;

import net.minecraft.server.level.GenerationChunkHolder;
import net.minecraft.util.StaticCache2D;
import net.minecraft.util.Util;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.status.ChunkStatusTask;
import net.minecraft.world.level.chunk.status.ChunkStep;
import net.minecraft.world.level.chunk.status.WorldGenContext;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;
import tqk114514.terracuda.worldgen.FeaturesGate;

/**
 * Moves the write-isolated generation stages off the serial stage dispatcher, behind
 * {@code -Dterracuda.offload}.
 *
 * <p>The experiment matrix in {@code docs/性能笔记.md} settled that this dispatcher is the wall
 * every configuration bounced off. Two kinds of stage are safe to move:
 *
 * <ul>
 *   <li><b>Write radius zero</b> (NOISE, SURFACE, CARVERS): the stage writes only its own chunk.
 *       Safety follows from the pyramid's own dependency semantics — a chunk in a radius-zero
 *       stage cannot have a neighbour in FEATURES, because FEATURES requires CARVERS at radius 1.
 *       These move unconditionally.</li>
 *   <li><b>Write radius one</b> (FEATURES): the stage may write to its 3×3 neighbourhood via
 *       features that cross chunk borders. Two FEATURES stages whose neighbourhoods do not
 *       overlap are safe to run in parallel; {@link FeaturesGate} holds the neighbourhood while
 *       the body runs and falls back to the serial dispatcher when a neighbour's FEATURES is
 *       already in flight. A large pregeneration has plenty of non-adjacent pairs.</li>
 * </ul>
 *
 * <p>The rest of the stages cost microseconds (STRUCTURE_STARTS 0.6 ms, BIOMES 67 µs, LIGHT,
 * SPAWN, FULL) and stay on the dispatcher; LIGHT is not thread-safe without the kind of audit
 * C2ME's {@code fixes-threading} modules represent, so it does not move either.
 *
 * <p>The wrap is on the {@code doWork} call rather than a rewrite of {@code apply}, so the status
 * bookkeeping, the profiling hook and the completion chain stay exactly vanilla on the dispatcher;
 * only the body of the stage moves. Off by default, non-required, inert without the flag.
 */
@Mixin(ChunkStep.class)
public abstract class ChunkStepMixin {

    /** Logged once per kind, so a run shows which stages moved or that none did. */
    private static final AtomicBoolean REPORTED_RADIUS_ZERO = new AtomicBoolean();
    private static final AtomicBoolean REPORTED_FEATURES = new AtomicBoolean();

    /**
     * The dedicated FEATURES executor, deliberately two threads behind a blocking gate.
     *
     * <p>One thread ran at 77% utilization at the measured 180 cps and was the only single-thread
     * left in the pipeline. Two threads double the ceiling to ~466/s — more than the device
     * thread's ~285/s — and the write-area gate keeps them off each other's chunks: each stage
     * blocking-acquires the 3×3 neighbourhood before running, so adjacent FEATURES stages
     * queue behind each other rather than racing on section semaphores. The deadlock that killed
     * the first attempt was the serial fallback running on the dispatcher against the dedicated
     * thread; with both stages on gate-holding executor threads, one waits and the other
     * finishes — no cycle.
     */
    private static final java.util.concurrent.ExecutorService FEATURES_EXECUTOR =
            java.util.concurrent.Executors.newSingleThreadExecutor(runnable -> {
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
                TerraCUDA.LOGGER.info("TerraCUDA: moving radius-zero stages off the serial "
                        + "dispatcher ({})", step.targetStatus());
            }
            // supplyAsync wraps doWork's own future in another future; thenCompose flattens the two
            // so the caller still sees the stage's completion, exactly as it would synchronously.
            return CompletableFuture.supplyAsync(() -> original.call(task, context, step, cache, chunk),
                            Util.backgroundExecutor())
                    .thenCompose(future -> future);
        }

        if (radius == 1 && step.targetStatus() == net.minecraft.world.level.chunk.status.ChunkStatus.FEATURES
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
