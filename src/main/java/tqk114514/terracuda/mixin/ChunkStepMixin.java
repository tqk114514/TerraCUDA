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

/**
 * Moves the radius-zero generation stages off the serial stage dispatcher, behind
 * {@code -Dterracuda.offload}.
 *
 * <p>The experiment matrix in {@code docs/性能笔记.md} settled that this dispatcher is the wall
 * every configuration bounced off: raising the player-ticket throttle only deepened the queue in
 * front of it. Its per-chunk cost is the stage bodies it applies synchronously, and the pyramid
 * gives exactly three of them a block-state write radius of zero — NOISE, SURFACE and CARVERS.
 * FEATURES writes its neighbours, so it stays; the rest cost microseconds.
 *
 * <p>Safety follows from the pyramid's own dependency semantics rather than from anything this
 * mod guarantees: a radius-zero stage writes only its own chunk, and a chunk only enters FEATURES
 * once every neighbour within radius 1 has finished CARVERS — so while a chunk is in a radius-zero
 * stage, no neighbour can be in FEATURES writing it, and two chunks in radius-zero stages write
 * disjoint chunks. The stage bodies are also the kind of work the dispatcher was already handing
 * to workers one pool hop away: NOISE's own body does exactly that internally.
 *
 * <p>The wrap is on the {@code doWork} call rather than a rewrite of {@code apply}, so the status
 * bookkeeping, the profiling hook and the completion chain stay exactly vanilla on the dispatcher;
 * only the body of the stage moves. Off by default, non-required, inert without the flag.
 */
@Mixin(ChunkStep.class)
public abstract class ChunkStepMixin {

    /** Logged once, so a run either shows the offload happened or shows that it did not. */
    private static final AtomicBoolean REPORTED_FIRST_OFFLOAD = new AtomicBoolean();

    @WrapOperation(method = "apply",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/world/level/chunk/status/ChunkStatusTask;"
                    + "doWork(Lnet/minecraft/world/level/chunk/status/WorldGenContext;"
                    + "Lnet/minecraft/world/level/chunk/status/ChunkStep;"
                    + "Lnet/minecraft/util/StaticCache2D;"
                    + "Lnet/minecraft/world/level/chunk/ChunkAccess;)"
                    + "Ljava/util/concurrent/CompletableFuture;"),
            require = 0)
    private CompletableFuture<ChunkAccess> terracuda$offloadRadiusZero(ChunkStatusTask task,
            WorldGenContext context, ChunkStep step, StaticCache2D<GenerationChunkHolder> cache,
            ChunkAccess chunk, Operation<CompletableFuture<ChunkAccess>> original) {
        if (!TerracudaConfig.offload() || step.blockStateWriteRadius() != 0) {
            return original.call(task, context, step, cache, chunk);
        }
        if (REPORTED_FIRST_OFFLOAD.compareAndSet(false, true)) {
            TerraCUDA.LOGGER.info("TerraCUDA: moving {} stages off the serial dispatcher ({})",
                    step.targetStatus(), TerracudaConfig.summary());
        }
        // supplyAsync wraps doWork's own future in another future; thenCompose flattens the two so
        // the caller still sees the stage's completion, exactly as it would synchronously.
        return CompletableFuture.supplyAsync(() -> original.call(task, context, step, cache, chunk),
                        Util.backgroundExecutor())
                .thenCompose(future -> future);
    }
}
