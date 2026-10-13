package tqk114514.terracuda.mixin;

import net.minecraft.server.level.GenerationChunkHolder;
import net.minecraft.world.level.chunk.status.ChunkStatus;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import net.minecraft.server.level.ChunkGenerationTask;

import tqk114514.terracuda.worldgen.LayerTiming;

/**
 * The probes for {@link LayerTiming}: per-layer schedule-to-complete attribution, straight out
 * of the task's own advancement loop.
 *
 * <p>The task runs one lane at a time — a continuation is only ever queued after the previous
 * one finished, and lanes hand the task over through executor queues that carry the
 * happens-before — so per-task probe state needs no synchronization of its own. The layer's
 * first step is seen in {@code scheduleChunkInLayer}; the layer is declared drained when the
 * next layer's first step arrives (for a synchronous layer that is the same lane pass, so the
 * wall is the body's own time) or when the claim is released (the last layer). A layer whose
 * completion parks the task contributes its requeue hop to the wall, and records that hop
 * separately as the park-to-continuation time.
 *
 * <p>Non-required: a renamed or reshaped {@code ChunkGenerationTask} degrades to a missing
 * diagnostic, never to a crash. Gated by {@code -Ptiming}; inert without it.
 */
@Mixin(ChunkGenerationTask.class)
public abstract class ChunkGenerationTaskMixin {

    @Unique
    private ChunkStatus terracuda$layerStatus;

    @Unique
    private long terracuda$layerFirstStepNanos;

    @Unique
    private int terracuda$layerSteps;

    @Unique
    private long terracuda$taskFirstStepNanos;

    @Unique
    private long terracuda$parkedNanos;

    /**
     * The first step of each new layer flushes the previous one; the special double-EMPTY pass
     * (a load attempt that discovers it must generate) naturally produces two EMPTY layers.
     */
    @Inject(method = "scheduleChunkInLayer", at = @At("HEAD"), require = 0)
    private void terracuda$noteStep(ChunkStatus status, boolean needsGeneration,
            GenerationChunkHolder chunkHolder, CallbackInfoReturnable<Boolean> cir) {
        if (!LayerTiming.enabled()) {
            return;
        }
        long now = System.nanoTime();
        if (this.terracuda$layerStatus != status) {
            this.terracuda$flushLayer(now);
            if (this.terracuda$taskFirstStepNanos == 0L) {
                this.terracuda$taskFirstStepNanos = now;
            }
            this.terracuda$layerStatus = status;
            this.terracuda$layerFirstStepNanos = now;
            this.terracuda$layerSteps = 0;
        }
        this.terracuda$layerSteps++;
    }

    /** Landing here with a park pending is the continuation round trip the lanes target. */
    @Inject(method = "runUntilWait", at = @At("HEAD"), require = 0)
    private void terracuda$noteRequeue(CallbackInfoReturnable<Object> cir) {
        if (!LayerTiming.enabled()) {
            return;
        }
        if (this.terracuda$parkedNanos != 0L) {
            if (this.terracuda$layerStatus != null) {
                LayerTiming.recordRequeue(this.terracuda$layerStatus, System.nanoTime() - this.terracuda$parkedNanos);
            }
            this.terracuda$parkedNanos = 0L;
        }
    }

    /** Returning a future means the task parks on this layer's completion. */
    @Inject(method = "runUntilWait", at = @At("RETURN"), require = 0)
    private void terracuda$notePark(CallbackInfoReturnable<Object> cir) {
        if (!LayerTiming.enabled()) {
            return;
        }
        if (cir.getReturnValue() != null) {
            this.terracuda$parkedNanos = System.nanoTime();
        }
    }

    /** The task ends: flush the last layer and record the dwell, cancellations included. */
    @Inject(method = "releaseClaim", at = @At("HEAD"), require = 0)
    private void terracuda$recordDwellAndFlush(CallbackInfo ci) {
        if (!LayerTiming.enabled()) {
            return;
        }
        long now = System.nanoTime();
        this.terracuda$flushLayer(now);
        if (this.terracuda$taskFirstStepNanos != 0L) {
            LayerTiming.recordDwell(now - this.terracuda$taskFirstStepNanos);
            this.terracuda$taskFirstStepNanos = 0L;
        }
        this.terracuda$parkedNanos = 0L;
    }

    @Unique
    private void terracuda$flushLayer(long nowNanos) {
        ChunkStatus status = this.terracuda$layerStatus;
        if (status == null) {
            return;
        }
        LayerTiming.recordSteps(status, this.terracuda$layerSteps);
        if (this.terracuda$layerFirstStepNanos != 0L) {
            LayerTiming.recordWall(status, nowNanos - this.terracuda$layerFirstStepNanos);
        }
        this.terracuda$layerStatus = null;
        this.terracuda$layerFirstStepNanos = 0L;
        this.terracuda$layerSteps = 0;
    }
}
