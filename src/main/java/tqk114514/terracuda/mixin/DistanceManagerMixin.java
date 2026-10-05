package tqk114514.terracuda.mixin;

import net.minecraft.server.level.DistanceManager;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;
import tqk114514.terracuda.worldgen.TicketThrottle;

/**
 * The experiment switch for the player-ticket throttle.
 *
 * <p>Vanilla admits four chunks to the loading pipeline at once and frees a slot only when a chunk
 * reaches entity-ticking status, so the ceiling on chunk loading is four divided by the per-chunk
 * dwell — about 65 a second for the ~62 ms dwell the timing windows measured. Everything upstream of
 * generation has been exonerated by measurement; this is the one knob vanilla exposes for it.
 *
 * <p>Off by default: without {@code -Dterracuda.maxInFlight} the argument passes through untouched.
 * Raising it is an experiment, not a speedup — the serial stage dispatcher and this mod's own threads
 * have their own ceilings, and the point of raising the limit is to find which one binds next.
 */
@Mixin(DistanceManager.class)
public abstract class DistanceManagerMixin {

    @ModifyArg(method = "<init>",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/server/level/ThrottlingChunkTaskDispatcher;"
                    + "<init>(Lnet/minecraft/util/thread/TaskScheduler;Ljava/util/concurrent/Executor;I)V"),
            require = 0)
    private int terracuda$maxChunksInExecution(int vanilla) {
        int configured = TicketThrottle.limitInFlight(vanilla);
        if (configured != vanilla) {
            TerraCUDA.LOGGER.info("TerraCUDA: player-ticket in-flight limit raised from {} to {} "
                    + "(experimental; the vanilla default of 4 keeps at most {} chunks moving "
                    + "towards loaded at once)", vanilla, configured, vanilla);
        }
        return configured;
    }
}
