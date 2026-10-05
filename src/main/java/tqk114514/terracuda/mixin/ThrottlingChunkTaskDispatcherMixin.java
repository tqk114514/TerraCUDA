package tqk114514.terracuda.mixin;

import java.util.concurrent.Executor;

import net.minecraft.server.level.ThrottlingChunkTaskDispatcher;
import net.minecraft.util.thread.TaskScheduler;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import tqk114514.terracuda.worldgen.TicketThrottle;

/**
 * Registers the player-ticket throttle with {@link TicketThrottle}, which reports its in-flight set
 * alongside the periodic timing windows.
 *
 * <p>Non-required: a future Minecraft renaming or reshaping this class degrades to a missing
 * diagnostic, never to a crash.
 */
@Mixin(ThrottlingChunkTaskDispatcher.class)
public abstract class ThrottlingChunkTaskDispatcherMixin {

    @Inject(method = "<init>", at = @At("TAIL"), require = 0)
    private void terracuda$observe(TaskScheduler<Runnable> executor, Executor dispatcherExecutor,
            int maxChunksInExecution, CallbackInfo ci) {
        TicketThrottle.register((ThrottlingChunkTaskDispatcher) (Object) this);
    }
}
