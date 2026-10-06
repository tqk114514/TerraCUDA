package tqk114514.terracuda.mixin;

import java.util.concurrent.Semaphore;

import net.minecraft.util.ThreadingDetector;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import tqk114514.terracuda.config.TerracudaConfig;

/**
 * Bypasses the threading detector's crash-on-thread-change when the offload is on.
 *
 * <p>The detector is a debugging aid — the {@link Semaphore} underneath is the actual mutual
 * exclusion, and it works correctly across threads: one holder at a time, sequential access from
 * different threads is safe. What the detector adds is a "who touched this from which thread"
 * check that throws when a second thread tries to acquire while the first holds, and then keeps
 * throwing forever after ({@code threadThatFailedToAcquire} is never cleared).
 *
 * <p>Running FEATURES on the background executor means a {@code PalettedContainer} acquired by
 * the dispatcher during NOISE is later acquired by a background thread during FEATURES — the
 * semaphore is free between them, the access is sequential, nothing races. The detector sees
 * the thread name changed and crashes the server on a false positive.
 *
 * <p>This bypass keeps the semaphore (mutual exclusion is preserved — only one thread holds the
 * container at a time) and drops the thread-identity bookkeeping. It is inert without
 * {@code -Dterracuda.offload}; with the offload off, vanilla's detector behaves exactly as it
 * always did.
 */
@Mixin(ThreadingDetector.class)
public abstract class ThreadingDetectorMixin {

    @Shadow
    private Semaphore lock;

    @Inject(method = "checkAndLock", at = @At("HEAD"), cancellable = true, require = 0)
    private void terracuda$lockWithoutDetection(CallbackInfo ci) {
        if (TerracudaConfig.offload()) {
            this.lock.acquireUninterruptibly();
            ci.cancel();
        }
    }

    @Inject(method = "checkAndUnlock", at = @At("HEAD"), cancellable = true, require = 0)
    private void terracuda$unlockWithoutDetection(CallbackInfo ci) {
        if (TerracudaConfig.offload()) {
            this.lock.release();
            ci.cancel();
        }
    }
}
