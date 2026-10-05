package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.server.level.ThrottlingChunkTaskDispatcher;
import net.minecraft.util.thread.TaskScheduler;
import net.minecraft.world.level.ChunkPos;
import org.junit.jupiter.api.Test;

/**
 * The throttle observer, including the part that proves the mixin is live.
 *
 * <p>Constructing a dispatcher must register it: in a real game the constructor injection is what
 * registers it, and this JVM applies the same mixins, so an empty {@link TicketThrottle#describe()}
 * while work is in flight means the mixin never bound — not that nothing is busy. The visible/idle
 * split has teeth too: a dispatched chunk must show up, and a released dispatcher must not, or the
 * periodic report would drown in idle lines from the dimensions nobody is loading.
 */
class TicketThrottleTest {

    @Test
    void aBusyDispatcherShowsUpAndAnIdleOneIsFilteredOut() {
        TaskScheduler<Runnable> scheduler = TaskScheduler.wrapExecutor("test throttler", Runnable::run);
        ThrottlingChunkTaskDispatcher dispatcher =
                new ThrottlingChunkTaskDispatcher(scheduler, Runnable::run, 4);

        long pos = new ChunkPos(1, 2).pack();
        dispatcher.submit(() -> { }, pos, () -> 0);

        String busy = TicketThrottle.describe();
        assertTrue(busy.contains("[1, 2]"),
                "a dispatcher holding a chunk in flight must show up: '" + busy + "'");
        assertTrue(busy.contains("test throttler"),
                "the status names its scheduler: '" + busy + "'");

        dispatcher.release(pos, () -> { }, false);
        String idle = TicketThrottle.describe();
        assertTrue(idle.isEmpty(), "an idle dispatcher is noise and must be filtered out: '" + idle + "'");
    }

    @Test
    void limitInFlightLeavesVanillaAloneWhenUnset() {
        assertEquals(4, TicketThrottle.limitInFlight(4),
                "without -Dterracuda.maxInFlight the vanilla limit must pass through untouched");
    }
}
