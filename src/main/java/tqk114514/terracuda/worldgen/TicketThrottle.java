package tqk114514.terracuda.worldgen;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import net.minecraft.server.level.ThrottlingChunkTaskDispatcher;
import tqk114514.terracuda.config.TerracudaConfig;

/**
 * The player-ticket throttle: the one thing upstream that measurements keep pointing at.
 *
 * <p>Vanilla admits at most {@code maxChunksInExecution} chunks — four of them — to be moving from
 * "given a player ticket" to "fully loaded" at once; the slot frees only when the chunk reaches
 * entity-ticking status, so the whole loading pipeline runs at four divided by the per-chunk dwell.
 * That arithmetic lands on ~65 chunks/s for the ~62 ms dwell we measured, which is the ceiling every
 * experiment so far has bounced off. This class is where the mod observes and, experimentally,
 * raises that limit.
 *
 * <p>Observation is via {@link ThrottlingChunkTaskDispatcher#getDebugStatus()}, which is public and
 * {@code @VisibleForTesting}: the in-flight chunk set and the queue's sleeping flag, one line per
 * dispatcher that is doing anything. It is read from a reporting thread while the dispatcher
 * mutates on the main thread, so a read may race — a diagnostic that throws would take the timing
 * report down with it, so it reports what it can and says so when it cannot.
 */
public final class TicketThrottle {

    /**
     * One dispatcher per level: the overworld, the nether and the end each have their own
     * DistanceManager. Idle ones are filtered out of {@link #describe()} rather than named, because
     * their scheduler name ("player ticket throttler") is identical and would only add noise.
     */
    private static final List<ThrottlingChunkTaskDispatcher> DISPATCHERS = new CopyOnWriteArrayList<>();

    private TicketThrottle() {
    }

    /** Called from the dispatcher's constructor; the instance is itself the instrument. */
    public static void register(ThrottlingChunkTaskDispatcher dispatcher) {
        DISPATCHERS.add(dispatcher);
    }

    /**
     * The in-flight limit to use instead of vanilla's four, or zero to leave vanilla alone.
     *
     * <p>Experimental by construction: raising it admits more chunks into the loading pipeline at
     * once, which nothing downstream is guaranteed to absorb. The serial stage dispatcher, the
     * device thread and the rules worker all have their own ceilings, and finding which one the
     * raised throttle runs into next is the point of the experiment.
     */
    public static int limitInFlight(int vanilla) {
        int configured = TerracudaConfig.maxInFlight();
        return configured > 0 ? configured : vanilla;
    }

    /**
     * One line per dispatcher that is doing anything, for the periodic timing report.
     *
     * @return empty when every dispatcher is idle — no chunks in flight and an empty queue.
     */
    public static String describe() {
        StringBuilder out = new StringBuilder(96);
        for (ThrottlingChunkTaskDispatcher dispatcher : DISPATCHERS) {
            String status;
            try {
                status = dispatcher.getDebugStatus();
            } catch (RuntimeException e) {
                // Reading a mutating set from another thread can race; a diagnostic must not throw.
                status = "(unreadable)";
            }
            if (status.endsWith("=[], s=true")) {
                continue; // idle: nothing in flight, queue drained
            }
            if (!out.isEmpty()) {
                out.append(" | ");
            }
            out.append(status);
        }
        return out.toString();
    }
}
