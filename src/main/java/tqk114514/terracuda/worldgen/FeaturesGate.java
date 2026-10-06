package tqk114514.terracuda.worldgen;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import net.minecraft.world.level.ChunkPos;

/**
 * The write-area gate for FEATURES, the one stage with a write radius of one.
 *
 * <p>FEATURES places features that can cross chunk borders — a tree, a village piece, a mineshaft
 * entrance — so vanilla serialises every FEATURES stage on the dispatcher thread. Two FEATURES
 * stages whose write areas do not overlap are safe to run in parallel, and on a large pregeneration
 * those pairs are plentiful: the gate is an optimistic per-position lock over the 3×3 neighbourhood
 * the stage may write. When all nine positions are free, the stage offloads to the background
 * executor and holds the positions until its body completes; when any is held, the stage runs
 * serially on the dispatcher, which is the behaviour that was there before.
 *
 * <p>The lock is try-acquire: the dispatcher never blocks on a neighbour, it just does the work
 * itself. That trades some parallelism for never stalling the pipeline on a lock — the same
 * reasoning as the replay queue's blocking put rather than an unbounded one, and the measurement
 * that decides whether the trade was right is the throughput.
 */
public final class FeaturesGate {

    /** One holder flag per chunk position the gate has seen, keyed by the position's packed long. */
    private static final ConcurrentHashMap<Long, AtomicInteger> HELD = new ConcurrentHashMap<>();

    private FeaturesGate() {
    }

    /**
     * Tries to reserve every position in the 3×3 neighbourhood of {@code center}.
     *
     * @return {@code true} if all nine were reserved and the caller must call
     *         {@link #release(ChunkPos)} when the stage body completes; {@code false} if any
     *         position was held by another FEATURES stage, in which case nothing was reserved.
     */
    public static boolean tryAcquire(ChunkPos center) {
        int centerX = center.x();
        int centerZ = center.z();
        List<Long> positions = new java.util.ArrayList<>(9);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                positions.add(pack(centerX + dx, centerZ + dz));
            }
        }
        // Canonical order: every caller visits the positions in the same order, so two
        // overlapping tryAcquires cannot interleave their reservations into a partial state.
        positions.sort(Long::compare);

        List<Long> reserved = new java.util.ArrayList<>(9);
        for (long position : positions) {
            AtomicInteger flag = HELD.computeIfAbsent(position, key -> new AtomicInteger(0));
            if (!flag.compareAndSet(0, 1)) {
                for (long held : reserved) {
                    HELD.get(held).set(0);
                }
                return false;
            }
            reserved.add(position);
        }
        return true;
    }

    /** Releases the positions {@link #tryAcquire(ChunkPos)} reserved. */
    public static void release(ChunkPos center) {
        int centerX = center.x();
        int centerZ = center.z();
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                AtomicInteger flag = HELD.get(pack(centerX + dx, centerZ + dz));
                if (flag != null) {
                    flag.set(0);
                }
            }
        }
    }

    /** Packs chunk coordinates into the long key the map is indexed by. */
    private static long pack(int x, int z) {
        return (long) x & 0xFFFFFFFFL | ((long) z & 0xFFFFFFFFL) << 32;
    }
}
