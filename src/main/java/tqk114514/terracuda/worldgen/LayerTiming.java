package tqk114514.terracuda.worldgen;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import net.minecraft.world.level.chunk.status.ChunkStatus;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;

/**
 * Per-layer schedule-to-complete attribution for the generation pipeline, gated by
 * {@code -Ptiming} and reported in windows like the rest of the timing kit.
 *
 * <p>Three measures, all recorded per {@link ChunkStatus} from the generation task's own
 * advancement (see {@code docs/26.1.2区块管线深度调研.md} for the pipeline anatomy):
 *
 * <ul>
 *   <li><b>wall</b> — the task's first step of a layer to the continuation that observed every
 *       future of that layer drained. For a synchronous layer that is the body's own time; for a
 *       layer whose completion parks the task (NOISE, light, FULL) it includes one requeue hop,
 *       which the requeue measure below separates.</li>
 *   <li><b>requeue</b> — when a layer's pending future parks the task, the time from that future
 *       completing to {@code runUntilWait} running again: the dispatcher round trip the parallel
 *       lanes and any event-driven continuation target.</li>
 *   <li><b>steps</b> — how many chunk steps the layer scheduled, the ripple's actual fan-out.</li>
 * </ul>
 *
 * <p>Plus <b>task dwell</b>: the task's first EMPTY step to its claim release — the per-chunk
 * dwell the benchmark cap converts into throughput. A cancellation still records its dwell; the
 * headless benchmark has almost none, and a diagnostic that silently drops the rest of the
 * population would lie about exactly the runs it exists for.
 *
 * <p>Thread model: the hooks run on whichever lane executes the task's continuation, one lane at
 * a time per task — the per-task state is handed between lanes through executor queues, which
 * carry the happens-before. Only this class's aggregation is shared, so only it synchronizes.
 */
public final class LayerTiming {

    /** Cached once: the hooks sit on a hot path and the flag is fixed at JVM start. */
    private static final boolean ENABLED = TerracudaConfig.timing();

    private static final long WINDOW_NANOS = 10_000_000_000L;

    /** A window's worth of samples is thousands; the cap guards a stuck reporter, not a run. */
    private static final int MAX_SAMPLES = 60_000;

    private static final Map<ChunkStatus, Bucket> BUCKETS = new LinkedHashMap<>();
    private static final List<Long> DWELLS = new ArrayList<>();
    private static long windowStartNanos;
    private static long droppedSamples;

    private LayerTiming() {
    }

    public static boolean enabled() {
        return ENABLED;
    }

    private static Bucket bucket(ChunkStatus status) {
        synchronized (BUCKETS) {
            return BUCKETS.computeIfAbsent(status, unused -> new Bucket());
        }
    }

    public static void recordWall(ChunkStatus status, long nanos) {
        Bucket bucket = bucket(status);
        synchronized (bucket) {
            bucket.walls.add(nanos);
        }
        capOrReport();
    }

    public static void recordRequeue(ChunkStatus status, long nanos) {
        Bucket bucket = bucket(status);
        synchronized (bucket) {
            bucket.requeues.add(nanos);
        }
        capOrReport();
    }

    public static void recordSteps(ChunkStatus status, int steps) {
        Bucket bucket = bucket(status);
        synchronized (bucket) {
            bucket.steps += steps;
        }
    }

    public static void recordDwell(long nanos) {
        synchronized (DWELLS) {
            DWELLS.add(nanos);
        }
    }

    /** Starts the window and the reporting thread; called from the first record, once. */
    private static void capOrReport() {
        long now = System.nanoTime();
        synchronized (LayerTiming.class) {
            if (windowStartNanos == 0L) {
                windowStartNanos = now;
                Thread reporter = new Thread(LayerTiming::reportLoop, "TerraCUDA-layer-timing");
                reporter.setDaemon(true);
                reporter.start();
                return;
            }
            if (now - windowStartNanos < WINDOW_NANOS) {
                return;
            }
            int live = 0;
            synchronized (BUCKETS) {
                for (Bucket bucket : BUCKETS.values()) {
                    synchronized (bucket) {
                        live += bucket.walls.size() + bucket.requeues.size();
                    }
                }
            }
            if (live > MAX_SAMPLES) {
                droppedSamples++;
                return;
            }
        }
        report();
    }

    private static void reportLoop() {
        while (true) {
            try {
                Thread.sleep(1_000);
                long now = System.nanoTime();
                synchronized (LayerTiming.class) {
                    if (windowStartNanos != 0L && now - windowStartNanos >= WINDOW_NANOS) {
                        report();
                    }
                }
            } catch (InterruptedException e) {
                return;
            }
        }
    }

    /** Drains one window and prints a line per status that ran, plus the dwell summary. */
    private static void report() {
        synchronized (LayerTiming.class) {
            long windowStart = windowStartNanos;
            windowStartNanos = System.nanoTime();
            if (windowStart == 0L) {
                return;
            }

            Map<ChunkStatus, Bucket> snapshot;
            List<Long> dwells;
            synchronized (BUCKETS) {
                snapshot = new LinkedHashMap<>();
                for (Map.Entry<ChunkStatus, Bucket> entry : BUCKETS.entrySet()) {
                    Bucket drained = new Bucket();
                    synchronized (entry.getValue()) {
                        drained.steps = entry.getValue().steps;
                        drained.walls = entry.getValue().walls;
                        drained.requeues = entry.getValue().requeues;
                        entry.getValue().steps = 0;
                        entry.getValue().walls = new ArrayList<>();
                        entry.getValue().requeues = new ArrayList<>();
                    }
                    if (drained.steps > 0 || !drained.walls.isEmpty()) {
                        snapshot.put(entry.getKey(), drained);
                    }
                }
            }
            synchronized (DWELLS) {
                dwells = new ArrayList<>(DWELLS);
                DWELLS.clear();
            }

            for (Map.Entry<ChunkStatus, Bucket> entry : snapshot.entrySet()) {
                Bucket bucket = entry.getValue();
                StringBuilder message = new StringBuilder(160)
                        .append("TerraCUDA: layer ").append(entry.getKey().getName())
                        .append(" steps ").append(bucket.steps);
                describe(message, " wall ", bucket.walls);
                describe(message, " requeue ", bucket.requeues);
                if (droppedSamples > 0) {
                    message.append(" [").append(droppedSamples).append(" capped windows]");
                    droppedSamples = 0;
                }
                TerraCUDA.LOGGER.info(message.toString());
            }
            if (!dwells.isEmpty()) {
                StringBuilder message = new StringBuilder(96)
                        .append("TerraCUDA: task dwell");
                describe(message, " ", dwells);
                message.append(" (").append(dwells.size()).append(" tasks)");
                TerraCUDA.LOGGER.info(message.toString());
            }
        }
    }

    /** Appends " p50 X.X, p95 Y.Y ms" — the mean tells capacity, the tail tells what a queue feels. */
    private static void describe(StringBuilder message, String label, List<Long> samples) {
        if (samples.isEmpty()) {
            return;
        }
        long[] sorted = new long[samples.size()];
        long sum = 0L;
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = samples.get(i);
            sum += sorted[i];
        }
        java.util.Arrays.sort(sorted);
        message.append(label).append("p50 ").append(String.format("%.1f", percentile(sorted, 50) / 1.0e6))
                .append(", p95 ").append(String.format("%.1f", percentile(sorted, 95) / 1.0e6))
                .append(", mean ").append(String.format("%.1f", sum / 1.0e6 / sorted.length)).append(" ms");
    }

    /** Nearest-rank percentile of a sorted array: the ceil(p · count)-th smallest sample. */
    private static long percentile(long[] sorted, int hundredths) {
        int rank = Math.min(sorted.length, (sorted.length * hundredths + 99) / 100);
        return sorted[rank - 1];
    }

    private static final class Bucket {
        private long steps;
        private List<Long> walls = new ArrayList<>();
        private List<Long> requeues = new ArrayList<>();
    }
}
