package tqk114514.terracuda.config;

/**
 * Runtime switches, read from system properties.
 *
 * <p>Two modes, and the difference between them is whether the mod is allowed to touch the chunk:
 *
 * <ul>
 *   <li><b>takeover</b> — {@code -Dterracuda.gpu=true}. The device path runs and its blocks are
 *       written into the chunk; vanilla's own NOISE stage is cancelled. This is the real thing.</li>
 *   <li><b>shadow</b> — {@code -Dterracuda.gpu=true -Dterracuda.shadow=true}. The device path runs and
 *       the result is thrown away. Terrain is exactly vanilla's, so it is the way to check a change
 *       against a live world without risking one.</li>
 * </ul>
 *
 * <p>Shadow mode wins when both are set: it is the safer of the two, and asking for it should never
 * accidentally turn generation over to the device.
 */
public final class TerracudaConfig {

    private static final String PREFIX = "terracuda.";

    private TerracudaConfig() {
    }

    /** Master switch for the GPU world-generation path. */
    public static boolean gpuEnabled() {
        return Boolean.getBoolean(PREFIX + "gpu");
    }

    /**
     * Compute each chunk's blocks on the device without changing what the game generates.
     *
     * <p>This is the design doc's parity mode in its cheapest form: it proves the whole chain — density
     * function lowering, module load, upload, launch, the material rules — works inside a real world,
     * with no risk of wrong terrain.
     */
    public static boolean shadowMode() {
        return Boolean.getBoolean(PREFIX + "shadow");
    }

    /** Whether the device path should build the chunk itself, rather than only observing. */
    public static boolean takeover() {
        return gpuEnabled() && !shadowMode();
    }

    /** Log a line for every chunk rather than every 256. */
    public static boolean verbose() {
        return Boolean.getBoolean(PREFIX + "verbose");
    }

    /**
     * Time the GPU thread and split a chunk's service time into its two halves.
     *
     * <p>This exists because the mod's own service time was inferred from a rate rather than
     * measured, and the inference disagreed with the offline profile: 52 chunks/s implies about
     * 19 ms per chunk, while {@code ChunkPassProfileTest} measures 6.7 ms. The gap is presumably
     * the chunk write-back, which the profile does not cover because it was taken in shadow mode.
     * Measuring where the thread's time actually goes is cheaper than arguing about it.
     */
    public static boolean timing() {
        return Boolean.getBoolean(PREFIX + "timing");
    }

    /**
     * Moves the write-isolated generation stages off vanilla's serial stage dispatcher.
     *
     * <p>Only the radius-zero stages move (NOISE, SURFACE, CARVERS): each writes nothing but its
     * own chunk, and the pyramid's own dependency semantics keep two of them off each other —
     * a chunk in a radius-zero stage cannot have a neighbour in a stage that writes it. The
     * stage's status bookkeeping, profiling hook and completion chain stay exactly vanilla on
     * the dispatcher; only the body runs on the shared background executor. See
     * {@code docs/26.1.2区块管线深度调研.md} for the safety argument and the measurements.
     */
    public static boolean offload() {
        return Boolean.getBoolean(PREFIX + "offload");
    }

    /**
     * Whether FEATURES runs on its dedicated single thread. Defaults to {@link #offload()}.
     *
     * <p>FEATURES writes its 3×3 neighbourhood, so two of them may never overlap — on the
     * vanilla serial dispatcher that holds by construction, and the one dedicated thread is the
     * replacement guarantee once the other stages have left the dispatcher. A single thread
     * also measured faster than a gate-policed pool: the write-area gate's polling ate the
     * parallelism it bought. {@code -Dterracuda.featuresOffload=false} keeps FEATURES on the
     * dispatcher as a diagnostic.
     */
    public static boolean featuresOffload() {
        String value = System.getProperty(PREFIX + "featuresOffload");
        return value == null ? offload() : Boolean.parseBoolean(value);
    }

    /**
     * How many rules-and-write-back workers run the CPU half of a chunk. The router interpreters
     * carry per-chunk state, so each worker owns a set.
     *
     * <p>The default of two comes from the walls on either side, not from taste: one worker drains
     * the ~7 ms rules+write-back at ~145 chunks/s and leaves the multi-stream device thread
     * waiting, while two clear it and hand the ceiling to the serial dispatcher's remaining stages
     * — a third would have nothing to pick up until that dispatcher is dealt with.
     */
    public static int rulesWorkers() {
        return Integer.getInteger(PREFIX + "rulesWorkers", 2);
    }

    public static String summary() {
        return "gpu=" + gpuEnabled() + ", shadow=" + shadowMode() + ", verbose=" + verbose()
                + ", timing=" + timing() + ", offload=" + offload()
                + ", rulesWorkers=" + rulesWorkers();
    }
}
