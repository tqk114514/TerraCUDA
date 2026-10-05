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
     * Raises vanilla's player-ticket in-flight limit above its default of four. Zero leaves vanilla
     * alone.
     *
     * <p>Vanilla frees a throttle slot only when a chunk reaches entity-ticking status, so loading
     * runs at four divided by the per-chunk dwell — the ~65 chunks/s ceiling every measurement has
     * bounced off. Raising it admits more chunks into the loading pipeline at once, which is an
     * experiment, not a speedup: the serial stage dispatcher and the device path have ceilings of
     * their own, and finding which one binds next is the point. See {@code docs/性能笔记.md}.
     */
    public static int maxInFlight() {
        return Integer.getInteger(PREFIX + "maxInFlight", 0);
    }

    public static String summary() {
        return "gpu=" + gpuEnabled() + ", shadow=" + shadowMode() + ", verbose=" + verbose()
                + ", timing=" + timing() + ", maxInFlight=" + maxInFlight();
    }
}
