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

    public static String summary() {
        return "gpu=" + gpuEnabled() + ", shadow=" + shadowMode() + ", verbose=" + verbose();
    }
}
