package tqk114514.terracuda.config;

/**
 * Runtime switches, read from system properties.
 *
 * <p>Everything defaults to off. The GPU path is not yet complete — the density tables are computed
 * but the chunk is still built by vanilla — so shipping it enabled would only cost startup time for
 * no benefit. {@code -Dterracuda.gpu=true -Dterracuda.shadow=true} turns on the diagnostic path that
 * exercises the device end to end inside a running game.
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
     * Computes each chunk's marker tables on the device without changing what the game generates.
     *
     * <p>This is the design doc's parity mode in its cheapest form: it proves the whole chain — density
     * function lowering, module load, upload, launch — works inside a real world, with no risk of
     * wrong terrain.
     */
    public static boolean shadowMode() {
        return Boolean.getBoolean(PREFIX + "shadow");
    }

    /** Log a line for every chunk rather than every 256. */
    public static boolean verbose() {
        return Boolean.getBoolean(PREFIX + "verbose");
    }

    public static String summary() {
        return "gpu=" + gpuEnabled() + ", shadow=" + shadowMode() + ", verbose=" + verbose();
    }
}
