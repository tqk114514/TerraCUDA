package tqk114514.terracuda.cuda;

/**
 * Immutable description of one CUDA device, as reported by the driver.
 *
 * @param index                zero-based ordinal used to address the device in later driver calls
 * @param name                 marketing name, e.g. {@code NVIDIA GeForce RTX 3080}
 * @param computeMajor         compute capability major version
 * @param computeMinor         compute capability minor version
 * @param totalMemoryBytes     total device memory in bytes
 * @param multiprocessorCount  number of streaming multiprocessors
 */
public record CudaDeviceInfo(
        int index,
        String name,
        int computeMajor,
        int computeMinor,
        long totalMemoryBytes,
        int multiprocessorCount) {

    /** Compute capability packed as {@code major*10 + minor}, e.g. {@code 86} for 8.6. */
    public int computeCapability() {
        return computeMajor * 10 + computeMinor;
    }

    /** Compute capability rendered as {@code "8.6"}. */
    public String computeCapabilityString() {
        return computeMajor + "." + computeMinor;
    }

    public long totalMemoryMiB() {
        return totalMemoryBytes / (1024L * 1024L);
    }
}
