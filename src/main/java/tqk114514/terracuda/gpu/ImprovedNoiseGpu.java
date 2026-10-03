package tqk114514.terracuda.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.KernelArguments;
import tqk114514.terracuda.noise.ImprovedNoise;

/**
 * GPU-backed {@link ImprovedNoise}: evaluates the noise for a batch of points on the device.
 *
 * <p>This is the M1 milestone's closed loop — upload an octave's permutation table and origin, run
 * the kernel, read the results back — and the first place where the bit-exactness of the CUDA port can
 * actually be observed. The Java {@link ImprovedNoise} is the reference it is checked against.
 *
 * <p>Buffers are sized once for {@code capacity} points and reused across calls, so the per-batch cost
 * is the two transfers plus the launch, with no allocation.
 *
 * <p>The kernel computes the {@code yScale == 0} form of the noise. See the note in
 * {@code terracuda_noise.cu}.
 */
public final class ImprovedNoiseGpu implements AutoCloseable {

    /** The exported kernel symbol, matching {@code terracuda_noise.cu}. */
    public static final String KERNEL_NAME = "terracuda_improved_noise_points";

    private static final int BLOCK_SIZE = 256;

    private final CudaContext context;
    private final CudaContext.DeviceBuffer permutation;
    private final CudaContext.DeviceBuffer points;
    private final CudaContext.DeviceBuffer results;
    private final int capacity;
    private final double xo;
    private final double yo;
    private final double zo;
    private boolean closed;

    /**
     * @param context  a context with the TerraCUDA module already loaded
     * @param noise    the Java octave whose parameters are mirrored to the device
     * @param capacity the largest number of points a single {@link #evaluate} call may carry
     */
    public ImprovedNoiseGpu(CudaContext context, ImprovedNoise noise, int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.context = context;
        this.capacity = capacity;
        this.xo = noise.xo;
        this.yo = noise.yo;
        this.zo = noise.zo;

        byte[] table = noise.permutation();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_BYTE, table);
            this.permutation = context.allocate(table.length);
            context.copyToDevice(host, this.permutation);
        }
        this.points = context.allocate(3L * capacity * Double.BYTES);
        this.results = context.allocate((long) capacity * Double.BYTES);
    }

    public int capacity() {
        return this.capacity;
    }

    /**
     * Evaluates the noise for {@code count} interleaved {@code (x, y, z)} triples.
     *
     * @param interleavedXyz exactly {@code 3 * count} doubles
     * @return one value per point, in the same order
     */
    public double[] evaluate(double[] interleavedXyz) {
        if (interleavedXyz.length % 3 != 0) {
            throw new IllegalArgumentException("expected interleaved xyz triples, got "
                    + interleavedXyz.length + " doubles");
        }
        int count = interleavedXyz.length / 3;
        if (count > this.capacity) {
            throw new IllegalArgumentException("batch of " + count + " points exceeds the "
                    + this.capacity + "-point capacity");
        }
        if (count == 0) {
            return new double[0];
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostPoints = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, interleavedXyz);
            this.context.copyToDevice(hostPoints, this.points);

            try (KernelArguments arguments = new KernelArguments(7)) {
                arguments.addDevicePointer(this.permutation.address())
                        .addDouble(this.xo)
                        .addDouble(this.yo)
                        .addDouble(this.zo)
                        .addDevicePointer(this.points.address())
                        .addDevicePointer(this.results.address())
                        .addInt(count);
                int grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                this.context.launch(KERNEL_NAME, grid, 1, 1, BLOCK_SIZE, 1, 1, arguments);
            }
            this.context.synchronize();

            MemorySegment hostResults = arena.allocate(ValueLayout.JAVA_DOUBLE, count);
            this.context.copyFromDevice(this.results, hostResults);
            return hostResults.toArray(ValueLayout.JAVA_DOUBLE);
        }
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        if (this.context.isClosed()) {
            return;
        }
        this.context.free(this.results);
        this.context.free(this.points);
        this.context.free(this.permutation);
    }
}
