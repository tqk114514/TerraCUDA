package tqk114514.terracuda.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.KernelArguments;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;

/**
 * GPU-backed evaluation of a lowered density program: the K2 kernel of the design doc.
 *
 * <p>One thread per block position, each running the whole instruction stream. The program is
 * uploaded once as a single read-only blob, so the per-batch cost is the point transfer plus the
 * launch.
 *
 * <p>The per-thread scratch is a global buffer of {@code capacity * instructionCount} doubles rather
 * than local memory: at a few hundred instructions per thread, local memory would be several kilobytes
 * each, and a device-side allocation per thread would be worse still.
 */
public final class DensityEvaluatorGpu implements AutoCloseable {

    /** The exported kernel symbol, matching {@code terracuda.cu}. */
    public static final String KERNEL_NAME = "terracuda_density_evaluate";

    private static final int BLOCK_SIZE = 128;

    private final CudaContext context;
    private final CudaContext.DeviceBuffer blob;
    private final CudaContext.DeviceBuffer offsets;
    private final CudaContext.DeviceBuffer points;
    private final CudaContext.DeviceBuffer scratch;
    private final CudaContext.DeviceBuffer results;
    private final int instructionCount;
    private final int root;
    private final int capacity;
    private boolean closed;

    public DensityEvaluatorGpu(CudaContext context, DensityProgram program, int capacity) {
        this(context, DensityProgramImage.of(program), capacity);
    }

    public DensityEvaluatorGpu(CudaContext context, DensityProgramImage image, int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.context = context;
        this.instructionCount = image.instructionCount();
        this.root = image.root();
        this.capacity = capacity;

        try (Arena arena = Arena.ofConfined()) {
            byte[] blobBytes = image.blob();
            MemorySegment hostBlob = arena.allocate(blobBytes.length);
            MemorySegment.copy(blobBytes, 0, hostBlob, ValueLayout.JAVA_BYTE, 0, blobBytes.length);
            this.blob = context.allocate(blobBytes.length);
            context.copyToDevice(hostBlob, this.blob);

            long[] offsetValues = image.offsets();
            MemorySegment hostOffsets = arena.allocateFrom(ValueLayout.JAVA_LONG, offsetValues);
            this.offsets = context.allocate(hostOffsets.byteSize());
            context.copyToDevice(hostOffsets, this.offsets);
        }

        this.points = context.allocate(3L * capacity * Integer.BYTES);
        this.scratch = context.allocate((long) capacity * this.instructionCount * Double.BYTES);
        this.results = context.allocate((long) capacity * Double.BYTES);
    }

    public int capacity() {
        return this.capacity;
    }

    public int instructionCount() {
        return this.instructionCount;
    }

    /**
     * Evaluates the program for {@code count} block positions.
     *
     * @param blockCoordinates interleaved {@code x, y, z} triples, exactly {@code 3 * count} entries
     * @return one density value per position
     */
    public double[] evaluate(int[] blockCoordinates) {
        if (blockCoordinates.length % 3 != 0) {
            throw new IllegalArgumentException("expected interleaved xyz triples, got "
                    + blockCoordinates.length + " ints");
        }
        int count = blockCoordinates.length / 3;
        if (count > this.capacity) {
            throw new IllegalArgumentException("batch of " + count + " points exceeds the "
                    + this.capacity + "-point capacity");
        }
        if (count == 0) {
            return new double[0];
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostPoints = arena.allocateFrom(ValueLayout.JAVA_INT, blockCoordinates);
            this.context.copyToDevice(hostPoints, this.points);

            try (KernelArguments arguments = new KernelArguments(8)) {
                arguments.addDevicePointer(this.blob.address())
                        .addDevicePointer(this.offsets.address())
                        .addInt(this.instructionCount)
                        .addInt(this.root)
                        .addDevicePointer(this.points.address())
                        .addDevicePointer(this.scratch.address())
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
        this.context.free(this.scratch);
        this.context.free(this.points);
        this.context.free(this.offsets);
        this.context.free(this.blob);
    }
}
