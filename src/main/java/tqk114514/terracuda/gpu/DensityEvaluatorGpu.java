package tqk114514.terracuda.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tqk114514.terracuda.cuda.CudaContext;
import tqk114514.terracuda.cuda.KernelArguments;
import tqk114514.terracuda.density.DensityProgram;
import tqk114514.terracuda.density.DensityProgramImage;

import net.minecraft.world.level.levelgen.NoiseSettings;

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
    private static final int MIN_CAPACITY = 2048;

    private final CudaContext context;
    private final DensityProgramImage image;
    private final CudaContext.DeviceBuffer blob;
    private final CudaContext.DeviceBuffer offsets;
    private final CudaContext.DeviceBuffer points;
    private final CudaContext.DeviceBuffer scratch;
    private final CudaContext.DeviceBuffer results;
    private CudaContext.DeviceBuffer roots;
    private final int instructionCount;
    private final int root;
    private final int capacity;
    private boolean closed;

    public DensityEvaluatorGpu(CudaContext context, DensityProgram program, int capacity) {
        this(context, DensityProgramImage.of(program), capacity);
    }

    /**
     * Builds an evaluator sized for {@code program}'s chunk-scale work.
     *
     * <p>The capacity is the program's total marker grid, because those grids go out in one launch. A
     * capacity that is too small is not wrong — the grids get split — but it gives up most of the win.
     */
    public static DensityEvaluatorGpu forProgram(CudaContext context, DensityProgram program,
            NoiseSettings settings) {
        DensityProgramImage image = DensityProgramImage.of(program);
        int budget = Math.max(MIN_CAPACITY, ChunkCornerTables.pointBudget(image, settings));
        return new DensityEvaluatorGpu(context, image, budget);
    }

    public DensityEvaluatorGpu(CudaContext context, DensityProgramImage image, int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.context = context;
        this.image = image;
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

    /** The uploaded program, including the marker roots used to build the chunk-level tables. */
    public DensityProgramImage image() {
        return this.image;
    }

    /**
     * Evaluates the program for {@code count} block positions.
     *
     * @param blockCoordinates interleaved {@code x, y, z} triples, exactly {@code 3 * count} entries
     * @return one density value per position
     */
    public double[] evaluate(int[] blockCoordinates) {
        return evaluate(blockCoordinates, this.root, null);
    }

    /**
     * Evaluates a sub-graph of the program, identified by the instruction it is rooted at.
     *
     * <p>Valid for any instruction index, because the image is emitted in post-order: everything an
     * instruction reads sits below it. This is how the {@code flat_cache} and {@code interpolated}
     * markers are evaluated on their own grids — each marker records the instruction it wraps.
     *
     * @param rootInstruction the instruction to take the value of, instead of the program root
     */
    public double[] evaluate(int[] blockCoordinates, int rootInstruction) {
        return evaluate(blockCoordinates, rootInstruction, null);
    }

    /**
     * Evaluates several sub-graphs of the same program in one launch.
     *
     * <p>This is the point of the whole class. A launch of 1225 points against a 450-step serial chain
     * leaves the device almost idle — the measured cost is flat at ~750 µs whether the launch carries
     * one point or twenty-five — so folding every marker's grid into a single launch is worth several
     * times the bookkeeping. The scratch is sized for the worst case, which is one instruction array
     * per point regardless of how many roots are in play.
     *
     * @param roots one instruction index per point, or {@code null} to use the program root for all
     */
    public double[] evaluate(int[] blockCoordinates, int[] roots) {
        return evaluate(blockCoordinates, this.root, roots);
    }

    private double[] evaluate(int[] blockCoordinates, int rootInstruction, int[] roots) {
        if (blockCoordinates.length % 3 != 0) {
            throw new IllegalArgumentException("expected interleaved xyz triples, got "
                    + blockCoordinates.length + " ints");
        }
        int count = blockCoordinates.length / 3;
        if (count > this.capacity) {
            throw new IllegalArgumentException("batch of " + count + " points exceeds the "
                    + this.capacity + "-point capacity");
        }
        if (roots != null && roots.length != count) {
            throw new IllegalArgumentException("expected " + count + " roots, got " + roots.length);
        }
        if (rootInstruction < 0 || rootInstruction >= this.instructionCount) {
            throw new IllegalArgumentException("root instruction " + rootInstruction
                    + " is outside [0, " + this.instructionCount + ")");
        }
        if (count == 0) {
            return new double[0];
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostPoints = arena.allocateFrom(ValueLayout.JAVA_INT, blockCoordinates);
            this.context.copyToDevice(hostPoints, this.points);

            long rootAddress = 0L;
            if (roots != null) {
                if (this.roots == null) {
                    this.roots = this.context.allocate((long) this.capacity * Integer.BYTES);
                }
                MemorySegment hostRoots = arena.allocateFrom(ValueLayout.JAVA_INT, roots);
                this.context.copyToDevice(hostRoots, this.roots);
                rootAddress = this.roots.address();
            }

            try (KernelArguments arguments = new KernelArguments(9)) {
                arguments.addDevicePointer(this.blob.address())
                        .addDevicePointer(this.offsets.address())
                        .addInt(this.instructionCount)
                        .addInt(rootInstruction)
                        .addDevicePointer(rootAddress)
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
        if (this.roots != null) {
            this.context.free(this.roots);
        }
        this.context.free(this.scratch);
        this.context.free(this.points);
        this.context.free(this.offsets);
        this.context.free(this.blob);
    }
}
