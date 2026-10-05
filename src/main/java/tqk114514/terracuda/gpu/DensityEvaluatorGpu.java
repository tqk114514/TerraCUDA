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
 * GPU-backed evaluation of a lowered density program: the K2 and K3 kernels of the design doc.
 *
 * <p>Two images of the same program live here, and the split between them is the point:
 *
 * <ul>
 *   <li>the full image, evaluated one thread per point against a root, produces the marker tables —
 *       the {@code 5 × 5 × 49} corner grid of every interpolated marker;</li>
 *   <li>the reduced image, from {@link DensityProgramImage#ofBlocks}, keeps only the DAG above those
 *       markers and leaves them as {@code OVERRIDE} leaves. One thread per block blends the corners
 *       down, drops the results into the leaves, and evaluates what is left.</li>
 * </ul>
 *
 * <p>The second image is why the chunk pass is cheap. Running the whole program per block would be
 * ten times the instructions for the same answer, and uploading the marker tables to blend them on
 * the host — which is what this did first — spends more on turning them back into a Java array than
 * the blend itself costs.
 *
 * <p>One thread per point, so the per-thread scratch is a global buffer rather than local memory: at
 * a few hundred instructions per thread, local memory would be several kilobytes each.
 */
public final class DensityEvaluatorGpu implements AutoCloseable {

    /** The exported kernel symbol for the marker-table pass, matching {@code terracuda.cu}. */
    public static final String KERNEL_NAME = "terracuda_density_evaluate";

    /** The exported kernel symbol for the per-block pass. */
    public static final String BLOCKS_KERNEL_NAME = "terracuda_density_blocks";

    private static final int BLOCK_SIZE = 128;
    private static final int MIN_CAPACITY = 2048;

    /** The chunk is 16 blocks across, which the block kernel needs to recover x and z. */
    private static final int CHUNK_SIZE_XZ = 16;

    private final CudaContext context;
    private final DensityProgramImage image;
    private final DensityProgramImage blockImage;
    private final CudaContext.DeviceBuffer blob;
    private final CudaContext.DeviceBuffer offsets;
    private final CudaContext.DeviceBuffer blockBlob;
    private final CudaContext.DeviceBuffer blockOffsets;
    private final CudaContext.DeviceBuffer points;
    private final CudaContext.DeviceBuffer scratch;
    private final CudaContext.DeviceBuffer results;
    private CudaContext.DeviceBuffer roots;
    private CudaContext.DeviceBuffer markerBases;
    private CudaContext.DeviceBuffer markerSlots;
    private CudaContext.DeviceBuffer blockScratch;
    private CudaContext.DeviceBuffer blockResults;
    private int[] combinedPoints;
    private int[] combinedRoots;
    /** Outlives a single call, so the staging segments can be reused instead of re-faulted. */
    private final Arena hostArena = Arena.ofShared();
    private MemorySegment hostPoints;
    private MemorySegment hostRoots;
    private MemorySegment hostDoubles;
    private final int instructionCount;
    private final int root;
    private final int capacity;

    /**
     * This evaluator's private stream. Work queued on it runs in submission order — the corner
     * tables, the block pass over them and the copy back need no events between them — while work on
     * different evaluators' streams overlaps on the device. Three programs used to run strictly one
     * after another behind a context-wide synchronize; the synchronize was the serialisation.
     */
    private final CudaContext.DeviceStream stream;

    /** What {@link #submitChunk} queued and {@link #awaitChunk} will collect. */
    private double[] pendingOut;
    private MemorySegment pendingHost;
    private int pendingCount;

    private boolean closed;

    public DensityEvaluatorGpu(CudaContext context, DensityProgram program, int capacity) {
        this(context, DensityProgramImage.of(program), DensityProgramImage.ofBlocks(program), capacity);
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
        return new DensityEvaluatorGpu(context, image, DensityProgramImage.ofBlocks(program), budget);
    }

    private DensityEvaluatorGpu(CudaContext context, DensityProgramImage image,
            DensityProgramImage blockImage, int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.context = context;
        this.image = image;
        this.blockImage = blockImage;
        this.instructionCount = image.instructionCount();
        this.root = image.root();
        this.capacity = capacity;

        this.blob = uploadBlob(image);
        this.offsets = uploadOffsets(image);
        this.blockBlob = uploadBlob(blockImage);
        this.blockOffsets = uploadOffsets(blockImage);

        this.points = context.allocate(3L * capacity * Integer.BYTES);
        this.scratch = context.allocate((long) capacity * this.instructionCount * Double.BYTES);
        this.results = context.allocate((long) capacity * Double.BYTES);
        this.stream = context.createStream();
    }

    /** Uploads one image's blob. */
    private CudaContext.DeviceBuffer uploadBlob(DensityProgramImage source) {
        try (Arena arena = Arena.ofConfined()) {
            byte[] blobBytes = source.blob();
            MemorySegment hostBlob = arena.allocate(blobBytes.length);
            MemorySegment.copy(blobBytes, 0, hostBlob, ValueLayout.JAVA_BYTE, 0, blobBytes.length);
            CudaContext.DeviceBuffer buffer = this.context.allocate(blobBytes.length);
            this.context.copyToDevice(hostBlob, buffer);
            return buffer;
        }
    }

    /** Uploads one image's offset table. */
    private CudaContext.DeviceBuffer uploadOffsets(DensityProgramImage source) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostOffsets = arena.allocateFrom(ValueLayout.JAVA_LONG, source.offsets());
            CudaContext.DeviceBuffer buffer = this.context.allocate(hostOffsets.byteSize());
            this.context.copyToDevice(hostOffsets, buffer);
            return buffer;
        }
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

    /** The reduced image the per-block pass runs: the DAG above the interpolated markers. */
    public DensityProgramImage blockImage() {
        return this.blockImage;
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
        launch(blockCoordinates, rootInstruction, roots);
        int count = blockCoordinates.length / 3;
        MemorySegment host = hostDoubles(count);
        this.context.copyFromDeviceAsync(this.results, host.asSlice(0, (long) count * Double.BYTES),
                this.stream);
        this.context.synchronizeStream(this.stream);
        return host.asSlice(0, (long) count * Double.BYTES).toArray(ValueLayout.JAVA_DOUBLE);
    }

    /**
     * Runs the marker-table evaluation and leaves the result in device memory.
     *
     * <p>Used by {@link #blocksForChunk}, where the corner tables are an input to the next kernel
     * rather than something the host wants: reading them back only to upload them again would be a
     * round trip through the slowest link in the system for no reason.
     *
     * <p>Everything on the host side is a cached segment. That is not tidiness: the points alone are
     * 73 KB, and a fresh native allocation of that size pays for its own page faults on first touch.
     * Measured across batch sizes, this call's fixed cost was 0.71 ms against a kernel that costs
     * 0.87 ms per chunk no matter how many chunks ride along — so the fixed cost is the whole of what
     * batching the marker tables would have bought, and reusing the buffers buys it directly.
     *
     * <p>The work is queued on this evaluator's stream and returns without waiting. Ordering within
     * the stream carries every dependency — the copy of the points lands before the kernel that reads
     * it, this kernel before whatever queued next — so no events are needed; the caller waits on the
     * stream when it wants the values, and only then may it overwrite the staging segments.
     */
    private void launch(int[] blockCoordinates, int rootInstruction, int[] roots) {
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
            return;
        }

        long pointBytes = 3L * count * Integer.BYTES;
        if (this.hostPoints == null || this.hostPoints.byteSize() < pointBytes) {
            this.hostPoints = this.hostArena.allocate(pointBytes);
        }
        MemorySegment.copy(blockCoordinates, 0, this.hostPoints, ValueLayout.JAVA_INT, 0, 3 * count);
        this.context.copyToDeviceAsync(this.hostPoints.asSlice(0, pointBytes), this.points, this.stream);

        long rootAddress = 0L;
        if (roots != null) {
            if (this.roots == null) {
                this.roots = this.context.allocate((long) this.capacity * Integer.BYTES);
            }
            long rootBytes = (long) count * Integer.BYTES;
            if (this.hostRoots == null || this.hostRoots.byteSize() < rootBytes) {
                this.hostRoots = this.hostArena.allocate(rootBytes);
            }
            MemorySegment.copy(roots, 0, this.hostRoots, ValueLayout.JAVA_INT, 0, count);
            this.context.copyToDeviceAsync(this.hostRoots.asSlice(0, rootBytes), this.roots, this.stream);
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
            this.context.launch(KERNEL_NAME, grid, 1, 1, BLOCK_SIZE, 1, 1, arguments, this.stream);
        }
    }

    /** A cached host segment of at least {@code doubles} doubles, page-locked for true async copies. */
    private MemorySegment hostDoubles(int doubles) {
        long bytes = (long) doubles * Double.BYTES;
        if (this.hostDoubles == null || this.hostDoubles.byteSize() < bytes) {
            MemorySegment replacement = this.hostArena.allocate(Math.max(bytes, 1L << 16));
            this.context.pin(replacement);
            if (this.hostDoubles != null) {
                this.context.unpin(this.hostDoubles);
            }
            this.hostDoubles = replacement;
        }
        return this.hostDoubles;
    }

    /**
     * The value of the program at every block of one chunk — marker tables, blend and the DAG above
     * them, all on the device.
     *
     * <p>This is K3's density half. Both stages it replaces were measured on the host and neither was
     * small: about seven and a half milliseconds of interpreting the DAG above the markers, and three
     * and a half of trilinear blend, against roughly a fifth of a millisecond to read the answer back.
     *
     * @param order {@link tqk114514.terracuda.density.CornerInterpolation#LERP3} or {@code INCREMENTAL}
     * @return {@code 16 * 16 * height} values, indexed {@code (x * 16 + z) * height + (y - minY)}
     */
    public double[] blocksForChunk(NoiseSettings settings, int chunkX, int chunkZ, int order) {
        double[] out = new double[CHUNK_SIZE_XZ * CHUNK_SIZE_XZ * settings.height()];
        blocksForChunk(settings, chunkX, chunkZ, order, out);
        return out;
    }

    /**
     * The same, writing into {@code out} rather than allocating.
     *
     * <p>A chunk pass allocates about five megabytes of scratch between the three programs, and the
     * host side of each is the same size every time. Callers that fill chunks in a loop — which is all
     * of them — should keep their own arrays and hand them in.
     */
    public void blocksForChunk(NoiseSettings settings, int chunkX, int chunkZ, int order,
            double[] out) {
        submitChunk(settings, chunkX, chunkZ, order, out);
        awaitChunk();
    }

    /**
     * Queues one chunk's whole device half — corner tables, block pass, and the copy back — and
     * returns without waiting.
     *
     * <p>This split exists so a caller can queue the same chunk on several evaluators before waiting
     * on any of them: the density, vein-toggle and vein-ridged programs of a chunk are independent,
     * and their passes used to run strictly one after another behind a context-wide synchronize. On
     * their own streams they overlap; the waits land on whichever finishes last.
     *
     * <p>Nothing may touch this evaluator's staging segments between {@code submitChunk} and
     * {@link #awaitChunk()} — the host-side points upload is asynchronous, and overwriting the
     * segment while the copy is in flight would corrupt the kernel's input. One caller at a time,
     * submit-then-wait, is the contract the production path already follows.
     *
     * @param order {@link tqk114514.terracuda.density.CornerInterpolation#LERP3} or {@code INCREMENTAL}
     */
    public void submitChunk(NoiseSettings settings, int chunkX, int chunkZ, int order, double[] out) {
        int[] cornerRoots = interpolatedRoots(this.image);
        int[] slots = interpolatedRoots(this.blockImage);
        if (cornerRoots.length != slots.length) {
            throw new IllegalStateException("the reduced image has " + slots.length
                    + " interpolated markers, the full one has " + cornerRoots.length);
        }

        int markerCount = cornerRoots.length;
        int countY = ChunkCornerTables.cornerCountY(settings);
        int perMarker = ChunkCornerTables.CORNERS_XZ * ChunkCornerTables.CORNERS_XZ * countY;
        int count = CHUNK_SIZE_XZ * CHUNK_SIZE_XZ * settings.height();
        if (out.length != count) {
            throw new IllegalArgumentException("expected " + count + " values, got " + out.length);
        }

        if (markerCount > 0) {
            int[] corners = ChunkCornerTables.cornerCoordinates(settings, chunkX, chunkZ);
            int pointsNeeded = markerCount * corners.length;
            if (this.combinedPoints == null || this.combinedPoints.length != pointsNeeded) {
                this.combinedPoints = new int[pointsNeeded];
                this.combinedRoots = new int[markerCount * perMarker];
            }
            for (int m = 0; m < markerCount; m++) {
                System.arraycopy(corners, 0, this.combinedPoints, m * corners.length, corners.length);
                java.util.Arrays.fill(this.combinedRoots, m * perMarker, (m + 1) * perMarker,
                        cornerRoots[m]);
            }
            launch(this.combinedPoints, this.root, this.combinedRoots);
        }

        int blockInstructions = this.blockImage.instructionCount();
        if (this.blockScratch == null) {
            this.markerBases = this.context.allocate((long) Math.max(1, markerCount) * Integer.BYTES);
            this.markerSlots = this.context.allocate((long) Math.max(1, markerCount) * Integer.BYTES);
            this.blockResults = this.context.allocate((long) count * Double.BYTES);
            this.blockScratch = this.context.allocate((long) count * blockInstructions * Double.BYTES);

            // Bases and slots never change — where a marker's corner grid starts in the results and
            // which instruction its blend lands in are properties of the program, not the chunk. They
            // were uploaded per chunk inside a confined arena, which also meant the arena had to
            // outlive nothing; now it really doesn't.
            try (Arena arena = Arena.ofConfined()) {
                int[] bases = new int[markerCount];
                for (int m = 0; m < markerCount; m++) {
                    bases[m] = m * perMarker;
                }
                this.context.copyToDevice(arena.allocateFrom(ValueLayout.JAVA_INT, bases),
                        this.markerBases);
                this.context.copyToDevice(arena.allocateFrom(ValueLayout.JAVA_INT, slots),
                        this.markerSlots);
            }
        }

        try (KernelArguments arguments = new KernelArguments(20)) {
            arguments.addDevicePointer(this.blockBlob.address())
                    .addDevicePointer(this.blockOffsets.address())
                    .addInt(blockInstructions)
                    .addInt(this.blockImage.root())
                    .addDevicePointer(this.results.address())
                    .addDevicePointer(this.markerBases.address())
                    .addDevicePointer(this.markerSlots.address())
                    .addInt(markerCount)
                    .addInt(settings.getCellWidth())
                    .addInt(settings.getCellHeight())
                    .addInt(countY)
                    .addInt(CHUNK_SIZE_XZ)
                    .addInt(settings.height())
                    .addInt(order)
                    .addInt(chunkX * CHUNK_SIZE_XZ)
                    .addInt(settings.minY())
                    .addInt(chunkZ * CHUNK_SIZE_XZ)
                    .addDevicePointer(this.blockScratch.address())
                    .addDevicePointer(this.blockResults.address())
                    .addInt(count);
            int grid = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            this.context.launch(BLOCKS_KERNEL_NAME, grid, 1, 1, BLOCK_SIZE, 1, 1, arguments,
                    this.stream);
        }

        // The staging segment is reused across calls: a fresh native allocation of this size pays
        // for its own page faults every time, which measured at more than the copy itself.
        MemorySegment host = hostDoubles(count);
        this.context.copyFromDeviceAsync(this.blockResults, host.asSlice(0, (long) count * Double.BYTES),
                this.stream);
        this.pendingOut = out;
        this.pendingHost = host;
        this.pendingCount = count;
    }

    /**
     * Waits for the work {@link #submitChunk} queued and fills the array it was given.
     *
     * <p>Safe to call once per submission; waiting for a second evaluator's stream in between does
     * not affect this one's results — the copy back already queued behind this stream's own kernels.
     */
    public void awaitChunk() {
        this.context.synchronizeStream(this.stream);
        MemorySegment.copy(this.pendingHost, ValueLayout.JAVA_DOUBLE, 0L, this.pendingOut, 0,
                this.pendingCount);
    }

    /** The image instruction each interpolated marker wraps, in the image's own marker order. */
    private static int[] interpolatedRoots(DensityProgramImage source) {
        int[] kinds = source.markerKinds();
        int[] roots = source.markerRoots();
        int markers = 0;
        for (int kind : kinds) {
            if (kind == DensityProgram.MARKER_INTERPOLATED) {
                markers++;
            }
        }
        int[] out = new int[markers];
        int at = 0;
        for (int i = 0; i < kinds.length; i++) {
            if (kinds[i] == DensityProgram.MARKER_INTERPOLATED) {
                out[at++] = roots[i];
            }
        }
        return out;
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        if (this.hostDoubles != null && !this.context.isClosed()) {
            this.context.unpin(this.hostDoubles);
        }
        this.hostArena.close();
        if (this.context.isClosed()) {
            return;
        }
        if (this.blockScratch != null) {
            this.context.free(this.blockScratch);
        }
        if (this.blockResults != null) {
            this.context.free(this.blockResults);
        }
        if (this.markerSlots != null) {
            this.context.free(this.markerSlots);
        }
        if (this.markerBases != null) {
            this.context.free(this.markerBases);
        }
        this.context.free(this.results);
        if (this.roots != null) {
            this.context.free(this.roots);
        }
        this.context.free(this.scratch);
        this.context.free(this.points);
        this.context.free(this.blockOffsets);
        this.context.free(this.blockBlob);
        this.context.free(this.offsets);
        this.context.free(this.blob);
    }
}
