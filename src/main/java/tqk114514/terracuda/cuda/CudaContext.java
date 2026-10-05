package tqk114514.terracuda.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An owning CUDA context: one {@code CUcontext}, one loaded module, and the buffer / launch helpers
 * built on top of it.
 *
 * <p>This is the layer the world-gen dispatcher will own one instance of. It wraps the parts of the
 * Driver API that have lifetime semantics, so that the rest of the mod never sees a raw
 * {@code CUcontext} or has to remember to free anything.
 *
 * <p><b>Threading.</b> {@code cuCtxCreate} makes the context current on the calling thread, and the
 * object owns a confined {@link Arena}. Create it, use it and {@link #close()} it on the same thread;
 * the dispatcher thread is expected to be that thread.
 *
 * <p>Every failure surfaces as a {@link CudaException} carrying the raw {@code CUresult}. Callers that
 * must not crash the game catch it and fall back to vanilla.
 */
public final class CudaContext implements AutoCloseable {

    private static final ValueLayout.OfInt CU_INT = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong CU_SIZE = ValueLayout.JAVA_LONG;

    /** A device-side allocation, identified by its {@code CUdeviceptr} address. */
    public record DeviceBuffer(long address, long size) {
    }

    private final CudaDriver driver;
    private final Arena arena = Arena.ofConfined();

    private final MethodHandle cuCtxCreate;
    private final MethodHandle cuCtxDestroy;
    private final MethodHandle cuModuleLoadData;
    private final MethodHandle cuModuleUnload;
    private final MethodHandle cuModuleGetFunction;
    private final MethodHandle cuMemAlloc;
    private final MethodHandle cuMemFree;
    private final MethodHandle cuMemcpyHtoD;
    private final MethodHandle cuMemcpyDtoH;
    private final MethodHandle cuMemcpyHtoDAsync;
    private final MethodHandle cuMemcpyDtoHAsync;
    private final MethodHandle cuLaunchKernel;
    private final MethodHandle cuCtxSynchronize;
    private final MethodHandle cuStreamCreate;
    private final MethodHandle cuStreamDestroy;
    private final MethodHandle cuStreamSynchronize;
    private final MethodHandle cuMemHostRegister;
    private final MethodHandle cuMemHostUnregister;

    private final MemorySegment contextHandle;
    private final Map<String, MemorySegment> functions = new HashMap<>();
    private final List<DeviceBuffer> allocations = new ArrayList<>();
    private final List<DeviceStream> streams = new ArrayList<>();

    private MemorySegment moduleHandle;
    private boolean closed;

    private CudaContext(CudaDriver driver, int device) {
        this.driver = driver;
        this.cuCtxCreate = driver.bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT, CU_INT),
                "cuCtxCreate_v2", "cuCtxCreate");
        this.cuCtxDestroy = driver.bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS),
                "cuCtxDestroy_v2", "cuCtxDestroy");
        this.cuModuleLoadData = driver.bind("cuModuleLoadData",
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.cuModuleUnload = driver.bind("cuModuleUnload", FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS));
        this.cuModuleGetFunction = driver.bind("cuModuleGetFunction",
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.cuMemAlloc = driver.bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_SIZE),
                "cuMemAlloc_v2", "cuMemAlloc");
        this.cuMemFree = driver.bindAny(FunctionDescriptor.of(CU_INT, CU_SIZE),
                "cuMemFree_v2", "cuMemFree");
        this.cuMemcpyHtoD = driver.bindAny(
                FunctionDescriptor.of(CU_INT, CU_SIZE, ValueLayout.ADDRESS, CU_SIZE),
                "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
        this.cuMemcpyDtoH = driver.bindAny(
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_SIZE, CU_SIZE),
                "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
        this.cuMemcpyHtoDAsync = driver.bindAny(
                FunctionDescriptor.of(CU_INT, CU_SIZE, ValueLayout.ADDRESS, CU_SIZE, ValueLayout.ADDRESS),
                "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
        this.cuMemcpyDtoHAsync = driver.bindAny(
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_SIZE, CU_SIZE, ValueLayout.ADDRESS),
                "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
        this.cuLaunchKernel = driver.bind("cuLaunchKernel", FunctionDescriptor.of(CU_INT,
                ValueLayout.ADDRESS,
                CU_INT, CU_INT, CU_INT,
                CU_INT, CU_INT, CU_INT,
                CU_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.cuCtxSynchronize = driver.bind("cuCtxSynchronize", FunctionDescriptor.of(CU_INT));
        this.cuStreamCreate = driver.bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT),
                "cuStreamCreate_v2", "cuStreamCreate");
        this.cuStreamDestroy = driver.bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS),
                "cuStreamDestroy_v2", "cuStreamDestroy");
        this.cuStreamSynchronize = driver.bind("cuStreamSynchronize",
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS));
        this.cuMemHostRegister = driver.bindAny(
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_SIZE, CU_INT),
                "cuMemHostRegister_v2", "cuMemHostRegister");
        this.cuMemHostUnregister = driver.bindAny(
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS),
                "cuMemHostUnregister_v2", "cuMemHostUnregister");

        MemorySegment handle = this.arena.allocate(ValueLayout.ADDRESS);
        int result = invokeInt(this.cuCtxCreate, handle, 0, device);
        this.driver.check(result, "cuCtxCreate");
        this.contextHandle = handle.get(ValueLayout.ADDRESS, 0);
    }

    /** Creates a context on the given device ordinal. */
    public static CudaContext create(CudaDriver driver, int device) {
        return new CudaContext(driver, device);
    }

    /**
     * Loads a module from an image.
     *
     * @param image a cubin binary, or NUL-terminated PTX text. Exactly one module may be loaded per
     *              context; call {@link #unloadModule()} first to replace it.
     */
    public void loadModule(byte[] image) {
        if (this.moduleHandle != null) {
            throw new IllegalStateException("a module is already loaded; call unloadModule() first");
        }
        if (image.length == 0) {
            throw new IllegalArgumentException("module image is empty");
        }
        try (Arena local = Arena.ofConfined()) {
            MemorySegment imageSegment = local.allocate(image.length);
            MemorySegment.copy(image, 0, imageSegment, ValueLayout.JAVA_BYTE, 0, image.length);
            MemorySegment handle = local.allocate(ValueLayout.ADDRESS);
            int result = invokeInt(this.cuModuleLoadData, handle, imageSegment);
            driver.check(result, "cuModuleLoadData");
            this.moduleHandle = handle.get(ValueLayout.ADDRESS, 0);
        }
        this.functions.clear();
    }

    public void unloadModule() {
        if (this.moduleHandle != null) {
            driver.check(invokeInt(this.cuModuleUnload, this.moduleHandle), "cuModuleUnload");
            this.moduleHandle = null;
            this.functions.clear();
        }
    }

    private MemorySegment function(String name) {
        if (this.moduleHandle == null) {
            throw new IllegalStateException("no module is loaded");
        }
        return this.functions.computeIfAbsent(name, key -> {
            try (Arena local = Arena.ofConfined()) {
                MemorySegment nameSegment = local.allocateFrom(key, StandardCharsets.UTF_8);
                MemorySegment handle = local.allocate(ValueLayout.ADDRESS);
                int result = invokeInt(this.cuModuleGetFunction, handle, this.moduleHandle, nameSegment);
                driver.check(result, "cuModuleGetFunction(" + key + ")");
                return handle.get(ValueLayout.ADDRESS, 0);
            }
        });
    }

    /** Allocates device memory. Freed automatically on {@link #close()}. */
    public DeviceBuffer allocate(long bytes) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment handle = local.allocate(CU_SIZE);
            int result = invokeInt(this.cuMemAlloc, handle, bytes);
            driver.check(result, "cuMemAlloc(" + bytes + " bytes)");
            DeviceBuffer buffer = new DeviceBuffer(handle.get(CU_SIZE, 0), bytes);
            this.allocations.add(buffer);
            return buffer;
        }
    }

    public void free(DeviceBuffer buffer) {
        driver.check(invokeInt(this.cuMemFree, buffer.address()), "cuMemFree");
        this.allocations.remove(buffer);
    }

    /** Copies host memory into a device buffer. The host segment must fit inside the buffer. */
    public void copyToDevice(MemorySegment host, DeviceBuffer device) {
        long bytes = host.byteSize();
        requireFits(device, bytes, "copyToDevice");
        driver.check(invokeInt(this.cuMemcpyHtoD, device.address(), host, bytes), "cuMemcpyHtoD");
    }

    /** Copies a device buffer back into host memory. The host segment must fit inside the buffer. */
    public void copyFromDevice(DeviceBuffer device, MemorySegment host) {
        long bytes = host.byteSize();
        requireFits(device, bytes, "copyFromDevice");
        driver.check(invokeInt(this.cuMemcpyDtoH, host, device.address(), bytes), "cuMemcpyDtoH");
    }

    private static void requireFits(DeviceBuffer device, long bytes, String operation) {
        if (bytes > device.size()) {
            throw new IllegalArgumentException(operation + ": " + bytes + " bytes exceeds the "
                    + device.size() + "-byte buffer");
        }
    }

    /** Launches a kernel on the default stream with the given grid and block dimensions. */
    public void launch(String kernel, int gridX, int gridY, int gridZ,
            int blockX, int blockY, int blockZ, KernelArguments arguments) {
        MemorySegment handle = function(kernel);
        int result = invokeInt(this.cuLaunchKernel, handle,
                gridX, gridY, gridZ, blockX, blockY, blockZ, 0,
                MemorySegment.NULL, arguments.pointers(), MemorySegment.NULL);
        driver.check(result, "cuLaunchKernel(" + kernel + ")");
    }

    /** Blocks until every previously issued call on the context has completed. */
    public void synchronize() {
        driver.check(invokeInt(this.cuCtxSynchronize), "cuCtxSynchronize");
    }

    /**
     * A stream, created and destroyed by the {@link CudaContext} that owns it.
     *
     * <p>Work queued on one stream executes in submission order, so a producer kernel and the copy
     * that reads it need no event between them — but work on different streams overlaps, which is
     * what this type exists for: the three per-chunk device passes used to run one after another
     * behind a context-wide synchronize, and the device sat idle between them.
     */
    public record DeviceStream(MemorySegment handle) {
    }

    /**
     * Creates a stream on this context. Destroyed with the context; no per-stream cleanup exists.
     */
    public DeviceStream createStream() {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment handle = local.allocate(ValueLayout.ADDRESS);
            int result = invokeInt(this.cuStreamCreate, handle, 0);
            this.driver.check(result, "cuStreamCreate");
            DeviceStream stream = new DeviceStream(handle.get(ValueLayout.ADDRESS, 0));
            this.streams.add(stream);
            return stream;
        }
    }

    /** Blocks until everything queued on {@code stream} has completed — and only that stream. */
    public void synchronizeStream(DeviceStream stream) {
        driver.check(invokeInt(this.cuStreamSynchronize, stream.handle()), "cuStreamSynchronize");
    }

    /**
     * Registers a host segment as page-locked, so async copies into and out of it truly queue.
     *
     * <p>The one asymmetry that matters here: an async copy out of pageable memory may block the
     * caller while the driver stages it — a copy <em>into</em> pageable memory always does, because
     * the driver cannot defer a write into memory the host might reuse. The first streams change
     * measured flat until this was pointed at the copy-back staging segment: every submission ends
     * in a device-to-host copy, and with a pageable destination that copy waited for the whole
     * stream behind it before submitting returned — the overlap the streams existed for never
     * happened.
     */
    public void pin(MemorySegment host) {
        driver.check(invokeInt(this.cuMemHostRegister, host, host.byteSize(), 0), "cuMemHostRegister");
    }

    /** Undoes {@link #pin(MemorySegment)}; must run before the segment's memory is released. */
    public void unpin(MemorySegment host) {
        driver.check(invokeInt(this.cuMemHostUnregister, host), "cuMemHostUnregister");
    }

    /**
     * Queues a host-to-device copy on {@code stream}, in order with the other work on it.
     *
     * <p>The host segment must stay unchanged until the copy completes; the callers own that by
     * waiting on the stream before reusing the segment.
     */
    public void copyToDeviceAsync(MemorySegment host, DeviceBuffer device, DeviceStream stream) {
        long bytes = host.byteSize();
        requireFits(device, bytes, "copyToDeviceAsync");
        driver.check(invokeInt(this.cuMemcpyHtoDAsync, device.address(), host, bytes, stream.handle()),
                "cuMemcpyHtoDAsync");
    }

    /**
     * Queues a device-to-host copy on {@code stream}, in order with the other work on it.
     *
     * <p>Read the destination only after {@link #synchronizeStream(DeviceStream)}.
     */
    public void copyFromDeviceAsync(DeviceBuffer device, MemorySegment host, DeviceStream stream) {
        long bytes = host.byteSize();
        requireFits(device, bytes, "copyFromDeviceAsync");
        driver.check(invokeInt(this.cuMemcpyDtoHAsync, host, device.address(), bytes, stream.handle()),
                "cuMemcpyDtoHAsync");
    }

    /**
     * Launches a kernel on {@code stream}, in order with the copies and kernels already queued on it.
     */
    public void launch(String kernel, int gridX, int gridY, int gridZ,
            int blockX, int blockY, int blockZ, KernelArguments arguments, DeviceStream stream) {
        MemorySegment handle = function(kernel);
        int result = invokeInt(this.cuLaunchKernel, handle,
                gridX, gridY, gridZ, blockX, blockY, blockZ, 0,
                stream.handle(), arguments.pointers(), MemorySegment.NULL);
        driver.check(result, "cuLaunchKernel(" + kernel + ")");
    }

    /** Whether {@link #close()} has run. Buffer owners check this before freeing. */
    public boolean isClosed() {
        return this.closed;
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        try {
            for (DeviceBuffer buffer : List.copyOf(this.allocations)) {
                invokeInt(this.cuMemFree, buffer.address());
            }
            this.allocations.clear();
            // Streams must be destroyed before the context they belong to.
            for (DeviceStream stream : List.copyOf(this.streams)) {
                invokeInt(this.cuStreamDestroy, stream.handle());
            }
            this.streams.clear();
            if (this.moduleHandle != null) {
                invokeInt(this.cuModuleUnload, this.moduleHandle);
                this.moduleHandle = null;
            }
            invokeInt(this.cuCtxDestroy, this.contextHandle);
        } finally {
            this.arena.close();
        }
    }

    private int invokeInt(MethodHandle handle, Object... arguments) {
        try {
            return (int) handle.invokeWithArguments(arguments);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw new CudaException(CudaException.BINDING_FAILURE, "CUDA_ERROR_BINDING",
                    "CUDA entry point could not be invoked", t);
        }
    }
}
