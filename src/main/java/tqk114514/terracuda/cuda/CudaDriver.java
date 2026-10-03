package tqk114514.terracuda.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

/**
 * Minimal Java FFM binding for the CUDA Driver API.
 *
 * <p>Only the calls required for device discovery are bound here; kernel loading, memory management
 * and launches are added in later milestones. The binding is deliberately thin: every method returns
 * the raw {@code CUresult} where that is meaningful, and {@link #check(int, String)} converts a
 * failure into a {@link CudaException}. Callers that must not crash the game catch that exception and
 * fall back to vanilla.
 *
 * <p>The library is resolved lazily through {@link SymbolLookup#libraryLookup(String, Arena)} so that
 * a machine without an NVIDIA driver simply gets {@link Optional#empty()} instead of an error.
 *
 * <p>Threading: the underlying CUDA driver is not thread-safe in the way the rest of this project
 * uses it. This class only performs per-call, self-contained driver calls and does not create a
 * context; the dispatcher thread owns all context-bound state.
 */
public final class CudaDriver implements AutoCloseable {

    /** Driver library name; {@code nvcuda} resolves to {@code nvcuda.dll} on Windows. */
    public static final String LIBRARY_NAME = "nvcuda";

    public static final int CUDA_SUCCESS = 0;
    public static final int CUDA_ERROR_INVALID_VALUE = 1;
    public static final int CUDA_ERROR_NO_DEVICE = 100;
    public static final int CUDA_ERROR_INVALID_DEVICE = 101;
    public static final int CUDA_ERROR_NOT_FOUND = 500;

    // CUdevice_attribute values. These are frozen in the driver ABI and safe to hard-code.
    public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    public static final int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
    public static final int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;
    public static final int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
    public static final int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12;
    public static final int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
    public static final int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    public static final int CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;
    public static final int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31;
    public static final int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36;
    public static final int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37;
    public static final int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38;
    public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    private static final ValueLayout.OfInt CU_INT = ValueLayout.JAVA_INT;
    /** {@code size_t} / {@code unsigned long long}; 64-bit on every platform this mod targets. */
    private static final ValueLayout.OfLong CU_SIZE = ValueLayout.JAVA_LONG;

    private static final int DEVICE_NAME_BYTES = 256;
    private static final int ERROR_STRING_BYTES = 256;

    private final Arena arena;
    private final SymbolLookup lookup;
    private final MethodHandle cuInit;
    private final MethodHandle cuDriverGetVersion;
    private final MethodHandle cuDeviceGetCount;
    private final MethodHandle cuDeviceGet;
    private final MethodHandle cuDeviceGetName;
    private final MethodHandle cuDeviceGetAttribute;
    private final MethodHandle cuDeviceTotalMem;
    private final MethodHandle cuGetErrorName;
    private final MethodHandle cuGetErrorString;

    private boolean closed;

    private CudaDriver(Arena arena, SymbolLookup lookup) {
        this.arena = arena;
        this.lookup = lookup;
        this.cuInit = bind("cuInit", FunctionDescriptor.of(CU_INT, CU_INT));
        this.cuDriverGetVersion = bind("cuDriverGetVersion", FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS));
        this.cuDeviceGetCount = bind("cuDeviceGetCount", FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS));
        this.cuDeviceGet = bind("cuDeviceGet", FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT));
        this.cuDeviceGetName = bind("cuDeviceGetName",
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT, CU_INT));
        this.cuDeviceGetAttribute = bind("cuDeviceGetAttribute",
                FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT, CU_INT));
        // cuDeviceTotalMem was versioned to _v2; some old drivers only export the unsuffixed name.
        this.cuDeviceTotalMem = bindAny(FunctionDescriptor.of(CU_INT, ValueLayout.ADDRESS, CU_INT),
                "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
        this.cuGetErrorName = bind("cuGetErrorName", FunctionDescriptor.of(CU_INT, CU_INT, ValueLayout.ADDRESS));
        this.cuGetErrorString = bind("cuGetErrorString", FunctionDescriptor.of(CU_INT, CU_INT, ValueLayout.ADDRESS));
    }

    /**
     * Loads the CUDA driver library and binds the entry points.
     *
     * @return a ready-to-use driver, or empty when the library is missing (no NVIDIA driver, or a
     *         non-Windows host without CUDA). Never throws for the "not available" case.
     */
    public static Optional<CudaDriver> tryLoad() {
        Arena arena = Arena.ofShared();
        try {
            SymbolLookup lookup = SymbolLookup.libraryLookup(LIBRARY_NAME, arena);
            return Optional.of(new CudaDriver(arena, lookup));
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            // IllegalArgumentException when the library is missing, UnsatisfiedLinkError when a
            // symbol cannot be resolved. Both mean "no usable CUDA driver here".
            arena.close();
            return Optional.empty();
        }
    }

    /**
     * Binds a further entry point from the loaded driver library.
     *
     * <p>Exposed so that the compute layer can keep each {@link FunctionDescriptor} next to the call
     * it describes, instead of every driver entry point living in this class. Callers are responsible
     * for getting the C type widths right: {@code size_t} and {@code CUdeviceptr} are 64-bit here.
     *
     * @throws IllegalStateException when the symbol is missing from the driver
     */
    public MethodHandle bind(String name, FunctionDescriptor descriptor) {
        MemorySegment symbol = lookup.find(name)
                .orElseThrow(() -> new IllegalStateException("CUDA symbol not found: " + name));
        return Linker.nativeLinker().downcallHandle(symbol, descriptor);
    }

    /** Like {@link #bind(String, FunctionDescriptor)} but tries each name in turn. */
    public MethodHandle bindAny(FunctionDescriptor descriptor, String... names) {
        for (String name : names) {
            Optional<MemorySegment> symbol = lookup.find(name);
            if (symbol.isPresent()) {
                return Linker.nativeLinker().downcallHandle(symbol.get(), descriptor);
            }
        }
        throw new IllegalStateException("none of the CUDA symbols were found: " + String.join(", ", names));
    }

    /** {@code cuInit}. Returns the raw {@code CUresult}; callers decide whether to escalate. */
    public int init() {
        try {
            return (int) cuInit.invokeExact(0);
        } catch (Throwable t) {
            throw bindingFailure("cuInit", t);
        }
    }

    /** {@code cuDriverGetVersion}, e.g. {@code 12060} for CUDA 12.6. */
    public int driverVersion() {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(CU_INT);
            int result = (int) cuDriverGetVersion.invokeExact(out);
            check(result, "cuDriverGetVersion");
            return out.get(CU_INT, 0);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDriverGetVersion", t);
        }
    }

    /** {@code cuDeviceGetCount}. */
    public int deviceCount() {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(CU_INT);
            int result = (int) cuDeviceGetCount.invokeExact(out);
            check(result, "cuDeviceGetCount");
            return out.get(CU_INT, 0);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDeviceGetCount", t);
        }
    }

    /**
     * {@code cuDeviceGet}. The returned handle is a plain {@code CUdevice} ordinal-scoped int, not a
     * pointer, so it is safe to hand around as an {@code int}.
     */
    public int deviceHandle(int ordinal) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(CU_INT);
            int result = (int) cuDeviceGet.invokeExact(out, ordinal);
            check(result, "cuDeviceGet");
            return out.get(CU_INT, 0);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDeviceGet", t);
        }
    }

    /** {@code cuDeviceGetName}. */
    public String deviceName(int device) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment buffer = local.allocate(DEVICE_NAME_BYTES);
            int result = (int) cuDeviceGetName.invokeExact(buffer, DEVICE_NAME_BYTES, device);
            check(result, "cuDeviceGetName");
            return buffer.getString(0, StandardCharsets.UTF_8);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDeviceGetName", t);
        }
    }

    /** {@code cuDeviceGetAttribute} for a {@code CUdevice_attribute}. */
    public int deviceAttribute(int device, int attribute) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(CU_INT);
            int result = (int) cuDeviceGetAttribute.invokeExact(out, attribute, device);
            check(result, "cuDeviceGetAttribute");
            return out.get(CU_INT, 0);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDeviceGetAttribute", t);
        }
    }

    /** {@code cuDeviceTotalMem}, in bytes. */
    public long deviceTotalMem(int device) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(CU_SIZE);
            int result = (int) cuDeviceTotalMem.invokeExact(out, device);
            check(result, "cuDeviceTotalMem");
            return out.get(CU_SIZE, 0);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw bindingFailure("cuDeviceTotalMem", t);
        }
    }

    /** {@code cuGetErrorName}, e.g. {@code CUDA_SUCCESS} or {@code CUDA_ERROR_INVALID_DEVICE}. */
    public String errorName(int code) {
        String value = readErrorString(cuGetErrorName, code);
        return value != null ? value : "CUDA_ERROR_UNKNOWN(" + code + ")";
    }

    /** {@code cuGetErrorString}: the human readable counterpart of {@link #errorName(int)}. */
    public String errorString(int code) {
        String value = readErrorString(cuGetErrorString, code);
        return value != null ? value : "unknown error code " + code;
    }

    /** Throws {@link CudaException} unless {@code result} is {@code CUDA_SUCCESS}. */
    public void check(int result, String operation) {
        if (result != CUDA_SUCCESS) {
            String name = errorName(result);
            throw new CudaException(result, name, operation + " failed: " + name + " - " + errorString(result));
        }
    }

    private String readErrorString(MethodHandle handle, int code) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment out = local.allocate(ValueLayout.ADDRESS);
            int result = (int) handle.invokeExact(code, out);
            if (result != CUDA_SUCCESS) {
                return null;
            }
            MemorySegment text = out.get(ValueLayout.ADDRESS, 0);
            if (text.address() == 0L) {
                return null;
            }
            return text.reinterpret(ERROR_STRING_BYTES).getString(0, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            return null;
        }
    }

    private static CudaException bindingFailure(String operation, Throwable cause) {
        return new CudaException(CudaException.BINDING_FAILURE, "CUDA_ERROR_BINDING",
                operation + " could not be invoked through the FFM binding", cause);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            arena.close();
        }
    }
}
