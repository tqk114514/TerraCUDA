package tqk114514.terracuda.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Runtime CUDA C++ compilation through NVRTC.
 *
 * <p>This exists because nvcc is not always usable: on Windows it hard-requires the MSVC host
 * compiler, and a machine can have a perfectly good CUDA driver and GPU without Visual Studio. NVRTC
 * has no such requirement — it compiles CUDA C++ straight to PTX, which the driver then JITs to SASS.
 * That is the "no matching cubin, fall back to PTX" rung of the design doc's degradation ladder.
 *
 * <p>The compiled PTX is byte-compatible with {@code nvcc -ptx}: same front end, same options. What
 * matters for correctness is that the caller passes {@code --fmad=false} and never
 * {@code --use_fast_math}, exactly as the build-time path does.
 *
 * <p>The library is located by probing {@code CUDA_PATH} and the standard toolkit install directory,
 * rather than relying on the DLL being on {@code PATH}. The {@code nvrtc-builtins} companion is
 * preloaded from the same directory because NVRTC resolves it lazily by name.
 */
public final class NvrtcCompiler implements AutoCloseable {

    private static final int NVRTC_SUCCESS = 0;

    private final Arena arena;
    private final MethodHandle nvrtcCreateProgram;
    private final MethodHandle nvrtcCompileProgram;
    private final MethodHandle nvrtcGetPTXSize;
    private final MethodHandle nvrtcGetPTX;
    private final MethodHandle nvrtcGetProgramLogSize;
    private final MethodHandle nvrtcGetProgramLog;
    private final MethodHandle nvrtcDestroyProgram;
    private final MethodHandle nvrtcGetErrorString;
    private final Path libraryPath;

    private NvrtcCompiler(Arena arena, SymbolLookup lookup, Path libraryPath) {
        this.arena = arena;
        this.libraryPath = libraryPath;
        Linker linker = Linker.nativeLinker();
        this.nvrtcCreateProgram = linker.downcallHandle(lookup.find("nvrtcCreateProgram").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.nvrtcCompileProgram = linker.downcallHandle(lookup.find("nvrtcCompileProgram").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS));
        this.nvrtcGetPTXSize = linker.downcallHandle(lookup.find("nvrtcGetPTXSize").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.nvrtcGetPTX = linker.downcallHandle(lookup.find("nvrtcGetPTX").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.nvrtcGetProgramLogSize = linker.downcallHandle(lookup.find("nvrtcGetProgramLogSize").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.nvrtcGetProgramLog = linker.downcallHandle(lookup.find("nvrtcGetProgramLog").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.nvrtcDestroyProgram = linker.downcallHandle(lookup.find("nvrtcDestroyProgram").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        this.nvrtcGetErrorString = linker.downcallHandle(lookup.find("nvrtcGetErrorString").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    }

    /** Locates and binds NVRTC, or returns empty when the toolkit is not installed. */
    public static Optional<NvrtcCompiler> tryLoad() {
        for (Path candidate : candidateLibraries()) {
            Optional<NvrtcCompiler> compiler = tryLoadFrom(candidate);
            if (compiler.isPresent()) {
                return compiler;
            }
        }
        return Optional.empty();
    }

    private static Optional<NvrtcCompiler> tryLoadFrom(Path library) {
        Arena arena = Arena.ofShared();
        try {
            preloadBuiltins(library, arena);
            SymbolLookup lookup = SymbolLookup.libraryLookup(library.toAbsolutePath().toString(), arena);
            return Optional.of(new NvrtcCompiler(arena, lookup, library));
        } catch (Throwable t) {
            arena.close();
            return Optional.empty();
        }
    }

    /**
     * NVRTC loads {@code nvrtc-builtins64_*.dll} lazily by name at compile time. Windows only resolves
     * that against the search path, which will not contain the toolkit's bin directory, so load it
     * ourselves first — an already-loaded module is found by name.
     */
    private static void preloadBuiltins(Path library, Arena arena) {
        Path directory = library.getParent();
        if (directory == null) {
            return;
        }
        try (Stream<Path> entries = Files.list(directory)) {
            for (Path entry : entries.toList()) {
                String name = entry.getFileName().toString().toLowerCase(Locale.ROOT);
                if (name.startsWith("nvrtc-builtins") && name.endsWith(".dll")) {
                    try {
                        SymbolLookup.libraryLookup(entry.toAbsolutePath().toString(), arena);
                    } catch (Throwable ignored) {
                        // A builtins library that refuses to load is not fatal on its own.
                    }
                }
            }
        } catch (Exception ignored) {
            // Directory listing failed; the main library load will report the real problem.
        }
    }

    private static List<Path> candidateLibraries() {
        List<Path> candidates = new ArrayList<>();
        for (String root : toolkitRoots()) {
            Path bin = Path.of(root, "bin", "x64");
            Path flatBin = Path.of(root, "bin");
            for (Path directory : List.of(bin, flatBin)) {
                if (!Files.isDirectory(directory)) {
                    continue;
                }
                try (Stream<Path> entries = Files.list(directory)) {
                    entries.filter(entry -> {
                        String name = entry.getFileName().toString().toLowerCase(Locale.ROOT);
                        return name.startsWith("nvrtc64_") && name.endsWith(".dll");
                    }).forEach(candidates::add);
                } catch (Exception ignored) {
                    // Unreadable directory: just skip it.
                }
            }
        }
        return candidates;
    }

    private static List<String> toolkitRoots() {
        List<String> roots = new ArrayList<>();
        String cudaPath = System.getenv("CUDA_PATH");
        if (cudaPath != null && !cudaPath.isBlank()) {
            roots.add(cudaPath);
        }
        Path toolkitRoot = Path.of("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
        if (Files.isDirectory(toolkitRoot)) {
            try (Stream<Path> versions = Files.list(toolkitRoot)) {
                versions.filter(Files::isDirectory)
                        .sorted((a, b) -> b.getFileName().toString().compareTo(a.getFileName().toString()))
                        .forEach(path -> roots.add(path.toString()));
            } catch (Exception ignored) {
                // Fall through with whatever we already have.
            }
        }
        return roots;
    }

    /** The NVRTC library that was loaded, for diagnostics. */
    public Path libraryPath() {
        return this.libraryPath;
    }

    /**
     * Compiles CUDA C++ to PTX.
     *
     * @param source             the translation unit
     * @param programName        a name used in diagnostics only
     * @param computeArchitecture e.g. {@code compute_86}; the driver JITs the result for the device
     * @param options            extra NVRTC options; {@code --fmad=false} is added by the caller
     * @return NUL-terminated PTX, ready for {@code cuModuleLoadData}
     */
    public byte[] compileToPtx(String source, String programName, String computeArchitecture,
            List<String> options) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment programOut = local.allocate(ValueLayout.ADDRESS);
            MemorySegment sourceSegment = local.allocateFrom(source, StandardCharsets.UTF_8);
            MemorySegment nameSegment = local.allocateFrom(programName, StandardCharsets.UTF_8);
            check(invoke(nvrtcCreateProgram, programOut, sourceSegment, nameSegment, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");

            MemorySegment program = programOut.get(ValueLayout.ADDRESS, 0);
            try {
                List<String> allOptions = new ArrayList<>(options);
                allOptions.add("--gpu-architecture=" + computeArchitecture);

                MemorySegment optionArray = local.allocate(ValueLayout.ADDRESS, allOptions.size());
                for (int i = 0; i < allOptions.size(); i++) {
                    optionArray.setAtIndex(ValueLayout.ADDRESS, i,
                            local.allocateFrom(allOptions.get(i), StandardCharsets.UTF_8));
                }

                int compileResult = invoke(nvrtcCompileProgram, program, allOptions.size(), optionArray);
                if (compileResult != NVRTC_SUCCESS) {
                    throw new CudaException(compileResult, errorName(compileResult),
                            "nvrtcCompileProgram failed for " + programName + ": " + programLog(program, local));
                }

                MemorySegment sizeOut = local.allocate(ValueLayout.JAVA_LONG);
                check(invoke(nvrtcGetPTXSize, program, sizeOut), "nvrtcGetPTXSize");
                long size = sizeOut.get(ValueLayout.JAVA_LONG, 0);
                if (size <= 0) {
                    throw new CudaException(CudaException.BINDING_FAILURE, "NVRTC_EMPTY_PTX",
                            "nvrtcGetPTXSize reported " + size + " bytes");
                }
                MemorySegment ptx = local.allocate(size);
                check(invoke(nvrtcGetPTX, program, ptx), "nvrtcGetPTX");
                return ptx.asSlice(0, size).toArray(ValueLayout.JAVA_BYTE);
            } finally {
                invoke(nvrtcDestroyProgram, programOut);
            }
        }
    }

    private String programLog(MemorySegment program, Arena arena) {
        try {
            MemorySegment sizeOut = arena.allocate(ValueLayout.JAVA_LONG);
            if (invoke(nvrtcGetProgramLogSize, program, sizeOut) != NVRTC_SUCCESS) {
                return "<no log available>";
            }
            long size = sizeOut.get(ValueLayout.JAVA_LONG, 0);
            if (size <= 1) {
                return "<empty log>";
            }
            MemorySegment log = arena.allocate(size);
            if (invoke(nvrtcGetProgramLog, program, log) != NVRTC_SUCCESS) {
                return "<no log available>";
            }
            return log.getString(0, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            return "<no log available>";
        }
    }

    private String errorName(int code) {
        try {
            MemorySegment text = (MemorySegment) nvrtcGetErrorString.invokeExact(code);
            if (text.address() == 0L) {
                return "NVRTC_ERROR(" + code + ")";
            }
            return text.reinterpret(128).getString(0, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            return "NVRTC_ERROR(" + code + ")";
        }
    }

    private void check(int result, String operation) {
        if (result != NVRTC_SUCCESS) {
            throw new CudaException(result, errorName(result), operation + " failed: " + errorName(result));
        }
    }

    private static int invoke(MethodHandle handle, Object... arguments) {
        try {
            return (int) handle.invokeWithArguments(arguments);
        } catch (CudaException e) {
            throw e;
        } catch (Throwable t) {
            throw new CudaException(CudaException.BINDING_FAILURE, "CUDA_ERROR_BINDING",
                    "NVRTC entry point could not be invoked", t);
        }
    }

    @Override
    public void close() {
        this.arena.close();
    }
}
