package tqk114514.terracuda.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Loads the TerraCUDA kernel module into a {@link CudaContext}, choosing the best available image.
 *
 * <p>The ladder mirrors the design doc's degradation matrix, best first:
 * <ol>
 *   <li>a cubin prebuilt for the device's exact compute capability — no JIT, fastest startup;</li>
 *   <li>a prebuilt PTX image, which the driver JITs for whatever device is present;</li>
 *   <li>the embedded CUDA source compiled through NVRTC at runtime, for machines with a driver and a
 *       GPU but no host compiler.</li>
 * </ol>
 *
 * <p>Rungs 1 and 2 are produced by the build when {@code nvcc} and a host compiler are present. When
 * they are not, rung 3 is what makes the GPU path work at all — see {@link NvrtcCompiler}.
 *
 * <p>Every image comes from the jar, so the kernel the mod runs always matches the kernel it was
 * built with.
 */
public final class CudaKernels {

    /** Which image the module was loaded from. */
    public enum Origin {
        /** A cubin compiled for this exact compute capability. */
        CUBIN,
        /** A prebuilt PTX image, JIT-compiled by the driver. */
        PTX,
        /** CUDA C++ compiled at runtime by NVRTC. */
        NVRTC
    }

    /**
     * @param origin how the module was produced
     * @param detail a human-readable note for the log, e.g. the resource path or NVRTC library name
     */
    public record LoadedModule(Origin origin, String detail) {
    }

    private static final String RESOURCE_DIR = "/META-INF/terracuda/cuda/";
    private static final String PROGRAM_NAME = "terracuda.cu";
    private static final String CUBIN_PREFIX = "terracuda_sm";
    private static final String PTX_NAME = "terracuda.ptx";

    /**
     * The options handed to NVRTC. {@code --fmad=false} is non-negotiable: with FMA contraction the
     * kernels would compute a different, better-rounded answer than Java does, and the terrain would
     * diverge. {@code --use_fast_math} must never be added.
     */
    private static final List<String> NVRTC_OPTIONS = List.of("--fmad=false");

    private CudaKernels() {
    }

    /**
     * Loads the kernel module into {@code context}, picking the best image for {@code device}.
     *
     * @throws CudaException when no image can be produced at all
     */
    public static LoadedModule loadModule(CudaContext context, CudaDeviceInfo device) {
        String cubinPath = RESOURCE_DIR + CUBIN_PREFIX + device.computeCapability() + ".cubin";
        byte[] cubin = readResource(cubinPath);
        if (cubin != null) {
            context.loadModule(cubin);
            return new LoadedModule(Origin.CUBIN, cubinPath + " (" + cubin.length + " bytes)");
        }

        String ptxPath = RESOURCE_DIR + PTX_NAME;
        byte[] ptx = readResource(ptxPath);
        if (ptx != null) {
            context.loadModule(nulTerminated(ptx));
            return new LoadedModule(Origin.PTX, ptxPath + " (" + ptx.length + " bytes)");
        }

        return loadFromSource(context, device);
    }

    private static LoadedModule loadFromSource(CudaContext context, CudaDeviceInfo device) {
        String sourcePath = RESOURCE_DIR + PROGRAM_NAME;
        String source = readResourceAsString(sourcePath);
        if (source == null) {
            throw new CudaException(CudaException.BINDING_FAILURE, "TERRACUDA_NO_KERNEL_IMAGE",
                    "no kernel image on the classpath: neither " + RESOURCE_DIR + CUBIN_PREFIX
                            + device.computeCapability() + ".cubin, " + RESOURCE_DIR + PTX_NAME
                            + ", nor " + sourcePath + " is present");
        }

        Optional<NvrtcCompiler> compiler = NvrtcCompiler.tryLoad();
        if (compiler.isEmpty()) {
            throw new CudaException(CudaException.BINDING_FAILURE, "TERRACUDA_NO_NVRTC",
                    "no prebuilt kernel image and NVRTC is not installed; "
                            + "install the CUDA Toolkit or build the mod with nvcc");
        }

        try (NvrtcCompiler nvrtc = compiler.get()) {
            byte[] compiled = nvrtc.compileToPtx(source, PROGRAM_NAME,
                    "compute_" + device.computeCapability(), NVRTC_OPTIONS);
            context.loadModule(compiled);
            return new LoadedModule(Origin.NVRTC,
                    "NVRTC " + nvrtc.libraryPath().getFileName() + " -> compute_"
                            + device.computeCapability() + " PTX (" + compiled.length + " bytes)");
        }
    }

    /** {@code cuModuleLoadData} wants a NUL-terminated image for PTX text. */
    private static byte[] nulTerminated(byte[] image) {
        if (image.length > 0 && image[image.length - 1] == 0) {
            return image;
        }
        return Arrays.copyOf(image, image.length + 1);
    }

    private static byte[] readResource(String path) {
        try (InputStream stream = CudaKernels.class.getResourceAsStream(path)) {
            return stream == null ? null : stream.readAllBytes();
        } catch (IOException e) {
            return null;
        }
    }

    private static String readResourceAsString(String path) {
        byte[] bytes = readResource(path);
        return bytes == null ? null : new String(bytes, StandardCharsets.UTF_8);
    }
}
