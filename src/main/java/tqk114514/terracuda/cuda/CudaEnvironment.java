package tqk114514.terracuda.cuda;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Result of probing the machine for usable CUDA hardware.
 *
 * <p>This is the single entry point the rest of the mod uses to decide whether the GPU path may be
 * taken. {@link #detect()} is contractually total: it never throws, and an unavailable environment
 * always carries a human-readable {@link #reason()} for the log. That keeps the "no NVIDIA card"
 * case to a single INFO line instead of a crash.
 *
 * @param available     whether a driver, a context-less device list and at least one device exist
 * @param reason        why CUDA is unavailable; blank when {@link #available()} is {@code true}
 * @param driverVersion raw {@code cuDriverGetVersion} value (e.g. {@code 12060}), or {@code 0}
 * @param devices       detected devices, empty when unavailable
 */
public record CudaEnvironment(
        boolean available,
        String reason,
        int driverVersion,
        List<CudaDeviceInfo> devices) {

    public CudaEnvironment {
        devices = List.copyOf(devices);
        if (available && devices.isEmpty()) {
            throw new IllegalArgumentException("an available CUDA environment must report at least one device");
        }
        if (!available && reason == null) {
            throw new IllegalArgumentException("an unavailable CUDA environment must carry a reason");
        }
    }

    /** An environment with no usable device and an explanatory reason. */
    public static CudaEnvironment unavailable(String reason) {
        return new CudaEnvironment(false, reason, 0, List.of());
    }

    /**
     * Probes the machine. Never throws; any failure becomes an {@link #unavailable(String)} result.
     */
    public static CudaEnvironment detect() {
        Optional<CudaDriver> loaded = CudaDriver.tryLoad();
        if (loaded.isEmpty()) {
            return unavailable("CUDA driver library '" + CudaDriver.LIBRARY_NAME + "' is not available");
        }
        try (CudaDriver driver = loaded.get()) {
            int initResult = driver.init();
            if (initResult != CudaDriver.CUDA_SUCCESS) {
                return unavailable("cuInit failed: " + driver.errorName(initResult));
            }
            int version = driver.driverVersion();
            int count = driver.deviceCount();
            if (count <= 0) {
                return new CudaEnvironment(false, "the driver reported no CUDA devices", version, List.of());
            }
            List<CudaDeviceInfo> devices = new ArrayList<>(count);
            for (int index = 0; index < count; index++) {
                devices.add(readDevice(driver, index));
            }
            return new CudaEnvironment(true, "", version, devices);
        } catch (CudaException e) {
            return unavailable(e.errorName() + ": " + e.getMessage());
        } catch (RuntimeException e) {
            return unavailable("unexpected failure while probing CUDA: " + e);
        }
    }

    private static CudaDeviceInfo readDevice(CudaDriver driver, int index) {
        int handle = driver.deviceHandle(index);
        return new CudaDeviceInfo(
                index,
                driver.deviceName(handle),
                driver.deviceAttribute(handle, CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                driver.deviceAttribute(handle, CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
                driver.deviceTotalMem(handle),
                driver.deviceAttribute(handle, CudaDriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
    }

    /** First device by ordinal, if any. */
    public Optional<CudaDeviceInfo> firstDevice() {
        return devices.isEmpty() ? Optional.empty() : Optional.of(devices.get(0));
    }

    /** Driver version rendered as {@code "12.6"}. */
    public String driverVersionString() {
        return (driverVersion / 1000) + "." + ((driverVersion % 1000) / 10);
    }

    /** One-line description suitable for a log message. */
    public String summary() {
        if (!available) {
            return "unavailable (" + reason + ")";
        }
        StringBuilder builder = new StringBuilder()
                .append("driver ").append(driverVersionString())
                .append(", ").append(devices.size()).append(" device(s): ");
        for (int i = 0; i < devices.size(); i++) {
            CudaDeviceInfo device = devices.get(i);
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(device.name())
                    .append(" [sm_").append(device.computeCapability())
                    .append(", ").append(device.totalMemoryMiB()).append(" MiB]");
        }
        return builder.toString();
    }
}
