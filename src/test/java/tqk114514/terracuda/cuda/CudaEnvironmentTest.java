package tqk114514.terracuda.cuda;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class CudaEnvironmentTest {

    @Test
    void detectIsTotalAndSelfConsistent() {
        CudaEnvironment environment = assertDoesNotThrow(CudaEnvironment::detect);

        if (environment.available()) {
            assertFalse(environment.devices().isEmpty(), "an available environment reports devices");
            assertTrue(environment.driverVersion() > 0, "an available environment reports a driver version");
            assertTrue(environment.firstDevice().isPresent());
            assertTrue(environment.summary().contains("driver "));
        } else {
            assertFalse(environment.reason().isBlank(), "an unavailable environment explains itself");
            assertTrue(environment.devices().isEmpty());
            assertTrue(environment.summary().contains(environment.reason()));
        }
    }

    @Test
    void detectedDevicesCarryPlausibleHardwareFacts() {
        CudaEnvironment environment = CudaEnvironment.detect();
        assumeTrue(environment.available(), "no CUDA device on this machine");

        for (CudaDeviceInfo device : environment.devices()) {
            assertFalse(device.name().isBlank());
            assertTrue(device.computeMajor() >= 3, "compute capability 3.0 is the floor for CUDA 12+");
            assertTrue(device.computeMinor() >= 0);
            assertTrue(device.multiprocessorCount() > 0);
            assertTrue(device.totalMemoryBytes() > 0);
            assertEquals(device.computeMajor() * 10 + device.computeMinor(), device.computeCapability());
            assertEquals(device.computeMajor() + "." + device.computeMinor(), device.computeCapabilityString());
            assertTrue(device.totalMemoryMiB() > 0);
        }
    }

    @Test
    void unavailableEnvironmentIsWellFormed() {
        CudaEnvironment environment = CudaEnvironment.unavailable("no NVIDIA driver");

        assertFalse(environment.available());
        assertEquals("no NVIDIA driver", environment.reason());
        assertEquals(0, environment.driverVersion());
        assertTrue(environment.devices().isEmpty());
        assertTrue(environment.firstDevice().isEmpty());
        assertEquals("unavailable (no NVIDIA driver)", environment.summary());
    }

    @Test
    void availableEnvironmentRequiresAtLeastOneDevice() {
        CudaDeviceInfo device = new CudaDeviceInfo(0, "Fake GPU", 8, 6, 1024L * 1024L * 1024L, 68);
        assertThrows(IllegalArgumentException.class, () -> new CudaEnvironment(true, "", 12060, List.of()));
        assertDoesNotThrow(() -> new CudaEnvironment(true, "", 12060, List.of(device)));
    }

    @Test
    void unavailableEnvironmentRequiresAReason() {
        assertThrows(IllegalArgumentException.class, () -> new CudaEnvironment(false, null, 0, List.of()));
    }

    @Test
    void driverVersionIsRenderedAsMajorDotMinor() {
        CudaDeviceInfo device = new CudaDeviceInfo(0, "Fake GPU", 8, 6, 1L, 1);
        assertEquals("12.6", new CudaEnvironment(true, "", 12060, List.of(device)).driverVersionString());
        assertEquals("11.8", new CudaEnvironment(true, "", 11080, List.of(device)).driverVersionString());
        assertEquals("13.0", new CudaEnvironment(true, "", 13000, List.of(device)).driverVersionString());
    }

    @Test
    void deviceInfoIsImmutable() {
        CudaDeviceInfo device = new CudaDeviceInfo(0, "Fake GPU", 8, 6, 1024L, 68);
        CudaEnvironment environment = new CudaEnvironment(true, "", 12060, List.of(device));

        assertThrows(UnsupportedOperationException.class, () -> environment.devices().add(device));
    }
}
