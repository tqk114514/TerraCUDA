package tqk114514.terracuda.cuda;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Tests for the raw CUDA driver binding.
 *
 * <p>Everything hardware-dependent is guarded by an assumption so the suite stays green on a machine
 * (or CI runner) without an NVIDIA driver, which is the normal state for the fallback path anyway.
 */
class CudaDriverTest {

    private static Optional<CudaDriver> loadedDriver() {
        return CudaDriver.tryLoad();
    }

    @Test
    void loadingIsTotalAndNeverThrows() {
        Optional<CudaDriver> driver = assertDoesNotThrow(CudaDriver::tryLoad);
        assertNotNull(driver);
        driver.ifPresent(CudaDriver::close);
    }

    @Test
    void initReturnsSuccessOrANamedError() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = maybe.get()) {
            int result = driver.init();
            if (result != CudaDriver.CUDA_SUCCESS) {
                // A present library that refuses to init is still a legitimate environment.
                assertFalse(driver.errorName(result).isBlank());
            } else {
                assertEquals(CudaDriver.CUDA_SUCCESS, result);
            }
        }
    }

    @Test
    void errorNameDecodesKnownResults() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = maybe.get()) {
            assertEquals("CUDA_SUCCESS", driver.errorName(CudaDriver.CUDA_SUCCESS));
            assertTrue(driver.errorString(CudaDriver.CUDA_SUCCESS).toLowerCase().contains("no error"));
        }
    }

    @Test
    void checkRaisesWithTheRawCodeForAFailure() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = maybe.get()) {
            CudaException failure = assertThrows(CudaException.class,
                    () -> driver.check(CudaDriver.CUDA_ERROR_INVALID_VALUE, "unit test"));
            assertEquals(CudaDriver.CUDA_ERROR_INVALID_VALUE, failure.code());
            assertEquals("CUDA_ERROR_INVALID_VALUE", failure.errorName());
            assertTrue(failure.getMessage().contains("unit test"));

            assertDoesNotThrow(() -> driver.check(CudaDriver.CUDA_SUCCESS, "unit test"));
        }
    }

    @Test
    void readingAnOutOfRangeDeviceFailsWithANamedError() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = maybe.get()) {
            assumeTrue(driver.init() == CudaDriver.CUDA_SUCCESS, "cuInit did not succeed");

            CudaException failure = assertThrows(CudaException.class, () -> driver.deviceHandle(9999));
            assertFalse(failure.errorName().isBlank());
            assertTrue(failure.code() != CudaDriver.CUDA_SUCCESS);
        }
    }

    @Test
    void driverVersionIsReportedWhenPresent() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        try (CudaDriver driver = maybe.get()) {
            assumeTrue(driver.init() == CudaDriver.CUDA_SUCCESS, "cuInit did not succeed");
            assertTrue(driver.driverVersion() > 0);
        }
    }

    @Test
    void closingTwiceIsHarmless() {
        Optional<CudaDriver> maybe = loadedDriver();
        assumeTrue(maybe.isPresent(), "no CUDA driver library on this machine");
        CudaDriver driver = maybe.get();
        assertDoesNotThrow(() -> {
            driver.close();
            driver.close();
        });
    }
}
