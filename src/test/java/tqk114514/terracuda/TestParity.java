package tqk114514.terracuda;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Helpers for asserting <em>bit</em> identity against vanilla.
 *
 * <p>{@code assertEquals(double, double)} already compares with {@code Double.compare}, but comparing
 * the raw bits makes the intent explicit and catches {@code -0.0} vs {@code 0.0} differences, which
 * would silently change terrain.
 */
public final class TestParity {

    private TestParity() {
    }

    public static void assertDoubleIdentical(double expected, double actual, String context) {
        assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual),
                () -> context + ": expected " + expected + " (" + Double.doubleToRawLongBits(expected)
                        + ") but got " + actual + " (" + Double.doubleToRawLongBits(actual) + ")");
    }

    public static void assertFloatIdentical(float expected, float actual, String context) {
        assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual),
                () -> context + ": expected " + expected + " but got " + actual);
    }
}
