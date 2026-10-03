package tqk114514.terracuda.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Builds the {@code void** kernelParams} array that {@code cuLaunchKernel} expects.
 *
 * <p>The CUDA Driver API does not take the arguments by value: it takes an array of pointers, each
 * pointing at the argument's bytes. This class owns the storage for both layers and keeps it alive
 * until {@link #close()}. Every slot is padded to 8 bytes so that {@code double} and 64-bit pointer
 * arguments are naturally aligned.
 *
 * <p>Not thread-safe, and confined to the thread that created it — it owns a confined {@link Arena}.
 */
public final class KernelArguments implements AutoCloseable {

    private static final long SLOT_BYTES = 8L;

    private final Arena arena = Arena.ofConfined();
    private final int capacity;
    private final MemorySegment values;
    private final MemorySegment pointers;
    private int count;

    public KernelArguments(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.capacity = capacity;
        this.values = this.arena.allocate(SLOT_BYTES * capacity);
        this.pointers = this.arena.allocate(SLOT_BYTES * capacity);
    }

    public KernelArguments addInt(int value) {
        int slot = reserve();
        this.values.set(ValueLayout.JAVA_INT, offset(slot), value);
        return this;
    }

    public KernelArguments addLong(long value) {
        int slot = reserve();
        this.values.set(ValueLayout.JAVA_LONG, offset(slot), value);
        return this;
    }

    public KernelArguments addDouble(double value) {
        int slot = reserve();
        this.values.set(ValueLayout.JAVA_DOUBLE, offset(slot), value);
        return this;
    }

    /**
     * Adds a {@code CUdeviceptr} argument. Device pointers are plain 64-bit addresses in the Driver
     * API, so they travel as a {@code long} rather than as a host pointer.
     */
    public KernelArguments addDevicePointer(long deviceAddress) {
        return addLong(deviceAddress);
    }

    /** The {@code void**} array itself. Valid until {@link #close()}. */
    public MemorySegment pointers() {
        return this.pointers;
    }

    public int count() {
        return this.count;
    }

    private int reserve() {
        if (this.count >= this.capacity) {
            throw new IllegalStateException("KernelArguments is full (capacity " + this.capacity + ")");
        }
        int slot = this.count++;
        // Each parameter pointer must address the storage of that argument.
        this.pointers.set(ValueLayout.ADDRESS, offset(slot), this.values.asSlice(offset(slot), SLOT_BYTES));
        return slot;
    }

    private static long offset(int slot) {
        return SLOT_BYTES * slot;
    }

    @Override
    public void close() {
        this.arena.close();
    }
}
