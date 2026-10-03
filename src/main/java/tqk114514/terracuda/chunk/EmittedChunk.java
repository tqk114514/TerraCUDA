package tqk114514.terracuda.chunk;

/**
 * One chunk as the device path produces it: the block state ids, plus a bitmap of the positions that
 * want queueing for a fluid tick.
 *
 * <p>The bitmap is not optional bookkeeping. Vanilla's {@code doFill} marks a position for
 * post-processing when the aquifer says the cell may flow and the block it placed is a fluid, and that
 * is what makes aquifer water run into the caves and overhangs next to it. Leaving it out produces
 * terrain that looks right and behaves wrong — still water where vanilla would have a waterfall.
 *
 * @param stateIds     block state ids, indexed {@code (x * 16 + z) * height + (y - minY)}
 * @param fluidUpdates one bit per index of {@code stateIds}, set when that block needs a fluid tick
 */
public record EmittedChunk(int[] stateIds, long[] fluidUpdates) {

    public EmittedChunk {
        if (fluidUpdates.length != (stateIds.length + 63) >>> 6) {
            throw new IllegalArgumentException("expected " + ((stateIds.length + 63) >>> 6)
                    + " bitmap words for " + stateIds.length + " blocks, got " + fluidUpdates.length);
        }
    }

    /** How many blocks this chunk carries. */
    public int size() {
        return this.stateIds.length;
    }

    /** Whether the block at {@code index} needs a fluid tick queued for it. */
    public boolean fluidUpdate(int index) {
        return (this.fluidUpdates[index >>> 6] & (1L << (index & 63))) != 0L;
    }
}
