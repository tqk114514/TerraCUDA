package tqk114514.terracuda.chunk;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.NoiseSettings;

/**
 * Writes a chunk's blocks from what the device path emitted.
 *
 * <p>This is the step the design doc calls the new bottleneck: once the density is cheap, writing the
 * blocks is what is left. It goes through {@link LevelChunkSection#setBlockState(int, int, int, BlockState, boolean)}
 * with threading checks off, which is what vanilla's own {@code doFill} does — the chunk-level
 * {@code setBlockState} would add locks, unsaved tracking and a heightmap predicate per block on top.
 *
 * <p>Three things happen here, and all three are in vanilla's {@code doFill}: the block state, the two
 * worldgen heightmaps, and {@code markPosForPostprocessing} for the positions the aquifer flagged. The
 * last one is easy to leave out and is the difference between water that flows and water that sits.
 *
 * <p>Air is skipped rather than written, matching {@code doFill}: a fresh section is already air.
 */
public final class ChunkReplay {

    /** {@code doFill} compares against this exact state, not {@code isAir}. */
    private static final BlockState AIR = Blocks.AIR.defaultBlockState();

    private ChunkReplay() {
    }

    /** Writes {@code emitted} into {@code chunk}, bringing its worldgen heightmaps up to date. */
    public static void write(ChunkAccess chunk, EmittedChunk emitted, NoiseSettings settings) {
        int minY = settings.minY();
        int height = settings.height();
        int[] stateIds = emitted.stateIds();
        if (stateIds.length != 16 * 16 * height) {
            throw new IllegalArgumentException("expected " + (16 * 16 * height)
                    + " block ids, got " + stateIds.length);
        }

        Heightmap oceanFloor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);
        Heightmap worldSurface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);
        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();

        // The flagged positions are collected rather than marked where they are found, because the
        // order matters and this loop is not the order vanilla uses. doFill walks cells, and each
        // section's post-processing list is appended to in that walk's order; this loop walks y
        // first. The contents come out identical either way, the sequence does not, and the sequence
        // is what ends up in the chunk's NBT.
        int[] flagged = new int[16];
        int flaggedCount = 0;

        LevelChunkSection section = null;
        int sectionIndex = Integer.MIN_VALUE;

        for (int yLocal = 0; yLocal < height; yLocal++) {
            int y = minY + yLocal;
            int index = chunk.getSectionIndex(y);
            if (index != sectionIndex) {
                sectionIndex = index;
                section = chunk.getSection(index);
            }
            int yInSection = y & 15;

            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    int blockIndex = (x * 16 + z) * height + yLocal;
                    BlockState state = Block.stateById(stateIds[blockIndex]);
                    if (state == AIR) {
                        continue;
                    }
                    section.setBlockState(x, yInSection, z, state, false);
                    oceanFloor.update(x, y, z, state);
                    worldSurface.update(x, y, z, state);
                    if (emitted.fluidUpdate(blockIndex)) {
                        if (flaggedCount == flagged.length) {
                            flagged = java.util.Arrays.copyOf(flagged, flaggedCount * 2);
                        }
                        flagged[flaggedCount++] = blockIndex;
                    }
                }
            }
        }

        markInCellOrder(chunk, flagged, flaggedCount, settings, minY, height, pos);
    }

    /**
     * Queues the flagged positions in {@code doFill}'s order.
     *
     * <p>Cell x ascending, then cell z ascending, then cell y descending, then y within the cell
     * descending, then x within the cell ascending, then z within the cell ascending. Any other
     * order puts the same positions in each section's list and in a different sequence.
     */
    private static void markInCellOrder(ChunkAccess chunk, int[] flagged, int count,
            NoiseSettings settings, int minY, int height, BlockPos.MutableBlockPos pos) {
        if (count == 0) {
            return;
        }
        int cellWidth = settings.getCellWidth();
        int cellHeight = settings.getCellHeight();
        Integer[] order = new Integer[count];
        long[] sortKeys = new long[count];
        for (int i = 0; i < count; i++) {
            order[i] = i;
        }
        for (int i = 0; i < count; i++) {
            int index = flagged[i];
            int yLocal = index % height;
            int rest = index / height;
            int z = rest % 16;
            int x = rest / 16;
            sortKeys[i] = ((long) (x / cellWidth) << 48)
                    | ((long) (z / cellWidth) << 40)
                    | ((long) (1023 - yLocal / cellHeight) << 30)
                    | ((long) (1023 - yLocal % cellHeight) << 20)
                    | ((long) (x % cellWidth) << 10)
                    | (z % cellWidth);
        }
        java.util.Arrays.sort(order, (a, b) -> Long.compare(sortKeys[a], sortKeys[b]));
        for (int i = 0; i < count; i++) {
            int index = flagged[order[i]];
            int yLocal = index % height;
            int rest = index / height;
            int z = rest % 16;
            int x = rest / 16;
            chunk.markPosForPostprocessing(pos.set(x, minY + yLocal, z));
        }
    }
}
