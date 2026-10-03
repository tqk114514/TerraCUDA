package tqk114514.terracuda.chunk;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Heightmap;

/**
 * Writes a chunk's blocks from an array of block state ids.
 *
 * <p>This is the step the design doc calls the new bottleneck: once the density is cheap, writing the
 * blocks is what is left. It goes through {@link LevelChunkSection#setBlockState(int, int, int, BlockState, boolean)}
 * with threading checks off, which is what vanilla's own {@code doFill} does — the chunk-level
 * {@code setBlockState} would add locks, unsaved tracking and a heightmap predicate per block on top.
 *
 * <p>Air is skipped rather than written: a fresh section is air, and vanilla skips it too.
 *
 * <p>Layout of {@code stateIds} is {@code (x * 16 + z) * height + (y - minY)}, the same as the device
 * emits and the same as {@code VanillaChunkReference} produces, so the two can be compared directly.
 */
public final class ChunkReplay {

    private ChunkReplay() {
    }

    /** Writes {@code stateIds} into {@code chunk} and brings its two worldgen heightmaps up to date. */
    public static void write(ChunkAccess chunk, int[] stateIds, int minY, int height) {
        if (stateIds.length != 16 * 16 * height) {
            throw new IllegalArgumentException("expected " + (16 * 16 * height)
                    + " block ids, got " + stateIds.length);
        }

        Heightmap oceanFloor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);
        Heightmap worldSurface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);

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
                    BlockState state = Block.stateById(stateIds[(x * 16 + z) * height + yLocal]);
                    if (state.isAir()) {
                        continue;
                    }
                    section.setBlockState(x, yInSection, z, state, false);
                    oceanFloor.update(x, y, z, state);
                    worldSurface.update(x, y, z, state);
                }
            }
        }
    }
}
