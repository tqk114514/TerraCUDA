package tqk114514.terracuda.worldgen;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;

/**
 * The M3 reference harness has to be trustworthy before it can be used as an oracle.
 *
 * <p>These checks are deliberately about shape and determinism rather than exact block counts: the
 * counts depend on the seed, and pinning them would turn a real terrain change into a test failure
 * without saying anything useful.
 */
class VanillaChunkReferenceTest {

    @Test
    void generatesOneWholeChunk() {
        RandomState randomState = OverworldFixture.randomState();
        VanillaChunkReference.ChunkBlocks chunk = VanillaChunkReference.generate(randomState, 0, 0);

        assertEquals(-64, chunk.minY());
        assertEquals(384, chunk.height());
        assertEquals(16 * 16 * 384, chunk.size());
    }

    @Test
    void producesTerrainRatherThanAFilledCube() {
        VanillaChunkReference.ChunkBlocks chunk =
                VanillaChunkReference.generate(OverworldFixture.randomState(), 0, 0);

        Map<Integer, Integer> histogram = new HashMap<>();
        for (int id : chunk.stateIds()) {
            histogram.merge(id, 1, Integer::sum);
        }

        int air = histogram.getOrDefault(Block.getId(Blocks.AIR.defaultBlockState()), 0);
        int stone = histogram.getOrDefault(Block.getId(Blocks.STONE.defaultBlockState()), 0);

        assertTrue(air > 0, "a chunk should have sky in it");
        assertTrue(stone > 0, "a chunk should have ground in it");
        assertEquals(chunk.size(), air + stone + histogram.entrySet().stream()
                .filter(e -> e.getKey() != Block.getId(Blocks.AIR.defaultBlockState())
                        && e.getKey() != Block.getId(Blocks.STONE.defaultBlockState()))
                .mapToInt(Map.Entry::getValue).sum());

        // The highest block is sky and the ground is well below it: a chunk that is one solid cube
        // means the harness is driving the interpolation loop wrong.
        assertEquals(Block.getId(Blocks.AIR.defaultBlockState()), chunk.at(8, 319, 8));
        assertTrue(air > 0 && stone > 0);
        assertTrue(stone < chunk.size() / 2, "a chunk is mostly sky above the surface");
    }

    @Test
    void isDeterministic() {
        RandomState randomState = OverworldFixture.randomState();
        VanillaChunkReference.ChunkBlocks first = VanillaChunkReference.generate(randomState, 3, -7);
        VanillaChunkReference.ChunkBlocks second = VanillaChunkReference.generate(randomState, 3, -7);

        assertArrayEquals(first.stateIds(), second.stateIds());
    }

    @Test
    void differentChunksDiffer() {
        RandomState randomState = OverworldFixture.randomState();
        VanillaChunkReference.ChunkBlocks a = VanillaChunkReference.generate(randomState, 0, 0);
        VanillaChunkReference.ChunkBlocks b = VanillaChunkReference.generate(randomState, 40, 40);

        boolean differs = false;
        for (int i = 0; i < a.size() && !differs; i++) {
            differs = a.stateIds()[i] != b.stateIds()[i];
        }
        assertTrue(differs, "two distant chunks should not be identical");
    }
}
