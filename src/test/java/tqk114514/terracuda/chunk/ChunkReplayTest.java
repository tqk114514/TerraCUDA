package tqk114514.terracuda.chunk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.core.Holder;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.IdMap;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.PalettedContainer;
import net.minecraft.world.level.chunk.PalettedContainerFactory;
import net.minecraft.world.level.chunk.ProtoChunk;
import net.minecraft.world.level.chunk.Strategy;
import net.minecraft.world.level.chunk.UpgradeData;
import net.minecraft.world.level.levelgen.Heightmap;
import org.junit.jupiter.api.Test;

/**
 * The write-back into a chunk: the one piece of the chain that had no test, and the reason the mixin
 * waited before taking over generation.
 *
 * <p>Nothing here needs a device. {@link ChunkReplay} takes an array of block state ids and writes it,
 * so the test can hand it a synthetic chunk — a stone floor, a few layers of water, air above — and
 * check what came back out. That is deliberate: the pieces either side of the write are covered by
 * bit-exact tests against vanilla, and what was untested was the writing itself.
 *
 * <p>Two things are checked beyond the block states. The heightmaps, because {@code doFill} maintains
 * them and the surface stage reads them. And the fluid-update flags, because they are what makes
 * aquifer water flow into the caves beside it; a chunk with the right blocks and the wrong flags looks
 * correct and behaves wrong.
 *
 * <p><b>Building the chunk.</b> {@code ProtoChunk} needs a {@link PalettedContainerFactory}, which
 * normally comes from a {@code RegistryAccess} carrying the datapack biome registry. There is no way
 * to build one of those in a unit test — {@code VanillaRegistries} returns a {@code HolderLookup}
 * rather than a {@code RegistryAccess}, and the static registries have no biome registry at all. So
 * the factory is assembled by hand with a one-entry biome stub. Nothing in this test reads biomes, so
 * the stub is never exercised; it exists because the record insists on having one.
 */
class ChunkReplayTest {

    private static final int MIN_Y = -64;
    private static final int HEIGHT = 384;
    private static final int COUNT = 16 * 16 * HEIGHT;

    /** Everything below this is stone. */
    private static final int STONE_TOP = 60;
    /** Water fills up to just under this. */
    private static final int WATER_TOP = 64;

    @Test
    void blocksHeightmapsAndFluidFlagsAllLand() {
        ProtoChunk chunk = newChunk();
        EmittedChunk emitted = syntheticChunk();

        ChunkReplay.write(chunk, emitted, MIN_Y, HEIGHT);

        int stone = Block.getId(Blocks.STONE.defaultBlockState());
        int water = Block.getId(Blocks.WATER.defaultBlockState());
        int air = Block.getId(Blocks.AIR.defaultBlockState());

        // The block states, at the corners and at the layer boundaries where an off-by-one would hide.
        for (int[] column : new int[][] {{0, 0}, {15, 15}, {7, 3}, {0, 15}}) {
            int x = column[0];
            int z = column[1];
            assertEquals(stone, Block.getId(chunk.getBlockState(new net.minecraft.core.BlockPos(x, MIN_Y, z))),
                    "bottom of the column should be stone");
            assertEquals(stone, Block.getId(chunk.getBlockState(
                    new net.minecraft.core.BlockPos(x, MIN_Y + STONE_TOP - 1, z))),
                    "the last stone layer");
            assertEquals(water, Block.getId(chunk.getBlockState(
                    new net.minecraft.core.BlockPos(x, MIN_Y + STONE_TOP, z))),
                    "the first water layer");
            assertEquals(water, Block.getId(chunk.getBlockState(
                    new net.minecraft.core.BlockPos(x, MIN_Y + WATER_TOP - 1, z))),
                    "the last water layer");
            assertEquals(air, Block.getId(chunk.getBlockState(
                    new net.minecraft.core.BlockPos(x, MIN_Y + WATER_TOP, z))),
                    "air above the water");
            assertEquals(air, Block.getId(chunk.getBlockState(
                    new net.minecraft.core.BlockPos(x, MIN_Y + HEIGHT - 1, z))),
                    "the top of the chunk");
        }

        // WORLD_SURFACE_WG counts any non-air block, so it stops on the water.
        // OCEAN_FLOOR_WG wants something that blocks motion, so it stops on the stone below.
        Heightmap surface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);
        Heightmap floor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                assertEquals(MIN_Y + WATER_TOP, surface.getFirstAvailable(x, z),
                        "WORLD_SURFACE_WG at (" + x + ", " + z + ")");
                assertEquals(MIN_Y + STONE_TOP, floor.getFirstAvailable(x, z),
                        "OCEAN_FLOOR_WG at (" + x + ", " + z + ")");
            }
        }

        // The fluid flags, which the synthetic chunk sets on the top water layer only.
        int flagged = 0;
        var postProcessing = chunk.getPostProcessing();
        for (int i = 0; i < COUNT; i++) {
            if (!emitted.fluidUpdate(i)) {
                continue;
            }
            flagged++;
        }
        assertEquals(16 * 16, flagged, "the top water layer is the flagged one");

        int queued = 0;
        for (var list : postProcessing) {
            if (list != null) {
                queued += list.size();
            }
        }
        assertEquals(flagged, queued,
                "every flagged block should have been queued for post-processing");

        // y = 63 is the top water layer, and section 7 is the one that contains it.
        int topWaterSection = chunk.getSectionIndex(MIN_Y + WATER_TOP - 1);
        assertNotNull(postProcessing[topWaterSection],
                "the section holding the flagged water should have a post-processing list");
        assertEquals(256, postProcessing[topWaterSection].size());
    }

    @Test
    void airIsNotWrittenAndDoesNotReachTheHeightmap() {
        ProtoChunk chunk = newChunk();
        // Every block air: the chunk should stay empty rather than be filled with anything.
        int air = Block.getId(Blocks.AIR.defaultBlockState());
        int[] ids = new int[COUNT];
        java.util.Arrays.fill(ids, air);

        ChunkReplay.write(chunk, new EmittedChunk(ids, new long[(COUNT + 63) >>> 6]), MIN_Y, HEIGHT);

        assertEquals(air, Block.getId(chunk.getBlockState(new net.minecraft.core.BlockPos(0, 0, 0))));
        // Heightmap.getFirstAvailable returns minY when nothing is there.
        assertEquals(MIN_Y,
                chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG)
                        .getFirstAvailable(0, 0));
    }

    @Test
    void aWrongSizedArrayIsRejected() {
        ProtoChunk chunk = newChunk();
        try {
            ChunkReplay.write(chunk, new EmittedChunk(new int[16], new long[1]), MIN_Y, HEIGHT);
            throw new AssertionError("expected a size check");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("block ids"), expected.getMessage());
        }
    }

    /** A chunk with the overworld's height and nothing in it. */
    private static ProtoChunk newChunk() {
        Bootstrap.bootStrap();
        return new ProtoChunk(new ChunkPos(3, -7), UpgradeData.EMPTY,
                LevelHeightAccessor.create(MIN_Y, HEIGHT), paletteFactory(), null);
    }

    /**
     * Stone up to {@link #STONE_TOP}, water up to {@link #WATER_TOP}, air above, with the top water
     * layer flagged for a fluid tick.
     */
    private static EmittedChunk syntheticChunk() {
        int stone = Block.getId(Blocks.STONE.defaultBlockState());
        int water = Block.getId(Blocks.WATER.defaultBlockState());
        int air = Block.getId(Blocks.AIR.defaultBlockState());

        int[] ids = new int[COUNT];
        long[] flags = new long[(COUNT + 63) >>> 6];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int yLocal = 0; yLocal < HEIGHT; yLocal++) {
                    int y = MIN_Y + yLocal;
                    int index = (x * 16 + z) * HEIGHT + yLocal;
                    if (y < MIN_Y + STONE_TOP) {
                        ids[index] = stone;
                    } else if (y < MIN_Y + WATER_TOP) {
                        ids[index] = water;
                        if (y == MIN_Y + WATER_TOP - 1) {
                            flags[index >>> 6] |= 1L << (index & 63);
                        }
                    } else {
                        ids[index] = air;
                    }
                }
            }
        }
        return new EmittedChunk(ids, flags);
    }

    /**
     * A palette factory built by hand.
     *
     * <p>The block half is the real thing — {@code Block.BLOCK_STATE_REGISTRY} is the game's own state
     * registry. The biome half is a single-entry stub around the real {@code plains} holder, because
     * there is no biome registry to be had without a datapack load.
     */
    private static PalettedContainerFactory paletteFactory() {
        HolderLookup.RegistryLookup<Biome> biomes =
                VanillaRegistries.createLookup().lookupOrThrow(Registries.BIOME);
        Holder.Reference<Biome> plains = biomes.getOrThrow(Biomes.PLAINS);

        IdMap<Holder<Biome>> biomeIds = new IdMap<>() {
            @Override
            public int getId(Holder<Biome> value) {
                return 0;
            }

            @Override
            public Holder<Biome> byId(int id) {
                return plains;
            }

            @Override
            public int size() {
                return 1;
            }

            @Override
            public java.util.Iterator<Holder<Biome>> iterator() {
                return java.util.List.of((Holder<Biome>) plains).iterator();
            }
        };

        Strategy<BlockState> blocks = Strategy.createForBlockStates(Block.BLOCK_STATE_REGISTRY);
        Strategy<Holder<Biome>> biomeStrategy = Strategy.createForBiomes(biomeIds);
        BlockState air = Blocks.AIR.defaultBlockState();
        return new PalettedContainerFactory(blocks, air,
                PalettedContainer.codecRW(BlockState.CODEC, blocks, air),
                biomeStrategy, plains,
                // Only read when a chunk is serialised, which this test does not do. The real one
                // comes from Registry.holderByNameCodec(), and there is no biome registry here.
                null);
    }
}
