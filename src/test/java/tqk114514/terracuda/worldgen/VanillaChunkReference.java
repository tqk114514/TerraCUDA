package tqk114514.terracuda.worldgen;

import java.lang.reflect.Method;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

/**
 * Drives vanilla's own {@link NoiseChunk} over a chunk and hands back the block states it produces.
 *
 * <p>This is the reference M3 is checked against. Everything before it compared numbers — noise
 * values, density values, corner tables — but the thing that actually has to come out right is the
 * block in each cell, and that is decided by the material rules (the aquifer and the ore veinifier)
 * sitting on top of the density. Reproducing those without a way to see what vanilla produced would be
 * guesswork, so this drives the real thing: the same cell iteration order as
 * {@code NoiseBasedChunkGenerator.doFill}, with {@code getInterpolatedState()} reached by reflection
 * because it is protected.
 *
 * <p>No world, no chunk object, no server: {@link NoiseChunk} has a public constructor, so a unit test
 * can run the NOISE stage on its own.
 */
public final class VanillaChunkReference {

    private static final Method GET_INTERPOLATED_STATE = interpolatedStateAccessor();

    /**
     * The block state ids of one chunk, indexed {@code (x * 16 + z) * height + (y - minY)}.
     *
     * @param stateIds {@link Block#getId(BlockState)} per block
     * @param minY     the chunk's lowest y
     * @param height   the chunk's height
     */
    public record ChunkBlocks(int[] stateIds, int minY, int height) {

        public int at(int x, int y, int z) {
            return this.stateIds[(x * 16 + z) * this.height + (y - this.minY)];
        }

        public int size() {
            return this.stateIds.length;
        }
    }

    private VanillaChunkReference() {
    }

    /** Generates one chunk with the overworld's own settings and rules. */
    public static ChunkBlocks generate(RandomState randomState, int chunkX, int chunkZ) {
        return generate(OverworldFixture.generatorSettings(), randomState, chunkX, chunkZ);
    }

    /**
     * Generates one chunk with the given settings.
     *
     * <p>The global fluid picker mirrors {@code NoiseBasedChunkGenerator}: lava below
     * {@code min(-54, seaLevel)}, water above it.
     */
    public static ChunkBlocks generate(NoiseGeneratorSettings settings, RandomState randomState,
            int chunkX, int chunkZ) {
        NoiseSettings noiseSettings = settings.noiseSettings();
        int cellWidth = noiseSettings.getCellWidth();
        int cellHeight = noiseSettings.getCellHeight();
        int cellCountXZ = 16 / cellWidth;
        int cellCountY = noiseSettings.height() / cellHeight;
        int cellMinY = Math.floorDiv(noiseSettings.minY(), cellHeight);

        int seaLevel = settings.seaLevel();
        Aquifer.FluidPicker fluidPicker = (x, y, z) -> y < Math.min(-54, seaLevel)
                ? new Aquifer.FluidStatus(-54, Blocks.LAVA.defaultBlockState())
                : new Aquifer.FluidStatus(seaLevel, Blocks.WATER.defaultBlockState());

        NoiseChunk chunk = new NoiseChunk(cellCountXZ, randomState, chunkX * 16, chunkZ * 16,
                noiseSettings, Beardifier.EMPTY, settings, fluidPicker, Blender.empty());

        int[] stateIds = new int[16 * 16 * noiseSettings.height()];
        int minY = noiseSettings.minY();
        int height = noiseSettings.height();
        int chunkMinBlockX = chunkX * 16;
        int chunkMinBlockZ = chunkZ * 16;

        chunk.initializeForFirstCellX();
        for (int cellX = 0; cellX < cellCountXZ; cellX++) {
            chunk.advanceCellX(cellX);
            for (int cellZ = 0; cellZ < cellCountXZ; cellZ++) {
                for (int cellY = cellCountY - 1; cellY >= 0; cellY--) {
                    chunk.selectCellYZ(cellY, cellZ);
                    for (int yInCell = cellHeight - 1; yInCell >= 0; yInCell--) {
                        int posY = (cellMinY + cellY) * cellHeight + yInCell;
                        chunk.updateForY(posY, (double) yInCell / cellHeight);
                        for (int xInCell = 0; xInCell < cellWidth; xInCell++) {
                            int x = cellX * cellWidth + xInCell;
                            // The interpolator is told the absolute block position; the block index
                            // into the chunk stays local.
                            chunk.updateForX(chunkMinBlockX + x, (double) xInCell / cellWidth);
                            for (int zInCell = 0; zInCell < cellWidth; zInCell++) {
                                int z = cellZ * cellWidth + zInCell;
                                chunk.updateForZ(chunkMinBlockZ + z, (double) zInCell / cellWidth);

                                BlockState state = interpolatedState(chunk);
                                if (state == null) {
                                    state = settings.defaultBlock();
                                }
                                stateIds[(x * 16 + z) * height + (posY - minY)] = Block.getId(state);
                            }
                        }
                    }
                }
            }
            chunk.swapSlices();
        }
        chunk.stopInterpolation();

        return new ChunkBlocks(stateIds, minY, height);
    }

    private static BlockState interpolatedState(NoiseChunk chunk) {
        try {
            return (BlockState) GET_INTERPOLATED_STATE.invoke(chunk);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw new IllegalStateException("NoiseChunk.getInterpolatedState threw", e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("vanilla layout changed: NoiseChunk.getInterpolatedState", e);
        }
    }

    private static Method interpolatedStateAccessor() {
        try {
            Method method = NoiseChunk.class.getDeclaredMethod("getInterpolatedState");
            method.setAccessible(true);
            return method;
        } catch (NoSuchMethodException e) {
            throw new IllegalStateException("vanilla layout changed: NoiseChunk.getInterpolatedState", e);
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot access NoiseChunk.getInterpolatedState", e);
        }
    }
}
