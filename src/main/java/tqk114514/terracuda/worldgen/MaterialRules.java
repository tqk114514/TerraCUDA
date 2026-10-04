package tqk114514.terracuda.worldgen;

import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.NoiseSettings;
import tqk114514.terracuda.density.DensityInterpreter;
import tqk114514.terracuda.math.VanillaMath;
import tqk114514.terracuda.random.PositionalRandomFactory;
import tqk114514.terracuda.random.XoroshiroRandom;

/**
 * The material rules: what block a cell holds, given its density.
 *
 * <p>This is the layer above the density. Vanilla hands the interpolated {@code final_density} to an
 * {@code Aquifer} and then to an {@code OreVeinifier}, and only if both decline does the cell get the
 * default block. Getting the density right is not enough on its own — the aquifer decides where the
 * water is, and that is most of what a player sees.
 *
 * <p>Ported from {@code Aquifer.NoiseBasedAquifer} and {@code OreVeinifier}, which are package-private
 * and take a live {@code NoiseChunk}. Everything they read has been replaced with something this
 * project computes: the router's nine noise functions come from lowered programs, the preliminary
 * surface levels come from K0, and the positional randomness comes from the exported seed pairs
 * through this project's own XORoshiro.
 *
 * <p>One instance is one chunk's aquifer state — the grid bounds and the two caches are sized from the
 * chunk position, exactly as vanilla does. Not thread-safe.
 */
public final class MaterialRules {

    /** Where the aquifer reads preliminary surface levels from; K0 in production. */
    public interface SurfaceLevels {
        int preliminarySurfaceLevel(int blockX, int blockZ);

        int maxPreliminarySurfaceLevel(int minBlockX, int minBlockZ, int maxBlockX, int maxBlockZ);
    }

    /**
     * A value that is already known per block.
     *
     * <p>Needed because two of the router's entries — {@code veinToggle} and {@code veinRidged} — are
     * {@code interpolated} markers, so the value at a block is a blend of coarse samples rather than
     * the sub-graph evaluated there. Evaluating them directly would give sharper vein boundaries than
     * vanilla, which is exactly the mistake the design doc warns about.
     */
    public interface PerBlock {
        double at(int blockX, int blockY, int blockZ);
    }

    /** The router's noise functions. The interpolated ones arrive as per-block values. */
    public record RouterNoises(
            DensityInterpreter barrierNoise,
            DensityInterpreter fluidLevelFloodednessNoise,
            DensityInterpreter fluidLevelSpreadNoise,
            DensityInterpreter lavaNoise,
            DensityInterpreter erosion,
            DensityInterpreter depth,
            DensityInterpreter veinGap,
            PerBlock veinToggle,
            PerBlock veinRidged) {
    }

    private static final int Y_SPACING = 12;
    private static final int X_SPACING_SHIFT = 4;
    private static final int Z_SPACING_SHIFT = 4;
    private static final double FLOWING_UPDATE_SIMILARITY = similarity(10 * 10, 12 * 12);
    private static final int SAMPLE_OFFSET_X = -5;
    private static final int SAMPLE_OFFSET_Z = -5;
    private static final int[][] SURFACE_SAMPLING_OFFSETS_IN_CHUNKS = {
            {0, 0}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {-3, 0}, {-2, 0}, {-1, 0}, {1, 0},
            {-2, 1}, {-1, 1}, {0, 1}, {1, 1}
    };
    /**
     * {@code DimensionType.WAY_BELOW_MIN_Y}, the sentinel the aquifer uses for "no ceiling above".
     *
     * <p>Read from the constant rather than written out: it is {@code MIN_Y << 4} with
     * {@code MIN_Y = -2032}, so the value is -32512, and hard-coding the overworld's -64 here would
     * silently skip the lava check on any dimension deeper than -768.
     */
    private static final int WAY_BELOW_MIN_Y = net.minecraft.world.level.dimension.DimensionType.WAY_BELOW_MIN_Y;

    private static final BlockState AIR = Blocks.AIR.defaultBlockState();
    private static final BlockState WATER = Blocks.WATER.defaultBlockState();
    private static final BlockState LAVA = Blocks.LAVA.defaultBlockState();

    /** The two ore veins, with their y ranges. */
    private enum VeinType {
        COPPER(Blocks.COPPER_ORE.defaultBlockState(), Blocks.RAW_COPPER_BLOCK.defaultBlockState(),
                Blocks.GRANITE.defaultBlockState(), 0, 50),
        IRON(Blocks.DEEPSLATE_IRON_ORE.defaultBlockState(), Blocks.RAW_IRON_BLOCK.defaultBlockState(),
                Blocks.TUFF.defaultBlockState(), -60, -8);

        private final BlockState ore;
        private final BlockState rawOreBlock;
        private final BlockState filler;
        private final int minY;
        private final int maxY;

        VeinType(BlockState ore, BlockState rawOreBlock, BlockState filler, int minY, int maxY) {
            this.ore = ore;
            this.rawOreBlock = rawOreBlock;
            this.filler = filler;
            this.minY = minY;
            this.maxY = maxY;
        }
    }

    private final RouterNoises noises;
    private final SurfaceLevels surfaces;
    private final Aquifer.FluidPicker globalFluidPicker;
    private final PositionalRandomFactory aquiferRandom;
    private final PositionalRandomFactory oreRandom;

    private final Aquifer.FluidStatus[] aquiferCache;
    private final long[] aquiferLocationCache;
    private final int minGridX;
    private final int minGridY;
    private final int minGridZ;
    private final int gridSizeX;
    private final int gridSizeY;
    private final int gridSizeZ;
    private final int skipSamplingAboveY;

    private boolean shouldScheduleFluidUpdate;
    private final boolean aquifersEnabled;
    private final boolean oreVeinsEnabled;
    private final double[] barrierNoiseScratch = {Double.NaN};

    private MaterialRules(RouterNoises noises, SurfaceLevels surfaces,
            Aquifer.FluidPicker globalFluidPicker, PositionalRandomFactory aquiferRandom,
            PositionalRandomFactory oreRandom, int chunkMinBlockX, int chunkMinBlockZ,
            NoiseSettings settings, boolean aquifersEnabled, boolean oreVeinsEnabled) {
        this.aquifersEnabled = aquifersEnabled;
        this.oreVeinsEnabled = oreVeinsEnabled;
        this.noises = noises;
        this.surfaces = surfaces;
        this.globalFluidPicker = globalFluidPicker;
        this.aquiferRandom = aquiferRandom;
        this.oreRandom = oreRandom;

        int chunkMaxBlockX = chunkMinBlockX + 15;
        int chunkMaxBlockZ = chunkMinBlockZ + 15;
        this.minGridX = gridX(chunkMinBlockX + SAMPLE_OFFSET_X);
        int maxGridX = gridX(chunkMaxBlockX + SAMPLE_OFFSET_X) + 1;
        this.gridSizeX = maxGridX - this.minGridX + 1;
        this.minGridY = gridY(settings.minY() + 1) - 1;
        int maxGridY = gridY(settings.minY() + settings.height() + 1) + 1;
        this.gridSizeY = maxGridY - this.minGridY + 1;
        this.minGridZ = gridZ(chunkMinBlockZ + SAMPLE_OFFSET_Z);
        int maxGridZ = gridZ(chunkMaxBlockZ + SAMPLE_OFFSET_Z) + 1;
        this.gridSizeZ = maxGridZ - this.minGridZ + 1;

        int total = this.gridSizeX * this.gridSizeY * this.gridSizeZ;
        this.aquiferCache = new Aquifer.FluidStatus[total];
        this.aquiferLocationCache = new long[total];
        java.util.Arrays.fill(this.aquiferLocationCache, Long.MAX_VALUE);

        int maxAdjustedSurfaceLevel = this.surfaces.maxPreliminarySurfaceLevel(
                fromGridX(this.minGridX, 0), fromGridZ(this.minGridZ, 0),
                fromGridX(maxGridX, 9), fromGridZ(maxGridZ, 9)) + 8;
        int skipSamplingAboveGridY = gridY(maxAdjustedSurfaceLevel + 12) + 1;
        this.skipSamplingAboveY = fromGridY(skipSamplingAboveGridY, 11) - 1;
    }

    /** Builds the aquifer state for one chunk. */
    public static MaterialRules forChunk(RouterNoises noises, SurfaceLevels surfaces,
            Aquifer.FluidPicker globalFluidPicker, PositionalRandomFactory aquiferRandom,
            PositionalRandomFactory oreRandom, int chunkX, int chunkZ, NoiseSettings settings,
            boolean aquifersEnabled, boolean oreVeinsEnabled) {
        return new MaterialRules(noises, surfaces, globalFluidPicker, aquiferRandom, oreRandom,
                chunkX * 16, chunkZ * 16, settings, aquifersEnabled, oreVeinsEnabled);
    }

    /**
     * The block for one cell, or {@code null} when the rules decline and the caller should use the
     * default block.
     *
     * @param density the interpolated {@code final_density} plus the beardifier
     */
    public BlockState compute(int posX, int posY, int posZ, double density) {
        BlockState aquifer = this.aquifersEnabled
                ? this.aquiferSubstance(posX, posY, posZ, density)
                : disabledAquifer(posX, posY, posZ, density);
        if (aquifer != null) {
            return aquifer;
        }
        return this.oreVeinsEnabled ? this.veinifier(posX, posY, posZ) : null;
    }

    /**
     * What {@code Aquifer.createDisabled} does: the global fluid picker, and nothing else.
     *
     * <p>It still consults the density — {@code density > 0.0} means sky — but it never picks a
     * neighbouring cell's status, and it never asks for a fluid tick. A dimension with aquifers off
     * is therefore a flat fluid level rather than a network of pockets, and without this branch the
     * nether would come out with overworld-style aquifers in it.
     */
    private BlockState disabledAquifer(int posX, int posY, int posZ, double density) {
        this.shouldScheduleFluidUpdate = false;
        return density > 0.0 ? null
                : this.globalFluidPicker.computeFluid(posX, posY, posZ).at(posY);
    }

    /** Whether the aquifer wants the position queued for a fluid tick. */
    public boolean shouldScheduleFluidUpdate() {
        return this.shouldScheduleFluidUpdate;
    }

    // ------------------------------------------------------------------------------------------
    // Aquifer
    // ------------------------------------------------------------------------------------------

    private BlockState aquiferSubstance(int posX, int posY, int posZ, double density) {
        if (density > 0.0) {
            this.shouldScheduleFluidUpdate = false;
            return null;
        }

        Aquifer.FluidStatus globalFluid = this.globalFluidPicker.computeFluid(posX, posY, posZ);
        if (posY > this.skipSamplingAboveY) {
            this.shouldScheduleFluidUpdate = false;
            return globalFluid.at(posY);
        }
        if (globalFluid.at(posY).is(Blocks.LAVA)) {
            this.shouldScheduleFluidUpdate = false;
            return LAVA;
        }

        int xAnchor = gridX(posX + SAMPLE_OFFSET_X);
        int yAnchor = gridY(posY + 1);
        int zAnchor = gridZ(posZ + SAMPLE_OFFSET_Z);
        int distanceSqr1 = Integer.MAX_VALUE;
        int distanceSqr2 = Integer.MAX_VALUE;
        int distanceSqr3 = Integer.MAX_VALUE;
        int distanceSqr4 = Integer.MAX_VALUE;
        int closestIndex1 = 0;
        int closestIndex2 = 0;
        int closestIndex3 = 0;
        int closestIndex4 = 0;

        for (int x1 = 0; x1 <= 1; x1++) {
            for (int y1 = -1; y1 <= 1; y1++) {
                for (int z1 = 0; z1 <= 1; z1++) {
                    int spacedGridX = xAnchor + x1;
                    int spacedGridY = yAnchor + y1;
                    int spacedGridZ = zAnchor + z1;
                    int index = getIndex(spacedGridX, spacedGridY, spacedGridZ);

                    long location = this.aquiferLocationCache[index];
                    if (location == Long.MAX_VALUE) {
                        XoroshiroRandom random = this.aquiferRandom.at(spacedGridX, spacedGridY, spacedGridZ);
                        location = pack(
                                fromGridX(spacedGridX, random.nextInt(10)),
                                fromGridY(spacedGridY, random.nextInt(9)),
                                fromGridZ(spacedGridZ, random.nextInt(10)));
                        this.aquiferLocationCache[index] = location;
                    }

                    int dx = unpackX(location) - posX;
                    int dy = unpackY(location) - posY;
                    int dz = unpackZ(location) - posZ;
                    int newDistance = dx * dx + dy * dy + dz * dz;
                    if (distanceSqr1 >= newDistance) {
                        closestIndex4 = closestIndex3;
                        closestIndex3 = closestIndex2;
                        closestIndex2 = closestIndex1;
                        closestIndex1 = index;
                        distanceSqr4 = distanceSqr3;
                        distanceSqr3 = distanceSqr2;
                        distanceSqr2 = distanceSqr1;
                        distanceSqr1 = newDistance;
                    } else if (distanceSqr2 >= newDistance) {
                        closestIndex4 = closestIndex3;
                        closestIndex3 = closestIndex2;
                        closestIndex2 = index;
                        distanceSqr4 = distanceSqr3;
                        distanceSqr3 = distanceSqr2;
                        distanceSqr2 = newDistance;
                    } else if (distanceSqr3 >= newDistance) {
                        closestIndex4 = closestIndex3;
                        closestIndex3 = index;
                        distanceSqr4 = distanceSqr3;
                        distanceSqr3 = newDistance;
                    } else if (distanceSqr4 >= newDistance) {
                        closestIndex4 = index;
                        distanceSqr4 = newDistance;
                    }
                }
            }
        }

        Aquifer.FluidStatus closest1 = getAquiferStatus(closestIndex1);
        double similarity12 = similarity(distanceSqr1, distanceSqr2);
        BlockState fluidState = closest1.at(posY);
        if (similarity12 <= 0.0) {
            if (similarity12 >= FLOWING_UPDATE_SIMILARITY) {
                this.shouldScheduleFluidUpdate = !closest1.equals(getAquiferStatus(closestIndex2));
            } else {
                this.shouldScheduleFluidUpdate = false;
            }
            return fluidState;
        }
        if (fluidState.is(Blocks.WATER)
                && this.globalFluidPicker.computeFluid(posX, posY - 1, posZ).at(posY - 1).is(Blocks.LAVA)) {
            this.shouldScheduleFluidUpdate = true;
            return fluidState;
        }

        this.barrierNoiseScratch[0] = Double.NaN;
        Aquifer.FluidStatus closest2 = getAquiferStatus(closestIndex2);
        double barrier12 = similarity12 * calculatePressure(posX, posY, posZ, closest1, closest2);
        if (density + barrier12 > 0.0) {
            this.shouldScheduleFluidUpdate = false;
            return null;
        }

        Aquifer.FluidStatus closest3 = getAquiferStatus(closestIndex3);
        double similarity13 = similarity(distanceSqr1, distanceSqr3);
        if (similarity13 > 0.0) {
            double barrier13 = similarity12 * similarity13
                    * calculatePressure(posX, posY, posZ, closest1, closest3);
            if (density + barrier13 > 0.0) {
                this.shouldScheduleFluidUpdate = false;
                return null;
            }
        }

        double similarity23 = similarity(distanceSqr2, distanceSqr3);
        if (similarity23 > 0.0) {
            double barrier23 = similarity12 * similarity23
                    * calculatePressure(posX, posY, posZ, closest2, closest3);
            if (density + barrier23 > 0.0) {
                this.shouldScheduleFluidUpdate = false;
                return null;
            }
        }

        boolean mayFlow12 = !closest1.equals(closest2);
        boolean mayFlow23 = similarity23 >= FLOWING_UPDATE_SIMILARITY && !closest2.equals(closest3);
        boolean mayFlow13 = similarity13 >= FLOWING_UPDATE_SIMILARITY && !closest1.equals(closest3);
        if (!mayFlow12 && !mayFlow23 && !mayFlow13) {
            this.shouldScheduleFluidUpdate = similarity13 >= FLOWING_UPDATE_SIMILARITY
                    && similarity(distanceSqr1, distanceSqr4) >= FLOWING_UPDATE_SIMILARITY
                    && !closest1.equals(getAquiferStatus(closestIndex4));
        } else {
            this.shouldScheduleFluidUpdate = true;
        }
        return fluidState;
    }

    private double calculatePressure(int posX, int posY, int posZ,
            Aquifer.FluidStatus status1, Aquifer.FluidStatus status2) {
        BlockState type1 = status1.at(posY);
        BlockState type2 = status2.at(posY);
        boolean oppositeFluids = (type1.is(Blocks.LAVA) && type2.is(Blocks.WATER))
                || (type1.is(Blocks.WATER) && type2.is(Blocks.LAVA));
        if (oppositeFluids) {
            return 2.0;
        }

        int fluidYDiff = Math.abs(status1.fluidLevel() - status2.fluidLevel());
        if (fluidYDiff == 0) {
            return 0.0;
        }

        double averageFluidY = 0.5 * (status1.fluidLevel() + status2.fluidLevel());
        double howFarAboveAverage = posY + 0.5 - averageFluidY;
        double baseValue = fluidYDiff / 2.0;
        double distanceFromBarrierEdge = baseValue - Math.abs(howFarAboveAverage);
        double gradient;
        if (howFarAboveAverage > 0.0) {
            double centerPoint = distanceFromBarrierEdge;
            gradient = centerPoint > 0.0 ? centerPoint / 1.5 : centerPoint / 2.5;
        } else {
            double centerPoint = 3.0 + distanceFromBarrierEdge;
            gradient = centerPoint > 0.0 ? centerPoint / 3.0 : centerPoint / 10.0;
        }

        double noiseValue;
        if (gradient >= -2.0 && gradient <= 2.0) {
            if (Double.isNaN(this.barrierNoiseScratch[0])) {
                this.barrierNoiseScratch[0] = this.noises.barrierNoise().evaluate(posX, posY, posZ);
            }
            noiseValue = this.barrierNoiseScratch[0];
        } else {
            noiseValue = 0.0;
        }
        return 2.0 * (noiseValue + gradient);
    }

    private Aquifer.FluidStatus getAquiferStatus(int index) {
        Aquifer.FluidStatus cached = this.aquiferCache[index];
        if (cached != null) {
            return cached;
        }
        long location = this.aquiferLocationCache[index];
        Aquifer.FluidStatus status = computeFluid(unpackX(location), unpackY(location), unpackZ(location));
        this.aquiferCache[index] = status;
        return status;
    }

    private Aquifer.FluidStatus computeFluid(int x, int y, int z) {
        Aquifer.FluidStatus globalFluid = this.globalFluidPicker.computeFluid(x, y, z);
        int lowestPreliminarySurface = Integer.MAX_VALUE;
        int topOfAquiferCell = y + 12;
        int bottomOfAquiferCell = y - 12;
        boolean surfaceAtCenterIsUnderGlobalFluidLevel = false;

        for (int[] offset : SURFACE_SAMPLING_OFFSETS_IN_CHUNKS) {
            int sampleX = x + (offset[0] << 4);
            int sampleZ = z + (offset[1] << 4);
            int preliminarySurfaceLevel = this.surfaces.preliminarySurfaceLevel(sampleX, sampleZ);
            int adjustedSurfaceLevel = preliminarySurfaceLevel + 8;
            boolean start = offset[0] == 0 && offset[1] == 0;
            if (start && bottomOfAquiferCell > adjustedSurfaceLevel) {
                return globalFluid;
            }

            boolean topPokesAboveSurface = topOfAquiferCell > adjustedSurfaceLevel;
            if (topPokesAboveSurface || start) {
                Aquifer.FluidStatus globalFluidAtSurface =
                        this.globalFluidPicker.computeFluid(sampleX, adjustedSurfaceLevel, sampleZ);
                if (!globalFluidAtSurface.at(adjustedSurfaceLevel).isAir()) {
                    if (start) {
                        surfaceAtCenterIsUnderGlobalFluidLevel = true;
                    }
                    if (topPokesAboveSurface) {
                        return globalFluidAtSurface;
                    }
                }
            }

            lowestPreliminarySurface = Math.min(lowestPreliminarySurface, preliminarySurfaceLevel);
        }

        int fluidSurfaceLevel = computeSurfaceLevel(x, y, z, globalFluid, lowestPreliminarySurface,
                surfaceAtCenterIsUnderGlobalFluidLevel);
        return new Aquifer.FluidStatus(fluidSurfaceLevel, computeFluidType(x, y, z, globalFluid, fluidSurfaceLevel));
    }

    private int computeSurfaceLevel(int x, int y, int z, Aquifer.FluidStatus globalFluid,
            int lowestPreliminarySurface, boolean surfaceAtCenterIsUnderGlobalFluidLevel) {
        double partiallyFloodedness;
        double fullyFloodedness;
        if (isDeepDarkRegion(x, y, z)) {
            partiallyFloodedness = -1.0;
            fullyFloodedness = -1.0;
        } else {
            int distanceBelowSurface = lowestPreliminarySurface + 8 - y;
            double floodednessFactor = surfaceAtCenterIsUnderGlobalFluidLevel
                    ? VanillaMath.clampedMap(distanceBelowSurface, 0.0, 64.0, 1.0, 0.0)
                    : 0.0;
            double floodednessNoise = VanillaMath.clamp(
                    this.noises.fluidLevelFloodednessNoise().evaluate(x, y, z), -1.0, 1.0);
            double fullyFloodedThreshold = VanillaMath.lerp(
                    VanillaMath.inverseLerp(floodednessFactor, 1.0, 0.0), -0.3, 0.8);
            double partiallyFloodedThreshold = VanillaMath.lerp(
                    VanillaMath.inverseLerp(floodednessFactor, 1.0, 0.0), -0.8, 0.4);
            partiallyFloodedness = floodednessNoise - partiallyFloodedThreshold;
            fullyFloodedness = floodednessNoise - fullyFloodedThreshold;
        }

        if (fullyFloodedness > 0.0) {
            return globalFluid.fluidLevel();
        }
        if (partiallyFloodedness > 0.0) {
            return computeRandomizedFluidSurfaceLevel(x, y, z, lowestPreliminarySurface);
        }
        return WAY_BELOW_MIN_Y;
    }

    private int computeRandomizedFluidSurfaceLevel(int x, int y, int z, int lowestPreliminarySurface) {
        int fluidLevelCellX = Math.floorDiv(x, 16);
        int fluidLevelCellY = Math.floorDiv(y, 40);
        int fluidLevelCellZ = Math.floorDiv(z, 16);
        int fluidCellMiddleY = fluidLevelCellY * 40 + 20;
        double spread = this.noises.fluidLevelSpreadNoise()
                .evaluate(fluidLevelCellX, fluidLevelCellY, fluidLevelCellZ) * 10.0;
        int quantized = VanillaMath.floor(spread / 3) * 3;
        int targetFluidSurfaceLevel = fluidCellMiddleY + quantized;
        return Math.min(lowestPreliminarySurface, targetFluidSurfaceLevel);
    }

    private BlockState computeFluidType(int x, int y, int z, Aquifer.FluidStatus globalFluid,
            int fluidSurfaceLevel) {
        BlockState fluidType = globalFluid.fluidType();
        if (fluidSurfaceLevel <= -10 && fluidSurfaceLevel != WAY_BELOW_MIN_Y && fluidType != LAVA) {
            int fluidTypeCellX = Math.floorDiv(x, 64);
            int fluidTypeCellY = Math.floorDiv(y, 40);
            int fluidTypeCellZ = Math.floorDiv(z, 64);
            double lavaNoise = this.noises.lavaNoise()
                    .evaluate(fluidTypeCellX, fluidTypeCellY, fluidTypeCellZ);
            if (Math.abs(lavaNoise) > 0.3) {
                fluidType = LAVA;
            }
        }
        return fluidType;
    }

    private boolean isDeepDarkRegion(int x, int y, int z) {
        // The literals are floats in vanilla, and the comparison widens them: -0.225F is
        // -0.22499999403953552 and 0.9F is 0.8999999761581421. Writing them as doubles moves both
        // boundaries by a few times 1e-8, which is enough to flip the aquifer's decision on the
        // narrow band where the deep-dark test sits.
        return this.noises.erosion().evaluate(x, y, z) < -0.225F
                && this.noises.depth().evaluate(x, y, z) > 0.9F;
    }

    private int getIndex(int gridX, int gridY, int gridZ) {
        int x = gridX - this.minGridX;
        int y = gridY - this.minGridY;
        int z = gridZ - this.minGridZ;
        return (y * this.gridSizeZ + z) * this.gridSizeX + x;
    }

    private static double similarity(int distanceSqr1, int distanceSqr2) {
        return 1.0 - (distanceSqr2 - distanceSqr1) / 25.0;
    }

    private static int gridX(int blockCoord) {
        return blockCoord >> X_SPACING_SHIFT;
    }

    private static int gridY(int blockCoord) {
        return Math.floorDiv(blockCoord, Y_SPACING);
    }

    private static int gridZ(int blockCoord) {
        return blockCoord >> Z_SPACING_SHIFT;
    }

    private static int fromGridX(int gridCoord, int blockOffset) {
        return (gridCoord << X_SPACING_SHIFT) + blockOffset;
    }

    private static int fromGridY(int gridCoord, int blockOffset) {
        return gridCoord * Y_SPACING + blockOffset;
    }

    private static int fromGridZ(int gridCoord, int blockOffset) {
        return (gridCoord << Z_SPACING_SHIFT) + blockOffset;
    }

    // BlockPos packing, reproduced so the port does not depend on vanilla's block state registry.
    private static long pack(int x, int y, int z) {
        return ((long) x & 0x3FFFFFFL) << 38 | ((long) z & 0x3FFFFFFL) << 12 | ((long) y & 0xFFFL);
    }

    private static int unpackX(long packed) {
        return (int) (packed >> 38);
    }

    private static int unpackY(long packed) {
        return (int) (packed << 52 >> 52);
    }

    private static int unpackZ(long packed) {
        return (int) (packed << 26 >> 38);
    }

    // ------------------------------------------------------------------------------------------
    // Ore veins
    // ------------------------------------------------------------------------------------------

    private BlockState veinifier(int posX, int posY, int posZ) {
        double veininess = this.noises.veinToggle().at(posX, posY, posZ);
        VeinType type = veininess > 0.0 ? VeinType.COPPER : VeinType.IRON;
        double ridged = Math.abs(veininess);
        int distanceFromTop = type.maxY - posY;
        int distanceFromBottom = posY - type.minY;
        if (distanceFromBottom < 0 || distanceFromTop < 0) {
            return null;
        }

        int distanceFromEdge = Math.min(distanceFromTop, distanceFromBottom);
        double edgeRoundoff = VanillaMath.clampedMap(distanceFromEdge, 0.0, 20.0, -0.2, 0.0);
        if (ridged + edgeRoundoff < 0.4F) {
            return null;
        }

        XoroshiroRandom random = this.oreRandom.at(posX, posY, posZ);
        if (random.nextFloat() > 0.7F) {
            return null;
        }
        if (this.noises.veinRidged().at(posX, posY, posZ) >= 0.0) {
            return null;
        }

        double richness = VanillaMath.clampedMap(ridged, 0.4F, 0.6F, 0.1F, 0.3F);
        if (random.nextFloat() < richness && this.noises.veinGap().evaluate(posX, posY, posZ) > -0.3F) {
            return random.nextFloat() < 0.02F ? type.rawOreBlock : type.ore;
        }
        return type.filler;
    }
}
