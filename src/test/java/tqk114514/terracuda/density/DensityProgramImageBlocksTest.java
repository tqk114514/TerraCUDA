package tqk114514.terracuda.density;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * The reduction behind {@link DensityProgramImage#ofBlocks}: what it keeps, and what it cuts.
 *
 * <p>The per-block pass runs on the device, and it is only worth running there because the image it
 * reads has the marker subgraphs removed — {@code final_density} is 450 instructions, of which 15 sit
 * above the interpolated markers. If that ratio ever collapses, the kernel's per-thread scratch goes
 * from tens of megabytes to hundreds and the reason for the whole split evaporates. Nothing else in
 * the suite would notice, because the values would still be right.
 *
 * <p>Opcodes are read out of the uploaded blob rather than through a new accessor: that is the exact
 * byte layout the kernel parses, so this checks the artifact that ships rather than a parallel view
 * of it.
 */
class DensityProgramImageBlocksTest {

    @Test
    void theBlockImageKeepsOnlyTheDagAboveTheMarkers() {
        RandomState randomState = OverworldFixture.randomState();
        DensityFunction finalDensity = randomState.router().finalDensity();

        DensityProgram program = DensityCompiler.lower(finalDensity);
        DensityProgramImage full = DensityProgramImage.of(program);
        DensityProgramImage blocks = DensityProgramImage.ofBlocks(program);

        assertTrue(blocks.instructionCount() * 10 < full.instructionCount(),
                "the block image should be a small fraction of the full one, but was "
                        + blocks.instructionCount() + " of " + full.instructionCount());

        // Every interpolated marker survives, cut down to an OVERRIDE leaf, and nothing else does:
        // the overworld's flat and 2D caches all live inside an interpolated marker.
        int interpolated = 0;
        for (int kind : full.markerKinds()) {
            if (kind == DensityProgram.MARKER_INTERPOLATED) {
                interpolated++;
            }
        }
        assertEquals(interpolated, blocks.markerCount(),
                "the block image should carry exactly the interpolated markers");

        int[] kinds = blocks.markerKinds();
        int[] roots = blocks.markerRoots();
        for (int i = 0; i < kinds.length; i++) {
            assertEquals(DensityProgram.MARKER_INTERPOLATED, kinds[i]);
            assertEquals(DensityProgram.OVERRIDE, opcode(blocks, roots[i]),
                    "marker " + i + " should be an OVERRIDE leaf in the block image");
        }
    }

    @Test
    void aProgramWithNoMarkersIsUnchanged() {
        // Nothing to cut, so the two images must agree instruction for instruction. This is the case
        // a data pack that removed the markers would hit.
        DensityProgram program = DensityCompiler.lower(
                OverworldFixture.randomState().router().barrierNoise());
        DensityProgramImage full = DensityProgramImage.of(program);
        DensityProgramImage blocks = DensityProgramImage.ofBlocks(program);

        assertEquals(full.instructionCount(), blocks.instructionCount());
        assertEquals(full.root(), blocks.root());
        assertEquals(0, blocks.markerCount());
        for (int pc = 0; pc < full.instructionCount(); pc++) {
            assertEquals(opcode(full, pc), opcode(blocks, pc), "opcode at " + pc);
        }
    }

    /** The opcode at {@code pc}, read the way the kernel reads it. */
    private static int opcode(DensityProgramImage image, int pc) {
        byte[] blob = image.blob();
        long at = image.offsets()[DensityProgramImage.OFF_OPS] + 4L * pc;
        return (blob[(int) at] & 0xFF)
                | (blob[(int) at + 1] & 0xFF) << 8
                | (blob[(int) at + 2] & 0xFF) << 16
                | (blob[(int) at + 3] & 0xFF) << 24;
    }
}
