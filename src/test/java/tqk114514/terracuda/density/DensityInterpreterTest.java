package tqk114514.terracuda.density;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.RandomState;
import org.junit.jupiter.api.Test;
import tqk114514.terracuda.TestParity;
import tqk114514.terracuda.worldgen.OverworldFixture;

/**
 * M0: the lowered program, evaluated by {@link DensityInterpreter}, must reproduce vanilla's own
 * {@code DensityFunction.compute} bit for bit on the <em>real</em> overworld density function.
 *
 * <p>This is the acceptance criterion for the milestone, and the thing that makes the GPU kernel
 * verifiable: whatever the CUDA interpreter produces is compared against this, not against a
 * hand-written expectation.
 */
class DensityInterpreterTest {

    @Test
    void theOverworldFinalDensityLowersWithoutUnsupportedNodes() {
        RandomState randomState = OverworldFixture.randomState();
        DensityProgram program = DensityCompiler.lower(randomState.router().finalDensity());

        assertTrue(program.size() > 100, "expected a large program, got " + program.size());
        assertTrue(program.noiseCount() > 10, "expected many distinct noises, got " + program.noiseCount());
        assertTrue(program.splineCount() > 0, "the overworld density function uses cubic splines");
        assertTrue(program.blendedCount() > 0, "base_3d_noise is a BlendedNoise");
        assertTrue(program.markerCount() > 0, "the overworld density function is marked up for caching");
    }

    @Test
    void theInterpreterMatchesVanillaBitForBit() {
        RandomState randomState = OverworldFixture.randomState();
        DensityFunction vanilla = randomState.router().finalDensity();
        DensityInterpreter interpreter = new DensityInterpreter(DensityCompiler.lower(vanilla));

        int checked = 0;
        for (int x = -64; x <= 64; x += 16) {
            for (int z = -64; z <= 64; z += 16) {
                for (int y = -64; y <= 320; y += 16) {
                    double expected = vanilla.compute(new DensityFunction.SinglePointContext(x, y, z));
                    double actual = interpreter.evaluate(x, y, z);
                    TestParity.assertDoubleIdentical(expected, actual, "(" + x + ", " + y + ", " + z + ")");
                    checked++;
                }
            }
        }
        assertTrue(checked >= 9 * 9 * 25, "expected a dense sample grid, checked " + checked);
    }

    @Test
    void theInterpreterMatchesVanillaAtIrregularPositions() {
        RandomState randomState = OverworldFixture.randomState();
        DensityFunction vanilla = randomState.router().finalDensity();
        DensityInterpreter interpreter = new DensityInterpreter(DensityCompiler.lower(vanilla));

        // Odd coordinates stress the floor / wrap boundaries that a regular grid can step over.
        for (int i = 0; i < 2000; i++) {
            int x = (int) Math.round(OverworldFixture.coordinate(i, 0) * 64.0);
            int y = (int) Math.round(OverworldFixture.coordinate(i, 1) * 40.0);
            int z = (int) Math.round(OverworldFixture.coordinate(i, 2) * 64.0);
            double expected = vanilla.compute(new DensityFunction.SinglePointContext(x, y, z));
            double actual = interpreter.evaluate(x, y, z);
            TestParity.assertDoubleIdentical(expected, actual, "(" + x + ", " + y + ", " + z + ")");
        }
    }

    @Test
    void repeatedEvaluationsAreStable() {
        RandomState randomState = OverworldFixture.randomState();
        DensityInterpreter interpreter =
                new DensityInterpreter(DensityCompiler.lower(randomState.router().finalDensity()));

        double first = interpreter.evaluate(12, 71, -33);
        // Interleave a different position: the per-point memo must not leak across contexts.
        interpreter.evaluate(-500, -12, 900);
        double again = interpreter.evaluate(12, 71, -33);

        TestParity.assertDoubleIdentical(first, again, "re-evaluating the same position");
    }

    @Test
    void anUnknownNodeTypeIsRejectedRatherThanGuessedAt() {
        // The design doc requires an unrecognised node (a data pack, another mod) to send the world
        // back to vanilla rather than be approximated. A node type the compiler has never seen is the
        // clearest way to exercise that.
        DensityFunction unknown = new UnsupportedNode();

        DensityCompiler.UnsupportedDensityFunctionException failure =
                assertThrows(DensityCompiler.UnsupportedDensityFunctionException.class,
                        () -> DensityCompiler.lower(unknown));
        assertTrue(failure.getMessage().contains("UnsupportedNode"), failure.getMessage());
    }

    /** A density function this compiler deliberately does not know how to lower. */
    private static final class UnsupportedNode implements DensityFunction {

        @Override
        public double compute(FunctionContext context) {
            return 0.0;
        }

        @Override
        public void fillArray(double[] output, ContextProvider contextProvider) {
            java.util.Arrays.fill(output, 0.0);
        }

        @Override
        public DensityFunction mapAll(Visitor visitor) {
            return visitor.apply(this);
        }

        @Override
        public double minValue() {
            return 0.0;
        }

        @Override
        public double maxValue() {
            return 0.0;
        }

        @Override
        public net.minecraft.util.KeyDispatchDataCodec<? extends DensityFunction> codec() {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    void theProgramReportsItsShape() {
        RandomState randomState = OverworldFixture.randomState();
        DensityProgram program = DensityCompiler.lower(randomState.router().finalDensity());

        assertTrue(program.summary().contains("instructions"));
        assertEquals(program.markerCount(), program.markerKinds().length);
        assertEquals(program.markerCount(), program.markerRoots().length);

        // Markers are kernel boundaries, not instructions, so there are far fewer of them.
        assertTrue(program.markerCount() > 0);
        assertTrue(program.markerCount() < program.size());

        for (int root : program.markerRoots()) {
            assertTrue(root >= 0 && root < program.size(), "marker root " + root + " is out of range");
        }
        assertTrue(program.root() >= 0 && program.root() < program.size());
    }
}
