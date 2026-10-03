package tqk114514.terracuda.mixin;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;
import tqk114514.terracuda.worldgen.GpuWorldgenService;

/**
 * Hooks {@code NoiseBasedChunkGenerator.fillFromNoise}, the entry point to the NOISE stage.
 *
 * <p>At this milestone the hook is deliberately <em>observational</em>: it runs the whole device path
 * for the chunk — lowering, upload, the density, the material rules, the block ids — and logs what came
 * out, but never cancels the vanilla call and never touches the chunk. Switching it to cancel is a
 * one-line change, and it is worth waiting for: the replay that writes those ids into the chunk is the
 * one piece with no unit test behind it, because building a chunk in a test needs registry plumbing
 * that has nothing to do with this code. Running it in a real world first is the cheaper way to find
 * out. What it does buy today is proof that the whole chain works against a live {@link RandomState}
 * inside a real game.
 *
 * <p>Failure containment is the point of the shape here: everything is inside a try/catch, the switch
 * is off by default, the injection is marked non-required, and the mixin config is marked non-required,
 * so a future Minecraft changing the signature degrades to a warning rather than a crash.
 */
@Mixin(NoiseBasedChunkGenerator.class)
public abstract class NoiseBasedChunkGeneratorMixin {

    private static final Map<RandomState, GpuWorldgenService> SERVICES = new ConcurrentHashMap<>();
    private static final AtomicLong CHUNKS = new AtomicLong();
    private static final AtomicBoolean REPORTED_FAILURE = new AtomicBoolean();

    @Inject(method = "fillFromNoise", at = @At("HEAD"), require = 0)
    private void terracuda$shadowFillFromNoise(Blender blender, RandomState randomState,
            StructureManager structureManager, ChunkAccess centerChunk,
            CallbackInfoReturnable<CompletableFuture<ChunkAccess>> cir) {
        if (!TerracudaConfig.gpuEnabled() || !TerracudaConfig.shadowMode()) {
            return;
        }
        try {
            NoiseBasedChunkGenerator self = (NoiseBasedChunkGenerator) (Object) this;
            NoiseGeneratorSettings settings = self.generatorSettings().value();

            GpuWorldgenService service = SERVICES.computeIfAbsent(randomState,
                    key -> GpuWorldgenService.of(key, settings));
            if (!service.isReady()) {
                if (REPORTED_FAILURE.compareAndSet(false, true)) {
                    TerraCUDA.LOGGER.info("TerraCUDA: shadow mode disabled, device unavailable: {}",
                            service.unavailableReason());
                }
                return;
            }

            ChunkPos pos = centerChunk.getPos();
            long start = System.nanoTime();
            Optional<int[]> ids = service.blockIds(pos.x(), pos.z());
            long micros = (System.nanoTime() - start) / 1000;

            long count = CHUNKS.incrementAndGet();
            if (TerracudaConfig.verbose() || count % 256 == 1) {
                TerraCUDA.LOGGER.info("TerraCUDA shadow: chunk {} -> {} block ids in {} us ({} chunks seen)",
                        pos, ids.map(blocks -> blocks.length).orElse(0), micros, count);
            }
        } catch (Throwable t) {
            // A diagnostic hook must never break world generation.
            if (REPORTED_FAILURE.compareAndSet(false, true)) {
                TerraCUDA.LOGGER.warn("TerraCUDA: shadow mode failed and will stay off", t);
            }
        }
    }
}
