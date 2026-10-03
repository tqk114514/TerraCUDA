package tqk114514.terracuda.mixin;

import java.util.List;
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
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import tqk114514.terracuda.TerraCUDA;
import tqk114514.terracuda.config.TerracudaConfig;
import tqk114514.terracuda.gpu.ChunkCornerTables;
import tqk114514.terracuda.worldgen.GpuWorldgenService;

/**
 * Hooks {@code NoiseBasedChunkGenerator.fillFromNoise}, the entry point to the NOISE stage.
 *
 * <p>At this milestone the hook is deliberately <em>observational</em>: it runs the device path for the
 * chunk and logs what it produced, but never cancels the vanilla call and never touches the chunk. The
 * reason is that producing a correct chunk also needs the material rules, the aquifer and the paletted
 * container replay — M3's K3 and K4 — and returning a half-filled chunk would be worse than not
 * running at all. What this does buy is proof that lowering, module loading, upload and launch all
 * work against a live {@link RandomState} inside a real world.
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
            NoiseSettings settings = self.generatorSettings().value().noiseSettings();

            GpuWorldgenService service = SERVICES.computeIfAbsent(randomState, GpuWorldgenService::of);
            if (!service.isReady()) {
                if (REPORTED_FAILURE.compareAndSet(false, true)) {
                    TerraCUDA.LOGGER.info("TerraCUDA: shadow mode disabled, device unavailable: {}",
                            service.unavailableReason());
                }
                return;
            }

            ChunkPos pos = centerChunk.getPos();
            long start = System.nanoTime();
            Optional<List<ChunkCornerTables.MarkerGrid>> tables =
                    service.markerTables(settings, pos.x(), pos.z());
            long micros = (System.nanoTime() - start) / 1000;

            long count = CHUNKS.incrementAndGet();
            if (TerracudaConfig.verbose() || count % 256 == 1) {
                TerraCUDA.LOGGER.info("TerraCUDA shadow: chunk {} -> {} marker tables in {} us ({} chunks seen)",
                        pos, tables.map(List::size).orElse(0), micros, count);
            }
        } catch (Throwable t) {
            // A diagnostic hook must never break world generation.
            if (REPORTED_FAILURE.compareAndSet(false, true)) {
                TerraCUDA.LOGGER.warn("TerraCUDA: shadow mode failed and will stay off", t);
            }
        }
    }
}
