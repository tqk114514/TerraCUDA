package tqk114514.terracuda;

import org.slf4j.Logger;

import com.mojang.logging.LogUtils;

import net.neoforged.bus.api.IEventBus;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.server.ServerStoppedEvent;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.Mod;
import tqk114514.terracuda.cuda.CudaEnvironment;
import tqk114514.terracuda.worldgen.GpuWorldgenService;

@Mod(TerraCUDA.MODID)
public class TerraCUDA {
    public static final String MODID = "terracuda";
    public static final Logger LOGGER = LogUtils.getLogger();

    public TerraCUDA(IEventBus modEventBus, ModContainer modContainer) {
        // Probing here (rather than lazily on the first chunk) makes the fallback decision visible in
        // the log at startup, and it is cheap: one cuInit plus a handful of attribute reads.
        // A world's device buffers and daemon thread are released when its server has fully
        // stopped, rather than being left for the life of the process. Loading another world
        // builds new ones. The event must be ServerStopped and not ServerStopping: stopping
        // fires before the shutdown drain, and closing there strands any fill job still queued
        // -- the chunk's future never completes, the drain's hasWork() never clears, and the
        // "Saving world" screen hangs forever. A quit mid-generation reproduced exactly that.
        // Stopped fires after the drain, when nothing is in flight and closing strands nothing.
        NeoForge.EVENT_BUS.addListener(ServerStoppedEvent.class,
                event -> GpuWorldgenService.releaseAll());

        CudaEnvironment environment = CudaEnvironment.detect();
        if (environment.available()) {
            LOGGER.info("TerraCUDA: GPU acceleration available - {}", environment.summary());
        } else {
            LOGGER.info("TerraCUDA: GPU acceleration unavailable, terrain generation stays on the CPU - {}",
                    environment.reason());
        }
    }
}
