package tqk114514.terracuda;

import org.slf4j.Logger;

import com.mojang.logging.LogUtils;

import net.neoforged.bus.api.IEventBus;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.server.ServerStoppingEvent;
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
        // A world's device buffers and daemon thread are released when its server stops, rather
        // than being left for the life of the process. Loading another world builds new ones.
        NeoForge.EVENT_BUS.addListener(ServerStoppingEvent.class,
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
