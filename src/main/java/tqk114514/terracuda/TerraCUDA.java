package tqk114514.terracuda;

import org.slf4j.Logger;

import com.mojang.logging.LogUtils;

import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.Mod;
import tqk114514.terracuda.cuda.CudaEnvironment;

@Mod(TerraCUDA.MODID)
public class TerraCUDA {
    public static final String MODID = "terracuda";
    public static final Logger LOGGER = LogUtils.getLogger();

    public TerraCUDA(IEventBus modEventBus, ModContainer modContainer) {
        // Probing here (rather than lazily on the first chunk) makes the fallback decision visible in
        // the log at startup, and it is cheap: one cuInit plus a handful of attribute reads.
        CudaEnvironment environment = CudaEnvironment.detect();
        if (environment.available()) {
            LOGGER.info("TerraCUDA: GPU acceleration available - {}", environment.summary());
        } else {
            LOGGER.info("TerraCUDA: GPU acceleration unavailable, terrain generation stays on the CPU - {}",
                    environment.reason());
        }
    }
}
