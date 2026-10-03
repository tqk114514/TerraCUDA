package tqk114514.terracuda;

import org.slf4j.Logger;

import com.mojang.logging.LogUtils;

import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.Mod;

@Mod(TerraCUDA.MODID)
public class TerraCUDA {
    public static final String MODID = "terracuda";
    public static final Logger LOGGER = LogUtils.getLogger();

    public TerraCUDA(IEventBus modEventBus, ModContainer modContainer) {
        LOGGER.info("TerraCUDA initializing");
    }
}
