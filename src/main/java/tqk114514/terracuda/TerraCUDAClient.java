package tqk114514.terracuda;

import net.neoforged.api.distmarker.Dist;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.common.EventBusSubscriber;
import net.neoforged.fml.common.Mod;
import net.neoforged.fml.event.lifecycle.FMLClientSetupEvent;

@Mod(value = TerraCUDA.MODID, dist = Dist.CLIENT)
@EventBusSubscriber(modid = TerraCUDA.MODID, value = Dist.CLIENT)
public class TerraCUDAClient {
    @SubscribeEvent
    static void onClientSetup(FMLClientSetupEvent event) {
        TerraCUDA.LOGGER.info("TerraCUDA client setup");
    }
}
