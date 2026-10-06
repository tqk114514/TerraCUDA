package tqk114514.terracuda.worldgen;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import net.minecraft.world.level.levelgen.PositionalRandomFactory;

/**
 * Reads the two seed halves out of a live vanilla {@code PositionalRandomFactory}.
 *
 * <p>The aquifer and the ore veinifier derive their per-grid-cell randomness from these factories, and
 * the device needs the same numbers. The alternative — re-deriving them from the world seed — would
 * duplicate the whole {@code fromHashOf} chain and break the moment anything upstream changed, which is
 * the design doc's section 2.4 decision applied to the positional factories rather than the noises.
 */
public final class VanillaRandomExport {

    /** Per-class reflection handles, shared across every rules worker; safe for concurrent compute. */
    private static final Map<Class<?>, Field[]> FIELDS = new ConcurrentHashMap<>();

    private VanillaRandomExport() {
    }

    /** The seed pair behind {@code factory}, as this project's own factory type. */
    public static tqk114514.terracuda.random.PositionalRandomFactory export(PositionalRandomFactory factory) {
        Field[] fields = fieldsOf(factory.getClass());
        return new tqk114514.terracuda.random.PositionalRandomFactory(
                (long) read(fields[0], factory), (long) read(fields[1], factory));
    }

    private static Field[] fieldsOf(Class<?> type) {
        return FIELDS.computeIfAbsent(type, t -> new Field[] {
                declared(t, "seedLo"), declared(t, "seedHi")
        });
    }

    private static Field declared(Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException("vanilla layout changed: " + owner.getName() + "." + name, e);
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot access " + owner.getName() + "." + name, e);
        }
    }

    private static Object read(Field field, Object target) {
        try {
            return field.get(target);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot read " + field, e);
        }
    }
}
