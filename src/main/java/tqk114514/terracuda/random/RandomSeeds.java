package tqk114514.terracuda.random;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Seed derivation helpers mirroring {@code net.minecraft.world.level.levelgen.RandomSupport}.
 *
 * <p>These are the functions that turn a world seed, a string name or a block position into the
 * 128-bit state a {@link Xoroshiro128PlusPlus} starts from. They must be reproduced exactly: a single
 * bit of drift here decorrelates every downstream noise sample.
 */
public final class RandomSeeds {

    /** {@code RandomSupport.GOLDEN_RATIO_64}. */
    public static final long GOLDEN_RATIO_64 = -7046029254386353131L;
    /** {@code RandomSupport.SILVER_RATIO_64}. */
    public static final long SILVER_RATIO_64 = 7640891576956012809L;

    private RandomSeeds() {
    }

    /** The Stafford variant 13 64-bit mix, used to expand a legacy seed to 128 bits. */
    public static long mixStafford13(long z) {
        z = (z ^ z >>> 30) * -4658895280553007687L;
        z = (z ^ z >>> 27) * -7723592293110705685L;
        return z ^ z >>> 31;
    }

    public static Seed128bit upgradeSeedTo128bitUnmixed(long legacySeed) {
        long lowBits = legacySeed ^ SILVER_RATIO_64;
        long highBits = lowBits + GOLDEN_RATIO_64;
        return new Seed128bit(lowBits, highBits);
    }

    public static Seed128bit upgradeSeedTo128bit(long legacySeed) {
        return upgradeSeedTo128bitUnmixed(legacySeed).mixed();
    }

    /** {@code MD5(name)} split into two big-endian {@code long}s. */
    public static Seed128bit seedFromHashOf(String input) {
        byte[] digest = md5(input.getBytes(StandardCharsets.UTF_8));
        return new Seed128bit(fromBytesBigEndian(digest, 0), fromBytesBigEndian(digest, 8));
    }

    private static byte[] md5(byte[] input) {
        try {
            return MessageDigest.getInstance("MD5").digest(input);
        } catch (NoSuchAlgorithmException e) {
            // MD5 is a required algorithm on every Java platform.
            throw new IllegalStateException("MD5 is not available", e);
        }
    }

    /** Equivalent to Guava's {@code Longs.fromBytes}, which is big-endian. */
    private static long fromBytesBigEndian(byte[] bytes, int offset) {
        long value = 0L;
        for (int i = 0; i < 8; i++) {
            value = value << 8 | bytes[offset + i] & 0xFFL;
        }
        return value;
    }

    /** The 128-bit seed pair; {@code seedLo} is the low half, {@code seedHi} the high half. */
    public record Seed128bit(long seedLo, long seedHi) {

        public Seed128bit xor(long lo, long hi) {
            return new Seed128bit(this.seedLo ^ lo, this.seedHi ^ hi);
        }

        public Seed128bit xor(Seed128bit other) {
            return xor(other.seedLo, other.seedHi);
        }

        public Seed128bit mixed() {
            return new Seed128bit(mixStafford13(this.seedLo), mixStafford13(this.seedHi));
        }
    }
}
