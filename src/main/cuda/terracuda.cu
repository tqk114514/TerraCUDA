// TerraCUDA — GPU kernels for Minecraft world generation.
//
// This file is compiled at build time into cubin / PTX images that ship inside the mod jar and are
// loaded through the CUDA Driver API (see tqk114514.terracuda.cuda).
//
// BIT-EXACTNESS CONTRACT
// ----------------------
// Every kernel here must reproduce the corresponding Minecraft code *bit for bit*, not approximately.
// Terrain is a chaotic function of the density field: a difference in the last mantissa bit changes
// where a cave wall lands. Two things therefore have to hold, and both are enforced by the build:
//
//   1. FMA contraction is disabled (-fmad=false) and --use_fast_math is off. Java evaluates
//      `a * b + c` as two correctly rounded operations; nvcc would otherwise fuse them into one.
//   2. The arithmetic below is transcribed with the same association and the same operand order as
//      the Java source. Do not "tidy" expressions, fold constants, or reorder additions.
//
// Double precision is used throughout. On consumer GPUs FP64 throughput is a fraction of FP32, but
// the whole point of the port is that the numbers match; see the design doc, section 5.1.
//
// This source is consumed two ways:
//   * nvcc -cubin / -ptx at build time, when a host compiler is present (produces the shipped images);
//   * NVRTC at runtime otherwise. NVRTC predefines __CUDACC_RTC__ and does not provide
//     cuda_runtime.h, hence the guard below. Nothing else in this file needs the runtime header.

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

namespace {

// ============================================================================================
// ImprovedNoise
// ============================================================================================

// SimplexNoise.GRADIENT — the 16 corner directions, transcribed verbatim.
__constant__ int GRADIENT[16][3] = {
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
    {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
    {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1}
};

__device__ __forceinline__ double gradDot(int hash, double x, double y, double z) {
    const int* g = GRADIENT[hash & 15];
    return g[0] * x + g[1] * y + g[2] * z;
}

__device__ __forceinline__ double lerp(double alpha, double start, double end) {
    return start + alpha * (end - start);
}

// Mth.lerp(float, float, float): the cubic spline path stays in float throughout, and widening it to
// double here would round differently.
__device__ __forceinline__ float lerpf(float alpha, float start, float end) {
    return start + alpha * (end - start);
}

__device__ __forceinline__ double lerp2(double a1, double a2,
                                        double x00, double x10, double x01, double x11) {
    return lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11));
}

__device__ __forceinline__ double lerp3(double a1, double a2, double a3,
                                        double x000, double x100, double x010, double x110,
                                        double x001, double x101, double x011, double x111) {
    return lerp(a3, lerp2(a1, a2, x000, x100, x010, x110),
                lerp2(a1, a2, x001, x101, x011, x111));
}

__device__ __forceinline__ double smoothstep(double x) {
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// p(x) = p[x & 0xFF] & 0xFF. The table is uploaded as unsigned char, so the Java `& 0xFF` that
// widens the signed byte is already accounted for.
__device__ __forceinline__ int perm(const unsigned char* p, int x) {
    return p[x & 0xFF];
}

// ImprovedNoise.noise(x, y, z, yScale, yFudge). The two extra parameters are zero for ordinary
// terrain noise, which short-circuits the fudge branch; BlendedNoise passes real values.
__device__ double improvedNoise(const unsigned char* p, double xo, double yo, double zo,
                                double inX, double inY, double inZ,
                                double yScale, double yFudge) {
    double x = inX + xo;
    double y = inY + yo;
    double z = inZ + zo;
    int xf = (int) floor(x);
    int yf = (int) floor(y);
    int zf = (int) floor(z);
    double xr = x - xf;
    double yr = y - yf;
    double zr = z - zf;

    double yrFudge;
    if (yScale != 0.0) {
        double fudgeLimit = (yFudge >= 0.0 && yFudge < yr) ? yFudge : yr;
        yrFudge = floor(fudgeLimit / yScale + 1.0E-7F) * yScale;
    } else {
        yrFudge = 0.0;
    }

    int x0 = perm(p, xf);
    int x1 = perm(p, xf + 1);
    int xy00 = perm(p, x0 + yf);
    int xy01 = perm(p, x0 + yf + 1);
    int xy10 = perm(p, x1 + yf);
    int xy11 = perm(p, x1 + yf + 1);

    double lerpYr = yr - yrFudge;
    double d000 = gradDot(perm(p, xy00 + zf), xr, lerpYr, zr);
    double d100 = gradDot(perm(p, xy10 + zf), xr - 1.0, lerpYr, zr);
    double d010 = gradDot(perm(p, xy01 + zf), xr, lerpYr - 1.0, zr);
    double d110 = gradDot(perm(p, xy11 + zf), xr - 1.0, lerpYr - 1.0, zr);
    double d001 = gradDot(perm(p, xy00 + zf + 1), xr, lerpYr, zr - 1.0);
    double d101 = gradDot(perm(p, xy10 + zf + 1), xr - 1.0, lerpYr, zr - 1.0);
    double d011 = gradDot(perm(p, xy01 + zf + 1), xr, lerpYr - 1.0, zr - 1.0);
    double d111 = gradDot(perm(p, xy11 + zf + 1), xr - 1.0, lerpYr - 1.0, zr - 1.0);

    double xAlpha = smoothstep(xr);
    double yAlpha = smoothstep(yr);
    double zAlpha = smoothstep(zr);
    return lerp3(xAlpha, yAlpha, zAlpha, d000, d100, d010, d110, d001, d101, d011, d111);
}

__device__ __forceinline__ double improvedNoise(const unsigned char* p, double xo, double yo,
                                                double zo, double x, double y, double z) {
    return improvedNoise(p, xo, yo, zo, x, y, z, 0.0, 0.0);
}

// ============================================================================================
// Density program interpreter
//
// Evaluates a lowered DensityProgram (see tqk114514.terracuda.density) at a batch of block
// positions. Every table lives in one read-only device blob; `offsets` indexes into it, which keeps
// the kernel signature small and the argument order out of the picture.
//
// The instruction stream is emitted parent-before-children, so every child has a HIGHER index than
// its parent. That makes a single reverse loop a valid topological evaluation: no stack, no
// recursion, and the per-thread scratch is one double per instruction.
// ============================================================================================

}  // namespace

#define TC_OFF_OPS 0
#define TC_OFF_IA 1
#define TC_OFF_IB 2
#define TC_OFF_IC 3
#define TC_OFF_ID 4
#define TC_OFF_DA 5
#define TC_OFF_DB 6
#define TC_OFF_NOISE_FIRST_OCTAVE 7
#define TC_OFF_NOISE_OCTAVE_START 8
#define TC_OFF_NOISE_OCTAVE_COUNT 9
#define TC_OFF_NOISE_VALUE_FACTOR 10
#define TC_OFF_NOISE_AMP_START 11
#define TC_OFF_OCTAVE_ORIGINS 12
#define TC_OFF_OCTAVE_PERMUTATIONS 13
#define TC_OFF_OCTAVE_VALID 14
#define TC_OFF_AMPLITUDES 15
#define TC_OFF_SPLINE_KIND 16
#define TC_OFF_SPLINE_CONSTANT 17
#define TC_OFF_SPLINE_COORDINATE 18
#define TC_OFF_SPLINE_LOC_START 19
#define TC_OFF_SPLINE_LOC_COUNT 20
#define TC_OFF_SPLINE_CHILD_START 21
#define TC_OFF_SPLINE_LOCATIONS 22
#define TC_OFF_SPLINE_DERIVATIVES 23
#define TC_OFF_SPLINE_CHILDREN 24
#define TC_OFF_BLENDED_OCTAVE_START 25
#define TC_OFF_BLENDED_XZ_MULT 26
#define TC_OFF_BLENDED_Y_MULT 27
#define TC_OFF_BLENDED_XZ_FACTOR 28
#define TC_OFF_BLENDED_Y_FACTOR 29
#define TC_OFF_BLENDED_SMEAR 30
#define TC_OFF_BLENDED_MIN_COUNT 31
#define TC_OFF_BLENDED_MAIN_COUNT 32
#define TC_OFFSET_COUNT 33

// Opcodes, mirroring DensityProgram.
#define TC_CONSTANT 1
#define TC_Y_CLAMPED_GRADIENT 2
#define TC_NOISE 3
#define TC_SHIFTED_NOISE 4
#define TC_SHIFT 5
#define TC_SHIFT_A 6
#define TC_SHIFT_B 7
#define TC_WEIRD_SCALED_TYPE1 8
#define TC_WEIRD_SCALED_TYPE2 9
#define TC_SPLINE 10
#define TC_RANGE_CHOICE 11
#define TC_BLEND_DENSITY 12
#define TC_BLEND_ALPHA 13
#define TC_BLEND_OFFSET 14
#define TC_BEARDIFIER 15
#define TC_BLENDED_NOISE 16
#define TC_ADD 17
#define TC_MUL 18
#define TC_MIN 19
#define TC_MAX 20
#define TC_MUL_ADD 21
#define TC_ABS 22
#define TC_SQUARE 23
#define TC_CUBE 24
#define TC_HALF_NEGATIVE 25
#define TC_QUARTER_NEGATIVE 26
#define TC_INVERT 27
#define TC_SQUEEZE 28
#define TC_CLAMP 29
#define TC_FIND_TOP_SURFACE 30
#define TC_OVERRIDE 31

#define TC_SPLINE_CONSTANT 0
#define TC_SPLINE_MULTIPOINT 1

// Blend orders, mirroring CornerInterpolation.
#define TC_ORDER_INCREMENTAL 0
#define TC_ORDER_LERP3 1

namespace {

struct TcProgram {
    const char* blob;
    const long long* offsets;
    int root;
    int instructionCount;
    int blockX;
    int blockY;
    int blockZ;
};

__device__ __forceinline__ const int* tcInts(const TcProgram& p, int index) {
    return (const int*) (p.blob + p.offsets[index]);
}

__device__ __forceinline__ const double* tcDoubles(const TcProgram& p, int index) {
    return (const double*) (p.blob + p.offsets[index]);
}

__device__ __forceinline__ const float* tcFloats(const TcProgram& p, int index) {
    return (const float*) (p.blob + p.offsets[index]);
}

__device__ __forceinline__ const unsigned char* tcBytes(const TcProgram& p, int index) {
    return (const unsigned char*) (p.blob + p.offsets[index]);
}

// The 2^25 coordinate wrap used by PerlinNoise.
__device__ __forceinline__ double tcWrap(double x) {
    return x - floor(x / 3.3554432E7 + 0.5) * 3.3554432E7;
}

// Java's Math.pow(2.0, n) is exact for integer n; CUDA's pow() is allowed up to 2 ulp of error, which
// is enough to change the last bits of the density. ldexp is exact by construction.
__device__ __forceinline__ double tcPow2(int exponent) {
    return ldexp(1.0, exponent);
}

__device__ __forceinline__ double tcClampedLerp(double factor, double min, double max) {
    if (factor < 0.0) {
        return min;
    }
    return factor > 1.0 ? max : min + factor * (max - min);
}

__device__ __forceinline__ double tcClampedMap(double value, double fromMin, double fromMax,
                                               double toMin, double toMax) {
    double t = (value - fromMin) / (fromMax - fromMin);
    if (t < 0.0) {
        return toMin;
    }
    if (t > 1.0) {
        return toMax;
    }
    return toMin + t * (toMax - toMin);
}

__device__ __forceinline__ double tcSpaghettiRarity3D(double r) {
    if (r < -0.5) {
        return 0.75;
    }
    if (r < 0.0) {
        return 1.0;
    }
    return r < 0.5 ? 1.5 : 2.0;
}

__device__ __forceinline__ double tcSpaghettiRarity2D(double r) {
    if (r < -0.75) {
        return 0.5;
    }
    if (r < -0.5) {
        return 0.75;
    }
    if (r < 0.5) {
        return 1.0;
    }
    return r < 0.75 ? 2.0 : 3.0;
}

// One PerlinNoise stack, evaluated exactly as PerlinNoise.getValue does.
__device__ double tcPerlin(const TcProgram& p, int octaveStart, int octaveCount, int amplitudeStart,
                           int firstOctave, double x, double y, double z) {
    const double* amplitudes = tcDoubles(p, TC_OFF_AMPLITUDES) + amplitudeStart;
    const double* origins = tcDoubles(p, TC_OFF_OCTAVE_ORIGINS);
    const unsigned char* permutations = tcBytes(p, TC_OFF_OCTAVE_PERMUTATIONS);
    const int* valid = tcInts(p, TC_OFF_OCTAVE_VALID);

    double factor = tcPow2(firstOctave);
    double valueFactor = tcPow2(octaveCount - 1) / (tcPow2(octaveCount) - 1.0);
    double value = 0.0;

    for (int i = 0; i < octaveCount; i++) {
        int octave = octaveStart + i;
        if (valid[octave]) {
            double noiseValue = improvedNoise(permutations + 256 * octave,
                    origins[3 * octave], origins[3 * octave + 1], origins[3 * octave + 2],
                    tcWrap(x * factor), tcWrap(y * factor), tcWrap(z * factor));
            value += amplitudes[i] * noiseValue * valueFactor;
        }
        factor *= 2.0;
        valueFactor /= 2.0;
    }
    return value;
}

// NormalNoise.getValue: two Perlin stacks, the second sampled at a slightly different scale.
__device__ double tcNormalNoise(const TcProgram& p, int noiseIndex, double x, double y, double z) {
    const int* firstOctave = tcInts(p, TC_OFF_NOISE_FIRST_OCTAVE);
    const int* octaveStart = tcInts(p, TC_OFF_NOISE_OCTAVE_START);
    const int* octaveCount = tcInts(p, TC_OFF_NOISE_OCTAVE_COUNT);
    const int* amplitudeStart = tcInts(p, TC_OFF_NOISE_AMP_START);
    const double* valueFactor = tcDoubles(p, TC_OFF_NOISE_VALUE_FACTOR);

    int count = octaveCount[noiseIndex];
    int start = octaveStart[noiseIndex];
    int amps = amplitudeStart[noiseIndex];
    int first = firstOctave[noiseIndex];

    double a = tcPerlin(p, start, count, amps, first, x, y, z);
    double b = tcPerlin(p, start + count, count, amps, first,
            x * 1.0181268882175227, y * 1.0181268882175227, z * 1.0181268882175227);
    return (a + b) * valueFactor[noiseIndex];
}

__device__ float tcSpline(const TcProgram& p, const double* values, int index);

__device__ __forceinline__ float tcLinearExtend(float input, const float* locations,
                                                const float* derivatives, int index, float value) {
    float derivative = derivatives[index];
    return derivative == 0.0F ? value : value + derivative * (input - locations[index]);
}

__device__ float tcSpline(const TcProgram& p, const double* values, int index) {
    if (tcInts(p, TC_OFF_SPLINE_KIND)[index] == TC_SPLINE_CONSTANT) {
        return tcFloats(p, TC_OFF_SPLINE_CONSTANT)[index];
    }

    const int* locationStart = tcInts(p, TC_OFF_SPLINE_LOC_START);
    const int* locationCount = tcInts(p, TC_OFF_SPLINE_LOC_COUNT);
    const int* childStart = tcInts(p, TC_OFF_SPLINE_CHILD_START);
    const float* locations = tcFloats(p, TC_OFF_SPLINE_LOCATIONS) + locationStart[index];
    const float* derivatives = tcFloats(p, TC_OFF_SPLINE_DERIVATIVES) + locationStart[index];
    const int* children = tcInts(p, TC_OFF_SPLINE_CHILDREN) + childStart[index];

    float input = (float) values[tcInts(p, TC_OFF_SPLINE_COORDINATE)[index]];
    int count = locationCount[index];

    // Mth.binarySearch(0, count, i -> input < locations[i]) - 1
    int from = 0;
    int length = count;
    while (length > 0) {
        int half = length / 2;
        int middle = from + half;
        if (input < locations[middle]) {
            length = half;
        } else {
            from = middle + 1;
            length -= half + 1;
        }
    }
    int start = from - 1;
    int lastIndex = count - 1;

    if (start < 0) {
        return tcLinearExtend(input, locations, derivatives, 0, tcSpline(p, values, children[0]));
    }
    if (start == lastIndex) {
        return tcLinearExtend(input, locations, derivatives, lastIndex,
                tcSpline(p, values, children[lastIndex]));
    }

    float x1 = locations[start];
    float x2 = locations[start + 1];
    float t = (input - x1) / (x2 - x1);
    float y1 = tcSpline(p, values, children[start]);
    float y2 = tcSpline(p, values, children[start + 1]);
    float d1 = derivatives[start];
    float d2 = derivatives[start + 1];
    float a = d1 * (x2 - x1) - (y2 - y1);
    float b = -d2 * (x2 - x1) + (y2 - y1);
    return lerpf(t, y1, y2) + t * (1.0F - t) * lerpf(t, a, b);
}

// BlendedNoise.compute, over the raw octave levels: min stack, max stack, then main.
__device__ double tcBlendedNoise(const TcProgram& p, int index, int blockX, int blockY, int blockZ) {
    const int* octaveStart = tcInts(p, TC_OFF_BLENDED_OCTAVE_START);
    const int* limitCount = tcInts(p, TC_OFF_BLENDED_MIN_COUNT);
    const int* mainCount = tcInts(p, TC_OFF_BLENDED_MAIN_COUNT);
    const double* xzMultiplier = tcDoubles(p, TC_OFF_BLENDED_XZ_MULT);
    const double* yMultiplier = tcDoubles(p, TC_OFF_BLENDED_Y_MULT);
    const double* xzFactor = tcDoubles(p, TC_OFF_BLENDED_XZ_FACTOR);
    const double* yFactor = tcDoubles(p, TC_OFF_BLENDED_Y_FACTOR);
    const double* smear = tcDoubles(p, TC_OFF_BLENDED_SMEAR);

    const double* origins = tcDoubles(p, TC_OFF_OCTAVE_ORIGINS);
    const unsigned char* permutations = tcBytes(p, TC_OFF_OCTAVE_PERMUTATIONS);

    int start = octaveStart[index];
    int limits = limitCount[index];
    int mains = mainCount[index];
    int mainBase = start + 2 * limits;

    double limitX = blockX * xzMultiplier[index];
    double limitY = blockY * yMultiplier[index];
    double limitZ = blockZ * xzMultiplier[index];
    double mainX = limitX / xzFactor[index];
    double mainY = limitY / yFactor[index];
    double mainZ = limitZ / xzFactor[index];
    double limitSmear = yMultiplier[index] * smear[index];
    double mainSmear = limitSmear / yFactor[index];

    double blendMin = 0.0;
    double blendMax = 0.0;
    double mainNoiseValue = 0.0;
    double pow2 = 1.0;

    for (int i = 0; i < 8 && i < mains; i++) {
        int octave = mainBase + (mains - 1 - i);
        mainNoiseValue += improvedNoise(permutations + 256 * octave,
                origins[3 * octave], origins[3 * octave + 1], origins[3 * octave + 2],
                tcWrap(mainX * pow2), tcWrap(mainY * pow2), tcWrap(mainZ * pow2),
                mainSmear * pow2, mainY * pow2) / pow2;
        pow2 /= 2.0;
    }

    double factor = (mainNoiseValue / 10.0 + 1.0) / 2.0;
    bool isMax = factor >= 1.0;
    bool isMin = factor <= 0.0;
    pow2 = 1.0;

    for (int i = 0; i < 16 && i < limits; i++) {
        double wx = tcWrap(limitX * pow2);
        double wy = tcWrap(limitY * pow2);
        double wz = tcWrap(limitZ * pow2);
        double yScalePow = limitSmear * pow2;
        if (!isMax) {
            int octave = start + (limits - 1 - i);
            blendMin += improvedNoise(permutations + 256 * octave,
                    origins[3 * octave], origins[3 * octave + 1], origins[3 * octave + 2],
                    wx, wy, wz, yScalePow, limitY * pow2) / pow2;
        }
        if (!isMin) {
            int octave = start + limits + (limits - 1 - i);
            blendMax += improvedNoise(permutations + 256 * octave,
                    origins[3 * octave], origins[3 * octave + 1], origins[3 * octave + 2],
                    wx, wy, wz, yScalePow, limitY * pow2) / pow2;
        }
        pow2 /= 2.0;
    }

    return tcClampedLerp(factor, blendMin / 512.0, blendMax / 512.0) / 128.0;
}

__device__ double tcCompute(TcProgram& p, double* values, int pc) {
    const int* ops = tcInts(p, TC_OFF_OPS);
    const int* ia = tcInts(p, TC_OFF_IA);
    const int* ib = tcInts(p, TC_OFF_IB);
    const int* ic = tcInts(p, TC_OFF_IC);
    const int* id = tcInts(p, TC_OFF_ID);
    const double* da = tcDoubles(p, TC_OFF_DA);
    const double* db = tcDoubles(p, TC_OFF_DB);

    switch (ops[pc]) {
        case TC_CONSTANT:
            return da[pc];
        case TC_Y_CLAMPED_GRADIENT:
            return tcClampedMap((double) p.blockY, (double) ia[pc], (double) ib[pc], da[pc], db[pc]);
        case TC_NOISE:
            return tcNormalNoise(p, ia[pc], p.blockX * da[pc], p.blockY * db[pc], p.blockZ * da[pc]);
        case TC_SHIFTED_NOISE: {
            double x = p.blockX * da[pc] + values[ia[pc]];
            double y = p.blockY * db[pc] + values[ib[pc]];
            double z = p.blockZ * da[pc] + values[ic[pc]];
            return tcNormalNoise(p, id[pc], x, y, z);
        }
        case TC_SHIFT:
            return tcNormalNoise(p, ia[pc], p.blockX * 0.25, p.blockY * 0.25, p.blockZ * 0.25) * 4.0;
        case TC_SHIFT_A:
            return tcNormalNoise(p, ia[pc], p.blockX * 0.25, 0.0, p.blockZ * 0.25) * 4.0;
        case TC_SHIFT_B:
            return tcNormalNoise(p, ia[pc], p.blockZ * 0.25, p.blockX * 0.25, 0.0) * 4.0;
        case TC_WEIRD_SCALED_TYPE1:
        case TC_WEIRD_SCALED_TYPE2: {
            double input = values[ia[pc]];
            double rarity = ops[pc] == TC_WEIRD_SCALED_TYPE1
                    ? tcSpaghettiRarity3D(input)
                    : tcSpaghettiRarity2D(input);
            return rarity * fabs(tcNormalNoise(p, ib[pc],
                    p.blockX / rarity, p.blockY / rarity, p.blockZ / rarity));
        }
        case TC_SPLINE:
            return (double) tcSpline(p, values, ia[pc]);
        case TC_RANGE_CHOICE: {
            double input = values[ia[pc]];
            return input >= da[pc] && input < db[pc] ? values[ib[pc]] : values[ic[pc]];
        }
        case TC_BLEND_DENSITY:
            return values[ia[pc]];
        case TC_BLEND_ALPHA:
            return 1.0;
        case TC_BLEND_OFFSET:
            return 0.0;
        case TC_BEARDIFIER:
            return 0.0;
        case TC_BLENDED_NOISE:
            return tcBlendedNoise(p, ia[pc], p.blockX, p.blockY, p.blockZ);
        case TC_ADD:
            return values[ia[pc]] + values[ib[pc]];
        case TC_MUL: {
            // Vanilla returns 0.0 without evaluating the right operand; the sign of zero is observable.
            double left = values[ia[pc]];
            return left == 0.0 ? 0.0 : left * values[ib[pc]];
        }
        case TC_MIN:
            return fmin(values[ia[pc]], values[ib[pc]]);
        case TC_MAX:
            return fmax(values[ia[pc]], values[ib[pc]]);
        case TC_MUL_ADD: {
            double input = values[ia[pc]];
            return ib[pc] == 1 ? input * da[pc] : input + da[pc];
        }
        case TC_ABS:
            return fabs(values[ia[pc]]);
        case TC_SQUARE: {
            double v = values[ia[pc]];
            return v * v;
        }
        case TC_CUBE: {
            double v = values[ia[pc]];
            return v * v * v;
        }
        case TC_HALF_NEGATIVE: {
            double v = values[ia[pc]];
            return v > 0.0 ? v : v * 0.5;
        }
        case TC_QUARTER_NEGATIVE: {
            double v = values[ia[pc]];
            return v > 0.0 ? v : v * 0.25;
        }
        case TC_INVERT:
            return 1.0 / values[ia[pc]];
        case TC_SQUEEZE: {
            double v = values[ia[pc]];
            double c = v < -1.0 ? -1.0 : fmin(v, 1.0);
            return c / 2.0 - c * c * c / 24.0;
        }
        case TC_CLAMP: {
            double v = values[ia[pc]];
            return v < da[pc] ? da[pc] : fmin(v, db[pc]);
        }
        case TC_FIND_TOP_SURFACE: {
            // Probes downwards in cellHeight steps until the density turns positive. Each probe is a
            // different y, so the density subtree is re-evaluated from scratch — which is only sound
            // when this instruction is the root, and the image is built to guarantee that.
            int densityPc = ia[pc];
            int lowerBound = ic[pc];
            int cellHeight = id[pc];
            int topY = (int) floor(values[ib[pc]] / (double) cellHeight) * cellHeight;
            if (topY <= lowerBound) {
                return (double) lowerBound;
            }

            int savedY = p.blockY;
            double result = (double) lowerBound;
            for (int probeY = topY; probeY >= lowerBound; probeY -= cellHeight) {
                p.blockY = probeY;
                for (int i = 0; i <= densityPc; i++) {
                    values[i] = tcCompute(p, values, i);
                }
                if (values[densityPc] > 0.0) {
                    result = (double) probeY;
                    break;
                }
            }
            p.blockY = savedY;
            return result;
        }
        case TC_OVERRIDE:
            // A leaf the caller filled in before the sweep — an interpolated marker blended down from
            // its corner grid. Returning the slot is what keeps the forward pass from clobbering it.
            return values[pc];
        default:
            // Anything newer than this interpreter. NaN rather than a guess: the caller compares
            // against the CPU reference and a mismatch is a loud failure, not silent wrong terrain.
            return __longlong_as_double(0x7FF8000000000000LL);
    }
}

// One block's trilinear blend out of a marker's corner grid. Both orders live here so the two
// kernels that need a blend cannot drift apart.
__device__ __forceinline__ double tcBlendMarker(const double* __restrict__ corners, int base,
                                                int cell, int cornerCountY, int order,
                                                double factorX, double factorY, double factorZ) {
    const double* v = corners + base;
    int stepX = 5 * cornerCountY;
    double n000 = v[cell];
    double n001 = v[cell + cornerCountY];
    double n100 = v[cell + stepX];
    double n101 = v[cell + stepX + cornerCountY];
    double n010 = v[cell + 1];
    double n011 = v[cell + cornerCountY + 1];
    double n110 = v[cell + stepX + 1];
    double n111 = v[cell + stepX + cornerCountY + 1];

    if (order == TC_ORDER_LERP3) {
        return lerp3(factorX, factorY, factorZ, n000, n100, n010, n110, n001, n101, n011, n111);
    }
    // NoiseInterpolator.updateForY/X/Z: Y, then X, then Z.
    double xz00 = lerp(factorY, n000, n010);
    double xz10 = lerp(factorY, n100, n110);
    double xz01 = lerp(factorY, n001, n011);
    double xz11 = lerp(factorY, n101, n111);
    double z0 = lerp(factorX, xz00, xz10);
    double z1 = lerp(factorX, xz01, xz11);
    return lerp(factorZ, z0, z1);
}

// Splits a linear block index into chunk-local coordinates, in the layout the caller uses for its
// per-block arrays: (x * sizeXZ + z) * height + y. Returned once so the block position and the cell
// coordinates cannot be derived two different ways and drift.
__device__ __forceinline__ void tcBlockLocal(int index, int sizeXZ, int height,
                                             int* xLocal, int* zLocal, int* yLocal) {
    *yLocal = index % height;
    int rest = index / height;
    *zLocal = rest % sizeXZ;
    *xLocal = rest / sizeXZ;
}

}  // namespace

// ============================================================================================
// Kernels
// ============================================================================================

extern "C" __global__ void terracuda_improved_noise_points(
        const unsigned char* __restrict__ permutation,
        double xo, double yo, double zo,
        const double* __restrict__ points,   // 3 * count, interleaved xyz
        double* __restrict__ results,        // count
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const double* point = points + 3 * (long long) index;
    results[index] = improvedNoise(permutation, xo, yo, zo, point[0], point[1], point[2]);
}

// One thread per block position. `scratch` is count * instructionCount doubles, owned by the caller:
// per-thread local memory would be far too large, and a device-side malloc per thread is worse.
//
// `roots` lets one launch evaluate several sub-graphs of the same program — a marker's root per point.
// That matters more than it looks: a launch of 1225 points against a 450-step serial chain leaves the
// device almost idle, so folding every marker's grid into one launch is worth several times the cost
// of the extra bookkeeping. Pass NULL to use the single `root` for every point.
extern "C" __global__ void terracuda_density_evaluate(
        const char* __restrict__ blob,
        const long long* __restrict__ offsets,
        int instructionCount,
        int root,
        const int* __restrict__ roots,
        const int* __restrict__ points,   // 3 * count block coordinates
        double* __restrict__ scratch,     // count * instructionCount
        double* __restrict__ results,     // count
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    TcProgram program;
    program.blob = blob;
    program.offsets = offsets;
    program.root = roots == 0 ? root : roots[index];
    program.instructionCount = instructionCount;
    program.blockX = points[3 * index];
    program.blockY = points[3 * index + 1];
    program.blockZ = points[3 * index + 2];

    double* values = scratch + (long long) index * instructionCount;

    // The image is emitted in post-order, so every instruction sits after the ones it reads and a
    // single forward pass is a valid evaluation.
    for (int pc = 0; pc < instructionCount; pc++) {
        values[pc] = tcCompute(program, values, pc);
    }
    results[index] = values[program.root];
}

// The chunk-level pass: blend each interpolated marker down to block resolution, then evaluate the
// DAG above it, all in one launch. One double comes back per block.
//
// Both halves used to run on the CPU and measured at about eleven milliseconds per chunk — seven and
// a half of it the interpreter walking the few dozen instructions above the markers, and three and a
// half the blend. Neither is large in absolute terms, and both are embarrassingly parallel, which is
// exactly the shape that does not belong on one core. The `blob` here is the reduced image built by
// `DensityProgramImage.ofBlocks`: the marker subgraphs are gone, replaced by TC_OVERRIDE leaves that
// the blend fills in below.
//
// `bases` and `slots` are parallel per-marker arrays: where the marker's corner grid starts in
// `corners`, and which instruction to write its blended value to.
extern "C" __global__ void terracuda_density_blocks(
        const char* __restrict__ blob,
        const long long* __restrict__ offsets,
        int instructionCount,
        int root,
        const double* __restrict__ corners,
        const int* __restrict__ bases,
        const int* __restrict__ slots,
        int markerCount,
        int cellWidth, int cellHeight, int cornerCountY, int sizeXZ, int height, int order,
        int chunkMinBlockX, int minY, int chunkMinBlockZ,
        double* __restrict__ scratch,     // count * instructionCount
        double* __restrict__ results,     // count
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    TcProgram program;
    program.blob = blob;
    program.offsets = offsets;
    program.root = root;
    program.instructionCount = instructionCount;

    int xLocal, zLocal, yLocal;
    tcBlockLocal(index, sizeXZ, height, &xLocal, &zLocal, &yLocal);
    program.blockX = chunkMinBlockX + xLocal;
    program.blockY = minY + yLocal;
    program.blockZ = chunkMinBlockZ + zLocal;

    int cellX = xLocal / cellWidth;
    int cellY = yLocal / cellHeight;
    int cellZ = zLocal / cellWidth;
    double factorX = (double) (xLocal % cellWidth) / (double) cellWidth;
    double factorY = (double) (yLocal % cellHeight) / (double) cellHeight;
    double factorZ = (double) (zLocal % cellWidth) / (double) cellWidth;

    double* values = scratch + (long long) index * instructionCount;

    int cell = (cellX * 5 + cellZ) * cornerCountY + cellY;
    for (int m = 0; m < markerCount; m++) {
        values[slots[m]] = tcBlendMarker(corners, bases[m], cell, cornerCountY, order,
                factorX, factorY, factorZ);
    }

    // The image is in post-order, so every instruction sits after the ones it reads. TC_OVERRIDE
    // returns its own slot, so the sweep leaves the blends above untouched.
    for (int pc = 0; pc < instructionCount; pc++) {
        values[pc] = tcCompute(program, values, pc);
    }
    results[index] = values[program.root];
}

// ============================================================================================
// Xoroshiro128++ — the positional draws the material rules need (G2a)
// ============================================================================================

// Mirrors tqk114514.terracuda.random.Xoroshiro128PlusPlus: two longs of state, vanilla's
// golden/silver substitution on an all-zero state, and the ++ scrambler. No floating point
// anywhere in the generator itself.
struct TcXoroshiro {
    unsigned long long seedLo;
    unsigned long long seedHi;
};

__device__ __forceinline__ TcXoroshiro tcXoroshiro(long long seedLo, long long seedHi) {
    TcXoroshiro state;
    state.seedLo = (unsigned long long) seedLo;
    state.seedHi = (unsigned long long) seedHi;
    if ((state.seedLo | state.seedHi) == 0ULL) {
        // RandomSupport.GOLDEN_RATIO_64 and SILVER_RATIO_64. Vanilla substitutes these for an
        // all-zero state instead of rejecting it, and the substitution is observable output
        // whenever a positional derivation lands on zero, so it is part of the port.
        state.seedLo = 0x9E3779B97F4A7C15ULL;
        state.seedHi = 0x6A09E667F3BCC909ULL;
    }
    return state;
}

__device__ __forceinline__ long long tcXoroshiroNextLong(TcXoroshiro* state) {
    unsigned long long s0 = state->seedLo;
    unsigned long long s1 = state->seedHi;
    long long result = (long long) (((s0 + s1) << 17) | ((s0 + s1) >> 47)) + (long long) s0;
    s1 ^= s0;
    state->seedLo = (((s0 << 49) | (s0 >> 15)) ^ s1 ^ (s1 << 21));
    state->seedHi = ((s1 << 28) | (s1 >> 36));
    return result;
}

// Mth.getSeed. The x product is a 32-bit int multiply in Java -- it wraps in 32 bits and only
// the widened result joins the long arithmetic -- so it is wrapped in unsigned 32 bits here and
// reinterpreted, which is the same two's-complement value, then sign-extended like the Java
// widening does. The rest is 64-bit throughout.
__device__ __forceinline__ long long tcGetSeed(int x, int y, int z) {
    int xProduct = (int) ((unsigned) x * 3129871u);
    long long seed = (long long) xProduct ^ ((long long) z) * 116129781LL ^ (long long) y;
    seed = seed * seed * 42317861LL + seed * 11LL;
    return seed >> 16;
}

// XoroshiroRandom.nextFloat: the top 24 bits widened to float with round-to-nearest, then a
// float multiply -- the same long-to-float conversion and float arithmetic Java's bytecode
// performs, not the double shortcut that would move the last bits.
__device__ __forceinline__ float tcXoroshiroNextFloat(TcXoroshiro* state) {
    unsigned long long bits = (unsigned long long) tcXoroshiroNextLong(state);
    return (float) (bits >> 40) * 5.9604645E-8f;
}

// XoroshiroRandom.nextInt(bound): vanilla's unbiased sampling. The draw is widened to
// unsigned, multiplied by the bound in 64 bits, and rejected from the low 32 bits until it
// lands outside the biased tail -- the loop is the point, and the modulo of the two's
// complement negation is computed exactly as Integer.remainderUnsigned does.
__device__ __forceinline__ int tcXoroshiroNextInt(TcXoroshiro* state, int bound) {
    unsigned long long randomBits = (unsigned int) (int) tcXoroshiroNextLong(state);
    unsigned long long multiplied = randomBits * (unsigned long long) (unsigned) bound;
    unsigned long long fractional = multiplied & 4294967295ULL;
    if (fractional < (unsigned long long) (unsigned) bound) {
        unsigned long long unbiasedStart =
                (unsigned long long) (unsigned) (~bound + 1) % (unsigned long long) (unsigned) bound;
        while (fractional < unbiasedStart) {
            randomBits = (unsigned int) (int) tcXoroshiroNextLong(state);
            multiplied = randomBits * (unsigned long long) (unsigned) bound;
            fractional = multiplied & 4294967295ULL;
        }
    }
    return (int) (multiplied >> 32);
}

// The first float draw of the positional source at each point: what the ore veinifier opens
// with, and what the G2a parity test pins the whole chain against.
extern "C" __global__ void terracuda_xoroshiro_float_points(
        long long seedLo, long long seedHi,
        const int* __restrict__ points,   // 3 * count positions
        float* __restrict__ results,      // count first draws
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    long long positionalSeed = tcGetSeed(points[3 * index], points[3 * index + 1],
            points[3 * index + 2]);
    TcXoroshiro state = tcXoroshiro(positionalSeed ^ seedLo, seedHi);
    results[index] = tcXoroshiroNextFloat(&state);
}

// ============================================================================================
// The material rules' per-block decision (G2a)
// ============================================================================================

// The decisions the per-block kernel hands back; the host maps them to block states.
#define TC_DECIDE_DEFAULT 0
#define TC_DECIDE_WATER 1
#define TC_DECIDE_LAVA 2
#define TC_DECIDE_COPPER_ORE 3
#define TC_DECIDE_RAW_COPPER 4
#define TC_DECIDE_GRANITE 5
#define TC_DECIDE_DEEPSLATE_IRON_ORE 6
#define TC_DECIDE_RAW_IRON 7
#define TC_DECIDE_TUFF 8
#define TC_DECIDE_AIR 9

// similarity(10*10, 12*12): the threshold below which two cells are too far apart to flow.
#define TC_FLOWING_UPDATE_SIMILARITY (1.0 - (144 - 100) / 25.0)


/** Java's Math.floorDiv for a positive divisor: the aquifer anchors floor their grid Y. */
__device__ __forceinline__ int tcFloorDiv(int value, int divisor) {
    int quotient = value / divisor;
    if (value % divisor != 0 && value < 0) {
        quotient--;
    }
    return quotient;
}

/** BlockPos unpacking, reproduced exactly: 26-bit x and z, 12-bit y, all sign-extended. */
__device__ __forceinline__ int tcUnpackX(long long packed) {
    return (int) (packed >> 38);
}

__device__ __forceinline__ int tcUnpackY(long long packed) {
    return (int) (packed << 52 >> 52);
}

__device__ __forceinline__ int tcUnpackZ(long long packed) {
    return (int) (packed << 26 >> 38);
}

/** One program evaluation at a block position: a plain forward sweep, no markers involved. */
__device__ __forceinline__ double tcEvaluatePlain(const char* blob, const long long* offsets,
                                                   int instructionCount, int root,
                                                   int blockX, int blockY, int blockZ,
                                                   double* scratch) {
    TcProgram program;
    program.blob = blob;
    program.offsets = offsets;
    program.root = root;
    program.instructionCount = instructionCount;
    program.blockX = blockX;
    program.blockY = blockY;
    program.blockZ = blockZ;
    for (int pc = 0; pc < instructionCount; pc++) {
        scratch[pc] = tcCompute(program, scratch, pc);
    }
    return scratch[root];
}

// The material grid's arithmetic helpers -- the clamped pair already exists for the spline
// code, but the plain lerp, inverse lerp and clamp are this half's own.
__device__ __forceinline__ double tcClamp(double value, double min, double max) {
    return value < min ? min : value > max ? max : value;
}

__device__ __forceinline__ double tcLerp(double alpha, double start, double end) {
    return start + alpha * (end - start);
}

__device__ __forceinline__ double tcInverseLerp(double value, double min, double max) {
    return (value - min) / (max - min);
}

// A router entry read with flat-cache semantics: wrapped entries answer from the chunk's quart
// columns (a table whose every entry is the program at y=0, so the entry is evaluated where it
// stands), and only a miss outside those columns falls back to the exact position -- the same
// branch MaterialRules.entry makes. Unwrapped entries are exact everywhere.
__device__ __forceinline__ double tcGridEntry(const char* blob, const long long* offsets,
                                             int instructionCount, int root, int wrapped,
                                             int firstQuartX, int firstQuartZ, int blockX,
                                             int blockY, int blockZ, double* scratch) {
    if (!wrapped) {
        return tcEvaluatePlain(blob, offsets, instructionCount, root, blockX, blockY, blockZ,
                scratch);
    }
    int quartX = (blockX >> 2) - firstQuartX;
    int quartZ = (blockZ >> 2) - firstQuartZ;
    if (quartX >= 0 && quartZ >= 0 && quartX < 5 && quartZ < 5) {
        return tcEvaluatePlain(blob, offsets, instructionCount, root,
                (firstQuartX + quartX) << 2, 0, (firstQuartZ + quartZ) << 2, scratch);
    }
    return tcEvaluatePlain(blob, offsets, instructionCount, root, blockX, blockY, blockZ,
            scratch);
}

// Aquifer's calculatePressure between two cells, verbatim. The barrier noise is evaluated at
// most once per block: the first pressure that needs it computes it through the memo the
// caller owns, and the rest reuse the value -- the same single-evaluation MaterialRules does
// with its NaN scratch slot.
__device__ __forceinline__ double tcCalculatePressure(int posY, int level1, int type1,
                                                       int level2, int type2, int posX,
                                                       int posZ, const char* barrierBlob,
                                                       const long long* barrierOffsets,
                                                       int barrierCount, int barrierRoot,
                                                       int barrierWrapped, int firstQuartX,
                                                       int firstQuartZ, double* scratch,
                                                       bool* barrierComputed,
                                                       double* barrierValue) {
    int at1 = posY < level1 ? type1 : TC_DECIDE_AIR;
    int at2 = posY < level2 ? type2 : TC_DECIDE_AIR;
    bool oppositeFluids = (at1 == TC_DECIDE_LAVA && at2 == TC_DECIDE_WATER)
            || (at1 == TC_DECIDE_WATER && at2 == TC_DECIDE_LAVA);
    if (oppositeFluids) {
        return 2.0;
    }

    int fluidYDiff = level1 < level2 ? level2 - level1 : level1 - level2;
    if (fluidYDiff == 0) {
        return 0.0;
    }

    double averageFluidY = 0.5 * (level1 + level2);
    double howFarAboveAverage = posY + 0.5 - averageFluidY;
    double baseValue = fluidYDiff / 2.0;
    double distanceFromBarrierEdge = baseValue - fabs(howFarAboveAverage);
    double gradient;
    if (howFarAboveAverage > 0.0) {
        double centerPoint = distanceFromBarrierEdge;
        gradient = centerPoint > 0.0 ? centerPoint / 1.5 : centerPoint / 2.5;
    } else {
        double centerPoint = 3.0 + distanceFromBarrierEdge;
        gradient = centerPoint > 0.0 ? centerPoint / 3.0 : centerPoint / 10.0;
    }

    double noiseValue = 0.0;
    if (gradient >= -2.0 && gradient <= 2.0) {
        if (!(*barrierComputed)) {
            *barrierValue = tcGridEntry(barrierBlob, barrierOffsets, barrierCount, barrierRoot,
                    barrierWrapped, firstQuartX, firstQuartZ, posX, posY, posZ, scratch);
            *barrierComputed = true;
        }
        noiseValue = *barrierValue;
    }
    return 2.0 * (noiseValue + gradient);
}

// The aquifer's per-block half, the veinifier's whole body, and the default behind them: a
// verbatim port of MaterialRules.compute. The grid cells' locations and fluid statuses arrive
// from the host (G2a's shape); the barrier and vein-gap noises evaluate on the device through
// their lowered programs, or read the chunk's five-by-five table when the entry is wrapped.
extern "C" __global__ void terracuda_material_decide(
        const double* __restrict__ density,     // count, the beard already folded in
        const double* __restrict__ veinToggle,  // count, or null with veins off
        const double* __restrict__ veinRidged,  // count, or null with veins off
        const int* __restrict__ cellLevels,     // gridTotal
        const int* __restrict__ cellTypes,      // gridTotal, TC_DECIDE_WATER or TC_DECIDE_LAVA
        int minGridX, int minGridY, int minGridZ,
        int gridSizeX, int gridSizeY, int gridSizeZ,
        int skipSamplingAboveY,
        int seaLevel,
        const char* __restrict__ barrierBlob, const long long* __restrict__ barrierOffsets,
        int barrierCount, int barrierRoot, int barrierWrapped,
        const char* __restrict__ veinGapBlob, const long long* __restrict__ veinGapOffsets,
        int veinGapCount, int veinGapRoot, int veinGapWrapped,
        long long oreSeedLo, long long oreSeedHi,
        long long aquiferSeedLo, long long aquiferSeedHi,
        int firstQuartX, int firstQuartZ,
        double* __restrict__ scratch,          // count * programScratch doubles
        int programScratch,
        int* __restrict__ decisions,           // count
        int* __restrict__ fluidFlags,          // count, the aquifer's fluid-tick request
        int chunkMinBlockX, int minY, int chunkMinBlockZ, int height,
        int aquifersEnabled, int oreVeinsEnabled,
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    int xLocal, zLocal, yLocal;
    tcBlockLocal(index, 16, height, &xLocal, &zLocal, &yLocal);
    int posX = chunkMinBlockX + xLocal;
    int posY = minY + yLocal;
    int posZ = chunkMinBlockZ + zLocal;
    double value = density[index];
    double* myScratch = scratch + (long long) index * programScratch;

    int decision = TC_DECIDE_DEFAULT;
    int fluidFlag = 0;

    // The filler's global fluid picker is a pure function of y: below both levels it is lava,
    // otherwise the sea. Each use derives its own pair the way computeFluid(x, y, z) would.
    if (aquifersEnabled && value <= 0.0) {
        bool lavaPick = posY < min(-54, seaLevel);
        int globalLevel = lavaPick ? -54 : seaLevel;
        int globalType = lavaPick ? TC_DECIDE_LAVA : TC_DECIDE_WATER;
        int globalAtY = posY < globalLevel ? globalType : TC_DECIDE_AIR;

        if (posY > skipSamplingAboveY) {
            decision = globalAtY;
        } else if (globalAtY == TC_DECIDE_LAVA) {
            decision = TC_DECIDE_LAVA;
        } else {
            int xAnchor = (posX - 5) >> 4;
            int yAnchor = tcFloorDiv(posY + 1, 12);
            int zAnchor = (posZ - 5) >> 4;
            int distanceSqr1 = 0x7FFFFFFF;
            int distanceSqr2 = 0x7FFFFFFF;
            int distanceSqr3 = 0x7FFFFFFF;
            int distanceSqr4 = 0x7FFFFFFF;
            int closestIndex1 = 0;
            int closestIndex2 = 0;
            int closestIndex3 = 0;
            int closestIndex4 = 0;

            for (int x1 = 0; x1 <= 1; x1++) {
                for (int y1 = -1; y1 <= 1; y1++) {
                    for (int z1 = 0; z1 <= 1; z1++) {
                        int spacedGridX = xAnchor + x1;
                        int spacedGridY = yAnchor + y1;
                        int spacedGridZ = zAnchor + z1;
                        int cellX = spacedGridX - minGridX;
                        int cellY = spacedGridY - minGridY;
                        int cellZ = spacedGridZ - minGridZ;
                        int idx = (cellY * gridSizeZ + cellZ) * gridSizeX + cellX;

                        // The cell's location is a pure function of its grid
                        // position, so it is recomputed where it is read instead of uploaded:
                        // the aquifer's positional source, three bounded draws, the pack.
                        TcXoroshiro cellRandom = tcXoroshiro(
                                tcGetSeed(spacedGridX, spacedGridY, spacedGridZ) ^ aquiferSeedLo,
                                aquiferSeedHi);
                        int cellOffsetX = tcXoroshiroNextInt(&cellRandom, 10);
                        int cellOffsetY = tcXoroshiroNextInt(&cellRandom, 9);
                        int cellOffsetZ = tcXoroshiroNextInt(&cellRandom, 10);
                        int cellBlockX = (spacedGridX << 4) + cellOffsetX;
                        int cellBlockY = spacedGridY * 12 + cellOffsetY;
                        int cellBlockZ = (spacedGridZ << 4) + cellOffsetZ;
                        long long location = (((long long) cellBlockX & 0x3FFFFFFLL) << 38)
                                | (((long long) cellBlockZ & 0x3FFFFFFLL) << 12)
                                | ((long long) cellBlockY & 0xFFFLL);
                        int dx = tcUnpackX(location) - posX;
                        int dy = tcUnpackY(location) - posY;
                        int dz = tcUnpackZ(location) - posZ;
                        int newDistance = dx * dx + dy * dy + dz * dz;
                        // The >= chain is verbatim: on a tie the later cell takes the earlier
                        // slot, and that ordering is terrain.
                        if (distanceSqr1 >= newDistance) {
                            closestIndex4 = closestIndex3;
                            closestIndex3 = closestIndex2;
                            closestIndex2 = closestIndex1;
                            closestIndex1 = idx;
                            distanceSqr4 = distanceSqr3;
                            distanceSqr3 = distanceSqr2;
                            distanceSqr2 = distanceSqr1;
                            distanceSqr1 = newDistance;
                        } else if (distanceSqr2 >= newDistance) {
                            closestIndex4 = closestIndex3;
                            closestIndex3 = closestIndex2;
                            closestIndex2 = idx;
                            distanceSqr4 = distanceSqr3;
                            distanceSqr3 = distanceSqr2;
                            distanceSqr2 = newDistance;
                        } else if (distanceSqr3 >= newDistance) {
                            closestIndex4 = closestIndex3;
                            closestIndex3 = idx;
                            distanceSqr4 = distanceSqr3;
                            distanceSqr3 = newDistance;
                        } else if (distanceSqr4 >= newDistance) {
                            closestIndex4 = idx;
                            distanceSqr4 = newDistance;
                        }
                    }
                }
            }

            int level1 = cellLevels[closestIndex1];
            int type1 = cellTypes[closestIndex1];
            int level2 = cellLevels[closestIndex2];
            int type2 = cellTypes[closestIndex2];
            int level3 = cellLevels[closestIndex3];
            int type3 = cellTypes[closestIndex3];
            double similarity12 = 1.0 - (double) (distanceSqr2 - distanceSqr1) / 25.0;
            int at1 = posY < level1 ? type1 : TC_DECIDE_AIR;

            if (similarity12 <= 0.0) {
                // Beyond the blend band: the nearest cell's fluid, with the flowing-update
                // similarity deciding whether the tick is worth scheduling.
                if (similarity12 >= TC_FLOWING_UPDATE_SIMILARITY
                        && !(level1 == level2 && type1 == type2)) {
                    fluidFlag = 1;
                }
                decision = at1;
            } else if (at1 == TC_DECIDE_WATER) {
                // The water-over-lava check reads the picker one block below.
                bool lavaBelow = (posY - 1) < min(-54, seaLevel) && (posY - 1) < -54;
                if (lavaBelow) {
                    fluidFlag = 1;
                    decision = at1;
                }
            }

            if (decision == TC_DECIDE_DEFAULT && similarity12 > 0.0
                    && (at1 != TC_DECIDE_WATER || !((posY - 1) < min(-54, seaLevel) && (posY - 1) < -54))) {
                // The barrier band: three pressures, the barrier evaluated once, and the
                // decline that hands the block to the veinifier or the default.
                bool barrierComputed = false;
                double barrierValue = 0.0;

                double barrier12 = similarity12 * tcCalculatePressure(posY, level1, type1,
                        level2, type2, posX, posZ, barrierBlob, barrierOffsets, barrierCount,
                        barrierRoot, barrierWrapped, firstQuartX, firstQuartZ, myScratch,
                        &barrierComputed, &barrierValue);
                bool declined = value + barrier12 > 0.0;
                double similarity13 = 1.0 - (double) (distanceSqr3 - distanceSqr1) / 25.0;
                if (!declined && similarity13 > 0.0) {
                    double barrier13 = similarity12 * similarity13 * tcCalculatePressure(posY,
                            level1, type1, level3, type3, posX, posZ, barrierBlob,
                            barrierOffsets, barrierCount, barrierRoot, barrierWrapped,
                            firstQuartX, firstQuartZ, myScratch, &barrierComputed, &barrierValue);
                    declined = value + barrier13 > 0.0;
                }
                double similarity23 = 1.0 - (double) (distanceSqr3 - distanceSqr2) / 25.0;
                if (!declined && similarity23 > 0.0) {
                    double barrier23 = similarity12 * similarity23 * tcCalculatePressure(posY,
                            level2, type2, level3, type3, posX, posZ, barrierBlob,
                            barrierOffsets, barrierCount, barrierRoot, barrierWrapped,
                            firstQuartX, firstQuartZ, myScratch, &barrierComputed, &barrierValue);
                    declined = value + barrier23 > 0.0;
                }

                if (!declined) {
                    // Nothing declined: the nearest cell's fluid, and the flow bookkeeping in
                    // the original's order.
                    bool mayFlow12 = !(level1 == level2 && type1 == type2);
                    bool mayFlow23 = similarity23 >= TC_FLOWING_UPDATE_SIMILARITY
                            && !(level2 == level3 && type2 == type3);
                    bool mayFlow13 = similarity13 >= TC_FLOWING_UPDATE_SIMILARITY
                            && !(level1 == level3 && type1 == type3);
                    if (!mayFlow12 && !mayFlow23 && !mayFlow13) {
                        int level4 = cellLevels[closestIndex4];
                        int type4 = cellTypes[closestIndex4];
                        fluidFlag = similarity13 >= TC_FLOWING_UPDATE_SIMILARITY
                                && (1.0 - (double) (distanceSqr4 - distanceSqr1) / 25.0)
                                           >= TC_FLOWING_UPDATE_SIMILARITY
                                && !(level1 == level4 && type1 == type4) ? 1 : 0;
                    } else {
                        fluidFlag = 1;
                    }
                    decision = at1;
                }
            }
        }
    } else if (!aquifersEnabled && value <= 0.0) {
        // The disabled aquifer: the global picker and nothing else.
        bool lavaPick = posY < min(-54, seaLevel);
        int globalLevel = lavaPick ? -54 : seaLevel;
        decision = posY < globalLevel
                ? (lavaPick ? TC_DECIDE_LAVA : TC_DECIDE_WATER)
                : TC_DECIDE_AIR;
    }

    // The veinifier, on whatever the aquifer declined. The float literals widen exactly the way
    // the Java comparisons do; the draw order is the veinifier's own.
    if (oreVeinsEnabled && decision == TC_DECIDE_DEFAULT) {
        double veininess = veinToggle[index];
        bool copper = veininess > 0.0;
        double ridged = fabs(veininess);
        int typeMaxY = copper ? 50 : -8;
        int typeMinY = copper ? 0 : -60;
        int distanceFromTop = typeMaxY - posY;
        int distanceFromBottom = posY - typeMinY;
        if (distanceFromBottom >= 0 && distanceFromTop >= 0) {
            int distanceFromEdge = min(distanceFromTop, distanceFromBottom);
            double edgeRoundoff = tcClampedMap((double) distanceFromEdge, 0.0, 20.0, -0.2, 0.0);
            if (ridged + edgeRoundoff >= (double) 0.4f) {
                TcXoroshiro random = tcXoroshiro(tcGetSeed(posX, posY, posZ) ^ oreSeedLo, oreSeedHi);
                if (tcXoroshiroNextFloat(&random) <= 0.7f && veinRidged[index] < 0.0) {
                    double richness = tcClampedMap(ridged, (double) 0.4f, (double) 0.6f,
                            (double) 0.1f, (double) 0.3f);
                    bool richEnough = tcXoroshiroNextFloat(&random) < richness;
                    if (richEnough) {
                        double gap = tcGridEntry(veinGapBlob, veinGapOffsets, veinGapCount,
                                veinGapRoot, veinGapWrapped, firstQuartX, firstQuartZ, posX,
                                posY, posZ, myScratch);
                        if (gap > (double) -0.3f) {
                            bool rawOre = tcXoroshiroNextFloat(&random) < 0.02f;
                            decision = copper
                                    ? (rawOre ? TC_DECIDE_RAW_COPPER : TC_DECIDE_COPPER_ORE)
                                    : (rawOre ? TC_DECIDE_RAW_IRON : TC_DECIDE_DEEPSLATE_IRON_ORE);
                        } else {
                            decision = copper ? TC_DECIDE_GRANITE : TC_DECIDE_TUFF;
                        }
                    } else {
                        decision = copper ? TC_DECIDE_GRANITE : TC_DECIDE_TUFF;
                    }
                }
            }
        }
    }

    decisions[index] = decision;
    fluidFlags[index] = fluidFlag;
}

// The aquifer grid's location chain at each cell: the three bounded draws, in the argument
// order MaterialRules draws them, packed the way it packs them. G2b's location supply is this
// chain computed where it is read, so the parity test drives it against the CPU reference.
extern "C" __global__ void terracuda_xoroshiro_grid_locations(
        long long seedLo, long long seedHi,
        const int* __restrict__ points,   // 3 * count grid cells
        long long* __restrict__ results,  // count packed locations
        int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    long long positionalSeed = tcGetSeed(points[3 * index], points[3 * index + 1],
            points[3 * index + 2]);
    TcXoroshiro state = tcXoroshiro(positionalSeed ^ seedLo, seedHi);
    int x = tcXoroshiroNextInt(&state, 10);
    int y = tcXoroshiroNextInt(&state, 9);
    int z = tcXoroshiroNextInt(&state, 10);
    // fromGridX/fromGridY/fromGridZ: the packed location holds the cell's block position,
    // not the raw draws -- the y scaling is what keeps the ranking three-dimensional.
    int blockX = (points[3 * index] << 4) + x;
    int blockY = points[3 * index + 1] * 12 + y;
    int blockZ = (points[3 * index + 2] << 4) + z;
    results[index] = (((long long) blockX & 0x3FFFFFFLL) << 38)
            | (((long long) blockZ & 0x3FFFFFFLL) << 12)
            | ((long long) blockY & 0xFFFLL);
}

// ============================================================================================
// The material rules' grid half (G2b): one thread per aquifer cell, fluid status on device
// ============================================================================================

// The material grid's arithmetic helpers -- the clamped pair already exists for the spline
// code, but the plain lerp, inverse lerp and clamp are this half's own.
// Vanilla's thirteen sampling offsets, in chunk-scale pairs: the aquifer's fluid status reads
// the preliminary surface around the cell, and this is the neighbourhood it reads.
__device__ __forceinline__ int tcSurfaceOffsetX(int index) {
    const int xs[13] = {0, -2, -1, 0, 1, -3, -2, -1, 1, -2, -1, 0, 1};
    return xs[index];
}

__device__ __forceinline__ int tcSurfaceOffsetZ(int index) {
    const int zs[13] = {0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
    return zs[index];
}

// One grid cell's fluid status, a verbatim port of MaterialRules.computeFluid, plus -- on the
// first thread only -- the sampling cutoff the constructor derives from the highest preliminary
// surface under the grid. The K0 program answers a column's level in one evaluation: the
// find-top-surface walk was lowered into it.
extern "C" __global__ void terracuda_material_grid(
        int minGridX, int minGridY, int minGridZ,
        int gridSizeX, int gridSizeY, int gridSizeZ,
        int seaLevel, long long aquiferSeedLo, long long aquiferSeedHi,
        int firstQuartX, int firstQuartZ,
        const char* __restrict__ k0Blob, const long long* __restrict__ k0Offsets,
        int k0Count, int k0Root,
        const char* __restrict__ floodednessBlob, const long long* __restrict__ floodednessOffsets,
        int floodednessCount, int floodednessRoot, int floodednessWrapped,
        const char* __restrict__ spreadBlob, const long long* __restrict__ spreadOffsets,
        int spreadCount, int spreadRoot, int spreadWrapped,
        const char* __restrict__ lavaBlob, const long long* __restrict__ lavaOffsets,
        int lavaCount, int lavaRoot, int lavaWrapped,
        const char* __restrict__ erosionBlob, const long long* __restrict__ erosionOffsets,
        int erosionCount, int erosionRoot, int erosionWrapped,
        const char* __restrict__ depthBlob, const long long* __restrict__ depthOffsets,
        int depthCount, int depthRoot, int depthWrapped,
        double* __restrict__ scratch,          // gridTotal * gridScratch doubles
        int gridScratch,
        int* __restrict__ cellLevels,          // gridTotal
        int* __restrict__ cellTypes,           // gridTotal, TC_DECIDE_WATER or TC_DECIDE_LAVA
        int* __restrict__ skipOut,              // one int, written by thread 0
        int gridTotal) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gridTotal) {
        return;
    }
    int cellX = index % gridSizeX;
    int remaining = index / gridSizeX;
    int cellZ = remaining % gridSizeZ;
    int cellY = remaining / gridSizeZ;
    int gridX = minGridX + cellX;
    int gridY = minGridY + cellY;
    int gridZ = minGridZ + cellZ;
    double* myScratch = scratch + (long long) index * gridScratch;

    // The cell's location: the same chain the decide kernel recomputes per block.
    TcXoroshiro cellRandom = tcXoroshiro(
            tcGetSeed(gridX, gridY, gridZ) ^ aquiferSeedLo, aquiferSeedHi);
    int cellBlockX = (gridX << 4) + tcXoroshiroNextInt(&cellRandom, 10);
    int cellBlockY = gridY * 12 + tcXoroshiroNextInt(&cellRandom, 9);
    int cellBlockZ = (gridZ << 4) + tcXoroshiroNextInt(&cellRandom, 10);

    // The global picker's status at the cell: below both levels it is lava, otherwise the sea.
    bool lavaPick = cellBlockY < min(-54, seaLevel);
    int globalLevel = lavaPick ? -54 : seaLevel;
    int globalType = lavaPick ? TC_DECIDE_LAVA : TC_DECIDE_WATER;

    int topOfAquiferCell = cellBlockY + 12;
    int bottomOfAquiferCell = cellBlockY - 12;
    int lowestPreliminarySurface = 0x7FFFFFFF;
    bool surfaceAtCenterIsUnderGlobalFluidLevel = false;
    int fluidSurfaceLevel = -32512;  // WAY_BELOW_MIN_Y when nothing floods
    int fluidType = globalType;

    bool earlyGlobal = false;
    for (int offset = 0; offset < 13; offset++) {
        int sampleX = cellBlockX + (tcSurfaceOffsetX(offset) << 4);
        int sampleZ = cellBlockZ + (tcSurfaceOffsetZ(offset) << 4);
        // The column quantises to quart positions before it is asked.
        int quartX = (sampleX >> 2) << 2;
        int quartZ = (sampleZ >> 2) << 2;
        double levelValue = tcEvaluatePlain(k0Blob, k0Offsets, k0Count, k0Root,
                quartX, 0, quartZ, myScratch);
        int preliminarySurfaceLevel = (int) floor(levelValue);
        int adjustedSurfaceLevel = preliminarySurfaceLevel + 8;
        bool start = offset == 0;
        if (start && bottomOfAquiferCell > adjustedSurfaceLevel) {
            // The cell sits below dry ground: the picker's status at the cell, nothing else.
            fluidSurfaceLevel = globalLevel;
            fluidType = globalType;
            earlyGlobal = true;
            break;
        }
        bool topPokesAboveSurface = topOfAquiferCell > adjustedSurfaceLevel;
        if (topPokesAboveSurface || start) {
            // The picker's status at the adjusted level, a pure function of that y; at(y) is
            // fluid only below the picker's own level.
            bool surfaceLavaPick = adjustedSurfaceLevel < min(-54, seaLevel);
            bool surfaceIsFluid = surfaceLavaPick ? adjustedSurfaceLevel < -54
                                                   : adjustedSurfaceLevel < seaLevel;
            if (surfaceIsFluid) {
                if (start) {
                    surfaceAtCenterIsUnderGlobalFluidLevel = true;
                }
                if (topPokesAboveSurface) {
                    // The aquifer pokes out under water: the surface's status, not the cell's.
                    fluidSurfaceLevel = surfaceLavaPick ? -54 : seaLevel;
                    fluidType = surfaceLavaPick ? TC_DECIDE_LAVA : TC_DECIDE_WATER;
                    earlyGlobal = true;
                    break;
                }
            }
        }
        lowestPreliminarySurface = min(lowestPreliminarySurface, preliminarySurfaceLevel);
    }

    if (!earlyGlobal) {
        // computeSurfaceLevel: the deep-dark shortcut, then the floodedness thresholds.
        double partiallyFloodedness;
        double fullyFloodedness;
        double erosionValue = tcGridEntry(erosionBlob, erosionOffsets, erosionCount, erosionRoot,
                erosionWrapped, firstQuartX, firstQuartZ, cellBlockX, cellBlockY, cellBlockZ,
                myScratch);
        double depthValue = tcGridEntry(depthBlob, depthOffsets, depthCount, depthRoot,
                depthWrapped, firstQuartX, firstQuartZ, cellBlockX, cellBlockY, cellBlockZ,
                myScratch);
        if (erosionValue < (double) -0.225f && depthValue > (double) 0.9f) {
            partiallyFloodedness = -1.0;
            fullyFloodedness = -1.0;
        } else {
            int distanceBelowSurface = lowestPreliminarySurface + 8 - cellBlockY;
            double floodednessFactor = surfaceAtCenterIsUnderGlobalFluidLevel
                    ? tcClampedMap((double) distanceBelowSurface, 0.0, 64.0, 1.0, 0.0)
                    : 0.0;
            double floodednessNoise = tcClamp(tcGridEntry(floodednessBlob, floodednessOffsets,
                    floodednessCount, floodednessRoot, floodednessWrapped, firstQuartX,
                    firstQuartZ, cellBlockX, cellBlockY, cellBlockZ, myScratch), -1.0, 1.0);
            double fullyFloodedThreshold = tcLerp(
                    tcInverseLerp(floodednessFactor, 1.0, 0.0), -0.3, 0.8);
            double partiallyFloodedThreshold = tcLerp(
                    tcInverseLerp(floodednessFactor, 1.0, 0.0), -0.8, 0.4);
            partiallyFloodedness = floodednessNoise - partiallyFloodedThreshold;
            fullyFloodedness = floodednessNoise - fullyFloodedThreshold;
        }

        if (fullyFloodedness > 0.0) {
            fluidSurfaceLevel = globalLevel;
        } else if (partiallyFloodedness > 0.0) {
            // computeRandomizedFluidSurfaceLevel: the spread noise at its own coarse cell.
            int fluidLevelCellX = tcFloorDiv(cellBlockX, 16);
            int fluidLevelCellY = tcFloorDiv(cellBlockY, 40);
            int fluidLevelCellZ = tcFloorDiv(cellBlockZ, 16);
            double spread = tcGridEntry(spreadBlob, spreadOffsets, spreadCount, spreadRoot,
                    spreadWrapped, firstQuartX, firstQuartZ, fluidLevelCellX, fluidLevelCellY,
                    fluidLevelCellZ, myScratch) * 10.0;
            int quantized = (int) floor(spread / 3.0) * 3;
            int targetFluidSurfaceLevel = fluidLevelCellY * 40 + 20 + quantized;
            fluidSurfaceLevel = min(lowestPreliminarySurface, targetFluidSurfaceLevel);
        }

        // computeFluidType: the lava noise decides the deep, cold cells.
        if (fluidSurfaceLevel <= -10 && fluidSurfaceLevel != -32512
                && fluidType != TC_DECIDE_LAVA) {
            double lavaNoise = tcGridEntry(lavaBlob, lavaOffsets, lavaCount, lavaRoot,
                    lavaWrapped, firstQuartX, firstQuartZ, tcFloorDiv(cellBlockX, 64),
                    tcFloorDiv(cellBlockY, 40), tcFloorDiv(cellBlockZ, 64), myScratch);
            if (fabs(lavaNoise) > 0.3) {
                fluidType = TC_DECIDE_LAVA;
            }
        }
    }

    cellLevels[index] = fluidSurfaceLevel;
    cellTypes[index] = fluidType;

    // Thread 0 derives the sampling cutoff the constructor used to: the highest preliminary
    // surface under the grid, in quart steps, then the grid arithmetic on top of it.
    if (index == 0) {
        int maxGridX = minGridX + gridSizeX - 1;
        int maxGridZ = minGridZ + gridSizeZ - 1;
        int minBlockX = (minGridX << 4);
        int minBlockZ = (minGridZ << 4);
        int maxBlockX = (maxGridX << 4) + 9;
        int maxBlockZ = (maxGridZ << 4) + 9;
        int highest = 0x80000000;
        for (int blockZ = minBlockZ; blockZ <= maxBlockZ; blockZ += 4) {
            for (int blockX = minBlockX; blockX <= maxBlockX; blockX += 4) {
                int quartX = (blockX >> 2) << 2;
                int quartZ = (blockZ >> 2) << 2;
                double levelValue = tcEvaluatePlain(k0Blob, k0Offsets, k0Count, k0Root,
                        quartX, 0, quartZ, myScratch);
                highest = max(highest, (int) floor(levelValue));
            }
        }
        int maxAdjusted = highest + 8;
        int skipGridY = tcFloorDiv(maxAdjusted + 12, 12) + 1;
        skipOut[0] = skipGridY * 12 + 11 - 1;
    }
}
