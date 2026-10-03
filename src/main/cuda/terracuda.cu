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

#define TC_SPLINE_CONSTANT 0
#define TC_SPLINE_MULTIPOINT 1

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

__device__ double tcCompute(const TcProgram& p, const double* values, int pc) {
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
        default:
            // TC_FIND_TOP_SURFACE and anything newer. NaN rather than a guess: the caller compares
            // against the CPU reference and a mismatch is a loud failure, not silent wrong terrain.
            return __longlong_as_double(0x7FF8000000000000LL);
    }
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
extern "C" __global__ void terracuda_density_evaluate(
        const char* __restrict__ blob,
        const long long* __restrict__ offsets,
        int instructionCount,
        int root,
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
    program.root = root;
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
    results[index] = values[root];
}
