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

// ImprovedNoise.noise(x, y, z) with yScale == 0, i.e. the form ordinary terrain noise uses.
// The yScale/yFudge branch only matters for BlendedNoise and is deliberately not ported yet.
__device__ __forceinline__ double improvedNoise(const unsigned char* p,
                                                double xo, double yo, double zo,
                                                double inX, double inY, double inZ) {
    double x = inX + xo;
    double y = inY + yo;
    double z = inZ + zo;
    int xf = (int) floor(x);
    int yf = (int) floor(y);
    int zf = (int) floor(z);
    double xr = x - xf;
    double yr = y - yf;
    double zr = z - zf;

    int x0 = perm(p, xf);
    int x1 = perm(p, xf + 1);
    int xy00 = perm(p, x0 + yf);
    int xy01 = perm(p, x0 + yf + 1);
    int xy10 = perm(p, x1 + yf);
    int xy11 = perm(p, x1 + yf + 1);

    double d000 = gradDot(perm(p, xy00 + zf), xr, yr, zr);
    double d100 = gradDot(perm(p, xy10 + zf), xr - 1.0, yr, zr);
    double d010 = gradDot(perm(p, xy01 + zf), xr, yr - 1.0, zr);
    double d110 = gradDot(perm(p, xy11 + zf), xr - 1.0, yr - 1.0, zr);
    double d001 = gradDot(perm(p, xy00 + zf + 1), xr, yr, zr - 1.0);
    double d101 = gradDot(perm(p, xy10 + zf + 1), xr - 1.0, yr, zr - 1.0);
    double d011 = gradDot(perm(p, xy01 + zf + 1), xr, yr - 1.0, zr - 1.0);
    double d111 = gradDot(perm(p, xy11 + zf + 1), xr - 1.0, yr - 1.0, zr - 1.0);

    double xAlpha = smoothstep(xr);
    double yAlpha = smoothstep(yr);
    double zAlpha = smoothstep(zr);
    return lerp3(xAlpha, yAlpha, zAlpha, d000, d100, d010, d110, d001, d101, d011, d111);
}

}  // namespace

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
