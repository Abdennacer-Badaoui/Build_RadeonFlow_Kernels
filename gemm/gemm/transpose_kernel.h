// Implementation of transpose kernel.
#pragma once

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../src/utils/arithmetic.h"
#include "../include/clangd_workaround.h"

DEVICE_CODE_BELOW

namespace transpose_kernel {



template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void transpose_kernel(Elem *odata, const Elem *idata) {
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;

    // avoid read bank conflict
    // VEC_SIZE * (TILE_DIM + d) * sizeof(Elem) = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE) * sizeof(Elem) + 128k
    // each warp read row = TILE_DIM (in VEC_SIZE reads), col = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    // warp 0                     warp 1
    // t0    t16    t32    t48    ...
    // ...
    // t1
    // ...
    // t15
    // don't know why padding to d as described above is not working, maybe gpu could merge contigious ds_read_u8 and
    // cause padding to be TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    constexpr auto PADDING = TBLOCK_Y / (BLOCK_SIZE / warpSize);
    __shared__ Elem tile[TILE_DIM][TILE_DIM + PADDING];

    int x = blockIdx.x * TILE_DIM + threadIdx.x * VEC_SIZE;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

// Load tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            tile[threadIdx.y + i][threadIdx.x * VEC_SIZE + v] = idata[(y + i) * N + x + v];
        }
    }

    __syncthreads();

    // Transpose indices
    x = blockIdx.y * TILE_DIM + threadIdx.x * VEC_SIZE;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

// Write tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            odata[(y + i) * M + x + v] = tile[threadIdx.x * VEC_SIZE + v][threadIdx.y + i];
        }
    }
}

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_transpose(Elem *out, const Elem *in, hipStream_t stream = 0) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>),
        dim3(N / TILE_DIM, M / TILE_DIM), dim3(TBLOCK_X, TBLOCK_Y), 0, stream,
        out, in);
}

#define DISPATCH_TRANSPOSE(DIM_0, DIM_1, TILE_DIM, BLOCK_SIZE, VEC_SIZE) else if constexpr(IN_DIM_0 == DIM_0 && IN_DIM_1 == DIM_1) launch_transpose<__FP8_TYPE, IN_DIM_0, IN_DIM_1, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(out, in, stream)

template <int DIM0, int DIM1>
struct unsupported_config {
    static_assert(DIM0 == -1, "Unsupported transpose configuration - check template parameters");
};

// Selecte best parameters for tranpose kernel.
template <int IN_DIM_0, int IN_DIM_1>
void transpose_fp8(__FP8_TYPE *out, const __FP8_TYPE *in, hipStream_t stream = 0) {
    if constexpr (false /* dummy*/ ) static_assert(true);
    DISPATCH_TRANSPOSE(   256,   1024,     64,    256, 4); // Optimized: 2.71 µs (193.46 GB/s)
    DISPATCH_TRANSPOSE(   256,   6144,     64,    256, 4); // Optimized: 2.72 µs (1157.37 GB/s)
    DISPATCH_TRANSPOSE(   256,   7168,     64,    256, 8); // Optimized: 2.99 µs (1225.38 GB/s)
    DISPATCH_TRANSPOSE(   512,   1024,     64,    512, 4); // Optimized: 2.55 µs (411.21 GB/s)
    DISPATCH_TRANSPOSE(   512,   4096,     64,    256, 4); // Optimized: 3.01 µs (1394.85 GB/s)
    DISPATCH_TRANSPOSE(   512,   6144,     64,    512, 4); // Optimized: 3.58 µs (1755.43 GB/s)
    DISPATCH_TRANSPOSE(  1536,   1024,     64,   1024, 4); // Optimized: 2.78 µs (1130.74 GB/s)
    DISPATCH_TRANSPOSE(  1536,   3072,     64,    512, 4); // Optimized: 3.57 µs (2641.99 GB/s)
    DISPATCH_TRANSPOSE(  1536,   6144,    128,   1024, 8); // Optimized: 7.09 µs (2661.36 GB/s)
    DISPATCH_TRANSPOSE(  2048,   1024,     64,   1024, 4); // Optimized: 2.84 µs (1477.91 GB/s)
    DISPATCH_TRANSPOSE(  2048,   6144,    128,    512, 8); // Optimized: 8.94 µs (2816.23 GB/s)
    DISPATCH_TRANSPOSE(  2048,   7168,    128,    512, 8); // Optimized: 9.56 µs (3070.50 GB/s)
    DISPATCH_TRANSPOSE(  2304,   1024,     64,   1024, 4); // Optimized: 3.08 µs (1532.51 GB/s)
    DISPATCH_TRANSPOSE(  2304,   6144,    128,    512, 8); // Optimized: 9.30 µs (3043.93 GB/s)
    DISPATCH_TRANSPOSE(  2304,   7168,    128,    512, 8); // Optimized: 10.39 µs (3179.95 GB/s)
    DISPATCH_TRANSPOSE(  7168,    512,     64,    512, 4); // Optimized: 3.25 µs (2257.78 GB/s)
    DISPATCH_TRANSPOSE(  7168,    576,     64,    512, 4); // Optimized: 3.44 µs (2403.24 GB/s)
    DISPATCH_TRANSPOSE(  7168,   1024,     64,    256, 4); // Optimized: 5.07 µs (2892.62 GB/s)
    DISPATCH_TRANSPOSE(  7168,   1536,    128,   1024, 8); // Optimized: 7.72 µs (2851.97 GB/s)
    DISPATCH_TRANSPOSE(  7168,   4608,    128,    512, 8); // Optimized: 16.87 µs (3915.84 GB/s)
    DISPATCH_TRANSPOSE(  7168,   6144,    128,    256, 8); // Optimized: 21.59 µs (4079.12 GB/s)
    else static_assert(false);
}

} // namespace transpose_kernel




#ifndef PARAMETERIZE_LIBRARY
int main() {}
#endif