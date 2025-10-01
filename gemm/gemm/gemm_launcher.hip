// Wrapped of gemm kernel launcher.
#include <unistd.h>
#include <chrono>
#define PARAMETERIZE_LIBRARY
#include "gemm_kernel.h"
#include "gemm_kernel_legacy.h"
#include "transpose_kernel.h"
#undef PARAMETERIZE_LIBRARY
#include "../include/gpu_types.h"
#include "../include/timer.h"
#include "../tests/checker/metrics.h"
#include <iostream>

#include <stdio.h>

HOST_CODE_BELOW

std::vector<std::shared_ptr<KernelTimer>> timers;

using namespace std;

float *c_splitk = nullptr;
__FP8_TYPE *a_trans = nullptr;
__FP8_TYPE *b_trans = nullptr;
constexpr int MAX_MATRIX_M = 6144;
constexpr int MAX_MATRIX_N = 7168;
constexpr int MAX_MATRIX_K = 7168;
constexpr int MAX_SPLITK_FACTOR = 8;

void init_workspace() {
    LIB_CALL(HOST_TYPE(Malloc)(&c_splitk, MAX_MATRIX_M * MAX_MATRIX_N * sizeof(float) * MAX_SPLITK_FACTOR));
    LIB_CALL(HOST_TYPE(Malloc)(&a_trans, MAX_MATRIX_M * MAX_MATRIX_K * sizeof(__FP8_TYPE)));
    LIB_CALL(HOST_TYPE(Malloc)(&b_trans, MAX_MATRIX_N * MAX_MATRIX_K * sizeof(__FP8_TYPE)));
    // LIB_CALL(HOST_TYPE(StreamCreateWithFlags)(&job_stream0, HOST_TYPE(StreamNonBlocking)));
    // job_stream0 = 0;
}


// Launch pipeline gemm kernels (most performant).
// 1. Transpose input A & B.
// 2. GEMM compute.
// 3. Reduce (if spilt-k is enable)
template <int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE,
          int SPLITK_FACTOR, int LOAD_BATCH_SIZE = 16>
void launch_gemm(const __FP8_TYPE *a, const __FP8_TYPE *b, __BF16_TYPE *c, const float *as, const float *bs, HOST_TYPE(Stream_t) job_stream0) {
    static_assert(M <= MAX_MATRIX_M, "M exceeds maximum supported size");
    static_assert(N <= MAX_MATRIX_N, "N exceeds maximum supported size");
    static_assert(K <= MAX_MATRIX_K, "K exceeds maximum supported size");
    static_assert(SPLITK_FACTOR <= MAX_SPLITK_FACTOR, "SPLITK_FACTOR exceeds maximum supported size");
    if (__builtin_expect(c_splitk == nullptr, 0)) {
        init_workspace();
        LIB_CALL(hipDeviceSynchronize());
    }
    
    transpose_kernel::transpose_fp8<K, N>(b_trans, b, job_stream0);
    transpose_kernel::transpose_fp8<K, M>(a_trans, a, job_stream0);
    // transpose_kernel::launch_transpose<__FP8_TYPE, K, N, 64, 512, 4>(b_trans, b, job_stream0);
    // transpose_kernel::launch_transpose<__FP8_TYPE, K, M, 64, 512, 4>(a_trans, a, job_stream0);
    // Busy wait for 150 microseconds
    // auto start = std::chrono::high_resolution_clock::now();
    // while (std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - start).count() < 150) {
    //     // Busy wait
    // }
    // be careful that blocksize < 1024, or there's a silent fault
    // gemm_kernel::check_trans<<<dim3(K / 32, M / 32), dim3(32, 32)>>>(a, a_trans, K, M);

    static_assert(K % SPLITK_FACTOR == 0, "K not divisible by SPLITK_FACTOR");
    dim3 grid(ceil_div(N, BN) << 1, ceil_div(M, BM) >> 1, SPLITK_FACTOR);
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be at least 32");
    dim3 block(BLOCK_SIZE);
    if constexpr (SPLITK_FACTOR == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::gemm_kernel<__FP8_TYPE, float, __BF16_TYPE, M, N, K, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N, K, K, LOAD_BATCH_SIZE>),
            grid, block, 0, job_stream0, 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(a_trans), 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(b_trans),
            reinterpret_cast<__BF16_TYPE(*)[N]>(c), reinterpret_cast<const float(*)[M]>(as),
            reinterpret_cast<const float(*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::gemm_kernel<__FP8_TYPE, float, float, M, N, K / SPLITK_FACTOR, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N, K, K, LOAD_BATCH_SIZE>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE(*)[K]>(a_trans), 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(b_trans),
            reinterpret_cast<float(*)[N]>(c_splitk), reinterpret_cast<const float(*)[M]>(as),
            reinterpret_cast<const float(*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs));
        constexpr uint32_t REDUCE_BLOCK = 256;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::reduce_kernel<M, N, SPLITK_FACTOR, REDUCE_BLOCK>),
            ceil_div(M * N / 4, REDUCE_BLOCK), REDUCE_BLOCK, 0, job_stream0,
            reinterpret_cast<const float(*)[M][N]>(c_splitk), 
            reinterpret_cast<__BF16_TYPE(*)[N]>(c)
        );    }
    auto err = HOST_TYPE(GetLastError)();
    if (err != HOST_TYPE(Success)) {
        std::cerr << "Kernel execution failed.\n" << HOST_TYPE(GetErrorString)(err) << std::endl;
        abort();
    }
}


// Launch legacy gemm kernel. (most compellable)
template <int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE, int SPLITK_FACTOR>
void launch_gemm_legacy(const __FP8_TYPE *a, const __FP8_TYPE *b, __BF16_TYPE *c, const float *as, const float *bs, HOST_TYPE(Stream_t) job_stream0) {
    static_assert(K % SPLITK_FACTOR == 0, "K not divisible by SPLITK_FACTOR");
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM), SPLITK_FACTOR);
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be at least 32");
    dim3 block(BLOCK_SIZE);
    if (__builtin_expect(c_splitk == nullptr, 0)) {
        init_workspace();
        LIB_CALL(hipDeviceSynchronize());
    }

    if constexpr (SPLITK_FACTOR == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::gemm_kernel<__FP8_TYPE, float, __BF16_TYPE, M, N, K, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE (*)[M]>(a),
            reinterpret_cast<const __FP8_TYPE (*)[N]>(b),
            reinterpret_cast<__BF16_TYPE (*)[N]>(c),
            reinterpret_cast<const float (*)[M]>(as),
            reinterpret_cast<const float (*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::gemm_kernel<__FP8_TYPE, float, float, M, N, K / SPLITK_FACTOR, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE (*)[M]>(a),
            reinterpret_cast<const __FP8_TYPE (*)[N]>(b),
            reinterpret_cast<float (*)[N]>(c_splitk),
            reinterpret_cast<const float (*)[M]>(as),
            reinterpret_cast<const float (*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
        constexpr uint32_t REDUCE_BLOCK = 256;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::reduce<0>),
            ceil_div(M * N, REDUCE_BLOCK), REDUCE_BLOCK, 0, job_stream0,
            M, N, SPLITK_FACTOR, c_splitk, (__BF16_TYPE *)c
        );
    }
    auto err = HOST_TYPE(GetLastError)();
    if (err != HOST_TYPE(Success)) {
        std::cerr << "Kernel execution failed.\n" << HOST_TYPE(GetErrorString)(err) << std::endl;
        abort();
    }
}

constexpr inline uint32_t pack_shape(uint32_t m, uint32_t n, uint32_t k) {
    // Pack m, n, k into a 32-bit integer
    // Use 8 bits for each dimension (supports 32-aligned values from 32 to 8192)
    // Divide each value by 32 to fit into 8 bits
    return ((m / 32) << 16) | ((n / 32) << 8) | (k / 32);
}
// int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE, int
// SPLITK_FACTOR, int LOAD_BATCH_SIZE
#define DISPATCH_GEMM(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK_FACTOR, LOAD_BATCH_SIZE)                                  \
    case pack_shape_checked<M, N, K>(): {                                                                              \
        launch_gemm<M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, 128, SPLITK_FACTOR, LOAD_BATCH_SIZE>(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, job_stream0);          \
        break;                                                                                                         \
    }

#define DISPATCH_GEMM_LEGACY(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK_FACTOR)                                                 \
    case pack_shape_checked<M, N, K>(): {                                                                                \
        launch_gemm_legacy<M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, 128, SPLITK_FACTOR>(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, job_stream0); \
        break;                                                                                                              \
    }

template <int M, int N, int K> constexpr inline uint32_t pack_shape_checked() {
    static_assert(M % 32 == 0, "M must be a multiple of 32");
    static_assert(N % 32 == 0, "N must be a multiple of 32");
    static_assert(K % 32 == 0, "K must be a multiple of 32");
    static_assert(M >= 32 && M <= 8192, "M must be between 32 and 8192");
    static_assert(N >= 32 && N <= 8192, "N must be between 32 and 8192");
    static_assert(K >= 32 && K <= 8192, "K must be between 32 and 8192");
    return pack_shape(M, N, K);
}



extern "C" {
// Basically, it dispatch GEMM to fatest implementations according to inputs' shape. 
void run(void *a, void *b, void *as, void *bs, void *c, int m, int n, int k, PerfMetrics *metrics, hipStream_t job_stream0) {
    // Cast pointers once
    const __FP8_TYPE *a_ptr = static_cast<const __FP8_TYPE *>(a);
    const __FP8_TYPE *b_ptr = static_cast<const __FP8_TYPE *>(b);
    __BF16_TYPE *c_ptr = static_cast<__BF16_TYPE *>(c);
    const float *as_ptr = static_cast<const float *>(as);
    const float *bs_ptr = static_cast<const float *>(bs);
    KernelTimerScoped timer(timers, 2LL * m * n * k,
        metrics ? &metrics->entries[0].time : nullptr,
        metrics ? &metrics->entries[0].gflops : nullptr, job_stream0);

    switch (pack_shape(m, n, k)) {
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
        // Test:      M,      N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SIZE,   SPLITK_FACTOR, LOAD_BATCH_SIZE
        DISPATCH_GEMM(64, 64, 128, 64, 64, 32, 1, 4, 128, 1, 16);
        DISPATCH_GEMM(64, 1536, 7168, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(64, 3072, 1536, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(64, 576, 7168, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(96, 7168, 256, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(96, 7168, 2048, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(96, 4608, 7168, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(128, 7168, 2304, 128, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(128, 512, 7168, 128, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(512, 4096, 512, 256, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(512, 1536, 7168, 256, 128, 64, 4, 2, 256, 1, 16);
        // Benchmark: M,      N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SIZE,   SPLITK_FACTOR, LOAD_BATCH_SIZE
        DISPATCH_GEMM(1024, 1536, 7168, 128, 128, 64, 1, 4, 128, 4, 16); // Optimized: 0.49 ms (45.65 TFlops)
        DISPATCH_GEMM(1024, 3072, 1536, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.19 ms (51.32 TFlops)
        DISPATCH_GEMM(1024, 576, 7168, 128, 64, 32, 4, 1, 128, 4, 16);   // Optimized: 0.30 ms (28.16 TFlops)
        DISPATCH_GEMM(1024, 7168, 256, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.08 ms (46.49 TFlops)
        DISPATCH_GEMM(1024, 7168, 2048, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.49 ms (61.92 TFlops)
        DISPATCH_GEMM(1024, 4608, 7168, 128, 128, 32, 2, 2, 128, 1, 16); // Optimized: 0.99 ms (68.16 TFlops)
        DISPATCH_GEMM(1024, 7168, 2304, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.51 ms (66.04 TFlops)
        DISPATCH_GEMM(1024, 512, 7168, 64, 128, 32, 2, 2, 128, 4, 16);   // Optimized: 0.26 ms (28.97 TFlops)
        DISPATCH_GEMM(1024, 4096, 512, 128, 256, 32, 2, 4, 256, 1, 16);  // Optimized: 0.08 ms (54.27 TFlops)
        DISPATCH_GEMM(6144, 1536, 7168, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 1.76 ms (76.76 TFlops)
        DISPATCH_GEMM(6144, 3072, 1536, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.88 ms (66.00 TFlops)
        DISPATCH_GEMM(6144, 576, 7168, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.84 ms (60.68 TFlops)
        DISPATCH_GEMM(6144, 7168, 256, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.49 ms (45.76 TFlops)
        DISPATCH_GEMM(6144, 7168, 2048, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 2.17 ms (83.11 TFlops)
        DISPATCH_GEMM(6144, 4608, 7168, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 4.56 ms (88.99 TFlops)
        DISPATCH_GEMM(6144, 7168, 2304, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 2.41 ms (84.32 TFlops)
        DISPATCH_GEMM(6144, 512, 7168, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.67 ms (67.45 TFlops)
        DISPATCH_GEMM(6144, 4096, 512, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.51 ms (50.79 TFlops)
#else                                                                // CDNA3, WAVE_SIZE = 64
      // Benchmark:   M,    N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SZ, SPLITK_F, LOAD_BS
        DISPATCH_GEMM(1024, 1536,   7168,   256,   128,   128,   4,      2,      512,      4,        16); // #0
        DISPATCH_GEMM(1024, 3072,   1536,   256,   128,   128,   4,      2,      512,      2,        16); // #1
        DISPATCH_GEMM(1024, 576,    7168,   256,   128,   128,   4,      2,      512,      8,        16); // #2
        DISPATCH_GEMM(1024, 7168,   256,    256,   128,   128,   4,      2,      512,      1,        16); // #3
        DISPATCH_GEMM(1024, 7168,   2048,   256,   128,   128,   4,      2,      512,      1,        16); // #4
        DISPATCH_GEMM(1024, 4608,   7168,   256,   128,   128,   4,      2,      512,      2,        16); // #5
        DISPATCH_GEMM(1024, 7168,   2304,   256,   128,   128,   4,      2,      512,      1,        16); // #6
        DISPATCH_GEMM(1024, 512,    7168,   256,   128,   128,   4,      2,      512,      8,        16); // #7
        DISPATCH_GEMM(1024, 4096,   512,    256,   128,   128,   4,      2,      512,      1,        16); // #8
        DISPATCH_GEMM(6144, 1536,   7168,   256,   128,   128,   4,      2,      512,      1,        16); // #9
        DISPATCH_GEMM(6144, 3072,   1536,   256,   128,   128,   4,      2,      512,      1,        16); // #10
        DISPATCH_GEMM(6144, 576,    7168,   256,   128,   128,   4,      2,      512,      2,        16); // #11
        DISPATCH_GEMM(6144, 7168,   256,    256,   128,   128,   4,      2,      512,      1,        16); // #12
        DISPATCH_GEMM(6144, 7168,   2048,   256,   128,   128,   4,      2,      512,      1,        16); // #13
        DISPATCH_GEMM(6144, 4608,   7168,   256,   128,   128,   4,      2,      512,      1,        16); // #14
        DISPATCH_GEMM(6144, 7168,   2304,   256,   128,   128,   4,      2,      512,      1,        16); // #15
        DISPATCH_GEMM(6144, 512,    7168,   256,   128,   128,   4,      2,      512,      2,        16); // #16
        DISPATCH_GEMM(6144, 4096,   512,    256,   128,   128,   4,      2,      512,      1,        16); // #17
    // Test:                 M,    N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SZ,    SPLITK_F, 
        DISPATCH_GEMM_LEGACY(64,   64,     128,    64,    64,    32,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   1536,   7168,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   3072,   1536,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   576,    7168,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   7168,   256,    96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   7168,   2048,   96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   4608,   7168,   96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(128,  7168,   2304,   128,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(128,  512,    7168,   128,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(512,  4096,   512,    256,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(512,  1536,   7168,   256,   128,   64,   4,      2,      512,          1);
#endif
    default: {
        printf("Error: Unsupported shape M=%d, K=%d, N=%d\n", m, k, n);
        abort();
    }
    }
}
} // extern "C"
