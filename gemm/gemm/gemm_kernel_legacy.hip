// Legacy version of gemm kernel, support all shape and various value of parameters (BM, BN, BK, etc.)
// It has been replace with faster pipeline version.
#pragma once
#include <cstdio>
#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../src/utils/arithmetic.h"
#include "../include/clangd_workaround.h"
#include <cstdlib>
#include <cfloat>

DEVICE_CODE_BELOW
namespace gemm_kernel_legacy {



template <typename data_type, int BATCH_SIZE> 
__device__ inline void read_batch(data_type *dst, const data_type *src) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == 2 * sizeof(ulong4)) {
        *(reinterpret_cast<ulong4 *>(dst) + 0) = *(reinterpret_cast<const ulong4 *>(src) + 0);
        *(reinterpret_cast<ulong4 *>(dst) + 1) = *(reinterpret_cast<const ulong4 *>(src) + 1);
    } else if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = *reinterpret_cast<const ulong4 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = *reinterpret_cast<const ulong2 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = *reinterpret_cast<const ulong1 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint1 *>(dst) = *reinterpret_cast<const uint1 *>(src);
    } else {
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = src[b];
        }
    }
}

template <typename data_type, int BATCH_SIZE> 
__device__ inline void zero_batch(data_type *dst) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = ulong4{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = ulong2{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = ulong1{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint *>(dst) = uint{};
    } else {
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = 0;
        }
    }
}

template <typename data_type, int DST_Y, int DST_X, int SRC_Y, int SRC_X, int BLOCK_DIM, int BATCH_SIZE>
__device__ inline void load_input(data_type dst[DST_Y][DST_X], const data_type src[SRC_Y][SRC_X],
                                         const int begin_x, const int begin_y) {
  static_assert(BATCH_SIZE > 0);
  /**
    Consider (SRC_X % DST_X == 0) && (SRC_Y % DST_Y == 0)
    Step 1:
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
    Step 2:
      [   ][   ][   ][   ]
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
  */
  static_assert((SRC_X % BATCH_SIZE == 0) && (SRC_Y % BATCH_SIZE == 0));
  static_assert((DST_X % BATCH_SIZE == 0) && (DST_Y % BATCH_SIZE == 0));
  static_assert(BATCH_SIZE <= DST_X && DST_X % BATCH_SIZE == 0);
  const int begin_idx = threadIdx.x * BATCH_SIZE;
  const constexpr int total_elements = DST_X * DST_Y;
  const constexpr int elements_per_step = BLOCK_DIM * BATCH_SIZE;
  // FIXME: loop unrolling
  #pragma unroll
  for (int k = begin_idx; k < total_elements; k += elements_per_step) {
    int l_kx = k % DST_X;
    int l_ky = k / DST_X;
    int g_kx = l_kx + begin_x;
    int g_ky = l_ky + begin_y;
    auto *dst_flatten = &dst[l_ky][l_kx];
    // const auto *src_flatten = &src[g_ky][g_kx];
    // read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    if (((SRC_X % DST_X == 0) || (g_kx < SRC_X)) && ((SRC_Y % DST_Y == 0) || (g_ky < SRC_Y))) {
      const auto *src_flatten = &src[g_ky][g_kx];
      read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    } else {
      zero_batch<data_type, BATCH_SIZE>(dst_flatten);
    }
  }
}

template <int PM, int PN, int QM, int QN, int QK, int QUANT_SIZE, int BLOCK_SIZE, int BATCH_SIZE>
__device__ void load_scale(float s_s[PM][PN], const float sa[QK][QM], const float sb[QK][QN], 
                const int m, const int n, const int k) {
    constexpr int total_elements = PM * PN;
    constexpr int elements_per_step = BLOCK_SIZE * BATCH_SIZE;
    // static_assert(PN % BATCH_SIZE)
    
    const int begin_idx = threadIdx.x * BATCH_SIZE;
    #pragma unroll
    for (int idx = begin_idx; idx < total_elements; idx += elements_per_step) {
        static_assert(BATCH_SIZE == 1);
        int i = idx / PN;
        int j = idx % PN;
        if (((QM % PM == 0) || (m + i < QM)) && ((QN % PN == 0) || ((n + j) / QUANT_SIZE < QN))) {
            s_s[i][j] = sa[k / QUANT_SIZE][(m + i)] * sb[k / QUANT_SIZE][(n) / QUANT_SIZE + j];
        } else {
            s_s[i][j] = 1.0f;
        }
    }
    
}

template <typename in_data_type, typename acc_data_type,
    typename FragC, typename FragA, typename FragB, 
    int PM, int PN,
    int BM, int BN, int BK, 
    int FRAG_M, int FRAG_N, int FRAG_K, 
    int WMMA_M, int WMMA_N, int WMMA_K,
    int WARP_M, int WARP_N,
    int BLOCK_SIZE, int BATCH_SIZE, int QUANT_SIZE>
__device__ void wmma_compute(
    const in_data_type s_a[BK][BM],
    const in_data_type s_b[BK][BN],
    const float s_s[PM][PN],
    FragC frag_r[FRAG_M][FRAG_N],
    const int comp_c_frag_m,
    const int comp_c_frag_n
) {
    FragA frag_a[FRAG_K][FRAG_M]; 
    FragB frag_b[FRAG_K][FRAG_N];

    // Spilt k over BK
    for (int k = 0; k < FRAG_K; ++k) {
        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            int s_a_row = k * WMMA_K;
            int s_a_col = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
            wmma::load_matrix_sync(frag_a[k][i], &s_a[s_a_row][s_a_col], BM);
        }
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            int s_b_row = k * WMMA_K;
            int s_b_col = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
            wmma::load_matrix_sync(frag_b[k][j], &s_b[s_b_row][s_b_col], BN);
        }
    }

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            FragC frag_c;
            wmma::fill_fragment(frag_c, 0.0F);
            #pragma unroll
            for (int k = 0; k < FRAG_K; ++k) {
                wmma::mma_sync(frag_c, frag_a[k][i], frag_b[k][j], frag_c);
            }
            #pragma unroll
            for (int k = 0; k < FragC::num_elements; ++k) {
                #ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
                int m = ((threadIdx.x & 16) >> 1) | (k & 7) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                #else // CDNA3, WAVE_SIZE = 64
                int m = ((threadIdx.x & 48) >> 2) | (k & 3) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                #endif
                int n = ((threadIdx.x & 15) | (comp_c_frag_n * FRAG_N + j) * WMMA_N) / QUANT_SIZE;
                float scale = s_s[m][n];
                frag_r[i][j].x[k] += (acc_data_type)scale * (acc_data_type)frag_c.x[k];
            }  
        }
    }
}


template <typename acc_data_type, typename out_data_type, 
typename FragC, typename FragOut, int WMMA_M, int WMMA_N, 
int BM, int BN, int M, int N, int FRAG_M, int FRAG_N>
__device__ inline void store_result(
    out_data_type c[M][N], 
    FragC frag_r[FRAG_M][FRAG_N], 
    const int m,
    const int n,
    const int comp_c_frag_m, 
    const int comp_c_frag_n
) {
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            int frag_m = comp_c_frag_m * FRAG_M + i;
            int frag_n = comp_c_frag_n * FRAG_N + j;
            int row = m + frag_m * WMMA_M;
            int col = n + frag_n * WMMA_N;
            if (((M % BM == 0) || (row < M)) && ((N % BN == 0) || (col < N))) {
                out_data_type *c_ptr = &c[row][col];
                if constexpr (sizeof(acc_data_type) == sizeof(out_data_type)) {
                    wmma::store_matrix_sync(reinterpret_cast<out_data_type*>(c_ptr), frag_r[i][j], N, wmma::mem_row_major);
                } else if constexpr (sizeof(out_data_type) == sizeof(half)) {
                    FragOut frag_out;
                    static_assert(sizeof(half) == sizeof(out_data_type));
                    static_assert(FragOut::num_elements == FragC::num_elements);
                    for (int k=0;k<FragOut::num_elements;++k) {
                        __hip_bfloat16 reg = frag_r[i][j].x[k];
                        frag_out.x[k] = *reinterpret_cast<half*>(&reg);
                    }
                    wmma::store_matrix_sync(reinterpret_cast<half*>(c_ptr), frag_out, N, wmma::mem_row_major);
                } else {
                    static_assert(0, "Unsupported data type for output");
                }

            }
        }
    }
}

// a dummy template to allow inlcuding this file
template<int Dummy=0>
__global__ void reduce(uint32_t m, uint32_t n, uint32_t splitk, const float *c_splitk, __hip_bfloat16 *c) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    float sum = 0;
    for (auto i = 0; i < splitk; ++i) {
        sum += c_splitk[i * (m * n) + tid];
    }
    c[tid] = sum;
}

    
#ifdef PARAMETERIZE_LIBRARY
template <
    typename in_data_type,
    typename acc_data_type, // Accumulator type (e.g., float)
    typename out_data_type, // Output type (e.g., __hip_bfloat16)
    int M, int N, int K,    // Matrix dimensions
    int BM, int BN, int BK, // Tile dimensions
    int QUANT_SIZE,         // Quantization block size
    int BLOCK_SIZE,         // Block size
    int WARP_M, int WARP_N // Warp dimensions
>
#else
using in_data_type = __FP8_TYPE;
using out_data_type = __BF16_TYPE;
using acc_data_type = float;
// constexpr int M = 4096, N = 4096, K = 4096;
constexpr int M = 96, N = 1024, K = 1024;
// constexpr int M = 512, N = 512, K = 512;
constexpr int BM = 64, BN = 256, BK = 32;
constexpr int QUANT_SIZE = 128, BLOCK_SIZE = 256;
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
constexpr int WARP_M = 4, WARP_N = 2;
#else // CDNA3, WAVE_SIZE = 64
constexpr int WARP_M = 2, WARP_N = 2;
#endif
#endif // End of parameterization
__global__ void gemm_kernel(
    const in_data_type a[K][M],
    const in_data_type b[K][N],
    out_data_type c[M][N],
    const float sa[ceil_div(K, QUANT_SIZE)][M / 1        ], // Assuming M is divisible by 1 (always true)
    const float sb[ceil_div(K, QUANT_SIZE)][ceil_div(N, QUANT_SIZE)]
) {
    // --- Start: Derived parameters and constants ---
    constexpr int WMMA_M = 16; // Fixed WMMA dimension M
    constexpr int WMMA_N = 16; // Fixed WMMA dimension N
    constexpr int WMMA_K = 32; // Fixed WMMA dimension K (for FP8)

    // WARP_M/N define the 2D arrangement of warps in the block grid.
    // These might need adjustment based on BLOCK_DIM_X/Y strategy.
    // Using fixed values based on the non-parameterized version for now.
    // TODO: Derive WARP_M/N from BLOCK_DIM_X/Y if a flexible strategy is needed.
    constexpr int WARP_NUM = WARP_M * WARP_N; // Total warps per block

    // Assertion: Check if the assumed warp layout matches the block size
    static_assert(WARP_NUM * WAVE_SIZE == BLOCK_SIZE, "WARP_M * WARP_N * WAVE_SIZE must equal BLOCK_SIZE");

    // Fragments per warp
    constexpr int FRAG_M_PER_WARP = BM / WMMA_M / WARP_M;
    constexpr int FRAG_N_PER_WARP = BN / WMMA_N / WARP_N;
    constexpr int FRAG_K = BK / WMMA_K; // Fragments along K dimension tile

    static_assert(BM % (WMMA_M * WARP_M) == 0, "BM must be divisible by WMMA_M * WARP_M");
    static_assert(BN % (WMMA_N * WARP_N) == 0, "BN must be divisible by WMMA_N * WARP_N");
    static_assert(BK % WMMA_K == 0, "BK must be divisible by WMMA_K");
    static_assert(BK >= 32, "BK must be at least 32");
    // --- End: Derived parameters and constants ---

    constexpr int QM = M; // Dimension M for scale A
    constexpr int QN = ceil_div(N, QUANT_SIZE); // Dimension N for scale B (quantized)
    constexpr int QK = ceil_div(K, QUANT_SIZE); // Dimension K for scales (quantized)
    constexpr int PM = BM; // Block size M for scale A * B
    constexpr int PN = ceil_div(BN, QUANT_SIZE); // Block size N for scale A * B

    // Ensure derived fragment counts are positive
    static_assert(FRAG_M_PER_WARP > 0, "FRAG_M_PER_WARP must be positive");
    static_assert(FRAG_N_PER_WARP > 0, "FRAG_N_PER_WARP must be positive");
    static_assert(FRAG_K > 0, "FRAG_K must be positive");

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, acc_data_type>;
    using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>; // Output uses half for storage via bfloat16 reinterpret

    __shared__ in_data_type  s_a[BK][BM];
    __shared__ in_data_type  s_b[BK][BN];
    __shared__ acc_data_type s_s[PM][PN]; // Accumulator type for scales
    FragC frag_r[FRAG_M_PER_WARP][FRAG_N_PER_WARP]; // Accumulator fragments

    // handle splitk
    a += blockIdx.z * K;
    b += blockIdx.z * K;
    c += blockIdx.z * M;
    sa += blockIdx.z * QK;
    sb += blockIdx.z * QK;

    int tid = threadIdx.x; // Linear thread ID within the block
    int wid = tid / WAVE_SIZE; // Warp ID within the block

    // Initialize the output accumulator fragments to zero
    #pragma unroll
    for (int i = 0; i < FRAG_M_PER_WARP; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N_PER_WARP; j++) {
            wmma::fill_fragment(frag_r[i][j], 0.0f); // Use float literal
        }
    }

    // Spilt and compute fragments
    constexpr int iteration_over_k = ceil_div(K, BK); // Use ceil_div for potentially non-divisible K
    constexpr int LOAD_BATCH_SIZE = (2 * sizeof(float4) / sizeof(in_data_type)) > 0 ? (2 * sizeof(float4) / sizeof(in_data_type)) : 1; // Ensure batch size > 0
    static_assert(LOAD_BATCH_SIZE > 0, "LOAD_BATCH_SIZE must be positive");

    for (int bk = 0; bk < iteration_over_k; bk++) {
        const int m = blockIdx.y * BM;
        const int n = blockIdx.x * BN;
        const int k = bk * BK;

        // Calculate remaining K for boundary checks if needed (not currently used by load_input)
        // const int k_rem = K - k;

        // Load data into shared memory
        load_input<in_data_type, BK, BM, K, M, BLOCK_SIZE, LOAD_BATCH_SIZE>(
            s_a, a, m, k);
        load_input<in_data_type, BK, BN, K, N, BLOCK_SIZE, LOAD_BATCH_SIZE>(
            s_b, b, n, k);
        // Load scales into shared memory (using acc_data_type for s_s)
        load_scale<PM, PN, QM, QN, QK, QUANT_SIZE, BLOCK_SIZE, 1>(
            s_s, sa, sb, m, n, k);
        __syncthreads();

        // Perform matrix multiplication using WMMA
        wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB,
            PM, PN, BM, BN, BK, FRAG_M_PER_WARP, FRAG_N_PER_WARP, FRAG_K,
            WMMA_M, WMMA_N, WMMA_K,
            WARP_M, WARP_N,
            BLOCK_SIZE, LOAD_BATCH_SIZE, QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
                s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
        __syncthreads();
    }
    // Store results from accumulator fragments to global memory
    store_result<acc_data_type, out_data_type, FragC, FragOut,
        WMMA_M, WMMA_N, BM, BN, M, N, FRAG_M_PER_WARP, FRAG_N_PER_WARP>(
            c, frag_r, blockIdx.y * BM, blockIdx.x * BN,
            wid / WARP_N, wid % WARP_N);


};


}; // namespace gemm_kernel_legacy