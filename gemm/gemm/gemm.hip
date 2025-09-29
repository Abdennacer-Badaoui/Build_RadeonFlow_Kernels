// Pipeline GEMM kernel. This version is rushed written and may not applied to all shape.
// Currently, only selected parameters is tested. (See gemm_launcher )
#ifndef GEMM_KERNEL
#define GEMM_KERNEL

#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../src/utils/arithmetic.h"
#include "../include/clangd_workaround.h"
#include <cstdlib>
#include <cfloat>

namespace gemm_kernel {

template <typename data_type, int BATCH_SIZE> __device__ inline void read_batch(data_type *dst, const data_type *src) {
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

template <typename data_type, int BATCH_SIZE> __device__ inline void zero_batch(data_type *dst) {
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
__device__ inline void load_input(data_type dst[DST_Y][DST_X], const data_type src[SRC_Y][SRC_X], const int begin_x,
                                  const int begin_y) {
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
__device__ void load_scale(float s_s[PM][PN], const float sa[QK][QM], const float sb[QK][QN], const int m, const int n,
                           const int k) {
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

// don't use __builtin_readcyclecounter(), which would insert waitcnt
__device__ auto getclock() {
    uint64_t clk;
    asm volatile("s_memtime %0" : "=r"(clk));
    return clk;
}


template <typename Elem> __global__ void check_trans(const Elem *origin, const Elem *tranposed, int m, int n) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < m && y < n) {
        if (origin[x * n + y] != tranposed[y * m + x]) {
            printf("Error: %d %d\n", x, y);
        }
    }
}

template <typename in_data_type, typename acc_data_type, typename FragC, typename FragA, typename FragB, int PM, int PN,
          int BM, int BN, int BK, int FRAG_M, int FRAG_N, int FRAG_K, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_M,
          int WARP_N, int BLOCK_SIZE, int BATCH_SIZE, int QUANT_SIZE>
__device__ void wmma_compute(const in_data_type s_a[BM][BK + 8], const in_data_type s_b[BN][BK + 8],
                             const float s_s[PN][PM], FragC frag_r[FRAG_M][FRAG_N], const int comp_c_frag_m,
                             const int comp_c_frag_n) {
    FragC frag_c[FRAG_M][FRAG_N];

#pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0F);
        }
    }

#pragma unroll
    for (int k = 0; k < FRAG_K; ++k) {
#pragma unroll
        for (int i = 0; i < FRAG_M; i++) {
            FragA frag_a;
            int s_a_row = k * WMMA_K;
            int s_a_col = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
            wmma::load_matrix_sync(frag_a, &s_a[s_a_col][s_a_row], BK + 8);
#pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                FragB frag_b;
                int s_b_row = k * WMMA_K;
                int s_b_col = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
                wmma::load_matrix_sync(frag_b, &s_b[s_b_col][s_b_row], BK + 8);

                wmma::mma_sync(frag_c[i][j], frag_a, frag_b, frag_c[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
#pragma unroll
            for (int k = 0; k < FragC::num_elements; ++k) {
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
                int m = ((threadIdx.x & 16) >> 1) | (k & 7) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
#else // CDNA3, WAVE_SIZE = 64
// int m = ((threadIdx.x & 48) >> 2) | (k & 3) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
#endif
                // int n = ((threadIdx.x & 15) | (comp_c_frag_n * FRAG_N + j) * WMMA_N) / QUANT_SIZE;
                auto lane = threadIdx.x % 64;
                int m, n;
                if constexpr (WMMA_M == 32) {
                    // C or D i: (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
                    // C or D j: (lane % 32)
                    m = (8 * (k / 4) % 32) + 4 * (lane / 32) + (k % 4);
                    n = lane % 32;
                } else {
                    // C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
                    // C or D j: (lane % 16)
                    m = 4 * (lane / 16) + (k % 4);
                    n = lane % 16;
                }
                m += (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                n += (comp_c_frag_n * FRAG_N + j) * WMMA_N;
                n = n / QUANT_SIZE;
                // if(threadIdx.x == 192 && blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0)
                // printf("m: %d, n: %d\n", m, n);
                float scale = s_s[n][m];
                frag_r[i][j].x[k] += (acc_data_type)scale * (acc_data_type)frag_c[i][j].x[k];
            }
        }
    }
}

__device__ rocwmma::bfloat16_t fast_f32tob16(float f) {
    union {
        float fp32;
        unsigned int u32;
    } u = {f};
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    auto ret = u.u32 >> 16;
    return reinterpret_cast<rocwmma::bfloat16_t &>(ret);
}

template <typename acc_data_type, typename out_data_type, typename FragC, typename FragOut, int WMMA_M, int WMMA_N,
          int BM, int BN, int M, int N, int FRAG_M, int FRAG_N>
__device__ inline void store_result(out_data_type c[M][N], FragC frag_r[FRAG_M][FRAG_N], const int m, const int n,
                                    const int comp_c_frag_m, const int comp_c_frag_n) {
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
                if constexpr (sizeof(acc_data_type) == sizeof(out_data_type)) { // split_k
                    auto lane = threadIdx.x % 64;
                    #pragma unroll
                    for (int k = 0; k < FragC::num_elements; ++k) {
                        int m, n;
                        if constexpr (WMMA_M == 32) {
                            // C or D i: (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
                            // C or D j: (lane % 32)
                            m = (8 * (k / 4) % 32) + 4 * (lane / 32) + (k % 4);
                            n = lane % 32;
                        } else {
                            // C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
                            // C or D j: (lane % 16)
                            m = 4 * (lane / 16) + (k % 4);
                            n = lane % 16;
                        }
                        c_ptr[m * N + n] = frag_r[i][j].x[k];;
                    }
                    
                    // wmma::store_matrix_sync(reinterpret_cast<out_data_type *>(c_ptr), frag_r[i][j], N,
                    //                         wmma::mem_row_major);
                } else if constexpr (sizeof(out_data_type) == sizeof(half)) {
                    FragOut frag_out;
                    static_assert(sizeof(half) == sizeof(out_data_type));
                    static_assert(FragOut::num_elements == FragC::num_elements);
                    for (int k = 0; k < FragOut::num_elements; ++k) {
                        auto reg = fast_f32tob16(frag_r[i][j].x[k]);
                        frag_out.x[k] = *reinterpret_cast<half *>(&reg);
                    }
                    wmma::store_matrix_sync(reinterpret_cast<half *>(c_ptr), frag_out, N, wmma::mem_row_major);
                } else {
                    static_assert(0, "Unsupported data type for output");
                }
            }
        }
    }
}

// a dummy template to allow inlcuding this file
template <int Splitk> __global__ void reduce(uint32_t m, uint32_t n, const float *c_splitk, __hip_bfloat16 *c) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    float4 sum{};
#pragma unroll
    for (auto i = 0; i < Splitk; ++i) {
        sum += *(float4 *)&c_splitk[i * (m * n) + tid * 4];
    }
    auto res =
        rocwmma::make_vector(fast_f32tob16(sum.x), fast_f32tob16(sum.y), fast_f32tob16(sum.z), fast_f32tob16(sum.w));
    *(decltype(res) *)&c[tid * 4] = res;
}

template<int M, int N, int SPLITK_FACTOR, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void reduce_kernel(const float c_splitk[SPLITK_FACTOR][M][N], __hip_bfloat16 c[M][N]) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) {
        return;
    }
    float4 sum{};
    #pragma unroll
    for (auto i = 0; i < SPLITK_FACTOR; ++i) {
        sum += *(float4 *)&reinterpret_cast<const float*>(c_splitk)[i * (M * N) + tid * 4];
    }
    auto res =
        rocwmma::make_vector(fast_f32tob16(sum.x), fast_f32tob16(sum.y), fast_f32tob16(sum.z), fast_f32tob16(sum.w));
    *(decltype(res) *)&reinterpret_cast< __BF16_TYPE*>(c)[tid * 4] = res;
}


#ifdef PARAMETERIZE_LIBRARY
template <typename in_data_type,
          typename acc_data_type, // Accumulator type (e.g., float)
          typename out_data_type, // Output type (e.g., __hip_bfloat16)
          int M, int N, int K,    // Matrix dimensions
          int BM, int BN, int BK, // Tile dimensions
          int QUANT_SIZE,         // Quantization block size
          int BLOCK_SIZE,         // Block size
          int WARP_M, int WARP_N, // Warp dimensions
          int LDA, int LDB,
          int LOAD_BATCH_SIZE>    // Load batch size for vectorized memory operations
#else
using in_data_type = __FP8_TYPE;
using out_data_type = __BF16_TYPE;
using acc_data_type = float;
// constexpr int M = 4096, N = 4096, K = 4096;
constexpr int M = 6144, N = 4608, K = 7168;
constexpr int LDA = K, LDB = K;
// constexpr int M = 512, N = 512, K = 512;
constexpr int BM = 256, BN = 128, BK = 128;
constexpr int QUANT_SIZE = 128, BLOCK_SIZE = 512;
constexpr int LOAD_BATCH_SIZE = 16;
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
constexpr int WARP_M = 4, WARP_N = 2;
#else                // CDNA3, WAVE_SIZE = 64
constexpr int WARP_M = 4, WARP_N = 2;
#endif
#endif // End of parameterization
__global__ __launch_bounds__(BLOCK_SIZE) void gemm_kernel(
    const in_data_type a[M][LDA], const in_data_type b[N][LDB], out_data_type c[M][N],
    const float sa[ceil_div(K, QUANT_SIZE)][M / 1], // Assuming M is divisible by 1 (always true)
    const float sb[ceil_div(K, QUANT_SIZE)][ceil_div(N, QUANT_SIZE)]) {
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

    constexpr int QM = M;                        // Dimension M for scale A
    constexpr int QN = ceil_div(N, QUANT_SIZE);  // Dimension N for scale B (quantized)
    constexpr int QK = ceil_div(K, QUANT_SIZE);  // Dimension K for scales (quantized)
    constexpr int PM = BM;                       // Block size M for scale A * B
    constexpr int PN = ceil_div(BN, QUANT_SIZE); // Block size N for scale A * B

    // Ensure derived fragment counts are positive
    static_assert(FRAG_M_PER_WARP > 0, "FRAG_M_PER_WARP must be positive");
    static_assert(FRAG_N_PER_WARP > 0, "FRAG_N_PER_WARP must be positive");
    static_assert(FRAG_K > 0, "FRAG_K must be positive");

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, acc_data_type>;
    using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                                   half>; // Output uses half for storage via bfloat16 reinterpret

    __shared__ in_data_type s_a[BM][BK + 8];
    __shared__ in_data_type s_b[BN][BK + 8];
    __shared__ acc_data_type s_s[PN][PM];           // Accumulator type for scales
    FragC frag_r[FRAG_M_PER_WARP][FRAG_N_PER_WARP]; // Accumulator fragments

    // handle splitk
    a = (decltype(a))((in_data_type *)a + blockIdx.z * K);
    b = (decltype(b))((in_data_type *)b + blockIdx.z * K);
    c += blockIdx.z * M;
    sa += blockIdx.z * QK;
    sb += blockIdx.z * QK;

    int tid = threadIdx.x;     // Linear thread ID within the block
    int wid = tid / WAVE_SIZE; // Warp ID within the block

    // Spilt and compute fragments
    constexpr int iteration_over_k = ceil_div(K, BK); // Use ceil_div for potentially non-divisible K
    static_assert(LOAD_BATCH_SIZE > 0, "LOAD_BATCH_SIZE must be positive");

    constexpr auto PIPELINE = true;
    // using LoadVec = rocwmma::VecT<float, LOAD_BATCH_SIZE / sizeof(float)>;
    using LoadVec = __attribute__((__vector_size__(LOAD_BATCH_SIZE))) float;
    static_assert(((BK * BM) % (BLOCK_SIZE * LOAD_BATCH_SIZE)) == 0,
                  "BK * BM must be divisible by BLOCK_SIZE * LOAD_BATCH_SIZE");
    static_assert(BK % LOAD_BATCH_SIZE == 0, "BK must be divisible by LOAD_BATCH_SIZE");
    LoadVec reg_a[BK * BM / BLOCK_SIZE / LOAD_BATCH_SIZE];
    LoadVec reg_b[BK * BN / BLOCK_SIZE / LOAD_BATCH_SIZE];
    constexpr auto PK = ceil_div(BK, QUANT_SIZE);
    static_assert(PK == 1, "PK must be 1 for now");
    float reg_sa[ceil_div(PM, BLOCK_SIZE)];
    float reg_sb[ceil_div(PN, BLOCK_SIZE)];

    // threadblock swizzle
    auto log_tile = 1;
    auto block_idx_x = blockIdx.x >> log_tile;
    auto block_idx_y = (blockIdx.y << log_tile) + ((blockIdx.x) & ((1 << (log_tile)) - 1));
    if (block_idx_x >= ceil_div(N, BN) || block_idx_y >= ceil_div(M, BM)) {
        return;
    }

    const int m = block_idx_y * BM;
    const int n = block_idx_x * BN;
    int k = 0;

    auto global2reg = [&]() {
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_sa) / sizeof(float); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid + reg * BLOCK_SIZE;
            // NOTE: don't branch here
            // if (t > PM) {
            //     break;
            // }
            int i = t / PM;
            int j = t % PM;
            reg_sa[reg] = sa[k / QUANT_SIZE][(m + j)];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_sb) / sizeof(float); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid + reg * BLOCK_SIZE;
            // NOTE: don't branch here
            // if (t > PN) {
            //     break;
            // }
            int i = t / PN;
            int j = t % PN;
            reg_sb[reg] = sb[k / QUANT_SIZE][(n) / QUANT_SIZE + j];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_a) / sizeof(LoadVec); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            reg_a[reg] = *(LoadVec *)&a[m + i][k + j];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_b) / sizeof(LoadVec); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            reg_b[reg] = *(LoadVec *)&b[n + i][k + j];
        }
    };

    auto reg2lds = [&]() {
#pragma unroll
        for (int rega = 0; rega < sizeof(reg_sa) / sizeof(float); rega++) {
            int ta = tid + rega * BLOCK_SIZE;
            int j = ta % PM;
#pragma unroll
            for (int regb = 0; regb < sizeof(reg_sb) / sizeof(float); regb++) {
                int tb = tid + regb * BLOCK_SIZE;
                int i = tb % PN;
                s_s[i][j] = reg_sa[rega] * reg_sb[regb];
            }
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_a) / sizeof(LoadVec); reg++) {
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            *(LoadVec *)&s_a[i][j] = reg_a[reg];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_b) / sizeof(LoadVec); reg++) {
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            *(LoadVec *)&s_b[i][j] = reg_b[reg];
        }
    };

    if constexpr (PIPELINE) {
        global2reg();
    }

// Initialize the output accumulator fragments to zero
#pragma unroll
    for (int i = 0; i < FRAG_M_PER_WARP; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N_PER_WARP; j++) {
            wmma::fill_fragment(frag_r[i][j], 0.0f); // Use float literal
        }
    }

    if constexpr (!PIPELINE) {
        global2reg();
    }

    reg2lds();

    for (int bk = 1; bk < iteration_over_k; bk++) {
        k = bk * BK;

        // Calculate remaining K for boundary checks if needed (not currently used by load_input)
        // const int k_rem = K - k;

        // Load data into shared memory
        // load_input<in_data_type, BK, BM, K, M, BLOCK_SIZE, 32>(
        //     s_a, a, m, k);
        // load_input<in_data_type, BK, BN, K, N, BLOCK_SIZE, 32>(
        //     s_b, b, n, k);
        // Load scales into shared memory (using acc_data_type for s_s)
        // load_scale<PM, PN, QM, QN, QK, QUANT_SIZE, BLOCK_SIZE, 1>(
        //     s_s, sa, sb, m, n, k);

        if constexpr (PIPELINE) {
            global2reg();
        }

        __syncthreads();

        // Perform matrix multiplication using WMMA
        wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB, PM, PN, BM, BN, BK, FRAG_M_PER_WARP,
                     FRAG_N_PER_WARP, FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE, LOAD_BATCH_SIZE,
                     QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
            s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
        __syncthreads();

        if constexpr (!PIPELINE) {
            global2reg();
        }

        // __builtin_amdgcn_sched_barrier(0);

        reg2lds();
    }
    __syncthreads();
    wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB, PM, PN, BM, BN, BK, FRAG_M_PER_WARP, FRAG_N_PER_WARP,
                 FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE, LOAD_BATCH_SIZE,
                 QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
        s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
    // Store results from accumulator fragments to global memory
    store_result<acc_data_type, out_data_type, FragC, FragOut, WMMA_M, WMMA_N, BM, BN, M, N, FRAG_M_PER_WARP,
                 FRAG_N_PER_WARP>(c, frag_r, block_idx_y * BM, block_idx_x * BN, wid / WARP_N, wid % WARP_N);
};

}; // namespace gemm_kernel

HOST_CODE_BELOW

#ifndef PARAMETERIZE_LIBRARY
// Define type aliases to match those in the namespace
using fp8_type = gemm_kernel::in_data_type;       // __hip_fp8_e4m3
using fp16_type = gemm_kernel::out_data_type;     // __hip_bfloat16
using acc_data_type = gemm_kernel::acc_data_type; // float

// Define constants to match those in the namespace
constexpr int M = gemm_kernel::M;   // 4096
constexpr int N = gemm_kernel::N;   // 4096
constexpr int K = gemm_kernel::K;   // 4096
constexpr int BM = gemm_kernel::BM; // 256
constexpr int BN = gemm_kernel::BN; // 128
constexpr int BK = gemm_kernel::BK; // 32
constexpr int BLOCK_SIZE = gemm_kernel::BLOCK_SIZE;
constexpr int QUANT_SIZE = gemm_kernel::QUANT_SIZE; // 128

// Define derived constants for the test
constexpr int QK = K / QUANT_SIZE;
constexpr int QM = M;
constexpr int QN = N / QUANT_SIZE;

// Helper function to check HIP errors
#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T> void check(T err, const char *const func, const char *const file, const int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", hipGetErrorString(err), func);
        exit(1);
    }
}

// Define a macro to check HIP errors
#define HIP_CALL(call)                                                                                                 \
    do {                                                                                                               \
        hipError_t err = call;                                                                                         \
        if (err != hipSuccess) {                                                                                       \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);                   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// CPU matrix multiplication implementation for result verification
void cpu_gemm(const fp8_type a[K][M], const fp8_type b[K][N], fp16_type c[M][N], const float sa[QK][QM],
              const float sb[QK][QN]) {
    float(*rc)[N] = new float[M][N];
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            rc[m][n] = 0.0f;
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float scale = sa[k / QUANT_SIZE][m] * sb[k / QUANT_SIZE][n / QUANT_SIZE];
                rc[m][n] += (scale * (float)a[k][m] * (float)b[k][n]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            c[m][n] = (fp16_type)rc[m][n];
        }
    }
    delete[] rc;
}

int main() {
    // Allocate host memory
    fp8_type(*h_a)[M] = new fp8_type[K][M];
    fp8_type(*h_b)[N] = new fp8_type[K][N];
    fp16_type(*h_c)[N] = new fp16_type[M][N];
    fp16_type(*h_c_ref)[N] = new fp16_type[M][N];

    // Allocate host memory for quantization scale factors
    float(*h_sa)[QM] = new float[QK][QM];
    float(*h_sb)[QN] = new float[QK][QN];

    // Initialize input data
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            h_a[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_b[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
        }
    }

    // Initialize quantization scale factors
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QM; ++j) {
            h_sa[i][j] = 1.0f;
        }
    }
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QN; ++j) {
            h_sb[i][j] = 1.0f;
        }
    }

    // Allocate device memory
    fp8_type(*d_a)[K];
    fp8_type(*d_b)[K];
    fp16_type(*d_c)[N];
    float(*d_sa)[QM];
    float(*d_sb)[QN];

    CHECK_HIP_ERROR(hipMalloc(&d_a, K * M * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, K * N * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, M * N * sizeof(fp16_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_sa, QK * QM * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_sb, QK * QN * sizeof(float)));

    // Copy data from host memory to device memory
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a, K * M * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b, K * N * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sa, h_sa, QK * QM * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sb, h_sb, QK * QN * sizeof(float), hipMemcpyHostToDevice));

    // Calculate grid and block sizes - ensure coverage of the entire matrix
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);

    // Ensure block size is a multiple of 32, since warp size is 32
    if (BLOCK_SIZE % 32 != 0) {
        printf("Error: Block size must be a multiple of warp size (32)\n");
        return 1;
    }

    // Check if device supports required compute capability
    int deviceId;
    HIP_CALL(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProp;
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));

    if (deviceProp.major < 7) {
        printf("Error: This kernel requires a GPU with compute capability 7.0 or higher\n");
        return 1;
    }

    printf("Running GEMM kernel with grid(%d,%d), block(%d)...\n", grid.x, grid.y, block.x);

    // Query and print kernel and device information
    printf("Querying kernel and device information...\n");

    // Get device properties
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("Device Name: %s\n", deviceProp.name);
    printf("Total Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
    printf("Shared Memory per Block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
    printf("Warp Size: %d\n", deviceProp.warpSize);
    printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);

    // Query kernel attributes
    hipFuncAttributes funcAttr;
    HIP_CALL(hipFuncGetAttributes(&funcAttr, (const void *)gemm_kernel::gemm_kernel));
    printf("Kernel Attributes:\n");
    printf("  Shared Memory Size: %lu bytes\n", funcAttr.sharedSizeBytes);
    printf("  Number of Registers: %d\n", funcAttr.numRegs);
    printf("  Max Threads per Block: %d\n", funcAttr.maxThreadsPerBlock);
    printf("  Local Memory Size: %lu bytes\n", funcAttr.localSizeBytes);

    // Zero the C matrix before launching kernel
    CHECK_HIP_ERROR(hipMemset(d_c, 0, M * N * sizeof(fp16_type)));

    // Perform warmup runs
    printf("Performing warmup runs...\n");
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Declare and create timing events
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    // Ensure device synchronization before formal timing
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(start));

    // Launch kernel
    printf("Launching kernel...\n");
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);

    // Record end time and calculate execution time
    HIP_CALL(hipEventRecord(stop));

    // Record end time and calculate execution time
    HIP_CALL(hipEventSynchronize(stop));
    float milliseconds = 0;
    HIP_CALL(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Check HIP errors
    CHECK_HIP_ERROR(hipGetLastError());

    // Calculate GPU performance metrics
    double operations = 2.0 * M * N * K; // Each multiply-add operation counts as 2 floating-point operations
    double seconds = milliseconds / 1000.0;
    double tflops = (operations / seconds) / 1e12;
    printf("GPU Performance: %.2f TFLOPS\n", tflops);

    return 0;

    // Copy results from device memory back to host memory
    CHECK_HIP_ERROR(hipMemcpy(h_c, d_c, M * N * sizeof(fp16_type), hipMemcpyDeviceToHost));

    // Calculate reference results
    printf("Computing reference result on CPU...\n");
    cpu_gemm(h_a, h_b, h_c_ref, h_sa, h_sb);

    // Print the first 10 values for comparison
    printf("First 10 values (GPU vs CPU):\n");
    int print_count = 0;
    for (int i = 0; i < M && print_count < 10; ++i) {
        for (int j = 0; j < N && print_count < 10; ++j) {
            printf("  [%d, %d]: GPU=%f, CPU=%f\n", i, j, (float)h_c[i][j], (float)h_c_ref[i][j]);
            print_count++;
        }
    }

    // Verify results
    printf("Verifying results...\n");
    int errors = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    struct ErrorInfo {
        int row, col;
        float gpu_val, cpu_val, abs_diff, rel_diff;
    };
    ErrorInfo first_10_errors[10];
    ErrorInfo max_10_errors[10] = {};

    // Add a configurable variable for the number of errors to output
    int max_errors_to_output = 10; // You can modify this value as needed

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = (float)h_c[i][j];
            float cpu_val = (float)h_c_ref[i][j];
            float abs_diff;
            float rel_diff;

            if (std::isnan(gpu_val) || std::isnan(cpu_val)) {
                abs_diff = INFINITY;
                rel_diff = INFINITY;
            } else {
                abs_diff = abs(gpu_val - cpu_val);
                rel_diff = abs_diff / (abs(cpu_val) + FLT_EPSILON);
            }

            // Track max absolute and relative differences
            max_abs_diff = fmaxf(max_abs_diff, abs_diff);
            max_rel_diff = fmaxf(max_rel_diff, rel_diff);

            // Record first 10 errors
            if (errors < max_errors_to_output && (rel_diff > 1e-2 || abs_diff > 1e-3)) {
                first_10_errors[errors] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
            }

            // Track top 10 largest errors
            if (rel_diff > 1e-2 || abs_diff > 1e-3) {
                errors++;
                for (int k = 0; k < max_errors_to_output; ++k) {
                    if (abs_diff > max_10_errors[k].abs_diff) {
                        for (int l = max_errors_to_output - 1; l > k; --l) {
                            max_10_errors[l] = max_10_errors[l - 1];
                        }
                        max_10_errors[k] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
                        break;
                    }
                }
            }
        }
    }

    // Print first 10 errors
    printf("First %d errors:\n", max_errors_to_output);
    for (int i = 0; i < fmin(errors, max_errors_to_output); ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n", first_10_errors[i].row,
               first_10_errors[i].col, first_10_errors[i].gpu_val, first_10_errors[i].cpu_val,
               first_10_errors[i].abs_diff, first_10_errors[i].rel_diff);
    }

    // Print top 10 largest errors
    printf("Top %d largest errors:\n", max_errors_to_output);
    for (int i = 0; i < max_errors_to_output && max_10_errors[i].abs_diff > 0; ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n", max_10_errors[i].row,
               max_10_errors[i].col, max_10_errors[i].gpu_val, max_10_errors[i].cpu_val, max_10_errors[i].abs_diff,
               max_10_errors[i].rel_diff);
    }

    printf("Max abs_diff: %f, Max rel_diff: %f\n", max_abs_diff, max_rel_diff);
    if (errors == 0) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED with %d errors\n", errors);
    }

    // Calculate performance
    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    delete[] h_sa;
    delete[] h_sb;
    HIP_CALL(hipFree(d_a));
    HIP_CALL(hipFree(d_b));
    HIP_CALL(hipFree(d_c));
    HIP_CALL(hipFree(d_sa));
    HIP_CALL(hipFree(d_sb));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));

    return 0;
}
#endif
#pragma clang diagnostic pop
#endif