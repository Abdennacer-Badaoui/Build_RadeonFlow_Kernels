#pragma once
#ifdef TEST_ON_CUDA
#define __FP8_TYPE __nv_fp8_e4m3
#define __FP8x4_TYPE __nv_fp8x4_e4m3
#define __BF16_TYPE __nv_bfloat16
#define __BF16x2_TYPE __nv_bfloat162

#else
#ifdef TEST_ON_RDNA4
#define __FP8_TYPE __hip_fp8_e4m3
#define __FP8x4_TYPE __hip_fp8x4_e4m3
constexpr const inline int WAVE_SIZE = 32;
constexpr const inline int XCD_SWIZZLE = 1;
#else
#define __FP8_TYPE __hip_fp8_e4m3_fnuz
#define __FP8x4_TYPE __hip_fp8x4_e4m3_fnuz
constexpr const inline int WAVE_SIZE = 64;
constexpr const inline int XCD_SWIZZLE = 8;
#endif

#define __BF16_TYPE __hip_bfloat16
#define __BF16x2_TYPE __hip_bfloat162
#define __FP16_TYPE __half
#define __INT16_TYPE int16_t

#endif

#define __FP32_TYPE float