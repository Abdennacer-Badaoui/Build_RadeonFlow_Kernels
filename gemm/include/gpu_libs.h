#ifdef TEST_ON_CUDA
#include <mma.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace wmma = nvcuda::wmma;

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) cuda##x

#else

#ifndef HIP_HEADERS__
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#define HIP_HEADERS__
#endif

namespace wmma = rocwmma;

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) hip##x

#endif

