#include <torch/all.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>

#include "registration.h"
#include "torch_binding.h"

// Forward declaration of the C function from gemm_launcher.hip
extern "C" {
    struct PerfMetrics;
    void run(void *a, void *b, void *as, void *bs, void *c, int m, int n, int k, PerfMetrics *metrics, hipStream_t job_stream0);
}

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs) {
    
    // Validate tensor properties
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on GPU device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on GPU device");
    TORCH_CHECK(as.device().is_cuda(), "Scale tensor as must be on GPU device");
    TORCH_CHECK(bs.device().is_cuda(), "Scale tensor bs must be on GPU device");
    TORCH_CHECK(out.device().is_cuda(), "Output tensor out must be on GPU device");
    
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(as.is_contiguous(), "Scale tensor as must be contiguous");
    TORCH_CHECK(bs.is_contiguous(), "Scale tensor bs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "Output tensor out must be contiguous");
    
    // Get matrix dimensions from tensor shapes
    // Assuming a is [M, K], b is [K, N], out is [M, N]
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    TORCH_CHECK(b.size(0) == K, "Matrix dimensions mismatch: a.size(1) != b.size(0)");
    TORCH_CHECK(out.size(0) == M, "Output tensor dimension mismatch: out.size(0) != M");
    TORCH_CHECK(out.size(1) == N, "Output tensor dimension mismatch: out.size(1) != N");
    
    // Use default HIP stream (stream 0)
    const hipStream_t stream = 0;
    
    // Call the C function
    run(a.data_ptr(), b.data_ptr(), as.data_ptr(), bs.data_ptr(), out.data_ptr(),
        M, N, K, nullptr, stream);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm(Tensor! out, Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> ()");
  ops.impl("gemm", torch::kCUDA, &gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
