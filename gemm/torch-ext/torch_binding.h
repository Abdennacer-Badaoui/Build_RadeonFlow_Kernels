#pragma once

#include <torch/torch.h>

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs);