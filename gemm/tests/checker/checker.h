
#pragma once
#include <iostream>
#include <vector>
#include "torch/torch.h"
#include "metrics.h"

enum class CheckerMode {
  kElementWise,
  kRowIndex,
  kJustDump,
};

struct Checkee {
  torch::Tensor *tensor;
  CheckerMode mode;
  std::string name;
};

void case_initialize();
int get_params_count();
void *case_get_input(int index);
std::vector<Checkee> case_run_kernel(void *input, PerfMetrics* metrics);
std::vector<Checkee> case_run_ref_kernel(void *input);
const char *case_get_name();
void get_error_tolerance(float *rtol, float *atol);
void case_destroy(void *input);
CheckerMode get_checker_mode();


// using OutputData = torch::Tensor;
// void ref_kernel(const BlockwiseMatmulInputs &data);
// BlockwiseMatmulInputs generate_input(int m, int n, int k, int seed);