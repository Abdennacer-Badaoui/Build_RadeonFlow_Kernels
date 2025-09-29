#include "checker.h"
#include <dlfcn.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <getopt.h>
#include <unistd.h>

std::pair<bool, std::string> verbose_allclose(const torch::Tensor &received, const torch::Tensor &expected,
                                              float rtol = 1e-05, float atol = 1e-08, int max_print = 5) {
    // Check if the shapes of the tensors match
    if (received.sizes() != expected.sizes()) {
        std::string expected_shape_str = "[";
        std::string received_shape_str = "[";
        auto expected_sizes = expected.sizes();
        auto received_sizes = received.sizes();

        for (int i = 0; i < expected_sizes.size(); i++) {
            expected_shape_str += std::to_string(expected_sizes[i]);
            if (i < expected_sizes.size() - 1)
                expected_shape_str += ", ";
        }
        expected_shape_str += "]";

        for (int i = 0; i < received_sizes.size(); i++) {
            received_shape_str += std::to_string(received_sizes[i]);
            if (i < received_sizes.size() - 1)
                received_shape_str += ", ";
        }
        received_shape_str += "]";

        return {false, "SIZE MISMATCH: expected " + expected_shape_str + " but got " + received_shape_str};
    }

    auto diff = torch::abs(received.to(torch::kFloat32) - expected.to(torch::kFloat32));

    auto tolerance = atol + rtol * torch::abs(expected);

    auto tol_mismatched = diff > tolerance;
    auto nan_mismatched = torch::logical_xor(torch::isnan(received), torch::isnan(expected));
    auto posinf_mismatched = torch::logical_xor(torch::isposinf(received), torch::isposinf(expected));
    auto neginf_mismatched = torch::logical_xor(torch::isneginf(received), torch::isneginf(expected));

    auto mismatched = torch::logical_or(torch::logical_or(tol_mismatched, nan_mismatched),
                                        torch::logical_or(posinf_mismatched, neginf_mismatched));

    auto mismatched_indices = torch::nonzero(mismatched);

    // Count the number of mismatched elements
    int64_t num_mismatched = mismatched.sum().item<int64_t>();

    // Generate detailed information if there are mismatches
    if (num_mismatched >= 1) {
        std::stringstream mismatch_details;
        auto sizes = received.sizes();
        mismatch_details << "Mismatch found in tensors with shape [";
        for (int i = 0; i < sizes.size(); i++) {
            mismatch_details << sizes[i];
            if (i < sizes.size() - 1)
                mismatch_details << ", ";
        }
        mismatch_details << "]:\n";
        mismatch_details << "Number of mismatched elements: " << num_mismatched << "\n";

        for (int i = 0; i < std::min(max_print, (int)mismatched_indices.size(0)); i++) {
            auto index = mismatched_indices[i];
            std::vector<int64_t> idx_vec;
            for (int j = 0; j < index.size(0); j++) {
                idx_vec.push_back(index[j].item<int64_t>());
            }

            // Format the index as a string
            std::string idx_str = "(";
            for (size_t j = 0; j < idx_vec.size(); j++) {
                idx_str += std::to_string(idx_vec[j]);
                if (j < idx_vec.size() - 1)
                    idx_str += ", ";
            }
            idx_str += ")";

            float received_val, expected_val;
            torch::Tensor received_elem = received;
            torch::Tensor expected_elem = expected;

            for (size_t j = 0; j < idx_vec.size(); j++) {
                received_elem = received_elem[idx_vec[j]];
                expected_elem = expected_elem[idx_vec[j]];
            }

            received_val = received_elem.item<float>();
            expected_val = expected_elem.item<float>();

            mismatch_details << "ERROR at " << idx_str << ": " << received_val << " " << expected_val << "\n";
        }

        if (num_mismatched > max_print) {
            mismatch_details << "... and " << (num_mismatched - max_print) << " more mismatched elements.";
        }

        return {false, mismatch_details.str()};
    }

    return {true, "Maximum error: " + std::to_string(diff.max().item<float>())};
}

// Check if implementation matches reference within tolerance
std::pair<bool, std::string> check_implementation(std::ofstream &fout, const torch::Tensor &output,
                                                  const torch::Tensor &expected, float rtol = 2e-02, float atol = 1e-03,
                                                  CheckerMode mode = CheckerMode::kElementWise) {
    if (mode == CheckerMode::kRowIndex) {
        // For row index mode, we need to sort each row before comparison
        // since the order of indices with the same values might differ
        auto sorted_output = output.clone();
        auto sorted_expected = expected.clone();

        sorted_output = std::get<0>(torch::sort(output, 1));
        sorted_expected = std::get<0>(torch::sort(expected, 1));

        return verbose_allclose(sorted_output, sorted_expected, rtol, atol);
    } else if (mode == CheckerMode::kJustDump) {
        // Dump output and expected tensors to file
        {
            fout << "=====OUTPUT=====" << std::endl;
            fout << output.sizes() << std::endl;

            // Manually print the full tensor to avoid truncation
            auto sizes = output.sizes();
            if (sizes.size() == 2) {
                // For 2D tensors (matrices)
                for (int64_t i = 0; i < sizes[0]; i++) {
                    for (int64_t j = 0; j < sizes[1]; j++) {
                        fout << std::setw(12) << std::setprecision(6) << output[i][j].item<float>() << " ";
                    }
                    fout << std::endl;
                }
            } else {
                // Fallback for other tensor dimensions
                fout << output << std::endl;
            }
        }

        {
            fout << "=====EXPECTED=====" << std::endl;
            fout << expected.sizes() << std::endl;

            // Manually print the full tensor to avoid truncation
            auto sizes = output.sizes();
            if (sizes.size() == 2) {
                // For 2D tensors (matrices)
                for (int64_t i = 0; i < sizes[0]; i++) {
                    for (int64_t j = 0; j < sizes[1]; j++) {
                        fout << std::setw(12) << std::setprecision(6) << expected[i][j].item<float>() << " ";
                    }
                    fout << std::endl;
                }
            } else {
                // Fallback for other tensor dimensions
                fout << output << std::endl;
            }
        }

        return {true, ""};
    }
    return verbose_allclose(output, expected, rtol, atol);
}

constexpr int BENCHMARK_ITERS = 5;

void preload() {
void *handle_rocblas = dlopen("/usr/local/lib/python3.10/dist-packages/torch/lib/librocblas.so", RTLD_NOW | RTLD_GLOBAL);
void *handle_hipblas = dlopen("/usr/local/lib/python3.10/dist-packages/torch/lib/libhipblas.so", RTLD_NOW | RTLD_GLOBAL);
void *handle_hipblaslt = dlopen("/usr/local/lib/python3.10/dist-packages/torch/lib/libhipblaslt.so", RTLD_NOW | RTLD_GLOBAL);

if (!handle_rocblas || !handle_hipblas || !handle_hipblaslt) {
    fprintf(stderr, "Failed to load required libraries: %s\n", dlerror());
    exit(1);
}
}

int main(int argc, char **argv) {
    // preload();
    // bool benchmark = false;
    bool benchmark = true;
    bool profile_mode = false;
    int target_test_case = -1;
    int target_sub_case = -1;
    int opt;

    while ((opt = getopt(argc, argv, "bpt:c:")) != -1) {
        switch (opt) {
        case 'b':
            benchmark = false;
            break;
        case 'p':
            profile_mode = true;
            break;
        case 't':
            target_sub_case = std::stoi(optarg);
            break;
        case 'c':
            target_test_case = std::stoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-b] [-p] [-t subcase_index] [-c test_case_index]\n", argv[0]);
            fprintf(stderr, "  -b: Disable benchmark mode\n");
            fprintf(stderr, "  -p: Enable profile mode (skips reference kernel and comparison)\n");
            fprintf(stderr, "  -t: Run only the specified subcase index\n");
            fprintf(stderr, "  -c: Run only the specified test case index\n");
            exit(EXIT_FAILURE);
        }
    }

    case_initialize();
    int num_params, passed_cases = 0;
    num_params = get_params_count();

    // Validate test case index if specified
    if (target_test_case >= 0) {
        if (target_test_case >= num_params) {
            std::cerr << "Error: Test case index " << target_test_case << " is out of range (0-" << (num_params - 1)
                      << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::vector<std::vector<PerfMetrics>> run_times(num_params);
    std::vector<std::tuple<bool, std::string, std::vector<std::pair<float, float>>>> results;

    // If targeting specific test case and subcase, run multiple times and output only the best time
    if (target_test_case >= 0 && target_sub_case >= 0) {
        void *input = case_get_input(target_test_case);
        std::vector<Checkee> output;
        float best_time = std::numeric_limits<float>::max();

        for (int j = 0; j < BENCHMARK_ITERS; j++) {
            PerfMetrics metrics;
            output = case_run_kernel(input, &metrics);

            if (metrics.count <= target_sub_case) {
                std::cerr << "Error: Subcase index " << target_sub_case << " is out of range (0-" << (metrics.count - 1)
                          << ")" << std::endl;
                exit(EXIT_FAILURE);
            }

            best_time = std::min(best_time, metrics.entries[target_sub_case].time);
        }

        std::cout << std::fixed << std::setprecision(6) << best_time * 1e3 << std::endl;
        case_destroy(input);
        return 0;
    }

    // Normal execution path
    if (!profile_mode && target_test_case < 0) {
        std::cout << "Found " << num_params << " test cases for " << case_get_name() << '\n';
    }
    if (benchmark) {
        std::cout << "Benchmark mode enabled\n";
    }
    if (profile_mode) {
        std::cout << "Profile mode enabled (skipping reference kernels and comparison)\n";
    }

    // Determine which test cases to run
    std::vector<int> test_cases_to_run;
    if (target_test_case >= 0) {
        test_cases_to_run.push_back(target_test_case);
    } else {
        for (int i = 0; i < num_params; i++) {
            test_cases_to_run.push_back(i);
        }
    }

    for (int i : test_cases_to_run) {
        std::ofstream *fout = nullptr;
        void *input = case_get_input(i);
        if (!profile_mode && target_test_case < 0) {
            std::cerr << "Running test case " << i << std::flush;
        }
        std::vector<Checkee> reference;
        if (!profile_mode) {
            reference = case_run_ref_kernel(input);
        }
        std::vector<Checkee> output;
        for (int j = 0; j < (benchmark ? BENCHMARK_ITERS : 1); j++) {
            PerfMetrics metrics;
            output = case_run_kernel(input, &metrics);
            run_times[i].push_back(metrics);
        }

        bool match = true;
        std::string case_message;

        if (!profile_mode) {
            if (reference.size() != output.size()) {
                std::cerr << "Wrong test definition: reference and output have different sizes" << '\n';
                abort();
            }

            for (int j = 0; j < reference.size(); j++) {
                float rtol, atol;
                get_error_tolerance(&rtol, &atol);
                if (output[j].mode == CheckerMode::kJustDump) {
                    if (!fout) {
                        fout = new std::ofstream(std::string("case_") + std::to_string(i) + ".txt");
                    }
                    *fout << "===== SUBCASE " << output[j].name << "=====" << std::endl;
                }
                auto [match_sub, message_sub] =
                    check_implementation(*fout, *output[j].tensor, *reference[j].tensor, rtol, atol, output[j].mode);
                if (!match_sub) {
                    case_message += "Err on sub case " + std::to_string(j) + ": " + message_sub + "\n";
                    match = false;
                }
            }
            if (match) {
                passed_cases++;
            }
        } else {
            match = true;
            passed_cases++;
        }

        std::vector<std::pair<float, float>> case_metrics;

        // Process metrics for each run
        for (const auto &run : run_times[i]) {
            if (run.count == 1) {
                // Backward compatibility: single metric case
                case_metrics.push_back({run.entries[0].time, run.entries[0].gflops});
            } else {
                // Multiple metrics case - first entry is the total result
                case_metrics.push_back({run.entries[0].time, run.entries[0].gflops});
            }
        }

        results.push_back(std::make_tuple(match, case_message, case_metrics));
        case_destroy(input);
        if (!profile_mode && target_test_case < 0) {
            std::cout << "\033[2K\r" << std::flush;
        }
    }

    // Only show detailed output if not in single test case mode
    if (target_test_case < 0) {
        std::cout << "=======================" << '\n';
        if (!profile_mode) {
            if (passed_cases == num_params) {
                std::cout << "✅ All " << num_params << " test cases passed!" << '\n';
            } else {
                std::cout << "❌ [" << num_params - passed_cases << "/" << num_params << "] test cases failed!" << '\n';
            }
        } else {
            std::cout << "Profile mode: results comparison skipped" << '\n';
        }
        std::cout << "-----------------------" << '\n';

        for (int i = 0; i < num_params; i++) {
            auto [match, message, metrics] = results[i];

            // Calculate best and worst metrics
            float best_time = std::numeric_limits<float>::max();
            float best_gflops = 0.0f;
            float worst_time = 0.0f;
            float worst_gflops = std::numeric_limits<float>::max();

            for (const auto &[time, gflops] : metrics) {
                best_time = std::min(best_time, time);
                best_gflops = std::max(best_gflops, gflops);
                worst_time = std::max(worst_time, time);
                worst_gflops = std::min(worst_gflops, gflops);
            }

            std::string timing_info;
            if (benchmark) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2);
                ss << "Best: [\033[1m" << best_time * 1e3 << "\033[0m us, \033[1m" << best_gflops / 1e3
                   << "\033[0m TFLOPS], "
                   << "\033[2mSlowest: [" << worst_time * 1e3 << " us, " << worst_gflops / 1e3 << " TFLOPS]\033[0m";
                timing_info = ss.str();
            } else {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2);
                ss << "Time: " << best_time * 1e3 << " us, TFLOPS: " << best_gflops / 1e3;
                timing_info = ss.str();
            }

            if (!profile_mode && !match) {
                std::cout << "❌ Test case " << i << ": " << timing_info << "\n" << message << '\n';
            } else {
                std::cout << "✅ Test case " << i << ": " << timing_info << "\n";
            }

            // Print sub-results if there are multiple metrics
            if (run_times[i][0].count > 1) {
                for (int j = 1; j < run_times[i][0].count; j++) {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2);
                    ss << "    - Sub-case " << run_times[i][0].entries[j].name << ": ";

                    if (benchmark) {
                        float sub_best_time = std::numeric_limits<float>::max();
                        float sub_best_gflops = 0.0f;
                        float sub_worst_time = 0.0f;
                        float sub_worst_gflops = std::numeric_limits<float>::max();

                        for (const auto &run : run_times[i]) {
                            sub_best_time = std::min(sub_best_time, run.entries[j].time);
                            sub_best_gflops = std::max(sub_best_gflops, run.entries[j].gflops);
                            sub_worst_time = std::max(sub_worst_time, run.entries[j].time);
                            sub_worst_gflops = std::min(sub_worst_gflops, run.entries[j].gflops);
                        }

                        ss << "Best: [\033[1m" << sub_best_time * 1e3 << "\033[0m us, \033[1m" << sub_best_gflops / 1e3
                           << "\033[0m TFLOPS], "
                           << "\033[2mSlowest: [" << sub_worst_time * 1e3 << " us, " << sub_worst_gflops / 1e3
                           << " TFLOPS]\033[0m";
                    } else {
                        ss << "Time: " << run_times[i][0].entries[j].time * 1e3
                           << " us, TFLOPS: " << run_times[i][0].entries[j].gflops / 1e3;
                    }

                    std::cout << ss.str() << std::endl;
                }
            }
        }
        std::cout << "-----------------------" << '\n';

        // Calculate geometric mean of time and GFLOPS
        double geo_mean_time = 1.0;
        double geo_mean_gflops = 1.0;

        for (int i = 0; i < num_params; i++) {
            auto [match, message, metrics] = results[i];
            // Always use the best performance metrics for geometric mean
            float best_time = std::numeric_limits<float>::max();
            float best_gflops = 0.0f;

            for (const auto &[time, gflops] : metrics) {
                best_time = std::min(best_time, time);
                best_gflops = std::max(best_gflops, gflops);
            }

            geo_mean_time *= best_time;
            geo_mean_gflops *= best_gflops;
        }

        geo_mean_time = std::pow(geo_mean_time, 1.0 / num_params);
        geo_mean_gflops = std::pow(geo_mean_gflops, 1.0 / num_params);

        if (benchmark) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << "GeoMean - Best Time: \033[1m" << geo_mean_time * 1e3 << "\033[0m us, Best TFLOPS: \033[1m"
               << geo_mean_gflops / 1e3 << "\033[0m";
            std::cout << ss.str() << std::endl;
        } else {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << "GeoMean - Time: " << geo_mean_time * 1e3 << " us, TFLOPS: " << geo_mean_gflops / 1e3;
            std::cout << ss.str() << std::endl;
        }
        std::cout << "=======================" << '\n';
    }

    return 0;
}
