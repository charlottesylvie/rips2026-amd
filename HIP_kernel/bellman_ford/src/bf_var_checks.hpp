#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <vector>

struct BellmanFordCsrVarChecksResult {
  std::vector<float> dist;
  int iterations_used = 0;
  bool converged = false;
  int check_interval = 0;
  int checks_performed = 0;
  int changed_checks = 0;
  int unchanged_checks = 0;
  double initialization_ms = 0.0;
  double bellman_ford_loop_ms = 0.0;
  double relaxation_ms = 0.0;
  double checked_merge_ms = 0.0;
  double unchecked_merge_ms = 0.0;
  double checked_iteration_ms = 0.0;
  double unchecked_iteration_ms = 0.0;
  double final_copy_ms = 0.0;
  double total_runtime_ms = 0.0;
  double estimated_check_overhead_ms = 0.0;
};

// Runs Bellman-Ford using a GPU-resident CSR min-plus adjacency matrix and
// checks convergence every check_interval iterations. For example, a
// check_interval of 10 checks iterations 10, 20, 30, ...
BellmanFordCsrVarChecksResult
bellman_ford_minplus_hip_csr_var_checks_with_stats(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream = nullptr,
    BellmanFordCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream = nullptr,
    BellmanFordCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int check_interval,
    hipStream_t stream = nullptr);

// Convenience overload that copies host CSR arrays to the GPU first.
BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream = nullptr,
    BellmanFordCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const HostCsrF32& adjacency,
    int source,
    int check_interval,
    hipStream_t stream = nullptr);
