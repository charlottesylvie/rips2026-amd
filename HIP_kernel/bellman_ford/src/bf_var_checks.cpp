// CSR min-plus Bellman-Ford implementation with variable convergence checks.
//
// This file is a library source; it does not have its own main().
// Compile it by linking it with a driver or benchmark from the repository root:
//
//   hipcc -std=c++17 -O3 -x hip \
//     <your_driver.cpp> \
//     HIP_kernel/bellman_ford/src/bf_var_checks.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o <your_program>
//
// Use bellman_ford_minplus_hip_csr_var_checks(..., check_interval, ...) to
// choose how often the algorithm checks for early convergence. For example,
// check_interval = 10 checks iterations 10, 20, 30, ...

#include "bf_var_checks.hpp"
#include "bf_hip_CSR_device_utils.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <utility>

namespace {

// Make sure the user asked for a sensible convergence-check frequency.
// A value of 10 means "check after iterations 10, 20, 30, ...".
void validate_check_interval(int check_interval) {
  if (check_interval <= 0) {
    throw std::invalid_argument("check_interval must be positive");
  }
}

// Small helper for converting C++ clock timestamps into milliseconds.
double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

BellmanFordCsrVarChecksResult
bellman_ford_minplus_hip_csr_var_checks_with_stats(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace bf_csr_detail;

  // Validate the graph, source node, and check interval before launching GPU
  // work. The graph is expected to already live on the GPU.
  validate_device_csr_adjacency(d_adjacency, source);
  validate_check_interval(check_interval);

  // A negative max_iters means "use the normal Bellman-Ford limit".
  // For a graph with n nodes, that is n - 1 relaxation rounds.
  if (max_iters < 0) {
    max_iters = static_cast<int>(d_adjacency.rows) - 1;
  }

  // This result object stores both the final distances and timing/statistics
  // that the benchmark program prints later.
  BellmanFordCsrVarChecksResult result;
  result.check_interval = check_interval;

  const auto total_start = std::chrono::steady_clock::now();

  // Create the starting distance vector on the GPU:
  //   source node distance = 0
  //   every other node = infinity, represented by absence from sparse CSR
  const auto init_start = std::chrono::steady_clock::now();
  DeviceCsrOwner d_dist =
      make_device_single_source_vector(d_adjacency.rows, source, stream);
  check_hip(hipStreamSynchronize(stream),
            "synchronize variable-check source initialization");
  const auto init_end = std::chrono::steady_clock::now();
  result.initialization_ms = elapsed_ms(init_start, init_end);

  // Main Bellman-Ford loop. Each iteration computes one more min-plus
  // relaxation and merges it into the current best distance vector.
  const auto loop_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < max_iters; ++iter) {
    const int iteration = iter + 1;

    // Only some iterations pay for the convergence check. For example, if
    // check_interval is 10, iterations 10, 20, 30, ... check whether any
    // distance changed.
    const bool check_convergence =
        (iteration % check_interval) == 0;

    // Relax all distances once using sparse min-plus matrix multiplication:
    //   d_relaxed = adjacency min-plus d_dist
    const auto iteration_start = std::chrono::steady_clock::now();
    const auto relaxation_start = std::chrono::steady_clock::now();
    DeviceCsrOwner d_relaxed =
        sparse_minplus_relax(d_adjacency, d_dist.view, stream);
    const auto relaxation_end = std::chrono::steady_clock::now();
    result.relaxation_ms += elapsed_ms(relaxation_start, relaxation_end);

    int changed = 0;

    // Merge old and relaxed distances on the GPU:
    //   d_next[v] = min(d_dist[v], d_relaxed[v])
    //
    // On check iterations, merge_distances_on_device also copies back one
    // small flag telling us whether any distance improved.
    // On non-check iterations, that flag is skipped to avoid CPU/GPU sync.
    const auto merge_start = std::chrono::steady_clock::now();
    DeviceCsrOwner d_next =
        merge_distances_on_device(d_dist.view,
                                  d_relaxed.view,
                                  check_convergence ? &changed : nullptr,
                                  stream);
    const auto merge_end = std::chrono::steady_clock::now();

    // Replace the old GPU distance vector with the newly merged one.
    d_dist = std::move(d_next);
    result.iterations_used = iteration;

    // Store timing separately for checked and unchecked iterations so the test
    // program can estimate how expensive convergence checks are.
    const double merge_ms = elapsed_ms(merge_start, merge_end);
    const double iteration_ms = elapsed_ms(iteration_start, merge_end);

    if (check_convergence) {
      ++result.checks_performed;
      result.checked_merge_ms += merge_ms;
      result.checked_iteration_ms += iteration_ms;
      if (changed) {
        ++result.changed_checks;
      } else {
        ++result.unchanged_checks;
      }
    } else {
      result.unchecked_merge_ms += merge_ms;
      result.unchecked_iteration_ms += iteration_ms;
    }

    // Optional callback for long-running benchmarks. It lets a caller print
    // progress without this library file knowing anything about stdout.
    if (progress_callback) {
      BellmanFordCsrProgress progress;
      progress.iteration = iteration;
      progress.max_iters = max_iters;
      progress.convergence_checked = check_convergence;
      progress.changed = check_convergence ? changed != 0 : false;
      progress_callback(progress, progress_user_data);
    }

    // If this was a check iteration and nothing changed, Bellman-Ford has
    // converged, so we can stop before max_iters.
    if (check_convergence && !changed) {
      result.converged = true;
      break;
    }
  }
  const auto loop_end = std::chrono::steady_clock::now();
  result.bellman_ford_loop_ms = elapsed_ms(loop_start, loop_end);

  // Copy the final sparse GPU distance vector back to a dense CPU vector so
  // the caller can inspect or compare the answer.
  const auto final_copy_start = std::chrono::steady_clock::now();
  result.dist = copy_dist_vector_to_host(d_dist.view, stream);
  const auto final_copy_end = std::chrono::steady_clock::now();
  result.final_copy_ms = elapsed_ms(final_copy_start, final_copy_end);

  // Estimate check overhead by comparing checked merge time to the average
  // unchecked merge time from the same run. This is impossible when every
  // iteration is checked, because then there are no unchecked iterations.
  const int unchecked_iterations =
      result.iterations_used - result.checks_performed;
  if (result.checks_performed > 0 && unchecked_iterations > 0) {
    const double average_unchecked_merge_ms =
        result.unchecked_merge_ms / static_cast<double>(unchecked_iterations);
    result.estimated_check_overhead_ms =
        result.checked_merge_ms -
        average_unchecked_merge_ms *
            static_cast<double>(result.checks_performed);
  }

  const auto total_end = std::chrono::steady_clock::now();
  result.total_runtime_ms = elapsed_ms(total_start, total_end);
  return result;
}

// Simpler wrapper for callers that only want the Bellman-Ford answer and do
// not care about the detailed timing/statistics.
BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  BellmanFordCsrVarChecksResult detailed =
      bellman_ford_minplus_hip_csr_var_checks_with_stats(d_adjacency,
                                                         source,
                                                         max_iters,
                                                         check_interval,
                                                         stream,
                                                         progress_callback,
                                                         progress_user_data);

  BellmanFordCsrResult result;
  result.dist = std::move(detailed.dist);
  result.iterations_used = detailed.iterations_used;
  result.converged = detailed.converged;
  return result;
}

// Convenience overload: GPU graph input, default n - 1 iteration limit.
BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int check_interval,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr_var_checks(
      d_adjacency,
      source,
      static_cast<int>(d_adjacency.rows) - 1,
      check_interval,
      stream,
      nullptr,
      nullptr);
}

// Convenience overload: CPU graph input. This copies the graph to the GPU once,
// then calls the main GPU implementation above.
BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    int check_interval,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace bf_csr_detail;

  validate_host_csr_adjacency(adjacency, source);
  validate_check_interval(check_interval);
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
  return bellman_ford_minplus_hip_csr_var_checks(d_adjacency.view,
                                                 source,
                                                 max_iters,
                                                 check_interval,
                                                 stream,
                                                 progress_callback,
                                                 progress_user_data);
}

// Convenience overload: CPU graph input, default n - 1 iteration limit.
BellmanFordCsrResult bellman_ford_minplus_hip_csr_var_checks(
    const HostCsrF32& adjacency,
    int source,
    int check_interval,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr_var_checks(
      adjacency,
      source,
      static_cast<int>(adjacency.rows) - 1,
      check_interval,
      stream,
      nullptr,
      nullptr);
}
