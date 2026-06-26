// CSR min-plus Bellman-Ford implementation without convergence checking.
//
// This file is a library source; it does not have its own main().
// Compile it by linking it with a driver or benchmark from the repository root:
//
//   hipcc -std=c++17 -O3 -x hip \
//     <your_driver.cpp> \
//     HIP_kernel/bellman_ford/src/bf_hip_no_checks_CSR.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o <your_program>
//
// Example benchmark with both CSR implementations:
//
//   hipcc -std=c++17 -O3 -x hip \
//     HIP_kernel/bellman_ford/tests/big_test.cpp \
//     HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
//     HIP_kernel/bellman_ford/src/bf_hip_no_checks_CSR.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o big_test
//   ./big_test USA-road-d.BAY.csrbin 1 50

#include "bf_hip_no_checks_CSR.hpp"
#include "bf_hip_CSR_device_utils.hpp"

#include <hip/hip_runtime.h>

#include <utility>

BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace bf_csr_detail;

  // Validate the GPU-resident CSR graph and source node before launching any
  // GPU kernels.
  validate_device_csr_adjacency(d_adjacency, source);

  // A negative max_iters means "use the standard Bellman-Ford limit", which is
  // n - 1 relaxation rounds for a graph with n nodes.
  if (max_iters < 0) {
    max_iters = static_cast<int>(d_adjacency.rows) - 1;
  }

  // Create the starting distance vector on the GPU:
  //   source node distance = 0
  //   every other node = infinity, represented by absence from sparse CSR
  DeviceCsrOwner d_dist =
      make_device_single_source_vector(d_adjacency.rows, source, stream);

  // The result will eventually hold the final distances and the number of
  // iterations that were run. This version never stops early.
  BellmanFordCsrNoChecksResult result;

  // Main Bellman-Ford loop. This version intentionally runs every requested
  // iteration and never asks whether the distances have stopped changing.
  for (int iter = 0; iter < max_iters; ++iter) {
    // Relax all distances once using sparse min-plus matrix multiplication:
    //   d_relaxed = adjacency min-plus d_dist
    DeviceCsrOwner d_relaxed =
        sparse_minplus_relax(d_adjacency, d_dist.view, stream);

    // Merge old and relaxed distances on the GPU:
    //   d_next[v] = min(d_dist[v], d_relaxed[v])
    //
    // Passing nullptr means no convergence flag is copied back to the CPU.
    DeviceCsrOwner d_next =
        merge_distances_on_device(d_dist.view, d_relaxed.view, nullptr, stream);

    // Replace the old GPU distance vector with the newly merged one.
    d_dist = std::move(d_next);
    result.iterations_used = iter + 1;

    // Optional callback for long-running benchmarks. Here convergence_checked
    // is always false because this file deliberately avoids convergence checks.
    if (progress_callback) {
      BellmanFordCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_iters;
      progress.convergence_checked = false;
      progress.changed = false;
      progress_callback(progress, progress_user_data);
    }
  }

  // Copy the final sparse GPU distance vector back to a dense CPU vector so
  // the caller can inspect or compare the answer.
  result.dist = copy_dist_vector_to_host(d_dist.view, stream);
  return result;
}

// Convenience overload: GPU graph input, default n - 1 iteration limit.
BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr_no_checks(
      d_adjacency,
      source,
      static_cast<int>(d_adjacency.rows) - 1,
      stream,
      nullptr,
      nullptr);
}

// Convenience overload: CPU graph input. This copies the graph to the GPU once,
// then calls the main GPU implementation above.
BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace bf_csr_detail;

  // Validate the CPU-side CSR arrays before copying them to the GPU.
  validate_host_csr_adjacency(adjacency, source);
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
  return bellman_ford_minplus_hip_csr_no_checks(d_adjacency.view,
                                                source,
                                                max_iters,
                                                stream,
                                                progress_callback,
                                                progress_user_data);
}

// Convenience overload: CPU graph input, default n - 1 iteration limit.
BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const HostCsrF32& adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr_no_checks(
      adjacency,
      source,
      static_cast<int>(adjacency.rows) - 1,
      stream,
      nullptr,
      nullptr);
}
