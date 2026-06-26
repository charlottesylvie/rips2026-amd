// CSR min-plus Bellman-Ford implementation with convergence checking.
//
// This file is a library source; it does not have its own main().
// Compile it by linking it with a driver or benchmark from the repository root:
//
//   hipcc -std=c++17 -O3 -x hip \
//     <your_driver.cpp> \
//     HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
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

#include "bf_hip_CSR.hpp"
#include "bf_hip_CSR_device_utils.hpp"

#include <hip/hip_runtime.h>

#include <utility>

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
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

  // The result will eventually hold the final distances, how many iterations
  // were used, and whether the algorithm stopped early.
  BellmanFordCsrResult result;

  // Main Bellman-Ford loop. This version checks for convergence after every
  // iteration and may stop before max_iters.
  for (int iter = 0; iter < max_iters; ++iter) {
    // Relax all distances once using sparse min-plus matrix multiplication:
    //   d_relaxed = adjacency min-plus d_dist
    DeviceCsrOwner d_relaxed =
        sparse_minplus_relax(d_adjacency, d_dist.view, stream);

    // Merge old and relaxed distances on the GPU:
    //   d_next[v] = min(d_dist[v], d_relaxed[v])
    //
    // Because this checked version passes &changed, the merge also copies back
    // one small flag telling us whether any distance improved.
    int changed = 0;
    DeviceCsrOwner d_next =
        merge_distances_on_device(d_dist.view, d_relaxed.view, &changed, stream);

    // Replace the old GPU distance vector with the newly merged one.
    d_dist = std::move(d_next);
    result.iterations_used = iter + 1;

    // Optional callback for long-running benchmarks. It lets a caller print
    // progress without this library file knowing anything about stdout.
    if (progress_callback) {
      BellmanFordCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_iters;
      progress.convergence_checked = true;
      progress.changed = changed != 0;
      progress_callback(progress, progress_user_data);
    }

    // If no distance changed, Bellman-Ford has converged and we can stop early.
    if (!changed) {
      result.converged = true;
      break;
    }
  }

  // Copy the final sparse GPU distance vector back to a dense CPU vector so
  // the caller can inspect or compare the answer.
  result.dist = copy_dist_vector_to_host(d_dist.view, stream);
  return result;
}

// Convenience overload: GPU graph input, default n - 1 iteration limit.
BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr(d_adjacency,
                                      source,
                                      static_cast<int>(d_adjacency.rows) - 1,
                                      stream,
                                      nullptr,
                                      nullptr);
}

// Convenience overload: CPU graph input. This copies the graph to the GPU once,
// then calls the main GPU implementation above.
BellmanFordCsrResult bellman_ford_minplus_hip_csr(
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
  return bellman_ford_minplus_hip_csr(d_adjacency.view,
                                      source,
                                      max_iters,
                                      stream,
                                      progress_callback,
                                      progress_user_data);
}

// Convenience overload: CPU graph input, default n - 1 iteration limit.
BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr(adjacency,
                                      source,
                                      static_cast<int>(adjacency.rows) - 1,
                                      stream,
                                      nullptr,
                                      nullptr);
}
