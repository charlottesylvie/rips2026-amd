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
    hipStream_t stream) {
  using namespace bf_csr_detail;

  validate_device_csr_adjacency(d_adjacency, source);

  if (max_iters < 0) {
    max_iters = static_cast<int>(d_adjacency.rows) - 1;
  }

  DeviceCsrOwner d_dist =
      make_device_single_source_vector(d_adjacency.rows, source, stream);

  BellmanFordCsrResult result;

  for (int iter = 0; iter < max_iters; ++iter) {
    DeviceCsrOwner d_relaxed =
        sparse_minplus_relax(d_adjacency, d_dist.view, stream);

    int changed = 0;
    DeviceCsrOwner d_next =
        merge_distances_on_device(d_dist.view, d_relaxed.view, &changed, stream);

    d_dist = std::move(d_next);
    result.iterations_used = iter + 1;

    if (!changed) {
      result.converged = true;
      break;
    }
  }

  result.dist = copy_dist_vector_to_host(d_dist.view, stream);
  return result;
}

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr(d_adjacency,
                                      source,
                                      static_cast<int>(d_adjacency.rows) - 1,
                                      stream);
}

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    hipStream_t stream) {
  using namespace bf_csr_detail;

  validate_host_csr_adjacency(adjacency, source);
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
  return bellman_ford_minplus_hip_csr(d_adjacency.view,
                                      source,
                                      max_iters,
                                      stream);
}

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_csr(adjacency,
                                      source,
                                      static_cast<int>(adjacency.rows) - 1,
                                      stream);
}
