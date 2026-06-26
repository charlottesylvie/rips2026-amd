#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <vector>

struct BellmanFordCsrNoChecksResult {
  std::vector<float> dist;
  int iterations_used = 0;
};

// Runs Bellman-Ford using a GPU-resident CSR min-plus adjacency matrix for
// exactly max_iters iterations. If max_iters is negative, this runs n - 1
// iterations. It does not stop early when distances stop changing.
BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    hipStream_t stream = nullptr);

BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    hipStream_t stream = nullptr);

// Convenience overload that copies host CSR arrays to the GPU first.
BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    hipStream_t stream = nullptr);

BellmanFordCsrNoChecksResult bellman_ford_minplus_hip_csr_no_checks(
    const HostCsrF32& adjacency,
    int source,
    hipStream_t stream = nullptr);
