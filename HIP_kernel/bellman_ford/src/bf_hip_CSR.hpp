#pragma once

#include "../../minplus_mm/src/minplus_sparse_hip.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <vector>

struct BellmanFordCsrProgress {
  int iteration = 0;
  int max_iters = 0;
  bool convergence_checked = false;
  bool changed = false;
};

using BellmanFordCsrProgressCallback =
    void (*)(const BellmanFordCsrProgress& progress, void* user_data);

struct BellmanFordCsrResult {
  std::vector<float> dist;
  std::vector<int> pred_node;
  std::vector<minplus_sparse::Offset> pred_edge;
  int iterations_used = 0;
  bool converged = false;
  int target = -1;
  float target_distance = std::numeric_limits<float>::infinity();
  bool target_reached = false;
  bool stopped_on_target = false;
};

struct HostCsrF32 {
  minplus_sparse::Offset rows = 0;
  minplus_sparse::Offset cols = 0;
  minplus_sparse::Offset nnz = 0;
  std::vector<minplus_sparse::Offset> rowptr;
  std::vector<minplus_sparse::Index> colind;
  std::vector<float> values;
};

// Runs Bellman-Ford using a GPU-resident CSR min-plus adjacency matrix.
//
// Matrix convention:
//   adjacency row v, column u = weight of edge u -> v
//
// This matches the dense implementation's incoming-edge orientation:
//   dense_adjacency[v * n + u] = weight of edge u -> v
BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int max_iters,
    hipStream_t stream = nullptr,
    BellmanFordCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    hipStream_t stream = nullptr);

// Convenience overload that copies host CSR arrays to the GPU first.
BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int max_iters,
    hipStream_t stream = nullptr,
    BellmanFordCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

BellmanFordCsrResult bellman_ford_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    hipStream_t stream = nullptr);
