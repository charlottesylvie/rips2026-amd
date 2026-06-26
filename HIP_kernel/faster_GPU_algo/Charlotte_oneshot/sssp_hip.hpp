#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

// Sparse graph in outgoing CSR orientation:
//   edges from u are in row_offsets[u] .. row_offsets[u + 1] - 1
//   destination = col_indices[e]
//   weight      = weights[e]
// All edge weights must be non-negative for ADDS / delta-stepping.
struct CsrGraph {
  int num_nodes = 0;
  int num_edges = 0;
  std::vector<int> row_offsets;
  std::vector<int> col_indices;
  std::vector<float> weights;
};

struct SSSPOptions {
  // max_iters means:
  //   Bellman-Ford: max relaxation rounds. Use <=0 for n - 1.
  //   ADDS-lite: max bucket advances. Use <=0 for a conservative default.
  int max_iters = -1;

  // ADDS-lite bucket scheduler controls.
  int num_buckets = 32;
  int bucket_capacity_per_bucket = 0;  // <=0 chooses 4*n, at least 4096.
  float initial_delta = 0.0f;          // <=0 chooses a graph-based heuristic.
  bool dynamic_delta = true;

  // Tuning knobs for dynamic_delta. Larger target values make the scheduler
  // coarser and more parallel. Smaller values preserve a tighter priority order.
  int target_low_work = 8192;
  int target_high_work = 262144;
};

struct SSSPResult {
  std::vector<float> dist;
  int iterations_used = 0;
  bool converged = false;
  bool has_negative_cycle = false;  // always false for ADDS; checked for BF.
  unsigned long long work_count = 0;
  float milliseconds = 0.0f;
  float final_delta = 0.0f;
};

// Same user-facing inputs for both algorithms: a sparse CSR graph and source.
// This is the version to benchmark on large sparse FPGA-routing-like graphs.
SSSPResult bellman_ford_sssp_hip(const CsrGraph& graph,
                                 int source,
                                 const SSSPOptions& options = {},
                                 hipStream_t stream = nullptr);

SSSPResult adds_sssp_hip(const CsrGraph& graph,
                         int source,
                         const SSSPOptions& options = {},
                         hipStream_t stream = nullptr);

// Convenience overloads that accept a dense row-major adjacency matrix.
// adjacency[u*n + v] is the weight of edge u -> v; infinity means no edge.
// These overloads exist only to make small correctness tests easy. Do not use
// dense adjacency for large sparse graphs.
SSSPResult bellman_ford_sssp_hip(const std::vector<float>& adjacency,
                                 int n,
                                 int source,
                                 const SSSPOptions& options = {},
                                 hipStream_t stream = nullptr);

SSSPResult adds_sssp_hip(const std::vector<float>& adjacency,
                         int n,
                         int source,
                         const SSSPOptions& options = {},
                         hipStream_t stream = nullptr);

// Helper for tests / input conversion.
CsrGraph dense_to_csr_outgoing(const std::vector<float>& adjacency, int n);
