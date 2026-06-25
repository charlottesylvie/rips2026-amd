#pragma once

#include <hip/hip_runtime.h>

#include <vector>

// Host-side result returned after the GPU Bellman-Ford run finishes.
struct BellmanFordResult {
  std::vector<float> dist;
  int iterations_used = 0;
  bool converged = false;
  bool has_negative_cycle = false;
};

// Runs Bellman-Ford using an already-GPU-resident dense min-plus adjacency matrix.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source,
                                           int max_iters,
                                           hipStream_t stream);

// Convenience overload for device adjacency with default n-1 Bellman-Ford iterations.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source,
                                           hipStream_t stream);

// Convenience overload for device adjacency using the default HIP stream.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source,
                                           int max_iters);

// Convenience overload for device adjacency with default iterations and default stream.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source);

// Copies a host row-major adjacency matrix to the GPU, then runs the device-pointer version.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           int max_iters,
                                           hipStream_t stream);

// Convenience overload for host adjacency with default n-1 Bellman-Ford iterations.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           hipStream_t stream);

// Convenience overload for host adjacency using the default HIP stream.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           int max_iters);

// Convenience overload for host adjacency with default iterations and default stream.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source);
