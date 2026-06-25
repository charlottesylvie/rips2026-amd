#include "../minplus_mm/minplus_hip.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Distance value used for unreachable nodes and missing edges.
constexpr float INF = std::numeric_limits<float>::infinity();

// Converts HIP errors into C++ exceptions with context.
void check_hip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(status));
  }
}

// Applies dist[v] = min(dist[v], relaxed[v]) on the GPU and records if anything changed.
__global__ void relax_dist_kernel(float* dist,
                                  const float* relaxed,
                                  int n,
                                  int* changed) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }

  const float old_dist = dist[v];
  const float new_dist = relaxed[v] < old_dist ? relaxed[v] : old_dist;

  if (new_dist < old_dist) {
    dist[v] = new_dist;
    atomicExch(changed, 1);
  }
}

// Checks whether one more relaxation could improve a distance, which signals a negative cycle.
__global__ void detect_change_kernel(const float* dist,
                                     const float* relaxed,
                                     int n,
                                     int* changed) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }

  if (relaxed[v] < dist[v]) {
    atomicExch(changed, 1);
  }
}

} 

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
                                           hipStream_t stream) {
  // Validate the graph size, source node, and iteration limit before using the GPU.
  if (d_adjacency == nullptr) {
    throw std::invalid_argument("d_adjacency must not be null");
  }
  if (n <= 0) {
    throw std::invalid_argument("n must be positive");
  }
  if (source < 0 || source >= n) {
    throw std::invalid_argument("source is outside graph");
  }
  if (max_iters < 0) {
    max_iters = n - 1;
  }

  const std::size_t vector_bytes = static_cast<std::size_t>(n) * sizeof(float);

  // Allocate GPU buffers for the current distances, relaxed candidates, and changed flag.
  float* d_dist = nullptr;
  float* d_relaxed = nullptr;
  int* d_changed = nullptr;

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_dist), vector_bytes),
            "hipMalloc dist");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_relaxed), vector_bytes),
            "hipMalloc relaxed");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_changed), sizeof(int)),
            "hipMalloc changed");

  // Initialize distances to INF except the source, which starts at distance 0.
  std::vector<float> initial_dist(n, INF);
  initial_dist[source] = 0.0f;
  check_hip(hipMemcpyAsync(d_dist,
                           initial_dist.data(),
                           vector_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy initial distances to device");

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  BellmanFordResult result;

  // Repeat min-plus relaxation up to n-1 times, stopping early if no distance changes.
  for (int iter = 0; iter < max_iters; ++iter) {
    int changed = 0;
    check_hip(hipMemcpyAsync(d_changed,
                             &changed,
                             sizeof(int),
                             hipMemcpyHostToDevice,
                             stream),
              "reset changed flag");

    // Compute relaxed = A (*) dist, treating dist as an n x 1 min-plus matrix.
    check_hip(minplus_gemm_f32(d_adjacency,
                               d_dist,
                               d_relaxed,
                               n,
                               1,
                               n,
                               n,
                               1,
                               1,
                               stream),
              "min-plus relaxation");

    // Merge the relaxed candidates into dist and mark whether any node improved.
    hipLaunchKernelGGL(relax_dist_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       d_dist,
                       d_relaxed,
                       n,
                       d_changed);
    check_hip(hipGetLastError(), "launch distance relaxation kernel");

    // Copy back only the small changed flag so the host can decide whether to stop.
    check_hip(hipMemcpyAsync(&changed,
                             d_changed,
                             sizeof(int),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy changed flag to host");
    check_hip(hipStreamSynchronize(stream), "synchronize Bellman-Ford iteration");

    result.iterations_used = iter + 1;
    if (!changed) {
      result.converged = true;
      break;
    }
  }

  // Run one extra relaxation pass to detect whether a reachable negative cycle exists.
  int changed = 0;
  check_hip(hipMemcpyAsync(d_changed,
                           &changed,
                           sizeof(int),
                           hipMemcpyHostToDevice,
                           stream),
            "reset negative-cycle flag");
  check_hip(minplus_gemm_f32(d_adjacency,
                             d_dist,
                             d_relaxed,
                             n,
                             1,
                             n,
                             n,
                             1,
                             1,
                             stream),
            "negative-cycle relaxation");

  // Compare the extra relaxed vector against the final distances without modifying dist.
  hipLaunchKernelGGL(detect_change_kernel,
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     d_dist,
                     d_relaxed,
                     n,
                     d_changed);
  check_hip(hipGetLastError(), "launch negative-cycle detection kernel");

  check_hip(hipMemcpyAsync(&changed,
                           d_changed,
                           sizeof(int),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy negative-cycle flag to host");
  check_hip(hipStreamSynchronize(stream), "synchronize negative-cycle check");
  result.has_negative_cycle = changed != 0;

  // Copy final distances back to the CPU for the caller.
  result.dist.resize(n);
  check_hip(hipMemcpyAsync(result.dist.data(),
                           d_dist,
                           vector_bytes,
                           hipMemcpyDeviceToHost,
                           stream),
            "copy final distances to host");
  check_hip(hipStreamSynchronize(stream), "synchronize final distance copy");

  // Release temporary GPU buffers owned by this function.
  check_hip(hipFree(d_dist), "hipFree dist");
  check_hip(hipFree(d_relaxed), "hipFree relaxed");
  check_hip(hipFree(d_changed), "hipFree changed");

  return result;
}

// Convenience overload for device adjacency with default n-1 Bellman-Ford iterations.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source,
                                           hipStream_t stream) {
  return bellman_ford_minplus_hip(d_adjacency, n, source, n - 1, stream);
}

// Convenience overload for device adjacency using the default HIP stream.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source,
                                           int max_iters) {
  return bellman_ford_minplus_hip(d_adjacency, n, source, max_iters, nullptr);
}

// Convenience overload for device adjacency with default iterations and default stream.
BellmanFordResult bellman_ford_minplus_hip(const float* d_adjacency,
                                           int n,
                                           int source) {
  return bellman_ford_minplus_hip(d_adjacency, n, source, n - 1, nullptr);
}

// Copies a host row-major adjacency matrix to the GPU, then runs the device-pointer version.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           int max_iters,
                                           hipStream_t stream) {
  // Check that the host matrix is exactly n x n before copying it to the GPU.
  if (n <= 0) {
    throw std::invalid_argument("n must be positive");
  }

  const std::size_t expected_size =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (adjacency.size() != expected_size) {
    throw std::invalid_argument("adjacency must contain n*n row-major values");
  }

  float* d_adjacency = nullptr;
  const std::size_t matrix_bytes = expected_size * sizeof(float);

  // Allocate device storage for the adjacency matrix and copy the host matrix once.
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_adjacency), matrix_bytes),
            "hipMalloc adjacency");
  check_hip(hipMemcpyAsync(d_adjacency,
                           adjacency.data(),
                           matrix_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy adjacency to device");

  // Reuse the main GPU implementation after the adjacency matrix is on device.
  BellmanFordResult result =
      bellman_ford_minplus_hip(d_adjacency, n, source, max_iters, stream);

  // Release the copied adjacency matrix before returning the result.
  check_hip(hipFree(d_adjacency), "hipFree adjacency");
  return result;
}

// Convenience overload for host adjacency with default n-1 Bellman-Ford iterations.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           hipStream_t stream) {
  return bellman_ford_minplus_hip(adjacency, n, source, n - 1, stream);
}

// Convenience overload for host adjacency using the default HIP stream.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source,
                                           int max_iters) {
  return bellman_ford_minplus_hip(adjacency, n, source, max_iters, nullptr);
}

// Convenience overload for host adjacency with default iterations and default stream.
BellmanFordResult bellman_ford_minplus_hip(const std::vector<float>& adjacency,
                                           int n,
                                           int source) {
  return bellman_ford_minplus_hip(adjacency, n, source, n - 1, nullptr);
}
