// Dense Gauss-Seidel-style Bellman-Ford implementation.
//
// This file is a library source; it does not have its own main().
// Compile it by linking it with a driver from the repository root:
//
//   hipcc -std=c++17 -O3 -x hip \
//     <your_driver.cpp> \
//     HIP_kernel/bellman_ford/src/bf_gauss_seidel.cpp \
//     -o <your_program>
//
// Run the resulting driver directly, for example:
//
//   ./<your_program>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

void check_hip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             hipGetErrorString(status));
  }
}

// Performs one ordered Gauss-Seidel sweep over the dense incoming-edge matrix.
// For target v, distances from nodes u < v may already be updated in d_dist,
// while distances from nodes u > v are read from d_prev.
__global__ void gauss_seidel_sweep_kernel(const float* adjacency,
                                          const float* prev,
                                          float* dist,
                                          int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int v = 0; v < n; ++v) {
    float best = prev[v];

    for (int u = 0; u < v; ++u) {
      const float weight = adjacency[v * n + u];
      const float candidate = dist[u] + weight;
      best = candidate < best ? candidate : best;
    }

    for (int u = v + 1; u < n; ++u) {
      const float weight = adjacency[v * n + u];
      const float candidate = prev[u] + weight;
      best = candidate < best ? candidate : best;
    }

    dist[v] = best;
  }
}

}  // namespace

struct BellmanFordGaussSeidelResult {
  std::vector<float> dist;
  int iterations_used = 0;
  bool converged = false;
  bool has_negative_cycle = false;
};

// Runs fixed-iteration min-plus Gauss-Seidel relaxation using an
// already-GPU-resident dense incoming-edge adjacency matrix:
// adjacency[v * n + u] = weight of edge u -> v.
BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const float* d_adjacency,
    int n,
    int source,
    int max_iters,
    hipStream_t stream) {
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

  float* d_dist = nullptr;
  float* d_prev = nullptr;

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_dist), vector_bytes),
            "hipMalloc dist");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_prev), vector_bytes),
            "hipMalloc prev");

  std::vector<float> initial_dist(n, INF);
  initial_dist[source] = 0.0f;
  check_hip(hipMemcpyAsync(d_dist,
                           initial_dist.data(),
                           vector_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy initial distances to device");

  BellmanFordGaussSeidelResult result;

  for (int iter = 0; iter < max_iters; ++iter) {
    check_hip(hipMemcpyAsync(d_prev,
                             d_dist,
                             vector_bytes,
                             hipMemcpyDeviceToDevice,
                             stream),
              "copy previous distances");

    hipLaunchKernelGGL(gauss_seidel_sweep_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       d_adjacency,
                       d_prev,
                       d_dist,
                       n);
    check_hip(hipGetLastError(), "launch Gauss-Seidel sweep kernel");

    result.iterations_used = iter + 1;
  }

  result.dist.resize(n);
  check_hip(hipMemcpyAsync(result.dist.data(),
                           d_dist,
                           vector_bytes,
                           hipMemcpyDeviceToHost,
                           stream),
            "copy final distances to host");
  check_hip(hipStreamSynchronize(stream), "synchronize final distance copy");

  check_hip(hipFree(d_dist), "hipFree dist");
  check_hip(hipFree(d_prev), "hipFree prev");

  return result;
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const float* d_adjacency,
    int n,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_gauss_seidel(d_adjacency,
                                               n,
                                               source,
                                               n - 1,
                                               stream);
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const float* d_adjacency,
    int n,
    int source,
    int max_iters) {
  return bellman_ford_minplus_hip_gauss_seidel(d_adjacency,
                                               n,
                                               source,
                                               max_iters,
                                               nullptr);
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const float* d_adjacency,
    int n,
    int source) {
  return bellman_ford_minplus_hip_gauss_seidel(d_adjacency,
                                               n,
                                               source,
                                               n - 1,
                                               nullptr);
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const std::vector<float>& adjacency,
    int n,
    int source,
    int max_iters,
    hipStream_t stream) {
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

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_adjacency), matrix_bytes),
            "hipMalloc adjacency");
  check_hip(hipMemcpyAsync(d_adjacency,
                           adjacency.data(),
                           matrix_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy adjacency to device");

  BellmanFordGaussSeidelResult result =
      bellman_ford_minplus_hip_gauss_seidel(d_adjacency,
                                            n,
                                            source,
                                            max_iters,
                                            stream);

  check_hip(hipFree(d_adjacency), "hipFree adjacency");
  return result;
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const std::vector<float>& adjacency,
    int n,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_gauss_seidel(adjacency,
                                               n,
                                               source,
                                               n - 1,
                                               stream);
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const std::vector<float>& adjacency,
    int n,
    int source,
    int max_iters) {
  return bellman_ford_minplus_hip_gauss_seidel(adjacency,
                                               n,
                                               source,
                                               max_iters,
                                               nullptr);
}

BellmanFordGaussSeidelResult bellman_ford_minplus_hip_gauss_seidel(
    const std::vector<float>& adjacency,
    int n,
    int source) {
  return bellman_ford_minplus_hip_gauss_seidel(adjacency,
                                               n,
                                               source,
                                               n - 1,
                                               nullptr);
}
