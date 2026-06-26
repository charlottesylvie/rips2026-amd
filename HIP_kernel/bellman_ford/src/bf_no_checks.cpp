// Dense min-plus Bellman-Ford implementation without convergence checking.
//
// This file is a library source; it does not have its own main().
// Compile it by linking it with a driver from the repository root:
//
//   hipcc -std=c++17 -O3 -x hip \
//     <your_driver.cpp> \
//     HIP_kernel/bellman_ford/src/bf_no_checks.cpp \
//     HIP_kernel/minplus_mm/src/minplus_hip.cpp \
//     -o <your_program>
//
// Run the resulting driver directly, for example:
//
//   ./<your_program>

#include "../../minplus_mm/src/minplus_hip.hpp"

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
    throw std::runtime_error(std::string(message) + ": " +
                             hipGetErrorString(status));
  }
}

// Applies dist[v] = min(dist[v], relaxed[v]) without recording convergence.
__global__ void relax_dist_no_checks_kernel(float* dist,
                                            const float* relaxed,
                                            int n) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }

  const float old_dist = dist[v];
  const float new_dist = relaxed[v] < old_dist ? relaxed[v] : old_dist;
  dist[v] = new_dist;
}

}  // namespace

// Host-side result returned after the fixed-iteration GPU Bellman-Ford run.
struct BellmanFordNoChecksResult {
  std::vector<float> dist;
  int iterations_used = 0;
  bool converged = false;
};

// Runs Bellman-Ford for exactly max_iters rounds using an already-GPU-resident
// dense min-plus adjacency matrix. If max_iters is negative, this runs n - 1
// rounds, which is the standard Bellman-Ford iteration count.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
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
  float* d_relaxed = nullptr;

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_dist), vector_bytes),
            "hipMalloc dist");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_relaxed), vector_bytes),
            "hipMalloc relaxed");

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

  BellmanFordNoChecksResult result;

  for (int iter = 0; iter < max_iters; ++iter) {
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

    hipLaunchKernelGGL(relax_dist_no_checks_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       d_dist,
                       d_relaxed,
                       n);
    check_hip(hipGetLastError(), "launch distance relaxation kernel");

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
  check_hip(hipFree(d_relaxed), "hipFree relaxed");

  return result;
}

// Convenience overload for device adjacency with default n-1 Bellman-Ford
// iterations.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const float* d_adjacency,
    int n,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_no_checks(d_adjacency,
                                            n,
                                            source,
                                            n - 1,
                                            stream);
}

// Convenience overload for device adjacency using the default HIP stream.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const float* d_adjacency,
    int n,
    int source,
    int max_iters) {
  return bellman_ford_minplus_hip_no_checks(d_adjacency,
                                            n,
                                            source,
                                            max_iters,
                                            nullptr);
}

// Convenience overload for device adjacency with default iterations and default
// stream.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const float* d_adjacency,
    int n,
    int source) {
  return bellman_ford_minplus_hip_no_checks(d_adjacency,
                                            n,
                                            source,
                                            n - 1,
                                            nullptr);
}

// Copies a host row-major adjacency matrix to the GPU, then runs the
// fixed-iteration device-pointer version.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
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

  BellmanFordNoChecksResult result =
      bellman_ford_minplus_hip_no_checks(d_adjacency,
                                         n,
                                         source,
                                         max_iters,
                                         stream);

  check_hip(hipFree(d_adjacency), "hipFree adjacency");
  return result;
}

// Convenience overload for host adjacency with default n-1 Bellman-Ford
// iterations.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const std::vector<float>& adjacency,
    int n,
    int source,
    hipStream_t stream) {
  return bellman_ford_minplus_hip_no_checks(adjacency,
                                            n,
                                            source,
                                            n - 1,
                                            stream);
}

// Convenience overload for host adjacency using the default HIP stream.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const std::vector<float>& adjacency,
    int n,
    int source,
    int max_iters) {
  return bellman_ford_minplus_hip_no_checks(adjacency,
                                            n,
                                            source,
                                            max_iters,
                                            nullptr);
}

// Convenience overload for host adjacency with default iterations and default
// stream.
BellmanFordNoChecksResult bellman_ford_minplus_hip_no_checks(
    const std::vector<float>& adjacency,
    int n,
    int source) {
  return bellman_ford_minplus_hip_no_checks(adjacency,
                                            n,
                                            source,
                                            n - 1,
                                            nullptr);
}
