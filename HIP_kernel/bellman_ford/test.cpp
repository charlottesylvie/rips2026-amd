#include "bellman_ford_hip.cpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

void require_close(float actual, float expected, int node) {
  if (std::isinf(expected)) {
    if (!std::isinf(actual)) {
      throw std::runtime_error("node " + std::to_string(node) +
                               ": expected INF, got finite value");
    }
    return;
  }

  const float diff = std::fabs(actual - expected);
  if (diff > 1e-4f) {
    throw std::runtime_error("node " + std::to_string(node) +
                             ": distance mismatch");
  }
}

}  // namespace

int main() {
  const int n = 5;
  const int source = 0;

  std::vector<float> adjacency(n * n, INF);

  // Incoming-edge orientation:
  // adjacency[v * n + u] = weight of edge u -> v.
  adjacency[1 * n + 0] = 2.0f;   // 0 -> 1
  adjacency[2 * n + 0] = 5.0f;   // 0 -> 2
  adjacency[2 * n + 1] = 1.0f;   // 1 -> 2
  adjacency[3 * n + 1] = 10.0f;  // 1 -> 3
  adjacency[3 * n + 2] = 4.0f;   // 2 -> 3
  adjacency[4 * n + 3] = 3.0f;   // 3 -> 4

  const BellmanFordResult result =
      bellman_ford_minplus_hip(adjacency, n, source);

  const std::vector<float> expected = {
      0.0f,   // source
      2.0f,   // 0 -> 1
      3.0f,   // 0 -> 1 -> 2
      7.0f,   // 0 -> 1 -> 2 -> 3
      10.0f,  // 0 -> 1 -> 2 -> 3 -> 4
  };

  for (int v = 0; v < n; ++v) {
    require_close(result.dist[v], expected[v], v);
  }

  if (result.has_negative_cycle) {
    throw std::runtime_error("unexpected negative cycle");
  }

  std::cout << "PASS\n";
  std::cout << "iterations_used: " << result.iterations_used << "\n";
  std::cout << "converged: " << (result.converged ? "true" : "false") << "\n";
  std::cout << "dist:";
  for (float value : result.dist) {
    std::cout << ' ' << value;
  }
  std::cout << "\n";

  return 0;
}
