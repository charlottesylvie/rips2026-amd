// Correctness tests for bellman_ford_hip.cpp.
//
// Compile from the repository root with:
//
//   hipcc -std=c++17 -O3 -x hip \
//     HIP_kernel/bellman_ford/tests/test_correctness.cpp \
//     HIP_kernel/bellman_ford/src/bellman_ford_hip.cpp \
//     HIP_kernel/minplus_mm/src/minplus_hip.cpp \
//     -o test_correctness
//
// Run:
//   ./test_correctness
//
// If your GPU is memory constrained, skip the 10000 x 10000 sparse case:
//   ./test_correctness --quick
//
// Matrix convention used by bellman_ford_hip:
//   adjacency[v * n + u] = weight of directed edge u -> v
//   adjacency[v * n + u] = INF when that edge is absent

#include "../src/bellman_ford_hip.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

struct Edge {
  int from = 0;
  int to = 0;
  float weight = 0.0f;
};

struct TestCase {
  std::string name;
  int n = 0;
  int source = 0;
  std::vector<float> explicit_flat_adjacency;
  std::vector<Edge> sparse_edges;
  std::vector<float> expected_dist;
  bool check_distances = true;
  bool large = false;
};

std::vector<float> expected_distances(
    int n,
    std::initializer_list<std::pair<int, float>> finite_values) {
  std::vector<float> dist(static_cast<std::size_t>(n), INF);
  for (const auto& [node, value] : finite_values) {
    if (node < 0 || node >= n) {
      throw std::invalid_argument("expected distance node is outside graph");
    }
    dist[static_cast<std::size_t>(node)] = value;
  }
  return dist;
}

std::vector<float> make_adjacency(int n, const std::vector<Edge>& edges) {
  const std::size_t n_size = static_cast<std::size_t>(n);
  std::vector<float> adjacency(n_size * n_size, INF);

  for (const Edge& edge : edges) {
    if (edge.from < 0 || edge.from >= n || edge.to < 0 || edge.to >= n) {
      throw std::invalid_argument("edge endpoint is outside graph");
    }

    const std::size_t index =
        static_cast<std::size_t>(edge.to) * n_size +
        static_cast<std::size_t>(edge.from);
    if (edge.weight < adjacency[index]) {
      adjacency[index] = edge.weight;
    }
  }

  return adjacency;
}

TestCase flat_case(std::string name,
                   int n,
                   int source,
                   std::vector<float> explicit_flat_adjacency,
                   std::vector<float> expected_dist) {
  return TestCase{std::move(name),
                  n,
                  source,
                  std::move(explicit_flat_adjacency),
                  {},
                  std::move(expected_dist),
                  true,
                  false};
}

TestCase sparse_case(std::string name,
                     int n,
                     int source,
                     std::vector<Edge> sparse_edges,
                     std::vector<float> expected_dist,
                     bool check_distances = true,
                     bool large = false) {
  return TestCase{std::move(name),
                  n,
                  source,
                  {},
                  std::move(sparse_edges),
                  std::move(expected_dist),
                  check_distances,
                  large};
}

bool close_enough(float actual, float expected) {
  if (std::isinf(actual) || std::isinf(expected)) {
    return std::isinf(actual) && std::isinf(expected);
  }

  const float diff = std::fabs(actual - expected);
  const float scale = 1.0f + std::fabs(actual) + std::fabs(expected);
  return diff <= 1e-4f * scale;
}

std::string format_distance(float value) {
  if (std::isinf(value)) {
    return "INF";
  }

  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

void print_reachable_paths(int source, const std::vector<float>& dist) {
  constexpr std::size_t max_printed_nodes = 24;

  std::vector<int> reachable_nodes;
  std::vector<float> reachable_distances;
  reachable_nodes.reserve(dist.size());
  reachable_distances.reserve(dist.size());

  for (std::size_t node = 0; node < dist.size(); ++node) {
    if (!std::isinf(dist[node])) {
      reachable_nodes.push_back(static_cast<int>(node));
      reachable_distances.push_back(dist[node]);
    }
  }

  const std::size_t printed =
      reachable_nodes.size() < max_printed_nodes ? reachable_nodes.size()
                                                 : max_printed_nodes;

  std::cout << "    paths found from node " << source << " to nodes [";
  for (std::size_t i = 0; i < printed; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << reachable_nodes[i];
  }
  if (printed < reachable_nodes.size()) {
    std::cout << ", ...";
  }
  std::cout << "] with distances [";

  for (std::size_t i = 0; i < printed; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << format_distance(reachable_distances[i]);
  }
  if (printed < reachable_distances.size()) {
    std::cout << ", ...";
  }
  std::cout << "] (" << reachable_nodes.size() << "/" << dist.size()
            << " reachable)\n";
}

void require_flat_size(const TestCase& test) {
  if (test.explicit_flat_adjacency.empty()) {
    return;
  }

  const std::size_t expected_entries =
      static_cast<std::size_t>(test.n) * static_cast<std::size_t>(test.n);
  if (test.explicit_flat_adjacency.size() != expected_entries) {
    throw std::runtime_error(test.name +
                             " explicit flattened matrix is not n*n");
  }
}

int count_finite_edges(const std::vector<float>& adjacency) {
  int count = 0;
  for (float value : adjacency) {
    if (!std::isinf(value)) {
      ++count;
    }
  }
  return count;
}

int check_distance_mismatches(const TestCase& test,
                              const std::vector<float>& actual) {
  if (!test.check_distances) {
    return 0;
  }

  if (actual.size() != test.expected_dist.size()) {
    std::cout << "    distance vector size mismatch: actual=" << actual.size()
              << " expected=" << test.expected_dist.size() << "\n";
    return 1;
  }

  int mismatches = 0;
  constexpr int max_printed_mismatches = 8;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (close_enough(actual[i], test.expected_dist[i])) {
      continue;
    }

    ++mismatches;
    if (mismatches <= max_printed_mismatches) {
      std::cout << "    node " << i << ": actual="
                << format_distance(actual[i])
                << " expected=" << format_distance(test.expected_dist[i])
                << "\n";
    }
  }

  if (mismatches > max_printed_mismatches) {
    std::cout << "    ... " << (mismatches - max_printed_mismatches)
              << " more distance mismatches\n";
  }

  return mismatches;
}

TestCase make_grid_10x10_case() {
  constexpr int width = 10;
  constexpr int height = 10;
  constexpr int n = width * height;

  std::vector<Edge> edges;
  edges.reserve(2 * n);
  std::vector<float> expected(static_cast<std::size_t>(n), INF);

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int node = row * width + col;
      expected[static_cast<std::size_t>(node)] =
          static_cast<float>(2 * row + col);

      if (col + 1 < width) {
        edges.push_back(Edge{node, node + 1, 1.0f});
      }
      if (row + 1 < height) {
        edges.push_back(Edge{node, node + width, 2.0f});
      }
    }
  }

  return sparse_case("sparse_grid_10x10_directed_right_down",
                     n,
                     0,
                     std::move(edges),
                     std::move(expected));
}

TestCase make_dense_complete_10_case() {
  constexpr int n = 10;
  std::vector<Edge> edges;
  edges.reserve(n * (n - 1));
  std::vector<float> expected(static_cast<std::size_t>(n), INF);

  for (int node = 0; node < n; ++node) {
    expected[static_cast<std::size_t>(node)] = static_cast<float>(node);
  }

  for (int from = 0; from < n; ++from) {
    for (int to = 0; to < n; ++to) {
      if (from == to) {
        continue;
      }

      const float weight =
          from < to ? static_cast<float>(to - from)
                    : static_cast<float>(100 + from - to);
      edges.push_back(Edge{from, to, weight});
    }
  }

  return sparse_case("dense_complete_10_positive_cycles",
                     n,
                     0,
                     std::move(edges),
                     std::move(expected));
}

TestCase make_medium_connected_1024_case() {
  constexpr int n = 1024;
  constexpr int backward_span = 16;
  constexpr int forward_span = 48;

  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) *
                (backward_span + forward_span + 1));
  std::vector<float> expected(static_cast<std::size_t>(n), INF);

  for (int node = 0; node < n; ++node) {
    expected[static_cast<std::size_t>(node)] = static_cast<float>(node);
  }

  for (int node = 1; node < n; ++node) {
    edges.push_back(Edge{0, node, static_cast<float>(node)});
  }

  for (int from = 0; from < n; ++from) {
    for (int offset = -backward_span; offset <= forward_span; ++offset) {
      if (offset == 0) {
        continue;
      }

      const int to = (from + offset + n) % n;
      const float slack = static_cast<float>((std::abs(offset) % 5) + 1);
      const float weight = static_cast<float>(to - from) + slack;
      edges.push_back(Edge{from, to, weight});
    }
  }

  return sparse_case("medium_connected_1024_local_density",
                     n,
                     0,
                     std::move(edges),
                     std::move(expected));
}

TestCase make_large_sparse_10000_case() {
  constexpr int n = 10000;
  std::vector<Edge> edges;
  edges.reserve(n + 1);
  std::vector<float> expected(static_cast<std::size_t>(n), INF);

  expected[0] = 0.0f;
  for (int node = 1; node < n; ++node) {
    const float direct_weight = static_cast<float>((node % 97) + 1);
    edges.push_back(Edge{0, node, direct_weight});
    expected[static_cast<std::size_t>(node)] = direct_weight;
  }

  edges.push_back(Edge{0, 5000, 1.0f});
  expected[5000] = 1.0f;

  edges.push_back(Edge{5000, 9999, -5.0f});
  expected[9999] = -4.0f;

  return sparse_case("large_sparse_star_10000_with_two_hop_improvement",
                     n,
                     0,
                     std::move(edges),
                     std::move(expected),
                     true,
                     true);
}

std::vector<TestCase> make_tests() {
  const float I = INF;
  std::vector<TestCase> tests;

  tests.push_back(sparse_case("single_vertex",
                              1,
                              0,
                              {},
                              expected_distances(1, {{0, 0.0f}})));

  tests.push_back(sparse_case("two_vertices_unreachable",
                              2,
                              0,
                              {},
                              expected_distances(2, {{0, 0.0f}})));

  tests.push_back(flat_case(
      "flattened_diamond_positive",
      4,
      0,
      {
          I, I, I, I,
          2.0f, I, I, I,
          5.0f, 1.0f, I, I,
          I, 4.0f, 1.0f, I,
      },
      expected_distances(4,
                         {{0, 0.0f}, {1, 2.0f}, {2, 3.0f}, {3, 4.0f}})));

  tests.push_back(flat_case(
      "flattened_negative_edges_no_cycle",
      5,
      0,
      {
          I, I, I, I, 2.0f,
          6.0f, I, I, -2.0f, I,
          7.0f, 8.0f, I, I, I,
          I, 5.0f, -3.0f, I, 7.0f,
          I, -4.0f, 9.0f, I, I,
      },
      expected_distances(5,
                         {{0, 0.0f},
                          {1, 2.0f},
                          {2, 7.0f},
                          {3, 4.0f},
                          {4, -2.0f}})));

  tests.push_back(sparse_case(
      "sparse_disconnected_components",
      6,
      0,
      {Edge{0, 1, 3.0f},
       Edge{1, 2, 4.0f},
       Edge{3, 4, 1.0f},
       Edge{4, 5, 1.0f}},
      expected_distances(6, {{0, 0.0f}, {1, 3.0f}, {2, 7.0f}})));

  tests.push_back(sparse_case(
      "sparse_zero_weight_edges",
      6,
      0,
      {Edge{0, 1, 0.0f},
       Edge{0, 2, 5.0f},
       Edge{1, 2, 0.0f},
       Edge{2, 3, 0.0f},
       Edge{1, 4, 2.0f},
       Edge{4, 5, 0.0f},
       Edge{3, 5, 4.0f}},
      expected_distances(6,
                         {{0, 0.0f},
                          {1, 0.0f},
                          {2, 0.0f},
                          {3, 0.0f},
                          {4, 2.0f},
                          {5, 2.0f}})));

  tests.push_back(sparse_case(
      "sparse_duplicate_edges_keep_min",
      5,
      0,
      {Edge{0, 1, 10.0f},
       Edge{0, 1, 3.0f},
       Edge{1, 2, 4.0f},
       Edge{0, 2, 20.0f},
       Edge{2, 3, 1.0f},
       Edge{1, 3, 100.0f},
       Edge{3, 4, -2.0f}},
      expected_distances(5,
                         {{0, 0.0f},
                          {1, 3.0f},
                          {2, 7.0f},
                          {3, 8.0f},
                          {4, 6.0f}})));

  tests.push_back(sparse_case(
      "sparse_source_not_zero",
      7,
      3,
      {Edge{3, 0, 2.0f},
       Edge{3, 4, 1.0f},
       Edge{4, 5, 1.0f},
       Edge{5, 6, 1.0f},
       Edge{0, 1, 3.0f},
       Edge{1, 2, 3.0f},
       Edge{3, 2, 10.0f},
       Edge{6, 2, -4.0f}},
      expected_distances(7,
                         {{0, 2.0f},
                          {1, 5.0f},
                          {2, -1.0f},
                          {3, 0.0f},
                          {4, 1.0f},
                          {5, 2.0f},
                          {6, 3.0f}})));

  tests.push_back(sparse_case(
      "sparse_negative_edges_branching",
      8,
      0,
      {Edge{0, 1, 4.0f},
       Edge{0, 2, 1.0f},
       Edge{2, 1, -2.0f},
       Edge{1, 3, 2.0f},
       Edge{2, 3, 5.0f},
       Edge{3, 4, 1.0f},
       Edge{1, 5, 10.0f},
       Edge{4, 5, -1.0f},
       Edge{5, 6, 2.0f},
       Edge{2, 7, 8.0f},
       Edge{6, 7, -3.0f}},
      expected_distances(8,
                         {{0, 0.0f},
                          {1, -1.0f},
                          {2, 1.0f},
                          {3, 1.0f},
                          {4, 2.0f},
                          {5, 1.0f},
                          {6, 3.0f},
                          {7, 0.0f}})));

  tests.push_back(sparse_case(
      "sparse_unreachable_negative_edges",
      5,
      0,
      {Edge{0, 1, 2.0f},
       Edge{1, 2, 2.0f},
       Edge{3, 4, -5.0f}},
      expected_distances(5, {{0, 0.0f}, {1, 2.0f}, {2, 4.0f}})));

  tests.push_back(make_grid_10x10_case());
  tests.push_back(make_dense_complete_10_case());
  tests.push_back(make_medium_connected_1024_case());
  tests.push_back(make_large_sparse_10000_case());

  return tests;
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " [--quick]\n";
  std::cerr << "  --quick  skip the 10000 x 10000 sparse correctness case\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick" || arg == "--skip-large") {
      quick = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      print_usage(argv[0]);
      return 1;
    }
  }

  int passed = 0;
  int failed = 0;
  int skipped = 0;

  std::cout << std::fixed << std::setprecision(3);

  for (const TestCase& test : make_tests()) {
    if (quick && test.large) {
      ++skipped;
      std::cout << "[SKIP] " << test.name << " (--quick)\n";
      continue;
    }

    std::cout << "[RUN ] " << test.name << "\n";

    try {
      require_flat_size(test);

      std::vector<float> adjacency =
          test.explicit_flat_adjacency.empty()
              ? make_adjacency(test.n, test.sparse_edges)
              : test.explicit_flat_adjacency;

      const int finite_edges = count_finite_edges(adjacency);
      const double density =
          static_cast<double>(finite_edges) /
          (static_cast<double>(test.n) * static_cast<double>(test.n));
      std::cout << "    n=" << test.n << " source=" << test.source
                << " # edges=" << finite_edges
                << " density=" << (100.0 * density) << "%\n";

      const auto start = std::chrono::steady_clock::now();
      const BellmanFordResult result =
          bellman_ford_minplus_hip(adjacency, test.n, test.source);
      const auto end = std::chrono::steady_clock::now();
      const double runtime_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      int mismatches = check_distance_mismatches(test, result.dist);
      std::cout << "    runtime_ms=" << runtime_ms
                << " iterations_used=" << result.iterations_used
                << "\n";
      print_reachable_paths(test.source, result.dist);

      if (mismatches == 0) {
        ++passed;
        std::cout << "[PASS] " << test.name << "\n";
      } else {
        ++failed;
        std::cout << "[FAIL] " << test.name << " (" << mismatches
                  << " mismatch(es))\n";
      }
    } catch (const std::exception& ex) {
      ++failed;
      std::cout << "[FAIL] " << test.name << " threw exception: " << ex.what()
                << "\n";
    }
  }

  std::cout << "\nsummary: passed=" << passed << " failed=" << failed
            << " skipped=" << skipped << "\n";
  return failed == 0 ? 0 : 1;
}
