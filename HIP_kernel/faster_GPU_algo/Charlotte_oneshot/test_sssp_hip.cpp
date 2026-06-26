#include "sssp_hip.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

struct CpuRefResult {
  std::vector<float> dist;
  double milliseconds = 0.0;
};

CsrGraph make_random_sparse_graph(int n, int avg_extra_degree, uint32_t seed) {
  if (n <= 0 || avg_extra_degree < 0) {
    throw std::invalid_argument("bad graph dimensions");
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> vertex_dist(0, n - 1);
  std::uniform_real_distribution<float> weight_dist(1.0f, 20.0f);

  std::vector<std::vector<std::pair<int, float>>> adj(n);

  // A directed ring makes every vertex reachable from source 0.
  for (int u = 0; u < n; ++u) {
    const int v = (u + 1) % n;
    adj[u].push_back({v, 1.0f});
  }

  for (int u = 0; u < n; ++u) {
    for (int k = 0; k < avg_extra_degree; ++k) {
      int v = vertex_dist(rng);
      if (v == u) {
        v = (v + 1) % n;
      }
      adj[u].push_back({v, weight_dist(rng)});
    }
  }

  CsrGraph g;
  g.num_nodes = n;
  g.row_offsets.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    g.row_offsets[u] = static_cast<int>(g.col_indices.size());
    for (const auto& edge : adj[u]) {
      g.col_indices.push_back(edge.first);
      g.weights.push_back(edge.second);
    }
  }
  g.row_offsets[n] = static_cast<int>(g.col_indices.size());
  g.num_edges = static_cast<int>(g.col_indices.size());
  return g;
}

CpuRefResult dijkstra_cpu(const CsrGraph& g, int source) {
  using Item = std::pair<float, int>;
  CpuRefResult result;
  result.dist.assign(g.num_nodes, INF);
  result.dist[source] = 0.0f;

  auto t0 = std::chrono::high_resolution_clock::now();
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  pq.push({0.0f, source});
  while (!pq.empty()) {
    const auto [du, u] = pq.top();
    pq.pop();
    if (du != result.dist[u]) {
      continue;
    }
    for (int e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
      const int v = g.col_indices[e];
      const float nd = du + g.weights[e];
      if (nd < result.dist[v]) {
        result.dist[v] = nd;
        pq.push({nd, v});
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  result.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return INF;
  }
  float err = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::isinf(a[i]) || std::isinf(b[i])) {
      if (std::isinf(a[i]) != std::isinf(b[i])) {
        return INF;
      }
      continue;
    }
    err = std::max(err, std::fabs(a[i] - b[i]));
  }
  return err;
}

void print_usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " [n] [extra_degree] [source] [seed] [run_bf]\n"
            << "  default: n=4096 extra_degree=8 source=0 seed=1 run_bf=1\n"
            << "  For large graphs, set run_bf=0 because Bellman-Ford is intentionally slow.\n";
}

int parse_int_arg(char** argv, int argc, int index, int fallback) {
  if (index >= argc) {
    return fallback;
  }
  return std::stoi(argv[index]);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      print_usage(argv[0]);
      return 0;
    }

    const int n = parse_int_arg(argv, argc, 1, 4096);
    const int extra_degree = parse_int_arg(argv, argc, 2, 8);
    const int source = parse_int_arg(argv, argc, 3, 0);
    const uint32_t seed = static_cast<uint32_t>(parse_int_arg(argv, argc, 4, 1));
    const bool run_bf = parse_int_arg(argv, argc, 5, 1) != 0;

    std::cout << "building graph...\n";
    CsrGraph g = make_random_sparse_graph(n, extra_degree, seed);
    std::cout << "nodes: " << g.num_nodes << " edges: " << g.num_edges
              << " avg_degree: " << (static_cast<double>(g.num_edges) / g.num_nodes) << "\n";

    std::cout << "running CPU Dijkstra reference...\n";
    CpuRefResult ref = dijkstra_cpu(g, source);
    std::cout << "CPU Dijkstra ms: " << ref.milliseconds << "\n";

    SSSPOptions opts;
    opts.num_buckets = 32;
    opts.dynamic_delta = true;
    opts.bucket_capacity_per_bucket = std::max(4096, 4 * n);

    std::cout << "running ADDS-lite HIP...\n";
    SSSPResult adds = adds_sssp_hip(g, source, opts);
    const float adds_err = max_abs_error(adds.dist, ref.dist);
    std::cout << "ADDS-lite HIP ms: " << adds.milliseconds
              << " iterations/bucket_advances: " << adds.iterations_used
              << " converged: " << (adds.converged ? "true" : "false")
              << " work_count(vertices processed): " << adds.work_count
              << " final_delta: " << adds.final_delta
              << " max_abs_error_vs_cpu: " << adds_err << "\n";

    if (adds_err > 1e-3f || !adds.converged) {
      std::cerr << "ADDS-lite validation failed\n";
      return 2;
    }

    if (run_bf) {
      std::cout << "running sparse Bellman-Ford HIP...\n";
      SSSPOptions bf_opts;
      bf_opts.max_iters = n - 1;
      SSSPResult bf = bellman_ford_sssp_hip(g, source, bf_opts);
      const float bf_err = max_abs_error(bf.dist, ref.dist);
      std::cout << "Bellman-Ford HIP ms: " << bf.milliseconds
                << " iterations: " << bf.iterations_used
                << " converged: " << (bf.converged ? "true" : "false")
                << " has_negative_cycle: " << (bf.has_negative_cycle ? "true" : "false")
                << " work_count(edge relax attempts): " << bf.work_count
                << " max_abs_error_vs_cpu: " << bf_err << "\n";
      if (bf_err > 1e-3f || bf.has_negative_cycle) {
        std::cerr << "Bellman-Ford validation failed\n";
        return 3;
      }

      if (adds.milliseconds > 0.0f) {
        std::cout << "speedup ADDS-lite over Bellman-Ford: "
                  << (bf.milliseconds / adds.milliseconds) << "x\n";
      }
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
