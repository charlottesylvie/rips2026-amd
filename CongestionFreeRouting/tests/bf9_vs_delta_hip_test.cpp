// Cross-engine distance regression test (requires an AMD HIP GPU).

#include "../bellman_ford/bf9.hpp"
#include "../delta_stepping/delta_stepping_hip_CSR.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

HostCsrF32 make_graph() {
  HostCsrF32 graph;
  graph.rows = 7;
  graph.cols = 7;
  graph.nnz = 8;
  graph.rowptr = {0, 3, 5, 6, 7, 8, 8, 8};
  graph.colind = {1, 2, 3, 2, 3, 4, 4, 5};
  graph.values = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f};
  return graph;
}

HostCsrF32 make_unit_graph() {
  HostCsrF32 graph;
  graph.rows = 6;
  graph.cols = 6;
  graph.nnz = 6;
  graph.rowptr = {0, 2, 3, 4, 5, 6, 6};
  graph.colind = {1, 2, 3, 3, 4, 5};
  graph.values.assign(static_cast<std::size_t>(graph.nnz), 1.0f);
  return graph;
}

void compare_distance(float bellman_ford, float delta, std::size_t target) {
  if (std::isinf(bellman_ford) || std::isinf(delta)) {
    if (!(std::isinf(bellman_ford) && std::isinf(delta))) {
      throw std::runtime_error("engines disagree on target reachability");
    }
    return;
  }
  if (std::fabs(bellman_ford - delta) > 1e-6f) {
    throw std::runtime_error(
        "engines disagree on target " + std::to_string(target) + " distance");
  }
}

void validate_compact_paths(const char* engine,
                            const HostCsrF32& graph,
                            const std::vector<int>& sources,
                            const std::vector<int>& targets,
                            const BellmanFordCsrResult& result) {
  const std::size_t target_count = targets.size();
  if (result.target_distances.size() != target_count ||
      result.target_sources.size() != target_count ||
      result.target_path_offsets.size() != target_count + 1 ||
      result.target_edge_offsets.size() != target_count + 1) {
    throw std::runtime_error(std::string(engine) +
                             " compact result sizes are inconsistent");
  }
  if (result.target_path_offsets.front() != 0 ||
      result.target_edge_offsets.front() != 0 ||
      result.target_path_offsets.back() < 0 ||
      result.target_edge_offsets.back() < 0 ||
      static_cast<std::size_t>(result.target_path_offsets.back()) !=
          result.target_path_nodes.size() ||
      static_cast<std::size_t>(result.target_edge_offsets.back()) !=
          result.target_path_edges.size()) {
    throw std::runtime_error(std::string(engine) +
                             " compact path boundaries are inconsistent");
  }

  for (std::size_t i = 0; i < target_count; ++i) {
    const int node_begin = result.target_path_offsets[i];
    const int node_end = result.target_path_offsets[i + 1];
    const int edge_begin = result.target_edge_offsets[i];
    const int edge_end = result.target_edge_offsets[i + 1];
    if (node_begin < 0 || node_end < node_begin ||
        edge_begin < 0 || edge_end < edge_begin ||
        static_cast<std::size_t>(node_end) > result.target_path_nodes.size() ||
        static_cast<std::size_t>(edge_end) > result.target_path_edges.size()) {
      throw std::runtime_error(std::string(engine) +
                               " compact path offsets are invalid");
    }

    const float distance = result.target_distances[i];
    if (std::isnan(distance)) {
      throw std::runtime_error(std::string(engine) +
                               " target distance is NaN");
    }
    if (!std::isfinite(distance)) {
      if (result.target_sources[i] != -1 || node_begin != node_end ||
          edge_begin != edge_end) {
        throw std::runtime_error(std::string(engine) +
                                 " unreachable target has path data");
      }
      continue;
    }

    const int selected_source = result.target_sources[i];
    if (std::find(sources.begin(), sources.end(), selected_source) ==
            sources.end() ||
        node_end - node_begin != edge_end - edge_begin + 1 ||
        result.target_path_nodes[static_cast<std::size_t>(node_begin)] !=
            selected_source ||
        result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] !=
            targets[i]) {
      throw std::runtime_error(std::string(engine) +
                               " compact path endpoints are invalid");
    }

    float path_cost = 0.0f;
    for (int j = 0; j < edge_end - edge_begin; ++j) {
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + j)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + j + 1)];
      const minplus_sparse::Offset edge = result.target_path_edges[
          static_cast<std::size_t>(edge_begin + j)];
      if (from < 0 || from >= graph.rows || edge < 0 || edge >= graph.nnz ||
          edge < graph.rowptr[static_cast<std::size_t>(from)] ||
          edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
          graph.colind[static_cast<std::size_t>(edge)] != to) {
        throw std::runtime_error(std::string(engine) +
                                 " compact path contains a non-CSR edge");
      }
      path_cost += graph.values[static_cast<std::size_t>(edge)];
    }
    if (std::fabs(path_cost - distance) > 1e-6f) {
      throw std::runtime_error(std::string(engine) +
                               " compact path cost does not match distance");
    }
  }
}

void compare_engines(const HostCsrF32& graph,
                     const std::vector<int>& sources,
                     const std::vector<int>& targets) {
  BellmanFordCsrWorkspace bellman_ford(graph, nullptr);
  DeltaSteppingCsrWorkspace delta(graph, nullptr);
  const BellmanFordCsrResult bellman_ford_result =
      bellman_ford.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  const DeltaSteppingCsrResult delta_result =
      delta.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);

  validate_compact_paths(
      "Bellman-Ford", graph, sources, targets, bellman_ford_result);
  validate_compact_paths(
      "delta-stepping", graph, sources, targets, delta_result);
  if (bellman_ford_result.target_distances.size() != targets.size() ||
      delta_result.target_distances.size() != targets.size()) {
    throw std::runtime_error("engine result count does not match target count");
  }
  for (std::size_t i = 0; i < targets.size(); ++i) {
    compare_distance(bellman_ford_result.target_distances[i],
                     delta_result.target_distances[i],
                     i);
  }
}

}  // namespace

int main() {
  try {
    const HostCsrF32 graph = make_graph();
    const std::vector<int> sources = {1, 0};
    const std::vector<int> targets = {4, 1, 5, 6, 4};

    compare_engines(graph, sources, targets);
    compare_engines(make_unit_graph(),
                    std::vector<int>{0, 1},
                    std::vector<int>{3, 4, 5, 5});

    std::cout << "bf9 and delta-stepping distances and compact paths are valid\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "bf9 versus delta-stepping test failed: " << ex.what() << '\n';
    return 1;
  }
}
