// BF10/BF9/delta-stepping cross-engine regression test (requires an AMD HIP GPU).
//
// Build and run from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip -DBF9_NO_MAIN -DBF10_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     CongestionFreeRouting/tests/bf10_vs_bf9_delta_hip_test.cpp \
//     CongestionFreeRouting/bellman_ford/bf9.cpp \
//     CongestionFreeRouting/bellman_ford/bf10.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     -o /tmp/bf10_vs_bf9_delta_hip_test
//   /tmp/bf10_vs_bf9_delta_hip_test

#include "../bellman_ford/bf9.hpp"
#include "../bellman_ford/bf10.hpp"
#include "../delta_stepping/delta_stepping_hip_CSR.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

HostCsrF32 make_weighted_graph() {
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

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

bool floats_are_bitwise_equal(float left, float right) {
  std::uint32_t left_bits = 0;
  std::uint32_t right_bits = 0;
  static_assert(sizeof(left_bits) == sizeof(left), "unexpected float width");
  std::memcpy(&left_bits, &left, sizeof(left));
  std::memcpy(&right_bits, &right, sizeof(right));
  return left_bits == right_bits;
}

void require_exact_float_vectors(const std::vector<float>& left,
                                 const std::vector<float>& right,
                                 const std::string& field) {
  require(left.size() == right.size(), field + " sizes differ");
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (!floats_are_bitwise_equal(left[i], right[i])) {
      fail(field + " differs at position " + std::to_string(i));
    }
  }
}

void require_exact_bellman_ford_results(
    const BellmanFordCsrResult& bf9,
    const BellmanFordCsrResult& bf10,
    const std::string& context) {
  require_exact_float_vectors(bf9.dist, bf10.dist, context + ": dist");
  require(bf9.pred_node == bf10.pred_node,
          context + ": predecessor nodes differ");
  require(bf9.pred_edge == bf10.pred_edge,
          context + ": predecessor edges differ");
  require(bf9.iterations_used == bf10.iterations_used,
          context + ": iteration counts differ");
  require(bf9.converged == bf10.converged,
          context + ": converged flags differ");
  require(bf9.target == bf10.target,
          context + ": target fields differ");
  require(floats_are_bitwise_equal(bf9.target_distance,
                                   bf10.target_distance),
          context + ": target distances differ");
  require(bf9.target_reached == bf10.target_reached,
          context + ": target-reached flags differ");
  require(bf9.stopped_on_target == bf10.stopped_on_target,
          context + ": stopped-on-target flags differ");
  require_exact_float_vectors(bf9.target_distances,
                              bf10.target_distances,
                              context + ": target distances");
  require(bf9.target_sources == bf10.target_sources,
          context + ": selected sources differ");
  require(bf9.target_path_offsets == bf10.target_path_offsets,
          context + ": compact node offsets differ");
  require(bf9.target_edge_offsets == bf10.target_edge_offsets,
          context + ": compact edge offsets differ");
  require(bf9.target_path_nodes == bf10.target_path_nodes,
          context + ": compact path nodes differ");
  require(bf9.target_path_edges == bf10.target_path_edges,
          context + ": compact path edges differ");
}

void validate_compact_paths(const char* engine,
                            const HostCsrF32& graph,
                            const std::vector<int>& sources,
                            const std::vector<int>& targets,
                            const BellmanFordCsrResult& result) {
  const std::string prefix = std::string(engine) + ": ";
  const std::size_t target_count = targets.size();
  require(result.target_distances.size() == target_count,
          prefix + "target-distance count is inconsistent");
  require(result.target_sources.size() == target_count,
          prefix + "target-source count is inconsistent");
  require(result.target_path_offsets.size() == target_count + 1,
          prefix + "compact node-offset count is inconsistent");
  require(result.target_edge_offsets.size() == target_count + 1,
          prefix + "compact edge-offset count is inconsistent");
  require(result.target_path_offsets.front() == 0 &&
              result.target_edge_offsets.front() == 0,
          prefix + "compact paths do not start at offset zero");
  require(result.target_path_offsets.back() >= 0 &&
              result.target_edge_offsets.back() >= 0,
          prefix + "compact path totals are negative");
  require(static_cast<std::size_t>(result.target_path_offsets.back()) ==
              result.target_path_nodes.size(),
          prefix + "final node offset does not match node storage");
  require(static_cast<std::size_t>(result.target_edge_offsets.back()) ==
              result.target_path_edges.size(),
          prefix + "final edge offset does not match edge storage");

  bool all_targets_reached = true;
  for (std::size_t i = 0; i < target_count; ++i) {
    const int node_begin = result.target_path_offsets[i];
    const int node_end = result.target_path_offsets[i + 1];
    const int edge_begin = result.target_edge_offsets[i];
    const int edge_end = result.target_edge_offsets[i + 1];
    require(node_begin >= 0 && node_end >= node_begin &&
                edge_begin >= 0 && edge_end >= edge_begin,
            prefix + "compact offsets are not monotonic at target " +
                std::to_string(i));
    require(static_cast<std::size_t>(node_end) <=
                    result.target_path_nodes.size() &&
                static_cast<std::size_t>(edge_end) <=
                    result.target_path_edges.size(),
            prefix + "compact offsets exceed storage at target " +
                std::to_string(i));

    const float distance = result.target_distances[i];
    require(!std::isnan(distance),
            prefix + "target distance is NaN at target " +
                std::to_string(i));
    if (!std::isfinite(distance)) {
      all_targets_reached = false;
      require(result.target_sources[i] == -1,
              prefix + "unreachable target has a selected source");
      require(node_begin == node_end && edge_begin == edge_end,
              prefix + "unreachable target has compact path data");
      continue;
    }

    const int selected_source = result.target_sources[i];
    require(std::find(sources.begin(), sources.end(), selected_source) !=
                sources.end(),
            prefix + "reachable target selected an unrequested source");
    require(node_end - node_begin == edge_end - edge_begin + 1,
            prefix + "path does not have exactly one fewer edge than nodes");
    require(result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                selected_source,
            prefix + "path does not begin at its selected source");
    require(result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                targets[i],
            prefix + "path does not end at its requested target");

    float path_cost = 0.0f;
    for (int path_index = 0; path_index < edge_end - edge_begin;
         ++path_index) {
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_index)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_index + 1)];
      const minplus_sparse::Offset edge = result.target_path_edges[
          static_cast<std::size_t>(edge_begin + path_index)];
      require(from >= 0 && from < graph.rows,
              prefix + "path contains an invalid source node");
      require(to >= 0 && to < graph.rows,
              prefix + "path contains an invalid destination node");
      require(edge >= 0 && edge < graph.nnz,
              prefix + "path contains an out-of-range edge ID");
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from + 1)] &&
                  graph.colind[static_cast<std::size_t>(edge)] == to,
              prefix + "path edge does not match outgoing CSR");
      path_cost += graph.values[static_cast<std::size_t>(edge)];
    }
    require(std::fabs(path_cost - distance) <= 1e-6f,
            prefix + "path cost does not match its reported distance");
  }

  require(result.target_reached == all_targets_reached,
          prefix + "aggregate reachability flag is inconsistent");
}

void compare_bellman_ford_to_delta(
    const BellmanFordCsrResult& bellman_ford,
    const DeltaSteppingCsrResult& delta,
    const std::vector<int>& targets,
    const std::string& context) {
  require(bellman_ford.target_distances.size() == targets.size() &&
              delta.target_distances.size() == targets.size(),
          context + ": engine result count does not match target count");
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const float bf_distance = bellman_ford.target_distances[i];
    const float delta_distance = delta.target_distances[i];
    const bool bf_reached = std::isfinite(bf_distance);
    const bool delta_reached = std::isfinite(delta_distance);
    require(bf_reached == delta_reached,
            context + ": engines disagree on reachability at target " +
                std::to_string(i));
    if (bf_reached) {
      require(std::fabs(bf_distance - delta_distance) <= 1e-6f,
              context + ": engines disagree on distance at target " +
                  std::to_string(i));
    }
  }
  require(bellman_ford.target_reached == delta.target_reached,
          context + ": aggregate reachability flags differ");
}

void compare_engines(const char* label,
                     const HostCsrF32& graph,
                     const std::vector<int>& sources,
                     const std::vector<int>& targets) {
  BellmanFordCsrWorkspace bf9(graph, nullptr);
  BellmanFord10CsrWorkspace bf10(graph, nullptr);
  DeltaSteppingCsrWorkspace delta(graph, nullptr);

  const BellmanFordCsrResult bf9_result =
      bf9.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  const BellmanFordCsrResult bf10_result =
      bf10.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  const DeltaSteppingCsrResult delta_result =
      delta.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);

  validate_compact_paths("BF9", graph, sources, targets, bf9_result);
  validate_compact_paths("BF10", graph, sources, targets, bf10_result);
  validate_compact_paths(
      "delta-stepping", graph, sources, targets, delta_result);
  require_exact_bellman_ford_results(bf9_result, bf10_result, label);
  compare_bellman_ford_to_delta(bf9_result, delta_result, targets, label);
  compare_bellman_ford_to_delta(bf10_result, delta_result, targets, label);

  // Reuse the same BF10 workspace so determinism also covers reusable device
  // buffers and state left behind by prior source runs.
  for (int repetition = 1; repetition <= 3; ++repetition) {
    const BellmanFordCsrResult repeated =
        bf10.run(sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
    validate_compact_paths("repeated BF10", graph, sources, targets, repeated);
    require_exact_bellman_ford_results(
        bf10_result,
        repeated,
        std::string(label) + ": BF10 repetition " +
            std::to_string(repetition));
  }
}

}  // namespace

int main() {
  try {
    compare_engines("weighted graph",
                    make_weighted_graph(),
                    std::vector<int>{1, 0, 1},
                    std::vector<int>{4, 1, 5, 6, 4});
    compare_engines("unit-weight graph",
                    make_unit_graph(),
                    std::vector<int>{0, 1, 0},
                    std::vector<int>{3, 4, 5, 5, 0});

    std::cout << "BF10 matches BF9 exactly and both agree with "
                 "delta-stepping distances and reachability\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "BF10/BF9/delta-stepping test failed: " << ex.what()
              << '\n';
    return 1;
  }
}
