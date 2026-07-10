#include "../unit_bfs/unit_bfs_hip_CSR.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O2 -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/tests/unit_bfs_hip_test.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     -o /tmp/unit_bfs_hip_test && /tmp/unit_bfs_hip_test

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;

static_assert(
    std::numeric_limits<Offset>::max() >
        static_cast<Offset>(std::numeric_limits<std::int32_t>::max()),
    "public unit-BFS edge IDs must retain more than 32 bits of range");

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

template <typename T>
void require_equal(const T& actual,
                   const T& expected,
                   const std::string& label) {
  if (!(actual == expected)) {
    fail(label + " does not match the expected value");
  }
}

HostCsrF32 make_unique_path_graph() {
  // The two branches have different lengths before they merge, so every target
  // below has a unique shortest predecessor chain.  This lets the test compare
  // exact edge IDs across the 32-bit and 64-bit internal representations without
  // depending on which GPU lane wins a same-level predecessor race.
  //
  //   0 --0--> 1 --2--> 2 --3--> 3 --4--\
  //    \                                      > 5 --6--> 6    7 (isolated)
  //     `--1--> 4 --------------------5--/
  HostCsrF32 graph;
  graph.rows = 8;
  graph.cols = 8;
  graph.nnz = 7;
  graph.rowptr = {0, 2, 3, 4, 5, 6, 7, 7, 7};
  graph.colind = {1, 4, 2, 3, 5, 5, 6};
  graph.values.assign(static_cast<std::size_t>(graph.nnz), 1.0f);
  return graph;
}

struct ExpectedTarget {
  int target = -1;
  float distance = std::numeric_limits<float>::infinity();
  int source = -1;
  std::vector<int> nodes;
  std::vector<Offset> edges;
};

struct ExpectedRun {
  int iterations_used = 0;
  bool converged = false;
  bool stopped_on_target = false;
  bool all_targets_reached = false;
  std::vector<ExpectedTarget> targets;
};

ExpectedTarget reached(int target,
                       float distance,
                       int source,
                       std::vector<int> nodes,
                       std::vector<Offset> edges) {
  return {target, distance, source, std::move(nodes), std::move(edges)};
}

ExpectedTarget unreachable(int target) {
  return {target,
          std::numeric_limits<float>::infinity(),
          -1,
          {},
          {}};
}

void validate_outgoing_edge_path(const HostCsrF32& graph,
                                 const ExpectedTarget& target,
                                 const std::vector<int>& nodes,
                                 const std::vector<Offset>& edges,
                                 const std::string& label) {
  require(!nodes.empty(), label + ": reached path must contain a node");
  require_equal(nodes.front(), target.source, label + ": path source");
  require_equal(nodes.back(), target.target, label + ": path target");
  require_equal(edges.size(), nodes.size() - 1, label + ": edge count");

  for (std::size_t i = 0; i < edges.size(); ++i) {
    const int from = nodes[i];
    const int to = nodes[i + 1];
    require(from >= 0 && static_cast<Offset>(from) < graph.rows,
            label + ": path source vertex is outside the graph");
    require(to >= 0 && static_cast<Offset>(to) < graph.rows,
            label + ": path destination vertex is outside the graph");

    const Offset edge = edges[i];
    require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                edge < graph.rowptr[static_cast<std::size_t>(from + 1)],
            label + ": returned edge ID is outside its outgoing CSR row");
    require_equal(graph.colind[static_cast<std::size_t>(edge)],
                  to,
                  label + ": returned edge destination");
  }
}

void validate_result(const HostCsrF32& graph,
                     const UnitBfsCsrResult& result,
                     const ExpectedRun& expected,
                     const std::string& label) {
  const std::size_t target_count = expected.targets.size();
  require_equal(result.target, -1, label + ": vector-run target marker");
  require_equal(result.iterations_used,
                expected.iterations_used,
                label + ": iterations_used");
  require_equal(result.converged, expected.converged, label + ": converged");
  require_equal(result.stopped_on_target,
                expected.stopped_on_target,
                label + ": stopped_on_target");
  require_equal(result.target_reached,
                expected.all_targets_reached,
                label + ": target_reached");

  require_equal(result.target_distances.size(),
                target_count,
                label + ": target distance count");
  require_equal(result.target_sources.size(),
                target_count,
                label + ": target source count");
  require_equal(result.target_path_offsets.size(),
                target_count + 1,
                label + ": node offset count");
  require_equal(result.target_edge_offsets.size(),
                target_count + 1,
                label + ": edge offset count");
  require_equal(result.target_path_offsets.front(), 0, label + ": first node offset");
  require_equal(result.target_edge_offsets.front(), 0, label + ": first edge offset");
  require_equal(static_cast<std::size_t>(result.target_path_offsets.back()),
                result.target_path_nodes.size(),
                label + ": final node offset");
  require_equal(static_cast<std::size_t>(result.target_edge_offsets.back()),
                result.target_path_edges.size(),
                label + ": final edge offset");

  for (std::size_t i = 0; i < target_count; ++i) {
    std::ostringstream item_label;
    item_label << label << ": target[" << i << "]=" << expected.targets[i].target;
    const std::string current_label = item_label.str();

    const int node_begin = result.target_path_offsets[i];
    const int node_end = result.target_path_offsets[i + 1];
    const int edge_begin = result.target_edge_offsets[i];
    const int edge_end = result.target_edge_offsets[i + 1];
    require(node_begin >= 0 && node_end >= node_begin,
            current_label + ": invalid node offsets");
    require(edge_begin >= 0 && edge_end >= edge_begin,
            current_label + ": invalid edge offsets");
    require(static_cast<std::size_t>(node_end) <= result.target_path_nodes.size(),
            current_label + ": node slice exceeds compact output");
    require(static_cast<std::size_t>(edge_end) <= result.target_path_edges.size(),
            current_label + ": edge slice exceeds compact output");

    const std::vector<int> nodes(
        result.target_path_nodes.begin() + node_begin,
        result.target_path_nodes.begin() + node_end);
    const std::vector<Offset> edges(
        result.target_path_edges.begin() + edge_begin,
        result.target_path_edges.begin() + edge_end);
    const ExpectedTarget& expected_target = expected.targets[i];

    if (std::isfinite(expected_target.distance)) {
      require(std::isfinite(result.target_distances[i]),
              current_label + ": reached target distance is not finite");
      require(std::fabs(result.target_distances[i] - expected_target.distance) < 1e-6f,
              current_label + ": target distance does not match");
      require_equal(result.target_sources[i],
                    expected_target.source,
                    current_label + ": source");
      require_equal(nodes, expected_target.nodes, current_label + ": path nodes");
      require_equal(edges, expected_target.edges, current_label + ": path edges");
      require_equal(static_cast<float>(edges.size()),
                    result.target_distances[i],
                    current_label + ": unit-weight path length");
      validate_outgoing_edge_path(graph,
                                  expected_target,
                                  nodes,
                                  edges,
                                  current_label);
    } else {
      require(std::isinf(result.target_distances[i]),
              current_label + ": unreachable target distance must be infinite");
      require_equal(result.target_sources[i], -1, current_label + ": unreachable source");
      require(nodes.empty(), current_label + ": unreachable target has path nodes");
      require(edges.empty(), current_label + ": unreachable target has path edges");
    }
  }
}

UnitBfsCsrResult run_case(UnitBfsCsrWorkspace& workspace,
                         const HostCsrF32& graph,
                         const std::vector<int>& sources,
                         const std::vector<int>& targets,
                         int max_depth,
                         const ExpectedRun& expected,
                         const std::string& label) {
  UnitBfsCsrResult result =
      workspace.run(sources, targets, 1.0f, max_depth, nullptr, nullptr, nullptr);
  validate_result(graph, result, expected, label);
  return result;
}

std::vector<UnitBfsCsrResult> run_mode_suite(
    const HostCsrF32& graph,
    UnitBfsCsrOffsetMode mode,
    bool expect_32_bit,
    const std::string& mode_label) {
  auto shared_graph =
      std::make_shared<UnitBfsCsrGraph>(graph, nullptr, mode);
  require_equal(shared_graph->uses_32_bit_offsets(),
                expect_32_bit,
                mode_label + ": selected offset representation");
  UnitBfsCsrWorkspace workspace(shared_graph);
  std::vector<UnitBfsCsrResult> results;

  const ExpectedRun full_paths{
      3,
      true,
      true,
      true,
      {reached(3, 3.0f, 0, {0, 1, 2, 3}, {0, 2, 3}),
       reached(6, 3.0f, 0, {0, 4, 5, 6}, {1, 5, 6})}};
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {3, 6},
                             -1,
                             full_paths,
                             mode_label + ": full paths"));

  const ExpectedRun discovered_duplicate_targets{
      3,
      true,
      true,
      true,
      {reached(3, 3.0f, 0, {0, 1, 2, 3}, {0, 2, 3}),
       reached(3, 3.0f, 0, {0, 1, 2, 3}, {0, 2, 3})}};
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {3, 3},
                             -1,
                             discovered_duplicate_targets,
                             mode_label + ": discovered duplicate targets"));

  const ExpectedRun duplicates_and_unreachable{
      4,
      true,
      false,
      false,
      {reached(4, 0.0f, 4, {4}, {}),
       reached(4, 0.0f, 4, {4}, {}),
       reached(3, 3.0f, 0, {0, 1, 2, 3}, {0, 2, 3}),
       unreachable(7)}};
  results.push_back(run_case(workspace,
                             graph,
                             {0, 0, 4, 4},
                             {4, 4, 3, 7},
                             -1,
                             duplicates_and_unreachable,
                             mode_label + ": duplicate sources and targets"));

  const ExpectedRun capped_before_targets{
      2,
      false,
      false,
      false,
      {unreachable(3), unreachable(6)}};
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {3, 6},
                             2,
                             capped_before_targets,
                             mode_label + ": max depth before targets"));

  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {3, 6},
                             3,
                             full_paths,
                             mode_label + ": targets at max depth"));

  const ExpectedRun source_is_target{
      0,
      true,
      true,
      true,
      {reached(0, 0.0f, 0, {0}, {})}};
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {0},
                             0,
                             source_is_target,
                             mode_label + ": source is target at depth zero"));

  const ExpectedRun zero_depth_miss{
      0,
      false,
      false,
      false,
      {unreachable(1)}};
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {1},
                             0,
                             zero_depth_miss,
                             mode_label + ": zero depth miss"));

  const ExpectedRun source_and_unreachable{
      1,
      true,
      false,
      false,
      {reached(6, 0.0f, 6, {6}, {}), unreachable(7)}};
  results.push_back(run_case(workspace,
                             graph,
                             {6},
                             {6, 7},
                             -1,
                             source_and_unreachable,
                             mode_label + ": source target plus empty frontier"));

  // Return to the original query after multiple early-stop, depth-limited, and
  // exhausted searches.  This specifically checks reset of the active internal
  // predecessor-width buffer and the shared level/target scratch arrays.
  results.push_back(run_case(workspace,
                             graph,
                             {0},
                             {3, 6},
                             -1,
                             full_paths,
                             mode_label + ": repeated full paths"));
  return results;
}

void require_equivalent(const UnitBfsCsrResult& left,
                        const UnitBfsCsrResult& right,
                        const std::string& label) {
  require_equal(left.iterations_used, right.iterations_used, label + ": iterations");
  require_equal(left.converged, right.converged, label + ": converged");
  require_equal(left.target_reached, right.target_reached, label + ": target_reached");
  require_equal(left.stopped_on_target,
                right.stopped_on_target,
                label + ": stopped_on_target");
  require_equal(left.target_distances,
                right.target_distances,
                label + ": target distances");
  require_equal(left.target_sources, right.target_sources, label + ": target sources");
  require_equal(left.target_path_offsets,
                right.target_path_offsets,
                label + ": target path offsets");
  require_equal(left.target_edge_offsets,
                right.target_edge_offsets,
                label + ": target edge offsets");
  require_equal(left.target_path_nodes,
                right.target_path_nodes,
                label + ": compact path nodes");
  require_equal(left.target_path_edges,
                right.target_path_edges,
                label + ": compact path edges");
}

}  // namespace

int main() {
  try {
    const HostCsrF32 graph = make_unique_path_graph();
    const std::vector<UnitBfsCsrResult> auto_results =
        run_mode_suite(graph,
                       UnitBfsCsrOffsetMode::kAuto,
                       true,
                       "auto/32-bit");
    const std::vector<UnitBfsCsrResult> forced_64_results =
        run_mode_suite(graph,
                       UnitBfsCsrOffsetMode::kForce64Bit,
                       false,
                       "forced-64-bit");

    UnitBfsCsrWorkspace direct_forced_64(
        graph, nullptr, UnitBfsCsrOffsetMode::kForce64Bit);
    const UnitBfsCsrResult direct_forced_64_result = direct_forced_64.run(
        std::vector<int>{0},
        std::vector<int>{3, 6},
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(!forced_64_results.empty(), "forced-64-bit suite returned no results");
    require_equivalent(direct_forced_64_result,
                       forced_64_results.front(),
                       "direct forced-64-bit workspace constructor");

    require_equal(auto_results.size(),
                  forced_64_results.size(),
                  "offset-mode result count");
    for (std::size_t i = 0; i < auto_results.size(); ++i) {
      std::ostringstream label;
      label << "auto/32-bit versus forced-64-bit case " << i;
      require_equivalent(auto_results[i], forced_64_results[i], label.str());
    }

    // The representation cutoff is compile-time production logic.  A giant CSR
    // fixture would be unsafe, and no public pure selection helper is exposed;
    // its INT32_MAX boundary is therefore covered by production static_asserts.
    std::cout << "Unit-BFS HIP 32-bit/64-bit offset test passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Unit-BFS HIP offset test failed: " << ex.what() << '\n';
    return 1;
  }
}
