#include "../unit_bfs/unit_bfs_hip_CSR.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/tests/unit_bfs_hip_test.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     -o /tmp/unit_bfs_hip_test && /tmp/unit_bfs_hip_test

#include <cmath>
#include <cstdint>
#include <future>
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

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

class HipStream {
 public:
  HipStream() {
    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
  }

  ~HipStream() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

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

HostCsrF32 make_long_chain_graph() {
  // 0 -> 1 -> ... -> 10, with vertex 11 isolated.  The depths cross both
  // sides of the traversal's first-level check and its four-level batches.
  constexpr int kChainVertices = 11;
  constexpr int kRows = 12;
  HostCsrF32 graph;
  graph.rows = kRows;
  graph.cols = kRows;
  graph.rowptr.resize(kRows + 1);
  for (int v = 0; v < kRows; ++v) {
    graph.rowptr[static_cast<std::size_t>(v)] =
        static_cast<Offset>(graph.colind.size());
    if (v + 1 < kChainVertices) {
      graph.colind.push_back(v + 1);
    }
  }
  graph.rowptr.back() = static_cast<Offset>(graph.colind.size());
  graph.nnz = static_cast<Offset>(graph.colind.size());
  graph.values.assign(graph.colind.size(), 1.0f);
  return graph;
}

HostCsrF32 make_wide_layered_graph() {
  // The first checked level has one vertex, then the queued four-level batch
  // sees frontiers of 64, 64, 64, and finally zero vertices.  Every reached
  // vertex still has a unique parent so exact compact paths remain stable.
  constexpr int kRows = 195;
  HostCsrF32 graph;
  graph.rows = kRows;
  graph.cols = kRows;
  graph.rowptr.resize(kRows + 1);
  for (int v = 0; v < kRows; ++v) {
    graph.rowptr[static_cast<std::size_t>(v)] =
        static_cast<Offset>(graph.colind.size());
    if (v == 0) {
      graph.colind.push_back(1);
    } else if (v == 1) {
      for (int dst = 2; dst <= 65; ++dst) {
        graph.colind.push_back(dst);
      }
    } else if (v >= 2 && v <= 65) {
      graph.colind.push_back(v + 64);
    } else if (v >= 66 && v <= 129) {
      graph.colind.push_back(v + 64);
    }
  }
  graph.rowptr.back() = static_cast<Offset>(graph.colind.size());
  graph.nnz = static_cast<Offset>(graph.colind.size());
  graph.values.assign(graph.colind.size(), 1.0f);
  return graph;
}

HostCsrF32 make_divergent_claim_graph() {
  // The second BFS level contains a full wave whose rows have zero through six
  // outgoing edges. Every nonempty row first claims a unique fresh vertex and
  // then revisits the source; lanes therefore execute different numbers of
  // queue-reservation calls with mixed true/false predicates. The third level
  // contends for one fresh vertex, and vertex 130 remains isolated.
  //
  //   0 -> 1..64
  //   v -> 64+v, then 0 repeated, with degree v%7
  //   65..128 -> 129                 130 (isolated)
  constexpr int kRows = 131;
  HostCsrF32 graph;
  graph.rows = kRows;
  graph.cols = kRows;
  graph.rowptr.resize(kRows + 1);
  for (int v = 0; v < kRows; ++v) {
    graph.rowptr[static_cast<std::size_t>(v)] =
        static_cast<Offset>(graph.colind.size());
    if (v == 0) {
      for (int dst = 1; dst <= 64; ++dst) {
        graph.colind.push_back(dst);
      }
    } else if (v >= 1 && v <= 64) {
      const int degree = v % 7;
      if (degree > 0) {
        graph.colind.push_back(64 + v);
        for (int edge = 1; edge < degree; ++edge) {
          graph.colind.push_back(0);
        }
      }
    } else if (v >= 65 && v <= 128) {
      graph.colind.push_back(129);
    }
  }
  graph.rowptr.back() = static_cast<Offset>(graph.colind.size());
  graph.nnz = static_cast<Offset>(graph.colind.size());
  graph.values.assign(graph.colind.size(), 1.0f);
  return graph;
}

ExpectedTarget reached_on_chain(int target) {
  std::vector<int> nodes;
  std::vector<Offset> edges;
  nodes.reserve(static_cast<std::size_t>(target) + 1);
  edges.reserve(static_cast<std::size_t>(target));
  for (int v = 0; v <= target; ++v) {
    nodes.push_back(v);
    if (v < target) {
      edges.push_back(static_cast<Offset>(v));
    }
  }
  return reached(target,
                 static_cast<float>(target),
                 0,
                 std::move(nodes),
                 std::move(edges));
}

void collect_progress(const UnitBfsCsrProgress& progress, void* user_data) {
  auto* trace = static_cast<std::vector<UnitBfsCsrProgress>*>(user_data);
  trace->push_back(progress);
}

void validate_progress_trace(const std::vector<UnitBfsCsrProgress>& trace,
                             int expected_iterations,
                             int expected_max_depth,
                             bool final_changed,
                             const std::string& label) {
  require_equal(trace.size(),
                static_cast<std::size_t>(expected_iterations),
                label + ": callback count");
  for (int i = 0; i < expected_iterations; ++i) {
    const UnitBfsCsrProgress& progress = trace[static_cast<std::size_t>(i)];
    require_equal(progress.iteration, i + 1, label + ": callback iteration");
    require_equal(progress.max_iters,
                  expected_max_depth,
                  label + ": callback max depth");
    require(progress.convergence_checked,
            label + ": callback must report a convergence check");
    const bool expected_changed =
        i + 1 == expected_iterations ? final_changed : true;
    require_equal(progress.changed,
                  expected_changed,
                  label + ": callback changed flag");
  }
}

void run_batching_suite(UnitBfsCsrOffsetMode mode,
                        const std::string& mode_label) {
  const HostCsrF32 graph = make_long_chain_graph();
  auto shared_graph = std::make_shared<UnitBfsCsrGraph>(graph, nullptr, mode);
  UnitBfsCsrWorkspace batched_workspace(shared_graph);

  UnitBfsCsrResult depth_six_result;
  for (const int target : std::vector<int>{1, 2, 4, 5, 6, 9, 10}) {
    const ExpectedRun expected{
        target, true, true, true, {reached_on_chain(target)}};
    UnitBfsCsrResult result = run_case(
        batched_workspace,
        graph,
        {0},
        {target},
        -1,
        expected,
        mode_label + ": batched target depth " + std::to_string(target));
    if (target == 6) {
      depth_six_result = std::move(result);
    }
  }

  const ExpectedRun two_targets{
      5,
      true,
      true,
      true,
      {reached_on_chain(2), reached_on_chain(5)}};
  run_case(batched_workspace,
           graph,
           {0},
           {2, 5},
           -1,
           two_targets,
           mode_label + ": targets across a batch");

  const ExpectedRun capped_before_target{
      4, false, false, false, {unreachable(5)}};
  run_case(batched_workspace,
           graph,
           {0},
           {5},
           4,
           capped_before_target,
           mode_label + ": cap before batch target");

  const ExpectedRun target_at_cap{
      5, true, true, true, {reached_on_chain(5)}};
  run_case(batched_workspace,
           graph,
           {0},
           {5},
           5,
           target_at_cap,
           mode_label + ": target at batch cap");

  const ExpectedRun exhausted{
      11, true, false, false, {unreachable(11)}};
  const UnitBfsCsrResult batched_exhausted = run_case(
      batched_workspace,
      graph,
      {0},
      {11},
      -1,
      exhausted,
      mode_label + ": frontier exhausted inside batch");

  // Installing a progress callback retains the historical one-level polling
  // behavior.  Compare it with the batched path to cover both controller modes
  // and ensure callback timing remains observable after every level.
  UnitBfsCsrWorkspace callback_workspace(shared_graph);
  std::vector<UnitBfsCsrProgress> target_trace;
  const UnitBfsCsrResult callback_target = callback_workspace.run(
      std::vector<int>{0},
      std::vector<int>{6},
      1.0f,
      6,
      nullptr,
      collect_progress,
      &target_trace);
  const ExpectedRun target_six{
      6, true, true, true, {reached_on_chain(6)}};
  validate_result(
      graph, callback_target, target_six, mode_label + ": callback target");
  validate_progress_trace(
      target_trace, 6, 6, true, mode_label + ": callback target");
  require_equivalent(depth_six_result,
                     callback_target,
                     mode_label + ": batched versus callback target");

  std::vector<UnitBfsCsrProgress> exhausted_trace;
  const UnitBfsCsrResult callback_exhausted = callback_workspace.run(
      std::vector<int>{0},
      std::vector<int>{11},
      1.0f,
      -1,
      nullptr,
      collect_progress,
      &exhausted_trace);
  validate_result(graph,
                  callback_exhausted,
                  exhausted,
                  mode_label + ": callback exhaustion");
  validate_progress_trace(
      exhausted_trace, 11, 12, false, mode_label + ": callback exhaustion");
  require_equivalent(batched_exhausted,
                     callback_exhausted,
                     mode_label + ": batched versus callback exhaustion");

  // Re-run a deep batched query after capped, exhausted, and callback-driven
  // traversals to catch stale device-controller or reset state.
  const ExpectedRun repeated_deep{
      9, true, true, true, {reached_on_chain(9)}};
  run_case(batched_workspace,
           graph,
           {0},
           {9},
           -1,
           repeated_deep,
           mode_label + ": repeated deep batched target");

  const HostCsrF32 wide_graph = make_wide_layered_graph();
  UnitBfsCsrWorkspace wide_workspace(wide_graph, nullptr, mode);
  const ExpectedRun wide_target{
      4,
      true,
      true,
      true,
      {reached(130,
               4.0f,
               0,
               {0, 1, 2, 66, 130},
               {static_cast<Offset>(0),
                static_cast<Offset>(1),
                static_cast<Offset>(65),
                static_cast<Offset>(129)})}};
  run_case(wide_workspace,
           wide_graph,
           {0},
           {130},
           -1,
           wide_target,
           mode_label + ": growing batched frontiers");

  const ExpectedRun wide_exhausted{
      5, true, false, false, {unreachable(194)}};
  run_case(wide_workspace,
           wide_graph,
           {0},
           {194},
           -1,
           wide_exhausted,
           mode_label + ": wide frontier exhaustion");

  const HostCsrF32 mixed_graph = make_divergent_claim_graph();
  UnitBfsCsrWorkspace mixed_workspace(mixed_graph, nullptr, mode);
  const ExpectedRun mixed_early_stop{
      2,
      true,
      true,
      true,
      {reached(65, 2.0f, 0, {0, 1, 65}, {0, 64})}};
  const ExpectedRun mixed_exhausted{
      4, true, false, false, {unreachable(130)}};
  constexpr int kMixedClaimReuseRuns = 64;
  for (int repetition = 0; repetition < kMixedClaimReuseRuns; ++repetition) {
    const std::string suffix =
        " reuse " + std::to_string(repetition);
    run_case(mixed_workspace,
             mixed_graph,
             {0},
             {65},
             -1,
             mixed_early_stop,
             mode_label + ": mixed-claim early stop" + suffix);
    run_case(mixed_workspace,
             mixed_graph,
             {0},
             {130},
             -1,
             mixed_exhausted,
             mode_label + ": mixed-claim exhaustion" + suffix);
  }
}

void run_parallel_workspace_suite() {
  const HostCsrF32 graph = make_divergent_claim_graph();
  auto shared_graph = std::make_shared<UnitBfsCsrGraph>(graph, nullptr);
  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");

  const ExpectedRun early_stop{
      2,
      true,
      true,
      true,
      {reached(65, 2.0f, 0, {0, 1, 65}, {0, 64})}};
  const ExpectedRun exhausted{
      4, true, false, false, {unreachable(130)}};
  constexpr int kWorkers = 8;
  constexpr int kRunsPerWorker = 16;
  std::vector<std::future<void>> workers;
  workers.reserve(kWorkers);
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker]() {
      check_hip(hipSetDevice(device), "hipSetDevice");
      HipStream stream;
      UnitBfsCsrWorkspace workspace(shared_graph, stream.get());
      for (int repetition = 0; repetition < kRunsPerWorker; ++repetition) {
        const std::string label =
            "parallel worker " + std::to_string(worker) + " reuse " +
            std::to_string(repetition);
        UnitBfsCsrResult early = workspace.run(
            std::vector<int>{0},
            std::vector<int>{65},
            1.0f,
            -1,
            stream.get(),
            nullptr,
            nullptr);
        validate_result(
            graph, early, early_stop, label + ": mixed-claim early stop");
        UnitBfsCsrResult full = workspace.run(
            std::vector<int>{0},
            std::vector<int>{130},
            1.0f,
            -1,
            stream.get(),
            nullptr,
            nullptr);
        validate_result(
            graph, full, exhausted, label + ": mixed-claim exhaustion");
      }
    }));
  }
  for (std::future<void>& worker : workers) {
    worker.get();
  }
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

    run_batching_suite(UnitBfsCsrOffsetMode::kAuto,
                       "auto/32-bit batching");
    run_batching_suite(UnitBfsCsrOffsetMode::kForce64Bit,
                       "forced-64-bit batching");
    run_parallel_workspace_suite();

    // The representation cutoff is compile-time production logic.  A giant CSR
    // fixture would be unsafe, and no public pure selection helper is exposed;
    // its INT32_MAX boundary is therefore covered by production static_asserts.
    std::cout << "Unit-BFS HIP offset/batching test passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Unit-BFS HIP offset/batching test failed: "
              << ex.what() << '\n';
    return 1;
  }
}
