#include "../delta_stepping/delta_stepping_hip_CSR.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/delta_stepping \
//     CongestionFreeRouting/tests/delta_stepping_hip_test.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     -o /tmp/delta_stepping_hip_test && /tmp/delta_stepping_hip_test

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kAbsoluteTolerance = 2e-3f;
constexpr float kRelativeTolerance = 2e-5f;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

template <typename Function>
void require_invalid_argument(const std::string& label, Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, label + ": expected std::invalid_argument");
}

void record_progress(const DeltaSteppingCsrProgress& progress,
                     void* user_data) {
  auto* records =
      static_cast<std::vector<DeltaSteppingCsrProgress>*>(user_data);
  records->push_back(progress);
}

class ExpectedProgressCallbackError : public std::runtime_error {
 public:
  ExpectedProgressCallbackError()
      : std::runtime_error("expected delta progress callback failure") {}
};

struct ThrowingProgressState {
  int calls = 0;
  int throw_on_call = 1;
};

void throw_progress_on_selected_call(const DeltaSteppingCsrProgress&,
                                     void* user_data) {
  auto* state = static_cast<ThrowingProgressState*>(user_data);
  ++state->calls;
  if (state->calls == state->throw_on_call) {
    throw ExpectedProgressCallbackError();
  }
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
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

struct EdgeSpec {
  int from = -1;
  int to = -1;
  float weight = 0.0f;
};

HostCsrF32 make_outgoing_csr(int vertex_count,
                            const std::vector<EdgeSpec>& edges) {
  if (vertex_count <= 0) {
    throw std::invalid_argument("test graph must contain a vertex");
  }

  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(vertex_count) + 1, 0);

  for (const EdgeSpec& edge : edges) {
    if (edge.from < 0 || edge.from >= vertex_count || edge.to < 0 ||
        edge.to >= vertex_count) {
      throw std::invalid_argument("test edge endpoint is outside the graph");
    }
    if (!std::isfinite(edge.weight) || edge.weight < 0.0f) {
      throw std::invalid_argument("test edge weight must be finite and nonnegative");
    }
    ++graph.rowptr[static_cast<std::size_t>(edge.from) + 1];
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex + 1)] +=
        graph.rowptr[static_cast<std::size_t>(vertex)];
  }

  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
  // Preserve input order within each row. The production kernels must not
  // depend on sorted columns, even though the FPGA converter normally sorts.
  for (const EdgeSpec& edge : edges) {
    const std::size_t row = static_cast<std::size_t>(edge.from);
    const std::size_t position = static_cast<std::size_t>(cursor[row]++);
    graph.colind[position] = edge.to;
    graph.values[position] = edge.weight;
  }
  return graph;
}

float effective_edge_weight(const HostCsrF32& graph,
                            Offset edge,
                            const std::vector<float>* vertex_costs) {
  const std::size_t edge_index = static_cast<std::size_t>(edge);
  const int destination = graph.colind[edge_index];
  const float multiplier =
      vertex_costs == nullptr
          ? 1.0f
          : (*vertex_costs)[static_cast<std::size_t>(destination)];
  return graph.values[edge_index] * multiplier;
}

std::vector<float> cpu_dijkstra_outgoing_multi_source(
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<float>* vertex_costs = nullptr) {
  require(!sources.empty(), "CPU reference requires a source");
  if (vertex_costs != nullptr) {
    require(vertex_costs->size() == static_cast<std::size_t>(graph.rows),
            "CPU reference vertex-cost size mismatch");
  }

  std::vector<float> distances(static_cast<std::size_t>(graph.rows), kInf);
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem,
                      std::vector<QueueItem>,
                      std::greater<QueueItem>>
      queue;

  for (const int source : sources) {
    require(source >= 0 && static_cast<Offset>(source) < graph.rows,
            "CPU reference source is outside the graph");
    if (distances[static_cast<std::size_t>(source)] != 0.0f) {
      distances[static_cast<std::size_t>(source)] = 0.0f;
      queue.push({0.0f, source});
    }
  }

  while (!queue.empty()) {
    const auto [distance, u] = queue.top();
    queue.pop();
    if (distance != distances[static_cast<std::size_t>(u)]) {
      continue;
    }

    for (Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const float candidate =
          distance + effective_edge_weight(graph, edge, vertex_costs);
      float& current = distances[static_cast<std::size_t>(v)];
      if (candidate < current) {
        current = candidate;
        queue.push({candidate, v});
      }
    }
  }
  return distances;
}

bool close_enough(float expected, float actual) {
  if (std::isinf(expected) || std::isinf(actual)) {
    return std::isinf(expected) && std::isinf(actual) &&
           std::signbit(expected) == std::signbit(actual);
  }
  if (!std::isfinite(expected) || !std::isfinite(actual)) {
    return false;
  }
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <=
         kAbsoluteTolerance + kRelativeTolerance * scale;
}

std::string distance_mismatch(const std::string& label,
                              std::size_t vertex,
                              float expected,
                              float actual) {
  std::ostringstream out;
  out << label << ": distance mismatch at vertex " << vertex
      << ", expected=" << expected << ", actual=" << actual;
  return out.str();
}

void validate_full_distances(const std::string& label,
                             const std::vector<float>& expected,
                             const DeltaSteppingCsrResult& result) {
  require(result.dist.size() == expected.size(),
          label + ": full-distance result has the wrong size");
  require(result.converged, label + ": unlimited full SSSP did not converge");
  require(result.target == -1, label + ": full SSSP target marker is not -1");
  for (std::size_t vertex = 0; vertex < expected.size(); ++vertex) {
    require(close_enough(expected[vertex], result.dist[vertex]),
            distance_mismatch(label,
                              vertex,
                              expected[vertex],
                              result.dist[vertex]));
  }
}

void validate_distances_only_result(
    const std::string& label,
    const std::vector<float>& expected,
    const DeltaSteppingCsrResult& result) {
  validate_full_distances(label, expected, result);
  require(result.pred_node.empty(),
          label + ": distances-only result contains predecessor nodes");
  require(result.pred_edge.empty(),
          label + ": distances-only result contains predecessor edges");
  require(result.target_distances.empty(),
          label + ": distances-only result contains target distances");
  require(result.target_sources.empty(),
          label + ": distances-only result contains target sources");
  require(result.target_path_offsets.empty(),
          label + ": distances-only result contains target path offsets");
  require(result.target_edge_offsets.empty(),
          label + ": distances-only result contains target edge offsets");
  require(result.target_path_nodes.empty(),
          label + ": distances-only result contains target path nodes");
  require(result.target_path_edges.empty(),
          label + ": distances-only result contains target path edges");
  require(std::isinf(result.target_distance) &&
              !std::signbit(result.target_distance),
          label + ": distances-only result changed the scalar target distance");
  require(!result.target_reached,
          label + ": distances-only result reports a reached target");
  require(!result.stopped_on_target,
          label + ": distances-only result reports a target stop");
}

void require_no_mutable_path_storage(
    const std::string& label,
    const DeltaSteppingCsrWorkspace& workspace) {
  const DeltaSteppingCsrAllocationState state = workspace.allocation_state();
  require(!state.parent_key,
          label + ": workspace retained compact parent keys");
  require(!state.predecessor_nodes,
          label + ": workspace retained predecessor nodes");
  require(!state.predecessor_edges,
          label + ": workspace retained predecessor edges");
  require(!state.target_storage,
          label + ": workspace retained target extraction storage");
  require(!state.path_nodes,
          label + ": workspace retained compact path nodes");
  require(!state.path_edges,
          label + ": workspace retained compact path edges");
}

void run_distances_and_check(
    const std::string& label,
    DeltaSteppingCsrWorkspace& workspace,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    float delta,
    hipStream_t stream,
    const std::vector<float>* vertex_costs = nullptr) {
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, sources, vertex_costs);
  const DeltaSteppingCsrResult result = workspace.run_distances(
      sources, delta, -1, stream, nullptr, nullptr);
  validate_distances_only_result(label, expected, result);
  require_no_mutable_path_storage(label, workspace);
}

bool contains_source(const std::vector<int>& sources, int vertex) {
  return std::find(sources.begin(), sources.end(), vertex) != sources.end();
}

void validate_compact_target_paths(
    const std::string& label,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<float>& expected_distances,
    const DeltaSteppingCsrResult& result,
    const std::vector<float>* vertex_costs = nullptr) {
  const std::size_t target_count = targets.size();
  require(result.target == -1, label + ": multi-target marker is not -1");
  require(result.target_distances.size() == target_count,
          label + ": target distance count mismatch");
  require(result.target_sources.size() == target_count,
          label + ": target source count mismatch");
  require(result.target_path_offsets.size() == target_count + 1,
          label + ": compact node-offset count mismatch");
  require(result.target_edge_offsets.size() == target_count + 1,
          label + ": compact edge-offset count mismatch");
  require(result.target_path_offsets.front() == 0,
          label + ": first compact node offset is not zero");
  require(result.target_edge_offsets.front() == 0,
          label + ": first compact edge offset is not zero");

  bool all_targets_reached = true;
  for (std::size_t target_index = 0; target_index < target_count;
       ++target_index) {
    std::ostringstream target_label_builder;
    target_label_builder << label << ": target[" << target_index
                         << "]=" << targets[target_index];
    const std::string target_label = target_label_builder.str();

    const int node_begin = result.target_path_offsets[target_index];
    const int node_end = result.target_path_offsets[target_index + 1];
    const int edge_begin = result.target_edge_offsets[target_index];
    const int edge_end = result.target_edge_offsets[target_index + 1];
    require(node_begin >= 0 && node_end >= node_begin,
            target_label + ": invalid node slice");
    require(edge_begin >= 0 && edge_end >= edge_begin,
            target_label + ": invalid edge slice");
    require(static_cast<std::size_t>(node_end) <=
                result.target_path_nodes.size(),
            target_label + ": node slice exceeds compact storage");
    require(static_cast<std::size_t>(edge_end) <=
                result.target_path_edges.size(),
            target_label + ": edge slice exceeds compact storage");

    const int target = targets[target_index];
    const float expected = expected_distances[static_cast<std::size_t>(target)];
    const float actual = result.target_distances[target_index];
    if (!std::isfinite(expected)) {
      all_targets_reached = false;
      require(std::isinf(actual),
              target_label + ": unreachable target distance is not infinity");
      require(result.target_sources[target_index] == -1,
              target_label + ": unreachable target has a source");
      require(node_begin == node_end,
              target_label + ": unreachable target has path nodes");
      require(edge_begin == edge_end,
              target_label + ": unreachable target has path edges");
      continue;
    }

    require(close_enough(expected, actual),
            target_label + ": compact target distance mismatch");
    require(node_end > node_begin,
            target_label + ": reached target has no path nodes");
    require(edge_end - edge_begin == node_end - node_begin - 1,
            target_label + ": compact node/edge counts disagree");
    const int source = result.target_sources[target_index];
    require(contains_source(sources, source),
            target_label + ": path begins at a non-source vertex");
    require(result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                source,
            target_label + ": source does not match first path node");
    require(result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                target,
            target_label + ": last path node is not the target");

    float path_distance = 0.0f;
    for (int edge_position = edge_begin; edge_position < edge_end;
         ++edge_position) {
      const int path_offset = edge_position - edge_begin;
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset + 1)];
      require(from >= 0 && static_cast<Offset>(from) < graph.rows &&
                  to >= 0 && static_cast<Offset>(to) < graph.rows,
              target_label + ": path vertex is outside the graph");

      const Offset edge = result.target_path_edges[
          static_cast<std::size_t>(edge_position)];
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from + 1)],
              target_label + ": returned edge ID is outside its outgoing row");
      require(graph.colind[static_cast<std::size_t>(edge)] == to,
              target_label + ": returned edge ID has the wrong destination");
      path_distance += effective_edge_weight(graph, edge, vertex_costs);
    }
    require(close_enough(expected, path_distance),
            target_label + ": path edge weights do not sum to CPU distance");
    require(close_enough(actual, path_distance),
            target_label + ": path edge weights do not sum to GPU distance");
  }

  require(static_cast<std::size_t>(result.target_path_offsets.back()) ==
              result.target_path_nodes.size(),
          label + ": final compact node offset mismatch");
  require(static_cast<std::size_t>(result.target_edge_offsets.back()) ==
              result.target_path_edges.size(),
          label + ": final compact edge offset mismatch");
  require(result.target_reached == all_targets_reached,
          label + ": aggregate target_reached flag mismatch");
}

void validate_parent_mode_agreement(
    const std::string& label,
    const DeltaSteppingCsrResult& automatic,
    const DeltaSteppingCsrResult& legacy) {
  require(automatic.target_distances.size() ==
              legacy.target_distances.size(),
          label + ": target distance count differs between parent modes");
  for (std::size_t i = 0; i < automatic.target_distances.size(); ++i) {
    require(close_enough(automatic.target_distances[i],
                         legacy.target_distances[i]),
            label + ": target distance differs between parent modes");
  }
  require(automatic.target_reached == legacy.target_reached,
          label + ": aggregate target status differs between parent modes");
  require(automatic.stopped_on_target == legacy.stopped_on_target,
          label + ": target-stop status differs between parent modes");
  require(automatic.converged == legacy.converged,
          label + ": convergence status differs between parent modes");
  require(automatic.iterations_used == legacy.iterations_used,
          label + ": bucket iteration count differs between parent modes");
}

void run_compact_parent_mode_comparison(
    const std::string& label,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    hipStream_t stream,
    const std::vector<float>* vertex_costs = nullptr,
    int max_iters = -1) {
  auto shared_graph =
      std::make_shared<DeltaSteppingCsrGraph>(graph, stream);
  DeltaSteppingCsrWorkspace automatic(shared_graph, stream);
  DeltaSteppingCsrWorkspace legacy(
      shared_graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
  if (vertex_costs != nullptr) {
    automatic.update_vertex_costs(*vertex_costs, stream);
    legacy.update_vertex_costs(*vertex_costs, stream);
  }

  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, sources, vertex_costs);
  const DeltaSteppingCsrResult automatic_result = automatic.run(
      sources, targets, delta, max_iters, stream, nullptr, nullptr);
  const DeltaSteppingCsrResult legacy_result = legacy.run(
      sources, targets, delta, max_iters, stream, nullptr, nullptr);
  validate_compact_target_paths(label + ": automatic",
                                graph,
                                sources,
                                targets,
                                expected,
                                automatic_result,
                                vertex_costs);
  validate_compact_target_paths(label + ": forced legacy",
                                graph,
                                sources,
                                targets,
                                expected,
                                legacy_result,
                                vertex_costs);
  validate_parent_mode_agreement(label, automatic_result, legacy_result);
  require(automatic_result.pred_node.empty() &&
              automatic_result.pred_edge.empty(),
          label + ": compact result unexpectedly materialized full predecessors");
}

void test_compact_edge_id_eligibility() {
  const std::uint64_t edge_id_count_limit = std::uint64_t{1} << 32;
  require(!delta_stepping_compact_edge_ids_eligible(
              static_cast<Offset>(-1)),
          "negative nnz was compact-edge eligible");
  require(delta_stepping_compact_edge_ids_eligible(0),
          "empty graph was not compact-edge eligible");
  require(delta_stepping_compact_edge_ids_eligible(1),
          "single-edge graph was not compact-edge eligible");
  require(delta_stepping_compact_edge_ids_eligible(
              static_cast<Offset>(edge_id_count_limit - 1)),
          "UINT32_MAX-edge graph was not compact-edge eligible");
  require(delta_stepping_compact_edge_ids_eligible(
              static_cast<Offset>(edge_id_count_limit)),
          "2^32-edge graph was not compact-edge eligible");
  require(!delta_stepping_compact_edge_ids_eligible(
              static_cast<Offset>(edge_id_count_limit + 1)),
          "graph above the 32-bit edge-ID boundary was eligible");
}

void test_braced_default_stream_constructor_compatibility() {
  const HostCsrF32 graph = make_outgoing_csr(1, {});
  DeltaSteppingCsrGraph shared_graph(graph, {});
  (void)shared_graph;
  DeltaSteppingCsrWorkspace workspace(graph, {});
  require(!workspace.allocation_state().predecessor_nodes &&
              !workspace.allocation_state().predecessor_edges,
          "default-stream construction eagerly allocated predecessors");
}

void validate_single_target_predecessors(
    const std::string& label,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    int target,
    float expected_distance,
    const DeltaSteppingCsrResult& result,
    const std::vector<float>* vertex_costs = nullptr) {
  require(std::isfinite(expected_distance),
          label + ": single-target validation requires a reachable target");
  require(result.target == target, label + ": single-target marker mismatch");
  require(result.target_reached, label + ": reachable target was not reached");
  require(result.stopped_on_target,
          label + ": reachable target did not trigger target stop");
  require(close_enough(expected_distance, result.target_distance),
          label + ": single-target distance mismatch");
  require(result.dist.size() == static_cast<std::size_t>(graph.rows),
          label + ": single-target full distance storage is missing");
  require(result.pred_node.size() == static_cast<std::size_t>(graph.rows),
          label + ": predecessor-node storage is missing");
  require(result.pred_edge.size() == static_cast<std::size_t>(graph.rows),
          label + ": predecessor-edge storage is missing");

  float path_distance = 0.0f;
  int current = target;
  for (Offset guard = 0; guard <= graph.rows; ++guard) {
    const int predecessor = result.pred_node[static_cast<std::size_t>(current)];
    if (predecessor == current ||
        (predecessor < 0 && contains_source(sources, current))) {
      require(contains_source(sources, current),
              label + ": predecessor chain ended at a non-source");
      require(close_enough(expected_distance, path_distance),
              label + ": predecessor edge weights do not sum to target distance");
      return;
    }
    require(predecessor >= 0 &&
                static_cast<Offset>(predecessor) < graph.rows &&
                predecessor != current,
            label + ": predecessor chain contains an invalid predecessor");
    const Offset edge = result.pred_edge[static_cast<std::size_t>(current)];
    require(edge >= graph.rowptr[static_cast<std::size_t>(predecessor)] &&
                edge < graph.rowptr[static_cast<std::size_t>(predecessor + 1)],
            label + ": predecessor edge is outside its outgoing row");
    require(graph.colind[static_cast<std::size_t>(edge)] == current,
            label + ": predecessor edge has the wrong destination");
    path_distance += effective_edge_weight(graph, edge, vertex_costs);
    current = predecessor;
  }
  fail(label + ": predecessor chain contains a cycle");
}

void run_full_and_target_checks(
    const std::string& label,
    DeltaSteppingCsrWorkspace& workspace,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    hipStream_t stream,
    const std::vector<float>* vertex_costs = nullptr,
    bool rerun_full_after_targets = false) {
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, sources, vertex_costs);

  const DeltaSteppingCsrResult full =
      workspace.run(sources, -1, delta, -1, stream, nullptr, nullptr);
  validate_full_distances(label + ": full", expected, full);

  const DeltaSteppingCsrResult targeted =
      workspace.run(sources, targets, delta, -1, stream, nullptr, nullptr);
  validate_compact_target_paths(label + ": compact",
                                graph,
                                sources,
                                targets,
                                expected,
                                targeted,
                                vertex_costs);
  const bool all_targets_reachable =
      std::all_of(targets.begin(), targets.end(), [&](int target) {
        return std::isfinite(expected[static_cast<std::size_t>(target)]);
      });
  if (all_targets_reachable) {
    require(targeted.stopped_on_target && !targeted.converged,
            label + ": reachable target set did not stop early");
  } else {
    require(targeted.converged && !targeted.stopped_on_target,
            label + ": unreachable target set did not exhaust the graph");
  }

  if (rerun_full_after_targets) {
    const DeltaSteppingCsrResult repeated =
        workspace.run(sources, -1, delta, -1, stream, nullptr, nullptr);
    validate_full_distances(label + ": repeated full", expected, repeated);
  }
}

HostCsrF32 make_weighted_corner_graph() {
  // Rows are intentionally supplied out of column order and include parallel
  // edges, harmless self-loops, a one-way zero edge, and disconnected nodes.
  return make_outgoing_csr(
      12,
      {{0, 2, 5.0f},
       {0, 1, 2.0f},
       {1, 3, 2.0f},
       {1, 2, 1.0f},
       {1, 3, 2.5f},
       {2, 5, 4.0f},
       {2, 3, 0.5f},
       {2, 2, 0.0f},
       {3, 4, 3.0f},
       {4, 7, 0.25f},
       {4, 5, 0.75f},
       {5, 6, 0.0f},
       {6, 7, 1.0f},
       {7, 7, 0.0f},
       {8, 7, 2.0f},
       {10, 11, 1.0f},
       {11, 10, 1.5f}});
}

void test_distances_only_graph_families(hipStream_t stream) {
  {
    const HostCsrF32 graph = make_outgoing_csr(1, {});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    const std::vector<float> expected =
        cpu_dijkstra_outgoing_multi_source(graph, {0});
    const DeltaSteppingCsrResult result = workspace.run_distances(
        0, 1.0f, -1, stream, nullptr, nullptr);
    validate_distances_only_result(
        "distances-only zero-edge singleton", expected, result);
    require_no_mutable_path_storage(
        "distances-only zero-edge singleton", workspace);
  }

  {
    const HostCsrF32 graph = make_outgoing_csr(6, {});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    run_distances_and_check("distances-only isolated vertices",
                            workspace,
                            graph,
                            {0, 4, 0},
                            0.5f,
                            stream);
    run_distances_and_check("distances-only isolated vertices reused",
                            workspace,
                            graph,
                            {2},
                            2.0f,
                            stream);
  }

  {
    const HostCsrF32 graph = make_outgoing_csr(
        8,
        {{0, 1, 1.0f},
         {0, 2, 1.0f},
         {1, 3, 1.0f},
         {2, 3, 1.0f},
         {3, 4, 1.0f},
         {4, 5, 1.0f},
         {6, 5, 1.0f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    run_distances_and_check("distances-only unit graph",
                            workspace,
                            graph,
                            {0, 0},
                            1.0f,
                            stream);
    require(std::isinf(cpu_dijkstra_outgoing_multi_source(graph, {0})[6]),
            "distances-only unit fixture lost its unreachable vertex");
  }

  {
    const HostCsrF32 graph = make_weighted_corner_graph();
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // This fixture contains heterogeneous and zero-weight edges, an unsorted
    // parallel edge pair, self-loops, and isolated/disconnected components.
    run_distances_and_check(
        "distances-only weighted zero/parallel graph",
        workspace,
        graph,
        {0, 0},
        1.0f,
        stream);
    run_distances_and_check(
        "distances-only weighted zero/parallel graph reused",
        workspace,
        graph,
        {8, 10, 8},
        0.75f,
        stream);

    const std::vector<float> vertex_costs = {
        1.0f, 2.0f, 0.5f, 1.5f, 0.0f, 3.0f,
        0.25f, 2.0f, 1.0f, 1.0f, 1.25f, 0.75f};
    workspace.update_vertex_costs(vertex_costs, stream);
    run_distances_and_check(
        "distances-only destination vertex costs",
        workspace,
        graph,
        {0, 10, 0},
        1.25f,
        stream,
        &vertex_costs);
  }
}

void test_distances_only_path_state_transitions(hipStream_t stream) {
  const HostCsrF32 weighted_graph = make_weighted_corner_graph();
  HostCsrF32 active_graph = weighted_graph;
  const std::vector<int> sources = {0, 8, 0};
  const std::vector<int> targets = {7, 4, 0, 8, 9, 7};
  DeltaSteppingCsrWorkspace workspace(weighted_graph, stream);

  const std::vector<float> weighted_expected =
      cpu_dijkstra_outgoing_multi_source(weighted_graph, sources);
  const DeltaSteppingCsrResult first_path = workspace.run(
      sources, targets, 1.0f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("distances-only transition initial compact",
                                weighted_graph,
                                sources,
                                targets,
                                weighted_expected,
                                first_path);
  {
    const DeltaSteppingCsrAllocationState state = workspace.allocation_state();
    require(state.edge_source,
            "initial compact run did not retain the immutable edge map");
    require(state.parent_key,
            "initial compact run did not allocate compact parent keys");
    require(!state.predecessor_nodes && !state.predecessor_edges,
            "initial compact run unexpectedly allocated legacy predecessors");
    require(state.target_storage && state.path_nodes && state.path_edges,
            "initial compact run did not allocate target/path storage");
  }

  run_distances_and_check("distances-only after compact path",
                          workspace,
                          weighted_graph,
                          sources,
                          1.0f,
                          stream);
  require(workspace.allocation_state().edge_source,
          "path-capable distance run released the immutable edge map");

  active_graph.values.assign(active_graph.values.size(), 1.0f);
  workspace.update_values(active_graph.values, stream);
  const std::vector<float> unit_expected =
      cpu_dijkstra_outgoing_multi_source(active_graph, sources);
  DeltaSteppingCsrTelemetry exact_telemetry;
  const DeltaSteppingCsrResult exact_path = workspace.run(
      sources,
      targets,
      1.0f,
      -1,
      DeltaSteppingCsrRunOptions{&exact_telemetry},
      stream,
      nullptr,
      nullptr);
  validate_compact_target_paths("distances-only transition exact unit",
                                active_graph,
                                sources,
                                targets,
                                unit_expected,
                                exact_path);
  require(exact_telemetry.collected && exact_telemetry.completed &&
              exact_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit,
          "path after distances-only did not re-enter the exact-unit path");
  {
    const DeltaSteppingCsrAllocationState state = workspace.allocation_state();
    require(!state.parent_key,
            "exact-unit transition unexpectedly allocated parent keys");
    require(state.predecessor_nodes && state.predecessor_edges,
            "exact-unit transition did not reallocate predecessors");
    require(state.target_storage && state.path_nodes && state.path_edges,
            "exact-unit transition did not reallocate target/path storage");
  }

  DeltaSteppingCsrTelemetry distances_telemetry;
  const DeltaSteppingCsrResult unit_distances = workspace.run_distances(
      sources,
      1.0f,
      -1,
      DeltaSteppingCsrRunOptions{&distances_telemetry},
      stream,
      nullptr,
      nullptr);
  validate_distances_only_result("distances-only after exact unit",
                                 unit_expected,
                                 unit_distances);
  require_no_mutable_path_storage(
      "distances-only after exact unit", workspace);
  require(distances_telemetry.collected && distances_telemetry.completed,
          "distances-only telemetry was not collected and completed");
  require(distances_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kGenericDistancesOnly,
          "distances-only telemetry reported the wrong execution path");
  require(std::string(delta_stepping_execution_path_name(
              distances_telemetry.execution_path)) ==
              "generic_distances_only",
          "distances-only execution-path name is unstable");
  require(close_enough(1.0f, distances_telemetry.resolved_delta),
          "distances-only telemetry reported the wrong delta");
  require(!distances_telemetry.force_legacy_parent,
          "distances-only telemetry reported forced legacy parents");
  require(distances_telemetry.wavefront_size == 32 ||
              distances_telemetry.wavefront_size == 64,
          "distances-only telemetry reported an unsupported wavefront size");

  workspace.update_values(weighted_graph.values, stream);
  const DeltaSteppingCsrResult restored_path = workspace.run(
      sources, targets, 1.0f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("distances-only transition restored compact",
                                weighted_graph,
                                sources,
                                targets,
                                weighted_expected,
                                restored_path);
  {
    const DeltaSteppingCsrAllocationState state = workspace.allocation_state();
    require(state.parent_key && state.target_storage && state.path_nodes &&
                state.path_edges,
            "compact path state was not reallocated after distances-only");
    require(!state.predecessor_nodes && !state.predecessor_edges,
            "restored compact run unexpectedly retained exact-unit predecessors");
  }
  run_distances_and_check("distances-only after restored compact",
                          workspace,
                          weighted_graph,
                          {10},
                          0.75f,
                          stream);
}

void test_distances_only_strict_storage_and_validation(hipStream_t stream) {
  const HostCsrF32 graph = make_weighted_corner_graph();
  const std::vector<int> sources = {0, 8, 0};
  const std::vector<int> targets = {4, 7};
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, sources);

  DeltaSteppingCsrWorkspace owned(
      graph, stream, DeltaSteppingCsrStorageMode::kDistancesOnly);
  require_no_mutable_path_storage("strict owned initial storage", owned);
  require(!owned.allocation_state().edge_source,
          "strict owned workspace retained the immutable edge map");

  DeltaSteppingCsrTelemetry telemetry;
  const DeltaSteppingCsrResult first = owned.run_distances(
      sources,
      1.25f,
      -1,
      DeltaSteppingCsrRunOptions{&telemetry},
      stream,
      nullptr,
      nullptr);
  validate_distances_only_result("strict owned distances", expected, first);
  require_no_mutable_path_storage("strict owned distances", owned);
  require(!owned.allocation_state().edge_source,
          "strict owned run allocated the immutable edge map");
  require(telemetry.collected && telemetry.completed &&
              telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kGenericDistancesOnly,
          "strict owned run reported the wrong telemetry mode");

  const std::vector<float> vertex_costs = {
      1.0f, 2.0f, 0.5f, 1.5f, 0.0f, 3.0f,
      0.25f, 2.0f, 1.0f, 1.0f, 1.25f, 0.75f};
  owned.update_vertex_costs(vertex_costs, stream);
  run_distances_and_check("strict owned destination costs reused",
                          owned,
                          graph,
                          {10},
                          0.75f,
                          stream,
                          &vertex_costs);
  require(!owned.allocation_state().edge_source,
          "strict owned reused run allocated the immutable edge map");

  require_invalid_argument("strict owned full path API", [&] {
    (void)owned.run(sources, -1, 1.25f, -1, stream, nullptr, nullptr);
  });
  require_invalid_argument("strict owned scalar-target path API", [&] {
    (void)owned.run(sources, 7, 1.25f, -1, stream, nullptr, nullptr);
  });
  require_invalid_argument("strict owned vector-target path API", [&] {
    (void)owned.run(sources, targets, 1.25f, -1, stream, nullptr, nullptr);
  });
  require_no_mutable_path_storage("strict owned rejected path APIs", owned);
  require(!owned.allocation_state().edge_source,
          "strict owned rejected path API allocated the immutable edge map");

  auto strict_graph = std::make_shared<DeltaSteppingCsrGraph>(
      graph, stream, DeltaSteppingCsrStorageMode::kDistancesOnly);
  DeltaSteppingCsrWorkspace shared(strict_graph, stream);
  require_no_mutable_path_storage("strict shared initial storage", shared);
  require(!shared.allocation_state().edge_source,
          "strict shared graph retained the immutable edge map");
  run_distances_and_check("strict shared distances",
                          shared,
                          graph,
                          sources,
                          1.25f,
                          stream);
  require(!shared.allocation_state().edge_source,
          "strict shared run allocated the immutable edge map");
  require_invalid_argument("strict shared vector-target path API", [&] {
    (void)shared.run(sources, targets, 1.25f, -1, stream, nullptr, nullptr);
  });

  DeltaSteppingCsrWorkspace forced_legacy(
      graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
  require_invalid_argument("distances-only forced legacy", [&] {
    (void)forced_legacy.run_distances(
        sources, 1.25f, -1, stream, nullptr, nullptr);
  });
  require_no_mutable_path_storage(
      "distances-only forced legacy rejection", forced_legacy);
}

void test_runtime_telemetry_modes_and_reset(hipStream_t stream) {
  const HostCsrF32 unit_graph = make_outgoing_csr(
      5,
      {{0, 1, 1.0f},
       {0, 2, 1.0f},
       {1, 3, 1.0f},
       {2, 3, 1.0f},
       {3, 4, 1.0f}});
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {4};
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(unit_graph, sources);

  DeltaSteppingCsrWorkspace automatic(unit_graph, stream);
  require(!automatic.allocation_state().telemetry_counters,
          "disabled telemetry allocated counters at construction");
  const DeltaSteppingCsrResult disabled = automatic.run(
      sources, targets, 1.0f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("telemetry disabled exact unit",
                                unit_graph,
                                sources,
                                targets,
                                expected,
                                disabled);
  require(!automatic.allocation_state().telemetry_counters,
          "disabled telemetry allocated device counters");

  DeltaSteppingCsrTelemetry exact;
  exact.outer_buckets_processed = UINT64_C(9999);
  exact.stale_frontier_entries = UINT64_C(9999);
  const DeltaSteppingCsrResult instrumented = automatic.run(
      sources,
      targets,
      1.0f,
      -1,
      DeltaSteppingCsrRunOptions{&exact},
      stream,
      nullptr,
      nullptr);
  validate_compact_target_paths("telemetry exact unit",
                                unit_graph,
                                sources,
                                targets,
                                expected,
                                instrumented);
  require(exact.collected && exact.completed,
          "exact-unit telemetry did not complete");
  require(exact.execution_path ==
              DeltaSteppingCsrExecutionPath::kExactUnit,
          "exact-unit telemetry reported the wrong path");
  require(exact.reached_vertices == 5,
          "exact-unit telemetry reached count mismatch");
  require(exact.successful_distance_relaxations == 4 &&
              exact.current_queue_insertions == 4,
          "exact-unit telemetry relaxation/queue counts mismatch");
  require(exact.current_queue_high_water == 5,
          "exact-unit telemetry queue peak mismatch");
  require(exact.frontier_entries_processed ==
              exact.active_vertices_processed &&
              exact.stale_frontier_entries == 0,
          "exact-unit telemetry frontier accounting mismatch");
  require(exact.distance_atomic_attempts >=
              exact.successful_distance_relaxations,
          "exact-unit telemetry atomic accounting is inconsistent");
  require(automatic.allocation_state().telemetry_counters,
          "enabled telemetry did not allocate counters");

  // Reuse the same output record for a zero-traversal source-target query.
  // Every prior counter must be reset rather than accumulated.
  exact.light_edge_visits = UINT64_C(8888);
  const DeltaSteppingCsrResult reset_run = automatic.run(
      sources,
      std::vector<int>{0},
      1.0f,
      -1,
      DeltaSteppingCsrRunOptions{&exact},
      stream,
      nullptr,
      nullptr);
  validate_compact_target_paths("telemetry reset exact unit",
                                unit_graph,
                                sources,
                                std::vector<int>{0},
                                expected,
                                reset_run);
  require(exact.collected && exact.completed &&
              exact.outer_buckets_processed == 0 &&
              exact.light_relaxation_rounds == 0 &&
              exact.frontier_entries_processed == 0 &&
              exact.light_edge_visits == 0 &&
              exact.successful_distance_relaxations == 0 &&
              exact.reached_vertices == 1 &&
              exact.current_queue_high_water == 1,
          "reused telemetry record was not reset deterministically");

  DeltaSteppingCsrWorkspace forced_generic(
      unit_graph,
      stream,
      DeltaSteppingCsrWorkspaceOptions{
          DeltaSteppingCsrParentMode::kAutomatic,
          DeltaSteppingCsrExecutionMode::kForceGeneric});
  DeltaSteppingCsrTelemetry generic;
  const DeltaSteppingCsrResult generic_result = forced_generic.run(
      sources, targets, 1.0f, -1,
      DeltaSteppingCsrRunOptions{&generic}, stream, nullptr, nullptr);
  validate_compact_target_paths("telemetry forced generic",
                                unit_graph,
                                sources,
                                targets,
                                expected,
                                generic_result);
  require(generic.collected && generic.completed && generic.force_generic &&
              generic.execution_path ==
                  DeltaSteppingCsrExecutionPath::kCompactGeneric,
          "forced-generic telemetry reported the wrong mode");

  DeltaSteppingCsrWorkspace forced_legacy(
      unit_graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
  DeltaSteppingCsrTelemetry legacy;
  const DeltaSteppingCsrResult legacy_result = forced_legacy.run(
      sources, targets, 1.0f, -1,
      DeltaSteppingCsrRunOptions{&legacy}, stream, nullptr, nullptr);
  validate_compact_target_paths("telemetry forced legacy",
                                unit_graph,
                                sources,
                                targets,
                                expected,
                                legacy_result);
  require(legacy.collected && legacy.completed &&
              legacy.force_legacy_parent &&
              legacy.execution_path ==
                  DeltaSteppingCsrExecutionPath::kLegacyGeneric,
          "forced-legacy telemetry reported the wrong mode");

  const HostCsrF32 weighted = make_outgoing_csr(
      5,
      {{0, 1, 0.25f},
       {0, 2, 4.0f},
       {1, 2, 0.25f},
       {1, 3, 3.0f},
       {2, 3, 0.5f},
       {3, 4, 2.0f}});
  const std::vector<float> weighted_expected =
      cpu_dijkstra_outgoing_multi_source(weighted, sources);
  DeltaSteppingCsrWorkspace weighted_workspace(weighted, stream);
  DeltaSteppingCsrTelemetry weighted_record;
  const DeltaSteppingCsrResult weighted_result = weighted_workspace.run(
      sources, targets, 1.0f, -1,
      DeltaSteppingCsrRunOptions{&weighted_record}, stream, nullptr, nullptr);
  validate_compact_target_paths("telemetry weighted generic",
                                weighted,
                                sources,
                                targets,
                                weighted_expected,
                                weighted_result);
  require(weighted_record.execution_path ==
              DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              !weighted_record.all_edges_light &&
              weighted_record.heavy_edge_phases > 0 &&
              weighted_record.light_edge_visits > 0 &&
              weighted_record.heavy_edge_visits > 0 &&
              weighted_record.distance_atomic_attempts >=
                  weighted_record.successful_distance_relaxations,
          "weighted telemetry counters are inconsistent");
}

void test_generic_compact_parent_modes(hipStream_t stream) {
  const HostCsrF32 all_light = make_outgoing_csr(
      6,
      {{0, 2, 1.5f},
       {0, 1, 0.25f},
       {0, 1, 0.5f},
       {1, 3, 0.75f},
       {2, 3, 0.25f},
       {3, 4, 1.25f}});
  run_compact_parent_mode_comparison("generic weighted all-light",
                                     all_light,
                                     {0, 0},
                                     {0, 4, 3, 5, 4},
                                     2.0f,
                                     stream);

  const HostCsrF32 all_heavy = make_outgoing_csr(
      6,
      {{0, 2, 1.25f},
       {0, 1, 0.75f},
       {1, 3, 0.75f},
       {2, 3, 1.5f},
       {3, 4, 0.75f}});
  run_compact_parent_mode_comparison("generic weighted all-heavy",
                                     all_heavy,
                                     {0},
                                     {4, 3, 0, 5},
                                     0.5f,
                                     stream);

  // Row zero deliberately mixes light/heavy work, keeps parallel edges, and
  // offers three equal-distance ways to reach vertex four. Parent modes may
  // choose different valid original edges; only distance/path validity is
  // compared.
  const HostCsrF32 mixed = make_outgoing_csr(
      6,
      {{0, 4, 1.25f},
       {0, 2, 0.75f},
       {0, 1, 0.25f},
       {0, 3, 2.5f},
       {0, 1, 0.25f},
       {1, 4, 1.0f},
       {2, 4, 0.5f},
       {3, 4, 0.25f}});
  run_compact_parent_mode_comparison("generic weighted mixed rows",
                                     mixed,
                                     {0, 0},
                                     {4, 4, 0, 5, 2},
                                     1.0f,
                                     stream);

  const HostCsrF32 cost_graph = make_outgoing_csr(
      6,
      {{0, 2, 0.75f},
       {0, 1, 0.75f},
       {1, 3, 0.75f},
       {2, 3, 0.75f},
       {1, 4, 0.75f},
       {3, 4, 0.75f}});
  std::vector<float> vertex_costs(
      static_cast<std::size_t>(cost_graph.rows), 1.0f);
  vertex_costs[1] = 0.5f;
  vertex_costs[2] = 2.0f;
  vertex_costs[3] = 2.0f;
  vertex_costs[4] = 0.25f;
  run_compact_parent_mode_comparison(
      "vertex costs reclassify light and heavy edges",
      cost_graph,
      {0},
      {4, 3, 2, 5, 0},
      1.0f,
      stream,
      &vertex_costs);
}

void test_compact_early_settlement_reset(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      5,
      {{0, 1, 0.25f},
       {0, 2, 5.0f},
       {2, 3, 0.25f},
       {3, 4, 0.25f}});
  DeltaSteppingCsrWorkspace workspace(graph, stream);
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, {0});

  const DeltaSteppingCsrResult early = workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("compact early settlement",
                                graph,
                                {0},
                                {1},
                                expected,
                                early);
  require(early.stopped_on_target && !early.converged,
          "compact early-settlement query did not stop on its target");

  // The first bucket also discovers vertex two in a future bucket. Repeating
  // from the same source requires reduced reset to clear that live pending
  // membership bit; otherwise vertex two cannot be enqueued again.
  const std::vector<int> later_targets = {3, 4, 1};
  const DeltaSteppingCsrResult repeated = workspace.run(
      std::vector<int>{0}, later_targets, 1.0f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("compact after early-settlement reset",
                                graph,
                                {0},
                                later_targets,
                                expected,
                                repeated);
  require(repeated.stopped_on_target,
          "repeated compact query lost pending work after early settlement");

  const HostCsrF32 all_light_graph = make_outgoing_csr(
      5,
      {{0, 1, 0.25f},
       {0, 2, 0.75f},
       {2, 3, 0.75f},
       {3, 4, 0.25f}});
  DeltaSteppingCsrWorkspace all_light_workspace(all_light_graph, stream);
  const std::vector<float> all_light_expected =
      cpu_dijkstra_outgoing_multi_source(all_light_graph, {0});
  const DeltaSteppingCsrResult all_light_early = all_light_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("all-light compact early settlement",
                                all_light_graph,
                                {0},
                                {1},
                                all_light_expected,
                                all_light_early);
  require(all_light_early.stopped_on_target && !all_light_early.converged,
          "all-light compact query did not stop with future work queued");

  // All individual edges are light, but the cumulative 0->2->3 distance is
  // in a future bucket. This repeats the live-pending reset check through the
  // specialization that intentionally omits the in_heavy reset store.
  const std::vector<int> all_light_later_targets = {3, 4};
  const DeltaSteppingCsrResult all_light_repeated = all_light_workspace.run(
      std::vector<int>{0}, all_light_later_targets, 1.0f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("all-light compact pending reset",
                                all_light_graph,
                                {0},
                                all_light_later_targets,
                                all_light_expected,
                                all_light_repeated);
  require(all_light_repeated.stopped_on_target,
          "all-light compact repeat lost future-bucket pending work");
}

void test_forced_generic_increasing_distance_reuse(hipStream_t stream) {
  // Alternate a one-edge path with a five-edge path to the same target. This
  // makes every other sparse-reset reuse replace a small finite distance and
  // compact parent chain with a larger one, covering the production pattern
  // implicated in mixed-generation measurement state on gfx1151 streams.
  const HostCsrF32 graph = make_outgoing_csr(
      7,
      {{0, 6, 1.0f},
       {1, 2, 1.0f},
       {2, 3, 1.0f},
       {3, 4, 1.0f},
       {4, 5, 1.0f},
       {5, 6, 1.0f}});
  const DeltaSteppingCsrWorkspaceOptions options{
      DeltaSteppingCsrParentMode::kAutomatic,
      DeltaSteppingCsrExecutionMode::kForceGeneric};
  DeltaSteppingCsrWorkspace workspace(graph, stream, options);
  const std::vector<float> short_expected =
      cpu_dijkstra_outgoing_multi_source(graph, {0});
  const std::vector<float> long_expected =
      cpu_dijkstra_outgoing_multi_source(graph, {1});

  constexpr int kReuseRuns = 128;
  for (int repetition = 0; repetition < kReuseRuns; ++repetition) {
    const bool use_long_path = (repetition & 1) != 0;
    const std::vector<int> sources = use_long_path
                                         ? std::vector<int>{1}
                                         : std::vector<int>{0};
    const DeltaSteppingCsrResult result = workspace.run(
        sources, std::vector<int>{6}, 1.0f, -1,
        stream, nullptr, nullptr);
    validate_compact_target_paths(
        "forced-generic increasing-distance reuse " +
            std::to_string(repetition),
        graph,
        sources,
        {6},
        use_long_path ? long_expected : short_expected,
        result);
  }
}

void test_compact_weight_class_updates(hipStream_t stream) {
  HostCsrF32 graph = make_outgoing_csr(
      6,
      {{0, 1, 0.25f},
       {0, 2, 0.75f},
       {1, 3, 0.5f},
       {2, 3, 0.25f},
       {3, 4, 0.5f},
       {4, 5, 0.25f}});
  DeltaSteppingCsrWorkspace workspace(graph, stream);
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {5, 4, 0, 5};
  auto validate_current = [&](const std::string& label,
                              const std::vector<float>* vertex_costs) {
    const DeltaSteppingCsrResult result = workspace.run(
        sources, targets, 1.0f, -1, stream, nullptr, nullptr);
    validate_compact_target_paths(
        label,
        graph,
        sources,
        targets,
        cpu_dijkstra_outgoing_multi_source(graph, sources, vertex_costs),
        result,
        vertex_costs);
  };

  validate_current("updated topology map all-light", nullptr);

  graph.values = {2.0f, 0.25f, 1.5f, 0.25f, 0.5f, 2.0f};
  workspace.update_values(graph.values, stream);
  validate_current("updated topology map mixed-heavy", nullptr);

  graph.values = {0.375f, 0.625f, 0.5f, 0.25f, 0.75f, 0.5f};
  workspace.update_values(graph.values, stream);
  validate_current("updated topology map back to all-light", nullptr);

  std::vector<float> vertex_costs(
      static_cast<std::size_t>(graph.rows), 1.0f);
  vertex_costs[1] = 4.0f;
  vertex_costs[3] = 3.0f;
  vertex_costs[5] = 2.5f;
  workspace.update_vertex_costs(vertex_costs, stream);
  validate_current("updated topology map with vertex costs", &vertex_costs);
}

void test_compact_to_unit_storage_transition(hipStream_t stream) {
  HostCsrF32 graph = make_weighted_corner_graph();
  DeltaSteppingCsrWorkspace workspace(graph, stream);
  const std::vector<int> sources = {0, 8, 0};
  const std::vector<int> targets = {4, 7, 0, 8, 7};

  const DeltaSteppingCsrResult compact_result = workspace.run(
      sources, targets, 1.25f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths(
      "compact storage before unit specialization",
      graph,
      sources,
      targets,
      cpu_dijkstra_outgoing_multi_source(graph, sources),
      compact_result);

  // The compact-parent run above deliberately never needs legacy predecessor
  // buffers. Switching the same workspace to exact unit weights must allocate
  // them lazily before the append-only specialization writes predecessor data.
  std::fill(graph.values.begin(), graph.values.end(), 1.0f);
  workspace.update_values(graph.values, stream);
  const DeltaSteppingCsrResult unit_result = workspace.run(
      sources, targets, 0.5f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths(
      "unit specialization after compact-only storage",
      graph,
      sources,
      targets,
      cpu_dijkstra_outgoing_multi_source(graph, sources),
      unit_result);
}

void test_empty_and_singleton_graphs(hipStream_t stream) {
  {
    const HostCsrF32 graph = make_outgoing_csr(1, {});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    run_full_and_target_checks("singleton",
                               workspace,
                               graph,
                               {0, 0},
                               {0, 0},
                               1.0f,
                               stream,
                               nullptr,
                               true);
  }
  {
    const HostCsrF32 graph = make_outgoing_csr(5, {});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    run_full_and_target_checks("empty edges",
                               workspace,
                               graph,
                               {1, 1},
                               {1, 0, 4, 1},
                               0.25f,
                               stream,
                               nullptr,
                               true);
    run_compact_parent_mode_comparison(
        "empty-edge compact parent modes",
        graph,
        {1, 1},
        {1, 0, 4, 1},
        0.25f,
        stream,
        nullptr,
        std::numeric_limits<int>::max());
  }
}

HostCsrF32 make_mixed_claim_unit_graph() {
  // A full frontier wave has zero through six outgoing edges per lane. Each
  // nonempty row first claims one fresh vertex and then revisits the source,
  // exercising differing adjacency-loop trip counts and mixed claim results.
  std::vector<EdgeSpec> edges;
  edges.reserve(320);
  for (int destination = 1; destination <= 64; ++destination) {
    edges.push_back({0, destination, 1.0f});
  }
  for (int vertex = 1; vertex <= 64; ++vertex) {
    const int degree = vertex % 7;
    if (degree > 0) {
      edges.push_back({vertex, 64 + vertex, 1.0f});
      for (int edge = 1; edge < degree; ++edge) {
        edges.push_back({vertex, 0, 1.0f});
      }
    }
  }
  for (int vertex = 65; vertex <= 128; ++vertex) {
    edges.push_back({vertex, 129, 1.0f});
  }
  return make_outgoing_csr(131, edges);
}

constexpr int kDeepWideLevels = 12;
constexpr int kDeepWideWidth = 4097;
constexpr int kDeepWideTarget = kDeepWideLevels * kDeepWideWidth;
constexpr int kDeepWideIsolated = kDeepWideTarget + 1;
static_assert(kDeepWideLevels > 8, "deep regression must exceed eight levels");
static_assert(kDeepWideWidth > 4096,
              "wide regression must span many 256-thread blocks");

HostCsrF32 make_deep_wide_layered_graph(float edge_weight) {
  // Keep 4,097 vertices live across twelve levels.  This spans seventeen
  // 256-thread blocks per level, reaches three four-round batches on the
  // default stream, and remains small enough for eight private workspaces.
  std::vector<EdgeSpec> edges;
  edges.reserve(static_cast<std::size_t>(kDeepWideLevels) * kDeepWideWidth);
  for (int lane = 0; lane < kDeepWideWidth; ++lane) {
    edges.push_back({0, lane + 1, edge_weight});
  }
  for (int level = 1; level < kDeepWideLevels; ++level) {
    const int level_begin = (level - 1) * kDeepWideWidth + 1;
    for (int lane = 0; lane < kDeepWideWidth; ++lane) {
      const int from = level_begin + lane;
      edges.push_back({from, from + kDeepWideWidth, edge_weight});
    }
  }
  return make_outgoing_csr(kDeepWideIsolated + 1, edges);
}

void validate_deep_wide_result(const std::string& label,
                               const HostCsrF32& graph,
                               const std::vector<float>& expected,
                               int target,
                               const DeltaSteppingCsrResult& result) {
  validate_compact_target_paths(
      label, graph, {0}, {target}, expected, result);
  if (target == kDeepWideTarget) {
    require(result.target_reached && result.stopped_on_target &&
                !result.converged,
            label + ": reachable deep target did not stop the traversal");
  } else {
    require(!result.target_reached && !result.stopped_on_target &&
                result.converged,
            label + ": isolated target did not exhaust the traversal");
  }
}

void test_default_stream_deep_wide_batching() {
  // Exact unit weights select Delta's append-only Unit-BFS specialization.
  // A null stream and no callback are the production batching preconditions.
  const HostCsrF32 unit_graph = make_deep_wide_layered_graph(1.0f);
  const std::vector<float> unit_expected =
      cpu_dijkstra_outgoing_multi_source(unit_graph, {0});
  DeltaSteppingCsrWorkspace unit_workspace(unit_graph, nullptr);
  validate_deep_wide_result(
      "default-stream deep-wide exact-unit batching",
      unit_graph,
      unit_expected,
      kDeepWideTarget,
      unit_workspace.run({0},
                         std::vector<int>{kDeepWideTarget},
                         4.0f,
                         -1,
                         nullptr,
                         nullptr,
                         nullptr));
  validate_deep_wide_result(
      "default-stream deep-wide exact-unit exhaustion",
      unit_graph,
      unit_expected,
      kDeepWideIsolated,
      unit_workspace.run({0},
                         std::vector<int>{kDeepWideIsolated},
                         4.0f,
                         -1,
                         nullptr,
                         nullptr,
                         nullptr));

  // Zero-weight edges force generic Delta-Stepping while keeping all twelve
  // levels inside one light-edge closure, so its four-round batching boundary
  // is exercised independently of the exact-unit specialization.
  const HostCsrF32 closure_graph = make_deep_wide_layered_graph(0.0f);
  const std::vector<float> closure_expected =
      cpu_dijkstra_outgoing_multi_source(closure_graph, {0});
  DeltaSteppingCsrWorkspace closure_workspace(closure_graph, nullptr);
  validate_deep_wide_result(
      "default-stream deep-wide generic closure batching",
      closure_graph,
      closure_expected,
      kDeepWideTarget,
      closure_workspace.run({0},
                            std::vector<int>{kDeepWideTarget},
                            1.0f,
                            -1,
                            nullptr,
                            nullptr,
                            nullptr));
  validate_deep_wide_result(
      "default-stream deep-wide generic closure exhaustion",
      closure_graph,
      closure_expected,
      kDeepWideIsolated,
      closure_workspace.run({0},
                            std::vector<int>{kDeepWideIsolated},
                            1.0f,
                            -1,
                            nullptr,
                            nullptr,
                            nullptr));
}

void test_parallel_deep_wide_explicit_streams() {
  const HostCsrF32 unit_graph = make_deep_wide_layered_graph(1.0f);
  const HostCsrF32 closure_graph = make_deep_wide_layered_graph(0.0f);
  const HostCsrF32 heavy_graph = make_deep_wide_layered_graph(2.0f);
  auto shared_unit_graph =
      std::make_shared<DeltaSteppingCsrGraph>(unit_graph, nullptr);
  auto shared_closure_graph =
      std::make_shared<DeltaSteppingCsrGraph>(closure_graph, nullptr);
  auto shared_heavy_graph =
      std::make_shared<DeltaSteppingCsrGraph>(heavy_graph, nullptr);
  const std::vector<float> unit_expected =
      cpu_dijkstra_outgoing_multi_source(unit_graph, {0});
  const std::vector<float> closure_expected =
      cpu_dijkstra_outgoing_multi_source(closure_graph, {0});
  const std::vector<float> heavy_expected =
      cpu_dijkstra_outgoing_multi_source(heavy_graph, {0});
  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");

  constexpr int kWorkers = 8;
  constexpr int kRunsPerWorker = 4;
  constexpr int kForceGenericMaxIterations =
      std::numeric_limits<int>::max();
  std::promise<void> start_promise;
  const std::shared_future<void> start = start_promise.get_future().share();
  std::atomic<int> ready{0};
  std::vector<std::future<void>> workers;
  workers.reserve(kWorkers);
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker]() {
      bool readiness_reported = false;
      try {
        check_hip(hipSetDevice(device), "hipSetDevice");
        HipStream stream;
        DeltaSteppingCsrWorkspace unit_workspace(shared_unit_graph,
                                                 stream.get());
        DeltaSteppingCsrWorkspace closure_workspace(shared_closure_graph,
                                                    stream.get());
        DeltaSteppingCsrWorkspace heavy_workspace(shared_heavy_graph,
                                                  stream.get());
        ready.fetch_add(1, std::memory_order_release);
        readiness_reported = true;
        start.wait();

        for (int repetition = 0; repetition < kRunsPerWorker; ++repetition) {
          const std::string prefix =
              "parallel deep-wide worker " + std::to_string(worker) +
              " reuse " + std::to_string(repetition);
          validate_deep_wide_result(
              prefix + ": exact-unit target",
              unit_graph,
              unit_expected,
              kDeepWideTarget,
              unit_workspace.run({0},
                                 std::vector<int>{kDeepWideTarget},
                                 4.0f,
                                 -1,
                                 stream.get(),
                                 nullptr,
                                 nullptr));
          validate_deep_wide_result(
              prefix + ": exact-unit exhaustion",
              unit_graph,
              unit_expected,
              kDeepWideIsolated,
              unit_workspace.run({0},
                                 std::vector<int>{kDeepWideIsolated},
                                 4.0f,
                                 -1,
                                 stream.get(),
                                 nullptr,
                                 nullptr));
          // A finite cap bypasses the exact-unit specialization.  With
          // delta=1 every 4,097-wide level crosses the pending-minimum and
          // compaction handoffs used by the production-sized generic graph.
          validate_deep_wide_result(
              prefix + ": generic unit-weight target",
              unit_graph,
              unit_expected,
              kDeepWideTarget,
              unit_workspace.run({0},
                                 std::vector<int>{kDeepWideTarget},
                                 1.0f,
                                 kForceGenericMaxIterations,
                                 stream.get(),
                                 nullptr,
                                 nullptr));
          validate_deep_wide_result(
              prefix + ": generic unit-weight exhaustion",
              unit_graph,
              unit_expected,
              kDeepWideIsolated,
              unit_workspace.run({0},
                                 std::vector<int>{kDeepWideIsolated},
                                 1.0f,
                                 kForceGenericMaxIterations,
                                 stream.get(),
                                 nullptr,
                                 nullptr));
          validate_deep_wide_result(
              prefix + ": generic closure target",
              closure_graph,
              closure_expected,
              kDeepWideTarget,
              closure_workspace.run({0},
                                    std::vector<int>{kDeepWideTarget},
                                    1.0f,
                                    -1,
                                    stream.get(),
                                    nullptr,
                                    nullptr));
          validate_deep_wide_result(
              prefix + ": generic closure exhaustion",
              closure_graph,
              closure_expected,
              kDeepWideIsolated,
              closure_workspace.run({0},
                                    std::vector<int>{kDeepWideIsolated},
                                    1.0f,
                                    -1,
                                    stream.get(),
                                    nullptr,
                                    nullptr));
          // Weight 2 with delta 1 forces every useful relaxation through the
          // once-per-bucket heavy queue before pending-minimum reduction and
          // compaction. Run it concurrently to cover the remaining explicit-
          // stream producer/consumer chain.
          validate_deep_wide_result(
              prefix + ": generic heavy target",
              heavy_graph,
              heavy_expected,
              kDeepWideTarget,
              heavy_workspace.run({0},
                                  std::vector<int>{kDeepWideTarget},
                                  1.0f,
                                  -1,
                                  stream.get(),
                                  nullptr,
                                  nullptr));
          validate_deep_wide_result(
              prefix + ": generic heavy exhaustion",
              heavy_graph,
              heavy_expected,
              kDeepWideIsolated,
              heavy_workspace.run({0},
                                  std::vector<int>{kDeepWideIsolated},
                                  1.0f,
                                  -1,
                                  stream.get(),
                                  nullptr,
                                  nullptr));
        }
      } catch (...) {
        if (!readiness_reported) {
          ready.fetch_add(1, std::memory_order_release);
        }
        throw;
      }
    }));
  }
  while (ready.load(std::memory_order_acquire) != kWorkers) {
    std::this_thread::yield();
  }
  start_promise.set_value();
  for (std::future<void>& worker : workers) {
    worker.get();
  }
}

void test_unit_weight_specialization(hipStream_t stream) {
  HostCsrF32 graph = make_outgoing_csr(
      10,
      {{0, 2, 1.0f},
       {0, 1, 1.0f},
       {1, 3, 1.0f},
       {2, 3, 1.0f},
       {2, 5, 1.0f},
       {3, 4, 1.0f},
       {4, 6, 1.0f},
       {5, 6, 1.0f},
       {6, 7, 1.0f},
       {7, 7, 1.0f},
       {8, 6, 1.0f}});
  DeltaSteppingCsrWorkspace workspace(graph, stream);
  const std::vector<float> unit_expected =
      cpu_dijkstra_outgoing_multi_source(graph, {0, 8, 0});

  const DeltaSteppingCsrResult all_source_targets = workspace.run(
      {0, 8, 0}, std::vector<int>{0, 8, 0}, 4.0f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("unit specialization all-source targets",
                                graph,
                                {0, 8, 0},
                                {0, 8, 0},
                                unit_expected,
                                all_source_targets);
  require(all_source_targets.iterations_used == 0 &&
              all_source_targets.target_reached &&
              all_source_targets.stopped_on_target &&
              !all_source_targets.converged,
          "unit specialization all-source stop is inconsistent");

  const std::vector<int> lazy_unit_targets = {4, 7, 7};
  const DeltaSteppingCsrResult lazy_unit_first = workspace.run(
      {0, 8, 0}, lazy_unit_targets, 4.0f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("fresh lazy unit specialization",
                                graph,
                                {0, 8, 0},
                                lazy_unit_targets,
                                unit_expected,
                                lazy_unit_first);
  require(lazy_unit_first.stopped_on_target && !lazy_unit_first.converged,
          "fresh lazy unit specialization did not stop on its targets");
  // Unlimited multi-target runs on exact unit weights take the append-only
  // specialization. Duplicates, source targets, an unreachable target, and a
  // subsequent generic full-distance run exercise unit-to-generic allocation
  // and all reset invariants.
  run_full_and_target_checks("exact unit-weight specialization",
                             workspace,
                             graph,
                             {0, 8, 0},
                             {7, 4, 0, 8, 9, 7},
                             4.0f,
                             stream,
                             nullptr,
                             true);

  const std::vector<int> reachable_targets = {4, 7, 7};
  const DeltaSteppingCsrResult early_stop = workspace.run(
      {0, 8, 0}, reachable_targets, 4.0f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("unit specialization reachable early stop",
                                graph,
                                {0, 8, 0},
                                reachable_targets,
                                unit_expected,
                                early_stop);
  require(early_stop.stopped_on_target,
          "unit specialization reachable targets did not stop early");
  require(!early_stop.converged,
          "delta target-stop result should not report full convergence");
  require(early_stop.iterations_used == 1,
          "unit specialization did not report the expected delta bucket round");

  const HostCsrF32 mixed_graph = make_mixed_claim_unit_graph();
  DeltaSteppingCsrWorkspace mixed_workspace(mixed_graph, stream);
  const std::vector<float> mixed_expected =
      cpu_dijkstra_outgoing_multi_source(mixed_graph, {0});
  // A finite cap bypasses the small-graph exact-unit specialization and
  // exercises the generic Delta-Stepping queues used by the production graph,
  // whose row count is above the specialization's 2^24 cutoff.
  constexpr int kForceGenericMaxIterations =
      std::numeric_limits<int>::max();
  constexpr int kMixedClaimReuseRuns = 64;
  for (int repetition = 0; repetition < kMixedClaimReuseRuns; ++repetition) {
    const std::string suffix = " reuse " + std::to_string(repetition);
    const DeltaSteppingCsrResult mixed_early = mixed_workspace.run(
        {0},
        std::vector<int>{65},
        4.0f,
        kForceGenericMaxIterations,
        stream,
        nullptr,
        nullptr);
    validate_compact_target_paths(
        "generic mixed-claim early stop" + suffix,
        mixed_graph,
        {0},
        {65},
        mixed_expected,
        mixed_early);
    require(mixed_early.stopped_on_target && !mixed_early.converged,
            "generic mixed-claim reachable target did not stop early" + suffix);

    const DeltaSteppingCsrResult mixed_exhausted = mixed_workspace.run(
        {0},
        std::vector<int>{130},
        4.0f,
        kForceGenericMaxIterations,
        stream,
        nullptr,
        nullptr);
    validate_compact_target_paths(
        "generic mixed-claim exhaustion" + suffix,
        mixed_graph,
        {0},
        {130},
        mixed_expected,
        mixed_exhausted);
    require(mixed_exhausted.converged && !mixed_exhausted.stopped_on_target,
            "generic mixed-claim unreachable target did not exhaust" + suffix);
  }

  graph.values.front() = 2.5f;
  workspace.update_values(graph.values, stream);
  run_full_and_target_checks("unit specialization to weighted fallback",
                             workspace,
                             graph,
                             {0, 8, 0},
                             {7, 4, 0, 8, 9, 7},
                             1.5f,
                             stream,
                             nullptr,
                             true);

  std::fill(graph.values.begin(), graph.values.end(), 1.0f);
  workspace.update_values(graph.values, stream);
  run_full_and_target_checks("weighted fallback back to unit specialization",
                             workspace,
                             graph,
                             {0, 8, 0},
                             {7, 4, 0, 8, 9, 7},
                             0.75f,
                             stream,
                             nullptr,
                             true);
}

void test_zero_weight_scc_predecessors(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      5,
      {{0, 1, 0.0f},
       {0, 2, 0.0f},
       {1, 2, 0.0f},
       {2, 1, 0.0f},
       {1, 3, 1.0f},
       {2, 3, 1.0f},
       {3, 4, 0.0f}});
  DeltaSteppingCsrWorkspace workspace(graph, stream);
  run_full_and_target_checks("zero-weight SCC parents",
                             workspace,
                             graph,
                             {0},
                             {1, 2, 3, 4},
                             0.5f,
                             stream,
                             nullptr,
                             true);
  run_compact_parent_mode_comparison("zero-weight SCC compact parent modes",
                                     graph,
                                     {0},
                                     {1, 2, 3, 4},
                                     0.5f,
                                     stream);
}

void test_float_bucket_boundaries_and_saturation(hipStream_t stream) {
  {
    const float delta = 0.1f;
    const float nominally_heavy =
        std::nextafter(delta, std::numeric_limits<float>::infinity());
    const HostCsrF32 graph = make_outgoing_csr(
        4,
        {{0, 1, 2.3f},
         {1, 2, nominally_heavy},
         {2, 3, 0.0f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // Float addition places the nominally-heavy 1->2 relaxation in the same
    // bucket as vertex 1. It must re-enter closure so vertex 3 is reached.
    run_full_and_target_checks("same-bucket nominally-heavy edge",
                               workspace,
                               graph,
                               {0},
                               {1, 2, 3},
                               delta,
                               stream,
                               nullptr,
                               true);
  }
  {
    const HostCsrF32 graph = make_outgoing_csr(
        4,
        {{0, 1, 1.1e9f},
         {1, 2, 2.0f},
         {2, 3, 0.0f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // All three finite distances clamp to the terminal representable bucket.
    // Terminal-bucket processing must close over every outgoing edge.
    run_full_and_target_checks("terminal bucket saturation",
                               workspace,
                               graph,
                               {0},
                               {1, 2, 3},
                               1.0f,
                               stream,
                               nullptr,
                               true);
  }
  {
    const HostCsrF32 graph = make_outgoing_csr(
        4,
        {{0, 1, 1.0f},
         {1, 2, 1.0f},
         {2, 3, 1.0f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    DeltaSteppingCsrWorkspace forced_generic_workspace(
        graph,
        stream,
        DeltaSteppingCsrWorkspaceOptions{
            DeltaSteppingCsrParentMode::kAutomatic,
            DeltaSteppingCsrExecutionMode::kForceGeneric});
    const std::vector<int> sources = {0};
    const std::vector<int> targets = {3};
    const std::vector<float> expected =
        cpu_dijkstra_outgoing_multi_source(graph, sources);
    constexpr float kSaturatingDelta = 1.0e-10f;

    const DeltaSteppingCsrResult specialized = workspace.run(
        sources, targets, kSaturatingDelta, -1, stream, nullptr, nullptr);
    validate_compact_target_paths("unit terminal-bucket specialization",
                                  graph,
                                  sources,
                                  targets,
                                  expected,
                                  specialized);

    const DeltaSteppingCsrResult generic = forced_generic_workspace.run(
        sources, targets, kSaturatingDelta, -1, stream, nullptr, nullptr);
    validate_compact_target_paths("unit terminal-bucket generic fallback",
                                  graph,
                                  sources,
                                  targets,
                                  expected,
                                  generic);
    require(specialized.iterations_used == 2,
            "unit terminal-bucket specialization counted BFS depths instead "
            "of distinct buckets");
    require(generic.iterations_used == specialized.iterations_used,
            "unit terminal-bucket specialization disagrees with generic "
            "bucket rounds");
  }
}

void test_batched_closure_and_iteration_fallback(hipStream_t stream) {
  for (const int edge_count : {1, 2, 3, 4, 5, 12}) {
    std::vector<EdgeSpec> edges;
    for (int vertex = 0; vertex < edge_count; ++vertex) {
      edges.push_back({vertex, vertex + 1, 0.0f});
    }
    const HostCsrF32 graph = make_outgoing_csr(edge_count + 1, edges);
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // After the immediate source round, edge counts 1..4 empty on each
    // possible speculative batch round, 5 crosses the boundary, and 12 spans
    // several batches. Repeating checks both ping-pong count parities.
    std::ostringstream label;
    label << "batched zero-weight closure length " << edge_count;
    run_full_and_target_checks(label.str(),
                               workspace,
                               graph,
                               {0},
                               {edge_count},
                               1.0f,
                               stream,
                               nullptr,
                               true);
  }
  {
    const HostCsrF32 graph = make_outgoing_csr(
        4,
        {{0, 1, 0.75f}, {1, 2, 1.25f}, {2, 3, 0.5f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    DeltaSteppingCsrWorkspace legacy(
        graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
    const std::vector<float> expected =
        cpu_dijkstra_outgoing_multi_source(graph, {0});
    const DeltaSteppingCsrResult zero_iterations = workspace.run(
        std::vector<int>{0}, std::vector<int>{3},
        1.0f, 0, stream, nullptr, nullptr);
    require(zero_iterations.iterations_used == 0 &&
                !zero_iterations.converged &&
                !zero_iterations.target_reached,
            "zero-iteration generic fallback did not preserve its cap");
    const DeltaSteppingCsrResult limited = workspace.run(
        std::vector<int>{0}, std::vector<int>{3},
        1.0f, 1, stream, nullptr, nullptr);
    require(limited.iterations_used == 1,
            "limited generic run did not stop at one bucket iteration");
    require(!limited.target_reached && !limited.stopped_on_target,
            "limited generic run reached a target beyond its iteration cap");
    require(!limited.converged,
            "limited generic run incorrectly reported full convergence");

    const std::vector<int> partial_targets = {1, 3};
    std::vector<float> partial_expected = expected;
    partial_expected[3] = kInf;
    const DeltaSteppingCsrResult automatic_partial = workspace.run(
        std::vector<int>{0}, partial_targets,
        1.0f, 1, stream, nullptr, nullptr);
    const DeltaSteppingCsrResult legacy_partial = legacy.run(
        std::vector<int>{0}, partial_targets,
        1.0f, 1, stream, nullptr, nullptr);
    validate_compact_target_paths("weighted finite-limit automatic",
                                  graph,
                                  {0},
                                  partial_targets,
                                  partial_expected,
                                  automatic_partial);
    validate_compact_target_paths("weighted finite-limit forced legacy",
                                  graph,
                                  {0},
                                  partial_targets,
                                  partial_expected,
                                  legacy_partial);
    validate_parent_mode_agreement("weighted finite-limit parent modes",
                                   automatic_partial,
                                   legacy_partial);
    require(automatic_partial.iterations_used == 1 &&
                legacy_partial.iterations_used == 1 &&
                !automatic_partial.stopped_on_target &&
                !legacy_partial.stopped_on_target &&
                !automatic_partial.converged &&
                !legacy_partial.converged,
            "weighted finite-limit parent modes did not preserve the cap");

    std::vector<DeltaSteppingCsrProgress> progress;
    const DeltaSteppingCsrResult callback_result = workspace.run(
        std::vector<int>{0}, std::vector<int>{3},
        1.0f, -1, stream, record_progress, &progress);
    validate_compact_target_paths("generic callback fallback",
                                  graph,
                                  {0},
                                  {3},
                                  expected,
                                  callback_result);
    require(!progress.empty(),
            "generic callback fallback did not report bucket progress");
    for (std::size_t i = 0; i < progress.size(); ++i) {
      require(progress[i].iteration == static_cast<int>(i + 1),
              "generic callback iteration sequence is not contiguous");
      require(progress[i].convergence_checked,
              "generic callback did not report a convergence check");
      require(progress[i].max_iters == std::numeric_limits<int>::max(),
              "generic callback reported the wrong unlimited iteration cap");
    }
    require(callback_result.stopped_on_target && !callback_result.converged,
            "weighted callback run did not stop on its target");

    std::vector<DeltaSteppingCsrProgress> legacy_progress;
    const DeltaSteppingCsrResult legacy_callback_result = legacy.run(
        std::vector<int>{0}, std::vector<int>{3},
        1.0f, -1, stream, record_progress, &legacy_progress);
    validate_compact_target_paths("forced-legacy weighted callback",
                                  graph,
                                  {0},
                                  {3},
                                  expected,
                                  legacy_callback_result);
    validate_parent_mode_agreement("weighted callback parent modes",
                                   callback_result,
                                   legacy_callback_result);
    require(!legacy_progress.empty(),
            "forced-legacy weighted callback did not report progress");
    for (std::size_t i = 0; i < legacy_progress.size(); ++i) {
      require(legacy_progress[i].iteration == static_cast<int>(i + 1) &&
                  legacy_progress[i].convergence_checked,
              "forced-legacy weighted callback sequence is invalid");
    }
    require(progress.size() == legacy_progress.size(),
            "callback count differs between compact parent modes");
    for (std::size_t i = 0; i < progress.size(); ++i) {
      require(progress[i].max_iters == legacy_progress[i].max_iters &&
                  progress[i].changed == legacy_progress[i].changed,
              "callback payload differs between compact parent modes");
    }
  }
  {
    const HostCsrF32 exact_unit_graph = make_outgoing_csr(
        4,
        {{0, 1, 1.0f}, {1, 2, 1.0f}, {2, 3, 1.0f}});
    DeltaSteppingCsrWorkspace exact_unit_workspace(exact_unit_graph, stream);
    const DeltaSteppingCsrResult limited_exact_unit =
        exact_unit_workspace.run(std::vector<int>{0},
                                 std::vector<int>{3},
                                 1.0f,
                                 1,
                                 stream,
                                 nullptr,
                                 nullptr);
    require(limited_exact_unit.iterations_used == 1 &&
                !limited_exact_unit.target_reached &&
                !limited_exact_unit.stopped_on_target &&
                !limited_exact_unit.converged,
            "finite iteration cap did not keep an exact-unit graph on its "
            "generic fallback");
  }
}

void test_capped_tentative_target_filtering(hipStream_t stream) {
  // Bucket zero discovers a valid but non-shortest direct path to vertex two
  // through a heavy edge.  The shorter path is not settled until later buckets.
  // A one-bucket cap must not expose the tentative distance/path as a reached
  // target.  A source target remains settled even when the cap is zero.
  const HostCsrF32 graph = make_outgoing_csr(
      3,
      {{0, 2, 10.0f}, {0, 1, 1.0f}, {1, 2, 1.0f}});
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, {0});

  auto exercise = [&](const std::string& label,
                      DeltaSteppingCsrWorkspace& workspace) {
    std::vector<float> zero_iteration_expected(
        static_cast<std::size_t>(graph.rows), kInf);
    zero_iteration_expected[0] = 0.0f;

    const std::vector<int> all_source_targets = {0, 0};
    const DeltaSteppingCsrResult all_sources = workspace.run(
        std::vector<int>{0}, all_source_targets, 1.0f, 0,
        stream, nullptr, nullptr);
    validate_compact_target_paths(label + ": all targets are sources",
                                  graph,
                                  {0},
                                  all_source_targets,
                                  zero_iteration_expected,
                                  all_sources);
    require(all_sources.iterations_used == 0 &&
                all_sources.target_reached &&
                all_sources.stopped_on_target &&
                !all_sources.converged,
            label + ": all-source target stop is inconsistent at zero cap");

    const std::vector<int> zero_iteration_targets = {0, 2};
    const DeltaSteppingCsrResult zero_iterations = workspace.run(
        std::vector<int>{0}, zero_iteration_targets, 1.0f, 0,
        stream, nullptr, nullptr);
    validate_compact_target_paths(label + ": zero-iteration source target",
                                  graph,
                                  {0},
                                  zero_iteration_targets,
                                  zero_iteration_expected,
                                  zero_iterations);
    require(zero_iterations.iterations_used == 0 &&
                !zero_iterations.converged &&
                !zero_iterations.stopped_on_target,
            label + ": zero-iteration result did not preserve its cap");
    require(zero_iterations.target_distances[0] == 0.0f &&
                std::isinf(zero_iterations.target_distances[1]),
            label + ": zero-iteration settled mask lost its source target");

    std::vector<float> capped_expected(
        static_cast<std::size_t>(graph.rows), kInf);
    const DeltaSteppingCsrResult capped = workspace.run(
        std::vector<int>{0}, std::vector<int>{2}, 1.0f, 1,
        stream, nullptr, nullptr);
    validate_compact_target_paths(label + ": tentative heavy target",
                                  graph,
                                  {0},
                                  {2},
                                  capped_expected,
                                  capped);
    require(capped.iterations_used == 1 && !capped.target_reached &&
                !capped.stopped_on_target && !capped.converged,
            label + ": finite cap exposed an unsettled target");

    const DeltaSteppingCsrResult completed = workspace.run(
        std::vector<int>{0}, std::vector<int>{2}, 1.0f, -1,
        stream, nullptr, nullptr);
    validate_compact_target_paths(label + ": reuse after capped cleanup",
                                  graph,
                                  {0},
                                  {2},
                                  expected,
                                  completed);
    require(completed.stopped_on_target && !completed.converged,
            label + ": completed rerun did not settle the target");
  };

  DeltaSteppingCsrWorkspace automatic(graph, stream);
  DeltaSteppingCsrWorkspace legacy(
      graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
  exercise("automatic compact parents", automatic);
  exercise("forced legacy parents", legacy);
}

void test_callback_exception_cleanup(hipStream_t stream) {
  // The forward query reaches its target in the second bucket.  Throw once at
  // the first next-bucket callback (with live pending membership), and once at
  // the later target-stop callback (with every vertex touched).  The reverse
  // recovery query changes the source; stale distances, flags, or predecessor
  // state from either aborted run would produce an immediately wrong path.
  const HostCsrF32 graph = make_outgoing_csr(
      4,
      {{0, 1, 0.75f},
       {1, 0, 0.75f},
       {1, 2, 1.25f},
       {2, 1, 1.25f},
       {2, 3, 0.5f},
       {3, 2, 0.5f}});
  const std::vector<float> reverse_expected =
      cpu_dijkstra_outgoing_multi_source(graph, {3});

  auto exercise = [&](const std::string& label,
                      DeltaSteppingCsrWorkspace& workspace) {
    for (const int throw_on_call : {1, 2}) {
      ThrowingProgressState throwing_state;
      throwing_state.throw_on_call = throw_on_call;
      bool caught_original = false;
      try {
        (void)workspace.run(std::vector<int>{0}, std::vector<int>{3}, 1.0f,
                            -1, stream, throw_progress_on_selected_call,
                            &throwing_state);
      } catch (const ExpectedProgressCallbackError&) {
        caught_original = true;
      } catch (const std::exception& error) {
        fail(label + ": callback cleanup replaced the original exception: " +
             error.what());
      }
      require(caught_original && throwing_state.calls == throw_on_call,
              label + ": selected callback exception was not preserved");

      const DeltaSteppingCsrResult recovered = workspace.run(
          std::vector<int>{3}, std::vector<int>{0}, 1.0f, -1,
          stream, nullptr, nullptr);
      const std::string recovery_label =
          label + ": reuse after callback " + std::to_string(throw_on_call);
      validate_compact_target_paths(recovery_label,
                                    graph,
                                    {3},
                                    {0},
                                    reverse_expected,
                                    recovered);
      require(recovered.stopped_on_target && !recovered.converged,
              recovery_label + ": workspace state was not restored");
    }
  };

  DeltaSteppingCsrWorkspace automatic(graph, stream);
  DeltaSteppingCsrWorkspace legacy(
      graph, stream, DeltaSteppingCsrParentMode::kForceLegacy);
  exercise("automatic callback cleanup", automatic);
  exercise("forced-legacy callback cleanup", legacy);
}

void test_wave_boundary_contention(hipStream_t stream) {
  constexpr int kFrontierSize = 257;
  {
    std::vector<int> sources;
    std::vector<EdgeSpec> edges;
    sources.reserve(kFrontierSize);
    edges.reserve(kFrontierSize);
    for (int source = 0; source < kFrontierSize; ++source) {
      const int child = kFrontierSize + source;
      sources.push_back(source);
      edges.push_back({source, child, 0.0f});
    }
    const HostCsrF32 graph =
        make_outgoing_csr(kFrontierSize * 2, edges);
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // 257 frontier threads discover distinct children together, exercising
    // full and partial wave boundaries. Every vertex is touched.
    run_full_and_target_checks("wave-boundary distinct appends",
                               workspace,
                               graph,
                               sources,
                               {kFrontierSize * 2 - 1},
                               1.0f,
                               stream,
                               nullptr,
                               true);
  }
  {
    constexpr int kSink = kFrontierSize + 1;
    std::vector<EdgeSpec> edges;
    edges.reserve(static_cast<std::size_t>(kFrontierSize) * 2);
    for (int vertex = 1; vertex <= kFrontierSize; ++vertex) {
      edges.push_back({0, vertex, 0.0f});
      edges.push_back({vertex, kSink, 0.0f});
    }
    const HostCsrF32 graph = make_outgoing_csr(kSink + 1, edges);
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    // One 257-thread frontier contends to claim a single sink, stressing CAS
    // authority and sparse successful appends across two blocks.
    run_full_and_target_checks("wave-boundary high-fan-in contention",
                               workspace,
                               graph,
                               {0},
                               {kSink, 256, kSink},
                               1.0f,
                               stream,
                               nullptr,
                               true);
  }
}

void test_shared_graph_workspaces(hipStream_t construction_stream) {
  const HostCsrF32 graph = make_weighted_corner_graph();
  auto shared_graph =
      std::make_shared<DeltaSteppingCsrGraph>(graph, construction_stream);
  HipStream other_stream;
  DeltaSteppingCsrWorkspace first(shared_graph, construction_stream);
  DeltaSteppingCsrWorkspace second(shared_graph, other_stream.get());

  const std::vector<int> first_sources = {0, 8};
  const std::vector<int> first_targets = {4, 7, 9, 11};
  const std::vector<int> second_sources = {10};
  const std::vector<int> second_targets = {11, 10, 0, 9};
  const std::vector<float> first_expected =
      cpu_dijkstra_outgoing_multi_source(graph, first_sources);
  const std::vector<float> second_expected =
      cpu_dijkstra_outgoing_multi_source(graph, second_sources);

  auto first_run = std::async(std::launch::async, [&]() {
    return first.run(first_sources,
                     first_targets,
                     1.25f,
                     -1,
                     construction_stream,
                     nullptr,
                     nullptr);
  });
  auto second_run = std::async(std::launch::async, [&]() {
    return second.run(second_sources,
                      second_targets,
                      2.5f,
                      -1,
                      other_stream.get(),
                      nullptr,
                      nullptr);
  });
  const DeltaSteppingCsrResult first_result = first_run.get();
  const DeltaSteppingCsrResult second_result = second_run.get();
  validate_compact_target_paths("shared graph first stream",
                                graph,
                                first_sources,
                                first_targets,
                                first_expected,
                                first_result);
  validate_compact_target_paths("shared graph second stream",
                                graph,
                                second_sources,
                                second_targets,
                                second_expected,
                                second_result);
  require(first_result.converged && !first_result.stopped_on_target &&
              second_result.converged && !second_result.stopped_on_target,
          "shared-graph unreachable target runs did not exhaust their graphs");

  bool rejected_update = false;
  try {
    first.update_values(graph.values, construction_stream);
  } catch (const std::logic_error&) {
    rejected_update = true;
  }
  require(rejected_update,
          "immutable shared delta graph accepted an edge-value update");

  std::vector<float> shared_vertex_costs(
      static_cast<std::size_t>(graph.rows), 1.0f);
  shared_vertex_costs[2] = 0.5f;
  shared_vertex_costs[4] = 1.75f;
  first.update_vertex_costs(shared_vertex_costs, construction_stream);
  const std::vector<int> cost_targets = {4, 7};
  const DeltaSteppingCsrResult cost_result = first.run(
      first_sources, cost_targets, 1.5f, -1,
      construction_stream, nullptr, nullptr);
  validate_compact_target_paths(
      "shared graph workspace-local vertex costs",
      graph,
      first_sources,
      cost_targets,
      cpu_dijkstra_outgoing_multi_source(graph, first_sources,
                                         &shared_vertex_costs),
      cost_result,
      &shared_vertex_costs);
  require(cost_result.stopped_on_target && !cost_result.converged,
          "shared vertex-cost run did not stop on reachable targets");

  const HostCsrF32 unit_graph = make_outgoing_csr(
      6,
      {{0, 1, 1.0f},
       {1, 2, 1.0f},
       {2, 3, 1.0f},
       {4, 2, 1.0f}});
  auto shared_unit_graph =
      std::make_shared<DeltaSteppingCsrGraph>(unit_graph,
                                              construction_stream);
  DeltaSteppingCsrWorkspace unit_first(shared_unit_graph,
                                       construction_stream);
  DeltaSteppingCsrWorkspace unit_second(shared_unit_graph,
                                        other_stream.get());
  auto unit_first_run = std::async(std::launch::async, [&]() {
    return unit_first.run({0}, std::vector<int>{3}, 2.0f, -1,
                          construction_stream, nullptr, nullptr);
  });
  auto unit_second_run = std::async(std::launch::async, [&]() {
    return unit_second.run({4}, std::vector<int>{2, 3}, 2.0f, -1,
                           other_stream.get(), nullptr, nullptr);
  });
  const DeltaSteppingCsrResult unit_first_result = unit_first_run.get();
  const DeltaSteppingCsrResult unit_second_result = unit_second_run.get();
  validate_compact_target_paths(
      "shared graph lazy unit first stream",
      unit_graph,
      {0},
      {3},
      cpu_dijkstra_outgoing_multi_source(unit_graph, {0}),
      unit_first_result);
  validate_compact_target_paths(
      "shared graph lazy unit second stream",
      unit_graph,
      {4},
      {2, 3},
      cpu_dijkstra_outgoing_multi_source(unit_graph, {4}),
      unit_second_result);
  require(unit_first_result.stopped_on_target &&
              unit_second_result.stopped_on_target &&
              !unit_first_result.converged &&
              !unit_second_result.converged,
          "shared lazy unit workspaces did not stop on reachable targets");

  // A workspace retains the immutable allocation it was constructed from,
  // even if the caller later replaces the movable public graph wrapper.
  const HostCsrF32 replacement_unit_graph = make_outgoing_csr(
      6,
      {{0, 5, 1.0f}});
  *shared_unit_graph =
      DeltaSteppingCsrGraph(replacement_unit_graph, construction_stream);
  const DeltaSteppingCsrResult retained_graph_result = unit_first.run(
      {0}, std::vector<int>{3}, 2.0f, -1,
      construction_stream, nullptr, nullptr);
  validate_compact_target_paths(
      "shared graph survives wrapper move assignment",
      unit_graph,
      {0},
      {3},
      cpu_dijkstra_outgoing_multi_source(unit_graph, {0}),
      retained_graph_result);
  require(retained_graph_result.stopped_on_target &&
              !retained_graph_result.converged,
          "shared workspace followed a replacement graph wrapper");
}

void test_parallel_divergent_workspaces(hipStream_t construction_stream) {
  const HostCsrF32 graph = make_mixed_claim_unit_graph();
  auto shared_graph =
      std::make_shared<DeltaSteppingCsrGraph>(graph, construction_stream);
  const std::vector<float> expected =
      cpu_dijkstra_outgoing_multi_source(graph, {0});
  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");

  constexpr int kWorkers = 8;
  constexpr int kRunsPerWorker = 16;
  constexpr int kForceGenericMaxIterations =
      std::numeric_limits<int>::max();
  std::vector<std::future<void>> workers;
  workers.reserve(kWorkers);
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker]() {
      check_hip(hipSetDevice(device), "hipSetDevice");
      HipStream stream;
      DeltaSteppingCsrWorkspace workspace(shared_graph, stream.get());
      for (int repetition = 0; repetition < kRunsPerWorker; ++repetition) {
        const std::string label =
            "parallel divergent worker " + std::to_string(worker) +
            " reuse " + std::to_string(repetition);
        const DeltaSteppingCsrResult early = workspace.run(
            std::vector<int>{0},
            std::vector<int>{65},
            4.0f,
            kForceGenericMaxIterations,
            stream.get(),
            nullptr,
            nullptr);
        validate_compact_target_paths(label + ": reachable",
                                      graph,
                                      {0},
                                      {65},
                                      expected,
                                      early);
        require(early.stopped_on_target && !early.converged,
                label + ": reachable target did not stop early");

        const DeltaSteppingCsrResult exhausted = workspace.run(
            std::vector<int>{0},
            std::vector<int>{130},
            4.0f,
            kForceGenericMaxIterations,
            stream.get(),
            nullptr,
            nullptr);
        validate_compact_target_paths(label + ": exhausted",
                                      graph,
                                      {0},
                                      {130},
                                      expected,
                                      exhausted);
        require(exhausted.converged && !exhausted.stopped_on_target,
                label + ": unreachable target did not exhaust");
      }
    }));
  }
  for (std::future<void>& worker : workers) {
    worker.get();
  }
}

void test_stream_affinity(hipStream_t construction_stream) {
  const HostCsrF32 graph =
      make_outgoing_csr(2, {{0, 1, 1.0f}});
  DeltaSteppingCsrWorkspace workspace(graph, construction_stream);
  HipStream other_stream;
  bool rejected = false;
  try {
    (void)workspace.run(
        std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
        other_stream.get(), nullptr, nullptr);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "delta workspace accepted a different HIP stream");
}

void test_workspace_reuse_and_updates(hipStream_t stream) {
  const HostCsrF32 original = make_weighted_corner_graph();
  HostCsrF32 current = original;
  DeltaSteppingCsrWorkspace workspace(original, stream);
  DeltaSteppingCsrWorkspace legacy_workspace(
      original, stream, DeltaSteppingCsrParentMode::kForceLegacy);

  const std::vector<int> primary_sources = {0, 8, 0};
  const std::vector<int> primary_targets = {7, 4, 0, 8, 9, 7, 11};
  run_full_and_target_checks("weighted baseline",
                             workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             2.25f,
                             stream,
                             nullptr,
                             true);
  run_full_and_target_checks("weighted baseline forced legacy",
                             legacy_workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             2.25f,
                             stream,
                             nullptr,
                             true);

  const std::vector<float> primary_expected =
      cpu_dijkstra_outgoing_multi_source(current, primary_sources);
  const DeltaSteppingCsrResult single_target = workspace.run(
      primary_sources, 4, 2.25f, -1, stream, nullptr, nullptr);
  validate_single_target_predecessors("weighted single target",
                                      current,
                                      primary_sources,
                                      4,
                                      primary_expected[4],
                                      single_target);

  // Change source components on the same scratch storage. This catches stale
  // distances, queue flags, and predecessor data from the preceding searches.
  run_full_and_target_checks("reused disconnected component",
                             workspace,
                             current,
                             {10},
                             {11, 10, 0, 9},
                             0.5f,
                             stream,
                             nullptr,
                             true);

  for (std::size_t edge = 0; edge < current.values.size(); ++edge) {
    const float old_weight = original.values[edge];
    current.values[edge] =
        old_weight == 0.0f
            ? 0.0f
            : 0.35f + old_weight *
                          (0.5f + 0.25f * static_cast<float>(edge % 5));
  }
  // Exercise the update API's host-lifetime contract: the temporary is gone
  // before the following run begins.
  workspace.update_values(std::vector<float>(current.values), stream);
  legacy_workspace.update_values(current.values, stream);
  run_full_and_target_checks("updated edge values",
                             workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             1.75f,
                             stream,
                             nullptr,
                             true);
  run_full_and_target_checks("updated edge values forced legacy",
                             legacy_workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             1.75f,
                             stream,
                             nullptr,
                             true);

  std::vector<float> vertex_costs(static_cast<std::size_t>(current.rows));
  for (std::size_t vertex = 0; vertex < vertex_costs.size(); ++vertex) {
    vertex_costs[vertex] =
        0.5f + 0.375f * static_cast<float>((vertex * 7) % 6);
  }
  workspace.update_vertex_costs(std::vector<float>(vertex_costs), stream);
  legacy_workspace.update_vertex_costs(vertex_costs, stream);
  run_full_and_target_checks("destination vertex costs",
                             workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             2.0f,
                             stream,
                             &vertex_costs,
                             true);
  run_full_and_target_checks("destination vertex costs forced legacy",
                             legacy_workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             2.0f,
                             stream,
                             &vertex_costs,
                             true);
}

void test_compact_legacy_mode_alternation(hipStream_t stream) {
  const HostCsrF32 graph = make_weighted_corner_graph();
  DeltaSteppingCsrWorkspace workspace(graph, stream);

  const std::vector<int> primary_sources = {0, 8, 0};
  const std::vector<int> primary_targets = {4, 7, 7, 0, 8};
  const std::vector<float> primary_expected =
      cpu_dijkstra_outgoing_multi_source(graph, primary_sources);

  // Start with compact mode on a fresh workspace. This proves its source
  // marker and core initialization do not depend on a preceding legacy run.
  const DeltaSteppingCsrResult compact_first = workspace.run(
      primary_sources, primary_targets, 1.25f, -1, stream, nullptr, nullptr);
  validate_compact_target_paths("fresh compact before legacy",
                                graph,
                                primary_sources,
                                primary_targets,
                                primary_expected,
                                compact_first);
  require(compact_first.stopped_on_target &&
              compact_first.pred_node.empty() &&
              compact_first.pred_edge.empty(),
          "fresh compact run materialized legacy predecessor storage");

  const std::vector<float> disconnected_expected =
      cpu_dijkstra_outgoing_multi_source(graph, {10});
  const DeltaSteppingCsrResult single_after_compact = workspace.run(
      std::vector<int>{10}, 11, 0.5f, -1, stream, nullptr, nullptr);
  validate_single_target_predecessors("single target after compact",
                                      graph,
                                      {10},
                                      11,
                                      disconnected_expected[11],
                                      single_after_compact);

  const DeltaSteppingCsrResult full_after_single = workspace.run(
      primary_sources, -1, 1.25f, -1, stream, nullptr, nullptr);
  validate_full_distances("full after compact and single",
                          primary_expected,
                          full_after_single);

  const std::vector<int> disconnected_targets = {11, 10, 0, 9, 11};
  const DeltaSteppingCsrResult compact_after_legacy = workspace.run(
      std::vector<int>{10}, disconnected_targets, 0.5f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("compact after legacy full run",
                                graph,
                                {10},
                                disconnected_targets,
                                disconnected_expected,
                                compact_after_legacy);
  require(compact_after_legacy.converged &&
              !compact_after_legacy.stopped_on_target,
          "compact run with unreachable targets did not exhaust its component");

  const DeltaSteppingCsrResult second_single = workspace.run(
      primary_sources, 4, 1.25f, -1, stream, nullptr, nullptr);
  validate_single_target_predecessors("second single target after compact",
                                      graph,
                                      primary_sources,
                                      4,
                                      primary_expected[4],
                                      second_single);
  require(second_single.pred_node[10] == -1 &&
              second_single.pred_edge[10] == static_cast<Offset>(-1),
          "compact mode left a legacy predecessor marker on an untouched source");
  const DeltaSteppingCsrResult repeated_compact = workspace.run(
      primary_sources, primary_targets, 1.25f, -1,
      stream, nullptr, nullptr);
  validate_compact_target_paths("repeated compact after legacy",
                                graph,
                                primary_sources,
                                primary_targets,
                                primary_expected,
                                repeated_compact);
}

HostCsrF32 make_random_graph(int vertex_count,
                            int active_vertex_count,
                            int extra_edges,
                            std::uint32_t seed) {
  require(active_vertex_count >= 2 &&
              active_vertex_count <= vertex_count - 3,
          "random graph requires room for a disconnected component");
  std::mt19937 random(seed);
  std::uniform_int_distribution<int> active_vertex(0,
                                                    active_vertex_count - 1);
  std::uniform_real_distribution<float> weight(0.125f, 18.0f);
  std::vector<EdgeSpec> edges;
  edges.reserve(static_cast<std::size_t>(active_vertex_count + extra_edges) *
                2);

  // Make the active component reachable from source zero while leaving plenty
  // of alternative weighted paths for Dijkstra/Delta-Stepping comparison.
  for (int vertex = 0; vertex + 1 < active_vertex_count; ++vertex) {
    edges.push_back({vertex, vertex + 1, weight(random)});
  }
  for (int edge_index = 0; edge_index < extra_edges; ++edge_index) {
    int from = active_vertex(random);
    int to = active_vertex(random);
    if (from == to && edge_index % 3 != 0) {
      to = (to + 1) % active_vertex_count;
    }
    const float edge_weight = weight(random);
    edges.push_back({from, to, edge_weight});
    if (edge_index % 19 == 0) {
      // Parallel edge with a distinct weight.
      edges.push_back({from, to, edge_weight * 0.625f});
    }
  }

  const int component_a = active_vertex_count;
  const int component_b = active_vertex_count + 1;
  edges.push_back({component_a, component_b, weight(random)});
  edges.push_back({component_b, component_a, weight(random)});
  // The final vertex remains isolated. Edges directed from disconnected nodes
  // into the active component do not make them reachable from active sources.
  edges.push_back({component_b, active_vertex(random), weight(random)});
  return make_outgoing_csr(vertex_count, edges);
}

void test_seeded_random_graphs(hipStream_t stream) {
  const std::vector<float> deltas = {0.5f, 1.25f, 3.75f, 8.0f, 32.0f};
  for (int case_index = 0; case_index < 8; ++case_index) {
    const int vertex_count = 37 + case_index * 13;
    const int active_count = vertex_count - 3;
    const std::uint32_t seed =
        static_cast<std::uint32_t>(0xC0FFEEu + 7919u * case_index);
    const HostCsrF32 graph =
        make_random_graph(vertex_count, active_count, active_count * 6, seed);
    DeltaSteppingCsrWorkspace workspace(graph, stream);
    const std::vector<int> sources = {0, active_count / 2, 0};
    const std::vector<int> targets = {
        0,
        active_count - 1,
        (active_count * 7 + 3) % active_count,
        active_count,
        active_count + 1,
        vertex_count - 1,
        active_count - 1,
    };

    std::ostringstream label;
    label << "random case " << case_index << " seed " << seed;
    run_full_and_target_checks(label.str(),
                               workspace,
                               graph,
                               sources,
                               targets,
                               deltas[static_cast<std::size_t>(case_index) %
                                      deltas.size()],
                               stream,
                               nullptr,
                               case_index % 3 == 0);
  }
}

}  // namespace

int main() {
  try {
    test_compact_edge_id_eligibility();
    test_braced_default_stream_constructor_compatibility();
    HipStream stream;
    test_empty_and_singleton_graphs(stream.get());
    test_distances_only_graph_families(stream.get());
    test_distances_only_path_state_transitions(stream.get());
    test_distances_only_strict_storage_and_validation(stream.get());
    test_runtime_telemetry_modes_and_reset(stream.get());
    test_generic_compact_parent_modes(stream.get());
    test_compact_early_settlement_reset(stream.get());
    test_forced_generic_increasing_distance_reuse(stream.get());
    test_compact_weight_class_updates(stream.get());
    test_compact_to_unit_storage_transition(stream.get());
    test_unit_weight_specialization(stream.get());
    test_default_stream_deep_wide_batching();
    test_zero_weight_scc_predecessors(stream.get());
    test_float_bucket_boundaries_and_saturation(stream.get());
    test_batched_closure_and_iteration_fallback(stream.get());
    test_capped_tentative_target_filtering(stream.get());
    test_callback_exception_cleanup(stream.get());
    test_wave_boundary_contention(stream.get());
    test_shared_graph_workspaces(stream.get());
    test_parallel_divergent_workspaces(stream.get());
    test_parallel_deep_wide_explicit_streams();
    test_workspace_reuse_and_updates(stream.get());
    test_compact_legacy_mode_alternation(stream.get());
    test_seeded_random_graphs(stream.get());
    test_stream_affinity(stream.get());
    check_hip(hipStreamSynchronize(stream.get()), "final hipStreamSynchronize");
    std::cout << "Delta-Stepping outgoing-CSR HIP regression test passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "Delta-Stepping outgoing-CSR HIP regression test failed: "
              << exception.what() << '\n';
    return 1;
  }
}
