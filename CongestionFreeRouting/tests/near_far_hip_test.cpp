#include "../near_far/near_far.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/near_far \
//     CongestionFreeRouting/tests/near_far_hip_test.cpp \
//     CongestionFreeRouting/near_far/near_far.cpp \
//     -o /tmp/near_far_hip_test && /tmp/near_far_hip_test

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kAbsoluteTolerance = 2e-3f;
constexpr float kRelativeTolerance = 2e-5f;

struct EdgeSpec {
  int from = -1;
  int to = -1;
  float weight = 0.0f;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

template <typename Exception, typename Function>
void require_throws(const std::string& label, Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception&) {
    rejected = true;
  }
  require(rejected, label + ": expected exception was not thrown");
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
    if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
  }

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

HostCsrF32 make_outgoing_csr(int vertex_count,
                            const std::vector<EdgeSpec>& edges) {
  require(vertex_count > 0, "test graph must contain at least one vertex");

  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
  for (const EdgeSpec& edge : edges) {
    require(edge.from >= 0 && edge.from < vertex_count &&
                edge.to >= 0 && edge.to < vertex_count,
            "test edge endpoint is outside the graph");
    require(std::isfinite(edge.weight) && edge.weight >= 0.0f,
            "test edge weight must be finite and nonnegative");
    ++graph.rowptr[static_cast<std::size_t>(edge.from) + 1];
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex + 1)] +=
        graph.rowptr[static_cast<std::size_t>(vertex)];
  }

  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
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
  const std::size_t index = static_cast<std::size_t>(edge);
  const int destination = graph.colind[index];
  const float cost =
      vertex_costs == nullptr
          ? 1.0f
          : (*vertex_costs)[static_cast<std::size_t>(destination)];
  return graph.values[index] * cost;
}

std::vector<float> cpu_dijkstra(
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
    if (distance != distances[static_cast<std::size_t>(u)]) continue;
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
  if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <=
         kAbsoluteTolerance + kRelativeTolerance * scale;
}

void validate_target_result(
    const std::string& label,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<float>& expected,
    const NearFarCsrResult& result,
    const std::vector<float>* vertex_costs = nullptr) {
  const std::size_t count = targets.size();
  require(result.target == -1, label + ": vector target marker is not -1");
  require(result.target_distances.size() == count,
          label + ": target distance count mismatch");
  require(result.target_sources.size() == count,
          label + ": target source count mismatch");
  require(result.target_path_offsets.size() == count + 1,
          label + ": target node-offset count mismatch");
  require(result.target_edge_offsets.size() == count + 1,
          label + ": target edge-offset count mismatch");
  require(result.target_path_offsets.front() == 0,
          label + ": first node offset is not zero");
  require(result.target_edge_offsets.front() == 0,
          label + ": first edge offset is not zero");
  require(result.target_path_offsets.back() ==
              static_cast<int>(result.target_path_nodes.size()),
          label + ": final node offset does not match compact storage");
  require(result.target_edge_offsets.back() ==
              static_cast<int>(result.target_path_edges.size()),
          label + ": final edge offset does not match compact storage");

  bool all_reached = true;
  for (std::size_t index = 0; index < count; ++index) {
    const int target = targets[index];
    const float expected_distance =
        expected[static_cast<std::size_t>(target)];
    const int node_begin = result.target_path_offsets[index];
    const int node_end = result.target_path_offsets[index + 1];
    const int edge_begin = result.target_edge_offsets[index];
    const int edge_end = result.target_edge_offsets[index + 1];
    require(node_begin >= 0 && node_begin <= node_end &&
                node_end <=
                    static_cast<int>(result.target_path_nodes.size()),
            label + ": target node slice is invalid");
    require(edge_begin >= 0 && edge_begin <= edge_end &&
                edge_end <=
                    static_cast<int>(result.target_path_edges.size()),
            label + ": target edge slice is invalid");

    if (!std::isfinite(expected_distance)) {
      all_reached = false;
      require(std::isinf(result.target_distances[index]) &&
                  result.target_sources[index] == -1 &&
                  node_begin == node_end && edge_begin == edge_end,
              label + ": unreachable target exposed a path");
      continue;
    }

    require(close_enough(expected_distance, result.target_distances[index]),
            label + ": target distance mismatch");
    require(node_end - node_begin >= 1,
            label + ": reached target has an empty node path");
    require(edge_end - edge_begin == node_end - node_begin - 1,
            label + ": reached target path has inconsistent edge count");
    const int source = result.target_sources[index];
    require(std::find(sources.begin(), sources.end(), source) != sources.end(),
            label + ": target source was not requested");
    require(result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                source,
            label + ": compact path does not start at target source");
    require(result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                target,
            label + ": compact path does not end at target");

    if (std::find(sources.begin(), sources.end(), target) != sources.end()) {
      require(source == target && node_end - node_begin == 1 &&
                  edge_begin == edge_end,
              label + ": requested-source target lost its identity path");
    }

    float path_distance = 0.0f;
    for (int step = 0; step < edge_end - edge_begin; ++step) {
      const int from =
          result.target_path_nodes[static_cast<std::size_t>(node_begin + step)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + step + 1)];
      const Offset edge = result.target_path_edges[
          static_cast<std::size_t>(edge_begin + step)];
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from + 1)],
              label + ": returned edge ID is outside its source CSR row");
      require(graph.colind[static_cast<std::size_t>(edge)] == to,
              label + ": returned edge ID has the wrong destination");
      path_distance += effective_edge_weight(graph, edge, vertex_costs);
    }
    require(close_enough(path_distance, result.target_distances[index]),
            label + ": compact path cost does not match target distance");
  }
  require(result.target_reached == all_reached,
          label + ": aggregate target_reached flag mismatch");
  require(result.converged || result.stopped_on_target,
          label + ": unlimited target run neither converged nor settled");
}

void validate_full_distances(const std::string& label,
                             const std::vector<float>& expected,
                             const NearFarCsrResult& result) {
  require(result.converged, label + ": full SSSP did not converge");
  require(!result.stopped_on_target,
          label + ": full SSSP reported target settlement");
  require(result.dist.size() == expected.size(),
          label + ": full distance count mismatch");
  require(result.pred_node.size() == expected.size() &&
              result.pred_edge.size() == expected.size(),
          label + ": full predecessor count mismatch");
  for (std::size_t vertex = 0; vertex < expected.size(); ++vertex) {
    require(close_enough(expected[vertex], result.dist[vertex]),
            label + ": full distance mismatch at vertex " +
                std::to_string(vertex));
  }
}

NearFarCsrResult run_targets_and_check(
    const std::string& label,
    NearFarCsrWorkspace& workspace,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    hipStream_t stream,
    const std::vector<float>* vertex_costs = nullptr) {
  const std::vector<float> expected =
      cpu_dijkstra(graph, sources, vertex_costs);
  NearFarCsrResult result =
      workspace.run(sources, targets, delta, -1, stream, nullptr, nullptr);
  validate_target_result(
      label, graph, sources, targets, expected, result, vertex_costs);
  return result;
}

void record_progress(const NearFarCsrProgress& progress, void* user_data) {
  static_cast<std::vector<NearFarCsrProgress>*>(user_data)
      ->push_back(progress);
}

void require_same_compact_result(const std::string& label,
                                 const NearFarCsrResult& expected,
                                 const NearFarCsrResult& actual) {
  require(expected.target_distances == actual.target_distances,
          label + ": target distances differ");
  require(expected.target_sources == actual.target_sources,
          label + ": target sources differ");
  require(expected.target_path_offsets == actual.target_path_offsets,
          label + ": node offsets differ");
  require(expected.target_edge_offsets == actual.target_edge_offsets,
          label + ": edge offsets differ");
  require(expected.target_path_nodes == actual.target_path_nodes,
          label + ": path nodes differ");
  require(expected.target_path_edges == actual.target_path_edges,
          label + ": path edges differ");
  require(expected.target_reached == actual.target_reached,
          label + ": target_reached differs");
}

void test_exact_unit_dispatch_epoch_wrap_and_reuse(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      7,
      {{0, 2, 1.0f},
       {1, 2, 1.0f},
       {0, 3, 1.0f},
       {1, 3, 1.0f},
       {2, 4, 1.0f},
       {2, 4, 1.0f},
       {3, 4, 1.0f},
       {4, 5, 1.0f},
       {5, 6, 1.0f}});
  const std::vector<int> sources{1, 0, 0};
  const std::vector<int> targets{0, 2, 4, 6, 2, 1};
  NearFarCsrWorkspace workspace(
      graph, stream, NearFarCsrWorkspaceOptions{3, 2});

  near_far_internal_reset_optimization_counters();
  const NearFarCsrResult controller = run_targets_and_check(
      "exact-unit controller",
      workspace,
      graph,
      sources,
      targets,
      1.0f,
      stream);
  require(near_far_internal_unit_controller_count() == 1,
          "exact-unit graph did not use the Near-Far controller");
  require(near_far_internal_shard_count_copy_count() == 0,
          "normal execution copied queue shard counts to the host");

  near_far_internal_force_generic(1);
  const NearFarCsrResult generic = run_targets_and_check(
      "forced generic exact-unit graph",
      workspace,
      graph,
      sources,
      targets,
      1.0f,
      stream);
  near_far_internal_force_generic(0);
  require_same_compact_result(
      "exact-unit controller versus generic", controller, generic);
  require(near_far_internal_controller_fallback_count() >= 1,
          "force-generic hook did not exercise controller fallback");

  near_far_internal_force_epoch_wrap(1);
  const NearFarCsrResult wrapped = run_targets_and_check(
      "forced packed-epoch wrap",
      workspace,
      graph,
      sources,
      targets,
      1.0f,
      stream);
  require_same_compact_result(
      "packed epoch wrap", controller, wrapped);

  near_far_internal_reset_optimization_counters();
  const NearFarCsrResult reused = run_targets_and_check(
      "allocation-free warmed reuse",
      workspace,
      graph,
      sources,
      targets,
      1.0f,
      stream);
  require_same_compact_result("warmed reuse", controller, reused);
  require(near_far_internal_device_allocation_count() == 0 &&
              near_far_internal_pinned_allocation_count() == 0 &&
              near_far_internal_target_growth_count() == 0 &&
              near_far_internal_path_growth_count() == 0,
          "warmed same-shape query allocated or grew persistent buffers");
  require(near_far_internal_status_copy_count() > 0 &&
              near_far_internal_path_transfer_count() == 1 &&
              near_far_internal_shard_count_copy_count() == 0,
          "optimized status/path transfer accounting is inconsistent");
}

void test_weighted_paths_and_ties(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      9,
      {{0, 0, 0.0f},
       {0, 1, 2.0f},
       {0, 1, 1.0f},
       {0, 1, 1.0f},
       {0, 2, 1.0f},
       {1, 3, 0.0f},
       {2, 3, 0.0f},
       {3, 1, 0.0f},
       {3, 4, 2.0f},
       {2, 4, 5.0f},
       {4, 5, 100.0f},
       {0, 5, 500.0f},
       {6, 6, 0.0f}});
  NearFarCsrWorkspace workspace(graph, stream);
  const std::vector<int> sources{6, 0, 0};
  const std::vector<int> targets{0, 3, 5, 7, 3, 6};
  const NearFarCsrResult result = run_targets_and_check(
      "weighted mixed/duplicate targets",
      workspace,
      graph,
      sources,
      targets,
      1.0f,
      stream);
  require(result.target_path_offsets[1] ==
              result.target_path_offsets[0] + 1,
          "source target did not use a one-node identity path");

  const HostCsrF32 source_tie_graph = make_outgoing_csr(
      4, {{0, 1, 0.0f}, {0, 2, 0.0f}, {1, 2, 0.0f}, {2, 3, 1.0f}});
  NearFarCsrWorkspace source_tie_workspace(source_tie_graph, stream);
  const NearFarCsrResult source_ties = run_targets_and_check(
      "zero-weight source ties",
      source_tie_workspace,
      source_tie_graph,
      {1, 0},
      {0, 1, 2, 3},
      0.5f,
      stream);
  require(source_ties.target_sources == std::vector<int>({0, 1, 0, 0}),
          "identity/lower-source zero-weight tie ordering is unstable");

  const HostCsrF32 parallel_graph =
      make_outgoing_csr(4, {{0, 3, 1.0f}, {0, 3, 1.0f}});
  NearFarCsrWorkspace parallel_workspace(parallel_graph, stream);
  const NearFarCsrResult parallel = run_targets_and_check(
      "equal parallel edges",
      parallel_workspace,
      parallel_graph,
      {0},
      {3},
      1.0f,
      stream);
  require(parallel.target_path_edges == std::vector<Offset>({0}),
          "equal parallel edges did not select the lower original CSR edge ID");
}

void test_thresholds_stale_entries_and_spill(hipStream_t stream) {
  const float below = std::nextafter(1.0f, 0.0f);
  const float above =
      std::nextafter(1.0f, std::numeric_limits<float>::infinity());
  const HostCsrF32 boundary_graph = make_outgoing_csr(
      7,
      {{0, 1, below},
       {0, 2, 1.0f},
       {0, 3, above},
       {1, 4, 0.25f},
       {2, 4, 0.125f},
       {3, 5, 8.0f},
       {4, 5, 0.0f},
       {5, 6, 64.0f}});
  NearFarCsrWorkspace boundary_workspace(boundary_graph, stream);
  run_targets_and_check("threshold boundaries and broad weights",
                        boundary_workspace,
                        boundary_graph,
                        {0},
                        {1, 2, 3, 4, 5, 6},
                        1.0f,
                        stream);

  const HostCsrF32 stale_graph = make_outgoing_csr(
      5,
      {{0, 1, 10.0f},
       {0, 2, 1.0f},
       {2, 1, 1.0f},
       {1, 3, 1.0f},
       {2, 3, 20.0f},
       {3, 4, 0.0f}});
  NearFarCsrWorkspace stale_workspace(stale_graph, stream);
  run_targets_and_check("stale deferred improvement",
                        stale_workspace,
                        stale_graph,
                        {0},
                        {1, 3, 4},
                        1.0f,
                        stream);

  std::vector<EdgeSpec> spill_edges;
  for (int vertex = 1; vertex <= 31; ++vertex) {
    spill_edges.push_back({0, vertex, 0.25f});
    spill_edges.push_back({vertex, 32, 1.0f});
  }
  const HostCsrF32 spill_graph = make_outgoing_csr(33, spill_edges);
  const NearFarCsrWorkspaceOptions tiny_urgent{
      2,
      1,
  };
  NearFarCsrWorkspace spill_workspace(
      spill_graph, stream, tiny_urgent);
  const std::vector<float> expected = cpu_dijkstra(spill_graph, {0});
  const NearFarCsrResult full = spill_workspace.run_distances(
      {0}, 1.0f, -1, stream, nullptr, nullptr);
  validate_full_distances("checked urgent spill", expected, full);
  run_targets_and_check("checked urgent spill compact paths",
                        spill_workspace,
                        spill_graph,
                        {0},
                        {1, 16, 31, 32},
                        1.0f,
                        stream);
}

void test_limits_callbacks_reuse_and_costs(hipStream_t stream) {
  const HostCsrF32 chain =
      make_outgoing_csr(5, {{0, 1, 1.0f},
                            {1, 2, 1.0f},
                            {2, 3, 1.0f},
                            {3, 4, 1.0f}});
  NearFarCsrWorkspace workspace(chain, stream);

  const NearFarCsrResult zero_iterations =
      workspace.run(std::vector<int>{0},
                    std::vector<int>{0, 1},
                    1.0f,
                    0,
                    stream,
                    nullptr,
                    nullptr);
  require(!zero_iterations.converged &&
              zero_iterations.iterations_used == 0 &&
              zero_iterations.target_distances[0] == 0.0f &&
              std::isinf(zero_iterations.target_distances[1]) &&
              zero_iterations.target_path_nodes == std::vector<int>({0}),
          "max_iters=0 exposed non-source tentative state");

  std::vector<NearFarCsrProgress> records;
  const NearFarCsrResult capped = workspace.run(
      std::vector<int>{0},
      std::vector<int>{4},
      1.0f,
      1,
      stream,
      record_progress,
      &records);
  require(!capped.converged && !capped.stopped_on_target &&
              capped.iterations_used == 1 &&
              !capped.target_reached && records.size() == 1,
          "capped run/callback semantics are inconsistent");
  require(records[0].iteration == 1 && records[0].max_iters == 1 &&
              records[0].convergence_checked && records[0].changed,
          "progress callback payload is incorrect");

  run_targets_and_check(
      "workspace reuse source zero", workspace, chain, {0}, {4}, 0.75f, stream);
  run_targets_and_check(
      "workspace reuse source two", workspace, chain, {2}, {0, 4}, 2.0f, stream);

  std::vector<EdgeSpec> fan_edges;
  for (int vertex = 1; vertex < 48; ++vertex) {
    fan_edges.push_back({0, vertex, static_cast<float>(vertex % 7)});
    fan_edges.push_back({vertex, 63, static_cast<float>(48 - vertex)});
  }
  fan_edges.push_back({63, 64, 0.0f});
  fan_edges.push_back({64, 65, 1.0f});
  const HostCsrF32 alternating = make_outgoing_csr(66, fan_edges);
  NearFarCsrWorkspace alternating_workspace(
      alternating,
      stream,
      NearFarCsrWorkspaceOptions{4, 3});
  const NearFarCsrResult deferred_exit = alternating_workspace.run(
      std::vector<int>{0},
      std::vector<int>{65},
      1.0f,
      1,
      stream,
      nullptr,
      nullptr);
  require(!deferred_exit.converged && !deferred_exit.stopped_on_target,
          "deferred-work capped run stopped unexpectedly");
  run_targets_and_check("reuse after deferred queues large target set",
                        alternating_workspace,
                        alternating,
                        {0, 7, 13, 13},
                        {0, 1, 2, 3, 7, 13, 31, 47, 63, 65},
                        2.0f,
                        stream);
  run_targets_and_check("reuse after deferred queues small target set",
                        alternating_workspace,
                        alternating,
                        {64},
                        {0, 65},
                        0.5f,
                        stream);
  const std::vector<float> alternating_expected =
      cpu_dijkstra(alternating, {63});
  validate_full_distances(
      "full distances after alternating target reuse",
      alternating_expected,
      alternating_workspace.run_distances(
          std::vector<int>{63}, 1.0f, -1, stream, nullptr, nullptr));

  const HostCsrF32 cost_graph = make_outgoing_csr(
      3, {{0, 1, 2.0f}, {0, 2, 1.0f}, {2, 1, 1.0f}});
  NearFarCsrWorkspace cost_workspace(cost_graph, stream);
  const std::vector<float> costs{1.0f, 3.0f, 0.5f};
  cost_workspace.update_vertex_costs(costs, stream);
  run_targets_and_check("destination vertex costs",
                        cost_workspace,
                        cost_graph,
                        {0},
                        {1, 2},
                        1.0f,
                        stream,
                        &costs);
  cost_workspace.clear_vertex_costs(stream);
  run_targets_and_check("cleared destination vertex costs",
                        cost_workspace,
                        cost_graph,
                        {0},
                        {1, 2},
                        1.0f,
                        stream);
}

void test_early_settlement_and_scalar_api(hipStream_t stream) {
  std::vector<EdgeSpec> edges{{0, 1, 1.0f}, {0, 2, 100.0f}};
  for (int vertex = 2; vertex < 63; ++vertex) {
    edges.push_back({vertex, vertex + 1, 1.0f});
  }
  const HostCsrF32 graph = make_outgoing_csr(64, edges);
  NearFarCsrWorkspace workspace(graph, stream);
  const NearFarCsrResult result =
      workspace.run(0, 1, 1.0f, -1, stream, nullptr, nullptr);
  require(result.target == 1 && result.target_reached &&
              close_enough(result.target_distance, 1.0f) &&
              result.stopped_on_target && !result.converged &&
              result.target_path_nodes == std::vector<int>({0, 1}),
          "safe early target settlement did not stop before distant work");
}

void test_validation_and_singleton() {
  const HostCsrF32 singleton = make_outgoing_csr(1, {});
  NearFarCsrWorkspace workspace(singleton, nullptr);
  const NearFarCsrResult identity = run_targets_and_check(
      "zero-edge singleton", workspace, singleton, {0, 0}, {0, 0}, 1.0f, nullptr);
  require(identity.target_path_nodes == std::vector<int>({0, 0}),
          "duplicate singleton targets did not preserve order");

  require_throws<std::invalid_argument>("empty sources", [&] {
    (void)workspace.run(std::vector<int>{},
                        std::vector<int>{0},
                        1.0f,
                        -1,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::invalid_argument>("empty targets", [&] {
    (void)workspace.run(std::vector<int>{0},
                        std::vector<int>{},
                        1.0f,
                        -1,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::invalid_argument>("zero delta", [&] {
    (void)workspace.run(std::vector<int>{0},
                        std::vector<int>{0},
                        0.0f,
                        -1,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::invalid_argument>("invalid max iterations", [&] {
    (void)workspace.run(std::vector<int>{0},
                        std::vector<int>{0},
                        1.0f,
                        -2,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::out_of_range>("invalid source", [&] {
    (void)workspace.run(std::vector<int>{1},
                        std::vector<int>{0},
                        1.0f,
                        -1,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::out_of_range>("invalid target", [&] {
    (void)workspace.run(std::vector<int>{0},
                        std::vector<int>{1},
                        1.0f,
                        -1,
                        nullptr,
                        nullptr,
                        nullptr);
  });
  require_throws<std::invalid_argument>("wrong vertex cost count", [&] {
    workspace.update_vertex_costs({}, nullptr);
  });
  require_throws<std::invalid_argument>("negative vertex cost", [&] {
    workspace.update_vertex_costs({-1.0f}, nullptr);
  });

  HostCsrF32 empty;
  empty.rows = 0;
  empty.cols = 0;
  empty.nnz = 0;
  empty.rowptr = {0};
  require_throws<std::invalid_argument>("empty graph", [&] {
    NearFarCsrGraph graph(empty, nullptr);
  });

  HostCsrF32 negative = make_outgoing_csr(2, {{0, 1, 1.0f}});
  negative.values[0] = -1.0f;
  require_throws<std::invalid_argument>("negative edge weight", [&] {
    NearFarCsrGraph graph(negative, nullptr);
  });
  require_throws<std::invalid_argument>("too many queue shards", [&] {
    NearFarCsrWorkspace bad(singleton,
                            nullptr,
                            NearFarCsrWorkspaceOptions{2, 0});
  });
  require_throws<std::invalid_argument>("oversized urgent queue", [&] {
    NearFarCsrWorkspace bad(singleton,
                            nullptr,
                            NearFarCsrWorkspaceOptions{1, 2});
  });
}

void test_shared_graph_stream_affinity_and_concurrency() {
  const HostCsrF32 graph = make_outgoing_csr(
      8,
      {{0, 1, 0.5f},
       {1, 2, 2.0f},
       {0, 3, 3.0f},
       {3, 2, 0.25f},
       {2, 4, 1.0f},
       {5, 6, 0.0f},
       {6, 4, 1.0f},
       {4, 7, 4.0f}});
  HipStream stream_a;
  HipStream stream_b;
  auto shared_graph =
      std::make_shared<NearFarCsrGraph>(graph, nullptr);

  NearFarCsrWorkspace affinity_workspace(shared_graph, stream_a.get());
  require_throws<std::invalid_argument>("wrong stream run", [&] {
    (void)affinity_workspace.run(std::vector<int>{0},
                                 std::vector<int>{4},
                                 1.0f,
                                 -1,
                                 stream_b.get(),
                                 nullptr,
                                 nullptr);
  });
  require_throws<std::invalid_argument>("wrong stream cost update", [&] {
    affinity_workspace.update_vertex_costs(
        std::vector<float>(8, 1.0f), stream_b.get());
  });

  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");
  auto run_async = [shared_graph, graph, device](hipStream_t stream,
                                                 std::vector<int> sources,
                                                 std::vector<int> targets,
                                                 const std::string& label) {
    check_hip(hipSetDevice(device), "hipSetDevice");
    NearFarCsrWorkspace workspace(shared_graph, stream);
    return run_targets_and_check(
        label, workspace, graph, sources, targets, 1.25f, stream);
  };

  auto first = std::async(std::launch::async,
                          run_async,
                          stream_a.get(),
                          std::vector<int>{0},
                          std::vector<int>{2, 4, 7},
                          "concurrent stream A");
  auto second = std::async(std::launch::async,
                           run_async,
                           stream_b.get(),
                           std::vector<int>{5},
                           std::vector<int>{4, 7, 0},
                           "concurrent stream B");
  (void)first.get();
  (void)second.get();

  NearFarCsrWorkspace retained(shared_graph, nullptr);
  *shared_graph = NearFarCsrGraph(make_outgoing_csr(1, {}), nullptr);
  run_targets_and_check("retained shared graph after wrapper move",
                        retained,
                        graph,
                        {0},
                        {7},
                        1.25f,
                        nullptr);
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const hipError_t count_status = hipGetDeviceCount(&device_count);
    if (count_status != hipSuccess || device_count == 0) {
      std::cout << "Near-Far HIP test skipped: no HIP device available\n";
      return 0;
    }

    HipStream stream;
    test_exact_unit_dispatch_epoch_wrap_and_reuse(stream.get());
    test_weighted_paths_and_ties(stream.get());
    test_thresholds_stale_entries_and_spill(stream.get());
    test_limits_callbacks_reuse_and_costs(stream.get());
    test_early_settlement_and_scalar_api(stream.get());
    test_validation_and_singleton();
    test_shared_graph_stream_affinity_and_concurrency();
    std::cout << "Near-Far HIP test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Near-Far HIP test failed: " << error.what() << '\n';
    return 1;
  }
}
