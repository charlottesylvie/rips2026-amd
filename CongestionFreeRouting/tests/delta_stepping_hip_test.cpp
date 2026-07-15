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

void record_progress(const DeltaSteppingCsrProgress& progress,
                     void* user_data) {
  auto* records =
      static_cast<std::vector<DeltaSteppingCsrProgress>*>(user_data);
  records->push_back(progress);
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

    std::vector<DeltaSteppingCsrProgress> progress;
    const DeltaSteppingCsrResult generic = workspace.run(
        sources, targets, kSaturatingDelta, -1, stream,
        record_progress, &progress);
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
        {{0, 1, 1.0f}, {1, 2, 1.0f}, {2, 3, 1.0f}});
    DeltaSteppingCsrWorkspace workspace(graph, stream);
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

    std::vector<DeltaSteppingCsrProgress> progress;
    const DeltaSteppingCsrResult callback_result = workspace.run(
        std::vector<int>{0}, std::vector<int>{3},
        1.0f, -1, stream, record_progress, &progress);
    const std::vector<float> expected =
        cpu_dijkstra_outgoing_multi_source(graph, {0});
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
            "unit-graph callback fallback did not stop on its target");
  }
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
    // full- and partial-wave coalesced reservations. Every vertex is touched.
    run_full_and_target_checks("wave-coalesced distinct appends",
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
  workspace.update_values(current.values, stream);
  run_full_and_target_checks("updated edge values",
                             workspace,
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
  workspace.update_vertex_costs(vertex_costs, stream);
  run_full_and_target_checks("destination vertex costs",
                             workspace,
                             current,
                             primary_sources,
                             primary_targets,
                             2.0f,
                             stream,
                             &vertex_costs,
                             true);
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
    HipStream stream;
    test_empty_and_singleton_graphs(stream.get());
    test_unit_weight_specialization(stream.get());
    test_zero_weight_scc_predecessors(stream.get());
    test_float_bucket_boundaries_and_saturation(stream.get());
    test_batched_closure_and_iteration_fallback(stream.get());
    test_wave_boundary_contention(stream.get());
    test_shared_graph_workspaces(stream.get());
    test_workspace_reuse_and_updates(stream.get());
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
