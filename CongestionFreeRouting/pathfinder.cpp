#include "pathfinder.hpp"

#include "bellman_ford/bf10.hpp"
#include "delta_stepping/delta_stepping_hip_CSR.hpp"
#include "profiling/roctx_ranges.hpp"
#include "unit_bfs/unit_bfs_hip_CSR.hpp"

// One-shot shortest-path router for the repository PathFinder flow.
//
// This keeps the same benchmark-facing and route JSON APIs, but the routing
// pass intentionally ignores present/historical congestion.  The default
// engine uses a unit-weight GPU BFS specialized for the converter's unit
// routing graph. GPU delta-stepping and Bellman-Ford bf10 remain selectable for
// comparison.
//
// Example GPU build from the repository root:
//   hipcc -std=c++17 -O3 -x hip -DBF10_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/pathfinder.cpp \
//     CongestionFreeRouting/bellman_ford/bf10.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     -pthread \
//     -o congestion_free_pathfinder
// Add -DPATHFINDER_ENABLE_ROCTX -lrocprofiler-sdk-roctx for profiler ranges.
//
// Run:
//   ./congestion_free_pathfinder design.csrbin design.csrbin.ifmeta.bin --net-limit 10

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace routing {
namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 4;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

int read_route_node(std::ifstream& in, const char* name) {
  const std::uint64_t raw = read_u64(in, name);
  if (raw == kNoIndex) {
    return -1;
  }
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) {
    return;
  }
  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

std::string read_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large for this host");
  }

  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed while reading metadata string bytes");
    }
  }
  return text;
}

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }
  for (minplus_sparse::Offset row = 0; row < graph.rows; ++row) {
    const minplus_sparse::Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const minplus_sparse::Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<minplus_sparse::Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

void validate_options(const PathfinderOptions& options) {
  if (options.sssp_engine != SsspEngine::kDeltaStep) {
    if (options.delta_force_generic ||
        options.delta_force_legacy_parent ||
        options.delta_telemetry ||
        options.delta_auto ||
        options.delta_multiplier != 1.0f ||
        options.delta_controls_explicit) {
      throw std::invalid_argument(
          "Delta-Stepping controls require --sssp-engine delta-step or "
          "--use-delta-step");
    }
  } else {
    if (!(options.delta_multiplier > 0.0f) ||
        !std::isfinite(options.delta_multiplier)) {
      throw std::invalid_argument(
          "PathFinder automatic delta multiplier must be finite and positive");
    }
    if (!options.delta_auto) {
      if (!(options.delta > 0.0f) || !std::isfinite(options.delta)) {
        throw std::invalid_argument(
            "PathFinder delta must be finite and positive");
      }
      if (options.delta_multiplier != 1.0f) {
        throw std::invalid_argument(
            "--delta-multiplier requires --delta auto");
      }
    }
  }
  if (options.capacity <= 0) {
    throw std::invalid_argument("capacity must be positive");
  }
}

bool valid_node(int node, minplus_sparse::Offset rows) {
  return node >= 0 && static_cast<minplus_sparse::Offset>(node) < rows;
}

void add_unique_node(std::vector<int>& nodes,
                     std::vector<std::uint32_t>& seen,
                     std::uint32_t tree_stamp,
                     int node) {
  if (node < 0 || static_cast<std::size_t>(node) >= seen.size()) {
    return;
  }
  if (seen[static_cast<std::size_t>(node)] == tree_stamp) {
    return;
  }
  seen[static_cast<std::size_t>(node)] = tree_stamp;
  nodes.push_back(node);
}

std::uint32_t next_tree_stamp(std::vector<std::uint32_t>& seen,
                              std::uint32_t* current_stamp) {
  if (*current_stamp == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(seen.begin(), seen.end(), 0);
    *current_stamp = 0;
  }
  ++(*current_stamp);
  return *current_stamp;
}

std::vector<int> nodes_from_path(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

std::uint64_t hash_path_nodes(const std::vector<int>& nodes,
                              std::size_t begin,
                              std::size_t end) {
  if (begin > end || end > nodes.size()) {
    return 0;
  }
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t i = begin; i < end; ++i) {
    std::uint32_t value = static_cast<std::uint32_t>(nodes[i]);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<std::uint8_t>(value & UINT32_C(0xff));
      hash *= UINT64_C(1099511628211);
      value >>= 8;
    }
  }
  return hash;
}

std::uint64_t hash_path_nodes(const std::vector<int>& nodes) {
  return hash_path_nodes(nodes, 0, nodes.size());
}

struct CpuUnitBfsObservation {
  int distance = -1;
  int source = -1;
  std::size_t edge_count = 0;
  std::uint64_t path_hash = 0;
};

CpuUnitBfsObservation cpu_unit_bfs_observation(
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    int target) {
  CpuUnitBfsObservation observation;
  if (!valid_node(target, graph.rows)) {
    return observation;
  }

  const std::size_t row_count = static_cast<std::size_t>(graph.rows);
  std::vector<int> parent(row_count, -1);
  std::vector<int> owner(row_count, -1);
  std::vector<int> distance(row_count, -1);
  std::vector<int> queue;
  queue.reserve(row_count);
  for (const int source : sources) {
    if (!valid_node(source, graph.rows) ||
        distance[static_cast<std::size_t>(source)] >= 0) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(source);
    parent[index] = source;
    owner[index] = source;
    distance[index] = 0;
    queue.push_back(source);
  }

  for (std::size_t head = 0;
       head < queue.size() && distance[static_cast<std::size_t>(target)] < 0;
       ++head) {
    const int u = queue[head];
    const int next_distance = distance[static_cast<std::size_t>(u)] + 1;
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t v_index = static_cast<std::size_t>(v);
      if (distance[v_index] >= 0) {
        continue;
      }
      parent[v_index] = u;
      owner[v_index] = owner[static_cast<std::size_t>(u)];
      distance[v_index] = next_distance;
      queue.push_back(v);
      if (v == target) {
        break;
      }
    }
  }

  const std::size_t target_index = static_cast<std::size_t>(target);
  if (distance[target_index] < 0) {
    return observation;
  }

  observation.distance = distance[target_index];
  observation.source = owner[target_index];
  observation.edge_count = static_cast<std::size_t>(observation.distance);
  std::vector<int> reversed;
  reversed.reserve(observation.edge_count + 1);
  for (int current = target;;
       current = parent[static_cast<std::size_t>(current)]) {
    reversed.push_back(current);
    if (parent[static_cast<std::size_t>(current)] == current) {
      break;
    }
  }
  std::reverse(reversed.begin(), reversed.end());
  observation.path_hash = hash_path_nodes(reversed);
  return observation;
}

template <typename SsspResult>
UnitBfsPathObservation target_observation(const SsspResult& result,
                                          std::size_t target_position,
                                          int target) {
  UnitBfsPathObservation observation;
  observation.captured = true;

  if (target_position < result.target_distances.size()) {
    observation.distance = result.target_distances[target_position];
    observation.reached = std::isfinite(observation.distance);
    if (target_position < result.target_sources.size()) {
      observation.source = result.target_sources[target_position];
    }
    if (target_position + 1 < result.target_path_offsets.size() &&
        target_position + 1 < result.target_edge_offsets.size()) {
      const int node_begin = result.target_path_offsets[target_position];
      const int node_end = result.target_path_offsets[target_position + 1];
      const int edge_begin = result.target_edge_offsets[target_position];
      const int edge_end = result.target_edge_offsets[target_position + 1];
      if (node_begin >= 0 && node_end >= node_begin && edge_begin >= 0 &&
          edge_end >= edge_begin &&
          static_cast<std::size_t>(node_end) <= result.target_path_nodes.size() &&
          static_cast<std::size_t>(edge_end) <= result.target_path_edges.size()) {
        observation.edge_count = static_cast<std::size_t>(edge_end - edge_begin);
        observation.path_hash = hash_path_nodes(
            result.target_path_nodes,
            static_cast<std::size_t>(node_begin),
            static_cast<std::size_t>(node_end));
      }
    }
    return observation;
  }

  if (valid_node(target, static_cast<minplus_sparse::Offset>(result.dist.size())) &&
      std::isfinite(result.dist[static_cast<std::size_t>(target)])) {
    observation.reached = true;
    observation.distance = result.dist[static_cast<std::size_t>(target)];
  }
  return observation;
}

bool observation_matches_cpu(const UnitBfsPathObservation& observation,
                             int cpu_distance) {
  if (!observation.captured) {
    return false;
  }
  if (cpu_distance < 0) {
    return !observation.reached;
  }
  return observation.reached &&
         observation.distance == static_cast<float>(cpu_distance) &&
         observation.edge_count == static_cast<std::size_t>(cpu_distance);
}

void classify_unit_bfs_diagnostic(UnitBfsPathDiagnostic* diagnostic) {
  if (diagnostic == nullptr) {
    return;
  }
  if (!diagnostic->all_unit_weights) {
    diagnostic->classification = "unsupported_non_unit_graph";
  } else if (!diagnostic->raw_batched.captured ||
             !diagnostic->fresh_original.captured ||
             !diagnostic->fresh_expanded_tree.captured) {
    diagnostic->classification = "diagnostic_not_captured";
  } else if (!observation_matches_cpu(diagnostic->fresh_original,
                                      diagnostic->cpu_original_distance)) {
    diagnostic->classification = "unit_bfs_single_target_mismatch";
  } else if (!observation_matches_cpu(diagnostic->raw_batched,
                                      diagnostic->cpu_original_distance)) {
    diagnostic->classification = "unit_bfs_batched_or_reuse_mismatch";
  } else if (!observation_matches_cpu(
                 diagnostic->fresh_expanded_tree,
                 diagnostic->cpu_expanded_tree_distance)) {
    diagnostic->classification = "unit_bfs_expanded_tree_mismatch";
  } else if (diagnostic->cpu_expanded_tree_distance < 0) {
    diagnostic->classification = diagnostic->attached_reached
                                     ? "pathfinder_attached_unreachable_sink"
                                     : "unreachable_consistently";
  } else if (!diagnostic->attached_reached) {
    diagnostic->classification = "pathfinder_failed_reachable_sink";
  } else if (diagnostic->attached_edge_count >
             static_cast<std::size_t>(diagnostic->cpu_expanded_tree_distance)) {
    diagnostic->classification = "pathfinder_cached_multi_sink_path";
  } else if (diagnostic->attached_edge_count <
             static_cast<std::size_t>(diagnostic->cpu_expanded_tree_distance)) {
    diagnostic->classification = "pathfinder_tree_state_mismatch";
  } else {
    diagnostic->classification = "no_mismatch_observed";
  }
}

bool tree_contains(const std::vector<std::uint32_t>& tree_seen,
                   std::uint32_t tree_stamp,
                   int node) {
  return node >= 0 &&
         static_cast<std::size_t>(node) < tree_seen.size() &&
         tree_seen[static_cast<std::size_t>(node)] == tree_stamp;
}

bool tight_edge(float from_dist, float weight, float to_dist) {
  if (!std::isfinite(from_dist) || !std::isfinite(to_dist)) {
    return false;
  }
  const float candidate = from_dist + weight;
  const float error = std::fabs(candidate - to_dist);
  const float tolerance =
      1e-3f * std::max({1.0f, std::fabs(candidate), std::fabs(to_dist)});
  return error <= tolerance;
}

std::vector<PathEdge> reconstruct_shortest_path_from_tree_dist(
    const HostCsrF32& graph,
    const std::vector<float>& dist,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    int target,
    int* source_out) {
  *source_out = -1;
  validate_csr(graph);
  if (!valid_node(target, graph.rows)) {
    throw std::out_of_range("target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows) ||
      tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance/tree vector size does not match CSR rows");
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<int> parent(static_cast<std::size_t>(graph.rows), -1);
  std::vector<minplus_sparse::Offset> parent_edge(
      static_cast<std::size_t>(graph.rows), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.rows));
  for (int node = 0; node < graph.rows; ++node) {
    if (!tree_contains(tree_seen, tree_stamp, node) ||
        !std::isfinite(dist[static_cast<std::size_t>(node)])) {
      continue;
    }
    parent[static_cast<std::size_t>(node)] = node;
    queue.push_back(node);
  }

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    if (u == target) {
      break;
    }
    const float du = dist[static_cast<std::size_t>(u)];
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t v_index = static_cast<std::size_t>(v);
      if (parent[v_index] >= 0 || v == u) {
        continue;
      }
      if (!tight_edge(du,
                      graph.values[static_cast<std::size_t>(edge)],
                      dist[v_index])) {
        continue;
      }
      parent[v_index] = u;
      parent_edge[v_index] = edge;
      queue.push_back(v);
      if (v == target) {
        head = queue.size();
        break;
      }
    }
  }

  if (parent[static_cast<std::size_t>(target)] < 0) {
    throw std::runtime_error("could not reconstruct shortest path through outgoing CSR");
  }

  std::vector<PathEdge> reversed;
  for (int current = target;
       parent[static_cast<std::size_t>(current)] != current;) {
    const int pred = parent[static_cast<std::size_t>(current)];
    const minplus_sparse::Offset edge =
        parent_edge[static_cast<std::size_t>(current)];
    if (!valid_node(pred, graph.rows) || edge < 0) {
      throw std::runtime_error("shortest path reconstruction found an invalid predecessor");
    }
    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }

  *source_out = reversed.empty() ? target : reversed.back().from;
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::vector<PathEdge> reconstruct_shortest_path_from_tree_pred(
    const HostCsrF32& graph,
    const DeltaSteppingCsrResult& sssp,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    int target,
    int* source_out) {
  *source_out = -1;
  if (sssp.pred_node.size() != static_cast<std::size_t>(graph.rows) ||
      sssp.pred_edge.size() != static_cast<std::size_t>(graph.rows) ||
      tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    return reconstruct_shortest_path_from_tree_dist(
        graph, sssp.dist, tree_seen, tree_stamp, target, source_out);
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (minplus_sparse::Offset guard = 0; guard < graph.rows; ++guard) {
    if (tree_contains(tree_seen, tree_stamp, current)) {
      *source_out = current;
      std::reverse(reversed.begin(), reversed.end());
      return reversed;
    }

    if (!valid_node(current, graph.rows)) {
      throw std::runtime_error("predecessor path left the CSR graph");
    }
    const std::size_t current_index = static_cast<std::size_t>(current);
    const int pred = sssp.pred_node[current_index];
    const minplus_sparse::Offset edge = sssp.pred_edge[current_index];
    if (!valid_node(pred, graph.rows) ||
        edge < graph.rowptr[static_cast<std::size_t>(pred)] ||
        edge >= graph.rowptr[static_cast<std::size_t>(pred + 1)] ||
        graph.colind[static_cast<std::size_t>(edge)] != current) {
      return reconstruct_shortest_path_from_tree_dist(
          graph, sssp.dist, tree_seen, tree_stamp, target, source_out);
    }

    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }

  throw std::runtime_error("predecessor path did not reach route tree");
}

bool attach_path_if_single_parent_tree(
    const std::vector<PathEdge>& edges,
    std::vector<int>& parent_by_child,
    std::vector<std::uint32_t>& parent_seen,
    std::uint32_t tree_stamp) {
  for (const PathEdge& edge : edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    if (child >= parent_by_child.size()) {
      return false;
    }
    if (parent_seen[child] == tree_stamp && parent_by_child[child] != edge.from) {
      return false;
    }
  }
  for (const PathEdge& edge : edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    parent_seen[child] = tree_stamp;
    parent_by_child[child] = edge.from;
  }
  return true;
}

template <typename SsspWorkspace>
auto run_sssp_with_optional_delta_telemetry(
    SsspWorkspace& workspace,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iterations,
    hipStream_t stream,
    DeltaSteppingCsrTelemetry*,
    float = std::numeric_limits<float>::infinity()) {
  return workspace.run(sources,
                       targets,
                       delta,
                       max_iterations,
                       stream,
                       nullptr,
                       nullptr);
}

DeltaSteppingCsrResult run_sssp_with_optional_delta_telemetry(
    DeltaSteppingCsrWorkspace& workspace,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iterations,
    hipStream_t stream,
    DeltaSteppingCsrTelemetry* telemetry,
    float exclusive_distance_limit =
        std::numeric_limits<float>::infinity()) {
  if (telemetry == nullptr && !std::isfinite(exclusive_distance_limit)) {
    return workspace.run(sources,
                         targets,
                         delta,
                         max_iterations,
                         stream,
                         nullptr,
                         nullptr);
  }
  return workspace.run(sources,
                       targets,
                       delta,
                       max_iterations,
                       DeltaSteppingCsrRunOptions{
                           telemetry, exclusive_distance_limit},
                       stream,
                       nullptr,
                       nullptr);
}

float routed_path_cost(const RoutedSink& sink) {
  float distance = 0.0f;
  for (const PathEdge& edge : sink.edges) {
    distance += edge.cost;
  }
  return distance;
}

void trim_routed_sink_to_tree(
    RoutedSink& sink,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp) {
  if (!std::isfinite(sink.distance)) {
    return;
  }
  if (sink.nodes.empty() || sink.edges.size() + 1 != sink.nodes.size() ||
      sink.nodes.back() != sink.target) {
    throw std::runtime_error("cached route path has an invalid shape");
  }

  std::size_t last_tree_node = sink.nodes.size();
  for (std::size_t i = 0; i < sink.nodes.size(); ++i) {
    if (tree_contains(tree_seen, tree_stamp, sink.nodes[i])) {
      last_tree_node = i;
    }
  }
  if (last_tree_node == sink.nodes.size()) {
    throw std::runtime_error("cached route path is detached from the route tree");
  }

  if (last_tree_node != 0) {
    sink.nodes.erase(sink.nodes.begin(),
                     sink.nodes.begin() +
                         static_cast<std::ptrdiff_t>(last_tree_node));
    sink.edges.erase(sink.edges.begin(),
                     sink.edges.begin() +
                         static_cast<std::ptrdiff_t>(last_tree_node));
  }
  sink.source = sink.nodes.front();
  sink.distance = routed_path_cost(sink);
}

template <typename SsspResult>
bool extract_routed_sink_candidate(
    const HostCsrF32& graph,
    const SsspResult& sssp,
    std::size_t target_pos,
    std::size_t target_count,
    int target,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    RoutedSink* candidate) {
  if (candidate == nullptr) {
    throw std::invalid_argument("route candidate output is null");
  }
  *candidate = RoutedSink{};
  candidate->target = target;
  candidate->distance = std::numeric_limits<float>::infinity();

  const bool has_compact_target_paths =
      sssp.target_distances.size() == target_count &&
      sssp.target_sources.size() == target_count &&
      sssp.target_path_offsets.size() == target_count + 1 &&
      sssp.target_edge_offsets.size() == target_count + 1;
  const bool has_any_compact_target_data =
      !sssp.target_distances.empty() || !sssp.target_sources.empty() ||
      !sssp.target_path_offsets.empty() ||
      !sssp.target_edge_offsets.empty() ||
      !sssp.target_path_nodes.empty() || !sssp.target_path_edges.empty();

  if (has_compact_target_paths) {
    if (target_pos >= target_count) {
      throw std::out_of_range("route target position is outside SSSP result");
    }
    const float distance = sssp.target_distances[target_pos];
    if (!std::isfinite(distance)) {
      return false;
    }

    const int reported_source = sssp.target_sources[target_pos];
    const int node_begin = sssp.target_path_offsets[target_pos];
    const int node_end = sssp.target_path_offsets[target_pos + 1];
    const int edge_begin = sssp.target_edge_offsets[target_pos];
    const int edge_end = sssp.target_edge_offsets[target_pos + 1];
    if (!valid_node(reported_source, graph.rows) ||
        !tree_contains(tree_seen, tree_stamp, reported_source)) {
      throw std::runtime_error(
          "SSSP returned a finite compact target path with an invalid "
          "route-tree source");
    }
    if (node_begin < 0 || node_end <= node_begin || edge_begin < 0 ||
        edge_end < edge_begin ||
        static_cast<std::size_t>(node_end) > sssp.target_path_nodes.size() ||
        static_cast<std::size_t>(edge_end) > sssp.target_path_edges.size() ||
        sssp.target_path_nodes[static_cast<std::size_t>(node_begin)] !=
            reported_source ||
        sssp.target_path_nodes[static_cast<std::size_t>(node_end - 1)] !=
            target ||
        edge_end - edge_begin + 1 != node_end - node_begin) {
      std::ostringstream message;
      message << "SSSP returned a malformed compact target path"
              << " (target=" << target
              << ", target_position=" << target_pos
              << ", source=" << reported_source
              << ", node_begin=" << node_begin
              << ", node_end=" << node_end
              << ", edge_begin=" << edge_begin
              << ", edge_end=" << edge_end
              << ", source_in_tree="
              << (tree_contains(tree_seen, tree_stamp, reported_source) ? 1 : 0)
              << ", first_node="
              << (node_begin >= 0 &&
                          static_cast<std::size_t>(node_begin) <
                              sssp.target_path_nodes.size()
                      ? sssp.target_path_nodes[static_cast<std::size_t>(node_begin)]
                      : -1)
              << ", last_node="
              << (node_end > 0 &&
                          static_cast<std::size_t>(node_end) <=
                              sssp.target_path_nodes.size()
                      ? sssp.target_path_nodes[static_cast<std::size_t>(node_end - 1)]
                      : -1)
              << ')';
      throw std::runtime_error(message.str());
    }

    candidate->source = reported_source;
    candidate->nodes.assign(
        sssp.target_path_nodes.begin() + node_begin,
        sssp.target_path_nodes.begin() + node_end);
    candidate->edges.reserve(static_cast<std::size_t>(edge_end - edge_begin));
    for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
      const std::size_t path_index =
          static_cast<std::size_t>(edge_index - edge_begin);
      const int from = candidate->nodes[path_index];
      const int to = candidate->nodes[path_index + 1];
      const minplus_sparse::Offset csr_edge =
          sssp.target_path_edges[static_cast<std::size_t>(edge_index)];
      if (!valid_node(from, graph.rows) || !valid_node(to, graph.rows) ||
          csr_edge < graph.rowptr[static_cast<std::size_t>(from)] ||
          csr_edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
          graph.colind[static_cast<std::size_t>(csr_edge)] != to) {
        throw std::runtime_error("SSSP compact path contains an invalid CSR edge");
      }
      candidate->edges.push_back(
          {from, to, csr_edge, graph.values[static_cast<std::size_t>(csr_edge)]});
    }
    candidate->distance = routed_path_cost(*candidate);
    trim_routed_sink_to_tree(*candidate, tree_seen, tree_stamp);
    return true;
  }

  if (has_any_compact_target_data) {
    throw std::runtime_error("SSSP returned inconsistent compact target arrays");
  }
  if (!valid_node(target,
                  static_cast<minplus_sparse::Offset>(sssp.dist.size())) ||
      !std::isfinite(sssp.dist[static_cast<std::size_t>(target)])) {
    return false;
  }

  int source = -1;
  candidate->edges = reconstruct_shortest_path_from_tree_pred(
      graph, sssp, tree_seen, tree_stamp, target, &source);
  if (!valid_node(source, graph.rows) ||
      !tree_contains(tree_seen, tree_stamp, source)) {
    throw std::runtime_error("SSSP predecessor path has an invalid tree root");
  }
  candidate->source = source;
  candidate->nodes = nodes_from_path(source, candidate->edges);
  candidate->distance = routed_path_cost(*candidate);
  trim_routed_sink_to_tree(*candidate, tree_seen, tree_stamp);
  return true;
}

template <typename SsspWorkspace>
RoutedNet route_net(const HostCsrF32& graph,
                    SsspWorkspace& workspace,
                    const RouteRequest& request,
                    std::vector<std::uint32_t>& tree_seen,
                    std::vector<int>& parent_by_child,
                    std::vector<std::uint32_t>& parent_seen,
                    std::uint32_t tree_stamp,
                    const PathfinderOptions& options,
                    hipStream_t stream,
                    std::vector<DeltaSteppingCsrTelemetry>* delta_telemetry =
                        nullptr,
                    UnitBfsPathDiagnostic* unit_bfs_diagnostic = nullptr) {
  PATHFINDER_PROFILE_RANGE("pathfinder.route_net");
  RoutedNet net;
  net.net_string = request.net_string;
  if (tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("route tree scratch size does not match CSR rows");
  }
  if (parent_by_child.size() != static_cast<std::size_t>(graph.rows) ||
      parent_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("route parent scratch size does not match CSR rows");
  }
  if (delta_telemetry != nullptr) {
    delta_telemetry->reserve(request.sinks.size());
  }

  std::vector<int> source_candidates;
  for (const SitePinNode& source : request.sources) {
    add_unique_node(source_candidates, tree_seen, tree_stamp, source.node);
  }
  std::vector<int> diagnostic_original_sources;
  if (unit_bfs_diagnostic != nullptr) {
    diagnostic_original_sources = source_candidates;
    unit_bfs_diagnostic->target =
        request.sinks[unit_bfs_diagnostic->sink_index].node;
    unit_bfs_diagnostic->original_source_count = source_candidates.size();
    unit_bfs_diagnostic->cpu_original_distance =
        cpu_unit_bfs_observation(graph,
                                 diagnostic_original_sources,
                                 unit_bfs_diagnostic->target)
            .distance;
  }

  if (source_candidates.empty()) {
    net.reached_all_sinks = false;
    if (unit_bfs_diagnostic != nullptr) {
      unit_bfs_diagnostic->classification = "no_valid_sources";
    }
    return net;
  }

  bool reached_all = true;
  net.sinks.resize(request.sinks.size());
  std::vector<int> initial_targets;
  std::vector<std::size_t> initial_target_sink_indices;
  initial_targets.reserve(request.sinks.size());
  initial_target_sink_indices.reserve(request.sinks.size());
  for (std::size_t sink_index = 0; sink_index < request.sinks.size(); ++sink_index) {
    const SitePinNode& sink = request.sinks[sink_index];
    RoutedSink& routed_sink = net.sinks[sink_index];
    routed_sink.target = sink.node;
    routed_sink.distance = std::numeric_limits<float>::infinity();
    if (!valid_node(sink.node, graph.rows)) {
      reached_all = false;
      continue;
    }
    if (tree_contains(tree_seen, tree_stamp, sink.node)) {
      routed_sink.source = sink.node;
      routed_sink.distance = 0.0f;
      routed_sink.reached = true;
      routed_sink.nodes.push_back(sink.node);
      if (unit_bfs_diagnostic != nullptr &&
          sink_index == unit_bfs_diagnostic->sink_index) {
        unit_bfs_diagnostic->raw_batched =
            {true,
             true,
             0.0f,
             sink.node,
             0,
             hash_path_nodes(routed_sink.nodes)};
      }
      continue;
    }
    initial_targets.push_back(sink.node);
    initial_target_sink_indices.push_back(sink_index);
  }

  if (!initial_targets.empty()) {
    DeltaSteppingCsrTelemetry initial_telemetry;
    auto initial_sssp = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.sssp");
      return run_sssp_with_optional_delta_telemetry(
          workspace,
          source_candidates,
          initial_targets,
          options.delta,
          options.max_sssp_iterations,
          stream,
          delta_telemetry == nullptr ? nullptr : &initial_telemetry);
    }();
    if (delta_telemetry != nullptr && initial_telemetry.collected) {
      delta_telemetry->push_back(std::move(initial_telemetry));
    }
    const bool initial_paths_certified =
        options.sssp_engine != SsspEngine::kBellmanFord ||
        initial_sssp.stopped_on_target || initial_sssp.converged;

    for (std::size_t target_pos = 0;
         target_pos < initial_target_sink_indices.size();
         ++target_pos) {
      const std::size_t sink_index = initial_target_sink_indices[target_pos];
      const int target = request.sinks[sink_index].node;
      if (unit_bfs_diagnostic != nullptr &&
          sink_index == unit_bfs_diagnostic->sink_index) {
        unit_bfs_diagnostic->raw_batched =
            target_observation(initial_sssp, target_pos, target);
      }
      RoutedSink candidate;
      if (initial_paths_certified &&
          extract_routed_sink_candidate(graph,
                                        initial_sssp,
                                        target_pos,
                                        initial_targets.size(),
                                        target,
                                        tree_seen,
                                        tree_stamp,
                                        &candidate)) {
        net.sinks[sink_index] = std::move(candidate);
      }
    }
  }

  auto capture_diagnostic_probes = [&](std::size_t sink_index) {
    if (unit_bfs_diagnostic == nullptr ||
        sink_index != unit_bfs_diagnostic->sink_index) {
      return;
    }
    unit_bfs_diagnostic->tree_source_count = source_candidates.size();
    unit_bfs_diagnostic->prior_sinks_reached = 0;
    for (std::size_t prior = 0; prior < sink_index; ++prior) {
      if (net.sinks[prior].reached) {
        ++unit_bfs_diagnostic->prior_sinks_reached;
      }
    }

    const int target = request.sinks[sink_index].node;
    unit_bfs_diagnostic->cpu_expanded_tree_distance =
        cpu_unit_bfs_observation(graph, source_candidates, target).distance;
    auto fresh_original = workspace.run(diagnostic_original_sources,
                                        std::vector<int>{target},
                                        options.delta,
                                        options.max_sssp_iterations,
                                        stream,
                                        nullptr,
                                        nullptr);
    unit_bfs_diagnostic->fresh_original =
        target_observation(fresh_original, 0, target);
    auto fresh_expanded = workspace.run(source_candidates,
                                        std::vector<int>{target},
                                        options.delta,
                                        options.max_sssp_iterations,
                                        stream,
                                        nullptr,
                                        nullptr);
    unit_bfs_diagnostic->fresh_expanded_tree =
        target_observation(fresh_expanded, 0, target);
  };

  // The original batch already accounts for every original source.  Only
  // nodes added by earlier sink branches can improve one of its cached paths.
  std::vector<int> added_tree_sources;
  for (std::size_t sink_index = 0; sink_index < request.sinks.size(); ++sink_index) {
    const int target = request.sinks[sink_index].node;
    RoutedSink& routed_sink = net.sinks[sink_index];
    if (!valid_node(target, graph.rows)) {
      continue;
    }

    if (tree_contains(tree_seen, tree_stamp, target)) {
      routed_sink = RoutedSink{};
      routed_sink.source = target;
      routed_sink.target = target;
      routed_sink.distance = 0.0f;
      routed_sink.reached = true;
      routed_sink.nodes.push_back(target);
      capture_diagnostic_probes(sink_index);
      continue;
    }

    bool incumbent_from_capped_repair = false;
    if (!std::isfinite(routed_sink.distance) &&
        (!added_tree_sources.empty() ||
         options.sssp_engine != SsspEngine::kUnitBfs) &&
        options.max_sssp_iterations >= 0) {
      // A finite user cap can hide a target from the original batch even
      // though a branch added later places it within the same cap. UnitBFS's
      // cap is a distance radius, so the new nodes suffice. Delta/BF caps are
      // scheduler iterations; retain their legacy full-current-tree semantics.
      const std::vector<int> capped_targets{target};
      const std::vector<int>& capped_sources =
          options.sssp_engine == SsspEngine::kUnitBfs
              ? added_tree_sources
              : source_candidates;
      DeltaSteppingCsrTelemetry capped_telemetry;
      auto capped_sssp = [&]() {
        PATHFINDER_PROFILE_RANGE("pathfinder.sssp_capped_tree_repair");
        return run_sssp_with_optional_delta_telemetry(
            workspace,
            capped_sources,
            capped_targets,
            options.delta,
            options.max_sssp_iterations,
            stream,
            delta_telemetry == nullptr ? nullptr : &capped_telemetry);
      }();
      if (delta_telemetry != nullptr && capped_telemetry.collected) {
        delta_telemetry->push_back(std::move(capped_telemetry));
      }
      RoutedSink capped_candidate;
      const bool capped_candidate_found = extract_routed_sink_candidate(
          graph,
          capped_sssp,
          0,
          1,
          target,
          tree_seen,
          tree_stamp,
          &capped_candidate);
      const bool capped_candidate_certified =
          capped_candidate_found &&
          (options.sssp_engine != SsspEngine::kBellmanFord ||
           capped_sssp.stopped_on_target || capped_sssp.converged);
      if (capped_candidate_certified) {
        routed_sink = std::move(capped_candidate);
        incumbent_from_capped_repair = true;
      }
    }
    if (!std::isfinite(routed_sink.distance)) {
      reached_all = false;
      capture_diagnostic_probes(sink_index);
      continue;
    }
    trim_routed_sink_to_tree(routed_sink, tree_seen, tree_stamp);

    bool run_correction = !added_tree_sources.empty() &&
                          !incumbent_from_capped_repair;
    int correction_iteration_limit = options.max_sssp_iterations;
    int unit_strict_depth = -1;
    float exclusive_distance_limit = routed_sink.distance;
    if (options.sssp_engine == SsspEngine::kUnitBfs) {
      // A non-tree target needs at least one edge.  A cached one-edge route is
      // therefore already optimal; otherwise only depths below its incumbent
      // edge count can improve it.
      if (routed_sink.edges.size() <= 1) {
        run_correction = false;
      } else {
        const std::size_t strict_depth_size = routed_sink.edges.size() - 1;
        unit_strict_depth =
            strict_depth_size >
                    static_cast<std::size_t>(std::numeric_limits<int>::max())
                ? std::numeric_limits<int>::max()
                : static_cast<int>(strict_depth_size);
        correction_iteration_limit =
            correction_iteration_limit < 0
                ? unit_strict_depth
                : std::min(correction_iteration_limit, unit_strict_depth);
      }
      exclusive_distance_limit = std::numeric_limits<float>::infinity();
    } else if (!(exclusive_distance_limit > 0.0f)) {
      run_correction = false;
    }

    if (run_correction) {
      const std::vector<int> correction_targets{target};
      DeltaSteppingCsrTelemetry correction_telemetry;
      auto correction_sssp = [&]() {
        PATHFINDER_PROFILE_RANGE("pathfinder.sssp_tree_repair");
        return run_sssp_with_optional_delta_telemetry(
            workspace,
            added_tree_sources,
            correction_targets,
            options.delta,
            correction_iteration_limit,
            stream,
            delta_telemetry == nullptr ? nullptr : &correction_telemetry,
            exclusive_distance_limit);
      }();
      if (delta_telemetry != nullptr && correction_telemetry.collected) {
        delta_telemetry->push_back(std::move(correction_telemetry));
      }

      RoutedSink correction;
      bool found_strict_improvement = false;
      if (extract_routed_sink_candidate(graph,
                                        correction_sssp,
                                        0,
                                        1,
                                        target,
                                        tree_seen,
                                        tree_stamp,
                                        &correction)) {
        const bool is_strict_improvement =
            options.sssp_engine == SsspEngine::kUnitBfs
                ? correction.edges.size() < routed_sink.edges.size()
                : correction.distance < routed_sink.distance;
        if (is_strict_improvement) {
          found_strict_improvement = true;
        }
      }
      bool correction_certified = false;
      switch (options.sssp_engine) {
        case SsspEngine::kUnitBfs:
          correction_certified =
              found_strict_improvement ||
              correction_iteration_limit >= unit_strict_depth;
          break;
        case SsspEngine::kDeltaStep:
          correction_certified =
              found_strict_improvement ||
              correction_sssp.stopped_on_distance_limit ||
              correction_sssp.stopped_on_target ||
              correction_sssp.converged;
          break;
        case SsspEngine::kBellmanFord:
          // Unlike level-ordered BFS and settled-target Delta-Stepping, a
          // capped BF10 run can expose a finite but still-tentative target.
          correction_certified = correction_sssp.stopped_on_target ||
                                 correction_sssp.converged;
          break;
      }
      if (!correction_certified) {
        reached_all = false;
        routed_sink.edges.clear();
        routed_sink.nodes.clear();
        routed_sink.distance = std::numeric_limits<float>::infinity();
        capture_diagnostic_probes(sink_index);
        continue;
      }
      if (found_strict_improvement) {
        routed_sink = std::move(correction);
      }
    }

    capture_diagnostic_probes(sink_index);
    if (!attach_path_if_single_parent_tree(routed_sink.edges,
                                           parent_by_child,
                                           parent_seen,
                                           tree_stamp)) {
      reached_all = false;
      routed_sink.edges.clear();
      routed_sink.nodes.clear();
      routed_sink.distance = std::numeric_limits<float>::infinity();
      continue;
    }
    routed_sink.reached = true;
    for (const int node : routed_sink.nodes) {
      if (!tree_contains(tree_seen, tree_stamp, node)) {
        add_unique_node(source_candidates, tree_seen, tree_stamp, node);
        added_tree_sources.push_back(node);
      }
    }
  }

  for (const RoutedSink& routed_sink : net.sinks) {
    if (!routed_sink.reached) {
      reached_all = false;
    } else {
      for (const int node : routed_sink.nodes) {
        add_unique_node(source_candidates, tree_seen, tree_stamp, node);
      }
    }
  }

  net.reached_all_sinks = reached_all;
  net.unique_nodes = std::move(source_candidates);
  if (unit_bfs_diagnostic != nullptr) {
    const RoutedSink& selected =
        net.sinks[unit_bfs_diagnostic->sink_index];
    unit_bfs_diagnostic->attached_reached = selected.reached;
    unit_bfs_diagnostic->attached_distance = selected.distance;
    unit_bfs_diagnostic->attached_source = selected.source;
    unit_bfs_diagnostic->attached_edge_count = selected.edges.size();
    unit_bfs_diagnostic->attached_path_hash = hash_path_nodes(selected.nodes);
    classify_unit_bfs_diagnostic(unit_bfs_diagnostic);
  }
  return net;
}

void commit_net_occupancy(const RoutedNet& net, std::vector<int>& occupancy) {
  if (!net.reached_all_sinks) {
    return;
  }
  for (const int node : net.unique_nodes) {
    if (node >= 0 && static_cast<std::size_t>(node) < occupancy.size()) {
      ++occupancy[static_cast<std::size_t>(node)];
    }
  }
}

void update_congestion_stats(const std::vector<int>& occupancy,
                             int capacity,
                             int* overused_nodes,
                             int* max_occupancy) {
  *overused_nodes = 0;
  *max_occupancy = 0;
  for (const int used : occupancy) {
    *max_occupancy = std::max(*max_occupancy, used);
    if (used > capacity) {
      ++(*overused_nodes);
    }
  }
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u";
          const char* hex = "0123456789abcdef";
          out << '0' << '0' << hex[(ch >> 4) & 0xf] << hex[ch & 0xf];
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

std::uint64_t edge_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

void print_pathfinder_progress(int iteration,
                               int max_iterations,
                               std::size_t completed_nets,
                               std::size_t total_nets) {
  constexpr int kWidth = 30;
  const int filled =
      total_nets == 0 ? kWidth : static_cast<int>((completed_nets * kWidth) / total_nets);
  std::cout << "[pathfinder] iter " << iteration << "/" << max_iterations << " [";
  for (int i = 0; i < kWidth; ++i) {
    std::cout << (i < filled ? '#' : '-');
  }
  std::cout << "] " << completed_nets << "/" << total_nets << " nets\n" << std::flush;
}

hipStream_t create_worker_stream() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  hipStream_t stream = nullptr;
  const hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipStreamCreateWithFlags failed: ") +
                             hipGetErrorString(status));
  }
  return stream;
#else
  return nullptr;
#endif
}

void destroy_worker_stream(hipStream_t stream) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  if (stream != nullptr) {
    (void)hipStreamDestroy(stream);
  }
#else
  (void)stream;
#endif
}

struct WorkerStream {
  explicit WorkerStream(bool create) : owns(create) {
    if (owns) {
      stream = create_worker_stream();
    }
  }

  WorkerStream(const WorkerStream&) = delete;
  WorkerStream& operator=(const WorkerStream&) = delete;

  ~WorkerStream() {
    if (owns) {
      destroy_worker_stream(stream);
    }
  }

  hipStream_t get(hipStream_t fallback) const {
    return owns ? stream : fallback;
  }

  hipStream_t stream = nullptr;
  bool owns = false;
};

int current_worker_device() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  int device = 0;
  const hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipGetDevice failed: ") +
                             hipGetErrorString(status));
  }
  return device;
#else
  return 0;
#endif
}

int current_device_wavefront_size() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  const int device = current_worker_device();
  hipDeviceProp_t properties{};
  const hipError_t status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                             hipGetErrorString(status));
  }
  if (properties.warpSize <= 0) {
    throw std::runtime_error(
        "HIP device reported a nonpositive runtime wavefront size");
  }
  return properties.warpSize;
#else
  throw std::runtime_error(
      "automatic delta requires a HIP runtime device query");
#endif
}

void select_worker_device(int device) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  const hipError_t status = hipSetDevice(device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipSetDevice failed: ") +
                             hipGetErrorString(status));
  }
#else
  (void)device;
#endif
}

std::size_t recommend_unit_bfs_worker_count(minplus_sparse::Offset rows,
                                            std::size_t route_request_count,
                                            hipStream_t stream) {
  if (stream != nullptr || rows <= 0 || route_request_count <= 1) {
    return 1;
  }

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    return 1;
  }
  (void)total_bytes;

  // Per-worker arrays consume 20 bytes/vertex with compact edge offsets and 24
  // bytes/vertex in the wide fallback. Budget 32 bytes per vertex plus 64 MiB
  // for query/path buffers, and leave 25% of currently free memory untouched
  // for HIP runtime allocations and unusually large routes.
  constexpr std::size_t kBytesPerVertexBudget = 32;
  constexpr std::size_t kPerWorkerReserve = 64ULL * 1024ULL * 1024ULL;
  constexpr std::size_t kMaxAutoWorkers = 8;
  const std::size_t row_count = static_cast<std::size_t>(rows);
  if (row_count >
      (std::numeric_limits<std::size_t>::max() - kPerWorkerReserve) /
          kBytesPerVertexBudget) {
    return 1;
  }
  const std::size_t per_worker_bytes =
      row_count * kBytesPerVertexBudget + kPerWorkerReserve;
  const std::size_t memory_budget = free_bytes - free_bytes / 4;
  const std::size_t memory_workers =
      std::max<std::size_t>(1, memory_budget / per_worker_bytes);
  const std::size_t cpu_threads =
      std::max<unsigned int>(1, std::thread::hardware_concurrency());
  return std::max<std::size_t>(
      1,
      std::min({kMaxAutoWorkers,
                route_request_count,
                cpu_threads,
                memory_workers}));
#else
  return 1;
#endif
}

std::size_t recommend_delta_worker_count(minplus_sparse::Offset rows,
                                         std::size_t route_request_count,
                                         bool unit_specialization,
                                         hipStream_t stream) {
  if (stream != nullptr || rows <= 0 || route_request_count <= 1) {
    return 1;
  }

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    return 1;
  }
  (void)total_bytes;

  // The immutable CSR has already been uploaded once and is reflected in
  // free_bytes. Exact-unit PathFinder queries allocate only their 24-byte per
  // vertex append-only traversal state. Generic weighted fallback lazily adds
  // membership flags, bucket queues, touched state, and race-safe parent keys,
  // reaching 60 bytes per vertex. Vertex costs are also lazy and PathFinder
  // never installs them. Leave a reserve for query and compact path buffers.
  const std::size_t bytes_per_vertex_budget = unit_specialization ? 24 : 60;
  constexpr std::size_t kPerWorkerReserve = 64ULL * 1024ULL * 1024ULL;
  constexpr std::size_t kMaxAutoWorkers = 8;
  const std::size_t row_count = static_cast<std::size_t>(rows);
  if (row_count >
          (std::numeric_limits<std::size_t>::max() - kPerWorkerReserve) /
              bytes_per_vertex_budget) {
    return 1;
  }
  const std::size_t vertex_bytes = row_count * bytes_per_vertex_budget;
  if (vertex_bytes >
      std::numeric_limits<std::size_t>::max() - kPerWorkerReserve) {
    return 1;
  }
  const std::size_t per_worker_bytes = vertex_bytes + kPerWorkerReserve;
  const std::size_t memory_budget = free_bytes - free_bytes / 4;
  const std::size_t memory_workers =
      std::max<std::size_t>(1, memory_budget / per_worker_bytes);
  const std::size_t cpu_threads =
      std::max<unsigned int>(1, std::thread::hardware_concurrency());
  return std::max<std::size_t>(
      1,
      std::min({kMaxAutoWorkers,
                route_request_count,
                cpu_threads,
                memory_workers}));
#else
  return 1;
#endif
}

template <typename WorkspaceFactory>
void route_all_nets_with_workspace(const HostCsrF32& base_graph,
                                   const RoutingMetadata& metadata,
                                   const PathfinderOptions& options,
                                   hipStream_t stream,
                                   std::size_t route_request_count,
                                   std::size_t progress_interval,
                                   std::vector<RoutedNet>& nets,
                                   WorkspaceFactory workspace_factory,
                                   std::vector<std::vector<DeltaSteppingCsrTelemetry>>*
                                       delta_telemetry_records = nullptr,
                                   UnitBfsPathDiagnostic*
                                       unit_bfs_diagnostic = nullptr) {
  if (delta_telemetry_records != nullptr &&
      delta_telemetry_records->size() != route_request_count) {
    throw std::invalid_argument(
        "Delta telemetry record count must match route request count");
  }
  std::size_t worker_count =
      std::min<std::size_t>(options.parallel_net_workers,
                            std::max<std::size_t>(1, route_request_count));
  if (stream != nullptr) {
    worker_count = 1;
  }

  if (worker_count <= 1 || route_request_count <= 1) {
    auto sssp_workspace = workspace_factory(stream);
    std::vector<std::uint32_t> route_tree_seen(static_cast<std::size_t>(base_graph.rows), 0);
    std::vector<int> route_parent_by_child(static_cast<std::size_t>(base_graph.rows), -1);
    std::vector<std::uint32_t> route_parent_seen(static_cast<std::size_t>(base_graph.rows), 0);
    std::uint32_t route_tree_stamp = 0;

    for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
      const RouteRequest& request = metadata.route_requests[net_index];
      const std::uint32_t tree_stamp =
          next_tree_stamp(route_tree_seen, &route_tree_stamp);
      try {
        nets[net_index] =
            route_net(base_graph,
                      sssp_workspace,
                      request,
                      route_tree_seen,
                      route_parent_by_child,
                      route_parent_seen,
                      tree_stamp,
                      options,
                      stream,
                      delta_telemetry_records == nullptr
                          ? nullptr
                          : &(*delta_telemetry_records)[net_index],
                      unit_bfs_diagnostic != nullptr &&
                              unit_bfs_diagnostic->net_index == net_index
                          ? unit_bfs_diagnostic
                          : nullptr);
      } catch (const std::exception& error) {
        throw std::runtime_error(
            "route request " + std::to_string(net_index) + " failed: " +
            error.what());
      }

      if ((net_index + 1) == route_request_count ||
          (net_index + 1) % progress_interval == 0) {
        print_pathfinder_progress(1, 1, net_index + 1, route_request_count);
      }
    }
    return;
  }

  std::atomic<std::size_t> next_net{0};
  std::atomic<std::size_t> completed_nets{0};
  std::atomic<bool> failed{false};
  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::mutex progress_mutex;
  std::size_t last_reported = 0;
  const int worker_device = current_worker_device();

  auto report_progress = [&](std::size_t completed) {
    std::lock_guard<std::mutex> lock(progress_mutex);
    // Completion counters are assigned atomically, but workers can reach this
    // mutex out of order. Never move last_reported backwards (or underflow the
    // unsigned subtraction below).
    if (completed <= last_reported) {
      return;
    }
    if (completed == route_request_count ||
        completed - last_reported >= progress_interval) {
      last_reported = completed;
      print_pathfinder_progress(1, 1, completed, route_request_count);
    }
  };

  auto worker = [&]() {
    try {
      select_worker_device(worker_device);
      WorkerStream worker_stream(stream == nullptr);
      hipStream_t local_stream = worker_stream.get(stream);
      auto sssp_workspace = workspace_factory(local_stream);
      std::vector<std::uint32_t> route_tree_seen(
          static_cast<std::size_t>(base_graph.rows), 0);
      std::vector<int> route_parent_by_child(
          static_cast<std::size_t>(base_graph.rows), -1);
      std::vector<std::uint32_t> route_parent_seen(
          static_cast<std::size_t>(base_graph.rows), 0);
      std::uint32_t route_tree_stamp = 0;

      while (!failed.load(std::memory_order_relaxed)) {
        const std::size_t net_index =
            next_net.fetch_add(1, std::memory_order_relaxed);
        if (net_index >= route_request_count) {
          break;
        }

        const RouteRequest& request = metadata.route_requests[net_index];
        const std::uint32_t tree_stamp =
            next_tree_stamp(route_tree_seen, &route_tree_stamp);
        try {
          nets[net_index] =
              route_net(base_graph,
                        sssp_workspace,
                        request,
                        route_tree_seen,
                        route_parent_by_child,
                        route_parent_seen,
                        tree_stamp,
                        options,
                        local_stream,
                        delta_telemetry_records == nullptr
                            ? nullptr
                            : &(*delta_telemetry_records)[net_index],
                        unit_bfs_diagnostic != nullptr &&
                                unit_bfs_diagnostic->net_index == net_index
                            ? unit_bfs_diagnostic
                            : nullptr);
        } catch (const std::exception& error) {
          throw std::runtime_error(
              "route request " + std::to_string(net_index) + " failed: " +
              error.what());
        }

        const std::size_t completed =
            completed_nets.fetch_add(1, std::memory_order_relaxed) + 1;
        report_progress(completed);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  auto join_workers = [&workers]() {
    for (std::thread& thread : workers) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  };
  try {
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers.emplace_back(worker);
    }
  } catch (...) {
    // If std::thread construction fails after one or more workers have
    // started, those joinable threads must be stopped and joined before the
    // vector is destroyed. Otherwise std::thread::~thread calls terminate()
    // instead of allowing the construction error to reach the caller.
    failed.store(true, std::memory_order_relaxed);
    join_workers();
    throw;
  }
  join_workers();
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

struct DeltaTelemetryTotals {
  std::uint64_t queries = 0;
  std::uint64_t completed_queries = 0;
  std::array<std::uint64_t, 4> path_counts{};
  DeltaSteppingCsrTelemetry sums;
  std::uint64_t current_queue_high_water = 0;
  std::uint64_t pending_queue_high_water = 0;
  std::uint64_t heavy_queue_high_water = 0;
};

DeltaTelemetryTotals aggregate_delta_telemetry(
    const std::vector<DeltaSteppingCsrTelemetry>& records) {
  DeltaTelemetryTotals totals;
  for (const DeltaSteppingCsrTelemetry& record : records) {
    if (!record.collected) continue;
    ++totals.queries;
    if (record.completed) ++totals.completed_queries;
    switch (record.execution_path) {
      case DeltaSteppingCsrExecutionPath::kExactUnit:
        ++totals.path_counts[0];
        break;
      case DeltaSteppingCsrExecutionPath::kCompactGeneric:
        ++totals.path_counts[1];
        break;
      case DeltaSteppingCsrExecutionPath::kLegacyGeneric:
        ++totals.path_counts[2];
        break;
      case DeltaSteppingCsrExecutionPath::kGenericDistancesOnly:
        ++totals.path_counts[3];
        break;
      case DeltaSteppingCsrExecutionPath::kNotRun:
        break;
    }
    totals.sums.outer_buckets_processed +=
        record.outer_buckets_processed;
    totals.sums.light_relaxation_rounds +=
        record.light_relaxation_rounds;
    totals.sums.heavy_edge_phases += record.heavy_edge_phases;
    totals.sums.frontier_entries_processed +=
        record.frontier_entries_processed;
    totals.sums.active_vertices_processed +=
        record.active_vertices_processed;
    totals.sums.stale_frontier_entries +=
        record.stale_frontier_entries;
    totals.sums.light_edge_visits += record.light_edge_visits;
    totals.sums.heavy_edge_visits += record.heavy_edge_visits;
    totals.sums.distance_atomic_attempts +=
        record.distance_atomic_attempts;
    totals.sums.successful_distance_relaxations +=
        record.successful_distance_relaxations;
    totals.sums.distance_cas_retries += record.distance_cas_retries;
    totals.sums.current_queue_insertions +=
        record.current_queue_insertions;
    totals.sums.pending_queue_insertions +=
        record.pending_queue_insertions;
    totals.sums.heavy_queue_insertions +=
        record.heavy_queue_insertions;
    totals.sums.bucket_insertions += record.bucket_insertions;
    totals.sums.pending_entry_examinations +=
        record.pending_entry_examinations;
    totals.sums.stale_pending_entry_examinations +=
        record.stale_pending_entry_examinations;
    totals.sums.reached_vertices += record.reached_vertices;
    totals.sums.controller_round_trips +=
        record.controller_round_trips;
    totals.sums.compact_parent_fallback_events +=
        record.compact_parent_fallback_events;
    totals.current_queue_high_water =
        std::max(totals.current_queue_high_water,
                 record.current_queue_high_water);
    totals.pending_queue_high_water =
        std::max(totals.pending_queue_high_water,
                 record.pending_queue_high_water);
    totals.heavy_queue_high_water =
        std::max(totals.heavy_queue_high_water,
                 record.heavy_queue_high_water);
  }
  return totals;
}

std::string delta_telemetry_aggregate_json(
    const std::vector<DeltaSteppingCsrTelemetry>& records,
    const PathfinderOptions& options,
    float resolved_delta,
    int wavefront_size,
    std::size_t worker_count) {
  const DeltaTelemetryTotals totals = aggregate_delta_telemetry(records);
  const DeltaSteppingCsrTelemetry& sums = totals.sums;
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  out << "{\"type\":\"delta_stepping_telemetry\""
      << ",\"schema_version\":1"
      << ",\"scope\":\"pathfinder_run\""
      << ",\"queries\":" << totals.queries
      << ",\"completed_queries\":" << totals.completed_queries
      << ",\"resolved_delta\":" << resolved_delta
      << ",\"wavefront_size\":" << wavefront_size
      << ",\"parallel_workers\":" << worker_count
      << ",\"delta_auto\":" << (options.delta_auto ? "true" : "false")
      << ",\"delta_multiplier\":" << options.delta_multiplier
      << ",\"force_generic\":"
      << (options.delta_force_generic ? "true" : "false")
      << ",\"force_legacy_parent\":"
      << (options.delta_force_legacy_parent ? "true" : "false")
      << ",\"execution_paths\":{"
      << "\"exact_unit\":" << totals.path_counts[0]
      << ",\"compact_generic\":" << totals.path_counts[1]
      << ",\"legacy_generic\":" << totals.path_counts[2]
      << ",\"generic_distances_only\":" << totals.path_counts[3]
      << "},\"counters\":{"
      << "\"outer_buckets_processed\":" << sums.outer_buckets_processed
      << ",\"light_relaxation_rounds\":" << sums.light_relaxation_rounds
      << ",\"heavy_edge_phases\":" << sums.heavy_edge_phases
      << ",\"frontier_entries_processed\":"
      << sums.frontier_entries_processed
      << ",\"active_vertices_processed\":"
      << sums.active_vertices_processed
      << ",\"stale_frontier_entries\":" << sums.stale_frontier_entries
      << ",\"light_edge_visits\":" << sums.light_edge_visits
      << ",\"heavy_edge_visits\":" << sums.heavy_edge_visits
      << ",\"distance_atomic_attempts\":"
      << sums.distance_atomic_attempts
      << ",\"successful_distance_relaxations\":"
      << sums.successful_distance_relaxations
      << ",\"distance_cas_retries\":" << sums.distance_cas_retries
      << ",\"current_queue_insertions\":"
      << sums.current_queue_insertions
      << ",\"pending_queue_insertions\":"
      << sums.pending_queue_insertions
      << ",\"heavy_queue_insertions\":"
      << sums.heavy_queue_insertions
      << ",\"bucket_insertions\":" << sums.bucket_insertions
      << ",\"pending_entry_examinations\":"
      << sums.pending_entry_examinations
      << ",\"stale_pending_entry_examinations\":"
      << sums.stale_pending_entry_examinations
      << ",\"reached_vertices\":" << sums.reached_vertices
      << ",\"controller_round_trips\":"
      << sums.controller_round_trips
      << ",\"compact_parent_fallback_events\":"
      << sums.compact_parent_fallback_events
      << "},\"maxima\":{"
      << "\"current_queue_high_water\":"
      << totals.current_queue_high_water
      << ",\"pending_queue_high_water\":"
      << totals.pending_queue_high_water
      << ",\"heavy_queue_high_water\":"
      << totals.heavy_queue_high_water << "}}";
  return out.str();
}

}  // namespace

std::filesystem::path default_metadata_path(const std::filesystem::path& csr_path) {
  std::filesystem::path path = csr_path;
  path += ".ifmeta.bin";
  return path;
}

int parse_int_arg(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

std::size_t parse_size_arg(const char* text, const char* name) {
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

float parse_float_arg(const char* text, const char* name) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

std::uint64_t parse_u64_arg(const char* text, const char* name) {
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::uint64_t>(value);
}

enum class DeltaBenchmarkWeights {
  kOriginal,
  kUnit,
  kAllLight,
  kAllHeavy,
  kMixed,
};

DeltaBenchmarkWeights parse_delta_benchmark_weights_arg(const char* text) {
  const std::string value(text);
  if (value == "unit") return DeltaBenchmarkWeights::kUnit;
  if (value == "all-light") return DeltaBenchmarkWeights::kAllLight;
  if (value == "all-heavy") return DeltaBenchmarkWeights::kAllHeavy;
  if (value == "mixed") return DeltaBenchmarkWeights::kMixed;
  throw std::runtime_error(
      "invalid delta-benchmark-weights: " + value +
      " (expected unit, all-light, all-heavy, or mixed)");
}

const char* delta_benchmark_weights_name(DeltaBenchmarkWeights mode) {
  switch (mode) {
    case DeltaBenchmarkWeights::kOriginal:
      return "original";
    case DeltaBenchmarkWeights::kUnit:
      return "unit";
    case DeltaBenchmarkWeights::kAllLight:
      return "all-light";
    case DeltaBenchmarkWeights::kAllHeavy:
      return "all-heavy";
    case DeltaBenchmarkWeights::kMixed:
      return "mixed";
  }
  return "unknown";
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

void apply_delta_benchmark_weights(HostCsrF32& graph,
                                   DeltaBenchmarkWeights mode,
                                   float delta,
                                   std::uint64_t seed) {
  if (mode == DeltaBenchmarkWeights::kOriginal) return;
  if (!(delta > 0.0f) || !std::isfinite(delta)) {
    throw std::invalid_argument(
        "delta benchmark weights require a finite positive numeric delta");
  }
  const float light = delta * 0.25f;
  const float heavy = delta * 4.0f;
  if (!std::isfinite(light) || !std::isfinite(heavy)) {
    throw std::invalid_argument(
        "numeric delta is too large for the benchmark weight family");
  }
  for (std::size_t edge = 0; edge < graph.values.size(); ++edge) {
    switch (mode) {
      case DeltaBenchmarkWeights::kOriginal:
        break;
      case DeltaBenchmarkWeights::kUnit:
        graph.values[edge] = 1.0f;
        break;
      case DeltaBenchmarkWeights::kAllLight:
        graph.values[edge] = light;
        break;
      case DeltaBenchmarkWeights::kAllHeavy:
        graph.values[edge] = heavy;
        break;
      case DeltaBenchmarkWeights::kMixed: {
        constexpr float kFactors[] = {0.0f, 0.25f, 1.0f, 4.0f};
        const std::uint64_t hash =
            splitmix64(static_cast<std::uint64_t>(edge) ^ seed);
        graph.values[edge] =
            delta * kFactors[static_cast<std::size_t>(hash & 3U)];
        break;
      }
    }
  }
}

void parse_delta_arg(const char* text, PathfinderOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("delta option destination must not be null");
  }
  if (std::string(text) == "auto") {
    options->delta_auto = true;
    options->delta_controls_explicit = true;
    return;
  }
  options->delta = parse_float_arg(text, "delta");
  options->delta_auto = false;
  options->delta_controls_explicit = true;
}

SsspEngine parse_sssp_engine_arg(const char* text) {
  const std::string value(text);
  if (value == "unit-bfs" || value == "bfs") {
    return SsspEngine::kUnitBfs;
  }
  if (value == "delta-step" || value == "delta-stepping" || value == "delta") {
    return SsspEngine::kDeltaStep;
  }
  if (value == "bellman-ford" || value == "bellman_ford" || value == "bf8" ||
      value == "bf9" || value == "bf10") {
    return SsspEngine::kBellmanFord;
  }
  throw std::runtime_error("invalid sssp-engine: " + value);
}

const char* sssp_engine_name(SsspEngine engine) {
  switch (engine) {
    case SsspEngine::kUnitBfs:
      return "unit-bfs";
    case SsspEngine::kDeltaStep:
      return "delta-step";
    case SsspEngine::kBellmanFord:
      return "bellman-ford";
  }
  return "unknown";
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <graph.csrbin> [metadata.ifmeta.bin] [options]\n\n"
      << "Options:\n"
      << "  --sssp-engine <unit-bfs|delta-step|bellman-ford|bf10>\n"
      << "                                  Shortest-path backend. bellman-ford and bf10 select BF10;\n"
      << "                                  bf8 and bf9 are compatibility aliases. Default: unit-bfs\n"
      << "  --use-delta-step                Use delta-step backend for comparison.\n"
      << "  --delta <float|auto>            Delta-stepping bucket width. Default: 1\n"
      << "  --delta-multiplier <float>      Positive sweep multiplier for --delta auto. Default: 1\n"
      << "  --max-sssp-iters <int>          Delta rounds, BFS depth, or Bellman-Ford rounds; -1 for default.\n"
      << "  --delta-force-generic           Bypass exact-unit specialization; retain weights and delta.\n"
      << "  --delta-force-legacy-parent     Force generic Delta predecessor recovery for A/B comparison.\n"
      << "  --delta-telemetry               Emit one aggregate Delta-Stepping telemetry JSON record.\n"
      << "  --delta-benchmark-weights <unit|all-light|all-heavy|mixed>\n"
      << "                                  Replace CSR weights deterministically for numeric-delta benchmarks.\n"
      << "  --delta-benchmark-weight-seed <uint>\n"
      << "                                  Seed for the mixed benchmark family. Default: 0\n"
      << "  --capacity <int>                Capacity used only for overuse diagnostics. Default: 1\n"
      << "  --net-limit <count>             Route only the first count requests.\n"
      << "  --parallel-net-workers <count>  Independent net workers. Default: 0 (engine-dependent auto).\n"
      << "  --diagnose-net <zero-based>     Replay through one request and emit UnitBFS path diagnostics.\n"
      << "  --diagnose-sink <zero-based>    Sink within --diagnose-net; both diagnostic options are required.\n"
      << "  --allow-unrouted                Write partial routes even if some sinks are unreached.\n"
      << "  --routes-out <path>             Write routed PIP tree data as JSONL.\n"
      << "\nCompatibility-only options accepted and ignored by this one-shot router:\n"
      << "  --max-pathfinder-iters <int>\n"
      << "  --present-factor <float>\n"
      << "  --present-multiplier <float>\n"
      << "  --history-factor <float>\n"
      << "  --route-batch-size <count>\n";
}

HostCsrF32 load_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(in, "CSR format version");
  const std::uint64_t orientation = read_u64(in, "CSR orientation");
  if (version != EXPECTED_CSR_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "CSR row count");
  const std::uint64_t cols = read_u64(in, "CSR column count");
  (void)read_u64(in, "declared edge count");
  (void)read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(in, "CSR colind count");
  const std::uint64_t values_count = read_u64(in, "CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<minplus_sparse::Offset>(rows);
  graph.cols = static_cast<minplus_sparse::Offset>(cols);
  graph.nnz = static_cast<minplus_sparse::Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(in, graph.colind, colind_count, "CSR colind");
  read_array(in, graph.values, values_count, "CSR values");
  validate_csr(graph);
  return graph;
}

RoutingMetadata load_interchange_metadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
  const std::uint64_t pip_data_count = read_u64(in, "metadata pip data count");
  const std::uint64_t site_pin_attr_count = read_u64(in, "metadata site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "metadata route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "metadata sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "metadata logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "metadata logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "metadata physical byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "metadata logical byte count");

  RoutingMetadata metadata;
  metadata.device_path_string = read_u64(in, "metadata device path string");
  metadata.physical_path_string = read_u64(in, "metadata physical path string");
  metadata.logical_path_string = read_u64(in, "metadata logical path string");
  metadata.logical_design_name_string = read_u64(in, "metadata logical design name");

  metadata.strings.reserve(static_cast<std::size_t>(string_count));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  read_array(in, metadata.node_device_ids, node_count, "metadata device node ids");
  read_array(in,
             metadata.node_min_x,
             node_count,
             "metadata node min x coordinates");
  read_array(in,
             metadata.node_max_x,
             node_count,
             "metadata node max x coordinates");
  read_array(in,
             metadata.node_min_y,
             node_count,
             "metadata node min y coordinates");
  read_array(in,
             metadata.node_max_y,
             node_count,
             "metadata node max y coordinates");
  read_array(in,
             metadata.node_tile_type_strings,
             node_count,
             "metadata node tile type strings");
  read_array(in,
             metadata.node_wire_type_strings,
             node_count,
             "metadata node wire type strings");

  metadata.edge_attrs.resize(static_cast<std::size_t>(edge_attr_count));
  for (EdgeAttr& attr : metadata.edge_attrs) {
    attr.tile_string = read_u64(in, "metadata edge tile string");
    attr.pip_data_index = read_u64(in, "metadata edge pip data index");
  }

  metadata.pip_data.resize(static_cast<std::size_t>(pip_data_count));
  for (PipData& pip : metadata.pip_data) {
    pip.wire0_string = read_u64(in, "metadata pip wire0 string");
    pip.wire1_string = read_u64(in, "metadata pip wire1 string");
    pip.forward = read_u64(in, "metadata pip forward flag") != 0;
  }

  metadata.site_pin_attrs.resize(static_cast<std::size_t>(site_pin_attr_count));
  for (SitePinNode& attr : metadata.site_pin_attrs) {
    attr.node = read_route_node(in, "metadata site pin node");
    attr.site_string = read_u64(in, "metadata site pin site");
    attr.pin_string = read_u64(in, "metadata site pin pin");
  }

  metadata.route_requests.resize(static_cast<std::size_t>(route_request_count));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");

    const std::uint64_t source_count = read_u64(in, "metadata source count");
    request.sources.resize(static_cast<std::size_t>(source_count));
    for (SitePinNode& source : request.sources) {
      source.node = read_route_node(in, "metadata source node");
      source.site_string = read_u64(in, "metadata source site");
      source.pin_string = read_u64(in, "metadata source pin");
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    request.sinks.resize(static_cast<std::size_t>(sink_count));
    for (SitePinNode& sink : request.sinks) {
      sink.node = read_route_node(in, "metadata sink node");
      sink.site_string = read_u64(in, "metadata sink site");
      sink.pin_string = read_u64(in, "metadata sink pin");
    }
  }

  for (std::uint64_t i = 0; i < logical_cell_count; ++i) {
    (void)read_u64(in, "metadata logical cell declaration");
    (void)read_u64(in, "metadata logical cell net begin");
    (void)read_u64(in, "metadata logical cell net count");
  }
  for (std::uint64_t i = 0; i < logical_net_count; ++i) {
    (void)read_u64(in, "metadata logical net name");
    (void)read_u64(in, "metadata logical net cell");
    (void)read_u64(in, "metadata logical net port begin");
    (void)read_u64(in, "metadata logical net port count");
  }
  for (std::uint64_t i = 0; i < logical_port_instance_count; ++i) {
    (void)read_u64(in, "metadata logical port name");
    (void)read_u64(in, "metadata logical instance name");
    (void)read_u64(in, "metadata logical port index");
    (void)read_u64(in, "metadata logical instance index");
    (void)read_u64(in, "metadata logical bus index");
    (void)read_u64(in, "metadata logical has bus");
    (void)read_u64(in, "metadata logical external port");
  }

  read_array(in, metadata.blocked_nodes, blocked_node_count, "metadata blocked nodes");
  read_array(in, metadata.sink_stop_nodes, sink_stop_node_count, "metadata sink stop nodes");

  std::vector<std::uint8_t> skipped_bytes;
  read_array(in, skipped_bytes, physical_netlist_byte_count, "metadata physical bytes");
  read_array(in, skipped_bytes, logical_netlist_byte_count, "metadata logical bytes");

  return metadata;
}

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dist,
                                                int source,
                                                int target) {
  validate_csr(graph);
  if (!valid_node(source, graph.rows) || !valid_node(target, graph.rows)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance vector size does not match CSR rows");
  }
  if (source == target) {
    return {};
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<int> parent(static_cast<std::size_t>(graph.rows), -1);
  std::vector<minplus_sparse::Offset> parent_edge(
      static_cast<std::size_t>(graph.rows), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.rows));
  parent[static_cast<std::size_t>(source)] = source;
  queue.push_back(source);

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    if (u == target) {
      break;
    }
    const float du = dist[static_cast<std::size_t>(u)];
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t v_index = static_cast<std::size_t>(v);
      if (parent[v_index] >= 0 || v == u ||
          !tight_edge(du, graph.values[static_cast<std::size_t>(edge)], dist[v_index])) {
        continue;
      }
      parent[v_index] = u;
      parent_edge[v_index] = edge;
      queue.push_back(v);
      if (v == target) {
        head = queue.size();
        break;
      }
    }
  }

  if (parent[static_cast<std::size_t>(target)] < 0) {
    throw std::runtime_error("shortest path reconstruction did not reach target");
  }

  std::vector<PathEdge> reversed;
  for (int current = target;
       parent[static_cast<std::size_t>(current)] != current;) {
    const int pred = parent[static_cast<std::size_t>(current)];
    const minplus_sparse::Offset edge =
        parent_edge[static_cast<std::size_t>(current)];
    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream,
                                UnitBfsPathDiagnostic* unit_bfs_diagnostic) {
  PATHFINDER_PROFILE_RANGE("pathfinder.run");
  validate_options(options);
  int automatic_delta_wavefront_size = 0;
  float resolved_automatic_delta = options.delta;
  if (options.sssp_engine == SsspEngine::kDeltaStep &&
      (options.delta_auto || options.delta_telemetry)) {
    automatic_delta_wavefront_size = current_device_wavefront_size();
  }
  if (options.sssp_engine == SsspEngine::kDeltaStep &&
      options.delta_auto) {
    PATHFINDER_PROFILE_RANGE("pathfinder.delta_auto_stats");
    // The resolver performs the same complete CSR validation while it gathers
    // the weight statistics. Avoid a second O(V + E) validation pass on large
    // device graphs.
    resolved_automatic_delta = delta_stepping_auto_delta(
        base_graph,
        automatic_delta_wavefront_size,
        options.delta_multiplier);
  } else {
    validate_csr(base_graph);
  }
  if (metadata.node_device_ids.size() != static_cast<std::size_t>(base_graph.rows)) {
    throw std::runtime_error("metadata node count does not match CSR row count");
  }
  if (metadata.node_min_x.size() != metadata.node_device_ids.size() ||
      metadata.node_max_x.size() != metadata.node_device_ids.size() ||
      metadata.node_min_y.size() != metadata.node_device_ids.size() ||
      metadata.node_max_y.size() != metadata.node_device_ids.size() ||
      metadata.node_tile_type_strings.size() != metadata.node_device_ids.size() ||
      metadata.node_wire_type_strings.size() != metadata.node_device_ids.size()) {
    throw std::runtime_error(
        "metadata node coordinate range/tile/wire type arrays do not match node count");
  }
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(base_graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }

  PathfinderResult result;
  result.occupancy.assign(static_cast<std::size_t>(base_graph.rows), 0);

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());
  if (unit_bfs_diagnostic != nullptr) {
    if (options.sssp_engine != SsspEngine::kUnitBfs) {
      throw std::invalid_argument(
          "UnitBFS path diagnostics require the unit-bfs engine");
    }
    const std::size_t diagnostic_net = unit_bfs_diagnostic->net_index;
    const std::size_t diagnostic_sink = unit_bfs_diagnostic->sink_index;
    if (diagnostic_net >= metadata.route_requests.size()) {
      throw std::out_of_range("diagnostic net index is outside route requests");
    }
    if (diagnostic_net >= route_request_count) {
      throw std::invalid_argument(
          "diagnostic net is excluded by the configured net limit");
    }
    if (diagnostic_sink >=
        metadata.route_requests[diagnostic_net].sinks.size()) {
      throw std::out_of_range("diagnostic sink index is outside the request");
    }
    *unit_bfs_diagnostic = UnitBfsPathDiagnostic{};
    unit_bfs_diagnostic->net_index = diagnostic_net;
    unit_bfs_diagnostic->sink_index = diagnostic_sink;
    unit_bfs_diagnostic->all_unit_weights =
        std::all_of(base_graph.values.begin(),
                    base_graph.values.end(),
                    [](float value) { return value == 1.0f; });
    if (!unit_bfs_diagnostic->all_unit_weights) {
      throw std::invalid_argument(
          "UnitBFS path diagnostics require exact unit edge weights");
    }
  }
  result.nets.resize(route_request_count);

  const std::size_t progress_interval =
      std::max<std::size_t>(1, route_request_count / 100);

  switch (options.sssp_engine) {
    case SsspEngine::kUnitBfs: {
      auto shared_graph = std::make_shared<UnitBfsCsrGraph>(base_graph, stream);
      PathfinderOptions unit_options = options;
      if (unit_options.parallel_net_workers == 0) {
        unit_options.parallel_net_workers = recommend_unit_bfs_worker_count(
            base_graph.rows, route_request_count, stream);
        std::cout << "[pathfinder] auto-selected "
                  << unit_options.parallel_net_workers
                  << " unit-BFS worker(s)\n";
      }
      if (unit_bfs_diagnostic != nullptr) {
        unit_bfs_diagnostic->worker_count =
            stream != nullptr
                ? 1
                : std::min<std::size_t>(
                      unit_options.parallel_net_workers,
                      std::max<std::size_t>(1, route_request_count));
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          unit_options,
          stream,
          route_request_count,
          progress_interval,
          result.nets,
          [shared_graph](hipStream_t worker_stream) {
            return UnitBfsCsrWorkspace(shared_graph, worker_stream);
          },
          nullptr,
          unit_bfs_diagnostic);
      break;
    }
    case SsspEngine::kDeltaStep: {
      PathfinderOptions delta_options = options;
      if (delta_options.delta_auto) {
        delta_options.delta = resolved_automatic_delta;
        std::ostringstream message;
        message.precision(std::numeric_limits<float>::max_digits10);
        message << "[pathfinder] resolved automatic delta="
                << delta_options.delta
                << " (wavefront=" << automatic_delta_wavefront_size
                << ", multiplier=" << delta_options.delta_multiplier << ")\n";
        std::cout << message.str();
      }
      auto shared_graph =
          std::make_shared<DeltaSteppingCsrGraph>(base_graph, stream);
      if (delta_options.parallel_net_workers == 0) {
        const bool uses_unit_specialization =
            !delta_options.delta_force_generic &&
            !delta_options.delta_force_legacy_parent &&
            delta_options.max_sssp_iterations < 0 &&
            base_graph.rows <=
                (static_cast<minplus_sparse::Offset>(1) <<
                 std::numeric_limits<float>::digits) &&
            std::all_of(base_graph.values.begin(),
                        base_graph.values.end(),
                        [](float value) { return value == 1.0f; });
        delta_options.parallel_net_workers = recommend_delta_worker_count(
            base_graph.rows, route_request_count, uses_unit_specialization,
            stream);
        std::cout << "[pathfinder] auto-selected "
                  << delta_options.parallel_net_workers
                  << " delta-step worker(s)\n";
      }
      const DeltaSteppingCsrWorkspaceOptions workspace_options{
          delta_options.delta_force_legacy_parent
              ? DeltaSteppingCsrParentMode::kForceLegacy
              : DeltaSteppingCsrParentMode::kAutomatic,
          delta_options.delta_force_generic
              ? DeltaSteppingCsrExecutionMode::kForceGeneric
              : DeltaSteppingCsrExecutionMode::kAutomatic};
      if (workspace_options.execution_mode ==
          DeltaSteppingCsrExecutionMode::kForceGeneric) {
        std::cout << "[pathfinder] selected forced generic delta execution\n";
      }
      if (workspace_options.parent_mode ==
          DeltaSteppingCsrParentMode::kForceLegacy) {
        std::cout << "[pathfinder] selected forced legacy parent mode for "
                     "generic vector-target delta runs\n";
      }
      std::vector<std::vector<DeltaSteppingCsrTelemetry>>
          delta_telemetry_records;
      if (delta_options.delta_telemetry) {
        delta_telemetry_records.resize(route_request_count);
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          delta_options,
          stream,
          route_request_count,
          progress_interval,
          result.nets,
          [shared_graph, workspace_options](hipStream_t worker_stream) {
            return DeltaSteppingCsrWorkspace(shared_graph, worker_stream,
                                             workspace_options);
          },
          delta_options.delta_telemetry ? &delta_telemetry_records : nullptr);
      if (delta_options.delta_telemetry) {
        std::vector<DeltaSteppingCsrTelemetry> flattened_telemetry;
        std::size_t telemetry_query_count = 0;
        for (const auto& net_records : delta_telemetry_records) {
          telemetry_query_count += net_records.size();
        }
        flattened_telemetry.reserve(telemetry_query_count);
        for (const auto& net_records : delta_telemetry_records) {
          flattened_telemetry.insert(flattened_telemetry.end(),
                                     net_records.begin(),
                                     net_records.end());
        }
        const std::size_t actual_worker_count =
            stream != nullptr
                ? 1
                : std::min<std::size_t>(
                      delta_options.parallel_net_workers,
                      std::max<std::size_t>(1, route_request_count));
        std::cout << delta_telemetry_aggregate_json(
                         flattened_telemetry,
                         delta_options,
                         delta_options.delta,
                         automatic_delta_wavefront_size,
                         actual_worker_count)
                  << '\n';
      }
      break;
    }
    case SsspEngine::kBellmanFord: {
      std::cout << "[pathfinder] selected Bellman-Ford bf10 backend\n";
      auto shared_graph =
          std::make_shared<BellmanFord10CsrGraph>(base_graph, stream);
      PathfinderOptions bellman_ford_options = options;
      if (bellman_ford_options.parallel_net_workers == 0) {
        // Start conservatively: every worker owns a full frontier state and
        // persistent reconstruction scratch buffers.
        bellman_ford_options.parallel_net_workers = 1;
        std::cout << "[pathfinder] auto-selected 1 Bellman-Ford worker(s)\n";
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          bellman_ford_options,
          stream,
          route_request_count,
          progress_interval,
          result.nets,
          [shared_graph](hipStream_t worker_stream) {
            return BellmanFord10CsrWorkspace(shared_graph, worker_stream);
          });
      break;
    }
  }

  for (const RoutedNet& net : result.nets) {
    commit_net_occupancy(net, result.occupancy);
  }

  bool all_sinks_reached = true;
  for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
    if (!result.nets[net_index].reached_all_sinks) {
      all_sinks_reached = false;
      break;
    }
  }

  result.iterations_used = 1;
  result.all_sinks_reached = all_sinks_reached;
  update_congestion_stats(result.occupancy,
                          options.capacity,
                          &result.overused_nodes,
                          &result.max_occupancy);
  result.routed = all_sinks_reached;
  return result;
}

std::string unit_bfs_path_diagnostic_json(
    const UnitBfsPathDiagnostic& diagnostic) {
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  auto write_distance = [&out](float distance) {
    if (std::isfinite(distance)) {
      out << distance;
    } else {
      out << "null";
    }
  };
  auto write_observation = [&out, &write_distance](
                               const UnitBfsPathObservation& observation) {
    out << "{\"captured\":" << (observation.captured ? "true" : "false")
        << ",\"reached\":" << (observation.reached ? "true" : "false")
        << ",\"distance\":";
    write_distance(observation.distance);
    out << ",\"source\":" << observation.source
        << ",\"edge_count\":" << observation.edge_count
        << ",\"path_hash\":" << observation.path_hash << '}';
  };

  out << "{\"type\":\"unit_bfs_path_diagnostic\""
      << ",\"schema_version\":1"
      << ",\"net_index\":" << diagnostic.net_index
      << ",\"sink_index\":" << diagnostic.sink_index
      << ",\"target\":" << diagnostic.target
      << ",\"worker_count\":" << diagnostic.worker_count
      << ",\"all_unit_weights\":"
      << (diagnostic.all_unit_weights ? "true" : "false")
      << ",\"original_source_count\":"
      << diagnostic.original_source_count
      << ",\"tree_source_count\":" << diagnostic.tree_source_count
      << ",\"prior_sinks_reached\":" << diagnostic.prior_sinks_reached
      << ",\"cpu_original_distance\":";
  if (diagnostic.cpu_original_distance >= 0) {
    out << diagnostic.cpu_original_distance;
  } else {
    out << "null";
  }
  out << ",\"cpu_expanded_tree_distance\":";
  if (diagnostic.cpu_expanded_tree_distance >= 0) {
    out << diagnostic.cpu_expanded_tree_distance;
  } else {
    out << "null";
  }
  out << ",\"raw_batched\":";
  write_observation(diagnostic.raw_batched);
  out << ",\"fresh_original\":";
  write_observation(diagnostic.fresh_original);
  out << ",\"fresh_expanded_tree\":";
  write_observation(diagnostic.fresh_expanded_tree);
  out << ",\"attached\":{\"reached\":"
      << (diagnostic.attached_reached ? "true" : "false")
      << ",\"distance\":";
  write_distance(diagnostic.attached_distance);
  out << ",\"source\":" << diagnostic.attached_source
      << ",\"edge_count\":" << diagnostic.attached_edge_count
      << ",\"path_hash\":" << diagnostic.attached_path_hash << '}'
      << ",\"classification\":";
  write_json_string(out, diagnostic.classification);
  out << '}';
  return out.str();
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) {
    return {};
  }
  if (index >= metadata.strings.size()) {
    std::ostringstream out;
    out << "<bad-string-" << index << ">";
    return out.str();
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result) {
  validate_csr(graph);
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }
  if (result.nets.size() > metadata.route_requests.size()) {
    throw std::runtime_error("pathfinder result has more nets than metadata requests");
  }
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("could not open routes output file: " + path.string());
  }

  for (std::size_t net_index = 0; net_index < result.nets.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const RoutedNet& net = result.nets[net_index];

    out << "{\"net\":";
    write_json_string(out, string_at(metadata, request.net_string));
    out << ",\"routed\":" << (net.reached_all_sinks ? "true" : "false");

    out << ",\"sources\":[";
    for (std::size_t i = 0; i < request.sources.size(); ++i) {
      const SitePinNode& source = request.sources[i];
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << source.node << ",\"site\":";
      write_json_string(out, string_at(metadata, source.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, source.pin_string));
      out << '}';
    }
    out << ']';

    out << ",\"sinks\":[";
    for (std::size_t i = 0; i < request.sinks.size(); ++i) {
      const SitePinNode& sink_pin = request.sinks[i];
      const bool has_sink_result = i < net.sinks.size();
      const RoutedSink* sink = has_sink_result ? &net.sinks[i] : nullptr;
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << sink_pin.node << ",\"site\":";
      write_json_string(out, string_at(metadata, sink_pin.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, sink_pin.pin_string));
      out << ",\"reached\":" << (sink != nullptr && sink->reached ? "true" : "false");
      out << ",\"source\":" << (sink != nullptr ? sink->source : -1);
      out << '}';
    }
    out << ']';

    std::size_t route_edge_count = 0;
    for (const RoutedSink& sink : net.sinks) {
      route_edge_count += sink.edges.size();
    }
    std::unordered_set<std::uint64_t> seen_edges;
    seen_edges.reserve(route_edge_count);
    out << ",\"edges\":[";
    bool first_edge = true;
    for (const RoutedSink& sink : net.sinks) {
      if (!sink.reached) {
        continue;
      }
      for (const PathEdge& path_edge : sink.edges) {
        if (path_edge.csr_edge < 0 ||
            path_edge.csr_edge >= graph.nnz ||
            !valid_node(path_edge.from, graph.rows) ||
            !valid_node(path_edge.to, graph.rows) ||
            path_edge.csr_edge < graph.rowptr[static_cast<std::size_t>(path_edge.from)] ||
            path_edge.csr_edge >= graph.rowptr[static_cast<std::size_t>(path_edge.from + 1)] ||
            graph.colind[static_cast<std::size_t>(path_edge.csr_edge)] != path_edge.to) {
          throw std::runtime_error("pathfinder result contains an invalid path edge");
        }
        if (!seen_edges.insert(edge_key(path_edge.from, path_edge.to)).second) {
          continue;
        }

        const EdgeAttr& attr =
            metadata.edge_attrs[static_cast<std::size_t>(path_edge.csr_edge)];
        if (attr.pip_data_index >= metadata.pip_data.size()) {
          throw std::runtime_error("route edge references invalid PIP data");
        }
        const PipData& pip =
            metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];

        if (!first_edge) {
          out << ',';
        }
        first_edge = false;
        out << "{\"from\":" << path_edge.from
            << ",\"to\":" << path_edge.to
            << ",\"csr_edge\":" << path_edge.csr_edge
            << ",\"tile\":";
        write_json_string(out, string_at(metadata, attr.tile_string));
        out << ",\"wire0\":";
        write_json_string(out, string_at(metadata, pip.wire0_string));
        out << ",\"wire1\":";
        write_json_string(out, string_at(metadata, pip.wire1_string));
        out << ",\"forward\":" << (pip.forward ? "true" : "false") << '}';
      }
    }
    out << "]}\n";
  }
}

}  // namespace routing

#ifndef ROUTING_PATHFINDER_NO_MAIN
int main(int argc, char** argv) {
  try {
    if (argc < 2 ||
        (argc == 2 && (std::string(argv[1]) == "-h" ||
                       std::string(argv[1]) == "--help"))) {
      routing::print_usage(argv[0]);
      return argc < 2 ? 1 : 0;
    }

    const std::filesystem::path csr_path = argv[1];
    std::filesystem::path metadata_path;
    std::filesystem::path routes_out_path;
    routing::PathfinderOptions options;
    bool allow_unrouted_routes = false;
    bool sssp_engine_control_seen = false;
    bool delta_option_seen = false;
    bool delta_benchmark_weights_seen = false;
    bool delta_benchmark_weight_seed_seen = false;
    bool net_limit_seen = false;
    bool diagnose_net_seen = false;
    bool diagnose_sink_seen = false;
    routing::UnitBfsPathDiagnostic unit_bfs_diagnostic;
    routing::DeltaBenchmarkWeights delta_benchmark_weights =
        routing::DeltaBenchmarkWeights::kOriginal;
    std::uint64_t delta_benchmark_weight_seed = 0;

    int arg = 2;
    if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
      metadata_path = argv[arg++];
    } else {
      metadata_path = routing::default_metadata_path(csr_path);
    }

    while (arg < argc) {
      const std::string option = argv[arg++];
      auto require_value = [&](const char* name) -> const char* {
        if (arg >= argc) {
          throw std::runtime_error(std::string(name) + " requires a value");
        }
        return argv[arg++];
      };

      if (option == "-h" || option == "--help") {
        routing::print_usage(argv[0]);
        return 0;
      }
      if (option == "--sssp-engine") {
        if (sssp_engine_control_seen) {
          throw std::runtime_error(
              "--sssp-engine and --use-delta-step are mutually exclusive");
        }
        sssp_engine_control_seen = true;
        options.sssp_engine =
            routing::parse_sssp_engine_arg(require_value("--sssp-engine"));
      } else if (option == "--use-delta-step") {
        if (sssp_engine_control_seen) {
          throw std::runtime_error(
              "--sssp-engine and --use-delta-step are mutually exclusive");
        }
        sssp_engine_control_seen = true;
        options.sssp_engine = routing::SsspEngine::kDeltaStep;
      } else if (option == "--delta") {
        routing::parse_delta_arg(require_value("--delta"), &options);
        delta_option_seen = true;
      } else if (option == "--delta-multiplier") {
        options.delta_multiplier = routing::parse_float_arg(
            require_value("--delta-multiplier"), "delta-multiplier");
        options.delta_controls_explicit = true;
      } else if (option == "--max-pathfinder-iters") {
        (void)routing::parse_int_arg(require_value("--max-pathfinder-iters"),
                                     "max-pathfinder-iters");
      } else if (option == "--max-sssp-iters") {
        options.max_sssp_iterations =
            routing::parse_int_arg(require_value("--max-sssp-iters"), "max-sssp-iters");
      } else if (option == "--delta-force-legacy-parent") {
        options.delta_force_legacy_parent = true;
      } else if (option == "--delta-force-generic") {
        options.delta_force_generic = true;
      } else if (option == "--delta-telemetry") {
        options.delta_telemetry = true;
      } else if (option == "--delta-benchmark-weights") {
        delta_benchmark_weights =
            routing::parse_delta_benchmark_weights_arg(
                require_value("--delta-benchmark-weights"));
        delta_benchmark_weights_seen = true;
      } else if (option == "--delta-benchmark-weight-seed") {
        delta_benchmark_weight_seed = routing::parse_u64_arg(
            require_value("--delta-benchmark-weight-seed"),
            "delta-benchmark-weight-seed");
        delta_benchmark_weight_seed_seen = true;
      } else if (option == "--capacity") {
        options.capacity = routing::parse_int_arg(require_value("--capacity"), "capacity");
      } else if (option == "--present-factor") {
        (void)routing::parse_float_arg(require_value("--present-factor"), "present-factor");
      } else if (option == "--present-multiplier") {
        (void)routing::parse_float_arg(require_value("--present-multiplier"),
                                       "present-multiplier");
      } else if (option == "--history-factor") {
        (void)routing::parse_float_arg(require_value("--history-factor"), "history-factor");
      } else if (option == "--net-limit") {
        options.net_limit =
            routing::parse_size_arg(require_value("--net-limit"), "net-limit");
        net_limit_seen = true;
      } else if (option == "--route-batch-size") {
        (void)routing::parse_size_arg(require_value("--route-batch-size"),
                                      "route-batch-size");
      } else if (option == "--parallel-net-workers") {
        options.parallel_net_workers =
            routing::parse_size_arg(require_value("--parallel-net-workers"),
                                    "parallel-net-workers");
      } else if (option == "--diagnose-net") {
        if (diagnose_net_seen) {
          throw std::runtime_error("--diagnose-net may be specified only once");
        }
        diagnose_net_seen = true;
        unit_bfs_diagnostic.net_index =
            routing::parse_size_arg(require_value("--diagnose-net"),
                                    "diagnose-net");
      } else if (option == "--diagnose-sink") {
        if (diagnose_sink_seen) {
          throw std::runtime_error("--diagnose-sink may be specified only once");
        }
        diagnose_sink_seen = true;
        unit_bfs_diagnostic.sink_index =
            routing::parse_size_arg(require_value("--diagnose-sink"),
                                    "diagnose-sink");
      } else if (option == "--allow-unrouted") {
        allow_unrouted_routes = true;
      } else if (option == "--routes-out") {
        routes_out_path = require_value("--routes-out");
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (delta_benchmark_weights_seen) {
      if (options.sssp_engine != routing::SsspEngine::kDeltaStep) {
        throw std::runtime_error(
            "--delta-benchmark-weights requires --sssp-engine delta-step "
            "or --use-delta-step");
      }
      if (!delta_option_seen || options.delta_auto) {
        throw std::runtime_error(
            "--delta-benchmark-weights requires an explicit numeric --delta");
      }
    }
    if (delta_benchmark_weight_seed_seen &&
        delta_benchmark_weights !=
            routing::DeltaBenchmarkWeights::kMixed) {
      throw std::runtime_error(
          "--delta-benchmark-weight-seed requires "
          "--delta-benchmark-weights mixed");
    }
    if (diagnose_net_seen != diagnose_sink_seen) {
      throw std::runtime_error(
          "--diagnose-net and --diagnose-sink must be specified together");
    }
    const bool diagnose_unit_bfs = diagnose_net_seen;
    if (diagnose_unit_bfs) {
      if (options.sssp_engine != routing::SsspEngine::kUnitBfs) {
        throw std::runtime_error(
            "UnitBFS path diagnostics require --sssp-engine unit-bfs");
      }
      if (net_limit_seen) {
        throw std::runtime_error(
            "--net-limit cannot be combined with --diagnose-net");
      }
      if (!routes_out_path.empty()) {
        throw std::runtime_error(
            "--routes-out cannot be combined with UnitBFS diagnostics");
      }
      if (unit_bfs_diagnostic.net_index ==
          std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("diagnostic net index is too large");
      }
      options.net_limit = unit_bfs_diagnostic.net_index + 1;
    }
    routing::validate_options(options);

    HostCsrF32 graph = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.load_csr");
      return routing::load_csrbin(csr_path);
    }();
    if (delta_benchmark_weights_seen) {
      routing::apply_delta_benchmark_weights(
          graph,
          delta_benchmark_weights,
          options.delta,
          delta_benchmark_weight_seed);
      std::cout << "[pathfinder] applied deterministic delta benchmark "
                << "weights="
                << routing::delta_benchmark_weights_name(
                       delta_benchmark_weights)
                << " seed=" << delta_benchmark_weight_seed << "\n";
    }

    routing::RoutingMetadata metadata = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.load_metadata");
      return routing::load_interchange_metadata(metadata_path);
    }();

    routing::PathfinderResult result =
        routing::run_pathfinder(
            graph,
            metadata,
            options,
            nullptr,
            diagnose_unit_bfs ? &unit_bfs_diagnostic : nullptr);

    if (diagnose_unit_bfs) {
      std::cout << routing::unit_bfs_path_diagnostic_json(
                       unit_bfs_diagnostic)
                << '\n';
      return 0;
    }

    if (!routes_out_path.empty()) {
      if (!result.routed && !allow_unrouted_routes) {
        std::cerr << "error: refusing to write routes because not all sinks were reached\n";
        return 2;
      }
      {
        PATHFINDER_PROFILE_RANGE("pathfinder.write_routes");
        routing::write_routes_jsonl(routes_out_path, graph, metadata, result);
      }
    }

    return result.routed || allow_unrouted_routes ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    if (argc < 2) {
      routing::print_usage(argv[0]);
    }
    return 1;
  }
}
#endif
