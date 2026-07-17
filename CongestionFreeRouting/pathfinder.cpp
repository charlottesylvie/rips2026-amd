#include "pathfinder.hpp"

#include "bellman_ford/bf8.hpp"
#include "delta_stepping/delta_stepping_hip_CSR.hpp"
#include "profiling/roctx_ranges.hpp"
#include "unit_bfs/unit_bfs_hip_CSR.hpp"

// CSR PathFinder router for the repository benchmark flow. Its default is a
// fast one-pass unit-BFS route, while --snapshot-iters enables deterministic
// congestion rip-up/reroute and freezes weighted CSR workloads for later SSSP
// comparisons. GPU delta-stepping and Bellman-Ford are selectable comparison
// engines.
//
// Example GPU build from the repository root:
//   hipcc -std=c++17 -O3 -x hip -DBF8_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/pathfinder.cpp \
//     CongestionFreeRouting/bellman_ford/bf8.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     -pthread \
//     -o congestion_free_pathfinder
// Add -DPATHFINDER_ENABLE_ROCTX -lrocprofiler-sdk-roctx for profiler ranges.
//
// Run:
//   ./congestion_free_pathfinder design.csrbin design.csrbin.ifmeta.bin --net-limit 10

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

void write_u64(std::ofstream& out, std::uint64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
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

template <typename T>
void write_array(std::ofstream& out,
                 const std::vector<T>& values,
                 const char* name) {
  if (values.empty()) {
    return;
  }
  const std::size_t bytes = values.size() * sizeof(T);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
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
  if (options.sssp_engine == SsspEngine::kDeltaStep &&
      (!(options.delta > 0.0f) || !std::isfinite(options.delta))) {
    throw std::invalid_argument("PathFinder delta must be finite and positive");
  }
  if (options.capacity <= 0) {
    throw std::invalid_argument("capacity must be positive");
  }
  if (options.max_pathfinder_iterations <= 0) {
    throw std::invalid_argument("max-pathfinder-iters must be positive");
  }
  if (!(options.initial_present_factor >= 0.0f) ||
      !std::isfinite(options.initial_present_factor)) {
    throw std::invalid_argument("present-factor must be finite and nonnegative");
  }
  if (!(options.present_factor_multiplier >= 1.0f) ||
      !std::isfinite(options.present_factor_multiplier)) {
    throw std::invalid_argument("present-multiplier must be finite and >= 1");
  }
  if (!(options.history_factor >= 0.0f) ||
      !std::isfinite(options.history_factor)) {
    throw std::invalid_argument("history-factor must be finite and nonnegative");
  }
  if (options.route_batch_size == 0) {
    throw std::invalid_argument("route-batch-size must be positive");
  }
  for (const int iteration : options.snapshot_iterations) {
    if (iteration <= 0) {
      throw std::invalid_argument("snapshot iterations must be positive");
    }
    if (iteration > options.max_pathfinder_iterations) {
      throw std::invalid_argument(
          "snapshot iteration exceeds max-pathfinder-iters");
    }
  }
  if (!options.snapshot_iterations.empty() && !options.snapshot_callback) {
    throw std::invalid_argument(
        "snapshot iterations require a Pathfinder snapshot callback");
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
RoutedNet route_net(const HostCsrF32& graph,
                    SsspWorkspace& workspace,
                    const RouteRequest& request,
                    std::vector<std::uint32_t>& tree_seen,
                    std::vector<int>& parent_by_child,
                    std::vector<std::uint32_t>& parent_seen,
                    std::uint32_t tree_stamp,
                    const PathfinderOptions& options,
                    hipStream_t stream) {
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

  std::vector<int> source_candidates;
  for (const SitePinNode& source : request.sources) {
    add_unique_node(source_candidates, tree_seen, tree_stamp, source.node);
  }

  if (source_candidates.empty()) {
    net.reached_all_sinks = false;
    return net;
  }

  std::vector<int> targets;
  targets.reserve(request.sinks.size());
  std::vector<std::size_t> target_sink_indices;
  target_sink_indices.reserve(request.sinks.size());
  bool reached_all = true;
  net.sinks.resize(request.sinks.size());
  for (std::size_t sink_index = 0; sink_index < request.sinks.size(); ++sink_index) {
    const SitePinNode& sink = request.sinks[sink_index];
    net.sinks[sink_index].target = sink.node;
    net.sinks[sink_index].distance = std::numeric_limits<float>::infinity();
    if (!valid_node(sink.node, graph.rows)) {
      reached_all = false;
      continue;
    }
    if (tree_contains(tree_seen, tree_stamp, sink.node)) {
      RoutedSink& routed_sink = net.sinks[sink_index];
      routed_sink.source = sink.node;
      routed_sink.distance = 0.0f;
      routed_sink.reached = true;
      routed_sink.nodes.push_back(sink.node);
      continue;
    }
    targets.push_back(sink.node);
    target_sink_indices.push_back(sink_index);
  }

  if (!targets.empty()) {
    auto sssp = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.sssp");
      return workspace.run(source_candidates,
                           targets,
                           options.delta,
                           options.max_sssp_iterations,
                           stream,
                           nullptr,
                           nullptr);
    }();

    const bool has_compact_target_paths =
        sssp.target_distances.size() == targets.size() &&
        sssp.target_sources.size() == targets.size() &&
        sssp.target_path_offsets.size() == targets.size() + 1 &&
        sssp.target_edge_offsets.size() == targets.size() + 1;

    for (std::size_t target_pos = 0; target_pos < target_sink_indices.size(); ++target_pos) {
      const std::size_t sink_index = target_sink_indices[target_pos];
      const int target = request.sinks[sink_index].node;
      RoutedSink& routed_sink = net.sinks[sink_index];

      if (has_compact_target_paths) {
        const float distance = sssp.target_distances[target_pos];
        const int node_begin = sssp.target_path_offsets[target_pos];
        const int node_end = sssp.target_path_offsets[target_pos + 1];
        const int edge_begin = sssp.target_edge_offsets[target_pos];
        const int edge_end = sssp.target_edge_offsets[target_pos + 1];
        if (!std::isfinite(distance) ||
            !valid_node(sssp.target_sources[target_pos], graph.rows) ||
            node_begin < 0 ||
            node_end <= node_begin ||
            edge_begin < 0 ||
            edge_end < edge_begin ||
            static_cast<std::size_t>(node_end) > sssp.target_path_nodes.size() ||
            static_cast<std::size_t>(edge_end) > sssp.target_path_edges.size()) {
          reached_all = false;
          continue;
        }

        std::size_t tree_start = 0;
        for (int node_index = node_begin; node_index < node_end; ++node_index) {
          const int path_node =
              sssp.target_path_nodes[static_cast<std::size_t>(node_index)];
          if (tree_contains(tree_seen, tree_stamp, path_node)) {
            tree_start = static_cast<std::size_t>(node_index - node_begin);
          }
        }

        routed_sink.source =
            sssp.target_path_nodes[static_cast<std::size_t>(node_begin) + tree_start];
        routed_sink.nodes.clear();
        routed_sink.nodes.reserve(static_cast<std::size_t>(node_end - node_begin) -
                                  tree_start);
        for (int node_index = node_begin + static_cast<int>(tree_start);
             node_index < node_end;
             ++node_index) {
          routed_sink.nodes.push_back(
              sssp.target_path_nodes[static_cast<std::size_t>(node_index)]);
        }
        const int trimmed_edge_begin = edge_begin + static_cast<int>(tree_start);
        if (routed_sink.nodes.empty() ||
            !valid_node(routed_sink.source, graph.rows) ||
            routed_sink.nodes.back() != target ||
            static_cast<std::size_t>(edge_end - trimmed_edge_begin) + 1 !=
                routed_sink.nodes.size()) {
          reached_all = false;
          routed_sink.nodes.clear();
          continue;
        }

        routed_sink.edges.reserve(static_cast<std::size_t>(edge_end - trimmed_edge_begin));
        bool valid_path = true;
        float path_distance = 0.0f;
        for (int edge_index = trimmed_edge_begin; edge_index < edge_end; ++edge_index) {
          const std::size_t path_index =
              static_cast<std::size_t>(edge_index - trimmed_edge_begin);
          const int from = routed_sink.nodes[path_index];
          const int to = routed_sink.nodes[path_index + 1];
          const minplus_sparse::Offset csr_edge =
              sssp.target_path_edges[static_cast<std::size_t>(edge_index)];
          if (!valid_node(from, graph.rows) ||
              !valid_node(to, graph.rows) ||
              csr_edge < graph.rowptr[static_cast<std::size_t>(from)] ||
              csr_edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
              graph.colind[static_cast<std::size_t>(csr_edge)] != to) {
            valid_path = false;
            break;
          }
          const float cost = graph.values[static_cast<std::size_t>(csr_edge)];
          path_distance += cost;
          routed_sink.edges.push_back(
              {from, to, csr_edge, cost});
        }
        if (!valid_path) {
          reached_all = false;
          routed_sink.edges.clear();
          routed_sink.nodes.clear();
          continue;
        }
        routed_sink.distance = path_distance;
      } else {
        if (static_cast<std::size_t>(target) >= sssp.dist.size() ||
            !std::isfinite(sssp.dist[static_cast<std::size_t>(target)])) {
          reached_all = false;
          continue;
        }

        int source = -1;
        routed_sink.edges =
            reconstruct_shortest_path_from_tree_pred(
                graph, sssp, tree_seen, tree_stamp, target, &source);
        if (!valid_node(source, graph.rows)) {
          reached_all = false;
          routed_sink.edges.clear();
          continue;
        }

        routed_sink.source = source;
        routed_sink.distance = sssp.dist[static_cast<std::size_t>(target)];
        routed_sink.nodes = nodes_from_path(source, routed_sink.edges);
      }

      if (!attach_path_if_single_parent_tree(routed_sink.edges,
                                             parent_by_child,
                                             parent_seen,
                                             tree_stamp)) {
        reached_all = false;
        routed_sink.edges.clear();
        routed_sink.nodes.clear();
        continue;
      }
      routed_sink.reached = true;
      for (const int node : routed_sink.nodes) {
        add_unique_node(source_candidates, tree_seen, tree_stamp, node);
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

void remove_net_occupancy(const RoutedNet& net, std::vector<int>& occupancy) {
  if (!net.reached_all_sinks) {
    return;
  }
  for (const int node : net.unique_nodes) {
    if (node < 0 || static_cast<std::size_t>(node) >= occupancy.size()) {
      continue;
    }
    int& used = occupancy[static_cast<std::size_t>(node)];
    if (used <= 0) {
      throw std::runtime_error("PathFinder rip-up found inconsistent occupancy");
    }
    --used;
  }
}

bool net_needs_reroute(const RoutedNet& net,
                       const std::vector<int>& occupancy,
                       int capacity) {
  if (!net.reached_all_sinks) {
    return true;
  }
  for (const int node : net.unique_nodes) {
    if (node >= 0 && static_cast<std::size_t>(node) < occupancy.size() &&
        occupancy[static_cast<std::size_t>(node)] > capacity) {
      return true;
    }
  }
  return false;
}

std::size_t route_request_weight(const RouteRequest& request) {
  return request.sources.size() + request.sinks.size();
}

// PathFinder charges for entering the destination routing node.  Keep this
// calculation on the host so the same costed CSR can be captured as a frozen
// workload and later reused by every SSSP comparison implementation.
void update_costed_graph_values(const HostCsrF32& base_graph,
                                const std::vector<int>& occupancy,
                                const std::vector<float>& history,
                                const PathfinderOptions& options,
                                float present_factor,
                                HostCsrF32& graph) {
  if (occupancy.size() != static_cast<std::size_t>(base_graph.rows) ||
      history.size() != static_cast<std::size_t>(base_graph.rows)) {
    throw std::invalid_argument("congestion state size does not match CSR rows");
  }
  // Keep the immutable topology allocated across batches.  The caller owns
  // this internal CSR and only its values are mutable between cost updates.
  if (graph.rows != base_graph.rows ||
      graph.cols != base_graph.cols ||
      graph.nnz != base_graph.nnz ||
      graph.rowptr.size() != base_graph.rowptr.size() ||
      graph.colind.size() != base_graph.colind.size()) {
    throw std::invalid_argument("costed CSR topology does not match base CSR");
  }
  graph.values.resize(base_graph.values.size());
  for (minplus_sparse::Offset source = 0; source < base_graph.rows; ++source) {
    for (minplus_sparse::Offset edge =
             base_graph.rowptr[static_cast<std::size_t>(source)];
         edge < base_graph.rowptr[static_cast<std::size_t>(source + 1)];
         ++edge) {
      const int destination = base_graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t destination_index = static_cast<std::size_t>(destination);
      const int overuse_if_taken =
          occupancy[destination_index] + 1 - options.capacity;
      const float present_cost = overuse_if_taken > 0
          ? 1.0f + present_factor * static_cast<float>(overuse_if_taken)
          : 1.0f;
      const float historical_cost = 1.0f + history[destination_index];
      graph.values[static_cast<std::size_t>(edge)] =
          base_graph.values[static_cast<std::size_t>(edge)] *
          present_cost * historical_cost;
    }
  }
}

HostCsrF32 make_costed_graph(const HostCsrF32& base_graph,
                             const std::vector<int>& occupancy,
                             const std::vector<float>& history,
                             const PathfinderOptions& options,
                             float present_factor) {
  HostCsrF32 graph = base_graph;
  update_costed_graph_values(base_graph,
                             occupancy,
                             history,
                             options,
                             present_factor,
                             graph);
  return graph;
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
                                   bool report_progress,
                                   std::vector<RoutedNet>& nets,
                                   WorkspaceFactory workspace_factory) {
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
      nets[net_index] =
          route_net(base_graph,
                    sssp_workspace,
                    request,
                    route_tree_seen,
                    route_parent_by_child,
                    route_parent_seen,
                    tree_stamp,
                    options,
                    stream);

      if (report_progress &&
          ((net_index + 1) == route_request_count ||
           (net_index + 1) % progress_interval == 0)) {
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

  auto report_completed_progress = [&](std::size_t completed) {
    std::lock_guard<std::mutex> lock(progress_mutex);
    if (report_progress &&
        (completed == route_request_count ||
         completed - last_reported >= progress_interval)) {
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
        nets[net_index] =
            route_net(base_graph,
                      sssp_workspace,
                      request,
                      route_tree_seen,
                      route_parent_by_child,
                      route_parent_seen,
                      tree_stamp,
                      options,
                      local_stream);

        const std::size_t completed =
            completed_nets.fetch_add(1, std::memory_order_relaxed) + 1;
        report_completed_progress(completed);
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
  for (std::size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back(worker);
  }
  for (std::thread& thread : workers) {
    thread.join();
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

bool congestion_iterations_requested(const PathfinderOptions& options) {
  return options.enable_congestion ||
         options.max_pathfinder_iterations > 1 ||
         !options.snapshot_iterations.empty();
}

bool snapshot_requested(const PathfinderOptions& options, int iteration) {
  return std::find(options.snapshot_iterations.begin(),
                   options.snapshot_iterations.end(),
                   iteration) != options.snapshot_iterations.end();
}

bool snapshot_requested_after(const PathfinderOptions& options, int iteration) {
  return std::any_of(options.snapshot_iterations.begin(),
                     options.snapshot_iterations.end(),
                     [iteration](int requested_iteration) {
                       return requested_iteration > iteration;
                     });
}

// This is intentionally sequential.  Every batch observes a single
// congestion snapshot, then commits its routes before the next batch is
// costed.  Parallel net workers would race that occupancy transition and make
// saved CSR snapshots nondeterministic.
PathfinderResult run_congestion_pathfinder(const HostCsrF32& base_graph,
                                           const RoutingMetadata& metadata,
                                           const PathfinderOptions& options,
                                           hipStream_t stream) {
  if (options.sssp_engine != SsspEngine::kDeltaStep) {
    throw std::invalid_argument(
        "congestion iterations require --sssp-engine delta-step; use the "
        "frozen CSR snapshots to compare other engines");
  }

  PathfinderResult result;
  result.occupancy.assign(static_cast<std::size_t>(base_graph.rows), 0);
  result.history.assign(static_cast<std::size_t>(base_graph.rows), 0.0f);
  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());
  result.nets.resize(route_request_count);

  DeltaSteppingCsrWorkspace workspace(base_graph, stream);
  HostCsrF32 costed_graph = base_graph;
  std::vector<std::uint32_t> route_tree_seen(
      static_cast<std::size_t>(base_graph.rows), 0);
  std::vector<int> route_parent_by_child(
      static_cast<std::size_t>(base_graph.rows), -1);
  std::vector<std::uint32_t> route_parent_seen(
      static_cast<std::size_t>(base_graph.rows), 0);
  std::uint32_t route_tree_stamp = 0;
  float present_factor = options.initial_present_factor;

  if (options.report_progress) {
    std::cout << "[pathfinder] congestion snapshot mode: iterations="
              << options.max_pathfinder_iterations
              << " batches=" << options.route_batch_size
              << " requested_snapshots=" << options.snapshot_iterations.size()
              << "\n";
  }

  for (int iteration = 1;
       iteration <= options.max_pathfinder_iterations;
       ++iteration) {
    PATHFINDER_PROFILE_RANGE("pathfinder.congestion_iteration");
    std::vector<std::size_t> reroute_indices;
    reroute_indices.reserve(route_request_count);
    if (iteration == 1) {
      std::fill(result.occupancy.begin(), result.occupancy.end(), 0);
      for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
        reroute_indices.push_back(net_index);
      }
    } else {
      for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
        if (net_needs_reroute(result.nets[net_index],
                              result.occupancy,
                              options.capacity)) {
          reroute_indices.push_back(net_index);
        }
      }
      for (const std::size_t net_index : reroute_indices) {
        remove_net_occupancy(result.nets[net_index], result.occupancy);
      }
    }

    std::stable_sort(
        reroute_indices.begin(), reroute_indices.end(),
        [&](std::size_t left, std::size_t right) {
          const std::size_t left_weight =
              route_request_weight(metadata.route_requests[left]);
          const std::size_t right_weight =
              route_request_weight(metadata.route_requests[right]);
          return left_weight == right_weight ? left < right
                                             : left_weight > right_weight;
        });

    const std::size_t reroute_count = reroute_indices.size();
    const std::size_t progress_interval =
        std::max<std::size_t>(1, reroute_count / 100);
    if (options.report_progress) {
      std::cout << "[pathfinder] iteration " << iteration
                << "/" << options.max_pathfinder_iterations
                << " present_factor=" << present_factor
                << " rerouting=" << reroute_count
                << "/" << route_request_count << "\n";
    }

    for (std::size_t batch_begin = 0;
         batch_begin < reroute_count;
         batch_begin += options.route_batch_size) {
      const std::size_t batch_end =
          std::min(reroute_count, batch_begin + options.route_batch_size);
      update_costed_graph_values(base_graph,
                                 result.occupancy,
                                 result.history,
                                 options,
                                 present_factor,
                                 costed_graph);
      workspace.update_values(costed_graph.values, stream);

      std::vector<std::size_t> completed_batch;
      completed_batch.reserve(batch_end - batch_begin);
      for (std::size_t route_pos = batch_begin; route_pos < batch_end; ++route_pos) {
        const std::size_t net_index = reroute_indices[route_pos];
        const std::uint32_t tree_stamp =
            next_tree_stamp(route_tree_seen, &route_tree_stamp);
        result.nets[net_index] = route_net(costed_graph,
                                           workspace,
                                           metadata.route_requests[net_index],
                                           route_tree_seen,
                                           route_parent_by_child,
                                           route_parent_seen,
                                           tree_stamp,
                                           options,
                                           stream);
        completed_batch.push_back(net_index);
        if (options.report_progress &&
            ((route_pos + 1) == reroute_count ||
             (route_pos + 1) % progress_interval == 0)) {
          print_pathfinder_progress(iteration,
                                    options.max_pathfinder_iterations,
                                    route_pos + 1,
                                    reroute_count);
        }
      }
      for (const std::size_t net_index : completed_batch) {
        commit_net_occupancy(result.nets[net_index], result.occupancy);
      }
    }

    result.all_sinks_reached = true;
    for (const RoutedNet& net : result.nets) {
      if (!net.reached_all_sinks) {
        result.all_sinks_reached = false;
        break;
      }
    }
    result.iterations_used = iteration;
    update_congestion_stats(result.occupancy,
                            options.capacity,
                            &result.overused_nodes,
                            &result.max_occupancy);
    result.routed = result.all_sinks_reached && result.overused_nodes == 0;

    // A frozen snapshot represents the completed iteration's global
    // congestion landscape. Overuse is added to history and the present
    // factor is advanced before materializing costs for an additional entrant
    // at each destination. It is deliberately a stable post-iteration
    // workload, not a literal later batch after the next iteration's rip-up.
    if (!result.routed) {
      for (std::size_t node = 0; node < result.occupancy.size(); ++node) {
        const int overuse = result.occupancy[node] - options.capacity;
        if (overuse > 0) {
          result.history[node] +=
              options.history_factor * static_cast<float>(overuse);
        }
      }
      present_factor *= options.present_factor_multiplier;
    }

    if (snapshot_requested(options, iteration)) {
      PathfinderSnapshot snapshot;
      snapshot.iteration = iteration;
      snapshot.present_factor = present_factor;
      snapshot.overused_nodes = result.overused_nodes;
      snapshot.max_occupancy = result.max_occupancy;
      snapshot.all_sinks_reached = result.all_sinks_reached;
      snapshot.occupancy = result.occupancy;
      snapshot.history = result.history;
      snapshot.costed_graph = make_costed_graph(base_graph,
                                                 snapshot.occupancy,
                                                 snapshot.history,
                                                 options,
                                                 snapshot.present_factor);
      options.snapshot_callback(snapshot);
    }

    if (options.report_progress) {
      std::cout << "[pathfinder] iteration " << iteration
                << " complete: all_sinks_reached="
                << (result.all_sinks_reached ? "true" : "false")
                << " overused_nodes=" << result.overused_nodes
                << " max_occupancy=" << result.max_occupancy
                << " routed=" << (result.routed ? "true" : "false")
                << "\n";
    }
    // A caller may explicitly request checkpoints after convergence.  Do not
    // silently omit those files: subsequent passes have no reroutes and thus
    // faithfully capture the unchanged legal congestion state.  Ordinary
    // congestion runs still stop as soon as they converge.
    if (result.routed && !snapshot_requested_after(options, iteration)) {
      break;
    }
  }

  return result;
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

SsspEngine parse_sssp_engine_arg(const char* text) {
  const std::string value(text);
  if (value == "unit-bfs" || value == "bfs") {
    return SsspEngine::kUnitBfs;
  }
  if (value == "delta-step" || value == "delta-stepping" || value == "delta") {
    return SsspEngine::kDeltaStep;
  }
  if (value == "bellman-ford" || value == "bellman_ford" || value == "bf8") {
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
      << "  --sssp-engine <unit-bfs|delta-step|bellman-ford>\n"
      << "                                  Shortest-path backend. Default: unit-bfs\n"
      << "  --use-delta-step                Use delta-step backend for comparison.\n"
      << "  --delta <float>                 Delta-stepping bucket width. Default: 4\n"
      << "  --max-sssp-iters <int>          Delta rounds, BFS depth, or Bellman-Ford rounds; -1 for default.\n"
      << "  --capacity <int>                Resource capacity for congestion costs. Default: 1\n"
      << "  --max-pathfinder-iters <int>    Enable delta-step congestion routing for this many rounds.\n"
      << "  --present-factor <float>        Initial present congestion factor. Default: 1\n"
      << "  --present-multiplier <float>    Per-round present-factor multiplier. Default: 2\n"
      << "  --history-factor <float>        Historical congestion increment. Default: 1\n"
      << "  --route-batch-size <count>      Nets routed per congestion-cost snapshot. Default: 256\n"
      << "  --snapshot-iters <1,3,5,...>    Save frozen costed CSRs after these PathFinder rounds.\n"
      << "  --snapshot-dir <path>           Directory for --snapshot-iters CSR/metadata pairs.\n"
      << "  --net-limit <count>             Route only the first count requests.\n"
      << "  --parallel-net-workers <count>  Independent net workers. Default: 0 (engine-dependent auto).\n"
      << "  --allow-unrouted                Write partial routes even if some sinks are unreached.\n"
      << "  --routes-out <path>             Write routed PIP tree data as JSONL.\n"
      << "\nCongestion runs require delta-step; snapshots capture the post-iteration global congestion landscape.\n";
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

void write_csrbin(const std::filesystem::path& path, const HostCsrF32& graph) {
  validate_csr(graph);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("could not open CSR output file: " + path.string());
  }

  out.write(CSR_MAGIC, sizeof(CSR_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing CSR magic");
  }
  const std::uint64_t rows = static_cast<std::uint64_t>(graph.rows);
  const std::uint64_t cols = static_cast<std::uint64_t>(graph.cols);
  const std::uint64_t nnz = static_cast<std::uint64_t>(graph.nnz);
  write_u64(out, EXPECTED_CSR_VERSION, "CSR format version");
  write_u64(out, EXPECTED_OUTGOING_EDGE_ORIENTATION, "CSR orientation");
  write_u64(out, rows, "CSR row count");
  write_u64(out, cols, "CSR column count");
  // HostCsrF32 intentionally does not retain the converter's declared and
  // loaded edge counts. Readers ignore those advisory fields, so the frozen
  // graph's actual edge count is the faithful value for both.
  write_u64(out, nnz, "CSR declared edge count");
  write_u64(out, nnz, "CSR loaded edge count");
  write_u64(out, nnz, "CSR nnz");
  write_u64(out,
            static_cast<std::uint64_t>(graph.rowptr.size()),
            "CSR rowptr count");
  write_u64(out,
            static_cast<std::uint64_t>(graph.colind.size()),
            "CSR colind count");
  write_u64(out,
            static_cast<std::uint64_t>(graph.values.size()),
            "CSR values count");
  write_array(out, graph.rowptr, "CSR rowptr");
  write_array(out, graph.colind, "CSR colind");
  write_array(out, graph.values, "CSR values");
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
                                hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("pathfinder.run");
  validate_csr(base_graph);
  validate_options(options);
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

  if (congestion_iterations_requested(options)) {
    return run_congestion_pathfinder(base_graph, metadata, options, stream);
  }

  PathfinderResult result;
  result.occupancy.assign(static_cast<std::size_t>(base_graph.rows), 0);

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());
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
        if (unit_options.report_progress) {
          std::cout << "[pathfinder] auto-selected "
                    << unit_options.parallel_net_workers
                    << " unit-BFS worker(s)\n";
        }
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          unit_options,
          stream,
          route_request_count,
          progress_interval,
          unit_options.report_progress,
          result.nets,
          [shared_graph](hipStream_t worker_stream) {
            return UnitBfsCsrWorkspace(shared_graph, worker_stream);
          });
      break;
    }
    case SsspEngine::kDeltaStep: {
      auto shared_graph =
          std::make_shared<DeltaSteppingCsrGraph>(base_graph, stream);
      PathfinderOptions delta_options = options;
      if (delta_options.parallel_net_workers == 0) {
        const bool uses_unit_specialization =
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
        if (delta_options.report_progress) {
          std::cout << "[pathfinder] auto-selected "
                    << delta_options.parallel_net_workers
                    << " delta-step worker(s)\n";
        }
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          delta_options,
          stream,
          route_request_count,
          progress_interval,
          delta_options.report_progress,
          result.nets,
          [shared_graph](hipStream_t worker_stream) {
            return DeltaSteppingCsrWorkspace(shared_graph, worker_stream);
      });
      break;
    }
    case SsspEngine::kBellmanFord: {
      auto shared_graph =
          std::make_shared<BellmanFordCsrGraph>(base_graph, stream);
      PathfinderOptions bellman_ford_options = options;
      if (bellman_ford_options.parallel_net_workers == 0) {
        // Start conservatively: every worker owns a full frontier state and
        // may copy one packed predecessor tree to the host per legal source.
        bellman_ford_options.parallel_net_workers = 1;
        if (bellman_ford_options.report_progress) {
          std::cout << "[pathfinder] auto-selected 1 Bellman-Ford worker(s)\n";
        }
      }
      route_all_nets_with_workspace(
          base_graph,
          metadata,
          bellman_ford_options,
          stream,
          route_request_count,
          progress_interval,
          bellman_ford_options.report_progress,
          result.nets,
          [shared_graph](hipStream_t worker_stream) {
            return BellmanFordCsrWorkspace(shared_graph, worker_stream);
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
    std::filesystem::path snapshot_dir;
    routing::PathfinderOptions options;
    bool allow_unrouted_routes = false;
    bool max_pathfinder_iters_was_set = false;
    bool sssp_engine_was_set = false;

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
        options.sssp_engine =
            routing::parse_sssp_engine_arg(require_value("--sssp-engine"));
        sssp_engine_was_set = true;
      } else if (option == "--use-delta-step") {
        options.sssp_engine = routing::SsspEngine::kDeltaStep;
        sssp_engine_was_set = true;
      } else if (option == "--delta") {
        options.delta = routing::parse_float_arg(require_value("--delta"), "delta");
      } else if (option == "--max-pathfinder-iters") {
        options.max_pathfinder_iterations = routing::parse_int_arg(
            require_value("--max-pathfinder-iters"), "max-pathfinder-iters");
        // Supplying the flag opts into the real PathFinder loop even for a
        // single round, which is useful when capturing/inspecting its costs.
        options.enable_congestion = true;
        max_pathfinder_iters_was_set = true;
      } else if (option == "--max-sssp-iters") {
        options.max_sssp_iterations =
            routing::parse_int_arg(require_value("--max-sssp-iters"), "max-sssp-iters");
      } else if (option == "--capacity") {
        options.capacity = routing::parse_int_arg(require_value("--capacity"), "capacity");
      } else if (option == "--present-factor") {
        options.initial_present_factor = routing::parse_float_arg(
            require_value("--present-factor"), "present-factor");
      } else if (option == "--present-multiplier") {
        options.present_factor_multiplier = routing::parse_float_arg(
            require_value("--present-multiplier"), "present-multiplier");
      } else if (option == "--history-factor") {
        options.history_factor = routing::parse_float_arg(
            require_value("--history-factor"), "history-factor");
      } else if (option == "--net-limit") {
        options.net_limit =
            routing::parse_size_arg(require_value("--net-limit"), "net-limit");
      } else if (option == "--route-batch-size") {
        options.route_batch_size = routing::parse_size_arg(
            require_value("--route-batch-size"), "route-batch-size");
      } else if (option == "--snapshot-iters") {
        const std::string values = require_value("--snapshot-iters");
        if (values.empty()) {
          throw std::runtime_error("--snapshot-iters requires at least one iteration");
        }
        std::size_t begin = 0;
        while (begin < values.size()) {
          const std::size_t comma = values.find(',', begin);
          const std::size_t end = comma == std::string::npos ? values.size() : comma;
          if (end == begin) {
            throw std::runtime_error("--snapshot-iters contains an empty value");
          }
          const std::string value = values.substr(begin, end - begin);
          options.snapshot_iterations.push_back(
              routing::parse_int_arg(value.c_str(), "snapshot iteration"));
          if (comma == std::string::npos) {
            break;
          }
          begin = comma + 1;
        }
        options.enable_congestion = true;
      } else if (option == "--snapshot-dir") {
        snapshot_dir = require_value("--snapshot-dir");
      } else if (option == "--parallel-net-workers") {
        options.parallel_net_workers =
            routing::parse_size_arg(require_value("--parallel-net-workers"),
                                    "parallel-net-workers");
      } else if (option == "--allow-unrouted") {
        allow_unrouted_routes = true;
      } else if (option == "--routes-out") {
        routes_out_path = require_value("--routes-out");
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (options.snapshot_iterations.empty() != snapshot_dir.empty()) {
      throw std::runtime_error(
          "--snapshot-iters and --snapshot-dir must be supplied together");
    }
    if (!options.snapshot_iterations.empty()) {
      std::sort(options.snapshot_iterations.begin(),
                options.snapshot_iterations.end());
      const auto duplicate = std::adjacent_find(options.snapshot_iterations.begin(),
                                                options.snapshot_iterations.end());
      if (duplicate != options.snapshot_iterations.end()) {
        throw std::runtime_error("--snapshot-iters must not contain duplicates");
      }
      const int last_snapshot = options.snapshot_iterations.back();
      if (max_pathfinder_iters_was_set &&
          options.max_pathfinder_iterations < last_snapshot) {
        throw std::runtime_error(
            "--max-pathfinder-iters is smaller than a requested snapshot iteration");
      }
      if (!max_pathfinder_iters_was_set) {
        options.max_pathfinder_iterations = last_snapshot;
      }
      const std::filesystem::path snapshot_root = snapshot_dir;
      const std::filesystem::path snapshot_metadata = metadata_path;
      const std::filesystem::path source_csr = csr_path;
      const int snapshot_capacity = options.capacity;
      const float snapshot_delta = options.delta;
      const float snapshot_initial_present_factor = options.initial_present_factor;
      const float snapshot_history_factor = options.history_factor;
      const float snapshot_present_multiplier = options.present_factor_multiplier;
      const int snapshot_max_pathfinder_iterations =
          options.max_pathfinder_iterations;
      const std::size_t snapshot_route_batch_size = options.route_batch_size;
      const std::size_t snapshot_net_limit = options.net_limit;
      const bool snapshot_report_progress = options.report_progress;
      options.snapshot_callback =
          [snapshot_root,
           snapshot_metadata,
           source_csr,
           snapshot_capacity,
           snapshot_delta,
           snapshot_initial_present_factor,
           snapshot_history_factor,
           snapshot_present_multiplier,
           snapshot_max_pathfinder_iterations,
           snapshot_route_batch_size,
           snapshot_net_limit,
           snapshot_report_progress](const routing::PathfinderSnapshot& snapshot) {
            std::filesystem::create_directories(snapshot_root);
            const std::string stem = "iteration-" + std::to_string(snapshot.iteration);
            const std::filesystem::path snapshot_csr =
                snapshot_root / (stem + ".csrbin");
            const std::filesystem::path snapshot_metadata_out =
                std::filesystem::path(snapshot_csr.string() + ".ifmeta.bin");
            const std::filesystem::path manifest_path =
                snapshot_root / (stem + ".snapshot.json");
            routing::write_csrbin(snapshot_csr, snapshot.costed_graph);
            std::filesystem::copy_file(snapshot_metadata,
                                       snapshot_metadata_out,
                                       std::filesystem::copy_options::overwrite_existing);

            std::ofstream manifest(manifest_path, std::ios::trunc);
            if (!manifest) {
              throw std::runtime_error(
                  "could not write snapshot manifest: " + manifest_path.string());
            }
            const auto write_json_string = [&manifest](const std::string& text) {
              manifest << '"';
              for (const unsigned char ch : text) {
                switch (ch) {
                  case '"': manifest << "\\\""; break;
                  case '\\': manifest << "\\\\"; break;
                  case '\n': manifest << "\\n"; break;
                  case '\r': manifest << "\\r"; break;
                  case '\t': manifest << "\\t"; break;
                  default: manifest << static_cast<char>(ch); break;
                }
              }
              manifest << '"';
            };
            int total_occupancy = 0;
            std::size_t history_nodes = 0;
            float max_history = 0.0f;
            for (std::size_t node = 0; node < snapshot.occupancy.size(); ++node) {
              total_occupancy += snapshot.occupancy[node];
              const float history = snapshot.history[node];
              if (history != 0.0f) {
                ++history_nodes;
              }
              max_history = std::max(max_history, history);
            }
            manifest << "{\n  \"cost_phase\": \"post_iteration_global_cost\",\n"
                     << "  \"iteration\": " << snapshot.iteration << ",\n"
                     << "  \"delta\": " << snapshot_delta << ",\n"
                     << "  \"initial_present_factor\": "
                     << snapshot_initial_present_factor << ",\n"
                     << "  \"present_factor\": " << snapshot.present_factor << ",\n"
                     << "  \"capacity\": " << snapshot_capacity << ",\n"
                     << "  \"history_factor\": " << snapshot_history_factor << ",\n"
                     << "  \"present_multiplier\": " << snapshot_present_multiplier << ",\n"
                     << "  \"max_pathfinder_iterations\": "
                     << snapshot_max_pathfinder_iterations << ",\n"
                     << "  \"route_batch_size\": "
                     << snapshot_route_batch_size << ",\n"
                     << "  \"net_limit\": " << snapshot_net_limit << ",\n"
                     << "  \"all_sinks_reached\": "
                     << (snapshot.all_sinks_reached ? "true" : "false") << ",\n"
                     << "  \"overused_nodes\": " << snapshot.overused_nodes << ",\n"
                     << "  \"max_occupancy\": " << snapshot.max_occupancy << ",\n"
                     << "  \"total_occupancy\": " << total_occupancy << ",\n"
                     << "  \"history_nodes\": " << history_nodes << ",\n"
                     << "  \"max_history\": " << max_history << ",\n"
                     << "  \"source_csr\": ";
            write_json_string(source_csr.string());
            manifest << ",\n  \"csrbin\": ";
            write_json_string(snapshot_csr.filename().string());
            manifest << ",\n  \"metadata\": ";
            write_json_string(snapshot_metadata_out.filename().string());
            manifest << "\n}\n";
            if (!manifest) {
              throw std::runtime_error(
                  "failed while writing snapshot manifest: " + manifest_path.string());
            }
            if (snapshot_report_progress) {
              std::cout << "[pathfinder] wrote congestion snapshot "
                        << snapshot_csr << "\n";
            }
          };
    }
    // The normal one-shot default is unit BFS. Any congestion run needs a
    // weighted implementation; choose delta-step unless the caller made an
    // explicit engine selection (which validation will check).
    if (options.enable_congestion && !sssp_engine_was_set) {
      options.sssp_engine = routing::SsspEngine::kDeltaStep;
    }

    HostCsrF32 graph = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.load_csr");
      return routing::load_csrbin(csr_path);
    }();

    routing::RoutingMetadata metadata = [&]() {
      PATHFINDER_PROFILE_RANGE("pathfinder.load_metadata");
      return routing::load_interchange_metadata(metadata_path);
    }();

    routing::PathfinderResult result =
        routing::run_pathfinder(graph, metadata, options, nullptr);

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

    return result.routed || allow_unrouted_routes ||
                   !options.snapshot_iterations.empty()
               ? 0
               : 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    if (argc < 2) {
      routing::print_usage(argv[0]);
    }
    return 1;
  }
}
#endif
