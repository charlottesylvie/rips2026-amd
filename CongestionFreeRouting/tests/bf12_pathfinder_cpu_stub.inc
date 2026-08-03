// Deterministic host substitute for BF12's PathFinder-facing API. Include it
// in a test translation unit after pathfinder12.hpp, and link pathfinder12.cpp
// as a separate translation unit with ROUTING_PATHFINDER12_NO_MAIN. Every
// descriptor runs an independent Dijkstra state, so shared vertices never
// share labels.

#include "../bellman_ford/bf12_worker_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bf12_cpu_stub {

struct Stats {
  std::size_t graph_constructions = 0;
  std::size_t workspace_constructions = 0;
  std::size_t dynamic_cost_uploads = 0;
  std::size_t cooperative_occupancy_queries = 0;
  std::vector<std::size_t> initial_batch_sizes;
  std::vector<std::vector<std::uint32_t>> initial_original_indices;
  std::vector<std::uint32_t> retried_original_indices;
};

inline Stats stats;
inline std::string architecture = "gfx1151";
inline int compute_unit_count = 16;
inline bool cooperative_supported = true;

inline void reset() {
  stats = {};
  architecture = "gfx1151";
  compute_unit_count = 16;
  cooperative_supported = true;
}

inline bool valid_node(int node, const HostCsrF32& graph) {
  return node >= 0 &&
         static_cast<minplus_sparse::Offset>(node) < graph.rows;
}

inline float effective_weight(const HostCsrF32& graph,
                              const std::vector<float>& base_cost,
                              const std::vector<float>& dynamic_cost,
                              minplus_sparse::Offset edge) {
  const int destination = graph.colind[static_cast<std::size_t>(edge)];
  const float edge_value = graph.values[static_cast<std::size_t>(edge)];
  const float dynamic = dynamic_cost[static_cast<std::size_t>(destination)];
  return edge_value == 0.0f || dynamic == 0.0f
             ? 0.0f
             : edge_value * base_cost[static_cast<std::size_t>(destination)] *
                   dynamic;
}

inline bool admitted(const BellmanFord12BoundingBox& bounds,
                     const std::vector<std::int32_t>& x,
                     const std::vector<std::int32_t>& y,
                     int node) {
  if (!bounds.enabled) return true;
  const std::int32_t node_x = x[static_cast<std::size_t>(node)];
  const std::int32_t node_y = y[static_cast<std::size_t>(node)];
  if (node_x == routing::interchange::kMissingRouteCoordinate &&
      node_y == routing::interchange::kMissingRouteCoordinate) {
    return true;
  }
  return node_x >= bounds.min_x && node_x <= bounds.max_x &&
         node_y >= bounds.min_y && node_y <= bounds.max_y;
}

struct CpuQuery {
  BellmanFordCsrResult result;
  BellmanFord12QueryStatus status;
};

inline CpuQuery run_query(const HostCsrF32& graph,
                          const std::vector<std::int32_t>& x,
                          const std::vector<std::int32_t>& y,
                          const std::vector<float>& base_cost,
                          const std::vector<float>& dynamic_cost,
                          const std::vector<int>& sources,
                          const std::vector<int>& targets,
                          BellmanFord12BoundingBox bounds,
                          int max_iterations) {
  const float infinity = std::numeric_limits<float>::infinity();
  const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
  std::vector<float> distance(vertex_count, infinity);
  std::vector<int> predecessor(vertex_count, -1);
  std::vector<minplus_sparse::Offset> predecessor_edge(vertex_count, -1);
  std::vector<int> hop_count(vertex_count, std::numeric_limits<int>::max());
  std::unordered_set<int> source_set;
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem, std::vector<QueueItem>,
                      std::greater<QueueItem>> queue;
  for (const int source : sources) {
    if (!valid_node(source, graph)) {
      throw std::out_of_range("BF12 CPU stub source is outside the graph");
    }
    if (bounds.enabled) {
      const std::int32_t source_x = x[static_cast<std::size_t>(source)];
      const std::int32_t source_y = y[static_cast<std::size_t>(source)];
      if (source_x == routing::interchange::kMissingRouteCoordinate ||
          source_y == routing::interchange::kMissingRouteCoordinate ||
          !admitted(bounds, x, y, source)) {
        throw std::invalid_argument("BF12 CPU stub bounded source is invalid");
      }
    }
    if (!source_set.insert(source).second) continue;
    distance[static_cast<std::size_t>(source)] = 0.0f;
    predecessor[static_cast<std::size_t>(source)] = source;
    hop_count[static_cast<std::size_t>(source)] = 0;
    queue.push({0.0f, source});
  }
  for (const int target : targets) {
    if (!valid_node(target, graph)) {
      throw std::out_of_range("BF12 CPU stub target is outside the graph");
    }
    if (bounds.enabled) {
      const std::int32_t target_x = x[static_cast<std::size_t>(target)];
      const std::int32_t target_y = y[static_cast<std::size_t>(target)];
      if (target_x == routing::interchange::kMissingRouteCoordinate ||
          target_y == routing::interchange::kMissingRouteCoordinate ||
          !admitted(bounds, x, y, target)) {
        throw std::invalid_argument("BF12 CPU stub bounded target is invalid");
      }
    }
  }

  const int iteration_limit =
      max_iterations < 0 ? static_cast<int>(graph.rows) : max_iterations;
  bool pruned_by_limit = false;
  while (!queue.empty()) {
    const auto [current_distance, node] = queue.top();
    queue.pop();
    if (current_distance != distance[static_cast<std::size_t>(node)]) continue;
    if (hop_count[static_cast<std::size_t>(node)] >= iteration_limit) {
      if (graph.rowptr[static_cast<std::size_t>(node)] !=
          graph.rowptr[static_cast<std::size_t>(node + 1)]) {
        pruned_by_limit = true;
      }
      continue;
    }
    for (minplus_sparse::Offset edge =
             graph.rowptr[static_cast<std::size_t>(node)];
         edge < graph.rowptr[static_cast<std::size_t>(node + 1)]; ++edge) {
      const int destination = graph.colind[static_cast<std::size_t>(edge)];
      if (!admitted(bounds, x, y, destination)) continue;
      const float candidate = current_distance +
                              effective_weight(graph, base_cost, dynamic_cost,
                                               edge);
      if (candidate < distance[static_cast<std::size_t>(destination)]) {
        distance[static_cast<std::size_t>(destination)] = candidate;
        predecessor[static_cast<std::size_t>(destination)] = node;
        predecessor_edge[static_cast<std::size_t>(destination)] = edge;
        hop_count[static_cast<std::size_t>(destination)] =
            hop_count[static_cast<std::size_t>(node)] + 1;
        queue.push({candidate, destination});
      }
    }
  }

  CpuQuery output;
  output.result.iterations_used = std::min(iteration_limit,
                                           static_cast<int>(graph.rows));
  output.result.converged = !pruned_by_limit;
  output.result.stopped_on_target = false;
  output.result.target_reached = true;
  output.result.target_path_offsets.push_back(0);
  output.result.target_edge_offsets.push_back(0);
  bool all_identity = true;
  for (const int target : targets) {
    const float target_distance = distance[static_cast<std::size_t>(target)];
    output.result.target_distances.push_back(target_distance);
    if (!std::isfinite(target_distance)) {
      output.result.target_sources.push_back(-1);
      output.result.target_reached = false;
      all_identity = false;
    } else {
      std::vector<int> reversed_nodes;
      std::vector<minplus_sparse::Offset> reversed_edges;
      int current = target;
      for (std::size_t guard = 0; guard <= vertex_count; ++guard) {
        reversed_nodes.push_back(current);
        if (source_set.find(current) != source_set.end()) break;
        const int parent = predecessor[static_cast<std::size_t>(current)];
        const minplus_sparse::Offset edge =
            predecessor_edge[static_cast<std::size_t>(current)];
        if (!valid_node(parent, graph) || edge < 0) {
          throw std::runtime_error("BF12 CPU stub predecessor chain is invalid");
        }
        reversed_edges.push_back(edge);
        current = parent;
      }
      if (source_set.find(reversed_nodes.back()) == source_set.end()) {
        throw std::runtime_error("BF12 CPU stub predecessor chain is cyclic");
      }
      output.result.target_sources.push_back(reversed_nodes.back());
      std::reverse(reversed_nodes.begin(), reversed_nodes.end());
      std::reverse(reversed_edges.begin(), reversed_edges.end());
      output.result.target_path_nodes.insert(
          output.result.target_path_nodes.end(), reversed_nodes.begin(),
          reversed_nodes.end());
      for (const minplus_sparse::Offset edge : reversed_edges) {
        output.result.target_path_edges.push_back(edge);
        output.result.target_path_edge_costs.push_back(
            effective_weight(graph, base_cost, dynamic_cost, edge));
      }
      all_identity = all_identity && reversed_edges.empty();
    }
    output.result.target_path_offsets.push_back(
        static_cast<int>(output.result.target_path_nodes.size()));
    output.result.target_edge_offsets.push_back(
        static_cast<int>(output.result.target_path_edges.size()));
  }
  output.status.converged = !pruned_by_limit;
  output.status.stopped_on_target = false;
  output.status.paths_certified = !pruned_by_limit;
  output.status.all_targets_reached = output.result.target_reached;
  output.status.hit_iteration_limit = pruned_by_limit;
  output.status.iterations = output.result.iterations_used;
  if (pruned_by_limit && all_identity && output.result.target_reached) {
    output.status.hit_iteration_limit = false;
    output.status.stopped_on_target = true;
    output.status.paths_certified = true;
    output.result.stopped_on_target = true;
  }
  output.status.bounded_search_miss =
      bounds.enabled && output.status.converged &&
      !output.status.all_targets_reached;
  return output;
}

}  // namespace bf12_cpu_stub

struct BellmanFord12CsrGraph::Impl {
  HostCsrF32 graph;
  std::vector<std::int32_t> x;
  std::vector<std::int32_t> y;
  std::vector<float> base_cost;
};

namespace bf12_cpu_stub {

inline std::shared_ptr<const BellmanFord12CsrGraph::Impl> make_graph(
    const HostCsrF32& graph,
    const std::vector<std::int32_t>& x,
    const std::vector<std::int32_t>& y,
    const std::vector<float>& base_cost) {
  if (graph.rows <= 0 || graph.rows != graph.cols || graph.nnz < 0 ||
      graph.rowptr.size() != static_cast<std::size_t>(graph.rows) + 1 ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("BF12 CPU stub CSR is malformed");
  }
  for (minplus_sparse::Offset row = 0; row < graph.rows; ++row) {
    const minplus_sparse::Offset begin =
        graph.rowptr[static_cast<std::size_t>(row)];
    const minplus_sparse::Offset end =
        graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::invalid_argument(
          "BF12 CPU stub CSR row offsets are malformed");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (!valid_node(graph.colind[edge], graph) ||
        !std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::invalid_argument(
          "BF12 CPU stub CSR edge data is malformed");
    }
  }
  if (x.size() != static_cast<std::size_t>(graph.rows) ||
      y.size() != static_cast<std::size_t>(graph.rows) ||
      base_cost.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("BF12 CPU stub sidecars must contain V entries");
  }
  for (std::size_t node = 0; node < x.size(); ++node) {
    const bool missing_x =
        x[node] == routing::interchange::kMissingRouteCoordinate;
    const bool missing_y =
        y[node] == routing::interchange::kMissingRouteCoordinate;
    if (missing_x != missing_y || (!missing_x && (x[node] < 0 || y[node] < 0)) ||
        !std::isfinite(base_cost[node]) || !(base_cost[node] > 0.0f)) {
      throw std::invalid_argument("BF12 CPU stub sidecar data is malformed");
    }
  }
  auto result = std::make_shared<BellmanFord12CsrGraph::Impl>();
  result->graph = graph;
  result->x = x;
  result->y = y;
  result->base_cost = base_cost;
  ++stats.graph_constructions;
  return result;
}

}  // namespace bf12_cpu_stub

BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t)
    : impl_(bf12_cpu_stub::make_graph(
          adjacency, sidecars.route_end_x, sidecars.route_end_y,
          sidecars.base_vertex_cost)) {}

BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    const HostCsrF32& adjacency,
    const BellmanFord12NodeSidecars& sidecars,
    hipStream_t)
    : impl_(bf12_cpu_stub::make_graph(
          adjacency, sidecars.route_end_x, sidecars.route_end_y,
          sidecars.base_vertex_costs)) {}

BellmanFord12CsrGraph::~BellmanFord12CsrGraph() = default;
BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    BellmanFord12CsrGraph&&) noexcept = default;
BellmanFord12CsrGraph& BellmanFord12CsrGraph::operator=(
    BellmanFord12CsrGraph&&) noexcept = default;
std::size_t BellmanFord12CsrGraph::vertex_count() const noexcept {
  return impl_ ? static_cast<std::size_t>(impl_->graph.rows) : 0;
}
std::size_t BellmanFord12CsrGraph::edge_count() const noexcept {
  return impl_ ? static_cast<std::size_t>(impl_->graph.nnz) : 0;
}
std::size_t BellmanFord12CsrGraph::device_memory_bytes() const noexcept {
  return impl_ ? bf12_worker_policy::shared_graph_device_bytes(
                     vertex_count(), edge_count())
               : 0;
}

struct BellmanFord12CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord12CsrGraph::Impl> graph;
  std::size_t maximum_batch = 0;
  hipStream_t stream = nullptr;
  std::vector<float> dynamic_cost;
  BellmanFord12BatchTelemetry telemetry;
};

namespace bf12_cpu_stub {

inline std::unique_ptr<BellmanFord12CsrWorkspace::Impl> make_workspace(
    std::shared_ptr<const BellmanFord12CsrGraph::Impl> graph,
    std::size_t maximum_batch,
    hipStream_t stream) {
  auto result = std::make_unique<BellmanFord12CsrWorkspace::Impl>();
  result->graph = std::move(graph);
  result->maximum_batch = maximum_batch;
  result->stream = stream;
  result->dynamic_cost.assign(
      static_cast<std::size_t>(result->graph->graph.rows), 1.0f);
  ++stats.workspace_constructions;
  return result;
}

inline BellmanFord12BatchResult execute_batch(
    BellmanFord12CsrWorkspace::Impl* workspace,
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options) {
  if (descriptors.empty()) {
    throw std::invalid_argument("BF12 CPU stub batch is empty");
  }
  if (workspace->maximum_batch != 0 &&
      descriptors.size() > workspace->maximum_batch) {
    throw std::invalid_argument("BF12 CPU stub batch exceeds workspace maximum");
  }
  stats.initial_batch_sizes.push_back(descriptors.size());
  stats.initial_original_indices.emplace_back();
  BellmanFord12BatchResult output;
  output.query_results.reserve(descriptors.size());
  output.query_statuses.reserve(descriptors.size());
  output.original_query_indices.reserve(descriptors.size());
  std::vector<std::size_t> retry_positions;
  for (std::size_t query = 0; query < descriptors.size(); ++query) {
    const BellmanFord12QueryDescriptor& descriptor = descriptors[query];
    const std::size_t source_end =
        static_cast<std::size_t>(descriptor.source_begin) +
        descriptor.source_count;
    const std::size_t target_end =
        static_cast<std::size_t>(descriptor.target_begin) +
        descriptor.target_count;
    if (descriptor.source_count == 0 || descriptor.target_count == 0 ||
        source_end > flattened_sources.size() ||
        target_end > flattened_targets.size()) {
      throw std::invalid_argument("BF12 CPU stub descriptor range is invalid");
    }
    std::vector<int> sources(
        flattened_sources.begin() + descriptor.source_begin,
        flattened_sources.begin() + source_end);
    std::vector<int> targets(
        flattened_targets.begin() + descriptor.target_begin,
        flattened_targets.begin() + target_end);
    BellmanFord12BoundingBox bounds = descriptor.bounds;
    bounds.enabled = bounds.enabled && options.enable_bounding_boxes;
    CpuQuery result = run_query(
        workspace->graph->graph, workspace->graph->x, workspace->graph->y,
        workspace->graph->base_cost, workspace->dynamic_cost, sources,
        targets, bounds, descriptor.max_iterations);
    if (result.status.bounded_search_miss &&
        options.enable_unbounded_retry) {
      retry_positions.push_back(query);
    }
    output.query_results.push_back(std::move(result.result));
    output.query_statuses.push_back(result.status);
    output.original_query_indices.push_back(descriptor.original_query_index);
    stats.initial_original_indices.back().push_back(
        descriptor.original_query_index);
  }
  for (const std::size_t query : retry_positions) {
    const BellmanFord12QueryDescriptor& descriptor = descriptors[query];
    const std::size_t source_end = descriptor.source_begin +
                                   descriptor.source_count;
    const std::size_t target_end = descriptor.target_begin +
                                   descriptor.target_count;
    std::vector<int> sources(flattened_sources.begin() + descriptor.source_begin,
                             flattened_sources.begin() + source_end);
    std::vector<int> targets(flattened_targets.begin() + descriptor.target_begin,
                             flattened_targets.begin() + target_end);
    CpuQuery retry = run_query(
        workspace->graph->graph, workspace->graph->x, workspace->graph->y,
        workspace->graph->base_cost, workspace->dynamic_cost, sources,
        targets, {}, descriptor.max_iterations);
    retry.status.bounded_search_miss = false;
    output.query_results[query] = std::move(retry.result);
    output.query_statuses[query] = retry.status;
    stats.retried_original_indices.push_back(descriptor.original_query_index);
  }

  BellmanFord12BatchTelemetry local;
  local.enabled = options.enable_telemetry;
  local.requested_batch_size = options.requested_batch_size;
  local.selected_batch_size = descriptors.size();
  local.submitted_query_count = descriptors.size();
  local.batch_count = 1 + retry_positions.size();
  local.batch_fill_ratio = 1.0;
  local.controller_mode_used = options.controller_mode ==
                                       BellmanFord12ControllerMode::Auto
                                   ? (workspace->stream == nullptr &&
                                              cooperative_supported
                                          ? BellmanFord12ControllerMode::
                                                CooperativeBatch
                                          : BellmanFord12ControllerMode::
                                                HostBatch)
                                   : options.controller_mode;
  local.cooperative_controller_launch_count =
      local.controller_mode_used ==
              BellmanFord12ControllerMode::CooperativeBatch
          ? local.batch_count
          : 0;
  local.bounded_query_retry_count = retry_positions.size();
  std::size_t nodes_used = 0;
  std::size_t edges_used = 0;
  for (const BellmanFordCsrResult& result : output.query_results) {
    nodes_used += result.target_path_nodes.size();
    edges_used += result.target_path_edges.size();
  }
  local.result_nodes_used = nodes_used;
  local.result_edges_used = edges_used;
  local.result_node_capacity =
      options.result_node_capacity == 0 ? nodes_used
                                        : options.result_node_capacity;
  local.result_edge_capacity =
      options.result_edge_capacity == 0 ? edges_used
                                        : options.result_edge_capacity;
  if ((options.result_node_capacity != 0 &&
       nodes_used > options.result_node_capacity) ||
      (options.result_edge_capacity != 0 &&
       edges_used > options.result_edge_capacity)) {
    local.arena_overflow_retry_count = 1;
  }
  workspace->telemetry.enabled = workspace->telemetry.enabled || local.enabled;
  workspace->telemetry.requested_batch_size = local.requested_batch_size;
  workspace->telemetry.selected_batch_size = local.selected_batch_size;
  workspace->telemetry.submitted_query_count += local.submitted_query_count;
  workspace->telemetry.batch_count += local.batch_count;
  workspace->telemetry.controller_mode_used = local.controller_mode_used;
  workspace->telemetry.cooperative_controller_launch_count +=
      local.cooperative_controller_launch_count;
  workspace->telemetry.bounded_query_retry_count +=
      local.bounded_query_retry_count;
  workspace->telemetry.arena_overflow_retry_count +=
      local.arena_overflow_retry_count;
  workspace->telemetry.result_nodes_used = local.result_nodes_used;
  workspace->telemetry.result_edges_used = local.result_edges_used;
  return output;
}

}  // namespace bf12_cpu_stub

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    std::size_t maximum_batch_size,
    hipStream_t stream)
    : impl_(bf12_cpu_stub::make_workspace(
          bf12_cpu_stub::make_graph(adjacency, sidecars.route_end_x,
                                    sidecars.route_end_y,
                                    sidecars.base_vertex_cost),
          maximum_batch_size, stream)) {}

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord12NodeSidecars& sidecars,
    std::size_t maximum_batch_size,
    hipStream_t stream)
    : impl_(bf12_cpu_stub::make_workspace(
          bf12_cpu_stub::make_graph(adjacency, sidecars.route_end_x,
                                    sidecars.route_end_y,
                                    sidecars.base_vertex_costs),
          maximum_batch_size, stream)) {}

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    std::shared_ptr<const BellmanFord12CsrGraph> adjacency,
    std::size_t maximum_batch_size,
    hipStream_t stream) {
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument("BF12 CPU stub shared graph is null");
  }
  impl_ = bf12_cpu_stub::make_workspace(adjacency->impl_, maximum_batch_size,
                                        stream);
}

BellmanFord12CsrWorkspace::~BellmanFord12CsrWorkspace() = default;
BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    BellmanFord12CsrWorkspace&&) noexcept = default;
BellmanFord12CsrWorkspace& BellmanFord12CsrWorkspace::operator=(
    BellmanFord12CsrWorkspace&&) noexcept = default;

void BellmanFord12CsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  if (!impl_ || stream != impl_->stream ||
      vertex_costs.size() != impl_->dynamic_cost.size()) {
    throw std::invalid_argument("BF12 CPU stub dynamic-cost update is invalid");
  }
  for (const float cost : vertex_costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument("BF12 CPU stub dynamic cost is invalid");
    }
  }
  impl_->dynamic_cost = vertex_costs;
  ++bf12_cpu_stub::stats.dynamic_cost_uploads;
}

BellmanFord12BatchResult BellmanFord12CsrWorkspace::run_batch(
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options,
    hipStream_t stream) {
  if (!impl_ || stream != impl_->stream) {
    throw std::invalid_argument("BF12 CPU stub run stream changed");
  }
  return bf12_cpu_stub::execute_batch(
      impl_.get(), flattened_sources, flattened_targets, descriptors, options);
}

BellmanFord12BatchResult BellmanFord12CsrWorkspace::run_batch(
    const std::vector<float>& dynamic_vertex_costs,
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options,
    hipStream_t stream) {
  update_vertex_costs(dynamic_vertex_costs, stream);
  return run_batch(flattened_sources, flattened_targets, descriptors, options,
                   stream);
}

BellmanFord12CooperativeCapability
BellmanFord12CsrWorkspace::cooperative_capability(
    bool require_occupancy) const {
  if (require_occupancy) ++bf12_cpu_stub::stats.cooperative_occupancy_queries;
  BellmanFord12CooperativeCapability result;
  result.device_supports_cooperative_launch =
      bf12_cpu_stub::cooperative_supported;
  result.default_stream_required = true;
  result.device = 0;
  result.compute_unit_count = bf12_cpu_stub::compute_unit_count;
  result.legally_resident_blocks =
      require_occupancy && bf12_cpu_stub::cooperative_supported ? 1 : 0;
  result.architecture = bf12_cpu_stub::architecture;
  result.reason = !bf12_cpu_stub::cooperative_supported
                      ? "CPU stub cooperative mode disabled"
                  : require_occupancy
                      ? "CPU stub cooperative mode"
                      : "CPU stub skipped cooperative occupancy";
  return result;
}

bool BellmanFord12CsrWorkspace::cooperative_execution_supported() const {
  return impl_ && impl_->stream == nullptr &&
         bf12_cpu_stub::cooperative_supported;
}

BellmanFord12MemoryEstimate BellmanFord12CsrWorkspace::estimate_memory(
    std::size_t batch_size,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity) const {
  return estimate_bellman_ford12_memory(
      static_cast<std::size_t>(impl_->graph->graph.rows),
      static_cast<std::size_t>(impl_->graph->graph.nnz), batch_size, 0, 0,
      result_node_capacity, result_edge_capacity, false);
}

std::size_t BellmanFord12CsrWorkspace::maximum_batch_size() const noexcept {
  if (!impl_) return 0;
  if (impl_->maximum_batch != 0) return impl_->maximum_batch;
  return bf12_worker_policy::maximum_uint32_composite_batch_size(
      static_cast<std::size_t>(impl_->graph->graph.rows));
}

BellmanFord12BatchTelemetry BellmanFord12CsrWorkspace::telemetry() const {
  if (!impl_) throw std::runtime_error("BF12 CPU stub workspace is moved-from");
  return impl_->telemetry;
}

void BellmanFord12CsrWorkspace::reset_telemetry() {
  if (!impl_) throw std::runtime_error("BF12 CPU stub workspace is moved-from");
  impl_->telemetry = {};
}

BellmanFord12MemoryEstimate estimate_bellman_ford12_memory(
    std::size_t vertex_count,
    std::size_t edge_count,
    std::size_t batch_size,
    std::size_t source_count,
    std::size_t target_count,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity,
    bool telemetry_enabled) {
  BellmanFord12MemoryEstimate result;
  result.graph_bytes =
      bf12_worker_policy::shared_graph_device_bytes(vertex_count, edge_count);
  result.shared_dynamic_cost_bytes =
      bf12_worker_policy::shared_dynamic_cost_device_bytes(vertex_count);
  result.fixed_query_state_bytes =
      bf12_worker_policy::query_vertex_state_device_bytes(batch_size,
                                                           vertex_count);
  result.result_arena_bytes = bf12_worker_policy::result_device_bytes_estimate(
      batch_size, target_count, result_node_capacity, result_edge_capacity);
  BellmanFord12WorkerPolicyInputs inputs;
  inputs.vertex_count = vertex_count;
  inputs.edge_count = edge_count;
  inputs.source_count = source_count;
  inputs.target_count = target_count;
  inputs.telemetry_enabled = telemetry_enabled;
  inputs.graph_already_resident = true;
  const std::size_t fixed =
      bf12_worker_policy::fixed_device_bytes_estimate(inputs, batch_size);
  result.control_bytes =
      fixed - result.shared_dynamic_cost_bytes - result.fixed_query_state_bytes;
  result.total_bytes = result.graph_bytes + fixed + result.result_arena_bytes;
  return result;
}

const char* bellman_ford12_controller_mode_name(
    BellmanFord12ControllerMode mode) noexcept {
  switch (mode) {
    case BellmanFord12ControllerMode::Auto: return "auto";
    case BellmanFord12ControllerMode::HostBatch: return "host-batch";
    case BellmanFord12ControllerMode::CooperativeBatch:
      return "cooperative-batch";
  }
  return "unknown";
}
