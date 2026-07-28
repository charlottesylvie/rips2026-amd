#define ROUTING_PATHFINDER_NO_MAIN
#define __HIP_PLATFORM_AMD__ 1

#include "../pathfinder.cpp"

#include <fstream>
#include <queue>

namespace {

std::atomic<int> g_multisource_delta_calls{0};
std::atomic<int> g_delta_graph_uploads{0};
std::atomic<int> g_bellman_ford_calls{0};
std::atomic<int> g_bellman_ford_graph_uploads{0};
std::atomic<int> g_bellman_ford_workspace_constructions{0};
std::atomic<int> g_unit_bfs_calls{0};
std::atomic<int> g_unit_bfs_graph_uploads{0};

struct CpuSsspResult {
  std::vector<float> dist;
  std::vector<int> pred_node;
  std::vector<minplus_sparse::Offset> pred_edge;
};

CpuSsspResult cpu_dijkstra_outgoing_csr_multi(const HostCsrF32& graph,
                                              const std::vector<int>& sources,
                                              const std::vector<DeltaSteppingCsrNodeBounds>* node_bounds = nullptr,
                                              DeltaSteppingCsrRunOptions::RouteWindow route_window = {},
                                              std::uint64_t* rejected_edges = nullptr,
                                              float exclusive_distance_limit =
                                                  std::numeric_limits<float>::infinity()) {
  struct OutEdge {
    int to = -1;
    float weight = 0.0f;
    minplus_sparse::Offset edge = -1;
  };

  std::vector<std::vector<OutEdge>> outgoing(
      static_cast<std::size_t>(graph.rows));
  for (int src = 0; src < graph.rows; ++src) {
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(src)];
         edge < graph.rowptr[static_cast<std::size_t>(src + 1)];
         ++edge) {
      const int dst = graph.colind[static_cast<std::size_t>(edge)];
      const float weight = graph.values[static_cast<std::size_t>(edge)];
      outgoing[static_cast<std::size_t>(src)].push_back({dst, weight, edge});
    }
  }

  constexpr float inf = std::numeric_limits<float>::infinity();
  CpuSsspResult result;
  result.dist.assign(static_cast<std::size_t>(graph.rows), inf);
  result.pred_node.assign(static_cast<std::size_t>(graph.rows), -1);
  result.pred_edge.assign(static_cast<std::size_t>(graph.rows), -1);
  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;

  if (0.0f < exclusive_distance_limit) {
    for (const int source : sources) {
      result.dist[static_cast<std::size_t>(source)] = 0.0f;
      queue.push({0.0f, source});
    }
  }
  while (!queue.empty()) {
    const auto [du, u] = queue.top();
    queue.pop();
    if (du != result.dist[static_cast<std::size_t>(u)]) {
      continue;
    }
    for (const OutEdge& edge : outgoing[static_cast<std::size_t>(u)]) {
      if (route_window.enabled && node_bounds != nullptr) {
        const DeltaSteppingCsrNodeBounds& bounds =
            (*node_bounds)[static_cast<std::size_t>(edge.to)];
        const bool intersects = bounds.valid == 0 ||
            (bounds.max_x >= route_window.min_x &&
             bounds.min_x <= route_window.max_x &&
             bounds.max_y >= route_window.min_y &&
             bounds.min_y <= route_window.max_y);
        if (!intersects) {
          if (rejected_edges != nullptr) ++*rejected_edges;
          continue;
        }
      }
      const float candidate = du + edge.weight;
      if (!(candidate < exclusive_distance_limit)) {
        continue;
      }
      if (candidate < result.dist[static_cast<std::size_t>(edge.to)]) {
        result.dist[static_cast<std::size_t>(edge.to)] = candidate;
        result.pred_node[static_cast<std::size_t>(edge.to)] = u;
        result.pred_edge[static_cast<std::size_t>(edge.to)] = edge.edge;
        queue.push({candidate, edge.to});
      }
    }
  }
  return result;
}

std::vector<float> cpu_dijkstra_outgoing_csr(const HostCsrF32& graph,
                                             int source) {
  return cpu_dijkstra_outgoing_csr_multi(graph, std::vector<int>{source}).dist;
}

void fill_compact_target_paths(const HostCsrF32& graph,
                               const std::vector<int>& sources,
                               const std::vector<int>& targets,
                               const CpuSsspResult& cpu_result,
                               BellmanFordCsrResult& result) {
  result.target_distances.resize(targets.size(), std::numeric_limits<float>::infinity());
  result.target_sources.resize(targets.size(), -1);
  result.target_path_offsets.assign(targets.size() + 1, 0);
  result.target_edge_offsets.assign(targets.size() + 1, 0);
  result.target_path_nodes.clear();
  result.target_path_edges.clear();
  result.target_reached = true;

  for (std::size_t i = 0; i < targets.size(); ++i) {
    const int target = targets[i];
    result.target_path_offsets[i] =
        static_cast<int>(result.target_path_nodes.size());
    result.target_edge_offsets[i] =
        static_cast<int>(result.target_path_edges.size());
    if (target < 0 ||
        static_cast<std::size_t>(target) >= cpu_result.dist.size() ||
        !std::isfinite(cpu_result.dist[static_cast<std::size_t>(target)])) {
      result.target_reached = false;
      continue;
    }

    result.target_distances[i] = cpu_result.dist[static_cast<std::size_t>(target)];
    std::vector<int> reversed_nodes;
    std::vector<minplus_sparse::Offset> reversed_edges;
    int current = target;
    for (int guard = 0; guard < graph.rows; ++guard) {
      reversed_nodes.push_back(current);
      if (std::find(sources.begin(), sources.end(), current) != sources.end()) {
        result.target_sources[i] = current;
        break;
      }
      const int pred = cpu_result.pred_node[static_cast<std::size_t>(current)];
      if (pred < 0) {
        break;
      }
      reversed_edges.push_back(cpu_result.pred_edge[static_cast<std::size_t>(current)]);
      current = pred;
    }
    if (result.target_sources[i] < 0) {
      result.target_reached = false;
      continue;
    }
    std::reverse(reversed_nodes.begin(), reversed_nodes.end());
    std::reverse(reversed_edges.begin(), reversed_edges.end());
    result.target_path_nodes.insert(result.target_path_nodes.end(),
                                    reversed_nodes.begin(),
                                    reversed_nodes.end());
    result.target_path_edges.insert(result.target_path_edges.end(),
                                    reversed_edges.begin(),
                                    reversed_edges.end());
  }
  result.target_path_offsets[targets.size()] =
      static_cast<int>(result.target_path_nodes.size());
  result.target_edge_offsets[targets.size()] =
      static_cast<int>(result.target_path_edges.size());
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void add_default_node_metadata(routing::RoutingMetadata& metadata) {
  const std::size_t node_count = metadata.node_device_ids.size();
  metadata.node_min_x.assign(node_count, 0);
  metadata.node_max_x.assign(node_count, 0);
  metadata.node_min_y.assign(node_count, 0);
  metadata.node_max_y.assign(node_count, 0);
  metadata.node_tile_type_strings.assign(node_count, routing::kNoIndex);
  metadata.node_wire_type_strings.assign(node_count, routing::kNoIndex);
}

HostCsrF32 make_tree_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 4;
  graph.rowptr = {0, 2, 4, 4, 4};
  graph.colind = {1, 3, 2, 3};
  graph.values = {1.0f, 10.0f, 1.0f, 1.0f};
  return graph;
}

HostCsrF32 make_self_loop_predecessor_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 2;
  graph.rowptr = {0, 0, 0, 1, 2};
  graph.colind = {2, 2};
  graph.values = {0.0f, 1.0f};
  return graph;
}

HostCsrF32 make_two_net_congestion_graph() {
  HostCsrF32 graph;
  graph.rows = 6;
  graph.cols = 6;
  graph.nnz = 6;
  graph.rowptr = {0, 1, 3, 5, 6, 6, 6};
  graph.colind = {2, 2, 3, 4, 5, 5};
  graph.values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  return graph;
}

HostCsrF32 make_three_net_overlap_graph() {
  HostCsrF32 graph;
  graph.rows = 8;
  graph.cols = 8;
  graph.nnz = 5;
  graph.rowptr = {0, 1, 2, 4, 4, 4, 4, 5, 5};
  graph.colind = {2, 2, 4, 5, 7};
  graph.values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  return graph;
}

routing::RoutingMetadata make_metadata() {
  routing::RoutingMetadata metadata;
  metadata.strings = {"net0",      "SRC_SITE", "SRC_PIN", "SINK_SITE_0",
                      "SINK_PIN_0", "SINK_SITE_1", "SINK_PIN_1",
                      "TILE_A",    "WIRE_0",   "WIRE_1", "WIRE_2",
                      "WIRE_3"};
  metadata.node_device_ids = {0, 1, 2, 3};
  add_default_node_metadata(metadata);
  metadata.edge_attrs = {
      {7, 0},
      {7, 1},
      {7, 2},
      {7, 3},
  };
  metadata.pip_data = {
      {8, 9, true},
      {9, 10, true},
      {9, 11, true},
      {8, 11, true},
  };

  routing::RouteRequest request;
  request.net_string = 0;
  request.sources.push_back({0, 1, 2});
  request.sinks.push_back({2, 3, 4});
  request.sinks.push_back({3, 5, 6});
  metadata.route_requests.push_back(std::move(request));
  return metadata;
}

routing::RoutingMetadata make_three_net_overlap_metadata(const HostCsrF32& graph) {
  routing::RoutingMetadata metadata;
  metadata.strings = {"net_a", "net_b", "net_c", "SRC_SITE_A", "SRC_SITE_B",
                      "SRC_SITE_C", "SRC_PIN", "SINK_SITE_A", "SINK_SITE_B",
                      "SINK_SITE_C", "SINK_PIN", "TILE_A", "WIRE_0", "WIRE_1"};
  metadata.node_device_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  add_default_node_metadata(metadata);
  metadata.edge_attrs.assign(static_cast<std::size_t>(graph.nnz), {11, 0});
  metadata.pip_data = {{12, 13, true}};

  routing::RouteRequest net_a;
  net_a.net_string = 0;
  net_a.sources.push_back({0, 3, 6});
  net_a.sinks.push_back({4, 7, 10});
  metadata.route_requests.push_back(std::move(net_a));

  routing::RouteRequest net_b;
  net_b.net_string = 1;
  net_b.sources.push_back({1, 4, 6});
  net_b.sinks.push_back({5, 8, 10});
  metadata.route_requests.push_back(std::move(net_b));

  routing::RouteRequest net_c;
  net_c.net_string = 2;
  net_c.sources.push_back({6, 5, 6});
  net_c.sinks.push_back({7, 9, 10});
  metadata.route_requests.push_back(std::move(net_c));

  return metadata;
}

routing::RoutingMetadata make_two_net_metadata(const HostCsrF32& graph) {
  routing::RoutingMetadata metadata;
  metadata.strings = {"net_a", "net_b", "SRC_SITE_A", "SRC_SITE_B",
                      "SRC_PIN", "SINK_SITE_A", "SINK_SITE_B", "SINK_PIN",
                      "TILE_A", "WIRE_0", "WIRE_1"};
  metadata.node_device_ids = {0, 1, 2, 3, 4, 5};
  add_default_node_metadata(metadata);
  metadata.edge_attrs.assign(static_cast<std::size_t>(graph.nnz), {8, 0});
  metadata.pip_data = {{9, 10, true}};

  routing::RouteRequest net_a;
  net_a.net_string = 0;
  net_a.sources.push_back({0, 2, 4});
  net_a.sinks.push_back({4, 5, 7});
  metadata.route_requests.push_back(std::move(net_a));

  routing::RouteRequest net_b;
  net_b.net_string = 1;
  net_b.sources.push_back({1, 3, 4});
  net_b.sinks.push_back({5, 6, 7});
  metadata.route_requests.push_back(std::move(net_b));
  return metadata;
}

routing::RoutingMetadata make_window_metadata(
    const HostCsrF32& graph,
    const std::vector<DeltaSteppingCsrNodeBounds>& bounds,
    int source,
    int sink) {
  if (bounds.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("test node bounds do not match graph rows");
  }
  routing::RoutingMetadata metadata;
  metadata.strings = {"window_net"};
  metadata.node_device_ids.resize(static_cast<std::size_t>(graph.rows));
  for (std::size_t i = 0; i < metadata.node_device_ids.size(); ++i) {
    metadata.node_device_ids[i] = i;
  }
  add_default_node_metadata(metadata);
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    if (bounds[i].valid == 0) {
      metadata.node_min_x[i] = -1;
      metadata.node_max_x[i] = -1;
      metadata.node_min_y[i] = -1;
      metadata.node_max_y[i] = -1;
    } else {
      metadata.node_min_x[i] = bounds[i].min_x;
      metadata.node_max_x[i] = bounds[i].max_x;
      metadata.node_min_y[i] = bounds[i].min_y;
      metadata.node_max_y[i] = bounds[i].max_y;
    }
  }
  metadata.edge_attrs.assign(static_cast<std::size_t>(graph.nnz), {});
  routing::RouteRequest request;
  request.net_string = 0;
  request.sources.push_back({source, 0, 0});
  request.sinks.push_back({sink, 0, 0});
  metadata.route_requests.push_back(std::move(request));
  return metadata;
}

routing::RoutingMetadata make_window_metadata_with_sinks(
    const HostCsrF32& graph,
    const std::vector<DeltaSteppingCsrNodeBounds>& bounds,
    int source,
    const std::vector<int>& sinks) {
  routing::RoutingMetadata metadata =
      make_window_metadata(graph, bounds, source, sinks.front());
  routing::RouteRequest& request = metadata.route_requests.front();
  request.sinks.clear();
  for (const int sink : sinks) {
    request.sinks.push_back({sink, 0, 0});
  }
  return metadata;
}

}  // namespace

struct BellmanFord10CsrGraph::Impl {
  explicit Impl(const HostCsrF32& adjacency) : graph(adjacency) {}

  HostCsrF32 graph;
};

BellmanFord10CsrGraph::BellmanFord10CsrGraph(const HostCsrF32& adjacency,
                                             hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency)) {
  (void)stream;
  ++g_bellman_ford_graph_uploads;
}

BellmanFord10CsrGraph::~BellmanFord10CsrGraph() = default;
BellmanFord10CsrGraph::BellmanFord10CsrGraph(BellmanFord10CsrGraph&&) noexcept = default;
BellmanFord10CsrGraph& BellmanFord10CsrGraph::operator=(
    BellmanFord10CsrGraph&&) noexcept = default;

struct BellmanFord10CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord10CsrGraph::Impl> graph;
};

BellmanFord10CsrWorkspace::BellmanFord10CsrWorkspace(const HostCsrF32& adjacency,
                                                     hipStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  ++g_bellman_ford_workspace_constructions;
  ++g_bellman_ford_graph_uploads;
  impl_->graph = std::make_shared<BellmanFord10CsrGraph::Impl>(adjacency);
}

BellmanFord10CsrWorkspace::BellmanFord10CsrWorkspace(
    std::shared_ptr<const BellmanFord10CsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  ++g_bellman_ford_workspace_constructions;
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument("Bellman-Ford shared graph must not be null");
  }
  impl_->graph = adjacency->impl_;
}

BellmanFord10CsrWorkspace::~BellmanFord10CsrWorkspace() = default;
BellmanFord10CsrWorkspace::BellmanFord10CsrWorkspace(
    BellmanFord10CsrWorkspace&&) noexcept = default;
BellmanFord10CsrWorkspace& BellmanFord10CsrWorkspace::operator=(
    BellmanFord10CsrWorkspace&&) noexcept = default;

BellmanFordCsrResult BellmanFord10CsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)delta;
  (void)max_iters;
  (void)stream;
  (void)progress_callback;
  (void)progress_user_data;
  ++g_bellman_ford_calls;

  BellmanFordCsrResult result;
  const HostCsrF32& graph = impl_->graph->graph;
  const CpuSsspResult cpu_result =
      cpu_dijkstra_outgoing_csr_multi(graph, sources);
  fill_compact_target_paths(graph, sources, targets, cpu_result, result);
  result.target = -1;
  result.iterations_used = 1;
  result.converged = true;
  result.stopped_on_target = result.target_reached;
  return result;
}

BellmanFordCsrResult BellmanFord10CsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  BellmanFordCsrResult result = run(sources,
                                    std::vector<int>{target},
                                    delta,
                                    max_iters,
                                    stream,
                                    progress_callback,
                                    progress_user_data);
  result.target = target;
  result.target_distance = result.target_distances.front();
  result.target_reached = std::isfinite(result.target_distance);
  return result;
}

BellmanFordCsrResult BellmanFord10CsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}

struct DeltaSteppingCsrGraph::Impl {
  explicit Impl(const HostCsrF32& adjacency,
                std::vector<DeltaSteppingCsrNodeBounds> bounds = {})
      : graph(adjacency), node_bounds(std::move(bounds)) {}

  HostCsrF32 graph;
  std::vector<DeltaSteppingCsrNodeBounds> node_bounds;
};

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                                             hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency)) {
  (void)stream;
  ++g_delta_graph_uploads;
}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    const std::vector<DeltaSteppingCsrNodeBounds>& node_bounds,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency, node_bounds)) {
  (void)stream;
  if (node_bounds.size() != static_cast<std::size_t>(adjacency.rows)) {
    throw std::invalid_argument("node bounds size does not match CSR rows");
  }
  for (const DeltaSteppingCsrNodeBounds& bounds : node_bounds) {
    if (bounds.valid != 0 &&
        (bounds.min_x > bounds.max_x || bounds.min_y > bounds.max_y)) {
      throw std::invalid_argument("node bounds have an invalid range");
    }
  }
  ++g_delta_graph_uploads;
}

DeltaSteppingCsrGraph::~DeltaSteppingCsrGraph() = default;
DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    DeltaSteppingCsrGraph&&) noexcept = default;
DeltaSteppingCsrGraph& DeltaSteppingCsrGraph::operator=(
    DeltaSteppingCsrGraph&&) noexcept = default;

struct DeltaSteppingCsrWorkspace::Impl {
  HostCsrF32 graph;
  std::vector<float> base_values;
  std::vector<DeltaSteppingCsrNodeBounds> node_bounds;
};

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                                                     hipStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  impl_->graph = adjacency;
  impl_->base_values = adjacency.values;
}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument(
        "delta-stepping shared graph must not be null");
  }
  impl_->graph = adjacency->impl_->graph;
  impl_->base_values = impl_->graph.values;
  impl_->node_bounds = adjacency->impl_->node_bounds;
}

DeltaSteppingCsrWorkspace::~DeltaSteppingCsrWorkspace() = default;
DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    DeltaSteppingCsrWorkspace&&) noexcept = default;
DeltaSteppingCsrWorkspace& DeltaSteppingCsrWorkspace::operator=(
    DeltaSteppingCsrWorkspace&&) noexcept = default;

void DeltaSteppingCsrWorkspace::update_values(const std::vector<float>& values,
                                              hipStream_t stream) {
  (void)stream;
  impl_->graph.values = values;
  impl_->base_values = values;
}

void DeltaSteppingCsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  (void)stream;
  impl_->graph.values.resize(impl_->base_values.size());
  for (int src = 0; src < impl_->graph.rows; ++src) {
    for (minplus_sparse::Offset edge = impl_->graph.rowptr[static_cast<std::size_t>(src)];
         edge < impl_->graph.rowptr[static_cast<std::size_t>(src + 1)];
         ++edge) {
      const int dst = impl_->graph.colind[static_cast<std::size_t>(edge)];
      const float node_cost = vertex_costs[static_cast<std::size_t>(dst)];
      impl_->graph.values[static_cast<std::size_t>(edge)] =
          impl_->base_values[static_cast<std::size_t>(edge)] * node_cost;
    }
  }
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(sources, std::vector<int>{target}, delta, max_iters, stream,
             progress_callback, progress_user_data);
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)delta;
  (void)max_iters;
  (void)stream;
  (void)progress_callback;
  (void)progress_user_data;
  if (active_route_window_.enabled && impl_->node_bounds.empty()) {
    throw std::logic_error(
        "route-window query requires uploaded immutable node bounds");
  }
  ++g_multisource_delta_calls;
  std::uint64_t rejected_edges = 0;
  const std::vector<DeltaSteppingCsrNodeBounds>* active_bounds =
      impl_->node_bounds.empty() ? nullptr : &impl_->node_bounds;
  CpuSsspResult cpu_result = cpu_dijkstra_outgoing_csr_multi(
      impl_->graph, sources, active_bounds, active_route_window_,
      &rejected_edges, active_distance_limit_);
  DeltaSteppingCsrResult result;
  result.dist = cpu_result.dist;
  result.pred_node = cpu_result.pred_node;
  result.pred_edge = cpu_result.pred_edge;
  result.iterations_used = 1;
  result.converged = true;
  result.stopped_on_target = true;
  fill_compact_target_paths(impl_->graph, sources, targets, cpu_result, result);
  if (active_telemetry_ != nullptr) {
    active_telemetry_->collected = true;
    active_telemetry_->completed = true;
    active_telemetry_->execution_path =
        std::all_of(impl_->graph.values.begin(), impl_->graph.values.end(),
                    [](float value) { return value == 1.0f; })
            ? DeltaSteppingCsrExecutionPath::kExactUnit
            : DeltaSteppingCsrExecutionPath::kCompactGeneric;
    active_telemetry_->window_rejected_edges = rejected_edges;
    active_telemetry_->window_unknown_coordinate_nodes =
        static_cast<std::uint64_t>(std::count_if(
            impl_->node_bounds.begin(), impl_->node_bounds.end(),
            [](const DeltaSteppingCsrNodeBounds& bounds) {
              return bounds.valid == 0;
            }));
  }
  result.dist.clear();
  result.pred_node.clear();
  result.pred_edge.clear();
  return result;
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_minplus_hip_csr(adjacency,
                                        std::vector<int>{source},
                                        target,
                                        delta,
                                        max_iters,
                                        stream,
                                        progress_callback,
                                        progress_user_data);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)delta;
  (void)max_iters;
  (void)stream;
  (void)progress_callback;
  (void)progress_user_data;
  ++g_multisource_delta_calls;

  DeltaSteppingCsrResult result;
  result.target = target;
  CpuSsspResult cpu_result = cpu_dijkstra_outgoing_csr_multi(adjacency, sources);
  result.dist = std::move(cpu_result.dist);
  result.pred_node = std::move(cpu_result.pred_node);
  result.pred_edge = std::move(cpu_result.pred_edge);
  result.iterations_used = 1;
  if (target >= 0) {
    result.target_distance = result.dist[static_cast<std::size_t>(target)];
    result.target_reached = std::isfinite(result.target_distance);
    result.stopped_on_target = result.target_reached;
  } else {
    result.target_reached = true;
    result.stopped_on_target = true;
  }
  result.converged = true;
  return result;
}

struct UnitBfsCsrGraph::Impl {
  HostCsrF32 graph;
  bool uses_32_bit_offsets = false;
};

UnitBfsCsrGraph::UnitBfsCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream)
    : UnitBfsCsrGraph(
          adjacency, stream, UnitBfsCsrOffsetMode::kAuto) {}

UnitBfsCsrGraph::UnitBfsCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream,
                                 UnitBfsCsrOffsetMode offset_mode)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  ++g_unit_bfs_graph_uploads;
  impl_->graph = adjacency;
  switch (offset_mode) {
    case UnitBfsCsrOffsetMode::kAuto:
      impl_->uses_32_bit_offsets =
          adjacency.nnz >= 0 &&
          adjacency.nnz <= std::numeric_limits<std::int32_t>::max();
      break;
    case UnitBfsCsrOffsetMode::kForce64Bit:
      impl_->uses_32_bit_offsets = false;
      break;
    default:
      throw std::invalid_argument("unknown unit BFS offset mode");
  }
}

UnitBfsCsrGraph::~UnitBfsCsrGraph() = default;
UnitBfsCsrGraph::UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept = default;
UnitBfsCsrGraph& UnitBfsCsrGraph::operator=(UnitBfsCsrGraph&&) noexcept = default;

bool UnitBfsCsrGraph::uses_32_bit_offsets() const noexcept {
  return impl_ && impl_->uses_32_bit_offsets;
}

struct UnitBfsCsrWorkspace::Impl {
  std::shared_ptr<const UnitBfsCsrGraph> graph;
};

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : UnitBfsCsrWorkspace(
          adjacency, stream, UnitBfsCsrOffsetMode::kAuto) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream,
                                         UnitBfsCsrOffsetMode offset_mode)
    : UnitBfsCsrWorkspace(
          std::make_shared<UnitBfsCsrGraph>(adjacency, stream, offset_mode),
          stream) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    std::shared_ptr<const UnitBfsCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument("unit BFS shared graph must not be null");
  }
  impl_->graph = std::move(adjacency);
}

UnitBfsCsrWorkspace::~UnitBfsCsrWorkspace() = default;
UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept = default;
UnitBfsCsrWorkspace& UnitBfsCsrWorkspace::operator=(
    UnitBfsCsrWorkspace&&) noexcept = default;

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)delta;
  (void)max_depth;
  (void)stream;
  (void)progress_callback;
  (void)progress_user_data;
  ++g_unit_bfs_calls;

  UnitBfsCsrResult result;
  const HostCsrF32& graph = impl_->graph->impl_->graph;
  CpuSsspResult cpu_result = cpu_dijkstra_outgoing_csr_multi(graph, sources);
  fill_compact_target_paths(graph, sources, targets, cpu_result, result);
  result.target = -1;
  result.iterations_used = 1;
  result.stopped_on_target = result.target_reached;
  result.converged = true;
  return result;
}

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  UnitBfsCsrResult result = run(sources,
                                std::vector<int>{target},
                                delta,
                                max_depth,
                                stream,
                                progress_callback,
                                progress_user_data);
  result.target = target;
  if (!result.target_distances.empty()) {
    result.target_distance = result.target_distances.front();
    result.target_reached = std::isfinite(result.target_distance);
    result.stopped_on_target = result.target_reached;
  }
  return result;
}

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_depth,
             stream,
             progress_callback,
             progress_user_data);
}

int main() {
  const HostCsrF32 graph = make_tree_graph();
  const std::vector<float> dist = cpu_dijkstra_outgoing_csr(graph, 0);
  const std::vector<routing::PathEdge> path =
      routing::reconstruct_shortest_path(graph, dist, 0, 2);
  require(path.size() == 2, "expected two edges in 0->2 reconstructed path");
  require(path[0].from == 0 && path[0].to == 1, "first path edge should be 0->1");
  require(path[1].from == 1 && path[1].to == 2, "second path edge should be 1->2");

  const HostCsrF32 self_loop_graph = make_self_loop_predecessor_graph();
  const std::vector<float> self_loop_dist =
      cpu_dijkstra_outgoing_csr(self_loop_graph, 3);
  const std::vector<routing::PathEdge> self_loop_path =
      routing::reconstruct_shortest_path(self_loop_graph, self_loop_dist, 3, 2);
  require(self_loop_path.size() == 1,
          "self-loop predecessor graph should reconstruct one real edge");
  require(self_loop_path[0].from == 3 && self_loop_path[0].to == 2,
          "reconstruction should ignore no-progress self-loop predecessors");

  const HostCsrF32 congestion_graph = make_two_net_congestion_graph();
  const routing::RoutingMetadata congestion_metadata =
      make_two_net_metadata(congestion_graph);
  routing::PathfinderOptions congestion_options;
  congestion_options.delta = 1.0f;
  routing::PathfinderResult congestion_result =
      routing::run_pathfinder(congestion_graph, congestion_metadata, congestion_options, nullptr);
  require(congestion_result.routed,
          "overlapping congestion-free graph should still be considered routed");
  require(congestion_result.overused_nodes > 0,
          "overlapping congestion-free graph should keep overuse only as a diagnostic");
  require(congestion_result.nets[0].sinks[0].nodes == std::vector<int>({0, 2, 4}),
          "first net should use the shared direct path");
  require(congestion_result.nets[1].sinks[0].nodes == std::vector<int>({1, 2, 5}),
          "second net should use its shortest path even when it overlaps");
  const std::filesystem::path overlap_routes_path =
      "/tmp/congestion_free_overlap_routes.jsonl";
  routing::write_routes_jsonl(overlap_routes_path,
                              congestion_graph,
                              congestion_metadata,
                              congestion_result);
  std::ifstream overlap_routes_file(overlap_routes_path);
  const std::string overlap_routes_json(
      (std::istreambuf_iterator<char>(overlap_routes_file)),
      std::istreambuf_iterator<char>());
  require(overlap_routes_json.find("\"net\":\"net_b\"") != std::string::npos,
          "overlapping route output should still include the second net");
  require(overlap_routes_json.find("\"from\":1,\"to\":2") != std::string::npos,
          "overlapping route output should include the shared shortest path");

  auto movable_delta_graph =
      std::make_shared<DeltaSteppingCsrGraph>(congestion_graph, nullptr);
  DeltaSteppingCsrWorkspace retained_delta_workspace(
      movable_delta_graph, nullptr);
  *movable_delta_graph = DeltaSteppingCsrGraph(self_loop_graph, nullptr);
  const DeltaSteppingCsrResult retained_delta_result =
      retained_delta_workspace.run(std::vector<int>{0},
                                   std::vector<int>{4},
                                   1.0f,
                                   -1,
                                   nullptr,
                                   nullptr,
                                   nullptr);
  require(retained_delta_result.target_path_nodes ==
              std::vector<int>({0, 2, 4}),
          "shared delta workspace should retain its graph after wrapper move "
          "assignment");

  routing::PathfinderOptions parallel_options;
  parallel_options.delta = 1.0f;
  parallel_options.parallel_net_workers = 2;
  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  g_unit_bfs_graph_uploads = 0;
  routing::PathfinderResult parallel_result =
      routing::run_pathfinder(congestion_graph,
                              congestion_metadata,
                              parallel_options,
                              nullptr);
  require(parallel_result.routed,
          "parallel congestion-free routing should preserve routed status");
  require(parallel_result.nets[0].sinks[0].nodes == std::vector<int>({0, 2, 4}),
          "parallel routing should preserve the first net route");
  require(parallel_result.nets[1].sinks[0].nodes == std::vector<int>({1, 2, 5}),
          "parallel routing should preserve the second net route");
  require(g_unit_bfs_calls == 2,
          "parallel congestion-free routing should call unit BFS once per net");
  require(g_unit_bfs_graph_uploads == 1,
          "parallel unit BFS workers should share one uploaded CSR graph");
  require(g_multisource_delta_calls == 0,
          "parallel default routing should not call delta-step");

  routing::PathfinderOptions parallel_delta_options = parallel_options;
  parallel_delta_options.sssp_engine = routing::SsspEngine::kDeltaStep;
  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  g_delta_graph_uploads = 0;
  routing::PathfinderResult parallel_delta_result =
      routing::run_pathfinder(congestion_graph,
                              congestion_metadata,
                              parallel_delta_options,
                              nullptr);
  require(parallel_delta_result.routed,
          "parallel delta-step routing should preserve routed status");
  require(g_multisource_delta_calls == 2,
          "parallel delta-step routing should call delta once per net");
  require(g_delta_graph_uploads == 1,
          "parallel delta workers should share one uploaded CSR graph");
  require(g_unit_bfs_calls == 0,
          "parallel explicit delta routing should not call unit BFS");

  routing::PathfinderOptions parallel_bellman_ford_options = parallel_options;
  parallel_bellman_ford_options.sssp_engine =
      routing::SsspEngine::kBellmanFord;
  g_bellman_ford_calls = 0;
  g_bellman_ford_graph_uploads = 0;
  g_bellman_ford_workspace_constructions = 0;
  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  const routing::PathfinderResult parallel_bellman_ford_result =
      routing::run_pathfinder(congestion_graph,
                              congestion_metadata,
                              parallel_bellman_ford_options,
                              nullptr);
  require(parallel_bellman_ford_result.routed,
          "parallel Bellman-Ford routing should preserve routed status");
  require(parallel_bellman_ford_result.nets[0].sinks[0].nodes ==
              std::vector<int>({0, 2, 4}),
          "Bellman-Ford should preserve the first net route");
  require(parallel_bellman_ford_result.nets[1].sinks[0].nodes ==
              std::vector<int>({1, 2, 5}),
          "Bellman-Ford should preserve the second net route");
  require(g_bellman_ford_calls == 2,
          "parallel Bellman-Ford routing should call Bellman-Ford once per net");
  require(g_bellman_ford_graph_uploads == 1,
          "parallel Bellman-Ford workers should share one uploaded CSR graph");
  require(g_bellman_ford_workspace_constructions == 2,
          "explicitly requested Bellman-Ford workers should own two workspaces");
  require(g_multisource_delta_calls == 0,
          "explicit Bellman-Ford routing should not call delta-step");
  require(g_unit_bfs_calls == 0,
          "explicit Bellman-Ford routing should not call unit BFS");

  const std::filesystem::path bellman_ford_routes_path =
      "/tmp/congestion_free_bellman_ford_routes.jsonl";
  routing::write_routes_jsonl(bellman_ford_routes_path,
                              congestion_graph,
                              congestion_metadata,
                              parallel_bellman_ford_result);
  std::ifstream bellman_ford_routes_file(bellman_ford_routes_path);
  const std::string bellman_ford_routes_json(
      (std::istreambuf_iterator<char>(bellman_ford_routes_file)),
      std::istreambuf_iterator<char>());
  require(bellman_ford_routes_json.find("\"net\":\"net_a\"") !=
              std::string::npos,
          "Bellman-Ford route output should use the existing net-level schema");
  require(bellman_ford_routes_json.find("\"sources\":[") !=
              std::string::npos &&
              bellman_ford_routes_json.find("\"sinks\":[") !=
              std::string::npos &&
              bellman_ford_routes_json.find("\"edges\":[") !=
              std::string::npos,
          "Bellman-Ford route output should retain sources, sinks, and PIP edges");
  require(bellman_ford_routes_json.find("\"type\":\"path\"") ==
              std::string::npos,
          "Bellman-Ford PathFinder output must not use standalone bf8/bf9 JSONL");

  require(routing::parse_sssp_engine_arg("bellman-ford") ==
              routing::SsspEngine::kBellmanFord,
          "bellman-ford engine name should parse");
  require(routing::parse_sssp_engine_arg("bf8") ==
              routing::SsspEngine::kBellmanFord,
          "bf8 engine alias should parse");
  require(routing::parse_sssp_engine_arg("bf9") ==
              routing::SsspEngine::kBellmanFord,
          "bf9 engine alias should parse");
  require(std::string(routing::sssp_engine_name(
              routing::SsspEngine::kBellmanFord)) == "bellman-ford",
          "Bellman-Ford engine should have a stable display name");

  routing::PathfinderOptions auto_worker_options = parallel_options;
  auto_worker_options.parallel_net_workers = 0;
  g_unit_bfs_calls = 0;
  g_unit_bfs_graph_uploads = 0;
  routing::PathfinderResult auto_worker_result =
      routing::run_pathfinder(congestion_graph,
                              congestion_metadata,
                              auto_worker_options,
                              nullptr);
  require(auto_worker_result.routed,
          "memory-aware unit BFS worker selection should preserve routed status");
  require(g_unit_bfs_calls == 2,
          "auto-selected unit BFS workers should still route every net");
  require(g_unit_bfs_graph_uploads == 1,
          "auto-selected unit BFS workers should share one uploaded CSR graph");

  for (const bool weighted_fallback : {false, true}) {
    HostCsrF32 auto_delta_graph = congestion_graph;
    if (weighted_fallback) {
      std::fill(auto_delta_graph.values.begin(),
                auto_delta_graph.values.end(),
                2.0f);
    }
    routing::PathfinderOptions auto_delta_options = parallel_delta_options;
    auto_delta_options.parallel_net_workers = 0;
    g_multisource_delta_calls = 0;
    g_delta_graph_uploads = 0;
    routing::PathfinderResult auto_delta_result =
        routing::run_pathfinder(auto_delta_graph,
                                congestion_metadata,
                                auto_delta_options,
                                nullptr);
    require(auto_delta_result.routed,
            "auto-selected delta workers should preserve routed status");
    require(g_multisource_delta_calls == 2,
            "auto-selected delta workers should route every net");
    require(g_delta_graph_uploads == 1,
            "auto-selected delta workers should share one uploaded CSR graph");
  }

  const HostCsrF32 overlap_graph = make_three_net_overlap_graph();
  const routing::RoutingMetadata overlap_metadata =
      make_three_net_overlap_metadata(overlap_graph);
  routing::PathfinderOptions overlap_options;
  overlap_options.delta = 1.0f;
  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  routing::PathfinderResult overlap_result =
      routing::run_pathfinder(overlap_graph,
                              overlap_metadata,
                              overlap_options,
                              nullptr);
  require(overlap_result.nets.size() == 3,
          "three-net overlap test should preserve all net result slots");
  require(overlap_result.overused_nodes == 1,
          "three-net overlap test should leave the intentionally unavoidable conflict");
  require(overlap_result.nets[2].sinks[0].nodes == std::vector<int>({6, 7}),
          "independent third net should keep its shortest path");
  require(g_unit_bfs_calls == 3,
          "congestion-free default router should call unit BFS once per net");
  require(g_multisource_delta_calls == 0,
          "congestion-free default router should not call delta-step");

  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  routing::PathfinderOptions options;
  options.delta = 1.0f;

  HostCsrF32 unit_graph = graph;
  unit_graph.nnz = 3;
  unit_graph.rowptr = {0, 1, 3, 3, 3};
  unit_graph.colind = {1, 2, 3};
  unit_graph.values = {1.0f, 1.0f, 1.0f};
  routing::RoutingMetadata metadata = make_metadata();
  metadata.edge_attrs.resize(static_cast<std::size_t>(unit_graph.nnz));
  routing::PathfinderResult result =
      routing::run_pathfinder(unit_graph, metadata, options, nullptr);
  require(result.routed, "PathFinder should route the simple tree graph");
  require(result.all_sinks_reached, "all sinks should be reached");
  require(result.overused_nodes == 0, "simple route should have no overused nodes");
  require(result.max_occupancy == 1, "each used node should have occupancy 1");
  require(result.nets.size() == 1, "expected exactly one routed net");
  require(result.nets[0].sinks.size() == 2, "expected two routed sinks");
  require(result.nets[0].sinks[0].source == 0, "first sink should route from source 0");
  require(result.nets[0].sinks[0].target == 2, "first sink target should be 2");
  require(result.nets[0].sinks[0].nodes == std::vector<int>({0, 1, 2}),
          "first sink should use path 0,1,2");
  require(result.nets[0].sinks[1].source == 1,
          "second sink should connect from existing route tree node 1");
  require(result.nets[0].sinks[1].target == 3, "second sink target should be 3");
  require(result.nets[0].sinks[1].nodes == std::vector<int>({1, 3}),
          "second sink should use path 1,3");
  require(result.nets[0].unique_nodes == std::vector<int>({0, 1, 2, 3}),
          "route tree should contain nodes 0,1,2,3");
  require(result.occupancy == std::vector<int>({1, 1, 1, 1}),
          "all route tree nodes should be occupied once");
  require(g_unit_bfs_calls == 2,
          "PathFinder should rerun unit BFS after expanding a multi-sink tree");
  require(g_multisource_delta_calls == 0,
          "default unit BFS path should not call delta-step");

  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  routing::PathfinderOptions delta_options = options;
  delta_options.sssp_engine = routing::SsspEngine::kDeltaStep;
  routing::PathfinderResult delta_result =
      routing::run_pathfinder(unit_graph, metadata, delta_options, nullptr);
  require(delta_result.routed, "delta-step comparison path should still route");
  require(delta_result.nets[0].sinks[0].nodes == std::vector<int>({0, 1, 2}),
          "delta-step comparison path should preserve first sink route");
  require(delta_result.nets[0].sinks[1].nodes == std::vector<int>({1, 3}),
          "delta-step comparison path should preserve second sink route");
  require(g_multisource_delta_calls == 2,
          "delta-step should rerun after expanding a multi-sink tree");
  require(g_unit_bfs_calls == 0,
          "explicit delta-step comparison path should not call unit BFS");

  const auto adaptive_window_options = [] {
    routing::PathfinderOptions window_options;
    window_options.sssp_engine = routing::SsspEngine::kDeltaStep;
    window_options.delta = 1.0f;
    window_options.route_window_enabled = true;
    window_options.route_window_min_margin = 0;
    window_options.route_window_max_margin = 0;
    window_options.route_window_margin_scale = 0.0f;
    return window_options;
  };

  // A complete in-window route retains its globally optimal cost after the
  // mandatory exclusive-cost verification query.
  HostCsrF32 bounded_optimal_graph;
  bounded_optimal_graph.rows = 3;
  bounded_optimal_graph.cols = 3;
  bounded_optimal_graph.nnz = 2;
  bounded_optimal_graph.rowptr = {0, 1, 2, 2};
  bounded_optimal_graph.colind = {1, 2};
  bounded_optimal_graph.values = {1.0f, 1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> bounded_optimal_bounds = {
      {0, 0, 0, 0, 1}, {1, 1, 0, 0, 1}, {2, 2, 0, 0, 1}};
  const routing::PathfinderResult bounded_optimal_result =
      routing::run_pathfinder(
          bounded_optimal_graph,
          make_window_metadata(bounded_optimal_graph, bounded_optimal_bounds,
                               0, 2),
          adaptive_window_options(), nullptr);
  require(bounded_optimal_result.routed &&
              bounded_optimal_result.nets[0].sinks[0].distance == 2.0f,
          "bounded route should retain the unbounded optimal cost");

  // The direct in-window path is reachable but expensive.  The unbounded
  // exclusive-cost verification must replace it with the outside-box path.
  HostCsrF32 cheaper_outside_graph;
  cheaper_outside_graph.rows = 4;
  cheaper_outside_graph.cols = 4;
  cheaper_outside_graph.nnz = 4;
  cheaper_outside_graph.rowptr = {0, 2, 3, 4, 4};
  cheaper_outside_graph.colind = {1, 2, 3, 3};
  cheaper_outside_graph.values = {5.0f, 1.0f, 5.0f, 1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> cheaper_outside_bounds = {
      {0, 0, 0, 0, 1}, {1, 1, 0, 0, 1},
      {100, 100, 0, 0, 1}, {2, 2, 0, 0, 1}};
  const routing::PathfinderResult cheaper_outside_result =
      routing::run_pathfinder(
          cheaper_outside_graph,
          make_window_metadata(cheaper_outside_graph, cheaper_outside_bounds,
                               0, 3),
          adaptive_window_options(), nullptr);
  require(cheaper_outside_result.routed &&
              cheaper_outside_result.nets[0].sinks[0].distance == 2.0f &&
              cheaper_outside_result.nets[0].sinks[0].nodes ==
                  std::vector<int>({0, 2, 3}),
          "a reachable but nonoptimal window path must be replaced globally");

  // A successful multi-sink bounded attempt performs one batched global
  // verification.  The first target has a finite equal-cost path outside the
  // box and the second is at the exclusive limit; both bounded incumbents
  // must be retained.
  HostCsrF32 multi_sink_retained_graph;
  multi_sink_retained_graph.rows = 7;
  multi_sink_retained_graph.cols = 7;
  multi_sink_retained_graph.nnz = 7;
  multi_sink_retained_graph.rowptr = {0, 3, 4, 5, 6, 6, 7, 7};
  multi_sink_retained_graph.colind = {1, 2, 3, 4, 4, 5, 6};
  multi_sink_retained_graph.values =
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> multi_sink_retained_bounds = {
      {0, 0, 0, 0, 1}, {100, 100, 0, 0, 1}, {1, 1, 0, 0, 1},
      {2, 2, 0, 0, 1}, {4, 4, 0, 0, 1}, {3, 3, 0, 0, 1},
      {6, 6, 0, 0, 1}};
  routing::PathfinderOptions multi_sink_retained_options =
      adaptive_window_options();
  multi_sink_retained_options.route_window_stats_out_path =
      "/tmp/pathfinder_multi_sink_retained_stats.jsonl";
  g_multisource_delta_calls = 0;
  const routing::PathfinderResult multi_sink_retained_result =
      routing::run_pathfinder(
          multi_sink_retained_graph,
          make_window_metadata_with_sinks(multi_sink_retained_graph,
                                          multi_sink_retained_bounds, 0,
                                          std::vector<int>{4, 6}),
          multi_sink_retained_options, nullptr);
  std::ifstream multi_sink_retained_stats_file(
      multi_sink_retained_options.route_window_stats_out_path);
  const std::string multi_sink_retained_stats(
      (std::istreambuf_iterator<char>(multi_sink_retained_stats_file)),
      std::istreambuf_iterator<char>());
  const std::string verification_kind = "\"kind\":\"verification\"";
  const std::size_t retained_verification =
      multi_sink_retained_stats.find(verification_kind);
  require(multi_sink_retained_result.routed &&
              multi_sink_retained_result.nets[0].sinks[0].distance == 2.0f &&
              multi_sink_retained_result.nets[0].sinks[0].nodes ==
                  std::vector<int>({0, 2, 4}) &&
              multi_sink_retained_result.nets[0].sinks[1].distance == 3.0f &&
              multi_sink_retained_result.nets[0].sinks[1].nodes ==
                  std::vector<int>({0, 3, 5, 6}) &&
              g_multisource_delta_calls == 2 &&
              retained_verification != std::string::npos &&
              multi_sink_retained_stats.find(
                  verification_kind,
                  retained_verification + verification_kind.size()) ==
                  std::string::npos &&
              multi_sink_retained_stats.find(
                  "\"kind\":\"verification\",\"fallback_triggered\":false,"
                  "\"attempt\":1,\"reason\":\"verification_no_cheaper_path\",") !=
                  std::string::npos &&
              multi_sink_retained_stats.find("\"target_count\":2") !=
                  std::string::npos,
          "multi-sink bounded candidates should be retained after one batched verification");

  // One target has a cheaper route outside the zero-margin endpoint box while
  // the other is already globally shortest.  Batched verification replaces
  // only the strictly cheaper target.
  HostCsrF32 multi_sink_mixed_graph;
  multi_sink_mixed_graph.rows = 6;
  multi_sink_mixed_graph.cols = 6;
  multi_sink_mixed_graph.nnz = 6;
  multi_sink_mixed_graph.rowptr = {0, 3, 4, 5, 5, 6, 6};
  multi_sink_mixed_graph.colind = {1, 2, 4, 3, 5, 3};
  multi_sink_mixed_graph.values = {5.0f, 1.0f, 1.0f, 5.0f, 1.0f, 1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> multi_sink_mixed_bounds = {
      {0, 0, 0, 0, 1}, {1, 1, 0, 0, 1}, {2, 2, 0, 0, 1},
      {3, 3, 0, 0, 1}, {100, 100, 0, 0, 1}, {5, 5, 0, 0, 1}};
  routing::PathfinderOptions multi_sink_mixed_options =
      adaptive_window_options();
  multi_sink_mixed_options.route_window_stats_out_path =
      "/tmp/pathfinder_multi_sink_mixed_stats.jsonl";
  g_multisource_delta_calls = 0;
  const routing::PathfinderResult multi_sink_mixed_result =
      routing::run_pathfinder(
          multi_sink_mixed_graph,
          make_window_metadata_with_sinks(multi_sink_mixed_graph,
                                          multi_sink_mixed_bounds, 0,
                                          std::vector<int>{3, 5}),
          multi_sink_mixed_options, nullptr);
  std::ifstream multi_sink_mixed_stats_file(
      multi_sink_mixed_options.route_window_stats_out_path);
  const std::string multi_sink_mixed_stats(
      (std::istreambuf_iterator<char>(multi_sink_mixed_stats_file)),
      std::istreambuf_iterator<char>());
  const std::size_t mixed_verification =
      multi_sink_mixed_stats.find(verification_kind);
  require(multi_sink_mixed_result.routed &&
              multi_sink_mixed_result.nets[0].sinks[0].distance == 2.0f &&
              multi_sink_mixed_result.nets[0].sinks[0].nodes ==
                  std::vector<int>({0, 4, 3}) &&
              multi_sink_mixed_result.nets[0].sinks[1].distance == 2.0f &&
              multi_sink_mixed_result.nets[0].sinks[1].nodes ==
                  std::vector<int>({0, 2, 5}) &&
              g_multisource_delta_calls == 2 &&
              mixed_verification != std::string::npos &&
              multi_sink_mixed_stats.find(
                  verification_kind,
                  mixed_verification + verification_kind.size()) ==
                  std::string::npos &&
              multi_sink_mixed_stats.find(
                  "\"reason\":\"verification_some_targets_cheaper\"") !=
                  std::string::npos,
          "batched verification should replace only strictly cheaper multi-sink paths");

  // The first zero-margin box misses the bridge; one geometric expansion is
  // required before the target can be found.
  HostCsrF32 expansion_graph;
  expansion_graph.rows = 3;
  expansion_graph.cols = 3;
  expansion_graph.nnz = 2;
  expansion_graph.rowptr = {0, 1, 2, 2};
  expansion_graph.colind = {1, 2};
  expansion_graph.values = {1.0f, 1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> expansion_bounds = {
      {0, 0, 0, 0, 1}, {1, 1, 0, 0, 1}, {0, 0, 0, 0, 1}};
  g_multisource_delta_calls = 0;
  const routing::PathfinderResult expansion_result = routing::run_pathfinder(
      expansion_graph,
      make_window_metadata(expansion_graph, expansion_bounds, 0, 2),
      adaptive_window_options(), nullptr);
  require(expansion_result.routed && g_multisource_delta_calls == 3,
          "adaptive windows should retry once before their verification query");

  // An unreachable target exhausts the device box and then performs the full
  // unbounded fallback rather than treating the final window as authoritative.
  HostCsrF32 fallback_graph;
  fallback_graph.rows = 3;
  fallback_graph.cols = 3;
  fallback_graph.nnz = 1;
  fallback_graph.rowptr = {0, 1, 1, 1};
  fallback_graph.colind = {1};
  fallback_graph.values = {1.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> fallback_bounds = {
      {0, 0, 0, 0, 1}, {10, 10, 0, 0, 1}, {0, 0, 0, 0, 1}};
  routing::PathfinderOptions fallback_options = adaptive_window_options();
  fallback_options.route_window_stats_out_path =
      "/tmp/pathfinder_window_fallback_stats.jsonl";
  const routing::PathfinderResult fallback_result = routing::run_pathfinder(
      fallback_graph, make_window_metadata(fallback_graph, fallback_bounds, 0, 2),
      fallback_options, nullptr);
  std::ifstream fallback_stats_file(fallback_options.route_window_stats_out_path);
  const std::string fallback_stats((std::istreambuf_iterator<char>(fallback_stats_file)),
                                   std::istreambuf_iterator<char>());
  require(!fallback_result.routed &&
              fallback_stats.find("device_bounds_fallback") != std::string::npos &&
              fallback_stats.find("unbounded_fallback") != std::string::npos,
          "saturated route windows must issue and report an unbounded fallback");

  // Unknown coordinates stay traversable, whereas coordinates above the old
  // uint16_t sentinel remain ordinary physical tile coordinates.
  const std::vector<DeltaSteppingCsrNodeBounds> unknown_bounds = {
      {0, 0, 0, 0, 1}, {}, {0, 0, 0, 0, 1}};
  const routing::PathfinderResult unknown_result = routing::run_pathfinder(
      bounded_optimal_graph,
      make_window_metadata(bounded_optimal_graph, unknown_bounds, 0, 2),
      adaptive_window_options(), nullptr);
  require(unknown_result.routed,
          "nodes with unknown coordinates must remain conservatively traversable");
  const std::vector<DeltaSteppingCsrNodeBounds> wide_coordinate_bounds = {
      {70000, 70000, 70000, 70000, 1},
      {70001, 70001, 70000, 70000, 1},
      {70002, 70002, 70000, 70000, 1}};
  const routing::PathfinderResult wide_coordinate_result =
      routing::run_pathfinder(
          bounded_optimal_graph,
          make_window_metadata(bounded_optimal_graph, wide_coordinate_bounds,
                               0, 2),
          adaptive_window_options(), nullptr);
  require(wide_coordinate_result.routed &&
              wide_coordinate_result.nets[0].sinks[0].distance == 2.0f,
          "coordinates at or above 65535 must not become missing bounds");

  // Direct workspace callers get the same fail-fast contract as PathFinder.
  bool missing_uploaded_bounds_threw = false;
  try {
    DeltaSteppingCsrWorkspace no_bounds_workspace(bounded_optimal_graph, nullptr);
    DeltaSteppingCsrRunOptions no_bounds_options;
    no_bounds_options.route_window = {true, 0, 2, 0, 0};
    (void)no_bounds_workspace.run(std::vector<int>{0}, std::vector<int>{2},
                                  1.0f, -1, no_bounds_options, nullptr,
                                  nullptr, nullptr);
  } catch (const std::logic_error&) {
    missing_uploaded_bounds_threw = true;
  }
  require(missing_uploaded_bounds_threw,
          "enabled windows without uploaded node bounds must fail fast");

  // One source has both a light and a heavy edge outside the active box.  The
  // CPU stub uses the same interval predicate for both relaxation families.
  HostCsrF32 filter_graph;
  filter_graph.rows = 3;
  filter_graph.cols = 3;
  filter_graph.nnz = 2;
  filter_graph.rowptr = {0, 2, 2, 2};
  filter_graph.colind = {1, 2};
  filter_graph.values = {0.5f, 2.0f};
  const std::vector<DeltaSteppingCsrNodeBounds> filter_bounds = {
      {0, 0, 0, 0, 1}, {10, 10, 0, 0, 1}, {11, 11, 0, 0, 1}};
  auto filter_shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
      filter_graph, filter_bounds, nullptr);
  DeltaSteppingCsrWorkspace filter_workspace(filter_shared_graph, nullptr);
  DeltaSteppingCsrRunOptions filter_options;
  DeltaSteppingCsrTelemetry filter_telemetry;
  filter_options.telemetry = &filter_telemetry;
  filter_options.route_window = {true, 0, 0, 0, 0};
  const DeltaSteppingCsrResult filter_result = filter_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1, filter_options,
      nullptr, nullptr, nullptr);
  require(!filter_result.target_reached &&
              filter_telemetry.window_rejected_edges == 2,
          "route-window filtering must reject both light and heavy edges");

  // A windowed exact-unit query stays on the specialization; a selected net
  // therefore cannot force an unrelated unit net into generic execution.
  DeltaSteppingCsrTelemetry exact_window_telemetry;
  DeltaSteppingCsrRunOptions exact_window_options;
  exact_window_options.telemetry = &exact_window_telemetry;
  exact_window_options.route_window = {true, 0, 2, 0, 0};
  auto exact_shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
      bounded_optimal_graph, bounded_optimal_bounds, nullptr);
  DeltaSteppingCsrWorkspace exact_window_workspace(exact_shared_graph, nullptr);
  (void)exact_window_workspace.run(std::vector<int>{0}, std::vector<int>{2},
                                   1.0f, -1, exact_window_options, nullptr,
                                   nullptr, nullptr);
  require(exact_window_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kExactUnit,
          "windowed exact-unit work should retain the unit specialization");

  // Select only the first unit-weight net.  Its bounded query may add work,
  // but the unselected net must retain the exact-unit dispatch.
  const std::filesystem::path selective_window_list =
      "/tmp/pathfinder_cpu_stub_selective_windows.jsonl";
  const std::filesystem::path selective_window_stats =
      "/tmp/pathfinder_cpu_stub_selective_windows_stats.jsonl";
  {
    std::ofstream selection_file(selective_window_list);
    selection_file << "{\"net_index\":0,\"net\":\"net_a\"}\n";
    require(static_cast<bool>(selection_file),
            "selective route-window test fixture must be writable");
  }
  routing::PathfinderOptions selective_window_options;
  selective_window_options.sssp_engine = routing::SsspEngine::kDeltaStep;
  selective_window_options.delta = 1.0f;
  selective_window_options.route_window_enabled = true;
  selective_window_options.route_window_net_list_path = selective_window_list;
  selective_window_options.route_window_stats_out_path = selective_window_stats;
  const routing::PathfinderResult selective_window_result =
      routing::run_pathfinder(congestion_graph, congestion_metadata,
                              selective_window_options, nullptr);
  std::ifstream selective_stats_file(selective_window_stats);
  const std::string selective_stats(
      (std::istreambuf_iterator<char>(selective_stats_file)),
      std::istreambuf_iterator<char>());
  const std::size_t net_b_stats_begin = selective_stats.find(
      "\"net_index\":1,\"net_string\":1,\"net\":\"net_b\",\"kind\":\"unbounded_baseline\"");
  const std::size_t net_b_stats_end =
      net_b_stats_begin == std::string::npos
          ? std::string::npos
          : selective_stats.find('\n', net_b_stats_begin);
  const std::string net_b_stats =
      net_b_stats_begin == std::string::npos
          ? std::string{}
          : selective_stats.substr(
                net_b_stats_begin,
                net_b_stats_end == std::string::npos
                    ? std::string::npos
                    : net_b_stats_end - net_b_stats_begin);
  require(selective_window_result.routed &&
              net_b_stats.find("\"execution_path\":\"exact_unit\"") !=
                  std::string::npos,
          "selective windows must not downgrade unselected exact-unit nets");

  routing::RoutingMetadata invalid_sink_metadata = make_metadata();
  invalid_sink_metadata.edge_attrs.resize(static_cast<std::size_t>(unit_graph.nnz));
  invalid_sink_metadata.strings.push_back("UNMAPPED_SINK_SITE");
  invalid_sink_metadata.strings.push_back("UNMAPPED_SINK_PIN");
  invalid_sink_metadata.route_requests[0].sinks.push_back({-1, 12, 13});
  routing::PathfinderResult invalid_sink_result =
      routing::run_pathfinder(unit_graph, invalid_sink_metadata, options, nullptr);
  require(!invalid_sink_result.routed,
          "an unmapped physical sink stub should prevent a routed result");
  require(!invalid_sink_result.all_sinks_reached,
          "all_sinks_reached must include invalid/unmapped sink stubs");
  require(!invalid_sink_result.nets[0].sinks.back().reached,
          "invalid sink should be preserved as an unreached sink result");

  const std::filesystem::path routes_path = "/tmp/pathfinder_cpu_stub_routes.jsonl";
  routing::write_routes_jsonl(routes_path, unit_graph, metadata, result);
  std::ifstream routes_file(routes_path);
  const std::string routes_json((std::istreambuf_iterator<char>(routes_file)),
                                std::istreambuf_iterator<char>());
  require(routes_json.find("\"net\":\"net0\"") != std::string::npos,
          "routes JSONL should include the routed net name");
  require(routes_json.find("\"site\":\"SINK_SITE_0\"") != std::string::npos,
          "routes JSONL should include sink site pins");
  require(routes_json.find("\"tile\":\"TILE_A\"") != std::string::npos,
          "routes JSONL should include PIP tile names");
  require(routes_json.find("\"from\":1,\"to\":3") != std::string::npos,
          "routes JSONL should include tree edge from existing route node");

  std::cout << "PathFinder CPU-stub test passed\n";
  return 0;
}
