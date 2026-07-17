// Build and run from the repository root:
//   g++ -std=c++17 -O2 -pthread \
//     -I Routing/tests/fake_hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/tests/pathfinder_bf10_cpu_stub_test.cpp \
//     -o /tmp/pathfinder_bf10_cpu_stub_test
//   /tmp/pathfinder_bf10_cpu_stub_test

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
                                              const std::vector<int>& sources) {
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

  for (const int source : sources) {
    result.dist[static_cast<std::size_t>(source)] = 0.0f;
    queue.push({0.0f, source});
  }
  while (!queue.empty()) {
    const auto [du, u] = queue.top();
    queue.pop();
    if (du != result.dist[static_cast<std::size_t>(u)]) {
      continue;
    }
    for (const OutEdge& edge : outgoing[static_cast<std::size_t>(u)]) {
      const float candidate = du + edge.weight;
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
  explicit Impl(const HostCsrF32& adjacency) : graph(adjacency) {}

  HostCsrF32 graph;
};

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                                             hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency)) {
  (void)stream;
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
  return delta_stepping_minplus_hip_csr(impl_->graph,
                                        sources,
                                        target,
                                        delta,
                                        max_iters,
                                        stream,
                                        progress_callback,
                                        progress_user_data);
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  DeltaSteppingCsrResult result =
      delta_stepping_minplus_hip_csr(impl_->graph,
                                     sources,
                                     -1,
                                     delta,
                                     max_iters,
                                     stream,
                                     progress_callback,
                                     progress_user_data);
  CpuSsspResult cpu_result;
  cpu_result.dist = result.dist;
  cpu_result.pred_node = result.pred_node;
  cpu_result.pred_edge = result.pred_edge;
  fill_compact_target_paths(impl_->graph, sources, targets, cpu_result, result);
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
      routing::parse_sssp_engine_arg("bellman-ford");
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
  require(routing::parse_sssp_engine_arg("bf10") ==
              routing::SsspEngine::kBellmanFord,
          "bf10 engine alias should parse");
  require(std::string(routing::sssp_engine_name(
              routing::SsspEngine::kBellmanFord)) == "bellman-ford",
          "Bellman-Ford engine should have a stable display name");

  for (const char* compatibility_alias : {"bf9", "bf10"}) {
    routing::PathfinderOptions alias_options = parallel_options;
    alias_options.sssp_engine =
        routing::parse_sssp_engine_arg(compatibility_alias);
    g_bellman_ford_calls = 0;
    g_bellman_ford_graph_uploads = 0;
    g_bellman_ford_workspace_constructions = 0;
    g_multisource_delta_calls = 0;
    g_unit_bfs_calls = 0;

    const routing::PathfinderResult alias_result =
        routing::run_pathfinder(congestion_graph,
                                congestion_metadata,
                                alias_options,
                                nullptr);
    require(alias_result.routed,
            "bf9/bf10 aliases should select a working BF10 backend");
    require(g_bellman_ford_calls == 2,
            "bf9/bf10 aliases should call the BF10 workspace once per net");
    require(g_bellman_ford_graph_uploads == 1,
            "bf9/bf10 aliases should share one BF10 graph upload");
    require(g_bellman_ford_workspace_constructions == 2,
            "bf9/bf10 aliases should construct the requested BF10 workers");
    require(g_multisource_delta_calls == 0 && g_unit_bfs_calls == 0,
            "bf9/bf10 aliases should not select another SSSP backend");
  }

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
  require(g_unit_bfs_calls == 1,
          "PathFinder should call unit BFS once per multi-sink net by default");
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
  require(g_multisource_delta_calls == 1,
          "explicit delta-step comparison path should call delta-step");
  require(g_unit_bfs_calls == 0,
          "explicit delta-step comparison path should not call unit BFS");

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
