#define ROUTING_PATHFINDER_NO_MAIN

#include "../pathfinder.cpp"

#include <fstream>
#include <queue>

namespace {

int g_multisource_delta_calls = 0;

struct CpuSsspResult {
  std::vector<float> dist;
  std::vector<int> pred_node;
  std::vector<minplus_sparse::Offset> pred_edge;
};

CpuSsspResult cpu_dijkstra_incoming_csr_multi(const HostCsrF32& graph,
                                              const std::vector<int>& sources) {
  struct OutEdge {
    int to = -1;
    float weight = 0.0f;
    minplus_sparse::Offset edge = -1;
  };

  std::vector<std::vector<OutEdge>> outgoing(
      static_cast<std::size_t>(graph.rows));
  for (int dst = 0; dst < graph.rows; ++dst) {
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(dst)];
         edge < graph.rowptr[static_cast<std::size_t>(dst + 1)];
         ++edge) {
      const int src = graph.colind[static_cast<std::size_t>(edge)];
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

std::vector<float> cpu_dijkstra_incoming_csr(const HostCsrF32& graph,
                                             int source) {
  return cpu_dijkstra_incoming_csr_multi(graph, std::vector<int>{source}).dist;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

HostCsrF32 make_tree_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 4;
  graph.rowptr = {0, 0, 1, 2, 4};
  graph.colind = {0, 1, 1, 0};
  graph.values = {1.0f, 1.0f, 1.0f, 10.0f};
  return graph;
}

HostCsrF32 make_self_loop_predecessor_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 2;
  graph.rowptr = {0, 0, 0, 2, 2};
  graph.colind = {2, 3};
  graph.values = {0.0f, 1.0f};
  return graph;
}

HostCsrF32 make_two_path_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 4;
  graph.rowptr = {0, 0, 1, 2, 4};
  graph.colind = {0, 0, 1, 2};
  graph.values = {1.0f, 1.0f, 1.0f, 1.0f};
  return graph;
}

HostCsrF32 make_two_net_congestion_graph() {
  HostCsrF32 graph;
  graph.rows = 6;
  graph.cols = 6;
  graph.nnz = 6;
  graph.rowptr = {0, 0, 0, 2, 3, 4, 6};
  graph.colind = {0, 1, 1, 2, 2, 3};
  graph.values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  return graph;
}

HostCsrF32 make_incremental_reroute_graph() {
  HostCsrF32 graph;
  graph.rows = 8;
  graph.cols = 8;
  graph.nnz = 5;
  graph.rowptr = {0, 0, 0, 2, 2, 3, 4, 4, 5};
  graph.colind = {0, 1, 2, 2, 6};
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

routing::RoutingMetadata make_incremental_reroute_metadata(const HostCsrF32& graph) {
  routing::RoutingMetadata metadata;
  metadata.strings = {"net_a", "net_b", "net_c", "SRC_SITE_A", "SRC_SITE_B",
                      "SRC_SITE_C", "SRC_PIN", "SINK_SITE_A", "SINK_SITE_B",
                      "SINK_SITE_C", "SINK_PIN", "TILE_A", "WIRE_0", "WIRE_1"};
  metadata.node_device_ids = {0, 1, 2, 3, 4, 5, 6, 7};
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
  for (int dst = 0; dst < impl_->graph.rows; ++dst) {
    const float node_cost = vertex_costs[static_cast<std::size_t>(dst)];
    for (minplus_sparse::Offset edge = impl_->graph.rowptr[static_cast<std::size_t>(dst)];
         edge < impl_->graph.rowptr[static_cast<std::size_t>(dst + 1)];
         ++edge) {
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
  result.target_distances.resize(targets.size(), std::numeric_limits<float>::infinity());
  result.target_sources.resize(targets.size(), -1);
  result.target_path_offsets.assign(targets.size() + 1, 0);
  result.target_edge_offsets.assign(targets.size() + 1, 0);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const int target = targets[i];
    result.target_path_offsets[i] =
        static_cast<int>(result.target_path_nodes.size());
    result.target_edge_offsets[i] =
        static_cast<int>(result.target_path_edges.size());
    if (target < 0 ||
        static_cast<std::size_t>(target) >= result.dist.size() ||
        !std::isfinite(result.dist[static_cast<std::size_t>(target)])) {
      result.target_reached = false;
      continue;
    }

    result.target_distances[i] = result.dist[static_cast<std::size_t>(target)];
    std::vector<int> reversed_nodes;
    std::vector<minplus_sparse::Offset> reversed_edges;
    int current = target;
    for (int guard = 0; guard < impl_->graph.rows; ++guard) {
      reversed_nodes.push_back(current);
      if (std::find(sources.begin(), sources.end(), current) != sources.end()) {
        result.target_sources[i] = current;
        break;
      }
      const int pred = result.pred_node[static_cast<std::size_t>(current)];
      if (pred < 0) {
        break;
      }
      reversed_edges.push_back(result.pred_edge[static_cast<std::size_t>(current)]);
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
  CpuSsspResult cpu_result = cpu_dijkstra_incoming_csr_multi(adjacency, sources);
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

int main() {
  const HostCsrF32 graph = make_tree_graph();
  const std::vector<float> dist = cpu_dijkstra_incoming_csr(graph, 0);
  const std::vector<routing::PathEdge> path =
      routing::reconstruct_shortest_path(graph, dist, 0, 2);
  require(path.size() == 2, "expected two edges in 0->2 reconstructed path");
  require(path[0].from == 0 && path[0].to == 1, "first path edge should be 0->1");
  require(path[1].from == 1 && path[1].to == 2, "second path edge should be 1->2");

  const HostCsrF32 self_loop_graph = make_self_loop_predecessor_graph();
  const std::vector<float> self_loop_dist =
      cpu_dijkstra_incoming_csr(self_loop_graph, 3);
  const std::vector<routing::PathEdge> self_loop_path =
      routing::reconstruct_shortest_path(self_loop_graph, self_loop_dist, 3, 2);
  require(self_loop_path.size() == 1,
          "self-loop predecessor graph should reconstruct one real edge");
  require(self_loop_path[0].from == 3 && self_loop_path[0].to == 2,
          "reconstruction should ignore no-progress self-loop predecessors");

  const HostCsrF32 two_path_graph = make_two_path_graph();
  std::vector<int> occupancy = {10, 1, 0, 0};
  std::vector<float> history = {0.0f, 0.0f, 0.0f, 0.0f};
  routing::PathfinderOptions cost_options;
  cost_options.capacity = 1;
  const HostCsrF32 costed_graph =
      routing::make_costed_graph(two_path_graph, occupancy, history, cost_options, 5.0f);
  require(std::fabs(costed_graph.values[0] - 6.0f) < 1e-6f,
          "entering occupied vertex 1 should receive present congestion cost");
  require(std::fabs(costed_graph.values[1] - 1.0f) < 1e-6f,
          "congested source vertex 0 should not inflate outgoing edge 0->2");
  const std::vector<float> costed_dist =
      cpu_dijkstra_incoming_csr(costed_graph, 0);
  const std::vector<routing::PathEdge> costed_path =
      routing::reconstruct_shortest_path(costed_graph, costed_dist, 0, 3);
  require(costed_path.size() == 2, "vertex-weighted two-path graph should use two edges");
  require(costed_path[0].from == 0 && costed_path[0].to == 2,
          "PathFinder vertex costs should steer away from occupied destination nodes");
  require(costed_path[1].from == 2 && costed_path[1].to == 3,
          "PathFinder vertex costs should preserve a valid path to the sink");

  const HostCsrF32 congestion_graph = make_two_net_congestion_graph();
  const routing::RoutingMetadata congestion_metadata =
      make_two_net_metadata(congestion_graph);
  routing::PathfinderOptions congestion_options;
  congestion_options.max_pathfinder_iterations = 1;
  congestion_options.delta = 1.0f;
  congestion_options.route_batch_size = 1;
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

  const HostCsrF32 incremental_graph = make_incremental_reroute_graph();
  const routing::RoutingMetadata incremental_metadata =
      make_incremental_reroute_metadata(incremental_graph);
  routing::PathfinderOptions incremental_options;
  incremental_options.max_pathfinder_iterations = 2;
  incremental_options.delta = 1.0f;
  incremental_options.route_batch_size = 3;
  g_multisource_delta_calls = 0;
  routing::PathfinderResult incremental_result =
      routing::run_pathfinder(incremental_graph,
                              incremental_metadata,
                              incremental_options,
                              nullptr);
  require(incremental_result.nets.size() == 3,
          "incremental test should preserve all net result slots");
  require(incremental_result.overused_nodes == 1,
          "incremental test should leave the intentionally unavoidable conflict");
  require(incremental_result.nets[2].sinks[0].nodes == std::vector<int>({6, 7}),
          "uncongested retained net should keep its previous route");
  require(g_multisource_delta_calls == 3,
          "congestion-free router should call SSSP once per net");

  g_multisource_delta_calls = 0;
  routing::PathfinderOptions options;
  options.max_pathfinder_iterations = 1;
  options.delta = 1.0f;

  const routing::RoutingMetadata metadata = make_metadata();
  routing::PathfinderResult result =
      routing::run_pathfinder(graph, metadata, options, nullptr);
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
  require(g_multisource_delta_calls == 1,
          "PathFinder should call multi-source delta stepping once per multi-sink net");

  const std::filesystem::path routes_path = "/tmp/pathfinder_cpu_stub_routes.jsonl";
  routing::write_routes_jsonl(routes_path, graph, metadata, result);
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
