#define ROUTING_PATHFINDER_NO_MAIN

#include "../pathfinder.cpp"

#include <queue>

namespace {

int g_targeted_delta_calls = 0;

std::vector<float> cpu_dijkstra_incoming_csr(const HostCsrF32& graph,
                                             int source) {
  std::vector<std::vector<std::pair<int, float>>> outgoing(
      static_cast<std::size_t>(graph.rows));
  for (int dst = 0; dst < graph.rows; ++dst) {
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(dst)];
         edge < graph.rowptr[static_cast<std::size_t>(dst + 1)];
         ++edge) {
      const int src = graph.colind[static_cast<std::size_t>(edge)];
      const float weight = graph.values[static_cast<std::size_t>(edge)];
      outgoing[static_cast<std::size_t>(src)].push_back({dst, weight});
    }
  }

  constexpr float inf = std::numeric_limits<float>::infinity();
  std::vector<float> dist(static_cast<std::size_t>(graph.rows), inf);
  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;

  dist[static_cast<std::size_t>(source)] = 0.0f;
  queue.push({0.0f, source});
  while (!queue.empty()) {
    const auto [du, u] = queue.top();
    queue.pop();
    if (du != dist[static_cast<std::size_t>(u)]) {
      continue;
    }
    for (const auto& [v, weight] : outgoing[static_cast<std::size_t>(u)]) {
      const float candidate = du + weight;
      if (candidate < dist[static_cast<std::size_t>(v)]) {
        dist[static_cast<std::size_t>(v)] = candidate;
        queue.push({candidate, v});
      }
    }
  }
  return dist;
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

routing::RoutingMetadata make_metadata() {
  routing::RoutingMetadata metadata;
  metadata.strings = {"net0"};
  metadata.node_device_ids = {0, 1, 2, 3};
  metadata.edge_attrs.resize(4);

  routing::RouteRequest request;
  request.net_string = 0;
  request.sources.push_back({0, 0, 0});
  request.sinks.push_back({2, 0, 0});
  request.sinks.push_back({3, 0, 0});
  metadata.route_requests.push_back(std::move(request));
  return metadata;
}

}  // namespace

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
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
  ++g_targeted_delta_calls;

  DeltaSteppingCsrResult result;
  result.target = target;
  result.dist = cpu_dijkstra_incoming_csr(adjacency, source);
  result.iterations_used = 1;
  result.target_distance = result.dist[static_cast<std::size_t>(target)];
  result.target_reached = std::isfinite(result.target_distance);
  result.stopped_on_target = result.target_reached;
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

  routing::PathfinderOptions options;
  options.max_pathfinder_iterations = 1;
  options.delta = 1.0f;

  routing::PathfinderResult result =
      routing::run_pathfinder(graph, make_metadata(), options, nullptr);
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
  require(g_targeted_delta_calls == 4,
          "PathFinder should call targeted delta stepping for source candidates");

  std::cout << "PathFinder CPU-stub test passed\n";
  return 0;
}

