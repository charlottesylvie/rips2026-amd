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
#include <sstream>

namespace {

std::atomic<int> g_multisource_delta_calls{0};
std::atomic<int> g_delta_graph_uploads{0};
std::atomic<int> g_delta_force_generic_calls{0};
std::atomic<int> g_delta_force_legacy_calls{0};
std::atomic<int> g_bellman_ford_calls{0};
std::atomic<int> g_bellman_ford_graph_uploads{0};
std::atomic<int> g_bellman_ford_workspace_constructions{0};
std::atomic<int> g_unit_bfs_calls{0};
std::atomic<int> g_unit_bfs_graph_uploads{0};
std::mutex g_delta_values_mutex;
std::vector<float> g_delta_values;

void clear_recorded_deltas() {
  std::lock_guard<std::mutex> lock(g_delta_values_mutex);
  g_delta_values.clear();
}

std::vector<float> recorded_deltas() {
  std::lock_guard<std::mutex> lock(g_delta_values_mutex);
  return g_delta_values;
}

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

void populate_stub_delta_telemetry(
    DeltaSteppingCsrTelemetry& telemetry,
    DeltaSteppingCsrExecutionPath execution_path,
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    float delta,
    bool force_generic,
    bool force_legacy_parent,
    bool has_vertex_costs) {
  const std::uint64_t token =
      sources.empty()
          ? 1
          : static_cast<std::uint64_t>(sources.front()) + 1;
  telemetry.collected = true;
  telemetry.completed = true;
  telemetry.execution_path = execution_path;
  telemetry.resolved_delta = delta;
  telemetry.wavefront_size = 64;
  telemetry.force_generic = force_generic;
  telemetry.force_legacy_parent = force_legacy_parent;
  telemetry.has_vertex_costs = has_vertex_costs;
  telemetry.all_edges_light =
      !has_vertex_costs &&
      std::all_of(graph.values.begin(), graph.values.end(),
                  [delta](float value) { return value <= delta; });
  telemetry.outer_buckets_processed = token;
  telemetry.light_relaxation_rounds = 2 * token;
  telemetry.heavy_edge_phases = 3 * token;
  telemetry.frontier_entries_processed = 4 * token;
  telemetry.active_vertices_processed = 5 * token;
  telemetry.stale_frontier_entries = 6 * token;
  telemetry.light_edge_visits = 7 * token;
  telemetry.heavy_edge_visits = 8 * token;
  telemetry.distance_atomic_attempts = 9 * token;
  telemetry.successful_distance_relaxations = 10 * token;
  telemetry.distance_cas_retries = 11 * token;
  telemetry.current_queue_insertions = 12 * token;
  telemetry.pending_queue_insertions = 13 * token;
  telemetry.heavy_queue_insertions = 14 * token;
  telemetry.bucket_insertions = 15 * token;
  telemetry.pending_entry_examinations = 16 * token;
  telemetry.stale_pending_entry_examinations = 17 * token;
  telemetry.reached_vertices = 18 * token;
  telemetry.current_queue_high_water = 19 * token;
  telemetry.pending_queue_high_water = 20 * token;
  telemetry.heavy_queue_high_water = 21 * token;
  telemetry.controller_round_trips = 22 * token;
  telemetry.compact_parent_fallback_events = 23 * token;
}

class ScopedCoutCapture {
 public:
  ScopedCoutCapture() : previous_(std::cout.rdbuf(output_.rdbuf())) {}

  ~ScopedCoutCapture() { std::cout.rdbuf(previous_); }

  ScopedCoutCapture(const ScopedCoutCapture&) = delete;
  ScopedCoutCapture& operator=(const ScopedCoutCapture&) = delete;

  std::string str() const { return output_.str(); }

 private:
  std::ostringstream output_;
  std::streambuf* previous_ = nullptr;
};

std::string single_delta_telemetry_json_line(const std::string& output) {
  constexpr const char* marker = "{\"type\":\"delta_stepping_telemetry\"";
  const std::size_t marker_pos = output.find(marker);
  require(marker_pos != std::string::npos,
          "telemetry-enabled PathFinder must emit a JSON record");
  require(output.find(marker, marker_pos + 1) == std::string::npos,
          "telemetry-enabled PathFinder must emit exactly one JSON record");
  const std::size_t preceding_newline = output.rfind('\n', marker_pos);
  const std::size_t line_begin =
      preceding_newline == std::string::npos ? 0 : preceding_newline + 1;
  const std::size_t following_newline = output.find('\n', marker_pos);
  const std::size_t line_end = following_newline == std::string::npos
                                   ? output.size()
                                   : following_newline;
  const std::string json = output.substr(line_begin, line_end - line_begin);
  require(!json.empty() && json.front() == '{' && json.back() == '}',
          "PathFinder telemetry must occupy one complete JSON line");
  return json;
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

HostCsrF32 make_cached_multi_sink_counterexample_graph() {
  // The initial shortest paths are uniquely 0->1->2 and 0->3->4.  Once the
  // first sink is attached, the exact route tree has a one-edge path 2->4,
  // which a cached source-0 result cannot discover by trimming its old path.
  HostCsrF32 graph;
  graph.rows = 5;
  graph.cols = 5;
  graph.nnz = 5;
  graph.rowptr = {0, 2, 3, 4, 5, 5};
  graph.colind = {1, 3, 2, 4, 4};
  graph.values.assign(static_cast<std::size_t>(graph.nnz), 1.0f);
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

routing::RoutingMetadata make_cached_multi_sink_counterexample_metadata(
    const HostCsrF32& graph) {
  routing::RoutingMetadata metadata;
  metadata.strings = {"cached_counterexample"};
  metadata.node_device_ids = {0, 1, 2, 3, 4};
  add_default_node_metadata(metadata);
  metadata.edge_attrs.resize(static_cast<std::size_t>(graph.nnz));

  routing::RouteRequest request;
  request.net_string = 0;
  request.sources.push_back({0, 0, 0});
  request.sinks.push_back({2, 0, 0});
  request.sinks.push_back({4, 0, 0});
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

struct DetachedCompactWorkspace {
  UnitBfsCsrResult run(const std::vector<int>& sources,
                       const std::vector<int>& targets,
                       float delta,
                       int max_depth,
                       hipStream_t stream,
                       UnitBfsCsrProgressCallback progress_callback,
                       void* progress_user_data) {
    (void)sources;
    (void)targets;
    (void)delta;
    (void)max_depth;
    (void)stream;
    (void)progress_callback;
    (void)progress_user_data;

    UnitBfsCsrResult result;
    result.target_distances = {1.0f};
    result.target_sources = {1};
    result.target_path_offsets = {0, 2};
    result.target_edge_offsets = {0, 1};
    result.target_path_nodes = {1, 2};
    result.target_path_edges = {2};
    result.target_reached = true;
    result.stopped_on_target = true;
    result.converged = true;
    return result;
  }
};

}  // namespace

const char* delta_stepping_execution_path_name(
    DeltaSteppingCsrExecutionPath path) noexcept {
  switch (path) {
    case DeltaSteppingCsrExecutionPath::kNotRun:
      return "not_run";
    case DeltaSteppingCsrExecutionPath::kExactUnit:
      return "exact_unit";
    case DeltaSteppingCsrExecutionPath::kCompactGeneric:
      return "compact_generic";
    case DeltaSteppingCsrExecutionPath::kLegacyGeneric:
      return "legacy_generic";
    case DeltaSteppingCsrExecutionPath::kGenericDistancesOnly:
      return "generic_distances_only";
  }
  return "unknown";
}

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
  bool has_vertex_costs = false;
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
  impl_->has_vertex_costs = true;
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
  DeltaSteppingCsrResult result = delta_stepping_minplus_hip_csr(
      impl_->graph,
      sources,
      target,
      delta,
      max_iters,
      stream,
      progress_callback,
      progress_user_data);
  if (active_telemetry_ != nullptr) {
    populate_stub_delta_telemetry(
        *active_telemetry_,
        DeltaSteppingCsrExecutionPath::kLegacyGeneric,
        impl_->graph,
        sources,
        delta,
        execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
        parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
        impl_->has_vertex_costs);
  }
  return result;
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  if (execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric) {
    ++g_delta_force_generic_calls;
  }
  if (parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy) {
    ++g_delta_force_legacy_calls;
  }
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
  if (active_telemetry_ != nullptr) {
    const bool exact_unit_path =
        execution_mode_ == DeltaSteppingCsrExecutionMode::kAutomatic &&
        parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
        !impl_->has_vertex_costs && max_iters < 0 &&
        progress_callback == nullptr &&
        std::all_of(impl_->graph.values.begin(), impl_->graph.values.end(),
                    [](float value) { return value == 1.0f; });
    const DeltaSteppingCsrExecutionPath execution_path =
        exact_unit_path
            ? DeltaSteppingCsrExecutionPath::kExactUnit
            : parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic
                  ? DeltaSteppingCsrExecutionPath::kCompactGeneric
                  : DeltaSteppingCsrExecutionPath::kLegacyGeneric;
    populate_stub_delta_telemetry(
        *active_telemetry_,
        execution_path,
        impl_->graph,
        sources,
        delta,
        execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
        parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
        impl_->has_vertex_costs);
  }
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
  (void)max_iters;
  (void)stream;
  (void)progress_callback;
  (void)progress_user_data;
  ++g_multisource_delta_calls;
  {
    std::lock_guard<std::mutex> lock(g_delta_values_mutex);
    g_delta_values.push_back(delta);
  }

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
  require(routing::PathfinderOptions{}.delta == 1.0f,
          "default delta-stepping bucket width must be one");
  require(!routing::PathfinderOptions{}.delta_auto,
          "automatic delta must remain an explicit opt-in");
  require(!routing::PathfinderOptions{}.delta_force_generic,
          "generic Delta forcing must remain disabled by default");
  require(!routing::PathfinderOptions{}.delta_telemetry,
          "Delta telemetry must remain disabled by default");
  const routing::PathfinderOptions legacy_positional_options{
      routing::SsspEngine::kDeltaStep, 2.0f, 7, true, 3, 4, 5};
  require(legacy_positional_options.max_sssp_iterations == 7 &&
              legacy_positional_options.delta_force_legacy_parent &&
              legacy_positional_options.capacity == 3 &&
              legacy_positional_options.net_limit == 4 &&
              legacy_positional_options.parallel_net_workers == 5 &&
              !legacy_positional_options.delta_auto &&
              legacy_positional_options.delta_multiplier == 1.0f &&
              !legacy_positional_options.delta_force_generic &&
              !legacy_positional_options.delta_telemetry,
          "new delta controls must preserve legacy aggregate field positions");

  HostCsrF32 delta_stats_graph;
  delta_stats_graph.rows = 4;
  delta_stats_graph.cols = 4;
  delta_stats_graph.nnz = 2;
  delta_stats_graph.rowptr = {0, 2, 2, 2, 2};
  delta_stats_graph.colind = {1, 2};
  delta_stats_graph.values = {2.0f, 6.0f};
  require(delta_stepping_auto_delta(delta_stats_graph, 64) == 512.0f,
          "automatic delta must include isolated rows in average degree");
  require(delta_stepping_auto_delta(delta_stats_graph, 32) == 256.0f,
          "automatic delta must use the runtime wavefront size");
  require(delta_stepping_auto_delta(delta_stats_graph, 64, 0.25f) ==
              128.0f,
          "automatic delta must apply its sweep multiplier to the seed");

  delta_stats_graph.values = {0.0f, 6.0f};
  require(delta_stepping_auto_delta(delta_stats_graph, 64) == 384.0f,
          "zero-weight edges must still contribute to average degree");
  delta_stats_graph.values = {2.0f, 6.0f};
  const std::vector<float> destination_costs = {1000.0f, 2.0f, 4.0f, 1.0f};
  require(delta_stepping_auto_delta(delta_stats_graph,
                                    destination_costs,
                                    64) == 1792.0f,
          "automatic delta must apply costs at edge destinations");
  require(delta_stepping_auto_delta(
              delta_stats_graph, std::vector<float>(4, 0.0f), 64, 0.5f) ==
              0.5f,
          "all-zero effective weights must use the documented positive fallback");

  HostCsrF32 empty_delta_graph;
  empty_delta_graph.rows = 4;
  empty_delta_graph.cols = 4;
  empty_delta_graph.nnz = 0;
  empty_delta_graph.rowptr = {0, 0, 0, 0, 0};
  require(delta_stepping_auto_delta(empty_delta_graph, 64, 2.0f) == 2.0f,
          "an edgeless graph must use the positive fallback and multiplier");
  require(delta_stepping_auto_delta(
              empty_delta_graph,
              64,
              std::numeric_limits<float>::denorm_min()) ==
              std::numeric_limits<float>::min(),
          "automatic delta must clamp tiny widths above device flush-to-zero");

  delta_stats_graph.values.assign(
      2, std::numeric_limits<float>::max());
  const std::vector<float> maximum_costs(
      4, std::numeric_limits<float>::max());
  require(delta_stepping_auto_delta(delta_stats_graph, maximum_costs, 64) ==
              std::numeric_limits<float>::max(),
          "automatic delta must clamp overflowing effective-weight seeds");

  auto require_auto_delta_rejects = [&](int wavefront, float multiplier) {
    bool rejected = false;
    try {
      (void)delta_stepping_auto_delta(
          empty_delta_graph, wavefront, multiplier);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected,
            "automatic delta must reject an invalid wavefront or multiplier");
  };
  require_auto_delta_rejects(0, 1.0f);
  require_auto_delta_rejects(64, 0.0f);
  require_auto_delta_rejects(64, -1.0f);
  require_auto_delta_rejects(
      64, std::numeric_limits<float>::infinity());
  require_auto_delta_rejects(
      64, std::numeric_limits<float>::quiet_NaN());

  routing::PathfinderOptions parsed_delta_options;
  routing::parse_delta_arg("auto", &parsed_delta_options);
  require(parsed_delta_options.delta_auto,
          "--delta auto must select automatic mode");
  routing::parse_delta_arg("2.5", &parsed_delta_options);
  require(!parsed_delta_options.delta_auto &&
              parsed_delta_options.delta == 2.5f,
          "a numeric --delta must restore explicit override mode");
  bool invalid_delta_text_rejected = false;
  try {
    routing::parse_delta_arg("aut", &parsed_delta_options);
  } catch (const std::runtime_error&) {
    invalid_delta_text_rejected = true;
  }
  require(invalid_delta_text_rejected,
          "--delta must reject strings other than exact 'auto'");

  routing::PathfinderOptions invalid_multiplier_options;
  invalid_multiplier_options.sssp_engine = routing::SsspEngine::kDeltaStep;
  invalid_multiplier_options.delta_multiplier = 2.0f;
  bool explicit_multiplier_rejected = false;
  try {
    routing::validate_options(invalid_multiplier_options);
  } catch (const std::invalid_argument&) {
    explicit_multiplier_rejected = true;
  }
  require(explicit_multiplier_rejected,
          "a multiplier with explicit delta must not be silently ignored");

  for (const int invalid_control : {0, 1, 2, 3}) {
    routing::PathfinderOptions invalid_engine_options;
    if (invalid_control == 0) {
      invalid_engine_options.delta_force_generic = true;
    } else if (invalid_control == 1) {
      invalid_engine_options.delta_force_legacy_parent = true;
    } else if (invalid_control == 2) {
      invalid_engine_options.delta_controls_explicit = true;
    } else {
      invalid_engine_options.delta_telemetry = true;
    }
    bool rejected = false;
    try {
      routing::validate_options(invalid_engine_options);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected,
            "a non-Delta engine must reject Delta-specific controls");
  }

  HostCsrF32 benchmark_weights_graph = make_tree_graph();
  routing::apply_delta_benchmark_weights(
      benchmark_weights_graph,
      routing::DeltaBenchmarkWeights::kAllLight,
      2.0f,
      0);
  require(std::all_of(benchmark_weights_graph.values.begin(),
                      benchmark_weights_graph.values.end(),
                      [](float value) { return value == 0.5f; }),
          "all-light benchmark weights must be derived from numeric delta");
  routing::apply_delta_benchmark_weights(
      benchmark_weights_graph,
      routing::DeltaBenchmarkWeights::kAllHeavy,
      2.0f,
      0);
  require(std::all_of(benchmark_weights_graph.values.begin(),
                      benchmark_weights_graph.values.end(),
                      [](float value) { return value == 8.0f; }),
          "all-heavy benchmark weights must be derived from numeric delta");
  routing::apply_delta_benchmark_weights(
      benchmark_weights_graph,
      routing::DeltaBenchmarkWeights::kMixed,
      2.0f,
      17);
  const std::vector<float> first_mixed_weights = benchmark_weights_graph.values;
  routing::apply_delta_benchmark_weights(
      benchmark_weights_graph,
      routing::DeltaBenchmarkWeights::kMixed,
      2.0f,
      17);
  require(benchmark_weights_graph.values == first_mixed_weights,
          "mixed benchmark weights must be reproducible for a fixed seed");

  const HostCsrF32 graph = make_tree_graph();
  const std::vector<float> dist = cpu_dijkstra_outgoing_csr(graph, 0);
  const std::vector<routing::PathEdge> path =
      routing::reconstruct_shortest_path(graph, dist, 0, 2);
  require(path.size() == 2, "expected two edges in 0->2 reconstructed path");
  require(path[0].from == 0 && path[0].to == 1, "first path edge should be 0->1");
  require(path[1].from == 1 && path[1].to == 2, "second path edge should be 1->2");

  routing::RouteRequest detached_request;
  detached_request.sources.push_back({0, 0, 0});
  detached_request.sinks.push_back({2, 0, 0});
  DetachedCompactWorkspace detached_workspace;
  std::vector<std::uint32_t> detached_tree_seen(
      static_cast<std::size_t>(graph.rows), 0);
  std::vector<int> detached_parent_by_child(
      static_cast<std::size_t>(graph.rows), -1);
  std::vector<std::uint32_t> detached_parent_seen(
      static_cast<std::size_t>(graph.rows), 0);
  routing::PathfinderOptions detached_options;
  const routing::RoutedNet detached_net = routing::route_net(
      graph,
      detached_workspace,
      detached_request,
      detached_tree_seen,
      detached_parent_by_child,
      detached_parent_seen,
      1,
      detached_options,
      nullptr);
  require(!detached_net.reached_all_sinks,
          "compact path rooted outside the submitted source tree must be rejected");
  require(detached_net.sinks.size() == 1 &&
              !detached_net.sinks.front().reached &&
              detached_net.sinks.front().edges.empty(),
          "rejected detached compact path must not leak route edges");

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

  require(std::string(delta_stepping_execution_path_name(
              DeltaSteppingCsrExecutionPath::kExactUnit)) == "exact_unit" &&
              std::string(delta_stepping_execution_path_name(
                  DeltaSteppingCsrExecutionPath::kCompactGeneric)) ==
                  "compact_generic" &&
              std::string(delta_stepping_execution_path_name(
                  DeltaSteppingCsrExecutionPath::kLegacyGeneric)) ==
                  "legacy_generic" &&
              std::string(delta_stepping_execution_path_name(
                  DeltaSteppingCsrExecutionPath::kGenericDistancesOnly)) ==
                  "generic_distances_only",
          "Delta execution paths must have stable telemetry names");

  DeltaSteppingCsrWorkspace telemetry_workspace(congestion_graph, nullptr);
  DeltaSteppingCsrTelemetry disabled_telemetry;
  disabled_telemetry.collected = true;
  disabled_telemetry.completed = true;
  disabled_telemetry.execution_path =
      DeltaSteppingCsrExecutionPath::kGenericDistancesOnly;
  disabled_telemetry.outer_buckets_processed = 999;
  disabled_telemetry.current_queue_high_water = 999;
  (void)telemetry_workspace.run(std::vector<int>{0},
                                std::vector<int>{4},
                                2.5f,
                                -1,
                                nullptr,
                                nullptr,
                                nullptr);
  require(disabled_telemetry.collected && disabled_telemetry.completed &&
              disabled_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kGenericDistancesOnly &&
              disabled_telemetry.outer_buckets_processed == 999 &&
              disabled_telemetry.current_queue_high_water == 999,
          "a run without RunOptions must leave unrelated telemetry disabled");

  DeltaSteppingCsrTelemetry low_level_telemetry;
  low_level_telemetry.outer_buckets_processed = 999;
  low_level_telemetry.current_queue_high_water = 999;
  (void)telemetry_workspace.run(
      std::vector<int>{0},
      std::vector<int>{4},
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.collected && low_level_telemetry.completed &&
              low_level_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit &&
              low_level_telemetry.resolved_delta == 2.5f &&
              low_level_telemetry.wavefront_size == 64 &&
              !low_level_telemetry.force_generic &&
              !low_level_telemetry.force_legacy_parent &&
              !low_level_telemetry.has_vertex_costs &&
              low_level_telemetry.all_edges_light &&
              low_level_telemetry.outer_buckets_processed == 1 &&
              low_level_telemetry.current_queue_high_water == 19,
          "RunOptions must collect a reset exact-unit telemetry record");

  low_level_telemetry.outer_buckets_processed = 999;
  low_level_telemetry.pending_queue_high_water = 999;
  (void)telemetry_workspace.run(
      std::vector<int>{1},
      std::vector<int>{5},
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.outer_buckets_processed == 2 &&
              low_level_telemetry.pending_queue_high_water == 40,
          "reused RunOptions telemetry must reset instead of accumulating");

  (void)telemetry_workspace.run(
      std::vector<int>{0},
      4,
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kLegacyGeneric,
          "full-distance workspace telemetry must report the legacy path");

  telemetry_workspace.update_vertex_costs(
      std::vector<float>(static_cast<std::size_t>(congestion_graph.rows),
                         1.0f),
      nullptr);
  (void)telemetry_workspace.run(
      std::vector<int>{0},
      std::vector<int>{4},
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              low_level_telemetry.has_vertex_costs &&
              !low_level_telemetry.all_edges_light,
          "vertex costs must be represented in generic-path telemetry");

  DeltaSteppingCsrWorkspace forced_compact_workspace(
      congestion_graph,
      nullptr,
      DeltaSteppingCsrWorkspaceOptions{
          DeltaSteppingCsrParentMode::kAutomatic,
          DeltaSteppingCsrExecutionMode::kForceGeneric});
  (void)forced_compact_workspace.run(
      std::vector<int>{0},
      std::vector<int>{4},
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              low_level_telemetry.force_generic &&
              !low_level_telemetry.force_legacy_parent,
          "force-generic must be visible in compact telemetry");

  DeltaSteppingCsrWorkspace forced_legacy_workspace(
      congestion_graph,
      nullptr,
      DeltaSteppingCsrWorkspaceOptions{
          DeltaSteppingCsrParentMode::kForceLegacy,
          DeltaSteppingCsrExecutionMode::kForceGeneric});
  (void)forced_legacy_workspace.run(
      std::vector<int>{0},
      std::vector<int>{4},
      2.5f,
      -1,
      DeltaSteppingCsrRunOptions{&low_level_telemetry},
      nullptr,
      nullptr,
      nullptr);
  require(low_level_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kLegacyGeneric &&
              low_level_telemetry.force_generic &&
              low_level_telemetry.force_legacy_parent,
          "forced legacy parent selection must be visible in telemetry");

  const std::array<DeltaSteppingCsrExecutionPath, 4> aggregate_paths = {
      DeltaSteppingCsrExecutionPath::kExactUnit,
      DeltaSteppingCsrExecutionPath::kCompactGeneric,
      DeltaSteppingCsrExecutionPath::kLegacyGeneric,
      DeltaSteppingCsrExecutionPath::kGenericDistancesOnly};
  std::vector<DeltaSteppingCsrTelemetry> aggregate_records;
  for (std::size_t i = 0; i < aggregate_paths.size(); ++i) {
    DeltaSteppingCsrTelemetry record;
    populate_stub_delta_telemetry(record,
                                  aggregate_paths[i],
                                  congestion_graph,
                                  std::vector<int>{static_cast<int>(i)},
                                  2.5f,
                                  false,
                                  false,
                                  false);
    aggregate_records.push_back(record);
  }
  aggregate_records[1].completed = false;
  DeltaSteppingCsrTelemetry ignored_record;
  ignored_record.outer_buckets_processed = 100000;
  ignored_record.current_queue_high_water = 100000;
  aggregate_records.push_back(ignored_record);

  const routing::DeltaTelemetryTotals telemetry_totals =
      routing::aggregate_delta_telemetry(aggregate_records);
  require(telemetry_totals.queries == 4 &&
              telemetry_totals.completed_queries == 3 &&
              telemetry_totals.path_counts ==
                  std::array<std::uint64_t, 4>{1, 1, 1, 1},
          "telemetry aggregation must count collected, completed, and path records");
  const DeltaSteppingCsrTelemetry& telemetry_sums = telemetry_totals.sums;
  require(telemetry_sums.outer_buckets_processed == 10 &&
              telemetry_sums.light_relaxation_rounds == 20 &&
              telemetry_sums.heavy_edge_phases == 30 &&
              telemetry_sums.frontier_entries_processed == 40 &&
              telemetry_sums.active_vertices_processed == 50 &&
              telemetry_sums.stale_frontier_entries == 60 &&
              telemetry_sums.light_edge_visits == 70 &&
              telemetry_sums.heavy_edge_visits == 80 &&
              telemetry_sums.distance_atomic_attempts == 90 &&
              telemetry_sums.successful_distance_relaxations == 100 &&
              telemetry_sums.distance_cas_retries == 110 &&
              telemetry_sums.current_queue_insertions == 120 &&
              telemetry_sums.pending_queue_insertions == 130 &&
              telemetry_sums.heavy_queue_insertions == 140 &&
              telemetry_sums.bucket_insertions == 150 &&
              telemetry_sums.pending_entry_examinations == 160 &&
              telemetry_sums.stale_pending_entry_examinations == 170 &&
              telemetry_sums.reached_vertices == 180 &&
              telemetry_sums.controller_round_trips == 220 &&
              telemetry_sums.compact_parent_fallback_events == 230,
          "telemetry aggregation must sum every counter and ignore empty slots");
  require(telemetry_totals.current_queue_high_water == 76 &&
              telemetry_totals.pending_queue_high_water == 80 &&
              telemetry_totals.heavy_queue_high_water == 84,
          "telemetry aggregation must take maxima instead of summing peaks");

  routing::PathfinderOptions aggregate_json_options;
  aggregate_json_options.sssp_engine = routing::SsspEngine::kDeltaStep;
  aggregate_json_options.delta_auto = true;
  aggregate_json_options.delta_multiplier = 0.25f;
  aggregate_json_options.delta_force_generic = true;
  aggregate_json_options.delta_force_legacy_parent = true;
  const std::string aggregate_json = routing::delta_telemetry_aggregate_json(
      aggregate_records, aggregate_json_options, 2.5f, 64, 3);
  require(aggregate_json.find('\n') == std::string::npos &&
              aggregate_json.find("\"queries\":4") != std::string::npos &&
              aggregate_json.find("\"completed_queries\":3") !=
                  std::string::npos &&
              aggregate_json.find(
                  "\"execution_paths\":{\"exact_unit\":1,"
                  "\"compact_generic\":1,\"legacy_generic\":1,"
                  "\"generic_distances_only\":1}") != std::string::npos &&
              aggregate_json.find(
                  "\"maxima\":{\"current_queue_high_water\":76,"
                  "\"pending_queue_high_water\":80,"
                  "\"heavy_queue_high_water\":84}") != std::string::npos,
          "aggregate telemetry JSON must preserve stable counts and maxima");
  const std::filesystem::path aggregate_telemetry_path =
      "/tmp/pathfinder_delta_telemetry_aggregate.json";
  std::ofstream aggregate_telemetry_file(aggregate_telemetry_path);
  aggregate_telemetry_file << aggregate_json << '\n';
  require(static_cast<bool>(aggregate_telemetry_file),
          "aggregate telemetry JSON fixture must be writable");

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
  parallel_delta_options.delta = 2.5f;
  g_multisource_delta_calls = 0;
  g_unit_bfs_calls = 0;
  g_delta_graph_uploads = 0;
  clear_recorded_deltas();
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
  const std::vector<float> explicit_deltas = recorded_deltas();
  require(explicit_deltas.size() == 2 &&
              std::all_of(explicit_deltas.begin(),
                          explicit_deltas.end(),
                          [](float value) { return value == 2.5f; }),
          "an explicit numeric delta must reach every worker unchanged");

  routing::PathfinderOptions parallel_telemetry_options =
      parallel_delta_options;
  parallel_telemetry_options.delta_telemetry = true;
  routing::PathfinderResult parallel_telemetry_result;
  std::string parallel_telemetry_stdout;
  {
    ScopedCoutCapture capture;
    parallel_telemetry_result =
        routing::run_pathfinder(congestion_graph,
                                congestion_metadata,
                                parallel_telemetry_options,
                                nullptr);
    parallel_telemetry_stdout = capture.str();
  }
  require(parallel_telemetry_result.routed,
          "telemetry collection must not change parallel routing results");
  const std::string parallel_telemetry_json =
      single_delta_telemetry_json_line(parallel_telemetry_stdout);
  require(parallel_telemetry_json.find("\"queries\":2") !=
                  std::string::npos &&
              parallel_telemetry_json.find("\"completed_queries\":2") !=
                  std::string::npos &&
              parallel_telemetry_json.find("\"resolved_delta\":2.5") !=
                  std::string::npos &&
              parallel_telemetry_json.find("\"wavefront_size\":64") !=
                  std::string::npos &&
              parallel_telemetry_json.find("\"parallel_workers\":2") !=
                  std::string::npos &&
              parallel_telemetry_json.find(
                  "\"execution_paths\":{\"exact_unit\":2,"
                  "\"compact_generic\":0,\"legacy_generic\":0,"
                  "\"generic_distances_only\":0}") != std::string::npos &&
              parallel_telemetry_json.find(
                  "\"outer_buckets_processed\":3") != std::string::npos &&
              parallel_telemetry_json.find(
                  "\"controller_round_trips\":66") != std::string::npos &&
              parallel_telemetry_json.find(
                  "\"maxima\":{\"current_queue_high_water\":38,"
                  "\"pending_queue_high_water\":40,"
                  "\"heavy_queue_high_water\":42}") != std::string::npos,
          "parallel telemetry must aggregate isolated per-net slots once");
  const std::filesystem::path parallel_telemetry_path =
      "/tmp/pathfinder_delta_telemetry.json";
  std::ofstream parallel_telemetry_file(parallel_telemetry_path);
  parallel_telemetry_file << parallel_telemetry_json << '\n';
  require(static_cast<bool>(parallel_telemetry_file),
          "PathFinder telemetry JSON fixture must be writable");

  routing::PathfinderOptions forced_generic_options =
      parallel_delta_options;
  forced_generic_options.delta_force_generic = true;
  forced_generic_options.delta_force_legacy_parent = true;
  g_delta_force_generic_calls = 0;
  g_delta_force_legacy_calls = 0;
  const routing::PathfinderResult forced_generic_result =
      routing::run_pathfinder(congestion_graph,
                              congestion_metadata,
                              forced_generic_options,
                              nullptr);
  require(forced_generic_result.routed,
          "forced generic/legacy Delta routing must preserve results");
  require(g_delta_force_generic_calls == 2,
          "force-generic must reach every parallel Delta worker");
  require(g_delta_force_legacy_calls == 2,
          "force-legacy must reach every parallel Delta worker");

  routing::PathfinderOptions graph_aware_delta_options =
      parallel_delta_options;
  graph_aware_delta_options.delta_auto = true;
  graph_aware_delta_options.delta_multiplier = 0.25f;
  HostCsrF32 graph_aware_weighted_graph = congestion_graph;
  std::fill(graph_aware_weighted_graph.values.begin(),
            graph_aware_weighted_graph.values.end(),
            2.0f);
  const float expected_graph_aware_delta = delta_stepping_auto_delta(
      graph_aware_weighted_graph,
      64,
      graph_aware_delta_options.delta_multiplier);
  g_multisource_delta_calls = 0;
  g_delta_graph_uploads = 0;
  clear_recorded_deltas();
  const routing::PathfinderResult graph_aware_delta_result =
      routing::run_pathfinder(graph_aware_weighted_graph,
                              congestion_metadata,
                              graph_aware_delta_options,
                              nullptr);
  require(graph_aware_delta_result.routed,
          "graph-aware delta routing should preserve routed status");
  require(g_multisource_delta_calls == 2 && g_delta_graph_uploads == 1,
          "graph-aware delta must preserve shared parallel dispatch");
  const std::vector<float> automatic_deltas = recorded_deltas();
  require(automatic_deltas.size() == 2 &&
              std::all_of(automatic_deltas.begin(),
                          automatic_deltas.end(),
                          [expected_graph_aware_delta](float value) {
                            return value == expected_graph_aware_delta;
                          }),
          "automatic delta must resolve once and reach every worker identically");

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

  {
    const HostCsrF32 cached_graph =
        make_cached_multi_sink_counterexample_graph();
    const routing::RoutingMetadata cached_metadata =
        make_cached_multi_sink_counterexample_metadata(cached_graph);
    routing::PathfinderOptions cached_options;
    cached_options.parallel_net_workers = 1;
    routing::UnitBfsPathDiagnostic diagnostic;
    diagnostic.net_index = 0;
    diagnostic.sink_index = 1;
    g_unit_bfs_calls = 0;
    const routing::PathfinderResult cached_result =
        routing::run_pathfinder(cached_graph,
                                cached_metadata,
                                cached_options,
                                nullptr,
                                &diagnostic);

    require(cached_result.nets[0].sinks[0].nodes ==
                std::vector<int>({0, 1, 2}),
            "counterexample first sink should establish the expanded tree");
    require(cached_result.nets[0].sinks[1].nodes ==
                std::vector<int>({0, 3, 4}),
            "diagnostic must preserve the current cached adapter behavior");
    require(diagnostic.cpu_original_distance == 2 &&
                diagnostic.raw_batched.reached &&
                diagnostic.raw_batched.edge_count == 2 &&
                diagnostic.fresh_original.edge_count == 2,
            "raw UnitBFS and CPU reference should agree before tree growth");
    require(diagnostic.tree_source_count == 3 &&
                diagnostic.prior_sinks_reached == 1 &&
                diagnostic.cpu_expanded_tree_distance == 1 &&
                diagnostic.fresh_expanded_tree.reached &&
                diagnostic.fresh_expanded_tree.edge_count == 1,
            "fresh UnitBFS should find the one-edge expanded-tree route");
    require(diagnostic.attached_edge_count == 2 &&
                diagnostic.classification ==
                    "pathfinder_cached_multi_sink_path",
            "diagnostic should classify the stale cached multi-sink path");
    require(g_unit_bfs_calls == 3,
            "diagnostic should add exactly two selected-sink reference runs");
  }

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
