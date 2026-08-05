// Test-only host implementation of BF11's PathFinder-facing surface.  The
// production implementation is HIP-only; CPU PathFinder tests include this
// after defining their deterministic Dijkstra helpers.

std::atomic<std::uint64_t> g_bf11_stub_requested_workers{0};
std::atomic<std::uint64_t> g_bf11_stub_effective_workers{0};
std::atomic<std::uint64_t> g_bf11_stub_free_before_workers{0};
std::atomic<std::uint64_t> g_bf11_stub_workspace_constructions{0};
std::atomic<std::uint64_t> g_bf11_stub_telemetry_workspaces{0};
std::atomic<bool> g_bf11_stub_telemetry_enabled{false};
std::mutex g_bf11_stub_workspace_records_mutex;
std::vector<BellmanFord11WorkspaceOptions>
    g_bf11_stub_recorded_workspace_options;
std::vector<SsspQueryCapacityHints> g_bf11_stub_recorded_capacity_hints;
// Test-only failure injection. Deliberately not cleared by PathFinder's stats
// reset so a test can arm it before run_pathfinder constructs worker state.
std::atomic<std::uint64_t> g_bf11_stub_throw_on_workspace_construction{0};

void record_bf11_stub_workspace_construction(
    const BellmanFord11WorkspaceOptions& options,
    const SsspQueryCapacityHints& capacity_hints) {
  const std::uint64_t ordinal = ++g_bf11_stub_workspace_constructions;
  if (ordinal == g_bf11_stub_throw_on_workspace_construction.load()) {
    throw std::runtime_error("injected BF11 workspace construction failure");
  }
  if (options.telemetry) ++g_bf11_stub_telemetry_workspaces;
  std::lock_guard<std::mutex> lock(g_bf11_stub_workspace_records_mutex);
  g_bf11_stub_recorded_workspace_options.push_back(options);
  g_bf11_stub_recorded_capacity_hints.push_back(capacity_hints);
}

std::vector<BellmanFord11WorkspaceOptions>
recorded_bf11_stub_workspace_options() {
  std::lock_guard<std::mutex> lock(g_bf11_stub_workspace_records_mutex);
  return g_bf11_stub_recorded_workspace_options;
}

std::vector<SsspQueryCapacityHints> recorded_bf11_stub_capacity_hints() {
  std::lock_guard<std::mutex> lock(g_bf11_stub_workspace_records_mutex);
  return g_bf11_stub_recorded_capacity_hints;
}

void reset_bellman_ford11_runtime_stats() {
  g_bf11_stub_requested_workers = 0;
  g_bf11_stub_effective_workers = 0;
  g_bf11_stub_free_before_workers = 0;
  g_bf11_stub_workspace_constructions = 0;
  g_bf11_stub_telemetry_workspaces = 0;
  g_bf11_stub_telemetry_enabled = false;
  std::lock_guard<std::mutex> lock(g_bf11_stub_workspace_records_mutex);
  g_bf11_stub_recorded_workspace_options.clear();
  g_bf11_stub_recorded_capacity_hints.clear();
}

void configure_bellman_ford11_runtime_stats(
    bool telemetry_enabled,
    std::uint64_t requested_workers,
    std::uint64_t effective_workers,
    std::uint64_t gpu_free_before_workers) {
  g_bf11_stub_telemetry_enabled = telemetry_enabled;
  g_bf11_stub_requested_workers = requested_workers;
  g_bf11_stub_effective_workers = effective_workers;
  g_bf11_stub_free_before_workers = gpu_free_before_workers;
}

BellmanFord11RuntimeStats bellman_ford11_runtime_stats() {
  BellmanFord11RuntimeStats stats;
  stats.telemetry_enabled = g_bf11_stub_telemetry_enabled;
  stats.requested_workers = g_bf11_stub_requested_workers;
  stats.effective_workers = g_bf11_stub_effective_workers;
  stats.workspace_state_initializations =
      g_bf11_stub_workspace_constructions;
  stats.gpu_free_before_workers = g_bf11_stub_free_before_workers;
  return stats;
}

struct BellmanFord11CsrGraph::Impl {
  Impl(const HostCsrF32& adjacency, const std::vector<float>& base_costs)
      : graph(adjacency), base_vertex_cost(base_costs) {}

  HostCsrF32 graph;
  std::vector<float> base_vertex_cost;
};

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency,
                                   sidecars.base_vertex_cost)) {
  (void)stream;
}

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency,
                                   sidecars.base_vertex_costs)) {
  (void)stream;
}

BellmanFord11CsrGraph::~BellmanFord11CsrGraph() = default;
BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    BellmanFord11CsrGraph&&) noexcept = default;
BellmanFord11CsrGraph& BellmanFord11CsrGraph::operator=(
    BellmanFord11CsrGraph&&) noexcept = default;

struct BellmanFord11CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord11CsrGraph::Impl> graph;
  BellmanFord11WorkspaceOptions options;
  SsspQueryCapacityHints capacity_hints;
  std::vector<float> dynamic_vertex_cost;
};

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : BellmanFord11CsrWorkspace(adjacency, sidecars, stream, options,
                                SsspQueryCapacityHints{}) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options,
    SsspQueryCapacityHints capacity_hints)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  impl_->graph = std::make_shared<BellmanFord11CsrGraph::Impl>(
      adjacency, sidecars.base_vertex_cost);
  impl_->options = options;
  impl_->capacity_hints = capacity_hints;
  impl_->dynamic_vertex_cost.assign(static_cast<std::size_t>(adjacency.rows),
                                    1.0f);
  record_bf11_stub_workspace_construction(options, capacity_hints);
}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : BellmanFord11CsrWorkspace(adjacency, sidecars, stream, options,
                                SsspQueryCapacityHints{}) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options,
    SsspQueryCapacityHints capacity_hints)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  impl_->graph = std::make_shared<BellmanFord11CsrGraph::Impl>(
      adjacency, sidecars.base_vertex_costs);
  impl_->options = options;
  impl_->capacity_hints = capacity_hints;
  impl_->dynamic_vertex_cost.assign(static_cast<std::size_t>(adjacency.rows),
                                    1.0f);
  record_bf11_stub_workspace_construction(options, capacity_hints);
}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : BellmanFord11CsrWorkspace(std::move(adjacency), stream, options,
                                SsspQueryCapacityHints{}) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options,
    SsspQueryCapacityHints capacity_hints)
    : impl_(std::make_unique<Impl>()) {
  (void)stream;
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument("BF11 shared graph must not be null");
  }
  impl_->graph = adjacency->impl_;
  impl_->options = options;
  impl_->capacity_hints = capacity_hints;
  impl_->dynamic_vertex_cost.assign(
      static_cast<std::size_t>(impl_->graph->graph.rows), 1.0f);
  record_bf11_stub_workspace_construction(options, capacity_hints);
}

BellmanFord11CsrWorkspace::~BellmanFord11CsrWorkspace() = default;
BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    BellmanFord11CsrWorkspace&&) noexcept = default;
BellmanFord11CsrWorkspace& BellmanFord11CsrWorkspace::operator=(
    BellmanFord11CsrWorkspace&&) noexcept = default;

void BellmanFord11CsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  (void)stream;
  if (vertex_costs.size() != impl_->dynamic_vertex_cost.size()) {
    throw std::invalid_argument("BF11 vertex-cost vector must contain V entries");
  }
  impl_->dynamic_vertex_cost = vertex_costs;
}

void BellmanFord11CsrWorkspace::update_vertex_costs_sparse(
    const std::vector<int>& nodes,
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  (void)stream;
  if (nodes.size() != vertex_costs.size()) {
    throw std::invalid_argument("BF11 sparse cost nodes and values differ in size");
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    impl_->dynamic_vertex_cost.at(static_cast<std::size_t>(nodes[i])) =
        vertex_costs[i];
  }
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
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

  BellmanFordCsrResult result;
  const HostCsrF32& graph = impl_->graph->graph;
  HostCsrF32 weighted_graph = graph;
  for (int source = 0; source < graph.rows; ++source) {
    for (minplus_sparse::Offset edge =
             graph.rowptr[static_cast<std::size_t>(source)];
         edge < graph.rowptr[static_cast<std::size_t>(source + 1)]; ++edge) {
      const int destination = graph.colind[static_cast<std::size_t>(edge)];
      weighted_graph.values[static_cast<std::size_t>(edge)] *=
          impl_->graph->base_vertex_cost[static_cast<std::size_t>(destination)] *
          impl_->dynamic_vertex_cost[static_cast<std::size_t>(destination)];
    }
  }
  const CpuSsspResult cpu_result =
      cpu_dijkstra_outgoing_csr_multi(weighted_graph, sources);
  fill_compact_target_paths(weighted_graph, sources, targets, cpu_result, result);
  result.target_path_edge_costs.reserve(result.target_path_edges.size());
  for (const minplus_sparse::Offset edge : result.target_path_edges) {
    result.target_path_edge_costs.push_back(
        weighted_graph.values[static_cast<std::size_t>(edge)]);
  }
  result.target = -1;
  result.iterations_used = 1;
  result.converged = true;
  result.stopped_on_target = result.target_reached;
  return result;
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    const BellmanFord11RunOptions& run_options,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)run_options;
  return run(sources, targets, delta, max_iters, stream, progress_callback,
             progress_user_data);
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  BellmanFordCsrResult result =
      run(sources, std::vector<int>{target}, delta, max_iters, stream,
          progress_callback, progress_user_data);
  result.target = target;
  result.target_distance = result.target_distances.front();
  result.target_reached = std::isfinite(result.target_distance);
  return result;
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source}, target, delta, max_iters, stream,
             progress_callback, progress_user_data);
}
