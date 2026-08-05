#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "../interchange/routing_csr_sidecars.hpp"
#include "../sssp_query_capacity.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

// Compatibility carrier for low-level callers that do not construct the
// interchange sidecar artifact. Production routing should pass
// routing::interchange::RoutingCsrSidecars directly.
struct BellmanFord11NodeSidecars {
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_costs;
};

// Inclusive destination-node bounds. enabled=false retains the complete graph.
// In a bounded run, known-coordinate nodes must lie in this rectangle;
// missing-coordinate spill resources remain admissible for routing
// completeness. Sources and targets must always have known coordinates and be
// inside an enabled explicit box.
struct BellmanFord11BoundingBox {
  bool enabled = false;
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
};

struct BellmanFord11RunOptions {
  BellmanFord11BoundingBox bounds{};
  // The persistent controller evaluates its exact nonnegative-distance target
  // certificate after every N completed relaxation rounds. N must be positive;
  // larger values can only delay an early stop.
  int target_check_interval = 1;
};

enum class BellmanFord11HipGraphMode {
  // Use graph replay only when the runtime path has been validated as
  // supported and segment_rounds is greater than one. Any setup or launch
  // failure falls back to direct segmented enqueue.
  kAuto,
  // Request graph replay, retaining the same safe direct-enqueue fallback.
  kOn,
  // Always enqueue the segment's ordinary kernels directly.
  kOff,
};

struct BellmanFord11WorkspaceOptions {
  // The legacy-shaped run() overload is unbounded unless this is enabled.
  bool auto_bounds = false;
  // BF11 admits inclusive bounds, so 2/14 matches the integer coordinate
  // layers admitted by RWRoute's strict 3/15 bounding-box test.
  std::int32_t auto_margin_x = 2;
  std::int32_t auto_margin_y = 14;
  // Applies only to auto-bounded legacy-shaped runs. Explicit run options never
  // widen silently. Missing-coordinate route-tree sources are still seeded,
  // but a target without coordinates selects an unbounded first run. A bounded
  // miss retries once.
  bool unbounded_fallback = false;
  int target_check_interval = 1;
  // Collect aggregate phase/work telemetry. Disabled workspaces do not create
  // HIP events or execute telemetry counter operations. Keep this before newer
  // controls so historical positional aggregate initialization retains its
  // meaning.
  bool telemetry = false;
  // Explicit-stream controllers enqueue this many relaxation/finalize rounds
  // before copying controller status to the host. Supported values are
  // 1, 2, 4, 8, and 16; one is the compatibility/control path.
  int segment_rounds = 1;
  BellmanFord11HipGraphMode hip_graph_mode =
      BellmanFord11HipGraphMode::kAuto;
  // Select a dense state reset when the touched fraction is greater than or
  // equal to this value. The device makes the choice without a host readback.
  double adaptive_reset_threshold = 0.25;
};

// Process-wide aggregate for one benchmark/run interval. PathFinder resets it
// immediately before constructing BF11 workspaces and emits one JSON snapshot
// after all nets finish. The counters are atomic because net workers run on
// independent CPU threads.
struct BellmanFord11RuntimeStats {
  std::uint64_t persistent_controller_runs = 0;
  std::uint64_t host_controller_runs = 0;
  std::uint64_t target_checks = 0;
  std::uint64_t auto_unbounded_retries = 0;
  std::uint64_t sparse_state_resets = 0;
  std::uint64_t workspace_state_initializations = 0;
  std::uint64_t defensive_dense_state_resets = 0;
  bool telemetry_enabled = false;
  std::uint64_t requested_workers = 0;
  std::uint64_t effective_workers = 0;
  std::uint64_t telemetry_queries = 0;
  std::uint64_t telemetry_completed_queries = 0;
  std::uint64_t rounds = 0;
  std::uint64_t segments = 0;
  std::uint64_t no_op_segment_rounds = 0;
  std::uint64_t direct_segments = 0;
  std::uint64_t hip_graph_segments = 0;
  std::uint64_t status_copies = 0;
  std::uint64_t stream_synchronizations = 0;
  std::uint64_t graph_fallbacks = 0;
  std::uint64_t adaptive_dense_state_resets = 0;
  std::uint64_t constant_one_queries = 0;
  std::uint64_t static_cost_queries = 0;
  std::uint64_t dynamic_cost_queries = 0;
  std::uint64_t first_discoveries = 0;
  std::uint64_t mark_cas_attempts = 0;
  std::uint64_t mark_cas_wins = 0;
  std::uint64_t queue_reservations = 0;
  std::uint64_t bounded_fallbacks = 0;
  std::uint64_t avoided_failed_attempt_extractions = 0;
  std::uint64_t total_query_nanoseconds = 0;
  std::uint64_t reset_seed_gpu_nanoseconds = 0;
  std::uint64_t relaxation_gpu_nanoseconds = 0;
  std::uint64_t target_check_gpu_nanoseconds = 0;
  std::uint64_t iteration_status_copy_gpu_nanoseconds = 0;
  std::uint64_t stream_synchronize_cpu_nanoseconds = 0;
  std::uint64_t target_summary_gpu_nanoseconds = 0;
  std::uint64_t target_prefix_gpu_nanoseconds = 0;
  std::uint64_t path_reconstruction_gpu_nanoseconds = 0;
  std::uint64_t output_transfer_gpu_nanoseconds = 0;
  std::uint64_t iterations = 0;
  std::uint64_t frontier_vertices_processed = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t successful_relaxations = 0;
  std::uint64_t touched_vertices = 0;
  std::uint64_t maximum_touched_vertices = 0;
  double maximum_touched_fraction = 0.0;
  std::uint64_t workspace_device_bytes_total = 0;
  std::uint64_t workspace_device_bytes_per_worker_max = 0;
  std::uint64_t workspace_device_bytes_current_total = 0;
  std::uint64_t gpu_free_before_workers = 0;
  std::uint64_t gpu_free_after_workers = 0;
};

void reset_bellman_ford11_runtime_stats();
void configure_bellman_ford11_runtime_stats(
    bool telemetry_enabled,
    std::uint64_t requested_workers,
    std::uint64_t effective_workers,
    std::uint64_t gpu_free_before_workers);
BellmanFord11RuntimeStats bellman_ford11_runtime_stats();

class BellmanFord11CsrGraph {
 public:
  struct Impl;

  BellmanFord11CsrGraph(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr);
  BellmanFord11CsrGraph(const HostCsrF32& adjacency,
                       const BellmanFord11NodeSidecars& sidecars,
                       hipStream_t stream = nullptr);
  ~BellmanFord11CsrGraph();

  BellmanFord11CsrGraph(const BellmanFord11CsrGraph&) = delete;
  BellmanFord11CsrGraph& operator=(const BellmanFord11CsrGraph&) = delete;
  BellmanFord11CsrGraph(BellmanFord11CsrGraph&&) noexcept;
  BellmanFord11CsrGraph& operator=(BellmanFord11CsrGraph&&) noexcept;

 private:
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFord11CsrWorkspace;
};

class BellmanFord11CsrWorkspace {
 public:
  struct Impl;

  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream,
      BellmanFord11WorkspaceOptions options,
      SsspQueryCapacityHints capacity_hints);
  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const BellmanFord11NodeSidecars& sidecars,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const BellmanFord11NodeSidecars& sidecars,
      hipStream_t stream,
      BellmanFord11WorkspaceOptions options,
      SsspQueryCapacityHints capacity_hints);
  explicit BellmanFord11CsrWorkspace(
      std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  BellmanFord11CsrWorkspace(
      std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
      hipStream_t stream,
      BellmanFord11WorkspaceOptions options,
      SsspQueryCapacityHints capacity_hints);
  ~BellmanFord11CsrWorkspace();

  BellmanFord11CsrWorkspace(const BellmanFord11CsrWorkspace&) = delete;
  BellmanFord11CsrWorkspace& operator=(const BellmanFord11CsrWorkspace&) = delete;
  BellmanFord11CsrWorkspace(BellmanFord11CsrWorkspace&&) noexcept;
  BellmanFord11CsrWorkspace& operator=(BellmanFord11CsrWorkspace&&) noexcept;

  // Replace all workspace-local destination congestion multipliers. Costs are
  // copied and completed on the workspace stream before this method returns.
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  // Mutate only the listed workspace-local multipliers. Node IDs must be unique.
  void update_vertex_costs_sparse(const std::vector<int>& nodes,
                                  const std::vector<float>& vertex_costs,
                                  hipStream_t stream = nullptr);

  // Legacy-shaped entry point. It applies this workspace's auto-bound/fallback
  // policy. delta is retained for PathFinder interface compatibility and is not
  // used by Bellman-Ford.
  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  // Exact fixed-box entry point. This never performs an unbounded retry.
  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      const BellmanFord11RunOptions& run_options,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  BellmanFordCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
