#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "../interchange/routing_csr_sidecars.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// BF12 executes independent multi-source/multi-target shortest-path queries in
// one GPU controller.  Query state is always addressed as q * V + vertex;
// immutable graph data and the dynamic-cost snapshot are the only shared data.

struct BellmanFord12BoundingBox {
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
  bool enabled = false;
};

struct BellmanFord12QueryDescriptor {
  std::uint32_t source_begin = 0;
  std::uint32_t source_count = 0;
  std::uint32_t target_begin = 0;
  std::uint32_t target_count = 0;
  BellmanFord12BoundingBox bounds{};
  // Negative selects V iterations.  Zero is useful only when every target is
  // also a source; otherwise the query reports an iteration-limit stop.
  int max_iterations = -1;
  // Zero inherits BellmanFord12BatchOptions::target_check_interval.
  int target_check_interval = 0;
  std::uint32_t original_query_index = 0;
};

enum class BellmanFord12ControllerMode {
  Auto,
  HostBatch,
  CooperativeBatch,
};

struct BellmanFord12BatchOptions {
  // 0 asks the conservative worker policy to select a maximum.  run_batch
  // still runs exactly the descriptors supplied by its caller and rejects a
  // batch that exceeds the safe selection.
  std::size_t requested_batch_size = 0;
  BellmanFord12ControllerMode controller_mode =
      BellmanFord12ControllerMode::Auto;
  int target_check_interval = 1;
  bool enable_bounding_boxes = true;
  bool enable_unbounded_retry = true;
  bool enable_telemetry = false;
  // Zero starts with a conservative demand-grown arena.
  std::size_t result_node_capacity = 0;
  std::size_t result_edge_capacity = 0;
  // Kept free when validating an automatic or explicit batch.  Zero selects
  // the policy default (25 percent of currently available device memory).
  std::size_t memory_safety_reserve_bytes = 0;
};

struct BellmanFord12QueryStatus {
  bool converged = false;
  bool stopped_on_target = false;
  // True only after frontier exhaustion or the exact nonnegative-distance
  // target certificate.  A finite label at an iteration limit is tentative.
  bool paths_certified = false;
  bool all_targets_reached = false;
  bool hit_iteration_limit = false;
  bool bounded_search_miss = false;
  // Terminal status only.  A recovered geometric arena retry is reported in
  // telemetry; every successfully returned query has this field false.
  bool result_arena_overflow = false;
  int iterations = 0;
};

struct BellmanFord12BatchTelemetry {
  bool enabled = false;
  std::size_t requested_batch_size = 0;
  std::size_t selected_batch_size = 0;
  std::size_t submitted_query_count = 0;
  std::uint64_t batch_count = 0;
  double batch_fill_ratio = 0.0;
  BellmanFord12ControllerMode controller_mode_used =
      BellmanFord12ControllerMode::HostBatch;
  std::uint64_t cooperative_controller_launch_count = 0;
  std::uint64_t host_fallback_round_count = 0;
  // Routing-epoch synchronizations: dynamic-cost upload, query-state growth,
  // host rounds, final result delivery, and overflow retries.  Graph/workspace
  // construction is intentionally outside this counter.
  std::uint64_t synchronization_count = 0;
  std::uint64_t global_traversal_rounds = 0;
  std::vector<std::uint32_t> active_queries_per_round;
  std::vector<std::uint64_t> combined_frontier_size_per_round;
  std::uint64_t edges_examined = 0;
  std::uint64_t successful_relaxations = 0;
  std::vector<std::uint64_t> touched_nodes_per_query;
  double maximum_touched_state_density = 0.0;
  // HIP-event device durations; zero when telemetry is disabled.
  std::uint64_t target_summary_nanoseconds = 0;
  std::uint64_t reconstruction_nanoseconds = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::size_t result_node_capacity = 0;
  std::size_t result_edge_capacity = 0;
  std::size_t result_nodes_used = 0;
  std::size_t result_edges_used = 0;
  std::uint64_t arena_overflow_retry_count = 0;
  std::uint64_t bounded_query_retry_count = 0;
  std::uint64_t total_batch_wall_nanoseconds = 0;
};

struct BellmanFord12BatchResult {
  // Results and statuses are returned in descriptor order, independent of
  // original_query_index.  Schedulers can restore global order with that field.
  std::vector<BellmanFordCsrResult> query_results;
  std::vector<BellmanFord12QueryStatus> query_statuses;
  std::vector<std::uint32_t> original_query_indices;
};

struct BellmanFord12MemoryEstimate {
  std::size_t graph_bytes = 0;
  std::size_t shared_dynamic_cost_bytes = 0;
  std::size_t fixed_query_state_bytes = 0;
  std::size_t result_arena_bytes = 0;
  std::size_t control_bytes = 0;
  std::size_t total_bytes = 0;
};

struct BellmanFord12CooperativeCapability {
  bool device_supports_cooperative_launch = false;
  bool default_stream_required = true;
  int device = -1;
  int compute_unit_count = 0;
  int legally_resident_blocks = 0;
  std::string architecture;
  std::string reason;
};

// Compatibility carrier for tests/tools that do not construct interchange
// routing sidecars.  Production callers should use RoutingCsrSidecars.
struct BellmanFord12NodeSidecars {
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_costs;
};

class BellmanFord12CsrGraph {
 public:
  struct Impl;

  BellmanFord12CsrGraph(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr);
  BellmanFord12CsrGraph(const HostCsrF32& adjacency,
                       const BellmanFord12NodeSidecars& sidecars,
                       hipStream_t stream = nullptr);
  ~BellmanFord12CsrGraph();

  BellmanFord12CsrGraph(const BellmanFord12CsrGraph&) = delete;
  BellmanFord12CsrGraph& operator=(const BellmanFord12CsrGraph&) = delete;
  BellmanFord12CsrGraph(BellmanFord12CsrGraph&&) noexcept;
  BellmanFord12CsrGraph& operator=(BellmanFord12CsrGraph&&) noexcept;

  std::size_t vertex_count() const noexcept;
  std::size_t edge_count() const noexcept;
  std::size_t device_memory_bytes() const noexcept;

 private:
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFord12CsrWorkspace;
};

class BellmanFord12CsrWorkspace {
 public:
  struct Impl;

  BellmanFord12CsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      // Zero means no caller-imposed cap beyond memory/index validation.
      std::size_t maximum_batch_size = 0,
      hipStream_t stream = nullptr);
  BellmanFord12CsrWorkspace(const HostCsrF32& adjacency,
                           const BellmanFord12NodeSidecars& sidecars,
                           std::size_t maximum_batch_size = 0,
                           hipStream_t stream = nullptr);
  explicit BellmanFord12CsrWorkspace(
      std::shared_ptr<const BellmanFord12CsrGraph> adjacency,
      std::size_t maximum_batch_size = 0,
      hipStream_t stream = nullptr);
  ~BellmanFord12CsrWorkspace();

  BellmanFord12CsrWorkspace(const BellmanFord12CsrWorkspace&) = delete;
  BellmanFord12CsrWorkspace& operator=(const BellmanFord12CsrWorkspace&) =
      delete;
  BellmanFord12CsrWorkspace(BellmanFord12CsrWorkspace&&) noexcept;
  BellmanFord12CsrWorkspace& operator=(BellmanFord12CsrWorkspace&&) noexcept;

  // Replace the one shared dynamic-cost snapshot.  No batch may be active.
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  BellmanFord12BatchResult run_batch(
      const std::vector<int>& flattened_sources,
      const std::vector<int>& flattened_targets,
      const std::vector<BellmanFord12QueryDescriptor>& descriptors,
      const BellmanFord12BatchOptions& options = {},
      hipStream_t stream = nullptr);

  // Convenience epoch entry point.  It uploads costs once, then submits the
  // batch while holding the workspace's exclusive operation lock.
  BellmanFord12BatchResult run_batch(
      const std::vector<float>& dynamic_vertex_costs,
      const std::vector<int>& flattened_sources,
      const std::vector<int>& flattened_targets,
      const std::vector<BellmanFord12QueryDescriptor>& descriptors,
      const BellmanFord12BatchOptions& options = {},
      hipStream_t stream = nullptr);

  // Set require_occupancy=false when cooperative execution will not be used;
  // this returns cached device properties without running kernel occupancy
  // discovery solely for an explicit host-batch schedule.
  BellmanFord12CooperativeCapability cooperative_capability(
      bool require_occupancy = true) const;
  bool cooperative_execution_supported() const;
  BellmanFord12MemoryEstimate estimate_memory(
      std::size_t batch_size,
      std::size_t result_node_capacity,
      std::size_t result_edge_capacity) const;
  std::size_t maximum_batch_size() const noexcept;
  BellmanFord12BatchTelemetry telemetry() const;
  void reset_telemetry();

 private:
  std::unique_ptr<Impl> impl_;
};

BellmanFord12MemoryEstimate estimate_bellman_ford12_memory(
    std::size_t vertex_count,
    std::size_t edge_count,
    std::size_t batch_size,
    std::size_t source_count,
    std::size_t target_count,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity,
    bool telemetry_enabled = false);

const char* bellman_ford12_controller_mode_name(
    BellmanFord12ControllerMode mode) noexcept;
