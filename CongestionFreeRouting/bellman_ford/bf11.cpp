// Bounded, dynamically weighted, true-multi-source active-frontier
// Bellman-Ford for PathFinder. The default-stream cooperative controller keeps
// every relaxation round and target certificate on the GPU; explicit worker
// streams use the complete host-checked controller because a BF11 cooperative
// grid consumes full device residency on the target gfx1151 runtime.

#include "bf11.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::atomic<std::uint64_t> g_bf11_gpu_controller_launches{0};
std::atomic<std::uint64_t> g_bf11_controller_fallbacks{0};
std::atomic<std::uint64_t> g_bf11_target_checks{0};
std::atomic<std::uint64_t> g_bf11_auto_unbounded_retries{0};
std::atomic<std::uint64_t> g_bf11_sparse_state_resets{0};
std::atomic<std::uint64_t> g_bf11_workspace_state_initializations{0};
std::atomic<std::uint64_t> g_bf11_defensive_dense_state_resets{0};
std::atomic<bool> g_bf11_telemetry_enabled{false};
std::atomic<std::uint64_t> g_bf11_requested_workers{0};
std::atomic<std::uint64_t> g_bf11_effective_workers{0};
std::atomic<std::uint64_t> g_bf11_telemetry_queries{0};
std::atomic<std::uint64_t> g_bf11_telemetry_completed_queries{0};
std::atomic<std::uint64_t> g_bf11_total_query_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_reset_seed_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_relaxation_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_target_check_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_status_copy_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_stream_sync_cpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_target_summary_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_reconstruction_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_telemetry_iterations{0};
std::atomic<std::uint64_t> g_bf11_frontier_vertices_processed{0};
std::atomic<std::uint64_t> g_bf11_edges_examined{0};
std::atomic<std::uint64_t> g_bf11_successful_relaxations{0};
std::atomic<std::uint64_t> g_bf11_touched_vertices{0};
std::atomic<std::uint64_t> g_bf11_maximum_touched_vertices{0};
std::atomic<std::uint64_t> g_bf11_telemetry_graph_rows{0};
std::atomic<std::uint64_t> g_bf11_workspace_device_bytes_total{0};
std::atomic<std::uint64_t> g_bf11_workspace_device_bytes_per_worker_max{0};
std::atomic<std::uint64_t> g_bf11_gpu_free_before_workers{0};
std::atomic<std::uint64_t> g_bf11_gpu_free_after_workers{0};
std::atomic<std::uint64_t> g_bf11_constructed_workers{0};
std::atomic<std::uint64_t> g_bf11_iterations_per_host_check{1};
std::atomic<std::uint64_t> g_bf11_host_check_rounds{0};
std::atomic<std::uint64_t> g_bf11_iteration_status_copies{0};
std::atomic<std::uint64_t> g_bf11_iteration_control_stream_syncs{0};
std::atomic<std::uint64_t> g_bf11_device_iterations_enqueued{0};
std::atomic<std::uint64_t> g_bf11_device_iterations_executed{0};
std::atomic<std::uint64_t> g_bf11_terminal_noop_iterations{0};

void atomic_max(std::atomic<std::uint64_t>& destination,
                std::uint64_t value) {
  std::uint64_t observed = destination.load(std::memory_order_relaxed);
  while (observed < value &&
         !destination.compare_exchange_weak(observed, value,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
  }
}

void reset_bf11_telemetry_counters() {
  g_bf11_telemetry_enabled.store(false, std::memory_order_relaxed);
  g_bf11_requested_workers.store(0, std::memory_order_relaxed);
  g_bf11_effective_workers.store(0, std::memory_order_relaxed);
  g_bf11_telemetry_queries.store(0, std::memory_order_relaxed);
  g_bf11_telemetry_completed_queries.store(0, std::memory_order_relaxed);
  g_bf11_total_query_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_reset_seed_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_relaxation_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_target_check_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_status_copy_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_stream_sync_cpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_target_summary_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_reconstruction_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_telemetry_iterations.store(0, std::memory_order_relaxed);
  g_bf11_frontier_vertices_processed.store(0, std::memory_order_relaxed);
  g_bf11_edges_examined.store(0, std::memory_order_relaxed);
  g_bf11_successful_relaxations.store(0, std::memory_order_relaxed);
  g_bf11_touched_vertices.store(0, std::memory_order_relaxed);
  g_bf11_maximum_touched_vertices.store(0, std::memory_order_relaxed);
  g_bf11_telemetry_graph_rows.store(0, std::memory_order_relaxed);
  g_bf11_workspace_device_bytes_total.store(0, std::memory_order_relaxed);
  g_bf11_workspace_device_bytes_per_worker_max.store(
      0, std::memory_order_relaxed);
  g_bf11_gpu_free_before_workers.store(0, std::memory_order_relaxed);
  g_bf11_gpu_free_after_workers.store(0, std::memory_order_relaxed);
  g_bf11_constructed_workers.store(0, std::memory_order_relaxed);
  g_bf11_iterations_per_host_check.store(1, std::memory_order_relaxed);
  g_bf11_host_check_rounds.store(0, std::memory_order_relaxed);
  g_bf11_iteration_status_copies.store(0, std::memory_order_relaxed);
  g_bf11_iteration_control_stream_syncs.store(0,
                                               std::memory_order_relaxed);
  g_bf11_device_iterations_enqueued.store(0, std::memory_order_relaxed);
  g_bf11_device_iterations_executed.store(0, std::memory_order_relaxed);
  g_bf11_terminal_noop_iterations.store(0, std::memory_order_relaxed);
}

}  // namespace

void reset_bellman_ford11_runtime_stats() {
  g_bf11_gpu_controller_launches.store(0, std::memory_order_relaxed);
  g_bf11_controller_fallbacks.store(0, std::memory_order_relaxed);
  g_bf11_target_checks.store(0, std::memory_order_relaxed);
  g_bf11_auto_unbounded_retries.store(0, std::memory_order_relaxed);
  g_bf11_sparse_state_resets.store(0, std::memory_order_relaxed);
  g_bf11_workspace_state_initializations.store(0,
                                                std::memory_order_relaxed);
  g_bf11_defensive_dense_state_resets.store(0,
                                             std::memory_order_relaxed);
  reset_bf11_telemetry_counters();
}

void configure_bellman_ford11_runtime_stats(
    bool telemetry_enabled,
    std::uint64_t requested_workers,
    std::uint64_t effective_workers,
    std::uint64_t gpu_free_before_workers,
    std::uint64_t iterations_per_host_check) {
  g_bf11_telemetry_enabled.store(telemetry_enabled,
                                 std::memory_order_relaxed);
  g_bf11_requested_workers.store(requested_workers,
                                 std::memory_order_relaxed);
  g_bf11_effective_workers.store(effective_workers,
                                 std::memory_order_relaxed);
  g_bf11_gpu_free_before_workers.store(gpu_free_before_workers,
                                       std::memory_order_relaxed);
  g_bf11_gpu_free_after_workers.store(0, std::memory_order_relaxed);
  g_bf11_constructed_workers.store(0, std::memory_order_relaxed);
  g_bf11_iterations_per_host_check.store(iterations_per_host_check,
                                          std::memory_order_relaxed);
}

BellmanFord11RuntimeStats bellman_ford11_runtime_stats() {
  BellmanFord11RuntimeStats stats;
  stats.persistent_controller_runs =
      g_bf11_gpu_controller_launches.load(std::memory_order_relaxed);
  stats.host_controller_runs =
      g_bf11_controller_fallbacks.load(std::memory_order_relaxed);
  stats.target_checks = g_bf11_target_checks.load(std::memory_order_relaxed);
  stats.auto_unbounded_retries =
      g_bf11_auto_unbounded_retries.load(std::memory_order_relaxed);
  stats.sparse_state_resets =
      g_bf11_sparse_state_resets.load(std::memory_order_relaxed);
  stats.workspace_state_initializations =
      g_bf11_workspace_state_initializations.load(std::memory_order_relaxed);
  stats.defensive_dense_state_resets =
      g_bf11_defensive_dense_state_resets.load(std::memory_order_relaxed);
  stats.telemetry_enabled =
      g_bf11_telemetry_enabled.load(std::memory_order_relaxed);
  stats.requested_workers =
      g_bf11_requested_workers.load(std::memory_order_relaxed);
  stats.effective_workers =
      g_bf11_effective_workers.load(std::memory_order_relaxed);
  stats.telemetry_queries =
      g_bf11_telemetry_queries.load(std::memory_order_relaxed);
  stats.telemetry_completed_queries =
      g_bf11_telemetry_completed_queries.load(std::memory_order_relaxed);
  stats.total_query_nanoseconds =
      g_bf11_total_query_nanoseconds.load(std::memory_order_relaxed);
  stats.reset_seed_gpu_nanoseconds =
      g_bf11_reset_seed_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.relaxation_gpu_nanoseconds =
      g_bf11_relaxation_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.target_check_gpu_nanoseconds =
      g_bf11_target_check_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.iteration_status_copy_gpu_nanoseconds =
      g_bf11_status_copy_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.stream_synchronize_cpu_nanoseconds =
      g_bf11_stream_sync_cpu_nanoseconds.load(std::memory_order_relaxed);
  stats.target_summary_gpu_nanoseconds =
      g_bf11_target_summary_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.path_reconstruction_gpu_nanoseconds =
      g_bf11_reconstruction_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.iterations =
      g_bf11_telemetry_iterations.load(std::memory_order_relaxed);
  stats.frontier_vertices_processed =
      g_bf11_frontier_vertices_processed.load(std::memory_order_relaxed);
  stats.edges_examined = g_bf11_edges_examined.load(std::memory_order_relaxed);
  stats.successful_relaxations =
      g_bf11_successful_relaxations.load(std::memory_order_relaxed);
  stats.touched_vertices =
      g_bf11_touched_vertices.load(std::memory_order_relaxed);
  stats.maximum_touched_vertices =
      g_bf11_maximum_touched_vertices.load(std::memory_order_relaxed);
  const std::uint64_t rows =
      g_bf11_telemetry_graph_rows.load(std::memory_order_relaxed);
  if (rows != 0) {
    stats.maximum_touched_fraction =
        static_cast<double>(stats.maximum_touched_vertices) /
        static_cast<double>(rows);
  }
  stats.workspace_device_bytes_total =
      g_bf11_workspace_device_bytes_total.load(std::memory_order_relaxed);
  stats.workspace_device_bytes_per_worker_max =
      g_bf11_workspace_device_bytes_per_worker_max.load(
          std::memory_order_relaxed);
  stats.gpu_free_before_workers =
      g_bf11_gpu_free_before_workers.load(std::memory_order_relaxed);
  stats.gpu_free_after_workers =
      g_bf11_gpu_free_after_workers.load(std::memory_order_relaxed);
  stats.iterations_per_host_check =
      g_bf11_iterations_per_host_check.load(std::memory_order_relaxed);
  stats.host_check_rounds =
      g_bf11_host_check_rounds.load(std::memory_order_relaxed);
  stats.iteration_status_copies =
      g_bf11_iteration_status_copies.load(std::memory_order_relaxed);
  stats.stream_synchronizations_for_iteration_control =
      g_bf11_iteration_control_stream_syncs.load(std::memory_order_relaxed);
  stats.device_iterations_enqueued =
      g_bf11_device_iterations_enqueued.load(std::memory_order_relaxed);
  stats.device_iterations_executed =
      g_bf11_device_iterations_executed.load(std::memory_order_relaxed);
  stats.terminal_noop_iterations =
      g_bf11_terminal_noop_iterations.load(std::memory_order_relaxed);
  stats.speculative_iterations = stats.terminal_noop_iterations;
  return stats;
}

extern "C" void bf11_internal_reset_counters() {
  reset_bellman_ford11_runtime_stats();
}

extern "C" std::uint64_t bf11_internal_gpu_controller_launch_count() {
  return g_bf11_gpu_controller_launches.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_controller_fallback_count() {
  return g_bf11_controller_fallbacks.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_target_check_count() {
  return g_bf11_target_checks.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_auto_unbounded_retry_count() {
  return g_bf11_auto_unbounded_retries.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_sparse_state_reset_count() {
  return g_bf11_sparse_state_resets.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_dense_state_reset_count() {
  return g_bf11_defensive_dense_state_resets.load(
      std::memory_order_relaxed);
}

namespace rips_sssp_bf11 {

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;
using DeviceOffset = std::uint32_t;

constexpr unsigned int kNoPredecessor = 0xffffffffu;
constexpr unsigned int kInfinityBits = 0x7f800000u;
constexpr unsigned kGridX = 65535u;
constexpr int kBlockSize = 256;
constexpr std::int32_t kMissingCoordinate =
    routing::interchange::kMissingRouteCoordinate;

static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4 &&
                  std::numeric_limits<float>::is_iec559,
              "BF11 packed distance ordering requires IEEE-754 float32");
static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int),
              "BF11 packed predecessor state requires a 64-bit CAS word");

struct DeviceGraph {
  Offset rows = 0;
  Offset nnz = 0;
  const DeviceOffset* rowptr = nullptr;
  const Index* to = nullptr;
  const float* edge_values = nullptr;
  const std::int32_t* route_end_x = nullptr;
  const std::int32_t* route_end_y = nullptr;
  const float* base_vertex_cost = nullptr;
};

struct IterationStatus {
  int next_count = 0;
  int error_status = 0;
  int reached_target_count = 0;
  unsigned int min_next_frontier_dist_bits = kInfinityBits;
  unsigned int max_target_dist_bits = 0;
};

struct ControllerResult {
  int iterations_used = 0;
  int error_status = 0;
  int frontier_count = 0;
  int current_frontier_index = 0;
  int converged = 0;
  int early_stopped = 0;
  int hit_max_iters = 0;
  int done = 0;
  int target_checks = 0;
};

enum TargetPathStatus : int {
  kTargetUnreachable = 0,
  kTargetPathValid = 1,
  kTargetPathInvalid = 2,
};

struct TargetSummary {
  unsigned long long state = 0;
  unsigned long long node_count = 0;
  unsigned long long edge_count = 0;
  int root = -1;
  int status = kTargetUnreachable;
};

// The automatic worker policy charges these exact device-side layouts. Keep
// the implementation and its pre-construction memory estimate from silently
// diverging if a control/result record changes.
static_assert(sizeof(IterationStatus) == 20,
              "BF11 IterationStatus device-memory estimate changed");
static_assert(sizeof(ControllerResult) == 36,
              "BF11 ControllerResult device-memory estimate changed");
static_assert(sizeof(TargetSummary) == 32,
              "BF11 TargetSummary device-memory estimate changed");
static_assert(sizeof(Index) == 4 && sizeof(DeviceOffset) == 4 &&
                  sizeof(int) == 4 && sizeof(float) == 4,
              "BF11 per-vertex device-memory estimate changed");
static_assert(sizeof(unsigned long long) + 3 * sizeof(Index) + sizeof(int) +
                      sizeof(unsigned char) + sizeof(float) ==
                  29,
              "BF11 graph-sized workspace estimate changed");

struct DeviceTelemetryCounters {
  unsigned long long frontier_vertices_processed = 0;
  unsigned long long edges_examined = 0;
  unsigned long long successful_relaxations = 0;
  unsigned long long reset_seed_wall_ticks = 0;
  unsigned long long relaxation_wall_ticks = 0;
  unsigned long long target_check_wall_ticks = 0;
};

static_assert(sizeof(DeviceTelemetryCounters) == 48,
              "BF11 telemetry device-memory estimate changed");

struct TelemetryEventPair {
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  bool pending = false;
};

struct DeviceGraphOwner {
  DeviceGraph view{};
  DeviceOffset* rowptr = nullptr;
  Index* to = nullptr;
  float* edge_values = nullptr;
  std::int32_t* route_end_x = nullptr;
  std::int32_t* route_end_y = nullptr;
  float* base_vertex_cost = nullptr;
};

struct DeviceWorkspace {
  Offset rows = 0;
  hipStream_t stream = nullptr;
  int iterations_per_host_check = 1;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  int* next_marks = nullptr;
  unsigned char* source_mask = nullptr;
  // Query reset is sparse after construction. Every node whose packed state,
  // frontier mark, or source bit can differ from the default state appears
  // exactly once in touched_nodes[0:touched_count]. Sources publish themselves
  // while seeding; a non-source is published by the unique relaxation that
  // replaces its infinity label. Any future writer of those arrays must
  // preserve this completeness invariant.
  Index* touched_nodes = nullptr;
  int* touched_count = nullptr;
  float* dynamic_vertex_cost = nullptr;
  IterationStatus* iteration_status = nullptr;
  IterationStatus* host_iteration_status = nullptr;
  ControllerResult* controller_result = nullptr;
  ControllerResult* host_controller_result = nullptr;
  Index* source_nodes = nullptr;
  Offset source_capacity = 0;
  Index* target_nodes = nullptr;
  Offset target_capacity = 0;
  Index* update_nodes = nullptr;
  float* update_costs = nullptr;
  Offset update_capacity = 0;
  TargetSummary* target_summaries = nullptr;
  TargetSummary* host_target_summaries = nullptr;
  unsigned long long* reconstruction_node_offsets = nullptr;
  unsigned long long* reconstruction_edge_offsets = nullptr;
  Offset reconstruction_capacity = 0;
  Index* compact_nodes = nullptr;
  DeviceOffset* compact_edges = nullptr;
  float* compact_edge_costs = nullptr;
  std::size_t compact_node_capacity = 0;
  std::size_t compact_edge_capacity = 0;
  int cooperative_blocks = -1;
  // A failed asynchronous operation can leave the sparse-reset invariant
  // uncertain. Reuse then takes the defensive dense reset path once.
  bool needs_full_state_reset = false;
  bool telemetry_enabled = false;
  int wall_clock_rate_khz = 0;
  DeviceTelemetryCounters* telemetry_counters = nullptr;
  DeviceTelemetryCounters* host_telemetry_counters = nullptr;
  int* host_touched_count = nullptr;
  TelemetryEventPair reset_seed_events;
  TelemetryEventPair relaxation_events;
  TelemetryEventPair target_check_events;
  TelemetryEventPair status_copy_events;
  TelemetryEventPair target_summary_events;
  TelemetryEventPair reconstruction_events;
  std::vector<TelemetryEventPair> batched_relaxation_events;
  std::vector<TelemetryEventPair> batched_target_check_events;
};

struct SsspStatus {
  int iterations_used = 0;
  bool converged = false;
  bool early_stopped = false;
  bool hit_max_iters = false;
};

struct HostControllerCounterBatch {
  std::uint64_t target_checks = 0;
  std::uint64_t host_check_rounds = 0;
  std::uint64_t iteration_status_copies = 0;
  std::uint64_t iteration_control_stream_syncs = 0;
  std::uint64_t device_iterations_enqueued = 0;
  std::uint64_t device_iterations_executed = 0;
  std::uint64_t terminal_noop_iterations = 0;

  ~HostControllerCounterBatch() noexcept {
    add(g_bf11_target_checks, target_checks);
    add(g_bf11_host_check_rounds, host_check_rounds);
    add(g_bf11_iteration_status_copies, iteration_status_copies);
    add(g_bf11_iteration_control_stream_syncs,
        iteration_control_stream_syncs);
    add(g_bf11_device_iterations_enqueued, device_iterations_enqueued);
    add(g_bf11_device_iterations_executed, device_iterations_executed);
    add(g_bf11_terminal_noop_iterations, terminal_noop_iterations);
  }

 private:
  static void add(std::atomic<std::uint64_t>& destination,
                  std::uint64_t value) noexcept {
    if (value != 0) destination.fetch_add(value, std::memory_order_relaxed);
  }
};

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
  }
}

// HIP allocations and streams are device-affine. Destruction can happen after
// the caller has selected another device, so temporarily restore the recorded
// owner instead of issuing invalid frees on the caller's current device. A
// failed switch makes cleanup a deliberate leak rather than risking cross-
// device corruption; destructors cannot report a useful exception safely.
class ScopedOwningHipDevice {
 public:
  explicit ScopedOwningHipDevice(int owner) noexcept {
    if (hipGetDevice(&previous_) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
    if (previous_ == owner) {
      active_ = true;
      return;
    }
    if (hipSetDevice(owner) == hipSuccess) {
      active_ = true;
      restore_ = true;
    } else {
      (void)hipGetLastError();
    }
  }

  ~ScopedOwningHipDevice() noexcept {
    if (restore_ && hipSetDevice(previous_) != hipSuccess) {
      (void)hipGetLastError();
    }
  }

  bool active() const noexcept { return active_; }

 private:
  int previous_ = -1;
  bool active_ = false;
  bool restore_ = false;
};

class DrainStreamOnException {
 public:
  explicit DrainStreamOnException(hipStream_t stream)
      : stream_(stream), exceptions_(std::uncaught_exceptions()) {}
  ~DrainStreamOnException() noexcept {
    if (std::uncaught_exceptions() > exceptions_) {
      (void)hipStreamSynchronize(stream_);
    }
  }

 private:
  hipStream_t stream_ = nullptr;
  int exceptions_ = 0;
};

class ScopedQueryTelemetry {
 public:
  explicit ScopedQueryTelemetry(bool enabled)
      : enabled_(enabled) {
    if (enabled_) {
      begin_ = std::chrono::steady_clock::now();
      g_bf11_telemetry_queries.fetch_add(1, std::memory_order_relaxed);
    }
  }

  ~ScopedQueryTelemetry() {
    if (!enabled_) return;
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin_);
    g_bf11_total_query_nanoseconds.fetch_add(
        static_cast<std::uint64_t>(elapsed.count()),
        std::memory_order_relaxed);
    if (completed_) {
      g_bf11_telemetry_completed_queries.fetch_add(
          1, std::memory_order_relaxed);
    }
  }

  void mark_completed() noexcept { completed_ = true; }

 private:
  bool enabled_ = false;
  bool completed_ = false;
  std::chrono::steady_clock::time_point begin_;
};

dim3 grid_for_items(Offset items) {
  if (items <= 0) return dim3(1, 1);
  const Offset blocks = (items + kBlockSize - 1) / kBlockSize;
  const unsigned gx = blocks < static_cast<Offset>(kGridX)
                          ? static_cast<unsigned>(blocks)
                          : kGridX;
  const unsigned gy = static_cast<unsigned>(
      (blocks + static_cast<Offset>(gx) - 1) / static_cast<Offset>(gx));
  return dim3(gx, gy);
}

dim3 sparse_reset_grid(Offset rows) {
  constexpr Offset kMaxSparseResetBlocks = 256;
  const Offset required =
      std::max<Offset>(1, (rows + kBlockSize - 1) / kBlockSize);
  return dim3(static_cast<unsigned>(
      std::min(required, kMaxSparseResetBlocks)));
}

dim3 host_batched_relaxation_grid(Offset rows, int known_frontier_count) {
  constexpr Offset kMinimumBlocks = 32;
  constexpr Offset kMaximumBlocks = 256;
  // Only the first frontier size in a host window is visible to the CPU.
  // Keep enough resident work for later device-produced frontiers, and use the
  // kernel's grid-stride loop when they outgrow this bounded launch geometry.
  const Offset row_blocks =
      std::max<Offset>(1, (rows + kBlockSize - 1) / kBlockSize);
  const Offset known_blocks = known_frontier_count <= 0
      ? 1
      : (static_cast<Offset>(known_frontier_count) + kBlockSize - 1) /
            kBlockSize;
  const Offset blocks = std::min(
      row_blocks,
      std::min(kMaximumBlocks, std::max(kMinimumBlocks, known_blocks)));
  return dim3(static_cast<unsigned>(blocks));
}

template <typename T>
T* device_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflow");
  }
  T* result = nullptr;
  check_hip(hipMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)), what);
  return result;
}

template <typename T>
T* pinned_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflow");
  }
  T* result = nullptr;
  check_hip(hipHostMalloc(reinterpret_cast<void**>(&result),
                         count * sizeof(T), hipHostMallocDefault),
            what);
  return result;
}

void create_telemetry_event_pair(TelemetryEventPair& events) {
  check_hip(hipEventCreate(&events.begin), "create BF11 telemetry start event");
  try {
    check_hip(hipEventCreate(&events.end), "create BF11 telemetry stop event");
  } catch (...) {
    (void)hipEventDestroy(events.begin);
    events.begin = nullptr;
    throw;
  }
}

void destroy_telemetry_event_pair(TelemetryEventPair& events) noexcept {
  if (events.begin) (void)hipEventDestroy(events.begin);
  if (events.end) (void)hipEventDestroy(events.end);
  events = {};
}

void ensure_host_window_telemetry_events(DeviceWorkspace& workspace) {
  if (!workspace.telemetry_enabled ||
      workspace.iterations_per_host_check <= 1) {
    return;
  }
  const std::size_t event_count =
      static_cast<std::size_t>(workspace.iterations_per_host_check);
  if (workspace.batched_relaxation_events.size() == event_count &&
      workspace.batched_target_check_events.size() == event_count) {
    return;
  }
  if (!workspace.batched_relaxation_events.empty() ||
      !workspace.batched_target_check_events.empty()) {
    throw std::runtime_error("BF11 host-window telemetry event state is invalid");
  }
  try {
    workspace.batched_relaxation_events.resize(event_count);
    workspace.batched_target_check_events.resize(event_count);
    for (std::size_t i = 0; i < event_count; ++i) {
      create_telemetry_event_pair(workspace.batched_relaxation_events[i]);
      create_telemetry_event_pair(workspace.batched_target_check_events[i]);
    }
  } catch (...) {
    for (TelemetryEventPair& events : workspace.batched_relaxation_events) {
      destroy_telemetry_event_pair(events);
    }
    for (TelemetryEventPair& events : workspace.batched_target_check_events) {
      destroy_telemetry_event_pair(events);
    }
    workspace.batched_relaxation_events.clear();
    workspace.batched_target_check_events.clear();
    throw;
  }
}

void initialize_workspace_telemetry(DeviceWorkspace& workspace) {
  if (!workspace.telemetry_enabled) return;
  try {
    workspace.telemetry_counters = device_allocate<DeviceTelemetryCounters>(
        1, "hipMalloc BF11 telemetry counters");
    workspace.host_telemetry_counters = pinned_allocate<DeviceTelemetryCounters>(
        1, "hipHostMalloc BF11 telemetry counters");
    workspace.host_touched_count =
        pinned_allocate<int>(1, "hipHostMalloc BF11 telemetry touched count");
    create_telemetry_event_pair(workspace.reset_seed_events);
    create_telemetry_event_pair(workspace.relaxation_events);
    create_telemetry_event_pair(workspace.target_check_events);
    create_telemetry_event_pair(workspace.status_copy_events);
    create_telemetry_event_pair(workspace.target_summary_events);
    create_telemetry_event_pair(workspace.reconstruction_events);
    if (workspace.iterations_per_host_check > 1 &&
        workspace.stream != nullptr) {
      ensure_host_window_telemetry_events(workspace);
    }
    int device = -1;
    check_hip(hipGetDevice(&device), "get BF11 telemetry HIP device");
    check_hip(hipDeviceGetAttribute(&workspace.wall_clock_rate_khz,
                                    hipDeviceAttributeWallClockRate, device),
              "get BF11 telemetry wall-clock rate");
  } catch (...) {
    destroy_telemetry_event_pair(workspace.reset_seed_events);
    destroy_telemetry_event_pair(workspace.relaxation_events);
    destroy_telemetry_event_pair(workspace.target_check_events);
    destroy_telemetry_event_pair(workspace.status_copy_events);
    destroy_telemetry_event_pair(workspace.target_summary_events);
    destroy_telemetry_event_pair(workspace.reconstruction_events);
    for (TelemetryEventPair& events : workspace.batched_relaxation_events) {
      destroy_telemetry_event_pair(events);
    }
    for (TelemetryEventPair& events : workspace.batched_target_check_events) {
      destroy_telemetry_event_pair(events);
    }
    workspace.batched_relaxation_events.clear();
    workspace.batched_target_check_events.clear();
    if (workspace.telemetry_counters) (void)hipFree(workspace.telemetry_counters);
    if (workspace.host_telemetry_counters)
      (void)hipHostFree(workspace.host_telemetry_counters);
    if (workspace.host_touched_count)
      (void)hipHostFree(workspace.host_touched_count);
    workspace.telemetry_counters = nullptr;
    workspace.host_telemetry_counters = nullptr;
    workspace.host_touched_count = nullptr;
    throw;
  }
}

void begin_telemetry_event(DeviceWorkspace& workspace,
                           TelemetryEventPair& events) {
  if (!workspace.telemetry_enabled) return;
  check_hip(hipEventRecord(events.begin, workspace.stream),
            "record BF11 telemetry start event");
}

void end_telemetry_event(DeviceWorkspace& workspace,
                         TelemetryEventPair& events) {
  if (!workspace.telemetry_enabled) return;
  check_hip(hipEventRecord(events.end, workspace.stream),
            "record BF11 telemetry stop event");
  events.pending = true;
}

void accumulate_telemetry_event(TelemetryEventPair& events,
                                std::atomic<std::uint64_t>& destination) {
  if (!events.pending) return;
  float milliseconds = 0.0f;
  check_hip(hipEventElapsedTime(&milliseconds, events.begin, events.end),
            "measure BF11 telemetry event interval");
  if (milliseconds > 0.0f && std::isfinite(milliseconds)) {
    const double nanoseconds = static_cast<double>(milliseconds) * 1.0e6;
    destination.fetch_add(static_cast<std::uint64_t>(nanoseconds + 0.5),
                          std::memory_order_relaxed);
  }
  events.pending = false;
}

void synchronize_query_stream(DeviceWorkspace& workspace, const char* what) {
  if (!workspace.telemetry_enabled) {
    check_hip(hipStreamSynchronize(workspace.stream), what);
    return;
  }
  const auto begin = std::chrono::steady_clock::now();
  check_hip(hipStreamSynchronize(workspace.stream), what);
  const auto end = std::chrono::steady_clock::now();
  const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - begin);
  g_bf11_stream_sync_cpu_nanoseconds.fetch_add(
      static_cast<std::uint64_t>(nanoseconds.count()),
      std::memory_order_relaxed);
}

std::uint64_t wall_ticks_to_nanoseconds(unsigned long long ticks,
                                        int wall_clock_rate_khz) {
  if (ticks == 0 || wall_clock_rate_khz <= 0) return 0;
  const long double nanoseconds =
      static_cast<long double>(ticks) * 1.0e6L /
      static_cast<long double>(wall_clock_rate_khz);
  if (nanoseconds >=
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(nanoseconds + 0.5L);
}

void aggregate_query_work_telemetry(DeviceWorkspace& workspace,
                                    int iterations_used) {
  if (!workspace.telemetry_enabled) return;
  const DeviceTelemetryCounters counters = *workspace.host_telemetry_counters;
  g_bf11_telemetry_iterations.fetch_add(
      static_cast<std::uint64_t>(std::max(0, iterations_used)),
      std::memory_order_relaxed);
  g_bf11_frontier_vertices_processed.fetch_add(
      counters.frontier_vertices_processed, std::memory_order_relaxed);
  g_bf11_edges_examined.fetch_add(counters.edges_examined,
                                  std::memory_order_relaxed);
  g_bf11_successful_relaxations.fetch_add(
      counters.successful_relaxations, std::memory_order_relaxed);
  const int observed_touched = *workspace.host_touched_count;
  const std::uint64_t touched = observed_touched <= 0
      ? 0
      : std::min<std::uint64_t>(
            static_cast<std::uint64_t>(observed_touched),
            static_cast<std::uint64_t>(workspace.rows));
  g_bf11_touched_vertices.fetch_add(touched, std::memory_order_relaxed);
  atomic_max(g_bf11_maximum_touched_vertices, touched);
  g_bf11_telemetry_graph_rows.store(
      static_cast<std::uint64_t>(workspace.rows), std::memory_order_relaxed);
  g_bf11_reset_seed_gpu_nanoseconds.fetch_add(
      wall_ticks_to_nanoseconds(counters.reset_seed_wall_ticks,
                                workspace.wall_clock_rate_khz),
      std::memory_order_relaxed);
  g_bf11_relaxation_gpu_nanoseconds.fetch_add(
      wall_ticks_to_nanoseconds(counters.relaxation_wall_ticks,
                                workspace.wall_clock_rate_khz),
      std::memory_order_relaxed);
  g_bf11_target_check_gpu_nanoseconds.fetch_add(
      wall_ticks_to_nanoseconds(counters.target_check_wall_ticks,
                                workspace.wall_clock_rate_khz),
      std::memory_order_relaxed);
}

__device__ __forceinline__ Offset logical_thread_id() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

__host__ __device__ __forceinline__ unsigned long long pack_state(
    unsigned int distance_bits,
    unsigned int predecessor) {
  return (static_cast<unsigned long long>(distance_bits) << 32) |
         static_cast<unsigned long long>(predecessor);
}

__host__ __device__ __forceinline__ unsigned int state_distance_bits(
    unsigned long long state) {
  return static_cast<unsigned int>(state >> 32);
}

__device__ __forceinline__ float state_distance(unsigned long long state) {
  return __uint_as_float(state_distance_bits(state));
}

float host_state_distance(unsigned long long state) {
  const unsigned int bits = state_distance_bits(state);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

__device__ __forceinline__ bool finite_device(float value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

// Path reconstruction is tiny compared with traversal, so recover an edge's
// source from CSR row offsets instead of retaining a 4-byte source sidecar for
// every edge. This saves substantial device memory and upload bandwidth on the
// production graph while preserving original CSR edge IDs.
__device__ __forceinline__ Index source_for_edge(
    const DeviceOffset* rowptr,
    Offset rows,
    DeviceOffset edge) {
  Offset low = 0;
  Offset high = rows;
  while (low + 1 < high) {
    const Offset middle = low + (high - low) / 2;
    if (rowptr[middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return rowptr[low] <= edge && edge < rowptr[low + 1]
             ? static_cast<Index>(low)
             : static_cast<Index>(-1);
}

__device__ __forceinline__ bool node_admitted(
    const DeviceGraph& graph,
    Index node,
    const BellmanFord11BoundingBox& bounds) {
  if (!bounds.enabled) return true;
  const std::int32_t x = graph.route_end_x[node];
  const std::int32_t y = graph.route_end_y[node];
  if (x == kMissingCoordinate && y == kMissingCoordinate) {
    return true;
  }
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

// Distance-only comparison is deliberate. Equal-distance predecessor rewrites
// can create cycles around zero-cost edges and are irrelevant to SSSP distance
// correctness. The first strict label improvement therefore owns its parent.
struct AtomicRelaxResult {
  bool improved = false;
  bool first_discovery = false;
};

__device__ __forceinline__ unsigned long long coherent_atomic_load(
    unsigned long long* address) {
#if defined(__has_builtin)
#if !defined(BF11_FORCE_CAS_ATOMIC_LOAD) && __has_builtin(__hip_atomic_load)
  return __hip_atomic_load(address, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
#else
  return atomicCAS(address, 0ULL, 0ULL);
#endif
#else
  return atomicCAS(address, 0ULL, 0ULL);
#endif
}

__device__ __forceinline__ AtomicRelaxResult atomic_relax_strict(
    unsigned long long* address,
    float candidate,
    DeviceOffset predecessor) {
  const unsigned int candidate_bits = __float_as_uint(candidate);
  // Reused workspaces are reset by an earlier dispatch. Observe that reset
  // through the coherent atomic path so a cached finite label from the prior
  // query cannot suppress the first claim (and its touched-list publication).
  unsigned long long old_state = coherent_atomic_load(address);
  while (candidate_bits < state_distance_bits(old_state)) {
    const unsigned long long desired = pack_state(candidate_bits, predecessor);
    const unsigned long long assumed = old_state;
    old_state = atomicCAS(address, assumed, desired);
    if (old_state == assumed) {
      return {true, state_distance_bits(assumed) == kInfinityBits};
    }
  }
  return {};
}

__device__ __forceinline__ float effective_edge_weight(
    float edge_value,
    float base_cost,
    float dynamic_cost) {
  // Preserve the exact zero-multiplier contract without allowing an earlier
  // large-factor product to overflow to infinity and then form inf * 0.
  return edge_value == 0.0f || dynamic_cost == 0.0f
             ? 0.0f
             : (edge_value * base_cost) * dynamic_cost;
}

template <bool CollectTelemetry>
__device__ __forceinline__ unsigned int relax_edge(
    const DeviceGraph& graph,
    Offset edge,
    float from_distance,
    const float* __restrict__ dynamic_vertex_cost,
    const BellmanFord11BoundingBox& bounds,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    const unsigned char* __restrict__ source_mask,
    Index* __restrict__ touched_nodes,
    int* __restrict__ touched_count,
    IterationStatus* __restrict__ status,
    DeviceTelemetryCounters* __restrict__ telemetry) {
  const Index dst = graph.to[edge];
  if (dst < 0 || static_cast<Offset>(dst) >= graph.rows) {
    atomicExch(&status->error_status, 1);
    return kInfinityBits;
  }
  if (source_mask[dst] != 0 || !node_admitted(graph, dst, bounds)) {
    return kInfinityBits;
  }

  const float edge_value = graph.edge_values[edge];
  const float base_cost = graph.base_vertex_cost[dst];
  const float dynamic_cost = dynamic_vertex_cost[dst];
  const float effective_weight =
      effective_edge_weight(edge_value, base_cost, dynamic_cost);
  if (!finite_device(effective_weight) || effective_weight < 0.0f) {
    atomicExch(&status->error_status, 3);
    return kInfinityBits;
  }
  const float candidate = from_distance + effective_weight;
  if (!finite_device(candidate)) return kInfinityBits;

  const AtomicRelaxResult relaxation =
      atomic_relax_strict(&best_state[dst], candidate,
                          static_cast<DeviceOffset>(edge));
  if (!relaxation.improved) {
    return kInfinityBits;
  }
  if constexpr (CollectTelemetry) {
    atomicAdd(&telemetry->successful_relaxations, 1ULL);
  }
  if (relaxation.first_discovery) {
    const int touched_slot = atomicAdd(touched_count, 1);
    if (touched_slot < 0 ||
        static_cast<Offset>(touched_slot) >= graph.rows) {
      atomicExch(&status->error_status, 5);
      return kInfinityBits;
    }
    touched_nodes[touched_slot] = dst;
  }
  const int old_mark = atomicExch(&next_marks[dst], mark_token);
  if (old_mark != mark_token) {
    const int slot = atomicAdd(&status->next_count, 1);
    if (slot < 0 || static_cast<Offset>(slot) >= graph.rows) {
      atomicExch(&status->error_status, 4);
      return kInfinityBits;
    }
    next_frontier[slot] = dst;
  }
  return __float_as_uint(candidate);
}

__global__ void clear_state_kernel(Offset rows,
                                   unsigned long long* best_state,
                                   int* next_marks,
                                   unsigned char* source_mask) {
  const Offset row = logical_thread_id();
  if (row >= rows) return;
  best_state[row] = pack_state(kInfinityBits, kNoPredecessor);
  next_marks[row] = 0;
  source_mask[row] = 0;
}

__global__ void clear_touched_state_kernel(
    Offset rows,
    const Index* touched_nodes,
    const int* touched_count,
    unsigned long long* best_state,
    int* next_marks,
    unsigned char* source_mask) {
  __shared__ int shared_count;
  if (threadIdx.x == 0) {
    // The list is immutable for this kernel and the preceding work is ordered
    // on the same stream, so an atomic read-modify-write only adds contention.
    shared_count = *touched_count;
  }
  __syncthreads();
  const Offset thread = logical_thread_id();
  const Offset thread_count =
      static_cast<Offset>(gridDim.x) * static_cast<Offset>(gridDim.y) *
      static_cast<Offset>(blockDim.x);
  const int observed_count = shared_count;
  Offset count = observed_count <= 0 ? 0 : static_cast<Offset>(observed_count);
  if (count > rows) count = rows;
  for (Offset item = thread; item < count; item += thread_count) {
    const Index node = touched_nodes[item];
    if (node < 0 || static_cast<Offset>(node) >= rows) continue;
    best_state[node] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[node] = 0;
    source_mask[node] = 0;
  }
}

__global__ void seed_sources_kernel(const Index* source_nodes,
                                    Offset source_count,
                                    unsigned long long* best_state,
                                    Index* frontier,
                                    unsigned char* source_mask,
                                    Index* touched_nodes,
                                    int* touched_count) {
  const Offset item = logical_thread_id();
  if (item >= source_count) return;
  const Index source = source_nodes[item];
  best_state[source] = pack_state(0u, kNoPredecessor);
  // Sources are stable-deduplicated before upload, so this prefix contains no
  // duplicate touched entries and needs no queue-tail atomic.
  touched_nodes[item] = source;
  source_mask[source] = 1;
  frontier[item] = source;
  if (item == 0) *touched_count = static_cast<int>(source_count);
}

__global__ void reset_iteration_status_kernel(IterationStatus* status) {
  if (logical_thread_id() != 0) return;
  status->next_count = 0;
  status->error_status = 0;
  status->reached_target_count = 0;
  status->min_next_frontier_dist_bits = kInfinityBits;
  status->max_target_dist_bits = 0;
}

__global__ void initialize_host_window_controller_kernel(
    int source_count,
    IterationStatus* iteration_status,
    ControllerResult* controller_result) {
  if (logical_thread_id() != 0) return;
  iteration_status->next_count = 0;
  iteration_status->error_status = 0;
  iteration_status->reached_target_count = 0;
  iteration_status->min_next_frontier_dist_bits = kInfinityBits;
  iteration_status->max_target_dist_bits = 0;
  controller_result->iterations_used = 0;
  controller_result->error_status = 0;
  controller_result->frontier_count = source_count;
  controller_result->current_frontier_index = 0;
  controller_result->converged = 0;
  controller_result->early_stopped = 0;
  controller_result->hit_max_iters = 0;
  controller_result->done = 0;
  controller_result->target_checks = 0;
}

template <bool CollectTelemetry>
__global__ void frontier_relax_kernel(
    DeviceGraph graph,
    const Index* frontier,
    int frontier_count,
    int mark_token,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    Index* next_frontier,
    int* next_marks,
    const unsigned char* source_mask,
    Index* touched_nodes,
    int* touched_count,
    IterationStatus* status,
    DeviceTelemetryCounters* telemetry) {
  const Offset item = logical_thread_id();
  if (item >= static_cast<Offset>(frontier_count)) return;
  if constexpr (CollectTelemetry) {
    if (item == 0) {
      atomicAdd(&telemetry->frontier_vertices_processed,
                static_cast<unsigned long long>(frontier_count));
    }
  }
  const Index from = frontier[item];
  if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
    atomicExch(&status->error_status, 1);
    return;
  }
  const float from_dist = state_distance(best_state[from]);
  if (!finite_device(from_dist)) return;
  const Offset begin = static_cast<Offset>(graph.rowptr[from]);
  const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
  if (end < begin || end > graph.nnz) {
    atomicExch(&status->error_status, 2);
    return;
  }
  if constexpr (CollectTelemetry) {
    atomicAdd(&telemetry->edges_examined,
              static_cast<unsigned long long>(end - begin));
  }
  unsigned int local_min = kInfinityBits;
  for (Offset edge = begin; edge < end; ++edge) {
    const unsigned int candidate = relax_edge<CollectTelemetry>(
        graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
        best_state, next_frontier, next_marks, source_mask, touched_nodes,
        touched_count, status, telemetry);
    local_min = candidate < local_min ? candidate : local_min;
  }
  if (local_min != kInfinityBits) {
    atomicMin(&status->min_next_frontier_dist_bits, local_min);
  }
}

__global__ void update_target_status_kernel(
    const unsigned long long* best_state,
    const Index* target_nodes,
    Offset target_count,
    IterationStatus* status) {
  const Offset item = logical_thread_id();
  if (item >= target_count) return;
  const unsigned int bits = state_distance_bits(best_state[target_nodes[item]]);
  if (bits == kInfinityBits) return;
  atomicAdd(&status->reached_target_count, 1);
  atomicMax(&status->max_target_dist_bits, bits);
}

template <bool CollectTelemetry>
__global__ void host_window_frontier_relax_kernel(
    DeviceGraph graph,
    Index* frontier,
    Index* next_frontier,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    int* next_marks,
    const unsigned char* source_mask,
    Index* touched_nodes,
    int* touched_count,
    IterationStatus* status,
    const ControllerResult* controller_result,
    DeviceTelemetryCounters* telemetry) {
  if (controller_result->done != 0) return;
  const int frontier_count = controller_result->frontier_count;
  const int current_index = controller_result->current_frontier_index;
  const int mark_token = controller_result->iterations_used + 1;
  const Index* current = current_index == 0 ? frontier : next_frontier;
  Index* next = current_index == 0 ? next_frontier : frontier;
  const Offset thread = logical_thread_id();
  const Offset thread_count =
      static_cast<Offset>(gridDim.x) * static_cast<Offset>(gridDim.y) *
      static_cast<Offset>(blockDim.x);
  if constexpr (CollectTelemetry) {
    if (thread == 0) {
      atomicAdd(&telemetry->frontier_vertices_processed,
                static_cast<unsigned long long>(frontier_count));
    }
  }
  for (Offset item = thread; item < static_cast<Offset>(frontier_count);
       item += thread_count) {
    const Index from = current[item];
    if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
      atomicExch(&status->error_status, 1);
      continue;
    }
    const float from_dist = state_distance(best_state[from]);
    if (!finite_device(from_dist)) continue;
    const Offset begin = static_cast<Offset>(graph.rowptr[from]);
    const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
    if (end < begin || end > graph.nnz) {
      atomicExch(&status->error_status, 2);
      continue;
    }
    if constexpr (CollectTelemetry) {
      atomicAdd(&telemetry->edges_examined,
                static_cast<unsigned long long>(end - begin));
    }
    unsigned int local_min = kInfinityBits;
    for (Offset edge = begin; edge < end; ++edge) {
      const unsigned int candidate = relax_edge<CollectTelemetry>(
          graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
          best_state, next, next_marks, source_mask, touched_nodes,
          touched_count, status, telemetry);
      local_min = candidate < local_min ? candidate : local_min;
    }
    if (local_min != kInfinityBits) {
      atomicMin(&status->min_next_frontier_dist_bits, local_min);
    }
  }
}

__global__ void update_host_window_target_status_kernel(
    const unsigned long long* best_state,
    const Index* target_nodes,
    Offset target_count,
    IterationStatus* status,
    const ControllerResult* controller_result) {
  if (controller_result->done != 0 || status->error_status != 0) return;
  const Offset item = logical_thread_id();
  if (item >= target_count) return;
  const unsigned int bits = state_distance_bits(best_state[target_nodes[item]]);
  if (bits == kInfinityBits) return;
  atomicAdd(&status->reached_target_count, 1);
  atomicMax(&status->max_target_dist_bits, bits);
}

__global__ void advance_host_window_controller_kernel(
    IterationStatus* iteration_status,
    ControllerResult* controller_result,
    int target_count,
    int max_iters,
    int target_check_interval) {
  if (logical_thread_id() != 0 || controller_result->done != 0) return;
  const int completed_iteration = controller_result->iterations_used + 1;
  const bool check_targets =
      completed_iteration % target_check_interval == 0;
  controller_result->iterations_used = completed_iteration;
  controller_result->frontier_count = iteration_status->next_count;
  if (check_targets) ++controller_result->target_checks;
  if (iteration_status->error_status != 0) {
    controller_result->error_status = iteration_status->error_status;
    controller_result->done = 1;
  } else if (iteration_status->next_count == 0) {
    controller_result->converged = 1;
    controller_result->done = 1;
  } else if (check_targets &&
             iteration_status->reached_target_count == target_count &&
             iteration_status->min_next_frontier_dist_bits >=
                 iteration_status->max_target_dist_bits) {
    controller_result->early_stopped = 1;
    controller_result->done = 1;
  } else if (completed_iteration >= max_iters) {
    controller_result->hit_max_iters = 1;
    controller_result->done = 1;
  } else {
    controller_result->current_frontier_index ^= 1;
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
    iteration_status->reached_target_count = 0;
    iteration_status->min_next_frontier_dist_bits = kInfinityBits;
    iteration_status->max_target_dist_bits = 0;
  }
}

__global__ void sparse_cost_update_kernel(const Index* nodes,
                                          const float* costs,
                                          Offset count,
                                          float* dynamic_vertex_cost) {
  const Offset item = logical_thread_id();
  if (item < count) dynamic_vertex_cost[nodes[item]] = costs[item];
}

__global__ void fill_cost_kernel(Offset count, float* costs, float value) {
  const Offset item = logical_thread_id();
  if (item < count) costs[item] = value;
}

template <bool CollectTelemetry>
__global__ void frontier_controller_kernel(
    DeviceGraph graph,
    const Index* source_nodes,
    Offset source_count,
    const Index* target_nodes,
    Offset target_count,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    int max_iters,
    int target_check_interval,
    unsigned long long* best_state,
    Index* frontier,
    Index* next_frontier,
    int* next_marks,
    unsigned char* source_mask,
    Index* touched_nodes,
    int* touched_count,
    IterationStatus* iteration_status,
    ControllerResult* controller_result,
    DeviceTelemetryCounters* telemetry) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const Offset thread = static_cast<Offset>(grid.thread_rank());
  const Offset thread_count = static_cast<Offset>(grid.size());
  unsigned long long reset_seed_start = 0;
  if constexpr (CollectTelemetry) {
    if (thread == 0) reset_seed_start = wall_clock64();
  }

  // Snapshot before clearing: touched_count must remain stable until every
  // block has consumed the prior query's list. Reusing the iteration status as
  // initialization scratch is safe because the first round resets it below.
  if (thread == 0) {
    const int observed = *touched_count;
    Offset prior_count = observed <= 0 ? 0 : static_cast<Offset>(observed);
    if (prior_count > graph.rows) prior_count = graph.rows;
    iteration_status->next_count = static_cast<int>(prior_count);
  }
  grid.sync();
  const int prior_touched_count = iteration_status->next_count;
  for (Offset item = thread;
       item < static_cast<Offset>(prior_touched_count);
       item += thread_count) {
    const Index node = touched_nodes[item];
    if (node < 0 || static_cast<Offset>(node) >= graph.rows) continue;
    best_state[node] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[node] = 0;
    source_mask[node] = 0;
  }
  grid.sync();
  for (Offset item = thread; item < source_count; item += thread_count) {
    const Index source = source_nodes[item];
    best_state[source] = pack_state(0u, kNoPredecessor);
    source_mask[source] = 1;
    frontier[item] = source;
    touched_nodes[item] = source;
  }
  if (thread == 0) {
    *touched_count = static_cast<int>(source_count);
    controller_result->iterations_used = 0;
    controller_result->error_status = 0;
    controller_result->frontier_count = static_cast<int>(source_count);
    controller_result->current_frontier_index = 0;
    controller_result->converged = 0;
    controller_result->early_stopped = 0;
    controller_result->hit_max_iters = max_iters == 0 ? 1 : 0;
    controller_result->done = max_iters == 0 ? 1 : 0;
    controller_result->target_checks = 0;
  }
  grid.sync();
  if constexpr (CollectTelemetry) {
    if (thread == 0) {
      telemetry->reset_seed_wall_ticks += wall_clock64() - reset_seed_start;
    }
  }

  while (controller_result->done == 0) {
    if (thread == 0) {
      iteration_status->next_count = 0;
      iteration_status->error_status = 0;
      iteration_status->reached_target_count = 0;
      iteration_status->min_next_frontier_dist_bits = kInfinityBits;
      iteration_status->max_target_dist_bits = 0;
    }
    unsigned long long relaxation_start = 0;
    if constexpr (CollectTelemetry) {
      if (thread == 0) {
        relaxation_start = wall_clock64();
        telemetry->frontier_vertices_processed += static_cast<unsigned long long>(
            controller_result->frontier_count);
      }
    }
    // Gate the measured work after thread zero samples the device wall clock.
    // Sampling after this barrier lets other grid threads complete a small
    // frontier before the interval begins and undercounts the phase.
    grid.sync();

    const int frontier_count = controller_result->frontier_count;
    const int current_index = controller_result->current_frontier_index;
    const int completed_iteration = controller_result->iterations_used + 1;
    const int mark_token = completed_iteration;
    const Index* current = current_index == 0 ? frontier : next_frontier;
    Index* next = current_index == 0 ? next_frontier : frontier;

    for (Offset item = thread; item < static_cast<Offset>(frontier_count);
         item += thread_count) {
      const Index from = current[item];
      if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
        atomicExch(&iteration_status->error_status, 1);
        continue;
      }
      const float from_dist = state_distance(best_state[from]);
      if (!finite_device(from_dist)) continue;
      const Offset begin = static_cast<Offset>(graph.rowptr[from]);
      const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
      if (end < begin || end > graph.nnz) {
        atomicExch(&iteration_status->error_status, 2);
        continue;
      }
      if constexpr (CollectTelemetry) {
        atomicAdd(&telemetry->edges_examined,
                  static_cast<unsigned long long>(end - begin));
      }
      unsigned int local_min = kInfinityBits;
      for (Offset edge = begin; edge < end; ++edge) {
        const unsigned int candidate = relax_edge<CollectTelemetry>(
            graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
            best_state, next, next_marks, source_mask, touched_nodes,
            touched_count, iteration_status, telemetry);
        local_min = candidate < local_min ? candidate : local_min;
      }
      if (local_min != kInfinityBits) {
        atomicMin(&iteration_status->min_next_frontier_dist_bits, local_min);
      }
    }
    grid.sync();
    if constexpr (CollectTelemetry) {
      if (thread == 0) {
        telemetry->relaxation_wall_ticks +=
            wall_clock64() - relaxation_start;
      }
    }

    const bool check_targets =
        completed_iteration % target_check_interval == 0;
    unsigned long long target_check_start = 0;
    if constexpr (CollectTelemetry) {
      if (thread == 0 && check_targets) target_check_start = wall_clock64();
      if (check_targets) grid.sync();
    }
    if (check_targets) {
      for (Offset item = thread; item < target_count; item += thread_count) {
        const unsigned int bits =
            state_distance_bits(best_state[target_nodes[item]]);
        if (bits != kInfinityBits) {
          atomicAdd(&iteration_status->reached_target_count, 1);
          atomicMax(&iteration_status->max_target_dist_bits, bits);
        }
      }
    }
    grid.sync();
    if constexpr (CollectTelemetry) {
      if (thread == 0 && check_targets) {
        telemetry->target_check_wall_ticks +=
            wall_clock64() - target_check_start;
      }
    }

    if (thread == 0) {
      controller_result->iterations_used = completed_iteration;
      controller_result->frontier_count = iteration_status->next_count;
      if (check_targets) ++controller_result->target_checks;
      if (iteration_status->error_status != 0) {
        controller_result->error_status = iteration_status->error_status;
        controller_result->done = 1;
      } else if (iteration_status->next_count == 0) {
        controller_result->converged = 1;
        controller_result->done = 1;
      } else if (check_targets &&
                 iteration_status->reached_target_count ==
                     static_cast<int>(target_count) &&
                 iteration_status->min_next_frontier_dist_bits >=
                     iteration_status->max_target_dist_bits) {
        // Every future path must begin at a next-frontier label whose distance
        // is at least this lower bound. Nonnegative frozen edge costs
        // therefore certify all requested target distances exactly.
        controller_result->early_stopped = 1;
        controller_result->done = 1;
      } else if (completed_iteration >= max_iters) {
        controller_result->hit_max_iters = 1;
        controller_result->done = 1;
      } else {
        controller_result->current_frontier_index ^= 1;
      }
    }
    grid.sync();
  }
}

__global__ void summarize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* to,
    const DeviceOffset* rowptr,
    const Index* targets,
    Offset target_count,
    TargetSummary* summaries) {
  const Offset item = logical_thread_id();
  if (item >= target_count) return;
  TargetSummary summary{};
  const Index target = targets[item];
  summary.state = best_state[target];
  if (state_distance_bits(summary.state) == kInfinityBits) {
    summaries[item] = summary;
    return;
  }

  Index current = target;
  unsigned long long edge_count = 0;
  for (Offset guard = 0; guard <= rows; ++guard) {
    const unsigned long long state = best_state[current];
    const DeviceOffset edge = static_cast<DeviceOffset>(state);
    if (edge == kNoPredecessor) {
      if (source_mask[current] != 0 && state_distance_bits(state) == 0u) {
        summary.node_count = edge_count + 1;
        summary.edge_count = edge_count;
        summary.root = current;
        summary.status = kTargetPathValid;
      } else {
        summary.status = kTargetPathInvalid;
      }
      summaries[item] = summary;
      return;
    }
    if (static_cast<Offset>(edge) >= nnz || to[edge] != current) {
      summary.status = kTargetPathInvalid;
      summaries[item] = summary;
      return;
    }
    const Index predecessor = source_for_edge(rowptr, rows, edge);
    if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) {
      summary.status = kTargetPathInvalid;
      summaries[item] = summary;
      return;
    }
    current = predecessor;
    ++edge_count;
  }
  summary.status = kTargetPathInvalid;
  summaries[item] = summary;
}

__global__ void materialize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* to,
    const DeviceOffset* rowptr,
    const Index* targets,
    const TargetSummary* summaries,
    const unsigned long long* node_offsets,
    const unsigned long long* edge_offsets,
    Offset target_count,
    Index* compact_nodes,
    DeviceOffset* compact_edges,
    const float* edge_values,
    const float* base_vertex_cost,
    const float* dynamic_vertex_cost,
    float* compact_edge_costs) {
  const Offset item = logical_thread_id();
  if (item >= target_count ||
      summaries[item].status != kTargetPathValid) {
    return;
  }
  const TargetSummary summary = summaries[item];
  const unsigned long long node_base = node_offsets[item];
  const unsigned long long edge_base = edge_offsets[item];
  Index current = targets[item];
  compact_nodes[node_base + summary.edge_count] = current;
  for (unsigned long long remaining = summary.edge_count; remaining > 0;
       --remaining) {
    const DeviceOffset edge =
        static_cast<DeviceOffset>(best_state[current]);
    if (edge == kNoPredecessor || static_cast<Offset>(edge) >= nnz ||
        to[edge] != current) {
      return;
    }
    const Index predecessor = source_for_edge(rowptr, rows, edge);
    if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) return;
    compact_edges[edge_base + remaining - 1] = edge;
    compact_edge_costs[edge_base + remaining - 1] = effective_edge_weight(
        edge_values[edge], base_vertex_cost[current],
        dynamic_vertex_cost[current]);
    compact_nodes[node_base + remaining - 1] = predecessor;
    current = predecessor;
  }
  // summarize_target_paths_kernel already certified this condition. Keep the
  // reads here so malformed concurrent mutation cannot silently appear valid.
  if (current != summary.root || source_mask[current] == 0) return;
}

struct HostSidecarView {
  const std::vector<std::int32_t>* route_end_x = nullptr;
  const std::vector<std::int32_t>* route_end_y = nullptr;
  const std::vector<float>* base_vertex_cost = nullptr;
};

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols ||
      graph.rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::invalid_argument(
        "BF11 requires a nonempty square CSR graph with int vertex IDs");
  }
  if (graph.nnz < 0 ||
      static_cast<unsigned long long>(graph.nnz) >= kNoPredecessor) {
    throw std::invalid_argument(
        "BF11 CSR edge IDs must fit below its predecessor sentinel");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("BF11 CSR arrays do not match rows and nnz");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::invalid_argument("BF11 CSR row offsets are not monotone");
    }
  }
  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const Index dst = graph.colind[static_cast<std::size_t>(edge)];
    const float value = graph.values[static_cast<std::size_t>(edge)];
    if (dst < 0 || static_cast<Offset>(dst) >= graph.rows ||
        !std::isfinite(value) || value < 0.0f) {
      throw std::invalid_argument(
          "BF11 CSR destinations and weights must be valid and nonnegative");
    }
  }
}

void validate_sidecar_view(const HostSidecarView& view, Offset rows) {
  if (!view.route_end_x || !view.route_end_y || !view.base_vertex_cost ||
      view.route_end_x->size() != static_cast<std::size_t>(rows) ||
      view.route_end_y->size() != static_cast<std::size_t>(rows) ||
      view.base_vertex_cost->size() != static_cast<std::size_t>(rows)) {
    throw std::invalid_argument("BF11 node sidecars must contain exactly V entries");
  }
  for (Offset row = 0; row < rows; ++row) {
    const std::int32_t x = (*view.route_end_x)[static_cast<std::size_t>(row)];
    const std::int32_t y = (*view.route_end_y)[static_cast<std::size_t>(row)];
    const bool missing_x = x == kMissingCoordinate;
    const bool missing_y = y == kMissingCoordinate;
    const float base =
        (*view.base_vertex_cost)[static_cast<std::size_t>(row)];
    if (missing_x != missing_y || (!missing_x && (x < 0 || y < 0))) {
      throw std::invalid_argument("BF11 route-end coordinates are malformed");
    }
    if (!std::isfinite(base) || !(base > 0.0f)) {
      throw std::invalid_argument(
          "BF11 base vertex costs must be finite and positive");
    }
  }
}

HostSidecarView common_sidecar_view(
    const routing::interchange::RoutingCsrSidecars& sidecars,
    const HostCsrF32& graph) {
  routing::interchange::validate_routing_csr_sidecars(
      sidecars, static_cast<std::size_t>(graph.rows),
      static_cast<std::size_t>(graph.nnz), false);
  return {&sidecars.route_end_x, &sidecars.route_end_y,
          &sidecars.base_vertex_cost};
}

HostSidecarView compatibility_sidecar_view(
    const BellmanFord11NodeSidecars& sidecars,
    const HostCsrF32& graph) {
  HostSidecarView view{&sidecars.route_end_x, &sidecars.route_end_y,
                       &sidecars.base_vertex_costs};
  validate_sidecar_view(view, graph.rows);
  return view;
}

DeviceGraphOwner copy_graph_to_device(const HostCsrF32& graph,
                                      const HostSidecarView& sidecars,
                                      hipStream_t stream) {
  DeviceGraphOwner owner;
  DrainStreamOnException drain(stream);
  try {
    std::vector<DeviceOffset> compact_rowptr;
    compact_rowptr.reserve(graph.rowptr.size());
    for (const Offset offset : graph.rowptr) {
      if (offset < 0 ||
          static_cast<unsigned long long>(offset) >= kNoPredecessor) {
        throw std::overflow_error("BF11 row offset cannot fit uint32 storage");
      }
      compact_rowptr.push_back(static_cast<DeviceOffset>(offset));
    }
    owner.rowptr = device_allocate<DeviceOffset>(compact_rowptr.size(),
                                                  "hipMalloc BF11 rowptr");
    check_hip(hipMemcpyAsync(owner.rowptr, compact_rowptr.data(),
                             compact_rowptr.size() * sizeof(DeviceOffset),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 rowptr");
    if (graph.nnz > 0) {
      const std::size_t edges = static_cast<std::size_t>(graph.nnz);
      owner.to = device_allocate<Index>(edges, "hipMalloc BF11 destinations");
      owner.edge_values =
          device_allocate<float>(edges, "hipMalloc BF11 edge values");
      check_hip(hipMemcpyAsync(owner.to, graph.colind.data(),
                               edges * sizeof(Index), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 destinations");
      check_hip(hipMemcpyAsync(owner.edge_values, graph.values.data(),
                               edges * sizeof(float), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 edge values");
    }
    const std::size_t rows = static_cast<std::size_t>(graph.rows);
    owner.route_end_x =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end X");
    owner.route_end_y =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end Y");
    owner.base_vertex_cost =
        device_allocate<float>(rows, "hipMalloc BF11 base vertex costs");
    check_hip(hipMemcpyAsync(owner.route_end_x, sidecars.route_end_x->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end X");
    check_hip(hipMemcpyAsync(owner.route_end_y, sidecars.route_end_y->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end Y");
    check_hip(hipMemcpyAsync(owner.base_vertex_cost,
                             sidecars.base_vertex_cost->data(),
                             rows * sizeof(float), hipMemcpyHostToDevice, stream),
              "copy BF11 base vertex costs");
    check_hip(hipStreamSynchronize(stream), "synchronize BF11 graph upload");

    owner.view = {graph.rows,
                  graph.nnz,
                  owner.rowptr,
                  owner.to,
                  owner.edge_values,
                  owner.route_end_x,
                  owner.route_end_y,
                  owner.base_vertex_cost};
    return owner;
  } catch (...) {
    // An allocation failure can occur after earlier asynchronous copies were
    // queued. Drain them before releasing their destinations.
    (void)hipStreamSynchronize(stream);
    if (owner.rowptr) (void)hipFree(owner.rowptr);
    if (owner.to) (void)hipFree(owner.to);
    if (owner.edge_values) (void)hipFree(owner.edge_values);
    if (owner.route_end_x) (void)hipFree(owner.route_end_x);
    if (owner.route_end_y) (void)hipFree(owner.route_end_y);
    if (owner.base_vertex_cost) (void)hipFree(owner.base_vertex_cost);
    throw;
  }
}

void free_graph(DeviceGraphOwner* owner) noexcept {
  if (!owner) return;
  if (owner->rowptr) (void)hipFree(owner->rowptr);
  if (owner->to) (void)hipFree(owner->to);
  if (owner->edge_values) (void)hipFree(owner->edge_values);
  if (owner->route_end_x) (void)hipFree(owner->route_end_x);
  if (owner->route_end_y) (void)hipFree(owner->route_end_y);
  if (owner->base_vertex_cost) (void)hipFree(owner->base_vertex_cost);
  *owner = {};
}

DeviceWorkspace make_workspace(Offset rows,
                               hipStream_t stream,
                               bool telemetry_enabled,
                               int iterations_per_host_check) {
  DeviceWorkspace workspace;
  workspace.rows = rows;
  workspace.stream = stream;
  workspace.telemetry_enabled = telemetry_enabled;
  workspace.iterations_per_host_check = iterations_per_host_check;
  try {
    const std::size_t count = static_cast<std::size_t>(rows);
    workspace.best_state = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 packed states");
    workspace.frontier =
        device_allocate<Index>(count, "hipMalloc BF11 frontier");
    workspace.next_frontier =
        device_allocate<Index>(count, "hipMalloc BF11 next frontier");
    workspace.next_marks =
        device_allocate<int>(count, "hipMalloc BF11 frontier marks");
    workspace.source_mask = device_allocate<unsigned char>(
        count, "hipMalloc BF11 source mask");
    workspace.touched_nodes =
        device_allocate<Index>(count, "hipMalloc BF11 touched nodes");
    workspace.touched_count =
        device_allocate<int>(1, "hipMalloc BF11 touched count");
    workspace.dynamic_vertex_cost = device_allocate<float>(
        count, "hipMalloc BF11 dynamic vertex costs");
    workspace.iteration_status = device_allocate<IterationStatus>(
        1, "hipMalloc BF11 iteration status");
    workspace.host_iteration_status = pinned_allocate<IterationStatus>(
        1, "hipHostMalloc BF11 iteration status");
    workspace.controller_result = device_allocate<ControllerResult>(
        1, "hipMalloc BF11 controller result");
    workspace.host_controller_result = pinned_allocate<ControllerResult>(
        1, "hipHostMalloc BF11 controller result");
    // Pay the dense state initialization once per workspace. Successful query
    // reuse clears only the nodes recorded in touched_nodes.
    hipLaunchKernelGGL(clear_state_kernel, grid_for_items(rows),
                       dim3(kBlockSize), 0, stream, rows,
                       workspace.best_state, workspace.next_marks,
                       workspace.source_mask);
    check_hip(hipGetLastError(), "initialize BF11 search state");
    check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int), stream),
              "initialize BF11 touched count");
    hipLaunchKernelGGL(fill_cost_kernel, grid_for_items(rows), dim3(kBlockSize),
                       0, stream, rows, workspace.dynamic_vertex_cost, 1.0f);
    check_hip(hipGetLastError(), "initialize BF11 dynamic vertex costs");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF11 workspace initialization");
    initialize_workspace_telemetry(workspace);
    g_bf11_workspace_state_initializations.fetch_add(
        1, std::memory_order_relaxed);
    return workspace;
  } catch (...) {
    // The initialization kernel uses the caller's stream. If an asynchronous
    // launch or synchronization failure surfaced above, drain best-effort
    // before freeing any destination allocation.
    (void)hipStreamSynchronize(stream);
    if (workspace.best_state) (void)hipFree(workspace.best_state);
    if (workspace.frontier) (void)hipFree(workspace.frontier);
    if (workspace.next_frontier) (void)hipFree(workspace.next_frontier);
    if (workspace.next_marks) (void)hipFree(workspace.next_marks);
    if (workspace.source_mask) (void)hipFree(workspace.source_mask);
    if (workspace.touched_nodes) (void)hipFree(workspace.touched_nodes);
    if (workspace.touched_count) (void)hipFree(workspace.touched_count);
    if (workspace.dynamic_vertex_cost)
      (void)hipFree(workspace.dynamic_vertex_cost);
    if (workspace.iteration_status) (void)hipFree(workspace.iteration_status);
    if (workspace.host_iteration_status)
      (void)hipHostFree(workspace.host_iteration_status);
    if (workspace.controller_result) (void)hipFree(workspace.controller_result);
    if (workspace.host_controller_result)
      (void)hipHostFree(workspace.host_controller_result);
    throw;
  }
}

void free_workspace(DeviceWorkspace* workspace) noexcept {
  if (!workspace) return;
  (void)hipStreamSynchronize(workspace->stream);
  destroy_telemetry_event_pair(workspace->reset_seed_events);
  destroy_telemetry_event_pair(workspace->relaxation_events);
  destroy_telemetry_event_pair(workspace->target_check_events);
  destroy_telemetry_event_pair(workspace->status_copy_events);
  destroy_telemetry_event_pair(workspace->target_summary_events);
  destroy_telemetry_event_pair(workspace->reconstruction_events);
  for (TelemetryEventPair& events : workspace->batched_relaxation_events) {
    destroy_telemetry_event_pair(events);
  }
  for (TelemetryEventPair& events : workspace->batched_target_check_events) {
    destroy_telemetry_event_pair(events);
  }
  if (workspace->best_state) (void)hipFree(workspace->best_state);
  if (workspace->frontier) (void)hipFree(workspace->frontier);
  if (workspace->next_frontier) (void)hipFree(workspace->next_frontier);
  if (workspace->next_marks) (void)hipFree(workspace->next_marks);
  if (workspace->source_mask) (void)hipFree(workspace->source_mask);
  if (workspace->touched_nodes) (void)hipFree(workspace->touched_nodes);
  if (workspace->touched_count) (void)hipFree(workspace->touched_count);
  if (workspace->dynamic_vertex_cost)
    (void)hipFree(workspace->dynamic_vertex_cost);
  if (workspace->iteration_status) (void)hipFree(workspace->iteration_status);
  if (workspace->host_iteration_status)
    (void)hipHostFree(workspace->host_iteration_status);
  if (workspace->controller_result) (void)hipFree(workspace->controller_result);
  if (workspace->host_controller_result)
    (void)hipHostFree(workspace->host_controller_result);
  if (workspace->source_nodes) (void)hipFree(workspace->source_nodes);
  if (workspace->target_nodes) (void)hipFree(workspace->target_nodes);
  if (workspace->update_nodes) (void)hipFree(workspace->update_nodes);
  if (workspace->update_costs) (void)hipFree(workspace->update_costs);
  if (workspace->target_summaries) (void)hipFree(workspace->target_summaries);
  if (workspace->host_target_summaries)
    (void)hipHostFree(workspace->host_target_summaries);
  if (workspace->reconstruction_node_offsets)
    (void)hipFree(workspace->reconstruction_node_offsets);
  if (workspace->reconstruction_edge_offsets)
    (void)hipFree(workspace->reconstruction_edge_offsets);
  if (workspace->compact_nodes) (void)hipFree(workspace->compact_nodes);
  if (workspace->compact_edges) (void)hipFree(workspace->compact_edges);
  if (workspace->compact_edge_costs)
    (void)hipFree(workspace->compact_edge_costs);
  if (workspace->telemetry_counters)
    (void)hipFree(workspace->telemetry_counters);
  if (workspace->host_telemetry_counters)
    (void)hipHostFree(workspace->host_telemetry_counters);
  if (workspace->host_touched_count)
    (void)hipHostFree(workspace->host_touched_count);
  *workspace = {};
}

std::uint64_t workspace_device_bytes(const DeviceWorkspace& workspace) {
  std::uint64_t bytes = 0;
  const auto add = [&](std::uint64_t count, std::size_t element_size) {
    if (count > (std::numeric_limits<std::uint64_t>::max() - bytes) /
                    element_size) {
      throw std::overflow_error("BF11 workspace byte count overflow");
    }
    bytes += count * static_cast<std::uint64_t>(element_size);
  };
  const std::uint64_t rows = static_cast<std::uint64_t>(workspace.rows);
  if (workspace.best_state) add(rows, sizeof(*workspace.best_state));
  if (workspace.frontier) add(rows, sizeof(*workspace.frontier));
  if (workspace.next_frontier) add(rows, sizeof(*workspace.next_frontier));
  if (workspace.next_marks) add(rows, sizeof(*workspace.next_marks));
  if (workspace.source_mask) add(rows, sizeof(*workspace.source_mask));
  if (workspace.touched_nodes) add(rows, sizeof(*workspace.touched_nodes));
  if (workspace.touched_count) add(1, sizeof(*workspace.touched_count));
  if (workspace.dynamic_vertex_cost)
    add(rows, sizeof(*workspace.dynamic_vertex_cost));
  if (workspace.iteration_status) add(1, sizeof(*workspace.iteration_status));
  if (workspace.controller_result) add(1, sizeof(*workspace.controller_result));
  if (workspace.source_nodes)
    add(static_cast<std::uint64_t>(workspace.source_capacity),
        sizeof(*workspace.source_nodes));
  if (workspace.target_nodes)
    add(static_cast<std::uint64_t>(workspace.target_capacity),
        sizeof(*workspace.target_nodes));
  if (workspace.update_nodes)
    add(static_cast<std::uint64_t>(workspace.update_capacity),
        sizeof(*workspace.update_nodes));
  if (workspace.update_costs)
    add(static_cast<std::uint64_t>(workspace.update_capacity),
        sizeof(*workspace.update_costs));
  if (workspace.target_summaries)
    add(static_cast<std::uint64_t>(workspace.reconstruction_capacity),
        sizeof(*workspace.target_summaries));
  if (workspace.reconstruction_node_offsets)
    add(static_cast<std::uint64_t>(workspace.reconstruction_capacity),
        sizeof(*workspace.reconstruction_node_offsets));
  if (workspace.reconstruction_edge_offsets)
    add(static_cast<std::uint64_t>(workspace.reconstruction_capacity),
        sizeof(*workspace.reconstruction_edge_offsets));
  if (workspace.compact_nodes)
    add(static_cast<std::uint64_t>(workspace.compact_node_capacity),
        sizeof(*workspace.compact_nodes));
  if (workspace.compact_edges)
    add(static_cast<std::uint64_t>(workspace.compact_edge_capacity),
        sizeof(*workspace.compact_edges));
  if (workspace.compact_edge_costs)
    add(static_cast<std::uint64_t>(workspace.compact_edge_capacity),
        sizeof(*workspace.compact_edge_costs));
  if (workspace.telemetry_counters)
    add(1, sizeof(*workspace.telemetry_counters));
  return bytes;
}

void fully_reset_workspace_state(DeviceWorkspace& workspace) {
  begin_telemetry_event(workspace, workspace.reset_seed_events);
  hipLaunchKernelGGL(clear_state_kernel, grid_for_items(workspace.rows),
                     dim3(kBlockSize), 0, workspace.stream, workspace.rows,
                     workspace.best_state, workspace.next_marks,
                     workspace.source_mask);
  check_hip(hipGetLastError(), "defensively reset BF11 search state");
  check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int),
                           workspace.stream),
            "defensively reset BF11 touched count");
  end_telemetry_event(workspace, workspace.reset_seed_events);
  synchronize_query_stream(workspace,
                           "synchronize defensive BF11 state reset");
  accumulate_telemetry_event(workspace.reset_seed_events,
                             g_bf11_reset_seed_gpu_nanoseconds);
  g_bf11_defensive_dense_state_resets.fetch_add(
      1, std::memory_order_relaxed);
  workspace.needs_full_state_reset = false;
}

Offset geometric_capacity(Offset current, Offset required, Offset limit) {
  Offset result = std::max<Offset>(1, current);
  while (result < required) {
    if (result > limit / 2) return limit;
    result *= 2;
  }
  return result;
}

void ensure_source_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.source_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.source_capacity, required, workspace.rows);
  Index* replacement = device_allocate<Index>(
      static_cast<std::size_t>(capacity), "hipMalloc BF11 source list");
  if (workspace.source_nodes) (void)hipFree(workspace.source_nodes);
  workspace.source_nodes = replacement;
  workspace.source_capacity = capacity;
}

void ensure_target_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.target_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.target_capacity, required, workspace.rows);
  Index* replacement = device_allocate<Index>(
      static_cast<std::size_t>(capacity), "hipMalloc BF11 target list");
  if (workspace.target_nodes) (void)hipFree(workspace.target_nodes);
  workspace.target_nodes = replacement;
  workspace.target_capacity = capacity;
}

void ensure_update_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.update_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.update_capacity, required, workspace.rows);
  Index* replacement_nodes = nullptr;
  float* replacement_costs = nullptr;
  try {
    replacement_nodes = device_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 sparse-cost nodes");
    replacement_costs = device_allocate<float>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 sparse-cost values");
  } catch (...) {
    if (replacement_nodes) (void)hipFree(replacement_nodes);
    if (replacement_costs) (void)hipFree(replacement_costs);
    throw;
  }
  if (workspace.update_nodes) (void)hipFree(workspace.update_nodes);
  if (workspace.update_costs) (void)hipFree(workspace.update_costs);
  workspace.update_nodes = replacement_nodes;
  workspace.update_costs = replacement_costs;
  workspace.update_capacity = capacity;
}

void ensure_reconstruction_capacity(DeviceWorkspace& workspace,
                                    Offset required) {
  if (required <= workspace.reconstruction_capacity) return;
  const Offset capacity = geometric_capacity(
      workspace.reconstruction_capacity, required, workspace.rows);
  const std::size_t count = static_cast<std::size_t>(capacity);
  TargetSummary* summaries = nullptr;
  TargetSummary* host_summaries = nullptr;
  unsigned long long* node_offsets = nullptr;
  unsigned long long* edge_offsets = nullptr;
  try {
    summaries = device_allocate<TargetSummary>(
        count, "hipMalloc BF11 target summaries");
    host_summaries = pinned_allocate<TargetSummary>(
        count, "hipHostMalloc BF11 target summaries");
    node_offsets = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 reconstruction node offsets");
    edge_offsets = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 reconstruction edge offsets");
  } catch (...) {
    if (summaries) (void)hipFree(summaries);
    if (host_summaries) (void)hipHostFree(host_summaries);
    if (node_offsets) (void)hipFree(node_offsets);
    if (edge_offsets) (void)hipFree(edge_offsets);
    throw;
  }
  if (workspace.target_summaries) (void)hipFree(workspace.target_summaries);
  if (workspace.host_target_summaries)
    (void)hipHostFree(workspace.host_target_summaries);
  if (workspace.reconstruction_node_offsets)
    (void)hipFree(workspace.reconstruction_node_offsets);
  if (workspace.reconstruction_edge_offsets)
    (void)hipFree(workspace.reconstruction_edge_offsets);
  workspace.target_summaries = summaries;
  workspace.host_target_summaries = host_summaries;
  workspace.reconstruction_node_offsets = node_offsets;
  workspace.reconstruction_edge_offsets = edge_offsets;
  workspace.reconstruction_capacity = capacity;
}

std::size_t grow_size(std::size_t current, std::size_t required) {
  std::size_t result = std::max<std::size_t>(1, current);
  while (result < required) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) return required;
    result *= 2;
  }
  return result;
}

void ensure_compact_capacity(DeviceWorkspace& workspace,
                             std::size_t required_nodes,
                             std::size_t required_edges) {
  if (required_nodes <= workspace.compact_node_capacity &&
      required_edges <= workspace.compact_edge_capacity) {
    return;
  }
  const std::size_t node_capacity =
      required_nodes > workspace.compact_node_capacity
          ? grow_size(workspace.compact_node_capacity, required_nodes)
          : workspace.compact_node_capacity;
  const std::size_t edge_capacity =
      required_edges > workspace.compact_edge_capacity
          ? grow_size(workspace.compact_edge_capacity, required_edges)
          : workspace.compact_edge_capacity;
  Index* nodes = nullptr;
  DeviceOffset* edges = nullptr;
  float* edge_costs = nullptr;
  try {
    nodes = device_allocate<Index>(node_capacity,
                                   "hipMalloc BF11 compact path nodes");
    edges = device_allocate<DeviceOffset>(edge_capacity,
                                          "hipMalloc BF11 compact path edges");
    edge_costs = device_allocate<float>(
        edge_capacity, "hipMalloc BF11 compact path edge costs");
  } catch (...) {
    if (nodes) (void)hipFree(nodes);
    if (edges) (void)hipFree(edges);
    if (edge_costs) (void)hipFree(edge_costs);
    throw;
  }
  if (workspace.compact_nodes) (void)hipFree(workspace.compact_nodes);
  if (workspace.compact_edges) (void)hipFree(workspace.compact_edges);
  if (workspace.compact_edge_costs)
    (void)hipFree(workspace.compact_edge_costs);
  workspace.compact_nodes = nodes;
  workspace.compact_edges = edges;
  workspace.compact_edge_costs = edge_costs;
  workspace.compact_node_capacity = node_capacity;
  workspace.compact_edge_capacity = edge_capacity;
}

void throw_controller_error(int status) {
  switch (status) {
    case 1:
      throw std::runtime_error("BF11 encountered an invalid vertex ID");
    case 2:
      throw std::runtime_error("BF11 encountered invalid CSR row offsets");
    case 3:
      throw std::runtime_error(
          "BF11 effective edge weight is nonfinite or negative");
    case 4:
      throw std::runtime_error("BF11 next-frontier capacity overflowed");
    case 5:
      throw std::runtime_error("BF11 touched-state capacity overflowed");
    default:
      throw std::runtime_error("BF11 controller returned an unknown error");
  }
}

int cooperative_block_count(DeviceWorkspace& workspace) {
  if (workspace.cooperative_blocks >= 0) return workspace.cooperative_blocks;
  int device = -1;
  check_hip(hipGetDevice(&device), "get BF11 HIP device");
  int supported = 0;
  const hipError_t attribute_status = hipDeviceGetAttribute(
      &supported, hipDeviceAttributeCooperativeLaunch, device);
  if (attribute_status != hipSuccess || supported == 0) {
    if (attribute_status != hipSuccess) (void)hipGetLastError();
    workspace.cooperative_blocks = 0;
    return 0;
  }
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, device),
            "get BF11 HIP device properties");
  int blocks_per_cu = 0;
  const hipError_t occupancy_status = workspace.telemetry_enabled
      ? hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_cu, frontier_controller_kernel<true>, kBlockSize, 0)
      : hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_cu, frontier_controller_kernel<false>, kBlockSize, 0);
  if (occupancy_status != hipSuccess || blocks_per_cu <= 0 ||
      properties.multiProcessorCount <= 0) {
    if (occupancy_status != hipSuccess) (void)hipGetLastError();
    workspace.cooperative_blocks = 0;
    return 0;
  }
  const Offset row_blocks = (workspace.rows + kBlockSize - 1) / kBlockSize;
  const Offset legal = static_cast<Offset>(blocks_per_cu) *
                       static_cast<Offset>(properties.multiProcessorCount);
  // A cooperative grid must fit concurrently, but there is no reason to cap
  // it at one block per CU. Use all occupancy-reported resident blocks so the
  // edge scan has enough waves to hide irregular-memory latency.
  const Offset blocks = std::min(row_blocks, legal);
  workspace.cooperative_blocks = blocks > 0 ? static_cast<int>(blocks) : 0;
  return workspace.cooperative_blocks;
}

SsspStatus run_gpu_controller(const DeviceGraph& graph,
                              DeviceWorkspace& workspace,
                              Offset source_count,
                              Offset target_count,
                              int max_iters,
                              const BellmanFord11RunOptions& options,
                              int blocks) {
  PATHFINDER_PROFILE_RANGE("bf11.gpu_controller");
  g_bf11_sparse_state_resets.fetch_add(1, std::memory_order_relaxed);
  DeviceGraph graph_arg = graph;
  const Index* sources_arg = workspace.source_nodes;
  Offset source_count_arg = source_count;
  const Index* targets_arg = workspace.target_nodes;
  Offset target_count_arg = target_count;
  const float* dynamic_cost_arg = workspace.dynamic_vertex_cost;
  BellmanFord11BoundingBox bounds_arg = options.bounds;
  int max_iters_arg = max_iters;
  int check_interval_arg = options.target_check_interval;
  unsigned long long* state_arg = workspace.best_state;
  Index* frontier_arg = workspace.frontier;
  Index* next_arg = workspace.next_frontier;
  int* marks_arg = workspace.next_marks;
  unsigned char* source_mask_arg = workspace.source_mask;
  Index* touched_nodes_arg = workspace.touched_nodes;
  int* touched_count_arg = workspace.touched_count;
  IterationStatus* iteration_arg = workspace.iteration_status;
  ControllerResult* result_arg = workspace.controller_result;
  DeviceTelemetryCounters* telemetry_arg = workspace.telemetry_counters;
  void* args[] = {&graph_arg,
                  &sources_arg,
                  &source_count_arg,
                  &targets_arg,
                  &target_count_arg,
                  &dynamic_cost_arg,
                  &bounds_arg,
                  &max_iters_arg,
                  &check_interval_arg,
                  &state_arg,
                  &frontier_arg,
                  &next_arg,
                  &marks_arg,
                  &source_mask_arg,
                  &touched_nodes_arg,
                  &touched_count_arg,
                  &iteration_arg,
                  &result_arg,
                  &telemetry_arg};
  if (workspace.telemetry_enabled) {
    check_hip(hipMemsetAsync(workspace.telemetry_counters, 0,
                             sizeof(DeviceTelemetryCounters), workspace.stream),
              "reset BF11 telemetry counters");
  }
  const hipError_t launch_status = workspace.telemetry_enabled
      ? hipLaunchCooperativeKernel(
            frontier_controller_kernel<true>,
            dim3(static_cast<unsigned>(blocks)), dim3(kBlockSize), args, 0,
            workspace.stream)
      : hipLaunchCooperativeKernel(
            frontier_controller_kernel<false>,
            dim3(static_cast<unsigned>(blocks)), dim3(kBlockSize), args, 0,
            workspace.stream);
  check_hip(launch_status,
            "launch BF11 cooperative controller");
  g_bf11_gpu_controller_launches.fetch_add(1, std::memory_order_relaxed);
  begin_telemetry_event(workspace, workspace.status_copy_events);
  check_hip(hipMemcpyAsync(workspace.host_controller_result,
                           workspace.controller_result,
                           sizeof(ControllerResult), hipMemcpyDeviceToHost,
                           workspace.stream),
            "copy BF11 controller result");
  end_telemetry_event(workspace, workspace.status_copy_events);
  if (workspace.telemetry_enabled) {
    check_hip(hipMemcpyAsync(workspace.host_telemetry_counters,
                             workspace.telemetry_counters,
                             sizeof(DeviceTelemetryCounters),
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF11 telemetry counters");
    check_hip(hipMemcpyAsync(workspace.host_touched_count,
                             workspace.touched_count, sizeof(int),
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF11 telemetry touched count");
  }
  synchronize_query_stream(workspace,
                           "synchronize BF11 controller result");
  accumulate_telemetry_event(workspace.status_copy_events,
                             g_bf11_status_copy_gpu_nanoseconds);
  const ControllerResult result = *workspace.host_controller_result;
  g_bf11_target_checks.fetch_add(
      static_cast<std::uint64_t>(std::max(0, result.target_checks)),
      std::memory_order_relaxed);
  if (result.error_status != 0) throw_controller_error(result.error_status);
  const int termination_count =
      result.converged + result.early_stopped + result.hit_max_iters;
  if (result.done != 1 || result.iterations_used < 0 ||
      result.iterations_used > max_iters || result.frontier_count < 0 ||
      static_cast<Offset>(result.frontier_count) > graph.rows ||
      result.current_frontier_index < 0 || result.current_frontier_index > 1 ||
      result.target_checks < 0 || termination_count != 1) {
    throw std::runtime_error("BF11 cooperative controller returned bad status");
  }
  return {result.iterations_used, result.converged != 0,
          result.early_stopped != 0, result.hit_max_iters != 0};
}

SsspStatus run_host_controller(const DeviceGraph& graph,
                               DeviceWorkspace& workspace,
                               Offset source_count,
                               Offset target_count,
                               int max_iters,
                               const BellmanFord11RunOptions& options) {
  PATHFINDER_PROFILE_RANGE("bf11.controller_fallback");
  g_bf11_controller_fallbacks.fetch_add(1, std::memory_order_relaxed);
  g_bf11_sparse_state_resets.fetch_add(1, std::memory_order_relaxed);
  HostControllerCounterBatch control_counters;
  ensure_host_window_telemetry_events(workspace);
  if (workspace.telemetry_enabled) {
    check_hip(hipMemsetAsync(workspace.telemetry_counters, 0,
                             sizeof(DeviceTelemetryCounters), workspace.stream),
              "reset BF11 telemetry counters");
  }
  begin_telemetry_event(workspace, workspace.reset_seed_events);
  hipLaunchKernelGGL(clear_touched_state_kernel,
                     sparse_reset_grid(graph.rows), dim3(kBlockSize), 0,
                     workspace.stream, graph.rows, workspace.touched_nodes,
                     workspace.touched_count, workspace.best_state,
                     workspace.next_marks, workspace.source_mask);
  check_hip(hipGetLastError(), "clear BF11 touched state");
  check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int),
                           workspace.stream),
            "reset BF11 touched count");
  hipLaunchKernelGGL(seed_sources_kernel, grid_for_items(source_count),
                     dim3(kBlockSize), 0, workspace.stream,
                     workspace.source_nodes, source_count,
                     workspace.best_state, workspace.frontier,
                     workspace.source_mask, workspace.touched_nodes,
                     workspace.touched_count);
  check_hip(hipGetLastError(), "seed BF11 sources");
  end_telemetry_event(workspace, workspace.reset_seed_events);
  // Same-stream ordering publishes reset and seed work to the first round; a
  // later controller-status or path-result transfer is the first required
  // host synchronization point.

  int frontier_count = static_cast<int>(source_count);
  SsspStatus result;
  if (workspace.iterations_per_host_check == 1) {
    // Preserve the original one-iteration host loop exactly for the control
    // arm of this experiment. Only the additive counters below are new.
    for (int iteration = 0; iteration < max_iters && frontier_count > 0;
         ++iteration) {
      hipLaunchKernelGGL(reset_iteration_status_kernel, dim3(1), dim3(1), 0,
                         workspace.stream, workspace.iteration_status);
      check_hip(hipGetLastError(), "reset BF11 iteration status");
      begin_telemetry_event(workspace, workspace.relaxation_events);
      if (workspace.telemetry_enabled) {
        hipLaunchKernelGGL((frontier_relax_kernel<true>),
                           grid_for_items(frontier_count), dim3(kBlockSize), 0,
                           workspace.stream, graph, workspace.frontier,
                           frontier_count, iteration + 1,
                           workspace.dynamic_vertex_cost, options.bounds,
                           workspace.best_state, workspace.next_frontier,
                           workspace.next_marks, workspace.source_mask,
                           workspace.touched_nodes, workspace.touched_count,
                           workspace.iteration_status,
                           workspace.telemetry_counters);
      } else {
        hipLaunchKernelGGL((frontier_relax_kernel<false>),
                           grid_for_items(frontier_count), dim3(kBlockSize), 0,
                           workspace.stream, graph, workspace.frontier,
                           frontier_count, iteration + 1,
                           workspace.dynamic_vertex_cost, options.bounds,
                           workspace.best_state, workspace.next_frontier,
                           workspace.next_marks, workspace.source_mask,
                           workspace.touched_nodes, workspace.touched_count,
                           workspace.iteration_status,
                           static_cast<DeviceTelemetryCounters*>(nullptr));
      }
      check_hip(hipGetLastError(), "launch BF11 frontier relaxation");
      end_telemetry_event(workspace, workspace.relaxation_events);
      const bool check_targets =
          (iteration + 1) % options.target_check_interval == 0;
      if (check_targets) {
        begin_telemetry_event(workspace, workspace.target_check_events);
        hipLaunchKernelGGL(update_target_status_kernel,
                           grid_for_items(target_count), dim3(kBlockSize), 0,
                           workspace.stream, workspace.best_state,
                           workspace.target_nodes, target_count,
                           workspace.iteration_status);
        check_hip(hipGetLastError(), "launch BF11 target check");
        end_telemetry_event(workspace, workspace.target_check_events);
        ++control_counters.target_checks;
      }
      ++control_counters.device_iterations_enqueued;
      begin_telemetry_event(workspace, workspace.status_copy_events);
      check_hip(hipMemcpyAsync(workspace.host_iteration_status,
                               workspace.iteration_status,
                               sizeof(IterationStatus), hipMemcpyDeviceToHost,
                               workspace.stream),
                "copy BF11 iteration status");
      ++control_counters.iteration_status_copies;
      end_telemetry_event(workspace, workspace.status_copy_events);
      synchronize_query_stream(workspace,
                               "synchronize BF11 iteration status");
      ++control_counters.iteration_control_stream_syncs;
      ++control_counters.host_check_rounds;
      accumulate_telemetry_event(workspace.reset_seed_events,
                                 g_bf11_reset_seed_gpu_nanoseconds);
      accumulate_telemetry_event(workspace.relaxation_events,
                                 g_bf11_relaxation_gpu_nanoseconds);
      accumulate_telemetry_event(workspace.target_check_events,
                                 g_bf11_target_check_gpu_nanoseconds);
      accumulate_telemetry_event(workspace.status_copy_events,
                                 g_bf11_status_copy_gpu_nanoseconds);
      const IterationStatus status = *workspace.host_iteration_status;
      ++control_counters.device_iterations_executed;
      if (status.error_status != 0) throw_controller_error(status.error_status);
      result.iterations_used = iteration + 1;
      frontier_count = status.next_count;
      if (frontier_count == 0) {
        result.converged = true;
        break;
      }
      if (check_targets &&
          status.reached_target_count == static_cast<int>(target_count) &&
          status.min_next_frontier_dist_bits >= status.max_target_dist_bits) {
        result.early_stopped = true;
        break;
      }
      std::swap(workspace.frontier, workspace.next_frontier);
    }
  } else if (max_iters > 0 && frontier_count > 0) {
    hipLaunchKernelGGL(initialize_host_window_controller_kernel,
                       dim3(1), dim3(1), 0, workspace.stream,
                       frontier_count, workspace.iteration_status,
                       workspace.controller_result);
    check_hip(hipGetLastError(),
              "initialize BF11 host-window controller");

    int iterations_enqueued = 0;
    int observed_iterations = 0;
    int observed_target_checks = 0;
    while (iterations_enqueued < max_iters) {
      const int window_size = std::min(
          workspace.iterations_per_host_check,
          max_iters - iterations_enqueued);
      const dim3 relaxation_grid =
          host_batched_relaxation_grid(graph.rows, frontier_count);
      for (int offset = 0; offset < window_size; ++offset) {
        TelemetryEventPair& relaxation_events =
            workspace.batched_relaxation_events.empty()
            ? workspace.relaxation_events
            : workspace.batched_relaxation_events[
                  static_cast<std::size_t>(offset)];
        begin_telemetry_event(workspace, relaxation_events);
        if (workspace.telemetry_enabled) {
          hipLaunchKernelGGL((host_window_frontier_relax_kernel<true>),
                             relaxation_grid, dim3(kBlockSize), 0,
                             workspace.stream, graph, workspace.frontier,
                             workspace.next_frontier,
                             workspace.dynamic_vertex_cost, options.bounds,
                             workspace.best_state, workspace.next_marks,
                             workspace.source_mask, workspace.touched_nodes,
                             workspace.touched_count,
                             workspace.iteration_status,
                             workspace.controller_result,
                             workspace.telemetry_counters);
        } else {
          hipLaunchKernelGGL((host_window_frontier_relax_kernel<false>),
                             relaxation_grid, dim3(kBlockSize), 0,
                             workspace.stream, graph, workspace.frontier,
                             workspace.next_frontier,
                             workspace.dynamic_vertex_cost, options.bounds,
                             workspace.best_state, workspace.next_marks,
                             workspace.source_mask, workspace.touched_nodes,
                             workspace.touched_count,
                             workspace.iteration_status,
                             workspace.controller_result,
                             static_cast<DeviceTelemetryCounters*>(nullptr));
        }
        check_hip(hipGetLastError(),
                  "launch BF11 host-window frontier relaxation");
        end_telemetry_event(workspace, relaxation_events);

        const int logical_iteration = iterations_enqueued + offset + 1;
        const bool check_targets =
            logical_iteration % options.target_check_interval == 0;
        if (check_targets) {
          TelemetryEventPair& target_events =
              workspace.batched_target_check_events.empty()
              ? workspace.target_check_events
              : workspace.batched_target_check_events[
                    static_cast<std::size_t>(offset)];
          begin_telemetry_event(workspace, target_events);
          hipLaunchKernelGGL(update_host_window_target_status_kernel,
                             grid_for_items(target_count), dim3(kBlockSize), 0,
                             workspace.stream, workspace.best_state,
                             workspace.target_nodes, target_count,
                             workspace.iteration_status,
                             workspace.controller_result);
          check_hip(hipGetLastError(),
                    "launch BF11 host-window target check");
          end_telemetry_event(workspace, target_events);
        }
        hipLaunchKernelGGL(advance_host_window_controller_kernel,
                           dim3(1), dim3(1), 0, workspace.stream,
                           workspace.iteration_status,
                           workspace.controller_result,
                           static_cast<int>(target_count), max_iters,
                           options.target_check_interval);
        check_hip(hipGetLastError(),
                  "advance BF11 host-window controller");
        ++control_counters.device_iterations_enqueued;
      }

      begin_telemetry_event(workspace, workspace.status_copy_events);
      check_hip(hipMemcpyAsync(workspace.host_controller_result,
                               workspace.controller_result,
                               sizeof(ControllerResult), hipMemcpyDeviceToHost,
                               workspace.stream),
                "copy BF11 host-window controller result");
      ++control_counters.iteration_status_copies;
      end_telemetry_event(workspace, workspace.status_copy_events);
      synchronize_query_stream(
          workspace, "synchronize BF11 host-window controller result");
      ++control_counters.iteration_control_stream_syncs;
      ++control_counters.host_check_rounds;

      accumulate_telemetry_event(workspace.reset_seed_events,
                                 g_bf11_reset_seed_gpu_nanoseconds);
      for (int offset = 0; offset < window_size; ++offset) {
        if (!workspace.batched_relaxation_events.empty()) {
          accumulate_telemetry_event(
              workspace.batched_relaxation_events[
                  static_cast<std::size_t>(offset)],
              g_bf11_relaxation_gpu_nanoseconds);
          accumulate_telemetry_event(
              workspace.batched_target_check_events[
                  static_cast<std::size_t>(offset)],
              g_bf11_target_check_gpu_nanoseconds);
        }
      }
      accumulate_telemetry_event(workspace.status_copy_events,
                                 g_bf11_status_copy_gpu_nanoseconds);

      const ControllerResult status = *workspace.host_controller_result;
      const int termination_count =
          status.converged + status.early_stopped + status.hit_max_iters;
      const int maximum_observed_iteration =
          iterations_enqueued + window_size;
      if (status.iterations_used < observed_iterations ||
          status.iterations_used > maximum_observed_iteration ||
          (status.error_status == 0 &&
           (status.frontier_count < 0 ||
            static_cast<Offset>(status.frontier_count) > graph.rows ||
            status.current_frontier_index < 0 ||
            status.current_frontier_index > 1)) ||
          status.target_checks < observed_target_checks ||
          status.target_checks > status.iterations_used ||
          status.done < 0 || status.done > 1 ||
          (status.done == 0 &&
           (status.iterations_used != maximum_observed_iteration ||
            status.frontier_count <= 0 ||
            status.iterations_used >= max_iters)) ||
          (status.error_status == 0 && status.done == 0 &&
           termination_count != 0) ||
          (status.error_status == 0 && status.done != 0 &&
           termination_count != 1) ||
          (status.error_status != 0 &&
           (status.done != 1 || termination_count != 0))) {
        throw std::runtime_error(
            "BF11 host-window controller returned bad status");
      }
      const int executed = status.iterations_used - observed_iterations;
      const int terminal_noops = window_size - executed;
      control_counters.device_iterations_executed +=
          static_cast<std::uint64_t>(executed);
      control_counters.terminal_noop_iterations +=
          static_cast<std::uint64_t>(terminal_noops);
      control_counters.target_checks += static_cast<std::uint64_t>(
          status.target_checks - observed_target_checks);
      observed_iterations = status.iterations_used;
      observed_target_checks = status.target_checks;
      iterations_enqueued += window_size;
      result.iterations_used = status.iterations_used;
      result.converged = status.converged != 0;
      result.early_stopped = status.early_stopped != 0;
      result.hit_max_iters = status.hit_max_iters != 0;
      frontier_count = status.frontier_count;
      if (status.error_status != 0) {
        throw_controller_error(status.error_status);
      }
      if (status.done != 0) break;
    }
  }
  if (workspace.telemetry_enabled) {
    check_hip(hipMemcpyAsync(workspace.host_telemetry_counters,
                             workspace.telemetry_counters,
                             sizeof(DeviceTelemetryCounters),
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF11 telemetry counters");
    check_hip(hipMemcpyAsync(workspace.host_touched_count,
                             workspace.touched_count, sizeof(int),
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF11 telemetry touched count");
  }
  result.hit_max_iters = !result.converged && !result.early_stopped;
  return result;
}

SsspStatus run_sssp(const DeviceGraph& graph,
                    DeviceWorkspace& workspace,
                    Offset source_count,
                    Offset target_count,
                    int max_iters,
                    const BellmanFord11RunOptions& options) {
  if (max_iters < 0) max_iters = static_cast<int>(graph.rows) - 1;
  // PathFinder's parallel workers use independent nonblocking streams. A BF11
  // cooperative grid consumes the device's full legal residency, so launching
  // one on each worker stream can corrupt the HIP runtime on gfx1151. Preserve
  // the one-round-trip persistent controller for the default stream and use
  // the complete host controller for explicit streams.
  const int blocks =
      workspace.stream == nullptr ? cooperative_block_count(workspace) : 0;
  return blocks > 0
             ? run_gpu_controller(graph, workspace, source_count, target_count,
                                  max_iters, options, blocks)
             : run_host_controller(graph, workspace, source_count, target_count,
                                   max_iters, options);
}

std::vector<int> deduplicate_nodes(const std::vector<int>& input,
                                   Offset rows,
                                   const char* kind) {
  if (input.empty()) {
    throw std::invalid_argument(std::string("BF11 requires at least one ") + kind);
  }
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string("BF11 ") + kind +
                              " count does not fit int");
  }
  std::vector<int> result;
  result.reserve(input.size());
  std::unordered_set<int> seen;
  seen.reserve(input.size());
  for (const int node : input) {
    if (node < 0 || static_cast<Offset>(node) >= rows) {
      throw std::out_of_range(std::string("BF11 ") + kind +
                              " is outside the CSR graph");
    }
    if (seen.insert(node).second) result.push_back(node);
  }
  return result;
}

bool known_coordinate(const HostSidecarView& sidecars, int node) {
  return (*sidecars.route_end_x)[static_cast<std::size_t>(node)] !=
             kMissingCoordinate &&
         (*sidecars.route_end_y)[static_cast<std::size_t>(node)] !=
             kMissingCoordinate;
}

bool in_bounds(const HostSidecarView& sidecars,
               int node,
               const BellmanFord11BoundingBox& bounds) {
  if (!bounds.enabled) return true;
  if (!known_coordinate(sidecars, node)) return false;
  const std::int32_t x =
      (*sidecars.route_end_x)[static_cast<std::size_t>(node)];
  const std::int32_t y =
      (*sidecars.route_end_y)[static_cast<std::size_t>(node)];
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

void validate_run_options(const BellmanFord11RunOptions& options) {
  if (options.target_check_interval <= 0) {
    throw std::invalid_argument("BF11 target-check interval must be positive");
  }
  if (options.bounds.enabled &&
      (options.bounds.min_x > options.bounds.max_x ||
       options.bounds.min_y > options.bounds.max_y)) {
    throw std::invalid_argument("BF11 bounding box is inverted");
  }
}

void validate_terminal_bounds(const HostSidecarView& sidecars,
                              const std::vector<int>& sources,
                              const std::vector<int>& targets,
                              const BellmanFord11BoundingBox& bounds,
                              bool allow_missing_sources) {
  if (!bounds.enabled) return;
  for (const int source : sources) {
    if (allow_missing_sources && !known_coordinate(sidecars, source)) {
      continue;
    }
    if (!in_bounds(sidecars, source, bounds)) {
      throw std::invalid_argument(
          "BF11 bounded source is missing coordinates or outside the box");
    }
  }
  for (const int target : targets) {
    if (!in_bounds(sidecars, target, bounds)) {
      throw std::invalid_argument(
          "BF11 bounded target is missing coordinates or outside the box");
    }
  }
}

std::int32_t saturating_margin(std::int32_t value,
                               std::int32_t margin,
                               bool subtract) {
  const std::int64_t widened =
      subtract ? static_cast<std::int64_t>(value) - margin
               : static_cast<std::int64_t>(value) + margin;
  return static_cast<std::int32_t>(std::max<std::int64_t>(
      std::numeric_limits<std::int32_t>::min(),
      std::min<std::int64_t>(std::numeric_limits<std::int32_t>::max(),
                             widened)));
}

bool make_auto_bounds(const HostSidecarView& sidecars,
                      const std::vector<int>& sources,
                      const std::vector<int>& targets,
                      const BellmanFord11WorkspaceOptions& options,
                      BellmanFord11BoundingBox* output) {
  std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_x = std::numeric_limits<std::int32_t>::min();
  std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_y = std::numeric_limits<std::int32_t>::min();
  auto include = [&](int node) {
    if (!known_coordinate(sidecars, node)) return false;
    const std::int32_t x =
        (*sidecars.route_end_x)[static_cast<std::size_t>(node)];
    const std::int32_t y =
        (*sidecars.route_end_y)[static_cast<std::size_t>(node)];
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    return true;
  };
  for (const int source : sources) (void)include(source);
  for (const int target : targets) {
    if (!include(target)) return false;
  }
  *output = {true,
             saturating_margin(min_x, options.auto_margin_x, true),
             saturating_margin(max_x, options.auto_margin_x, false),
             saturating_margin(min_y, options.auto_margin_y, true),
             saturating_margin(max_y, options.auto_margin_y, false)};
  return true;
}

int saturated_add(int left, int right) {
  if (right > 0 && left > std::numeric_limits<int>::max() - right) {
    return std::numeric_limits<int>::max();
  }
  return left + right;
}

}  // namespace rips_sssp_bf11

struct BellmanFord11CsrGraph::Impl {
  minplus_sparse::Offset rows = 0;
  minplus_sparse::Offset nnz = 0;
  // Auto-bound queries need random host access after construction. Own only
  // the two compact coordinate columns; base costs and spatial shards are
  // upload-only and are deliberately not duplicated here.
  std::vector<std::int32_t> host_route_end_x;
  std::vector<std::int32_t> host_route_end_y;
  rips_sssp_bf11::HostSidecarView host_sidecars;
  rips_sssp_bf11::DeviceGraphOwner device;
  int hip_device = -1;

  Impl(const HostCsrF32& adjacency,
       rips_sssp_bf11::HostSidecarView sidecars,
       hipStream_t stream)
      : rows(adjacency.rows),
        nnz(adjacency.nnz),
        host_route_end_x(*sidecars.route_end_x),
        host_route_end_y(*sidecars.route_end_y),
        host_sidecars{&host_route_end_x, &host_route_end_y, nullptr} {
    rips_sssp_bf11::validate_csr(adjacency);
    rips_sssp_bf11::validate_sidecar_view(sidecars, rows);
    rips_sssp_bf11::check_hip(hipGetDevice(&hip_device),
                              "get BF11 graph HIP device");
    PATHFINDER_PROFILE_RANGE("bf11.upload_graph");
    device = rips_sssp_bf11::copy_graph_to_device(adjacency, sidecars, stream);
  }

  ~Impl() {
    rips_sssp_bf11::ScopedOwningHipDevice owner_device(hip_device);
    if (owner_device.active()) rips_sssp_bf11::free_graph(&device);
  }
};

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency,
          rips_sssp_bf11::common_sidecar_view(sidecars, adjacency), stream)) {}

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency,
          rips_sssp_bf11::compatibility_sidecar_view(sidecars, adjacency),
          stream)) {}

BellmanFord11CsrGraph::~BellmanFord11CsrGraph() = default;
BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    BellmanFord11CsrGraph&&) noexcept = default;
BellmanFord11CsrGraph& BellmanFord11CsrGraph::operator=(
    BellmanFord11CsrGraph&&) noexcept = default;

struct BellmanFord11CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord11CsrGraph::Impl> graph;
  rips_sssp_bf11::DeviceWorkspace workspace;
  hipStream_t stream = nullptr;
  BellmanFord11WorkspaceOptions options;
  std::uint64_t maximum_workspace_device_bytes = 0;
  // Serializes a complete cost epoch (update or run). In particular, a caller
  // cannot change destination multipliers between rounds of the capability
  // fallback controller.
  std::mutex operation_mutex;

  static std::shared_ptr<const BellmanFord11CsrGraph::Impl> require_graph(
      const std::shared_ptr<const BellmanFord11CsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("BF11 shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const BellmanFord11CsrGraph::Impl> graph_in,
       hipStream_t stream_in,
       BellmanFord11WorkspaceOptions options_in)
      : graph(std::move(graph_in)), stream(stream_in), options(options_in) {
    if (options.auto_margin_x < 0 || options.auto_margin_y < 0 ||
        options.target_check_interval <= 0 ||
        options.iterations_per_host_check <= 0 ||
        options.iterations_per_host_check >
            kBellmanFord11MaximumIterationsPerHostCheck) {
      throw std::invalid_argument(
          "BF11 auto-bound margins and target interval must be "
          "nonnegative/positive, and iterations per host check must be in "
          "[1, 64]");
    }
    int current_device = -1;
    rips_sssp_bf11::check_hip(hipGetDevice(&current_device),
                              "get BF11 workspace HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument("BF11 graph belongs to another HIP device");
    }
    workspace = rips_sssp_bf11::make_workspace(
        graph->rows, stream, options.telemetry,
        options.iterations_per_host_check);
    if (options.telemetry) {
      maximum_workspace_device_bytes =
          rips_sssp_bf11::workspace_device_bytes(workspace);
      const std::uint64_t constructed =
          g_bf11_constructed_workers.fetch_add(1, std::memory_order_relaxed) + 1;
      const std::uint64_t expected =
          g_bf11_effective_workers.load(std::memory_order_relaxed);
      if (expected != 0 && constructed >= expected) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
          (void)total_bytes;
          g_bf11_gpu_free_after_workers.store(
              static_cast<std::uint64_t>(free_bytes),
              std::memory_order_relaxed);
        } else {
          (void)hipGetLastError();
        }
      }
    }
  }

  ~Impl() {
    if (options.telemetry) {
      g_bf11_workspace_device_bytes_total.fetch_add(
          maximum_workspace_device_bytes, std::memory_order_relaxed);
      atomic_max(g_bf11_workspace_device_bytes_per_worker_max,
                 maximum_workspace_device_bytes);
    }
    rips_sssp_bf11::ScopedOwningHipDevice owner_device(graph->hip_device);
    if (owner_device.active()) rips_sssp_bf11::free_workspace(&workspace);
  }

  void note_workspace_size() {
    if (!options.telemetry) return;
    maximum_workspace_device_bytes = std::max(
        maximum_workspace_device_bytes,
        rips_sssp_bf11::workspace_device_bytes(workspace));
  }

  void require_stream(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "BF11 workspace is stream-affine; use its construction stream");
    }
    int current_device = -1;
    rips_sssp_bf11::check_hip(hipGetDevice(&current_device),
                              "get BF11 run HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument("BF11 workspace is on another HIP device");
    }
  }

  BellmanFordCsrResult run_once(const std::vector<int>& sources,
                                const std::vector<int>& targets,
                                int max_iters,
                                const BellmanFord11RunOptions& run_options,
                                bool allow_missing_bounded_sources = false) {
    using namespace rips_sssp_bf11;
    ScopedQueryTelemetry query_telemetry(workspace.telemetry_enabled);
    validate_run_options(run_options);
    const std::vector<int> unique_sources =
        deduplicate_nodes(sources, graph->rows, "source");
    const std::vector<int> unique_targets =
        deduplicate_nodes(targets, graph->rows, "target");
    validate_terminal_bounds(graph->host_sidecars, unique_sources,
                             unique_targets, run_options.bounds,
                             allow_missing_bounded_sources);

    const Offset source_count = static_cast<Offset>(unique_sources.size());
    const Offset target_count = static_cast<Offset>(unique_targets.size());
    ensure_source_capacity(workspace, source_count);
    ensure_target_capacity(workspace, target_count);
    ensure_reconstruction_capacity(workspace, target_count);
    note_workspace_size();
    if (workspace.needs_full_state_reset) {
      fully_reset_workspace_state(workspace);
    }
    // Leave this set across every asynchronous phase. Only a completely
    // materialized host result proves that the touched-list invariant is safe
    // for sparse reuse; any exception forces one dense reset next time.
    workspace.needs_full_state_reset = true;
    DrainStreamOnException drain(stream);
    check_hip(hipMemcpyAsync(workspace.source_nodes, unique_sources.data(),
                             unique_sources.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 sources");
    check_hip(hipMemcpyAsync(workspace.target_nodes, unique_targets.data(),
                             unique_targets.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 targets");

    SsspStatus status = run_sssp(graph->device.view, workspace,
                                 source_count, target_count, max_iters,
                                 run_options);
    if (status.iterations_used == 0 &&
        std::all_of(unique_targets.begin(), unique_targets.end(),
                    [&](int target) {
                      return std::find(unique_sources.begin(),
                                       unique_sources.end(),
                                       target) != unique_sources.end();
                    })) {
      // Source labels are exact before the first relaxation. Preserve useful
      // max_iters=0 identity queries as certified target stops rather than
      // exposing them as finite-but-tentative results to PathFinder.
      status.early_stopped = true;
      status.hit_max_iters = false;
    }

    begin_telemetry_event(workspace, workspace.target_summary_events);
    hipLaunchKernelGGL(summarize_target_paths_kernel,
                       grid_for_items(target_count), dim3(kBlockSize), 0, stream,
                       workspace.best_state, workspace.source_mask, graph->rows,
                       graph->nnz, graph->device.view.to,
                       graph->device.view.rowptr, workspace.target_nodes,
                       target_count, workspace.target_summaries);
    check_hip(hipGetLastError(), "summarize BF11 target paths");
    check_hip(hipMemcpyAsync(workspace.host_target_summaries,
                             workspace.target_summaries,
                             unique_targets.size() * sizeof(TargetSummary),
                             hipMemcpyDeviceToHost, stream),
              "copy BF11 target summaries");
    end_telemetry_event(workspace, workspace.target_summary_events);
    synchronize_query_stream(workspace,
                             "synchronize BF11 target summaries");
    accumulate_telemetry_event(workspace.reset_seed_events,
                               g_bf11_reset_seed_gpu_nanoseconds);
    accumulate_telemetry_event(workspace.target_summary_events,
                               g_bf11_target_summary_gpu_nanoseconds);
    aggregate_query_work_telemetry(workspace, status.iterations_used);

    std::vector<unsigned long long> node_offsets(unique_targets.size(), 0);
    std::vector<unsigned long long> edge_offsets(unique_targets.size(), 0);
    unsigned long long total_nodes = 0;
    unsigned long long total_edges = 0;
    for (std::size_t i = 0; i < unique_targets.size(); ++i) {
      const TargetSummary& summary = workspace.host_target_summaries[i];
      node_offsets[i] = total_nodes;
      edge_offsets[i] = total_edges;
      if (summary.status == kTargetPathInvalid) {
        throw std::runtime_error(
            "BF11 produced an invalid or cyclic predecessor chain");
      }
      if (summary.status != kTargetPathValid) continue;
      if (summary.node_count == 0 ||
          summary.node_count != summary.edge_count + 1 ||
          summary.node_count > ~total_nodes ||
          summary.edge_count > ~total_edges) {
        throw std::overflow_error("BF11 compact target paths overflow uint64");
      }
      total_nodes += summary.node_count;
      total_edges += summary.edge_count;
    }
    if (total_nodes > std::numeric_limits<std::size_t>::max() ||
        total_edges > std::numeric_limits<std::size_t>::max() ||
        total_nodes >
            static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        total_edges >
            static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "BF11 compact paths exceed host/result offset capacity");
    }
    const std::size_t compact_node_count =
        static_cast<std::size_t>(total_nodes);
    const std::size_t compact_edge_count =
        static_cast<std::size_t>(total_edges);
    ensure_compact_capacity(workspace, compact_node_count, compact_edge_count);
    note_workspace_size();
    begin_telemetry_event(workspace, workspace.reconstruction_events);
    check_hip(hipMemcpyAsync(workspace.reconstruction_node_offsets,
                             node_offsets.data(),
                             node_offsets.size() * sizeof(unsigned long long),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 path node offsets");
    check_hip(hipMemcpyAsync(workspace.reconstruction_edge_offsets,
                             edge_offsets.data(),
                             edge_offsets.size() * sizeof(unsigned long long),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 path edge offsets");
    hipLaunchKernelGGL(
        materialize_target_paths_kernel, grid_for_items(target_count),
        dim3(kBlockSize), 0, stream, workspace.best_state,
        workspace.source_mask, graph->rows, graph->nnz,
        graph->device.view.to, graph->device.view.rowptr,
        workspace.target_nodes, workspace.target_summaries,
        workspace.reconstruction_node_offsets,
        workspace.reconstruction_edge_offsets, target_count,
        workspace.compact_nodes, workspace.compact_edges,
        graph->device.view.edge_values,
        graph->device.view.base_vertex_cost,
        workspace.dynamic_vertex_cost,
        workspace.compact_edge_costs);
    check_hip(hipGetLastError(), "materialize BF11 target paths");
    std::vector<Index> compact_nodes(compact_node_count);
    std::vector<DeviceOffset> compact_edges(compact_edge_count);
    std::vector<float> compact_edge_costs(compact_edge_count);
    if (!compact_nodes.empty()) {
      check_hip(hipMemcpyAsync(compact_nodes.data(), workspace.compact_nodes,
                               compact_nodes.size() * sizeof(Index),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact nodes");
    }
    if (!compact_edges.empty()) {
      check_hip(hipMemcpyAsync(compact_edges.data(), workspace.compact_edges,
                               compact_edges.size() * sizeof(DeviceOffset),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edges");
      check_hip(hipMemcpyAsync(compact_edge_costs.data(),
                               workspace.compact_edge_costs,
                               compact_edge_costs.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edge costs");
    }
    end_telemetry_event(workspace, workspace.reconstruction_events);
    synchronize_query_stream(workspace,
                             "synchronize BF11 compact target paths");
    accumulate_telemetry_event(workspace.reconstruction_events,
                               g_bf11_reconstruction_gpu_nanoseconds);

    std::unordered_map<int, std::size_t> target_index;
    target_index.reserve(unique_targets.size() * 2 + 1);
    for (std::size_t i = 0; i < unique_targets.size(); ++i) {
      target_index.emplace(unique_targets[i], i);
    }
    BellmanFordCsrResult result;
    result.target = -1;
    result.iterations_used = status.iterations_used;
    result.converged = status.converged;
    result.stopped_on_target = status.early_stopped;
    result.target_reached = true;
    result.target_distances.reserve(targets.size());
    result.target_sources.reserve(targets.size());
    result.target_path_offsets.reserve(targets.size() + 1);
    result.target_edge_offsets.reserve(targets.size() + 1);
    result.target_path_offsets.push_back(0);
    result.target_edge_offsets.push_back(0);
    for (const int target : targets) {
      const std::size_t unique = target_index.at(target);
      const TargetSummary& summary = workspace.host_target_summaries[unique];
      if (summary.status == kTargetUnreachable) {
        result.target_distances.push_back(
            std::numeric_limits<float>::infinity());
        result.target_sources.push_back(-1);
        result.target_reached = false;
      } else {
        const std::size_t node_begin =
            static_cast<std::size_t>(node_offsets[unique]);
        const std::size_t edge_begin =
            static_cast<std::size_t>(edge_offsets[unique]);
        const std::size_t node_end =
            node_begin + static_cast<std::size_t>(summary.node_count);
        const std::size_t edge_end =
            edge_begin + static_cast<std::size_t>(summary.edge_count);
        if (node_end > compact_nodes.size() || edge_end > compact_edges.size() ||
            compact_nodes[node_begin] != summary.root ||
            compact_nodes[node_end - 1] != target) {
          throw std::runtime_error("BF11 compact target path is malformed");
        }
        result.target_distances.push_back(host_state_distance(summary.state));
        result.target_sources.push_back(summary.root);
        result.target_path_nodes.insert(result.target_path_nodes.end(),
                                        compact_nodes.begin() + node_begin,
                                        compact_nodes.begin() + node_end);
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          result.target_path_edges.push_back(
              static_cast<minplus_sparse::Offset>(compact_edges[edge]));
          result.target_path_edge_costs.push_back(compact_edge_costs[edge]);
        }
      }
      if (result.target_path_nodes.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          result.target_path_edges.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BF11 result path offsets exceed int");
      }
      result.target_path_offsets.push_back(
          static_cast<int>(result.target_path_nodes.size()));
      result.target_edge_offsets.push_back(
          static_cast<int>(result.target_path_edges.size()));
    }
    workspace.needs_full_state_reset = false;
    query_telemetry.mark_completed();
    return result;
  }
};

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::common_sidecar_view(sidecars, adjacency), stream),
          stream, options)) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::compatibility_sidecar_view(sidecars, adjacency),
              stream),
          stream, options)) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(Impl::require_graph(adjacency), stream,
                                   options)) {}

BellmanFord11CsrWorkspace::~BellmanFord11CsrWorkspace() = default;
BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    BellmanFord11CsrWorkspace&&) noexcept = default;
BellmanFord11CsrWorkspace& BellmanFord11CsrWorkspace::operator=(
    BellmanFord11CsrWorkspace&&) noexcept = default;

void BellmanFord11CsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("bf11.update_vertex_costs");
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (vertex_costs.size() != static_cast<std::size_t>(impl_->graph->rows)) {
    throw std::invalid_argument("BF11 vertex-cost vector must contain V entries");
  }
  for (const float cost : vertex_costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument(
          "BF11 dynamic vertex costs must be finite and nonnegative");
    }
  }
  rips_sssp_bf11::DrainStreamOnException drain(stream);
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.dynamic_vertex_cost,
                     vertex_costs.data(), vertex_costs.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream),
      "copy BF11 dynamic vertex costs");
  rips_sssp_bf11::check_hip(hipStreamSynchronize(stream),
                            "synchronize BF11 dynamic vertex costs");
}

void BellmanFord11CsrWorkspace::update_vertex_costs_sparse(
    const std::vector<int>& nodes,
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("bf11.update_vertex_costs_sparse");
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (nodes.size() != vertex_costs.size()) {
    throw std::invalid_argument("BF11 sparse cost nodes and values differ in size");
  }
  if (nodes.empty()) return;
  std::unordered_set<int> seen;
  seen.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i] < 0 ||
        static_cast<minplus_sparse::Offset>(nodes[i]) >= impl_->graph->rows) {
      throw std::out_of_range("BF11 sparse cost node is outside the graph");
    }
    if (!seen.insert(nodes[i]).second) {
      throw std::invalid_argument("BF11 sparse cost nodes must be unique");
    }
    if (!std::isfinite(vertex_costs[i]) || vertex_costs[i] < 0.0f) {
      throw std::invalid_argument(
          "BF11 dynamic vertex costs must be finite and nonnegative");
    }
  }
  if (nodes.size() > static_cast<std::size_t>(impl_->graph->rows)) {
    throw std::invalid_argument("BF11 sparse cost update exceeds V entries");
  }
  const rips_sssp_bf11::Offset count =
      static_cast<rips_sssp_bf11::Offset>(nodes.size());
  rips_sssp_bf11::ensure_update_capacity(impl_->workspace, count);
  impl_->note_workspace_size();
  rips_sssp_bf11::DrainStreamOnException drain(stream);
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.update_nodes, nodes.data(),
                     nodes.size() * sizeof(int), hipMemcpyHostToDevice, stream),
      "copy BF11 sparse cost nodes");
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.update_costs, vertex_costs.data(),
                     vertex_costs.size() * sizeof(float), hipMemcpyHostToDevice,
                     stream),
      "copy BF11 sparse cost values");
  hipLaunchKernelGGL(rips_sssp_bf11::sparse_cost_update_kernel,
                     rips_sssp_bf11::grid_for_items(count),
                     dim3(rips_sssp_bf11::kBlockSize), 0, stream,
                     impl_->workspace.update_nodes,
                     impl_->workspace.update_costs, count,
                     impl_->workspace.dynamic_vertex_cost);
  rips_sssp_bf11::check_hip(hipGetLastError(),
                            "apply BF11 sparse vertex costs");
  rips_sssp_bf11::check_hip(hipStreamSynchronize(stream),
                            "synchronize BF11 sparse vertex costs");
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("bf11.run");
  (void)delta;
  (void)progress_user_data;
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (progress_callback) {
    throw std::invalid_argument(
        "BF11 persistent controller does not expose per-round callbacks");
  }
  const std::vector<int> unique_sources = rips_sssp_bf11::deduplicate_nodes(
      sources, impl_->graph->rows, "source");
  const std::vector<int> unique_targets = rips_sssp_bf11::deduplicate_nodes(
      targets, impl_->graph->rows, "target");

  BellmanFord11RunOptions run_options;
  run_options.target_check_interval = impl_->options.target_check_interval;
  if (!impl_->options.auto_bounds) {
    return impl_->run_once(sources, targets, max_iters, run_options);
  }

  const bool have_terminal_coordinates = rips_sssp_bf11::make_auto_bounds(
      impl_->graph->host_sidecars, unique_sources, unique_targets,
      impl_->options, &run_options.bounds);
  if (!have_terminal_coordinates) {
    if (!impl_->options.unbounded_fallback) {
      throw std::invalid_argument(
          "BF11 cannot auto-bound a target without coordinates");
    }
    run_options.bounds = {};
    return impl_->run_once(sources, targets, max_iters, run_options);
  }

  BellmanFordCsrResult bounded =
      impl_->run_once(sources, targets, max_iters, run_options, true);
  if (bounded.target_reached || !impl_->options.unbounded_fallback) {
    return bounded;
  }
  g_bf11_auto_unbounded_retries.fetch_add(1, std::memory_order_relaxed);
  run_options.bounds = {};
  BellmanFordCsrResult unbounded =
      impl_->run_once(sources, targets, max_iters, run_options);
  unbounded.iterations_used = rips_sssp_bf11::saturated_add(
      bounded.iterations_used, unbounded.iterations_used);
  return unbounded;
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
  PATHFINDER_PROFILE_RANGE("bf11.run_explicit");
  (void)delta;
  (void)progress_user_data;
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (progress_callback) {
    throw std::invalid_argument(
        "BF11 persistent controller does not expose per-round callbacks");
  }
  return impl_->run_once(sources, targets, max_iters, run_options);
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

#ifndef BF11_NO_MAIN
int main(int, char**) {
  std::cerr
      << "bf11 is a PathFinder library backend; build it with BF11_NO_MAIN "
         "and construct BellmanFord11CsrWorkspace with routing sidecars.\n";
  return 2;
}
#endif
