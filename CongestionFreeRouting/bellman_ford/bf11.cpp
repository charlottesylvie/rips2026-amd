// Bounded, dynamically weighted, true-multi-source active-frontier
// Bellman-Ford for PathFinder. The default-stream cooperative controller keeps
// every relaxation round and target certificate on the GPU; explicit worker
// streams use ordinary device-controlled K-round segments because a BF11
// cooperative grid consumes full device residency on the target gfx1151
// runtime and must never overlap another full-residency grid.

#include "bf11.hpp"
#include "bf11_execution_policy.hpp"
#include "bf11_graph_execution_policy.hpp"
#include "bf11_worker_policy.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
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
std::atomic<std::uint64_t> g_bf11_rounds{0};
std::atomic<std::uint64_t> g_bf11_segments{0};
std::atomic<std::uint64_t> g_bf11_no_op_segment_rounds{0};
std::atomic<std::uint64_t> g_bf11_direct_segments{0};
std::atomic<std::uint64_t> g_bf11_hip_graph_segments{0};
std::atomic<std::uint64_t> g_bf11_status_copies{0};
std::atomic<std::uint64_t> g_bf11_stream_synchronizations{0};
std::atomic<std::uint64_t> g_bf11_graph_fallbacks{0};
std::atomic<std::uint64_t> g_bf11_adaptive_dense_state_resets{0};
std::atomic<std::uint64_t> g_bf11_constant_one_queries{0};
std::atomic<std::uint64_t> g_bf11_static_cost_queries{0};
std::atomic<std::uint64_t> g_bf11_dynamic_cost_queries{0};
std::atomic<std::uint64_t> g_bf11_first_discoveries{0};
std::atomic<std::uint64_t> g_bf11_mark_cas_attempts{0};
std::atomic<std::uint64_t> g_bf11_mark_cas_wins{0};
std::atomic<std::uint64_t> g_bf11_queue_reservations{0};
std::atomic<std::uint64_t> g_bf11_bounded_fallbacks{0};
std::atomic<std::uint64_t> g_bf11_avoided_failed_attempt_extractions{0};
std::atomic<std::uint64_t> g_bf11_total_query_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_reset_seed_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_relaxation_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_target_check_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_status_copy_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_stream_sync_cpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_target_summary_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_target_prefix_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_reconstruction_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_output_transfer_gpu_nanoseconds{0};
std::atomic<std::uint64_t> g_bf11_telemetry_iterations{0};
std::atomic<std::uint64_t> g_bf11_frontier_vertices_processed{0};
std::atomic<std::uint64_t> g_bf11_edges_examined{0};
std::atomic<std::uint64_t> g_bf11_successful_relaxations{0};
std::atomic<std::uint64_t> g_bf11_touched_vertices{0};
std::atomic<std::uint64_t> g_bf11_maximum_touched_vertices{0};
std::atomic<std::uint64_t> g_bf11_telemetry_graph_rows{0};
std::atomic<std::uint64_t> g_bf11_workspace_device_bytes_total{0};
std::atomic<std::uint64_t> g_bf11_workspace_device_bytes_per_worker_max{0};
std::atomic<std::uint64_t> g_bf11_workspace_device_bytes_current_total{0};
std::atomic<std::uint64_t> g_bf11_gpu_free_before_workers{0};
std::atomic<std::uint64_t> g_bf11_gpu_free_after_workers{0};
std::atomic<std::uint64_t> g_bf11_constructed_workers{0};
std::atomic<std::uint64_t> g_bf11_query_telemetry_sequence{0};
std::mutex g_bf11_query_telemetry_output_mutex;
// Tests can force the wrap path without allocating a 32-bit number of rounds.
// Production retains the full nonzero uint32 token range.
std::atomic<std::uint64_t> g_bf11_mark_generation_limit{
    std::numeric_limits<std::uint32_t>::max()};

#if defined(BF11_ENABLE_HIP_GRAPHS)
// Test-only rendezvous. A zero participant count is the production default and
// takes only the atomic fast path on a workspace's first capture. The AMD
// regression enables it so every worker has actually begun capture before any
// captured kernel is enqueued, deterministically exercising concurrent first
// use rather than relying on host scheduling luck.
std::mutex g_bf11_graph_capture_barrier_mutex;
std::condition_variable g_bf11_graph_capture_barrier_condition;
std::atomic<int> g_bf11_graph_capture_barrier_participants{0};
int g_bf11_graph_capture_barrier_arrived = 0;
std::uint64_t g_bf11_graph_capture_barrier_generation = 0;

void wait_for_bf11_graph_capture_barrier() {
  if (g_bf11_graph_capture_barrier_participants.load(
          std::memory_order_acquire) <= 1) {
    return;
  }
  std::unique_lock<std::mutex> lock(g_bf11_graph_capture_barrier_mutex);
  const int participants =
      g_bf11_graph_capture_barrier_participants.load(
          std::memory_order_relaxed);
  if (participants <= 1) return;
  const std::uint64_t generation =
      g_bf11_graph_capture_barrier_generation;
  ++g_bf11_graph_capture_barrier_arrived;
  if (g_bf11_graph_capture_barrier_arrived == participants) {
    g_bf11_graph_capture_barrier_arrived = 0;
    ++g_bf11_graph_capture_barrier_generation;
    g_bf11_graph_capture_barrier_condition.notify_all();
    return;
  }
  g_bf11_graph_capture_barrier_condition.wait(lock, [&] {
    return g_bf11_graph_capture_barrier_generation != generation;
  });
}
#endif

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
  g_bf11_rounds.store(0, std::memory_order_relaxed);
  g_bf11_segments.store(0, std::memory_order_relaxed);
  g_bf11_no_op_segment_rounds.store(0, std::memory_order_relaxed);
  g_bf11_direct_segments.store(0, std::memory_order_relaxed);
  g_bf11_hip_graph_segments.store(0, std::memory_order_relaxed);
  g_bf11_status_copies.store(0, std::memory_order_relaxed);
  g_bf11_stream_synchronizations.store(0, std::memory_order_relaxed);
  g_bf11_graph_fallbacks.store(0, std::memory_order_relaxed);
  g_bf11_adaptive_dense_state_resets.store(0, std::memory_order_relaxed);
  g_bf11_constant_one_queries.store(0, std::memory_order_relaxed);
  g_bf11_static_cost_queries.store(0, std::memory_order_relaxed);
  g_bf11_dynamic_cost_queries.store(0, std::memory_order_relaxed);
  g_bf11_first_discoveries.store(0, std::memory_order_relaxed);
  g_bf11_mark_cas_attempts.store(0, std::memory_order_relaxed);
  g_bf11_mark_cas_wins.store(0, std::memory_order_relaxed);
  g_bf11_queue_reservations.store(0, std::memory_order_relaxed);
  g_bf11_bounded_fallbacks.store(0, std::memory_order_relaxed);
  g_bf11_avoided_failed_attempt_extractions.store(0,
                                                   std::memory_order_relaxed);
  g_bf11_total_query_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_reset_seed_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_relaxation_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_target_check_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_status_copy_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_stream_sync_cpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_target_summary_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_target_prefix_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_reconstruction_gpu_nanoseconds.store(0, std::memory_order_relaxed);
  g_bf11_output_transfer_gpu_nanoseconds.store(0,
                                                std::memory_order_relaxed);
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
  g_bf11_workspace_device_bytes_current_total.store(
      0, std::memory_order_relaxed);
  g_bf11_gpu_free_before_workers.store(0, std::memory_order_relaxed);
  g_bf11_gpu_free_after_workers.store(0, std::memory_order_relaxed);
  g_bf11_constructed_workers.store(0, std::memory_order_relaxed);
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
  g_bf11_query_telemetry_sequence.store(0, std::memory_order_relaxed);
  reset_bf11_telemetry_counters();
}

void configure_bellman_ford11_runtime_stats(
    bool telemetry_enabled,
    std::uint64_t requested_workers,
    std::uint64_t effective_workers,
    std::uint64_t gpu_free_before_workers) {
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
  stats.rounds = g_bf11_rounds.load(std::memory_order_relaxed);
  stats.segments = g_bf11_segments.load(std::memory_order_relaxed);
  stats.no_op_segment_rounds =
      g_bf11_no_op_segment_rounds.load(std::memory_order_relaxed);
  stats.direct_segments =
      g_bf11_direct_segments.load(std::memory_order_relaxed);
  stats.hip_graph_segments =
      g_bf11_hip_graph_segments.load(std::memory_order_relaxed);
  stats.status_copies = g_bf11_status_copies.load(std::memory_order_relaxed);
  stats.stream_synchronizations =
      g_bf11_stream_synchronizations.load(std::memory_order_relaxed);
  stats.graph_fallbacks =
      g_bf11_graph_fallbacks.load(std::memory_order_relaxed);
  stats.adaptive_dense_state_resets =
      g_bf11_adaptive_dense_state_resets.load(std::memory_order_relaxed);
  stats.constant_one_queries =
      g_bf11_constant_one_queries.load(std::memory_order_relaxed);
  stats.static_cost_queries =
      g_bf11_static_cost_queries.load(std::memory_order_relaxed);
  stats.dynamic_cost_queries =
      g_bf11_dynamic_cost_queries.load(std::memory_order_relaxed);
  stats.first_discoveries =
      g_bf11_first_discoveries.load(std::memory_order_relaxed);
  stats.mark_cas_attempts =
      g_bf11_mark_cas_attempts.load(std::memory_order_relaxed);
  stats.mark_cas_wins =
      g_bf11_mark_cas_wins.load(std::memory_order_relaxed);
  stats.queue_reservations =
      g_bf11_queue_reservations.load(std::memory_order_relaxed);
  stats.bounded_fallbacks =
      g_bf11_bounded_fallbacks.load(std::memory_order_relaxed);
  stats.avoided_failed_attempt_extractions =
      g_bf11_avoided_failed_attempt_extractions.load(
          std::memory_order_relaxed);
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
  stats.target_prefix_gpu_nanoseconds =
      g_bf11_target_prefix_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.path_reconstruction_gpu_nanoseconds =
      g_bf11_reconstruction_gpu_nanoseconds.load(std::memory_order_relaxed);
  stats.output_transfer_gpu_nanoseconds =
      g_bf11_output_transfer_gpu_nanoseconds.load(std::memory_order_relaxed);
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
  stats.workspace_device_bytes_current_total =
      g_bf11_workspace_device_bytes_current_total.load(
          std::memory_order_relaxed);
  stats.gpu_free_before_workers =
      g_bf11_gpu_free_before_workers.load(std::memory_order_relaxed);
  stats.gpu_free_after_workers =
      g_bf11_gpu_free_after_workers.load(std::memory_order_relaxed);
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

extern "C" void bf11_internal_set_mark_generation_limit(
    std::uint64_t limit) {
  const std::uint64_t maximum =
      std::numeric_limits<std::uint32_t>::max();
  g_bf11_mark_generation_limit.store(
      std::max<std::uint64_t>(1, std::min(limit, maximum)),
      std::memory_order_relaxed);
}

extern "C" void bf11_internal_set_graph_capture_barrier(int participants) {
#if defined(BF11_ENABLE_HIP_GRAPHS)
  std::lock_guard<std::mutex> lock(g_bf11_graph_capture_barrier_mutex);
  g_bf11_graph_capture_barrier_participants.store(
      std::max(0, participants), std::memory_order_release);
  g_bf11_graph_capture_barrier_arrived = 0;
  ++g_bf11_graph_capture_barrier_generation;
  g_bf11_graph_capture_barrier_condition.notify_all();
#else
  (void)participants;
#endif
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
static_assert(alignof(unsigned long long) >= 8,
              "BF11 packed state requires naturally aligned 64-bit atomics");

struct DeviceGraph {
  Offset rows = 0;
  Offset nnz = 0;
  const DeviceOffset* rowptr = nullptr;
  const Index* to = nullptr;
  // One sequential load supplies the complete immutable static cost for the
  // general weighted path. Constant-one queries do not touch this array.
  const float* static_edge_cost = nullptr;
  const std::int32_t* route_end_x = nullptr;
  const std::int32_t* route_end_y = nullptr;
  int constant_one = 0;
};

enum class DeviceCostMode : int {
  kConstantOne = 0,
  kStatic = 1,
  kDynamic = 2,
};

struct alignas(64) QueueStatus {
  int next_count = 0;
  int error_status = 0;
};

struct alignas(64) MinimumStatus {
  unsigned int min_next_frontier_dist_bits = kInfinityBits;
};

struct alignas(64) TargetCheckStatus {
  int reached_target_count = 0;
  unsigned int max_target_dist_bits = 0;
  int requested = 0;
  int certificate_due = 0;
  int current = 0;
};

// The explicit-stream controller is entirely device resident inside a
// segment. Hot queue-tail, minimum-label, and target fields occupy distinct
// cache-line-aligned regions so their atomics do not contend on one line.
struct alignas(64) ControllerDescriptor {
  QueueStatus queue{};
  MinimumStatus minimum{};
  TargetCheckStatus target{};
  unsigned long long phase_start_wall_tick = 0;
  int iterations_used = 0;
  int frontier_count = 0;
  int current_frontier_index = 0;
  int converged = 0;
  int early_stopped = 0;
  int hit_max_iters = 0;
  int done = 0;
  int target_checks = 0;
  int all_targets_reached = 0;
  int max_iters = 0;
  int target_count = 0;
  int target_check_interval = 1;
  unsigned int mark_token_base = 1;
  int rounds_executed = 0;
  int no_op_rounds = 0;
  int reset_mode = 0;
  int cost_mode = static_cast<int>(DeviceCostMode::kStatic);
};

struct ExtractionHeader {
  unsigned long long total_nodes = 0;
  unsigned long long total_edges = 0;
  unsigned long long required_nodes = 0;
  unsigned long long required_edges = 0;
  int status = 0;
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
static_assert(alignof(ControllerDescriptor) >= 64,
              "BF11 controller hot fields must remain cache-line aligned");
static_assert(sizeof(ControllerDescriptor) ==
                  bf11_worker_policy::kControllerDescriptorBytes,
              "BF11 controller and worker memory policy diverged");
static_assert(sizeof(ExtractionHeader) ==
                  bf11_worker_policy::kExtractionHeaderBytes,
              "BF11 extraction and worker memory policy diverged");
static_assert(sizeof(TargetSummary) == 32,
              "BF11 TargetSummary device-memory estimate changed");
static_assert(sizeof(Index) == 4 && sizeof(DeviceOffset) == 4 &&
                  sizeof(int) == 4 && sizeof(float) == 4,
              "BF11 per-vertex device-memory estimate changed");
static_assert(sizeof(unsigned long long) + 3 * sizeof(Index) + sizeof(int) ==
                  24,
              "BF11 graph-sized workspace estimate changed");

struct DeviceTelemetryCounters {
  unsigned long long frontier_vertices_processed = 0;
  unsigned long long edges_examined = 0;
  unsigned long long successful_relaxations = 0;
  unsigned long long first_discoveries = 0;
  unsigned long long mark_cas_attempts = 0;
  unsigned long long mark_cas_wins = 0;
  unsigned long long queue_reservations = 0;
  unsigned long long sparse_resets = 0;
  unsigned long long dense_resets = 0;
  unsigned long long reset_seed_wall_ticks = 0;
  unsigned long long relaxation_wall_ticks = 0;
  unsigned long long target_check_wall_ticks = 0;
};

static_assert(sizeof(DeviceTelemetryCounters) == 96,
              "BF11 telemetry device-memory estimate changed");

struct TelemetryEventPair {
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  bool pending = false;
};

struct QueryTelemetry {
  std::uint64_t attempts = 0;
  std::uint64_t iterations = 0;
  std::uint64_t rounds = 0;
  std::uint64_t no_op_rounds = 0;
  std::uint64_t target_checks = 0;
  std::uint64_t frontier_vertices_processed = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t successful_relaxations = 0;
  std::uint64_t first_discoveries = 0;
  std::uint64_t mark_cas_attempts = 0;
  std::uint64_t mark_cas_wins = 0;
  std::uint64_t queue_reservations = 0;
  std::uint64_t touched_vertices = 0;
  std::uint64_t reset_seed_gpu_nanoseconds = 0;
  std::uint64_t relaxation_gpu_nanoseconds = 0;
  std::uint64_t target_check_gpu_nanoseconds = 0;
};

struct DeviceGraphOwner {
  DeviceGraph view{};
  DeviceOffset* rowptr = nullptr;
  Index* to = nullptr;
  float* static_edge_cost = nullptr;
  std::int32_t* route_end_x = nullptr;
  std::int32_t* route_end_y = nullptr;
  bool constant_one = false;
};

struct DeviceWorkspace {
  Offset rows = 0;
  hipStream_t stream = nullptr;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  unsigned int* next_marks = nullptr;
  // Query reset is sparse after construction. Every node whose packed state
  // or frontier mark can differ from the default state appears
  // exactly once in touched_nodes[0:touched_count]. Sources publish themselves
  // while seeding; a non-source is published by the unique relaxation that
  // replaces its infinity label. Any future writer of those arrays must
  // preserve this completeness invariant.
  Index* touched_nodes = nullptr;
  int* touched_count = nullptr;
  // Null means the immutable identity multiplier. Allocate lazily on the
  // first sparse/full non-identity update; a complete all-ones replacement
  // re-enters identity mode without paying hot-loop dynamic loads.
  float* dynamic_vertex_cost = nullptr;
  bool dynamic_cost_identity = true;
  bool dynamic_storage_identity = false;
  // A failed update may have partially modified the live device array. Runs
  // reject a poisoned non-identity epoch until a complete replacement commits.
  // Identity mode remains valid because traversal never reads the array.
  bool dynamic_cost_epoch_valid = true;
  ControllerDescriptor* controller = nullptr;
  ControllerDescriptor* host_controller = nullptr;
  unsigned int next_mark_generation = 1;
  Index* source_nodes = nullptr;
  Index* host_source_nodes = nullptr;
  Offset source_capacity = 0;
  Index* target_nodes = nullptr;
  Index* host_target_nodes = nullptr;
  Offset target_capacity = 0;
  Index* update_nodes = nullptr;
  float* update_costs = nullptr;
  Offset update_capacity = 0;
  TargetSummary* target_summaries = nullptr;
  TargetSummary* host_target_summaries = nullptr;
  unsigned long long* reconstruction_node_offsets = nullptr;
  unsigned long long* reconstruction_edge_offsets = nullptr;
  ExtractionHeader* extraction_header = nullptr;
  ExtractionHeader* host_extraction_header = nullptr;
  Offset reconstruction_capacity = 0;
  Index* compact_nodes = nullptr;
  Index* host_compact_nodes = nullptr;
  DeviceOffset* compact_edges = nullptr;
  DeviceOffset* host_compact_edges = nullptr;
  float* compact_edge_costs = nullptr;
  float* host_compact_edge_costs = nullptr;
  std::size_t compact_node_capacity = 0;
  std::size_t compact_edge_capacity = 0;
  // Device arenas retain their allocation high-water marks. The D2H windows
  // track recent useful output separately so one long path does not force all
  // later queries to transfer the full retained arenas.
  std::size_t compact_node_transfer_capacity = 0;
  std::size_t compact_edge_transfer_capacity = 0;
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
  TelemetryEventPair status_copy_events;
  TelemetryEventPair target_summary_events;
  TelemetryEventPair target_prefix_events;
  TelemetryEventPair reconstruction_events;
  TelemetryEventPair output_transfer_events;
  // Once graph setup/replay is known unavailable, every later segment in this
  // workspace goes direct. Keep this outside the compile guard so a graph-on
  // request in a graph-disabled binary also records only one sticky fallback.
  bool graph_disabled = false;
  QueryTelemetry query_telemetry;
#if defined(BF11_ENABLE_HIP_GRAPHS)
  enum class GraphFailureInjection {
    kNone,
    kBeginCapture,
    kCapturedEnqueue,
    kEndCapture,
    kInstantiate,
    kLaunch,
    // Launch the real cached graph successfully, then exercise the same
    // quiesce + whole-query restart path used for a non-atomic launch error.
    // This is a target-only recovery hook, not an emulated HIP failure.
    kLaunchAfterSubmit,
  };
  hipGraph_t segment_graph = nullptr;
  hipGraphExec_t segment_graph_exec = nullptr;
  int graph_segment_rounds = 0;
  int graph_cost_mode = -1;
  BellmanFord11BoundingBox graph_bounds{};
  GraphFailureInjection graph_failure_injection =
      GraphFailureInjection::kNone;
#endif
};

struct SsspStatus {
  int iterations_used = 0;
  bool converged = false;
  bool early_stopped = false;
  bool hit_max_iters = false;
  bool all_targets_reached = false;
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
      g_bf11_telemetry_queries.fetch_add(1, std::memory_order_relaxed);
      begin_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedQueryTelemetry() {
    if (enabled_ && completed_) {
      g_bf11_telemetry_completed_queries.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (!enabled_) return;
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin_);
    g_bf11_total_query_nanoseconds.fetch_add(
        static_cast<std::uint64_t>(elapsed.count()),
        std::memory_order_relaxed);
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
    create_telemetry_event_pair(workspace.status_copy_events);
    create_telemetry_event_pair(workspace.target_summary_events);
    create_telemetry_event_pair(workspace.target_prefix_events);
    create_telemetry_event_pair(workspace.reconstruction_events);
    create_telemetry_event_pair(workspace.output_transfer_events);
    int device = -1;
    check_hip(hipGetDevice(&device), "get BF11 telemetry HIP device");
    check_hip(hipDeviceGetAttribute(&workspace.wall_clock_rate_khz,
                                    hipDeviceAttributeWallClockRate, device),
              "get BF11 telemetry wall-clock rate");
  } catch (...) {
    destroy_telemetry_event_pair(workspace.reset_seed_events);
    destroy_telemetry_event_pair(workspace.status_copy_events);
    destroy_telemetry_event_pair(workspace.target_summary_events);
    destroy_telemetry_event_pair(workspace.target_prefix_events);
    destroy_telemetry_event_pair(workspace.reconstruction_events);
    destroy_telemetry_event_pair(workspace.output_transfer_events);
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
  g_bf11_stream_synchronizations.fetch_add(1, std::memory_order_relaxed);
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
  QueryTelemetry& query = workspace.query_telemetry;
  query.iterations += static_cast<std::uint64_t>(std::max(0, iterations_used));
  query.frontier_vertices_processed += counters.frontier_vertices_processed;
  query.edges_examined += counters.edges_examined;
  query.successful_relaxations += counters.successful_relaxations;
  query.first_discoveries += counters.first_discoveries;
  query.mark_cas_attempts += counters.mark_cas_attempts;
  query.mark_cas_wins += counters.mark_cas_wins;
  query.queue_reservations += counters.queue_reservations;
  g_bf11_telemetry_iterations.fetch_add(
      static_cast<std::uint64_t>(std::max(0, iterations_used)),
      std::memory_order_relaxed);
  g_bf11_frontier_vertices_processed.fetch_add(
      counters.frontier_vertices_processed, std::memory_order_relaxed);
  g_bf11_edges_examined.fetch_add(counters.edges_examined,
                                  std::memory_order_relaxed);
  g_bf11_successful_relaxations.fetch_add(
      counters.successful_relaxations, std::memory_order_relaxed);
  g_bf11_first_discoveries.fetch_add(counters.first_discoveries,
                                     std::memory_order_relaxed);
  g_bf11_mark_cas_attempts.fetch_add(counters.mark_cas_attempts,
                                     std::memory_order_relaxed);
  g_bf11_mark_cas_wins.fetch_add(counters.mark_cas_wins,
                                 std::memory_order_relaxed);
  g_bf11_queue_reservations.fetch_add(counters.queue_reservations,
                                      std::memory_order_relaxed);
  const int observed_touched = *workspace.host_touched_count;
  const std::uint64_t touched = observed_touched <= 0
      ? 0
      : std::min<std::uint64_t>(
            static_cast<std::uint64_t>(observed_touched),
            static_cast<std::uint64_t>(workspace.rows));
  query.touched_vertices += touched;
  g_bf11_touched_vertices.fetch_add(touched, std::memory_order_relaxed);
  atomic_max(g_bf11_maximum_touched_vertices, touched);
  g_bf11_telemetry_graph_rows.store(
      static_cast<std::uint64_t>(workspace.rows), std::memory_order_relaxed);
  const std::uint64_t reset_seed_ns = wall_ticks_to_nanoseconds(
      counters.reset_seed_wall_ticks, workspace.wall_clock_rate_khz);
  const std::uint64_t relaxation_ns = wall_ticks_to_nanoseconds(
      counters.relaxation_wall_ticks, workspace.wall_clock_rate_khz);
  const std::uint64_t target_check_ns = wall_ticks_to_nanoseconds(
      counters.target_check_wall_ticks, workspace.wall_clock_rate_khz);
  query.reset_seed_gpu_nanoseconds += reset_seed_ns;
  query.relaxation_gpu_nanoseconds += relaxation_ns;
  query.target_check_gpu_nanoseconds += target_check_ns;
  g_bf11_reset_seed_gpu_nanoseconds.fetch_add(
      reset_seed_ns,
      std::memory_order_relaxed);
  g_bf11_relaxation_gpu_nanoseconds.fetch_add(
      relaxation_ns,
      std::memory_order_relaxed);
  g_bf11_target_check_gpu_nanoseconds.fetch_add(
      target_check_ns,
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

__host__ __device__ __forceinline__ DeviceOffset state_predecessor_edge(
    unsigned long long state) {
  return static_cast<DeviceOffset>(state);
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

__device__ __forceinline__ unsigned int coherent_atomic_load_u32(
    unsigned int* address) {
#if defined(__has_builtin)
#if !defined(BF11_FORCE_CAS_ATOMIC_LOAD) && __has_builtin(__hip_atomic_load)
  return __hip_atomic_load(address, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
#else
  return atomicCAS(address, 0u, 0u);
#endif
#else
  return atomicCAS(address, 0u, 0u);
#endif
}

template <DeviceCostMode CostMode>
__device__ __forceinline__ float effective_edge_weight(
    const DeviceGraph& graph,
    Offset edge,
    Index destination,
    const float* __restrict__ dynamic_vertex_cost,
    bool* valid) {
  float weight = 1.0f;
  if constexpr (CostMode != DeviceCostMode::kConstantOne) {
    const float static_cost = graph.static_edge_cost[edge];
    if constexpr (CostMode == DeviceCostMode::kStatic) {
      weight = static_cost;
    } else {
      const float multiplier = dynamic_vertex_cost[destination];
      // Preserve the exact zero-multiplier contract without forming inf * 0.
      weight = multiplier == 0.0f ? 0.0f : static_cost * multiplier;
    }
  }
  *valid = finite_device(weight) && weight >= 0.0f;
  return weight;
}

struct RelaxPublication {
  unsigned int candidate_bits = kInfinityBits;
  Index destination = -1;
  bool enqueue = false;
  bool first_discovery = false;
};

template <DeviceCostMode CostMode, bool CollectTelemetry>
__device__ __forceinline__ RelaxPublication relax_edge(
    const DeviceGraph& graph,
    Offset edge,
    float from_distance,
    const float* __restrict__ dynamic_vertex_cost,
    const BellmanFord11BoundingBox& bounds,
    unsigned int mark_token,
    unsigned long long* __restrict__ best_state,
    unsigned int* __restrict__ next_marks,
    ControllerDescriptor* __restrict__ controller,
    DeviceTelemetryCounters* __restrict__ telemetry) {
  RelaxPublication publication;
  const Index dst = graph.to[edge];
  if (dst < 0 || static_cast<Offset>(dst) >= graph.rows) {
    atomicExch(&controller->queue.error_status, 1);
    return publication;
  }
  if (!node_admitted(graph, dst, bounds)) {
    return publication;
  }

  bool valid_weight = false;
  const float weight = effective_edge_weight<CostMode>(
      graph, edge, dst, dynamic_vertex_cost, &valid_weight);
  if (!valid_weight) {
    atomicExch(&controller->queue.error_status, 3);
    return publication;
  }
  const float candidate = from_distance + weight;
  if (!finite_device(candidate)) return publication;

  const AtomicRelaxResult relaxation =
      atomic_relax_strict(&best_state[dst], candidate,
                          static_cast<DeviceOffset>(edge));
  if (!relaxation.improved) {
    return publication;
  }
  if constexpr (CollectTelemetry) {
    atomicAdd(&telemetry->successful_relaxations, 1ULL);
    if (relaxation.first_discovery) {
      atomicAdd(&telemetry->first_discoveries, 1ULL);
    }
  }

  unsigned int observed = coherent_atomic_load_u32(&next_marks[dst]);
  while (observed != mark_token) {
    if constexpr (CollectTelemetry) {
      atomicAdd(&telemetry->mark_cas_attempts, 1ULL);
    }
    const unsigned int previous = atomicCAS(&next_marks[dst], observed,
                                             mark_token);
    if (previous == observed) {
      publication.enqueue = true;
      if constexpr (CollectTelemetry) {
        atomicAdd(&telemetry->mark_cas_wins, 1ULL);
      }
      break;
    }
    observed = previous;
  }
  publication.candidate_bits = __float_as_uint(candidate);
  publication.destination = dst;
  publication.first_discovery = relaxation.first_discovery;
  return publication;
}

__global__ void clear_state_kernel(Offset rows,
                                   unsigned long long* best_state,
                                   unsigned int* next_marks) {
  const Offset row = logical_thread_id();
  if (row >= rows) return;
  best_state[row] = pack_state(kInfinityBits, kNoPredecessor);
  next_marks[row] = 0;
}

template <bool CollectTelemetry>
__global__ void clear_touched_state_kernel(
    Offset rows,
    const Index* touched_nodes,
    const int* touched_count,
    Offset dense_threshold_count,
    unsigned long long* best_state,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  __shared__ int shared_count;
  __shared__ int dense;
  if (threadIdx.x == 0) {
    // The list is immutable for this kernel and the preceding work is ordered
    // on the same stream, so an atomic read-modify-write only adds contention.
    shared_count = *touched_count;
    const int nonnegative_count = shared_count > 0 ? shared_count : 0;
    dense = static_cast<Offset>(nonnegative_count) >= dense_threshold_count
                ? 1
                : 0;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
      controller->reset_mode = dense;
      if constexpr (CollectTelemetry) {
        if (dense) {
          atomicAdd(&telemetry->dense_resets, 1ULL);
        } else {
          atomicAdd(&telemetry->sparse_resets, 1ULL);
        }
      }
    }
  }
  __syncthreads();
  const Offset thread = logical_thread_id();
  const Offset thread_count =
      static_cast<Offset>(gridDim.x) * static_cast<Offset>(gridDim.y) *
      static_cast<Offset>(blockDim.x);
  if (dense) {
    for (Offset row = thread; row < rows; row += thread_count) {
      best_state[row] = pack_state(kInfinityBits, kNoPredecessor);
    }
  } else {
    const int observed_count = shared_count;
    Offset count = observed_count <= 0 ? 0 : static_cast<Offset>(observed_count);
    if (count > rows) count = rows;
    for (Offset item = thread; item < count; item += thread_count) {
      const Index node = touched_nodes[item];
      if (node < 0 || static_cast<Offset>(node) >= rows) continue;
      best_state[node] = pack_state(kInfinityBits, kNoPredecessor);
    }
  }
}

__global__ void clear_marks_kernel(Offset rows, unsigned int* next_marks) {
  const Offset row = logical_thread_id();
  if (row < rows) next_marks[row] = 0;
}

__global__ void seed_sources_kernel(const Index* source_nodes,
                                    Offset source_count,
                                    unsigned long long* best_state,
                                    Index* frontier,
                                    Index* touched_nodes,
                                    int* touched_count) {
  const Offset item = logical_thread_id();
  if (item >= source_count) return;
  const Index source = source_nodes[item];
  best_state[source] = pack_state(0u, kNoPredecessor);
  // Sources are stable-deduplicated before upload, so this prefix contains no
  // duplicate touched entries and needs no queue-tail atomic.
  touched_nodes[item] = source;
  frontier[item] = source;
  if (item == 0) *touched_count = static_cast<int>(source_count);
}

__global__ void initialize_controller_kernel(
    Offset source_count,
    Offset target_count,
    int max_iters,
    int target_check_interval,
    unsigned int mark_token_base,
    DeviceCostMode cost_mode,
    ControllerDescriptor* controller) {
  if (logical_thread_id() != 0) return;
  const int reset_mode = controller->reset_mode;
  *controller = {};
  controller->frontier_count = static_cast<int>(source_count);
  controller->max_iters = max_iters;
  controller->target_count = static_cast<int>(target_count);
  controller->target_check_interval = target_check_interval;
  controller->mark_token_base = mark_token_base;
  controller->cost_mode = static_cast<int>(cost_mode);
  controller->reset_mode = reset_mode;
  if (max_iters == 0) {
    controller->hit_max_iters = 1;
    controller->done = 1;
  }
}

template <bool CollectTelemetry>
__global__ void begin_segment_round_kernel(
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  if (logical_thread_id() != 0) return;
  if (controller->done != 0) {
    ++controller->no_op_rounds;
    return;
  }
  controller->queue.next_count = 0;
  controller->queue.error_status = 0;
  controller->minimum.min_next_frontier_dist_bits = kInfinityBits;
  controller->target.reached_target_count = 0;
  controller->target.max_target_dist_bits = 0;
  const int completed = controller->iterations_used + 1;
  controller->target.certificate_due =
      completed % controller->target_check_interval == 0;
  controller->target.requested = controller->target.certificate_due != 0 ||
                                 completed >= controller->max_iters;
  controller->target.current = 0;
  if constexpr (CollectTelemetry) {
    (void)telemetry;
    controller->phase_start_wall_tick = wall_clock64();
  }
}

template <bool CollectTelemetry>
__device__ __forceinline__ void reserve_direct(
    Index value,
    Index* output,
    int* tail,
    Offset capacity,
    int error_code,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  const int slot = atomicAdd(tail, 1);
  if constexpr (CollectTelemetry) {
    atomicAdd(&telemetry->queue_reservations, 1ULL);
  }
  if (slot < 0 || static_cast<Offset>(slot) >= capacity) {
    atomicExch(&controller->queue.error_status, error_code);
    return;
  }
  output[slot] = value;
}

template <bool CollectTelemetry>
__device__ __forceinline__ void block_reserve_one(
    bool publish,
    Index value,
    Index* output,
    int* tail,
    Offset capacity,
    int error_code,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry,
    int* scan,
    int* shared_base,
    int* shared_count) {
  // Queue/touched slots are ordinary stores after one aggregate tail atomic.
  // They are never consumed in this kernel: the next ordinary round observes
  // them after a kernel boundary, and the cooperative controller observes
  // them only after grid.sync(). Those boundaries provide the publication
  // ordering, so stronger per-entry atomics would add cost without safety.
  const int lane = static_cast<int>(threadIdx.x);
  scan[lane] = publish ? 1 : 0;
  __syncthreads();
  if (lane == 0) {
    int count = 0;
    for (int item = 0; item < static_cast<int>(blockDim.x); ++item) {
      const int flag = scan[item];
      scan[item] = count;
      count += flag;
    }
    *shared_count = count;
    *shared_base = count == 0 ? 0 : atomicAdd(tail, count);
    if constexpr (CollectTelemetry) {
      if (count != 0) atomicAdd(&telemetry->queue_reservations, 1ULL);
    }
    if (count != 0 &&
        (*shared_base < 0 ||
         static_cast<unsigned long long>(*shared_base) +
                 static_cast<unsigned long long>(count) >
             static_cast<unsigned long long>(capacity))) {
      atomicExch(&controller->queue.error_status, error_code);
    }
  }
  __syncthreads();
  if (publish && *shared_count != 0 && *shared_base >= 0 &&
      static_cast<unsigned long long>(*shared_base) +
              static_cast<unsigned long long>(*shared_count) <=
          static_cast<unsigned long long>(capacity)) {
    output[*shared_base + scan[lane]] = value;
  }
  __syncthreads();
}

template <DeviceCostMode CostMode, bool CollectTelemetry>
__device__ __forceinline__ unsigned int relax_frontier_vertex(
    DeviceGraph graph,
    Index from,
    unsigned int mark_token,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    Index* next_frontier,
    unsigned int* next_marks,
    Index* touched_nodes,
    int* touched_count,
    ControllerDescriptor* controller,
    bool* pending_queue,
    Index* pending_queue_node,
    bool* pending_touched,
    Index* pending_touched_node,
    DeviceTelemetryCounters* telemetry) {
  if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
    atomicExch(&controller->queue.error_status, 1);
    return kInfinityBits;
  }
  // Current-frontier vertices can also be improved by another vertex in this
  // round. Observe the packed label coherently because those writers use CAS;
  // one relaxed load per frontier vertex avoids an atomic/non-atomic race.
  const float from_dist =
      state_distance(coherent_atomic_load(&best_state[from]));
  if (!finite_device(from_dist)) return kInfinityBits;
  const Offset begin = static_cast<Offset>(graph.rowptr[from]);
  const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
  if (end < begin || end > graph.nnz) {
    atomicExch(&controller->queue.error_status, 2);
    return kInfinityBits;
  }
  if constexpr (CollectTelemetry) {
    atomicAdd(&telemetry->edges_examined,
              static_cast<unsigned long long>(end - begin));
  }
  unsigned int local_min = kInfinityBits;
  for (Offset edge = begin; edge < end; ++edge) {
    const RelaxPublication publication = relax_edge<CostMode, CollectTelemetry>(
        graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
        best_state, next_marks, controller, telemetry);
    local_min = publication.candidate_bits < local_min
                    ? publication.candidate_bits
                    : local_min;
    if (publication.first_discovery) {
      if (!*pending_touched) {
        *pending_touched = true;
        *pending_touched_node = publication.destination;
      } else {
        reserve_direct<CollectTelemetry>(
            publication.destination, touched_nodes, touched_count, graph.rows,
            5, controller, telemetry);
      }
    }
    if (publication.enqueue) {
      if (!*pending_queue) {
        *pending_queue = true;
        *pending_queue_node = publication.destination;
      } else {
        reserve_direct<CollectTelemetry>(
            publication.destination, next_frontier,
            &controller->queue.next_count, graph.rows, 4, controller,
            telemetry);
      }
    }
  }
  return local_min;
}

template <DeviceCostMode CostMode, bool CollectTelemetry>
__global__ void segmented_frontier_relax_kernel(
    DeviceGraph graph,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    Index* frontier,
    Index* next_frontier,
    unsigned int* next_marks,
    Index* touched_nodes,
    int* touched_count,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  if (controller->done != 0) return;
  __shared__ int scan[kBlockSize];
  __shared__ int shared_base;
  __shared__ int shared_count;
  __shared__ unsigned int block_min[kBlockSize];
  const Offset block = static_cast<Offset>(blockIdx.x) +
                       static_cast<Offset>(blockIdx.y) * gridDim.x;
  const Offset blocks = static_cast<Offset>(gridDim.x) * gridDim.y;
  const int frontier_count = controller->frontier_count;
  const bool first_buffer = controller->current_frontier_index == 0;
  const Index* current = first_buffer ? frontier : next_frontier;
  Index* next = first_buffer ? next_frontier : frontier;
  const unsigned int mark_token =
      controller->mark_token_base +
      static_cast<unsigned int>(controller->iterations_used);
  if constexpr (CollectTelemetry) {
    if (block == 0 && threadIdx.x == 0) {
      atomicAdd(&telemetry->frontier_vertices_processed,
                static_cast<unsigned long long>(frontier_count));
    }
  }
  unsigned int thread_min = kInfinityBits;
  for (Offset base = block * blockDim.x;
       base < static_cast<Offset>(frontier_count);
       base += blocks * blockDim.x) {
    const Offset item = base + threadIdx.x;
    bool pending_queue = false;
    bool pending_touched = false;
    Index queue_node = -1;
    Index touched_node = -1;
    if (item < static_cast<Offset>(frontier_count)) {
      const unsigned int candidate = relax_frontier_vertex<
          CostMode, CollectTelemetry>(
          graph, current[item], mark_token, dynamic_vertex_cost, bounds,
          best_state, next, next_marks, touched_nodes, touched_count,
          controller, &pending_queue, &queue_node, &pending_touched,
          &touched_node, telemetry);
      thread_min = candidate < thread_min ? candidate : thread_min;
    }
    block_reserve_one<CollectTelemetry>(
        pending_touched, touched_node, touched_nodes, touched_count, graph.rows,
        5, controller, telemetry, scan, &shared_base, &shared_count);
    block_reserve_one<CollectTelemetry>(
        pending_queue, queue_node, next, &controller->queue.next_count,
        graph.rows, 4, controller, telemetry, scan, &shared_base,
        &shared_count);
  }
  block_min[threadIdx.x] = thread_min;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride) {
      block_min[threadIdx.x] =
          min(block_min[threadIdx.x], block_min[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && block_min[0] != kInfinityBits) {
    atomicMin(&controller->minimum.min_next_frontier_dist_bits, block_min[0]);
  }
}

template <bool CollectTelemetry>
__global__ void update_target_status_kernel(
    const unsigned long long* best_state,
    const Index* target_nodes,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  if (controller->done != 0) return;
  const bool convergence_scan = controller->queue.next_count == 0;
  const Offset thread = logical_thread_id();
  if constexpr (CollectTelemetry) {
    if (thread == 0) {
      const unsigned long long now = wall_clock64();
      telemetry->relaxation_wall_ticks +=
          now - controller->phase_start_wall_tick;
      if (controller->target.requested != 0 || convergence_scan) {
        controller->phase_start_wall_tick = now;
      }
    }
  }
  if (controller->target.requested == 0 && !convergence_scan) return;
  const Offset thread_count =
      static_cast<Offset>(gridDim.x) * gridDim.y * blockDim.x;
  for (Offset item = thread;
       item < static_cast<Offset>(controller->target_count);
       item += thread_count) {
    const unsigned int bits =
        state_distance_bits(best_state[target_nodes[item]]);
    if (bits == kInfinityBits) continue;
    atomicAdd(&controller->target.reached_target_count, 1);
    atomicMax(&controller->target.max_target_dist_bits, bits);
  }
}

__device__ __forceinline__ void finalize_completed_round(
    ControllerDescriptor* controller) {
  const int completed = controller->iterations_used + 1;
  controller->iterations_used = completed;
  ++controller->rounds_executed;
  controller->frontier_count = controller->queue.next_count;
  const bool targets_scanned = controller->target.requested != 0 ||
                               controller->queue.next_count == 0;
  if (targets_scanned) {
    controller->target.current = 1;
    controller->all_targets_reached =
        controller->target.reached_target_count == controller->target_count;
    ++controller->target_checks;
  }
  const auto reason =
      bf11_execution_policy::completed_round_terminal_reason(
          controller->queue.error_status,
          static_cast<unsigned int>(controller->queue.next_count),
          controller->target.certificate_due != 0,
          controller->all_targets_reached != 0,
          controller->minimum.min_next_frontier_dist_bits,
          controller->target.max_target_dist_bits,
          static_cast<unsigned int>(completed),
          static_cast<unsigned int>(controller->max_iters));
  switch (reason) {
    case bf11_execution_policy::ControllerTerminalReason::kRunning:
      controller->current_frontier_index ^= 1;
      break;
    case bf11_execution_policy::ControllerTerminalReason::kConverged:
      controller->converged = 1;
      controller->done = 1;
      break;
    case bf11_execution_policy::ControllerTerminalReason::kTargetCertified:
      // Every future path starts at a next-frontier label no smaller than the
      // largest target label. Frozen nonnegative costs make this exact.
      controller->early_stopped = 1;
      controller->done = 1;
      break;
    case bf11_execution_policy::ControllerTerminalReason::kMaxIterations:
      controller->hit_max_iters = 1;
      controller->done = 1;
      break;
    default:
      controller->done = 1;
      break;
  }
}

template <bool CollectTelemetry>
__global__ void finalize_segment_round_kernel(
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  if (logical_thread_id() != 0 || controller->done != 0) return;
  if constexpr (CollectTelemetry) {
    if (controller->target.requested != 0 ||
        controller->queue.next_count == 0) {
      telemetry->target_check_wall_ticks +=
          wall_clock64() - controller->phase_start_wall_tick;
    }
  }
  finalize_completed_round(controller);
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

template <DeviceCostMode CostMode, bool CollectTelemetry>
__global__ void frontier_controller_kernel(
    DeviceGraph graph,
    const Index* target_nodes,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    Index* frontier,
    Index* next_frontier,
    unsigned int* next_marks,
    Index* touched_nodes,
    int* touched_count,
    ControllerDescriptor* controller,
    DeviceTelemetryCounters* telemetry) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const Offset thread = static_cast<Offset>(grid.thread_rank());
  const Offset block = static_cast<Offset>(blockIdx.x);
  const Offset blocks = static_cast<Offset>(gridDim.x);
  __shared__ int scan[kBlockSize];
  __shared__ int shared_base;
  __shared__ int shared_count;
  __shared__ unsigned int block_min[kBlockSize];

  while (controller->done == 0) {
    unsigned long long relaxation_start = 0;
    if (thread == 0) {
      controller->queue.next_count = 0;
      controller->queue.error_status = 0;
      controller->minimum.min_next_frontier_dist_bits = kInfinityBits;
      controller->target.reached_target_count = 0;
      controller->target.max_target_dist_bits = 0;
      const int completed = controller->iterations_used + 1;
      controller->target.certificate_due =
          completed % controller->target_check_interval == 0;
      controller->target.requested =
          controller->target.certificate_due != 0 ||
          completed >= controller->max_iters;
      controller->target.current = 0;
      if constexpr (CollectTelemetry) {
        relaxation_start = wall_clock64();
        telemetry->frontier_vertices_processed +=
            static_cast<unsigned long long>(controller->frontier_count);
      }
    }
    grid.sync();

    const int frontier_count = controller->frontier_count;
    const bool first_buffer = controller->current_frontier_index == 0;
    const Index* current = first_buffer ? frontier : next_frontier;
    Index* next = first_buffer ? next_frontier : frontier;
    const unsigned int mark_token =
        controller->mark_token_base +
        static_cast<unsigned int>(controller->iterations_used);
    unsigned int thread_min = kInfinityBits;
    for (Offset base = block * blockDim.x;
         base < static_cast<Offset>(frontier_count);
         base += blocks * blockDim.x) {
      const Offset item = base + threadIdx.x;
      bool pending_queue = false;
      bool pending_touched = false;
      Index queue_node = -1;
      Index touched_node = -1;
      if (item < static_cast<Offset>(frontier_count)) {
        const unsigned int candidate = relax_frontier_vertex<
            CostMode, CollectTelemetry>(
            graph, current[item], mark_token, dynamic_vertex_cost, bounds,
            best_state, next, next_marks, touched_nodes, touched_count,
            controller, &pending_queue, &queue_node, &pending_touched,
            &touched_node, telemetry);
        thread_min = candidate < thread_min ? candidate : thread_min;
      }
      block_reserve_one<CollectTelemetry>(
          pending_touched, touched_node, touched_nodes, touched_count,
          graph.rows, 5, controller, telemetry, scan, &shared_base,
          &shared_count);
      block_reserve_one<CollectTelemetry>(
          pending_queue, queue_node, next, &controller->queue.next_count,
          graph.rows, 4, controller, telemetry, scan, &shared_base,
          &shared_count);
    }
    block_min[threadIdx.x] = thread_min;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
      if (static_cast<int>(threadIdx.x) < stride) {
        block_min[threadIdx.x] =
            min(block_min[threadIdx.x], block_min[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && block_min[0] != kInfinityBits) {
      atomicMin(&controller->minimum.min_next_frontier_dist_bits,
                block_min[0]);
    }
    grid.sync();
    if constexpr (CollectTelemetry) {
      if (thread == 0) {
        telemetry->relaxation_wall_ticks +=
            wall_clock64() - relaxation_start;
      }
    }

    const bool scan_targets =
        controller->target.requested != 0 ||
        controller->queue.next_count == 0;
    unsigned long long target_start = 0;
    if constexpr (CollectTelemetry) {
      if (thread == 0 && scan_targets) target_start = wall_clock64();
      if (scan_targets) grid.sync();
    }
    if (scan_targets) {
      for (Offset item = thread;
           item < static_cast<Offset>(controller->target_count);
           item += static_cast<Offset>(grid.size())) {
        const unsigned int bits =
            state_distance_bits(best_state[target_nodes[item]]);
        if (bits != kInfinityBits) {
          atomicAdd(&controller->target.reached_target_count, 1);
          atomicMax(&controller->target.max_target_dist_bits, bits);
        }
      }
    }
    grid.sync();
    if constexpr (CollectTelemetry) {
      if (thread == 0 && scan_targets) {
        telemetry->target_check_wall_ticks +=
            wall_clock64() - target_start;
      }
    }
    if (thread == 0) {
      if (scan_targets) controller->target.current = 1;
      finalize_completed_round(controller);
    }
    grid.sync();
  }
}
__global__ void summarize_target_paths_kernel(
    const unsigned long long* best_state,
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
    const DeviceOffset edge = state_predecessor_edge(state);
    if (edge == kNoPredecessor) {
      // Sources are initialized to (distance=0,no predecessor). With frozen
      // nonnegative costs and strict-only relaxation, no source can ever
      // receive a better label or parent. This pair is therefore the complete
      // root predicate and a graph-sized source mask is unnecessary.
      if (state_distance_bits(state) == 0u) {
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

__global__ void prefix_target_paths_kernel(
    const TargetSummary* summaries,
    Offset target_count,
    unsigned long long node_capacity,
    unsigned long long edge_capacity,
    unsigned long long* node_offsets,
    unsigned long long* edge_offsets,
    ExtractionHeader* header) {
  if (logical_thread_id() != 0) return;
  ExtractionHeader result{};
  for (Offset item = 0; item < target_count; ++item) {
    node_offsets[item] = result.required_nodes;
    edge_offsets[item] = result.required_edges;
    const TargetSummary summary = summaries[item];
    if (summary.status == kTargetUnreachable) {
      if (summary.node_count != 0 || summary.edge_count != 0) {
        result.status = 1;
        break;
      }
      continue;
    }
    if (summary.status != kTargetPathValid) {
      result.status = 1;
      break;
    }
    if (summary.node_count == 0 ||
        summary.node_count != summary.edge_count + 1 ||
        summary.node_count > ~result.required_nodes ||
        summary.edge_count > ~result.required_edges) {
      result.status = 1;
      break;
    }
    result.required_nodes += summary.node_count;
    result.required_edges += summary.edge_count;
    if (result.required_nodes >
            static_cast<unsigned long long>(
                std::numeric_limits<int>::max()) ||
        result.required_edges >
            static_cast<unsigned long long>(
                std::numeric_limits<int>::max())) {
      result.status = 1;
      break;
    }
  }
  result.total_nodes = result.required_nodes;
  result.total_edges = result.required_edges;
  if (result.status == 0 &&
      (result.required_nodes > node_capacity ||
       result.required_edges > edge_capacity)) {
    result.status = 2;
  }
  *header = result;
}

template <DeviceCostMode CostMode>
__global__ void materialize_target_paths_kernel(
    const unsigned long long* best_state,
    Offset rows,
    Offset nnz,
    const Index* to,
    const DeviceOffset* rowptr,
    const Index* targets,
    const TargetSummary* summaries,
    const unsigned long long* node_offsets,
    const unsigned long long* edge_offsets,
    const ExtractionHeader* header,
    Offset target_count,
    Index* compact_nodes,
    DeviceOffset* compact_edges,
    DeviceGraph graph,
    const float* dynamic_vertex_cost,
    float* compact_edge_costs) {
  const Offset item = logical_thread_id();
  if (header->status != 0 || item >= target_count ||
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
    const DeviceOffset edge = state_predecessor_edge(best_state[current]);
    if (edge == kNoPredecessor || static_cast<Offset>(edge) >= nnz ||
        to[edge] != current) {
      return;
    }
    const Index predecessor = source_for_edge(rowptr, rows, edge);
    if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) return;
    compact_edges[edge_base + remaining - 1] = edge;
    bool valid_weight = false;
    compact_edge_costs[edge_base + remaining - 1] =
        effective_edge_weight<CostMode>(graph, edge, current,
                                        dynamic_vertex_cost,
                                        &valid_weight);
    if (!valid_weight) return;
    compact_nodes[node_base + remaining - 1] = predecessor;
    current = predecessor;
  }
  // summarize_target_paths_kernel already certified this condition. Keep the
  // reads here so malformed concurrent mutation cannot silently appear valid.
  const unsigned long long root_state = best_state[current];
  if (current != summary.root ||
      state_distance_bits(root_state) != 0u ||
      static_cast<DeviceOffset>(root_state) != kNoPredecessor) {
    return;
  }
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
      owner.static_edge_cost =
          device_allocate<float>(edges, "hipMalloc BF11 static edge costs");
      std::vector<float> static_costs(edges);
      owner.constant_one = true;
      for (std::size_t edge = 0; edge < edges; ++edge) {
        const Index destination = graph.colind[edge];
        const float edge_value = graph.values[edge];
        const float base =
            (*sidecars.base_vertex_cost)[static_cast<std::size_t>(destination)];
        const float cost = edge_value == 0.0f ? 0.0f : edge_value * base;
        static_costs[edge] = cost;
        owner.constant_one = owner.constant_one && cost == 1.0f;
      }
      check_hip(hipMemcpyAsync(owner.to, graph.colind.data(),
                               edges * sizeof(Index), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 destinations");
      check_hip(hipMemcpyAsync(owner.static_edge_cost, static_costs.data(),
                               edges * sizeof(float), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 static edge costs");
    } else {
      owner.constant_one = true;
    }
    const std::size_t rows = static_cast<std::size_t>(graph.rows);
    owner.route_end_x =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end X");
    owner.route_end_y =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end Y");
    check_hip(hipMemcpyAsync(owner.route_end_x, sidecars.route_end_x->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end X");
    check_hip(hipMemcpyAsync(owner.route_end_y, sidecars.route_end_y->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end Y");
    check_hip(hipStreamSynchronize(stream), "synchronize BF11 graph upload");

    owner.view = {graph.rows,
                  graph.nnz,
                  owner.rowptr,
                  owner.to,
                  owner.static_edge_cost,
                  owner.route_end_x,
                  owner.route_end_y,
                  owner.constant_one ? 1 : 0};
    return owner;
  } catch (...) {
    // An allocation failure can occur after earlier asynchronous copies were
    // queued. Drain them before releasing their destinations.
    (void)hipStreamSynchronize(stream);
    if (owner.rowptr) (void)hipFree(owner.rowptr);
    if (owner.to) (void)hipFree(owner.to);
    if (owner.static_edge_cost) (void)hipFree(owner.static_edge_cost);
    if (owner.route_end_x) (void)hipFree(owner.route_end_x);
    if (owner.route_end_y) (void)hipFree(owner.route_end_y);
    throw;
  }
}

void free_graph(DeviceGraphOwner* owner) noexcept {
  if (!owner) return;
  if (owner->rowptr) (void)hipFree(owner->rowptr);
  if (owner->to) (void)hipFree(owner->to);
  if (owner->static_edge_cost) (void)hipFree(owner->static_edge_cost);
  if (owner->route_end_x) (void)hipFree(owner->route_end_x);
  if (owner->route_end_y) (void)hipFree(owner->route_end_y);
  *owner = {};
}

DeviceWorkspace make_workspace(Offset rows,
                               hipStream_t stream,
                               bool telemetry_enabled) {
  DeviceWorkspace workspace;
  workspace.rows = rows;
  workspace.stream = stream;
  workspace.telemetry_enabled = telemetry_enabled;
#if defined(BF11_ENABLE_HIP_GRAPHS)
  const char* graph_failure =
      std::getenv("BF11_TEST_HIP_GRAPH_FAILURE_STAGE");
  if (graph_failure != nullptr && graph_failure[0] != '\0' &&
      std::strcmp(graph_failure, "none") != 0) {
    using Injection = DeviceWorkspace::GraphFailureInjection;
    if (std::strcmp(graph_failure, "begin") == 0) {
      workspace.graph_failure_injection = Injection::kBeginCapture;
    } else if (std::strcmp(graph_failure, "enqueue") == 0) {
      workspace.graph_failure_injection = Injection::kCapturedEnqueue;
    } else if (std::strcmp(graph_failure, "end") == 0) {
      workspace.graph_failure_injection = Injection::kEndCapture;
    } else if (std::strcmp(graph_failure, "instantiate") == 0) {
      workspace.graph_failure_injection = Injection::kInstantiate;
    } else if (std::strcmp(graph_failure, "launch") == 0) {
      workspace.graph_failure_injection = Injection::kLaunch;
    } else if (std::strcmp(graph_failure, "launch-after-submit") == 0) {
      workspace.graph_failure_injection = Injection::kLaunchAfterSubmit;
    } else {
      throw std::invalid_argument(
          "BF11_TEST_HIP_GRAPH_FAILURE_STAGE must be begin, enqueue, end, "
          "instantiate, launch, launch-after-submit, or none");
    }
  }
#endif
  try {
    const std::size_t count = static_cast<std::size_t>(rows);
    workspace.best_state = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 packed states");
    workspace.frontier =
        device_allocate<Index>(count, "hipMalloc BF11 frontier");
    workspace.next_frontier =
        device_allocate<Index>(count, "hipMalloc BF11 next frontier");
    workspace.next_marks =
        device_allocate<unsigned int>(count, "hipMalloc BF11 frontier marks");
    workspace.touched_nodes =
        device_allocate<Index>(count, "hipMalloc BF11 touched nodes");
    workspace.touched_count =
        device_allocate<int>(1, "hipMalloc BF11 touched count");
    workspace.controller = device_allocate<ControllerDescriptor>(
        1, "hipMalloc BF11 controller descriptor");
    workspace.host_controller = pinned_allocate<ControllerDescriptor>(
        1, "hipHostMalloc BF11 controller descriptor");
    workspace.extraction_header = device_allocate<ExtractionHeader>(
        1, "hipMalloc BF11 extraction header");
    workspace.host_extraction_header = pinned_allocate<ExtractionHeader>(
        1, "hipHostMalloc BF11 extraction header");
    // Pay the dense state initialization once per workspace. Successful query
    // reuse clears only the nodes recorded in touched_nodes.
    hipLaunchKernelGGL(clear_state_kernel, grid_for_items(rows),
                       dim3(kBlockSize), 0, stream, rows,
                       workspace.best_state, workspace.next_marks);
    check_hip(hipGetLastError(), "initialize BF11 search state");
    check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int), stream),
              "initialize BF11 touched count");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF11 workspace initialization");
    initialize_workspace_telemetry(workspace);
    if (telemetry_enabled) {
      g_bf11_workspace_state_initializations.fetch_add(
          1, std::memory_order_relaxed);
    }
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
    if (workspace.touched_nodes) (void)hipFree(workspace.touched_nodes);
    if (workspace.touched_count) (void)hipFree(workspace.touched_count);
    if (workspace.dynamic_vertex_cost)
      (void)hipFree(workspace.dynamic_vertex_cost);
    if (workspace.controller) (void)hipFree(workspace.controller);
    if (workspace.host_controller) (void)hipHostFree(workspace.host_controller);
    if (workspace.extraction_header) (void)hipFree(workspace.extraction_header);
    if (workspace.host_extraction_header)
      (void)hipHostFree(workspace.host_extraction_header);
    throw;
  }
}

void free_workspace(DeviceWorkspace* workspace) noexcept {
  if (!workspace) return;
  (void)hipStreamSynchronize(workspace->stream);
  destroy_telemetry_event_pair(workspace->reset_seed_events);
  destroy_telemetry_event_pair(workspace->status_copy_events);
  destroy_telemetry_event_pair(workspace->target_summary_events);
  destroy_telemetry_event_pair(workspace->target_prefix_events);
  destroy_telemetry_event_pair(workspace->reconstruction_events);
  destroy_telemetry_event_pair(workspace->output_transfer_events);
#if defined(BF11_ENABLE_HIP_GRAPHS)
  if (workspace->segment_graph_exec)
    (void)hipGraphExecDestroy(workspace->segment_graph_exec);
  if (workspace->segment_graph) (void)hipGraphDestroy(workspace->segment_graph);
#endif
  if (workspace->best_state) (void)hipFree(workspace->best_state);
  if (workspace->frontier) (void)hipFree(workspace->frontier);
  if (workspace->next_frontier) (void)hipFree(workspace->next_frontier);
  if (workspace->next_marks) (void)hipFree(workspace->next_marks);
  if (workspace->touched_nodes) (void)hipFree(workspace->touched_nodes);
  if (workspace->touched_count) (void)hipFree(workspace->touched_count);
  if (workspace->dynamic_vertex_cost)
    (void)hipFree(workspace->dynamic_vertex_cost);
  if (workspace->controller) (void)hipFree(workspace->controller);
  if (workspace->host_controller) (void)hipHostFree(workspace->host_controller);
  if (workspace->extraction_header) (void)hipFree(workspace->extraction_header);
  if (workspace->host_extraction_header)
    (void)hipHostFree(workspace->host_extraction_header);
  if (workspace->source_nodes) (void)hipFree(workspace->source_nodes);
  if (workspace->host_source_nodes) (void)hipHostFree(workspace->host_source_nodes);
  if (workspace->target_nodes) (void)hipFree(workspace->target_nodes);
  if (workspace->host_target_nodes) (void)hipHostFree(workspace->host_target_nodes);
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
  if (workspace->host_compact_nodes)
    (void)hipHostFree(workspace->host_compact_nodes);
  if (workspace->compact_edges) (void)hipFree(workspace->compact_edges);
  if (workspace->host_compact_edges)
    (void)hipHostFree(workspace->host_compact_edges);
  if (workspace->compact_edge_costs)
    (void)hipFree(workspace->compact_edge_costs);
  if (workspace->host_compact_edge_costs)
    (void)hipHostFree(workspace->host_compact_edge_costs);
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
  if (workspace.touched_nodes) add(rows, sizeof(*workspace.touched_nodes));
  if (workspace.touched_count) add(1, sizeof(*workspace.touched_count));
  if (workspace.dynamic_vertex_cost)
    add(rows, sizeof(*workspace.dynamic_vertex_cost));
  if (workspace.controller) add(1, sizeof(*workspace.controller));
  if (workspace.extraction_header) add(1, sizeof(*workspace.extraction_header));
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
  PATHFINDER_PROFILE_RANGE("bf11.reset.defensive_dense");
  begin_telemetry_event(workspace, workspace.reset_seed_events);
  hipLaunchKernelGGL(clear_state_kernel, grid_for_items(workspace.rows),
                     dim3(kBlockSize), 0, workspace.stream, workspace.rows,
                     workspace.best_state, workspace.next_marks);
  check_hip(hipGetLastError(), "defensively reset BF11 search state");
  check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int),
                           workspace.stream),
            "defensively reset BF11 touched count");
  end_telemetry_event(workspace, workspace.reset_seed_events);
  synchronize_query_stream(workspace,
                           "synchronize defensive BF11 state reset");
  accumulate_telemetry_event(workspace.reset_seed_events,
                             g_bf11_reset_seed_gpu_nanoseconds);
  if (workspace.telemetry_enabled) {
    g_bf11_defensive_dense_state_resets.fetch_add(
        1, std::memory_order_relaxed);
  }
  workspace.needs_full_state_reset = false;
  workspace.next_mark_generation = 1;
}

Offset geometric_capacity(Offset current, Offset required, Offset limit) {
  Offset result = std::max<Offset>(1, current);
  while (result < required) {
    if (result > limit / 2) return limit;
    result *= 2;
  }
  return result;
}

void invalidate_segment_graph(DeviceWorkspace& workspace) {
#if defined(BF11_ENABLE_HIP_GRAPHS)
  const hipGraphExec_t executable = workspace.segment_graph_exec;
  const hipGraph_t graph = workspace.segment_graph;
  workspace.segment_graph_exec = nullptr;
  workspace.segment_graph = nullptr;
  workspace.graph_segment_rounds = 0;
  workspace.graph_cost_mode = -1;
  workspace.graph_bounds = {};

  // Attempt both releases exactly once even if the first reports an error.
  // A cleanup error is not a recoverable graph-capability failure: surface it
  // instead of silently losing the only handle and continuing direct.
  hipError_t first_error = hipSuccess;
  if (executable != nullptr) {
    const hipError_t status = hipGraphExecDestroy(executable);
    if (status != hipSuccess) first_error = status;
  }
  if (graph != nullptr) {
    const hipError_t status = hipGraphDestroy(graph);
    if (status != hipSuccess && first_error == hipSuccess) first_error = status;
  }
  check_hip(first_error, "destroy cached BF11 HIP Graph resources");
#else
  (void)workspace;
#endif
}

bool same_bounds(const BellmanFord11BoundingBox& lhs,
                 const BellmanFord11BoundingBox& rhs) {
  return lhs.enabled == rhs.enabled && lhs.min_x == rhs.min_x &&
         lhs.max_x == rhs.max_x && lhs.min_y == rhs.min_y &&
         lhs.max_y == rhs.max_y;
}

void ensure_source_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.source_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.source_capacity, required, workspace.rows);
  Index* replacement = nullptr;
  Index* host_replacement = nullptr;
  try {
    replacement = device_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 source list");
    host_replacement = pinned_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipHostMalloc BF11 source list");
  } catch (...) {
    if (replacement) (void)hipFree(replacement);
    if (host_replacement) (void)hipHostFree(host_replacement);
    throw;
  }
  if (workspace.source_nodes) (void)hipFree(workspace.source_nodes);
  if (workspace.host_source_nodes) (void)hipHostFree(workspace.host_source_nodes);
  workspace.source_nodes = replacement;
  workspace.host_source_nodes = host_replacement;
  workspace.source_capacity = capacity;
}

void ensure_target_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.target_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.target_capacity, required, workspace.rows);
  Index* replacement = nullptr;
  Index* host_replacement = nullptr;
  try {
    replacement = device_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 target list");
    host_replacement = pinned_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipHostMalloc BF11 target list");
  } catch (...) {
    if (replacement) (void)hipFree(replacement);
    if (host_replacement) (void)hipHostFree(host_replacement);
    throw;
  }
  try {
    invalidate_segment_graph(workspace);
  } catch (...) {
    if (replacement) (void)hipFree(replacement);
    if (host_replacement) (void)hipHostFree(host_replacement);
    throw;
  }
  if (workspace.target_nodes) (void)hipFree(workspace.target_nodes);
  if (workspace.host_target_nodes) (void)hipHostFree(workspace.host_target_nodes);
  workspace.target_nodes = replacement;
  workspace.host_target_nodes = host_replacement;
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

void ensure_dynamic_cost_storage(DeviceWorkspace& workspace) {
  if (workspace.dynamic_vertex_cost) return;
  workspace.dynamic_vertex_cost = device_allocate<float>(
      static_cast<std::size_t>(workspace.rows),
      "hipMalloc BF11 lazy dynamic vertex costs");
  workspace.dynamic_storage_identity = false;
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
  if (required_nodes > workspace.compact_node_capacity) {
    const std::size_t capacity =
        grow_size(workspace.compact_node_capacity, required_nodes);
    Index* device_nodes = nullptr;
    Index* host_nodes = nullptr;
    try {
      device_nodes = device_allocate<Index>(
          capacity, "hipMalloc BF11 compact path nodes");
      host_nodes = pinned_allocate<Index>(
          capacity, "hipHostMalloc BF11 compact path nodes");
    } catch (...) {
      if (device_nodes) (void)hipFree(device_nodes);
      if (host_nodes) (void)hipHostFree(host_nodes);
      throw;
    }
    if (workspace.compact_nodes) (void)hipFree(workspace.compact_nodes);
    if (workspace.host_compact_nodes)
      (void)hipHostFree(workspace.host_compact_nodes);
    workspace.compact_nodes = device_nodes;
    workspace.host_compact_nodes = host_nodes;
    workspace.compact_node_capacity = capacity;
    if (workspace.compact_node_transfer_capacity == 0) {
      workspace.compact_node_transfer_capacity = capacity;
    }
  }
  if (required_edges > workspace.compact_edge_capacity) {
    const std::size_t capacity =
        grow_size(workspace.compact_edge_capacity, required_edges);
    DeviceOffset* device_edges = nullptr;
    DeviceOffset* host_edges = nullptr;
    float* device_costs = nullptr;
    float* host_costs = nullptr;
    try {
      device_edges = device_allocate<DeviceOffset>(
          capacity, "hipMalloc BF11 compact path edges");
      host_edges = pinned_allocate<DeviceOffset>(
          capacity, "hipHostMalloc BF11 compact path edges");
      device_costs = device_allocate<float>(
          capacity, "hipMalloc BF11 compact path edge costs");
      host_costs = pinned_allocate<float>(
          capacity, "hipHostMalloc BF11 compact path edge costs");
    } catch (...) {
      if (device_edges) (void)hipFree(device_edges);
      if (host_edges) (void)hipHostFree(host_edges);
      if (device_costs) (void)hipFree(device_costs);
      if (host_costs) (void)hipHostFree(host_costs);
      throw;
    }
    if (workspace.compact_edges) (void)hipFree(workspace.compact_edges);
    if (workspace.host_compact_edges)
      (void)hipHostFree(workspace.host_compact_edges);
    if (workspace.compact_edge_costs)
      (void)hipFree(workspace.compact_edge_costs);
    if (workspace.host_compact_edge_costs)
      (void)hipHostFree(workspace.host_compact_edge_costs);
    workspace.compact_edges = device_edges;
    workspace.host_compact_edges = host_edges;
    workspace.compact_edge_costs = device_costs;
    workspace.host_compact_edge_costs = host_costs;
    workspace.compact_edge_capacity = capacity;
    if (workspace.compact_edge_transfer_capacity == 0) {
      workspace.compact_edge_transfer_capacity = capacity;
    }
  }
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
            &blocks_per_cu,
            frontier_controller_kernel<DeviceCostMode::kDynamic, true>,
            kBlockSize, 0)
      : hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_cu,
            frontier_controller_kernel<DeviceCostMode::kDynamic, false>,
            kBlockSize, 0);
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

dim3 segmented_controller_grid(Offset rows) {
  constexpr Offset kMaximumBlocks = 256;
  const Offset required =
      std::max<Offset>(1, (rows + kBlockSize - 1) / kBlockSize);
  return dim3(static_cast<unsigned>(std::min(required, kMaximumBlocks)));
}

DeviceCostMode workspace_cost_mode(const DeviceGraph& graph,
                                   const DeviceWorkspace& workspace) {
  if (!workspace.dynamic_cost_identity &&
      !workspace.dynamic_cost_epoch_valid) {
    throw std::runtime_error(
        "BF11 dynamic costs require a complete replacement after a failed update");
  }
  if (!workspace.dynamic_cost_identity) return DeviceCostMode::kDynamic;
  return graph.constant_one != 0 ? DeviceCostMode::kConstantOne
                                 : DeviceCostMode::kStatic;
}

unsigned int reserve_query_mark_tokens(DeviceWorkspace& workspace,
                                       int max_iters) {
  const auto requested = static_cast<std::uint32_t>(std::max(0, max_iters));
  const auto limit = static_cast<std::uint32_t>(
      g_bf11_mark_generation_limit.load(std::memory_order_relaxed));
  const bf11_execution_policy::MarkTokenReservation reservation =
      bf11_execution_policy::reserve_mark_tokens(
          workspace.next_mark_generation, requested, limit);
  if (!reservation.valid) {
    throw std::overflow_error(
        "BF11 mark-token test range is smaller than one query");
  }
  if (reservation.dense_reset_required) {
    hipLaunchKernelGGL(clear_marks_kernel, grid_for_items(workspace.rows),
                       dim3(kBlockSize), 0, workspace.stream, workspace.rows,
                       workspace.next_marks);
    check_hip(hipGetLastError(), "reset BF11 frontier mark generations");
  }
  workspace.next_mark_generation = reservation.next_token;
  return requested == 0 ? workspace.next_mark_generation
                        : reservation.first_token;
}

void prepare_query_controller(const DeviceGraph& graph,
                              DeviceWorkspace& workspace,
                              Offset source_count,
                              Offset target_count,
                              int max_iters,
                              const BellmanFord11RunOptions& options,
                              double adaptive_reset_threshold,
                              DeviceCostMode cost_mode) {
  PATHFINDER_PROFILE_RANGE("bf11.reset_seed");
  if (workspace.telemetry_enabled) {
    check_hip(hipMemsetAsync(workspace.telemetry_counters, 0,
                             sizeof(DeviceTelemetryCounters),
                             workspace.stream),
              "reset BF11 telemetry counters");
  }
  const unsigned int mark_token_base =
      reserve_query_mark_tokens(workspace, max_iters);
  const Offset dense_threshold_count = std::min<Offset>(
      graph.rows,
      std::max<Offset>(
          1, static_cast<Offset>(std::ceil(
                 adaptive_reset_threshold *
                 static_cast<double>(graph.rows)))));
  begin_telemetry_event(workspace, workspace.reset_seed_events);
  if (workspace.telemetry_enabled) {
    hipLaunchKernelGGL(
        (clear_touched_state_kernel<true>), sparse_reset_grid(graph.rows),
        dim3(kBlockSize), 0, workspace.stream, graph.rows,
        workspace.touched_nodes, workspace.touched_count,
        dense_threshold_count, workspace.best_state, workspace.controller,
        workspace.telemetry_counters);
  } else {
    hipLaunchKernelGGL(
        (clear_touched_state_kernel<false>), sparse_reset_grid(graph.rows),
        dim3(kBlockSize), 0, workspace.stream, graph.rows,
        workspace.touched_nodes, workspace.touched_count,
        dense_threshold_count, workspace.best_state, workspace.controller,
        workspace.telemetry_counters);
  }
  check_hip(hipGetLastError(), "adaptively clear BF11 search state");
  check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(int),
                           workspace.stream),
            "reset BF11 touched count");
  hipLaunchKernelGGL(
      seed_sources_kernel, grid_for_items(source_count), dim3(kBlockSize), 0,
      workspace.stream, workspace.source_nodes, source_count,
      workspace.best_state, workspace.frontier, workspace.touched_nodes,
      workspace.touched_count);
  check_hip(hipGetLastError(), "seed BF11 sources");
  hipLaunchKernelGGL(
      initialize_controller_kernel, dim3(1), dim3(1), 0, workspace.stream,
      source_count, target_count, max_iters, options.target_check_interval,
      mark_token_base, cost_mode, workspace.controller);
  check_hip(hipGetLastError(), "initialize BF11 device controller");
  end_telemetry_event(workspace, workspace.reset_seed_events);
}

struct SegmentEnqueueResult {
  hipError_t status = hipSuccess;
  const char* operation = nullptr;
};

template <DeviceCostMode CostMode, bool CollectTelemetry>
SegmentEnqueueResult try_enqueue_one_segment_round(
    const DeviceGraph& graph,
    DeviceWorkspace& workspace,
    const BellmanFord11RunOptions& options) {
  hipLaunchKernelGGL((begin_segment_round_kernel<CollectTelemetry>), dim3(1),
                     dim3(1), 0, workspace.stream, workspace.controller,
                     workspace.telemetry_counters);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    return {status, "begin BF11 segmented round"};
  }
  hipLaunchKernelGGL(
      (segmented_frontier_relax_kernel<CostMode, CollectTelemetry>),
      segmented_controller_grid(graph.rows), dim3(kBlockSize), 0,
      workspace.stream, graph, workspace.dynamic_vertex_cost, options.bounds,
      workspace.best_state, workspace.frontier, workspace.next_frontier,
      workspace.next_marks, workspace.touched_nodes, workspace.touched_count,
      workspace.controller, workspace.telemetry_counters);
  status = hipGetLastError();
  if (status != hipSuccess) {
    return {status, "launch BF11 segmented relaxation"};
  }
  hipLaunchKernelGGL(
      (update_target_status_kernel<CollectTelemetry>),
      segmented_controller_grid(graph.rows),
      dim3(kBlockSize), 0, workspace.stream, workspace.best_state,
      workspace.target_nodes, workspace.controller,
      workspace.telemetry_counters);
  status = hipGetLastError();
  if (status != hipSuccess) {
    return {status, "launch BF11 segmented target check"};
  }
  hipLaunchKernelGGL((finalize_segment_round_kernel<CollectTelemetry>),
                     dim3(1), dim3(1), 0, workspace.stream,
                     workspace.controller, workspace.telemetry_counters);
  status = hipGetLastError();
  return {status, status == hipSuccess ? nullptr
                                      : "finalize BF11 segmented round"};
}

SegmentEnqueueResult try_enqueue_direct_segment(
    const DeviceGraph& graph,
    DeviceWorkspace& workspace,
    const BellmanFord11RunOptions& options,
    int segment_rounds,
    DeviceCostMode cost_mode) {
  for (int round = 0; round < segment_rounds; ++round) {
    SegmentEnqueueResult result;
    if (workspace.telemetry_enabled) {
      switch (cost_mode) {
        case DeviceCostMode::kConstantOne:
          result = try_enqueue_one_segment_round<
              DeviceCostMode::kConstantOne, true>(graph, workspace, options);
          break;
        case DeviceCostMode::kStatic:
          result = try_enqueue_one_segment_round<DeviceCostMode::kStatic,
                                                  true>(
              graph, workspace, options);
          break;
        case DeviceCostMode::kDynamic:
          result = try_enqueue_one_segment_round<DeviceCostMode::kDynamic,
                                                  true>(
              graph, workspace, options);
          break;
      }
    } else {
      switch (cost_mode) {
        case DeviceCostMode::kConstantOne:
          result = try_enqueue_one_segment_round<
              DeviceCostMode::kConstantOne, false>(graph, workspace, options);
          break;
        case DeviceCostMode::kStatic:
          result = try_enqueue_one_segment_round<DeviceCostMode::kStatic,
                                                  false>(
              graph, workspace, options);
          break;
        case DeviceCostMode::kDynamic:
          result = try_enqueue_one_segment_round<DeviceCostMode::kDynamic,
                                                  false>(
              graph, workspace, options);
          break;
      }
    }
    if (result.status != hipSuccess) return result;
  }
  return {};
}

void enqueue_direct_segment(const DeviceGraph& graph,
                            DeviceWorkspace& workspace,
                            const BellmanFord11RunOptions& options,
                            int segment_rounds,
                            DeviceCostMode cost_mode) {
  const SegmentEnqueueResult result = try_enqueue_direct_segment(
      graph, workspace, options, segment_rounds, cost_mode);
  if (result.status != hipSuccess) {
    check_hip(result.status, result.operation);
  }
}

#if defined(BF11_ENABLE_HIP_GRAPHS)
class HipGraphSegmentBackend {
 public:
  using Graph = hipGraph_t;
  using Executable = hipGraphExec_t;

  HipGraphSegmentBackend(const DeviceGraph& graph,
                         DeviceWorkspace& workspace,
                         const BellmanFord11RunOptions& options,
                         int segment_rounds,
                         DeviceCostMode cost_mode)
      : graph_(graph),
        workspace_(workspace),
        options_(options),
        segment_rounds_(segment_rounds),
        cost_mode_(cost_mode) {}

  bool disabled() const { return workspace_.graph_disabled; }
  bool has_cached_executable() const {
    return workspace_.segment_graph_exec != nullptr;
  }
  Graph null_graph() const { return nullptr; }
  Executable null_executable() const { return nullptr; }
  bool valid_graph(Graph graph) const { return graph != nullptr; }
  bool valid_executable(Executable executable) const {
    return executable != nullptr;
  }

  bool begin_capture() {
    // A stale error predating capture is not a recoverable Graph failure. It
    // must be surfaced before the fallback path is allowed to consume only
    // the status produced by a Graph API operation.
    check_hip(hipPeekAtLastError(),
              "check BF11 HIP Graph capture precondition");
    if (injected(DeviceWorkspace::GraphFailureInjection::kBeginCapture)) {
      return false;
    }
    const hipError_t status = hipStreamBeginCapture(
        workspace_.stream, hipStreamCaptureModeThreadLocal);
    // The production fast path is a no-op. The AMD regression holds every
    // successfully begun capture here until all configured workspaces have
    // called BeginCapture, guaranteeing overlap at first use.
    wait_for_bf11_graph_capture_barrier();
    if (status != hipSuccess) {
      record_explicit_failure(status);
      (void)verify_not_capturing();
    }
    return status == hipSuccess;
  }

  bool enqueue_captured_segment() {
    if (injected(DeviceWorkspace::GraphFailureInjection::kCapturedEnqueue)) {
      // Simulate a recoverable captured-enqueue failure without deliberately
      // invalidating the runtime stream. Some supported ROCm releases do not
      // restore an invalidated stream after EndCapture, so using a prohibited
      // synchronization here would make this test hook corrupt an otherwise
      // healthy caller-owned stream. EndCapture still releases the partial
      // capture, and the policy tests cover the invalidated/no-graph branch.
      return false;
    }
    const SegmentEnqueueResult result = try_enqueue_direct_segment(
        graph_, workspace_, options_, segment_rounds_, cost_mode_);
    // Kernel launch status is obtained through hipGetLastError above, so it
    // has already been consumed and must not be cleared a second time here.
    // Only capture-specific launch statuses are safe graph fallbacks. A bad
    // kernel configuration, invalid device function, or device-side fault
    // would also fail on the direct path and must remain fatal.
    if (result.status != hipSuccess &&
        !is_recoverable_capture_enqueue_error(result.status)) {
      record_consumed_unrecoverable_failure(result.status);
    }
    return result.status == hipSuccess;
  }

  bool end_capture(Graph* captured) {
    const hipError_t status = hipStreamEndCapture(workspace_.stream, captured);
    if (status != hipSuccess) record_explicit_failure(status);
    const bool capture_terminated = verify_not_capturing();
    return capture_terminated && status == hipSuccess &&
           !injected(DeviceWorkspace::GraphFailureInjection::kEndCapture);
  }

  bool instantiate(Executable* executable, Graph graph) {
    const hipError_t status =
        hipGraphInstantiate(executable, graph, nullptr, nullptr, 0);
    if (status != hipSuccess) record_explicit_failure(status);
    return status == hipSuccess &&
           !injected(DeviceWorkspace::GraphFailureInjection::kInstantiate);
  }

  void adopt(Graph graph, Executable executable) {
    workspace_.segment_graph = graph;
    workspace_.segment_graph_exec = executable;
    workspace_.graph_segment_rounds = segment_rounds_;
    workspace_.graph_cost_mode = static_cast<int>(cost_mode_);
    workspace_.graph_bounds = options_.bounds;
  }

  bool launch_cached() {
    if (injected(DeviceWorkspace::GraphFailureInjection::kLaunch)) {
      real_launch_attempted_ = false;
      return false;
    }
    const hipError_t preflight = hipPeekAtLastError();
    if (preflight != hipSuccess) {
      real_launch_attempted_ = false;
      record_unrecoverable_failure(preflight);
      return false;
    }
    real_launch_attempted_ = true;
    const hipError_t status =
        hipGraphLaunch(workspace_.segment_graph_exec, workspace_.stream);
    if (status != hipSuccess) record_explicit_failure(status);
    return status == hipSuccess &&
           !injected(
               DeviceWorkspace::GraphFailureInjection::kLaunchAfterSubmit);
  }

  bool launch_failure_requires_restart() const {
    return real_launch_attempted_;
  }

  void prepare_launch_failure() {
    if (!real_launch_attempted_) return;
    const hipError_t status = synchronize_graph_stream();
    if (status != hipSuccess) record_unrecoverable_failure(status);
  }

  void destroy_graph(Graph graph) {
    if (graph == nullptr) return;
    const hipError_t status = hipGraphDestroy(graph);
    if (status != hipSuccess) record_unrecoverable_failure(status);
  }

  void destroy_executable(Executable executable) {
    if (executable == nullptr) return;
    const hipError_t status = hipGraphExecDestroy(executable);
    if (status != hipSuccess) record_unrecoverable_failure(status);
  }

  void invalidate_cached() {
    const Executable executable = workspace_.segment_graph_exec;
    const Graph graph = workspace_.segment_graph;
    workspace_.segment_graph_exec = nullptr;
    workspace_.segment_graph = nullptr;
    workspace_.graph_segment_rounds = 0;
    workspace_.graph_cost_mode = -1;
    workspace_.graph_bounds = {};
    destroy_executable(executable);
    destroy_graph(graph);
  }

  void disable(bf11_graph_execution_policy::FailureStage) {
    workspace_.graph_disabled = true;
    if (workspace_.telemetry_enabled) {
      g_bf11_graph_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void finish_failure() {
    const hipError_t unexpected = unexpected_error_;
    unexpected_error_ = hipSuccess;
    if (capture_not_terminated_) {
      capture_not_terminated_ = false;
      throw std::runtime_error(
          "BF11 HIP Graph capture did not return the stream to non-capturing "
          "state");
    }
    if (unexpected != hipSuccess) {
      check_hip(unexpected,
                "BF11 HIP Graph fallback observed an unrelated HIP error");
    }
  }

 private:
  hipError_t synchronize_graph_stream() {
    const auto begin = std::chrono::steady_clock::now();
    const hipError_t status = hipStreamSynchronize(workspace_.stream);
    const auto end = std::chrono::steady_clock::now();
    if (workspace_.telemetry_enabled) {
      g_bf11_stream_synchronizations.fetch_add(1,
                                                std::memory_order_relaxed);
      g_bf11_stream_sync_cpu_nanoseconds.fetch_add(
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                  .count()),
          std::memory_order_relaxed);
    }
    return status;
  }

  static bool is_recoverable_capture_enqueue_error(hipError_t status) {
    const int code = static_cast<int>(status);
    return code >= static_cast<int>(hipErrorStreamCaptureUnsupported) &&
           code <= static_cast<int>(hipErrorStreamCaptureWrongThread);
  }

  bool injected(DeviceWorkspace::GraphFailureInjection stage) const {
    return workspace_.graph_failure_injection == stage;
  }

  bool verify_not_capturing() {
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    const hipError_t query_status =
        hipStreamIsCapturing(workspace_.stream, &capture_status);
    if (query_status != hipSuccess) {
      record_unrecoverable_failure(query_status);
      return false;
    }
    if (capture_status != hipStreamCaptureStatusNone) {
      capture_not_terminated_ = true;
      return false;
    }
    return true;
  }

  void record_explicit_failure(hipError_t expected) {
    // Explicit HIP API calls return their own status, but some runtimes also
    // leave it as the thread's last error. Consume that exact duplicate so a
    // direct kernel launch cannot inherit it. Preserve correctness by saving
    // any different (for example asynchronous) error for finish_failure().
    const hipError_t observed = hipGetLastError();
    if (observed != hipSuccess && observed != expected &&
        unexpected_error_ == hipSuccess) {
      unexpected_error_ = observed;
    }
  }

  void record_unrecoverable_failure(hipError_t status) {
    const hipError_t observed = hipGetLastError();
    if (unexpected_error_ == hipSuccess) {
      unexpected_error_ =
          observed != hipSuccess && observed != status ? observed : status;
    }
  }

  void record_consumed_unrecoverable_failure(hipError_t status) {
    if (unexpected_error_ == hipSuccess) unexpected_error_ = status;
  }

  const DeviceGraph& graph_;
  DeviceWorkspace& workspace_;
  const BellmanFord11RunOptions& options_;
  int segment_rounds_ = 1;
  DeviceCostMode cost_mode_ = DeviceCostMode::kStatic;
  hipError_t unexpected_error_ = hipSuccess;
  bool capture_not_terminated_ = false;
  bool real_launch_attempted_ = false;
};
#endif

bf11_graph_execution_policy::AttemptResult enqueue_graph_segment(
    const DeviceGraph& graph,
    DeviceWorkspace& workspace,
    const BellmanFord11RunOptions& options,
    int segment_rounds,
    DeviceCostMode cost_mode,
    BellmanFord11HipGraphMode graph_mode) {
  if (segment_rounds <= 1 ||
      graph_mode == BellmanFord11HipGraphMode::kOff) {
    return bf11_graph_execution_policy::AttemptResult::kDirectFallback;
  }
  if (workspace.graph_disabled) {
    return bf11_graph_execution_policy::AttemptResult::kDirectFallback;
  }
#if defined(BF11_ENABLE_HIP_GRAPHS)
  if (workspace.segment_graph_exec &&
      (workspace.graph_segment_rounds != segment_rounds ||
       workspace.graph_cost_mode != static_cast<int>(cost_mode) ||
       !same_bounds(workspace.graph_bounds, options.bounds))) {
    invalidate_segment_graph(workspace);
  }
  HipGraphSegmentBackend backend(graph, workspace, options, segment_rounds,
                                 cost_mode);
  return bf11_graph_execution_policy::try_launch(&backend);
#else
  (void)graph;
  (void)workspace;
  (void)options;
  (void)cost_mode;
  if (graph_mode == BellmanFord11HipGraphMode::kOn) {
    workspace.graph_disabled = true;
    if (workspace.telemetry_enabled) {
      g_bf11_graph_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return bf11_graph_execution_policy::AttemptResult::kDirectFallback;
#endif
}

ControllerDescriptor copy_controller_to_host(DeviceWorkspace& workspace,
                                             const char* synchronization) {
  PATHFINDER_PROFILE_RANGE("bf11.runtime_copy.controller_status");
  begin_telemetry_event(workspace, workspace.status_copy_events);
  check_hip(hipMemcpyAsync(workspace.host_controller, workspace.controller,
                           sizeof(ControllerDescriptor), hipMemcpyDeviceToHost,
                           workspace.stream),
            "copy BF11 controller descriptor");
  if (workspace.telemetry_enabled) {
    g_bf11_status_copies.fetch_add(1, std::memory_order_relaxed);
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
  end_telemetry_event(workspace, workspace.status_copy_events);
  synchronize_query_stream(workspace, synchronization);
  accumulate_telemetry_event(workspace.reset_seed_events,
                             g_bf11_reset_seed_gpu_nanoseconds);
  accumulate_telemetry_event(workspace.status_copy_events,
                             g_bf11_status_copy_gpu_nanoseconds);
  return *workspace.host_controller;
}

void record_controller_result(const DeviceGraph& graph,
                              const ControllerDescriptor& descriptor,
                              int max_iters,
                              DeviceWorkspace* workspace) {
  if (descriptor.queue.error_status != 0) {
    throw_controller_error(descriptor.queue.error_status);
  }
  const int termination_count = descriptor.converged +
                                descriptor.early_stopped +
                                descriptor.hit_max_iters;
  if (descriptor.done != 1 || descriptor.iterations_used < 0 ||
      descriptor.iterations_used > max_iters ||
      descriptor.frontier_count < 0 ||
      static_cast<Offset>(descriptor.frontier_count) > graph.rows ||
      descriptor.current_frontier_index < 0 ||
      descriptor.current_frontier_index > 1 ||
      descriptor.target_checks < 0 || termination_count != 1) {
    throw std::runtime_error("BF11 device controller returned bad status");
  }
  if (workspace != nullptr && workspace->telemetry_enabled) {
    workspace->query_telemetry.attempts += 1;
    workspace->query_telemetry.rounds +=
        static_cast<std::uint64_t>(descriptor.rounds_executed);
    workspace->query_telemetry.no_op_rounds +=
        static_cast<std::uint64_t>(descriptor.no_op_rounds);
    workspace->query_telemetry.target_checks +=
        static_cast<std::uint64_t>(descriptor.target_checks);
    g_bf11_rounds.fetch_add(
        static_cast<std::uint64_t>(descriptor.rounds_executed),
        std::memory_order_relaxed);
    g_bf11_no_op_segment_rounds.fetch_add(
        static_cast<std::uint64_t>(descriptor.no_op_rounds),
        std::memory_order_relaxed);
    g_bf11_target_checks.fetch_add(
        static_cast<std::uint64_t>(descriptor.target_checks),
        std::memory_order_relaxed);
    if (descriptor.reset_mode == 0) {
      g_bf11_sparse_state_resets.fetch_add(1, std::memory_order_relaxed);
    } else {
      g_bf11_adaptive_dense_state_resets.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
}

template <DeviceCostMode CostMode, bool CollectTelemetry>
hipError_t launch_cooperative_controller(const DeviceGraph& graph,
                                         DeviceWorkspace& workspace,
                                         const BellmanFord11RunOptions& options,
                                         int blocks) {
  DeviceGraph graph_arg = graph;
  const Index* targets_arg = workspace.target_nodes;
  const float* dynamic_arg = workspace.dynamic_vertex_cost;
  BellmanFord11BoundingBox bounds_arg = options.bounds;
  unsigned long long* state_arg = workspace.best_state;
  Index* frontier_arg = workspace.frontier;
  Index* next_arg = workspace.next_frontier;
  unsigned int* marks_arg = workspace.next_marks;
  Index* touched_arg = workspace.touched_nodes;
  int* touched_count_arg = workspace.touched_count;
  ControllerDescriptor* controller_arg = workspace.controller;
  DeviceTelemetryCounters* telemetry_arg = workspace.telemetry_counters;
  void* args[] = {&graph_arg,      &targets_arg, &dynamic_arg,
                  &bounds_arg,     &state_arg,   &frontier_arg,
                  &next_arg,       &marks_arg,   &touched_arg,
                  &touched_count_arg, &controller_arg, &telemetry_arg};
  return hipLaunchCooperativeKernel(
      frontier_controller_kernel<CostMode, CollectTelemetry>,
      dim3(static_cast<unsigned>(blocks)), dim3(kBlockSize), args, 0,
      workspace.stream);
}

SsspStatus run_gpu_controller(const DeviceGraph& graph,
                              DeviceWorkspace& workspace,
                              int max_iters,
                              const BellmanFord11RunOptions& options,
                              int blocks,
                              DeviceCostMode cost_mode) {
  PATHFINDER_PROFILE_RANGE("bf11.gpu_controller");
  hipError_t launch_status = hipSuccess;
  if (workspace.telemetry_enabled) {
    switch (cost_mode) {
      case DeviceCostMode::kConstantOne:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kConstantOne, true>(
                graph, workspace, options, blocks);
        break;
      case DeviceCostMode::kStatic:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kStatic, true>(
                graph, workspace, options, blocks);
        break;
      case DeviceCostMode::kDynamic:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kDynamic, true>(
                graph, workspace, options, blocks);
        break;
    }
  } else {
    switch (cost_mode) {
      case DeviceCostMode::kConstantOne:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kConstantOne, false>(
                graph, workspace, options, blocks);
        break;
      case DeviceCostMode::kStatic:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kStatic, false>(
                graph, workspace, options, blocks);
        break;
      case DeviceCostMode::kDynamic:
        launch_status =
            launch_cooperative_controller<DeviceCostMode::kDynamic, false>(
                graph, workspace, options, blocks);
        break;
    }
  }
  check_hip(launch_status, "launch BF11 cooperative controller");
  if (workspace.telemetry_enabled) {
    g_bf11_gpu_controller_launches.fetch_add(1, std::memory_order_relaxed);
  }
  const ControllerDescriptor descriptor =
      copy_controller_to_host(workspace,
                              "synchronize BF11 cooperative controller");
  record_controller_result(graph, descriptor, max_iters,
                           &workspace);
  return {descriptor.iterations_used, descriptor.converged != 0,
          descriptor.early_stopped != 0, descriptor.hit_max_iters != 0,
          descriptor.all_targets_reached != 0};
}

struct SegmentedControllerRunResult {
  SsspStatus status{};
  bool restart_query_direct = false;
};

SegmentedControllerRunResult run_segmented_controller(
    const DeviceGraph& graph,
    DeviceWorkspace& workspace,
    int max_iters,
    const BellmanFord11RunOptions& options,
    int segment_rounds,
    BellmanFord11HipGraphMode graph_mode,
    DeviceCostMode cost_mode) {
  PATHFINDER_PROFILE_RANGE("bf11.segmented_controller");
  ControllerDescriptor descriptor{};
  if (max_iters == 0) {
    descriptor = copy_controller_to_host(
        workspace, "synchronize BF11 zero-round controller");
  } else {
    do {
      const bf11_graph_execution_policy::AttemptResult graph_result =
          enqueue_graph_segment(
              graph, workspace, options, segment_rounds, cost_mode,
              graph_mode);
      if (graph_result ==
          bf11_graph_execution_policy::AttemptResult::kRestartQueryDirect) {
        return {{}, true};
      }
      if (graph_result ==
          bf11_graph_execution_policy::AttemptResult::kGraphLaunched) {
        if (workspace.telemetry_enabled) {
          g_bf11_hip_graph_segments.fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        enqueue_direct_segment(graph, workspace, options, segment_rounds,
                               cost_mode);
        if (workspace.telemetry_enabled) {
          g_bf11_direct_segments.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (workspace.telemetry_enabled) {
        g_bf11_segments.fetch_add(1, std::memory_order_relaxed);
      }
      descriptor = copy_controller_to_host(
          workspace, "synchronize BF11 segmented controller status");
    } while (descriptor.done == 0);
  }
  if (max_iters == 0) {
    // No relaxation segment ran, but reset/seed and its status copy completed.
    descriptor.done = 1;
  }
  record_controller_result(graph, descriptor, max_iters,
                           &workspace);
  return {{descriptor.iterations_used, descriptor.converged != 0,
           descriptor.early_stopped != 0, descriptor.hit_max_iters != 0,
           descriptor.all_targets_reached != 0},
          false};
}

SsspStatus run_sssp(const DeviceGraph& graph,
                    DeviceWorkspace& workspace,
                    Offset source_count,
                    Offset target_count,
                    int max_iters,
                    const BellmanFord11RunOptions& options,
                    const BellmanFord11WorkspaceOptions& workspace_options) {
  if (max_iters < 0) max_iters = static_cast<int>(graph.rows) - 1;
  const DeviceCostMode cost_mode = workspace_cost_mode(graph, workspace);
  if (workspace.telemetry_enabled) {
    switch (cost_mode) {
      case DeviceCostMode::kConstantOne:
        g_bf11_constant_one_queries.fetch_add(1, std::memory_order_relaxed);
        break;
      case DeviceCostMode::kStatic:
        g_bf11_static_cost_queries.fetch_add(1, std::memory_order_relaxed);
        break;
      case DeviceCostMode::kDynamic:
        g_bf11_dynamic_cost_queries.fetch_add(1, std::memory_order_relaxed);
        break;
    }
  }
  prepare_query_controller(graph, workspace, source_count, target_count,
                           max_iters, options,
                           workspace_options.adaptive_reset_threshold,
                           cost_mode);
  // Never overlap full-residency cooperative grids. Explicit worker streams
  // always use ordinary segmented kernels with fixed launch geometry.
  const bool retain_default_cooperative_path =
      workspace.stream == nullptr && workspace_options.segment_rounds == 1 &&
      workspace_options.hip_graph_mode == BellmanFord11HipGraphMode::kAuto;
  const int blocks = retain_default_cooperative_path
                         ? cooperative_block_count(workspace)
                         : 0;
  if (blocks > 0 && max_iters != 0) {
    return run_gpu_controller(graph, workspace, max_iters, options, blocks,
                              cost_mode);
  }
  if (workspace.telemetry_enabled) {
    // This is a per-query controller choice. A non-atomic graph-launch error
    // can run the segmented loop twice after a full reset, but it remains one
    // host-controller query.
    g_bf11_controller_fallbacks.fetch_add(1, std::memory_order_relaxed);
  }
  SegmentedControllerRunResult segmented = run_segmented_controller(
      graph, workspace, max_iters, options, workspace_options.segment_rounds,
      workspace_options.hip_graph_mode, cost_mode);
  if (!segmented.restart_query_direct) return segmented.status;

  // hipGraphLaunch is not submission-atomic on every HIP runtime. The graph
  // backend quiesced a possibly submitted prefix before requesting this
  // restart. Reset/seed the complete query and use the now-sticky direct path;
  // continuing from a partially executed segment could duplicate rounds.
  prepare_query_controller(graph, workspace, source_count, target_count,
                           max_iters, options,
                           workspace_options.adaptive_reset_threshold,
                           cost_mode);
  segmented = run_segmented_controller(
      graph, workspace, max_iters, options, workspace_options.segment_rounds,
      workspace_options.hip_graph_mode, cost_mode);
  if (segmented.restart_query_direct) {
    throw std::runtime_error(
        "BF11 HIP Graph direct restart unexpectedly requested another restart");
  }
  return segmented.status;
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
  std::unordered_map<int, std::size_t> target_index;
  std::vector<unsigned long long> host_node_offsets;
  std::vector<unsigned long long> host_edge_offsets;
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
       BellmanFord11WorkspaceOptions options_in,
       SsspQueryCapacityHints capacity_hints)
      : graph(std::move(graph_in)), stream(stream_in), options(options_in) {
    if (options.query_telemetry && !options.telemetry) {
      throw std::invalid_argument(
          "BF11 query telemetry requires aggregate telemetry collection");
    }
    if (options.auto_margin_x < 0 || options.auto_margin_y < 0 ||
        options.target_check_interval <= 0 ||
        !(options.adaptive_reset_threshold > 0.0) ||
        options.adaptive_reset_threshold > 1.0 ||
        !std::isfinite(options.adaptive_reset_threshold)) {
      throw std::invalid_argument(
          "BF11 workspace bounds, target interval, or reset threshold is invalid");
    }
    bf11_execution_policy::validate_segment_rounds(
        static_cast<std::uint32_t>(options.segment_rounds));
    switch (options.hip_graph_mode) {
      case BellmanFord11HipGraphMode::kAuto:
      case BellmanFord11HipGraphMode::kOn:
      case BellmanFord11HipGraphMode::kOff:
        break;
      default:
        throw std::invalid_argument("invalid BF11 HIP Graph mode");
    }
    sssp_capacity::validate_reservation(capacity_hints);
    int current_device = -1;
    rips_sssp_bf11::check_hip(hipGetDevice(&current_device),
                              "get BF11 workspace HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument("BF11 graph belongs to another HIP device");
    }
    workspace = rips_sssp_bf11::make_workspace(
        graph->rows, stream, options.telemetry);
    try {
      const auto source_hint = static_cast<rips_sssp_bf11::Offset>(
          std::min<std::size_t>(capacity_hints.max_sources,
                                static_cast<std::size_t>(graph->rows)));
      const auto target_hint = static_cast<rips_sssp_bf11::Offset>(
          std::min<std::size_t>(capacity_hints.max_targets,
                                static_cast<std::size_t>(graph->rows)));
      rips_sssp_bf11::ensure_source_capacity(workspace, source_hint);
      rips_sssp_bf11::ensure_target_capacity(workspace, target_hint);
      rips_sssp_bf11::ensure_reconstruction_capacity(workspace, target_hint);
      const std::size_t vertex_count = static_cast<std::size_t>(graph->rows);
      const std::size_t target_count = static_cast<std::size_t>(target_hint);
      const std::size_t nodes_per_target = std::min(
          vertex_count, bf11_worker_policy::kInitialNodesPerTarget);
      const std::size_t edges_per_target = vertex_count == 0
          ? 0
          : std::min(vertex_count - 1,
                     bf11_worker_policy::kInitialEdgesPerTarget);
      const std::size_t initial_nodes =
          nodes_per_target != 0 &&
                  target_count >
                      bf11_worker_policy::kInitialArenaElementLimit /
                          nodes_per_target
              ? bf11_worker_policy::kInitialArenaElementLimit
              : std::min(bf11_worker_policy::kInitialArenaElementLimit,
                         target_count * nodes_per_target);
      const std::size_t initial_edges =
          edges_per_target != 0 &&
                  target_count >
                      bf11_worker_policy::kInitialArenaElementLimit /
                          edges_per_target
              ? bf11_worker_policy::kInitialArenaElementLimit
              : std::min(bf11_worker_policy::kInitialArenaElementLimit,
                         target_count * edges_per_target);
      rips_sssp_bf11::ensure_compact_capacity(
          workspace, initial_nodes, initial_edges);
      if (options.telemetry) {
        maximum_workspace_device_bytes =
            rips_sssp_bf11::workspace_device_bytes(workspace);
        const std::uint64_t constructed =
            g_bf11_constructed_workers.fetch_add(
                1, std::memory_order_relaxed) + 1;
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
    } catch (...) {
      // A throwing constructor does not run Impl::~Impl. Release both the
      // fixed workspace and any partially grown hinted arenas here.
      rips_sssp_bf11::free_workspace(&workspace);
      throw;
    }
  }

  ~Impl() {
    if (options.telemetry) {
      g_bf11_workspace_device_bytes_current_total.fetch_add(
          rips_sssp_bf11::workspace_device_bytes(workspace),
          std::memory_order_relaxed);
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

  void begin_query_telemetry() noexcept {
    if (options.query_telemetry) {
      workspace.query_telemetry = {};
    }
  }

  void emit_query_telemetry() const {
    if (!options.query_telemetry) return;
    const rips_sssp_bf11::QueryTelemetry& query =
        workspace.query_telemetry;
    const pathfinder_profile::QueryIdentity identity =
        pathfinder_profile::query_identity();
    const std::uint64_t sequence =
        g_bf11_query_telemetry_sequence.fetch_add(
            1, std::memory_order_relaxed);
    std::ostringstream out;
    out << "{\"type\":\"bf11_query_telemetry\",\"schema_version\":1"
        << ",\"sequence\":" << sequence << ",\"net_index\":";
    if (identity.valid) {
      out << identity.net_index;
    } else {
      out << "null";
    }
    out << ",\"worker_index\":";
    if (identity.valid) {
      out << identity.worker_index;
    } else {
      out << "null";
    }
    out << ",\"attempts\":" << query.attempts
        << ",\"iterations\":" << query.iterations
        << ",\"rounds\":" << query.rounds
        << ",\"no_op_rounds\":" << query.no_op_rounds
        << ",\"target_checks\":" << query.target_checks
        << ",\"frontier_vertices_processed\":"
        << query.frontier_vertices_processed
        << ",\"edges_examined\":" << query.edges_examined
        << ",\"successful_relaxations\":"
        << query.successful_relaxations
        << ",\"first_discoveries\":" << query.first_discoveries
        << ",\"mark_cas_attempts\":" << query.mark_cas_attempts
        << ",\"mark_cas_wins\":" << query.mark_cas_wins
        << ",\"queue_reservations\":" << query.queue_reservations
        << ",\"touched_vertices\":" << query.touched_vertices
        << ",\"reset_seed_gpu_nanoseconds\":"
        << query.reset_seed_gpu_nanoseconds
        << ",\"relaxation_gpu_nanoseconds\":"
        << query.relaxation_gpu_nanoseconds
        << ",\"target_check_gpu_nanoseconds\":"
        << query.target_check_gpu_nanoseconds << "}\n";
    std::lock_guard<std::mutex> lock(g_bf11_query_telemetry_output_mutex);
    std::cout << out.str();
  }

  rips_sssp_bf11::SsspStatus search_once(
                         const std::vector<int>& unique_sources,
                         const std::vector<int>& unique_targets,
                         int max_iters,
                         const BellmanFord11RunOptions& run_options,
                         const char* attempt_type,
                         bool allow_missing_bounded_sources = false) {
    using namespace rips_sssp_bf11;
    validate_run_options(run_options);
    validate_terminal_bounds(graph->host_sidecars, unique_sources,
                             unique_targets, run_options.bounds,
                             allow_missing_bounded_sources);
#if defined(PATHFINDER_ENABLE_ROCTX)
    const int effective_max_iters =
        max_iters < 0 ? static_cast<int>(graph->rows) - 1 : max_iters;
    const bool cooperative_candidate =
        workspace.stream == nullptr && options.segment_rounds == 1 &&
        options.hip_graph_mode == BellmanFord11HipGraphMode::kAuto;
    const int cooperative_blocks =
        cooperative_candidate ? cooperative_block_count(workspace) : 0;
    const pathfinder_profile::QueryIdentity identity =
        pathfinder_profile::query_identity();
    std::string identity_range = "bf11.query net=" +
        std::string(identity.valid ? std::to_string(identity.net_index)
                                   : "unknown") +
        " worker=" +
        std::string(identity.valid ? std::to_string(identity.worker_index)
                                   : "unknown") +
        (cooperative_blocks > 0 && effective_max_iters != 0
             ? " controller=cooperative"
             : " controller=segmented") +
        " attempt=" + std::string(attempt_type);
    PATHFINDER_PROFILE_RANGE(identity_range.c_str());
#else
    (void)attempt_type;
#endif
    const Offset source_count = static_cast<Offset>(unique_sources.size());
    const Offset target_count = static_cast<Offset>(unique_targets.size());
    ensure_source_capacity(workspace, source_count);
    ensure_target_capacity(workspace, target_count);
    ensure_reconstruction_capacity(workspace, target_count);
    note_workspace_size();
    if (workspace.needs_full_state_reset) {
      fully_reset_workspace_state(workspace);
    }

    workspace.needs_full_state_reset = true;
    DrainStreamOnException drain(stream);
    std::copy(unique_sources.begin(), unique_sources.end(),
              workspace.host_source_nodes);
    std::copy(unique_targets.begin(), unique_targets.end(),
              workspace.host_target_nodes);
    check_hip(hipMemcpyAsync(workspace.source_nodes,
                             workspace.host_source_nodes,
                             unique_sources.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 sources from pinned staging");
    check_hip(hipMemcpyAsync(workspace.target_nodes,
                             workspace.host_target_nodes,
                             unique_targets.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 targets from pinned staging");

    SsspStatus status =
        run_sssp(graph->device.view, workspace, source_count, target_count,
                 max_iters, run_options, options);
    if (status.iterations_used == 0) {
      status.all_targets_reached =
          std::all_of(unique_targets.begin(), unique_targets.end(),
                      [&](int target) {
                        return std::find(unique_sources.begin(),
                                         unique_sources.end(), target) !=
                               unique_sources.end();
                      });
      if (status.all_targets_reached) {
        // Source labels are exact before the first relaxation.
        status.early_stopped = true;
        status.hit_max_iters = false;
      }
    }
    aggregate_query_work_telemetry(workspace, status.iterations_used);
    // Traversal completed and all producer kernels were synchronized by the
    // controller copy. Sparse reset is safe for a bounded retry; extraction
    // separately reinstates the defensive exception guard.
    workspace.needs_full_state_reset = false;
    return status;
  }

  void enqueue_extraction(bool include_summary,
                          rips_sssp_bf11::Offset target_count,
                          rips_sssp_bf11::DeviceCostMode cost_mode) {
    PATHFINDER_PROFILE_RANGE("bf11.extraction");
    using namespace rips_sssp_bf11;
    if (include_summary) {
      begin_telemetry_event(workspace, workspace.target_summary_events);
      hipLaunchKernelGGL(
          summarize_target_paths_kernel, grid_for_items(target_count),
          dim3(kBlockSize), 0, stream, workspace.best_state, graph->rows,
          graph->nnz, graph->device.view.to, graph->device.view.rowptr,
          workspace.target_nodes, target_count, workspace.target_summaries);
      check_hip(hipGetLastError(), "summarize BF11 target paths");
      end_telemetry_event(workspace, workspace.target_summary_events);
    }

    begin_telemetry_event(workspace, workspace.target_prefix_events);
    hipLaunchKernelGGL(
        prefix_target_paths_kernel, dim3(1), dim3(1), 0, stream,
        workspace.target_summaries, target_count,
        static_cast<unsigned long long>(
            workspace.compact_node_transfer_capacity),
        static_cast<unsigned long long>(
            workspace.compact_edge_transfer_capacity),
        workspace.reconstruction_node_offsets,
        workspace.reconstruction_edge_offsets, workspace.extraction_header);
    check_hip(hipGetLastError(), "prefix BF11 target paths");
    end_telemetry_event(workspace, workspace.target_prefix_events);

    begin_telemetry_event(workspace, workspace.reconstruction_events);
    switch (cost_mode) {
      case DeviceCostMode::kConstantOne:
        hipLaunchKernelGGL(
            (materialize_target_paths_kernel<
                DeviceCostMode::kConstantOne>),
            grid_for_items(target_count), dim3(kBlockSize), 0, stream,
            workspace.best_state, graph->rows, graph->nnz,
            graph->device.view.to, graph->device.view.rowptr,
            workspace.target_nodes, workspace.target_summaries,
            workspace.reconstruction_node_offsets,
            workspace.reconstruction_edge_offsets,
            workspace.extraction_header, target_count,
            workspace.compact_nodes, workspace.compact_edges,
            graph->device.view, workspace.dynamic_vertex_cost,
            workspace.compact_edge_costs);
        break;
      case DeviceCostMode::kStatic:
        hipLaunchKernelGGL(
            (materialize_target_paths_kernel<DeviceCostMode::kStatic>),
            grid_for_items(target_count), dim3(kBlockSize), 0, stream,
            workspace.best_state, graph->rows, graph->nnz,
            graph->device.view.to, graph->device.view.rowptr,
            workspace.target_nodes, workspace.target_summaries,
            workspace.reconstruction_node_offsets,
            workspace.reconstruction_edge_offsets,
            workspace.extraction_header, target_count,
            workspace.compact_nodes, workspace.compact_edges,
            graph->device.view, workspace.dynamic_vertex_cost,
            workspace.compact_edge_costs);
        break;
      case DeviceCostMode::kDynamic:
        hipLaunchKernelGGL(
            (materialize_target_paths_kernel<DeviceCostMode::kDynamic>),
            grid_for_items(target_count), dim3(kBlockSize), 0, stream,
            workspace.best_state, graph->rows, graph->nnz,
            graph->device.view.to, graph->device.view.rowptr,
            workspace.target_nodes, workspace.target_summaries,
            workspace.reconstruction_node_offsets,
            workspace.reconstruction_edge_offsets,
            workspace.extraction_header, target_count,
            workspace.compact_nodes, workspace.compact_edges,
            graph->device.view, workspace.dynamic_vertex_cost,
            workspace.compact_edge_costs);
        break;
    }
    check_hip(hipGetLastError(), "materialize BF11 target paths");
    end_telemetry_event(workspace, workspace.reconstruction_events);

    begin_telemetry_event(workspace, workspace.output_transfer_events);
    check_hip(hipMemcpyAsync(workspace.host_extraction_header,
                             workspace.extraction_header,
                             sizeof(ExtractionHeader), hipMemcpyDeviceToHost,
                             stream),
              "copy BF11 extraction header");
    if (include_summary) {
      check_hip(hipMemcpyAsync(workspace.host_target_summaries,
                               workspace.target_summaries,
                               static_cast<std::size_t>(target_count) *
                                   sizeof(TargetSummary),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 target summaries");
    }
    if (workspace.compact_node_transfer_capacity != 0) {
      check_hip(hipMemcpyAsync(
                    workspace.host_compact_nodes, workspace.compact_nodes,
                    workspace.compact_node_transfer_capacity * sizeof(Index),
                    hipMemcpyDeviceToHost, stream),
                "copy BF11 compact node transfer window");
    }
    if (workspace.compact_edge_transfer_capacity != 0) {
      check_hip(hipMemcpyAsync(
                    workspace.host_compact_edges, workspace.compact_edges,
                    workspace.compact_edge_transfer_capacity *
                        sizeof(DeviceOffset),
                    hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edge transfer window");
      check_hip(hipMemcpyAsync(
                    workspace.host_compact_edge_costs,
                    workspace.compact_edge_costs,
                    workspace.compact_edge_transfer_capacity * sizeof(float),
                    hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edge-cost transfer window");
    }
    end_telemetry_event(workspace, workspace.output_transfer_events);
  }

  BellmanFordCsrResult extract_result(
      const std::vector<int>& unique_targets,
      const std::vector<int>& requested_targets,
      rips_sssp_bf11::SsspStatus status) {
    PATHFINDER_PROFILE_RANGE("bf11.extract_result");
#if defined(PATHFINDER_ENABLE_ROCTX)
    const pathfinder_profile::QueryIdentity identity =
        pathfinder_profile::query_identity();
    std::string extraction_range = "bf11.query net=" +
        std::string(identity.valid ? std::to_string(identity.net_index)
                                   : "unknown") +
        " worker=" +
        std::string(identity.valid ? std::to_string(identity.worker_index)
                                   : "unknown") +
        " controller=extraction attempt=extraction";
    PATHFINDER_PROFILE_RANGE(extraction_range.c_str());
#endif
    using namespace rips_sssp_bf11;
    const Offset target_count = static_cast<Offset>(unique_targets.size());
    ensure_reconstruction_capacity(workspace, target_count);
    const DeviceCostMode cost_mode =
        workspace_cost_mode(graph->device.view, workspace);
    workspace.needs_full_state_reset = true;
    DrainStreamOnException drain(stream);

    enqueue_extraction(true, target_count, cost_mode);
    synchronize_query_stream(workspace,
                             "synchronize BF11 device-side extraction");
    accumulate_telemetry_event(workspace.target_summary_events,
                               g_bf11_target_summary_gpu_nanoseconds);
    accumulate_telemetry_event(workspace.target_prefix_events,
                               g_bf11_target_prefix_gpu_nanoseconds);
    accumulate_telemetry_event(workspace.reconstruction_events,
                               g_bf11_reconstruction_gpu_nanoseconds);
    accumulate_telemetry_event(workspace.output_transfer_events,
                               g_bf11_output_transfer_gpu_nanoseconds);

    ExtractionHeader header = *workspace.host_extraction_header;
    if (header.status == 2) {
      if (header.required_nodes >
              static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max()) ||
          header.required_edges >
              static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max()) ||
          header.required_nodes >
              static_cast<unsigned long long>(
                  std::numeric_limits<int>::max()) ||
          header.required_edges >
              static_cast<unsigned long long>(
                  std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "BF11 compact paths exceed host/result capacity");
      }
      ensure_compact_capacity(
          workspace, static_cast<std::size_t>(header.required_nodes),
          static_cast<std::size_t>(header.required_edges));
      // A physical grow is not always necessary: status 2 can also mean that
      // the retained arena was large enough but its recent D2H window was not.
      // Replay only extraction with exactly the required transfer extents.
      workspace.compact_node_transfer_capacity =
          static_cast<std::size_t>(header.required_nodes);
      workspace.compact_edge_transfer_capacity =
          static_cast<std::size_t>(header.required_edges);
      note_workspace_size();
      enqueue_extraction(false, target_count, cost_mode);
      synchronize_query_stream(
          workspace, "synchronize grown BF11 extraction arena");
      accumulate_telemetry_event(workspace.target_prefix_events,
                                 g_bf11_target_prefix_gpu_nanoseconds);
      accumulate_telemetry_event(workspace.reconstruction_events,
                                 g_bf11_reconstruction_gpu_nanoseconds);
      accumulate_telemetry_event(workspace.output_transfer_events,
                                 g_bf11_output_transfer_gpu_nanoseconds);
      header = *workspace.host_extraction_header;
    }
    if (header.status == 1) {
      throw std::runtime_error(
          "BF11 produced an invalid or overflowing predecessor chain");
    }
    if (header.status != 0 ||
        header.total_nodes > workspace.compact_node_capacity ||
        header.total_edges > workspace.compact_edge_capacity ||
        header.total_nodes > workspace.compact_node_transfer_capacity ||
        header.total_edges > workspace.compact_edge_transfer_capacity) {
      throw std::runtime_error("BF11 extraction arena returned bad status");
    }

    // Keep a modest target-count floor for the next query, but shrink away a
    // prior long-path transfer window. Device allocation capacities remain at
    // their independent high-water marks.
    const std::size_t target_size = static_cast<std::size_t>(target_count);
    const std::size_t node_floor =
        target_size > bf11_worker_policy::kInitialArenaElementLimit /
                          bf11_worker_policy::kInitialNodesPerTarget
            ? bf11_worker_policy::kInitialArenaElementLimit
            : std::min(bf11_worker_policy::kInitialArenaElementLimit,
                       target_size *
                           bf11_worker_policy::kInitialNodesPerTarget);
    const std::size_t edge_floor =
        target_size > bf11_worker_policy::kInitialArenaElementLimit /
                          bf11_worker_policy::kInitialEdgesPerTarget
            ? bf11_worker_policy::kInitialArenaElementLimit
            : std::min(bf11_worker_policy::kInitialArenaElementLimit,
                       target_size *
                           bf11_worker_policy::kInitialEdgesPerTarget);
    workspace.compact_node_transfer_capacity = std::min(
        workspace.compact_node_capacity,
        std::max(node_floor, static_cast<std::size_t>(header.total_nodes)));
    workspace.compact_edge_transfer_capacity = std::min(
        workspace.compact_edge_capacity,
        std::max(edge_floor, static_cast<std::size_t>(header.total_edges)));

    host_node_offsets.assign(unique_targets.size(), 0);
    host_edge_offsets.assign(unique_targets.size(), 0);
    unsigned long long total_nodes = 0;
    unsigned long long total_edges = 0;
    for (std::size_t item = 0; item < unique_targets.size(); ++item) {
      host_node_offsets[item] = total_nodes;
      host_edge_offsets[item] = total_edges;
      const TargetSummary& summary = workspace.host_target_summaries[item];
      if (summary.status == kTargetUnreachable) continue;
      if (summary.status != kTargetPathValid ||
          summary.node_count == 0 ||
          summary.node_count != summary.edge_count + 1) {
        throw std::runtime_error("BF11 target summary is malformed");
      }
      total_nodes += summary.node_count;
      total_edges += summary.edge_count;
    }
    if (total_nodes != header.total_nodes ||
        total_edges != header.total_edges) {
      throw std::runtime_error(
          "BF11 host and device target prefixes disagree");
    }

    target_index.clear();
    target_index.reserve(unique_targets.size() * 2 + 1);
    for (std::size_t item = 0; item < unique_targets.size(); ++item) {
      target_index.emplace(unique_targets[item], item);
    }

    BellmanFordCsrResult result;
    result.target = -1;
    result.iterations_used = status.iterations_used;
    result.converged = status.converged;
    result.stopped_on_target = status.early_stopped;
    result.target_reached = true;
    result.target_distances.reserve(requested_targets.size());
    result.target_sources.reserve(requested_targets.size());
    result.target_path_offsets.reserve(requested_targets.size() + 1);
    result.target_edge_offsets.reserve(requested_targets.size() + 1);
    result.target_path_offsets.push_back(0);
    result.target_edge_offsets.push_back(0);

    for (const int target : requested_targets) {
      const std::size_t unique = target_index.at(target);
      const TargetSummary& summary = workspace.host_target_summaries[unique];
      if (summary.status == kTargetUnreachable) {
        result.target_distances.push_back(
            std::numeric_limits<float>::infinity());
        result.target_sources.push_back(-1);
        result.target_reached = false;
      } else {
        const std::size_t node_begin =
            static_cast<std::size_t>(host_node_offsets[unique]);
        const std::size_t edge_begin =
            static_cast<std::size_t>(host_edge_offsets[unique]);
        const std::size_t node_end =
            node_begin + static_cast<std::size_t>(summary.node_count);
        const std::size_t edge_end =
            edge_begin + static_cast<std::size_t>(summary.edge_count);
        if (node_end > header.total_nodes || edge_end > header.total_edges ||
            workspace.host_compact_nodes[node_begin] != summary.root ||
            workspace.host_compact_nodes[node_end - 1] != target) {
          throw std::runtime_error("BF11 compact target path is malformed");
        }
        result.target_distances.push_back(host_state_distance(summary.state));
        result.target_sources.push_back(summary.root);
        result.target_path_nodes.insert(
            result.target_path_nodes.end(),
            workspace.host_compact_nodes + node_begin,
            workspace.host_compact_nodes + node_end);
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          result.target_path_edges.push_back(
              static_cast<minplus_sparse::Offset>(
                  workspace.host_compact_edges[edge]));
          result.target_path_edge_costs.push_back(
              workspace.host_compact_edge_costs[edge]);
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
    return result;
  }
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
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::common_sidecar_view(sidecars, adjacency), stream),
          stream, options, capacity_hints)) {}

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
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::compatibility_sidecar_view(sidecars, adjacency),
              stream),
          stream, options, capacity_hints)) {}

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
    : impl_(std::make_unique<Impl>(Impl::require_graph(adjacency), stream,
                                   options, capacity_hints)) {}

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
  const bool all_identity =
      std::all_of(vertex_costs.begin(), vertex_costs.end(),
                  [](float cost) { return cost == 1.0f; });
  if (all_identity) {
    // A complete replacement proves identity. Retained dynamic storage may be
    // stale, but identity traversal/reconstruction do not read it; a later
    // sparse departure reinitializes the array before applying its delta.
    impl_->workspace.dynamic_cost_identity = true;
    impl_->workspace.dynamic_storage_identity = false;
    impl_->workspace.dynamic_cost_epoch_valid = true;
    return;
  }
  rips_sssp_bf11::ensure_dynamic_cost_storage(impl_->workspace);
  impl_->note_workspace_size();
  const bool previous_identity = impl_->workspace.dynamic_cost_identity;
  impl_->workspace.dynamic_cost_epoch_valid = false;
  try {
    rips_sssp_bf11::DrainStreamOnException drain(stream);
    rips_sssp_bf11::check_hip(
        hipMemcpyAsync(impl_->workspace.dynamic_vertex_cost,
                       vertex_costs.data(),
                       vertex_costs.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream),
        "copy BF11 dynamic vertex costs");
    rips_sssp_bf11::check_hip(hipStreamSynchronize(stream),
                              "synchronize BF11 dynamic vertex costs");
    impl_->workspace.dynamic_cost_identity = false;
    impl_->workspace.dynamic_storage_identity = false;
    impl_->workspace.dynamic_cost_epoch_valid = true;
  } catch (...) {
    // If the committed epoch was identity it remains usable because this
    // partially written array is dormant. A committed dynamic epoch is now
    // poisoned and requires a complete update_vertex_costs replacement.
    impl_->workspace.dynamic_cost_epoch_valid = previous_identity;
    impl_->workspace.dynamic_storage_identity = false;
    throw;
  }
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
  const bool all_identity =
      std::all_of(vertex_costs.begin(), vertex_costs.end(),
                  [](float cost) { return cost == 1.0f; });
  if (impl_->workspace.dynamic_cost_identity && all_identity) return;
  if (!impl_->workspace.dynamic_cost_identity &&
      !impl_->workspace.dynamic_cost_epoch_valid) {
    throw std::runtime_error(
        "BF11 sparse dynamic-cost update cannot recover a failed dynamic epoch; "
        "use a complete replacement");
  }
  const rips_sssp_bf11::Offset count =
      static_cast<rips_sssp_bf11::Offset>(nodes.size());
  rips_sssp_bf11::ensure_dynamic_cost_storage(impl_->workspace);
  rips_sssp_bf11::ensure_update_capacity(impl_->workspace, count);
  impl_->note_workspace_size();
  const bool previous_identity = impl_->workspace.dynamic_cost_identity;
  impl_->workspace.dynamic_cost_epoch_valid = false;
  try {
    rips_sssp_bf11::DrainStreamOnException drain(stream);
    if (previous_identity &&
        !impl_->workspace.dynamic_storage_identity) {
      // Do not claim that the retained array is identity until the whole
      // sparse update succeeds. On failure a later identity departure refills
      // all V entries before reuse.
      impl_->workspace.dynamic_storage_identity = false;
      hipLaunchKernelGGL(rips_sssp_bf11::fill_cost_kernel,
                         rips_sssp_bf11::grid_for_items(impl_->graph->rows),
                         dim3(rips_sssp_bf11::kBlockSize), 0, stream,
                         impl_->graph->rows,
                         impl_->workspace.dynamic_vertex_cost, 1.0f);
      rips_sssp_bf11::check_hip(
          hipGetLastError(), "initialize lazy BF11 dynamic vertex costs");
    }
    rips_sssp_bf11::check_hip(
        hipMemcpyAsync(impl_->workspace.update_nodes, nodes.data(),
                       nodes.size() * sizeof(int), hipMemcpyHostToDevice,
                       stream),
        "copy BF11 sparse cost nodes");
    rips_sssp_bf11::check_hip(
        hipMemcpyAsync(impl_->workspace.update_costs, vertex_costs.data(),
                       vertex_costs.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream),
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
    impl_->workspace.dynamic_cost_identity = false;
    impl_->workspace.dynamic_storage_identity = false;
    impl_->workspace.dynamic_cost_epoch_valid = true;
  } catch (...) {
    impl_->workspace.dynamic_cost_epoch_valid = previous_identity;
    impl_->workspace.dynamic_storage_identity = false;
    throw;
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
  impl_->begin_query_telemetry();
  rips_sssp_bf11::ScopedQueryTelemetry query_telemetry(
      impl_->workspace.telemetry_enabled);

  BellmanFord11RunOptions run_options;
  run_options.target_check_interval = impl_->options.target_check_interval;
  if (!impl_->options.auto_bounds) {
    const auto status = impl_->search_once(
        unique_sources, unique_targets, max_iters, run_options,
        "unbounded");
    BellmanFordCsrResult result =
        impl_->extract_result(unique_targets, targets, status);
    impl_->emit_query_telemetry();
    query_telemetry.mark_completed();
    return result;
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
    const auto status = impl_->search_once(
        unique_sources, unique_targets, max_iters, run_options,
        "unbounded");
    BellmanFordCsrResult result =
        impl_->extract_result(unique_targets, targets, status);
    impl_->emit_query_telemetry();
    query_telemetry.mark_completed();
    return result;
  }

  auto bounded_status = impl_->search_once(
      unique_sources, unique_targets, max_iters, run_options, "bounded",
      true);
  if (bounded_status.all_targets_reached ||
      !impl_->options.unbounded_fallback) {
    BellmanFordCsrResult result =
        impl_->extract_result(unique_targets, targets, bounded_status);
    impl_->emit_query_telemetry();
    query_telemetry.mark_completed();
    return result;
  }
  if (impl_->workspace.telemetry_enabled) {
    g_bf11_auto_unbounded_retries.fetch_add(1, std::memory_order_relaxed);
    g_bf11_bounded_fallbacks.fetch_add(1, std::memory_order_relaxed);
    g_bf11_avoided_failed_attempt_extractions.fetch_add(
        1, std::memory_order_relaxed);
  }
  run_options.bounds = {};
  auto unbounded_status = impl_->search_once(
      unique_sources, unique_targets, max_iters, run_options,
      "unbounded_retry");
  unbounded_status.iterations_used = rips_sssp_bf11::saturated_add(
      bounded_status.iterations_used, unbounded_status.iterations_used);
  BellmanFordCsrResult result =
      impl_->extract_result(unique_targets, targets, unbounded_status);
  impl_->emit_query_telemetry();
  query_telemetry.mark_completed();
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
  const std::vector<int> unique_sources = rips_sssp_bf11::deduplicate_nodes(
      sources, impl_->graph->rows, "source");
  const std::vector<int> unique_targets = rips_sssp_bf11::deduplicate_nodes(
      targets, impl_->graph->rows, "target");
  impl_->begin_query_telemetry();
  rips_sssp_bf11::ScopedQueryTelemetry query_telemetry(
      impl_->workspace.telemetry_enabled);
  const auto status = impl_->search_once(
      unique_sources, unique_targets, max_iters, run_options,
      run_options.bounds.enabled ? "bounded" : "unbounded");
  BellmanFordCsrResult result =
      impl_->extract_result(unique_targets, targets, status);
  impl_->emit_query_telemetry();
  query_telemetry.mark_completed();
  return result;
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
