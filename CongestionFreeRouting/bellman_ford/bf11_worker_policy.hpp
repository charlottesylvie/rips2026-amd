#pragma once

#include "../sssp_query_capacity.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace bf11_worker_policy {

// DeviceWorkspace retains these graph-sized arrays for its lifetime:
// packed best state (8), two frontiers (4 each), frontier mark (4), source
// mask (1), touched-node list (4), and dynamic destination cost (4).
inline constexpr std::size_t kPersistentBytesPerVertex = 29;

// Device-side scalar control storage consists of touched_count (one int32),
// IterationStatus (five int32 fields), and ControllerResult (nine int32
// fields). Pinned host mirrors are deliberately not device-memory charges.
inline constexpr std::size_t kFixedDeviceStatusBytes =
    (1 + 5 + 9) * sizeof(std::int32_t);

// A retained source slot stores one int32 node. A retained target slot stores
// one int32 node, a 32-byte TargetSummary, and two uint64 reconstruction
// offsets. Compact materialized paths remain demand-grown and are not part of
// this persistent query-capacity estimate.
inline constexpr std::size_t kDeviceBytesPerSourceCapacity =
    sizeof(std::int32_t);
inline constexpr std::size_t kDeviceBytesPerTargetCapacity =
    sizeof(std::int32_t) + 3 * sizeof(std::uint64_t) +
    2 * sizeof(std::int32_t) + 2 * sizeof(std::uint64_t);
inline constexpr std::size_t kTelemetryDeviceBytes =
    6 * sizeof(std::uint64_t);
inline constexpr std::size_t kDeviceBytesPerCompactNode =
    sizeof(std::int32_t);
inline constexpr std::size_t kDeviceBytesPerCompactEdge =
    sizeof(std::uint32_t) + sizeof(float);

inline constexpr std::size_t kMaxAutomaticWorkers = 4;
inline constexpr std::size_t kGfx1151PreferredWorkers = 3;
inline constexpr std::size_t kUnmeasuredArchitecturePreferredWorkers = 1;
inline constexpr std::size_t kFreeMemoryReserveDivisor = 4;

static_assert(sizeof(float) == sizeof(std::uint32_t),
              "BF11 workspace accounting requires 32-bit float storage");
static_assert(kFixedDeviceStatusBytes == 60,
              "BF11 device status layout accounting changed");
static_assert(kDeviceBytesPerTargetCapacity == 52,
              "BF11 target-capacity accounting changed");
static_assert(kTelemetryDeviceBytes == 48,
              "BF11 telemetry-counter accounting changed");
static_assert(kDeviceBytesPerCompactNode == 4 &&
                  kDeviceBytesPerCompactEdge == 8,
              "BF11 compact-path accounting changed");

// Mirror bf11.cpp's geometric_capacity(0, required, vertex_count). Query
// endpoint lists are stable-deduplicated before allocation, so no retained
// source or target capacity can exceed V even if raw metadata hints do.
constexpr std::size_t retained_query_capacity(std::size_t vertex_count,
                                              std::size_t required) {
  if (vertex_count == 0 || required == 0) return 0;
  required = std::min(required, vertex_count);
  std::size_t result = 1;
  while (result < required) {
    if (result > vertex_count / 2) return vertex_count;
    result *= 2;
  }
  return result;
}

constexpr std::size_t query_capacity_device_bytes(
    std::size_t vertex_count,
    std::size_t maximum_source_count,
    std::size_t maximum_target_count) {
  const std::size_t source_capacity =
      retained_query_capacity(vertex_count, maximum_source_count);
  const std::size_t target_capacity =
      retained_query_capacity(vertex_count, maximum_target_count);
  std::size_t bytes = sssp_capacity::checked_multiply(
      source_capacity, kDeviceBytesPerSourceCapacity);
  return sssp_capacity::checked_add(
      bytes,
      sssp_capacity::checked_multiply(
          target_capacity, kDeviceBytesPerTargetCapacity));
}

constexpr std::size_t fixed_device_workspace_bytes(
    std::size_t vertex_count,
    bool telemetry_enabled = false) {
  std::size_t bytes = sssp_capacity::checked_multiply(
      vertex_count, kPersistentBytesPerVertex);
  bytes = sssp_capacity::checked_add(bytes, kFixedDeviceStatusBytes);
  return telemetry_enabled
             ? sssp_capacity::checked_add(bytes, kTelemetryDeviceBytes)
             : bytes;
}

// Exact device bytes at BF11's retained graph/state and query-capacity
// ceiling. Duplicate endpoints make the query portion conservative because
// BF11 stable-deduplicates them before allocation. Compact materialized paths
// are accounted separately below; PathFinder never allocates sparse-cost
// staging.
constexpr std::size_t persistent_device_workspace_bytes(
    std::size_t vertex_count,
    std::size_t maximum_source_count,
    std::size_t maximum_target_count,
    bool telemetry_enabled = false) {
  return sssp_capacity::checked_add(
      fixed_device_workspace_bytes(vertex_count, telemetry_enabled),
      query_capacity_device_bytes(vertex_count, maximum_source_count,
                                  maximum_target_count));
}

// Mirror ensure_compact_capacity's power-of-two retained growth. Successful
// BF11 results reject aggregate node/edge counts above INT_MAX before device
// allocation, so that is the largest capacity this estimate needs to cover.
constexpr std::size_t rounded_compact_capacity(std::size_t required) {
  if (required == 0) return 0;
  std::size_t result = 1;
  while (result < required) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) {
      return required;
    }
    result *= 2;
  }
  return result;
}

constexpr std::size_t capped_product(std::size_t left,
                                     std::size_t right,
                                     std::size_t limit) {
  if (left == 0 || right == 0) return 0;
  return left > limit / right ? limit : left * right;
}

constexpr std::size_t compact_path_device_bytes_ceiling(
    std::size_t vertex_count,
    std::size_t maximum_target_count) {
  const std::size_t target_count =
      std::min(vertex_count, maximum_target_count);
  constexpr std::size_t result_item_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const std::size_t required_nodes =
      capped_product(target_count, vertex_count, result_item_limit);
  const std::size_t required_edges =
      capped_product(target_count,
                     vertex_count == 0 ? 0 : vertex_count - 1,
                     result_item_limit);
  const std::size_t node_capacity =
      rounded_compact_capacity(required_nodes);
  const std::size_t edge_capacity =
      rounded_compact_capacity(required_edges);
  return sssp_capacity::checked_add(
      sssp_capacity::checked_multiply(node_capacity,
                                      kDeviceBytesPerCompactNode),
      sssp_capacity::checked_multiply(edge_capacity,
                                      kDeviceBytesPerCompactEdge));
}

// Conservative per-worker allocation peak for automatic selection. BF11
// allocates replacement query/path buffers before releasing the prior buffers,
// so double their retained ceiling. Graph-sized state and telemetry counters
// are fixed and never overlap a replacement. The separate 25% free-memory
// reserve remains runtime/allocator headroom rather than an unverified compact-
// path allowance.
constexpr std::size_t automatic_worker_device_bytes_estimate(
    std::size_t vertex_count,
    std::size_t maximum_source_count,
    std::size_t maximum_target_count,
    bool telemetry_enabled = false) {
  const std::size_t fixed =
      fixed_device_workspace_bytes(vertex_count, telemetry_enabled);
  std::size_t replaceable = query_capacity_device_bytes(
      vertex_count, maximum_source_count, maximum_target_count);
  replaceable = sssp_capacity::checked_add(
      replaceable,
      compact_path_device_bytes_ceiling(vertex_count, maximum_target_count));
  return sssp_capacity::checked_add(
      fixed, sssp_capacity::checked_multiply(replaceable, 2));
}

constexpr bool is_measured_gfx1151(std::string_view architecture) {
  constexpr std::string_view name = "gfx1151";
  return architecture.size() >= name.size() &&
         architecture.substr(0, name.size()) == name &&
         (architecture.size() == name.size() ||
          architecture[name.size()] == ':');
}

struct Inputs {
  std::size_t route_request_count = 0;
  std::size_t cpu_hardware_concurrency = 0;
  std::size_t free_device_bytes = 0;
  std::size_t workspace_device_bytes_estimate = 0;
  std::string_view device_architecture{};
  int compute_unit_count = 0;
};

struct Recommendation {
  std::size_t worker_count = 1;
  std::size_t performance_preference = 1;
  std::size_t resource_limit = 1;
  std::size_t route_limit = 1;
  std::size_t cpu_limit = 1;
  std::size_t memory_limit = 0;
  std::size_t compute_unit_limit = 1;
  std::size_t memory_budget_bytes = 0;
  bool uses_measured_gfx1151_policy = false;
};

// Recommend only the automatic count. An explicit --parallel-net-workers N
// override is resolved by PathFinder before/after this pure policy function.
// Unmeasured architectures remain at one worker until target evidence supports
// a larger plateau; gfx1151 uses the measured smallest-plateau preference of
// three when every resource ceiling permits it.
constexpr Recommendation recommend(const Inputs& inputs) {
  Recommendation result;
  result.route_limit = std::max<std::size_t>(1, inputs.route_request_count);
  result.cpu_limit =
      std::max<std::size_t>(1, inputs.cpu_hardware_concurrency);
  result.compute_unit_limit = inputs.compute_unit_count > 0
                                  ? static_cast<std::size_t>(
                                        inputs.compute_unit_count)
                                  : 1;
  result.memory_budget_bytes =
      inputs.free_device_bytes -
      inputs.free_device_bytes / kFreeMemoryReserveDivisor;
  if (inputs.workspace_device_bytes_estimate != 0) {
    result.memory_limit =
        result.memory_budget_bytes / inputs.workspace_device_bytes_estimate;
  }

  const std::size_t usable_memory_limit =
      std::max<std::size_t>(1, result.memory_limit);
  result.resource_limit =
      std::min({kMaxAutomaticWorkers,
                result.route_limit,
                result.cpu_limit,
                usable_memory_limit,
                result.compute_unit_limit});
  result.resource_limit = std::max<std::size_t>(1, result.resource_limit);

  result.uses_measured_gfx1151_policy =
      is_measured_gfx1151(inputs.device_architecture);
  result.performance_preference =
      result.uses_measured_gfx1151_policy
          ? kGfx1151PreferredWorkers
          : kUnmeasuredArchitecturePreferredWorkers;
  result.worker_count =
      std::max<std::size_t>(
          1, std::min(result.performance_preference, result.resource_limit));
  return result;
}

}  // namespace bf11_worker_policy
