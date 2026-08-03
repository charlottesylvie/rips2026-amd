#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

// Inputs and the decision are deliberately independent of HIP headers.  This
// keeps the policy usable by command-line validation and ordinary CPU tests.
struct BellmanFord12WorkerPolicyInputs {
  // 0 selects the conservative automatic policy.  A positive value requests
  // at most that many queries; it is not permission to exceed a safety limit.
  std::size_t requested_batch_size = 0;
  std::size_t vertex_count = 0;
  std::size_t edge_count = 0;
  std::size_t queries_waiting = 0;

  // Conservative requested counts for the flattened composite endpoint-state
  // arrays retained by the candidate batch.  The estimator rounds endpoint
  // and compact-result buffers to BF12's geometric high-water capacities.
  // Counts are intentionally not prorated when memory
  // pressure reduces the batch, because the policy does not know which query
  // descriptors the scheduler will choose.
  std::size_t source_count = 0;
  std::size_t target_count = 0;

  // hipMemGetInfo's free-byte result at the policy decision point.  The
  // configured reserve is subtracted before any BF12 allocation is admitted.
  std::size_t available_device_bytes = 0;
  // Requested minima; the estimate includes geometric retained-capacity
  // rounding, so callers should not pre-round these values.
  std::size_t result_node_capacity = 0;
  std::size_t result_edge_capacity = 0;
  std::string_view gpu_architecture{};
  int compute_unit_count = 0;
  bool cooperative_launch_supported = false;
  std::size_t memory_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
  bool telemetry_enabled = false;

  // Set this only when available_device_bytes was sampled after the immutable
  // CSR graph and its sidecars were uploaded.  Dynamic vertex costs are a BF12
  // workspace allocation and remain charged exactly once either way.
  bool graph_already_resident = false;
};

struct BellmanFord12WorkerDecision {
  std::size_t selected_batch_size = 0;
  // Device bytes retained outside the bounded result arenas.  If no batch can
  // fit, these two estimates describe the rejected batch-size-1 baseline so a
  // diagnostic can report the actual minimum requirement.
  std::size_t estimated_fixed_bytes = 0;
  std::size_t estimated_result_bytes = 0;
  bool cooperative_allowed = false;
  std::string limiting_reason;

  // Additional audit fields used by diagnostics and policy unit tests.
  std::size_t memory_budget_bytes = 0;
  std::size_t memory_reserve_bytes = 0;
  std::size_t automatic_preference = 0;
  // Exact retained graph-sized query capacity.  Smaller endpoint/result
  // buffers may independently retain geometric high-water capacities.
  std::size_t retained_batch_capacity = 0;
  std::size_t maximum_index_batch_size = 0;
  // Largest memory-fitting batch among all waiting/index-safe queries.  This
  // remains independent of the explicit request or automatic preference.
  std::size_t maximum_memory_batch_size = 0;
  bool used_gfx1151_default = false;
};

namespace bf12_worker_policy {

using Inputs = ::BellmanFord12WorkerPolicyInputs;
using Decision = ::BellmanFord12WorkerDecision;

inline constexpr std::size_t kBestLabelAndPredecessorBytesPerState = 8;
inline constexpr std::size_t kTwoFrontiersBytesPerState = 8;
inline constexpr std::size_t kNextFrontierMarkBytesPerState = 4;
inline constexpr std::size_t kSourceMaskBytesPerState = 1;
inline constexpr std::size_t kTouchedNodeBytesPerState = 4;
inline constexpr std::size_t kQueryStateBytesPerVertex =
    kBestLabelAndPredecessorBytesPerState +
    kTwoFrontiersBytesPerState +
    kNextFrontierMarkBytesPerState +
    kSourceMaskBytesPerState +
    kTouchedNodeBytesPerState;

inline constexpr std::size_t kCsrOffsetBytes = sizeof(std::uint32_t);
inline constexpr std::size_t kVertexIndexBytes = sizeof(std::uint32_t);
inline constexpr std::size_t kEdgeWeightBytes = sizeof(float);
inline constexpr std::size_t kCoordinateBytes = sizeof(std::int32_t);
inline constexpr std::size_t kBaseVertexCostBytes = sizeof(float);
inline constexpr std::size_t kDynamicVertexCostBytes = sizeof(float);
inline constexpr std::size_t kEndpointStateBytes = sizeof(std::uint32_t);
inline constexpr std::size_t kResultNodeBytes = sizeof(std::int32_t);
inline constexpr std::size_t kResultEdgeBytes =
    sizeof(std::uint32_t) + sizeof(float);
inline constexpr std::size_t kTargetSummaryBytes = 32;
inline constexpr std::size_t kQueryResultHeaderBytes = 64;
inline constexpr std::size_t kAllocationAlignment = 256;
// Descriptor and mutable control records are separate allocations.  Sixty-four
// bytes each is a conservative upper bound for the initial device layouts.
inline constexpr std::size_t kPerQueryDescriptorBytes = 64;
inline constexpr std::size_t kPerQueryControlBytes = 64;
// Touched-count, batch-control, and reconstruction-header records are also
// separate tiny allocations, each charged one allocator-alignment block.
inline constexpr std::size_t kGlobalControlBytes =
    3 * kAllocationAlignment;
inline constexpr std::size_t kTelemetryCounterBytes = 16;
inline constexpr std::size_t kMaximumTelemetryRounds = 1ULL << 20;

inline constexpr std::size_t kAutomaticGfx1151BatchSize = 4;
inline constexpr std::size_t kAutomaticDefaultBatchSize = 3;

static_assert(sizeof(float) == 4,
              "BF12 policy requires 32-bit graph and result weights");
static_assert(kQueryStateBytesPerVertex == 25,
              "BF12 fixed query-state accounting changed");
static_assert(kResultEdgeBytes == 8,
              "BF12 compact result-edge accounting changed");

constexpr std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("BF12 worker-policy byte addition overflow");
  }
  return left + right;
}

constexpr std::size_t checked_multiply(std::size_t left,
                                       std::size_t right) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(
        "BF12 worker-policy byte multiplication overflow");
  }
  return left * right;
}

constexpr std::size_t aligned_allocation_bytes(std::size_t bytes) {
  if (bytes == 0) return 0;
  const std::size_t remainder = bytes % kAllocationAlignment;
  return remainder == 0
             ? bytes
             : checked_add(bytes, kAllocationAlignment - remainder);
}

constexpr std::size_t add_allocation(std::size_t total,
                                     std::size_t bytes) {
  return checked_add(total, aligned_allocation_bytes(bytes));
}

// Endpoint, transfer, and compact-result buffers retain high-water capacities
// grown to powers of two.  Graph-sized query state grows exactly to B because
// rounding B itself could add hundreds of megabytes per vertex segment.
// Close to the size_t ceiling no larger power of two is representable, so the
// allocator's checked terminal capacity is the exact requirement.
constexpr std::size_t rounded_retained_capacity(std::size_t required) {
  if (required == 0) return 0;
  std::size_t capacity = 1;
  while (capacity < required) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      return required;
    }
    capacity *= 2;
  }
  return capacity;
}

// The immutable graph is shared by every query: compact CSR offsets,
// destination IDs, edge weights, X/Y coordinates, and one base-cost column.
constexpr std::size_t shared_graph_device_bytes(std::size_t vertex_count,
                                                std::size_t edge_count) {
  std::size_t bytes = 0;
  bytes = add_allocation(
      bytes,
      checked_multiply(checked_add(vertex_count, 1), kCsrOffsetBytes));
  bytes = add_allocation(
      bytes, checked_multiply(edge_count, kVertexIndexBytes));
  bytes = add_allocation(
      bytes, checked_multiply(edge_count, kEdgeWeightBytes));
  bytes = add_allocation(
      bytes, checked_multiply(vertex_count, kCoordinateBytes));
  bytes = add_allocation(
      bytes, checked_multiply(vertex_count, kCoordinateBytes));
  bytes = add_allocation(
      bytes, checked_multiply(vertex_count, kBaseVertexCostBytes));
  return bytes;
}

// Dynamic costs describe one frozen routing epoch.  This allocation is shared
// by the batch and must never be multiplied by the number of queries.
constexpr std::size_t shared_dynamic_cost_device_bytes(
    std::size_t vertex_count) {
  return aligned_allocation_bytes(
      checked_multiply(vertex_count, kDynamicVertexCostBytes));
}

constexpr std::size_t raw_query_vertex_state_bytes(
    std::size_t batch_size,
    std::size_t vertex_count) {
  return checked_multiply(
      checked_multiply(batch_size, vertex_count),
      kQueryStateBytesPerVertex);
}

// Account for the six separately allocated segmented arrays (the two frontier
// buffers are distinct allocations).  Charging each allocation's alignment
// makes this no smaller than the nominal 25 B/state.
constexpr std::size_t query_vertex_state_device_bytes(
    std::size_t batch_size,
    std::size_t vertex_count) {
  const std::size_t state_count =
      checked_multiply(batch_size, vertex_count);
  std::size_t bytes = 0;
  bytes = add_allocation(
      bytes,
      checked_multiply(state_count,
                       kBestLabelAndPredecessorBytesPerState));
  bytes = add_allocation(
      bytes,
      checked_multiply(state_count, kVertexIndexBytes));
  bytes = add_allocation(
      bytes,
      checked_multiply(state_count, kVertexIndexBytes));
  bytes = add_allocation(
      bytes,
      checked_multiply(state_count, kNextFrontierMarkBytesPerState));
  bytes = add_allocation(
      bytes, checked_multiply(state_count, kSourceMaskBytesPerState));
  bytes = add_allocation(
      bytes, checked_multiply(state_count, kTouchedNodeBytesPerState));
  return bytes;
}

constexpr std::size_t result_device_bytes_estimate(
    std::size_t batch_size,
    std::size_t target_count,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity) {
  const std::size_t node_capacity =
      rounded_retained_capacity(result_node_capacity);
  const std::size_t edge_capacity =
      rounded_retained_capacity(result_edge_capacity);
  const std::size_t target_capacity =
      rounded_retained_capacity(target_count);
  const std::size_t query_capacity = batch_size;
  std::size_t bytes = 0;
  bytes = add_allocation(
      bytes, checked_multiply(node_capacity, kResultNodeBytes));
  bytes = add_allocation(
      bytes,
      checked_multiply(edge_capacity, sizeof(std::uint32_t)));
  bytes = add_allocation(
      bytes, checked_multiply(edge_capacity, sizeof(float)));
  bytes = add_allocation(
      bytes, checked_multiply(target_capacity, kTargetSummaryBytes));
  const std::size_t target_offset_count = checked_add(target_capacity, 1);
  bytes = add_allocation(
      bytes,
      checked_multiply(target_offset_count, sizeof(std::uint64_t)));
  bytes = add_allocation(
      bytes,
      checked_multiply(target_offset_count, sizeof(std::uint64_t)));
  bytes = add_allocation(
      bytes, checked_multiply(query_capacity, kQueryResultHeaderBytes));
  // One contiguous D2H header contains query status, target summary and
  // offsets, plus batch/arena status. Query descriptors remain host-resident
  // and are not copied back. The operational arrays remain separate on device,
  // so conservatively charge the packed transfer block as an additional
  // allocation.
  std::size_t transfer_header_bytes = 0;
  transfer_header_bytes = checked_add(
      transfer_header_bytes,
      checked_multiply(query_capacity, kPerQueryControlBytes));
  transfer_header_bytes = checked_add(
      transfer_header_bytes,
      checked_multiply(target_capacity,
                       kTargetSummaryBytes + 2 * sizeof(std::uint64_t)));
  transfer_header_bytes = checked_add(transfer_header_bytes,
                                      kAllocationAlignment);
  bytes = checked_add(
      bytes,
      aligned_allocation_bytes(
          rounded_retained_capacity(transfer_header_bytes)));
  return bytes;
}

constexpr std::size_t fixed_device_bytes_estimate(
    const Inputs& inputs,
    std::size_t batch_size) {
  const std::size_t source_capacity =
      rounded_retained_capacity(inputs.source_count);
  const std::size_t target_capacity =
      rounded_retained_capacity(inputs.target_count);
  const std::size_t query_capacity = batch_size;
  std::size_t bytes = 0;
  if (!inputs.graph_already_resident) {
    bytes = checked_add(
        bytes,
        shared_graph_device_bytes(inputs.vertex_count, inputs.edge_count));
  }
  bytes = checked_add(
      bytes, shared_dynamic_cost_device_bytes(inputs.vertex_count));
  bytes = checked_add(
      bytes,
      query_vertex_state_device_bytes(batch_size, inputs.vertex_count));
  bytes = add_allocation(
      bytes, checked_multiply(source_capacity, kEndpointStateBytes));
  bytes = add_allocation(
      bytes, checked_multiply(target_capacity, kEndpointStateBytes));
  bytes = add_allocation(
      bytes,
      checked_multiply(query_capacity, kPerQueryDescriptorBytes));
  bytes = add_allocation(
      bytes, checked_multiply(query_capacity, kPerQueryControlBytes));
  bytes = add_allocation(bytes, kGlobalControlBytes);
  if (inputs.telemetry_enabled) {
    bytes = add_allocation(bytes, kTelemetryCounterBytes);
    bytes = add_allocation(
        bytes,
        checked_multiply(kMaximumTelemetryRounds, sizeof(std::uint32_t)));
    bytes = add_allocation(
        bytes,
        checked_multiply(kMaximumTelemetryRounds, sizeof(std::uint64_t)));
  }
  return bytes;
}

constexpr std::size_t total_device_bytes_estimate(
    const Inputs& inputs,
    std::size_t batch_size) {
  return checked_add(
      fixed_device_bytes_estimate(inputs, batch_size),
      result_device_bytes_estimate(
          batch_size, inputs.target_count, inputs.result_node_capacity,
          inputs.result_edge_capacity));
}

constexpr bool is_gfx1151(std::string_view architecture) {
  constexpr std::string_view name = "gfx1151";
  return architecture.size() >= name.size() &&
         architecture.substr(0, name.size()) == name &&
         (architecture.size() == name.size() ||
          architecture[name.size()] == ':');
}

constexpr std::size_t automatic_batch_preference(const Inputs& inputs) {
  return is_gfx1151(inputs.gpu_architecture)
             ? kAutomaticGfx1151BatchSize
             : kAutomaticDefaultBatchSize;
}

// BF12's initial flattened-frontier representation uses uint32_t composite
// indices and reserves 0xffffffff as an invalid/sentinel value.  Query storage
// grows exactly to the submitted count, so B * V must be strictly below the
// sentinel.
constexpr std::size_t maximum_uint32_composite_batch_size(
    std::size_t vertex_count) {
  if (vertex_count == 0) return 0;
  return static_cast<std::size_t>(
             std::numeric_limits<std::uint32_t>::max() - 1) /
         vertex_count;
}

inline Decision decide(const Inputs& inputs) {
  Decision result;
  result.memory_reserve_bytes = inputs.memory_reserve_bytes;
  result.automatic_preference = automatic_batch_preference(inputs);
  result.used_gfx1151_default = is_gfx1151(inputs.gpu_architecture);

  if (inputs.queries_waiting == 0) {
    result.limiting_reason = "no independent routing queries are waiting";
    return result;
  }
  if (inputs.vertex_count == 0) {
    result.limiting_reason = "the routing graph has zero vertices";
    return result;
  }

  result.maximum_index_batch_size =
      maximum_uint32_composite_batch_size(inputs.vertex_count);
  if (result.maximum_index_batch_size == 0) {
    result.limiting_reason =
        "one query cannot fit the 32-bit flattened-state index";
    return result;
  }

  // Compute the minimum useful allocation before inspecting the memory
  // report.  Arithmetic mistakes are rejected as overflow exceptions rather
  // than being wrapped into a falsely safe decision.
  const std::size_t baseline_fixed =
      fixed_device_bytes_estimate(inputs, 1);
  const std::size_t baseline_result = result_device_bytes_estimate(
      1, inputs.target_count, inputs.result_node_capacity,
      inputs.result_edge_capacity);
  result.estimated_fixed_bytes = baseline_fixed;
  result.estimated_result_bytes = baseline_result;

  if (inputs.available_device_bytes == 0) {
    result.limiting_reason =
        "available GPU memory was not reported; refusing an unsafe batch";
    return result;
  }
  if (inputs.memory_reserve_bytes >= inputs.available_device_bytes) {
    result.limiting_reason =
        "the GPU memory safety reserve leaves no allocation budget";
    return result;
  }
  result.memory_budget_bytes =
      inputs.available_device_bytes - inputs.memory_reserve_bytes;

  const bool automatic = inputs.requested_batch_size == 0;
  const std::size_t requested_or_preferred =
      automatic ? result.automatic_preference
                : inputs.requested_batch_size;
  const std::size_t hard_candidate_limit =
      std::min(inputs.queries_waiting,
               result.maximum_index_batch_size);
  std::size_t performance_candidate_limit =
      std::min(requested_or_preferred, inputs.queries_waiting);
  performance_candidate_limit =
      std::min(performance_candidate_limit,
               result.maximum_index_batch_size);
  if (automatic && inputs.compute_unit_count > 0) {
    performance_candidate_limit = std::min(
        performance_candidate_limit,
        static_cast<std::size_t>(inputs.compute_unit_count));
  }

  if (performance_candidate_limit == 0) {
    result.limiting_reason = "no valid BF12 batch candidate remains";
    return result;
  }

  // The estimate is monotonic in batch_size, so find the largest fitting
  // candidate without a potentially enormous linear scan for explicit input.
  std::size_t low = 1;
  std::size_t high = hard_candidate_limit;
  std::size_t fitting = 0;
  while (low <= high) {
    const std::size_t middle = low + (high - low) / 2;
    if (total_device_bytes_estimate(inputs, middle) <=
        result.memory_budget_bytes) {
      fitting = middle;
      if (middle == high) break;
      low = middle + 1;
    } else {
      if (middle == 0) break;
      high = middle - 1;
    }
  }
  result.maximum_memory_batch_size = fitting;

  if (fitting == 0) {
    result.limiting_reason =
        "batch size 1 exceeds available GPU memory after the safety reserve";
    return result;
  }

  result.selected_batch_size =
      std::min(performance_candidate_limit, fitting);
  result.retained_batch_capacity = result.selected_batch_size;
  result.estimated_fixed_bytes =
      fixed_device_bytes_estimate(inputs, result.selected_batch_size);
  result.estimated_result_bytes = result_device_bytes_estimate(
      result.selected_batch_size, inputs.target_count,
      inputs.result_node_capacity,
      inputs.result_edge_capacity);
  result.cooperative_allowed = inputs.cooperative_launch_supported;

  if (fitting < performance_candidate_limit) {
    result.limiting_reason =
        "batch size limited by available GPU memory after the safety reserve";
  } else if (performance_candidate_limit < requested_or_preferred) {
    const std::size_t cu_limit =
        inputs.compute_unit_count > 0
            ? static_cast<std::size_t>(inputs.compute_unit_count)
            : std::numeric_limits<std::size_t>::max();
    if (result.maximum_index_batch_size <= inputs.queries_waiting &&
        result.maximum_index_batch_size <= cu_limit &&
        result.maximum_index_batch_size < requested_or_preferred) {
      result.limiting_reason =
          "batch size limited by the 32-bit flattened-state index";
    } else if (automatic && cu_limit <= inputs.queries_waiting &&
               cu_limit < requested_or_preferred) {
      result.limiting_reason =
          "automatic batch size limited by the reported compute-unit count";
    } else {
      result.limiting_reason =
          "batch size limited by the number of waiting queries";
    }
  } else if (automatic) {
    result.limiting_reason = result.used_gfx1151_default
                                 ? "automatic conservative gfx1151 default (4)"
                                 : "automatic conservative default (3)";
  } else if (result.selected_batch_size == 1) {
    result.limiting_reason = "explicit correctness baseline batch size 1";
  } else {
    result.limiting_reason = "explicit requested batch size is safe";
  }

  if (!result.cooperative_allowed) {
    result.limiting_reason +=
        "; cooperative execution is unavailable, use host-batch mode";
  }
  return result;
}

}  // namespace bf12_worker_policy
