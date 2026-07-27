#include "delta_stepping_hip_CSR.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ds_delta_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;
using CompactRowOffset = std::uint32_t;

static_assert(sizeof(CompactRowOffset) == 4,
              "Delta-Stepping compact row offsets must be 32-bit");

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kNoBucket = 0x3fffffff;
constexpr int kUnitCooperativeLevelsPerLaunch = 32;
constexpr int kMaxConcurrentCooperativeWorkers = 8;
constexpr int kTargetConcurrentCooperativeWorkers = 4;
constexpr Offset kMaxUnitSpecializationRows =
    static_cast<Offset>(1) << std::numeric_limits<float>::digits;
constexpr unsigned long long kNoParentKey =
    std::numeric_limits<unsigned long long>::max();

enum DeviceTelemetryCounter : int {
  kTelemetryFrontierEntries = 0,
  kTelemetryActiveVertices,
  kTelemetryStaleFrontierEntries,
  kTelemetryLightEdgeVisits,
  kTelemetryHeavyEdgeVisits,
  kTelemetryDistanceAtomicAttempts,
  kTelemetrySuccessfulRelaxations,
  kTelemetryDistanceCasRetries,
  kTelemetryCurrentQueueInsertions,
  kTelemetryPendingQueueInsertions,
  kTelemetryHeavyQueueInsertions,
  kTelemetryPendingEntryExaminations,
  kTelemetryStalePendingEntryExaminations,
  kTelemetryCurrentQueueHighWater,
  kTelemetryPendingQueueHighWater,
  kTelemetryHeavyQueueHighWater,
  kTelemetryCounterCount,
};

enum UnitStatusIndex : int {
  kUnitStatusQueueTail = 0,
  kUnitStatusFoundCount,
  kUnitStatusFrontierBegin,
  kUnitStatusFrontierEnd,
  kUnitStatusCompletedDepth,
  kUnitStatusActive,
  kUnitStatusBucket,
  kUnitStatusBucketRounds,
  kUnitStatusCount,
};

enum TargetOffsetScanStatus : int {
  kTargetOffsetScanNotPublished = 0,
  kTargetOffsetScanValid = 1,
  kTargetOffsetScanInvalidMetadata = -1,
  kTargetOffsetScanOverflow = -2,
};

struct TargetPathTotals {
  int total_nodes = 0;
  int total_edges = 0;
  int status = kTargetOffsetScanNotPublished;
  std::uint32_t extraction_epoch = 0;
};

static_assert(sizeof(TargetPathTotals) == 4 * sizeof(std::uint32_t),
              "Delta target totals must remain one packed control record");

inline void hip_check(hipError_t status, const char* expr, const char* file, int line) {
  if (status != hipSuccess) {
    std::ostringstream os;
    os << "HIP error at " << file << ':' << line << " for " << expr << ": "
       << hipGetErrorString(status);
    throw std::runtime_error(os.str());
  }
}

#define DS_DELTA_HIP_CHECK(expr) \
  ::ds_delta_detail::hip_check((expr), #expr, __FILE__, __LINE__)

inline int current_hip_device() {
  int device = 0;
  DS_DELTA_HIP_CHECK(hipGetDevice(&device));
  return device;
}

inline int current_hip_wavefront_size() {
  hipDeviceProp_t properties{};
  DS_DELTA_HIP_CHECK(
      hipGetDeviceProperties(&properties, current_hip_device()));
  if (properties.warpSize <= 0) {
    throw std::runtime_error(
        "HIP device reported a nonpositive wavefront size");
  }
  return properties.warpSize;
}

inline bool device_memory_request_fits(std::size_t requested_bytes,
                                       std::size_t* free_bytes_out = nullptr,
                                       std::size_t* reserve_out = nullptr) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  DS_DELTA_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  const std::size_t reserve =
      delta_stepping_device_memory_reserve(free_bytes, total_bytes);
  if (free_bytes_out != nullptr) *free_bytes_out = free_bytes;
  if (reserve_out != nullptr) *reserve_out = reserve;
  return delta_stepping_device_memory_request_fits(
      requested_bytes, free_bytes, total_bytes);
}

inline void require_device_memory_headroom(std::size_t requested_bytes,
                                           const char* what) {
  if (requested_bytes == 0) return;
  std::size_t free_bytes = 0;
  std::size_t reserve = 0;
  if (device_memory_request_fits(requested_bytes, &free_bytes, &reserve)) {
    return;
  }
  std::ostringstream message;
  message << what << " needs " << requested_bytes
          << " device bytes, but only " << free_bytes
          << " are free and Delta-Stepping retains a shared reserve of "
          << reserve;
  throw std::runtime_error(message.str());
}

template <typename T>
inline void add_device_allocation_bytes(std::size_t count,
                                        std::size_t& total_bytes) {
  total_bytes = sssp_capacity::checked_add(
      total_bytes, sssp_capacity::checked_bytes<T>(count));
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { reset(count); }
  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(std::move(other)); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(count_, other.count_);
  }

  void reset(std::size_t count) {
    if (count == count_) return;
    const std::size_t bytes = sssp_capacity::checked_bytes<T>(count);
    T* candidate = nullptr;
    if (count != 0) {
      DS_DELTA_HIP_CHECK(
          hipMalloc(reinterpret_cast<void**>(&candidate), bytes));
    }
    release();
    ptr_ = candidate;
    count_ = count;
  }

  hipError_t try_reset(std::size_t count) noexcept {
    if (count == count_) {
      return hipSuccess;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return hipErrorOutOfMemory;
    }
    T* candidate = nullptr;
    if (count != 0) {
      const hipError_t status =
          hipMalloc(reinterpret_cast<void**>(&candidate),
                    count * sizeof(T));
      if (status != hipSuccess) {
        return status;
      }
    }
    release();
    ptr_ = candidate;
    count_ = count;
    return hipSuccess;
  }

  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  void move_from(DeviceBuffer&& other) noexcept {
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  explicit PinnedHostBuffer(std::size_t count) { reset(count); }
  ~PinnedHostBuffer() { release(); }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept {
    move_from(std::move(other));
  }
  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  void reset(std::size_t count) {
    if (count == count_) return;
    T* candidate = nullptr;
    if (count != 0) {
      DS_DELTA_HIP_CHECK(
          hipHostMalloc(reinterpret_cast<void**>(&candidate),
                        sssp_capacity::checked_bytes<T>(count),
                        hipHostMallocDefault));
    }
    release();
    ptr_ = candidate;
    count_ = count;
  }

  void swap(PinnedHostBuffer& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(count_, other.count_);
  }

  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) {
      (void)hipHostFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  void move_from(PinnedHostBuffer&& other) noexcept {
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

class HipEvent {
 public:
  HipEvent() = default;
  explicit HipEvent(unsigned int flags) { reset(flags); }
  ~HipEvent() { release(); }

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  HipEvent(HipEvent&& other) noexcept { move_from(std::move(other)); }
  HipEvent& operator=(HipEvent&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  void reset(unsigned int flags) {
    hipEvent_t candidate = nullptr;
    DS_DELTA_HIP_CHECK(hipEventCreateWithFlags(&candidate, flags));
    release();
    event_ = candidate;
  }

  void swap(HipEvent& other) noexcept {
    std::swap(event_, other.event_);
  }

  hipEvent_t get() const { return event_; }

 private:
  void release() noexcept {
    if (event_ != nullptr) {
      (void)hipEventDestroy(event_);
      event_ = nullptr;
    }
  }

  void move_from(HipEvent&& other) noexcept {
    event_ = other.event_;
    other.event_ = nullptr;
  }

  hipEvent_t event_ = nullptr;
};

template <typename RowOffset>
struct DeviceCsrView {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const RowOffset* rowptr = nullptr;
  const Index* colind = nullptr;
  const float* values = nullptr;
  bool implicit_unit_weights = false;
};

inline DeviceCsrView<Offset> wide_device_csr_view(
    const minplus_sparse::DeviceCsrF32& graph) {
  return {graph.rows, graph.cols, graph.nnz, graph.rowptr, graph.colind,
          graph.values, false};
}

struct DeviceCsrOwner {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  bool uses_32_bit_offsets = false;
  bool implicit_unit_weights = false;
  // Cooperative occupancy depends on both the row-offset width and whether
  // per-edge telemetry is compiled into the exact-unit kernel.
  int unit_cooperative_launch_blocks = 0;
  int unit_telemetry_cooperative_launch_blocks = 0;
  DeviceBuffer<CompactRowOffset> rowptr32;
  DeviceBuffer<Offset> rowptr64;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  DeviceBuffer<std::uint32_t> edge_source;
  // Kept separate from edge_source.get(): an eligible empty graph has a
  // deliberately null zero-length map.
  bool edge_source_available = false;

  DeviceCsrOwner() = default;

  template <typename RowOffset>
  DeviceCsrView<RowOffset> view() const {
    if constexpr (std::is_same<RowOffset, CompactRowOffset>::value) {
      if (!uses_32_bit_offsets) {
        throw std::logic_error(
            "requested compact view of wide Delta-Stepping CSR");
      }
      return {rows, cols, nnz, rowptr32.get(), colind.get(), values.get(),
              implicit_unit_weights};
    } else {
      static_assert(std::is_same<RowOffset, Offset>::value,
                    "unsupported Delta-Stepping row offset type");
      if (uses_32_bit_offsets) {
        throw std::logic_error(
            "requested wide view of compact Delta-Stepping CSR");
      }
      return {rows, cols, nnz, rowptr64.get(), colind.get(), values.get(),
              implicit_unit_weights};
    }
  }
};

struct DeltaSteppingScratch {
  Offset rows = 0;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<int> target_settled;
  DeviceBuffer<float> dist;
  DeviceBuffer<std::uint32_t> in_current;
  DeviceBuffer<int> in_pending;
  DeviceBuffer<int> in_heavy;
  DeviceBuffer<int> current_queue;
  DeviceBuffer<int> next_queue;
  DeviceBuffer<int> pending_a;
  DeviceBuffer<int> pending_b;
  DeviceBuffer<int> touched_queue;
  DeviceBuffer<int> heavy_queue;
  DeviceBuffer<int> current_count;
  DeviceBuffer<int> next_count;
  DeviceBuffer<int> pending_count;
  DeviceBuffer<int> new_pending_count;
  DeviceBuffer<int> heavy_count;
  DeviceBuffer<int> touched_count;
  DeviceBuffer<int> settled_target_count;
  DeviceBuffer<int> min_pending_bucket;
  // Unit-weight specialization status: append tail, targets reached, frontier
  // begin/end, next depth, whether the last expansion was active, last
  // discovered bucket, and distinct bucket rounds.
  DeviceBuffer<int> unit_status;
  // Lazily allocated only for telemetry-enabled invocations. Disabled runs do
  // not reset, copy, or pass this buffer to a kernel.
  DeviceBuffer<unsigned long long> telemetry_counters;
  // Lazy: compact edge-parent searches need neither legacy predecessor array.
  // Unit specialization and legacy/wide generic paths allocate both together.
  DeviceBuffer<int> pred_node;
  DeviceBuffer<Offset> pred_edge;
  DeviceBuffer<unsigned long long> parent_key;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_sources;
  DeviceBuffer<int> target_path_status;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  DeviceBuffer<TargetPathTotals> target_path_totals;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<std::uint32_t> compact_path_edges32;
  DeviceBuffer<Offset> compact_path_edges64;
  PinnedHostBuffer<int> host_scalar;
  PinnedHostBuffer<float> host_float_scalar;
  PinnedHostBuffer<int> host_unit_status;
  PinnedHostBuffer<float> host_target_distances;
  PinnedHostBuffer<int> host_target_path_lengths;
  PinnedHostBuffer<int> host_target_sources;
  PinnedHostBuffer<int> host_target_path_status;
  PinnedHostBuffer<int> host_target_node_offsets;
  PinnedHostBuffer<int> host_target_edge_offsets;
  PinnedHostBuffer<TargetPathTotals> host_target_path_totals;
  PinnedHostBuffer<int> host_compact_path_nodes;
  PinnedHostBuffer<std::uint32_t> host_compact_path_edges32;
  PinnedHostBuffer<Offset> host_compact_path_edges64;
  HipEvent control_ready_event;
  bool unit_initialized = false;
  bool generic_initialized = false;
  bool parent_key_initialized = false;
  bool legacy_predecessors_initialized = false;
  std::uint32_t current_generation = 0;
  std::uint32_t extraction_epoch = 0;

  DeltaSteppingScratch() = default;
  explicit DeltaSteppingScratch(Offset rows_) : rows(rows_) {
    if (rows_ < 0) {
      throw std::invalid_argument(
          "Delta-Stepping scratch rows must be nonnegative");
    }
    const std::size_t vertex_count = static_cast<std::size_t>(rows_);
    std::size_t device_bytes = 0;
    add_device_allocation_bytes<float>(vertex_count, device_bytes);
    add_device_allocation_bytes<int>(vertex_count, device_bytes);
    add_device_allocation_bytes<int>(vertex_count, device_bytes);
    add_device_allocation_bytes<int>(kUnitStatusCount, device_bytes);
    require_device_memory_headroom(
        device_bytes, "Delta-Stepping base workspace allocation");

    DeviceBuffer<float> next_dist(vertex_count);
    DeviceBuffer<int> next_in_pending(vertex_count);
    DeviceBuffer<int> next_current_queue(vertex_count);
    DeviceBuffer<int> next_unit_status(kUnitStatusCount);
    PinnedHostBuffer<int> next_host_scalar(1);
    PinnedHostBuffer<float> next_host_float_scalar(1);
    PinnedHostBuffer<int> next_host_unit_status(kUnitStatusCount);
    HipEvent next_control_ready_event(hipEventDisableTiming);
    dist.swap(next_dist);
    in_pending.swap(next_in_pending);
    current_queue.swap(next_current_queue);
    unit_status.swap(next_unit_status);
    host_scalar.swap(next_host_scalar);
    host_float_scalar.swap(next_host_float_scalar);
    host_unit_status.swap(next_host_unit_status);
    control_ready_event.swap(next_control_ready_event);
  }

  void ensure_legacy_predecessor_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    const bool grow_nodes = pred_node.size() < vertex_count;
    const bool grow_edges = pred_edge.size() < vertex_count;
    if (!grow_nodes && !grow_edges) return;

    std::size_t device_bytes = 0;
    if (grow_nodes) {
      add_device_allocation_bytes<int>(vertex_count, device_bytes);
    }
    if (grow_edges) {
      add_device_allocation_bytes<Offset>(vertex_count, device_bytes);
    }
    require_device_memory_headroom(
        device_bytes, "Delta-Stepping predecessor growth");

    DeviceBuffer<int> next_nodes;
    DeviceBuffer<Offset> next_edges;
    if (grow_nodes) next_nodes.reset(vertex_count);
    if (grow_edges) next_edges.reset(vertex_count);
    if (grow_nodes) pred_node.swap(next_nodes);
    if (grow_edges) pred_edge.swap(next_edges);
    legacy_predecessors_initialized = false;
  }

  void ensure_generic_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    const bool grow_in_current = in_current.size() < vertex_count;
    const bool grow_in_heavy = in_heavy.size() < vertex_count;
    const bool grow_next_queue = next_queue.size() < vertex_count;
    const bool grow_pending_a = pending_a.size() < vertex_count;
    const bool grow_pending_b = pending_b.size() < vertex_count;
    const bool grow_touched_queue = touched_queue.size() < vertex_count;
    const bool grow_heavy_queue = heavy_queue.size() < vertex_count;
    const bool grow_current_count = current_count.size() == 0;
    const bool grow_next_count = next_count.size() == 0;
    const bool grow_pending_count = pending_count.size() == 0;
    const bool grow_new_pending_count = new_pending_count.size() == 0;
    const bool grow_heavy_count = heavy_count.size() == 0;
    const bool grow_touched_count = touched_count.size() == 0;
    const bool grow_settled_target_count =
        settled_target_count.size() == 0;
    const bool grow_min_pending_bucket = min_pending_bucket.size() == 0;
    if (!grow_in_current && !grow_in_heavy && !grow_next_queue &&
        !grow_pending_a && !grow_pending_b && !grow_touched_queue &&
        !grow_heavy_queue && !grow_current_count && !grow_next_count &&
        !grow_pending_count && !grow_new_pending_count &&
        !grow_heavy_count && !grow_touched_count &&
        !grow_settled_target_count && !grow_min_pending_bucket) {
      return;
    }

    std::size_t device_bytes = 0;
    if (grow_in_current) {
      add_device_allocation_bytes<std::uint32_t>(vertex_count, device_bytes);
    }
    for (const bool grow : {grow_in_heavy, grow_next_queue, grow_pending_a,
                            grow_pending_b, grow_touched_queue,
                            grow_heavy_queue}) {
      if (grow) add_device_allocation_bytes<int>(vertex_count, device_bytes);
    }
    for (const bool grow :
         {grow_current_count, grow_next_count, grow_pending_count,
          grow_new_pending_count, grow_heavy_count, grow_touched_count,
          grow_settled_target_count, grow_min_pending_bucket}) {
      if (grow) add_device_allocation_bytes<int>(1, device_bytes);
    }
    require_device_memory_headroom(device_bytes,
                                   "Delta-Stepping generic workspace growth");

    DeviceBuffer<std::uint32_t> next_in_current;
    DeviceBuffer<int> next_in_heavy;
    DeviceBuffer<int> next_next_queue;
    DeviceBuffer<int> next_pending_a;
    DeviceBuffer<int> next_pending_b;
    DeviceBuffer<int> next_touched_queue;
    DeviceBuffer<int> next_heavy_queue;
    DeviceBuffer<int> next_current_count;
    DeviceBuffer<int> next_next_count;
    DeviceBuffer<int> next_pending_count;
    DeviceBuffer<int> next_new_pending_count;
    DeviceBuffer<int> next_heavy_count;
    DeviceBuffer<int> next_touched_count;
    DeviceBuffer<int> next_settled_target_count;
    DeviceBuffer<int> next_min_pending_bucket;
    if (grow_in_current) next_in_current.reset(vertex_count);
    if (grow_in_heavy) next_in_heavy.reset(vertex_count);
    if (grow_next_queue) next_next_queue.reset(vertex_count);
    if (grow_pending_a) next_pending_a.reset(vertex_count);
    if (grow_pending_b) next_pending_b.reset(vertex_count);
    if (grow_touched_queue) next_touched_queue.reset(vertex_count);
    if (grow_heavy_queue) next_heavy_queue.reset(vertex_count);
    if (grow_current_count) next_current_count.reset(1);
    if (grow_next_count) next_next_count.reset(1);
    if (grow_pending_count) next_pending_count.reset(1);
    if (grow_new_pending_count) next_new_pending_count.reset(1);
    if (grow_heavy_count) next_heavy_count.reset(1);
    if (grow_touched_count) next_touched_count.reset(1);
    if (grow_settled_target_count) next_settled_target_count.reset(1);
    if (grow_min_pending_bucket) next_min_pending_bucket.reset(1);

    if (grow_in_current) in_current.swap(next_in_current);
    if (grow_in_heavy) in_heavy.swap(next_in_heavy);
    if (grow_next_queue) next_queue.swap(next_next_queue);
    if (grow_pending_a) pending_a.swap(next_pending_a);
    if (grow_pending_b) pending_b.swap(next_pending_b);
    if (grow_touched_queue) touched_queue.swap(next_touched_queue);
    if (grow_heavy_queue) heavy_queue.swap(next_heavy_queue);
    if (grow_current_count) current_count.swap(next_current_count);
    if (grow_next_count) next_count.swap(next_next_count);
    if (grow_pending_count) pending_count.swap(next_pending_count);
    if (grow_new_pending_count) {
      new_pending_count.swap(next_new_pending_count);
    }
    if (grow_heavy_count) heavy_count.swap(next_heavy_count);
    if (grow_touched_count) touched_count.swap(next_touched_count);
    if (grow_settled_target_count) {
      settled_target_count.swap(next_settled_target_count);
    }
    if (grow_min_pending_bucket) {
      min_pending_bucket.swap(next_min_pending_bucket);
    }
    generic_initialized = false;
  }

  void ensure_parent_key_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    if (parent_key.size() < vertex_count) {
      require_device_memory_headroom(
          sssp_capacity::checked_bytes<unsigned long long>(vertex_count),
          "Delta-Stepping parent-key growth");
      DeviceBuffer<unsigned long long> next_parent_key(vertex_count);
      parent_key.swap(next_parent_key);
      parent_key_initialized = false;
    }
  }

  void ensure_source_capacity(std::size_t source_count) {
    source_count = sssp_capacity::checked_device_count(source_count);
    if (sources.size() < source_count) {
      const std::size_t capacity =
          delta_stepping_device_geometric_capacity(sources.size(),
                                                   source_count);
      require_device_memory_headroom(
          sssp_capacity::checked_bytes<int>(capacity),
          "Delta-Stepping source growth");
      DeviceBuffer<int> next_sources(capacity);
      sources.swap(next_sources);
    }
  }

  void ensure_telemetry_storage() {
    if (telemetry_counters.size() < kTelemetryCounterCount) {
      require_device_memory_headroom(
          sssp_capacity::checked_bytes<unsigned long long>(
              kTelemetryCounterCount),
          "Delta-Stepping telemetry growth");
      DeviceBuffer<unsigned long long> next_telemetry(kTelemetryCounterCount);
      telemetry_counters.swap(next_telemetry);
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    target_count = sssp_capacity::checked_device_count(target_count);
    const std::size_t capacity =
        delta_stepping_device_geometric_capacity(targets.size(),
                                                 target_count);
    const std::size_t required_offset_count =
        sssp_capacity::checked_target_offset_count(target_count);
    const std::size_t offset_capacity =
        sssp_capacity::checked_target_offset_count(capacity);
    const bool grow_targets = targets.size() < target_count;
    const bool grow_settled = target_settled.size() < target_count;
    const bool grow_distances = target_distances.size() < target_count;
    const bool grow_lengths = target_path_lengths.size() < target_count;
    const bool grow_sources = target_sources.size() < target_count;
    const bool grow_status = target_path_status.size() < target_count;
    const bool grow_node_offsets =
        target_node_offsets.size() < required_offset_count;
    const bool grow_edge_offsets =
        target_edge_offsets.size() < required_offset_count;
    const bool grow_totals = target_path_totals.size() == 0;
    const bool grow_host_distances =
        host_target_distances.size() < target_count;
    const bool grow_host_lengths =
        host_target_path_lengths.size() < target_count;
    const bool grow_host_sources =
        host_target_sources.size() < target_count;
    const bool grow_host_status =
        host_target_path_status.size() < target_count;
    const bool grow_host_node_offsets =
        host_target_node_offsets.size() < required_offset_count;
    const bool grow_host_edge_offsets =
        host_target_edge_offsets.size() < required_offset_count;
    const bool grow_host_totals = host_target_path_totals.size() == 0;
    if (!grow_targets && !grow_settled && !grow_distances &&
        !grow_lengths && !grow_sources && !grow_status &&
        !grow_node_offsets && !grow_edge_offsets && !grow_totals &&
        !grow_host_distances && !grow_host_lengths && !grow_host_sources &&
        !grow_host_status && !grow_host_node_offsets &&
        !grow_host_edge_offsets && !grow_host_totals) {
      return;
    }

    std::size_t device_bytes = 0;
    for (const bool grow :
         {grow_targets, grow_settled, grow_lengths, grow_sources,
          grow_status}) {
      if (grow) add_device_allocation_bytes<int>(capacity, device_bytes);
    }
    if (grow_distances) {
      add_device_allocation_bytes<float>(capacity, device_bytes);
    }
    if (grow_node_offsets) {
      add_device_allocation_bytes<int>(offset_capacity, device_bytes);
    }
    if (grow_edge_offsets) {
      add_device_allocation_bytes<int>(offset_capacity, device_bytes);
    }
    if (grow_totals) {
      add_device_allocation_bytes<TargetPathTotals>(1, device_bytes);
    }
    require_device_memory_headroom(device_bytes,
                                   "Delta-Stepping target growth");

    DeviceBuffer<int> next_targets;
    DeviceBuffer<int> next_settled;
    DeviceBuffer<float> next_distances;
    DeviceBuffer<int> next_lengths;
    DeviceBuffer<int> next_sources;
    DeviceBuffer<int> next_status;
    DeviceBuffer<int> next_node_offsets;
    DeviceBuffer<int> next_edge_offsets;
    DeviceBuffer<TargetPathTotals> next_totals;
    PinnedHostBuffer<float> next_host_distances;
    PinnedHostBuffer<int> next_host_lengths;
    PinnedHostBuffer<int> next_host_sources;
    PinnedHostBuffer<int> next_host_status;
    PinnedHostBuffer<int> next_host_node_offsets;
    PinnedHostBuffer<int> next_host_edge_offsets;
    PinnedHostBuffer<TargetPathTotals> next_host_totals;
    if (grow_targets) next_targets.reset(capacity);
    if (grow_settled) next_settled.reset(capacity);
    if (grow_distances) next_distances.reset(capacity);
    if (grow_lengths) next_lengths.reset(capacity);
    if (grow_sources) next_sources.reset(capacity);
    if (grow_status) next_status.reset(capacity);
    if (grow_node_offsets) next_node_offsets.reset(offset_capacity);
    if (grow_edge_offsets) next_edge_offsets.reset(offset_capacity);
    if (grow_totals) next_totals.reset(1);
    if (grow_host_distances) next_host_distances.reset(capacity);
    if (grow_host_lengths) next_host_lengths.reset(capacity);
    if (grow_host_sources) next_host_sources.reset(capacity);
    if (grow_host_status) next_host_status.reset(capacity);
    if (grow_host_node_offsets) {
      next_host_node_offsets.reset(offset_capacity);
    }
    if (grow_host_edge_offsets) {
      next_host_edge_offsets.reset(offset_capacity);
    }
    if (grow_host_totals) next_host_totals.reset(1);
    if (grow_targets) targets.swap(next_targets);
    if (grow_settled) target_settled.swap(next_settled);
    if (grow_distances) target_distances.swap(next_distances);
    if (grow_lengths) target_path_lengths.swap(next_lengths);
    if (grow_sources) target_sources.swap(next_sources);
    if (grow_status) target_path_status.swap(next_status);
    if (grow_node_offsets) target_node_offsets.swap(next_node_offsets);
    if (grow_edge_offsets) target_edge_offsets.swap(next_edge_offsets);
    if (grow_totals) target_path_totals.swap(next_totals);
    if (grow_host_distances) {
      host_target_distances.swap(next_host_distances);
    }
    if (grow_host_lengths) {
      host_target_path_lengths.swap(next_host_lengths);
    }
    if (grow_host_sources) host_target_sources.swap(next_host_sources);
    if (grow_host_status) host_target_path_status.swap(next_host_status);
    if (grow_host_node_offsets) {
      host_target_node_offsets.swap(next_host_node_offsets);
    }
    if (grow_host_edge_offsets) {
      host_target_edge_offsets.swap(next_host_edge_offsets);
    }
    if (grow_host_totals) host_target_path_totals.swap(next_host_totals);
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count,
                                    bool compact_edge_ids) {
    node_count = sssp_capacity::checked_device_count(node_count);
    edge_count = sssp_capacity::checked_device_count(edge_count);
    const bool grow_nodes = compact_path_nodes.size() < node_count;
    const std::size_t node_capacity =
        grow_nodes
            ? delta_stepping_device_geometric_capacity(
                  compact_path_nodes.size(), node_count)
            : compact_path_nodes.size();
    bool grow_edges = false;
    std::size_t edge_capacity = 0;
    if (compact_edge_ids) {
      grow_edges = compact_path_edges32.size() < edge_count;
      edge_capacity =
          grow_edges
              ? delta_stepping_device_geometric_capacity(
                    compact_path_edges32.size(), edge_count)
              : compact_path_edges32.size();
    } else {
      grow_edges = compact_path_edges64.size() < edge_count;
      edge_capacity =
          grow_edges
              ? delta_stepping_device_geometric_capacity(
                    compact_path_edges64.size(), edge_count)
              : compact_path_edges64.size();
    }
    const bool grow_host_nodes =
        host_compact_path_nodes.size() < node_count;
    const bool grow_host_edges =
        compact_edge_ids
            ? host_compact_path_edges32.size() < edge_count
            : host_compact_path_edges64.size() < edge_count;
    if (!grow_nodes && !grow_edges && !grow_host_nodes &&
        !grow_host_edges) {
      return;
    }

    std::size_t device_bytes = 0;
    if (grow_nodes) {
      add_device_allocation_bytes<int>(node_capacity, device_bytes);
    }
    if (grow_edges && compact_edge_ids) {
      add_device_allocation_bytes<std::uint32_t>(edge_capacity, device_bytes);
    } else if (grow_edges) {
      add_device_allocation_bytes<Offset>(edge_capacity, device_bytes);
    }
    require_device_memory_headroom(
        device_bytes, "Delta-Stepping compact-path growth");

    DeviceBuffer<int> next_nodes;
    DeviceBuffer<std::uint32_t> next_edges32;
    DeviceBuffer<Offset> next_edges64;
    PinnedHostBuffer<int> next_host_nodes;
    PinnedHostBuffer<std::uint32_t> next_host_edges32;
    PinnedHostBuffer<Offset> next_host_edges64;
    if (grow_nodes) next_nodes.reset(node_capacity);
    if (grow_edges && compact_edge_ids) {
      next_edges32.reset(edge_capacity);
    } else if (grow_edges) {
      next_edges64.reset(edge_capacity);
    }
    if (grow_host_nodes) next_host_nodes.reset(node_capacity);
    if (grow_host_edges && compact_edge_ids) {
      next_host_edges32.reset(edge_capacity);
    } else if (grow_host_edges) {
      next_host_edges64.reset(edge_capacity);
    }
    if (grow_nodes) compact_path_nodes.swap(next_nodes);
    if (grow_edges && compact_edge_ids) {
      compact_path_edges32.swap(next_edges32);
    } else if (grow_edges) {
      compact_path_edges64.swap(next_edges64);
    }
    if (grow_host_nodes) host_compact_path_nodes.swap(next_host_nodes);
    if (grow_host_edges && compact_edge_ids) {
      host_compact_path_edges32.swap(next_host_edges32);
    } else if (grow_host_edges) {
      host_compact_path_edges64.swap(next_host_edges64);
    }
  }

  std::uint32_t begin_target_extraction() {
    if (extraction_epoch == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "Delta-Stepping target extraction epoch exhausted");
    }
    return ++extraction_epoch;
  }

  void reserve_query_capacity(const SsspQueryCapacityHints& hints,
                              bool path_capable) {
    sssp_capacity::validate_reservation(hints);
    if (hints.max_sources != 0) {
      ensure_source_capacity(hints.max_sources);
    }
    if (path_capable && hints.max_targets != 0) {
      ensure_target_capacity(hints.max_targets);
    }
  }

  void release_parent_storage() {
    pred_node.reset(0);
    pred_edge.reset(0);
    parent_key.reset(0);
    unit_initialized = false;
    parent_key_initialized = false;
    legacy_predecessors_initialized = false;
  }

  void release_parent_and_path_storage() {
    release_parent_storage();
    targets.reset(0);
    target_settled.reset(0);
    target_distances.reset(0);
    target_path_lengths.reset(0);
    target_sources.reset(0);
    target_path_status.reset(0);
    target_node_offsets.reset(0);
    target_edge_offsets.reset(0);
    target_path_totals.reset(0);
    compact_path_nodes.reset(0);
    compact_path_edges32.reset(0);
    compact_path_edges64.reset(0);
    host_target_distances.reset(0);
    host_target_path_lengths.reset(0);
    host_target_sources.reset(0);
    host_target_path_status.reset(0);
    host_target_node_offsets.reset(0);
    host_target_edge_offsets.reset(0);
    host_target_path_totals.reset(0);
    host_compact_path_nodes.reset(0);
    host_compact_path_edges32.reset(0);
    host_compact_path_edges64.reset(0);
  }

};

inline int grid_for_items(Offset items, int block_size = kBlockSize) {
  if (items <= 0) return 1;
  const long long blocks = (static_cast<long long>(items) + block_size - 1) / block_size;
  return static_cast<int>(std::min<long long>(kMaxGridX, std::max<long long>(1, blocks)));
}

inline int grid_for_frontier(int items) {
  if (items <= 0) return 1;
  const long long blocks =
      (static_cast<long long>(items) + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(
      std::min<long long>(kMaxGridX, std::max<long long>(1, blocks)));
}

inline std::size_t checked_size(Offset x, const char* name) {
  if (x < 0) {
    throw std::invalid_argument(std::string(name) + " must be nonnegative");
  }
  return static_cast<std::size_t>(x);
}

inline void validate_delta(float delta) {
  if (!(delta > 0.0f) || !std::isfinite(delta)) {
    throw std::invalid_argument("delta must be a finite positive float");
  }
}

inline void validate_common_shape(Offset rows,
                                  Offset cols,
                                  Offset nnz,
                                  int source,
                                  int target) {
  if (rows <= 0) throw std::invalid_argument("CSR graph must contain at least one vertex");
  if (rows != cols) throw std::invalid_argument("SSSP expects a square CSR adjacency matrix");
  if (nnz < 0) throw std::invalid_argument("CSR nnz must be nonnegative");
  if (source < 0 || static_cast<Offset>(source) >= rows) {
    throw std::out_of_range("source vertex is outside CSR row range");
  }
  if (target < -1 || static_cast<Offset>(target) >= rows) {
    throw std::out_of_range("target vertex is outside CSR row range");
  }
  if (static_cast<unsigned long long>(rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("frontier vertices are stored as int; rows must fit in int");
  }
}

inline void validate_source_list_common_shape(Offset rows,
                                              Offset cols,
                                              Offset nnz,
                                              const std::vector<int>& sources,
                                              int target) {
  if (rows <= 0) throw std::invalid_argument("CSR graph must contain at least one vertex");
  if (rows != cols) throw std::invalid_argument("SSSP expects a square CSR adjacency matrix");
  if (nnz < 0) throw std::invalid_argument("CSR nnz must be nonnegative");
  if (sources.empty()) {
    throw std::invalid_argument("at least one source vertex is required");
  }
  if (sources.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("source count must fit in int");
  }
  for (const int source : sources) {
    if (source < 0 || static_cast<Offset>(source) >= rows) {
      throw std::out_of_range("source vertex is outside CSR row range");
    }
  }
  if (target < -1 || static_cast<Offset>(target) >= rows) {
    throw std::out_of_range("target vertex is outside CSR row range");
  }
  if (static_cast<unsigned long long>(rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("frontier vertices are stored as int; rows must fit in int");
  }
}

inline void validate_target_list_common_shape(Offset rows,
                                              const std::vector<int>& targets) {
  if (targets.empty()) {
    throw std::invalid_argument("at least one target vertex is required");
  }
  if (targets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("target count must fit in int");
  }
  for (const int target : targets) {
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      throw std::out_of_range("target vertex is outside CSR row range");
    }
  }
}

struct StableTargetDeduplication {
  std::vector<int> unique_targets;
  std::vector<std::size_t> original_to_unique;

  bool has_duplicates() const noexcept {
    return unique_targets.size() != original_to_unique.size();
  }
};

StableTargetDeduplication stable_deduplicate_targets(
    const std::vector<int>& targets) {
  StableTargetDeduplication deduplication;
  deduplication.unique_targets.reserve(targets.size());
  deduplication.original_to_unique.reserve(targets.size());

  std::unordered_map<int, std::size_t> unique_index;
  unique_index.reserve(targets.size());
  for (const int target : targets) {
    const std::size_t next_unique = deduplication.unique_targets.size();
    const auto inserted = unique_index.emplace(target, next_unique);
    if (inserted.second) {
      deduplication.unique_targets.push_back(target);
    }
    deduplication.original_to_unique.push_back(inserted.first->second);
  }
  return deduplication;
}

DeltaSteppingCsrResult fan_out_deduplicated_target_result(
    DeltaSteppingCsrResult result,
    const StableTargetDeduplication& deduplication) {
  if (!deduplication.has_duplicates()) {
    return result;
  }

  const std::size_t unique_count = deduplication.unique_targets.size();
  const std::size_t original_count =
      deduplication.original_to_unique.size();
  const std::size_t unique_offset_count =
      sssp_capacity::checked_target_offset_count(unique_count);
  if (result.target_distances.size() != unique_count ||
      result.target_sources.size() != unique_count ||
      result.target_path_offsets.size() != unique_offset_count ||
      result.target_edge_offsets.size() != unique_offset_count) {
    throw std::runtime_error(
        "deduplicated Delta target result has an invalid shape");
  }

  const std::vector<float> unique_distances =
      std::move(result.target_distances);
  const std::vector<int> unique_sources = std::move(result.target_sources);
  const std::vector<int> unique_node_offsets =
      std::move(result.target_path_offsets);
  const std::vector<int> unique_edge_offsets =
      std::move(result.target_edge_offsets);
  const std::vector<int> unique_path_nodes =
      std::move(result.target_path_nodes);
  const std::vector<Offset> unique_path_edges =
      std::move(result.target_path_edges);

  result.target_distances.resize(original_count);
  result.target_sources.resize(original_count);
  result.target_path_offsets.assign(
      sssp_capacity::checked_target_offset_count(original_count), 0);
  result.target_edge_offsets.assign(
      sssp_capacity::checked_target_offset_count(original_count), 0);

  std::size_t total_nodes = 0;
  std::size_t total_edges = 0;
  for (std::size_t original = 0; original < original_count; ++original) {
    const std::size_t unique =
        deduplication.original_to_unique[original];
    if (unique >= unique_count) {
      throw std::runtime_error(
          "Delta target fan-out index is outside the unique target set");
    }
    const int node_begin = unique_node_offsets[unique];
    const int node_end = unique_node_offsets[unique + 1];
    const int edge_begin = unique_edge_offsets[unique];
    const int edge_end = unique_edge_offsets[unique + 1];
    if (node_begin < 0 || node_end < node_begin ||
        edge_begin < 0 || edge_end < edge_begin ||
        static_cast<std::size_t>(node_end) > unique_path_nodes.size() ||
        static_cast<std::size_t>(edge_end) > unique_path_edges.size()) {
      throw std::runtime_error(
          "deduplicated Delta target result contains an invalid path slice");
    }

    result.target_distances[original] = unique_distances[unique];
    result.target_sources[original] = unique_sources[unique];
    result.target_path_offsets[original] = static_cast<int>(total_nodes);
    result.target_edge_offsets[original] = static_cast<int>(total_edges);
    total_nodes = sssp_capacity::checked_add(
        total_nodes,
        static_cast<std::size_t>(node_end - node_begin));
    total_edges = sssp_capacity::checked_add(
        total_edges,
        static_cast<std::size_t>(edge_end - edge_begin));
    if (total_nodes >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        total_edges >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "fanned-out Delta target paths are too large for int offsets");
    }
  }
  result.target_path_offsets[original_count] =
      static_cast<int>(total_nodes);
  result.target_edge_offsets[original_count] =
      static_cast<int>(total_edges);
  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);

  std::size_t node_destination = 0;
  std::size_t edge_destination = 0;
  for (const std::size_t unique : deduplication.original_to_unique) {
    const std::size_t node_begin =
        static_cast<std::size_t>(unique_node_offsets[unique]);
    const std::size_t node_end =
        static_cast<std::size_t>(unique_node_offsets[unique + 1]);
    const std::size_t edge_begin =
        static_cast<std::size_t>(unique_edge_offsets[unique]);
    const std::size_t edge_end =
        static_cast<std::size_t>(unique_edge_offsets[unique + 1]);
    const std::size_t node_count = node_end - node_begin;
    const std::size_t edge_count = edge_end - edge_begin;
    if (node_count != 0) {
      std::copy_n(unique_path_nodes.data() + node_begin,
                  node_count,
                  result.target_path_nodes.data() + node_destination);
      node_destination += node_count;
    }
    if (edge_count != 0) {
      std::copy_n(unique_path_edges.data() + edge_begin,
                  edge_count,
                  result.target_path_edges.data() + edge_destination);
      edge_destination += edge_count;
    }
  }
  return result;
}

inline void validate_host_csr_arrays(const HostCsrF32& g) {
  const std::size_t rows = checked_size(g.rows, "rows");
  const std::size_t nnz = checked_size(g.nnz, "nnz");
  if (g.rowptr.size() != sssp_capacity::checked_add(rows, 1)) {
    throw std::invalid_argument("rowptr.size() must equal rows + 1");
  }
  if (g.colind.size() != nnz) throw std::invalid_argument("colind.size() must equal nnz");
  if (g.values.size() != nnz) throw std::invalid_argument("values.size() must equal nnz");
  if (g.rowptr.front() != 0 || g.rowptr.back() != g.nnz) {
    throw std::invalid_argument("CSR rowptr must start at 0 and end at nnz");
  }
  for (std::size_t r = 0; r < rows; ++r) {
    if (g.rowptr[r] < 0 || g.rowptr[r + 1] < g.rowptr[r] || g.rowptr[r + 1] > g.nnz) {
      throw std::invalid_argument("CSR rowptr must be monotone and within [0, nnz]");
    }
  }
  for (std::size_t e = 0; e < nnz; ++e) {
    if (g.colind[e] < 0 || static_cast<Offset>(g.colind[e]) >= g.cols) {
      throw std::invalid_argument("CSR colind contains an out-of-range destination vertex");
    }
    if (!std::isfinite(g.values[e]) || g.values[e] < 0.0f) {
      throw std::invalid_argument("delta-stepping requires finite nonnegative edge weights");
    }
  }
}

inline float max_edge_value(const std::vector<float>& values) {
  float max_value = 0.0f;
  for (const float value : values) {
    max_value = std::max(max_value, value);
  }
  return max_value;
}

inline bool has_exact_unit_edge_values(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) {
    return value == 1.0f;
  });
}

inline void validate_host_csr(const HostCsrF32& g, int source, int target, float delta) {
  validate_common_shape(g.rows, g.cols, g.nnz, source, target);
  validate_delta(delta);
  validate_host_csr_arrays(g);
}

inline void validate_host_csr(const HostCsrF32& g,
                              const std::vector<int>& sources,
                              int target,
                              float delta) {
  validate_source_list_common_shape(g.rows, g.cols, g.nnz, sources, target);
  validate_delta(delta);
  validate_host_csr_arrays(g);
}

template <typename RowOffset>
inline void validate_device_csr_shape(const DeviceCsrView<RowOffset>& g,
                                      int source,
                                      int target,
                                      float delta) {
  validate_common_shape(g.rows, g.cols, g.nnz, source, target);
  validate_delta(delta);
  if (g.rowptr == nullptr) throw std::invalid_argument("device CSR rowptr is null");
  if (g.nnz > 0 &&
      (g.colind == nullptr ||
       (!g.implicit_unit_weights && g.values == nullptr))) {
    throw std::invalid_argument(
        "device CSR colind/values are null for a nonempty graph");
  }
}

template <typename RowOffset>
inline void validate_device_csr_shape(const DeviceCsrView<RowOffset>& g,
                                      const std::vector<int>& sources,
                                      int target,
                                      float delta) {
  validate_source_list_common_shape(g.rows, g.cols, g.nnz, sources, target);
  validate_delta(delta);
  if (g.rowptr == nullptr) throw std::invalid_argument("device CSR rowptr is null");
  if (g.nnz > 0 &&
      (g.colind == nullptr ||
       (!g.implicit_unit_weights && g.values == nullptr))) {
    throw std::invalid_argument(
        "device CSR colind/values are null for a nonempty graph");
  }
}

inline void wait_for_control_copy(hipEvent_t event, hipStream_t stream) {
  if (event == nullptr) {
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    return;
  }
  DS_DELTA_HIP_CHECK(hipEventRecord(event, stream));
  DS_DELTA_HIP_CHECK(hipEventSynchronize(event));
}

template <typename T>
inline T copy_scalar_to_host(const T* d_value,
                             hipStream_t stream,
                             T* host_staging = nullptr,
                             hipEvent_t completion_event = nullptr) {
  T pageable_value{};
  T* const destination = host_staging == nullptr ? &pageable_value : host_staging;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(destination,
                                    d_value,
                                    sizeof(T),
                                    hipMemcpyDeviceToHost,
                                    stream));
  wait_for_control_copy(completion_event, stream);
  return *destination;
}

inline void copy_unit_status_to_host(DeltaSteppingScratch& scratch,
                                     hipStream_t stream) {
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_unit_status.get(), scratch.unit_status.get(),
      sssp_capacity::checked_bytes<int>(kUnitStatusCount),
      hipMemcpyDeviceToHost, stream));
  wait_for_control_copy(scratch.control_ready_event.get(), stream);
}

inline void reset_int_zero_async(int* d_value, hipStream_t stream) {
  DS_DELTA_HIP_CHECK(hipMemsetAsync(d_value, 0, sizeof(int), stream));
}

inline void synchronize_explicit_stream(hipStream_t stream) {
  if (stream != nullptr) {
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  }
}

// Keep the telemetry-disabled relaxation primitive identical to the
// production implementation that predates opt-in instrumentation. In
// particular, return the old distance as a scalar so disabled telemetry does
// not change the ABI/register handoff that controls parent publication.
__device__ inline float atomic_min_float_nonnegative(float* addr,
                                                     float value) {
  auto* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *addr_as_uint;
  while (value < __uint_as_float(old)) {
    const unsigned int assumed = old;
    old = atomicCAS(addr_as_uint, assumed, __float_as_uint(value));
    if (old == assumed) break;
  }
  return __uint_as_float(old);
}

struct AtomicMinFloatResult {
  float old_value;
  unsigned int cas_retries;
};

__device__ inline unsigned int atomic_load_uint(
    const unsigned int* address) {
  // Compact extraction consumes state produced by atomic relaxations in an
  // earlier dispatch. Keep validation reads on the coherent atomic path so a
  // reused gfx1151 worker cannot combine values from different routes.
  return atomicCAS(const_cast<unsigned int*>(address), 0U, 0U);
}

__device__ inline int atomic_load_int(const int* address) {
  return atomicAdd(const_cast<int*>(address), 0);
}

__device__ inline unsigned long long atomic_load_parent_key(
    const unsigned long long* address) {
  return atomicCAS(const_cast<unsigned long long*>(address), 0ULL, 0ULL);
}

__device__ inline float atomic_load_float(const float* address) {
  return __uint_as_float(atomic_load_uint(
      reinterpret_cast<const unsigned int*>(address)));
}

template <bool CollectTelemetry>
__device__ inline AtomicMinFloatResult atomic_min_float_nonnegative(
    float* addr,
    float value) {
  auto* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *addr_as_uint;
  unsigned int retries = 0;
  while (value < __uint_as_float(old)) {
    const unsigned int assumed = old;
    old = atomicCAS(addr_as_uint, assumed, __float_as_uint(value));
    if (old == assumed) break;
    if constexpr (CollectTelemetry) {
      ++retries;
    }
  }
  return {__uint_as_float(old), retries};
}

__device__ inline bool finite_float(float value) {
  return (__float_as_uint(value) & 0x7f800000U) != 0x7f800000U;
}

__device__ inline bool infinite_float(float value) {
  return (__float_as_uint(value) & 0x7fffffffU) == 0x7f800000U;
}

template <bool ImplicitUnitWeights>
__device__ inline float device_edge_weight(const float* values, Offset edge) {
  if constexpr (ImplicitUnitWeights) {
    return 1.0f;
  }
  return values[edge];
}

template <bool TrackParents, bool UseEdgeParent>
__device__ inline void publish_parent_candidate(
    unsigned long long* parent_key,
    int predecessor,
    Offset edge,
    float candidate_distance) {
  if constexpr (TrackParents) {
    std::uint32_t payload = 0;
    if constexpr (UseEdgeParent) {
      payload = static_cast<std::uint32_t>(edge);
    } else {
      payload = static_cast<std::uint32_t>(predecessor);
    }
    const unsigned long long candidate_key =
        (static_cast<unsigned long long>(
             __float_as_uint(candidate_distance)) << 32) |
        payload;
    unsigned long long old = *parent_key;
    while (candidate_key < old) {
      const unsigned long long assumed = old;
      old = atomicCAS(parent_key, assumed, candidate_key);
      if (old == assumed) break;
    }
  }
}

template <typename RowOffset>
__global__ void build_edge_source_kernel(Offset rows,
                                         const RowOffset* rowptr,
                                         std::uint32_t* edge_source) {
  for (Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<Offset>(blockDim.x) * gridDim.x) {
    for (Offset edge = static_cast<Offset>(rowptr[row]);
         edge < static_cast<Offset>(rowptr[row + 1]); ++edge) {
      edge_source[edge] = static_cast<std::uint32_t>(row);
    }
  }
}

inline int bucket_index_host(float distance, float delta) {
  if (!std::isfinite(distance)) return kNoBucket;
  const float bucket_f = distance / delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  // Distances are nonnegative, so integer truncation is floor without an
  // additional libm operation.
  return static_cast<int>(bucket_f);
}

__device__ inline int bucket_index(float distance, float delta) {
  if (!finite_float(distance)) return kNoBucket;
  const float bucket_f = distance / delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  return static_cast<int>(bucket_f);
}

__device__ inline int append_position(bool append, int* queue_tail) {
  // Callers include divergent adjacency loops, where different lanes execute
  // different numbers of iterations. Wave collectives are not valid at those
  // call sites, so reserve each successful append independently.
  return append ? atomicAdd(queue_tail, 1) : -1;
}

__device__ inline int atomic_load_unit_status(int* address) {
  // Keep the device-resident unit-BFS controller on the same coherent atomic
  // path as its queue-tail and target-count writers.  Plain reads here can
  // otherwise retain stale uniform data between kernels on gfx1151.
  return atomicAdd(address, 0);
}

__device__ inline void atomic_store_unit_status(int* address, int value) {
  atomicExch(address, value);
}

__device__ inline int atomic_load_counter(const int* address) {
  // Queue counts are produced with atomicAdd and then consumed as uniform
  // values by a later kernel.  Keep the read on the coherent atomic path; a
  // plain scalar load can retain the previous dispatch's count on gfx1151.
  return atomicAdd(const_cast<int*>(address), 0);
}

__device__ inline void telemetry_atomic_max(unsigned long long* address,
                                            unsigned long long value) {
  unsigned long long old = *address;
  while (old < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(address, assumed, value);
    if (old == assumed) break;
  }
}

template <int CounterCount>
__device__ inline void add_block_telemetry(
    const unsigned long long (&local)[CounterCount],
    unsigned long long* global_counters) {
  __shared__ unsigned long long reduced[CounterCount];
  for (int counter = threadIdx.x; counter < CounterCount;
       counter += blockDim.x) {
    reduced[counter] = 0;
  }
  __syncthreads();
  for (int counter = 0; counter < CounterCount; ++counter) {
    if (local[counter] != 0) {
      atomicAdd(reduced + counter, local[counter]);
    }
  }
  __syncthreads();
  for (int counter = threadIdx.x; counter < CounterCount;
       counter += blockDim.x) {
    if (reduced[counter] != 0) {
      atomicAdd(global_counters + counter, reduced[counter]);
    }
  }
}

__device__ inline void update_block_queue_peaks(
    unsigned long long current_peak,
    unsigned long long pending_peak,
    unsigned long long heavy_peak,
    unsigned long long* global_counters) {
  __shared__ unsigned long long peaks[3];
  if (threadIdx.x < 3) peaks[threadIdx.x] = 0;
  __syncthreads();
  telemetry_atomic_max(peaks + 0, current_peak);
  telemetry_atomic_max(peaks + 1, pending_peak);
  telemetry_atomic_max(peaks + 2, heavy_peak);
  __syncthreads();
  if (threadIdx.x == 0) {
    telemetry_atomic_max(
        global_counters + kTelemetryCurrentQueueHighWater, peaks[0]);
    telemetry_atomic_max(
        global_counters + kTelemetryPendingQueueHighWater, peaks[1]);
    telemetry_atomic_max(
        global_counters + kTelemetryHeavyQueueHighWater, peaks[2]);
  }
}

template <typename RowOffset>
__global__ void validate_device_csr_kernel(Offset rows,
                                           Offset cols,
                                           Offset nnz,
                                           const RowOffset* rowptr,
                                           const Index* colind,
                                           const float* values,
                                           int* invalid) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (rowptr[0] != 0 || rowptr[rows] != nnz) atomicExch(invalid, 1);
  }
  for (Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<Offset>(blockDim.x) * gridDim.x) {
    const Offset begin = static_cast<Offset>(rowptr[row]);
    const Offset end = static_cast<Offset>(rowptr[row + 1]);
    if (begin < 0 || end < begin || end > nnz) {
      atomicExch(invalid, 1);
      continue;
    }
    for (Offset e = begin; e < end; ++e) {
      const Index dst = colind[e];
      // Raw device views reject null values before launch. Internal
      // exact-unit views deliberately omit the values allocation.
      const float w = values == nullptr ? 1.0f : values[e];
      if (dst < 0 || static_cast<Offset>(dst) >= cols ||
          !finite_float(w) || w < 0.0f) {
        atomicExch(invalid, 1);
      }
    }
  }
}

__global__ void initialize_delta_arrays_kernel(Offset n,
                                                float inf,
                                                float* dist,
                                                std::uint32_t* in_current,
                                                int* in_pending,
                                                int* in_heavy,
                                                int* current_count,
                                                int* next_count,
                                                int* pending_count,
                                                int* heavy_count,
                                                int* touched_count) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < n;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    dist[v] = inf;
    in_current[v] = 0;
    in_pending[v] = 0;
    in_heavy[v] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *current_count = 0;
    *next_count = 0;
    *pending_count = 0;
    *heavy_count = 0;
    *touched_count = 0;
  }
}

__global__ void initialize_parent_keys_kernel(
    Offset n,
    unsigned long long* parent_key) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < n;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    parent_key[v] = kNoParentKey;
  }
}

__global__ void initialize_legacy_predecessors_kernel(Offset n,
                                                       int* pred_node,
                                                       Offset* pred_edge) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < n;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
  }
}

__global__ void initialize_unit_arrays_kernel(Offset n,
                                              float inf,
                                              float* dist,
                                              int* target_multiplicity,
                                              int* pred_node,
                                              Offset* pred_edge) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < n;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    dist[v] = inf;
    target_multiplicity[v] = 0;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
  }
}

template <bool InitializeLegacyPredecessors>
__global__ void initialize_delta_sources_kernel(const int* sources,
                                                int source_count,
                                                float* dist,
                                                std::uint32_t* in_current,
                                                std::uint32_t current_generation,
                                                int* current_queue,
                                                int* current_count,
                                                int* pred_node,
                                                Offset* pred_edge,
                                                int* touched_queue,
                                                int* touched_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *current_count = source_count;
    *touched_count = source_count;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    dist[source] = 0.0f;
    in_current[source] = current_generation;
    if constexpr (InitializeLegacyPredecessors) {
      pred_node[source] = source;
      pred_edge[source] = static_cast<Offset>(-1);
    }
    current_queue[i] = source;
    touched_queue[i] = source;
  }
}

__global__ void mark_unit_target_multiplicity_kernel(
    const int* targets,
    int target_count,
    int* target_multiplicity) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&target_multiplicity[targets[i]], 1);
  }
}

__global__ void clear_unit_target_multiplicity_kernel(
    const int* targets,
    int target_count,
    int* target_multiplicity) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    // Duplicate target entries may map several lanes to the same word.
    atomicExch(&target_multiplicity[targets[i]], 0);
  }
}

__global__ void initialize_unit_sources_kernel(const int* sources,
                                               int source_count,
                                               int initially_found,
                                               int target_count,
                                               int max_depth,
                                               float* dist,
                                               int* pred_node,
                                               Offset* pred_edge,
                                               int* frontier_queue,
                                               int* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomic_store_unit_status(status + kUnitStatusQueueTail, source_count);
    atomic_store_unit_status(
        status + kUnitStatusFoundCount, initially_found);
    atomic_store_unit_status(status + kUnitStatusFrontierBegin, 0);
    atomic_store_unit_status(
        status + kUnitStatusFrontierEnd, source_count);
    atomic_store_unit_status(status + kUnitStatusCompletedDepth, 0);
    atomic_store_unit_status(status + kUnitStatusBucket, 0);
    atomic_store_unit_status(
        status + kUnitStatusBucketRounds,
        source_count > 0 && initially_found < target_count && max_depth > 0
            ? 1
            : 0);
    __threadfence();
    atomic_store_unit_status(
        status + kUnitStatusActive,
        source_count > 0 && initially_found < target_count && max_depth > 0);
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    dist[source] = 0.0f;
    pred_node[source] = source;
    pred_edge[source] = static_cast<Offset>(-1);
    frontier_queue[i] = source;
  }
}

template <typename RowOffset, bool CollectTelemetry>
__device__ inline void expand_unit_frontier_range(
    int frontier_begin,
    int frontier_end,
    int next_depth,
    const RowOffset* out_rowptr,
    const Index* out_colind,
    float* dist,
    int* pred_node,
    Offset* pred_edge,
    int* frontier_queue,
    int* queue_tail,
    int* found_count,
    const int* target_multiplicity,
    unsigned long long* telemetry_counters) {
  const float next_distance = static_cast<float>(next_depth);
  const unsigned int infinity_bits = __float_as_uint(INFINITY);
  const unsigned int next_distance_bits = __float_as_uint(next_distance);
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long current_peak = 0;
  for (int i = frontier_begin + blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_end;
       i += blockDim.x * gridDim.x) {
    if constexpr (CollectTelemetry) {
      ++telemetry[kTelemetryFrontierEntries];
      ++telemetry[kTelemetryActiveVertices];
    }
    const int u = frontier_queue[i];
    for (Offset edge = static_cast<Offset>(out_rowptr[u]);
         edge < static_cast<Offset>(out_rowptr[u + 1]); ++edge) {
      if constexpr (CollectTelemetry) {
        ++telemetry[kTelemetryLightEdgeVisits];
      }
      const int v = static_cast<int>(out_colind[edge]);
      auto* const distance_bits =
          reinterpret_cast<unsigned int*>(&dist[v]);
      const bool attempted = *distance_bits == infinity_bits;
      const bool claimed = attempted &&
                           atomicCAS(distance_bits,
                                     infinity_bits,
                                     next_distance_bits) == infinity_bits;
      if constexpr (CollectTelemetry) {
        if (attempted) {
          ++telemetry[kTelemetryDistanceAtomicAttempts];
          if (!claimed) ++telemetry[kTelemetryDistanceCasRetries];
        }
        if (claimed) {
          ++telemetry[kTelemetrySuccessfulRelaxations];
          ++telemetry[kTelemetryCurrentQueueInsertions];
        }
      }
      if (claimed) {
        pred_node[v] = u;
        pred_edge[v] = edge;
      }
      const int pos = append_position(claimed, queue_tail);
      if (claimed) {
        frontier_queue[pos] = v;
        if constexpr (CollectTelemetry) {
          const unsigned long long observed_peak =
              static_cast<unsigned long long>(pos) + 1;
          if (observed_peak > current_peak) current_peak = observed_peak;
        }
        const int multiplicity = target_multiplicity[v];
        if (multiplicity > 0) {
          atomicAdd(found_count, multiplicity);
        }
      }
    }
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, telemetry_counters);
    update_block_queue_peaks(current_peak, 0, 0, telemetry_counters);
  }
}

template <typename RowOffset, bool CollectTelemetry>
__global__ void expand_unit_frontier_kernel(const RowOffset* out_rowptr,
                                            const Index* out_colind,
                                            float* dist,
                                            int* pred_node,
                                            Offset* pred_edge,
                                            int* frontier_queue,
                                            int* status,
                                            const int* target_multiplicity,
                                            unsigned long long* telemetry_counters) {
  __shared__ int controller[4];
  if (threadIdx.x == 0) {
    controller[0] = atomic_load_unit_status(status + kUnitStatusActive);
    if (controller[0] != 0) {
      controller[1] =
          atomic_load_unit_status(status + kUnitStatusFrontierBegin);
      controller[2] =
          atomic_load_unit_status(status + kUnitStatusFrontierEnd);
      controller[3] =
          atomic_load_unit_status(status + kUnitStatusCompletedDepth);
    }
  }
  __syncthreads();
  if (controller[0] == 0) return;
  expand_unit_frontier_range<RowOffset, CollectTelemetry>(
      controller[1], controller[2], controller[3] + 1, out_rowptr,
      out_colind, dist, pred_node, pred_edge, frontier_queue,
      status + kUnitStatusQueueTail, status + kUnitStatusFoundCount,
      target_multiplicity, telemetry_counters);
}

template <typename RowOffset, bool CollectTelemetry>
__global__ void expand_unit_frontier_host_controlled_kernel(
    int frontier_begin,
    int frontier_end,
    int next_depth,
    const RowOffset* out_rowptr,
    const Index* out_colind,
    float* dist,
    int* pred_node,
    Offset* pred_edge,
    int* frontier_queue,
    int* queue_tail,
    int* found_count,
    const int* target_multiplicity,
    unsigned long long* telemetry_counters) {
  expand_unit_frontier_range<RowOffset, CollectTelemetry>(
      frontier_begin, frontier_end, next_depth, out_rowptr, out_colind, dist,
      pred_node, pred_edge, frontier_queue, queue_tail, found_count,
      target_multiplicity, telemetry_counters);
}

__global__ void advance_unit_frontier_kernel(int* status,
                                             float delta,
                                             int target_count,
                                             int max_depth) {
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      atomic_load_unit_status(status + kUnitStatusActive) != 0) {
    const int queue_tail =
        atomic_load_unit_status(status + kUnitStatusQueueTail);
    const int previous_end =
        atomic_load_unit_status(status + kUnitStatusFrontierEnd);
    const int completed_depth =
        atomic_load_unit_status(status + kUnitStatusCompletedDepth) + 1;
    const int found_count =
        atomic_load_unit_status(status + kUnitStatusFoundCount);
    int bucket = atomic_load_unit_status(status + kUnitStatusBucket);
    int bucket_rounds =
        atomic_load_unit_status(status + kUnitStatusBucketRounds);
    if (queue_tail > previous_end) {
      const int discovered_depth = completed_depth;
      const int discovered_bucket =
          bucket_index(static_cast<float>(discovered_depth), delta);
      if (discovered_bucket != bucket) {
        bucket = discovered_bucket;
        ++bucket_rounds;
      }
    }
    atomic_store_unit_status(status + kUnitStatusBucket, bucket);
    atomic_store_unit_status(
        status + kUnitStatusBucketRounds, bucket_rounds);
    atomic_store_unit_status(
        status + kUnitStatusFrontierBegin, previous_end);
    atomic_store_unit_status(status + kUnitStatusFrontierEnd, queue_tail);
    atomic_store_unit_status(
        status + kUnitStatusCompletedDepth, completed_depth);
    __threadfence();
    atomic_store_unit_status(
        status + kUnitStatusActive,
        previous_end < queue_tail && found_count < target_count &&
            completed_depth < max_depth);
  }
}

template <typename RowOffset, bool CollectTelemetry>
__global__ void cooperative_unit_frontier_controller_kernel(
    const RowOffset* out_rowptr,
    const Index* out_colind,
    float delta,
    int target_count,
    int max_depth,
    int level_budget,
    float* dist,
    int* pred_node,
    Offset* pred_edge,
    int* frontier_queue,
    int* status,
    const int* target_multiplicity,
    unsigned long long* telemetry_counters) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __shared__ int controller[4];

  // Source initialization publishes the complete initial state before this
  // launch. Every later transition remains inside this cooperative kernel and
  // is separated from frontier expansion by a grid-wide barrier. This avoids
  // both per-level host decisions and the cross-dispatch controller handoff
  // that is unreliable on explicit gfx1151 worker streams.
  int levels_this_launch = 0;
  while (levels_this_launch < level_budget) {
    if (threadIdx.x == 0) {
      controller[0] =
          atomic_load_unit_status(status + kUnitStatusActive);
      if (controller[0] != 0) {
        controller[1] =
            atomic_load_unit_status(status + kUnitStatusFrontierBegin);
        controller[2] =
            atomic_load_unit_status(status + kUnitStatusFrontierEnd);
        controller[3] =
            atomic_load_unit_status(status + kUnitStatusCompletedDepth);
      }
    }
    __syncthreads();
    if (controller[0] == 0) break;

    const int frontier_begin = controller[1];
    const int frontier_end = controller[2];
    const int next_depth = controller[3] + 1;
    expand_unit_frontier_range<RowOffset, CollectTelemetry>(
        frontier_begin, frontier_end, next_depth, out_rowptr, out_colind, dist,
        pred_node, pred_edge, frontier_queue,
        status + kUnitStatusQueueTail,
        status + kUnitStatusFoundCount,
        target_multiplicity, telemetry_counters);
    grid.sync();

    if (grid.thread_rank() == 0) {
      const int queue_tail =
          atomic_load_unit_status(status + kUnitStatusQueueTail);
      const int found_count =
          atomic_load_unit_status(status + kUnitStatusFoundCount);
      int bucket = atomic_load_unit_status(status + kUnitStatusBucket);
      int bucket_rounds =
          atomic_load_unit_status(status + kUnitStatusBucketRounds);
      if (queue_tail > frontier_end) {
        const int discovered_bucket =
            bucket_index(static_cast<float>(next_depth), delta);
        if (discovered_bucket != bucket) {
          bucket = discovered_bucket;
          ++bucket_rounds;
        }
      }
      atomic_store_unit_status(status + kUnitStatusBucket, bucket);
      atomic_store_unit_status(
          status + kUnitStatusBucketRounds, bucket_rounds);
      atomic_store_unit_status(
          status + kUnitStatusFrontierBegin, frontier_end);
      atomic_store_unit_status(
          status + kUnitStatusFrontierEnd, queue_tail);
      atomic_store_unit_status(
          status + kUnitStatusCompletedDepth, next_depth);
      __threadfence();
      atomic_store_unit_status(
          status + kUnitStatusActive,
          frontier_end < queue_tail && found_count < target_count &&
              next_depth < max_depth);
    }
    grid.sync();
    ++levels_this_launch;
  }
}

template <typename RowOffset, bool CollectTelemetry>
int unit_cooperative_controller_blocks(Offset rows) {
  int device = -1;
  DS_DELTA_HIP_CHECK(hipGetDevice(&device));
  int cooperative_launch = 0;
  const hipError_t capability_status =
      hipDeviceGetAttribute(&cooperative_launch,
                            hipDeviceAttributeCooperativeLaunch,
                            device);
  if (capability_status != hipSuccess || cooperative_launch == 0) {
    if (capability_status != hipSuccess) {
      (void)hipGetLastError();
    }
    return 0;
  }

  hipDeviceProp_t properties{};
  DS_DELTA_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  int active_blocks_per_compute_unit = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks_per_compute_unit,
          cooperative_unit_frontier_controller_kernel<RowOffset,
                                                      CollectTelemetry>,
          kBlockSize,
          0);
  if (occupancy_status != hipSuccess ||
      active_blocks_per_compute_unit <= 0 ||
      properties.multiProcessorCount <= 0) {
    if (occupancy_status != hipSuccess) {
      (void)hipGetLastError();
    }
    return 0;
  }

  const Offset row_blocks =
      (rows + static_cast<Offset>(kBlockSize) - 1) /
      static_cast<Offset>(kBlockSize);
  const Offset legal_resident_limit =
      static_cast<Offset>(active_blocks_per_compute_unit) *
      static_cast<Offset>(properties.multiProcessorCount);
  // Pathfinder may run eight workers at once. Reserve only a fraction of the
  // legal cooperative residency per workspace, targeting roughly one
  // aggregate block per CU at the common four-worker setting. Small routing
  // frontiers therefore avoid an oversized device-wide barrier, while all
  // launched blocks remain simultaneously resident as grid.sync requires.
  const Offset per_worker_resident_limit =
      std::max<Offset>(
          1,
          legal_resident_limit /
              static_cast<Offset>(kMaxConcurrentCooperativeWorkers));
  const Offset balanced_worker_blocks =
      (static_cast<Offset>(properties.multiProcessorCount) +
       static_cast<Offset>(kTargetConcurrentCooperativeWorkers) - 1) /
      static_cast<Offset>(kTargetConcurrentCooperativeWorkers);
  const Offset concurrency_friendly_limit =
      std::min(balanced_worker_blocks, per_worker_resident_limit);
  const Offset blocks =
      std::min(row_blocks,
               std::min(concurrency_friendly_limit, legal_resident_limit));
  if (blocks <= 0 ||
      blocks > static_cast<Offset>(std::numeric_limits<int>::max())) {
    return 0;
  }
  return static_cast<int>(blocks);
}

template <typename RowOffset, bool ImplicitUnitWeights>
__global__ void materialize_predecessors_kernel(
    const int* touched_vertices,
    int touched_count,
    const RowOffset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    const float* dist,
    const unsigned long long* parent_key,
    int* pred_node,
    Offset* pred_edge) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_vertices[i];
    if (pred_node[v] == v) {
      // Sources are marked directly and have no predecessor edge.
      continue;
    }
    const unsigned long long key = parent_key[v];
    if (key == kNoParentKey) continue;
    const int u = static_cast<int>(static_cast<unsigned int>(key));
    const float du = dist[u];
    const float dv = dist[v];
    if (!finite_float(du) || !finite_float(dv)) continue;

    for (Offset edge = static_cast<Offset>(rowptr[u]);
         edge < static_cast<Offset>(rowptr[u + 1]); ++edge) {
      if (static_cast<int>(colind[edge]) != v) continue;
      const float edge_cost =
          device_edge_weight<ImplicitUnitWeights>(values, edge) *
          (vertex_costs == nullptr ? 1.0f : vertex_costs[v]);
      const float candidate = du + edge_cost;
      if (__float_as_uint(candidate) == __float_as_uint(dv)) {
        pred_node[v] = u;
        pred_edge[v] = edge;
        break;
      }
    }
  }
}

__global__ void mark_settled_targets_kernel(const int* targets,
                                            int target_count,
                                            int current_bucket,
                                            float delta,
                                            const float* dist,
                                            int* target_settled,
                                            int* settled_count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (target_settled[i] != 0) continue;
    const int target = targets[i];
    const float target_distance = dist[target];
    if (finite_float(target_distance) &&
        bucket_index(target_distance, delta) <= current_bucket &&
        atomicCAS(&target_settled[i], 0, 1) == 0) {
      atomicAdd(settled_count, 1);
    }
  }
}

__global__ void measure_target_paths_kernel(const int* targets,
                                            int target_count,
                                            Offset rows,
                                            const int* target_settled,
                                            const float* dist,
                                            const int* pred_node,
                                            float* target_distances,
                                            int* path_lengths,
                                            int* path_sources,
                                            int* path_status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    if (target_settled != nullptr && target_settled[i] == 0) {
      target_distances[i] = INFINITY;
      path_lengths[i] = 0;
      path_sources[i] = -1;
      path_status[i] = 0;
      continue;
    }
    const float target_distance = dist[target];
    target_distances[i] = target_distance;
    path_lengths[i] = 0;
    path_sources[i] = -1;
    path_status[i] = 0;
    if (!finite_float(target_distance)) {
      continue;
    }

    int current = target;
    int length = 1;
    for (Offset guard = 0; guard < rows; ++guard) {
      const int source_marker = pred_node[current];
      if (source_marker == current) {
        path_lengths[i] = length;
        path_sources[i] = current;
        path_status[i] = 1;
        break;
      }

      const int pred = pred_node[current];
      if (pred < 0 || static_cast<Offset>(pred) >= rows) {
        break;
      }
      current = pred;
      ++length;
    }
  }
}

__global__ void measure_unit_target_paths_kernel(const int* targets,
                                                 int target_count,
                                                 const float* dist,
                                                 float exclusive_distance_limit,
                                                 float* target_distances,
                                                 int* path_lengths,
                                                 int* path_sources,
                                                 int* path_status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const float target_distance = dist[targets[i]];
    path_sources[i] = -1;
    if (!finite_float(target_distance) ||
        !(target_distance < exclusive_distance_limit)) {
      target_distances[i] = INFINITY;
      path_lengths[i] = 0;
      path_status[i] = 0;
      continue;
    }
    target_distances[i] = target_distance;
    // In the unit specialization, distance is exactly the BFS edge depth.
    path_lengths[i] = static_cast<int>(target_distance) + 1;
    path_status[i] = 1;
  }
}

__global__ void scan_target_path_offsets_kernel(
    int target_count,
    const float* target_distances,
    const int* path_lengths,
    const int* path_status,
    std::uint32_t extraction_epoch,
    int* node_offsets,
    int* edge_offsets,
    TargetPathTotals* totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  totals->total_nodes = 0;
  totals->total_edges = 0;
  totals->extraction_epoch = extraction_epoch;
  atomicExch(&totals->status, kTargetOffsetScanNotPublished);
  node_offsets[0] = 0;
  edge_offsets[0] = 0;

  int total_nodes = 0;
  int total_edges = 0;
  int scan_status = kTargetOffsetScanValid;
  for (int i = 0; i < target_count; ++i) {
    const float distance = target_distances[i];
    const int length = path_lengths[i];
    const int status = path_status[i];
    const bool reached = finite_float(distance);
    const bool unreachable = infinite_float(distance);
    if ((!reached && !unreachable) ||
        (reached && (status == 0 || length <= 0)) ||
        (unreachable && (status != 0 || length != 0))) {
      scan_status = kTargetOffsetScanInvalidMetadata;
      break;
    }

    const int nodes = reached ? length : 0;
    const int edges = nodes > 0 ? nodes - 1 : 0;
    if (nodes < 0 || edges < 0 ||
        total_nodes > std::numeric_limits<int>::max() - nodes ||
        total_edges > std::numeric_limits<int>::max() - edges) {
      scan_status = kTargetOffsetScanOverflow;
      break;
    }
    total_nodes += nodes;
    total_edges += edges;
    node_offsets[i + 1] = total_nodes;
    edge_offsets[i + 1] = total_edges;
  }

  totals->total_nodes = total_nodes;
  totals->total_edges = total_edges;
  __threadfence_system();
  atomicExch(&totals->status, scan_status);
}

template <typename RowOffset>
__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const RowOffset* rowptr,
                                         const Index* colind,
                                         const int* pred_node,
                                         const Offset* pred_edge,
                                         const int* path_lengths,
                                         int* path_status,
                                         int* path_sources,
                                         const int* node_offsets,
                                         const int* edge_offsets,
                                         int* path_nodes,
                                         Offset* path_edges) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (path_status[i] == 0) {
      continue;
    }
    const int length = path_lengths[i];
    int current = targets[i];
    const int node_begin = node_offsets[i];
    const int edge_begin = edge_offsets[i];
    bool path_valid = true;
    path_nodes[node_begin + length - 1] = current;
    for (int j = length - 1; j > 0; --j) {
      const int source_marker = pred_node[current];
      if (source_marker == current) {
        path_valid = false;
        break;
      }
      const int pred = source_marker;
      if (pred < 0 || static_cast<Offset>(pred) >= rows) {
        path_valid = false;
        break;
      }
      const Offset edge = pred_edge[current];
      if (edge < static_cast<Offset>(rowptr[pred]) ||
          edge >= static_cast<Offset>(rowptr[pred + 1]) ||
          static_cast<int>(colind[edge]) != current) {
        path_valid = false;
        break;
      }
      path_edges[edge_begin + j - 1] = edge;
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
    if (path_valid && pred_node[current] == current) {
      path_sources[i] = current;
    } else {
      path_status[i] = 0;
      path_sources[i] = -1;
    }
  }
}

template <typename RowOffset, bool ImplicitUnitWeights>
__device__ inline bool decode_tight_edge_parent(
    int current,
    Offset rows,
    Offset nnz,
    const RowOffset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const float* dist,
    unsigned long long key,
    Offset* edge_out,
    int* predecessor_out) {
  const float current_distance = atomic_load_float(&dist[current]);
  if (key == kNoParentKey ||
      static_cast<unsigned int>(key >> 32) !=
          __float_as_uint(current_distance)) {
    return false;
  }

  const Offset edge =
      static_cast<Offset>(static_cast<std::uint32_t>(key));
  if (edge < 0 || edge >= nnz ||
      static_cast<int>(colind[edge]) != current) {
    return false;
  }
  const std::uint32_t predecessor_bits = edge_source[edge];
  if (static_cast<unsigned long long>(predecessor_bits) >=
      static_cast<unsigned long long>(rows)) {
    return false;
  }
  const int predecessor = static_cast<int>(predecessor_bits);
  if (edge < static_cast<Offset>(rowptr[predecessor]) ||
      edge >= static_cast<Offset>(rowptr[predecessor + 1])) {
    return false;
  }

  const float predecessor_distance = atomic_load_float(&dist[predecessor]);
  if (!finite_float(predecessor_distance) || !finite_float(current_distance)) {
    return false;
  }
  const float effective_weight =
      device_edge_weight<ImplicitUnitWeights>(values, edge) *
      (vertex_costs == nullptr ? 1.0f : vertex_costs[current]);
  if (__float_as_uint(predecessor_distance + effective_weight) !=
      __float_as_uint(current_distance)) {
    return false;
  }
  *edge_out = edge;
  *predecessor_out = predecessor;
  return true;
}

template <typename RowOffset, bool ImplicitUnitWeights>
__global__ void measure_edge_parent_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    Offset nnz,
    const RowOffset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const int* target_settled,
    const float* dist,
    const unsigned long long* parent_key,
    float* target_distances,
    int* path_lengths,
    int* path_sources,
    int* path_status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    if (target_settled != nullptr &&
        atomic_load_int(&target_settled[i]) == 0) {
      target_distances[i] = INFINITY;
      path_lengths[i] = 0;
      path_sources[i] = -1;
      atomicExch(&path_status[i], 0);
      continue;
    }
    const float target_distance = atomic_load_float(&dist[target]);
    target_distances[i] = target_distance;
    path_lengths[i] = 0;
    path_sources[i] = -1;
    atomicExch(&path_status[i], 0);
    if (!finite_float(target_distance)) {
      continue;
    }

    int current = target;
    int length = 1;
    for (Offset guard = 0; guard < rows; ++guard) {
      const unsigned long long key =
          atomic_load_parent_key(&parent_key[current]);
      if (key == kNoParentKey) {
        path_lengths[i] = length;
        path_sources[i] = current;
        atomicExch(&path_status[i], 1);
        break;
      }

      Offset edge = 0;
      int predecessor = -1;
      if (!decode_tight_edge_parent<RowOffset, ImplicitUnitWeights>(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        break;
      }
      current = predecessor;
      ++length;
    }
  }
}

template <typename RowOffset, bool ImplicitUnitWeights>
__global__ void fill_edge_parent_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    Offset nnz,
    const RowOffset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const float* dist,
    const unsigned long long* parent_key,
    const int* path_lengths,
    int* path_status,
    int* path_sources,
    const int* node_offsets,
    const int* edge_offsets,
    int* path_nodes,
    std::uint32_t* path_edges) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (atomic_load_int(&path_status[i]) == 0) {
      continue;
    }
    const int length = path_lengths[i];
    int current = targets[i];
    const int node_begin = node_offsets[i];
    const int edge_begin = edge_offsets[i];
    bool path_valid = true;
    path_nodes[node_begin + length - 1] = current;
    for (int j = length - 1; j > 0; --j) {
      const unsigned long long key =
          atomic_load_parent_key(&parent_key[current]);
      Offset edge = 0;
      int predecessor = -1;
      if (!decode_tight_edge_parent<RowOffset, ImplicitUnitWeights>(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        path_valid = false;
        break;
      }
      path_edges[edge_begin + j - 1] = static_cast<std::uint32_t>(edge);
      current = predecessor;
      path_nodes[node_begin + j - 1] = current;
    }
    if (path_valid &&
        atomic_load_parent_key(&parent_key[current]) == kNoParentKey) {
      path_sources[i] = current;
    } else {
      atomicExch(&path_status[i], 0);
      path_sources[i] = -1;
    }
  }
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectHeavy,
          bool AllEdgesLight,
          bool CollectTelemetry>
__global__ void relax_light_edges_kernel(const int* frontier,
                                         const int* frontier_count_ptr,
                                         int current_bucket,
                                         float delta,
                                         float exclusive_distance_limit,
                                         const RowOffset* out_rowptr,
                                         const Index* out_colind,
                                         const float* out_values,
                                         const float* vertex_costs,
                                         float* dist,
                                         unsigned long long* parent_key,
                                         std::uint32_t* in_current,
                                         std::uint32_t next_current_generation,
                                         int* in_pending,
                                         int* in_heavy,
                                         int* touched_queue,
                                         int* touched_count,
                                         int* next_frontier,
                                         int* next_count,
                                         int* pending_queue,
                                         int* pending_count,
                                         int* heavy_queue,
                                         int* heavy_count,
                                         unsigned long long* telemetry_counters) {
  // FPGA routing graphs have short outgoing rows.  Assign one active vertex to
  // each thread so a 256-thread block can process up to 256 rows concurrently,
  // matching the unit-BFS traversal instead of idling most of a block per row.
  __shared__ int frontier_count;
  if (threadIdx.x == 0) {
    frontier_count = atomic_load_counter(frontier_count_ptr);
  }
  __syncthreads();
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long current_peak =
      threadIdx.x == 0 ? static_cast<unsigned long long>(frontier_count) : 0;
  unsigned long long pending_peak = 0;
  unsigned long long heavy_peak = 0;
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < frontier_count;
       fi += blockDim.x * gridDim.x) {
    if constexpr (CollectTelemetry) {
      ++telemetry[kTelemetryFrontierEntries];
    }
    const int u = frontier[fi];
    // Boolean membership for this whole frontier is cleared by a separate
    // kernel before relaxation starts. Generation membership instead claims a
    // fresh next-frontier token. Do not mutate either representation here:
    // doing so races with other blocks that may need to re-enqueue this vertex
    // after a same-bucket distance decrease.
    const float du = dist[u];
    const bool active =
        finite_float(du) && bucket_index(du, delta) == current_bucket;
    if constexpr (CollectTelemetry) {
      ++telemetry[active ? kTelemetryActiveVertices
                         : kTelemetryStaleFrontierEntries];
    }
    if constexpr (CollectHeavy) {
      const bool append_heavy =
          active && atomicCAS(&in_heavy[u], 0, 1) == 0;
      const int heavy_pos = append_position(append_heavy, heavy_count);
      if (append_heavy) {
        heavy_queue[heavy_pos] = u;
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryHeavyQueueInsertions];
          const unsigned long long observed_peak =
              static_cast<unsigned long long>(heavy_pos) + 1;
          if (observed_peak > heavy_peak) heavy_peak = observed_peak;
        }
      }
    }
    if (active) {
      for (Offset e = static_cast<Offset>(out_rowptr[u]);
           e < static_cast<Offset>(out_rowptr[u + 1]); ++e) {
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryLightEdgeVisits];
        }
        const float w =
            device_edge_weight<ImplicitUnitWeights>(out_values, e);
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            HasVertexCosts ? w * vertex_costs[v] : w;
        const float candidate = du + effective_w;
        const bool below_distance_limit =
            candidate < exclusive_distance_limit;
        int candidate_bucket = kNoBucket;
        // Float addition can place a nominally heavy edge in the current
        // bucket (and all very large distances share the terminal bucket).
        // Such work belongs to light closure or its descendants can be lost.
        bool light = true;
        if constexpr (!AllEdgesLight) {
          candidate_bucket = below_distance_limit
                                 ? bucket_index(candidate, delta)
                                 : kNoBucket;
          light = below_distance_limit &&
                  (effective_w <= delta ||
                   candidate_bucket == current_bucket);
        } else {
          light = below_distance_limit;
        }
        const float nd = light ? candidate : INFINITY;
        float old = INFINITY;
        if constexpr (CollectTelemetry) {
          if (light) {
            const AtomicMinFloatResult atomic_result =
                atomic_min_float_nonnegative<true>(&dist[v], nd);
            old = atomic_result.old_value;
            ++telemetry[kTelemetryDistanceAtomicAttempts];
            telemetry[kTelemetryDistanceCasRetries] +=
                atomic_result.cas_retries;
          }
        } else {
          old = light
                    ? atomic_min_float_nonnegative(&dist[v], nd)
                    : INFINITY;
        }
        const bool decreased = light && nd < old;
        const bool append_touched = decreased && infinite_float(old);
        bool append_current = false;
        bool append_pending = false;
        if (decreased) {
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetrySuccessfulRelaxations];
          }
          if constexpr (AllEdgesLight) {
            candidate_bucket = bucket_index(nd, delta);
          }
          if constexpr (TrackParents) {
            publish_parent_candidate<true, UseEdgeParent>(
                &parent_key[v], u, e, nd);
          }
          const int b = candidate_bucket;
          if (b == current_bucket) {
            // Keep any existing pending token marked until compaction/reset.
            // Clearing it here lets a delayed, already-successful relaxation
            // set the flag again and append a second token for v. The pending
            // reduction ignores tokens whose live distance is in this bucket.
            if constexpr (UseCurrentGenerations) {
              append_current =
                  atomicExch(&in_current[v], next_current_generation) !=
                  next_current_generation;
            } else {
              append_current = atomicCAS(&in_current[v], 0U, 1U) == 0U;
            }
          } else if (b > current_bucket && b < kNoBucket) {
            append_pending = atomicCAS(&in_pending[v], 0, 1) == 0;
          }
        }
        const int touched_pos =
            append_position(append_touched, touched_count);
        const int current_pos =
            append_position(append_current, next_count);
        const int pending_pos =
            append_position(append_pending, pending_count);
        if (append_touched) touched_queue[touched_pos] = v;
        if (append_current) {
          next_frontier[current_pos] = v;
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryCurrentQueueInsertions];
            const unsigned long long observed_peak =
                static_cast<unsigned long long>(current_pos) + 1;
            if (observed_peak > current_peak) current_peak = observed_peak;
          }
        }
        if (append_pending) {
          pending_queue[pending_pos] = v;
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryPendingQueueInsertions];
            const unsigned long long observed_peak =
                static_cast<unsigned long long>(pending_pos) + 1;
            if (observed_peak > pending_peak) pending_peak = observed_peak;
          }
        }
      }
    }
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, telemetry_counters);
    update_block_queue_peaks(current_peak, pending_peak, heavy_peak,
                             telemetry_counters);
  }
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectTelemetry>
__global__ void relax_heavy_edges_kernel(const int* heavy_vertices,
                                         const int* heavy_count_ptr,
                                         int current_bucket,
                                         float delta,
                                         float exclusive_distance_limit,
                                         const RowOffset* out_rowptr,
                                         const Index* out_colind,
                                         const float* out_values,
                                         const float* vertex_costs,
                                         float* dist,
                                         unsigned long long* parent_key,
                                         int* in_pending,
                                         int* touched_queue,
                                         int* touched_count,
                                         int* pending_queue,
                                         int* pending_count,
                                         int* in_heavy,
                                         unsigned long long* telemetry_counters) {
  __shared__ int heavy_count_value;
  if (threadIdx.x == 0) {
    heavy_count_value = atomic_load_counter(heavy_count_ptr);
  }
  __syncthreads();
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long pending_peak = 0;
  const unsigned long long heavy_peak =
      threadIdx.x == 0
          ? static_cast<unsigned long long>(heavy_count_value)
          : 0;
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < heavy_count_value;
       fi += blockDim.x * gridDim.x) {
    const int u = heavy_vertices[fi];
    const float du = dist[u];
    if (finite_float(du)) {
      for (Offset e = static_cast<Offset>(out_rowptr[u]);
           e < static_cast<Offset>(out_rowptr[u + 1]); ++e) {
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryHeavyEdgeVisits];
        }
        const float w =
            device_edge_weight<ImplicitUnitWeights>(out_values, e);
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            HasVertexCosts ? w * vertex_costs[v] : w;
        const float candidate = du + effective_w;
        const bool below_distance_limit =
            candidate < exclusive_distance_limit;
        const int candidate_bucket =
            below_distance_limit ? bucket_index(candidate, delta) : kNoBucket;
        // Same-bucket nominally-heavy edges were already processed by the
        // light-closure kernel. Only future-bucket candidates belong here.
        const bool heavy = below_distance_limit && effective_w > delta &&
                           candidate_bucket > current_bucket &&
                           candidate_bucket < kNoBucket;
        const float nd = heavy ? candidate : INFINITY;
        float old = INFINITY;
        if constexpr (CollectTelemetry) {
          if (heavy) {
            const AtomicMinFloatResult atomic_result =
                atomic_min_float_nonnegative<true>(&dist[v], nd);
            old = atomic_result.old_value;
            ++telemetry[kTelemetryDistanceAtomicAttempts];
            telemetry[kTelemetryDistanceCasRetries] +=
                atomic_result.cas_retries;
          }
        } else {
          old = heavy
                    ? atomic_min_float_nonnegative(&dist[v], nd)
                    : INFINITY;
        }
        const bool decreased = heavy && nd < old;
        const bool append_touched = decreased && infinite_float(old);
        bool append_pending = false;
        if (decreased) {
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetrySuccessfulRelaxations];
          }
          if constexpr (TrackParents) {
            publish_parent_candidate<true, UseEdgeParent>(
                &parent_key[v], u, e, nd);
          }
          const int b = candidate_bucket;
          if (b > current_bucket && b < kNoBucket) {
            append_pending = atomicCAS(&in_pending[v], 0, 1) == 0;
          }
        }
        const int touched_pos =
            append_position(append_touched, touched_count);
        const int pending_pos =
            append_position(append_pending, pending_count);
        if (append_touched) touched_queue[touched_pos] = v;
        if (append_pending) {
          pending_queue[pending_pos] = v;
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryPendingQueueInsertions];
            const unsigned long long observed_peak =
                static_cast<unsigned long long>(pending_pos) + 1;
            if (observed_peak > pending_peak) pending_peak = observed_peak;
          }
        }
      }
    }
    in_heavy[u] = 0;
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, telemetry_counters);
    update_block_queue_peaks(0, pending_peak, heavy_peak,
                             telemetry_counters);
  }
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectHeavy,
          bool AllEdgesLight,
          bool CollectTelemetry>
void launch_relax_light_edges(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    const float* vertex_costs,
    const int* current_queue,
    const int* current_count,
    int launch_blocks,
    int current_bucket,
    float delta,
    float exclusive_distance_limit,
    int* next_queue,
    int* next_count,
    std::uint32_t next_current_generation,
    int* pending_queue,
    hipStream_t stream) {
  relax_light_edges_kernel<RowOffset, ImplicitUnitWeights,
                           UseCurrentGenerations, TrackParents, UseEdgeParent,
                           HasVertexCosts, CollectHeavy, AllEdgesLight,
                           CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta,
          exclusive_distance_limit,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(),
          scratch.in_current.get(), next_current_generation,
          scratch.in_pending.get(),
          scratch.in_heavy.get(), scratch.touched_queue.get(),
          scratch.touched_count.get(), next_queue, next_count,
          pending_queue, scratch.pending_count.get(), scratch.heavy_queue.get(),
          scratch.heavy_count.get(),
          CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectTelemetry>
void launch_relax_heavy_edges(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    const float* vertex_costs,
    int launch_blocks,
    int current_bucket,
    float delta,
    float exclusive_distance_limit,
    int* pending_queue,
    hipStream_t stream) {
  relax_heavy_edges_kernel<RowOffset, ImplicitUnitWeights, TrackParents,
                           UseEdgeParent, HasVertexCosts, CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          scratch.heavy_queue.get(), scratch.heavy_count.get(),
          current_bucket, delta, exclusive_distance_limit,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(), scratch.in_pending.get(),
          scratch.touched_queue.get(), scratch.touched_count.get(),
          pending_queue, scratch.pending_count.get(), scratch.in_heavy.get(),
          CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

__global__ void clear_flags_from_queue_kernel(const int* vertices,
                                              const int* count_ptr,
                                              std::uint32_t* flags) {
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = atomic_load_counter(count_ptr);
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count;
       i += blockDim.x * gridDim.x) {
    flags[vertices[i]] = 0U;
  }
}

template <bool ResetCurrentMembership>
__global__ void reset_touched_vertices_kernel(const int* touched_queue,
                                              int touched_count,
                                              float inf,
                                              float* dist,
                                              std::uint32_t* in_current,
                                              int* in_pending,
                                              int* in_heavy,
                                              int* pred_node,
                                              Offset* pred_edge,
                                              unsigned long long* parent_key) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    if constexpr (ResetCurrentMembership) {
      in_current[v] = 0U;
    }
    in_pending[v] = 0;
    in_heavy[v] = 0;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
    parent_key[v] = kNoParentKey;
  }
}

template <bool ResetCurrentMembership>
__global__ void reset_distance_only_touched_vertices_kernel(
    const int* touched_queue,
    int touched_count,
    float inf,
    float* dist,
    std::uint32_t* in_current,
    int* in_pending,
    int* in_heavy) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    if constexpr (ResetCurrentMembership) {
      in_current[v] = 0U;
    }
    in_pending[v] = 0;
    in_heavy[v] = 0;
  }
}

template <bool ResetCurrentMembership, bool ResetHeavyMembership>
__global__ void reset_compact_parent_touched_vertices_kernel(
    const int* touched_queue,
    int touched_count,
    float inf,
    float* dist,
    std::uint32_t* in_current,
    int* in_pending,
    int* in_heavy,
    unsigned long long* parent_key) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    parent_key[v] = kNoParentKey;
    if constexpr (ResetCurrentMembership) {
      in_current[v] = 0U;
    }
    in_pending[v] = 0;
    if constexpr (ResetHeavyMembership) {
      in_heavy[v] = 0;
    }
  }
}

__global__ void reset_unit_visited_kernel(const int* visited_queue,
                                          int visited_count,
                                          float inf,
                                          float* dist,
                                          int* pred_node,
                                          Offset* pred_edge) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < visited_count;
       i += blockDim.x * gridDim.x) {
    const int v = visited_queue[i];
    dist[v] = inf;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
  }
}

template <bool CollectTelemetry>
__global__ void reduce_min_pending_bucket_kernel(const int* pending_queue,
                                                 const int* pending_count_ptr,
                                                 int previous_bucket,
                                                 float delta,
                                                 const float* dist,
                                                 const int* in_pending,
                                                 int* global_min,
                                                 unsigned long long* telemetry_counters) {
  __shared__ int s_min[kBlockSize];
  __shared__ int shared_pending_count;
  const int tid = threadIdx.x;
  if (tid == 0) {
    shared_pending_count = atomic_load_counter(pending_count_ptr);
  }
  __syncthreads();
  const int pending_count = shared_pending_count;
  int local_min = kNoBucket;
  unsigned long long examined = 0;
  unsigned long long stale = 0;
  for (int i = blockIdx.x * blockDim.x + tid;
       i < pending_count;
       i += blockDim.x * gridDim.x) {
    const int v = pending_queue[i];
    const bool active = in_pending[v] != 0;
    if constexpr (CollectTelemetry) {
      ++examined;
    }
    if (active) {
      const int b = bucket_index(dist[v], delta);
      if (b > previous_bucket && b < local_min) local_min = b;
      if constexpr (CollectTelemetry) {
        if (b <= previous_bucket || b >= kNoBucket) ++stale;
      }
    } else if constexpr (CollectTelemetry) {
      ++stale;
    }
  }
  s_min[tid] = local_min;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_min[tid] = (s_min[tid] < s_min[tid + stride]) ? s_min[tid] : s_min[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0 && s_min[0] < kNoBucket) {
    atomicMin(global_min, s_min[0]);
  }
  if constexpr (CollectTelemetry) {
    unsigned long long telemetry[kTelemetryCounterCount] = {};
    telemetry[kTelemetryPendingEntryExaminations] = examined;
    telemetry[kTelemetryStalePendingEntryExaminations] = stale;
    add_block_telemetry(telemetry, telemetry_counters);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      telemetry_atomic_max(
          telemetry_counters + kTelemetryPendingQueueHighWater,
          static_cast<unsigned long long>(pending_count));
    }
  }
}

template <bool UseCurrentGenerations, bool CollectTelemetry>
__global__ void compact_pending_to_current_bucket_kernel(const int* pending_in,
                                                         const int* pending_count_ptr,
                                                         int selected_bucket,
                                                         float delta,
                                                         const float* dist,
                                                         int* in_pending,
                                                         std::uint32_t* in_current,
                                                         std::uint32_t current_generation,
                                                         int* current_queue,
                                                         int* current_count,
                                                         int* pending_out,
                                                         int* new_pending_count,
                                                         unsigned long long* telemetry_counters) {
  __shared__ int pending_count;
  if (threadIdx.x == 0) {
    pending_count = atomic_load_counter(pending_count_ptr);
  }
  __syncthreads();
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long current_peak = 0;
  unsigned long long pending_peak = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < pending_count;
       i += blockDim.x * gridDim.x) {
    const int v = pending_in[i];
    const bool active = in_pending[v] != 0;
    const int b = active ? bucket_index(dist[v], delta) : kNoBucket;
    if constexpr (CollectTelemetry) {
      ++telemetry[kTelemetryPendingEntryExaminations];
      if (!active || b < selected_bucket || b >= kNoBucket) {
        ++telemetry[kTelemetryStalePendingEntryExaminations];
      }
    }
    bool append_current = false;
    const bool keep_pending = active && b > selected_bucket && b < kNoBucket;
    if (active && b == selected_bucket) {
      atomicExch(&in_pending[v], 0);
      if constexpr (UseCurrentGenerations) {
        append_current = atomicExch(&in_current[v], current_generation) !=
                         current_generation;
      } else {
        append_current = atomicCAS(&in_current[v], 0U, 1U) == 0U;
      }
    } else if (active && !keep_pending) {
      atomicExch(&in_pending[v], 0);
    }
    const int current_pos =
        append_position(append_current, current_count);
    const int pending_pos =
        append_position(keep_pending, new_pending_count);
    if (append_current) {
      current_queue[current_pos] = v;
      if constexpr (CollectTelemetry) {
        ++telemetry[kTelemetryCurrentQueueInsertions];
        const unsigned long long observed_peak =
            static_cast<unsigned long long>(current_pos) + 1;
        if (observed_peak > current_peak) current_peak = observed_peak;
      }
    }
    if (keep_pending) {
      pending_out[pending_pos] = v;
      if constexpr (CollectTelemetry) {
        const unsigned long long observed_peak =
            static_cast<unsigned long long>(pending_pos) + 1;
        if (observed_peak > pending_peak) pending_peak = observed_peak;
      }
    }
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, telemetry_counters);
    update_block_queue_peaks(current_peak, pending_peak, 0,
                             telemetry_counters);
  }
}

DeviceCsrOwner copy_host_csr_to_device(const HostCsrF32& h,
                                       hipStream_t stream,
                                       bool build_compact_edge_source,
                                       DeltaSteppingCsrOffsetMode offset_mode) {
  const bool uses_32_bit_offsets =
      delta_stepping_device_row_offset_width(h.nnz, offset_mode) ==
      DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit;
  const bool implicit_unit_weights = has_exact_unit_edge_values(h.values);
  const std::size_t rows = checked_size(h.rows, "rows");
  const std::size_t nnz = checked_size(h.nnz, "nnz");
  const std::size_t row_offset_count =
      sssp_capacity::checked_add(rows, 1);
  std::vector<std::uint32_t> compact_rowptr;

  std::size_t graph_device_bytes = 0;
  if (uses_32_bit_offsets) {
    add_device_allocation_bytes<CompactRowOffset>(
        row_offset_count, graph_device_bytes);
  } else {
    add_device_allocation_bytes<Offset>(row_offset_count,
                                        graph_device_bytes);
  }
  add_device_allocation_bytes<Index>(nnz, graph_device_bytes);
  if (!implicit_unit_weights) {
    add_device_allocation_bytes<float>(nnz, graph_device_bytes);
  }
  require_device_memory_headroom(
      graph_device_bytes, "Delta-Stepping graph upload");

  if (uses_32_bit_offsets) {
    compact_rowptr = delta_stepping_compact_row_offsets(h.rowptr);
  }
  DeviceBuffer<CompactRowOffset> next_rowptr32;
  DeviceBuffer<Offset> next_rowptr64;
  DeviceBuffer<Index> next_colind;
  DeviceBuffer<float> next_values;
  if (uses_32_bit_offsets) {
    next_rowptr32.reset(row_offset_count);
  } else {
    next_rowptr64.reset(row_offset_count);
  }
  next_colind.reset(nnz);
  if (!implicit_unit_weights) {
    next_values.reset(nnz);
  }

  DeviceCsrOwner d;
  d.rows = h.rows;
  d.cols = h.cols;
  d.nnz = h.nnz;
  d.uses_32_bit_offsets = uses_32_bit_offsets;
  d.implicit_unit_weights = implicit_unit_weights;
  if (implicit_unit_weights) {
    if (uses_32_bit_offsets) {
      d.unit_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<CompactRowOffset, false>(h.rows);
      d.unit_telemetry_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<CompactRowOffset, true>(h.rows);
    } else {
      d.unit_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<Offset, false>(h.rows);
      d.unit_telemetry_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<Offset, true>(h.rows);
    }
  }
  if (uses_32_bit_offsets) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        next_rowptr32.get(), compact_rowptr.data(),
        sssp_capacity::checked_bytes<CompactRowOffset>(row_offset_count),
        hipMemcpyHostToDevice, stream));
  } else {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        next_rowptr64.get(), h.rowptr.data(),
        sssp_capacity::checked_bytes<Offset>(row_offset_count),
        hipMemcpyHostToDevice, stream));
  }

  DeviceBuffer<std::uint32_t> next_edge_source;
  bool edge_source_available = false;
  if (build_compact_edge_source &&
      delta_stepping_compact_edge_ids_eligible(h.nnz)) {
    const std::size_t edge_source_bytes =
        sssp_capacity::checked_bytes<std::uint32_t>(nnz);
    const bool optional_storage_fits =
        edge_source_bytes == 0 ||
        device_memory_request_fits(edge_source_bytes);
    const bool allocation_attempted = optional_storage_fits;
    const hipError_t allocation_status =
        optional_storage_fits
            ? next_edge_source.try_reset(nnz)
            : hipErrorOutOfMemory;
    if (allocation_status == hipSuccess) {
      edge_source_available = true;
      if (nnz != 0) {
        // edge_source consumes rowptr in a later dispatch.  Explicit worker
        // streams on gfx1151 require a real completion boundary for dependent
        // dispatches.
        synchronize_explicit_stream(stream);
        if (uses_32_bit_offsets) {
          build_edge_source_kernel<CompactRowOffset>
              <<<grid_for_items(h.rows), kBlockSize, 0, stream>>>(
                  h.rows, next_rowptr32.get(), next_edge_source.get());
        } else {
          build_edge_source_kernel<Offset>
              <<<grid_for_items(h.rows), kBlockSize, 0, stream>>>(
                  h.rows, next_rowptr64.get(), next_edge_source.get());
        }
        DS_DELTA_HIP_CHECK(hipGetLastError());
      }
    } else if (allocation_status == hipErrorOutOfMemory) {
      // This allocation is optional. Clear the swallowed per-thread runtime
      // error so the next launch check does not turn the intended legacy
      // fallback into a delayed exception.
      if (allocation_attempted) {
        (void)hipGetLastError();
      }
    } else {
      DS_DELTA_HIP_CHECK(allocation_status);
    }
  }
  if (nnz != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        next_colind.get(), h.colind.data(),
        sssp_capacity::checked_bytes<Index>(nnz), hipMemcpyHostToDevice,
        stream));
    if (!implicit_unit_weights) {
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          next_values.get(), h.values.data(),
          sssp_capacity::checked_bytes<float>(nnz), hipMemcpyHostToDevice,
          stream));
    }
  }
  // The host vectors may be temporary, and PathFinder worker streams consume
  // the graph immediately after construction. Publish only a completed upload.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  if (uses_32_bit_offsets) {
    d.rowptr32.swap(next_rowptr32);
  } else {
    d.rowptr64.swap(next_rowptr64);
  }
  d.colind.swap(next_colind);
  if (!implicit_unit_weights) {
    d.values.swap(next_values);
  }
  if (edge_source_available) {
    d.edge_source.swap(next_edge_source);
  }
  d.edge_source_available = edge_source_available;
  return d;
}

template <typename RowOffset>
void validate_device_csr_contents(const DeviceCsrView<RowOffset>& g,
                                  hipStream_t stream) {
  DeviceBuffer<int> d_invalid(1);
  DS_DELTA_HIP_CHECK(hipMemsetAsync(d_invalid.get(), 0, sizeof(int), stream));
  synchronize_explicit_stream(stream);
  validate_device_csr_kernel<RowOffset>
      <<<grid_for_items(g.rows), kBlockSize, 0, stream>>>(
      g.rows, g.cols, g.nnz, g.rowptr, g.colind, g.values, d_invalid.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  if (copy_scalar_to_host(d_invalid.get(), stream) != 0) {
    throw std::invalid_argument(
        "invalid device CSR: rowptr/colind must be in range and weights finite nonnegative");
  }
}

void initialize_scratch_storage_once(DeltaSteppingScratch& scratch,
                                     Offset n,
                                     float inf,
                                     hipStream_t stream) {
  scratch.ensure_generic_storage();
  if (scratch.generic_initialized) return;
  initialize_delta_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, scratch.dist.get(), scratch.in_current.get(), scratch.in_pending.get(),
      scratch.in_heavy.get(), scratch.current_count.get(), scratch.next_count.get(),
      scratch.pending_count.get(), scratch.heavy_count.get(),
      scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.generic_initialized = true;
}

void initialize_parent_keys_once(DeltaSteppingScratch& scratch,
                                 Offset n,
                                 hipStream_t stream) {
  scratch.ensure_parent_key_storage();
  if (scratch.parent_key_initialized) return;
  initialize_parent_keys_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, scratch.parent_key.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.parent_key_initialized = true;
}

void prepare_device_telemetry(DeltaSteppingScratch& scratch,
                              hipStream_t stream) {
  scratch.ensure_telemetry_storage();
  DS_DELTA_HIP_CHECK(hipMemsetAsync(
      scratch.telemetry_counters.get(),
      0,
      sssp_capacity::checked_bytes<unsigned long long>(
          kTelemetryCounterCount),
      stream));
}

void copy_device_telemetry_to_host(DeltaSteppingScratch& scratch,
                                   DeltaSteppingCsrTelemetry& telemetry,
                                   hipStream_t stream) {
  std::array<unsigned long long, kTelemetryCounterCount> counters{};
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      counters.data(),
      scratch.telemetry_counters.get(),
      sssp_capacity::checked_bytes<unsigned long long>(counters.size()),
      hipMemcpyDeviceToHost,
      stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  telemetry.frontier_entries_processed =
      counters[kTelemetryFrontierEntries];
  telemetry.active_vertices_processed =
      counters[kTelemetryActiveVertices];
  telemetry.stale_frontier_entries =
      counters[kTelemetryStaleFrontierEntries];
  telemetry.light_edge_visits = counters[kTelemetryLightEdgeVisits];
  telemetry.heavy_edge_visits = counters[kTelemetryHeavyEdgeVisits];
  telemetry.distance_atomic_attempts =
      counters[kTelemetryDistanceAtomicAttempts];
  telemetry.successful_distance_relaxations =
      counters[kTelemetrySuccessfulRelaxations];
  telemetry.distance_cas_retries =
      counters[kTelemetryDistanceCasRetries];
  telemetry.current_queue_insertions =
      counters[kTelemetryCurrentQueueInsertions];
  telemetry.pending_queue_insertions =
      counters[kTelemetryPendingQueueInsertions];
  telemetry.heavy_queue_insertions =
      counters[kTelemetryHeavyQueueInsertions];
  telemetry.bucket_insertions = telemetry.current_queue_insertions +
                                telemetry.pending_queue_insertions;
  telemetry.pending_entry_examinations =
      counters[kTelemetryPendingEntryExaminations];
  telemetry.stale_pending_entry_examinations =
      counters[kTelemetryStalePendingEntryExaminations];
  telemetry.current_queue_high_water =
      counters[kTelemetryCurrentQueueHighWater];
  telemetry.pending_queue_high_water =
      counters[kTelemetryPendingQueueHighWater];
  telemetry.heavy_queue_high_water =
      counters[kTelemetryHeavyQueueHighWater];
}

void initialize_legacy_predecessors_once(DeltaSteppingScratch& scratch,
                                         Offset n,
                                         hipStream_t stream) {
  if (scratch.legacy_predecessors_initialized) return;
  scratch.ensure_legacy_predecessor_storage();
  initialize_legacy_predecessors_kernel
      <<<grid_for_items(n), kBlockSize, 0, stream>>>(
          n, scratch.pred_node.get(), scratch.pred_edge.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.legacy_predecessors_initialized = true;
}

void initialize_unit_scratch_storage_once(DeltaSteppingScratch& scratch,
                                          Offset n,
                                          float inf,
                                          hipStream_t stream) {
  if (scratch.unit_initialized) return;
  scratch.ensure_legacy_predecessor_storage();
  initialize_unit_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, scratch.dist.get(), scratch.in_pending.get(),
      scratch.pred_node.get(), scratch.pred_edge.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.unit_initialized = true;
  scratch.legacy_predecessors_initialized = true;
  // Only the first query has initialization work to publish. Reused explicit
  // workspaces must not pay an empty host synchronization on every route.
  synchronize_explicit_stream(stream);
}

void prepare_delta_scratch(DeltaSteppingScratch& scratch,
                           Offset n,
                           float inf,
                           hipStream_t stream) {
  scratch.ensure_generic_storage();
  if (!scratch.generic_initialized) {
    initialize_scratch_storage_once(scratch, n, inf, stream);
    return;
  }
  reset_int_zero_async(scratch.current_count.get(), stream);
  reset_int_zero_async(scratch.next_count.get(), stream);
  reset_int_zero_async(scratch.pending_count.get(), stream);
  reset_int_zero_async(scratch.new_pending_count.get(), stream);
  reset_int_zero_async(scratch.heavy_count.get(), stream);
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

template <bool UseCurrentGenerations>
std::uint32_t acquire_current_generation(DeltaSteppingScratch& scratch,
                                         hipStream_t stream) {
  if constexpr (!UseCurrentGenerations) {
    return 1U;
  }
  const DeltaSteppingGenerationAdvance advance =
      delta_stepping_advance_generation(scratch.current_generation);
  if (advance.reset_required) {
    DS_DELTA_HIP_CHECK(hipMemsetAsync(
        scratch.in_current.get(), 0,
        sssp_capacity::checked_bytes<std::uint32_t>(
            static_cast<std::size_t>(scratch.rows)),
        stream));
    // Token reuse is safe only after the old generations are fully cleared.
    synchronize_explicit_stream(stream);
  }
  scratch.current_generation = advance.token;
  return advance.token;
}

template <bool ResetCurrentMembership>
void reset_touched_vertices(DeltaSteppingScratch& scratch,
                            float inf,
                            hipStream_t stream,
                            int known_touched_count = -1) {
  PATHFINDER_PROFILE_RANGE("delta_step.legacy_reset");
  const int touched_count =
      known_touched_count >= 0
          ? known_touched_count
          : copy_scalar_to_host(scratch.touched_count.get(),
                                stream,
                                scratch.host_scalar.get(),
                                scratch.control_ready_event.get());
  if (touched_count < 0 || static_cast<Offset>(touched_count) > scratch.rows) {
    throw std::runtime_error("delta touched-vertex count is outside graph bounds");
  }
  if (touched_count > 0) {
    reset_touched_vertices_kernel<ResetCurrentMembership>
        <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
        scratch.touched_queue.get(), touched_count, inf, scratch.dist.get(),
        scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
        scratch.pred_node.get(), scratch.pred_edge.get(), scratch.parent_key.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

template <bool ResetCurrentMembership>
void reset_distance_only_touched_vertices(
    DeltaSteppingScratch& scratch,
    float inf,
    hipStream_t stream,
    int known_touched_count = -1) {
  PATHFINDER_PROFILE_RANGE("delta_step.distance_only_reset");
  const int touched_count =
      known_touched_count >= 0
          ? known_touched_count
          : copy_scalar_to_host(scratch.touched_count.get(), stream,
                                scratch.host_scalar.get(),
                                scratch.control_ready_event.get());
  if (touched_count < 0 ||
      static_cast<Offset>(touched_count) > scratch.rows) {
    throw std::runtime_error(
        "delta distance-only touched count is outside graph bounds");
  }
  if (touched_count > 0) {
    reset_distance_only_touched_vertices_kernel<ResetCurrentMembership>
        <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
            scratch.touched_queue.get(), touched_count, inf,
            scratch.dist.get(), scratch.in_current.get(),
            scratch.in_pending.get(), scratch.in_heavy.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

template <bool ResetCurrentMembership>
void reset_compact_parent_touched_vertices(DeltaSteppingScratch& scratch,
                                           float inf,
                                           hipStream_t stream,
                                           int touched_count,
                                           bool reset_heavy_membership) {
  PATHFINDER_PROFILE_RANGE("delta_step.compact_parent_reset");
  if (touched_count < 0) {
    throw std::logic_error(
        "compact-parent reset requires an already-known touched count");
  }
  if (static_cast<Offset>(touched_count) > scratch.rows) {
    throw std::runtime_error(
        "delta compact-parent touched count is outside graph bounds");
  }
  if (touched_count > 0) {
    if (reset_heavy_membership) {
      reset_compact_parent_touched_vertices_kernel<ResetCurrentMembership,
                                                   true>
          <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
              scratch.touched_queue.get(), touched_count, inf,
              scratch.dist.get(), scratch.in_current.get(),
              scratch.in_pending.get(), scratch.in_heavy.get(),
              scratch.parent_key.get());
    } else {
      reset_compact_parent_touched_vertices_kernel<ResetCurrentMembership,
                                                   false>
          <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
              scratch.touched_queue.get(), touched_count, inf,
              scratch.dist.get(), scratch.in_current.get(),
              scratch.in_pending.get(), scratch.in_heavy.get(),
              scratch.parent_key.get());
    }
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

std::vector<float> copy_dist_to_host(const float* d_dist, Offset n, hipStream_t stream) {
  std::vector<float> h(static_cast<std::size_t>(n));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(h.data(), d_dist,
                                    sssp_capacity::checked_bytes<float>(
                                        static_cast<std::size_t>(n)),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return h;
}

float copy_dist_value_to_host(const float* d_dist,
                              int vertex,
                              DeltaSteppingScratch& scratch,
                              hipStream_t stream) {
  return copy_scalar_to_host(d_dist + vertex, stream,
                             scratch.host_float_scalar.get(),
                             scratch.control_ready_event.get());
}

template <typename RowOffset>
void copy_predecessors_to_result(DeltaSteppingCsrResult& result,
                                 const DeviceCsrView<RowOffset>& graph,
                                 int* d_pred_node,
                                 Offset* d_pred_edge,
                                 hipStream_t stream) {
  result.pred_node.resize(static_cast<std::size_t>(graph.rows));
  result.pred_edge.resize(static_cast<std::size_t>(graph.rows));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_node.data(), d_pred_node,
                                    sssp_capacity::checked_bytes<int>(
                                        static_cast<std::size_t>(graph.rows)),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_edge.data(), d_pred_edge,
                                    sssp_capacity::checked_bytes<Offset>(
                                        static_cast<std::size_t>(graph.rows)),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename RowOffset, bool ImplicitUnitWeights>
int materialize_predecessors_from_keys(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    const float* vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("delta_step.predecessor_materialization");
  // Relaxation atomically keeps the predecessor associated with the smallest
  // winning distance. Materialize only touched vertices and recover the exact
  // original CSR edge from the winning predecessor's normally short row.
  const int touched_count = copy_scalar_to_host(
      scratch.touched_count.get(), stream, scratch.host_scalar.get(),
      scratch.control_ready_event.get());
  if (touched_count < 0 || static_cast<Offset>(touched_count) > graph.rows) {
    throw std::runtime_error(
        "delta predecessor touched count is outside graph bounds");
  }
  if (touched_count == 0) {
    return touched_count;
  }
  materialize_predecessors_kernel<RowOffset, ImplicitUnitWeights>
      <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
          scratch.touched_queue.get(), touched_count, graph.rowptr,
          graph.colind, graph.values, vertex_costs, scratch.dist.get(),
          scratch.parent_key.get(), scratch.pred_node.get(),
          scratch.pred_edge.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  synchronize_explicit_stream(stream);
  return touched_count;
}

enum class TargetPathParentMode {
  kLegacyPredecessor,
  kCompactEdge,
  kUnitWeight,
};

template <typename RowOffset,
          TargetPathParentMode ParentMode,
          bool ImplicitUnitWeights = false>
void extract_target_paths_to_result(
    DeltaSteppingCsrResult& result,
    DeltaSteppingScratch& scratch,
    const DeviceCsrView<RowOffset>& graph,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const std::vector<int>& targets,
    hipStream_t stream,
    const int* target_settled = nullptr,
    float exclusive_distance_limit =
        std::numeric_limits<float>::infinity()) {
  PATHFINDER_PROFILE_RANGE(
      ParentMode == TargetPathParentMode::kCompactEdge
          ? "delta_step.compact_edge_path_extraction"
          : "delta_step.legacy_path_extraction");
  const int target_count = static_cast<int>(targets.size());
  if constexpr (ParentMode == TargetPathParentMode::kUnitWeight) {
    measure_unit_target_paths_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.dist.get(),
            exclusive_distance_limit,
            scratch.target_distances.get(), scratch.target_path_lengths.get(),
            scratch.target_sources.get(), scratch.target_path_status.get());
  } else if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
    measure_edge_parent_target_paths_kernel<RowOffset, ImplicitUnitWeights>
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.rows, graph.nnz,
            graph.rowptr, graph.colind, graph.values, vertex_costs,
            edge_source, target_settled, scratch.dist.get(),
            scratch.parent_key.get(),
            scratch.target_distances.get(), scratch.target_path_lengths.get(),
            scratch.target_sources.get(), scratch.target_path_status.get());
  } else {
    measure_target_paths_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.rows,
            target_settled, scratch.dist.get(), scratch.pred_node.get(),
            scratch.target_distances.get(), scratch.target_path_lengths.get(),
            scratch.target_sources.get(), scratch.target_path_status.get());
  }
  DS_DELTA_HIP_CHECK(hipGetLastError());
  // On nonblocking gfx1151 worker streams, keep the measurement producer
  // behind a real completion boundary before the copy engine consumes its
  // status/length/distance tuple. Same-stream enqueue ordering alone has
  // exposed stale mixed-generation tuples on reused workspaces.
  synchronize_explicit_stream(stream);

  // Compute offsets on-device. Only this fixed-size totals record needs to
  // cross to the host before transactional output-buffer growth.
  const std::uint32_t extraction_epoch = scratch.begin_target_extraction();
  scan_target_path_offsets_kernel<<<1, 1, 0, stream>>>(
      target_count, scratch.target_distances.get(),
      scratch.target_path_lengths.get(), scratch.target_path_status.get(),
      extraction_epoch, scratch.target_node_offsets.get(),
      scratch.target_edge_offsets.get(), scratch.target_path_totals.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_path_totals.get(),
      scratch.target_path_totals.get(), sizeof(TargetPathTotals),
      hipMemcpyDeviceToHost, stream));
  wait_for_control_copy(scratch.control_ready_event.get(), stream);

  const TargetPathTotals totals = *scratch.host_target_path_totals.get();
  if (totals.extraction_epoch != extraction_epoch) {
    throw std::runtime_error(
        "delta target-offset scan returned a stale control record");
  }
  if (totals.status == kTargetOffsetScanOverflow) {
    throw std::overflow_error(
        "compact target paths are too large for int offsets");
  }
  if (totals.status != kTargetOffsetScanValid) {
    throw std::runtime_error(
        "delta predecessor path failed device validation during "
        "measurement");
  }
  if (totals.total_nodes < 0 || totals.total_edges < 0 ||
      totals.total_edges > totals.total_nodes) {
    throw std::runtime_error(
        "delta target-offset scan returned invalid compact-path totals");
  }
  const std::size_t total_nodes =
      static_cast<std::size_t>(totals.total_nodes);
  const std::size_t total_edges =
      static_cast<std::size_t>(totals.total_edges);
  scratch.ensure_compact_path_capacity(
      total_nodes, total_edges,
      ParentMode == TargetPathParentMode::kCompactEdge);

  result.target_distances.resize(targets.size());
  result.target_sources.resize(targets.size());
  const std::size_t target_offset_count =
      sssp_capacity::checked_target_offset_count(targets.size());
  result.target_path_offsets.resize(target_offset_count);
  result.target_edge_offsets.resize(target_offset_count);
  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
      fill_edge_parent_target_paths_kernel<RowOffset, ImplicitUnitWeights>
          <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
              scratch.targets.get(), target_count, scratch.rows, graph.nnz,
              graph.rowptr, graph.colind, graph.values, vertex_costs,
              edge_source, scratch.dist.get(), scratch.parent_key.get(),
              scratch.target_path_lengths.get(),
              scratch.target_path_status.get(), scratch.target_sources.get(),
              scratch.target_node_offsets.get(),
              scratch.target_edge_offsets.get(),
              scratch.compact_path_nodes.get(),
              scratch.compact_path_edges32.get());
    } else {
      fill_target_paths_kernel<RowOffset>
          <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
              scratch.targets.get(), target_count, scratch.rows,
              graph.rowptr, graph.colind,
              scratch.pred_node.get(), scratch.pred_edge.get(),
              scratch.target_path_lengths.get(),
              scratch.target_path_status.get(), scratch.target_sources.get(),
              scratch.target_node_offsets.get(),
              scratch.target_edge_offsets.get(),
              scratch.compact_path_nodes.get(),
              scratch.compact_path_edges64.get());
    }
    DS_DELTA_HIP_CHECK(hipGetLastError());
    // Path status is validated by the fill kernel and copied immediately
    // below. Publish the complete path and status together on explicit worker
    // streams before any D2H consumer starts.
    synchronize_explicit_stream(stream);
  }

  // Use persistent pinned mirrors and one completion event for the complete
  // host-visible output batch.
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_distances.get(), scratch.target_distances.get(),
      sssp_capacity::checked_bytes<float>(targets.size()),
      hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_path_lengths.get(),
      scratch.target_path_lengths.get(),
      sssp_capacity::checked_bytes<int>(targets.size()),
      hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_sources.get(), scratch.target_sources.get(),
      sssp_capacity::checked_bytes<int>(targets.size()),
      hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_path_status.get(),
      scratch.target_path_status.get(),
      sssp_capacity::checked_bytes<int>(targets.size()),
      hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_node_offsets.get(),
      scratch.target_node_offsets.get(),
      sssp_capacity::checked_bytes<int>(target_offset_count),
      hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      scratch.host_target_edge_offsets.get(),
      scratch.target_edge_offsets.get(),
      sssp_capacity::checked_bytes<int>(target_offset_count),
      hipMemcpyDeviceToHost, stream));
  if (total_nodes != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        scratch.host_compact_path_nodes.get(),
        scratch.compact_path_nodes.get(),
        sssp_capacity::checked_bytes<int>(total_nodes),
        hipMemcpyDeviceToHost, stream));
  }
  if (total_edges != 0) {
    if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          scratch.host_compact_path_edges32.get(),
          scratch.compact_path_edges32.get(),
          sssp_capacity::checked_bytes<std::uint32_t>(total_edges),
          hipMemcpyDeviceToHost, stream));
    } else {
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          scratch.host_compact_path_edges64.get(),
          scratch.compact_path_edges64.get(),
          sssp_capacity::checked_bytes<Offset>(total_edges),
          hipMemcpyDeviceToHost, stream));
    }
  }
  wait_for_control_copy(scratch.control_ready_event.get(), stream);

  std::copy_n(scratch.host_target_distances.get(), targets.size(),
              result.target_distances.begin());
  std::copy_n(scratch.host_target_sources.get(), targets.size(),
              result.target_sources.begin());
  std::copy_n(scratch.host_target_node_offsets.get(), target_offset_count,
              result.target_path_offsets.begin());
  std::copy_n(scratch.host_target_edge_offsets.get(), target_offset_count,
              result.target_edge_offsets.begin());
  if (result.target_path_offsets.front() != 0 ||
      result.target_edge_offsets.front() != 0 ||
      result.target_path_offsets.back() != totals.total_nodes ||
      result.target_edge_offsets.back() != totals.total_edges) {
    throw std::runtime_error(
        "delta compact target offsets do not match device totals");
  }
  bool all_targets_reached = true;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const int node_begin = result.target_path_offsets[i];
    const int node_end = result.target_path_offsets[i + 1];
    const int edge_begin = result.target_edge_offsets[i];
    const int edge_end = result.target_edge_offsets[i + 1];
    const int length = scratch.host_target_path_lengths.get()[i];
    const int status = scratch.host_target_path_status.get()[i];
    const float distance = result.target_distances[i];
    if (node_begin < 0 || node_end < node_begin ||
        edge_begin < 0 || edge_end < edge_begin) {
      throw std::runtime_error(
          "delta compact target offsets are not monotone");
    }
    const bool reached = std::isfinite(distance);
    if (reached &&
        (status == 0 || length <= 0 || node_end - node_begin != length ||
         edge_end - edge_begin != length - 1)) {
      throw std::runtime_error(
          "delta predecessor path failed device validation for target index " +
          std::to_string(i));
    }
    if (!reached &&
        (!std::isinf(distance) || status != 0 || length != 0 ||
         node_end != node_begin || edge_end != edge_begin)) {
      throw std::runtime_error(
          "delta unreachable target returned inconsistent path metadata for "
          "target index " +
          std::to_string(i));
    }
    all_targets_reached = all_targets_reached && reached;
  }

  // A failed fill can leave the unused portion of a persistent staging buffer
  // untouched. Validate status first, then read only fully published paths.
  if (total_nodes != 0) {
    std::copy_n(scratch.host_compact_path_nodes.get(), total_nodes,
                result.target_path_nodes.begin());
  }
  if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
    for (std::size_t edge_index = 0; edge_index < total_edges;
         ++edge_index) {
      const std::uint32_t edge =
          scratch.host_compact_path_edges32.get()[edge_index];
      if (static_cast<Offset>(edge) >= graph.nnz) {
        throw std::runtime_error(
            "delta compact path contains an out-of-range edge id");
      }
      result.target_path_edges[edge_index] = static_cast<Offset>(edge);
    }
  } else if (total_edges != 0) {
    std::copy_n(scratch.host_compact_path_edges64.get(), total_edges,
                result.target_path_edges.begin());
  }
  result.target_reached = all_targets_reached;
}

template <typename RowOffset, bool CollectTelemetry>
void launch_cooperative_unit_controller(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    int cooperative_launch_blocks,
    float delta,
    int target_count,
    int max_depth,
    int level_budget,
    hipStream_t stream) {
  const RowOffset* rowptr_arg = graph.rowptr;
  const Index* colind_arg = graph.colind;
  float delta_arg = delta;
  int target_count_arg = target_count;
  int max_depth_arg = max_depth;
  int level_budget_arg = level_budget;
  float* dist_arg = scratch.dist.get();
  int* pred_node_arg = scratch.pred_node.get();
  Offset* pred_edge_arg = scratch.pred_edge.get();
  int* frontier_queue_arg = scratch.current_queue.get();
  int* status_arg = scratch.unit_status.get();
  const int* target_multiplicity_arg = scratch.in_pending.get();
  unsigned long long* telemetry_counters_arg =
      CollectTelemetry ? scratch.telemetry_counters.get() : nullptr;
  void* kernel_args[] = {
      &rowptr_arg,
      &colind_arg,
      &delta_arg,
      &target_count_arg,
      &max_depth_arg,
      &level_budget_arg,
      &dist_arg,
      &pred_node_arg,
      &pred_edge_arg,
      &frontier_queue_arg,
      &status_arg,
      &target_multiplicity_arg,
      &telemetry_counters_arg,
  };

  DS_DELTA_HIP_CHECK(hipLaunchCooperativeKernel(
      cooperative_unit_frontier_controller_kernel<RowOffset,
                                                  CollectTelemetry>,
      dim3(static_cast<unsigned>(cooperative_launch_blocks)),
      dim3(kBlockSize),
      kernel_args,
      0,
      stream));
}

template <typename RowOffset, bool CollectTelemetry>
DeltaSteppingCsrResult run_unit_weight_specialization(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    float exclusive_distance_limit,
    int cooperative_launch_blocks,
    hipStream_t stream,
    DeltaSteppingCsrTelemetry* telemetry) {
  // With identical positive edge weights, delta-stepping and multi-source BFS
  // have the same shortest paths.  The routing converter emits exactly this
  // case, so use an append-only frontier and claim each vertex once.  This
  // removes bucket scans, light-closure bookkeeping, and the O(E) predecessor
  // rebuild while preserving original CSR edge IDs.
  std::vector<int> deduplicated_sources;
  std::unordered_set<int> source_set;
  const std::vector<int>* effective_sources = &sources;
  int initially_found = 0;
  const bool zero_distance_within_limit =
      !std::isfinite(exclusive_distance_limit) ||
      0.0f < exclusive_distance_limit;
  if (sources.size() == 1) {
    if (zero_distance_within_limit) {
      for (const int target : targets) {
        if (target == sources.front()) {
          ++initially_found;
        }
      }
    }
  } else {
    source_set.reserve(sources.size());
    deduplicated_sources.reserve(sources.size());
    for (const int source : sources) {
      if (source_set.insert(source).second) {
        deduplicated_sources.push_back(source);
      }
    }
    effective_sources = &deduplicated_sources;
    if (zero_distance_within_limit) {
      for (const int target : targets) {
        if (source_set.find(target) != source_set.end()) {
          ++initially_found;
        }
      }
    }
  }
  const auto is_effective_source = [&](int candidate) {
    return sources.size() == 1
               ? candidate == sources.front()
               : source_set.find(candidate) != source_set.end();
  };

  const int source_count = static_cast<int>(effective_sources->size());
  const int target_count = static_cast<int>(targets.size());
  const int vertex_count = static_cast<int>(graph.rows);
  int max_depth = vertex_count;
  if (std::isfinite(exclusive_distance_limit)) {
    if (!(exclusive_distance_limit > 0.0f)) {
      max_depth = 0;
    } else {
      const double largest_allowed_depth =
          std::ceil(static_cast<double>(exclusive_distance_limit)) - 1.0;
      max_depth = static_cast<int>(std::min<double>(
          static_cast<double>(vertex_count), largest_allowed_depth));
    }
  }
  const float inf = std::numeric_limits<float>::infinity();
  if constexpr (CollectTelemetry) {
    prepare_device_telemetry(scratch, stream);
  }
  scratch.ensure_source_capacity(effective_sources->size());
  scratch.ensure_target_capacity(targets.size());
  initialize_unit_scratch_storage_once(scratch, graph.rows, inf, stream);

  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(),
                                    effective_sources->data(),
                                    sssp_capacity::checked_bytes<int>(
                                        effective_sources->size()),
                                    hipMemcpyHostToDevice,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.targets.get(),
                                    targets.data(),
                                    sssp_capacity::checked_bytes<int>(
                                        targets.size()),
                                    hipMemcpyHostToDevice,
                                    stream));
  synchronize_explicit_stream(stream);

  int frontier_begin = 0;
  int frontier_end = source_count;
  int queue_tail = source_count;
  int current_count = source_count;
  int found_count = initially_found;
  int bucket = 0;
  // Source initialization alone is not a processed bucket.  This matters when
  // every requested target is already a source and traversal stops at depth 0.
  int bucket_rounds =
      initially_found >= target_count || max_depth == 0 ? 0 : 1;
  std::uint64_t controller_round_trips = 0;
  DeltaSteppingCsrResult result;
  result.target = -1;
  const bool use_cooperative_controller = cooperative_launch_blocks > 0;
  const bool use_batched_device_controller = stream == nullptr;

  try {
    // From the first target mark onward, any failure must restore all reusable
    // unit-search state. Upload failures above have not mutated device state.
    mark_unit_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.in_pending.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    initialize_unit_sources_kernel
        <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
            scratch.sources.get(), source_count, initially_found, target_count,
            max_depth, scratch.dist.get(), scratch.pred_node.get(),
            scratch.pred_edge.get(), scratch.current_queue.get(),
            scratch.unit_status.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    synchronize_explicit_stream(stream);

    while (current_count > 0 && found_count < target_count &&
           result.iterations_used < max_depth) {
    const int previous_queue_tail = queue_tail;
    const int previous_found_count = found_count;
    const int previous_frontier_end = frontier_end;
    const int previous_depth = result.iterations_used;

    if (use_cooperative_controller) {
      const int levels_budgeted =
          std::min(kUnitCooperativeLevelsPerLaunch,
                   max_depth - previous_depth);
      launch_cooperative_unit_controller<RowOffset, CollectTelemetry>(
          graph, scratch, cooperative_launch_blocks, delta, target_count,
          max_depth, levels_budgeted, stream);
      copy_unit_status_to_host(scratch, stream);
      ++controller_round_trips;

      queue_tail = scratch.host_unit_status.get()[kUnitStatusQueueTail];
      found_count = scratch.host_unit_status.get()[kUnitStatusFoundCount];
      frontier_begin =
          scratch.host_unit_status.get()[kUnitStatusFrontierBegin];
      frontier_end =
          scratch.host_unit_status.get()[kUnitStatusFrontierEnd];
      result.iterations_used =
          scratch.host_unit_status.get()[kUnitStatusCompletedDepth];
      bucket = scratch.host_unit_status.get()[kUnitStatusBucket];
      bucket_rounds =
          scratch.host_unit_status.get()[kUnitStatusBucketRounds];
      current_count = frontier_end - frontier_begin;
      const int expected_active =
          current_count > 0 && found_count < target_count &&
          result.iterations_used < max_depth;
      const int active = scratch.host_unit_status.get()[kUnitStatusActive];
      if (queue_tail < previous_queue_tail || queue_tail > vertex_count ||
          frontier_begin < previous_frontier_end ||
          frontier_end < frontier_begin || frontier_end != queue_tail ||
          found_count < previous_found_count || found_count > target_count ||
          active != expected_active ||
          result.iterations_used <= previous_depth ||
          result.iterations_used > previous_depth + levels_budgeted ||
          bucket < 0 || bucket >= kNoBucket || bucket_rounds < 1 ||
          bucket_rounds > result.iterations_used + 1) {
        std::ostringstream message;
        message << "delta unit-weight cooperative frontier state is inconsistent"
                << " (queue_tail=" << queue_tail
                << ", previous_queue_tail=" << previous_queue_tail
                << ", frontier_begin=" << frontier_begin
                << ", frontier_end=" << frontier_end
                << ", previous_frontier_end=" << previous_frontier_end
                << ", found_count=" << found_count
                << ", previous_found_count=" << previous_found_count
                << ", completed_depth=" << result.iterations_used
                << ", previous_depth=" << previous_depth
                << ", levels_budgeted=" << levels_budgeted
                << ", active=" << active
                << ", expected_active=" << expected_active
                << ", bucket=" << bucket
                << ", bucket_rounds=" << bucket_rounds
                << ", rows=" << vertex_count
                << ", sources=" << source_count
                << ", targets=" << target_count
                << ", max_depth=" << max_depth << ')';
        throw std::runtime_error(message.str());
      }
    } else if (!use_batched_device_controller) {
      // Explicit streams are used by parallel PathFinder workers.  Keep their
      // frontier bounds and depth on the host so there is no dependent
      // expansion -> controller-advance dispatch.  gfx1151 has repeatedly made
      // the expansion's queue updates visible without the following advance,
      // despite both kernels being submitted to the same stream.
      expand_unit_frontier_host_controlled_kernel<RowOffset,
                                                   CollectTelemetry>
          <<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
              frontier_begin, frontier_end, previous_depth + 1, graph.rowptr,
              graph.colind, scratch.dist.get(), scratch.pred_node.get(),
              scratch.pred_edge.get(), scratch.current_queue.get(),
              scratch.unit_status.get() + kUnitStatusQueueTail,
              scratch.unit_status.get() + kUnitStatusFoundCount,
              scratch.in_pending.get(),
              CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
      DS_DELTA_HIP_CHECK(hipGetLastError());
      copy_unit_status_to_host(scratch, stream);
      ++controller_round_trips;

      const int observed_queue_tail =
          scratch.host_unit_status.get()[kUnitStatusQueueTail];
      const int observed_found_count =
          scratch.host_unit_status.get()[kUnitStatusFoundCount];
      if (frontier_begin < 0 || frontier_end < frontier_begin ||
          frontier_end != previous_queue_tail ||
          current_count != frontier_end - frontier_begin ||
          observed_queue_tail < previous_queue_tail ||
          observed_queue_tail > vertex_count ||
          observed_found_count < previous_found_count ||
          observed_found_count > target_count) {
        std::ostringstream message;
        message << "delta unit-weight host-controlled frontier state is inconsistent"
                << " (queue_tail=" << observed_queue_tail
                << ", previous_queue_tail=" << previous_queue_tail
                << ", frontier_begin=" << frontier_begin
                << ", frontier_end=" << frontier_end
                << ", found_count=" << observed_found_count
                << ", previous_found_count=" << previous_found_count
                << ", completed_depth=" << previous_depth + 1
                << ", rows=" << vertex_count
                << ", sources=" << source_count
                << ", targets=" << target_count << ')';
        throw std::runtime_error(message.str());
      }

      queue_tail = observed_queue_tail;
      found_count = observed_found_count;
      frontier_begin = previous_frontier_end;
      frontier_end = queue_tail;
      result.iterations_used = previous_depth + 1;
      if (queue_tail > previous_queue_tail) {
        const int discovered_bucket =
            bucket_index_host(static_cast<float>(result.iterations_used), delta);
        if (discovered_bucket != bucket) {
          bucket = discovered_bucket;
          ++bucket_rounds;
        }
      }
    } else {
      // Preserve null-stream controller batching, which has not exhibited the
      // multi-stream dispatch failure and amortizes status transfers.
      const int rounds_to_enqueue =
          previous_depth == 0
              ? 1
              : std::min(4, max_depth - previous_depth);
      const int launch_blocks =
          rounds_to_enqueue == 1
              ? grid_for_frontier(current_count)
              : std::min(grid_for_items(graph.rows),
                         std::max(grid_for_frontier(current_count), 32));
      for (int round = 0; round < rounds_to_enqueue; ++round) {
        expand_unit_frontier_kernel<RowOffset, CollectTelemetry>
            <<<launch_blocks, kBlockSize, 0, stream>>>(
                graph.rowptr, graph.colind, scratch.dist.get(),
                scratch.pred_node.get(), scratch.pred_edge.get(),
                scratch.current_queue.get(), scratch.unit_status.get(),
                scratch.in_pending.get(),
                CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
        DS_DELTA_HIP_CHECK(hipGetLastError());
        advance_unit_frontier_kernel<<<1, 1, 0, stream>>>(
            scratch.unit_status.get(), delta, target_count, max_depth);
        DS_DELTA_HIP_CHECK(hipGetLastError());
      }
      copy_unit_status_to_host(scratch, stream);
      ++controller_round_trips;
      queue_tail = scratch.host_unit_status.get()[kUnitStatusQueueTail];
      found_count = scratch.host_unit_status.get()[kUnitStatusFoundCount];
      frontier_begin =
          scratch.host_unit_status.get()[kUnitStatusFrontierBegin];
      frontier_end =
          scratch.host_unit_status.get()[kUnitStatusFrontierEnd];
      result.iterations_used =
          scratch.host_unit_status.get()[kUnitStatusCompletedDepth];
      bucket = scratch.host_unit_status.get()[kUnitStatusBucket];
      bucket_rounds =
          scratch.host_unit_status.get()[kUnitStatusBucketRounds];
      const int expected_active =
          frontier_begin < frontier_end && found_count < target_count &&
          result.iterations_used < max_depth;
      const int active = scratch.host_unit_status.get()[kUnitStatusActive];
      if (queue_tail < previous_queue_tail || queue_tail > vertex_count ||
          frontier_begin < previous_frontier_end ||
          frontier_end < frontier_begin || frontier_end != queue_tail ||
          found_count < previous_found_count || found_count > target_count ||
          active != expected_active ||
          result.iterations_used <= previous_depth ||
          result.iterations_used > previous_depth + rounds_to_enqueue ||
          bucket < 0 || bucket >= kNoBucket || bucket_rounds < 1 ||
          bucket_rounds > result.iterations_used + 1) {
        std::ostringstream message;
        message << "delta unit-weight device frontier state is inconsistent"
                << " (queue_tail=" << queue_tail
                << ", previous_queue_tail=" << previous_queue_tail
                << ", frontier_begin=" << frontier_begin
                << ", frontier_end=" << frontier_end
                << ", previous_frontier_end=" << previous_frontier_end
                << ", found_count=" << found_count
                << ", previous_found_count=" << previous_found_count
                << ", completed_depth=" << result.iterations_used
                << ", previous_depth=" << previous_depth
                << ", rounds_enqueued=" << rounds_to_enqueue
                << ", active=" << active
                << ", expected_active=" << expected_active
                << ", bucket=" << bucket
                << ", bucket_rounds=" << bucket_rounds
                << ", rows=" << vertex_count
                << ", sources=" << source_count
                << ", targets=" << target_count << ')';
        throw std::runtime_error(message.str());
      }
    }
      current_count = frontier_end - frontier_begin;
    }
  } catch (...) {
    // A cooperative launch can be accepted after construction-time capability
    // checks and still fail because of transient residency or stream state.
    // At that point queue_tail may no longer be host-visible, so sparse cleanup
    // is unsafe. Densely restore every unit-search field and target mark before
    // preserving the original exception. If cleanup itself cannot run, force
    // the next query to perform the full one-time initialization again.
    const std::exception_ptr traversal_exception = std::current_exception();
    try {
      (void)hipStreamSynchronize(stream);
      (void)hipGetLastError();
      initialize_unit_arrays_kernel
          <<<grid_for_items(graph.rows), kBlockSize, 0, stream>>>(
              graph.rows, inf, scratch.dist.get(), scratch.in_pending.get(),
              scratch.pred_node.get(), scratch.pred_edge.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    } catch (...) {
      scratch.unit_initialized = false;
    }
    std::rethrow_exception(traversal_exception);
  }

  result.stopped_on_target = found_count >= target_count;
  result.stopped_on_distance_limit =
      std::isfinite(exclusive_distance_limit) &&
      !result.stopped_on_target;
  result.converged = !result.stopped_on_target &&
                     !result.stopped_on_distance_limit &&
                     current_count == 0;
  try {
    clear_unit_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.in_pending.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    extract_target_paths_to_result<RowOffset,
                                   TargetPathParentMode::kUnitWeight>(
        result, scratch, graph, nullptr, nullptr, targets, stream, nullptr,
        exclusive_distance_limit);
    for (std::size_t i = 0; i < result.target_distances.size(); ++i) {
      if (std::isfinite(result.target_distances[i]) &&
          !is_effective_source(result.target_sources[i])) {
        throw std::runtime_error(
            "delta unit-weight target path root is not a requested source "
            "for target index " +
            std::to_string(i));
      }
    }
  } catch (...) {
    const std::exception_ptr extraction_exception = std::current_exception();
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    clear_unit_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.in_pending.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    if (queue_tail > 0) {
      reset_unit_visited_kernel
          <<<grid_for_items(queue_tail), kBlockSize, 0, stream>>>(
              scratch.current_queue.get(),
              queue_tail,
              inf,
              scratch.dist.get(),
              scratch.pred_node.get(),
              scratch.pred_edge.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
    }
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    std::rethrow_exception(extraction_exception);
  }

  // Report distinct nonempty delta buckets rather than BFS expansion depth.
  // Tracking transitions on the device also handles float bucket collisions
  // and the shared terminal bucket after index saturation.
  const int completed_depth = result.iterations_used;
  result.iterations_used = bucket_rounds;

  if constexpr (CollectTelemetry) {
    copy_device_telemetry_to_host(scratch, *telemetry, stream);
    telemetry->outer_buckets_processed =
        static_cast<std::uint64_t>(bucket_rounds);
    telemetry->light_relaxation_rounds =
        static_cast<std::uint64_t>(completed_depth);
    telemetry->reached_vertices = static_cast<std::uint64_t>(queue_tail);
    telemetry->current_queue_high_water = std::max(
        telemetry->current_queue_high_water,
        static_cast<std::uint64_t>(source_count));
    telemetry->controller_round_trips = controller_round_trips;
    telemetry->gpu_resident_controller = use_cooperative_controller;
  }

  if (queue_tail > 0) {
    reset_unit_visited_kernel
        <<<grid_for_items(queue_tail), kBlockSize, 0, stream>>>(
            scratch.current_queue.get(),
            queue_tail,
            inf,
            scratch.dist.get(),
            scratch.pred_node.get(),
            scratch.pred_edge.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  // Runs reuse this sparse state immediately.  Do not let the next query's
  // source initialization race a queued reset on an explicit worker stream.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  if constexpr (CollectTelemetry) {
    telemetry->completed = true;
  }
  return result;
}

template <bool CollectTelemetry>
int find_min_pending_bucket(const int* d_pending_queue,
                            const int* d_pending_count,
                            int launch_blocks,
                            int current_bucket,
                            float delta,
                            const float* d_dist,
                            const int* d_in_pending,
                            int* d_min_bucket,
                            int* h_min_bucket,
                            hipEvent_t completion_event,
                            hipStream_t stream,
                            unsigned long long* telemetry_counters) {
  const int initial_min = kNoBucket;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(d_min_bucket,
                                    &initial_min,
                                    sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));
  synchronize_explicit_stream(stream);
  reduce_min_pending_bucket_kernel<CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
      d_pending_queue, d_pending_count, current_bucket, delta, d_dist,
      d_in_pending, d_min_bucket, telemetry_counters);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  return copy_scalar_to_host(d_min_bucket, stream, h_min_bucket,
                             completion_event);
}

int mark_and_count_settled_targets(DeltaSteppingScratch& scratch,
                                   int target_count,
                                   int current_bucket,
                                   float delta,
                                   hipStream_t stream) {
  mark_settled_targets_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(), target_count, current_bucket, delta,
      scratch.dist.get(), scratch.target_settled.get(),
      scratch.settled_target_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  return copy_scalar_to_host(scratch.settled_target_count.get(),
                             stream,
                             scratch.host_scalar.get(),
                             scratch.control_ready_event.get());
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
DeltaSteppingCsrResult run_delta_stepping_impl(
    const DeviceCsrView<RowOffset>& d_adjacency,
    const std::uint32_t* edge_source,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    bool skip_heavy_edges,
    float delta,
    int max_iters,
    float exclusive_distance_limit,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (max_iters < 0) max_iters = std::numeric_limits<int>::max();

  const Offset n = d_adjacency.rows;
  std::vector<int> deduplicated_sources;
  std::unordered_set<int> source_set;
  const std::vector<int>* effective_sources = &sources;
  if (sources.size() > 1) {
    source_set.reserve(sources.size());
    deduplicated_sources.reserve(sources.size());
    for (const int source : sources) {
      if (source_set.insert(source).second) {
        deduplicated_sources.push_back(source);
      }
    }
    effective_sources = &deduplicated_sources;
  }
  const auto is_effective_source = [&](int candidate) {
    return sources.size() == 1
               ? candidate == sources.front()
               : source_set.find(candidate) != source_set.end();
  };
  const int source_count = static_cast<int>(effective_sources->size());
  const bool use_target_set = targets != nullptr;
  const bool zero_distance_within_limit =
      !std::isfinite(exclusive_distance_limit) ||
      0.0f < exclusive_distance_limit;
  if constexpr (!TrackParents) {
    if (target >= 0 || use_target_set) {
      throw std::logic_error(
          "distance-only Delta-Stepping cannot materialize target paths");
    }
  }
  if constexpr (UseEdgeParent) {
    static_assert(TrackParents,
                  "edge-parent mode requires parent tracking");
    if (!use_target_set ||
        !delta_stepping_compact_edge_ids_eligible(d_adjacency.nnz) ||
        (d_adjacency.nnz != 0 && edge_source == nullptr)) {
      throw std::logic_error(
          "compact edge parents require eligible vector-target graph storage");
    }
  }
  const int target_count = use_target_set ? static_cast<int>(targets->size()) : 0;
  // A source target is settled before any bucket is processed, including when
  // max_iters is zero.  Keep this host storage alive for the duration of the
  // run because its copies are submitted asynchronously on the workspace
  // stream.
  std::vector<int> initial_target_settled;
  int initial_settled_target_count = 0;
  if (use_target_set) {
    initial_target_settled.assign(static_cast<std::size_t>(target_count), 0);
    for (int i = 0; i < target_count; ++i) {
      const int candidate = (*targets)[static_cast<std::size_t>(i)];
      if (zero_distance_within_limit && is_effective_source(candidate)) {
        initial_target_settled[static_cast<std::size_t>(i)] = 1;
        ++initial_settled_target_count;
      }
    }
  }
  const float inf = std::numeric_limits<float>::infinity();
  if constexpr (CollectTelemetry) {
    prepare_device_telemetry(scratch, stream);
  }
  const bool target_is_source =
      !use_target_set && target >= 0 && zero_distance_within_limit &&
      is_effective_source(target);

  scratch.ensure_source_capacity(effective_sources->size());
  // Generic state is lazy because exact-unit workers never need it. Allocate
  // and initialize its scalars before target setup touches settled_count.
  prepare_delta_scratch(scratch, n, inf, stream);
  if constexpr (TrackParents) {
    initialize_parent_keys_once(scratch, n, stream);
  }
  if constexpr (TrackParents && !UseEdgeParent) {
    initialize_legacy_predecessors_once(scratch, n, stream);
  }
  if (use_target_set) {
    scratch.ensure_target_capacity(static_cast<std::size_t>(target_count));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.targets.get(), targets->data(),
                                      sssp_capacity::checked_bytes<int>(
                                          static_cast<std::size_t>(
                                              target_count)),
                                      hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        scratch.target_settled.get(), initial_target_settled.data(),
        sssp_capacity::checked_bytes<int>(
            static_cast<std::size_t>(target_count)),
        hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        scratch.settled_target_count.get(), &initial_settled_target_count,
        sizeof(int), hipMemcpyHostToDevice, stream));
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(), effective_sources->data(),
                                    sssp_capacity::checked_bytes<int>(
                                        static_cast<std::size_t>(
                                            source_count)),
                                    hipMemcpyHostToDevice, stream));
  // The source kernel consumes the uploaded list and the state initialized or
  // reset above.  Complete both producers before launching it on an explicit
  // worker stream.
  synchronize_explicit_stream(stream);
  const std::uint32_t source_generation =
      acquire_current_generation<UseCurrentGenerations>(scratch, stream);
  initialize_delta_sources_kernel<TrackParents && !UseEdgeParent>
      <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
          scratch.sources.get(), source_count, scratch.dist.get(),
          scratch.in_current.get(), source_generation,
          scratch.current_queue.get(),
          scratch.current_count.get(), scratch.pred_node.get(),
          scratch.pred_edge.get(), scratch.touched_queue.get(),
          scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  synchronize_explicit_stream(stream);

  int* current_queue = scratch.current_queue.get();
  int* next_queue = scratch.next_queue.get();
  int* current_count_device = scratch.current_count.get();
  int* next_count_device = scratch.next_count.get();
  int* pending_queue = scratch.pending_a.get();
  int* pending_scratch = scratch.pending_b.get();
  int current_bucket = 0;
  const bool has_distance_limit =
      std::isfinite(exclusive_distance_limit);
  const int last_allowed_bucket =
      !has_distance_limit || !(exclusive_distance_limit > 0.0f)
          ? -1
          : bucket_index_host(
                std::nextafter(exclusive_distance_limit,
                               -std::numeric_limits<float>::infinity()),
                delta);
  // Sources are deduplicated on the host and initialization writes exactly
  // this many queue entries, so no device round trip is needed here.
  int current_count = source_count;
  std::uint64_t total_light_rounds = 0;
  std::uint64_t heavy_edge_phases = 0;
  std::uint64_t controller_round_trips = 0;
  // Device-counted heavy/pending kernels need a host-independent launch size.
  // Cap it to avoid pathological grids; tune 32--256 on the target GPU.
  const int device_count_blocks = std::min(grid_for_items(n), 256);
  DeltaSteppingCsrResult result;
  result.target = target;
  if (use_target_set && initial_settled_target_count == target_count) {
    // Every requested target is already a source.  This is a complete target
    // stop even when max_iters is zero; no bucket work is necessary.
    result.target_reached = true;
    result.stopped_on_target = true;
  }
  if (target_is_source) {
    result.target_distance = 0.0f;
    result.target_reached = true;
    result.stopped_on_target = true;
    try {
      if constexpr (TrackParents) {
        copy_predecessors_to_result<RowOffset>(
            result, d_adjacency, scratch.pred_node.get(),
            scratch.pred_edge.get(), stream);
      }
      result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
    } catch (...) {
      const std::exception_ptr materialization_exception =
          std::current_exception();
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      if constexpr (TrackParents) {
        reset_touched_vertices<!UseCurrentGenerations>(
            scratch, inf, stream, source_count);
      } else {
        reset_distance_only_touched_vertices<!UseCurrentGenerations>(
            scratch, inf, stream, source_count);
      }
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      std::rethrow_exception(materialization_exception);
    }
    if constexpr (CollectTelemetry) {
      copy_device_telemetry_to_host(scratch, *telemetry, stream);
      telemetry->reached_vertices =
          static_cast<std::uint64_t>(source_count);
      telemetry->current_queue_high_water =
          static_cast<std::uint64_t>(source_count);
    }
    if constexpr (TrackParents) {
      reset_touched_vertices<!UseCurrentGenerations>(scratch, inf, stream);
    } else {
      reset_distance_only_touched_vertices<!UseCurrentGenerations>(
          scratch, inf, stream);
    }
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    if constexpr (CollectTelemetry) {
      telemetry->completed = true;
    }
    return result;
  }

  auto report_progress = [&](const DeltaSteppingCsrProgress& progress) {
    try {
      progress_callback(progress, progress_user_data);
    } catch (...) {
      // A callback is allowed to abort a run, but it must not poison this
      // reusable workspace.  Capture the original exception before cleanup so
      // a successful reset preserves its exact dynamic type and payload.
      const std::exception_ptr callback_exception = std::current_exception();
      const int touched_count = copy_scalar_to_host(
          scratch.touched_count.get(), stream, scratch.host_scalar.get(),
          scratch.control_ready_event.get());
      if constexpr (UseEdgeParent) {
        // Exception cleanup is cold; clear heavy membership unconditionally so
        // it remains correct regardless of which callback site was reached.
        reset_compact_parent_touched_vertices<!UseCurrentGenerations>(
            scratch, inf, stream, touched_count, true);
      } else if constexpr (TrackParents) {
        reset_touched_vertices<!UseCurrentGenerations>(
            scratch, inf, stream, touched_count);
      } else {
        reset_distance_only_touched_vertices<!UseCurrentGenerations>(
            scratch, inf, stream, touched_count);
      }
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      std::rethrow_exception(callback_exception);
    }
  };

  for (int iter = 0;
       iter < max_iters && !result.stopped_on_target;
       ++iter) {
    if (has_distance_limit && current_bucket > last_allowed_bucket) {
      result.stopped_on_distance_limit = true;
      break;
    }
    reset_int_zero_async(scratch.heavy_count.get(), stream);

    int light_rounds = 0;
    while (current_count > 0) {
      // Check the first round immediately so shallow buckets pay no speculative
      // launch cost.  Only the default stream batches later device-counted
      // ping-pong rounds.  Every explicit stream retains a host-visible count
      // boundary between rounds after observed controller-state inconsistencies
      // when several nonblocking streams batch dependent kernels concurrently.
      const bool batch_rounds =
          stream == nullptr && progress_callback == nullptr &&
          light_rounds > 0;
      const int rounds_to_enqueue = batch_rounds ? 4 : 1;
      const int graph_blocks = grid_for_items(n);
      const int launch_blocks =
          batch_rounds
              ? std::min(graph_blocks,
                         std::max(grid_for_frontier(current_count), 32))
              : grid_for_frontier(current_count);

      for (int round = 0; round < rounds_to_enqueue; ++round) {
        reset_int_zero_async(next_count_device, stream);
        const std::uint32_t next_current_generation =
            acquire_current_generation<UseCurrentGenerations>(scratch,
                                                               stream);
        if constexpr (!UseCurrentGenerations) {
          clear_flags_from_queue_kernel
              <<<launch_blocks, kBlockSize, 0, stream>>>(
                  current_queue, current_count_device,
                  scratch.in_current.get());
          DS_DELTA_HIP_CHECK(hipGetLastError());
          // Preserve the known gfx1151 dependency boundary on the established
          // Boolean path.  The generation path has no clear producer.
          synchronize_explicit_stream(stream);
        }

        const bool terminal_bucket = current_bucket == kNoBucket - 1;
        if (terminal_bucket && vertex_costs != nullptr) {
          launch_relax_light_edges<RowOffset, ImplicitUnitWeights,
                                   UseCurrentGenerations, TrackParents,
                                   UseEdgeParent, true, false, true,
                                   CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, stream);
        } else if (terminal_bucket || skip_heavy_edges) {
          launch_relax_light_edges<RowOffset, ImplicitUnitWeights,
                                   UseCurrentGenerations, TrackParents,
                                   UseEdgeParent, false, false, true,
                                   CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, stream);
        } else if (vertex_costs != nullptr) {
          launch_relax_light_edges<RowOffset, ImplicitUnitWeights,
                                   UseCurrentGenerations, TrackParents,
                                   UseEdgeParent, true, true, false,
                                   CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, stream);
        } else {
          launch_relax_light_edges<RowOffset, ImplicitUnitWeights,
                                   UseCurrentGenerations, TrackParents,
                                   UseEdgeParent, false, true, false,
                                   CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, stream);
        }
        std::swap(current_queue, next_queue);
        std::swap(current_count_device, next_count_device);
        ++light_rounds;
        ++total_light_rounds;
      }
      current_count = copy_scalar_to_host(
          current_count_device, stream, scratch.host_scalar.get(),
          scratch.control_ready_event.get());
      ++controller_round_trips;
      if (current_count < 0 || static_cast<Offset>(current_count) > n) {
        throw std::runtime_error(
            "delta light-closure frontier count is outside graph bounds");
      }
    }

    if (!skip_heavy_edges && current_bucket != kNoBucket - 1) {
      ++heavy_edge_phases;
      if (vertex_costs != nullptr) {
        launch_relax_heavy_edges<RowOffset, ImplicitUnitWeights,
                                 TrackParents, UseEdgeParent, true,
                                 CollectTelemetry>(
            d_adjacency, scratch, vertex_costs, device_count_blocks,
            current_bucket, delta, exclusive_distance_limit, pending_queue,
            stream);
      } else {
        launch_relax_heavy_edges<RowOffset, ImplicitUnitWeights,
                                 TrackParents, UseEdgeParent, false,
                                 CollectTelemetry>(
            d_adjacency, scratch, nullptr, device_count_blocks,
            current_bucket, delta, exclusive_distance_limit, pending_queue,
            stream);
      }
      // Vector-target settlement is the only immediate device consumer here.
      // Scalar target copies synchronize on their D2H transfer, while the
      // minimum-bucket helper synchronizes its H2D initializer before reduce.
      if (use_target_set) {
        synchronize_explicit_stream(stream);
      }
    }

    result.iterations_used = iter + 1;
    if (use_target_set) {
      const int settled_count =
          mark_and_count_settled_targets(scratch, target_count, current_bucket,
                                         delta, stream);
      ++controller_round_trips;
      if (settled_count >= target_count) {
        result.target_reached = true;
        result.stopped_on_target = true;
        if (progress_callback) {
          DeltaSteppingCsrProgress progress;
          progress.iteration = result.iterations_used;
          progress.max_iters = max_iters;
          progress.convergence_checked = true;
          progress.changed = true;
          report_progress(progress);
        }
        break;
      }
    } else if (target >= 0) {
      const float target_distance =
          copy_dist_value_to_host(scratch.dist.get(), target, scratch,
                                  stream);
      ++controller_round_trips;
      const bool target_settled =
          std::isfinite(target_distance) &&
          bucket_index_host(target_distance, delta) <= current_bucket;
      if (target_settled) {
        result.target_distance = target_distance;
        result.target_reached = true;
        result.stopped_on_target = true;
        if (progress_callback) {
          DeltaSteppingCsrProgress progress;
          progress.iteration = result.iterations_used;
          progress.max_iters = max_iters;
          progress.convergence_checked = true;
          progress.changed = true;
          report_progress(progress);
        }
        break;
      }
    }

    const int next_bucket = find_min_pending_bucket<CollectTelemetry>(
        pending_queue, scratch.pending_count.get(), device_count_blocks,
        current_bucket, delta, scratch.dist.get(), scratch.in_pending.get(),
        scratch.min_pending_bucket.get(), scratch.host_scalar.get(),
        scratch.control_ready_event.get(), stream,
        CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
    ++controller_round_trips;
    const bool changed = (next_bucket != kNoBucket);
    if (changed &&
        (next_bucket <= current_bucket || next_bucket >= kNoBucket)) {
      throw std::runtime_error(
          "delta pending-bucket reduction returned an invalid successor");
    }
    if (progress_callback) {
      DeltaSteppingCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_iters;
      progress.convergence_checked = true;
      progress.changed = changed;
      report_progress(progress);
    }
    if (!changed) {
      if (has_distance_limit) {
        result.stopped_on_distance_limit = true;
      } else {
        result.converged = true;
      }
      break;
    }

    if (has_distance_limit && next_bucket > last_allowed_bucket) {
      result.stopped_on_distance_limit = true;
      break;
    }

    current_bucket = next_bucket;
    current_queue = scratch.current_queue.get();
    next_queue = scratch.next_queue.get();
    current_count_device = scratch.current_count.get();
    next_count_device = scratch.next_count.get();
    reset_int_zero_async(scratch.current_count.get(), stream);
    reset_int_zero_async(scratch.new_pending_count.get(), stream);
    synchronize_explicit_stream(stream);
    const std::uint32_t compacted_current_generation =
        acquire_current_generation<UseCurrentGenerations>(scratch, stream);

    compact_pending_to_current_bucket_kernel<UseCurrentGenerations,
                                             CollectTelemetry>
        <<<device_count_blocks, kBlockSize, 0, stream>>>(
        pending_queue, scratch.pending_count.get(), current_bucket, delta,
        scratch.dist.get(),
        scratch.in_pending.get(), scratch.in_current.get(),
        compacted_current_generation, current_queue,
        scratch.current_count.get(), pending_scratch,
        scratch.new_pending_count.get(),
        CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
    DS_DELTA_HIP_CHECK(hipGetLastError());

    current_count = copy_scalar_to_host(
        scratch.current_count.get(), stream, scratch.host_scalar.get(),
        scratch.control_ready_event.get());
    ++controller_round_trips;
    if (current_count < 0 || static_cast<Offset>(current_count) > n) {
      throw std::runtime_error(
          "delta compacted frontier count is outside graph bounds");
    }
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.pending_count.get(),
                                      scratch.new_pending_count.get(),
                                      sizeof(int),
                                      hipMemcpyDeviceToDevice,
                                      stream));
    // The next bucket appends to this count and consumes the compacted queue.
    synchronize_explicit_stream(stream);
    std::swap(pending_queue, pending_scratch);
  }

  int touched_count_for_reset = -1;
  try {
    const int* const settled_target_filter =
        use_target_set && !result.converged && !result.stopped_on_target
            ? scratch.target_settled.get()
            : nullptr;
    if constexpr (!TrackParents) {
      result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
    } else if constexpr (UseEdgeParent) {
      // Extraction already synchronizes for compact path sizes and output.
      // Stage the touched count before it so reduced reset adds no host
      // synchronization.
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.host_scalar.get(),
                                        scratch.touched_count.get(),
                                        sizeof(int),
                                        hipMemcpyDeviceToHost,
                                        stream));
      extract_target_paths_to_result<RowOffset,
                                     TargetPathParentMode::kCompactEdge,
                                     ImplicitUnitWeights>(
          result, scratch, d_adjacency, vertex_costs, edge_source, *targets,
          stream, settled_target_filter);
      touched_count_for_reset = *scratch.host_scalar.get();
    } else {
      if (use_target_set || target >= 0) {
        touched_count_for_reset =
            materialize_predecessors_from_keys<RowOffset,
                                               ImplicitUnitWeights>(
                d_adjacency, scratch, vertex_costs, stream);
      }
      if (use_target_set) {
        extract_target_paths_to_result<
            RowOffset, TargetPathParentMode::kLegacyPredecessor,
            ImplicitUnitWeights>(
            result, scratch, d_adjacency, vertex_costs, nullptr, *targets,
            stream, settled_target_filter);
      } else if (target >= 0) {
        copy_predecessors_to_result<RowOffset>(
            result, d_adjacency, scratch.pred_node.get(),
            scratch.pred_edge.get(), stream);
      }
    }
    if constexpr (TrackParents) {
      if (use_target_set) {
        for (std::size_t i = 0; i < result.target_distances.size(); ++i) {
          if (std::isfinite(result.target_distances[i]) &&
              !is_effective_source(result.target_sources[i])) {
            throw std::runtime_error(
                "delta target path root is not a requested source for target "
                "index " +
                std::to_string(i));
          }
        }
        result.target = -1;
      }
    }
    if constexpr (TrackParents) {
      if (!use_target_set && target >= 0) {
        result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
        result.target_distance =
            result.dist[static_cast<std::size_t>(target)];
        if (has_distance_limit &&
            !(result.target_distance < exclusive_distance_limit)) {
          result.target_distance = inf;
        }
        result.target_reached =
            !std::isinf(result.target_distance) &&
            (result.target_reached || result.converged);
      } else if (!use_target_set) {
        result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
      }
    }
    if (has_distance_limit && !use_target_set) {
      for (std::size_t node = 0; node < result.dist.size(); ++node) {
        if (result.dist[node] < exclusive_distance_limit) {
          continue;
        }
        result.dist[node] = inf;
        if (node < result.pred_node.size()) {
          result.pred_node[node] = -1;
        }
        if (node < result.pred_edge.size()) {
          result.pred_edge[node] = static_cast<Offset>(-1);
        }
      }
    }
  } catch (...) {
    // Counts are still trusted here: traversal completed successfully and the
    // failure arose only while materializing host-visible results.
    const std::exception_ptr extraction_exception = std::current_exception();
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    const int touched_count = copy_scalar_to_host(
        scratch.touched_count.get(), stream, scratch.host_scalar.get(),
        scratch.control_ready_event.get());
    if constexpr (UseEdgeParent) {
      reset_compact_parent_touched_vertices<!UseCurrentGenerations>(
          scratch, inf, stream, touched_count, true);
    } else if constexpr (TrackParents) {
      reset_touched_vertices<!UseCurrentGenerations>(
          scratch, inf, stream, touched_count);
    } else {
      reset_distance_only_touched_vertices<!UseCurrentGenerations>(
          scratch, inf, stream, touched_count);
    }
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    std::rethrow_exception(extraction_exception);
  }
  if constexpr (CollectTelemetry) {
    if (touched_count_for_reset < 0) {
      touched_count_for_reset = copy_scalar_to_host(
          scratch.touched_count.get(), stream, scratch.host_scalar.get(),
          scratch.control_ready_event.get());
    }
    copy_device_telemetry_to_host(scratch, *telemetry, stream);
    telemetry->outer_buckets_processed =
        static_cast<std::uint64_t>(result.iterations_used);
    telemetry->light_relaxation_rounds = total_light_rounds;
    telemetry->heavy_edge_phases = heavy_edge_phases;
    telemetry->reached_vertices =
        static_cast<std::uint64_t>(touched_count_for_reset);
    telemetry->current_queue_high_water = std::max(
        telemetry->current_queue_high_water,
        static_cast<std::uint64_t>(source_count));
    telemetry->controller_round_trips = controller_round_trips;
  }
  if constexpr (UseEdgeParent) {
    reset_compact_parent_touched_vertices<!UseCurrentGenerations>(
        scratch, inf, stream, touched_count_for_reset, !skip_heavy_edges);
  } else if constexpr (TrackParents) {
    reset_touched_vertices<!UseCurrentGenerations>(
        scratch, inf, stream, touched_count_for_reset);
  } else {
    reset_distance_only_touched_vertices<!UseCurrentGenerations>(
        scratch, inf, stream, touched_count_for_reset);
  }
  // Sparse cleanup is part of the run's completion contract.  Parallel
  // PathFinder workers immediately reuse this workspace for another query, so
  // returning with reset kernels still queued can race the next source setup.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  if constexpr (CollectTelemetry) {
    telemetry->completed = true;
  }
  return result;
}

template <typename RowOffset,
          bool ImplicitUnitWeights,
          bool TrackParents,
          bool UseEdgeParent>
DeltaSteppingCsrResult dispatch_delta_stepping_weight_mode(
    const DeviceCsrView<RowOffset>& d_adjacency,
    const std::uint32_t* edge_source,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    bool skip_heavy_edges,
    float delta,
    int max_iters,
    float exclusive_distance_limit,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data,
    DeltaSteppingCsrCurrentMembershipMode current_membership_mode,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (current_membership_mode ==
      DeltaSteppingCsrCurrentMembershipMode::kGeneration) {
    if (telemetry != nullptr) {
      return run_delta_stepping_impl<RowOffset, ImplicitUnitWeights, true,
                                     TrackParents, UseEdgeParent, true>(
          d_adjacency, edge_source, scratch, sources, target, targets,
          vertex_costs, skip_heavy_edges, delta, max_iters,
          exclusive_distance_limit, stream,
          progress_callback, progress_user_data, telemetry);
    }
    return run_delta_stepping_impl<RowOffset, ImplicitUnitWeights, true,
                                   TrackParents, UseEdgeParent, false>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        exclusive_distance_limit, stream,
        progress_callback, progress_user_data, nullptr);
  }
  if (current_membership_mode !=
      DeltaSteppingCsrCurrentMembershipMode::kBoolean) {
    throw std::invalid_argument(
        "unknown Delta-Stepping current-membership mode");
  }
  if (telemetry != nullptr) {
    return run_delta_stepping_impl<RowOffset, ImplicitUnitWeights, false,
                                   TrackParents, UseEdgeParent, true>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        exclusive_distance_limit, stream,
        progress_callback, progress_user_data, telemetry);
  }
  return run_delta_stepping_impl<RowOffset, ImplicitUnitWeights, false,
                                 TrackParents, UseEdgeParent, false>(
      d_adjacency, edge_source, scratch, sources, target, targets,
      vertex_costs, skip_heavy_edges, delta, max_iters,
      exclusive_distance_limit, stream,
      progress_callback, progress_user_data, nullptr);
}

template <typename RowOffset, bool TrackParents, bool UseEdgeParent>
DeltaSteppingCsrResult dispatch_delta_stepping_impl(
    const DeviceCsrView<RowOffset>& d_adjacency,
    const std::uint32_t* edge_source,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    bool skip_heavy_edges,
    float delta,
    int max_iters,
    float exclusive_distance_limit,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data,
    DeltaSteppingCsrCurrentMembershipMode current_membership_mode,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (d_adjacency.implicit_unit_weights) {
    return dispatch_delta_stepping_weight_mode<RowOffset, true, TrackParents,
                                               UseEdgeParent>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        exclusive_distance_limit, stream, progress_callback,
        progress_user_data, current_membership_mode, telemetry);
  }
  return dispatch_delta_stepping_weight_mode<RowOffset, false, TrackParents,
                                             UseEdgeParent>(
      d_adjacency, edge_source, scratch, sources, target, targets,
      vertex_costs, skip_heavy_edges, delta, max_iters,
      exclusive_distance_limit, stream, progress_callback,
      progress_user_data, current_membership_mode, telemetry);
}

void begin_telemetry_record(DeltaSteppingCsrTelemetry* telemetry,
                            DeltaSteppingCsrExecutionPath path,
                            float delta,
                            bool force_generic,
                            bool force_legacy_parent,
                            bool has_vertex_costs,
                            bool all_edges_light,
                            bool compact_parent_fallback) {
  if (telemetry == nullptr) return;
  telemetry->collected = true;
  telemetry->execution_path = path;
  telemetry->resolved_delta = delta;
  telemetry->wavefront_size = current_hip_wavefront_size();
  telemetry->force_generic = force_generic;
  telemetry->force_legacy_parent = force_legacy_parent;
  telemetry->has_vertex_costs = has_vertex_costs;
  telemetry->all_edges_light = all_edges_light;
  telemetry->compact_parent_fallback_events =
      compact_parent_fallback ? 1 : 0;
}

}  // namespace ds_delta_detail

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

struct DeltaSteppingCsrGraph::Impl {
  int device = 0;
  ds_delta_detail::DeviceCsrOwner adjacency;
  float max_edge_value = 0.0f;
  bool has_exact_unit_edge_values = false;
  bool path_capable = true;

  Impl(const HostCsrF32& host,
       hipStream_t stream,
       DeltaSteppingCsrStorageMode storage_mode,
       DeltaSteppingCsrOffsetMode offset_mode)
      : device(ds_delta_detail::current_hip_device()),
        adjacency(ds_delta_detail::copy_host_csr_to_device(
            host, stream,
            storage_mode == DeltaSteppingCsrStorageMode::kPathCapable,
            offset_mode)),
        max_edge_value(ds_delta_detail::max_edge_value(host.values)),
        has_exact_unit_edge_values(
            ds_delta_detail::has_exact_unit_edge_values(host.values)),
        path_capable(
            storage_mode == DeltaSteppingCsrStorageMode::kPathCapable) {}
};

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                                             hipStream_t stream)
    : DeltaSteppingCsrGraph(
          adjacency, stream, DeltaSteppingCsrStorageMode::kPathCapable,
          DeltaSteppingCsrOffsetMode::kAuto) {}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode)
    : DeltaSteppingCsrGraph(adjacency, stream, storage_mode,
                            DeltaSteppingCsrOffsetMode::kAuto) {}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrOffsetMode offset_mode)
    : DeltaSteppingCsrGraph(adjacency, stream,
                            DeltaSteppingCsrStorageMode::kPathCapable,
                            offset_mode) {}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrGraphOptions options)
    : DeltaSteppingCsrGraph(adjacency, stream, options.storage_mode,
                            options.offset_mode) {}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode,
    DeltaSteppingCsrOffsetMode offset_mode) {
  PATHFINDER_PROFILE_RANGE("delta_step.upload_graph");
  using namespace ds_delta_detail;
  validate_host_csr_arrays(adjacency);
  if (adjacency.rows <= 0 || adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR graph must be nonempty and square");
  }
  if (static_cast<unsigned long long>(adjacency.rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "frontier vertices are stored as int; rows must fit in int");
  }
  switch (offset_mode) {
    case DeltaSteppingCsrOffsetMode::kAuto:
    case DeltaSteppingCsrOffsetMode::kForce64Bit:
      break;
    default:
      throw std::invalid_argument("unknown Delta-Stepping offset mode");
  }
  impl_ = std::make_shared<Impl>(adjacency, stream, storage_mode, offset_mode);
}

DeltaSteppingCsrGraph::~DeltaSteppingCsrGraph() = default;
DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    DeltaSteppingCsrGraph&&) noexcept = default;
DeltaSteppingCsrGraph& DeltaSteppingCsrGraph::operator=(
    DeltaSteppingCsrGraph&&) noexcept = default;

bool DeltaSteppingCsrGraph::uses_32_bit_offsets() const noexcept {
  return impl_ && impl_->adjacency.uses_32_bit_offsets;
}

struct DeltaSteppingCsrWorkspace::Impl {
  std::shared_ptr<const DeltaSteppingCsrGraph::Impl> shared_graph;
  std::unique_ptr<ds_delta_detail::DeviceCsrOwner> owned_adjacency;
  ds_delta_detail::DeltaSteppingScratch scratch;
  ds_delta_detail::DeviceBuffer<float> vertex_costs;
  float max_edge_value = 0.0f;
  bool has_exact_unit_edge_values = false;
  bool has_vertex_costs = false;
  bool path_capable = true;
  hipStream_t stream = nullptr;
  int device = 0;

  Impl(const HostCsrF32& host,
       hipStream_t stream,
       DeltaSteppingCsrStorageMode storage_mode,
       DeltaSteppingCsrOffsetMode offset_mode)
      : owned_adjacency(std::make_unique<ds_delta_detail::DeviceCsrOwner>(
            ds_delta_detail::copy_host_csr_to_device(
                host, stream,
                storage_mode == DeltaSteppingCsrStorageMode::kPathCapable,
                offset_mode))),
        scratch(host.rows),
        max_edge_value(ds_delta_detail::max_edge_value(host.values)),
        has_exact_unit_edge_values(
            ds_delta_detail::has_exact_unit_edge_values(host.values)),
        path_capable(
            storage_mode == DeltaSteppingCsrStorageMode::kPathCapable),
        stream(stream),
        device(ds_delta_detail::current_hip_device()) {}

  static std::shared_ptr<const DeltaSteppingCsrGraph::Impl>
  require_shared_graph(
      const std::shared_ptr<const DeltaSteppingCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument(
          "delta-stepping shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const DeltaSteppingCsrGraph> graph,
       hipStream_t stream)
      : shared_graph(require_shared_graph(graph)),
        scratch(shared_graph->adjacency.rows),
        max_edge_value(shared_graph->max_edge_value),
        has_exact_unit_edge_values(shared_graph->has_exact_unit_edge_values),
        path_capable(shared_graph->path_capable),
        stream(stream),
        device(shared_graph->device) {
    if (ds_delta_detail::current_hip_device() != device) {
      throw std::invalid_argument(
          "delta-stepping shared graph belongs to a different HIP device");
    }
  }

  const ds_delta_detail::DeviceCsrOwner& adjacency() const {
    if (shared_graph) {
      return shared_graph->adjacency;
    }
    return *owned_adjacency;
  }

  ds_delta_detail::DeviceCsrOwner& mutable_adjacency() {
    if (!owned_adjacency) {
      throw std::logic_error(
          "update_values is unavailable for an immutable shared delta graph");
    }
    return *owned_adjacency;
  }

  void require_run_context(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "DeltaSteppingCsrWorkspace is stream-affine; use its construction stream");
    }
    if (ds_delta_detail::current_hip_device() != device) {
      throw std::invalid_argument(
          "DeltaSteppingCsrWorkspace is running on a different HIP device");
    }
  }
};

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                                                     hipStream_t stream)
    : DeltaSteppingCsrWorkspace(
          adjacency, stream, DeltaSteppingCsrStorageMode::kPathCapable,
          DeltaSteppingCsrOffsetMode::kAuto) {}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode)
    : DeltaSteppingCsrWorkspace(adjacency, stream, storage_mode,
                                DeltaSteppingCsrOffsetMode::kAuto) {}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrOffsetMode offset_mode)
    : DeltaSteppingCsrWorkspace(adjacency, stream,
                                DeltaSteppingCsrStorageMode::kPathCapable,
                                offset_mode) {}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode,
    DeltaSteppingCsrOffsetMode offset_mode) {
  using namespace ds_delta_detail;
  validate_host_csr_arrays(adjacency);
  if (adjacency.rows <= 0 || adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR graph must be nonempty and square");
  }
  if (static_cast<unsigned long long>(adjacency.rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("frontier vertices are stored as int; rows must fit in int");
  }
  switch (offset_mode) {
    case DeltaSteppingCsrOffsetMode::kAuto:
    case DeltaSteppingCsrOffsetMode::kForce64Bit:
      break;
    default:
      throw std::invalid_argument("unknown Delta-Stepping offset mode");
  }
  impl_ = std::make_unique<Impl>(adjacency, stream, storage_mode, offset_mode);
}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrWorkspaceOptions options)
    : DeltaSteppingCsrWorkspace(adjacency, stream) {
  parent_mode_ = options.parent_mode;
  execution_mode_ = options.execution_mode;
  current_membership_mode_ = options.current_membership_mode;
  impl_->scratch.reserve_query_capacity(options.capacity_hints,
                                        impl_->path_capable);
}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(std::move(adjacency), stream)) {}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
    hipStream_t stream,
    DeltaSteppingCsrWorkspaceOptions options)
    : DeltaSteppingCsrWorkspace(std::move(adjacency), stream) {
  parent_mode_ = options.parent_mode;
  execution_mode_ = options.execution_mode;
  current_membership_mode_ = options.current_membership_mode;
  impl_->scratch.reserve_query_capacity(options.capacity_hints,
                                        impl_->path_capable);
}

DeltaSteppingCsrWorkspace::~DeltaSteppingCsrWorkspace() = default;
DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    DeltaSteppingCsrWorkspace&&) noexcept = default;
DeltaSteppingCsrWorkspace& DeltaSteppingCsrWorkspace::operator=(
    DeltaSteppingCsrWorkspace&&) noexcept = default;

void DeltaSteppingCsrWorkspace::update_values(const std::vector<float>& values,
                                              hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("delta_step.update_edge_weights");
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_run_context(stream);
  DeviceCsrOwner& adjacency = impl_->mutable_adjacency();
  if (values.size() != static_cast<std::size_t>(adjacency.nnz)) {
    throw std::invalid_argument("updated CSR values size does not match workspace nnz");
  }
  for (const float value : values) {
    if (!std::isfinite(value) || value < 0.0f) {
      throw std::invalid_argument(
          "updated CSR values must be finite and nonnegative");
    }
  }
  const bool next_implicit_unit_weights =
      ds_delta_detail::has_exact_unit_edge_values(values);
  const float next_max_edge_value = max_edge_value(values);
  int next_unit_cooperative_launch_blocks =
      adjacency.unit_cooperative_launch_blocks;
  int next_unit_telemetry_cooperative_launch_blocks =
      adjacency.unit_telemetry_cooperative_launch_blocks;
  if (next_implicit_unit_weights && !adjacency.implicit_unit_weights) {
    if (adjacency.uses_32_bit_offsets) {
      next_unit_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<CompactRowOffset, false>(
              adjacency.rows);
      next_unit_telemetry_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<CompactRowOffset, true>(
              adjacency.rows);
    } else {
      next_unit_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<Offset, false>(adjacency.rows);
      next_unit_telemetry_cooperative_launch_blocks =
          unit_cooperative_controller_blocks<Offset, true>(adjacency.rows);
    }
  }
  if (adjacency.nnz == 0) {
    adjacency.implicit_unit_weights = true;
    adjacency.unit_cooperative_launch_blocks =
        next_unit_cooperative_launch_blocks;
    adjacency.unit_telemetry_cooperative_launch_blocks =
        next_unit_telemetry_cooperative_launch_blocks;
    impl_->has_exact_unit_edge_values = true;
    impl_->max_edge_value = next_max_edge_value;
    return;
  }
  if (!next_implicit_unit_weights) {
    // Build the replacement before publishing any semantic state. Allocation,
    // upload, or synchronization failure therefore leaves the previous graph
    // representation usable by the next run.
    require_device_memory_headroom(
        sssp_capacity::checked_bytes<float>(values.size()),
        "Delta-Stepping edge-value update");
    DeviceBuffer<float> replacement(values.size());
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        replacement.get(), values.data(),
        sssp_capacity::checked_bytes<float>(values.size()),
        hipMemcpyHostToDevice, stream));
    // The caller retains ownership of values and may have passed a temporary.
    // Complete the upload before that storage can be destroyed or reused.
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    adjacency.values.swap(replacement);
  }
  // A weighted-to-unit transition deliberately retains the previous explicit
  // allocation as high-water storage. Kernels select the compile-time unit
  // path from this flag and never dereference those stale values.
  adjacency.implicit_unit_weights = next_implicit_unit_weights;
  adjacency.unit_cooperative_launch_blocks =
      next_unit_cooperative_launch_blocks;
  adjacency.unit_telemetry_cooperative_launch_blocks =
      next_unit_telemetry_cooperative_launch_blocks;
  impl_->has_exact_unit_edge_values = next_implicit_unit_weights;
  impl_->max_edge_value = next_max_edge_value;
}

void DeltaSteppingCsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("delta_step.update_vertex_costs");
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_run_context(stream);
  const DeviceCsrOwner& adjacency = impl_->adjacency();
  if (vertex_costs.size() != static_cast<std::size_t>(adjacency.rows)) {
    throw std::invalid_argument("vertex cost size does not match workspace rows");
  }
  for (const float cost : vertex_costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument("vertex costs must be finite nonnegative values");
    }
  }
  require_device_memory_headroom(
      sssp_capacity::checked_bytes<float>(vertex_costs.size()),
      "Delta-Stepping vertex-cost update");
  DeviceBuffer<float> replacement(vertex_costs.size());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      replacement.get(), vertex_costs.data(),
      sssp_capacity::checked_bytes<float>(vertex_costs.size()),
      hipMemcpyHostToDevice, stream));
  // Match update_values(): this API does not require callers to keep the host
  // vector alive after it returns, and a failed upload must retain the
  // previously installed costs.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  impl_->vertex_costs.swap(replacement);
  impl_->has_vertex_costs = true;
}

DeltaSteppingCsrAllocationState
DeltaSteppingCsrWorkspace::allocation_state() const noexcept {
  DeltaSteppingCsrAllocationState state;
  if (!impl_) return state;
  const auto& scratch = impl_->scratch;
  const auto& adjacency = impl_->adjacency();
  state.edge_source = adjacency.edge_source.size() != 0;
  state.edge_values = adjacency.values.size() != 0;
  state.implicit_unit_weights = adjacency.implicit_unit_weights;
  state.parent_key = scratch.parent_key.size() != 0;
  state.predecessor_nodes = scratch.pred_node.size() != 0;
  state.predecessor_edges = scratch.pred_edge.size() != 0;
  state.target_storage = scratch.targets.size() != 0 ||
                         scratch.target_settled.size() != 0 ||
                         scratch.target_distances.size() != 0 ||
                         scratch.target_path_lengths.size() != 0 ||
                         scratch.target_sources.size() != 0 ||
                         scratch.target_path_status.size() != 0 ||
                         scratch.target_node_offsets.size() != 0 ||
                         scratch.target_edge_offsets.size() != 0;
  state.path_nodes = scratch.compact_path_nodes.size() != 0;
  state.path_edges = scratch.compact_path_edges32.size() != 0 ||
                     scratch.compact_path_edges64.size() != 0;
  state.compact_path_edges_32_bit =
      scratch.compact_path_edges32.size() != 0;
  state.telemetry_counters = scratch.telemetry_counters.size() != 0;
  return state;
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run_distances(
    const std::vector<int>& sources,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_run_context(stream);
  if (parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy) {
    throw std::invalid_argument(
        "run_distances is incompatible with forced legacy parent mode");
  }
  const DeviceCsrOwner& adjacency = impl_->adjacency();
  // Distances-only execution drops parent representations but preserves the
  // target/path high-water buffers of a path-capable workspace.
  impl_->scratch.release_parent_storage();
  // Strict distances-only storage never allocates or retains path state.
  if (!impl_->path_capable) {
    impl_->scratch.release_parent_and_path_storage();
  }
  PATHFINDER_PROFILE_RANGE("delta_step.generic_distances_only");
  const bool skip_heavy_edges =
      !impl_->has_vertex_costs && impl_->max_edge_value <= delta;
  begin_telemetry_record(
      active_telemetry_,
      DeltaSteppingCsrExecutionPath::kGenericDistancesOnly,
      delta,
      execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
      false,
      impl_->has_vertex_costs,
      skip_heavy_edges,
      false);
  const auto run_typed = [&](const auto& graph) {
    using RowOffset = typename std::remove_cv<typename std::remove_pointer<
        decltype(graph.rowptr)>::type>::type;
    validate_device_csr_shape(graph, sources, -1, delta);
    return dispatch_delta_stepping_impl<RowOffset, false, false>(
        graph, nullptr, impl_->scratch, sources, -1, nullptr,
        impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
        skip_heavy_edges, delta, max_iters, active_distance_limit_, stream,
        progress_callback, progress_user_data, current_membership_mode_,
        active_telemetry_);
  };
  if (adjacency.uses_32_bit_offsets) {
    return run_typed(adjacency.view<CompactRowOffset>());
  }
  return run_typed(adjacency.view<Offset>());
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_run_context(stream);
  if (!impl_->path_capable) {
    throw std::invalid_argument(
        "distances-only Delta-Stepping storage supports only run_distances");
  }
  const DeviceCsrOwner& adjacency = impl_->adjacency();
  PATHFINDER_PROFILE_RANGE("delta_step.generic");
  const bool skip_heavy_edges =
      !impl_->has_vertex_costs && impl_->max_edge_value <= delta;
  begin_telemetry_record(
      active_telemetry_, DeltaSteppingCsrExecutionPath::kLegacyGeneric,
      delta,
      execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
      parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
      impl_->has_vertex_costs, skip_heavy_edges, false);
  const auto run_typed = [&](const auto& graph) {
    using RowOffset = typename std::remove_cv<typename std::remove_pointer<
        decltype(graph.rowptr)>::type>::type;
    validate_device_csr_shape(graph, sources, target, delta);
    return dispatch_delta_stepping_impl<RowOffset, true, false>(
        graph, nullptr, impl_->scratch, sources, target, nullptr,
        impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
        skip_heavy_edges, delta, max_iters, active_distance_limit_, stream,
        progress_callback, progress_user_data, current_membership_mode_,
        active_telemetry_);
  };
  if (adjacency.uses_32_bit_offsets) {
    return run_typed(adjacency.view<CompactRowOffset>());
  }
  return run_typed(adjacency.view<Offset>());
}

DeltaSteppingCsrResult DeltaSteppingCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_run_context(stream);
  if (!impl_->path_capable) {
    throw std::invalid_argument(
        "distances-only Delta-Stepping storage supports only run_distances");
  }
  const DeviceCsrOwner& adjacency = impl_->adjacency();
  validate_target_list_common_shape(adjacency.rows, targets);
  const StableTargetDeduplication target_deduplication =
      stable_deduplicate_targets(targets);
  const std::vector<int>& traversal_targets =
      target_deduplication.has_duplicates()
          ? target_deduplication.unique_targets
          : targets;
  const auto run_typed = [&](const auto& graph) {
    using RowOffset = typename std::remove_cv<typename std::remove_pointer<
        decltype(graph.rowptr)>::type>::type;
    validate_device_csr_shape(graph, sources, -1, delta);
    if (execution_mode_ == DeltaSteppingCsrExecutionMode::kAutomatic &&
        parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
        impl_->has_exact_unit_edge_values &&
        !impl_->has_vertex_costs &&
        graph.rows <= kMaxUnitSpecializationRows &&
        max_iters < 0 && progress_callback == nullptr) {
      PATHFINDER_PROFILE_RANGE("delta_step.unit_specialization");
      begin_telemetry_record(
          active_telemetry_, DeltaSteppingCsrExecutionPath::kExactUnit,
          delta, false, false, false,
          impl_->max_edge_value <= delta, false);
      if (active_telemetry_ != nullptr) {
        return run_unit_weight_specialization<RowOffset, true>(
            graph, impl_->scratch, sources, traversal_targets, delta,
            active_distance_limit_,
            adjacency.unit_telemetry_cooperative_launch_blocks,
            stream, active_telemetry_);
      }
      return run_unit_weight_specialization<RowOffset, false>(
          graph, impl_->scratch, sources, traversal_targets, delta,
          active_distance_limit_, adjacency.unit_cooperative_launch_blocks,
          stream, nullptr);
    }
    PATHFINDER_PROFILE_RANGE("delta_step.generic");
    const float* const vertex_costs =
        impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr;
    const bool skip_heavy_edges =
        !impl_->has_vertex_costs && impl_->max_edge_value <= delta;
    if (parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
        adjacency.edge_source_available) {
      begin_telemetry_record(
          active_telemetry_, DeltaSteppingCsrExecutionPath::kCompactGeneric,
          delta,
          execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
          false, impl_->has_vertex_costs, skip_heavy_edges, false);
      return dispatch_delta_stepping_impl<RowOffset, true, true>(
          graph, adjacency.edge_source.get(), impl_->scratch, sources,
          -1, &traversal_targets, vertex_costs, skip_heavy_edges, delta,
          max_iters,
          active_distance_limit_, stream, progress_callback,
          progress_user_data, current_membership_mode_, active_telemetry_);
    }
    const bool compact_parent_fallback =
        parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
        !adjacency.edge_source_available;
    begin_telemetry_record(
        active_telemetry_, DeltaSteppingCsrExecutionPath::kLegacyGeneric,
        delta,
        execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
        parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
        impl_->has_vertex_costs, skip_heavy_edges, compact_parent_fallback);
    return dispatch_delta_stepping_impl<RowOffset, true, false>(
        graph, nullptr, impl_->scratch, sources, -1, &traversal_targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        active_distance_limit_, stream, progress_callback,
        progress_user_data, current_membership_mode_, active_telemetry_);
  };
  DeltaSteppingCsrResult result;
  if (adjacency.uses_32_bit_offsets) {
    result = run_typed(adjacency.view<CompactRowOffset>());
  } else {
    result = run_typed(adjacency.view<Offset>());
  }
  return fan_out_deduplicated_target_result(
      std::move(result), target_deduplication);
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
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_detail;

  const DeviceCsrView<Offset> graph = wide_device_csr_view(d_adjacency);
  validate_device_csr_shape(graph, sources, target, delta);
  validate_device_csr_contents(graph, stream);
  DeltaSteppingScratch scratch(d_adjacency.rows);
  return dispatch_delta_stepping_impl<Offset, true, false>(
      graph, nullptr, scratch, sources, target, nullptr, nullptr, false, delta,
      max_iters, std::numeric_limits<float>::infinity(), stream,
      progress_callback, progress_user_data,
      DeltaSteppingCsrCurrentMembershipMode::kBoolean, nullptr);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_minplus_hip_csr(d_adjacency,
                                        std::vector<int>{source},
                                        target,
                                        delta,
                                        max_iters,
                                        stream,
                                        progress_callback,
                                        progress_user_data);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, -1, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, delta, -1, stream, nullptr, nullptr);
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
  using namespace ds_delta_detail;
  validate_host_csr(adjacency, source, target, delta);
  DeviceCsrOwner d_adjacency =
      copy_host_csr_to_device(adjacency, stream, false,
                              DeltaSteppingCsrOffsetMode::kAuto);
  DeltaSteppingScratch scratch(adjacency.rows);
  const std::vector<int> sources{source};
  if (d_adjacency.uses_32_bit_offsets) {
    return dispatch_delta_stepping_impl<CompactRowOffset, true, false>(
        d_adjacency.view<CompactRowOffset>(), nullptr, scratch, sources,
        target, nullptr, nullptr, false, delta, max_iters,
        std::numeric_limits<float>::infinity(), stream, progress_callback,
        progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
        nullptr);
  }
  return dispatch_delta_stepping_impl<Offset, true, false>(
      d_adjacency.view<Offset>(), nullptr, scratch, sources, target, nullptr,
      nullptr, false, delta, max_iters,
      std::numeric_limits<float>::infinity(), stream, progress_callback,
      progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
      nullptr);
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
  using namespace ds_delta_detail;
  validate_host_csr(adjacency, sources, target, delta);
  DeviceCsrOwner d_adjacency =
      copy_host_csr_to_device(adjacency, stream, false,
                              DeltaSteppingCsrOffsetMode::kAuto);
  DeltaSteppingScratch scratch(adjacency.rows);
  if (d_adjacency.uses_32_bit_offsets) {
    return dispatch_delta_stepping_impl<CompactRowOffset, true, false>(
        d_adjacency.view<CompactRowOffset>(), nullptr, scratch, sources,
        target, nullptr, nullptr, false, delta, max_iters,
        std::numeric_limits<float>::infinity(), stream, progress_callback,
        progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
        nullptr);
  }
  return dispatch_delta_stepping_impl<Offset, true, false>(
      d_adjacency.view<Offset>(), nullptr, scratch, sources, target, nullptr,
      nullptr, false, delta, max_iters,
      std::numeric_limits<float>::infinity(), stream, progress_callback,
      progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
      nullptr);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_minplus_hip_csr(adjacency, source, -1, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream) {
  return delta_stepping_minplus_hip_csr(adjacency, source, delta, -1, stream, nullptr, nullptr);
}
