#include "near_far.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> g_near_far_force_generic{false};
std::atomic<bool> g_near_far_force_epoch_wrap{false};
std::atomic<std::uint64_t> g_near_far_status_copy_count{0};
std::atomic<std::uint64_t> g_near_far_shard_count_copy_count{0};
std::atomic<std::uint64_t> g_near_far_unit_controller_count{0};
std::atomic<std::uint64_t> g_near_far_controller_fallback_count{0};
std::atomic<std::uint64_t> g_near_far_path_transfer_count{0};
std::atomic<std::uint64_t> g_near_far_target_growth_count{0};
std::atomic<std::uint64_t> g_near_far_path_growth_count{0};
std::atomic<std::uint64_t> g_near_far_device_allocation_count{0};
std::atomic<std::uint64_t> g_near_far_pinned_allocation_count{0};

}  // namespace

extern "C" void near_far_internal_reset_optimization_counters() {
  g_near_far_status_copy_count.store(0, std::memory_order_relaxed);
  g_near_far_shard_count_copy_count.store(0, std::memory_order_relaxed);
  g_near_far_unit_controller_count.store(0, std::memory_order_relaxed);
  g_near_far_controller_fallback_count.store(0, std::memory_order_relaxed);
  g_near_far_path_transfer_count.store(0, std::memory_order_relaxed);
  g_near_far_target_growth_count.store(0, std::memory_order_relaxed);
  g_near_far_path_growth_count.store(0, std::memory_order_relaxed);
  g_near_far_device_allocation_count.store(0, std::memory_order_relaxed);
  g_near_far_pinned_allocation_count.store(0, std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_status_copy_count() {
  return g_near_far_status_copy_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_shard_count_copy_count() {
  return g_near_far_shard_count_copy_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_unit_controller_count() {
  return g_near_far_unit_controller_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_controller_fallback_count() {
  return g_near_far_controller_fallback_count.load(
      std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_path_transfer_count() {
  return g_near_far_path_transfer_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_target_growth_count() {
  return g_near_far_target_growth_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_path_growth_count() {
  return g_near_far_path_growth_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_device_allocation_count() {
  return g_near_far_device_allocation_count.load(
      std::memory_order_relaxed);
}

extern "C" std::uint64_t near_far_internal_pinned_allocation_count() {
  return g_near_far_pinned_allocation_count.load(
      std::memory_order_relaxed);
}

extern "C" void near_far_internal_force_generic(int enabled) {
  g_near_far_force_generic.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" void near_far_internal_force_epoch_wrap(int enabled) {
  g_near_far_force_epoch_wrap.store(enabled != 0,
                                    std::memory_order_relaxed);
}

namespace near_far_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;
using CompactOffset = std::uint32_t;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kMaxBlocksPerShard = 32;
constexpr int kCurrentQueue = 0;
constexpr int kNextQueue = 1;
constexpr int kNearFarQueue = 2;
constexpr int kFarQueue = 3;
constexpr int kScratchQueue = 4;
constexpr int kQueueCount = 5;
constexpr unsigned int kQueueMask =
    (1U << static_cast<unsigned int>(kQueueCount)) - 1U;
constexpr unsigned int kEpochShift = kQueueCount;
constexpr unsigned int kMaxRunEpoch =
    std::numeric_limits<unsigned int>::max() >> kEpochShift;
constexpr CompactOffset kNoCompactEdge =
    std::numeric_limits<CompactOffset>::max();
constexpr int kDefaultSubpartitions = 16;
constexpr int kUrgentWavesPerComputeUnit = 32;
constexpr int kTargetPathValid = 1;

enum DeviceError : int {
  kNoDeviceError = 0,
  kDeferredQueueOverflow = 1,
  kVersionOverflow = 2,
  kInvalidVertexState = 3,
};

inline void hip_check(hipError_t status,
                      const char* expression,
                      const char* file,
                      int line) {
  if (status != hipSuccess) {
    std::ostringstream message;
    message << "HIP error at " << file << ':' << line << " for "
            << expression << ": " << hipGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

#define NEAR_FAR_HIP_CHECK(expr) \
  ::near_far_detail::hip_check((expr), #expr, __FILE__, __LINE__)

inline int current_hip_device() {
  int device = 0;
  NEAR_FAR_HIP_CHECK(hipGetDevice(&device));
  return device;
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

  void reset(std::size_t count) {
    release();
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Near-Far device allocation size overflow");
    }
    T* candidate = nullptr;
    NEAR_FAR_HIP_CHECK(
        hipMalloc(reinterpret_cast<void**>(&candidate), count * sizeof(T)));
    pointer_ = candidate;
    count_ = count;
    g_near_far_device_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)hipFree(pointer_);
      pointer_ = nullptr;
    }
    count_ = 0;
  }

  void move_from(DeviceBuffer&& other) noexcept {
    pointer_ = other.pointer_;
    count_ = other.count_;
    other.pointer_ = nullptr;
    other.count_ = 0;
  }

  T* pointer_ = nullptr;
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
    release();
    if (count == 0) return;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error(
          "Near-Far pinned allocation size overflow");
    }
    T* candidate = nullptr;
    NEAR_FAR_HIP_CHECK(hipHostMalloc(
        reinterpret_cast<void**>(&candidate),
        count * sizeof(T),
        hipHostMallocDefault));
    pointer_ = candidate;
    count_ = count;
    g_near_far_pinned_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)hipHostFree(pointer_);
      pointer_ = nullptr;
    }
    count_ = 0;
  }

  void move_from(PinnedHostBuffer&& other) noexcept {
    pointer_ = other.pointer_;
    count_ = other.count_;
    other.pointer_ = nullptr;
    other.count_ = 0;
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

class DisableTimingEvent {
 public:
  DisableTimingEvent() {
    NEAR_FAR_HIP_CHECK(
        hipEventCreateWithFlags(&event_, hipEventDisableTiming));
  }
  ~DisableTimingEvent() {
    if (event_ != nullptr) (void)hipEventDestroy(event_);
  }

  DisableTimingEvent(const DisableTimingEvent&) = delete;
  DisableTimingEvent& operator=(const DisableTimingEvent&) = delete;

  hipEvent_t get() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

inline std::size_t checked_size(Offset value, const char* name) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be nonnegative");
  }
  return static_cast<std::size_t>(value);
}

inline int grid_for_items(std::size_t count) {
  if (count == 0) {
    return 1;
  }
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(
      std::min<std::size_t>(kMaxGridX, std::max<std::size_t>(1, blocks)));
}

inline int grid_for_shard(int shard_capacity) {
  const int blocks =
      (shard_capacity + kBlockSize - 1) / kBlockSize;
  return std::max(1, std::min(kMaxBlocksPerShard, blocks));
}

void validate_delta(float delta) {
  if (!(delta > 0.0f) || !std::isfinite(delta)) {
    throw std::invalid_argument(
        "Near-Far delta must be a finite positive float");
  }
}

void validate_host_csr(const HostCsrF32& graph) {
  const std::size_t rows = checked_size(graph.rows, "rows");
  const std::size_t nnz = checked_size(graph.nnz, "nnz");
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::invalid_argument(
        "Near-Far requires a nonempty square CSR graph");
  }
  if (graph.rows >
      static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Near-Far queue vertices are stored as int");
  }
  if (graph.rowptr.size() != rows + 1 ||
      graph.colind.size() != nnz ||
      graph.values.size() != nnz) {
    throw std::invalid_argument(
        "Near-Far CSR array sizes do not match rows and nnz");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument(
        "Near-Far CSR rowptr must start at zero and end at nnz");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    if (graph.rowptr[row] < 0 ||
        graph.rowptr[row + 1] < graph.rowptr[row] ||
        graph.rowptr[row + 1] > graph.nnz) {
      throw std::invalid_argument(
          "Near-Far CSR rowptr must be monotone and in range");
    }
  }
  for (std::size_t edge = 0; edge < nnz; ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::invalid_argument(
          "Near-Far CSR contains an out-of-range destination");
    }
    if (!std::isfinite(graph.values[edge]) ||
        graph.values[edge] < 0.0f) {
      throw std::invalid_argument(
          "Near-Far requires finite nonnegative edge weights");
    }
  }
}

void validate_sources(Offset rows, const std::vector<int>& sources) {
  if (sources.empty()) {
    throw std::invalid_argument(
        "Near-Far requires at least one source vertex");
  }
  if (sources.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Near-Far source count must fit in int");
  }
  for (const int source : sources) {
    if (source < 0 || static_cast<Offset>(source) >= rows) {
      throw std::out_of_range(
          "Near-Far source vertex is outside the CSR graph");
    }
  }
}

void validate_targets(Offset rows, const std::vector<int>& targets) {
  if (targets.empty()) {
    throw std::invalid_argument(
        "Near-Far vector-target run requires at least one target");
  }
  if (targets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Near-Far target count must fit in int");
  }
  for (const int target : targets) {
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      throw std::out_of_range(
          "Near-Far target vertex is outside the CSR graph");
    }
  }
}

struct DeviceCsrOwner {
  DeviceBuffer<CompactOffset> compact_rowptr;
  DeviceBuffer<Offset> wide_rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  bool uses_compact_offsets = false;
  bool unit_weights = false;

  DeviceCsrOwner(const HostCsrF32& host, hipStream_t stream)
      : colind(static_cast<std::size_t>(host.nnz)),
        rows(host.rows),
        cols(host.cols),
        nnz(host.nnz),
        uses_compact_offsets(
            static_cast<std::uint64_t>(host.nnz) <=
            static_cast<std::uint64_t>(kNoCompactEdge)),
        unit_weights(std::all_of(host.values.begin(),
                                 host.values.end(),
                                 [](float value) { return value == 1.0f; })) {
    if (uses_compact_offsets) {
      std::vector<CompactOffset> rowptr(host.rowptr.size());
      std::transform(host.rowptr.begin(),
                     host.rowptr.end(),
                     rowptr.begin(),
                     [](Offset value) {
                       return static_cast<CompactOffset>(value);
                     });
      compact_rowptr.reset(rowptr.size());
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          compact_rowptr.get(),
          rowptr.data(),
          rowptr.size() * sizeof(CompactOffset),
          hipMemcpyHostToDevice,
          stream));
    } else {
      wide_rowptr.reset(host.rowptr.size());
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          wide_rowptr.get(),
          host.rowptr.data(),
          host.rowptr.size() * sizeof(Offset),
          hipMemcpyHostToDevice,
          stream));
    }
    if (host.nnz != 0) {
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          colind.get(),
          host.colind.data(),
          host.colind.size() * sizeof(Index),
          hipMemcpyHostToDevice,
          stream));
      if (!unit_weights) {
        values.reset(host.values.size());
        NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
            values.get(),
            host.values.data(),
            host.values.size() * sizeof(float),
            hipMemcpyHostToDevice,
            stream));
      }
    }
    // Shared workspaces may subsequently use different streams.
    NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
  }
};

struct SchedulerStatus {
  int queue_totals[kQueueCount] = {};
  unsigned int queue_min_bits[kQueueCount] = {};
  int error = 0;
  int unreached_targets = 0;
  unsigned int max_target_bits = 0;
  unsigned int min_pending_bits = 0;
  // Keep callback-only state last so callback-free checkpoints can omit it.
  int changed = 0;
};

struct PathTotals {
  int node_count = 0;
  int edge_count = 0;
  int all_reached = 0;
  int error = 0;
};

struct DeviceTelemetry {
  unsigned long long edge_visits = 0;
  unsigned long long successful_relaxations = 0;
  unsigned long long lock_retries = 0;
  unsigned long long queue_insertions = 0;
  unsigned int queue_high_water[kQueueCount] = {};
};

#if defined(NEAR_FAR_ENABLE_TELEMETRY)
#define NEAR_FAR_TELEMETRY_ADD(view, field, value)                  \
  do {                                                              \
    if ((view).telemetry != nullptr) {                               \
      atomicAdd(&(view).telemetry->field,                            \
                static_cast<unsigned long long>(value));             \
    }                                                               \
  } while (false)
#define NEAR_FAR_TELEMETRY_HIGH_WATER(view, value)                  \
  do {                                                              \
    if ((view).telemetry != nullptr) {                               \
      atomicMax((view).telemetry->queue_high_water +                 \
                    (view).queue_index,                              \
                static_cast<unsigned int>(value));                   \
    }                                                               \
  } while (false)
constexpr std::size_t kTelemetryAllocationCount = 1;
#else
#define NEAR_FAR_TELEMETRY_ADD(view, field, value) ((void)0)
#define NEAR_FAR_TELEMETRY_HIGH_WATER(view, value) ((void)0)
constexpr std::size_t kTelemetryAllocationCount = 0;
#endif

struct DeviceQueueView {
  int* vertices = nullptr;
  int* shard_counts = nullptr;
  int* total_claims = nullptr;
  unsigned int* epoch_queue = nullptr;
  SchedulerStatus* status = nullptr;
  DeviceTelemetry* telemetry = nullptr;
  unsigned int epoch = 1;
  unsigned int membership_bit = 0;
  int queue_index = 0;
  int shards = 1;
  int shard_mask = 0;
  int shard_capacity = 0;
  int logical_capacity = 0;
  bool bounded = false;
};

struct QueueStorage {
  DeviceBuffer<int> vertices;
  DeviceBuffer<int> shard_counts;
  DeviceBuffer<int> total_claims;
  int shards = 1;
  int shard_mask = 0;
  int shard_capacity = 0;
  int logical_capacity = 0;
  unsigned int membership_bit = 0;
  int queue_index = 0;
  bool bounded = false;

  static int capacity_per_shard(int capacity, int shard_count) {
    return capacity / shard_count +
           (capacity % shard_count == 0 ? 0 : 1);
  }

  QueueStorage() = default;
  QueueStorage(int row_count,
               int shard_count,
               int capacity,
               int queue_index)
      : vertices(static_cast<std::size_t>(shard_count) *
                 static_cast<std::size_t>(
                     capacity_per_shard(capacity, shard_count))),
        shard_counts(static_cast<std::size_t>(shard_count)),
        total_claims(capacity < row_count ? 1 : 0),
        shards(shard_count),
        shard_mask((shard_count & (shard_count - 1)) == 0
                       ? shard_count - 1
                       : -1),
        shard_capacity(capacity_per_shard(capacity, shard_count)),
        logical_capacity(capacity),
        membership_bit(1U << static_cast<unsigned int>(queue_index)),
        queue_index(queue_index),
        bounded(capacity < row_count) {}

  DeviceQueueView view(unsigned int* epoch_queue,
                       SchedulerStatus* status,
                       unsigned int epoch,
                       DeviceTelemetry* telemetry = nullptr) {
    return {vertices.get(),
            shard_counts.get(),
            total_claims.get(),
            epoch_queue,
            status,
            telemetry,
            epoch,
            membership_bit,
            queue_index,
            shards,
            shard_mask,
            shard_capacity,
            logical_capacity,
            bounded};
  }
};

struct QueueSetView {
  DeviceQueueView queues[kQueueCount]{};
};

struct Scratch {
  Offset rows = 0;
  int shards = 1;
  int urgent_capacity = 1;
  bool compact_parents = false;

  DeviceBuffer<float> dist;
  DeviceBuffer<int> owner_source;
  DeviceBuffer<std::uint32_t> hops;
  DeviceBuffer<unsigned long long> compact_parent;
  DeviceBuffer<int> wide_pred_node;
  DeviceBuffer<Offset> wide_pred_edge;
  DeviceBuffer<std::uint32_t> version;
  DeviceBuffer<std::uint32_t> processed_version;
  DeviceBuffer<unsigned int> epoch_queue;
  DeviceBuffer<unsigned int> locks;

  std::array<QueueStorage, kQueueCount> queues;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_sources;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_path_status;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<CompactOffset> compact_path_edges;
  DeviceBuffer<Offset> wide_path_edges;
  DeviceBuffer<SchedulerStatus> scheduler_status;
  DeviceBuffer<PathTotals> path_totals;
  DeviceBuffer<DeviceTelemetry> telemetry;

  PinnedHostBuffer<SchedulerStatus> host_scheduler_status;
  PinnedHostBuffer<PathTotals> host_path_totals;
  PinnedHostBuffer<float> host_target_distances;
  PinnedHostBuffer<int> host_target_sources;
  PinnedHostBuffer<int> host_target_path_status;
  PinnedHostBuffer<int> host_target_node_offsets;
  PinnedHostBuffer<int> host_target_edge_offsets;
  PinnedHostBuffer<int> host_compact_path_nodes;
  PinnedHostBuffer<CompactOffset> host_compact_path_edges;
  PinnedHostBuffer<Offset> host_wide_path_edges;
  DisableTimingEvent ready_event;

  Scratch(Offset row_count,
          int shard_count,
          int urgent_count,
          bool use_compact_parents)
      : rows(row_count),
        shards(shard_count),
        urgent_capacity(urgent_count),
        compact_parents(use_compact_parents),
        dist(static_cast<std::size_t>(row_count)),
        owner_source(static_cast<std::size_t>(row_count)),
        hops(static_cast<std::size_t>(row_count)),
        compact_parent(use_compact_parents
                           ? static_cast<std::size_t>(row_count)
                           : 0),
        wide_pred_node(use_compact_parents
                           ? 0
                           : static_cast<std::size_t>(row_count)),
        wide_pred_edge(use_compact_parents
                           ? 0
                           : static_cast<std::size_t>(row_count)),
        version(static_cast<std::size_t>(row_count)),
        processed_version(static_cast<std::size_t>(row_count)),
        epoch_queue(static_cast<std::size_t>(row_count)),
        locks(static_cast<std::size_t>(row_count)),
        scheduler_status(1),
        path_totals(1),
        telemetry(kTelemetryAllocationCount),
        host_scheduler_status(1),
        host_path_totals(1) {
    const int rows_as_int = static_cast<int>(row_count);
    queues[kCurrentQueue] =
        QueueStorage(rows_as_int, 1, urgent_capacity, kCurrentQueue);
    queues[kNextQueue] =
        QueueStorage(rows_as_int, 1, urgent_capacity, kNextQueue);
    queues[kNearFarQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int, kNearFarQueue);
    queues[kFarQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int, kFarQueue);
    queues[kScratchQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int, kScratchQueue);
  }

  void ensure_source_capacity(std::size_t count) {
    if (sources.size() < count) {
      sources.reset(grown_capacity(sources.size(), count));
    }
  }

  void ensure_target_capacity(std::size_t count) {
    const std::size_t capacity = grown_capacity(targets.size(), count);
    bool grew = false;
    if (targets.size() < count) {
      targets.reset(capacity);
      grew = true;
    }
    if (target_distances.size() < count) {
      target_distances.reset(capacity);
      host_target_distances.reset(capacity);
      grew = true;
    }
    if (target_sources.size() < count) {
      target_sources.reset(capacity);
      host_target_sources.reset(capacity);
      grew = true;
    }
    if (target_path_lengths.size() < count) {
      target_path_lengths.reset(capacity);
      grew = true;
    }
    if (target_path_status.size() < count) {
      target_path_status.reset(capacity);
      host_target_path_status.reset(capacity);
      grew = true;
    }
    if (target_node_offsets.size() < count + 1) {
      target_node_offsets.reset(capacity + 1);
      host_target_node_offsets.reset(capacity + 1);
      grew = true;
    }
    if (target_edge_offsets.size() < count + 1) {
      target_edge_offsets.reset(capacity + 1);
      host_target_edge_offsets.reset(capacity + 1);
      grew = true;
    }
    if (grew) {
      g_near_far_target_growth_count.fetch_add(1,
                                               std::memory_order_relaxed);
    }
  }

  void ensure_compact_path_capacity(std::size_t nodes, std::size_t edges) {
    bool grew = false;
    if (compact_path_nodes.size() < nodes) {
      const std::size_t capacity =
          grown_capacity(compact_path_nodes.size(), nodes);
      compact_path_nodes.reset(capacity);
      host_compact_path_nodes.reset(capacity);
      grew = true;
    }
    if (compact_parents) {
      if (compact_path_edges.size() < edges) {
        const std::size_t capacity =
            grown_capacity(compact_path_edges.size(), edges);
        compact_path_edges.reset(capacity);
        host_compact_path_edges.reset(capacity);
        grew = true;
      }
    } else if (wide_path_edges.size() < edges) {
      const std::size_t capacity =
          grown_capacity(wide_path_edges.size(), edges);
      wide_path_edges.reset(capacity);
      host_wide_path_edges.reset(capacity);
      grew = true;
    }
    if (grew) {
      g_near_far_path_growth_count.fetch_add(1,
                                             std::memory_order_relaxed);
    }
  }

 private:
  static std::size_t grown_capacity(std::size_t current,
                                    std::size_t required) {
    if (current >= required) return current;
    if (current == 0) return required;
    const std::size_t growth = current / 2 + 1;
    if (current > std::numeric_limits<std::size_t>::max() - growth) {
      return required;
    }
    return std::max(required, current + growth);
  }
};

__device__ inline void set_device_error(SchedulerStatus* status,
                                        DeviceError value) {
  atomicCAS(&status->error,
            static_cast<int>(kNoDeviceError),
            static_cast<int>(value));
}

__device__ inline bool try_lock_vertex(unsigned int* lock) {
  return atomicCAS(lock, 0U, 1U) == 0U;
}

__device__ inline void unlock_vertex(unsigned int* lock) {
  __threadfence();
  atomicExch(lock, 0U);
}

__device__ inline bool bump_version(std::uint32_t* version,
                                    SchedulerStatus* status) {
  std::uint32_t old = atomicAdd(version, 0U);
  while (old != std::numeric_limits<std::uint32_t>::max()) {
    const std::uint32_t observed = atomicCAS(version, old, old + 1U);
    if (observed == old) return true;
    old = observed;
  }
  set_device_error(status, kVersionOverflow);
  return false;
}

__device__ inline bool finite_float(float value) {
  return (__float_as_uint(value) & 0x7f800000U) != 0x7f800000U;
}

__device__ inline unsigned int packed_epoch(unsigned int word) {
  return word >> kEpochShift;
}

__device__ inline bool state_is_current(const unsigned int* epoch_queue,
                                        int vertex,
                                        unsigned int epoch) {
  return packed_epoch(epoch_queue[vertex]) == epoch;
}

template <bool CompactParents>
__device__ inline int load_parent_node(
    int vertex,
    const unsigned long long* compact_parent,
    const int* wide_pred_node) {
  if constexpr (CompactParents) {
    return static_cast<int>(
        static_cast<std::uint32_t>(compact_parent[vertex] >> 32));
  } else {
    return wide_pred_node[vertex];
  }
}

template <bool CompactParents>
__device__ inline Offset load_parent_edge(
    int vertex,
    const unsigned long long* compact_parent,
    const Offset* wide_pred_edge) {
  if constexpr (CompactParents) {
    const CompactOffset edge =
        static_cast<CompactOffset>(compact_parent[vertex]);
    return edge == kNoCompactEdge ? static_cast<Offset>(-1)
                                  : static_cast<Offset>(edge);
  } else {
    return wide_pred_edge[vertex];
  }
}

template <bool CompactParents>
__device__ inline void store_parent(
    int vertex,
    int predecessor,
    Offset edge,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge) {
  if constexpr (CompactParents) {
    const CompactOffset compact_edge =
        edge < 0 ? kNoCompactEdge : static_cast<CompactOffset>(edge);
    compact_parent[vertex] =
        (static_cast<unsigned long long>(
             static_cast<std::uint32_t>(predecessor))
         << 32) |
        static_cast<unsigned long long>(compact_edge);
  } else {
    wide_pred_node[vertex] = predecessor;
    wide_pred_edge[vertex] = edge;
  }
}

__device__ inline void clear_queue_bit(DeviceQueueView queue, int vertex) {
  unsigned int observed = atomicAdd(queue.epoch_queue + vertex, 0U);
  while (packed_epoch(observed) == queue.epoch &&
         (observed & queue.membership_bit) != 0U) {
    const unsigned int desired = observed & ~queue.membership_bit;
    const unsigned int previous =
        atomicCAS(queue.epoch_queue + vertex, observed, desired);
    if (previous == observed) return;
    observed = previous;
  }
}

__device__ inline bool claim_queue_bit(DeviceQueueView queue,
                                       int vertex,
                                       bool* inserted) {
  const unsigned int epoch_bits = queue.epoch << kEpochShift;
  unsigned int observed = atomicAdd(queue.epoch_queue + vertex, 0U);
  while (true) {
    if (packed_epoch(observed) == queue.epoch &&
        (observed & queue.membership_bit) != 0U) {
      *inserted = false;
      return true;
    }
    const unsigned int current_mask =
        packed_epoch(observed) == queue.epoch ? observed & kQueueMask : 0U;
    const unsigned int desired =
        epoch_bits | current_mask | queue.membership_bit;
    const unsigned int previous =
        atomicCAS(queue.epoch_queue + vertex, observed, desired);
    if (previous == observed) {
      *inserted = true;
      return true;
    }
    observed = previous;
  }
}

__device__ inline bool enqueue_unique(DeviceQueueView queue,
                                      int vertex,
                                      bool overflow_is_error) {
  bool inserted = false;
  if (queue.bounded) {
    const unsigned int word = atomicAdd(queue.epoch_queue + vertex, 0U);
    if (packed_epoch(word) == queue.epoch &&
        (word & queue.membership_bit) != 0U) {
      return true;
    }

    const int reservation = atomicAdd(queue.total_claims, 1);
    if (reservation >= queue.logical_capacity) {
      atomicSub(queue.total_claims, 1);
      const unsigned int latest =
          atomicAdd(queue.epoch_queue + vertex, 0U);
      if (packed_epoch(latest) == queue.epoch &&
          (latest & queue.membership_bit) != 0U) {
        return true;
      }
      if (overflow_is_error) {
        set_device_error(queue.status, kDeferredQueueOverflow);
      }
      return false;
    }
    (void)claim_queue_bit(queue, vertex, &inserted);
    if (!inserted) {
      atomicSub(queue.total_claims, 1);
      return true;
    }
  } else {
    (void)claim_queue_bit(queue, vertex, &inserted);
    if (!inserted) return true;
  }

  const int shard = queue.shard_mask >= 0
                        ? vertex & queue.shard_mask
                        : vertex % queue.shards;
  const int position = atomicAdd(queue.shard_counts + shard, 1);
  if (position >= queue.shard_capacity) {
    set_device_error(queue.status, kDeferredQueueOverflow);
    return false;
  }
  queue.vertices[shard * queue.shard_capacity + position] = vertex;
  NEAR_FAR_TELEMETRY_ADD(queue, queue_insertions, 1);
  NEAR_FAR_TELEMETRY_HIGH_WATER(queue, position + 1);
  return true;
}

template <bool CompactParents>
__device__ inline void initialize_vertex_for_epoch(
    int vertex,
    unsigned int epoch,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* epoch_queue) {
  unsigned int observed = atomicAdd(epoch_queue + vertex, 0U);
  if (packed_epoch(observed) == epoch) return;

  dist[vertex] = INFINITY;
  owner_source[vertex] = std::numeric_limits<int>::max();
  hops[vertex] = std::numeric_limits<std::uint32_t>::max();
  store_parent<CompactParents>(vertex,
                               -1,
                               -1,
                               compact_parent,
                               wide_pred_node,
                               wide_pred_edge);
  version[vertex] = 0;
  processed_version[vertex] = 0;
  __threadfence();

  const unsigned int desired = epoch << kEpochShift;
  while (packed_epoch(observed) != epoch) {
    const unsigned int previous =
        atomicCAS(epoch_queue + vertex, observed, desired);
    if (previous == observed) return;
    observed = previous;
  }
}

__device__ inline void enqueue_by_distance(int vertex,
                                           float distance,
                                           float near_far_threshold,
                                           DeviceQueueView urgent,
                                           DeviceQueueView near_far,
                                           DeviceQueueView far) {
  if (distance < near_far_threshold) {
    if (!enqueue_unique(urgent, vertex, false)) {
      (void)enqueue_unique(near_far, vertex, true);
    }
  } else {
    (void)enqueue_unique(far, vertex, true);
  }
}

template <bool CompactParents>
__global__ void initialize_sources_kernel(
    const int* sources,
    int source_count,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* epoch_queue,
    unsigned int* locks,
    unsigned int epoch,
    DeviceQueueView primary,
    DeviceQueueView spill) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < source_count;
       index += gridDim.x * blockDim.x) {
    const int source = sources[index];
    if (!try_lock_vertex(locks + source)) continue;
    initialize_vertex_for_epoch<CompactParents>(source,
                                                 epoch,
                                                 dist,
                                                 owner_source,
                                                 hops,
                                                 compact_parent,
                                                 wide_pred_node,
                                                 wide_pred_edge,
                                                 version,
                                                 processed_version,
                                                 epoch_queue);
    if (dist[source] != 0.0f || owner_source[source] > source ||
        hops[source] != 0) {
      dist[source] = 0.0f;
      owner_source[source] = source;
      hops[source] = 0;
      store_parent<CompactParents>(source,
                                   source,
                                   -1,
                                   compact_parent,
                                   wide_pred_node,
                                   wide_pred_edge);
      version[source] = 1;
    }
    unlock_vertex(locks + source);
    if (!enqueue_unique(primary, source, false)) {
      (void)enqueue_unique(spill, source, true);
    }
  }
}

__device__ inline bool candidate_is_better(
    float candidate_distance,
    int candidate_source,
    std::uint32_t candidate_hops,
    Offset candidate_edge,
    float current_distance,
    int current_source,
    std::uint32_t current_hops,
    Offset current_edge) {
  if (candidate_distance < current_distance) return true;
  if (candidate_distance > current_distance) return false;
  const bool candidate_is_identity = candidate_hops == 0;
  const bool current_is_identity = current_hops == 0;
  if (candidate_is_identity != current_is_identity) {
    return candidate_is_identity;
  }
  if (candidate_source < current_source) return true;
  if (candidate_source > current_source) return false;
  if (candidate_hops < current_hops) return true;
  if (candidate_hops > current_hops) return false;
  return current_edge < 0 || candidate_edge < current_edge;
}

template <typename RowOffset,
          bool UnitWeights,
          bool HasVertexCosts,
          bool CompactParents,
          bool TrackChanged>
__global__ void expand_queue_kernel(
    DeviceQueueView input,
    const RowOffset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    float near_far_threshold,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* locks,
    DeviceQueueView next,
    DeviceQueueView near_far,
    DeviceQueueView far) {
  const int shard = static_cast<int>(blockIdx.y);
  if (shard >= input.shards) return;
  const int queue_count = input.shard_counts[shard];
  if (queue_count < 0 || queue_count > input.shard_capacity) {
    set_device_error(input.status, kDeferredQueueOverflow);
    return;
  }
  const int* queue_vertices =
      input.vertices + shard * input.shard_capacity;
  for (int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
       queue_index < queue_count;
       queue_index += gridDim.x * blockDim.x) {
    const int u = queue_vertices[queue_index];
    clear_queue_bit(input, u);

    if (!try_lock_vertex(locks + u)) {
      NEAR_FAR_TELEMETRY_ADD(input, lock_retries, 1);
      (void)enqueue_unique(near_far, u, true);
      continue;
    }
    if (!state_is_current(input.epoch_queue, u, input.epoch)) {
      unlock_vertex(locks + u);
      continue;
    }
    const std::uint32_t current_version = atomicAdd(version + u, 0U);
    if (current_version == 0 ||
        current_version <= processed_version[u]) {
      unlock_vertex(locks + u);
      continue;
    }
    processed_version[u] = current_version;
    const float source_distance = dist[u];
    const int source_owner = owner_source[u];
    const std::uint32_t source_hops = hops[u];
    unlock_vertex(locks + u);

    if (!finite_float(source_distance) ||
        source_owner == std::numeric_limits<int>::max() ||
        source_hops >= std::numeric_limits<std::uint32_t>::max() - 1U) {
      set_device_error(input.status, kInvalidVertexState);
      continue;
    }

    bool retry_source = false;
    const Offset edge_begin = static_cast<Offset>(rowptr[u]);
    const Offset edge_end = static_cast<Offset>(rowptr[u + 1]);
    for (Offset edge = edge_begin; edge < edge_end; ++edge) {
      NEAR_FAR_TELEMETRY_ADD(input, edge_visits, 1);
      const int v = colind[edge];
      float effective_weight = 1.0f;
      if constexpr (!UnitWeights) effective_weight = values[edge];
      if constexpr (HasVertexCosts) effective_weight *= vertex_costs[v];
      const float candidate_distance = source_distance + effective_weight;
      if (!finite_float(candidate_distance)) continue;
      const std::uint32_t candidate_hops = source_hops + 1U;

      bool updated = false;
      if (!try_lock_vertex(locks + v)) {
        NEAR_FAR_TELEMETRY_ADD(input, lock_retries, 1);
        retry_source = true;
        continue;
      }
      initialize_vertex_for_epoch<CompactParents>(v,
                                                   input.epoch,
                                                   dist,
                                                   owner_source,
                                                   hops,
                                                   compact_parent,
                                                   wide_pred_node,
                                                   wide_pred_edge,
                                                   version,
                                                   processed_version,
                                                   input.epoch_queue);
      const float current_distance = dist[v];
      bool better = candidate_distance < current_distance;
      if (candidate_distance == current_distance) {
        const Offset current_edge =
            load_parent_edge<CompactParents>(
                v, compact_parent, wide_pred_edge);
        better = candidate_is_better(candidate_distance,
                                     source_owner,
                                     candidate_hops,
                                     edge,
                                     current_distance,
                                     owner_source[v],
                                     hops[v],
                                     current_edge);
      }
      if (better && bump_version(version + v, input.status)) {
        dist[v] = candidate_distance;
        owner_source[v] = source_owner;
        hops[v] = candidate_hops;
        store_parent<CompactParents>(v,
                                     u,
                                     edge,
                                     compact_parent,
                                     wide_pred_node,
                                     wide_pred_edge);
        updated = true;
        NEAR_FAR_TELEMETRY_ADD(input, successful_relaxations, 1);
      }
      unlock_vertex(locks + v);

      if (updated) {
        if constexpr (TrackChanged) atomicExch(&input.status->changed, 1);
        enqueue_by_distance(v,
                            candidate_distance,
                            near_far_threshold,
                            next,
                            near_far,
                            far);
      }
    }
    if (retry_source && bump_version(version + u, input.status)) {
      (void)enqueue_unique(near_far, u, true);
    }
  }
}

template <typename RowOffset, bool CompactParents>
__global__ void expand_exact_unit_kernel(
    DeviceQueueView input,
    const RowOffset* rowptr,
    const Index* colind,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* locks,
    DeviceQueueView next,
    DeviceQueueView retry) {
  const int shard = static_cast<int>(blockIdx.y);
  if (shard >= input.shards) return;
  const int queue_count = input.shard_counts[shard];
  if (queue_count < 0 || queue_count > input.shard_capacity) {
    set_device_error(input.status, kDeferredQueueOverflow);
    return;
  }
  const int* queue_vertices =
      input.vertices + shard * input.shard_capacity;
  for (int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
       queue_index < queue_count;
       queue_index += gridDim.x * blockDim.x) {
    const int u = queue_vertices[queue_index];
    clear_queue_bit(input, u);
    if (!try_lock_vertex(locks + u)) {
      NEAR_FAR_TELEMETRY_ADD(input, lock_retries, 1);
      (void)enqueue_unique(retry, u, true);
      continue;
    }
    if (!state_is_current(input.epoch_queue, u, input.epoch) ||
        version[u] == 0) {
      unlock_vertex(locks + u);
      continue;
    }
    const std::uint32_t source_version = version[u];
    const float source_distance = dist[u];
    const int source_owner = owner_source[u];
    const std::uint32_t source_hops = hops[u];
    unlock_vertex(locks + u);

    if (!finite_float(source_distance) ||
        source_owner == std::numeric_limits<int>::max() ||
        source_hops >= std::numeric_limits<std::uint32_t>::max() - 1U) {
      set_device_error(input.status, kInvalidVertexState);
      continue;
    }

    bool retry_source = false;
    const Offset edge_begin = static_cast<Offset>(rowptr[u]);
    const Offset edge_end = static_cast<Offset>(rowptr[u + 1]);
    for (Offset edge = edge_begin; edge < edge_end; ++edge) {
      NEAR_FAR_TELEMETRY_ADD(input, edge_visits, 1);
      const int v = colind[edge];
      const float candidate_distance = source_distance + 1.0f;
      const std::uint32_t candidate_hops = source_hops + 1U;
      bool updated = false;
      if (!try_lock_vertex(locks + v)) {
        NEAR_FAR_TELEMETRY_ADD(input, lock_retries, 1);
        retry_source = true;
        continue;
      }
      initialize_vertex_for_epoch<CompactParents>(v,
                                                   input.epoch,
                                                   dist,
                                                   owner_source,
                                                   hops,
                                                   compact_parent,
                                                   wide_pred_node,
                                                   wide_pred_edge,
                                                   version,
                                                   processed_version,
                                                   input.epoch_queue);
      const float current_distance = dist[v];
      bool better = candidate_distance < current_distance;
      if (candidate_distance == current_distance) {
        const Offset current_edge =
            load_parent_edge<CompactParents>(
                v, compact_parent, wide_pred_edge);
        better = candidate_is_better(candidate_distance,
                                     source_owner,
                                     candidate_hops,
                                     edge,
                                     current_distance,
                                     owner_source[v],
                                     hops[v],
                                     current_edge);
      }
      if (better && bump_version(version + v, input.status)) {
        dist[v] = candidate_distance;
        owner_source[v] = source_owner;
        hops[v] = candidate_hops;
        store_parent<CompactParents>(v,
                                     u,
                                     edge,
                                     compact_parent,
                                     wide_pred_node,
                                     wide_pred_edge);
        updated = true;
        NEAR_FAR_TELEMETRY_ADD(input, successful_relaxations, 1);
      }
      unlock_vertex(locks + v);
      if (updated) (void)enqueue_unique(next, v, true);
    }

    if (retry_source) {
      (void)enqueue_unique(retry, u, true);
    } else if (try_lock_vertex(locks + u)) {
      if (state_is_current(input.epoch_queue, u, input.epoch)) {
        processed_version[u] =
            processed_version[u] > source_version
                ? processed_version[u]
                : source_version;
      }
      unlock_vertex(locks + u);
    } else {
      NEAR_FAR_TELEMETRY_ADD(input, lock_retries, 1);
      (void)enqueue_unique(retry, u, true);
    }
  }
}

__global__ void reclassify_queue_kernel(
    DeviceQueueView input,
    float near_far_threshold,
    const float* dist,
    const std::uint32_t* version,
    const std::uint32_t* processed_version,
    DeviceQueueView urgent,
    DeviceQueueView near_far,
    DeviceQueueView far) {
  const int shard = static_cast<int>(blockIdx.y);
  if (shard >= input.shards) return;
  const int input_count = input.shard_counts[shard];
  if (input_count < 0 || input_count > input.shard_capacity) {
    set_device_error(input.status, kDeferredQueueOverflow);
    return;
  }
  const int* input_vertices =
      input.vertices + shard * input.shard_capacity;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < input_count;
       index += gridDim.x * blockDim.x) {
    const int vertex = input_vertices[index];
    clear_queue_bit(input, vertex);
    if (!state_is_current(input.epoch_queue, vertex, input.epoch) ||
        version[vertex] == 0 ||
        version[vertex] <= processed_version[vertex]) {
      continue;
    }
    enqueue_by_distance(vertex,
                        dist[vertex],
                        near_far_threshold,
                        urgent,
                        near_far,
                        far);
  }
}

__global__ void clear_queue_membership_kernel(DeviceQueueView queue) {
  const int shard = static_cast<int>(blockIdx.y);
  if (shard >= queue.shards) return;
  const int count = queue.shard_counts[shard];
  if (count < 0 || count > queue.shard_capacity) {
    set_device_error(queue.status, kDeferredQueueOverflow);
    return;
  }
  const int* vertices = queue.vertices + shard * queue.shard_capacity;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += gridDim.x * blockDim.x) {
    clear_queue_bit(queue, vertices[index]);
  }
}

__global__ void reset_checkpoint_status_kernel(SchedulerStatus* status,
                                               bool reset_changed) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int queue = 0; queue < kQueueCount; ++queue) {
    status->queue_totals[queue] = 0;
    status->queue_min_bits[queue] =
        std::numeric_limits<unsigned int>::max();
  }
  status->unreached_targets = 0;
  status->max_target_bits = 0;
  status->min_pending_bits =
      std::numeric_limits<unsigned int>::max();
  if (reset_changed) status->changed = 0;
}

__global__ void collect_queue_counts_kernel(QueueSetView queue_set,
                                            SchedulerStatus* status) {
  for (int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
       queue_index < kQueueCount;
       queue_index += gridDim.x * blockDim.x) {
    const DeviceQueueView queue = queue_set.queues[queue_index];
    int total = 0;
    for (int shard = 0; shard < queue.shards; ++shard) {
      const int count = queue.shard_counts[shard];
      if (count < 0 || count > queue.shard_capacity) {
        set_device_error(status, kDeferredQueueOverflow);
        return;
      }
      total += count;
    }
    status->queue_totals[queue_index] = total;
  }
}

__global__ void collect_queue_status_kernel(
    QueueSetView queue_set,
    const float* dist,
    const std::uint32_t* version,
    const std::uint32_t* processed_version,
    SchedulerStatus* status,
    bool collect_minima) {
  const int queue_index = static_cast<int>(blockIdx.z);
  if (queue_index >= kQueueCount) return;
  const DeviceQueueView queue = queue_set.queues[queue_index];
  const int shard = static_cast<int>(blockIdx.y);
  if (shard >= queue.shards) return;
  const int count = queue.shard_counts[shard];
  if (count < 0 || count > queue.shard_capacity) {
    set_device_error(status, kDeferredQueueOverflow);
    return;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(status->queue_totals + queue_index, count);
  }
  if (!collect_minima) return;
  const int first_index =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
  if (first_index >= count) return;

  // Queue entries are distributed across many blocks. Reduce within each
  // block before touching the two global minima instead of issuing two global
  // atomics for every live entry.
  __shared__ unsigned int block_min_bits[kBlockSize];
  unsigned int local_min_bits =
      std::numeric_limits<unsigned int>::max();
  const int* vertices = queue.vertices + shard * queue.shard_capacity;
  for (int index = first_index + static_cast<int>(threadIdx.x);
       index < count;
       index += gridDim.x * blockDim.x) {
    const int vertex = vertices[index];
    if (state_is_current(queue.epoch_queue, vertex, queue.epoch) &&
        version[vertex] > processed_version[vertex]) {
      const unsigned int bits = __float_as_uint(dist[vertex]);
      local_min_bits =
          local_min_bits < bits ? local_min_bits : bits;
    }
  }
  block_min_bits[threadIdx.x] = local_min_bits;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_min_bits[threadIdx.x] =
          block_min_bits[threadIdx.x] <
                  block_min_bits[threadIdx.x + stride]
              ? block_min_bits[threadIdx.x]
              : block_min_bits[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 &&
      block_min_bits[0] !=
          std::numeric_limits<unsigned int>::max()) {
    atomicMin(status->queue_min_bits + queue_index,
              block_min_bits[0]);
    atomicMin(&status->min_pending_bits, block_min_bits[0]);
  }
}

__global__ void collect_target_bounds_kernel(const int* targets,
                                             int target_count,
                                             const float* dist,
                                             const unsigned int* epoch_queue,
                                             unsigned int epoch,
                                             SchedulerStatus* status) {
  __shared__ int block_unreached[kBlockSize];
  __shared__ unsigned int block_max_bits[kBlockSize];
  int local_unreached = 0;
  unsigned int local_max_bits = 0;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    const int target = targets[index];
    if (!state_is_current(epoch_queue, target, epoch)) {
      ++local_unreached;
      continue;
    }
    const float value = dist[target];
    if (!finite_float(value)) {
      ++local_unreached;
    } else {
      const unsigned int bits = __float_as_uint(value);
      local_max_bits =
          local_max_bits > bits ? local_max_bits : bits;
    }
  }
  block_unreached[threadIdx.x] = local_unreached;
  block_max_bits[threadIdx.x] = local_max_bits;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_unreached[threadIdx.x] +=
          block_unreached[threadIdx.x + stride];
      block_max_bits[threadIdx.x] =
          block_max_bits[threadIdx.x] >
                  block_max_bits[threadIdx.x + stride]
              ? block_max_bits[threadIdx.x]
              : block_max_bits[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (block_unreached[0] != 0) {
      atomicAdd(&status->unreached_targets, block_unreached[0]);
    }
    if (block_max_bits[0] != 0) {
      atomicMax(&status->max_target_bits, block_max_bits[0]);
    }
  }
}

template <typename RowOffset, bool CompactParents>
__global__ void measure_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    const RowOffset* rowptr,
    const Index* colind,
    const float* dist,
    const int* owner_source,
    const std::uint32_t* hops,
    const unsigned long long* compact_parent,
    const int* wide_pred_node,
    const Offset* wide_pred_edge,
    const unsigned int* epoch_queue,
    unsigned int epoch,
    bool paths_certified,
    float* target_distances,
    int* target_sources,
    int* path_lengths,
    int* path_status) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    const int target = targets[index];
    target_distances[index] = INFINITY;
    target_sources[index] = -1;
    path_lengths[index] = 0;
    path_status[index] = 0;
    if (!state_is_current(epoch_queue, target, epoch)) continue;

    const float target_distance = dist[target];
    const std::uint32_t target_hops = hops[target];
    if (!finite_float(target_distance) ||
        target_hops == std::numeric_limits<std::uint32_t>::max() ||
        (!paths_certified && target_hops != 0) ||
        target_hops >= static_cast<std::uint32_t>(rows)) {
      continue;
    }

    int current = target;
    std::uint32_t current_hops = target_hops;
    const int source = owner_source[target];
    bool valid = source >= 0 && static_cast<Offset>(source) < rows;
    while (valid && current_hops > 0) {
      const int predecessor =
          load_parent_node<CompactParents>(
              current, compact_parent, wide_pred_node);
      const Offset edge =
          load_parent_edge<CompactParents>(
              current, compact_parent, wide_pred_edge);
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows ||
          !state_is_current(epoch_queue, predecessor, epoch) ||
          edge < static_cast<Offset>(rowptr[predecessor]) ||
          edge >= static_cast<Offset>(rowptr[predecessor + 1]) ||
          colind[edge] != current ||
          owner_source[predecessor] != source ||
          hops[predecessor] + 1U != current_hops) {
        valid = false;
        break;
      }
      current = predecessor;
      --current_hops;
    }
    valid =
        valid && current == source && hops[current] == 0 &&
        load_parent_node<CompactParents>(
            current, compact_parent, wide_pred_node) == current;
    if (!valid) continue;

    target_distances[index] = target_distance;
    target_sources[index] = source;
    path_lengths[index] = static_cast<int>(target_hops) + 1;
    path_status[index] = kTargetPathValid;
  }
}

__global__ void scan_target_offsets_kernel(
    int target_count,
    const float* target_distances,
    const int* path_lengths,
    int* path_status,
    int* node_offsets,
    int* edge_offsets,
    PathTotals* totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int node_total = 0;
  int edge_total = 0;
  int all_reached = 1;
  int error = 0;
  for (int index = 0; index < target_count; ++index) {
    node_offsets[index] = node_total;
    edge_offsets[index] = edge_total;
    const int length = path_lengths[index];
    if (path_status[index] != kTargetPathValid || length <= 0 ||
        !finite_float(target_distances[index])) {
      all_reached = 0;
      continue;
    }
    if (length > std::numeric_limits<int>::max() - node_total ||
        length - 1 > std::numeric_limits<int>::max() - edge_total) {
      path_status[index] = 0;
      all_reached = 0;
      error = 1;
      continue;
    }
    node_total += length;
    edge_total += length - 1;
  }
  node_offsets[target_count] = node_total;
  edge_offsets[target_count] = edge_total;
  totals->node_count = node_total;
  totals->edge_count = edge_total;
  totals->all_reached = all_reached;
  totals->error = error;
}

template <typename RowOffset, bool CompactParents>
__global__ void fill_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    const RowOffset* rowptr,
    const Index* colind,
    const int* owner_source,
    const std::uint32_t* hops,
    const unsigned long long* compact_parent,
    const int* wide_pred_node,
    const Offset* wide_pred_edge,
    const unsigned int* epoch_queue,
    unsigned int epoch,
    const int* path_lengths,
    int* path_status,
    const int* node_offsets,
    const int* edge_offsets,
    int* path_nodes,
    CompactOffset* compact_path_edges,
    Offset* wide_path_edges) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    if (path_status[index] != kTargetPathValid ||
        path_lengths[index] <= 0) {
      continue;
    }
    int current = targets[index];
    std::uint32_t current_hops = hops[current];
    const int source = owner_source[current];
    const int node_begin = node_offsets[index];
    const int edge_begin = edge_offsets[index];
    bool valid =
        path_lengths[index] == static_cast<int>(current_hops) + 1;

    while (valid) {
      path_nodes[node_begin + static_cast<int>(current_hops)] = current;
      if (current_hops == 0) {
        valid =
            current == source &&
            load_parent_node<CompactParents>(
                current, compact_parent, wide_pred_node) == current;
        break;
      }
      const int predecessor =
          load_parent_node<CompactParents>(
              current, compact_parent, wide_pred_node);
      const Offset edge =
          load_parent_edge<CompactParents>(
              current, compact_parent, wide_pred_edge);
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows ||
          !state_is_current(epoch_queue, predecessor, epoch) ||
          edge < static_cast<Offset>(rowptr[predecessor]) ||
          edge >= static_cast<Offset>(rowptr[predecessor + 1]) ||
          colind[edge] != current ||
          owner_source[predecessor] != source ||
          hops[predecessor] + 1U != current_hops) {
        valid = false;
        break;
      }
      const int output = edge_begin + static_cast<int>(current_hops) - 1;
      if constexpr (CompactParents) {
        compact_path_edges[output] = static_cast<CompactOffset>(edge);
      } else {
        wide_path_edges[output] = edge;
      }
      current = predecessor;
      --current_hops;
    }
    if (!valid) path_status[index] = 0;
  }
}

template <bool CompactParents>
__global__ void materialize_full_result_kernel(
    Offset rows,
    unsigned int epoch,
    const unsigned int* epoch_queue,
    float* dist,
    unsigned long long* compact_parent,
    int* wide_pred_node,
    Offset* wide_pred_edge) {
  for (Offset vertex =
           static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       vertex < rows;
       vertex += static_cast<Offset>(gridDim.x) * blockDim.x) {
    const int index = static_cast<int>(vertex);
    if (state_is_current(epoch_queue, index, epoch)) continue;
    dist[index] = INFINITY;
    store_parent<CompactParents>(index,
                                 -1,
                                 -1,
                                 compact_parent,
                                 wide_pred_node,
                                 wide_pred_edge);
  }
}

void reset_queue_counts(QueueStorage& queue, hipStream_t stream) {
  NEAR_FAR_HIP_CHECK(hipMemsetAsync(queue.shard_counts.get(),
                                     0,
                                     static_cast<std::size_t>(queue.shards) *
                                         sizeof(int),
                                     stream));
  if (queue.total_claims.get() != nullptr) {
    NEAR_FAR_HIP_CHECK(
        hipMemsetAsync(queue.total_claims.get(), 0, sizeof(int), stream));
  }
}

void wait_for_event(DisableTimingEvent& event, hipStream_t stream) {
  NEAR_FAR_HIP_CHECK(hipEventRecord(event.get(), stream));
  NEAR_FAR_HIP_CHECK(hipEventSynchronize(event.get()));
}

float float_from_bits(unsigned int bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits), "float size mismatch");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int choose_queue_shards(Offset rows,
                        std::size_t requested,
                        const hipDeviceProp_t& properties) {
  const std::size_t max_shards =
      std::min<std::size_t>(kDefaultSubpartitions,
                            static_cast<std::size_t>(rows));
  if (requested != 0) {
    if (requested > max_shards) {
      throw std::invalid_argument(
          "Near-Far queue shard count exceeds min(rows, 16)");
    }
    return static_cast<int>(requested);
  }
  const int compute_units = std::max(1, properties.multiProcessorCount);
  return static_cast<int>(
      std::max<std::size_t>(
          1,
          std::min<std::size_t>(max_shards,
                                static_cast<std::size_t>(compute_units))));
}

int choose_urgent_capacity(Offset rows,
                           Offset nnz,
                           std::size_t requested,
                           const hipDeviceProp_t& properties) {
  if (requested != 0) {
    if (requested > static_cast<std::size_t>(rows)) {
      throw std::invalid_argument(
          "Near-Far urgent queue capacity exceeds graph rows");
    }
    return static_cast<int>(requested);
  }
  const double average_degree =
      rows == 0 ? 0.0 : static_cast<double>(nnz) / rows;
  const double useful_rows =
      average_degree > 0.0
          ? (static_cast<double>(
                 std::max(1, properties.multiProcessorCount)) *
             static_cast<double>(std::max(1, properties.warpSize)) *
             kUrgentWavesPerComputeUnit) /
                average_degree
          : static_cast<double>(rows);
  return static_cast<int>(
      std::max<double>(1.0,
                       std::min<double>(static_cast<double>(rows),
                                        std::ceil(useful_rows))));
}

float next_near_far_threshold(float current,
                              float delta,
                              float minimum_far_distance) {
  const double current_step =
      static_cast<double>(current) + static_cast<double>(delta);
  const double containing_step =
      (std::floor(static_cast<double>(minimum_far_distance) /
                  static_cast<double>(delta)) +
       1.0) *
      static_cast<double>(delta);
  double candidate = std::max(current_step, containing_step);
  if (!std::isfinite(candidate) ||
      !(candidate > static_cast<double>(minimum_far_distance))) {
    return std::nextafter(minimum_far_distance,
                          std::numeric_limits<float>::infinity());
  }
  const float narrowed = static_cast<float>(candidate);
  if (!(narrowed > minimum_far_distance)) {
    return std::nextafter(minimum_far_distance,
                          std::numeric_limits<float>::infinity());
  }
  return narrowed;
}

const char* device_error_message(int error) {
  switch (error) {
    case kNoDeviceError:
      return "no error";
    case kDeferredQueueOverflow:
      return "a deferred queue exceeded its checked capacity";
    case kVersionOverflow:
      return "a vertex update generation overflowed";
    case kInvalidVertexState:
      return "an active vertex had invalid state";
    default:
      return "an unknown device error occurred";
  }
}

}  // namespace near_far_detail

struct NearFarCsrGraph::Impl {
  near_far_detail::DeviceCsrOwner adjacency;
  int device = 0;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : adjacency(host, stream),
        device(near_far_detail::current_hip_device()) {}
};

struct NearFarCsrWorkspace::Impl {
  std::shared_ptr<const NearFarCsrGraph::Impl> graph;
  near_far_detail::Scratch scratch;
  near_far_detail::DeviceBuffer<float> vertex_costs;
  bool has_vertex_costs = false;
  hipStream_t stream = nullptr;
  int device = 0;
  unsigned int run_epoch = 0;
  near_far_detail::SchedulerStatus scheduler_status{};

  struct RunOutcome {
    int iterations = 0;
    bool converged = false;
    bool stopped_on_target = false;
  };

  static std::shared_ptr<const NearFarCsrGraph::Impl> require_graph(
      const std::shared_ptr<const NearFarCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument(
          "Near-Far shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const NearFarCsrGraph> shared_graph,
       hipStream_t stream_in,
       NearFarCsrWorkspaceOptions options)
      : graph(require_graph(shared_graph)),
        scratch(graph->adjacency.rows,
                select_shards(graph->adjacency,
                              options.queue_shards),
                select_urgent_capacity(
                    graph->adjacency,
                    options.urgent_queue_capacity),
                graph->adjacency.uses_compact_offsets),
        stream(stream_in),
        device(graph->device) {
    require_context(stream);
    initialize_epoch_storage();
    near_far_detail::wait_for_event(scratch.ready_event, stream);
  }

  static hipDeviceProp_t device_properties() {
    hipDeviceProp_t properties{};
    NEAR_FAR_HIP_CHECK(hipGetDeviceProperties(
        &properties, near_far_detail::current_hip_device()));
    if (properties.warpSize <= 0 ||
        properties.multiProcessorCount <= 0) {
      throw std::runtime_error(
          "HIP device reported invalid Near-Far scheduling properties");
    }
    return properties;
  }

  static int select_shards(const near_far_detail::DeviceCsrOwner& graph,
                           std::size_t requested) {
    return near_far_detail::choose_queue_shards(
        graph.rows, requested, device_properties());
  }

  static int select_urgent_capacity(
      const near_far_detail::DeviceCsrOwner& graph,
      std::size_t requested) {
    return near_far_detail::choose_urgent_capacity(
        graph.rows, graph.nnz, requested, device_properties());
  }

  void require_context(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "NearFarCsrWorkspace is stream-affine; use its construction stream");
    }
    if (near_far_detail::current_hip_device() != device) {
      throw std::invalid_argument(
          "NearFarCsrWorkspace is running on a different HIP device");
    }
  }

  void reset_queues() {
    using namespace near_far_detail;
    for (QueueStorage& queue : scratch.queues) {
      reset_queue_counts(queue, stream);
    }
  }

  void initialize_epoch_storage() {
    using namespace near_far_detail;
    NEAR_FAR_HIP_CHECK(hipMemsetAsync(
        scratch.epoch_queue.get(),
        0,
        static_cast<std::size_t>(scratch.rows) * sizeof(unsigned int),
        stream));
    NEAR_FAR_HIP_CHECK(hipMemsetAsync(
        scratch.locks.get(),
        0,
        static_cast<std::size_t>(scratch.rows) * sizeof(unsigned int),
        stream));
    NEAR_FAR_HIP_CHECK(hipMemsetAsync(
        scratch.scheduler_status.get(), 0, sizeof(SchedulerStatus), stream));
    if (scratch.telemetry.get() != nullptr) {
      NEAR_FAR_HIP_CHECK(hipMemsetAsync(
          scratch.telemetry.get(), 0, sizeof(DeviceTelemetry), stream));
    }
    reset_queues();
  }

  void advance_epoch() {
    using namespace near_far_detail;
    const bool force_wrap =
        g_near_far_force_epoch_wrap.exchange(false,
                                             std::memory_order_relaxed);
    if (force_wrap || run_epoch >= kMaxRunEpoch) {
      initialize_epoch_storage();
      run_epoch = 0;
    }
    ++run_epoch;
  }

  near_far_detail::DeviceQueueView queue_view(int queue_index) {
    return scratch.queues[queue_index].view(
        scratch.epoch_queue.get(),
        scratch.scheduler_status.get(),
        run_epoch,
        scratch.telemetry.get());
  }

  near_far_detail::QueueSetView queue_set_view() {
    near_far_detail::QueueSetView result;
    for (int queue = 0; queue < near_far_detail::kQueueCount; ++queue) {
      result.queues[queue] = queue_view(queue);
    }
    return result;
  }

  int largest_queue_shard_capacity() const {
    int capacity = 1;
    for (const near_far_detail::QueueStorage& queue : scratch.queues) {
      capacity = std::max(capacity, queue.shard_capacity);
    }
    return capacity;
  }

  void throw_on_device_error() const {
    if (scheduler_status.error != near_far_detail::kNoDeviceError) {
      throw std::runtime_error(
          std::string("Near-Far device scheduler failed: ") +
          near_far_detail::device_error_message(
              scheduler_status.error));
    }
  }

  void checkpoint(int target_count,
                  bool copy_changed,
                  bool force_minima = false) {
    using namespace near_far_detail;
    reset_checkpoint_status_kernel<<<1, 1, 0, stream>>>(
        scratch.scheduler_status.get(), false);
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    const bool collect_minima = target_count > 0 || force_minima;
    if (collect_minima) {
      collect_queue_status_kernel
          <<<dim3(grid_for_shard(largest_queue_shard_capacity()),
                   static_cast<unsigned int>(scratch.shards),
                   static_cast<unsigned int>(kQueueCount)),
             dim3(kBlockSize),
             0,
             stream>>>(queue_set_view(),
                       scratch.dist.get(),
                       scratch.version.get(),
                       scratch.processed_version.get(),
                       scratch.scheduler_status.get(),
                       true);
    } else {
      collect_queue_counts_kernel<<<1, kQueueCount, 0, stream>>>(
          queue_set_view(), scratch.scheduler_status.get());
    }
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    if (target_count > 0) {
      collect_target_bounds_kernel
          <<<grid_for_items(static_cast<std::size_t>(target_count)),
             kBlockSize,
             0,
             stream>>>(scratch.targets.get(),
                       target_count,
                       scratch.dist.get(),
                       scratch.epoch_queue.get(),
                       run_epoch,
                       scratch.scheduler_status.get());
      NEAR_FAR_HIP_CHECK(hipGetLastError());
    }

    const std::size_t copy_bytes =
        copy_changed ? sizeof(SchedulerStatus)
                     : offsetof(SchedulerStatus, changed);
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_scheduler_status.get(),
        scratch.scheduler_status.get(),
        copy_bytes,
        hipMemcpyDeviceToHost,
        stream));
    wait_for_event(scratch.ready_event, stream);
    std::memcpy(&scheduler_status,
                scratch.host_scheduler_status.get(),
                copy_bytes);
    g_near_far_status_copy_count.fetch_add(1,
                                           std::memory_order_relaxed);
    throw_on_device_error();
  }

  bool targets_are_settled(int target_count) const {
    if (target_count == 0 || scheduler_status.unreached_targets != 0) {
      return false;
    }
    const unsigned int no_minimum =
        std::numeric_limits<unsigned int>::max();
    return scheduler_status.min_pending_bits != no_minimum &&
           scheduler_status.min_pending_bits >
               scheduler_status.max_target_bits;
  }

  void prepare(const std::vector<int>& sources,
               const std::vector<int>* targets,
               bool exact_unit_controller) {
    using namespace near_far_detail;
    advance_epoch();
    scratch.ensure_source_capacity(sources.size());
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.sources.get(),
        sources.data(),
        sources.size() * sizeof(int),
        hipMemcpyHostToDevice,
        stream));
    if (targets != nullptr) {
      scratch.ensure_target_capacity(targets->size());
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          scratch.targets.get(),
          targets->data(),
          targets->size() * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
    }

    reset_queues();
    NEAR_FAR_HIP_CHECK(hipMemsetAsync(
        scratch.scheduler_status.get(), 0, sizeof(SchedulerStatus), stream));
    const int primary_index =
        exact_unit_controller ? kNearFarQueue : kCurrentQueue;
    const int spill_index = kNearFarQueue;
    if (scratch.compact_parents) {
      initialize_sources_kernel<true>
          <<<grid_for_items(sources.size()), kBlockSize, 0, stream>>>(
              scratch.sources.get(),
              static_cast<int>(sources.size()),
              scratch.dist.get(),
              scratch.owner_source.get(),
              scratch.hops.get(),
              scratch.compact_parent.get(),
              scratch.wide_pred_node.get(),
              scratch.wide_pred_edge.get(),
              scratch.version.get(),
              scratch.processed_version.get(),
              scratch.epoch_queue.get(),
              scratch.locks.get(),
              run_epoch,
              queue_view(primary_index),
              queue_view(spill_index));
    } else {
      initialize_sources_kernel<false>
          <<<grid_for_items(sources.size()), kBlockSize, 0, stream>>>(
              scratch.sources.get(),
              static_cast<int>(sources.size()),
              scratch.dist.get(),
              scratch.owner_source.get(),
              scratch.hops.get(),
              scratch.compact_parent.get(),
              scratch.wide_pred_node.get(),
              scratch.wide_pred_edge.get(),
              scratch.version.get(),
              scratch.processed_version.get(),
              scratch.epoch_queue.get(),
              scratch.locks.get(),
              run_epoch,
              queue_view(primary_index),
              queue_view(spill_index));
    }
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    checkpoint(targets == nullptr ? 0 : static_cast<int>(targets->size()),
               false);
  }

  template <typename RowOffset,
            bool UnitWeights,
            bool HasVertexCosts,
            bool CompactParents,
            bool TrackChanged>
  void launch_current_expansion_typed(float threshold) {
    using namespace near_far_detail;
    QueueStorage& current = scratch.queues[kCurrentQueue];
    const RowOffset* rowptr = nullptr;
    if constexpr (std::is_same<RowOffset, CompactOffset>::value) {
      rowptr = graph->adjacency.compact_rowptr.get();
    } else {
      rowptr = graph->adjacency.wide_rowptr.get();
    }
    expand_queue_kernel<RowOffset,
                        UnitWeights,
                        HasVertexCosts,
                        CompactParents,
                        TrackChanged>
        <<<dim3(grid_for_shard(current.shard_capacity),
                 static_cast<unsigned int>(current.shards)),
           dim3(kBlockSize),
           0,
           stream>>>(queue_view(kCurrentQueue),
                     rowptr,
                     graph->adjacency.colind.get(),
                     graph->adjacency.values.get(),
                     HasVertexCosts ? vertex_costs.get() : nullptr,
                     threshold,
                     scratch.dist.get(),
                     scratch.owner_source.get(),
                     scratch.hops.get(),
                     scratch.compact_parent.get(),
                     scratch.wide_pred_node.get(),
                     scratch.wide_pred_edge.get(),
                     scratch.version.get(),
                     scratch.processed_version.get(),
                     scratch.locks.get(),
                     queue_view(kNextQueue),
                     queue_view(kNearFarQueue),
                     queue_view(kFarQueue));
    NEAR_FAR_HIP_CHECK(hipGetLastError());
  }

  template <bool TrackChanged>
  void launch_current_expansion(float threshold) {
    if (graph->adjacency.uses_compact_offsets) {
      if (graph->adjacency.unit_weights) {
        if (has_vertex_costs) {
          launch_current_expansion_typed<
              near_far_detail::CompactOffset, true, true, true, TrackChanged>(
              threshold);
        } else {
          launch_current_expansion_typed<
              near_far_detail::CompactOffset, true, false, true, TrackChanged>(
              threshold);
        }
      } else if (has_vertex_costs) {
        launch_current_expansion_typed<
            near_far_detail::CompactOffset, false, true, true, TrackChanged>(
            threshold);
      } else {
        launch_current_expansion_typed<
            near_far_detail::CompactOffset, false, false, true, TrackChanged>(
            threshold);
      }
    } else if (graph->adjacency.unit_weights) {
      if (has_vertex_costs) {
        launch_current_expansion_typed<
            near_far_detail::Offset, true, true, false, TrackChanged>(
            threshold);
      } else {
        launch_current_expansion_typed<
            near_far_detail::Offset, true, false, false, TrackChanged>(
            threshold);
      }
    } else if (has_vertex_costs) {
      launch_current_expansion_typed<
          near_far_detail::Offset, false, true, false, TrackChanged>(
          threshold);
    } else {
      launch_current_expansion_typed<
          near_far_detail::Offset, false, false, false, TrackChanged>(
          threshold);
    }
  }

  bool process_current(float threshold,
                       int target_count,
                       NearFarCsrProgressCallback callback,
                       void* callback_data,
                       int iteration,
                       int max_iters) {
    using namespace near_far_detail;
    if (callback != nullptr) {
      reset_checkpoint_status_kernel<<<1, 1, 0, stream>>>(
          scratch.scheduler_status.get(), true);
      NEAR_FAR_HIP_CHECK(hipGetLastError());
      launch_current_expansion<true>(threshold);
    } else {
      launch_current_expansion<false>(threshold);
    }
    reset_queue_counts(scratch.queues[kCurrentQueue], stream);
    checkpoint(target_count, callback != nullptr);
    const bool changed =
        callback != nullptr && scheduler_status.changed != 0;
    if (callback != nullptr) {
      callback(NearFarCsrProgress{
                   iteration,
                   max_iters,
                   true,
                   changed},
               callback_data);
    }
    return changed;
  }

  void promote_queue(int input_index,
                     int deferred_output_index,
                     float threshold,
                     int target_count) {
    using namespace near_far_detail;
    QueueStorage& input = scratch.queues[input_index];
    QueueStorage& deferred_output =
        scratch.queues[deferred_output_index];
    QueueStorage& scratch_output = scratch.queues[kScratchQueue];
    if (scheduler_status.queue_totals[kScratchQueue] != 0) {
      throw std::logic_error(
          "Near-Far deferred scratch queue was not empty");
    }

    const DeviceQueueView input_view = queue_view(input_index);
    DeviceQueueView urgent = queue_view(kCurrentQueue);
    DeviceQueueView near_far =
        (deferred_output_index == kNearFarQueue
             ? scratch_output
             : scratch.queues[kNearFarQueue])
            .view(scratch.epoch_queue.get(),
                  scratch.scheduler_status.get(),
                  run_epoch,
                  scratch.telemetry.get());
    DeviceQueueView far =
        (deferred_output_index == kFarQueue
             ? scratch_output
             : scratch.queues[kFarQueue])
            .view(scratch.epoch_queue.get(),
                  scratch.scheduler_status.get(),
                  run_epoch,
                  scratch.telemetry.get());

    reclassify_queue_kernel
        <<<dim3(grid_for_shard(input.shard_capacity),
                 static_cast<unsigned int>(input.shards)),
           dim3(kBlockSize),
           0,
           stream>>>(input_view,
                     threshold,
                     scratch.dist.get(),
                     scratch.version.get(),
                     scratch.processed_version.get(),
                     urgent,
                     near_far,
                     far);
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    reset_queue_counts(input, stream);
    std::swap(deferred_output, scratch_output);
    reset_queue_counts(scratch_output, stream);
    checkpoint(target_count, false);
  }

  float minimum_queue_distance(int queue_index) {
    const unsigned int bits =
        scheduler_status.queue_min_bits[queue_index];
    if (bits == std::numeric_limits<unsigned int>::max()) {
      return std::numeric_limits<float>::infinity();
    }
    return near_far_detail::float_from_bits(bits);
  }

  void clear_queue(int queue_index, int target_count) {
    using namespace near_far_detail;
    QueueStorage& queue = scratch.queues[queue_index];
    clear_queue_membership_kernel
        <<<dim3(grid_for_shard(queue.shard_capacity),
                 static_cast<unsigned int>(queue.shards)),
           dim3(kBlockSize),
           0,
           stream>>>(queue_view(queue_index));
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    reset_queue_counts(queue, stream);
    checkpoint(target_count, false);
  }

  void swap_queue_roles(int destination, int source) {
    using namespace near_far_detail;
    std::swap(scratch.queues[destination], scratch.queues[source]);
    std::swap(scheduler_status.queue_totals[destination],
              scheduler_status.queue_totals[source]);
    std::swap(scheduler_status.queue_min_bits[destination],
              scheduler_status.queue_min_bits[source]);
    reset_queue_counts(scratch.queues[source], stream);
    scheduler_status.queue_totals[source] = 0;
    scheduler_status.queue_min_bits[source] =
        std::numeric_limits<unsigned int>::max();
  }

  template <typename RowOffset, bool CompactParents>
  void launch_exact_unit_expansion_typed(int input,
                                         int next,
                                         int retry) {
    using namespace near_far_detail;
    QueueStorage& input_queue = scratch.queues[input];
    const RowOffset* rowptr = nullptr;
    if constexpr (std::is_same<RowOffset, CompactOffset>::value) {
      rowptr = graph->adjacency.compact_rowptr.get();
    } else {
      rowptr = graph->adjacency.wide_rowptr.get();
    }
    expand_exact_unit_kernel<RowOffset, CompactParents>
        <<<dim3(grid_for_shard(input_queue.shard_capacity),
                 static_cast<unsigned int>(input_queue.shards)),
           dim3(kBlockSize),
           0,
           stream>>>(queue_view(input),
                     rowptr,
                     graph->adjacency.colind.get(),
                     scratch.dist.get(),
                     scratch.owner_source.get(),
                     scratch.hops.get(),
                     scratch.compact_parent.get(),
                     scratch.wide_pred_node.get(),
                     scratch.wide_pred_edge.get(),
                     scratch.version.get(),
                     scratch.processed_version.get(),
                     scratch.locks.get(),
                     queue_view(next),
                     queue_view(retry));
    NEAR_FAR_HIP_CHECK(hipGetLastError());
  }

  void launch_exact_unit_expansion(int input, int next, int retry) {
    if (graph->adjacency.uses_compact_offsets) {
      launch_exact_unit_expansion_typed<
          near_far_detail::CompactOffset, true>(input, next, retry);
    } else {
      launch_exact_unit_expansion_typed<
          near_far_detail::Offset, false>(input, next, retry);
    }
  }

  RunOutcome run_exact_unit_scheduler(
      const std::vector<int>& sources,
      const std::vector<int>* targets) {
    using namespace near_far_detail;
    prepare(sources, targets, true);
    g_near_far_unit_controller_count.fetch_add(
        1, std::memory_order_relaxed);
    const int target_count =
        targets == nullptr ? 0 : static_cast<int>(targets->size());
    constexpr int input = kNearFarQueue;
    constexpr int next = kFarQueue;
    constexpr int retry = kScratchQueue;
    RunOutcome outcome;

    while (true) {
      if (scheduler_status.queue_totals[input] == 0) {
        if (scheduler_status.queue_totals[retry] != 0) {
          swap_queue_roles(input, retry);
          continue;
        }
        if (scheduler_status.queue_totals[next] != 0) {
          swap_queue_roles(input, next);
          continue;
        }
        outcome.converged = true;
        break;
      }

      ++outcome.iterations;
      launch_exact_unit_expansion(input, next, retry);
      reset_queue_counts(scratch.queues[input], stream);
      checkpoint(target_count, false);
      if (target_count != 0 &&
          scheduler_status.queue_totals[retry] == 0 &&
          targets_are_settled(target_count)) {
        outcome.stopped_on_target = true;
        break;
      }
    }
    return outcome;
  }

  template <typename RowOffset, bool CompactParents>
  void launch_measure_target_paths(int target_count,
                                   bool paths_certified) {
    using namespace near_far_detail;
    const RowOffset* rowptr = nullptr;
    if constexpr (std::is_same<RowOffset, CompactOffset>::value) {
      rowptr = graph->adjacency.compact_rowptr.get();
    } else {
      rowptr = graph->adjacency.wide_rowptr.get();
    }
    measure_target_paths_kernel<RowOffset, CompactParents>
        <<<grid_for_items(static_cast<std::size_t>(target_count)),
           kBlockSize,
           0,
           stream>>>(scratch.targets.get(),
                     target_count,
                     scratch.rows,
                     rowptr,
                     graph->adjacency.colind.get(),
                     scratch.dist.get(),
                     scratch.owner_source.get(),
                     scratch.hops.get(),
                     scratch.compact_parent.get(),
                     scratch.wide_pred_node.get(),
                     scratch.wide_pred_edge.get(),
                     scratch.epoch_queue.get(),
                     run_epoch,
                     paths_certified,
                     scratch.target_distances.get(),
                     scratch.target_sources.get(),
                     scratch.target_path_lengths.get(),
                     scratch.target_path_status.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
  }

  template <typename RowOffset, bool CompactParents>
  void launch_fill_target_paths(int target_count) {
    using namespace near_far_detail;
    const RowOffset* rowptr = nullptr;
    if constexpr (std::is_same<RowOffset, CompactOffset>::value) {
      rowptr = graph->adjacency.compact_rowptr.get();
    } else {
      rowptr = graph->adjacency.wide_rowptr.get();
    }
    fill_target_paths_kernel<RowOffset, CompactParents>
        <<<grid_for_items(static_cast<std::size_t>(target_count)),
           kBlockSize,
           0,
           stream>>>(scratch.targets.get(),
                     target_count,
                     scratch.rows,
                     rowptr,
                     graph->adjacency.colind.get(),
                     scratch.owner_source.get(),
                     scratch.hops.get(),
                     scratch.compact_parent.get(),
                     scratch.wide_pred_node.get(),
                     scratch.wide_pred_edge.get(),
                     scratch.epoch_queue.get(),
                     run_epoch,
                     scratch.target_path_lengths.get(),
                     scratch.target_path_status.get(),
                     scratch.target_node_offsets.get(),
                     scratch.target_edge_offsets.get(),
                     scratch.compact_path_nodes.get(),
                     scratch.compact_path_edges.get(),
                     scratch.wide_path_edges.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
  }

  NearFarCsrResult extract_targets(const std::vector<int>& targets,
                                   bool paths_certified) {
    using namespace near_far_detail;
    const int target_count = static_cast<int>(targets.size());
    if (scratch.compact_parents) {
      launch_measure_target_paths<CompactOffset, true>(
          target_count, paths_certified);
    } else {
      launch_measure_target_paths<Offset, false>(
          target_count, paths_certified);
    }
    scan_target_offsets_kernel<<<1, 1, 0, stream>>>(
        target_count,
        scratch.target_distances.get(),
        scratch.target_path_lengths.get(),
        scratch.target_path_status.get(),
        scratch.target_node_offsets.get(),
        scratch.target_edge_offsets.get(),
        scratch.path_totals.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_path_totals.get(),
        scratch.path_totals.get(),
        sizeof(PathTotals),
        hipMemcpyDeviceToHost,
        stream));
    wait_for_event(scratch.ready_event, stream);

    const PathTotals totals = *scratch.host_path_totals.get();
    if (totals.error != 0 || totals.node_count < 0 ||
        totals.edge_count < 0) {
      throw std::overflow_error(
          "Near-Far compact paths exceed int offsets");
    }
    const std::size_t total_nodes =
        static_cast<std::size_t>(totals.node_count);
    const std::size_t total_edges =
        static_cast<std::size_t>(totals.edge_count);
    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    if (total_nodes != 0) {
      if (scratch.compact_parents) {
        launch_fill_target_paths<CompactOffset, true>(target_count);
      } else {
        launch_fill_target_paths<Offset, false>(target_count);
      }
    }

    const std::size_t target_bytes =
        static_cast<std::size_t>(target_count);
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_target_distances.get(),
        scratch.target_distances.get(),
        target_bytes * sizeof(float),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_target_sources.get(),
        scratch.target_sources.get(),
        target_bytes * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_target_path_status.get(),
        scratch.target_path_status.get(),
        target_bytes * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_target_node_offsets.get(),
        scratch.target_node_offsets.get(),
        (target_bytes + 1) * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.host_target_edge_offsets.get(),
        scratch.target_edge_offsets.get(),
        (target_bytes + 1) * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    if (total_nodes != 0) {
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          scratch.host_compact_path_nodes.get(),
          scratch.compact_path_nodes.get(),
          total_nodes * sizeof(int),
          hipMemcpyDeviceToHost,
          stream));
    }
    if (total_edges != 0) {
      if (scratch.compact_parents) {
        NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
            scratch.host_compact_path_edges.get(),
            scratch.compact_path_edges.get(),
            total_edges * sizeof(CompactOffset),
            hipMemcpyDeviceToHost,
            stream));
      } else {
        NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
            scratch.host_wide_path_edges.get(),
            scratch.wide_path_edges.get(),
            total_edges * sizeof(Offset),
            hipMemcpyDeviceToHost,
            stream));
      }
    }
    wait_for_event(scratch.ready_event, stream);
    g_near_far_path_transfer_count.fetch_add(
        1, std::memory_order_relaxed);

    NearFarCsrResult result;
    result.target_distances.assign(
        scratch.host_target_distances.get(),
        scratch.host_target_distances.get() + target_count);
    result.target_sources.assign(
        scratch.host_target_sources.get(),
        scratch.host_target_sources.get() + target_count);
    result.target_path_offsets.assign(
        scratch.host_target_node_offsets.get(),
        scratch.host_target_node_offsets.get() + target_count + 1);
    result.target_edge_offsets.assign(
        scratch.host_target_edge_offsets.get(),
        scratch.host_target_edge_offsets.get() + target_count + 1);
    if (total_nodes != 0) {
      result.target_path_nodes.assign(
          scratch.host_compact_path_nodes.get(),
          scratch.host_compact_path_nodes.get() + total_nodes);
    }
    result.target_path_edges.resize(total_edges);
    for (std::size_t edge = 0; edge < total_edges; ++edge) {
      result.target_path_edges[edge] =
          scratch.compact_parents
              ? static_cast<Offset>(
                    scratch.host_compact_path_edges.get()[edge])
              : scratch.host_wide_path_edges.get()[edge];
    }
    for (int index = 0; index < target_count; ++index) {
      const int reserved_nodes =
          result.target_path_offsets[static_cast<std::size_t>(index) + 1] -
          result.target_path_offsets[static_cast<std::size_t>(index)];
      if (reserved_nodes > 0 &&
          scratch.host_target_path_status.get()[index] !=
              kTargetPathValid) {
        throw std::runtime_error(
            "Near-Far predecessor path failed device validation");
      }
    }
    result.target_reached = totals.all_reached != 0;
    return result;
  }

  RunOutcome run_generic_scheduler(
      const std::vector<int>& sources,
      const std::vector<int>* targets,
      float delta,
      int max_iters,
      NearFarCsrProgressCallback callback,
      void* callback_data) {
    using namespace near_far_detail;
    prepare(sources, targets, false);
    const int target_count =
        targets == nullptr ? 0 : static_cast<int>(targets->size());
    float threshold = delta;
    RunOutcome outcome;

    while (true) {
      if (scheduler_status.queue_totals[kCurrentQueue] == 0) {
        if (scheduler_status.queue_totals[kNextQueue] != 0) {
          swap_queue_roles(kCurrentQueue, kNextQueue);
          continue;
        }

        if (scheduler_status.queue_totals[kNearFarQueue] != 0) {
          promote_queue(
              kNearFarQueue, kNearFarQueue, threshold, target_count);
          continue;
        }

        if (scheduler_status.queue_totals[kFarQueue] != 0) {
          if (target_count == 0 &&
              scheduler_status.queue_min_bits[kFarQueue] ==
                  std::numeric_limits<unsigned int>::max()) {
            checkpoint(0, false, true);
          }
          const float minimum = minimum_queue_distance(kFarQueue);
          if (std::isfinite(minimum)) {
            threshold = next_near_far_threshold(
                threshold, delta, minimum);
            promote_queue(kFarQueue,
                          kFarQueue,
                          threshold,
                          target_count);
            continue;
          } else {
            clear_queue(kFarQueue, target_count);
          }
        }

        outcome.converged = true;
        break;
      }

      if (max_iters >= 0 && outcome.iterations >= max_iters) {
        break;
      }
      ++outcome.iterations;
      (void)process_current(
          threshold,
          target_count,
          callback,
          callback_data,
          outcome.iterations,
          max_iters);

      if (targets_are_settled(target_count)) {
        outcome.stopped_on_target = true;
        break;
      }
    }
    return outcome;
  }

  NearFarCsrResult extract_full_result() {
    using namespace near_far_detail;
    NearFarCsrResult result;
    const std::size_t rows = static_cast<std::size_t>(scratch.rows);
    result.dist.resize(rows);
    result.pred_node.resize(rows);
    result.pred_edge.resize(rows);
    if (scratch.compact_parents) {
      materialize_full_result_kernel<true>
          <<<grid_for_items(rows), kBlockSize, 0, stream>>>(
              scratch.rows,
              run_epoch,
              scratch.epoch_queue.get(),
              scratch.dist.get(),
              scratch.compact_parent.get(),
              scratch.wide_pred_node.get(),
              scratch.wide_pred_edge.get());
    } else {
      materialize_full_result_kernel<false>
          <<<grid_for_items(rows), kBlockSize, 0, stream>>>(
              scratch.rows,
              run_epoch,
              scratch.epoch_queue.get(),
              scratch.dist.get(),
              scratch.compact_parent.get(),
              scratch.wide_pred_node.get(),
              scratch.wide_pred_edge.get());
    }
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        result.dist.data(),
        scratch.dist.get(),
        rows * sizeof(float),
        hipMemcpyDeviceToHost,
        stream));
    if (scratch.compact_parents) {
      std::vector<unsigned long long> packed_parent(rows);
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          packed_parent.data(),
          scratch.compact_parent.get(),
          rows * sizeof(unsigned long long),
          hipMemcpyDeviceToHost,
          stream));
      wait_for_event(scratch.ready_event, stream);
      for (std::size_t vertex = 0; vertex < rows; ++vertex) {
        const unsigned long long packed = packed_parent[vertex];
        const std::uint32_t node =
            static_cast<std::uint32_t>(packed >> 32);
        result.pred_node[vertex] =
            node == std::numeric_limits<std::uint32_t>::max()
                ? -1
                : static_cast<int>(node);
        const CompactOffset edge = static_cast<CompactOffset>(packed);
        result.pred_edge[vertex] =
            edge == kNoCompactEdge ? static_cast<Offset>(-1)
                                   : static_cast<Offset>(edge);
      }
    } else {
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.pred_node.data(),
          scratch.wide_pred_node.get(),
          rows * sizeof(int),
          hipMemcpyDeviceToHost,
          stream));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.pred_edge.data(),
          scratch.wide_pred_edge.get(),
          rows * sizeof(Offset),
          hipMemcpyDeviceToHost,
          stream));
      wait_for_event(scratch.ready_event, stream);
    }
    result.target_reached = true;
    return result;
  }

  static NearFarCsrResult fan_out_target_result(
      const NearFarCsrResult& unique,
      const std::vector<int>& original_to_unique) {
    NearFarCsrResult result;
    const std::size_t count = original_to_unique.size();
    result.target_distances.resize(count);
    result.target_sources.resize(count);
    result.target_path_offsets.resize(count + 1, 0);
    result.target_edge_offsets.resize(count + 1, 0);
    for (std::size_t index = 0; index < count; ++index) {
      const std::size_t unique_index =
          static_cast<std::size_t>(original_to_unique[index]);
      result.target_distances[index] =
          unique.target_distances[unique_index];
      result.target_sources[index] =
          unique.target_sources[unique_index];
      const int node_begin = unique.target_path_offsets[unique_index];
      const int node_end = unique.target_path_offsets[unique_index + 1];
      const int edge_begin = unique.target_edge_offsets[unique_index];
      const int edge_end = unique.target_edge_offsets[unique_index + 1];
      const std::size_t new_nodes =
          result.target_path_nodes.size() +
          static_cast<std::size_t>(node_end - node_begin);
      const std::size_t new_edges =
          result.target_path_edges.size() +
          static_cast<std::size_t>(edge_end - edge_begin);
      if (new_nodes >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          new_edges >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Near-Far duplicate target paths exceed int offsets");
      }
      result.target_path_nodes.insert(
          result.target_path_nodes.end(),
          unique.target_path_nodes.begin() + node_begin,
          unique.target_path_nodes.begin() + node_end);
      result.target_path_edges.insert(
          result.target_path_edges.end(),
          unique.target_path_edges.begin() + edge_begin,
          unique.target_path_edges.begin() + edge_end);
      result.target_path_offsets[index + 1] =
          static_cast<int>(new_nodes);
      result.target_edge_offsets[index + 1] =
          static_cast<int>(new_edges);
    }
    result.target_reached = unique.target_reached;
    return result;
  }

  NearFarCsrResult run_impl(
      const std::vector<int>& sources,
      const std::vector<int>* targets,
      float delta,
      int max_iters,
      NearFarCsrProgressCallback callback,
      void* callback_data) {
    std::vector<int> unique_targets;
    std::vector<int> original_to_unique;
    const std::vector<int>* device_targets = nullptr;
    if (targets != nullptr) {
      unique_targets.reserve(targets->size());
      original_to_unique.reserve(targets->size());
      std::unordered_map<int, int> unique_indices;
      unique_indices.reserve(targets->size());
      for (const int target : *targets) {
        const auto insertion = unique_indices.emplace(
            target, static_cast<int>(unique_targets.size()));
        if (insertion.second) unique_targets.push_back(target);
        original_to_unique.push_back(insertion.first->second);
      }
      device_targets = &unique_targets;
    }

    const bool unit_candidate =
        graph->adjacency.unit_weights && !has_vertex_costs;
    const bool use_unit_controller =
        unit_candidate && max_iters == -1 && callback == nullptr &&
        !g_near_far_force_generic.load(std::memory_order_relaxed);
    RunOutcome outcome;
    if (use_unit_controller) {
      outcome = run_exact_unit_scheduler(sources, device_targets);
    } else {
      if (unit_candidate) {
        g_near_far_controller_fallback_count.fetch_add(
            1, std::memory_order_relaxed);
      }
      outcome = run_generic_scheduler(sources,
                                      device_targets,
                                      delta,
                                      max_iters,
                                      callback,
                                      callback_data);
    }

    NearFarCsrResult result;
    if (device_targets == nullptr) {
      result = extract_full_result();
    } else {
      NearFarCsrResult unique_result = extract_targets(
          *device_targets,
          outcome.converged || outcome.stopped_on_target);
      result = fan_out_target_result(unique_result, original_to_unique);
    }
    result.iterations_used = outcome.iterations;
    result.converged = outcome.converged;
    result.stopped_on_target = outcome.stopped_on_target;
    return result;
  }
};

NearFarCsrGraph::NearFarCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("near_far.upload_graph");
  near_far_detail::validate_host_csr(adjacency);
  impl_ = std::make_shared<Impl>(adjacency, stream);
}

NearFarCsrGraph::~NearFarCsrGraph() = default;
NearFarCsrGraph::NearFarCsrGraph(NearFarCsrGraph&&) noexcept = default;
NearFarCsrGraph& NearFarCsrGraph::operator=(NearFarCsrGraph&&) noexcept =
    default;

NearFarCsrWorkspace::NearFarCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : NearFarCsrWorkspace(
          adjacency, stream, NearFarCsrWorkspaceOptions{}) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    NearFarCsrWorkspaceOptions options)
    : NearFarCsrWorkspace(
          std::make_shared<NearFarCsrGraph>(adjacency, stream),
          stream,
          options) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    std::shared_ptr<const NearFarCsrGraph> adjacency,
    hipStream_t stream)
    : NearFarCsrWorkspace(
          std::move(adjacency),
          stream,
          NearFarCsrWorkspaceOptions{}) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    std::shared_ptr<const NearFarCsrGraph> adjacency,
    hipStream_t stream,
    NearFarCsrWorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(adjacency), stream, options)) {}

NearFarCsrWorkspace::~NearFarCsrWorkspace() = default;
NearFarCsrWorkspace::NearFarCsrWorkspace(
    NearFarCsrWorkspace&&) noexcept = default;
NearFarCsrWorkspace& NearFarCsrWorkspace::operator=(
    NearFarCsrWorkspace&&) noexcept = default;

void NearFarCsrWorkspace::update_vertex_costs(
    const std::vector<float>& costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("near_far.update_vertex_costs");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  if (costs.size() !=
      static_cast<std::size_t>(impl_->graph->adjacency.rows)) {
    throw std::invalid_argument(
        "Near-Far vertex cost count does not match graph rows");
  }
  for (const float cost : costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument(
          "Near-Far vertex costs must be finite and nonnegative");
    }
  }
  if (impl_->vertex_costs.size() < costs.size()) {
    impl_->vertex_costs.reset(costs.size());
  }
  NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
      impl_->vertex_costs.get(),
      costs.data(),
      costs.size() * sizeof(float),
      hipMemcpyHostToDevice,
      stream));
  near_far_detail::wait_for_event(impl_->scratch.ready_event, stream);
  impl_->has_vertex_costs = true;
}

void NearFarCsrWorkspace::clear_vertex_costs(hipStream_t stream) {
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  impl_->has_vertex_costs = false;
}

NearFarCsrResult NearFarCsrWorkspace::run_distances(
    const std::vector<int>& sources,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("near_far.run_distances");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  near_far_detail::validate_sources(
      impl_->graph->adjacency.rows, sources);
  near_far_detail::validate_delta(delta);
  if (max_iters < -1) {
    throw std::invalid_argument(
        "Near-Far max iterations must be -1 or nonnegative");
  }
  return impl_->run_impl(sources,
                         nullptr,
                         delta,
                         max_iters,
                         progress_callback,
                         progress_user_data);
}

NearFarCsrResult NearFarCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("near_far.run");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  near_far_detail::validate_sources(
      impl_->graph->adjacency.rows, sources);
  near_far_detail::validate_targets(
      impl_->graph->adjacency.rows, targets);
  near_far_detail::validate_delta(delta);
  if (max_iters < -1) {
    throw std::invalid_argument(
        "Near-Far max iterations must be -1 or nonnegative");
  }
  NearFarCsrResult result = impl_->run_impl(
      sources,
      &targets,
      delta,
      max_iters,
      progress_callback,
      progress_user_data);
  result.target = -1;
  return result;
}

NearFarCsrResult NearFarCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  if (target < 0 ||
      static_cast<minplus_sparse::Offset>(target) >=
          impl_->graph->adjacency.rows) {
    throw std::out_of_range(
        "Near-Far target vertex is outside the CSR graph");
  }
  NearFarCsrResult result =
      run(sources,
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

NearFarCsrResult NearFarCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}
