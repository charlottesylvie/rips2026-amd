#include "delta_stepping_hip_CSR.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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
constexpr int kMaxHostCheckedReductionBlocks = 256;
constexpr int kNoBucket = 0x3fffffff;
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
    release();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return hipErrorOutOfMemory;
    }
    if (count == 0) {
      return hipSuccess;
    }
    T* candidate = nullptr;
    const hipError_t status =
        hipMalloc(reinterpret_cast<void**>(&candidate), count * sizeof(T));
    if (status == hipSuccess) {
      ptr_ = candidate;
      count_ = count;
    }
    return status;
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
  explicit PinnedHostBuffer(std::size_t count) {
    if (count != 0) {
      DS_DELTA_HIP_CHECK(
          hipHostMalloc(reinterpret_cast<void**>(&ptr_),
                        sssp_capacity::checked_bytes<T>(count),
                        hipHostMallocDefault));
    }
  }

  ~PinnedHostBuffer() {
    if (ptr_ != nullptr) {
      (void)hipHostFree(ptr_);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

template <typename RowOffset>
struct DeviceCsrView {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const RowOffset* rowptr = nullptr;
  const Index* colind = nullptr;
  const float* values = nullptr;
};

inline DeviceCsrView<Offset> wide_device_csr_view(
    const minplus_sparse::DeviceCsrF32& graph) {
  return {graph.rows, graph.cols, graph.nnz, graph.rowptr, graph.colind,
          graph.values};
}

struct DeviceCsrOwner {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  bool uses_32_bit_offsets = false;
  DeviceBuffer<CompactRowOffset> rowptr32;
  DeviceBuffer<Offset> rowptr64;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  DeviceBuffer<std::uint32_t> edge_source;
  // Kept separate from edge_source.get(): an eligible empty graph has a
  // deliberately null zero-length map.
  bool edge_source_available = false;

  DeviceCsrOwner() = default;
  DeviceCsrOwner(Offset rows_,
                 Offset cols_,
                 Offset nnz_,
                 bool uses_32_bit_offsets_)
      : rows(rows_),
        cols(cols_),
        nnz(nnz_),
        uses_32_bit_offsets(uses_32_bit_offsets_),
        rowptr32(uses_32_bit_offsets
                     ? sssp_capacity::checked_target_offset_count(
                           static_cast<std::size_t>(rows))
                     : 0),
        rowptr64(uses_32_bit_offsets
                     ? 0
                     : sssp_capacity::checked_target_offset_count(
                           static_cast<std::size_t>(rows))),
        colind(static_cast<std::size_t>(nnz)),
        values(static_cast<std::size_t>(nnz)) {}

  template <typename RowOffset>
  DeviceCsrView<RowOffset> view() const {
    if constexpr (std::is_same<RowOffset, CompactRowOffset>::value) {
      if (!uses_32_bit_offsets) {
        throw std::logic_error(
            "requested compact view of wide Delta-Stepping CSR");
      }
      return {rows, cols, nnz, rowptr32.get(), colind.get(), values.get()};
    } else {
      static_assert(std::is_same<RowOffset, Offset>::value,
                    "unsupported Delta-Stepping row offset type");
      if (uses_32_bit_offsets) {
        throw std::logic_error(
            "requested wide view of compact Delta-Stepping CSR");
      }
      return {rows, cols, nnz, rowptr64.get(), colind.get(), values.get()};
    }
  }
};

// The public 64-byte descriptor is the only state copied back after a bounded
// cooperative-controller launch.  Queue parity and phase-local counters stay
// device-resident across publications so a host check never has to rebuild
// controller state from independently copied scalars.
struct CooperativeDeltaControllerState {
  DeltaSteppingCsrControllerDescriptor descriptor{};
  std::uint32_t current_queue_parity = 0;
  std::uint32_t pending_queue_parity = 0;
  std::uint32_t generation_cursor = 0;
  std::uint32_t reserved = 0;
  unsigned long long grid_barriers = 0;
};

static_assert(std::is_standard_layout<CooperativeDeltaControllerState>::value,
              "cooperative Delta controller state must have standard layout");
static_assert(std::is_trivially_copyable<
                  CooperativeDeltaControllerState>::value,
              "cooperative Delta controller state must be trivially copyable");
static_assert(offsetof(CooperativeDeltaControllerState, descriptor) == 0,
              "the published Delta controller descriptor must be first");

struct CooperativeLaunchConfiguration {
  bool initialized = false;
  int max_blocks = 0;
  int active_blocks_per_compute_unit = 0;
  int compute_units = 0;
  int selected_blocks_per_compute_unit = 0;
  DeltaSteppingCsrControllerFallbackReason fallback_reason =
      DeltaSteppingCsrControllerFallbackReason::kNone;
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
  DeviceBuffer<CooperativeDeltaControllerState> controller_state;
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
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  PinnedHostBuffer<int> host_scalar;
  PinnedHostBuffer<int> host_unit_status;
  // Persistent pinned staging for the host-checked controller's per-block
  // pending-bucket minima.  It outlives every asynchronous copy that uses it.
  std::unique_ptr<PinnedHostBuffer<int>> host_pending_bucket_block_mins;
  // One workspace-owned pinned record stages both the initial H2D state and
  // each descriptor D2H publication. It outlives every asynchronous transfer
  // on the workspace stream, including exception cleanup.
  std::unique_ptr<PinnedHostBuffer<CooperativeDeltaControllerState>>
      host_controller_state;
  bool unit_initialized = false;
  bool generic_initialized = false;
  bool parent_key_initialized = false;
  bool legacy_predecessors_initialized = false;
  std::uint32_t current_generation = 0;
  std::uint32_t controller_query_sequence = 0;
  std::uint32_t controller_concurrency_hint = 1;
  std::shared_ptr<DeltaSteppingCsrBatchCoordinator>
      controller_batch_coordinator;
  // Each bit selects one compile-time controller trait: current-generation,
  // parent tracking, compact edge parent, telemetry, and physical multi-query
  // batching. A workspace is bound to one immutable graph/device, so both
  // successful and unsupported occupancy results remain valid for its
  // lifetime.
  std::array<CooperativeLaunchConfiguration, 32>
      cooperative_launch_configurations{};

  DeltaSteppingScratch() = default;
  explicit DeltaSteppingScratch(Offset rows_)
      : rows(rows_),
        dist(static_cast<std::size_t>(rows_)),
        in_pending(static_cast<std::size_t>(rows_)),
        current_queue(static_cast<std::size_t>(rows_)),
        unit_status(kUnitStatusCount),
        host_scalar(1),
        host_unit_status(kUnitStatusCount) {}

  void ensure_legacy_predecessor_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    if (pred_node.size() < vertex_count) {
      pred_node.reset(vertex_count);
    }
    if (pred_edge.size() < vertex_count) {
      pred_edge.reset(vertex_count);
    }
  }

  void ensure_generic_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    auto ensure_vertices = [vertex_count](auto& buffer) {
      if (buffer.size() < vertex_count) buffer.reset(vertex_count);
    };
    auto ensure_scalar = [](auto& buffer) {
      if (buffer.size() == 0) buffer.reset(1);
    };
    ensure_vertices(in_current);
    ensure_vertices(in_heavy);
    ensure_vertices(next_queue);
    ensure_vertices(pending_a);
    ensure_vertices(pending_b);
    ensure_vertices(touched_queue);
    ensure_vertices(heavy_queue);
    ensure_scalar(current_count);
    ensure_scalar(next_count);
    ensure_scalar(pending_count);
    ensure_scalar(new_pending_count);
    ensure_scalar(heavy_count);
    ensure_scalar(touched_count);
    ensure_scalar(settled_target_count);
    ensure_scalar(min_pending_bucket);
  }

  void ensure_host_checked_reduction_storage() {
    if (!host_pending_bucket_block_mins) {
      host_pending_bucket_block_mins =
          std::make_unique<PinnedHostBuffer<int>>(
              kMaxHostCheckedReductionBlocks);
    }
  }

  void ensure_parent_key_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    if (parent_key.size() < vertex_count) {
      parent_key.reset(vertex_count);
      parent_key_initialized = false;
    }
  }

  void ensure_source_capacity(std::size_t source_count) {
    source_count = sssp_capacity::checked_device_count(source_count);
    if (sources.size() < source_count) {
      sources.reset(
          delta_stepping_device_geometric_capacity(sources.size(),
                                                   source_count));
    }
  }

  void ensure_telemetry_storage() {
    if (telemetry_counters.size() < kTelemetryCounterCount) {
      telemetry_counters.reset(kTelemetryCounterCount);
    }
  }

  void ensure_controller_storage() {
    if (controller_state.size() == 0) {
      controller_state.reset(1);
    }
    if (!host_controller_state) {
      host_controller_state = std::make_unique<
          PinnedHostBuffer<CooperativeDeltaControllerState>>(1);
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    target_count = sssp_capacity::checked_device_count(target_count);
    const std::size_t capacity =
        delta_stepping_device_geometric_capacity(targets.size(),
                                                 target_count);
    if (targets.size() < target_count) {
      targets.reset(capacity);
    }
    if (target_settled.size() < target_count) {
      target_settled.reset(capacity);
    }
    if (target_distances.size() < target_count) {
      target_distances.reset(capacity);
    }
    if (target_path_lengths.size() < target_count) {
      target_path_lengths.reset(capacity);
    }
    if (target_sources.size() < target_count) {
      target_sources.reset(capacity);
    }
    if (target_path_status.size() < target_count) {
      target_path_status.reset(capacity);
    }
    const std::size_t required_offset_count =
        sssp_capacity::checked_target_offset_count(target_count);
    const std::size_t offset_capacity =
        sssp_capacity::checked_target_offset_count(capacity);
    if (target_node_offsets.size() < required_offset_count) {
      target_node_offsets.reset(offset_capacity);
    }
    if (target_edge_offsets.size() < required_offset_count) {
      target_edge_offsets.reset(offset_capacity);
    }
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count) {
    node_count = sssp_capacity::checked_device_count(node_count);
    edge_count = sssp_capacity::checked_device_count(edge_count);
    if (compact_path_nodes.size() < node_count) {
      compact_path_nodes.reset(
          delta_stepping_device_geometric_capacity(
              compact_path_nodes.size(), node_count));
    }
    if (compact_path_edges.size() < edge_count) {
      compact_path_edges.reset(
          delta_stepping_device_geometric_capacity(
              compact_path_edges.size(), edge_count));
    }
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
    compact_path_nodes.reset(0);
    compact_path_edges.reset(0);
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
  if (g.nnz > 0 && (g.colind == nullptr || g.values == nullptr)) {
    throw std::invalid_argument("device CSR colind/values are null for a nonempty graph");
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
  if (g.nnz > 0 && (g.colind == nullptr || g.values == nullptr)) {
    throw std::invalid_argument("device CSR colind/values are null for a nonempty graph");
  }
}

template <typename T>
inline T copy_scalar_to_host(const T* d_value,
                             hipStream_t stream,
                             T* host_staging = nullptr) {
  T pageable_value{};
  T* const destination = host_staging == nullptr ? &pageable_value : host_staging;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(destination,
                                    d_value,
                                    sizeof(T),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return *destination;
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
      const float w = values[e];
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

template <typename RowOffset>
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
          values[edge] * (vertex_costs == nullptr ? 1.0f : vertex_costs[v]);
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

template <typename RowOffset>
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
      values[edge] *
      (vertex_costs == nullptr ? 1.0f : vertex_costs[current]);
  if (__float_as_uint(predecessor_distance + effective_weight) !=
      __float_as_uint(current_distance)) {
    return false;
  }
  *edge_out = edge;
  *predecessor_out = predecessor;
  return true;
}

template <typename RowOffset>
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
      if (!decode_tight_edge_parent<RowOffset>(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        break;
      }
      current = predecessor;
      ++length;
    }
  }
}

template <typename RowOffset>
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
    Offset* path_edges) {
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
      if (!decode_tight_edge_parent<RowOffset>(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        path_valid = false;
        break;
      }
      path_edges[edge_begin + j - 1] = edge;
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
        const float w = out_values[e];
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
        const float w = out_values[e];
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
    int* pending_count,
    hipStream_t stream) {
  relax_light_edges_kernel<RowOffset, UseCurrentGenerations, TrackParents,
                           UseEdgeParent, HasVertexCosts, CollectHeavy,
                           AllEdgesLight, CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta,
          exclusive_distance_limit,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(),
          scratch.in_current.get(), next_current_generation,
          scratch.in_pending.get(),
          scratch.in_heavy.get(), scratch.touched_queue.get(),
          scratch.touched_count.get(), next_queue, next_count,
          pending_queue, pending_count, scratch.heavy_queue.get(),
          scratch.heavy_count.get(),
          CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

template <typename RowOffset,
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
    int* pending_count,
    hipStream_t stream) {
  relax_heavy_edges_kernel<RowOffset, TrackParents, UseEdgeParent,
                           HasVertexCosts, CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          scratch.heavy_queue.get(), scratch.heavy_count.get(),
          current_bucket, delta, exclusive_distance_limit,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(), scratch.in_pending.get(),
          scratch.touched_queue.get(), scratch.touched_count.get(),
          pending_queue, pending_count, scratch.in_heavy.get(),
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
                                                 int* block_mins,
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
  if (tid == 0) {
    // Each block publishes a self-contained result.  The previous global
    // atomic minimum had to be initialized by an H2D scalar copy followed by
    // a host stream synchronization before this kernel.  Per-block outputs
    // remove that producer dependency entirely; the already-required D2H
    // completion below certifies every block before the host reduces them.
    // Preserve an explicit system-visible publication on gfx1151 without a
    // host drain between this kernel and the copy engine.
    atomicExch(block_mins + blockIdx.x, s_min[0]);
    __threadfence_system();
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

template <typename RowOffset>
struct CooperativeDeltaControllerArgs {
  Offset rows;
  const RowOffset* rowptr;
  const Index* colind;
  const float* values;
  const float* vertex_costs;
  float delta;
  float exclusive_distance_limit;
  float* dist;
  unsigned long long* parent_key;
  std::uint32_t* in_current;
  int* in_pending;
  int* in_heavy;
  int* touched_queue;
  int* touched_count;
  int* current_queue_0;
  int* current_queue_1;
  int* current_count_0;
  int* current_count_1;
  int* pending_queue_0;
  int* pending_queue_1;
  int* pending_count_0;
  int* pending_count_1;
  int* heavy_queue;
  int* heavy_count;
  int* min_pending_bucket;
  const int* targets;
  int target_count;
  int* target_settled;
  int* settled_target_count;
  int scalar_target;
  int max_iters;
  int last_allowed_bucket;
  std::uint32_t batch_size;
  std::uint32_t generation_first;
  int has_distance_limit;
  int skip_heavy_edges;
  unsigned long long* telemetry_counters;
  CooperativeDeltaControllerState* state;
};

template <typename RowOffset>
struct CooperativeDeltaBatchSlot {
  CooperativeDeltaControllerArgs<RowOffset> args;
  std::uint64_t submission_sequence;
  std::uint32_t slot_index;
  std::uint32_t reserved;
};

struct CooperativeDeltaBatchPublication {
  std::uint64_t submission_sequence;
  std::uint32_t slot_index;
  std::uint32_t reserved;
  DeltaSteppingCsrControllerDescriptor descriptor;
};

static_assert(
    std::is_standard_layout<CooperativeDeltaBatchSlot<CompactRowOffset>>::value,
    "compact cooperative Delta batch slots must have standard layout");
static_assert(
    std::is_trivially_copyable<
        CooperativeDeltaBatchSlot<CompactRowOffset>>::value,
    "compact cooperative Delta batch slots must be trivially copyable");
static_assert(
    std::is_standard_layout<CooperativeDeltaBatchSlot<Offset>>::value,
    "wide cooperative Delta batch slots must have standard layout");
static_assert(
    std::is_trivially_copyable<CooperativeDeltaBatchSlot<Offset>>::value,
    "wide cooperative Delta batch slots must be trivially copyable");
static_assert(std::is_standard_layout<CooperativeDeltaBatchPublication>::value,
              "cooperative Delta batch publications must have standard "
              "layout");
static_assert(
    std::is_trivially_copyable<CooperativeDeltaBatchPublication>::value,
    "cooperative Delta batch publications must be trivially copyable");
static_assert(offsetof(CooperativeDeltaBatchPublication, descriptor) == 16,
              "cooperative Delta batch publication identity must precede "
              "the descriptor without hidden padding");
static_assert(
    sizeof(CooperativeDeltaBatchPublication) ==
        16 + sizeof(DeltaSteppingCsrControllerDescriptor),
    "cooperative Delta batch publication ABI must be tightly bounded");

__device__ inline unsigned int controller_atomic_load_u32(
    const unsigned int* address) {
  return atomicAdd(const_cast<unsigned int*>(address), 0U);
}

// These reads are used only after a cooperative grid barrier. Unlike the
// atomic status read used while sticky failure bits may still be arriving,
// the referenced values are stable and globally visible at those sites.
// Avoid turning uniform controller state into one contended no-op atomic per
// participating thread.
__device__ inline unsigned int controller_stable_load_u32(
    const unsigned int* address) {
  return *reinterpret_cast<const volatile unsigned int*>(address);
}

__device__ inline int controller_stable_load_int(const int* address) {
  return *reinterpret_cast<const volatile int*>(address);
}

__device__ inline std::uint64_t controller_stable_load_u64(
    const std::uint64_t* address) {
  return *reinterpret_cast<const volatile std::uint64_t*>(address);
}

__device__ inline DeltaSteppingCsrControllerPhase
controller_phase_after_grid_sync(
    const CooperativeDeltaControllerState* state) {
  const auto* const phase = reinterpret_cast<const unsigned int*>(
      &state->descriptor.phase);
  return static_cast<DeltaSteppingCsrControllerPhase>(
      controller_stable_load_u32(phase));
}

__device__ inline void controller_set_status(
    CooperativeDeltaControllerState* state,
    DeltaSteppingCsrControllerStatus status) {
  atomicOr(reinterpret_cast<unsigned int*>(&state->descriptor.status),
           static_cast<unsigned int>(status));
}

__device__ inline unsigned int controller_status_bits(
    const CooperativeDeltaControllerState* state) {
  return controller_atomic_load_u32(reinterpret_cast<const unsigned int*>(
      &state->descriptor.status));
}

__device__ inline bool controller_has_fatal_status_after_grid_sync(
    const CooperativeDeltaControllerState* state) {
  constexpr unsigned int kFatal =
      static_cast<unsigned int>(
          DeltaSteppingCsrControllerStatus::kQueueOverflow) |
      static_cast<unsigned int>(
          DeltaSteppingCsrControllerStatus::kInvalidState);
  const auto* const status =
      reinterpret_cast<const unsigned int*>(&state->descriptor.status);
  return (controller_stable_load_u32(status) & kFatal) != 0U;
}

__device__ inline DeltaSteppingCsrControllerAction
controller_terminal_action_device(unsigned int status) {
  if ((status & static_cast<unsigned int>(
                    DeltaSteppingCsrControllerStatus::kInvalidState)) != 0U) {
    return DeltaSteppingCsrControllerAction::kStopInvalidState;
  }
  if ((status & static_cast<unsigned int>(
                    DeltaSteppingCsrControllerStatus::kQueueOverflow)) != 0U) {
    return DeltaSteppingCsrControllerAction::kStopQueueOverflow;
  }
  if ((status & static_cast<unsigned int>(
                    DeltaSteppingCsrControllerStatus::kIterationLimit)) != 0U) {
    return DeltaSteppingCsrControllerAction::kStopIterationLimit;
  }
  if ((status & static_cast<unsigned int>(
                    DeltaSteppingCsrControllerStatus::kTargetSettled)) != 0U) {
    return DeltaSteppingCsrControllerAction::kStopTargetSettled;
  }
  if ((status & static_cast<unsigned int>(
                    DeltaSteppingCsrControllerStatus::kComplete)) != 0U) {
    return DeltaSteppingCsrControllerAction::kStopComplete;
  }
  return DeltaSteppingCsrControllerAction::kStopInvalidState;
}

__device__ inline void controller_finish(
    CooperativeDeltaControllerState* state,
    DeltaSteppingCsrControllerStatus status) {
  if (status != DeltaSteppingCsrControllerStatus::kNone) {
    controller_set_status(state, status);
  }
  unsigned int observed = controller_status_bits(state);
  if (observed == 0U) {
    controller_set_status(
        state, DeltaSteppingCsrControllerStatus::kInvalidState);
    observed = static_cast<unsigned int>(
        DeltaSteppingCsrControllerStatus::kInvalidState);
  }
  state->descriptor.phase = DeltaSteppingCsrControllerPhase::kFinished;
  state->descriptor.action = controller_terminal_action_device(observed);
}

__device__ inline int controller_bounded_append(
    bool append,
    int* queue_tail,
    int capacity,
    CooperativeDeltaControllerState* state) {
  if (!append) return -1;
  // Membership/distance claims bound successful reservations to one per
  // vertex. A single atomic reservation is therefore sufficient on the valid
  // path; the post-check retains sticky diagnostics and prevents any OOB
  // payload write if an invariant is violated.
  const int observed = atomicAdd(queue_tail, 1);
  if (observed >= 0 && observed < capacity) return observed;
  controller_set_status(
      state,
      observed < 0 ? DeltaSteppingCsrControllerStatus::kInvalidState
                   : DeltaSteppingCsrControllerStatus::kQueueOverflow);
  return -1;
}

template <bool CollectTelemetry, typename Grid>
__device__ inline void controller_grid_release_sync(
    Grid& grid,
    CooperativeDeltaControllerState* state) {
  // Cooperative grid synchronization completes prior global/shared accesses
  // and makes them visible to the next grid phase. The old per-thread
  // __threadfence duplicated that contract at every phase boundary and was a
  // major source of controller work amplification. Keep the system-scope
  // fence at final host publication below.
  if constexpr (CollectTelemetry) {
    if (grid.thread_rank() == 0) ++state->grid_barriers;
  }
  grid.sync();
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
__device__ void cooperative_relax_light_range(
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    const int* frontier,
    int frontier_count,
    int current_bucket,
    int* next_frontier,
    int* next_count,
    std::uint32_t next_current_generation,
    int* pending_queue,
    int* pending_count) {
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long current_peak = 0;
  unsigned long long pending_peak = 0;
  unsigned long long heavy_peak = 0;
  const int capacity = static_cast<int>(args.rows);
  const bool terminal_bucket = current_bucket == kNoBucket - 1;
  const bool all_edges_light = terminal_bucket || args.skip_heavy_edges != 0;
  const bool collect_heavy = !terminal_bucket && args.skip_heavy_edges == 0;
  // Match the established launch matrix exactly: skip-heavy all-light runs
  // intentionally omit vertex costs, while the terminal bucket retains them.
  const bool use_vertex_costs =
      args.vertex_costs != nullptr &&
      (args.skip_heavy_edges == 0 || terminal_bucket);
  const long long global_thread =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long global_stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long fi = global_thread; fi < frontier_count;
       fi += global_stride) {
    if constexpr (CollectTelemetry) {
      ++telemetry[kTelemetryFrontierEntries];
    }
    const int u = frontier[fi];
    if (u < 0 || u >= capacity) {
      controller_set_status(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      continue;
    }
    const float du = args.dist[u];
    const bool active =
        finite_float(du) && bucket_index(du, args.delta) == current_bucket;
    if constexpr (CollectTelemetry) {
      ++telemetry[active ? kTelemetryActiveVertices
                         : kTelemetryStaleFrontierEntries];
    }
    if (collect_heavy) {
      const bool append_heavy =
          active && atomicCAS(&args.in_heavy[u], 0, 1) == 0;
      const int heavy_pos = controller_bounded_append(
          append_heavy, args.heavy_count, capacity, args.state);
      if (heavy_pos >= 0) {
        args.heavy_queue[heavy_pos] = u;
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryHeavyQueueInsertions];
          const auto observed_peak =
              static_cast<unsigned long long>(heavy_pos) + 1;
          if (observed_peak > heavy_peak) heavy_peak = observed_peak;
        }
      }
    }
    if (!active) continue;
    for (Offset e = static_cast<Offset>(args.rowptr[u]);
         e < static_cast<Offset>(args.rowptr[u + 1]); ++e) {
      if constexpr (CollectTelemetry) {
        ++telemetry[kTelemetryLightEdgeVisits];
      }
      const int v = static_cast<int>(args.colind[e]);
      if (v < 0 || v >= capacity) {
        controller_set_status(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
        continue;
      }
      const float effective_w =
          use_vertex_costs ? args.values[e] * args.vertex_costs[v]
                           : args.values[e];
      const float candidate = du + effective_w;
      const bool below_distance_limit =
          candidate < args.exclusive_distance_limit;
      int candidate_bucket = kNoBucket;
      bool light = true;
      if (!all_edges_light) {
        candidate_bucket = below_distance_limit
                               ? bucket_index(candidate, args.delta)
                               : kNoBucket;
        light = below_distance_limit &&
                (effective_w <= args.delta ||
                 candidate_bucket == current_bucket);
      } else {
        light = below_distance_limit;
      }
      const float nd = light ? candidate : INFINITY;
      float old = INFINITY;
      if constexpr (CollectTelemetry) {
        if (light) {
          const AtomicMinFloatResult atomic_result =
              atomic_min_float_nonnegative<true>(&args.dist[v], nd);
          old = atomic_result.old_value;
          ++telemetry[kTelemetryDistanceAtomicAttempts];
          telemetry[kTelemetryDistanceCasRetries] +=
              atomic_result.cas_retries;
        }
      } else {
        old = light ? atomic_min_float_nonnegative(&args.dist[v], nd)
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
        if (all_edges_light) candidate_bucket = bucket_index(nd, args.delta);
        if constexpr (TrackParents) {
          publish_parent_candidate<true, UseEdgeParent>(
              &args.parent_key[v], u, e, nd);
        }
        if (candidate_bucket == current_bucket) {
          if constexpr (UseCurrentGenerations) {
            append_current =
                atomicExch(&args.in_current[v], next_current_generation) !=
                next_current_generation;
          } else {
            append_current =
                atomicCAS(&args.in_current[v], 0U, 1U) == 0U;
          }
        } else if (candidate_bucket > current_bucket &&
                   candidate_bucket < kNoBucket) {
          append_pending = atomicCAS(&args.in_pending[v], 0, 1) == 0;
        }
      }
      const int touched_pos = controller_bounded_append(
          append_touched, args.touched_count, capacity, args.state);
      const int current_pos = controller_bounded_append(
          append_current, next_count, capacity, args.state);
      const int pending_pos = controller_bounded_append(
          append_pending, pending_count, capacity, args.state);
      if (touched_pos >= 0) args.touched_queue[touched_pos] = v;
      if (current_pos >= 0) {
        next_frontier[current_pos] = v;
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryCurrentQueueInsertions];
          const auto observed_peak =
              static_cast<unsigned long long>(current_pos) + 1;
          if (observed_peak > current_peak) current_peak = observed_peak;
        }
      }
      if (pending_pos >= 0) {
        pending_queue[pending_pos] = v;
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryPendingQueueInsertions];
          const auto observed_peak =
              static_cast<unsigned long long>(pending_pos) + 1;
          if (observed_peak > pending_peak) pending_peak = observed_peak;
        }
      }
    }
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, args.telemetry_counters);
    update_block_queue_peaks(current_peak, pending_peak, heavy_peak,
                             args.telemetry_counters);
  }
}

template <typename RowOffset,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
__device__ void cooperative_relax_heavy_range(
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    int current_bucket,
    int* pending_queue,
    int* pending_count,
    int heavy_count) {
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long pending_peak = 0;
  const unsigned long long heavy_peak =
      blockIdx.x == 0 && threadIdx.x == 0
          ? static_cast<unsigned long long>(heavy_count)
          : 0;
  const int capacity = static_cast<int>(args.rows);
  const long long global_thread =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long global_stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long fi = global_thread; fi < heavy_count;
       fi += global_stride) {
    const int u = args.heavy_queue[fi];
    if (u < 0 || u >= capacity) {
      controller_set_status(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      continue;
    }
    const float du = args.dist[u];
    if (finite_float(du)) {
      for (Offset e = static_cast<Offset>(args.rowptr[u]);
           e < static_cast<Offset>(args.rowptr[u + 1]); ++e) {
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryHeavyEdgeVisits];
        }
        const int v = static_cast<int>(args.colind[e]);
        if (v < 0 || v >= capacity) {
          controller_set_status(
              args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
          continue;
        }
        const float effective_w =
            args.vertex_costs == nullptr
                ? args.values[e]
                : args.values[e] * args.vertex_costs[v];
        const float candidate = du + effective_w;
        const bool below_distance_limit =
            candidate < args.exclusive_distance_limit;
        const int candidate_bucket =
            below_distance_limit ? bucket_index(candidate, args.delta)
                                 : kNoBucket;
        const bool heavy = below_distance_limit && effective_w > args.delta &&
                           candidate_bucket > current_bucket &&
                           candidate_bucket < kNoBucket;
        const float nd = heavy ? candidate : INFINITY;
        float old = INFINITY;
        if constexpr (CollectTelemetry) {
          if (heavy) {
            const AtomicMinFloatResult atomic_result =
                atomic_min_float_nonnegative<true>(&args.dist[v], nd);
            old = atomic_result.old_value;
            ++telemetry[kTelemetryDistanceAtomicAttempts];
            telemetry[kTelemetryDistanceCasRetries] +=
                atomic_result.cas_retries;
          }
        } else {
          old = heavy ? atomic_min_float_nonnegative(&args.dist[v], nd)
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
                &args.parent_key[v], u, e, nd);
          }
          append_pending =
              atomicCAS(&args.in_pending[v], 0, 1) == 0;
        }
        const int touched_pos = controller_bounded_append(
            append_touched, args.touched_count, capacity, args.state);
        const int pending_pos = controller_bounded_append(
            append_pending, pending_count, capacity, args.state);
        if (touched_pos >= 0) args.touched_queue[touched_pos] = v;
        if (pending_pos >= 0) {
          pending_queue[pending_pos] = v;
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryPendingQueueInsertions];
            const auto observed_peak =
                static_cast<unsigned long long>(pending_pos) + 1;
            if (observed_peak > pending_peak) pending_peak = observed_peak;
          }
        }
      }
    }
    args.in_heavy[u] = 0;
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, args.telemetry_counters);
    update_block_queue_peaks(0, pending_peak, heavy_peak,
                             args.telemetry_counters);
  }
}

template <typename RowOffset, bool CollectTelemetry>
__device__ void cooperative_reduce_min_pending(
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    const int* pending_queue,
    int pending_count,
    int previous_bucket) {
  unsigned long long examined = 0;
  unsigned long long stale = 0;
  const int capacity = static_cast<int>(args.rows);
  const long long global_thread =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long global_stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  int local_min = kNoBucket;
  for (long long i = global_thread; i < pending_count; i += global_stride) {
    const int v = pending_queue[i];
    if (v < 0 || v >= capacity) {
      controller_set_status(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      if constexpr (CollectTelemetry) {
        ++examined;
        ++stale;
      }
      continue;
    }
    const bool active = args.in_pending[v] != 0;
    if constexpr (CollectTelemetry) ++examined;
    if (active) {
      const int bucket = bucket_index(args.dist[v], args.delta);
      if (bucket > previous_bucket && bucket < local_min) local_min = bucket;
      if constexpr (CollectTelemetry) {
        if (bucket <= previous_bucket || bucket >= kNoBucket) ++stale;
      }
    } else if constexpr (CollectTelemetry) {
      ++stale;
    }
  }
  // Publish at most one global minimum candidate per block. The previous
  // implementation sent one contended atomicMin from every participating
  // thread, even when most threads observed no pending entry.
  __shared__ int block_min[kBlockSize];
  block_min[threadIdx.x] = local_min;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int rhs = block_min[threadIdx.x + stride];
      if (rhs < block_min[threadIdx.x]) block_min[threadIdx.x] = rhs;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && block_min[0] < kNoBucket) {
    atomicMin(args.min_pending_bucket, block_min[0]);
  }
  if constexpr (CollectTelemetry) {
    if (examined != 0) {
      atomicAdd(args.telemetry_counters + kTelemetryPendingEntryExaminations,
                examined);
    }
    if (stale != 0) {
      atomicAdd(
          args.telemetry_counters + kTelemetryStalePendingEntryExaminations,
          stale);
    }
  }
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool CollectTelemetry>
__device__ void cooperative_compact_pending(
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    const int* pending_in,
    int pending_count,
    int selected_bucket,
    std::uint32_t current_generation,
    int* current_queue,
    int* current_count,
    int* pending_out,
    int* new_pending_count) {
  unsigned long long telemetry[kTelemetryCounterCount] = {};
  unsigned long long current_peak = 0;
  unsigned long long pending_peak = 0;
  const int capacity = static_cast<int>(args.rows);
  const long long global_thread =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long global_stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long i = global_thread; i < pending_count; i += global_stride) {
    const int v = pending_in[i];
    if (v < 0 || v >= capacity) {
      controller_set_status(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      if constexpr (CollectTelemetry) {
        ++telemetry[kTelemetryPendingEntryExaminations];
        ++telemetry[kTelemetryStalePendingEntryExaminations];
      }
      continue;
    }
    const bool active = args.in_pending[v] != 0;
    const int bucket = active ? bucket_index(args.dist[v], args.delta)
                              : kNoBucket;
    if constexpr (CollectTelemetry) {
      ++telemetry[kTelemetryPendingEntryExaminations];
      if (!active || bucket < selected_bucket || bucket >= kNoBucket) {
        ++telemetry[kTelemetryStalePendingEntryExaminations];
      }
    }
    bool append_current = false;
    const bool keep_pending =
        active && bucket > selected_bucket && bucket < kNoBucket;
    if (active && bucket == selected_bucket) {
      atomicExch(&args.in_pending[v], 0);
      if constexpr (UseCurrentGenerations) {
        append_current =
            atomicExch(&args.in_current[v], current_generation) !=
            current_generation;
      } else {
        append_current =
            atomicCAS(&args.in_current[v], 0U, 1U) == 0U;
      }
    } else if (active && !keep_pending) {
      atomicExch(&args.in_pending[v], 0);
    }
    const int current_pos = controller_bounded_append(
        append_current, current_count, capacity, args.state);
    const int pending_pos = controller_bounded_append(
        keep_pending, new_pending_count, capacity, args.state);
    if (current_pos >= 0) {
      current_queue[current_pos] = v;
      if constexpr (CollectTelemetry) {
        ++telemetry[kTelemetryCurrentQueueInsertions];
        const auto observed_peak =
            static_cast<unsigned long long>(current_pos) + 1;
        if (observed_peak > current_peak) current_peak = observed_peak;
      }
    }
    if (pending_pos >= 0) {
      pending_out[pending_pos] = v;
      if constexpr (CollectTelemetry) {
        const auto observed_peak =
            static_cast<unsigned long long>(pending_pos) + 1;
        if (observed_peak > pending_peak) pending_peak = observed_peak;
      }
    }
  }
  if constexpr (CollectTelemetry) {
    add_block_telemetry(telemetry, args.telemetry_counters);
    update_block_queue_peaks(current_peak, pending_peak, 0,
                             args.telemetry_counters);
  }
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
__device__ void cooperative_delta_controller_slot(
    cooperative_groups::grid_group& grid,
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    CooperativeDeltaBatchPublication* publication,
    std::uint64_t submission_sequence,
    std::uint32_t slot_index) {
  const bool leader = grid.thread_rank() == 0;
  const int capacity = static_cast<int>(args.rows);
  if (leader) {
    const bool resumable =
        args.state->descriptor.version ==
            DeltaSteppingCsrControllerDescriptor::kVersion &&
        args.state->descriptor.status ==
            DeltaSteppingCsrControllerStatus::kNone &&
        args.state->descriptor.phase ==
            DeltaSteppingCsrControllerPhase::kLightClosure &&
        (args.state->descriptor.action ==
             DeltaSteppingCsrControllerAction::kContinueDevice ||
         args.state->descriptor.action ==
             DeltaSteppingCsrControllerAction::kPublishHostCheck) &&
        args.state->descriptor.current_count != 0 &&
        args.batch_size != 0 && args.generation_first != 0;
    if (!resumable) {
      controller_finish(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
    } else {
      args.state->descriptor.action =
          DeltaSteppingCsrControllerAction::kContinueDevice;
      args.state->descriptor.rounds_since_host_check = 0;
      args.state->generation_cursor = args.generation_first;
    }
  }
  controller_grid_release_sync<CollectTelemetry>(grid, args.state);

  for (std::uint32_t action = 0; action < args.batch_size; ++action) {
    if (controller_has_fatal_status_after_grid_sync(args.state))
      break;
    const std::uint32_t current_parity =
        controller_stable_load_u32(&args.state->current_queue_parity);
    const std::uint32_t pending_parity =
        controller_stable_load_u32(&args.state->pending_queue_parity);
    if (current_parity > 1U || pending_parity > 1U) {
      if (leader) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      }
      controller_grid_release_sync<CollectTelemetry>(grid, args.state);
      break;
    }
    int* const current_queue =
        current_parity == 0U ? args.current_queue_0 : args.current_queue_1;
    int* const next_queue =
        current_parity == 0U ? args.current_queue_1 : args.current_queue_0;
    int* const current_count_ptr =
        current_parity == 0U ? args.current_count_0 : args.current_count_1;
    int* const next_count_ptr =
        current_parity == 0U ? args.current_count_1 : args.current_count_0;
    int* const pending_queue =
        pending_parity == 0U ? args.pending_queue_0 : args.pending_queue_1;
    int* const pending_count_ptr =
        pending_parity == 0U ? args.pending_count_0 : args.pending_count_1;
    const int current_count = controller_stable_load_int(current_count_ptr);
    const std::uint64_t raw_current_bucket = controller_stable_load_u64(
        &args.state->descriptor.current_bucket);
    const int current_bucket =
        raw_current_bucket <= static_cast<std::uint64_t>(
                                  std::numeric_limits<int>::max())
            ? static_cast<int>(raw_current_bucket)
            : -1;
    if (leader) {
      if (current_count <= 0 || current_count > capacity ||
          current_bucket < 0 || current_bucket >= kNoBucket) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else {
        atomicExch(next_count_ptr, 0);
        args.state->descriptor.phase =
            DeltaSteppingCsrControllerPhase::kLightClosure;
        args.state->descriptor.current_count =
            static_cast<std::uint32_t>(current_count);
        args.state->descriptor.next_bucket =
            kDeltaSteppingCsrNoControllerBucket;
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_has_fatal_status_after_grid_sync(args.state))
      break;

    if constexpr (!UseCurrentGenerations) {
      const long long global_thread =
          static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
      const long long global_stride =
          static_cast<long long>(blockDim.x) * gridDim.x;
      for (long long i = global_thread; i < current_count;
           i += global_stride) {
        const int vertex = current_queue[i];
        if (vertex < 0 || vertex >= capacity) {
          controller_set_status(
              args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
        } else {
          args.in_current[vertex] = 0U;
        }
      }
      controller_grid_release_sync<CollectTelemetry>(grid, args.state);
      if (controller_has_fatal_status_after_grid_sync(args.state))
        break;
    }

    const std::uint32_t next_generation =
        controller_stable_load_u32(&args.state->generation_cursor);
    cooperative_relax_light_range<RowOffset, UseCurrentGenerations,
                                   TrackParents, UseEdgeParent,
                                   CollectTelemetry>(
        args, current_queue, current_count, current_bucket, next_queue,
        next_count_ptr, next_generation, pending_queue, pending_count_ptr);
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    const int next_count = controller_stable_load_int(next_count_ptr);
    const int observed_pending_count =
        controller_stable_load_int(pending_count_ptr);
    if (leader) {
      if (next_count < 0 || next_count > capacity ||
          observed_pending_count < 0 || observed_pending_count > capacity) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else if (args.state->descriptor.light_rounds ==
                     std::numeric_limits<std::uint32_t>::max() ||
                 args.state->descriptor.rounds_since_host_check ==
                     std::numeric_limits<std::uint32_t>::max()) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else {
        args.state->current_queue_parity = current_parity ^ 1U;
        args.state->descriptor.current_count =
            static_cast<std::uint32_t>(next_count);
        args.state->descriptor.pending_count =
            static_cast<std::uint32_t>(observed_pending_count);
        ++args.state->descriptor.light_rounds;
        ++args.state->descriptor.rounds_since_host_check;
        if constexpr (UseCurrentGenerations) {
          ++args.state->generation_cursor;
        }
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_has_fatal_status_after_grid_sync(args.state))
      break;
    if (next_count > 0)
      continue;

    if (args.skip_heavy_edges == 0 && current_bucket != kNoBucket - 1) {
      const int heavy_count = controller_stable_load_int(args.heavy_count);
      if (leader && (heavy_count < 0 || heavy_count > capacity)) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      }
      controller_grid_release_sync<CollectTelemetry>(grid, args.state);
      if (controller_has_fatal_status_after_grid_sync(args.state))
        break;
      cooperative_relax_heavy_range<RowOffset, TrackParents, UseEdgeParent,
                                    CollectTelemetry>(
          args, current_bucket, pending_queue, pending_count_ptr, heavy_count);
      controller_grid_release_sync<CollectTelemetry>(grid, args.state);
      if (controller_has_fatal_status_after_grid_sync(args.state))
        break;
    }

    // Preserve the established host controller's observable stop boundary:
    // complete this bucket's heavy relaxations before settling targets. This
    // matters for scalar-target calls, which expose the partial full-distance
    // and predecessor arrays in addition to the requested target.
    if (args.target_count > 0) {
      const long long global_thread =
          static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
      const long long global_stride =
          static_cast<long long>(blockDim.x) * gridDim.x;
      for (long long i = global_thread; i < args.target_count;
           i += global_stride) {
        if (args.target_settled[i] != 0) continue;
        const int target = args.targets[i];
        if (target < 0 || target >= capacity) {
          controller_set_status(
              args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
          continue;
        }
        const float target_distance = args.dist[target];
        if (finite_float(target_distance) &&
            bucket_index(target_distance, args.delta) <= current_bucket &&
            atomicCAS(&args.target_settled[i], 0, 1) == 0) {
          atomicAdd(args.settled_target_count, 1);
        }
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (leader) {
      if (args.state->descriptor.iterations ==
          std::numeric_limits<std::uint64_t>::max()) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else {
        ++args.state->descriptor.iterations;
      }
      bool target_settled = false;
      if (args.target_count > 0) {
        const int settled = atomic_load_counter(args.settled_target_count);
        if (settled < 0 || settled > args.target_count) {
          controller_finish(
              args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
        } else {
          target_settled = settled == args.target_count;
        }
      } else if (args.scalar_target >= 0) {
        const float target_distance = args.dist[args.scalar_target];
        target_settled =
            finite_float(target_distance) &&
            bucket_index(target_distance, args.delta) <= current_bucket;
      }
      if (target_settled) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kTargetSettled);
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_has_fatal_status_after_grid_sync(args.state) ||
        controller_phase_after_grid_sync(args.state) ==
            DeltaSteppingCsrControllerPhase::kFinished) {
      break;
    }

    const int pending_count = controller_stable_load_int(pending_count_ptr);
    if (leader) {
      if (pending_count < 0 || pending_count > capacity) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else {
        atomicExch(args.min_pending_bucket, kNoBucket);
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_has_fatal_status_after_grid_sync(args.state))
      break;
    cooperative_reduce_min_pending<RowOffset, CollectTelemetry>(
        args, pending_queue, pending_count, current_bucket);
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    const int next_bucket = controller_stable_load_int(args.min_pending_bucket);
    if constexpr (CollectTelemetry) {
      if (leader) {
        telemetry_atomic_max(
            args.telemetry_counters + kTelemetryPendingQueueHighWater,
            static_cast<unsigned long long>(pending_count));
      }
    }
    if (leader) {
      args.state->descriptor.next_bucket =
          next_bucket == kNoBucket
              ? kDeltaSteppingCsrNoControllerBucket
              : static_cast<std::uint64_t>(next_bucket);
      if (next_bucket != kNoBucket &&
          (next_bucket <= current_bucket || next_bucket >= kNoBucket)) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else if (next_bucket == kNoBucket ||
                 (args.has_distance_limit != 0 &&
                  next_bucket > args.last_allowed_bucket)) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kComplete);
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_phase_after_grid_sync(args.state) ==
        DeltaSteppingCsrControllerPhase::kFinished) {
      break;
    }

    const std::uint32_t next_pending_parity = pending_parity ^ 1U;
    int* const new_pending_queue =
        next_pending_parity == 0U ? args.pending_queue_0
                                  : args.pending_queue_1;
    int* const new_pending_count_ptr =
        next_pending_parity == 0U ? args.pending_count_0
                                  : args.pending_count_1;
    const std::uint32_t compact_generation =
        controller_stable_load_u32(&args.state->generation_cursor);
    if (leader) {
      atomicExch(next_count_ptr, 0);
      atomicExch(new_pending_count_ptr, 0);
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    cooperative_compact_pending<RowOffset, UseCurrentGenerations,
                                CollectTelemetry>(
        args, pending_queue, pending_count, next_bucket, compact_generation,
        next_queue, next_count_ptr, new_pending_queue, new_pending_count_ptr);
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    const int compacted_current_count =
        controller_stable_load_int(next_count_ptr);
    const int compacted_pending_count =
        controller_stable_load_int(new_pending_count_ptr);
    if (leader) {
      if (compacted_current_count <= 0 ||
          compacted_current_count > capacity ||
          compacted_pending_count < 0 ||
          compacted_pending_count > capacity) {
        controller_finish(
            args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
      } else {
        args.state->pending_queue_parity = next_pending_parity;
        args.state->descriptor.current_bucket =
            static_cast<std::uint64_t>(next_bucket);
        args.state->descriptor.current_count =
            static_cast<std::uint32_t>(compacted_current_count);
        args.state->descriptor.pending_count =
            static_cast<std::uint32_t>(compacted_pending_count);
        args.state->descriptor.next_bucket =
            kDeltaSteppingCsrNoControllerBucket;
        args.state->descriptor.phase =
            DeltaSteppingCsrControllerPhase::kLightClosure;
        args.state->descriptor.action =
            DeltaSteppingCsrControllerAction::kContinueDevice;
        if constexpr (UseCurrentGenerations) {
          ++args.state->generation_cursor;
        }
        atomicExch(args.heavy_count, 0);
        if (args.state->descriptor.iterations >=
            static_cast<std::uint64_t>(args.max_iters)) {
          controller_finish(
              args.state,
              DeltaSteppingCsrControllerStatus::kIterationLimit);
        }
      }
    }
    controller_grid_release_sync<CollectTelemetry>(grid, args.state);
    if (controller_phase_after_grid_sync(args.state) ==
        DeltaSteppingCsrControllerPhase::kFinished) {
      break;
    }
  }

  if (leader) {
    if (controller_has_fatal_status_after_grid_sync(args.state)) {
      controller_finish(args.state, DeltaSteppingCsrControllerStatus::kNone);
    } else if (controller_phase_after_grid_sync(args.state) !=
               DeltaSteppingCsrControllerPhase::kFinished) {
      args.state->descriptor.phase =
          DeltaSteppingCsrControllerPhase::kLightClosure;
      args.state->descriptor.action =
          DeltaSteppingCsrControllerAction::kPublishHostCheck;
    }
    if (args.state->descriptor.publication_sequence ==
        std::numeric_limits<std::uint32_t>::max()) {
      controller_finish(
          args.state, DeltaSteppingCsrControllerStatus::kInvalidState);
    } else {
      ++args.state->descriptor.publication_sequence;
    }
    if constexpr (CollectTelemetry) {
      ++args.state->grid_barriers;
    }
    if (publication != nullptr) {
      publication->submission_sequence = submission_sequence;
      publication->slot_index = slot_index;
      publication->reserved = 0;
      publication->descriptor = args.state->descriptor;
    }
    __threadfence_system();
  }
  grid.sync();
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
__global__ void cooperative_delta_controller_kernel(
    CooperativeDeltaControllerArgs<RowOffset> args) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cooperative_delta_controller_slot<RowOffset, UseCurrentGenerations,
                                    TrackParents, UseEdgeParent,
                                    CollectTelemetry>(
      grid, args, nullptr, 0, 0);
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
__global__ void cooperative_delta_controller_batch_kernel(
    const CooperativeDeltaBatchSlot<RowOffset>* slots,
    std::uint32_t active_slots,
    CooperativeDeltaBatchPublication* publications) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  // Every block traverses every active slot in the same order. Each slot ends
  // with the controller's existing full-grid publication barrier, so slots may
  // take different uniform controller paths without allowing any block to
  // advance to the next query early.
  for (std::uint32_t slot_position = 0; slot_position < active_slots;
       ++slot_position) {
    const CooperativeDeltaBatchSlot<RowOffset>& slot = slots[slot_position];
    cooperative_delta_controller_slot<RowOffset, UseCurrentGenerations,
                                      TrackParents, UseEdgeParent,
                                      CollectTelemetry>(
        grid, slot.args, &publications[slot_position],
        slot.submission_sequence, slot.slot_index);
  }
}

template <typename RowOffset, bool UseCurrentGenerations, bool TrackParents,
          bool UseEdgeParent, bool CollectTelemetry, bool Batched>
CooperativeLaunchConfiguration
query_cooperative_launch_configuration(Offset rows,
                                       std::uint32_t concurrency_hint,
                                       std::uint32_t requested_batch_blocks_per_cu) {
  CooperativeLaunchConfiguration configuration;
  configuration.initialized = true;
  int device = -1;
  const hipError_t device_status = hipGetDevice(&device);
  if (device_status != hipSuccess) {
    (void)hipGetLastError();
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kCapabilityQueryFailed;
    return configuration;
  }
  int cooperative_launch = 0;
  const hipError_t capability_status = hipDeviceGetAttribute(
      &cooperative_launch, hipDeviceAttributeCooperativeLaunch, device);
  if (capability_status != hipSuccess) {
    (void)hipGetLastError();
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kCapabilityQueryFailed;
    return configuration;
  }
  if (cooperative_launch == 0) {
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kCooperativeUnsupported;
    return configuration;
  }

  hipDeviceProp_t properties{};
  const hipError_t properties_status =
      hipGetDeviceProperties(&properties, device);
  if (properties_status != hipSuccess) {
    (void)hipGetLastError();
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kCapabilityQueryFailed;
    return configuration;
  }
  configuration.compute_units = properties.multiProcessorCount;
  int active_blocks_per_compute_unit = 0;
  hipError_t occupancy_status = hipSuccess;
  if constexpr (Batched) {
    occupancy_status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_compute_unit,
        cooperative_delta_controller_batch_kernel<
            RowOffset, UseCurrentGenerations, TrackParents, UseEdgeParent,
            CollectTelemetry>,
        kBlockSize, 0);
  } else {
    occupancy_status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_compute_unit,
        cooperative_delta_controller_kernel<RowOffset, UseCurrentGenerations,
                                            TrackParents, UseEdgeParent,
                                            CollectTelemetry>,
        kBlockSize, 0);
  }
  if (occupancy_status != hipSuccess) {
    (void)hipGetLastError();
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kOccupancyQueryFailed;
    return configuration;
  }
  configuration.active_blocks_per_compute_unit = active_blocks_per_compute_unit;
  if (active_blocks_per_compute_unit <= 0 ||
      properties.multiProcessorCount <= 0) {
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kNoResidentGrid;
    return configuration;
  }

  const Offset row_blocks =
      (rows + static_cast<Offset>(kBlockSize) - 1) /
      static_cast<Offset>(kBlockSize);
  const Offset legal_resident_limit =
      static_cast<Offset>(active_blocks_per_compute_unit) *
      static_cast<Offset>(properties.multiProcessorCount);
  // A physical multi-query launch is the device's only cooperative grid and
  // visits slots sequentially. Its explicit blocks-per-CU cap is conservative
  // by default because every additional resident block participates in every
  // whole-grid barrier. It is always bounded by the occupancy-derived legal
  // resident limit. The legacy single-query path retains its defensive
  // concurrency partition for low-level callers without a coordinator.
  Offset concurrency_friendly_limit = 0;
  if constexpr (Batched) {
    const std::uint32_t selected_blocks_per_cu =
        delta_stepping_effective_batch_blocks_per_compute_unit(
            requested_batch_blocks_per_cu,
            static_cast<std::uint32_t>(active_blocks_per_compute_unit));
    if (selected_blocks_per_cu == 0) {
      configuration.fallback_reason =
          DeltaSteppingCsrControllerFallbackReason::kNoResidentGrid;
      return configuration;
    }
    configuration.selected_blocks_per_compute_unit =
        static_cast<int>(selected_blocks_per_cu);
    concurrency_friendly_limit =
        static_cast<Offset>(selected_blocks_per_cu) *
        static_cast<Offset>(properties.multiProcessorCount);
  } else {
    concurrency_friendly_limit = std::max<Offset>(
        1,
        static_cast<Offset>(properties.multiProcessorCount) /
            static_cast<Offset>(concurrency_hint));
  }
  const Offset blocks = std::min(
      row_blocks, std::min(legal_resident_limit, concurrency_friendly_limit));
  if (blocks <= 0 ||
      blocks > static_cast<Offset>(std::numeric_limits<int>::max())) {
    configuration.fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kNoResidentGrid;
    return configuration;
  }
  configuration.max_blocks = static_cast<int>(blocks);
  return configuration;
}

template <typename RowOffset, bool UseCurrentGenerations, bool TrackParents,
          bool UseEdgeParent, bool CollectTelemetry>
const CooperativeLaunchConfiguration& cooperative_launch_configuration(
    DeltaSteppingScratch& scratch,
    Offset rows) {
  const bool batched = scratch.controller_batch_coordinator != nullptr;
  const std::size_t key = (UseCurrentGenerations ? std::size_t{1} : 0) |
                          (TrackParents ? std::size_t{2} : 0) |
                          (UseEdgeParent ? std::size_t{4} : 0) |
                          (CollectTelemetry ? std::size_t{8} : 0) |
                          (batched ? std::size_t{16} : 0);
  CooperativeLaunchConfiguration& configuration =
      scratch.cooperative_launch_configurations[key];
  if (!configuration.initialized) {
    const std::uint32_t requested_batch_blocks_per_cu = batched
        ? scratch.controller_batch_coordinator
              ->configured_blocks_per_compute_unit()
        : kDeltaSteppingCsrRecommendedBatchBlocksPerComputeUnit;
    if (batched) {
      configuration = query_cooperative_launch_configuration<
          RowOffset, UseCurrentGenerations, TrackParents, UseEdgeParent,
          CollectTelemetry, true>(rows, scratch.controller_concurrency_hint,
                                  requested_batch_blocks_per_cu);
    } else {
      configuration = query_cooperative_launch_configuration<
          RowOffset, UseCurrentGenerations, TrackParents, UseEdgeParent,
          CollectTelemetry, false>(rows, scratch.controller_concurrency_hint,
                                   requested_batch_blocks_per_cu);
    }
  }
  return configuration;
}

inline bool controller_generation_batch_is_representable(
    std::uint32_t batch_size) {
  const std::uint64_t tokens = std::uint64_t{2} * batch_size;
  return tokens != 0 &&
         tokens < std::numeric_limits<std::uint32_t>::max();
}

template <bool UseCurrentGenerations>
bool reserve_controller_generations(DeltaSteppingScratch& scratch,
                                    std::uint32_t batch_size,
                                    hipStream_t stream,
                                    std::uint32_t* generation_first) {
  if constexpr (!UseCurrentGenerations) {
    *generation_first = 1;
    return true;
  }
  if (!controller_generation_batch_is_representable(batch_size)) {
    return false;
  }
  const std::uint64_t token_count = std::uint64_t{2} * batch_size;
  const std::uint64_t current = scratch.current_generation;
  const std::uint64_t maximum =
      std::numeric_limits<std::uint32_t>::max();
  if (current + token_count > maximum) {
    // No controller phase is in flight at a publication boundary.  The queue
    // is authoritative there, so clearing tags cannot lose frontier work.
    DS_DELTA_HIP_CHECK(hipMemsetAsync(
        scratch.in_current.get(),
        0,
        sssp_capacity::checked_bytes<std::uint32_t>(
            static_cast<std::size_t>(scratch.rows)),
        stream));
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    scratch.current_generation = 0;
  }
  *generation_first = scratch.current_generation + 1U;
  scratch.current_generation = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(scratch.current_generation) + token_count);
  return true;
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
void launch_cooperative_delta_controller(
    CooperativeDeltaControllerArgs<RowOffset> args,
    int blocks,
    hipStream_t stream) {
  void* kernel_args[] = {&args};
  DS_DELTA_HIP_CHECK(hipLaunchCooperativeKernel(
      cooperative_delta_controller_kernel<
          RowOffset, UseCurrentGenerations, TrackParents,
          UseEdgeParent, CollectTelemetry>,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(kBlockSize),
      kernel_args,
      0,
      stream));
}

inline DeltaSteppingCsrControllerDescriptor
copy_controller_descriptor_to_host(DeltaSteppingScratch& scratch,
                                   hipStream_t stream) {
  auto* const destination =
      &scratch.host_controller_state->get()->descriptor;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      destination,
      scratch.controller_state.get(),
      sizeof(DeltaSteppingCsrControllerDescriptor),
      hipMemcpyDeviceToHost,
      stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return *destination;
}

struct CooperativeDeltaBatchSubmitResult {
  DeltaSteppingCsrControllerDescriptor descriptor{};
  std::uint64_t logical_query_sequence = 0;
  std::uint32_t grid_blocks = 0;
};

}  // namespace ds_delta_detail

struct DeltaSteppingCsrBatchCoordinator::Impl {
  enum class ProducerState : std::uint8_t {
    kExpected,
    kActive,
    kRetired,
    kAbandoned,
  };

  struct Completion {
    bool done = false;
    std::exception_ptr failure;
    ds_delta_detail::CooperativeDeltaBatchPublication publication{};
    std::uint64_t logical_query_sequence = 0;
    std::uint32_t grid_blocks = 0;
  };

  struct ExecutorBase;
  using ExecutorFactory =
      std::unique_ptr<ExecutorBase> (*)(std::size_t capacity);

  static constexpr std::size_t kMaxArgumentBytes =
      sizeof(ds_delta_detail::CooperativeDeltaControllerArgs<
                 ds_delta_detail::CompactRowOffset>) >
              sizeof(ds_delta_detail::CooperativeDeltaControllerArgs<
                     minplus_sparse::Offset>)
          ? sizeof(ds_delta_detail::CooperativeDeltaControllerArgs<
                   ds_delta_detail::CompactRowOffset>)
          : sizeof(ds_delta_detail::CooperativeDeltaControllerArgs<
                   minplus_sparse::Offset>);

  struct Record {
    std::uint32_t family_key = 0;
    alignas(std::max_align_t)
        std::array<std::byte, kMaxArgumentBytes> argument_bytes{};
    std::size_t argument_size = 0;
    ExecutorFactory executor_factory = nullptr;
    std::uint64_t submission_sequence = 0;
    std::uint64_t logical_query_sequence = 0;
    std::uint32_t expected_query_sequence = 0;
    std::uint32_t expected_publication_sequence = 0;
    std::uint32_t previous_light_rounds = 0;
    std::uint32_t action_budget = 0;
    std::uint32_t launch_blocks = 0;
    std::uint32_t selected_blocks_per_compute_unit = 0;
    std::uint32_t occupancy_active_blocks_per_compute_unit = 0;
    std::uint32_t compute_units = 0;
    Completion completion{};
  };

  struct ExecutorBase {
    virtual ~ExecutorBase() = default;
    virtual std::uint32_t family_key() const noexcept = 0;
    virtual void execute(Record* const* records,
                         std::size_t record_count,
                         hipStream_t stream,
                         Impl* coordinator) = 0;
  };

  Impl(std::size_t expected_producers,
       std::size_t requested_width,
       std::uint32_t requested_blocks_per_compute_unit,
       int device)
      : expected_producers(expected_producers),
        width(static_cast<std::size_t>(
            delta_stepping_effective_query_batch_width(
                static_cast<std::uint32_t>(std::min<std::size_t>(
                    expected_producers,
                    std::numeric_limits<std::uint32_t>::max())),
                static_cast<std::uint32_t>(std::min<std::size_t>(
                    requested_width,
                    std::numeric_limits<std::uint32_t>::max()))))),
        remaining_producers(expected_producers),
        producer_states(expected_producers, ProducerState::kExpected),
        record_pool(expected_producers),
        ready_queue(expected_producers, nullptr),
        requested_blocks_per_compute_unit(
            requested_blocks_per_compute_unit),
        device(device) {
    if (expected_producers == 0 || requested_width == 0 || width == 0) {
      throw std::invalid_argument(
          "Delta-Stepping query batch coordinator requires positive "
          "producer and width counts");
    }
    if (!delta_stepping_batch_blocks_per_compute_unit_is_valid(
            requested_blocks_per_compute_unit)) {
      throw std::invalid_argument(
          "Delta-Stepping batch blocks per compute unit must be in [1, 8]");
    }
    telemetry_record.configured_width = static_cast<std::uint32_t>(
        std::min<std::size_t>(requested_width,
                              std::numeric_limits<std::uint32_t>::max()));
    telemetry_record.effective_width =
        static_cast<std::uint32_t>(width);
    telemetry_record.requested_blocks_per_compute_unit =
        requested_blocks_per_compute_unit;
    free_records.reserve(expected_producers);
    for (Record& record : record_pool) free_records.push_back(&record);
    coordinator_thread = std::thread([this] { run(); });
  }

  ~Impl() = default;

  void run() noexcept;
  ds_delta_detail::CooperativeDeltaBatchSubmitResult submit_record(
      std::uint32_t family_key,
      const void* argument_bytes,
      std::size_t argument_size,
      ExecutorFactory executor_factory,
      std::uint64_t logical_query_sequence,
      std::uint32_t expected_query_sequence,
      std::uint32_t expected_publication_sequence,
      std::uint32_t previous_light_rounds,
      std::uint32_t action_budget,
      std::uint32_t launch_blocks,
      std::uint32_t selected_blocks_per_compute_unit,
      std::uint32_t occupancy_active_blocks_per_compute_unit,
      std::uint32_t compute_units);
  void acquire(std::size_t producer_index);
  void retire(std::size_t producer_index) noexcept;
  void abandon_from(std::size_t first_producer) noexcept;
  void request_cancel(std::exception_ptr failure) noexcept;
  void join_and_rethrow();

  void note_launch_attempt(
      std::size_t active_slots,
      std::uint32_t blocks,
      std::uint32_t selected_blocks_per_compute_unit,
      std::uint32_t occupancy_active_blocks_per_compute_unit,
      std::uint32_t compute_units) {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    ++telemetry_record.physical_launch_attempts;
    telemetry_record.slot_dispatches += active_slots;
    telemetry_record.active_slots_sum += active_slots;
    const auto observed = static_cast<std::uint32_t>(active_slots);
    if (telemetry_record.active_slots_min == 0) {
      telemetry_record.active_slots_min = observed;
    } else {
      telemetry_record.active_slots_min =
          std::min(telemetry_record.active_slots_min, observed);
    }
    telemetry_record.active_slots_max =
        std::max(telemetry_record.active_slots_max, observed);
    telemetry_record.unused_slots += width - active_slots;
    if (telemetry_record.grid_blocks_min == 0) {
      telemetry_record.grid_blocks_min = blocks;
    } else {
      telemetry_record.grid_blocks_min =
          std::min(telemetry_record.grid_blocks_min, blocks);
    }
    telemetry_record.grid_blocks_max =
        std::max(telemetry_record.grid_blocks_max, blocks);
    auto update_nonzero_range = [](std::uint32_t observed,
                                   std::uint32_t* minimum,
                                   std::uint32_t* maximum) {
      if (*minimum == 0) {
        *minimum = observed;
      } else {
        *minimum = std::min(*minimum, observed);
      }
      *maximum = std::max(*maximum, observed);
    };
    update_nonzero_range(
        selected_blocks_per_compute_unit,
        &telemetry_record.selected_blocks_per_compute_unit_min,
        &telemetry_record.selected_blocks_per_compute_unit_max);
    update_nonzero_range(
        occupancy_active_blocks_per_compute_unit,
        &telemetry_record.occupancy_active_blocks_per_compute_unit_min,
        &telemetry_record.occupancy_active_blocks_per_compute_unit_max);
    if (telemetry_record.compute_units == 0) {
      telemetry_record.compute_units = compute_units;
    }
  }

  void note_launch_accepted() {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    ++telemetry_record.physical_launches;
    ++in_flight_launches;
    telemetry_record.max_concurrent_cooperative_launches = std::max(
        telemetry_record.max_concurrent_cooperative_launches,
        in_flight_launches);
  }

  void note_launch_finished(bool completed) {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    if (completed) ++telemetry_record.physical_completions;
    if (in_flight_launches != 0) --in_flight_launches;
  }

  void note_launch_failure() {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    ++telemetry_record.launch_failures;
  }

  void note_descriptor_failure() {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    ++telemetry_record.descriptor_failures;
  }

  void note_validated_slot(const Record& record,
                           const DeltaSteppingCsrControllerDescriptor& descriptor,
                           std::uint64_t completed_actions) {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    if (record.expected_publication_sequence == 1) {
      ++telemetry_record.logical_queries_admitted;
    } else {
      ++telemetry_record.slot_relaunches;
    }
    if (descriptor.status != DeltaSteppingCsrControllerStatus::kNone) {
      ++telemetry_record.logical_queries_completed;
    }
    telemetry_record.action_slots_budgeted += record.action_budget;
    telemetry_record.actions_completed += completed_actions;
    telemetry_record.unused_action_slots +=
        static_cast<std::uint64_t>(record.action_budget) - completed_actions;
  }

  DeltaSteppingCsrBatchTelemetry telemetry() const {
    std::lock_guard<std::mutex> lock(telemetry_mutex);
    return telemetry_record;
  }

  void fail_records_locked(
      Record* const* records,
      std::size_t record_count,
      const std::exception_ptr& failure) {
    for (std::size_t i = 0; i < record_count; ++i) {
      Record* const record = records[i];
      record->completion.failure = failure;
      record->completion.done = true;
    }
  }

  void push_ready_locked(Record* record) noexcept {
    ready_queue[ready_tail] = record;
    ready_tail = (ready_tail + 1) % ready_queue.size();
    ++ready_count;
  }

  Record* pop_ready_locked() noexcept {
    Record* const record = ready_queue[ready_head];
    ready_queue[ready_head] = nullptr;
    ready_head = (ready_head + 1) % ready_queue.size();
    --ready_count;
    return record;
  }

  void fail_ready_locked(const std::exception_ptr& failure) {
    while (ready_count != 0) {
      Record* const record = pop_ready_locked();
      record->completion.failure = failure;
      record->completion.done = true;
    }
  }

  const std::size_t expected_producers;
  const std::size_t width;
  std::size_t remaining_producers;
  std::vector<ProducerState> producer_states;
  const std::uint32_t requested_blocks_per_compute_unit;
  const int device;

  mutable std::mutex mutex;
  std::condition_variable condition;
  std::vector<Record> record_pool;
  std::vector<Record*> free_records;
  std::vector<Record*> ready_queue;
  std::size_t ready_head = 0;
  std::size_t ready_tail = 0;
  std::size_t ready_count = 0;
  std::exception_ptr failure;
  bool cancel_requested = false;
  bool stopped = false;
  std::uint64_t next_submission_sequence = 1;
  std::uint64_t next_logical_query_sequence = 1;
  std::thread coordinator_thread;

  mutable std::mutex telemetry_mutex;
  DeltaSteppingCsrBatchTelemetry telemetry_record;
  std::uint32_t in_flight_launches = 0;
};

namespace {

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
constexpr std::uint32_t delta_batch_family_key() noexcept {
  return (std::is_same<RowOffset, ds_delta_detail::CompactRowOffset>::value
              ? std::uint32_t{1}
              : std::uint32_t{0}) |
         (UseCurrentGenerations ? std::uint32_t{2} : 0) |
         (TrackParents ? std::uint32_t{4} : 0) |
         (UseEdgeParent ? std::uint32_t{8} : 0) |
         (CollectTelemetry ? std::uint32_t{16} : 0);
}

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
class DeltaBatchExecutor final
    : public DeltaSteppingCsrBatchCoordinator::Impl::ExecutorBase {
 public:
  using Args = ds_delta_detail::CooperativeDeltaControllerArgs<RowOffset>;
  using Slot = ds_delta_detail::CooperativeDeltaBatchSlot<RowOffset>;
  using Publication = ds_delta_detail::CooperativeDeltaBatchPublication;
  using Record = DeltaSteppingCsrBatchCoordinator::Impl::Record;

  explicit DeltaBatchExecutor(std::size_t capacity)
      : capacity_(capacity),
        host_slots_(capacity),
        host_publications_(capacity),
        device_slots_(capacity),
        device_publications_(capacity) {}

  std::uint32_t family_key() const noexcept override {
    return delta_batch_family_key<RowOffset, UseCurrentGenerations,
                                  TrackParents, UseEdgeParent,
                                  CollectTelemetry>();
  }

  void execute(Record* const* records,
               std::size_t record_count,
               hipStream_t stream,
               DeltaSteppingCsrBatchCoordinator::Impl* coordinator) override {
    if (records == nullptr || record_count == 0 || record_count > capacity_) {
      throw std::logic_error(
          "Delta-Stepping cooperative query batch has invalid active width");
    }

    const RowOffset* shared_rowptr = nullptr;
    const minplus_sparse::Index* shared_colind = nullptr;
    const float* shared_values = nullptr;
    minplus_sparse::Offset shared_rows = 0;
    std::uint32_t launch_blocks = 0;
    std::uint32_t selected_blocks_per_compute_unit = 0;
    std::uint32_t occupancy_active_blocks_per_compute_unit = 0;
    std::uint32_t compute_units = 0;
    std::array<std::array<const void*, 23>,
               kDeltaSteppingCsrMaxQueryBatchWidth>
        query_owned_addresses_by_slot{};

    for (std::size_t i = 0; i < record_count; ++i) {
      const Record* const record = records[i];
      if (record->family_key != family_key() ||
          record->argument_size != sizeof(Args) ||
          record->launch_blocks == 0 ||
          record->selected_blocks_per_compute_unit == 0 ||
          record->occupancy_active_blocks_per_compute_unit == 0 ||
          record->compute_units == 0) {
        throw std::logic_error(
            "Delta-Stepping cooperative query batch mixes kernel variants");
      }
      Args args{};
      std::memcpy(&args, record->argument_bytes.data(), sizeof(args));
      if (i == 0) {
        shared_rows = args.rows;
        shared_rowptr = args.rowptr;
        shared_colind = args.colind;
        shared_values = args.values;
        launch_blocks = record->launch_blocks;
        selected_blocks_per_compute_unit =
            record->selected_blocks_per_compute_unit;
        occupancy_active_blocks_per_compute_unit =
            record->occupancy_active_blocks_per_compute_unit;
        compute_units = record->compute_units;
      } else if (args.rows != shared_rows || args.rowptr != shared_rowptr ||
                 args.colind != shared_colind ||
                 args.values != shared_values ||
                 record->launch_blocks != launch_blocks ||
                 record->selected_blocks_per_compute_unit !=
                     selected_blocks_per_compute_unit ||
                 record->occupancy_active_blocks_per_compute_unit !=
                     occupancy_active_blocks_per_compute_unit ||
                 record->compute_units != compute_units) {
        throw std::logic_error(
            "Delta-Stepping cooperative query batch mixes graph or launch "
            "configuration");
      }

      auto& query_owned_addresses = query_owned_addresses_by_slot[i];
      query_owned_addresses = {
          args.dist,
          args.parent_key,
          args.in_current,
          args.in_pending,
          args.in_heavy,
          args.touched_queue,
          args.touched_count,
          args.current_queue_0,
          args.current_queue_1,
          args.current_count_0,
          args.current_count_1,
          args.pending_queue_0,
          args.pending_queue_1,
          args.pending_count_0,
          args.pending_count_1,
          args.heavy_queue,
          args.heavy_count,
          args.min_pending_bucket,
          args.targets,
          args.target_settled,
          args.settled_target_count,
          args.telemetry_counters,
          args.state};
      // Width is bounded to eight, so pairwise checks are cheaper and more
      // predictable than constructing a hash table on every publication.
      // Check both accidental aliases inside one workspace and cross-slot
      // reuse of storage that may still be in flight.
      for (std::size_t address_index = 0;
           address_index < query_owned_addresses.size(); ++address_index) {
        const void* const address = query_owned_addresses[address_index];
        if (address == nullptr) continue;
        for (std::size_t previous_index = 0;
             previous_index < address_index; ++previous_index) {
          if (address == query_owned_addresses[previous_index]) {
            throw std::logic_error(
                "Delta-Stepping cooperative query batch aliases query-owned "
                "storage");
          }
        }
        for (std::size_t previous_slot = 0; previous_slot < i;
             ++previous_slot) {
          for (const void* const previous_address :
               query_owned_addresses_by_slot[previous_slot]) {
            if (address == previous_address) {
              throw std::logic_error(
                  "Delta-Stepping cooperative query batch aliases "
                  "query-owned storage across slots");
            }
          }
        }
      }

      host_slots_.get()[i] =
          Slot{args, record->submission_sequence,
               static_cast<std::uint32_t>(i), 0};
    }

    coordinator->note_launch_attempt(
        record_count, launch_blocks, selected_blocks_per_compute_unit,
        occupancy_active_blocks_per_compute_unit, compute_units);
    bool launch_started = false;
    try {
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          device_slots_.get(), host_slots_.get(),
          sssp_capacity::checked_bytes<Slot>(record_count),
          hipMemcpyHostToDevice, stream));
      const Slot* slot_pointer = device_slots_.get();
      std::uint32_t active_slots =
          static_cast<std::uint32_t>(record_count);
      Publication* publication_pointer = device_publications_.get();
      void* kernel_arguments[] = {
          &slot_pointer, &active_slots, &publication_pointer};
      DS_DELTA_HIP_CHECK(hipLaunchCooperativeKernel(
          ds_delta_detail::cooperative_delta_controller_batch_kernel<
              RowOffset, UseCurrentGenerations, TrackParents, UseEdgeParent,
              CollectTelemetry>,
          dim3(launch_blocks), dim3(ds_delta_detail::kBlockSize),
          kernel_arguments, 0, stream));
      coordinator->note_launch_accepted();
      launch_started = true;
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          host_publications_.get(), device_publications_.get(),
          sssp_capacity::checked_bytes<Publication>(record_count),
          hipMemcpyDeviceToHost, stream));
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      coordinator->note_launch_finished(true);
      launch_started = false;
    } catch (...) {
      // Host slot/publication staging and every referenced workspace must stay
      // alive until the one batch stream has stopped consuming them.
      (void)hipStreamSynchronize(stream);
      if (launch_started) coordinator->note_launch_finished(false);
      coordinator->note_launch_failure();
      throw;
    }

    // Validate every publication before exposing any result to a worker. One
    // corrupt slot cancels the complete physical batch atomically.
    for (std::size_t i = 0; i < record_count; ++i) {
      const auto& record = *records[i];
      const Publication& publication = host_publications_.get()[i];
      const auto& descriptor = publication.descriptor;
      const std::uint64_t completed_actions =
          static_cast<std::uint64_t>(descriptor.light_rounds) -
          static_cast<std::uint64_t>(record.previous_light_rounds);
      const bool valid =
          publication.submission_sequence == record.submission_sequence &&
          publication.slot_index == i &&
          delta_stepping_controller_descriptor_is_valid(descriptor) &&
          descriptor.query_sequence == record.expected_query_sequence &&
          descriptor.publication_sequence ==
              record.expected_publication_sequence &&
          descriptor.light_rounds > record.previous_light_rounds &&
          completed_actions <= record.action_budget &&
          (descriptor.status != DeltaSteppingCsrControllerStatus::kNone ||
           completed_actions == record.action_budget);
      if (!valid) {
        coordinator->note_descriptor_failure();
        throw std::runtime_error(
            "Delta-Stepping cooperative query batch published an invalid "
            "slot descriptor");
      }
    }

    for (std::size_t i = 0; i < record_count; ++i) {
      auto& record = *records[i];
      const Publication publication = host_publications_.get()[i];
      const std::uint64_t completed_actions =
          static_cast<std::uint64_t>(publication.descriptor.light_rounds) -
          static_cast<std::uint64_t>(record.previous_light_rounds);
      record.completion.publication = publication;
      record.completion.grid_blocks = launch_blocks;
      coordinator->note_validated_slot(
          record, publication.descriptor, completed_actions);
    }
  }

 private:
  std::size_t capacity_;
  ds_delta_detail::PinnedHostBuffer<Slot> host_slots_;
  ds_delta_detail::PinnedHostBuffer<Publication> host_publications_;
  ds_delta_detail::DeviceBuffer<Slot> device_slots_;
  ds_delta_detail::DeviceBuffer<Publication> device_publications_;
};

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
std::unique_ptr<DeltaSteppingCsrBatchCoordinator::Impl::ExecutorBase>
make_delta_batch_executor(std::size_t capacity) {
  return std::make_unique<
      DeltaBatchExecutor<RowOffset, UseCurrentGenerations, TrackParents,
                         UseEdgeParent, CollectTelemetry>>(capacity);
}

}  // namespace

void DeltaSteppingCsrBatchCoordinator::Impl::run() noexcept {
  hipStream_t stream = nullptr;
  std::unique_ptr<ExecutorBase> executor;
  std::array<Record*, kDeltaSteppingCsrMaxQueryBatchWidth> batch{};
  try {
    DS_DELTA_HIP_CHECK(hipSetDevice(device));
    DS_DELTA_HIP_CHECK(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    while (true) {
      std::size_t batch_size = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] {
          return cancel_requested || remaining_producers == 0 ||
                 ready_count >= width ||
                 (ready_count != 0 &&
                  ready_count >= remaining_producers);
        });
        if (cancel_requested) {
          fail_ready_locked(failure);
          stopped = true;
          condition.notify_all();
          break;
        }
        if (remaining_producers == 0 && ready_count == 0) {
          stopped = true;
          condition.notify_all();
          break;
        }

        batch_size = std::min(width, ready_count);
        if (batch_size == 0) continue;
        for (std::size_t i = 0; i < batch_size; ++i) {
          batch[i] = pop_ready_locked();
        }
      }

      std::exception_ptr batch_failure;
      try {
        if (!executor) {
          if (batch[0]->executor_factory == nullptr) {
            throw std::logic_error(
                "Delta-Stepping query batch has no kernel executor");
          }
          executor = batch[0]->executor_factory(width);
        }
        for (std::size_t i = 0; i < batch_size; ++i) {
          const Record* const record = batch[i];
          if (record->family_key != executor->family_key() ||
              record->executor_factory != batch[0]->executor_factory) {
            throw std::logic_error(
                "Delta-Stepping query batch mixes controller "
                "specializations");
          }
        }
        executor->execute(batch.data(), batch_size, stream, this);
      } catch (...) {
        batch_failure = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        if (batch_failure && !failure) failure = batch_failure;
        if (cancel_requested || batch_failure) {
          cancel_requested = true;
          const std::exception_ptr reported =
              failure ? failure : batch_failure;
          fail_records_locked(batch.data(), batch_size, reported);
          fail_ready_locked(reported);
        } else {
          for (std::size_t i = 0; i < batch_size; ++i) {
            batch[i]->completion.done = true;
          }
        }
        condition.notify_all();
        if (cancel_requested) {
          stopped = true;
          break;
        }
      }
    }

    // Release all batch device/pinned storage on the selected coordinator
    // device before destroying its sole launch stream.
    executor.reset();
    DS_DELTA_HIP_CHECK(hipStreamDestroy(stream));
    stream = nullptr;
  } catch (...) {
    const std::exception_ptr coordinator_failure = std::current_exception();
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      executor.reset();
      (void)hipStreamDestroy(stream);
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (!failure) failure = coordinator_failure;
    cancel_requested = true;
    stopped = true;
    fail_ready_locked(failure);
    condition.notify_all();
  }
}

ds_delta_detail::CooperativeDeltaBatchSubmitResult
DeltaSteppingCsrBatchCoordinator::Impl::submit_record(
    std::uint32_t family_key,
    const void* argument_bytes,
    std::size_t argument_size,
    ExecutorFactory executor_factory,
    std::uint64_t logical_query_sequence,
    std::uint32_t expected_query_sequence,
    std::uint32_t expected_publication_sequence,
    std::uint32_t previous_light_rounds,
    std::uint32_t action_budget,
    std::uint32_t launch_blocks,
    std::uint32_t selected_blocks_per_compute_unit,
    std::uint32_t occupancy_active_blocks_per_compute_unit,
    std::uint32_t compute_units) {
  if (argument_bytes == nullptr || argument_size == 0 ||
      argument_size > kMaxArgumentBytes || executor_factory == nullptr) {
    throw std::invalid_argument(
        "Delta-Stepping query batch submission has invalid fixed storage");
  }
  std::unique_lock<std::mutex> lock(mutex);
  if (failure) std::rethrow_exception(failure);
  if (cancel_requested || stopped) {
    throw std::runtime_error(
        "Delta-Stepping query batch coordinator is stopped");
  }
  if (logical_query_sequence == 0) {
    if (next_logical_query_sequence == 0) {
      throw std::overflow_error(
          "Delta-Stepping batch logical query sequence exhausted");
    }
    logical_query_sequence = next_logical_query_sequence++;
  }
  if (next_submission_sequence == 0) {
    throw std::overflow_error(
        "Delta-Stepping batch submission sequence exhausted");
  }
  // A producer cannot have more than one outstanding synchronous submission.
  // The pool therefore has exactly the expected producer count and exhaustion
  // identifies API misuse rather than a condition worth waiting on with the
  // caller's stack-local argument record still uncopied.
  if (free_records.empty()) {
    throw std::logic_error(
        "Delta-Stepping query batch fixed submission pool is exhausted");
  }
  Record* const record = free_records.back();
  free_records.pop_back();
  *record = Record{};
  record->family_key = family_key;
  record->argument_size = argument_size;
  std::memcpy(record->argument_bytes.data(), argument_bytes, argument_size);
  record->executor_factory = executor_factory;
  record->logical_query_sequence = logical_query_sequence;
  record->submission_sequence = next_submission_sequence++;
  record->expected_query_sequence = expected_query_sequence;
  record->expected_publication_sequence = expected_publication_sequence;
  record->previous_light_rounds = previous_light_rounds;
  record->action_budget = action_budget;
  record->launch_blocks = launch_blocks;
  record->selected_blocks_per_compute_unit =
      selected_blocks_per_compute_unit;
  record->occupancy_active_blocks_per_compute_unit =
      occupancy_active_blocks_per_compute_unit;
  record->compute_units = compute_units;
  record->completion.logical_query_sequence = logical_query_sequence;
  push_ready_locked(record);
  condition.notify_all();
  condition.wait(lock, [&] { return record->completion.done; });

  const std::exception_ptr completion_failure = record->completion.failure;
  ds_delta_detail::CooperativeDeltaBatchSubmitResult result;
  result.descriptor = record->completion.publication.descriptor;
  result.logical_query_sequence =
      record->completion.logical_query_sequence;
  result.grid_blocks = record->completion.grid_blocks;
  free_records.push_back(record);
  condition.notify_all();
  // The coordinator completed its HIP stream before taking mutex and setting
  // done. This acquire resumes the worker only after all batch writes and D2H
  // descriptors are complete; subsequent extraction may safely enqueue on the
  // worker's original stream without a device-wide synchronization.
  lock.unlock();
  if (completion_failure) std::rethrow_exception(completion_failure);
  return result;
}

void DeltaSteppingCsrBatchCoordinator::Impl::acquire(
    std::size_t producer_index) {
  std::lock_guard<std::mutex> lock(mutex);
  if (failure) std::rethrow_exception(failure);
  if (producer_index >= producer_states.size() ||
      producer_states[producer_index] != ProducerState::kExpected) {
    throw std::logic_error(
        "Delta-Stepping batch producer lease is invalid or duplicated");
  }
  producer_states[producer_index] = ProducerState::kActive;
}

void DeltaSteppingCsrBatchCoordinator::Impl::retire(
    std::size_t producer_index) noexcept {
  std::lock_guard<std::mutex> lock(mutex);
  if (producer_index >= producer_states.size() ||
      producer_states[producer_index] != ProducerState::kActive) {
    return;
  }
  producer_states[producer_index] = ProducerState::kRetired;
  if (remaining_producers != 0) --remaining_producers;
  condition.notify_all();
}

void DeltaSteppingCsrBatchCoordinator::Impl::abandon_from(
    std::size_t first_producer) noexcept {
  std::lock_guard<std::mutex> lock(mutex);
  for (std::size_t i = first_producer; i < producer_states.size(); ++i) {
    if (producer_states[i] == ProducerState::kExpected) {
      producer_states[i] = ProducerState::kAbandoned;
      if (remaining_producers != 0) --remaining_producers;
    }
  }
  condition.notify_all();
}

void DeltaSteppingCsrBatchCoordinator::Impl::request_cancel(
    std::exception_ptr requested_failure) noexcept {
  if (!requested_failure) {
    try {
      throw std::runtime_error(
          "Delta-Stepping query batch coordinator cancelled");
    } catch (...) {
      requested_failure = std::current_exception();
    }
  }
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex);
    ++telemetry_record.cancellations;
  }
  std::lock_guard<std::mutex> lock(mutex);
  if (!failure) failure = requested_failure;
  cancel_requested = true;
  condition.notify_all();
}

void DeltaSteppingCsrBatchCoordinator::Impl::join_and_rethrow() {
  if (coordinator_thread.joinable()) coordinator_thread.join();
  std::lock_guard<std::mutex> lock(mutex);
  if (failure) std::rethrow_exception(failure);
}

DeltaSteppingCsrBatchCoordinator::DeltaSteppingCsrBatchCoordinator(
    std::size_t expected_producers,
    std::size_t requested_width,
    std::uint32_t requested_blocks_per_compute_unit) {
  int device = 0;
  DS_DELTA_HIP_CHECK(hipGetDevice(&device));
  impl_ = std::make_unique<Impl>(expected_producers, requested_width,
                                 requested_blocks_per_compute_unit, device);
}

DeltaSteppingCsrBatchCoordinator::~DeltaSteppingCsrBatchCoordinator() {
  if (impl_ && impl_->coordinator_thread.joinable()) {
    impl_->request_cancel(nullptr);
    impl_->coordinator_thread.join();
  }
}

DeltaSteppingCsrBatchCoordinator::ProducerLease::ProducerLease(
    DeltaSteppingCsrBatchCoordinator* owner,
    std::size_t producer_index) noexcept
    : owner_(owner), producer_index_(producer_index) {}

DeltaSteppingCsrBatchCoordinator::ProducerLease::~ProducerLease() {
  release();
}

DeltaSteppingCsrBatchCoordinator::ProducerLease::ProducerLease(
    ProducerLease&& other) noexcept
    : owner_(other.owner_), producer_index_(other.producer_index_) {
  other.owner_ = nullptr;
}

DeltaSteppingCsrBatchCoordinator::ProducerLease&
DeltaSteppingCsrBatchCoordinator::ProducerLease::operator=(
    ProducerLease&& other) noexcept {
  if (this != &other) {
    release();
    owner_ = other.owner_;
    producer_index_ = other.producer_index_;
    other.owner_ = nullptr;
  }
  return *this;
}

void DeltaSteppingCsrBatchCoordinator::ProducerLease::release() noexcept {
  if (owner_ != nullptr) {
    owner_->retire_producer(producer_index_);
    owner_ = nullptr;
  }
}

DeltaSteppingCsrBatchCoordinator::ProducerLease
DeltaSteppingCsrBatchCoordinator::acquire_producer(
    std::size_t producer_index) {
  if (!impl_) throw std::logic_error("Delta batch coordinator is moved");
  impl_->acquire(producer_index);
  return ProducerLease(this, producer_index);
}

void DeltaSteppingCsrBatchCoordinator::retire_producer(
    std::size_t producer_index) noexcept {
  if (impl_) impl_->retire(producer_index);
}

void DeltaSteppingCsrBatchCoordinator::abandon_unstarted_producers(
    std::size_t first_producer) noexcept {
  if (impl_) impl_->abandon_from(first_producer);
}

void DeltaSteppingCsrBatchCoordinator::cancel(
    std::exception_ptr failure) noexcept {
  if (impl_) impl_->request_cancel(failure);
}

void DeltaSteppingCsrBatchCoordinator::finish() {
  if (impl_) impl_->join_and_rethrow();
}

DeltaSteppingCsrBatchTelemetry
DeltaSteppingCsrBatchCoordinator::telemetry() const {
  return impl_ ? impl_->telemetry() : DeltaSteppingCsrBatchTelemetry{};
}

std::uint32_t
DeltaSteppingCsrBatchCoordinator::configured_blocks_per_compute_unit()
    const noexcept {
  return impl_ ? impl_->requested_blocks_per_compute_unit : 0;
}

DeltaSteppingCsrBatchCoordinator::Impl*
DeltaSteppingCsrBatchCoordinator::implementation_for_delta() noexcept {
  return impl_.get();
}

namespace ds_delta_detail {

template <typename RowOffset,
          bool UseCurrentGenerations,
          bool TrackParents,
          bool UseEdgeParent,
          bool CollectTelemetry>
CooperativeDeltaBatchSubmitResult submit_cooperative_delta_controller_batch(
    DeltaSteppingCsrBatchCoordinator& coordinator,
    const CooperativeDeltaControllerArgs<RowOffset>& args,
    std::uint64_t logical_query_sequence,
    std::uint32_t expected_query_sequence,
    std::uint32_t expected_publication_sequence,
    std::uint32_t previous_light_rounds,
    std::uint32_t action_budget,
    std::uint32_t launch_blocks,
    std::uint32_t selected_blocks_per_compute_unit,
    std::uint32_t occupancy_active_blocks_per_compute_unit,
    std::uint32_t compute_units) {
  auto* const implementation = coordinator.implementation_for_delta();
  if (implementation == nullptr) {
    throw std::logic_error(
        "Delta-Stepping query batch coordinator has no implementation");
  }
  if (!delta_stepping_query_batch_action_bound_is_valid(
          static_cast<std::uint32_t>(implementation->width),
          action_budget)) {
    throw std::invalid_argument(
        "Delta-Stepping cooperative query batch exceeds its bounded action "
        "watchdog");
  }
  // submit_record copies the complete value into its preallocated fixed pool
  // before this stack-local argument record can go out of scope. Steady-state
  // publications perform no general heap allocation.
  return implementation->submit_record(
      delta_batch_family_key<RowOffset, UseCurrentGenerations, TrackParents,
                             UseEdgeParent, CollectTelemetry>(),
      &args, sizeof(args),
      &make_delta_batch_executor<RowOffset, UseCurrentGenerations,
                                 TrackParents, UseEdgeParent,
                                 CollectTelemetry>,
      logical_query_sequence, expected_query_sequence,
      expected_publication_sequence, previous_light_rounds, action_budget,
      launch_blocks, selected_blocks_per_compute_unit,
      occupancy_active_blocks_per_compute_unit, compute_units);
}

template <bool TrackParents, bool UseEdgeParent>
void fully_reinitialize_after_controller_error(DeltaSteppingScratch& scratch,
                                               Offset n,
                                               float inf,
                                               hipStream_t stream) {
  initialize_delta_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, scratch.dist.get(), scratch.in_current.get(),
      scratch.in_pending.get(), scratch.in_heavy.get(),
      scratch.current_count.get(), scratch.next_count.get(),
      scratch.pending_count.get(), scratch.heavy_count.get(),
      scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  reset_int_zero_async(scratch.new_pending_count.get(), stream);
  reset_int_zero_async(scratch.settled_target_count.get(), stream);
  reset_int_zero_async(scratch.min_pending_bucket.get(), stream);
  if constexpr (TrackParents) {
    initialize_parent_keys_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
        n, scratch.parent_key.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  if constexpr (TrackParents && !UseEdgeParent) {
    initialize_legacy_predecessors_kernel
        <<<grid_for_items(n), kBlockSize, 0, stream>>>(
            n, scratch.pred_node.get(), scratch.pred_edge.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  scratch.current_generation = 0;
  scratch.generic_initialized = true;
  if constexpr (TrackParents) scratch.parent_key_initialized = true;
  if constexpr (TrackParents && !UseEdgeParent) {
    scratch.legacy_predecessors_initialized = true;
  }
}

DeviceCsrOwner copy_host_csr_to_device(const HostCsrF32& h,
                                       hipStream_t stream,
                                       bool build_compact_edge_source,
                                       DeltaSteppingCsrOffsetMode offset_mode) {
  const bool uses_32_bit_offsets =
      delta_stepping_device_row_offset_width(h.nnz, offset_mode) ==
      DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit;
  DeviceCsrOwner d(h.rows, h.cols, h.nnz, uses_32_bit_offsets);
  const std::size_t rows = checked_size(h.rows, "rows");
  const std::size_t nnz = checked_size(h.nnz, "nnz");
  const std::size_t row_offset_count =
      sssp_capacity::checked_add(rows, 1);
  std::vector<std::uint32_t> compact_rowptr;
  if (uses_32_bit_offsets) {
    compact_rowptr = delta_stepping_compact_row_offsets(h.rowptr);
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        d.rowptr32.get(), compact_rowptr.data(),
        sssp_capacity::checked_bytes<CompactRowOffset>(row_offset_count),
        hipMemcpyHostToDevice, stream));
  } else {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        d.rowptr64.get(), h.rowptr.data(),
        sssp_capacity::checked_bytes<Offset>(row_offset_count),
        hipMemcpyHostToDevice, stream));
  }
  if (build_compact_edge_source &&
      delta_stepping_compact_edge_ids_eligible(h.nnz)) {
    const hipError_t allocation_status = d.edge_source.try_reset(nnz);
    if (allocation_status == hipSuccess) {
      d.edge_source_available = true;
      if (nnz != 0) {
        // edge_source consumes rowptr in a later dispatch.  Explicit worker
        // streams on gfx1151 require a real completion boundary for dependent
        // dispatches.
        synchronize_explicit_stream(stream);
        if (uses_32_bit_offsets) {
          build_edge_source_kernel<CompactRowOffset>
              <<<grid_for_items(h.rows), kBlockSize, 0, stream>>>(
                  h.rows, d.rowptr32.get(), d.edge_source.get());
        } else {
          build_edge_source_kernel<Offset>
              <<<grid_for_items(h.rows), kBlockSize, 0, stream>>>(
                  h.rows, d.rowptr64.get(), d.edge_source.get());
        }
        DS_DELTA_HIP_CHECK(hipGetLastError());
      }
    } else if (allocation_status == hipErrorOutOfMemory) {
      // This allocation is optional. Clear the swallowed per-thread runtime
      // error so the next launch check does not turn the intended legacy
      // fallback into a delayed exception.
      (void)hipGetLastError();
    } else {
      DS_DELTA_HIP_CHECK(allocation_status);
    }
  }
  if (nnz != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        d.colind.get(), h.colind.data(),
        sssp_capacity::checked_bytes<Index>(nnz), hipMemcpyHostToDevice,
        stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        d.values.get(), h.values.data(),
        sssp_capacity::checked_bytes<float>(nnz), hipMemcpyHostToDevice,
        stream));
  }
  // The host vectors may be temporary, and PathFinder worker streams consume
  // the graph immediately after construction. Publish only a completed upload.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
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
  unsigned long long controller_grid_barriers = 0;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      counters.data(),
      scratch.telemetry_counters.get(),
      sssp_capacity::checked_bytes<unsigned long long>(counters.size()),
      hipMemcpyDeviceToHost,
      stream));
  if (telemetry.controller_backend ==
          DeltaSteppingCsrControllerBackend::kCooperativeGrid &&
      scratch.controller_state.size() != 0 &&
      telemetry.controller_publications != 0) {
    const auto* const controller_bytes =
        reinterpret_cast<const unsigned char*>(scratch.controller_state.get());
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        &controller_grid_barriers,
        controller_bytes +
            offsetof(CooperativeDeltaControllerState, grid_barriers),
        sizeof(controller_grid_barriers), hipMemcpyDeviceToHost, stream));
  }
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
  telemetry.cooperative_grid_barriers = controller_grid_barriers;
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
                                scratch.host_scalar.get());
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
                                scratch.host_scalar.get());
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

float copy_dist_value_to_host(const float* d_dist, int vertex, hipStream_t stream) {
  return copy_scalar_to_host(d_dist + vertex, stream);
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

template <typename RowOffset>
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
      scratch.touched_count.get(), stream, scratch.host_scalar.get());
  if (touched_count < 0 || static_cast<Offset>(touched_count) > graph.rows) {
    throw std::runtime_error(
        "delta predecessor touched count is outside graph bounds");
  }
  if (touched_count == 0) {
    return touched_count;
  }
  materialize_predecessors_kernel<RowOffset>
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

template <typename RowOffset, TargetPathParentMode ParentMode>
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
    measure_edge_parent_target_paths_kernel<RowOffset>
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

  result.target_distances.resize(targets.size());
  result.target_sources.resize(targets.size());
  std::vector<int> path_lengths(targets.size());
  std::vector<int> path_status(targets.size());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_distances.data(),
                                    scratch.target_distances.get(),
                                    sssp_capacity::checked_bytes<float>(
                                        targets.size()),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_lengths.data(),
                                    scratch.target_path_lengths.get(),
                                    sssp_capacity::checked_bytes<int>(
                                        targets.size()),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_status.data(),
                                    scratch.target_path_status.get(),
                                    sssp_capacity::checked_bytes<int>(
                                        targets.size()),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));

  const std::size_t target_offset_count =
      sssp_capacity::checked_target_offset_count(targets.size());
  result.target_path_offsets.assign(target_offset_count, 0);
  result.target_edge_offsets.assign(target_offset_count, 0);
  bool all_targets_reached = true;
  std::size_t total_nodes = 0;
  std::size_t total_edges = 0;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    result.target_path_offsets[i] = static_cast<int>(total_nodes);
    result.target_edge_offsets[i] = static_cast<int>(total_edges);
    if (std::isfinite(result.target_distances[i]) &&
        (path_status[i] == 0 || path_lengths[i] <= 0)) {
      throw std::runtime_error(
          "delta predecessor path failed device validation during "
          "measurement for target index " +
          std::to_string(i));
    }
    if (path_status[i] == 0 || path_lengths[i] <= 0 ||
        !std::isfinite(result.target_distances[i])) {
      all_targets_reached = false;
      continue;
    }
    total_nodes = sssp_capacity::checked_add(
        total_nodes, static_cast<std::size_t>(path_lengths[i]));
    total_edges = sssp_capacity::checked_add(
        total_edges, static_cast<std::size_t>(path_lengths[i] - 1));
    if (total_nodes > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        total_edges > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("compact target paths are too large for int offsets");
    }
  }
  result.target_path_offsets[targets.size()] = static_cast<int>(total_nodes);
  result.target_edge_offsets[targets.size()] = static_cast<int>(total_edges);

  if (targets.empty()) {
    result.target_reached = true;
    return;
  }

  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.target_node_offsets.get(),
                                      result.target_path_offsets.data(),
                                      sssp_capacity::checked_bytes<int>(
                                          target_offset_count),
                                      hipMemcpyHostToDevice,
                                      stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.target_edge_offsets.get(),
                                      result.target_edge_offsets.data(),
                                      sssp_capacity::checked_bytes<int>(
                                          target_offset_count),
                                      hipMemcpyHostToDevice,
                                      stream));
    synchronize_explicit_stream(stream);

    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
      fill_edge_parent_target_paths_kernel<RowOffset>
          <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
              scratch.targets.get(), target_count, scratch.rows, graph.nnz,
              graph.rowptr, graph.colind, graph.values, vertex_costs,
              edge_source, scratch.dist.get(), scratch.parent_key.get(),
              scratch.target_path_lengths.get(),
              scratch.target_path_status.get(), scratch.target_sources.get(),
              scratch.target_node_offsets.get(),
              scratch.target_edge_offsets.get(),
              scratch.compact_path_nodes.get(),
              scratch.compact_path_edges.get());
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
              scratch.compact_path_edges.get());
    }
    DS_DELTA_HIP_CHECK(hipGetLastError());
    // Path status is validated by the fill kernel and copied immediately
    // below. Publish the complete path and status together on explicit worker
    // streams before any D2H consumer starts.
    synchronize_explicit_stream(stream);
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_path_nodes.data(),
                                      scratch.compact_path_nodes.get(),
                                      sssp_capacity::checked_bytes<int>(
                                          total_nodes),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  if (total_edges != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_path_edges.data(),
                                      scratch.compact_path_edges.get(),
                                      sssp_capacity::checked_bytes<Offset>(
                                          total_edges),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_sources.data(),
                                    scratch.target_sources.get(),
                                    sssp_capacity::checked_bytes<int>(
                                        targets.size()),
                                    hipMemcpyDeviceToHost,
                                    stream));
  if (total_nodes != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_status.data(),
                                      scratch.target_path_status.get(),
                                      sssp_capacity::checked_bytes<int>(
                                          targets.size()),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  for (std::size_t i = 0; i < targets.size(); ++i) {
    if (path_lengths[i] > 0 && path_status[i] == 0) {
      throw std::runtime_error(
          "delta predecessor path failed device validation for target index " +
          std::to_string(i));
    }
  }
  result.target_reached = all_targets_reached;
}

template <typename RowOffset, bool CollectTelemetry>
DeltaSteppingCsrResult run_unit_weight_specialization(
    const DeviceCsrView<RowOffset>& graph,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    float exclusive_distance_limit,
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
  mark_unit_target_multiplicity_kernel
      <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
          scratch.targets.get(),
          target_count,
          scratch.in_pending.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  initialize_unit_sources_kernel
      <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
          scratch.sources.get(),
          source_count,
          initially_found,
          target_count,
          max_depth,
          scratch.dist.get(),
          scratch.pred_node.get(),
          scratch.pred_edge.get(),
          scratch.current_queue.get(),
          scratch.unit_status.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
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
  const bool use_device_controller = stream == nullptr;

  while (current_count > 0 && found_count < target_count &&
         result.iterations_used < max_depth) {
    const int previous_queue_tail = queue_tail;
    const int previous_found_count = found_count;
    const int previous_frontier_end = frontier_end;
    const int previous_depth = result.iterations_used;

    if (!use_device_controller) {
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
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.host_unit_status.get(),
                                        scratch.unit_status.get(),
                                        sssp_capacity::checked_bytes<int>(
                                            kUnitStatusCount),
                                        hipMemcpyDeviceToHost, stream));
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
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
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.host_unit_status.get(),
                                        scratch.unit_status.get(),
                                        sssp_capacity::checked_bytes<int>(
                                            kUnitStatusCount),
                                        hipMemcpyDeviceToHost, stream));
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
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
                            int* d_block_mins,
                            int* h_block_mins,
                            bool pending_updates_synchronized,
                            hipStream_t stream,
                            unsigned long long* telemetry_counters) {
  if (launch_blocks <= 0 ||
      launch_blocks > kMaxHostCheckedReductionBlocks) {
    throw std::logic_error(
        "delta pending-bucket reduction block count is outside bounds");
  }
  // A preceding host observation already completed the producer in the
  // multi-target PathFinder path.  Retain the established explicit-stream
  // guard only for low-level no-target runs whose heavy phase has no such
  // completion boundary.
  if (!pending_updates_synchronized) {
    synchronize_explicit_stream(stream);
  }
  reduce_min_pending_bucket_kernel<CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
      d_pending_queue, d_pending_count, current_bucket, delta, d_dist,
      d_in_pending, d_block_mins, telemetry_counters);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      h_block_mins,
      d_block_mins,
      sssp_capacity::checked_bytes<int>(
          static_cast<std::size_t>(launch_blocks)),
      hipMemcpyDeviceToHost,
      stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return *std::min_element(h_block_mins,
                           h_block_mins + launch_blocks);
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
                             scratch.host_scalar.get());
}

template <typename RowOffset,
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
    DeltaSteppingCsrControllerMode controller_mode,
    std::uint32_t controller_batch_size,
    std::uint32_t controller_generation_seed_for_testing,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (max_iters < 0) max_iters = std::numeric_limits<int>::max();

  const DeltaSteppingCsrControllerPolicy controller_policy{
      controller_mode, controller_batch_size};
  delta_stepping_validate_controller_policy(controller_policy);

  const Offset n = d_adjacency.rows;
  const bool cooperative_controller_requested =
      controller_mode == DeltaSteppingCsrControllerMode::kFusedHostChecked ||
      controller_mode == DeltaSteppingCsrControllerMode::kReducedRoundTrip;
  const bool use_query_batch_coordinator =
      scratch.controller_batch_coordinator != nullptr;
  if (cooperative_controller_requested &&
      scratch.controller_concurrency_hint > 1 &&
      !use_query_batch_coordinator && progress_callback == nullptr) {
    throw std::invalid_argument(
        "multiworker cooperative Delta-Stepping requires one shared query "
        "batch coordinator");
  }
  const std::uint32_t cooperative_action_budget =
      delta_stepping_effective_controller_batch_size(controller_policy);
  CooperativeLaunchConfiguration cooperative_configuration;
  DeltaSteppingCsrControllerFallbackReason controller_fallback_reason =
      DeltaSteppingCsrControllerFallbackReason::kNone;
  if (cooperative_controller_requested && progress_callback != nullptr) {
    controller_fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kProgressCallback;
  } else if (cooperative_controller_requested && UseCurrentGenerations &&
             !controller_generation_batch_is_representable(
                 cooperative_action_budget)) {
    controller_fallback_reason =
        DeltaSteppingCsrControllerFallbackReason::kGenerationBudget;
  } else if (cooperative_controller_requested) {
    cooperative_configuration =
        cooperative_launch_configuration<RowOffset, UseCurrentGenerations,
                                         TrackParents, UseEdgeParent,
                                         CollectTelemetry>(scratch, n);
    controller_fallback_reason = cooperative_configuration.fallback_reason;
  }
  const bool use_cooperative_controller =
      cooperative_controller_requested && progress_callback == nullptr &&
      controller_fallback_reason ==
          DeltaSteppingCsrControllerFallbackReason::kNone &&
      cooperative_configuration.max_blocks > 0;
  if constexpr (CollectTelemetry) {
    telemetry->effective_controller_mode =
        use_cooperative_controller
            ? controller_mode
            : DeltaSteppingCsrControllerMode::kHostChecked;
    telemetry->effective_controller_batch_size =
        use_cooperative_controller ? cooperative_action_budget : 1U;
    telemetry->controller_fallback =
        cooperative_controller_requested && !use_cooperative_controller;
    telemetry->controller_backend =
        use_cooperative_controller
            ? DeltaSteppingCsrControllerBackend::kCooperativeGrid
            : DeltaSteppingCsrControllerBackend::kScalarHost;
    telemetry->controller_fallback_reason = controller_fallback_reason;
    telemetry->cooperative_active_blocks_per_compute_unit =
        static_cast<std::uint32_t>(std::max(
            0, cooperative_configuration.active_blocks_per_compute_unit));
    telemetry->cooperative_compute_units = static_cast<std::uint32_t>(
        std::max(0, cooperative_configuration.compute_units));
  }
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
  // Keep the per-block staging lazy: a supported reduced controller never
  // performs this host reduction.  Allocate before any query work is enqueued
  // so allocation failure cannot leave partially mutated asynchronous state.
  if (!use_cooperative_controller) {
    scratch.ensure_host_checked_reduction_storage();
  } else {
    // Allocate publication storage before any query work is enqueued.
    // hipMalloc/hipHostMalloc may synchronize internally, but this first-use
    // cost remains outside the repeated controller path.
    scratch.ensure_controller_storage();
  }
  if constexpr (CollectTelemetry) {
    prepare_device_telemetry(scratch, stream);
  }
  const bool target_is_source =
      !use_target_set && target >= 0 && zero_distance_within_limit &&
      is_effective_source(target);

  CooperativeDeltaControllerState initial_controller_state{};
  if (use_cooperative_controller) {
    if (scratch.controller_query_sequence ==
        std::numeric_limits<std::uint32_t>::max()) {
      scratch.controller_query_sequence = 1;
    } else {
      ++scratch.controller_query_sequence;
      if (scratch.controller_query_sequence == 0) {
        scratch.controller_query_sequence = 1;
      }
    }
    initial_controller_state.descriptor.version =
        DeltaSteppingCsrControllerDescriptor::kVersion;
    initial_controller_state.descriptor.query_sequence =
        scratch.controller_query_sequence;
    initial_controller_state.descriptor.phase =
        DeltaSteppingCsrControllerPhase::kLightClosure;
    initial_controller_state.descriptor.action =
        DeltaSteppingCsrControllerAction::kContinueDevice;
    initial_controller_state.descriptor.current_count =
        static_cast<std::uint32_t>(source_count);
    initial_controller_state.descriptor.current_bucket = 0;
    initial_controller_state.descriptor.next_bucket =
        kDeltaSteppingCsrNoControllerBucket;
  }

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
  if constexpr (UseCurrentGenerations) {
    if (controller_generation_seed_for_testing != 0) {
      DS_DELTA_HIP_CHECK(hipMemsetAsync(
          scratch.in_current.get(),
          0,
          sssp_capacity::checked_bytes<std::uint32_t>(
              static_cast<std::size_t>(n)),
          stream));
      // This hook deliberately exercises token reuse. Publish the full clear
      // even on the null stream before installing the synthetic predecessor.
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      scratch.current_generation =
          controller_generation_seed_for_testing;
    }
  }
  const std::uint32_t source_generation =
      acquire_current_generation<UseCurrentGenerations>(scratch, stream);
  try {
    if (use_cooperative_controller) {
      // The controller state and source frontier are independent producers.
      // Enqueue both before the existing source-publication completion so a
      // cooperative query does not add a third setup drain.
      *scratch.host_controller_state->get() = initial_controller_state;
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(
          scratch.controller_state.get(), scratch.host_controller_state->get(),
          sizeof(initial_controller_state), hipMemcpyHostToDevice, stream));
    }
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
  } catch (...) {
    // A synchronous launch rejection can leave the preceding pinned-state
    // upload queued. Complete it before callers may retry and overwrite that
    // workspace-owned staging record.
    const std::exception_ptr setup_exception = std::current_exception();
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    std::rethrow_exception(setup_exception);
  }

  int* current_queue = scratch.current_queue.get();
  int* next_queue = scratch.next_queue.get();
  int* current_count_device = scratch.current_count.get();
  int* next_count_device = scratch.next_count.get();
  int* pending_queue = scratch.pending_a.get();
  int* pending_scratch = scratch.pending_b.get();
  int* pending_count_device = scratch.pending_count.get();
  int* pending_scratch_count_device = scratch.new_pending_count.get();
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
  const int device_count_blocks =
      std::min(grid_for_items(n), kMaxHostCheckedReductionBlocks);
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
          scratch.touched_count.get(), stream, scratch.host_scalar.get());
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

  if (use_cooperative_controller && !result.stopped_on_target &&
      max_iters > 0 &&
      !(has_distance_limit && current_bucket > last_allowed_bucket)) {
    PATHFINDER_PROFILE_RANGE("delta_step.cooperative_controller");
    std::uint32_t previous_publication = 0;
    std::uint64_t coordinator_query_sequence = 0;
    DeltaSteppingCsrControllerDescriptor previous_descriptor =
        initial_controller_state.descriptor;
    try {
      while (true) {
        std::uint32_t generation_first = 0;
        if (!reserve_controller_generations<UseCurrentGenerations>(
                scratch, cooperative_action_budget, stream,
                &generation_first)) {
          throw std::runtime_error(
              "Delta-Stepping controller generation budget is not "
              "representable");
        }
        CooperativeDeltaControllerArgs<RowOffset> args{};
        args.rows = n;
        args.rowptr = d_adjacency.rowptr;
        args.colind = d_adjacency.colind;
        args.values = d_adjacency.values;
        args.vertex_costs = vertex_costs;
        args.delta = delta;
        args.exclusive_distance_limit = exclusive_distance_limit;
        args.dist = scratch.dist.get();
        args.parent_key = scratch.parent_key.get();
        args.in_current = scratch.in_current.get();
        args.in_pending = scratch.in_pending.get();
        args.in_heavy = scratch.in_heavy.get();
        args.touched_queue = scratch.touched_queue.get();
        args.touched_count = scratch.touched_count.get();
        args.current_queue_0 = scratch.current_queue.get();
        args.current_queue_1 = scratch.next_queue.get();
        args.current_count_0 = scratch.current_count.get();
        args.current_count_1 = scratch.next_count.get();
        args.pending_queue_0 = scratch.pending_a.get();
        args.pending_queue_1 = scratch.pending_b.get();
        args.pending_count_0 = scratch.pending_count.get();
        args.pending_count_1 = scratch.new_pending_count.get();
        args.heavy_queue = scratch.heavy_queue.get();
        args.heavy_count = scratch.heavy_count.get();
        args.min_pending_bucket = scratch.min_pending_bucket.get();
        args.targets = use_target_set ? scratch.targets.get() : nullptr;
        args.target_count = target_count;
        args.target_settled =
            use_target_set ? scratch.target_settled.get() : nullptr;
        args.settled_target_count = scratch.settled_target_count.get();
        args.scalar_target = target;
        args.max_iters = max_iters;
        args.last_allowed_bucket = last_allowed_bucket;
        args.batch_size = cooperative_action_budget;
        args.generation_first = generation_first;
        args.has_distance_limit = has_distance_limit ? 1 : 0;
        args.skip_heavy_edges = skip_heavy_edges ? 1 : 0;
        args.telemetry_counters =
            CollectTelemetry ? scratch.telemetry_counters.get() : nullptr;
        args.state = scratch.controller_state.get();
        const int launch_blocks = cooperative_configuration.max_blocks;
        if (launch_blocks <= 0) {
          throw std::runtime_error(
              "Delta-Stepping cooperative controller selected an empty "
              "concurrency-partitioned grid");
        }
        if constexpr (CollectTelemetry) {
          const std::uint32_t observed_blocks =
              static_cast<std::uint32_t>(launch_blocks);
          if (telemetry->cooperative_grid_blocks_min == 0) {
            telemetry->cooperative_grid_blocks_min = observed_blocks;
          } else {
            telemetry->cooperative_grid_blocks_min = std::min(
                telemetry->cooperative_grid_blocks_min, observed_blocks);
          }
          telemetry->cooperative_grid_blocks_max =
              std::max(telemetry->cooperative_grid_blocks_max, observed_blocks);
          if (!use_query_batch_coordinator) {
            ++telemetry->cooperative_launches;
          }
          telemetry->controller_action_slots_budgeted +=
              cooperative_action_budget;
        }
        DeltaSteppingCsrControllerDescriptor descriptor{};
        if (use_query_batch_coordinator) {
          const CooperativeDeltaBatchSubmitResult batch_result =
              submit_cooperative_delta_controller_batch<
                  RowOffset, UseCurrentGenerations, TrackParents,
                  UseEdgeParent, CollectTelemetry>(
                  *scratch.controller_batch_coordinator, args,
                  coordinator_query_sequence,
                  scratch.controller_query_sequence,
                  previous_publication + 1U,
                  previous_descriptor.light_rounds,
                  cooperative_action_budget,
                  static_cast<std::uint32_t>(launch_blocks),
                  static_cast<std::uint32_t>(
                      cooperative_configuration
                          .selected_blocks_per_compute_unit),
                  static_cast<std::uint32_t>(
                      cooperative_configuration
                          .active_blocks_per_compute_unit),
                  static_cast<std::uint32_t>(
                      cooperative_configuration.compute_units));
          descriptor = batch_result.descriptor;
          coordinator_query_sequence =
              batch_result.logical_query_sequence;
          if (batch_result.grid_blocks !=
              static_cast<std::uint32_t>(launch_blocks)) {
            throw std::runtime_error(
                "Delta-Stepping query batch returned a mismatched grid "
                "configuration");
          }
        } else {
          launch_cooperative_delta_controller<
              RowOffset, UseCurrentGenerations, TrackParents, UseEdgeParent,
              CollectTelemetry>(args, launch_blocks, stream);
          descriptor = copy_controller_descriptor_to_host(scratch, stream);
        }
        ++controller_round_trips;
        if (!delta_stepping_controller_descriptor_is_valid(descriptor) ||
            descriptor.query_sequence != scratch.controller_query_sequence ||
            descriptor.publication_sequence != previous_publication + 1U) {
          throw std::runtime_error(
              "Delta-Stepping cooperative controller published an invalid "
              "descriptor");
        }
        const std::uint64_t completed_actions =
            static_cast<std::uint64_t>(descriptor.light_rounds) -
            static_cast<std::uint64_t>(previous_descriptor.light_rounds);
        if (descriptor.light_rounds <= previous_descriptor.light_rounds ||
            completed_actions > cooperative_action_budget ||
            (descriptor.status == DeltaSteppingCsrControllerStatus::kNone &&
             completed_actions != cooperative_action_budget)) {
          throw std::runtime_error(
              "Delta-Stepping cooperative controller published an invalid "
              "action count");
        }
        previous_publication = descriptor.publication_sequence;
        if constexpr (CollectTelemetry) {
          telemetry->controller_actions_completed += completed_actions;
          telemetry->controller_unused_action_slots +=
              static_cast<std::uint64_t>(cooperative_action_budget) -
              completed_actions;
          ++telemetry->controller_publications;
          if (descriptor.status == DeltaSteppingCsrControllerStatus::kNone) {
            ++telemetry->controller_nonterminal_publications;
          } else {
            ++telemetry->controller_terminal_publications;
          }
        }
        previous_descriptor = descriptor;
        total_light_rounds = descriptor.light_rounds;
        result.iterations_used = static_cast<int>(descriptor.iterations);
        if (descriptor.status ==
            DeltaSteppingCsrControllerStatus::kNone) {
          if (descriptor.action !=
              DeltaSteppingCsrControllerAction::kPublishHostCheck) {
            throw std::runtime_error(
                "Delta-Stepping cooperative controller published a "
                "nonterminal action");
          }
          continue;
        }
        if (delta_stepping_controller_has_status(
                descriptor.status,
                DeltaSteppingCsrControllerStatus::kInvalidState) ||
            delta_stepping_controller_has_status(
                descriptor.status,
                DeltaSteppingCsrControllerStatus::kQueueOverflow)) {
          throw std::runtime_error(
              "Delta-Stepping cooperative controller detected invalid or "
              "overflowed queue state");
        }
        if (!skip_heavy_edges) {
          heavy_edge_phases = descriptor.iterations;
          const bool final_bucket_skipped_heavy =
              !delta_stepping_controller_has_status(
                  descriptor.status,
                  DeltaSteppingCsrControllerStatus::kIterationLimit) &&
              descriptor.current_bucket ==
                  static_cast<std::uint64_t>(kNoBucket - 1);
          if (final_bucket_skipped_heavy && heavy_edge_phases != 0) {
            --heavy_edge_phases;
          }
        }
        if (delta_stepping_controller_has_status(
                descriptor.status,
                DeltaSteppingCsrControllerStatus::kTargetSettled)) {
          result.target_reached = true;
          result.stopped_on_target = true;
        } else if (delta_stepping_controller_has_status(
                       descriptor.status,
                       DeltaSteppingCsrControllerStatus::kComplete)) {
          if (has_distance_limit) {
            result.stopped_on_distance_limit = true;
          } else {
            result.converged = true;
          }
        } else if (!delta_stepping_controller_has_status(
                       descriptor.status,
                       DeltaSteppingCsrControllerStatus::kIterationLimit)) {
          throw std::runtime_error(
              "Delta-Stepping cooperative controller published an unknown "
              "terminal status");
        }
        break;
      }
    } catch (...) {
      const std::exception_ptr controller_exception =
          std::current_exception();
      fully_reinitialize_after_controller_error<TrackParents, UseEdgeParent>(
          scratch, n, inf, stream);
      std::rethrow_exception(controller_exception);
    }
  } else if (use_cooperative_controller && has_distance_limit &&
             current_bucket > last_allowed_bucket &&
             !result.stopped_on_target) {
    result.stopped_on_distance_limit = true;
  }

  if (!use_cooperative_controller) {
    PATHFINDER_PROFILE_RANGE("delta_step.scalar_host_controller");
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
          launch_relax_light_edges<RowOffset, UseCurrentGenerations,
                                   TrackParents, UseEdgeParent, true, false,
                                   true, CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, pending_count_device,
              stream);
        } else if (terminal_bucket || skip_heavy_edges) {
          launch_relax_light_edges<RowOffset, UseCurrentGenerations,
                                   TrackParents, UseEdgeParent, false, false,
                                   true, CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, pending_count_device,
              stream);
        } else if (vertex_costs != nullptr) {
          launch_relax_light_edges<RowOffset, UseCurrentGenerations,
                                   TrackParents, UseEdgeParent, true, true,
                                   false, CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, pending_count_device,
              stream);
        } else {
          launch_relax_light_edges<RowOffset, UseCurrentGenerations,
                                   TrackParents, UseEdgeParent, false, true,
                                   false, CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              exclusive_distance_limit, next_queue, next_count_device,
              next_current_generation, pending_queue, pending_count_device,
              stream);
        }
        std::swap(current_queue, next_queue);
        std::swap(current_count_device, next_count_device);
        ++light_rounds;
        ++total_light_rounds;
      }
      current_count = copy_scalar_to_host(
          current_count_device, stream, scratch.host_scalar.get());
      ++controller_round_trips;
      if (current_count < 0 || static_cast<Offset>(current_count) > n) {
        throw std::runtime_error(
            "delta light-closure frontier count is outside graph bounds");
      }
    }

    // The final frontier-count transfer completed every light-phase update.
    // A heavy launch below makes the pending queue live again until a target
    // observation (or an explicit low-level guard) completes it.
    bool pending_updates_synchronized = true;
    if (!skip_heavy_edges && current_bucket != kNoBucket - 1) {
      ++heavy_edge_phases;
      if (vertex_costs != nullptr) {
        launch_relax_heavy_edges<RowOffset, TrackParents, UseEdgeParent, true,
                                 CollectTelemetry>(
            d_adjacency, scratch, vertex_costs, device_count_blocks,
            current_bucket, delta, exclusive_distance_limit, pending_queue,
            pending_count_device, stream);
      } else {
        launch_relax_heavy_edges<RowOffset, TrackParents, UseEdgeParent,
                                 false, CollectTelemetry>(
            d_adjacency, scratch, nullptr, device_count_blocks,
            current_bucket, delta, exclusive_distance_limit, pending_queue,
            pending_count_device, stream);
      }
      pending_updates_synchronized = false;
      // Vector-target settlement is the only immediate device consumer here.
      // Preserve the established gfx1151 producer boundary before it. Scalar
      // target copies synchronize on their own D2H transfer.
      if (use_target_set) {
        synchronize_explicit_stream(stream);
        pending_updates_synchronized = true;
      }
    }

    result.iterations_used = iter + 1;
    if (use_target_set) {
      const int settled_count =
          mark_and_count_settled_targets(scratch, target_count, current_bucket,
                                         delta, stream);
      pending_updates_synchronized = true;
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
      const float target_distance = copy_dist_value_to_host(scratch.dist.get(), target, stream);
      pending_updates_synchronized = true;
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

    // The inactive pending queue has n entries, while device_count_blocks is
    // no greater than min(ceil(n / kBlockSize), 256).  Its first entries are
    // therefore safe temporary storage for block minima.  The D2H completion
    // in this helper finishes those writes before compaction overwrites the
    // inactive queue with the next pending generation.
    const int next_bucket = find_min_pending_bucket<CollectTelemetry>(
        pending_queue, pending_count_device, device_count_blocks,
        current_bucket, delta, scratch.dist.get(), scratch.in_pending.get(),
        pending_scratch,
        scratch.host_pending_bucket_block_mins->get(),
        pending_updates_synchronized,
        stream,
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
    reset_int_zero_async(pending_scratch_count_device, stream);
    synchronize_explicit_stream(stream);
    const std::uint32_t compacted_current_generation =
        acquire_current_generation<UseCurrentGenerations>(scratch, stream);

    compact_pending_to_current_bucket_kernel<UseCurrentGenerations,
                                             CollectTelemetry>
        <<<device_count_blocks, kBlockSize, 0, stream>>>(
        pending_queue, pending_count_device, current_bucket, delta,
        scratch.dist.get(),
        scratch.in_pending.get(), scratch.in_current.get(),
        compacted_current_generation, current_queue,
        scratch.current_count.get(), pending_scratch,
        pending_scratch_count_device,
        CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
    DS_DELTA_HIP_CHECK(hipGetLastError());

    current_count = copy_scalar_to_host(
        scratch.current_count.get(), stream, scratch.host_scalar.get());
    ++controller_round_trips;
    if (current_count < 0 || static_cast<Offset>(current_count) > n) {
      throw std::runtime_error(
          "delta compacted frontier count is outside graph bounds");
    }
    // The current-count D2H completion above also certifies the compaction's
    // queue and pending-count outputs.  Advance queue and count parity
    // together instead of copying the count back to one fixed address and
    // introducing another explicit stream barrier.
    std::swap(pending_queue, pending_scratch);
    std::swap(pending_count_device, pending_scratch_count_device);
    }
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
                                     TargetPathParentMode::kCompactEdge>(
          result, scratch, d_adjacency, vertex_costs, edge_source, *targets,
          stream, settled_target_filter);
      touched_count_for_reset = *scratch.host_scalar.get();
    } else {
      if (use_target_set || target >= 0) {
        touched_count_for_reset = materialize_predecessors_from_keys<RowOffset>(
            d_adjacency, scratch, vertex_costs, stream);
      }
      if (use_target_set) {
        extract_target_paths_to_result<
            RowOffset, TargetPathParentMode::kLegacyPredecessor>(
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
        scratch.touched_count.get(), stream, scratch.host_scalar.get());
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
          scratch.touched_count.get(), stream, scratch.host_scalar.get());
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
    DeltaSteppingCsrControllerMode controller_mode,
    std::uint32_t controller_batch_size,
    std::uint32_t controller_generation_seed_for_testing,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (current_membership_mode ==
      DeltaSteppingCsrCurrentMembershipMode::kGeneration) {
    if (telemetry != nullptr) {
      return run_delta_stepping_impl<RowOffset, true, TrackParents,
                                     UseEdgeParent, true>(
          d_adjacency, edge_source, scratch, sources, target, targets,
          vertex_costs, skip_heavy_edges, delta, max_iters,
          exclusive_distance_limit, stream,
          progress_callback, progress_user_data, controller_mode,
          controller_batch_size, controller_generation_seed_for_testing,
          telemetry);
    }
    return run_delta_stepping_impl<RowOffset, true, TrackParents,
                                   UseEdgeParent, false>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        exclusive_distance_limit, stream,
        progress_callback, progress_user_data, controller_mode,
        controller_batch_size, controller_generation_seed_for_testing,
        nullptr);
  }
  if (current_membership_mode !=
      DeltaSteppingCsrCurrentMembershipMode::kBoolean) {
    throw std::invalid_argument(
        "unknown Delta-Stepping current-membership mode");
  }
  if (telemetry != nullptr) {
    return run_delta_stepping_impl<RowOffset, false, TrackParents,
                                   UseEdgeParent, true>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        exclusive_distance_limit, stream,
        progress_callback, progress_user_data, controller_mode,
        controller_batch_size, controller_generation_seed_for_testing,
        telemetry);
  }
  return run_delta_stepping_impl<RowOffset, false, TrackParents,
                                 UseEdgeParent, false>(
      d_adjacency, edge_source, scratch, sources, target, targets,
      vertex_costs, skip_heavy_edges, delta, max_iters,
      exclusive_distance_limit, stream,
      progress_callback, progress_user_data, controller_mode,
      controller_batch_size, controller_generation_seed_for_testing,
      nullptr);
}

void begin_telemetry_record(DeltaSteppingCsrTelemetry* telemetry,
                            DeltaSteppingCsrExecutionPath path,
                            float delta,
                            bool force_generic,
                            bool force_legacy_parent,
                            bool has_vertex_costs,
                            bool all_edges_light,
                            bool compact_parent_fallback,
                            DeltaSteppingCsrControllerMode controller_mode,
                            std::uint32_t controller_batch_size) {
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
  telemetry->requested_controller_mode = controller_mode;
  telemetry->effective_controller_mode =
      DeltaSteppingCsrControllerMode::kHostChecked;
  telemetry->requested_controller_batch_size = controller_batch_size;
  telemetry->effective_controller_batch_size = 1;
  telemetry->controller_fallback = false;
  telemetry->controller_backend = DeltaSteppingCsrControllerBackend::kNotRun;
  telemetry->controller_fallback_reason =
      DeltaSteppingCsrControllerFallbackReason::kNone;
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

const char* delta_stepping_controller_backend_name(
    DeltaSteppingCsrControllerBackend backend) noexcept {
  switch (backend) {
    case DeltaSteppingCsrControllerBackend::kNotRun:
      return "not_run";
    case DeltaSteppingCsrControllerBackend::kScalarHost:
      return "scalar_host";
    case DeltaSteppingCsrControllerBackend::kCooperativeGrid:
      return "cooperative_grid";
    case DeltaSteppingCsrControllerBackend::kExactUnit:
      return "exact_unit";
  }
  return "unknown";
}

const char* delta_stepping_controller_fallback_reason_name(
    DeltaSteppingCsrControllerFallbackReason reason) noexcept {
  switch (reason) {
    case DeltaSteppingCsrControllerFallbackReason::kNone:
      return "none";
    case DeltaSteppingCsrControllerFallbackReason::kProgressCallback:
      return "progress_callback";
    case DeltaSteppingCsrControllerFallbackReason::kExactUnitBypass:
      return "exact_unit_bypass";
    case DeltaSteppingCsrControllerFallbackReason::kGenerationBudget:
      return "generation_budget";
    case DeltaSteppingCsrControllerFallbackReason::kCooperativeUnsupported:
      return "cooperative_unsupported";
    case DeltaSteppingCsrControllerFallbackReason::kCapabilityQueryFailed:
      return "capability_query_failed";
    case DeltaSteppingCsrControllerFallbackReason::kOccupancyQueryFailed:
      return "occupancy_query_failed";
    case DeltaSteppingCsrControllerFallbackReason::kNoResidentGrid:
      return "no_resident_grid";
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
  delta_stepping_validate_controller_policy(
      {options.controller_mode, options.controller_batch_size});
  controller_mode_ = options.controller_mode;
  controller_batch_size_ = options.controller_batch_size;
  controller_generation_seed_for_testing_ =
      options.controller_generation_seed_for_testing;
  if (options.controller_concurrency_hint == 0) {
    throw std::invalid_argument(
        "Delta-Stepping controller concurrency hint must be positive");
  }
  if (options.controller_batch_coordinator != nullptr && stream == nullptr) {
    throw std::invalid_argument(
        "Delta-Stepping query batching requires an explicit producer stream");
  }
  impl_->scratch.controller_concurrency_hint =
      options.controller_concurrency_hint;
  impl_->scratch.controller_batch_coordinator =
      std::move(options.controller_batch_coordinator);
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
  delta_stepping_validate_controller_policy(
      {options.controller_mode, options.controller_batch_size});
  controller_mode_ = options.controller_mode;
  controller_batch_size_ = options.controller_batch_size;
  controller_generation_seed_for_testing_ =
      options.controller_generation_seed_for_testing;
  if (options.controller_concurrency_hint == 0) {
    throw std::invalid_argument(
        "Delta-Stepping controller concurrency hint must be positive");
  }
  if (options.controller_batch_coordinator != nullptr && stream == nullptr) {
    throw std::invalid_argument(
        "Delta-Stepping query batching requires an explicit producer stream");
  }
  impl_->scratch.controller_concurrency_hint =
      options.controller_concurrency_hint;
  impl_->scratch.controller_batch_coordinator =
      std::move(options.controller_batch_coordinator);
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
  impl_->has_exact_unit_edge_values =
      ds_delta_detail::has_exact_unit_edge_values(values);
  if (adjacency.nnz == 0) {
    impl_->max_edge_value = 0.0f;
    return;
  }
  impl_->max_edge_value = max_edge_value(values);
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(adjacency.values.get(),
                                    values.data(),
                                    sssp_capacity::checked_bytes<float>(
                                        values.size()),
                                    hipMemcpyHostToDevice,
                                    stream));
  // The caller retains ownership of values and may have passed a temporary.
  // Complete the upload before that storage can be destroyed or reused.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
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
  if (impl_->vertex_costs.size() < vertex_costs.size()) {
    impl_->vertex_costs.reset(vertex_costs.size());
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(impl_->vertex_costs.get(),
                                    vertex_costs.data(),
                                    sssp_capacity::checked_bytes<float>(
                                        vertex_costs.size()),
                                    hipMemcpyHostToDevice,
                                    stream));
  // Match update_values(): this API does not require callers to keep the host
  // vector alive after it returns.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  impl_->has_vertex_costs = true;
}

DeltaSteppingCsrAllocationState
DeltaSteppingCsrWorkspace::allocation_state() const noexcept {
  DeltaSteppingCsrAllocationState state;
  if (!impl_) return state;
  const auto& scratch = impl_->scratch;
  state.edge_source = impl_->adjacency().edge_source.size() != 0;
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
  state.path_edges = scratch.compact_path_edges.size() != 0;
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
      false,
      controller_mode_,
      controller_batch_size_);
  const auto run_typed = [&](const auto& graph) {
    using RowOffset = typename std::remove_cv<typename std::remove_pointer<
        decltype(graph.rowptr)>::type>::type;
    validate_device_csr_shape(graph, sources, -1, delta);
    return dispatch_delta_stepping_impl<RowOffset, false, false>(
        graph, nullptr, impl_->scratch, sources, -1, nullptr,
        impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
        skip_heavy_edges, delta, max_iters, active_distance_limit_, stream,
        progress_callback, progress_user_data, current_membership_mode_,
        controller_mode_, controller_batch_size_,
        controller_generation_seed_for_testing_, active_telemetry_);
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
      impl_->has_vertex_costs, skip_heavy_edges, false,
      controller_mode_, controller_batch_size_);
  const auto run_typed = [&](const auto& graph) {
    using RowOffset = typename std::remove_cv<typename std::remove_pointer<
        decltype(graph.rowptr)>::type>::type;
    validate_device_csr_shape(graph, sources, target, delta);
    return dispatch_delta_stepping_impl<RowOffset, true, false>(
        graph, nullptr, impl_->scratch, sources, target, nullptr,
        impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
        skip_heavy_edges, delta, max_iters, active_distance_limit_, stream,
        progress_callback, progress_user_data, current_membership_mode_,
        controller_mode_, controller_batch_size_,
        controller_generation_seed_for_testing_, active_telemetry_);
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
          impl_->max_edge_value <= delta, false,
          controller_mode_, controller_batch_size_);
      if (active_telemetry_ != nullptr) {
        active_telemetry_->controller_backend =
            DeltaSteppingCsrControllerBackend::kExactUnit;
        if (controller_mode_ != DeltaSteppingCsrControllerMode::kHostChecked) {
          // Cooperative controllers are defined only for classic generic
          // Delta. Exact-unit specialization bypasses them explicitly.
          active_telemetry_->controller_fallback = true;
          active_telemetry_->controller_fallback_reason =
              DeltaSteppingCsrControllerFallbackReason::kExactUnitBypass;
        }
      }
      if (active_telemetry_ != nullptr) {
        return run_unit_weight_specialization<RowOffset, true>(
            graph, impl_->scratch, sources, targets, delta,
            active_distance_limit_, stream, active_telemetry_);
      }
      return run_unit_weight_specialization<RowOffset, false>(
          graph, impl_->scratch, sources, targets, delta,
          active_distance_limit_, stream, nullptr);
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
          false, impl_->has_vertex_costs, skip_heavy_edges, false,
          controller_mode_, controller_batch_size_);
      return dispatch_delta_stepping_impl<RowOffset, true, true>(
          graph, adjacency.edge_source.get(), impl_->scratch, sources,
          -1, &targets, vertex_costs, skip_heavy_edges, delta, max_iters,
          active_distance_limit_, stream, progress_callback,
          progress_user_data, current_membership_mode_, controller_mode_,
          controller_batch_size_, controller_generation_seed_for_testing_,
          active_telemetry_);
    }
    const bool compact_parent_fallback =
        parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
        !adjacency.edge_source_available;
    begin_telemetry_record(
        active_telemetry_, DeltaSteppingCsrExecutionPath::kLegacyGeneric,
        delta,
        execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
        parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
        impl_->has_vertex_costs, skip_heavy_edges, compact_parent_fallback,
        controller_mode_, controller_batch_size_);
    return dispatch_delta_stepping_impl<RowOffset, true, false>(
        graph, nullptr, impl_->scratch, sources, -1, &targets,
        vertex_costs, skip_heavy_edges, delta, max_iters,
        active_distance_limit_, stream, progress_callback,
        progress_user_data, current_membership_mode_, controller_mode_,
        controller_batch_size_, controller_generation_seed_for_testing_,
        active_telemetry_);
  };
  if (adjacency.uses_32_bit_offsets) {
    return run_typed(adjacency.view<CompactRowOffset>());
  }
  return run_typed(adjacency.view<Offset>());
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
      DeltaSteppingCsrCurrentMembershipMode::kBoolean,
      DeltaSteppingCsrControllerMode::kHostChecked,
      kDeltaSteppingCsrRecommendedControllerBatchSize, 0, nullptr);
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
        DeltaSteppingCsrControllerMode::kHostChecked,
        kDeltaSteppingCsrRecommendedControllerBatchSize, 0, nullptr);
  }
  return dispatch_delta_stepping_impl<Offset, true, false>(
      d_adjacency.view<Offset>(), nullptr, scratch, sources, target, nullptr,
      nullptr, false, delta, max_iters,
      std::numeric_limits<float>::infinity(), stream, progress_callback,
      progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
      DeltaSteppingCsrControllerMode::kHostChecked,
      kDeltaSteppingCsrRecommendedControllerBatchSize, 0, nullptr);
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
        DeltaSteppingCsrControllerMode::kHostChecked,
        kDeltaSteppingCsrRecommendedControllerBatchSize, 0, nullptr);
  }
  return dispatch_delta_stepping_impl<Offset, true, false>(
      d_adjacency.view<Offset>(), nullptr, scratch, sources, target, nullptr,
      nullptr, false, delta, max_iters,
      std::numeric_limits<float>::infinity(), stream, progress_callback,
      progress_user_data, DeltaSteppingCsrCurrentMembershipMode::kBoolean,
      DeltaSteppingCsrControllerMode::kHostChecked,
      kDeltaSteppingCsrRecommendedControllerBatchSize, 0, nullptr);
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
