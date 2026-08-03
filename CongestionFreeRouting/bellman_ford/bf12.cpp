// Batched, independent multi-source/multi-target shortest paths for PathFinder.
// BF12 deliberately owns its kernels and never includes another implementation
// file.  Every mutable search address is q * V + vertex.

#include "bf12.hpp"

#include "bf12_worker_policy.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::mutex g_bf12_cooperative_controller_mutex;
std::atomic<std::uint64_t> g_bf12_live_cooperative_controllers{0};

template <typename T>
bool checked_add(T left, T right, T* result) {
  if (right > std::numeric_limits<T>::max() - left) return false;
  *result = left + right;
  return true;
}

template <typename T>
bool checked_multiply(T left, T right, T* result) {
  if (left != 0 && right > std::numeric_limits<T>::max() / left) return false;
  *result = left * right;
  return true;
}

std::size_t require_add(std::size_t left,
                        std::size_t right,
                        const char* what) {
  std::size_t result = 0;
  if (!checked_add(left, right, &result)) throw std::overflow_error(what);
  return result;
}

std::size_t require_multiply(std::size_t left,
                             std::size_t right,
                             const char* what) {
  std::size_t result = 0;
  if (!checked_multiply(left, right, &result)) throw std::overflow_error(what);
  return result;
}

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
  }
}

class HipEventTimer {
 public:
  explicit HipEventTimer(bool enabled) : enabled_(enabled) {
    if (!enabled_) return;
    check_hip(hipEventCreate(&begin_), "create BF12 timing start event");
    try {
      check_hip(hipEventCreate(&end_), "create BF12 timing end event");
    } catch (...) {
      (void)hipEventDestroy(begin_);
      begin_ = nullptr;
      throw;
    }
  }

  ~HipEventTimer() {
    if (end_) (void)hipEventDestroy(end_);
    if (begin_) (void)hipEventDestroy(begin_);
  }

  HipEventTimer(const HipEventTimer&) = delete;
  HipEventTimer& operator=(const HipEventTimer&) = delete;

  void begin(hipStream_t stream) {
    if (enabled_) {
      check_hip(hipEventRecord(begin_, stream), "record BF12 timing start event");
    }
  }

  void end(hipStream_t stream) {
    if (enabled_) {
      check_hip(hipEventRecord(end_, stream), "record BF12 timing end event");
    }
  }

  std::uint64_t elapsed_nanoseconds() const {
    if (!enabled_) return 0;
    float milliseconds = 0.0f;
    check_hip(hipEventElapsedTime(&milliseconds, begin_, end_),
              "read BF12 GPU timing events");
    if (!std::isfinite(milliseconds) || milliseconds < 0.0f) {
      throw std::runtime_error("BF12 GPU timing event returned an invalid value");
    }
    const long double nanoseconds =
        static_cast<long double>(milliseconds) * 1'000'000.0L;
    if (nanoseconds >=
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(nanoseconds + 0.5L);
  }

 private:
  bool enabled_ = false;
  hipEvent_t begin_ = nullptr;
  hipEvent_t end_ = nullptr;
};

template <typename T>
T* device_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  void* pointer = nullptr;
  check_hip(hipMalloc(&pointer, require_multiply(count, sizeof(T), what)), what);
  return static_cast<T*>(pointer);
}

template <typename T>
T* pinned_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  void* pointer = nullptr;
  check_hip(hipHostMalloc(&pointer, require_multiply(count, sizeof(T), what)),
            what);
  return static_cast<T*>(pointer);
}

template <typename T>
void device_free(T*& pointer) noexcept {
  if (pointer) (void)hipFree(pointer);
  pointer = nullptr;
}

template <typename T>
void pinned_free(T*& pointer) noexcept {
  if (pointer) (void)hipHostFree(pointer);
  pointer = nullptr;
}

std::size_t geometric_capacity(std::size_t current,
                               std::size_t required,
                               std::size_t limit) {
  if (required > limit) throw std::length_error("BF12 capacity exceeds limit");
  if (required == 0) return 0;
  std::size_t capacity = std::max<std::size_t>(1, current);
  while (capacity < required) {
    if (capacity > limit / 2) return limit;
    capacity *= 2;
  }
  return capacity;
}

struct DrainStreamOnException {
  hipStream_t stream = nullptr;
  int uncaught = std::uncaught_exceptions();
  ~DrainStreamOnException() {
    if (std::uncaught_exceptions() > uncaught) (void)hipStreamSynchronize(stream);
  }
};

class CooperativeControllerLease {
 public:
  CooperativeControllerLease()
      : lock_(g_bf12_cooperative_controller_mutex) {
    const std::uint64_t prior =
        g_bf12_live_cooperative_controllers.fetch_add(1,
                                                       std::memory_order_acq_rel);
    if (prior != 0) {
      g_bf12_live_cooperative_controllers.fetch_sub(1,
                                                     std::memory_order_acq_rel);
      throw std::logic_error("BF12 cooperative controller overlap detected");
    }
  }
  ~CooperativeControllerLease() {
    g_bf12_live_cooperative_controllers.fetch_sub(1,
                                                   std::memory_order_acq_rel);
  }

 private:
  std::unique_lock<std::mutex> lock_;
};

class ScopedHipDevice {
 public:
  explicit ScopedHipDevice(int owner) noexcept : owner_(owner) {
    if (hipGetDevice(&previous_) != hipSuccess) {
      (void)hipGetLastError();
      previous_ = -1;
      return;
    }
    if (previous_ != owner_ && hipSetDevice(owner_) != hipSuccess) {
      (void)hipGetLastError();
      previous_ = -1;
      return;
    }
    active_ = true;
  }
  ~ScopedHipDevice() {
    if (active_ && previous_ >= 0 && previous_ != owner_) {
      (void)hipSetDevice(previous_);
    }
  }
  bool active() const noexcept { return active_; }

 private:
  int owner_ = -1;
  int previous_ = -1;
  bool active_ = false;
};

}  // namespace

namespace rips_sssp_bf12 {

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;
using DeviceOffset = std::uint32_t;
using StateIndex = std::uint32_t;

constexpr unsigned int kNoPredecessor = 0xffffffffu;
constexpr unsigned int kInfinityBits = 0x7f800000u;
constexpr int kBlockSize = 256;
constexpr unsigned int kMaximumGridX = 65535u;
constexpr std::size_t kMaximumTelemetryRounds = 1u << 20;
constexpr std::int32_t kMissingCoordinate =
    routing::interchange::kMissingRouteCoordinate;

static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4 &&
                  sizeof(StateIndex) == 4 &&
                  std::numeric_limits<float>::is_iec559,
              "BF12 requires IEEE float32 and 32-bit composite indices");

struct DeviceGraph {
  Offset rows = 0;
  Offset nnz = 0;
  const DeviceOffset* rowptr = nullptr;
  const Index* destinations = nullptr;
  const float* edge_values = nullptr;
  const std::int32_t* route_end_x = nullptr;
  const std::int32_t* route_end_y = nullptr;
  const float* base_vertex_cost = nullptr;
};

struct DeviceGraphOwner {
  DeviceGraph view{};
  DeviceOffset* rowptr = nullptr;
  Index* destinations = nullptr;
  float* edge_values = nullptr;
  std::int32_t* route_end_x = nullptr;
  std::int32_t* route_end_y = nullptr;
  float* base_vertex_cost = nullptr;
};

struct DeviceBounds {
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
  int enabled = 0;
};

struct DeviceQueryDescriptor {
  std::uint32_t source_begin = 0;
  std::uint32_t source_count = 0;
  std::uint32_t target_begin = 0;
  std::uint32_t target_count = 0;
  DeviceBounds bounds{};
  int max_iterations = 0;
  int target_check_interval = 1;
  std::uint32_t original_query_index = 0;
};

struct DeviceQueryControl {
  int next_count = 0;
  int error_status = 0;
  int reached_target_count = 0;
  unsigned int min_next_frontier_dist_bits = kInfinityBits;
  unsigned int max_target_dist_bits = 0;
  int iterations = 0;
  int finished = 0;
  int converged = 0;
  int early_stopped = 0;
  int hit_iteration_limit = 0;
  unsigned int touched_count = 0;
};

// The host controller only needs these four values to decide whether and how
// to launch the next round.  Keep them as the prefix of DeviceBatchControl so
// one small D2H copy can read the round status without transferring per-query
// controls or the device-only counters.
struct DeviceHostRoundStatus {
  unsigned int frontier_count = 0;
  unsigned int active_query_count = 0;
  unsigned int current_frontier_index = 0;
  int error_status = 0;
};

struct DeviceBatchControl {
  unsigned int frontier_count = 0;
  unsigned int active_query_count = 0;
  unsigned int current_frontier_index = 0;
  int error_status = 0;
  unsigned int next_frontier_count = 0;
  unsigned int global_rounds = 0;
  unsigned int prior_touched_count = 0;
};

static_assert(sizeof(DeviceHostRoundStatus) == 4 * sizeof(unsigned int));
static_assert(offsetof(DeviceBatchControl, frontier_count) ==
              offsetof(DeviceHostRoundStatus, frontier_count));
static_assert(offsetof(DeviceBatchControl, active_query_count) ==
              offsetof(DeviceHostRoundStatus, active_query_count));
static_assert(offsetof(DeviceBatchControl, current_frontier_index) ==
              offsetof(DeviceHostRoundStatus, current_frontier_index));
static_assert(offsetof(DeviceBatchControl, error_status) ==
              offsetof(DeviceHostRoundStatus, error_status));
static_assert(offsetof(DeviceBatchControl, next_frontier_count) ==
              sizeof(DeviceHostRoundStatus));

enum TargetPathStatus : int {
  kTargetUnreachable = 0,
  kTargetPathValid = 1,
  kTargetPathInvalid = 2,
};

struct DeviceTargetSummary {
  unsigned long long state = 0;
  unsigned long long node_count = 0;
  unsigned long long edge_count = 0;
  int root = -1;
  int status = kTargetUnreachable;
};

struct DeviceResultHeader {
  unsigned long long total_nodes = 0;
  unsigned long long total_edges = 0;
  int arena_overflow = 0;
  int invalid_path = 0;
};

struct DeviceTelemetry {
  unsigned long long edges_examined = 0;
  unsigned long long successful_relaxations = 0;
};

struct TransferHeaderLayout {
  std::size_t query_controls = 0;
  std::size_t batch_control = 0;
  std::size_t target_summaries = 0;
  std::size_t node_offsets = 0;
  std::size_t edge_offsets = 0;
  std::size_t result_header = 0;
  std::size_t total_bytes = 0;
};

std::size_t align_transfer_offset(std::size_t offset,
                                  std::size_t alignment) {
  const std::size_t remainder = offset % alignment;
  return remainder == 0
             ? offset
             : require_add(offset, alignment - remainder,
                           "BF12 transfer-header alignment overflow");
}

template <typename T>
std::size_t append_transfer_array(std::size_t* offset,
                                  std::size_t count) {
  *offset = align_transfer_offset(*offset, alignof(T));
  const std::size_t begin = *offset;
  *offset = require_add(
      *offset,
      require_multiply(count, sizeof(T),
                       "BF12 transfer-header byte multiplication overflow"),
      "BF12 transfer-header byte addition overflow");
  return begin;
}

TransferHeaderLayout transfer_header_layout(std::size_t query_count,
                                            std::size_t target_count) {
  TransferHeaderLayout layout;
  std::size_t offset = 0;
  layout.query_controls =
      append_transfer_array<DeviceQueryControl>(&offset, query_count);
  layout.batch_control =
      append_transfer_array<DeviceBatchControl>(&offset, 1);
  layout.target_summaries =
      append_transfer_array<DeviceTargetSummary>(&offset, target_count);
  layout.node_offsets =
      append_transfer_array<unsigned long long>(&offset, target_count + 1);
  layout.edge_offsets =
      append_transfer_array<unsigned long long>(&offset, target_count + 1);
  layout.result_header =
      append_transfer_array<DeviceResultHeader>(&offset, 1);
  layout.total_bytes = align_transfer_offset(offset, alignof(unsigned long long));
  return layout;
}

static_assert(sizeof(DeviceTargetSummary) == 32,
              "BF12 target summary layout changed");

struct HostSidecarView {
  const std::vector<std::int32_t>* route_end_x = nullptr;
  const std::vector<std::int32_t>* route_end_y = nullptr;
  const std::vector<float>* base_vertex_cost = nullptr;
  bool already_validated = false;
};

struct NormalizedBatch {
  std::vector<StateIndex> source_states;
  std::vector<Index> targets;
  std::vector<StateIndex> target_states;
  std::vector<DeviceQueryDescriptor> descriptors;
};

struct DeviceWorkspace {
  Offset rows = 0;
  hipStream_t stream = nullptr;
  std::size_t query_capacity = 0;
  std::size_t state_capacity = 0;
  unsigned long long* best_state = nullptr;
  StateIndex* frontier = nullptr;
  StateIndex* next_frontier = nullptr;
  unsigned int* next_marks = nullptr;
  unsigned char* source_mask = nullptr;
  StateIndex* touched_states = nullptr;
  unsigned int* touched_count = nullptr;
  float* dynamic_vertex_cost = nullptr;
  DeviceQueryDescriptor* descriptors = nullptr;
  DeviceQueryControl* query_controls = nullptr;
  DeviceQueryControl* host_query_controls = nullptr;
  DeviceBatchControl* batch_control = nullptr;
  DeviceBatchControl* host_batch_control = nullptr;
  DeviceHostRoundStatus* host_round_status = nullptr;
  StateIndex* source_states = nullptr;
  std::size_t source_capacity = 0;
  StateIndex* target_states = nullptr;
  std::size_t target_capacity = 0;
  DeviceTargetSummary* target_summaries = nullptr;
  DeviceTargetSummary* host_target_summaries = nullptr;
  unsigned long long* node_offsets = nullptr;
  unsigned long long* edge_offsets = nullptr;
  unsigned long long* host_node_offsets = nullptr;
  unsigned long long* host_edge_offsets = nullptr;
  std::size_t reconstruction_target_capacity = 0;
  DeviceResultHeader* result_header = nullptr;
  DeviceResultHeader* host_result_header = nullptr;
  unsigned char* transfer_header = nullptr;
  unsigned char* host_transfer_header = nullptr;
  std::size_t transfer_header_capacity = 0;
  unsigned char* result_payload = nullptr;
  unsigned char* host_result_payload = nullptr;
  std::size_t result_payload_bytes = 0;
  Index* result_nodes = nullptr;
  Index* host_result_nodes = nullptr;
  DeviceOffset* result_edges = nullptr;
  DeviceOffset* host_result_edges = nullptr;
  float* result_edge_costs = nullptr;
  float* host_result_edge_costs = nullptr;
  std::size_t result_node_capacity = 0;
  std::size_t result_edge_capacity = 0;
  DeviceTelemetry* telemetry = nullptr;
  DeviceTelemetry* host_telemetry = nullptr;
  unsigned int* telemetry_active_queries = nullptr;
  unsigned int* host_telemetry_active_queries = nullptr;
  unsigned long long* telemetry_frontier_sizes = nullptr;
  unsigned long long* host_telemetry_frontier_sizes = nullptr;
  std::size_t telemetry_round_capacity = 0;
  bool needs_dense_reset = true;
  int cooperative_blocks = -1;
  bool device_capability_initialized = false;
  BellmanFord12CooperativeCapability cached_capability{};
};

__host__ __device__ inline unsigned long long pack_state(
    unsigned int distance_bits,
    unsigned int predecessor) {
  return (static_cast<unsigned long long>(distance_bits) << 32) |
         static_cast<unsigned long long>(predecessor);
}

__host__ __device__ inline unsigned int state_distance_bits(
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

__host__ __device__ inline StateIndex make_state_index(std::uint32_t query,
                                                        std::uint32_t vertex,
                                                        std::uint32_t rows) {
  return query * rows + vertex;
}

__host__ __device__ inline std::uint32_t state_query(StateIndex state,
                                                      std::uint32_t rows) {
  return state / rows;
}

__host__ __device__ inline std::uint32_t state_vertex(StateIndex state,
                                                       std::uint32_t rows) {
  return state % rows;
}

dim3 grid_for_items(std::size_t items) {
  if (items == 0) return dim3(1);
  const std::size_t blocks = (items + kBlockSize - 1) / kBlockSize;
  if (blocks <= kMaximumGridX) return dim3(static_cast<unsigned int>(blocks));
  const std::size_t y = (blocks + kMaximumGridX - 1) / kMaximumGridX;
  if (y > kMaximumGridX) throw std::length_error("BF12 launch grid is too large");
  return dim3(kMaximumGridX, static_cast<unsigned int>(y));
}

dim3 sparse_reset_grid(std::size_t state_capacity) {
  constexpr std::size_t kMaximumSparseResetBlocks = 256;
  const std::size_t required = std::max<std::size_t>(
      1, (state_capacity + kBlockSize - 1) / kBlockSize);
  return dim3(static_cast<unsigned int>(
      std::min(required, kMaximumSparseResetBlocks)));
}

__device__ __forceinline__ std::size_t logical_thread_id() {
  return (static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) *
             blockDim.x +
         threadIdx.x;
}

__device__ __forceinline__ std::size_t logical_thread_count() {
  return static_cast<std::size_t>(gridDim.x) * gridDim.y * blockDim.x;
}

__global__ void pack_transfer_header_kernel(
    const DeviceQueryControl* query_controls,
    std::size_t query_count,
    const DeviceBatchControl* batch_control,
    const DeviceTargetSummary* target_summaries,
    std::size_t target_count,
    const unsigned long long* node_offsets,
    const unsigned long long* edge_offsets,
    const DeviceResultHeader* result_header,
    unsigned char* transfer_header,
    TransferHeaderLayout layout) {
  DeviceQueryControl* packed_query_controls =
      reinterpret_cast<DeviceQueryControl*>(
          transfer_header + layout.query_controls);
  DeviceTargetSummary* packed_target_summaries =
      reinterpret_cast<DeviceTargetSummary*>(
          transfer_header + layout.target_summaries);
  unsigned long long* packed_node_offsets =
      reinterpret_cast<unsigned long long*>(
          transfer_header + layout.node_offsets);
  unsigned long long* packed_edge_offsets =
      reinterpret_cast<unsigned long long*>(
          transfer_header + layout.edge_offsets);
  for (std::size_t query = logical_thread_id(); query < query_count;
       query += logical_thread_count()) {
    packed_query_controls[query] = query_controls[query];
  }
  for (std::size_t target = logical_thread_id(); target < target_count;
       target += logical_thread_count()) {
    packed_target_summaries[target] = target_summaries[target];
  }
  for (std::size_t item = logical_thread_id(); item <= target_count;
       item += logical_thread_count()) {
    packed_node_offsets[item] = node_offsets[item];
    packed_edge_offsets[item] = edge_offsets[item];
  }
  if (logical_thread_id() == 0) {
    *reinterpret_cast<DeviceBatchControl*>(
        transfer_header + layout.batch_control) = *batch_control;
    *reinterpret_cast<DeviceResultHeader*>(
        transfer_header + layout.result_header) = *result_header;
  }
}

__device__ __forceinline__ bool finite_device(float value) {
  return isfinite(value);
}

__device__ __forceinline__ bool node_admitted(const DeviceGraph& graph,
                                               Index node,
                                               const DeviceBounds& bounds) {
  if (bounds.enabled == 0) return true;
  const std::int32_t x = graph.route_end_x[node];
  const std::int32_t y = graph.route_end_y[node];
  if (x == kMissingCoordinate && y == kMissingCoordinate) return true;
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

__device__ __forceinline__ float effective_edge_weight(float edge_value,
                                                        float base_cost,
                                                        float dynamic_cost) {
  return edge_value == 0.0f || dynamic_cost == 0.0f
             ? 0.0f
             : (edge_value * base_cost) * dynamic_cost;
}

__device__ __forceinline__ unsigned long long coherent_atomic_load(
    unsigned long long* address) {
#if defined(__has_builtin)
#if !defined(BF12_FORCE_CAS_ATOMIC_LOAD) && __has_builtin(__hip_atomic_load)
  return __hip_atomic_load(address, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
#else
  return atomicCAS(address, 0ULL, 0ULL);
#endif
#else
  return atomicCAS(address, 0ULL, 0ULL);
#endif
}

struct AtomicRelaxResult {
  bool improved = false;
  bool first_discovery = false;
};

__device__ __forceinline__ AtomicRelaxResult atomic_relax_strict(
    unsigned long long* address,
    float candidate,
    DeviceOffset predecessor) {
  const unsigned int candidate_bits = __float_as_uint(candidate);
  unsigned long long observed = coherent_atomic_load(address);
  while (candidate_bits < state_distance_bits(observed)) {
    const unsigned long long desired =
        pack_state(candidate_bits, predecessor);
    const unsigned long long assumed = observed;
    observed = atomicCAS(address, assumed, desired);
    if (observed == assumed) {
      return {true, state_distance_bits(assumed) == kInfinityBits};
    }
  }
  return {};
}

__device__ __forceinline__ Index source_for_edge(
    const DeviceOffset* rowptr,
    Offset rows,
    DeviceOffset edge) {
  Offset low = 0;
  Offset high = rows;
  while (low < high) {
    const Offset middle = low + (high - low) / 2;
    if (rowptr[middle + 1] <= edge) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low >= rows || rowptr[low] > edge || rowptr[low + 1] <= edge) return -1;
  return static_cast<Index>(low);
}

__global__ void clear_dense_state_kernel(std::size_t state_count,
                                         unsigned long long* best_state,
                                         unsigned int* next_marks,
                                         unsigned char* source_mask) {
  for (std::size_t state = logical_thread_id(); state < state_count;
       state += logical_thread_count()) {
    best_state[state] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[state] = 0;
    source_mask[state] = 0;
  }
}

__global__ void clear_sparse_state_kernel(
    std::size_t state_capacity,
    const StateIndex* touched_states,
    const unsigned int* touched_count,
    unsigned long long* best_state,
    unsigned int* next_marks,
    unsigned char* source_mask) {
  __shared__ unsigned int count;
  if (threadIdx.x == 0) count = *touched_count;
  __syncthreads();
  const std::size_t observed = static_cast<std::size_t>(count);
  const std::size_t safe_count =
      observed < state_capacity ? observed : state_capacity;
  for (std::size_t item = logical_thread_id(); item < safe_count;
       item += logical_thread_count()) {
    const StateIndex state = touched_states[item];
    if (static_cast<std::size_t>(state) >= state_capacity) continue;
    best_state[state] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[state] = 0;
    source_mask[state] = 0;
  }
}

__global__ void fill_float_kernel(std::size_t count,
                                  float* values,
                                  float value) {
  for (std::size_t item = logical_thread_id(); item < count;
       item += logical_thread_count()) {
    values[item] = value;
  }
}

__global__ void initialize_batch_kernel(
    std::size_t query_count,
    unsigned int source_count,
    unsigned int active_query_count,
    const DeviceQueryDescriptor* descriptors,
    DeviceQueryControl* controls,
    DeviceBatchControl* batch,
    unsigned int* touched_count,
    DeviceTelemetry* telemetry) {
  const std::size_t item = logical_thread_id();
  if (item < query_count) {
    DeviceQueryControl value{};
    value.finished = descriptors[item].max_iterations == 0 ? 1 : 0;
    value.hit_iteration_limit = value.finished;
    value.touched_count = descriptors[item].source_count;
    controls[item] = value;
  }
  if (item == 0) {
    batch->frontier_count = source_count;
    batch->next_frontier_count = 0;
    batch->active_query_count = active_query_count;
    batch->global_rounds = 0;
    batch->current_frontier_index = 0;
    batch->error_status = 0;
    batch->prior_touched_count = 0;
    *touched_count = source_count;
    if (telemetry) *telemetry = {};
  }
}

__global__ void seed_sources_kernel(
    const StateIndex* source_states,
    unsigned int source_count,
    unsigned long long* best_state,
    StateIndex* frontier,
    unsigned char* source_mask,
    StateIndex* touched_states) {
  for (std::size_t item = logical_thread_id(); item < source_count;
       item += logical_thread_count()) {
    const StateIndex state = source_states[item];
    best_state[state] = pack_state(0u, kNoPredecessor);
    frontier[item] = state;
    source_mask[state] = 1;
    touched_states[item] = state;
  }
}

__global__ void reset_round_kernel(std::size_t query_count,
                                   DeviceQueryControl* controls,
                                   DeviceBatchControl* batch) {
  const std::size_t query = logical_thread_id();
  if (query < query_count && controls[query].finished == 0) {
    controls[query].next_count = 0;
    controls[query].error_status = 0;
    controls[query].reached_target_count = 0;
    controls[query].min_next_frontier_dist_bits = kInfinityBits;
    controls[query].max_target_dist_bits = 0;
  }
  if (query == 0) batch->next_frontier_count = 0;
}

__device__ __forceinline__ void relax_frontier_item(
    const DeviceGraph& graph,
    StateIndex source_state,
    unsigned int query_count,
    unsigned int mark_token,
    const DeviceQueryDescriptor* descriptors,
    const float* dynamic_vertex_cost,
    unsigned long long* best_state,
    StateIndex* next_frontier,
    unsigned int* next_marks,
    const unsigned char* source_mask,
    StateIndex* touched_states,
    unsigned int* touched_count,
    DeviceQueryControl* controls,
    DeviceBatchControl* batch,
    DeviceTelemetry* telemetry,
    bool collect_telemetry) {
  const std::uint32_t rows = static_cast<std::uint32_t>(graph.rows);
  const std::uint32_t query = state_query(source_state, rows);
  const std::uint32_t from = state_vertex(source_state, rows);
  if (query >= query_count || from >= rows) {
    atomicExch(&batch->error_status, 1);
    return;
  }
  DeviceQueryControl* control = &controls[query];
  if (control->finished != 0) return;
  const float from_distance = state_distance(best_state[source_state]);
  if (!finite_device(from_distance)) return;
  const DeviceOffset begin = graph.rowptr[from];
  const DeviceOffset end = graph.rowptr[from + 1];
  if (end < begin || static_cast<Offset>(end) > graph.nnz) {
    atomicExch(&control->error_status, 2);
    atomicExch(&batch->error_status, 2);
    return;
  }
  if (collect_telemetry) {
    atomicAdd(&telemetry->edges_examined,
              static_cast<unsigned long long>(end - begin));
  }
  unsigned int local_min = kInfinityBits;
  const DeviceBounds bounds = descriptors[query].bounds;
  for (DeviceOffset edge = begin; edge < end; ++edge) {
    const Index destination = graph.destinations[edge];
    if (destination < 0 || static_cast<Offset>(destination) >= graph.rows) {
      atomicExch(&control->error_status, 1);
      atomicExch(&batch->error_status, 1);
      continue;
    }
    const StateIndex destination_state =
        make_state_index(query, static_cast<std::uint32_t>(destination), rows);
    if (source_mask[destination_state] != 0 ||
        !node_admitted(graph, destination, bounds)) {
      continue;
    }
    const float weight = effective_edge_weight(
        graph.edge_values[edge], graph.base_vertex_cost[destination],
        dynamic_vertex_cost[destination]);
    if (!finite_device(weight) || weight < 0.0f) {
      atomicExch(&control->error_status, 3);
      atomicExch(&batch->error_status, 3);
      continue;
    }
    const float candidate = from_distance + weight;
    if (!finite_device(candidate)) continue;
    const AtomicRelaxResult relaxation = atomic_relax_strict(
        &best_state[destination_state], candidate, edge);
    if (!relaxation.improved) continue;
    if (collect_telemetry) {
      atomicAdd(&telemetry->successful_relaxations, 1ULL);
    }
    if (relaxation.first_discovery) {
      const unsigned int touched_slot = atomicAdd(touched_count, 1u);
      if (static_cast<std::size_t>(touched_slot) >=
          static_cast<std::size_t>(query_count) * rows) {
        atomicExch(&control->error_status, 5);
        atomicExch(&batch->error_status, 5);
      } else {
        touched_states[touched_slot] = destination_state;
        atomicAdd(&control->touched_count, 1u);
      }
    }
    const unsigned int old_mark =
        atomicExch(&next_marks[destination_state], mark_token);
    if (old_mark != mark_token) {
      const unsigned int slot = atomicAdd(&batch->next_frontier_count, 1u);
      if (static_cast<std::size_t>(slot) >=
          static_cast<std::size_t>(query_count) * rows) {
        atomicExch(&control->error_status, 4);
        atomicExch(&batch->error_status, 4);
      } else {
        next_frontier[slot] = destination_state;
        atomicAdd(&control->next_count, 1);
      }
    }
    const unsigned int bits = __float_as_uint(candidate);
    local_min = bits < local_min ? bits : local_min;
  }
  if (local_min != kInfinityBits) {
    atomicMin(&control->min_next_frontier_dist_bits, local_min);
  }
}

__global__ void relax_frontier_kernel(
    DeviceGraph graph,
    const StateIndex* frontier,
    unsigned int frontier_count,
    unsigned int query_count,
    unsigned int mark_token,
    const DeviceQueryDescriptor* descriptors,
    const float* dynamic_vertex_cost,
    unsigned long long* best_state,
    StateIndex* next_frontier,
    unsigned int* next_marks,
    const unsigned char* source_mask,
    StateIndex* touched_states,
    unsigned int* touched_count,
    DeviceQueryControl* controls,
    DeviceBatchControl* batch,
    DeviceTelemetry* telemetry,
    int collect_telemetry) {
  for (std::size_t item = logical_thread_id(); item < frontier_count;
       item += logical_thread_count()) {
    relax_frontier_item(graph, frontier[item], query_count, mark_token,
                        descriptors, dynamic_vertex_cost, best_state,
                        next_frontier, next_marks, source_mask, touched_states,
                        touched_count, controls, batch, telemetry,
                        collect_telemetry != 0);
  }
}

__global__ void update_target_status_kernel(
    const StateIndex* target_states,
    std::size_t target_count,
    const DeviceQueryDescriptor* descriptors,
    const unsigned long long* best_state,
    DeviceQueryControl* controls,
    std::uint32_t rows) {
  for (std::size_t item = logical_thread_id(); item < target_count;
       item += logical_thread_count()) {
    const StateIndex state = target_states[item];
    const std::uint32_t query = state_query(state, rows);
    DeviceQueryControl* control = &controls[query];
    if (control->finished != 0) continue;
    const int completed_iteration = control->iterations + 1;
    if (completed_iteration % descriptors[query].target_check_interval != 0) {
      continue;
    }
    const unsigned int bits = state_distance_bits(best_state[state]);
    if (bits == kInfinityBits) continue;
    atomicAdd(&control->reached_target_count, 1);
    atomicMax(&control->max_target_dist_bits, bits);
  }
}

__global__ void finish_round_kernel(
    std::size_t query_count,
    const DeviceQueryDescriptor* descriptors,
    DeviceQueryControl* controls,
    DeviceBatchControl* batch,
    unsigned int* telemetry_active_queries,
    unsigned long long* telemetry_frontier_sizes,
    std::size_t telemetry_round_capacity) {
  const std::size_t query = logical_thread_id();
  if (query < query_count) {
    DeviceQueryControl* control = &controls[query];
    if (control->finished == 0) {
      const int completed = control->iterations + 1;
      control->iterations = completed;
      const bool check_targets =
          completed % descriptors[query].target_check_interval == 0;
      bool finish = false;
      if (control->error_status != 0 || batch->error_status != 0) {
        finish = true;
      } else if (control->next_count == 0) {
        control->converged = 1;
        finish = true;
      } else if (check_targets &&
                 control->reached_target_count ==
                     static_cast<int>(descriptors[query].target_count) &&
                 control->min_next_frontier_dist_bits >=
                     control->max_target_dist_bits) {
        control->early_stopped = 1;
        finish = true;
      } else if (completed >= descriptors[query].max_iterations) {
        control->hit_iteration_limit = 1;
        finish = true;
      }
      if (finish) {
        control->finished = 1;
        atomicSub(&batch->active_query_count, 1u);
      }
    }
  }
  if (query == 0) {
    const unsigned int round = batch->global_rounds;
    if (round < telemetry_round_capacity) {
      telemetry_active_queries[round] = batch->active_query_count;
      telemetry_frontier_sizes[round] = batch->frontier_count;
    }
    batch->frontier_count = batch->next_frontier_count;
    batch->current_frontier_index ^= 1u;
    ++batch->global_rounds;
  }
}

__global__ void cooperative_batch_controller_kernel(
    DeviceGraph graph,
    const StateIndex* source_states,
    unsigned int source_count,
    const StateIndex* target_states,
    unsigned int target_count,
    unsigned int query_count,
    unsigned int initial_active_query_count,
    const DeviceQueryDescriptor* descriptors,
    const float* dynamic_vertex_cost,
    unsigned long long* best_state,
    StateIndex* frontier,
    StateIndex* next_frontier,
    unsigned int* next_marks,
    unsigned char* source_mask,
    StateIndex* touched_states,
    unsigned int* touched_count,
    std::size_t state_capacity,
    DeviceQueryControl* controls,
    DeviceBatchControl* batch,
    DeviceTelemetry* telemetry,
    unsigned int* telemetry_active_queries,
    unsigned long long* telemetry_frontier_sizes,
    std::size_t telemetry_round_capacity,
    int collect_telemetry) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const std::size_t thread = grid.thread_rank();
  const std::size_t thread_count = grid.size();

  if (thread == 0) {
    const unsigned int observed = *touched_count;
    const std::size_t observed_size = static_cast<std::size_t>(observed);
    batch->prior_touched_count = static_cast<unsigned int>(
        observed_size < state_capacity ? observed_size : state_capacity);
  }
  grid.sync();
  for (std::size_t item = thread; item < batch->prior_touched_count;
       item += thread_count) {
    const StateIndex state = touched_states[item];
    if (static_cast<std::size_t>(state) >= state_capacity) continue;
    best_state[state] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[state] = 0;
    source_mask[state] = 0;
  }
  for (std::size_t query = thread; query < query_count;
       query += thread_count) {
    DeviceQueryControl value{};
    value.finished = descriptors[query].max_iterations == 0 ? 1 : 0;
    value.hit_iteration_limit = value.finished;
    value.touched_count = descriptors[query].source_count;
    controls[query] = value;
  }
  grid.sync();
  for (std::size_t item = thread; item < source_count; item += thread_count) {
    const StateIndex state = source_states[item];
    best_state[state] = pack_state(0u, kNoPredecessor);
    source_mask[state] = 1;
    frontier[item] = state;
    touched_states[item] = state;
  }
  if (thread == 0) {
    *touched_count = source_count;
    batch->frontier_count = source_count;
    batch->next_frontier_count = 0;
    batch->active_query_count = initial_active_query_count;
    batch->global_rounds = 0;
    batch->current_frontier_index = 0;
    batch->error_status = 0;
    if (telemetry) *telemetry = {};
  }
  grid.sync();

  while (batch->active_query_count != 0) {
    for (std::size_t query = thread; query < query_count;
         query += thread_count) {
      DeviceQueryControl* control = &controls[query];
      if (control->finished == 0) {
        control->next_count = 0;
        control->error_status = 0;
        control->reached_target_count = 0;
        control->min_next_frontier_dist_bits = kInfinityBits;
        control->max_target_dist_bits = 0;
      }
    }
    if (thread == 0) batch->next_frontier_count = 0;
    grid.sync();

    const unsigned int current_count = batch->frontier_count;
    const unsigned int current_index = batch->current_frontier_index;
    const unsigned int mark_token = batch->global_rounds + 1u;
    const StateIndex* current = current_index == 0 ? frontier : next_frontier;
    StateIndex* next = current_index == 0 ? next_frontier : frontier;
    for (std::size_t item = thread; item < current_count;
         item += thread_count) {
      relax_frontier_item(
          graph, current[item], query_count, mark_token, descriptors,
          dynamic_vertex_cost, best_state, next, next_marks, source_mask,
          touched_states, touched_count, controls, batch, telemetry,
          collect_telemetry != 0);
    }
    grid.sync();

    const std::uint32_t rows = static_cast<std::uint32_t>(graph.rows);
    for (std::size_t item = thread; item < target_count;
         item += thread_count) {
      const StateIndex state = target_states[item];
      const std::uint32_t query = state_query(state, rows);
      DeviceQueryControl* control = &controls[query];
      if (control->finished != 0) continue;
      const int completed = control->iterations + 1;
      if (completed % descriptors[query].target_check_interval != 0) continue;
      const unsigned int bits = state_distance_bits(best_state[state]);
      if (bits == kInfinityBits) continue;
      atomicAdd(&control->reached_target_count, 1);
      atomicMax(&control->max_target_dist_bits, bits);
    }
    grid.sync();

    for (std::size_t query = thread; query < query_count;
         query += thread_count) {
      DeviceQueryControl* control = &controls[query];
      if (control->finished != 0) continue;
      const int completed = control->iterations + 1;
      control->iterations = completed;
      const bool check_targets =
          completed % descriptors[query].target_check_interval == 0;
      bool finish = false;
      if (control->error_status != 0 || batch->error_status != 0) {
        finish = true;
      } else if (control->next_count == 0) {
        control->converged = 1;
        finish = true;
      } else if (check_targets &&
                 control->reached_target_count ==
                     static_cast<int>(descriptors[query].target_count) &&
                 control->min_next_frontier_dist_bits >=
                     control->max_target_dist_bits) {
        control->early_stopped = 1;
        finish = true;
      } else if (completed >= descriptors[query].max_iterations) {
        control->hit_iteration_limit = 1;
        finish = true;
      }
      if (finish) {
        control->finished = 1;
        atomicSub(&batch->active_query_count, 1u);
      }
    }
    grid.sync();
    if (thread == 0) {
      const unsigned int round = batch->global_rounds;
      if (round < telemetry_round_capacity) {
        telemetry_active_queries[round] = batch->active_query_count;
        telemetry_frontier_sizes[round] = batch->frontier_count;
      }
      batch->frontier_count = batch->next_frontier_count;
      batch->current_frontier_index ^= 1u;
      ++batch->global_rounds;
    }
    grid.sync();
  }
}

__global__ void summarize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* destinations,
    const DeviceOffset* rowptr,
    const StateIndex* target_states,
    std::size_t target_count,
    DeviceTargetSummary* summaries) {
  const std::uint32_t compact_rows = static_cast<std::uint32_t>(rows);
  for (std::size_t item = logical_thread_id(); item < target_count;
       item += logical_thread_count()) {
    DeviceTargetSummary summary{};
    const StateIndex target_state = target_states[item];
    const std::uint32_t query = state_query(target_state, compact_rows);
    const Index target = static_cast<Index>(state_vertex(target_state,
                                                          compact_rows));
    summary.state = best_state[target_state];
    if (state_distance_bits(summary.state) == kInfinityBits) {
      summaries[item] = summary;
      continue;
    }
    Index current = target;
    unsigned long long edge_count = 0;
    bool stored = false;
    for (Offset guard = 0; guard <= rows; ++guard) {
      const StateIndex current_state = make_state_index(
          query, static_cast<std::uint32_t>(current), compact_rows);
      const unsigned long long state = best_state[current_state];
      const DeviceOffset edge = static_cast<DeviceOffset>(state);
      if (edge == kNoPredecessor) {
        if (source_mask[current_state] != 0 &&
            state_distance_bits(state) == 0u) {
          summary.node_count = edge_count + 1;
          summary.edge_count = edge_count;
          summary.root = current;
          summary.status = kTargetPathValid;
        } else {
          summary.status = kTargetPathInvalid;
        }
        summaries[item] = summary;
        stored = true;
        break;
      }
      if (static_cast<Offset>(edge) >= nnz || destinations[edge] != current) {
        summary.status = kTargetPathInvalid;
        summaries[item] = summary;
        stored = true;
        break;
      }
      const Index predecessor = source_for_edge(rowptr, rows, edge);
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) {
        summary.status = kTargetPathInvalid;
        summaries[item] = summary;
        stored = true;
        break;
      }
      current = predecessor;
      ++edge_count;
    }
    if (!stored) {
      summary.status = kTargetPathInvalid;
      summaries[item] = summary;
    }
  }
}

__global__ void scan_target_offsets_kernel(
    const DeviceTargetSummary* summaries,
    std::size_t target_count,
    unsigned long long node_capacity,
    unsigned long long edge_capacity,
    unsigned long long* node_offsets,
    unsigned long long* edge_offsets,
    DeviceResultHeader* header) {
  if (logical_thread_id() != 0) return;
  DeviceResultHeader result{};
  for (std::size_t item = 0; item < target_count; ++item) {
    node_offsets[item] = result.total_nodes;
    edge_offsets[item] = result.total_edges;
    const DeviceTargetSummary summary = summaries[item];
    if (summary.status == kTargetPathInvalid) {
      result.invalid_path = 1;
      continue;
    }
    if (summary.status != kTargetPathValid) continue;
    if (summary.node_count == 0 ||
        summary.node_count != summary.edge_count + 1 ||
        summary.node_count > ~result.total_nodes ||
        summary.edge_count > ~result.total_edges) {
      result.invalid_path = 1;
      continue;
    }
    result.total_nodes += summary.node_count;
    result.total_edges += summary.edge_count;
  }
  node_offsets[target_count] = result.total_nodes;
  edge_offsets[target_count] = result.total_edges;
  if (result.total_nodes > node_capacity || result.total_edges > edge_capacity) {
    result.arena_overflow = 1;
  }
  *header = result;
}

__global__ void materialize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* destinations,
    const DeviceOffset* rowptr,
    const StateIndex* target_states,
    const DeviceTargetSummary* summaries,
    const unsigned long long* node_offsets,
    const unsigned long long* edge_offsets,
    std::size_t target_count,
    const DeviceResultHeader* header,
    Index* result_nodes,
    DeviceOffset* result_edges,
    const float* edge_values,
    const float* base_vertex_cost,
    const float* dynamic_vertex_cost,
    float* result_edge_costs) {
  if (header->arena_overflow != 0 || header->invalid_path != 0) return;
  const std::uint32_t compact_rows = static_cast<std::uint32_t>(rows);
  for (std::size_t item = logical_thread_id(); item < target_count;
       item += logical_thread_count()) {
    const DeviceTargetSummary summary = summaries[item];
    if (summary.status != kTargetPathValid) continue;
    const StateIndex target_state = target_states[item];
    const std::uint32_t query = state_query(target_state, compact_rows);
    Index current = static_cast<Index>(state_vertex(target_state, compact_rows));
    const unsigned long long node_base = node_offsets[item];
    const unsigned long long edge_base = edge_offsets[item];
    result_nodes[node_base + summary.edge_count] = current;
    for (unsigned long long remaining = summary.edge_count; remaining > 0;
         --remaining) {
      const StateIndex current_state = make_state_index(
          query, static_cast<std::uint32_t>(current), compact_rows);
      const DeviceOffset edge =
          static_cast<DeviceOffset>(best_state[current_state]);
      if (edge == kNoPredecessor || static_cast<Offset>(edge) >= nnz ||
          destinations[edge] != current) {
        return;
      }
      const Index predecessor = source_for_edge(rowptr, rows, edge);
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) return;
      result_edges[edge_base + remaining - 1] = edge;
      result_edge_costs[edge_base + remaining - 1] = effective_edge_weight(
          edge_values[edge], base_vertex_cost[current],
          dynamic_vertex_cost[current]);
      result_nodes[node_base + remaining - 1] = predecessor;
      current = predecessor;
    }
    const StateIndex root_state = make_state_index(
        query, static_cast<std::uint32_t>(current), compact_rows);
    if (current != summary.root || source_mask[root_state] == 0) return;
  }
}

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols ||
      graph.rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::invalid_argument(
        "BF12 requires a nonempty square CSR graph with int vertex IDs");
  }
  if (graph.nnz < 0 ||
      static_cast<unsigned long long>(graph.nnz) >= kNoPredecessor) {
    throw std::invalid_argument("BF12 CSR edge IDs must fit uint32");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("BF12 CSR arrays do not match rows and nnz");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::invalid_argument("BF12 CSR row offsets are not monotone");
    }
  }
  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const Index destination = graph.colind[static_cast<std::size_t>(edge)];
    const float value = graph.values[static_cast<std::size_t>(edge)];
    if (destination < 0 || static_cast<Offset>(destination) >= graph.rows ||
        !std::isfinite(value) || value < 0.0f) {
      throw std::invalid_argument(
          "BF12 CSR destinations and weights must be valid and nonnegative");
    }
  }
}

void validate_sidecars(const HostSidecarView& sidecars, Offset rows) {
  if (!sidecars.route_end_x || !sidecars.route_end_y ||
      !sidecars.base_vertex_cost ||
      sidecars.route_end_x->size() != static_cast<std::size_t>(rows) ||
      sidecars.route_end_y->size() != static_cast<std::size_t>(rows) ||
      sidecars.base_vertex_cost->size() != static_cast<std::size_t>(rows)) {
    throw std::invalid_argument("BF12 node sidecars must contain V entries");
  }
  for (Offset row = 0; row < rows; ++row) {
    const std::int32_t x = (*sidecars.route_end_x)[row];
    const std::int32_t y = (*sidecars.route_end_y)[row];
    const bool missing_x = x == kMissingCoordinate;
    const bool missing_y = y == kMissingCoordinate;
    const float base = (*sidecars.base_vertex_cost)[row];
    if (missing_x != missing_y || (!missing_x && (x < 0 || y < 0))) {
      throw std::invalid_argument("BF12 route-end coordinates are malformed");
    }
    if (!std::isfinite(base) || !(base > 0.0f)) {
      throw std::invalid_argument(
          "BF12 base vertex costs must be finite and positive");
    }
  }
}

HostSidecarView sidecar_view(
    const routing::interchange::RoutingCsrSidecars& sidecars,
    const HostCsrF32& graph) {
  // This is the public/untrusted RoutingCsrSidecars boundary. Validate the
  // complete carrier once, including a supplied spatial index, even though
  // BF12 uploads only the three node arrays below. The Impl observes the tag
  // and does not repeat the node scan.
  routing::interchange::validate_routing_csr_sidecars(
      sidecars, static_cast<std::size_t>(graph.rows),
      static_cast<std::size_t>(graph.nnz), false);
  return {&sidecars.route_end_x, &sidecars.route_end_y,
          &sidecars.base_vertex_cost, true};
}

HostSidecarView sidecar_view(const BellmanFord12NodeSidecars& sidecars) {
  return {&sidecars.route_end_x, &sidecars.route_end_y,
          &sidecars.base_vertex_costs, false};
}

DeviceGraphOwner copy_graph_to_device(const HostCsrF32& graph,
                                      const HostSidecarView& sidecars,
                                      hipStream_t stream) {
  DeviceGraphOwner owner;
  DrainStreamOnException drain{stream};
  try {
    std::vector<DeviceOffset> compact_rowptr;
    compact_rowptr.reserve(graph.rowptr.size());
    for (const Offset offset : graph.rowptr) {
      if (offset < 0 ||
          static_cast<unsigned long long>(offset) >= kNoPredecessor) {
        throw std::overflow_error("BF12 row offset cannot fit uint32");
      }
      compact_rowptr.push_back(static_cast<DeviceOffset>(offset));
    }
    owner.rowptr = device_allocate<DeviceOffset>(compact_rowptr.size(),
                                                  "allocate BF12 rowptr");
    check_hip(hipMemcpyAsync(owner.rowptr, compact_rowptr.data(),
                             compact_rowptr.size() * sizeof(DeviceOffset),
                             hipMemcpyHostToDevice, stream),
              "copy BF12 rowptr");
    const std::size_t edge_count = static_cast<std::size_t>(graph.nnz);
    owner.destinations =
        device_allocate<Index>(edge_count, "allocate BF12 destinations");
    owner.edge_values =
        device_allocate<float>(edge_count, "allocate BF12 edge values");
    if (edge_count != 0) {
      check_hip(hipMemcpyAsync(owner.destinations, graph.colind.data(),
                               edge_count * sizeof(Index),
                               hipMemcpyHostToDevice, stream),
                "copy BF12 destinations");
      check_hip(hipMemcpyAsync(owner.edge_values, graph.values.data(),
                               edge_count * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "copy BF12 edge values");
    }
    const std::size_t rows = static_cast<std::size_t>(graph.rows);
    owner.route_end_x =
        device_allocate<std::int32_t>(rows, "allocate BF12 route-end X");
    owner.route_end_y =
        device_allocate<std::int32_t>(rows, "allocate BF12 route-end Y");
    owner.base_vertex_cost =
        device_allocate<float>(rows, "allocate BF12 base costs");
    check_hip(hipMemcpyAsync(owner.route_end_x, sidecars.route_end_x->data(),
                             rows * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream),
              "copy BF12 route-end X");
    check_hip(hipMemcpyAsync(owner.route_end_y, sidecars.route_end_y->data(),
                             rows * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream),
              "copy BF12 route-end Y");
    check_hip(hipMemcpyAsync(owner.base_vertex_cost,
                             sidecars.base_vertex_cost->data(),
                             rows * sizeof(float), hipMemcpyHostToDevice,
                             stream),
              "copy BF12 base costs");
    check_hip(hipStreamSynchronize(stream), "synchronize BF12 graph upload");
    owner.view = {graph.rows,
                  graph.nnz,
                  owner.rowptr,
                  owner.destinations,
                  owner.edge_values,
                  owner.route_end_x,
                  owner.route_end_y,
                  owner.base_vertex_cost};
    return owner;
  } catch (...) {
    (void)hipStreamSynchronize(stream);
    device_free(owner.rowptr);
    device_free(owner.destinations);
    device_free(owner.edge_values);
    device_free(owner.route_end_x);
    device_free(owner.route_end_y);
    device_free(owner.base_vertex_cost);
    throw;
  }
}

void free_graph(DeviceGraphOwner& owner) noexcept {
  device_free(owner.rowptr);
  device_free(owner.destinations);
  device_free(owner.edge_values);
  device_free(owner.route_end_x);
  device_free(owner.route_end_y);
  device_free(owner.base_vertex_cost);
  owner.view = {};
}

DeviceWorkspace make_workspace(Offset rows, hipStream_t stream) {
  DeviceWorkspace workspace;
  workspace.rows = rows;
  workspace.stream = stream;
  try {
    const std::size_t count = static_cast<std::size_t>(rows);
    workspace.dynamic_vertex_cost =
        device_allocate<float>(count, "allocate BF12 dynamic costs");
    workspace.touched_count =
        device_allocate<unsigned int>(1, "allocate BF12 touched count");
    workspace.batch_control =
        device_allocate<DeviceBatchControl>(1, "allocate BF12 batch control");
    workspace.host_batch_control = pinned_allocate<DeviceBatchControl>(
        1, "allocate BF12 host batch control");
    workspace.host_round_status = pinned_allocate<DeviceHostRoundStatus>(
        1, "allocate BF12 host round status");
    workspace.result_header =
        device_allocate<DeviceResultHeader>(1, "allocate BF12 result header");
    workspace.host_result_header = pinned_allocate<DeviceResultHeader>(
        1, "allocate BF12 host result header");
    workspace.telemetry =
        device_allocate<DeviceTelemetry>(1, "allocate BF12 telemetry");
    workspace.host_telemetry = pinned_allocate<DeviceTelemetry>(
        1, "allocate BF12 host telemetry");
    check_hip(hipMemsetAsync(workspace.touched_count, 0,
                             sizeof(unsigned int), stream),
              "initialize BF12 touched count");
    hipLaunchKernelGGL(fill_float_kernel, grid_for_items(count),
                       dim3(kBlockSize), 0, stream, count,
                       workspace.dynamic_vertex_cost, 1.0f);
    check_hip(hipGetLastError(), "initialize BF12 dynamic costs");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF12 workspace initialization");
    return workspace;
  } catch (...) {
    (void)hipStreamSynchronize(stream);
    device_free(workspace.dynamic_vertex_cost);
    device_free(workspace.touched_count);
    device_free(workspace.batch_control);
    pinned_free(workspace.host_batch_control);
    pinned_free(workspace.host_round_status);
    device_free(workspace.result_header);
    pinned_free(workspace.host_result_header);
    device_free(workspace.telemetry);
    pinned_free(workspace.host_telemetry);
    throw;
  }
}

void free_workspace(DeviceWorkspace& workspace) noexcept {
  (void)hipStreamSynchronize(workspace.stream);
  device_free(workspace.best_state);
  device_free(workspace.frontier);
  device_free(workspace.next_frontier);
  device_free(workspace.next_marks);
  device_free(workspace.source_mask);
  device_free(workspace.touched_states);
  device_free(workspace.touched_count);
  device_free(workspace.dynamic_vertex_cost);
  device_free(workspace.descriptors);
  device_free(workspace.query_controls);
  pinned_free(workspace.host_query_controls);
  device_free(workspace.batch_control);
  pinned_free(workspace.host_batch_control);
  pinned_free(workspace.host_round_status);
  device_free(workspace.source_states);
  device_free(workspace.target_states);
  device_free(workspace.target_summaries);
  pinned_free(workspace.host_target_summaries);
  device_free(workspace.node_offsets);
  device_free(workspace.edge_offsets);
  pinned_free(workspace.host_node_offsets);
  pinned_free(workspace.host_edge_offsets);
  device_free(workspace.result_header);
  pinned_free(workspace.host_result_header);
  device_free(workspace.transfer_header);
  pinned_free(workspace.host_transfer_header);
  device_free(workspace.result_payload);
  pinned_free(workspace.host_result_payload);
  workspace.result_nodes = nullptr;
  workspace.host_result_nodes = nullptr;
  workspace.result_edges = nullptr;
  workspace.host_result_edges = nullptr;
  workspace.result_edge_costs = nullptr;
  workspace.host_result_edge_costs = nullptr;
  device_free(workspace.telemetry);
  pinned_free(workspace.host_telemetry);
  device_free(workspace.telemetry_active_queries);
  pinned_free(workspace.host_telemetry_active_queries);
  device_free(workspace.telemetry_frontier_sizes);
  pinned_free(workspace.host_telemetry_frontier_sizes);
  workspace = {};
}

bool ensure_query_capacity(DeviceWorkspace& workspace,
                           std::size_t required_queries) {
  if (required_queries <= workspace.query_capacity) return false;
  const std::size_t rows = static_cast<std::size_t>(workspace.rows);
  const std::size_t maximum_capacity =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() - 1) /
      rows;
  if (required_queries > maximum_capacity) {
    throw std::overflow_error("BF12 query capacity exceeds composite indexing");
  }
  // Query/vertex state dominates memory, so grow it exactly.  Endpoint and
  // compact-result buffers use geometric growth, but silently rounding B up
  // here could exceed the worker policy's 25*B*V safety calculation.
  const std::size_t new_capacity = required_queries;
  const std::size_t states = require_multiply(
      new_capacity, rows, "BF12 composite state capacity overflow");
  if (states >= std::numeric_limits<StateIndex>::max()) {
    throw std::overflow_error("BF12 composite state index reaches uint32 sentinel");
  }

  unsigned long long* best_state = nullptr;
  StateIndex* frontier = nullptr;
  StateIndex* next_frontier = nullptr;
  unsigned int* next_marks = nullptr;
  unsigned char* source_mask = nullptr;
  StateIndex* touched_states = nullptr;
  DeviceQueryDescriptor* descriptors = nullptr;
  DeviceQueryControl* controls = nullptr;
  DeviceQueryControl* host_controls = nullptr;
  workspace.needs_dense_reset = true;
  try {
    best_state = device_allocate<unsigned long long>(
        states, "allocate BF12 packed query states");
    frontier = device_allocate<StateIndex>(states, "allocate BF12 frontier");
    next_frontier =
        device_allocate<StateIndex>(states, "allocate BF12 next frontier");
    next_marks =
        device_allocate<unsigned int>(states, "allocate BF12 frontier marks");
    source_mask =
        device_allocate<unsigned char>(states, "allocate BF12 source masks");
    touched_states =
        device_allocate<StateIndex>(states, "allocate BF12 touched states");
    descriptors = device_allocate<DeviceQueryDescriptor>(
        new_capacity, "allocate BF12 descriptors");
    controls = device_allocate<DeviceQueryControl>(
        new_capacity, "allocate BF12 query controls");
    host_controls = pinned_allocate<DeviceQueryControl>(
        new_capacity, "allocate BF12 host query controls");
    hipLaunchKernelGGL(clear_dense_state_kernel, grid_for_items(states),
                       dim3(kBlockSize), 0, workspace.stream, states,
                       best_state, next_marks, source_mask);
    check_hip(hipGetLastError(), "initialize grown BF12 query state");
    check_hip(hipMemsetAsync(workspace.touched_count, 0,
                             sizeof(unsigned int), workspace.stream),
              "reset BF12 touched count after growth");
    check_hip(hipStreamSynchronize(workspace.stream),
              "synchronize grown BF12 query state and touched count");
  } catch (...) {
    (void)hipStreamSynchronize(workspace.stream);
    device_free(best_state);
    device_free(frontier);
    device_free(next_frontier);
    device_free(next_marks);
    device_free(source_mask);
    device_free(touched_states);
    device_free(descriptors);
    device_free(controls);
    pinned_free(host_controls);
    throw;
  }

  device_free(workspace.best_state);
  device_free(workspace.frontier);
  device_free(workspace.next_frontier);
  device_free(workspace.next_marks);
  device_free(workspace.source_mask);
  device_free(workspace.touched_states);
  device_free(workspace.descriptors);
  device_free(workspace.query_controls);
  pinned_free(workspace.host_query_controls);
  workspace.best_state = best_state;
  workspace.frontier = frontier;
  workspace.next_frontier = next_frontier;
  workspace.next_marks = next_marks;
  workspace.source_mask = source_mask;
  workspace.touched_states = touched_states;
  workspace.descriptors = descriptors;
  workspace.query_controls = controls;
  workspace.host_query_controls = host_controls;
  workspace.query_capacity = new_capacity;
  workspace.state_capacity = states;
  workspace.needs_dense_reset = false;
  return true;
}

void ensure_endpoint_capacity(DeviceWorkspace& workspace,
                              std::size_t required_sources,
                              std::size_t required_targets) {
  if (required_sources > workspace.source_capacity) {
    const std::size_t capacity = geometric_capacity(
        workspace.source_capacity, required_sources,
        std::numeric_limits<std::uint32_t>::max());
    StateIndex* states = nullptr;
    try {
      states =
          device_allocate<StateIndex>(capacity, "allocate BF12 source states");
    } catch (...) {
      device_free(states);
      throw;
    }
    device_free(workspace.source_states);
    workspace.source_states = states;
    workspace.source_capacity = capacity;
  }
  if (required_targets > workspace.target_capacity) {
    const std::size_t capacity = geometric_capacity(
        workspace.target_capacity, required_targets,
        std::numeric_limits<std::uint32_t>::max());
    StateIndex* states = nullptr;
    try {
      states =
          device_allocate<StateIndex>(capacity, "allocate BF12 target states");
    } catch (...) {
      device_free(states);
      throw;
    }
    device_free(workspace.target_states);
    workspace.target_states = states;
    workspace.target_capacity = capacity;
  }
}

void ensure_reconstruction_target_capacity(DeviceWorkspace& workspace,
                                           std::size_t required_targets) {
  if (required_targets <= workspace.reconstruction_target_capacity) return;
  const std::size_t capacity = geometric_capacity(
      workspace.reconstruction_target_capacity, required_targets,
      std::numeric_limits<std::uint32_t>::max());
  DeviceTargetSummary* summaries = nullptr;
  DeviceTargetSummary* host_summaries = nullptr;
  unsigned long long* node_offsets = nullptr;
  unsigned long long* edge_offsets = nullptr;
  unsigned long long* host_node_offsets = nullptr;
  unsigned long long* host_edge_offsets = nullptr;
  try {
    summaries = device_allocate<DeviceTargetSummary>(
        capacity, "allocate BF12 target summaries");
    host_summaries = pinned_allocate<DeviceTargetSummary>(
        capacity, "allocate BF12 host target summaries");
    node_offsets = device_allocate<unsigned long long>(
        capacity + 1, "allocate BF12 node offsets");
    edge_offsets = device_allocate<unsigned long long>(
        capacity + 1, "allocate BF12 edge offsets");
    host_node_offsets = pinned_allocate<unsigned long long>(
        capacity + 1, "allocate BF12 host node offsets");
    host_edge_offsets = pinned_allocate<unsigned long long>(
        capacity + 1, "allocate BF12 host edge offsets");
  } catch (...) {
    device_free(summaries);
    pinned_free(host_summaries);
    device_free(node_offsets);
    device_free(edge_offsets);
    pinned_free(host_node_offsets);
    pinned_free(host_edge_offsets);
    throw;
  }
  device_free(workspace.target_summaries);
  pinned_free(workspace.host_target_summaries);
  device_free(workspace.node_offsets);
  device_free(workspace.edge_offsets);
  pinned_free(workspace.host_node_offsets);
  pinned_free(workspace.host_edge_offsets);
  workspace.target_summaries = summaries;
  workspace.host_target_summaries = host_summaries;
  workspace.node_offsets = node_offsets;
  workspace.edge_offsets = edge_offsets;
  workspace.host_node_offsets = host_node_offsets;
  workspace.host_edge_offsets = host_edge_offsets;
  workspace.reconstruction_target_capacity = capacity;
}

std::size_t result_payload_size(std::size_t node_capacity,
                                std::size_t edge_capacity) {
  const std::size_t node_bytes = require_multiply(
      node_capacity, sizeof(Index), "BF12 result-node payload overflow");
  const std::size_t edge_index_bytes = require_multiply(
      edge_capacity, sizeof(DeviceOffset),
      "BF12 result-edge payload overflow");
  const std::size_t edge_cost_bytes = require_multiply(
      edge_capacity, sizeof(float),
      "BF12 result-edge-cost payload overflow");
  return require_add(
      node_bytes, require_add(edge_index_bytes, edge_cost_bytes,
                              "BF12 result payload overflow"),
      "BF12 result payload overflow");
}

void validate_result_growth_reserve(DeviceWorkspace& workspace,
                                    std::size_t required_nodes,
                                    std::size_t required_edges,
                                    std::size_t configured_reserve_bytes) {
  const std::size_t node_capacity = geometric_capacity(
      workspace.result_node_capacity, required_nodes,
      static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const std::size_t edge_capacity = geometric_capacity(
      workspace.result_edge_capacity, required_edges,
      static_cast<std::size_t>(std::numeric_limits<int>::max()));
  if (node_capacity == workspace.result_node_capacity &&
      edge_capacity == workspace.result_edge_capacity) {
    return;
  }
  const std::size_t allocation_bytes =
      result_payload_size(node_capacity, edge_capacity);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_hip(hipMemGetInfo(&free_bytes, &total_bytes),
            "refresh BF12 memory before result-arena growth");
  (void)total_bytes;
  const std::size_t reserve = configured_reserve_bytes != 0
                                  ? configured_reserve_bytes
                                  : free_bytes / 4;
  if (reserve >= free_bytes || allocation_bytes > free_bytes - reserve) {
    throw std::runtime_error(
        "BF12 result-arena growth would violate the GPU memory safety reserve");
  }
}

void ensure_result_capacity(DeviceWorkspace& workspace,
                            std::size_t required_nodes,
                            std::size_t required_edges) {
  const std::size_t node_capacity = geometric_capacity(
      workspace.result_node_capacity, required_nodes,
      static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const std::size_t edge_capacity = geometric_capacity(
      workspace.result_edge_capacity, required_edges,
      static_cast<std::size_t>(std::numeric_limits<int>::max()));
  if (node_capacity == workspace.result_node_capacity &&
      edge_capacity == workspace.result_edge_capacity) {
    return;
  }

  const std::size_t node_bytes = require_multiply(
      node_capacity, sizeof(Index), "BF12 result-node payload overflow");
  const std::size_t edge_index_bytes = require_multiply(
      edge_capacity, sizeof(DeviceOffset),
      "BF12 result-edge payload overflow");
  const std::size_t payload_bytes =
      result_payload_size(node_capacity, edge_capacity);

  unsigned char* device_payload = nullptr;
  unsigned char* host_payload = nullptr;
  try {
    device_payload = device_allocate<unsigned char>(
        payload_bytes, "allocate BF12 compact result payload");
    host_payload = pinned_allocate<unsigned char>(
        payload_bytes, "allocate BF12 pinned compact result payload");
  } catch (...) {
    device_free(device_payload);
    pinned_free(host_payload);
    throw;
  }

  device_free(workspace.result_payload);
  pinned_free(workspace.host_result_payload);
  workspace.result_payload = device_payload;
  workspace.host_result_payload = host_payload;
  workspace.result_payload_bytes = payload_bytes;
  workspace.result_node_capacity = node_capacity;
  workspace.result_edge_capacity = edge_capacity;
  workspace.result_nodes =
      node_capacity == 0
          ? nullptr
          : reinterpret_cast<Index*>(device_payload);
  workspace.host_result_nodes =
      node_capacity == 0
          ? nullptr
          : reinterpret_cast<Index*>(host_payload);
  workspace.result_edges =
      edge_capacity == 0
          ? nullptr
          : reinterpret_cast<DeviceOffset*>(device_payload + node_bytes);
  workspace.host_result_edges =
      edge_capacity == 0
          ? nullptr
          : reinterpret_cast<DeviceOffset*>(host_payload + node_bytes);
  workspace.result_edge_costs =
      edge_capacity == 0
          ? nullptr
          : reinterpret_cast<float*>(device_payload + node_bytes +
                                     edge_index_bytes);
  workspace.host_result_edge_costs =
      edge_capacity == 0
          ? nullptr
          : reinterpret_cast<float*>(host_payload + node_bytes +
                                     edge_index_bytes);
}

void ensure_transfer_header_capacity(DeviceWorkspace& workspace,
                                     std::size_t required_bytes) {
  if (required_bytes <= workspace.transfer_header_capacity) return;
  const std::size_t capacity = geometric_capacity(
      workspace.transfer_header_capacity, required_bytes,
      std::numeric_limits<std::size_t>::max());
  unsigned char* device = nullptr;
  unsigned char* host = nullptr;
  try {
    device = device_allocate<unsigned char>(
        capacity, "allocate BF12 batched transfer header");
    host = pinned_allocate<unsigned char>(
        capacity, "allocate BF12 pinned batched transfer header");
  } catch (...) {
    device_free(device);
    pinned_free(host);
    throw;
  }
  device_free(workspace.transfer_header);
  pinned_free(workspace.host_transfer_header);
  workspace.transfer_header = device;
  workspace.host_transfer_header = host;
  workspace.transfer_header_capacity = capacity;
}

void ensure_telemetry_round_capacity(DeviceWorkspace& workspace,
                                     std::size_t required_rounds) {
  required_rounds = std::min(required_rounds, kMaximumTelemetryRounds);
  if (required_rounds <= workspace.telemetry_round_capacity) return;
  unsigned int* active = nullptr;
  unsigned int* host_active = nullptr;
  unsigned long long* frontiers = nullptr;
  unsigned long long* host_frontiers = nullptr;
  try {
    active = device_allocate<unsigned int>(required_rounds,
                                            "allocate BF12 round activity");
    host_active = pinned_allocate<unsigned int>(
        required_rounds, "allocate BF12 host round activity");
    frontiers = device_allocate<unsigned long long>(
        required_rounds, "allocate BF12 round frontiers");
    host_frontiers = pinned_allocate<unsigned long long>(
        required_rounds, "allocate BF12 host round frontiers");
  } catch (...) {
    device_free(active);
    pinned_free(host_active);
    device_free(frontiers);
    pinned_free(host_frontiers);
    throw;
  }
  device_free(workspace.telemetry_active_queries);
  pinned_free(workspace.host_telemetry_active_queries);
  device_free(workspace.telemetry_frontier_sizes);
  pinned_free(workspace.host_telemetry_frontier_sizes);
  workspace.telemetry_active_queries = active;
  workspace.host_telemetry_active_queries = host_active;
  workspace.telemetry_frontier_sizes = frontiers;
  workspace.host_telemetry_frontier_sizes = host_frontiers;
  workspace.telemetry_round_capacity = required_rounds;
}

bool terminal_inside_bounds(const HostSidecarView& sidecars,
                            Index node,
                            const DeviceBounds& bounds) {
  const std::int32_t x = (*sidecars.route_end_x)[node];
  const std::int32_t y = (*sidecars.route_end_y)[node];
  return x != kMissingCoordinate && y != kMissingCoordinate &&
         x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

NormalizedBatch normalize_batch(
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options,
    Offset rows,
    const HostSidecarView& sidecars) {
  if (descriptors.empty()) {
    throw std::invalid_argument("BF12 batch must contain at least one query");
  }
  if (descriptors.size() >=
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error("BF12 query count exceeds uint32");
  }
  const std::size_t state_count = require_multiply(
      descriptors.size(), static_cast<std::size_t>(rows),
      "BF12 composite state count overflow");
  if (state_count >= std::numeric_limits<StateIndex>::max()) {
    throw std::overflow_error(
        "BF12 batch_size * vertex_count exceeds uint32 composite indexing");
  }
  if (options.target_check_interval <= 0) {
    throw std::invalid_argument("BF12 target-check interval must be positive");
  }

  NormalizedBatch normalized;
  normalized.descriptors.reserve(descriptors.size());
  for (std::size_t query = 0; query < descriptors.size(); ++query) {
    const BellmanFord12QueryDescriptor& input = descriptors[query];
    const std::uint64_t source_end =
        static_cast<std::uint64_t>(input.source_begin) + input.source_count;
    const std::uint64_t target_end =
        static_cast<std::uint64_t>(input.target_begin) + input.target_count;
    if (source_end > flattened_sources.size() ||
        target_end > flattened_targets.size()) {
      throw std::out_of_range("BF12 descriptor endpoint range is invalid");
    }
    if (input.source_count == 0 || input.target_count == 0) {
      throw std::invalid_argument(
          "BF12 queries require at least one source and one target");
    }
    if (input.target_count >
        static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "BF12 per-query target count exceeds signed device counters");
    }
    DeviceQueryDescriptor device{};
    device.source_begin =
        static_cast<std::uint32_t>(normalized.source_states.size());
    device.target_begin = static_cast<std::uint32_t>(normalized.targets.size());
    device.max_iterations =
        input.max_iterations < 0 ? static_cast<int>(rows)
                                 : input.max_iterations;
    device.target_check_interval = input.target_check_interval == 0
                                       ? options.target_check_interval
                                       : input.target_check_interval;
    if (device.target_check_interval <= 0) {
      throw std::invalid_argument(
          "BF12 per-query target-check interval must be positive");
    }
    device.original_query_index = input.original_query_index;
    if (options.enable_bounding_boxes && input.bounds.enabled) {
      if (input.bounds.min_x > input.bounds.max_x ||
          input.bounds.min_y > input.bounds.max_y) {
        throw std::invalid_argument("BF12 bounding box is inverted");
      }
      device.bounds = {input.bounds.min_x, input.bounds.max_x,
                       input.bounds.min_y, input.bounds.max_y, 1};
    }

    std::unordered_set<int> seen_sources;
    seen_sources.reserve(static_cast<std::size_t>(input.source_count));
    for (std::uint64_t item = input.source_begin; item < source_end; ++item) {
      const int source = flattened_sources[static_cast<std::size_t>(item)];
      if (source < 0 || static_cast<Offset>(source) >= rows) {
        throw std::out_of_range("BF12 source is outside the graph");
      }
      if (device.bounds.enabled != 0 &&
          !terminal_inside_bounds(sidecars, source, device.bounds)) {
        throw std::invalid_argument(
            "BF12 bounded source lacks an in-bounds coordinate");
      }
      if (!seen_sources.insert(source).second) continue;
      normalized.source_states.push_back(make_state_index(
          static_cast<std::uint32_t>(query), static_cast<std::uint32_t>(source),
          static_cast<std::uint32_t>(rows)));
    }
    if (normalized.source_states.size() == device.source_begin) {
      throw std::invalid_argument("BF12 query has no unique source");
    }
    device.source_count = static_cast<std::uint32_t>(
        normalized.source_states.size() - device.source_begin);
    for (std::uint64_t item = input.target_begin; item < target_end; ++item) {
      const int target = flattened_targets[static_cast<std::size_t>(item)];
      if (target < 0 || static_cast<Offset>(target) >= rows) {
        throw std::out_of_range("BF12 target is outside the graph");
      }
      if (device.bounds.enabled != 0 &&
          !terminal_inside_bounds(sidecars, target, device.bounds)) {
        throw std::invalid_argument(
            "BF12 bounded target lacks an in-bounds coordinate");
      }
      normalized.targets.push_back(target);
      normalized.target_states.push_back(make_state_index(
          static_cast<std::uint32_t>(query), static_cast<std::uint32_t>(target),
          static_cast<std::uint32_t>(rows)));
    }
    device.target_count = input.target_count;
    normalized.descriptors.push_back(device);
  }
  if (normalized.source_states.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      normalized.targets.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error("BF12 flattened endpoints exceed uint32");
  }
  return normalized;
}

void throw_controller_error(int error_status) {
  switch (error_status) {
    case 0:
      return;
    case 1:
      throw std::runtime_error("BF12 encountered an invalid vertex ID");
    case 2:
      throw std::runtime_error("BF12 encountered invalid CSR row offsets");
    case 3:
      throw std::runtime_error(
          "BF12 effective edge weight is nonfinite or negative");
    case 4:
      throw std::runtime_error("BF12 combined frontier overflowed");
    case 5:
      throw std::runtime_error("BF12 touched-state list overflowed");
    default:
      throw std::runtime_error("BF12 controller returned an unknown error");
  }
}

BellmanFord12CooperativeCapability query_cooperative_capability(
    DeviceWorkspace& workspace,
    bool require_occupancy = true) {
  if (!workspace.device_capability_initialized) {
    BellmanFord12CooperativeCapability capability;
    capability.default_stream_required = true;
    int device = -1;
    check_hip(hipGetDevice(&device), "get BF12 HIP device");
    capability.device = device;
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, device),
              "get BF12 HIP device properties");
    capability.compute_unit_count = properties.multiProcessorCount;
    capability.architecture = properties.gcnArchName;
    int supported = 0;
    const hipError_t attribute_status = hipDeviceGetAttribute(
        &supported, hipDeviceAttributeCooperativeLaunch, device);
    if (attribute_status != hipSuccess || supported == 0) {
      if (attribute_status != hipSuccess) (void)hipGetLastError();
      capability.reason = "device does not support cooperative launch";
      workspace.cooperative_blocks = 0;
    } else {
      capability.device_supports_cooperative_launch = true;
      capability.reason =
          "cooperative occupancy was not queried for host-batch mode";
    }
    workspace.cached_capability = std::move(capability);
    workspace.device_capability_initialized = true;
  }

  BellmanFord12CooperativeCapability capability =
      workspace.cached_capability;
  if (!capability.device_supports_cooperative_launch ||
      workspace.cooperative_blocks == 0) {
    return capability;
  }
  if (workspace.cooperative_blocks > 0) {
    capability.legally_resident_blocks = workspace.cooperative_blocks;
    return capability;
  }
  if (!require_occupancy) return capability;

  int blocks_per_cu = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_cu, cooperative_batch_controller_kernel, kBlockSize, 0);
  if (occupancy_status != hipSuccess || blocks_per_cu <= 0 ||
      capability.compute_unit_count <= 0) {
    if (occupancy_status != hipSuccess) (void)hipGetLastError();
    capability.reason =
        "cooperative-controller occupancy could not be established";
    workspace.cooperative_blocks = 0;
    workspace.cached_capability = capability;
    return capability;
  }
  const long long legal =
      static_cast<long long>(blocks_per_cu) * capability.compute_unit_count;
  capability.legally_resident_blocks =
      legal > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max()
                                              : static_cast<int>(legal);
  workspace.cooperative_blocks = capability.legally_resident_blocks;
  capability.reason = "cooperative controller is available on the default stream";
  workspace.cached_capability = capability;
  return capability;
}

void upload_batch_inputs(DeviceWorkspace& workspace,
                         const NormalizedBatch& batch) {
  ensure_endpoint_capacity(workspace, batch.source_states.size(),
                           batch.targets.size());
  ensure_reconstruction_target_capacity(workspace, batch.targets.size());
  ensure_transfer_header_capacity(
      workspace,
      transfer_header_layout(batch.descriptors.size(), batch.targets.size())
          .total_bytes);
  if (!batch.source_states.empty()) {
    check_hip(hipMemcpyAsync(workspace.source_states,
                             batch.source_states.data(),
                             batch.source_states.size() * sizeof(StateIndex),
                             hipMemcpyHostToDevice, workspace.stream),
              "copy BF12 source states");
  }
  if (!batch.targets.empty()) {
    check_hip(hipMemcpyAsync(workspace.target_states,
                             batch.target_states.data(),
                             batch.target_states.size() * sizeof(StateIndex),
                             hipMemcpyHostToDevice, workspace.stream),
              "copy BF12 target states");
  }
  check_hip(hipMemcpyAsync(workspace.descriptors, batch.descriptors.data(),
                           batch.descriptors.size() *
                               sizeof(DeviceQueryDescriptor),
                           hipMemcpyHostToDevice, workspace.stream),
            "copy BF12 query descriptors");
}

void defensive_dense_reset(DeviceWorkspace& workspace) {
  hipLaunchKernelGGL(clear_dense_state_kernel,
                     grid_for_items(workspace.state_capacity),
                     dim3(kBlockSize), 0, workspace.stream,
                     workspace.state_capacity, workspace.best_state,
                     workspace.next_marks, workspace.source_mask);
  check_hip(hipGetLastError(), "defensively reset BF12 query state");
  check_hip(hipMemsetAsync(workspace.touched_count, 0, sizeof(unsigned int),
                           workspace.stream),
            "defensively reset BF12 touched count");
}

unsigned int initial_active_query_count(const NormalizedBatch& batch) {
  return static_cast<unsigned int>(std::count_if(
      batch.descriptors.begin(), batch.descriptors.end(),
      [](const DeviceQueryDescriptor& descriptor) {
        return descriptor.max_iterations != 0;
      }));
}

bool target_check_due(const NormalizedBatch& batch,
                      unsigned int completed_round) {
  return std::any_of(
      batch.descriptors.begin(), batch.descriptors.end(),
      [completed_round](const DeviceQueryDescriptor& descriptor) {
        return completed_round %
                   static_cast<unsigned int>(
                       descriptor.target_check_interval) ==
               0;
      });
}

void initialize_host_batch(DeviceWorkspace& workspace,
                           const NormalizedBatch& batch) {
  const bool defensive_reset_required = workspace.needs_dense_reset;
  workspace.needs_dense_reset = true;
  if (defensive_reset_required) {
    defensive_dense_reset(workspace);
  } else {
    hipLaunchKernelGGL(clear_sparse_state_kernel,
                       sparse_reset_grid(workspace.state_capacity),
                       dim3(kBlockSize), 0, workspace.stream,
                       workspace.state_capacity, workspace.touched_states,
                       workspace.touched_count, workspace.best_state,
                       workspace.next_marks, workspace.source_mask);
    check_hip(hipGetLastError(), "sparse-reset BF12 query state");
  }
  const unsigned int sources =
      static_cast<unsigned int>(batch.source_states.size());
  hipLaunchKernelGGL(
      initialize_batch_kernel, grid_for_items(batch.descriptors.size()),
      dim3(kBlockSize), 0, workspace.stream, batch.descriptors.size(), sources,
      initial_active_query_count(batch), workspace.descriptors,
      workspace.query_controls, workspace.batch_control,
      workspace.touched_count, workspace.telemetry);
  check_hip(hipGetLastError(), "initialize BF12 host batch");
  hipLaunchKernelGGL(seed_sources_kernel,
                     grid_for_items(batch.source_states.size()),
                     dim3(kBlockSize), 0, workspace.stream,
                     workspace.source_states, sources, workspace.best_state,
                     workspace.frontier, workspace.source_mask,
                     workspace.touched_states);
  check_hip(hipGetLastError(), "seed BF12 host batch");
}

void run_host_controller(DeviceGraph graph,
                         DeviceWorkspace& workspace,
                         const NormalizedBatch& batch,
                         bool telemetry_enabled,
                         BellmanFord12BatchTelemetry& telemetry) {
  initialize_host_batch(workspace, batch);
  unsigned int active = initial_active_query_count(batch);
  unsigned int frontier_count =
      static_cast<unsigned int>(batch.source_states.size());
  unsigned int current_index = 0;
  unsigned int round = 0;
  while (active != 0) {
    hipLaunchKernelGGL(reset_round_kernel,
                       grid_for_items(batch.descriptors.size()),
                       dim3(kBlockSize), 0, workspace.stream,
                       batch.descriptors.size(), workspace.query_controls,
                       workspace.batch_control);
    check_hip(hipGetLastError(), "reset BF12 host-controller round");
    const StateIndex* current =
        current_index == 0 ? workspace.frontier : workspace.next_frontier;
    StateIndex* next =
        current_index == 0 ? workspace.next_frontier : workspace.frontier;
    hipLaunchKernelGGL(
        relax_frontier_kernel, grid_for_items(frontier_count),
        dim3(kBlockSize), 0, workspace.stream, graph, current, frontier_count,
        static_cast<unsigned int>(batch.descriptors.size()), round + 1u,
        workspace.descriptors, workspace.dynamic_vertex_cost,
        workspace.best_state, next, workspace.next_marks,
        workspace.source_mask, workspace.touched_states,
        workspace.touched_count, workspace.query_controls,
        workspace.batch_control, workspace.telemetry,
        telemetry_enabled ? 1 : 0);
    check_hip(hipGetLastError(), "relax BF12 combined host frontier");
    if (target_check_due(batch, round + 1u)) {
      hipLaunchKernelGGL(
          update_target_status_kernel, grid_for_items(batch.targets.size()),
          dim3(kBlockSize), 0, workspace.stream, workspace.target_states,
          batch.targets.size(), workspace.descriptors, workspace.best_state,
          workspace.query_controls, static_cast<std::uint32_t>(graph.rows));
      check_hip(hipGetLastError(), "update BF12 host target status");
    }
    hipLaunchKernelGGL(
        finish_round_kernel, grid_for_items(batch.descriptors.size()),
        dim3(kBlockSize), 0, workspace.stream, batch.descriptors.size(),
        workspace.descriptors, workspace.query_controls,
        workspace.batch_control, workspace.telemetry_active_queries,
        workspace.telemetry_frontier_sizes, std::size_t{0});
    check_hip(hipGetLastError(), "finish BF12 host-controller round");
    check_hip(hipMemcpyAsync(workspace.host_round_status,
                             workspace.batch_control,
                             sizeof(DeviceHostRoundStatus),
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF12 host-controller batch status");
    check_hip(hipStreamSynchronize(workspace.stream),
              "synchronize BF12 host-controller round");
    ++telemetry.synchronization_count;
    ++telemetry.host_fallback_round_count;
    ++telemetry.global_traversal_rounds;
    telemetry.device_to_host_bytes += sizeof(DeviceHostRoundStatus);
    if (telemetry_enabled) {
      telemetry.active_queries_per_round.push_back(
          workspace.host_round_status->active_query_count);
      telemetry.combined_frontier_size_per_round.push_back(frontier_count);
    }
    throw_controller_error(workspace.host_round_status->error_status);
    active = workspace.host_round_status->active_query_count;
    frontier_count = workspace.host_round_status->frontier_count;
    current_index = workspace.host_round_status->current_frontier_index;
    ++round;
    if (round == std::numeric_limits<unsigned int>::max()) {
      throw std::overflow_error("BF12 global round counter overflowed");
    }
  }
}

void launch_cooperative_controller(DeviceGraph graph,
                                   DeviceWorkspace& workspace,
                                   const NormalizedBatch& batch,
                                   bool telemetry_enabled,
                                   int blocks) {
  const StateIndex* sources = workspace.source_states;
  unsigned int source_count =
      static_cast<unsigned int>(batch.source_states.size());
  const StateIndex* targets = workspace.target_states;
  unsigned int target_count =
      static_cast<unsigned int>(batch.target_states.size());
  unsigned int query_count =
      static_cast<unsigned int>(batch.descriptors.size());
  unsigned int active_count = initial_active_query_count(batch);
  const DeviceQueryDescriptor* descriptors = workspace.descriptors;
  const float* dynamic_cost = workspace.dynamic_vertex_cost;
  unsigned long long* best_state = workspace.best_state;
  StateIndex* frontier = workspace.frontier;
  StateIndex* next_frontier = workspace.next_frontier;
  unsigned int* marks = workspace.next_marks;
  unsigned char* source_mask = workspace.source_mask;
  StateIndex* touched_states = workspace.touched_states;
  unsigned int* touched_count = workspace.touched_count;
  std::size_t state_capacity = workspace.state_capacity;
  DeviceQueryControl* controls = workspace.query_controls;
  DeviceBatchControl* batch_control = workspace.batch_control;
  DeviceTelemetry* device_telemetry = workspace.telemetry;
  unsigned int* round_active = workspace.telemetry_active_queries;
  unsigned long long* round_frontier = workspace.telemetry_frontier_sizes;
  std::size_t telemetry_round_capacity =
      telemetry_enabled ? workspace.telemetry_round_capacity : 0;
  int collect_telemetry = telemetry_enabled ? 1 : 0;
  void* arguments[] = {
      &graph,          &sources,          &source_count,
      &targets,        &target_count,     &query_count,
      &active_count,   &descriptors,      &dynamic_cost,
      &best_state,     &frontier,         &next_frontier,
      &marks,          &source_mask,      &touched_states,
      &touched_count,  &state_capacity,   &controls,
      &batch_control,  &device_telemetry, &round_active,
      &round_frontier, &telemetry_round_capacity,
      &collect_telemetry};
  check_hip(hipLaunchCooperativeKernel(
                cooperative_batch_controller_kernel,
                dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
                arguments, 0, workspace.stream),
            "launch BF12 cooperative batch controller");
}

std::size_t default_node_arena_capacity(std::size_t target_count) {
  return std::max<std::size_t>(1, require_multiply(
      target_count, 64, "BF12 default node arena capacity overflow"));
}

std::size_t default_edge_arena_capacity(std::size_t target_count) {
  return std::max<std::size_t>(1, require_multiply(
      target_count, 63, "BF12 default edge arena capacity overflow"));
}

void launch_reconstruction(DeviceGraph graph,
                           DeviceWorkspace& workspace,
                           std::size_t target_count) {
  hipLaunchKernelGGL(
      scan_target_offsets_kernel, dim3(1), dim3(1), 0, workspace.stream,
      workspace.target_summaries, target_count,
      static_cast<unsigned long long>(workspace.result_node_capacity),
      static_cast<unsigned long long>(workspace.result_edge_capacity),
      workspace.node_offsets, workspace.edge_offsets,
      workspace.result_header);
  check_hip(hipGetLastError(), "scan BF12 reconstruction offsets");
  hipLaunchKernelGGL(
      materialize_target_paths_kernel, grid_for_items(target_count),
      dim3(kBlockSize), 0, workspace.stream, workspace.best_state,
      workspace.source_mask, graph.rows, graph.nnz, graph.destinations,
      graph.rowptr, workspace.target_states, workspace.target_summaries,
      workspace.node_offsets, workspace.edge_offsets, target_count,
      workspace.result_header, workspace.result_nodes,
      workspace.result_edges, graph.edge_values, graph.base_vertex_cost,
      workspace.dynamic_vertex_cost, workspace.result_edge_costs);
  check_hip(hipGetLastError(), "materialize BF12 target paths");
}

std::uint64_t enqueue_result_copies(DeviceWorkspace& workspace,
                                    std::size_t query_count,
                                    std::size_t target_count,
                                    bool copy_telemetry) {
  std::uint64_t bytes = 0;
  auto add_bytes = [&](std::size_t count) {
    bytes = static_cast<std::uint64_t>(require_add(
        static_cast<std::size_t>(bytes), count,
        "BF12 device-to-host byte counter overflow"));
  };
  const TransferHeaderLayout layout =
      transfer_header_layout(query_count, target_count);
  if (layout.total_bytes > workspace.transfer_header_capacity) {
    throw std::logic_error("BF12 transfer header was not preallocated");
  }
  check_hip(hipMemsetAsync(workspace.transfer_header, 0, layout.total_bytes,
                           workspace.stream),
            "clear BF12 batched transfer header");
  const std::size_t packed_items =
      std::max(query_count, require_add(target_count, 1,
                                        "BF12 packed-header item overflow"));
  hipLaunchKernelGGL(
      pack_transfer_header_kernel, grid_for_items(packed_items),
      dim3(kBlockSize), 0, workspace.stream, workspace.query_controls,
      query_count, workspace.batch_control, workspace.target_summaries,
      target_count, workspace.node_offsets, workspace.edge_offsets,
      workspace.result_header, workspace.transfer_header, layout);
  check_hip(hipGetLastError(), "pack BF12 result transfer header");
  check_hip(hipMemcpyAsync(workspace.host_transfer_header,
                           workspace.transfer_header, layout.total_bytes,
                           hipMemcpyDeviceToHost, workspace.stream),
            "copy BF12 batched header and offsets");
  add_bytes(layout.total_bytes);
  if (workspace.result_payload_bytes != 0) {
    check_hip(hipMemcpyAsync(workspace.host_result_payload,
                             workspace.result_payload,
                             workspace.result_payload_bytes,
                             hipMemcpyDeviceToHost, workspace.stream),
              "copy BF12 compact result payload");
    add_bytes(workspace.result_payload_bytes);
  }
  if (copy_telemetry) {
    check_hip(hipMemcpyAsync(workspace.host_telemetry, workspace.telemetry,
                             sizeof(DeviceTelemetry), hipMemcpyDeviceToHost,
                             workspace.stream),
              "copy BF12 telemetry counters");
    add_bytes(sizeof(DeviceTelemetry));
    const std::size_t rounds = workspace.telemetry_round_capacity;
    if (rounds != 0) {
      check_hip(hipMemcpyAsync(
                    workspace.host_telemetry_active_queries,
                    workspace.telemetry_active_queries,
                    rounds * sizeof(unsigned int), hipMemcpyDeviceToHost,
                    workspace.stream),
                "copy BF12 round activity");
      check_hip(hipMemcpyAsync(
                    workspace.host_telemetry_frontier_sizes,
                    workspace.telemetry_frontier_sizes,
                    rounds * sizeof(unsigned long long),
                    hipMemcpyDeviceToHost, workspace.stream),
                "copy BF12 round frontiers");
      add_bytes(rounds *
                (sizeof(unsigned int) + sizeof(unsigned long long)));
    }
  }
  return bytes;
}

void unpack_result_header(DeviceWorkspace& workspace,
                          std::size_t query_count,
                          std::size_t target_count) {
  const TransferHeaderLayout layout =
      transfer_header_layout(query_count, target_count);
  if (layout.total_bytes > workspace.transfer_header_capacity) {
    throw std::logic_error("BF12 copied transfer header exceeds its capacity");
  }
  const unsigned char* header = workspace.host_transfer_header;
  std::memcpy(workspace.host_query_controls,
              header + layout.query_controls,
              query_count * sizeof(DeviceQueryControl));
  std::memcpy(workspace.host_batch_control,
              header + layout.batch_control,
              sizeof(DeviceBatchControl));
  std::memcpy(workspace.host_target_summaries,
              header + layout.target_summaries,
              target_count * sizeof(DeviceTargetSummary));
  std::memcpy(workspace.host_node_offsets,
              header + layout.node_offsets,
              (target_count + 1) * sizeof(unsigned long long));
  std::memcpy(workspace.host_edge_offsets,
              header + layout.edge_offsets,
              (target_count + 1) * sizeof(unsigned long long));
  std::memcpy(workspace.host_result_header,
              header + layout.result_header,
              sizeof(DeviceResultHeader));
}

BellmanFord12BatchResult build_host_results(
    const NormalizedBatch& batch,
    DeviceWorkspace& workspace,
    bool bounded_run,
    bool arena_overflow_was_retried) {
  BellmanFord12BatchResult output;
  output.query_results.reserve(batch.descriptors.size());
  output.query_statuses.reserve(batch.descriptors.size());
  output.original_query_indices.reserve(batch.descriptors.size());
  for (std::size_t query = 0; query < batch.descriptors.size(); ++query) {
    const DeviceQueryDescriptor& descriptor = batch.descriptors[query];
    const DeviceQueryControl& control = workspace.host_query_controls[query];
    throw_controller_error(control.error_status);
    BellmanFordCsrResult result;
    result.target = -1;
    result.iterations_used = control.iterations;
    result.converged = control.converged != 0;
    result.stopped_on_target = control.early_stopped != 0;
    result.target_reached = true;
    result.target_distances.reserve(descriptor.target_count);
    result.target_sources.reserve(descriptor.target_count);
    result.target_path_offsets.reserve(descriptor.target_count + 1);
    result.target_edge_offsets.reserve(descriptor.target_count + 1);
    result.target_path_offsets.push_back(0);
    result.target_edge_offsets.push_back(0);
    bool identity_only = true;
    for (std::size_t local = 0; local < descriptor.target_count; ++local) {
      const std::size_t target_item = descriptor.target_begin + local;
      const DeviceTargetSummary& summary =
          workspace.host_target_summaries[target_item];
      if (summary.status == kTargetPathInvalid) {
        throw std::runtime_error(
            "BF12 produced an invalid or cyclic predecessor chain");
      }
      if (summary.status == kTargetUnreachable) {
        result.target_distances.push_back(
            std::numeric_limits<float>::infinity());
        result.target_sources.push_back(-1);
        result.target_reached = false;
        identity_only = false;
      } else {
        const std::size_t node_begin = static_cast<std::size_t>(
            workspace.host_node_offsets[target_item]);
        const std::size_t node_end = static_cast<std::size_t>(
            workspace.host_node_offsets[target_item + 1]);
        const std::size_t edge_begin = static_cast<std::size_t>(
            workspace.host_edge_offsets[target_item]);
        const std::size_t edge_end = static_cast<std::size_t>(
            workspace.host_edge_offsets[target_item + 1]);
        if (node_begin > node_end || edge_begin > edge_end ||
            node_end > workspace.result_node_capacity ||
            edge_end > workspace.result_edge_capacity ||
            node_end - node_begin != summary.node_count ||
            edge_end - edge_begin != summary.edge_count ||
            node_begin == node_end ||
            workspace.host_result_nodes[node_begin] != summary.root ||
            workspace.host_result_nodes[node_end - 1] !=
                batch.targets[target_item]) {
          throw std::runtime_error("BF12 compact target path is malformed");
        }
        result.target_distances.push_back(host_state_distance(summary.state));
        result.target_sources.push_back(summary.root);
        result.target_path_nodes.insert(
            result.target_path_nodes.end(),
            workspace.host_result_nodes + node_begin,
            workspace.host_result_nodes + node_end);
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          result.target_path_edges.push_back(
              static_cast<Offset>(workspace.host_result_edges[edge]));
          result.target_path_edge_costs.push_back(
              workspace.host_result_edge_costs[edge]);
        }
        identity_only = identity_only && summary.edge_count == 0 &&
                        host_state_distance(summary.state) == 0.0f;
      }
      if (result.target_path_nodes.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          result.target_path_edges.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BF12 result offsets exceed int");
      }
      result.target_path_offsets.push_back(
          static_cast<int>(result.target_path_nodes.size()));
      result.target_edge_offsets.push_back(
          static_cast<int>(result.target_path_edges.size()));
    }

    BellmanFord12QueryStatus status;
    status.converged = control.converged != 0;
    status.stopped_on_target = control.early_stopped != 0;
    status.paths_certified = status.converged || status.stopped_on_target;
    status.all_targets_reached = result.target_reached;
    status.hit_iteration_limit = control.hit_iteration_limit != 0;
    status.iterations = control.iterations;
    if (status.hit_iteration_limit && identity_only &&
        status.all_targets_reached) {
      status.hit_iteration_limit = false;
      status.stopped_on_target = true;
      status.paths_certified = true;
      result.stopped_on_target = true;
    }
    status.bounded_search_miss =
        bounded_run && descriptor.bounds.enabled != 0 &&
        !status.all_targets_reached;
    status.result_arena_overflow = false;
    (void)arena_overflow_was_retried;
    output.query_results.push_back(std::move(result));
    output.query_statuses.push_back(status);
    output.original_query_indices.push_back(descriptor.original_query_index);
  }
  return output;
}

}  // namespace rips_sssp_bf12

struct BellmanFord12CsrGraph::Impl {
  using Offset = rips_sssp_bf12::Offset;

  Offset rows = 0;
  Offset nnz = 0;
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_cost;
  rips_sssp_bf12::DeviceGraphOwner device;
  int owner_device = -1;
  std::size_t device_bytes = 0;

  Impl(const HostCsrF32& graph,
       const rips_sssp_bf12::HostSidecarView& sidecars,
       hipStream_t stream) {
    rips_sssp_bf12::validate_csr(graph);
    if (!sidecars.already_validated) {
      rips_sssp_bf12::validate_sidecars(sidecars, graph.rows);
    }
    check_hip(hipGetDevice(&owner_device), "get BF12 graph owner device");
    rows = graph.rows;
    nnz = graph.nnz;
    route_end_x = *sidecars.route_end_x;
    route_end_y = *sidecars.route_end_y;
    base_vertex_cost = *sidecars.base_vertex_cost;
    rips_sssp_bf12::HostSidecarView owned{
        &route_end_x, &route_end_y, &base_vertex_cost};
    device = rips_sssp_bf12::copy_graph_to_device(graph, owned, stream);
    device_bytes =
        bf12_worker_policy::shared_graph_device_bytes(
            static_cast<std::size_t>(rows), static_cast<std::size_t>(nnz));
  }

  ~Impl() {
    ScopedHipDevice scope(owner_device);
    if (scope.active()) rips_sssp_bf12::free_graph(device);
  }

  rips_sssp_bf12::HostSidecarView host_sidecars() const {
    return {&route_end_x, &route_end_y, &base_vertex_cost};
  }
};

struct BellmanFord12CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord12CsrGraph::Impl> graph;
  rips_sssp_bf12::DeviceWorkspace workspace;
  std::size_t maximum_batch = 0;
  hipStream_t stream = nullptr;
  int owner_device = -1;
  mutable std::mutex operation_mutex;
  BellmanFord12BatchTelemetry aggregate_telemetry;
  std::uint64_t aggregate_selected_query_slots = 0;
  bool dynamic_costs_are_unit = true;

  Impl(std::shared_ptr<const BellmanFord12CsrGraph::Impl> graph_arg,
       std::size_t maximum_batch_size,
       hipStream_t stream_arg)
      : graph(std::move(graph_arg)),
        maximum_batch(maximum_batch_size),
        stream(stream_arg) {
    if (!graph) throw std::invalid_argument("BF12 graph is null");
    check_hip(hipGetDevice(&owner_device), "get BF12 workspace owner device");
    if (owner_device != graph->owner_device) {
      throw std::invalid_argument(
          "BF12 graph and workspace must use the same HIP device");
    }
    workspace = rips_sssp_bf12::make_workspace(graph->rows, stream);
  }

  ~Impl() {
    ScopedHipDevice scope(owner_device);
    if (scope.active()) rips_sssp_bf12::free_workspace(workspace);
  }

  void require_stream(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "BF12 workspace calls must use its construction stream");
    }
    int current = -1;
    check_hip(hipGetDevice(&current), "get BF12 current device");
    if (current != owner_device) {
      throw std::invalid_argument(
          "BF12 workspace call selected a different HIP device");
    }
  }

  void upload_costs(const std::vector<float>& costs) {
    if (costs.size() != static_cast<std::size_t>(graph->rows)) {
      throw std::invalid_argument(
          "BF12 dynamic vertex-cost vector must contain V entries");
    }
    bool all_unit = true;
    for (const float cost : costs) {
      if (!std::isfinite(cost) || cost < 0.0f) {
        throw std::invalid_argument(
            "BF12 dynamic vertex costs must be finite and nonnegative");
      }
      all_unit = all_unit && cost == 1.0f;
    }
    // make_workspace initializes the device epoch to all ones. Pathfinder's
    // initial epoch commonly uploads the same vector immediately afterward;
    // validation above is retained, but the redundant H2D copy and sync are
    // not. A failed nontrivial upload clears this cache before work is queued,
    // so a later unit reset is never incorrectly skipped.
    if (all_unit && dynamic_costs_are_unit) return;
    dynamic_costs_are_unit = false;
    DrainStreamOnException drain{stream};
    check_hip(hipMemcpyAsync(workspace.dynamic_vertex_cost, costs.data(),
                             costs.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream),
              "copy BF12 dynamic vertex costs");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF12 dynamic-cost epoch");
    dynamic_costs_are_unit = all_unit;
    ++aggregate_telemetry.synchronization_count;
  }

  BellmanFord12WorkerDecision policy_decision(
      const rips_sssp_bf12::NormalizedBatch& batch,
      const BellmanFord12BatchOptions& options,
      const BellmanFord12CooperativeCapability& capability,
      std::size_t node_capacity,
      std::size_t edge_capacity) const {
    BellmanFord12WorkerPolicyInputs inputs;
    inputs.requested_batch_size = options.requested_batch_size;
    inputs.vertex_count = static_cast<std::size_t>(graph->rows);
    inputs.edge_count = static_cast<std::size_t>(graph->nnz);
    inputs.queries_waiting = batch.descriptors.size();
    inputs.source_count = batch.source_states.size();
    inputs.target_count = batch.targets.size();
    inputs.result_node_capacity = node_capacity;
    inputs.result_edge_capacity = edge_capacity;
    inputs.gpu_architecture = capability.architecture;
    inputs.compute_unit_count = capability.compute_unit_count;
    inputs.cooperative_launch_supported =
        capability.device_supports_cooperative_launch &&
        capability.legally_resident_blocks > 0;
    inputs.telemetry_enabled = options.enable_telemetry;
    inputs.graph_already_resident = true;

    std::size_t required_telemetry_rounds = 0;
    if (options.enable_telemetry) {
      for (const auto& descriptor : batch.descriptors) {
        required_telemetry_rounds = std::max(
            required_telemetry_rounds,
            static_cast<std::size_t>(std::max(0, descriptor.max_iterations)));
      }
      required_telemetry_rounds = std::min(
          required_telemetry_rounds,
          bf12_worker_policy::kMaximumTelemetryRounds);
    }

    // hipMemGetInfo excludes retained workspace storage.  Credit only the
    // exact policy charges whose backing capacity is already sufficient for
    // this batch.  Algebraically those reused charges cancel from both sides
    // of the admission test, while any growing allocation remains charged at
    // its full replacement-before-release peak.  This prevents a second full
    // batch from being rejected merely because the first batch's B*V state is
    // now resident, without weakening the configured free-memory reserve.
    std::size_t reusable_credit = 0;
    const auto add_credit = [&](std::size_t value) {
      reusable_credit =
          value > std::numeric_limits<std::size_t>::max() - reusable_credit
              ? std::numeric_limits<std::size_t>::max()
              : reusable_credit + value;
    };
    add_credit(bf12_worker_policy::shared_dynamic_cost_device_bytes(
        inputs.vertex_count));
    if (batch.descriptors.size() <= workspace.query_capacity) {
      add_credit(bf12_worker_policy::query_vertex_state_device_bytes(
          batch.descriptors.size(), inputs.vertex_count));
      add_credit(bf12_worker_policy::aligned_allocation_bytes(
          bf12_worker_policy::checked_multiply(
              batch.descriptors.size(),
              bf12_worker_policy::kPerQueryDescriptorBytes)));
      add_credit(bf12_worker_policy::aligned_allocation_bytes(
          bf12_worker_policy::checked_multiply(
              batch.descriptors.size(),
              bf12_worker_policy::kPerQueryControlBytes)));
    }
    const std::size_t retained_sources =
        bf12_worker_policy::rounded_retained_capacity(
            batch.source_states.size());
    if (retained_sources <= workspace.source_capacity) {
      const std::size_t one_source_array =
          bf12_worker_policy::aligned_allocation_bytes(
              bf12_worker_policy::checked_multiply(
                  retained_sources, sizeof(std::uint32_t)));
      add_credit(one_source_array);
    }
    const std::size_t retained_targets =
        bf12_worker_policy::rounded_retained_capacity(batch.targets.size());
    if (retained_targets <= workspace.target_capacity) {
      const std::size_t one_target_array =
          bf12_worker_policy::aligned_allocation_bytes(
              bf12_worker_policy::checked_multiply(
                  retained_targets, sizeof(std::uint32_t)));
      add_credit(one_target_array);
    }
    add_credit(bf12_worker_policy::aligned_allocation_bytes(
        bf12_worker_policy::kGlobalControlBytes));
    if (options.enable_telemetry) {
      if (required_telemetry_rounds <= workspace.telemetry_round_capacity) {
        add_credit(bf12_worker_policy::aligned_allocation_bytes(
            bf12_worker_policy::kTelemetryCounterBytes));
        add_credit(bf12_worker_policy::aligned_allocation_bytes(
            bf12_worker_policy::checked_multiply(
                bf12_worker_policy::kMaximumTelemetryRounds,
                sizeof(std::uint32_t))));
        add_credit(bf12_worker_policy::aligned_allocation_bytes(
            bf12_worker_policy::checked_multiply(
                bf12_worker_policy::kMaximumTelemetryRounds,
                sizeof(std::uint64_t))));
      }
    }
    const std::size_t retained_nodes =
        bf12_worker_policy::rounded_retained_capacity(node_capacity);
    const std::size_t retained_edges =
        bf12_worker_policy::rounded_retained_capacity(edge_capacity);
    const std::size_t transfer_bytes =
        rips_sssp_bf12::transfer_header_layout(
            batch.descriptors.size(), batch.targets.size())
            .total_bytes;
    const bool result_storage_reusable =
        batch.descriptors.size() <= workspace.query_capacity &&
        retained_targets <= workspace.reconstruction_target_capacity &&
        retained_nodes <= workspace.result_node_capacity &&
        retained_edges <= workspace.result_edge_capacity &&
        transfer_bytes <= workspace.transfer_header_capacity;
    if (result_storage_reusable) {
      add_credit(bf12_worker_policy::result_device_bytes_estimate(
          batch.descriptors.size(), batch.targets.size(), node_capacity,
          edge_capacity));
    }

    const bool all_policy_storage_reusable =
        batch.descriptors.size() <= workspace.query_capacity &&
        retained_sources <= workspace.source_capacity &&
        retained_targets <= workspace.target_capacity &&
        result_storage_reusable &&
        (!options.enable_telemetry ||
         required_telemetry_rounds <= workspace.telemetry_round_capacity);
    if (all_policy_storage_reusable) {
      // No policy-governed device allocation can occur in this run. Ask the
      // pure policy to enforce request/CU/index limits against the already
      // admitted batch without issuing a synchronous hipMemGetInfo query.
      // A configured reserve remains visible in the audit fields, but it need
      // not be backed by currently free memory when no allocation is pending.
      inputs.memory_reserve_bytes = options.memory_safety_reserve_bytes;
      const std::size_t admitted_bytes =
          bf12_worker_policy::total_device_bytes_estimate(
              inputs, batch.descriptors.size());
      inputs.available_device_bytes =
          inputs.memory_reserve_bytes >
                  std::numeric_limits<std::size_t>::max() - admitted_bytes
              ? std::numeric_limits<std::size_t>::max()
              : admitted_bytes + inputs.memory_reserve_bytes;
      return bf12_worker_policy::decide(inputs);
    }

    std::size_t current_free_bytes = 0;
    std::size_t current_total_bytes = 0;
    check_hip(hipMemGetInfo(&current_free_bytes, &current_total_bytes),
              "refresh BF12 available device memory");
    (void)current_total_bytes;
    inputs.memory_reserve_bytes = options.memory_safety_reserve_bytes != 0
                                      ? options.memory_safety_reserve_bytes
                                      : current_free_bytes / 4;
    inputs.available_device_bytes =
        reusable_credit > std::numeric_limits<std::size_t>::max() -
                              current_free_bytes
            ? std::numeric_limits<std::size_t>::max()
            : current_free_bytes + reusable_credit;
    return bf12_worker_policy::decide(inputs);
  }

  void merge_telemetry(const BellmanFord12BatchTelemetry& value) {
    aggregate_telemetry.enabled = aggregate_telemetry.enabled || value.enabled;
    aggregate_telemetry.requested_batch_size = std::max(
        aggregate_telemetry.requested_batch_size, value.requested_batch_size);
    aggregate_telemetry.selected_batch_size = std::max(
        aggregate_telemetry.selected_batch_size, value.selected_batch_size);
    aggregate_telemetry.submitted_query_count += value.submitted_query_count;
    aggregate_telemetry.batch_count += value.batch_count;
    aggregate_telemetry.controller_mode_used = value.controller_mode_used;
    aggregate_telemetry.cooperative_controller_launch_count +=
        value.cooperative_controller_launch_count;
    aggregate_telemetry.host_fallback_round_count +=
        value.host_fallback_round_count;
    aggregate_telemetry.synchronization_count += value.synchronization_count;
    aggregate_telemetry.global_traversal_rounds +=
        value.global_traversal_rounds;
    aggregate_telemetry.active_queries_per_round.insert(
        aggregate_telemetry.active_queries_per_round.end(),
        value.active_queries_per_round.begin(),
        value.active_queries_per_round.end());
    aggregate_telemetry.combined_frontier_size_per_round.insert(
        aggregate_telemetry.combined_frontier_size_per_round.end(),
        value.combined_frontier_size_per_round.begin(),
        value.combined_frontier_size_per_round.end());
    aggregate_telemetry.edges_examined += value.edges_examined;
    aggregate_telemetry.successful_relaxations +=
        value.successful_relaxations;
    aggregate_telemetry.touched_nodes_per_query.insert(
        aggregate_telemetry.touched_nodes_per_query.end(),
        value.touched_nodes_per_query.begin(),
        value.touched_nodes_per_query.end());
    aggregate_telemetry.maximum_touched_state_density = std::max(
        aggregate_telemetry.maximum_touched_state_density,
        value.maximum_touched_state_density);
    aggregate_telemetry.target_summary_nanoseconds +=
        value.target_summary_nanoseconds;
    aggregate_telemetry.reconstruction_nanoseconds +=
        value.reconstruction_nanoseconds;
    aggregate_telemetry.device_to_host_bytes += value.device_to_host_bytes;
    aggregate_telemetry.result_node_capacity = value.result_node_capacity;
    aggregate_telemetry.result_edge_capacity = value.result_edge_capacity;
    aggregate_telemetry.result_nodes_used = value.result_nodes_used;
    aggregate_telemetry.result_edges_used = value.result_edges_used;
    aggregate_telemetry.arena_overflow_retry_count +=
        value.arena_overflow_retry_count;
    aggregate_telemetry.bounded_query_retry_count +=
        value.bounded_query_retry_count;
    aggregate_telemetry.total_batch_wall_nanoseconds +=
        value.total_batch_wall_nanoseconds;
    const std::uint64_t value_slots =
        value.selected_batch_size >
                std::numeric_limits<std::uint64_t>::max() /
                    std::max<std::uint64_t>(1, value.batch_count)
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(value.selected_batch_size) *
                  value.batch_count;
    aggregate_selected_query_slots =
        value_slots > std::numeric_limits<std::uint64_t>::max() -
                          aggregate_selected_query_slots
            ? std::numeric_limits<std::uint64_t>::max()
            : aggregate_selected_query_slots + value_slots;
    aggregate_telemetry.batch_fill_ratio =
        aggregate_selected_query_slots == 0
            ? 0.0
            : static_cast<double>(aggregate_telemetry.submitted_query_count) /
                  static_cast<double>(aggregate_selected_query_slots);
  }

  BellmanFord12BatchResult run_once(
      const std::vector<int>& flattened_sources,
      const std::vector<int>& flattened_targets,
      const std::vector<BellmanFord12QueryDescriptor>& descriptors,
      const BellmanFord12BatchOptions& options) {
    using Clock = std::chrono::steady_clock;
    const auto batch_begin = Clock::now();
    rips_sssp_bf12::NormalizedBatch batch = rips_sssp_bf12::normalize_batch(
        flattened_sources, flattened_targets, descriptors, options,
        graph->rows, graph->host_sidecars());
    if (maximum_batch != 0 && batch.descriptors.size() > maximum_batch) {
      throw std::invalid_argument(
          "BF12 batch exceeds the workspace's caller-imposed maximum");
    }

    BellmanFord12CooperativeCapability capability =
        rips_sssp_bf12::query_cooperative_capability(
            workspace,
            options.controller_mode != BellmanFord12ControllerMode::HostBatch);
    const std::size_t initial_node_capacity =
        options.result_node_capacity != 0
            ? options.result_node_capacity
            : rips_sssp_bf12::default_node_arena_capacity(
                  batch.targets.size());
    const std::size_t initial_edge_capacity =
        options.result_edge_capacity != 0
            ? options.result_edge_capacity
            : rips_sssp_bf12::default_edge_arena_capacity(
                  batch.targets.size());
    const BellmanFord12WorkerDecision decision = policy_decision(
        batch, options, capability, initial_node_capacity,
        initial_edge_capacity);
    if (decision.selected_batch_size == 0) {
      throw std::runtime_error("BF12 cannot allocate a safe batch: " +
                               decision.limiting_reason);
    }
    if (batch.descriptors.size() > decision.selected_batch_size) {
      throw std::invalid_argument(
          "BF12 submitted batch exceeds the safe worker-policy selection (" +
          std::to_string(decision.selected_batch_size) + "): " +
          decision.limiting_reason);
    }

    BellmanFord12ControllerMode mode = options.controller_mode;
    const bool cooperative_available =
        capability.device_supports_cooperative_launch &&
        capability.legally_resident_blocks > 0 && stream == nullptr;
    if (mode == BellmanFord12ControllerMode::Auto) {
      mode = cooperative_available
                 ? BellmanFord12ControllerMode::CooperativeBatch
                 : BellmanFord12ControllerMode::HostBatch;
    } else if (mode == BellmanFord12ControllerMode::CooperativeBatch &&
               !cooperative_available) {
      throw std::runtime_error(
          stream != nullptr
              ? "BF12 cooperative batching requires the null/default stream"
              : "BF12 cooperative batching is unsupported: " +
                    capability.reason);
    }

    // Declare the cooperative lease before the stream guard so unwinding
    // drains accepted asynchronous work before releasing full-residency
    // exclusion.  The guard is active before endpoint uploads: their source
    // vectors are local to this call and must outlive every accepted copy.
    std::unique_ptr<CooperativeControllerLease> cooperative_lease;
    DrainStreamOnException exception_drain{stream};

    const bool query_state_grew = rips_sssp_bf12::ensure_query_capacity(
        workspace, batch.descriptors.size());
    if (options.enable_telemetry) {
      std::size_t maximum_rounds = 0;
      for (const auto& descriptor : batch.descriptors) {
        maximum_rounds = std::max(
            maximum_rounds,
            static_cast<std::size_t>(std::max(0, descriptor.max_iterations)));
      }
      rips_sssp_bf12::ensure_telemetry_round_capacity(workspace,
                                                       maximum_rounds);
    }
    rips_sssp_bf12::validate_result_growth_reserve(
        workspace, initial_node_capacity, initial_edge_capacity,
        options.memory_safety_reserve_bytes);
    rips_sssp_bf12::ensure_result_capacity(
        workspace, initial_node_capacity, initial_edge_capacity);
    rips_sssp_bf12::upload_batch_inputs(workspace, batch);

    BellmanFord12BatchTelemetry local;
    local.enabled = options.enable_telemetry;
    local.requested_batch_size = options.requested_batch_size;
    local.selected_batch_size = decision.selected_batch_size;
    local.submitted_query_count = batch.descriptors.size();
    local.batch_count = 1;
    local.batch_fill_ratio = decision.selected_batch_size == 0
                                 ? 0.0
                                 : static_cast<double>(batch.descriptors.size()) /
                                       decision.selected_batch_size;
    local.controller_mode_used = mode;
    if (query_state_grew) ++local.synchronization_count;

    if (mode == BellmanFord12ControllerMode::HostBatch) {
      rips_sssp_bf12::run_host_controller(
          graph->device.view, workspace, batch, options.enable_telemetry,
          local);
    } else {
      cooperative_lease = std::make_unique<CooperativeControllerLease>();
      const bool defensive_reset_required = workspace.needs_dense_reset;
      workspace.needs_dense_reset = true;
      if (defensive_reset_required) {
        rips_sssp_bf12::defensive_dense_reset(workspace);
      }
      rips_sssp_bf12::launch_cooperative_controller(
          graph->device.view, workspace, batch, options.enable_telemetry,
          capability.legally_resident_blocks);
      ++local.cooperative_controller_launch_count;
    }

    HipEventTimer summary_timer(options.enable_telemetry);
    summary_timer.begin(stream);
    hipLaunchKernelGGL(
        rips_sssp_bf12::summarize_target_paths_kernel,
        rips_sssp_bf12::grid_for_items(batch.targets.size()),
        dim3(rips_sssp_bf12::kBlockSize), 0, stream, workspace.best_state,
        workspace.source_mask, graph->rows, graph->nnz,
        graph->device.view.destinations, graph->device.view.rowptr,
        workspace.target_states, batch.targets.size(),
        workspace.target_summaries);
    check_hip(hipGetLastError(), "summarize BF12 target paths");
    summary_timer.end(stream);

    bool overflow_retried = false;
    HipEventTimer reconstruction_timer(options.enable_telemetry);
    reconstruction_timer.begin(stream);
    rips_sssp_bf12::launch_reconstruction(graph->device.view, workspace,
                                           batch.targets.size());
    reconstruction_timer.end(stream);
    local.device_to_host_bytes += rips_sssp_bf12::enqueue_result_copies(
        workspace, batch.descriptors.size(), batch.targets.size(),
        options.enable_telemetry);
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF12 batched result payload");
    ++local.synchronization_count;
    rips_sssp_bf12::unpack_result_header(
        workspace, batch.descriptors.size(), batch.targets.size());
    local.target_summary_nanoseconds += summary_timer.elapsed_nanoseconds();
    local.reconstruction_nanoseconds +=
        reconstruction_timer.elapsed_nanoseconds();
    rips_sssp_bf12::throw_controller_error(
        workspace.host_batch_control->error_status);
    if (workspace.host_result_header->invalid_path != 0) {
      throw std::runtime_error(
          "BF12 predecessor validation failed during reconstruction");
    }
    if (workspace.host_result_header->arena_overflow != 0) {
      overflow_retried = true;
      ++local.arena_overflow_retry_count;
      if (workspace.host_result_header->total_nodes >
              static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max()) ||
          workspace.host_result_header->total_edges >
              static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("BF12 result arena usage exceeds size_t");
      }
      rips_sssp_bf12::validate_result_growth_reserve(
          workspace,
          static_cast<std::size_t>(workspace.host_result_header->total_nodes),
          static_cast<std::size_t>(workspace.host_result_header->total_edges),
          options.memory_safety_reserve_bytes);
      rips_sssp_bf12::ensure_result_capacity(
          workspace,
          static_cast<std::size_t>(workspace.host_result_header->total_nodes),
          static_cast<std::size_t>(workspace.host_result_header->total_edges));
      HipEventTimer retry_reconstruction_timer(options.enable_telemetry);
      retry_reconstruction_timer.begin(stream);
      rips_sssp_bf12::launch_reconstruction(graph->device.view, workspace,
                                             batch.targets.size());
      retry_reconstruction_timer.end(stream);
      local.device_to_host_bytes += rips_sssp_bf12::enqueue_result_copies(
          workspace, batch.descriptors.size(), batch.targets.size(),
          options.enable_telemetry);
      check_hip(hipStreamSynchronize(stream),
                "synchronize grown BF12 result arena");
      ++local.synchronization_count;
      rips_sssp_bf12::unpack_result_header(
          workspace, batch.descriptors.size(), batch.targets.size());
      local.reconstruction_nanoseconds +=
          retry_reconstruction_timer.elapsed_nanoseconds();
      if (workspace.host_result_header->arena_overflow != 0) {
        throw std::runtime_error("BF12 result arena growth did not resolve overflow");
      }
      if (workspace.host_result_header->invalid_path != 0) {
        throw std::runtime_error(
            "BF12 predecessor validation failed after arena growth");
      }
    }
    local.result_node_capacity = workspace.result_node_capacity;
    local.result_edge_capacity = workspace.result_edge_capacity;
    local.result_nodes_used = static_cast<std::size_t>(
        workspace.host_result_header->total_nodes);
    local.result_edges_used = static_cast<std::size_t>(
        workspace.host_result_header->total_edges);
    if (mode == BellmanFord12ControllerMode::CooperativeBatch) {
      local.global_traversal_rounds =
          workspace.host_batch_control->global_rounds;
      const std::size_t recorded_rounds = std::min<std::size_t>(
          workspace.host_batch_control->global_rounds,
          workspace.telemetry_round_capacity);
      if (options.enable_telemetry) {
        local.active_queries_per_round.assign(
            workspace.host_telemetry_active_queries,
            workspace.host_telemetry_active_queries + recorded_rounds);
        local.combined_frontier_size_per_round.assign(
            workspace.host_telemetry_frontier_sizes,
            workspace.host_telemetry_frontier_sizes + recorded_rounds);
      }
    }
    if (options.enable_telemetry) {
      local.edges_examined = workspace.host_telemetry->edges_examined;
      local.successful_relaxations =
          workspace.host_telemetry->successful_relaxations;
    }
    local.touched_nodes_per_query.reserve(batch.descriptors.size());
    for (std::size_t query = 0; query < batch.descriptors.size(); ++query) {
      const std::uint64_t touched =
          workspace.host_query_controls[query].touched_count;
      local.touched_nodes_per_query.push_back(touched);
      local.maximum_touched_state_density = std::max(
          local.maximum_touched_state_density,
          static_cast<double>(touched) / static_cast<double>(graph->rows));
    }

    BellmanFord12BatchResult result = rips_sssp_bf12::build_host_results(
        batch, workspace, options.enable_bounding_boxes, overflow_retried);
    workspace.needs_dense_reset = false;
    local.total_batch_wall_nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             batch_begin)
            .count();
    merge_telemetry(local);
    return result;
  }

  BellmanFord12BatchResult run_with_retry(
      const std::vector<int>& flattened_sources,
      const std::vector<int>& flattened_targets,
      const std::vector<BellmanFord12QueryDescriptor>& descriptors,
      const BellmanFord12BatchOptions& options) {
    BellmanFord12BatchResult result = run_once(
        flattened_sources, flattened_targets, descriptors, options);
    if (!options.enable_bounding_boxes ||
        !options.enable_unbounded_retry) {
      return result;
    }
    std::vector<std::size_t> retry_positions;
    for (std::size_t query = 0; query < result.query_statuses.size(); ++query) {
      if (result.query_statuses[query].bounded_search_miss) {
        retry_positions.push_back(query);
      }
    }
    if (retry_positions.empty()) return result;

    std::vector<int> retry_sources;
    std::vector<int> retry_targets;
    std::vector<BellmanFord12QueryDescriptor> retry_descriptors;
    retry_descriptors.reserve(retry_positions.size());
    for (const std::size_t position : retry_positions) {
      const BellmanFord12QueryDescriptor& input = descriptors[position];
      BellmanFord12QueryDescriptor retry = input;
      retry.source_begin = static_cast<std::uint32_t>(retry_sources.size());
      retry.target_begin = static_cast<std::uint32_t>(retry_targets.size());
      retry.bounds.enabled = false;
      retry_sources.insert(
          retry_sources.end(),
          flattened_sources.begin() + input.source_begin,
          flattened_sources.begin() + input.source_begin + input.source_count);
      retry_targets.insert(
          retry_targets.end(),
          flattened_targets.begin() + input.target_begin,
          flattened_targets.begin() + input.target_begin + input.target_count);
      retry_descriptors.push_back(retry);
    }
    BellmanFord12BatchOptions retry_options = options;
    retry_options.enable_bounding_boxes = false;
    retry_options.enable_unbounded_retry = false;
    // The retry contains only misses, so an explicit original maximum must not
    // make the safe policy expect absent queries.
    if (retry_options.requested_batch_size != 0) {
      retry_options.requested_batch_size = std::min(
          retry_options.requested_batch_size, retry_descriptors.size());
    }
    BellmanFord12BatchResult retried = run_once(
        retry_sources, retry_targets, retry_descriptors, retry_options);
    aggregate_telemetry.bounded_query_retry_count += retry_positions.size();
    for (std::size_t item = 0; item < retry_positions.size(); ++item) {
      const std::size_t destination = retry_positions[item];
      const int first_iterations = result.query_statuses[destination].iterations;
      const int retry_iterations = retried.query_statuses[item].iterations;
      const int combined =
          retry_iterations > std::numeric_limits<int>::max() - first_iterations
              ? std::numeric_limits<int>::max()
              : first_iterations + retry_iterations;
      retried.query_statuses[item].iterations = combined;
      retried.query_results[item].iterations_used = combined;
      result.query_results[destination] = std::move(retried.query_results[item]);
      result.query_statuses[destination] = retried.query_statuses[item];
      result.original_query_indices[destination] =
          retried.original_query_indices[item];
    }
    return result;
  }
};

BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency, rips_sssp_bf12::sidecar_view(sidecars, adjacency),
          stream)) {}

BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    const HostCsrF32& adjacency,
    const BellmanFord12NodeSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency, rips_sssp_bf12::sidecar_view(sidecars),
          stream)) {}

BellmanFord12CsrGraph::~BellmanFord12CsrGraph() = default;
BellmanFord12CsrGraph::BellmanFord12CsrGraph(
    BellmanFord12CsrGraph&&) noexcept = default;
BellmanFord12CsrGraph& BellmanFord12CsrGraph::operator=(
    BellmanFord12CsrGraph&&) noexcept = default;

std::size_t BellmanFord12CsrGraph::vertex_count() const noexcept {
  return impl_ ? static_cast<std::size_t>(impl_->rows) : 0;
}

std::size_t BellmanFord12CsrGraph::edge_count() const noexcept {
  return impl_ ? static_cast<std::size_t>(impl_->nnz) : 0;
}

std::size_t BellmanFord12CsrGraph::device_memory_bytes() const noexcept {
  return impl_ ? impl_->device_bytes : 0;
}

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    std::size_t maximum_batch_size,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord12CsrGraph::Impl>(
              adjacency, rips_sssp_bf12::sidecar_view(sidecars, adjacency),
              stream),
          maximum_batch_size, stream)) {}

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord12NodeSidecars& sidecars,
    std::size_t maximum_batch_size,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord12CsrGraph::Impl>(
              adjacency, rips_sssp_bf12::sidecar_view(sidecars),
              stream),
          maximum_batch_size, stream)) {}

BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    std::shared_ptr<const BellmanFord12CsrGraph> adjacency,
    std::size_t maximum_batch_size,
    hipStream_t stream) {
  if (!adjacency || !adjacency->impl_) {
    throw std::invalid_argument("BF12 shared graph is null or moved-from");
  }
  impl_ = std::make_unique<Impl>(adjacency->impl_, maximum_batch_size, stream);
}

BellmanFord12CsrWorkspace::~BellmanFord12CsrWorkspace() = default;
BellmanFord12CsrWorkspace::BellmanFord12CsrWorkspace(
    BellmanFord12CsrWorkspace&&) noexcept = default;
BellmanFord12CsrWorkspace& BellmanFord12CsrWorkspace::operator=(
    BellmanFord12CsrWorkspace&&) noexcept = default;

void BellmanFord12CsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  impl_->upload_costs(vertex_costs);
}

BellmanFord12BatchResult BellmanFord12CsrWorkspace::run_batch(
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options,
    hipStream_t stream) {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  return impl_->run_with_retry(flattened_sources, flattened_targets,
                               descriptors, options);
}

BellmanFord12BatchResult BellmanFord12CsrWorkspace::run_batch(
    const std::vector<float>& dynamic_vertex_costs,
    const std::vector<int>& flattened_sources,
    const std::vector<int>& flattened_targets,
    const std::vector<BellmanFord12QueryDescriptor>& descriptors,
    const BellmanFord12BatchOptions& options,
    hipStream_t stream) {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  impl_->upload_costs(dynamic_vertex_costs);
  return impl_->run_with_retry(flattened_sources, flattened_targets,
                               descriptors, options);
}

BellmanFord12CooperativeCapability
BellmanFord12CsrWorkspace::cooperative_capability(
    bool require_occupancy) const {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  impl_->require_stream(impl_->stream);
  return rips_sssp_bf12::query_cooperative_capability(
      impl_->workspace, require_occupancy);
}

bool BellmanFord12CsrWorkspace::cooperative_execution_supported() const {
  const BellmanFord12CooperativeCapability capability =
      cooperative_capability();
  return capability.device_supports_cooperative_launch &&
         capability.legally_resident_blocks > 0 && impl_->stream == nullptr;
}

BellmanFord12MemoryEstimate BellmanFord12CsrWorkspace::estimate_memory(
    std::size_t batch_size,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity) const {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  return estimate_bellman_ford12_memory(
      static_cast<std::size_t>(impl_->graph->rows),
      static_cast<std::size_t>(impl_->graph->nnz), batch_size, 0, 0,
      result_node_capacity, result_edge_capacity, false);
}

std::size_t BellmanFord12CsrWorkspace::maximum_batch_size() const noexcept {
  if (!impl_) return 0;
  if (impl_->maximum_batch != 0) return impl_->maximum_batch;
  return bf12_worker_policy::maximum_uint32_composite_batch_size(
      static_cast<std::size_t>(impl_->graph->rows));
}

BellmanFord12BatchTelemetry BellmanFord12CsrWorkspace::telemetry() const {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  return impl_->aggregate_telemetry;
}

void BellmanFord12CsrWorkspace::reset_telemetry() {
  if (!impl_) throw std::runtime_error("BF12 workspace is moved-from");
  std::lock_guard<std::mutex> lock(impl_->operation_mutex);
  impl_->aggregate_telemetry = {};
  impl_->aggregate_selected_query_slots = 0;
}

BellmanFord12MemoryEstimate estimate_bellman_ford12_memory(
    std::size_t vertex_count,
    std::size_t edge_count,
    std::size_t batch_size,
    std::size_t source_count,
    std::size_t target_count,
    std::size_t result_node_capacity,
    std::size_t result_edge_capacity,
    bool telemetry_enabled) {
  BellmanFord12MemoryEstimate estimate;
  estimate.graph_bytes = bf12_worker_policy::shared_graph_device_bytes(
      vertex_count, edge_count);
  estimate.shared_dynamic_cost_bytes =
      bf12_worker_policy::shared_dynamic_cost_device_bytes(vertex_count);
  estimate.fixed_query_state_bytes =
      bf12_worker_policy::query_vertex_state_device_bytes(batch_size,
                                                           vertex_count);
  estimate.result_arena_bytes =
      bf12_worker_policy::result_device_bytes_estimate(
          batch_size, target_count, result_node_capacity,
          result_edge_capacity);
  BellmanFord12WorkerPolicyInputs inputs;
  inputs.vertex_count = vertex_count;
  inputs.edge_count = edge_count;
  inputs.source_count = source_count;
  inputs.target_count = target_count;
  inputs.telemetry_enabled = telemetry_enabled;
  inputs.graph_already_resident = true;
  const std::size_t fixed_with_controls =
      bf12_worker_policy::fixed_device_bytes_estimate(inputs, batch_size);
  const std::size_t fixed_base = require_add(
      estimate.shared_dynamic_cost_bytes, estimate.fixed_query_state_bytes,
      "BF12 memory estimate overflow");
  estimate.control_bytes = fixed_with_controls >= fixed_base
                               ? fixed_with_controls - fixed_base
                               : 0;
  estimate.total_bytes = require_add(
      require_add(
          require_add(estimate.graph_bytes,
                      estimate.shared_dynamic_cost_bytes,
                      "BF12 memory estimate overflow"),
          estimate.fixed_query_state_bytes,
          "BF12 memory estimate overflow"),
      require_add(estimate.result_arena_bytes, estimate.control_bytes,
                  "BF12 memory estimate overflow"),
      "BF12 memory estimate overflow");
  return estimate;
}

const char* bellman_ford12_controller_mode_name(
    BellmanFord12ControllerMode mode) noexcept {
  switch (mode) {
    case BellmanFord12ControllerMode::Auto:
      return "auto";
    case BellmanFord12ControllerMode::HostBatch:
      return "host-batch";
    case BellmanFord12ControllerMode::CooperativeBatch:
      return "cooperative-batch";
  }
  return "unknown";
}

#ifndef BF12_NO_MAIN
int main(int, char**) {
  std::cerr
      << "bf12 is a PathFinder library backend; build it with BF12_NO_MAIN "
         "and use BellmanFord12CsrWorkspace.\n";
  return 2;
}
#endif
