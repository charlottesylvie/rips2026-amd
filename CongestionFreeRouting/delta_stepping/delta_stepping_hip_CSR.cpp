#include "delta_stepping_hip_CSR.hpp"

#include "../profiling/roctx_ranges.hpp"

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
#include <unordered_set>
#include <utility>
#include <vector>

namespace ds_delta_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
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
    release();
    if (count != 0) {
      T* candidate = nullptr;
      DS_DELTA_HIP_CHECK(
          hipMalloc(reinterpret_cast<void**>(&candidate), count * sizeof(T)));
      ptr_ = candidate;
      count_ = count;
    }
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
                        count * sizeof(T),
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

struct DeviceCsrOwner {
  DeviceBuffer<Offset> rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  DeviceBuffer<std::uint32_t> edge_source;
  // Kept separate from edge_source.get(): an eligible empty graph has a
  // deliberately null zero-length map.
  bool edge_source_available = false;
  minplus_sparse::DeviceCsrF32 view{};

  DeviceCsrOwner() = default;
  DeviceCsrOwner(Offset rows, Offset cols, Offset nnz)
      : rowptr(static_cast<std::size_t>(rows) + 1),
        colind(static_cast<std::size_t>(nnz)),
        values(static_cast<std::size_t>(nnz)) {
    view.rows = rows;
    view.cols = cols;
    view.nnz = nnz;
    view.rowptr = rowptr.get();
    view.colind = colind.get();
    view.values = values.get();
  }
};

struct DeltaSteppingScratch {
  Offset rows = 0;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<int> target_settled;
  DeviceBuffer<float> dist;
  DeviceBuffer<int> in_current;
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
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  PinnedHostBuffer<int> host_scalar;
  PinnedHostBuffer<int> host_unit_status;
  bool unit_initialized = false;
  bool generic_initialized = false;
  bool parent_key_initialized = false;
  bool legacy_predecessors_initialized = false;

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

  void ensure_parent_key_storage() {
    const std::size_t vertex_count = static_cast<std::size_t>(rows);
    if (parent_key.size() < vertex_count) {
      parent_key.reset(vertex_count);
      parent_key_initialized = false;
    }
  }

  void ensure_source_capacity(std::size_t source_count) {
    if (sources.size() < source_count) {
      sources.reset(grown_capacity(sources.size(), source_count));
    }
  }

  void ensure_telemetry_storage() {
    if (telemetry_counters.size() < kTelemetryCounterCount) {
      telemetry_counters.reset(kTelemetryCounterCount);
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    const std::size_t capacity =
        grown_capacity(targets.size(), target_count);
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
    if (target_node_offsets.size() < target_count + 1) {
      target_node_offsets.reset(capacity + 1);
    }
    if (target_edge_offsets.size() < target_count + 1) {
      target_edge_offsets.reset(capacity + 1);
    }
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count) {
    if (compact_path_nodes.size() < node_count) {
      compact_path_nodes.reset(
          grown_capacity(compact_path_nodes.size(), node_count));
    }
    if (compact_path_edges.size() < edge_count) {
      compact_path_edges.reset(
          grown_capacity(compact_path_edges.size(), edge_count));
    }
  }

  void release_parent_and_path_storage() {
    pred_node.reset(0);
    pred_edge.reset(0);
    parent_key.reset(0);
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
    unit_initialized = false;
    parent_key_initialized = false;
    legacy_predecessors_initialized = false;
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
  if (g.rowptr.size() != rows + 1) throw std::invalid_argument("rowptr.size() must equal rows + 1");
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

inline void validate_device_csr_shape(const minplus_sparse::DeviceCsrF32& g,
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

inline void validate_device_csr_shape(const minplus_sparse::DeviceCsrF32& g,
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

struct AtomicMinFloatResult {
  float old_value;
  unsigned int cas_retries;
};

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

__global__ void build_edge_source_kernel(Offset rows,
                                         const Offset* rowptr,
                                         std::uint32_t* edge_source) {
  for (Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<Offset>(blockDim.x) * gridDim.x) {
    for (Offset edge = rowptr[row]; edge < rowptr[row + 1]; ++edge) {
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

__global__ void validate_device_csr_kernel(Offset rows,
                                           Offset cols,
                                           Offset nnz,
                                           const Offset* rowptr,
                                           const Index* colind,
                                           const float* values,
                                           int* invalid) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (rowptr[0] != 0 || rowptr[rows] != nnz) atomicExch(invalid, 1);
  }
  for (Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows;
       row += static_cast<Offset>(blockDim.x) * gridDim.x) {
    const Offset begin = rowptr[row];
    const Offset end = rowptr[row + 1];
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
                                                int* in_current,
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
                                                int* in_current,
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
    in_current[source] = 1;
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
    atomic_store_unit_status(status + kUnitStatusBucketRounds, 1);
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

template <bool CollectTelemetry>
__device__ inline void expand_unit_frontier_range(
    int frontier_begin,
    int frontier_end,
    int next_depth,
    const Offset* out_rowptr,
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
    for (Offset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
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

template <bool CollectTelemetry>
__global__ void expand_unit_frontier_kernel(const Offset* out_rowptr,
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
  expand_unit_frontier_range<CollectTelemetry>(
      controller[1], controller[2], controller[3] + 1, out_rowptr,
      out_colind, dist, pred_node, pred_edge, frontier_queue,
      status + kUnitStatusQueueTail, status + kUnitStatusFoundCount,
      target_multiplicity, telemetry_counters);
}

template <bool CollectTelemetry>
__global__ void expand_unit_frontier_host_controlled_kernel(
    int frontier_begin,
    int frontier_end,
    int next_depth,
    const Offset* out_rowptr,
    const Index* out_colind,
    float* dist,
    int* pred_node,
    Offset* pred_edge,
    int* frontier_queue,
    int* queue_tail,
    int* found_count,
    const int* target_multiplicity,
    unsigned long long* telemetry_counters) {
  expand_unit_frontier_range<CollectTelemetry>(
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

__global__ void materialize_predecessors_kernel(
    const int* touched_vertices,
    int touched_count,
    const Offset* rowptr,
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

    for (Offset edge = rowptr[u]; edge < rowptr[u + 1]; ++edge) {
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
                                                 float* target_distances,
                                                 int* path_lengths,
                                                 int* path_sources,
                                                 int* path_status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const float target_distance = dist[targets[i]];
    target_distances[i] = target_distance;
    path_sources[i] = -1;
    if (!finite_float(target_distance)) {
      path_lengths[i] = 0;
      path_status[i] = 0;
      continue;
    }
    // In the unit specialization, distance is exactly the BFS edge depth.
    path_lengths[i] = static_cast<int>(target_distance) + 1;
    path_status[i] = 1;
  }
}

__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const Offset* rowptr,
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
      if (edge < rowptr[pred] || edge >= rowptr[pred + 1] ||
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

__device__ inline bool decode_tight_edge_parent(
    int current,
    Offset rows,
    Offset nnz,
    const Offset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const float* dist,
    unsigned long long key,
    Offset* edge_out,
    int* predecessor_out) {
  const float current_distance = dist[current];
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
  if (edge < rowptr[predecessor] || edge >= rowptr[predecessor + 1]) {
    return false;
  }

  const float predecessor_distance = dist[predecessor];
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

__global__ void measure_edge_parent_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    Offset nnz,
    const Offset* rowptr,
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
      const unsigned long long key = parent_key[current];
      if (key == kNoParentKey) {
        path_lengths[i] = length;
        path_sources[i] = current;
        path_status[i] = 1;
        break;
      }

      Offset edge = 0;
      int predecessor = -1;
      if (!decode_tight_edge_parent(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        break;
      }
      current = predecessor;
      ++length;
    }
  }
}

__global__ void fill_edge_parent_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    Offset nnz,
    const Offset* rowptr,
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
      const unsigned long long key = parent_key[current];
      Offset edge = 0;
      int predecessor = -1;
      if (!decode_tight_edge_parent(
              current, rows, nnz, rowptr, colind, values, vertex_costs,
              edge_source, dist, key, &edge, &predecessor)) {
        path_valid = false;
        break;
      }
      path_edges[edge_begin + j - 1] = edge;
      current = predecessor;
      path_nodes[node_begin + j - 1] = current;
    }
    if (path_valid && parent_key[current] == kNoParentKey) {
      path_sources[i] = current;
    } else {
      path_status[i] = 0;
      path_sources[i] = -1;
    }
  }
}

template <bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectHeavy,
          bool AllEdgesLight,
          bool CollectTelemetry>
__global__ void relax_light_edges_kernel(const int* frontier,
                                         const int* frontier_count_ptr,
                                         int current_bucket,
                                         float delta,
                                         const Offset* out_rowptr,
                                         const Index* out_colind,
                                         const float* out_values,
                                         const float* vertex_costs,
                                         float* dist,
                                         unsigned long long* parent_key,
                                         int* in_current,
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
    // The in_current flags for this whole frontier are cleared by a separate
    // kernel before relaxation starts.  Do not clear them here: doing so races
    // with other blocks that may need to re-enqueue this vertex after a same-
    // bucket distance decrease.
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
      for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryLightEdgeVisits];
        }
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            HasVertexCosts ? w * vertex_costs[v] : w;
        const float candidate = du + effective_w;
        int candidate_bucket = kNoBucket;
        // Float addition can place a nominally heavy edge in the current
        // bucket (and all very large distances share the terminal bucket).
        // Such work belongs to light closure or its descendants can be lost.
        bool light = true;
        if constexpr (!AllEdgesLight) {
          candidate_bucket = bucket_index(candidate, delta);
          light = effective_w <= delta ||
                  candidate_bucket == current_bucket;
        }
        const float nd = light ? candidate : INFINITY;
        AtomicMinFloatResult atomic_result{INFINITY, 0};
        if (light) {
          atomic_result =
              atomic_min_float_nonnegative<CollectTelemetry>(&dist[v], nd);
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryDistanceAtomicAttempts];
            telemetry[kTelemetryDistanceCasRetries] +=
                atomic_result.cas_retries;
          }
        }
        const float old = atomic_result.old_value;
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
            append_current = atomicCAS(&in_current[v], 0, 1) == 0;
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

template <bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectTelemetry>
__global__ void relax_heavy_edges_kernel(const int* heavy_vertices,
                                         const int* heavy_count_ptr,
                                         int current_bucket,
                                         float delta,
                                         const Offset* out_rowptr,
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
      for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
        if constexpr (CollectTelemetry) {
          ++telemetry[kTelemetryHeavyEdgeVisits];
        }
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            HasVertexCosts ? w * vertex_costs[v] : w;
        const float candidate = du + effective_w;
        const int candidate_bucket = bucket_index(candidate, delta);
        // Same-bucket nominally-heavy edges were already processed by the
        // light-closure kernel. Only future-bucket candidates belong here.
        const bool heavy = effective_w > delta &&
                           candidate_bucket > current_bucket &&
                           candidate_bucket < kNoBucket;
        const float nd = heavy ? candidate : INFINITY;
        AtomicMinFloatResult atomic_result{INFINITY, 0};
        if (heavy) {
          atomic_result =
              atomic_min_float_nonnegative<CollectTelemetry>(&dist[v], nd);
          if constexpr (CollectTelemetry) {
            ++telemetry[kTelemetryDistanceAtomicAttempts];
            telemetry[kTelemetryDistanceCasRetries] +=
                atomic_result.cas_retries;
          }
        }
        const float old = atomic_result.old_value;
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

template <bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectHeavy,
          bool AllEdgesLight,
          bool CollectTelemetry>
void launch_relax_light_edges(
    const minplus_sparse::DeviceCsrF32& graph,
    DeltaSteppingScratch& scratch,
    const float* vertex_costs,
    const int* current_queue,
    const int* current_count,
    int launch_blocks,
    int current_bucket,
    float delta,
    int* next_queue,
    int* next_count,
    int* pending_queue,
    hipStream_t stream) {
  relax_light_edges_kernel<TrackParents, UseEdgeParent, HasVertexCosts,
                           CollectHeavy, AllEdgesLight, CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(),
          scratch.in_current.get(), scratch.in_pending.get(),
          scratch.in_heavy.get(), scratch.touched_queue.get(),
          scratch.touched_count.get(), next_queue, next_count,
          pending_queue, scratch.pending_count.get(), scratch.heavy_queue.get(),
          scratch.heavy_count.get(),
          CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

template <bool TrackParents,
          bool UseEdgeParent,
          bool HasVertexCosts,
          bool CollectTelemetry>
void launch_relax_heavy_edges(
    const minplus_sparse::DeviceCsrF32& graph,
    DeltaSteppingScratch& scratch,
    const float* vertex_costs,
    int launch_blocks,
    int current_bucket,
    float delta,
    int* pending_queue,
    hipStream_t stream) {
  relax_heavy_edges_kernel<TrackParents, UseEdgeParent, HasVertexCosts,
                           CollectTelemetry>
      <<<launch_blocks, kBlockSize, 0, stream>>>(
          scratch.heavy_queue.get(), scratch.heavy_count.get(),
          current_bucket, delta,
          graph.rowptr, graph.colind, graph.values, vertex_costs,
          scratch.dist.get(), scratch.parent_key.get(), scratch.in_pending.get(),
          scratch.touched_queue.get(), scratch.touched_count.get(),
          pending_queue, scratch.pending_count.get(), scratch.in_heavy.get(),
          CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

__global__ void clear_flags_from_queue_kernel(const int* vertices,
                                              const int* count_ptr,
                                              int* flags) {
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = atomic_load_counter(count_ptr);
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count;
       i += blockDim.x * gridDim.x) {
    flags[vertices[i]] = 0;
  }
}

__global__ void reset_touched_vertices_kernel(const int* touched_queue,
                                              int touched_count,
                                              float inf,
                                              float* dist,
                                              int* in_current,
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
    in_current[v] = 0;
    in_pending[v] = 0;
    in_heavy[v] = 0;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
    parent_key[v] = kNoParentKey;
  }
}

__global__ void reset_distance_only_touched_vertices_kernel(
    const int* touched_queue,
    int touched_count,
    float inf,
    float* dist,
    int* in_current,
    int* in_pending,
    int* in_heavy) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    in_current[v] = 0;
    in_pending[v] = 0;
    in_heavy[v] = 0;
  }
}

template <bool ResetHeavyMembership>
__global__ void reset_compact_parent_touched_vertices_kernel(
    const int* touched_queue,
    int touched_count,
    float inf,
    float* dist,
    int* in_current,
    int* in_pending,
    int* in_heavy,
    unsigned long long* parent_key) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    parent_key[v] = kNoParentKey;
    in_current[v] = 0;
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

template <bool CollectTelemetry>
__global__ void compact_pending_to_current_bucket_kernel(const int* pending_in,
                                                         const int* pending_count_ptr,
                                                         int selected_bucket,
                                                         float delta,
                                                         const float* dist,
                                                         int* in_pending,
                                                         int* in_current,
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
      append_current = atomicCAS(&in_current[v], 0, 1) == 0;
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
                                       bool build_compact_edge_source) {
  DeviceCsrOwner d(h.rows, h.cols, h.nnz);
  const std::size_t rows = checked_size(h.rows, "rows");
  const std::size_t nnz = checked_size(h.nnz, "nnz");
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(d.rowptr.get(), h.rowptr.data(),
                                    (rows + 1) * sizeof(Offset), hipMemcpyHostToDevice, stream));
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
        build_edge_source_kernel<<<grid_for_items(h.rows), kBlockSize, 0, stream>>>(
            h.rows, d.rowptr.get(), d.edge_source.get());
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
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(d.colind.get(), h.colind.data(),
                                      nnz * sizeof(Index), hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(d.values.get(), h.values.data(),
                                      nnz * sizeof(float), hipMemcpyHostToDevice, stream));
  }
  // The host vectors may be temporary, and PathFinder worker streams consume
  // the graph immediately after construction. Publish only a completed upload.
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return d;
}

void validate_device_csr_contents(const minplus_sparse::DeviceCsrF32& g, hipStream_t stream) {
  DeviceBuffer<int> d_invalid(1);
  DS_DELTA_HIP_CHECK(hipMemsetAsync(d_invalid.get(), 0, sizeof(int), stream));
  synchronize_explicit_stream(stream);
  validate_device_csr_kernel<<<grid_for_items(g.rows), kBlockSize, 0, stream>>>(
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
      kTelemetryCounterCount * sizeof(unsigned long long),
      stream));
}

void copy_device_telemetry_to_host(DeltaSteppingScratch& scratch,
                                   DeltaSteppingCsrTelemetry& telemetry,
                                   hipStream_t stream) {
  std::array<unsigned long long, kTelemetryCounterCount> counters{};
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(
      counters.data(),
      scratch.telemetry_counters.get(),
      counters.size() * sizeof(unsigned long long),
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
    reset_touched_vertices_kernel<<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
        scratch.touched_queue.get(), touched_count, inf, scratch.dist.get(),
        scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
        scratch.pred_node.get(), scratch.pred_edge.get(), scratch.parent_key.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

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
    reset_distance_only_touched_vertices_kernel
        <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
            scratch.touched_queue.get(), touched_count, inf,
            scratch.dist.get(), scratch.in_current.get(),
            scratch.in_pending.get(), scratch.in_heavy.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

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
      reset_compact_parent_touched_vertices_kernel<true>
          <<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
              scratch.touched_queue.get(), touched_count, inf,
              scratch.dist.get(), scratch.in_current.get(),
              scratch.in_pending.get(), scratch.in_heavy.get(),
              scratch.parent_key.get());
    } else {
      reset_compact_parent_touched_vertices_kernel<false>
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
                                    static_cast<std::size_t>(n) * sizeof(float),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return h;
}

float copy_dist_value_to_host(const float* d_dist, int vertex, hipStream_t stream) {
  return copy_scalar_to_host(d_dist + vertex, stream);
}

void copy_predecessors_to_result(DeltaSteppingCsrResult& result,
                                 const minplus_sparse::DeviceCsrF32& graph,
                                 int* d_pred_node,
                                 Offset* d_pred_edge,
                                 hipStream_t stream) {
  result.pred_node.resize(static_cast<std::size_t>(graph.rows));
  result.pred_edge.resize(static_cast<std::size_t>(graph.rows));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_node.data(), d_pred_node,
                                    static_cast<std::size_t>(graph.rows) * sizeof(int),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_edge.data(), d_pred_edge,
                                    static_cast<std::size_t>(graph.rows) * sizeof(Offset),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
}

int materialize_predecessors_from_keys(
    const minplus_sparse::DeviceCsrF32& graph,
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
  materialize_predecessors_kernel
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

template <TargetPathParentMode ParentMode>
void extract_target_paths_to_result(
    DeltaSteppingCsrResult& result,
    DeltaSteppingScratch& scratch,
    const minplus_sparse::DeviceCsrF32& graph,
    const float* vertex_costs,
    const std::uint32_t* edge_source,
    const std::vector<int>& targets,
    hipStream_t stream,
    const int* target_settled = nullptr) {
  PATHFINDER_PROFILE_RANGE(
      ParentMode == TargetPathParentMode::kCompactEdge
          ? "delta_step.compact_edge_path_extraction"
          : "delta_step.legacy_path_extraction");
  const int target_count = static_cast<int>(targets.size());
  if constexpr (ParentMode == TargetPathParentMode::kUnitWeight) {
    measure_unit_target_paths_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.dist.get(),
            scratch.target_distances.get(), scratch.target_path_lengths.get(),
            scratch.target_sources.get(), scratch.target_path_status.get());
  } else if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
    measure_edge_parent_target_paths_kernel
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

  result.target_distances.resize(targets.size());
  result.target_sources.resize(targets.size());
  std::vector<int> path_lengths(targets.size());
  std::vector<int> path_status(targets.size());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_distances.data(),
                                    scratch.target_distances.get(),
                                    targets.size() * sizeof(float),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_lengths.data(),
                                    scratch.target_path_lengths.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_status.data(),
                                    scratch.target_path_status.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));

  result.target_path_offsets.assign(targets.size() + 1, 0);
  result.target_edge_offsets.assign(targets.size() + 1, 0);
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
    total_nodes += static_cast<std::size_t>(path_lengths[i]);
    total_edges += static_cast<std::size_t>(path_lengths[i] - 1);
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
                                      (targets.size() + 1) * sizeof(int),
                                      hipMemcpyHostToDevice,
                                      stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.target_edge_offsets.get(),
                                      result.target_edge_offsets.data(),
                                      (targets.size() + 1) * sizeof(int),
                                      hipMemcpyHostToDevice,
                                      stream));
    synchronize_explicit_stream(stream);

    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    if constexpr (ParentMode == TargetPathParentMode::kCompactEdge) {
      fill_edge_parent_target_paths_kernel
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
      fill_target_paths_kernel
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
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_path_nodes.data(),
                                      scratch.compact_path_nodes.get(),
                                      total_nodes * sizeof(int),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  if (total_edges != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_path_edges.data(),
                                      scratch.compact_path_edges.get(),
                                      total_edges * sizeof(Offset),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_sources.data(),
                                    scratch.target_sources.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  if (total_nodes != 0) {
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(path_status.data(),
                                      scratch.target_path_status.get(),
                                      targets.size() * sizeof(int),
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

template <bool CollectTelemetry>
DeltaSteppingCsrResult run_unit_weight_specialization(
    const minplus_sparse::DeviceCsrF32& graph,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
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
  if (sources.size() == 1) {
    for (const int target : targets) {
      if (target == sources.front()) {
        ++initially_found;
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
    for (const int target : targets) {
      if (source_set.find(target) != source_set.end()) {
        ++initially_found;
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
  const float inf = std::numeric_limits<float>::infinity();
  if constexpr (CollectTelemetry) {
    prepare_device_telemetry(scratch, stream);
  }
  scratch.ensure_source_capacity(effective_sources->size());
  scratch.ensure_target_capacity(targets.size());
  initialize_unit_scratch_storage_once(scratch, graph.rows, inf, stream);

  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(),
                                    effective_sources->data(),
                                    effective_sources->size() * sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.targets.get(),
                                    targets.data(),
                                    targets.size() * sizeof(int),
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
          vertex_count,
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
  int bucket_rounds = initially_found >= target_count ? 0 : 1;
  std::uint64_t controller_round_trips = 0;
  DeltaSteppingCsrResult result;
  result.target = -1;
  const bool use_device_controller = stream == nullptr;

  while (current_count > 0 && found_count < target_count &&
         result.iterations_used < vertex_count) {
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
      expand_unit_frontier_host_controlled_kernel<CollectTelemetry>
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
                                        kUnitStatusCount * sizeof(int),
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
      const int rounds_to_enqueue = previous_depth == 0 ? 1 : 4;
      const int launch_blocks =
          rounds_to_enqueue == 1
              ? grid_for_frontier(current_count)
              : std::min(grid_for_items(graph.rows),
                         std::max(grid_for_frontier(current_count), 32));
      for (int round = 0; round < rounds_to_enqueue; ++round) {
        expand_unit_frontier_kernel<CollectTelemetry>
            <<<launch_blocks, kBlockSize, 0, stream>>>(
                graph.rowptr, graph.colind, scratch.dist.get(),
                scratch.pred_node.get(), scratch.pred_edge.get(),
                scratch.current_queue.get(), scratch.unit_status.get(),
                scratch.in_pending.get(),
                CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
        DS_DELTA_HIP_CHECK(hipGetLastError());
        advance_unit_frontier_kernel<<<1, 1, 0, stream>>>(
            scratch.unit_status.get(), delta, target_count, vertex_count);
        DS_DELTA_HIP_CHECK(hipGetLastError());
      }
      DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.host_unit_status.get(),
                                        scratch.unit_status.get(),
                                        kUnitStatusCount * sizeof(int),
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
          result.iterations_used < vertex_count;
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
  result.converged = !result.stopped_on_target && current_count == 0;
  try {
    clear_unit_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.in_pending.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    extract_target_paths_to_result<TargetPathParentMode::kUnitWeight>(
        result, scratch, graph, nullptr, nullptr, targets, stream);
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
                            int* d_min_bucket,
                            int* h_min_bucket,
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
  return copy_scalar_to_host(d_min_bucket, stream, h_min_bucket);
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

template <bool TrackParents, bool UseEdgeParent, bool CollectTelemetry>
DeltaSteppingCsrResult run_delta_stepping_impl(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::uint32_t* edge_source,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    bool skip_heavy_edges,
    float delta,
    int max_iters,
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
      if (is_effective_source(candidate)) {
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
      !use_target_set && target >= 0 && is_effective_source(target);

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
                                      static_cast<std::size_t>(target_count) * sizeof(int),
                                      hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        scratch.target_settled.get(), initial_target_settled.data(),
        static_cast<std::size_t>(target_count) * sizeof(int),
        hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(
        scratch.settled_target_count.get(), &initial_settled_target_count,
        sizeof(int), hipMemcpyHostToDevice, stream));
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(), effective_sources->data(),
                                    static_cast<std::size_t>(source_count) * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
  // The source kernel consumes the uploaded list and the state initialized or
  // reset above.  Complete both producers before launching it on an explicit
  // worker stream.
  synchronize_explicit_stream(stream);
  initialize_delta_sources_kernel<TrackParents && !UseEdgeParent>
      <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
          scratch.sources.get(), source_count, scratch.dist.get(),
          scratch.in_current.get(), scratch.current_queue.get(),
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
        copy_predecessors_to_result(result, d_adjacency,
                                    scratch.pred_node.get(),
                                    scratch.pred_edge.get(), stream);
      }
      result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
    } catch (...) {
      const std::exception_ptr materialization_exception =
          std::current_exception();
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      if constexpr (TrackParents) {
        reset_touched_vertices(scratch, inf, stream, source_count);
      } else {
        reset_distance_only_touched_vertices(
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
      reset_touched_vertices(scratch, inf, stream);
    } else {
      reset_distance_only_touched_vertices(scratch, inf, stream);
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
        reset_compact_parent_touched_vertices(
            scratch, inf, stream, touched_count, true);
      } else if constexpr (TrackParents) {
        reset_touched_vertices(scratch, inf, stream, touched_count);
      } else {
        reset_distance_only_touched_vertices(
            scratch, inf, stream, touched_count);
      }
      DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
      std::rethrow_exception(callback_exception);
    }
  };

  for (int iter = 0;
       iter < max_iters && !result.stopped_on_target;
       ++iter) {
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
        clear_flags_from_queue_kernel
            <<<launch_blocks, kBlockSize, 0, stream>>>(
                current_queue, current_count_device,
                scratch.in_current.get());
        DS_DELTA_HIP_CHECK(hipGetLastError());
        // next_count and the frontier membership flags are both consumed by
        // the relaxation kernel.  An explicit-stream completion boundary is
        // required before that dependent dispatch on the affected runtime.
        synchronize_explicit_stream(stream);

        const bool terminal_bucket = current_bucket == kNoBucket - 1;
        if (terminal_bucket && vertex_costs != nullptr) {
          launch_relax_light_edges<TrackParents, UseEdgeParent, true, false,
                                   true, CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              next_queue, next_count_device, pending_queue, stream);
        } else if (terminal_bucket || skip_heavy_edges) {
          launch_relax_light_edges<TrackParents, UseEdgeParent, false, false,
                                   true, CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              next_queue, next_count_device, pending_queue, stream);
        } else if (vertex_costs != nullptr) {
          launch_relax_light_edges<TrackParents, UseEdgeParent, true, true,
                                   false, CollectTelemetry>(
              d_adjacency, scratch, vertex_costs, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              next_queue, next_count_device, pending_queue, stream);
        } else {
          launch_relax_light_edges<TrackParents, UseEdgeParent, false, true,
                                   false, CollectTelemetry>(
              d_adjacency, scratch, nullptr, current_queue,
              current_count_device, launch_blocks, current_bucket, delta,
              next_queue, next_count_device, pending_queue, stream);
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

    if (!skip_heavy_edges && current_bucket != kNoBucket - 1) {
      ++heavy_edge_phases;
      if (vertex_costs != nullptr) {
        launch_relax_heavy_edges<TrackParents, UseEdgeParent, true,
                                 CollectTelemetry>(
            d_adjacency, scratch, vertex_costs, device_count_blocks,
            current_bucket, delta, pending_queue, stream);
      } else {
        launch_relax_heavy_edges<TrackParents, UseEdgeParent, false,
                                 CollectTelemetry>(
            d_adjacency, scratch, nullptr, device_count_blocks,
            current_bucket, delta, pending_queue, stream);
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
      const float target_distance = copy_dist_value_to_host(scratch.dist.get(), target, stream);
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
        scratch.min_pending_bucket.get(), scratch.host_scalar.get(), stream,
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
      result.converged = true;
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

    compact_pending_to_current_bucket_kernel<CollectTelemetry>
        <<<device_count_blocks, kBlockSize, 0, stream>>>(
        pending_queue, scratch.pending_count.get(), current_bucket, delta,
        scratch.dist.get(),
        scratch.in_pending.get(), scratch.in_current.get(), current_queue,
        scratch.current_count.get(), pending_scratch,
        scratch.new_pending_count.get(),
        CollectTelemetry ? scratch.telemetry_counters.get() : nullptr);
    DS_DELTA_HIP_CHECK(hipGetLastError());

    current_count = copy_scalar_to_host(
        scratch.current_count.get(), stream, scratch.host_scalar.get());
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
      extract_target_paths_to_result<TargetPathParentMode::kCompactEdge>(
          result, scratch, d_adjacency, vertex_costs, edge_source, *targets,
          stream, settled_target_filter);
      touched_count_for_reset = *scratch.host_scalar.get();
    } else {
      if (use_target_set || target >= 0) {
        touched_count_for_reset = materialize_predecessors_from_keys(
            d_adjacency, scratch, vertex_costs, stream);
      }
      if (use_target_set) {
        extract_target_paths_to_result<
            TargetPathParentMode::kLegacyPredecessor>(
            result, scratch, d_adjacency, vertex_costs, nullptr, *targets,
            stream, settled_target_filter);
      } else if (target >= 0) {
        copy_predecessors_to_result(result, d_adjacency,
                                    scratch.pred_node.get(),
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
        result.target_reached =
            !std::isinf(result.target_distance) &&
            (result.target_reached || result.converged);
      } else if (!use_target_set) {
        result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
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
      reset_compact_parent_touched_vertices(
          scratch, inf, stream, touched_count, true);
    } else if constexpr (TrackParents) {
      reset_touched_vertices(scratch, inf, stream, touched_count);
    } else {
      reset_distance_only_touched_vertices(
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
    reset_compact_parent_touched_vertices(
        scratch, inf, stream, touched_count_for_reset, !skip_heavy_edges);
  } else if constexpr (TrackParents) {
    reset_touched_vertices(scratch, inf, stream, touched_count_for_reset);
  } else {
    reset_distance_only_touched_vertices(
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

template <bool TrackParents, bool UseEdgeParent>
DeltaSteppingCsrResult dispatch_delta_stepping_impl(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::uint32_t* edge_source,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    bool skip_heavy_edges,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingCsrProgressCallback progress_callback,
    void* progress_user_data,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (telemetry != nullptr) {
    return run_delta_stepping_impl<TrackParents, UseEdgeParent, true>(
        d_adjacency, edge_source, scratch, sources, target, targets,
        vertex_costs, skip_heavy_edges, delta, max_iters, stream,
        progress_callback, progress_user_data, telemetry);
  }
  return run_delta_stepping_impl<TrackParents, UseEdgeParent, false>(
      d_adjacency, edge_source, scratch, sources, target, targets,
      vertex_costs, skip_heavy_edges, delta, max_iters, stream,
      progress_callback, progress_user_data, nullptr);
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
       DeltaSteppingCsrStorageMode storage_mode)
      : device(ds_delta_detail::current_hip_device()),
        adjacency(ds_delta_detail::copy_host_csr_to_device(
            host, stream,
            storage_mode == DeltaSteppingCsrStorageMode::kPathCapable)),
        max_edge_value(ds_delta_detail::max_edge_value(host.values)),
        has_exact_unit_edge_values(
            ds_delta_detail::has_exact_unit_edge_values(host.values)),
        path_capable(
            storage_mode == DeltaSteppingCsrStorageMode::kPathCapable) {}
};

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                                             hipStream_t stream)
    : DeltaSteppingCsrGraph(
          adjacency, stream, DeltaSteppingCsrStorageMode::kPathCapable) {}

DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode) {
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
  impl_ = std::make_shared<Impl>(adjacency, stream, storage_mode);
}

DeltaSteppingCsrGraph::~DeltaSteppingCsrGraph() = default;
DeltaSteppingCsrGraph::DeltaSteppingCsrGraph(
    DeltaSteppingCsrGraph&&) noexcept = default;
DeltaSteppingCsrGraph& DeltaSteppingCsrGraph::operator=(
    DeltaSteppingCsrGraph&&) noexcept = default;

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
       DeltaSteppingCsrStorageMode storage_mode)
      : owned_adjacency(std::make_unique<ds_delta_detail::DeviceCsrOwner>(
            ds_delta_detail::copy_host_csr_to_device(
                host, stream,
                storage_mode == DeltaSteppingCsrStorageMode::kPathCapable))),
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
        scratch(shared_graph->adjacency.view.rows),
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
          adjacency, stream, DeltaSteppingCsrStorageMode::kPathCapable) {}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    DeltaSteppingCsrStorageMode storage_mode) {
  using namespace ds_delta_detail;
  validate_host_csr_arrays(adjacency);
  if (adjacency.rows <= 0 || adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR graph must be nonempty and square");
  }
  if (static_cast<unsigned long long>(adjacency.rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("frontier vertices are stored as int; rows must fit in int");
  }
  impl_ = std::make_unique<Impl>(adjacency, stream, storage_mode);
}

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(std::move(adjacency), stream)) {}

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
  if (values.size() != static_cast<std::size_t>(adjacency.view.nnz)) {
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
  if (adjacency.view.nnz == 0) {
    impl_->max_edge_value = 0.0f;
    return;
  }
  impl_->max_edge_value = max_edge_value(values);
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(adjacency.values.get(),
                                    values.data(),
                                    values.size() * sizeof(float),
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
  if (vertex_costs.size() != static_cast<std::size_t>(adjacency.view.rows)) {
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
                                    vertex_costs.size() * sizeof(float),
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
  validate_device_csr_shape(adjacency.view, sources, -1, delta);
  // Every predecessor/key/target/path buffer is unnecessary in this mode.
  // Prior runs are fully synchronized before returning, so releasing retained
  // path state here cannot race queued work.
  impl_->scratch.release_parent_and_path_storage();
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
  return dispatch_delta_stepping_impl<false, false>(
      adjacency.view, nullptr, impl_->scratch, sources, -1, nullptr,
      impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
      skip_heavy_edges, delta, max_iters, stream, progress_callback,
      progress_user_data, active_telemetry_);
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
  validate_device_csr_shape(adjacency.view, sources, target, delta);
  PATHFINDER_PROFILE_RANGE("delta_step.generic");
  const bool skip_heavy_edges =
      !impl_->has_vertex_costs && impl_->max_edge_value <= delta;
  begin_telemetry_record(
      active_telemetry_, DeltaSteppingCsrExecutionPath::kLegacyGeneric,
      delta,
      execution_mode_ == DeltaSteppingCsrExecutionMode::kForceGeneric,
      parent_mode_ == DeltaSteppingCsrParentMode::kForceLegacy,
      impl_->has_vertex_costs, skip_heavy_edges, false);
  return dispatch_delta_stepping_impl<true, false>(
      adjacency.view, nullptr, impl_->scratch, sources, target, nullptr,
      impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
      skip_heavy_edges, delta, max_iters, stream, progress_callback,
      progress_user_data, active_telemetry_);
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
  validate_device_csr_shape(adjacency.view, sources, -1, delta);
  validate_target_list_common_shape(adjacency.view.rows, targets);
  if (execution_mode_ == DeltaSteppingCsrExecutionMode::kAutomatic &&
      parent_mode_ == DeltaSteppingCsrParentMode::kAutomatic &&
      impl_->has_exact_unit_edge_values &&
      !impl_->has_vertex_costs &&
      adjacency.view.rows <= kMaxUnitSpecializationRows &&
      max_iters < 0 &&
      progress_callback == nullptr) {
    PATHFINDER_PROFILE_RANGE("delta_step.unit_specialization");
    begin_telemetry_record(
        active_telemetry_, DeltaSteppingCsrExecutionPath::kExactUnit,
        delta, false, false, false,
        impl_->max_edge_value <= delta, false);
    if (active_telemetry_ != nullptr) {
      return run_unit_weight_specialization<true>(
          adjacency.view, impl_->scratch, sources, targets, delta, stream,
          active_telemetry_);
    }
    return run_unit_weight_specialization<false>(
        adjacency.view, impl_->scratch, sources, targets, delta, stream,
        nullptr);
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
    return dispatch_delta_stepping_impl<true, true>(
        adjacency.view, adjacency.edge_source.get(), impl_->scratch, sources,
        -1, &targets, vertex_costs, skip_heavy_edges, delta, max_iters,
        stream, progress_callback, progress_user_data, active_telemetry_);
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
  return dispatch_delta_stepping_impl<true, false>(
      adjacency.view, nullptr, impl_->scratch, sources, -1, &targets,
      vertex_costs, skip_heavy_edges, delta, max_iters, stream,
      progress_callback, progress_user_data, active_telemetry_);
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

  validate_device_csr_shape(d_adjacency, sources, target, delta);
  validate_device_csr_contents(d_adjacency, stream);
  DeltaSteppingScratch scratch(d_adjacency.rows);
  return dispatch_delta_stepping_impl<true, false>(
      d_adjacency, nullptr, scratch, sources, target, nullptr, nullptr, false,
      delta, max_iters, stream, progress_callback, progress_user_data,
      nullptr);
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
      copy_host_csr_to_device(adjacency, stream, false);
  return delta_stepping_minplus_hip_csr(d_adjacency.view, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
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
      copy_host_csr_to_device(adjacency, stream, false);
  return delta_stepping_minplus_hip_csr(d_adjacency.view, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
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
