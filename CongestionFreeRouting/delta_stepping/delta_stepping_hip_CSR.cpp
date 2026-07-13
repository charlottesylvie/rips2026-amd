#include "delta_stepping_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    count_ = count;
    if (count_ != 0) {
      DS_DELTA_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }
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
  // Unit-weight specialization: append tail and number of targets reached.
  DeviceBuffer<int> unit_status;
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
  bool initialized = false;

  DeltaSteppingScratch() = default;
  explicit DeltaSteppingScratch(Offset rows_)
      : rows(rows_),
        dist(static_cast<std::size_t>(rows_)),
        in_current(static_cast<std::size_t>(rows_)),
        in_pending(static_cast<std::size_t>(rows_)),
        in_heavy(static_cast<std::size_t>(rows_)),
        current_queue(static_cast<std::size_t>(rows_)),
        next_queue(static_cast<std::size_t>(rows_)),
        pending_a(static_cast<std::size_t>(rows_)),
        pending_b(static_cast<std::size_t>(rows_)),
        touched_queue(static_cast<std::size_t>(rows_)),
        heavy_queue(static_cast<std::size_t>(rows_)),
        current_count(1),
        next_count(1),
        pending_count(1),
        new_pending_count(1),
        heavy_count(1),
        touched_count(1),
        settled_target_count(1),
        min_pending_bucket(1),
        unit_status(2),
        pred_node(static_cast<std::size_t>(rows_)),
        pred_edge(static_cast<std::size_t>(rows_)),
        parent_key(static_cast<std::size_t>(rows_)),
        host_scalar(1),
        host_unit_status(2) {}

  void ensure_source_capacity(std::size_t source_count) {
    if (sources.size() < source_count) {
      sources.reset(source_count);
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    if (targets.size() < target_count) {
      targets.reset(target_count);
      target_settled.reset(target_count);
      target_distances.reset(target_count);
      target_path_lengths.reset(target_count);
      target_sources.reset(target_count);
      target_path_status.reset(target_count);
      target_node_offsets.reset(target_count + 1);
      target_edge_offsets.reset(target_count + 1);
    }
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count) {
    if (compact_path_nodes.size() < node_count) {
      compact_path_nodes.reset(node_count);
    }
    if (compact_path_edges.size() < edge_count) {
      compact_path_edges.reset(edge_count);
    }
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
  if (!std::isfinite(1.0f / delta)) {
    throw std::invalid_argument(
        "delta is too small to have a finite float reciprocal");
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

__device__ inline float atomic_min_float_nonnegative(float* addr, float value) {
  auto* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *addr_as_uint;
  while (value < __uint_as_float(old)) {
    const unsigned int assumed = old;
    old = atomicCAS(addr_as_uint, assumed, __float_as_uint(value));
    if (old == assumed) break;
  }
  return __uint_as_float(old);
}

__device__ inline void publish_parent_candidate(
    unsigned long long* parent_key,
    int predecessor,
    float candidate_distance) {
  const unsigned long long candidate_key =
      (static_cast<unsigned long long>(__float_as_uint(candidate_distance)) << 32) |
      static_cast<unsigned int>(predecessor);
  unsigned long long old = *parent_key;
  while (candidate_key < old) {
    const unsigned long long assumed = old;
    old = atomicCAS(parent_key, assumed, candidate_key);
    if (old == assumed) break;
  }
}

inline int bucket_index_host(float distance, float inv_delta) {
  if (!std::isfinite(distance)) return kNoBucket;
  const float bucket_f = distance * inv_delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  // Distances are nonnegative, so integer truncation is floor without an
  // additional libm operation.
  return static_cast<int>(bucket_f);
}

__device__ inline int bucket_index(float distance, float inv_delta) {
  if (!isfinite(distance)) return kNoBucket;
  const float bucket_f = distance * inv_delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  return static_cast<int>(bucket_f);
}

__device__ inline int wave_append_position(bool append, int* queue_tail) {
  const unsigned long long mask = __ballot(append);
  if (!append) {
    return -1;
  }

  const int lane = static_cast<int>(threadIdx.x % warpSize);
  const int leader = __ffsll(mask) - 1;
  int base = 0;
  if (lane == leader) {
    base = atomicAdd(queue_tail, __popcll(mask));
  }
  base = __shfl(base, leader);
  const unsigned long long lower_lanes =
      lane == 0 ? 0ULL : mask & ((1ULL << lane) - 1ULL);
  return base + __popcll(lower_lanes);
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
      if (dst < 0 || static_cast<Offset>(dst) >= cols || !isfinite(w) || w < 0.0f) {
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
                                                int* pred_node,
                                                Offset* pred_edge,
                                                unsigned long long* parent_key,
                                                int* touched_count) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < n;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    dist[v] = inf;
    in_current[v] = 0;
    in_pending[v] = 0;
    in_heavy[v] = 0;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
    parent_key[v] = kNoParentKey;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *current_count = 0;
    *next_count = 0;
    *pending_count = 0;
    *heavy_count = 0;
    *touched_count = 0;
  }
}

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
    pred_node[source] = source;
    pred_edge[source] = static_cast<Offset>(-1);
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
                                               float* dist,
                                               int* pred_node,
                                               Offset* pred_edge,
                                               int* frontier_queue,
                                               int* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status[0] = source_count;
    status[1] = initially_found;
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

__global__ void expand_unit_frontier_kernel(int frontier_begin,
                                            int frontier_end,
                                            float next_distance,
                                            const Offset* out_rowptr,
                                            const Index* out_colind,
                                            float* dist,
                                            int* pred_node,
                                            Offset* pred_edge,
                                            int* frontier_queue,
                                            int* queue_tail,
                                            const int* target_multiplicity,
                                            int* found_count) {
  const unsigned int infinity_bits = __float_as_uint(INFINITY);
  const unsigned int next_distance_bits = __float_as_uint(next_distance);
  for (int i = frontier_begin + blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_end;
       i += blockDim.x * gridDim.x) {
    const int u = frontier_queue[i];
    for (Offset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
      const int v = static_cast<int>(out_colind[edge]);
      auto* const distance_bits =
          reinterpret_cast<unsigned int*>(&dist[v]);
      const bool claimed =
          *distance_bits == infinity_bits &&
          atomicCAS(distance_bits, infinity_bits, next_distance_bits) ==
              infinity_bits;
      if (claimed) {
        pred_node[v] = u;
        pred_edge[v] = edge;
      }
      const int pos = wave_append_position(claimed, queue_tail);
      if (claimed) {
        frontier_queue[pos] = v;
        const int multiplicity = target_multiplicity[v];
        if (multiplicity > 0) {
          atomicAdd(found_count, multiplicity);
        }
      }
    }
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
    if (!isfinite(du) || !isfinite(dv)) continue;

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
                                            float inv_delta,
                                            const float* dist,
                                            int* target_settled,
                                            int* settled_count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (target_settled[i] != 0) continue;
    const int target = targets[i];
    const float target_distance = dist[target];
    if (isfinite(target_distance) &&
        bucket_index(target_distance, inv_delta) <= current_bucket &&
        atomicCAS(&target_settled[i], 0, 1) == 0) {
      atomicAdd(settled_count, 1);
    }
  }
}

__global__ void measure_target_paths_kernel(const int* targets,
                                            int target_count,
                                            Offset rows,
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
    const float target_distance = dist[target];
    target_distances[i] = target_distance;
    path_lengths[i] = 0;
    path_sources[i] = -1;
    path_status[i] = 0;
    if (!isfinite(target_distance)) {
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
    if (!isfinite(target_distance)) {
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
                                         const int* pred_node,
                                         const Offset* pred_edge,
                                         const int* path_lengths,
                                         const int* path_status,
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
    path_nodes[node_begin + length - 1] = current;
    for (int j = length - 1; j > 0; --j) {
      const int source_marker = pred_node[current];
      if (source_marker == current) {
        return;
      }
      const int pred = source_marker;
      if (pred < 0 || static_cast<Offset>(pred) >= rows) {
        return;
      }
      path_edges[edge_begin + j - 1] = pred_edge[current];
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
    path_sources[i] = current;
  }
}

__global__ void relax_light_edges_kernel(const int* frontier,
                                         int frontier_count,
                                         int current_bucket,
                                         float delta,
                                         float inv_delta,
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
                                         int collect_heavy) {
  // FPGA routing graphs have short outgoing rows.  Assign one active vertex to
  // each thread so a 256-thread block can process up to 256 rows concurrently,
  // matching the unit-BFS traversal instead of idling most of a block per row.
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < frontier_count;
       fi += blockDim.x * gridDim.x) {
    const int u = frontier[fi];
    // The in_current flags for this whole frontier are cleared by a separate
    // kernel before relaxation starts.  Do not clear them here: doing so races
    // with other blocks that may need to re-enqueue this vertex after a same-
    // bucket distance decrease.
    const float du = dist[u];
    const bool active = isfinite(du) && bucket_index(du, inv_delta) == current_bucket;
    const bool append_heavy =
        active && collect_heavy && atomicCAS(&in_heavy[u], 0, 1) == 0;
    const int heavy_pos = wave_append_position(append_heavy, heavy_count);
    if (append_heavy) {
      heavy_queue[heavy_pos] = u;
    }
    if (active) {
      for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            vertex_costs == nullptr ? w : w * vertex_costs[v];
        const bool light = effective_w <= delta;
        const float nd = light ? du + effective_w : INFINITY;
        const float old =
            light ? atomic_min_float_nonnegative(&dist[v], nd) : INFINITY;
        const bool decreased = light && nd < old;
        const bool append_touched = decreased && isinf(old);
        bool append_current = false;
        bool append_pending = false;
        if (decreased) {
          publish_parent_candidate(&parent_key[v], u, nd);
          const int b = bucket_index(nd, inv_delta);
          if (b == current_bucket) {
            // If v had previously been discovered in a later bucket, the new
            // distance has moved it back into the current bucket.  Clear the
            // pending flag so stale pending-queue entries cannot affect later
            // bucket selection.
            atomicExch(&in_pending[v], 0);
            append_current = atomicCAS(&in_current[v], 0, 1) == 0;
          } else if (b > current_bucket && b < kNoBucket) {
            append_pending = atomicCAS(&in_pending[v], 0, 1) == 0;
          }
        }
        const int touched_pos =
            wave_append_position(append_touched, touched_count);
        const int current_pos =
            wave_append_position(append_current, next_count);
        const int pending_pos =
            wave_append_position(append_pending, pending_count);
        if (append_touched) touched_queue[touched_pos] = v;
        if (append_current) next_frontier[current_pos] = v;
        if (append_pending) pending_queue[pending_pos] = v;
      }
    }
  }
}

__global__ void relax_heavy_edges_kernel(const int* heavy_vertices,
                                         int heavy_count_value,
                                         int current_bucket,
                                         float delta,
                                         float inv_delta,
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
                                         int* in_heavy) {
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < heavy_count_value;
       fi += blockDim.x * gridDim.x) {
    const int u = heavy_vertices[fi];
    const float du = dist[u];
    if (isfinite(du)) {
      for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            vertex_costs == nullptr ? w : w * vertex_costs[v];
        const bool heavy = effective_w > delta;
        const float nd = heavy ? du + effective_w : INFINITY;
        const float old =
            heavy ? atomic_min_float_nonnegative(&dist[v], nd) : INFINITY;
        const bool decreased = heavy && nd < old;
        const bool append_touched = decreased && isinf(old);
        bool append_pending = false;
        if (decreased) {
          publish_parent_candidate(&parent_key[v], u, nd);
          const int b = bucket_index(nd, inv_delta);
          if (b > current_bucket && b < kNoBucket) {
            append_pending = atomicCAS(&in_pending[v], 0, 1) == 0;
          }
        }
        const int touched_pos =
            wave_append_position(append_touched, touched_count);
        const int pending_pos =
            wave_append_position(append_pending, pending_count);
        if (append_touched) touched_queue[touched_pos] = v;
        if (append_pending) pending_queue[pending_pos] = v;
      }
    }
    in_heavy[u] = 0;
  }
}

__global__ void clear_flags_from_queue_kernel(const int* vertices, int count, int* flags) {
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

__global__ void reduce_min_pending_bucket_kernel(const int* pending_queue,
                                                 int pending_count,
                                                 int previous_bucket,
                                                 float inv_delta,
                                                 const float* dist,
                                                 const int* in_pending,
                                                 int* global_min) {
  __shared__ int s_min[kBlockSize];
  const int tid = threadIdx.x;
  int local_min = kNoBucket;
  for (int i = blockIdx.x * blockDim.x + tid;
       i < pending_count;
       i += blockDim.x * gridDim.x) {
    const int v = pending_queue[i];
    if (in_pending[v]) {
      const int b = bucket_index(dist[v], inv_delta);
      if (b > previous_bucket && b < local_min) local_min = b;
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
}

__global__ void compact_pending_to_current_bucket_kernel(const int* pending_in,
                                                         int pending_count,
                                                         int selected_bucket,
                                                         float inv_delta,
                                                         const float* dist,
                                                         int* in_pending,
                                                         int* in_current,
                                                         int* current_queue,
                                                         int* current_count,
                                                         int* pending_out,
                                                         int* new_pending_count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < pending_count;
       i += blockDim.x * gridDim.x) {
    const int v = pending_in[i];
    const bool active = in_pending[v] != 0;
    const int b = active ? bucket_index(dist[v], inv_delta) : kNoBucket;
    bool append_current = false;
    const bool keep_pending = active && b > selected_bucket && b < kNoBucket;
    if (active && b == selected_bucket) {
      atomicExch(&in_pending[v], 0);
      append_current = atomicCAS(&in_current[v], 0, 1) == 0;
    } else if (active && !keep_pending) {
      atomicExch(&in_pending[v], 0);
    }
    const int current_pos =
        wave_append_position(append_current, current_count);
    const int pending_pos =
        wave_append_position(keep_pending, new_pending_count);
    if (append_current) current_queue[current_pos] = v;
    if (keep_pending) pending_out[pending_pos] = v;
  }
}

DeviceCsrOwner copy_host_csr_to_device(const HostCsrF32& h, hipStream_t stream) {
  DeviceCsrOwner d(h.rows, h.cols, h.nnz);
  const std::size_t rows = checked_size(h.rows, "rows");
  const std::size_t nnz = checked_size(h.nnz, "nnz");
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(d.rowptr.get(), h.rowptr.data(),
                                    (rows + 1) * sizeof(Offset), hipMemcpyHostToDevice, stream));
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
  if (scratch.initialized) return;
  initialize_delta_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, scratch.dist.get(), scratch.in_current.get(), scratch.in_pending.get(),
      scratch.in_heavy.get(), scratch.current_count.get(), scratch.next_count.get(),
      scratch.pending_count.get(), scratch.heavy_count.get(),
      scratch.pred_node.get(), scratch.pred_edge.get(), scratch.parent_key.get(),
      scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.initialized = true;
}

void prepare_delta_scratch(DeltaSteppingScratch& scratch,
                           Offset n,
                           float inf,
                           hipStream_t stream) {
  if (!scratch.initialized) {
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
  const int touched_count =
      known_touched_count >= 0
          ? known_touched_count
          : copy_scalar_to_host(scratch.touched_count.get(),
                                stream,
                                scratch.host_scalar.get());
  if (touched_count > 0) {
    reset_touched_vertices_kernel<<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
        scratch.touched_queue.get(), touched_count, inf, scratch.dist.get(),
        scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
        scratch.pred_node.get(), scratch.pred_edge.get(), scratch.parent_key.get());
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
  // Relaxation atomically keeps the predecessor associated with the smallest
  // winning distance. Materialize only touched vertices and recover the exact
  // original CSR edge from the winning predecessor's normally short row.
  const int touched_count = copy_scalar_to_host(
      scratch.touched_count.get(), stream, scratch.host_scalar.get());
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
  return touched_count;
}

void extract_target_paths_to_result(DeltaSteppingCsrResult& result,
                                    DeltaSteppingScratch& scratch,
                                    const std::vector<int>& targets,
                                    hipStream_t stream,
                                    bool unit_weight_specialization = false) {
  const int target_count = static_cast<int>(targets.size());
  if (unit_weight_specialization) {
    measure_unit_target_paths_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.dist.get(),
            scratch.target_distances.get(), scratch.target_path_lengths.get(),
            scratch.target_sources.get(), scratch.target_path_status.get());
  } else {
    measure_target_paths_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(), target_count, scratch.rows,
            scratch.dist.get(), scratch.pred_node.get(),
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

  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    fill_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
        scratch.targets.get(), target_count, scratch.rows, scratch.pred_node.get(),
        scratch.pred_edge.get(),
        scratch.target_path_lengths.get(), scratch.target_path_status.get(),
        scratch.target_sources.get(),
        scratch.target_node_offsets.get(), scratch.target_edge_offsets.get(),
        scratch.compact_path_nodes.get(),
        scratch.compact_path_edges.get());
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
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  result.target_reached = all_targets_reached;
}

DeltaSteppingCsrResult run_unit_weight_specialization(
    const minplus_sparse::DeviceCsrF32& graph,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    hipStream_t stream) {
  // With identical positive edge weights, delta-stepping and multi-source BFS
  // have the same shortest paths.  The routing converter emits exactly this
  // case, so use an append-only frontier and claim each vertex once.  This
  // removes bucket scans, light-closure bookkeeping, and the O(E) predecessor
  // rebuild while preserving original CSR edge IDs.
  std::vector<int> deduplicated_sources;
  const std::vector<int>* effective_sources = &sources;
  int initially_found = 0;
  if (sources.size() == 1) {
    for (const int target : targets) {
      if (target == sources.front()) {
        ++initially_found;
      }
    }
  } else {
    std::unordered_set<int> source_set;
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

  const int source_count = static_cast<int>(effective_sources->size());
  const int target_count = static_cast<int>(targets.size());
  const int vertex_count = static_cast<int>(graph.rows);
  const float inf = std::numeric_limits<float>::infinity();
  scratch.ensure_source_capacity(effective_sources->size());
  scratch.ensure_target_capacity(targets.size());
  initialize_scratch_storage_once(scratch, graph.rows, inf, stream);

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
          scratch.dist.get(),
          scratch.pred_node.get(),
          scratch.pred_edge.get(),
          scratch.current_queue.get(),
          scratch.unit_status.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());

  int frontier_begin = 0;
  int frontier_end = source_count;
  int queue_tail = source_count;
  int current_count = source_count;
  int found_count = initially_found;
  int max_reached_distance = 0;
  DeltaSteppingCsrResult result;
  result.target = -1;

  for (int depth = 0;
       current_count > 0 && found_count < target_count && depth < vertex_count;
       ++depth) {
    expand_unit_frontier_kernel
        <<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
            frontier_begin,
            frontier_end,
            static_cast<float>(depth + 1),
            graph.rowptr,
            graph.colind,
            scratch.dist.get(),
            scratch.pred_node.get(),
            scratch.pred_edge.get(),
            scratch.current_queue.get(),
            scratch.unit_status.get(),
            scratch.in_pending.get(),
            scratch.unit_status.get() + 1);
    DS_DELTA_HIP_CHECK(hipGetLastError());
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.host_unit_status.get(),
                                      scratch.unit_status.get(),
                                      2 * sizeof(int),
                                      hipMemcpyDeviceToHost,
                                      stream));
    DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
    queue_tail = scratch.host_unit_status.get()[0];
    found_count = scratch.host_unit_status.get()[1];
    if (queue_tail < frontier_end || queue_tail > vertex_count) {
      throw std::runtime_error(
          "delta unit-weight append-only frontier queue overflow");
    }
    if (queue_tail > frontier_end) {
      max_reached_distance = depth + 1;
    }
    current_count = queue_tail - frontier_end;
    frontier_begin = frontier_end;
    frontier_end = queue_tail;
    result.iterations_used = depth + 1;
  }

  result.stopped_on_target = found_count >= target_count;
  result.converged = !result.stopped_on_target && current_count == 0;
  clear_unit_target_multiplicity_kernel
      <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
          scratch.targets.get(),
          target_count,
          scratch.in_pending.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  extract_target_paths_to_result(result, scratch, targets, stream, true);

  // Report nonempty delta bucket rounds rather than BFS expansion depth.
  int reporting_distance = max_reached_distance;
  if (result.target_reached) {
    reporting_distance = 0;
    for (const float distance : result.target_distances) {
      reporting_distance =
          std::max(reporting_distance, static_cast<int>(distance));
    }
  }
  // For delta >= 1, integer path lengths visit every bucket through the last
  // relevant bucket. For delta < 1, unit edges are heavy and bucket indices can
  // skip, but exactly one nonempty bucket is processed per path level.
  result.iterations_used =
      delta >= 1.0f
          ? bucket_index_host(static_cast<float>(reporting_distance),
                              1.0f / delta) + 1
          : reporting_distance + 1;

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
  return result;
}

int find_min_pending_bucket(const int* d_pending_queue,
                            int pending_count,
                            int current_bucket,
                            float inv_delta,
                            const float* d_dist,
                            const int* d_in_pending,
                            int* d_min_bucket,
                            int* h_min_bucket,
                            hipStream_t stream) {
  if (pending_count <= 0) return kNoBucket;
  const int blocks = grid_for_frontier(pending_count);
  const int initial_min = kNoBucket;
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(d_min_bucket,
                                    &initial_min,
                                    sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));
  reduce_min_pending_bucket_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_pending_queue, pending_count, current_bucket, inv_delta, d_dist,
      d_in_pending, d_min_bucket);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  return copy_scalar_to_host(d_min_bucket, stream, h_min_bucket);
}

int mark_and_count_settled_targets(DeltaSteppingScratch& scratch,
                                   int target_count,
                                   int current_bucket,
                                   float inv_delta,
                                   hipStream_t stream) {
  mark_settled_targets_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(), target_count, current_bucket, inv_delta,
      scratch.dist.get(), scratch.target_settled.get(),
      scratch.settled_target_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  return copy_scalar_to_host(scratch.settled_target_count.get(),
                             stream,
                             scratch.host_scalar.get());
}

DeltaSteppingCsrResult run_delta_stepping_impl(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
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
    void* progress_user_data) {
  if (max_iters < 0) max_iters = std::numeric_limits<int>::max();

  const Offset n = d_adjacency.rows;
  std::vector<int> deduplicated_sources;
  const std::vector<int>* effective_sources = &sources;
  if (sources.size() > 1) {
    std::unordered_set<int> seen;
    seen.reserve(sources.size());
    deduplicated_sources.reserve(sources.size());
    for (const int source : sources) {
      if (seen.insert(source).second) {
        deduplicated_sources.push_back(source);
      }
    }
    effective_sources = &deduplicated_sources;
  }
  const int source_count = static_cast<int>(effective_sources->size());
  const bool use_target_set = targets != nullptr;
  const int target_count = use_target_set ? static_cast<int>(targets->size()) : 0;
  const float inv_delta = 1.0f / delta;
  const float inf = std::numeric_limits<float>::infinity();
  const bool target_is_source =
      !use_target_set &&
      target >= 0 && std::find(sources.begin(), sources.end(), target) != sources.end();

  scratch.ensure_source_capacity(effective_sources->size());
  if (use_target_set) {
    scratch.ensure_target_capacity(static_cast<std::size_t>(target_count));
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.targets.get(), targets->data(),
                                      static_cast<std::size_t>(target_count) * sizeof(int),
                                      hipMemcpyHostToDevice, stream));
    DS_DELTA_HIP_CHECK(hipMemsetAsync(scratch.target_settled.get(),
                                      0,
                                      static_cast<std::size_t>(target_count) * sizeof(int),
                                      stream));
    reset_int_zero_async(scratch.settled_target_count.get(), stream);
  }
  prepare_delta_scratch(scratch, n, inf, stream);
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(), effective_sources->data(),
                                    static_cast<std::size_t>(source_count) * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
  initialize_delta_sources_kernel<<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
      scratch.sources.get(), source_count, scratch.dist.get(), scratch.in_current.get(),
      scratch.current_queue.get(), scratch.current_count.get(),
      scratch.pred_node.get(), scratch.pred_edge.get(),
      scratch.touched_queue.get(), scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());

  int* current_queue = scratch.current_queue.get();
  int* next_queue = scratch.next_queue.get();
  int* pending_queue = scratch.pending_a.get();
  int* pending_scratch = scratch.pending_b.get();
  int current_bucket = 0;
  int current_count = copy_scalar_to_host(
      scratch.current_count.get(), stream, scratch.host_scalar.get());
  int pending_count = 0;
  DeltaSteppingCsrResult result;
  result.target = target;
  if (target_is_source) {
    result.target_distance = 0.0f;
    result.target_reached = true;
    result.stopped_on_target = true;
    copy_predecessors_to_result(result, d_adjacency, scratch.pred_node.get(),
                                scratch.pred_edge.get(), stream);
    result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
    reset_touched_vertices(scratch, inf, stream);
    return result;
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    reset_int_zero_async(scratch.heavy_count.get(), stream);

    while (current_count > 0) {
      reset_int_zero_async(scratch.next_count.get(), stream);

      clear_flags_from_queue_kernel<<<grid_for_items(current_count), kBlockSize, 0, stream>>>(
          current_queue, current_count, scratch.in_current.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());

      relax_light_edges_kernel<<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta, inv_delta,
          d_adjacency.rowptr, d_adjacency.colind, d_adjacency.values,
          vertex_costs, scratch.dist.get(), scratch.parent_key.get(),
          scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
          scratch.touched_queue.get(), scratch.touched_count.get(),
          next_queue, scratch.next_count.get(),
          pending_queue, scratch.pending_count.get(), scratch.heavy_queue.get(),
          scratch.heavy_count.get(), skip_heavy_edges ? 0 : 1);
      DS_DELTA_HIP_CHECK(hipGetLastError());
      current_count = copy_scalar_to_host(
          scratch.next_count.get(), stream, scratch.host_scalar.get());
      std::swap(current_queue, next_queue);
    }

    const int heavy_count =
        skip_heavy_edges
            ? 0
            : copy_scalar_to_host(scratch.heavy_count.get(),
                                  stream,
                                  scratch.host_scalar.get());
    if (heavy_count > 0) {
      relax_heavy_edges_kernel<<<grid_for_frontier(heavy_count), kBlockSize, 0, stream>>>(
          scratch.heavy_queue.get(), heavy_count, current_bucket, delta, inv_delta,
          d_adjacency.rowptr, d_adjacency.colind, d_adjacency.values,
          vertex_costs, scratch.dist.get(), scratch.parent_key.get(),
          scratch.in_pending.get(),
          scratch.touched_queue.get(), scratch.touched_count.get(),
          pending_queue,
          scratch.pending_count.get(), scratch.in_heavy.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
    }

    result.iterations_used = iter + 1;
    if (use_target_set) {
      const int settled_count =
          mark_and_count_settled_targets(scratch, target_count, current_bucket,
                                         inv_delta, stream);
      if (settled_count >= target_count) {
        result.target_reached = true;
        result.stopped_on_target = true;
        if (progress_callback) {
          DeltaSteppingCsrProgress progress;
          progress.iteration = result.iterations_used;
          progress.max_iters = max_iters;
          progress.convergence_checked = true;
          progress.changed = true;
          progress_callback(progress, progress_user_data);
        }
        break;
      }
    } else if (target >= 0) {
      const float target_distance = copy_dist_value_to_host(scratch.dist.get(), target, stream);
      const bool target_settled =
          std::isfinite(target_distance) &&
          bucket_index_host(target_distance, inv_delta) <= current_bucket;
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
          progress_callback(progress, progress_user_data);
        }
        break;
      }
    }

    pending_count = copy_scalar_to_host(
        scratch.pending_count.get(), stream, scratch.host_scalar.get());
    const int next_bucket =
        find_min_pending_bucket(pending_queue, pending_count, current_bucket,
                                inv_delta, scratch.dist.get(), scratch.in_pending.get(),
                                scratch.min_pending_bucket.get(),
                                scratch.host_scalar.get(), stream);
    const bool changed = (next_bucket != kNoBucket);
    if (progress_callback) {
      DeltaSteppingCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_iters;
      progress.convergence_checked = true;
      progress.changed = changed;
      progress_callback(progress, progress_user_data);
    }
    if (!changed) {
      result.converged = true;
      break;
    }

    current_bucket = next_bucket;
    current_queue = scratch.current_queue.get();
    next_queue = scratch.next_queue.get();
    reset_int_zero_async(scratch.current_count.get(), stream);
    reset_int_zero_async(scratch.new_pending_count.get(), stream);

    compact_pending_to_current_bucket_kernel<<<grid_for_items(pending_count), kBlockSize, 0, stream>>>(
        pending_queue, pending_count, current_bucket, inv_delta, scratch.dist.get(),
        scratch.in_pending.get(), scratch.in_current.get(), current_queue,
        scratch.current_count.get(), pending_scratch, scratch.new_pending_count.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());

    current_count = copy_scalar_to_host(
        scratch.current_count.get(), stream, scratch.host_scalar.get());
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.pending_count.get(),
                                      scratch.new_pending_count.get(),
                                      sizeof(int),
                                      hipMemcpyDeviceToDevice,
                                      stream));
    std::swap(pending_queue, pending_scratch);
  }

  int touched_count_for_reset = -1;
  if (use_target_set || target >= 0) {
    touched_count_for_reset =
        materialize_predecessors_from_keys(
            d_adjacency, scratch, vertex_costs, stream);
  }
  if (use_target_set) {
    extract_target_paths_to_result(result, scratch, *targets, stream);
  } else if (target >= 0) {
    copy_predecessors_to_result(result, d_adjacency, scratch.pred_node.get(),
                                scratch.pred_edge.get(), stream);
  }
  if (use_target_set) {
    result.target = -1;
  } else if (target >= 0) {
    result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
    result.target_distance = result.dist[static_cast<std::size_t>(target)];
    result.target_reached =
        !std::isinf(result.target_distance) &&
        (result.target_reached || result.converged);
  } else {
    result.dist = copy_dist_to_host(scratch.dist.get(), n, stream);
  }
  reset_touched_vertices(scratch, inf, stream, touched_count_for_reset);
  return result;
}

}  // namespace ds_delta_detail

struct DeltaSteppingCsrWorkspace::Impl {
  ds_delta_detail::DeviceCsrOwner adjacency;
  ds_delta_detail::DeltaSteppingScratch scratch;
  ds_delta_detail::DeviceBuffer<float> vertex_costs;
  float max_edge_value = 0.0f;
  bool has_exact_unit_edge_values = false;
  bool has_vertex_costs = false;
  hipStream_t stream = nullptr;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : adjacency(ds_delta_detail::copy_host_csr_to_device(host, stream)),
        scratch(host.rows),
        vertex_costs(static_cast<std::size_t>(host.rows)),
        max_edge_value(ds_delta_detail::max_edge_value(host.values)),
        has_exact_unit_edge_values(
            ds_delta_detail::has_exact_unit_edge_values(host.values)),
        stream(stream) {}

  void require_stream(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "DeltaSteppingCsrWorkspace is stream-affine; use its construction stream");
    }
  }
};

DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                                                     hipStream_t stream) {
  using namespace ds_delta_detail;
  validate_host_csr_arrays(adjacency);
  if (adjacency.rows <= 0 || adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR graph must be nonempty and square");
  }
  if (static_cast<unsigned long long>(adjacency.rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("frontier vertices are stored as int; rows must fit in int");
  }
  impl_ = std::make_unique<Impl>(adjacency, stream);
}

DeltaSteppingCsrWorkspace::~DeltaSteppingCsrWorkspace() = default;
DeltaSteppingCsrWorkspace::DeltaSteppingCsrWorkspace(
    DeltaSteppingCsrWorkspace&&) noexcept = default;
DeltaSteppingCsrWorkspace& DeltaSteppingCsrWorkspace::operator=(
    DeltaSteppingCsrWorkspace&&) noexcept = default;

void DeltaSteppingCsrWorkspace::update_values(const std::vector<float>& values,
                                              hipStream_t stream) {
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_stream(stream);
  if (values.size() != static_cast<std::size_t>(impl_->adjacency.view.nnz)) {
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
  if (impl_->adjacency.view.nnz == 0) {
    impl_->max_edge_value = 0.0f;
    return;
  }
  impl_->max_edge_value = max_edge_value(values);
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(impl_->adjacency.values.get(),
                                    values.data(),
                                    values.size() * sizeof(float),
                                    hipMemcpyHostToDevice,
                                    stream));
}

void DeltaSteppingCsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  using namespace ds_delta_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingCsrWorkspace has no implementation");
  }
  impl_->require_stream(stream);
  if (vertex_costs.size() != static_cast<std::size_t>(impl_->adjacency.view.rows)) {
    throw std::invalid_argument("vertex cost size does not match workspace rows");
  }
  for (const float cost : vertex_costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument("vertex costs must be finite nonnegative values");
    }
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(impl_->vertex_costs.get(),
                                    vertex_costs.data(),
                                    vertex_costs.size() * sizeof(float),
                                    hipMemcpyHostToDevice,
                                    stream));
  impl_->has_vertex_costs = true;
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
  impl_->require_stream(stream);
  validate_device_csr_shape(impl_->adjacency.view, sources, target, delta);
  return run_delta_stepping_impl(impl_->adjacency.view,
                                 impl_->scratch,
                                 sources,
                                 target,
                                 nullptr,
                                 impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
                                 !impl_->has_vertex_costs && impl_->max_edge_value <= delta,
                                 delta,
                                 max_iters,
                                 stream,
                                 progress_callback,
                                 progress_user_data);
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
  impl_->require_stream(stream);
  validate_device_csr_shape(impl_->adjacency.view, sources, -1, delta);
  validate_target_list_common_shape(impl_->adjacency.view.rows, targets);
  if (impl_->has_exact_unit_edge_values &&
      !impl_->has_vertex_costs &&
      impl_->adjacency.view.rows <= kMaxUnitSpecializationRows &&
      max_iters < 0 &&
      progress_callback == nullptr) {
    return run_unit_weight_specialization(impl_->adjacency.view,
                                          impl_->scratch,
                                          sources,
                                          targets,
                                          delta,
                                          stream);
  }
  return run_delta_stepping_impl(impl_->adjacency.view,
                                 impl_->scratch,
                                 sources,
                                 -1,
                                 &targets,
                                 impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
                                 !impl_->has_vertex_costs && impl_->max_edge_value <= delta,
                                 delta,
                                 max_iters,
                                 stream,
                                 progress_callback,
                                 progress_user_data);
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
  return run_delta_stepping_impl(d_adjacency, scratch, sources, target,
                                 nullptr, nullptr, false, delta, max_iters, stream, progress_callback,
                                 progress_user_data);
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
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
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
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
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
