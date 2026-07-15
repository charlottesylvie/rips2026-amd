#include "delta_stepping_hip_CSR_threads.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ds_delta_threads_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kNoBucket = 0x3fffffff;

// Compile-time switch for the relaxation mapping.
//   1: one GPU thread handles one active low-degree vertex (recommended for FPGA RRGs).
//   0: original one block cooperates on one active vertex.
#ifndef DS_USE_LOWDEG_RELAX
#define DS_USE_LOWDEG_RELAX 1
#endif

// Queue capacity is one slot per vertex and all hot queues are guarded by
// per-vertex membership flags, so overflow is not expected in normal runs.
// Keep the expensive host-side overflow checks available for debug builds.
#ifndef DS_CHECK_QUEUE_OVERFLOW
#define DS_CHECK_QUEUE_OVERFLOW 0
#endif

inline void hip_check(hipError_t status, const char* expr, const char* file, int line) {
  if (status != hipSuccess) {
    std::ostringstream os;
    os << "HIP error at " << file << ':' << line << " for " << expr << ": "
       << hipGetErrorString(status);
    throw std::runtime_error(os.str());
  }
}

#define DS_DELTA_HIP_CHECK(expr) \
  ::ds_delta_threads_detail::hip_check((expr), #expr, __FILE__, __LINE__)

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

// Retained only for the legacy transpose helpers below.  The public threaded
// solver no longer constructs or uses this representation.
struct OutgoingCsrOwner {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  DeviceBuffer<Offset> rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  DeviceBuffer<Offset> incoming_edge;

  OutgoingCsrOwner() = default;
  OutgoingCsrOwner(Offset rows_, Offset cols_, Offset nnz_)
      : rows(rows_), cols(cols_), nnz(nnz_),
        rowptr(static_cast<std::size_t>(rows_) + 1),
        colind(static_cast<std::size_t>(nnz_)),
        values(static_cast<std::size_t>(nnz_)),
        incoming_edge(static_cast<std::size_t>(nnz_)) {}
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
  DeviceBuffer<int> in_touched;
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
  DeviceBuffer<int> overflow;
  DeviceBuffer<int> settled_target_count;
  DeviceBuffer<int> min_pending_bucket;
  DeviceBuffer<int> pred_node;
  DeviceBuffer<Offset> pred_edge;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_sources;
  DeviceBuffer<int> target_path_status;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  bool initialized = false;

  DeltaSteppingScratch() = default;
  explicit DeltaSteppingScratch(Offset rows_)
      : rows(rows_),
        dist(static_cast<std::size_t>(rows_)),
        in_current(static_cast<std::size_t>(rows_)),
        in_pending(static_cast<std::size_t>(rows_)),
        in_heavy(static_cast<std::size_t>(rows_)),
        in_touched(static_cast<std::size_t>(rows_)),
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
        overflow(1),
        settled_target_count(1),
        min_pending_bucket(1),
        pred_node(static_cast<std::size_t>(rows_)),
        pred_edge(static_cast<std::size_t>(rows_)) {}

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
  return std::min(kMaxGridX, items);
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
      throw std::invalid_argument("CSR colind contains an out-of-range source vertex");
    }
    if (!std::isfinite(g.values[e]) || g.values[e] < 0.0f) {
      throw std::invalid_argument("delta-stepping requires finite nonnegative edge weights");
    }
  }
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
inline T copy_scalar_to_host(const T* d_value, hipStream_t stream) {
  T h{};
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(&h, d_value, sizeof(T), hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  return h;
}

inline void reset_int_zero_async(int* d_value, hipStream_t stream) {
  DS_DELTA_HIP_CHECK(hipMemsetAsync(d_value, 0, sizeof(int), stream));
}

inline void reset_overflow_async(int* d_value, hipStream_t stream) {
#if DS_CHECK_QUEUE_OVERFLOW
  reset_int_zero_async(d_value, stream);
#else
  (void)d_value;
  (void)stream;
#endif
}

__device__ inline Offset atomic_add_offset(Offset* ptr, Offset value) {
  static_assert(sizeof(Offset) == 4 || sizeof(Offset) == 8,
                "Offset must be a 32-bit or 64-bit integer type");
  if constexpr (sizeof(Offset) == 8) {
    return static_cast<Offset>(atomicAdd(reinterpret_cast<unsigned long long*>(ptr),
                                         static_cast<unsigned long long>(value)));
  } else {
    return static_cast<Offset>(atomicAdd(reinterpret_cast<unsigned int*>(ptr),
                                         static_cast<unsigned int>(value)));
  }
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

inline int bucket_index_host(float distance, float inv_delta) {
  if (!std::isfinite(distance)) return kNoBucket;
  const float bucket_f = distance * inv_delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  return static_cast<int>(bucket_f);
}

__device__ inline int bucket_index(float distance, float inv_delta) {
  if (!isfinite(distance)) return kNoBucket;
  const float bucket_f = distance * inv_delta;
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  return static_cast<int>(bucket_f);
}

__device__ inline void enqueue_unique(int v,
                                      int* flags,
                                      int* queue,
                                      int* count,
                                      int capacity,
                                      int* overflow) {
  if (atomicCAS(&flags[v], 0, 1) == 0) {
    const int pos = atomicAdd(count, 1);
    if (pos < capacity) {
      queue[pos] = v;
    } else {
      atomicExch(overflow, 1);
      atomicExch(&flags[v], 0);
    }
  }
}

__device__ inline void mark_touched(int v,
                                    int* in_touched,
                                    int* touched_queue,
                                    int* touched_count,
                                    int capacity,
                                    int* overflow) {
  if (atomicCAS(&in_touched[v], 0, 1) == 0) {
    const int pos = atomicAdd(touched_count, 1);
    if (pos < capacity) {
      touched_queue[pos] = v;
    } else {
      atomicExch(overflow, 1);
    }
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
      const Index src = colind[e];
      const float w = values[e];
      if (src < 0 || static_cast<Offset>(src) >= cols || !isfinite(w) || w < 0.0f) {
        atomicExch(invalid, 1);
      }
    }
  }
}

__global__ void count_out_degrees_from_incoming_kernel(Offset rows,
                                                       const Offset* in_rowptr,
                                                       const Index* in_colind,
                                                       Offset* out_degree) {
  // Public CSR row dst enumerates incoming edges src -> dst. Count by src.
  for (Offset dst = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       dst < rows;
       dst += static_cast<Offset>(blockDim.x) * gridDim.x) {
    for (Offset e = in_rowptr[dst]; e < in_rowptr[dst + 1]; ++e) {
      const Offset src = static_cast<Offset>(in_colind[e]);
      atomic_add_offset(&out_degree[src], static_cast<Offset>(1));
    }
  }
}

__global__ void set_last_rowptr_kernel(Offset* rowptr, Offset rows, Offset nnz) {
  if (blockIdx.x == 0 && threadIdx.x == 0) rowptr[rows] = nnz;
}

__global__ void fill_outgoing_from_incoming_kernel(Offset rows,
                                                   const Offset* in_rowptr,
                                                   const Index* in_colind,
                                                   const float* in_values,
                                                   Offset* cursor,
                                                   Index* out_colind,
                                                   float* out_values,
                                                   Offset* out_incoming_edge) {
  // Transpose incoming-edge CSR into outgoing CSR: output row src stores dst.
  for (Offset dst = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       dst < rows;
       dst += static_cast<Offset>(blockDim.x) * gridDim.x) {
    for (Offset e = in_rowptr[dst]; e < in_rowptr[dst + 1]; ++e) {
      const Offset src = static_cast<Offset>(in_colind[e]);
      const Offset pos = atomic_add_offset(&cursor[src], static_cast<Offset>(1));
      out_colind[pos] = static_cast<Index>(dst);
      out_values[pos] = in_values[e];
      out_incoming_edge[pos] = e;
    }
  }
}

__global__ void update_outgoing_values_from_incoming_kernel(Offset nnz,
                                                            const float* in_values,
                                                            const Offset* out_incoming_edge,
                                                            float* out_values) {
  for (Offset e = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       e < nnz;
       e += static_cast<Offset>(blockDim.x) * gridDim.x) {
    out_values[e] = in_values[out_incoming_edge[e]];
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
                                                int* overflow,
                                                int* pred_node,
                                                Offset* pred_edge,
                                                int* in_touched,
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
    in_touched[v] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *current_count = 0;
    *next_count = 0;
    *pending_count = 0;
    *heavy_count = 0;
    *overflow = 0;
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
                                                int* in_touched,
                                                int* touched_queue,
                                                int* touched_count,
                                                int queue_capacity,
                                                int* overflow) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    dist[source] = 0.0f;
    pred_node[source] = source;
    pred_edge[source] = static_cast<Offset>(-1);
    mark_touched(source, in_touched, touched_queue, touched_count,
                 queue_capacity, overflow);
    if (atomicCAS(&in_current[source], 0, 1) == 0) {
      const int pos = atomicAdd(current_count, 1);
      current_queue[pos] = source;
    }
  }
}

__global__ void compute_predecessors_kernel(Offset rows,
                                            const Offset* rowptr,
                                            const Index* colind,
                                            const float* values,
                                            const float* vertex_costs,
                                            const float* dist,
                                            int* pred_node,
                                            Offset* pred_edge) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    const float current_dist = dist[v];
    int best_pred = -1;
    Offset best_edge = static_cast<Offset>(-1);
    float best_error = INFINITY;

    if (isfinite(current_dist)) {
      for (Offset edge = rowptr[v]; edge < rowptr[v + 1]; ++edge) {
        const int pred = static_cast<int>(colind[edge]);
        if (pred == static_cast<int>(v)) continue;
        const float pred_dist = dist[pred];
        if (!isfinite(pred_dist)) continue;
        const float edge_cost =
            values[edge] * (vertex_costs == nullptr ? 1.0f : vertex_costs[v]);
        const float candidate = pred_dist + edge_cost;
        const float error = fabsf(candidate - current_dist);
        const float tolerance =
            1e-3f * fmaxf(1.0f, fmaxf(fabsf(candidate), fabsf(current_dist)));
        if (error <= tolerance &&
            (error < best_error ||
             (error == best_error && (best_pred < 0 || pred < best_pred)))) {
          best_error = error;
          best_pred = pred;
          best_edge = edge;
        }
      }
    }

    pred_node[v] = best_pred;
    pred_edge[v] = best_edge;
  }
}

__device__ inline bool find_final_distance_predecessor(Offset rows,
                                                       const Offset* rowptr,
                                                       const Index* colind,
                                                       const float* values,
                                                       const float* vertex_costs,
                                                       const float* dist,
                                                       int current,
                                                       int* pred_out,
                                                       Offset* edge_out) {
  const float current_dist = dist[current];
  int best_pred = -1;
  Offset best_edge = static_cast<Offset>(-1);
  float best_error = INFINITY;

  if (!isfinite(current_dist)) {
    return false;
  }
  for (Offset edge = rowptr[current]; edge < rowptr[current + 1]; ++edge) {
    const int pred = static_cast<int>(colind[edge]);
    if (pred == current) continue;
    const float pred_dist = dist[pred];
    if (!isfinite(pred_dist)) continue;
    const float edge_cost =
        values[edge] * (vertex_costs == nullptr ? 1.0f : vertex_costs[current]);
    const float candidate = pred_dist + edge_cost;
    const float error = fabsf(candidate - current_dist);
    const float tolerance =
        1e-3f * fmaxf(1.0f, fmaxf(fabsf(candidate), fabsf(current_dist)));
    if (error <= tolerance &&
        (error < best_error ||
         (error == best_error && (best_pred < 0 || pred < best_pred)))) {
      best_error = error;
      best_pred = pred;
      best_edge = edge;
    }
  }
  if (best_pred < 0) {
    return false;
  }
  *pred_out = best_pred;
  *edge_out = best_edge;
  return true;
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
      if (pred < 0 || pred == current) {
        break;
      }
      (void)edge;
      current = pred;
      ++length;
    }
  }
}

__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const int* pred_node,
                                         const Offset* pred_edge,
                                         const int* path_lengths,
                                         const int* path_status,
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
      const int pred = pred_node[current];
      const Offset edge = pred_edge[current];
      if (pred < 0 || pred == current || edge < 0) {
        return;
      }
      path_edges[edge_begin + j - 1] = edge;
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
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
                                         int* in_current,
                                         int* in_pending,
                                         int* in_heavy,
                                         int* pred_node,
                                         Offset* pred_edge,
                                         int* in_touched,
                                         int* touched_queue,
                                         int* touched_count,
                                         int* next_frontier,
                                         int* next_count,
                                         int* pending_queue,
                                         int* pending_count,
                                         int* heavy_queue,
                                         int* heavy_count,
                                         int queue_capacity,
                                         int* overflow) {
#if DS_USE_LOWDEG_RELAX
  // Low-degree mapping: one GPU thread processes one active vertex and loops
  // over that vertex's few outgoing edges. This is intended for routing graphs
  // where most vertices have small degree, e.g. 4-9 outgoing edges.
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < frontier_count;
       fi += blockDim.x * gridDim.x) {
    const int u = frontier[fi];
    in_current[u] = 0;
    const float du = dist[u];
    if (!isfinite(du) || bucket_index(du, inv_delta) != current_bucket) {
      continue;
    }

    enqueue_unique(u, in_heavy, heavy_queue, heavy_count, queue_capacity, overflow);

    for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
      const float w = out_values[e];
      const int v = static_cast<int>(out_colind[e]);
      const float effective_w =
          vertex_costs == nullptr ? w : w * vertex_costs[v];
      if (effective_w > delta) continue;
      const float nd = du + effective_w;
      const float old = atomic_min_float_nonnegative(&dist[v], nd);
      if (nd < old) {
        // A later, shorter relaxation may already have won before this
        // thread records its parent.  Only publish a parent for the distance
        // that is still current, so compact outgoing paths stay consistent.
        if (dist[v] == nd) {
          pred_node[v] = u;
          pred_edge[v] = e;
        }
        mark_touched(v, in_touched, touched_queue, touched_count,
                     queue_capacity, overflow);
        const int b = bucket_index(nd, inv_delta);
        if (b == current_bucket) {
          atomicExch(&in_pending[v], 0);
          enqueue_unique(v, in_current, next_frontier, next_count, queue_capacity, overflow);
        } else if (b > current_bucket && b < kNoBucket) {
          enqueue_unique(v, in_pending, pending_queue, pending_count, queue_capacity, overflow);
        }
      }
    }
  }
#else
  // Original mapping: one block cooperates on one active vertex at a time;
  // useful for high-degree vertices but wasteful when most vertices have only
  // a handful of outgoing edges.
  for (int fi = blockIdx.x; fi < frontier_count; fi += gridDim.x) {
    const int u = frontier[fi];
    if (threadIdx.x == 0) {
      in_current[u] = 0;
    }
    __syncthreads();
    const float du = dist[u];
    const bool active = isfinite(du) && bucket_index(du, inv_delta) == current_bucket;
    if (active) {
      if (threadIdx.x == 0) {
        enqueue_unique(u, in_heavy, heavy_queue, heavy_count, queue_capacity, overflow);
      }
      for (Offset e = out_rowptr[u] + threadIdx.x; e < out_rowptr[u + 1]; e += blockDim.x) {
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            vertex_costs == nullptr ? w : w * vertex_costs[v];
        if (effective_w > delta) continue;
        const float nd = du + effective_w;
        const float old = atomic_min_float_nonnegative(&dist[v], nd);
        if (nd < old) {
          if (dist[v] == nd) {
            pred_node[v] = u;
            pred_edge[v] = e;
          }
          mark_touched(v, in_touched, touched_queue, touched_count,
                       queue_capacity, overflow);
          const int b = bucket_index(nd, inv_delta);
          if (b == current_bucket) {
            atomicExch(&in_pending[v], 0);
            enqueue_unique(v, in_current, next_frontier, next_count, queue_capacity, overflow);
          } else if (b > current_bucket && b < kNoBucket) {
            enqueue_unique(v, in_pending, pending_queue, pending_count, queue_capacity, overflow);
          }
        }
      }
    }
    __syncthreads();
  }
#endif
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
                                         int* in_heavy,
                                         int* in_pending,
                                         int* pred_node,
                                         Offset* pred_edge,
                                         int* in_touched,
                                         int* touched_queue,
                                         int* touched_count,
                                         int* pending_queue,
                                         int* pending_count,
                                         int queue_capacity,
                                         int* overflow) {
#if DS_USE_LOWDEG_RELAX
  // Low-degree mapping: one GPU thread processes one heavy-frontier vertex.
  for (int fi = blockIdx.x * blockDim.x + threadIdx.x;
       fi < heavy_count_value;
       fi += blockDim.x * gridDim.x) {
    const int u = heavy_vertices[fi];
    in_heavy[u] = 0;
    const float du = dist[u];
    if (!isfinite(du)) continue;

    for (Offset e = out_rowptr[u]; e < out_rowptr[u + 1]; ++e) {
      const float w = out_values[e];
      const int v = static_cast<int>(out_colind[e]);
      const float effective_w =
          vertex_costs == nullptr ? w : w * vertex_costs[v];
      if (effective_w <= delta) continue;
      const float nd = du + effective_w;
      const float old = atomic_min_float_nonnegative(&dist[v], nd);
      if (nd < old) {
        if (dist[v] == nd) {
          pred_node[v] = u;
          pred_edge[v] = e;
        }
        mark_touched(v, in_touched, touched_queue, touched_count,
                     queue_capacity, overflow);
        const int b = bucket_index(nd, inv_delta);
        if (b > current_bucket && b < kNoBucket) {
          enqueue_unique(v, in_pending, pending_queue, pending_count, queue_capacity, overflow);
        }
      }
    }
  }
#else
  for (int fi = blockIdx.x; fi < heavy_count_value; fi += gridDim.x) {
    const int u = heavy_vertices[fi];
    if (threadIdx.x == 0) {
      in_heavy[u] = 0;
    }
    __syncthreads();
    const float du = dist[u];
    if (isfinite(du)) {
      for (Offset e = out_rowptr[u] + threadIdx.x; e < out_rowptr[u + 1]; e += blockDim.x) {
        const float w = out_values[e];
        const int v = static_cast<int>(out_colind[e]);
        const float effective_w =
            vertex_costs == nullptr ? w : w * vertex_costs[v];
        if (effective_w <= delta) continue;
        const float nd = du + effective_w;
        const float old = atomic_min_float_nonnegative(&dist[v], nd);
        if (nd < old) {
          if (dist[v] == nd) {
            pred_node[v] = u;
            pred_edge[v] = e;
          }
          mark_touched(v, in_touched, touched_queue, touched_count,
                       queue_capacity, overflow);
          const int b = bucket_index(nd, inv_delta);
          if (b > current_bucket && b < kNoBucket) {
            enqueue_unique(v, in_pending, pending_queue, pending_count, queue_capacity, overflow);
          }
        }
      }
    }
    __syncthreads();
  }
#endif
}

__global__ void reset_touched_vertices_kernel(const int* touched_queue,
                                              int touched_count,
                                              float inf,
                                              float* dist,
                                              int* in_current,
                                              int* in_pending,
                                              int* in_heavy,
                                              int* in_touched,
                                              int* pred_node,
                                              Offset* pred_edge) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    dist[v] = inf;
    in_current[v] = 0;
    in_pending[v] = 0;
    in_heavy[v] = 0;
    in_touched[v] = 0;
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
                                                 int* min_bucket) {
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
    atomicMin(min_bucket, s_min[0]);
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
                                                         int* new_pending_count,
                                                         int queue_capacity,
                                                         int* overflow) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < pending_count;
       i += blockDim.x * gridDim.x) {
    const int v = pending_in[i];
    if (!in_pending[v]) continue;
    const int b = bucket_index(dist[v], inv_delta);
    if (b == selected_bucket) {
      atomicExch(&in_pending[v], 0);
      enqueue_unique(v, in_current, current_queue, current_count, queue_capacity, overflow);
    } else if (b > selected_bucket && b < kNoBucket) {
      const int pos = atomicAdd(new_pending_count, 1);
      if (pos < queue_capacity) pending_out[pos] = v;
      else atomicExch(overflow, 1);
    } else {
      atomicExch(&in_pending[v], 0);
    }
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

OutgoingCsrOwner build_outgoing_csr_from_incoming(const minplus_sparse::DeviceCsrF32& in,
                                                  hipStream_t stream) {
  OutgoingCsrOwner out(in.rows, in.cols, in.nnz);
  DeviceBuffer<Offset> degree(static_cast<std::size_t>(in.rows));
  DeviceBuffer<Offset> cursor(static_cast<std::size_t>(in.rows));

  DS_DELTA_HIP_CHECK(hipMemsetAsync(degree.get(), 0,
                                    static_cast<std::size_t>(in.rows) * sizeof(Offset), stream));
  count_out_degrees_from_incoming_kernel<<<grid_for_items(in.rows), kBlockSize, 0, stream>>>(
      in.rows, in.rowptr, in.colind, degree.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());

  std::size_t temp_bytes = 0;
  const std::size_t scan_items = static_cast<std::size_t>(in.rows);

  // Same rocPRIM argument order used by bf_hip_CSR_device_utils.hpp:
  //   input, output, initial_value, size, scan_op, stream.
  DS_DELTA_HIP_CHECK(rocprim::exclusive_scan(nullptr,
                                             temp_bytes,
                                             degree.get(),
                                             out.rowptr.get(),
                                             static_cast<Offset>(0),
                                             scan_items,
                                             rocprim::plus<Offset>(),
                                             stream));
  DeviceBuffer<unsigned char> temp(temp_bytes);
  DS_DELTA_HIP_CHECK(rocprim::exclusive_scan(temp.get(),
                                             temp_bytes,
                                             degree.get(),
                                             out.rowptr.get(),
                                             static_cast<Offset>(0),
                                             scan_items,
                                             rocprim::plus<Offset>(),
                                             stream));

  set_last_rowptr_kernel<<<1, 1, 0, stream>>>(out.rowptr.get(), in.rows, in.nnz);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(cursor.get(), out.rowptr.get(),
                                    static_cast<std::size_t>(in.rows) * sizeof(Offset),
                                    hipMemcpyDeviceToDevice, stream));

  if (in.nnz != 0) {
    fill_outgoing_from_incoming_kernel<<<grid_for_items(in.rows), kBlockSize, 0, stream>>>(
        in.rows, in.rowptr, in.colind, in.values, cursor.get(), out.colind.get(),
        out.values.get(), out.incoming_edge.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  return out;
}

void update_outgoing_values_from_incoming(OutgoingCsrOwner& outgoing,
                                          const minplus_sparse::DeviceCsrF32& incoming,
                                          hipStream_t stream) {
  if (incoming.nnz == 0) {
    return;
  }
  update_outgoing_values_from_incoming_kernel<<<grid_for_items(incoming.nnz),
                                                kBlockSize,
                                                0,
                                                stream>>>(
      incoming.nnz, incoming.values, outgoing.incoming_edge.get(), outgoing.values.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
}

void initialize_scratch_once(DeltaSteppingScratch& scratch,
                             Offset n,
                             float inf,
                             hipStream_t stream) {
  if (scratch.initialized) {
    reset_int_zero_async(scratch.current_count.get(), stream);
    reset_int_zero_async(scratch.next_count.get(), stream);
    reset_int_zero_async(scratch.pending_count.get(), stream);
    reset_int_zero_async(scratch.new_pending_count.get(), stream);
    reset_int_zero_async(scratch.heavy_count.get(), stream);
    reset_int_zero_async(scratch.touched_count.get(), stream);
    reset_overflow_async(scratch.overflow.get(), stream);
    return;
  }

  initialize_delta_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, scratch.dist.get(), scratch.in_current.get(), scratch.in_pending.get(),
      scratch.in_heavy.get(), scratch.current_count.get(), scratch.next_count.get(),
      scratch.pending_count.get(), scratch.heavy_count.get(), scratch.overflow.get(),
      scratch.pred_node.get(), scratch.pred_edge.get(), scratch.in_touched.get(),
      scratch.touched_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  scratch.initialized = true;
}

void reset_touched_vertices(DeltaSteppingScratch& scratch,
                            float inf,
                            hipStream_t stream) {
  const int touched_count = copy_scalar_to_host(scratch.touched_count.get(), stream);
  if (touched_count > 0) {
    reset_touched_vertices_kernel<<<grid_for_items(touched_count), kBlockSize, 0, stream>>>(
        scratch.touched_queue.get(), touched_count, inf, scratch.dist.get(),
        scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
        scratch.in_touched.get(), scratch.pred_node.get(), scratch.pred_edge.get());
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

void copy_predecessors_to_result(DeltaSteppingThreadsCsrResult& result,
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

void extract_target_paths_to_result(DeltaSteppingThreadsCsrResult& result,
                                    const minplus_sparse::DeviceCsrF32& graph,
                                    DeltaSteppingScratch& scratch,
                                    const std::vector<int>& targets,
                                    hipStream_t stream) {
  const int target_count = static_cast<int>(targets.size());
  measure_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(), target_count, scratch.rows, scratch.dist.get(),
      scratch.pred_node.get(), scratch.target_distances.get(),
      scratch.target_path_lengths.get(), scratch.target_sources.get(),
      scratch.target_path_status.get());
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
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.target_sources.data(),
                                    scratch.target_sources.get(),
                                    targets.size() * sizeof(int),
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
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  result.target_reached = all_targets_reached;
}

void throw_if_overflow(const int* d_overflow, hipStream_t stream) {
#if DS_CHECK_QUEUE_OVERFLOW
  if (copy_scalar_to_host(d_overflow, stream) != 0) {
    throw std::runtime_error("delta-stepping unique frontier queue overflow");
  }
#else
  (void)d_overflow;
  (void)stream;
#endif
}

int find_min_pending_bucket(const int* d_pending_queue,
                            int pending_count,
                            int current_bucket,
                            float inv_delta,
                            const float* d_dist,
                            const int* d_in_pending,
                            int* d_min_bucket,
                            hipStream_t stream) {
  if (pending_count <= 0) return kNoBucket;
  const int blocks = std::min(kMaxGridX, std::max(1, (pending_count + kBlockSize - 1) / kBlockSize));
  DS_DELTA_HIP_CHECK(hipMemsetAsync(d_min_bucket, 0x7f, sizeof(int), stream));
  reduce_min_pending_bucket_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_pending_queue, pending_count, current_bucket, inv_delta, d_dist,
      d_in_pending, d_min_bucket);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  const int min_bucket = copy_scalar_to_host(d_min_bucket, stream);
  return min_bucket < kNoBucket ? min_bucket : kNoBucket;
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
  return copy_scalar_to_host(scratch.settled_target_count.get(), stream);
}

DeltaSteppingThreadsCsrResult run_delta_stepping_impl(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    DeltaSteppingScratch& scratch,
    const std::vector<int>& sources,
    int target,
    const std::vector<int>* targets,
    const float* vertex_costs,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  if (max_iters < 0) max_iters = std::numeric_limits<int>::max();

  const Offset n = d_adjacency.rows;
  const int n_int = static_cast<int>(n);
  const int source_count = static_cast<int>(sources.size());
  const bool use_target_set = targets != nullptr;
  const int target_count = use_target_set ? static_cast<int>(targets->size()) : 0;
  const float inv_delta = 1.0f / delta;
  const float inf = std::numeric_limits<float>::infinity();
  const bool target_is_source =
      !use_target_set &&
      target >= 0 && std::find(sources.begin(), sources.end(), target) != sources.end();

  scratch.ensure_source_capacity(static_cast<std::size_t>(source_count));
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
  initialize_scratch_once(scratch, n, inf, stream);
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(), sources.data(),
                                    static_cast<std::size_t>(source_count) * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
  initialize_delta_sources_kernel<<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
      scratch.sources.get(), source_count, scratch.dist.get(), scratch.in_current.get(),
      scratch.current_queue.get(), scratch.current_count.get(),
      scratch.pred_node.get(), scratch.pred_edge.get(),
      scratch.in_touched.get(), scratch.touched_queue.get(), scratch.touched_count.get(),
      n_int, scratch.overflow.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  throw_if_overflow(scratch.overflow.get(), stream);

  int* current_queue = scratch.current_queue.get();
  int* next_queue = scratch.next_queue.get();
  int* pending_queue = scratch.pending_a.get();
  int* pending_scratch = scratch.pending_b.get();
  int current_bucket = 0;
  int current_count = copy_scalar_to_host(scratch.current_count.get(), stream);
  int pending_count = 0;
  DeltaSteppingThreadsCsrResult result;
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
      reset_overflow_async(scratch.overflow.get(), stream);

#if DS_USE_LOWDEG_RELAX
      const int light_grid = grid_for_items(current_count);
#else
      const int light_grid = grid_for_frontier(current_count);
#endif
      relax_light_edges_kernel<<<light_grid, kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta, inv_delta,
          d_adjacency.rowptr, d_adjacency.colind, d_adjacency.values,
          vertex_costs, scratch.dist.get(),
          scratch.in_current.get(), scratch.in_pending.get(), scratch.in_heavy.get(),
          scratch.pred_node.get(), scratch.pred_edge.get(),
          scratch.in_touched.get(), scratch.touched_queue.get(), scratch.touched_count.get(),
          next_queue, scratch.next_count.get(),
          pending_queue, scratch.pending_count.get(), scratch.heavy_queue.get(),
          scratch.heavy_count.get(), n_int, scratch.overflow.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
      throw_if_overflow(scratch.overflow.get(), stream);
      current_count = copy_scalar_to_host(scratch.next_count.get(), stream);
      std::swap(current_queue, next_queue);
    }

    const int heavy_count = copy_scalar_to_host(scratch.heavy_count.get(), stream);
    if (heavy_count > 0) {
      reset_overflow_async(scratch.overflow.get(), stream);
#if DS_USE_LOWDEG_RELAX
      const int heavy_grid = grid_for_items(heavy_count);
#else
      const int heavy_grid = grid_for_frontier(heavy_count);
#endif
      relax_heavy_edges_kernel<<<heavy_grid, kBlockSize, 0, stream>>>(
          scratch.heavy_queue.get(), heavy_count, current_bucket, delta, inv_delta,
          d_adjacency.rowptr, d_adjacency.colind, d_adjacency.values,
          vertex_costs, scratch.dist.get(),
          scratch.in_heavy.get(), scratch.in_pending.get(), scratch.pred_node.get(),
          scratch.pred_edge.get(), scratch.in_touched.get(), scratch.touched_queue.get(),
          scratch.touched_count.get(), pending_queue,
          scratch.pending_count.get(), n_int, scratch.overflow.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
      throw_if_overflow(scratch.overflow.get(), stream);
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
          DeltaSteppingThreadsCsrProgress progress;
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
          DeltaSteppingThreadsCsrProgress progress;
          progress.iteration = result.iterations_used;
          progress.max_iters = max_iters;
          progress.convergence_checked = true;
          progress.changed = true;
          progress_callback(progress, progress_user_data);
        }
        break;
      }
    }

    pending_count = copy_scalar_to_host(scratch.pending_count.get(), stream);
    const int next_bucket =
        find_min_pending_bucket(pending_queue, pending_count, current_bucket,
                                inv_delta, scratch.dist.get(), scratch.in_pending.get(),
                                scratch.min_pending_bucket.get(), stream);
    const bool changed = (next_bucket != kNoBucket);
    if (progress_callback) {
      DeltaSteppingThreadsCsrProgress progress;
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
    reset_overflow_async(scratch.overflow.get(), stream);

    compact_pending_to_current_bucket_kernel<<<grid_for_items(pending_count), kBlockSize, 0, stream>>>(
        pending_queue, pending_count, current_bucket, inv_delta, scratch.dist.get(),
        scratch.in_pending.get(), scratch.in_current.get(), current_queue,
        scratch.current_count.get(), pending_scratch, scratch.new_pending_count.get(),
        n_int, scratch.overflow.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    throw_if_overflow(scratch.overflow.get(), stream);

    current_count = copy_scalar_to_host(scratch.current_count.get(), stream);
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(scratch.pending_count.get(),
                                      scratch.new_pending_count.get(),
                                      sizeof(int),
                                      hipMemcpyDeviceToDevice,
                                      stream));
    std::swap(pending_queue, pending_scratch);
  }

  if (use_target_set) {
    extract_target_paths_to_result(result, d_adjacency, scratch, *targets, stream);
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
  reset_touched_vertices(scratch, inf, stream);
  return result;
}

}  // namespace ds_delta_threads_detail

struct DeltaSteppingThreadsCsrWorkspace::Impl {
  ds_delta_threads_detail::DeviceCsrOwner adjacency;
  ds_delta_threads_detail::DeltaSteppingScratch scratch;
  ds_delta_threads_detail::DeviceBuffer<float> vertex_costs;
  bool has_vertex_costs = false;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : adjacency(ds_delta_threads_detail::copy_host_csr_to_device(host, stream)),
        scratch(host.rows),
        vertex_costs(static_cast<std::size_t>(host.rows)) {}
};

DeltaSteppingThreadsCsrWorkspace::DeltaSteppingThreadsCsrWorkspace(const HostCsrF32& adjacency,
                                                     hipStream_t stream) {
  using namespace ds_delta_threads_detail;
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

DeltaSteppingThreadsCsrWorkspace::~DeltaSteppingThreadsCsrWorkspace() = default;
DeltaSteppingThreadsCsrWorkspace::DeltaSteppingThreadsCsrWorkspace(
    DeltaSteppingThreadsCsrWorkspace&&) noexcept = default;
DeltaSteppingThreadsCsrWorkspace& DeltaSteppingThreadsCsrWorkspace::operator=(
    DeltaSteppingThreadsCsrWorkspace&&) noexcept = default;

void DeltaSteppingThreadsCsrWorkspace::update_values(const std::vector<float>& values,
                                              hipStream_t stream) {
  using namespace ds_delta_threads_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingThreadsCsrWorkspace has no implementation");
  }
  if (values.size() != static_cast<std::size_t>(impl_->adjacency.view.nnz)) {
    throw std::invalid_argument("updated CSR values size does not match workspace nnz");
  }
  if (impl_->adjacency.view.nnz == 0) {
    return;
  }
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(impl_->adjacency.values.get(),
                                    values.data(),
                                    values.size() * sizeof(float),
                                    hipMemcpyHostToDevice,
                                    stream));
}

void DeltaSteppingThreadsCsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  using namespace ds_delta_threads_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingThreadsCsrWorkspace has no implementation");
  }
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

DeltaSteppingThreadsCsrResult DeltaSteppingThreadsCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_threads_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingThreadsCsrWorkspace has no implementation");
  }
  validate_device_csr_shape(impl_->adjacency.view, sources, target, delta);
  return run_delta_stepping_impl(impl_->adjacency.view,
                                 impl_->scratch,
                                 sources,
                                 target,
                                 nullptr,
                                 impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
                                 delta,
                                 max_iters,
                                 stream,
                                 progress_callback,
                                 progress_user_data);
}

DeltaSteppingThreadsCsrResult DeltaSteppingThreadsCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_threads_detail;
  if (!impl_) {
    throw std::runtime_error("DeltaSteppingThreadsCsrWorkspace has no implementation");
  }
  validate_device_csr_shape(impl_->adjacency.view, sources, -1, delta);
  validate_target_list_common_shape(impl_->adjacency.view.rows, targets);
  return run_delta_stepping_impl(impl_->adjacency.view,
                                 impl_->scratch,
                                 sources,
                                 -1,
                                 &targets,
                                 impl_->has_vertex_costs ? impl_->vertex_costs.get() : nullptr,
                                 delta,
                                 max_iters,
                                 stream,
                                 progress_callback,
                                 progress_user_data);
}

DeltaSteppingThreadsCsrResult DeltaSteppingThreadsCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_threads_detail;

  validate_device_csr_shape(d_adjacency, sources, target, delta);
  validate_device_csr_contents(d_adjacency, stream);
  DeltaSteppingScratch scratch(d_adjacency.rows);
  return run_delta_stepping_impl(d_adjacency, scratch, sources, target,
                                 nullptr, nullptr, delta, max_iters, stream, progress_callback,
                                 progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_threads_minplus_hip_csr(d_adjacency,
                                        std::vector<int>{source},
                                        target,
                                        delta,
                                        max_iters,
                                        stream,
                                        progress_callback,
                                        progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_threads_minplus_hip_csr(d_adjacency, source, -1, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream) {
  return delta_stepping_threads_minplus_hip_csr(d_adjacency, source, delta, -1, stream, nullptr, nullptr);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_threads_detail;
  validate_host_csr(adjacency, source, target, delta);
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
  return delta_stepping_threads_minplus_hip_csr(d_adjacency.view, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  using namespace ds_delta_threads_detail;
  validate_host_csr(adjacency, sources, target, delta);
  DeviceCsrOwner d_adjacency = copy_host_csr_to_device(adjacency, stream);
  return delta_stepping_threads_minplus_hip_csr(d_adjacency.view, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream,
    DeltaSteppingThreadsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return delta_stepping_threads_minplus_hip_csr(adjacency, source, -1, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream) {
  return delta_stepping_threads_minplus_hip_csr(adjacency, source, delta, -1, stream, nullptr, nullptr);
}
