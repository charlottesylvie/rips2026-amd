#include "delta_stepping_hip_CSR.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ds_delta_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kNoBucket = 0x3fffffff;

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

struct OutgoingCsrOwner {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  DeviceBuffer<Offset> rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;

  OutgoingCsrOwner() = default;
  OutgoingCsrOwner(Offset rows_, Offset cols_, Offset nnz_)
      : rows(rows_),
        cols(cols_),
        nnz(nnz_),
        rowptr(static_cast<std::size_t>(rows_) + 1),
        colind(static_cast<std::size_t>(nnz_)),
        values(static_cast<std::size_t>(nnz_)) {}
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
  const float bucket_f = std::floor(distance * inv_delta);
  if (bucket_f < 0.0f) return 0;
  if (bucket_f >= static_cast<float>(kNoBucket)) return kNoBucket - 1;
  return static_cast<int>(bucket_f);
}

__device__ inline int bucket_index(float distance, float inv_delta) {
  if (!isfinite(distance)) return kNoBucket;
  const float bucket_f = floorf(distance * inv_delta);
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
                                                   float* out_values) {
  // Transpose incoming-edge CSR into outgoing CSR: output row src stores dst.
  for (Offset dst = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       dst < rows;
       dst += static_cast<Offset>(blockDim.x) * gridDim.x) {
    for (Offset e = in_rowptr[dst]; e < in_rowptr[dst + 1]; ++e) {
      const Offset src = static_cast<Offset>(in_colind[e]);
      const Offset pos = atomic_add_offset(&cursor[src], static_cast<Offset>(1));
      out_colind[pos] = static_cast<Index>(dst);
      out_values[pos] = in_values[e];
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
                                                int* overflow) {
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
    *overflow = 0;
  }
}

__global__ void initialize_delta_sources_kernel(const int* sources,
                                                int source_count,
                                                float* dist,
                                                int* in_current,
                                                int* current_queue,
                                                int* current_count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    dist[source] = 0.0f;
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
        const float candidate = pred_dist + values[edge];
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

__global__ void relax_light_edges_kernel(const int* frontier,
                                         int frontier_count,
                                         int current_bucket,
                                         float delta,
                                         float inv_delta,
                                         const Offset* out_rowptr,
                                         const Index* out_colind,
                                         const float* out_values,
                                         float* dist,
                                         int* in_current,
                                         int* in_pending,
                                         int* in_heavy,
                                         int* next_frontier,
                                         int* next_count,
                                         int* pending_queue,
                                         int* pending_count,
                                         int* heavy_queue,
                                         int* heavy_count,
                                         int queue_capacity,
                                         int* overflow) {
  // One block cooperates on one active vertex at a time; grid-stride over frontier.
  for (int fi = blockIdx.x; fi < frontier_count; fi += gridDim.x) {
    const int u = frontier[fi];
    // The in_current flags for this whole frontier are cleared by a separate
    // kernel before relaxation starts.  Do not clear them here: doing so races
    // with other blocks that may need to re-enqueue this vertex after a same-
    // bucket distance decrease.
    const float du = dist[u];
    const bool active = isfinite(du) && bucket_index(du, inv_delta) == current_bucket;
    if (active) {
      if (threadIdx.x == 0) {
        enqueue_unique(u, in_heavy, heavy_queue, heavy_count, queue_capacity, overflow);
      }
      for (Offset e = out_rowptr[u] + threadIdx.x; e < out_rowptr[u + 1]; e += blockDim.x) {
        const float w = out_values[e];
        if (w > delta) continue;
        const int v = static_cast<int>(out_colind[e]);
        const float nd = du + w;
        const float old = atomic_min_float_nonnegative(&dist[v], nd);
        if (nd < old) {
          const int b = bucket_index(nd, inv_delta);
          if (b == current_bucket) {
            // If v had previously been discovered in a later bucket, the new
            // distance has moved it back into the current bucket.  Clear the
            // pending flag so stale pending-queue entries cannot affect later
            // bucket selection.
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
}

__global__ void relax_heavy_edges_kernel(const int* heavy_vertices,
                                         int heavy_count_value,
                                         int current_bucket,
                                         float delta,
                                         float inv_delta,
                                         const Offset* out_rowptr,
                                         const Index* out_colind,
                                         const float* out_values,
                                         float* dist,
                                         int* in_pending,
                                         int* pending_queue,
                                         int* pending_count,
                                         int queue_capacity,
                                         int* overflow) {
  for (int fi = blockIdx.x; fi < heavy_count_value; fi += gridDim.x) {
    const int u = heavy_vertices[fi];
    const float du = dist[u];
    if (isfinite(du)) {
      for (Offset e = out_rowptr[u] + threadIdx.x; e < out_rowptr[u + 1]; e += blockDim.x) {
        const float w = out_values[e];
        if (w <= delta) continue;
        const int v = static_cast<int>(out_colind[e]);
        const float nd = du + w;
        const float old = atomic_min_float_nonnegative(&dist[v], nd);
        if (nd < old) {
          const int b = bucket_index(nd, inv_delta);
          if (b > current_bucket && b < kNoBucket) {
            enqueue_unique(v, in_pending, pending_queue, pending_count, queue_capacity, overflow);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void clear_flags_from_queue_kernel(const int* vertices, int count, int* flags) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count;
       i += blockDim.x * gridDim.x) {
    flags[vertices[i]] = 0;
  }
}

__global__ void reduce_min_pending_bucket_kernel(const int* pending_queue,
                                                 int pending_count,
                                                 int previous_bucket,
                                                 float inv_delta,
                                                 const float* dist,
                                                 const int* in_pending,
                                                 int* block_mins) {
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
  if (tid == 0) block_mins[blockIdx.x] = s_min[0];
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
        in.rows, in.rowptr, in.colind, in.values, cursor.get(), out.colind.get(), out.values.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
  }
  return out;
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
                                 const float* d_dist,
                                 hipStream_t stream) {
  DeviceBuffer<int> d_pred_node(static_cast<std::size_t>(graph.rows));
  DeviceBuffer<Offset> d_pred_edge(static_cast<std::size_t>(graph.rows));
  compute_predecessors_kernel<<<grid_for_items(graph.rows), kBlockSize, 0, stream>>>(
      graph.rows, graph.rowptr, graph.colind, graph.values, d_dist,
      d_pred_node.get(), d_pred_edge.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());

  result.pred_node.resize(static_cast<std::size_t>(graph.rows));
  result.pred_edge.resize(static_cast<std::size_t>(graph.rows));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_node.data(), d_pred_node.get(),
                                    static_cast<std::size_t>(graph.rows) * sizeof(int),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(result.pred_edge.data(), d_pred_edge.get(),
                                    static_cast<std::size_t>(graph.rows) * sizeof(Offset),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
}

void throw_if_overflow(const int* d_overflow, hipStream_t stream) {
  if (copy_scalar_to_host(d_overflow, stream) != 0) {
    throw std::runtime_error("delta-stepping unique frontier queue overflow");
  }
}

int find_min_pending_bucket(const int* d_pending_queue,
                            int pending_count,
                            int current_bucket,
                            float inv_delta,
                            const float* d_dist,
                            const int* d_in_pending,
                            int* d_block_mins,
                            std::vector<int>& h_block_mins,
                            hipStream_t stream) {
  if (pending_count <= 0) return kNoBucket;
  const int blocks = std::min(kMaxGridX, std::max(1, (pending_count + kBlockSize - 1) / kBlockSize));
  reduce_min_pending_bucket_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_pending_queue, pending_count, current_bucket, inv_delta, d_dist, d_in_pending, d_block_mins);
  DS_DELTA_HIP_CHECK(hipGetLastError());
  h_block_mins.resize(static_cast<std::size_t>(blocks));
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(h_block_mins.data(), d_block_mins,
                                    static_cast<std::size_t>(blocks) * sizeof(int),
                                    hipMemcpyDeviceToHost, stream));
  DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));
  int min_bucket = kNoBucket;
  for (int b : h_block_mins) min_bucket = std::min(min_bucket, b);
  return min_bucket;
}

}  // namespace ds_delta_detail

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
  if (max_iters < 0) max_iters = std::numeric_limits<int>::max();

  const Offset n = d_adjacency.rows;
  const int n_int = static_cast<int>(n);
  const int source_count = static_cast<int>(sources.size());
  const float inv_delta = 1.0f / delta;
  const float inf = std::numeric_limits<float>::infinity();
  const bool target_is_source =
      target >= 0 && std::find(sources.begin(), sources.end(), target) != sources.end();

  OutgoingCsrOwner outgoing = build_outgoing_csr_from_incoming(d_adjacency, stream);

  DeviceBuffer<int> d_sources(static_cast<std::size_t>(source_count));
  DeviceBuffer<float> d_dist(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_in_current(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_in_pending(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_in_heavy(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_current_queue(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_next_queue(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_pending_a(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_pending_b(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_heavy_queue(static_cast<std::size_t>(n));
  DeviceBuffer<int> d_current_count(1), d_next_count(1), d_pending_count(1);
  DeviceBuffer<int> d_new_pending_count(1), d_heavy_count(1), d_overflow(1);
  DeviceBuffer<int> d_block_mins(static_cast<std::size_t>((n_int + kBlockSize - 1) / kBlockSize + 1));

  initialize_delta_arrays_kernel<<<grid_for_items(n), kBlockSize, 0, stream>>>(
      n, inf, d_dist.get(), d_in_current.get(), d_in_pending.get(), d_in_heavy.get(),
      d_current_count.get(), d_next_count.get(), d_pending_count.get(),
      d_heavy_count.get(), d_overflow.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());
  DS_DELTA_HIP_CHECK(hipMemcpyAsync(d_sources.get(), sources.data(),
                                    static_cast<std::size_t>(source_count) * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
  initialize_delta_sources_kernel<<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
      d_sources.get(), source_count, d_dist.get(), d_in_current.get(),
      d_current_queue.get(), d_current_count.get());
  DS_DELTA_HIP_CHECK(hipGetLastError());

  int* current_queue = d_current_queue.get();
  int* next_queue = d_next_queue.get();
  int* pending_queue = d_pending_a.get();
  int* pending_scratch = d_pending_b.get();
  int current_bucket = 0;
  int current_count = copy_scalar_to_host(d_current_count.get(), stream);
  int pending_count = 0;
  DeltaSteppingCsrResult result;
  result.target = target;
  if (target_is_source) {
    result.target_distance = 0.0f;
    result.target_reached = true;
    result.stopped_on_target = true;
    copy_predecessors_to_result(result, d_adjacency, d_dist.get(), stream);
    result.dist = copy_dist_to_host(d_dist.get(), n, stream);
    return result;
  }
  std::vector<int> h_block_mins;

  for (int iter = 0; iter < max_iters; ++iter) {
    reset_int_zero_async(d_heavy_count.get(), stream);

    while (current_count > 0) {
      reset_int_zero_async(d_next_count.get(), stream);
      reset_int_zero_async(d_overflow.get(), stream);

      // Correctness-critical: clear membership for every vertex in the current
      // frontier before any block starts relaxing light edges.  The previous
      // implementation cleared one vertex inside relax_light_edges_kernel;
      // other blocks could then observe stale in_current[v] == 1 and drop a
      // required same-bucket re-enqueue, causing premature bucket convergence.
      clear_flags_from_queue_kernel<<<grid_for_items(current_count), kBlockSize, 0, stream>>>(
          current_queue, current_count, d_in_current.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());

      relax_light_edges_kernel<<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
          current_queue, current_count, current_bucket, delta, inv_delta,
          outgoing.rowptr.get(), outgoing.colind.get(), outgoing.values.get(), d_dist.get(),
          d_in_current.get(), d_in_pending.get(), d_in_heavy.get(), next_queue, d_next_count.get(),
          pending_queue, d_pending_count.get(), d_heavy_queue.get(), d_heavy_count.get(),
          n_int, d_overflow.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
      throw_if_overflow(d_overflow.get(), stream);
      current_count = copy_scalar_to_host(d_next_count.get(), stream);
      std::swap(current_queue, next_queue);
    }

    const int heavy_count = copy_scalar_to_host(d_heavy_count.get(), stream);
    if (heavy_count > 0) {
      reset_int_zero_async(d_overflow.get(), stream);
      relax_heavy_edges_kernel<<<grid_for_frontier(heavy_count), kBlockSize, 0, stream>>>(
          d_heavy_queue.get(), heavy_count, current_bucket, delta, inv_delta,
          outgoing.rowptr.get(), outgoing.colind.get(), outgoing.values.get(), d_dist.get(),
          d_in_pending.get(), pending_queue, d_pending_count.get(), n_int, d_overflow.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
      throw_if_overflow(d_overflow.get(), stream);
      clear_flags_from_queue_kernel<<<grid_for_items(heavy_count), kBlockSize, 0, stream>>>(
          d_heavy_queue.get(), heavy_count, d_in_heavy.get());
      DS_DELTA_HIP_CHECK(hipGetLastError());
    }

    result.iterations_used = iter + 1;
    if (target >= 0) {
      const float target_distance = copy_dist_value_to_host(d_dist.get(), target, stream);
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

    pending_count = copy_scalar_to_host(d_pending_count.get(), stream);
    const int next_bucket = find_min_pending_bucket(pending_queue, pending_count, current_bucket,
                                                    inv_delta, d_dist.get(), d_in_pending.get(),
                                                    d_block_mins.get(), h_block_mins, stream);
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
    current_queue = d_current_queue.get();
    next_queue = d_next_queue.get();
    reset_int_zero_async(d_current_count.get(), stream);
    reset_int_zero_async(d_new_pending_count.get(), stream);
    reset_int_zero_async(d_overflow.get(), stream);

    compact_pending_to_current_bucket_kernel<<<grid_for_items(pending_count), kBlockSize, 0, stream>>>(
        pending_queue, pending_count, current_bucket, inv_delta, d_dist.get(), d_in_pending.get(),
        d_in_current.get(), current_queue, d_current_count.get(), pending_scratch,
        d_new_pending_count.get(), n_int, d_overflow.get());
    DS_DELTA_HIP_CHECK(hipGetLastError());
    throw_if_overflow(d_overflow.get(), stream);

    current_count = copy_scalar_to_host(d_current_count.get(), stream);
    DS_DELTA_HIP_CHECK(hipMemcpyAsync(d_pending_count.get(), d_new_pending_count.get(), sizeof(int),
                                      hipMemcpyDeviceToDevice, stream));
    std::swap(pending_queue, pending_scratch);
  }

  if (target >= 0) {
    copy_predecessors_to_result(result, d_adjacency, d_dist.get(), stream);
  }
  result.dist = copy_dist_to_host(d_dist.get(), n, stream);
  if (target >= 0) {
    result.target_distance = result.dist[static_cast<std::size_t>(target)];
    result.target_reached =
        !std::isinf(result.target_distance) &&
        (result.target_reached || result.converged);
  }
  return result;
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
