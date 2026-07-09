#include "unit_bfs_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace unit_bfs_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kUnvisited = std::numeric_limits<int>::max();

inline void hip_check(hipError_t status, const char* expr, const char* file, int line) {
  if (status != hipSuccess) {
    std::ostringstream os;
    os << "HIP error at " << file << ':' << line << " for " << expr << ": "
       << hipGetErrorString(status);
    throw std::runtime_error(os.str());
  }
}

#define UNIT_BFS_HIP_CHECK(expr) \
  ::unit_bfs_detail::hip_check((expr), #expr, __FILE__, __LINE__)

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
      UNIT_BFS_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
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

struct OutgoingCsrOwner {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  DeviceBuffer<Offset> rowptr;
  DeviceBuffer<Index> colind;

  OutgoingCsrOwner() = default;
  OutgoingCsrOwner(Offset rows_, Offset cols_, Offset nnz_)
      : rows(rows_),
        cols(cols_),
        nnz(nnz_),
        rowptr(static_cast<std::size_t>(rows_) + 1),
        colind(static_cast<std::size_t>(nnz_)) {}
};

struct UnitBfsScratch {
  Offset rows = 0;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<int> target_multiplicity;
  DeviceBuffer<int> level;
  DeviceBuffer<int> pred_node;
  DeviceBuffer<Offset> pred_edge;
  DeviceBuffer<int> current_queue;
  DeviceBuffer<int> next_queue;
  DeviceBuffer<int> touched_queue;
  DeviceBuffer<int> in_touched;
  DeviceBuffer<int> current_count;
  DeviceBuffer<int> next_count;
  DeviceBuffer<int> touched_count;
  DeviceBuffer<int> found_count;
  DeviceBuffer<int> overflow;
  DeviceBuffer<int> status;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_sources;
  DeviceBuffer<int> target_path_status;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  bool initialized = false;

  UnitBfsScratch() = default;
  explicit UnitBfsScratch(Offset rows_)
      : rows(rows_),
        target_multiplicity(static_cast<std::size_t>(rows_)),
        level(static_cast<std::size_t>(rows_)),
        pred_node(static_cast<std::size_t>(rows_)),
        pred_edge(static_cast<std::size_t>(rows_)),
        current_queue(static_cast<std::size_t>(rows_)),
        next_queue(static_cast<std::size_t>(rows_)),
        touched_queue(static_cast<std::size_t>(rows_)),
        in_touched(static_cast<std::size_t>(rows_)),
        current_count(1),
        next_count(1),
        touched_count(1),
        found_count(1),
        overflow(1),
        status(3) {}

  void ensure_source_capacity(std::size_t source_count) {
    if (sources.size() < source_count) {
      sources.reset(source_count);
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    if (targets.size() < target_count) {
      targets.reset(target_count);
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
  return std::min(kMaxGridX, std::max(1, (items + kBlockSize - 1) / kBlockSize));
}

inline std::size_t checked_size(Offset value, const char* name) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be nonnegative");
  }
  return static_cast<std::size_t>(value);
}

inline void validate_host_csr_arrays(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::invalid_argument("unit BFS expects a nonempty square CSR graph");
  }
  if (static_cast<unsigned long long>(graph.rows) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("unit BFS stores vertices as int; rows must fit in int");
  }
  const std::size_t rows = checked_size(graph.rows, "rows");
  const std::size_t nnz = checked_size(graph.nnz, "nnz");
  if (graph.rowptr.size() != rows + 1) {
    throw std::invalid_argument("CSR rowptr size must equal rows + 1");
  }
  if (graph.colind.size() != nnz || graph.values.size() != nnz) {
    throw std::invalid_argument("CSR colind and values sizes must equal nnz");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("CSR rowptr must start at 0 and end at nnz");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    if (graph.rowptr[row] < 0 ||
        graph.rowptr[row + 1] < graph.rowptr[row] ||
        graph.rowptr[row + 1] > graph.nnz) {
      throw std::invalid_argument("CSR rowptr must be monotone and within [0, nnz]");
    }
  }
  for (std::size_t edge = 0; edge < nnz; ++edge) {
    if (graph.colind[edge] < 0 || static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::invalid_argument("CSR colind contains an out-of-range destination vertex");
    }
    if (!std::isfinite(graph.values[edge]) ||
        std::fabs(graph.values[edge] - 1.0f) > 1e-5f) {
      throw std::invalid_argument("unit BFS requires all CSR edge weights to be exactly 1");
    }
  }
}

inline void validate_sources_targets(Offset rows,
                                     const std::vector<int>& sources,
                                     const std::vector<int>& targets) {
  if (sources.empty()) {
    throw std::invalid_argument("unit BFS requires at least one source");
  }
  if (targets.empty()) {
    throw std::invalid_argument("unit BFS requires at least one target");
  }
  if (sources.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      targets.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("source and target counts must fit in int");
  }
  for (const int source : sources) {
    if (source < 0 || static_cast<Offset>(source) >= rows) {
      throw std::out_of_range("unit BFS source vertex is outside CSR row range");
    }
  }
  for (const int target : targets) {
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      throw std::out_of_range("unit BFS target vertex is outside CSR row range");
    }
  }
}

template <typename T>
inline T copy_scalar_to_host(const T* d_value, hipStream_t stream) {
  T h{};
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(&h, d_value, sizeof(T), hipMemcpyDeviceToHost, stream));
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  return h;
}

inline void reset_int_zero_async(int* d_value, hipStream_t stream) {
  UNIT_BFS_HIP_CHECK(hipMemsetAsync(d_value, 0, sizeof(int), stream));
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

__device__ inline void count_target_if_reached(int v,
                                               const int* target_multiplicity,
                                               int* found_count) {
  const int multiplicity = target_multiplicity[v];
  if (multiplicity > 0) {
    atomicAdd(found_count, multiplicity);
  }
}

__global__ void initialize_bfs_arrays_kernel(Offset rows,
                                             int* level,
                                             int* pred_node,
                                             Offset* pred_edge,
                                             int* target_multiplicity,
                                             int* in_touched,
                                             int* current_count,
                                             int* next_count,
                                             int* touched_count,
                                             int* found_count,
                                             int* overflow) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    level[v] = kUnvisited;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
    target_multiplicity[v] = 0;
    in_touched[v] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *current_count = 0;
    *next_count = 0;
    *touched_count = 0;
    *found_count = 0;
    *overflow = 0;
  }
}

__global__ void reset_touched_vertices_kernel(const int* touched_queue,
                                              int touched_count,
                                              int* level,
                                              int* pred_node,
                                              Offset* pred_edge,
                                              int* in_touched) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < touched_count;
       i += blockDim.x * gridDim.x) {
    const int v = touched_queue[i];
    level[v] = kUnvisited;
    pred_node[v] = -1;
    pred_edge[v] = static_cast<Offset>(-1);
    in_touched[v] = 0;
  }
}

__global__ void initialize_sources_kernel(const int* sources,
                                          int source_count,
                                          int* level,
                                          int* pred_node,
                                          Offset* pred_edge,
                                          int* current_queue,
                                          int* current_count,
                                          const int* target_multiplicity,
                                          int* found_count,
                                          int* in_touched,
                                          int* touched_queue,
                                          int* touched_count,
                                          int queue_capacity,
                                          int* overflow) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    if (atomicCAS(&level[source], kUnvisited, 0) == kUnvisited) {
      pred_node[source] = source;
      pred_edge[source] = static_cast<Offset>(-1);
      mark_touched(source, in_touched, touched_queue, touched_count,
                   queue_capacity, overflow);
      const int pos = atomicAdd(current_count, 1);
      if (pos < queue_capacity) {
        current_queue[pos] = source;
      } else {
        atomicExch(overflow, 1);
      }
      count_target_if_reached(source, target_multiplicity, found_count);
    }
  }
}

__global__ void expand_frontier_kernel(const int* frontier,
                                       int frontier_count,
                                       int next_level,
                                       const Offset* out_rowptr,
                                       const Index* out_colind,
                                       int* level,
                                       int* pred_node,
                                       Offset* pred_edge,
                                       int* next_frontier,
                                       int* next_count,
                                       const int* target_multiplicity,
                                       int* found_count,
                                       int* in_touched,
                                       int* touched_queue,
                                       int* touched_count,
                                       int queue_capacity,
                                       int* overflow) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_count;
       i += blockDim.x * gridDim.x) {
    const int u = frontier[i];
    for (Offset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
      const int v = static_cast<int>(out_colind[edge]);
      if (atomicCAS(&level[v], kUnvisited, next_level) == kUnvisited) {
        pred_node[v] = u;
        pred_edge[v] = edge;
        mark_touched(v, in_touched, touched_queue, touched_count,
                     queue_capacity, overflow);
        const int pos = atomicAdd(next_count, 1);
        if (pos < queue_capacity) {
          next_frontier[pos] = v;
        } else {
          atomicExch(overflow, 1);
        }
        count_target_if_reached(v, target_multiplicity, found_count);
      }
    }
  }
}

__global__ void mark_target_multiplicity_kernel(const int* targets,
                                                int target_count,
                                                int* target_multiplicity) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&target_multiplicity[targets[i]], 1);
  }
}

__global__ void clear_target_multiplicity_kernel(const int* targets,
                                                 int target_count,
                                                 int* target_multiplicity) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    target_multiplicity[targets[i]] = 0;
  }
}

__global__ void pack_status_kernel(const int* queue_count,
                                   const int* found_count,
                                   const int* overflow,
                                   int* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status[0] = *queue_count;
    status[1] = *found_count;
    status[2] = *overflow;
  }
}

__global__ void measure_target_paths_kernel(const int* targets,
                                            int target_count,
                                            Offset rows,
                                            const int* level,
                                            const int* pred_node,
                                            float* target_distances,
                                            int* path_lengths,
                                            int* path_sources,
                                            int* path_status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    const int target_level = level[target];
    target_distances[i] =
        target_level == kUnvisited ? INFINITY : static_cast<float>(target_level);
    path_lengths[i] = 0;
    path_sources[i] = -1;
    path_status[i] = 0;
    if (target_level == kUnvisited) {
      continue;
    }

    int current = target;
    int length = 1;
    for (Offset guard = 0; guard < rows; ++guard) {
      const int pred = pred_node[current];
      if (pred == current) {
        path_lengths[i] = length;
        path_sources[i] = current;
        path_status[i] = 1;
        break;
      }
      if (pred < 0 || static_cast<Offset>(pred) >= rows) {
        break;
      }
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
    const int node_begin = node_offsets[i];
    const int edge_begin = edge_offsets[i];
    int current = targets[i];
    path_nodes[node_begin + length - 1] = current;

    for (int j = length - 1; j > 0; --j) {
      const int pred = pred_node[current];
      if (pred < 0 || static_cast<Offset>(pred) >= rows || pred == current) {
        return;
      }
      path_edges[edge_begin + j - 1] = pred_edge[current];
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
  }
}

OutgoingCsrOwner copy_host_csr_to_device(const HostCsrF32& host, hipStream_t stream) {
  OutgoingCsrOwner device(host.rows, host.cols, host.nnz);
  const std::size_t rows = checked_size(host.rows, "rows");
  const std::size_t nnz = checked_size(host.nnz, "nnz");
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.rowptr.get(),
                                    host.rowptr.data(),
                                    (rows + 1) * sizeof(Offset),
                                    hipMemcpyHostToDevice,
                                    stream));
  if (nnz != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.colind.get(),
                                      host.colind.data(),
                                      nnz * sizeof(Index),
                                      hipMemcpyHostToDevice,
                                      stream));
  }
  return device;
}

void initialize_scratch_once(UnitBfsScratch& scratch, Offset rows, hipStream_t stream) {
  if (scratch.initialized) {
    reset_int_zero_async(scratch.current_count.get(), stream);
    reset_int_zero_async(scratch.next_count.get(), stream);
    reset_int_zero_async(scratch.touched_count.get(), stream);
    reset_int_zero_async(scratch.found_count.get(), stream);
    reset_int_zero_async(scratch.overflow.get(), stream);
    return;
  }

  initialize_bfs_arrays_kernel<<<grid_for_items(rows), kBlockSize, 0, stream>>>(
      rows,
      scratch.level.get(),
      scratch.pred_node.get(),
      scratch.pred_edge.get(),
      scratch.target_multiplicity.get(),
      scratch.in_touched.get(),
      scratch.current_count.get(),
      scratch.next_count.get(),
      scratch.touched_count.get(),
      scratch.found_count.get(),
      scratch.overflow.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  scratch.initialized = true;
}

void reset_touched_vertices(UnitBfsScratch& scratch, hipStream_t stream) {
  const int touched_count = copy_scalar_to_host(scratch.touched_count.get(), stream);
  if (touched_count > 0) {
    reset_touched_vertices_kernel<<<grid_for_items(touched_count),
                                    kBlockSize,
                                    0,
                                    stream>>>(
        scratch.touched_queue.get(),
        touched_count,
        scratch.level.get(),
        scratch.pred_node.get(),
        scratch.pred_edge.get(),
        scratch.in_touched.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
  }
  reset_int_zero_async(scratch.touched_count.get(), stream);
}

std::array<int, 3> copy_status_to_host(const int* queue_count,
                                       UnitBfsScratch& scratch,
                                       hipStream_t stream) {
  pack_status_kernel<<<1, 1, 0, stream>>>(
      queue_count, scratch.found_count.get(), scratch.overflow.get(), scratch.status.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  std::array<int, 3> status{};
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(status.data(),
                                    scratch.status.get(),
                                    status.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  return status;
}

void throw_if_overflow(const std::array<int, 3>& status) {
  if (status[2] != 0) {
    throw std::runtime_error("unit BFS frontier/touched queue overflow");
  }
}

void extract_target_paths_to_result(UnitBfsCsrResult& result,
                                    UnitBfsScratch& scratch,
                                    const std::vector<int>& targets,
                                    hipStream_t stream) {
  const int target_count = static_cast<int>(targets.size());
  measure_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.rows,
      scratch.level.get(),
      scratch.pred_node.get(),
      scratch.target_distances.get(),
      scratch.target_path_lengths.get(),
      scratch.target_sources.get(),
      scratch.target_path_status.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  result.target_distances.resize(targets.size());
  result.target_sources.resize(targets.size());
  std::vector<int> path_lengths(targets.size());
  std::vector<int> path_status(targets.size());

  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_distances.data(),
                                    scratch.target_distances.get(),
                                    targets.size() * sizeof(float),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_sources.data(),
                                    scratch.target_sources.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(path_lengths.data(),
                                    scratch.target_path_lengths.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(path_status.data(),
                                    scratch.target_path_status.get(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));

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
      throw std::overflow_error("compact unit BFS target paths are too large");
    }
  }
  result.target_path_offsets[targets.size()] = static_cast<int>(total_nodes);
  result.target_edge_offsets[targets.size()] = static_cast<int>(total_edges);

  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.target_node_offsets.get(),
                                    result.target_path_offsets.data(),
                                    (targets.size() + 1) * sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.target_edge_offsets.get(),
                                    result.target_edge_offsets.data(),
                                    (targets.size() + 1) * sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));

  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    fill_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
        scratch.targets.get(),
        target_count,
        scratch.rows,
        scratch.pred_node.get(),
        scratch.pred_edge.get(),
        scratch.target_path_lengths.get(),
        scratch.target_path_status.get(),
        scratch.target_node_offsets.get(),
        scratch.target_edge_offsets.get(),
        scratch.compact_path_nodes.get(),
        scratch.compact_path_edges.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_path_nodes.data(),
                                      scratch.compact_path_nodes.get(),
                                      total_nodes * sizeof(int),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  if (total_edges != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_path_edges.data(),
                                      scratch.compact_path_edges.get(),
                                      total_edges * sizeof(Offset),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  result.target_reached = all_targets_reached;
}

UnitBfsCsrResult run_unit_bfs_impl(const OutgoingCsrOwner& outgoing,
                                   UnitBfsScratch& scratch,
                                   const std::vector<int>& sources,
                                   const std::vector<int>& targets,
                                   int max_depth,
                                   hipStream_t stream,
                                   UnitBfsCsrProgressCallback progress_callback,
                                   void* progress_user_data) {
  if (max_depth < 0) {
    max_depth = static_cast<int>(scratch.rows);
  }

  const int n_int = static_cast<int>(scratch.rows);
  const int source_count = static_cast<int>(sources.size());
  const int target_count = static_cast<int>(targets.size());
  scratch.ensure_source_capacity(sources.size());
  scratch.ensure_target_capacity(targets.size());

  initialize_scratch_once(scratch, scratch.rows, stream);
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(),
                                    sources.data(),
                                    sources.size() * sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.targets.get(),
                                    targets.data(),
                                    targets.size() * sizeof(int),
                                    hipMemcpyHostToDevice,
                                    stream));

  mark_target_multiplicity_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.target_multiplicity.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  initialize_sources_kernel<<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
      scratch.sources.get(),
      source_count,
      scratch.level.get(),
      scratch.pred_node.get(),
      scratch.pred_edge.get(),
      scratch.current_queue.get(),
      scratch.current_count.get(),
      scratch.target_multiplicity.get(),
      scratch.found_count.get(),
      scratch.in_touched.get(),
      scratch.touched_queue.get(),
      scratch.touched_count.get(),
      n_int,
      scratch.overflow.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  std::array<int, 3> status =
      copy_status_to_host(scratch.current_count.get(), scratch, stream);
  throw_if_overflow(status);
  int current_count = status[0];
  int found_count = status[1];

  int* current_queue = scratch.current_queue.get();
  int* next_queue = scratch.next_queue.get();
  UnitBfsCsrResult result;
  result.target = -1;
  result.iterations_used = 0;

  for (int depth = 0;
       current_count > 0 && found_count < target_count && depth < max_depth;
       ++depth) {
    reset_int_zero_async(scratch.next_count.get(), stream);
    reset_int_zero_async(scratch.overflow.get(), stream);

    expand_frontier_kernel<<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
        current_queue,
        current_count,
        depth + 1,
        outgoing.rowptr.get(),
        outgoing.colind.get(),
        scratch.level.get(),
        scratch.pred_node.get(),
        scratch.pred_edge.get(),
        next_queue,
        scratch.next_count.get(),
        scratch.target_multiplicity.get(),
        scratch.found_count.get(),
        scratch.in_touched.get(),
        scratch.touched_queue.get(),
        scratch.touched_count.get(),
        n_int,
        scratch.overflow.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());

    status = copy_status_to_host(scratch.next_count.get(), scratch, stream);
    throw_if_overflow(status);
    current_count = status[0];
    found_count = status[1];
    result.iterations_used = depth + 1;

    if (progress_callback) {
      UnitBfsCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_depth;
      progress.convergence_checked = true;
      progress.changed = current_count > 0;
      progress_callback(progress, progress_user_data);
    }

    std::swap(current_queue, next_queue);
  }

  result.converged = current_count == 0 || found_count >= target_count;
  result.stopped_on_target = found_count >= target_count;
  extract_target_paths_to_result(result, scratch, targets, stream);
  clear_target_multiplicity_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.target_multiplicity.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  reset_touched_vertices(scratch, stream);
  return result;
}

}  // namespace unit_bfs_detail

struct UnitBfsCsrWorkspace::Impl {
  unit_bfs_detail::OutgoingCsrOwner outgoing;
  unit_bfs_detail::UnitBfsScratch scratch;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : outgoing(unit_bfs_detail::copy_host_csr_to_device(host, stream)),
        scratch(host.rows) {}
};

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream) {
  unit_bfs_detail::validate_host_csr_arrays(adjacency);
  impl_ = std::make_unique<Impl>(adjacency, stream);
}

UnitBfsCsrWorkspace::~UnitBfsCsrWorkspace() = default;
UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept = default;
UnitBfsCsrWorkspace& UnitBfsCsrWorkspace::operator=(
    UnitBfsCsrWorkspace&&) noexcept = default;

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  (void)delta;
  using namespace unit_bfs_detail;
  if (!impl_) {
    throw std::runtime_error("UnitBfsCsrWorkspace has no implementation");
  }
  validate_sources_targets(impl_->scratch.rows, sources, targets);
  return run_unit_bfs_impl(impl_->outgoing,
                           impl_->scratch,
                           sources,
                           targets,
                           max_depth,
                           stream,
                           progress_callback,
                           progress_user_data);
}

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  UnitBfsCsrResult result = run(sources,
                                std::vector<int>{target},
                                delta,
                                max_depth,
                                stream,
                                progress_callback,
                                progress_user_data);
  result.target = target;
  if (!result.target_distances.empty()) {
    result.target_distance = result.target_distances.front();
    result.target_reached = std::isfinite(result.target_distance);
  }
  return result;
}

UnitBfsCsrResult UnitBfsCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_depth,
    hipStream_t stream,
    UnitBfsCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_depth,
             stream,
             progress_callback,
             progress_user_data);
}
