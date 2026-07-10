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
#include <unordered_set>
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

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  explicit PinnedHostBuffer(std::size_t count) {
    if (count != 0) {
      UNIT_BFS_HIP_CHECK(
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
  // The queue is append-only for one BFS run.  The current frontier is a
  // half-open range within it, and every successfully claimed vertex is also
  // the complete list of levels that must be reset before the next run.
  DeviceBuffer<int> frontier_queue;
  // [0] = append position / visited count, [1] = number of targets found.
  DeviceBuffer<int> status;
  PinnedHostBuffer<int> host_status;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_sources;
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
        frontier_queue(static_cast<std::size_t>(rows_)),
        status(2),
        host_status(2) {}

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

__device__ inline void count_target_if_reached(int v,
                                               const int* target_multiplicity,
                                               int* found_count) {
  const int multiplicity = target_multiplicity[v];
  if (multiplicity > 0) {
    atomicAdd(found_count, multiplicity);
  }
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

__global__ void initialize_bfs_arrays_kernel(Offset rows,
                                             int* level,
                                             int* target_multiplicity) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    level[v] = kUnvisited;
    target_multiplicity[v] = 0;
  }
}

__global__ void reset_visited_levels_kernel(const int* frontier_queue,
                                            int visited_count,
                                            int* level) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < visited_count;
       i += blockDim.x * gridDim.x) {
    const int v = frontier_queue[i];
    level[v] = kUnvisited;
  }
}

__global__ void initialize_sources_kernel(const int* sources,
                                          int source_count,
                                          int initially_found,
                                          int* level,
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
    // Sources are deduplicated on the host, so neither claiming nor queue
    // allocation needs a global atomic operation.
    level[source] = 0;
    pred_node[source] = source;
    pred_edge[source] = static_cast<Offset>(-1);
    frontier_queue[i] = source;
  }
}

__global__ void expand_frontier_kernel(int frontier_begin,
                                       int frontier_end,
                                       int next_level,
                                       const Offset* out_rowptr,
                                       const Index* out_colind,
                                       int* level,
                                       int* pred_node,
                                       Offset* pred_edge,
                                       int* frontier_queue,
                                       int* queue_tail,
                                       const int* target_multiplicity,
                                       int* found_count) {
  for (int i = frontier_begin + blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_end;
       i += blockDim.x * gridDim.x) {
    const int u = frontier_queue[i];
    for (Offset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
      const int v = static_cast<int>(out_colind[edge]);
      // Most examined edges point to a vertex already reached by this BFS.
      // Avoid an atomic read-modify-write for that common case; a stale load is
      // harmless because the CAS remains the authority for claiming v.
      const bool claimed =
          level[v] == kUnvisited &&
          atomicCAS(&level[v], kUnvisited, next_level) == kUnvisited;
      if (claimed) {
        pred_node[v] = u;
        pred_edge[v] = edge;
      }
      // Reserve queue positions once per active GPU wave instead of forcing
      // every discovered vertex through the same global atomic counter.
      const int pos = wave_append_position(claimed, queue_tail);
      if (claimed) {
        frontier_queue[pos] = v;
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
    // Duplicate target entries may map several lanes to the same word.
    atomicExch(&target_multiplicity[targets[i]], 0);
  }
}

__global__ void measure_target_paths_kernel(const int* targets,
                                            int target_count,
                                            const int* level,
                                            float* target_distances,
                                            int* path_lengths,
                                            int* path_sources) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    const int target_level = level[target];
    target_distances[i] =
        target_level == kUnvisited ? INFINITY : static_cast<float>(target_level);
    path_lengths[i] = 0;
    path_sources[i] = -1;
    if (target_level == kUnvisited) {
      continue;
    }
    // In unit BFS, the level is exactly the edge count.  The previous
    // implementation walked the predecessor chain here and then walked it a
    // second time while filling the path.  Derive the size in O(1) and let the
    // fill pass perform the only pointer chase.
    path_lengths[i] = target_level + 1;
  }
}

__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const int* pred_node,
                                         const Offset* pred_edge,
                                         const int* path_lengths,
                                         int* path_sources,
                                         const int* node_offsets,
                                         const int* edge_offsets,
                                         int* path_nodes,
                                         Offset* path_edges) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (path_lengths[i] <= 0) {
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
    path_sources[i] = current;
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
    return;
  }

  initialize_bfs_arrays_kernel<<<grid_for_items(rows), kBlockSize, 0, stream>>>(
      rows,
      scratch.level.get(),
      scratch.target_multiplicity.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  scratch.initialized = true;
}

void reset_visited_levels(UnitBfsScratch& scratch,
                          int visited_count,
                          hipStream_t stream) {
  if (visited_count > 0) {
    reset_visited_levels_kernel<<<grid_for_items(visited_count),
                                  kBlockSize,
                                  0,
                                  stream>>>(
        scratch.frontier_queue.get(), visited_count, scratch.level.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
  }
}

std::array<int, 2> copy_status_to_host(UnitBfsScratch& scratch,
                                      hipStream_t stream) {
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_status.get(),
                                    scratch.status.get(),
                                    2 * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  return {scratch.host_status.get()[0], scratch.host_status.get()[1]};
}

void extract_target_paths_to_result(UnitBfsCsrResult& result,
                                    UnitBfsScratch& scratch,
                                    const std::vector<int>& targets,
                                    int visited_count,
                                    hipStream_t stream) {
  const int target_count = static_cast<int>(targets.size());
  measure_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.level.get(),
      scratch.target_distances.get(),
      scratch.target_path_lengths.get(),
      scratch.target_sources.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  result.target_distances.resize(targets.size());
  result.target_sources.assign(targets.size(), -1);
  std::vector<int> path_lengths(targets.size());

  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_distances.data(),
                                    scratch.target_distances.get(),
                                    targets.size() * sizeof(float),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(path_lengths.data(),
                                    scratch.target_path_lengths.get(),
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
    if (path_lengths[i] <= 0 ||
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
        scratch.target_sources.get(),
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
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(result.target_sources.data(),
                                      scratch.target_sources.get(),
                                      targets.size() * sizeof(int),
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
  // Reset only the state used to claim a vertex.  Predecessors are overwritten
  // whenever a future search claims that vertex and never need cleanup.
  reset_visited_levels(scratch, visited_count, stream);
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
  const int target_count = static_cast<int>(targets.size());
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
    std::unordered_set<int> source_seen;
    source_seen.reserve(sources.size());
    deduplicated_sources.reserve(sources.size());
    for (const int source : sources) {
      if (source_seen.insert(source).second) {
        deduplicated_sources.push_back(source);
      }
    }
    effective_sources = &deduplicated_sources;
    for (const int target : targets) {
      if (source_seen.find(target) != source_seen.end()) {
        ++initially_found;
      }
    }
  }
  const int source_count = static_cast<int>(effective_sources->size());

  scratch.ensure_source_capacity(effective_sources->size());
  scratch.ensure_target_capacity(targets.size());

  initialize_scratch_once(scratch, scratch.rows, stream);
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.sources.get(),
                                    effective_sources->data(),
                                    effective_sources->size() * sizeof(int),
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
      initially_found,
      scratch.level.get(),
      scratch.pred_node.get(),
      scratch.pred_edge.get(),
      scratch.frontier_queue.get(),
      scratch.status.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  int frontier_begin = 0;
  int frontier_end = source_count;
  int queue_tail = source_count;
  int current_count = source_count;
  int found_count = initially_found;
  UnitBfsCsrResult result;
  result.target = -1;
  result.iterations_used = 0;

  for (int depth = 0;
       current_count > 0 && found_count < target_count && depth < max_depth;
       ++depth) {
    expand_frontier_kernel<<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
        frontier_begin,
        frontier_end,
        depth + 1,
        outgoing.rowptr.get(),
        outgoing.colind.get(),
        scratch.level.get(),
        scratch.pred_node.get(),
        scratch.pred_edge.get(),
        scratch.frontier_queue.get(),
        scratch.status.get(),
        scratch.target_multiplicity.get(),
        scratch.status.get() + 1);
    UNIT_BFS_HIP_CHECK(hipGetLastError());

    const std::array<int, 2> status = copy_status_to_host(scratch, stream);
    queue_tail = status[0];
    if (queue_tail < frontier_end || queue_tail > n_int) {
      throw std::runtime_error("unit BFS append-only frontier queue overflow");
    }
    current_count = queue_tail - frontier_end;
    found_count = status[1];
    result.iterations_used = depth + 1;
    frontier_begin = frontier_end;
    frontier_end = queue_tail;

    if (progress_callback) {
      UnitBfsCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_depth;
      progress.convergence_checked = true;
      progress.changed = current_count > 0;
      progress_callback(progress, progress_user_data);
    }
  }

  result.converged = current_count == 0 || found_count >= target_count;
  result.stopped_on_target = found_count >= target_count;
  clear_target_multiplicity_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.target_multiplicity.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  extract_target_paths_to_result(result, scratch, targets, queue_tail, stream);
  return result;
}

}  // namespace unit_bfs_detail

struct UnitBfsCsrGraph::Impl {
  unit_bfs_detail::OutgoingCsrOwner outgoing;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : outgoing(unit_bfs_detail::copy_host_csr_to_device(host, stream)) {}
};

UnitBfsCsrGraph::UnitBfsCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream) {
  unit_bfs_detail::validate_host_csr_arrays(adjacency);
  impl_ = std::make_unique<Impl>(adjacency, stream);
  // The shared graph may be consumed by worker streams created after this
  // constructor returns.  Complete the one-time upload before publishing it.
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
}

UnitBfsCsrGraph::~UnitBfsCsrGraph() = default;
UnitBfsCsrGraph::UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept = default;
UnitBfsCsrGraph& UnitBfsCsrGraph::operator=(UnitBfsCsrGraph&&) noexcept = default;

struct UnitBfsCsrWorkspace::Impl {
  std::shared_ptr<const UnitBfsCsrGraph> graph;
  unit_bfs_detail::UnitBfsScratch scratch;

  static minplus_sparse::Offset require_graph_rows(
      const std::shared_ptr<const UnitBfsCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("unit BFS shared graph must not be null");
    }
    return candidate->impl_->outgoing.rows;
  }

  Impl(std::shared_ptr<const UnitBfsCsrGraph> graph_, hipStream_t stream)
      : graph(std::move(graph_)),
        scratch(require_graph_rows(graph)) {
    (void)stream;
  }
};

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : UnitBfsCsrWorkspace(
          std::make_shared<UnitBfsCsrGraph>(adjacency, stream), stream) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    std::shared_ptr<const UnitBfsCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(std::move(adjacency), stream)) {}

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
  return run_unit_bfs_impl(impl_->graph->impl_->outgoing,
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
