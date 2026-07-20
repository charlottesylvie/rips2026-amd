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
using CompactOffset = std::int32_t;

static_assert(sizeof(CompactOffset) == 4,
              "unit BFS compact offsets must be 32-bit");
static_assert(std::numeric_limits<CompactOffset>::max() <
                  std::numeric_limits<Offset>::max(),
              "unit BFS wide offsets must exceed the compact range");

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kUnvisited = std::numeric_limits<int>::max();
constexpr int kLevelsPerStatusCheck = 4;
constexpr int kBatchBlocksPerComputeUnit = 4;

enum UnitBfsStatusIndex : int {
  kStatusQueueTail = 0,
  kStatusFoundCount,
  kStatusFrontierBegin,
  kStatusFrontierEnd,
  kStatusCompletedDepth,
  kStatusActive,
  kStatusCount,
};

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
  bool uses_32_bit_offsets = false;
  int batched_launch_blocks = 1;
  DeviceBuffer<CompactOffset> rowptr32;
  DeviceBuffer<Offset> rowptr64;
  DeviceBuffer<Index> colind;

  OutgoingCsrOwner() = default;
  OutgoingCsrOwner(Offset rows_,
                   Offset cols_,
                   Offset nnz_,
                   bool uses_32_bit_offsets_)
      : rows(rows_),
        cols(cols_),
        nnz(nnz_),
        uses_32_bit_offsets(uses_32_bit_offsets_),
        rowptr32(uses_32_bit_offsets
                     ? static_cast<std::size_t>(rows_) + 1
                     : 0),
        rowptr64(uses_32_bit_offsets
                     ? 0
                     : static_cast<std::size_t>(rows_) + 1),
        colind(static_cast<std::size_t>(nnz_)) {}
};

struct UnitBfsScratch {
  Offset rows = 0;
  bool uses_32_bit_offsets = false;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<int> target_multiplicity;
  DeviceBuffer<int> level;
  DeviceBuffer<int> pred_node;
  DeviceBuffer<CompactOffset> pred_edge32;
  DeviceBuffer<Offset> pred_edge64;
  // The queue is append-only for one BFS run.  The current frontier is a
  // half-open range within it, and every successfully claimed vertex is also
  // the complete list of levels that must be reset before the next run.
  DeviceBuffer<int> frontier_queue;
  // Device-resident traversal controller.  Keeping frontier bounds, completed
  // depth, and the stopping condition here allows several BFS levels to be
  // enqueued before the host checks progress.
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
  UnitBfsScratch(Offset rows_, bool uses_32_bit_offsets_)
      : rows(rows_),
        uses_32_bit_offsets(uses_32_bit_offsets_),
        target_multiplicity(static_cast<std::size_t>(rows_)),
        level(static_cast<std::size_t>(rows_)),
        pred_node(static_cast<std::size_t>(rows_)),
        pred_edge32(uses_32_bit_offsets
                        ? static_cast<std::size_t>(rows_)
                        : 0),
        pred_edge64(uses_32_bit_offsets
                        ? 0
                        : static_cast<std::size_t>(rows_)),
        frontier_queue(static_cast<std::size_t>(rows_)),
        status(kStatusCount),
        host_status(kStatusCount) {}

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

template <typename EdgeOffset>
__global__ void initialize_sources_kernel(const int* sources,
                                          int source_count,
                                          int initially_found,
                                          int target_count,
                                          int max_depth,
                                          int* level,
                                          int* pred_node,
                                          EdgeOffset* pred_edge,
                                          int* frontier_queue,
                                          int* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status[kStatusQueueTail] = source_count;
    status[kStatusFoundCount] = initially_found;
    status[kStatusFrontierBegin] = 0;
    status[kStatusFrontierEnd] = source_count;
    status[kStatusCompletedDepth] = 0;
    status[kStatusActive] =
        source_count > 0 && initially_found < target_count && max_depth > 0;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    // Sources are deduplicated on the host, so neither claiming nor queue
    // allocation needs a global atomic operation.
    level[source] = 0;
    pred_node[source] = source;
    pred_edge[source] = static_cast<EdgeOffset>(-1);
    frontier_queue[i] = source;
  }
}

template <typename EdgeOffset>
__global__ void expand_frontier_kernel(const EdgeOffset* out_rowptr,
                                       const Index* out_colind,
                                       int* level,
                                       int* pred_node,
                                       EdgeOffset* pred_edge,
                                       int* frontier_queue,
                                       const int* target_multiplicity,
                                       int* status) {
  if (status[kStatusActive] == 0) {
    return;
  }
  const int frontier_begin = status[kStatusFrontierBegin];
  const int frontier_end = status[kStatusFrontierEnd];
  const int next_level = status[kStatusCompletedDepth] + 1;
  for (int i = frontier_begin + blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_end;
       i += blockDim.x * gridDim.x) {
    const int u = frontier_queue[i];
    for (EdgeOffset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
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
      const int pos =
          wave_append_position(claimed, status + kStatusQueueTail);
      if (claimed) {
        frontier_queue[pos] = v;
        count_target_if_reached(
            v, target_multiplicity, status + kStatusFoundCount);
      }
    }
  }
}

__global__ void advance_frontier_kernel(int target_count,
                                        int max_depth,
                                        int* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      status[kStatusActive] == 0) {
    return;
  }

  status[kStatusFrontierBegin] = status[kStatusFrontierEnd];
  status[kStatusFrontierEnd] = status[kStatusQueueTail];
  ++status[kStatusCompletedDepth];
  status[kStatusActive] =
      status[kStatusFrontierBegin] < status[kStatusFrontierEnd] &&
      status[kStatusFoundCount] < target_count &&
      status[kStatusCompletedDepth] < max_depth;
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

template <typename EdgeOffset>
__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const int* pred_node,
                                         const EdgeOffset* pred_edge,
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
      path_edges[edge_begin + j - 1] =
          static_cast<Offset>(pred_edge[current]);
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
    path_sources[i] = current;
  }
}

constexpr bool nnz_fits_32_bit_offsets(Offset nnz) noexcept {
  return nnz >= 0 &&
         nnz <= static_cast<Offset>(
                    std::numeric_limits<CompactOffset>::max());
}

inline bool select_32_bit_offsets(Offset nnz,
                                  UnitBfsCsrOffsetMode offset_mode) {
  switch (offset_mode) {
    case UnitBfsCsrOffsetMode::kAuto:
      return nnz_fits_32_bit_offsets(nnz);
    case UnitBfsCsrOffsetMode::kForce64Bit:
      return false;
  }
  throw std::invalid_argument("unknown unit BFS offset mode");
}

static_assert(nnz_fits_32_bit_offsets(
                  static_cast<Offset>(
                      std::numeric_limits<CompactOffset>::max())),
              "INT32_MAX edges must use compact offsets");
static_assert(!nnz_fits_32_bit_offsets(
                  static_cast<Offset>(
                      std::numeric_limits<CompactOffset>::max()) + 1),
              "graphs above INT32_MAX edges must retain wide offsets");

OutgoingCsrOwner copy_host_csr_to_device(
    const HostCsrF32& host,
    hipStream_t stream,
    UnitBfsCsrOffsetMode offset_mode) {
  const bool uses_32_bit_offsets =
      select_32_bit_offsets(host.nnz, offset_mode);
  OutgoingCsrOwner device(
      host.rows, host.cols, host.nnz, uses_32_bit_offsets);
  int current_device = 0;
  hipDeviceProp_t device_properties{};
  UNIT_BFS_HIP_CHECK(hipGetDevice(&current_device));
  UNIT_BFS_HIP_CHECK(
      hipGetDeviceProperties(&device_properties, current_device));
  const long long compute_units =
      std::max<long long>(1, device_properties.multiProcessorCount);
  device.batched_launch_blocks = static_cast<int>(std::min<long long>(
      grid_for_items(host.rows),
      std::min<long long>(kMaxGridX,
                          compute_units * kBatchBlocksPerComputeUnit)));
  const std::size_t rows = checked_size(host.rows, "rows");
  const std::size_t nnz = checked_size(host.nnz, "nnz");
  if (nnz != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.colind.get(),
                                      host.colind.data(),
                                      nnz * sizeof(Index),
                                      hipMemcpyHostToDevice,
                                      stream));
  }
  std::vector<CompactOffset> compact_rowptr;
  if (uses_32_bit_offsets) {
    compact_rowptr.resize(rows + 1);
    std::transform(host.rowptr.begin(),
                   host.rowptr.end(),
                   compact_rowptr.begin(),
                   [](Offset offset) {
                     return static_cast<CompactOffset>(offset);
                   });
    // This converted host buffer is temporary, so use a blocking copy rather
    // than relying on pageable-host async-copy behavior for its lifetime.
    UNIT_BFS_HIP_CHECK(hipMemcpy(device.rowptr32.get(),
                                 compact_rowptr.data(),
                                 (rows + 1) * sizeof(CompactOffset),
                                 hipMemcpyHostToDevice));
  } else {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.rowptr64.get(),
                                      host.rowptr.data(),
                                      (rows + 1) * sizeof(Offset),
                                      hipMemcpyHostToDevice,
                                      stream));
  }
  // Worker streams may consume the graph immediately after construction.
  // Finish the one-time asynchronous portions before publishing it.
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
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

std::array<int, kStatusCount> copy_status_to_host(UnitBfsScratch& scratch,
                                                  hipStream_t stream) {
  UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_status.get(),
                                    scratch.status.get(),
                                    kStatusCount * sizeof(int),
                                    hipMemcpyDeviceToHost,
                                    stream));
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  std::array<int, kStatusCount> status{};
  std::copy_n(scratch.host_status.get(), kStatusCount, status.begin());
  return status;
}

template <typename EdgeOffset>
void extract_target_paths_to_result(UnitBfsCsrResult& result,
                                    UnitBfsScratch& scratch,
                                    const EdgeOffset* pred_edge,
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
    fill_target_paths_kernel<EdgeOffset>
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.rows,
            scratch.pred_node.get(),
            pred_edge,
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

template <typename EdgeOffset>
UnitBfsCsrResult run_unit_bfs_with_offsets(
    const OutgoingCsrOwner& outgoing,
    const EdgeOffset* out_rowptr,
    UnitBfsScratch& scratch,
    EdgeOffset* pred_edge,
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

  initialize_sources_kernel<EdgeOffset>
      <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
          scratch.sources.get(),
          source_count,
          initially_found,
          target_count,
          max_depth,
          scratch.level.get(),
          scratch.pred_node.get(),
          pred_edge,
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

  while (current_count > 0 && found_count < target_count &&
         result.iterations_used < max_depth) {
    // Check the first expansion immediately so the next batch can be sized
    // from a real frontier.  Thereafter, enqueue four device-controlled levels
    // per status copy.  Kernels after target discovery, frontier exhaustion, or
    // max_depth become no-ops through kStatusActive.
    const int remaining_depth = max_depth - result.iterations_used;
    const int rounds_to_enqueue =
        progress_callback != nullptr || result.iterations_used == 0
            ? 1
            : std::min(kLevelsPerStatusCheck, remaining_depth);
    const int launch_blocks =
        rounds_to_enqueue == 1
            ? grid_for_frontier(current_count)
            : std::min(grid_for_items(scratch.rows),
                       std::max(grid_for_frontier(current_count),
                                outgoing.batched_launch_blocks));

    for (int round = 0; round < rounds_to_enqueue; ++round) {
      expand_frontier_kernel<EdgeOffset>
          <<<launch_blocks, kBlockSize, 0, stream>>>(
              out_rowptr,
              outgoing.colind.get(),
              scratch.level.get(),
              scratch.pred_node.get(),
              pred_edge,
              scratch.frontier_queue.get(),
              scratch.target_multiplicity.get(),
              scratch.status.get());
      UNIT_BFS_HIP_CHECK(hipGetLastError());
      advance_frontier_kernel<<<1, 1, 0, stream>>>(
          target_count, max_depth, scratch.status.get());
      UNIT_BFS_HIP_CHECK(hipGetLastError());
    }

    const std::array<int, kStatusCount> status =
        copy_status_to_host(scratch, stream);
    queue_tail = status[kStatusQueueTail];
    found_count = status[kStatusFoundCount];
    frontier_begin = status[kStatusFrontierBegin];
    frontier_end = status[kStatusFrontierEnd];
    result.iterations_used = status[kStatusCompletedDepth];
    if (queue_tail < 0 || queue_tail > n_int || frontier_begin < 0 ||
        frontier_end < frontier_begin || frontier_end != queue_tail ||
        result.iterations_used < 0 || result.iterations_used > max_depth) {
      throw std::runtime_error("unit BFS device frontier state is inconsistent");
    }
    current_count = frontier_end - frontier_begin;

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
  extract_target_paths_to_result(
      result, scratch, pred_edge, targets, queue_tail, stream);
  return result;
}

UnitBfsCsrResult run_unit_bfs_impl(const OutgoingCsrOwner& outgoing,
                                   UnitBfsScratch& scratch,
                                   const std::vector<int>& sources,
                                   const std::vector<int>& targets,
                                   int max_depth,
                                   hipStream_t stream,
                                   UnitBfsCsrProgressCallback progress_callback,
                                   void* progress_user_data) {
  if (scratch.uses_32_bit_offsets != outgoing.uses_32_bit_offsets) {
    throw std::logic_error(
        "unit BFS graph and workspace offset representations do not match");
  }
  if (outgoing.uses_32_bit_offsets) {
    return run_unit_bfs_with_offsets(
        outgoing,
        outgoing.rowptr32.get(),
        scratch,
        scratch.pred_edge32.get(),
        sources,
        targets,
        max_depth,
        stream,
        progress_callback,
        progress_user_data);
  }
  return run_unit_bfs_with_offsets(outgoing,
                                   outgoing.rowptr64.get(),
                                   scratch,
                                   scratch.pred_edge64.get(),
                                   sources,
                                   targets,
                                   max_depth,
                                   stream,
                                   progress_callback,
                                   progress_user_data);
}

}  // namespace unit_bfs_detail

struct UnitBfsCsrGraph::Impl {
  unit_bfs_detail::OutgoingCsrOwner outgoing;

  Impl(const HostCsrF32& host,
       hipStream_t stream,
       UnitBfsCsrOffsetMode offset_mode)
      : outgoing(unit_bfs_detail::copy_host_csr_to_device(
            host, stream, offset_mode)) {}
};

UnitBfsCsrGraph::UnitBfsCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream)
    : UnitBfsCsrGraph(
          adjacency, stream, UnitBfsCsrOffsetMode::kAuto) {}

UnitBfsCsrGraph::UnitBfsCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream,
                                 UnitBfsCsrOffsetMode offset_mode) {
  unit_bfs_detail::validate_host_csr_arrays(adjacency);
  impl_ = std::make_unique<Impl>(adjacency, stream, offset_mode);
}

UnitBfsCsrGraph::~UnitBfsCsrGraph() = default;
UnitBfsCsrGraph::UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept = default;
UnitBfsCsrGraph& UnitBfsCsrGraph::operator=(UnitBfsCsrGraph&&) noexcept = default;

bool UnitBfsCsrGraph::uses_32_bit_offsets() const noexcept {
  return impl_ && impl_->outgoing.uses_32_bit_offsets;
}

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

  static bool require_graph_uses_32_bit_offsets(
      const std::shared_ptr<const UnitBfsCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("unit BFS shared graph must not be null");
    }
    return candidate->impl_->outgoing.uses_32_bit_offsets;
  }

  Impl(std::shared_ptr<const UnitBfsCsrGraph> graph_, hipStream_t stream)
      : graph(std::move(graph_)),
        scratch(require_graph_rows(graph),
                require_graph_uses_32_bit_offsets(graph)) {
    (void)stream;
  }
};

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : UnitBfsCsrWorkspace(
          adjacency, stream, UnitBfsCsrOffsetMode::kAuto) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream,
                                         UnitBfsCsrOffsetMode offset_mode)
    : UnitBfsCsrWorkspace(
          std::make_shared<UnitBfsCsrGraph>(adjacency, stream, offset_mode),
          stream) {}

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
