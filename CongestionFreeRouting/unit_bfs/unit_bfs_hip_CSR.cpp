#include "unit_bfs_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
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

enum UnitBfsPathStatus : int {
  kPathNotValidated = 0,
  kPathValid = 1,
  kPathInvalidOffsets = -1,
  kPathInvalidPredecessor = -2,
  kPathInvalidEdge = -3,
  kPathInvalidLevel = -4,
  kPathInvalidRoot = -5,
  kPathInvalidEpoch = -6,
};

struct TargetPathMetadata {
  float distance;
  int length;
  int source;
  int status;
  std::uint32_t query_epoch;
  std::uint32_t validation_epoch;
};

static_assert(sizeof(TargetPathMetadata) == 6 * sizeof(std::uint32_t),
              "unit BFS target metadata must remain one packed 24-byte record");

const char* path_status_name(int status) noexcept {
  switch (status) {
    case kPathNotValidated:
      return "not validated";
    case kPathValid:
      return "valid";
    case kPathInvalidOffsets:
      return "invalid compact offsets";
    case kPathInvalidPredecessor:
      return "invalid predecessor vertex";
    case kPathInvalidEdge:
      return "invalid predecessor edge";
    case kPathInvalidLevel:
      return "predecessor level mismatch";
    case kPathInvalidRoot:
      return "invalid source root";
    case kPathInvalidEpoch:
      return "stale query epoch";
    default:
      return "unknown status";
  }
}

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

inline int current_hip_device() {
  int device = 0;
  UNIT_BFS_HIP_CHECK(hipGetDevice(&device));
  return device;
}

inline void synchronize_explicit_stream(hipStream_t stream) {
  if (stream != nullptr) {
    UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  }
}

inline void copy_control_synchronously(void* destination,
                                       const void* source,
                                       std::size_t bytes,
                                       hipMemcpyKind kind,
                                       hipStream_t producer_stream) {
  // The target gfx1151 runtime has returned stale-but-plausible control data
  // from an asynchronous copy on one hardware queue while other worker queues
  // remained active.  Finish the producing stream, then use a host-synchronous
  // copy for the small source/target, frontier-status, metadata, and offset
  // records.  Large compact paths remain asynchronous.
  synchronize_explicit_stream(producer_stream);
  UNIT_BFS_HIP_CHECK(hipMemcpy(destination, source, bytes, kind));
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
      UNIT_BFS_HIP_CHECK(
          hipMalloc(reinterpret_cast<void**>(&candidate), count * sizeof(T)));
      ptr_ = candidate;
      count_ = count;
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
  explicit PinnedHostBuffer(std::size_t count) { reset(count); }

  void reset(std::size_t count) {
    if (count == count_) {
      return;
    }
    T* candidate = nullptr;
    if (count != 0) {
      UNIT_BFS_HIP_CHECK(
          hipHostMalloc(reinterpret_cast<void**>(&candidate),
                        count * sizeof(T),
                        hipHostMallocDefault));
    }
    if (ptr_ != nullptr) {
      (void)hipHostFree(ptr_);
    }
    ptr_ = candidate;
    count_ = count;
  }

  ~PinnedHostBuffer() {
    if (ptr_ != nullptr) {
      (void)hipHostFree(ptr_);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
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
  // Measurement and validation publish one record per target. A synchronized
  // pinned transfer then moves the tuple together instead of issuing separate
  // pageable copies for length, source, distance, and validation status.
  DeviceBuffer<TargetPathMetadata> target_metadata;
  PinnedHostBuffer<TargetPathMetadata> host_target_metadata;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  PinnedHostBuffer<int> host_target_node_offsets;
  PinnedHostBuffer<int> host_target_edge_offsets;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  PinnedHostBuffer<int> host_compact_path_nodes;
  PinnedHostBuffer<Offset> host_compact_path_edges;
  std::uint32_t query_epoch = 0;
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
    }
    if (target_metadata.size() < target_count) {
      target_metadata.reset(target_count);
    }
    if (host_target_metadata.size() < target_count) {
      host_target_metadata.reset(target_count);
    }
    if (target_node_offsets.size() < target_count + 1) {
      target_node_offsets.reset(target_count + 1);
    }
    if (host_target_node_offsets.size() < target_count + 1) {
      host_target_node_offsets.reset(target_count + 1);
    }
    if (target_edge_offsets.size() < target_count + 1) {
      target_edge_offsets.reset(target_count + 1);
    }
    if (host_target_edge_offsets.size() < target_count + 1) {
      host_target_edge_offsets.reset(target_count + 1);
    }
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count) {
    if (compact_path_nodes.size() < node_count) {
      compact_path_nodes.reset(node_count);
    }
    if (host_compact_path_nodes.size() < node_count) {
      host_compact_path_nodes.reset(node_count);
    }
    if (compact_path_edges.size() < edge_count) {
      compact_path_edges.reset(edge_count);
    }
    if (host_compact_path_edges.size() < edge_count) {
      host_compact_path_edges.reset(edge_count);
    }
  }

  std::uint32_t begin_query() {
    if (query_epoch == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("unit BFS workspace query epoch exhausted");
    }
    return ++query_epoch;
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
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] != 1.0f) {
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

__device__ inline int atomic_load_int(const int* address) {
  // Reused workspace state is written atomically below. Read it through the
  // same coherent path so an ordinary cached value from a prior route cannot
  // suppress a claim or validate a detached predecessor chain.
  return atomicAdd(const_cast<int*>(address), 0);
}

__device__ inline void count_target_if_reached(int v,
                                               const int* target_multiplicity,
                                               int* found_count) {
  const int multiplicity = atomic_load_int(target_multiplicity + v);
  if (multiplicity > 0) {
    atomicAdd(found_count, multiplicity);
  }
}

__device__ inline int atomic_load_status(int* address) {
  return atomic_load_int(address);
}

__device__ inline void atomic_store_status(int* address, int value) {
  atomicExch(address, value);
}

__device__ inline int append_position(bool append, int* queue_tail) {
  // This helper is called from adjacency loops whose trip counts differ by
  // lane. A ballot/shuffle reservation is invalid there because lanes do not
  // reach the same dynamic collective calls. One atomic per successful claim
  // preserves the append-only queue invariant for arbitrary row degrees.
  return append ? atomicAdd(queue_tail, 1) : -1;
}

__global__ void initialize_bfs_arrays_kernel(Offset rows,
                                             int* level,
                                             int* target_multiplicity) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    atomicExch(&level[v], kUnvisited);
    atomicExch(&target_multiplicity[v], 0);
  }
}

__global__ void reset_visited_levels_kernel(const int* frontier_queue,
                                            int visited_count,
                                            int* level) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < visited_count;
       i += blockDim.x * gridDim.x) {
    const int v = frontier_queue[i];
    atomicExch(&level[v], kUnvisited);
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
    atomic_store_status(status + kStatusQueueTail, source_count);
    atomic_store_status(status + kStatusFoundCount, initially_found);
    atomic_store_status(status + kStatusFrontierBegin, 0);
    atomic_store_status(status + kStatusFrontierEnd, source_count);
    atomic_store_status(status + kStatusCompletedDepth, 0);
    __threadfence();
    atomic_store_status(
        status + kStatusActive,
        source_count > 0 && initially_found < target_count && max_depth > 0);
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < source_count;
       i += blockDim.x * gridDim.x) {
    const int source = sources[i];
    // Sources are deduplicated on the host. Publish their complete predecessor
    // state before making level zero visible to later atomic claims/loads.
    pred_node[source] = source;
    pred_edge[source] = static_cast<EdgeOffset>(-1);
    frontier_queue[i] = source;
    __threadfence();
    atomicExch(&level[source], 0);
  }
}

template <typename EdgeOffset>
__device__ inline void expand_frontier_range(
    int frontier_begin,
    int frontier_end,
    int next_level,
    const EdgeOffset* out_rowptr,
    const Index* out_colind,
    int* level,
    int* pred_node,
    EdgeOffset* pred_edge,
    int* frontier_queue,
    const int* target_multiplicity,
    int* queue_tail,
    int* found_count) {
  for (int i = frontier_begin + blockIdx.x * blockDim.x + threadIdx.x;
       i < frontier_end;
       i += blockDim.x * gridDim.x) {
    const int u = frontier_queue[i];
    for (EdgeOffset edge = out_rowptr[u]; edge < out_rowptr[u + 1]; ++edge) {
      const int v = static_cast<int>(out_colind[edge]);
      // The CAS must be unconditional. An ordinary precheck can retain a
      // finite value from the previous route after sparse reset and skip the
      // authoritative claim, leaving an old level/predecessor pair in place.
      const bool claimed =
          atomicCAS(&level[v], kUnvisited, next_level) == kUnvisited;
      if (claimed) {
        pred_node[v] = u;
        pred_edge[v] = edge;
      }
      // Each successful claim reserves one queue position. This call is inside
      // a variable-trip adjacency loop, so it must not use a wave collective.
      const int pos = append_position(claimed, queue_tail);
      if (claimed) {
        frontier_queue[pos] = v;
        count_target_if_reached(v, target_multiplicity, found_count);
      }
    }
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
  __shared__ int controller[4];
  if (threadIdx.x == 0) {
    controller[0] = atomic_load_status(status + kStatusActive);
    if (controller[0] != 0) {
      controller[1] = atomic_load_status(status + kStatusFrontierBegin);
      controller[2] = atomic_load_status(status + kStatusFrontierEnd);
      controller[3] = atomic_load_status(status + kStatusCompletedDepth);
    }
  }
  __syncthreads();
  if (controller[0] == 0) {
    return;
  }
  expand_frontier_range(controller[1],
                        controller[2],
                        controller[3] + 1,
                        out_rowptr,
                        out_colind,
                        level,
                        pred_node,
                        pred_edge,
                        frontier_queue,
                        target_multiplicity,
                        status + kStatusQueueTail,
                        status + kStatusFoundCount);
}

template <typename EdgeOffset>
__global__ void expand_frontier_host_controlled_kernel(
    int frontier_begin,
    int frontier_end,
    int next_level,
    const EdgeOffset* out_rowptr,
    const Index* out_colind,
    int* level,
    int* pred_node,
    EdgeOffset* pred_edge,
    int* frontier_queue,
    const int* target_multiplicity,
    int* queue_tail,
    int* found_count) {
  expand_frontier_range(frontier_begin,
                        frontier_end,
                        next_level,
                        out_rowptr,
                        out_colind,
                        level,
                        pred_node,
                        pred_edge,
                        frontier_queue,
                        target_multiplicity,
                        queue_tail,
                        found_count);
}

__global__ void advance_frontier_kernel(int target_count,
                                        int max_depth,
                                        int* status) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      atomic_load_status(status + kStatusActive) == 0) {
    return;
  }

  const int frontier_begin =
      atomic_load_status(status + kStatusFrontierEnd);
  const int frontier_end = atomic_load_status(status + kStatusQueueTail);
  const int completed_depth =
      atomic_load_status(status + kStatusCompletedDepth) + 1;
  const int found_count =
      atomic_load_status(status + kStatusFoundCount);
  atomic_store_status(status + kStatusFrontierBegin, frontier_begin);
  atomic_store_status(status + kStatusFrontierEnd, frontier_end);
  atomic_store_status(status + kStatusCompletedDepth, completed_depth);
  __threadfence();
  atomic_store_status(
      status + kStatusActive,
      frontier_begin < frontier_end && found_count < target_count &&
          completed_depth < max_depth);
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
                                            std::uint32_t query_epoch,
                                            TargetPathMetadata* metadata) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    const int target_level = atomic_load_int(level + target);
    metadata[i].distance =
        target_level == kUnvisited ? INFINITY : static_cast<float>(target_level);
    metadata[i].length =
        target_level == kUnvisited ? 0 : target_level + 1;
    metadata[i].source = -1;
    metadata[i].query_epoch = query_epoch;
    metadata[i].validation_epoch = 0;
    // Measurement establishes only reachability and size. Validation owns the
    // success transition. Publish the complete measurement before its status
    // so a stale or torn record cannot be accepted for the current query.
    __threadfence_system();
    atomic_store_status(&metadata[i].status, kPathNotValidated);
  }
}

__device__ inline void publish_path_status(TargetPathMetadata* metadata,
                                           std::uint32_t query_epoch,
                                           int status) {
  metadata->validation_epoch = query_epoch;
  __threadfence_system();
  atomic_store_status(&metadata->status, status);
}

template <typename EdgeOffset>
__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const EdgeOffset* out_rowptr,
                                         const Index* out_colind,
                                         const int* pred_node,
                                         const EdgeOffset* pred_edge,
                                         const int* level,
                                         TargetPathMetadata* metadata,
                                         std::uint32_t query_epoch,
                                         const int* node_offsets,
                                         const int* edge_offsets,
                                         int path_node_capacity,
                                         int path_edge_capacity,
                                         int* path_nodes,
                                         Offset* path_edges) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    if (metadata[i].query_epoch != query_epoch) {
      publish_path_status(
          &metadata[i], query_epoch, kPathInvalidEpoch);
      continue;
    }
    if (metadata[i].length <= 0) {
      continue;
    }
    const int length = metadata[i].length;
    const int node_begin = node_offsets[i];
    const int node_end = node_offsets[i + 1];
    const int edge_begin = edge_offsets[i];
    const int edge_end = edge_offsets[i + 1];
    if (node_begin < 0 || node_end < node_begin ||
        node_end > path_node_capacity || node_end - node_begin != length ||
        edge_begin < 0 || edge_end < edge_begin ||
        edge_end > path_edge_capacity ||
        edge_end - edge_begin != length - 1) {
      publish_path_status(
          &metadata[i], query_epoch, kPathInvalidOffsets);
      continue;
    }
    int current = targets[i];
    int failure_status = kPathValid;
    path_nodes[node_begin + length - 1] = current;

    for (int j = length - 1; j > 0; --j) {
      const int pred = atomic_load_int(pred_node + current);
      if (pred < 0 || static_cast<Offset>(pred) >= rows || pred == current) {
        failure_status = kPathInvalidPredecessor;
        break;
      }
      const EdgeOffset edge = pred_edge[current];
      if (edge < out_rowptr[pred] || edge >= out_rowptr[pred + 1] ||
          out_colind[edge] != current) {
        failure_status = kPathInvalidEdge;
        break;
      }
      if (atomic_load_int(level + pred) != j - 1) {
        failure_status = kPathInvalidLevel;
        break;
      }
      path_edges[edge_begin + j - 1] = static_cast<Offset>(edge);
      current = pred;
      path_nodes[node_begin + j - 1] = current;
    }
    if (failure_status != kPathValid) {
      publish_path_status(&metadata[i], query_epoch, failure_status);
      continue;
    }
    if (atomic_load_int(level + current) != 0 ||
        atomic_load_int(pred_node + current) != current) {
      publish_path_status(
          &metadata[i], query_epoch, kPathInvalidRoot);
      continue;
    }
    metadata[i].source = current;
    publish_path_status(&metadata[i], query_epoch, kPathValid);
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
  // This is the only full-array initialization for the workspace. Publish its
  // completion before the first query so later runs depend only on their sparse
  // per-route reset, not on another kernel in the initial dispatch burst.
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
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
  copy_control_synchronously(scratch.host_status.get(),
                             scratch.status.get(),
                             kStatusCount * sizeof(int),
                             hipMemcpyDeviceToHost,
                             stream);
  std::array<int, kStatusCount> status{};
  std::copy_n(scratch.host_status.get(), kStatusCount, status.begin());
  return status;
}

template <typename EdgeOffset>
void extract_target_paths_to_result(UnitBfsCsrResult& result,
                                    UnitBfsScratch& scratch,
                                    const EdgeOffset* out_rowptr,
                                    const Index* out_colind,
                                    const EdgeOffset* pred_edge,
                                    const std::vector<int>& targets,
                                    int visited_count,
                                    std::uint32_t query_epoch,
                                    hipStream_t stream) {
  const int target_count = static_cast<int>(targets.size());
  measure_target_paths_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.level.get(),
      query_epoch,
      scratch.target_metadata.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  copy_control_synchronously(scratch.host_target_metadata.get(),
                             scratch.target_metadata.get(),
                             targets.size() * sizeof(TargetPathMetadata),
                             hipMemcpyDeviceToHost,
                             stream);

  result.target_distances.resize(targets.size());
  result.target_sources.assign(targets.size(), -1);
  result.target_path_offsets.assign(targets.size() + 1, 0);
  result.target_edge_offsets.assign(targets.size() + 1, 0);
  bool all_targets_reached = true;
  std::size_t total_nodes = 0;
  std::size_t total_edges = 0;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const TargetPathMetadata& metadata = scratch.host_target_metadata.get()[i];
    if (metadata.query_epoch != query_epoch ||
        metadata.validation_epoch != 0 ||
        metadata.status != kPathNotValidated) {
      std::ostringstream message;
      message << "unit BFS measured stale target metadata"
              << " (target_index=" << i
              << ", target=" << targets[i]
              << ", query_epoch=" << metadata.query_epoch
              << ", expected_epoch=" << query_epoch
              << ", validation_epoch=" << metadata.validation_epoch
              << ", status=" << metadata.status << ')';
      throw std::runtime_error(message.str());
    }
    result.target_distances[i] = metadata.distance;
    result.target_path_offsets[i] = static_cast<int>(total_nodes);
    result.target_edge_offsets[i] = static_cast<int>(total_edges);
    if (metadata.length <= 0 || !std::isfinite(metadata.distance)) {
      all_targets_reached = false;
      continue;
    }
    total_nodes += static_cast<std::size_t>(metadata.length);
    total_edges += static_cast<std::size_t>(metadata.length - 1);
    if (total_nodes > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        total_edges > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("compact unit BFS target paths are too large");
    }
  }
  result.target_path_offsets[targets.size()] = static_cast<int>(total_nodes);
  result.target_edge_offsets[targets.size()] = static_cast<int>(total_edges);

  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    std::copy(result.target_path_offsets.begin(),
              result.target_path_offsets.end(),
              scratch.host_target_node_offsets.get());
    std::copy(result.target_edge_offsets.begin(),
              result.target_edge_offsets.end(),
              scratch.host_target_edge_offsets.get());
    copy_control_synchronously(scratch.target_node_offsets.get(),
                               scratch.host_target_node_offsets.get(),
                               (targets.size() + 1) * sizeof(int),
                               hipMemcpyHostToDevice,
                               stream);
    copy_control_synchronously(scratch.target_edge_offsets.get(),
                               scratch.host_target_edge_offsets.get(),
                               (targets.size() + 1) * sizeof(int),
                               hipMemcpyHostToDevice,
                               stream);

    fill_target_paths_kernel<EdgeOffset>
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.rows,
            out_rowptr,
            out_colind,
            scratch.pred_node.get(),
            pred_edge,
            scratch.level.get(),
            scratch.target_metadata.get(),
            query_epoch,
            scratch.target_node_offsets.get(),
            scratch.target_edge_offsets.get(),
            static_cast<int>(total_nodes),
            static_cast<int>(total_edges),
            scratch.compact_path_nodes.get(),
            scratch.compact_path_edges.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
    // Validation metadata controls whether the compact path is accepted, so
    // make that small transfer host-synchronous. The potentially large node
    // and edge payloads remain asynchronous after the fill is complete.
    copy_control_synchronously(scratch.host_target_metadata.get(),
                               scratch.target_metadata.get(),
                               targets.size() * sizeof(TargetPathMetadata),
                               hipMemcpyDeviceToHost,
                               stream);
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_compact_path_nodes.get(),
                                      scratch.compact_path_nodes.get(),
                                      total_nodes * sizeof(int),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  if (total_edges != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_compact_path_edges.get(),
                                      scratch.compact_path_edges.get(),
                                      total_edges * sizeof(Offset),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  // Reset only the state used to claim a vertex.  Predecessors are overwritten
  // whenever a future search claims that vertex and never need cleanup.
  reset_visited_levels(scratch, visited_count, stream);
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const TargetPathMetadata& metadata = scratch.host_target_metadata.get()[i];
    if (metadata.query_epoch != query_epoch) {
      std::ostringstream message;
      message << "unit BFS target metadata belongs to a stale query"
              << " (target_index=" << i
              << ", target=" << targets[i]
              << ", query_epoch=" << metadata.query_epoch
              << ", expected_epoch=" << query_epoch << ')';
      throw std::runtime_error(message.str());
    }
    if (metadata.length > 0 &&
        metadata.validation_epoch != query_epoch) {
      std::ostringstream message;
      message << "unit BFS target validation was not published"
              << " (target_index=" << i
              << ", target=" << targets[i]
              << ", query_epoch=" << metadata.query_epoch
              << ", validation_epoch=" << metadata.validation_epoch
              << ", expected_epoch=" << query_epoch
              << ", status=" << metadata.status << ')';
      throw std::runtime_error(message.str());
    }
    if (metadata.length > 0 && metadata.status != kPathValid) {
      std::ostringstream message;
      message << "unit BFS predecessor path failed device validation"
              << " (target_index=" << i
              << ", target=" << targets[i]
              << ", status=" << metadata.status
              << " [" << path_status_name(metadata.status) << ']'
              << ", path_length=" << metadata.length
              << ", distance=" << metadata.distance
              << ", query_epoch=" << metadata.query_epoch
              << ", validation_epoch=" << metadata.validation_epoch << ')';
      throw std::runtime_error(message.str());
    }
    if (metadata.length > 0) {
      result.target_sources[i] = metadata.source;
    }
  }
  if (total_nodes != 0) {
    std::copy_n(scratch.host_compact_path_nodes.get(),
                total_nodes,
                result.target_path_nodes.begin());
  }
  if (total_edges != 0) {
    std::copy_n(scratch.host_compact_path_edges.get(),
                total_edges,
                result.target_path_edges.begin());
  }
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
  const std::uint32_t query_epoch = scratch.begin_query();
  copy_control_synchronously(scratch.sources.get(),
                             effective_sources->data(),
                             effective_sources->size() * sizeof(int),
                             hipMemcpyHostToDevice,
                             stream);
  copy_control_synchronously(scratch.targets.get(),
                             targets.data(),
                             targets.size() * sizeof(int),
                             hipMemcpyHostToDevice,
                             stream);

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
  // Publish target marks, sources, controller counters, and the initial queue
  // before the first host-controlled expansion.  This costs one boundary per
  // route, not per BFS level, and does not affect null-stream batching.
  synchronize_explicit_stream(stream);

  int frontier_begin = 0;
  int frontier_end = source_count;
  int queue_tail = source_count;
  int current_count = source_count;
  int found_count = initially_found;
  UnitBfsCsrResult result;
  result.target = -1;
  result.iterations_used = 0;
  const bool use_device_controller =
      stream == nullptr && progress_callback == nullptr;

  while (current_count > 0 && found_count < target_count &&
         result.iterations_used < max_depth) {
    const int previous_queue_tail = queue_tail;
    const int previous_found_count = found_count;
    const int previous_frontier_end = frontier_end;
    const int previous_depth = result.iterations_used;

    if (!use_device_controller) {
      // Parallel PathFinder workers use explicit nonblocking streams.  Keep
      // their frontier bounds and depth on the host, as the pre-batching
      // implementation did.  On gfx1151, an expansion's queue-tail updates
      // have repeatedly become host-visible without the immediately following
      // controller-advance kernel, even though both were submitted to the same
      // stream.  Removing that dependent kernel also removes the failure mode.
      expand_frontier_host_controlled_kernel<EdgeOffset>
          <<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
              frontier_begin,
              frontier_end,
              previous_depth + 1,
              out_rowptr,
              outgoing.colind.get(),
              scratch.level.get(),
              scratch.pred_node.get(),
              pred_edge,
              scratch.frontier_queue.get(),
              scratch.target_multiplicity.get(),
              scratch.status.get() + kStatusQueueTail,
              scratch.status.get() + kStatusFoundCount);
      UNIT_BFS_HIP_CHECK(hipGetLastError());

      const std::array<int, kStatusCount> status =
          copy_status_to_host(scratch, stream);
      const int observed_queue_tail = status[kStatusQueueTail];
      const int observed_found_count = status[kStatusFoundCount];
      if (frontier_begin < 0 || frontier_end < frontier_begin ||
          frontier_end != previous_queue_tail ||
          current_count != frontier_end - frontier_begin ||
          observed_queue_tail < previous_queue_tail ||
          observed_queue_tail > n_int ||
          observed_found_count < previous_found_count ||
          observed_found_count > target_count) {
        std::ostringstream message;
        message << "unit BFS host-controlled frontier state is inconsistent"
                << " (queue_tail=" << observed_queue_tail
                << ", previous_queue_tail=" << previous_queue_tail
                << ", frontier_begin=" << frontier_begin
                << ", frontier_end=" << frontier_end
                << ", found_count=" << observed_found_count
                << ", previous_found_count=" << previous_found_count
                << ", completed_depth=" << previous_depth + 1
                << ", rows=" << n_int
                << ", sources=" << source_count
                << ", targets=" << target_count
                << ", max_depth=" << max_depth << ')';
        throw std::runtime_error(message.str());
      }

      queue_tail = observed_queue_tail;
      found_count = observed_found_count;
      frontier_begin = previous_frontier_end;
      frontier_end = queue_tail;
      result.iterations_used = previous_depth + 1;
    } else {
      // Check the first expansion immediately so the next null-stream batch can
      // be sized from a real frontier.  Later batches enqueue up to four
      // device-controlled levels before copying the controller to the host.
      const int remaining_depth = max_depth - previous_depth;
      const int rounds_to_enqueue =
          previous_depth == 0
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
      const int expected_active =
          frontier_begin < frontier_end && found_count < target_count &&
          result.iterations_used < max_depth;
      if (queue_tail < previous_queue_tail || queue_tail > n_int ||
          frontier_begin < previous_frontier_end ||
          frontier_end < frontier_begin || frontier_end != queue_tail ||
          found_count < previous_found_count || found_count > target_count ||
          status[kStatusActive] != expected_active ||
          result.iterations_used <= previous_depth ||
          result.iterations_used > previous_depth + rounds_to_enqueue) {
        std::ostringstream message;
        message << "unit BFS device frontier state is inconsistent"
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
                << ", active=" << status[kStatusActive]
                << ", expected_active=" << expected_active
                << ", rows=" << n_int
                << ", sources=" << source_count
                << ", targets=" << target_count
                << ", max_depth=" << max_depth << ')';
        throw std::runtime_error(message.str());
      }
    }
    current_count = frontier_end - frontier_begin;

    if (progress_callback) {
      UnitBfsCsrProgress progress;
      progress.iteration = result.iterations_used;
      progress.max_iters = max_depth;
      progress.convergence_checked = true;
      progress.changed = current_count > 0;
      try {
        progress_callback(progress, progress_user_data);
      } catch (...) {
        // A callback is ordinary host code and may throw. Restore every piece
        // of per-run state whose sparse cleanup normally happens below so the
        // workspace remains reusable after the exception.
        clear_target_multiplicity_kernel
            <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
                scratch.targets.get(),
                target_count,
                scratch.target_multiplicity.get());
        UNIT_BFS_HIP_CHECK(hipGetLastError());
        reset_visited_levels(scratch, queue_tail, stream);
        UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
        throw;
      }
    }
  }

  result.converged = current_count == 0 || found_count >= target_count;
  result.stopped_on_target = found_count >= target_count;
  try {
    clear_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.target_multiplicity.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
    extract_target_paths_to_result(result,
                                   scratch,
                                   out_rowptr,
                                   outgoing.colind.get(),
                                   pred_edge,
                                   targets,
                                   queue_tail,
                                   query_epoch,
                                   stream);
    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (!std::isfinite(result.target_distances[i])) {
        continue;
      }
      const int path_source = result.target_sources[i];
      const bool belongs_to_current_sources =
          effective_sources->size() == 1
              ? path_source == effective_sources->front()
              : std::find(effective_sources->begin(),
                          effective_sources->end(),
                          path_source) != effective_sources->end();
      if (!belongs_to_current_sources) {
        throw std::runtime_error(
            "unit BFS predecessor path is detached from the current source set "
            "for target index " +
            std::to_string(i));
      }
    }
  } catch (...) {
    // Result allocation and predecessor validation are host-visible failure
    // points after traversal has completed. Restore the reusable sparse state
    // before preserving the original exception. PathFinder discards a failed
    // worker, but public workspace callers may legitimately recover and retry.
    const std::exception_ptr extraction_exception = std::current_exception();
    // Extraction can throw after submitting a measurement/fill kernel but
    // before its normal copy synchronization. Drain those readers before the
    // cleanup kernels overwrite level/predecessor state.
    UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
    clear_target_multiplicity_kernel
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.target_multiplicity.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
    reset_visited_levels(scratch, queue_tail, stream);
    UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
    std::rethrow_exception(extraction_exception);
  }
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
  int device = 0;
  unit_bfs_detail::OutgoingCsrOwner outgoing;

  Impl(const HostCsrF32& host,
       hipStream_t stream,
       UnitBfsCsrOffsetMode offset_mode)
      : device(unit_bfs_detail::current_hip_device()),
        outgoing(unit_bfs_detail::copy_host_csr_to_device(
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
  impl_ = std::make_shared<Impl>(adjacency, stream, offset_mode);
}

UnitBfsCsrGraph::~UnitBfsCsrGraph() = default;
UnitBfsCsrGraph::UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept = default;
UnitBfsCsrGraph& UnitBfsCsrGraph::operator=(UnitBfsCsrGraph&&) noexcept = default;

bool UnitBfsCsrGraph::uses_32_bit_offsets() const noexcept {
  return impl_ && impl_->outgoing.uses_32_bit_offsets;
}

struct UnitBfsCsrWorkspace::Impl {
  std::shared_ptr<const UnitBfsCsrGraph::Impl> graph;
  unit_bfs_detail::UnitBfsScratch scratch;
  hipStream_t stream = nullptr;

  static std::shared_ptr<const UnitBfsCsrGraph::Impl> require_graph(
      const std::shared_ptr<const UnitBfsCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("unit BFS shared graph must not be null");
    }
    const std::shared_ptr<const UnitBfsCsrGraph::Impl> graph =
        candidate->impl_;
    if (unit_bfs_detail::current_hip_device() != graph->device) {
      throw std::invalid_argument(
          "unit BFS shared graph belongs to a different HIP device");
    }
    return graph;
  }

  Impl(std::shared_ptr<const UnitBfsCsrGraph> graph_, hipStream_t stream)
      : graph(require_graph(graph_)),
        scratch(graph->outgoing.rows,
                graph->outgoing.uses_32_bit_offsets),
        stream(stream) {}

  void require_run_context(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "UnitBfsCsrWorkspace is stream-affine; use its construction stream");
    }
    if (unit_bfs_detail::current_hip_device() != graph->device) {
      throw std::invalid_argument(
          "UnitBfsCsrWorkspace is running on a different HIP device");
    }
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
  impl_->require_run_context(stream);
  validate_sources_targets(impl_->scratch.rows, sources, targets);
  return run_unit_bfs_impl(impl_->graph->outgoing,
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
