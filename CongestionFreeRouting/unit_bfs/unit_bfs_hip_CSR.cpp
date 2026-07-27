#include "unit_bfs_hip_CSR.hpp"
#include "unit_bfs_policy.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
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
using PackedVisitState = unsigned long long;

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
constexpr int kCooperativeLevelsPerLaunch = 32;
constexpr int kMaxConcurrentCooperativeWorkers = 8;
constexpr int kTargetConcurrentCooperativeWorkers = 4;

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

enum UnitBfsOffsetScanStatus : int {
  kOffsetScanNotPublished = 0,
  kOffsetScanValid = 1,
  kOffsetScanInvalidMetadata = -1,
  kOffsetScanOverflow = -2,
};

struct TargetPathMetadata {
  float distance;
  int length;
  int source;
  int status;
  std::uint32_t query_epoch;
  std::uint32_t validation_epoch;
};

struct TargetPathTotals {
  int total_nodes;
  int total_edges;
  int status;
  std::uint32_t query_epoch;
};

static_assert(sizeof(TargetPathTotals) == 4 * sizeof(std::uint32_t),
              "unit BFS totals descriptor must remain one packed record");

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
    if (count == count_) return;
    if (count == 0) {
      release();
      return;
    }
    T* candidate = nullptr;
    UNIT_BFS_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&candidate),
                                sssp_capacity::checked_bytes<T>(count)));
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
    }
    ptr_ = candidate;
    count_ = count;
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
                        sssp_capacity::checked_bytes<T>(count),
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
  int sparse_cooperative_launch_blocks = 0;
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
                     ? sssp_capacity::checked_add(
                           static_cast<std::size_t>(rows_), 1)
                     : 0),
        rowptr64(uses_32_bit_offsets
                     ? 0
                     : sssp_capacity::checked_add(
                           static_cast<std::size_t>(rows_), 1)),
        colind(static_cast<std::size_t>(nnz_)) {}
};

struct UnitBfsScratch {
  Offset rows = 0;
  bool uses_32_bit_offsets = false;
  UnitBfsCsrExtractionMode extraction_mode =
      UnitBfsCsrExtractionMode::kHostOffsets;
  UnitBfsCsrVisitationMode visitation_mode =
      UnitBfsCsrVisitationMode::kSparseReset;
  int generation_cooperative_launch_blocks = 0;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<int> target_multiplicity;
  DeviceBuffer<int> level;
  DeviceBuffer<PackedVisitState> generation_level;
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
  DeviceBuffer<TargetPathTotals> target_totals;
  PinnedHostBuffer<TargetPathTotals> host_target_totals;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;
  PinnedHostBuffer<int> host_compact_path_nodes;
  PinnedHostBuffer<Offset> host_compact_path_edges;
  std::uint32_t query_epoch = 0;
  std::uint32_t visitation_generation = 0;
  bool initialized = false;

  UnitBfsScratch() = default;
  UnitBfsScratch(Offset rows_,
                 bool uses_32_bit_offsets_,
                 UnitBfsCsrWorkspaceOptions options)
      : rows(rows_),
        uses_32_bit_offsets(uses_32_bit_offsets_),
        extraction_mode(options.extraction_mode),
        visitation_mode(options.visitation_mode),
        target_multiplicity(static_cast<std::size_t>(rows_)),
        level(visitation_mode == UnitBfsCsrVisitationMode::kSparseReset
                  ? static_cast<std::size_t>(rows_)
                  : 0),
        generation_level(
            visitation_mode == UnitBfsCsrVisitationMode::kGenerationStamped
                ? static_cast<std::size_t>(rows_)
                : 0),
        pred_node(static_cast<std::size_t>(rows_)),
        pred_edge32(uses_32_bit_offsets
                        ? static_cast<std::size_t>(rows_)
                        : 0),
        pred_edge64(uses_32_bit_offsets
                        ? 0
                        : static_cast<std::size_t>(rows_)),
        frontier_queue(static_cast<std::size_t>(rows_)),
        status(kStatusCount),
        host_status(kStatusCount),
        target_totals(
            extraction_mode == UnitBfsCsrExtractionMode::kDeviceOffsets ? 1
                                                                         : 0),
        host_target_totals(
            extraction_mode == UnitBfsCsrExtractionMode::kDeviceOffsets ? 1
                                                                         : 0) {
    sssp_capacity::validate_reservation(options.capacity_hints);
    ensure_source_capacity(options.capacity_hints.max_sources);
    ensure_target_capacity(options.capacity_hints.max_targets);
  }

  void ensure_source_capacity(std::size_t source_count) {
    if (sources.size() < source_count) {
      sources.reset(unit_bfs_policy::bounded_geometric_capacity(
          sources.size(),
          sssp_capacity::checked_device_count(source_count),
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    if (target_count == 0) return;
    const std::size_t capacity =
        unit_bfs_policy::bounded_geometric_capacity(
            targets.size(),
            sssp_capacity::checked_device_count(target_count),
            static_cast<std::size_t>(std::numeric_limits<int>::max()));
    const std::size_t offset_capacity =
        sssp_capacity::checked_target_offset_count(capacity);
    if (targets.size() < target_count) {
      targets.reset(capacity);
    }
    if (target_metadata.size() < target_count) {
      target_metadata.reset(capacity);
    }
    if (host_target_metadata.size() < target_count) {
      host_target_metadata.reset(capacity);
    }
    const std::size_t required_offsets =
        sssp_capacity::checked_target_offset_count(target_count);
    if (target_node_offsets.size() < required_offsets) {
      target_node_offsets.reset(offset_capacity);
    }
    if (host_target_node_offsets.size() < required_offsets) {
      host_target_node_offsets.reset(offset_capacity);
    }
    if (target_edge_offsets.size() < required_offsets) {
      target_edge_offsets.reset(offset_capacity);
    }
    if (host_target_edge_offsets.size() < required_offsets) {
      host_target_edge_offsets.reset(offset_capacity);
    }
  }

  void ensure_compact_path_capacity(std::size_t node_count,
                                    std::size_t edge_count) {
    const std::size_t device_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (compact_path_nodes.size() < node_count) {
      compact_path_nodes.reset(unit_bfs_policy::bounded_geometric_capacity(
          compact_path_nodes.size(),
          sssp_capacity::checked_device_count(node_count),
          device_limit));
    }
    if (host_compact_path_nodes.size() < node_count) {
      host_compact_path_nodes.reset(compact_path_nodes.size());
    }
    if (compact_path_edges.size() < edge_count) {
      compact_path_edges.reset(unit_bfs_policy::bounded_geometric_capacity(
          compact_path_edges.size(),
          sssp_capacity::checked_device_count(edge_count),
          device_limit));
    }
    if (host_compact_path_edges.size() < edge_count) {
      host_compact_path_edges.reset(compact_path_edges.size());
    }
  }

  std::uint32_t begin_query() {
    if (query_epoch == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("unit BFS workspace query epoch exhausted");
    }
    return ++query_epoch;
  }

  std::uint32_t begin_generation_query() {
    if (visitation_generation == std::numeric_limits<std::uint32_t>::max()) {
      return 0;
    }
    return ++visitation_generation;
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

__device__ inline PackedVisitState atomic_load_packed_visit(
    const PackedVisitState* address) {
  return atomicCAS(const_cast<PackedVisitState*>(address), 0ULL, 0ULL);
}

__device__ inline PackedVisitState packed_visit(std::uint32_t generation,
                                                int level) {
  return (static_cast<PackedVisitState>(generation) << 32) |
         static_cast<std::uint32_t>(level);
}

template <bool UseGeneration>
__device__ inline int load_visit_level(const int* level,
                                       const PackedVisitState* generation_level,
                                       int vertex,
                                       std::uint32_t generation) {
  if constexpr (!UseGeneration) {
    return atomic_load_int(level + vertex);
  } else {
    const PackedVisitState state =
        atomic_load_packed_visit(generation_level + vertex);
    return static_cast<std::uint32_t>(state >> 32) == generation
               ? static_cast<int>(static_cast<std::uint32_t>(state))
               : kUnvisited;
  }
}

template <bool UseGeneration>
__device__ inline bool claim_visit(int* level,
                                   PackedVisitState* generation_level,
                                   int vertex,
                                   std::uint32_t generation,
                                   int next_level) {
  if constexpr (!UseGeneration) {
    return atomicCAS(&level[vertex], kUnvisited, next_level) == kUnvisited;
  } else {
    PackedVisitState observed =
        atomic_load_packed_visit(generation_level + vertex);
    const PackedVisitState desired = packed_visit(generation, next_level);
    while (static_cast<std::uint32_t>(observed >> 32) != generation) {
      const PackedVisitState prior =
          atomicCAS(generation_level + vertex, observed, desired);
      if (prior == observed) return true;
      observed = prior;
    }
    return false;
  }
}

template <bool UseGeneration>
__device__ inline void publish_source_visit(
    int* level,
    PackedVisitState* generation_level,
    int vertex,
    std::uint32_t generation) {
  if constexpr (!UseGeneration) {
    atomicExch(level + vertex, 0);
  } else {
    atomicExch(generation_level + vertex, packed_visit(generation, 0));
  }
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

template <bool UseGeneration>
__global__ void initialize_bfs_arrays_kernel(
    Offset rows,
    int* level,
    PackedVisitState* generation_level,
    int* target_multiplicity) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    if constexpr (!UseGeneration) {
      atomicExch(&level[v], kUnvisited);
    } else {
      atomicExch(&generation_level[v], 0ULL);
    }
    atomicExch(&target_multiplicity[v], 0);
  }
}

__global__ void reset_generation_levels_kernel(
    Offset rows,
    PackedVisitState* generation_level) {
  for (Offset v = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < rows;
       v += static_cast<Offset>(blockDim.x) * gridDim.x) {
    atomicExch(&generation_level[v], 0ULL);
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

template <typename EdgeOffset, bool UseGeneration>
__global__ void initialize_sources_kernel(const int* sources,
                                          int source_count,
                                          int initially_found,
                                          int target_count,
                                          int max_depth,
                                          std::uint32_t generation,
                                          int* level,
                                          PackedVisitState* generation_level,
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
    publish_source_visit<UseGeneration>(
        level, generation_level, source, generation);
  }
}

template <typename EdgeOffset, bool UseGeneration>
__device__ inline void expand_frontier_range(
    int frontier_begin,
    int frontier_end,
    int next_level,
    const EdgeOffset* out_rowptr,
    const Index* out_colind,
    std::uint32_t generation,
    int* level,
    PackedVisitState* generation_level,
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
      const bool claimed = claim_visit<UseGeneration>(
          level, generation_level, v, generation, next_level);
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

template <typename EdgeOffset, bool UseGeneration>
__global__ void expand_frontier_kernel(const EdgeOffset* out_rowptr,
                                       const Index* out_colind,
                                       std::uint32_t generation,
                                       int* level,
                                       PackedVisitState* generation_level,
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
  expand_frontier_range<EdgeOffset, UseGeneration>(
      controller[1],
      controller[2],
      controller[3] + 1,
      out_rowptr,
      out_colind,
      generation,
      level,
      generation_level,
      pred_node,
      pred_edge,
      frontier_queue,
      target_multiplicity,
      status + kStatusQueueTail,
      status + kStatusFoundCount);
}

template <typename EdgeOffset, bool UseGeneration>
__global__ void expand_frontier_host_controlled_kernel(
    int frontier_begin,
    int frontier_end,
    int next_level,
    const EdgeOffset* out_rowptr,
    const Index* out_colind,
    std::uint32_t generation,
    int* level,
    PackedVisitState* generation_level,
    int* pred_node,
    EdgeOffset* pred_edge,
    int* frontier_queue,
    const int* target_multiplicity,
    int* queue_tail,
    int* found_count) {
  expand_frontier_range<EdgeOffset, UseGeneration>(frontier_begin,
                                                   frontier_end,
                                                   next_level,
                                                   out_rowptr,
                                                   out_colind,
                                                   generation,
                                                   level,
                                                   generation_level,
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

template <typename EdgeOffset, bool UseGeneration>
__global__ void cooperative_frontier_controller_kernel(
    const EdgeOffset* out_rowptr,
    const Index* out_colind,
    int target_count,
    int max_depth,
    int level_budget,
    std::uint32_t generation,
    int* level,
    PackedVisitState* generation_level,
    int* pred_node,
    EdgeOffset* pred_edge,
    int* frontier_queue,
    const int* target_multiplicity,
    int* status) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __shared__ int controller[4];

  // initialize_sources_kernel publishes a complete initial controller before
  // this kernel is submitted.  Every later controller transition happens in
  // this kernel, separated from frontier expansion by a grid-wide barrier, so
  // no cross-kernel controller handoff or per-level host copy is required.
  int levels_this_launch = 0;
  while (levels_this_launch < level_budget) {
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
      break;
    }

    const int frontier_begin = controller[1];
    const int frontier_end = controller[2];
    const int next_level = controller[3] + 1;

    expand_frontier_range<EdgeOffset, UseGeneration>(
        frontier_begin,
        frontier_end,
        next_level,
        out_rowptr,
        out_colind,
        generation,
        level,
        generation_level,
        pred_node,
        pred_edge,
        frontier_queue,
        target_multiplicity,
        status + kStatusQueueTail,
        status + kStatusFoundCount);
    grid.sync();

    if (grid.thread_rank() == 0) {
      const int next_frontier_begin = frontier_end;
      const int next_frontier_end =
          atomic_load_status(status + kStatusQueueTail);
      const int found_count =
          atomic_load_status(status + kStatusFoundCount);
      atomic_store_status(
          status + kStatusFrontierBegin, next_frontier_begin);
      atomic_store_status(status + kStatusFrontierEnd, next_frontier_end);
      atomic_store_status(status + kStatusCompletedDepth, next_level);
      __threadfence();
      atomic_store_status(
          status + kStatusActive,
          next_frontier_begin < next_frontier_end &&
              found_count < target_count && next_level < max_depth);
    }
    grid.sync();
    ++levels_this_launch;
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

template <bool UseGeneration>
__global__ void measure_target_paths_kernel(
    const int* targets,
    int target_count,
    const int* level,
    const PackedVisitState* generation_level,
    std::uint32_t query_epoch,
    TargetPathMetadata* metadata) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < target_count;
       i += blockDim.x * gridDim.x) {
    const int target = targets[i];
    const int target_level = load_visit_level<UseGeneration>(
        level, generation_level, target, query_epoch);
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

__global__ void scan_target_path_offsets_kernel(
    const TargetPathMetadata* metadata,
    int target_count,
    std::uint32_t query_epoch,
    int* node_offsets,
    int* edge_offsets,
    TargetPathTotals* totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  totals->query_epoch = query_epoch;
  totals->total_nodes = 0;
  totals->total_edges = 0;
  atomic_store_status(&totals->status, kOffsetScanNotPublished);
  node_offsets[0] = 0;
  edge_offsets[0] = 0;
  int total_nodes = 0;
  int total_edges = 0;
  int scan_status = kOffsetScanValid;
  for (int i = 0; i < target_count; ++i) {
    const TargetPathMetadata item = metadata[i];
    if (item.query_epoch != query_epoch || item.validation_epoch != 0 ||
        item.status != kPathNotValidated) {
      scan_status = kOffsetScanInvalidMetadata;
      break;
    }
    const int nodes = item.length > 0 && isfinite(item.distance) ? item.length : 0;
    const int edges = nodes > 0 ? nodes - 1 : 0;
    if (nodes < 0 || edges < 0 || total_nodes > INT_MAX - nodes ||
        total_edges > INT_MAX - edges) {
      scan_status = kOffsetScanOverflow;
      break;
    }
    total_nodes += nodes;
    total_edges += edges;
    node_offsets[i + 1] = total_nodes;
    edge_offsets[i + 1] = total_edges;
  }
  totals->total_nodes = total_nodes;
  totals->total_edges = total_edges;
  __threadfence_system();
  atomic_store_status(&totals->status, scan_status);
}

__device__ inline void publish_path_status(TargetPathMetadata* metadata,
                                           std::uint32_t query_epoch,
                                           int status) {
  metadata->validation_epoch = query_epoch;
  __threadfence_system();
  atomic_store_status(&metadata->status, status);
}

template <typename EdgeOffset, bool UseGeneration>
__global__ void fill_target_paths_kernel(const int* targets,
                                         int target_count,
                                         Offset rows,
                                         const EdgeOffset* out_rowptr,
                                         const Index* out_colind,
                                         const int* pred_node,
                                         const EdgeOffset* pred_edge,
                                         const int* level,
                                         const PackedVisitState* generation_level,
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
      if (load_visit_level<UseGeneration>(
              level, generation_level, pred, query_epoch) != j - 1) {
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
    if (load_visit_level<UseGeneration>(
            level, generation_level, current, query_epoch) != 0 ||
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

template <typename EdgeOffset, bool UseGeneration>
int cooperative_controller_blocks(Offset rows) {
  int device = -1;
  UNIT_BFS_HIP_CHECK(hipGetDevice(&device));
  int cooperative_launch = 0;
  const hipError_t capability_status =
      hipDeviceGetAttribute(&cooperative_launch,
                            hipDeviceAttributeCooperativeLaunch,
                            device);
  if (capability_status != hipSuccess || cooperative_launch == 0) {
    if (capability_status != hipSuccess) {
      (void)hipGetLastError();
    }
    return 0;
  }

  hipDeviceProp_t properties{};
  UNIT_BFS_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  int active_blocks_per_compute_unit = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks_per_compute_unit,
          cooperative_frontier_controller_kernel<EdgeOffset, UseGeneration>,
          kBlockSize,
          0);
  if (occupancy_status != hipSuccess ||
      active_blocks_per_compute_unit <= 0 ||
      properties.multiProcessorCount <= 0) {
    if (occupancy_status != hipSuccess) {
      (void)hipGetLastError();
    }
    return 0;
  }

  const Offset row_blocks =
      (rows + static_cast<Offset>(kBlockSize) - 1) /
      static_cast<Offset>(kBlockSize);
  const Offset legal_resident_limit =
      static_cast<Offset>(active_blocks_per_compute_unit) *
      static_cast<Offset>(properties.multiProcessorCount);
  // PathFinder can run eight UnitBFS workspaces at once. Divide legal residency
  // across that maximum, then target roughly one aggregate block per CU at the
  // measured four-worker baseline. This avoids device-wide barriers across a
  // mostly idle CU-count grid for every small routing frontier while retaining
  // enough residency headroom for the eight-worker comparison.
  const Offset per_worker_resident_limit =
      std::max<Offset>(
          1,
          legal_resident_limit /
              static_cast<Offset>(kMaxConcurrentCooperativeWorkers));
  const Offset balanced_worker_blocks =
      (static_cast<Offset>(properties.multiProcessorCount) +
       static_cast<Offset>(kTargetConcurrentCooperativeWorkers) - 1) /
      static_cast<Offset>(kTargetConcurrentCooperativeWorkers);
  const Offset concurrency_friendly_limit =
      std::min(balanced_worker_blocks, per_worker_resident_limit);
  const Offset blocks =
      std::min(row_blocks,
               std::min(concurrency_friendly_limit, legal_resident_limit));
  if (blocks <= 0 ||
      blocks > static_cast<Offset>(std::numeric_limits<int>::max())) {
    return 0;
  }
  return static_cast<int>(blocks);
}

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
  device.sparse_cooperative_launch_blocks =
      uses_32_bit_offsets
          ? cooperative_controller_blocks<CompactOffset, false>(host.rows)
          : cooperative_controller_blocks<Offset, false>(host.rows);
  const std::size_t rows = checked_size(host.rows, "rows");
  const std::size_t nnz = checked_size(host.nnz, "nnz");
  if (nnz != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.colind.get(),
                                      host.colind.data(),
                                      sssp_capacity::checked_bytes<Index>(nnz),
                                      hipMemcpyHostToDevice,
                                      stream));
  }
  std::vector<CompactOffset> compact_rowptr;
  if (uses_32_bit_offsets) {
    compact_rowptr.resize(sssp_capacity::checked_add(rows, 1));
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
                                 sssp_capacity::checked_bytes<CompactOffset>(
                                     sssp_capacity::checked_add(rows, 1)),
                                 hipMemcpyHostToDevice));
  } else {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(device.rowptr64.get(),
                                      host.rowptr.data(),
                                      sssp_capacity::checked_bytes<Offset>(
                                          sssp_capacity::checked_add(rows, 1)),
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

  if (scratch.visitation_mode == UnitBfsCsrVisitationMode::kSparseReset) {
    initialize_bfs_arrays_kernel<false>
        <<<grid_for_items(rows), kBlockSize, 0, stream>>>(
            rows,
            scratch.level.get(),
            nullptr,
            scratch.target_multiplicity.get());
  } else {
    initialize_bfs_arrays_kernel<true>
        <<<grid_for_items(rows), kBlockSize, 0, stream>>>(
            rows,
            nullptr,
            scratch.generation_level.get(),
            scratch.target_multiplicity.get());
  }
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
  if (scratch.visitation_mode ==
      UnitBfsCsrVisitationMode::kGenerationStamped) {
    return;
  }
  if (visited_count > 0) {
    reset_visited_levels_kernel<<<grid_for_items(visited_count),
                                  kBlockSize,
                                  0,
                                  stream>>>(
        scratch.frontier_queue.get(), visited_count, scratch.level.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
  }
}

std::uint32_t begin_query_epoch(UnitBfsScratch& scratch,
                                hipStream_t stream) {
  if (scratch.visitation_mode == UnitBfsCsrVisitationMode::kSparseReset) {
    return scratch.begin_query();
  }
  std::uint32_t generation = scratch.begin_generation_query();
  if (generation != 0) return generation;

  // No earlier query may still observe the generation being recycled. Every
  // successful/exceptional run already drains this stream, and this explicit
  // boundary makes rollover safe even if that contract changes later.
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  reset_generation_levels_kernel
      <<<grid_for_items(scratch.rows), kBlockSize, 0, stream>>>(
          scratch.rows, scratch.generation_level.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  scratch.visitation_generation = 0;
  generation = scratch.begin_generation_query();
  if (generation == 0) {
    throw std::logic_error("unit BFS generation rollover did not restart");
  }
  return generation;
}

std::array<int, kStatusCount> copy_status_to_host(UnitBfsScratch& scratch,
                                                  hipStream_t stream) {
  copy_control_synchronously(scratch.host_status.get(),
                             scratch.status.get(),
                             sssp_capacity::checked_bytes<int>(kStatusCount),
                             hipMemcpyDeviceToHost,
                             stream);
  std::array<int, kStatusCount> status{};
  std::copy_n(scratch.host_status.get(), kStatusCount, status.begin());
  return status;
}

template <typename EdgeOffset, bool UseGeneration>
void launch_cooperative_controller(
    const OutgoingCsrOwner& outgoing,
    const EdgeOffset* out_rowptr,
    UnitBfsScratch& scratch,
    EdgeOffset* pred_edge,
    std::uint32_t generation,
    int target_count,
    int max_depth,
    hipStream_t stream) {
  const EdgeOffset* out_rowptr_arg = out_rowptr;
  const Index* out_colind_arg = outgoing.colind.get();
  int target_count_arg = target_count;
  int max_depth_arg = max_depth;
  int level_budget_arg = kCooperativeLevelsPerLaunch;
  std::uint32_t generation_arg = generation;
  int* level_arg = scratch.level.get();
  PackedVisitState* generation_level_arg = scratch.generation_level.get();
  int* pred_node_arg = scratch.pred_node.get();
  EdgeOffset* pred_edge_arg = pred_edge;
  int* frontier_queue_arg = scratch.frontier_queue.get();
  const int* target_multiplicity_arg = scratch.target_multiplicity.get();
  int* status_arg = scratch.status.get();
  void* kernel_args[] = {
      &out_rowptr_arg,
      &out_colind_arg,
      &target_count_arg,
      &max_depth_arg,
      &level_budget_arg,
      &generation_arg,
      &level_arg,
      &generation_level_arg,
      &pred_node_arg,
      &pred_edge_arg,
      &frontier_queue_arg,
      &target_multiplicity_arg,
      &status_arg,
  };

  UNIT_BFS_HIP_CHECK(hipLaunchCooperativeKernel(
      cooperative_frontier_controller_kernel<EdgeOffset, UseGeneration>,
      dim3(static_cast<unsigned>(
          UseGeneration ? scratch.generation_cooperative_launch_blocks
                        : outgoing.sparse_cooperative_launch_blocks)),
      dim3(kBlockSize),
      kernel_args,
      0,
      stream));
}

template <typename EdgeOffset, bool UseGeneration>
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
  const std::size_t target_offset_count =
      sssp_capacity::checked_target_offset_count(targets.size());
  measure_target_paths_kernel<UseGeneration>
      <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
          scratch.targets.get(),
          target_count,
          scratch.level.get(),
          scratch.generation_level.get(),
          query_epoch,
          scratch.target_metadata.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  result.target_distances.resize(targets.size());
  result.target_sources.assign(targets.size(), -1);
  result.target_path_offsets.assign(target_offset_count, 0);
  result.target_edge_offsets.assign(target_offset_count, 0);
  std::size_t total_nodes = 0;
  std::size_t total_edges = 0;
  const bool use_device_offsets =
      scratch.extraction_mode == UnitBfsCsrExtractionMode::kDeviceOffsets;
  if (!use_device_offsets) {
    copy_control_synchronously(
        scratch.host_target_metadata.get(),
        scratch.target_metadata.get(),
        sssp_capacity::checked_bytes<TargetPathMetadata>(targets.size()),
        hipMemcpyDeviceToHost,
        stream);
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
      result.target_path_offsets[i] = static_cast<int>(total_nodes);
      result.target_edge_offsets[i] = static_cast<int>(total_edges);
      if (metadata.length <= 0 || !std::isfinite(metadata.distance)) continue;
      total_nodes = sssp_capacity::checked_add(
          total_nodes, static_cast<std::size_t>(metadata.length));
      total_edges = sssp_capacity::checked_add(
          total_edges, static_cast<std::size_t>(metadata.length - 1));
      if (total_nodes >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          total_edges >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("compact unit BFS target paths are too large");
      }
    }
    result.target_path_offsets[targets.size()] = static_cast<int>(total_nodes);
    result.target_edge_offsets[targets.size()] = static_cast<int>(total_edges);
  } else {
    scan_target_path_offsets_kernel<<<1, 1, 0, stream>>>(
        scratch.target_metadata.get(),
        target_count,
        query_epoch,
        scratch.target_node_offsets.get(),
        scratch.target_edge_offsets.get(),
        scratch.target_totals.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
    copy_control_synchronously(
        scratch.host_target_totals.get(),
        scratch.target_totals.get(),
        sssp_capacity::checked_bytes<TargetPathTotals>(1),
        hipMemcpyDeviceToHost,
        stream);
    const TargetPathTotals totals = scratch.host_target_totals.get()[0];
    if (totals.query_epoch != query_epoch ||
        totals.status != kOffsetScanValid) {
      std::ostringstream message;
      message << "unit BFS device compact-offset scan failed"
              << " (status=" << totals.status
              << ", query_epoch=" << totals.query_epoch
              << ", expected_epoch=" << query_epoch << ')';
      if (totals.status == kOffsetScanOverflow) {
        throw std::overflow_error(message.str());
      }
      throw std::runtime_error(message.str());
    }
    if (totals.total_nodes < 0 || totals.total_edges < 0 ||
        totals.total_edges > totals.total_nodes) {
      throw std::runtime_error(
          "unit BFS device compact-offset totals are inconsistent");
    }
    total_nodes = static_cast<std::size_t>(totals.total_nodes);
    total_edges = static_cast<std::size_t>(totals.total_edges);
  }

  result.target_path_nodes.resize(total_nodes);
  result.target_path_edges.resize(total_edges);
  if (total_nodes != 0) {
    scratch.ensure_compact_path_capacity(total_nodes, total_edges);
    if (!use_device_offsets) {
      std::copy(result.target_path_offsets.begin(),
                result.target_path_offsets.end(),
                scratch.host_target_node_offsets.get());
      std::copy(result.target_edge_offsets.begin(),
                result.target_edge_offsets.end(),
                scratch.host_target_edge_offsets.get());
      copy_control_synchronously(
          scratch.target_node_offsets.get(),
          scratch.host_target_node_offsets.get(),
          sssp_capacity::checked_bytes<int>(target_offset_count),
          hipMemcpyHostToDevice,
          stream);
      copy_control_synchronously(
          scratch.target_edge_offsets.get(),
          scratch.host_target_edge_offsets.get(),
          sssp_capacity::checked_bytes<int>(target_offset_count),
          hipMemcpyHostToDevice,
          stream);
    }

    fill_target_paths_kernel<EdgeOffset, UseGeneration>
        <<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.rows,
            out_rowptr,
            out_colind,
            scratch.pred_node.get(),
            pred_edge,
            scratch.level.get(),
            scratch.generation_level.get(),
            scratch.target_metadata.get(),
            query_epoch,
            scratch.target_node_offsets.get(),
            scratch.target_edge_offsets.get(),
            static_cast<int>(total_nodes),
            static_cast<int>(total_edges),
            scratch.compact_path_nodes.get(),
            scratch.compact_path_edges.get());
    UNIT_BFS_HIP_CHECK(hipGetLastError());
  }
  if (use_device_offsets || total_nodes != 0) {
    // Validation metadata controls whether the compact path is accepted. In
    // device-offset mode this is also the first metadata transfer: the earlier
    // host prefix sum and its two H2D offset transfers are absent.
    copy_control_synchronously(scratch.host_target_metadata.get(),
                               scratch.target_metadata.get(),
                               sssp_capacity::checked_bytes<TargetPathMetadata>(
                                   targets.size()),
                               hipMemcpyDeviceToHost,
                               stream);
  }
  if (use_device_offsets) {
    // Offsets are control records: preserve the gfx1151 guarded transfer used
    // for the totals/metadata descriptors. The new path still eliminates the
    // host prefix sum and both H2D offset copies.
    copy_control_synchronously(
        scratch.host_target_node_offsets.get(),
        scratch.target_node_offsets.get(),
        sssp_capacity::checked_bytes<int>(target_offset_count),
        hipMemcpyDeviceToHost,
        stream);
    copy_control_synchronously(
        scratch.host_target_edge_offsets.get(),
        scratch.target_edge_offsets.get(),
        sssp_capacity::checked_bytes<int>(target_offset_count),
        hipMemcpyDeviceToHost,
        stream);
  }
  if (total_nodes != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_compact_path_nodes.get(),
                                      scratch.compact_path_nodes.get(),
                                      sssp_capacity::checked_bytes<int>(
                                          total_nodes),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  if (total_edges != 0) {
    UNIT_BFS_HIP_CHECK(hipMemcpyAsync(scratch.host_compact_path_edges.get(),
                                      scratch.compact_path_edges.get(),
                                      sssp_capacity::checked_bytes<Offset>(
                                          total_edges),
                                      hipMemcpyDeviceToHost,
                                      stream));
  }
  // Reset only the state used to claim a vertex.  Predecessors are overwritten
  // whenever a future search claims that vertex and never need cleanup.
  reset_visited_levels(scratch, visited_count, stream);
  UNIT_BFS_HIP_CHECK(hipStreamSynchronize(stream));
  if (use_device_offsets) {
    std::copy_n(scratch.host_target_node_offsets.get(),
                target_offset_count,
                result.target_path_offsets.begin());
    std::copy_n(scratch.host_target_edge_offsets.get(),
                target_offset_count,
                result.target_edge_offsets.begin());
    if (result.target_path_offsets.front() != 0 ||
        result.target_edge_offsets.front() != 0 ||
        result.target_path_offsets.back() != static_cast<int>(total_nodes) ||
        result.target_edge_offsets.back() != static_cast<int>(total_edges)) {
      throw std::runtime_error(
          "unit BFS copied compact offsets do not match device totals");
    }
  }
  bool all_targets_reached = true;
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
    result.target_distances[i] = metadata.distance;
    if (metadata.length <= 0 || !std::isfinite(metadata.distance)) {
      all_targets_reached = false;
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

template <typename EdgeOffset, bool UseGeneration>
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
  const std::uint32_t query_epoch = begin_query_epoch(scratch, stream);
  copy_control_synchronously(scratch.sources.get(),
                             effective_sources->data(),
                             sssp_capacity::checked_bytes<int>(
                                 effective_sources->size()),
                             hipMemcpyHostToDevice,
                             stream);
  copy_control_synchronously(scratch.targets.get(),
                             targets.data(),
                             sssp_capacity::checked_bytes<int>(targets.size()),
                             hipMemcpyHostToDevice,
                             stream);

  mark_target_multiplicity_kernel<<<grid_for_items(target_count), kBlockSize, 0, stream>>>(
      scratch.targets.get(),
      target_count,
      scratch.target_multiplicity.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());

  initialize_sources_kernel<EdgeOffset, UseGeneration>
      <<<grid_for_items(source_count), kBlockSize, 0, stream>>>(
          scratch.sources.get(),
          source_count,
          initially_found,
          target_count,
          max_depth,
          query_epoch,
          scratch.level.get(),
          scratch.generation_level.get(),
          scratch.pred_node.get(),
          pred_edge,
          scratch.frontier_queue.get(),
          scratch.status.get());
  UNIT_BFS_HIP_CHECK(hipGetLastError());
  // Publish target marks, sources, controller counters, and the initial queue
  // before traversal starts. This one explicit-stream boundary also isolates
  // the cooperative controller from the target runtime's unreliable
  // cross-kernel controller handoff.
  synchronize_explicit_stream(stream);

  int frontier_begin = 0;
  int frontier_end = source_count;
  int queue_tail = source_count;
  int current_count = source_count;
  int found_count = initially_found;
  UnitBfsCsrResult result;
  result.target = -1;
  result.iterations_used = 0;
  const int cooperative_launch_blocks =
      UseGeneration ? scratch.generation_cooperative_launch_blocks
                    : outgoing.sparse_cooperative_launch_blocks;
  const bool use_cooperative_controller =
      unit_bfs_policy::use_cooperative_controller(
          progress_callback != nullptr,
          stream != nullptr,
          cooperative_launch_blocks);
  const bool use_batched_device_controller =
      stream == nullptr && progress_callback == nullptr;

  while (current_count > 0 && found_count < target_count &&
         result.iterations_used < max_depth) {
    const int previous_queue_tail = queue_tail;
    const int previous_found_count = found_count;
    const int previous_frontier_end = frontier_end;
    const int previous_depth = result.iterations_used;

    if (use_cooperative_controller) {
      launch_cooperative_controller<EdgeOffset, UseGeneration>(outgoing,
                                                               out_rowptr,
                                                               scratch,
                                                               pred_edge,
                                                               query_epoch,
                                                               target_count,
                                                               max_depth,
                                                               stream);
      const std::array<int, kStatusCount> status =
          copy_status_to_host(scratch, stream);
      queue_tail = status[kStatusQueueTail];
      found_count = status[kStatusFoundCount];
      frontier_begin = status[kStatusFrontierBegin];
      frontier_end = status[kStatusFrontierEnd];
      result.iterations_used = status[kStatusCompletedDepth];
      current_count = frontier_end - frontier_begin;
      const int expected_active =
          current_count > 0 && found_count < target_count &&
          result.iterations_used < max_depth;
      const int remaining_depth = max_depth - previous_depth;
      const int levels_budgeted =
          std::min(kCooperativeLevelsPerLaunch, remaining_depth);
      if (queue_tail < previous_queue_tail || queue_tail > n_int ||
          frontier_begin < previous_frontier_end ||
          frontier_end < frontier_begin || frontier_end != queue_tail ||
          found_count < previous_found_count || found_count > target_count ||
          result.iterations_used <= previous_depth ||
          result.iterations_used > previous_depth + levels_budgeted ||
          status[kStatusActive] != expected_active) {
        std::ostringstream message;
        message << "unit BFS cooperative frontier state is inconsistent"
                << " (queue_tail=" << queue_tail
                << ", previous_queue_tail=" << previous_queue_tail
                << ", frontier_begin=" << frontier_begin
                << ", frontier_end=" << frontier_end
                << ", previous_frontier_end=" << previous_frontier_end
                << ", found_count=" << found_count
                << ", previous_found_count=" << previous_found_count
                << ", completed_depth=" << result.iterations_used
                << ", previous_depth=" << previous_depth
                << ", levels_budgeted=" << levels_budgeted
                << ", active=" << status[kStatusActive]
                << ", expected_active=" << expected_active
                << ", rows=" << n_int
                << ", sources=" << source_count
                << ", targets=" << target_count
                << ", max_depth=" << max_depth << ')';
        throw std::runtime_error(message.str());
      }
      continue;
    }

    if (!use_batched_device_controller) {
      // Parallel PathFinder workers use explicit nonblocking streams.  Keep
      // their frontier bounds and depth on the host, as the pre-batching
      // implementation did.  On gfx1151, an expansion's queue-tail updates
      // have repeatedly become host-visible without the immediately following
      // controller-advance kernel, even though both were submitted to the same
      // stream.  Removing that dependent kernel also removes the failure mode.
      expand_frontier_host_controlled_kernel<EdgeOffset, UseGeneration>
          <<<grid_for_frontier(current_count), kBlockSize, 0, stream>>>(
              frontier_begin,
              frontier_end,
              previous_depth + 1,
              out_rowptr,
              outgoing.colind.get(),
              query_epoch,
              scratch.level.get(),
              scratch.generation_level.get(),
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
        expand_frontier_kernel<EdgeOffset, UseGeneration>
            <<<launch_blocks, kBlockSize, 0, stream>>>(
                out_rowptr,
                outgoing.colind.get(),
                query_epoch,
                scratch.level.get(),
                scratch.generation_level.get(),
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
    extract_target_paths_to_result<EdgeOffset, UseGeneration>(
        result,
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
    if (scratch.visitation_mode ==
        UnitBfsCsrVisitationMode::kGenerationStamped) {
      return run_unit_bfs_with_offsets<CompactOffset, true>(
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
    return run_unit_bfs_with_offsets<CompactOffset, false>(
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
  if (scratch.visitation_mode ==
      UnitBfsCsrVisitationMode::kGenerationStamped) {
    return run_unit_bfs_with_offsets<Offset, true>(outgoing,
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
  return run_unit_bfs_with_offsets<Offset, false>(outgoing,
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

  static UnitBfsCsrWorkspaceOptions validate_options(
      UnitBfsCsrWorkspaceOptions options) {
    switch (options.extraction_mode) {
      case UnitBfsCsrExtractionMode::kHostOffsets:
      case UnitBfsCsrExtractionMode::kDeviceOffsets:
        break;
      default:
        throw std::invalid_argument("unknown unit BFS extraction mode");
    }
    switch (options.visitation_mode) {
      case UnitBfsCsrVisitationMode::kSparseReset:
      case UnitBfsCsrVisitationMode::kGenerationStamped:
        break;
      default:
        throw std::invalid_argument("unknown unit BFS visitation mode");
    }
    sssp_capacity::validate_reservation(options.capacity_hints);
    return options;
  }

  Impl(std::shared_ptr<const UnitBfsCsrGraph> graph_,
       hipStream_t stream,
       UnitBfsCsrWorkspaceOptions options)
      : graph(require_graph(graph_)),
        scratch(graph->outgoing.rows,
                graph->outgoing.uses_32_bit_offsets,
                validate_options(options)),
        stream(stream) {
    if (scratch.visitation_mode ==
        UnitBfsCsrVisitationMode::kGenerationStamped) {
      scratch.generation_cooperative_launch_blocks =
          graph->outgoing.uses_32_bit_offsets
              ? unit_bfs_detail::cooperative_controller_blocks<
                    unit_bfs_detail::CompactOffset, true>(graph->outgoing.rows)
              : unit_bfs_detail::cooperative_controller_blocks<
                    minplus_sparse::Offset, true>(graph->outgoing.rows);
    }
  }

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
          adjacency,
          stream,
          UnitBfsCsrOffsetMode::kAuto,
          UnitBfsCsrWorkspaceOptions{}) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream,
                                         UnitBfsCsrOffsetMode offset_mode)
    : UnitBfsCsrWorkspace(
          adjacency, stream, offset_mode, UnitBfsCsrWorkspaceOptions{}) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    UnitBfsCsrWorkspaceOptions options)
    : UnitBfsCsrWorkspace(
          adjacency, stream, UnitBfsCsrOffsetMode::kAuto, options) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    UnitBfsCsrOffsetMode offset_mode,
    UnitBfsCsrWorkspaceOptions options)
    : UnitBfsCsrWorkspace(
          std::make_shared<UnitBfsCsrGraph>(adjacency, stream, offset_mode),
          stream,
          options) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    std::shared_ptr<const UnitBfsCsrGraph> adjacency,
    hipStream_t stream)
    : UnitBfsCsrWorkspace(
          std::move(adjacency), stream, UnitBfsCsrWorkspaceOptions{}) {}

UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(
    std::shared_ptr<const UnitBfsCsrGraph> adjacency,
    hipStream_t stream,
    UnitBfsCsrWorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(adjacency), stream, options)) {}

UnitBfsCsrWorkspace::~UnitBfsCsrWorkspace() = default;
UnitBfsCsrWorkspace::UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept = default;
UnitBfsCsrWorkspace& UnitBfsCsrWorkspace::operator=(
    UnitBfsCsrWorkspace&&) noexcept = default;

UnitBfsCsrAllocationState UnitBfsCsrWorkspace::allocation_state() const noexcept {
  UnitBfsCsrAllocationState state;
  if (!impl_) return state;
  const auto& scratch = impl_->scratch;
  state.source_capacity = scratch.sources.size();
  state.target_capacity = scratch.targets.size();
  state.target_metadata_capacity =
      std::min(scratch.target_metadata.size(),
               scratch.host_target_metadata.size());
  state.target_offset_capacity =
      std::min({scratch.target_node_offsets.size(),
                scratch.host_target_node_offsets.size(),
                scratch.target_edge_offsets.size(),
                scratch.host_target_edge_offsets.size()});
  state.compact_path_node_capacity =
      std::min(scratch.compact_path_nodes.size(),
               scratch.host_compact_path_nodes.size());
  state.compact_path_edge_capacity =
      std::min(scratch.compact_path_edges.size(),
               scratch.host_compact_path_edges.size());
  return state;
}

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
