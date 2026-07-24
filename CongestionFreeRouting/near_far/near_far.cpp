#include "near_far.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace near_far_detail {

using minplus_sparse::Index;
using minplus_sparse::Offset;

constexpr int kBlockSize = 256;
constexpr int kMaxGridX = 65535;
constexpr int kCurrentQueue = 0;
constexpr int kNextQueue = 1;
constexpr int kNearFarQueue = 2;
constexpr int kFarQueue = 3;
constexpr int kScratchQueue = 4;
constexpr int kQueueCount = 5;
constexpr int kDefaultSubpartitions = 16;
constexpr int kUrgentWavesPerComputeUnit = 32;
constexpr int kTargetPathValid = 1;

enum DeviceError : int {
  kNoDeviceError = 0,
  kDeferredQueueOverflow = 1,
  kVersionOverflow = 2,
  kInvalidVertexState = 3,
};

inline void hip_check(hipError_t status,
                      const char* expression,
                      const char* file,
                      int line) {
  if (status != hipSuccess) {
    std::ostringstream message;
    message << "HIP error at " << file << ':' << line << " for "
            << expression << ": " << hipGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

#define NEAR_FAR_HIP_CHECK(expr) \
  ::near_far_detail::hip_check((expr), #expr, __FILE__, __LINE__)

inline int current_hip_device() {
  int device = 0;
  NEAR_FAR_HIP_CHECK(hipGetDevice(&device));
  return device;
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
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Near-Far device allocation size overflow");
    }
    T* candidate = nullptr;
    NEAR_FAR_HIP_CHECK(
        hipMalloc(reinterpret_cast<void**>(&candidate), count * sizeof(T)));
    pointer_ = candidate;
    count_ = count;
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)hipFree(pointer_);
      pointer_ = nullptr;
    }
    count_ = 0;
  }

  void move_from(DeviceBuffer&& other) noexcept {
    pointer_ = other.pointer_;
    count_ = other.count_;
    other.pointer_ = nullptr;
    other.count_ = 0;
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

inline std::size_t checked_size(Offset value, const char* name) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be nonnegative");
  }
  return static_cast<std::size_t>(value);
}

inline int grid_for_items(std::size_t count) {
  if (count == 0) {
    return 1;
  }
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(
      std::min<std::size_t>(kMaxGridX, std::max<std::size_t>(1, blocks)));
}

void validate_delta(float delta) {
  if (!(delta > 0.0f) || !std::isfinite(delta)) {
    throw std::invalid_argument(
        "Near-Far delta must be a finite positive float");
  }
}

void validate_host_csr(const HostCsrF32& graph) {
  const std::size_t rows = checked_size(graph.rows, "rows");
  const std::size_t nnz = checked_size(graph.nnz, "nnz");
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::invalid_argument(
        "Near-Far requires a nonempty square CSR graph");
  }
  if (graph.rows >
      static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Near-Far queue vertices are stored as int");
  }
  if (graph.rowptr.size() != rows + 1 ||
      graph.colind.size() != nnz ||
      graph.values.size() != nnz) {
    throw std::invalid_argument(
        "Near-Far CSR array sizes do not match rows and nnz");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument(
        "Near-Far CSR rowptr must start at zero and end at nnz");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    if (graph.rowptr[row] < 0 ||
        graph.rowptr[row + 1] < graph.rowptr[row] ||
        graph.rowptr[row + 1] > graph.nnz) {
      throw std::invalid_argument(
          "Near-Far CSR rowptr must be monotone and in range");
    }
  }
  for (std::size_t edge = 0; edge < nnz; ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::invalid_argument(
          "Near-Far CSR contains an out-of-range destination");
    }
    if (!std::isfinite(graph.values[edge]) ||
        graph.values[edge] < 0.0f) {
      throw std::invalid_argument(
          "Near-Far requires finite nonnegative edge weights");
    }
  }
}

void validate_sources(Offset rows, const std::vector<int>& sources) {
  if (sources.empty()) {
    throw std::invalid_argument(
        "Near-Far requires at least one source vertex");
  }
  if (sources.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Near-Far source count must fit in int");
  }
  for (const int source : sources) {
    if (source < 0 || static_cast<Offset>(source) >= rows) {
      throw std::out_of_range(
          "Near-Far source vertex is outside the CSR graph");
    }
  }
}

void validate_targets(Offset rows, const std::vector<int>& targets) {
  if (targets.empty()) {
    throw std::invalid_argument(
        "Near-Far vector-target run requires at least one target");
  }
  if (targets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Near-Far target count must fit in int");
  }
  for (const int target : targets) {
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      throw std::out_of_range(
          "Near-Far target vertex is outside the CSR graph");
    }
  }
}

struct DeviceCsrOwner {
  DeviceBuffer<Offset> rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  minplus_sparse::DeviceCsrF32 view{};

  DeviceCsrOwner(const HostCsrF32& host, hipStream_t stream)
      : rowptr(static_cast<std::size_t>(host.rows) + 1),
        colind(static_cast<std::size_t>(host.nnz)),
        values(static_cast<std::size_t>(host.nnz)) {
    view.rows = host.rows;
    view.cols = host.cols;
    view.nnz = host.nnz;
    view.rowptr = rowptr.get();
    view.colind = colind.get();
    view.values = values.get();

    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        rowptr.get(),
        host.rowptr.data(),
        host.rowptr.size() * sizeof(Offset),
        hipMemcpyHostToDevice,
        stream));
    if (host.nnz != 0) {
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          colind.get(),
          host.colind.data(),
          host.colind.size() * sizeof(Index),
          hipMemcpyHostToDevice,
          stream));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          values.get(),
          host.values.data(),
          host.values.size() * sizeof(float),
          hipMemcpyHostToDevice,
          stream));
    }
    // Shared workspaces may subsequently use different streams.
    NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
  }
};

struct DeviceQueueView {
  int* vertices = nullptr;
  int* shard_counts = nullptr;
  int* total_claims = nullptr;
  int* membership = nullptr;
  int* error = nullptr;
  int rows = 0;
  int shards = 1;
  int shard_capacity = 0;
  int logical_capacity = 0;
};

struct QueueStorage {
  DeviceBuffer<int> vertices;
  DeviceBuffer<int> shard_counts;
  DeviceBuffer<int> total_claims;
  DeviceBuffer<int> membership;
  int rows = 0;
  int shards = 1;
  int shard_capacity = 0;
  int logical_capacity = 0;

  static int capacity_per_shard(int capacity, int shard_count) {
    return capacity / shard_count +
           (capacity % shard_count == 0 ? 0 : 1);
  }

  QueueStorage() = default;
  QueueStorage(int row_count, int shard_count, int capacity)
      : vertices(static_cast<std::size_t>(shard_count) *
                 static_cast<std::size_t>(
                     capacity_per_shard(capacity, shard_count))),
        shard_counts(static_cast<std::size_t>(shard_count)),
        total_claims(1),
        membership(static_cast<std::size_t>(row_count)),
        rows(row_count),
        shards(shard_count),
        shard_capacity(capacity_per_shard(capacity, shard_count)),
        logical_capacity(capacity) {}

  DeviceQueueView view(int* error) {
    return {vertices.get(),
            shard_counts.get(),
            total_claims.get(),
            membership.get(),
            error,
            rows,
            shards,
            shard_capacity,
            logical_capacity};
  }
};

struct Scratch {
  Offset rows = 0;
  int shards = 1;
  int urgent_capacity = 1;

  DeviceBuffer<float> dist;
  DeviceBuffer<int> owner_source;
  DeviceBuffer<std::uint32_t> hops;
  DeviceBuffer<int> pred_node;
  DeviceBuffer<Offset> pred_edge;
  DeviceBuffer<std::uint32_t> version;
  DeviceBuffer<std::uint32_t> processed_version;
  DeviceBuffer<unsigned int> locks;

  std::array<QueueStorage, kQueueCount> queues;
  DeviceBuffer<int> sources;
  DeviceBuffer<int> targets;
  DeviceBuffer<float> target_distances;
  DeviceBuffer<int> target_sources;
  DeviceBuffer<int> target_path_lengths;
  DeviceBuffer<int> target_path_status;
  DeviceBuffer<int> target_node_offsets;
  DeviceBuffer<int> target_edge_offsets;
  DeviceBuffer<int> compact_path_nodes;
  DeviceBuffer<Offset> compact_path_edges;

  DeviceBuffer<int> error;
  DeviceBuffer<int> changed;
  DeviceBuffer<unsigned int> min_pending_bits;
  DeviceBuffer<int> unreached_targets;
  DeviceBuffer<unsigned int> max_target_bits;

  Scratch(Offset row_count, int shard_count, int urgent_count)
      : rows(row_count),
        shards(shard_count),
        urgent_capacity(urgent_count),
        dist(static_cast<std::size_t>(row_count)),
        owner_source(static_cast<std::size_t>(row_count)),
        hops(static_cast<std::size_t>(row_count)),
        pred_node(static_cast<std::size_t>(row_count)),
        pred_edge(static_cast<std::size_t>(row_count)),
        version(static_cast<std::size_t>(row_count)),
        processed_version(static_cast<std::size_t>(row_count)),
        locks(static_cast<std::size_t>(row_count)),
        error(1),
        changed(1),
        min_pending_bits(1),
        unreached_targets(1),
        max_target_bits(1) {
    const int rows_as_int = static_cast<int>(row_count);
    queues[kCurrentQueue] =
        QueueStorage(rows_as_int, shards, urgent_capacity);
    queues[kNextQueue] =
        QueueStorage(rows_as_int, shards, urgent_capacity);
    queues[kNearFarQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int);
    queues[kFarQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int);
    queues[kScratchQueue] =
        QueueStorage(rows_as_int, shards, rows_as_int);
  }

  void ensure_source_capacity(std::size_t count) {
    if (sources.size() < count) {
      sources.reset(grown_capacity(sources.size(), count));
    }
  }

  void ensure_target_capacity(std::size_t count) {
    const std::size_t capacity = grown_capacity(targets.size(), count);
    if (targets.size() < count) targets.reset(capacity);
    if (target_distances.size() < count) target_distances.reset(capacity);
    if (target_sources.size() < count) target_sources.reset(capacity);
    if (target_path_lengths.size() < count) {
      target_path_lengths.reset(capacity);
    }
    if (target_path_status.size() < count) {
      target_path_status.reset(capacity);
    }
    if (target_node_offsets.size() < count + 1) {
      target_node_offsets.reset(capacity + 1);
    }
    if (target_edge_offsets.size() < count + 1) {
      target_edge_offsets.reset(capacity + 1);
    }
  }

  void ensure_compact_path_capacity(std::size_t nodes, std::size_t edges) {
    if (compact_path_nodes.size() < nodes) {
      compact_path_nodes.reset(
          grown_capacity(compact_path_nodes.size(), nodes));
    }
    if (compact_path_edges.size() < edges) {
      compact_path_edges.reset(
          grown_capacity(compact_path_edges.size(), edges));
    }
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

__device__ inline void set_device_error(int* error, DeviceError value) {
  atomicCAS(error, kNoDeviceError, static_cast<int>(value));
}

__device__ inline bool try_lock_vertex(unsigned int* lock) {
  return atomicCAS(lock, 0U, 1U) == 0U;
}

__device__ inline void unlock_vertex(unsigned int* lock) {
  __threadfence();
  atomicExch(lock, 0U);
}

__device__ inline bool bump_version(std::uint32_t* version, int* error) {
  std::uint32_t old = atomicAdd(version, 0U);
  while (old != std::numeric_limits<std::uint32_t>::max()) {
    const std::uint32_t observed = atomicCAS(version, old, old + 1U);
    if (observed == old) return true;
    old = observed;
  }
  set_device_error(error, kVersionOverflow);
  return false;
}

__device__ inline bool finite_float(float value) {
  return (__float_as_uint(value) & 0x7f800000U) != 0x7f800000U;
}

__device__ inline bool enqueue_unique(DeviceQueueView queue,
                                      int vertex,
                                      bool overflow_is_error) {
  if (atomicCAS(queue.membership + vertex, 0, 1) != 0) {
    return true;
  }

  const int total = atomicAdd(queue.total_claims, 1);
  if (total >= queue.logical_capacity) {
    atomicSub(queue.total_claims, 1);
    atomicExch(queue.membership + vertex, 0);
    if (overflow_is_error) {
      set_device_error(queue.error, kDeferredQueueOverflow);
    }
    return false;
  }

  const int shard = vertex % queue.shards;
  const int position = atomicAdd(queue.shard_counts + shard, 1);
  if (position >= queue.shard_capacity) {
    atomicSub(queue.shard_counts + shard, 1);
    atomicSub(queue.total_claims, 1);
    atomicExch(queue.membership + vertex, 0);
    if (overflow_is_error) {
      set_device_error(queue.error, kDeferredQueueOverflow);
    }
    return false;
  }
  queue.vertices[shard * queue.shard_capacity + position] = vertex;
  return true;
}

__device__ inline void enqueue_by_distance(int vertex,
                                           float distance,
                                           float near_far_threshold,
                                           DeviceQueueView urgent,
                                           DeviceQueueView near_far,
                                           DeviceQueueView far) {
  if (distance < near_far_threshold) {
    if (!enqueue_unique(urgent, vertex, false)) {
      (void)enqueue_unique(near_far, vertex, true);
    }
  } else {
    (void)enqueue_unique(far, vertex, true);
  }
}

__global__ void initialize_state_kernel(
    Offset rows,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    int* pred_node,
    Offset* pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* locks) {
  for (Offset vertex =
           static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       vertex < rows;
       vertex += static_cast<Offset>(gridDim.x) * blockDim.x) {
    const std::size_t index = static_cast<std::size_t>(vertex);
    dist[index] = INFINITY;
    owner_source[index] = std::numeric_limits<int>::max();
    hops[index] = std::numeric_limits<std::uint32_t>::max();
    pred_node[index] = -1;
    pred_edge[index] = -1;
    version[index] = 0;
    processed_version[index] = 0;
    locks[index] = 0;
  }
}

__global__ void initialize_sources_kernel(
    const int* sources,
    int source_count,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    int* pred_node,
    Offset* pred_edge,
    std::uint32_t* version,
    unsigned int* locks,
    DeviceQueueView current,
    DeviceQueueView near_far) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < source_count;
       index += gridDim.x * blockDim.x) {
    const int source = sources[index];
    if (!try_lock_vertex(locks + source)) {
      continue;
    }
    if (dist[source] != 0.0f || owner_source[source] > source ||
        hops[source] != 0) {
      dist[source] = 0.0f;
      owner_source[source] = source;
      hops[source] = 0;
      pred_node[source] = source;
      pred_edge[source] = -1;
      version[source] = 1;
    }
    unlock_vertex(locks + source);
    if (!enqueue_unique(current, source, false)) {
      (void)enqueue_unique(near_far, source, true);
    }
  }
}

__device__ inline bool candidate_is_better(
    float candidate_distance,
    int candidate_source,
    std::uint32_t candidate_hops,
    Offset candidate_edge,
    float current_distance,
    int current_source,
    std::uint32_t current_hops,
    Offset current_edge) {
  if (candidate_distance < current_distance) return true;
  if (candidate_distance > current_distance) return false;
  const bool candidate_is_identity = candidate_hops == 0;
  const bool current_is_identity = current_hops == 0;
  if (candidate_is_identity != current_is_identity) {
    return candidate_is_identity;
  }
  if (candidate_source < current_source) return true;
  if (candidate_source > current_source) return false;
  if (candidate_hops < current_hops) return true;
  if (candidate_hops > current_hops) return false;
  return current_edge < 0 || candidate_edge < current_edge;
}

__global__ void expand_queue_shard_kernel(
    const int* queue_vertices,
    int queue_count,
    int* queue_membership,
    const Offset* rowptr,
    const Index* colind,
    const float* values,
    const float* vertex_costs,
    bool has_vertex_costs,
    float near_far_threshold,
    float* dist,
    int* owner_source,
    std::uint32_t* hops,
    int* pred_node,
    Offset* pred_edge,
    std::uint32_t* version,
    std::uint32_t* processed_version,
    unsigned int* locks,
    DeviceQueueView next,
    DeviceQueueView near_far,
    DeviceQueueView far,
    int* changed,
    int* error) {
  for (int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
       queue_index < queue_count;
       queue_index += gridDim.x * blockDim.x) {
    const int u = queue_vertices[queue_index];
    atomicExch(queue_membership + u, 0);

    if (!try_lock_vertex(locks + u)) {
      (void)enqueue_unique(near_far, u, true);
      continue;
    }
    const std::uint32_t current_version = atomicAdd(version + u, 0U);
    if (current_version == 0 ||
        current_version <= processed_version[u]) {
      unlock_vertex(locks + u);
      continue;
    }
    processed_version[u] = current_version;
    const float source_distance = dist[u];
    const int source_owner = owner_source[u];
    const std::uint32_t source_hops = hops[u];
    unlock_vertex(locks + u);

    if (!finite_float(source_distance) ||
        source_owner == std::numeric_limits<int>::max() ||
        source_hops == std::numeric_limits<std::uint32_t>::max()) {
      set_device_error(error, kInvalidVertexState);
      continue;
    }
    if (source_hops == std::numeric_limits<std::uint32_t>::max() - 1U) {
      set_device_error(error, kInvalidVertexState);
      continue;
    }

    bool retry_source = false;
    for (Offset edge = rowptr[u]; edge < rowptr[u + 1]; ++edge) {
      const int v = colind[edge];
      const float destination_cost =
          has_vertex_costs ? vertex_costs[v] : 1.0f;
      const float candidate_distance =
          source_distance + values[edge] * destination_cost;
      if (!finite_float(candidate_distance)) {
        continue;
      }
      const std::uint32_t candidate_hops = source_hops + 1U;

      bool updated = false;
      if (!try_lock_vertex(locks + v)) {
        retry_source = true;
        continue;
      }
      if (candidate_is_better(candidate_distance,
                              source_owner,
                              candidate_hops,
                              edge,
                              dist[v],
                              owner_source[v],
                              hops[v],
                              pred_edge[v])) {
        if (bump_version(version + v, error)) {
          dist[v] = candidate_distance;
          owner_source[v] = source_owner;
          hops[v] = candidate_hops;
          pred_node[v] = u;
          pred_edge[v] = edge;
          updated = true;
        }
      }
      unlock_vertex(locks + v);

      if (updated) {
        atomicExch(changed, 1);
        enqueue_by_distance(
            v, candidate_distance, near_far_threshold, next, near_far, far);
      }
    }
    if (retry_source && bump_version(version + u, error)) {
      (void)enqueue_unique(near_far, u, true);
    }
  }
}

__global__ void reclassify_queue_shard_kernel(
    const int* input_vertices,
    int input_count,
    int* input_membership,
    float near_far_threshold,
    const float* dist,
    const std::uint32_t* version,
    const std::uint32_t* processed_version,
    DeviceQueueView urgent,
    DeviceQueueView near_far,
    DeviceQueueView far) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < input_count;
       index += gridDim.x * blockDim.x) {
    const int vertex = input_vertices[index];
    atomicExch(input_membership + vertex, 0);
    if (version[vertex] == 0 ||
        version[vertex] <= processed_version[vertex]) {
      continue;
    }
    enqueue_by_distance(vertex,
                        dist[vertex],
                        near_far_threshold,
                        urgent,
                        near_far,
                        far);
  }
}

__global__ void clear_queue_membership_shard_kernel(
    const int* vertices,
    int count,
    int* membership) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += gridDim.x * blockDim.x) {
    atomicExch(membership + vertices[index], 0);
  }
}

__global__ void queue_min_kernel(
    const int* vertices,
    int count,
    const float* dist,
    const std::uint32_t* version,
    const std::uint32_t* processed_version,
    unsigned int* min_bits) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += gridDim.x * blockDim.x) {
    const int vertex = vertices[index];
    if (version[vertex] > processed_version[vertex]) {
      atomicMin(min_bits, __float_as_uint(dist[vertex]));
    }
  }
}

__global__ void pending_min_kernel(
    Offset rows,
    const float* dist,
    const std::uint32_t* version,
    const std::uint32_t* processed_version,
    unsigned int* min_bits) {
  for (Offset vertex =
           static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
       vertex < rows;
       vertex += static_cast<Offset>(gridDim.x) * blockDim.x) {
    if (version[vertex] > processed_version[vertex]) {
      atomicMin(min_bits, __float_as_uint(dist[vertex]));
    }
  }
}

__global__ void target_bounds_kernel(const int* targets,
                                     int target_count,
                                     const float* dist,
                                     int* unreached,
                                     unsigned int* max_bits) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    const float value = dist[targets[index]];
    if (!finite_float(value)) {
      atomicAdd(unreached, 1);
    } else {
      atomicMax(max_bits, __float_as_uint(value));
    }
  }
}

__global__ void measure_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    const Offset* rowptr,
    const Index* colind,
    const float* dist,
    const int* owner_source,
    const std::uint32_t* hops,
    const int* pred_node,
    const Offset* pred_edge,
    bool paths_certified,
    float* target_distances,
    int* target_sources,
    int* path_lengths,
    int* path_status) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    const int target = targets[index];
    const float target_distance = dist[target];
    const std::uint32_t target_hops = hops[target];
    target_distances[index] = INFINITY;
    target_sources[index] = -1;
    path_lengths[index] = 0;
    path_status[index] = 0;

    if (!finite_float(target_distance) ||
        target_hops == std::numeric_limits<std::uint32_t>::max() ||
        (!paths_certified && target_hops != 0) ||
        target_hops >= static_cast<std::uint32_t>(rows)) {
      continue;
    }

    int current = target;
    std::uint32_t current_hops = target_hops;
    const int source = owner_source[target];
    bool valid = source >= 0 && static_cast<Offset>(source) < rows;
    while (valid && current_hops > 0) {
      const int predecessor = pred_node[current];
      const Offset edge = pred_edge[current];
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows ||
          edge < rowptr[predecessor] || edge >= rowptr[predecessor + 1] ||
          colind[edge] != current ||
          owner_source[predecessor] != source ||
          hops[predecessor] + 1U != current_hops) {
        valid = false;
        break;
      }
      current = predecessor;
      --current_hops;
    }
    valid = valid && current == source && hops[current] == 0 &&
            pred_node[current] == current;
    if (!valid) {
      continue;
    }

    target_distances[index] = target_distance;
    target_sources[index] = source;
    path_lengths[index] = static_cast<int>(target_hops) + 1;
    path_status[index] = kTargetPathValid;
  }
}

__global__ void fill_target_paths_kernel(
    const int* targets,
    int target_count,
    Offset rows,
    const Offset* rowptr,
    const Index* colind,
    const int* owner_source,
    const std::uint32_t* hops,
    const int* pred_node,
    const Offset* pred_edge,
    const int* path_lengths,
    int* path_status,
    const int* node_offsets,
    const int* edge_offsets,
    int* path_nodes,
    Offset* path_edges) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < target_count;
       index += gridDim.x * blockDim.x) {
    if (path_status[index] != kTargetPathValid ||
        path_lengths[index] <= 0) {
      continue;
    }

    int current = targets[index];
    std::uint32_t current_hops = hops[current];
    const int source = owner_source[current];
    const int node_begin = node_offsets[index];
    const int edge_begin = edge_offsets[index];
    bool valid =
        path_lengths[index] == static_cast<int>(current_hops) + 1;

    while (valid) {
      path_nodes[node_begin + static_cast<int>(current_hops)] = current;
      if (current_hops == 0) {
        valid = current == source && pred_node[current] == current;
        break;
      }
      const int predecessor = pred_node[current];
      const Offset edge = pred_edge[current];
      if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows ||
          edge < rowptr[predecessor] || edge >= rowptr[predecessor + 1] ||
          colind[edge] != current ||
          owner_source[predecessor] != source ||
          hops[predecessor] + 1U != current_hops) {
        valid = false;
        break;
      }
      path_edges[edge_begin + static_cast<int>(current_hops) - 1] = edge;
      current = predecessor;
      --current_hops;
    }
    if (!valid) {
      path_status[index] = 0;
    }
  }
}

template <typename T>
T copy_scalar_to_host(const T* device_value, hipStream_t stream) {
  T host_value{};
  NEAR_FAR_HIP_CHECK(hipMemcpyAsync(&host_value,
                                     device_value,
                                     sizeof(T),
                                     hipMemcpyDeviceToHost,
                                     stream));
  NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
  return host_value;
}

std::vector<int> copy_queue_counts(const QueueStorage& queue,
                                   hipStream_t stream) {
  std::vector<int> counts(static_cast<std::size_t>(queue.shards));
  NEAR_FAR_HIP_CHECK(hipMemcpyAsync(counts.data(),
                                     queue.shard_counts.get(),
                                     counts.size() * sizeof(int),
                                     hipMemcpyDeviceToHost,
                                     stream));
  NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
  for (const int count : counts) {
    if (count < 0 || count > queue.shard_capacity) {
      throw std::runtime_error(
          "Near-Far queue shard count is outside its allocation");
    }
  }
  return counts;
}

std::size_t queue_size(const QueueStorage& queue, hipStream_t stream) {
  const std::vector<int> counts = copy_queue_counts(queue, stream);
  std::size_t total = 0;
  for (const int count : counts) {
    total += static_cast<std::size_t>(count);
  }
  return total;
}

void reset_queue_counts(QueueStorage& queue, hipStream_t stream) {
  NEAR_FAR_HIP_CHECK(hipMemsetAsync(queue.shard_counts.get(),
                                     0,
                                     static_cast<std::size_t>(queue.shards) *
                                         sizeof(int),
                                     stream));
  NEAR_FAR_HIP_CHECK(
      hipMemsetAsync(queue.total_claims.get(), 0, sizeof(int), stream));
}

void clear_queue(QueueStorage& queue, hipStream_t stream) {
  const std::vector<int> counts = copy_queue_counts(queue, stream);
  for (int shard = 0; shard < queue.shards; ++shard) {
    const int count = counts[static_cast<std::size_t>(shard)];
    if (count == 0) continue;
    clear_queue_membership_shard_kernel
        <<<grid_for_items(static_cast<std::size_t>(count)),
           kBlockSize,
           0,
           stream>>>(
            queue.vertices.get() + shard * queue.shard_capacity,
            count,
            queue.membership.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
  }
  reset_queue_counts(queue, stream);
}

int choose_queue_shards(Offset rows,
                        std::size_t requested,
                        const hipDeviceProp_t& properties) {
  const std::size_t max_shards =
      std::min<std::size_t>(kDefaultSubpartitions,
                            static_cast<std::size_t>(rows));
  if (requested != 0) {
    if (requested > max_shards) {
      throw std::invalid_argument(
          "Near-Far queue shard count exceeds min(rows, 16)");
    }
    return static_cast<int>(requested);
  }
  const int compute_units = std::max(1, properties.multiProcessorCount);
  return static_cast<int>(
      std::max<std::size_t>(
          1,
          std::min<std::size_t>(max_shards,
                                static_cast<std::size_t>(compute_units))));
}

int choose_urgent_capacity(Offset rows,
                           Offset nnz,
                           std::size_t requested,
                           const hipDeviceProp_t& properties) {
  if (requested != 0) {
    if (requested > static_cast<std::size_t>(rows)) {
      throw std::invalid_argument(
          "Near-Far urgent queue capacity exceeds graph rows");
    }
    return static_cast<int>(requested);
  }
  const double average_degree =
      rows == 0 ? 0.0 : static_cast<double>(nnz) / rows;
  const double useful_rows =
      average_degree > 0.0
          ? (static_cast<double>(
                 std::max(1, properties.multiProcessorCount)) *
             static_cast<double>(std::max(1, properties.warpSize)) *
             kUrgentWavesPerComputeUnit) /
                average_degree
          : static_cast<double>(rows);
  return static_cast<int>(
      std::max<double>(1.0,
                       std::min<double>(static_cast<double>(rows),
                                        std::ceil(useful_rows))));
}

float next_near_far_threshold(float current,
                              float delta,
                              float minimum_far_distance) {
  const double current_step =
      static_cast<double>(current) + static_cast<double>(delta);
  const double containing_step =
      (std::floor(static_cast<double>(minimum_far_distance) /
                  static_cast<double>(delta)) +
       1.0) *
      static_cast<double>(delta);
  double candidate = std::max(current_step, containing_step);
  if (!std::isfinite(candidate) ||
      !(candidate > static_cast<double>(minimum_far_distance))) {
    return std::nextafter(minimum_far_distance,
                          std::numeric_limits<float>::infinity());
  }
  const float narrowed = static_cast<float>(candidate);
  if (!(narrowed > minimum_far_distance)) {
    return std::nextafter(minimum_far_distance,
                          std::numeric_limits<float>::infinity());
  }
  return narrowed;
}

const char* device_error_message(int error) {
  switch (error) {
    case kNoDeviceError:
      return "no error";
    case kDeferredQueueOverflow:
      return "a deferred queue exceeded its checked capacity";
    case kVersionOverflow:
      return "a vertex update generation overflowed";
    case kInvalidVertexState:
      return "an active vertex had invalid state";
    default:
      return "an unknown device error occurred";
  }
}

}  // namespace near_far_detail

struct NearFarCsrGraph::Impl {
  near_far_detail::DeviceCsrOwner adjacency;
  int device = 0;

  Impl(const HostCsrF32& host, hipStream_t stream)
      : adjacency(host, stream),
        device(near_far_detail::current_hip_device()) {}
};

struct NearFarCsrWorkspace::Impl {
  std::shared_ptr<const NearFarCsrGraph::Impl> graph;
  near_far_detail::Scratch scratch;
  near_far_detail::DeviceBuffer<float> vertex_costs;
  bool has_vertex_costs = false;
  hipStream_t stream = nullptr;
  int device = 0;

  static std::shared_ptr<const NearFarCsrGraph::Impl> require_graph(
      const std::shared_ptr<const NearFarCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument(
          "Near-Far shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const NearFarCsrGraph> shared_graph,
       hipStream_t stream_in,
       NearFarCsrWorkspaceOptions options)
      : graph(require_graph(shared_graph)),
        scratch(graph->adjacency.view.rows,
                select_shards(graph->adjacency.view,
                              options.queue_shards),
                select_urgent_capacity(graph->adjacency.view,
                                       options.urgent_queue_capacity)),
        stream(stream_in),
        device(graph->device) {
    require_context(stream);
  }

  static hipDeviceProp_t device_properties() {
    hipDeviceProp_t properties{};
    NEAR_FAR_HIP_CHECK(hipGetDeviceProperties(
        &properties, near_far_detail::current_hip_device()));
    if (properties.warpSize <= 0 ||
        properties.multiProcessorCount <= 0) {
      throw std::runtime_error(
          "HIP device reported invalid Near-Far scheduling properties");
    }
    return properties;
  }

  static int select_shards(const minplus_sparse::DeviceCsrF32& graph,
                           std::size_t requested) {
    return near_far_detail::choose_queue_shards(
        graph.rows, requested, device_properties());
  }

  static int select_urgent_capacity(
      const minplus_sparse::DeviceCsrF32& graph,
      std::size_t requested) {
    return near_far_detail::choose_urgent_capacity(
        graph.rows, graph.nnz, requested, device_properties());
  }

  void require_context(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "NearFarCsrWorkspace is stream-affine; use its construction stream");
    }
    if (near_far_detail::current_hip_device() != device) {
      throw std::invalid_argument(
          "NearFarCsrWorkspace is running on a different HIP device");
    }
  }

  void reset_queues() {
    using namespace near_far_detail;
    for (QueueStorage& queue : scratch.queues) {
      NEAR_FAR_HIP_CHECK(hipMemsetAsync(
          queue.membership.get(),
          0,
          static_cast<std::size_t>(queue.rows) * sizeof(int),
          stream));
      reset_queue_counts(queue, stream);
    }
  }

  void prepare(const std::vector<int>& sources,
               const std::vector<int>* targets) {
    using namespace near_far_detail;
    scratch.ensure_source_capacity(sources.size());
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.sources.get(),
        sources.data(),
        sources.size() * sizeof(int),
        hipMemcpyHostToDevice,
        stream));
    if (targets != nullptr) {
      scratch.ensure_target_capacity(targets->size());
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          scratch.targets.get(),
          targets->data(),
          targets->size() * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
    }

    reset_queues();
    NEAR_FAR_HIP_CHECK(
        hipMemsetAsync(scratch.error.get(), 0, sizeof(int), stream));
    initialize_state_kernel
        <<<grid_for_items(static_cast<std::size_t>(scratch.rows)),
           kBlockSize,
           0,
           stream>>>(
            scratch.rows,
            scratch.dist.get(),
            scratch.owner_source.get(),
            scratch.hops.get(),
            scratch.pred_node.get(),
            scratch.pred_edge.get(),
            scratch.version.get(),
            scratch.processed_version.get(),
            scratch.locks.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());

    DeviceQueueView current =
        scratch.queues[kCurrentQueue].view(scratch.error.get());
    DeviceQueueView near_far =
        scratch.queues[kNearFarQueue].view(scratch.error.get());
    initialize_sources_kernel
        <<<grid_for_items(sources.size()), kBlockSize, 0, stream>>>(
            scratch.sources.get(),
            static_cast<int>(sources.size()),
            scratch.dist.get(),
            scratch.owner_source.get(),
            scratch.hops.get(),
            scratch.pred_node.get(),
            scratch.pred_edge.get(),
            scratch.version.get(),
            scratch.locks.get(),
            current,
            near_far);
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    check_device_error();
  }

  void check_device_error() {
    const int error =
        near_far_detail::copy_scalar_to_host(scratch.error.get(), stream);
    if (error != near_far_detail::kNoDeviceError) {
      throw std::runtime_error(
          std::string("Near-Far device scheduler failed: ") +
          near_far_detail::device_error_message(error));
    }
  }

  bool process_current(float threshold,
                       NearFarCsrProgressCallback callback,
                       void* callback_data,
                       int iteration,
                       int max_iters) {
    using namespace near_far_detail;
    QueueStorage& current = scratch.queues[kCurrentQueue];
    const std::vector<int> counts = copy_queue_counts(current, stream);
    NEAR_FAR_HIP_CHECK(
        hipMemsetAsync(scratch.changed.get(), 0, sizeof(int), stream));

    DeviceQueueView next =
        scratch.queues[kNextQueue].view(scratch.error.get());
    DeviceQueueView near_far =
        scratch.queues[kNearFarQueue].view(scratch.error.get());
    DeviceQueueView far =
        scratch.queues[kFarQueue].view(scratch.error.get());
    for (int shard = 0; shard < current.shards; ++shard) {
      const int count = counts[static_cast<std::size_t>(shard)];
      if (count == 0) continue;
      expand_queue_shard_kernel
          <<<grid_for_items(static_cast<std::size_t>(count)),
             kBlockSize,
             0,
             stream>>>(
              current.vertices.get() + shard * current.shard_capacity,
              count,
              current.membership.get(),
              graph->adjacency.view.rowptr,
              graph->adjacency.view.colind,
              graph->adjacency.view.values,
              has_vertex_costs ? vertex_costs.get() : nullptr,
              has_vertex_costs,
              threshold,
              scratch.dist.get(),
              scratch.owner_source.get(),
              scratch.hops.get(),
              scratch.pred_node.get(),
              scratch.pred_edge.get(),
              scratch.version.get(),
              scratch.processed_version.get(),
              scratch.locks.get(),
              next,
              near_far,
              far,
              scratch.changed.get(),
              scratch.error.get());
      NEAR_FAR_HIP_CHECK(hipGetLastError());
    }

    const bool changed =
        copy_scalar_to_host(scratch.changed.get(), stream) != 0;
    check_device_error();
    reset_queue_counts(current, stream);
    if (callback != nullptr) {
      callback(NearFarCsrProgress{
                   iteration,
                   max_iters,
                   true,
                   changed},
               callback_data);
    }
    return changed;
  }

  void promote_queue(int input_index,
                     int deferred_output_index,
                     float threshold) {
    using namespace near_far_detail;
    QueueStorage& input = scratch.queues[input_index];
    QueueStorage& deferred_output =
        scratch.queues[deferred_output_index];
    QueueStorage& scratch_output = scratch.queues[kScratchQueue];
    if (queue_size(scratch_output, stream) != 0) {
      throw std::logic_error(
          "Near-Far deferred scratch queue was not empty");
    }

    const std::vector<int> counts = copy_queue_counts(input, stream);
    DeviceQueueView urgent =
        scratch.queues[kCurrentQueue].view(scratch.error.get());
    DeviceQueueView near_far =
        (deferred_output_index == kNearFarQueue
             ? scratch_output
             : scratch.queues[kNearFarQueue])
            .view(scratch.error.get());
    DeviceQueueView far =
        (deferred_output_index == kFarQueue
             ? scratch_output
             : scratch.queues[kFarQueue])
            .view(scratch.error.get());

    for (int shard = 0; shard < input.shards; ++shard) {
      const int count = counts[static_cast<std::size_t>(shard)];
      if (count == 0) continue;
      reclassify_queue_shard_kernel
          <<<grid_for_items(static_cast<std::size_t>(count)),
             kBlockSize,
             0,
             stream>>>(
              input.vertices.get() + shard * input.shard_capacity,
              count,
              input.membership.get(),
              threshold,
              scratch.dist.get(),
              scratch.version.get(),
              scratch.processed_version.get(),
              urgent,
              near_far,
              far);
      NEAR_FAR_HIP_CHECK(hipGetLastError());
    }
    check_device_error();
    reset_queue_counts(input, stream);
    std::swap(deferred_output, scratch_output);
    reset_queue_counts(scratch_output, stream);
  }

  float minimum_queue_distance(int queue_index) {
    using namespace near_far_detail;
    QueueStorage& queue = scratch.queues[queue_index];
    const std::vector<int> counts = copy_queue_counts(queue, stream);
    const unsigned int initial = std::numeric_limits<unsigned int>::max();
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.min_pending_bits.get(),
        &initial,
        sizeof(initial),
        hipMemcpyHostToDevice,
        stream));
    for (int shard = 0; shard < queue.shards; ++shard) {
      const int count = counts[static_cast<std::size_t>(shard)];
      if (count == 0) continue;
      queue_min_kernel
          <<<grid_for_items(static_cast<std::size_t>(count)),
             kBlockSize,
             0,
             stream>>>(
              queue.vertices.get() + shard * queue.shard_capacity,
              count,
              scratch.dist.get(),
              scratch.version.get(),
              scratch.processed_version.get(),
              scratch.min_pending_bits.get());
      NEAR_FAR_HIP_CHECK(hipGetLastError());
    }
    const unsigned int bits =
        copy_scalar_to_host(scratch.min_pending_bits.get(), stream);
    if (bits == initial) {
      return std::numeric_limits<float>::infinity();
    }
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits), "float size mismatch");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  bool targets_are_settled(const std::vector<int>& targets) {
    using namespace near_far_detail;
    NEAR_FAR_HIP_CHECK(
        hipMemsetAsync(scratch.unreached_targets.get(), 0, sizeof(int), stream));
    NEAR_FAR_HIP_CHECK(
        hipMemsetAsync(scratch.max_target_bits.get(), 0, sizeof(unsigned int),
                       stream));
    target_bounds_kernel
        <<<grid_for_items(targets.size()), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            static_cast<int>(targets.size()),
            scratch.dist.get(),
            scratch.unreached_targets.get(),
            scratch.max_target_bits.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    const int unreached =
        copy_scalar_to_host(scratch.unreached_targets.get(), stream);
    if (unreached != 0) return false;
    const unsigned int max_bits =
        copy_scalar_to_host(scratch.max_target_bits.get(), stream);

    const unsigned int initial = std::numeric_limits<unsigned int>::max();
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        scratch.min_pending_bits.get(),
        &initial,
        sizeof(initial),
        hipMemcpyHostToDevice,
        stream));
    pending_min_kernel
        <<<grid_for_items(static_cast<std::size_t>(scratch.rows)),
           kBlockSize,
           0,
           stream>>>(
            scratch.rows,
            scratch.dist.get(),
            scratch.version.get(),
            scratch.processed_version.get(),
            scratch.min_pending_bits.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());
    const unsigned int min_bits =
        copy_scalar_to_host(scratch.min_pending_bits.get(), stream);
    return min_bits != initial && min_bits > max_bits;
  }

  NearFarCsrResult extract_targets(const std::vector<int>& targets,
                                   bool paths_certified) {
    using namespace near_far_detail;
    NearFarCsrResult result;
    const int target_count = static_cast<int>(targets.size());
    measure_target_paths_kernel
        <<<grid_for_items(targets.size()), kBlockSize, 0, stream>>>(
            scratch.targets.get(),
            target_count,
            scratch.rows,
            graph->adjacency.view.rowptr,
            graph->adjacency.view.colind,
            scratch.dist.get(),
            scratch.owner_source.get(),
            scratch.hops.get(),
            scratch.pred_node.get(),
            scratch.pred_edge.get(),
            paths_certified,
            scratch.target_distances.get(),
            scratch.target_sources.get(),
            scratch.target_path_lengths.get(),
            scratch.target_path_status.get());
    NEAR_FAR_HIP_CHECK(hipGetLastError());

    result.target_distances.resize(targets.size());
    result.target_sources.resize(targets.size());
    std::vector<int> lengths(targets.size());
    std::vector<int> status(targets.size());
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        result.target_distances.data(),
        scratch.target_distances.get(),
        targets.size() * sizeof(float),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        result.target_sources.data(),
        scratch.target_sources.get(),
        targets.size() * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        lengths.data(),
        scratch.target_path_lengths.get(),
        targets.size() * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
        status.data(),
        scratch.target_path_status.get(),
        targets.size() * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));

    result.target_path_offsets.assign(targets.size() + 1, 0);
    result.target_edge_offsets.assign(targets.size() + 1, 0);
    std::size_t total_nodes = 0;
    std::size_t total_edges = 0;
    bool all_reached = true;
    for (std::size_t index = 0; index < targets.size(); ++index) {
      result.target_path_offsets[index] = static_cast<int>(total_nodes);
      result.target_edge_offsets[index] = static_cast<int>(total_edges);
      if (status[index] != kTargetPathValid || lengths[index] <= 0 ||
          !std::isfinite(result.target_distances[index])) {
        all_reached = false;
        continue;
      }
      total_nodes += static_cast<std::size_t>(lengths[index]);
      total_edges += static_cast<std::size_t>(lengths[index] - 1);
      if (total_nodes >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          total_edges >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Near-Far compact paths exceed int offsets");
      }
    }
    result.target_path_offsets.back() = static_cast<int>(total_nodes);
    result.target_edge_offsets.back() = static_cast<int>(total_edges);

    result.target_path_nodes.resize(total_nodes);
    result.target_path_edges.resize(total_edges);
    if (total_nodes != 0) {
      scratch.ensure_compact_path_capacity(total_nodes, total_edges);
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          scratch.target_node_offsets.get(),
          result.target_path_offsets.data(),
          (targets.size() + 1) * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          scratch.target_edge_offsets.get(),
          result.target_edge_offsets.data(),
          (targets.size() + 1) * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
      fill_target_paths_kernel
          <<<grid_for_items(targets.size()), kBlockSize, 0, stream>>>(
              scratch.targets.get(),
              target_count,
              scratch.rows,
              graph->adjacency.view.rowptr,
              graph->adjacency.view.colind,
              scratch.owner_source.get(),
              scratch.hops.get(),
              scratch.pred_node.get(),
              scratch.pred_edge.get(),
              scratch.target_path_lengths.get(),
              scratch.target_path_status.get(),
              scratch.target_node_offsets.get(),
              scratch.target_edge_offsets.get(),
              scratch.compact_path_nodes.get(),
              scratch.compact_path_edges.get());
      NEAR_FAR_HIP_CHECK(hipGetLastError());
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.target_path_nodes.data(),
          scratch.compact_path_nodes.get(),
          total_nodes * sizeof(int),
          hipMemcpyDeviceToHost,
          stream));
      if (total_edges != 0) {
        NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
            result.target_path_edges.data(),
            scratch.compact_path_edges.get(),
            total_edges * sizeof(Offset),
            hipMemcpyDeviceToHost,
            stream));
      }
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          status.data(),
          scratch.target_path_status.get(),
          targets.size() * sizeof(int),
          hipMemcpyDeviceToHost,
          stream));
      NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
      for (std::size_t index = 0; index < status.size(); ++index) {
        if (lengths[index] > 0 && status[index] != kTargetPathValid) {
          throw std::runtime_error(
              "Near-Far predecessor path failed device validation");
        }
      }
    }
    result.target_reached = all_reached;
    return result;
  }

  NearFarCsrResult run_impl(
      const std::vector<int>& sources,
      const std::vector<int>* targets,
      float delta,
      int max_iters,
      NearFarCsrProgressCallback callback,
      void* callback_data) {
    using namespace near_far_detail;
    prepare(sources, targets);
    float threshold = delta;
    int iterations = 0;
    bool converged = false;
    bool stopped_on_target = false;

    while (true) {
      if (queue_size(scratch.queues[kCurrentQueue], stream) == 0) {
        if (queue_size(scratch.queues[kNextQueue], stream) != 0) {
          std::swap(scratch.queues[kCurrentQueue],
                    scratch.queues[kNextQueue]);
          reset_queue_counts(scratch.queues[kNextQueue], stream);
          continue;
        }

        if (queue_size(scratch.queues[kNearFarQueue], stream) != 0) {
          promote_queue(kNearFarQueue, kNearFarQueue, threshold);
          if (queue_size(scratch.queues[kCurrentQueue], stream) != 0) {
            continue;
          }
        }

        if (queue_size(scratch.queues[kFarQueue], stream) != 0) {
          const float minimum = minimum_queue_distance(kFarQueue);
          if (std::isfinite(minimum)) {
            threshold = next_near_far_threshold(
                threshold, delta, minimum);
            promote_queue(kFarQueue, kFarQueue, threshold);
            if (queue_size(scratch.queues[kCurrentQueue], stream) != 0 ||
                queue_size(scratch.queues[kNearFarQueue], stream) != 0 ||
                queue_size(scratch.queues[kFarQueue], stream) != 0) {
              continue;
            }
          } else {
            clear_queue(scratch.queues[kFarQueue], stream);
          }
        }

        converged = true;
        break;
      }

      if (max_iters >= 0 && iterations >= max_iters) {
        break;
      }
      ++iterations;
      (void)process_current(
          threshold, callback, callback_data, iterations, max_iters);

      if (targets != nullptr && targets_are_settled(*targets)) {
        stopped_on_target = true;
        break;
      }
    }

    NearFarCsrResult result;
    if (targets != nullptr) {
      result = extract_targets(
          *targets, converged || stopped_on_target);
    } else {
      result.dist.resize(static_cast<std::size_t>(scratch.rows));
      result.pred_node.resize(static_cast<std::size_t>(scratch.rows));
      result.pred_edge.resize(static_cast<std::size_t>(scratch.rows));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.dist.data(),
          scratch.dist.get(),
          result.dist.size() * sizeof(float),
          hipMemcpyDeviceToHost,
          stream));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.pred_node.data(),
          scratch.pred_node.get(),
          result.pred_node.size() * sizeof(int),
          hipMemcpyDeviceToHost,
          stream));
      NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
          result.pred_edge.data(),
          scratch.pred_edge.get(),
          result.pred_edge.size() * sizeof(Offset),
          hipMemcpyDeviceToHost,
          stream));
      NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
      result.target_reached = true;
    }
    result.iterations_used = iterations;
    result.converged = converged;
    result.stopped_on_target = stopped_on_target;
    return result;
  }
};

NearFarCsrGraph::NearFarCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("near_far.upload_graph");
  near_far_detail::validate_host_csr(adjacency);
  impl_ = std::make_shared<Impl>(adjacency, stream);
}

NearFarCsrGraph::~NearFarCsrGraph() = default;
NearFarCsrGraph::NearFarCsrGraph(NearFarCsrGraph&&) noexcept = default;
NearFarCsrGraph& NearFarCsrGraph::operator=(NearFarCsrGraph&&) noexcept =
    default;

NearFarCsrWorkspace::NearFarCsrWorkspace(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : NearFarCsrWorkspace(
          adjacency, stream, NearFarCsrWorkspaceOptions{}) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    const HostCsrF32& adjacency,
    hipStream_t stream,
    NearFarCsrWorkspaceOptions options)
    : NearFarCsrWorkspace(
          std::make_shared<NearFarCsrGraph>(adjacency, stream),
          stream,
          options) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    std::shared_ptr<const NearFarCsrGraph> adjacency,
    hipStream_t stream)
    : NearFarCsrWorkspace(
          std::move(adjacency),
          stream,
          NearFarCsrWorkspaceOptions{}) {}

NearFarCsrWorkspace::NearFarCsrWorkspace(
    std::shared_ptr<const NearFarCsrGraph> adjacency,
    hipStream_t stream,
    NearFarCsrWorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(adjacency), stream, options)) {}

NearFarCsrWorkspace::~NearFarCsrWorkspace() = default;
NearFarCsrWorkspace::NearFarCsrWorkspace(
    NearFarCsrWorkspace&&) noexcept = default;
NearFarCsrWorkspace& NearFarCsrWorkspace::operator=(
    NearFarCsrWorkspace&&) noexcept = default;

void NearFarCsrWorkspace::update_vertex_costs(
    const std::vector<float>& costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("near_far.update_vertex_costs");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  if (costs.size() !=
      static_cast<std::size_t>(impl_->graph->adjacency.view.rows)) {
    throw std::invalid_argument(
        "Near-Far vertex cost count does not match graph rows");
  }
  for (const float cost : costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument(
          "Near-Far vertex costs must be finite and nonnegative");
    }
  }
  if (impl_->vertex_costs.size() < costs.size()) {
    impl_->vertex_costs.reset(costs.size());
  }
  NEAR_FAR_HIP_CHECK(hipMemcpyAsync(
      impl_->vertex_costs.get(),
      costs.data(),
      costs.size() * sizeof(float),
      hipMemcpyHostToDevice,
      stream));
  NEAR_FAR_HIP_CHECK(hipStreamSynchronize(stream));
  impl_->has_vertex_costs = true;
}

void NearFarCsrWorkspace::clear_vertex_costs(hipStream_t stream) {
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  impl_->has_vertex_costs = false;
}

NearFarCsrResult NearFarCsrWorkspace::run_distances(
    const std::vector<int>& sources,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("near_far.run_distances");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  near_far_detail::validate_sources(
      impl_->graph->adjacency.view.rows, sources);
  near_far_detail::validate_delta(delta);
  if (max_iters < -1) {
    throw std::invalid_argument(
        "Near-Far max iterations must be -1 or nonnegative");
  }
  return impl_->run_impl(sources,
                         nullptr,
                         delta,
                         max_iters,
                         progress_callback,
                         progress_user_data);
}

NearFarCsrResult NearFarCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("near_far.run");
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  impl_->require_context(stream);
  near_far_detail::validate_sources(
      impl_->graph->adjacency.view.rows, sources);
  near_far_detail::validate_targets(
      impl_->graph->adjacency.view.rows, targets);
  near_far_detail::validate_delta(delta);
  if (max_iters < -1) {
    throw std::invalid_argument(
        "Near-Far max iterations must be -1 or nonnegative");
  }
  NearFarCsrResult result = impl_->run_impl(
      sources,
      &targets,
      delta,
      max_iters,
      progress_callback,
      progress_user_data);
  result.target = -1;
  return result;
}

NearFarCsrResult NearFarCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  if (!impl_) {
    throw std::runtime_error(
        "NearFarCsrWorkspace has no implementation");
  }
  if (target < 0 ||
      static_cast<minplus_sparse::Offset>(target) >=
          impl_->graph->adjacency.view.rows) {
    throw std::out_of_range(
        "Near-Far target vertex is outside the CSR graph");
  }
  NearFarCsrResult result =
      run(sources,
          std::vector<int>{target},
          delta,
          max_iters,
          stream,
          progress_callback,
          progress_user_data);
  result.target = target;
  result.target_distance = result.target_distances.front();
  result.target_reached = std::isfinite(result.target_distance);
  return result;
}

NearFarCsrResult NearFarCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    NearFarCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}
