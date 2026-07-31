// Bounded, dynamically weighted, true-multi-source active-frontier
// Bellman-Ford for PathFinder. The cooperative controller keeps every
// relaxation round and target certificate on the GPU.

#include "bf11.hpp"

#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::atomic<std::uint64_t> g_bf11_gpu_controller_launches{0};
std::atomic<std::uint64_t> g_bf11_controller_fallbacks{0};
std::atomic<std::uint64_t> g_bf11_target_checks{0};
std::atomic<std::uint64_t> g_bf11_auto_unbounded_retries{0};

}  // namespace

extern "C" void bf11_internal_reset_counters() {
  g_bf11_gpu_controller_launches.store(0, std::memory_order_relaxed);
  g_bf11_controller_fallbacks.store(0, std::memory_order_relaxed);
  g_bf11_target_checks.store(0, std::memory_order_relaxed);
  g_bf11_auto_unbounded_retries.store(0, std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_gpu_controller_launch_count() {
  return g_bf11_gpu_controller_launches.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_controller_fallback_count() {
  return g_bf11_controller_fallbacks.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_target_check_count() {
  return g_bf11_target_checks.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf11_internal_auto_unbounded_retry_count() {
  return g_bf11_auto_unbounded_retries.load(std::memory_order_relaxed);
}

namespace rips_sssp_bf11 {

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;
using DeviceOffset = std::uint32_t;

constexpr unsigned int kNoPredecessor = 0xffffffffu;
constexpr unsigned int kInfinityBits = 0x7f800000u;
constexpr unsigned kGridX = 65535u;
constexpr int kBlockSize = 256;
constexpr std::int32_t kMissingCoordinate =
    routing::interchange::kMissingRouteCoordinate;

static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4 &&
                  std::numeric_limits<float>::is_iec559,
              "BF11 packed distance ordering requires IEEE-754 float32");
static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int),
              "BF11 packed predecessor state requires a 64-bit CAS word");

struct DeviceGraph {
  Offset rows = 0;
  Offset nnz = 0;
  const DeviceOffset* rowptr = nullptr;
  const Index* to = nullptr;
  const float* edge_values = nullptr;
  const std::int32_t* route_end_x = nullptr;
  const std::int32_t* route_end_y = nullptr;
  const float* base_vertex_cost = nullptr;
};

struct IterationStatus {
  int next_count = 0;
  int error_status = 0;
  int reached_target_count = 0;
  unsigned int min_next_frontier_dist_bits = kInfinityBits;
  unsigned int max_target_dist_bits = 0;
};

struct ControllerResult {
  int iterations_used = 0;
  int error_status = 0;
  int frontier_count = 0;
  int current_frontier_index = 0;
  int converged = 0;
  int early_stopped = 0;
  int hit_max_iters = 0;
  int done = 0;
  int target_checks = 0;
};

enum TargetPathStatus : int {
  kTargetUnreachable = 0,
  kTargetPathValid = 1,
  kTargetPathInvalid = 2,
};

struct TargetSummary {
  unsigned long long state = 0;
  unsigned long long node_count = 0;
  unsigned long long edge_count = 0;
  int root = -1;
  int status = kTargetUnreachable;
};

struct DeviceGraphOwner {
  DeviceGraph view{};
  DeviceOffset* rowptr = nullptr;
  Index* to = nullptr;
  float* edge_values = nullptr;
  std::int32_t* route_end_x = nullptr;
  std::int32_t* route_end_y = nullptr;
  float* base_vertex_cost = nullptr;
};

struct DeviceWorkspace {
  Offset rows = 0;
  hipStream_t stream = nullptr;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  int* next_marks = nullptr;
  unsigned char* source_mask = nullptr;
  float* dynamic_vertex_cost = nullptr;
  IterationStatus* iteration_status = nullptr;
  IterationStatus* host_iteration_status = nullptr;
  ControllerResult* controller_result = nullptr;
  ControllerResult* host_controller_result = nullptr;
  Index* source_nodes = nullptr;
  Offset source_capacity = 0;
  Index* target_nodes = nullptr;
  Offset target_capacity = 0;
  Index* update_nodes = nullptr;
  float* update_costs = nullptr;
  Offset update_capacity = 0;
  TargetSummary* target_summaries = nullptr;
  TargetSummary* host_target_summaries = nullptr;
  unsigned long long* reconstruction_node_offsets = nullptr;
  unsigned long long* reconstruction_edge_offsets = nullptr;
  Offset reconstruction_capacity = 0;
  Index* compact_nodes = nullptr;
  DeviceOffset* compact_edges = nullptr;
  float* compact_edge_costs = nullptr;
  std::size_t compact_node_capacity = 0;
  std::size_t compact_edge_capacity = 0;
  int cooperative_blocks = -1;
};

struct SsspStatus {
  int iterations_used = 0;
  bool converged = false;
  bool early_stopped = false;
  bool hit_max_iters = false;
};

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
  }
}

// HIP allocations and streams are device-affine. Destruction can happen after
// the caller has selected another device, so temporarily restore the recorded
// owner instead of issuing invalid frees on the caller's current device. A
// failed switch makes cleanup a deliberate leak rather than risking cross-
// device corruption; destructors cannot report a useful exception safely.
class ScopedOwningHipDevice {
 public:
  explicit ScopedOwningHipDevice(int owner) noexcept {
    if (hipGetDevice(&previous_) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
    if (previous_ == owner) {
      active_ = true;
      return;
    }
    if (hipSetDevice(owner) == hipSuccess) {
      active_ = true;
      restore_ = true;
    } else {
      (void)hipGetLastError();
    }
  }

  ~ScopedOwningHipDevice() noexcept {
    if (restore_ && hipSetDevice(previous_) != hipSuccess) {
      (void)hipGetLastError();
    }
  }

  bool active() const noexcept { return active_; }

 private:
  int previous_ = -1;
  bool active_ = false;
  bool restore_ = false;
};

class DrainStreamOnException {
 public:
  explicit DrainStreamOnException(hipStream_t stream)
      : stream_(stream), exceptions_(std::uncaught_exceptions()) {}
  ~DrainStreamOnException() noexcept {
    if (std::uncaught_exceptions() > exceptions_) {
      (void)hipStreamSynchronize(stream_);
    }
  }

 private:
  hipStream_t stream_ = nullptr;
  int exceptions_ = 0;
};

dim3 grid_for_items(Offset items) {
  if (items <= 0) return dim3(1, 1);
  const Offset blocks = (items + kBlockSize - 1) / kBlockSize;
  const unsigned gx = blocks < static_cast<Offset>(kGridX)
                          ? static_cast<unsigned>(blocks)
                          : kGridX;
  const unsigned gy = static_cast<unsigned>(
      (blocks + static_cast<Offset>(gx) - 1) / static_cast<Offset>(gx));
  return dim3(gx, gy);
}

template <typename T>
T* device_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflow");
  }
  T* result = nullptr;
  check_hip(hipMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)), what);
  return result;
}

template <typename T>
T* pinned_allocate(std::size_t count, const char* what) {
  if (count == 0) return nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflow");
  }
  T* result = nullptr;
  check_hip(hipHostMalloc(reinterpret_cast<void**>(&result),
                         count * sizeof(T), hipHostMallocDefault),
            what);
  return result;
}

__device__ __forceinline__ Offset logical_thread_id() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

__host__ __device__ __forceinline__ unsigned long long pack_state(
    unsigned int distance_bits,
    unsigned int predecessor) {
  return (static_cast<unsigned long long>(distance_bits) << 32) |
         static_cast<unsigned long long>(predecessor);
}

__host__ __device__ __forceinline__ unsigned int state_distance_bits(
    unsigned long long state) {
  return static_cast<unsigned int>(state >> 32);
}

__device__ __forceinline__ float state_distance(unsigned long long state) {
  return __uint_as_float(state_distance_bits(state));
}

float host_state_distance(unsigned long long state) {
  const unsigned int bits = state_distance_bits(state);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

__device__ __forceinline__ bool finite_device(float value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

// Path reconstruction is tiny compared with traversal, so recover an edge's
// source from CSR row offsets instead of retaining a 4-byte source sidecar for
// every edge. This saves substantial device memory and upload bandwidth on the
// production graph while preserving original CSR edge IDs.
__device__ __forceinline__ Index source_for_edge(
    const DeviceOffset* rowptr,
    Offset rows,
    DeviceOffset edge) {
  Offset low = 0;
  Offset high = rows;
  while (low + 1 < high) {
    const Offset middle = low + (high - low) / 2;
    if (rowptr[middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return rowptr[low] <= edge && edge < rowptr[low + 1]
             ? static_cast<Index>(low)
             : static_cast<Index>(-1);
}

__device__ __forceinline__ bool node_admitted(
    const DeviceGraph& graph,
    Index node,
    const BellmanFord11BoundingBox& bounds) {
  if (!bounds.enabled) return true;
  const std::int32_t x = graph.route_end_x[node];
  const std::int32_t y = graph.route_end_y[node];
  if (x == kMissingCoordinate && y == kMissingCoordinate) {
    return true;
  }
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

// Distance-only comparison is deliberate. Equal-distance predecessor rewrites
// can create cycles around zero-cost edges and are irrelevant to SSSP distance
// correctness. The first strict label improvement therefore owns its parent.
__device__ __forceinline__ bool atomic_relax_strict(
    unsigned long long* address,
    float candidate,
    DeviceOffset predecessor) {
  const unsigned int candidate_bits = __float_as_uint(candidate);
  unsigned long long old_state = *address;
  while (candidate_bits < state_distance_bits(old_state)) {
    const unsigned long long desired = pack_state(candidate_bits, predecessor);
    const unsigned long long assumed = old_state;
    old_state = atomicCAS(address, assumed, desired);
    if (old_state == assumed) return true;
  }
  return false;
}

__device__ __forceinline__ float effective_edge_weight(
    float edge_value,
    float base_cost,
    float dynamic_cost) {
  // Preserve the exact zero-multiplier contract without allowing an earlier
  // large-factor product to overflow to infinity and then form inf * 0.
  return edge_value == 0.0f || dynamic_cost == 0.0f
             ? 0.0f
             : (edge_value * base_cost) * dynamic_cost;
}

__device__ __forceinline__ unsigned int relax_edge(
    const DeviceGraph& graph,
    Offset edge,
    float from_distance,
    const float* __restrict__ dynamic_vertex_cost,
    const BellmanFord11BoundingBox& bounds,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    const unsigned char* __restrict__ source_mask,
    IterationStatus* __restrict__ status) {
  const Index dst = graph.to[edge];
  if (dst < 0 || static_cast<Offset>(dst) >= graph.rows) {
    atomicExch(&status->error_status, 1);
    return kInfinityBits;
  }
  if (source_mask[dst] != 0 || !node_admitted(graph, dst, bounds)) {
    return kInfinityBits;
  }

  const float edge_value = graph.edge_values[edge];
  const float base_cost = graph.base_vertex_cost[dst];
  const float dynamic_cost = dynamic_vertex_cost[dst];
  const float effective_weight =
      effective_edge_weight(edge_value, base_cost, dynamic_cost);
  if (!finite_device(effective_weight) || effective_weight < 0.0f) {
    atomicExch(&status->error_status, 3);
    return kInfinityBits;
  }
  const float candidate = from_distance + effective_weight;
  if (!finite_device(candidate)) return kInfinityBits;

  if (!atomic_relax_strict(&best_state[dst], candidate,
                           static_cast<DeviceOffset>(edge))) {
    return kInfinityBits;
  }
  const int old_mark = atomicExch(&next_marks[dst], mark_token);
  if (old_mark != mark_token) {
    const int slot = atomicAdd(&status->next_count, 1);
    if (slot < 0 || static_cast<Offset>(slot) >= graph.rows) {
      atomicExch(&status->error_status, 4);
      return kInfinityBits;
    }
    next_frontier[slot] = dst;
  }
  return __float_as_uint(candidate);
}

__global__ void clear_state_kernel(Offset rows,
                                   unsigned long long* best_state,
                                   int* next_marks,
                                   unsigned char* source_mask) {
  const Offset row = logical_thread_id();
  if (row >= rows) return;
  best_state[row] = pack_state(kInfinityBits, kNoPredecessor);
  next_marks[row] = 0;
  source_mask[row] = 0;
}

__global__ void seed_sources_kernel(const Index* source_nodes,
                                    Offset source_count,
                                    unsigned long long* best_state,
                                    Index* frontier,
                                    unsigned char* source_mask) {
  const Offset item = logical_thread_id();
  if (item >= source_count) return;
  const Index source = source_nodes[item];
  best_state[source] = pack_state(0u, kNoPredecessor);
  source_mask[source] = 1;
  frontier[item] = source;
}

__global__ void reset_iteration_status_kernel(IterationStatus* status) {
  if (logical_thread_id() != 0) return;
  status->next_count = 0;
  status->error_status = 0;
  status->reached_target_count = 0;
  status->min_next_frontier_dist_bits = kInfinityBits;
  status->max_target_dist_bits = 0;
}

__global__ void frontier_relax_kernel(
    DeviceGraph graph,
    const Index* frontier,
    int frontier_count,
    int mark_token,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    unsigned long long* best_state,
    Index* next_frontier,
    int* next_marks,
    const unsigned char* source_mask,
    IterationStatus* status) {
  const Offset item = logical_thread_id();
  if (item >= static_cast<Offset>(frontier_count)) return;
  const Index from = frontier[item];
  if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
    atomicExch(&status->error_status, 1);
    return;
  }
  const float from_dist = state_distance(best_state[from]);
  if (!finite_device(from_dist)) return;
  const Offset begin = static_cast<Offset>(graph.rowptr[from]);
  const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
  if (end < begin || end > graph.nnz) {
    atomicExch(&status->error_status, 2);
    return;
  }
  unsigned int local_min = kInfinityBits;
  for (Offset edge = begin; edge < end; ++edge) {
    const unsigned int candidate = relax_edge(
        graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
        best_state, next_frontier, next_marks, source_mask, status);
    local_min = candidate < local_min ? candidate : local_min;
  }
  if (local_min != kInfinityBits) {
    atomicMin(&status->min_next_frontier_dist_bits, local_min);
  }
}

__global__ void update_target_status_kernel(
    const unsigned long long* best_state,
    const Index* target_nodes,
    Offset target_count,
    IterationStatus* status) {
  const Offset item = logical_thread_id();
  if (item >= target_count) return;
  const unsigned int bits = state_distance_bits(best_state[target_nodes[item]]);
  if (bits == kInfinityBits) return;
  atomicAdd(&status->reached_target_count, 1);
  atomicMax(&status->max_target_dist_bits, bits);
}

__global__ void sparse_cost_update_kernel(const Index* nodes,
                                          const float* costs,
                                          Offset count,
                                          float* dynamic_vertex_cost) {
  const Offset item = logical_thread_id();
  if (item < count) dynamic_vertex_cost[nodes[item]] = costs[item];
}

__global__ void fill_cost_kernel(Offset count, float* costs, float value) {
  const Offset item = logical_thread_id();
  if (item < count) costs[item] = value;
}

__global__ void frontier_controller_kernel(
    DeviceGraph graph,
    const Index* source_nodes,
    Offset source_count,
    const Index* target_nodes,
    Offset target_count,
    const float* dynamic_vertex_cost,
    BellmanFord11BoundingBox bounds,
    int max_iters,
    int target_check_interval,
    unsigned long long* best_state,
    Index* frontier,
    Index* next_frontier,
    int* next_marks,
    unsigned char* source_mask,
    IterationStatus* iteration_status,
    ControllerResult* controller_result) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const Offset thread = static_cast<Offset>(grid.thread_rank());
  const Offset thread_count = static_cast<Offset>(grid.size());

  for (Offset row = thread; row < graph.rows; row += thread_count) {
    best_state[row] = pack_state(kInfinityBits, kNoPredecessor);
    next_marks[row] = 0;
    source_mask[row] = 0;
  }
  grid.sync();
  for (Offset item = thread; item < source_count; item += thread_count) {
    const Index source = source_nodes[item];
    best_state[source] = pack_state(0u, kNoPredecessor);
    source_mask[source] = 1;
    frontier[item] = source;
  }
  if (thread == 0) {
    controller_result->iterations_used = 0;
    controller_result->error_status = 0;
    controller_result->frontier_count = static_cast<int>(source_count);
    controller_result->current_frontier_index = 0;
    controller_result->converged = 0;
    controller_result->early_stopped = 0;
    controller_result->hit_max_iters = max_iters == 0 ? 1 : 0;
    controller_result->done = max_iters == 0 ? 1 : 0;
    controller_result->target_checks = 0;
  }
  grid.sync();

  while (controller_result->done == 0) {
    if (thread == 0) {
      iteration_status->next_count = 0;
      iteration_status->error_status = 0;
      iteration_status->reached_target_count = 0;
      iteration_status->min_next_frontier_dist_bits = kInfinityBits;
      iteration_status->max_target_dist_bits = 0;
    }
    grid.sync();

    const int frontier_count = controller_result->frontier_count;
    const int current_index = controller_result->current_frontier_index;
    const int completed_iteration = controller_result->iterations_used + 1;
    const int mark_token = completed_iteration;
    const Index* current = current_index == 0 ? frontier : next_frontier;
    Index* next = current_index == 0 ? next_frontier : frontier;

    for (Offset item = thread; item < static_cast<Offset>(frontier_count);
         item += thread_count) {
      const Index from = current[item];
      if (from < 0 || static_cast<Offset>(from) >= graph.rows) {
        atomicExch(&iteration_status->error_status, 1);
        continue;
      }
      const float from_dist = state_distance(best_state[from]);
      if (!finite_device(from_dist)) continue;
      const Offset begin = static_cast<Offset>(graph.rowptr[from]);
      const Offset end = static_cast<Offset>(graph.rowptr[from + 1]);
      if (end < begin || end > graph.nnz) {
        atomicExch(&iteration_status->error_status, 2);
        continue;
      }
      unsigned int local_min = kInfinityBits;
      for (Offset edge = begin; edge < end; ++edge) {
        const unsigned int candidate = relax_edge(
            graph, edge, from_dist, dynamic_vertex_cost, bounds, mark_token,
            best_state, next, next_marks, source_mask, iteration_status);
        local_min = candidate < local_min ? candidate : local_min;
      }
      if (local_min != kInfinityBits) {
        atomicMin(&iteration_status->min_next_frontier_dist_bits, local_min);
      }
    }
    grid.sync();

    const bool check_targets =
        completed_iteration % target_check_interval == 0;
    if (check_targets) {
      for (Offset item = thread; item < target_count; item += thread_count) {
        const unsigned int bits =
            state_distance_bits(best_state[target_nodes[item]]);
        if (bits != kInfinityBits) {
          atomicAdd(&iteration_status->reached_target_count, 1);
          atomicMax(&iteration_status->max_target_dist_bits, bits);
        }
      }
    }
    grid.sync();

    if (thread == 0) {
      controller_result->iterations_used = completed_iteration;
      controller_result->frontier_count = iteration_status->next_count;
      if (check_targets) ++controller_result->target_checks;
      if (iteration_status->error_status != 0) {
        controller_result->error_status = iteration_status->error_status;
        controller_result->done = 1;
      } else if (iteration_status->next_count == 0) {
        controller_result->converged = 1;
        controller_result->done = 1;
      } else if (check_targets &&
                 iteration_status->reached_target_count ==
                     static_cast<int>(target_count) &&
                 iteration_status->min_next_frontier_dist_bits >=
                     iteration_status->max_target_dist_bits) {
        // Every future path must begin at a next-frontier label whose distance
        // is at least this lower bound. Nonnegative frozen edge costs
        // therefore certify all requested target distances exactly.
        controller_result->early_stopped = 1;
        controller_result->done = 1;
      } else if (completed_iteration >= max_iters) {
        controller_result->hit_max_iters = 1;
        controller_result->done = 1;
      } else {
        controller_result->current_frontier_index ^= 1;
      }
    }
    grid.sync();
  }
}

__global__ void summarize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* to,
    const DeviceOffset* rowptr,
    const Index* targets,
    Offset target_count,
    TargetSummary* summaries) {
  const Offset item = logical_thread_id();
  if (item >= target_count) return;
  TargetSummary summary{};
  const Index target = targets[item];
  summary.state = best_state[target];
  if (state_distance_bits(summary.state) == kInfinityBits) {
    summaries[item] = summary;
    return;
  }

  Index current = target;
  unsigned long long edge_count = 0;
  for (Offset guard = 0; guard <= rows; ++guard) {
    const unsigned long long state = best_state[current];
    const DeviceOffset edge = static_cast<DeviceOffset>(state);
    if (edge == kNoPredecessor) {
      if (source_mask[current] != 0 && state_distance_bits(state) == 0u) {
        summary.node_count = edge_count + 1;
        summary.edge_count = edge_count;
        summary.root = current;
        summary.status = kTargetPathValid;
      } else {
        summary.status = kTargetPathInvalid;
      }
      summaries[item] = summary;
      return;
    }
    if (static_cast<Offset>(edge) >= nnz || to[edge] != current) {
      summary.status = kTargetPathInvalid;
      summaries[item] = summary;
      return;
    }
    const Index predecessor = source_for_edge(rowptr, rows, edge);
    if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) {
      summary.status = kTargetPathInvalid;
      summaries[item] = summary;
      return;
    }
    current = predecessor;
    ++edge_count;
  }
  summary.status = kTargetPathInvalid;
  summaries[item] = summary;
}

__global__ void materialize_target_paths_kernel(
    const unsigned long long* best_state,
    const unsigned char* source_mask,
    Offset rows,
    Offset nnz,
    const Index* to,
    const DeviceOffset* rowptr,
    const Index* targets,
    const TargetSummary* summaries,
    const unsigned long long* node_offsets,
    const unsigned long long* edge_offsets,
    Offset target_count,
    Index* compact_nodes,
    DeviceOffset* compact_edges,
    const float* edge_values,
    const float* base_vertex_cost,
    const float* dynamic_vertex_cost,
    float* compact_edge_costs) {
  const Offset item = logical_thread_id();
  if (item >= target_count ||
      summaries[item].status != kTargetPathValid) {
    return;
  }
  const TargetSummary summary = summaries[item];
  const unsigned long long node_base = node_offsets[item];
  const unsigned long long edge_base = edge_offsets[item];
  Index current = targets[item];
  compact_nodes[node_base + summary.edge_count] = current;
  for (unsigned long long remaining = summary.edge_count; remaining > 0;
       --remaining) {
    const DeviceOffset edge =
        static_cast<DeviceOffset>(best_state[current]);
    if (edge == kNoPredecessor || static_cast<Offset>(edge) >= nnz ||
        to[edge] != current) {
      return;
    }
    const Index predecessor = source_for_edge(rowptr, rows, edge);
    if (predecessor < 0 || static_cast<Offset>(predecessor) >= rows) return;
    compact_edges[edge_base + remaining - 1] = edge;
    compact_edge_costs[edge_base + remaining - 1] = effective_edge_weight(
        edge_values[edge], base_vertex_cost[current],
        dynamic_vertex_cost[current]);
    compact_nodes[node_base + remaining - 1] = predecessor;
    current = predecessor;
  }
  // summarize_target_paths_kernel already certified this condition. Keep the
  // reads here so malformed concurrent mutation cannot silently appear valid.
  if (current != summary.root || source_mask[current] == 0) return;
}

struct HostSidecarView {
  const std::vector<std::int32_t>* route_end_x = nullptr;
  const std::vector<std::int32_t>* route_end_y = nullptr;
  const std::vector<float>* base_vertex_cost = nullptr;
};

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols ||
      graph.rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::invalid_argument(
        "BF11 requires a nonempty square CSR graph with int vertex IDs");
  }
  if (graph.nnz < 0 ||
      static_cast<unsigned long long>(graph.nnz) >= kNoPredecessor) {
    throw std::invalid_argument(
        "BF11 CSR edge IDs must fit below its predecessor sentinel");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("BF11 CSR arrays do not match rows and nnz");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::invalid_argument("BF11 CSR row offsets are not monotone");
    }
  }
  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const Index dst = graph.colind[static_cast<std::size_t>(edge)];
    const float value = graph.values[static_cast<std::size_t>(edge)];
    if (dst < 0 || static_cast<Offset>(dst) >= graph.rows ||
        !std::isfinite(value) || value < 0.0f) {
      throw std::invalid_argument(
          "BF11 CSR destinations and weights must be valid and nonnegative");
    }
  }
}

void validate_sidecar_view(const HostSidecarView& view, Offset rows) {
  if (!view.route_end_x || !view.route_end_y || !view.base_vertex_cost ||
      view.route_end_x->size() != static_cast<std::size_t>(rows) ||
      view.route_end_y->size() != static_cast<std::size_t>(rows) ||
      view.base_vertex_cost->size() != static_cast<std::size_t>(rows)) {
    throw std::invalid_argument("BF11 node sidecars must contain exactly V entries");
  }
  for (Offset row = 0; row < rows; ++row) {
    const std::int32_t x = (*view.route_end_x)[static_cast<std::size_t>(row)];
    const std::int32_t y = (*view.route_end_y)[static_cast<std::size_t>(row)];
    const bool missing_x = x == kMissingCoordinate;
    const bool missing_y = y == kMissingCoordinate;
    const float base =
        (*view.base_vertex_cost)[static_cast<std::size_t>(row)];
    if (missing_x != missing_y || (!missing_x && (x < 0 || y < 0))) {
      throw std::invalid_argument("BF11 route-end coordinates are malformed");
    }
    if (!std::isfinite(base) || !(base > 0.0f)) {
      throw std::invalid_argument(
          "BF11 base vertex costs must be finite and positive");
    }
  }
}

HostSidecarView common_sidecar_view(
    const routing::interchange::RoutingCsrSidecars& sidecars,
    const HostCsrF32& graph) {
  routing::interchange::validate_routing_csr_sidecars(
      sidecars, static_cast<std::size_t>(graph.rows),
      static_cast<std::size_t>(graph.nnz), false);
  return {&sidecars.route_end_x, &sidecars.route_end_y,
          &sidecars.base_vertex_cost};
}

HostSidecarView compatibility_sidecar_view(
    const BellmanFord11NodeSidecars& sidecars,
    const HostCsrF32& graph) {
  HostSidecarView view{&sidecars.route_end_x, &sidecars.route_end_y,
                       &sidecars.base_vertex_costs};
  validate_sidecar_view(view, graph.rows);
  return view;
}

DeviceGraphOwner copy_graph_to_device(const HostCsrF32& graph,
                                      const HostSidecarView& sidecars,
                                      hipStream_t stream) {
  DeviceGraphOwner owner;
  DrainStreamOnException drain(stream);
  try {
    std::vector<DeviceOffset> compact_rowptr;
    compact_rowptr.reserve(graph.rowptr.size());
    for (const Offset offset : graph.rowptr) {
      if (offset < 0 ||
          static_cast<unsigned long long>(offset) >= kNoPredecessor) {
        throw std::overflow_error("BF11 row offset cannot fit uint32 storage");
      }
      compact_rowptr.push_back(static_cast<DeviceOffset>(offset));
    }
    owner.rowptr = device_allocate<DeviceOffset>(compact_rowptr.size(),
                                                  "hipMalloc BF11 rowptr");
    check_hip(hipMemcpyAsync(owner.rowptr, compact_rowptr.data(),
                             compact_rowptr.size() * sizeof(DeviceOffset),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 rowptr");
    if (graph.nnz > 0) {
      const std::size_t edges = static_cast<std::size_t>(graph.nnz);
      owner.to = device_allocate<Index>(edges, "hipMalloc BF11 destinations");
      owner.edge_values =
          device_allocate<float>(edges, "hipMalloc BF11 edge values");
      check_hip(hipMemcpyAsync(owner.to, graph.colind.data(),
                               edges * sizeof(Index), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 destinations");
      check_hip(hipMemcpyAsync(owner.edge_values, graph.values.data(),
                               edges * sizeof(float), hipMemcpyHostToDevice,
                               stream),
                "copy BF11 edge values");
    }
    const std::size_t rows = static_cast<std::size_t>(graph.rows);
    owner.route_end_x =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end X");
    owner.route_end_y =
        device_allocate<std::int32_t>(rows, "hipMalloc BF11 route-end Y");
    owner.base_vertex_cost =
        device_allocate<float>(rows, "hipMalloc BF11 base vertex costs");
    check_hip(hipMemcpyAsync(owner.route_end_x, sidecars.route_end_x->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end X");
    check_hip(hipMemcpyAsync(owner.route_end_y, sidecars.route_end_y->data(),
                             rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream),
              "copy BF11 route-end Y");
    check_hip(hipMemcpyAsync(owner.base_vertex_cost,
                             sidecars.base_vertex_cost->data(),
                             rows * sizeof(float), hipMemcpyHostToDevice, stream),
              "copy BF11 base vertex costs");
    check_hip(hipStreamSynchronize(stream), "synchronize BF11 graph upload");

    owner.view = {graph.rows,
                  graph.nnz,
                  owner.rowptr,
                  owner.to,
                  owner.edge_values,
                  owner.route_end_x,
                  owner.route_end_y,
                  owner.base_vertex_cost};
    return owner;
  } catch (...) {
    // An allocation failure can occur after earlier asynchronous copies were
    // queued. Drain them before releasing their destinations.
    (void)hipStreamSynchronize(stream);
    if (owner.rowptr) (void)hipFree(owner.rowptr);
    if (owner.to) (void)hipFree(owner.to);
    if (owner.edge_values) (void)hipFree(owner.edge_values);
    if (owner.route_end_x) (void)hipFree(owner.route_end_x);
    if (owner.route_end_y) (void)hipFree(owner.route_end_y);
    if (owner.base_vertex_cost) (void)hipFree(owner.base_vertex_cost);
    throw;
  }
}

void free_graph(DeviceGraphOwner* owner) noexcept {
  if (!owner) return;
  if (owner->rowptr) (void)hipFree(owner->rowptr);
  if (owner->to) (void)hipFree(owner->to);
  if (owner->edge_values) (void)hipFree(owner->edge_values);
  if (owner->route_end_x) (void)hipFree(owner->route_end_x);
  if (owner->route_end_y) (void)hipFree(owner->route_end_y);
  if (owner->base_vertex_cost) (void)hipFree(owner->base_vertex_cost);
  *owner = {};
}

DeviceWorkspace make_workspace(Offset rows, hipStream_t stream) {
  DeviceWorkspace workspace;
  workspace.rows = rows;
  workspace.stream = stream;
  try {
    const std::size_t count = static_cast<std::size_t>(rows);
    workspace.best_state = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 packed states");
    workspace.frontier =
        device_allocate<Index>(count, "hipMalloc BF11 frontier");
    workspace.next_frontier =
        device_allocate<Index>(count, "hipMalloc BF11 next frontier");
    workspace.next_marks =
        device_allocate<int>(count, "hipMalloc BF11 frontier marks");
    workspace.source_mask = device_allocate<unsigned char>(
        count, "hipMalloc BF11 source mask");
    workspace.dynamic_vertex_cost = device_allocate<float>(
        count, "hipMalloc BF11 dynamic vertex costs");
    workspace.iteration_status = device_allocate<IterationStatus>(
        1, "hipMalloc BF11 iteration status");
    workspace.host_iteration_status = pinned_allocate<IterationStatus>(
        1, "hipHostMalloc BF11 iteration status");
    workspace.controller_result = device_allocate<ControllerResult>(
        1, "hipMalloc BF11 controller result");
    workspace.host_controller_result = pinned_allocate<ControllerResult>(
        1, "hipHostMalloc BF11 controller result");
    hipLaunchKernelGGL(fill_cost_kernel, grid_for_items(rows), dim3(kBlockSize),
                       0, stream, rows, workspace.dynamic_vertex_cost, 1.0f);
    check_hip(hipGetLastError(), "initialize BF11 dynamic vertex costs");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF11 workspace initialization");
    return workspace;
  } catch (...) {
    // The initialization kernel uses the caller's stream. If an asynchronous
    // launch or synchronization failure surfaced above, drain best-effort
    // before freeing any destination allocation.
    (void)hipStreamSynchronize(stream);
    if (workspace.best_state) (void)hipFree(workspace.best_state);
    if (workspace.frontier) (void)hipFree(workspace.frontier);
    if (workspace.next_frontier) (void)hipFree(workspace.next_frontier);
    if (workspace.next_marks) (void)hipFree(workspace.next_marks);
    if (workspace.source_mask) (void)hipFree(workspace.source_mask);
    if (workspace.dynamic_vertex_cost)
      (void)hipFree(workspace.dynamic_vertex_cost);
    if (workspace.iteration_status) (void)hipFree(workspace.iteration_status);
    if (workspace.host_iteration_status)
      (void)hipHostFree(workspace.host_iteration_status);
    if (workspace.controller_result) (void)hipFree(workspace.controller_result);
    if (workspace.host_controller_result)
      (void)hipHostFree(workspace.host_controller_result);
    throw;
  }
}

void free_workspace(DeviceWorkspace* workspace) noexcept {
  if (!workspace) return;
  (void)hipStreamSynchronize(workspace->stream);
  if (workspace->best_state) (void)hipFree(workspace->best_state);
  if (workspace->frontier) (void)hipFree(workspace->frontier);
  if (workspace->next_frontier) (void)hipFree(workspace->next_frontier);
  if (workspace->next_marks) (void)hipFree(workspace->next_marks);
  if (workspace->source_mask) (void)hipFree(workspace->source_mask);
  if (workspace->dynamic_vertex_cost)
    (void)hipFree(workspace->dynamic_vertex_cost);
  if (workspace->iteration_status) (void)hipFree(workspace->iteration_status);
  if (workspace->host_iteration_status)
    (void)hipHostFree(workspace->host_iteration_status);
  if (workspace->controller_result) (void)hipFree(workspace->controller_result);
  if (workspace->host_controller_result)
    (void)hipHostFree(workspace->host_controller_result);
  if (workspace->source_nodes) (void)hipFree(workspace->source_nodes);
  if (workspace->target_nodes) (void)hipFree(workspace->target_nodes);
  if (workspace->update_nodes) (void)hipFree(workspace->update_nodes);
  if (workspace->update_costs) (void)hipFree(workspace->update_costs);
  if (workspace->target_summaries) (void)hipFree(workspace->target_summaries);
  if (workspace->host_target_summaries)
    (void)hipHostFree(workspace->host_target_summaries);
  if (workspace->reconstruction_node_offsets)
    (void)hipFree(workspace->reconstruction_node_offsets);
  if (workspace->reconstruction_edge_offsets)
    (void)hipFree(workspace->reconstruction_edge_offsets);
  if (workspace->compact_nodes) (void)hipFree(workspace->compact_nodes);
  if (workspace->compact_edges) (void)hipFree(workspace->compact_edges);
  if (workspace->compact_edge_costs)
    (void)hipFree(workspace->compact_edge_costs);
  *workspace = {};
}

Offset geometric_capacity(Offset current, Offset required, Offset limit) {
  Offset result = std::max<Offset>(1, current);
  while (result < required) {
    if (result > limit / 2) return limit;
    result *= 2;
  }
  return result;
}

void ensure_source_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.source_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.source_capacity, required, workspace.rows);
  Index* replacement = device_allocate<Index>(
      static_cast<std::size_t>(capacity), "hipMalloc BF11 source list");
  if (workspace.source_nodes) (void)hipFree(workspace.source_nodes);
  workspace.source_nodes = replacement;
  workspace.source_capacity = capacity;
}

void ensure_target_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.target_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.target_capacity, required, workspace.rows);
  Index* replacement = device_allocate<Index>(
      static_cast<std::size_t>(capacity), "hipMalloc BF11 target list");
  if (workspace.target_nodes) (void)hipFree(workspace.target_nodes);
  workspace.target_nodes = replacement;
  workspace.target_capacity = capacity;
}

void ensure_update_capacity(DeviceWorkspace& workspace, Offset required) {
  if (required <= workspace.update_capacity) return;
  const Offset capacity =
      geometric_capacity(workspace.update_capacity, required, workspace.rows);
  Index* replacement_nodes = nullptr;
  float* replacement_costs = nullptr;
  try {
    replacement_nodes = device_allocate<Index>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 sparse-cost nodes");
    replacement_costs = device_allocate<float>(
        static_cast<std::size_t>(capacity), "hipMalloc BF11 sparse-cost values");
  } catch (...) {
    if (replacement_nodes) (void)hipFree(replacement_nodes);
    if (replacement_costs) (void)hipFree(replacement_costs);
    throw;
  }
  if (workspace.update_nodes) (void)hipFree(workspace.update_nodes);
  if (workspace.update_costs) (void)hipFree(workspace.update_costs);
  workspace.update_nodes = replacement_nodes;
  workspace.update_costs = replacement_costs;
  workspace.update_capacity = capacity;
}

void ensure_reconstruction_capacity(DeviceWorkspace& workspace,
                                    Offset required) {
  if (required <= workspace.reconstruction_capacity) return;
  const Offset capacity = geometric_capacity(
      workspace.reconstruction_capacity, required, workspace.rows);
  const std::size_t count = static_cast<std::size_t>(capacity);
  TargetSummary* summaries = nullptr;
  TargetSummary* host_summaries = nullptr;
  unsigned long long* node_offsets = nullptr;
  unsigned long long* edge_offsets = nullptr;
  try {
    summaries = device_allocate<TargetSummary>(
        count, "hipMalloc BF11 target summaries");
    host_summaries = pinned_allocate<TargetSummary>(
        count, "hipHostMalloc BF11 target summaries");
    node_offsets = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 reconstruction node offsets");
    edge_offsets = device_allocate<unsigned long long>(
        count, "hipMalloc BF11 reconstruction edge offsets");
  } catch (...) {
    if (summaries) (void)hipFree(summaries);
    if (host_summaries) (void)hipHostFree(host_summaries);
    if (node_offsets) (void)hipFree(node_offsets);
    if (edge_offsets) (void)hipFree(edge_offsets);
    throw;
  }
  if (workspace.target_summaries) (void)hipFree(workspace.target_summaries);
  if (workspace.host_target_summaries)
    (void)hipHostFree(workspace.host_target_summaries);
  if (workspace.reconstruction_node_offsets)
    (void)hipFree(workspace.reconstruction_node_offsets);
  if (workspace.reconstruction_edge_offsets)
    (void)hipFree(workspace.reconstruction_edge_offsets);
  workspace.target_summaries = summaries;
  workspace.host_target_summaries = host_summaries;
  workspace.reconstruction_node_offsets = node_offsets;
  workspace.reconstruction_edge_offsets = edge_offsets;
  workspace.reconstruction_capacity = capacity;
}

std::size_t grow_size(std::size_t current, std::size_t required) {
  std::size_t result = std::max<std::size_t>(1, current);
  while (result < required) {
    if (result > std::numeric_limits<std::size_t>::max() / 2) return required;
    result *= 2;
  }
  return result;
}

void ensure_compact_capacity(DeviceWorkspace& workspace,
                             std::size_t required_nodes,
                             std::size_t required_edges) {
  if (required_nodes <= workspace.compact_node_capacity &&
      required_edges <= workspace.compact_edge_capacity) {
    return;
  }
  const std::size_t node_capacity =
      required_nodes > workspace.compact_node_capacity
          ? grow_size(workspace.compact_node_capacity, required_nodes)
          : workspace.compact_node_capacity;
  const std::size_t edge_capacity =
      required_edges > workspace.compact_edge_capacity
          ? grow_size(workspace.compact_edge_capacity, required_edges)
          : workspace.compact_edge_capacity;
  Index* nodes = nullptr;
  DeviceOffset* edges = nullptr;
  float* edge_costs = nullptr;
  try {
    nodes = device_allocate<Index>(node_capacity,
                                   "hipMalloc BF11 compact path nodes");
    edges = device_allocate<DeviceOffset>(edge_capacity,
                                          "hipMalloc BF11 compact path edges");
    edge_costs = device_allocate<float>(
        edge_capacity, "hipMalloc BF11 compact path edge costs");
  } catch (...) {
    if (nodes) (void)hipFree(nodes);
    if (edges) (void)hipFree(edges);
    if (edge_costs) (void)hipFree(edge_costs);
    throw;
  }
  if (workspace.compact_nodes) (void)hipFree(workspace.compact_nodes);
  if (workspace.compact_edges) (void)hipFree(workspace.compact_edges);
  if (workspace.compact_edge_costs)
    (void)hipFree(workspace.compact_edge_costs);
  workspace.compact_nodes = nodes;
  workspace.compact_edges = edges;
  workspace.compact_edge_costs = edge_costs;
  workspace.compact_node_capacity = node_capacity;
  workspace.compact_edge_capacity = edge_capacity;
}

void throw_controller_error(int status) {
  switch (status) {
    case 1:
      throw std::runtime_error("BF11 encountered an invalid vertex ID");
    case 2:
      throw std::runtime_error("BF11 encountered invalid CSR row offsets");
    case 3:
      throw std::runtime_error(
          "BF11 effective edge weight is nonfinite or negative");
    case 4:
      throw std::runtime_error("BF11 next-frontier capacity overflowed");
    default:
      throw std::runtime_error("BF11 controller returned an unknown error");
  }
}

int cooperative_block_count(DeviceWorkspace& workspace) {
  if (workspace.cooperative_blocks >= 0) return workspace.cooperative_blocks;
  int device = -1;
  check_hip(hipGetDevice(&device), "get BF11 HIP device");
  int supported = 0;
  const hipError_t attribute_status = hipDeviceGetAttribute(
      &supported, hipDeviceAttributeCooperativeLaunch, device);
  if (attribute_status != hipSuccess || supported == 0) {
    if (attribute_status != hipSuccess) (void)hipGetLastError();
    workspace.cooperative_blocks = 0;
    return 0;
  }
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, device),
            "get BF11 HIP device properties");
  int blocks_per_cu = 0;
  const hipError_t occupancy_status =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_cu, frontier_controller_kernel, kBlockSize, 0);
  if (occupancy_status != hipSuccess || blocks_per_cu <= 0 ||
      properties.multiProcessorCount <= 0) {
    if (occupancy_status != hipSuccess) (void)hipGetLastError();
    workspace.cooperative_blocks = 0;
    return 0;
  }
  const Offset row_blocks = (workspace.rows + kBlockSize - 1) / kBlockSize;
  const Offset legal = static_cast<Offset>(blocks_per_cu) *
                       static_cast<Offset>(properties.multiProcessorCount);
  // A cooperative grid must fit concurrently, but there is no reason to cap
  // it at one block per CU. Use all occupancy-reported resident blocks so the
  // edge scan has enough waves to hide irregular-memory latency.
  const Offset blocks = std::min(row_blocks, legal);
  workspace.cooperative_blocks = blocks > 0 ? static_cast<int>(blocks) : 0;
  return workspace.cooperative_blocks;
}

SsspStatus run_gpu_controller(const DeviceGraph& graph,
                              DeviceWorkspace& workspace,
                              Offset source_count,
                              Offset target_count,
                              int max_iters,
                              const BellmanFord11RunOptions& options,
                              int blocks) {
  PATHFINDER_PROFILE_RANGE("bf11.gpu_controller");
  DeviceGraph graph_arg = graph;
  const Index* sources_arg = workspace.source_nodes;
  Offset source_count_arg = source_count;
  const Index* targets_arg = workspace.target_nodes;
  Offset target_count_arg = target_count;
  const float* dynamic_cost_arg = workspace.dynamic_vertex_cost;
  BellmanFord11BoundingBox bounds_arg = options.bounds;
  int max_iters_arg = max_iters;
  int check_interval_arg = options.target_check_interval;
  unsigned long long* state_arg = workspace.best_state;
  Index* frontier_arg = workspace.frontier;
  Index* next_arg = workspace.next_frontier;
  int* marks_arg = workspace.next_marks;
  unsigned char* source_mask_arg = workspace.source_mask;
  IterationStatus* iteration_arg = workspace.iteration_status;
  ControllerResult* result_arg = workspace.controller_result;
  void* args[] = {&graph_arg,
                  &sources_arg,
                  &source_count_arg,
                  &targets_arg,
                  &target_count_arg,
                  &dynamic_cost_arg,
                  &bounds_arg,
                  &max_iters_arg,
                  &check_interval_arg,
                  &state_arg,
                  &frontier_arg,
                  &next_arg,
                  &marks_arg,
                  &source_mask_arg,
                  &iteration_arg,
                  &result_arg};
  check_hip(hipLaunchCooperativeKernel(
                frontier_controller_kernel,
                dim3(static_cast<unsigned>(blocks)), dim3(kBlockSize), args, 0,
                workspace.stream),
            "launch BF11 cooperative controller");
  g_bf11_gpu_controller_launches.fetch_add(1, std::memory_order_relaxed);
  check_hip(hipMemcpyAsync(workspace.host_controller_result,
                           workspace.controller_result,
                           sizeof(ControllerResult), hipMemcpyDeviceToHost,
                           workspace.stream),
            "copy BF11 controller result");
  check_hip(hipStreamSynchronize(workspace.stream),
            "synchronize BF11 controller result");
  const ControllerResult result = *workspace.host_controller_result;
  g_bf11_target_checks.fetch_add(
      static_cast<std::uint64_t>(std::max(0, result.target_checks)),
      std::memory_order_relaxed);
  if (result.error_status != 0) throw_controller_error(result.error_status);
  const int termination_count =
      result.converged + result.early_stopped + result.hit_max_iters;
  if (result.done != 1 || result.iterations_used < 0 ||
      result.iterations_used > max_iters || result.frontier_count < 0 ||
      static_cast<Offset>(result.frontier_count) > graph.rows ||
      result.current_frontier_index < 0 || result.current_frontier_index > 1 ||
      result.target_checks < 0 || termination_count != 1) {
    throw std::runtime_error("BF11 cooperative controller returned bad status");
  }
  return {result.iterations_used, result.converged != 0,
          result.early_stopped != 0, result.hit_max_iters != 0};
}

SsspStatus run_host_controller(const DeviceGraph& graph,
                               DeviceWorkspace& workspace,
                               Offset source_count,
                               Offset target_count,
                               int max_iters,
                               const BellmanFord11RunOptions& options) {
  PATHFINDER_PROFILE_RANGE("bf11.controller_fallback");
  g_bf11_controller_fallbacks.fetch_add(1, std::memory_order_relaxed);
  hipLaunchKernelGGL(clear_state_kernel, grid_for_items(graph.rows),
                     dim3(kBlockSize), 0, workspace.stream, graph.rows,
                     workspace.best_state, workspace.next_marks,
                     workspace.source_mask);
  check_hip(hipGetLastError(), "clear BF11 state");
  hipLaunchKernelGGL(seed_sources_kernel, grid_for_items(source_count),
                     dim3(kBlockSize), 0, workspace.stream,
                     workspace.source_nodes, source_count,
                     workspace.best_state, workspace.frontier,
                     workspace.source_mask);
  check_hip(hipGetLastError(), "seed BF11 sources");

  int frontier_count = static_cast<int>(source_count);
  SsspStatus result;
  for (int iteration = 0; iteration < max_iters && frontier_count > 0;
       ++iteration) {
    hipLaunchKernelGGL(reset_iteration_status_kernel, dim3(1), dim3(1), 0,
                       workspace.stream, workspace.iteration_status);
    check_hip(hipGetLastError(), "reset BF11 iteration status");
    hipLaunchKernelGGL(frontier_relax_kernel, grid_for_items(frontier_count),
                       dim3(kBlockSize), 0, workspace.stream, graph,
                       workspace.frontier, frontier_count, iteration + 1,
                       workspace.dynamic_vertex_cost, options.bounds,
                       workspace.best_state, workspace.next_frontier,
                       workspace.next_marks, workspace.source_mask,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch BF11 frontier relaxation");
    const bool check_targets =
        (iteration + 1) % options.target_check_interval == 0;
    if (check_targets) {
      hipLaunchKernelGGL(update_target_status_kernel,
                         grid_for_items(target_count), dim3(kBlockSize), 0,
                         workspace.stream, workspace.best_state,
                         workspace.target_nodes, target_count,
                         workspace.iteration_status);
      check_hip(hipGetLastError(), "launch BF11 target check");
      g_bf11_target_checks.fetch_add(1, std::memory_order_relaxed);
    }
    check_hip(hipMemcpyAsync(workspace.host_iteration_status,
                             workspace.iteration_status,
                             sizeof(IterationStatus), hipMemcpyDeviceToHost,
                             workspace.stream),
              "copy BF11 iteration status");
    check_hip(hipStreamSynchronize(workspace.stream),
              "synchronize BF11 iteration status");
    const IterationStatus status = *workspace.host_iteration_status;
    if (status.error_status != 0) throw_controller_error(status.error_status);
    result.iterations_used = iteration + 1;
    frontier_count = status.next_count;
    if (frontier_count == 0) {
      result.converged = true;
      break;
    }
    if (check_targets &&
        status.reached_target_count == static_cast<int>(target_count) &&
        status.min_next_frontier_dist_bits >= status.max_target_dist_bits) {
      result.early_stopped = true;
      break;
    }
    std::swap(workspace.frontier, workspace.next_frontier);
  }
  result.hit_max_iters = !result.converged && !result.early_stopped;
  return result;
}

SsspStatus run_sssp(const DeviceGraph& graph,
                    DeviceWorkspace& workspace,
                    Offset source_count,
                    Offset target_count,
                    int max_iters,
                    const BellmanFord11RunOptions& options) {
  if (max_iters < 0) max_iters = static_cast<int>(graph.rows) - 1;
  const int blocks = cooperative_block_count(workspace);
  return blocks > 0
             ? run_gpu_controller(graph, workspace, source_count, target_count,
                                  max_iters, options, blocks)
             : run_host_controller(graph, workspace, source_count, target_count,
                                   max_iters, options);
}

std::vector<int> deduplicate_nodes(const std::vector<int>& input,
                                   Offset rows,
                                   const char* kind) {
  if (input.empty()) {
    throw std::invalid_argument(std::string("BF11 requires at least one ") + kind);
  }
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string("BF11 ") + kind +
                              " count does not fit int");
  }
  std::vector<int> result;
  result.reserve(input.size());
  std::unordered_set<int> seen;
  seen.reserve(input.size());
  for (const int node : input) {
    if (node < 0 || static_cast<Offset>(node) >= rows) {
      throw std::out_of_range(std::string("BF11 ") + kind +
                              " is outside the CSR graph");
    }
    if (seen.insert(node).second) result.push_back(node);
  }
  return result;
}

bool known_coordinate(const HostSidecarView& sidecars, int node) {
  return (*sidecars.route_end_x)[static_cast<std::size_t>(node)] !=
             kMissingCoordinate &&
         (*sidecars.route_end_y)[static_cast<std::size_t>(node)] !=
             kMissingCoordinate;
}

bool in_bounds(const HostSidecarView& sidecars,
               int node,
               const BellmanFord11BoundingBox& bounds) {
  if (!bounds.enabled) return true;
  if (!known_coordinate(sidecars, node)) return false;
  const std::int32_t x =
      (*sidecars.route_end_x)[static_cast<std::size_t>(node)];
  const std::int32_t y =
      (*sidecars.route_end_y)[static_cast<std::size_t>(node)];
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

void validate_run_options(const BellmanFord11RunOptions& options) {
  if (options.target_check_interval <= 0) {
    throw std::invalid_argument("BF11 target-check interval must be positive");
  }
  if (options.bounds.enabled &&
      (options.bounds.min_x > options.bounds.max_x ||
       options.bounds.min_y > options.bounds.max_y)) {
    throw std::invalid_argument("BF11 bounding box is inverted");
  }
}

void validate_terminal_bounds(const HostSidecarView& sidecars,
                              const std::vector<int>& sources,
                              const std::vector<int>& targets,
                              const BellmanFord11BoundingBox& bounds,
                              bool allow_missing_sources) {
  if (!bounds.enabled) return;
  for (const int source : sources) {
    if (allow_missing_sources && !known_coordinate(sidecars, source)) {
      continue;
    }
    if (!in_bounds(sidecars, source, bounds)) {
      throw std::invalid_argument(
          "BF11 bounded source is missing coordinates or outside the box");
    }
  }
  for (const int target : targets) {
    if (!in_bounds(sidecars, target, bounds)) {
      throw std::invalid_argument(
          "BF11 bounded target is missing coordinates or outside the box");
    }
  }
}

std::int32_t saturating_margin(std::int32_t value,
                               std::int32_t margin,
                               bool subtract) {
  const std::int64_t widened =
      subtract ? static_cast<std::int64_t>(value) - margin
               : static_cast<std::int64_t>(value) + margin;
  return static_cast<std::int32_t>(std::max<std::int64_t>(
      std::numeric_limits<std::int32_t>::min(),
      std::min<std::int64_t>(std::numeric_limits<std::int32_t>::max(),
                             widened)));
}

bool make_auto_bounds(const HostSidecarView& sidecars,
                      const std::vector<int>& sources,
                      const std::vector<int>& targets,
                      const BellmanFord11WorkspaceOptions& options,
                      BellmanFord11BoundingBox* output) {
  std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_x = std::numeric_limits<std::int32_t>::min();
  std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_y = std::numeric_limits<std::int32_t>::min();
  auto include = [&](int node) {
    if (!known_coordinate(sidecars, node)) return false;
    const std::int32_t x =
        (*sidecars.route_end_x)[static_cast<std::size_t>(node)];
    const std::int32_t y =
        (*sidecars.route_end_y)[static_cast<std::size_t>(node)];
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    return true;
  };
  for (const int source : sources) (void)include(source);
  for (const int target : targets) {
    if (!include(target)) return false;
  }
  *output = {true,
             saturating_margin(min_x, options.auto_margin_x, true),
             saturating_margin(max_x, options.auto_margin_x, false),
             saturating_margin(min_y, options.auto_margin_y, true),
             saturating_margin(max_y, options.auto_margin_y, false)};
  return true;
}

int saturated_add(int left, int right) {
  if (right > 0 && left > std::numeric_limits<int>::max() - right) {
    return std::numeric_limits<int>::max();
  }
  return left + right;
}

}  // namespace rips_sssp_bf11

struct BellmanFord11CsrGraph::Impl {
  minplus_sparse::Offset rows = 0;
  minplus_sparse::Offset nnz = 0;
  // Auto-bound queries need random host access after construction. Own only
  // the two compact coordinate columns; base costs and spatial shards are
  // upload-only and are deliberately not duplicated here.
  std::vector<std::int32_t> host_route_end_x;
  std::vector<std::int32_t> host_route_end_y;
  rips_sssp_bf11::HostSidecarView host_sidecars;
  rips_sssp_bf11::DeviceGraphOwner device;
  int hip_device = -1;

  Impl(const HostCsrF32& adjacency,
       rips_sssp_bf11::HostSidecarView sidecars,
       hipStream_t stream)
      : rows(adjacency.rows),
        nnz(adjacency.nnz),
        host_route_end_x(*sidecars.route_end_x),
        host_route_end_y(*sidecars.route_end_y),
        host_sidecars{&host_route_end_x, &host_route_end_y, nullptr} {
    rips_sssp_bf11::validate_csr(adjacency);
    rips_sssp_bf11::validate_sidecar_view(sidecars, rows);
    rips_sssp_bf11::check_hip(hipGetDevice(&hip_device),
                              "get BF11 graph HIP device");
    PATHFINDER_PROFILE_RANGE("bf11.upload_graph");
    device = rips_sssp_bf11::copy_graph_to_device(adjacency, sidecars, stream);
  }

  ~Impl() {
    rips_sssp_bf11::ScopedOwningHipDevice owner_device(hip_device);
    if (owner_device.active()) rips_sssp_bf11::free_graph(&device);
  }
};

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency,
          rips_sssp_bf11::common_sidecar_view(sidecars, adjacency), stream)) {}

BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream)
    : impl_(std::make_shared<Impl>(
          adjacency,
          rips_sssp_bf11::compatibility_sidecar_view(sidecars, adjacency),
          stream)) {}

BellmanFord11CsrGraph::~BellmanFord11CsrGraph() = default;
BellmanFord11CsrGraph::BellmanFord11CsrGraph(
    BellmanFord11CsrGraph&&) noexcept = default;
BellmanFord11CsrGraph& BellmanFord11CsrGraph::operator=(
    BellmanFord11CsrGraph&&) noexcept = default;

struct BellmanFord11CsrWorkspace::Impl {
  std::shared_ptr<const BellmanFord11CsrGraph::Impl> graph;
  rips_sssp_bf11::DeviceWorkspace workspace;
  hipStream_t stream = nullptr;
  BellmanFord11WorkspaceOptions options;
  // Serializes a complete cost epoch (update or run). In particular, a caller
  // cannot change destination multipliers between rounds of the capability
  // fallback controller.
  std::mutex operation_mutex;

  static std::shared_ptr<const BellmanFord11CsrGraph::Impl> require_graph(
      const std::shared_ptr<const BellmanFord11CsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("BF11 shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const BellmanFord11CsrGraph::Impl> graph_in,
       hipStream_t stream_in,
       BellmanFord11WorkspaceOptions options_in)
      : graph(std::move(graph_in)), stream(stream_in), options(options_in) {
    if (options.auto_margin_x < 0 || options.auto_margin_y < 0 ||
        options.target_check_interval <= 0) {
      throw std::invalid_argument(
          "BF11 auto-bound margins and target interval must be nonnegative/positive");
    }
    int current_device = -1;
    rips_sssp_bf11::check_hip(hipGetDevice(&current_device),
                              "get BF11 workspace HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument("BF11 graph belongs to another HIP device");
    }
    workspace = rips_sssp_bf11::make_workspace(graph->rows, stream);
  }

  ~Impl() {
    rips_sssp_bf11::ScopedOwningHipDevice owner_device(graph->hip_device);
    if (owner_device.active()) rips_sssp_bf11::free_workspace(&workspace);
  }

  void require_stream(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "BF11 workspace is stream-affine; use its construction stream");
    }
    int current_device = -1;
    rips_sssp_bf11::check_hip(hipGetDevice(&current_device),
                              "get BF11 run HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument("BF11 workspace is on another HIP device");
    }
  }

  BellmanFordCsrResult run_once(const std::vector<int>& sources,
                                const std::vector<int>& targets,
                                int max_iters,
                                const BellmanFord11RunOptions& run_options,
                                bool allow_missing_bounded_sources = false) {
    using namespace rips_sssp_bf11;
    validate_run_options(run_options);
    const std::vector<int> unique_sources =
        deduplicate_nodes(sources, graph->rows, "source");
    const std::vector<int> unique_targets =
        deduplicate_nodes(targets, graph->rows, "target");
    validate_terminal_bounds(graph->host_sidecars, unique_sources,
                             unique_targets, run_options.bounds,
                             allow_missing_bounded_sources);

    const Offset source_count = static_cast<Offset>(unique_sources.size());
    const Offset target_count = static_cast<Offset>(unique_targets.size());
    ensure_source_capacity(workspace, source_count);
    ensure_target_capacity(workspace, target_count);
    ensure_reconstruction_capacity(workspace, target_count);
    DrainStreamOnException drain(stream);
    check_hip(hipMemcpyAsync(workspace.source_nodes, unique_sources.data(),
                             unique_sources.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 sources");
    check_hip(hipMemcpyAsync(workspace.target_nodes, unique_targets.data(),
                             unique_targets.size() * sizeof(Index),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 targets");

    SsspStatus status = run_sssp(graph->device.view, workspace,
                                 source_count, target_count, max_iters,
                                 run_options);
    if (status.iterations_used == 0 &&
        std::all_of(unique_targets.begin(), unique_targets.end(),
                    [&](int target) {
                      return std::find(unique_sources.begin(),
                                       unique_sources.end(),
                                       target) != unique_sources.end();
                    })) {
      // Source labels are exact before the first relaxation. Preserve useful
      // max_iters=0 identity queries as certified target stops rather than
      // exposing them as finite-but-tentative results to PathFinder.
      status.early_stopped = true;
      status.hit_max_iters = false;
    }

    hipLaunchKernelGGL(summarize_target_paths_kernel,
                       grid_for_items(target_count), dim3(kBlockSize), 0, stream,
                       workspace.best_state, workspace.source_mask, graph->rows,
                       graph->nnz, graph->device.view.to,
                       graph->device.view.rowptr, workspace.target_nodes,
                       target_count, workspace.target_summaries);
    check_hip(hipGetLastError(), "summarize BF11 target paths");
    check_hip(hipMemcpyAsync(workspace.host_target_summaries,
                             workspace.target_summaries,
                             unique_targets.size() * sizeof(TargetSummary),
                             hipMemcpyDeviceToHost, stream),
              "copy BF11 target summaries");
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF11 target summaries");

    std::vector<unsigned long long> node_offsets(unique_targets.size(), 0);
    std::vector<unsigned long long> edge_offsets(unique_targets.size(), 0);
    unsigned long long total_nodes = 0;
    unsigned long long total_edges = 0;
    for (std::size_t i = 0; i < unique_targets.size(); ++i) {
      const TargetSummary& summary = workspace.host_target_summaries[i];
      node_offsets[i] = total_nodes;
      edge_offsets[i] = total_edges;
      if (summary.status == kTargetPathInvalid) {
        throw std::runtime_error(
            "BF11 produced an invalid or cyclic predecessor chain");
      }
      if (summary.status != kTargetPathValid) continue;
      if (summary.node_count == 0 ||
          summary.node_count != summary.edge_count + 1 ||
          summary.node_count > ~total_nodes ||
          summary.edge_count > ~total_edges) {
        throw std::overflow_error("BF11 compact target paths overflow uint64");
      }
      total_nodes += summary.node_count;
      total_edges += summary.edge_count;
    }
    if (total_nodes > std::numeric_limits<std::size_t>::max() ||
        total_edges > std::numeric_limits<std::size_t>::max() ||
        total_nodes >
            static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        total_edges >
            static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "BF11 compact paths exceed host/result offset capacity");
    }
    const std::size_t compact_node_count =
        static_cast<std::size_t>(total_nodes);
    const std::size_t compact_edge_count =
        static_cast<std::size_t>(total_edges);
    ensure_compact_capacity(workspace, compact_node_count, compact_edge_count);
    check_hip(hipMemcpyAsync(workspace.reconstruction_node_offsets,
                             node_offsets.data(),
                             node_offsets.size() * sizeof(unsigned long long),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 path node offsets");
    check_hip(hipMemcpyAsync(workspace.reconstruction_edge_offsets,
                             edge_offsets.data(),
                             edge_offsets.size() * sizeof(unsigned long long),
                             hipMemcpyHostToDevice, stream),
              "copy BF11 path edge offsets");
    hipLaunchKernelGGL(
        materialize_target_paths_kernel, grid_for_items(target_count),
        dim3(kBlockSize), 0, stream, workspace.best_state,
        workspace.source_mask, graph->rows, graph->nnz,
        graph->device.view.to, graph->device.view.rowptr,
        workspace.target_nodes, workspace.target_summaries,
        workspace.reconstruction_node_offsets,
        workspace.reconstruction_edge_offsets, target_count,
        workspace.compact_nodes, workspace.compact_edges,
        graph->device.view.edge_values,
        graph->device.view.base_vertex_cost,
        workspace.dynamic_vertex_cost,
        workspace.compact_edge_costs);
    check_hip(hipGetLastError(), "materialize BF11 target paths");
    std::vector<Index> compact_nodes(compact_node_count);
    std::vector<DeviceOffset> compact_edges(compact_edge_count);
    std::vector<float> compact_edge_costs(compact_edge_count);
    if (!compact_nodes.empty()) {
      check_hip(hipMemcpyAsync(compact_nodes.data(), workspace.compact_nodes,
                               compact_nodes.size() * sizeof(Index),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact nodes");
    }
    if (!compact_edges.empty()) {
      check_hip(hipMemcpyAsync(compact_edges.data(), workspace.compact_edges,
                               compact_edges.size() * sizeof(DeviceOffset),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edges");
      check_hip(hipMemcpyAsync(compact_edge_costs.data(),
                               workspace.compact_edge_costs,
                               compact_edge_costs.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "copy BF11 compact edge costs");
    }
    check_hip(hipStreamSynchronize(stream),
              "synchronize BF11 compact target paths");

    std::unordered_map<int, std::size_t> target_index;
    target_index.reserve(unique_targets.size() * 2 + 1);
    for (std::size_t i = 0; i < unique_targets.size(); ++i) {
      target_index.emplace(unique_targets[i], i);
    }
    BellmanFordCsrResult result;
    result.target = -1;
    result.iterations_used = status.iterations_used;
    result.converged = status.converged;
    result.stopped_on_target = status.early_stopped;
    result.target_reached = true;
    result.target_distances.reserve(targets.size());
    result.target_sources.reserve(targets.size());
    result.target_path_offsets.reserve(targets.size() + 1);
    result.target_edge_offsets.reserve(targets.size() + 1);
    result.target_path_offsets.push_back(0);
    result.target_edge_offsets.push_back(0);
    for (const int target : targets) {
      const std::size_t unique = target_index.at(target);
      const TargetSummary& summary = workspace.host_target_summaries[unique];
      if (summary.status == kTargetUnreachable) {
        result.target_distances.push_back(
            std::numeric_limits<float>::infinity());
        result.target_sources.push_back(-1);
        result.target_reached = false;
      } else {
        const std::size_t node_begin =
            static_cast<std::size_t>(node_offsets[unique]);
        const std::size_t edge_begin =
            static_cast<std::size_t>(edge_offsets[unique]);
        const std::size_t node_end =
            node_begin + static_cast<std::size_t>(summary.node_count);
        const std::size_t edge_end =
            edge_begin + static_cast<std::size_t>(summary.edge_count);
        if (node_end > compact_nodes.size() || edge_end > compact_edges.size() ||
            compact_nodes[node_begin] != summary.root ||
            compact_nodes[node_end - 1] != target) {
          throw std::runtime_error("BF11 compact target path is malformed");
        }
        result.target_distances.push_back(host_state_distance(summary.state));
        result.target_sources.push_back(summary.root);
        result.target_path_nodes.insert(result.target_path_nodes.end(),
                                        compact_nodes.begin() + node_begin,
                                        compact_nodes.begin() + node_end);
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          result.target_path_edges.push_back(
              static_cast<minplus_sparse::Offset>(compact_edges[edge]));
          result.target_path_edge_costs.push_back(compact_edge_costs[edge]);
        }
      }
      if (result.target_path_nodes.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          result.target_path_edges.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BF11 result path offsets exceed int");
      }
      result.target_path_offsets.push_back(
          static_cast<int>(result.target_path_nodes.size()));
      result.target_edge_offsets.push_back(
          static_cast<int>(result.target_path_edges.size()));
    }
    return result;
  }
};

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const routing::interchange::RoutingCsrSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::common_sidecar_view(sidecars, adjacency), stream),
          stream, options)) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    const HostCsrF32& adjacency,
    const BellmanFord11NodeSidecars& sidecars,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(
          std::make_shared<BellmanFord11CsrGraph::Impl>(
              adjacency,
              rips_sssp_bf11::compatibility_sidecar_view(sidecars, adjacency),
              stream),
          stream, options)) {}

BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
    hipStream_t stream,
    BellmanFord11WorkspaceOptions options)
    : impl_(std::make_unique<Impl>(Impl::require_graph(adjacency), stream,
                                   options)) {}

BellmanFord11CsrWorkspace::~BellmanFord11CsrWorkspace() = default;
BellmanFord11CsrWorkspace::BellmanFord11CsrWorkspace(
    BellmanFord11CsrWorkspace&&) noexcept = default;
BellmanFord11CsrWorkspace& BellmanFord11CsrWorkspace::operator=(
    BellmanFord11CsrWorkspace&&) noexcept = default;

void BellmanFord11CsrWorkspace::update_vertex_costs(
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("bf11.update_vertex_costs");
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (vertex_costs.size() != static_cast<std::size_t>(impl_->graph->rows)) {
    throw std::invalid_argument("BF11 vertex-cost vector must contain V entries");
  }
  for (const float cost : vertex_costs) {
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::invalid_argument(
          "BF11 dynamic vertex costs must be finite and nonnegative");
    }
  }
  rips_sssp_bf11::DrainStreamOnException drain(stream);
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.dynamic_vertex_cost,
                     vertex_costs.data(), vertex_costs.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream),
      "copy BF11 dynamic vertex costs");
  rips_sssp_bf11::check_hip(hipStreamSynchronize(stream),
                            "synchronize BF11 dynamic vertex costs");
}

void BellmanFord11CsrWorkspace::update_vertex_costs_sparse(
    const std::vector<int>& nodes,
    const std::vector<float>& vertex_costs,
    hipStream_t stream) {
  PATHFINDER_PROFILE_RANGE("bf11.update_vertex_costs_sparse");
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (nodes.size() != vertex_costs.size()) {
    throw std::invalid_argument("BF11 sparse cost nodes and values differ in size");
  }
  if (nodes.empty()) return;
  std::unordered_set<int> seen;
  seen.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i] < 0 ||
        static_cast<minplus_sparse::Offset>(nodes[i]) >= impl_->graph->rows) {
      throw std::out_of_range("BF11 sparse cost node is outside the graph");
    }
    if (!seen.insert(nodes[i]).second) {
      throw std::invalid_argument("BF11 sparse cost nodes must be unique");
    }
    if (!std::isfinite(vertex_costs[i]) || vertex_costs[i] < 0.0f) {
      throw std::invalid_argument(
          "BF11 dynamic vertex costs must be finite and nonnegative");
    }
  }
  if (nodes.size() > static_cast<std::size_t>(impl_->graph->rows)) {
    throw std::invalid_argument("BF11 sparse cost update exceeds V entries");
  }
  const rips_sssp_bf11::Offset count =
      static_cast<rips_sssp_bf11::Offset>(nodes.size());
  rips_sssp_bf11::ensure_update_capacity(impl_->workspace, count);
  rips_sssp_bf11::DrainStreamOnException drain(stream);
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.update_nodes, nodes.data(),
                     nodes.size() * sizeof(int), hipMemcpyHostToDevice, stream),
      "copy BF11 sparse cost nodes");
  rips_sssp_bf11::check_hip(
      hipMemcpyAsync(impl_->workspace.update_costs, vertex_costs.data(),
                     vertex_costs.size() * sizeof(float), hipMemcpyHostToDevice,
                     stream),
      "copy BF11 sparse cost values");
  hipLaunchKernelGGL(rips_sssp_bf11::sparse_cost_update_kernel,
                     rips_sssp_bf11::grid_for_items(count),
                     dim3(rips_sssp_bf11::kBlockSize), 0, stream,
                     impl_->workspace.update_nodes,
                     impl_->workspace.update_costs, count,
                     impl_->workspace.dynamic_vertex_cost);
  rips_sssp_bf11::check_hip(hipGetLastError(),
                            "apply BF11 sparse vertex costs");
  rips_sssp_bf11::check_hip(hipStreamSynchronize(stream),
                            "synchronize BF11 sparse vertex costs");
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("bf11.run");
  (void)delta;
  (void)progress_user_data;
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (progress_callback) {
    throw std::invalid_argument(
        "BF11 persistent controller does not expose per-round callbacks");
  }
  const std::vector<int> unique_sources = rips_sssp_bf11::deduplicate_nodes(
      sources, impl_->graph->rows, "source");
  const std::vector<int> unique_targets = rips_sssp_bf11::deduplicate_nodes(
      targets, impl_->graph->rows, "target");

  BellmanFord11RunOptions run_options;
  run_options.target_check_interval = impl_->options.target_check_interval;
  if (!impl_->options.auto_bounds) {
    return impl_->run_once(sources, targets, max_iters, run_options);
  }

  const bool have_terminal_coordinates = rips_sssp_bf11::make_auto_bounds(
      impl_->graph->host_sidecars, unique_sources, unique_targets,
      impl_->options, &run_options.bounds);
  if (!have_terminal_coordinates) {
    if (!impl_->options.unbounded_fallback) {
      throw std::invalid_argument(
          "BF11 cannot auto-bound a target without coordinates");
    }
    run_options.bounds = {};
    return impl_->run_once(sources, targets, max_iters, run_options);
  }

  BellmanFordCsrResult bounded =
      impl_->run_once(sources, targets, max_iters, run_options, true);
  if (bounded.target_reached || !impl_->options.unbounded_fallback) {
    return bounded;
  }
  g_bf11_auto_unbounded_retries.fetch_add(1, std::memory_order_relaxed);
  run_options.bounds = {};
  BellmanFordCsrResult unbounded =
      impl_->run_once(sources, targets, max_iters, run_options);
  unbounded.iterations_used = rips_sssp_bf11::saturated_add(
      bounded.iterations_used, unbounded.iterations_used);
  return unbounded;
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    const BellmanFord11RunOptions& run_options,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("bf11.run_explicit");
  (void)delta;
  (void)progress_user_data;
  if (!impl_) throw std::runtime_error("BF11 workspace has no implementation");
  std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
  impl_->require_stream(stream);
  if (progress_callback) {
    throw std::invalid_argument(
        "BF11 persistent controller does not expose per-round callbacks");
  }
  return impl_->run_once(sources, targets, max_iters, run_options);
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  BellmanFordCsrResult result =
      run(sources, std::vector<int>{target}, delta, max_iters, stream,
          progress_callback, progress_user_data);
  result.target = target;
  result.target_distance = result.target_distances.front();
  result.target_reached = std::isfinite(result.target_distance);
  return result;
}

BellmanFordCsrResult BellmanFord11CsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source}, target, delta, max_iters, stream,
             progress_callback, progress_user_data);
}

#ifndef BF11_NO_MAIN
int main(int, char**) {
  std::cerr
      << "bf11 is a PathFinder library backend; build it with BF11_NO_MAIN "
         "and construct BellmanFord11CsrWorkspace with routing sidecars.\n";
  return 2;
}
#endif
