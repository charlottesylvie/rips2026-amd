// Outgoing-CSR Bellman-Ford driver using a local active-frontier relaxation
// with optional validator-compatible source-to-sink path JSONL.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     CongestionFreeRouting/bellman_ford/bf9.cpp \
//     -o bf9
//
// Run summary-only; JSONL path output is disabled by default:
//   ./bf9 design.csrbin \
//     design.csrbin.ifmeta.bin \
//     bf9_summary.csv \
//     --source-progress-every 100
//
// Add --paths-output bf9.paths.jsonl
// to emit validator-compatible source-to-sink path JSONL.
// Add --print-frontiers to print the active frontier vertex ids for detailed
// source progress lines.
// Add --print-jsonl-entry-ids to print the corresponding JSONL path entry ids for
// detailed source progress lines.

#include "bf9.hpp"
#include "../profiling/roctx_ranges.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Internal regression counters intentionally stay out of the public BF9
// interface. They cover exceptional full-state traffic and infrequent scratch
// growth, and are exposed by C-linkage test hooks below so production routing
// results remain untouched.
std::atomic<std::uint64_t> g_bf9_full_state_copy_count{0};
std::atomic<std::uint64_t> g_bf9_full_state_fallback_count{0};
std::atomic<std::uint64_t> g_bf9_target_buffer_growth_count{0};
std::atomic<std::uint64_t> g_bf9_path_buffer_growth_count{0};

}  // namespace

extern "C" void bf9_internal_reset_full_state_counters() {
  g_bf9_full_state_copy_count.store(0, std::memory_order_relaxed);
  g_bf9_full_state_fallback_count.store(0, std::memory_order_relaxed);
}

extern "C" std::uint64_t bf9_internal_full_state_copy_count() {
  return g_bf9_full_state_copy_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf9_internal_full_state_fallback_count() {
  return g_bf9_full_state_fallback_count.load(std::memory_order_relaxed);
}

extern "C" void bf9_internal_reset_buffer_growth_counters() {
  g_bf9_target_buffer_growth_count.store(0, std::memory_order_relaxed);
  g_bf9_path_buffer_growth_count.store(0, std::memory_order_relaxed);
}

extern "C" std::uint64_t bf9_internal_target_buffer_growth_count() {
  return g_bf9_target_buffer_growth_count.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t bf9_internal_path_buffer_growth_count() {
  return g_bf9_path_buffer_growth_count.load(std::memory_order_relaxed);
}

namespace rips_sssp_new {

using Offset = std::int64_t;
using Index = int;
using DeviceOffset = std::uint32_t;

constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;
constexpr unsigned int kPackedInfinityBits = 0x7f800000u;
constexpr unsigned kGridX = 65535u;
constexpr int kBlockSize = 256;

struct DeviceOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;

  // Outgoing CSR orientation: row u stores directed edges u -> v.
  // bf9 stores row offsets in 32 bits because packed predecessor edge IDs
  // already constrain nnz below UINT32_MAX. Degree and edge-id arrays are
  // derived from adjacent row offsets and the CSR position in the kernel.
  const DeviceOffset* rowptr = nullptr;
  const Index* to = nullptr;
  const float* values = nullptr;
  bool unit_weights = false;
};

struct IterationStatus {
  int next_count = 0;
  int error_status = 0;
  int reached_target_count = 0;
  unsigned int min_next_frontier_dist_bits = kPackedInfinityBits;
  unsigned int max_target_dist_bits = 0;
};

struct DeviceSsspWorkspace {
  Offset rows = 0;
  Offset gather_capacity = 0;
  hipStream_t stream = nullptr;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  int* next_marks = nullptr;
  IterationStatus* iteration_status = nullptr;
  IterationStatus* host_iteration_status = nullptr;
  hipEvent_t status_ready_event = nullptr;
  Index* gather_nodes = nullptr;
  unsigned long long* gather_states = nullptr;
};

struct OutgoingSsspResult {
  int iterations_used = 0;
  bool converged = false;
  bool early_stopped = false;
  bool hit_max_iters = false;
};

// CPU helper. Inputs: HIP status and operation label. Output: none, or throws.
// Purpose: convert HIP errors from host-side launches/copies into exceptions.
void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

// CPU helper. Inputs: item count and block size. Output: a 2D HIP grid.
// Purpose: keep launches scalable for large frontier or row counts.
dim3 grid_for_items(Offset items, int block_size) {
  if (items <= 0) return dim3(1, 1);
  const Offset blocks_needed = (items + block_size - 1) / block_size;
  const unsigned gx =
      blocks_needed < static_cast<Offset>(kGridX)
          ? static_cast<unsigned>(blocks_needed)
          : kGridX;
  const unsigned gy =
      static_cast<unsigned>((blocks_needed + static_cast<Offset>(gx) - 1) /
                            static_cast<Offset>(gx));
  return dim3(gx, gy);
}

// Hybrid CPU/GPU host helper. Inputs: graph view. Output: none, or throws.
// Purpose: validate the device-pointer graph shape before launching kernels.
void validate_device_outgoing_csr(const DeviceOutgoingCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR expects a nonempty square graph");
  }
  if (graph.rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for frontier node ids");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (!graph.rowptr) {
    throw std::runtime_error("outgoing CSR rowptr must not be null");
  }
  if (graph.nnz > 0 &&
      (!graph.to || (!graph.unit_weights && !graph.values))) {
    throw std::runtime_error(
        "outgoing CSR edge arrays must not be null when nnz > 0");
  }
}

// Hybrid CPU/GPU host helper. Inputs: row count, gather capacity, HIP stream.
// Output: allocated workspace whose pointers live on the GPU.
// Purpose: allocate packed state, active-frontier arrays, and optional gather scratch.
DeviceSsspWorkspace make_sssp_workspace(Offset rows,
                                        Offset gather_capacity,
                                        hipStream_t stream = nullptr) {
  if (rows <= 0) {
    throw std::runtime_error("SSSP workspace rows must be positive");
  }
  if (rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::runtime_error("SSSP workspace has too many rows for frontier node ids");
  }
  if (gather_capacity < 0 || gather_capacity > rows) {
    throw std::runtime_error("SSSP gather capacity is outside graph bounds");
  }

  DeviceSsspWorkspace workspace;
  workspace.rows = rows;
  workspace.gather_capacity = gather_capacity;
  workspace.stream = stream;
  try {
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.best_state),
                        static_cast<std::size_t>(rows) * sizeof(unsigned long long)),
              "hipMalloc SSSP best state");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.frontier),
                        static_cast<std::size_t>(rows) * sizeof(Index)),
              "hipMalloc SSSP frontier");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.next_frontier),
                        static_cast<std::size_t>(rows) * sizeof(Index)),
              "hipMalloc SSSP next frontier");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.next_marks),
                        static_cast<std::size_t>(rows) * sizeof(int)),
              "hipMalloc SSSP frontier marks");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.iteration_status),
                        sizeof(IterationStatus)),
              "hipMalloc SSSP frontier status");
    check_hip(hipHostMalloc(
                  reinterpret_cast<void**>(&workspace.host_iteration_status),
                  sizeof(IterationStatus),
                  hipHostMallocDefault),
              "hipHostMalloc SSSP frontier status");
    check_hip(hipEventCreateWithFlags(&workspace.status_ready_event,
                                      hipEventDisableTiming),
              "hipEventCreateWithFlags SSSP frontier status");

    if (gather_capacity > 0) {
      check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_nodes),
                          static_cast<std::size_t>(gather_capacity) * sizeof(Index)),
                "hipMalloc SSSP gather nodes");
      check_hip(
          hipMalloc(reinterpret_cast<void**>(&workspace.gather_states),
                    static_cast<std::size_t>(gather_capacity) *
                        sizeof(unsigned long long)),
          "hipMalloc SSSP gather states");
    }

    return workspace;
  } catch (...) {
    if (workspace.best_state) (void)hipFree(workspace.best_state);
    if (workspace.frontier) (void)hipFree(workspace.frontier);
    if (workspace.next_frontier) (void)hipFree(workspace.next_frontier);
    if (workspace.next_marks) (void)hipFree(workspace.next_marks);
    if (workspace.iteration_status) (void)hipFree(workspace.iteration_status);
    if (workspace.status_ready_event) {
      (void)hipEventDestroy(workspace.status_ready_event);
    }
    if (workspace.host_iteration_status) {
      (void)hipHostFree(workspace.host_iteration_status);
    }
    if (workspace.gather_nodes) (void)hipFree(workspace.gather_nodes);
    if (workspace.gather_states) (void)hipFree(workspace.gather_states);
    throw;
  }
}

// Hybrid CPU/GPU host helper. Inputs: workspace to release. Output: reset fields.
// Purpose: free allocations created by make_sssp_workspace.
void free_sssp_workspace(DeviceSsspWorkspace* workspace) {
  if (!workspace) return;
  // A failed launch/copy may unwind before its normal event wait. Keep queued
  // operations from outliving device or pinned workspace storage.
  (void)hipStreamSynchronize(workspace->stream);
  if (workspace->best_state) {
    (void)hipFree(workspace->best_state);
    workspace->best_state = nullptr;
  }
  if (workspace->frontier) {
    (void)hipFree(workspace->frontier);
    workspace->frontier = nullptr;
  }
  if (workspace->next_frontier) {
    (void)hipFree(workspace->next_frontier);
    workspace->next_frontier = nullptr;
  }
  if (workspace->next_marks) {
    (void)hipFree(workspace->next_marks);
    workspace->next_marks = nullptr;
  }
  if (workspace->iteration_status) {
    (void)hipFree(workspace->iteration_status);
    workspace->iteration_status = nullptr;
  }
  if (workspace->status_ready_event) {
    (void)hipEventDestroy(workspace->status_ready_event);
    workspace->status_ready_event = nullptr;
  }
  if (workspace->host_iteration_status) {
    (void)hipHostFree(workspace->host_iteration_status);
    workspace->host_iteration_status = nullptr;
  }
  if (workspace->gather_nodes) {
    (void)hipFree(workspace->gather_nodes);
    workspace->gather_nodes = nullptr;
  }
  if (workspace->gather_states) {
    (void)hipFree(workspace->gather_states);
    workspace->gather_states = nullptr;
  }
  workspace->rows = 0;
  workspace->gather_capacity = 0;
  workspace->stream = nullptr;
}

// GPU device helper. Inputs: HIP block/thread ids. Output: logical 1D thread id.
// Purpose: support 2D-grid linearization.
__device__ __forceinline__ Offset logical_thread_id_1d() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

// GPU device helper. Inputs: distance bits and predecessor edge id. Output: packed state.
// Purpose: preserve the existing sortable 64-bit SSSP state format.
__device__ __forceinline__ unsigned long long pack_state_bits(unsigned int dist_bits,
                                                              unsigned int edge) {
  return (static_cast<unsigned long long>(dist_bits) << 32) |
         static_cast<unsigned long long>(edge);
}

// GPU device helper. Inputs: float distance and predecessor edge id. Output: packed state.
// Purpose: pack a candidate relaxation state for atomic comparison.
__device__ __forceinline__ unsigned long long pack_state_float(float dist,
                                                               unsigned int edge) {
  return pack_state_bits(__float_as_uint(dist), edge);
}

// GPU device helper. Inputs: packed state. Output: unpacked float distance.
// Purpose: read the current best known distance for a vertex.
__device__ __forceinline__ float unpack_state_dist(unsigned long long state) {
  return __uint_as_float(static_cast<unsigned int>(state >> 32));
}

// GPU device helper. Inputs: address, candidate distance, predecessor edge id.
// Output: true if the packed state was improved.
// Purpose: atomically apply min-plus relaxation with predecessor tie-breaking.
__device__ __forceinline__ bool atomic_relax_state(unsigned long long* addr,
                                                   float value,
                                                   unsigned int edge) {
  const unsigned long long desired = pack_state_float(value, edge);
  unsigned long long old_state = *addr;

  while (desired < old_state) {
    const unsigned long long assumed = old_state;
    old_state = atomicCAS(addr, assumed, desired);
    if (old_state == assumed) {
      return true;
    }
  }

  return false;
}

// GPU device helper. Inputs: float. Output: whether it is finite.
// Purpose: match the existing device-side NaN and infinity guard.
__device__ __forceinline__ bool finite_device_float(float value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

// GPU device helper. Inputs: edge fields, source distance, frontier workspace.
// Output: improved distance bits, or infinity when the edge did not improve.
// Purpose: apply one outgoing edge, queue its destination once, and let the
// caller aggregate the frontier-distance minimum before issuing a global atomic.
__device__ __forceinline__ unsigned int relax_outgoing_edge(
    Index dst,
    float weight,
    DeviceOffset pred_edge,
    Offset rows,
    float from_dist,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    IterationStatus* __restrict__ iteration_status) {
  if (dst < 0 || static_cast<Offset>(dst) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return kPackedInfinityBits;
  }

  if (!finite_device_float(weight) || weight < 0.0f) {
    atomicExch(&iteration_status->error_status, 3);
    return kPackedInfinityBits;
  }

  const float candidate = from_dist + weight;
  if (!finite_device_float(candidate)) {
    return kPackedInfinityBits;
  }

  if (atomic_relax_state(&best_state[static_cast<Offset>(dst)],
                         candidate,
                         pred_edge)) {
    const int old_mark =
        atomicExch(&next_marks[static_cast<Offset>(dst)], mark_token);
    if (old_mark != mark_token) {
      const int slot = atomicAdd(&iteration_status->next_count, 1);
      if (slot < 0 || static_cast<Offset>(slot) >= rows) {
        atomicExch(&iteration_status->error_status, 4);
        return kPackedInfinityBits;
      }
      next_frontier[slot] = dst;
    }
    return __float_as_uint(candidate);
  }

  return kPackedInfinityBits;
}

// GPU kernel. Inputs: row count, source node, workspace arrays.
// Output: initializes best_state, frontier, marks, and status on the GPU.
// Purpose: start frontier Bellman-Ford from exactly the requested source.
__global__ void init_outgoing_sssp_kernel(
    Offset rows,
    int source,
    unsigned long long* best_state,
    Index* frontier,
    int* next_marks,
    IterationStatus* iteration_status) {
  const Offset row = logical_thread_id_1d();
  if (row < rows) {
    best_state[row] =
        row == static_cast<Offset>(source)
            ? pack_state_bits(0u, 0u)
            : pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
    next_marks[row] = 0;
  }
  if (row == 0) {
    frontier[0] = static_cast<Index>(source);
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
    iteration_status->reached_target_count = 0;
    iteration_status->min_next_frontier_dist_bits = kPackedInfinityBits;
    iteration_status->max_target_dist_bits = 0;
  }
}

// GPU kernel. Inputs: iteration status pointer. Output: zeroed per-iteration counters.
// Purpose: reset status before building the next frontier.
__global__ void reset_iteration_status_kernel(IterationStatus* iteration_status) {
  if (logical_thread_id_1d() == 0) {
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
    iteration_status->reached_target_count = 0;
    iteration_status->min_next_frontier_dist_bits = kPackedInfinityBits;
    iteration_status->max_target_dist_bits = 0;
  }
}

// GPU kernel. Inputs: target node list and current packed best_state.
// Output: reached target count and largest finite target distance in iteration_status.
// Purpose: support nonnegative distance-bound stopping without copying target states to CPU.
__global__ void update_target_status_kernel(
    const unsigned long long* __restrict__ best_state,
    Offset rows,
    const Index* __restrict__ target_nodes,
    Offset target_count,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= target_count) {
    return;
  }

  const Index target = target_nodes[item];
  if (target < 0 || static_cast<Offset>(target) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  const unsigned long long state = best_state[static_cast<Offset>(target)];
  const unsigned int dist_bits = static_cast<unsigned int>(state >> 32);
  if (dist_bits != kPackedInfinityBits) {
    atomicAdd(&iteration_status->reached_target_count, 1);
    atomicMax(&iteration_status->max_target_dist_bits, dist_bits);
  }
}

// GPU kernel. Inputs: packed state array and a compact node-id list. Output:
// packed states in the same list order plus a malformed-node error flag.
// Purpose: transfer only target or active predecessor states requested by the
// host without changing their packed distance/predecessor representation.
__global__ void gather_packed_states_kernel(
    const unsigned long long* __restrict__ best_state,
    Offset rows,
    const Index* __restrict__ nodes,
    Offset node_count,
    unsigned long long* __restrict__ gathered_states,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= node_count) {
    return;
  }

  const Index node = nodes[item];
  if (node < 0 || static_cast<Offset>(node) >= rows) {
    gathered_states[item] =
        pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
    atomicExch(&iteration_status->error_status, 1);
    return;
  }
  gathered_states[item] = best_state[static_cast<Offset>(node)];
}

// GPU kernel. Inputs: outgoing CSR, current frontier, and packed best_state.
// Output: relaxed best_state and deduplicated next_frontier.
// Purpose: perform one active-frontier Bellman-Ford relaxation step.
template <bool UnitWeights>
__global__ void outgoing_frontier_relax_kernel(
    const DeviceOffset* __restrict__ rowptr,
    const Index* __restrict__ to,
    const float* __restrict__ values,
    Offset rows,
    Offset nnz,
    const Index* __restrict__ frontier,
    int frontier_count,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= static_cast<Offset>(frontier_count)) {
    return;
  }

  const Index from = frontier[item];
  if (from < 0 || static_cast<Offset>(from) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  const float from_dist = unpack_state_dist(best_state[static_cast<Offset>(from)]);
  if (!finite_device_float(from_dist)) {
    return;
  }

  const Offset begin =
      static_cast<Offset>(rowptr[static_cast<Offset>(from)]);
  const Offset end =
      static_cast<Offset>(rowptr[static_cast<Offset>(from) + 1]);
  if (end < begin || end > nnz) {
    atomicExch(&iteration_status->error_status, 2);
    return;
  }

  unsigned int local_min_dist_bits = kPackedInfinityBits;
  for (Offset edge = begin; edge < end; ++edge) {
    float weight = 1.0f;
    if constexpr (!UnitWeights) {
      weight = values[edge];
    }
    const unsigned int improved_dist_bits = relax_outgoing_edge(
        to[edge],
        weight,
        static_cast<DeviceOffset>(edge),
        rows,
        from_dist,
        mark_token,
        best_state,
        next_frontier,
        next_marks,
        iteration_status);
    local_min_dist_bits =
        improved_dist_bits < local_min_dist_bits
            ? improved_dist_bits
            : local_min_dist_bits;
  }

  if (local_min_dist_bits != kPackedInfinityBits) {
    // Project edge weights are nonnegative, so finite float bit ordering matches
    // numeric ordering for the frontier distance lower-bound test. One atomic
    // per active vertex replaces one atomic per successful outgoing edge.
    atomicMin(&iteration_status->min_next_frontier_dist_bits,
              local_min_dist_bits);
  }
}

// Hybrid CPU/GPU host wrapper. Inputs: workspace, source, stream. Output: initialized GPU state.
// Purpose: launch init_outgoing_sssp_kernel for a frontier Bellman-Ford source.
void initialize_outgoing_sssp_state(DeviceSsspWorkspace& workspace,
                                    int source,
                                    hipStream_t stream = nullptr) {
  if (workspace.rows <= 0 || !workspace.best_state || !workspace.frontier ||
      !workspace.next_frontier || !workspace.next_marks ||
      !workspace.iteration_status) {
    throw std::runtime_error("SSSP frontier workspace is not allocated");
  }
  if (source < 0 || static_cast<Offset>(source) >= workspace.rows) {
    throw std::runtime_error("source is outside workspace rows");
  }

  hipLaunchKernelGGL(init_outgoing_sssp_kernel,
                     grid_for_items(workspace.rows, kBlockSize),
                     dim3(kBlockSize),
                     0,
                     stream,
                     workspace.rows,
                     source,
                     workspace.best_state,
                     workspace.frontier,
                     workspace.next_marks,
                     workspace.iteration_status);
  check_hip(hipGetLastError(), "launch outgoing SSSP frontier initialization");
  // The first relaxation and any result copy use the same stream, so no host
  // decision depends on initialization completing here.
}

// Hybrid CPU/GPU host helper. Inputs: a queued status copy and workspace event.
// Output: the pinned status is safe for host access after the event completes.
// Purpose: wait only through the status copy instead of draining later work on
// the whole stream. The host dependency itself remains mandatory.
void wait_for_workspace_status(DeviceSsspWorkspace& workspace,
                               hipStream_t stream,
                               const char* record_what,
                               const char* wait_what) {
  if (!workspace.host_iteration_status || !workspace.status_ready_event) {
    throw std::runtime_error("SSSP pinned status workspace is not allocated");
  }
  if (workspace.stream != stream) {
    throw std::invalid_argument("SSSP workspace status event used on another stream");
  }
  check_hip(hipEventRecord(workspace.status_ready_event, stream), record_what);
  check_hip(hipEventSynchronize(workspace.status_ready_event), wait_what);
}

// Hybrid CPU/GPU host helper. Inputs: compact device node/state buffers, a
// pinned host result buffer, and the workspace stream. Output: packed states in
// node-list order. Purpose: share one validated gather path for persistent
// targets and batched predecessor-chain nodes.
void gather_packed_states_to_host(DeviceSsspWorkspace& workspace,
                                  const Index* device_nodes,
                                  unsigned long long* device_states,
                                  unsigned long long* host_states,
                                  Offset node_count,
                                  hipStream_t stream,
                                  const char* operation_name) {
  if (node_count < 0 || node_count > workspace.rows) {
    throw std::invalid_argument(std::string(operation_name) +
                                " count is outside graph bounds");
  }
  if (node_count == 0) {
    return;
  }
  if (!device_nodes || !device_states || !host_states) {
    throw std::runtime_error(std::string(operation_name) +
                             " buffers are not allocated");
  }
  if (workspace.stream != stream) {
    throw std::invalid_argument(std::string(operation_name) +
                                " used on another stream");
  }

  // The completed SSSP status is no longer needed. Clear it before the gather
  // so a malformed node can be reported without allocating another device
  // scalar or performing another synchronization.
  check_hip(hipMemsetAsync(workspace.iteration_status,
                           0,
                           sizeof(IterationStatus),
                           stream),
            "clear outgoing SSSP gather status");
  hipLaunchKernelGGL(gather_packed_states_kernel,
                     grid_for_items(node_count, kBlockSize),
                     dim3(kBlockSize),
                     0,
                     stream,
                     workspace.best_state,
                     workspace.rows,
                     device_nodes,
                     node_count,
                     device_states,
                     workspace.iteration_status);
  check_hip(hipGetLastError(), "launch outgoing SSSP packed-state gather");
  check_hip(hipMemcpyAsync(host_states,
                           device_states,
                           static_cast<std::size_t>(node_count) *
                               sizeof(unsigned long long),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy outgoing SSSP gathered packed states");
  check_hip(hipMemcpyAsync(workspace.host_iteration_status,
                           workspace.iteration_status,
                           sizeof(IterationStatus),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy outgoing SSSP gather status");
  wait_for_workspace_status(workspace,
                            stream,
                            "record outgoing SSSP gather event",
                            "wait for outgoing SSSP gather event");
  if (workspace.host_iteration_status->error_status != 0) {
    throw std::runtime_error(std::string(operation_name) +
                             " contains a node outside the outgoing CSR graph");
  }
}

// Hybrid CPU/GPU host helper. Inputs: workspace, current frontier count, and stream.
// Output: sorted CPU vector containing the current frontier vertex ids.
// Purpose: copy optional debug trace data without feeding it back into the algorithm.
std::vector<Index> copy_current_frontier_to_host(
    const DeviceSsspWorkspace& workspace,
    int frontier_count,
    hipStream_t stream = nullptr) {
  if (frontier_count < 0 ||
      static_cast<Offset>(frontier_count) > workspace.rows ||
      !workspace.frontier) {
    throw std::runtime_error("SSSP frontier trace requested an invalid frontier");
  }

  std::vector<Index> frontier(static_cast<std::size_t>(frontier_count));
  if (!frontier.empty()) {
    check_hip(hipMemcpyAsync(frontier.data(),
                             workspace.frontier,
                             frontier.size() * sizeof(Index),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy outgoing SSSP frontier trace");
    check_hip(hipStreamSynchronize(stream),
              "synchronize outgoing SSSP frontier trace copy");
    std::sort(frontier.begin(), frontier.end());
  }
  return frontier;
}

// Hybrid CPU/GPU host wrapper. Inputs: graph, workspace, source, target sinks,
// max iterations, early-stop flag, stream, and optional trace vector.
// Output: termination status and optional per-iteration frontier lists.
// Purpose: run active-frontier Bellman-Ford with optional nonnegative distance-bound stopping.
OutgoingSsspResult run_outgoing_frontier_sssp_distance_early_stop(
    const DeviceOutgoingCsrF32& graph,
    DeviceSsspWorkspace& workspace,
    int source,
    const std::vector<Index>& target_nodes,
    int max_iters,
    bool distance_early_stop_enabled,
    hipStream_t stream = nullptr,
    std::vector<std::vector<Index>>* frontier_trace = nullptr,
    bool upload_target_nodes = true) {
  validate_device_outgoing_csr(graph);
  if (workspace.rows != graph.rows || !workspace.best_state ||
      !workspace.frontier || !workspace.next_frontier || !workspace.next_marks ||
      !workspace.iteration_status || !workspace.host_iteration_status ||
      !workspace.status_ready_event) {
    throw std::runtime_error("SSSP frontier workspace does not match outgoing graph");
  }
  if (workspace.stream != stream) {
    throw std::invalid_argument("SSSP frontier workspace used on another stream");
  }
  if (source < 0 || static_cast<Offset>(source) >= graph.rows) {
    throw std::runtime_error("source is outside outgoing graph");
  }
  const Offset target_count =
      distance_early_stop_enabled ? static_cast<Offset>(target_nodes.size()) : 0;
  if (target_count > 0 &&
      (target_nodes.size() > static_cast<std::size_t>(workspace.gather_capacity) ||
       !workspace.gather_nodes)) {
    throw std::runtime_error("SSSP target workspace capacity is too small");
  }
  if (max_iters < 0) {
    max_iters = static_cast<int>(graph.rows) - 1;
  }

  if (target_count > 0 && upload_target_nodes) {
    check_hip(hipMemcpyAsync(workspace.gather_nodes,
                             target_nodes.data(),
                             target_nodes.size() * sizeof(Index),
                             hipMemcpyHostToDevice,
                             stream),
              "copy outgoing SSSP target nodes");
  }

  initialize_outgoing_sssp_state(workspace, source, stream);
  if (frontier_trace) {
    frontier_trace->clear();
  }

  int frontier_count = 1;
  int launched_iterations = 0;
  OutgoingSsspResult result;
  for (int iter = 0; iter < max_iters && frontier_count > 0; ++iter) {
    if (frontier_trace) {
      frontier_trace->push_back(
          copy_current_frontier_to_host(workspace, frontier_count, stream));
    }

    hipLaunchKernelGGL(reset_iteration_status_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch outgoing SSSP frontier status reset");

    const int mark_token = iter + 1;
    if (graph.unit_weights) {
      hipLaunchKernelGGL(outgoing_frontier_relax_kernel<true>,
                         grid_for_items(frontier_count, kBlockSize),
                         dim3(kBlockSize),
                         0,
                         stream,
                         graph.rowptr,
                         graph.to,
                         graph.values,
                         graph.rows,
                         graph.nnz,
                         workspace.frontier,
                         frontier_count,
                         mark_token,
                         workspace.best_state,
                         workspace.next_frontier,
                         workspace.next_marks,
                         workspace.iteration_status);
    } else {
      hipLaunchKernelGGL(outgoing_frontier_relax_kernel<false>,
                         grid_for_items(frontier_count, kBlockSize),
                         dim3(kBlockSize),
                         0,
                         stream,
                         graph.rowptr,
                         graph.to,
                         graph.values,
                         graph.rows,
                         graph.nnz,
                         workspace.frontier,
                         frontier_count,
                         mark_token,
                         workspace.best_state,
                         workspace.next_frontier,
                         workspace.next_marks,
                         workspace.iteration_status);
    }
    check_hip(hipGetLastError(), "launch outgoing SSSP frontier relaxation");

    if (distance_early_stop_enabled && target_count > 0) {
      hipLaunchKernelGGL(update_target_status_kernel,
                         grid_for_items(target_count, kBlockSize),
                         dim3(kBlockSize),
                         0,
                         stream,
                         workspace.best_state,
                         graph.rows,
                         workspace.gather_nodes,
                         target_count,
                         workspace.iteration_status);
      check_hip(hipGetLastError(), "launch outgoing SSSP target status");
    }

    check_hip(hipMemcpyAsync(workspace.host_iteration_status,
                             workspace.iteration_status,
                             sizeof(IterationStatus),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy outgoing SSSP frontier status");
    wait_for_workspace_status(workspace,
                              stream,
                              "record outgoing SSSP frontier status event",
                              "wait for outgoing SSSP frontier status event");
    const IterationStatus status = *workspace.host_iteration_status;
    if (status.error_status != 0) {
      throw std::runtime_error("outgoing SSSP frontier relaxation saw invalid graph data");
    }

    ++launched_iterations;
    result.iterations_used = launched_iterations;
    frontier_count = status.next_count;
    if (distance_early_stop_enabled &&
        frontier_count > 0 &&
        target_count > 0 &&
        status.reached_target_count == static_cast<int>(target_count) &&
        status.min_next_frontier_dist_bits > status.max_target_dist_bits) {
      // With nonnegative weights, every future relaxation from next_frontier has
      // distance at least min(next_frontier). If all queried targets already have
      // distance <= max_target, no target distance can still improve.
      result.early_stopped = true;
      break;
    }
    std::swap(workspace.frontier, workspace.next_frontier);
  }

  result.converged = frontier_count == 0;
  result.hit_max_iters = !result.converged && !result.early_stopped;
  return result;
}

}  // namespace rips_sssp_new

namespace {

using Offset = rips_sssp_new::Offset;
using Index = rips_sssp_new::Index;
using DeviceOffset = rips_sssp_new::DeviceOffset;

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t MIN_METADATA_VERSION = 3;
constexpr std::uint64_t CURRENT_METADATA_VERSION = 4;
constexpr std::uint64_t NODE_PHYSICAL_METADATA_VERSION = 4;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;

struct HostOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> to;
  std::vector<float> values;
};

struct DeviceOutgoingCsrOwner {
  rips_sssp_new::DeviceOutgoingCsrF32 view{};
  rips_sssp_new::DeviceOffset* rowptr = nullptr;
  Index* to = nullptr;
  float* values = nullptr;
};

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = kNoIndex;
  std::uint64_t pin_string = kNoIndex;
};

struct RouteRequest {
  std::uint64_t net_string = kNoIndex;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::uint64_t metadata_node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::vector<std::string> strings;
  std::vector<RouteRequest> route_requests;
};

struct Query {
  std::size_t net_index = 0;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::string net_name;
  int source = -1;
  int target = -1;
};

struct SourceWork {
  int source = -1;
  std::vector<Query> queries;
};

struct WorkBuildResult {
  std::uint64_t raw_source_count = 0;
  std::uint64_t invalid_source_count = 0;
  std::uint64_t invalid_sink_count = 0;
  std::uint64_t total_queries = 0;
  std::vector<SourceWork> work;
};

struct PathEdge {
  int from = -1;
  int to = -1;
  Offset csr_edge = -1;
  float cost = 0.0f;
};

struct Options {
  int max_iters = -1;
  bool distance_early_stop = true;
  bool print_frontiers = false;
  bool print_jsonl_entry_ids = false;
  std::size_t source_progress_every = 100;
  std::size_t source_limit = 0;
};

struct ParsedArgs {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path output_path;
  std::filesystem::path paths_output_path;
  Options options;
};

// CPU helper. Inputs: binary stream and field name. Output: one little-endian u64.
// Purpose: read fixed-width counts and ids from RIPS binary files.
std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

// CPU helper. Inputs: binary stream, destination vector, element count, field name.
// Output: vector resized and filled from the stream.
// Purpose: read CSR arrays while checking host-size limits.
template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) return;

  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

// CPU helper. Inputs: item count, bytes per item, field name. Output: total byte count.
// Purpose: detect overflow before skipping repeated binary records.
std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t bytes_per_item,
                                 const char* name) {
  if (bytes_per_item != 0 &&
      count > std::numeric_limits<std::uint64_t>::max() / bytes_per_item) {
    throw std::runtime_error(std::string(name) + " byte count overflow");
  }
  return count * bytes_per_item;
}

// CPU helper. Inputs: binary stream, byte count, field name. Output: stream advanced.
// Purpose: skip metadata sections bf9 does not need.
void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) return;
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) + " byte count is too large to seek");
  }
  in.seekg(static_cast<std::streamoff>(count), std::ios::cur);
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

// CPU helper. Inputs: binary stream. Output: one length-prefixed metadata string.
// Purpose: keep net names for JSONL path records.
std::string read_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large for this host");
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed while reading metadata string bytes");
    }
  }
  return text;
}

// CPU helper. Inputs: u64 count and field name. Output: host-size count.
// Purpose: guard vector resize calls while reading metadata.
std::size_t checked_size(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  return static_cast<std::size_t>(count);
}

// CPU helper. Inputs: compact metadata node id. Output: int node id, or -1.
// Purpose: keep malformed oversized metadata nodes out of route work.
int node_from_u64(std::uint64_t node) {
  if (node > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(node);
}

// CPU helper. Inputs: host outgoing CSR. Output: none, or throws.
// Purpose: validate outgoing graph shape and array consistency before GPU copy.
void validate_outgoing_csr(const HostOutgoingCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph.rows > static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for int node ids");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.to.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("outgoing CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("outgoing CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const Offset row_degree = end - begin;
    if (row_degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
  }

  for (std::size_t edge = 0; edge < graph.to.size(); ++edge) {
    if (graph.to[edge] < 0 || static_cast<Offset>(graph.to[edge]) >= graph.cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
  }
}

// CPU adapter. Inputs: PathFinder's outgoing HostCsrF32. Output: bf9's
// validated outgoing host view.
// Purpose: let the reusable PathFinder engine use exactly the same device graph
// and protected kernels as the standalone bf9 driver.
HostOutgoingCsrF32 make_outgoing_csr(const HostCsrF32& input) {
  if (input.rows < 0 || input.cols < 0 || input.nnz < 0) {
    throw std::runtime_error("outgoing CSR counts must be nonnegative");
  }
  if (input.rows == 0 || input.rows != input.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (input.rows > static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for int node ids");
  }
  if (input.nnz >= static_cast<Offset>(kPackedNoPredEdge)) {
    throw std::runtime_error(
        "outgoing CSR has too many edges for packed predecessor edge ids");
  }

  HostOutgoingCsrF32 graph;
  graph.rows = input.rows;
  graph.cols = input.cols;
  graph.nnz = input.nnz;
  graph.rowptr.assign(input.rowptr.begin(), input.rowptr.end());
  graph.to.assign(input.colind.begin(), input.colind.end());
  graph.values = input.values;

  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1)) {
    throw std::runtime_error("outgoing CSR rowptr size does not match row count");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const Offset row_degree = end - begin;
    if (row_degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
  }
  validate_outgoing_csr(graph);
  return graph;
}

// CPU helper. Inputs: outgoing RIPSCSR1 CSR path. Output: host outgoing CSR arrays.
// Purpose: load interchange_to_csr output and derive fields used by the kernels.
HostOutgoingCsrF32 load_outgoing_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open outgoing CSR file: " + path.string());
  }

  char magic[sizeof(CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error(
        "input is not an interchange_to_csr RIPSCSR1 file");
  }

  const std::uint64_t version = read_u64(in, "outgoing CSR format version");
  const std::uint64_t orientation = read_u64(in, "outgoing CSR orientation");
  if (version != EXPECTED_CSR_VERSION) {
    throw std::runtime_error(
        "unsupported RIPSCSR1 format version (expected version 1)");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error(
        "unsupported RIPSCSR1 orientation (expected outgoing orientation 2)");
  }

  const std::uint64_t rows = read_u64(in, "outgoing CSR row count");
  const std::uint64_t cols = read_u64(in, "outgoing CSR column count");
  const std::uint64_t declared_edges =
      read_u64(in, "outgoing CSR declared edge count");
  const std::uint64_t loaded_edges =
      read_u64(in, "outgoing CSR loaded edge count");
  const std::uint64_t nnz = read_u64(in, "outgoing CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "outgoing CSR rowptr count");
  const std::uint64_t colind_count =
      read_u64(in, "outgoing CSR colind count");
  const std::uint64_t values_count = read_u64(in, "outgoing CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this API");
  }
  if (nnz >= static_cast<std::uint64_t>(kPackedNoPredEdge)) {
    throw std::runtime_error(
        "outgoing CSR has too many edges for packed predecessor edge ids");
  }
  if (loaded_edges > declared_edges || nnz > loaded_edges) {
    throw std::runtime_error(
        "outgoing CSR declared/loaded/nnz edge counts are inconsistent");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz ||
      values_count != nnz) {
    throw std::runtime_error(
        "outgoing CSR rowptr/colind/values counts are inconsistent");
  }

  HostOutgoingCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  read_array(in, graph.to, colind_count, "outgoing CSR colind destinations");
  read_array(in, graph.values, values_count, "outgoing CSR values");

  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error(
        "outgoing CSR rowptr must start at 0 and end at nnz");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const Offset row_degree = end - begin;
    if (row_degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
  }
  validate_outgoing_csr(graph);
  return graph;
}

// Hybrid CPU/GPU helper. Inputs: host outgoing CSR and HIP stream.
// Output: device owner with graph arrays copied once to GPU memory.
// Purpose: avoid transferring the CSR inside Bellman-Ford iterations or per source.
DeviceOutgoingCsrOwner copy_outgoing_csr_to_device(const HostOutgoingCsrF32& host,
                                                   hipStream_t stream) {
  DeviceOutgoingCsrOwner device;
  try {
    std::vector<DeviceOffset> compact_rowptr;
    compact_rowptr.reserve(host.rowptr.size());
    for (const Offset offset : host.rowptr) {
      if (offset < 0 ||
          static_cast<std::uint64_t>(offset) >=
              static_cast<std::uint64_t>(kPackedNoPredEdge)) {
        throw std::runtime_error(
            "outgoing CSR row offset cannot be represented on the GPU");
      }
      compact_rowptr.push_back(static_cast<DeviceOffset>(offset));
    }
    const bool unit_weights =
        std::all_of(host.values.begin(), host.values.end(),
                    [](float value) { return value == 1.0f; });
    const std::size_t rowptr_bytes =
        compact_rowptr.size() * sizeof(DeviceOffset);
    rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr),
                                       rowptr_bytes),
                             "hipMalloc outgoing rowptr");
    rips_sssp_new::check_hip(hipMemcpyAsync(device.rowptr,
                                            compact_rowptr.data(),
                                            rowptr_bytes,
                                            hipMemcpyHostToDevice,
                                            stream),
                             "copy outgoing rowptr to device");

    if (host.nnz > 0) {
      const std::size_t to_bytes = host.to.size() * sizeof(Index);
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.to),
                                         to_bytes),
                               "hipMalloc outgoing destinations");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.to,
                                              host.to.data(),
                                              to_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing destinations to device");
      if (!unit_weights) {
        const std::size_t values_bytes = host.values.size() * sizeof(float);
        rips_sssp_new::check_hip(
            hipMalloc(reinterpret_cast<void**>(&device.values), values_bytes),
            "hipMalloc outgoing values");
        rips_sssp_new::check_hip(hipMemcpyAsync(device.values,
                                                host.values.data(),
                                                values_bytes,
                                                hipMemcpyHostToDevice,
                                                stream),
                                 "copy outgoing values to device");
      }
    }

    rips_sssp_new::check_hip(hipStreamSynchronize(stream),
                             "synchronize outgoing CSR copy");
    device.view.rows = host.rows;
    device.view.cols = host.cols;
    device.view.nnz = host.nnz;
    device.view.rowptr = device.rowptr;
    device.view.to = device.to;
    device.view.values = device.values;
    device.view.unit_weights = unit_weights;
    return device;
  } catch (...) {
    if (device.rowptr) (void)hipFree(device.rowptr);
    if (device.to) (void)hipFree(device.to);
    if (device.values) (void)hipFree(device.values);
    throw;
  }
}

// Hybrid CPU/GPU helper. Inputs: device owner. Output: freed GPU graph pointers.
// Purpose: release the one-time outgoing CSR device copy.
void free_device_outgoing_csr(DeviceOutgoingCsrOwner* device) {
  if (!device) return;
  if (device->rowptr) {
    (void)hipFree(device->rowptr);
    device->rowptr = nullptr;
  }
  if (device->to) {
    (void)hipFree(device->to);
    device->to = nullptr;
  }
  if (device->values) {
    (void)hipFree(device->values);
    device->values = nullptr;
  }
  device->view = {};
}

// CPU helper. Inputs: metadata path and graph nnz. Output: route request metadata.
// Purpose: parse outgoing RIPSIFM1 metadata needed to emit source-to-sink paths.
RoutingMetadata load_routing_metadata(const std::filesystem::path& path,
                                      Offset graph_nnz) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPSIFM1 metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version < MIN_METADATA_VERSION || version > CURRENT_METADATA_VERSION) {
    throw std::runtime_error(
        "unsupported RIPSIFM1 metadata version (expected version 3 or 4)");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error(
        "unsupported RIPSIFM1 orientation (expected outgoing orientation 2)");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
  if (edge_attr_count != static_cast<std::uint64_t>(graph_nnz)) {
    throw std::runtime_error(
        "metadata edge attribute count does not match outgoing CSR nnz");
  }
  const std::uint64_t pip_data_count = read_u64(in, "metadata pip data count");
  const std::uint64_t site_pin_attr_count = read_u64(in, "metadata site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "metadata route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "metadata sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "metadata logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "metadata logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "metadata physical byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "metadata logical byte count");

  (void)read_u64(in, "metadata device path string");
  (void)read_u64(in, "metadata physical path string");
  (void)read_u64(in, "metadata logical path string");
  (void)read_u64(in, "metadata logical design name");

  RoutingMetadata metadata;
  metadata.metadata_node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.strings.reserve(checked_size(string_count, "metadata string"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  skip_bytes(in,
             checked_byte_count(node_count, sizeof(std::uint64_t), "metadata device node ids"),
             "metadata device node ids");
  if (version >= NODE_PHYSICAL_METADATA_VERSION) {
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::int32_t),
                                  "metadata node min x coordinates"),
               "metadata node min x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::int32_t),
                                  "metadata node max x coordinates"),
               "metadata node max x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::int32_t),
                                  "metadata node min y coordinates"),
               "metadata node min y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::int32_t),
                                  "metadata node max y coordinates"),
               "metadata node max y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::uint64_t),
                                  "metadata node tile type strings"),
               "metadata node tile type strings");
    skip_bytes(in,
               checked_byte_count(node_count,
                                  sizeof(std::uint64_t),
                                  "metadata node wire type strings"),
               "metadata node wire type strings");
  }
  skip_bytes(in,
             checked_byte_count(edge_attr_count, 2 * sizeof(std::uint64_t), "metadata edge attrs"),
             "metadata edge attrs");
  skip_bytes(in,
             checked_byte_count(pip_data_count, 3 * sizeof(std::uint64_t), "metadata pip data"),
             "metadata pip data");
  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t),
                                "metadata site pin attrs"),
             "metadata site pin attrs");

  metadata.route_requests.resize(checked_size(route_request_count, "metadata route request"));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");
    const std::uint64_t source_count = read_u64(in, "metadata source count");
    request.sources.resize(checked_size(source_count, "metadata source"));
    for (SitePinNode& source : request.sources) {
      source.node = node_from_u64(read_u64(in, "metadata source node"));
      source.site_string = read_u64(in, "metadata source site");
      source.pin_string = read_u64(in, "metadata source pin");
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    request.sinks.resize(checked_size(sink_count, "metadata sink"));
    for (SitePinNode& sink : request.sinks) {
      sink.node = node_from_u64(read_u64(in, "metadata sink node"));
      sink.site_string = read_u64(in, "metadata sink site");
      sink.pin_string = read_u64(in, "metadata sink pin");
    }
  }

  skip_bytes(in,
             checked_byte_count(logical_cell_count,
                                3 * sizeof(std::uint64_t),
                                "metadata logical cells"),
             "metadata logical cells");
  skip_bytes(in,
             checked_byte_count(logical_net_count,
                                4 * sizeof(std::uint64_t),
                                "metadata logical nets"),
             "metadata logical nets");
  skip_bytes(in,
             checked_byte_count(logical_port_instance_count,
                                7 * sizeof(std::uint64_t),
                                "metadata logical port instances"),
             "metadata logical port instances");
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t), "metadata blocked nodes"),
             "metadata blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count,
                                sizeof(std::uint64_t),
                                "metadata sink stop nodes"),
             "metadata sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "metadata physical bytes");
  skip_bytes(in, logical_netlist_byte_count, "metadata logical bytes");
  return metadata;
}

// CPU helper. Inputs: graph and node id. Output: whether node id is usable.
// Purpose: filter route requests before sending source work to the GPU.
bool valid_node(const HostOutgoingCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

// CPU helper. Inputs: metadata string table and string id. Output: string value.
// Purpose: turn route request net ids into JSONL net names.
std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) return "";
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

// CPU helper. Inputs: route metadata and outgoing graph. Output: grouped source work.
// Purpose: run one SSSP per unique source while preserving every source-sink query.
WorkBuildResult build_source_work(const RoutingMetadata& metadata,
                                  const HostOutgoingCsrF32& graph) {
  WorkBuildResult result;
  std::unordered_map<int, std::size_t> source_to_work;
  source_to_work.reserve(metadata.route_requests.size());

  for (std::size_t net_index = 0; net_index < metadata.route_requests.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const std::string net_name = string_at(metadata, request.net_string);

    for (const SitePinNode& source_pin : request.sources) {
      ++result.raw_source_count;
      if (!valid_node(graph, source_pin.node)) {
        ++result.invalid_source_count;
        std::cout << "[warning] skipping out-of-range source node "
                  << source_pin.node << " for net_index=" << net_index << '\n';
        continue;
      }

      for (const SitePinNode& sink_pin : request.sinks) {
        if (!valid_node(graph, sink_pin.node)) {
          ++result.invalid_sink_count;
          std::cout << "[warning] skipping out-of-range sink node "
                    << sink_pin.node << " for net_index=" << net_index << '\n';
          continue;
        }

        std::size_t work_index = 0;
        const auto found = source_to_work.find(source_pin.node);
        if (found == source_to_work.end()) {
          work_index = result.work.size();
          source_to_work.emplace(source_pin.node, work_index);
          SourceWork source_work;
          source_work.source = source_pin.node;
          result.work.push_back(std::move(source_work));
        } else {
          work_index = found->second;
        }

        Query query;
        query.net_index = net_index;
        query.logical_net_index = request.logical_net_index;
        query.net_name = net_name;
        query.source = source_pin.node;
        query.target = sink_pin.node;
        result.work[work_index].queries.push_back(std::move(query));
        ++result.total_queries;
      }
    }
  }

  return result;
}

// CPU helper. Inputs: grouped work and source count. Output: query count covered.
// Purpose: report how many JSONL records the selected source limit can emit.
std::uint64_t count_selected_queries(const std::vector<SourceWork>& work,
                                     std::size_t source_count) {
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < source_count; ++i) {
    total += static_cast<std::uint64_t>(work[i].queries.size());
  }
  return total;
}

// CPU helper. Inputs: grouped source work, selected source count, and graph rows.
// Output: maximum de-duplicated sink count needed by any source.
// Purpose: size the GPU target-node buffer to the workload, not to every graph node.
Offset max_unique_targets_per_source(const std::vector<SourceWork>& work,
                                     std::size_t source_count,
                                     Offset rows) {
  std::size_t max_targets = 0;
  for (std::size_t i = 0; i < source_count; ++i) {
    std::unordered_set<int> targets;
    targets.reserve(work[i].queries.size());
    for (const Query& query : work[i].queries) {
      targets.insert(query.target);
    }
    max_targets = std::max(max_targets, targets.size());
  }
  if (max_targets > static_cast<std::size_t>(rows)) {
    throw std::runtime_error("unique sink count exceeds graph row count");
  }
  return static_cast<Offset>(max_targets);
}

// CPU helper. Inputs: one source's route queries. Output: unique target nodes.
// Purpose: copy each source's early-stop target set to the GPU once per source.
std::vector<Index> unique_targets_for_queries(const std::vector<Query>& queries) {
  std::vector<Index> targets;
  targets.reserve(queries.size());
  std::unordered_set<int> seen;
  seen.reserve(queries.size() * 2 + 1);
  for (const Query& query : queries) {
    if (seen.insert(query.target).second) {
      targets.push_back(static_cast<Index>(query.target));
    }
  }
  return targets;
}

// CPU helper. Inputs: packed best_state value. Output: distance as float.
// Purpose: decode bf9's packed predecessor state on the host.
float unpack_state_dist(unsigned long long state) {
  const unsigned int bits = static_cast<unsigned int>(state >> 32);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// CPU helper. Inputs: packed best_state value. Output: outgoing CSR predecessor edge id.
// Purpose: decode the low 32 bits used by bf9's frontier kernels.
Offset unpack_state_edge(unsigned long long state) {
  return static_cast<Offset>(static_cast<unsigned int>(state));
}

// CPU helper. Inputs: outgoing graph. Output: from node for each outgoing edge id.
// Purpose: reconstruct predecessor chains from edge ids stored in best_state.
std::vector<int> build_from_for_edge(const HostOutgoingCsrF32& graph) {
  std::vector<int> from_for_edge(static_cast<std::size_t>(graph.nnz), -1);
  for (Offset from = 0; from < graph.rows; ++from) {
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
         edge < graph.rowptr[static_cast<std::size_t>(from + 1)];
         ++edge) {
      from_for_edge[static_cast<std::size_t>(edge)] = static_cast<int>(from);
    }
  }
  return from_for_edge;
}

// CPU helper. Inputs: workspace state pointer and row count. Output: host packed states.
// Purpose: copy one completed SSSP predecessor tree back for path reconstruction.
std::vector<unsigned long long> copy_best_states_to_host(
    const rips_sssp_new::DeviceSsspWorkspace& workspace,
    Offset rows,
    hipStream_t stream) {
  std::vector<unsigned long long> states(static_cast<std::size_t>(rows));
  if (rows == 0) {
    return states;
  }
  rips_sssp_new::check_hip(
      hipMemcpyAsync(states.data(),
                     workspace.best_state,
                     states.size() * sizeof(unsigned long long),
                     hipMemcpyDeviceToHost,
                     stream),
      "copy outgoing SSSP best_state to host");
  rips_sssp_new::check_hip(hipStreamSynchronize(stream),
                           "synchronize outgoing SSSP best_state copy");
  g_bf9_full_state_copy_count.fetch_add(1, std::memory_order_relaxed);
  return states;
}

// CPU helper. Inputs: source and reconstructed edges. Output: source-to-target nodes.
// Purpose: emit the compact validator JSONL path shape.
std::vector<int> nodes_from_edges(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

// CPU helper. Inputs: graph, from map, packed states, query endpoints. Output: path edges.
// Purpose: follow outgoing CSR predecessor edge ids backward from target to source.
std::vector<PathEdge> reconstruct_shortest_path(
    const HostOutgoingCsrF32& graph,
    const std::vector<int>& from_for_edge,
    const std::vector<unsigned long long>& states,
    int source,
    int target) {
  if (!valid_node(graph, source) || !valid_node(graph, target)) {
    throw std::out_of_range("source or target is outside the outgoing CSR graph");
  }
  if (states.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("best_state vector size does not match CSR rows");
  }
  if (source == target) {
    return {};
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (Offset guard = 0; guard < graph.rows && current != source; ++guard) {
    const unsigned long long state = states[static_cast<std::size_t>(current)];
    const Offset edge = unpack_state_edge(state);
    if (edge == static_cast<Offset>(kPackedNoPredEdge)) {
      throw std::runtime_error("shortest path predecessor is missing");
    }
    if (edge < 0 || edge >= graph.nnz) {
      throw std::runtime_error("shortest path predecessor edge is outside outgoing CSR");
    }
    if (graph.to[static_cast<std::size_t>(edge)] != current) {
      throw std::runtime_error(
          "shortest path predecessor edge does not end at current node");
    }

    const int pred = from_for_edge[static_cast<std::size_t>(edge)];
    if (!valid_node(graph, pred)) {
      throw std::runtime_error("shortest path predecessor source is invalid");
    }

    PathEdge path_edge;
    path_edge.from = pred;
    path_edge.to = current;
    path_edge.csr_edge = edge;
    path_edge.cost = graph.values[static_cast<std::size_t>(edge)];
    reversed.push_back(path_edge);
    current = pred;
  }

  if (current != source) {
    throw std::runtime_error("shortest path reconstruction did not reach source");
  }

  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

// CPU helper. Inputs: text and field name. Output: parsed size_t.
// Purpose: parse nonnegative integer CLI options.
std::size_t parse_size(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::size_t>(value);
}

// CPU helper. Inputs: text and field name. Output: parsed int.
// Purpose: parse --max-iters, including -1 for the default n - 1 limit.
int parse_int(const std::string& text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid integer for ") + name);
  }
  return static_cast<int>(value);
}

// CPU helper. Inputs: argv index and option name. Output: option value string.
// Purpose: consume the value following a CLI option.
std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

// CPU helper. Inputs: program name. Output: usage text on stderr.
// Purpose: document bf9's path-outputting command line.
void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <outgoing.csrbin> <metadata.ifmeta.bin> [summary.csv] [options]\n\n"
      << "Options:\n"
      << "  --max-iters <n>             Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --no-distance-early-stop   Disable distance-bound target early stopping.\n"
      << "  --paths-output <path>       Enable JSONL path output at this path. Disabled by default.\n"
      << "  --print-frontiers           Print per-iteration frontier vertex ids for detailed source lines.\n"
      << "  --print-jsonl-entry-ids     Print JSONL path entry ids for detailed source lines.\n"
      << "  --source-progress-every <n> Print detailed progress every n sources. "
         "Default: 100; 0 disables detailed lines.\n"
      << "  --source-limit <n>          Run only the first n unique valid source nodes. Default: 0 = all.\n"
      << "  --help                      Print this message.\n";
}

// CPU helper. Inputs: argc and argv. Output: parsed paths and options.
// Purpose: accept required CSR/metadata paths, optional summary CSV, and simple controls.
ParsedArgs parse_args(int argc, char** argv) {
  ParsedArgs parsed;
  int positional_count = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--max-iters") {
      parsed.options.max_iters =
          parse_int(require_value(&i, argc, argv, "--max-iters"), "--max-iters");
    } else if (arg == "--no-distance-early-stop") {
      parsed.options.distance_early_stop = false;
    } else if (arg == "--paths-output") {
      parsed.paths_output_path = require_value(&i, argc, argv, "--paths-output");
    } else if (arg == "--print-frontiers") {
      parsed.options.print_frontiers = true;
    } else if (arg == "--print-jsonl-entry-ids" || arg == "--print-jsonl-records") {
      parsed.options.print_jsonl_entry_ids = true;
    } else if (arg == "--source-progress-every") {
      parsed.options.source_progress_every =
          parse_size(require_value(&i, argc, argv, "--source-progress-every"),
                     "--source-progress-every");
    } else if (arg == "--source-limit") {
      parsed.options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      if (positional_count == 0) {
        parsed.csr_path = arg;
      } else if (positional_count == 1) {
        parsed.metadata_path = arg;
      } else if (positional_count == 2) {
        parsed.output_path = arg;
      } else {
        throw std::runtime_error("too many positional arguments");
      }
      ++positional_count;
    }
  }

  if (parsed.csr_path.empty() || parsed.metadata_path.empty()) {
    throw std::runtime_error("expected <outgoing.csrbin> <metadata.ifmeta.bin>");
  }
  if (parsed.options.max_iters < -1) {
    throw std::runtime_error("--max-iters must be -1 or nonnegative");
  }
  return parsed;
}

// CPU helper. Inputs: completed count, total count, visual width. Output: progress bar string.
// Purpose: create a compact progress bar for source-level status.
std::string progress_bar(std::size_t completed, std::size_t total, int width) {
  if (width <= 0) return "";
  const double fraction =
      total == 0 ? 1.0 : std::min(1.0, static_cast<double>(completed) /
                                           static_cast<double>(total));
  const int filled = static_cast<int>(fraction * width);
  std::string bar;
  bar.reserve(static_cast<std::size_t>(width) + 2);
  bar.push_back('[');
  for (int i = 0; i < width; ++i) {
    bar.push_back(i < filled ? '#' : '.');
  }
  bar.push_back(']');
  return bar;
}

// CPU helper. Inputs: completed count, total count, and whether to end the line.
// Output: progress bar printed to stdout.
// Purpose: show how many unique sources have completed SSSP.
void print_source_progress(std::size_t completed, std::size_t total, bool final_line) {
  const double percent =
      total == 0 ? 100.0 : 100.0 * static_cast<double>(completed) /
                                static_cast<double>(total);
  std::ostringstream percent_text;
  percent_text << std::fixed
               << std::setprecision(percent > 0.0 && percent < 0.1 ? 4 : 1)
               << std::min(100.0, percent);
  std::cout << '\r'
            << "[bf9] sources "
            << progress_bar(completed, total, 30)
            << ' ' << percent_text.str() << "%"
            << " " << completed << "/" << total
            << "        "
            << std::flush;
  if (final_line) {
    std::cout << '\n';
  }
}

// CPU helper. Inputs: optional output path. Output: opened CSV stream, or unopened stream.
// Purpose: provide minimal per-source summary output without copying distances.
std::ofstream open_summary_output(const std::filesystem::path& output_path) {
  std::ofstream out;
  if (output_path.empty()) {
    return out;
  }
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  out.open(output_path);
  if (!out) {
    throw std::runtime_error("could not open summary output: " + output_path.string());
  }
  out << "source,iterations_used,converged\n";
  return out;
}

// CPU helper. Inputs: summary stream, source node, result. Output: one CSV row if stream is open.
// Purpose: record minimal Bellman-Ford status per source.
void write_summary_record(std::ofstream& out,
                          int source,
                          const rips_sssp_new::OutgoingSsspResult& result) {
  if (!out) return;
  out << source << ','
      << result.iterations_used << ','
      << (result.converged ? "yes" : "no") << '\n';
}

// CPU helper. Inputs: SSSP result. Output: stable termination label for logs.
// Purpose: make early-stop progress lines explicit without changing the CSV schema.
const char* termination_label(const rips_sssp_new::OutgoingSsspResult& result) {
  if (result.converged) return "full_convergence";
  if (result.early_stopped) return "early_distance_bound";
  if (result.hit_max_iters) return "max_iters";
  return "unknown";
}

// CPU helper. Inputs: optional JSONL output path. Output: opened path stream.
// Purpose: create SSSP-new/paths output directories on demand.
std::ofstream open_paths_output(const std::filesystem::path& output_path) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("could not open paths output: " + output_path.string());
  }
  return out;
}

// CPU helper. Inputs: raw string. Output: JSON string contents without quotes.
// Purpose: write validator-compatible JSONL without a third-party JSON library.
std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

// CPU helper. Inputs: output stream and string. Output: quoted JSON string.
// Purpose: keep path records compact and schema-compatible.
void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

// CPU helper. Inputs: output stream and per-iteration frontier trace.
// Output: JSON-like list of vertex-id lists.
// Purpose: print optional frontier diagnostics for source progress lines.
void write_frontier_trace(std::ostream& out,
                          const std::vector<std::vector<Index>>& frontiers) {
  out << '[';
  for (std::size_t iter = 0; iter < frontiers.size(); ++iter) {
    if (iter != 0) out << ',';
    out << '[';
    for (std::size_t item = 0; item < frontiers[iter].size(); ++item) {
      if (item != 0) out << ',';
      out << frontiers[iter][item];
    }
    out << ']';
  }
  out << ']';
}

// CPU helper. Inputs: output stream and unsigned ids. Output: JSON-like id list.
// Purpose: print compact JSONL entry diagnostics without printing full JSONL records.
void write_u64_list(std::ostream& out, const std::vector<std::uint64_t>& ids) {
  out << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) out << ',';
    out << ids[i];
  }
  out << ']';
}

// CPU helper. Inputs: output stream, graph, metadata, selected source/query counts.
// Output: one JSONL metadata line.
// Purpose: identify this as outgoing SSSP-new path output for validators and humans.
void write_metadata_record(std::ostream& out,
                           const HostOutgoingCsrF32& graph,
                           const RoutingMetadata& metadata,
                           std::size_t selected_sources,
                           std::uint64_t selected_queries) {
  out << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\""
      << ",\"producer\":\"bf9\""
      << ",\"node_count\":" << graph.rows
      << ",\"edge_count\":" << graph.nnz
      << ",\"route_request_count\":" << metadata.route_requests.size()
      << ",\"selected_source_count\":" << selected_sources
      << ",\"selected_query_count\":" << selected_queries
      << ",\"edge_orientation\":\"outgoing\""
      << ",\"description\":\"row u stores directed edges u -> v\""
      << "}\n";
}

// CPU helper. Inputs: query result and reconstructed path. Output: one JSONL path line.
// Purpose: emit the compact schema consumed by SSSP validators and comparators.
void write_path_record(std::ostream& out,
                       const Query& query,
                       bool reached,
                       float distance,
                       const std::vector<int>& nodes,
                       const std::vector<PathEdge>& edges) {
  out << "{\"type\":\"path\"";
  out << ",\"net\":";
  write_json_string(out, query.net_name);
  out << ",\"net_index\":" << query.net_index;
  if (query.logical_net_index == kNoLogicalNetIndex) {
    out << ",\"logical_net_index\":null";
  } else {
    out << ",\"logical_net_index\":" << query.logical_net_index;
  }
  out << ",\"source\":" << query.source
      << ",\"target\":" << query.target
      << ",\"reached\":" << (reached ? "true" : "false");

  if (reached) {
    out << ",\"distance\":" << std::setprecision(9) << distance;
  } else {
    out << ",\"distance\":null";
  }

  out << ",\"nodes\":[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) out << ',';
    out << nodes[i];
  }
  out << "]";

  out << ",\"csr_edges\":[";
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (i != 0) out << ',';
    out << edges[i].csr_edge;
  }
  out << "]}\n";
}

// CPU function. Inputs: command-line arguments. Output: process exit code.
// Purpose: load inputs once, run one SSSP per source, and optionally emit paths.
int main_impl(int argc, char** argv) {
  const auto total_begin = std::chrono::steady_clock::now();
  const ParsedArgs args = parse_args(argc, argv);
  const bool emit_paths = !args.paths_output_path.empty();
  const bool print_jsonl_entry_ids = args.options.print_jsonl_entry_ids;

  std::cout << "[bf9] loading outgoing CSR: " << args.csr_path.string() << '\n';
  HostOutgoingCsrF32 graph = load_outgoing_csrbin(args.csr_path);
  std::cout << "[bf9] graph rows=" << graph.rows
            << " nnz=" << graph.nnz
            << " orientation=outgoing(row u stores edges u -> v)\n";

  std::cout << "[bf9] loading metadata: " << args.metadata_path.string() << '\n';
  RoutingMetadata metadata = load_routing_metadata(args.metadata_path, graph.nnz);
  if (metadata.metadata_node_count != static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error("metadata node count does not match outgoing CSR rows");
  }

  std::vector<int> from_for_edge;
  if (emit_paths) {
    from_for_edge = build_from_for_edge(graph);
  }
  WorkBuildResult work = build_source_work(metadata, graph);
  const std::size_t source_count =
      args.options.source_limit == 0
          ? work.work.size()
          : std::min(args.options.source_limit, work.work.size());
  const std::uint64_t selected_queries =
      count_selected_queries(work.work, source_count);
  std::cout << "[bf9] raw_metadata_sources=" << work.raw_source_count
            << " invalid_sources=" << work.invalid_source_count
            << " invalid_sinks=" << work.invalid_sink_count
            << " unique_valid_sources=" << work.work.size()
            << " sources_to_run=" << source_count
            << " source_sink_queries=" << selected_queries << '\n';
  const Offset target_capacity =
      args.options.distance_early_stop
          ? max_unique_targets_per_source(work.work, source_count, graph.rows)
          : 0;
  std::cout << "[bf9] distance_early_stop="
            << (args.options.distance_early_stop ? "enabled" : "disabled");
  if (args.options.distance_early_stop) {
    std::cout << " max_unique_sinks_per_source=" << target_capacity;
  }
  std::cout << '\n';

  std::ofstream summary_out = open_summary_output(args.output_path);
  std::ofstream paths_out;
  if (emit_paths) {
    paths_out = open_paths_output(args.paths_output_path);
    write_metadata_record(paths_out, graph, metadata, source_count, selected_queries);
    std::cout << "[bf9] writing paths JSONL: "
              << args.paths_output_path.string() << '\n';
  } else {
    std::cout << "[bf9] paths JSONL disabled; pass --paths-output <path> to emit paths\n";
  }
  if (args.options.print_frontiers) {
    std::cout << "[bf9] frontier printing enabled for detailed source progress lines";
    if (args.options.source_progress_every == 0) {
      std::cout << " but --source-progress-every 0 disables detailed source lines";
    }
    std::cout << '\n';
  }
  if (print_jsonl_entry_ids) {
    std::cout << "[bf9] JSONL entry-id printing enabled for detailed source progress lines";
    if (!emit_paths) {
      std::cout << " but --paths-output is disabled, so no JSONL entry ids exist";
    }
    if (args.options.source_progress_every == 0) {
      std::cout << " but --source-progress-every 0 disables detailed source lines";
    }
    std::cout << '\n';
  }

  if (source_count == 0) {
    print_source_progress(0, 0, true);
    std::cout << "[bf9] no valid unique sources to run\n";
    if (emit_paths) {
      paths_out.flush();
      if (!paths_out) {
        throw std::runtime_error("failed while writing paths JSONL");
      }
      std::cout << "[bf9] wrote paths: " << args.paths_output_path.string() << '\n';
    }
    return 0;
  }

  hipStream_t stream = nullptr;
  DeviceOutgoingCsrOwner d_graph;
  rips_sssp_new::DeviceSsspWorkspace workspace;
  bool stream_created = false;
  bool graph_copied = false;
  bool workspace_allocated = false;

  try {
    rips_sssp_new::check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    stream_created = true;

    std::cout << "[bf9] copying outgoing CSR to GPU once\n";
    d_graph = copy_outgoing_csr_to_device(graph, stream);
    graph_copied = true;

    std::cout << "[bf9] allocating reusable frontier SSSP workspace\n";
    workspace = rips_sssp_new::make_sssp_workspace(graph.rows,
                                                   target_capacity,
                                                   stream);
    workspace_allocated = true;

    print_source_progress(0, source_count, false);
    std::uint64_t paths_written = 0;
    std::uint64_t reached_count = 0;
    std::uint64_t unreached_count = 0;
    std::uint64_t full_convergence_count = 0;
    std::uint64_t early_stop_count = 0;
    std::uint64_t max_iter_count = 0;
    for (std::size_t source_index = 0; source_index < source_count; ++source_index) {
      const SourceWork& source_work = work.work[source_index];
      const int source = source_work.source;
      const std::size_t completed = source_index + 1;
      const bool final_source = completed == source_count;
      const bool detailed =
          args.options.source_progress_every > 0 &&
          (completed % args.options.source_progress_every == 0 || final_source);
      const bool collect_source_jsonl_ids = print_jsonl_entry_ids && emit_paths && detailed;
      std::vector<std::vector<Index>> frontier_trace;
      std::vector<std::vector<Index>>* frontier_trace_output =
          args.options.print_frontiers && detailed ? &frontier_trace : nullptr;
      std::vector<std::uint64_t> source_jsonl_entry_ids;
      std::vector<std::uint64_t> source_jsonl_line_numbers;
      const std::vector<Index> target_nodes =
          args.options.distance_early_stop
              ? unique_targets_for_queries(source_work.queries)
              : std::vector<Index>{};
      const rips_sssp_new::OutgoingSsspResult result =
          rips_sssp_new::run_outgoing_frontier_sssp_distance_early_stop(
              d_graph.view,
              workspace,
              source,
              target_nodes,
              args.options.max_iters,
              args.options.distance_early_stop,
              stream,
              frontier_trace_output);
      if (result.converged) {
        ++full_convergence_count;
      } else if (result.early_stopped) {
        ++early_stop_count;
      } else if (result.hit_max_iters) {
        ++max_iter_count;
      }
      write_summary_record(summary_out, source, result);

      const std::uint64_t paths_before = paths_written;
      if (emit_paths) {
        const std::vector<unsigned long long> states =
            copy_best_states_to_host(workspace, graph.rows, stream);
        for (const Query& query : source_work.queries) {
          const float distance =
              query.source == query.target
                  ? 0.0f
                  : unpack_state_dist(states[static_cast<std::size_t>(query.target)]);
          const bool reached =
              query.source == query.target || std::isfinite(distance);
          if (collect_source_jsonl_ids) {
            source_jsonl_entry_ids.push_back(paths_written + 1);
            source_jsonl_line_numbers.push_back(paths_written + 2);
          }
          if (!reached) {
            write_path_record(paths_out, query, false, distance, {}, {});
            ++unreached_count;
            ++paths_written;
            continue;
          }

          std::vector<PathEdge> edges =
              reconstruct_shortest_path(graph,
                                        from_for_edge,
                                        states,
                                        query.source,
                                        query.target);
          std::vector<int> nodes = nodes_from_edges(query.source, edges);
          write_path_record(paths_out, query, true, distance, nodes, edges);
          ++reached_count;
          ++paths_written;
        }
      }

      print_source_progress(completed, source_count, detailed || final_source);

      if (detailed) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - total_begin).count();
        const double sources_per_sec =
            elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
        std::cout << "[bf9] source " << completed << "/" << source_count
                  << " node=" << source
                  << " queries=" << source_work.queries.size()
                  << " iterations_used=" << result.iterations_used
                  << " termination=" << termination_label(result)
                  << " converged=" << (result.converged ? "yes" : "no");
        if (emit_paths) {
          std::cout << " source_paths_written=" << (paths_written - paths_before)
                    << " total_paths_written=" << paths_written;
        }
        std::cout << " elapsed_sec=" << std::fixed << std::setprecision(3)
                  << elapsed
                  << " avg_sources_per_sec=" << std::setprecision(3)
                  << sources_per_sec << '\n';
        if (args.options.print_frontiers) {
          std::cout << "[bf9] frontier_trace source " << completed << "/"
                    << source_count
                    << " node=" << source
                    << " iterations_used=" << result.iterations_used
                    << " frontiers=";
          write_frontier_trace(std::cout, frontier_trace);
          std::cout << '\n';
        }
        if (print_jsonl_entry_ids) {
          std::cout << "[bf9] jsonl_entry_ids source " << completed << "/"
                    << source_count
                    << " node=" << source
                    << " count=" << source_jsonl_entry_ids.size();
          if (!emit_paths) {
            std::cout << " unavailable_no_paths_output=yes\n";
          } else {
            std::cout << " jsonl_entry_ids=";
            write_u64_list(std::cout, source_jsonl_entry_ids);
            std::cout << " jsonl_line_numbers=";
            write_u64_list(std::cout, source_jsonl_line_numbers);
            std::cout << '\n';
          }
        }
      }
    }
    std::cout << "[bf9] termination_counts"
              << " full_convergence=" << full_convergence_count
              << " early_distance_bound=" << early_stop_count
              << " max_iters=" << max_iter_count << '\n';
    if (emit_paths) {
      paths_out.flush();
      if (!paths_out) {
        throw std::runtime_error("failed while writing paths JSONL");
      }
      std::cout << "[bf9] path_records=" << paths_written
                << " reached=" << reached_count
                << " unreached=" << unreached_count << '\n';
    }

    rips_sssp_new::check_hip(hipStreamSynchronize(stream), "final stream sync");
    if (workspace_allocated) {
      rips_sssp_new::free_sssp_workspace(&workspace);
      workspace_allocated = false;
    }
    if (graph_copied) {
      free_device_outgoing_csr(&d_graph);
      graph_copied = false;
    }
    rips_sssp_new::check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    stream_created = false;
  } catch (...) {
    if (workspace_allocated) {
      rips_sssp_new::free_sssp_workspace(&workspace);
    }
    if (graph_copied) {
      free_device_outgoing_csr(&d_graph);
    }
    if (stream_created) {
      const hipError_t destroy_status = hipStreamDestroy(stream);
      if (destroy_status != hipSuccess) {
        std::cerr << "[bf9] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
    }
    throw;
  }

  const auto total_end = std::chrono::steady_clock::now();
  const double total_seconds =
      std::chrono::duration<double>(total_end - total_begin).count();
  std::cout << "[bf9] completed " << source_count
            << " source SSSP runs"
            << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
            << total_seconds << '\n';
  if (!args.output_path.empty()) {
    std::cout << "[bf9] wrote summary: " << args.output_path.string() << '\n';
  }
  if (emit_paths) {
    std::cout << "[bf9] wrote paths: " << args.paths_output_path.string() << '\n';
  }
  return 0;
}

}  // namespace

namespace {

struct BellmanFordTargetCandidate {
  float distance = std::numeric_limits<float>::infinity();
  int source = -1;
  std::vector<int> nodes;
  std::vector<Offset> edges;
};

struct BellmanFordPendingPath {
  std::size_t unique_target_index = 0;
  int target = -1;
  int current = -1;
  bool complete = false;
  bool needs_full_state_fallback = false;
  std::vector<PathEdge> reversed_edges;
  std::unordered_set<int> seen_nodes;
};

// CPU helper. Inputs: gathered distance/source/target and the current winner.
// Output: whether path reconstruction can affect the result. Purpose: reject
// definite losers before any predecessor-state traffic.
bool bellman_ford_candidate_can_improve(
    float distance,
    int source,
    int target,
    const BellmanFordTargetCandidate& current) {
  if (!std::isfinite(distance)) {
    return false;
  }
  if (!std::isfinite(current.distance)) {
    return true;
  }
  if (distance != current.distance) {
    return distance < current.distance;
  }
  if (source == target && current.source != target) {
    return true;
  }
  if (current.source == target && source != target) {
    return false;
  }
  if (source != current.source) {
    // Sources are stable-deduplicated and run exactly once. For two different
    // source IDs the lower source decides an equal-distance tie before path
    // edges are considered, so a cross-source lexicographic path comparison
    // cannot change the winner. Packed edge ordering still deterministically
    // selects the predecessor chain within each source's GPU run.
    return source < current.source;
  }

  // This cannot normally occur because a unique source runs once, but retain
  // the path-based final comparison if that invariant changes later.
  return true;
}

// Adapter-only path fallback. The protected packed predecessor ordering can
// make a zero-cost predecessor cycle even though its final distances are
// correct. Keep the packed predecessor path as the primary result; only when
// that reconstruction fails, walk finite distance-tight outgoing edges in CSR
// edge order to recover a deterministic simple source-to-target path.
std::vector<PathEdge> reconstruct_bellman_ford_adapter_path(
    const HostOutgoingCsrF32& graph,
    const std::vector<int>& from_for_edge,
    const std::vector<unsigned long long>& states,
    int source,
    int target) {
  try {
    return reconstruct_shortest_path(
        graph, from_for_edge, states, source, target);
  } catch (const std::runtime_error& predecessor_error) {
    std::vector<int> predecessor_node(
        static_cast<std::size_t>(graph.rows), -1);
    std::vector<Offset> predecessor_edge(
        static_cast<std::size_t>(graph.rows), -1);
    std::vector<unsigned char> seen(
        static_cast<std::size_t>(graph.rows), 0);
    std::vector<int> queue;
    queue.reserve(static_cast<std::size_t>(graph.rows));
    queue.push_back(source);
    seen[static_cast<std::size_t>(source)] = 1;

    for (std::size_t head = 0;
         head < queue.size() && !seen[static_cast<std::size_t>(target)];
         ++head) {
      const int from = queue[head];
      const float from_distance =
          unpack_state_dist(states[static_cast<std::size_t>(from)]);
      if (!std::isfinite(from_distance)) {
        continue;
      }
      for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
           edge < graph.rowptr[static_cast<std::size_t>(from + 1)];
           ++edge) {
        const int to = graph.to[static_cast<std::size_t>(edge)];
        if (seen[static_cast<std::size_t>(to)]) {
          continue;
        }
        const float to_distance =
            unpack_state_dist(states[static_cast<std::size_t>(to)]);
        const float candidate =
            from_distance + graph.values[static_cast<std::size_t>(edge)];
        if (!std::isfinite(to_distance) || candidate != to_distance) {
          continue;
        }
        predecessor_node[static_cast<std::size_t>(to)] = from;
        predecessor_edge[static_cast<std::size_t>(to)] = edge;
        seen[static_cast<std::size_t>(to)] = 1;
        queue.push_back(to);
        if (to == target) {
          break;
        }
      }
    }

    if (!seen[static_cast<std::size_t>(target)]) {
      throw std::runtime_error(
          std::string("Bellman-Ford packed predecessor reconstruction failed: ") +
          predecessor_error.what() +
          "; no distance-tight fallback path reached the target");
    }

    std::vector<PathEdge> reversed;
    int current = target;
    while (current != source) {
      const Offset edge =
          predecessor_edge[static_cast<std::size_t>(current)];
      const int from = predecessor_node[static_cast<std::size_t>(current)];
      if (edge < 0 || from < 0) {
        throw std::runtime_error(
            "Bellman-Ford distance-tight fallback predecessor is missing");
      }
      reversed.push_back(
          {from,
           current,
           edge,
           graph.values[static_cast<std::size_t>(edge)]});
      current = from;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
  }
}

bool prefer_bellman_ford_candidate(
    float distance,
    int source,
    int target,
    const std::vector<PathEdge>& edges,
    const BellmanFordTargetCandidate& current) {
  if (!std::isfinite(current.distance)) {
    return true;
  }
  if (distance != current.distance) {
    return distance < current.distance;
  }
  if (source == target && current.source != target) {
    return true;
  }
  if (current.source == target && source != target) {
    return false;
  }
  if (source != current.source) {
    return source < current.source;
  }

  std::vector<Offset> candidate_edges;
  candidate_edges.reserve(edges.size());
  for (const PathEdge& edge : edges) {
    candidate_edges.push_back(edge.csr_edge);
  }
  return std::lexicographical_compare(candidate_edges.begin(),
                                      candidate_edges.end(),
                                      current.edges.begin(),
                                      current.edges.end());
}

std::vector<int> deduplicate_bellman_ford_sources(
    const std::vector<int>& sources,
    Offset rows) {
  if (sources.empty()) {
    throw std::invalid_argument("Bellman-Ford requires at least one source");
  }
  if (sources.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Bellman-Ford source count must fit in int");
  }

  std::vector<int> unique_sources;
  unique_sources.reserve(sources.size());
  std::unordered_set<int> seen;
  seen.reserve(sources.size());
  for (const int source : sources) {
    if (source < 0 || static_cast<Offset>(source) >= rows) {
      throw std::out_of_range("Bellman-Ford source is outside the CSR graph");
    }
    if (seen.insert(source).second) {
      unique_sources.push_back(source);
    }
  }
  return unique_sources;
}

std::vector<int> validate_and_deduplicate_bellman_ford_targets(
    const std::vector<int>& targets,
    Offset rows) {
  if (targets.empty()) {
    throw std::invalid_argument("Bellman-Ford requires at least one target");
  }
  if (targets.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Bellman-Ford target count must fit in int");
  }
  std::vector<int> unique_targets;
  unique_targets.reserve(targets.size());
  std::unordered_set<int> seen;
  seen.reserve(targets.size());
  for (const int target : targets) {
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      throw std::out_of_range("Bellman-Ford target is outside the CSR graph");
    }
    if (seen.insert(target).second) {
      unique_targets.push_back(target);
    }
  }
  return unique_targets;
}

int saturated_iteration_sum(int current, int added) {
  if (added > 0 && current > std::numeric_limits<int>::max() - added) {
    return std::numeric_limits<int>::max();
  }
  return current + added;
}

}  // namespace

struct BellmanFordCsrGraph::Impl {
  HostOutgoingCsrF32 host;
  DeviceOutgoingCsrOwner device;
  std::vector<int> from_for_edge;
  int hip_device = -1;

  Impl(const HostCsrF32& adjacency, hipStream_t stream)
      : host(make_outgoing_csr(adjacency)),
        from_for_edge(build_from_for_edge(host)) {
    PATHFINDER_PROFILE_RANGE("bf9.upload_graph");
    rips_sssp_new::check_hip(hipGetDevice(&hip_device),
                             "get Bellman-Ford graph HIP device");
    device = copy_outgoing_csr_to_device(host, stream);
  }

  ~Impl() {
    free_device_outgoing_csr(&device);
  }
};

BellmanFordCsrGraph::BellmanFordCsrGraph(const HostCsrF32& adjacency,
                                         hipStream_t stream)
    : impl_(std::make_shared<Impl>(adjacency, stream)) {}

BellmanFordCsrGraph::~BellmanFordCsrGraph() = default;
BellmanFordCsrGraph::BellmanFordCsrGraph(BellmanFordCsrGraph&&) noexcept = default;
BellmanFordCsrGraph& BellmanFordCsrGraph::operator=(
    BellmanFordCsrGraph&&) noexcept = default;

struct BellmanFordCsrWorkspace::Impl {
  std::shared_ptr<const BellmanFordCsrGraph::Impl> graph;
  rips_sssp_new::DeviceSsspWorkspace workspace;
  hipStream_t stream = nullptr;
  unsigned long long* host_target_states = nullptr;
  Offset path_gather_capacity = 0;
  Index* path_gather_nodes = nullptr;
  unsigned long long* path_gather_states = nullptr;
  Index* host_path_gather_nodes = nullptr;
  unsigned long long* host_path_gather_states = nullptr;

  static std::shared_ptr<const BellmanFordCsrGraph::Impl> require_graph(
      const std::shared_ptr<const BellmanFordCsrGraph>& candidate) {
    if (!candidate || !candidate->impl_) {
      throw std::invalid_argument("Bellman-Ford shared graph must not be null");
    }
    return candidate->impl_;
  }

  Impl(std::shared_ptr<const BellmanFordCsrGraph::Impl> graph_in,
       hipStream_t stream_in)
      : graph(std::move(graph_in)), stream(stream_in) {
    int current_device = -1;
    rips_sssp_new::check_hip(hipGetDevice(&current_device),
                             "get Bellman-Ford workspace HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument(
          "Bellman-Ford shared graph belongs to a different HIP device");
    }
    workspace = rips_sssp_new::make_sssp_workspace(graph->host.rows, 0, stream);
  }

  Impl(const HostCsrF32& adjacency, hipStream_t stream_in)
      : Impl(std::make_shared<BellmanFordCsrGraph::Impl>(adjacency, stream_in),
             stream_in) {}

  Impl(const std::shared_ptr<const BellmanFordCsrGraph>& adjacency,
       hipStream_t stream_in)
      : Impl(require_graph(adjacency), stream_in) {}

  ~Impl() {
    rips_sssp_new::free_sssp_workspace(&workspace);
    if (path_gather_nodes) (void)hipFree(path_gather_nodes);
    if (path_gather_states) (void)hipFree(path_gather_states);
    if (host_target_states) (void)hipHostFree(host_target_states);
    if (host_path_gather_nodes) (void)hipHostFree(host_path_gather_nodes);
    if (host_path_gather_states) (void)hipHostFree(host_path_gather_states);
  }

  void require_stream(hipStream_t candidate) const {
    if (candidate != stream) {
      throw std::invalid_argument(
          "BellmanFordCsrWorkspace is stream-affine; use its construction stream");
    }
    int current_device = -1;
    rips_sssp_new::check_hip(hipGetDevice(&current_device),
                             "get Bellman-Ford run HIP device");
    if (current_device != graph->hip_device) {
      throw std::invalid_argument(
          "Bellman-Ford workspace is running on a different HIP device");
    }
  }

  void ensure_target_capacity(std::size_t target_count) {
    if (target_count <= static_cast<std::size_t>(workspace.gather_capacity)) {
      return;
    }
    if (target_count > static_cast<std::size_t>(graph->host.rows)) {
      throw std::invalid_argument(
          "Bellman-Ford target count exceeds the CSR row count");
    }

    Offset new_capacity = std::max<Offset>(1, workspace.gather_capacity);
    const Offset required = static_cast<Offset>(target_count);
    while (new_capacity < required) {
      if (new_capacity > graph->host.rows / 2) {
        new_capacity = graph->host.rows;
        break;
      }
      new_capacity *= 2;
    }

    Index* new_gather_nodes = nullptr;
    unsigned long long* new_gather_states = nullptr;
    unsigned long long* new_host_target_states = nullptr;
    try {
      rips_sssp_new::check_hip(
          hipMalloc(reinterpret_cast<void**>(&new_gather_nodes),
                    static_cast<std::size_t>(new_capacity) * sizeof(Index)),
          "hipMalloc Bellman-Ford target nodes");
      rips_sssp_new::check_hip(
          hipMalloc(reinterpret_cast<void**>(&new_gather_states),
                    static_cast<std::size_t>(new_capacity) *
                        sizeof(unsigned long long)),
          "hipMalloc Bellman-Ford target states");
      rips_sssp_new::check_hip(
          hipHostMalloc(reinterpret_cast<void**>(&new_host_target_states),
                        static_cast<std::size_t>(new_capacity) *
                            sizeof(unsigned long long),
                        hipHostMallocDefault),
          "hipHostMalloc Bellman-Ford target states");
    } catch (...) {
      if (new_gather_nodes) (void)hipFree(new_gather_nodes);
      if (new_gather_states) (void)hipFree(new_gather_states);
      if (new_host_target_states) (void)hipHostFree(new_host_target_states);
      throw;
    }

    if (workspace.gather_nodes) (void)hipFree(workspace.gather_nodes);
    if (workspace.gather_states) (void)hipFree(workspace.gather_states);
    if (host_target_states) (void)hipHostFree(host_target_states);
    workspace.gather_nodes = new_gather_nodes;
    workspace.gather_states = new_gather_states;
    host_target_states = new_host_target_states;
    workspace.gather_capacity = new_capacity;
    g_bf9_target_buffer_growth_count.fetch_add(1,
                                               std::memory_order_relaxed);
  }

  void ensure_path_gather_capacity(std::size_t active_path_count) {
    if (active_path_count <= static_cast<std::size_t>(path_gather_capacity)) {
      return;
    }
    if (active_path_count > static_cast<std::size_t>(graph->host.rows)) {
      throw std::invalid_argument(
          "Bellman-Ford active path count exceeds the CSR row count");
    }

    Offset new_capacity = std::max<Offset>(1, path_gather_capacity);
    const Offset required = static_cast<Offset>(active_path_count);
    while (new_capacity < required) {
      if (new_capacity > graph->host.rows / 2) {
        new_capacity = graph->host.rows;
        break;
      }
      new_capacity *= 2;
    }

    Index* new_path_gather_nodes = nullptr;
    unsigned long long* new_path_gather_states = nullptr;
    Index* new_host_path_gather_nodes = nullptr;
    unsigned long long* new_host_path_gather_states = nullptr;
    try {
      rips_sssp_new::check_hip(
          hipMalloc(reinterpret_cast<void**>(&new_path_gather_nodes),
                    static_cast<std::size_t>(new_capacity) * sizeof(Index)),
          "hipMalloc Bellman-Ford path gather nodes");
      rips_sssp_new::check_hip(
          hipMalloc(reinterpret_cast<void**>(&new_path_gather_states),
                    static_cast<std::size_t>(new_capacity) *
                        sizeof(unsigned long long)),
          "hipMalloc Bellman-Ford path gather states");
      rips_sssp_new::check_hip(
          hipHostMalloc(reinterpret_cast<void**>(&new_host_path_gather_nodes),
                        static_cast<std::size_t>(new_capacity) * sizeof(Index),
                        hipHostMallocDefault),
          "hipHostMalloc Bellman-Ford path gather nodes");
      rips_sssp_new::check_hip(
          hipHostMalloc(reinterpret_cast<void**>(&new_host_path_gather_states),
                        static_cast<std::size_t>(new_capacity) *
                            sizeof(unsigned long long),
                        hipHostMallocDefault),
          "hipHostMalloc Bellman-Ford path gather states");
    } catch (...) {
      if (new_path_gather_nodes) (void)hipFree(new_path_gather_nodes);
      if (new_path_gather_states) (void)hipFree(new_path_gather_states);
      if (new_host_path_gather_nodes) {
        (void)hipHostFree(new_host_path_gather_nodes);
      }
      if (new_host_path_gather_states) {
        (void)hipHostFree(new_host_path_gather_states);
      }
      throw;
    }

    if (path_gather_nodes) (void)hipFree(path_gather_nodes);
    if (path_gather_states) (void)hipFree(path_gather_states);
    if (host_path_gather_nodes) (void)hipHostFree(host_path_gather_nodes);
    if (host_path_gather_states) (void)hipHostFree(host_path_gather_states);
    path_gather_nodes = new_path_gather_nodes;
    path_gather_states = new_path_gather_states;
    host_path_gather_nodes = new_host_path_gather_nodes;
    host_path_gather_states = new_host_path_gather_states;
    path_gather_capacity = new_capacity;
    g_bf9_path_buffer_growth_count.fetch_add(1,
                                             std::memory_order_relaxed);
  }

  void gather_target_states(std::size_t target_count) {
    if (target_count == 0 ||
        target_count > static_cast<std::size_t>(workspace.gather_capacity) ||
        !workspace.gather_nodes || !workspace.gather_states ||
        !host_target_states) {
      throw std::runtime_error(
          "Bellman-Ford target-state gather buffers are not allocated");
    }
    rips_sssp_new::gather_packed_states_to_host(
        workspace,
        workspace.gather_nodes,
        workspace.gather_states,
        host_target_states,
        static_cast<Offset>(target_count),
        stream,
        "Bellman-Ford target-state gather");
  }

  std::vector<std::vector<PathEdge>> reconstruct_candidate_paths(
      int source,
      const std::vector<std::size_t>& candidate_target_indices,
      const std::vector<int>& unique_targets,
      const unsigned long long* gathered_target_states) {
    std::vector<std::vector<PathEdge>> completed_paths(
        candidate_target_indices.size());
    if (candidate_target_indices.empty()) {
      return completed_paths;
    }
    if (!gathered_target_states) {
      throw std::runtime_error(
          "Bellman-Ford gathered target states are not allocated");
    }

    std::vector<BellmanFordPendingPath> paths;
    paths.reserve(candidate_target_indices.size());
    for (const std::size_t unique_target_index : candidate_target_indices) {
      if (unique_target_index >= unique_targets.size()) {
        throw std::logic_error(
            "Bellman-Ford candidate target index is outside unique targets");
      }
      const int target = unique_targets[unique_target_index];
      if (target == source) {
        throw std::logic_error(
            "Bellman-Ford identity path reached predecessor reconstruction");
      }
      BellmanFordPendingPath path;
      path.unique_target_index = unique_target_index;
      path.target = target;
      path.current = target;
      path.seen_nodes.reserve(16);
      path.seen_nodes.insert(target);
      paths.push_back(std::move(path));
    }

    auto advance_path = [&](BellmanFordPendingPath& path,
                            unsigned long long state) {
      const Offset edge = unpack_state_edge(state);
      if (edge == static_cast<Offset>(kPackedNoPredEdge) ||
          edge < 0 || edge >= graph->host.nnz ||
          graph->host.to[static_cast<std::size_t>(edge)] != path.current) {
        path.needs_full_state_fallback = true;
        return false;
      }

      const int predecessor =
          graph->from_for_edge[static_cast<std::size_t>(edge)];
      if (!valid_node(graph->host, predecessor)) {
        path.needs_full_state_fallback = true;
        return false;
      }

      path.reversed_edges.push_back(
          {predecessor,
           path.current,
           edge,
           graph->host.values[static_cast<std::size_t>(edge)]});
      if (predecessor == source) {
        path.complete = true;
        return false;
      }
      if (!path.seen_nodes.insert(predecessor).second) {
        path.needs_full_state_fallback = true;
        return false;
      }
      path.current = predecessor;
      return true;
    };

    // The compact target gather is already the first predecessor-depth batch.
    // Consume those packed states directly instead of uploading and copying the
    // same target nodes again through path scratch.
    std::vector<std::size_t> active_paths;
    active_paths.reserve(paths.size());
    for (std::size_t i = 0; i < paths.size(); ++i) {
      const std::size_t unique_target_index =
          paths[i].unique_target_index;
      if (advance_path(paths[i],
                       gathered_target_states[unique_target_index])) {
        active_paths.push_back(i);
      }
    }
    ensure_path_gather_capacity(active_paths.size());
    std::vector<std::size_t> gather_paths;
    std::vector<std::size_t> next_active_paths;
    gather_paths.reserve(paths.size());
    next_active_paths.reserve(paths.size());

    // Depth zero above consumed one predecessor; allow at most graph.rows total
    // predecessor steps before treating a nonterminating chain as malformed.
    const Offset remaining_step_limit = graph->host.rows - 1;
    for (Offset depth = 0;
         depth < remaining_step_limit && !active_paths.empty();
         ++depth) {
      gather_paths.clear();
      for (const std::size_t path_index : active_paths) {
        BellmanFordPendingPath& path = paths[path_index];
        if (!valid_node(graph->host, path.current)) {
          path.needs_full_state_fallback = true;
          continue;
        }
        host_path_gather_nodes[gather_paths.size()] = path.current;
        gather_paths.push_back(path_index);
      }
      if (gather_paths.empty()) {
        active_paths.clear();
        break;
      }

      rips_sssp_new::check_hip(
          hipMemcpyAsync(path_gather_nodes,
                         host_path_gather_nodes,
                         gather_paths.size() * sizeof(Index),
                         hipMemcpyHostToDevice,
                         stream),
          "copy Bellman-Ford path gather nodes");
      rips_sssp_new::gather_packed_states_to_host(
          workspace,
          path_gather_nodes,
          path_gather_states,
          host_path_gather_states,
          static_cast<Offset>(gather_paths.size()),
          stream,
          "Bellman-Ford predecessor-state gather");

      next_active_paths.clear();
      for (std::size_t gather_index = 0;
           gather_index < gather_paths.size();
           ++gather_index) {
        const std::size_t path_index = gather_paths[gather_index];
        BellmanFordPendingPath& path = paths[path_index];
        if (advance_path(path, host_path_gather_states[gather_index])) {
          next_active_paths.push_back(path_index);
        }
      }
      active_paths.swap(next_active_paths);
    }

    for (const std::size_t path_index : active_paths) {
      paths[path_index].needs_full_state_fallback = true;
    }

    bool needs_full_state_fallback = false;
    for (std::size_t i = 0; i < paths.size(); ++i) {
      BellmanFordPendingPath& path = paths[i];
      if (!path.complete) {
        path.needs_full_state_fallback = true;
      }
      if (path.needs_full_state_fallback) {
        needs_full_state_fallback = true;
        continue;
      }
      std::reverse(path.reversed_edges.begin(), path.reversed_edges.end());
      completed_paths[i] = std::move(path.reversed_edges);
    }

    if (needs_full_state_fallback) {
      PATHFINDER_PROFILE_RANGE("bf9.full_state_fallback");
      g_bf9_full_state_fallback_count.fetch_add(1,
                                                std::memory_order_relaxed);
      // The full distance field is needed only by the deterministic
      // distance-tight repair. Cache one copy for every affected candidate of
      // this source; never copy it separately per target.
      const std::vector<unsigned long long> full_states =
          copy_best_states_to_host(workspace, graph->host.rows, stream);
      for (std::size_t i = 0; i < paths.size(); ++i) {
        const BellmanFordPendingPath& path = paths[i];
        if (!path.needs_full_state_fallback) {
          continue;
        }
        completed_paths[i] = reconstruct_bellman_ford_adapter_path(
            graph->host,
            graph->from_for_edge,
            full_states,
            source,
            path.target);
      }
    }
    return completed_paths;
  }
};

BellmanFordCsrWorkspace::BellmanFordCsrWorkspace(const HostCsrF32& adjacency,
                                                 hipStream_t stream)
    : impl_(std::make_unique<Impl>(adjacency, stream)) {}

BellmanFordCsrWorkspace::BellmanFordCsrWorkspace(
    std::shared_ptr<const BellmanFordCsrGraph> adjacency,
    hipStream_t stream)
    : impl_(std::make_unique<Impl>(adjacency, stream)) {}

BellmanFordCsrWorkspace::~BellmanFordCsrWorkspace() = default;
BellmanFordCsrWorkspace::BellmanFordCsrWorkspace(
    BellmanFordCsrWorkspace&&) noexcept = default;
BellmanFordCsrWorkspace& BellmanFordCsrWorkspace::operator=(
    BellmanFordCsrWorkspace&&) noexcept = default;

BellmanFordCsrResult BellmanFordCsrWorkspace::run(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  PATHFINDER_PROFILE_RANGE("bf9.run");
  (void)delta;
  if (!impl_) {
    throw std::runtime_error("BellmanFordCsrWorkspace has no implementation");
  }
  if (progress_callback) {
    throw std::invalid_argument(
        "Bellman-Ford progress callbacks are not available without changing "
        "the bf9 iteration loop");
  }
  (void)progress_user_data;
  impl_->require_stream(stream);
  const HostOutgoingCsrF32& graph = impl_->graph->host;
  const std::vector<int> unique_sources =
      deduplicate_bellman_ford_sources(sources, graph.rows);
  const std::vector<int> unique_targets =
      validate_and_deduplicate_bellman_ford_targets(targets, graph.rows);
  impl_->ensure_target_capacity(unique_targets.size());

  std::unordered_map<int, std::size_t> unique_target_index;
  unique_target_index.reserve(unique_targets.size() * 2 + 1);
  for (std::size_t index = 0; index < unique_targets.size(); ++index) {
    unique_target_index.emplace(unique_targets[index], index);
  }
  std::vector<std::size_t> target_to_unique_index;
  target_to_unique_index.reserve(targets.size());
  for (const int target : targets) {
    const auto found = unique_target_index.find(target);
    if (found == unique_target_index.end()) {
      throw std::logic_error(
          "Bellman-Ford validated target is missing from the unique target map");
    }
    target_to_unique_index.push_back(found->second);
  }

  // Duplicate target positions share one candidate and one reconstruction;
  // the compact result builder below fans that candidate back out in the
  // caller's original target order.
  std::vector<BellmanFordTargetCandidate> best(unique_targets.size());
  BellmanFordCsrResult result;
  result.target = -1;
  result.converged = true;
  bool any_distance_early_stop = false;
  bool any_max_iteration_stop = false;
  bool target_nodes_uploaded = false;

  for (const int source : unique_sources) {
    const rips_sssp_new::OutgoingSsspResult source_result =
        rips_sssp_new::run_outgoing_frontier_sssp_distance_early_stop(
            impl_->graph->device.view,
            impl_->workspace,
            source,
            unique_targets,
            max_iters,
            true,
            stream,
            nullptr,
            !target_nodes_uploaded);
    target_nodes_uploaded = true;
    result.iterations_used = saturated_iteration_sum(
        result.iterations_used, source_result.iterations_used);
    result.converged = result.converged && source_result.converged;
    any_distance_early_stop =
        any_distance_early_stop || source_result.early_stopped;
    any_max_iteration_stop =
        any_max_iteration_stop || source_result.hit_max_iters;

    {
      PATHFINDER_PROFILE_RANGE("bf9.gather_targets");
      impl_->gather_target_states(unique_targets.size());
    }

    std::vector<std::size_t> candidate_target_indices;
    candidate_target_indices.reserve(unique_targets.size());
    for (std::size_t target_index = 0;
         target_index < unique_targets.size();
         ++target_index) {
      const int target = unique_targets[target_index];
      const float distance =
          source == target
              ? 0.0f
              : unpack_state_dist(impl_->host_target_states[target_index]);
      if (!bellman_ford_candidate_can_improve(
              distance, source, target, best[target_index])) {
        continue;
      }

      if (source != target) {
        candidate_target_indices.push_back(target_index);
        continue;
      }

      const std::vector<PathEdge> identity_edges;
      if (!prefer_bellman_ford_candidate(
              distance,
              source,
              target,
              identity_edges,
              best[target_index])) {
        continue;
      }

      BellmanFordTargetCandidate candidate;
      candidate.distance = distance;
      candidate.source = source;
      candidate.nodes.push_back(source);
      best[target_index] = std::move(candidate);
    }

    std::vector<std::vector<PathEdge>> candidate_paths;
    {
      PATHFINDER_PROFILE_RANGE("bf9.reconstruct_paths");
      candidate_paths = impl_->reconstruct_candidate_paths(
          source,
          candidate_target_indices,
          unique_targets,
          impl_->host_target_states);
    }
    for (std::size_t candidate_index = 0;
         candidate_index < candidate_target_indices.size();
         ++candidate_index) {
      const std::size_t target_index =
          candidate_target_indices[candidate_index];
      const int target = unique_targets[target_index];
      const float distance =
          unpack_state_dist(impl_->host_target_states[target_index]);
      const std::vector<PathEdge>& edges = candidate_paths[candidate_index];
      if (!prefer_bellman_ford_candidate(
              distance, source, target, edges, best[target_index])) {
        continue;
      }

      BellmanFordTargetCandidate candidate;
      candidate.distance = distance;
      candidate.source = source;
      candidate.nodes = nodes_from_edges(source, edges);
      candidate.edges.reserve(edges.size());
      for (const PathEdge& edge : edges) {
        candidate.edges.push_back(edge.csr_edge);
      }
      best[target_index] = std::move(candidate);
    }
  }

  result.target_distances.assign(
      targets.size(), std::numeric_limits<float>::infinity());
  result.target_sources.assign(targets.size(), -1);
  result.target_path_offsets.assign(targets.size() + 1, 0);
  result.target_edge_offsets.assign(targets.size() + 1, 0);
  result.target_reached = true;

  for (std::size_t target_index = 0;
       target_index < targets.size();
       ++target_index) {
    if (result.target_path_nodes.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        result.target_path_edges.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "Bellman-Ford compact target paths exceed int offsets");
    }
    result.target_path_offsets[target_index] =
        static_cast<int>(result.target_path_nodes.size());
    result.target_edge_offsets[target_index] =
        static_cast<int>(result.target_path_edges.size());

    const BellmanFordTargetCandidate& candidate =
        best[target_to_unique_index[target_index]];
    if (!std::isfinite(candidate.distance)) {
      result.target_reached = false;
      continue;
    }
    result.target_distances[target_index] = candidate.distance;
    result.target_sources[target_index] = candidate.source;
    result.target_path_nodes.insert(result.target_path_nodes.end(),
                                    candidate.nodes.begin(),
                                    candidate.nodes.end());
    result.target_path_edges.insert(result.target_path_edges.end(),
                                    candidate.edges.begin(),
                                    candidate.edges.end());
  }

  if (result.target_path_nodes.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      result.target_path_edges.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Bellman-Ford compact target paths exceed int offsets");
  }
  result.target_path_offsets[targets.size()] =
      static_cast<int>(result.target_path_nodes.size());
  result.target_edge_offsets[targets.size()] =
      static_cast<int>(result.target_path_edges.size());

  // Aggregate status for the repeated-single-source adapter:
  // - iterations_used is the saturated sum of launched protected BF rounds;
  // - converged is true only if every source reached an empty frontier;
  // - stopped_on_target means at least one source used bf9's distance bound
  //   and no constituent source run stopped at the iteration limit.
  result.stopped_on_target =
      any_distance_early_stop && !any_max_iteration_stop;
  return result;
}

BellmanFordCsrResult BellmanFordCsrWorkspace::run(
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  BellmanFordCsrResult result = run(sources,
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

BellmanFordCsrResult BellmanFordCsrWorkspace::run(
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream,
    BellmanFordCsrProgressCallback progress_callback,
    void* progress_user_data) {
  return run(std::vector<int>{source},
             target,
             delta,
             max_iters,
             stream,
             progress_callback,
             progress_user_data);
}

// CPU function. Inputs: command-line arguments. Output: process exit code.
// Purpose: top-level exception boundary for the bf9 executable.
#ifndef BF9_NO_MAIN
int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "[bf9] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
#endif
