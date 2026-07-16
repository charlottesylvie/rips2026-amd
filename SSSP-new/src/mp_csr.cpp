// Reusable outgoing-CSR min-plus Bellman-Ford relaxation.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip -c SSSP-new/src/mp_csr.cpp -o mp_csr.o
//
// This file intentionally implements the outgoing dense relaxation used by
// SSSP/src/bf_original.cpp, not generic sparse matrix-matrix multiplication.
// The core one-step operation is:
//
//   best_state_next[v] = min(best_state[v],
//                            best_state[u].distance + weight(u -> v))
//
// for every outgoing edge u -> v in an outgoing CSR graph. The packed state is
// identical to bf_original: high 32 bits are the float distance bits, and low
// 32 bits are the predecessor edge id in the outgoing CSR.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace rips_sssp_new {

using Offset = std::int64_t;
using Index = int;

constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;
constexpr unsigned int kPackedInfinityBits = 0x7f800000u;
constexpr unsigned kGridX = 65535u;
constexpr int kBlockSize = 256;

struct DeviceOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;

  // Outgoing CSR orientation: row u stores directed edges u -> v.
  const Offset* rowptr = nullptr;        // length rows + 1
  const Index* degree = nullptr;         // length rows
  const Index* to = nullptr;             // length nnz, destination v
  const float* values = nullptr;         // length nnz, edge weights
  const Offset* edge_id = nullptr;       // length nnz, outgoing CSR edge id
};

struct DeviceSsspWorkspace {
  Offset rows = 0;
  Offset gather_capacity = 0;

  // Packed state per vertex: upper 32 bits distance, lower 32 bits predecessor.
  unsigned long long* best_state = nullptr;
  int* changed_flag = nullptr;
  int* status = nullptr;

  // Optional scratch for gather_best_states_kernel.
  Index* gather_nodes = nullptr;
  unsigned long long* gather_states = nullptr;
};

struct OutgoingSsspResult {
  int iterations_used = 0;
  bool converged = false;
};

// CPU helper. Inputs: HIP status and operation label. Output: none, or throws.
// Purpose: convert HIP errors from host-side launches/copies into exceptions.
void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

// CPU helper. Inputs: item count and block size. Output: a 2D HIP grid.
// Purpose: mirror bf_original's large-1D-launch mapping.
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

// CPU helper. Inputs: float bit pattern. Output: decoded float value.
// Purpose: let host code decode the high 32 bits of packed best_state entries.
float host_float_from_bits(unsigned int bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Hybrid CPU/GPU host helper. Inputs: graph view. Output: none, or throws.
// Purpose: validate the device-pointer graph shape before launching kernels.
void validate_device_outgoing_csr(const DeviceOutgoingCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR expects a nonempty square graph");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (!graph.rowptr || !graph.degree) {
    throw std::runtime_error("outgoing CSR rowptr and degree must not be null");
  }
  if (graph.nnz > 0 && (!graph.to || !graph.values || !graph.edge_id)) {
    throw std::runtime_error("outgoing CSR edge arrays must not be null when nnz > 0");
  }
}

// Hybrid CPU/GPU host helper. Inputs: row count, gather capacity, HIP stream.
// Output: allocated workspace whose pointers live on the GPU.
// Purpose: allocate the dense packed-state vector and optional gather scratch.
DeviceSsspWorkspace make_sssp_workspace(Offset rows,
                                        Offset gather_capacity,
                                        hipStream_t stream = nullptr) {
  if (rows <= 0) {
    throw std::runtime_error("SSSP workspace rows must be positive");
  }
  if (gather_capacity < 0 || gather_capacity > rows) {
    throw std::runtime_error("SSSP gather capacity is outside graph bounds");
  }

  DeviceSsspWorkspace workspace;
  workspace.rows = rows;
  workspace.gather_capacity = gather_capacity;
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.best_state),
                      static_cast<std::size_t>(rows) * sizeof(unsigned long long)),
            "hipMalloc SSSP best state");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.changed_flag), sizeof(int)),
            "hipMalloc SSSP changed flag");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.status), sizeof(int)),
            "hipMalloc SSSP status");

  if (gather_capacity > 0) {
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_nodes),
                        static_cast<std::size_t>(gather_capacity) * sizeof(Index)),
              "hipMalloc SSSP gather nodes");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_states),
                        static_cast<std::size_t>(gather_capacity) *
                            sizeof(unsigned long long)),
              "hipMalloc SSSP gather states");
  }

  check_hip(hipMemsetAsync(workspace.changed_flag, 0, sizeof(int), stream),
            "reset SSSP changed flag");
  check_hip(hipStreamSynchronize(stream), "synchronize SSSP workspace allocation");
  return workspace;
}

// Hybrid CPU/GPU host helper. Inputs: workspace to release. Output: reset fields.
// Purpose: free allocations created by make_sssp_workspace.
void free_sssp_workspace(DeviceSsspWorkspace* workspace) {
  if (!workspace) return;
  if (workspace->best_state) {
    (void)hipFree(workspace->best_state);
    workspace->best_state = nullptr;
  }
  if (workspace->changed_flag) {
    (void)hipFree(workspace->changed_flag);
    workspace->changed_flag = nullptr;
  }
  if (workspace->status) {
    (void)hipFree(workspace->status);
    workspace->status = nullptr;
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
}

// GPU device helper. Inputs: HIP block/thread ids. Output: logical 1D thread id.
// Purpose: support bf_original's 2D-grid linearization.
__device__ __forceinline__ Offset logical_thread_id_1d() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

// GPU device helper. Inputs: distance bits and predecessor edge id. Output: packed state.
// Purpose: preserve bf_original's sortable 64-bit state format.
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
// Purpose: match bf_original's device-side NaN and infinity guard.
__device__ __forceinline__ bool finite_device_float(float value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

// GPU device helper. Inputs: edge destination/weight/id, source distance, state arrays.
// Output: updates best_state, changed_flag, or status in device memory.
// Purpose: apply one outgoing edge's min-plus candidate relaxation.
__device__ __forceinline__ void relax_outgoing_edge(
    Index dst,
    float weight,
    Offset pred_edge,
    Offset rows,
    float from_dist,
    unsigned long long* __restrict__ best_state,
    int* __restrict__ changed_flag,
    int* __restrict__ status) {
  if (dst < 0 || static_cast<Offset>(dst) >= rows) {
    atomicExch(status, 1);
    return;
  }

  if (!finite_device_float(weight) || weight < 0.0f) {
    atomicExch(status, 3);
    return;
  }

  const float candidate = from_dist + weight;
  if (!finite_device_float(candidate)) {
    return;
  }

  if (pred_edge < 0 ||
      pred_edge >= static_cast<Offset>(kPackedNoPredEdge)) {
    atomicExch(status, 5);
    return;
  }

  if (atomic_relax_state(&best_state[static_cast<Offset>(dst)],
                         candidate,
                         static_cast<unsigned int>(pred_edge))) {
    atomicExch(changed_flag, 1);
  }
}

// GPU kernel. Inputs: row count, source node, workspace arrays.
// Output: initializes best_state, changed_flag, and status on the GPU.
// Purpose: set source distance to zero and all other distances to +infinity.
__global__ void init_outgoing_sssp_kernel(Offset rows,
                                          int source,
                                          unsigned long long* best_state,
                                          int* changed_flag,
                                          int* status) {
  const Offset row = logical_thread_id_1d();
  if (row < rows) {
    best_state[row] =
        row == static_cast<Offset>(source)
            ? pack_state_bits(0u, 0u)
            : pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
  }
  if (row == 0) {
    *changed_flag = 0;
    *status = 0;
  }
}

// GPU kernel. Inputs: packed state vector, requested device node ids.
// Output: packed states for requested nodes in device memory.
// Purpose: read selected rows without copying the full dense state vector.
__global__ void gather_best_states_kernel(
    const unsigned long long* __restrict__ best_state,
    Offset rows,
    const Index* __restrict__ nodes,
    Offset node_count,
    unsigned long long* __restrict__ states) {
  const Offset item = logical_thread_id_1d();
  if (item >= node_count) {
    return;
  }

  const Index node = nodes[item];
  states[item] =
      node >= 0 && static_cast<Offset>(node) < rows
          ? best_state[static_cast<Offset>(node)]
          : pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
}

// GPU kernel. Inputs: outgoing CSR arrays and current packed best_state.
// Output: relaxed best_state, changed_flag, and status in device memory.
// Purpose: perform one dense outgoing min-plus matrix-vector relaxation step.
__global__ void outgoing_dense_relax_kernel(
    const Offset* __restrict__ rowptr,
    const Index* __restrict__ degree,
    const Index* __restrict__ to,
    const float* __restrict__ values,
    const Offset* __restrict__ edge_id,
    Offset rows,
    Offset nnz,
    unsigned long long* __restrict__ best_state,
    int* __restrict__ changed_flag,
    int* __restrict__ status) {
  const Offset from_row = logical_thread_id_1d();
  if (from_row >= rows) {
    return;
  }

  const Index from = static_cast<Index>(from_row);
  if (from < 0 || static_cast<Offset>(from) >= rows) {
    atomicExch(status, 1);
    return;
  }

  const float from_dist = unpack_state_dist(best_state[static_cast<Offset>(from)]);
  if (!finite_device_float(from_dist)) {
    return;
  }

  const Offset begin = rowptr[static_cast<Offset>(from)];
  const int deg = degree[static_cast<Offset>(from)];
  const Offset end = begin + static_cast<Offset>(deg);
  if (begin < 0 || deg < 0 || end < begin || end > nnz) {
    atomicExch(status, 2);
    return;
  }

  for (Offset edge = begin; edge < end; ++edge) {
    relax_outgoing_edge(to[edge],
                        values[edge],
                        edge_id[edge],
                        rows,
                        from_dist,
                        best_state,
                        changed_flag,
                        status);
  }
}

// Hybrid CPU/GPU host wrapper. Inputs: workspace, source, stream. Output: initialized GPU state.
// Purpose: launch init_outgoing_sssp_kernel for a Bellman-Ford source.
void initialize_outgoing_sssp_state(DeviceSsspWorkspace& workspace,
                                    int source,
                                    hipStream_t stream = nullptr) {
  if (workspace.rows <= 0 || !workspace.best_state ||
      !workspace.changed_flag || !workspace.status) {
    throw std::runtime_error("SSSP workspace is not allocated");
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
                     workspace.changed_flag,
                     workspace.status);
  check_hip(hipGetLastError(), "launch outgoing SSSP initialization");
  check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP initialization");
}

// Hybrid CPU/GPU host wrapper. Inputs: graph, initialized workspace, stream.
// Output: true if any distance changed during this one relaxation step.
// Purpose: expose the reusable min-plus matrix-vector relaxation iteration.
bool run_outgoing_dense_relaxation_iteration(const DeviceOutgoingCsrF32& graph,
                                             DeviceSsspWorkspace& workspace,
                                             hipStream_t stream = nullptr) {
  validate_device_outgoing_csr(graph);
  if (workspace.rows != graph.rows || !workspace.best_state ||
      !workspace.changed_flag || !workspace.status) {
    throw std::runtime_error("SSSP workspace does not match outgoing graph");
  }

  check_hip(hipMemsetAsync(workspace.changed_flag, 0, sizeof(int), stream),
            "reset outgoing SSSP changed flag");
  check_hip(hipMemsetAsync(workspace.status, 0, sizeof(int), stream),
            "reset outgoing SSSP status");

  hipLaunchKernelGGL(outgoing_dense_relax_kernel,
                     grid_for_items(graph.rows, kBlockSize),
                     dim3(kBlockSize),
                     0,
                     stream,
                     graph.rowptr,
                     graph.degree,
                     graph.to,
                     graph.values,
                     graph.edge_id,
                     graph.rows,
                     graph.nnz,
                     workspace.best_state,
                     workspace.changed_flag,
                     workspace.status);
  check_hip(hipGetLastError(), "launch outgoing SSSP relaxation");

  int changed_flag = 0;
  int h_status = 0;
  check_hip(hipMemcpyAsync(&changed_flag,
                           workspace.changed_flag,
                           sizeof(int),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy outgoing SSSP changed flag");
  check_hip(hipMemcpyAsync(&h_status,
                           workspace.status,
                           sizeof(int),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy outgoing SSSP status");
  check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP iteration");
  if (h_status != 0) {
    throw std::runtime_error("outgoing SSSP relaxation saw invalid graph data");
  }
  return changed_flag != 0;
}

// Hybrid CPU/GPU host wrapper. Inputs: workspace, device nodes, count, output states, stream.
// Output: requested packed states in d_states.
// Purpose: launch gather_best_states_kernel for callers that need selected distances/predecessors.
void gather_best_states(const DeviceSsspWorkspace& workspace,
                        const Index* d_nodes,
                        Offset node_count,
                        unsigned long long* d_states,
                        hipStream_t stream = nullptr) {
  if (workspace.rows <= 0 || !workspace.best_state) {
    throw std::runtime_error("SSSP workspace is not allocated");
  }
  if (node_count < 0 || node_count > workspace.rows) {
    throw std::runtime_error("gather node count is outside graph bounds");
  }
  if (node_count > 0 && (!d_nodes || !d_states)) {
    throw std::runtime_error("gather input and output device pointers must not be null");
  }
  if (node_count == 0) {
    return;
  }

  hipLaunchKernelGGL(gather_best_states_kernel,
                     grid_for_items(node_count, kBlockSize),
                     dim3(kBlockSize),
                     0,
                     stream,
                     workspace.best_state,
                     workspace.rows,
                     d_nodes,
                     node_count,
                     d_states);
  check_hip(hipGetLastError(), "launch sparse SSSP state gather");
}

// Hybrid CPU/GPU host wrapper. Inputs: graph, workspace, source, max_iters, stream.
// Output: iteration count and convergence flag; final packed states remain in workspace.
// Purpose: compatibility loop equivalent to bf_original's run_outgoing_dense_sssp.
OutgoingSsspResult run_outgoing_dense_sssp(const DeviceOutgoingCsrF32& graph,
                                           DeviceSsspWorkspace& workspace,
                                           int source,
                                           int max_iters,
                                           hipStream_t stream = nullptr) {
  validate_device_outgoing_csr(graph);
  if (workspace.rows != graph.rows) {
    throw std::runtime_error("SSSP workspace row count does not match graph");
  }
  if (source < 0 || static_cast<Offset>(source) >= graph.rows) {
    throw std::runtime_error("source is outside outgoing graph");
  }
  if (max_iters < 0) {
    max_iters = static_cast<int>(graph.rows) - 1;
  }

  initialize_outgoing_sssp_state(workspace, source, stream);

  bool changed = true;
  OutgoingSsspResult result;
  for (int iter = 0; iter < max_iters && changed; ++iter) {
    changed = run_outgoing_dense_relaxation_iteration(graph, workspace, stream);
    result.iterations_used = iter + 1;
  }

  result.converged = !changed;
  return result;
}

}  // namespace rips_sssp_new
