// Outgoing-CSR Bellman-Ford driver using a local active-frontier relaxation,
// safe target-reachability early stopping, and optional path JSONL.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     SSSP-new/src/bf5.cpp \
//     -o bf5
//
// Run summary-only; JSONL path output is disabled by default:
//   ./bf5 SSSP-new/data/logicnets_jscl.outgoing.csrbin \
//     SSSP-new/data/logicnets_jscl.outgoing.csrbin.ifmeta.bin \
//     bf5_summary.csv \
//     --source-progress-every 100
//
// Target early stopping is enabled by default; pass --no-target-early-stop
// to run the full bf5-named equivalent of bf3.
//
// Add --paths-output SSSP-new/paths/bf5.logicnets_jscl.outgoing.paths.jsonl
// to emit validator-compatible source-to-sink path JSONL.

#include <hip/hip_runtime.h>

#include <algorithm>
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
#include <utility>
#include <vector>

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
  const Offset* rowptr = nullptr;
  const Index* degree = nullptr;
  const Index* to = nullptr;
  const float* values = nullptr;
  const Offset* edge_id = nullptr;
};

struct IterationStatus {
  int next_count = 0;
  int error_status = 0;
};

struct DeviceSsspWorkspace {
  Offset rows = 0;
  Offset gather_capacity = 0;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  int* next_marks = nullptr;
  IterationStatus* iteration_status = nullptr;
  int* can_reach_target = nullptr;
  int* target_relevant_flag = nullptr;
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
  if (!graph.rowptr || !graph.degree) {
    throw std::runtime_error("outgoing CSR rowptr and degree must not be null");
  }
  if (graph.nnz > 0 && (!graph.to || !graph.values || !graph.edge_id)) {
    throw std::runtime_error("outgoing CSR edge arrays must not be null when nnz > 0");
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
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.can_reach_target),
                        static_cast<std::size_t>(rows) * sizeof(int)),
              "hipMalloc SSSP target reachability mask");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.target_relevant_flag),
                        sizeof(int)),
              "hipMalloc SSSP target relevance flag");

    if (gather_capacity > 0) {
      check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_nodes),
                          static_cast<std::size_t>(gather_capacity) * sizeof(Index)),
                "hipMalloc SSSP gather nodes");
      check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_states),
                          static_cast<std::size_t>(gather_capacity) *
                              sizeof(unsigned long long)),
                "hipMalloc SSSP gather states");
    }

    check_hip(hipStreamSynchronize(stream), "synchronize SSSP workspace allocation");
    return workspace;
  } catch (...) {
    if (workspace.best_state) (void)hipFree(workspace.best_state);
    if (workspace.frontier) (void)hipFree(workspace.frontier);
    if (workspace.next_frontier) (void)hipFree(workspace.next_frontier);
    if (workspace.next_marks) (void)hipFree(workspace.next_marks);
    if (workspace.iteration_status) (void)hipFree(workspace.iteration_status);
    if (workspace.can_reach_target) (void)hipFree(workspace.can_reach_target);
    if (workspace.target_relevant_flag) (void)hipFree(workspace.target_relevant_flag);
    if (workspace.gather_nodes) (void)hipFree(workspace.gather_nodes);
    if (workspace.gather_states) (void)hipFree(workspace.gather_states);
    throw;
  }
}

// Hybrid CPU/GPU host helper. Inputs: workspace to release. Output: reset fields.
// Purpose: free allocations created by make_sssp_workspace.
void free_sssp_workspace(DeviceSsspWorkspace* workspace) {
  if (!workspace) return;
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
  if (workspace->can_reach_target) {
    (void)hipFree(workspace->can_reach_target);
    workspace->can_reach_target = nullptr;
  }
  if (workspace->target_relevant_flag) {
    (void)hipFree(workspace->target_relevant_flag);
    workspace->target_relevant_flag = nullptr;
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
// Output: updates best_state and queues improved destinations once for next iteration.
// Purpose: apply one outgoing edge's frontier relaxation candidate.
__device__ __forceinline__ void relax_outgoing_edge(
    Index dst,
    float weight,
    Offset pred_edge,
    Offset rows,
    float from_dist,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    const int* __restrict__ can_reach_target,
    int* __restrict__ target_relevant_flag,
    int target_early_stop_enabled,
    IterationStatus* __restrict__ iteration_status) {
  if (dst < 0 || static_cast<Offset>(dst) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  if (!finite_device_float(weight) || weight < 0.0f) {
    atomicExch(&iteration_status->error_status, 3);
    return;
  }

  const float candidate = from_dist + weight;
  if (!finite_device_float(candidate)) {
    return;
  }

  if (pred_edge < 0 ||
      pred_edge >= static_cast<Offset>(kPackedNoPredEdge)) {
    atomicExch(&iteration_status->error_status, 5);
    return;
  }

  if (atomic_relax_state(&best_state[static_cast<Offset>(dst)],
                         candidate,
                         static_cast<unsigned int>(pred_edge))) {
    const int old_mark = atomicExch(&next_marks[static_cast<Offset>(dst)], mark_token);
    if (old_mark != mark_token) {
      const int slot = atomicAdd(&iteration_status->next_count, 1);
      if (slot < 0 || static_cast<Offset>(slot) >= rows) {
        atomicExch(&iteration_status->error_status, 4);
        return;
      }
      next_frontier[slot] = dst;
      if (target_early_stop_enabled &&
          can_reach_target[static_cast<Offset>(dst)] != 0) {
        atomicExch(target_relevant_flag, 1);
      }
    }
  }
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
  }
}

// GPU kernel. Inputs: iteration status pointer. Output: zeroed per-iteration counters.
// Purpose: reset status before building the next frontier.
__global__ void reset_iteration_status_kernel(IterationStatus* iteration_status) {
  if (logical_thread_id_1d() == 0) {
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
  }
}

// GPU kernel. Inputs: outgoing CSR, current frontier, and packed best_state.
// Output: relaxed best_state and deduplicated next_frontier.
// Purpose: perform one active-frontier Bellman-Ford relaxation step.
__global__ void outgoing_frontier_relax_kernel(
    const Offset* __restrict__ rowptr,
    const Index* __restrict__ degree,
    const Index* __restrict__ to,
    const float* __restrict__ values,
    const Offset* __restrict__ edge_id,
    Offset rows,
    Offset nnz,
    const Index* __restrict__ frontier,
    int frontier_count,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    const int* __restrict__ can_reach_target,
    int* __restrict__ target_relevant_flag,
    int target_early_stop_enabled,
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

  const Offset begin = rowptr[static_cast<Offset>(from)];
  const int deg = degree[static_cast<Offset>(from)];
  const Offset end = begin + static_cast<Offset>(deg);
  if (begin < 0 || deg < 0 || end < begin || end > nnz) {
    atomicExch(&iteration_status->error_status, 2);
    return;
  }

  for (Offset edge = begin; edge < end; ++edge) {
    relax_outgoing_edge(to[edge],
                        values[edge],
                        edge_id[edge],
                        rows,
                        from_dist,
                        mark_token,
                        best_state,
                        next_frontier,
                        next_marks,
                        can_reach_target,
                        target_relevant_flag,
                        target_early_stop_enabled,
                        iteration_status);
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
  check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP initialization");
}

// Hybrid CPU/GPU host wrapper. Inputs: graph, workspace, source, max iterations,
// optional target reachability mask, early-stop switch, stream.
// Output: iteration count and convergence flag; final packed states remain in workspace.
// Purpose: run active-frontier Bellman-Ford until convergence or target-safe stop.
OutgoingSsspResult run_outgoing_frontier_sssp_target_early_stop(
    const DeviceOutgoingCsrF32& graph,
    DeviceSsspWorkspace& workspace,
    int source,
    int max_iters,
    const int* d_can_reach_target,
    bool target_early_stop_enabled,
    hipStream_t stream = nullptr) {
  validate_device_outgoing_csr(graph);
  if (workspace.rows != graph.rows || !workspace.best_state ||
      !workspace.frontier || !workspace.next_frontier || !workspace.next_marks ||
      !workspace.iteration_status) {
    throw std::runtime_error("SSSP frontier workspace does not match outgoing graph");
  }
  if (target_early_stop_enabled &&
      (!d_can_reach_target || !workspace.target_relevant_flag)) {
    throw std::runtime_error("target early stop requires a device reachability mask");
  }
  if (source < 0 || static_cast<Offset>(source) >= graph.rows) {
    throw std::runtime_error("source is outside outgoing graph");
  }
  if (max_iters < 0) {
    max_iters = static_cast<int>(graph.rows) - 1;
  }

  initialize_outgoing_sssp_state(workspace, source, stream);

  int frontier_count = 1;
  OutgoingSsspResult result;
  for (int iter = 0; iter < max_iters && frontier_count > 0; ++iter) {
    hipLaunchKernelGGL(reset_iteration_status_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch outgoing SSSP frontier status reset");
    if (target_early_stop_enabled) {
      check_hip(hipMemsetAsync(workspace.target_relevant_flag, 0, sizeof(int), stream),
                "reset outgoing SSSP target relevance flag");
    }

    const int mark_token = iter + 1;
    hipLaunchKernelGGL(outgoing_frontier_relax_kernel,
                       grid_for_items(frontier_count, kBlockSize),
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
                       workspace.frontier,
                       frontier_count,
                       mark_token,
                       workspace.best_state,
                       workspace.next_frontier,
                       workspace.next_marks,
                       d_can_reach_target,
                       workspace.target_relevant_flag,
                       target_early_stop_enabled ? 1 : 0,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch outgoing SSSP frontier relaxation");

    IterationStatus status;
    int target_relevant = 1;
    check_hip(hipMemcpyAsync(&status,
                             workspace.iteration_status,
                             sizeof(status),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy outgoing SSSP frontier status");
    if (target_early_stop_enabled) {
      check_hip(hipMemcpyAsync(&target_relevant,
                               workspace.target_relevant_flag,
                               sizeof(target_relevant),
                               hipMemcpyDeviceToHost,
                               stream),
                "copy outgoing SSSP target relevance flag");
    }
    check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP frontier iteration");
    if (status.error_status != 0) {
      throw std::runtime_error("outgoing SSSP frontier relaxation saw invalid graph data");
    }

    frontier_count = status.next_count;
    std::swap(workspace.frontier, workspace.next_frontier);
    result.iterations_used = iter + 1;
    if (target_early_stop_enabled &&
        frontier_count > 0 &&
        target_relevant == 0) {
      result.early_stopped = true;
      break;
    }
  }

  result.converged = frontier_count == 0;
  result.hit_max_iters = !result.converged && !result.early_stopped;
  return result;
}

}  // namespace rips_sssp_new

namespace {

using Offset = rips_sssp_new::Offset;
using Index = rips_sssp_new::Index;

constexpr char OUTGOING_CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'O', 'C', 'S', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t MIN_OUTGOING_CSR_VERSION = 3;
constexpr std::uint64_t CURRENT_OUTGOING_CSR_VERSION = 3;
// SSSP-new metadata version 3 is outgoing-edge aligned: edge_attrs[k]
// describes outgoing CSR edge k, not Routing's incoming-oriented edge order.
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 3;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;

struct HostOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> degree;
  std::vector<Index> to;
  std::vector<float> values;
  std::vector<Offset> edge_id;
};

struct ReverseCsr {
  std::vector<Offset> rowptr;
  std::vector<Index> predecessor;
};

struct DeviceOutgoingCsrOwner {
  rips_sssp_new::DeviceOutgoingCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* degree = nullptr;
  Index* to = nullptr;
  float* values = nullptr;
  Offset* edge_id = nullptr;
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

struct TargetReachabilityWorkspace {
  std::vector<int> can_reach_target;
  std::vector<int> queue;
  std::vector<int> touched;
};

struct Options {
  int max_iters = -1;
  bool target_early_stop = true;
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
// Purpose: skip metadata sections bf5 does not need.
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
      graph.degree.size() != static_cast<std::size_t>(graph.rows) ||
      graph.to.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.edge_id.size() != static_cast<std::size_t>(graph.nnz)) {
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
    if (graph.degree[static_cast<std::size_t>(row)] !=
        static_cast<Index>(row_degree)) {
      throw std::runtime_error("outgoing CSR degree array does not match rowptr");
    }
  }

  for (std::size_t edge = 0; edge < graph.to.size(); ++edge) {
    if (graph.to[edge] < 0 || static_cast<Offset>(graph.to[edge]) >= graph.cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (graph.edge_id[edge] < 0 ||
        graph.edge_id[edge] >= static_cast<Offset>(kPackedNoPredEdge)) {
      throw std::runtime_error("outgoing CSR edge id cannot be packed");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
  }
}

// CPU helper. Inputs: host outgoing CSR. Output: reverse adjacency by predecessor.
// Purpose: support per-source target reachability without rebuilding reverse edges.
ReverseCsr build_reverse_csr(const HostOutgoingCsrF32& graph) {
  ReverseCsr reverse;
  reverse.rowptr.assign(static_cast<std::size_t>(graph.rows + 1), 0);
  reverse.predecessor.resize(static_cast<std::size_t>(graph.nnz));

  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const Index dst = graph.to[static_cast<std::size_t>(edge)];
    ++reverse.rowptr[static_cast<std::size_t>(dst) + 1];
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    reverse.rowptr[static_cast<std::size_t>(row + 1)] +=
        reverse.rowptr[static_cast<std::size_t>(row)];
  }

  std::vector<Offset> cursor = reverse.rowptr;
  for (Offset from = 0; from < graph.rows; ++from) {
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
         edge < graph.rowptr[static_cast<std::size_t>(from + 1)];
         ++edge) {
      const Index dst = graph.to[static_cast<std::size_t>(edge)];
      const Offset slot = cursor[static_cast<std::size_t>(dst)]++;
      reverse.predecessor[static_cast<std::size_t>(slot)] = static_cast<Index>(from);
    }
  }

  return reverse;
}

// CPU helper. Inputs: outgoing RIPSOCS1 CSR path. Output: host outgoing CSR arrays.
// Purpose: load the outgoing CSR format consumed by bf5's frontier kernels.
HostOutgoingCsrF32 load_outgoing_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open outgoing CSR file: " + path.string());
  }

  char magic[sizeof(OUTGOING_CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, OUTGOING_CSR_MAGIC, sizeof(OUTGOING_CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS outgoing CSR file");
  }

  const std::uint64_t version = read_u64(in, "outgoing CSR format version");
  const std::uint64_t orientation = read_u64(in, "outgoing CSR orientation");
  if (version < MIN_OUTGOING_CSR_VERSION ||
      version > CURRENT_OUTGOING_CSR_VERSION) {
    throw std::runtime_error("unsupported outgoing CSR format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported outgoing CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "outgoing CSR row count");
  const std::uint64_t cols = read_u64(in, "outgoing CSR column count");
  const std::uint64_t nnz = read_u64(in, "outgoing CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "outgoing CSR rowptr count");
  const std::uint64_t degree_count = read_u64(in, "outgoing CSR degree count");
  const std::uint64_t to_count = read_u64(in, "outgoing CSR destination count");
  const std::uint64_t values_count = read_u64(in, "outgoing CSR values count");
  const std::uint64_t edge_id_count =
      read_u64(in, "outgoing CSR edge-id count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || degree_count != rows ||
      to_count != nnz || values_count != nnz || edge_id_count != nnz) {
    throw std::runtime_error("outgoing CSR header counts are inconsistent");
  }

  HostOutgoingCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  read_array(in, graph.degree, degree_count, "outgoing CSR degree");
  read_array(in, graph.to, to_count, "outgoing CSR destinations");
  read_array(in, graph.values, values_count, "outgoing CSR values");
  read_array(in, graph.edge_id, edge_id_count, "outgoing CSR edge-id map");
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
    const std::size_t rowptr_bytes = host.rowptr.size() * sizeof(Offset);
    const std::size_t degree_bytes = host.degree.size() * sizeof(Index);
    rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr),
                                       rowptr_bytes),
                             "hipMalloc outgoing rowptr");
    rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.degree),
                                       degree_bytes),
                             "hipMalloc outgoing degree");
    rips_sssp_new::check_hip(hipMemcpyAsync(device.rowptr,
                                            host.rowptr.data(),
                                            rowptr_bytes,
                                            hipMemcpyHostToDevice,
                                            stream),
                             "copy outgoing rowptr to device");
    rips_sssp_new::check_hip(hipMemcpyAsync(device.degree,
                                            host.degree.data(),
                                            degree_bytes,
                                            hipMemcpyHostToDevice,
                                            stream),
                             "copy outgoing degree to device");

    if (host.nnz > 0) {
      const std::size_t to_bytes = host.to.size() * sizeof(Index);
      const std::size_t values_bytes = host.values.size() * sizeof(float);
      const std::size_t edge_id_bytes = host.edge_id.size() * sizeof(Offset);
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.to),
                                         to_bytes),
                               "hipMalloc outgoing destinations");
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.values),
                                         values_bytes),
                               "hipMalloc outgoing values");
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.edge_id),
                                         edge_id_bytes),
                               "hipMalloc outgoing edge-id map");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.to,
                                              host.to.data(),
                                              to_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing destinations to device");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.values,
                                              host.values.data(),
                                              values_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing values to device");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.edge_id,
                                              host.edge_id.data(),
                                              edge_id_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing edge-id map to device");
    }

    rips_sssp_new::check_hip(hipStreamSynchronize(stream),
                             "synchronize outgoing CSR copy");
    device.view.rows = host.rows;
    device.view.cols = host.cols;
    device.view.nnz = host.nnz;
    device.view.rowptr = device.rowptr;
    device.view.degree = device.degree;
    device.view.to = device.to;
    device.view.values = device.values;
    device.view.edge_id = device.edge_id;
    return device;
  } catch (...) {
    if (device.rowptr) (void)hipFree(device.rowptr);
    if (device.degree) (void)hipFree(device.degree);
    if (device.to) (void)hipFree(device.to);
    if (device.values) (void)hipFree(device.values);
    if (device.edge_id) (void)hipFree(device.edge_id);
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
  if (device->degree) {
    (void)hipFree(device->degree);
    device->degree = nullptr;
  }
  if (device->to) {
    (void)hipFree(device->to);
    device->to = nullptr;
  }
  if (device->values) {
    (void)hipFree(device->values);
    device->values = nullptr;
  }
  if (device->edge_id) {
    (void)hipFree(device->edge_id);
    device->edge_id = nullptr;
  }
  device->view = {};
}

// CPU helper. Inputs: metadata path and graph nnz. Output: route request metadata.
// Purpose: parse outgoing SSSP-new metadata needed to emit source-to-sink paths.
RoutingMetadata load_routing_metadata(const std::filesystem::path& path,
                                      Offset graph_nnz) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS SSSP-new metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
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

// CPU helper. Inputs: reverse graph, source queries, host graph, reusable workspace.
// Output: workspace.can_reach_target marks nodes that can reach any queried target.
// Purpose: compute the safe target cone for early stopping without per-source allocation.
const std::vector<int>& mark_nodes_that_can_reach_targets(
    const ReverseCsr& reverse,
    const SourceWork& source_work,
    const HostOutgoingCsrF32& graph,
    TargetReachabilityWorkspace& workspace) {
  const std::size_t row_count = static_cast<std::size_t>(graph.rows);
  if (workspace.can_reach_target.size() != row_count) {
    workspace.can_reach_target.assign(row_count, 0);
    workspace.touched.clear();
    workspace.queue.clear();
  } else {
    for (const int node : workspace.touched) {
      workspace.can_reach_target[static_cast<std::size_t>(node)] = 0;
    }
    workspace.touched.clear();
    workspace.queue.clear();
  }

  auto mark = [&](int node) {
    if (!valid_node(graph, node)) {
      return;
    }
    int& value = workspace.can_reach_target[static_cast<std::size_t>(node)];
    if (value != 0) {
      return;
    }
    value = 1;
    workspace.touched.push_back(node);
    workspace.queue.push_back(node);
  };

  for (const Query& query : source_work.queries) {
    mark(query.target);
  }

  for (std::size_t head = 0; head < workspace.queue.size(); ++head) {
    const int node = workspace.queue[head];
    const Offset begin = reverse.rowptr[static_cast<std::size_t>(node)];
    const Offset end = reverse.rowptr[static_cast<std::size_t>(node + 1)];
    for (Offset item = begin; item < end; ++item) {
      mark(reverse.predecessor[static_cast<std::size_t>(item)]);
    }
  }

  return workspace.can_reach_target;
}

// CPU helper. Inputs: packed best_state value. Output: distance as float.
// Purpose: decode bf5's packed predecessor state on the host.
float unpack_state_dist(unsigned long long state) {
  const unsigned int bits = static_cast<unsigned int>(state >> 32);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// CPU helper. Inputs: packed best_state value. Output: outgoing CSR predecessor edge id.
// Purpose: decode the low 32 bits used by bf5's frontier kernels.
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
      if (graph.edge_id[static_cast<std::size_t>(edge)] != edge) {
        throw std::runtime_error(
            "bf5 path output requires outgoing CSR edge_id[k] == k");
      }
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
  return states;
}

// Hybrid CPU/GPU helper. Inputs: host target cone and workspace. Output: copied GPU mask.
// Purpose: give the relaxation kernel a per-source reachability mask without copying frontiers.
void copy_target_reachability_to_device(
    const std::vector<int>& can_reach_target,
    const rips_sssp_new::DeviceSsspWorkspace& workspace,
    hipStream_t stream) {
  if (workspace.rows <= 0 || !workspace.can_reach_target) {
    throw std::runtime_error("SSSP target reachability workspace is not allocated");
  }
  if (can_reach_target.size() != static_cast<std::size_t>(workspace.rows)) {
    throw std::runtime_error("target reachability mask size does not match workspace rows");
  }
  rips_sssp_new::check_hip(
      hipMemcpyAsync(workspace.can_reach_target,
                     can_reach_target.data(),
                     can_reach_target.size() * sizeof(int),
                     hipMemcpyHostToDevice,
                     stream),
      "copy target reachability mask to device");
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
// Purpose: document bf5's path-outputting command line.
void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <outgoing.csrbin> <metadata.ifmeta.bin> [summary.csv] [options]\n\n"
      << "Options:\n"
      << "  --max-iters <n>             Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --no-target-early-stop     Disable safe target-reachability early stopping.\n"
      << "  --paths-output <path>       Enable JSONL path output at this path. Disabled by default.\n"
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
    } else if (arg == "--no-target-early-stop") {
      parsed.options.target_early_stop = false;
    } else if (arg == "--paths-output") {
      parsed.paths_output_path = require_value(&i, argc, argv, "--paths-output");
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
            << "[bf5] sources "
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

// CPU helper. Inputs: SSSP result. Output: human-readable termination reason.
// Purpose: keep early target stopping distinct from full-graph convergence in logs.
const char* termination_label(const rips_sssp_new::OutgoingSsspResult& result) {
  if (result.converged) {
    return "full_convergence";
  }
  if (result.early_stopped) {
    return "early_target_unreachable";
  }
  if (result.hit_max_iters) {
    return "max_iters";
  }
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

// CPU helper. Inputs: output stream, graph, metadata, selected source/query counts.
// Output: one JSONL metadata line.
// Purpose: identify this as outgoing SSSP-new path output for validators and humans.
void write_metadata_record(std::ostream& out,
                           const HostOutgoingCsrF32& graph,
                           const RoutingMetadata& metadata,
                           std::size_t selected_sources,
                           std::uint64_t selected_queries) {
  out << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\""
      << ",\"producer\":\"bf5\""
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

  std::cout << "[bf5] loading outgoing CSR: " << args.csr_path.string() << '\n';
  HostOutgoingCsrF32 graph = load_outgoing_csrbin(args.csr_path);
  std::cout << "[bf5] graph rows=" << graph.rows
            << " nnz=" << graph.nnz
            << " orientation=outgoing(row u stores edges u -> v)\n";

  std::cout << "[bf5] loading metadata: " << args.metadata_path.string() << '\n';
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
  std::cout << "[bf5] raw_metadata_sources=" << work.raw_source_count
            << " invalid_sources=" << work.invalid_source_count
            << " invalid_sinks=" << work.invalid_sink_count
            << " unique_valid_sources=" << work.work.size()
            << " sources_to_run=" << source_count
            << " source_sink_queries=" << selected_queries << '\n';
  std::cout << "[bf5] target_early_stop="
            << (args.options.target_early_stop ? "enabled" : "disabled")
            << '\n';

  std::ofstream summary_out = open_summary_output(args.output_path);
  std::ofstream paths_out;
  if (emit_paths) {
    paths_out = open_paths_output(args.paths_output_path);
    write_metadata_record(paths_out, graph, metadata, source_count, selected_queries);
    std::cout << "[bf5] writing paths JSONL: "
              << args.paths_output_path.string() << '\n';
  } else {
    std::cout << "[bf5] paths JSONL disabled; pass --paths-output <path> to emit paths\n";
  }

  if (source_count == 0) {
    print_source_progress(0, 0, true);
    std::cout << "[bf5] no valid unique sources to run\n";
    if (emit_paths) {
      paths_out.flush();
      if (!paths_out) {
        throw std::runtime_error("failed while writing paths JSONL");
      }
      std::cout << "[bf5] wrote paths: " << args.paths_output_path.string() << '\n';
    }
    return 0;
  }

  ReverseCsr reverse_graph;
  TargetReachabilityWorkspace target_workspace;
  if (args.options.target_early_stop) {
    std::cout << "[bf5] building reverse CSR once for target reachability\n";
    reverse_graph = build_reverse_csr(graph);
    target_workspace.can_reach_target.assign(static_cast<std::size_t>(graph.rows), 0);
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

    std::cout << "[bf5] copying outgoing CSR to GPU once\n";
    d_graph = copy_outgoing_csr_to_device(graph, stream);
    graph_copied = true;

    std::cout << "[bf5] allocating reusable frontier SSSP workspace\n";
    workspace = rips_sssp_new::make_sssp_workspace(graph.rows, 0, stream);
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
      if (args.options.target_early_stop) {
        const std::vector<int>& can_reach_target =
            mark_nodes_that_can_reach_targets(reverse_graph,
                                              source_work,
                                              graph,
                                              target_workspace);
        copy_target_reachability_to_device(can_reach_target, workspace, stream);
      }
      const rips_sssp_new::OutgoingSsspResult result =
          rips_sssp_new::run_outgoing_frontier_sssp_target_early_stop(
              d_graph.view,
              workspace,
              source,
              args.options.max_iters,
              args.options.target_early_stop ? workspace.can_reach_target : nullptr,
              args.options.target_early_stop,
              stream);
      write_summary_record(summary_out, source, result);
      if (result.converged) {
        ++full_convergence_count;
      } else if (result.early_stopped) {
        ++early_stop_count;
      } else if (result.hit_max_iters) {
        ++max_iter_count;
      }

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

      const std::size_t completed = source_index + 1;
      const bool final_source = completed == source_count;
      const bool detailed =
          args.options.source_progress_every > 0 &&
          (completed % args.options.source_progress_every == 0 || final_source);
      print_source_progress(completed, source_count, detailed || final_source);

      if (detailed) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - total_begin).count();
        const double sources_per_sec =
            elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
        std::cout << "[bf5] source " << completed << "/" << source_count
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
      }
    }
    std::cout << "[bf5] termination_counts"
              << " full_convergence=" << full_convergence_count
              << " early_target_unreachable=" << early_stop_count
              << " max_iters=" << max_iter_count << '\n';
    if (emit_paths) {
      paths_out.flush();
      if (!paths_out) {
        throw std::runtime_error("failed while writing paths JSONL");
      }
      std::cout << "[bf5] path_records=" << paths_written
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
        std::cerr << "[bf5] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
    }
    throw;
  }

  const auto total_end = std::chrono::steady_clock::now();
  const double total_seconds =
      std::chrono::duration<double>(total_end - total_begin).count();
  std::cout << "[bf5] completed " << source_count
            << " source SSSP runs"
            << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
            << total_seconds << '\n';
  if (!args.output_path.empty()) {
    std::cout << "[bf5] wrote summary: " << args.output_path.string() << '\n';
  }
  if (emit_paths) {
    std::cout << "[bf5] wrote paths: " << args.paths_output_path.string() << '\n';
  }
  return 0;
}

}  // namespace

// CPU function. Inputs: command-line arguments. Output: process exit code.
// Purpose: top-level exception boundary for the bf5 executable.
int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "[bf5] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
