#pragma once

#include "bf_hip_CSR.hpp"
#include "delta_stepping_auto_delta.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// Delta-Stepping SSSP for nonnegative edge weights over outgoing-edge CSR:
//   adjacency row u, column v = weight of directed edge u -> v.
// The converter emits this orientation directly so the kernel can traverse
// frontiers without an O(E) GPU transpose.
// Unlimited multi-target workspace runs with exact unit weights and no vertex
// costs use an equivalent append-only traversal specialized for that case.
using DeltaSteppingCsrProgress = BellmanFordCsrProgress;
using DeltaSteppingCsrProgressCallback = BellmanFordCsrProgressCallback;
using DeltaSteppingCsrResult = BellmanFordCsrResult;

enum class DeltaSteppingCsrExecutionPath {
  kNotRun,
  kExactUnit,
  kCompactGeneric,
  kLegacyGeneric,
  kGenericDistancesOnly,
};

// Per-invocation telemetry. Every counter is exact under the semantics below;
// pending-entry counters count examinations and may therefore count one token
// more than once across scheduler scans. Queue peaks count entries, not bytes.
struct DeltaSteppingCsrTelemetry {
  bool collected = false;
  bool completed = false;
  DeltaSteppingCsrExecutionPath execution_path =
      DeltaSteppingCsrExecutionPath::kNotRun;
  float resolved_delta = 0.0f;
  int wavefront_size = 0;
  bool force_generic = false;
  bool force_legacy_parent = false;
  bool has_vertex_costs = false;
  bool all_edges_light = false;
  std::uint64_t outer_buckets_processed = 0;
  std::uint64_t light_relaxation_rounds = 0;
  std::uint64_t heavy_edge_phases = 0;
  std::uint64_t frontier_entries_processed = 0;
  std::uint64_t active_vertices_processed = 0;
  std::uint64_t stale_frontier_entries = 0;
  std::uint64_t light_edge_visits = 0;
  std::uint64_t heavy_edge_visits = 0;
  std::uint64_t distance_atomic_attempts = 0;
  std::uint64_t successful_distance_relaxations = 0;
  std::uint64_t distance_cas_retries = 0;
  std::uint64_t current_queue_insertions = 0;
  std::uint64_t pending_queue_insertions = 0;
  std::uint64_t heavy_queue_insertions = 0;
  std::uint64_t bucket_insertions = 0;
  std::uint64_t pending_entry_examinations = 0;
  std::uint64_t stale_pending_entry_examinations = 0;
  std::uint64_t reached_vertices = 0;
  std::uint64_t current_queue_high_water = 0;
  std::uint64_t pending_queue_high_water = 0;
  std::uint64_t heavy_queue_high_water = 0;
  std::uint64_t controller_round_trips = 0;
  std::uint64_t compact_parent_fallback_events = 0;
};

struct DeltaSteppingCsrRunOptions {
  // Null keeps telemetry completely disabled. A nonnull record is reset before
  // dispatch and remains completed=false if the invocation throws.
  DeltaSteppingCsrTelemetry* telemetry = nullptr;
};

const char* delta_stepping_execution_path_name(
    DeltaSteppingCsrExecutionPath path) noexcept;

enum class DeltaSteppingCsrParentMode {
  // Applies to vector-target workspace runs. Single-target/full-predecessor,
  // explicit distances-only, and raw-device APIs retain their own policies.
  kAutomatic,
  // Bypass exact-unit dispatch and use the generic scheduler with legacy
  // predecessor-node/edge recovery. Incompatible with run_distances().
  kForceLegacy,
};

enum class DeltaSteppingCsrExecutionMode {
  // Preserve the existing exact-unit specialization whenever all of its
  // correctness guards are satisfied.
  kAutomatic,
  // Deliberately bypass the exact-unit specialization without changing edge
  // weights, destination costs, delta, or parent representation.
  kForceGeneric,
};

enum class DeltaSteppingCsrStorageMode {
  // Retain the immutable edge-to-source map needed by compact path recovery.
  kPathCapable,
  // Omit path-only immutable storage. Only run_distances() is valid on a
  // workspace backed by this graph/storage mode.
  kDistancesOnly,
};

struct DeltaSteppingCsrAllocationState {
  // True means a nonzero device allocation is resident. An eligible zero-edge
  // graph therefore reports edge_source=false even in path-capable mode.
  bool edge_source = false;
  bool parent_key = false;
  bool predecessor_nodes = false;
  bool predecessor_edges = false;
  bool target_storage = false;
  bool path_nodes = false;
  bool path_edges = false;
  bool telemetry_counters = false;
};

struct DeltaSteppingCsrWorkspaceOptions {
  DeltaSteppingCsrParentMode parent_mode =
      DeltaSteppingCsrParentMode::kAutomatic;
  DeltaSteppingCsrExecutionMode execution_mode =
      DeltaSteppingCsrExecutionMode::kAutomatic;
};

// A graph with 2^32 edges is still eligible: its largest original CSR edge ID
// is UINT32_MAX. Keep this helper allocation-free so boundary behavior can be
// validated without constructing an impractically large graph.
constexpr bool delta_stepping_compact_edge_ids_eligible(
    minplus_sparse::Offset nnz) noexcept {
  return nnz >= 0 &&
         static_cast<std::uint64_t>(nnz) <=
             (std::uint64_t{1} << std::numeric_limits<std::uint32_t>::digits);
}

// Immutable device CSR that can be shared by independent Delta-Stepping
// workspaces. The graph and every workspace using it must remain on the HIP
// device that was current when the graph was constructed.
class DeltaSteppingCsrGraph {
 public:
  struct Impl;

  explicit DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream = nullptr);
  DeltaSteppingCsrGraph(const HostCsrF32& adjacency,
                        hipStream_t stream,
                        DeltaSteppingCsrStorageMode storage_mode);
  ~DeltaSteppingCsrGraph();

  DeltaSteppingCsrGraph(const DeltaSteppingCsrGraph&) = delete;
  DeltaSteppingCsrGraph& operator=(const DeltaSteppingCsrGraph&) = delete;
  DeltaSteppingCsrGraph(DeltaSteppingCsrGraph&&) noexcept;
  DeltaSteppingCsrGraph& operator=(DeltaSteppingCsrGraph&&) noexcept;

 private:
  // Workspaces retain this immutable backing allocation directly, so moving
  // or replacing the public graph wrapper cannot invalidate live workspaces.
  std::shared_ptr<const Impl> impl_;
  friend class DeltaSteppingCsrWorkspace;
};

class DeltaSteppingCsrWorkspace {
 public:
  struct Impl;

  // A workspace is stream- and device-affine: construction, updates, and
  // every run must use the same stream handle while its construction device is
  // current. Separate workspaces may use separate streams.
  explicit DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                                     hipStream_t stream = nullptr);
  DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                            hipStream_t stream,
                            DeltaSteppingCsrStorageMode storage_mode);
  DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                            hipStream_t stream,
                            DeltaSteppingCsrParentMode parent_mode)
      : DeltaSteppingCsrWorkspace(adjacency, stream) {
    parent_mode_ = parent_mode;
  }
  DeltaSteppingCsrWorkspace(
      const HostCsrF32& adjacency,
      hipStream_t stream,
      DeltaSteppingCsrWorkspaceOptions options)
      : DeltaSteppingCsrWorkspace(adjacency, stream) {
    parent_mode_ = options.parent_mode;
    execution_mode_ = options.execution_mode;
  }
  // Shared-graph workspaces keep private mutable search state but reuse the
  // immutable CSR. update_values() is intentionally unavailable for this form;
  // update_vertex_costs() remains workspace-local and is supported.
  explicit DeltaSteppingCsrWorkspace(
      std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
      hipStream_t stream = nullptr);
  DeltaSteppingCsrWorkspace(
      std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
      hipStream_t stream,
      DeltaSteppingCsrParentMode parent_mode)
      : DeltaSteppingCsrWorkspace(adjacency, stream) {
    parent_mode_ = parent_mode;
  }
  DeltaSteppingCsrWorkspace(
      std::shared_ptr<const DeltaSteppingCsrGraph> adjacency,
      hipStream_t stream,
      DeltaSteppingCsrWorkspaceOptions options)
      : DeltaSteppingCsrWorkspace(std::move(adjacency), stream) {
    parent_mode_ = options.parent_mode;
    execution_mode_ = options.execution_mode;
  }
  ~DeltaSteppingCsrWorkspace();

  DeltaSteppingCsrWorkspace(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace& operator=(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace(DeltaSteppingCsrWorkspace&&) noexcept;
  DeltaSteppingCsrWorkspace& operator=(DeltaSteppingCsrWorkspace&&) noexcept;

  void update_values(const std::vector<float>& values,
                     hipStream_t stream = nullptr);

  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  // Compile-time no-parent Delta-Stepping specialization. Only dist is
  // populated in the result; predecessor and compact target/path vectors stay
  // empty. Entering this mode releases any path-only mutable workspace state.
  DeltaSteppingCsrResult run_distances(
      const std::vector<int>& sources,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingCsrResult run_distances(
      const std::vector<int>& sources,
      float delta,
      int max_iters,
      DeltaSteppingCsrRunOptions run_options,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_with_telemetry(run_options, [&] {
      return run_distances(sources, delta, max_iters, stream,
                           progress_callback, progress_user_data);
    });
  }

  DeltaSteppingCsrResult run_distances(
      int source,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_distances(std::vector<int>{source}, delta, max_iters, stream,
                         progress_callback, progress_user_data);
  }

  DeltaSteppingCsrResult run_distances(
      int source,
      float delta,
      int max_iters,
      DeltaSteppingCsrRunOptions run_options,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_with_telemetry(run_options, [&] {
      return run_distances(source, delta, max_iters, stream,
                           progress_callback, progress_user_data);
    });
  }

  DeltaSteppingCsrAllocationState allocation_state() const noexcept;

  DeltaSteppingCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      DeltaSteppingCsrRunOptions run_options,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_with_telemetry(run_options, [&] {
      return run(sources, target, delta, max_iters, stream,
                 progress_callback, progress_user_data);
    });
  }

  DeltaSteppingCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      DeltaSteppingCsrRunOptions run_options,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_with_telemetry(run_options, [&] {
      return run(sources, targets, delta, max_iters, stream,
                 progress_callback, progress_user_data);
    });
  }

  DeltaSteppingCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      DeltaSteppingCsrRunOptions run_options,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) {
    return run_with_telemetry(run_options, [&] {
      return run(source, target, delta, max_iters, stream,
                 progress_callback, progress_user_data);
    });
  }

 private:
  template <typename Run>
  DeltaSteppingCsrResult run_with_telemetry(
      DeltaSteppingCsrRunOptions run_options,
      Run&& run) {
    if (active_telemetry_ != nullptr) {
      throw std::logic_error(
          "DeltaSteppingCsrWorkspace does not support nested telemetry runs");
    }
    if (run_options.telemetry != nullptr) {
      *run_options.telemetry = DeltaSteppingCsrTelemetry{};
    }
    active_telemetry_ = run_options.telemetry;
    try {
      DeltaSteppingCsrResult result = run();
      active_telemetry_ = nullptr;
      return result;
    } catch (...) {
      active_telemetry_ = nullptr;
      throw;
    }
  }

  DeltaSteppingCsrParentMode parent_mode_ =
      DeltaSteppingCsrParentMode::kAutomatic;
  DeltaSteppingCsrExecutionMode execution_mode_ =
      DeltaSteppingCsrExecutionMode::kAutomatic;
  DeltaSteppingCsrTelemetry* active_telemetry_ = nullptr;
  std::unique_ptr<Impl> impl_;
};

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream = nullptr);

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, source, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}
