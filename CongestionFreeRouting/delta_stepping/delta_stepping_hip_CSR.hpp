#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
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

enum class DeltaSteppingCsrParentMode {
  // Applies only to eligible generic vector-target workspace runs. Unit-BFS,
  // single-target/full-predecessor, full-distance, and raw-device APIs retain
  // their existing implementations.
  kAutomatic,
  kForceLegacy,
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
                            DeltaSteppingCsrParentMode parent_mode)
      : DeltaSteppingCsrWorkspace(adjacency, stream) {
    parent_mode_ = parent_mode;
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
  ~DeltaSteppingCsrWorkspace();

  DeltaSteppingCsrWorkspace(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace& operator=(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace(DeltaSteppingCsrWorkspace&&) noexcept;
  DeltaSteppingCsrWorkspace& operator=(DeltaSteppingCsrWorkspace&&) noexcept;

  void update_values(const std::vector<float>& values,
                     hipStream_t stream = nullptr);

  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

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
      const std::vector<int>& targets,
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
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  DeltaSteppingCsrParentMode parent_mode_ =
      DeltaSteppingCsrParentMode::kAutomatic;
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
