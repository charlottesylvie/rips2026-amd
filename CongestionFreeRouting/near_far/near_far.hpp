#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

using NearFarCsrProgress = BellmanFordCsrProgress;
using NearFarCsrProgressCallback = BellmanFordCsrProgressCallback;
using NearFarCsrResult = BellmanFordCsrResult;

struct NearFarCsrWorkspaceOptions {
  // Zero selects a device-derived value capped at 16.
  std::size_t queue_shards = 0;
  // Zero selects a CU-, wave-, graph-, and average-degree-derived cap.
  std::size_t urgent_queue_capacity = 0;
};

// Immutable outgoing CSR shared by independent stream-affine workspaces.
class NearFarCsrGraph {
 public:
  struct Impl;

  explicit NearFarCsrGraph(const HostCsrF32& adjacency,
                           hipStream_t stream = nullptr);
  ~NearFarCsrGraph();

  NearFarCsrGraph(const NearFarCsrGraph&) = delete;
  NearFarCsrGraph& operator=(const NearFarCsrGraph&) = delete;
  NearFarCsrGraph(NearFarCsrGraph&&) noexcept;
  NearFarCsrGraph& operator=(NearFarCsrGraph&&) noexcept;

 private:
  std::shared_ptr<const Impl> impl_;
  friend class NearFarCsrWorkspace;
};

class NearFarCsrWorkspace {
 public:
  struct Impl;

  explicit NearFarCsrWorkspace(const HostCsrF32& adjacency,
                               hipStream_t stream = nullptr);
  NearFarCsrWorkspace(const HostCsrF32& adjacency,
                      hipStream_t stream,
                      NearFarCsrWorkspaceOptions options);
  explicit NearFarCsrWorkspace(
      std::shared_ptr<const NearFarCsrGraph> adjacency,
      hipStream_t stream = nullptr);
  NearFarCsrWorkspace(std::shared_ptr<const NearFarCsrGraph> adjacency,
                      hipStream_t stream,
                      NearFarCsrWorkspaceOptions options);
  ~NearFarCsrWorkspace();

  NearFarCsrWorkspace(const NearFarCsrWorkspace&) = delete;
  NearFarCsrWorkspace& operator=(const NearFarCsrWorkspace&) = delete;
  NearFarCsrWorkspace(NearFarCsrWorkspace&&) noexcept;
  NearFarCsrWorkspace& operator=(NearFarCsrWorkspace&&) noexcept;

  // Destination costs are workspace-local. Effective edge weight is
  // edge_weight(u,v) * vertex_cost(v).
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);
  void clear_vertex_costs(hipStream_t stream = nullptr);

  NearFarCsrResult run_distances(
      const std::vector<int>& sources,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      NearFarCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  NearFarCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      NearFarCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  NearFarCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      NearFarCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  NearFarCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      NearFarCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
