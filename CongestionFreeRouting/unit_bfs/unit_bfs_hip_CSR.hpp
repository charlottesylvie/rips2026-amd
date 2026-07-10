#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

using UnitBfsCsrProgress = BellmanFordCsrProgress;
using UnitBfsCsrProgressCallback = BellmanFordCsrProgressCallback;
using UnitBfsCsrResult = BellmanFordCsrResult;

// Immutable device CSR that can be shared by independent BFS workspaces.
// The graph and every workspace using it must remain on the HIP device that
// was current when the graph was constructed.
class UnitBfsCsrGraph {
 public:
  struct Impl;

  explicit UnitBfsCsrGraph(const HostCsrF32& adjacency,
                           hipStream_t stream = nullptr);
  ~UnitBfsCsrGraph();

  UnitBfsCsrGraph(const UnitBfsCsrGraph&) = delete;
  UnitBfsCsrGraph& operator=(const UnitBfsCsrGraph&) = delete;
  UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept;
  UnitBfsCsrGraph& operator=(UnitBfsCsrGraph&&) noexcept;

 private:
  std::unique_ptr<Impl> impl_;
  friend class UnitBfsCsrWorkspace;
};

class UnitBfsCsrWorkspace {
 public:
  struct Impl;

  explicit UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                               hipStream_t stream = nullptr);
  explicit UnitBfsCsrWorkspace(
      std::shared_ptr<const UnitBfsCsrGraph> adjacency,
      hipStream_t stream = nullptr);
  ~UnitBfsCsrWorkspace();

  UnitBfsCsrWorkspace(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace& operator=(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept;
  UnitBfsCsrWorkspace& operator=(UnitBfsCsrWorkspace&&) noexcept;

  UnitBfsCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  UnitBfsCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  UnitBfsCsrResult run(
      int source,
      int target,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
