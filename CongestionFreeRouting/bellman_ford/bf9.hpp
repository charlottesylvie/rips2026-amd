#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

// PathFinder-facing adapter for bf9's outgoing active-frontier Bellman-Ford.
// The graph is immutable and may be shared by independent, stream-affine
// workspaces. Each workspace preserves bf9's single-source kernel sequence and
// combines repeated single-source runs when PathFinder supplies multiple legal
// sources.
class BellmanFordCsrGraph {
 public:
  struct Impl;

  explicit BellmanFordCsrGraph(const HostCsrF32& adjacency,
                               hipStream_t stream = nullptr);
  ~BellmanFordCsrGraph();

  BellmanFordCsrGraph(const BellmanFordCsrGraph&) = delete;
  BellmanFordCsrGraph& operator=(const BellmanFordCsrGraph&) = delete;
  BellmanFordCsrGraph(BellmanFordCsrGraph&&) noexcept;
  BellmanFordCsrGraph& operator=(BellmanFordCsrGraph&&) noexcept;

 private:
  // Workspaces retain the immutable backing allocation directly, so moving or
  // replacing the public wrapper cannot invalidate an existing workspace.
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFordCsrWorkspace;
};

class BellmanFordCsrWorkspace {
 public:
  struct Impl;

  explicit BellmanFordCsrWorkspace(const HostCsrF32& adjacency,
                                   hipStream_t stream = nullptr);
  explicit BellmanFordCsrWorkspace(
      std::shared_ptr<const BellmanFordCsrGraph> adjacency,
      hipStream_t stream = nullptr);
  ~BellmanFordCsrWorkspace();

  BellmanFordCsrWorkspace(const BellmanFordCsrWorkspace&) = delete;
  BellmanFordCsrWorkspace& operator=(const BellmanFordCsrWorkspace&) = delete;
  BellmanFordCsrWorkspace(BellmanFordCsrWorkspace&&) noexcept;
  BellmanFordCsrWorkspace& operator=(BellmanFordCsrWorkspace&&) noexcept;

  // PathFinder passes a null progress callback. A non-null callback is rejected
  // because bf9's optimized host iteration loop has no callback hook.
  // Multi-source calls run the single-source implementation once per
  // stable-deduplicated source. Equal-distance candidates prefer an identity
  // path, then the lower source node id, then the existing packed
  // predecessor-edge ordering.
  // If zero-weight ties make the final packed predecessors cyclic, adapter-only
  // reconstruction falls back to deterministic CSR-order distance-tight edges.
  // iterations_used is the saturated sum across those runs; converged requires
  // every run to exhaust its frontier; stopped_on_target requires at least one
  // distance-bound stop and no max-iteration stop.
  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  BellmanFordCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
