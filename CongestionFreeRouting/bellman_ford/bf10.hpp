#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

// PathFinder-facing adapter for bf10's outgoing active-frontier Bellman-Ford.
// The graph is immutable and may be shared by independent, stream-affine
// workspaces. Each workspace preserves bf10's repeated-single-source semantics,
// runs the iteration controller cooperatively when supported, and reconstructs
// improving predecessor chains into deterministic compact GPU buffers.
class BellmanFord10CsrGraph {
 public:
  struct Impl;

  explicit BellmanFord10CsrGraph(const HostCsrF32& adjacency,
                                 hipStream_t stream = nullptr);
  ~BellmanFord10CsrGraph();

  BellmanFord10CsrGraph(const BellmanFord10CsrGraph&) = delete;
  BellmanFord10CsrGraph& operator=(const BellmanFord10CsrGraph&) = delete;
  BellmanFord10CsrGraph(BellmanFord10CsrGraph&&) noexcept;
  BellmanFord10CsrGraph& operator=(BellmanFord10CsrGraph&&) noexcept;

 private:
  // Workspaces retain the immutable backing allocation directly, so moving or
  // replacing the public wrapper cannot invalidate an existing workspace.
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFord10CsrWorkspace;
};

class BellmanFord10CsrWorkspace {
 public:
  struct Impl;

  explicit BellmanFord10CsrWorkspace(const HostCsrF32& adjacency,
                                     hipStream_t stream = nullptr);
  explicit BellmanFord10CsrWorkspace(
      std::shared_ptr<const BellmanFord10CsrGraph> adjacency,
      hipStream_t stream = nullptr);
  ~BellmanFord10CsrWorkspace();

  BellmanFord10CsrWorkspace(const BellmanFord10CsrWorkspace&) = delete;
  BellmanFord10CsrWorkspace& operator=(const BellmanFord10CsrWorkspace&) = delete;
  BellmanFord10CsrWorkspace(BellmanFord10CsrWorkspace&&) noexcept;
  BellmanFord10CsrWorkspace& operator=(BellmanFord10CsrWorkspace&&) noexcept;

  // PathFinder passes a null progress callback. A non-null callback is rejected
  // because bf10's GPU-controlled iteration loop has no per-round host hook.
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
