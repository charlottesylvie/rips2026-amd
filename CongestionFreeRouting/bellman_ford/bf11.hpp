#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "../interchange/routing_csr_sidecars.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

// Compatibility carrier for low-level callers that do not construct the
// interchange sidecar artifact. Production routing should pass
// routing::interchange::RoutingCsrSidecars directly.
struct BellmanFord11NodeSidecars {
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_costs;
};

// Inclusive destination-node bounds. enabled=false retains the complete graph.
// In a bounded run, known-coordinate nodes must lie in this rectangle;
// missing-coordinate spill resources remain admissible for routing
// completeness. Sources and targets must always have known coordinates and be
// inside an enabled explicit box.
struct BellmanFord11BoundingBox {
  bool enabled = false;
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
};

struct BellmanFord11RunOptions {
  BellmanFord11BoundingBox bounds{};
  // The persistent controller evaluates its exact nonnegative-distance target
  // certificate after every N completed relaxation rounds. N must be positive;
  // larger values can only delay an early stop.
  int target_check_interval = 1;
};

struct BellmanFord11WorkspaceOptions {
  // The legacy-shaped run() overload is unbounded unless this is enabled.
  bool auto_bounds = false;
  std::int32_t auto_margin_x = 3;
  std::int32_t auto_margin_y = 15;
  // Applies only to auto-bounded legacy-shaped runs. Explicit run options never
  // widen silently. Missing-coordinate route-tree sources are still seeded,
  // but a target without coordinates selects an unbounded first run. A bounded
  // miss retries once.
  bool unbounded_fallback = false;
  int target_check_interval = 1;
};

class BellmanFord11CsrGraph {
 public:
  struct Impl;

  BellmanFord11CsrGraph(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr);
  BellmanFord11CsrGraph(const HostCsrF32& adjacency,
                       const BellmanFord11NodeSidecars& sidecars,
                       hipStream_t stream = nullptr);
  ~BellmanFord11CsrGraph();

  BellmanFord11CsrGraph(const BellmanFord11CsrGraph&) = delete;
  BellmanFord11CsrGraph& operator=(const BellmanFord11CsrGraph&) = delete;
  BellmanFord11CsrGraph(BellmanFord11CsrGraph&&) noexcept;
  BellmanFord11CsrGraph& operator=(BellmanFord11CsrGraph&&) noexcept;

 private:
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFord11CsrWorkspace;
};

class BellmanFord11CsrWorkspace {
 public:
  struct Impl;

  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  BellmanFord11CsrWorkspace(
      const HostCsrF32& adjacency,
      const BellmanFord11NodeSidecars& sidecars,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  explicit BellmanFord11CsrWorkspace(
      std::shared_ptr<const BellmanFord11CsrGraph> adjacency,
      hipStream_t stream = nullptr,
      BellmanFord11WorkspaceOptions options = {});
  ~BellmanFord11CsrWorkspace();

  BellmanFord11CsrWorkspace(const BellmanFord11CsrWorkspace&) = delete;
  BellmanFord11CsrWorkspace& operator=(const BellmanFord11CsrWorkspace&) = delete;
  BellmanFord11CsrWorkspace(BellmanFord11CsrWorkspace&&) noexcept;
  BellmanFord11CsrWorkspace& operator=(BellmanFord11CsrWorkspace&&) noexcept;

  // Replace all workspace-local destination congestion multipliers. Costs are
  // copied and completed on the workspace stream before this method returns.
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  // Mutate only the listed workspace-local multipliers. Node IDs must be unique.
  void update_vertex_costs_sparse(const std::vector<int>& nodes,
                                  const std::vector<float>& vertex_costs,
                                  hipStream_t stream = nullptr);

  // Legacy-shaped entry point. It applies this workspace's auto-bound/fallback
  // policy. delta is retained for PathFinder interface compatibility and is not
  // used by Bellman-Ford.
  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      BellmanFordCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  // Exact fixed-box entry point. This never performs an unbounded retry.
  BellmanFordCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      const BellmanFord11RunOptions& run_options,
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
