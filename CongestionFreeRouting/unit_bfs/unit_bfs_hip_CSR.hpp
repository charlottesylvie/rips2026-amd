#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "../sssp_query_capacity.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

using UnitBfsCsrProgress = BellmanFordCsrProgress;
using UnitBfsCsrProgressCallback = BellmanFordCsrProgressCallback;
using UnitBfsCsrResult = BellmanFordCsrResult;

// Automatic mode uses 32-bit device row offsets and predecessor-edge IDs when
// nnz fits in a signed 32-bit value.  The forced-wide mode is primarily useful
// for correctness/performance A/B tests; public path edge IDs remain 64-bit in
// both modes.
enum class UnitBfsCsrOffsetMode {
  kAuto,
  kForce64Bit,
};

// The established host-prefix extraction and sparse-reset visitation paths
// remain the defaults until the opt-in paths have been validated on AMD HIP.
enum class UnitBfsCsrExtractionMode {
  kHostOffsets,
  kDeviceOffsets,
};

enum class UnitBfsCsrVisitationMode {
  kSparseReset,
  kGenerationStamped,
};

struct UnitBfsCsrWorkspaceOptions {
  UnitBfsCsrExtractionMode extraction_mode =
      UnitBfsCsrExtractionMode::kHostOffsets;
  UnitBfsCsrVisitationMode visitation_mode =
      UnitBfsCsrVisitationMode::kSparseReset;
  // Zero-valued fields are equivalent to no reservation hint. Hints reserve
  // only source/target-derived buffers; compact paths remain demand-driven.
  SsspQueryCapacityHints capacity_hints{};
};

struct UnitBfsCsrAllocationState {
  std::size_t source_capacity = 0;
  std::size_t target_capacity = 0;
  std::size_t target_metadata_capacity = 0;
  std::size_t target_offset_capacity = 0;
  std::size_t compact_path_node_capacity = 0;
  std::size_t compact_path_edge_capacity = 0;
};

// Immutable device CSR that can be shared by independent BFS workspaces.
// The graph and every workspace using it must remain on the HIP device that
// was current when the graph was constructed.
class UnitBfsCsrGraph {
 public:
  struct Impl;

  explicit UnitBfsCsrGraph(const HostCsrF32& adjacency,
                           hipStream_t stream = nullptr);
  UnitBfsCsrGraph(const HostCsrF32& adjacency,
                  hipStream_t stream,
                  UnitBfsCsrOffsetMode offset_mode);
  ~UnitBfsCsrGraph();

  UnitBfsCsrGraph(const UnitBfsCsrGraph&) = delete;
  UnitBfsCsrGraph& operator=(const UnitBfsCsrGraph&) = delete;
  UnitBfsCsrGraph(UnitBfsCsrGraph&&) noexcept;
  UnitBfsCsrGraph& operator=(UnitBfsCsrGraph&&) noexcept;

  // A moved-from graph has no storage and reports false.
  bool uses_32_bit_offsets() const noexcept;

 private:
  std::shared_ptr<Impl> impl_;
  friend class UnitBfsCsrWorkspace;
};

class UnitBfsCsrWorkspace {
 public:
  struct Impl;

  // A workspace is bound to its construction stream and HIP device. Every run
  // must use that same stream; independent streams require independent
  // workspaces, while the immutable UnitBfsCsrGraph may still be shared.
  explicit UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                               hipStream_t stream = nullptr);
  UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                      hipStream_t stream,
                      UnitBfsCsrOffsetMode offset_mode);
  UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                      hipStream_t stream,
                      UnitBfsCsrWorkspaceOptions options);
  UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                      hipStream_t stream,
                      UnitBfsCsrOffsetMode offset_mode,
                      UnitBfsCsrWorkspaceOptions options);
  explicit UnitBfsCsrWorkspace(
      std::shared_ptr<const UnitBfsCsrGraph> adjacency,
      hipStream_t stream = nullptr);
  UnitBfsCsrWorkspace(
      std::shared_ptr<const UnitBfsCsrGraph> adjacency,
      hipStream_t stream,
      UnitBfsCsrWorkspaceOptions options);
  ~UnitBfsCsrWorkspace();

  UnitBfsCsrWorkspace(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace& operator=(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept;
  UnitBfsCsrWorkspace& operator=(UnitBfsCsrWorkspace&&) noexcept;

  UnitBfsCsrAllocationState allocation_state() const noexcept;

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
