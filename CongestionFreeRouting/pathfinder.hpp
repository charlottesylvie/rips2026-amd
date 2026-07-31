#pragma once

#include "../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "delta_stepping/delta_stepping_policy.hpp"
#include "interchange/import_policy.hpp"
#include "sssp_query_capacity.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace routing {

constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();

struct EdgeAttr {
  std::uint64_t tile_string = 0;
  std::uint64_t pip_data_index = 0;
};

struct PipData {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
};

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::optional<interchange::InterchangeArtifactPairId> artifact_pair_id;
  std::vector<std::string> strings;
  std::vector<std::uint64_t> node_device_ids;
  std::vector<std::int32_t> node_min_x;
  std::vector<std::int32_t> node_max_x;
  std::vector<std::int32_t> node_min_y;
  std::vector<std::int32_t> node_max_y;
  std::vector<std::uint64_t> node_tile_type_strings;
  std::vector<std::uint64_t> node_wire_type_strings;
  std::vector<EdgeAttr> edge_attrs;
  std::vector<PipData> pip_data;
  std::vector<SitePinNode> site_pin_attrs;
  std::vector<RouteRequest> route_requests;
  std::vector<std::uint64_t> blocked_nodes;
  std::vector<std::uint64_t> sink_stop_nodes;
  std::uint64_t device_path_string = kNoIndex;
  std::uint64_t physical_path_string = kNoIndex;
  std::uint64_t logical_path_string = kNoIndex;
  std::uint64_t logical_design_name_string = kNoIndex;
  // The routing-only v6 sidecar deliberately omits graph-sized per-node
  // metadata arrays.  Keep the declared header count separately so callers
  // can still verify that the sidecar and CSR describe the same graph.  Zero
  // preserves the historical direct-test convention of deriving the count
  // from node_device_ids.
  std::uint64_t declared_node_count = 0;
  // Like declared_node_count, this lets the routing-only loader validate the
  // CSR/metadata pairing without retaining the graph-sized edge/PIP tables.
  std::uint64_t declared_edge_attr_count = 0;
};

enum class InterchangeMetadataLoadMode {
  // Materialize every section represented by RoutingMetadata.  V6 still has
  // no per-node arrays because those sections are absent from the format.
  kFull,
  // Load only strings and route requests needed by PathFinder. Large unused
  // sections are range-checked and skipped without throwaway allocations.
  kRoutingOnly,
  // Load the routing fields plus the edge/PIP tables needed by the routes
  // JSONL writer, while still skipping graph-sized per-node metadata.
  kRoutingWithRouteOutput,
};

struct PathEdge {
  int from = -1;
  int to = -1;
  minplus_sparse::Offset csr_edge = -1;
  float cost = 0.0f;
};

struct RoutedSink {
  int source = -1;
  int target = -1;
  float distance = 0.0f;
  bool reached = false;
  std::vector<int> nodes;
  std::vector<PathEdge> edges;
};

struct RoutedNet {
  std::uint64_t net_string = 0;
  bool reached_all_sinks = false;
  std::vector<RoutedSink> sinks;
  std::vector<int> unique_nodes;
};

enum class SsspEngine {
  kUnitBfs,
  kDeltaStep,
  kBellmanFord,
};

struct PathfinderOptions {
  SsspEngine sssp_engine = SsspEngine::kUnitBfs;
  float delta = 1.0f;
  int max_sssp_iterations = -1;
  // Explicit A/B control for generic vector-target Delta-Stepping runs.
  bool delta_force_legacy_parent = false;
  int capacity = 1;
  std::size_t net_limit = 0;
  // Zero enables engine-dependent automatic worker selection.
  std::size_t parallel_net_workers = 0;
  // Appended to preserve positional aggregate initialization of every older
  // option field. A numeric delta remains an explicit override.
  bool delta_auto = false;
  float delta_multiplier = 1.0f;
  // Appended controls preserve positional aggregate compatibility.
  bool delta_force_generic = false;
  // Set by CLI/configuration adapters when --delta or --delta-multiplier was
  // explicitly supplied, so non-Delta engines cannot silently ignore it.
  bool delta_controls_explicit = false;
  bool delta_telemetry = false;
  // The historical host-checked controller remains the default. These fields
  // are appended so older positional aggregate initializers retain meaning.
  DeltaSteppingCsrControllerMode delta_controller_mode =
      DeltaSteppingCsrControllerMode::kHostChecked;
  int delta_controller_batch_size =
      static_cast<int>(kDeltaSteppingCsrRecommendedControllerBatchSize);
  bool delta_controller_controls_explicit = false;
  DeltaSteppingCsrCurrentMembershipMode delta_current_membership_mode =
      DeltaSteppingCsrCurrentMembershipMode::kBoolean;
  bool delta_current_membership_controls_explicit = false;
};

struct PathfinderResult {
  bool routed = false;
  bool all_sinks_reached = false;
  int iterations_used = 0;
  int overused_nodes = 0;
  int max_occupancy = 0;
  std::vector<int> occupancy;
  std::vector<RoutedNet> nets;
};

struct UnitBfsPathObservation {
  bool captured = false;
  bool reached = false;
  float distance = std::numeric_limits<float>::infinity();
  int source = -1;
  std::size_t edge_count = 0;
  std::uint64_t path_hash = 0;
};

// Opt-in diagnostic state for one UnitBFS sink.  When supplied to
// run_pathfinder, the selected request is observed inside its real worker and
// workspace after the preceding request prefix has run.  Normal routing does
// not allocate or execute any of these probes when the pointer is null.
struct UnitBfsPathDiagnostic {
  std::size_t net_index = std::numeric_limits<std::size_t>::max();
  std::size_t sink_index = std::numeric_limits<std::size_t>::max();
  int target = -1;
  bool all_unit_weights = false;
  std::size_t original_source_count = 0;
  std::size_t tree_source_count = 0;
  std::size_t prior_sinks_reached = 0;
  std::size_t worker_count = 0;
  int cpu_original_distance = -1;
  int cpu_expanded_tree_distance = -1;
  // The selected target's result from the original multi-target batch,
  // before any sink branch expands the route tree.
  UnitBfsPathObservation raw_batched;
  UnitBfsPathObservation fresh_original;
  UnitBfsPathObservation fresh_expanded_tree;
  bool attached_reached = false;
  float attached_distance = std::numeric_limits<float>::infinity();
  int attached_source = -1;
  std::size_t attached_edge_count = 0;
  std::uint64_t attached_path_hash = 0;
  std::string classification;
};

HostCsrF32 load_csrbin(
    const std::filesystem::path& path,
    std::optional<interchange::InterchangeArtifactPairId>* artifact_pair_id =
        nullptr);
RoutingMetadata load_interchange_metadata(
    const std::filesystem::path& path,
    InterchangeMetadataLoadMode mode = InterchangeMetadataLoadMode::kFull);

// Derive conservative initial workspace reservations from exactly the routed
// request prefix. Raw endpoint counts are retained (including duplicates), and
// the result is only a hint: low-level workspaces may still grow later.
SsspQueryCapacityHints derive_query_capacity_hints(
    const RoutingMetadata& metadata,
    std::size_t routed_request_count);

std::vector<PathEdge> reconstruct_shortest_path(
    const HostCsrF32& graph,
    const std::vector<float>& dist,
    int source,
    int target);

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream = nullptr,
                                UnitBfsPathDiagnostic* unit_bfs_diagnostic = nullptr);

std::string unit_bfs_path_diagnostic_json(
    const UnitBfsPathDiagnostic& diagnostic);

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index);

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result);

}  // namespace routing
