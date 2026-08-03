#pragma once

#include "bellman_ford/bf12.hpp"
#include "pathfinder.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace routing {

// BF12 is deliberately a standalone one-shot router.  Every batch observes
// one frozen dynamic-cost epoch; this preserves exact shortest paths for each
// query but is not a negotiated-congestion update after every individual net.
struct Pathfinder12Options {
  std::size_t bf12_batch_size = 0;
  BellmanFord12ControllerMode bf12_controller =
      BellmanFord12ControllerMode::Auto;
  int bf12_target_check_interval = 1;
  bool bf12_enable_bounding_boxes = true;
  bool bf12_enable_unbounded_retry = true;
  bool bf12_enable_telemetry = false;
  std::size_t bf12_result_node_capacity = 0;
  std::size_t bf12_result_edge_capacity = 0;
  std::size_t bf12_memory_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
  int bf12_bbox_margin_x = 2;
  int bf12_bbox_margin_y = 14;

  int max_sssp_iterations = -1;
  int capacity = 1;
  std::size_t net_limit = 0;

  // Accepted only as a compatibility diagnostic.  BF12 always owns one GPU
  // workspace/controller; values greater than one are rejected rather than
  // launching multiple full-residency cooperative kernels.
  std::size_t parallel_net_workers = 0;
};

struct Pathfinder12PreparedQuery {
  std::size_t original_net_index = 0;
  BellmanFord12QueryDescriptor descriptor{};
  std::vector<int> sources;
  std::vector<int> targets;
  // Maps each descriptor target back to the original RouteRequest sink.
  std::vector<std::size_t> target_sink_indices;
};

struct Pathfinder12PreparedBatch {
  std::vector<int> flattened_sources;
  std::vector<int> flattened_targets;
  std::vector<BellmanFord12QueryDescriptor> descriptors;
  std::vector<Pathfinder12PreparedQuery> queries;
};

struct Pathfinder12BatchSchedule {
  std::size_t requested_batch_size = 0;
  std::size_t selected_batch_size = 0;
  std::size_t next_net_index = 0;
  bool final_partial_batch = false;
};

struct Pathfinder12RoutingStatistics {
  std::size_t requested_batch_size = 0;
  std::size_t selected_batch_size = 0;
  std::size_t net_count = 0;
  std::size_t submitted_query_count = 0;
  std::size_t batch_count = 0;
  std::size_t full_batch_count = 0;
  std::size_t partial_batch_count = 0;
  std::size_t bounded_descriptor_count = 0;
  std::size_t unbounded_descriptor_count = 0;
  std::size_t successful_net_count = 0;
  BellmanFord12BatchTelemetry backend_telemetry{};
};

using Pathfinder12ProgressCallback =
    std::function<void(std::size_t completed_nets,
                       std::size_t total_nets,
                       std::size_t completed_batches)>;

struct Pathfinder12CommandLine {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path routes_out_path;
  Pathfinder12Options options{};
  bool allow_unrouted = false;
  bool show_help = false;
};

BellmanFord12ControllerMode parse_bf12_controller(const std::string& value);
void validate_pathfinder12_options(const Pathfinder12Options& options);
Pathfinder12CommandLine parse_pathfinder12_args(int argc, char** argv);
std::string pathfinder12_usage(const std::string& executable);

// Checked public boundary for preparing eligible endpoint ranges. Requests
// without a valid source, and requests with no valid target, are represented
// directly in the final PathfinderResult and are therefore not submitted to
// BF12. route_all_nets_bf12 uses an internal already-validated context so its
// batches do not repeat the public O(V+E) checks.
Pathfinder12PreparedBatch prepare_bf12_batch(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata,
    const interchange::RoutingCsrSidecars& sidecars,
    const std::vector<std::size_t>& net_indices,
    const Pathfinder12Options& options);

// Checked public boundary for converting one compact BF12 query result into
// the existing JSONL-compatible routed-net contract. This validates graph and
// source/target membership, every CSR predecessor edge, path cost consistency,
// tree trimming, and single-parent attachment. route_all_nets_bf12 reuses
// generation-stamped graph-sized scratch behind an internal validated helper.
RoutedNet finalize_bf12_routed_net(
    const HostCsrF32& graph,
    const RouteRequest& request,
    const Pathfinder12PreparedQuery& prepared,
    const BellmanFordCsrResult& result,
    const BellmanFord12QueryStatus& status);

// Route all selected nets with exactly one reusable BF12 workspace and its
// frozen all-ones dynamic-cost epoch. Graph/sidecars are validated once, and
// route-tree scratch is allocated once and generation-stamped between queries.
// Full and final partial batches are restored to stable input-net order before
// returning.
PathfinderResult route_all_nets_bf12(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata,
    const interchange::RoutingCsrSidecars* sidecars,
    const Pathfinder12Options& options = {},
    hipStream_t stream = nullptr,
    Pathfinder12RoutingStatistics* statistics = nullptr,
    Pathfinder12ProgressCallback progress = {});

// Standalone artifact helpers.  They intentionally use BF12-specific names so
// pathfinder12.cpp can be linked without pathfinder.cpp and can also coexist
// with it in test tools without duplicate public symbols.
HostCsrF32 load_pathfinder12_csrbin(
    const std::filesystem::path& path,
    std::optional<interchange::InterchangeArtifactPairId>* artifact_pair_id,
    interchange::RoutingCsrSidecars* routing_sidecars);
RoutingMetadata load_pathfinder12_metadata(
    const std::filesystem::path& path,
    InterchangeMetadataLoadMode mode =
        InterchangeMetadataLoadMode::kRoutingOnly);
void write_pathfinder12_routes_jsonl(const std::filesystem::path& path,
                                     const HostCsrF32& graph,
                                     const RoutingMetadata& metadata,
                                     const PathfinderResult& result);

}  // namespace routing
