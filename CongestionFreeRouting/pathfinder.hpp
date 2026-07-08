#pragma once

#include "../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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
  std::vector<std::string> strings;
  std::vector<std::uint64_t> node_device_ids;
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

struct PathfinderOptions {
  float delta = 4.0f;
  int max_pathfinder_iterations = 30;
  int max_sssp_iterations = -1;
  int capacity = 1;
  float initial_present_factor = 1.0f;
  float present_factor_multiplier = 2.0f;
  float history_factor = 1.0f;
  std::size_t net_limit = 0;
  std::size_t route_batch_size = 256;
  std::size_t parallel_net_workers = 1;
};

struct PathfinderResult {
  bool routed = false;
  bool all_sinks_reached = false;
  int iterations_used = 0;
  int overused_nodes = 0;
  int max_occupancy = 0;
  std::vector<int> occupancy;
  std::vector<float> history;
  std::vector<RoutedNet> nets;
};

HostCsrF32 load_csrbin(const std::filesystem::path& path);
RoutingMetadata load_interchange_metadata(const std::filesystem::path& path);

std::vector<PathEdge> reconstruct_shortest_path(
    const HostCsrF32& graph,
    const std::vector<float>& dist,
    int source,
    int target);

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream = nullptr);

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index);

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result);

}  // namespace routing
