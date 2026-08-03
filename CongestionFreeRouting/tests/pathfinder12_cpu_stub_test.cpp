// Build and run from the repository root:
//   g++ -std=c++17 -O2 -pthread -D__HIP_PLATFORM_AMD__=1 \
//     -I Routing/tests/fake_hip -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/pathfinder12_cpu_stub_test.cpp \
//     CongestionFreeRouting/pathfinder12.cpp \
//     -DROUTING_PATHFINDER12_NO_MAIN \
//     -o /tmp/pathfinder12_cpu_stub_test
//   /tmp/pathfinder12_cpu_stub_test
// Repeat the build with -DROUTING_PATHFINDER12_TEST_STAMP_LIMIT=2 to force
// multiple generation-wrap resets during the scratch-reuse test.

#include "../pathfinder12.hpp"
#include "bf12_pathfinder_cpu_stub.inc"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_rejected(Function&& function, const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

HostCsrF32 make_graph() {
  HostCsrF32 graph;
  graph.rows = 8;
  graph.cols = 8;
  graph.nnz = 10;
  graph.rowptr = {0, 2, 4, 6, 7, 7, 9, 10, 10};
  graph.colind = {1, 5, 3, 4, 1, 4, 4, 4, 6, 4};
  graph.values = {1.0f, 5.0f, 1.0f, 2.0f, 0.0f,
                  10.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  return graph;
}

routing::interchange::RoutingCsrSidecars make_sidecars() {
  routing::interchange::RoutingCsrSidecars sidecars;
  sidecars.route_end_x = {0, 100, 0, 0, 0, 1, 2, 3};
  sidecars.route_end_y.assign(8, 0);
  sidecars.base_vertex_cost.assign(8, 1.0f);
  sidecars.base_vertex_cost[4] = 2.0f;
  return sidecars;
}

routing::SitePinNode pin(int node) {
  routing::SitePinNode result;
  result.node = node;
  return result;
}

routing::RouteRequest request(std::uint64_t identity,
                              std::vector<int> sources,
                              std::vector<int> targets) {
  routing::RouteRequest result;
  result.net_string = identity;
  for (const int source : sources) result.sources.push_back(pin(source));
  for (const int target : targets) result.sinks.push_back(pin(target));
  return result;
}

routing::RoutingMetadata make_metadata() {
  routing::RoutingMetadata metadata;
  metadata.route_requests = {
      request(100, {0}, {3, 4}),
      request(101, {2}, {4}),
      request(102, {0, 2}, {3}),
      request(103, {3}, {4}),
      request(104, {5}, {4}),
  };
  return metadata;
}

void require_path(const routing::RoutedSink& sink,
                  const std::vector<int>& nodes,
                  float distance,
                  const std::string& name) {
  require(sink.reached, name + " was not reached");
  require(sink.nodes == nodes, name + " path nodes changed");
  require(std::fabs(sink.distance - distance) < 1e-6f,
          name + " path distance changed");
  require(sink.edges.size() + 1 == sink.nodes.size(),
          name + " path shape is invalid");
}

void require_same_routes(const routing::PathfinderResult& left,
                         const routing::PathfinderResult& right,
                         const std::string& name) {
  require(left.routed == right.routed &&
              left.all_sinks_reached == right.all_sinks_reached &&
              left.nets.size() == right.nets.size(),
          name + " summary changed");
  for (std::size_t net_index = 0; net_index < left.nets.size(); ++net_index) {
    const routing::RoutedNet& left_net = left.nets[net_index];
    const routing::RoutedNet& right_net = right.nets[net_index];
    require(left_net.net_string == right_net.net_string &&
                left_net.reached_all_sinks == right_net.reached_all_sinks &&
                left_net.unique_nodes == right_net.unique_nodes &&
                left_net.sinks.size() == right_net.sinks.size(),
            name + " net changed");
    for (std::size_t sink_index = 0; sink_index < left_net.sinks.size();
         ++sink_index) {
      const routing::RoutedSink& left_sink = left_net.sinks[sink_index];
      const routing::RoutedSink& right_sink = right_net.sinks[sink_index];
      require(left_sink.source == right_sink.source &&
                  left_sink.target == right_sink.target &&
                  left_sink.distance == right_sink.distance &&
                  left_sink.reached == right_sink.reached &&
                  left_sink.nodes == right_sink.nodes &&
                  left_sink.edges.size() == right_sink.edges.size(),
              name + " sink changed");
      for (std::size_t edge_index = 0;
           edge_index < left_sink.edges.size(); ++edge_index) {
        const routing::PathEdge& left_edge = left_sink.edges[edge_index];
        const routing::PathEdge& right_edge = right_sink.edges[edge_index];
        require(left_edge.from == right_edge.from &&
                    left_edge.to == right_edge.to &&
                    left_edge.csr_edge == right_edge.csr_edge &&
                    left_edge.cost == right_edge.cost,
                name + " edge changed");
      }
    }
  }
}

void test_explicit_full_and_partial_batches() {
  bf12_cpu_stub::reset();
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  const routing::RoutingMetadata metadata = make_metadata();
  routing::Pathfinder12Options options;
  options.bf12_batch_size = 2;
  options.bf12_controller = BellmanFord12ControllerMode::HostBatch;
  options.bf12_enable_bounding_boxes = false;
  options.bf12_enable_telemetry = true;
  options.bf12_result_node_capacity = 1;
  options.bf12_result_edge_capacity = 1;

  routing::Pathfinder12RoutingStatistics stats;
  std::vector<std::array<std::size_t, 3>> progress_updates;
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      graph, metadata, &sidecars, options, nullptr, &stats,
      [&progress_updates](std::size_t completed,
                          std::size_t total,
                          std::size_t batches) {
        progress_updates.push_back({completed, total, batches});
      });
  require(result.routed && result.all_sinks_reached,
          "unbounded BF12 CPU run did not route all nets");
  require(result.nets.size() == metadata.route_requests.size(),
          "BF12 result net count changed");
  for (std::size_t net = 0; net < result.nets.size(); ++net) {
    require(result.nets[net].net_string ==
                metadata.route_requests[net].net_string,
            "BF12 did not restore stable net order");
  }
  require_path(result.nets[0].sinks[0], {0, 1, 3}, 2.0f,
               "net0 sink3");
  // The second independently reconstructed shortest path is trimmed at the
  // last point already in this net's tree (node 3).
  require_path(result.nets[0].sinks[1], {3, 4}, 2.0f,
               "net0 sink4 trimmed branch");
  require_path(result.nets[1].sinks[0], {2, 1, 3, 4}, 3.0f,
               "net1 independent overlapping path");
  require_path(result.nets[2].sinks[0], {2, 1, 3}, 1.0f,
               "net2 multi-source zero-edge path");
  require_path(result.nets[3].sinks[0], {3, 4}, 2.0f,
               "net3 overlapping target");
  require_path(result.nets[4].sinks[0], {5, 4}, 2.0f,
               "net4 final partial batch");
  require(result.nets[0].sinks[0].distance !=
              result.nets[1].sinks[0].distance,
          "independent nets appear to have shared their distance labels");

  require(bf12_cpu_stub::stats.graph_constructions == 1 &&
              bf12_cpu_stub::stats.workspace_constructions == 1,
          "route_all_nets_bf12 did not retain one validated graph/workspace");
  require(bf12_cpu_stub::stats.dynamic_cost_uploads == 0,
          "Pathfinder12 redundantly uploaded the workspace's all-one epoch");
  require(bf12_cpu_stub::stats.cooperative_occupancy_queries == 0,
          "forced host-batch queried cooperative kernel occupancy");
  require(bf12_cpu_stub::stats.initial_batch_sizes ==
              std::vector<std::size_t>({2, 2, 1}),
          "explicit BF12 full/final-partial scheduling changed");
  require(stats.batch_count == 3 && stats.full_batch_count == 2 &&
              stats.partial_batch_count == 1,
          "Pathfinder12 batch statistics are inconsistent");
  require(progress_updates ==
              std::vector<std::array<std::size_t, 3>>(
                  {{2, 5, 1}, {4, 5, 2}, {5, 5, 3}}),
          "Pathfinder12 progress callbacks changed batch completion order");
  require(stats.backend_telemetry.arena_overflow_retry_count == 3,
          "small result arenas did not exercise geometric retry telemetry");
}

void test_generation_stamped_route_tree_reuse() {
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  routing::RoutingMetadata metadata;
  metadata.route_requests = {
      request(500, {0}, {3, 4}),
      request(501, {2}, {3, 4}),
      request(502, {3}, {3, 4, 4}),
      request(503, {5}, {4}),
      request(504, {0, 2}, {3, 4}),
      request(505, {0}, {3, 4}),
  };
  routing::Pathfinder12Options options;
  options.bf12_batch_size = 1;
  options.bf12_controller = BellmanFord12ControllerMode::HostBatch;
  options.bf12_enable_bounding_boxes = false;

  bf12_cpu_stub::reset();
  const routing::PathfinderResult first = routing::route_all_nets_bf12(
      graph, metadata, &sidecars, options);
  require(first.routed && first.all_sinks_reached,
          "generation-stamped scratch did not route every query");
  require_path(first.nets[0].sinks[0], {0, 1, 3}, 2.0f,
               "scratch net0 sink3");
  require_path(first.nets[1].sinks[0], {2, 1, 3}, 1.0f,
               "scratch conflicting-parent net1 sink3");
  require_path(first.nets[2].sinks[0], {3}, 0.0f,
               "scratch source-as-target");
  require_path(first.nets[2].sinks[1], {3, 4}, 2.0f,
               "scratch first duplicate target");
  require_path(first.nets[2].sinks[2], {4}, 0.0f,
               "scratch already-attached duplicate target");
  require_path(first.nets[4].sinks[0], {2, 1, 3}, 1.0f,
               "scratch multi-source/multi-target sink3");
  require_path(first.nets[4].sinks[1], {3, 4}, 2.0f,
               "scratch multi-source/multi-target sink4");

  bf12_cpu_stub::reset();
  const routing::PathfinderResult second = routing::route_all_nets_bf12(
      graph, metadata, &sidecars, options);
  require_same_routes(first, second,
                      "repeated generation-stamped scratch run");
  require(bf12_cpu_stub::stats.initial_batch_sizes ==
              std::vector<std::size_t>(metadata.route_requests.size(), 1),
          "scratch-reuse test did not execute batch-size-one queries");
  require(bf12_cpu_stub::stats.dynamic_cost_uploads == 0,
          "scratch reuse restored the redundant cost upload");
  require(bf12_cpu_stub::stats.cooperative_occupancy_queries == 0,
          "scratch reuse queried cooperative occupancy in host mode");
}

void test_checked_helpers_retain_validation() {
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  const routing::RoutingMetadata metadata = make_metadata();
  routing::Pathfinder12Options options;
  options.bf12_enable_bounding_boxes = false;

  HostCsrF32 invalid_graph = graph;
  invalid_graph.values[0] = -1.0f;
  require_rejected(
      [&] {
        (void)routing::prepare_bf12_batch(
            invalid_graph, metadata, sidecars, {0}, options);
      },
      "public BF12 preparation skipped graph validation");

  auto invalid_sidecars = sidecars;
  invalid_sidecars.base_vertex_cost.pop_back();
  require_rejected(
      [&] {
        (void)routing::prepare_bf12_batch(
            graph, metadata, invalid_sidecars, {0}, options);
      },
      "public BF12 preparation skipped sidecar validation");

  routing::Pathfinder12PreparedQuery empty_prepared;
  BellmanFordCsrResult empty_result;
  BellmanFord12QueryStatus empty_status;
  require_rejected(
      [&] {
        (void)routing::finalize_bf12_routed_net(
            invalid_graph, metadata.route_requests[0], empty_prepared,
            empty_result, empty_status);
      },
      "public BF12 finalization skipped graph validation");

  require_rejected(
      [&] {
        (void)routing::route_all_nets_bf12(
            invalid_graph, metadata, &sidecars, options);
      },
      "public BF12 routing skipped backend graph validation");

  routing::RoutingMetadata empty_metadata;
  require_rejected(
      [&] {
        (void)routing::route_all_nets_bf12(
            invalid_graph, empty_metadata, nullptr, options);
      },
      "empty public BF12 routing skipped graph validation");
}

void test_direct_finalization_scratch_resizes_safely() {
  HostCsrF32 singleton;
  singleton.rows = 1;
  singleton.cols = 1;
  singleton.nnz = 0;
  singleton.rowptr = {0, 0};
  const routing::RouteRequest singleton_request = request(600, {0}, {0});
  routing::Pathfinder12PreparedQuery singleton_prepared;
  singleton_prepared.sources = {0};
  singleton_prepared.targets = {0};
  singleton_prepared.target_sink_indices = {0};
  singleton_prepared.descriptor.target_count = 1;
  BellmanFordCsrResult singleton_result;
  singleton_result.target_distances = {0.0f};
  singleton_result.target_sources = {0};
  singleton_result.target_path_offsets = {0, 1};
  singleton_result.target_edge_offsets = {0, 0};
  singleton_result.target_path_nodes = {0};
  BellmanFord12QueryStatus uncertified;
  uncertified.paths_certified = false;

  const routing::RoutedNet first = routing::finalize_bf12_routed_net(
      singleton, singleton_request, singleton_prepared, singleton_result,
      uncertified);
  require(first.reached_all_sinks && first.sinks[0].reached,
          "direct finalization failed before scratch resize");

  const HostCsrF32 graph = make_graph();
  const routing::RouteRequest larger_request = request(601, {3}, {3});
  routing::Pathfinder12PreparedQuery larger_prepared;
  larger_prepared.sources = {3};
  larger_prepared.targets = {3};
  larger_prepared.target_sink_indices = {0};
  larger_prepared.descriptor.target_count = 1;
  BellmanFordCsrResult larger_result;
  larger_result.target_distances = {0.0f};
  larger_result.target_sources = {3};
  larger_result.target_path_offsets = {0, 1};
  larger_result.target_edge_offsets = {0, 0};
  larger_result.target_path_nodes = {3};
  const routing::RoutedNet larger = routing::finalize_bf12_routed_net(
      graph, larger_request, larger_prepared, larger_result, uncertified);
  require(larger.reached_all_sinks && larger.sinks[0].reached,
          "direct finalization failed after scratch growth");

  const routing::RoutedNet second = routing::finalize_bf12_routed_net(
      singleton, singleton_request, singleton_prepared, singleton_result,
      uncertified);
  require_same_routes(
      routing::PathfinderResult{true, true, 0, 0, 0, {}, {first}},
      routing::PathfinderResult{true, true, 0, 0, 0, {}, {second}},
      "direct finalization scratch resize");
}

void test_automatic_four_plus_one_schedule() {
  bf12_cpu_stub::reset();
  routing::Pathfinder12Options options;
  options.bf12_batch_size = 0;
  options.bf12_enable_bounding_boxes = false;
  options.bf12_enable_telemetry = true;
  routing::Pathfinder12RoutingStatistics stats;
  const auto sidecars = make_sidecars();
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      make_graph(), make_metadata(), &sidecars, options, nullptr, &stats);
  require(result.routed, "automatic BF12 schedule failed to route");
  require(stats.selected_batch_size == 4,
          "gfx1151 automatic policy did not select conservative batch 4");
  require(bf12_cpu_stub::stats.initial_batch_sizes ==
              std::vector<std::size_t>({4, 1}),
          "automatic BF12 schedule did not retain the final partial batch");
  require(bf12_cpu_stub::stats.cooperative_occupancy_queries == 1,
          "automatic controller skipped required cooperative occupancy");
  require(stats.backend_telemetry.selected_batch_size == 4 &&
              stats.backend_telemetry.batch_fill_ratio == 0.625,
          "automatic BF12 telemetry did not account for partial-batch slots");
}

void test_automatic_host_schedule_skips_occupancy() {
  bf12_cpu_stub::reset();
  routing::Pathfinder12Options options;
  options.bf12_batch_size = 0;
  options.bf12_controller = BellmanFord12ControllerMode::HostBatch;
  options.bf12_enable_bounding_boxes = false;
  routing::Pathfinder12RoutingStatistics stats;
  const auto sidecars = make_sidecars();
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      make_graph(), make_metadata(), &sidecars, options, nullptr, &stats);
  require(result.routed && stats.selected_batch_size == 4,
          "forced host mode lost architecture-aware automatic scheduling");
  require(bf12_cpu_stub::stats.cooperative_occupancy_queries == 0,
          "automatic forced-host scheduling queried cooperative occupancy");
}

void test_bounded_retry_only_missed_query() {
  bf12_cpu_stub::reset();
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  routing::RoutingMetadata metadata;
  metadata.route_requests.push_back(request(200, {0}, {3, 4}));
  routing::Pathfinder12Options options;
  options.bf12_batch_size = 4;
  options.bf12_enable_bounding_boxes = true;
  options.bf12_bbox_margin_x = 2;
  options.bf12_bbox_margin_y = 0;
  options.bf12_enable_unbounded_retry = true;
  options.bf12_enable_telemetry = true;
  routing::Pathfinder12RoutingStatistics stats;
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      graph, metadata, &sidecars, options, nullptr, &stats);
  require(result.routed, "bounded miss was not recovered by unbounded retry");
  require(bf12_cpu_stub::stats.retried_original_indices ==
              std::vector<std::uint32_t>({0}),
          "bounded fallback retried more than the missed query");
  require(stats.backend_telemetry.bounded_query_retry_count == 1,
          "bounded retry telemetry changed");
  require_path(result.nets[0].sinks[0], {0, 1, 3}, 2.0f,
               "retried bounded sink3");
  require_path(result.nets[0].sinks[1], {3, 4}, 2.0f,
               "retried bounded sink4");
}

void test_finalization_validation_and_uncertified_results() {
  bf12_cpu_stub::reset();
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  const routing::RoutingMetadata metadata = make_metadata();
  routing::Pathfinder12Options options;
  options.bf12_enable_bounding_boxes = false;
  const routing::Pathfinder12PreparedBatch prepared =
      routing::prepare_bf12_batch(graph, metadata, sidecars, {0}, options);
  BellmanFord12CsrWorkspace workspace(graph, sidecars);
  BellmanFord12BatchOptions backend_options;
  backend_options.controller_mode = BellmanFord12ControllerMode::HostBatch;
  backend_options.enable_bounding_boxes = false;
  const BellmanFord12BatchResult backend = workspace.run_batch(
      prepared.flattened_sources, prepared.flattened_targets,
      prepared.descriptors, backend_options);
  require(backend.query_results.size() == 1,
          "CPU stub setup returned the wrong query count");

  BellmanFordCsrResult bad_root = backend.query_results[0];
  bad_root.target_sources[0] = 7;
  require_rejected(
      [&] {
        (void)routing::finalize_bf12_routed_net(
            graph, metadata.route_requests[0], prepared.queries[0], bad_root,
            backend.query_statuses[0]);
      },
      "finalization accepted a path rooted outside the request sources");

  BellmanFordCsrResult bad_edge = backend.query_results[0];
  bad_edge.target_path_edges[0] = 7;  // row-5 edge, not row 0.
  require_rejected(
      [&] {
        (void)routing::finalize_bf12_routed_net(
            graph, metadata.route_requests[0], prepared.queries[0], bad_edge,
            backend.query_statuses[0]);
      },
      "finalization accepted an inconsistent predecessor edge");

  BellmanFord12QueryStatus uncertified = backend.query_statuses[0];
  uncertified.paths_certified = false;
  uncertified.hit_iteration_limit = true;
  const routing::RoutedNet rejected = routing::finalize_bf12_routed_net(
      graph, metadata.route_requests[0], prepared.queries[0],
      backend.query_results[0], uncertified);
  require(!rejected.reached_all_sinks && !rejected.sinks[0].reached &&
              !rejected.sinks[1].reached,
          "uncertified finite BF12 labels were consumed as final paths");

  routing::RoutingMetadata identity_metadata;
  identity_metadata.route_requests.push_back(request(300, {3}, {3}));
  const routing::Pathfinder12PreparedBatch identity_prepared =
      routing::prepare_bf12_batch(graph, identity_metadata, sidecars, {0},
                                  options);
  BellmanFordCsrResult identity_result;
  identity_result.target_distances = {0.0f};
  identity_result.target_sources = {3};
  identity_result.target_path_offsets = {0, 1};
  identity_result.target_edge_offsets = {0, 0};
  identity_result.target_path_nodes = {3};
  BellmanFord12QueryStatus identity_status;
  identity_status.paths_certified = false;
  identity_status.hit_iteration_limit = true;
  const routing::RoutedNet identity = routing::finalize_bf12_routed_net(
      graph, identity_metadata.route_requests[0],
      identity_prepared.queries[0], identity_result, identity_status);
  require(identity.reached_all_sinks && identity.sinks[0].reached &&
              identity.sinks[0].distance == 0.0f,
          "source-target identity was not trivially finalized");
}

void test_empty_and_invalid_requests_keep_order() {
  bf12_cpu_stub::reset();
  routing::RoutingMetadata metadata;
  metadata.route_requests = {
      request(400, {-1}, {3}),
      request(401, {0}, {}),
      request(402, {0}, {99}),
  };
  const auto sidecars = make_sidecars();
  routing::Pathfinder12Options options;
  options.bf12_enable_bounding_boxes = false;
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      make_graph(), metadata, &sidecars, options);
  require(result.nets.size() == 3 && result.nets[0].net_string == 400 &&
              result.nets[1].net_string == 401 &&
              result.nets[2].net_string == 402,
          "trivial/invalid requests changed stable output order");
  require(!result.nets[0].reached_all_sinks,
          "request without a valid source was routed");
  require(result.nets[1].reached_all_sinks,
          "source-only request should be trivially complete");
  require(!result.nets[2].reached_all_sinks,
          "out-of-range sink was routed");
  require(bf12_cpu_stub::stats.initial_batch_sizes.empty(),
          "empty/invalid requests were submitted to BF12");
}

void test_jsonl_schema_and_stable_order() {
  bf12_cpu_stub::reset();
  const HostCsrF32 graph = make_graph();
  const auto sidecars = make_sidecars();
  routing::RoutingMetadata metadata = make_metadata();
  metadata.strings.resize(110);
  metadata.strings[0] = "";
  metadata.strings[1] = "TILE";
  metadata.strings[2] = "WIRE0";
  metadata.strings[3] = "WIRE1";
  for (std::size_t index = 0; index < metadata.route_requests.size(); ++index) {
    metadata.strings[100 + index] = "net" + std::to_string(100 + index);
  }
  metadata.edge_attrs.resize(static_cast<std::size_t>(graph.nnz));
  for (routing::EdgeAttr& attribute : metadata.edge_attrs) {
    attribute.tile_string = 1;
    attribute.pip_data_index = 0;
  }
  metadata.pip_data.push_back({2, 3, true});

  routing::Pathfinder12Options options;
  options.bf12_batch_size = 2;
  options.bf12_enable_bounding_boxes = false;
  const routing::PathfinderResult result = routing::route_all_nets_bf12(
      graph, metadata, &sidecars, options);
  require(result.routed, "JSONL setup did not route all test nets");

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() /
      "pathfinder12_cpu_stub_routes.jsonl";
  const std::filesystem::path unstable_path =
      std::filesystem::temp_directory_path() /
      "pathfinder12_cpu_stub_unstable_routes.jsonl";
  const std::filesystem::path sequential_path =
      std::filesystem::temp_directory_path() /
      "pathfinder12_cpu_stub_sequential_routes.jsonl";
  std::error_code ignored;
  std::filesystem::remove(output_path, ignored);
  std::filesystem::remove(unstable_path, ignored);
  std::filesystem::remove(sequential_path, ignored);
  routing::write_pathfinder12_routes_jsonl(output_path, graph, metadata,
                                           result);

  std::ifstream input(output_path);
  require(static_cast<bool>(input), "JSONL output could not be reopened");
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    lines.push_back(std::move(line));
  }
  input.close();
  std::filesystem::remove(output_path, ignored);

  routing::Pathfinder12Options sequential_options = options;
  sequential_options.bf12_batch_size = 1;
  const routing::PathfinderResult sequential_result =
      routing::route_all_nets_bf12(graph, metadata, &sidecars,
                                   sequential_options);
  require(sequential_result.routed,
          "sequential JSONL comparison did not route all test nets");
  routing::write_pathfinder12_routes_jsonl(
      sequential_path, graph, metadata, sequential_result);
  std::ifstream sequential_input(sequential_path);
  require(static_cast<bool>(sequential_input),
          "sequential JSONL output could not be reopened");
  std::vector<std::string> sequential_lines;
  for (std::string line; std::getline(sequential_input, line);) {
    sequential_lines.push_back(std::move(line));
  }
  sequential_input.close();
  std::filesystem::remove(sequential_path, ignored);
  require(sequential_lines == lines,
          "batched JSONL output differs from sequential BF12 output");

  require(lines.size() == metadata.route_requests.size(),
          "JSONL writer emitted the wrong number of net records");
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const std::string expected_net =
        "\"net\":\"net" + std::to_string(100 + index) + "\"";
    require(lines[index].find(expected_net) != std::string::npos,
            "JSONL writer changed stable net order");
    require(lines[index].find("\"routed\":true") != std::string::npos &&
                lines[index].find("\"sources\":[") != std::string::npos &&
                lines[index].find("\"sinks\":[") != std::string::npos &&
                lines[index].find("\"edges\":[") != std::string::npos,
            "JSONL net record is missing a required schema field");
  }
  for (const std::string& edge_field :
       {"\"from\":", "\"to\":", "\"csr_edge\":", "\"tile\":",
        "\"wire0\":", "\"wire1\":", "\"forward\":"}) {
    require(lines[0].find(edge_field) != std::string::npos,
            "JSONL edge record is missing a required schema field");
  }

  routing::PathfinderResult unstable = result;
  std::swap(unstable.nets[0], unstable.nets[1]);
  require_rejected(
      [&] {
        routing::write_pathfinder12_routes_jsonl(
            unstable_path, graph, metadata, unstable);
      },
      "JSONL writer accepted unstable result-net ordering");
  std::filesystem::remove(unstable_path, ignored);
}

}  // namespace

int main() {
  try {
    test_explicit_full_and_partial_batches();
    test_generation_stamped_route_tree_reuse();
    test_checked_helpers_retain_validation();
    test_direct_finalization_scratch_resizes_safely();
    test_automatic_four_plus_one_schedule();
    test_automatic_host_schedule_skips_occupancy();
    test_bounded_retry_only_missed_query();
    test_finalization_validation_and_uncertified_results();
    test_empty_and_invalid_requests_keep_order();
    test_jsonl_schema_and_stable_order();
    std::cout << "Pathfinder12 CPU stub test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Pathfinder12 CPU stub test failed: " << error.what() << '\n';
    return 1;
  }
}
