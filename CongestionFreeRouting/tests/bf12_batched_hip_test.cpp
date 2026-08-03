// Build from the repository root on a ROCm host:
//   hipcc -std=c++17 -O2 -x hip -DBF12_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src -I CongestionFreeRouting \
//     -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/bf12_batched_hip_test.cpp \
//     CongestionFreeRouting/bellman_ford/bf12.cpp -pthread \
//     -o /tmp/bf12_batched_hip_test

#include "../bellman_ford/bf12.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;

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

HostCsrF32 make_graph(
    int rows,
    const std::vector<std::tuple<int, int, float>>& edges) {
  HostCsrF32 graph;
  graph.rows = rows;
  graph.cols = rows;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(rows + 1), 0);
  for (const auto& edge : edges) {
    const int from = std::get<0>(edge);
    const int to = std::get<1>(edge);
    const float weight = std::get<2>(edge);
    require(from >= 0 && from < rows && to >= 0 && to < rows &&
                std::isfinite(weight) && weight >= 0.0f,
            "test graph edge is invalid");
    ++graph.rowptr[static_cast<std::size_t>(from + 1)];
  }
  for (int row = 0; row < rows; ++row) {
    graph.rowptr[static_cast<std::size_t>(row + 1)] +=
        graph.rowptr[static_cast<std::size_t>(row)];
  }
  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
  for (const auto& item : edges) {
    const int from = std::get<0>(item);
    const Offset edge = cursor[static_cast<std::size_t>(from)]++;
    graph.colind[static_cast<std::size_t>(edge)] = std::get<1>(item);
    graph.values[static_cast<std::size_t>(edge)] = std::get<2>(item);
  }
  return graph;
}

BellmanFord12NodeSidecars make_sidecars(int rows) {
  BellmanFord12NodeSidecars sidecars;
  for (int node = 0; node < rows; ++node) {
    sidecars.route_end_x.push_back(node);
    sidecars.route_end_y.push_back(0);
    sidecars.base_vertex_costs.push_back(1.0f);
  }
  return sidecars;
}

struct CpuResult {
  std::vector<float> distance;
};

CpuResult cpu_dijkstra(const HostCsrF32& graph,
                       const std::vector<float>& base_cost,
                       const std::vector<float>& dynamic_cost,
                       const std::vector<int>& sources,
                       const BellmanFord12BoundingBox& bounds,
                       const BellmanFord12NodeSidecars& sidecars) {
  const float infinity = std::numeric_limits<float>::infinity();
  CpuResult result;
  result.distance.assign(static_cast<std::size_t>(graph.rows), infinity);
  std::vector<unsigned char> source_mask(static_cast<std::size_t>(graph.rows),
                                         0);
  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  for (const int source : sources) {
    source_mask[static_cast<std::size_t>(source)] = 1;
    if (result.distance[static_cast<std::size_t>(source)] != 0.0f) {
      result.distance[static_cast<std::size_t>(source)] = 0.0f;
      queue.push({0.0f, source});
    }
  }
  auto admitted = [&](int node) {
    if (!bounds.enabled) return true;
    const int x = sidecars.route_end_x[static_cast<std::size_t>(node)];
    const int y = sidecars.route_end_y[static_cast<std::size_t>(node)];
    return (x == routing::interchange::kMissingRouteCoordinate &&
            y == routing::interchange::kMissingRouteCoordinate) ||
           (x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
            y <= bounds.max_y);
  };
  while (!queue.empty()) {
    const auto [distance, from] = queue.top();
    queue.pop();
    if (distance != result.distance[static_cast<std::size_t>(from)]) continue;
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
         edge < graph.rowptr[static_cast<std::size_t>(from + 1)]; ++edge) {
      const int to = graph.colind[static_cast<std::size_t>(edge)];
      if (source_mask[static_cast<std::size_t>(to)] != 0 || !admitted(to)) {
        continue;
      }
      const float raw = graph.values[static_cast<std::size_t>(edge)];
      const float dynamic = dynamic_cost[static_cast<std::size_t>(to)];
      const float weight = raw == 0.0f || dynamic == 0.0f
                               ? 0.0f
                               : raw * base_cost[static_cast<std::size_t>(to)] *
                                     dynamic;
      const float candidate = distance + weight;
      if (candidate < result.distance[static_cast<std::size_t>(to)]) {
        result.distance[static_cast<std::size_t>(to)] = candidate;
        queue.push({candidate, to});
      }
    }
  }
  return result;
}

int source_for_edge(const HostCsrF32& graph, Offset edge) {
  const auto iterator = std::upper_bound(graph.rowptr.begin(),
                                         graph.rowptr.end(), edge);
  if (iterator == graph.rowptr.begin()) return -1;
  const std::ptrdiff_t row = iterator - graph.rowptr.begin() - 1;
  return row >= 0 && row < graph.rows ? static_cast<int>(row) : -1;
}

void validate_compact_result(const HostCsrF32& graph,
                             const std::vector<int>& sources,
                             const std::vector<int>& targets,
                             const BellmanFordCsrResult& result) {
  require(result.target_distances.size() == targets.size() &&
              result.target_sources.size() == targets.size() &&
              result.target_path_offsets.size() == targets.size() + 1 &&
              result.target_edge_offsets.size() == targets.size() + 1,
          "BF12 compact result shape is inconsistent");
  for (std::size_t target = 0; target < targets.size(); ++target) {
    const int node_begin = result.target_path_offsets[target];
    const int node_end = result.target_path_offsets[target + 1];
    const int edge_begin = result.target_edge_offsets[target];
    const int edge_end = result.target_edge_offsets[target + 1];
    if (!std::isfinite(result.target_distances[target])) {
      require(node_begin == node_end && edge_begin == edge_end &&
                  result.target_sources[target] == -1,
              "unreachable BF12 target has a materialized path");
      continue;
    }
    require(node_begin >= 0 && edge_begin >= 0 && node_end > node_begin &&
                node_end <= static_cast<int>(result.target_path_nodes.size()) &&
                edge_end >= edge_begin &&
                edge_end <= static_cast<int>(result.target_path_edges.size()) &&
                node_end - node_begin == edge_end - edge_begin + 1,
            "BF12 materialized path offsets are malformed");
    require(result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                    result.target_sources[target] &&
                std::find(sources.begin(), sources.end(),
                          result.target_sources[target]) != sources.end() &&
                result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                    targets[target],
            "BF12 materialized path endpoints are invalid");
    float cost = 0.0f;
    for (int item = 0; item < edge_end - edge_begin; ++item) {
      const Offset edge = result.target_path_edges[
          static_cast<std::size_t>(edge_begin + item)];
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + item)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + item + 1)];
      require(edge >= 0 && edge < graph.nnz && source_for_edge(graph, edge) == from &&
                  graph.colind[static_cast<std::size_t>(edge)] == to,
              "BF12 materialized predecessor edge is invalid");
      cost += result.target_path_edge_costs[
          static_cast<std::size_t>(edge_begin + item)];
    }
    require(std::fabs(cost - result.target_distances[target]) < 1e-4f,
            "BF12 compact path cost differs from its distance");
  }
}

struct QueryInput {
  std::vector<int> sources;
  std::vector<int> targets;
  BellmanFord12BoundingBox bounds{};
  int max_iterations = -1;
  std::uint32_t original_index = 0;
};

struct FlattenedInput {
  std::vector<int> sources;
  std::vector<int> targets;
  std::vector<BellmanFord12QueryDescriptor> descriptors;
};

FlattenedInput flatten(const std::vector<QueryInput>& queries) {
  FlattenedInput result;
  for (const QueryInput& query : queries) {
    BellmanFord12QueryDescriptor descriptor;
    descriptor.source_begin = static_cast<std::uint32_t>(result.sources.size());
    descriptor.source_count = static_cast<std::uint32_t>(query.sources.size());
    descriptor.target_begin = static_cast<std::uint32_t>(result.targets.size());
    descriptor.target_count = static_cast<std::uint32_t>(query.targets.size());
    descriptor.bounds = query.bounds;
    descriptor.max_iterations = query.max_iterations;
    descriptor.original_query_index = query.original_index;
    result.sources.insert(result.sources.end(), query.sources.begin(),
                          query.sources.end());
    result.targets.insert(result.targets.end(), query.targets.begin(),
                          query.targets.end());
    result.descriptors.push_back(descriptor);
  }
  return result;
}

void compare_batch(const HostCsrF32& graph,
                   const BellmanFord12NodeSidecars& sidecars,
                   const std::vector<float>& dynamic_cost,
                   BellmanFord12CsrWorkspace* workspace,
                   const std::vector<QueryInput>& queries,
                   BellmanFord12ControllerMode mode,
                   std::size_t requested_batch_size,
                   bool tiny_arena,
                   bool telemetry_enabled = true) {
  const FlattenedInput input = flatten(queries);
  BellmanFord12BatchOptions options;
  options.requested_batch_size = requested_batch_size;
  options.controller_mode = mode;
  options.enable_bounding_boxes = true;
  options.enable_unbounded_retry = false;
  options.enable_telemetry = telemetry_enabled;
  options.memory_safety_reserve_bytes = 1;
  if (tiny_arena) {
    options.result_node_capacity = 1;
    options.result_edge_capacity = 1;
  }
  const BellmanFord12BatchResult actual = workspace->run_batch(
      input.sources, input.targets, input.descriptors, options);
  require(actual.query_results.size() == queries.size() &&
              actual.query_statuses.size() == queries.size(),
          "BF12 returned the wrong query count");
  for (std::size_t query = 0; query < queries.size(); ++query) {
    const CpuResult expected = cpu_dijkstra(
        graph, sidecars.base_vertex_costs, dynamic_cost,
        queries[query].sources, queries[query].bounds, sidecars);
    validate_compact_result(graph, queries[query].sources,
                            queries[query].targets,
                            actual.query_results[query]);
    for (std::size_t target = 0; target < queries[query].targets.size();
         ++target) {
      const float want = expected.distance[static_cast<std::size_t>(
          queries[query].targets[target])];
      const float got = actual.query_results[query].target_distances[target];
      require((std::isinf(want) && std::isinf(got)) ||
                  std::fabs(want - got) < 1e-4f,
              "BF12 distance differs from independent CPU Dijkstra for "
              "query " + std::to_string(query) + " target " +
              std::to_string(target) + ": got " + std::to_string(got) +
              ", expected " + std::to_string(want) + ", requested batch " +
              std::to_string(requested_batch_size) +
              (tiny_arena ? ", tiny arena" : ", default arena"));
    }
    require(actual.original_query_indices[query] ==
                queries[query].original_index,
            "BF12 changed original query order");
    require(actual.query_statuses[query].paths_certified,
            "fully explored test query was not certified");
  }
}

void test_single_query_endpoint_semantics_and_exception_recovery() {
  const HostCsrF32 graph = make_graph(
      6, {{0, 1, 2.0f}, {1, 2, 1.0f}, {4, 5, 3.0f}});
  BellmanFord12NodeSidecars sidecars = make_sidecars(6);
  std::vector<float> unit_costs(6, 1.0f);
  BellmanFord12CsrWorkspace workspace(graph, sidecars, 1);

  // The workspace device epoch is already initialized to one. The public
  // update still validates the vector but must not add a redundant upload
  // synchronization.
  workspace.reset_telemetry();
  workspace.update_vertex_costs(unit_costs);
  require(workspace.telemetry().synchronization_count == 0,
          "unit BF12 initial epoch performed a redundant upload");

  const std::vector<QueryInput> identity_and_duplicates = {
      {{0, 4}, {4, 5, 5}, {}, -1, 250}};
  compare_batch(graph, sidecars, unit_costs, &workspace,
                identity_and_duplicates,
                BellmanFord12ControllerMode::HostBatch, 1, false, false);
  BellmanFord12BatchTelemetry telemetry = workspace.telemetry();
  require(!telemetry.enabled &&
              telemetry.controller_mode_used ==
                  BellmanFord12ControllerMode::HostBatch &&
              telemetry.cooperative_controller_launch_count == 0 &&
              telemetry.host_fallback_round_count ==
                  telemetry.global_traversal_rounds,
          "telemetry-disabled BF12 host baseline accounting is inconsistent");

  // Repeat with telemetry enabled to exercise sparse reset and retained
  // capacities while preserving duplicate and source-as-target ordering.
  workspace.reset_telemetry();
  compare_batch(graph, sidecars, unit_costs, &workspace,
                identity_and_duplicates,
                BellmanFord12ControllerMode::HostBatch, 1, false, true);
  telemetry = workspace.telemetry();
  require(telemetry.enabled && telemetry.selected_batch_size == 1 &&
              telemetry.cooperative_controller_launch_count == 0 &&
              telemetry.host_fallback_round_count ==
                  telemetry.global_traversal_rounds,
          "telemetry-enabled BF12 host baseline accounting is inconsistent");

  // A finite input can still overflow while forming an effective edge weight
  // on device. That failure occurs after query state mutation and therefore
  // verifies the next run takes the defensive dense-reset recovery path.
  std::vector<float> overflowing_costs = unit_costs;
  overflowing_costs[1] = std::numeric_limits<float>::max();
  workspace.update_vertex_costs(overflowing_costs);
  const FlattenedInput failing = flatten({{{0}, {2}, {}, -1, 251}});
  BellmanFord12BatchOptions options;
  options.requested_batch_size = 1;
  options.controller_mode = BellmanFord12ControllerMode::HostBatch;
  options.memory_safety_reserve_bytes = 1;
  require_rejected(
      [&] {
        (void)workspace.run_batch(failing.sources, failing.targets,
                                  failing.descriptors, options);
      },
      "BF12 accepted an overflowing effective edge weight");
  workspace.update_vertex_costs(unit_costs);
  compare_batch(graph, sidecars, unit_costs, &workspace,
                identity_and_duplicates,
                BellmanFord12ControllerMode::HostBatch, 1, false, false);
}

void test_batch_sizes_and_independence() {
  std::vector<std::tuple<int, int, float>> edges;
  for (int node = 0; node < 10; ++node) {
    edges.push_back({node, node + 1, static_cast<float>(node % 3 + 1)});
    if (node + 2 <= 10) {
      edges.push_back({node, node + 2,
                       node % 2 == 0 ? 0.0f : 2.5f});
    }
    if (node + 4 <= 10) edges.push_back({node, node + 4, 4.0f});
  }
  const HostCsrF32 graph = make_graph(12, edges);  // node 11 is unreachable.
  BellmanFord12NodeSidecars sidecars = make_sidecars(12);
  sidecars.base_vertex_costs[4] = 2.0f;
  std::vector<float> dynamic_cost(12, 1.0f);
  dynamic_cost[7] = 1.5f;
  BellmanFord12CsrWorkspace workspace(graph, sidecars, 8);
  workspace.update_vertex_costs(dynamic_cost);

  std::vector<QueryInput> queries = {
      {{0}, {10}, {}, -1, 100},
      {{5}, {10}, {}, -1, 101},
      {{0, 4}, {9, 10}, {}, -1, 102},
      {{0}, {8}, {}, -1, 103},       // overlapping source
      {{2}, {10}, {}, -1, 104},      // overlapping target
      {{3}, {11}, {}, -1, 105},      // unreachable
      {{1, 1, 2}, {7}, {}, -1, 106}, // within-query source dedup
      {{4}, {6, 10}, {}, -1, 107},
  };
  for (const std::size_t batch_size : {1u, 2u, 3u, 4u, 8u}) {
    std::vector<QueryInput> prefix(queries.begin(),
                                   queries.begin() + batch_size);
    compare_batch(graph, sidecars, dynamic_cost, &workspace, prefix,
                  BellmanFord12ControllerMode::HostBatch, batch_size,
                  batch_size == 1);
  }

  // Simulate a scheduler's final partial batch: 4 + 4 + 2.
  std::vector<QueryInput> ten = queries;
  ten.push_back({{1}, {6}, {}, -1, 108});
  ten.push_back({{2, 4}, {8}, {}, -1, 109});
  for (std::size_t begin = 0; begin < ten.size(); begin += 4) {
    const std::size_t end = std::min(ten.size(), begin + 4);
    compare_batch(graph, sidecars, dynamic_cost, &workspace,
                  std::vector<QueryInput>(ten.begin() + begin,
                                          ten.begin() + end),
                  BellmanFord12ControllerMode::HostBatch, 4, false);
  }
  require(workspace.telemetry().arena_overflow_retry_count > 0,
          "tiny BF12 result arena did not report a retry");
  require(workspace.telemetry().selected_batch_size == 8 &&
              workspace.telemetry().batch_fill_ratio > 0.0 &&
              workspace.telemetry().batch_fill_ratio <= 1.0,
          "aggregate BF12 batch-fill telemetry is inconsistent");

  // An invalid submission must not poison subsequent sparse workspace reuse.
  const FlattenedInput invalid = flatten({{{}, {3}, {}, -1, 200}});
  BellmanFord12BatchOptions invalid_options;
  invalid_options.requested_batch_size = 1;
  invalid_options.controller_mode = BellmanFord12ControllerMode::HostBatch;
  require_rejected(
      [&] {
        (void)workspace.run_batch(invalid.sources, invalid.targets,
                                  invalid.descriptors, invalid_options);
      },
      "BF12 accepted an empty source list");
  const FlattenedInput empty_targets = flatten({{{0}, {}, {}, -1, 201}});
  require_rejected(
      [&] {
        (void)workspace.run_batch(empty_targets.sources,
                                  empty_targets.targets,
                                  empty_targets.descriptors, invalid_options);
      },
      "BF12 accepted an empty target list");
  compare_batch(graph, sidecars, dynamic_cost, &workspace, {queries[0]},
                BellmanFord12ControllerMode::HostBatch, 1, false);
}

void test_bounds_retry_and_iteration_limit() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 1, 1.0f}, {1, 3, 1.0f}, {3, 4, 1.0f}});
  BellmanFord12NodeSidecars sidecars = make_sidecars(5);
  sidecars.route_end_x = {0, 10, 2, 3, 4};
  BellmanFord12CsrWorkspace workspace(graph, sidecars, 3);
  const BellmanFord12BoundingBox bounds{0, 3, 0, 0, true};
  const BellmanFord12BoundingBox successful_bounds{3, 4, 0, 0, true};
  const FlattenedInput input = flatten({{{0}, {3}, bounds, -1, 300},
                                        {{3}, {4}, successful_bounds, -1, 301},
                                        {{0}, {4}, {}, 1, 302}});
  BellmanFord12BatchOptions options;
  options.requested_batch_size = 3;
  options.controller_mode = BellmanFord12ControllerMode::HostBatch;
  options.enable_bounding_boxes = true;
  options.enable_unbounded_retry = true;
  options.enable_telemetry = true;
  options.memory_safety_reserve_bytes = 1;
  const BellmanFord12BatchResult result = workspace.run_batch(
      input.sources, input.targets, input.descriptors, options);
  require(result.query_results[0].target_distances[0] == 2.0f,
          "bounded miss was not recovered unbounded");
  require(result.query_results[1].target_distances[0] == 1.0f &&
              result.query_results[1].target_path_nodes ==
                  std::vector<int>({3, 4}) &&
              result.query_statuses[1].paths_certified &&
              !result.query_statuses[1].bounded_search_miss,
          "nontrivial bounded-success path changed during selective retry");
  require(result.query_statuses[2].hit_iteration_limit &&
              !result.query_statuses[2].paths_certified,
          "maximum-iteration failure was not reported as tentative");
  require(workspace.telemetry().bounded_query_retry_count == 1,
          "BF12 did not retry exactly the missed bounded query");
}

void test_randomized_and_cooperative_equivalence() {
  std::mt19937 generator(0xBF12u);
  std::uniform_real_distribution<float> weight(0.0f, 5.0f);
  for (int trial = 0; trial < 8; ++trial) {
    constexpr int rows = 9;
    std::vector<std::tuple<int, int, float>> edges;
    for (int from = 0; from < rows; ++from) {
      for (int to = 0; to < rows; ++to) {
        if (from != to && (generator() % 5u) == 0u) {
          float value = weight(generator);
          if ((generator() & 7u) == 0u) value = 0.0f;
          edges.push_back({from, to, value});
        }
      }
    }
    const HostCsrF32 graph = make_graph(rows, edges);
    BellmanFord12NodeSidecars sidecars = make_sidecars(rows);
    std::vector<float> dynamic(rows, 1.0f);
    BellmanFord12CsrWorkspace workspace(graph, sidecars, 4);
    workspace.update_vertex_costs(dynamic);
    std::vector<QueryInput> queries;
    for (int query = 0; query < 4; ++query) {
      queries.push_back({{query, (query + 3) % rows},
                         {(query + 5) % rows, (query + 7) % rows},
                         {}, -1, static_cast<std::uint32_t>(400 + query)});
    }
    compare_batch(graph, sidecars, dynamic, &workspace, queries,
                  BellmanFord12ControllerMode::HostBatch, 4, false);
    if (workspace.cooperative_execution_supported()) {
      compare_batch(graph, sidecars, dynamic, &workspace, queries,
                    BellmanFord12ControllerMode::CooperativeBatch, 4, false);
    }
  }
}

}  // namespace

int main() {
  int device_count = 0;
  const hipError_t device_status = hipGetDeviceCount(&device_count);
  if (device_status != hipSuccess || device_count == 0) {
    std::cout << "BF12 batched HIP test skipped: no HIP device\n";
    return 0;
  }
  try {
    test_single_query_endpoint_semantics_and_exception_recovery();
    test_batch_sizes_and_independence();
    test_bounds_retry_and_iteration_limit();
    test_randomized_and_cooperative_equivalence();
    std::cout << "BF12 batched HIP test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF12 batched HIP test failed: " << error.what() << '\n';
    return 1;
  }
}
