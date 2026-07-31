// BF11 bounded/dynamic/true-multi-source regression test (AMD HIP GPU).
// Build from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
//     CongestionFreeRouting/bellman_ford/bf11.cpp \
//     -o /tmp/bf11_bounded_dynamic_hip_test
// Run on a ROCm host with a visible AMD GPU:
//   /tmp/bf11_bounded_dynamic_hip_test

#include "../bellman_ford/bf11.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" void bf11_internal_reset_counters();
extern "C" std::uint64_t bf11_internal_gpu_controller_launch_count();
extern "C" std::uint64_t bf11_internal_controller_fallback_count();
extern "C" std::uint64_t bf11_internal_target_check_count();
extern "C" std::uint64_t bf11_internal_auto_unbounded_retry_count();

namespace ri = routing::interchange;
using Offset = minplus_sparse::Offset;

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception = std::exception, typename Function>
void require_throws(const std::string& label, Function&& function) {
  bool caught = false;
  try {
    function();
  } catch (const Exception&) {
    caught = true;
  }
  require(caught, label + " was not rejected");
}

bool close_enough(float expected, float actual) {
  if (std::isinf(expected) || std::isinf(actual)) {
    return std::isinf(expected) && std::isinf(actual) &&
           std::signbit(expected) == std::signbit(actual);
  }
  if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <= 1e-5f * scale;
}

struct EdgeSpec {
  int from = -1;
  int to = -1;
  float value = 0.0f;
};

HostCsrF32 make_graph(int vertex_count, const std::vector<EdgeSpec>& edges) {
  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
  for (const EdgeSpec& edge : edges) {
    require(edge.from >= 0 && edge.from < vertex_count && edge.to >= 0 &&
                edge.to < vertex_count,
            "test edge lies outside its graph");
    require(std::isfinite(edge.value) && edge.value >= 0.0f,
            "test edge has an invalid value");
    ++graph.rowptr[static_cast<std::size_t>(edge.from) + 1];
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex + 1)] +=
        graph.rowptr[static_cast<std::size_t>(vertex)];
  }
  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
  for (const EdgeSpec& edge : edges) {
    const std::size_t row = static_cast<std::size_t>(edge.from);
    const std::size_t position = static_cast<std::size_t>(cursor[row]++);
    graph.colind[position] = edge.to;
    graph.values[position] = edge.value;
  }
  return graph;
}

ri::RoutingCsrSidecars make_sidecars(
    std::vector<std::int32_t> x,
    std::vector<std::int32_t> y,
    std::vector<float> base_cost = {}) {
  require(x.size() == y.size(), "test coordinate columns have different sizes");
  if (base_cost.empty()) base_cost.assign(x.size(), 1.0f);
  ri::RoutingCsrSidecars sidecars;
  sidecars.route_end_x = std::move(x);
  sidecars.route_end_y = std::move(y);
  sidecars.base_vertex_cost = std::move(base_cost);
  return sidecars;
}

bool admitted(const ri::RoutingCsrSidecars& sidecars,
              int node,
              const BellmanFord11BoundingBox& bounds) {
  if (!bounds.enabled) return true;
  const std::int32_t x = sidecars.route_end_x[static_cast<std::size_t>(node)];
  const std::int32_t y = sidecars.route_end_y[static_cast<std::size_t>(node)];
  if (!ri::has_route_coordinate(x, y)) return true;
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

float effective_weight(const HostCsrF32& graph,
                       const ri::RoutingCsrSidecars& sidecars,
                       const std::vector<float>& dynamic_cost,
                       Offset edge) {
  const std::size_t index = static_cast<std::size_t>(edge);
  const int destination = graph.colind[index];
  const float edge_cost = graph.values[index];
  const float multiplier =
      dynamic_cost[static_cast<std::size_t>(destination)];
  if (edge_cost == 0.0f || multiplier == 0.0f) return 0.0f;
  return edge_cost *
         sidecars.base_vertex_cost[static_cast<std::size_t>(destination)] *
         multiplier;
}

std::vector<float> cpu_bounded_dijkstra(
    const HostCsrF32& graph,
    const ri::RoutingCsrSidecars& sidecars,
    const std::vector<float>& dynamic_cost,
    const std::vector<int>& sources,
    const BellmanFord11BoundingBox& bounds = {}) {
  std::vector<float> distance(static_cast<std::size_t>(graph.rows), kInfinity);
  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  for (const int source : sources) {
    if (distance[static_cast<std::size_t>(source)] != 0.0f) {
      distance[static_cast<std::size_t>(source)] = 0.0f;
      queue.push({0.0f, source});
    }
  }
  while (!queue.empty()) {
    const auto [du, from] = queue.top();
    queue.pop();
    if (du != distance[static_cast<std::size_t>(from)]) continue;
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
         edge < graph.rowptr[static_cast<std::size_t>(from + 1)]; ++edge) {
      const int to = graph.colind[static_cast<std::size_t>(edge)];
      if (!admitted(sidecars, to, bounds)) continue;
      const float candidate =
          du + effective_weight(graph, sidecars, dynamic_cost, edge);
      float& current = distance[static_cast<std::size_t>(to)];
      if (candidate < current) {
        current = candidate;
        queue.push({candidate, to});
      }
    }
  }
  return distance;
}

void validate_paths(const std::string& label,
                    const HostCsrF32& graph,
                    const ri::RoutingCsrSidecars& sidecars,
                    const std::vector<float>& dynamic_cost,
                    const std::vector<int>& sources,
                    const std::vector<int>& targets,
                    const BellmanFordCsrResult& result,
                    const BellmanFord11BoundingBox& bounds = {}) {
  const std::vector<float> expected = cpu_bounded_dijkstra(
      graph, sidecars, dynamic_cost, sources, bounds);
  require(result.target_distances.size() == targets.size() &&
              result.target_sources.size() == targets.size() &&
              result.target_path_offsets.size() == targets.size() + 1 &&
              result.target_edge_offsets.size() == targets.size() + 1,
          label + ": compact target array sizes are inconsistent");
  require(result.target_path_offsets.front() == 0 &&
              result.target_edge_offsets.front() == 0,
          label + ": compact offsets do not begin at zero");
  require(result.target_path_edge_costs.size() ==
              result.target_path_edges.size(),
          label + ": effective edge costs are not aligned with compact edges");

  bool all_reached = true;
  for (std::size_t target_index = 0; target_index < targets.size();
       ++target_index) {
    const int target = targets[target_index];
    const float expected_distance = expected[static_cast<std::size_t>(target)];
    const float actual_distance = result.target_distances[target_index];
    const int node_begin = result.target_path_offsets[target_index];
    const int node_end = result.target_path_offsets[target_index + 1];
    const int edge_begin = result.target_edge_offsets[target_index];
    const int edge_end = result.target_edge_offsets[target_index + 1];
    require(node_begin >= 0 && node_end >= node_begin && edge_begin >= 0 &&
                edge_end >= edge_begin &&
                static_cast<std::size_t>(node_end) <=
                    result.target_path_nodes.size() &&
                static_cast<std::size_t>(edge_end) <=
                    result.target_path_edges.size(),
            label + ": compact target slice is invalid");
    if (!std::isfinite(expected_distance)) {
      all_reached = false;
      require(std::isinf(actual_distance) &&
                  result.target_sources[target_index] == -1 &&
                  node_begin == node_end && edge_begin == edge_end,
              label + ": unreachable target contains finite path data");
      continue;
    }

    require(close_enough(expected_distance, actual_distance),
            label + ": target distance disagrees with bounded CPU Dijkstra");
    require(node_end - node_begin == edge_end - edge_begin + 1 &&
                node_end > node_begin,
            label + ": reached compact path has inconsistent lengths");
    const int root = result.target_sources[target_index];
    require(std::find(sources.begin(), sources.end(), root) != sources.end() &&
                result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                    root &&
                result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                    target,
            label + ": compact path has the wrong root or target");

    float path_cost = 0.0f;
    for (int position = edge_begin; position < edge_end; ++position) {
      const int path_offset = position - edge_begin;
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset + 1)];
      const Offset edge =
          result.target_path_edges[static_cast<std::size_t>(position)];
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from + 1)] &&
                  graph.colind[static_cast<std::size_t>(edge)] == to,
              label + ": compact path edge is not an original CSR edge");
      require(admitted(sidecars, to, bounds),
              label + ": compact path traverses a known node outside its box");
      const float expected_edge_cost =
          effective_weight(graph, sidecars, dynamic_cost, edge);
      const float reported_edge_cost =
          result.target_path_edge_costs[static_cast<std::size_t>(position)];
      require(close_enough(expected_edge_cost, reported_edge_cost),
              label + ": compact path reports the wrong effective edge cost");
      path_cost += reported_edge_cost;
    }
    require(close_enough(expected_distance, path_cost) &&
                close_enough(actual_distance, path_cost),
            label + ": compact path does not sum to its effective cost");
  }
  require(static_cast<std::size_t>(result.target_path_offsets.back()) ==
                  result.target_path_nodes.size() &&
              static_cast<std::size_t>(result.target_edge_offsets.back()) ==
                  result.target_path_edges.size(),
          label + ": final compact offsets do not match storage");
  require(result.target_reached == all_reached,
          label + ": aggregate target reachability is inconsistent");
}

void test_validation_and_dynamic_updates() {
  const HostCsrF32 graph = make_graph(
      4, {{0, 1, 0.5f}, {0, 2, 1.0f}, {1, 3, 1.0f}, {2, 3, 1.0f}});
  const ri::RoutingCsrSidecars sidecars = make_sidecars(
      {0, 1, 1, 2}, {0, 0, 1, 0}, {1.0f, 2.0f, 1.5f, 2.0f});

  require_throws("short BF11 coordinate sidecar", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.route_end_x.pop_back();
    BellmanFord11CsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("half-missing BF11 coordinate", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.route_end_x[0] = ri::kMissingRouteCoordinate;
    BellmanFord11CsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("zero BF11 base cost", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.base_vertex_cost[0] = 0.0f;
    BellmanFord11CsrGraph rejected(graph, invalid, nullptr);
  });

  auto shared_graph =
      std::make_shared<BellmanFord11CsrGraph>(graph, sidecars, nullptr);
  BellmanFord11CsrWorkspace first(shared_graph, nullptr);
  BellmanFord11CsrWorkspace independent(shared_graph, nullptr);
  const std::vector<float> unit_dynamic(4, 1.0f);
  const BellmanFordCsrResult baseline = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 999.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 factored-cost baseline", graph, sidecars, unit_dynamic,
                 {0}, {3}, baseline);
  require(baseline.target_path_nodes == std::vector<int>({0, 1, 3}),
          "base costs did not select the expected first diamond arm");

  const std::vector<float> changed_dynamic = {1.0f, 4.0f, 0.5f, 1.0f};
  first.update_vertex_costs(std::vector<float>(changed_dynamic), nullptr);
  const BellmanFordCsrResult changed = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 0.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 full dynamic-cost update", graph, sidecars,
                 changed_dynamic, {0}, {3}, changed);
  require(changed.target_path_nodes == std::vector<int>({0, 2, 3}),
          "dynamic costs did not flip the selected diamond arm");

  const BellmanFordCsrResult still_independent = independent.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 workspace-local dynamic costs", graph, sidecars,
                 unit_dynamic, {0}, {3}, still_independent);
  require(still_independent.target_path_nodes == baseline.target_path_nodes,
          "one workspace's dynamic update leaked into a shared-graph peer");

  first.update_vertex_costs(unit_dynamic, nullptr);
  first.update_vertex_costs_sparse({1, 2}, {4.0f, 0.5f}, nullptr);
  const BellmanFordCsrResult sparse = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 sparse dynamic-cost update", graph, sidecars,
                 changed_dynamic, {0}, {3}, sparse);
  require(sparse.target_path_nodes == changed.target_path_nodes,
          "sparse and full dynamic-cost updates disagree");

  require_throws<std::invalid_argument>("short full dynamic-cost vector", [&] {
    first.update_vertex_costs({1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("negative dynamic cost", [&] {
    first.update_vertex_costs({1.0f, -1.0f, 1.0f, 1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("NaN dynamic cost", [&] {
    first.update_vertex_costs(
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f},
        nullptr);
  });
  require_throws<std::invalid_argument>("mismatched sparse update columns", [&] {
    first.update_vertex_costs_sparse({1, 2}, {1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("duplicate sparse update node", [&] {
    first.update_vertex_costs_sparse({1, 1}, {1.0f, 2.0f}, nullptr);
  });
  require_throws<std::out_of_range>("out-of-range sparse update node", [&] {
    first.update_vertex_costs_sparse({4}, {1.0f}, nullptr);
  });

  const float largest = std::numeric_limits<float>::max();
  const HostCsrF32 zero_scaled_graph =
      make_graph(2, {{0, 1, largest}});
  const ri::RoutingCsrSidecars zero_scaled_sidecars =
      make_sidecars({0, 1}, {0, 0}, {1.0f, largest});
  BellmanFord11CsrWorkspace zero_scaled(zero_scaled_graph,
                                       zero_scaled_sidecars, nullptr);
  zero_scaled.update_vertex_costs({1.0f, 0.0f}, nullptr);
  const BellmanFordCsrResult zero_scaled_result = zero_scaled.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 exact zero dynamic multiplier", zero_scaled_graph,
                 zero_scaled_sidecars, {1.0f, 0.0f}, {0}, {1},
                 zero_scaled_result);
  require(zero_scaled_result.target_distances == std::vector<float>({0.0f}),
          "zero dynamic multiplier was lost to intermediate overflow");

  // The shared graph owns all host data needed after its synchronous upload.
  // Mutating and destroying the caller's carrier must not alter later bounds
  // or costs.
  std::shared_ptr<BellmanFord11CsrGraph> owned_graph;
  {
    ri::RoutingCsrSidecars ephemeral =
        make_sidecars({0, 1}, {0, 0}, {1.0f, 3.0f});
    owned_graph = std::make_shared<BellmanFord11CsrGraph>(
        make_graph(2, {{0, 1, 2.0f}}), ephemeral, nullptr);
    ephemeral.route_end_x.assign(2, ri::kMissingRouteCoordinate);
    ephemeral.route_end_y.assign(2, ri::kMissingRouteCoordinate);
    ephemeral.base_vertex_cost.assign(2, 99.0f);
  }
  BellmanFord11WorkspaceOptions owned_options;
  owned_options.auto_bounds = true;
  owned_options.auto_margin_x = 0;
  owned_options.auto_margin_y = 0;
  BellmanFord11CsrWorkspace owned_workspace(
      std::move(owned_graph), nullptr, owned_options);
  const BellmanFordCsrResult owned_result = owned_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  require(owned_result.stopped_on_target &&
              close_enough(6.0f, owned_result.target_distances[0]) &&
              owned_result.target_path_nodes == std::vector<int>({0, 1}),
          "BF11 retained dangling or caller-mutable graph sidecars");
}

void test_true_multi_source() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 3, 4.0f}, {1, 3, 1.0f}, {2, 1, 0.0f}, {3, 4, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 4}, {0, 0, 0, 0, 0});
  BellmanFord11CsrWorkspace workspace(graph, sidecars, nullptr);
  const std::vector<int> sources = {0, 1, 2, 1};
  const std::vector<int> targets = {3, 1, 4, 3};
  const std::vector<float> dynamic(5, 1.0f);

  bf11_internal_reset_counters();
  const BellmanFordCsrResult result = workspace.run(
      sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  validate_paths("BF11 true multi-source", graph, sidecars, dynamic,
                 sources, targets, result);
  require(result.target_sources == std::vector<int>({1, 1, 1, 1}),
          "true multi-source routing selected the wrong protected root");
  require(result.target_path_nodes ==
              std::vector<int>({1, 3, 1, 1, 3, 4, 1, 3}),
          "true multi-source compact paths or identity target changed");
  require(bf11_internal_gpu_controller_launch_count() +
                  bf11_internal_controller_fallback_count() ==
              1,
          "one true multi-source query used more than one controller run");
  require(result.iterations_used <= graph.rows,
          "true multi-source iterations look like a sum of source runs");

  const BellmanFordCsrResult repeated = workspace.run(
      sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  require(repeated.target_distances == result.target_distances &&
              repeated.target_sources == result.target_sources &&
              repeated.target_path_nodes == result.target_path_nodes &&
              repeated.target_path_edges == result.target_path_edges,
          "true multi-source result is not deterministic across reuse");

  const BellmanFordCsrResult zero_round_identity = workspace.run(
      std::vector<int>{1}, std::vector<int>{1}, 1.0f, 0,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 zero-round identity target", graph, sidecars, dynamic,
                 {1}, {1}, zero_round_identity);
  require(zero_round_identity.iterations_used == 0 &&
              zero_round_identity.stopped_on_target &&
              !zero_round_identity.converged &&
              zero_round_identity.target_path_nodes == std::vector<int>({1}),
          "BF11 lost a source-equals-target path at max_iters=0");
}

void test_explicit_bounds_and_missing_spill() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 4, 0.1f}, {0, 1, 1.0f}, {1, 2, 1.0f},
          {2, 3, 1.0f}, {4, 3, 0.1f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 1}, {0, 0, 0, 0, 10});
  BellmanFord11CsrWorkspace workspace(graph, sidecars, nullptr);
  const std::vector<float> dynamic(5, 1.0f);
  BellmanFord11RunOptions bounded_options;
  bounded_options.bounds = {true, 0, 3, 0, 0};

  const BellmanFordCsrResult bounded = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      bounded_options, nullptr, nullptr, nullptr);
  validate_paths("BF11 explicit fixed box", graph, sidecars, dynamic,
                 {0}, {3}, bounded, bounded_options.bounds);
  require(close_enough(3.0f, bounded.target_distances[0]) &&
              bounded.target_path_nodes == std::vector<int>({0, 1, 2, 3}),
          "bounded BF11 did not choose the exact path inside its fixed box");

  const BellmanFordCsrResult unbounded = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 bounded-to-unbounded reuse", graph, sidecars, dynamic,
                 {0}, {3}, unbounded);
  require(close_enough(0.2f, unbounded.target_distances[0]) &&
              unbounded.target_path_nodes == std::vector<int>({0, 4, 3}),
          "unbounded BF11 did not recover the cheaper out-of-box path");

  const BellmanFordCsrResult bounded_again = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      bounded_options, nullptr, nullptr, nullptr);
  require(bounded_again.target_path_nodes == bounded.target_path_nodes,
          "an unbounded run leaked into later explicit bound state");

  BellmanFord11RunOptions inverted = bounded_options;
  inverted.bounds.min_x = 4;
  inverted.bounds.max_x = 3;
  require_throws<std::invalid_argument>("inverted BF11 bounds", [&] {
    (void)workspace.run({0}, {3}, 1.0f, -1, inverted,
                        nullptr, nullptr, nullptr);
  });
  BellmanFord11RunOptions terminal_outside = bounded_options;
  terminal_outside.bounds.max_x = 2;
  require_throws<std::invalid_argument>("target outside BF11 bounds", [&] {
    (void)workspace.run({0}, {3}, 1.0f, -1, terminal_outside,
                        nullptr, nullptr, nullptr);
  });

  const HostCsrF32 spill_graph =
      make_graph(3, {{0, 1, 1.0f}, {1, 2, 1.0f}});
  const ri::RoutingCsrSidecars spill_sidecars = make_sidecars(
      {0, ri::kMissingRouteCoordinate, 2},
      {0, ri::kMissingRouteCoordinate, 0});
  BellmanFord11CsrWorkspace spill_workspace(
      spill_graph, spill_sidecars, nullptr);
  BellmanFord11RunOptions tight;
  tight.bounds = {true, 0, 2, 0, 0};
  const BellmanFordCsrResult through_spill = spill_workspace.run(
      std::vector<int>{0}, std::vector<int>{2}, 1.0f, -1,
      tight, nullptr, nullptr, nullptr);
  validate_paths("BF11 missing-coordinate spill admission", spill_graph,
                 spill_sidecars, std::vector<float>(3, 1.0f), {0}, {2},
                 through_spill, tight.bounds);
  require(through_spill.target_path_nodes == std::vector<int>({0, 1, 2}),
          "bounded BF11 excluded a missing-coordinate spill resource");
  require_throws<std::invalid_argument>("missing-coordinate BF11 terminal", [&] {
    (void)spill_workspace.run({0}, {1}, 1.0f, -1, tight,
                              nullptr, nullptr, nullptr);
  });
}

void test_auto_bounds_and_fallback() {
  const HostCsrF32 graph =
      make_graph(3, {{0, 2, 0.5f}, {2, 1, 0.5f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 2, 1}, {0, 0, 10});

  BellmanFord11WorkspaceOptions bounded_only_options;
  bounded_only_options.auto_bounds = true;
  bounded_only_options.auto_margin_x = 0;
  bounded_only_options.auto_margin_y = 0;
  BellmanFord11CsrWorkspace bounded_only(
      graph, sidecars, nullptr, bounded_only_options);
  const BellmanFordCsrResult miss = bounded_only.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  require(!miss.target_reached && std::isinf(miss.target_distances[0]),
          "auto-bounded BF11 unexpectedly crossed an excluded detour");

  BellmanFord11WorkspaceOptions fallback_options = bounded_only_options;
  fallback_options.unbounded_fallback = true;
  BellmanFord11CsrWorkspace fallback(
      graph, sidecars, nullptr, fallback_options);
  bf11_internal_reset_counters();
  const BellmanFordCsrResult recovered = fallback.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 automatic unbounded fallback", graph, sidecars,
                 std::vector<float>(3, 1.0f), {0}, {1}, recovered);
  require(close_enough(1.0f, recovered.target_distances[0]) &&
              recovered.target_path_nodes == std::vector<int>({0, 2, 1}),
          "BF11 automatic fallback did not recover the out-of-box detour");
  require(bf11_internal_auto_unbounded_retry_count() == 1 &&
              bf11_internal_gpu_controller_launch_count() +
                      bf11_internal_controller_fallback_count() ==
                  2,
          "BF11 fallback did not execute exactly one bounded and one unbounded run");

  const HostCsrF32 missing_terminal_graph =
      make_graph(2, {{0, 1, 1.0f}});
  const ri::RoutingCsrSidecars missing_terminal_sidecars = make_sidecars(
      {0, ri::kMissingRouteCoordinate},
      {0, ri::kMissingRouteCoordinate});
  BellmanFord11CsrWorkspace missing_terminal_workspace(
      missing_terminal_graph, missing_terminal_sidecars, nullptr,
      fallback_options);
  bf11_internal_reset_counters();
  const BellmanFordCsrResult missing_terminal =
      missing_terminal_workspace.run(
          std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
          nullptr, nullptr, nullptr);
  validate_paths("BF11 missing-terminal direct unbounded fallback",
                 missing_terminal_graph, missing_terminal_sidecars,
                 std::vector<float>(2, 1.0f), {0}, {1}, missing_terminal);
  require(bf11_internal_auto_unbounded_retry_count() == 0 &&
              bf11_internal_gpu_controller_launch_count() +
                      bf11_internal_controller_fallback_count() ==
                  1,
          "a missing auto-bound terminal did not select one unbounded first run");

  const HostCsrF32 missing_source_graph = make_graph(
      3, {{0, 1, 1.0f}, {0, 2, 5.0f}, {1, 2, 1.0f}});
  const ri::RoutingCsrSidecars missing_source_sidecars = make_sidecars(
      {ri::kMissingRouteCoordinate, 10, 2},
      {ri::kMissingRouteCoordinate, 0, 0});
  BellmanFord11CsrWorkspace missing_source_workspace(
      missing_source_graph, missing_source_sidecars, nullptr,
      fallback_options);
  const BellmanFordCsrResult missing_source = missing_source_workspace.run(
      std::vector<int>{0}, std::vector<int>{2}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("BF11 missing-source bounded seed", missing_source_graph,
                 missing_source_sidecars, std::vector<float>(3, 1.0f), {0},
                 {2}, missing_source, {true, 2, 2, 0, 0});
  require(close_enough(5.0f, missing_source.target_distances[0]) &&
              missing_source.target_path_nodes == std::vector<int>({0, 2}),
          "a missing-coordinate route-tree source forced an unbounded first run");
}

BellmanFordCsrResult run_with_interval(BellmanFord11CsrWorkspace& workspace,
                                       int interval) {
  BellmanFord11RunOptions options;
  options.target_check_interval = interval;
  return workspace.run(std::vector<int>{0}, std::vector<int>{1},
                       1.0f, -1, options, nullptr, nullptr, nullptr);
}

void test_target_check_interval_and_settlement() {
  const HostCsrF32 interval_graph = make_graph(
      7, {{0, 1, 1.0f}, {0, 2, 100.0f}, {2, 3, 1.0f},
          {3, 4, 1.0f}, {4, 5, 1.0f}, {5, 6, 1.0f}});
  const ri::RoutingCsrSidecars interval_sidecars =
      make_sidecars({0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0});
  BellmanFord11CsrWorkspace workspace(
      interval_graph, interval_sidecars, nullptr);
  const std::vector<float> dynamic(7, 1.0f);

  bf11_internal_reset_counters();
  const BellmanFordCsrResult every_round = run_with_interval(workspace, 1);
  const std::uint64_t every_round_checks =
      bf11_internal_target_check_count();
  validate_paths("BF11 target checks every round", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_round);

  bf11_internal_reset_counters();
  const BellmanFordCsrResult every_two = run_with_interval(workspace, 2);
  const std::uint64_t every_two_checks = bf11_internal_target_check_count();
  validate_paths("BF11 target checks every two rounds", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_two);

  bf11_internal_reset_counters();
  const BellmanFordCsrResult every_four = run_with_interval(workspace, 4);
  const std::uint64_t every_four_checks = bf11_internal_target_check_count();
  validate_paths("BF11 target checks every four rounds", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_four);

  require(every_round.target_path_nodes == every_two.target_path_nodes &&
              every_round.target_path_nodes == every_four.target_path_nodes &&
              every_round.target_distances == every_two.target_distances &&
              every_round.target_distances == every_four.target_distances,
          "target-check interval changed the certified path");
  require(every_round.iterations_used == 1 && every_two.iterations_used == 2 &&
              every_four.iterations_used == 4,
          "target-check interval did not only delay early termination");
  require(every_round_checks == 1 && every_two_checks == 1 &&
              every_four_checks == 1,
          "BF11 did not scan targets at the configured round interval");
  require(every_round.stopped_on_target && every_two.stopped_on_target &&
              every_four.stopped_on_target,
          "an exact distance certificate did not report a target stop");
  require_throws<std::invalid_argument>("zero BF11 target-check interval", [&] {
    (void)run_with_interval(workspace, 0);
  });

  // A direct expensive target label is tentative: the two-edge route improves
  // it in the following round. A check interval must never turn discovery into
  // settlement, including when the frontier converges before the first check.
  const HostCsrF32 tentative_graph =
      make_graph(3, {{0, 1, 10.0f}, {0, 2, 1.0f}, {2, 1, 1.0f}});
  const ri::RoutingCsrSidecars tentative_sidecars =
      make_sidecars({0, 2, 1}, {0, 0, 0});
  BellmanFord11CsrWorkspace tentative_workspace(
      tentative_graph, tentative_sidecars, nullptr);
  for (const int interval : {1, 2, 4}) {
    const BellmanFordCsrResult result =
        run_with_interval(tentative_workspace, interval);
    validate_paths("BF11 tentative-target settlement interval " +
                       std::to_string(interval),
                   tentative_graph, tentative_sidecars,
                   std::vector<float>(3, 1.0f), {0}, {1}, result);
    require(close_enough(2.0f, result.target_distances[0]) &&
                result.target_path_nodes == std::vector<int>({0, 2, 1}),
            "BF11 accepted a tentative direct target label");
    require(result.converged || result.stopped_on_target,
            "BF11 returned an exact tentative-target path without certifying it");
  }

  const BellmanFordCsrResult iteration_limited = tentative_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, 1,
      nullptr, nullptr, nullptr);
  require(!iteration_limited.converged &&
              !iteration_limited.stopped_on_target &&
              iteration_limited.target_reached &&
              close_enough(10.0f, iteration_limited.target_distances[0]) &&
              iteration_limited.target_path_nodes ==
                  std::vector<int>({0, 1}),
          "BF11 mislabeled a finite max-iteration result as certified");

  // The target is first discovered on the final legal BF round. The >=
  // nonnegative-distance certificate must win over the max-iteration branch;
  // a strict comparison would return the right label but fail to certify it.
  constexpr int kChainVertices = 6;
  std::vector<EdgeSpec> chain_edges;
  for (int node = 0; node + 1 < kChainVertices; ++node) {
    chain_edges.push_back({node, node + 1, 1.0f});
  }
  const HostCsrF32 chain_graph = make_graph(kChainVertices, chain_edges);
  const ri::RoutingCsrSidecars chain_sidecars = make_sidecars(
      {0, 1, 2, 3, 4, 5}, {0, 0, 0, 0, 0, 0});
  BellmanFord11CsrWorkspace chain_workspace(
      chain_graph, chain_sidecars, nullptr);
  const BellmanFordCsrResult chain = chain_workspace.run(
      std::vector<int>{0}, std::vector<int>{kChainVertices - 1},
      1.0f, -1, nullptr, nullptr, nullptr);
  validate_paths("BF11 V-1 target certificate", chain_graph, chain_sidecars,
                 std::vector<float>(kChainVertices, 1.0f), {0},
                 {kChainVertices - 1}, chain);
  require(chain.iterations_used == kChainVertices - 1 &&
              chain.stopped_on_target && !chain.converged,
          "BF11 failed to certify a target first reached on round V-1");
}

}  // namespace

int main() {
  try {
    test_validation_and_dynamic_updates();
    test_true_multi_source();
    test_explicit_bounds_and_missing_spill();
    test_auto_bounds_and_fallback();
    test_target_check_interval_and_settlement();
    std::cout << "BF11 bounded dynamic HIP tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 bounded dynamic HIP test failed: " << error.what()
              << '\n';
    return 1;
  }
}
