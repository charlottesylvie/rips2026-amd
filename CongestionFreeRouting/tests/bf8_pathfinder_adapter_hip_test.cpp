// Production bf8 PathFinder-adapter regression test (requires an AMD HIP GPU).
// Build from the repository root:
//   hipcc -std=c++17 -O2 -x hip -DBF8_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/bf8_pathfinder_adapter_hip_test.cpp \
//     CongestionFreeRouting/bellman_ford/bf8.cpp \
//     -o /tmp/bf8_pathfinder_adapter_hip_test

#include "../bellman_ford/bf8.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

HostCsrF32 make_weighted_graph() {
  HostCsrF32 graph;
  graph.rows = 7;
  graph.cols = 7;
  graph.nnz = 8;
  graph.rowptr = {0, 3, 5, 6, 7, 8, 8, 8};
  graph.colind = {
      1, 2, 3,  // 0 -> 1, 2, 3
      2, 3,     // 1 -> 2, 3
      4,        // 2 -> 4
      4,        // 3 -> 4
      5,        // 4 -> 5
  };
  graph.values = {
      0.0f, 1.0f, 1.0f,
      1.0f, 1.0f,
      2.0f,
      2.0f,
      1.0f,
  };
  return graph;
}

HostCsrF32 make_zero_weight_predecessor_cycle_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 3;
  graph.rowptr = {0, 1, 2, 2, 3};
  graph.colind = {1, 0, 0};
  graph.values = {0.0f, 0.0f, 0.0f};
  return graph;
}

void require_float(float actual, float expected, const std::string& message) {
  require(std::fabs(actual - expected) <= 1e-6f, message);
}

}  // namespace

int main() {
  try {
    const HostCsrF32 graph = make_weighted_graph();
    BellmanFordCsrWorkspace workspace(graph, nullptr);

    const BellmanFordCsrResult one_target = workspace.run(
        std::vector<int>{0},
        std::vector<int>{4},
        123.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(one_target.target_distances.size() == 1,
            "single-target result should contain one distance");
    require_float(one_target.target_distances[0], 3.0f,
                  "weighted 0->4 distance should be three");
    require(one_target.target_sources == std::vector<int>{0},
            "single-target result should retain source zero");
    require(one_target.target_path_offsets == std::vector<int>({0, 3}),
            "single-target node offsets should delimit one three-node path");
    require(one_target.target_edge_offsets == std::vector<int>({0, 2}),
            "single-target edge offsets should delimit two edges");
    require(one_target.target_path_nodes == std::vector<int>({0, 2, 4}),
            "packed predecessor tie-breaking should choose 0->2->4");
    require(one_target.target_path_edges ==
                std::vector<minplus_sparse::Offset>({1, 5}),
            "single-target path should retain outgoing CSR edge IDs");

    const BellmanFordCsrResult many_targets = workspace.run(
        std::vector<int>{1, 0, 1},
        std::vector<int>{4, 4, 1, 5, 6},
        0.001f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(many_targets.target_distances.size() == 5,
            "duplicate targets must retain separate result positions");
    require_float(many_targets.target_distances[0], 3.0f,
                  "first target four distance should be three");
    require_float(many_targets.target_distances[1], 3.0f,
                  "duplicate target four distance should be three");
    require_float(many_targets.target_distances[2], 0.0f,
                  "source-equals-target distance should be zero");
    require_float(many_targets.target_distances[3], 4.0f,
                  "target five distance should be four");
    require(std::isinf(many_targets.target_distances[4]),
            "unreachable target should retain infinity");
    require(many_targets.target_sources == std::vector<int>({0, 0, 1, 0, -1}),
            "multi-source ties should be deterministic and prefer identity paths");
    require(many_targets.target_path_offsets ==
                std::vector<int>({0, 3, 6, 7, 11, 11}),
            "node offsets should preserve duplicate and unreachable targets");
    require(many_targets.target_edge_offsets ==
                std::vector<int>({0, 2, 4, 4, 7, 7}),
            "edge offsets should preserve duplicate and unreachable targets");
    require(many_targets.target_path_nodes ==
                std::vector<int>({0, 2, 4,
                                  0, 2, 4,
                                  1,
                                  0, 2, 4, 5}),
            "compact node paths should preserve every requested target position");
    require(many_targets.target_path_edges ==
                std::vector<minplus_sparse::Offset>({1, 5,
                                                     1, 5,
                                                     1, 5, 7}),
            "compact edge paths should remain aligned with outgoing CSR");
    require(!many_targets.target_reached,
            "one unreachable target should clear aggregate target_reached");

    const BellmanFordCsrResult single_overload = workspace.run(
        0, 5, 999.0f, -1, nullptr, nullptr, nullptr);
    require(single_overload.target == 5,
            "single-target overload should report its target");
    require_float(single_overload.target_distance, 4.0f,
                  "single-target overload should report its target distance");

    bool callback_rejected = false;
    auto callback = [](const BellmanFordCsrProgress&, void*) {};
    try {
      (void)workspace.run(std::vector<int>{0},
                          std::vector<int>{4},
                          1.0f,
                          -1,
                          nullptr,
                          callback,
                          nullptr);
    } catch (const std::invalid_argument&) {
      callback_rejected = true;
    }
    require(callback_rejected,
            "non-null callbacks should be rejected without changing bf8's loop");

    const HostCsrF32 zero_cycle_graph =
        make_zero_weight_predecessor_cycle_graph();
    BellmanFordCsrWorkspace zero_cycle_workspace(zero_cycle_graph, nullptr);
    const BellmanFordCsrResult zero_cycle_result = zero_cycle_workspace.run(
        std::vector<int>{3},
        std::vector<int>{0},
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require_float(zero_cycle_result.target_distances[0], 0.0f,
                  "zero-weight-cycle target distance should remain zero");
    require(zero_cycle_result.target_path_nodes == std::vector<int>({3, 0}),
            "adapter should recover a simple tight path from cyclic packed predecessors");
    require(zero_cycle_result.target_path_edges ==
                std::vector<minplus_sparse::Offset>({2}),
            "zero-weight-cycle fallback should retain the outgoing CSR edge ID");

    std::cout << "bf8 PathFinder adapter HIP tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "bf8 PathFinder adapter HIP test failed: " << ex.what() << '\n';
    return 1;
  }
}
