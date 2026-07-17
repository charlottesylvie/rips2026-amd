// Production bf9 PathFinder-adapter regression test (requires an AMD HIP GPU).
// Build from the repository root:
//   hipcc -std=c++17 -O2 -pthread -x hip -DBF9_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/bf9_pathfinder_adapter_hip_test.cpp \
//     CongestionFreeRouting/bellman_ford/bf9.cpp \
//     -o /tmp/bf9_pathfinder_adapter_hip_test

#include "../bellman_ford/bf9.hpp"

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void bf9_internal_reset_full_state_counters();
extern "C" std::uint64_t bf9_internal_full_state_copy_count();
extern "C" std::uint64_t bf9_internal_full_state_fallback_count();
extern "C" void bf9_internal_reset_buffer_growth_counters();
extern "C" std::uint64_t bf9_internal_target_buffer_growth_count();
extern "C" std::uint64_t bf9_internal_path_buffer_growth_count();

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class HipStreamOwner {
 public:
  HipStreamOwner() {
    const hipError_t status =
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (status != hipSuccess) {
      throw std::runtime_error(std::string("hipStreamCreateWithFlags: ") +
                               hipGetErrorString(status));
    }
  }

  ~HipStreamOwner() {
    if (stream_) {
      (void)hipStreamDestroy(stream_);
    }
  }

  HipStreamOwner(const HipStreamOwner&) = delete;
  HipStreamOwner& operator=(const HipStreamOwner&) = delete;

  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

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

HostCsrF32 make_unit_weight_graph() {
  HostCsrF32 graph;
  graph.rows = 6;
  graph.cols = 6;
  graph.nnz = 6;
  graph.rowptr = {0, 2, 3, 4, 5, 6, 6};
  graph.colind = {1, 2, 3, 3, 4, 5};
  graph.values.assign(static_cast<std::size_t>(graph.nnz), 1.0f);
  return graph;
}

HostCsrF32 make_early_stop_graph() {
  HostCsrF32 graph;
  graph.rows = 4;
  graph.cols = 4;
  graph.nnz = 3;
  graph.rowptr = {0, 2, 2, 3, 3};
  graph.colind = {1, 2, 3};
  graph.values = {1.0f, 10.0f, 1.0f};
  return graph;
}

HostCsrF32 make_zero_edge_graph() {
  HostCsrF32 graph;
  graph.rows = 2;
  graph.cols = 2;
  graph.nnz = 0;
  graph.rowptr = {0, 0, 0};
  return graph;
}

HostCsrF32 make_batched_path_graph() {
  HostCsrF32 graph;
  graph.rows = 12;
  graph.cols = 12;
  graph.nnz = 11;
  graph.rowptr = {0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11};
  graph.colind = {
      1, 2,  // 0 -> 1, 2
      3, 4,  // 1 -> 3, 4
      5,     // 2 -> 5
      6,     // 3 -> 6
      7,     // 4 -> 7
      8,     // 5 -> 8
      9,     // 6 -> 9
      10,    // 7 -> 10
      10,    // 8 -> 10
  };
  graph.values.assign(static_cast<std::size_t>(graph.nnz), 1.0f);
  return graph;
}

HostCsrF32 make_source_selection_graph() {
  HostCsrF32 graph;
  graph.rows = 6;
  graph.cols = 6;
  graph.nnz = 4;
  graph.rowptr = {0, 1, 2, 3, 4, 4, 4};
  graph.colind = {4, 4, 5, 5};
  graph.values = {5.0f, 2.0f, 1.0f, 1.0f};
  return graph;
}

HostCsrF32 make_long_chain_graph() {
  HostCsrF32 graph;
  graph.rows = 66;
  graph.cols = 66;
  graph.nnz = 64;
  graph.rowptr.resize(67);
  graph.colind.reserve(64);
  graph.values.assign(64, 1.0f);
  for (int node = 0; node < 66; ++node) {
    graph.rowptr[static_cast<std::size_t>(node)] =
        static_cast<minplus_sparse::Offset>(graph.colind.size());
    if (node < 64) {
      graph.colind.push_back(node + 1);
    }
  }
  graph.rowptr[66] = static_cast<minplus_sparse::Offset>(graph.colind.size());
  return graph;
}

void require_float(float actual, float expected, const std::string& message) {
  require(std::fabs(actual - expected) <= 1e-6f, message);
}

}  // namespace

int main() {
  try {
    bf9_internal_reset_full_state_counters();
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

    const BellmanFordCsrResult zero_iterations = workspace.run(
        0, 4, 1.0f, 0, nullptr, nullptr, nullptr);
    require(zero_iterations.iterations_used == 0,
            "zero max iterations should launch no relaxation rounds");
    require(!zero_iterations.converged && !zero_iterations.stopped_on_target,
            "zero max iterations should report an iteration-limit stop");
    require(std::isinf(zero_iterations.target_distance),
            "zero max iterations should leave a non-source target unreachable");

    const BellmanFordCsrResult one_iteration = workspace.run(
        0, 1, 1.0f, 1, nullptr, nullptr, nullptr);
    require(one_iteration.iterations_used == 1,
            "one max iteration should launch exactly one relaxation round");
    require(!one_iteration.converged && !one_iteration.stopped_on_target,
            "a nonempty frontier at the limit should not report convergence");
    require_float(one_iteration.target_distance, 0.0f,
                  "one relaxation should discover the zero-weight neighbor");

    const HostCsrF32 unit_graph = make_unit_weight_graph();
    BellmanFordCsrWorkspace unit_workspace(unit_graph, nullptr);
    const BellmanFordCsrResult unit_result = unit_workspace.run(
        0, 4, 1.0f, -1, nullptr, nullptr, nullptr);
    require_float(unit_result.target_distance, 3.0f,
                  "unit-weight specialization should preserve distance");
    require(unit_result.target_path_nodes == std::vector<int>({0, 1, 3, 4}),
            "unit-weight specialization should preserve deterministic paths");
    require(unit_result.target_path_edges ==
                std::vector<minplus_sparse::Offset>({0, 2, 4}),
            "unit-weight specialization should preserve CSR edge IDs");

    const HostCsrF32 early_stop_graph = make_early_stop_graph();
    BellmanFordCsrWorkspace early_stop_workspace(early_stop_graph, nullptr);
    const BellmanFordCsrResult early_stop_result = early_stop_workspace.run(
        0, 1, 1.0f, -1, nullptr, nullptr, nullptr);
    require_float(early_stop_result.target_distance, 1.0f,
                  "distance-bound stop should preserve the target distance");
    require(early_stop_result.iterations_used == 2,
            "distance-bound graph should stop after two rounds");
    require(!early_stop_result.converged && early_stop_result.stopped_on_target,
            "distance-bound graph should report a target stop before convergence");

    BellmanFordCsrWorkspace zero_edge_workspace(make_zero_edge_graph(), nullptr);
    const BellmanFordCsrResult zero_edge_result = zero_edge_workspace.run(
        0, 1, 1.0f, -1, nullptr, nullptr, nullptr);
    require(std::isinf(zero_edge_result.target_distance) &&
                zero_edge_result.converged,
            "an empty graph should converge with its non-source target unreachable");

    const HostCsrF32 batched_graph = make_batched_path_graph();
    BellmanFordCsrWorkspace batched_workspace(batched_graph, nullptr);
    const BellmanFordCsrResult batched_result = batched_workspace.run(
        std::vector<int>{0, 2},
        std::vector<int>{9, 10, 4, 2, 11, 9},
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(batched_result.target_distances.size() == 6,
            "batched reconstruction should retain every target position");
    require_float(batched_result.target_distances[0], 4.0f,
                  "first batched path should have distance four");
    require_float(batched_result.target_distances[1], 3.0f,
                  "later source should provide a strictly shorter route");
    require_float(batched_result.target_distances[2], 2.0f,
                  "third batched path should have distance two");
    require_float(batched_result.target_distances[3], 0.0f,
                  "batched identity target should have distance zero");
    require(std::isinf(batched_result.target_distances[4]),
            "batched unreachable target should retain infinity");
    require_float(batched_result.target_distances[5], 4.0f,
                  "duplicate batched target should retain its distance");
    require(batched_result.target_sources ==
                std::vector<int>({0, 2, 0, 2, -1, 0}),
            "batched source choices should preserve shorter and identity winners");
    require(batched_result.target_path_offsets ==
                std::vector<int>({0, 5, 9, 12, 13, 13, 18}),
            "batched node offsets should preserve duplicates and unreachable targets");
    require(batched_result.target_edge_offsets ==
                std::vector<int>({0, 4, 7, 9, 9, 9, 13}),
            "batched edge offsets should stay one shorter than node paths");
    require(batched_result.target_path_nodes ==
                std::vector<int>({0, 1, 3, 6, 9,
                                  2, 5, 8, 10,
                                  0, 1, 4,
                                  2,
                                  0, 1, 3, 6, 9}),
            "batched predecessor rounds should reconstruct exact compact paths");
    require(batched_result.target_path_edges ==
                std::vector<minplus_sparse::Offset>({0, 2, 5, 8,
                                                     4, 7, 10,
                                                     0, 3,
                                                     0, 2, 5, 8}),
            "batched predecessor rounds should preserve exact CSR edge IDs");

    const HostCsrF32 selection_graph = make_source_selection_graph();
    BellmanFordCsrWorkspace selection_workspace(selection_graph, nullptr);
    const BellmanFordCsrResult shorter_later_source = selection_workspace.run(
        std::vector<int>{0, 1},
        4,
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require_float(shorter_later_source.target_distance, 2.0f,
                  "a later source should replace a strictly longer candidate");
    require(shorter_later_source.target_sources == std::vector<int>({1}) &&
                shorter_later_source.target_path_nodes ==
                    std::vector<int>({1, 4}),
            "strictly shorter later-source selection should retain its path");
    const BellmanFordCsrResult lower_id_later_source = selection_workspace.run(
        std::vector<int>{3, 2},
        5,
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require_float(lower_id_later_source.target_distance, 1.0f,
                  "equal-distance source candidates should retain their distance");
    require(lower_id_later_source.target_sources == std::vector<int>({2}) &&
                lower_id_later_source.target_path_nodes ==
                    std::vector<int>({2, 5}),
            "a later lower-ID source should win an equal-distance tie");
    const BellmanFordCsrResult higher_id_later_source = selection_workspace.run(
        std::vector<int>{2, 3},
        5,
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(higher_id_later_source.target_sources == std::vector<int>({2}) &&
                higher_id_later_source.target_path_nodes ==
                    std::vector<int>({2, 5}),
            "a later higher-ID equal-distance source should not replace the winner");

    bf9_internal_reset_buffer_growth_counters();
    const HostCsrF32 long_chain_graph = make_long_chain_graph();
    BellmanFordCsrWorkspace long_chain_workspace(long_chain_graph, nullptr);
    const BellmanFordCsrResult small_before_growth = long_chain_workspace.run(
        0, 2, 1.0f, -1, nullptr, nullptr, nullptr);
    require(small_before_growth.target_path_nodes ==
                std::vector<int>({0, 1, 2}),
            "small path should work before target/path scratch growth");
    require(bf9_internal_target_buffer_growth_count() == 1 &&
                bf9_internal_path_buffer_growth_count() == 1,
            "the first small request should allocate each compact scratch once");

    std::vector<int> long_chain_targets;
    for (int target = 1; target <= 64; ++target) {
      long_chain_targets.push_back(target);
    }
    const BellmanFordCsrResult long_chain_result = long_chain_workspace.run(
        std::vector<int>{0},
        long_chain_targets,
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(long_chain_result.target_path_nodes.size() == 2144,
            "long batched chains should contain the expected compact nodes");
    require(long_chain_result.target_path_edges.size() == 2080,
            "long batched chains should contain the expected compact edges");
    require(bf9_internal_target_buffer_growth_count() == 2 &&
                bf9_internal_path_buffer_growth_count() == 2,
            "the larger request should grow each compact scratch exactly once");
    for (int target = 1; target <= 64; ++target) {
      const std::size_t index = static_cast<std::size_t>(target - 1);
      require_float(long_chain_result.target_distances[index],
                    static_cast<float>(target),
                    "long chain distance should equal its target node");
      require(long_chain_result.target_sources[index] == 0,
              "long chain target should retain source zero");
      const int node_begin = long_chain_result.target_path_offsets[index];
      const int node_end = long_chain_result.target_path_offsets[index + 1];
      const int edge_begin = long_chain_result.target_edge_offsets[index];
      const int edge_end = long_chain_result.target_edge_offsets[index + 1];
      require(node_end - node_begin == target + 1,
              "long chain node slice should include both endpoints");
      require(edge_end - edge_begin == target,
              "long chain edge slice should be one shorter than its nodes");
      for (int node = 0; node <= target; ++node) {
        require(long_chain_result.target_path_nodes[
                    static_cast<std::size_t>(node_begin + node)] == node,
                "long chain node slice should be contiguous");
      }
      for (int edge = 0; edge < target; ++edge) {
        require(long_chain_result.target_path_edges[
                    static_cast<std::size_t>(edge_begin + edge)] == edge,
                "long chain edge slice should retain CSR IDs");
      }
    }
    const BellmanFordCsrResult repeated_large_batch = long_chain_workspace.run(
        std::vector<int>{0},
        long_chain_targets,
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require(repeated_large_batch.target_path_nodes ==
                long_chain_result.target_path_nodes &&
                repeated_large_batch.target_path_edges ==
                    long_chain_result.target_path_edges,
            "grown target/path scratch should be reusable without changing paths");
    require(bf9_internal_target_buffer_growth_count() == 2 &&
                bf9_internal_path_buffer_growth_count() == 2,
            "repeating the large request should reuse compact scratch buffers");
    const BellmanFordCsrResult small_after_growth = long_chain_workspace.run(
        0, 2, 1.0f, -1, nullptr, nullptr, nullptr);
    require(small_after_growth.target_path_nodes ==
                std::vector<int>({0, 1, 2}),
            "grown target/path scratch should remain reusable for smaller calls");
    require(bf9_internal_target_buffer_growth_count() == 2 &&
                bf9_internal_path_buffer_growth_count() == 2,
            "a smaller follow-up should not shrink or reallocate compact scratch");

    require(bf9_internal_full_state_copy_count() == 0,
            "normal target/path gathers must not copy the full best_state array");
    require(bf9_internal_full_state_fallback_count() == 0,
            "normal weighted, unit, batched, and long paths must not fall back");

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
            "non-null callbacks should be rejected without changing bf9's loop");

    const HostCsrF32 zero_cycle_graph =
        make_zero_weight_predecessor_cycle_graph();
    BellmanFordCsrWorkspace zero_cycle_workspace(zero_cycle_graph, nullptr);
    const BellmanFordCsrResult zero_cycle_result = zero_cycle_workspace.run(
        std::vector<int>{3},
        std::vector<int>{0, 1, 0},
        1.0f,
        -1,
        nullptr,
        nullptr,
        nullptr);
    require_float(zero_cycle_result.target_distances[0], 0.0f,
                  "zero-weight-cycle target distance should remain zero");
    require_float(zero_cycle_result.target_distances[1], 0.0f,
                  "second zero-weight-cycle target distance should remain zero");
    require_float(zero_cycle_result.target_distances[2], 0.0f,
                  "duplicate zero-weight-cycle target distance should remain zero");
    require(zero_cycle_result.target_sources ==
                std::vector<int>({3, 3, 3}) &&
                zero_cycle_result.target_reached,
            "cycle fallback should reach every target from the selected source");
    require(zero_cycle_result.target_path_offsets ==
                std::vector<int>({0, 2, 5, 7}) &&
                zero_cycle_result.target_edge_offsets ==
                    std::vector<int>({0, 1, 3, 4}),
            "cycle fallback should preserve duplicate target output positions");
    require(zero_cycle_result.target_path_nodes ==
                std::vector<int>({3, 0, 3, 0, 1, 3, 0}),
            "adapter should recover simple tight paths from cyclic packed predecessors");
    require(zero_cycle_result.target_path_edges ==
                std::vector<minplus_sparse::Offset>({2, 2, 0, 2}),
            "zero-weight-cycle fallback should retain outgoing CSR edge IDs");
    require(bf9_internal_full_state_fallback_count() == 1,
            "a packed predecessor cycle should trigger one source fallback");
    require(bf9_internal_full_state_copy_count() == 1,
            "a packed predecessor cycle should copy full state at most once per source");
    bf9_internal_reset_full_state_counters();

    HipStreamOwner stream_a;
    HipStreamOwner stream_b;
    require(stream_a.get() != nullptr && stream_b.get() != nullptr &&
                stream_a.get() != stream_b.get(),
            "parallel workspaces should use two distinct non-null HIP streams");
    {
      auto shared_graph =
          std::make_shared<BellmanFordCsrGraph>(batched_graph, nullptr);
      BellmanFordCsrWorkspace workspace_a(shared_graph, stream_a.get());
      BellmanFordCsrWorkspace workspace_b(shared_graph, stream_b.get());
      BellmanFordCsrResult result_a;
      BellmanFordCsrResult result_b;
      int shared_device = -1;
      require(hipGetDevice(&shared_device) == hipSuccess,
              "shared graph HIP device should be queryable");
      std::exception_ptr error_a;
      std::exception_ptr error_b;
      std::mutex start_mutex;
      std::condition_variable start_condition;
      int ready_threads = 0;
      bool start_threads = false;
      auto wait_for_start = [&] {
        std::unique_lock<std::mutex> lock(start_mutex);
        ++ready_threads;
        start_condition.notify_all();
        start_condition.wait(lock, [&] { return start_threads; });
      };
      std::thread thread_a([&] {
        try {
          wait_for_start();
          if (hipSetDevice(shared_device) != hipSuccess) {
            throw std::runtime_error("thread A could not select the graph HIP device");
          }
          result_a = workspace_a.run(
              std::vector<int>{0, 2},
              std::vector<int>{9, 10, 4, 2, 11, 9},
              1.0f,
              -1,
              stream_a.get(),
              nullptr,
              nullptr);
        } catch (...) {
          error_a = std::current_exception();
        }
      });
      std::thread thread_b([&] {
        try {
          wait_for_start();
          if (hipSetDevice(shared_device) != hipSuccess) {
            throw std::runtime_error("thread B could not select the graph HIP device");
          }
          result_b = workspace_b.run(
              std::vector<int>{2},
              std::vector<int>{8, 10, 2, 11, 8},
              1.0f,
              -1,
              stream_b.get(),
              nullptr,
              nullptr);
        } catch (...) {
          error_b = std::current_exception();
        }
      });
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [&] { return ready_threads == 2; });
        start_threads = true;
      }
      start_condition.notify_all();
      thread_a.join();
      thread_b.join();
      if (error_a) std::rethrow_exception(error_a);
      if (error_b) std::rethrow_exception(error_b);
      require(result_a.target_distances == batched_result.target_distances &&
                  result_a.target_sources == batched_result.target_sources &&
                  result_a.target_path_nodes == batched_result.target_path_nodes &&
                  result_a.target_path_edges == batched_result.target_path_edges,
              "first concurrent workspace should retain its complete compact result");
      require(result_b.target_distances.size() == 5,
              "second concurrent workspace should retain all target positions");
      require_float(result_b.target_distances[0], 2.0f,
                    "second workspace target eight distance should be two");
      require_float(result_b.target_distances[1], 3.0f,
                    "second workspace target ten distance should be three");
      require_float(result_b.target_distances[2], 0.0f,
                    "second workspace identity target should be zero");
      require(std::isinf(result_b.target_distances[3]),
              "second workspace unreachable target should retain infinity");
      require_float(result_b.target_distances[4], 2.0f,
                    "second workspace duplicate target should retain distance");
      require(result_b.target_sources ==
                  std::vector<int>({2, 2, 2, -1, 2}) &&
                  result_b.target_path_offsets ==
                      std::vector<int>({0, 3, 7, 8, 8, 11}) &&
                  result_b.target_edge_offsets ==
                      std::vector<int>({0, 2, 5, 5, 5, 7}) &&
                  result_b.target_path_nodes ==
                      std::vector<int>({2, 5, 8,
                                        2, 5, 8, 10,
                                        2,
                                        2, 5, 8}) &&
                  result_b.target_path_edges ==
                      std::vector<minplus_sparse::Offset>({4, 7,
                                                           4, 7, 10,
                                                           4, 7}),
              "second concurrent workspace should retain exact compact paths");

      bool wrong_stream_rejected = false;
      try {
        (void)workspace_a.run(
            0, 9, 1.0f, -1, stream_b.get(), nullptr, nullptr);
      } catch (const std::invalid_argument&) {
        wrong_stream_rejected = true;
      }
      require(wrong_stream_rejected,
              "a workspace should reject a stream other than its construction stream");
    }

    require(bf9_internal_full_state_copy_count() == 0 &&
                bf9_internal_full_state_fallback_count() == 0,
            "parallel stream-affine workspaces should use only compact gathers");

    std::cout << "bf9 PathFinder adapter HIP tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "bf9 PathFinder adapter HIP test failed: " << ex.what() << '\n';
    return 1;
  }
}
