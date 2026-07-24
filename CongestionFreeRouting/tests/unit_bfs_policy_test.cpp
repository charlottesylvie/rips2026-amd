#include "../unit_bfs/unit_bfs_policy.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
  bool threw = false;
  try {
    callback();
  } catch (const Exception&) {
    threw = true;
  }
  require(threw, message);
}

void test_compact_offset_scan() {
  using unit_bfs_policy::scan_target_node_counts;
  const auto zero = scan_target_node_counts({});
  require(zero.node_offsets == std::vector<int>{0} &&
              zero.edge_offsets == std::vector<int>{0} &&
              zero.total_nodes == 0 && zero.total_edges == 0,
          "empty target scan must publish zero totals");

  const auto singleton = scan_target_node_counts({1});
  require(singleton.node_offsets == std::vector<int>({0, 1}) &&
              singleton.edge_offsets == std::vector<int>({0, 0}),
          "singleton source-target path offsets are incorrect");

  const auto mixed = scan_target_node_counts({4, 0, 2});
  require(mixed.node_offsets == std::vector<int>({0, 4, 4, 6}) &&
              mixed.edge_offsets == std::vector<int>({0, 3, 3, 4}) &&
              mixed.total_nodes == 6 && mixed.total_edges == 4,
          "mixed reachable/unreachable scan is incorrect");

  const auto duplicate = scan_target_node_counts({3, 3, 0});
  require(duplicate.node_offsets == std::vector<int>({0, 3, 6, 6}) &&
              duplicate.edge_offsets == std::vector<int>({0, 2, 4, 4}),
          "duplicate targets must retain separate ordered slices");

  require_throws<std::invalid_argument>(
      [] { (void)scan_target_node_counts({-1}); },
      "negative target lengths must be rejected");
  require_throws<std::overflow_error>(
      [] {
        (void)scan_target_node_counts(
            {std::numeric_limits<int>::max(), 1});
      },
      "compact path total overflow must be rejected");
}

void test_capacity_policy() {
  unit_bfs_policy::CapacityPlan plan;
  plan.reserve_queries({0, 0});
  require(plan.sources == 0 && plan.targets == 0 &&
              plan.target_offsets == 0,
          "zero hints must not reserve query buffers");
  plan.reserve_queries({3, 4});
  require(plan.sources == 3 && plan.targets == 4 &&
              plan.target_offsets == 5,
          "capacity hints must reserve source/target-derived storage");
  require(plan.compact_path_nodes == 0 && plan.compact_path_edges == 0,
          "capacity hints must not reserve compact paths");

  plan.grow_queries(4, 6);
  const std::size_t grown_sources = plan.sources;
  const std::size_t grown_targets = plan.targets;
  const std::size_t grown_offsets = plan.target_offsets;
  require(grown_sources >= 4 && grown_targets >= 6 && grown_offsets >= 7,
          "query capacity must grow to the requested size");
  plan.grow_queries(1, 2);
  require(plan.sources == grown_sources && plan.targets == grown_targets &&
              plan.target_offsets == grown_offsets,
          "smaller queries must not request shrinkage");

  plan.grow_compact_paths(5, 4);
  const std::size_t node_high_water = plan.compact_path_nodes;
  const std::size_t edge_high_water = plan.compact_path_edges;
  plan.grow_compact_paths(1, 0);
  require(plan.compact_path_nodes == node_high_water &&
              plan.compact_path_edges == edge_high_water,
          "compact path high-water capacity must be retained");

  require_throws<std::overflow_error>(
      [] {
        unit_bfs_policy::CapacityPlan too_large;
        too_large.grow_queries(
            static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1,
            0);
      },
      "device query-count overflow must be rejected");
  require_throws<std::overflow_error>(
      [] {
        (void)sssp_capacity::checked_bytes<std::uint64_t>(
            std::numeric_limits<std::size_t>::max());
      },
      "allocation byte overflow must be rejected");
}

void test_generation_model() {
  unit_bfs_policy::GenerationVisitationModel model(6);
  require(!model.begin_query() && model.generation() == 1,
          "first query must use generation one without rollover");
  require(model.claim(0, 0), "source claim must succeed");
  require(model.claim(1, 1), "first discovery must succeed");
  require(!model.claim(1, 1) && !model.claim(1, 2),
          "duplicate discovery claims must fail within one query");
  require(model.level(1) == 1, "winning level must remain stable");
  require(!model.level(5).has_value(),
          "unreachable vertex must remain unvisited");

  // Early stop leaves claimed entries in place; the next token makes them stale.
  require(model.claim(2, 2), "early-stop frontier claim must succeed");
  require(!model.begin_query() && model.generation() == 2,
          "reused workspace must advance its token");
  require(!model.level(0).has_value() && !model.level(2).has_value(),
          "prior-query entries must be stale without clearing");
  require(model.claim(2, 0), "stale vertex must be claimable in a later query");

  model.abort_query();
  require(!model.begin_query(), "abort reuse must advance without full reset");
  require(!model.level(2).has_value(),
          "aborted-query entries must be stale on reuse");
  require(model.claim(4, 0), "workspace must remain usable after abort");

  model.set_generation_for_test(std::numeric_limits<std::uint32_t>::max());
  require(model.begin_query(), "token rollover must request a full reset");
  require(model.generation() == 1,
          "token rollover must restart at generation one");
  require(!model.level(4).has_value(),
          "rollover full reset must remove aliased old entries");
  require(model.claim(4, 0), "vertex must be claimable after rollover reset");
}

}  // namespace

int main() {
  try {
    test_compact_offset_scan();
    test_capacity_policy();
    test_generation_model();
    std::cout << "Unit-BFS host policy test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Unit-BFS host policy test failed: " << error.what() << '\n';
    return 1;
  }
}
