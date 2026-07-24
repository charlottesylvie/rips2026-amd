#pragma once

#include "../sssp_query_capacity.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace unit_bfs_policy {

inline std::size_t bounded_geometric_capacity(std::size_t current,
                                              std::size_t required,
                                              std::size_t limit) {
  if (current > limit || required > limit) {
    throw std::overflow_error("unit BFS capacity exceeds its device limit");
  }
  if (current >= required) return current;
  return std::min(sssp_capacity::geometric_capacity(current, required), limit);
}

struct CapacityPlan {
  std::size_t sources = 0;
  std::size_t targets = 0;
  std::size_t target_offsets = 0;
  std::size_t compact_path_nodes = 0;
  std::size_t compact_path_edges = 0;

  void reserve_queries(const SsspQueryCapacityHints& hints) {
    sssp_capacity::validate_reservation(hints);
    grow_queries(hints.max_sources, hints.max_targets);
  }

  void grow_queries(std::size_t source_count, std::size_t target_count) {
    const std::size_t device_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    sources = bounded_geometric_capacity(
        sources, sssp_capacity::checked_device_count(source_count), device_limit);
    targets = bounded_geometric_capacity(
        targets, sssp_capacity::checked_device_count(target_count), device_limit);
    if (targets != 0) {
      target_offsets = sssp_capacity::checked_target_offset_count(targets);
    }
  }

  void grow_compact_paths(std::size_t node_count, std::size_t edge_count) {
    const std::size_t device_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    compact_path_nodes = bounded_geometric_capacity(
        compact_path_nodes,
        sssp_capacity::checked_device_count(node_count),
        device_limit);
    compact_path_edges = bounded_geometric_capacity(
        compact_path_edges,
        sssp_capacity::checked_device_count(edge_count),
        device_limit);
  }
};

struct CompactOffsetScan {
  std::vector<int> node_offsets;
  std::vector<int> edge_offsets;
  int total_nodes = 0;
  int total_edges = 0;
};

// A zero node count represents an unreachable target. Repeated counts are
// retained in input order, matching duplicate-target extraction semantics.
inline CompactOffsetScan scan_target_node_counts(
    const std::vector<int>& node_counts) {
  CompactOffsetScan scan;
  scan.node_offsets.reserve(
      sssp_capacity::checked_target_offset_count(node_counts.size()));
  scan.edge_offsets.reserve(
      sssp_capacity::checked_target_offset_count(node_counts.size()));
  scan.node_offsets.push_back(0);
  scan.edge_offsets.push_back(0);

  std::size_t total_nodes = 0;
  std::size_t total_edges = 0;
  const std::size_t device_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  for (const int node_count : node_counts) {
    if (node_count < 0) {
      throw std::invalid_argument("unit BFS target node count is negative");
    }
    const std::size_t nodes = static_cast<std::size_t>(node_count);
    const std::size_t edges = nodes == 0 ? 0 : nodes - 1;
    total_nodes = sssp_capacity::checked_add(total_nodes, nodes);
    total_edges = sssp_capacity::checked_add(total_edges, edges);
    if (total_nodes > device_limit || total_edges > device_limit) {
      throw std::overflow_error("compact unit BFS target paths are too large");
    }
    scan.node_offsets.push_back(static_cast<int>(total_nodes));
    scan.edge_offsets.push_back(static_cast<int>(total_edges));
  }
  scan.total_nodes = static_cast<int>(total_nodes);
  scan.total_edges = static_cast<int>(total_edges);
  return scan;
}

struct GenerationEntry {
  std::uint32_t generation = 0;
  int level = 0;
};

// Sequential model for the packed device visitation invariant. It deliberately
// models only first-discovery membership; queue/controller behavior remains
// covered by the HIP integration suite.
class GenerationVisitationModel {
 public:
  explicit GenerationVisitationModel(std::size_t vertex_count)
      : entries_(vertex_count) {}

  // Returns true when rollover required a full reset before generation one.
  bool begin_query() {
    if (generation_ == std::numeric_limits<std::uint32_t>::max()) {
      std::fill(entries_.begin(), entries_.end(), GenerationEntry{});
      generation_ = 1;
      return true;
    }
    ++generation_;
    return false;
  }

  bool claim(std::size_t vertex, int level) {
    if (vertex >= entries_.size()) {
      throw std::out_of_range("unit BFS visitation model vertex is out of range");
    }
    if (generation_ == 0 || level < 0) {
      throw std::logic_error("unit BFS visitation claim has no active query");
    }
    GenerationEntry& entry = entries_[vertex];
    if (entry.generation == generation_) return false;
    entry = {generation_, level};
    return true;
  }

  std::optional<int> level(std::size_t vertex) const {
    if (vertex >= entries_.size()) {
      throw std::out_of_range("unit BFS visitation model vertex is out of range");
    }
    const GenerationEntry& entry = entries_[vertex];
    if (entry.generation != generation_) return std::nullopt;
    return entry.level;
  }

  void abort_query() noexcept {
    // No per-entry cleanup is required. The next generation makes every entry
    // from the aborted query stale.
  }

  std::uint32_t generation() const noexcept { return generation_; }

  void set_generation_for_test(std::uint32_t generation) noexcept {
    generation_ = generation;
  }

 private:
  std::vector<GenerationEntry> entries_;
  std::uint32_t generation_ = 0;
};

}  // namespace unit_bfs_policy
