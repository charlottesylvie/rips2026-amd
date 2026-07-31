#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace routing::interchange {

// Device tiles use nonnegative integer coordinates.  Keep one explicit
// sentinel for routing resources that have no physical tile coordinate; such
// resources are placed in the spatial spill shard and must remain admissible
// to preserve routing completeness.
constexpr std::int32_t kMissingRouteCoordinate = -1;

struct RoutingBoundingBox {
  std::int32_t min_x = 0;
  std::int32_t max_x = -1;
  std::int32_t min_y = 0;
  std::int32_t max_y = -1;

  bool valid() const {
    return min_x <= max_x && min_y <= max_y;
  }
};

// Every filtered CSR edge appears exactly once, in the shard selected by its
// destination node's representative route-end coordinate.  Known coordinate
// shards are a dense, row-major tile grid.  One final spill shard contains all
// edges whose destination has no physical coordinate.
struct SpatialEdgeShards {
  std::int32_t min_x = 0;
  std::int32_t min_y = 0;
  std::uint64_t width = 0;
  std::uint64_t height = 0;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> edge_ids;

  std::uint64_t regular_shard_count() const {
    if (width != 0 && height >
                          std::numeric_limits<std::uint64_t>::max() / width) {
      throw std::overflow_error("spatial shard grid size overflows uint64");
    }
    return width * height;
  }

  std::uint64_t spill_shard() const {
    return regular_shard_count();
  }
};

struct RoutingCsrSidecars {
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_cost;
  SpatialEdgeShards spatial_edges;
};

inline bool has_route_coordinate(std::int32_t x, std::int32_t y) {
  return x != kMissingRouteCoordinate && y != kMissingRouteCoordinate;
}

inline std::uint64_t maximum_dense_spatial_cells(std::size_t node_count) {
  constexpr std::uint64_t kMinimumSpatialCellBudget = 1ULL << 20;
  return node_count >
                 (std::numeric_limits<std::uint64_t>::max() -
                  kMinimumSpatialCellBudget) /
                     4
             ? std::numeric_limits<std::uint64_t>::max()
             : kMinimumSpatialCellBudget +
                   4 * static_cast<std::uint64_t>(node_count);
}

inline SpatialEdgeShards build_destination_spatial_edge_shards(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    const std::vector<std::int32_t>& csr_destinations) {
  if (route_end_x.size() != route_end_y.size()) {
    throw std::invalid_argument(
        "route-end coordinate arrays do not match for spatial sharding");
  }
  if (csr_destinations.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error(
        "spatial edge IDs require filtered CSR nnz to fit uint32");
  }

  const std::size_t node_count = route_end_x.size();
  std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_x = -1;
  std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_y = -1;
  for (std::size_t node = 0; node < node_count; ++node) {
    const std::int32_t x = route_end_x[node];
    const std::int32_t y = route_end_y[node];
    const bool x_missing = x == kMissingRouteCoordinate;
    const bool y_missing = y == kMissingRouteCoordinate;
    if (x_missing != y_missing || (!x_missing && (x < 0 || y < 0))) {
      throw std::invalid_argument(
          "invalid route-end coordinate pair for spatial sharding");
    }
    if (x_missing) {
      continue;
    }
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }

  SpatialEdgeShards shards;
  std::uint64_t regular_shard_count = 0;
  if (max_x >= 0 && max_y >= 0) {
    shards.min_x = min_x;
    shards.min_y = min_y;
    shards.width = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(max_x) - min_x + 1);
    shards.height = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(max_y) - min_y + 1);
    regular_shard_count = shards.regular_shard_count();

    // A corrupt legacy artifact with two extremely distant coordinates must
    // not trigger an unbounded dense-grid allocation. Real FPGA tile grids are
    // compact; allow at least one million cells plus four per routing node.
    if (regular_shard_count > maximum_dense_spatial_cells(node_count)) {
      throw std::runtime_error(
          "route-end coordinate grid is implausibly sparse for dense spatial shards");
    }
  }
  if (regular_shard_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() - 2)) {
    throw std::overflow_error("spatial shard count exceeds host size_t");
  }

  const std::size_t spill_shard =
      static_cast<std::size_t>(regular_shard_count);
  const std::size_t shard_count = spill_shard + 1;
  std::vector<std::uint64_t> counts(shard_count, 0);
  auto shard_for_destination = [&](std::int32_t destination) {
    if (destination < 0 ||
        static_cast<std::size_t>(destination) >= node_count) {
      throw std::invalid_argument(
          "CSR contains an invalid spatial-shard destination");
    }
    const std::size_t node = static_cast<std::size_t>(destination);
    const std::int32_t x = route_end_x[node];
    const std::int32_t y = route_end_y[node];
    if (!has_route_coordinate(x, y)) {
      return spill_shard;
    }
    const std::uint64_t local_x =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(x) - shards.min_x);
    const std::uint64_t local_y =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(y) - shards.min_y);
    if (local_x >= shards.width || local_y >= shards.height) {
      throw std::logic_error(
          "valid route coordinate fell outside its spatial grid");
    }
    return static_cast<std::size_t>(local_y * shards.width + local_x);
  };

  for (const std::int32_t destination : csr_destinations) {
    ++counts[shard_for_destination(destination)];
  }
  shards.offsets.resize(shard_count + 1, 0);
  for (std::size_t shard = 0; shard < shard_count; ++shard) {
    shards.offsets[shard + 1] = shards.offsets[shard] + counts[shard];
  }
  shards.edge_ids.resize(csr_destinations.size());
  std::vector<std::uint64_t> cursors(shards.offsets.begin(),
                                     shards.offsets.end() - 1);
  for (std::size_t edge = 0; edge < csr_destinations.size(); ++edge) {
    const std::size_t shard = shard_for_destination(csr_destinations[edge]);
    const std::uint64_t output = cursors[shard]++;
    shards.edge_ids[static_cast<std::size_t>(output)] =
        static_cast<std::uint32_t>(edge);
  }
  return shards;
}

inline void validate_routing_csr_sidecars(
    const RoutingCsrSidecars& sidecars,
    std::size_t vertex_count,
    std::size_t edge_count,
    bool spatial_index_required = true) {
  if (edge_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error(
        "routing CSR spatial edge IDs require nnz to fit uint32");
  }
  if (sidecars.route_end_x.size() != vertex_count ||
      sidecars.route_end_y.size() != vertex_count ||
      sidecars.base_vertex_cost.size() != vertex_count) {
    throw std::runtime_error(
        "routing CSR node sidecar arrays do not match vertex count");
  }
  bool have_known_coordinate = false;
  std::int32_t known_min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t known_max_x = -1;
  std::int32_t known_min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t known_max_y = -1;
  for (std::size_t node = 0; node < vertex_count; ++node) {
    const std::int32_t x = sidecars.route_end_x[node];
    const std::int32_t y = sidecars.route_end_y[node];
    const bool x_missing = x == kMissingRouteCoordinate;
    const bool y_missing = y == kMissingRouteCoordinate;
    if (x_missing != y_missing || (!x_missing && (x < 0 || y < 0))) {
      throw std::runtime_error(
          "routing CSR contains an invalid route-end coordinate pair");
    }
    if (!x_missing) {
      have_known_coordinate = true;
      known_min_x = std::min(known_min_x, x);
      known_max_x = std::max(known_max_x, x);
      known_min_y = std::min(known_min_y, y);
      known_max_y = std::max(known_max_y, y);
    }
    const float cost = sidecars.base_vertex_cost[node];
    if (!std::isfinite(cost) || !(cost > 0.0f)) {
      throw std::runtime_error(
          "routing CSR base vertex costs must be finite and positive");
    }
  }

  const SpatialEdgeShards& shards = sidecars.spatial_edges;
  if (shards.offsets.empty()) {
    if (spatial_index_required || !shards.edge_ids.empty() ||
        shards.width != 0 || shards.height != 0) {
      throw std::runtime_error("routing CSR spatial edge shards are missing");
    }
    return;
  }
  if ((shards.width == 0) != (shards.height == 0)) {
    throw std::runtime_error(
        "routing CSR spatial shard grid dimensions are inconsistent");
  }
  if (have_known_coordinate) {
    const std::uint64_t expected_width = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(known_max_x) - known_min_x + 1);
    const std::uint64_t expected_height = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(known_max_y) - known_min_y + 1);
    if (shards.min_x != known_min_x || shards.min_y != known_min_y ||
        shards.width != expected_width || shards.height != expected_height) {
      throw std::runtime_error(
          "routing CSR spatial shard grid does not cover node coordinates exactly");
    }
  } else if (shards.width != 0 || shards.height != 0) {
    throw std::runtime_error(
        "routing CSR has regular spatial shards without known coordinates");
  }
  if (shards.width != 0) {
    if (shards.min_x < 0 || shards.min_y < 0 ||
        shards.width > static_cast<std::uint64_t>(
                           std::numeric_limits<std::int32_t>::max()) -
                               static_cast<std::uint64_t>(shards.min_x) + 1 ||
        shards.height > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int32_t>::max()) -
                                static_cast<std::uint64_t>(shards.min_y) + 1) {
      throw std::runtime_error(
          "routing CSR spatial shard grid exceeds coordinate range");
    }
  }
  const std::uint64_t regular_shards = shards.regular_shard_count();
  if (regular_shards > maximum_dense_spatial_cells(vertex_count)) {
    throw std::runtime_error(
        "routing CSR spatial shard grid is implausibly sparse");
  }
  if (regular_shards >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() - 2)) {
    throw std::runtime_error(
        "routing CSR spatial shard offset count exceeds host size_t");
  }
  const std::size_t expected_offsets =
      static_cast<std::size_t>(regular_shards) + 2;
  if (shards.offsets.size() != expected_offsets || shards.offsets.front() != 0 ||
      shards.offsets.back() != shards.edge_ids.size() ||
      shards.edge_ids.size() != edge_count) {
    throw std::runtime_error(
        "routing CSR spatial shard arrays do not match edge count");
  }
  std::uint64_t previous = 0;
  for (const std::uint64_t offset : shards.offsets) {
    if (offset < previous || offset > shards.edge_ids.size()) {
      throw std::runtime_error(
          "routing CSR spatial shard offsets are not monotone");
    }
    previous = offset;
  }

  // Count plus range checks are not sufficient to detect a duplicated edge ID
  // paired with one omitted ID.  A compact bitset makes the permutation
  // invariant authoritative without another edge-sized integer allocation.
  const std::size_t bit_words = edge_count / 64 + (edge_count % 64 != 0);
  std::vector<std::uint64_t> seen(bit_words, 0);
  for (const std::uint32_t edge : shards.edge_ids) {
    if (static_cast<std::uint64_t>(edge) >= edge_count) {
      throw std::runtime_error(
          "routing CSR spatial shard contains an invalid edge ID");
    }
    const std::size_t index = static_cast<std::size_t>(edge);
    const std::uint64_t mask = std::uint64_t{1} << (index & 63);
    std::uint64_t& word = seen[index >> 6];
    if ((word & mask) != 0) {
      throw std::runtime_error(
          "routing CSR spatial shard contains a duplicate edge ID");
    }
    word |= mask;
  }
}

// Validate the semantic half of the shard index as well as its structural
// permutation: each compact edge ID must live in the shard selected by that
// edge's destination coordinate. This check is intended for artifact creation
// and full shard loading; node-sidecar-only BF11 loads deliberately skip the
// unused edge permutation.
inline void validate_destination_spatial_edge_shards(
    const RoutingCsrSidecars& sidecars,
    std::size_t vertex_count,
    const std::vector<std::int32_t>& csr_destinations) {
  validate_routing_csr_sidecars(sidecars, vertex_count,
                                csr_destinations.size(), true);
  const SpatialEdgeShards& shards = sidecars.spatial_edges;
  const std::uint64_t regular_shards = shards.regular_shard_count();
  const std::uint64_t spill = regular_shards;
  for (std::uint64_t shard = 0; shard <= spill; ++shard) {
    const std::uint64_t begin =
        shards.offsets[static_cast<std::size_t>(shard)];
    const std::uint64_t end =
        shards.offsets[static_cast<std::size_t>(shard + 1)];
    for (std::uint64_t position = begin; position < end; ++position) {
      const std::uint32_t edge =
          shards.edge_ids[static_cast<std::size_t>(position)];
      const std::int32_t destination =
          csr_destinations[static_cast<std::size_t>(edge)];
      if (destination < 0 ||
          static_cast<std::size_t>(destination) >=
              sidecars.route_end_x.size()) {
        throw std::runtime_error(
            "routing CSR spatial shard refers to an invalid destination");
      }
      const std::size_t node = static_cast<std::size_t>(destination);
      const std::int32_t x = sidecars.route_end_x[node];
      const std::int32_t y = sidecars.route_end_y[node];
      std::uint64_t expected = spill;
      if (has_route_coordinate(x, y)) {
        if (shards.width == 0 || shards.height == 0) {
          throw std::runtime_error(
              "known route coordinate has no regular spatial shard grid");
        }
        const std::uint64_t local_x = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(x) - shards.min_x);
        const std::uint64_t local_y = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(y) - shards.min_y);
        if (local_x >= shards.width || local_y >= shards.height) {
          throw std::runtime_error(
              "route coordinate falls outside the spatial shard grid");
        }
        expected = local_y * shards.width + local_x;
      }
      if (shard != expected) {
        throw std::runtime_error(
            "routing CSR edge is stored in the wrong destination shard");
      }
    }
  }
}

// Return every known-coordinate tile shard intersecting an inclusive box,
// followed by the conservative unknown-coordinate spill shard.  An empty grid
// has only the spill shard.  Callers with no spatial sidecar receive an empty
// list and should use their ordinary unbounded traversal.
inline std::vector<std::uint64_t> spatial_shards_for_box(
    const SpatialEdgeShards& shards,
    const RoutingBoundingBox& box) {
  if (shards.offsets.empty() || !box.valid()) {
    return {};
  }
  std::vector<std::uint64_t> selected;
  const std::uint64_t spill = shards.spill_shard();
  if (shards.width == 0 || shards.height == 0) {
    selected.push_back(spill);
    return selected;
  }

  const std::int64_t grid_max_x =
      static_cast<std::int64_t>(shards.min_x) +
      static_cast<std::int64_t>(shards.width) - 1;
  const std::int64_t grid_max_y =
      static_cast<std::int64_t>(shards.min_y) +
      static_cast<std::int64_t>(shards.height) - 1;
  const std::int64_t min_x =
      std::max<std::int64_t>(box.min_x, shards.min_x);
  const std::int64_t max_x =
      std::min<std::int64_t>(box.max_x, grid_max_x);
  const std::int64_t min_y =
      std::max<std::int64_t>(box.min_y, shards.min_y);
  const std::int64_t max_y =
      std::min<std::int64_t>(box.max_y, grid_max_y);
  if (min_x <= max_x && min_y <= max_y) {
    const std::uint64_t selected_width =
        static_cast<std::uint64_t>(max_x - min_x + 1);
    const std::uint64_t selected_height =
        static_cast<std::uint64_t>(max_y - min_y + 1);
    const std::uint64_t host_max =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (selected_width != 0 &&
        selected_height > (host_max - 1) / selected_width) {
      throw std::overflow_error("selected spatial shard count overflows size_t");
    }
    const std::uint64_t selected_count =
        selected_width * selected_height;
    selected.reserve(static_cast<std::size_t>(selected_count + 1));
    for (std::int64_t y = min_y; y <= max_y; ++y) {
      const std::uint64_t row =
          static_cast<std::uint64_t>(y - shards.min_y) * shards.width;
      for (std::int64_t x = min_x; x <= max_x; ++x) {
        selected.push_back(
            row + static_cast<std::uint64_t>(x - shards.min_x));
      }
    }
  }
  selected.push_back(spill);
  return selected;
}

}  // namespace routing::interchange
