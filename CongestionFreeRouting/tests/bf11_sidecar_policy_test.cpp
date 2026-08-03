#include "../interchange/routing_csr_sidecars.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ri = routing::interchange;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_rejected(const std::string& label, Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, label + " was accepted");
}

ri::RoutingCsrSidecars make_sidecars() {
  ri::RoutingCsrSidecars sidecars;
  sidecars.route_end_x = {10, 11, 10, 12, ri::kMissingRouteCoordinate};
  sidecars.route_end_y = {20, 20, 21, 22, ri::kMissingRouteCoordinate};
  sidecars.base_vertex_cost = {1.0f, 1.25f, 2.0f, 3.5f, 7.0f};

  // Three by three regular tile shards plus one spill shard. Every edge ID
  // appears exactly once, but the edge order is deliberately not numeric so
  // validation cannot accidentally rely on a sorted permutation.
  sidecars.spatial_edges.min_x = 10;
  sidecars.spatial_edges.min_y = 20;
  sidecars.spatial_edges.width = 3;
  sidecars.spatial_edges.height = 3;
  sidecars.spatial_edges.offsets = {0, 2, 3, 3, 4, 4, 4, 4, 4, 5, 7};
  sidecars.spatial_edges.edge_ids = {0, 4, 1, 2, 3, 5, 6};
  return sidecars;
}

void test_valid_sidecars_and_selection() {
  const ri::RoutingCsrSidecars sidecars = make_sidecars();
  ri::validate_routing_csr_sidecars(sidecars, 5, 7);
  require(sidecars.spatial_edges.regular_shard_count() == 9 &&
              sidecars.spatial_edges.spill_shard() == 9,
          "spatial shard count or spill index changed");

  require(ri::has_route_coordinate(0, 0),
          "origin should be a valid route coordinate");
  require(!ri::has_route_coordinate(ri::kMissingRouteCoordinate,
                                    ri::kMissingRouteCoordinate),
          "paired missing coordinates should not be considered physical");

  const std::vector<std::uint64_t> first_row =
      ri::spatial_shards_for_box(sidecars.spatial_edges, {10, 11, 20, 20});
  require(first_row == std::vector<std::uint64_t>({0, 1, 9}),
          "inclusive first-row box selected the wrong regular/spill shards");

  const std::vector<std::uint64_t> clipped =
      ri::spatial_shards_for_box(sidecars.spatial_edges, {9, 10, 19, 21});
  require(clipped == std::vector<std::uint64_t>({0, 3, 9}),
          "partially out-of-grid box did not clip to intersecting shards");

  const std::vector<std::uint64_t> outside =
      ri::spatial_shards_for_box(sidecars.spatial_edges, {-5, -1, -5, -1});
  require(outside == std::vector<std::uint64_t>({9}),
          "out-of-grid box must retain only the conservative spill shard");

  require(ri::spatial_shards_for_box(sidecars.spatial_edges, {4, 3, 0, 0})
              .empty(),
          "an invalid box should not select spatial shards");

  ri::SpatialEdgeShards spill_only;
  spill_only.offsets = {0, 2};
  spill_only.edge_ids = {1, 0};
  ri::RoutingCsrSidecars no_regular = sidecars;
  no_regular.spatial_edges = spill_only;
  no_regular.route_end_x.assign(2, ri::kMissingRouteCoordinate);
  no_regular.route_end_y.assign(2, ri::kMissingRouteCoordinate);
  no_regular.base_vertex_cost.assign(2, 1.0f);
  ri::validate_routing_csr_sidecars(no_regular, 2, 2);
  require(ri::spatial_shards_for_box(spill_only, {0, 0, 0, 0}) ==
              std::vector<std::uint64_t>({0}),
          "an empty regular grid should still expose its spill shard");

  ri::RoutingCsrSidecars legacy;
  legacy.route_end_x.assign(2, ri::kMissingRouteCoordinate);
  legacy.route_end_y.assign(2, ri::kMissingRouteCoordinate);
  legacy.base_vertex_cost.assign(2, 1.0f);
  ri::validate_routing_csr_sidecars(legacy, 2, 0, false);
  require(ri::spatial_shards_for_box(legacy.spatial_edges, {0, 1, 0, 1})
              .empty(),
          "an absent legacy spatial index should select no shards");
}

void test_node_sidecar_validation() {
  const ri::RoutingCsrSidecars valid = make_sidecars();

  require_rejected("short route-end X column", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.route_end_x.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("short route-end Y column", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.route_end_y.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("short base-cost column", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.base_vertex_cost.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("half-missing coordinate", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.route_end_x[0] = ri::kMissingRouteCoordinate;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("negative physical coordinate", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.route_end_x[0] = -2;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  for (const float invalid_cost :
       {0.0f,
        -1.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    require_rejected("invalid base vertex cost", [&, invalid_cost] {
      ri::RoutingCsrSidecars broken = valid;
      broken.base_vertex_cost[0] = invalid_cost;
      ri::validate_routing_csr_sidecars(broken, 5, 7);
    });
  }
}

void test_spatial_permutation_validation() {
  const ri::RoutingCsrSidecars valid = make_sidecars();

  require_rejected("missing required spatial index", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges = {};
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("one zero grid dimension", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.width = 0;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("wrong spatial offset count", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.offsets.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("nonzero first spatial offset", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.offsets.front() = 1;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("nonmonotone spatial offsets", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.offsets[4] = 2;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("spatial terminal offset mismatch", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.offsets.back() = 6;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("spatial edge-count mismatch", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.edge_ids.pop_back();
    broken.spatial_edges.offsets.back() = 6;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("out-of-range spatial edge ID", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.edge_ids[0] = 7;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("duplicated spatial edge ID", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.edge_ids[0] = broken.spatial_edges.edge_ids[1];
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
  require_rejected("overflowing spatial grid", [&] {
    ri::RoutingCsrSidecars broken = valid;
    broken.spatial_edges.width =
        std::numeric_limits<std::uint64_t>::max();
    broken.spatial_edges.height = 2;
    ri::validate_routing_csr_sidecars(broken, 5, 7);
  });
}

void test_destination_shard_builder() {
  const std::vector<std::int32_t> x =
      {10, 11, 10, 12, ri::kMissingRouteCoordinate};
  const std::vector<std::int32_t> y =
      {20, 20, 21, 22, ri::kMissingRouteCoordinate};
  // These are destinations in the already filtered CSR. In particular, edge
  // IDs zero through six are compact post-filter IDs, not static-device IDs.
  const std::vector<std::int32_t> destinations = {0, 1, 4, 2, 3, 4, 1};
  const ri::SpatialEdgeShards shards =
      ri::build_destination_spatial_edge_shards(x, y, destinations);
  require(shards.min_x == 10 && shards.min_y == 20 && shards.width == 3 &&
              shards.height == 3,
          "destination shard builder produced the wrong dense grid");
  require(shards.offsets ==
              std::vector<std::uint64_t>({0, 1, 3, 3, 4, 4, 4, 4, 4, 5, 7}),
          "destination shard builder lost empty-cell boundaries");
  require(shards.edge_ids ==
              std::vector<std::uint32_t>({0, 1, 6, 3, 4, 2, 5}),
          "destination shard builder did not retain compact edge IDs in stable order");

  ri::RoutingCsrSidecars built = make_sidecars();
  built.spatial_edges = shards;
  ri::validate_routing_csr_sidecars(built, x.size(), destinations.size());
  ri::validate_destination_spatial_edge_shards(built, x.size(), destinations);
  const std::uint64_t spill = shards.spill_shard();
  require(spill == 9 && shards.offsets[spill] == 5 &&
              shards.offsets[spill + 1] == 7 &&
              shards.edge_ids[5] == 2 && shards.edge_ids[6] == 5,
          "missing-coordinate destinations did not enter the spill shard");

  require_rejected("mismatched shard-builder coordinate columns", [&] {
    (void)ri::build_destination_spatial_edge_shards(
        std::vector<std::int32_t>{0}, std::vector<std::int32_t>{}, {});
  });
  require_rejected("negative shard-builder destination", [&] {
    (void)ri::build_destination_spatial_edge_shards(x, y, {-1});
  });
  require_rejected("out-of-range shard-builder destination", [&] {
    (void)ri::build_destination_spatial_edge_shards(x, y, {5});
  });
  require_rejected("implausibly sparse shard-builder grid", [&] {
    (void)ri::build_destination_spatial_edge_shards(
        std::vector<std::int32_t>{0, 2'000'000},
        std::vector<std::int32_t>{0, 0},
        std::vector<std::int32_t>{0, 1});
  });
  require_rejected("edge stored in the wrong destination shard", [&] {
    ri::RoutingCsrSidecars broken = built;
    std::swap(broken.spatial_edges.edge_ids[0],
              broken.spatial_edges.edge_ids[1]);
    // The edge IDs remain a valid permutation; only semantic shard placement
    // is corrupt.
    ri::validate_destination_spatial_edge_shards(broken, x.size(),
                                                 destinations);
  });
}

}  // namespace

int main() {
  try {
    test_valid_sidecars_and_selection();
    test_node_sidecar_validation();
    test_spatial_permutation_validation();
    test_destination_shard_builder();
    std::cout << "BF11 routing-sidecar policy tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 routing-sidecar policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
