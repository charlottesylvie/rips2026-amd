#include "../interchange/device_routing_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ri = routing::interchange;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool reachable(const ri::CsrGraph& graph, std::int32_t source,
               std::int32_t target) {
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(graph.rows), 0);
  std::vector<std::int32_t> queue = {source};
  seen[static_cast<std::size_t>(source)] = 1;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::int32_t node = queue[head];
    if (node == target) {
      return true;
    }
    for (std::int64_t edge = graph.rowptr[static_cast<std::size_t>(node)];
         edge < graph.rowptr[static_cast<std::size_t>(node + 1)]; ++edge) {
      const std::int32_t next = graph.colind[static_cast<std::size_t>(edge)];
      if (!seen[static_cast<std::size_t>(next)]) {
        seen[static_cast<std::size_t>(next)] = 1;
        queue.push_back(next);
      }
    }
  }
  return false;
}

ri::CsrGraph reference_filter_with_separate_destination_masks(
    const ri::DeviceRoutingGraph& graph,
    const std::vector<std::uint8_t>& blocked,
    const std::vector<std::uint8_t>& sink_stops,
    const std::vector<std::uint8_t>& exclusive_sources) {
  ri::CsrGraph filtered;
  const std::size_t node_count =
      ri::device_routing_graph_node_count(graph);
  filtered.rows = static_cast<std::int64_t>(node_count);
  filtered.cols = filtered.rows;
  filtered.declared_edges = graph.declared_edges;
  filtered.loaded_edges = graph.loaded_edges;
  filtered.rowptr.resize(node_count + 1, 0);
  for (std::size_t row = 0; row < node_count; ++row) {
    if (!blocked[row] && !sink_stops[row]) {
      for (std::int64_t edge = graph.rowptr[row];
           edge < graph.rowptr[row + 1]; ++edge) {
        const std::size_t index = static_cast<std::size_t>(edge);
        const std::size_t destination =
            static_cast<std::size_t>(graph.colind[index]);
        if (!blocked[destination] && !exclusive_sources[destination]) {
          filtered.colind.push_back(graph.colind[index]);
          filtered.edge_attrs.push_back(graph.edge_attrs[index]);
        }
      }
    }
    filtered.rowptr[row + 1] =
        static_cast<std::int64_t>(filtered.colind.size());
  }
  filtered.values.assign(filtered.colind.size(), 1.0f);
  return filtered;
}

bool same_csr(const ri::CsrGraph& lhs, const ri::CsrGraph& rhs) {
  if (lhs.rows != rhs.rows || lhs.cols != rhs.cols ||
      lhs.declared_edges != rhs.declared_edges ||
      lhs.loaded_edges != rhs.loaded_edges || lhs.rowptr != rhs.rowptr ||
      lhs.colind != rhs.colind || lhs.values != rhs.values ||
      lhs.edge_attrs.size() != rhs.edge_attrs.size()) {
    return false;
  }
  for (std::size_t edge = 0; edge < lhs.edge_attrs.size(); ++edge) {
    if (lhs.edge_attrs[edge].tile_string !=
            rhs.edge_attrs[edge].tile_string ||
        lhs.edge_attrs[edge].pip_data_index !=
            rhs.edge_attrs[edge].pip_data_index) {
      return false;
    }
  }
  return true;
}

ri::DeviceRoutingGraph make_graph() {
  ri::DeviceRoutingGraph graph;
  graph.device_fingerprint = 0x123456789abcdef0ULL;
  graph.device_path_string = graph.string_table.intern("fixture.device");
  graph.device_name_string = graph.string_table.intern("xcvu3p");
  const std::uint64_t tile0 = graph.string_table.intern("TILE_X0Y0");
  const std::uint64_t tile1 = graph.string_table.intern("TILE_X1Y0");
  const std::uint64_t wire0 = graph.string_table.intern("WIRE0");
  const std::uint64_t wire1 = graph.string_table.intern("WIRE1");
  const std::uint64_t site0 = graph.string_table.intern("SITE0");
  const std::uint64_t site_type0 = graph.string_table.intern("TYPE0");
  const std::uint64_t pin0 = graph.string_table.intern("PIN0");
  const std::uint64_t tile_type = graph.string_table.intern("TILE_TYPE");
  const std::uint64_t wire_type = graph.string_table.intern("WIRE_TYPE");

  graph.bounds = {0, 10, 0, 10};
  graph.node_bounds_mode = ri::NodeBoundsMode::kPocBaseWire;
  graph.declared_edges = 6;
  graph.loaded_edges = 5;
  graph.node_device_ids = {10, 20, 30, 40};
  graph.node_min_x = {0, 0, 1, 1};
  graph.node_max_x = graph.node_min_x;
  graph.node_min_y = {0, 0, 0, 0};
  graph.node_max_y = graph.node_min_y;
  graph.node_tile_type_strings.assign(4, tile_type);
  graph.node_wire_type_strings.assign(4, wire_type);
  graph.node_route_end_x = {0, 4, 1, ri::kMissingRouteCoordinate};
  graph.node_route_end_y = {0, 6, 2, ri::kMissingRouteCoordinate};
  graph.node_base_vertex_cost = {1.0f, 1.25f, 2.5f, 4.0f};

  graph.pip_data = {
      {wire0, wire1, true},
      {wire0, wire1, false},
  };
  graph.rowptr = {0, 2, 4, 5, 5};
  graph.colind = {1, 2, 2, 3, 3};
  graph.edge_attrs = {
      {tile0, 0},
      {tile0, 1},
      {tile0, 0},
      {tile1, 0},
      {tile1, 1},
  };
  graph.tile_wire_nodes = {
      {ri::checked_lookup_string_id(tile0),
       ri::checked_lookup_string_id(wire0), 0, 0},
      {ri::checked_lookup_string_id(tile1),
       ri::checked_lookup_string_id(wire1), 3, 0},
  };
  graph.site_pin_nodes = {
      {ri::checked_lookup_string_id(site0),
       ri::checked_lookup_string_id(site_type0),
       ri::checked_lookup_string_id(pin0), 3},
  };
  std::sort(graph.tile_wire_nodes.begin(), graph.tile_wire_nodes.end());
  std::sort(graph.site_pin_nodes.begin(), graph.site_pin_nodes.end());
  return graph;
}

void compare_graphs(const ri::DeviceRoutingGraph& expected,
                    const ri::DeviceRoutingGraph& actual,
                    bool expect_physical_node_arrays = true,
                    bool expect_routing_sidecars = true) {
  require(actual.device_fingerprint == expected.device_fingerprint,
          "fingerprint changed across roundtrip");
  require(actual.device_path_string == expected.device_path_string,
          "device path changed across roundtrip");
  require(actual.device_name_string == expected.device_name_string,
          "device name changed across roundtrip");
  require(actual.node_bounds_mode == expected.node_bounds_mode,
          "node-bounds mode changed across roundtrip");
  require(actual.bounds.min_x == expected.bounds.min_x &&
              actual.bounds.max_x == expected.bounds.max_x &&
              actual.bounds.min_y == expected.bounds.min_y &&
              actual.bounds.max_y == expected.bounds.max_y,
          "bounds changed across roundtrip");
  require(actual.string_table.strings == expected.string_table.strings,
          "string table changed across roundtrip");
  require(ri::device_routing_graph_node_count(actual) ==
              expected.node_device_ids.size(),
          "retained node count changed across roundtrip");
  if (expect_physical_node_arrays) {
    require(actual.node_device_ids == expected.node_device_ids,
            "node IDs changed across roundtrip");
    require(actual.node_min_x == expected.node_min_x &&
                actual.node_max_x == expected.node_max_x &&
                actual.node_min_y == expected.node_min_y &&
                actual.node_max_y == expected.node_max_y,
            "node coordinates changed across roundtrip");
    require(actual.node_tile_type_strings ==
                    expected.node_tile_type_strings &&
                actual.node_wire_type_strings ==
                    expected.node_wire_type_strings,
            "node type strings changed across roundtrip");
  } else {
    require(actual.node_device_ids.empty() && actual.node_min_x.empty() &&
                actual.node_max_x.empty() && actual.node_min_y.empty() &&
                actual.node_max_y.empty() &&
                actual.node_tile_type_strings.empty() &&
                actual.node_wire_type_strings.empty(),
            "filtering projection retained physical node arrays");
  }
  if (expect_routing_sidecars) {
    require(actual.node_route_end_x == expected.node_route_end_x &&
                actual.node_route_end_y == expected.node_route_end_y &&
                actual.node_base_vertex_cost ==
                    expected.node_base_vertex_cost,
            "BF11 node sidecars changed across roundtrip");
  } else {
    require(actual.node_route_end_x.empty() &&
                actual.node_route_end_y.empty() &&
                actual.node_base_vertex_cost.empty(),
            "filtering projection retained BF11 node sidecars");
  }
  require(actual.declared_edges == expected.declared_edges &&
              actual.loaded_edges == expected.loaded_edges,
          "diagnostic edge counts changed across roundtrip");
  require(actual.rowptr == expected.rowptr,
          "row pointers changed across roundtrip");
  require(actual.colind == expected.colind,
          "columns changed across roundtrip");
  require(actual.edge_attrs.size() == expected.edge_attrs.size(),
          "edge attribute count changed across roundtrip");
  for (std::size_t edge = 0; edge < expected.edge_attrs.size(); ++edge) {
    require(actual.edge_attrs[edge].tile_string ==
                    expected.edge_attrs[edge].tile_string &&
                actual.edge_attrs[edge].pip_data_index ==
                    expected.edge_attrs[edge].pip_data_index,
            "edge attributes changed across roundtrip");
  }
  require(actual.pip_data.size() == expected.pip_data.size(),
          "PIP data count changed across roundtrip");
  for (std::size_t index = 0; index < expected.pip_data.size(); ++index) {
    require(actual.pip_data[index].wire0_string ==
                    expected.pip_data[index].wire0_string &&
                actual.pip_data[index].wire1_string ==
                    expected.pip_data[index].wire1_string &&
                actual.pip_data[index].forward ==
                    expected.pip_data[index].forward,
            "PIP data changed across roundtrip");
  }
  const auto compare_pair_lookups = [](
                                        const std::vector<ri::PairNodeLookup>& lhs,
                                        const std::vector<ri::PairNodeLookup>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      if (lhs[index].first_string != rhs[index].first_string ||
          lhs[index].second_string != rhs[index].second_string ||
          lhs[index].node != rhs[index].node ||
          lhs[index].reserved != rhs[index].reserved) {
        return false;
      }
    }
    return true;
  };
  const auto compare_site_pin_lookups = [](
      const std::vector<ri::SitePinNodeLookup>& lhs,
      const std::vector<ri::SitePinNodeLookup>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      if (lhs[index].site_string != rhs[index].site_string ||
          lhs[index].site_type_string != rhs[index].site_type_string ||
          lhs[index].pin_string != rhs[index].pin_string ||
          lhs[index].node != rhs[index].node) {
        return false;
      }
    }
    return true;
  };
  require(compare_pair_lookups(actual.tile_wire_nodes,
                               expected.tile_wire_nodes),
          "tile-wire lookup changed across roundtrip");
  require(compare_site_pin_lookups(actual.site_pin_nodes,
                                   expected.site_pin_nodes),
          "site-pin lookup changed across roundtrip");
}

void write_legacy_v3_fixture(const std::filesystem::path& v4_path,
                             const std::filesystem::path& v3_path,
                             const ri::DeviceRoutingGraph& graph) {
  std::ifstream input(v4_path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  require(static_cast<bool>(input) || input.eof(),
          "could not read the version-4 device-graph fixture");

  constexpr std::size_t kMagicBytes = 8;
  constexpr std::size_t kFixedHeaderBytes =
      kMagicBytes + 17 * sizeof(std::uint64_t);
  require(bytes.size() >= kFixedHeaderBytes,
          "version-4 fixture is shorter than its fixed header");
  const std::uint64_t legacy_version = 3;
  std::memcpy(bytes.data() + kMagicBytes,
              &legacy_version,
              sizeof(legacy_version));

  std::size_t node_arrays_begin = kFixedHeaderBytes;
  for (const std::string& text : graph.string_table.strings) {
    node_arrays_begin += sizeof(std::uint64_t) + text.size();
  }
  const std::size_t node_count = graph.node_device_ids.size();
  constexpr std::size_t kLegacyNodeBytes =
      3 * sizeof(std::uint64_t) + 4 * sizeof(std::int32_t);
  constexpr std::size_t kRoutingSidecarBytes =
      2 * sizeof(std::int32_t) + sizeof(float);
  const std::size_t sidecars_begin =
      node_arrays_begin + node_count * kLegacyNodeBytes;
  const std::size_t sidecars_end =
      sidecars_begin + node_count * kRoutingSidecarBytes;
  require(sidecars_end <= bytes.size(),
          "version-4 fixture does not contain complete BF11 sidecars");
  bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(sidecars_begin),
              bytes.begin() + static_cast<std::ptrdiff_t>(sidecars_end));

  std::ofstream output(v3_path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(output),
          "could not write the legacy version-3 fixture");
}

ri::DeviceRoutingGraph legacy_v3_expectation(
    const ri::DeviceRoutingGraph& current) {
  ri::DeviceRoutingGraph expected = current;
  const auto midpoint = [](std::int32_t minimum, std::int32_t maximum) {
    if (minimum == ri::kMissingRouteCoordinate &&
        maximum == ri::kMissingRouteCoordinate) {
      return ri::kMissingRouteCoordinate;
    }
    return static_cast<std::int32_t>(
        static_cast<std::int64_t>(minimum) +
        (static_cast<std::int64_t>(maximum) - minimum) / 2);
  };
  for (std::size_t node = 0; node < expected.node_device_ids.size(); ++node) {
    expected.node_route_end_x[node] =
        midpoint(expected.node_min_x[node], expected.node_max_x[node]);
    expected.node_route_end_y[node] =
        midpoint(expected.node_min_y[node], expected.node_max_y[node]);
  }
  expected.node_base_vertex_cost.assign(expected.node_device_ids.size(), 1.0f);
  return expected;
}

}  // namespace

int main() {
  std::vector<std::filesystem::path> cleanup;
  try {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() /
        ("rips-device-graph-test-" + std::to_string(nonce));
    const std::filesystem::path split_path = base.string() + ".split";
    const std::filesystem::path streamed_path = base.string() + ".streamed";
    const std::filesystem::path version_three_path =
        base.string() + ".version-three";
    const std::filesystem::path legacy_path = base.string() + ".legacy";
    const std::filesystem::path trailing_path = base.string() + ".trailing";
    const std::filesystem::path projected_truncated_path =
        base.string() + ".projected-truncated";
    const std::filesystem::path sidecar_truncated_path =
        base.string() + ".sidecar-truncated";
    cleanup = {split_path,
               streamed_path,
               version_three_path,
               legacy_path,
               trailing_path,
               projected_truncated_path,
               sidecar_truncated_path};

    const std::filesystem::path staged_one =
        ri::create_unique_staging_path(base.string() + ".output");
    const std::filesystem::path staged_two =
        ri::create_unique_staging_path(base.string() + ".output");
    cleanup.push_back(staged_one);
    cleanup.push_back(staged_two);
    require(staged_one != staged_two &&
                std::filesystem::is_regular_file(staged_one) &&
                std::filesystem::is_regular_file(staged_two),
            "staging paths were not created exclusively");

    std::vector<std::int64_t> raw_rowptr = {0, 4, 5};
    std::vector<ri::StaticCsrEntry> raw_entries = {
        {2, 0, {10, 10}},
        {1, 1, {11, 11}},
        {2, 2, {12, 12}},
        {1, 3, {13, 13}},
        {0, 0, {14, 14}},
    };
    ri::sort_and_deduplicate_static_csr(raw_rowptr, raw_entries);
    require(raw_rowptr == std::vector<std::int64_t>({0, 2, 3}),
            "row-local deduplication produced wrong row pointers");
    require(raw_entries.size() == 3 && raw_entries[0].col == 1 &&
                raw_entries[0].attr.tile_string == 13 &&
                raw_entries[1].col == 2 &&
                raw_entries[1].attr.tile_string == 12 &&
                raw_entries[2].col == 0,
            "row-local deduplication did not keep latest sorted edges");

    std::vector<std::int64_t> empty_rowptr = {0, 0, 4, 4, 5, 5};
    std::vector<ri::StaticCsrEntry> empty_row_entries = {
        {2, 0, {10, 10}},
        {1, 1, {11, 11}},
        {2, 2, {12, 12}},
        {1, 3, {13, 13}},
        {0, 0, {14, 14}},
    };
    ri::sort_and_deduplicate_static_csr(empty_rowptr, empty_row_entries);
    require(empty_rowptr ==
                std::vector<std::int64_t>({0, 0, 2, 2, 3, 3}) &&
                empty_row_entries.size() == 3 &&
                empty_row_entries[0].attr.tile_string == 13 &&
                empty_row_entries[1].attr.tile_string == 12 &&
                empty_row_entries[2].attr.tile_string == 14,
            "in-place row compaction lost boundaries around empty rows");

    std::vector<ri::PairNodeLookup> identical_aliases = {
        {2, 3, 7, 0}, {1, 4, 8, 0}, {2, 3, 7, 0}};
    require(ri::sort_and_deduplicate_pair_node_lookups(
                identical_aliases, ri::LookupConflictPolicy::kReject,
                "test") == 0 &&
                identical_aliases.size() == 2 &&
                identical_aliases[0].first_string == 1,
            "identical lookup aliases did not deduplicate deterministically");

    std::vector<ri::PairNodeLookup> conflicting_lookup = {
        {2, 3, 7, 0}, {2, 3, 9, 0}};
    bool rejected_lookup_conflict = false;
    try {
      (void)ri::sort_and_deduplicate_pair_node_lookups(
          conflicting_lookup, ri::LookupConflictPolicy::kReject, "test");
    } catch (const std::runtime_error&) {
      rejected_lookup_conflict = true;
    }
    require(rejected_lookup_conflict,
            "conflicting lookup nodes were silently selected");

    conflicting_lookup = {{2, 3, 7, 0}, {2, 3, 9, 0}, {4, 5, 6, 0}};
    require(ri::sort_and_deduplicate_pair_node_lookups(
                conflicting_lookup,
                ri::LookupConflictPolicy::kDropAmbiguous, "test") == 1 &&
                conflicting_lookup.size() == 1 &&
                conflicting_lookup[0].node == 6,
            "ambiguous untyped alias was not omitted");

    std::vector<ri::SitePinNodeLookup> typed_aliases = {
        {1, 10, 3, 7},
        {1, 11, 3, 9},
        {1, 10, 3, 7},
    };
    ri::sort_and_deduplicate_site_pin_lookups(typed_aliases);
    require(typed_aliases.size() == 2 && typed_aliases[0].node == 7 &&
                typed_aliases[1].node == 9,
            "typed site-pin aliases were dropped or misordered");
    bool rejected_typed_conflict = false;
    try {
      typed_aliases = {{1, 10, 3, 7}, {1, 10, 3, 9}};
      ri::sort_and_deduplicate_site_pin_lookups(typed_aliases);
    } catch (const std::runtime_error&) {
      rejected_typed_conflict = true;
    }
    require(rejected_typed_conflict,
            "conflicting typed site-pin key was silently selected");

    const ri::DeviceRoutingGraph expected = make_graph();

    bool rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.node_tile_type_strings[0] =
          static_cast<std::uint64_t>(invalid.string_table.strings.size());
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted an out-of-range node type string");

    rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.rowptr[2] = 1;
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted non-monotone row pointers");

    rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.string_table.strings[invalid.device_name_string].clear();
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted an empty cached device name");

    rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.node_route_end_x.pop_back();
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted a short route-end coordinate column");

    rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.node_route_end_x[0] = ri::kMissingRouteCoordinate;
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted a half-missing route-end coordinate");

    rejected_invalid_graph = false;
    try {
      ri::DeviceRoutingGraph invalid = make_graph();
      invalid.node_base_vertex_cost[0] = 0.0f;
      ri::validate_device_routing_graph(invalid);
    } catch (const std::runtime_error&) {
      rejected_invalid_graph = true;
    }
    require(rejected_invalid_graph,
            "validation accepted a nonpositive base vertex cost");

    ri::write_device_routing_graph(expected, split_path);
    const ri::DeviceRoutingGraph split =
        ri::read_device_routing_graph(split_path);
    compare_graphs(expected, split);
    const ri::DeviceRoutingGraph deferred =
        ri::read_device_routing_graph_for_filtering(split_path);
    compare_graphs(expected, deferred, false, false);
    const ri::DeviceRoutingGraph routing_projection =
        ri::read_device_routing_graph_for_routing(split_path);
    compare_graphs(expected, routing_projection, false, true);

    write_legacy_v3_fixture(split_path, version_three_path, expected);
    const ri::DeviceRoutingGraph expected_v3 =
        legacy_v3_expectation(expected);
    const ri::DeviceRoutingGraph version_three =
        ri::read_device_routing_graph(version_three_path);
    compare_graphs(expected_v3, version_three);
    const ri::DeviceRoutingGraph version_three_filtering =
        ri::read_device_routing_graph_for_filtering(version_three_path);
    compare_graphs(expected_v3, version_three_filtering, false, false);
    const ri::DeviceRoutingGraph version_three_routing =
        ri::read_device_routing_graph_for_routing(version_three_path);
    compare_graphs(expected_v3, version_three_routing, false, true);

    // Truncate one byte before the end of the skipped 40-byte/node block.
    // The projection must check the available file extent instead of letting
    // a relative seek silently move beyond EOF.
    std::filesystem::copy_file(
        split_path, projected_truncated_path,
        std::filesystem::copy_options::overwrite_existing);
    constexpr std::uintmax_t kDeviceGraphFixedHeaderBytes =
        8 + 17 * sizeof(std::uint64_t);
    std::uintmax_t node_arrays_begin = kDeviceGraphFixedHeaderBytes;
    for (const std::string& text : expected.string_table.strings) {
      node_arrays_begin += sizeof(std::uint64_t) + text.size();
    }
    const std::uintmax_t node_array_bytes =
        expected.node_device_ids.size() * 40;
    require(node_array_bytes != 0,
            "projection truncation fixture has no node metadata");
    std::filesystem::resize_file(
        projected_truncated_path,
        node_arrays_begin + node_array_bytes - 1);
    bool rejected_projected_truncation = false;
    try {
      (void)ri::read_device_routing_graph_for_filtering(
          projected_truncated_path);
    } catch (const std::runtime_error&) {
      rejected_projected_truncation = true;
    }
    require(rejected_projected_truncation,
            "filtering projection accepted a truncated node block");

    std::filesystem::copy_file(
        split_path, sidecar_truncated_path,
        std::filesystem::copy_options::overwrite_existing);
    constexpr std::uintmax_t kLegacyNodeBytes =
        3 * sizeof(std::uint64_t) + 4 * sizeof(std::int32_t);
    constexpr std::uintmax_t kRoutingSidecarBytes =
        2 * sizeof(std::int32_t) + sizeof(float);
    std::filesystem::resize_file(
        sidecar_truncated_path,
        node_arrays_begin +
            expected.node_device_ids.size() *
                (kLegacyNodeBytes + kRoutingSidecarBytes) -
            1);
    bool rejected_sidecar_truncation = false;
    try {
      (void)ri::read_device_routing_graph_for_routing(
          sidecar_truncated_path);
    } catch (const std::runtime_error&) {
      rejected_sidecar_truncation = true;
    }
    require(rejected_sidecar_truncation,
            "routing projection accepted a truncated BF11 sidecar block");

    std::filesystem::copy_file(
        split_path, legacy_path,
        std::filesystem::copy_options::overwrite_existing);
    {
      std::fstream legacy(legacy_path,
                          std::ios::in | std::ios::out | std::ios::binary);
      require(static_cast<bool>(legacy),
              "could not open legacy-version fixture");
      legacy.seekp(8);
      const std::uint64_t old_version = 2;
      legacy.write(reinterpret_cast<const char*>(&old_version),
                   sizeof(old_version));
      legacy.close();
      require(static_cast<bool>(legacy),
              "could not write legacy-version fixture");
    }
    bool rejected_legacy_version = false;
    try {
      (void)ri::read_device_routing_graph(legacy_path);
    } catch (const std::runtime_error&) {
      rejected_legacy_version = true;
    }
    require(rejected_legacy_version,
            "device graph accepted a version-2 cache");

    std::filesystem::copy_file(
        split_path, trailing_path,
        std::filesystem::copy_options::overwrite_existing);
    {
      std::ofstream trailing(trailing_path, std::ios::binary | std::ios::app);
      trailing.put('\0');
    }
    bool rejected_trailing_data = false;
    try {
      (void)ri::read_device_routing_graph(trailing_path);
    } catch (const std::runtime_error&) {
      rejected_trailing_data = true;
    }
    require(rejected_trailing_data,
            "device graph accepted trailing bytes");
    rejected_trailing_data = false;
    try {
      (void)ri::read_device_routing_graph_for_filtering(trailing_path);
    } catch (const std::runtime_error&) {
      rejected_trailing_data = true;
    }
    require(rejected_trailing_data,
            "device graph filtering projection accepted trailing bytes");

    const auto site = ri::find_site_pin_node(
        split.site_pin_nodes, split.string_table, "SITE0",
        std::optional<std::string>("TYPE0"), "PIN0");
    require(site.has_value() && *site == 3, "site-pin lookup failed");
    require(!ri::find_site_pin_node(
                 split.site_pin_nodes, split.string_table, "SITE0",
                 std::optional<std::string>("TYPE0"), "MISSING")
                 .has_value(),
            "missing site pin unexpectedly resolved");

    ri::DeviceRoutingGraph typed_graph = make_graph();
    const std::uint32_t site_id = typed_graph.site_pin_nodes[0].site_string;
    const std::uint32_t pin_id = typed_graph.site_pin_nodes[0].pin_string;
    const std::uint32_t type1 = ri::checked_lookup_string_id(
        typed_graph.string_table.intern("TYPE1"));
    typed_graph.site_pin_nodes.push_back({site_id, type1, pin_id, 2});
    ri::sort_and_deduplicate_site_pin_lookups(
        typed_graph.site_pin_nodes);
    require(ri::find_site_pin_node(
                typed_graph.site_pin_nodes, typed_graph.string_table,
                "SITE0", std::optional<std::string>("TYPE0"), "PIN0") ==
                std::optional<ri::NodeId>(3) &&
                ri::find_site_pin_node(
                    typed_graph.site_pin_nodes, typed_graph.string_table,
                    "SITE0", std::optional<std::string>("TYPE1"),
                    "PIN0") == std::optional<ri::NodeId>(2),
            "active site type did not disambiguate a site pin");
    require(!ri::find_site_pin_node(
                 typed_graph.site_pin_nodes, typed_graph.string_table,
                 "SITE0", std::nullopt, "PIN0")
                 .has_value(),
            "ambiguous untyped site pin was guessed");
    const std::vector<ri::NodeId> candidates =
        ri::find_site_pin_candidates(typed_graph.site_pin_nodes,
                                     typed_graph.string_table, "SITE0",
                                     "PIN0");
    require(candidates == std::vector<ri::NodeId>({2, 3}),
            "fixed-resource fallback lost possible site-pin nodes");

    ri::DeviceRoutingGraph agreeing_graph = make_graph();
    const std::uint32_t agreeing_type = ri::checked_lookup_string_id(
        agreeing_graph.string_table.intern("TYPE1"));
    agreeing_graph.site_pin_nodes.push_back(
        {agreeing_graph.site_pin_nodes[0].site_string, agreeing_type,
         agreeing_graph.site_pin_nodes[0].pin_string, 3});
    ri::sort_and_deduplicate_site_pin_lookups(
        agreeing_graph.site_pin_nodes);
    require(ri::find_site_pin_node(
                agreeing_graph.site_pin_nodes, agreeing_graph.string_table,
                "SITE0", std::nullopt, "PIN0") ==
                std::optional<ri::NodeId>(3),
            "agreeing typed aliases did not permit an untyped fallback");
    (void)agreeing_graph.string_table.intern("TYPE_UNUSED");
    require(!ri::find_site_pin_node(
                 agreeing_graph.site_pin_nodes,
                 agreeing_graph.string_table, "SITE0",
                 std::optional<std::string>("TYPE_UNUSED"), "PIN0")
                 .has_value(),
            "explicit active-type miss fell back to an inactive alias");

    std::vector<std::uint8_t> blocked = {0, 0, 1, 0};
    std::vector<std::uint8_t> sink_stop = {0, 1, 0, 0};
    // Destination unavailability is the union of blocked resources and
    // exclusive route sources. Rows remain governed by blocked/sink_stop.
    std::vector<std::uint8_t> unavailable_destinations = blocked;
    const ri::CsrGraph filtered =
        ri::filter_device_routing_graph(deferred, blocked, sink_stop,
                                        unavailable_destinations);
    require(filtered.rowptr == std::vector<std::int64_t>({0, 1, 1, 1, 1}),
            "filtered row pointers are wrong");
    require(filtered.colind == std::vector<std::int32_t>({1}),
            "filtered destinations are wrong");
    require(filtered.values == std::vector<float>({1.0f}),
            "filtered weights are wrong");
    require(filtered.edge_attrs.size() == 1 &&
                filtered.edge_attrs[0].tile_string ==
                    expected.edge_attrs[0].tile_string &&
                filtered.edge_attrs[0].pip_data_index ==
                    expected.edge_attrs[0].pip_data_index,
            "filtering broke CSR-edge/PIP alignment");
    require(filtered.declared_edges == expected.declared_edges &&
                filtered.loaded_edges == expected.loaded_edges,
            "filtering changed diagnostic base edge counts");

    require(reachable(filtered, 0, 1),
            "terminal sink lost its incoming route");
    require(!reachable(filtered, 0, 3),
            "a route traversed another net's terminal sink");
    // This conservative shared-CSR rule also rejects a same-net path that
    // would need 0 -> sink(1) -> 3 through a PINBOUNCE node. RWRoute can make
    // that ownership-aware exception; this representation cannot do so yet.

    unavailable_destinations[1] = 1;
    // When one node is both a source and sink of the same net, importer policy
    // deliberately leaves its row live while the exclusive-source mask still
    // removes incoming edges from every other row.
    const ri::CsrGraph source_sink_overlap =
        ri::filter_device_routing_graph(
            deferred, std::vector<std::uint8_t>({0, 0, 0, 0}),
            std::vector<std::uint8_t>({0, 0, 0, 0}),
            unavailable_destinations);
    require(!reachable(source_sink_overlap, 0, 1),
            "another net entered an exclusive source node");
    require(reachable(source_sink_overlap, 1, 3),
            "source/sink overlap lost its outgoing routing row");

    bool rejected_invalid_edge = false;
    try {
      ri::DeviceRoutingGraph invalid = deferred;
      invalid.colind[0] = 99;
      (void)ri::filter_device_routing_graph(
          invalid, blocked, sink_stop,
          unavailable_destinations);
    } catch (const std::runtime_error&) {
      rejected_invalid_edge = true;
    }
    require(rejected_invalid_edge,
            "deferred edge validation accepted an invalid destination");

    bool rejected_uncombined_destination_mask = false;
    try {
      (void)ri::filter_device_routing_graph(
          deferred, blocked, sink_stop,
          std::vector<std::uint8_t>({0, 0, 0, 0}));
    } catch (const std::runtime_error&) {
      rejected_uncombined_destination_mask = true;
    }
    require(rejected_uncombined_destination_mask,
            "filter accepted a destination mask missing a blocked node");

    // Exhaust every combination of the three four-node masks. This compares
    // the precombined production representation against the original policy:
    // blocked nodes lose incoming and outgoing edges, sink-stop nodes lose
    // only outgoing edges, and exclusive sources lose only incoming edges.
    const auto four_node_mask = [](unsigned bits) {
      std::vector<std::uint8_t> result(4, 0);
      for (unsigned node = 0; node < 4; ++node) {
        result[node] = static_cast<std::uint8_t>((bits >> node) & 1U);
      }
      return result;
    };
    for (unsigned blocked_bits = 0; blocked_bits < 16; ++blocked_bits) {
      for (unsigned sink_bits = 0; sink_bits < 16; ++sink_bits) {
        for (unsigned exclusive_bits = 0; exclusive_bits < 16;
             ++exclusive_bits) {
          const std::vector<std::uint8_t> case_blocked =
              four_node_mask(blocked_bits);
          const std::vector<std::uint8_t> case_sink_stops =
              four_node_mask(sink_bits);
          const std::vector<std::uint8_t> case_exclusive =
              four_node_mask(exclusive_bits);
          std::vector<std::uint8_t> case_unavailable = case_exclusive;
          for (std::size_t node = 0; node < case_unavailable.size(); ++node) {
            case_unavailable[node] |= case_blocked[node];
          }
          const ri::CsrGraph reference =
              reference_filter_with_separate_destination_masks(
                  deferred, case_blocked, case_sink_stops, case_exclusive);
          const ri::CsrGraph production = ri::filter_device_routing_graph(
              deferred, case_blocked, case_sink_stops, case_unavailable);
          require(same_csr(reference, production),
                  "precombined destination mask changed filter semantics");
        }
      }
    }

    ri::DeviceRoutingGraph streamed_source = make_graph();
    std::vector<ri::StaticCsrEntry> entries;
    entries.reserve(streamed_source.colind.size());
    for (std::size_t row = 0; row < streamed_source.node_device_ids.size();
         ++row) {
      for (std::int64_t edge = streamed_source.rowptr[row];
           edge < streamed_source.rowptr[row + 1]; ++edge) {
        const std::size_t index = static_cast<std::size_t>(edge);
        entries.push_back({streamed_source.colind[index],
                           static_cast<std::uint32_t>(
                               edge - streamed_source.rowptr[row]),
                           streamed_source.edge_attrs[index]});
      }
    }
    streamed_source.colind.clear();
    streamed_source.edge_attrs.clear();
    ri::write_device_routing_graph(streamed_source, entries, streamed_path);
    const ri::DeviceRoutingGraph streamed =
        ri::read_device_routing_graph(streamed_path);
    compare_graphs(expected, streamed);

    for (const std::filesystem::path& path : cleanup) {
      std::filesystem::remove(path);
    }
    std::cout << "device routing graph roundtrip/filter test passed\n";
    return 0;
  } catch (const std::exception& error) {
    for (const std::filesystem::path& path : cleanup) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
    std::cerr << "device routing graph test failed: " << error.what() << '\n';
    return 1;
  }
}
