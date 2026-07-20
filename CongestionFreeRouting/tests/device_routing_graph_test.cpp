#include "../interchange/device_routing_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

ri::DeviceRoutingGraph make_graph() {
  ri::DeviceRoutingGraph graph;
  graph.device_fingerprint = 0x123456789abcdef0ULL;
  graph.device_path_string = graph.string_table.intern("fixture.device");
  const std::uint64_t tile0 = graph.string_table.intern("TILE_X0Y0");
  const std::uint64_t tile1 = graph.string_table.intern("TILE_X1Y0");
  const std::uint64_t wire0 = graph.string_table.intern("WIRE0");
  const std::uint64_t wire1 = graph.string_table.intern("WIRE1");
  const std::uint64_t site0 = graph.string_table.intern("SITE0");
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
       ri::checked_lookup_string_id(pin0), 3, 0},
  };
  std::sort(graph.tile_wire_nodes.begin(), graph.tile_wire_nodes.end());
  std::sort(graph.site_pin_nodes.begin(), graph.site_pin_nodes.end());
  return graph;
}

void compare_graphs(const ri::DeviceRoutingGraph& expected,
                    const ri::DeviceRoutingGraph& actual) {
  require(actual.device_fingerprint == expected.device_fingerprint,
          "fingerprint changed across roundtrip");
  require(actual.device_path_string == expected.device_path_string,
          "device path changed across roundtrip");
  require(actual.node_bounds_mode == expected.node_bounds_mode,
          "node-bounds mode changed across roundtrip");
  require(actual.bounds.min_x == expected.bounds.min_x &&
              actual.bounds.max_x == expected.bounds.max_x &&
              actual.bounds.min_y == expected.bounds.min_y &&
              actual.bounds.max_y == expected.bounds.max_y,
          "bounds changed across roundtrip");
  require(actual.string_table.strings == expected.string_table.strings,
          "string table changed across roundtrip");
  require(actual.node_device_ids == expected.node_device_ids,
          "node IDs changed across roundtrip");
  require(actual.node_min_x == expected.node_min_x &&
              actual.node_max_x == expected.node_max_x &&
              actual.node_min_y == expected.node_min_y &&
              actual.node_max_y == expected.node_max_y,
          "node coordinates changed across roundtrip");
  require(actual.node_tile_type_strings == expected.node_tile_type_strings &&
              actual.node_wire_type_strings ==
                  expected.node_wire_type_strings,
          "node type strings changed across roundtrip");
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
  const auto compare_lookups = [](const std::vector<ri::PairNodeLookup>& lhs,
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
  require(compare_lookups(actual.tile_wire_nodes,
                          expected.tile_wire_nodes),
          "tile-wire lookup changed across roundtrip");
  require(compare_lookups(actual.site_pin_nodes, expected.site_pin_nodes),
          "site-pin lookup changed across roundtrip");
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
    cleanup = {split_path, streamed_path};

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

    ri::write_device_routing_graph(expected, split_path);
    const ri::DeviceRoutingGraph split =
        ri::read_device_routing_graph(split_path);
    compare_graphs(expected, split);
    const ri::DeviceRoutingGraph deferred =
        ri::read_device_routing_graph_for_filtering(split_path);
    compare_graphs(expected, deferred);

    const auto site = ri::find_pair_node(
        split.site_pin_nodes, split.string_table, "SITE0", "PIN0");
    require(site.has_value() && *site == 3, "site-pin lookup failed");
    require(!ri::find_pair_node(split.site_pin_nodes, split.string_table,
                                "SITE0", "MISSING")
                 .has_value(),
            "missing site pin unexpectedly resolved");

    std::vector<std::uint8_t> blocked = {0, 0, 1, 0};
    std::vector<std::uint8_t> sink_stop = {0, 1, 0, 0};
    const ri::CsrGraph filtered =
        ri::filter_device_routing_graph(deferred, blocked, sink_stop);
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

    bool rejected_invalid_edge = false;
    try {
      ri::DeviceRoutingGraph invalid = deferred;
      invalid.colind[0] = 99;
      (void)ri::filter_device_routing_graph(invalid, blocked, sink_stop);
    } catch (const std::runtime_error&) {
      rejected_invalid_edge = true;
    }
    require(rejected_invalid_edge,
            "deferred edge validation accepted an invalid destination");

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
