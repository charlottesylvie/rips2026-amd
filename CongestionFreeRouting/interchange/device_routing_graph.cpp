#include "device_routing_graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace routing::interchange {
namespace {

constexpr char DEVICE_GRAPH_MAGIC[8] = {'R', 'I', 'P', 'S', 'D', 'R', 'G', '1'};
constexpr std::uint64_t DEVICE_GRAPH_VERSION = 1;

static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");
static_assert(sizeof(EdgeAttr) == 16, "EdgeAttr disk layout changed");
static_assert(std::is_trivially_copyable<EdgeAttr>::value,
              "EdgeAttr must support bulk I/O");
static_assert(sizeof(PairNodeLookup) == 16,
              "PairNodeLookup disk layout changed");
static_assert(std::is_trivially_copyable<PairNodeLookup>::value,
              "PairNodeLookup must support bulk I/O");

struct PipDataDisk {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
};

static_assert(sizeof(PipDataDisk) == 24, "PipDataDisk layout changed");

std::size_t checked_size(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds host size_t");
  }
  return static_cast<std::size_t>(count);
}

void write_u64(std::ofstream& out, std::uint64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

void write_i64(std::ofstream& out, std::int64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

std::int64_t read_i64(std::ifstream& in, const char* name) {
  std::int64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

template <typename T>
void write_array(std::ofstream& out,
                 const std::vector<T>& values,
                 const char* name) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary arrays must be trivially copyable");
  if (values.empty()) {
    return;
  }
  const std::size_t byte_count = values.size() * sizeof(T);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(byte_count));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary arrays must be trivially copyable");
  values.resize(checked_size(count, name));
  if (values.empty()) {
    return;
  }
  const std::size_t byte_count = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(byte_count));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

void write_string(std::ofstream& out, const std::string& text) {
  write_u64(out, static_cast<std::uint64_t>(text.size()), "string length");
  if (!text.empty()) {
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  if (!out) {
    throw std::runtime_error("failed while writing device-graph string");
  }
}

std::string read_string(std::ifstream& in) {
  const std::uint64_t byte_count = read_u64(in, "string length");
  std::string text(checked_size(byte_count, "string length"), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
  }
  if (!in) {
    throw std::runtime_error("failed while reading device-graph string");
  }
  return text;
}

void validate_node_arrays(const DeviceRoutingGraph& graph) {
  const std::size_t node_count = graph.node_device_ids.size();
  if (node_count == 0 ||
      graph.node_min_x.size() != node_count ||
      graph.node_max_x.size() != node_count ||
      graph.node_min_y.size() != node_count ||
      graph.node_max_y.size() != node_count ||
      graph.node_tile_type_strings.size() != node_count ||
      graph.node_wire_type_strings.size() != node_count) {
    throw std::runtime_error(
        "device-graph node metadata arrays do not match node count");
  }
}

void validate_string_table_index(const StringTable& table) {
  if (table.ids.size() != table.strings.size()) {
    throw std::runtime_error(
        "device-graph string table index is incomplete or contains "
        "duplicate strings");
  }
  for (std::size_t index = 0; index < table.strings.size(); ++index) {
    const auto found = table.ids.find(table.strings[index]);
    if (found == table.ids.end() || found->second != index) {
      throw std::runtime_error(
          "device-graph string table index is inconsistent");
    }
  }
}

bool is_valid_node_bounds_mode(NodeBoundsMode mode) {
  switch (mode) {
    case NodeBoundsMode::kPocBaseWire:
    case NodeBoundsMode::kFullyContained:
    case NodeBoundsMode::kIntersects:
      return true;
  }
  return false;
}

bool is_valid_optional_string(std::uint64_t id, std::size_t string_count) {
  return id == kNoStringIndex || id < string_count;
}

void validate_lookup_records(const std::vector<PairNodeLookup>& records,
                             std::size_t string_count,
                             std::size_t node_count,
                             const char* name) {
  if (!std::is_sorted(records.begin(), records.end())) {
    throw std::runtime_error(std::string(name) + " lookup is not sorted");
  }
  for (std::size_t index = 0; index < records.size(); ++index) {
    const PairNodeLookup& record = records[index];
    if (record.first_string >= string_count ||
        record.second_string >= string_count || record.node < 0 ||
        static_cast<std::size_t>(record.node) >= node_count ||
        record.reserved != 0) {
      throw std::runtime_error(std::string(name) +
                               " lookup contains an invalid record");
    }
    if (index > 0 &&
        record.first_string == records[index - 1].first_string &&
        record.second_string == records[index - 1].second_string) {
      throw std::runtime_error(std::string(name) +
                               " lookup contains a duplicate key");
    }
  }
}

void validate_static_metadata(const DeviceRoutingGraph& graph) {
  validate_node_arrays(graph);
  const std::size_t node_count = graph.node_device_ids.size();
  const std::size_t string_count = graph.string_table.strings.size();
  if (node_count > static_cast<std::size_t>(
                       std::numeric_limits<NodeId>::max())) {
    throw std::runtime_error("device graph has too many nodes for int32 IDs");
  }
  if (graph.device_fingerprint == 0) {
    throw std::runtime_error("device graph has an empty device fingerprint");
  }
  if (!is_valid_node_bounds_mode(graph.node_bounds_mode)) {
    throw std::runtime_error("device graph has an invalid node-bounds mode");
  }
  if (graph.bounds.min_x > graph.bounds.max_x ||
      graph.bounds.min_y > graph.bounds.max_y) {
    throw std::runtime_error("device graph has invalid coordinate bounds");
  }
  if (graph.device_path_string >= string_count) {
    throw std::runtime_error("device path string is out of range");
  }
  if (graph.loaded_edges > graph.declared_edges) {
    throw std::runtime_error(
        "device graph has more loaded than declared edges");
  }
  for (std::size_t node = 0; node < node_count; ++node) {
    if (graph.node_min_x[node] > graph.node_max_x[node] ||
        graph.node_min_y[node] > graph.node_max_y[node]) {
      throw std::runtime_error(
          "device graph contains an invalid node coordinate range");
    }
    if (!is_valid_optional_string(graph.node_tile_type_strings[node],
                                  string_count) ||
        !is_valid_optional_string(graph.node_wire_type_strings[node],
                                  string_count)) {
      throw std::runtime_error(
          "device graph contains an invalid node type string");
    }
  }
  for (const PipData& pip : graph.pip_data) {
    if (pip.wire0_string >= string_count ||
        pip.wire1_string >= string_count) {
      throw std::runtime_error("device graph contains invalid PIP strings");
    }
  }
  validate_lookup_records(graph.tile_wire_nodes, string_count, node_count,
                          "tile-wire");
  validate_lookup_records(graph.site_pin_nodes, string_count, node_count,
                          "site-pin");
}

std::size_t validate_row_pointers(const DeviceRoutingGraph& graph) {
  const std::size_t expected_count = graph.node_device_ids.size() + 1;
  if (graph.rowptr.size() != expected_count || graph.rowptr.front() != 0) {
    throw std::runtime_error("device graph has invalid CSR row pointers");
  }
  std::int64_t previous = 0;
  for (const std::int64_t offset : graph.rowptr) {
    if (offset < previous) {
      throw std::runtime_error("device graph row pointers are not monotone");
    }
    previous = offset;
  }
  return checked_size(static_cast<std::uint64_t>(graph.rowptr.back()),
                      "base CSR edge count");
}

std::size_t validate_csr_shape(const DeviceRoutingGraph& graph) {
  validate_static_metadata(graph);
  const std::size_t edge_count = validate_row_pointers(graph);
  if (graph.loaded_edges != edge_count || graph.colind.size() != edge_count ||
      graph.edge_attrs.size() != edge_count) {
    throw std::runtime_error("device graph edge counts are inconsistent");
  }
  return edge_count;
}

void write_header_and_static_prefix(std::ofstream& out,
                                    const DeviceRoutingGraph& graph,
                                    std::uint64_t edge_count) {
  out.write(DEVICE_GRAPH_MAGIC, sizeof(DEVICE_GRAPH_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing device-graph magic");
  }

  write_u64(out, DEVICE_GRAPH_VERSION, "device-graph version");
  write_u64(out, graph.device_fingerprint, "device fingerprint");
  write_u64(out, static_cast<std::uint64_t>(graph.node_bounds_mode),
            "node bounds mode");
  write_i64(out, graph.bounds.min_x, "minimum X bound");
  write_i64(out, graph.bounds.max_x, "maximum X bound");
  write_i64(out, graph.bounds.min_y, "minimum Y bound");
  write_i64(out, graph.bounds.max_y, "maximum Y bound");
  write_u64(out, static_cast<std::uint64_t>(graph.string_table.strings.size()),
            "string count");
  write_u64(out, static_cast<std::uint64_t>(graph.node_device_ids.size()),
            "node count");
  write_u64(out, edge_count, "edge count");
  write_u64(out, static_cast<std::uint64_t>(graph.pip_data.size()),
            "pip data count");
  write_u64(out, static_cast<std::uint64_t>(graph.tile_wire_nodes.size()),
            "tile-wire lookup count");
  write_u64(out, static_cast<std::uint64_t>(graph.site_pin_nodes.size()),
            "site-pin lookup count");
  write_u64(out, graph.declared_edges, "declared edge count");
  write_u64(out, graph.loaded_edges, "loaded edge count");
  write_u64(out, graph.device_path_string, "device path string");

  for (const std::string& text : graph.string_table.strings) {
    write_string(out, text);
  }

  write_array(out, graph.node_device_ids, "device node IDs");
  write_array(out, graph.node_min_x, "node minimum X coordinates");
  write_array(out, graph.node_max_x, "node maximum X coordinates");
  write_array(out, graph.node_min_y, "node minimum Y coordinates");
  write_array(out, graph.node_max_y, "node maximum Y coordinates");
  write_array(out, graph.node_tile_type_strings, "node tile type strings");
  write_array(out, graph.node_wire_type_strings, "node wire type strings");
  write_array(out, graph.rowptr, "base CSR row pointers");
}

void write_static_suffix(std::ofstream& out,
                         const DeviceRoutingGraph& graph) {
  std::vector<PipDataDisk> pip_disk;
  pip_disk.reserve(graph.pip_data.size());
  for (const PipData& pip : graph.pip_data) {
    pip_disk.push_back(
        {pip.wire0_string, pip.wire1_string, pip.forward ? 1ULL : 0ULL});
  }
  write_array(out, pip_disk, "PIP data");
  write_array(out, graph.tile_wire_nodes, "tile-wire lookup");
  write_array(out, graph.site_pin_nodes, "site-pin lookup");
}

void ensure_parent_directory(const std::filesystem::path& path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

}  // namespace

const char* node_bounds_mode_name(NodeBoundsMode mode) {
  switch (mode) {
    case NodeBoundsMode::kPocBaseWire:
      return "poc-base-wire";
    case NodeBoundsMode::kFullyContained:
      return "fully-contained";
    case NodeBoundsMode::kIntersects:
      return "intersects";
  }
  return "unknown";
}

NodeBoundsMode parse_node_bounds_mode(const std::string& text) {
  if (text == "poc-base-wire" || text == "poc") {
    return NodeBoundsMode::kPocBaseWire;
  }
  if (text == "fully-contained" || text == "contained") {
    return NodeBoundsMode::kFullyContained;
  }
  if (text == "intersects" || text == "any") {
    return NodeBoundsMode::kIntersects;
  }
  throw std::runtime_error("unknown node bounds mode: " + text);
}

std::uint64_t StringTable::intern(const std::string& text) {
  const auto found = ids.find(text);
  if (found != ids.end()) {
    return found->second;
  }
  const std::uint64_t id = static_cast<std::uint64_t>(strings.size());
  strings.push_back(text);
  ids.emplace(strings.back(), id);
  return id;
}

std::optional<std::uint64_t> StringTable::find(
    const std::string& text) const {
  const auto found = ids.find(text);
  if (found == ids.end()) {
    return std::nullopt;
  }
  return found->second;
}

void StringTable::rebuild_index() {
  ids.clear();
  ids.reserve(strings.size());
  for (std::size_t index = 0; index < strings.size(); ++index) {
    if (!ids.emplace(strings[index], static_cast<std::uint64_t>(index)).second) {
      throw std::runtime_error("device-graph string table contains duplicates");
    }
  }
}

bool operator<(const PairNodeLookup& lhs, const PairNodeLookup& rhs) {
  if (lhs.first_string != rhs.first_string) {
    return lhs.first_string < rhs.first_string;
  }
  if (lhs.second_string != rhs.second_string) {
    return lhs.second_string < rhs.second_string;
  }
  return lhs.node < rhs.node;
}

std::uint32_t checked_lookup_string_id(std::uint64_t id) {
  if (id >= static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error(
        "device-graph string table is too large for compact lookups");
  }
  return static_cast<std::uint32_t>(id);
}

std::optional<NodeId> find_pair_node(
    const std::vector<PairNodeLookup>& records,
    const StringTable& strings,
    const std::string& first,
    const std::string& second) {
  const std::optional<std::uint64_t> first_id = strings.find(first);
  const std::optional<std::uint64_t> second_id = strings.find(second);
  if (!first_id.has_value() || !second_id.has_value() ||
      *first_id > std::numeric_limits<std::uint32_t>::max() ||
      *second_id > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }

  PairNodeLookup key;
  key.first_string = static_cast<std::uint32_t>(*first_id);
  key.second_string = static_cast<std::uint32_t>(*second_id);
  key.node = std::numeric_limits<NodeId>::min();
  const auto found = std::lower_bound(records.begin(), records.end(), key);
  if (found == records.end() || found->first_string != key.first_string ||
      found->second_string != key.second_string) {
    return std::nullopt;
  }
  return found->node;
}

void validate_device_routing_graph(const DeviceRoutingGraph& graph) {
  const std::size_t node_count = graph.node_device_ids.size();
  const std::size_t edge_count = validate_csr_shape(graph);
  for (std::size_t row = 0; row < node_count; ++row) {
    const std::int64_t begin = graph.rowptr[row];
    const std::int64_t end = graph.rowptr[row + 1];
    if (begin < 0 || end < begin ||
        static_cast<std::size_t>(end) > edge_count) {
      throw std::runtime_error("device graph row pointers are not monotone");
    }
    std::int32_t previous = -1;
    for (std::int64_t edge = begin; edge < end; ++edge) {
      const std::int32_t col = graph.colind[static_cast<std::size_t>(edge)];
      if (col < 0 || static_cast<std::size_t>(col) >= node_count ||
          col <= previous) {
        throw std::runtime_error(
            "device graph row destinations are invalid or not unique/sorted");
      }
      previous = col;
      const EdgeAttr& attr = graph.edge_attrs[static_cast<std::size_t>(edge)];
      if (attr.tile_string >= graph.string_table.strings.size() ||
          attr.pip_data_index >= graph.pip_data.size()) {
        throw std::runtime_error("device graph contains an invalid edge attr");
      }
    }
  }
}

namespace {

DeviceRoutingGraph read_device_routing_graph_impl(
    const std::filesystem::path& path,
    bool validate_edge_records) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open device-routing graph: " +
                             path.string());
  }

  char magic[sizeof(DEVICE_GRAPH_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, DEVICE_GRAPH_MAGIC,
                         sizeof(DEVICE_GRAPH_MAGIC)) != 0) {
    throw std::runtime_error(
        "input is not a recognized RIPS device-routing graph");
  }
  if (read_u64(in, "device-graph version") != DEVICE_GRAPH_VERSION) {
    throw std::runtime_error("unsupported device-routing graph version");
  }

  DeviceRoutingGraph graph;
  graph.device_fingerprint = read_u64(in, "device fingerprint");
  const std::uint64_t raw_mode = read_u64(in, "node bounds mode");
  if (raw_mode > static_cast<std::uint64_t>(NodeBoundsMode::kIntersects)) {
    throw std::runtime_error("invalid cached node bounds mode");
  }
  graph.node_bounds_mode = static_cast<NodeBoundsMode>(raw_mode);
  const auto read_bound = [&](const char* name) {
    const std::int64_t value = read_i64(in, name);
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error(std::string(name) + " is outside int32 range");
    }
    return static_cast<std::int32_t>(value);
  };
  graph.bounds.min_x = read_bound("minimum X bound");
  graph.bounds.max_x = read_bound("maximum X bound");
  graph.bounds.min_y = read_bound("minimum Y bound");
  graph.bounds.max_y = read_bound("maximum Y bound");

  const std::uint64_t string_count = read_u64(in, "string count");
  const std::uint64_t node_count = read_u64(in, "node count");
  const std::uint64_t edge_count = read_u64(in, "edge count");
  const std::uint64_t pip_count = read_u64(in, "PIP data count");
  const std::uint64_t tile_wire_count =
      read_u64(in, "tile-wire lookup count");
  const std::uint64_t site_pin_count = read_u64(in, "site-pin lookup count");
  graph.declared_edges = read_u64(in, "declared edge count");
  graph.loaded_edges = read_u64(in, "loaded edge count");
  graph.device_path_string = read_u64(in, "device path string");

  if (node_count == 0 ||
      node_count > static_cast<std::uint64_t>(
                       std::numeric_limits<NodeId>::max()) ||
      edge_count > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int64_t>::max()) ||
      graph.loaded_edges != edge_count) {
    throw std::runtime_error("device-routing graph header counts are invalid");
  }

  graph.string_table.strings.reserve(checked_size(string_count, "strings"));
  for (std::uint64_t index = 0; index < string_count; ++index) {
    graph.string_table.strings.push_back(read_string(in));
  }
  graph.string_table.rebuild_index();

  read_array(in, graph.node_device_ids, node_count, "device node IDs");
  read_array(in, graph.node_min_x, node_count, "node minimum X coordinates");
  read_array(in, graph.node_max_x, node_count, "node maximum X coordinates");
  read_array(in, graph.node_min_y, node_count, "node minimum Y coordinates");
  read_array(in, graph.node_max_y, node_count, "node maximum Y coordinates");
  read_array(in, graph.node_tile_type_strings, node_count,
             "node tile type strings");
  read_array(in, graph.node_wire_type_strings, node_count,
             "node wire type strings");
  read_array(in, graph.rowptr, node_count + 1, "base CSR row pointers");
  read_array(in, graph.colind, edge_count, "base CSR destinations");
  read_array(in, graph.edge_attrs, edge_count, "base CSR edge attributes");

  std::vector<PipDataDisk> pip_disk;
  read_array(in, pip_disk, pip_count, "PIP data");
  graph.pip_data.reserve(pip_disk.size());
  for (const PipDataDisk& pip : pip_disk) {
    if (pip.forward > 1) {
      throw std::runtime_error("device graph contains an invalid PIP flag");
    }
    graph.pip_data.push_back(
        {pip.wire0_string, pip.wire1_string, pip.forward != 0});
  }
  read_array(in, graph.tile_wire_nodes, tile_wire_count, "tile-wire lookup");
  read_array(in, graph.site_pin_nodes, site_pin_count, "site-pin lookup");

  if (validate_edge_records) {
    validate_device_routing_graph(graph);
  } else {
    (void)validate_csr_shape(graph);
  }
  return graph;
}

}  // namespace

DeviceRoutingGraph read_device_routing_graph(
    const std::filesystem::path& path) {
  return read_device_routing_graph_impl(path, true);
}

DeviceRoutingGraph read_device_routing_graph_for_filtering(
    const std::filesystem::path& path) {
  return read_device_routing_graph_impl(path, false);
}

void write_device_routing_graph(const DeviceRoutingGraph& graph,
                                const std::filesystem::path& path) {
  validate_string_table_index(graph.string_table);
  validate_device_routing_graph(graph);
  ensure_parent_directory(path);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open device-routing output: " +
                             path.string());
  }
  write_header_and_static_prefix(out, graph, graph.loaded_edges);
  write_array(out, graph.colind, "base CSR destinations");
  write_array(out, graph.edge_attrs, "base CSR edge attributes");
  write_static_suffix(out, graph);
}

void write_device_routing_graph(
    const DeviceRoutingGraph& graph,
    const std::vector<StaticCsrEntry>& static_entries,
    const std::filesystem::path& path) {
  validate_string_table_index(graph.string_table);
  validate_static_metadata(graph);
  const std::size_t edge_count = validate_row_pointers(graph);
  if (!graph.colind.empty() || !graph.edge_attrs.empty() ||
      edge_count != static_entries.size() ||
      graph.loaded_edges != static_entries.size()) {
    throw std::runtime_error(
        "preprocessor graph and static CSR entries are inconsistent");
  }
  for (std::size_t row = 0; row < graph.node_device_ids.size(); ++row) {
    const std::size_t begin = static_cast<std::size_t>(graph.rowptr[row]);
    const std::size_t end = static_cast<std::size_t>(graph.rowptr[row + 1]);
    std::int32_t previous = -1;
    for (std::size_t edge = begin; edge < end; ++edge) {
      const StaticCsrEntry& entry = static_entries[edge];
      if (entry.col < 0 ||
          static_cast<std::size_t>(entry.col) >=
              graph.node_device_ids.size() ||
          entry.col <= previous ||
          entry.attr.tile_string >= graph.string_table.strings.size() ||
          entry.attr.pip_data_index >= graph.pip_data.size()) {
        throw std::runtime_error("preprocessor CSR entries are invalid");
      }
      previous = entry.col;
    }
  }

  ensure_parent_directory(path);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open device-routing output: " +
                             path.string());
  }
  write_header_and_static_prefix(out, graph, static_entries.size());

  constexpr std::size_t kChunkEntries = 1 << 20;
  std::vector<std::int32_t> columns;
  columns.reserve(kChunkEntries);
  for (std::size_t begin = 0; begin < static_entries.size();
       begin += kChunkEntries) {
    const std::size_t end =
        std::min(static_entries.size(), begin + kChunkEntries);
    columns.clear();
    for (std::size_t index = begin; index < end; ++index) {
      columns.push_back(static_entries[index].col);
    }
    write_array(out, columns, "base CSR destinations");
  }

  std::vector<EdgeAttr> attrs;
  attrs.reserve(kChunkEntries);
  for (std::size_t begin = 0; begin < static_entries.size();
       begin += kChunkEntries) {
    const std::size_t end =
        std::min(static_entries.size(), begin + kChunkEntries);
    attrs.clear();
    for (std::size_t index = begin; index < end; ++index) {
      attrs.push_back(static_entries[index].attr);
    }
    write_array(out, attrs, "base CSR edge attributes");
  }

  write_static_suffix(out, graph);
}

void sort_and_deduplicate_static_csr(
    std::vector<std::int64_t>& rowptr,
    std::vector<StaticCsrEntry>& entries) {
  if (rowptr.empty() || rowptr.front() != 0 || rowptr.back() < 0 ||
      static_cast<std::size_t>(rowptr.back()) != entries.size()) {
    throw std::runtime_error(
        "raw static CSR row pointers do not match its entries");
  }

  std::vector<std::int64_t> unique_rowptr(rowptr.size(), 0);
  std::size_t output_edge = 0;
  for (std::size_t row = 0; row + 1 < rowptr.size(); ++row) {
    if (rowptr[row] < 0 || rowptr[row + 1] < rowptr[row]) {
      throw std::runtime_error("raw static CSR row pointers are not monotone");
    }
    const std::size_t begin = static_cast<std::size_t>(rowptr[row]);
    const std::size_t end = static_cast<std::size_t>(rowptr[row + 1]);
    if (end - begin > 1) {
      std::sort(entries.begin() + static_cast<std::ptrdiff_t>(begin),
                entries.begin() + static_cast<std::ptrdiff_t>(end),
                [](const StaticCsrEntry& lhs, const StaticCsrEntry& rhs) {
                  return lhs.col != rhs.col ? lhs.col < rhs.col
                                            : lhs.ordinal < rhs.ordinal;
                });
    }
    for (std::size_t group_begin = begin; group_begin < end;) {
      std::size_t group_end = group_begin + 1;
      while (group_end < end &&
             entries[group_end].col == entries[group_begin].col) {
        ++group_end;
      }
      if (output_edge != group_end - 1) {
        entries[output_edge] = entries[group_end - 1];
      }
      ++output_edge;
      group_begin = group_end;
    }
    unique_rowptr[row + 1] = static_cast<std::int64_t>(output_edge);
  }
  entries.resize(output_edge);
  rowptr.swap(unique_rowptr);
}

CsrGraph filter_device_routing_graph(
    const DeviceRoutingGraph& graph,
    const std::vector<std::uint8_t>& blocked_node,
    const std::vector<std::uint8_t>& sink_node_stops) {
  const std::size_t node_count = graph.node_device_ids.size();
  if (blocked_node.size() != node_count ||
      sink_node_stops.size() != node_count) {
    throw std::runtime_error("design masks do not match device graph rows");
  }
  if (graph.rowptr.size() != node_count + 1 || graph.rowptr.front() != 0 ||
      graph.rowptr.back() < 0 ||
      static_cast<std::uint64_t>(graph.rowptr.back()) !=
          graph.loaded_edges ||
      graph.loaded_edges != graph.colind.size() ||
      graph.colind.size() != graph.edge_attrs.size()) {
    throw std::runtime_error("device graph CSR shape is inconsistent");
  }

  CsrGraph csr;
  csr.rows = static_cast<std::int64_t>(node_count);
  csr.cols = csr.rows;
  csr.declared_edges = graph.declared_edges;
  csr.loaded_edges = graph.loaded_edges;
  csr.rowptr.resize(node_count + 1, 0);
  csr.colind.reserve(graph.colind.size());
  csr.edge_attrs.reserve(graph.edge_attrs.size());

  // Validate and compact in one pass. Contest masks are sparse, so reserving
  // the base edge count avoids reallocations without value-initializing a
  // second full graph; retained records are appended directly in row order.
  // A heavily masked non-contest workload may prefer an exact counting pass
  // to reduce reserved virtual memory.
  for (std::size_t row = 0; row < node_count; ++row) {
    const std::int64_t begin = graph.rowptr[row];
    const std::int64_t end = graph.rowptr[row + 1];
    if (begin < 0 || end < begin ||
        static_cast<std::uint64_t>(end) > graph.loaded_edges) {
      throw std::runtime_error("device graph row pointers are not monotone");
    }
    std::int32_t previous = -1;
    const bool source_is_active =
        !blocked_node[row] && !sink_node_stops[row];
    for (std::int64_t edge = begin; edge < end; ++edge) {
      const std::size_t input_edge = static_cast<std::size_t>(edge);
      const std::int32_t col = graph.colind[input_edge];
      const EdgeAttr& attr = graph.edge_attrs[input_edge];
      if (col < 0 || static_cast<std::size_t>(col) >= node_count ||
          col <= previous ||
          attr.tile_string >= graph.string_table.strings.size() ||
          attr.pip_data_index >= graph.pip_data.size()) {
        throw std::runtime_error(
            "device graph edge records are invalid or not sorted");
      }
      previous = col;
      if (source_is_active &&
          !blocked_node[static_cast<std::size_t>(col)]) {
        csr.colind.push_back(col);
        csr.edge_attrs.push_back(attr);
      }
    }
    if (csr.colind.size() > static_cast<std::size_t>(
                                 std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("filtered CSR edge count overflows int64");
    }
    csr.rowptr[row + 1] = static_cast<std::int64_t>(csr.colind.size());
  }

  csr.values.assign(csr.colind.size(), 1.0f);
  return csr;
}

}  // namespace routing::interchange
