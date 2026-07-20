// One-time FPGA Interchange DeviceResources preprocessor.
//
// Expensive fabric parsing is invariant across benchmarks that share the same
// device, tile bounds, and node-bounds policy. This executable converts that
// data into a versioned RIPSDRG1 artifact consumed by interchange_to_csr.
//
// Example build:
//   g++ -std=c++17 -O3 -I<generated-schema-dir> \
//     CongestionFreeRouting/device_to_routing_graph.cpp \
//     CongestionFreeRouting/interchange/device_routing_graph.cpp \
//     <generated-schema-dir>/DeviceResources.capnp.c++ \
//     <generated-schema-dir>/LogicalNetlist.capnp.c++ \
//     <generated-schema-dir>/References.capnp.c++ \
//     -lcapnp -lkj -lz -o device_to_routing_graph
//
// Example use:
//   ./device_to_routing_graph xcvu3p.device \
//     xcvu3p.full-poc-base-wire.devicegraph --full-device

#include "DeviceResources.capnp.h"
#include "interchange/device_routing_graph.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using routing::interchange::Bounds;
using routing::interchange::DeviceRoutingGraph;
using routing::interchange::EdgeAttr;
using routing::interchange::NodeBoundsMode;
using routing::interchange::NodeId;
using routing::interchange::PairNodeLookup;
using routing::interchange::PipData;
using routing::interchange::StaticCsrEntry;
using routing::interchange::checked_lookup_string_id;
using routing::interchange::kInvalidRouteNode;
using routing::interchange::kNoIndex;
using routing::interchange::kNoStringIndex;
using routing::interchange::node_bounds_mode_name;
using routing::interchange::parse_node_bounds_mode;
using routing::interchange::sort_and_deduplicate_static_csr;
using routing::interchange::write_device_routing_graph;

constexpr int NXROUTE_MIN_X = 36;
constexpr int NXROUTE_MAX_X = 90;
constexpr int NXROUTE_MIN_Y = 60;
constexpr int NXROUTE_MAX_Y = 239;

struct Options {
  std::filesystem::path device_path;
  std::filesystem::path output_path;
  Bounds bounds;
  NodeBoundsMode node_bounds_mode = NodeBoundsMode::kPocBaseWire;
};

int parse_int_arg(const char* text, const char* name) {
  try {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != std::strlen(text) ||
        value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("outside int32 range");
    }
    return static_cast<int>(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n  " << program
      << " <device.device> <output.devicegraph> [options]\n\n"
      << "Options:\n"
      << "  --bounds <minX> <maxX> <minY> <maxY>\n"
      << "                                 Tile-coordinate subset to import.\n"
      << "  --nxroute-bounds               Import X36..X90, Y60..Y239.\n"
      << "  --node-bounds-mode <mode>      poc-base-wire, fully-contained, "
         "or intersects.\n"
      << "  --full-device                  Import every tile with XY coords "
         "(default).\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<std::filesystem::path> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--bounds") {
      if (index + 4 >= argc) {
        throw std::runtime_error("--bounds requires four integers");
      }
      options.bounds.min_x = parse_int_arg(argv[++index], "minX");
      options.bounds.max_x = parse_int_arg(argv[++index], "maxX");
      options.bounds.min_y = parse_int_arg(argv[++index], "minY");
      options.bounds.max_y = parse_int_arg(argv[++index], "maxY");
      continue;
    }
    if (arg == "--nxroute-bounds") {
      options.bounds =
          {NXROUTE_MIN_X, NXROUTE_MAX_X, NXROUTE_MIN_Y, NXROUTE_MAX_Y};
      continue;
    }
    if (arg == "--node-bounds-mode") {
      if (index + 1 >= argc) {
        throw std::runtime_error("--node-bounds-mode requires a mode");
      }
      options.node_bounds_mode = parse_node_bounds_mode(argv[++index]);
      continue;
    }
    if (arg == "--full-device") {
      options.bounds = Bounds{};
      continue;
    }
    if (!arg.empty() && arg.front() == '-') {
      throw std::runtime_error("unknown option: " + arg);
    }
    positional.emplace_back(arg);
  }
  if (positional.size() != 2) {
    throw std::runtime_error(
        "expected <device.device> <output.devicegraph>");
  }
  options.device_path = positional[0];
  options.output_path = positional[1];
  if (options.bounds.min_x > options.bounds.max_x ||
      options.bounds.min_y > options.bounds.max_y) {
    throw std::runtime_error("invalid bounds: min is greater than max");
  }
  return options;
}

struct DevicePayload {
  std::vector<capnp::word> words;
  std::size_t decoded_bytes = 0;
  std::uint64_t fingerprint = 1469598103934665603ULL;
};

// Read directly into Cap'n Proto-aligned storage. The old converter first
// retained a byte vector and then copied the entire decompressed device into a
// word vector; avoiding that second multi-gigabyte resident copy matters on a
// full xcvu3p graph.
DevicePayload read_device_payload(const std::filesystem::path& path) {
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (!file) {
    throw std::runtime_error("could not open device file: " + path.string());
  }
  (void)gzbuffer(file, 1 << 20);

  DevicePayload payload;
  std::array<std::uint8_t, 1 << 20> buffer{};
  try {
    while (true) {
      const int read_count = gzread(
          file, buffer.data(), static_cast<unsigned int>(buffer.size()));
      if (read_count == 0) {
        break;
      }
      if (read_count < 0) {
        int zlib_error = 0;
        const char* message = gzerror(file, &zlib_error);
        throw std::runtime_error(
            "failed while reading " + path.string() + ": " +
            (message ? message : "zlib error"));
      }

      const std::size_t chunk_size = static_cast<std::size_t>(read_count);
      if (chunk_size > std::numeric_limits<std::size_t>::max() -
                           payload.decoded_bytes) {
        throw std::runtime_error("decoded device is too large for this host");
      }
      const std::size_t old_size = payload.decoded_bytes;
      payload.decoded_bytes += chunk_size;
      const std::size_t word_count =
          (payload.decoded_bytes + sizeof(capnp::word) - 1) /
          sizeof(capnp::word);
      payload.words.resize(word_count);
      std::memcpy(reinterpret_cast<std::uint8_t*>(payload.words.data()) +
                      old_size,
                  buffer.data(), chunk_size);

      // A word-at-a-time content fingerprint is sufficient for cache identity
      // and avoids adding a byte-serial hash pass over a multi-gigabyte device.
      std::size_t offset = 0;
      while (offset + sizeof(std::uint64_t) <= chunk_size) {
        std::uint64_t lane = 0;
        std::memcpy(&lane, buffer.data() + offset, sizeof(lane));
        payload.fingerprint ^=
            lane + 0x9e3779b97f4a7c15ULL + old_size + offset;
        payload.fingerprint *= 0xbf58476d1ce4e5b9ULL;
        payload.fingerprint = (payload.fingerprint << 27) |
                              (payload.fingerprint >> (64 - 27));
        offset += sizeof(lane);
      }
      std::uint64_t tail = static_cast<std::uint64_t>(chunk_size - offset)
                           << 56;
      std::memcpy(&tail, buffer.data() + offset, chunk_size - offset);
      payload.fingerprint ^= tail + payload.decoded_bytes;
      payload.fingerprint *= 0x94d049bb133111ebULL;
    }
  } catch (...) {
    gzclose(file);
    throw;
  }
  gzclose(file);
  if (payload.decoded_bytes == 0) {
    throw std::runtime_error("device file is empty: " + path.string());
  }
  return payload;
}

void check_device_payload_size(const std::filesystem::path& path,
                               std::size_t decoded_bytes) {
  constexpr std::size_t kMinExpectedDeviceBytes = 1 << 20;
  if (path.filename() == "xcvu3p.device" &&
      decoded_bytes < kMinExpectedDeviceBytes) {
    throw std::runtime_error(
        "device resources file is suspiciously small: " + path.string() +
        " has only " + std::to_string(decoded_bytes) +
        " decoded bytes; regenerate xcvu3p.device");
  }
}

class TextCache {
 public:
  explicit TextCache(capnp::List<capnp::Text>::Reader strings)
      : strings_(strings), cache_(strings.size()) {}

  const std::string& get(std::uint32_t index) {
    if (index >= cache_.size()) {
      throw std::runtime_error("FPGAIF string index is out of range");
    }
    std::optional<std::string>& cached = cache_[index];
    if (!cached.has_value()) {
      const capnp::Text::Reader text = strings_[index];
      cached.emplace(text.cStr(), text.size());
    }
    return *cached;
  }

  std::size_t size() const { return cache_.size(); }

 private:
  capnp::List<capnp::Text>::Reader strings_;
  std::vector<std::optional<std::string>> cache_;
};

std::optional<std::pair<std::int32_t, std::int32_t>> parse_tile_xy(
    const std::string& tile_name) {
  const std::size_t marker = tile_name.rfind("_X");
  if (marker == std::string::npos) {
    return std::nullopt;
  }
  std::size_t pos = marker + 2;
  if (pos >= tile_name.size() || tile_name[pos] < '0' ||
      tile_name[pos] > '9') {
    return std::nullopt;
  }
  std::int64_t x = 0;
  while (pos < tile_name.size() && tile_name[pos] >= '0' &&
         tile_name[pos] <= '9') {
    x = x * 10 + (tile_name[pos++] - '0');
    if (x > std::numeric_limits<std::int32_t>::max()) {
      return std::nullopt;
    }
  }
  if (pos >= tile_name.size() || tile_name[pos++] != 'Y' ||
      pos >= tile_name.size() || tile_name[pos] < '0' ||
      tile_name[pos] > '9') {
    return std::nullopt;
  }
  std::int64_t y = 0;
  while (pos < tile_name.size() && tile_name[pos] >= '0' &&
         tile_name[pos] <= '9') {
    y = y * 10 + (tile_name[pos++] - '0');
    if (y > std::numeric_limits<std::int32_t>::max()) {
      return std::nullopt;
    }
  }
  return std::make_pair(static_cast<std::int32_t>(x),
                        static_cast<std::int32_t>(y));
}

bool has_prefix(const std::string& text, const char* prefix) {
  const std::size_t size = std::strlen(prefix);
  return text.size() >= size && text.compare(0, size, prefix) == 0;
}

std::uint64_t pair_key(std::uint32_t first, std::uint32_t second) {
  return (static_cast<std::uint64_t>(first) << 32) | second;
}

std::uint64_t mix_key(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct NumericPairNode {
  std::uint32_t first = 0;
  std::uint32_t second = 0;
  NodeId node = kInvalidRouteNode;
};

// Open-addressed numeric lookup used only while preprocessing. It replaces
// nested unordered_map<string, unordered_map<string, NodeId>> and therefore
// avoids one allocation and multiple string hashes for every device wire.
class FlatPairNodeMap {
 public:
  explicit FlatPairNodeMap(std::size_t expected_size) {
    std::size_t capacity = 8;
    const std::size_t target = expected_size >
                                       std::numeric_limits<std::size_t>::max() /
                                           10
                                   ? expected_size
                                   : (expected_size * 10 + 6) / 7;
    while (capacity < target) {
      if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::runtime_error("tile-wire lookup is too large");
      }
      capacity *= 2;
    }
    keys_.resize(capacity);
    values_.assign(capacity, kInvalidRouteNode);
    mask_ = capacity - 1;
  }

  void insert(std::uint32_t first, std::uint32_t second, NodeId node) {
    const std::uint64_t key = pair_key(first, second);
    std::size_t slot = static_cast<std::size_t>(mix_key(key)) & mask_;
    while (values_[slot] != kInvalidRouteNode && keys_[slot] != key) {
      slot = (slot + 1) & mask_;
    }
    keys_[slot] = key;
    values_[slot] = node;
  }

  std::optional<NodeId> find(std::uint32_t first,
                             std::uint32_t second) const {
    const std::uint64_t key = pair_key(first, second);
    std::size_t slot = static_cast<std::size_t>(mix_key(key)) & mask_;
    while (values_[slot] != kInvalidRouteNode) {
      if (keys_[slot] == key) {
        return values_[slot];
      }
      slot = (slot + 1) & mask_;
    }
    return std::nullopt;
  }

 private:
  std::vector<std::uint64_t> keys_;
  std::vector<NodeId> values_;
  std::size_t mask_ = 0;
};

struct TileInfo {
  std::int32_t x = -1;
  std::int32_t y = -1;
  std::uint32_t tile_index = std::numeric_limits<std::uint32_t>::max();
  bool has_xy = false;
  bool in_bounds = false;
};

struct PipKey {
  std::uint32_t wire0 = 0;
  std::uint32_t wire1 = 0;
  bool forward = true;

  bool operator==(const PipKey& other) const {
    return wire0 == other.wire0 && wire1 == other.wire1 &&
           forward == other.forward;
  }
};

struct PipKeyHash {
  std::size_t operator()(const PipKey& key) const {
    return static_cast<std::size_t>(
        mix_key(pair_key(key.wire0, key.wire1) ^
                (key.forward ? 0xd6e8feb86659fd93ULL : 0)));
  }
};

struct PipTemplate {
  std::uint32_t wire0 = 0;
  std::uint32_t wire1 = 0;
  std::uint64_t forward_pip_data = kNoIndex;
  std::uint64_t reverse_pip_data = kNoIndex;
  bool bidirectional = false;
  bool conventional = true;
};

struct TileInstance {
  std::uint32_t tile_index = 0;
  std::uint32_t tile_name = 0;
  std::uint32_t tile_type = 0;
  std::uint64_t tile_string = 0;
  bool restrict_to_conventional = false;
};

struct SitePinTemplate {
  std::uint32_t pin_name = 0;
  std::uint32_t tile_wire = 0;
};

void sort_and_deduplicate_lookups(std::vector<PairNodeLookup>& records) {
  std::sort(records.begin(), records.end());
  std::size_t write = 0;
  for (std::size_t begin = 0; begin < records.size();) {
    std::size_t end = begin + 1;
    while (end < records.size() &&
           records[end].first_string == records[begin].first_string &&
           records[end].second_string == records[begin].second_string) {
      ++end;
    }
    records[write++] = records[end - 1];
    begin = end;
  }
  records.resize(write);
}

struct BuildResult {
  DeviceRoutingGraph graph;
  std::vector<StaticCsrEntry> entries;
};

template <typename Container>
void release_storage(Container& container) {
  Container empty;
  container.swap(empty);
}

BuildResult build_device_routing_graph(const Options& options) {
  DevicePayload payload = read_device_payload(options.device_path);
  check_device_payload_size(options.device_path, payload.decoded_bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.data(), payload.words.size()), reader_options);
  const auto device = reader.getRoot<DeviceResources::Device>();
  TextCache strings(device.getStrList());

  const auto tile_list = device.getTileList();
  const auto tile_types = device.getTileTypeList();
  const auto wires = device.getWires();
  const auto nodes = device.getNodes();
  const auto wire_types = device.getWireTypes();

  if (nodes.size() > static_cast<std::uint64_t>(
                         std::numeric_limits<NodeId>::max())) {
    throw std::runtime_error("device has too many nodes for int32 CSR IDs");
  }

  BuildResult result;
  DeviceRoutingGraph& graph = result.graph;
  graph.device_fingerprint = payload.fingerprint;
  graph.bounds = options.bounds;
  graph.node_bounds_mode = options.node_bounds_mode;
  graph.device_path_string =
      graph.string_table.intern(options.device_path.string());

  std::vector<std::uint64_t> local_string_by_device_id(strings.size(),
                                                        kNoStringIndex);
  auto intern_device_string = [&](std::uint32_t device_string) {
    if (device_string >= local_string_by_device_id.size()) {
      throw std::runtime_error("FPGAIF string index is out of range");
    }
    std::uint64_t& local = local_string_by_device_id[device_string];
    if (local == kNoStringIndex) {
      local = graph.string_table.intern(strings.get(device_string));
    }
    return local;
  };

  // Parse each tile coordinate once and retain a dense StringIdx-indexed table.
  // The old implementation reparsed a tile name for every wire in every node.
  std::vector<TileInfo> tile_info(strings.size());
  std::vector<std::uint32_t> in_bounds_tile_indices;
  in_bounds_tile_indices.reserve(tile_list.size());
  for (std::uint32_t tile_index = 0; tile_index < tile_list.size();
       ++tile_index) {
    const auto tile = tile_list[tile_index];
    if (tile.getName() >= tile_info.size()) {
      throw std::runtime_error("tile name string index is out of range");
    }
    TileInfo& info = tile_info[tile.getName()];
    info.tile_index = tile_index;
    const auto xy = parse_tile_xy(strings.get(tile.getName()));
    if (!xy.has_value()) {
      continue;
    }
    info.has_xy = true;
    info.x = xy->first;
    info.y = xy->second;
    info.in_bounds = info.x >= options.bounds.min_x &&
                     info.x <= options.bounds.max_x &&
                     info.y >= options.bounds.min_y &&
                     info.y <= options.bounds.max_y;
    if (info.in_bounds) {
      in_bounds_tile_indices.push_back(tile_index);
    }
  }

  const std::size_t node_capacity = static_cast<std::size_t>(nodes.size());
  graph.node_device_ids.reserve(node_capacity);
  graph.node_min_x.reserve(node_capacity);
  graph.node_max_x.reserve(node_capacity);
  graph.node_min_y.reserve(node_capacity);
  graph.node_max_y.reserve(node_capacity);
  graph.node_tile_type_strings.reserve(node_capacity);
  graph.node_wire_type_strings.reserve(node_capacity);

  std::vector<NumericPairNode> numeric_tile_wire_nodes;
  numeric_tile_wire_nodes.reserve(static_cast<std::size_t>(wires.size()));

  for (std::uint32_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const auto node_wires = nodes[node_index].getWires();
    if (node_wires.size() == 0) {
      continue;
    }
    const auto base_wire = wires[node_wires[0]];
    const bool base_in_bounds =
        base_wire.getTile() < tile_info.size() &&
        tile_info[base_wire.getTile()].in_bounds;

    bool include = base_in_bounds;
    std::uint32_t metadata_wire_index = node_wires[0];
    if (options.node_bounds_mode != NodeBoundsMode::kPocBaseWire) {
      bool any_in_bounds = false;
      bool all_in_bounds = true;
      for (std::uint32_t offset = 0; offset < node_wires.size(); ++offset) {
        const auto wire = wires[node_wires[offset]];
        const bool in_bounds = wire.getTile() < tile_info.size() &&
                               tile_info[wire.getTile()].in_bounds;
        if (in_bounds && !any_in_bounds) {
          metadata_wire_index = node_wires[offset];
        }
        any_in_bounds = any_in_bounds || in_bounds;
        all_in_bounds = all_in_bounds && in_bounds;
      }
      include = options.node_bounds_mode == NodeBoundsMode::kFullyContained
                    ? all_in_bounds
                    : any_in_bounds;
    }
    if (!include) {
      continue;
    }

    const NodeId compact_node =
        static_cast<NodeId>(graph.node_device_ids.size());
    graph.node_device_ids.push_back(node_index);

    std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
    std::int32_t max_x = std::numeric_limits<std::int32_t>::min();
    std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
    std::int32_t max_y = std::numeric_limits<std::int32_t>::min();
    bool has_xy = false;
    for (std::uint32_t offset = 0; offset < node_wires.size(); ++offset) {
      const auto wire = wires[node_wires[offset]];
      if (wire.getTile() < tile_info.size()) {
        const TileInfo& info = tile_info[wire.getTile()];
        if (info.has_xy) {
          has_xy = true;
          min_x = std::min(min_x, info.x);
          max_x = std::max(max_x, info.x);
          min_y = std::min(min_y, info.y);
          max_y = std::max(max_y, info.y);
        }
      }
      numeric_tile_wire_nodes.push_back(
          {checked_lookup_string_id(
               intern_device_string(wire.getTile())),
           checked_lookup_string_id(
               intern_device_string(wire.getWire())),
           compact_node});
    }
    if (!has_xy) {
      min_x = max_x = min_y = max_y = -1;
    }
    graph.node_min_x.push_back(min_x);
    graph.node_max_x.push_back(max_x);
    graph.node_min_y.push_back(min_y);
    graph.node_max_y.push_back(max_y);

    const auto metadata_wire = wires[metadata_wire_index];
    std::uint64_t tile_type_string = kNoStringIndex;
    if (metadata_wire.getTile() < tile_info.size()) {
      const std::uint32_t tile_index =
          tile_info[metadata_wire.getTile()].tile_index;
      if (tile_index != std::numeric_limits<std::uint32_t>::max()) {
        const std::uint32_t tile_type_index = tile_list[tile_index].getType();
        if (tile_type_index < tile_types.size()) {
          tile_type_string = intern_device_string(
              tile_types[tile_type_index].getName());
        }
      }
    }
    std::uint64_t wire_type_string = kNoStringIndex;
    if (metadata_wire.getType() < wire_types.size()) {
      wire_type_string =
          intern_device_string(wire_types[metadata_wire.getType()].getName());
    }
    graph.node_tile_type_strings.push_back(tile_type_string);
    graph.node_wire_type_strings.push_back(wire_type_string);
  }

  if (graph.node_device_ids.empty()) {
    throw std::runtime_error("no routing nodes were imported from device");
  }
  release_storage(tile_info);

  FlatPairNodeMap tile_wire_map(numeric_tile_wire_nodes.size());
  for (const NumericPairNode& record : numeric_tile_wire_nodes) {
    tile_wire_map.insert(record.first, record.second, record.node);
  }

  graph.tile_wire_nodes.reserve(numeric_tile_wire_nodes.size());
  for (const NumericPairNode& record : numeric_tile_wire_nodes) {
    graph.tile_wire_nodes.push_back(
        {record.first, record.second, record.node, 0});
  }
  sort_and_deduplicate_lookups(graph.tile_wire_nodes);
  release_storage(numeric_tile_wire_nodes);

  std::vector<TileInstance> tile_instances;
  tile_instances.reserve(in_bounds_tile_indices.size());
  std::vector<std::uint8_t> used_tile_type(tile_types.size(), 0);
  for (const std::uint32_t tile_index : in_bounds_tile_indices) {
    const auto tile = tile_list[tile_index];
    if (tile.getType() >= tile_types.size()) {
      throw std::runtime_error("tile type index is out of range");
    }
    const std::string& tile_name = strings.get(tile.getName());
    const std::uint64_t tile_string = intern_device_string(tile.getName());
    tile_instances.push_back(
        {tile_index, checked_lookup_string_id(tile_string), tile.getType(),
         tile_string,
         has_prefix(tile_name, "CLE") || has_prefix(tile_name, "RCLK")});
    used_tile_type[tile.getType()] = 1;
  }
  release_storage(in_bounds_tile_indices);

  // Expand PIP descriptors only once per tile type. Tile instances now reuse
  // numeric wire IDs and PIP-data IDs instead of rebuilding string keys.
  std::unordered_map<PipKey, std::uint64_t, PipKeyHash> pip_data_by_key;
  std::vector<std::vector<PipTemplate>> pip_templates(tile_types.size());
  auto intern_pip_data = [&](std::uint32_t wire0, std::uint32_t wire1,
                             bool forward) {
    const PipKey key{wire0, wire1, forward};
    const auto found = pip_data_by_key.find(key);
    if (found != pip_data_by_key.end()) {
      return found->second;
    }
    const std::uint64_t index = graph.pip_data.size();
    graph.pip_data.push_back({wire0, wire1, forward});
    pip_data_by_key.emplace(key, index);
    return index;
  };

  for (std::uint32_t type_index = 0; type_index < tile_types.size();
       ++type_index) {
    if (!used_tile_type[type_index]) {
      continue;
    }
    const auto tile_type = tile_types[type_index];
    const auto type_wires = tile_type.getWires();
    const auto pips = tile_type.getPips();
    auto& templates = pip_templates[type_index];
    templates.reserve(pips.size());
    for (std::uint32_t pip_index = 0; pip_index < pips.size(); ++pip_index) {
      const auto pip = pips[pip_index];
      if (pip.getWire0() >= type_wires.size() ||
          pip.getWire1() >= type_wires.size()) {
        throw std::runtime_error("tile-type PIP wire index is out of range");
      }
      PipTemplate descriptor;
      descriptor.wire0 = checked_lookup_string_id(
          intern_device_string(type_wires[pip.getWire0()]));
      descriptor.wire1 = checked_lookup_string_id(
          intern_device_string(type_wires[pip.getWire1()]));
      descriptor.conventional = pip.isConventional();
      descriptor.bidirectional = !pip.getDirectional();
      templates.push_back(descriptor);
    }
  }

  auto for_each_edge = [&](auto&& callback) {
    for (const TileInstance& tile : tile_instances) {
      for (PipTemplate& pip : pip_templates[tile.tile_type]) {
        if (tile.restrict_to_conventional && !pip.conventional) {
          continue;
        }
        const std::optional<NodeId> node0 =
            tile_wire_map.find(tile.tile_name, pip.wire0);
        const std::optional<NodeId> node1 =
            tile_wire_map.find(tile.tile_name, pip.wire1);
        if (!node0.has_value() || !node1.has_value()) {
          continue;
        }
        if (pip.forward_pip_data == kNoIndex) {
          pip.forward_pip_data =
              intern_pip_data(pip.wire0, pip.wire1, true);
        }
        callback(*node0, *node1, tile.tile_string,
                 pip.forward_pip_data);
        if (pip.bidirectional) {
          if (pip.reverse_pip_data == kNoIndex) {
            pip.reverse_pip_data =
                intern_pip_data(pip.wire0, pip.wire1, false);
          }
          callback(*node1, *node0, tile.tile_string,
                   pip.reverse_pip_data);
        }
      }
    }
  };

  std::vector<std::int64_t> row_counts(graph.node_device_ids.size(), 0);
  for_each_edge([&](NodeId from, NodeId to, std::uint64_t,
                    std::uint64_t) {
    ++graph.declared_edges;
    if (from == to) {
      return;
    }
    std::int64_t& count = row_counts[static_cast<std::size_t>(from)];
    if (count == std::numeric_limits<std::int64_t>::max()) {
      throw std::runtime_error("device graph row degree overflows int64");
    }
    ++count;
  });

  graph.rowptr.resize(graph.node_device_ids.size() + 1, 0);
  for (std::size_t row = 0; row < row_counts.size(); ++row) {
    if (row_counts[row] > std::numeric_limits<std::int64_t>::max() -
                               graph.rowptr[row]) {
      throw std::runtime_error("device graph edge count overflows int64");
    }
    graph.rowptr[row + 1] = graph.rowptr[row] + row_counts[row];
  }
  result.entries.resize(static_cast<std::size_t>(graph.rowptr.back()));
  std::vector<std::int64_t> cursor = graph.rowptr;
  for_each_edge([&](NodeId from, NodeId to, std::uint64_t tile_string,
                    std::uint64_t pip_data_index) {
    if (from == to) {
      return;
    }
    const std::size_t row = static_cast<std::size_t>(from);
    const std::int64_t position = cursor[row]++;
    const std::int64_t ordinal = position - graph.rowptr[row];
    if (ordinal > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("device graph row degree exceeds uint32");
    }
    result.entries[static_cast<std::size_t>(position)] =
        {to, static_cast<std::uint32_t>(ordinal),
         EdgeAttr{tile_string, pip_data_index}};
  });
  release_storage(cursor);
  release_storage(row_counts);

  sort_and_deduplicate_static_csr(graph.rowptr, result.entries);
  graph.loaded_edges = result.entries.size();
  release_storage(pip_data_by_key);
  release_storage(pip_templates);

  // Resolve concrete site/pin pairs once. The per-design converter performs a
  // single string-ID lookup plus binary search instead of rebuilding the three
  // nested device tables on every benchmark.
  const auto site_type_list = device.getSiteTypeList();
  std::vector<std::vector<std::uint32_t>> site_type_pin_names(
      site_type_list.size());
  for (std::uint32_t type_index = 0; type_index < site_type_list.size();
       ++type_index) {
    const auto pins = site_type_list[type_index].getPins();
    auto& names = site_type_pin_names[type_index];
    names.reserve(pins.size());
    for (std::uint32_t pin = 0; pin < pins.size(); ++pin) {
      names.push_back(checked_lookup_string_id(
          intern_device_string(pins[pin].getName())));
    }
  }

  std::vector<std::vector<std::vector<SitePinTemplate>>> site_pin_templates(
      tile_types.size());
  for (std::uint32_t tile_type_index = 0;
       tile_type_index < tile_types.size(); ++tile_type_index) {
    if (!used_tile_type[tile_type_index]) {
      continue;
    }
    const auto site_types = tile_types[tile_type_index].getSiteTypes();
    auto& by_site_type = site_pin_templates[tile_type_index];
    by_site_type.resize(site_types.size());
    for (std::uint32_t site_type_index = 0;
         site_type_index < site_types.size(); ++site_type_index) {
      const auto site_type = site_types[site_type_index];
      const std::uint32_t primary_type = site_type.getPrimaryType();
      if (primary_type >= site_type_pin_names.size()) {
        continue;
      }
      const auto wires_for_pins = site_type.getPrimaryPinsToTileWires();
      const auto& pin_names = site_type_pin_names[primary_type];
      const std::uint32_t count = std::min<std::uint32_t>(
          wires_for_pins.size(), pin_names.size());
      auto& templates = by_site_type[site_type_index];
      templates.reserve(count);
      for (std::uint32_t pin = 0; pin < count; ++pin) {
        templates.push_back(
            {pin_names[pin], checked_lookup_string_id(
                                 intern_device_string(wires_for_pins[pin]))});
      }
    }
  }

  for (const TileInstance& tile_instance : tile_instances) {
    const auto tile = tile_list[tile_instance.tile_index];
    const auto sites = tile.getSites();
    const auto& by_site_type = site_pin_templates[tile_instance.tile_type];
    for (std::uint32_t site_index = 0; site_index < sites.size(); ++site_index) {
      const auto site = sites[site_index];
      if (site.getType() >= by_site_type.size()) {
        continue;
      }
      for (const SitePinTemplate& pin : by_site_type[site.getType()]) {
        const std::optional<NodeId> node =
            tile_wire_map.find(tile_instance.tile_name, pin.tile_wire);
        if (!node.has_value()) {
          continue;
        }
        graph.site_pin_nodes.push_back(
            {checked_lookup_string_id(
                 intern_device_string(site.getName())),
             pin.pin_name,
             *node, 0});
      }
    }
  }
  sort_and_deduplicate_lookups(graph.site_pin_nodes);

  return result;
}

double mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void write_graph_atomically(const DeviceRoutingGraph& graph,
                            const std::vector<StaticCsrEntry>& entries,
                            const std::filesystem::path& output_path) {
  std::filesystem::path temporary = output_path;
  temporary += ".tmp";
  try {
    write_device_routing_graph(graph, entries, temporary);
    std::filesystem::rename(temporary, output_path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::cout << "device: " << options.device_path << '\n'
              << "bounds: X" << options.bounds.min_x << "..X"
              << options.bounds.max_x << ", Y" << options.bounds.min_y
              << "..Y" << options.bounds.max_y << '\n'
              << "node_bounds_mode: "
              << node_bounds_mode_name(options.node_bounds_mode) << '\n';

    BuildResult result = build_device_routing_graph(options);
    write_graph_atomically(result.graph, result.entries, options.output_path);

    const std::uint64_t node_count = result.graph.node_device_ids.size();
    const std::uint64_t edge_count = result.entries.size();
    const std::uint64_t compact_bytes =
        (node_count + 1) * sizeof(std::int64_t) +
        edge_count * (sizeof(std::int32_t) + sizeof(EdgeAttr));
    std::cout << "device_fingerprint: " << result.graph.device_fingerprint
              << '\n'
              << "imported_nodes: " << node_count << '\n'
              << "declared_edges: " << result.graph.declared_edges << '\n'
              << "unique_edges: " << edge_count << '\n'
              << "base_csr_and_attrs_mib: " << mib(compact_bytes) << '\n'
              << "wrote_device_graph: " << options.output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    if (argc < 2) {
      print_usage(argv[0]);
    }
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
