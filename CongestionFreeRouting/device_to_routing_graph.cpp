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
#include "interchange/gzip_io.hpp"
#include "interchange/import_policy.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
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
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using routing::interchange::Bounds;
using routing::interchange::DeviceRoutingGraph;
using routing::interchange::EdgeAttr;
using routing::interchange::EndpointAttachment;
using routing::interchange::EndpointAttachmentRole;
using routing::interchange::NodeBoundsMode;
using routing::interchange::NodeId;
using routing::interchange::LookupConflictPolicy;
using routing::interchange::PairNodeLookup;
using routing::interchange::PipData;
using routing::interchange::PseudoCellPinDirection;
using routing::interchange::PseudoCellPinResource;
using routing::interchange::StaticCsrEntry;
using routing::interchange::checked_lookup_string_id;
using routing::interchange::create_unique_staging_path;
using routing::interchange::kInvalidRouteNode;
using routing::interchange::kMissingRouteCoordinate;
using routing::interchange::kNoIndex;
using routing::interchange::kNoStringIndex;
using routing::interchange::node_bounds_mode_name;
using routing::interchange::parse_node_bounds_mode;
using routing::interchange::read_gzip_or_plain_chunks;
using routing::interchange::require_distinct_interchange_paths;
using routing::interchange::rebuild_endpoint_attachment_lookups;
using routing::interchange::sort_and_deduplicate_pair_node_lookups;
using routing::interchange::sort_and_deduplicate_site_pin_lookups;
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
  DevicePayload payload;
  read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t chunk_size) {
        if (chunk_size > std::numeric_limits<std::size_t>::max() -
                             payload.decoded_bytes) {
          throw std::runtime_error(
              "decoded device is too large for this host");
        }
        const std::size_t old_size = payload.decoded_bytes;
        payload.decoded_bytes += chunk_size;
        if (payload.decoded_bytes >
            std::numeric_limits<std::size_t>::max() -
                (sizeof(capnp::word) - 1)) {
          throw std::runtime_error(
              "decoded device word count overflows size_t");
        }
        const std::size_t word_count =
            (payload.decoded_bytes + sizeof(capnp::word) - 1) /
            sizeof(capnp::word);
        payload.words.resize(word_count);
        std::memcpy(reinterpret_cast<std::uint8_t*>(payload.words.data()) +
                        old_size,
                    data, chunk_size);

        // A word-at-a-time content fingerprint is sufficient for cache identity
        // and avoids another hash pass over a multi-gigabyte device.
        std::size_t offset = 0;
        while (offset + sizeof(std::uint64_t) <= chunk_size) {
          std::uint64_t lane = 0;
          std::memcpy(&lane, data + offset, sizeof(lane));
          payload.fingerprint ^=
              lane + 0x9e3779b97f4a7c15ULL + old_size + offset;
          payload.fingerprint *= 0xbf58476d1ce4e5b9ULL;
          payload.fingerprint = (payload.fingerprint << 27) |
                                (payload.fingerprint >> (64 - 27));
          offset += sizeof(lane);
        }
        std::uint64_t tail =
            static_cast<std::uint64_t>(chunk_size - offset) << 56;
        std::memcpy(&tail, data + offset, chunk_size - offset);
        payload.fingerprint ^= tail + payload.decoded_bytes;
        payload.fingerprint *= 0x94d049bb133111ebULL;
      });
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

std::optional<std::pair<std::int32_t, std::int32_t>> parse_tile_xy(
    std::string_view tile_name) {
  const std::size_t marker = tile_name.rfind("_X");
  if (marker == std::string_view::npos) {
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

std::uint64_t pair_key(std::uint32_t first, std::uint32_t second) {
  return (static_cast<std::uint64_t>(first) << 32) | second;
}

std::uint64_t mix_key(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

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
    if (values_[slot] != kInvalidRouteNode && values_[slot] != node) {
      throw std::runtime_error(
          "tile-wire lookup maps one key to multiple nodes");
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
};

struct AuditedAttachmentCandidate {
  routing::interchange::IobAttachmentRole policy_role =
      routing::interchange::IobAttachmentRole::kSource;
  NodeId from_node = kInvalidRouteNode;
  NodeId to_node = kInvalidRouteNode;
  std::uint32_t tile_string = 0;
  std::uint64_t pip_data_index = kNoIndex;
  std::uint32_t traversed_site_string = 0;
  std::vector<std::uint32_t> traversed_site_type_strings;
  std::vector<PseudoCellPinResource> pseudo_cell_pins;

  std::uint32_t endpoint_site_string = 0;
  std::uint32_t endpoint_site_type_string = 0;
  std::uint32_t endpoint_pin_string = 0;
  NodeId endpoint_node = kInvalidRouteNode;
};

struct AuditedEndpoint {
  routing::interchange::IobAttachmentRole role =
      routing::interchange::IobAttachmentRole::kSource;
  std::uint32_t site_string = 0;
  std::uint32_t site_type_string = 0;
  std::uint32_t pin_string = 0;
  NodeId node = kInvalidRouteNode;
};

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
  if (payload.decoded_bytes % sizeof(capnp::word) != 0) {
    throw std::runtime_error(
        "decoded Cap'n Proto input is not word-aligned: " +
        options.device_path.string() + " has " +
        std::to_string(payload.decoded_bytes) + " bytes");
  }

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.data(), payload.words.size()), reader_options);
  const auto device = reader.getRoot<DeviceResources::Device>();
  const auto device_strings = device.getStrList();

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
  const capnp::Text::Reader device_name = device.getName();
  graph.device_name_string = graph.string_table.intern(
      std::string(device_name.cStr(), device_name.size()));

  // All serialized lookup string IDs are required to fit below UINT32_MAX.
  // Use that otherwise-invalid value as the dense cache sentinel, cutting this
  // DeviceResources-sized array in half. The local string table is the only
  // decoded-string cache: keeping a second optional<string> for every FPGAIF
  // string consumed substantial memory on full devices.
  constexpr std::uint32_t kUnmappedLocalString =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> local_string_by_device_id(
      device_strings.size(), kUnmappedLocalString);
  auto intern_device_string = [&](std::uint32_t device_string) {
    if (device_string >= local_string_by_device_id.size()) {
      throw std::runtime_error("FPGAIF string index is out of range");
    }
    std::uint32_t& local = local_string_by_device_id[device_string];
    if (local == kUnmappedLocalString) {
      const capnp::Text::Reader text = device_strings[device_string];
      local = checked_lookup_string_id(graph.string_table.intern(
          std::string(text.cStr(), text.size())));
    }
    return local;
  };

  // Parse each tile coordinate once and retain a dense StringIdx-indexed table.
  // The old implementation reparsed a tile name for every wire in every node.
  std::vector<TileInfo> tile_info(device_strings.size());
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
    const capnp::Text::Reader tile_name = device_strings[tile.getName()];
    const auto xy = parse_tile_xy(
        std::string_view(tile_name.cStr(), tile_name.size()));
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
  graph.node_route_end_x.reserve(node_capacity);
  graph.node_route_end_y.reserve(node_capacity);
  graph.node_base_vertex_cost.reserve(node_capacity);

  // Build the final compact lookup records directly. A separate numeric
  // staging vector held the same key/node triples and briefly doubled this
  // per-wire storage while it was copied into graph.tile_wire_nodes.
  graph.tile_wire_nodes.reserve(static_cast<std::size_t>(wires.size()));

  for (std::uint32_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const auto node_wires = nodes[node_index].getWires();
    if (node_wires.size() == 0) {
      continue;
    }
    for (std::uint32_t offset = 0; offset < node_wires.size(); ++offset) {
      if (node_wires[offset] >= wires.size()) {
        throw std::runtime_error(
            "device node refers to an out-of-range wire");
      }
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
      graph.tile_wire_nodes.push_back(
          {checked_lookup_string_id(
               intern_device_string(wire.getTile())),
           checked_lookup_string_id(
               intern_device_string(wire.getWire())),
           compact_node,
           0});
    }
    if (!has_xy) {
      min_x = max_x = min_y = max_y = -1;
    }
    graph.node_min_x.push_back(min_x);
    graph.node_max_x.push_back(max_x);
    graph.node_min_y.push_back(min_y);
    graph.node_max_y.push_back(max_y);

    const auto metadata_wire = wires[metadata_wire_index];
    std::int32_t route_end_x = kMissingRouteCoordinate;
    std::int32_t route_end_y = kMissingRouteCoordinate;
    if (metadata_wire.getTile() < tile_info.size()) {
      const TileInfo& route_tile = tile_info[metadata_wire.getTile()];
      if (route_tile.has_xy) {
        route_end_x = route_tile.x;
        route_end_y = route_tile.y;
      }
    }
    graph.node_route_end_x.push_back(route_end_x);
    graph.node_route_end_y.push_back(route_end_y);
    // Static cost is deliberately factorized from mutable negotiated
    // congestion.  Unit cost preserves the current routing objective until a
    // characterized wire/type cost model replaces it.
    graph.node_base_vertex_cost.push_back(1.0f);
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

  FlatPairNodeMap tile_wire_map(graph.tile_wire_nodes.size());
  for (const PairNodeLookup& record : graph.tile_wire_nodes) {
    tile_wire_map.insert(record.first_string, record.second_string,
                         record.node);
  }
  (void)sort_and_deduplicate_pair_node_lookups(
      graph.tile_wire_nodes, LookupConflictPolicy::kReject, "tile-wire");

  std::vector<TileInstance> tile_instances;
  tile_instances.reserve(in_bounds_tile_indices.size());
  std::vector<std::uint8_t> used_tile_type(tile_types.size(), 0);
  for (const std::uint32_t tile_index : in_bounds_tile_indices) {
    const auto tile = tile_list[tile_index];
    if (tile.getType() >= tile_types.size()) {
      throw std::runtime_error("tile type index is out of range");
    }
    const std::uint32_t tile_string =
        intern_device_string(tile.getName());
    tile_instances.push_back(
        {tile_index, tile_string, tile.getType()});
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

  // Resolve concrete site/pin pairs before admitting endpoint attachments.
  // The audited pseudo-PIP corridors are authorized by an exact typed IOB
  // endpoint, so their construction needs the same primary/alternate mapping
  // used later by the per-design importer.
  const auto site_type_list = device.getSiteTypeList();
  std::vector<std::vector<std::uint32_t>> site_type_pin_names(
      site_type_list.size());
  std::vector<std::uint32_t> site_type_name_strings(site_type_list.size());
  for (std::uint32_t type_index = 0; type_index < site_type_list.size();
       ++type_index) {
    site_type_name_strings[type_index] = checked_lookup_string_id(
        intern_device_string(site_type_list[type_index].getName()));
    const auto pins = site_type_list[type_index].getPins();
    auto& names = site_type_pin_names[type_index];
    names.reserve(pins.size());
    for (std::uint32_t pin = 0; pin < pins.size(); ++pin) {
      names.push_back(checked_lookup_string_id(
          intern_device_string(pins[pin].getName())));
    }
  }

  std::vector<std::vector<std::vector<routing::interchange::SitePinTemplate>>>
      site_pin_templates(tile_types.size());
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
        throw std::runtime_error(
            "tile site type has an invalid primary site type");
      }
      const auto wires_for_pins = site_type.getPrimaryPinsToTileWires();
      const auto& pin_names = site_type_pin_names[primary_type];
      if (wires_for_pins.size() != pin_names.size()) {
        throw std::runtime_error(
            "primary site-pin and tile-wire counts do not match");
      }
      std::vector<std::uint32_t> primary_wires;
      primary_wires.reserve(wires_for_pins.size());
      for (std::uint32_t pin = 0; pin < wires_for_pins.size(); ++pin) {
        primary_wires.push_back(checked_lookup_string_id(
            intern_device_string(wires_for_pins[pin])));
      }

      const auto primary_site_type = site_type_list[primary_type];
      const auto alternate_types = primary_site_type.getAltSiteTypes();
      const auto alternate_parent_maps = site_type.getAltPinsToPrimaryPins();
      if (alternate_types.size() != alternate_parent_maps.size()) {
        throw std::runtime_error(
            "alternate site-type and parent-map counts do not match");
      }
      std::vector<routing::interchange::AlternateSitePinMap> alternates;
      alternates.reserve(alternate_types.size());
      for (std::uint32_t alternate = 0;
           alternate < alternate_types.size(); ++alternate) {
        const std::uint32_t alternate_type = alternate_types[alternate];
        if (alternate_type >= site_type_pin_names.size()) {
          throw std::runtime_error(
              "alternate site type index is out of range");
        }
        routing::interchange::AlternateSitePinMap alternate_map;
        alternate_map.site_type_string =
            site_type_name_strings[alternate_type];
        alternate_map.pin_strings = site_type_pin_names[alternate_type];
        const auto parent_pins = alternate_parent_maps[alternate].getPins();
        alternate_map.primary_pin_indices.reserve(parent_pins.size());
        for (std::uint32_t pin = 0; pin < parent_pins.size(); ++pin) {
          alternate_map.primary_pin_indices.push_back(parent_pins[pin]);
        }
        alternates.push_back(std::move(alternate_map));
      }
      by_site_type[site_type_index] =
          routing::interchange::build_site_pin_templates(
              site_type_name_strings[primary_type], pin_names,
              primary_wires, alternates);
    }
  }

  for (const TileInstance& tile_instance : tile_instances) {
    const auto tile = tile_list[tile_instance.tile_index];
    const auto sites = tile.getSites();
    const auto& by_site_type = site_pin_templates[tile_instance.tile_type];
    for (std::uint32_t site_index = 0; site_index < sites.size(); ++site_index) {
      const auto site = sites[site_index];
      if (site.getType() >= by_site_type.size()) {
        throw std::runtime_error(
            "tile site refers to an invalid tile-site-type index");
      }
      for (const routing::interchange::SitePinTemplate& pin :
           by_site_type[site.getType()]) {
        const std::optional<NodeId> node =
            tile_wire_map.find(tile_instance.tile_name,
                               pin.tile_wire_string);
        if (!node.has_value()) {
          continue;
        }
        graph.site_pin_nodes.push_back(
            {checked_lookup_string_id(intern_device_string(site.getName())),
             pin.site_type_string, pin.pin_string, *node});
      }
    }
  }
  sort_and_deduplicate_site_pin_lookups(graph.site_pin_nodes);

  // First identify the exact xcvu3p pseudo-PIP signatures and retain all
  // legality metadata that would otherwise be lost by the compact CSR.  A
  // candidate is not emitted until a conventional one-edge corridor proves
  // that it belongs to one exact typed IOB endpoint.
  std::vector<AuditedAttachmentCandidate> attachment_candidates;
  const std::string cached_device_name =
      graph.string_table.strings[graph.device_name_string];
  for (const TileInstance& tile_instance : tile_instances) {
    const auto tile_type = tile_types[tile_instance.tile_type];
    const std::uint32_t tile_type_name_string =
        intern_device_string(tile_type.getName());
    const std::string tile_type_name =
        graph.string_table.strings[tile_type_name_string];
    if (cached_device_name != "xcvu3p" ||
        tile_type_name != "XIPHY_BYTE_L") {
      continue;
    }

    const auto type_wires = tile_type.getWires();
    const auto pips = tile_type.getPips();
    for (std::uint32_t pip_index = 0; pip_index < pips.size(); ++pip_index) {
      const auto pip = pips[pip_index];
      const PipTemplate& pip_template =
          pip_templates[tile_instance.tile_type][pip_index];
      const std::string wire0 =
          graph.string_table.strings[pip_template.wire0];
      const std::string wire1 =
          graph.string_table.strings[pip_template.wire1];
      const auto policy =
          routing::interchange::classify_audited_iob_attachment_pip(
              cached_device_name, tile_type_name, wire0, wire1,
              pip_template.conventional, pip.getDirectional());
      if (!policy.has_value()) {
        continue;
      }
      if (!pip.isPseudoCells()) {
        throw std::runtime_error(
            "audited IOB attachment PIP has no pseudo-cell metadata");
      }

      const std::optional<NodeId> node0 = tile_wire_map.find(
          tile_instance.tile_name, pip_template.wire0);
      const std::optional<NodeId> node1 = tile_wire_map.find(
          tile_instance.tile_name, pip_template.wire1);
      if (!node0.has_value() || !node1.has_value()) {
        continue;
      }

      // Locate the unique traversed site by requiring both PIP wires to map
      // to the expected primary site pins in one SiteTypeInTileType record.
      const auto tile_site_types = tile_type.getSiteTypes();
      std::optional<std::uint32_t> traversed_tile_site_type;
      std::uint32_t from_primary_pin = 0;
      std::uint32_t to_primary_pin = 0;
      for (std::uint32_t site_type_index = 0;
           site_type_index < tile_site_types.size(); ++site_type_index) {
        const auto site_type = tile_site_types[site_type_index];
        const auto pin_wires = site_type.getPrimaryPinsToTileWires();
        std::optional<std::uint32_t> found_from;
        std::optional<std::uint32_t> found_to;
        for (std::uint32_t pin = 0; pin < pin_wires.size(); ++pin) {
          if (pin_wires[pin] == type_wires[pip.getWire0()]) {
            if (found_from.has_value()) {
              throw std::runtime_error(
                  "audited IOB PIP wire maps to duplicate site pins");
            }
            found_from = pin;
          }
          if (pin_wires[pin] == type_wires[pip.getWire1()]) {
            if (found_to.has_value()) {
              throw std::runtime_error(
                  "audited IOB PIP wire maps to duplicate site pins");
            }
            found_to = pin;
          }
        }
        if (!found_from.has_value() || !found_to.has_value()) {
          continue;
        }
        if (traversed_tile_site_type.has_value()) {
          throw std::runtime_error(
              "audited IOB PIP maps through more than one site");
        }
        traversed_tile_site_type = site_type_index;
        from_primary_pin = *found_from;
        to_primary_pin = *found_to;
      }
      if (!traversed_tile_site_type.has_value()) {
        throw std::runtime_error(
            "audited IOB PIP does not map through a concrete site");
      }

      const auto traversed_site_type_in_tile =
          tile_site_types[*traversed_tile_site_type];
      const std::uint32_t primary_type_index =
          traversed_site_type_in_tile.getPrimaryType();
      if (primary_type_index >= site_type_list.size()) {
        throw std::runtime_error(
            "audited IOB PIP primary site type is out of range");
      }
      const auto primary_site_type = site_type_list[primary_type_index];
      const auto primary_pins = primary_site_type.getPins();
      if (from_primary_pin >= primary_pins.size() ||
          to_primary_pin >= primary_pins.size()) {
        throw std::runtime_error(
            "audited IOB PIP primary site pin is out of range");
      }
      const std::string from_site_pin = graph.string_table.strings[
          intern_device_string(primary_pins[from_primary_pin].getName())];
      const std::string to_site_pin = graph.string_table.strings[
          intern_device_string(primary_pins[to_primary_pin].getName())];
      if (from_site_pin != policy->from_site_pin ||
          to_site_pin != policy->to_site_pin) {
        throw std::runtime_error(
            "audited IOB PIP site-pin mapping changed unexpectedly");
      }

      const auto concrete_sites =
          tile_list[tile_instance.tile_index].getSites();
      std::optional<std::uint32_t> traversed_site_string;
      for (std::uint32_t site_index = 0; site_index < concrete_sites.size();
           ++site_index) {
        const auto site = concrete_sites[site_index];
        if (site.getType() != *traversed_tile_site_type) {
          continue;
        }
        if (traversed_site_string.has_value()) {
          throw std::runtime_error(
              "audited IOB PIP maps to duplicate concrete sites");
        }
        traversed_site_string = checked_lookup_string_id(
            intern_device_string(site.getName()));
      }
      if (!traversed_site_string.has_value()) {
        throw std::runtime_error(
            "audited IOB PIP concrete traversed site is missing");
      }

      AuditedAttachmentCandidate candidate;
      candidate.policy_role = policy->role;
      candidate.from_node = *node0;
      candidate.to_node = *node1;
      candidate.tile_string = tile_instance.tile_name;
      candidate.traversed_site_string = *traversed_site_string;
      candidate.traversed_site_type_strings.push_back(
          site_type_name_strings[primary_type_index]);

      // An alternate type is legal only when it exposes both primary pins of
      // this directed crossing. This admits BITSLICE_COMPONENT_RX_TX for the
      // benchmark while rejecting TX-only use of the RX attachment and vice
      // versa. DeviceResources defines the PIP's pseudoCells in the primary
      // site namespace; component alternates intentionally omit some of those
      // routing BELs, so their canonical resource keys remain the primary
      // BEL/pin names and are checked against placement/fixed occupancy later.
      const auto alternate_types = primary_site_type.getAltSiteTypes();
      const auto alternate_parent_maps =
          traversed_site_type_in_tile.getAltPinsToPrimaryPins();
      if (alternate_types.size() != alternate_parent_maps.size()) {
        throw std::runtime_error(
            "audited IOB alternate site-type maps are inconsistent");
      }
      for (std::uint32_t alternate = 0;
           alternate < alternate_types.size(); ++alternate) {
        const std::uint32_t alternate_type = alternate_types[alternate];
        if (alternate_type >= site_type_name_strings.size()) {
          throw std::runtime_error(
              "audited IOB alternate site type is out of range");
        }
        bool maps_from = false;
        bool maps_to = false;
        const auto parents = alternate_parent_maps[alternate].getPins();
        for (std::uint32_t pin = 0; pin < parents.size(); ++pin) {
          maps_from = maps_from || parents[pin] == from_primary_pin;
          maps_to = maps_to || parents[pin] == to_primary_pin;
        }
        if (maps_from && maps_to) {
          candidate.traversed_site_type_strings.push_back(
              site_type_name_strings[alternate_type]);
        }
      }
      std::sort(candidate.traversed_site_type_strings.begin(),
                candidate.traversed_site_type_strings.end());
      candidate.traversed_site_type_strings.erase(
          std::unique(candidate.traversed_site_type_strings.begin(),
                      candidate.traversed_site_type_strings.end()),
          candidate.traversed_site_type_strings.end());

      const auto pseudo_cells = pip.getPseudoCells();
      if (pseudo_cells.size() != 3) {
        throw std::runtime_error(
            "audited IOB PIP pseudo-cell count changed unexpectedly");
      }
      std::array<PseudoCellPinResource, 4> ordered_resources{};
      std::uint32_t resource_mask = 0;
      std::uint32_t resource_count = 0;
      const auto primary_bel_pins = primary_site_type.getBelPins();
      for (std::uint32_t cell_index = 0; cell_index < pseudo_cells.size();
           ++cell_index) {
        const auto pseudo_cell = pseudo_cells[cell_index];
        const std::uint32_t bel_string = checked_lookup_string_id(
            intern_device_string(pseudo_cell.getBel()));
        const std::string bel = graph.string_table.strings[bel_string];
        const auto pins = pseudo_cell.getPins();
        for (std::uint32_t pin_index = 0; pin_index < pins.size();
             ++pin_index) {
          ++resource_count;
          const std::uint32_t pin_string = checked_lookup_string_id(
              intern_device_string(pins[pin_index]));
          const std::string pin = graph.string_table.strings[pin_string];
          const auto bit =
              routing::interchange::audited_iob_pseudo_resource_bit(
                  policy->kind, bel, pin);
          if (!bit.has_value() || *bit >= ordered_resources.size() ||
              (resource_mask & (1U << *bit)) != 0) {
            throw std::runtime_error(
                "audited IOB PIP pseudo-cell signature changed");
          }

          std::optional<PseudoCellPinDirection> direction;
          for (std::uint32_t bel_pin_index = 0;
               bel_pin_index < primary_bel_pins.size(); ++bel_pin_index) {
            const auto bel_pin = primary_bel_pins[bel_pin_index];
            if (bel_pin.getBel() != pseudo_cell.getBel() ||
                bel_pin.getName() != pins[pin_index]) {
              continue;
            }
            if (direction.has_value()) {
              throw std::runtime_error(
                  "audited IOB pseudo-cell BEL pin is ambiguous");
            }
            const std::uint32_t raw_direction =
                static_cast<std::uint32_t>(bel_pin.getDir());
            if (raw_direction >
                static_cast<std::uint32_t>(PseudoCellPinDirection::kInout)) {
              throw std::runtime_error(
                  "audited IOB pseudo-cell BEL pin direction is invalid");
            }
            direction = static_cast<PseudoCellPinDirection>(raw_direction);
          }
          if (!direction.has_value()) {
            throw std::runtime_error(
                "audited IOB pseudo-cell BEL pin is absent from primary type");
          }
          ordered_resources[*bit] = {bel_string, pin_string, *direction};
          resource_mask |= 1U << *bit;
        }
      }
      if (resource_count != ordered_resources.size() ||
          resource_mask != 0xfU) {
        throw std::runtime_error(
            "audited IOB PIP pseudo-cell resources are incomplete");
      }
      candidate.pseudo_cell_pins.assign(ordered_resources.begin(),
                                        ordered_resources.end());
      std::sort(candidate.pseudo_cell_pins.begin(),
                candidate.pseudo_cell_pins.end(),
                [](const PseudoCellPinResource& lhs,
                   const PseudoCellPinResource& rhs) {
                  if (lhs.bel_string != rhs.bel_string) {
                    return lhs.bel_string < rhs.bel_string;
                  }
                  if (lhs.pin_string != rhs.pin_string) {
                    return lhs.pin_string < rhs.pin_string;
                  }
                  return static_cast<std::uint32_t>(lhs.direction) <
                         static_cast<std::uint32_t>(rhs.direction);
                });

      // Attachment PipData records are intentionally not globally
      // deduplicated. Their unique index is the sparse identity carried from
      // this concrete site through final route reconstruction.
      candidate.pip_data_index = graph.pip_data.size();
      graph.pip_data.push_back(
          {pip_template.wire0, pip_template.wire1, true});
      attachment_candidates.push_back(std::move(candidate));
    }
  }

  auto for_each_conventional_edge = [&](auto&& callback) {
    for (const TileInstance& tile : tile_instances) {
      for (PipTemplate& pip : pip_templates[tile.tile_type]) {
        // Unsupported pseudo PIPs remain excluded. Audited IOB crossings are
        // appended separately only after exact endpoint authorization.
        if (!routing::interchange::include_pip_in_static_graph(
                pip.conventional)) {
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
        callback(*node0, *node1, tile.tile_name,
                 pip.forward_pip_data);
        if (pip.bidirectional) {
          if (pip.reverse_pip_data == kNoIndex) {
            pip.reverse_pip_data =
                intern_pip_data(pip.wire0, pip.wire1, false);
          }
          callback(*node1, *node0, tile.tile_name,
                   pip.reverse_pip_data);
        }
      }
    }
  };

  std::vector<AuditedEndpoint> audited_endpoints;
  std::unordered_map<NodeId, std::vector<std::size_t>> endpoints_by_node;
  for (const auto& endpoint : graph.site_pin_nodes) {
    const std::string& site =
        graph.string_table.strings[endpoint.site_string];
    const std::string& site_type =
        graph.string_table.strings[endpoint.site_type_string];
    const std::string& pin =
        graph.string_table.strings[endpoint.pin_string];
    for (const auto role : {routing::interchange::IobAttachmentRole::kSource,
                            routing::interchange::IobAttachmentRole::kSink}) {
      if (!routing::interchange::is_audited_iob_endpoint(
              role, site, site_type, pin)) {
        continue;
      }
      const std::size_t index = audited_endpoints.size();
      audited_endpoints.push_back({role, endpoint.site_string,
                                   endpoint.site_type_string,
                                   endpoint.pin_string, endpoint.node});
      endpoints_by_node[endpoint.node].push_back(index);
    }
  }

  std::unordered_map<NodeId, std::vector<std::size_t>>
      source_candidates_by_from;
  std::unordered_map<NodeId, std::vector<std::size_t>> sink_candidates_by_to;
  for (std::size_t index = 0; index < attachment_candidates.size(); ++index) {
    const auto& candidate = attachment_candidates[index];
    if (candidate.policy_role ==
        routing::interchange::IobAttachmentRole::kSource) {
      source_candidates_by_from[candidate.from_node].push_back(index);
    } else {
      sink_candidates_by_to[candidate.to_node].push_back(index);
    }
  }

  auto authorize_candidate = [&](std::size_t candidate_index,
                                 const AuditedEndpoint& endpoint) {
    AuditedAttachmentCandidate& candidate =
        attachment_candidates[candidate_index];
    if (candidate.policy_role != endpoint.role) {
      return;
    }
    if (candidate.endpoint_node != kInvalidRouteNode) {
      if (candidate.endpoint_node != endpoint.node ||
          candidate.endpoint_site_string != endpoint.site_string ||
          candidate.endpoint_site_type_string != endpoint.site_type_string ||
          candidate.endpoint_pin_string != endpoint.pin_string) {
        throw std::runtime_error(
            "audited IOB attachment corridor reaches multiple endpoints");
      }
      return;
    }
    candidate.endpoint_site_string = endpoint.site_string;
    candidate.endpoint_site_type_string = endpoint.site_type_string;
    candidate.endpoint_pin_string = endpoint.pin_string;
    candidate.endpoint_node = endpoint.node;
  };

  // Match only the exact conventional edge adjacent to the endpoint. This
  // proves the fixed two-edge source/sink corridor and supplies its concrete
  // endpoint ownership without admitting the pseudo edge as general transit.
  for_each_conventional_edge(
      [&](NodeId from, NodeId to, std::uint64_t, std::uint64_t) {
        const auto source_candidates = source_candidates_by_from.find(to);
        const auto source_endpoints = endpoints_by_node.find(from);
        if (source_candidates != source_candidates_by_from.end() &&
            source_endpoints != endpoints_by_node.end()) {
          for (const std::size_t candidate : source_candidates->second) {
            for (const std::size_t endpoint : source_endpoints->second) {
              authorize_candidate(candidate, audited_endpoints[endpoint]);
            }
          }
        }

        const auto sink_candidates = sink_candidates_by_to.find(from);
        const auto sink_endpoints = endpoints_by_node.find(to);
        if (sink_candidates != sink_candidates_by_to.end() &&
            sink_endpoints != endpoints_by_node.end()) {
          for (const std::size_t candidate : sink_candidates->second) {
            for (const std::size_t endpoint : sink_endpoints->second) {
              authorize_candidate(candidate, audited_endpoints[endpoint]);
            }
          }
        }
      });

  std::vector<std::size_t> accepted_attachment_candidates;
  accepted_attachment_candidates.reserve(attachment_candidates.size());
  for (std::size_t candidate_index = 0;
       candidate_index < attachment_candidates.size(); ++candidate_index) {
    const AuditedAttachmentCandidate& candidate =
        attachment_candidates[candidate_index];
    if (candidate.endpoint_node == kInvalidRouteNode) {
      continue;
    }
    EndpointAttachment attachment;
    attachment.endpoint_site_string = candidate.endpoint_site_string;
    attachment.endpoint_site_type_string =
        candidate.endpoint_site_type_string;
    attachment.endpoint_pin_string = candidate.endpoint_pin_string;
    attachment.role =
        candidate.policy_role ==
                routing::interchange::IobAttachmentRole::kSource
            ? EndpointAttachmentRole::kSource
            : EndpointAttachmentRole::kSink;
    attachment.endpoint_node = candidate.endpoint_node;
    attachment.from_node = candidate.from_node;
    attachment.to_node = candidate.to_node;
    attachment.pip_data_index = candidate.pip_data_index;
    attachment.traversed_site_string = candidate.traversed_site_string;
    attachment.traversed_site_type_begin =
        graph.endpoint_attachment_traversed_site_types.size();
    attachment.traversed_site_type_count =
        candidate.traversed_site_type_strings.size();
    graph.endpoint_attachment_traversed_site_types.insert(
        graph.endpoint_attachment_traversed_site_types.end(),
        candidate.traversed_site_type_strings.begin(),
        candidate.traversed_site_type_strings.end());
    attachment.pseudo_cell_pin_begin =
        graph.endpoint_attachment_pseudo_cell_pins.size();
    attachment.pseudo_cell_pin_count = candidate.pseudo_cell_pins.size();
    graph.endpoint_attachment_pseudo_cell_pins.insert(
        graph.endpoint_attachment_pseudo_cell_pins.end(),
        candidate.pseudo_cell_pins.begin(), candidate.pseudo_cell_pins.end());
    graph.endpoint_attachments.push_back(attachment);
    accepted_attachment_candidates.push_back(candidate_index);
  }
  rebuild_endpoint_attachment_lookups(graph);

  auto for_each_edge = [&](auto&& callback) {
    for_each_conventional_edge(callback);
    for (const std::size_t candidate_index :
         accepted_attachment_candidates) {
      const AuditedAttachmentCandidate& candidate =
          attachment_candidates[candidate_index];
      callback(candidate.from_node, candidate.to_node,
               candidate.tile_string, candidate.pip_data_index);
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

  return result;
}

double mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void write_graph_atomically(const DeviceRoutingGraph& graph,
                            const std::vector<StaticCsrEntry>& entries,
                            const std::filesystem::path& output_path) {
  const std::filesystem::path temporary =
      create_unique_staging_path(output_path);
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
    require_distinct_interchange_paths(
        {options.device_path, options.output_path});
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
              << "typed_site_pin_aliases: "
              << result.graph.site_pin_nodes.size() << '\n'
              << "endpoint_attachments: "
              << result.graph.endpoint_attachments.size() << '\n'
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
