// Converts FPGA Interchange DeviceResources plus a design's PhysicalNetlist
// and LogicalNetlist into the CSR binary format consumed by
// HIP_kernel/bellman_ford.
//
// The primary output is a RIPSCSR1 .csrbin file with the same layout as
// HIP_kernel/bellman_ford/src/gr_to_csr.cpp:
//
//   CSR row v, column u = routing edge u -> v, with unit edge weight.
//
// This incoming-edge orientation is what the min-plus Bellman-Ford kernels
// expect. A second sidecar file preserves the FPGA-specific NetworkX-style
// attributes:
//
//   edge attribute "pip" = (tileName, pipDataIndex)
//   pipData[pipDataIndex] = (wire0Name, wire1Name, forward)
//   node attribute "sp" = (siteName, pinName), for sink site pins
//
// The sidecar also records route requests extracted from PhysicalNetlist stubs
// and source site pins, logical net summaries extracted from LogicalNetlist,
// and the original decompressed .phys/.netlist payloads. That metadata is
// intentionally CPU-side: the GPU CSR remains compact and compatible with the
// current Bellman-Ford readers, while later post-processing has enough context
// to turn SSSP paths back into PhysPIP route branches in a routed .phys file.
//
// Expected generated schema headers:
//   DeviceResources.capnp.h
//   PhysicalNetlist.capnp.h
//   LogicalNetlist.capnp.h
//
// Example compile command, after generating the C++ Cap'n Proto schema files:
//
//   g++ -std=c++17 -O3 \
//     -I<generated-schema-dir> \
//     CongestionFreeRouting/interchange_to_csr.cpp \
//     <generated-schema-dir>/DeviceResources.capnp.c++ \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     <generated-schema-dir>/LogicalNetlist.capnp.c++ \
//     <generated-schema-dir>/References.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o interchange_to_csr
//
// Example use:
//
//   ./interchange_to_csr benchmarks/vtr_mcml_unrouted.phys \
//     benchmarks/vtr_mcml.netlist \
//     HIP_kernel/bellman_ford/data/vtr_mcml_fpga.csrbin \
//     --device xcvu3p.device

#include "DeviceResources.capnp.h"
#include "LogicalNetlist.capnp.h"
#include "PhysicalNetlist.capnp.h"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Keep the on-disk binary formats explicit. The CSR constants match the
// existing Bellman-Ford loader; the metadata constants identify the sidecar
// that stores FPGA-specific information not present in plain CSR.
static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t CSR_FORMAT_VERSION = 1;
constexpr std::uint64_t METADATA_FORMAT_VERSION = 2;
constexpr std::uint64_t INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t kNoIndex =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;
constexpr std::uint64_t kNoStringIndex = kNoIndex;

using NodeId = std::int32_t;
constexpr NodeId kInvalidRouteNode = -1;

// Controls how the bounded tile box is applied to DeviceResources.nodes.
// The Python proof of concept uses the first wire of each node as the "base"
// wire and admits the node if that base wire's tile is in bounds.
enum class NodeBoundsMode {
  kPocBaseWire,
  kFullyContained,
  kIntersects,
};

constexpr int NXROUTE_MIN_X = 36;
constexpr int NXROUTE_MAX_X = 90;
constexpr int NXROUTE_MIN_Y = 60;
constexpr int NXROUTE_MAX_Y = 239;

// Import every XY tile by default. Use --nxroute-bounds only for apples-to-
// apples comparison with the NetworkX proof-of-concept router.
struct Bounds {
  int min_x = 0;
  int max_x = std::numeric_limits<int>::max();
  int min_y = 0;
  int max_y = std::numeric_limits<int>::max();
};

// Concrete site instances are resolved through their tile instance, tile type,
// and site type. This is the C++ equivalent of site2tileAndTypes in Python.
struct TileAndTypes {
  std::string tile_name;
  std::uint32_t tile_type = 0;
  std::uint32_t site_type = 0;
};

// Key used for tile-type-local site pin lookup:
//   (site type within tile type, site pin name) -> tile wire name.
struct SitePinKey {
  std::uint32_t site_type = 0;
  std::string pin_name;

  bool operator==(const SitePinKey& other) const {
    return site_type == other.site_type && pin_name == other.pin_name;
  }
};

struct SitePinKeyHash {
  std::size_t operator()(const SitePinKey& key) const {
    const std::size_t h1 = std::hash<std::uint32_t>{}(key.site_type);
    const std::size_t h2 = std::hash<std::string>{}(key.pin_name);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// Sidecar metadata uses a compact local string table. CSR itself only needs
// numeric row/column IDs, while reconstruction later needs names like tile,
// wire, site, pin, and net.
struct StringTable {
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint64_t> ids;

  std::uint64_t intern(const std::string& text) {
    auto found = ids.find(text);
    if (found != ids.end()) {
      return found->second;
    }

    const std::uint64_t id = static_cast<std::uint64_t>(strings.size());
    strings.push_back(text);
    ids.emplace(strings.back(), id);
    return id;
  }
};

// FPGAIF stores most names as integer indexes into strList. This cache turns
// those indexes into std::string once, avoiding repeated Cap'n Proto text copies
// while parsing large devices.
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
      cached = std::string(text.cStr(), text.size());
    }
    return *cached;
  }

 private:
  capnp::List<capnp::Text>::Reader strings_;
  std::vector<std::optional<std::string>> cache_;
};

// Per-PIP metadata follows the Python proof of concept:
//   pipData[pipDataIndex] = (wire0Name, wire1Name, forward)
struct PipData {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
};

// Per-edge metadata follows the Python NetworkX edge attribute:
//   edge["pip"] = (tileName, pipDataIndex)
struct EdgeAttr {
  std::uint64_t tile_string = 0;
  std::uint64_t pip_data_index = 0;
};

// Temporary edge representation while building the graph in natural routing
// direction. The final CSR is written in incoming-edge orientation.
struct BuildEdge {
  NodeId from = 0;
  NodeId to = 0;
  EdgeAttr attr;
};

// Site pin metadata is attached to sink nodes, matching the Python node
// attribute "sp" = (siteName, pinName).
struct SitePinNode {
  NodeId node = -1;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

// A design-specific route request extracted from a PhysicalNetlist. Sources
// and sinks are stored as graph node IDs plus their human-readable site pins.
struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

// One logical port instance inside a LogicalNetlist net. This is CPU metadata
// only; it lets a later routed-.phys writer relate physical route requests back
// to logical connectivity if it needs to validate or annotate the result.
struct LogicalPortInstanceSummary {
  std::uint64_t port_string = 0;
  std::uint64_t instance_string = kNoStringIndex;
  std::uint32_t port_index = 0;
  std::uint32_t instance_index = 0;
  std::uint32_t bus_index = 0;
  bool has_bus_index = false;
  bool is_external_port = false;
};

// One LogicalNetlist net, stored with a slice into logical_port_instances.
struct LogicalNetSummary {
  std::uint64_t name_string = 0;
  std::uint64_t cell_index = 0;
  std::uint64_t port_instance_begin = 0;
  std::uint64_t port_instance_count = 0;
};

// One LogicalNetlist cell, stored with a slice into logical_nets.
struct LogicalCellSummary {
  std::uint64_t declaration_name_string = 0;
  std::uint64_t net_begin = 0;
  std::uint64_t net_count = 0;
};

// Plain CSR payload consumed by HIP_kernel/bellman_ford. edge_attrs is kept in
// the same order as colind/values so each CSR edge can recover its PIP metadata.
struct CsrGraph {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<float> values;
  std::vector<EdgeAttr> edge_attrs;
};

// All CPU-side information produced while parsing FPGAIF. The CSR arrays are
// derived from this structure; the metadata sidecar also serializes selected
// fields from it.
struct RoutingGraph {
  StringTable string_table;
  std::vector<std::uint64_t> node_device_ids;
  std::vector<BuildEdge> edges;
  std::vector<PipData> pip_data;
  std::uint64_t raw_edges_seen = 0;

  std::unordered_map<std::string, std::unordered_map<std::string, NodeId>>
      tile2wire2node;
  std::unordered_map<std::string, TileAndTypes> site2tile_and_types;
  std::unordered_map<
      std::uint32_t,
      std::unordered_map<SitePinKey, std::string, SitePinKeyHash>>
      tile_type2site_type_pin2wire;

  std::vector<std::uint8_t> blocked_node;
  std::vector<std::uint8_t> sink_node_stops;
  std::vector<std::int64_t> site_pin_attr_by_node;
  std::vector<SitePinNode> site_pin_attrs;
  std::vector<RouteRequest> route_requests;

  std::uint64_t device_path_string = 0;
  std::uint64_t physical_path_string = 0;
  std::uint64_t logical_path_string = 0;
  std::uint64_t logical_design_name_string = 0;
  std::vector<LogicalCellSummary> logical_cells;
  std::vector<LogicalNetSummary> logical_nets;
  std::vector<LogicalPortInstanceSummary> logical_port_instances;
  std::unordered_map<std::string, std::uint64_t> logical_net_index_by_name;
  std::vector<std::uint8_t> physical_netlist_bytes;
  std::vector<std::uint8_t> logical_netlist_bytes;
};

// Lightweight site-pin name pair used while walking PhysicalNetlist route
// branches before those names are mapped into graph nodes.
struct SitePinName {
  std::string site;
  std::string pin;
};

// Command-line options: PhysicalNetlist and LogicalNetlist inputs are required.
// DeviceResources is still required for topology, but it defaults to
// xcvu3p.device or can be supplied by --device.
struct Options {
  std::filesystem::path device_path = "xcvu3p.device";
  std::filesystem::path phys_path;
  std::filesystem::path logical_path;
  std::filesystem::path output_path;
  std::filesystem::path metadata_path;
  Bounds bounds;
  NodeBoundsMode node_bounds_mode = NodeBoundsMode::kPocBaseWire;
};

bool has_prefix(const std::string& text, const char* prefix) {
  const std::string prefix_text(prefix);
  return text.size() >= prefix_text.size() &&
         text.compare(0, prefix_text.size(), prefix_text) == 0;
}

std::optional<std::pair<int, int>> parse_tile_xy(const std::string& tile_name) {
  const std::size_t marker = tile_name.rfind("_X");
  if (marker == std::string::npos) {
    return std::nullopt;
  }

  std::size_t pos = marker + 2;
  if (pos >= tile_name.size() || tile_name[pos] < '0' || tile_name[pos] > '9') {
    return std::nullopt;
  }

  int x = 0;
  while (pos < tile_name.size() && tile_name[pos] >= '0' && tile_name[pos] <= '9') {
    x = x * 10 + (tile_name[pos] - '0');
    ++pos;
  }

  if (pos >= tile_name.size() || tile_name[pos] != 'Y') {
    return std::nullopt;
  }
  ++pos;
  if (pos >= tile_name.size() || tile_name[pos] < '0' || tile_name[pos] > '9') {
    return std::nullopt;
  }

  int y = 0;
  while (pos < tile_name.size() && tile_name[pos] >= '0' && tile_name[pos] <= '9') {
    y = y * 10 + (tile_name[pos] - '0');
    ++pos;
  }

  return std::make_pair(x, y);
}

// Print the node-bounds mode in logs so graph size can be interpreted without
// guessing which subset policy was used.
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

// Parse the explicit node filtering policy. The default is poc-base-wire
// because that matches nxroute-poc.py. fully-contained is useful when you want
// the graph to contain only routing nodes whose every wire lies in the box.
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

// Print the supported invocation forms. The .phys and .netlist files describe
// the design; DeviceResources describes the routing fabric and is supplied with
// --device or defaults to ./xcvu3p.device.
void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <unrouted.phys> <logical.netlist> <output.csrbin> [options]\n"
      << "  " << program
      << " <device.device> <unrouted.phys> <logical.netlist> "
         "<output.csrbin> [options]\n\n"
      << "Options:\n"
      << "  --device <path>                FPGAIF DeviceResources input.\n"
      << "  --metadata <path>              Sidecar FPGA metadata output.\n"
      << "  --bounds <minX> <maxX> <minY> <maxY>\n"
      << "                                 Tile-coordinate subset to import.\n"
      << "  --nxroute-bounds               Import nxroute-poc subset: X36..X90, Y60..Y239.\n"
      << "  --node-bounds-mode <mode>      poc-base-wire, fully-contained, "
         "or intersects.\n"
      << "  --full-device                  Import every tile that has XY coords (default).\n\n"
      << "Default bounds import the whole device graph.\n";
}

std::filesystem::path default_metadata_path(
    const std::filesystem::path& output_path) {
  std::filesystem::path path = output_path;
  path += ".ifmeta.bin";
  return path;
}

// Parse integer CLI arguments with a useful error message. This is used both
// for explicit --bounds and for tile coordinate text captured from tile names.
int parse_int_arg(const char* text, const char* name) {
  try {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
}

// Split positional inputs from options:
//   physical netlist + logical netlist + output
//   device + physical netlist + logical netlist + output
// Device path can also be supplied by --device. Metadata path and tile bounds
// are independent options.
Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<std::filesystem::path> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--device") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--device requires a path");
      }
      options.device_path = argv[++i];
      continue;
    }

    if (arg == "--metadata") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--metadata requires a path");
      }
      options.metadata_path = argv[++i];
      continue;
    }

    if (arg == "--bounds") {
      if (i + 4 >= argc) {
        throw std::runtime_error("--bounds requires four integer arguments");
      }
      options.bounds.min_x = parse_int_arg(argv[++i], "minX");
      options.bounds.max_x = parse_int_arg(argv[++i], "maxX");
      options.bounds.min_y = parse_int_arg(argv[++i], "minY");
      options.bounds.max_y = parse_int_arg(argv[++i], "maxY");
      continue;
    }

    if (arg == "--nxroute-bounds") {
      options.bounds.min_x = NXROUTE_MIN_X;
      options.bounds.max_x = NXROUTE_MAX_X;
      options.bounds.min_y = NXROUTE_MIN_Y;
      options.bounds.max_y = NXROUTE_MAX_Y;
      continue;
    }

    if (arg == "--node-bounds-mode") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--node-bounds-mode requires a mode");
      }
      options.node_bounds_mode = parse_node_bounds_mode(argv[++i]);
      continue;
    }

    if (arg == "--full-device") {
      options.bounds.min_x = 0;
      options.bounds.min_y = 0;
      options.bounds.max_x = std::numeric_limits<int>::max();
      options.bounds.max_y = std::numeric_limits<int>::max();
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    }

    positional.emplace_back(arg);
  }

  if (positional.size() == 3) {
    options.phys_path = positional[0];
    options.logical_path = positional[1];
    options.output_path = positional[2];
  } else if (positional.size() == 4) {
    options.device_path = positional[0];
    options.phys_path = positional[1];
    options.logical_path = positional[2];
    options.output_path = positional[3];
  } else {
    throw std::runtime_error("expected three or four positional arguments");
  }

  if (options.metadata_path.empty()) {
    options.metadata_path = default_metadata_path(options.output_path);
  }

  if (options.bounds.min_x > options.bounds.max_x ||
      options.bounds.min_y > options.bounds.max_y) {
    throw std::runtime_error("invalid bounds: min is greater than max");
  }

  return options;
}

// FPGAIF files used by the contest are gzipped Cap'n Proto messages. zlib's
// gzopen also handles plain files, which makes this helper tolerant of either
// compressed or already-decompressed inputs.
std::vector<std::uint8_t> read_gzip_or_plain_file(
    const std::filesystem::path& path) {
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (!file) {
    throw std::runtime_error("could not open input file: " + path.string());
  }

  std::vector<std::uint8_t> bytes;
  std::array<std::uint8_t, 1 << 20> buffer{};

  while (true) {
    const int read_count =
        gzread(file, buffer.data(), static_cast<unsigned int>(buffer.size()));
    if (read_count > 0) {
      const std::size_t old_size = bytes.size();
      bytes.resize(old_size + static_cast<std::size_t>(read_count));
      std::memcpy(bytes.data() + old_size,
                  buffer.data(),
                  static_cast<std::size_t>(read_count));
      continue;
    }

    if (read_count == 0) {
      break;
    }

    int zlib_error = 0;
    const char* message = gzerror(file, &zlib_error);
    gzclose(file);
    throw std::runtime_error("failed while reading " + path.string() + ": " +
                             (message ? message : "zlib error"));
  }

  gzclose(file);

  if (bytes.empty()) {
    throw std::runtime_error("input file is empty: " + path.string());
  }

  return bytes;
}

// Cap'n Proto's FlatArrayMessageReader expects word-aligned storage. The file
// is byte-oriented, so this copies it into a padded vector<capnp::word>.
std::vector<capnp::word> bytes_to_words(
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t word_size = sizeof(capnp::word);
  const std::size_t word_count = (bytes.size() + word_size - 1) / word_size;
  std::vector<capnp::word> words(word_count);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

void check_device_payload_size(const std::filesystem::path& path,
                               std::size_t decoded_bytes) {
  constexpr std::size_t kMinExpectedDeviceBytes = 1 << 20;
  if (path.filename() != "xcvu3p.device" ||
      decoded_bytes >= kMinExpectedDeviceBytes) {
    return;
  }

  throw std::runtime_error(
      "device resources file is suspiciously small: " + path.string() +
      " has only " + std::to_string(decoded_bytes) +
      " decoded bytes; regenerate it with `rm -f xcvu3p.device && make "
      "xcvu3p.device`");
}

std::uint64_t edge_key(NodeId from, NodeId to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

// Validate signed offsets before writing them into unsigned binary headers.
std::uint64_t as_u64(std::int64_t value, const char* name) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " is negative");
  }
  return static_cast<std::uint64_t>(value);
}

void write_u64(std::ofstream& out, std::uint64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

void write_route_node(std::ofstream& out, NodeId node, const char* name) {
  const std::uint64_t encoded =
      node < 0 ? kNoIndex : static_cast<std::uint64_t>(node);
  write_u64(out, encoded, name);
}

// Arrays are written raw after explicit count fields in the header. The file
// format therefore depends on the fixed-width type checks near the top.
template <typename T>
void write_array(std::ofstream& out,
                 const std::vector<T>& values,
                 const char* name) {
  if (values.empty()) {
    return;
  }

  const std::size_t bytes = values.size() * sizeof(T);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

// Strings in the metadata sidecar are length-prefixed byte strings. All later
// metadata records refer to these strings by numeric index.
void write_string(std::ofstream& out, const std::string& text) {
  write_u64(out, static_cast<std::uint64_t>(text.size()), "string length");
  if (!text.empty()) {
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  if (!out) {
    throw std::runtime_error("failed while writing metadata string");
  }
}

// Resolve a PhysicalNetlist site pin back to the routing graph:
//   site name -> concrete tile/type info
//   tile type + site type + pin name -> tile wire name
//   tile name + tile wire name -> compact CSR node ID
std::optional<NodeId> get_node_from_site_pin(const RoutingGraph& graph,
                                             const std::string& site_name,
                                             const std::string& pin_name) {
  const auto site_it = graph.site2tile_and_types.find(site_name);
  if (site_it == graph.site2tile_and_types.end()) {
    return std::nullopt;
  }

  const TileAndTypes& tile_and_types = site_it->second;
  const auto tile_type_it =
      graph.tile_type2site_type_pin2wire.find(tile_and_types.tile_type);
  if (tile_type_it == graph.tile_type2site_type_pin2wire.end()) {
    return std::nullopt;
  }

  SitePinKey key;
  key.site_type = tile_and_types.site_type;
  key.pin_name = pin_name;

  const auto wire_it = tile_type_it->second.find(key);
  if (wire_it == tile_type_it->second.end()) {
    return std::nullopt;
  }

  const auto tile_it = graph.tile2wire2node.find(tile_and_types.tile_name);
  if (tile_it == graph.tile2wire2node.end()) {
    return std::nullopt;
  }

  const auto node_it = tile_it->second.find(wire_it->second);
  if (node_it == tile_it->second.end()) {
    return std::nullopt;
  }

  return node_it->second;
}

// Record the sink-site-pin node attribute once. The sidecar stores this as the
// equivalent of NetworkX's node attribute "sp".
void add_site_pin_attr(RoutingGraph& graph,
                       NodeId node,
                       std::uint64_t site_string,
                       std::uint64_t pin_string) {
  if (node < 0 ||
      static_cast<std::size_t>(node) >= graph.site_pin_attr_by_node.size()) {
    throw std::runtime_error("site pin node is outside graph");
  }

  std::int64_t& attr_index = graph.site_pin_attr_by_node[node];
  if (attr_index >= 0) {
    return;
  }

  attr_index = static_cast<std::int64_t>(graph.site_pin_attrs.size());
  SitePinNode attr;
  attr.node = node;
  attr.site_string = site_string;
  attr.pin_string = pin_string;
  graph.site_pin_attrs.push_back(attr);
}

// Walk a PhysicalNetlist route-branch forest and collect every sitePin segment.
// For unrouted signal nets, stubs hold sinks and sources hold legal starts.
std::vector<SitePinName> extract_site_pins(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches,
    TextCache& strings) {
  using RouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;

  std::vector<RouteBranch::Reader> queue;
  queue.reserve(branches.size());
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    queue.push_back(branches[i]);
  }

  std::vector<SitePinName> pins;
  while (!queue.empty()) {
    const RouteBranch::Reader branch = queue.back();
    queue.pop_back();

    const auto segment = branch.getRouteSegment();
    if (segment.isSitePin()) {
      const auto site_pin = segment.getSitePin();
      pins.push_back(
          {strings.get(site_pin.getSite()), strings.get(site_pin.getPin())});
    }

    const auto child_branches = branch.getBranches();
    for (std::uint32_t i = 0; i < child_branches.size(); ++i) {
      queue.push_back(child_branches[i]);
    }
  }

  return pins;
}

// Look up a graph node directly from a tile/wire pair. This is used when an
// already-routed PhysPIP names its driven wire and that node must be blocked.
std::optional<NodeId> find_tile_wire_node(const RoutingGraph& graph,
                                          const std::string& tile_name,
                                          const std::string& wire_name) {
  const auto tile_it = graph.tile2wire2node.find(tile_name);
  if (tile_it == graph.tile2wire2node.end()) {
    return std::nullopt;
  }

  const auto wire_it = tile_it->second.find(wire_name);
  if (wire_it == tile_it->second.end()) {
    return std::nullopt;
  }

  return wire_it->second;
}

// Convert Cap'n Proto text into an owned string. strList entries use TextCache,
// while root names like LogicalNetlist.name are direct Text fields.
std::string capnp_text_to_string(capnp::Text::Reader text) {
  return std::string(text.cStr(), text.size());
}

void parse_logical_netlist(const std::filesystem::path& logical_path,
                           RoutingGraph& graph) {
  using Netlist = LogicalNetlist::Netlist;

  // LogicalNetlist is design connectivity: logical cells, nets, ports, and
  // port instances. It does not contain routing wires or PIPs, but preserving
  // it lets later tools relate physical route results back to logical nets.
  const std::vector<std::uint8_t> bytes =
      read_gzip_or_plain_file(logical_path);
  graph.logical_netlist_bytes = bytes;
  std::vector<capnp::word> words = bytes_to_words(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(words.data(), words.size()), reader_options);
  const auto netlist = reader.getRoot<Netlist>();
  TextCache strings(netlist.getStrList());

  // Store the top-level logical design name in the shared metadata string
  // table. This helps a routed-.phys reconstruction tool sanity-check that it
  // is using metadata from the intended design.
  graph.logical_design_name_string =
      graph.string_table.intern(capnp_text_to_string(netlist.getName()));

  const auto port_list = netlist.getPortList();
  const auto cell_decls = netlist.getCellDecls();
  const auto inst_list = netlist.getInstList();
  const auto cell_list = netlist.getCellList();

  // Walk every logical cell. Each cell summary stores a slice into the flat
  // logical_nets array, keeping metadata compact and easy to stream from disk.
  graph.logical_cells.reserve(cell_list.size());
  for (std::uint32_t cell_index = 0; cell_index < cell_list.size();
       ++cell_index) {
    const auto cell = cell_list[cell_index];
    const std::uint32_t declaration_index = cell.getIndex();

    LogicalCellSummary cell_summary;
    cell_summary.net_begin =
        static_cast<std::uint64_t>(graph.logical_nets.size());
    if (declaration_index < cell_decls.size()) {
      const auto declaration = cell_decls[declaration_index];
      cell_summary.declaration_name_string =
          graph.string_table.intern(strings.get(declaration.getName()));
    }

    // Walk every logical net in this cell. Physical net names usually match
    // logical net names; the name index map below lets route requests point
    // back to these logical net summaries.
    const auto nets = cell.getNets();
    for (std::uint32_t net_index = 0; net_index < nets.size(); ++net_index) {
      const auto net = nets[net_index];
      const std::string& net_name = strings.get(net.getName());

      LogicalNetSummary net_summary;
      net_summary.name_string = graph.string_table.intern(net_name);
      net_summary.cell_index =
          static_cast<std::uint64_t>(graph.logical_cells.size());
      net_summary.port_instance_begin =
          static_cast<std::uint64_t>(graph.logical_port_instances.size());

      // Walk each port instance on the logical net. This captures which cell
      // instance or top-level port participates in the net and which port/bus
      // bit it uses.
      const auto port_insts = net.getPortInsts();
      for (std::uint32_t port_inst_index = 0;
           port_inst_index < port_insts.size();
           ++port_inst_index) {
        const auto port_inst = port_insts[port_inst_index];

        LogicalPortInstanceSummary port_summary;
        port_summary.port_index = port_inst.getPort();

        if (port_summary.port_index < port_list.size()) {
          const auto port = port_list[port_summary.port_index];
          port_summary.port_string =
              graph.string_table.intern(strings.get(port.getName()));
        }

        const auto bus_idx = port_inst.getBusIdx();
        if (bus_idx.isIdx()) {
          port_summary.has_bus_index = true;
          port_summary.bus_index = bus_idx.getIdx();
        }

        if (port_inst.isInst()) {
          port_summary.instance_index = port_inst.getInst();
          if (port_summary.instance_index < inst_list.size()) {
            const auto inst = inst_list[port_summary.instance_index];
            port_summary.instance_string =
                graph.string_table.intern(strings.get(inst.getName()));
          }
        } else {
          port_summary.is_external_port = true;
        }

        graph.logical_port_instances.push_back(port_summary);
      }

      net_summary.port_instance_count =
          static_cast<std::uint64_t>(graph.logical_port_instances.size()) -
          net_summary.port_instance_begin;

      const std::uint64_t logical_net_index =
          static_cast<std::uint64_t>(graph.logical_nets.size());
      graph.logical_nets.push_back(net_summary);

      // If duplicate logical net names exist in different scopes, keep the
      // first match for the simple physical-net-name bridge and still serialize
      // every logical net summary in full.
      graph.logical_net_index_by_name.emplace(net_name, logical_net_index);
    }

    cell_summary.net_count =
        static_cast<std::uint64_t>(graph.logical_nets.size()) -
        cell_summary.net_begin;
    graph.logical_cells.push_back(cell_summary);
  }
}

void build_device_routing_graph(const std::filesystem::path& device_path,
                                const Bounds& bounds,
                                NodeBoundsMode node_bounds_mode,
                                RoutingGraph& graph) {
  // Load the whole DeviceResources message into memory, matching the Python
  // proof of concept. DeviceResources is the fabric database: tiles, wires,
  // routing nodes, tile types, PIPs, site types, and site instances.
  const std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(device_path);
  check_device_payload_size(device_path, bytes.size());
  std::vector<capnp::word> words = bytes_to_words(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(words.data(), words.size()), reader_options);
  const auto device = reader.getRoot<DeviceResources::Device>();
  TextCache strings(device.getStrList());

  const auto tile_list = device.getTileList();
  const auto tile_types = device.getTileTypeList();
  const auto wires = device.getWires();
  const auto nodes = device.getNodes();

  // First pass over tileList: keep only tiles whose names carry X/Y coordinates
  // inside the selected bounds. Store the FPGAIF StringIdx values exactly like
  // nxroute-poc.py, which compares Wire.tile StringIdx against Tile.name
  // StringIdx.
  std::unordered_set<std::uint32_t> in_bounds_tile_name_ids;
  std::vector<std::uint32_t> in_bounds_tile_indices;

  for (std::uint32_t tile_index = 0; tile_index < tile_list.size();
       ++tile_index) {
    const auto tile = tile_list[tile_index];
    const std::string& tile_name = strings.get(tile.getName());

    const std::optional<std::pair<int, int>> xy = parse_tile_xy(tile_name);
    if (!xy.has_value()) {
      continue;
    }

    const int x = xy->first;
    const int y = xy->second;
    if (x < bounds.min_x || x > bounds.max_x || y < bounds.min_y ||
        y > bounds.max_y) {
      continue;
    }

    in_bounds_tile_name_ids.insert(tile.getName());
    in_bounds_tile_indices.push_back(tile_index);
  }

  if (nodes.size() > static_cast<std::uint64_t>(
                         std::numeric_limits<NodeId>::max())) {
    throw std::runtime_error("device has too many nodes for 32-bit CSR IDs");
  }

  // Build CSR/compact node IDs from DeviceResources.nodes. FPGAIF already
  // groups electrically-equivalent wires into routing nodes. The default
  // node-bounds mode is the POC rule:
  //   baseWireIdx = node.wires[0]
  //   include node iff wires[baseWireIdx].tile is in tileNames
  // Other node-bounds modes are kept as options for experiments, but they do
  // not change how metadata is stored once a node is imported.
  for (std::uint32_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const auto node = nodes[node_index];
    const auto node_wires = node.getWires();
    if (node_wires.size() == 0) {
      continue;
    }

    const auto base_wire = wires[node_wires[0]];
    const bool base_wire_in_bounds =
        in_bounds_tile_name_ids.find(base_wire.getTile()) !=
        in_bounds_tile_name_ids.end();

    bool any_wire_in_bounds = false;
    bool all_wires_in_bounds = true;
    for (std::uint32_t wire_offset = 0; wire_offset < node_wires.size();
         ++wire_offset) {
      const auto wire = wires[node_wires[wire_offset]];
      const bool wire_in_bounds =
          in_bounds_tile_name_ids.find(wire.getTile()) !=
          in_bounds_tile_name_ids.end();
      any_wire_in_bounds = any_wire_in_bounds || wire_in_bounds;
      all_wires_in_bounds = all_wires_in_bounds && wire_in_bounds;
    }

    bool include_node = false;
    switch (node_bounds_mode) {
      case NodeBoundsMode::kPocBaseWire:
        include_node = base_wire_in_bounds;
        break;
      case NodeBoundsMode::kFullyContained:
        include_node = all_wires_in_bounds;
        break;
      case NodeBoundsMode::kIntersects:
        include_node = any_wire_in_bounds;
        break;
    }

    if (!include_node) {
      continue;
    }

    const NodeId compact_node =
        static_cast<NodeId>(graph.node_device_ids.size());
    graph.node_device_ids.push_back(node_index);

    for (std::uint32_t wire_offset = 0; wire_offset < node_wires.size();
         ++wire_offset) {
      const auto wire = wires[node_wires[wire_offset]];
      const std::string& tile_name = strings.get(wire.getTile());
      const std::string& wire_name = strings.get(wire.getWire());
      graph.tile2wire2node[tile_name][wire_name] = compact_node;
    }
  }

  if (graph.node_device_ids.empty()) {
    throw std::runtime_error("no routing nodes were imported from device");
  }

  graph.blocked_node.assign(graph.node_device_ids.size(), 0);
  graph.sink_node_stops.assign(graph.node_device_ids.size(), 0);
  graph.site_pin_attr_by_node.assign(graph.node_device_ids.size(), -1);

  // Edge deduplication is by compact node pair. If two PIPs produce the same
  // source/destination pair, the latest attribute is kept, which mirrors the
  // "single pip attribute per edge" simplification in the NetworkX example.
  std::unordered_map<std::uint64_t, std::size_t> edge_index_by_pair;
  edge_index_by_pair.reserve(graph.node_device_ids.size() * 4);

  // PIP metadata is stored once per unique (wire0, wire1, forward) tuple.
  // Edges then point at this table through pip_data_index.
  std::unordered_map<std::string, std::uint64_t> pip_data_index_by_key;

  auto intern_pip_data = [&](const std::string& wire0_name,
                             const std::string& wire1_name,
                             bool forward) -> std::uint64_t {
    std::string key;
    key.reserve(wire0_name.size() + wire1_name.size() + 3);
    key.append(wire0_name);
    key.push_back('\x1f');
    key.append(wire1_name);
    key.push_back('\x1f');
    key.push_back(forward ? '1' : '0');

    const auto found = pip_data_index_by_key.find(key);
    if (found != pip_data_index_by_key.end()) {
      return found->second;
    }

    PipData data;
    data.wire0_string = graph.string_table.intern(wire0_name);
    data.wire1_string = graph.string_table.intern(wire1_name);
    data.forward = forward;

    const std::uint64_t index =
        static_cast<std::uint64_t>(graph.pip_data.size());
    graph.pip_data.push_back(data);
    pip_data_index_by_key.emplace(std::move(key), index);
    return index;
  };

  auto add_or_update_edge = [&](NodeId from, NodeId to, EdgeAttr attr) {
    if (from < 0 || to < 0) {
      return;
    }

    ++graph.raw_edges_seen;
    if (from == to) {
      return;
    }

    const std::uint64_t key = edge_key(from, to);
    const auto found = edge_index_by_pair.find(key);
    if (found != edge_index_by_pair.end()) {
      graph.edges[found->second].attr = attr;
      return;
    }

    BuildEdge edge;
    edge.from = from;
    edge.to = to;
    edge.attr = attr;
    edge_index_by_pair.emplace(key, graph.edges.size());
    graph.edges.push_back(edge);
  };

  // Build graph edges from tile type PIPs. A tile instance chooses a tile type;
  // the tile type lists possible PIPs by wire index within that tile type.
  // Those wire names are mapped through tile2wire2node to compact graph nodes.
  for (const std::uint32_t tile_index : in_bounds_tile_indices) {
    const auto tile = tile_list[tile_index];
    const std::string& tile_name = strings.get(tile.getName());
    const auto wire2node_it = graph.tile2wire2node.find(tile_name);
    if (wire2node_it == graph.tile2wire2node.end()) {
      continue;
    }

    const bool is_cle_or_rclk_tile =
        has_prefix(tile_name, "CLE") || has_prefix(tile_name, "RCLK");
    const auto tile_type = tile_types[tile.getType()];
    const auto tile_wires = tile_type.getWires();
    const auto pips = tile_type.getPips();

    for (std::uint32_t pip_index = 0; pip_index < pips.size(); ++pip_index) {
      const auto pip = pips[pip_index];
      if (is_cle_or_rclk_tile && !pip.isConventional()) {
        continue;
      }

      const std::string& wire0_name = strings.get(tile_wires[pip.getWire0()]);
      const std::string& wire1_name = strings.get(tile_wires[pip.getWire1()]);

      const auto node0_it = wire2node_it->second.find(wire0_name);
      if (node0_it == wire2node_it->second.end()) {
        continue;
      }

      const auto node1_it = wire2node_it->second.find(wire1_name);
      if (node1_it == wire2node_it->second.end()) {
        continue;
      }

      EdgeAttr forward_attr;
      forward_attr.tile_string = graph.string_table.intern(tile_name);
      forward_attr.pip_data_index =
          intern_pip_data(wire0_name, wire1_name, true);
      add_or_update_edge(node0_it->second, node1_it->second, forward_attr);

      if (!pip.getDirectional()) {
        EdgeAttr reverse_attr;
        reverse_attr.tile_string = forward_attr.tile_string;
        reverse_attr.pip_data_index =
            intern_pip_data(wire0_name, wire1_name, false);
        add_or_update_edge(node1_it->second, node0_it->second, reverse_attr);
      }
    }
  }

  // Build site type pin-name tables so site pin indexes from siteTypeList can
  // later be paired with primaryPinsToTileWires in tileTypeList.
  const auto site_type_list = device.getSiteTypeList();
  std::vector<std::vector<std::string>> site_type_pin_names(
      site_type_list.size());

  for (std::uint32_t site_type_index = 0;
       site_type_index < site_type_list.size();
       ++site_type_index) {
    const auto site_type = site_type_list[site_type_index];
    const auto pins = site_type.getPins();
    auto& pin_names = site_type_pin_names[site_type_index];
    pin_names.reserve(pins.size());
    for (std::uint32_t pin_index = 0; pin_index < pins.size(); ++pin_index) {
      pin_names.push_back(strings.get(pins[pin_index].getName()));
    }
  }

  // Build tile type local mapping:
  //   tile type -> (site type in tile type, pin name) -> tile wire name.
  // This is the first half of translating a PhysicalNetlist site pin into a
  // routing graph node.
  for (std::uint32_t tile_type_index = 0; tile_type_index < tile_types.size();
       ++tile_type_index) {
    const auto tile_type = tile_types[tile_type_index];
    const auto site_types = tile_type.getSiteTypes();
    auto& pin_to_wire = graph.tile_type2site_type_pin2wire[tile_type_index];

    for (std::uint32_t site_type_index = 0;
         site_type_index < site_types.size();
         ++site_type_index) {
      const auto site_type = site_types[site_type_index];
      const std::uint32_t primary_type = site_type.getPrimaryType();
      if (primary_type >= site_type_pin_names.size()) {
        continue;
      }

      const auto primary_pins_to_tile_wires =
          site_type.getPrimaryPinsToTileWires();
      const auto& pin_names = site_type_pin_names[primary_type];
      const std::uint32_t pin_count = std::min<std::uint32_t>(
          primary_pins_to_tile_wires.size(), pin_names.size());

      for (std::uint32_t pin_index = 0; pin_index < pin_count; ++pin_index) {
        SitePinKey key;
        key.site_type = site_type_index;
        key.pin_name = pin_names[pin_index];
        pin_to_wire.emplace(std::move(key),
                            strings.get(primary_pins_to_tile_wires[pin_index]));
      }
    }
  }

  // Build concrete site-instance mapping:
  //   site name -> tile name, tile type, site type.
  // Combined with the tile-type pin map above, this completes sitePin -> node.
  for (const std::uint32_t tile_index : in_bounds_tile_indices) {
    const auto tile = tile_list[tile_index];
    const std::string& tile_name = strings.get(tile.getName());
    const auto sites = tile.getSites();

    for (std::uint32_t site_index = 0; site_index < sites.size();
         ++site_index) {
      const auto site = sites[site_index];
      TileAndTypes tile_and_types;
      tile_and_types.tile_name = tile_name;
      tile_and_types.tile_type = tile.getType();
      tile_and_types.site_type = site.getType();
      graph.site2tile_and_types[strings.get(site.getName())] =
          std::move(tile_and_types);
    }
  }
}

void parse_physical_netlist(const std::filesystem::path& phys_path,
                            RoutingGraph& graph) {
  using PhysNetlist = PhysicalNetlist::PhysNetlist;
  using RouteBranch = PhysNetlist::RouteBranch;

  // PhysicalNetlist is design-specific. It supplies unrouted signal stubs,
  // legal source site pins, and already-occupied routing from fixed or
  // pre-routed nets.
  const std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(phys_path);
  graph.physical_netlist_bytes = bytes;
  std::vector<capnp::word> words = bytes_to_words(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(words.data(), words.size()), reader_options);
  const auto netlist = reader.getRoot<PhysNetlist>();
  TextCache strings(netlist.getStrList());

  const auto phys_nets = netlist.getPhysNets();
  for (std::uint32_t net_index = 0; net_index < phys_nets.size();
       ++net_index) {
    const auto net = phys_nets[net_index];

    // Signal nets with stubs are the nets still needing routing. Stubs are
    // interpreted as sink site pins; sources are interpreted as start pins.
    const bool is_signal = net.getType() == PhysNetlist::NetType::SIGNAL;
    if (is_signal && net.getStubs().size() > 0) {
      const std::vector<SitePinName> sink_pins =
          extract_site_pins(net.getStubs(), strings);
      if (sink_pins.empty()) {
        continue;
      }

      RouteRequest request;
      const std::string& physical_net_name = strings.get(net.getName());
      request.net_string = graph.string_table.intern(physical_net_name);
      const auto logical_match =
          graph.logical_net_index_by_name.find(physical_net_name);
      if (logical_match != graph.logical_net_index_by_name.end()) {
        request.logical_net_index = logical_match->second;
      }

      // Map each source site pin to the routing node that drives/enters the
      // inter-site routing fabric.
      const std::vector<SitePinName> source_pins =
          extract_site_pins(net.getSources(), strings);
      bool has_valid_source = false;
      for (const SitePinName& source_pin : source_pins) {
        SitePinNode source;
        source.node = kInvalidRouteNode;
        source.site_string = graph.string_table.intern(source_pin.site);
        source.pin_string = graph.string_table.intern(source_pin.pin);

        const std::optional<NodeId> source_node =
            get_node_from_site_pin(graph, source_pin.site, source_pin.pin);
        if (source_node.has_value()) {
          source.node = *source_node;
          has_valid_source = true;
        }

        request.sources.push_back(source);
      }

      // Preserve every physical sink stub in the request. If bounds exclude
      // its routing node, keep the site pin with an invalid node so routing
      // reports the net as unreached instead of emitting a partial route.
      for (const SitePinName& sink_pin : sink_pins) {
        SitePinNode sink;
        sink.node = kInvalidRouteNode;
        sink.site_string = graph.string_table.intern(sink_pin.site);
        sink.pin_string = graph.string_table.intern(sink_pin.pin);

        const std::optional<NodeId> sink_node =
            get_node_from_site_pin(graph, sink_pin.site, sink_pin.pin);
        if (sink_node.has_value()) {
          const NodeId node = *sink_node;
          sink.node = node;
          if (!has_valid_source) {
            graph.blocked_node[node] = 1;
          } else {
            graph.sink_node_stops[node] = 1;
            add_site_pin_attr(graph, node, sink.site_string, sink.pin_string);
          }
        }
        request.sinks.push_back(sink);
      }

      if (!request.sinks.empty()) {
        graph.route_requests.push_back(std::move(request));
      }
      continue;
    }

    // Nets without stubs are already routed, and non-signal nets are treated
    // as fixed resources. RapidWright's PhysNetlistReader imports both source
    // branches and stub branches, then adds explicit stubNodes as null-end
    // PIPs. Mirror that occupancy here by blocking the driven node of every
    // PhysPIP we can resolve, plus every explicit stub node.
    std::vector<RouteBranch::Reader> queue;
    const auto sources = net.getSources();
    const auto stubs = net.getStubs();
    queue.reserve(sources.size() + stubs.size());
    for (std::uint32_t i = 0; i < sources.size(); ++i) {
      queue.push_back(sources[i]);
    }
    for (std::uint32_t i = 0; i < stubs.size(); ++i) {
      queue.push_back(stubs[i]);
    }

    while (!queue.empty()) {
      const RouteBranch::Reader branch = queue.back();
      queue.pop_back();

      const auto segment = branch.getRouteSegment();
      if (segment.isPip()) {
        const auto pip = segment.getPip();
        const std::string& tile_name = strings.get(pip.getTile());
        const std::string& driven_wire =
            strings.get(pip.getForward() ? pip.getWire1() : pip.getWire0());
        const std::optional<NodeId> blocked =
            find_tile_wire_node(graph, tile_name, driven_wire);
        if (blocked.has_value()) {
          graph.blocked_node[*blocked] = 1;
        }
      }

      const auto child_branches = branch.getBranches();
      for (std::uint32_t i = 0; i < child_branches.size(); ++i) {
        queue.push_back(child_branches[i]);
      }
    }

    const auto stub_nodes = net.getStubNodes();
    for (std::uint32_t i = 0; i < stub_nodes.size(); ++i) {
      const auto stub_node = stub_nodes[i];
      const std::string& tile_name = strings.get(stub_node.getTile());
      const std::string& wire_name = strings.get(stub_node.getWire());
      const std::optional<NodeId> blocked =
          find_tile_wire_node(graph, tile_name, wire_name);
      if (blocked.has_value()) {
        graph.blocked_node[*blocked] = 1;
      }
    }
  }
}

struct CsrEntry {
  std::int32_t col = 0;
  float value = 1.0f;
  EdgeAttr attr;
};

// Convert the natural directed routing graph into Bellman-Ford CSR:
//   incoming row v contains columns u for every routing edge u -> v.
// Blocked nodes and sink-stop source nodes are filtered out here, so the GPU
// graph never sees those edges.
CsrGraph make_incoming_csr(const RoutingGraph& graph) {
  CsrGraph csr;
  csr.rows = static_cast<std::int64_t>(graph.node_device_ids.size());
  csr.cols = csr.rows;
  csr.declared_edges = graph.raw_edges_seen;
  csr.loaded_edges = static_cast<std::uint64_t>(graph.edges.size());
  csr.rowptr.resize(static_cast<std::size_t>(csr.rows) + 1);

  auto edge_is_importable = [&](const BuildEdge& edge) {
    if (edge.from < 0 || edge.to < 0 || edge.from >= csr.rows ||
        edge.to >= csr.rows) {
      return false;
    }

    if (graph.blocked_node[edge.from] || graph.blocked_node[edge.to]) {
      return false;
    }

    if (graph.sink_node_stops[edge.from]) {
      return false;
    }
    return true;
  };

  for (const BuildEdge& edge : graph.edges) {
    if (edge_is_importable(edge)) {
      ++csr.rowptr[static_cast<std::size_t>(edge.to) + 1];
    }
  }

  for (std::int64_t row = 0; row < csr.rows; ++row) {
    csr.rowptr[static_cast<std::size_t>(row + 1)] +=
        csr.rowptr[static_cast<std::size_t>(row)];
  }

  std::vector<CsrEntry> entries(static_cast<std::size_t>(csr.rowptr.back()));
  std::vector<std::int64_t> cursor = csr.rowptr;
  for (const BuildEdge& edge : graph.edges) {
    if (!edge_is_importable(edge)) {
      continue;
    }

    const std::size_t row = static_cast<std::size_t>(edge.to);
    const std::size_t pos = static_cast<std::size_t>(cursor[row]++);
    entries[pos] = {edge.from, 1.0f, edge.attr};
  }

  csr.colind.resize(entries.size());
  csr.values.resize(entries.size());
  csr.edge_attrs.resize(entries.size());
  for (std::int64_t row = 0; row < csr.rows; ++row) {
    const std::size_t begin =
        static_cast<std::size_t>(csr.rowptr[static_cast<std::size_t>(row)]);
    const std::size_t end =
        static_cast<std::size_t>(csr.rowptr[static_cast<std::size_t>(row + 1)]);
    std::sort(entries.begin() + static_cast<std::ptrdiff_t>(begin),
              entries.begin() + static_cast<std::ptrdiff_t>(end),
              [](const CsrEntry& lhs, const CsrEntry& rhs) {
                return lhs.col < rhs.col;
              });

    for (std::size_t index = begin; index < end; ++index) {
      const CsrEntry& entry = entries[index];
      csr.colind[index] = entry.col;
      csr.values[index] = entry.value;
      csr.edge_attrs[index] = entry.attr;
    }
  }
  return csr;
}


void write_csr_graph(const CsrGraph& graph,
                     const std::filesystem::path& output_path) {
  // This binary file intentionally contains only the generic graph structure:
  // rowptr, colind, and unit weights. FPGA routing names and PIP details live
  // in the RIPSIFM1 metadata sidecar.
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open output file: " +
                             output_path.string());
  }

  out.write(CSR_MAGIC, sizeof(CSR_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing CSR magic");
  }

  const std::uint64_t nnz = static_cast<std::uint64_t>(graph.values.size());
  write_u64(out, CSR_FORMAT_VERSION, "format version");
  write_u64(out, INCOMING_EDGE_ORIENTATION, "orientation");
  write_u64(out, as_u64(graph.rows, "rows"), "row count");
  write_u64(out, as_u64(graph.cols, "cols"), "column count");
  write_u64(out, graph.declared_edges, "declared edge count");
  write_u64(out, graph.loaded_edges, "loaded edge count");
  write_u64(out, nnz, "nnz");
  write_u64(out, static_cast<std::uint64_t>(graph.rowptr.size()),
            "rowptr count");
  write_u64(out, static_cast<std::uint64_t>(graph.colind.size()),
            "colind count");
  write_u64(out, static_cast<std::uint64_t>(graph.values.size()),
            "values count");

  write_array(out, graph.rowptr, "rowptr");
  write_array(out, graph.colind, "colind");
  write_array(out, graph.values, "values");
}

void write_metadata(const RoutingGraph& graph,
                    const CsrGraph& csr,
                    const std::filesystem::path& metadata_path) {
  // RIPSIFM1 sidecar layout:
  //   char[8] magic
  //   u64 version, orientation
  //   u64 string_count, node_count, edge_attr_count, pip_data_count
  //   u64 site_pin_attr_count, route_request_count
  //   u64 blocked_node_count, sink_stop_node_count
  //   u64 logical_cell_count, logical_net_count, logical_port_instance_count
  //   u64 physical_netlist_byte_count, logical_netlist_byte_count
  //   u64 device_path_string, physical_path_string, logical_path_string
  //   u64 logical_design_name_string
  //   repeated strings: u64 byte_length, bytes
  //   u64[node_count] original DeviceResources node ids
  //   edge_attr_count records: u64 tile_string, u64 pip_data_index
  //   pip_data_count records: u64 wire0_string, u64 wire1_string, u64 forward
  //   site_pin_attr_count records: u64 node, u64 site_string, u64 pin_string
  //   route requests with logical net index and variable source/sink records
  //   logical cell/net/port-instance summary records
  //   u64[blocked_node_count], u64[sink_stop_node_count]
  //   physical_netlist_byte_count raw bytes from decompressed .phys
  //   logical_netlist_byte_count raw bytes from decompressed .netlist
  if (metadata_path.has_parent_path()) {
    std::filesystem::create_directories(metadata_path.parent_path());
  }

  std::ofstream out(metadata_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open metadata file: " +
                             metadata_path.string());
  }

  std::vector<std::uint64_t> blocked_nodes;
  std::vector<std::uint64_t> sink_stop_nodes;
  // Store node-level masks separately from CSR. blocked_nodes were removed
  // from the CSR graph; sink_stop_nodes identify site-pin targets whose
  // outgoing edges were suppressed.
  for (std::size_t node = 0; node < graph.node_device_ids.size(); ++node) {
    if (graph.blocked_node[node]) {
      blocked_nodes.push_back(static_cast<std::uint64_t>(node));
    }
    if (graph.sink_node_stops[node]) {
      sink_stop_nodes.push_back(static_cast<std::uint64_t>(node));
    }
  }

  out.write(METADATA_MAGIC, sizeof(METADATA_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing metadata magic");
  }

  write_u64(out, METADATA_FORMAT_VERSION, "metadata format version");
  write_u64(out, INCOMING_EDGE_ORIENTATION, "metadata orientation");
  write_u64(out, static_cast<std::uint64_t>(graph.string_table.strings.size()),
            "string count");
  write_u64(out, static_cast<std::uint64_t>(graph.node_device_ids.size()),
            "node count");
  write_u64(out, static_cast<std::uint64_t>(csr.edge_attrs.size()),
            "edge attribute count");
  write_u64(out, static_cast<std::uint64_t>(graph.pip_data.size()),
            "pip data count");
  write_u64(out, static_cast<std::uint64_t>(graph.site_pin_attrs.size()),
            "site pin attr count");
  write_u64(out, static_cast<std::uint64_t>(graph.route_requests.size()),
            "route request count");
  write_u64(out, static_cast<std::uint64_t>(blocked_nodes.size()),
            "blocked node count");
  write_u64(out, static_cast<std::uint64_t>(sink_stop_nodes.size()),
            "sink stop node count");
  write_u64(out, static_cast<std::uint64_t>(graph.logical_cells.size()),
            "logical cell count");
  write_u64(out, static_cast<std::uint64_t>(graph.logical_nets.size()),
            "logical net count");
  write_u64(out,
            static_cast<std::uint64_t>(
                graph.logical_port_instances.size()),
            "logical port instance count");
  write_u64(out,
            static_cast<std::uint64_t>(graph.physical_netlist_bytes.size()),
            "physical netlist byte count");
  write_u64(out,
            static_cast<std::uint64_t>(graph.logical_netlist_bytes.size()),
            "logical netlist byte count");
  write_u64(out, graph.device_path_string, "device path string");
  write_u64(out, graph.physical_path_string, "physical path string");
  write_u64(out, graph.logical_path_string, "logical path string");
  write_u64(out, graph.logical_design_name_string,
            "logical design name string");

  for (const std::string& text : graph.string_table.strings) {
    write_string(out, text);
  }

  // For each compact CSR node, preserve the original DeviceResources node
  // index. This gives later code a bridge back to FPGAIF device data.
  write_array(out, graph.node_device_ids, "device node ids");

  // Edge attributes are aligned exactly with CSR colind/values order. For edge
  // k, csr.colind[k], csr.values[k], and edge_attrs[k] describe one PIP edge.
  for (const EdgeAttr& attr : csr.edge_attrs) {
    write_u64(out, attr.tile_string, "edge tile string");
    write_u64(out, attr.pip_data_index, "edge pip data index");
  }

  // PIP data table stores the wire pair and direction referenced by each edge
  // attribute. tile name lives on EdgeAttr because the same wire pair appears
  // in many tile instances.
  for (const PipData& data : graph.pip_data) {
    write_u64(out, data.wire0_string, "pip wire0 string");
    write_u64(out, data.wire1_string, "pip wire1 string");
    write_u64(out, data.forward ? 1 : 0, "pip forward flag");
  }

  // Sink site-pin node attributes, matching NetworkX's node attribute "sp".
  for (const SitePinNode& attr : graph.site_pin_attrs) {
    write_u64(out, static_cast<std::uint64_t>(attr.node),
              "site pin attr node");
    write_u64(out, attr.site_string, "site pin attr site string");
    write_u64(out, attr.pin_string, "site pin attr pin string");
  }

  // Route requests preserve net -> source nodes and sink nodes. A future
  // router can run shortest paths/Bellman-Ford over the CSR using these node
  // IDs, then recover PIPs through edge_attrs and pip_data. logical_net_index
  // links the physical route request back to LogicalNetlist metadata when a
  // net-name match was available.
  for (const RouteRequest& request : graph.route_requests) {
    write_u64(out, request.net_string, "route request net string");
    write_u64(out, request.logical_net_index,
              "route request logical net index");
    write_u64(out, static_cast<std::uint64_t>(request.sources.size()),
              "route request source count");
    for (const SitePinNode& source : request.sources) {
      write_route_node(out, source.node, "route request source node");
      write_u64(out, source.site_string, "route request source site");
      write_u64(out, source.pin_string, "route request source pin");
    }

    write_u64(out, static_cast<std::uint64_t>(request.sinks.size()),
              "route request sink count");
    for (const SitePinNode& sink : request.sinks) {
      write_route_node(out, sink.node, "route request sink node");
      write_u64(out, sink.site_string, "route request sink site");
      write_u64(out, sink.pin_string, "route request sink pin");
    }
  }

  // Logical cell summaries are slices into the flat logical_nets array.
  for (const LogicalCellSummary& cell : graph.logical_cells) {
    write_u64(out, cell.declaration_name_string,
              "logical cell declaration name");
    write_u64(out, cell.net_begin, "logical cell net begin");
    write_u64(out, cell.net_count, "logical cell net count");
  }

  // Logical net summaries are slices into the flat logical_port_instances
  // array. These records preserve logical connectivity without bloating CSR.
  for (const LogicalNetSummary& net : graph.logical_nets) {
    write_u64(out, net.name_string, "logical net name");
    write_u64(out, net.cell_index, "logical net cell index");
    write_u64(out, net.port_instance_begin,
              "logical net port instance begin");
    write_u64(out, net.port_instance_count,
              "logical net port instance count");
  }

  // Logical port-instance summaries say which port and instance/top-level port
  // participates in each logical net.
  for (const LogicalPortInstanceSummary& port :
       graph.logical_port_instances) {
    write_u64(out, port.port_string, "logical port name");
    write_u64(out, port.instance_string, "logical instance name");
    write_u64(out, port.port_index, "logical port index");
    write_u64(out, port.instance_index, "logical instance index");
    write_u64(out, port.bus_index, "logical bus index");
    write_u64(out, port.has_bus_index ? 1 : 0,
              "logical has bus index");
    write_u64(out, port.is_external_port ? 1 : 0,
              "logical is external port");
  }

  // Write the node masks last because they are auxiliary metadata rather than
  // per-edge routing information.
  write_array(out, blocked_nodes, "blocked nodes");
  write_array(out, sink_stop_nodes, "sink stop nodes");

  // Preserve original decompressed FPGAIF messages. A future writer can load
  // the physical bytes as a PhysNetlist builder, insert routed PhysPIP
  // branches derived from CSR paths, and then serialize a routed .phys. The
  // logical bytes provide the matching LogicalNetlist for validation or net
  // correlation during that process.
  write_array(out, graph.physical_netlist_bytes, "physical netlist bytes");
  write_array(out, graph.logical_netlist_bytes, "logical netlist bytes");
}

double mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Parse inputs first so the rest of main reads as the converter pipeline:
    // DeviceResources -> LogicalNetlist -> PhysicalNetlist -> CSR + metadata
    // sidecar.
    const Options options = parse_options(argc, argv);

    std::cout << "device: " << options.device_path << "\n";
    std::cout << "physical_netlist: " << options.phys_path << "\n";
    std::cout << "logical_netlist: " << options.logical_path << "\n";
    std::cout << "bounds: X" << options.bounds.min_x << "..X"
              << options.bounds.max_x << ", Y" << options.bounds.min_y
              << "..Y" << options.bounds.max_y << "\n";
    std::cout << "node_bounds_mode: "
              << node_bounds_mode_name(options.node_bounds_mode) << "\n";

    // Store input paths in the metadata string table. They are not used by the
    // GPU, but they make sidecar provenance and later routed-.phys generation
    // much easier to audit.
    RoutingGraph graph;
    graph.device_path_string =
        graph.string_table.intern(options.device_path.string());
    graph.physical_path_string =
        graph.string_table.intern(options.phys_path.string());
    graph.logical_path_string =
        graph.string_table.intern(options.logical_path.string());

    // DeviceResources parsing builds the full fabric graph and all static
    // lookup tables needed to translate site pins and PIP edges.
    build_device_routing_graph(options.device_path,
                               options.bounds,
                               options.node_bounds_mode,
                               graph);

    std::cout << "imported_nodes: " << graph.node_device_ids.size() << "\n";
    std::cout << "unique_edges: " << graph.edges.size() << "\n";

    // LogicalNetlist parsing records logical cells/nets/port instances and
    // builds a name index that physical route requests can reference.
    parse_logical_netlist(options.logical_path, graph);

    // PhysicalNetlist parsing adds design-specific route requests and blockage.
    // It also stores the original .phys payload so later code can patch routes
    // into that exact physical netlist structure.
    parse_physical_netlist(options.phys_path, graph);

    // CSR is the GPU-facing graph. Metadata is the CPU-facing FPGA context
    // needed to map CSR edges back to tile/wire PIPs and site-pin targets.
    CsrGraph csr = make_incoming_csr(graph);
    write_csr_graph(csr, options.output_path);
    write_metadata(graph, csr, options.metadata_path);

    const std::uint64_t rowptr_bytes =
        static_cast<std::uint64_t>(csr.rowptr.size() * sizeof(std::int64_t));
    const std::uint64_t colind_bytes =
        static_cast<std::uint64_t>(csr.colind.size() * sizeof(std::int32_t));
    const std::uint64_t values_bytes =
        static_cast<std::uint64_t>(csr.values.size() * sizeof(float));
    const std::uint64_t attr_bytes =
        static_cast<std::uint64_t>(csr.edge_attrs.size() * sizeof(EdgeAttr));
    const std::uint64_t csr_bytes = rowptr_bytes + colind_bytes + values_bytes;

    std::cout << "csr_rows: " << csr.rows << "\n";
    std::cout << "csr_nnz: " << csr.values.size() << "\n";
    std::cout << "csr_total_mib: " << mib(csr_bytes) << "\n";
    std::cout << "csr_rowptr_mib: " << mib(rowptr_bytes) << "\n";
    std::cout << "csr_colind_mib: " << mib(colind_bytes) << "\n";
    std::cout << "csr_values_mib: " << mib(values_bytes) << "\n";
    std::cout << "metadata_edge_attr_mib: " << mib(attr_bytes) << "\n";

    std::cout << "wrote_csr: " << options.output_path << "\n";
    std::cout << "wrote_metadata: " << options.metadata_path << "\n";

  } catch (const std::exception& ex) {
    if (argc < 2) {
      print_usage(argv[0]);
    }
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
