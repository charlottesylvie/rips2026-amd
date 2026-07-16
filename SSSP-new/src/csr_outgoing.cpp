// Converts FPGA Interchange DeviceResources plus a design's PhysicalNetlist
// and LogicalNetlist into the outgoing CSR binary format consumed by
// SSSP-new/src/mp_csr.cpp.
//
// The primary output is a RIPSOCS1 .outgoing.csrbin file:
//
//   CSR row u stores routing edges u -> v, with unit edge weight.
//
// The outgoing format also stores edge_id[k], the outgoing CSR edge id used as
// the packed predecessor in mp_csr.cpp. The RIPSIFM1 sidecar is outgoing-edge
// aligned: metadata edge record k describes outgoing CSR edge k.
//
//   edge attribute "pip" = (tileName, pipDataIndex)
//   pipData[pipDataIndex] = (wire0Name, wire1Name, forward)
//   node attribute "sp" = (siteName, pinName), for sink site pins
//
// The sidecar also records route requests extracted from PhysicalNetlist stubs
// and source site pins, logical net summaries extracted from LogicalNetlist,
// and the original decompressed .phys/.netlist payloads. That metadata is
// intentionally CPU-side: the GPU CSR remains compact and compatible with
// mp_csr.cpp, while later post-processing has enough context to turn SSSP paths
// back into PhysPIP route branches in a routed .phys file.
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
//     SSSP-new/src/csr_outgoing.cpp \
//     <generated-schema-dir>/DeviceResources.capnp.c++ \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     <generated-schema-dir>/LogicalNetlist.capnp.c++ \
//     <generated-schema-dir>/References.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o csr_outgoing
//
// Example use:
//
//   ./csr_outgoing xcvu3p.device benchmarks/vtr_mcml_unrouted.phys \
//     benchmarks/vtr_mcml.netlist --x-fraction 0.0 0.5 --y-fraction 0.0 1.0

#include "DeviceResources.capnp.h"
#include "LogicalNetlist.capnp.h"
#include "PhysicalNetlist.capnp.h"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Keep the on-disk binary formats explicit. The outgoing CSR constants match
// SSSP-new/src/bf1.cpp and mp_csr.cpp; the metadata constants identify the
// sidecar that stores FPGA-specific information not present in plain CSR.
static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

constexpr char OUTGOING_CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'O', 'C', 'S', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t OUTGOING_CSR_FORMAT_VERSION = 3;
// SSSP-new metadata version 3 is outgoing-edge aligned: edge_attrs[k]
// describes outgoing CSR edge k. Routing's older sidecar is incoming-oriented.
constexpr std::uint64_t METADATA_FORMAT_VERSION = 3;
constexpr std::uint64_t OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kPackedNoPredEdge = 0xffffffffULL;
constexpr std::uint64_t kNoIndex =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;
constexpr std::uint64_t kNoStringIndex = kNoIndex;

using NodeId = std::int32_t;

// Controls how the bounded tile box is applied to DeviceResources.nodes. The
// default for this converter is fully-contained so a region selection includes
// only nodes whose wires all live inside the selected tile-coordinate window.
enum class NodeBoundsMode {
  kPocBaseWire,
  kFullyContained,
  kIntersects,
};

// Absolute tile-coordinate bounds. The default covers every nonnegative X/Y
// coordinate; fraction selections are resolved to this form after the device's
// actual X/Y extent is known.
struct Bounds {
  int min_x = 0;
  int max_x = std::numeric_limits<int>::max();
  int min_y = 0;
  int max_y = std::numeric_limits<int>::max();
};

struct FractionBounds {
  double min_x = 0.0;
  double max_x = 1.0;
  double min_y = 0.0;
  double max_y = 1.0;
};

enum class BoundsSelectionMode {
  kFullDevice,
  kAbsoluteBounds,
  kFractionBounds,
};

struct TileCoord {
  std::uint32_t tile_index = 0;
  std::uint32_t tile_name_id = 0;
  int x = 0;
  int y = 0;
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
// direction. The final CSR is written in outgoing-edge orientation.
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

// Internal incoming CSR payload used only as a simple transpose source when
// building the final outgoing CSR.
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

// Outgoing CSR payload consumed by SSSP-new/src/mp_csr.cpp. For each outgoing
// edge k, edge_id[k] is also k; mp_csr packs that outgoing edge id as the
// predecessor edge.
struct OutgoingCsrGraph {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t nnz = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> degree;
  std::vector<std::int32_t> to;
  std::vector<float> values;
  std::vector<std::int64_t> edge_id;
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
  Bounds selected_bounds;
  std::uint64_t xy_tile_count = 0;
  std::uint64_t selected_tile_count = 0;
};

// Lightweight site-pin name pair used while walking PhysicalNetlist route
// branches before those names are mapped into graph nodes.
struct SitePinName {
  std::string site;
  std::string pin;
};

// Command-line options: PhysicalNetlist and LogicalNetlist inputs are required.
// DeviceResources is still required for topology, but it defaults to
// xcvu3p.device or can be supplied by --device or as the first positional input.
struct Options {
  std::filesystem::path device_path = "xcvu3p.device";
  std::filesystem::path phys_path;
  std::filesystem::path logical_path;
  std::filesystem::path output_dir;
  std::filesystem::path output_path;
  std::filesystem::path metadata_path;
  std::string output_name;
  Bounds bounds;
  FractionBounds fractions;
  BoundsSelectionMode bounds_mode = BoundsSelectionMode::kFullDevice;
  NodeBoundsMode node_bounds_mode = NodeBoundsMode::kFullyContained;
};

bool has_prefix(const std::string& text, const char* prefix) {
  const std::string prefix_text(prefix);
  return text.size() >= prefix_text.size() &&
         text.compare(0, prefix_text.size(), prefix_text) == 0;
}

bool has_suffix(const std::string& text, const char* suffix) {
  const std::string suffix_text(suffix);
  return text.size() >= suffix_text.size() &&
         text.compare(text.size() - suffix_text.size(),
                      suffix_text.size(),
                      suffix_text) == 0;
}

std::filesystem::path default_output_dir() {
  const std::filesystem::path source_path(__FILE__);
  const std::filesystem::path source_dir = source_path.parent_path();
  if (source_dir.filename() == "src" &&
      source_dir.parent_path().filename() == "SSSP-new") {
    return source_dir.parent_path() / "data";
  }

  const std::filesystem::path cwd = std::filesystem::current_path();
  if (cwd.filename() == "src" && cwd.parent_path().filename() == "SSSP-new") {
    return cwd.parent_path() / "data";
  }
  if (cwd.filename() == "SSSP-new" && std::filesystem::exists(cwd / "src")) {
    return cwd / "data";
  }
  if (std::filesystem::exists(cwd / "SSSP-new" / "src")) {
    return cwd / "SSSP-new" / "data";
  }
  if (std::filesystem::exists(cwd / "rips2026-amd" / "SSSP-new" / "src")) {
    return cwd / "rips2026-amd" / "SSSP-new" / "data";
  }
  return std::filesystem::path("rips2026-amd") / "SSSP-new" / "data";
}

std::string default_output_name(const std::filesystem::path& phys_path) {
  std::string name = phys_path.stem().string();
  if (has_suffix(name, ".phys")) {
    name.resize(name.size() - std::string(".phys").size());
  }
  return name.empty() ? std::string("graph") : name;
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

const char* bounds_selection_mode_name(BoundsSelectionMode mode) {
  switch (mode) {
    case BoundsSelectionMode::kFullDevice:
      return "full-device";
    case BoundsSelectionMode::kAbsoluteBounds:
      return "absolute-bounds";
    case BoundsSelectionMode::kFractionBounds:
      return "fraction-bounds";
  }
  return "unknown";
}

// Parse the explicit node filtering policy. The default is fully-contained so a
// selected coordinate window does not import nodes that cross the region edge.
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
      << " <unrouted.phys> <logical.netlist> [options]\n"
      << "  " << program
      << " <device.device> <unrouted.phys> <logical.netlist> "
         "[options]\n\n"
      << "Options:\n"
      << "  --device <path>                FPGAIF DeviceResources input.\n"
      << "  --output-dir <path>            Output directory. Default: "
         "rips2026-amd/SSSP-new/data.\n"
      << "  --output <path>                Outgoing RIPSOCS1 CSR output path.\n"
      << "  --metadata <path>              RIPSIFM1 metadata sidecar output path.\n"
      << "  --name <stem>                  Output filename stem when using "
         "--output-dir.\n"
      << "  --bounds <minX> <maxX> <minY> <maxY>\n"
      << "                                 Absolute tile-coordinate window.\n"
      << "  --fraction <minX> <maxX> <minY> <maxY>\n"
      << "                                 Fractional coordinate window, each 0..1.\n"
      << "  --x-fraction <min> <max>       Fractional X window, each 0..1.\n"
      << "  --y-fraction <min> <max>       Fractional Y window, each 0..1.\n"
      << "  --node-bounds-mode <mode>      poc-base-wire, fully-contained, "
         "or intersects.\n"
      << "  --full-device                  Import every tile that has XY coords.\n\n"
      << "Default: full-device, fully-contained nodes, outputs in "
         "SSSP-new/data.\n";
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

double parse_fraction_arg(const char* text, const char* name) {
  try {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(value) ||
        value < 0.0 || value > 1.0) {
      throw std::invalid_argument("invalid fraction");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name +
                             " fraction: " + text);
  }
}

// Split positional inputs from options:
//   physical netlist + logical netlist
//   device + physical netlist + logical netlist
// Device path can also be supplied by --device. Output paths default to
// SSSP-new/data and tile bounds default to the full device.
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

    if (arg == "--output-dir") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--output-dir requires a path");
      }
      options.output_dir = argv[++i];
      continue;
    }

    if (arg == "--output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--output requires a path");
      }
      options.output_path = argv[++i];
      continue;
    }

    if (arg == "--name") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--name requires a filename stem");
      }
      options.output_name = argv[++i];
      continue;
    }

    if (arg == "--bounds") {
      if (i + 4 >= argc) {
        throw std::runtime_error("--bounds requires four integer arguments");
      }
      if (options.bounds_mode == BoundsSelectionMode::kFractionBounds) {
        throw std::runtime_error("--bounds cannot be combined with fraction bounds");
      }
      options.bounds.min_x = parse_int_arg(argv[++i], "minX");
      options.bounds.max_x = parse_int_arg(argv[++i], "maxX");
      options.bounds.min_y = parse_int_arg(argv[++i], "minY");
      options.bounds.max_y = parse_int_arg(argv[++i], "maxY");
      options.bounds_mode = BoundsSelectionMode::kAbsoluteBounds;
      continue;
    }

    if (arg == "--fraction") {
      if (i + 4 >= argc) {
        throw std::runtime_error("--fraction requires four 0..1 arguments");
      }
      if (options.bounds_mode == BoundsSelectionMode::kAbsoluteBounds) {
        throw std::runtime_error("--fraction cannot be combined with --bounds");
      }
      options.fractions.min_x = parse_fraction_arg(argv[++i], "minX");
      options.fractions.max_x = parse_fraction_arg(argv[++i], "maxX");
      options.fractions.min_y = parse_fraction_arg(argv[++i], "minY");
      options.fractions.max_y = parse_fraction_arg(argv[++i], "maxY");
      options.bounds_mode = BoundsSelectionMode::kFractionBounds;
      continue;
    }

    if (arg == "--x-fraction") {
      if (i + 2 >= argc) {
        throw std::runtime_error("--x-fraction requires two 0..1 arguments");
      }
      if (options.bounds_mode == BoundsSelectionMode::kAbsoluteBounds) {
        throw std::runtime_error("--x-fraction cannot be combined with --bounds");
      }
      options.fractions.min_x = parse_fraction_arg(argv[++i], "minX");
      options.fractions.max_x = parse_fraction_arg(argv[++i], "maxX");
      options.bounds_mode = BoundsSelectionMode::kFractionBounds;
      continue;
    }

    if (arg == "--y-fraction") {
      if (i + 2 >= argc) {
        throw std::runtime_error("--y-fraction requires two 0..1 arguments");
      }
      if (options.bounds_mode == BoundsSelectionMode::kAbsoluteBounds) {
        throw std::runtime_error("--y-fraction cannot be combined with --bounds");
      }
      options.fractions.min_y = parse_fraction_arg(argv[++i], "minY");
      options.fractions.max_y = parse_fraction_arg(argv[++i], "maxY");
      options.bounds_mode = BoundsSelectionMode::kFractionBounds;
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
      options.fractions = FractionBounds{};
      options.bounds_mode = BoundsSelectionMode::kFullDevice;
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    }

    positional.emplace_back(arg);
  }

  if (positional.size() == 2) {
    options.phys_path = positional[0];
    options.logical_path = positional[1];
  } else if (positional.size() == 3) {
    options.device_path = positional[0];
    options.phys_path = positional[1];
    options.logical_path = positional[2];
  } else {
    throw std::runtime_error("expected two or three positional arguments");
  }

  if (options.output_dir.empty()) {
    options.output_dir = default_output_dir();
  }
  if (options.output_name.empty()) {
    options.output_name = default_output_name(options.phys_path);
  }
  if (options.output_path.empty()) {
    options.output_path =
        options.output_dir / (options.output_name + ".outgoing.csrbin");
  }
  if (options.metadata_path.empty()) {
    options.metadata_path = default_metadata_path(options.output_path);
  }

  if (options.bounds.min_x > options.bounds.max_x ||
      options.bounds.min_y > options.bounds.max_y) {
    throw std::runtime_error("invalid bounds: min is greater than max");
  }
  if (options.fractions.min_x > options.fractions.max_x ||
      options.fractions.min_y > options.fractions.max_y) {
    throw std::runtime_error("invalid fraction bounds: min is greater than max");
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
      bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + read_count);
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

int coordinate_from_fraction(int min_value, int max_value, double fraction) {
  const double span = static_cast<double>(max_value) - static_cast<double>(min_value);
  const double coordinate = static_cast<double>(min_value) + span * fraction;
  return static_cast<int>(std::floor(coordinate));
}

Bounds resolve_selected_bounds(const Options& options,
                               int device_min_x,
                               int device_max_x,
                               int device_min_y,
                               int device_max_y) {
  if (options.bounds_mode == BoundsSelectionMode::kFullDevice) {
    Bounds bounds;
    bounds.min_x = device_min_x;
    bounds.max_x = device_max_x;
    bounds.min_y = device_min_y;
    bounds.max_y = device_max_y;
    return bounds;
  }
  if (options.bounds_mode == BoundsSelectionMode::kAbsoluteBounds) {
    return options.bounds;
  }

  Bounds bounds;
  bounds.min_x = coordinate_from_fraction(
      device_min_x, device_max_x, options.fractions.min_x);
  bounds.max_x = coordinate_from_fraction(
      device_min_x, device_max_x, options.fractions.max_x);
  bounds.min_y = coordinate_from_fraction(
      device_min_y, device_max_y, options.fractions.min_y);
  bounds.max_y = coordinate_from_fraction(
      device_min_y, device_max_y, options.fractions.max_y);
  return bounds;
}

void build_device_routing_graph(const Options& options, RoutingGraph& graph) {
  // Load the whole DeviceResources message into memory, matching the Python
  // proof of concept. DeviceResources is the fabric database: tiles, wires,
  // routing nodes, tile types, PIPs, site types, and site instances.
  const std::vector<std::uint8_t> bytes =
      read_gzip_or_plain_file(options.device_path);
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

  // First pass over tileList: collect every tile whose name carries X/Y
  // coordinates. Fractional region requests are resolved against this full
  // device extent, then selected tiles are kept by absolute X/Y bounds.
  std::unordered_set<std::uint32_t> in_bounds_tile_name_ids;
  std::vector<std::uint32_t> in_bounds_tile_indices;
  std::vector<TileCoord> xy_tiles;
  const std::regex tile_xy_regex(R"([A-Z0-9_]+_X([0-9]+)Y([0-9]+))");
  int device_min_x = std::numeric_limits<int>::max();
  int device_max_x = std::numeric_limits<int>::min();
  int device_min_y = std::numeric_limits<int>::max();
  int device_max_y = std::numeric_limits<int>::min();

  for (std::uint32_t tile_index = 0; tile_index < tile_list.size();
       ++tile_index) {
    const auto tile = tile_list[tile_index];
    const std::string& tile_name = strings.get(tile.getName());

    std::smatch match;
    if (!std::regex_search(tile_name, match, tile_xy_regex) ||
        match.position() != 0) {
      continue;
    }

    const int x = parse_int_arg(match[1].str().c_str(), "tile X");
    const int y = parse_int_arg(match[2].str().c_str(), "tile Y");
    TileCoord coord;
    coord.tile_index = tile_index;
    coord.tile_name_id = tile.getName();
    coord.x = x;
    coord.y = y;
    xy_tiles.push_back(coord);

    device_min_x = std::min(device_min_x, x);
    device_max_x = std::max(device_max_x, x);
    device_min_y = std::min(device_min_y, y);
    device_max_y = std::max(device_max_y, y);
  }

  if (xy_tiles.empty()) {
    throw std::runtime_error("device has no XY-coordinate tiles to import");
  }

  const Bounds selected_bounds = resolve_selected_bounds(options,
                                                        device_min_x,
                                                        device_max_x,
                                                        device_min_y,
                                                        device_max_y);
  graph.selected_bounds = selected_bounds;
  graph.xy_tile_count = static_cast<std::uint64_t>(xy_tiles.size());

  for (const TileCoord& coord : xy_tiles) {
    if (coord.x < selected_bounds.min_x || coord.x > selected_bounds.max_x ||
        coord.y < selected_bounds.min_y || coord.y > selected_bounds.max_y) {
      continue;
    }

    in_bounds_tile_name_ids.insert(coord.tile_name_id);
    in_bounds_tile_indices.push_back(coord.tile_index);
  }
  graph.selected_tile_count =
      static_cast<std::uint64_t>(in_bounds_tile_indices.size());

  if (in_bounds_tile_indices.empty()) {
    throw std::runtime_error("selected tile-coordinate window contains no tiles");
  }

  if (nodes.size() > static_cast<std::uint64_t>(
                         std::numeric_limits<NodeId>::max())) {
    throw std::runtime_error("device has too many nodes for 32-bit CSR IDs");
  }

  // Build CSR/compact node IDs from DeviceResources.nodes. FPGAIF already
  // groups electrically-equivalent wires into routing nodes. The default
  // fully-contained mode imports nodes only when every wire is in the selected
  // X/Y window, which keeps region-limited outgoing CSR edges inside the region.
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
    switch (options.node_bounds_mode) {
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
      for (const SitePinName& source_pin : source_pins) {
        const std::optional<NodeId> source_node =
            get_node_from_site_pin(graph, source_pin.site, source_pin.pin);
        if (!source_node.has_value()) {
          continue;
        }

        SitePinNode source;
        source.node = *source_node;
        source.site_string = graph.string_table.intern(source_pin.site);
        source.pin_string = graph.string_table.intern(source_pin.pin);
        request.sources.push_back(source);
      }

      // Map each sink site pin to a routing node. Sinks become route targets
      // and get stored as node metadata ("sp") in the sidecar.
      for (const SitePinName& sink_pin : sink_pins) {
        const std::optional<NodeId> sink_node =
            get_node_from_site_pin(graph, sink_pin.site, sink_pin.pin);
        if (!sink_node.has_value()) {
          continue;
        }

        const NodeId node = *sink_node;
        if (request.sources.empty()) {
          graph.blocked_node[node] = 1;
          continue;
        }

        SitePinNode sink;
        sink.node = node;
        sink.site_string = graph.string_table.intern(sink_pin.site);
        sink.pin_string = graph.string_table.intern(sink_pin.pin);
        request.sinks.push_back(sink);
        graph.sink_node_stops[node] = 1;
        add_site_pin_attr(graph, node, sink.site_string, sink.pin_string);
      }

      if (!request.sources.empty() && !request.sinks.empty()) {
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

  std::vector<std::vector<CsrEntry>> incoming_rows(
      static_cast<std::size_t>(csr.rows));

  for (const BuildEdge& edge : graph.edges) {
    if (edge.from < 0 || edge.to < 0 || edge.from >= csr.rows ||
        edge.to >= csr.rows) {
      continue;
    }

    if (graph.blocked_node[edge.from] || graph.blocked_node[edge.to]) {
      continue;
    }

    if (graph.sink_node_stops[edge.from]) {
      continue;
    }

    CsrEntry entry;
    entry.col = edge.from;
    entry.value = 1.0f;
    entry.attr = edge.attr;
    incoming_rows[edge.to].push_back(entry);
  }

  for (std::int64_t row = 0; row < csr.rows; ++row) {
    auto& entries = incoming_rows[static_cast<std::size_t>(row)];
    csr.rowptr[static_cast<std::size_t>(row)] =
        static_cast<std::int64_t>(csr.colind.size());

    std::sort(entries.begin(),
              entries.end(),
              [](const CsrEntry& lhs, const CsrEntry& rhs) {
                return lhs.col < rhs.col;
              });

    for (const CsrEntry& entry : entries) {
      csr.colind.push_back(entry.col);
      csr.values.push_back(entry.value);
      csr.edge_attrs.push_back(entry.attr);
    }
  }

  csr.rowptr[static_cast<std::size_t>(csr.rows)] =
      static_cast<std::int64_t>(csr.colind.size());
  return csr;
}

void validate_outgoing_csr(const OutgoingCsrGraph& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph.rows > static_cast<std::int64_t>(
                       std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for 32-bit node ids");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.degree.size() != static_cast<std::size_t>(graph.rows) ||
      graph.to.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.edge_id.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.edge_attrs.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("outgoing CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("outgoing CSR rowptr must start at 0 and end at nnz");
  }

  for (std::int64_t row = 0; row < graph.rows; ++row) {
    const std::int64_t begin = graph.rowptr[static_cast<std::size_t>(row)];
    const std::int64_t end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const std::int64_t row_degree = end - begin;
    if (row_degree > static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    if (graph.degree[static_cast<std::size_t>(row)] !=
        static_cast<std::int32_t>(row_degree)) {
      throw std::runtime_error("outgoing CSR degree array does not match rowptr");
    }
  }

  for (std::size_t edge = 0; edge < graph.to.size(); ++edge) {
    if (graph.to[edge] < 0 ||
        static_cast<std::int64_t>(graph.to[edge]) >= graph.cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (graph.edge_id[edge] < 0 ||
        static_cast<std::uint64_t>(graph.edge_id[edge]) >=
            kPackedNoPredEdge) {
      throw std::runtime_error("outgoing CSR edge id cannot be packed");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
  }
}

OutgoingCsrGraph make_outgoing_csr(const CsrGraph& incoming) {
  OutgoingCsrGraph outgoing;
  outgoing.rows = incoming.rows;
  outgoing.cols = incoming.cols;
  outgoing.nnz = static_cast<std::int64_t>(incoming.values.size());
  outgoing.rowptr.assign(static_cast<std::size_t>(incoming.rows + 1), 0);
  outgoing.degree.assign(static_cast<std::size_t>(incoming.rows), 0);
  outgoing.to.resize(static_cast<std::size_t>(outgoing.nnz));
  outgoing.values.resize(static_cast<std::size_t>(outgoing.nnz));
  outgoing.edge_id.resize(static_cast<std::size_t>(outgoing.nnz));
  outgoing.edge_attrs.resize(static_cast<std::size_t>(outgoing.nnz));

  for (std::int64_t edge = 0; edge < outgoing.nnz; ++edge) {
    const std::int32_t from = incoming.colind[static_cast<std::size_t>(edge)];
    if (from < 0 || static_cast<std::int64_t>(from) >= incoming.rows) {
      throw std::runtime_error("incoming CSR colind contains an invalid source");
    }
    ++outgoing.rowptr[static_cast<std::size_t>(from + 1)];
  }

  for (std::int64_t row = 0; row < incoming.rows; ++row) {
    outgoing.rowptr[static_cast<std::size_t>(row + 1)] +=
        outgoing.rowptr[static_cast<std::size_t>(row)];
  }

  for (std::int64_t row = 0; row < incoming.rows; ++row) {
    const std::int64_t row_degree =
        outgoing.rowptr[static_cast<std::size_t>(row + 1)] -
        outgoing.rowptr[static_cast<std::size_t>(row)];
    if (row_degree > static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    outgoing.degree[static_cast<std::size_t>(row)] =
        static_cast<std::int32_t>(row_degree);
  }

  std::vector<std::int64_t> cursor = outgoing.rowptr;
  for (std::int64_t to = 0; to < incoming.rows; ++to) {
    for (std::int64_t edge = incoming.rowptr[static_cast<std::size_t>(to)];
         edge < incoming.rowptr[static_cast<std::size_t>(to + 1)];
         ++edge) {
      const std::int32_t from = incoming.colind[static_cast<std::size_t>(edge)];
      const std::int64_t dst = cursor[static_cast<std::size_t>(from)]++;
      outgoing.to[static_cast<std::size_t>(dst)] = static_cast<std::int32_t>(to);
      outgoing.values[static_cast<std::size_t>(dst)] =
          incoming.values[static_cast<std::size_t>(edge)];
      outgoing.edge_id[static_cast<std::size_t>(dst)] = dst;
      outgoing.edge_attrs[static_cast<std::size_t>(dst)] =
          incoming.edge_attrs[static_cast<std::size_t>(edge)];
    }
  }

  validate_outgoing_csr(outgoing);
  return outgoing;
}

void write_outgoing_csr_graph(const OutgoingCsrGraph& graph,
                              const std::filesystem::path& output_path) {
  // This binary file intentionally contains only the outgoing graph structure
  // needed by mp_csr.cpp. FPGA routing names and PIP details live in the
  // outgoing-edge-aligned RIPSIFM1 metadata sidecar.
  validate_outgoing_csr(graph);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open output file: " +
                             output_path.string());
  }

  out.write(OUTGOING_CSR_MAGIC, sizeof(OUTGOING_CSR_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing outgoing CSR magic");
  }

  const std::uint64_t nnz = as_u64(graph.nnz, "nnz");
  write_u64(out, OUTGOING_CSR_FORMAT_VERSION, "format version");
  write_u64(out, OUTGOING_EDGE_ORIENTATION, "orientation");
  write_u64(out, as_u64(graph.rows, "rows"), "row count");
  write_u64(out, as_u64(graph.cols, "cols"), "column count");
  write_u64(out, nnz, "nnz");
  write_u64(out, static_cast<std::uint64_t>(graph.rowptr.size()),
            "rowptr count");
  write_u64(out, static_cast<std::uint64_t>(graph.degree.size()),
            "degree count");
  write_u64(out, static_cast<std::uint64_t>(graph.to.size()),
            "destination count");
  write_u64(out, static_cast<std::uint64_t>(graph.values.size()),
            "values count");
  write_u64(out, static_cast<std::uint64_t>(graph.edge_id.size()),
            "edge-id count");

  write_array(out, graph.rowptr, "rowptr");
  write_array(out, graph.degree, "degree");
  write_array(out, graph.to, "destinations");
  write_array(out, graph.values, "values");
  write_array(out, graph.edge_id, "edge-id map");
}

void write_metadata(const RoutingGraph& graph,
                    const OutgoingCsrGraph& csr,
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
  //   edge_attr_count outgoing CSR records: u64 tile_string, u64 pip_data_index
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
  // SSSP-new metadata is outgoing-edge aligned. This version/orientation pair
  // intentionally differs from Routing's incoming-oriented RIPSIFM1 sidecar so
  // old readers fail clearly instead of silently using the wrong edge order.
  write_u64(out, OUTGOING_EDGE_ORIENTATION, "metadata orientation");
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

  // Edge attributes are aligned exactly with outgoing CSR order. For outgoing
  // edge k, csr.to[k], csr.values[k], and edge_attrs[k] describe one PIP edge.
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
      write_u64(out, static_cast<std::uint64_t>(source.node),
                "route request source node");
      write_u64(out, source.site_string, "route request source site");
      write_u64(out, source.pin_string, "route request source pin");
    }

    write_u64(out, static_cast<std::uint64_t>(request.sinks.size()),
              "route request sink count");
    for (const SitePinNode& sink : request.sinks) {
      write_u64(out, static_cast<std::uint64_t>(sink.node),
                "route request sink node");
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
    std::cout << "output_csr: " << options.output_path << "\n";
    std::cout << "metadata: " << options.metadata_path << "\n";
    std::cout << "bounds_mode: "
              << bounds_selection_mode_name(options.bounds_mode) << "\n";
    if (options.bounds_mode == BoundsSelectionMode::kAbsoluteBounds) {
      std::cout << "requested_bounds: X" << options.bounds.min_x << "..X"
                << options.bounds.max_x << ", Y" << options.bounds.min_y
                << "..Y" << options.bounds.max_y << "\n";
    } else if (options.bounds_mode == BoundsSelectionMode::kFractionBounds) {
      std::cout << "requested_fraction_bounds: X" << options.fractions.min_x
                << ".." << options.fractions.max_x << ", Y"
                << options.fractions.min_y << ".." << options.fractions.max_y
                << "\n";
    }
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
    build_device_routing_graph(options, graph);

    std::cout << "xy_tiles: " << graph.xy_tile_count << "\n";
    std::cout << "selected_tiles: " << graph.selected_tile_count << "\n";
    std::cout << "selected_bounds: X" << graph.selected_bounds.min_x << "..X"
              << graph.selected_bounds.max_x << ", Y"
              << graph.selected_bounds.min_y << "..Y"
              << graph.selected_bounds.max_y << "\n";
    std::cout << "imported_nodes: " << graph.node_device_ids.size() << "\n";
    std::cout << "unique_edges: " << graph.edges.size() << "\n";

    // LogicalNetlist parsing records logical cells/nets/port instances and
    // builds a name index that physical route requests can reference.
    parse_logical_netlist(options.logical_path, graph);

    // PhysicalNetlist parsing adds design-specific route requests and blockage.
    // It also stores the original .phys payload so later code can patch routes
    // into that exact physical netlist structure.
    parse_physical_netlist(options.phys_path, graph);

    // The outgoing CSR is the GPU-facing graph. The sidecar is aligned to the
    // same outgoing edge order, so metadata edge k describes outgoing edge k.
    CsrGraph incoming = make_incoming_csr(graph);
    OutgoingCsrGraph outgoing = make_outgoing_csr(incoming);
    write_outgoing_csr_graph(outgoing, options.output_path);
    write_metadata(graph, outgoing, options.metadata_path);

    const std::uint64_t rowptr_bytes =
        static_cast<std::uint64_t>(outgoing.rowptr.size() * sizeof(std::int64_t));
    const std::uint64_t degree_bytes =
        static_cast<std::uint64_t>(outgoing.degree.size() * sizeof(std::int32_t));
    const std::uint64_t to_bytes =
        static_cast<std::uint64_t>(outgoing.to.size() * sizeof(std::int32_t));
    const std::uint64_t values_bytes =
        static_cast<std::uint64_t>(outgoing.values.size() * sizeof(float));
    const std::uint64_t edge_id_bytes =
        static_cast<std::uint64_t>(
            outgoing.edge_id.size() * sizeof(std::int64_t));
    const std::uint64_t attr_bytes =
        static_cast<std::uint64_t>(outgoing.edge_attrs.size() * sizeof(EdgeAttr));
    const std::uint64_t csr_bytes =
        rowptr_bytes + degree_bytes + to_bytes + values_bytes +
        edge_id_bytes;

    std::cout << "csr_orientation: outgoing\n";
    std::cout << "csr_rows: " << outgoing.rows << "\n";
    std::cout << "csr_nnz: " << outgoing.nnz << "\n";
    std::cout << "csr_total_mib: " << mib(csr_bytes) << "\n";
    std::cout << "csr_rowptr_mib: " << mib(rowptr_bytes) << "\n";
    std::cout << "csr_degree_mib: " << mib(degree_bytes) << "\n";
    std::cout << "csr_to_mib: " << mib(to_bytes) << "\n";
    std::cout << "csr_values_mib: " << mib(values_bytes) << "\n";
    std::cout << "csr_edge_id_mib: " << mib(edge_id_bytes) << "\n";
    std::cout << "metadata_edge_attr_mib: " << mib(attr_bytes) << "\n";

    std::cout << "wrote_outgoing_csr: " << options.output_path << "\n";
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
