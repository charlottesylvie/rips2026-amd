// Converts a preprocessed RIPSDRG1 device-routing graph plus a design's
// PhysicalNetlist and LogicalNetlist into the CSR binary format consumed by
// the CongestionFreeRouting HIP kernels.
//
// The primary output is a RIPSCSR1 .csrbin file:
//
//   CSR row u, column v = routing edge u -> v, with unit edge weight.
//
// This outgoing-edge orientation is emitted directly so the low-level HIP
// kernels can traverse source frontiers without first transposing the graph.
// A second sidecar file preserves the FPGA-specific NetworkX-style attributes:
//
//   edge attribute "pip" = (tileName, pipDataIndex)
//   pipData[pipDataIndex] = (wire0Name, wire1Name, forward)
//   node attribute "sp" = (siteName, pinName), for sink site pins
//
// The sidecar also records route requests extracted from PhysicalNetlist stubs
// and source site pins, logical net summaries extracted from LogicalNetlist,
// and the original decompressed .phys/.netlist payloads. That metadata is
// intentionally CPU-side: the GPU CSR remains compact and compatible with the
// routing kernels, while later post-processing has enough context
// to turn SSSP paths back into PhysPIP route branches in a routed .phys file.
//
// DeviceResources parsing and invariant graph construction live in
// device_to_routing_graph.cpp and run once per (device, bounds, bounds mode).
// This per-benchmark stage only needs these generated schema headers:
//   PhysicalNetlist.capnp.h
//   LogicalNetlist.capnp.h
//
// Example compile command, after generating the C++ Cap'n Proto schema files:
//
//   g++ -std=c++17 -O3 \
//     -I<generated-schema-dir> \
//     CongestionFreeRouting/interchange_to_csr.cpp \
//     CongestionFreeRouting/interchange/device_routing_graph.cpp \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     <generated-schema-dir>/LogicalNetlist.capnp.c++ \
//     <generated-schema-dir>/References.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o interchange_to_csr
//
// Example use:
//
//   ./interchange_to_csr xcvu3p.full-poc-base-wire.devicegraph \
//     benchmarks/vtr_mcml_unrouted.phys \
//     benchmarks/vtr_mcml.netlist \
//     HIP_kernel/bellman_ford/data/vtr_mcml_fpga.csrbin

#include "LogicalNetlist.capnp.h"
#include "PhysicalNetlist.capnp.h"
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
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Keep the on-disk binary formats explicit. The CSR constants identify the
// generic graph payload; the metadata constants identify the sidecar that
// stores FPGA-specific information not present in plain CSR.
static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

struct PipDataDisk {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
};

static_assert(sizeof(PipDataDisk) == 3 * sizeof(std::uint64_t),
              "PipDataDisk metadata layout changed");

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t CSR_FORMAT_VERSION = 1;
constexpr std::uint64_t METADATA_FORMAT_VERSION = 4;
constexpr std::uint64_t OUTGOING_EDGE_ORIENTATION = 2;
using routing::interchange::CsrGraph;
using routing::interchange::DeviceRoutingGraph;
using routing::interchange::EdgeAttr;
using routing::interchange::NodeId;
using routing::interchange::PipData;
using routing::interchange::StringTable;
using routing::interchange::filter_device_routing_graph;
using routing::interchange::find_pair_node;
using routing::interchange::kInvalidRouteNode;
using routing::interchange::kNoIndex;
using routing::interchange::kNoLogicalNetIndex;
using routing::interchange::kNoStringIndex;
using routing::interchange::node_bounds_mode_name;
using routing::interchange::read_device_routing_graph_for_filtering;

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

// Static device fields are loaded from the preprocessed artifact. This derived
// structure adds only benchmark-specific masks, requests, logical summaries,
// and provenance.
struct RoutingGraph : DeviceRoutingGraph {
  explicit RoutingGraph(DeviceRoutingGraph&& device_graph)
      : DeviceRoutingGraph(std::move(device_graph)) {
    blocked_node.assign(node_device_ids.size(), 0);
    sink_node_stops.assign(node_device_ids.size(), 0);
    site_pin_attr_by_node.assign(node_device_ids.size(), -1);
  }

  std::vector<std::uint8_t> blocked_node;
  std::vector<std::uint8_t> sink_node_stops;
  std::vector<std::int64_t> site_pin_attr_by_node;
  std::vector<SitePinNode> site_pin_attrs;
  std::vector<RouteRequest> route_requests;

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

// The static device graph is an explicit positional input so a benchmark can
// never silently pay the DeviceResources build cost.
struct Options {
  std::filesystem::path device_graph_path;
  std::filesystem::path phys_path;
  std::filesystem::path logical_path;
  std::filesystem::path output_path;
  std::filesystem::path metadata_path;
};

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <device.devicegraph> <unrouted.phys> <logical.netlist> "
         "<output.csrbin> [options]\n\n"
      << "Options:\n"
      << "  --metadata <path>              Sidecar FPGA metadata output.\n\n"
      << "Generate <device.devicegraph> once with device_to_routing_graph.\n";
}

std::filesystem::path default_metadata_path(
    const std::filesystem::path& output_path) {
  std::filesystem::path path = output_path;
  path += ".ifmeta.bin";
  return path;
}


Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<std::filesystem::path> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--metadata") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--metadata requires a path");
      }
      options.metadata_path = argv[++i];
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    }

    positional.emplace_back(arg);
  }

  if (positional.size() == 4) {
    options.device_graph_path = positional[0];
    options.phys_path = positional[1];
    options.logical_path = positional[2];
    options.output_path = positional[3];
  } else {
    throw std::runtime_error("expected four positional arguments");
  }

  if (options.metadata_path.empty()) {
    options.metadata_path = default_metadata_path(options.output_path);
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
  (void)gzbuffer(file, 1 << 20);

  std::vector<std::uint8_t> bytes;
  std::error_code size_error;
  const std::uintmax_t encoded_size =
      std::filesystem::file_size(path, size_error);
  if (!size_error &&
      encoded_size <= std::numeric_limits<std::size_t>::max()) {
    bytes.reserve(static_cast<std::size_t>(encoded_size));
  }
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

// Direct lookup records were resolved once by device_to_routing_graph.
std::optional<NodeId> get_node_from_site_pin(const RoutingGraph& graph,
                                             const std::string& site_name,
                                             const std::string& pin_name) {
  return find_pair_node(graph.site_pin_nodes, graph.string_table, site_name,
                        pin_name);
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
  return find_pair_node(graph.tile_wire_nodes, graph.string_table, tile_name,
                        wire_name);
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
  std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(logical_path);
  std::vector<capnp::word> words = bytes_to_words(bytes);
  graph.logical_netlist_bytes = std::move(bytes);

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

void parse_physical_netlist(const std::filesystem::path& phys_path,
                            RoutingGraph& graph) {
  using PhysNetlist = PhysicalNetlist::PhysNetlist;
  using RouteBranch = PhysNetlist::RouteBranch;

  // PhysicalNetlist is design-specific. It supplies unrouted signal stubs,
  // legal source site pins, and already-occupied routing from fixed or
  // pre-routed nets.
  std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(phys_path);
  std::vector<capnp::word> words = bytes_to_words(bytes);
  graph.physical_netlist_bytes = std::move(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(words.data(), words.size()), reader_options);
  const auto netlist = reader.getRoot<PhysNetlist>();
  TextCache strings(netlist.getStrList());

  const auto phys_nets = netlist.getPhysNets();
  graph.route_requests.reserve(phys_nets.size());
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
      request.sources.reserve(source_pins.size());
      request.sinks.reserve(sink_pins.size());
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

CsrGraph make_outgoing_csr(const RoutingGraph& graph) {
  return filter_device_routing_graph(graph, graph.blocked_node,
                                     graph.sink_node_stops);
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
  write_u64(out, OUTGOING_EDGE_ORIENTATION, "orientation");
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
  //   i32[node_count] min X tile coordinate, or -1
  //   i32[node_count] max X tile coordinate, or -1
  //   i32[node_count] min Y tile coordinate, or -1
  //   i32[node_count] max Y tile coordinate, or -1
  //   u64[node_count] representative tile-type string index, or kNoIndex
  //   u64[node_count] representative wire-type string index, or kNoIndex
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
  if (graph.node_min_x.size() != graph.node_device_ids.size() ||
      graph.node_max_x.size() != graph.node_device_ids.size() ||
      graph.node_min_y.size() != graph.node_device_ids.size() ||
      graph.node_max_y.size() != graph.node_device_ids.size() ||
      graph.node_tile_type_strings.size() != graph.node_device_ids.size() ||
      graph.node_wire_type_strings.size() != graph.node_device_ids.size()) {
    throw std::runtime_error("node metadata arrays do not match node count");
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
  // index plus lightweight physical metadata useful to GPU cost models.
  write_array(out, graph.node_device_ids, "device node ids");
  write_array(out, graph.node_min_x, "node min x coordinates");
  write_array(out, graph.node_max_x, "node max x coordinates");
  write_array(out, graph.node_min_y, "node min y coordinates");
  write_array(out, graph.node_max_y, "node max y coordinates");
  write_array(out, graph.node_tile_type_strings, "node tile type strings");
  write_array(out, graph.node_wire_type_strings, "node wire type strings");

  // Edge attributes are aligned exactly with CSR colind/values order. For edge
  // k, csr.colind[k], csr.values[k], and edge_attrs[k] describe one PIP edge.
  static_assert(sizeof(EdgeAttr) == 2 * sizeof(std::uint64_t),
                "EdgeAttr metadata layout changed");
  write_array(out, csr.edge_attrs, "edge attributes");

  // PIP data table stores the wire pair and direction referenced by each edge
  // attribute. tile name lives on EdgeAttr because the same wire pair appears
  // in many tile instances.
  std::vector<PipDataDisk> pip_data;
  pip_data.reserve(graph.pip_data.size());
  for (const PipData& data : graph.pip_data) {
    pip_data.push_back(
        {data.wire0_string, data.wire1_string, data.forward ? 1ULL : 0ULL});
  }
  write_array(out, pip_data, "pip data");

  // Sink site-pin node attributes, matching NetworkX's node attribute "sp".
  for (const SitePinNode& attr : graph.site_pin_attrs) {
    write_u64(out, static_cast<std::uint64_t>(attr.node),
              "site pin attr node");
    write_u64(out, attr.site_string, "site pin attr site string");
    write_u64(out, attr.pin_string, "site pin attr pin string");
  }

  // Route requests preserve net -> source nodes and sink nodes. A future
  // router can run shortest paths over the CSR using these node
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

template <typename Container>
void release_storage(Container& container) {
  Container empty;
  container.swap(empty);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    std::cout << "device_graph: " << options.device_graph_path << "\n";
    std::cout << "physical_netlist: " << options.phys_path << "\n";
    std::cout << "logical_netlist: " << options.logical_path << "\n";
    RoutingGraph graph(
        read_device_routing_graph_for_filtering(options.device_graph_path));
    std::cout << "device_fingerprint: " << graph.device_fingerprint << "\n";
    std::cout << "bounds: X" << graph.bounds.min_x << "..X"
              << graph.bounds.max_x << ", Y" << graph.bounds.min_y << "..Y"
              << graph.bounds.max_y << "\n";
    std::cout << "node_bounds_mode: "
              << node_bounds_mode_name(graph.node_bounds_mode) << "\n";

    // Static string IDs are loaded first and remain stable. Benchmark-specific
    // provenance and net names are appended to that namespace.
    graph.physical_path_string =
        graph.string_table.intern(options.phys_path.string());
    graph.logical_path_string =
        graph.string_table.intern(options.logical_path.string());

    std::cout << "imported_nodes: " << graph.node_device_ids.size() << "\n";
    std::cout << "unique_edges: " << graph.loaded_edges << "\n";

    // LogicalNetlist parsing records logical cells/nets/port instances and
    // builds a name index that physical route requests can reference.
    parse_logical_netlist(options.logical_path, graph);

    // PhysicalNetlist parsing adds design-specific route requests and blockage.
    // It also stores the original .phys payload so later code can patch routes
    // into that exact physical netlist structure.
    parse_physical_netlist(options.phys_path, graph);

    // These indexes exist only to resolve names while parsing the two design
    // netlists. Release them before allocating the filtered CSR; on a full
    // device they otherwise overlap several other multi-gigabyte arrays.
    release_storage(graph.tile_wire_nodes);
    release_storage(graph.site_pin_nodes);
    release_storage(graph.string_table.ids);
    release_storage(graph.logical_net_index_by_name);
    release_storage(graph.site_pin_attr_by_node);

    // CSR is the GPU-facing graph. Metadata is the CPU-facing FPGA context
    // needed to map CSR edges back to tile/wire PIPs and site-pin targets.
    CsrGraph csr = make_outgoing_csr(graph);

    // Filtering has copied every retained destination and edge attribute.
    // Drop the immutable base CSR before serializing either design output.
    release_storage(graph.rowptr);
    release_storage(graph.colind);
    release_storage(graph.edge_attrs);

    const std::uint64_t rowptr_bytes =
        static_cast<std::uint64_t>(csr.rowptr.size() * sizeof(std::int64_t));
    const std::uint64_t colind_bytes =
        static_cast<std::uint64_t>(csr.colind.size() * sizeof(std::int32_t));
    const std::uint64_t values_bytes =
        static_cast<std::uint64_t>(csr.values.size() * sizeof(float));
    const std::uint64_t attr_bytes =
        static_cast<std::uint64_t>(csr.edge_attrs.size() * sizeof(EdgeAttr));
    const std::uint64_t csr_bytes = rowptr_bytes + colind_bytes + values_bytes;
    const std::int64_t csr_rows = csr.rows;
    const std::size_t csr_nnz = csr.values.size();

    write_csr_graph(csr, options.output_path);

    // The metadata sidecar needs edge attributes, but not the three generic
    // CSR arrays that were just written.
    release_storage(csr.rowptr);
    release_storage(csr.colind);
    release_storage(csr.values);
    write_metadata(graph, csr, options.metadata_path);

    std::cout << "csr_rows: " << csr_rows << "\n";
    std::cout << "csr_nnz: " << csr_nnz << "\n";
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
