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
#include "interchange/gzip_io.hpp"
#include "interchange/import_policy.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <algorithm>
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
constexpr std::uint64_t CSR_FORMAT_VERSION = 2;
constexpr std::uint64_t METADATA_FORMAT_VERSION = 5;
constexpr std::uint64_t OUTGOING_EDGE_ORIENTATION = 2;
using routing::interchange::CsrGraph;
using routing::interchange::DeviceRoutingGraph;
using routing::interchange::EdgeAttr;
using routing::interchange::InterchangeArtifactPairId;
using routing::interchange::NodeId;
using routing::interchange::PipData;
using routing::interchange::StringTable;
using routing::interchange::filter_device_routing_graph;
using routing::interchange::find_pair_node;
using routing::interchange::find_site_pin_candidates;
using routing::interchange::find_site_pin_node;
using routing::interchange::kInvalidRouteNode;
using routing::interchange::kNoIndex;
using routing::interchange::kNoLogicalNetIndex;
using routing::interchange::kNoStringIndex;
using routing::interchange::node_bounds_mode_name;
using routing::interchange::read_device_routing_graph_for_filtering;
using routing::interchange::read_gzip_or_plain_chunks;

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
    // With a shared immutable CSR, terminal rows are the conservative guard
    // against one net traversing another net's exclusive sink site pin.  This
    // matches the contest POC.  RWRoute can make a narrower same-net pinbounce
    // exception because its traversal carries connection ownership and intent;
    // this graph currently cannot.
    sink_node_stops.assign(node_device_ids.size(), 0);
    exclusive_source_nodes.assign(node_device_ids.size(), 0);
  }

  std::vector<std::uint8_t> blocked_node;
  std::vector<std::uint8_t> sink_node_stops;
  std::vector<std::uint8_t> exclusive_source_nodes;
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

// PhysicalNetlist.siteInsts gives the active type for each concrete site.  A
// type is required to distinguish primary/alternate pin names that can map to
// different device nodes.
class ActiveSiteTypes {
 public:
  void insert(const std::string& site, const std::string& type) {
    const auto [found, inserted] = type_by_site_.emplace(site, type);
    if (!inserted && found->second != type) {
      throw std::runtime_error(
          "PhysicalNetlist assigns conflicting active types to site " +
          site);
    }
  }

  std::optional<std::string> find(const std::string& site) const {
    const auto found = type_by_site_.find(site);
    if (found == type_by_site_.end()) {
      return std::nullopt;
    }
    return found->second;
  }

 private:
  std::unordered_map<std::string, std::string> type_by_site_;
};

// The static device graph is an explicit positional input so a benchmark can
// never silently pay the DeviceResources build cost.
struct Options {
  std::filesystem::path device_graph_path;
  std::filesystem::path phys_path;
  std::filesystem::path logical_path;
  std::filesystem::path output_path;
  std::filesystem::path metadata_path;
  bool allow_unsupported_preserved_nets = false;
};

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <device.devicegraph> <unrouted.phys> <logical.netlist> "
         "<output.csrbin> [options]\n\n"
      << "Options:\n"
      << "  --metadata <path>              Sidecar FPGA metadata output.\n\n"
      << "  --allow-unsupported-preserved-nets\n"
      << "                                 Keep unsupported partial/static "
         "work unchanged.\n\n"
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

    if (arg == "--allow-unsupported-preserved-nets") {
      options.allow_unsupported_preserved_nets = true;
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
  std::vector<std::uint8_t> bytes;
  std::error_code size_error;
  const std::uintmax_t encoded_size =
      std::filesystem::file_size(path, size_error);
  if (!size_error &&
      encoded_size <= std::numeric_limits<std::size_t>::max()) {
    bytes.reserve(static_cast<std::size_t>(encoded_size));
  }
  read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t byte_count) {
        if (byte_count > bytes.max_size() - bytes.size()) {
          throw std::runtime_error("decoded input is too large: " +
                                   path.string());
        }
        const std::size_t old_size = bytes.size();
        bytes.resize(old_size + byte_count);
        std::memcpy(bytes.data() + old_size, data, byte_count);
      });

  if (bytes.empty()) {
    throw std::runtime_error("input file is empty: " + path.string());
  }

  return bytes;
}

// Cap'n Proto's FlatArrayMessageReader expects exact word-aligned storage.
// Copy only after rejecting a partial final word instead of hiding it with
// zero-padding.
std::vector<capnp::word> bytes_to_words(
    const std::vector<std::uint8_t>& bytes,
    const std::filesystem::path& path) {
  const std::size_t word_size = sizeof(capnp::word);
  if (bytes.size() % word_size != 0) {
    throw std::runtime_error(
        "decoded Cap'n Proto input is not word-aligned: " + path.string() +
        " has " + std::to_string(bytes.size()) + " bytes");
  }
  const std::size_t word_count = bytes.size() / word_size;
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

template <typename T>
std::size_t checked_array_bytes(std::size_t count, const char* name) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error(std::string(name) + " byte count overflows size_t");
  }
  const std::size_t bytes = count * sizeof(T);
  if (bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error(std::string(name) +
                             " byte count exceeds streamsize");
  }
  return bytes;
}

std::uint64_t checked_add_u64(std::uint64_t lhs,
                              std::uint64_t rhs,
                              const char* name) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64");
  }
  return lhs + rhs;
}

void finish_output(std::ofstream& out,
                   const std::filesystem::path& path) {
  out.flush();
  if (!out) {
    throw std::runtime_error("failed while flushing output: " +
                             path.string());
  }
  out.close();
  if (!out) {
    throw std::runtime_error("failed while closing output: " +
                             path.string());
  }
}

InterchangeArtifactPairId make_artifact_pair_id() {
  static_assert(sizeof(InterchangeArtifactPairId) == 16,
                "artifact pair id layout changed");
  std::ifstream entropy("/dev/urandom", std::ios::binary);
  InterchangeArtifactPairId result;
  entropy.read(reinterpret_cast<char*>(&result), sizeof(result));
  if (!entropy || result.is_zero()) {
    throw std::runtime_error(
        "could not obtain a nonzero artifact pair id from /dev/urandom");
  }
  return result;
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

  const std::size_t bytes = checked_array_bytes<T>(values.size(), name);
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
    const std::size_t bytes =
        checked_array_bytes<char>(text.size(), "metadata string");
    out.write(text.data(), static_cast<std::streamsize>(bytes));
  }
  if (!out) {
    throw std::runtime_error("failed while writing metadata string");
  }
}

// Direct lookup records were resolved once by device_to_routing_graph.  Use
// the design's active site type when present; the low-level lookup permits an
// untyped fallback only if every possible type agrees on one node.
std::optional<NodeId> get_node_from_site_pin(
    const RoutingGraph& graph,
    const ActiveSiteTypes& active_site_types,
    const std::string& site_name,
    const std::string& pin_name) {
  return find_site_pin_node(graph.site_pin_nodes, graph.string_table,
                            site_name, active_site_types.find(site_name),
                            pin_name);
}

// Preserve every sink alias. Multiple physical site pins can intentionally
// resolve to one routing node (for example pinbounce/alternate endpoints), so
// silently retaining only the first name loses reconstruction metadata.
void add_site_pin_attr(RoutingGraph& graph,
                       NodeId node,
                       std::uint64_t site_string,
                       std::uint64_t pin_string) {
  if (node < 0 ||
      static_cast<std::size_t>(node) >= graph.node_device_ids.size()) {
    throw std::runtime_error("site pin node is outside graph");
  }
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

bool route_forest_has_pip(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches) {
  using RouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;
  std::vector<RouteBranch::Reader> queue;
  queue.reserve(branches.size());
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    queue.push_back(branches[i]);
  }
  while (!queue.empty()) {
    const RouteBranch::Reader branch = queue.back();
    queue.pop_back();
    if (branch.getRouteSegment().isPip()) {
      return true;
    }
    const auto children = branch.getBranches();
    for (std::uint32_t i = 0; i < children.size(); ++i) {
      queue.push_back(children[i]);
    }
  }
  return false;
}

bool route_forest_site_pins_are_leaves(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches) {
  using RouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;
  std::vector<RouteBranch::Reader> queue;
  queue.reserve(branches.size());
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    queue.push_back(branches[i]);
  }
  while (!queue.empty()) {
    const RouteBranch::Reader branch = queue.back();
    queue.pop_back();
    const auto children = branch.getBranches();
    if (branch.getRouteSegment().isSitePin() && children.size() != 0) {
      return false;
    }
    for (std::uint32_t i = 0; i < children.size(); ++i) {
      queue.push_back(children[i]);
    }
  }
  return true;
}

bool top_level_stubs_are_site_pins(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader stubs) {
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    if (!stubs[i].getRouteSegment().isSitePin()) {
      return false;
    }
  }
  return true;
}

std::vector<SitePinName> extract_top_level_site_pins(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches,
    TextCache& strings) {
  std::vector<SitePinName> pins;
  pins.reserve(branches.size());
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    const auto segment = branches[i].getRouteSegment();
    if (!segment.isSitePin()) {
      throw std::runtime_error(
          "routable net contains a non-sitePin top-level stub");
    }
    const auto site_pin = segment.getSitePin();
    pins.push_back(
        {strings.get(site_pin.getSite()), strings.get(site_pin.getPin())});
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

bool is_full_device_graph(const RoutingGraph& graph) {
  return graph.bounds.min_x == 0 && graph.bounds.min_y == 0 &&
         graph.bounds.max_x == std::numeric_limits<std::int32_t>::max() &&
         graph.bounds.max_y == std::numeric_limits<std::int32_t>::max();
}

void require_or_count_route_endpoint(bool resolved,
                                     const RoutingGraph& graph,
                                     std::size_t* unresolved_count,
                                     const std::string& description) {
  if (resolved) {
    return;
  }
  ++*unresolved_count;
  if (is_full_device_graph(graph)) {
    throw std::runtime_error("full-device graph cannot resolve " +
                             description);
  }
}

void require_or_count_fixed_resource(bool resolved,
                                     const RoutingGraph& graph,
                                     std::size_t* unresolved_count,
                                     const std::string& description) {
  if (resolved) {
    return;
  }
  ++*unresolved_count;
  if (routing::interchange::unresolved_resource_is_fatal(
          is_full_device_graph(graph))) {
    throw std::runtime_error("full-device graph cannot resolve " +
                             description);
  }
}

struct PhysicalImportStats {
  std::size_t route_requests = 0;
  std::size_t excluded_driverless_nets = 0;
  std::size_t preserved_nets = 0;
  std::size_t unsupported_partial_signal_nets = 0;
  std::size_t unsupported_signal_shape_nets = 0;
  std::size_t unsupported_static_nets = 0;
  std::size_t unresolved_endpoints = 0;
  std::size_t unresolved_fixed_resources = 0;
  std::size_t conservative_fixed_site_pin_fallbacks = 0;
};

using RouteEndpointOwners =
    std::unordered_map<std::int32_t, std::size_t>;

void preserve_unowned_node(RoutingGraph& graph,
                           const RouteEndpointOwners& endpoint_owners,
                           NodeId node,
                           const std::string& preserving_net) {
  const auto owner = endpoint_owners.find(node);
  if (owner != endpoint_owners.end()) {
    throw std::runtime_error(
        "preserved net " + preserving_net +
        " overlaps a routable endpoint owned by physical net index " +
        std::to_string(owner->second));
  }
  routing::interchange::preserve_node(graph.blocked_node, node);
}

void preserve_fixed_site_pin(const std::string& site,
                             const std::string& pin,
                             const ActiveSiteTypes& active_site_types,
                             const RouteEndpointOwners& endpoint_owners,
                             RoutingGraph& graph,
                             const std::string& net_name,
                             PhysicalImportStats* stats) {
  const std::optional<std::string> active_site_type =
      active_site_types.find(site);
  const std::optional<NodeId> exact = find_site_pin_node(
      graph.site_pin_nodes, graph.string_table, site, active_site_type, pin);
  if (exact.has_value()) {
    preserve_unowned_node(graph, endpoint_owners, *exact, net_name);
    return;
  }

  if (!routing::interchange::fixed_site_pin_candidate_fallback_allowed(
          active_site_type.has_value())) {
    require_or_count_fixed_resource(
        false, graph, &stats->unresolved_fixed_resources,
        "typed fixed site pin " + site + "/" + pin + " on net " +
            net_name);
    return;
  }

  // If siteInst metadata is missing, blocking every possible typed alias is
  // conservative. Silently dropping an ambiguous alias would
  // leave a live fixed resource available to a newly routed net.
  const std::vector<NodeId> candidates = find_site_pin_candidates(
      graph.site_pin_nodes, graph.string_table, site, pin);
  if (candidates.empty()) {
    require_or_count_fixed_resource(
        false, graph, &stats->unresolved_fixed_resources,
        "fixed site pin " + site + "/" + pin + " on net " + net_name);
    return;
  }
  for (const NodeId candidate : candidates) {
    preserve_unowned_node(graph, endpoint_owners, candidate, net_name);
  }
  ++stats->conservative_fixed_site_pin_fallbacks;
}

void preserve_route_forest(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches,
    TextCache& strings,
    const ActiveSiteTypes& active_site_types,
    const RouteEndpointOwners& endpoint_owners,
    RoutingGraph& graph,
    const std::string& net_name,
    bool preserve_static_output_pair,
    PhysicalImportStats* stats) {
  using RouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;
  std::vector<RouteBranch::Reader> queue;
  queue.reserve(branches.size());
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    queue.push_back(branches[i]);
  }

  while (!queue.empty()) {
    const RouteBranch::Reader branch = queue.back();
    queue.pop_back();
    const auto segment = branch.getRouteSegment();
    if (segment.isSitePin()) {
      const auto site_pin = segment.getSitePin();
      const std::string& site = strings.get(site_pin.getSite());
      const std::string& pin = strings.get(site_pin.getPin());
      preserve_fixed_site_pin(site, pin, active_site_types, endpoint_owners,
                              graph, net_name, stats);
      if (preserve_static_output_pair) {
        const std::string& device_name =
            graph.string_table.strings[graph.device_name_string];
        const std::optional<std::string> paired_pin =
            routing::interchange::paired_static_slice_output_pin(
                device_name, site, active_site_types.find(site), pin);
        if (paired_pin.has_value()) {
          preserve_fixed_site_pin(site, *paired_pin, active_site_types,
                                  endpoint_owners, graph, net_name, stats);
        }
      }
    } else if (segment.isPip()) {
      const auto pip = segment.getPip();
      const std::string& tile = strings.get(pip.getTile());
      const std::string& wire0 = strings.get(pip.getWire0());
      const std::string& wire1 = strings.get(pip.getWire1());
      const std::optional<NodeId> node0 =
          find_tile_wire_node(graph, tile, wire0);
      const std::optional<NodeId> node1 =
          find_tile_wire_node(graph, tile, wire1);
      require_or_count_fixed_resource(
          node0.has_value(), graph, &stats->unresolved_fixed_resources,
          "fixed PIP endpoint " + tile + "/" + wire0 + " on net " +
              net_name);
      require_or_count_fixed_resource(
          node1.has_value(), graph, &stats->unresolved_fixed_resources,
          "fixed PIP endpoint " + tile + "/" + wire1 + " on net " +
              net_name);
      if (node0.has_value() && node1.has_value()) {
        preserve_unowned_node(graph, endpoint_owners, *node0, net_name);
        preserve_unowned_node(graph, endpoint_owners, *node1, net_name);
      } else {
        if (node0.has_value()) {
          preserve_unowned_node(graph, endpoint_owners, *node0, net_name);
        }
        if (node1.has_value()) {
          preserve_unowned_node(graph, endpoint_owners, *node1, net_name);
        }
      }
    }

    const auto children = branch.getBranches();
    for (std::uint32_t i = 0; i < children.size(); ++i) {
      queue.push_back(children[i]);
    }
  }
}

void preserve_physical_net(
    PhysicalNetlist::PhysNetlist::PhysNet::Reader net,
    TextCache& strings,
    const ActiveSiteTypes& active_site_types,
    const RouteEndpointOwners& endpoint_owners,
    RoutingGraph& graph,
    const std::string& net_name,
    PhysicalImportStats* stats) {
  const bool is_static =
      net.getType() != PhysicalNetlist::PhysNetlist::NetType::SIGNAL;
  preserve_route_forest(net.getSources(), strings, active_site_types,
                        endpoint_owners, graph, net_name, is_static, stats);
  preserve_route_forest(net.getStubs(), strings, active_site_types,
                        endpoint_owners, graph, net_name, is_static, stats);
  const auto stub_nodes = net.getStubNodes();
  for (std::uint32_t i = 0; i < stub_nodes.size(); ++i) {
    const auto stub_node = stub_nodes[i];
    const std::string& tile = strings.get(stub_node.getTile());
    const std::string& wire = strings.get(stub_node.getWire());
    const std::optional<NodeId> node =
        find_tile_wire_node(graph, tile, wire);
    if (node.has_value()) {
      preserve_unowned_node(graph, endpoint_owners, *node, net_name);
    } else {
      require_or_count_fixed_resource(
          false, graph, &stats->unresolved_fixed_resources,
          "fixed stub node " + tile + "/" + wire + " on net " +
              net_name);
    }
  }
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
  std::vector<capnp::word> words = bytes_to_words(bytes, logical_path);
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
  std::unordered_set<std::string> ambiguous_logical_net_names;

  // Walk every logical cell. Each cell summary stores a slice into the flat
  // logical_nets array, keeping metadata compact and easy to stream from disk.
  graph.logical_cells.reserve(cell_list.size());
  for (std::uint32_t cell_index = 0; cell_index < cell_list.size();
       ++cell_index) {
    const auto cell = cell_list[cell_index];
    const std::uint32_t declaration_index = cell.getIndex();
    if (declaration_index >= cell_decls.size()) {
      throw std::runtime_error(
          "LogicalNetlist cell refers to an invalid declaration");
    }

    LogicalCellSummary cell_summary;
    cell_summary.net_begin =
        static_cast<std::uint64_t>(graph.logical_nets.size());
    const auto declaration = cell_decls[declaration_index];
    cell_summary.declaration_name_string =
        graph.string_table.intern(strings.get(declaration.getName()));

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

        if (port_summary.port_index >= port_list.size()) {
          throw std::runtime_error(
              "LogicalNetlist port instance refers to an invalid port");
        }
        const auto port = port_list[port_summary.port_index];
        port_summary.port_string =
            graph.string_table.intern(strings.get(port.getName()));

        const auto bus_idx = port_inst.getBusIdx();
        if (bus_idx.isIdx()) {
          port_summary.has_bus_index = true;
          port_summary.bus_index = bus_idx.getIdx();
        }

        if (port_inst.isInst()) {
          port_summary.instance_index = port_inst.getInst();
          if (port_summary.instance_index >= inst_list.size()) {
            throw std::runtime_error(
                "LogicalNetlist port instance refers to an invalid cell "
                "instance");
          }
          const auto inst = inst_list[port_summary.instance_index];
          port_summary.instance_string =
              graph.string_table.intern(strings.get(inst.getName()));
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

      routing::interchange::index_unambiguous_logical_net_name(
          graph.logical_net_index_by_name, ambiguous_logical_net_names,
          net_name, logical_net_index);
    }

    cell_summary.net_count =
        static_cast<std::uint64_t>(graph.logical_nets.size()) -
        cell_summary.net_begin;
    graph.logical_cells.push_back(cell_summary);
  }
}

PhysicalImportStats parse_physical_netlist(
    const std::filesystem::path& phys_path,
    RoutingGraph& graph) {
  using PhysNetlist = PhysicalNetlist::PhysNetlist;

  std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(phys_path);
  std::vector<capnp::word> words = bytes_to_words(bytes, phys_path);
  graph.physical_netlist_bytes = std::move(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(words.data(), words.size()), reader_options);
  const auto netlist = reader.getRoot<PhysNetlist>();
  TextCache strings(netlist.getStrList());

  const std::string physical_part = capnp_text_to_string(netlist.getPart());
  const std::string& device_name =
      graph.string_table.strings[graph.device_name_string];
  if (!routing::interchange::physical_part_matches_device(device_name,
                                                           physical_part)) {
    throw std::runtime_error("PhysicalNetlist part " + physical_part +
                             " does not match cached device " +
                             device_name);
  }

  ActiveSiteTypes active_site_types;
  const auto site_instances = netlist.getSiteInsts();
  for (std::uint32_t index = 0; index < site_instances.size(); ++index) {
    const auto site_instance = site_instances[index];
    active_site_types.insert(strings.get(site_instance.getSite()),
                             strings.get(site_instance.getType()));
  }

  const auto phys_nets = netlist.getPhysNets();
  graph.route_requests.reserve(phys_nets.size());
  PhysicalImportStats stats;
  RouteEndpointOwners endpoint_owners;
  std::unordered_map<std::string, std::size_t> physical_net_name_owners;
  auto claim_endpoint = [&](NodeId node, std::size_t owner,
                            const std::string& net_name) {
    if (graph.blocked_node[static_cast<std::size_t>(node)] != 0) {
      throw std::runtime_error("routable endpoint on net " + net_name +
                               " overlaps a preserved resource");
    }
    const auto claim = routing::interchange::claim_route_endpoint(
        endpoint_owners, node, owner);
    if (claim == routing::interchange::EndpointClaim::kDifferentOwner) {
      throw std::runtime_error(
          "routable endpoint node is claimed by multiple physical nets; "
          "second claimant is " + net_name);
    }
  };
  for (std::uint32_t net_index = 0; net_index < phys_nets.size();
       ++net_index) {
    const auto net = phys_nets[net_index];
    const std::string& physical_net_name = strings.get(net.getName());
    if (!routing::interchange::claim_unique_physical_net_name(
            physical_net_name_owners, physical_net_name, net_index)) {
      throw std::runtime_error("duplicate PhysicalNetlist net name: " +
                               physical_net_name);
    }
    const auto sources = net.getSources();
    const auto stubs = net.getStubs();
    const std::vector<SitePinName> source_pins =
        extract_site_pins(sources, strings);
    routing::interchange::PhysicalNetRoutingFacts facts;
    facts.is_signal = net.getType() == PhysNetlist::NetType::SIGNAL;
    facts.top_level_source_count = sources.size();
    facts.top_level_stub_count = stubs.size();
    facts.source_site_pin_count = source_pins.size();
    facts.source_site_pins_are_leaves =
        route_forest_site_pins_are_leaves(sources);
    facts.has_inter_site_pip =
        route_forest_has_pip(sources) || route_forest_has_pip(stubs);
    facts.has_stub_nodes = net.getStubNodes().size() != 0;
    facts.top_level_stubs_are_site_pins =
        top_level_stubs_are_site_pins(stubs);
    const auto disposition =
        routing::interchange::classify_physical_net(facts);

    if (disposition !=
        routing::interchange::PhysicalNetDisposition::kRouteSignal) {
      preserve_physical_net(net, strings, active_site_types,
                            endpoint_owners, graph, physical_net_name,
                            &stats);
      switch (disposition) {
        case routing::interchange::PhysicalNetDisposition::
            kPreserveCompleteOrLoadless:
          ++stats.preserved_nets;
          break;
        case routing::interchange::PhysicalNetDisposition::
            kExcludeDriverlessSignal:
          ++stats.excluded_driverless_nets;
          std::cout << "excluded_driverless_net: " << physical_net_name
                    << '\n';
          break;
        case routing::interchange::PhysicalNetDisposition::
            kPreserveUnsupportedPartialSignal:
          ++stats.unsupported_partial_signal_nets;
          std::cout << "preserved_unsupported_partial_signal_net: "
                    << physical_net_name << '\n';
          break;
        case routing::interchange::PhysicalNetDisposition::
            kPreserveUnsupportedSignalShape:
          ++stats.unsupported_signal_shape_nets;
          std::cout << "preserved_unsupported_signal_shape_net: "
                    << physical_net_name << '\n';
          break;
        case routing::interchange::PhysicalNetDisposition::
            kPreserveUnsupportedStatic:
          ++stats.unsupported_static_nets;
          std::cout << "preserved_unsupported_static_net: "
                    << physical_net_name << '\n';
          break;
        case routing::interchange::PhysicalNetDisposition::kRouteSignal:
          break;
      }
      continue;
    }

    const std::vector<SitePinName> sink_pins =
        extract_top_level_site_pins(stubs, strings);
    RouteRequest request;
    request.net_string = graph.string_table.intern(physical_net_name);
    const auto logical_match =
        graph.logical_net_index_by_name.find(physical_net_name);
    if (logical_match != graph.logical_net_index_by_name.end()) {
      request.logical_net_index = logical_match->second;
    }
    request.sources.reserve(source_pins.size());
    request.sinks.reserve(sink_pins.size());
    bool has_valid_source = false;
    for (const SitePinName& source_pin : source_pins) {
      SitePinNode source;
      source.node = kInvalidRouteNode;
      source.site_string = graph.string_table.intern(source_pin.site);
      source.pin_string = graph.string_table.intern(source_pin.pin);
      const std::optional<NodeId> source_node =
          get_node_from_site_pin(graph, active_site_types, source_pin.site,
                                 source_pin.pin);
      require_or_count_route_endpoint(
          source_node.has_value(), graph, &stats.unresolved_endpoints,
          "route source " + source_pin.site + "/" + source_pin.pin +
              " on net " + physical_net_name);
      if (source_node.has_value()) {
        source.node = *source_node;
        claim_endpoint(*source_node, net_index, physical_net_name);
        routing::interchange::mark_source_exclusive(
            graph.exclusive_source_nodes, *source_node);
        has_valid_source = true;
      }
      request.sources.push_back(source);
    }
    for (const SitePinName& sink_pin : sink_pins) {
      SitePinNode sink;
      sink.node = kInvalidRouteNode;
      sink.site_string = graph.string_table.intern(sink_pin.site);
      sink.pin_string = graph.string_table.intern(sink_pin.pin);
      const std::optional<NodeId> sink_node =
          get_node_from_site_pin(graph, active_site_types, sink_pin.site,
                                 sink_pin.pin);
      require_or_count_route_endpoint(
          sink_node.has_value(), graph, &stats.unresolved_endpoints,
          "route sink " + sink_pin.site + "/" + sink_pin.pin +
              " on net " + physical_net_name);
      if (sink_node.has_value()) {
        sink.node = *sink_node;
        claim_endpoint(*sink_node, net_index, physical_net_name);
        if (has_valid_source) {
          const bool is_source_of_same_net =
              graph.exclusive_source_nodes[
                  static_cast<std::size_t>(*sink_node)] != 0;
          if (routing::interchange::sink_requires_terminal_row(
                  is_source_of_same_net)) {
            routing::interchange::mark_sink_terminal(
                graph.sink_node_stops, *sink_node);
          }
          add_site_pin_attr(graph, *sink_node, sink.site_string,
                            sink.pin_string);
        } else {
          // A syntactically present source may lie outside a bounded cache.
          // Keep the request so strict routing reports it as unreachable, and
          // fully reserve any in-bounds sinks from use by other nets.
          routing::interchange::preserve_node(graph.blocked_node,
                                              *sink_node);
        }
      }
      request.sinks.push_back(sink);
    }
    graph.route_requests.push_back(std::move(request));
    ++stats.route_requests;
  }
  return stats;
}

CsrGraph make_outgoing_csr(const RoutingGraph& graph) {
  return filter_device_routing_graph(graph, graph.blocked_node,
                                     graph.sink_node_stops,
                                     graph.exclusive_source_nodes);
}

void write_csr_graph(const CsrGraph& graph,
                     const InterchangeArtifactPairId& artifact_pair_id,
                     const std::filesystem::path& output_path) {
  // This binary file intentionally contains only the generic graph structure:
  // rowptr, colind, and unit weights. FPGA routing names and PIP details live
  // in the RIPSIFM1 metadata sidecar.
  if (artifact_pair_id.is_zero()) {
    throw std::runtime_error("cannot write CSR with a zero artifact pair id");
  }
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
  write_u64(out, artifact_pair_id.high, "artifact pair id high");
  write_u64(out, artifact_pair_id.low, "artifact pair id low");
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
  finish_output(out, output_path);
}

void write_metadata(const RoutingGraph& graph,
                    const CsrGraph& csr,
                    const InterchangeArtifactPairId& artifact_pair_id,
                    const std::filesystem::path& metadata_path) {
  // RIPSIFM1 sidecar layout:
  //   char[8] magic
  //   u64 version, orientation, artifact_pair_id_high, artifact_pair_id_low
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
  if (artifact_pair_id.is_zero()) {
    throw std::runtime_error(
        "cannot write metadata with a zero artifact pair id");
  }
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
  write_u64(out, artifact_pair_id.high, "metadata artifact pair id high");
  write_u64(out, artifact_pair_id.low, "metadata artifact pair id low");
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
  finish_output(out, metadata_path);
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
    const std::filesystem::path csr_publication_marker_path =
        routing::interchange::interchange_publication_marker_path(
            options.output_path);
    const std::filesystem::path metadata_publication_marker_path =
        routing::interchange::interchange_publication_marker_path(
            options.metadata_path);
    const std::filesystem::path publication_generation_path =
        routing::interchange::interchange_publication_generation_path(
            options.metadata_path);
    routing::interchange::require_distinct_interchange_paths(
        {options.device_graph_path, options.phys_path,
         options.logical_path, options.output_path, options.metadata_path,
         csr_publication_marker_path, metadata_publication_marker_path,
         publication_generation_path});
    for (const std::filesystem::path& publication_marker_path :
         {csr_publication_marker_path, metadata_publication_marker_path}) {
      std::error_code marker_error;
      const bool marker_exists =
          std::filesystem::exists(publication_marker_path, marker_error);
      if (marker_error) {
        throw std::runtime_error(
            "could not inspect interchange publication marker: " +
            marker_error.message());
      }
      if (marker_exists) {
        throw std::runtime_error(
            "an interrupted interchange publication marker requires "
            "inspection before conversion: " +
            publication_marker_path.string());
      }
    }
    for (const std::filesystem::path& output :
         {options.output_path, options.metadata_path,
          publication_generation_path}) {
      std::error_code error;
      const bool exists = std::filesystem::exists(output, error);
      if (error) {
        throw std::runtime_error("could not inspect output path: " +
                                 output.string() + ": " + error.message());
      }
      if (exists && !std::filesystem::is_regular_file(output, error)) {
        throw std::runtime_error("output path is not a regular file: " +
                                 output.string());
      }
      if (error) {
        throw std::runtime_error("could not inspect output path type: " +
                                 output.string() + ": " + error.message());
      }
    }

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
    const PhysicalImportStats import_stats =
        parse_physical_netlist(options.phys_path, graph);
    std::cout << "route_requests: " << import_stats.route_requests << "\n";
    std::cout << "excluded_driverless_nets: "
              << import_stats.excluded_driverless_nets << "\n";
    std::cout << "preserved_nets: " << import_stats.preserved_nets << "\n";
    std::cout << "preserved_unsupported_partial_signal_nets: "
              << import_stats.unsupported_partial_signal_nets << "\n";
    std::cout << "preserved_unsupported_signal_shape_nets: "
              << import_stats.unsupported_signal_shape_nets << "\n";
    std::cout << "preserved_unsupported_static_nets: "
              << import_stats.unsupported_static_nets << "\n";
    std::cout << "unresolved_route_endpoints: "
              << import_stats.unresolved_endpoints << "\n";
    std::cout << "unresolved_fixed_resources: "
              << import_stats.unresolved_fixed_resources << "\n";
    std::cout << "conservative_fixed_site_pin_fallbacks: "
              << import_stats.conservative_fixed_site_pin_fallbacks << "\n";

    const std::size_t unsupported_net_count =
        import_stats.unsupported_partial_signal_nets +
        import_stats.unsupported_signal_shape_nets +
        import_stats.unsupported_static_nets;
    if (unsupported_net_count != 0 &&
        !options.allow_unsupported_preserved_nets) {
      throw std::runtime_error(
          "PhysicalNetlist contains " +
          std::to_string(unsupported_net_count) +
          " unsupported net(s); no CSR was written. Use "
          "--allow-unsupported-preserved-nets only to preserve those nets "
          "unchanged for diagnostics");
    }

    // These indexes exist only to resolve names while parsing the two design
    // netlists. Release them before allocating the filtered CSR; on a full
    // device they otherwise overlap several other multi-gigabyte arrays.
    release_storage(graph.tile_wire_nodes);
    release_storage(graph.site_pin_nodes);
    release_storage(graph.string_table.ids);
    release_storage(graph.logical_net_index_by_name);

    // CSR is the GPU-facing graph. Metadata is the CPU-facing FPGA context
    // needed to map CSR edges back to tile/wire PIPs and site-pin targets.
    CsrGraph csr = make_outgoing_csr(graph);

    // Filtering has copied every retained destination and edge attribute.
    // Drop the immutable base CSR before serializing either design output.
    release_storage(graph.rowptr);
    release_storage(graph.colind);
    release_storage(graph.edge_attrs);
    release_storage(graph.exclusive_source_nodes);

    const std::uint64_t rowptr_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int64_t>(csr.rowptr.size(),
                                          "CSR row pointers"));
    const std::uint64_t colind_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int32_t>(csr.colind.size(),
                                          "CSR destinations"));
    const std::uint64_t values_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<float>(csr.values.size(), "CSR values"));
    const std::uint64_t attr_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<EdgeAttr>(csr.edge_attrs.size(),
                                      "metadata edge attributes"));
    const std::uint64_t csr_bytes = checked_add_u64(
        checked_add_u64(rowptr_bytes, colind_bytes, "CSR byte count"),
        values_bytes, "CSR byte count");
    const std::int64_t csr_rows = csr.rows;
    const std::size_t csr_nnz = csr.values.size();

    const std::filesystem::path staged_csr_path =
        routing::interchange::create_unique_staging_path(
            options.output_path);
    std::filesystem::path staged_metadata_path;
    std::filesystem::path staged_generation_path;
    try {
      staged_metadata_path =
          routing::interchange::create_unique_staging_path(
              options.metadata_path);
      staged_generation_path =
          routing::interchange::create_unique_staging_path(
              publication_generation_path);
      const InterchangeArtifactPairId artifact_pair_id =
          make_artifact_pair_id();
      write_csr_graph(csr, artifact_pair_id, staged_csr_path);

      // The metadata sidecar needs edge attributes, but not the three generic
      // CSR arrays that were just written.
      release_storage(csr.rowptr);
      release_storage(csr.colind);
      release_storage(csr.values);
      write_metadata(graph, csr, artifact_pair_id, staged_metadata_path);
      std::ofstream generation(staged_generation_path,
                               std::ios::binary | std::ios::trunc);
      if (!generation) {
        throw std::runtime_error(
            "could not open interchange publication generation: " +
            staged_generation_path.string());
      }
      generation << routing::interchange::interchange_artifact_pair_id_string(
                        artifact_pair_id)
                 << '\n';
      finish_output(generation, staged_generation_path);
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove(staged_csr_path, ignored);
      if (!staged_metadata_path.empty()) {
        std::filesystem::remove(staged_metadata_path, ignored);
      }
      if (!staged_generation_path.empty()) {
        std::filesystem::remove(staged_generation_path, ignored);
      }
      throw;
    }

    std::vector<std::filesystem::path> publication_marker_paths = {
        csr_publication_marker_path, metadata_publication_marker_path};
    std::sort(publication_marker_paths.begin(),
              publication_marker_paths.end(),
              [](const std::filesystem::path& lhs,
                 const std::filesystem::path& rhs) {
                return routing::interchange::normalized_interchange_path(lhs)
                           .string() <
                       routing::interchange::normalized_interchange_path(rhs)
                           .string();
              });
    std::vector<std::filesystem::path> created_publication_markers;
    bool csr_published = false;
    bool metadata_published = false;
    bool generation_published = false;
    bool publication_complete = false;
    try {
      for (const std::filesystem::path& publication_marker_path :
           publication_marker_paths) {
        std::error_code marker_error;
        const bool marker_created =
            std::filesystem::create_directory(publication_marker_path,
                                              marker_error);
        if (!marker_created || marker_error) {
          throw std::runtime_error(
              "another interchange publication is active, or an interrupted "
              "publication marker requires inspection: " +
              publication_marker_path.string());
        }
        created_publication_markers.push_back(publication_marker_path);
      }
      std::filesystem::rename(staged_csr_path, options.output_path);
      csr_published = true;
      std::filesystem::rename(staged_metadata_path,
                              options.metadata_path);
      metadata_published = true;
      std::filesystem::rename(staged_generation_path,
                              publication_generation_path);
      generation_published = true;
      publication_complete = true;
      for (const std::filesystem::path& publication_marker_path :
           created_publication_markers) {
        std::error_code marker_error;
        if (!std::filesystem::remove(publication_marker_path, marker_error) ||
            marker_error) {
          throw std::runtime_error(
              "could not clear interchange publication marker: " +
              publication_marker_path.string());
        }
      }
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove(staged_csr_path, ignored);
      std::filesystem::remove(staged_metadata_path, ignored);
      std::filesystem::remove(staged_generation_path, ignored);

      // Once all three files are in place they are a coherent generation. If
      // marker cleanup itself fails, retain the remaining marker so readers
      // fail closed and leave the valid pair for manual inspection. For a
      // partial publication, remove every exposed new file and clear markers
      // only if every removal succeeded.
      if (!publication_complete) {
        bool exposed_files_removed = true;
        const auto remove_published = [&](bool published,
                                          const std::filesystem::path& path) {
          if (!published) {
            return;
          }
          std::error_code remove_error;
          (void)std::filesystem::remove(path, remove_error);
          if (remove_error) {
            exposed_files_removed = false;
          }
        };
        remove_published(csr_published, options.output_path);
        remove_published(metadata_published, options.metadata_path);
        remove_published(generation_published, publication_generation_path);
        if (exposed_files_removed) {
          for (const std::filesystem::path& publication_marker_path :
               created_publication_markers) {
            std::error_code marker_error;
            (void)std::filesystem::remove(publication_marker_path,
                                          marker_error);
          }
        }
      }
      throw;
    }

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
