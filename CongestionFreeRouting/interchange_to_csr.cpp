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
// and source site pins plus the minimal logical-net name/index correlation
// used by diagnostics. That metadata is intentionally CPU-side: the GPU CSR
// remains compact and compatible with the routing kernels, while later
// post-processing reloads the original .phys input to turn SSSP paths into
// PhysPIP route branches.
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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace {

// Keep the on-disk binary formats explicit. The CSR constants identify the
// generic graph payload; the metadata constants identify the sidecar that
// stores FPGA-specific information not present in plain CSR.
static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

struct PipDataDisk {
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  std::uint32_t forward = 0;
};

static_assert(sizeof(PipDataDisk) == 3 * sizeof(std::uint32_t),
              "PipDataDisk metadata layout changed");

// Metadata v7+ sparse endpoint-owned pseudo-PIP record. Keep this layout in
// lockstep with pathfinder.cpp and routes_to_phys.cpp.
struct EndpointPipDisk {
  std::uint64_t csr_edge = 0;
  std::uint64_t from = 0;
  std::uint64_t to = 0;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
  std::uint64_t site_string = 0;
  std::uint64_t endpoint_node = 0;
  std::uint64_t role = 0;
};

static_assert(sizeof(EndpointPipDisk) == 10 * sizeof(std::uint64_t),
              "EndpointPipDisk metadata layout changed");

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t CSR_IMPLICIT_UNIT_NO_SHARD_VERSION = 4;
constexpr std::uint64_t CSR_FORMAT_VERSION =
    CSR_IMPLICIT_UNIT_NO_SHARD_VERSION;
// Metadata v8 retains the v7 endpoint-PIP policy while compacting graph-sized
// records and dropping payloads/summaries unused by active readers. CSR v4 is
// versioned independently.
constexpr std::uint64_t METADATA_FORMAT_VERSION = 8;
constexpr std::uint64_t OUTGOING_EDGE_ORIENTATION = 2;
using routing::interchange::CsrGraph;
using routing::interchange::DeviceRoutingGraph;
using routing::interchange::DeviceRoutingGraphReadTelemetry;
using routing::interchange::EdgeAttr;
using routing::interchange::EndpointAttachment;
using routing::interchange::EndpointAttachmentRole;
using routing::interchange::FixedEndpointAttachmentIndex;
using routing::interchange::FixedEndpointAttachmentSiteSlice;
using routing::interchange::InterchangeArtifactPairId;
using routing::interchange::NodeId;
using routing::interchange::PipData;
using routing::interchange::RoutingCsrSidecars;
using routing::interchange::SpatialEdgeShards;
using routing::interchange::StringTable;
using routing::interchange::attachment_traversed_site_type_is_compatible;
using routing::interchange::device_routing_graph_node_count;
using routing::interchange::filter_device_routing_graph;
using routing::interchange::find_endpoint_attachment_index;
using routing::interchange::find_pair_node;
using routing::interchange::find_site_pin_candidates;
using routing::interchange::find_site_pin_node;
using routing::interchange::kInvalidRouteNode;
using routing::interchange::kNoIndex;
using routing::interchange::kNoLogicalNetIndex;
using routing::interchange::kNoStringIndex;
using routing::interchange::node_bounds_mode_name;
using routing::interchange::read_device_routing_graph_for_routing;
using routing::interchange::read_gzip_or_plain_chunks;
using routing::interchange::validate_destination_spatial_edge_shards;
using routing::interchange::validate_routing_csr_sidecars;

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
  // Devicegraph attachment identity during import, remapped to the sparse v7
  // EndpointPip table after final CSR filtering.
  std::uint64_t endpoint_attachment_index = kNoIndex;
  std::uint64_t endpoint_pip_index = kNoIndex;
};

struct EndpointPipMetadata {
  std::uint64_t csr_edge = 0;
  NodeId from = kInvalidRouteNode;
  NodeId to = kInvalidRouteNode;
  std::uint64_t tile_string = kNoStringIndex;
  std::uint64_t wire0_string = kNoStringIndex;
  std::uint64_t wire1_string = kNoStringIndex;
  bool forward = true;
  std::uint64_t site_string = kNoStringIndex;
  NodeId endpoint_node = kInvalidRouteNode;
  EndpointAttachmentRole role = EndpointAttachmentRole::kSource;
};

// A design-specific route request extracted from a PhysicalNetlist. Sources
// and sinks are stored as graph node IDs plus their human-readable site pins.
struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

// Static device fields are loaded from the preprocessed artifact. This derived
// structure adds only benchmark-specific masks, requests, minimal logical-net
// correlation, and provenance.
struct RoutingGraph : DeviceRoutingGraph {
  explicit RoutingGraph(DeviceRoutingGraph&& device_graph)
      : DeviceRoutingGraph(std::move(device_graph)) {
    const std::size_t node_count = device_routing_graph_node_count(*this);
    blocked_node.assign(node_count, 0);
    // With a shared immutable CSR, terminal rows are the conservative guard
    // against one net traversing another net's exclusive sink site pin.  This
    // matches the contest POC.  RWRoute can make a narrower same-net pinbounce
    // exception because its traversal carries connection ownership and intent;
    // this graph currently cannot.
    sink_node_stops.assign(node_count, 0);
    unavailable_destination_nodes.assign(node_count, 0);
    enabled_endpoint_attachments.assign(endpoint_attachments.size(), 0);
  }

  std::vector<std::uint8_t> blocked_node;
  std::vector<std::uint8_t> sink_node_stops;
  // Exclusive route sources while physical endpoints are being validated.
  // Blocked resources are ORed into this mask exactly once before final CSR
  // filtering, when source/sink overlap checks no longer need to distinguish
  // the two policies. The filter then needs one random destination-mask read.
  std::vector<std::uint8_t> unavailable_destination_nodes;
  std::vector<std::uint8_t> enabled_endpoint_attachments;
  std::vector<SitePinNode> site_pin_attrs;
  std::vector<RouteRequest> route_requests;
  std::vector<EndpointPipMetadata> endpoint_pips;

  std::uint64_t physical_path_string = 0;
  std::uint64_t logical_path_string = 0;
  std::uint64_t logical_design_name_string = 0;
  std::vector<std::uint64_t> logical_net_name_strings;
  std::unordered_map<std::string, std::uint64_t> logical_net_index_by_name;
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
  void reserve(std::size_t site_count) {
    type_by_site_.reserve(site_count);
  }

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

// Pseudo-cell pins are design resources, not merely graph edges. Placements
// and fixed site routing claim an entire BEL; an attachment claims each exact
// BEL/pin listed by DeviceResources. Different owners may not share an exact
// pin or overlap an entire-BEL claim. This keeps the audited IOB transition
// legal without globally enabling arbitrary site route-throughs.
class AttachmentResourceClaims {
 public:
  // Placements and their already-routed fixed BEL/site-PIP segments describe
  // one pre-existing design occupancy state and may legitimately overlap.
  // Keep them under one sentinel owner; newly enabled attachments retain
  // their physical-net owner so they still conflict with existing occupancy
  // and with attachments belonging to other nets.
  static constexpr std::size_t kExistingOccupancyOwner =
      std::numeric_limits<std::size_t>::max();

  void claim_placement_bel(const std::string& site,
                           const std::string& bel,
                           const std::string& description) {
    claim_whole_bel(site, bel, kExistingOccupancyOwner, description);
  }

  void claim_whole_bel(const std::string& site,
                       const std::string& bel,
                       std::size_t owner,
                       const std::string& description) {
    const std::string bel_key = make_key(site, bel);
    const auto whole = whole_bel_owners_.find(bel_key);
    if (whole != whole_bel_owners_.end() && whole->second != owner) {
      throw std::runtime_error(description +
                               " conflicts with an occupied BEL " + site +
                               "/" + bel);
    }
    const auto pins = pin_owners_by_bel_.find(bel_key);
    if (pins != pin_owners_by_bel_.end()) {
      for (const std::size_t pin_owner : pins->second) {
        if (pin_owner != owner) {
          throw std::runtime_error(description +
                                   " conflicts with an attachment resource " +
                                   site + "/" + bel);
        }
      }
    }
    whole_bel_owners_.emplace(bel_key, owner);
  }

  void claim_pin(const std::string& site,
                 const std::string& bel,
                 const std::string& pin,
                 std::size_t owner,
                 const std::string& description) {
    const std::string bel_key = make_key(site, bel);
    const auto whole = whole_bel_owners_.find(bel_key);
    if (whole != whole_bel_owners_.end() && whole->second != owner) {
      throw std::runtime_error(description +
                               " conflicts with an occupied BEL " + site +
                               "/" + bel);
    }
    const std::string pin_key = make_key(bel_key, pin);
    const auto exact = pin_owners_.find(pin_key);
    if (exact != pin_owners_.end() && exact->second != owner) {
      throw std::runtime_error(description +
                               " conflicts with an owned pseudo-cell pin " +
                               site + "/" + bel + "/" + pin);
    }
    pin_owners_.emplace(pin_key, owner);
    pin_owners_by_bel_[bel_key].insert(owner);
  }

 private:
  static std::string make_key(const std::string& first,
                              const std::string& second) {
    std::string key;
    key.reserve(first.size() + second.size() + 1);
    key.append(first);
    key.push_back('\0');
    key.append(second);
    return key;
  }

  std::unordered_map<std::string, std::size_t> whole_bel_owners_;
  std::unordered_map<std::string, std::size_t> pin_owners_;
  std::unordered_map<std::string, std::unordered_set<std::size_t>>
      pin_owners_by_bel_;
};

bool attachment_allows_traversed_site_type(
    const RoutingGraph& graph,
    const EndpointAttachment& attachment,
    const std::string& active_site_type) {
  const std::uint64_t end =
      attachment.traversed_site_type_begin +
      attachment.traversed_site_type_count;
  if (end < attachment.traversed_site_type_begin ||
      end > graph.endpoint_attachment_traversed_site_types.size()) {
    throw std::runtime_error(
        "endpoint attachment traversed-site-type slice is invalid");
  }
  for (std::uint64_t index = attachment.traversed_site_type_begin;
       index < end; ++index) {
    const std::uint32_t string_index =
        graph.endpoint_attachment_traversed_site_types[
            static_cast<std::size_t>(index)];
    if (string_index >= graph.string_table.strings.size()) {
      throw std::runtime_error(
          "endpoint attachment references an invalid traversed site type");
    }
    if (graph.string_table.strings[string_index] == active_site_type) {
      return true;
    }
  }
  return false;
}

void claim_attachment_resources(
    const RoutingGraph& graph,
    const EndpointAttachment& attachment,
    const ActiveSiteTypes& active_site_types,
    AttachmentResourceClaims& resource_claims,
    std::size_t owner,
    const std::string& description) {
  if (attachment.traversed_site_string >=
      graph.string_table.strings.size()) {
    throw std::runtime_error(
        "endpoint attachment has an invalid traversed site string");
  }
  const std::string& traversed_site =
      graph.string_table.strings[attachment.traversed_site_string];
  const std::optional<std::string> traversed_type =
      active_site_types.find(traversed_site);
  const bool active_type_is_allowed =
      traversed_type.has_value() &&
      attachment_allows_traversed_site_type(graph, attachment,
                                            *traversed_type);
  if (!attachment_traversed_site_type_is_compatible(
          traversed_type.has_value(), active_type_is_allowed)) {
    throw std::runtime_error(
        description + " traverses site " + traversed_site +
        " without a compatible active PhysicalNetlist site type");
  }

  const std::uint64_t pin_end =
      attachment.pseudo_cell_pin_begin + attachment.pseudo_cell_pin_count;
  if (pin_end < attachment.pseudo_cell_pin_begin ||
      pin_end > graph.endpoint_attachment_pseudo_cell_pins.size()) {
    throw std::runtime_error(
        "endpoint attachment pseudo-cell resource slice is invalid");
  }
  for (std::uint64_t index = attachment.pseudo_cell_pin_begin;
       index < pin_end; ++index) {
    const auto& resource = graph.endpoint_attachment_pseudo_cell_pins[
        static_cast<std::size_t>(index)];
    if (resource.bel_string >= graph.string_table.strings.size() ||
        resource.pin_string >= graph.string_table.strings.size()) {
      throw std::runtime_error(
          "endpoint attachment references an invalid pseudo-cell resource");
    }
    resource_claims.claim_pin(
        traversed_site, graph.string_table.strings[resource.bel_string],
        graph.string_table.strings[resource.pin_string], owner,
        description);
  }
}

void claim_endpoint_attachment(RoutingGraph& graph,
                               std::uint32_t attachment_index,
                               EndpointAttachmentRole expected_role,
                               NodeId endpoint_node,
                               const ActiveSiteTypes& active_site_types,
                               AttachmentResourceClaims& resource_claims,
                               std::size_t owner,
                               const std::string& net_name) {
  if (attachment_index >= graph.endpoint_attachments.size()) {
    throw std::runtime_error(
        "endpoint attachment lookup returned an invalid index");
  }
  const EndpointAttachment& attachment =
      graph.endpoint_attachments[attachment_index];
  if (attachment.role != expected_role ||
      attachment.endpoint_node != endpoint_node) {
    throw std::runtime_error(
        "typed route endpoint does not match its attachment corridor");
  }
  claim_attachment_resources(graph, attachment, active_site_types,
                             resource_claims, owner,
                             "IOB attachment on net " + net_name);
  graph.enabled_endpoint_attachments[attachment_index] = 1;
}

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

struct WordAlignedPayload {
  kj::Array<capnp::word> words;
  std::size_t decoded_bytes = 0;
  std::uint64_t peak_allocation_bytes = 0;

  std::size_t word_count() const {
    return decoded_bytes / sizeof(capnp::word);
  }
};

std::uint64_t word_aligned_payload_allocation_bytes(
    const WordAlignedPayload& payload,
    const char* name) {
  if (payload.words.size() >
      std::numeric_limits<std::uint64_t>::max() / sizeof(capnp::word)) {
    throw std::runtime_error(std::string(name) +
                             " allocation bytes overflow uint64");
  }
  const std::uint64_t resident_bytes =
      static_cast<std::uint64_t>(payload.words.size()) * sizeof(capnp::word);
  if (payload.peak_allocation_bytes < resident_bytes) {
    throw std::runtime_error(std::string(name) +
                             " peak allocation accounting is inconsistent");
  }
  return payload.peak_allocation_bytes;
}

// Decode directly into Cap'n Proto-aligned storage. read_gzip_or_plain_chunks
// owns gzip terminal-status and truncation checks; this layer preserves exact
// decoded length and rejects a partial final word instead of accepting padded
// data. zlib also permits already-decompressed input through the same path.
WordAlignedPayload read_word_aligned_payload(
    const std::filesystem::path& path) {
  WordAlignedPayload payload;
  constexpr std::size_t kWordBytes = sizeof(capnp::word);
  read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t byte_count) {
        if (byte_count == 0) {
          return;
        }
        if (byte_count >
            std::numeric_limits<std::size_t>::max() -
                payload.decoded_bytes) {
          throw std::runtime_error("decoded input is too large: " +
                                   path.string());
        }
        const std::size_t old_size = payload.decoded_bytes;
        payload.decoded_bytes += byte_count;
        if (payload.decoded_bytes >
            std::numeric_limits<std::size_t>::max() - (kWordBytes - 1)) {
          throw std::runtime_error(
              "decoded Cap'n Proto word count overflows size_t: " +
              path.string());
        }
        const std::size_t word_count =
            (payload.decoded_bytes + kWordBytes - 1) / kWordBytes;
        constexpr std::size_t kMaximumWordCount =
            std::numeric_limits<std::size_t>::max() / sizeof(capnp::word);
        if (word_count > kMaximumWordCount) {
          throw std::runtime_error(
              "decoded Cap'n Proto word count exceeds host capacity: " +
              path.string());
        }
        if (word_count > payload.words.size()) {
          std::size_t grown_word_count = payload.words.size();
          if (grown_word_count == 0) {
            grown_word_count = word_count;
          } else if (grown_word_count > kMaximumWordCount / 2) {
            grown_word_count = kMaximumWordCount;
          } else {
            grown_word_count *= 2;
          }
          grown_word_count = std::max(grown_word_count, word_count);
          if (payload.words.size() >
              std::numeric_limits<std::size_t>::max() - grown_word_count) {
            throw std::runtime_error(
                "decoded Cap'n Proto peak allocation overflows size_t: " +
                path.string());
          }
          const std::size_t peak_words =
              payload.words.size() + grown_word_count;
          if (peak_words >
              std::numeric_limits<std::uint64_t>::max() / kWordBytes) {
            throw std::runtime_error(
                "decoded Cap'n Proto peak allocation overflows uint64: " +
                path.string());
          }
          payload.peak_allocation_bytes = std::max(
              payload.peak_allocation_bytes,
              static_cast<std::uint64_t>(peak_words) * kWordBytes);
          kj::Array<capnp::word> grown =
              kj::heapArray<capnp::word>(grown_word_count);
          if (old_size != 0) {
            std::memcpy(grown.begin(), payload.words.begin(), old_size);
          }
          payload.words = kj::mv(grown);
        }
        std::memcpy(reinterpret_cast<std::uint8_t*>(payload.words.begin()) +
                        old_size,
                    data, byte_count);
      });

  if (payload.decoded_bytes == 0) {
    throw std::runtime_error("input file is empty: " + path.string());
  }
  if (payload.decoded_bytes % kWordBytes != 0) {
    throw std::runtime_error(
        "decoded Cap'n Proto input is not word-aligned: " + path.string() +
        " has " + std::to_string(payload.decoded_bytes) + " bytes");
  }
  return payload;
}

// Validate signed offsets before writing them into unsigned binary headers.
std::uint64_t as_u64(std::int64_t value, const char* name) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " is negative");
  }
  return static_cast<std::uint64_t>(value);
}

std::uint32_t checked_narrow_u32(std::uint64_t value, const char* name) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string(name) + " exceeds uint32");
  }
  return static_cast<std::uint32_t>(value);
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

using TelemetryClock = std::chrono::steady_clock;

double elapsed_seconds(TelemetryClock::time_point begin,
                       TelemetryClock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

std::uint64_t process_peak_rss_bytes() {
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return 0;
  }
  const std::uint64_t raw = static_cast<std::uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
  return raw;
#else
  if (raw > std::numeric_limits<std::uint64_t>::max() / 1024) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return raw * 1024;
#endif
}

void emit_stage_telemetry(
    const char* stage,
    double seconds,
    std::initializer_list<std::pair<const char*, std::uint64_t>> fields = {}) {
  std::cout << std::setprecision(9)
            << "{\"type\":\"interchange_to_csr_stage\","
               "\"schema_version\":1,\"stage\":\""
            << stage << "\",\"seconds\":" << seconds
            << ",\"peak_rss_bytes\":" << process_peak_rss_bytes();
  for (const auto& field : fields) {
    std::cout << ",\"" << field.first << "\":" << field.second;
  }
  std::cout << "}\n";
}

template <typename T>
std::uint64_t vector_allocation_bytes(const std::vector<T>& values,
                                      const char* name) {
  if (values.capacity() >
      std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
    throw std::runtime_error(std::string(name) +
                             " allocation bytes overflow uint64");
  }
  return static_cast<std::uint64_t>(values.capacity()) * sizeof(T);
}

struct DecodeParseTelemetry {
  double decode_seconds = 0.0;
  double parse_seconds = 0.0;
  std::uint64_t decoded_bytes = 0;
  std::uint64_t decode_peak_allocation_bytes = 0;
};

// Declare this before the decode/parse working sets. Its destructor then runs
// after their cleanup, so parse wall time does not leave a potentially large
// unreported gap between the stage event and its caller.
class DecodeParseTelemetryRecorder {
 public:
  explicit DecodeParseTelemetryRecorder(DecodeParseTelemetry* telemetry)
      : telemetry_(telemetry), decode_begin_(TelemetryClock::now()) {}

  void mark_decode_complete(std::uint64_t decoded_bytes,
                            std::uint64_t peak_allocation_bytes) {
    decode_end_ = TelemetryClock::now();
    decoded_bytes_ = decoded_bytes;
    decode_peak_allocation_bytes_ = peak_allocation_bytes;
    decode_complete_ = true;
  }

  ~DecodeParseTelemetryRecorder() {
    if (telemetry_ == nullptr || !decode_complete_) {
      return;
    }
    const auto parse_end = TelemetryClock::now();
    telemetry_->decode_seconds =
        elapsed_seconds(decode_begin_, decode_end_);
    telemetry_->parse_seconds = elapsed_seconds(decode_end_, parse_end);
    telemetry_->decoded_bytes = decoded_bytes_;
    telemetry_->decode_peak_allocation_bytes =
        decode_peak_allocation_bytes_;
  }

 private:
  DecodeParseTelemetry* telemetry_ = nullptr;
  TelemetryClock::time_point decode_begin_;
  TelemetryClock::time_point decode_end_;
  std::uint64_t decoded_bytes_ = 0;
  std::uint64_t decode_peak_allocation_bytes_ = 0;
  bool decode_complete_ = false;
};

struct CsrBuildTelemetry {
  double filtering_seconds = 0.0;
  double endpoint_binding_seconds = 0.0;
  double spatial_sidecar_seconds = 0.0;
  std::uint64_t filtering_output_allocation_bytes = 0;
  std::uint64_t endpoint_binding_allocation_bytes = 0;
  std::uint64_t spatial_sidecar_allocation_bytes = 0;
};

struct MetadataWriteTelemetry {
  std::uint64_t packing_allocation_bytes = 0;
};

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

std::optional<std::uint32_t> get_endpoint_attachment(
    const RoutingGraph& graph,
    const ActiveSiteTypes& active_site_types,
    const std::string& site_name,
    const std::string& pin_name,
    EndpointAttachmentRole role) {
  const std::optional<std::string> active_type =
      active_site_types.find(site_name);
  if (!active_type.has_value()) {
    return std::nullopt;
  }
  return find_endpoint_attachment_index(
      graph.endpoint_attachment_lookups, graph.string_table, site_name,
      *active_type, pin_name, role);
}

// Preserve every sink alias. Multiple physical site pins can intentionally
// resolve to one routing node (for example pinbounce/alternate endpoints), so
// silently retaining only the first name loses reconstruction metadata.
void add_site_pin_attr(RoutingGraph& graph,
                       NodeId node,
                       std::uint64_t site_string,
                       std::uint64_t pin_string) {
  if (node < 0 ||
      static_cast<std::size_t>(node) >=
          device_routing_graph_node_count(graph)) {
    throw std::runtime_error("site pin node is outside graph");
  }
  SitePinNode attr;
  attr.node = node;
  attr.site_string = site_string;
  attr.pin_string = pin_string;
  graph.site_pin_attrs.push_back(attr);
}

using PhysicalRouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;
using RouteBranchStack = std::vector<PhysicalRouteBranch::Reader>;

void seed_route_branch_stack(
    capnp::List<PhysicalRouteBranch>::Reader branches,
    RouteBranchStack& stack) {
  stack.clear();
  if (stack.capacity() < branches.size()) {
    stack.reserve(branches.size());
  }
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    stack.push_back(branches[i]);
  }
}

struct SourceForestAnalysis {
  bool site_pins_are_leaves = true;
  bool has_pip = false;
};

// Collect source pins and the two source-forest classification facts in one
// LIFO traversal. The previous correctness checks walked the same forest three
// times; retaining the original root/child push order preserves serialized
// source order exactly.
SourceForestAnalysis analyze_source_forest(
    capnp::List<PhysicalRouteBranch>::Reader branches,
    TextCache& strings,
    std::vector<SitePinName>& pins,
    RouteBranchStack& stack) {
  pins.clear();
  if (pins.capacity() < branches.size()) {
    pins.reserve(branches.size());
  }
  seed_route_branch_stack(branches, stack);

  SourceForestAnalysis analysis;
  while (!stack.empty()) {
    const PhysicalRouteBranch::Reader branch = stack.back();
    stack.pop_back();

    const auto segment = branch.getRouteSegment();
    const auto children = branch.getBranches();
    if (segment.isSitePin()) {
      const auto site_pin = segment.getSitePin();
      pins.push_back(
          {strings.get(site_pin.getSite()), strings.get(site_pin.getPin())});
      if (children.size() != 0) {
        analysis.site_pins_are_leaves = false;
      }
    } else if (segment.isPip()) {
      analysis.has_pip = true;
    }

    for (std::uint32_t i = 0; i < children.size(); ++i) {
      stack.push_back(children[i]);
    }
  }

  return analysis;
}

bool route_forest_has_pip(
    capnp::List<PhysicalRouteBranch>::Reader branches,
    RouteBranchStack& stack) {
  seed_route_branch_stack(branches, stack);
  while (!stack.empty()) {
    const PhysicalRouteBranch::Reader branch = stack.back();
    stack.pop_back();
    if (branch.getRouteSegment().isPip()) {
      return true;
    }
    const auto children = branch.getBranches();
    for (std::uint32_t i = 0; i < children.size(); ++i) {
      stack.push_back(children[i]);
    }
  }
  return false;
}

struct SitePinStringIndices {
  std::uint32_t site = 0;
  std::uint32_t pin = 0;
};

// Validate the routable-stub shape and retain its site/pin indices in one
// Cap'n Proto list traversal. Decode names only after every top-level branch
// has passed the shape check. That preserves the old failure behavior for an
// unsupported non-sitePin stub while avoiding a second branch-list scan for a
// routable net.
bool analyze_top_level_site_pin_stubs(
    capnp::List<PhysicalRouteBranch>::Reader branches,
    TextCache& strings,
    std::vector<SitePinStringIndices>& pin_indices,
    std::vector<SitePinName>& pins) {
  pin_indices.clear();
  pins.clear();
  if (pin_indices.capacity() < branches.size()) {
    pin_indices.reserve(branches.size());
  }
  if (pins.capacity() < branches.size()) {
    pins.reserve(branches.size());
  }
  for (std::uint32_t i = 0; i < branches.size(); ++i) {
    const auto segment = branches[i].getRouteSegment();
    if (!segment.isSitePin()) {
      return false;
    }
    const auto site_pin = segment.getSitePin();
    pin_indices.push_back({site_pin.getSite(), site_pin.getPin()});
  }
  for (const SitePinStringIndices& site_pin : pin_indices) {
    pins.push_back(
        {strings.get(site_pin.site), strings.get(site_pin.pin)});
  }
  return true;
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
    capnp::List<PhysicalRouteBranch>::Reader branches,
    TextCache& strings,
    const ActiveSiteTypes& active_site_types,
    const RouteEndpointOwners& endpoint_owners,
    RoutingGraph& graph,
    const FixedEndpointAttachmentIndex& fixed_attachment_index,
    AttachmentResourceClaims& attachment_resource_claims,
    std::size_t owner,
    const std::string& net_name,
    bool preserve_static_output_pair,
    PhysicalImportStats* stats,
    RouteBranchStack& stack) {
  seed_route_branch_stack(branches, stack);

  while (!stack.empty()) {
    const PhysicalRouteBranch::Reader branch = stack.back();
    stack.pop_back();
    const auto segment = branch.getRouteSegment();
    if (segment.isBelPin()) {
      const auto bel_pin = segment.getBelPin();
      attachment_resource_claims.claim_whole_bel(
          strings.get(bel_pin.getSite()), strings.get(bel_pin.getBel()),
          owner, "fixed BEL pin on net " + net_name);
    } else if (segment.isSitePin()) {
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
      if (pip.isSite()) {
        const std::string& site = strings.get(pip.getSite());
        const std::optional<std::uint32_t> attachment =
            routing::interchange::find_fixed_endpoint_attachment(
                fixed_attachment_index, graph.string_table, tile, wire0,
                wire1, pip.getForward(), site);
        if (attachment.has_value()) {
          claim_attachment_resources(
              graph, graph.endpoint_attachments[*attachment],
              active_site_types, attachment_resource_claims, owner,
              "fixed IOB attachment on net " + net_name);
        } else {
          // Unsupported fixed pseudo route-throughs stay preserved and absent
          // from the routable CSR. Conservatively reserve every audited
          // attachment resource at the same traversed site so newly routed
          // nets cannot overlap the fixed site's unknown pseudo resources.
          const FixedEndpointAttachmentSiteSlice slice =
              routing::interchange::find_fixed_endpoint_attachment_site_slice(
                  fixed_attachment_index, graph.string_table, site);
          if (slice.attachment_begin >
                  fixed_attachment_index.attachments_by_traversed_site.size() ||
              slice.attachment_count >
                  fixed_attachment_index.attachments_by_traversed_site.size() -
                      slice.attachment_begin) {
            throw std::runtime_error(
                "fixed endpoint attachment site slice is invalid");
          }
          const std::size_t slice_end =
              slice.attachment_begin + slice.attachment_count;
          for (std::size_t offset = slice.attachment_begin;
               offset < slice_end; ++offset) {
            const std::uint32_t candidate_index =
                fixed_attachment_index.attachments_by_traversed_site[offset];
            if (candidate_index >= graph.endpoint_attachments.size()) {
              throw std::runtime_error(
                  "fixed endpoint attachment site slice is invalid");
            }
            claim_attachment_resources(
                graph, graph.endpoint_attachments[candidate_index],
                active_site_types, attachment_resource_claims, owner,
                "fixed pseudo PIP on net " + net_name);
          }
        }
      }
    } else if (segment.isSitePIP()) {
      const auto site_pip = segment.getSitePIP();
      attachment_resource_claims.claim_whole_bel(
          strings.get(site_pip.getSite()), strings.get(site_pip.getBel()),
          owner, "fixed site PIP on net " + net_name);
    }

    const auto children = branch.getBranches();
    for (std::uint32_t i = 0; i < children.size(); ++i) {
      stack.push_back(children[i]);
    }
  }
}

void preserve_physical_net(
    PhysicalNetlist::PhysNetlist::PhysNet::Reader net,
    TextCache& strings,
    const ActiveSiteTypes& active_site_types,
    const RouteEndpointOwners& endpoint_owners,
    RoutingGraph& graph,
    const FixedEndpointAttachmentIndex& fixed_attachment_index,
    AttachmentResourceClaims& attachment_resource_claims,
    std::size_t owner,
    const std::string& net_name,
    PhysicalImportStats* stats,
    RouteBranchStack& stack) {
  const bool is_static =
      net.getType() != PhysicalNetlist::PhysNetlist::NetType::SIGNAL;
  preserve_route_forest(net.getSources(), strings, active_site_types,
                        endpoint_owners, graph, fixed_attachment_index,
                        attachment_resource_claims, owner, net_name,
                        is_static, stats, stack);
  preserve_route_forest(net.getStubs(), strings, active_site_types,
                        endpoint_owners, graph, fixed_attachment_index,
                        attachment_resource_claims, owner, net_name,
                        is_static, stats, stack);
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
                           RoutingGraph& graph,
                           DecodeParseTelemetry* telemetry) {
  using Netlist = LogicalNetlist::Netlist;

  // LogicalNetlist is design connectivity: logical cells, nets, ports, and
  // port instances. Metadata v8 retains only the ordered logical-net names,
  // but this parser still traverses and validates the hierarchy references.
  DecodeParseTelemetryRecorder telemetry_recorder(telemetry);
  WordAlignedPayload payload = read_word_aligned_payload(logical_path);
  const std::uint64_t decoded_bytes =
      static_cast<std::uint64_t>(payload.decoded_bytes);
  const std::uint64_t decode_peak_allocation_bytes =
      word_aligned_payload_allocation_bytes(
          payload, "logical Cap'n Proto words");
  telemetry_recorder.mark_decode_complete(decoded_bytes,
                                          decode_peak_allocation_bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.begin(), payload.word_count()),
      reader_options);
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

  // Walk every logical cell and validate the same declaration/port/instance
  // references as older writers without allocating hierarchy summary tables.
  for (std::uint32_t cell_index = 0; cell_index < cell_list.size();
       ++cell_index) {
    const auto cell = cell_list[cell_index];
    const std::uint32_t declaration_index = cell.getIndex();
    if (declaration_index >= cell_decls.size()) {
      throw std::runtime_error(
          "LogicalNetlist cell refers to an invalid declaration");
    }
    const auto declaration = cell_decls[declaration_index];
    (void)strings.get(declaration.getName());

    // Physical net names usually match logical net names. Preserve their flat
    // deterministic order so a route request can retain one compact index.
    const auto nets = cell.getNets();
    for (std::uint32_t net_index = 0; net_index < nets.size(); ++net_index) {
      const auto net = nets[net_index];
      const std::string& net_name = strings.get(net.getName());
      const std::uint64_t name_string =
          graph.string_table.intern(net_name);

      // These records are intentionally not serialized in v8, but walking
      // them preserves invalid-port/instance/string rejection.
      const auto port_insts = net.getPortInsts();
      for (std::uint32_t port_inst_index = 0;
           port_inst_index < port_insts.size();
           ++port_inst_index) {
        const auto port_inst = port_insts[port_inst_index];
        const std::uint32_t port_index = port_inst.getPort();
        if (port_index >= port_list.size()) {
          throw std::runtime_error(
              "LogicalNetlist port instance refers to an invalid port");
        }
        (void)strings.get(port_list[port_index].getName());

        const auto bus_idx = port_inst.getBusIdx();
        if (bus_idx.isIdx()) {
          (void)bus_idx.getIdx();
        }

        if (port_inst.isInst()) {
          const std::uint32_t instance_index = port_inst.getInst();
          if (instance_index >= inst_list.size()) {
            throw std::runtime_error(
                "LogicalNetlist port instance refers to an invalid cell "
                "instance");
          }
          (void)strings.get(inst_list[instance_index].getName());
        }
      }

      const std::uint64_t logical_net_index =
          static_cast<std::uint64_t>(graph.logical_net_name_strings.size());
      graph.logical_net_name_strings.push_back(name_string);

      routing::interchange::index_unambiguous_logical_net_name(
          graph.logical_net_index_by_name, ambiguous_logical_net_names,
          net_name, logical_net_index);
    }
  }
}

PhysicalImportStats parse_physical_netlist(
    const std::filesystem::path& phys_path,
    RoutingGraph& graph,
    DecodeParseTelemetry* telemetry) {
  using PhysNetlist = PhysicalNetlist::PhysNetlist;

  DecodeParseTelemetryRecorder telemetry_recorder(telemetry);
  WordAlignedPayload payload = read_word_aligned_payload(phys_path);
  const std::uint64_t decoded_bytes =
      static_cast<std::uint64_t>(payload.decoded_bytes);
  const std::uint64_t decode_peak_allocation_bytes =
      word_aligned_payload_allocation_bytes(
          payload, "physical Cap'n Proto words");
  telemetry_recorder.mark_decode_complete(decoded_bytes,
                                          decode_peak_allocation_bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.begin(), payload.word_count()),
      reader_options);
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
  const FixedEndpointAttachmentIndex fixed_attachment_index =
      routing::interchange::build_fixed_endpoint_attachment_index(graph);

  ActiveSiteTypes active_site_types;
  const auto site_instances = netlist.getSiteInsts();
  active_site_types.reserve(site_instances.size());
  for (std::uint32_t index = 0; index < site_instances.size(); ++index) {
    const auto site_instance = site_instances[index];
    active_site_types.insert(strings.get(site_instance.getSite()),
                             strings.get(site_instance.getType()));
  }

  AttachmentResourceClaims attachment_resource_claims;
  const auto placements = netlist.getPlacements();
  for (std::uint32_t index = 0; index < placements.size(); ++index) {
    const auto placement = placements[index];
    const std::string& site = strings.get(placement.getSite());
    const std::string& bel = strings.get(placement.getBel());
    if (!site.empty() && !bel.empty()) {
      attachment_resource_claims.claim_placement_bel(
          site, bel, "cell placement " + strings.get(placement.getCellName()));
    }
    const auto other_bels = placement.getOtherBels();
    for (std::uint32_t other = 0; other < other_bels.size(); ++other) {
      const std::string& other_bel = strings.get(other_bels[other]);
      if (!site.empty() && !other_bel.empty()) {
        attachment_resource_claims.claim_placement_bel(
            site, other_bel,
            "cell placement " + strings.get(placement.getCellName()));
      }
    }
    const auto pin_map = placement.getPinMap();
    for (std::uint32_t pin = 0; pin < pin_map.size(); ++pin) {
      const std::string& mapped_bel = strings.get(pin_map[pin].getBel());
      if (!site.empty() && !mapped_bel.empty()) {
        attachment_resource_claims.claim_placement_bel(
            site, mapped_bel,
            "cell placement pin map " +
                strings.get(placement.getCellName()));
      }
    }
  }

  const auto phys_nets = netlist.getPhysNets();
  graph.route_requests.reserve(phys_nets.size());
  PhysicalImportStats stats;
  RouteEndpointOwners endpoint_owners;
  std::unordered_map<std::string, std::size_t> physical_net_name_owners;
  endpoint_owners.reserve(phys_nets.size());
  physical_net_name_owners.reserve(phys_nets.size());
  RouteBranchStack route_branch_stack;
  std::vector<SitePinName> source_pins;
  std::vector<SitePinName> sink_pins;
  std::vector<SitePinStringIndices> sink_pin_indices;
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
    routing::interchange::PhysicalNetRoutingFacts facts;
    facts.is_signal = net.getType() == PhysNetlist::NetType::SIGNAL;
    facts.top_level_source_count = sources.size();
    facts.top_level_stub_count = stubs.size();

    // GLOBAL_USEDNET is RapidWright's signal-net occupancy sentinel, not a
    // logical connection. Its arbitrary routing shape is supported here by
    // preserving every represented resource without creating a route request.
    if (routing::interchange::is_reserved_used_resource_net(
            physical_net_name, facts.is_signal)) {
      preserve_physical_net(net, strings, active_site_types,
                            endpoint_owners, graph, fixed_attachment_index,
                            attachment_resource_claims,
                            AttachmentResourceClaims::kExistingOccupancyOwner,
                            physical_net_name, &stats, route_branch_stack);
      ++stats.preserved_nets;
      continue;
    }

    source_pins.clear();
    sink_pins.clear();
    sink_pin_indices.clear();
    routing::interchange::PhysicalNetDisposition disposition;
    // Match classify_physical_net's decision order so nets already known to
    // be preservation-only do not pay to scan large routed forests merely to
    // compute facts that cannot affect their disposition.
    if (!facts.is_signal || facts.top_level_stub_count == 0 ||
        facts.top_level_source_count == 0) {
      disposition = routing::interchange::classify_physical_net(facts);
    } else {
      const SourceForestAnalysis source_analysis = analyze_source_forest(
          sources, strings, source_pins, route_branch_stack);
      facts.source_site_pin_count = source_pins.size();
      facts.source_site_pins_are_leaves =
          source_analysis.site_pins_are_leaves;

      if (facts.source_site_pin_count == 0 ||
          !facts.source_site_pins_are_leaves) {
        disposition = routing::interchange::classify_physical_net(facts);
      } else {
        facts.has_stub_nodes = net.getStubNodes().size() != 0;
        facts.has_inter_site_pip = source_analysis.has_pip;
        if (!facts.has_inter_site_pip && !facts.has_stub_nodes) {
          facts.has_inter_site_pip =
              route_forest_has_pip(stubs, route_branch_stack);
        }
        if (!facts.has_inter_site_pip && !facts.has_stub_nodes) {
          facts.top_level_stubs_are_site_pins =
              analyze_top_level_site_pin_stubs(
                  stubs, strings, sink_pin_indices, sink_pins);
        }
        disposition = routing::interchange::classify_physical_net(facts);
      }
    }

    if (disposition !=
        routing::interchange::PhysicalNetDisposition::kRouteSignal) {
      preserve_physical_net(net, strings, active_site_types,
                            endpoint_owners, graph, fixed_attachment_index,
                            attachment_resource_claims,
                            AttachmentResourceClaims::kExistingOccupancyOwner,
                            physical_net_name, &stats, route_branch_stack);
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

    RouteRequest request;
    request.net_string = graph.string_table.intern(physical_net_name);
    const auto logical_match =
        graph.logical_net_index_by_name.find(physical_net_name);
    if (logical_match != graph.logical_net_index_by_name.end()) {
      request.logical_net_index = logical_match->second;
    }
    request.sources.reserve(source_pins.size());
    request.sinks.reserve(sink_pins.size());
    std::unordered_map<NodeId, std::uint64_t> source_attachment_by_node;
    std::unordered_map<NodeId, std::uint64_t> sink_attachment_by_node;
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
        const std::optional<std::uint32_t> attachment =
            get_endpoint_attachment(
                graph, active_site_types, source_pin.site, source_pin.pin,
                EndpointAttachmentRole::kSource);
        const std::optional<std::string> active_type =
            active_site_types.find(source_pin.site);
        if (active_type.has_value() &&
            routing::interchange::is_audited_iob_endpoint(
                routing::interchange::IobAttachmentRole::kSource,
                source_pin.site, *active_type, source_pin.pin)) {
          require_or_count_route_endpoint(
              attachment.has_value(), graph, &stats.unresolved_endpoints,
              "audited IOB source attachment " + source_pin.site + "/" +
                  source_pin.pin + " on net " + physical_net_name);
        }
        if (attachment.has_value()) {
          claim_endpoint_attachment(
              graph, *attachment, EndpointAttachmentRole::kSource,
              *source_node, active_site_types, attachment_resource_claims,
              net_index, physical_net_name);
          source.endpoint_attachment_index = *attachment;
        }
        const auto alias = source_attachment_by_node.emplace(
            *source_node, source.endpoint_attachment_index);
        if (!alias.second &&
            alias.first->second != source.endpoint_attachment_index) {
          throw std::runtime_error(
              "source aliases on net " + physical_net_name +
              " share a node but require different IOB attachments");
        }
        claim_endpoint(*source_node, net_index, physical_net_name);
        routing::interchange::mark_source_exclusive(
            graph.unavailable_destination_nodes, *source_node);
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
              graph.unavailable_destination_nodes[
                  static_cast<std::size_t>(*sink_node)] != 0;
          if (!is_source_of_same_net) {
            const std::optional<std::uint32_t> attachment =
                get_endpoint_attachment(
                    graph, active_site_types, sink_pin.site, sink_pin.pin,
                    EndpointAttachmentRole::kSink);
            const std::optional<std::string> active_type =
                active_site_types.find(sink_pin.site);
            if (active_type.has_value() &&
                routing::interchange::is_audited_iob_endpoint(
                    routing::interchange::IobAttachmentRole::kSink,
                    sink_pin.site, *active_type, sink_pin.pin)) {
              require_or_count_route_endpoint(
                  attachment.has_value(), graph,
                  &stats.unresolved_endpoints,
                  "audited IOB sink attachment " + sink_pin.site + "/" +
                      sink_pin.pin + " on net " + physical_net_name);
            }
            if (attachment.has_value()) {
              claim_endpoint_attachment(
                  graph, *attachment, EndpointAttachmentRole::kSink,
                  *sink_node, active_site_types,
                  attachment_resource_claims, net_index, physical_net_name);
              sink.endpoint_attachment_index = *attachment;
            }
          }
          const auto alias = sink_attachment_by_node.emplace(
              *sink_node, sink.endpoint_attachment_index);
          if (!alias.second &&
              alias.first->second != sink.endpoint_attachment_index) {
            throw std::runtime_error(
                "sink aliases on net " + physical_net_name +
                " share a node but require different IOB attachments");
          }
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

CsrGraph make_outgoing_csr(RoutingGraph& graph,
                           CsrBuildTelemetry* telemetry) {
  const auto filtering_begin = TelemetryClock::now();
  if (graph.unavailable_destination_nodes.size() !=
      graph.blocked_node.size()) {
    throw std::runtime_error(
        "destination-unavailability mask does not match blocked-node mask");
  }
  for (std::size_t node = 0; node < graph.blocked_node.size(); ++node) {
    graph.unavailable_destination_nodes[node] = static_cast<std::uint8_t>(
        graph.unavailable_destination_nodes[node] |
        graph.blocked_node[node]);
  }
  CsrGraph csr = filter_device_routing_graph(
      graph, graph.blocked_node, graph.sink_node_stops,
      graph.unavailable_destination_nodes,
      graph.enabled_endpoint_attachments);
  const auto filtering_end = TelemetryClock::now();
  const auto endpoint_binding_begin = filtering_end;

  // Bind each enabled devicegraph attachment to its exact retained CSR edge.
  // The resulting sparse table is the only attachment identity exposed to
  // routing/reconstruction; disabled pseudo edges never enter this CSR.
  std::vector<std::uint64_t> endpoint_pip_by_attachment(
      graph.endpoint_attachments.size(), kNoIndex);
  if (csr.edge_attrs.size() != csr.colind.size()) {
    throw std::runtime_error(
        "filtered CSR edge attributes are not aligned with destinations");
  }
  routing::interchange::sort_and_validate_retained_endpoint_attachment_edges(
      csr.retained_endpoint_attachment_edges);
  for (const auto& retained : csr.retained_endpoint_attachment_edges) {
    if (retained.attachment_index >= graph.endpoint_attachments.size() ||
        retained.attachment_index >=
            graph.enabled_endpoint_attachments.size() ||
        graph.enabled_endpoint_attachments[retained.attachment_index] == 0 ||
        retained.csr_edge >= csr.edge_attrs.size()) {
      throw std::runtime_error(
          "filtered endpoint attachment binding is invalid");
    }
    const std::uint32_t attachment_index = retained.attachment_index;
    const std::size_t edge_index =
        static_cast<std::size_t>(retained.csr_edge);
    const EndpointAttachment& attachment =
        graph.endpoint_attachments[attachment_index];
    if (attachment.from_node < 0 || csr.rowptr.empty() ||
        static_cast<std::size_t>(attachment.from_node) >=
            csr.rowptr.size() - 1) {
      throw std::runtime_error(
          "filtered endpoint attachment edge does not match devicegraph");
    }
    const std::size_t row = static_cast<std::size_t>(attachment.from_node);
    const std::int64_t row_begin = csr.rowptr[row];
    const std::int64_t row_end = csr.rowptr[row + 1];
    if (row_begin < 0 || row_end < row_begin ||
        retained.csr_edge < static_cast<std::uint64_t>(row_begin) ||
        retained.csr_edge >= static_cast<std::uint64_t>(row_end)) {
      throw std::runtime_error(
          "filtered endpoint attachment edge does not match devicegraph");
    }
    const EdgeAttr& attr = csr.edge_attrs[edge_index];
    if (csr.colind[edge_index] != attachment.to_node ||
        attr.pip_data_index != attachment.pip_data_index ||
        attr.pip_data_index >= graph.pip_data.size() ||
        attr.tile_string >= graph.string_table.strings.size() ||
        attachment.traversed_site_string >=
            graph.string_table.strings.size()) {
      throw std::runtime_error(
          "filtered endpoint attachment edge does not match devicegraph");
    }
    if (endpoint_pip_by_attachment[attachment_index] != kNoIndex) {
      throw std::runtime_error(
          "filtered CSR contains an endpoint attachment more than once");
    }
    const PipData& pip = graph.pip_data[
        static_cast<std::size_t>(attachment.pip_data_index)];
    if (pip.wire0_string >= graph.string_table.strings.size() ||
        pip.wire1_string >= graph.string_table.strings.size()) {
      throw std::runtime_error(
          "endpoint attachment PIP references invalid wire strings");
    }
    const std::uint64_t endpoint_pip_index = graph.endpoint_pips.size();
    graph.endpoint_pips.push_back(
        {retained.csr_edge,
         attachment.from_node,
         attachment.to_node,
         attr.tile_string,
         pip.wire0_string,
         pip.wire1_string,
         pip.forward,
         attachment.traversed_site_string,
         attachment.endpoint_node,
         attachment.role});
    endpoint_pip_by_attachment[attachment_index] = endpoint_pip_index;
  }
  for (std::size_t attachment_index = 0;
       attachment_index < graph.enabled_endpoint_attachments.size();
       ++attachment_index) {
    if (graph.enabled_endpoint_attachments[attachment_index] != 0 &&
        endpoint_pip_by_attachment[attachment_index] == kNoIndex) {
      throw std::runtime_error(
          "enabled IOB attachment was removed by design CSR filtering");
    }
  }
  for (RouteRequest& request : graph.route_requests) {
    for (SitePinNode& source : request.sources) {
      if (source.endpoint_attachment_index == kNoIndex) {
        continue;
      }
      if (source.endpoint_attachment_index >=
          endpoint_pip_by_attachment.size()) {
        throw std::runtime_error(
            "route source references an invalid endpoint attachment");
      }
      source.endpoint_pip_index = endpoint_pip_by_attachment[
          static_cast<std::size_t>(source.endpoint_attachment_index)];
      if (source.endpoint_pip_index == kNoIndex ||
          graph.endpoint_pips[source.endpoint_pip_index].role !=
              EndpointAttachmentRole::kSource) {
        throw std::runtime_error(
            "route source attachment is absent or has the wrong role");
      }
    }
    for (SitePinNode& sink : request.sinks) {
      if (sink.endpoint_attachment_index == kNoIndex) {
        continue;
      }
      if (sink.endpoint_attachment_index >= endpoint_pip_by_attachment.size()) {
        throw std::runtime_error(
            "route sink references an invalid endpoint attachment");
      }
      sink.endpoint_pip_index = endpoint_pip_by_attachment[
          static_cast<std::size_t>(sink.endpoint_attachment_index)];
      if (sink.endpoint_pip_index == kNoIndex ||
          graph.endpoint_pips[sink.endpoint_pip_index].role !=
              EndpointAttachmentRole::kSink) {
        throw std::runtime_error(
            "route sink attachment is absent or has the wrong role");
      }
    }
  }
  const auto endpoint_binding_end = TelemetryClock::now();
  const auto spatial_sidecar_begin = endpoint_binding_end;

  // The routing projection owns only these compact node columns. Move them
  // into the design-specific CSR artifact after edge filtering rather than
  // carrying a second graph-sized copy through conversion.
  csr.routing_sidecars.route_end_x = std::move(graph.node_route_end_x);
  csr.routing_sidecars.route_end_y = std::move(graph.node_route_end_y);
  csr.routing_sidecars.base_vertex_cost =
      std::move(graph.node_base_vertex_cost);

  const std::size_t node_count = static_cast<std::size_t>(csr.rows);
  const std::size_t edge_count = csr.colind.size();
  validate_routing_csr_sidecars(csr.routing_sidecars, node_count, edge_count,
                                false);
  const auto spatial_sidecar_end = TelemetryClock::now();
  if (telemetry != nullptr) {
    telemetry->filtering_seconds =
        elapsed_seconds(filtering_begin, filtering_end);
    telemetry->endpoint_binding_seconds =
        elapsed_seconds(endpoint_binding_begin, endpoint_binding_end);
    telemetry->spatial_sidecar_seconds =
        elapsed_seconds(spatial_sidecar_begin, spatial_sidecar_end);
    telemetry->filtering_output_allocation_bytes = checked_add_u64(
        checked_add_u64(
            vector_allocation_bytes(csr.rowptr, "filtered row pointers"),
            vector_allocation_bytes(csr.colind, "filtered destinations"),
            "filtered CSR allocation bytes"),
        checked_add_u64(
            vector_allocation_bytes(csr.edge_attrs,
                                    "filtered edge attributes"),
            vector_allocation_bytes(
                csr.retained_endpoint_attachment_edges,
                "retained endpoint attachment edges"),
            "filtered CSR allocation bytes"),
        "filtered CSR allocation bytes");
    telemetry->endpoint_binding_allocation_bytes = checked_add_u64(
        vector_allocation_bytes(endpoint_pip_by_attachment,
                                "endpoint attachment binding table"),
        vector_allocation_bytes(graph.endpoint_pips,
                                "endpoint PIP metadata"),
        "endpoint binding allocation bytes");
    telemetry->spatial_sidecar_allocation_bytes = checked_add_u64(
        vector_allocation_bytes(csr.routing_sidecars.spatial_edges.offsets,
                                "spatial shard offsets"),
        vector_allocation_bytes(csr.routing_sidecars.spatial_edges.edge_ids,
                                "spatial shard edge IDs"),
        "spatial sidecar allocation bytes");
  }
  return csr;
}

void write_csr_graph(const CsrGraph& graph,
                     const InterchangeArtifactPairId& artifact_pair_id,
                     const std::filesystem::path& output_path) {
  // Version 4 keeps the v3 header shape but makes unit edge weights implicit
  // and omits the destination-coordinate edge permutation. Representative
  // route-end coordinates and static vertex costs remain explicit. Version-3
  // readers must never infer these distinct payload semantics from ordering.
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

  const std::uint64_t nnz = static_cast<std::uint64_t>(graph.colind.size());
  if (graph.rows < 0 || graph.cols < 0) {
    throw std::runtime_error("cannot write a CSR with negative dimensions");
  }
  validate_routing_csr_sidecars(
      graph.routing_sidecars, static_cast<std::size_t>(graph.rows),
      graph.colind.size(), false);
  if (!graph.routing_sidecars.spatial_edges.offsets.empty() ||
      !graph.routing_sidecars.spatial_edges.edge_ids.empty() ||
      graph.routing_sidecars.spatial_edges.width != 0 ||
      graph.routing_sidecars.spatial_edges.height != 0) {
    throw std::runtime_error(
        "CSR v4 writer received a destination spatial-edge permutation");
  }
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
  write_u64(out, 0, "implicit values count");
  write_u64(out,
            static_cast<std::uint64_t>(
                graph.routing_sidecars.route_end_x.size()),
            "route-end X count");
  write_u64(out,
            static_cast<std::uint64_t>(
                graph.routing_sidecars.route_end_y.size()),
            "route-end Y count");
  write_u64(out,
            static_cast<std::uint64_t>(
                graph.routing_sidecars.base_vertex_cost.size()),
            "base vertex cost count");
  write_i64(out, 0, "unused spatial shard minimum X");
  write_i64(out, 0, "unused spatial shard minimum Y");
  write_u64(out, 0, "spatial shard width");
  write_u64(out, 0, "spatial shard height");
  write_u64(out, 0, "spatial shard offset count");
  write_u64(out, 0, "spatial shard edge ID count");

  write_array(out, graph.rowptr, "rowptr");
  write_array(out, graph.colind, "colind");
  write_array(out, graph.routing_sidecars.route_end_x, "route-end X values");
  write_array(out, graph.routing_sidecars.route_end_y, "route-end Y values");
  write_array(out, graph.routing_sidecars.base_vertex_cost,
              "base vertex costs");
  finish_output(out, output_path);
}

void write_metadata(const RoutingGraph& graph,
                    const CsrGraph& csr,
                    const InterchangeArtifactPairId& artifact_pair_id,
                    const std::filesystem::path& metadata_path,
                    MetadataWriteTelemetry* telemetry) {
  // RIPSIFM1 sidecar layout:
  //   char[8] magic
  //   u64 version, orientation, artifact_pair_id_high, artifact_pair_id_low
  //   u64 string_count, node_count, edge_attr_count, pip_data_count
  //   u64 endpoint_pip_count
  //   u64 site_pin_attr_count, route_request_count
  //   u64 blocked_node_count, sink_stop_node_count
  //   u64 logical_cell_count, logical_net_count, logical_port_instance_count
  //   u64 physical_netlist_byte_count, logical_netlist_byte_count
  //   u64 device_path_string, physical_path_string, logical_path_string
  //   u64 logical_design_name_string
  //   repeated strings: u64 byte_length, bytes
  //   Version 4/5 only: seven node metadata arrays totaling 40 bytes/node.
  //   Versions 6-8 omit them; node_count remains for CSR binding.
  //   edge_attr_count records: u32 tile_string, u32 pip_data_index
  //   pip_data_count records: u32 wire0_string, u32 wire1_string, u32 forward
  //   endpoint_pip_count records: ten u64 fields (EndpointPipDisk)
  //   site_pin_attr_count records: u64 node, u64 site_string, u64 pin_string
  //   route requests with logical net index and variable source/sink records
  //   logical_net_count records: u64 logical-net name string
  //   u64[blocked_node_count], u64[sink_stop_node_count]
  // Metadata v8 requires logical cell/port and both embedded payload counts to
  // be zero. Reconstruction reloads the original paths named by the header.
  if (artifact_pair_id.is_zero()) {
    throw std::runtime_error(
        "cannot write metadata with a zero artifact pair id");
  }
  if (graph.string_table.strings.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error(
        "metadata v8 string count exceeds uint32");
  }
  if (graph.pip_data.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error("metadata v8 PIP count exceeds uint32");
  }
  for (const std::uint64_t name_string : graph.logical_net_name_strings) {
    if (name_string >= graph.string_table.strings.size()) {
      throw std::runtime_error(
          "metadata v8 logical net references an invalid string");
    }
  }
  for (const RouteRequest& request : graph.route_requests) {
    if (request.net_string >= graph.string_table.strings.size()) {
      throw std::runtime_error(
          "metadata v8 route request references an invalid net string");
    }
    if (request.logical_net_index == kNoLogicalNetIndex) {
      continue;
    }
    if (request.logical_net_index >= graph.logical_net_name_strings.size() ||
        graph.logical_net_name_strings[static_cast<std::size_t>(
            request.logical_net_index)] != request.net_string) {
      throw std::runtime_error(
          "metadata v8 physical/logical net-name correlation mismatch");
    }
  }
  if (metadata_path.has_parent_path()) {
    std::filesystem::create_directories(metadata_path.parent_path());
  }
  const std::uint64_t node_count_u64 =
      as_u64(csr.rows, "metadata node count");
  if (csr.cols != csr.rows) {
    throw std::runtime_error("metadata CSR must be square");
  }
  if (node_count_u64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata node count does not fit size_t");
  }
  const std::size_t node_count = static_cast<std::size_t>(node_count_u64);
  if (graph.blocked_node.size() != node_count ||
      graph.sink_node_stops.size() != node_count) {
    throw std::runtime_error("node masks do not match metadata node count");
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
  for (std::size_t node = 0; node < node_count; ++node) {
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
  write_u64(out, node_count_u64, "node count");
  write_u64(out, static_cast<std::uint64_t>(csr.edge_attrs.size()),
            "edge attribute count");
  write_u64(out, static_cast<std::uint64_t>(graph.pip_data.size()),
            "pip data count");
  write_u64(out, static_cast<std::uint64_t>(graph.endpoint_pips.size()),
            "endpoint PIP count");
  write_u64(out, static_cast<std::uint64_t>(graph.site_pin_attrs.size()),
            "site pin attr count");
  write_u64(out, static_cast<std::uint64_t>(graph.route_requests.size()),
            "route request count");
  write_u64(out, static_cast<std::uint64_t>(blocked_nodes.size()),
            "blocked node count");
  write_u64(out, static_cast<std::uint64_t>(sink_stop_nodes.size()),
            "sink stop node count");
  write_u64(out, 0, "logical cell count");
  write_u64(out,
            static_cast<std::uint64_t>(
                graph.logical_net_name_strings.size()),
            "logical net count");
  write_u64(out, 0, "logical port instance count");
  write_u64(out, 0, "physical netlist byte count");
  write_u64(out, 0, "logical netlist byte count");
  write_u64(out, graph.device_path_string, "device path string");
  write_u64(out, graph.physical_path_string, "physical path string");
  write_u64(out, graph.logical_path_string, "logical path string");
  write_u64(out, graph.logical_design_name_string,
            "logical design name string");

  for (const std::string& text : graph.string_table.strings) {
    write_string(out, text);
  }

  // Edge attributes are aligned exactly with CSR colind order. For edge k,
  // csr.colind[k] and edge_attrs[k] describe one implicit-unit PIP edge.
  static_assert(sizeof(EdgeAttr) == 2 * sizeof(std::uint32_t),
                "EdgeAttr metadata layout changed");
  write_array(out, csr.edge_attrs, "edge attributes");

  // PIP data table stores the wire pair and direction referenced by each edge
  // attribute. tile name lives on EdgeAttr because the same wire pair appears
  // in many tile instances.
  std::vector<PipDataDisk> pip_data;
  pip_data.reserve(graph.pip_data.size());
  for (const PipData& data : graph.pip_data) {
    pip_data.push_back({
        checked_narrow_u32(data.wire0_string,
                           "metadata PIP wire0 string"),
        checked_narrow_u32(data.wire1_string,
                           "metadata PIP wire1 string"),
        static_cast<std::uint32_t>(data.forward ? 1 : 0)});
  }
  write_array(out, pip_data, "pip data");

  std::vector<EndpointPipDisk> endpoint_pips;
  endpoint_pips.reserve(graph.endpoint_pips.size());
  for (const EndpointPipMetadata& endpoint_pip : graph.endpoint_pips) {
    if (endpoint_pip.from < 0 || endpoint_pip.to < 0 ||
        endpoint_pip.endpoint_node < 0) {
      throw std::runtime_error(
          "cannot serialize endpoint PIP with an invalid node");
    }
    endpoint_pips.push_back(
        {endpoint_pip.csr_edge,
         static_cast<std::uint64_t>(endpoint_pip.from),
         static_cast<std::uint64_t>(endpoint_pip.to),
         endpoint_pip.tile_string,
         endpoint_pip.wire0_string,
         endpoint_pip.wire1_string,
         endpoint_pip.forward ? 1ULL : 0ULL,
         endpoint_pip.site_string,
         static_cast<std::uint64_t>(endpoint_pip.endpoint_node),
         static_cast<std::uint64_t>(endpoint_pip.role)});
  }
  write_array(out, endpoint_pips, "endpoint PIPs");

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
      write_u64(out, source.endpoint_pip_index,
                "route request source endpoint PIP");
    }

    write_u64(out, static_cast<std::uint64_t>(request.sinks.size()),
              "route request sink count");
    for (const SitePinNode& sink : request.sinks) {
      write_route_node(out, sink.node, "route request sink node");
      write_u64(out, sink.site_string, "route request sink site");
      write_u64(out, sink.pin_string, "route request sink pin");
      write_u64(out, sink.endpoint_pip_index,
                "route request sink endpoint PIP");
    }
  }

  // Minimal physical/logical correlation retained by v8 diagnostics. Route
  // requests index this flat deterministic list.
  write_array(out, graph.logical_net_name_strings,
              "logical net name strings");

  // Write the node masks last because they are auxiliary metadata rather than
  // per-edge routing information.
  write_array(out, blocked_nodes, "blocked nodes");
  write_array(out, sink_stop_nodes, "sink stop nodes");

  if (telemetry != nullptr) {
    std::uint64_t allocation_bytes = checked_add_u64(
        vector_allocation_bytes(blocked_nodes, "packed blocked nodes"),
        vector_allocation_bytes(sink_stop_nodes, "packed sink-stop nodes"),
        "metadata packing allocation bytes");
    allocation_bytes = checked_add_u64(
        allocation_bytes,
        vector_allocation_bytes(pip_data, "packed PIP data"),
        "metadata packing allocation bytes");
    allocation_bytes = checked_add_u64(
        allocation_bytes,
        vector_allocation_bytes(endpoint_pips, "packed endpoint PIPs"),
        "metadata packing allocation bytes");
    telemetry->packing_allocation_bytes = allocation_bytes;
  }
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

std::uint64_t routing_projection_bulk_allocation_bytes(
    const RoutingGraph& graph) {
  std::uint64_t bytes = 0;
  const auto add = [&](std::uint64_t value) {
    bytes = checked_add_u64(bytes, value,
                            "devicegraph bulk allocation bytes");
  };
  add(vector_allocation_bytes(graph.node_route_end_x, "route-end X"));
  add(vector_allocation_bytes(graph.node_route_end_y, "route-end Y"));
  add(vector_allocation_bytes(graph.node_base_vertex_cost, "base costs"));
  add(vector_allocation_bytes(graph.rowptr, "devicegraph row pointers"));
  add(vector_allocation_bytes(graph.colind, "devicegraph destinations"));
  add(vector_allocation_bytes(graph.edge_attrs, "devicegraph edge attrs"));
  add(vector_allocation_bytes(graph.pip_data, "devicegraph PIP data"));
  add(vector_allocation_bytes(graph.tile_wire_nodes,
                              "tile-wire lookup"));
  add(vector_allocation_bytes(graph.site_pin_nodes, "site-pin lookup"));
  add(vector_allocation_bytes(graph.endpoint_attachments,
                              "endpoint attachments"));
  add(vector_allocation_bytes(
      graph.endpoint_attachment_traversed_site_types,
      "endpoint attachment traversed site types"));
  add(vector_allocation_bytes(graph.endpoint_attachment_pseudo_cell_pins,
                              "endpoint pseudo-cell pins"));
  add(vector_allocation_bytes(graph.endpoint_attachment_lookups,
                              "endpoint attachment lookups"));
  return bytes;
}

std::uint64_t checked_file_size(const std::filesystem::path& path,
                                const char* name) {
  std::error_code error;
  const std::uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error(std::string("could not measure ") + name +
                             ": " + error.message());
  }
  if (bytes > std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error(std::string(name) +
                             " size exceeds uint64");
  }
  return static_cast<std::uint64_t>(bytes);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_begin = TelemetryClock::now();
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
    DeviceRoutingGraphReadTelemetry devicegraph_telemetry;
    RoutingGraph graph(read_device_routing_graph_for_routing(
        options.device_graph_path, true, &devicegraph_telemetry));
    const std::uint64_t string_object_bytes =
        vector_allocation_bytes(graph.string_table.strings,
                                "devicegraph string objects");
    emit_stage_telemetry(
        "devicegraph_strings",
        devicegraph_telemetry.string_loading_seconds +
            devicegraph_telemetry.string_index_seconds,
        {{"string_count", devicegraph_telemetry.string_count},
         {"payload_bytes", devicegraph_telemetry.string_payload_bytes},
         {"file_bytes", devicegraph_telemetry.string_file_bytes},
         {"string_object_bytes", string_object_bytes},
         {"index_entries",
          static_cast<std::uint64_t>(graph.string_table.ids.size())}});
    emit_stage_telemetry(
        "devicegraph_bulk_arrays",
        devicegraph_telemetry.bulk_array_loading_seconds,
        {{"file_bytes", devicegraph_telemetry.bulk_file_bytes},
         {"allocation_bytes",
          routing_projection_bulk_allocation_bytes(graph)}});
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

    std::cout << "imported_nodes: "
              << device_routing_graph_node_count(graph) << "\n";
    std::cout << "unique_edges: " << graph.loaded_edges << "\n";

    // LogicalNetlist parsing records logical cells/nets/port instances and
    // builds a name index that physical route requests can reference.
    DecodeParseTelemetry logical_telemetry;
    parse_logical_netlist(options.logical_path, graph, &logical_telemetry);
    emit_stage_telemetry(
        "logical_decode", logical_telemetry.decode_seconds,
        {{"decoded_bytes", logical_telemetry.decoded_bytes},
         {"allocation_bytes",
          logical_telemetry.decode_peak_allocation_bytes}});
    emit_stage_telemetry(
        "logical_parse", logical_telemetry.parse_seconds,
        {{"logical_nets", static_cast<std::uint64_t>(
                              graph.logical_net_name_strings.size())}});

    // PhysicalNetlist parsing adds design-specific route requests and blockage.
    // Metadata v8 retains its input path; reconstruction reloads that exact
    // .phys instead of embedding a decompressed copy in every sidecar.
    DecodeParseTelemetry physical_telemetry;
    const PhysicalImportStats import_stats =
        parse_physical_netlist(options.phys_path, graph,
                               &physical_telemetry);
    emit_stage_telemetry(
        "physical_decode", physical_telemetry.decode_seconds,
        {{"decoded_bytes", physical_telemetry.decoded_bytes},
         {"allocation_bytes",
          physical_telemetry.decode_peak_allocation_bytes}});
    emit_stage_telemetry(
        "physical_parse", physical_telemetry.parse_seconds,
        {{"route_requests",
          static_cast<std::uint64_t>(import_stats.route_requests)},
         {"enabled_attachments",
          static_cast<std::uint64_t>(std::count(
              graph.enabled_endpoint_attachments.begin(),
              graph.enabled_endpoint_attachments.end(),
              static_cast<std::uint8_t>(1)))}});
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
    CsrBuildTelemetry csr_build_telemetry;
    CsrGraph csr = make_outgoing_csr(graph, &csr_build_telemetry);
    emit_stage_telemetry(
        "graph_filtering", csr_build_telemetry.filtering_seconds,
        {{"base_edges", graph.loaded_edges},
         {"retained_edges", static_cast<std::uint64_t>(csr.colind.size())},
         {"allocation_bytes",
          csr_build_telemetry.filtering_output_allocation_bytes}});
    emit_stage_telemetry(
        "endpoint_pip_binding",
        csr_build_telemetry.endpoint_binding_seconds,
        {{"endpoint_pips",
          static_cast<std::uint64_t>(graph.endpoint_pips.size())},
         {"allocation_bytes",
          csr_build_telemetry.endpoint_binding_allocation_bytes}});
    emit_stage_telemetry(
        "spatial_sidecar", csr_build_telemetry.spatial_sidecar_seconds,
        {{"spatial_offset_count",
          static_cast<std::uint64_t>(
              csr.routing_sidecars.spatial_edges.offsets.size())},
         {"spatial_edge_count",
          static_cast<std::uint64_t>(
              csr.routing_sidecars.spatial_edges.edge_ids.size())},
         {"allocation_bytes",
          csr_build_telemetry.spatial_sidecar_allocation_bytes}});
    std::cout << "enabled_endpoint_pips: " << graph.endpoint_pips.size()
              << "\n";

    // Versions 6-8 do not serialize the seven physical node-metadata arrays.
    // The filtering reader normally projected them out before this point;
    // release them defensively if a full graph is ever supplied here.
    release_storage(graph.node_device_ids);
    release_storage(graph.node_min_x);
    release_storage(graph.node_max_x);
    release_storage(graph.node_min_y);
    release_storage(graph.node_max_y);
    release_storage(graph.node_tile_type_strings);
    release_storage(graph.node_wire_type_strings);

    // Filtering has copied every retained destination and edge attribute.
    // Drop the immutable base CSR before serializing either design output.
    release_storage(graph.rowptr);
    release_storage(graph.colind);
    release_storage(graph.edge_attrs);
    release_storage(graph.unavailable_destination_nodes);
    release_storage(graph.enabled_endpoint_attachments);
    release_storage(graph.endpoint_attachments);
    release_storage(graph.endpoint_attachment_traversed_site_types);
    release_storage(graph.endpoint_attachment_pseudo_cell_pins);
    release_storage(graph.endpoint_attachment_lookups);

    const std::uint64_t rowptr_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int64_t>(csr.rowptr.size(),
                                          "CSR row pointers"));
    const std::uint64_t colind_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int32_t>(csr.colind.size(),
                                          "CSR destinations"));
    constexpr std::uint64_t values_bytes = 0;
    const std::uint64_t route_x_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int32_t>(
            csr.routing_sidecars.route_end_x.size(), "route-end X values"));
    const std::uint64_t route_y_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<std::int32_t>(
            csr.routing_sidecars.route_end_y.size(), "route-end Y values"));
    const std::uint64_t base_cost_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<float>(
            csr.routing_sidecars.base_vertex_cost.size(),
            "base vertex costs"));
    constexpr std::uint64_t shard_offset_bytes = 0;
    constexpr std::uint64_t shard_edge_bytes = 0;
    const std::uint64_t attr_bytes = static_cast<std::uint64_t>(
        checked_array_bytes<EdgeAttr>(csr.edge_attrs.size(),
                                      "metadata edge attributes"));
    std::uint64_t csr_bytes = checked_add_u64(
        checked_add_u64(rowptr_bytes, colind_bytes, "CSR byte count"),
        values_bytes, "CSR byte count");
    for (const std::uint64_t sidecar_bytes :
         {route_x_bytes, route_y_bytes, base_cost_bytes, shard_offset_bytes,
          shard_edge_bytes}) {
      csr_bytes =
          checked_add_u64(csr_bytes, sidecar_bytes, "CSR byte count");
    }
    const std::int64_t csr_rows = csr.rows;
    const std::size_t csr_nnz = csr.colind.size();

    const std::filesystem::path staged_csr_path =
        routing::interchange::create_unique_staging_path(
            options.output_path);
    std::filesystem::path staged_metadata_path;
    std::filesystem::path staged_generation_path;
    std::uint64_t csr_output_bytes = 0;
    std::uint64_t metadata_output_bytes = 0;
    std::uint64_t generation_output_bytes = 0;
    std::uint64_t output_bytes = 0;
    try {
      staged_metadata_path =
          routing::interchange::create_unique_staging_path(
              options.metadata_path);
      staged_generation_path =
          routing::interchange::create_unique_staging_path(
              publication_generation_path);
      const InterchangeArtifactPairId artifact_pair_id =
          make_artifact_pair_id();
      const auto csr_write_begin = TelemetryClock::now();
      write_csr_graph(csr, artifact_pair_id, staged_csr_path);
      const auto csr_write_end = TelemetryClock::now();
      csr_output_bytes =
          checked_file_size(staged_csr_path, "staged CSR");
      emit_stage_telemetry(
          "csr_writing", elapsed_seconds(csr_write_begin, csr_write_end),
          {{"output_bytes", csr_output_bytes},
           {"row_count", static_cast<std::uint64_t>(csr.rows)},
           {"retained_edges", static_cast<std::uint64_t>(csr.colind.size())}});

      // The metadata sidecar needs edge attributes, but not the three generic
      // CSR arrays that were just written.
      release_storage(csr.rowptr);
      release_storage(csr.colind);
      release_storage(csr.retained_endpoint_attachment_edges);
      release_storage(csr.routing_sidecars.route_end_x);
      release_storage(csr.routing_sidecars.route_end_y);
      release_storage(csr.routing_sidecars.base_vertex_cost);
      release_storage(csr.routing_sidecars.spatial_edges.offsets);
      release_storage(csr.routing_sidecars.spatial_edges.edge_ids);
      MetadataWriteTelemetry metadata_write_telemetry;
      const auto metadata_write_begin = TelemetryClock::now();
      write_metadata(graph, csr, artifact_pair_id, staged_metadata_path,
                     &metadata_write_telemetry);
      const auto metadata_write_end = TelemetryClock::now();
      metadata_output_bytes =
          checked_file_size(staged_metadata_path, "staged metadata");
      emit_stage_telemetry(
          "metadata_packing_writing",
          elapsed_seconds(metadata_write_begin, metadata_write_end),
          {{"output_bytes", metadata_output_bytes},
           {"allocation_bytes",
            metadata_write_telemetry.packing_allocation_bytes},
           {"pip_count",
            static_cast<std::uint64_t>(graph.pip_data.size())}});
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
      generation_output_bytes = checked_file_size(
          staged_generation_path, "staged generation file");
      output_bytes = checked_add_u64(
          checked_add_u64(csr_output_bytes, metadata_output_bytes,
                          "published output bytes"),
          generation_output_bytes, "published output bytes");
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

    const auto publication_begin = TelemetryClock::now();
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
    const auto publication_end = TelemetryClock::now();
    emit_stage_telemetry(
        "publication", elapsed_seconds(publication_begin, publication_end),
        {{"csr_bytes", csr_output_bytes},
         {"metadata_bytes", metadata_output_bytes},
         {"generation_bytes", generation_output_bytes},
         {"output_bytes", output_bytes}});
    emit_stage_telemetry(
        "total", elapsed_seconds(total_begin, TelemetryClock::now()),
        {{"nodes", static_cast<std::uint64_t>(csr_rows)},
         {"base_edges", graph.loaded_edges},
         {"retained_edges", static_cast<std::uint64_t>(csr_nnz)},
         {"pip_count", static_cast<std::uint64_t>(graph.pip_data.size())},
         {"output_bytes", output_bytes}});

    std::cout << "csr_rows: " << csr_rows << "\n";
    std::cout << "csr_nnz: " << csr_nnz << "\n";
    std::cout << "csr_total_mib: " << mib(csr_output_bytes) << "\n";
    std::cout << "csr_payload_mib: " << mib(csr_bytes) << "\n";
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
