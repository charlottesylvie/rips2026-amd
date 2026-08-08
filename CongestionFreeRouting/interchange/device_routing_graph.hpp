#pragma once

#include "routing_csr_sidecars.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace routing::interchange {

using NodeId = std::int32_t;

constexpr NodeId kInvalidRouteNode = -1;
constexpr std::uint64_t kNoIndex =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;
constexpr std::uint64_t kNoStringIndex = kNoIndex;
constexpr std::uint64_t kMinimumDeviceRoutingGraphVersion = 3;
// Version 5 introduced the endpoint-attachment trailer layout. Version 6
// keeps that layout but requires the complete xcvu3p I/OP/TSP endpoint policy;
// production conversion must reject v5 graphs generated without TSP sinks.
constexpr std::uint64_t kEndpointAttachmentDeviceRoutingGraphVersion = 5;
constexpr std::uint64_t kCompleteIobAttachmentDeviceRoutingGraphVersion = 6;
constexpr std::uint64_t kCurrentDeviceRoutingGraphVersion = 6;

enum class NodeBoundsMode : std::uint64_t {
  kPocBaseWire = 0,
  kFullyContained = 1,
  kIntersects = 2,
};

struct Bounds {
  std::int32_t min_x = 0;
  std::int32_t max_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t min_y = 0;
  std::int32_t max_y = std::numeric_limits<std::int32_t>::max();
};

const char* node_bounds_mode_name(NodeBoundsMode mode);
NodeBoundsMode parse_node_bounds_mode(const std::string& text);

// Create one empty, exclusively named temporary file next to a final output.
// Keeping staging on the same filesystem makes the eventual rename atomic and
// avoids temporary-file clobbering between concurrent converter processes.
std::filesystem::path create_unique_staging_path(
    const std::filesystem::path& final_path);

// The cache owns one stable string-ID namespace. The per-design converter
// appends benchmark strings to this table, so all static node/PIP/lookup IDs
// remain valid without a remapping pass.
struct StringTable {
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint64_t> ids;

  std::uint64_t intern(const std::string& text);
  std::optional<std::uint64_t> find(const std::string& text) const;
  void rebuild_index();
};

struct PipData {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
};

// This record is intentionally two adjacent u64s. Both the static cache and
// final metadata file can serialize large edge-attribute arrays with one bulk
// write instead of two stream calls per edge.
struct EdgeAttr {
  std::uint64_t tile_string = 0;
  std::uint64_t pip_data_index = 0;
};

// Compact on-disk/runtime lookup record. Static cache string tables are
// required to have fewer than 2^32 entries; node IDs are already int32.
struct PairNodeLookup {
  std::uint32_t first_string = 0;
  std::uint32_t second_string = 0;
  NodeId node = kInvalidRouteNode;
  std::uint32_t reserved = 0;
};

bool operator<(const PairNodeLookup& lhs, const PairNodeLookup& rhs);

// A physical site can legally be instantiated as its primary type or as one
// of several alternate types.  Pin names are meaningful only together with
// that active type: the same (site, pin) spelling can map to different device
// nodes for two possible types.  Keep the type in the reusable cache so the
// per-design importer can resolve against PhysicalNetlist.siteInsts.
struct SitePinNodeLookup {
  std::uint32_t site_string = 0;
  std::uint32_t site_type_string = 0;
  std::uint32_t pin_string = 0;
  NodeId node = kInvalidRouteNode;
};

bool operator<(const SitePinNodeLookup& lhs,
               const SitePinNodeLookup& rhs);

// Audited pseudo PIPs are admitted only as fixed directed corridors between a
// concrete route endpoint and ordinary fabric. They are never general-purpose
// route-throughs.
enum class EndpointAttachmentRole : std::uint32_t {
  kSource = 0,
  kSink = 1,
};

// Direction of one BEL pin consumed by a pseudo PIP. This mirrors the three
// FPGA Interchange logical directions without retaining a schema dependency in
// the persistent device graph.
enum class PseudoCellPinDirection : std::uint32_t {
  kInput = 0,
  kOutput = 1,
  kInout = 2,
};

struct PseudoCellPinResource {
  std::uint32_t bel_string = 0;
  std::uint32_t pin_string = 0;
  PseudoCellPinDirection direction = PseudoCellPinDirection::kInput;
};

// One record describes one concrete, directed pseudo PIP. pip_data_index is
// unique across this array and identifies the otherwise unchanged PipData and
// EdgeAttr records. Source corridors are endpoint_node -> from_node -> to_node;
// sink corridors are from_node -> to_node -> endpoint_node. The first/last
// edge respectively is conventional and the middle edge is this attachment.
struct EndpointAttachment {
  std::uint32_t endpoint_site_string = 0;
  std::uint32_t endpoint_site_type_string = 0;
  std::uint32_t endpoint_pin_string = 0;
  EndpointAttachmentRole role = EndpointAttachmentRole::kSource;
  NodeId endpoint_node = kInvalidRouteNode;
  NodeId from_node = kInvalidRouteNode;
  NodeId to_node = kInvalidRouteNode;
  std::uint64_t pip_data_index = kNoIndex;
  std::uint32_t traversed_site_string = 0;
  std::uint64_t traversed_site_type_begin = 0;
  std::uint64_t traversed_site_type_count = 0;
  std::uint64_t pseudo_cell_pin_begin = 0;
  std::uint64_t pseudo_cell_pin_count = 0;
};

// Exact endpoint active-type authorization key. attachment_index indexes
// DeviceRoutingGraph::endpoint_attachments. The allowed active types of the
// distinct traversed site live in that attachment's traversed-site-type slice.
struct EndpointAttachmentLookup {
  std::uint32_t endpoint_site_string = 0;
  std::uint32_t endpoint_site_type_string = 0;
  std::uint32_t endpoint_pin_string = 0;
  EndpointAttachmentRole role = EndpointAttachmentRole::kSource;
  std::uint32_t attachment_index = 0;
};

bool operator<(const EndpointAttachmentLookup& lhs,
               const EndpointAttachmentLookup& rhs);

enum class LookupConflictPolicy {
  kReject,
  kDropAmbiguous,
};

// Sort lookup keys, collapse identical key->node aliases, and either reject
// or omit keys that map to more than one node.  Silently selecting one of two
// physical nodes can fabricate a route, so callers must make the ambiguity
// policy explicit.
std::size_t sort_and_deduplicate_pair_node_lookups(
    std::vector<PairNodeLookup>& records,
    LookupConflictPolicy conflict_policy,
    const char* lookup_name);

// Typed site-pin keys must be unique.  Multiple possible site types may still
// contain the same pin spelling; those are distinct keys and are retained.
void sort_and_deduplicate_site_pin_lookups(
    std::vector<SitePinNodeLookup>& records);

// Preprocessor-only interleaved entry used while sorting/deduplicating rows.
// The cache writer gathers its columns and attributes into compact contiguous
// sections without materializing another full multi-gigabyte edge copy.
struct StaticCsrEntry {
  NodeId col = 0;
  std::uint32_t ordinal = 0;
  EdgeAttr attr;
};

// Immutable data determined solely by DeviceResources plus bounds policy.
struct DeviceRoutingGraph {
  StringTable string_table;
  // The writer always emits kCurrentDeviceRoutingGraphVersion. Readers retain
  // the original version so production callers can reject stale v3/v4 caches
  // while generic inspection remains backward compatible.
  std::uint64_t format_version = kCurrentDeviceRoutingGraphVersion;
  std::uint64_t device_fingerprint = 0;
  std::uint64_t device_path_string = 0;
  std::uint64_t device_name_string = 0;
  Bounds bounds;
  NodeBoundsMode node_bounds_mode = NodeBoundsMode::kPocBaseWire;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;

  // The filtering projection retains the on-disk node count without loading
  // the seven physical node arrays. Full graphs leave this equal to the array
  // size; programmatically constructed graphs may leave it at zero and derive
  // their count from node_device_ids.
  std::size_t retained_node_count = 0;

  std::vector<std::uint64_t> node_device_ids;
  std::vector<std::int32_t> node_min_x;
  std::vector<std::int32_t> node_max_x;
  std::vector<std::int32_t> node_min_y;
  std::vector<std::int32_t> node_max_y;
  std::vector<std::uint64_t> node_tile_type_strings;
  std::vector<std::uint64_t> node_wire_type_strings;

  // BF11-facing static sidecars.  route_end_{x,y} is the representative tile
  // of the node wire selected for routing metadata, or
  // kMissingRouteCoordinate for special resources without a physical tile.
  // base_vertex_cost is factored from mutable negotiated-congestion cost so a
  // routing epoch can update the latter without rewriting immutable CSR data.
  std::vector<std::int32_t> node_route_end_x;
  std::vector<std::int32_t> node_route_end_y;
  std::vector<float> node_base_vertex_cost;

  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<EdgeAttr> edge_attrs;
  std::vector<PipData> pip_data;

  // Sorted lexicographically for allocation-free binary-search lookups during
  // physical-netlist parsing.
  std::vector<PairNodeLookup> tile_wire_nodes;
  std::vector<SitePinNodeLookup> site_pin_nodes;

  // Sparse v5+ endpoint-attachment metadata. Allowed active types of the
  // concrete traversed site are compact string IDs sliced by
  // EndpointAttachment. Pseudo-cell BEL/pin resources use that same site.
  std::vector<EndpointAttachment> endpoint_attachments;
  std::vector<std::uint32_t> endpoint_attachment_traversed_site_types;
  std::vector<PseudoCellPinResource> endpoint_attachment_pseudo_cell_pins;
  std::vector<EndpointAttachmentLookup> endpoint_attachment_lookups;
};

// Return the authoritative node count for either a full graph or the compact
// filtering projection, rejecting a retained count that contradicts loaded
// node IDs.
std::size_t device_routing_graph_node_count(
    const DeviceRoutingGraph& graph);

// Design-filtered outgoing CSR. Values are always exact unit weights.
struct CsrGraph {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<float> values;
  std::vector<EdgeAttr> edge_attrs;
  RoutingCsrSidecars routing_sidecars;
};

std::uint32_t checked_lookup_string_id(std::uint64_t id);

std::optional<NodeId> find_pair_node(
    const std::vector<PairNodeLookup>& records,
    const StringTable& strings,
    const std::string& first,
    const std::string& second);

// Resolve an exact active-type key.  If active_site_type is absent, accept an
// untyped fallback only when every possible type maps (site, pin) to the same
// node.  This makes incomplete legacy PhysicalNetlists usable without ever
// guessing between distinct physical resources.
std::optional<NodeId> find_site_pin_node(
    const std::vector<SitePinNodeLookup>& records,
    const StringTable& strings,
    const std::string& site,
    const std::optional<std::string>& active_site_type,
    const std::string& pin);

// Return all distinct possible nodes for an untyped site/pin.  Fixed-resource
// preservation uses this as a conservative fallback when siteInst metadata is
// missing or inconsistent.
std::vector<NodeId> find_site_pin_candidates(
    const std::vector<SitePinNodeLookup>& records,
    const StringTable& strings,
    const std::string& site,
    const std::string& pin);

// Sort exact endpoint-attachment lookup keys, collapse identical aliases, and
// reject a key that names more than one attachment.
void sort_and_deduplicate_endpoint_attachment_lookups(
    std::vector<EndpointAttachmentLookup>& records);

// Rebuild the exact endpoint lookup. Attachments themselves must already be
// ordered by unique pip_data_index.
void rebuild_endpoint_attachment_lookups(DeviceRoutingGraph& graph);

std::optional<std::uint32_t> find_endpoint_attachment_index(
    const std::vector<EndpointAttachmentLookup>& records,
    const StringTable& strings,
    const std::string& endpoint_site,
    const std::string& endpoint_site_type,
    const std::string& endpoint_pin,
    EndpointAttachmentRole role);

// Test one active type of the concrete traversed site against an attachment's
// sorted authorization slice. The endpoint site's active type is a separate,
// exact scalar used by find_endpoint_attachment_index().
bool endpoint_attachment_allows_traversed_site_type(
    const DeviceRoutingGraph& graph,
    std::uint32_t attachment_index,
    const std::string& traversed_site_type);

// Fail with an explicit regeneration diagnostic when a production path needs
// complete I/OP/TSP endpoint-attachment semantics but was given a stale
// v3-v5 cache.
void require_endpoint_attachment_device_graph(
    const DeviceRoutingGraph& graph);

void validate_device_routing_graph(const DeviceRoutingGraph& graph);

DeviceRoutingGraph read_device_routing_graph(
    const std::filesystem::path& path);

// Converter fast path: seeks over the seven unused physical node arrays,
// retains their node count, and validates the header, strings, lookups, and
// CSR shape. Individual edge checks are deferred to
// filter_device_routing_graph(), avoiding both 40 bytes/node of input and a
// second scan of every large edge record in the per-design pipeline.
DeviceRoutingGraph read_device_routing_graph_for_filtering(
    const std::filesystem::path& path,
    bool require_endpoint_attachments = false);

// Per-design filtering needs the immutable CSR/lookups plus BF11's compact
// route-end/cost columns, but not the legacy 40-byte physical-node columns.
// Version-3 artifacts synthesize representative coordinates from node extents
// and unit base costs; versions 4+ read the authored sidecars directly.
DeviceRoutingGraph read_device_routing_graph_for_routing(
    const std::filesystem::path& path,
    bool require_endpoint_attachments = true);

// Standard writer used by tests and tools that already own split arrays.
void write_device_routing_graph(const DeviceRoutingGraph& graph,
                                const std::filesystem::path& path);

// Memory-conscious preprocessor writer. graph.rowptr and static_entries must
// describe the same sorted/deduplicated CSR; graph.colind/edge_attrs may be
// empty so no second full edge representation is required.
void write_device_routing_graph(
    const DeviceRoutingGraph& graph,
    const std::vector<StaticCsrEntry>& static_entries,
    const std::filesystem::path& path);

// Sort each source row by destination, collapse parallel node-pair edges, and
// retain the greatest per-row ordinal (the original converter's "latest PIP
// wins" behavior). rowptr is rewritten for the compacted entries.
void sort_and_deduplicate_static_csr(
    std::vector<std::int64_t>& rowptr,
    std::vector<StaticCsrEntry>& entries);

CsrGraph filter_device_routing_graph(
    const DeviceRoutingGraph& graph,
    const std::vector<std::uint8_t>& blocked_node,
    const std::vector<std::uint8_t>& sink_node_stops,
    // Union of blocked nodes and exclusive route-source nodes. Keeping this
    // precombined avoids two unrelated random mask reads per destination edge.
    const std::vector<std::uint8_t>& unavailable_destination_nodes,
    // Indexed by endpoint_attachments. Empty means that every attachment is
    // disabled. Non-attachment edges retain the existing conventional policy.
    const std::vector<std::uint8_t>& enabled_endpoint_attachments = {});

}  // namespace routing::interchange
