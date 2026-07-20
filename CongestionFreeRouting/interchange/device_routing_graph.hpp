#pragma once

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
  std::uint64_t device_fingerprint = 0;
  std::uint64_t device_path_string = 0;
  Bounds bounds;
  NodeBoundsMode node_bounds_mode = NodeBoundsMode::kPocBaseWire;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;

  std::vector<std::uint64_t> node_device_ids;
  std::vector<std::int32_t> node_min_x;
  std::vector<std::int32_t> node_max_x;
  std::vector<std::int32_t> node_min_y;
  std::vector<std::int32_t> node_max_y;
  std::vector<std::uint64_t> node_tile_type_strings;
  std::vector<std::uint64_t> node_wire_type_strings;

  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<EdgeAttr> edge_attrs;
  std::vector<PipData> pip_data;

  // Sorted lexicographically for allocation-free binary-search lookups during
  // physical-netlist parsing.
  std::vector<PairNodeLookup> tile_wire_nodes;
  std::vector<PairNodeLookup> site_pin_nodes;
};

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
};

std::uint32_t checked_lookup_string_id(std::uint64_t id);

std::optional<NodeId> find_pair_node(
    const std::vector<PairNodeLookup>& records,
    const StringTable& strings,
    const std::string& first,
    const std::string& second);

void validate_device_routing_graph(const DeviceRoutingGraph& graph);

DeviceRoutingGraph read_device_routing_graph(
    const std::filesystem::path& path);

// Converter fast path: validates the header, static metadata, lookups, and CSR
// shape while deferring individual edge checks to
// filter_device_routing_graph(). This avoids scanning every large edge record
// twice in the per-design pipeline.
DeviceRoutingGraph read_device_routing_graph_for_filtering(
    const std::filesystem::path& path);

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
    const std::vector<std::uint8_t>& sink_node_stops);

}  // namespace routing::interchange
