// Build this host-only CSR/metadata format test once for each standalone reader:
//   g++ -std=c++17 -O0 -pthread -DBF_STANDALONE_READER=8 \
//     -I Routing/tests/fake_hip -I HIP_kernel/bellman_ford/src \
//     CongestionFreeRouting/tests/bf_standalone_csr_format_test.cpp \
//     -o /tmp/bf8_csr_format_test
// Repeat with BF_STANDALONE_READER=9 and 10.

#ifndef BF_STANDALONE_READER
#error "define BF_STANDALONE_READER to 8, 9, or 10"
#endif

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>

// The repository's compile-only HIP shim exposes the nonblocking constructor
// used by PathFinder. Standalone BF8/9/10 use the equivalent default-stream
// spelling, so provide that one narrow adapter inside this test translation
// unit without expanding the shared fake-HIP surface.
inline hipError_t hipStreamCreate(hipStream_t* stream) {
  return hipStreamCreateWithFlags(stream, 0);
}

#if BF_STANDALONE_READER == 9 || BF_STANDALONE_READER == 10
constexpr unsigned int hipEventDisableTiming = 2;
inline hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int) {
  return hipEventCreate(event);
}
inline hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
#endif

#if BF_STANDALONE_READER == 8
#define BF8_NO_MAIN
#include "../bellman_ford/bf8.cpp"
#elif BF_STANDALONE_READER == 9
#define BF9_NO_MAIN
#include "../bellman_ford/bf9.cpp"
#elif BF_STANDALONE_READER == 10
#define BF10_NO_MAIN
#include "../bellman_ford/bf10.cpp"
#else
#error "BF_STANDALONE_READER must be 8, 9, or 10"
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using PairId = routing::interchange::InterchangeArtifactPairId;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_u64(std::ofstream& out, std::uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ofstream& out, const std::string& value) {
  write_u64(out, static_cast<std::uint64_t>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

struct HeaderOverrides {
  std::optional<std::uint64_t> values_count;
  std::optional<std::uint64_t> route_x_count;
  std::optional<std::uint64_t> route_y_count;
  std::optional<std::uint64_t> base_cost_count;
  std::optional<std::uint64_t> spatial_width;
  std::optional<std::uint64_t> spatial_height;
  std::optional<std::uint64_t> spatial_offset_count;
  std::optional<std::uint64_t> spatial_edge_id_count;
  bool omit_v3_spatial_payload = false;
};

void write_csr_fixture(const std::filesystem::path& path,
                       std::uint64_t version,
                       const PairId& pair,
                       const HeaderOverrides& overrides = {}) {
  const bool has_pair = version == 2 || version == 3 || version == 4;
  const bool has_node_sidecars = version == 3 || version == 4;
  const bool has_explicit_values =
      version == 1 || version == 2 || version == 3;
  const bool has_spatial_shards = version == 3;

  const std::uint64_t values_count = overrides.values_count.value_or(
      has_explicit_values ? 1 : 0);
  const std::uint64_t route_x_count =
      overrides.route_x_count.value_or(has_node_sidecars ? 2 : 0);
  const std::uint64_t route_y_count =
      overrides.route_y_count.value_or(has_node_sidecars ? 2 : 0);
  const std::uint64_t base_cost_count =
      overrides.base_cost_count.value_or(has_node_sidecars ? 2 : 0);
  const std::uint64_t spatial_width =
      overrides.spatial_width.value_or(has_spatial_shards ? 1 : 0);
  const std::uint64_t spatial_height =
      overrides.spatial_height.value_or(has_spatial_shards ? 1 : 0);
  const std::uint64_t spatial_offset_count =
      overrides.spatial_offset_count.value_or(has_spatial_shards ? 3 : 0);
  const std::uint64_t spatial_edge_id_count =
      overrides.spatial_edge_id_count.value_or(has_spatial_shards ? 1 : 0);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write("RIPSCSR1", 8);
  write_u64(out, version);
  write_u64(out, 2);  // outgoing orientation
  if (has_pair) {
    write_u64(out, pair.high);
    write_u64(out, pair.low);
  }
  write_u64(out, 2);  // rows
  write_u64(out, 2);  // cols
  write_u64(out, 1);  // declared edges
  write_u64(out, 1);  // loaded edges
  write_u64(out, 1);  // nnz
  write_u64(out, 3);  // rowptr count
  write_u64(out, 1);  // colind count
  write_u64(out, values_count);
  if (has_node_sidecars) {
    write_u64(out, route_x_count);
    write_u64(out, route_y_count);
    write_u64(out, base_cost_count);
    // Origins are retained header fields but are irrelevant when v4 has no
    // spatial permutation. Nonzero values prove readers do not invent a new
    // restriction beyond the specified zero dimensions/counts.
    write_u64(out, version == 4 ? 17 : 0);
    write_u64(out, version == 4 ? 29 : 0);
    write_u64(out, spatial_width);
    write_u64(out, spatial_height);
    write_u64(out, spatial_offset_count);
    write_u64(out, spatial_edge_id_count);
  }

  const std::int64_t rowptr[3] = {0, 1, 1};
  const std::int32_t colind[1] = {1};
  const float value[1] = {2.5f};
  const std::int32_t route_end_x[2] = {3, 4};
  const std::int32_t route_end_y[2] = {5, 6};
  const float base_cost[2] = {1.0f, 2.0f};
  const std::uint64_t shard_offsets[3] = {0, 1, 1};
  const std::uint32_t shard_edge_ids[1] = {0};
  out.write(reinterpret_cast<const char*>(rowptr), sizeof(rowptr));
  out.write(reinterpret_cast<const char*>(colind), sizeof(colind));
  if (has_explicit_values) {
    out.write(reinterpret_cast<const char*>(value), sizeof(value));
  }
  if (has_node_sidecars) {
    out.write(reinterpret_cast<const char*>(route_end_x), sizeof(route_end_x));
    out.write(reinterpret_cast<const char*>(route_end_y), sizeof(route_end_y));
    out.write(reinterpret_cast<const char*>(base_cost), sizeof(base_cost));
  }
  if (has_spatial_shards && !overrides.omit_v3_spatial_payload) {
    out.write(reinterpret_cast<const char*>(shard_offsets),
              sizeof(shard_offsets));
    out.write(reinterpret_cast<const char*>(shard_edge_ids),
              sizeof(shard_edge_ids));
  }
  if (!out) {
    throw std::runtime_error("failed to write standalone CSR fixture");
  }
}

struct MetadataHeaderOverrides {
  std::optional<std::uint64_t> string_count;
  std::optional<std::uint64_t> pip_data_count;
  std::optional<std::uint64_t> logical_cell_count;
  std::optional<std::uint64_t> logical_net_count;
  std::optional<std::uint64_t> logical_port_instance_count;
  std::optional<std::uint64_t> physical_netlist_byte_count;
  std::optional<std::uint64_t> logical_netlist_byte_count;
  std::optional<std::uint64_t> route_net_string;
  std::optional<std::uint64_t> route_logical_net_index;
  std::optional<std::uint64_t> first_logical_net_name_string;
};

enum class MetadataFixtureTruncation {
  kNone,
  kCompactEdgeAttr,
  kCompactPip,
  kFlatLogicalNetNames,
};

void write_metadata_fixture(
    const std::filesystem::path& path,
    std::uint64_t version,
    const PairId& pair,
    const MetadataHeaderOverrides& overrides = {},
    MetadataFixtureTruncation truncation = MetadataFixtureTruncation::kNone) {
  if (version < 3 || version > 8) {
    throw std::invalid_argument("metadata fixture version must be 3 through 8");
  }
  const bool has_pair =
      version == 5 || version == 6 || version == 7 || version == 8;
  const bool has_legacy_node_arrays =
      version == 3 || version == 4 || version == 5;
  const bool has_node_physical_arrays = version == 4 || version == 5;
  const bool has_endpoint_pips = version == 7 || version == 8;
  const bool has_compact_edge_pip_records = version == 8;
  const bool has_flat_logical_net_names = version == 8;

  constexpr std::uint64_t kStringCount = 5;
  constexpr std::uint64_t kNodeCount = 2;
  constexpr std::uint64_t kEdgeAttrCount = 1;
  constexpr std::uint64_t kPipDataCount = 1;
  constexpr std::uint64_t kEndpointPipCount = 1;
  constexpr std::uint64_t kSitePinAttrCount = 1;
  constexpr std::uint64_t kRouteRequestCount = 1;
  constexpr std::uint64_t kBlockedNodeCount = 1;
  constexpr std::uint64_t kSinkStopNodeCount = 1;
  constexpr std::uint64_t kLogicalNetCount = 2;

  const std::uint64_t logical_cell_count =
      overrides.logical_cell_count.value_or(version == 8 ? 0 : 1);
  const std::uint64_t logical_port_instance_count =
      overrides.logical_port_instance_count.value_or(version == 8 ? 0 : 1);
  const std::uint64_t physical_netlist_byte_count =
      overrides.physical_netlist_byte_count.value_or(version == 8 ? 0 : 3);
  const std::uint64_t logical_netlist_byte_count =
      overrides.logical_netlist_byte_count.value_or(version == 8 ? 0 : 2);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write("RIPSIFM1", 8);
  write_u64(out, version);
  write_u64(out, 2);  // outgoing orientation
  if (has_pair) {
    write_u64(out, pair.high);
    write_u64(out, pair.low);
  }
  write_u64(out, overrides.string_count.value_or(kStringCount));
  write_u64(out, kNodeCount);
  write_u64(out, kEdgeAttrCount);
  write_u64(out, overrides.pip_data_count.value_or(kPipDataCount));
  if (has_endpoint_pips) {
    write_u64(out, kEndpointPipCount);
  }
  write_u64(out, kSitePinAttrCount);
  write_u64(out, kRouteRequestCount);
  write_u64(out, kBlockedNodeCount);
  write_u64(out, kSinkStopNodeCount);
  write_u64(out, logical_cell_count);
  write_u64(out, overrides.logical_net_count.value_or(kLogicalNetCount));
  write_u64(out, logical_port_instance_count);
  write_u64(out, physical_netlist_byte_count);
  write_u64(out, logical_netlist_byte_count);
  write_u64(out, 0);  // device path string
  write_u64(out, 0);  // physical path string
  write_u64(out, 0);  // logical path string
  write_u64(out, 0);  // logical design name string

  write_string(out, "net");
  write_string(out, "site");
  write_string(out, "pin");
  write_string(out, "logical0");
  write_string(out, "logical1");

  if (has_legacy_node_arrays) {
    const std::uint64_t device_node_ids[kNodeCount] = {10, 11};
    out.write(reinterpret_cast<const char*>(device_node_ids),
              sizeof(device_node_ids));
    if (has_node_physical_arrays) {
      const std::int32_t coordinates[kNodeCount] = {1, 2};
      const std::uint64_t type_strings[kNodeCount] = {3, 4};
      for (int i = 0; i < 4; ++i) {
        out.write(reinterpret_cast<const char*>(coordinates),
                  sizeof(coordinates));
      }
      out.write(reinterpret_cast<const char*>(type_strings),
                sizeof(type_strings));
      out.write(reinterpret_cast<const char*>(type_strings),
                sizeof(type_strings));
    }
  }

  if (has_compact_edge_pip_records) {
    const std::uint32_t edge_attr[2] = {3, 0};
    const std::streamsize edge_bytes =
        static_cast<std::streamsize>(sizeof(edge_attr));
    out.write(reinterpret_cast<const char*>(edge_attr),
              truncation == MetadataFixtureTruncation::kCompactEdgeAttr
                  ? edge_bytes - 1
                  : edge_bytes);
    if (truncation == MetadataFixtureTruncation::kCompactEdgeAttr) return;

    const std::uint32_t pip_data[3] = {2, 3, 1};
    const std::streamsize pip_bytes =
        static_cast<std::streamsize>(sizeof(pip_data));
    out.write(reinterpret_cast<const char*>(pip_data),
              truncation == MetadataFixtureTruncation::kCompactPip
                  ? pip_bytes - 1
                  : pip_bytes);
    if (truncation == MetadataFixtureTruncation::kCompactPip) return;
  } else {
    const std::uint64_t edge_attr[2] = {3, 0};
    const std::uint64_t pip_data[3] = {2, 3, 1};
    out.write(reinterpret_cast<const char*>(edge_attr), sizeof(edge_attr));
    out.write(reinterpret_cast<const char*>(pip_data), sizeof(pip_data));
  }

  if (has_endpoint_pips) {
    // csr_edge, from, to, tile, wire0, wire1, forward, site, endpoint, role
    const std::uint64_t endpoint_pip[10] = {0, 0, 1, 3, 2,
                                            3, 1, 1, 0, 1};
    out.write(reinterpret_cast<const char*>(endpoint_pip),
              sizeof(endpoint_pip));
  }
  const std::uint64_t site_pin_attr[3] = {1, 2, 0};
  out.write(reinterpret_cast<const char*>(site_pin_attr),
            sizeof(site_pin_attr));

  write_u64(out, overrides.route_net_string.value_or(0));
  write_u64(out, overrides.route_logical_net_index.value_or(0));
  write_u64(out, 1);  // source count
  write_u64(out, 0);  // source node
  write_u64(out, 1);  // source site string
  write_u64(out, 2);  // source pin string
  if (has_endpoint_pips) write_u64(out, 0);
  write_u64(out, 1);  // sink count
  write_u64(out, 1);  // sink node
  write_u64(out, 1);  // sink site string
  write_u64(out, 2);  // sink pin string
  if (has_endpoint_pips) write_u64(out, 0);

  if (has_flat_logical_net_names) {
    const std::uint64_t logical_net_names[kLogicalNetCount] = {
        overrides.first_logical_net_name_string.value_or(0), 4};
    const std::streamsize logical_net_bytes =
        static_cast<std::streamsize>(sizeof(logical_net_names));
    out.write(
        reinterpret_cast<const char*>(logical_net_names),
        truncation == MetadataFixtureTruncation::kFlatLogicalNetNames
            ? logical_net_bytes - 1
            : logical_net_bytes);
    if (truncation == MetadataFixtureTruncation::kFlatLogicalNetNames) return;
  } else {
    const std::uint64_t logical_cell[3] = {0, 0, 0};
    const std::uint64_t logical_nets[kLogicalNetCount][4] = {
        {3, 0, 0, 0}, {4, 0, 0, 0}};
    const std::uint64_t logical_port_instance[7] = {0, 0, 0, 0, 0, 0, 0};
    out.write(reinterpret_cast<const char*>(logical_cell),
              sizeof(logical_cell));
    out.write(reinterpret_cast<const char*>(logical_nets),
              sizeof(logical_nets));
    out.write(reinterpret_cast<const char*>(logical_port_instance),
              sizeof(logical_port_instance));
  }

  write_u64(out, 1);  // blocked node
  write_u64(out, 1);  // sink-stop node
  for (std::uint64_t i = 0; i < physical_netlist_byte_count; ++i) {
    const char byte = static_cast<char>(0x20 + (i & 0x1f));
    out.write(&byte, 1);
  }
  for (std::uint64_t i = 0; i < logical_netlist_byte_count; ++i) {
    const char byte = static_cast<char>(0x40 + (i & 0x1f));
    out.write(&byte, 1);
  }
  if (!out) {
    throw std::runtime_error("failed to write standalone metadata fixture");
  }
}

template <typename Callback>
void require_failure(Callback&& callback, const char* message) {
  bool failed = false;
  try {
    callback();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, message);
}

void test_reader_formats(const std::filesystem::path& path) {
  const PairId pair{0x123456789abcdef0ULL, 0x0fedcba987654321ULL};
  for (std::uint64_t version = 1; version <= 4; ++version) {
    write_csr_fixture(path, version, pair);
    const HostOutgoingCsrF32 graph = load_outgoing_csrbin(path);
    require(graph.rows == 2 && graph.cols == 2 && graph.nnz == 1,
            "standalone reader changed CSR dimensions");
    const std::vector<float> expected_values =
        version == 4 ? std::vector<float>({1.0f})
                     : std::vector<float>({2.5f});
    require(graph.rowptr.size() == 3 && graph.rowptr[0] == 0 &&
                graph.rowptr[1] == 1 && graph.rowptr[2] == 1 &&
                graph.to == std::vector<Index>({1}) &&
                graph.values == expected_values,
            "standalone reader changed explicit or implicit CSR values");
    require(version == 1 ? !graph.artifact_pair_id.has_value()
                         : graph.artifact_pair_id == pair,
            "standalone reader changed artifact-pair compatibility");
  }

  HeaderOverrides missing_v3_shards;
  missing_v3_shards.spatial_width = 0;
  missing_v3_shards.spatial_height = 0;
  missing_v3_shards.spatial_offset_count = 0;
  missing_v3_shards.spatial_edge_id_count = 0;
  missing_v3_shards.omit_v3_spatial_payload = true;
  write_csr_fixture(path, 3, pair, missing_v3_shards);
  require_failure(
      [&] { (void)load_outgoing_csrbin(path); },
      "standalone reader reinterpreted shard-less CSR v3 as CSR v4");

  std::vector<HeaderOverrides> malformed_v4(8);
  malformed_v4[0].values_count = 1;
  malformed_v4[1].route_x_count = 1;
  malformed_v4[2].route_y_count = 1;
  malformed_v4[3].base_cost_count = 1;
  malformed_v4[4].spatial_width = 1;
  malformed_v4[5].spatial_height = 1;
  malformed_v4[6].spatial_offset_count = 1;
  malformed_v4[7].spatial_edge_id_count = 1;
  for (const HeaderOverrides& malformed : malformed_v4) {
    write_csr_fixture(path, 4, pair, malformed);
    require_failure(
        [&] { (void)load_outgoing_csrbin(path); },
        "standalone reader accepted malformed CSR v4 counts");
  }

  write_csr_fixture(path, 4, pair);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  require_failure(
      [&] { (void)load_outgoing_csrbin(path); },
      "standalone reader accepted a truncated CSR v4 payload");

  write_csr_fixture(path, 4, pair);
  {
    std::ofstream trailing(path, std::ios::binary | std::ios::app);
    const char byte = '\x7f';
    trailing.write(&byte, sizeof(byte));
    if (!trailing) {
      throw std::runtime_error("failed to append trailing CSR fixture byte");
    }
  }
  require_failure(
      [&] { (void)load_outgoing_csrbin(path); },
      "standalone reader accepted trailing bytes after a CSR v4 payload");

  write_csr_fixture(path, 4, PairId{});
  require_failure(
      [&] { (void)load_outgoing_csrbin(path); },
      "standalone reader accepted a zero CSR v4 artifact pair id");
}

void test_metadata_formats(const std::filesystem::path& path) {
  const PairId pair{0x123456789abcdef0ULL, 0x0fedcba987654321ULL};
  for (std::uint64_t version = 3; version <= 8; ++version) {
    write_metadata_fixture(path, version, pair);
    const RoutingMetadata metadata =
        load_routing_metadata(path, static_cast<Offset>(1));
    require(metadata.metadata_node_count == 2 &&
                metadata.edge_attr_count == 1 &&
                metadata.strings ==
                    std::vector<std::string>({"net", "site", "pin",
                                              "logical0", "logical1"}),
            "standalone reader changed metadata header/string fields");
    require(version < 5 ? !metadata.artifact_pair_id.has_value()
                        : metadata.artifact_pair_id == pair,
            "standalone reader changed metadata artifact-pair compatibility");
    require(metadata.route_requests.size() == 1,
            "standalone reader changed metadata route request count");
    const RouteRequest& request = metadata.route_requests.front();
    require(request.net_string == 0 && request.logical_net_index == 0 &&
                request.sources.size() == 1 && request.sinks.size() == 1 &&
                request.sources.front().node == 0 &&
                request.sources.front().site_string == 1 &&
                request.sources.front().pin_string == 2 &&
                request.sinks.front().node == 1 &&
                request.sinks.front().site_string == 1 &&
                request.sinks.front().pin_string == 2,
            "standalone reader changed metadata route records");
    const std::uint64_t expected_endpoint_pip =
        version == 7 || version == 8 ? 0 : kNoIndex;
    require(request.sources.front().endpoint_pip_index ==
                    expected_endpoint_pip &&
                request.sinks.front().endpoint_pip_index ==
                    expected_endpoint_pip,
            "standalone reader changed endpoint-PIP route fields");
  }

  std::vector<MetadataHeaderOverrides> malformed_v8(4);
  malformed_v8[0].logical_cell_count = 1;
  malformed_v8[1].logical_port_instance_count = 1;
  malformed_v8[2].physical_netlist_byte_count = 1;
  malformed_v8[3].logical_netlist_byte_count = 1;
  for (const MetadataHeaderOverrides& malformed : malformed_v8) {
    write_metadata_fixture(path, 8, pair, malformed);
    require_failure(
        [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
        "standalone reader accepted nonzero omitted metadata-v8 counts");
  }

  for (const bool oversized_strings : {true, false}) {
    MetadataHeaderOverrides malformed;
    const std::uint64_t beyond_u32 =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) +
        1;
    if (oversized_strings) {
      malformed.string_count = beyond_u32;
    } else {
      malformed.pip_data_count = beyond_u32;
    }
    write_metadata_fixture(path, 8, pair, malformed);
    require_failure(
        [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
        "standalone reader accepted a metadata-v8 compact count beyond "
        "UINT32_MAX");
  }

  {
    MetadataHeaderOverrides malformed;
    malformed.logical_net_count = std::numeric_limits<std::uint64_t>::max();
    write_metadata_fixture(path, 8, pair, malformed);
    require_failure(
        [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
        "standalone reader accepted an overflowing metadata-v8 logical-net "
        "table count");
  }

  for (const auto& [malformed, message] :
       std::vector<std::pair<MetadataHeaderOverrides, const char*>>{
           {[] {
              MetadataHeaderOverrides value;
              value.route_logical_net_index = 2;
              return value;
            }(),
            "standalone reader accepted an out-of-range metadata-v8 logical "
            "net index"},
           {[] {
              MetadataHeaderOverrides value;
              value.first_logical_net_name_string = 5;
              return value;
            }(),
            "standalone reader accepted an out-of-range metadata-v8 logical "
            "net string"},
           {[] {
              MetadataHeaderOverrides value;
              value.route_net_string = 5;
              return value;
            }(),
            "standalone reader accepted an out-of-range metadata-v8 route net "
            "string"},
           {[] {
              MetadataHeaderOverrides value;
              value.first_logical_net_name_string = 3;
              return value;
            }(),
            "standalone reader accepted a mismatched metadata-v8 logical net "
            "correlation"}}) {
    write_metadata_fixture(path, 8, pair, malformed);
    require_failure(
        [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
        message);
  }

  for (MetadataFixtureTruncation truncation :
       {MetadataFixtureTruncation::kCompactEdgeAttr,
        MetadataFixtureTruncation::kCompactPip,
        MetadataFixtureTruncation::kFlatLogicalNetNames}) {
    write_metadata_fixture(path, 8, pair, {}, truncation);
    require_failure(
        [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
        "standalone reader accepted a truncated metadata-v8 payload");
  }

  write_metadata_fixture(path, 8, pair);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  require_failure(
      [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
      "standalone reader accepted terminally truncated metadata v8");

  write_metadata_fixture(path, 8, pair);
  {
    std::ofstream trailing(path, std::ios::binary | std::ios::app);
    const char byte = '\x7f';
    trailing.write(&byte, sizeof(byte));
    if (!trailing) {
      throw std::runtime_error(
          "failed to append trailing standalone metadata byte");
    }
  }
  require_failure(
      [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
      "standalone reader accepted trailing bytes after metadata v8");

  write_metadata_fixture(path, 8, PairId{});
  require_failure(
      [&] { (void)load_routing_metadata(path, static_cast<Offset>(1)); },
      "standalone reader accepted a zero metadata-v8 artifact pair id");
}

}  // namespace

int main() {
  try {
    // Suppressing the standalone executable entry point leaves main_impl in
    // the included source; take its address so -Werror host builds still
    // verify the complete translation unit.
    (void)&main_impl;
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("bf_standalone_csr_format_" + std::to_string(nonce));
    std::filesystem::create_directory(directory);
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    } cleanup{directory};
    test_reader_formats(directory / "graph.csrbin");
    test_metadata_formats(directory / "routing.ifmeta.bin");
    std::cout << "BF" << BF_STANDALONE_READER
              << " standalone CSR v1-v4/metadata v3-v8 format test passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "standalone CSR/metadata format test failed: " << ex.what()
              << '\n';
    return 1;
  }
}
