// Generic metadata-driven all-sources GPU SSSP driver.
//
// This translation unit deliberately owns input parsing, metadata grouping,
// engine selection, source orchestration, progress, cumulative reporting, and
// optional JSONL output. Algorithm-specific GPU state remains behind the
// reusable engine APIs.

#include "preds_gpu_engine.hpp"
#include "../CongestionFreeRouting/interchange/import_policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace all_sssp {

using Clock = std::chrono::steady_clock;
using Offset = std::int64_t;
using Index = std::int32_t;
using ArtifactPairId = routing::interchange::InterchangeArtifactPairId;

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char kMetadataMagic[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t kMinimumCsrVersion = 1;
constexpr std::uint64_t kCurrentCsrVersion = 2;
constexpr std::uint64_t kArtifactPairCsrVersion = 2;
constexpr std::uint64_t kMinimumMetadataVersion = 3;
constexpr std::uint64_t kCurrentMetadataVersion = 5;
constexpr std::uint64_t kNodePhysicalMetadataVersion = 4;
constexpr std::uint64_t kArtifactPairMetadataVersion = 5;
constexpr std::uint64_t kOutgoingOrientation = 2;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;

template <typename Rep, typename Period>
double seconds(std::chrono::duration<Rep, Period> duration) {
  return std::chrono::duration<double>(duration).count();
}

struct HostGraph {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
  std::optional<ArtifactPairId> artifact_pair_id;
};

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = kNoIndex;
  std::uint64_t pin_string = kNoIndex;
};

struct RouteRequest {
  std::uint64_t net_string = kNoIndex;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::uint64_t version = 0;
  std::uint64_t metadata_node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::vector<std::string> strings;
  std::vector<RouteRequest> route_requests;
  std::optional<ArtifactPairId> artifact_pair_id;
};

struct Query {
  std::uint64_t ordinal = 0;
  std::size_t net_index = 0;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  std::string net_name;
  int source = -1;
  int target = -1;
};

struct SourceWork {
  int source = -1;
  std::vector<Query> queries;
  std::vector<Index> unique_targets;
};

struct WorkBuildResult {
  std::uint64_t raw_source_occurrences = 0;
  std::uint64_t unique_valid_sources = 0;
  std::uint64_t invalid_source_occurrences = 0;
  std::uint64_t raw_source_sink_queries = 0;
  std::uint64_t valid_source_sink_queries = 0;
  std::uint64_t invalid_sink_occurrences = 0;
  std::uint64_t invalid_source_queries = 0;
  std::uint64_t invalid_sink_queries = 0;
  std::uint64_t unique_target_nodes = 0;
  std::uint64_t unique_source_target_pairs = 0;
  std::vector<SourceWork> work;
};

struct Options {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::string algorithm;
  bool algorithm_set = false;
  rips_predicates_gpu::PredicateMode predicate =
      rips_predicates_gpu::PredicateMode::kInSimple;
  bool predicate_set = false;
  bool early_stop = false;
  bool termination_mode_set = false;
  std::size_t stats_every = 100;
  bool stats_every_set = false;
  std::size_t source_limit = 0;
  bool source_limit_set = false;
  std::filesystem::path paths_output;
  bool paths_output_set = false;
  bool help = false;
};

struct AlgorithmDescriptor {
  std::string_view name;
  bool implemented = false;
  bool uses_predicate = false;
  std::string_view unavailable_reason;
};

constexpr std::array<AlgorithmDescriptor, 4> kAlgorithmRegistry{{
    {"preds-gpu", true, true, ""},
    {"bf10",
     false,
     false,
     "its reusable surface lacks independent early/full and capture-path "
     "controls, empty-target full runs, explicit max/termination status, "
     "workload reservation, and all_SSSP telemetry; current dependencies are "
     "also absent from this checkout"},
    {"delta-step",
     false,
     false,
     "its vector API always target-stops and materializes paths, rejects empty "
     "targets, and lacks explicit full-vs-early termination plus all_SSSP "
     "transfer/timing telemetry"},
    {"unit-bfs",
     false,
     false,
     "its vector API always target-stops and materializes paths, rejects empty "
     "targets, and lacks explicit full-vs-early termination, all_SSSP "
     "transfer/timing telemetry, and a strict adapter unit-weight gate"},
}};

const AlgorithmDescriptor* find_algorithm_descriptor(
    std::string_view name) noexcept {
  for (const AlgorithmDescriptor& descriptor : kAlgorithmRegistry) {
    if (descriptor.name == name) {
      return &descriptor;
    }
  }
  return nullptr;
}

const AlgorithmDescriptor& algorithm_descriptor(const Options& options) {
  const AlgorithmDescriptor* descriptor =
      find_algorithm_descriptor(options.algorithm);
  if (descriptor == nullptr) {
    throw std::logic_error(
        "selected algorithm is absent from the registry");
  }
  return *descriptor;
}

std::string configured_predicate_name(const Options& options) {
  const AlgorithmDescriptor& descriptor = algorithm_descriptor(options);
  return descriptor.uses_predicate
             ? rips_predicates_gpu::predicate_mode_name(options.predicate)
             : "n/a";
}

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* what) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error(std::string(what) + " overflows uint64");
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* what) {
  if (right != 0 &&
      left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error(std::string(what) + " overflows uint64");
  }
  return left * right;
}

std::size_t checked_hash_reserve(std::size_t count,
                                 const char* what) {
  if (count >
      (std::numeric_limits<std::size_t>::max() - 1) / 2) {
    throw std::overflow_error(
        std::string(what) + " reserve capacity overflows size_t");
  }
  return count * 2 + 1;
}

std::size_t checked_host_size(std::uint64_t count, const char* what) {
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(what) +
                              " is too large for this host");
  }
  return static_cast<std::size_t>(count);
}

std::uint64_t read_u64(std::ifstream& input, const char* what) {
  std::uint64_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error(std::string("failed while reading ") + what);
  }
  return value;
}

template <typename T>
void read_array(std::ifstream& input,
                std::vector<T>& destination,
                std::uint64_t count,
                const char* what) {
  const std::size_t host_count = checked_host_size(count, what);
  if (host_count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflows");
  }
  const std::size_t bytes = host_count * sizeof(T);
  if (bytes >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(std::string(what) +
                              " exceeds stream size limits");
  }
  destination.resize(host_count);
  if (bytes == 0) {
    return;
  }
  input.read(reinterpret_cast<char*>(destination.data()),
             static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error(std::string("failed while reading ") + what);
  }
}

std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t width,
                                 const char* what) {
  return checked_multiply(count, width, what);
}

void skip_bytes(std::ifstream& input,
                std::uint64_t byte_count,
                const char* what) {
  if (byte_count == 0) {
    return;
  }
  if (byte_count >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::streamoff>::max())) {
    throw std::overflow_error(std::string(what) +
                              " is too large to seek");
  }
  const std::streampos begin = input.tellg();
  if (begin == std::streampos(-1)) {
    throw std::runtime_error(
        std::string("failed while locating ") + what);
  }
  input.seekg(0, std::ios::end);
  const std::streampos end = input.tellg();
  if (end == std::streampos(-1) || end < begin) {
    throw std::runtime_error(
        std::string("failed while sizing ") + what);
  }
  const std::uint64_t remaining =
      static_cast<std::uint64_t>(end - begin);
  if (byte_count > remaining) {
    throw std::runtime_error(
        std::string("truncated file while skipping ") + what);
  }
  input.seekg(
      begin + static_cast<std::streamoff>(byte_count),
      std::ios::beg);
  if (!input) {
    throw std::runtime_error(std::string("failed while skipping ") + what);
  }
}

std::string read_string(std::ifstream& input) {
  const std::uint64_t count = read_u64(input, "metadata string length");
  const std::size_t size = checked_host_size(count, "metadata string");
  if (size >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error("metadata string exceeds stream size limits");
  }
  std::string text(size, '\0');
  if (!text.empty()) {
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!input) {
      throw std::runtime_error(
          "failed while reading metadata string bytes");
    }
  }
  return text;
}

int node_from_u64(std::uint64_t node) {
  if (node >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(node);
}

void require_end_of_file(std::ifstream& input, const char* what) {
  const int trailing = input.peek();
  if (trailing != std::char_traits<char>::eof()) {
    throw std::runtime_error(std::string(what) +
                             " contains unexpected trailing bytes");
  }
  if (!input.eof()) {
    throw std::runtime_error(std::string("failed while checking ") + what);
  }
}

void validate_graph(const HostGraph& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph.rows > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::runtime_error(
        "outgoing CSR has too many rows for 32-bit node ids");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error(
        "outgoing CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error(
        "outgoing CSR rowptr must start at zero and end at nnz");
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    const Index destination = graph.colind[edge];
    if (destination < 0 ||
        static_cast<Offset>(destination) >= graph.cols) {
      throw std::runtime_error(
          "outgoing CSR contains an out-of-range destination");
    }
    const float weight = graph.values[edge];
    if (!std::isfinite(weight) || weight < 0.0f) {
      throw std::runtime_error(
          "outgoing CSR weights must be finite and nonnegative");
    }
  }
}

HostGraph load_graph(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open outgoing CSR file: " +
                             path.string());
  }

  char magic[sizeof(kCsrMagic)] = {};
  input.read(magic, sizeof(magic));
  if (!input ||
      std::memcmp(magic, kCsrMagic, sizeof(kCsrMagic)) != 0) {
    throw std::runtime_error(
        "input is not a recognized RIPSCSR1 file");
  }

  const std::uint64_t version =
      read_u64(input, "outgoing CSR format version");
  const std::uint64_t orientation =
      read_u64(input, "outgoing CSR orientation");
  if (version < kMinimumCsrVersion || version > kCurrentCsrVersion) {
    throw std::runtime_error(
        "unsupported RIPSCSR1 version (expected version 1 or 2)");
  }
  if (orientation != kOutgoingOrientation) {
    throw std::runtime_error(
        "unsupported RIPSCSR1 orientation (expected outgoing orientation 2)");
  }

  std::optional<ArtifactPairId> artifact_pair_id;
  if (version >= kArtifactPairCsrVersion) {
    const ArtifactPairId id{
        read_u64(input, "outgoing CSR artifact pair id high"),
        read_u64(input, "outgoing CSR artifact pair id low")};
    if (id.is_zero()) {
      throw std::runtime_error(
          "outgoing CSR artifact pair id must not be zero");
    }
    artifact_pair_id = id;
  }

  const std::uint64_t rows = read_u64(input, "outgoing CSR row count");
  const std::uint64_t cols = read_u64(input, "outgoing CSR column count");
  const std::uint64_t declared_edges =
      read_u64(input, "outgoing CSR declared edge count");
  const std::uint64_t loaded_edges =
      read_u64(input, "outgoing CSR loaded edge count");
  const std::uint64_t nnz = read_u64(input, "outgoing CSR nnz");
  const std::uint64_t rowptr_count =
      read_u64(input, "outgoing CSR rowptr count");
  const std::uint64_t colind_count =
      read_u64(input, "outgoing CSR colind count");
  const std::uint64_t values_count =
      read_u64(input, "outgoing CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows >
          static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows >
          static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz >
          static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this API");
  }
  if (loaded_edges > declared_edges || nnz > loaded_edges) {
    throw std::runtime_error(
        "outgoing CSR declared/loaded/nnz counts are inconsistent");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz ||
      values_count != nnz) {
    throw std::runtime_error(
        "outgoing CSR rowptr/colind/values counts are inconsistent");
  }

  HostGraph graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  graph.declared_edges = declared_edges;
  graph.loaded_edges = loaded_edges;
  graph.artifact_pair_id = artifact_pair_id;
  read_array(input, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  read_array(input, graph.colind, colind_count, "outgoing CSR colind");
  read_array(input, graph.values, values_count, "outgoing CSR values");
  require_end_of_file(input, "outgoing CSR");
  validate_graph(graph);
  return graph;
}

RoutingMetadata load_metadata(const std::filesystem::path& path,
                              Offset graph_nnz) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open metadata file: " +
                             path.string());
  }

  char magic[sizeof(kMetadataMagic)] = {};
  input.read(magic, sizeof(magic));
  if (!input ||
      std::memcmp(magic, kMetadataMagic, sizeof(kMetadataMagic)) != 0) {
    throw std::runtime_error(
        "input is not a recognized RIPSIFM1 metadata file");
  }

  const std::uint64_t version =
      read_u64(input, "metadata format version");
  const std::uint64_t orientation =
      read_u64(input, "metadata orientation");
  if (version < kMinimumMetadataVersion ||
      version > kCurrentMetadataVersion) {
    throw std::runtime_error(
        "unsupported RIPSIFM1 version (expected version 3, 4, or 5)");
  }
  if (orientation != kOutgoingOrientation) {
    throw std::runtime_error(
        "unsupported RIPSIFM1 orientation (expected outgoing orientation 2)");
  }

  std::optional<ArtifactPairId> artifact_pair_id;
  if (version >= kArtifactPairMetadataVersion) {
    const ArtifactPairId id{
        read_u64(input, "metadata artifact pair id high"),
        read_u64(input, "metadata artifact pair id low")};
    if (id.is_zero()) {
      throw std::runtime_error(
          "metadata artifact pair id must not be zero");
    }
    artifact_pair_id = id;
  }

  const std::uint64_t string_count =
      read_u64(input, "metadata string count");
  const std::uint64_t node_count =
      read_u64(input, "metadata node count");
  const std::uint64_t edge_attr_count =
      read_u64(input, "metadata edge attribute count");
  if (graph_nnz < 0 ||
      edge_attr_count != static_cast<std::uint64_t>(graph_nnz)) {
    throw std::runtime_error(
        "metadata edge attribute count does not match outgoing CSR nnz");
  }
  const std::uint64_t pip_data_count =
      read_u64(input, "metadata pip data count");
  const std::uint64_t site_pin_attr_count =
      read_u64(input, "metadata site pin attr count");
  const std::uint64_t route_request_count =
      read_u64(input, "metadata route request count");
  const std::uint64_t blocked_node_count =
      read_u64(input, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count =
      read_u64(input, "metadata sink stop node count");
  const std::uint64_t logical_cell_count =
      read_u64(input, "metadata logical cell count");
  const std::uint64_t logical_net_count =
      read_u64(input, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(input, "metadata logical port instance count");
  const std::uint64_t physical_bytes =
      read_u64(input, "metadata physical byte count");
  const std::uint64_t logical_bytes =
      read_u64(input, "metadata logical byte count");

  (void)read_u64(input, "metadata device path string");
  (void)read_u64(input, "metadata physical path string");
  (void)read_u64(input, "metadata logical path string");
  (void)read_u64(input, "metadata logical design name");

  RoutingMetadata metadata;
  metadata.version = version;
  metadata.metadata_node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.artifact_pair_id = artifact_pair_id;
  metadata.strings.reserve(
      checked_host_size(string_count, "metadata string count"));
  for (std::uint64_t index = 0; index < string_count; ++index) {
    metadata.strings.push_back(read_string(input));
  }

  skip_bytes(
      input,
      checked_byte_count(node_count, sizeof(std::uint64_t),
                         "metadata device node ids"),
      "metadata device node ids");
  if (version >= kNodePhysicalMetadataVersion) {
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "metadata node min x"),
               "metadata node min x");
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "metadata node max x"),
               "metadata node max x");
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "metadata node min y"),
               "metadata node min y");
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "metadata node max y"),
               "metadata node max y");
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "metadata node tile type strings"),
               "metadata node tile type strings");
    skip_bytes(input,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "metadata node wire type strings"),
               "metadata node wire type strings");
  }
  skip_bytes(
      input,
      checked_byte_count(edge_attr_count, 2 * sizeof(std::uint64_t),
                         "metadata edge attrs"),
      "metadata edge attrs");
  skip_bytes(
      input,
      checked_byte_count(pip_data_count, 3 * sizeof(std::uint64_t),
                         "metadata pip data"),
      "metadata pip data");
  skip_bytes(
      input,
      checked_byte_count(site_pin_attr_count, 3 * sizeof(std::uint64_t),
                         "metadata site pin attrs"),
      "metadata site pin attrs");

  metadata.route_requests.resize(
      checked_host_size(route_request_count, "metadata route requests"));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(input, "metadata route request net");
    request.logical_net_index =
        read_u64(input, "metadata route request logical net");
    const std::uint64_t source_count =
        read_u64(input, "metadata source count");
    request.sources.resize(
        checked_host_size(source_count, "metadata sources"));
    for (SitePinNode& source : request.sources) {
      source.node =
          node_from_u64(read_u64(input, "metadata source node"));
      source.site_string = read_u64(input, "metadata source site");
      source.pin_string = read_u64(input, "metadata source pin");
    }
    const std::uint64_t sink_count =
        read_u64(input, "metadata sink count");
    request.sinks.resize(
        checked_host_size(sink_count, "metadata sinks"));
    for (SitePinNode& sink : request.sinks) {
      sink.node = node_from_u64(read_u64(input, "metadata sink node"));
      sink.site_string = read_u64(input, "metadata sink site");
      sink.pin_string = read_u64(input, "metadata sink pin");
    }
  }

  skip_bytes(
      input,
      checked_byte_count(logical_cell_count, 3 * sizeof(std::uint64_t),
                         "metadata logical cells"),
      "metadata logical cells");
  skip_bytes(
      input,
      checked_byte_count(logical_net_count, 4 * sizeof(std::uint64_t),
                         "metadata logical nets"),
      "metadata logical nets");
  skip_bytes(
      input,
      checked_byte_count(logical_port_instance_count,
                         7 * sizeof(std::uint64_t),
                         "metadata logical port instances"),
      "metadata logical port instances");
  skip_bytes(
      input,
      checked_byte_count(blocked_node_count, sizeof(std::uint64_t),
                         "metadata blocked nodes"),
      "metadata blocked nodes");
  // sink_stop_nodes describe rows suppressed by conversion. They are not the
  // route request's SSSP endpoints and are intentionally skipped here.
  skip_bytes(
      input,
      checked_byte_count(sink_stop_node_count, sizeof(std::uint64_t),
                         "metadata sink stop nodes"),
      "metadata sink stop nodes");
  skip_bytes(input, physical_bytes, "metadata physical bytes");
  skip_bytes(input, logical_bytes, "metadata logical bytes");
  require_end_of_file(input, "metadata");
  return metadata;
}

bool valid_node(Offset rows, int node) {
  return node >= 0 && static_cast<Offset>(node) < rows;
}

std::string string_at(const RoutingMetadata& metadata,
                      std::uint64_t index) {
  if (index == kNoIndex) {
    return "";
  }
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

std::vector<Index> unique_targets_for_queries(
    const std::vector<Query>& queries) {
  std::vector<Index> targets;
  targets.reserve(queries.size());
  std::unordered_set<int> seen;
  seen.reserve(
      checked_hash_reserve(queries.size(), "unique target set"));
  for (const Query& query : queries) {
    if (seen.insert(query.target).second) {
      targets.push_back(static_cast<Index>(query.target));
    }
  }
  return targets;
}

WorkBuildResult build_source_work(const RoutingMetadata& metadata,
                                  Offset rows,
                                  std::ostream& warnings) {
  WorkBuildResult result;
  std::unordered_map<int, std::size_t> source_to_work;
  source_to_work.reserve(
      checked_hash_reserve(
          metadata.route_requests.size(), "source grouping map"));
  std::unordered_set<int> global_targets;
  std::uint64_t query_ordinal = 0;

  for (std::size_t net_index = 0;
       net_index < metadata.route_requests.size();
       ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const std::string net_name = string_at(metadata, request.net_string);

    const std::uint64_t source_count =
        static_cast<std::uint64_t>(request.sources.size());
    const std::uint64_t sink_count =
        static_cast<std::uint64_t>(request.sinks.size());
    result.raw_source_sink_queries = checked_add(
        result.raw_source_sink_queries,
        checked_multiply(source_count, sink_count,
                         "raw source-sink query count"),
        "raw source-sink query count");

    std::uint64_t invalid_sinks_in_request = 0;
    for (const SitePinNode& sink : request.sinks) {
      if (!valid_node(rows, sink.node)) {
        ++invalid_sinks_in_request;
        result.invalid_sink_occurrences = checked_add(
            result.invalid_sink_occurrences, 1,
            "invalid sink occurrence count");
        warnings << "[all_SSSP] warning: skipping out-of-range sink node "
                 << sink.node << " for net_index=" << net_index << '\n';
      }
    }
    result.invalid_sink_queries = checked_add(
        result.invalid_sink_queries,
        checked_multiply(source_count, invalid_sinks_in_request,
                         "invalid sink query count"),
        "invalid sink query count");

    for (const SitePinNode& source : request.sources) {
      result.raw_source_occurrences = checked_add(
          result.raw_source_occurrences, 1,
          "raw source occurrence count");
      const bool source_valid = valid_node(rows, source.node);
      if (!source_valid) {
        result.invalid_source_occurrences = checked_add(
            result.invalid_source_occurrences, 1,
            "invalid source occurrence count");
        result.invalid_source_queries = checked_add(
            result.invalid_source_queries, sink_count,
            "invalid source query count");
        warnings << "[all_SSSP] warning: skipping out-of-range source node "
                 << source.node << " for net_index=" << net_index << '\n';
      }

      std::size_t work_index = 0;
      if (source_valid) {
        const auto found = source_to_work.find(source.node);
        if (found == source_to_work.end()) {
          work_index = result.work.size();
          source_to_work.emplace(source.node, work_index);
          SourceWork source_work;
          source_work.source = source.node;
          result.work.push_back(std::move(source_work));
        } else {
          work_index = found->second;
        }
      }

      for (const SitePinNode& sink : request.sinks) {
        const std::uint64_t ordinal = query_ordinal;
        query_ordinal = checked_add(
            query_ordinal, 1, "source-sink query ordinal");
        if (!source_valid || !valid_node(rows, sink.node)) {
          continue;
        }
        Query query;
        query.ordinal = ordinal;
        query.net_index = net_index;
        query.logical_net_index = request.logical_net_index;
        query.net_name = net_name;
        query.source = source.node;
        query.target = sink.node;
        result.work[work_index].queries.push_back(std::move(query));
        result.valid_source_sink_queries = checked_add(
            result.valid_source_sink_queries, 1,
            "valid source-sink query count");
        global_targets.insert(sink.node);
      }
    }
  }

  result.unique_valid_sources =
      static_cast<std::uint64_t>(result.work.size());
  result.unique_target_nodes =
      static_cast<std::uint64_t>(global_targets.size());
  for (SourceWork& source : result.work) {
    source.unique_targets = unique_targets_for_queries(source.queries);
    result.unique_source_target_pairs = checked_add(
        result.unique_source_target_pairs,
        static_cast<std::uint64_t>(source.unique_targets.size()),
        "unique source-target pair count");
  }
  return result;
}

std::uint64_t count_selected_queries(
    const std::vector<SourceWork>& work,
    std::size_t source_count) {
  if (source_count > work.size()) {
    throw std::out_of_range("selected source count exceeds source work");
  }
  std::uint64_t total = 0;
  for (std::size_t index = 0; index < source_count; ++index) {
    total = checked_add(
        total,
        static_cast<std::uint64_t>(work[index].queries.size()),
        "selected query count");
  }
  return total;
}

std::vector<const Query*> selected_queries_in_metadata_order(
    const std::vector<SourceWork>& work,
    std::size_t source_count) {
  const std::uint64_t query_count =
      count_selected_queries(work, source_count);
  std::vector<const Query*> ordered;
  ordered.reserve(
      checked_host_size(query_count, "selected path query count"));
  for (std::size_t source_index = 0;
       source_index < source_count;
       ++source_index) {
    for (const Query& query : work[source_index].queries) {
      ordered.push_back(&query);
    }
  }
  std::sort(
      ordered.begin(),
      ordered.end(),
      [](const Query* left, const Query* right) {
        return left->ordinal < right->ordinal;
      });
  return ordered;
}

std::size_t max_unique_targets_per_source(
    const std::vector<SourceWork>& work,
    std::size_t source_count,
    Offset rows) {
  if (source_count > work.size()) {
    throw std::out_of_range("selected source count exceeds source work");
  }
  std::size_t maximum = 0;
  for (std::size_t index = 0; index < source_count; ++index) {
    maximum = std::max(maximum, work[index].unique_targets.size());
  }
  if (rows < 0 || maximum > static_cast<std::size_t>(rows)) {
    throw std::runtime_error("unique target count exceeds graph rows");
  }
  return maximum;
}

std::size_t selected_source_count(std::size_t available,
                                  std::size_t source_limit) {
  return source_limit == 0
             ? available
             : std::min(source_limit, available);
}

std::uint64_t parse_u64(const std::string& text,
                        const char* option_name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(
        std::string(option_name) + " requires a nonnegative integer");
  }
  std::size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(
        std::string(option_name) + " requires a nonnegative integer");
  }
  if (parsed != text.size()) {
    throw std::invalid_argument(
        std::string(option_name) + " requires a nonnegative integer");
  }
  return static_cast<std::uint64_t>(value);
}

std::size_t parse_size(const std::string& text,
                       const char* option_name) {
  const std::uint64_t value = parse_u64(text, option_name);
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::out_of_range(std::string(option_name) +
                            " is too large for this host");
  }
  return static_cast<std::size_t>(value);
}

bool starts_with_dash(const std::string& text) {
  return !text.empty() && text.front() == '-';
}

void print_usage(std::ostream& output, const char* program) {
  output
      << "Usage:\n"
      << "  " << program
      << " <graph.csrbin> <graph.csrbin.ifmeta.bin> [options]\n\n"
      << "Implemented GPU algorithms:\n";
  for (const AlgorithmDescriptor& descriptor : kAlgorithmRegistry) {
    if (descriptor.implemented) {
      output << "  " << descriptor.name << '\n';
    }
  }
  output
      << "\nOptions:\n"
      << "  --algorithm <name>       Select the GPU SSSP engine (required).\n"
      << "  --predicate <MODE>       Required for preds-gpu. Modes:\n"
      << "                           IN_SIMPLE, OUT_SIMPLE,\n"
      << "                           IN_SIMPLE_OR_OUT_SIMPLE, IN_STATIC,\n"
      << "                           OUT_STATIC, IN_STATIC_OR_OUT_STATIC\n"
      << "  --early-stop             Stop after every unique sink is settled.\n"
      << "  --no-early-stop          Run every source to full convergence "
         "(default).\n"
      << "  --stats-every <count>    Print a cumulative snapshot every count "
         "sources.\n"
      << "                           Zero disables intermediate snapshots.\n"
      << "  --source-limit <count>   Process at most count unique sources; "
         "zero means all.\n"
      << "  --paths-output <path>    Write metadata query paths as JSONL.\n"
      << "  -h, --help               Show this help message.\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  std::vector<std::string> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto require_value = [&](const char* option_name) -> std::string {
      if (index + 1 >= argc ||
          starts_with_dash(std::string(argv[index + 1]))) {
        throw std::invalid_argument(
            std::string(option_name) + " requires a value");
      }
      return argv[++index];
    };

    if (argument == "-h" || argument == "--help") {
      if (options.help) {
        throw std::invalid_argument(
            "-h/--help may be specified only once");
      }
      options.help = true;
    } else if (argument == "--algorithm") {
      if (options.algorithm_set) {
        throw std::invalid_argument(
            "--algorithm may be specified only once");
      }
      options.algorithm = require_value("--algorithm");
      options.algorithm_set = true;
    } else if (argument == "--predicate") {
      if (options.predicate_set) {
        throw std::invalid_argument(
            "--predicate may be specified only once");
      }
      options.predicate = rips_predicates_gpu::parse_predicate_mode(
          require_value("--predicate"));
      options.predicate_set = true;
    } else if (argument == "--early-stop" ||
               argument == "--no-early-stop") {
      if (options.termination_mode_set) {
        throw std::invalid_argument(
            "--early-stop/--no-early-stop may be specified only once");
      }
      options.early_stop = argument == "--early-stop";
      options.termination_mode_set = true;
    } else if (argument == "--stats-every") {
      if (options.stats_every_set) {
        throw std::invalid_argument(
            "--stats-every may be specified only once");
      }
      options.stats_every =
          parse_size(require_value("--stats-every"), "--stats-every");
      options.stats_every_set = true;
    } else if (argument == "--source-limit") {
      if (options.source_limit_set) {
        throw std::invalid_argument(
            "--source-limit may be specified only once");
      }
      options.source_limit =
          parse_size(require_value("--source-limit"), "--source-limit");
      options.source_limit_set = true;
    } else if (argument == "--paths-output") {
      if (options.paths_output_set) {
        throw std::invalid_argument(
            "--paths-output may be specified only once");
      }
      options.paths_output = require_value("--paths-output");
      options.paths_output_set = true;
    } else if (starts_with_dash(argument)) {
      throw std::invalid_argument("unknown option: " + argument);
    } else {
      positional.push_back(argument);
    }
  }

  if (options.help) {
    return options;
  }
  if (positional.size() != 2) {
    throw std::invalid_argument(
        "expected <graph.csrbin> <graph.csrbin.ifmeta.bin>");
  }
  options.csr_path = positional[0];
  options.metadata_path = positional[1];
  if (!options.algorithm_set) {
    throw std::invalid_argument("--algorithm <name> is required");
  }
  const AlgorithmDescriptor* descriptor =
      find_algorithm_descriptor(options.algorithm);
  if (descriptor == nullptr) {
    throw std::invalid_argument(
        "unknown algorithm '" + options.algorithm +
        "'; recognized names: preds-gpu, bf10, delta-step, unit-bfs");
  }
  if (options.predicate_set && !descriptor->uses_predicate) {
    throw std::invalid_argument(
        "--predicate is not valid for algorithm '" + options.algorithm + "'");
  }
  if (descriptor->uses_predicate && !options.predicate_set) {
    throw std::invalid_argument(
        "--predicate <MODE> is required for algorithm '" +
        options.algorithm + "'");
  }
  if (!descriptor->implemented) {
    throw std::invalid_argument(
        "algorithm '" + options.algorithm +
        "' is recognized but unavailable in this build: " +
        std::string(descriptor->unavailable_reason));
  }
  if (options.paths_output_set) {
    routing::interchange::require_distinct_interchange_paths(
        {options.paths_output,
         options.csr_path,
         options.metadata_path,
         routing::interchange::interchange_publication_generation_path(
             options.metadata_path),
         routing::interchange::interchange_publication_marker_path(
             options.csr_path),
         routing::interchange::interchange_publication_marker_path(
             options.metadata_path)});
  }
  return options;
}

enum class TerminationReason {
  kFullConvergence,
  kAllTargetsConfirmed,
  kMaxIterations,
  kError,
};

const char* termination_reason_name(TerminationReason reason) {
  switch (reason) {
    case TerminationReason::kFullConvergence:
      return "full_convergence";
    case TerminationReason::kAllTargetsConfirmed:
      return "all_targets_confirmed";
    case TerminationReason::kMaxIterations:
      return "max_iterations";
    case TerminationReason::kError:
      return "error";
  }
  return "error";
}

struct TargetResult {
  Index target = -1;
  bool reached = false;
  float distance = std::numeric_limits<float>::infinity();
  bool path_captured = false;
  std::vector<Index> nodes;
  std::vector<Offset> csr_edges;
};

struct PerRunStatistics {
  std::uint64_t vertices_settled = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxation_attempts = 0;
  std::uint64_t successful_distance_updates = 0;
  std::uint64_t kernel_launches = 0;
  std::uint64_t reset_kernel_launches = 0;
  std::uint64_t target_h2d_transfers = 0;
  std::uint64_t target_h2d_bytes = 0;
  std::uint64_t d2h_transfers = 0;
  std::uint64_t d2h_bytes = 0;
  double predicate_gpu_ms = 0.0;
  double relaxation_gpu_ms = 0.0;
  double reset_gpu_ms = 0.0;
  double target_transfer_ms = 0.0;
  double d2h_transfer_ms = 0.0;
  double gpu_sssp_ms = 0.0;
  double run_wall_ms = 0.0;
};

struct SsspRequest {
  Index source = -1;
  std::vector<Index> unique_targets;
  bool early_stop = false;
  bool capture_paths = false;
};

struct SsspRunResult {
  TerminationReason termination = TerminationReason::kError;
  bool fully_converged = false;
  bool all_targets_confirmed = false;
  std::uint64_t phases_or_iterations = 0;
  PerRunStatistics statistics;
  std::vector<TargetResult> targets;
};

struct EngineStatistics {
  std::uint64_t runs = 0;
  std::uint64_t initial_graph_h2d_transfers = 0;
  std::uint64_t initial_graph_h2d_bytes = 0;
  std::uint64_t target_h2d_transfers = 0;
  std::uint64_t target_h2d_bytes = 0;
  std::uint64_t d2h_transfers = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t kernel_launches = 0;
  std::uint64_t reset_kernel_launches = 0;
  double preprocessing_ms = 0.0;
  double initial_graph_transfer_ms = 0.0;
  double reset_gpu_ms = 0.0;
  double target_transfer_ms = 0.0;
  double d2h_transfer_ms = 0.0;
  double predicate_gpu_ms = 0.0;
  double relaxation_gpu_ms = 0.0;
  double gpu_sssp_ms = 0.0;
  double run_wall_ms = 0.0;
  std::size_t allocated_device_bytes = 0;
};

class GpuSsspEngine {
 public:
  virtual ~GpuSsspEngine() = default;
  virtual const char* algorithm_name() const noexcept = 0;
  virtual const char* predicate_name() const noexcept = 0;
  virtual SsspRunResult run(const SsspRequest& request) = 0;
  virtual void validate_captured_target_paths(
      Index source,
      const std::vector<TargetResult>& targets) const = 0;
  virtual EngineStatistics cumulative_statistics() const = 0;
};

#ifndef ALL_SSSP_HOST_TEST

class PredsGpuEngine final : public GpuSsspEngine {
 public:
  PredsGpuEngine(HostGraph graph,
                 rips_predicates_gpu::PredicateMode predicate,
                 std::size_t maximum_targets,
                 bool reserve_paths)
      : predicate_(predicate),
        context_(to_preds_graph(std::move(graph)),
                 predicate,
                 maximum_targets,
                 reserve_paths,
                 false) {}

  const char* algorithm_name() const noexcept override {
    return "preds-gpu";
  }

  const char* predicate_name() const noexcept override {
    return rips_predicates_gpu::predicate_mode_name(predicate_);
  }

  SsspRunResult run(const SsspRequest& request) override {
    rips_predicates_gpu::RunRequest native_request;
    native_request.source = static_cast<rips_predicates_gpu::Index>(
        request.source);
    native_request.unique_targets = request.unique_targets;
    native_request.early_stop = request.early_stop;
    native_request.capture_paths = request.capture_paths;
    native_request.capture_phase_histogram = false;

    const rips_predicates_gpu::RunResult native =
        context_.run(native_request);
    SsspRunResult result;
    result.termination =
        native.termination ==
                rips_predicates_gpu::TerminationReason::kFullConvergence
            ? TerminationReason::kFullConvergence
            : TerminationReason::kAllTargetsConfirmed;
    result.fully_converged = native.fully_converged;
    result.all_targets_confirmed = native.all_targets_confirmed;
    result.phases_or_iterations = native.phases_or_iterations;

    const auto& algorithm = native.telemetry.algorithm;
    const auto& transfers = native.telemetry.transfers;
    const auto& launches = native.telemetry.launches;
    result.statistics.vertices_settled = algorithm.vertices_settled;
    result.statistics.edges_examined = algorithm.edges_examined;
    result.statistics.relaxation_attempts =
        algorithm.relaxations_attempted;
    result.statistics.successful_distance_updates =
        algorithm.successful_distance_updates;
    result.statistics.kernel_launches = checked_add(
        launches.sssp, launches.output_reconstruction,
        "per-run kernel launch count");
    result.statistics.reset_kernel_launches =
        algorithm.reset_kernel_launches;
    result.statistics.target_h2d_transfers = transfers.h2d_operations;
    result.statistics.target_h2d_bytes = transfers.h2d_bytes;
    result.statistics.d2h_transfers = transfers.d2h_operations;
    result.statistics.d2h_bytes = transfers.d2h_bytes;
    result.statistics.predicate_gpu_ms =
        algorithm.predicate_evaluation_ms;
    result.statistics.relaxation_gpu_ms = algorithm.relaxation_ms;
    result.statistics.reset_gpu_ms = algorithm.reset_ms;
    result.statistics.target_transfer_ms = transfers.h2d_ms;
    result.statistics.d2h_transfer_ms = transfers.d2h_ms;
    result.statistics.gpu_sssp_ms =
        algorithm.reset_ms + algorithm.predicate_evaluation_ms +
        algorithm.relaxation_ms;
    result.statistics.run_wall_ms = native.telemetry.runtime_ms;

    result.targets.reserve(native.targets.size());
    for (const rips_predicates_gpu::TargetResult& target :
         native.targets) {
      TargetResult converted;
      converted.target = target.target;
      converted.reached = target.reached;
      converted.distance = target.distance;
      converted.path_captured = target.path_captured;
      converted.nodes.assign(target.nodes.begin(), target.nodes.end());
      converted.csr_edges.assign(
          target.csr_edges.begin(), target.csr_edges.end());
      result.targets.push_back(std::move(converted));
    }
    return result;
  }

  void validate_captured_target_paths(
      Index source,
      const std::vector<TargetResult>& targets) const override {
    for (const TargetResult& target : targets) {
      const rips_predicates_gpu::CapturedTargetPathView view{
          target.target,
          target.distance,
          target.reached,
          target.path_captured,
          target.nodes.data(),
          target.nodes.size(),
          target.csr_edges.data(),
          target.csr_edges.size()};
      context_.validate_captured_target_path(source, view);
    }
  }

  EngineStatistics cumulative_statistics() const override {
    const rips_predicates_gpu::CumulativeStatistics native =
        context_.cumulative_statistics();
    EngineStatistics result;
    result.runs = native.runs;
    result.initial_graph_h2d_transfers =
        native.graph_setup_transfers.h2d_operations;
    result.initial_graph_h2d_bytes =
        native.graph_setup_transfers.h2d_bytes;
    result.target_h2d_transfers =
        native.run_transfers.h2d_operations;
    result.target_h2d_bytes = native.run_transfers.h2d_bytes;
    result.d2h_transfers = native.run_transfers.d2h_operations;
    result.d2h_bytes = native.run_transfers.d2h_bytes;
    result.kernel_launches = checked_add(
        native.launches.sssp, native.launches.output_reconstruction,
        "cumulative kernel launch count");
    result.reset_kernel_launches =
        native.algorithm.reset_kernel_launches;
    result.preprocessing_ms = native.preprocessing_ms;
    result.initial_graph_transfer_ms =
        native.graph_setup_transfers.h2d_ms;
    result.reset_gpu_ms = native.algorithm.reset_ms;
    result.target_transfer_ms = native.run_transfers.h2d_ms;
    result.d2h_transfer_ms = native.run_transfers.d2h_ms;
    result.predicate_gpu_ms =
        native.algorithm.predicate_evaluation_ms;
    result.relaxation_gpu_ms = native.algorithm.relaxation_ms;
    result.gpu_sssp_ms =
        native.algorithm.reset_ms +
        native.algorithm.predicate_evaluation_ms +
        native.algorithm.relaxation_ms;
    result.run_wall_ms = native.runtime_ms;
    result.allocated_device_bytes = native.allocated_device_bytes;
    return result;
  }

 private:
  static rips_predicates_gpu::CsrGraph to_preds_graph(
      HostGraph graph) {
    rips_predicates_gpu::CsrGraph result;
    result.rows = graph.rows;
    result.cols = graph.cols;
    result.nnz = graph.nnz;
    result.rowptr = std::move(graph.rowptr);
    result.colind = std::move(graph.colind);
    result.values = std::move(graph.values);
    return result;
  }

  rips_predicates_gpu::PredicateMode predicate_;
  rips_predicates_gpu::PredsGpuContext context_;
};

std::unique_ptr<GpuSsspEngine> create_engine(
    const Options& options,
    HostGraph graph,
    std::size_t maximum_targets) {
  const AlgorithmDescriptor& descriptor = algorithm_descriptor(options);
  if (descriptor.name == "preds-gpu" && descriptor.implemented) {
    return std::make_unique<PredsGpuEngine>(
        std::move(graph),
        options.predicate,
        maximum_targets,
        options.paths_output_set);
  }
  throw std::invalid_argument(
      "no implemented adapter is registered for algorithm '" +
      options.algorithm + "'");
}

#endif  // ALL_SSSP_HOST_TEST

bool stdout_is_terminal() {
#if defined(_WIN32)
  return ::_isatty(::_fileno(stdout)) != 0;
#else
  return ::isatty(::fileno(stdout)) != 0;
#endif
}

class ProgressReporter {
 public:
  ProgressReporter(std::size_t total,
                   std::size_t statistics_interval,
                   Clock::time_point begin,
                   std::ostream& output = std::cout,
                   std::optional<bool> terminal_override = std::nullopt)
      : total_(total),
        nonterminal_interval_(
            statistics_interval != 0
                ? statistics_interval
                : std::max<std::size_t>(
                      1, total == 0 ? 1 : (total + 9) / 10)),
        begin_(begin),
        output_(output),
        terminal_(terminal_override.value_or(
            &output == &std::cout && stdout_is_terminal())),
        last_refresh_(begin - std::chrono::seconds(1)) {}

  ~ProgressReporter() {
    if (terminal_ && line_open_) {
      output_ << '\n';
    }
  }

  void update(std::size_t completed, bool force = false) {
    const Clock::time_point now = Clock::now();
    if (terminal_) {
      if (!force &&
          now - last_refresh_ < std::chrono::milliseconds(100)) {
        return;
      }
      output_ << '\r' << make_record(completed, now) << "\033[K"
              << std::flush;
      last_refresh_ = now;
      line_open_ = true;
      if (completed >= total_) {
        output_ << '\n';
        line_open_ = false;
      }
      return;
    }

    const bool periodic =
        completed != 0 &&
        completed % nonterminal_interval_ == 0;
    const bool final = completed >= total_;
    if ((force || periodic || final) &&
        (!last_nonterminal_.has_value() ||
         *last_nonterminal_ != completed)) {
      output_ << make_record(completed, now) << '\n';
      last_nonterminal_ = completed;
    }
  }

  void finish_line() {
    if (terminal_ && line_open_) {
      output_ << '\n';
      line_open_ = false;
    }
  }

  void before_block() {
    finish_line();
  }

  void after_block(std::size_t completed) {
    if (terminal_ && completed < total_) {
      update(completed, true);
    }
  }

 private:
  std::string make_record(std::size_t completed,
                          Clock::time_point now) const {
    completed = std::min(completed, total_);
    const double elapsed = std::max(0.0, seconds(now - begin_));
    const double fraction =
        total_ == 0
            ? 1.0
            : static_cast<double>(completed) /
                  static_cast<double>(total_);
    const double rate =
        elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;

    constexpr int width = 20;
    const int filled = static_cast<int>(
        std::floor(std::min(1.0, fraction) * width));
    std::string bar;
    bar.reserve(width);
    for (int index = 0; index < width; ++index) {
      bar.push_back(index < filled ? '#' : '-');
    }

    std::ostringstream record;
    record << "[all_SSSP] sources [" << bar << "] "
           << std::fixed << std::setprecision(1)
           << (100.0 * std::min(1.0, fraction)) << "% "
           << completed << "/" << total_
           << " elapsed=" << elapsed << "s"
           << " rate=" << rate << " src/s";
    if (rate > 0.0 && completed < total_) {
      record << " ETA="
             << static_cast<double>(total_ - completed) / rate << "s";
    } else if (completed < total_) {
      record << " ETA=n/a";
    } else {
      record << " ETA=0.0s";
    }
    return record.str();
  }

  std::size_t total_ = 0;
  std::size_t nonterminal_interval_ = 1;
  Clock::time_point begin_;
  std::ostream& output_;
  bool terminal_ = false;
  bool line_open_ = false;
  Clock::time_point last_refresh_;
  std::optional<std::size_t> last_nonterminal_;
};

struct AggregateStatistics {
  std::uint64_t completed = 0;
  std::uint64_t queries_covered = 0;
  std::uint64_t full_convergence = 0;
  std::uint64_t early_stops = 0;
  std::uint64_t max_iterations = 0;
  std::uint64_t failures = 0;
  std::uint64_t every_sink_reachable = 0;
  std::uint64_t contains_unreachable_sink = 0;
  std::uint64_t total_phases = 0;
  std::optional<std::uint64_t> minimum_phases;
  std::uint64_t maximum_phases = 0;
  std::uint64_t vertices_settled = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxation_attempts = 0;
  std::uint64_t successful_distance_updates = 0;

  void add_success(const SourceWork& work,
                   const SsspRunResult& result) {
    completed = checked_add(completed, 1, "completed source count");
    queries_covered = checked_add(
        queries_covered,
        static_cast<std::uint64_t>(work.queries.size()),
        "covered query count");
    switch (result.termination) {
      case TerminationReason::kFullConvergence:
        full_convergence =
            checked_add(full_convergence, 1, "full convergence count");
        break;
      case TerminationReason::kAllTargetsConfirmed:
        early_stops = checked_add(early_stops, 1, "early stop count");
        break;
      case TerminationReason::kMaxIterations:
        max_iterations =
            checked_add(max_iterations, 1, "max iteration count");
        break;
      case TerminationReason::kError:
        failures = checked_add(failures, 1, "failure count");
        break;
    }

    bool all_reached = true;
    for (const TargetResult& target : result.targets) {
      all_reached = all_reached && target.reached;
    }
    if (all_reached) {
      every_sink_reachable = checked_add(
          every_sink_reachable, 1,
          "every-sink-reachable source count");
    } else {
      contains_unreachable_sink = checked_add(
          contains_unreachable_sink, 1,
          "unreachable-sink source count");
    }

    total_phases = checked_add(
        total_phases, result.phases_or_iterations,
        "phase/iteration count");
    minimum_phases =
        minimum_phases.has_value()
            ? std::min(*minimum_phases, result.phases_or_iterations)
            : result.phases_or_iterations;
    maximum_phases =
        std::max(maximum_phases, result.phases_or_iterations);
    vertices_settled = checked_add(
        vertices_settled, result.statistics.vertices_settled,
        "settled vertex count");
    edges_examined = checked_add(
        edges_examined, result.statistics.edges_examined,
        "examined edge count");
    relaxation_attempts = checked_add(
        relaxation_attempts, result.statistics.relaxation_attempts,
        "relaxation attempt count");
    successful_distance_updates = checked_add(
        successful_distance_updates,
        result.statistics.successful_distance_updates,
        "successful update count");
  }
};

std::string json_escape(const std::string& text) {
  std::ostringstream output;
  for (const unsigned char character : text) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4)
                 << std::setfill('0') << static_cast<int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

void write_json_string(std::ostream& output, const std::string& text) {
  output << '"' << json_escape(text) << '"';
}

std::ofstream open_paths_output(const std::filesystem::path& path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open paths output: " +
                             path.string());
  }
  return output;
}

void write_paths_metadata(std::ostream& output,
                          Offset rows,
                          Offset nnz,
                          const RoutingMetadata& metadata,
                          const Options& options,
                          const std::string& predicate_name,
                          std::size_t selected_sources,
                          std::uint64_t selected_queries) {
  output << "{\"type\":\"metadata\""
         << ",\"format\":\"rips-sssp-paths-v1\""
         << ",\"producer\":\"all_SSSP\""
         << ",\"algorithm\":";
  write_json_string(output, options.algorithm);
  output << ",\"predicate_mode\":";
  write_json_string(output, predicate_name);
  output << ",\"early_stop\":"
         << (options.early_stop ? "true" : "false")
         << ",\"node_count\":" << rows
         << ",\"edge_count\":" << nnz
         << ",\"route_request_count\":"
         << metadata.route_requests.size()
         << ",\"selected_source_count\":" << selected_sources
         << ",\"selected_query_count\":" << selected_queries
         << ",\"edge_orientation\":\"outgoing\""
         << ",\"non_target_distances_complete\":"
         << (options.early_stop ? "false" : "true")
         << "}\n";
}

void write_path_record(std::ostream& output,
                       const Query& query,
                       const TargetResult& result) {
  output << "{\"type\":\"path\",\"net\":";
  write_json_string(output, query.net_name);
  output << ",\"net_index\":" << query.net_index;
  if (query.logical_net_index == kNoLogicalNetIndex) {
    output << ",\"logical_net_index\":null";
  } else {
    output << ",\"logical_net_index\":" << query.logical_net_index;
  }
  output << ",\"source\":" << query.source
         << ",\"target\":" << query.target
         << ",\"reached\":" << (result.reached ? "true" : "false");
  if (result.reached) {
    output << ",\"distance\":" << std::setprecision(9)
           << result.distance;
  } else {
    output << ",\"distance\":null";
  }
  output << ",\"nodes\":[";
  for (std::size_t index = 0; index < result.nodes.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << result.nodes[index];
  }
  output << "],\"csr_edges\":[";
  for (std::size_t index = 0; index < result.csr_edges.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << result.csr_edges[index];
  }
  output << "]}\n";
}

void print_snapshot(const char* label,
                    const Options& options,
                    std::string_view predicate_name,
                    const AggregateStatistics& aggregate,
                    std::size_t total_sources,
                    const std::optional<EngineStatistics>& engine,
                    Clock::time_point source_loop_begin,
                    Clock::time_point end_to_end_begin,
                    Clock::time_point source_loop_sample,
                    Clock::time_point end_to_end_sample) {
  const double source_loop_seconds =
      std::max(0.0, seconds(source_loop_sample - source_loop_begin));
  const double end_to_end_seconds =
      std::max(0.0, seconds(end_to_end_sample - end_to_end_begin));
  const double percent =
      total_sources == 0
          ? 100.0
          : 100.0 * static_cast<double>(aggregate.completed) /
                static_cast<double>(total_sources);
  const double sources_per_second =
      source_loop_seconds > 0.0
          ? static_cast<double>(aggregate.completed) /
                source_loop_seconds
          : 0.0;
  const double queries_per_second =
      source_loop_seconds > 0.0
          ? static_cast<double>(aggregate.queries_covered) /
                source_loop_seconds
          : 0.0;

  std::cout << "\n[all_SSSP] " << label << " statistics\n"
            << "  Selected GPU algorithm: " << options.algorithm << '\n'
            << "  Predicate mode: " << predicate_name << '\n'
            << "  Early stopping: "
            << (options.early_stop ? "enabled" : "disabled") << '\n'
            << "  Sources completed: " << aggregate.completed << '\n'
            << "  Total sources: " << total_sources << '\n'
            << "  Percentage complete: " << std::fixed
            << std::setprecision(1) << percent << "%\n"
            << "  Source-sink queries covered: "
            << aggregate.queries_covered << '\n'
            << "  Full-convergence count: "
            << aggregate.full_convergence << '\n'
            << "  Early-stop count: " << aggregate.early_stops << '\n'
            << "  Maximum-iteration count: "
            << aggregate.max_iterations << '\n'
            << "  Failure count: " << aggregate.failures << '\n'
            << "  Sources with every sink reachable: "
            << aggregate.every_sink_reachable << '\n'
            << "  Sources containing unreachable sinks: "
            << aggregate.contains_unreachable_sink << '\n'
            << "  Total phases/iterations: "
            << aggregate.total_phases << '\n';
  if (aggregate.completed == 0 ||
      !aggregate.minimum_phases.has_value()) {
    std::cout << "  Minimum phases/iterations per source: n/a\n"
              << "  Maximum phases/iterations per source: n/a\n"
              << "  Mean phases/iterations per source: n/a\n";
  } else {
    std::cout << "  Minimum phases/iterations per source: "
              << *aggregate.minimum_phases << '\n'
              << "  Maximum phases/iterations per source: "
              << aggregate.maximum_phases << '\n'
              << "  Mean phases/iterations per source: "
              << std::setprecision(3)
              << static_cast<double>(aggregate.total_phases) /
                     static_cast<double>(aggregate.completed)
              << '\n';
  }
  std::cout << "  Total vertices settled: "
            << aggregate.vertices_settled << '\n'
            << "  Total edges examined: "
            << aggregate.edges_examined << '\n'
            << "  Total relaxation attempts: "
            << aggregate.relaxation_attempts << '\n'
            << "  Total successful distance updates: "
            << aggregate.successful_distance_updates << '\n';

  if (engine.has_value()) {
    const EngineStatistics& statistics = *engine;
    std::cout << "  Total predicate-evaluation GPU time (ms): "
              << std::setprecision(3) << statistics.predicate_gpu_ms << '\n'
              << "  Total relaxation GPU time (ms): "
              << statistics.relaxation_gpu_ms << '\n'
              << "  GPU kernel-launch count: "
              << statistics.kernel_launches << '\n'
              << "  Per-source reset-kernel count: "
              << statistics.reset_kernel_launches << " total";
    if (statistics.runs != 0) {
      std::cout << " ("
                << static_cast<double>(statistics.reset_kernel_launches) /
                       static_cast<double>(statistics.runs)
                << " per source)";
    }
    std::cout << '\n'
              << "  Initial graph H2D transfer count: "
              << statistics.initial_graph_h2d_transfers << '\n'
              << "  Initial graph H2D transfer bytes: "
              << statistics.initial_graph_h2d_bytes << '\n'
              << "  Per-source target H2D transfer count: "
              << statistics.target_h2d_transfers << '\n'
              << "  Per-source target H2D transfer bytes: "
              << statistics.target_h2d_bytes << '\n'
              << "  D2H transfer count: "
              << statistics.d2h_transfers << '\n'
              << "  D2H transfer bytes: "
              << statistics.d2h_bytes << '\n'
              << "  Preprocessing time (ms): "
              << statistics.preprocessing_ms << '\n'
              << "  Initial graph-transfer time (ms): "
              << statistics.initial_graph_transfer_ms << '\n'
              << "  Cumulative GPU SSSP time (ms): "
              << statistics.gpu_sssp_ms << '\n'
              << "  Cumulative reset time (ms): "
              << statistics.reset_gpu_ms << '\n'
              << "  Cumulative target-transfer time (ms): "
              << statistics.target_transfer_ms << '\n'
              << "  Cumulative D2H transfer time (ms): "
              << statistics.d2h_transfer_ms << '\n'
              << "  Reusable device allocation high-water bytes: "
              << statistics.allocated_device_bytes << '\n';
  } else {
    std::cout
        << "  Total predicate-evaluation GPU time (ms): n/a\n"
        << "  Total relaxation GPU time (ms): n/a\n"
        << "  GPU kernel-launch count: n/a\n"
        << "  Per-source reset-kernel count: n/a\n"
        << "  Initial graph H2D transfer count: n/a\n"
        << "  Initial graph H2D transfer bytes: n/a\n"
        << "  Per-source target H2D transfer count: n/a\n"
        << "  Per-source target H2D transfer bytes: n/a\n"
        << "  D2H transfer count: n/a\n"
        << "  D2H transfer bytes: n/a\n"
        << "  Preprocessing time (ms): n/a\n"
        << "  Initial graph-transfer time (ms): n/a\n"
        << "  Cumulative GPU SSSP time (ms): n/a\n"
        << "  Cumulative reset time (ms): n/a\n"
        << "  Cumulative target-transfer time (ms): n/a\n"
        << "  Cumulative D2H transfer time (ms): n/a\n"
        << "  Reusable device allocation high-water bytes: n/a\n";
  }

  std::cout << "  Elapsed source-loop wall time (s): "
            << std::setprecision(3) << source_loop_seconds << '\n'
            << "  End-to-end elapsed time (s): "
            << end_to_end_seconds << '\n'
            << "  Average sources per second: "
            << sources_per_second << '\n'
            << "  Average source-sink queries per second: "
            << queries_per_second << '\n'
            << "  Estimated remaining time (s): ";
  if (aggregate.completed >= total_sources) {
    std::cout << "0.000\n";
  } else if (sources_per_second > 0.0) {
    std::cout
        << static_cast<double>(total_sources - aggregate.completed) /
               sources_per_second
        << '\n';
  } else {
    std::cout << "n/a\n";
  }
  if (options.early_stop) {
    std::cout
        << "  Early-stop completeness: metadata sink distances are confirmed; "
           "non-target distances may be incomplete.\n";
  }
}

void validate_run_result(const SourceWork& work,
                         const Options& options,
                         const SsspRunResult& result,
                         const GpuSsspEngine& engine) {
  if (result.termination == TerminationReason::kError) {
    throw std::runtime_error("engine reported an SSSP runtime error");
  }
  if (result.termination == TerminationReason::kMaxIterations) {
    throw std::runtime_error(
        "engine reached its maximum iteration limit before completion");
  }
  if (!result.all_targets_confirmed) {
    throw std::runtime_error(
        "engine returned without confirming all requested targets");
  }
  if (!options.early_stop && !result.fully_converged) {
    throw std::runtime_error(
        "engine did not fully converge with --no-early-stop");
  }
  if (work.unique_targets.empty() && !result.fully_converged) {
    throw std::runtime_error(
        "empty target set caused an invalid vacuous early stop");
  }
  if (result.termination == TerminationReason::kAllTargetsConfirmed &&
      result.fully_converged) {
    throw std::runtime_error(
        "engine labeled a fully converged run as an early stop");
  }
  if (result.termination == TerminationReason::kFullConvergence &&
      !result.fully_converged) {
    throw std::runtime_error(
        "engine labeled an incomplete run as full convergence");
  }
  if (result.targets.size() != work.unique_targets.size()) {
    throw std::runtime_error(
        "engine target result count does not match the unique target set");
  }

  std::unordered_set<int> expected(
      work.unique_targets.begin(), work.unique_targets.end());
  std::unordered_set<int> returned;
  returned.reserve(
      checked_hash_reserve(
          result.targets.size(), "returned target set"));
  bool any_unreachable = false;
  for (const TargetResult& target : result.targets) {
    if (expected.find(target.target) == expected.end() ||
        !returned.insert(target.target).second) {
      throw std::runtime_error(
          "engine returned an unexpected or duplicate target result");
    }
    if (target.target == work.source &&
        (!target.reached || target.distance != 0.0f)) {
      throw std::runtime_error(
          "engine did not return the required zero-distance source-to-self "
          "result");
    }
    if (options.paths_output_set && !target.path_captured) {
      throw std::runtime_error(
          "engine did not capture a requested target path");
    }
    if (target.reached) {
      if (!std::isfinite(target.distance) || target.distance < 0.0f) {
        throw std::runtime_error(
            "engine returned an invalid reached-target distance");
      }
      if (options.paths_output_set) {
        if (target.nodes.empty() ||
            target.nodes.front() != work.source ||
            target.nodes.back() != target.target ||
            target.csr_edges.size() + 1 != target.nodes.size()) {
          throw std::runtime_error(
              "engine returned an invalid compact target path");
        }
        if (target.target == work.source &&
            (target.nodes.size() != 1 ||
             target.nodes.front() != work.source ||
             !target.csr_edges.empty())) {
          throw std::runtime_error(
              "engine returned a nontrivial source-to-self path");
        }
      }
    } else {
      any_unreachable = true;
      if (target.distance != std::numeric_limits<float>::infinity()) {
        throw std::runtime_error(
            "engine returned a distance other than positive infinity for an "
            "unreachable target");
      }
      if (options.paths_output_set &&
          (!target.nodes.empty() || !target.csr_edges.empty())) {
        throw std::runtime_error(
            "engine returned path data for an unreachable target");
      }
    }
  }
  if (any_unreachable &&
      result.termination == TerminationReason::kAllTargetsConfirmed) {
    throw std::runtime_error(
        "an unreachable target incorrectly triggered target-only stopping");
  }
  if (options.paths_output_set) {
    engine.validate_captured_target_paths(work.source, result.targets);
  }
}

#ifndef ALL_SSSP_HOST_TEST

int run(const Options& options, Clock::time_point end_to_end_begin) {
  const routing::interchange::InterchangePublicationSnapshot
      publication_snapshot =
          routing::interchange::snapshot_interchange_publication(
              options.csr_path, options.metadata_path);

  std::cout << "[all_SSSP] loading outgoing CSR: "
            << options.csr_path.string() << '\n';
  HostGraph graph = load_graph(options.csr_path);
  std::cout << "[all_SSSP] loading RIPSIFM1 metadata: "
            << options.metadata_path.string() << '\n';
  RoutingMetadata metadata =
      load_metadata(options.metadata_path, graph.nnz);

  routing::interchange::verify_interchange_publication(
      options.csr_path,
      options.metadata_path,
      publication_snapshot);
  routing::interchange::require_matching_interchange_pair_ids(
      graph.artifact_pair_id,
      metadata.artifact_pair_id,
      publication_snapshot.generation);
  if (metadata.metadata_node_count !=
      static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error(
        "metadata node count does not match outgoing CSR rows");
  }

  WorkBuildResult work =
      build_source_work(metadata, graph.rows, std::cerr);
  const std::size_t selected_sources =
      selected_source_count(work.work.size(), options.source_limit);
  const std::uint64_t selected_queries =
      count_selected_queries(work.work, selected_sources);
  const std::size_t maximum_targets =
      max_unique_targets_per_source(
          work.work, selected_sources, graph.rows);

  std::uint64_t selected_unique_source_targets = 0;
  std::unordered_set<int> selected_global_targets;
  for (std::size_t index = 0; index < selected_sources; ++index) {
    selected_unique_source_targets = checked_add(
        selected_unique_source_targets,
        static_cast<std::uint64_t>(
            work.work[index].unique_targets.size()),
        "selected unique source-target count");
    selected_global_targets.insert(
        work.work[index].unique_targets.begin(),
        work.work[index].unique_targets.end());
  }

  std::cout
      << "[all_SSSP] graph rows=" << graph.rows
      << " nnz=" << graph.nnz
      << " orientation=outgoing\n"
      << "[all_SSSP] metadata_version=" << metadata.version
      << " route_requests=" << metadata.route_requests.size()
      << " raw_metadata_source_occurrences="
      << work.raw_source_occurrences
      << " unique_valid_source_nodes="
      << work.unique_valid_sources
      << " invalid_source_count="
      << work.invalid_source_occurrences
      << " raw_source_sink_query_count="
      << work.raw_source_sink_queries
      << " valid_source_sink_query_count="
      << work.valid_source_sink_queries
      << " invalid_sink_count="
      << work.invalid_sink_occurrences
      << " invalid_source_queries="
      << work.invalid_source_queries
      << " invalid_sink_queries="
      << work.invalid_sink_queries
      << " unique_target_count="
      << work.unique_target_nodes
      << " unique_source_target_pairs="
      << work.unique_source_target_pairs
      << " sources_selected_after_limit="
      << selected_sources
      << " selected_queries=" << selected_queries
      << " selected_unique_target_nodes="
      << selected_global_targets.size()
      << " selected_unique_source_target_pairs="
      << selected_unique_source_targets
      << " max_unique_targets_per_source="
      << maximum_targets << '\n';

  const Offset graph_rows = graph.rows;
  const Offset graph_nnz = graph.nnz;
  const std::string predicate_name =
      configured_predicate_name(options);
  std::ofstream paths_output;
  if (options.paths_output_set) {
    paths_output = open_paths_output(options.paths_output);
    write_paths_metadata(
        paths_output,
        graph_rows,
        graph_nnz,
        metadata,
        options,
        predicate_name,
        selected_sources,
        selected_queries);
  } else {
    std::cout
        << "[all_SSSP] path output disabled; use --paths-output <path> "
           "to emit metadata query paths\n";
  }

  AggregateStatistics aggregate;
  const Clock::time_point empty_loop_begin = Clock::now();
  if (selected_sources == 0) {
    ProgressReporter progress(
        0, options.stats_every, empty_loop_begin);
    progress.update(0, true);
    if (paths_output) {
      paths_output.flush();
      if (!paths_output) {
        throw std::runtime_error("failed while writing paths JSONL");
      }
    }
    const Clock::time_point final_sample = Clock::now();
    print_snapshot(
        "final",
        options,
        predicate_name,
        aggregate,
        0,
        std::nullopt,
        empty_loop_begin,
        end_to_end_begin,
        final_sample,
        final_sample);
    return 0;
  }

  std::cout
      << "[all_SSSP] preparing reusable " << options.algorithm
      << " graph and workspace once\n";
  std::unique_ptr<GpuSsspEngine> engine =
      create_engine(options, std::move(graph), maximum_targets);
  if (options.algorithm != engine->algorithm_name() ||
      predicate_name != engine->predicate_name()) {
    throw std::logic_error(
        "engine registry and adapter reporting metadata disagree");
  }
  const Clock::time_point source_loop_begin = Clock::now();
  ProgressReporter progress(
      selected_sources, options.stats_every, end_to_end_begin);
  progress.update(0, true);

  using TargetMap = std::unordered_map<int, TargetResult>;
  std::vector<TargetMap> stored_results;
  if (options.paths_output_set) {
    stored_results.resize(selected_sources);
  }

  Clock::time_point source_loop_end = source_loop_begin;
  for (std::size_t source_index = 0;
       source_index < selected_sources;
       ++source_index) {
    const SourceWork& source_work = work.work[source_index];
    SsspRequest request;
    request.source = source_work.source;
    request.unique_targets = source_work.unique_targets;
    request.early_stop = options.early_stop;
    request.capture_paths = options.paths_output_set;

    SsspRunResult result;
    try {
      result = engine->run(request);
      validate_run_result(source_work, options, result, *engine);
    } catch (const std::exception& error) {
      aggregate.failures =
          checked_add(aggregate.failures, 1, "failure count");
      progress.finish_line();
      const Clock::time_point failure_sample = Clock::now();
      print_snapshot(
          "failure",
          options,
          engine->predicate_name(),
          aggregate,
          selected_sources,
          engine->cumulative_statistics(),
          source_loop_begin,
          end_to_end_begin,
          failure_sample,
          failure_sample);
      std::ostringstream message;
      message << "source run failed at unique source index "
              << source_index << " (CSR node " << source_work.source
              << "), algorithm=" << options.algorithm
              << ", predicate=" << engine->predicate_name()
              << ": " << error.what();
      throw std::runtime_error(message.str());
    }

    if (options.paths_output_set) {
      TargetMap& target_map = stored_results[source_index];
      target_map.reserve(
          checked_hash_reserve(
              result.targets.size(), "stored target result map"));
      for (TargetResult& target : result.targets) {
        const int target_node = target.target;
        const bool inserted =
            target_map.emplace(target_node, std::move(target)).second;
        if (!inserted) {
          throw std::logic_error(
              "duplicate target escaped run-result validation");
        }
      }
    }
    aggregate.add_success(source_work, result);

    const std::size_t completed = source_index + 1;
    const bool final_source = completed == selected_sources;
    if (!final_source) {
      progress.update(completed);
    }
    const bool intermediate_snapshot =
        !final_source && options.stats_every != 0 &&
        completed % options.stats_every == 0;
    if (intermediate_snapshot) {
      progress.before_block();
      const Clock::time_point snapshot_sample = Clock::now();
      print_snapshot(
          "cumulative",
          options,
          engine->predicate_name(),
          aggregate,
          selected_sources,
          engine->cumulative_statistics(),
          source_loop_begin,
          end_to_end_begin,
          snapshot_sample,
          snapshot_sample);
      progress.after_block(completed);
    }
  }
  source_loop_end = Clock::now();

  if (options.paths_output_set) {
    const std::vector<const Query*> selected_query_order =
        selected_queries_in_metadata_order(
            work.work, selected_sources);
    if (selected_query_order.size() !=
        checked_host_size(selected_queries, "selected path query count")) {
      throw std::logic_error(
          "selected path query ordering lost a query");
    }

    std::unordered_map<int, std::size_t> source_index_by_node;
    source_index_by_node.reserve(
        checked_hash_reserve(
            selected_sources, "selected source index map"));
    for (std::size_t source_index = 0;
         source_index < selected_sources;
         ++source_index) {
      source_index_by_node.emplace(
          work.work[source_index].source, source_index);
    }
    for (const Query* query : selected_query_order) {
      const auto source_found =
          source_index_by_node.find(query->source);
      if (source_found == source_index_by_node.end()) {
        throw std::logic_error(
            "selected path query lost its source result");
      }
      const TargetMap& target_map =
          stored_results[source_found->second];
      const auto target_found = target_map.find(query->target);
      if (target_found == target_map.end()) {
        throw std::logic_error(
            "selected path query lost its target result");
      }
      write_path_record(paths_output, *query, target_found->second);
    }
    paths_output.flush();
    if (!paths_output) {
      throw std::runtime_error("failed while writing paths JSONL");
    }
    std::cout << "[all_SSSP] wrote " << selected_query_order.size()
              << " path record(s) to "
              << options.paths_output.string() << '\n';
  }

  progress.update(selected_sources, true);
  const Clock::time_point final_sample = Clock::now();
  print_snapshot(
      "final",
      options,
      engine->predicate_name(),
      aggregate,
      selected_sources,
      engine->cumulative_statistics(),
      source_loop_begin,
      end_to_end_begin,
      source_loop_end,
      final_sample);
  return 0;
}

#endif  // ALL_SSSP_HOST_TEST

}  // namespace all_sssp

#if !defined(ALL_SSSP_NO_MAIN) && !defined(ALL_SSSP_HOST_TEST)
int main(int argc, char** argv) {
  const auto end_to_end_begin = all_sssp::Clock::now();
  try {
    const all_sssp::Options options = all_sssp::parse_args(argc, argv);
    if (options.help) {
      all_sssp::print_usage(std::cout, argv[0]);
      return 0;
    }
    return all_sssp::run(options, end_to_end_begin);
  } catch (const std::exception& error) {
    std::cerr << "all_SSSP error: " << error.what() << '\n';
    return 1;
  }
}
#endif
