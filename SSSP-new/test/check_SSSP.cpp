// SSSP-new outgoing path validator.
//
// Build from rips2026-amd:
//   g++ -std=c++17 -O2 SSSP-new/test/check_SSSP.cpp -o check_SSSP
//
// Run all checks:
//   ./check_SSSP SSSP-new/data/example.outgoing.csrbin \
//     SSSP-new/data/example.outgoing.csrbin.ifmeta.bin \
//     SSSP-new/paths/example.paths.jsonl --all

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Offset = std::int64_t;
using Index = int;

constexpr char OUTGOING_CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'O', 'C', 'S', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t MIN_OUTGOING_CSR_VERSION = 3;
constexpr std::uint64_t CURRENT_OUTGOING_CSR_VERSION = 3;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 3;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoLogicalNetIndex = kNoIndex;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr std::size_t kMaxPrintedFailures = 80;

struct HostOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> degree;
  std::vector<Index> to;
  std::vector<float> values;
  std::vector<Offset> edge_id;
  std::vector<Index> from_for_edge;
  bool all_weights_one = false;
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
  std::uint64_t metadata_node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::vector<std::string> strings;
  std::vector<RouteRequest> route_requests;
};

struct QueryKey {
  std::uint64_t net_index = 0;
  std::uint64_t logical_net_index = kNoLogicalNetIndex;
  int source = -1;
  int target = -1;

  bool operator==(const QueryKey& other) const {
    return net_index == other.net_index &&
           logical_net_index == other.logical_net_index &&
           source == other.source &&
           target == other.target;
  }
};

struct QueryKeyHash {
  std::size_t operator()(const QueryKey& key) const {
    std::size_t h = std::hash<std::uint64_t>{}(key.net_index);
    h ^= std::hash<std::uint64_t>{}(key.logical_net_index) + 0x9e3779b97f4a7c15ull +
         (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.source) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.target) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
  }
};

struct Query {
  QueryKey key;
  std::string net_name;
};

struct ExpectedInfo {
  std::string net_name;
  std::uint64_t count = 0;
};

struct ExpectedQueries {
  std::uint64_t raw_source_count = 0;
  std::uint64_t invalid_source_count = 0;
  std::uint64_t invalid_sink_count = 0;
  std::vector<Query> queries;
  std::unordered_map<QueryKey, ExpectedInfo, QueryKeyHash> by_key;
};

struct PathRecord {
  std::uint64_t line_no = 0;
  QueryKey key;
  std::string net;
  bool reached = true;
  bool has_distance = false;
  double distance = 0.0;
  std::vector<int> nodes;
  std::vector<Offset> csr_edges;
};

struct JsonlData {
  std::uint64_t jsonl_lines = 0;
  std::uint64_t metadata_records = 0;
  std::vector<PathRecord> paths;
  std::unordered_map<QueryKey, std::vector<std::size_t>, QueryKeyHash> by_key;
};

struct Options {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path jsonl_path;
  bool check_continuity = false;
  bool check_shortest = false;
  bool check_completeness = false;
  std::size_t shortest_source_limit = 0;
  double abs_tolerance = 1e-4;
};

struct CheckStats {
  std::uint64_t checked = 0;
  std::uint64_t failures = 0;
  std::uint64_t sources_checked = 0;
  std::uint64_t sources_skipped = 0;
};

struct FailureSink {
  std::vector<std::string> first_failures;

  void add(const std::string& message) {
    if (first_failures.size() < kMaxPrintedFailures) {
      first_failures.push_back(message);
    }
  }
};

std::string key_string(const QueryKey& key) {
  std::ostringstream out;
  out << "net_index=" << key.net_index
      << " logical_net_index=";
  if (key.logical_net_index == kNoLogicalNetIndex) {
    out << "null";
  } else {
    out << key.logical_net_index;
  }
  out << " source=" << key.source
      << " target=" << key.target;
  return out.str();
}

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) return;

  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t bytes_per_item,
                                 const char* name) {
  if (bytes_per_item != 0 &&
      count > std::numeric_limits<std::uint64_t>::max() / bytes_per_item) {
    throw std::runtime_error(std::string(name) + " byte count overflow");
  }
  return count * bytes_per_item;
}

void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) return;
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) + " byte count is too large to seek");
  }
  in.seekg(static_cast<std::streamoff>(count), std::ios::cur);
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

std::string read_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large for this host");
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed while reading metadata string bytes");
    }
  }
  return text;
}

std::size_t checked_size(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  return static_cast<std::size_t>(count);
}

int node_from_u64(std::uint64_t node) {
  if (node > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(node);
}

void validate_outgoing_csr(HostOutgoingCsrF32* graph) {
  if (graph->rows <= 0 || graph->rows != graph->cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph->rows > static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for int node ids");
  }
  if (graph->nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph->rowptr.size() != static_cast<std::size_t>(graph->rows + 1) ||
      graph->degree.size() != static_cast<std::size_t>(graph->rows) ||
      graph->to.size() != static_cast<std::size_t>(graph->nnz) ||
      graph->values.size() != static_cast<std::size_t>(graph->nnz) ||
      graph->edge_id.size() != static_cast<std::size_t>(graph->nnz)) {
    throw std::runtime_error("outgoing CSR array sizes do not match header counts");
  }
  if (graph->rowptr.front() != 0 || graph->rowptr.back() != graph->nnz) {
    throw std::runtime_error("outgoing CSR rowptr must start at 0 and end at nnz");
  }

  graph->from_for_edge.assign(static_cast<std::size_t>(graph->nnz), -1);
  graph->all_weights_one = true;
  for (Offset row = 0; row < graph->rows; ++row) {
    const Offset begin = graph->rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph->rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph->nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const Offset row_degree = end - begin;
    if (row_degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    if (graph->degree[static_cast<std::size_t>(row)] != static_cast<Index>(row_degree)) {
      throw std::runtime_error("outgoing CSR degree array does not match rowptr");
    }
    for (Offset edge = begin; edge < end; ++edge) {
      graph->from_for_edge[static_cast<std::size_t>(edge)] = static_cast<Index>(row);
    }
  }

  for (Offset edge = 0; edge < graph->nnz; ++edge) {
    const std::size_t i = static_cast<std::size_t>(edge);
    if (graph->to[i] < 0 || static_cast<Offset>(graph->to[i]) >= graph->cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (graph->edge_id[i] != edge) {
      throw std::runtime_error("check_SSSP expects outgoing CSR edge_id[k] == k");
    }
    if (!std::isfinite(graph->values[i]) || graph->values[i] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
    if (graph->values[i] != 1.0f) {
      graph->all_weights_one = false;
    }
  }
}

HostOutgoingCsrF32 load_outgoing_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open outgoing CSR file: " + path.string());
  }

  char magic[sizeof(OUTGOING_CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, OUTGOING_CSR_MAGIC, sizeof(OUTGOING_CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPSOCS1 outgoing CSR file");
  }

  const std::uint64_t version = read_u64(in, "outgoing CSR format version");
  const std::uint64_t orientation = read_u64(in, "outgoing CSR orientation");
  if (version < MIN_OUTGOING_CSR_VERSION ||
      version > CURRENT_OUTGOING_CSR_VERSION) {
    throw std::runtime_error("unsupported outgoing CSR format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported outgoing CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "outgoing CSR row count");
  const std::uint64_t cols = read_u64(in, "outgoing CSR column count");
  const std::uint64_t nnz = read_u64(in, "outgoing CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "outgoing CSR rowptr count");
  const std::uint64_t degree_count = read_u64(in, "outgoing CSR degree count");
  const std::uint64_t to_count = read_u64(in, "outgoing CSR destination count");
  const std::uint64_t values_count = read_u64(in, "outgoing CSR values count");
  const std::uint64_t edge_id_count = read_u64(in, "outgoing CSR edge-id count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this validator");
  }
  if (rowptr_count != rows + 1 || degree_count != rows ||
      to_count != nnz || values_count != nnz || edge_id_count != nnz) {
    throw std::runtime_error("outgoing CSR header counts are inconsistent");
  }

  HostOutgoingCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  read_array(in, graph.degree, degree_count, "outgoing CSR degree");
  read_array(in, graph.to, to_count, "outgoing CSR destinations");
  read_array(in, graph.values, values_count, "outgoing CSR values");
  read_array(in, graph.edge_id, edge_id_count, "outgoing CSR edge-id map");
  validate_outgoing_csr(&graph);
  return graph;
}

RoutingMetadata load_routing_metadata(const std::filesystem::path& path,
                                      Offset graph_nnz) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPSIFM1 metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
  if (edge_attr_count != static_cast<std::uint64_t>(graph_nnz)) {
    throw std::runtime_error("metadata edge attribute count does not match outgoing CSR nnz");
  }
  const std::uint64_t pip_data_count = read_u64(in, "metadata pip data count");
  const std::uint64_t site_pin_attr_count = read_u64(in, "metadata site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "metadata route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "metadata sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "metadata logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "metadata logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "metadata physical byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "metadata logical byte count");

  (void)read_u64(in, "metadata device path string");
  (void)read_u64(in, "metadata physical path string");
  (void)read_u64(in, "metadata logical path string");
  (void)read_u64(in, "metadata logical design name");

  RoutingMetadata metadata;
  metadata.metadata_node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.strings.reserve(checked_size(string_count, "metadata string"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  skip_bytes(in,
             checked_byte_count(node_count, sizeof(std::uint64_t), "metadata device node ids"),
             "metadata device node ids");
  skip_bytes(in,
             checked_byte_count(edge_attr_count, 2 * sizeof(std::uint64_t), "metadata edge attrs"),
             "metadata edge attrs");
  skip_bytes(in,
             checked_byte_count(pip_data_count, 3 * sizeof(std::uint64_t), "metadata pip data"),
             "metadata pip data");
  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t),
                                "metadata site pin attrs"),
             "metadata site pin attrs");

  metadata.route_requests.resize(checked_size(route_request_count, "metadata route request"));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");
    const std::uint64_t source_count = read_u64(in, "metadata source count");
    request.sources.resize(checked_size(source_count, "metadata source"));
    for (SitePinNode& source : request.sources) {
      source.node = node_from_u64(read_u64(in, "metadata source node"));
      source.site_string = read_u64(in, "metadata source site");
      source.pin_string = read_u64(in, "metadata source pin");
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    request.sinks.resize(checked_size(sink_count, "metadata sink"));
    for (SitePinNode& sink : request.sinks) {
      sink.node = node_from_u64(read_u64(in, "metadata sink node"));
      sink.site_string = read_u64(in, "metadata sink site");
      sink.pin_string = read_u64(in, "metadata sink pin");
    }
  }

  skip_bytes(in,
             checked_byte_count(logical_cell_count, 3 * sizeof(std::uint64_t),
                                "metadata logical cells"),
             "metadata logical cells");
  skip_bytes(in,
             checked_byte_count(logical_net_count, 4 * sizeof(std::uint64_t),
                                "metadata logical nets"),
             "metadata logical nets");
  skip_bytes(in,
             checked_byte_count(logical_port_instance_count, 7 * sizeof(std::uint64_t),
                                "metadata logical port instances"),
             "metadata logical port instances");
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t),
                                "metadata blocked nodes"),
             "metadata blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count, sizeof(std::uint64_t),
                                "metadata sink stop nodes"),
             "metadata sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "metadata physical bytes");
  skip_bytes(in, logical_netlist_byte_count, "metadata logical bytes");
  return metadata;
}

bool valid_node(const HostOutgoingCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) return "";
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

ExpectedQueries build_expected_queries(const RoutingMetadata& metadata,
                                       const HostOutgoingCsrF32& graph) {
  ExpectedQueries expected;
  expected.queries.reserve(metadata.route_requests.size());
  for (std::size_t net_index = 0; net_index < metadata.route_requests.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const std::string net_name = string_at(metadata, request.net_string);
    for (const SitePinNode& source_pin : request.sources) {
      ++expected.raw_source_count;
      if (!valid_node(graph, source_pin.node)) {
        ++expected.invalid_source_count;
        continue;
      }
      for (const SitePinNode& sink_pin : request.sinks) {
        if (!valid_node(graph, sink_pin.node)) {
          ++expected.invalid_sink_count;
          continue;
        }
        Query query;
        query.key.net_index = static_cast<std::uint64_t>(net_index);
        query.key.logical_net_index = request.logical_net_index;
        query.key.source = source_pin.node;
        query.key.target = sink_pin.node;
        query.net_name = net_name;
        expected.queries.push_back(query);

        ExpectedInfo& info = expected.by_key[query.key];
        if (info.count == 0) {
          info.net_name = net_name;
        }
        ++info.count;
      }
    }
  }
  return expected;
}

struct JsonValue {
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_number() const { return std::holds_alternative<double>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_array() const { return std::holds_alternative<Array>(value); }
  bool is_object() const { return std::holds_alternative<Object>(value); }

  bool as_bool(const char* name) const {
    if (const auto* v = std::get_if<bool>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not bool: ") + name);
  }

  double as_number(const char* name) const {
    if (const auto* v = std::get_if<double>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not number: ") + name);
  }

  const std::string& as_string(const char* name) const {
    if (const auto* v = std::get_if<std::string>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not string: ") + name);
  }

  const Array& as_array(const char* name) const {
    if (const auto* v = std::get_if<Array>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not array: ") + name);
  }

  const Object& as_object(const char* name) const {
    if (const auto* v = std::get_if<Object>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not object: ") + name);
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters after JSON value");
    }
    return value;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    return text_[pos_];
  }

  bool consume(char expected) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    const char ch = peek();
    if (ch == '{') return JsonValue{parse_object()};
    if (ch == '[') return JsonValue{parse_array()};
    if (ch == '"') return JsonValue{parse_string()};
    if (ch == 't') {
      expect_literal("true");
      return JsonValue{true};
    }
    if (ch == 'f') {
      expect_literal("false");
      return JsonValue{false};
    }
    if (ch == 'n') {
      expect_literal("null");
      return JsonValue{nullptr};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return JsonValue{parse_number()};
    }
    throw std::runtime_error("unexpected JSON value");
  }

  JsonValue::Object parse_object() {
    if (!consume('{')) throw std::runtime_error("expected JSON object");
    JsonValue::Object object;
    if (consume('}')) return object;
    while (true) {
      std::string key = parse_string();
      if (!consume(':')) throw std::runtime_error("expected ':' in JSON object");
      object.emplace(std::move(key), parse_value());
      if (consume('}')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON object");
    }
    return object;
  }

  JsonValue::Array parse_array() {
    if (!consume('[')) throw std::runtime_error("expected JSON array");
    JsonValue::Array array;
    if (consume(']')) return array;
    while (true) {
      array.push_back(parse_value());
      if (consume(']')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON array");
    }
    return array;
  }

  std::string parse_string() {
    if (!consume('"')) throw std::runtime_error("expected JSON string");
    std::string out;
    while (pos_ < text_.size()) {
      const char ch = text_[pos_++];
      if (ch == '"') return out;
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
          throw std::runtime_error("unicode JSON escapes are not supported");
        default:
          throw std::runtime_error("unknown JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  double parse_number() {
    const std::size_t begin = pos_;
    if (text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }

    const std::string token = text_.substr(begin, pos_ - begin);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
      throw std::runtime_error("invalid JSON number");
    }
    return value;
  }

  void expect_literal(std::string_view literal) {
    if (text_.compare(pos_, literal.size(), literal) != 0) {
      throw std::runtime_error("invalid JSON literal");
    }
    pos_ += literal.size();
  }

  std::string text_;
  std::size_t pos_ = 0;
};

const JsonValue* find_field(const JsonValue::Object& object, const char* key) {
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

std::int64_t require_i64(const JsonValue::Object& object, const char* key) {
  const JsonValue* value = find_field(object, key);
  if (value == nullptr) {
    throw std::runtime_error(std::string("missing required JSON field: ") + key);
  }
  const double number = value->as_number(key);
  if (!std::isfinite(number) ||
      number < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      number > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      std::floor(number) != number) {
    throw std::runtime_error(std::string("JSON integer field is invalid: ") + key);
  }
  return static_cast<std::int64_t>(number);
}

std::uint64_t require_u64(const JsonValue::Object& object, const char* key) {
  const std::int64_t value = require_i64(object, key);
  if (value < 0) {
    throw std::runtime_error(std::string("JSON u64 field is negative: ") + key);
  }
  return static_cast<std::uint64_t>(value);
}

int require_node_id(const JsonValue::Object& object, const char* key) {
  const std::int64_t value = require_i64(object, key);
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("node id is outside int range: ") + key);
  }
  return static_cast<int>(value);
}

std::uint64_t optional_nullable_u64(const JsonValue::Object& object,
                                    const char* key,
                                    std::uint64_t fallback) {
  const JsonValue* value = find_field(object, key);
  if (value == nullptr || value->is_null()) return fallback;
  const double number = value->as_number(key);
  if (!std::isfinite(number) || number < 0.0 ||
      number > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
      std::floor(number) != number) {
    throw std::runtime_error(std::string("JSON u64 field is invalid: ") + key);
  }
  return static_cast<std::uint64_t>(number);
}

bool optional_bool(const JsonValue::Object& object, const char* key, bool fallback) {
  const JsonValue* value = find_field(object, key);
  return value == nullptr || value->is_null() ? fallback : value->as_bool(key);
}

std::string optional_string(const JsonValue::Object& object,
                            const char* key,
                            std::string fallback) {
  const JsonValue* value = find_field(object, key);
  return value == nullptr || value->is_null() ? std::move(fallback) : value->as_string(key);
}

std::vector<int> parse_node_array(const JsonValue::Object& object, const char* key) {
  std::vector<int> out;
  const JsonValue* value = find_field(object, key);
  if (value == nullptr || value->is_null()) return out;

  const auto& array = value->as_array(key);
  out.reserve(array.size());
  for (const JsonValue& item : array) {
    const double number = item.as_number(key);
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<int>::min()) ||
        number > static_cast<double>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(std::string("invalid node id in array: ") + key);
    }
    out.push_back(static_cast<int>(number));
  }
  return out;
}

std::vector<Offset> parse_offset_array(const JsonValue::Object& object,
                                       const char* key) {
  std::vector<Offset> out;
  const JsonValue* value = find_field(object, key);
  if (value == nullptr || value->is_null()) return out;

  const auto& array = value->as_array(key);
  out.reserve(array.size());
  for (const JsonValue& item : array) {
    const double number = item.as_number(key);
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<Offset>::min()) ||
        number > static_cast<double>(std::numeric_limits<Offset>::max())) {
      throw std::runtime_error(std::string("invalid offset in array: ") + key);
    }
    out.push_back(static_cast<Offset>(number));
  }
  return out;
}

PathRecord parse_path_record(const JsonValue::Object& object, std::uint64_t line_no) {
  PathRecord record;
  record.line_no = line_no;
  record.net = optional_string(object, "net", "");
  record.key.net_index = require_u64(object, "net_index");
  record.key.logical_net_index =
      optional_nullable_u64(object, "logical_net_index", kNoLogicalNetIndex);
  record.key.source = require_node_id(object, "source");
  record.key.target = require_node_id(object, "target");
  record.reached = optional_bool(object, "reached", true);

  if (const JsonValue* distance = find_field(object, "distance")) {
    if (!distance->is_null()) {
      record.has_distance = true;
      record.distance = distance->as_number("distance");
    }
  }

  record.nodes = parse_node_array(object, "nodes");
  record.csr_edges = parse_offset_array(object, "csr_edges");
  return record;
}

std::string parse_record_type(const JsonValue::Object& object) {
  return optional_string(object, "type", "");
}

void validate_metadata_record(const JsonValue::Object& object, std::uint64_t line_no) {
  const std::string format = optional_string(object, "format", "");
  if (format != "rips-sssp-paths-v1") {
    throw std::runtime_error("line " + std::to_string(line_no) +
                             ": metadata record format is not rips-sssp-paths-v1");
  }
  const std::string edge_orientation = optional_string(object, "edge_orientation", "");
  if (!edge_orientation.empty() && edge_orientation != "outgoing") {
    throw std::runtime_error("line " + std::to_string(line_no) +
                             ": metadata edge_orientation is not outgoing");
  }
}

JsonlData load_jsonl(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("could not open JSONL file: " + path.string());
  }

  JsonlData data;
  std::string line;
  bool saw_first_nonempty = false;
  while (std::getline(in, line)) {
    ++data.jsonl_lines;
    if (line.empty()) continue;

    JsonValue root = JsonParser(line).parse();
    const auto& object = root.as_object("JSONL record");
    const std::string type = parse_record_type(object);
    if (!saw_first_nonempty) {
      saw_first_nonempty = true;
      if (type != "metadata") {
        throw std::runtime_error("first nonempty JSONL line must be a metadata record");
      }
    }

    if (type == "metadata") {
      ++data.metadata_records;
      validate_metadata_record(object, data.jsonl_lines);
    } else if (type == "path") {
      PathRecord record = parse_path_record(object, data.jsonl_lines);
      const std::size_t index = data.paths.size();
      data.by_key[record.key].push_back(index);
      data.paths.push_back(std::move(record));
    } else {
      throw std::runtime_error("line " + std::to_string(data.jsonl_lines) +
                               ": unknown JSONL record type '" + type + "'");
    }
  }

  if (data.metadata_records == 0) {
    throw std::runtime_error("JSONL file has no metadata record");
  }
  return data;
}

bool nearly_equal(double a, double b, double abs_tolerance) {
  return std::fabs(a - b) <= abs_tolerance;
}

void add_failure(CheckStats* stats, FailureSink* sink, const std::string& message) {
  ++stats->failures;
  sink->add(message);
}

CheckStats check_continuity(const HostOutgoingCsrF32& graph,
                            const JsonlData& jsonl,
                            double abs_tolerance,
                            FailureSink* failures) {
  CheckStats stats;
  for (const PathRecord& record : jsonl.paths) {
    ++stats.checked;
    const std::string prefix =
        "line " + std::to_string(record.line_no) + " " + key_string(record.key) + ": ";

    if (!valid_node(graph, record.key.source) || !valid_node(graph, record.key.target)) {
      add_failure(&stats, failures, prefix + "source or target is outside CSR rows");
      continue;
    }

    if (!record.reached) {
      if (record.has_distance) {
        add_failure(&stats, failures, prefix + "unreached path has non-null distance");
      }
      if (!record.nodes.empty()) {
        add_failure(&stats, failures, prefix + "unreached path has nonempty nodes");
      }
      if (!record.csr_edges.empty()) {
        add_failure(&stats, failures, prefix + "unreached path has nonempty csr_edges");
      }
      continue;
    }

    if (!record.has_distance || !std::isfinite(record.distance)) {
      add_failure(&stats, failures, prefix + "reached path lacks finite distance");
      continue;
    }
    if (record.nodes.empty()) {
      add_failure(&stats, failures, prefix + "reached path has empty nodes");
      continue;
    }
    if (record.nodes.front() != record.key.source) {
      add_failure(&stats, failures, prefix + "nodes.front() does not match source");
    }
    if (record.nodes.back() != record.key.target) {
      add_failure(&stats, failures, prefix + "nodes.back() does not match target");
    }
    if (record.csr_edges.size() + 1 != record.nodes.size()) {
      add_failure(&stats, failures, prefix + "csr_edges.size() is not nodes.size() - 1");
      continue;
    }

    double path_cost = 0.0;
    for (std::size_t i = 0; i < record.csr_edges.size(); ++i) {
      const Offset edge = record.csr_edges[i];
      if (edge < 0 || edge >= graph.nnz) {
        add_failure(&stats, failures,
                    prefix + "csr_edges[" + std::to_string(i) + "] is out of range: " +
                    std::to_string(edge));
        continue;
      }
      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const int expected_from = record.nodes[i];
      const int expected_to = record.nodes[i + 1];
      if (graph.from_for_edge[edge_index] != expected_from) {
        add_failure(&stats, failures,
                    prefix + "edge " + std::to_string(edge) + " has from=" +
                    std::to_string(graph.from_for_edge[edge_index]) +
                    " but nodes step expects " + std::to_string(expected_from));
      }
      if (graph.to[edge_index] != expected_to) {
        add_failure(&stats, failures,
                    prefix + "edge " + std::to_string(edge) + " has to=" +
                    std::to_string(graph.to[edge_index]) +
                    " but nodes step expects " + std::to_string(expected_to));
      }
      path_cost += static_cast<double>(graph.values[edge_index]);
    }

    if (!nearly_equal(path_cost, record.distance, abs_tolerance)) {
      std::ostringstream message;
      message << prefix << "distance mismatch: JSON distance=" << record.distance
              << " edge-sum=" << path_cost;
      add_failure(&stats, failures, message.str());
    }
  }
  return stats;
}

std::unordered_map<int, double> dijkstra_to_targets(const HostOutgoingCsrF32& graph,
                                                    int source,
                                                    const std::unordered_set<int>& targets) {
  if (!valid_node(graph, source)) {
    return {};
  }

  std::vector<double> dist(static_cast<std::size_t>(graph.rows), kInf);
  std::unordered_set<int> remaining = targets;
  using Item = std::pair<double, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;

  dist[static_cast<std::size_t>(source)] = 0.0;
  heap.emplace(0.0, source);
  while (!heap.empty() && !remaining.empty()) {
    const auto [current_dist, node] = heap.top();
    heap.pop();
    if (current_dist != dist[static_cast<std::size_t>(node)]) {
      continue;
    }
    remaining.erase(node);
    const Offset begin = graph.rowptr[static_cast<std::size_t>(node)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(node + 1)];
    for (Offset edge = begin; edge < end; ++edge) {
      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const int dst = graph.to[edge_index];
      const double candidate = current_dist + static_cast<double>(graph.values[edge_index]);
      double& old = dist[static_cast<std::size_t>(dst)];
      if (candidate < old) {
        old = candidate;
        heap.emplace(candidate, dst);
      }
    }
  }

  std::unordered_map<int, double> out;
  out.reserve(targets.size());
  for (int target : targets) {
    out.emplace(target, valid_node(graph, target) ? dist[static_cast<std::size_t>(target)] : kInf);
  }
  return out;
}

CheckStats check_shortest(const HostOutgoingCsrF32& graph,
                          const JsonlData& jsonl,
                          double abs_tolerance,
                          std::size_t source_limit,
                          FailureSink* failures) {
  CheckStats stats;
  std::unordered_map<int, std::vector<std::size_t>> by_source;
  std::unordered_set<int> seen_sources;
  std::vector<int> source_order;
  for (std::size_t i = 0; i < jsonl.paths.size(); ++i) {
    const int source = jsonl.paths[i].key.source;
    if (seen_sources.insert(source).second) {
      source_order.push_back(source);
    }
    by_source[source].push_back(i);
  }

  for (int source : source_order) {
    if (source_limit != 0 &&
        stats.sources_checked >= static_cast<std::uint64_t>(source_limit)) {
      stats.sources_skipped =
          static_cast<std::uint64_t>(source_order.size()) - stats.sources_checked;
      break;
    }

    ++stats.sources_checked;
    const std::vector<std::size_t>& record_indices = by_source.at(source);
    std::unordered_set<int> targets;
    targets.reserve(record_indices.size());
    for (std::size_t index : record_indices) {
      targets.insert(jsonl.paths[index].key.target);
    }

    const std::unordered_map<int, double> shortest =
        dijkstra_to_targets(graph, source, targets);
    for (std::size_t index : record_indices) {
      const PathRecord& record = jsonl.paths[index];
      ++stats.checked;
      const std::string prefix =
          "line " + std::to_string(record.line_no) + " " + key_string(record.key) + ": ";
      const auto found = shortest.find(record.key.target);
      const double expected =
          found == shortest.end() ? kInf : found->second;

      if (record.reached) {
        if (!record.has_distance || !std::isfinite(record.distance)) {
          add_failure(&stats, failures, prefix + "reached path lacks finite distance");
          continue;
        }
        if (!std::isfinite(expected)) {
          add_failure(&stats, failures, prefix + "JSON says reached but Dijkstra cannot reach target");
          continue;
        }
        if (!nearly_equal(record.distance, expected, abs_tolerance)) {
          std::ostringstream message;
          message << prefix << "not shortest: JSON distance=" << record.distance
                  << " shortest=" << expected;
          add_failure(&stats, failures, message.str());
        }
      } else if (std::isfinite(expected)) {
        std::ostringstream message;
        message << prefix << "JSON says unreached but shortest distance is " << expected;
        add_failure(&stats, failures, message.str());
      }
    }
  }
  return stats;
}

CheckStats check_completeness(const ExpectedQueries& expected,
                              const JsonlData& jsonl,
                              FailureSink* failures) {
  CheckStats stats;
  stats.checked = expected.queries.size();

  for (const auto& [key, info] : expected.by_key) {
    const auto actual = jsonl.by_key.find(key);
    const std::uint64_t actual_count =
        actual == jsonl.by_key.end() ? 0 : static_cast<std::uint64_t>(actual->second.size());
    if (actual_count == 0) {
      add_failure(&stats, failures, "missing JSONL path for " + key_string(key));
      continue;
    }
    if (actual_count != info.count) {
      add_failure(&stats, failures,
                  "expected " + std::to_string(info.count) + " JSONL path record(s) for " +
                  key_string(key) + " but found " + std::to_string(actual_count));
    }
    if (actual_count > 1) {
      add_failure(&stats, failures, "duplicate JSONL path records for " + key_string(key));
    }
    if (actual != jsonl.by_key.end()) {
      for (std::size_t record_index : actual->second) {
        const PathRecord& record = jsonl.paths[record_index];
        if (record.net != info.net_name) {
          add_failure(&stats, failures,
                      "line " + std::to_string(record.line_no) + " " +
                      key_string(record.key) + ": net name mismatch; expected '" +
                      info.net_name + "' got '" + record.net + "'");
        }
      }
    }
  }

  for (const auto& [key, record_indices] : jsonl.by_key) {
    if (expected.by_key.find(key) == expected.by_key.end()) {
      add_failure(&stats, failures,
                  "extra JSONL path record not present in metadata: " + key_string(key) +
                  " line=" + std::to_string(jsonl.paths[record_indices.front()].line_no));
    }
  }
  return stats;
}

double parse_double_arg(const std::string& text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid number for ") + name);
  }
  return value;
}

std::size_t parse_size_arg(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' ||
      value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string("invalid size for ") + name);
  }
  return static_cast<std::size_t>(value);
}

std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <outgoing.csrbin> <metadata.ifmeta.bin> <paths.jsonl> [options]\n\n"
      << "Validation modes:\n"
      << "  --check-continuity    Validate node/edge continuity and path distance sums.\n"
      << "  --check-shortest      Validate requested source-target shortest distances with Dijkstra.\n"
      << "  --check-completeness  Validate JSONL covers every valid metadata source-sink query.\n"
      << "  --all                 Enable all validation modes. Default if no mode is passed.\n\n"
	      << "Other options:\n"
	      << "  --abs-tol <value>     Absolute distance tolerance. Default: 1e-4.\n"
	      << "  --shortest-source-limit <n>\n"
	      << "                         Run Dijkstra for only the first n unique JSONL sources.\n"
	      << "                         Applies only to --check-shortest. Default: 0 = all.\n"
	      << "  --help                Print this message.\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  int positional_count = 0;
  bool mode_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--check-continuity") {
      options.check_continuity = true;
      mode_seen = true;
    } else if (arg == "--check-shortest") {
      options.check_shortest = true;
      mode_seen = true;
    } else if (arg == "--check-completeness") {
      options.check_completeness = true;
      mode_seen = true;
    } else if (arg == "--all") {
      options.check_continuity = true;
      options.check_shortest = true;
      options.check_completeness = true;
      mode_seen = true;
	    } else if (arg == "--abs-tol") {
	      options.abs_tolerance =
	          parse_double_arg(require_value(&i, argc, argv, "--abs-tol"), "--abs-tol");
	    } else if (arg == "--shortest-source-limit") {
	      options.shortest_source_limit =
	          parse_size_arg(require_value(&i, argc, argv, "--shortest-source-limit"),
	                         "--shortest-source-limit");
	    } else if (!arg.empty() && arg[0] == '-') {
	      throw std::runtime_error("unknown option: " + arg);
    } else {
      if (positional_count == 0) {
        options.csr_path = arg;
      } else if (positional_count == 1) {
        options.metadata_path = arg;
      } else if (positional_count == 2) {
        options.jsonl_path = arg;
      } else {
        throw std::runtime_error("too many positional arguments");
      }
      ++positional_count;
    }
  }

  if (options.csr_path.empty() || options.metadata_path.empty() || options.jsonl_path.empty()) {
    throw std::runtime_error("expected <outgoing.csrbin> <metadata.ifmeta.bin> <paths.jsonl>");
  }
  if (!mode_seen) {
    options.check_continuity = true;
    options.check_shortest = true;
    options.check_completeness = true;
  }
  if (options.abs_tolerance < 0.0) {
    throw std::runtime_error("--abs-tol must be nonnegative");
  }
  return options;
}

std::string enabled_modes_string(const Options& options) {
  std::vector<std::string> modes;
  if (options.check_continuity) modes.push_back("continuity");
  if (options.check_shortest) modes.push_back("shortest");
  if (options.check_completeness) modes.push_back("completeness");
  std::ostringstream out;
  for (std::size_t i = 0; i < modes.size(); ++i) {
    if (i != 0) out << ',';
    out << modes[i];
  }
  return out.str();
}

void print_check_summary(const char* name, const CheckStats& stats) {
  std::cout << "[check] " << name
            << " checked=" << stats.checked;
  if (stats.sources_checked != 0 || stats.sources_skipped != 0) {
    std::cout << " sources_checked=" << stats.sources_checked
              << " sources_skipped=" << stats.sources_skipped;
  }
  std::cout
            << " failures=" << stats.failures
            << " status=" << (stats.failures == 0 ? "PASS" : "FAIL") << '\n';
}

int main_impl(int argc, char** argv) {
  const Options options = parse_args(argc, argv);

  HostOutgoingCsrF32 graph = load_outgoing_csrbin(options.csr_path);
  RoutingMetadata metadata = load_routing_metadata(options.metadata_path, graph.nnz);
  if (metadata.metadata_node_count != static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error("metadata node count does not match outgoing CSR rows");
  }
  ExpectedQueries expected = build_expected_queries(metadata, graph);
  JsonlData jsonl = load_jsonl(options.jsonl_path);

  std::cout << "[summary] csr_rows=" << graph.rows
            << " csr_nnz=" << graph.nnz
            << " orientation=outgoing"
            << " all_weights_one=" << (graph.all_weights_one ? "yes" : "no") << '\n';
  std::cout << "[summary] metadata_route_requests=" << metadata.route_requests.size()
            << " raw_sources=" << expected.raw_source_count
            << " invalid_sources=" << expected.invalid_source_count
            << " invalid_sinks=" << expected.invalid_sink_count
            << " expected_source_sink_queries=" << expected.queries.size() << '\n';
  std::cout << "[summary] jsonl_lines=" << jsonl.jsonl_lines
            << " metadata_records=" << jsonl.metadata_records
            << " path_records=" << jsonl.paths.size() << '\n';
  std::cout << "[summary] checks_enabled=" << enabled_modes_string(options)
            << " abs_tol=" << options.abs_tolerance
            << " shortest_source_limit=" << options.shortest_source_limit << '\n';
  std::cout << "[summary] shortest_check=targeted_per_source_dijkstra_not_all_pairs\n";

  FailureSink failures;
  std::uint64_t total_failures = 0;
  if (options.check_continuity) {
    const CheckStats stats =
        check_continuity(graph, jsonl, options.abs_tolerance, &failures);
    print_check_summary("continuity", stats);
    total_failures += stats.failures;
  }
	  if (options.check_shortest) {
	    const CheckStats stats =
	        check_shortest(graph,
	                       jsonl,
	                       options.abs_tolerance,
	                       options.shortest_source_limit,
	                       &failures);
    print_check_summary("shortest", stats);
    total_failures += stats.failures;
  }
  if (options.check_completeness) {
    const CheckStats stats = check_completeness(expected, jsonl, &failures);
    print_check_summary("completeness", stats);
    total_failures += stats.failures;
  }

  if (total_failures != 0) {
    std::cerr << "[failures] total=" << total_failures << '\n';
    for (const std::string& failure : failures.first_failures) {
      std::cerr << "  - " << failure << '\n';
    }
    if (total_failures > failures.first_failures.size()) {
      std::cerr << "  - ... "
                << (total_failures - failures.first_failures.size())
                << " additional failure(s) omitted\n";
    }
    return 2;
  }

  std::cout << "[summary] status=PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }
}
