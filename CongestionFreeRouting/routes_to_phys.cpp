// Reconstructs a routed FPGA Interchange PhysicalNetlist from PathFinder's
// route JSONL output and the RIPS interchange metadata sidecar.
//
// Benchmark-facing use, after interchange_to_csr and pathfinder have run:
//   routes_to_phys <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys>
//
// Expected generated schema header:
//   PhysicalNetlist.capnp.h
//
// Example compile command:
//   g++ -std=c++17 -O3 \
//     -I<generated-schema-dir> \
//     CongestionFreeRouting/routes_to_phys.cpp \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o routes_to_phys

#include "PhysicalNetlist.capnp.h"
#include "interchange/gzip_io.hpp"
#include "interchange/import_policy.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t LEGACY_METADATA_VERSION = 4;
constexpr std::uint64_t CURRENT_METADATA_VERSION = 7;
constexpr std::uint64_t ARTIFACT_PAIR_METADATA_VERSION = 5;
constexpr std::uint64_t COMPACT_METADATA_VERSION = 6;
constexpr std::uint64_t ENDPOINT_PIP_METADATA_VERSION = 7;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kInvalidRouteNode =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoEndpointPip =
    std::numeric_limits<std::uint64_t>::max();

struct SitePinKey {
  std::string site;
  std::string pin;

  bool operator<(const SitePinKey& other) const {
    return std::tie(site, pin) < std::tie(other.site, other.pin);
  }
};

struct RouteSitePin {
  int node = -1;
  std::string site;
  std::string pin;
  bool reached = true;
  std::uint64_t endpoint_pip_index = kNoEndpointPip;
};

struct RouteEdge {
  int from = -1;
  int to = -1;
  std::uint64_t csr_edge = 0;
  std::string tile;
  std::string wire0;
  std::string wire1;
  bool forward = true;
  bool attachment_field_present = false;
  std::optional<std::uint64_t> attachment;
  bool site_field_present = false;
  std::optional<std::string> site;
};

struct NetRoute {
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::string net;
  bool routed = false;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
  std::vector<RouteEdge> edges;
};

struct StoredStubBranch {
  SitePinKey key;
  std::uint32_t branch_index = 0;
  bool consumed = false;
};

struct StubBranchStore {
  std::vector<StoredStubBranch> stubs;
  std::map<SitePinKey, std::vector<std::size_t>> by_key;
  std::map<SitePinKey, std::size_t> cursor_by_key;
};

struct MetadataRouteRequest {
  std::string net;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
};

enum class MetadataEndpointPipRole : std::uint64_t {
  kSource = 0,
  kSink = 1,
};

struct MetadataEndpointPip {
  std::uint64_t csr_edge = 0;
  int from = -1;
  int to = -1;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
  std::uint64_t site_string = 0;
  int endpoint_node = -1;
  MetadataEndpointPipRole role = MetadataEndpointPipRole::kSource;
};

struct RoutingMetadataSummary {
  std::uint64_t version = 0;
  std::uint64_t node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::vector<std::string> strings;
  std::vector<MetadataEndpointPip> endpoint_pips;
  std::vector<MetadataRouteRequest> route_requests;
};

struct JsonValue {
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
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
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          if (pos_ + 4 > text_.size()) {
            throw std::runtime_error("truncated JSON unicode escape");
          }
          unsigned code = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_++];
            code <<= 4;
            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
            else throw std::runtime_error("bad JSON unicode escape");
          }
          if (code <= 0x7f) {
            out.push_back(static_cast<char>(code));
          } else {
            throw std::runtime_error("non-ASCII JSON unicode escapes are not supported");
          }
          break;
        }
        default:
          throw std::runtime_error("bad JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  double parse_number() {
    skip_ws();
    const std::size_t begin = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      throw std::runtime_error("invalid JSON number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
      if (pos_ < text_.size() &&
          std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("JSON number has a leading zero");
      }
    } else {
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      const std::size_t fractional_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == fractional_begin) {
        throw std::runtime_error("JSON number has an empty fraction");
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      const std::size_t exponent_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == exponent_begin) {
        throw std::runtime_error("JSON number has an empty exponent");
      }
    }
    const std::string token = text_.substr(begin, pos_ - begin);
    std::size_t consumed = 0;
    const double value = std::stod(token, &consumed);
    if (consumed != token.size() || !std::isfinite(value)) {
      throw std::runtime_error("invalid finite JSON number");
    }
    return value;
  }

  void expect_literal(const char* literal) {
    const std::size_t n = std::strlen(literal);
    if (text_.compare(pos_, n, literal) != 0) {
      throw std::runtime_error(std::string("expected JSON literal ") + literal);
    }
    pos_ += n;
  }

  std::string text_;
  std::size_t pos_ = 0;
};

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error(std::string("failed while reading ") + name);
  return value;
}

int read_route_node(std::ifstream& in, const char* name) {
  const std::uint64_t raw = read_u64(in, name);
  if (raw == kInvalidRouteNode) {
    return -1;
  }
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

int checked_nonnegative_int(std::uint64_t raw, const char* name) {
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

std::size_t checked_size_count(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds size_t range");
  }
  return static_cast<std::size_t>(count);
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

// Seek over unused bulk metadata without reading it through a temporary
// buffer. Checking the remaining file length first is required because a
// standard seek is otherwise allowed to move beyond end-of-file silently.
void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) {
    return;
  }
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) +
                             " byte count is too large to seek");
  }
  const std::streampos current = in.tellg();
  if (current == std::streampos(-1)) {
    throw std::runtime_error(std::string("failed while locating ") + name);
  }
  in.seekg(0, std::ios::end);
  const std::streampos end = in.tellg();
  if (!in || end == std::streampos(-1) || end < current ||
      static_cast<std::uint64_t>(end - current) < count) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
  in.seekg(current + static_cast<std::streamoff>(count));
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

std::string read_metadata_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large");
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) throw std::runtime_error("failed while reading metadata string");
  }
  return text;
}

const std::string& string_at(const RoutingMetadataSummary& metadata,
                             std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata references an invalid string index: " +
                             std::to_string(index));
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

RoutingMetadataSummary load_metadata_summary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open metadata file: " + path.string());

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version < LEGACY_METADATA_VERSION ||
      version > CURRENT_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  if (version >= ARTIFACT_PAIR_METADATA_VERSION) {
    routing::interchange::InterchangeArtifactPairId id;
    id.high = read_u64(in, "metadata artifact pair id high");
    id.low = read_u64(in, "metadata artifact pair id low");
    if (id.is_zero()) {
      throw std::runtime_error("metadata artifact pair id must not be zero");
    }
    artifact_pair_id = id;
  }

  const std::uint64_t string_count = read_u64(in, "string count");
  const std::uint64_t node_count = read_u64(in, "node count");
  const std::uint64_t edge_attr_count = read_u64(in, "edge attr count");
  const std::uint64_t pip_data_count = read_u64(in, "pip data count");
  const std::uint64_t endpoint_pip_count =
      version >= ENDPOINT_PIP_METADATA_VERSION
          ? read_u64(in, "endpoint PIP count")
          : 0;
  const std::uint64_t site_pin_attr_count = read_u64(in, "site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "physical netlist byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "logical netlist byte count");

  (void)read_u64(in, "device path string");
  (void)read_u64(in, "physical path string");
  (void)read_u64(in, "logical path string");
  (void)read_u64(in, "logical design name string");

  RoutingMetadataSummary metadata;
  metadata.version = version;
  metadata.node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.artifact_pair_id = artifact_pair_id;
  metadata.strings.reserve(checked_size_count(string_count, "string count"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_metadata_string(in));
  }

  if (version < COMPACT_METADATA_VERSION) {
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node ids"),
               "node ids");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node min x coordinates"),
               "node min x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node max x coordinates"),
               "node max x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node min y coordinates"),
               "node min y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node max y coordinates"),
               "node max y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node tile type strings"),
               "node tile type strings");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node wire type strings"),
               "node wire type strings");
  }
  // These full-device tables can contain tens of millions of records.  Route
  // reconstruction only needs the sparse, self-contained EndpointPip table,
  // so retain the original streaming behavior here.
  skip_bytes(in,
             checked_byte_count(edge_attr_count,
                                2 * sizeof(std::uint64_t), "edge attrs"),
             "edge attrs");
  skip_bytes(in,
             checked_byte_count(pip_data_count,
                                3 * sizeof(std::uint64_t), "PIP data"),
             "PIP data");

  metadata.endpoint_pips.reserve(
      checked_size_count(endpoint_pip_count, "endpoint PIP count"));
  std::set<std::uint64_t> endpoint_pip_csr_edges;
  for (std::uint64_t i = 0; i < endpoint_pip_count; ++i) {
    MetadataEndpointPip endpoint;
    endpoint.csr_edge = read_u64(in, "endpoint PIP CSR edge");
    endpoint.from = checked_nonnegative_int(
        read_u64(in, "endpoint PIP source node"),
        "endpoint PIP source node");
    endpoint.to = checked_nonnegative_int(
        read_u64(in, "endpoint PIP destination node"),
        "endpoint PIP destination node");
    endpoint.tile_string = read_u64(in, "endpoint PIP tile string");
    endpoint.wire0_string = read_u64(in, "endpoint PIP wire0 string");
    endpoint.wire1_string = read_u64(in, "endpoint PIP wire1 string");
    const std::uint64_t forward = read_u64(in, "endpoint PIP forward flag");
    endpoint.site_string = read_u64(in, "endpoint PIP site string");
    endpoint.endpoint_node = checked_nonnegative_int(
        read_u64(in, "endpoint PIP endpoint node"),
        "endpoint PIP endpoint node");
    const std::uint64_t role = read_u64(in, "endpoint PIP role");

    if (forward > 1) {
      throw std::runtime_error(
          "metadata endpoint PIP has an invalid forward flag");
    }
    endpoint.forward = forward != 0;
    if (role == static_cast<std::uint64_t>(
                    MetadataEndpointPipRole::kSource)) {
      endpoint.role = MetadataEndpointPipRole::kSource;
    } else if (role == static_cast<std::uint64_t>(
                           MetadataEndpointPipRole::kSink)) {
      endpoint.role = MetadataEndpointPipRole::kSink;
    } else {
      throw std::runtime_error("metadata endpoint PIP has an invalid role");
    }
    if (endpoint.csr_edge >= edge_attr_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid CSR edge");
    }
    if (static_cast<std::uint64_t>(endpoint.from) >= node_count ||
        static_cast<std::uint64_t>(endpoint.to) >= node_count ||
        static_cast<std::uint64_t>(endpoint.endpoint_node) >= node_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid node");
    }
    if (endpoint.from == endpoint.to ||
        endpoint.endpoint_node == endpoint.from ||
        endpoint.endpoint_node == endpoint.to) {
      throw std::runtime_error(
          "metadata endpoint PIP has invalid endpoint alignment");
    }
    (void)string_at(metadata, endpoint.tile_string);
    (void)string_at(metadata, endpoint.wire0_string);
    (void)string_at(metadata, endpoint.wire1_string);
    if (string_at(metadata, endpoint.site_string).empty()) {
      throw std::runtime_error(
          "metadata endpoint PIP has an empty concrete site");
    }
    if (!endpoint_pip_csr_edges.insert(endpoint.csr_edge).second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }

    metadata.endpoint_pips.push_back(endpoint);
  }

  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t), "site pin attrs"),
             "site pin attrs");

  metadata.route_requests.reserve(
      checked_size_count(route_request_count, "route request count"));
  for (std::uint64_t i = 0; i < route_request_count; ++i) {
    MetadataRouteRequest request;
    request.net = string_at(metadata, read_u64(in, "route request net"));
    (void)read_u64(in, "route request logical net");

    const std::uint64_t source_count = read_u64(in, "source count");
    request.sources.reserve(checked_size_count(source_count, "source count"));
    for (std::uint64_t s = 0; s < source_count; ++s) {
      RouteSitePin source;
      source.node = read_route_node(in, "source node");
      source.site = string_at(metadata, read_u64(in, "source site"));
      source.pin = string_at(metadata, read_u64(in, "source pin"));
      source.endpoint_pip_index =
          version >= ENDPOINT_PIP_METADATA_VERSION
              ? read_u64(in, "source endpoint PIP index")
              : kNoEndpointPip;
      if (source.endpoint_pip_index != kNoEndpointPip) {
        if (source.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata source references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(source.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSource ||
            endpoint.endpoint_node != source.node) {
          throw std::runtime_error(
              "metadata source references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sources.push_back(std::move(source));
    }

    const std::uint64_t sink_count = read_u64(in, "sink count");
    request.sinks.reserve(checked_size_count(sink_count, "sink count"));
    for (std::uint64_t s = 0; s < sink_count; ++s) {
      RouteSitePin sink;
      sink.node = read_route_node(in, "sink node");
      sink.site = string_at(metadata, read_u64(in, "sink site"));
      sink.pin = string_at(metadata, read_u64(in, "sink pin"));
      sink.endpoint_pip_index =
          version >= ENDPOINT_PIP_METADATA_VERSION
              ? read_u64(in, "sink endpoint PIP index")
              : kNoEndpointPip;
      if (sink.endpoint_pip_index != kNoEndpointPip) {
        if (sink.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata sink references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(sink.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSink ||
            endpoint.endpoint_node != sink.node) {
          throw std::runtime_error(
              "metadata sink references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sinks.push_back(std::move(sink));
    }
    metadata.route_requests.push_back(std::move(request));
  }

  skip_bytes(in,
             checked_byte_count(logical_cell_count,
                                3 * sizeof(std::uint64_t), "logical cells"),
             "logical cells");
  skip_bytes(in,
             checked_byte_count(logical_net_count,
                                4 * sizeof(std::uint64_t), "logical nets"),
             "logical nets");
  skip_bytes(in,
             checked_byte_count(logical_port_instance_count,
                                7 * sizeof(std::uint64_t),
                                "logical port instances"),
             "logical port instances");
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t),
                                "blocked nodes"),
             "blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count, sizeof(std::uint64_t),
                                "sink stop nodes"),
             "sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "physical netlist bytes");
  skip_bytes(in, logical_netlist_byte_count, "logical netlist bytes");

  return metadata;
}

int json_int(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
  const double value = found->second.as_number(key);
  if (!std::isfinite(value) || std::trunc(value) != value ||
      value < static_cast<double>(std::numeric_limits<int>::min()) ||
      value > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string("JSON field is not an in-range integer: ") + key);
  }
  return static_cast<int>(value);
}

std::uint64_t json_u64(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::runtime_error(std::string("missing JSON key: ") + key);
  }
  const double value = found->second.as_number(key);
  // JsonValue deliberately uses double for the small route JSON parser.  The
  // router's 64-bit CSR offsets are exact throughout the binary formats; JSON
  // integers remain exact through 2^53-1, well beyond the former int32 cap.
  constexpr double kLargestExactJsonInteger = 9007199254740991.0;
  if (!std::isfinite(value) || std::trunc(value) != value || value < 0.0 ||
      value > kLargestExactJsonInteger) {
    throw std::runtime_error(
        std::string("JSON field is not an exact nonnegative integer: ") +
        key);
  }
  return static_cast<std::uint64_t>(value);
}

std::string json_string(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
  return found->second.as_string(key);
}

std::optional<std::string> optional_json_string(
    const JsonValue::Object& object,
    const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    return std::nullopt;
  }
  return found->second.as_string(key);
}

std::optional<std::uint64_t> nullable_json_u64(
    const JsonValue::Object& object,
    const char* key,
    bool* present) {
  const auto found = object.find(key);
  *present = found != object.end();
  if (found == object.end() || found->second.is_null()) {
    return std::nullopt;
  }
  const double value = found->second.as_number(key);
  constexpr double kLargestExactJsonInteger = 9007199254740991.0;
  if (!std::isfinite(value) || std::trunc(value) != value || value < 0.0 ||
      value > kLargestExactJsonInteger) {
    throw std::runtime_error(
        std::string("JSON field is not an exact nonnegative integer: ") +
        key);
  }
  return static_cast<std::uint64_t>(value);
}

std::optional<std::string> nullable_json_string(
    const JsonValue::Object& object,
    const char* key,
    bool* present) {
  const auto found = object.find(key);
  *present = found != object.end();
  if (found == object.end() || found->second.is_null()) {
    return std::nullopt;
  }
  return found->second.as_string(key);
}

bool json_bool(const JsonValue::Object& object, const char* key, bool fallback = false) {
  const auto found = object.find(key);
  if (found == object.end()) return fallback;
  return found->second.as_bool(key);
}

RouteSitePin parse_route_site_pin(const JsonValue& value, bool require_reached) {
  const auto& object = value.as_object("site pin");
  RouteSitePin pin;
  pin.node = json_int(object, "node");
  pin.site = json_string(object, "site");
  pin.pin = json_string(object, "pin");
  pin.reached = json_bool(object, "reached", true);
  if (require_reached && !pin.reached) {
    throw std::runtime_error("route contains an unreached sink");
  }
  return pin;
}

NetRoute parse_route_line(const std::string& line) {
  const JsonValue root = JsonParser(line).parse();
  const auto& object = root.as_object("route");
  NetRoute route;
  if (const std::optional<std::string> raw_id =
          optional_json_string(object, "artifact_pair_id");
      raw_id.has_value()) {
    route.artifact_pair_id =
        routing::interchange::parse_interchange_artifact_pair_id(*raw_id);
  }
  route.net = json_string(object, "net");
  route.routed = json_bool(object, "routed", false);

  for (const JsonValue& value : object.at("sources").as_array("sources")) {
    route.sources.push_back(parse_route_site_pin(value, false));
  }
  for (const JsonValue& value : object.at("sinks").as_array("sinks")) {
    route.sinks.push_back(parse_route_site_pin(value, false));
  }
  for (const JsonValue& value : object.at("edges").as_array("edges")) {
    const auto& edge_object = value.as_object("edge");
    RouteEdge edge;
    edge.from = json_int(edge_object, "from");
    edge.to = json_int(edge_object, "to");
    edge.csr_edge = json_u64(edge_object, "csr_edge");
    edge.tile = json_string(edge_object, "tile");
    edge.wire0 = json_string(edge_object, "wire0");
    edge.wire1 = json_string(edge_object, "wire1");
    edge.forward = json_bool(edge_object, "forward", true);
    edge.attachment = nullable_json_u64(
        edge_object, "attachment", &edge.attachment_field_present);
    edge.site =
        nullable_json_string(edge_object, "site", &edge.site_field_present);
    route.edges.push_back(std::move(edge));
  }
  return route;
}

std::unordered_map<std::string, NetRoute> load_routes_jsonl(
    const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("could not open routes file: " + path.string());

  std::unordered_map<std::string, NetRoute> routes;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    NetRoute route = parse_route_line(line);
    const std::string net = route.net;
    if (!routes.emplace(net, std::move(route)).second) {
      throw std::runtime_error("duplicate route entry for net: " + net);
    }
  }
  if (routes.empty()) {
    throw std::runtime_error("routes file is empty: " + path.string());
  }
  return routes;
}

void validate_routes_against_metadata(
    const std::unordered_map<std::string, NetRoute>& routes,
    const RoutingMetadataSummary& metadata) {
  std::unordered_map<std::string, const MetadataRouteRequest*> requests_by_net;
  for (const MetadataRouteRequest& request : metadata.route_requests) {
    if (!requests_by_net.emplace(request.net, &request).second) {
      throw std::runtime_error(
          "metadata contains duplicate route request: " + request.net);
    }
  }

  std::unordered_map<std::uint64_t, std::size_t> endpoint_pip_by_csr_edge;
  endpoint_pip_by_csr_edge.reserve(metadata.endpoint_pips.size());
  for (std::size_t index = 0; index < metadata.endpoint_pips.size(); ++index) {
    const MetadataEndpointPip& endpoint = metadata.endpoint_pips[index];
    if (!endpoint_pip_by_csr_edge.emplace(endpoint.csr_edge, index).second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }
  }

  for (const auto& [net, route] : routes) {
    const auto found = requests_by_net.find(net);
    if (found == requests_by_net.end()) {
      throw std::runtime_error("route file contains net not present in metadata: " + net);
    }
    const MetadataRouteRequest& request = *found->second;
    if (route.sources.size() != request.sources.size()) {
      std::ostringstream out;
      out << "route source count for " << net << " is "
          << route.sources.size() << " but metadata expects "
          << request.sources.size();
      throw std::runtime_error(out.str());
    }
    if (route.sinks.size() != request.sinks.size()) {
      std::ostringstream out;
      out << "route sink count for " << net << " is " << route.sinks.size()
          << " but metadata expects " << request.sinks.size();
      throw std::runtime_error(out.str());
    }
    const auto require_same_endpoint = [&](const RouteSitePin& actual,
                                           const RouteSitePin& expected,
                                           const char* role,
                                           std::size_t index) {
      if (actual.node != expected.node || actual.site != expected.site ||
          actual.pin != expected.pin) {
        throw std::runtime_error(
            "route " + std::string(role) + " " +
            std::to_string(index) + " does not match metadata for net " +
            net);
      }
    };
    for (std::size_t index = 0; index < route.sources.size(); ++index) {
      require_same_endpoint(route.sources[index], request.sources[index],
                            "source", index);
    }
    for (std::size_t index = 0; index < route.sinks.size(); ++index) {
      require_same_endpoint(route.sinks[index], request.sinks[index],
                            "sink", index);
    }

    std::set<std::uint64_t> authorized_source_attachments;
    std::set<std::uint64_t> authorized_reached_sink_attachments;
    std::map<int, std::uint64_t> source_attachment_by_node;
    for (std::size_t index = 0; index < request.sources.size(); ++index) {
      const RouteSitePin& source = request.sources[index];
      if (source.endpoint_pip_index == kNoEndpointPip) {
        continue;
      }
      authorized_source_attachments.insert(source.endpoint_pip_index);
      const auto inserted = source_attachment_by_node.emplace(
          source.node, source.endpoint_pip_index);
      if (!inserted.second &&
          inserted.first->second != source.endpoint_pip_index) {
        throw std::runtime_error(
            "metadata has ambiguous source attachments for one node in net " +
            net);
      }
    }
    for (std::size_t index = 0; index < request.sinks.size(); ++index) {
      if (route.sinks[index].reached &&
          request.sinks[index].endpoint_pip_index != kNoEndpointPip) {
        authorized_reached_sink_attachments.insert(
            request.sinks[index].endpoint_pip_index);
      }
    }

    std::map<int, const RouteEdge*> incoming_by_node;
    std::map<int, std::vector<const RouteEdge*>> outgoing_by_node;
    std::set<std::pair<int, int>> node_pairs;
    std::set<std::uint64_t> csr_edges;
    std::set<std::uint64_t> used_attachments;

    for (const RouteEdge& edge : route.edges) {
      if (edge.from < 0 || edge.to < 0 || edge.from == edge.to) {
        throw std::runtime_error("route contains an invalid edge for net " +
                                 net);
      }
      if (static_cast<std::uint64_t>(edge.from) >= metadata.node_count ||
          static_cast<std::uint64_t>(edge.to) >= metadata.node_count) {
        throw std::runtime_error(
            "route edge references an invalid node for net " + net);
      }
      if (edge.csr_edge >= metadata.edge_attr_count) {
        throw std::runtime_error(
            "route edge references an invalid CSR edge for net " + net);
      }
      if (!node_pairs.emplace(edge.from, edge.to).second ||
          !csr_edges.insert(edge.csr_edge).second) {
        throw std::runtime_error(
            "route contains a duplicate edge for net " + net);
      }
      const auto incoming = incoming_by_node.emplace(edge.to, &edge);
      if (!incoming.second && incoming.first->second->from != edge.from) {
        throw std::runtime_error(
            "route drives one node from multiple parents: " + net);
      }
      outgoing_by_node[edge.from].push_back(&edge);

      if (metadata.version >= ENDPOINT_PIP_METADATA_VERSION &&
          (!edge.attachment_field_present || !edge.site_field_present)) {
        throw std::runtime_error(
            "v7 route edge is missing attachment/site fields for net " + net);
      }
      if (edge.attachment.has_value() != edge.site.has_value()) {
        throw std::runtime_error(
            "route edge must pair attachment and site for net " + net);
      }

      const auto endpoint_for_edge =
          endpoint_pip_by_csr_edge.find(edge.csr_edge);
      if (endpoint_for_edge == endpoint_pip_by_csr_edge.end()) {
        if (edge.attachment.has_value() || edge.site.has_value()) {
          throw std::runtime_error(
              "conventional route edge must not carry attachment/site for net " +
              net);
        }
        continue;
      }

      const std::size_t expected_index = endpoint_for_edge->second;
      if (!edge.attachment.has_value() || !edge.site.has_value()) {
        throw std::runtime_error(
            "endpoint attachment edge is encoded as conventional for net " +
            net);
      }
      if (*edge.attachment != expected_index ||
          *edge.attachment >= metadata.endpoint_pips.size()) {
        throw std::runtime_error(
            "route attachment index does not match its CSR edge for net " +
            net);
      }
      if (!used_attachments.insert(*edge.attachment).second) {
        throw std::runtime_error(
            "route reuses one endpoint attachment in net " + net);
      }

      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[expected_index];
      if (edge.from != endpoint.from || edge.to != endpoint.to ||
          edge.csr_edge != endpoint.csr_edge ||
          edge.tile != string_at(metadata, endpoint.tile_string) ||
          edge.wire0 != string_at(metadata, endpoint.wire0_string) ||
          edge.wire1 != string_at(metadata, endpoint.wire1_string) ||
          edge.forward != endpoint.forward ||
          *edge.site != string_at(metadata, endpoint.site_string)) {
        throw std::runtime_error(
            "route attachment does not exactly match sparse metadata for net " +
            net);
      }

      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        if (authorized_source_attachments.count(*edge.attachment) == 0) {
          throw std::runtime_error(
              "route source attachment belongs to a different endpoint for net " +
              net);
        }
      } else if (authorized_reached_sink_attachments.count(
                     *edge.attachment) == 0) {
        throw std::runtime_error(
            "route sink attachment belongs to a different or unreached "
            "endpoint for net " + net);
      }
    }

    for (std::uint64_t index : used_attachments) {
      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[static_cast<std::size_t>(index)];
      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        const auto corridor = incoming_by_node.find(endpoint.from);
        const auto attachment_children = outgoing_by_node.find(endpoint.from);
        const auto root_children = outgoing_by_node.find(endpoint.endpoint_node);
        const bool root_contains_corridor =
            corridor != incoming_by_node.end() &&
            root_children != outgoing_by_node.end() &&
            std::find(root_children->second.begin(),
                      root_children->second.end(), corridor->second) !=
                root_children->second.end();
        const bool source_contains_attachment =
            attachment_children != outgoing_by_node.end() &&
            std::any_of(
                attachment_children->second.begin(),
                attachment_children->second.end(),
                [&](const RouteEdge* edge) {
                  return edge->attachment == index;
                });
        // The filtered graph makes endpoint_node source-exclusive and gives
        // from exactly one incoming edge: this corridor. Extra outgoing edges
        // are same-net source fanout rather than attachment transit.
        if (corridor == incoming_by_node.end() ||
            corridor->second->from != endpoint.endpoint_node ||
            corridor->second->attachment.has_value() ||
            !root_contains_corridor || !source_contains_attachment ||
            incoming_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "source attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      } else {
        const auto corridor = outgoing_by_node.find(endpoint.to);
        if (corridor == outgoing_by_node.end() ||
            corridor->second.size() != 1 ||
            corridor->second.front()->to != endpoint.endpoint_node ||
            corridor->second.front()->attachment.has_value() ||
            outgoing_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "sink attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      }
    }

    // Endpoint metadata authorizes an audited pseudo-PIP when a route needs
    // that BITSLICE crossing.  A route that stays entirely on conventional CSR
    // edges is already represented exactly and need not consume the optional
    // attachment.  Used attachments remain subject to the strict corridor and
    // no-transit checks above.
  }
}

std::vector<std::uint8_t> read_gzip_or_plain_file(const std::filesystem::path& path) {
  std::vector<std::uint8_t> bytes;
  routing::interchange::read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t byte_count) {
        if (byte_count > bytes.max_size() - bytes.size()) {
          throw std::runtime_error("decoded input is too large: " +
                                   path.string());
        }
        const std::size_t old_size = bytes.size();
        bytes.resize(old_size + byte_count);
        std::memcpy(bytes.data() + old_size, data, byte_count);
      });
  if (bytes.empty()) throw std::runtime_error("input file is empty: " + path.string());
  return bytes;
}

std::vector<capnp::word> bytes_to_words(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0) {
    throw std::runtime_error(
        "decoded physical netlist is not Cap'n Proto word-aligned");
  }
  const std::size_t word_count = bytes.size() / sizeof(capnp::word);
  std::vector<capnp::word> words(word_count);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

void write_gzip_file(const std::filesystem::path& path,
                     kj::ArrayPtr<const kj::byte> bytes) {
  gzFile file = gzopen(path.string().c_str(), "wb6");
  if (!file) throw std::runtime_error("could not open output file: " + path.string());

  // gzwrite takes an unsigned length but reports the byte count as int.  One
  // call cannot safely represent a multi-gigabyte flattened PhysicalNetlist.
  // Bounded chunks also make every return value unambiguous on all zlib builds.
  constexpr std::size_t kWriteChunkBytes = 64ULL * 1024ULL * 1024ULL;
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const unsigned int chunk_size = static_cast<unsigned int>(
        std::min<std::size_t>(bytes.size() - offset, kWriteChunkBytes));
    const int written = gzwrite(file, bytes.begin() + offset, chunk_size);
    if (written != static_cast<int>(chunk_size)) {
      int zlib_error = Z_OK;
      const char* raw_message = gzerror(file, &zlib_error);
      // gzerror's pointer belongs to the gzFile and becomes invalid at close.
      std::string message = raw_message == nullptr ? std::string() : raw_message;
      const int saved_errno = errno;
      (void)gzclose(file);
      if (message.empty()) {
        message = zlib_error == Z_ERRNO
                      ? std::strerror(saved_errno)
                      : zError(zlib_error);
      }
      if (message.empty()) {
        message = "zlib write error " + std::to_string(zlib_error);
      }
      throw std::runtime_error("failed while writing " + path.string() + ": " +
                               message);
    }
    offset += static_cast<std::size_t>(written);
  }

  const int close_status = gzclose(file);
  if (close_status != Z_OK) {
    const int saved_errno = errno;
    const char* zlib_message = zError(close_status);
    const std::string message =
        close_status == Z_ERRNO
            ? std::strerror(saved_errno)
            : (zlib_message == nullptr ? "zlib error" : zlib_message);
    throw std::runtime_error("failed while closing " + path.string() + ": " +
                             message);
  }
}

std::vector<std::string> copy_string_list(
    capnp::List<capnp::Text>::Builder str_list,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  std::vector<std::string> strings;
  strings.reserve(str_list.size());
  for (std::uint32_t i = 0; i < str_list.size(); ++i) {
    capnp::Text::Builder text = str_list[i];
    strings.emplace_back(text.cStr(), text.size());
    string_to_index.emplace(strings.back(), i);
  }
  return strings;
}

std::uint32_t string_index(const std::string& text,
                           std::vector<std::string>& strings,
                           std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  const auto found = string_to_index.find(text);
  if (found != string_to_index.end()) return found->second;
  if (strings.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("PhysicalNetlist strList exceeds uint32_t");
  }
  const std::uint32_t index = static_cast<std::uint32_t>(strings.size());
  strings.push_back(text);
  string_to_index.emplace(strings.back(), index);
  return index;
}

std::string phys_string_at(const std::vector<std::string>& strings, std::uint32_t index) {
  if (index >= strings.size()) throw std::runtime_error("PhysicalNetlist string index out of range");
  return strings[static_cast<std::size_t>(index)];
}

void copy_route_branch(PhysicalNetlist::PhysNetlist::RouteBranch::Builder source,
                       PhysicalNetlist::PhysNetlist::RouteBranch::Builder destination) {
  auto source_segment = source.getRouteSegment();
  auto destination_segment = destination.initRouteSegment();
  if (source_segment.isBelPin()) {
    auto source_bel_pin = source_segment.getBelPin();
    auto destination_bel_pin = destination_segment.initBelPin();
    destination_bel_pin.setSite(source_bel_pin.getSite());
    destination_bel_pin.setBel(source_bel_pin.getBel());
    destination_bel_pin.setPin(source_bel_pin.getPin());
  } else if (source_segment.isSitePin()) {
    auto source_site_pin = source_segment.getSitePin();
    auto destination_site_pin = destination_segment.initSitePin();
    destination_site_pin.setSite(source_site_pin.getSite());
    destination_site_pin.setPin(source_site_pin.getPin());
  } else if (source_segment.isPip()) {
    auto source_pip = source_segment.getPip();
    auto destination_pip = destination_segment.initPip();
    destination_pip.setTile(source_pip.getTile());
    destination_pip.setWire0(source_pip.getWire0());
    destination_pip.setWire1(source_pip.getWire1());
    destination_pip.setForward(source_pip.getForward());
    destination_pip.setIsFixed(source_pip.getIsFixed());
    if (source_pip.isSite()) {
      destination_pip.setSite(source_pip.getSite());
    } else {
      destination_pip.setNoSite();
    }
  } else if (source_segment.isSitePIP()) {
    auto source_site_pip = source_segment.getSitePIP();
    auto destination_site_pip = destination_segment.initSitePIP();
    destination_site_pip.setSite(source_site_pip.getSite());
    destination_site_pip.setBel(source_site_pip.getBel());
    destination_site_pip.setPin(source_site_pip.getPin());
    destination_site_pip.setIsFixed(source_site_pip.getIsFixed());
    if (source_site_pip.isIsInverting()) {
      destination_site_pip.setIsInverting(source_site_pip.getIsInverting());
    } else {
      destination_site_pip.setInverts();
    }
  } else {
    throw std::runtime_error("unsupported PhysicalNetlist route segment");
  }

  auto source_children = source.getBranches();
  auto destination_children =
      destination.initBranches(static_cast<std::uint32_t>(source_children.size()));
  for (std::uint32_t i = 0; i < source_children.size(); ++i) {
    copy_route_branch(source_children[i], destination_children[i]);
  }
}

StubBranchStore snapshot_top_level_stubs(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder stubs,
    const std::vector<std::string>& strings,
    const std::string& net_name) {
  StubBranchStore store;
  store.stubs.reserve(stubs.size());
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    auto stub = stubs[i];
    auto segment = stub.getRouteSegment();
    if (!segment.isSitePin()) {
      throw std::runtime_error("non-sitePin stub found in net: " + net_name);
    }
    auto site_pin = segment.getSitePin();
    SitePinKey key{phys_string_at(strings, site_pin.getSite()),
                   phys_string_at(strings, site_pin.getPin())};
    StoredStubBranch stored;
    stored.key = key;
    stored.branch_index = i;
    const std::size_t index = store.stubs.size();
    store.stubs.push_back(std::move(stored));
    store.by_key[key].push_back(index);
  }
  return store;
}

std::uint32_t consume_stub_branch(StubBranchStore& store,
                                  const SitePinKey& key,
                                  const std::string& net_name) {
  const auto found = store.by_key.find(key);
  if (found == store.by_key.end()) {
    throw std::runtime_error("routed sink was not present as a stub in net: " + net_name);
  }

  std::size_t& cursor = store.cursor_by_key[key];
  if (cursor >= found->second.size()) {
    throw std::runtime_error("routed sink used more times than its stubs in net: " + net_name);
  }
  const std::size_t stub_index = found->second[cursor++];
  store.stubs[stub_index].consumed = true;
  return store.stubs[stub_index].branch_index;
}

void collect_site_pin_branches(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    const std::vector<std::string>& strings,
    std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>& out) {
  auto segment = branch.getRouteSegment();
  if (segment.isSitePin()) {
    auto site_pin = segment.getSitePin();
    out.push_back({{phys_string_at(strings, site_pin.getSite()),
                    phys_string_at(strings, site_pin.getPin())},
                   branch});
  }

  auto children = branch.getBranches();
  for (std::uint32_t i = 0; i < children.size(); ++i) {
    collect_site_pin_branches(children[i], strings, out);
  }
}

struct RouteTables {
  std::map<int, std::vector<RouteEdge>> children_by_node;
  std::map<int, std::vector<SitePinKey>> sinks_by_node;
  std::map<SitePinKey, int> source_node_by_pin;
  int edge_count = 0;
};

RouteTables build_route_tables(const NetRoute& route) {
  RouteTables tables;
  std::map<int, int> parent_by_child;

  for (const RouteEdge& edge : route.edges) {
    const auto parent = parent_by_child.find(edge.to);
    if (parent != parent_by_child.end() && parent->second != edge.from) {
      throw std::runtime_error("route drives one node from multiple parents: " + route.net);
    }
    parent_by_child[edge.to] = edge.from;
    tables.children_by_node[edge.from].push_back(edge);
    ++tables.edge_count;
  }
  for (const RouteSitePin& source : route.sources) {
    if (source.node < 0) continue;
    SitePinKey key{source.site, source.pin};
    const auto [found, inserted] =
        tables.source_node_by_pin.emplace(key, source.node);
    if (!inserted && found->second != source.node) {
      throw std::runtime_error(
          "one source site pin maps to multiple nodes in route: " +
          route.net);
    }
  }
  for (const RouteSitePin& sink : route.sinks) {
    if (!sink.reached || sink.node < 0) continue;
    tables.sinks_by_node[sink.node].push_back({sink.site, sink.pin});
  }
  return tables;
}

bool has_reached_sink(const NetRoute& route) {
  for (const RouteSitePin& sink : route.sinks) {
    if (sink.reached) {
      return true;
    }
  }
  return false;
}

bool top_level_stubs_are_site_pins(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder stubs) {
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    if (!stubs[i].getRouteSegment().isSitePin()) {
      return false;
    }
  }
  return true;
}

int insert_route_tree(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    int node,
    const NetRoute& route,
    const RouteTables& tables,
    StubBranchStore& stub_store,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder old_stubs,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index,
    std::vector<int> ancestors) {
  if (std::find(ancestors.begin(), ancestors.end(), node) != ancestors.end()) {
    throw std::runtime_error("route tree has a cycle in net: " + route.net);
  }
  ancestors.push_back(node);

  const auto children_it = tables.children_by_node.find(node);
  const auto sinks_it = tables.sinks_by_node.find(node);
  const std::size_t child_count =
      children_it == tables.children_by_node.end() ? 0 : children_it->second.size();
  const std::size_t sink_count =
      sinks_it == tables.sinks_by_node.end() ? 0 : sinks_it->second.size();
  const std::size_t branch_count = child_count + sink_count;
  if (branch_count == 0) return 0;
  if (branch.getBranches().size() != 0) {
    throw std::runtime_error("source route branch already has children in net: " + route.net);
  }

  auto new_branches = branch.initBranches(static_cast<std::uint32_t>(branch_count));
  std::uint32_t out_index = 0;
  int emitted_edges = 0;

  if (children_it != tables.children_by_node.end()) {
    for (const RouteEdge& edge : children_it->second) {
      auto child = new_branches[out_index++];
      auto pip = child.initRouteSegment().initPip();
      pip.setTile(string_index(edge.tile, strings, string_to_index));
      pip.setWire0(string_index(edge.wire0, strings, string_to_index));
      pip.setWire1(string_index(edge.wire1, strings, string_to_index));
      pip.setIsFixed(false);
      pip.setForward(edge.forward);
      if (edge.attachment.has_value()) {
        // Validation above guarantees that the site is present and exactly
        // matches the sparse EndpointPip record.
        pip.setSite(string_index(*edge.site, strings, string_to_index));
      } else {
        // Select the union arm explicitly; do not rely on schema defaults.
        pip.setNoSite();
      }
      emitted_edges += 1 + insert_route_tree(child,
                                             edge.to,
                                             route,
                                             tables,
                                             stub_store,
                                             old_stubs,
                                             strings,
                                             string_to_index,
                                             ancestors);
    }
  }

  if (sinks_it != tables.sinks_by_node.end()) {
    for (const SitePinKey& sink : sinks_it->second) {
      auto child = new_branches[out_index++];
      copy_route_branch(old_stubs[consume_stub_branch(stub_store, sink, route.net)],
                        child);
    }
  }

  return emitted_edges;
}

void write_routed_phys(const std::filesystem::path& input_phys,
                       const std::filesystem::path& output_phys,
                       const std::unordered_map<std::string, NetRoute>& routes,
                       bool allow_unrouted_stubs) {
  const std::vector<std::uint8_t> bytes = read_gzip_or_plain_file(input_phys);
  std::vector<capnp::word> words = bytes_to_words(bytes);

  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords = std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(kj::arrayPtr(words.data(), words.size()),
                                       reader_options);

  capnp::MallocMessageBuilder builder;
  builder.setRoot(reader.getRoot<PhysicalNetlist::PhysNetlist>());
  auto netlist = builder.getRoot<PhysicalNetlist::PhysNetlist>();

  std::unordered_map<std::string, std::uint32_t> string_to_index;
  std::vector<std::string> strings =
      copy_string_list(netlist.getStrList(), string_to_index);

  std::unordered_map<std::string, bool> routed_seen;
  int total_pips = 0;
  int total_nets = 0;

  auto phys_nets = netlist.getPhysNets();
  for (std::uint32_t net_index = 0; net_index < phys_nets.size(); ++net_index) {
    auto net = phys_nets[net_index];
    const std::string net_name = phys_string_at(strings, net.getName());
    const auto route_it = routes.find(net_name);
    if (route_it == routes.end()) continue;

    const NetRoute& route = route_it->second;
    const bool reached_any_sink = has_reached_sink(route);
    if (!reached_any_sink) {
      if (!route.edges.empty()) {
        throw std::runtime_error("route has PIP edges but no reached sinks: " +
                                 net_name);
      }
      if (!allow_unrouted_stubs) {
        throw std::runtime_error("unrouted route has no reached sinks: " +
                                 net_name);
      }
      routed_seen.emplace(net_name, true);
      continue;
    }
    if (!route.routed && allow_unrouted_stubs &&
        !top_level_stubs_are_site_pins(net.getStubs())) {
      routed_seen.emplace(net_name, true);
      continue;
    }

    RouteTables tables = build_route_tables(route);

    auto old_stubs_orphan = net.disownStubs();
    auto old_stubs = old_stubs_orphan.get();
    StubBranchStore stub_store =
        snapshot_top_level_stubs(old_stubs, strings, net_name);

    std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
        source_branches;
    auto sources = net.getSources();
    for (std::uint32_t i = 0; i < sources.size(); ++i) {
      collect_site_pin_branches(sources[i], strings, source_branches);
    }

    int emitted_edges = 0;
    std::set<int> emitted_source_nodes;
    for (const auto& [source_key, source_branch] : source_branches) {
      const auto source_node_it = tables.source_node_by_pin.find(source_key);
      if (source_node_it == tables.source_node_by_pin.end()) continue;
      const int source_node = source_node_it->second;
      if (tables.children_by_node.find(source_node) == tables.children_by_node.end() &&
          tables.sinks_by_node.find(source_node) == tables.sinks_by_node.end()) {
        continue;
      }
      // Alternate or duplicate source site pins can legally resolve to the
      // same routing node. Emit that node's tree from the first physical root
      // only; inserting it below every alias duplicates PIPs and consumes the
      // same sink stubs more than once.
      if (!emitted_source_nodes.insert(source_node).second) {
        continue;
      }
      emitted_edges += insert_route_tree(source_branch,
                                         source_node,
                                         route,
                                         tables,
                                         stub_store,
                                         old_stubs,
                                         strings,
                                         string_to_index,
                                         {});
    }

    if (emitted_edges != tables.edge_count) {
      std::ostringstream out;
      out << "emitted " << emitted_edges << " PIPs for " << net_name
          << " but route contains " << tables.edge_count;
      throw std::runtime_error(out.str());
    }
    const std::size_t expected_reached_stubs =
        static_cast<std::size_t>(std::count_if(
            route.sinks.begin(), route.sinks.end(),
            [](const RouteSitePin& sink) {
              return sink.reached;
            }));
    const std::size_t consumed_reached_stubs =
        static_cast<std::size_t>(std::count_if(
            stub_store.stubs.begin(), stub_store.stubs.end(),
            [](const StoredStubBranch& stub) { return stub.consumed; }));
    if (consumed_reached_stubs != expected_reached_stubs) {
      throw std::runtime_error(
          "one or more reached sinks were not attached to a routed source in " +
          net_name);
    }
    std::size_t remaining_stub_count = 0;
    for (const StoredStubBranch& stub : stub_store.stubs) {
      if (!stub.consumed) {
        if (!allow_unrouted_stubs) {
          throw std::runtime_error("unrouted stub remains in routed net: " + net_name);
        }
        ++remaining_stub_count;
      }
    }

    auto new_stubs = net.initStubs(static_cast<std::uint32_t>(remaining_stub_count));
    std::uint32_t stub_index = 0;
    for (const StoredStubBranch& stub : stub_store.stubs) {
      if (stub.consumed) continue;
      copy_route_branch(old_stubs[stub.branch_index], new_stubs[stub_index++]);
    }
    routed_seen.emplace(net_name, true);
    total_pips += emitted_edges;
    ++total_nets;
  }

  for (const auto& [net, route] : routes) {
    (void)route;
    if (routed_seen.find(net) == routed_seen.end()) {
      throw std::runtime_error("route net was not found in PhysicalNetlist: " + net);
    }
  }

  auto new_str_list = netlist.initStrList(static_cast<std::uint32_t>(strings.size()));
  for (std::uint32_t i = 0; i < strings.size(); ++i) {
    new_str_list.set(i, strings[i]);
  }

  if (output_phys.has_parent_path()) {
    std::filesystem::create_directories(output_phys.parent_path());
  }
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(builder);
  write_gzip_file(output_phys, flat.asBytes());
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys> "
         "[--allow-unrouted-stubs]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "-h" ||
                      std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }
    bool allow_unrouted_stubs = false;
    if (argc == 6 && std::string(argv[5]) == "--allow-unrouted-stubs") {
      allow_unrouted_stubs = true;
    } else if (argc != 5) {
      print_usage(argv[0]);
      return 1;
    }

    const std::filesystem::path input_phys = argv[1];
    const std::filesystem::path metadata_path = argv[2];
    const std::filesystem::path routes_path = argv[3];
    const std::filesystem::path output_phys = argv[4];

    const routing::interchange::InterchangePublicationSnapshot
        publication_snapshot =
            routing::interchange::snapshot_interchange_publication(
                metadata_path);
    RoutingMetadataSummary metadata = load_metadata_summary(metadata_path);
    std::unordered_map<std::string, NetRoute> routes = load_routes_jsonl(routes_path);
    routing::interchange::verify_interchange_publication(
        metadata_path, publication_snapshot);
    for (const auto& [net, route] : routes) {
      (void)net;
      routing::interchange::require_matching_interchange_pair_ids(
          route.artifact_pair_id, metadata.artifact_pair_id,
          publication_snapshot.generation);
    }
    validate_routes_against_metadata(routes, metadata);
    write_routed_phys(input_phys, output_phys, routes, allow_unrouted_stubs);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
