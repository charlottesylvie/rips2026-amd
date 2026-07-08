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
//     Routing/routes_to_phys.cpp \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o routes_to_phys

#include "PhysicalNetlist.capnp.h"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
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
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t kInvalidRouteNode =
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
};

struct RouteEdge {
  int from = -1;
  int to = -1;
  int csr_edge = -1;
  std::string tile;
  std::string wire0;
  std::string wire1;
  bool forward = true;
};

struct NetRoute {
  std::string net;
  bool routed = false;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
  std::vector<RouteEdge> edges;
};

struct MetadataRouteRequest {
  std::string net;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
};

struct RoutingMetadataSummary {
  std::vector<std::string> strings;
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
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    return std::stod(text_.substr(begin, pos_ - begin));
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

void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  constexpr std::uint64_t kChunk = 1 << 20;
  std::array<char, kChunk> buffer{};
  while (count != 0) {
    const std::uint64_t take = std::min<std::uint64_t>(count, buffer.size());
    in.read(buffer.data(), static_cast<std::streamsize>(take));
    if (!in) throw std::runtime_error(std::string("failed while skipping ") + name);
    count -= take;
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

std::string string_at(const RoutingMetadataSummary& metadata, std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    std::ostringstream out;
    out << "<bad-string-" << index << ">";
    return out.str();
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
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "string count");
  const std::uint64_t node_count = read_u64(in, "node count");
  const std::uint64_t edge_attr_count = read_u64(in, "edge attr count");
  const std::uint64_t pip_data_count = read_u64(in, "pip data count");
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
  metadata.strings.reserve(static_cast<std::size_t>(string_count));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_metadata_string(in));
  }

  skip_bytes(in, node_count * sizeof(std::uint64_t), "node ids");
  skip_bytes(in, edge_attr_count * 2 * sizeof(std::uint64_t), "edge attrs");
  skip_bytes(in, pip_data_count * 3 * sizeof(std::uint64_t), "pip data");
  skip_bytes(in, site_pin_attr_count * 3 * sizeof(std::uint64_t), "site pin attrs");

  metadata.route_requests.reserve(static_cast<std::size_t>(route_request_count));
  for (std::uint64_t i = 0; i < route_request_count; ++i) {
    MetadataRouteRequest request;
    request.net = string_at(metadata, read_u64(in, "route request net"));
    (void)read_u64(in, "route request logical net");

    const std::uint64_t source_count = read_u64(in, "source count");
    request.sources.reserve(static_cast<std::size_t>(source_count));
    for (std::uint64_t s = 0; s < source_count; ++s) {
      RouteSitePin source;
      source.node = read_route_node(in, "source node");
      source.site = string_at(metadata, read_u64(in, "source site"));
      source.pin = string_at(metadata, read_u64(in, "source pin"));
      request.sources.push_back(std::move(source));
    }

    const std::uint64_t sink_count = read_u64(in, "sink count");
    request.sinks.reserve(static_cast<std::size_t>(sink_count));
    for (std::uint64_t s = 0; s < sink_count; ++s) {
      RouteSitePin sink;
      sink.node = read_route_node(in, "sink node");
      sink.site = string_at(metadata, read_u64(in, "sink site"));
      sink.pin = string_at(metadata, read_u64(in, "sink pin"));
      request.sinks.push_back(std::move(sink));
    }
    metadata.route_requests.push_back(std::move(request));
  }

  skip_bytes(in, logical_cell_count * 3 * sizeof(std::uint64_t), "logical cells");
  skip_bytes(in, logical_net_count * 4 * sizeof(std::uint64_t), "logical nets");
  skip_bytes(in, logical_port_instance_count * 7 * sizeof(std::uint64_t),
             "logical port instances");
  skip_bytes(in, blocked_node_count * sizeof(std::uint64_t), "blocked nodes");
  skip_bytes(in, sink_stop_node_count * sizeof(std::uint64_t), "sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "physical netlist bytes");
  skip_bytes(in, logical_netlist_byte_count, "logical netlist bytes");

  return metadata;
}

int json_int(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
  return static_cast<int>(found->second.as_number(key));
}

std::string json_string(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
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
    edge.csr_edge = json_int(edge_object, "csr_edge");
    edge.tile = json_string(edge_object, "tile");
    edge.wire0 = json_string(edge_object, "wire0");
    edge.wire1 = json_string(edge_object, "wire1");
    edge.forward = json_bool(edge_object, "forward", true);
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
    requests_by_net.emplace(request.net, &request);
  }

  for (const auto& [net, route] : routes) {
    const auto found = requests_by_net.find(net);
    if (found == requests_by_net.end()) {
      throw std::runtime_error("route file contains net not present in metadata: " + net);
    }
    const MetadataRouteRequest& request = *found->second;
    if (route.sinks.size() != request.sinks.size()) {
      std::ostringstream out;
      out << "route sink count for " << net << " is " << route.sinks.size()
          << " but metadata expects " << request.sinks.size();
      throw std::runtime_error(out.str());
    }
  }
}

std::vector<std::uint8_t> read_gzip_or_plain_file(const std::filesystem::path& path) {
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (!file) throw std::runtime_error("could not open input file: " + path.string());

  std::vector<std::uint8_t> bytes;
  std::array<std::uint8_t, 1 << 20> buffer{};
  while (true) {
    const int read_count =
        gzread(file, buffer.data(), static_cast<unsigned int>(buffer.size()));
    if (read_count > 0) {
      bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + read_count);
      continue;
    }
    if (read_count == 0) break;

    int zlib_error = 0;
    const char* message = gzerror(file, &zlib_error);
    gzclose(file);
    throw std::runtime_error("failed while reading " + path.string() + ": " +
                             (message ? message : "zlib error"));
  }
  gzclose(file);
  if (bytes.empty()) throw std::runtime_error("input file is empty: " + path.string());
  return bytes;
}

std::vector<capnp::word> bytes_to_words(const std::vector<std::uint8_t>& bytes) {
  const std::size_t word_count = (bytes.size() + sizeof(capnp::word) - 1) /
                                 sizeof(capnp::word);
  std::vector<capnp::word> words(word_count);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

void write_gzip_file(const std::filesystem::path& path,
                     kj::ArrayPtr<const kj::byte> bytes) {
  gzFile file = gzopen(path.string().c_str(), "wb6");
  if (!file) throw std::runtime_error("could not open output file: " + path.string());
  const int written =
      gzwrite(file, bytes.begin(), static_cast<unsigned int>(bytes.size()));
  if (written != static_cast<int>(bytes.size())) {
    int zlib_error = 0;
    const char* message = gzerror(file, &zlib_error);
    gzclose(file);
    throw std::runtime_error("failed while writing " + path.string() + ": " +
                             (message ? message : "zlib error"));
  }
  gzclose(file);
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
    if (!tables.source_node_by_pin.emplace(key, source.node).second) {
      throw std::runtime_error("duplicate source site pin in route: " + route.net);
    }
  }
  for (const RouteSitePin& sink : route.sinks) {
    if (!sink.reached || sink.node < 0) continue;
    tables.sinks_by_node[sink.node].push_back({sink.site, sink.pin});
  }
  return tables;
}

int insert_route_tree(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    int node,
    const NetRoute& route,
    const RouteTables& tables,
    std::map<SitePinKey, int>& remaining_stub_counts,
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
      emitted_edges += 1 + insert_route_tree(child,
                                             edge.to,
                                             route,
                                             tables,
                                             remaining_stub_counts,
                                             strings,
                                             string_to_index,
                                             ancestors);
    }
  }

  if (sinks_it != tables.sinks_by_node.end()) {
    for (const SitePinKey& sink : sinks_it->second) {
      auto found = remaining_stub_counts.find(sink);
      if (found == remaining_stub_counts.end() || found->second <= 0) {
        throw std::runtime_error("routed sink was not present as a stub in net: " + route.net);
      }
      --found->second;

      auto child = new_branches[out_index++];
      auto site_pin = child.initRouteSegment().initSitePin();
      site_pin.setSite(string_index(sink.site, strings, string_to_index));
      site_pin.setPin(string_index(sink.pin, strings, string_to_index));
      child.initBranches(0);
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
    RouteTables tables = build_route_tables(route);

    std::map<SitePinKey, int> stub_counts;
    auto stubs = net.getStubs();
    for (std::uint32_t i = 0; i < stubs.size(); ++i) {
      std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
          stub_pins;
      collect_site_pin_branches(stubs[i], strings, stub_pins);
      for (const auto& [key, branch] : stub_pins) {
        (void)branch;
        ++stub_counts[key];
      }
    }

    std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
        source_branches;
    auto sources = net.getSources();
    for (std::uint32_t i = 0; i < sources.size(); ++i) {
      collect_site_pin_branches(sources[i], strings, source_branches);
    }

    int emitted_edges = 0;
    for (const auto& [source_key, source_branch] : source_branches) {
      const auto source_node_it = tables.source_node_by_pin.find(source_key);
      if (source_node_it == tables.source_node_by_pin.end()) continue;
      const int source_node = source_node_it->second;
      if (tables.children_by_node.find(source_node) == tables.children_by_node.end() &&
          tables.sinks_by_node.find(source_node) == tables.sinks_by_node.end()) {
        continue;
      }
      emitted_edges += insert_route_tree(source_branch,
                                         source_node,
                                         route,
                                         tables,
                                         stub_counts,
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
    std::size_t remaining_stub_count = 0;
    for (const auto& [stub, count] : stub_counts) {
      if (count != 0) {
        if (!allow_unrouted_stubs) {
          throw std::runtime_error("unrouted stub remains in routed net: " + net_name);
        }
        remaining_stub_count += static_cast<std::size_t>(count);
      }
    }

    auto new_stubs = net.initStubs(static_cast<std::uint32_t>(remaining_stub_count));
    std::uint32_t stub_index = 0;
    for (const auto& [stub, count] : stub_counts) {
      for (int i = 0; i < count; ++i) {
        auto branch = new_stubs[stub_index++];
        auto site_pin = branch.initRouteSegment().initSitePin();
        site_pin.setSite(string_index(stub.site, strings, string_to_index));
        site_pin.setPin(string_index(stub.pin, strings, string_to_index));
        branch.initBranches(0);
      }
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

    RoutingMetadataSummary metadata = load_metadata_summary(metadata_path);
    std::unordered_map<std::string, NetRoute> routes = load_routes_jsonl(routes_path);
    validate_routes_against_metadata(routes, metadata);
    write_routed_phys(input_phys, output_phys, routes, allow_unrouted_stubs);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
