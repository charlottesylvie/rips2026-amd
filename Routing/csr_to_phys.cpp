// Reconstruct a routed FPGA Interchange PhysicalNetlist (.phys) from:
//   1. a CSR graph (.csrbin),
//   2. the matching RIPSIFM1 metadata sidecar produced by
//      Routing/interchange_to_csr.cpp, and
//   3. a text file describing routed CSR node paths.
//
// The tool expects one routed path per line:
//
//   net=<physical-net-name> source=<csr-node> sink=<csr-node> \
//   path=<n0>,<n1>,...,<nk>
//
// Example:
//
//   net=net_42 source=101 sink=887 path=101,220,404,700,887
//
// Each path must begin at the given source node, end at the given sink node,
// and every consecutive (u, v) pair must exist as a directed routing edge in
// the CSR graph. The metadata sidecar supplies the tile / wire / direction
// attributes needed to turn each CSR edge back into a PhysPIP.
//
// Compile after generating the FPGA Interchange C++ schema files:
//
//   g++ -std=c++17 -O3 \
//     -I<generated-schema-dir> \
//     Routing/csr_to_phys.cpp \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     <generated-schema-dir>/References.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o csr_to_phys
//
// Example use:
//
//   ./csr_to_phys \
//     HIP_kernel/bellman_ford/data/design.csrbin \
//     HIP_kernel/bellman_ford/data/design.csrbin.ifmeta.bin \
//     routes.txt \
//     design_routed.phys

#include "PhysicalNetlist.capnp.h"

#include <capnp/message.h>
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
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t kExpectedMetadataVersion = 2;
constexpr std::uint64_t kIncomingEdgeOrientation = 1;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();

using NodeId = std::int32_t;

struct Options {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path routes_path;
  std::filesystem::path output_phys_path;
  bool gzip_output = false;
};

struct HostCsr {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<float> values;
};

struct SitePinNode {
  NodeId node = -1;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

struct EdgeAttr {
  std::uint64_t tile_string = 0;
  std::uint64_t pip_data_index = 0;
};

struct PipData {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
};

struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct Metadata {
  std::vector<std::string> strings;
  std::vector<std::uint64_t> node_device_ids;
  std::vector<EdgeAttr> edge_attrs;
  std::vector<PipData> pip_data;
  std::vector<RouteRequest> route_requests;
  std::vector<std::uint8_t> physical_netlist_bytes;
  std::uint64_t device_path_string = 0;
  std::uint64_t physical_path_string = 0;
  std::uint64_t logical_path_string = 0;
  std::uint64_t logical_design_name_string = 0;
};

struct RoutedPath {
  std::string net_name;
  NodeId source = -1;
  NodeId sink = -1;
  std::vector<NodeId> nodes;
};

struct StringTable {
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint32_t> index_by_text;

  std::uint32_t intern(const std::string& text) {
    const auto found = index_by_text.find(text);
    if (found != index_by_text.end()) {
      return found->second;
    }
    const std::uint32_t index = static_cast<std::uint32_t>(strings.size());
    strings.push_back(text);
    index_by_text.emplace(strings.back(), index);
    return index;
  }
};

struct PathAssignment {
  const RouteRequest* request = nullptr;
  const SitePinNode* source = nullptr;
  const SitePinNode* sink = nullptr;
  std::vector<std::size_t> edge_indices;
  std::vector<NodeId> nodes;
};

struct SegmentKey {
  enum class Kind { kSitePin, kPip };

  Kind kind = Kind::kSitePin;
  std::string a;
  std::string b;
  std::string c;
  bool flag = false;

  bool operator==(const SegmentKey& other) const {
    return kind == other.kind && a == other.a && b == other.b && c == other.c &&
           flag == other.flag;
  }
};

struct SegmentKeyHash {
  std::size_t operator()(const SegmentKey& key) const {
    const std::size_t h0 = std::hash<int>{}(static_cast<int>(key.kind));
    const std::size_t h1 = std::hash<std::string>{}(key.a);
    const std::size_t h2 = std::hash<std::string>{}(key.b);
    const std::size_t h3 = std::hash<std::string>{}(key.c);
    const std::size_t h4 = std::hash<bool>{}(key.flag);
    return h0 ^ (h1 << 1) ^ (h2 << 7) ^ (h3 << 13) ^ (h4 << 19);
  }
};

struct RouteTreeNode {
  SegmentKey key;
  std::vector<std::unique_ptr<RouteTreeNode>> children;
  std::unordered_map<SegmentKey, std::size_t, SegmentKeyHash> child_index;
};

struct NetRouteForest {
  std::vector<std::unique_ptr<RouteTreeNode>> roots;
  std::unordered_map<NodeId, std::size_t> root_index_by_source_node;
};

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <graph.csrbin> <graph.ifmeta.bin> <routes.txt> <output.phys>"
         " [--gzip-output]\n\n"
      << "Route file format:\n"
      << "  net=<name> source=<csr-node> sink=<csr-node>"
         " path=<n0>,<n1>,...,<nk>\n";
}

Options parse_options(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    throw std::runtime_error("expected four positional arguments");
  }

  Options options;
  options.csr_path = argv[1];
  options.metadata_path = argv[2];
  options.routes_path = argv[3];
  options.output_phys_path = argv[4];

  for (int i = 5; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--gzip-output") {
      options.gzip_output = true;
      continue;
    }
    throw std::runtime_error("unknown option: " + arg);
  }

  return options;
}

std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open file: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to stat file: " + path.string());
  }
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  if (!in) {
    throw std::runtime_error("failed while reading file: " + path.string());
  }
  return bytes;
}

std::vector<capnp::word> bytes_to_words(const std::vector<std::uint8_t>& bytes) {
  const std::size_t word_size = sizeof(capnp::word);
  const std::size_t word_count = (bytes.size() + word_size - 1) / word_size;
  std::vector<capnp::word> words(word_count);
  if (!bytes.empty()) {
    std::memcpy(words.data(), bytes.data(), bytes.size());
  }
  return words;
}

std::string trim(std::string text) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && is_space(static_cast<unsigned char>(text.front()))) {
    text.erase(text.begin());
  }
  while (!text.empty() && is_space(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }
  return text;
}

std::vector<std::string> split_ws(const std::string& text) {
  std::istringstream in(text);
  std::vector<std::string> parts;
  std::string part;
  while (in >> part) {
    parts.push_back(part);
  }
  return parts;
}

std::vector<std::string> split(const std::string& text, char delim) {
  std::vector<std::string> parts;
  std::string current;
  std::istringstream in(text);
  while (std::getline(in, current, delim)) {
    parts.push_back(current);
  }
  return parts;
}

NodeId parse_node_id(const std::string& text, const char* field_name) {
  try {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    if (value < std::numeric_limits<NodeId>::min() ||
        value > std::numeric_limits<NodeId>::max()) {
      throw std::out_of_range("node id out of range");
    }
    return static_cast<NodeId>(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + field_name + ": " +
                             text);
  }
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& bytes, std::size_t& at,
                       const char* name) {
  if (at + sizeof(std::uint64_t) > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             name);
  }
  std::uint64_t value = 0;
  std::memcpy(&value, bytes.data() + at, sizeof(value));
  at += sizeof(value);
  return value;
}

template <typename T>
std::vector<T> read_array(const std::vector<std::uint8_t>& bytes,
                          std::size_t& at,
                          std::size_t count,
                          const char* name) {
  const std::size_t need = count * sizeof(T);
  if (at + need > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             name);
  }
  std::vector<T> out(count);
  if (need > 0) {
    std::memcpy(out.data(), bytes.data() + at, need);
  }
  at += need;
  return out;
}

std::string read_string(const std::vector<std::uint8_t>& bytes, std::size_t& at,
                        const char* name) {
  const std::uint64_t length = read_u64(bytes, at, name);
  if (length > bytes.size() - at) {
    throw std::runtime_error(std::string("invalid string length for ") + name);
  }
  std::string text;
  text.resize(static_cast<std::size_t>(length));
  if (length > 0) {
    std::memcpy(text.data(), bytes.data() + at, static_cast<std::size_t>(length));
  }
  at += static_cast<std::size_t>(length);
  return text;
}

std::vector<std::uint8_t> read_bytes(const std::vector<std::uint8_t>& bytes,
                                     std::size_t& at,
                                     std::size_t count,
                                     const char* name) {
  if (count > bytes.size() - at) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             name);
  }
  std::vector<std::uint8_t> out(count);
  if (count > 0) {
    std::memcpy(out.data(), bytes.data() + at, count);
  }
  at += count;
  return out;
}

std::uint64_t edge_key(NodeId from, NodeId to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

HostCsr load_csr(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_binary_file(path);
  std::size_t at = 0;

  if (bytes.size() < sizeof(CSR_MAGIC)) {
    throw std::runtime_error("CSR file is too small: " + path.string());
  }
  if (std::memcmp(bytes.data(), CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error("CSR magic mismatch: " + path.string());
  }
  at += sizeof(CSR_MAGIC);

  const std::uint64_t version = read_u64(bytes, at, "csr version");
  const std::uint64_t orientation = read_u64(bytes, at, "csr orientation");
  if (version != 1) {
    throw std::runtime_error("unsupported CSR version");
  }
  if (orientation != kIncomingEdgeOrientation) {
    throw std::runtime_error("CSR orientation is not incoming-edge");
  }

  HostCsr csr;
  csr.rows = static_cast<std::int64_t>(read_u64(bytes, at, "rows"));
  csr.cols = static_cast<std::int64_t>(read_u64(bytes, at, "cols"));
  csr.declared_edges = read_u64(bytes, at, "declared edges");
  csr.loaded_edges = read_u64(bytes, at, "loaded edges");
  const std::uint64_t nnz = read_u64(bytes, at, "nnz");
  const std::size_t rowptr_count =
      static_cast<std::size_t>(read_u64(bytes, at, "rowptr count"));
  const std::size_t colind_count =
      static_cast<std::size_t>(read_u64(bytes, at, "colind count"));
  const std::size_t values_count =
      static_cast<std::size_t>(read_u64(bytes, at, "values count"));

  csr.rowptr = read_array<std::int64_t>(bytes, at, rowptr_count, "rowptr");
  csr.colind = read_array<std::int32_t>(bytes, at, colind_count, "colind");
  csr.values = read_array<float>(bytes, at, values_count, "values");

  if (csr.rowptr.size() != static_cast<std::size_t>(csr.rows + 1)) {
    throw std::runtime_error("CSR rowptr size mismatch");
  }
  if (csr.colind.size() != static_cast<std::size_t>(nnz) ||
      csr.values.size() != static_cast<std::size_t>(nnz)) {
    throw std::runtime_error("CSR nnz count mismatch");
  }
  return csr;
}

Metadata load_metadata(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_binary_file(path);
  std::size_t at = 0;

  if (bytes.size() < sizeof(METADATA_MAGIC)) {
    throw std::runtime_error("metadata file is too small: " + path.string());
  }
  if (std::memcmp(bytes.data(), METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("metadata magic mismatch: " + path.string());
  }
  at += sizeof(METADATA_MAGIC);

  const std::uint64_t version = read_u64(bytes, at, "metadata version");
  const std::uint64_t orientation = read_u64(bytes, at, "metadata orientation");
  if (version != kExpectedMetadataVersion) {
    throw std::runtime_error("unsupported metadata version");
  }
  if (orientation != kIncomingEdgeOrientation) {
    throw std::runtime_error("metadata orientation is not incoming-edge");
  }

  const std::size_t string_count =
      static_cast<std::size_t>(read_u64(bytes, at, "string count"));
  const std::size_t node_count =
      static_cast<std::size_t>(read_u64(bytes, at, "node count"));
  const std::size_t edge_attr_count =
      static_cast<std::size_t>(read_u64(bytes, at, "edge attr count"));
  const std::size_t pip_data_count =
      static_cast<std::size_t>(read_u64(bytes, at, "pip data count"));
  const std::size_t site_pin_attr_count =
      static_cast<std::size_t>(read_u64(bytes, at, "site pin attr count"));
  const std::size_t route_request_count =
      static_cast<std::size_t>(read_u64(bytes, at, "route request count"));
  const std::size_t blocked_node_count =
      static_cast<std::size_t>(read_u64(bytes, at, "blocked node count"));
  const std::size_t sink_stop_node_count =
      static_cast<std::size_t>(read_u64(bytes, at, "sink stop node count"));
  const std::size_t logical_cell_count =
      static_cast<std::size_t>(read_u64(bytes, at, "logical cell count"));
  const std::size_t logical_net_count =
      static_cast<std::size_t>(read_u64(bytes, at, "logical net count"));
  const std::size_t logical_port_instance_count = static_cast<std::size_t>(
      read_u64(bytes, at, "logical port instance count"));
  const std::size_t physical_netlist_byte_count = static_cast<std::size_t>(
      read_u64(bytes, at, "physical netlist byte count"));
  const std::size_t logical_netlist_byte_count = static_cast<std::size_t>(
      read_u64(bytes, at, "logical netlist byte count"));

  Metadata meta;
  meta.device_path_string = read_u64(bytes, at, "device path string");
  meta.physical_path_string = read_u64(bytes, at, "physical path string");
  meta.logical_path_string = read_u64(bytes, at, "logical path string");
  meta.logical_design_name_string =
      read_u64(bytes, at, "logical design name string");

  meta.strings.reserve(string_count);
  for (std::size_t i = 0; i < string_count; ++i) {
    meta.strings.push_back(read_string(bytes, at, "metadata string"));
  }

  meta.node_device_ids =
      read_array<std::uint64_t>(bytes, at, node_count, "device node ids");

  meta.edge_attrs.reserve(edge_attr_count);
  for (std::size_t i = 0; i < edge_attr_count; ++i) {
    EdgeAttr attr;
    attr.tile_string = read_u64(bytes, at, "edge tile string");
    attr.pip_data_index = read_u64(bytes, at, "edge pip data index");
    meta.edge_attrs.push_back(attr);
  }

  meta.pip_data.reserve(pip_data_count);
  for (std::size_t i = 0; i < pip_data_count; ++i) {
    PipData data;
    data.wire0_string = read_u64(bytes, at, "pip wire0 string");
    data.wire1_string = read_u64(bytes, at, "pip wire1 string");
    data.forward = read_u64(bytes, at, "pip forward flag") != 0;
    meta.pip_data.push_back(data);
  }

  for (std::size_t i = 0; i < site_pin_attr_count; ++i) {
    (void)read_u64(bytes, at, "site pin attr node");
    (void)read_u64(bytes, at, "site pin attr site");
    (void)read_u64(bytes, at, "site pin attr pin");
  }

  meta.route_requests.reserve(route_request_count);
  for (std::size_t i = 0; i < route_request_count; ++i) {
    RouteRequest request;
    request.net_string = read_u64(bytes, at, "route request net string");
    request.logical_net_index = read_u64(bytes, at, "route request logical net");

    const std::size_t source_count =
        static_cast<std::size_t>(read_u64(bytes, at, "route request source count"));
    request.sources.reserve(source_count);
    for (std::size_t s = 0; s < source_count; ++s) {
      SitePinNode source;
      source.node = static_cast<NodeId>(
          read_u64(bytes, at, "route request source node"));
      source.site_string = read_u64(bytes, at, "route request source site");
      source.pin_string = read_u64(bytes, at, "route request source pin");
      request.sources.push_back(source);
    }

    const std::size_t sink_count =
        static_cast<std::size_t>(read_u64(bytes, at, "route request sink count"));
    request.sinks.reserve(sink_count);
    for (std::size_t s = 0; s < sink_count; ++s) {
      SitePinNode sink;
      sink.node = static_cast<NodeId>(
          read_u64(bytes, at, "route request sink node"));
      sink.site_string = read_u64(bytes, at, "route request sink site");
      sink.pin_string = read_u64(bytes, at, "route request sink pin");
      request.sinks.push_back(sink);
    }

    meta.route_requests.push_back(std::move(request));
  }

  for (std::size_t i = 0; i < logical_cell_count; ++i) {
    (void)read_u64(bytes, at, "logical cell declaration name");
    (void)read_u64(bytes, at, "logical cell net begin");
    (void)read_u64(bytes, at, "logical cell net count");
  }
  for (std::size_t i = 0; i < logical_net_count; ++i) {
    (void)read_u64(bytes, at, "logical net name");
    (void)read_u64(bytes, at, "logical net cell index");
    (void)read_u64(bytes, at, "logical net port instance begin");
    (void)read_u64(bytes, at, "logical net port instance count");
  }
  for (std::size_t i = 0; i < logical_port_instance_count; ++i) {
    (void)read_u64(bytes, at, "logical port name");
    (void)read_u64(bytes, at, "logical instance name");
    (void)read_u64(bytes, at, "logical port index");
    (void)read_u64(bytes, at, "logical instance index");
    (void)read_u64(bytes, at, "logical bus index");
    (void)read_u64(bytes, at, "logical has bus index");
    (void)read_u64(bytes, at, "logical is external port");
  }

  (void)read_array<std::uint64_t>(bytes, at, blocked_node_count, "blocked nodes");
  (void)read_array<std::uint64_t>(bytes, at, sink_stop_node_count,
                                  "sink stop nodes");
  meta.physical_netlist_bytes = read_bytes(bytes, at, physical_netlist_byte_count,
                                           "physical netlist bytes");
  (void)read_bytes(bytes, at, logical_netlist_byte_count,
                   "logical netlist bytes");

  if (at != bytes.size()) {
    throw std::runtime_error("metadata file contains trailing bytes");
  }

  return meta;
}

std::string metadata_string(const Metadata& meta, std::uint64_t index,
                            const char* what) {
  if (index >= meta.strings.size()) {
    throw std::runtime_error(std::string("metadata string index out of range for ") +
                             what);
  }
  return meta.strings[static_cast<std::size_t>(index)];
}

std::vector<RoutedPath> load_routes(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("could not open routes file: " + path.string());
  }

  std::vector<RoutedPath> routes;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const std::string cleaned = trim(line);
    if (cleaned.empty() || cleaned[0] == '#') {
      continue;
    }

    RoutedPath route;
    bool have_net = false;
    bool have_source = false;
    bool have_sink = false;
    bool have_path = false;

    for (const std::string& token : split_ws(cleaned)) {
      const std::size_t eq = token.find('=');
      if (eq == std::string::npos) {
        throw std::runtime_error("routes line " + std::to_string(line_number) +
                                 " has malformed token: " + token);
      }
      const std::string key = token.substr(0, eq);
      const std::string value = token.substr(eq + 1);
      if (key == "net") {
        route.net_name = value;
        have_net = true;
      } else if (key == "source") {
        route.source = parse_node_id(value, "source");
        have_source = true;
      } else if (key == "sink") {
        route.sink = parse_node_id(value, "sink");
        have_sink = true;
      } else if (key == "path") {
        const std::vector<std::string> nodes = split(value, ',');
        if (nodes.empty()) {
          throw std::runtime_error("routes line " + std::to_string(line_number) +
                                   " has empty path");
        }
        route.nodes.reserve(nodes.size());
        for (const std::string& node_text : nodes) {
          route.nodes.push_back(parse_node_id(node_text, "path node"));
        }
        have_path = true;
      } else {
        throw std::runtime_error("routes line " + std::to_string(line_number) +
                                 " has unknown key: " + key);
      }
    }

    if (!have_net || !have_source || !have_sink || !have_path) {
      throw std::runtime_error("routes line " + std::to_string(line_number) +
                               " is missing one of net/source/sink/path");
    }
    if (route.nodes.front() != route.source || route.nodes.back() != route.sink) {
      throw std::runtime_error("routes line " + std::to_string(line_number) +
                               " path endpoints do not match source/sink");
    }
    routes.push_back(std::move(route));
  }

  return routes;
}

const SitePinNode* find_site_pin_by_node(const std::vector<SitePinNode>& pins,
                                         NodeId node) {
  for (const SitePinNode& pin : pins) {
    if (pin.node == node) {
      return &pin;
    }
  }
  return nullptr;
}

std::unordered_map<std::string, const RouteRequest*> build_request_map(
    const Metadata& meta) {
  std::unordered_map<std::string, const RouteRequest*> by_name;
  for (const RouteRequest& request : meta.route_requests) {
    const std::string name = metadata_string(meta, request.net_string, "net name");
    by_name.emplace(name, &request);
  }
  return by_name;
}

std::unordered_map<std::uint64_t, std::size_t> build_edge_index(
    const HostCsr& csr, const Metadata& meta) {
  if (meta.edge_attrs.size() != csr.colind.size()) {
    throw std::runtime_error("metadata edge attributes do not align with CSR nnz");
  }

  std::unordered_map<std::uint64_t, std::size_t> edge_index_by_key;
  edge_index_by_key.reserve(meta.edge_attrs.size());
  for (NodeId to = 0; to < csr.rows; ++to) {
    const std::int64_t begin = csr.rowptr[static_cast<std::size_t>(to)];
    const std::int64_t end = csr.rowptr[static_cast<std::size_t>(to + 1)];
    for (std::int64_t p = begin; p < end; ++p) {
      const NodeId from = csr.colind[static_cast<std::size_t>(p)];
      const std::uint64_t key = edge_key(from, to);
      const auto [it, inserted] =
          edge_index_by_key.emplace(key, static_cast<std::size_t>(p));
      if (!inserted) {
        throw std::runtime_error(
            "parallel CSR edges detected for the same (from,to) pair; "
            "path node lists are ambiguous for reconstruction");
      }
    }
  }
  return edge_index_by_key;
}

std::unordered_map<std::string, std::vector<PathAssignment>> validate_routes(
    const std::vector<RoutedPath>& routes,
    const Metadata& meta,
    const std::unordered_map<std::string, const RouteRequest*>& request_by_name,
    const std::unordered_map<std::uint64_t, std::size_t>& edge_index_by_key) {
  std::unordered_map<std::string, std::vector<PathAssignment>> assignments;

  for (const RoutedPath& route : routes) {
    const auto request_it = request_by_name.find(route.net_name);
    if (request_it == request_by_name.end()) {
      throw std::runtime_error("route refers to net not present in metadata: " +
                               route.net_name);
    }

    const RouteRequest* request = request_it->second;
    const SitePinNode* source = find_site_pin_by_node(request->sources, route.source);
    const SitePinNode* sink = find_site_pin_by_node(request->sinks, route.sink);
    if (source == nullptr) {
      throw std::runtime_error("route source node is not a legal source for net " +
                               route.net_name);
    }
    if (sink == nullptr) {
      throw std::runtime_error("route sink node is not a legal sink for net " +
                               route.net_name);
    }
    if (route.nodes.size() < 2) {
      throw std::runtime_error("route path for net " + route.net_name +
                               " must contain at least two nodes");
    }

    PathAssignment assignment;
    assignment.request = request;
    assignment.source = source;
    assignment.sink = sink;
    assignment.nodes = route.nodes;
    assignment.edge_indices.reserve(route.nodes.size() - 1);

    for (std::size_t i = 0; i + 1 < route.nodes.size(); ++i) {
      const NodeId from = route.nodes[i];
      const NodeId to = route.nodes[i + 1];
      const auto edge_it = edge_index_by_key.find(edge_key(from, to));
      if (edge_it == edge_index_by_key.end()) {
        throw std::runtime_error("route path for net " + route.net_name +
                                 " uses a non-existent CSR edge " +
                                 std::to_string(from) + " -> " +
                                 std::to_string(to));
      }
      assignment.edge_indices.push_back(edge_it->second);
    }

    assignments[route.net_name].push_back(std::move(assignment));
  }

  for (const auto& [net_name, net_assignments] : assignments) {
    const RouteRequest* request = request_by_name.at(net_name);
    std::unordered_map<NodeId, std::size_t> sink_counts;
    for (const PathAssignment& assignment : net_assignments) {
      ++sink_counts[assignment.sink->node];
    }
    for (const SitePinNode& sink : request->sinks) {
      const auto found = sink_counts.find(sink.node);
      if (found == sink_counts.end()) {
        throw std::runtime_error("net " + net_name +
                                 " is missing a routed path for sink node " +
                                 std::to_string(sink.node));
      }
      if (found->second != 1) {
        throw std::runtime_error("net " + net_name +
                                 " has duplicate routed paths for sink node " +
                                 std::to_string(sink.node));
      }
    }
  }

  return assignments;
}

RouteTreeNode* add_child(RouteTreeNode& parent, SegmentKey key) {
  const auto found = parent.child_index.find(key);
  if (found != parent.child_index.end()) {
    return parent.children[found->second].get();
  }
  const std::size_t index = parent.children.size();
  parent.child_index.emplace(key, index);
  parent.children.push_back(std::make_unique<RouteTreeNode>());
  parent.children.back()->key = std::move(key);
  return parent.children.back().get();
}

NetRouteForest build_route_forest(
    const std::vector<PathAssignment>& assignments,
    const Metadata& meta) {
  NetRouteForest forest;

  for (const PathAssignment& assignment : assignments) {
    const auto root_it =
        forest.root_index_by_source_node.find(assignment.source->node);
    RouteTreeNode* current = nullptr;
    if (root_it == forest.root_index_by_source_node.end()) {
      const std::size_t root_index = forest.roots.size();
      forest.root_index_by_source_node.emplace(assignment.source->node, root_index);
      forest.roots.push_back(std::make_unique<RouteTreeNode>());
      RouteTreeNode& root = *forest.roots.back();
      root.key.kind = SegmentKey::Kind::kSitePin;
      root.key.a = metadata_string(meta, assignment.source->site_string,
                                   "source site");
      root.key.b = metadata_string(meta, assignment.source->pin_string,
                                   "source pin");
      current = &root;
    } else {
      current = forest.roots[root_it->second].get();
    }

    for (const std::size_t edge_index : assignment.edge_indices) {
      const EdgeAttr& attr = meta.edge_attrs[edge_index];
      if (attr.pip_data_index >= meta.pip_data.size()) {
        throw std::runtime_error("pip data index out of range while building tree");
      }
      const PipData& pip = meta.pip_data[static_cast<std::size_t>(attr.pip_data_index)];

      SegmentKey key;
      key.kind = SegmentKey::Kind::kPip;
      key.a = metadata_string(meta, attr.tile_string, "pip tile");
      key.b = metadata_string(meta, pip.wire0_string, "pip wire0");
      key.c = metadata_string(meta, pip.wire1_string, "pip wire1");
      key.flag = pip.forward;
      current = add_child(*current, std::move(key));
    }

    SegmentKey sink_key;
    sink_key.kind = SegmentKey::Kind::kSitePin;
    sink_key.a = metadata_string(meta, assignment.sink->site_string, "sink site");
    sink_key.b = metadata_string(meta, assignment.sink->pin_string, "sink pin");
    (void)add_child(*current, std::move(sink_key));
  }

  return forest;
}

void collect_required_route_strings(StringTable& out_strings,
                                    const Metadata& meta,
                                    const std::vector<PathAssignment>& assignments) {
  for (const PathAssignment& assignment : assignments) {
    out_strings.intern(metadata_string(meta, assignment.source->site_string, "source site"));
    out_strings.intern(metadata_string(meta, assignment.source->pin_string, "source pin"));
    out_strings.intern(metadata_string(meta, assignment.sink->site_string, "sink site"));
    out_strings.intern(metadata_string(meta, assignment.sink->pin_string, "sink pin"));
    for (const std::size_t edge_index : assignment.edge_indices) {
      const EdgeAttr& attr = meta.edge_attrs[edge_index];
      const PipData& pip = meta.pip_data[static_cast<std::size_t>(attr.pip_data_index)];
      out_strings.intern(metadata_string(meta, attr.tile_string, "pip tile"));
      out_strings.intern(metadata_string(meta, pip.wire0_string, "pip wire0"));
      out_strings.intern(metadata_string(meta, pip.wire1_string, "pip wire1"));
    }
  }
}

void copy_text_list(capnp::List<capnp::Text>::Builder out,
                    const std::vector<std::string>& strings) {
  for (std::uint32_t i = 0; i < out.size(); ++i) {
    out.set(i, strings[i]);
  }
}

void copy_route_branch(
    PhysicalNetlist::PhysNetlist::RouteBranch::Reader src,
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder dst,
    const std::vector<std::uint32_t>& old_string_to_new) {
  auto src_segment = src.getRouteSegment();
  auto dst_segment = dst.getRouteSegment();

  if (src_segment.isBelPin()) {
    const auto in = src_segment.getBelPin();
    auto out = dst_segment.initBelPin();
    out.setSite(old_string_to_new[in.getSite()]);
    out.setBel(old_string_to_new[in.getBel()]);
    out.setPin(old_string_to_new[in.getPin()]);
  } else if (src_segment.isSitePin()) {
    const auto in = src_segment.getSitePin();
    auto out = dst_segment.initSitePin();
    out.setSite(old_string_to_new[in.getSite()]);
    out.setPin(old_string_to_new[in.getPin()]);
  } else if (src_segment.isPip()) {
    const auto in = src_segment.getPip();
    auto out = dst_segment.initPip();
    out.setTile(old_string_to_new[in.getTile()]);
    out.setWire0(old_string_to_new[in.getWire0()]);
    out.setWire1(old_string_to_new[in.getWire1()]);
    out.setForward(in.getForward());
    out.setIsFixed(in.getIsFixed());
    if (in.isSite()) {
      out.setSite(old_string_to_new[in.getSite()]);
    } else {
      out.setNoSite();
    }
  } else if (src_segment.isSitePIP()) {
    const auto in = src_segment.getSitePIP();
    auto out = dst_segment.initSitePIP();
    out.setSite(old_string_to_new[in.getSite()]);
    out.setBel(old_string_to_new[in.getBel()]);
    out.setPin(old_string_to_new[in.getPin()]);
    out.setIsFixed(in.getIsFixed());
    if (in.isInverts()) {
      out.setInverts();
    } else {
      out.setIsInverting(in.getIsInverting());
    }
  } else {
    throw std::runtime_error("unknown route segment kind while copying route branch");
  }

  const auto src_children = src.getBranches();
  auto dst_children = dst.initBranches(src_children.size());
  for (std::uint32_t i = 0; i < src_children.size(); ++i) {
    copy_route_branch(src_children[i], dst_children[i], old_string_to_new);
  }
}

void copy_phys_node(PhysicalNetlist::PhysNetlist::PhysNode::Reader src,
                    PhysicalNetlist::PhysNetlist::PhysNode::Builder dst,
                    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setTile(old_string_to_new[src.getTile()]);
  dst.setWire(old_string_to_new[src.getWire()]);
  dst.setIsFixed(src.getIsFixed());
}

void copy_phys_net(
    PhysicalNetlist::PhysNetlist::PhysNet::Reader src,
    PhysicalNetlist::PhysNetlist::PhysNet::Builder dst,
    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setName(old_string_to_new[src.getName()]);
  dst.setType(src.getType());

  const auto src_sources = src.getSources();
  auto dst_sources = dst.initSources(src_sources.size());
  for (std::uint32_t i = 0; i < src_sources.size(); ++i) {
    copy_route_branch(src_sources[i], dst_sources[i], old_string_to_new);
  }

  const auto src_stubs = src.getStubs();
  auto dst_stubs = dst.initStubs(src_stubs.size());
  for (std::uint32_t i = 0; i < src_stubs.size(); ++i) {
    copy_route_branch(src_stubs[i], dst_stubs[i], old_string_to_new);
  }

  const auto src_stub_nodes = src.getStubNodes();
  auto dst_stub_nodes = dst.initStubNodes(src_stub_nodes.size());
  for (std::uint32_t i = 0; i < src_stub_nodes.size(); ++i) {
    copy_phys_node(src_stub_nodes[i], dst_stub_nodes[i], old_string_to_new);
  }
}

void copy_pin_mapping(
    PhysicalNetlist::PhysNetlist::PinMapping::Reader src,
    PhysicalNetlist::PhysNetlist::PinMapping::Builder dst,
    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setCellPin(old_string_to_new[src.getCellPin()]);
  dst.setBel(old_string_to_new[src.getBel()]);
  dst.setBelPin(old_string_to_new[src.getBelPin()]);
  dst.setIsFixed(src.getIsFixed());
  if (src.isMulti()) {
    dst.setMulti();
  } else {
    const auto in = src.getOtherCell();
    auto out = dst.initOtherCell();
    out.setMultiCell(old_string_to_new[in.getMultiCell()]);
    out.setMultiType(old_string_to_new[in.getMultiType()]);
  }
}

void copy_cell_placement(
    PhysicalNetlist::PhysNetlist::CellPlacement::Reader src,
    PhysicalNetlist::PhysNetlist::CellPlacement::Builder dst,
    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setCellName(old_string_to_new[src.getCellName()]);
  dst.setType(old_string_to_new[src.getType()]);
  dst.setSite(old_string_to_new[src.getSite()]);
  dst.setBel(old_string_to_new[src.getBel()]);

  const auto src_pin_map = src.getPinMap();
  auto dst_pin_map = dst.initPinMap(src_pin_map.size());
  for (std::uint32_t i = 0; i < src_pin_map.size(); ++i) {
    copy_pin_mapping(src_pin_map[i], dst_pin_map[i], old_string_to_new);
  }

  const auto src_other_bels = src.getOtherBels();
  auto dst_other_bels = dst.initOtherBels(src_other_bels.size());
  for (std::uint32_t i = 0; i < src_other_bels.size(); ++i) {
    dst_other_bels.set(i, old_string_to_new[src_other_bels[i]]);
  }

  dst.setIsBelFixed(src.getIsBelFixed());
  dst.setIsSiteFixed(src.getIsSiteFixed());
  dst.setAltSiteType(old_string_to_new[src.getAltSiteType()]);
}

void copy_phys_cell(PhysicalNetlist::PhysNetlist::PhysCell::Reader src,
                    PhysicalNetlist::PhysNetlist::PhysCell::Builder dst,
                    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setCellName(old_string_to_new[src.getCellName()]);
  dst.setPhysType(src.getPhysType());
}

void copy_site_instance(
    PhysicalNetlist::PhysNetlist::SiteInstance::Reader src,
    PhysicalNetlist::PhysNetlist::SiteInstance::Builder dst,
    const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setSite(old_string_to_new[src.getSite()]);
  dst.setType(old_string_to_new[src.getType()]);
}

void copy_property(PhysicalNetlist::PhysNetlist::Property::Reader src,
                   PhysicalNetlist::PhysNetlist::Property::Builder dst,
                   const std::vector<std::uint32_t>& old_string_to_new) {
  dst.setKey(old_string_to_new[src.getKey()]);
  dst.setValue(old_string_to_new[src.getValue()]);
}

void write_tree_branch(RouteTreeNode& node,
                       PhysicalNetlist::PhysNetlist::RouteBranch::Builder out,
                       StringTable& strings) {
  auto segment = out.getRouteSegment();
  if (node.key.kind == SegmentKey::Kind::kSitePin) {
    auto site_pin = segment.initSitePin();
    site_pin.setSite(strings.intern(node.key.a));
    site_pin.setPin(strings.intern(node.key.b));
  } else {
    auto pip = segment.initPip();
    pip.setTile(strings.intern(node.key.a));
    pip.setWire0(strings.intern(node.key.b));
    pip.setWire1(strings.intern(node.key.c));
    pip.setForward(node.key.flag);
    pip.setIsFixed(false);
    pip.setNoSite();
  }

  auto children = out.initBranches(node.children.size());
  for (std::uint32_t i = 0; i < node.children.size(); ++i) {
    write_tree_branch(*node.children[i], children[i], strings);
  }
}

void write_routed_net(
    PhysicalNetlist::PhysNetlist::PhysNet::Reader original,
    PhysicalNetlist::PhysNetlist::PhysNet::Builder out,
    StringTable& strings,
    std::uint32_t remapped_name,
    const NetRouteForest& forest) {
  out.setName(remapped_name);
  out.setType(original.getType());

  auto out_sources = out.initSources(forest.roots.size());
  for (std::uint32_t i = 0; i < forest.roots.size(); ++i) {
    write_tree_branch(*forest.roots[i], out_sources[i], strings);
  }

  out.initStubs(0);
  out.initStubNodes(0);
}

void write_output_file(const std::filesystem::path& path,
                       const kj::Array<capnp::word>& words,
                       bool gzip_output) {
  const auto bytes = words.asBytes();
  if (!gzip_output) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      throw std::runtime_error("could not open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes.begin()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
      throw std::runtime_error("failed while writing output file: " + path.string());
    }
    return;
  }

  gzFile file = gzopen(path.string().c_str(), "wb");
  if (!file) {
    throw std::runtime_error("could not open gzip output file: " + path.string());
  }
  const int written =
      gzwrite(file, bytes.begin(), static_cast<unsigned int>(bytes.size()));
  if (written == 0) {
    int zerr = 0;
    const char* msg = gzerror(file, &zerr);
    gzclose(file);
    throw std::runtime_error("failed while gz-writing output file: " +
                             std::string(msg ? msg : "zlib error"));
  }
  gzclose(file);
}

void rebuild_phys(const HostCsr& csr,
                  const Metadata& meta,
                  const std::unordered_map<std::string, std::vector<PathAssignment>>&
                      assignments_by_net,
                  const std::filesystem::path& output_path,
                  bool gzip_output) {
  (void)csr;
  std::vector<capnp::word> phys_words = bytes_to_words(meta.physical_netlist_bytes);
  capnp::ReaderOptions reader_options;
  reader_options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  reader_options.nestingLimit = 1 << 20;

  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(phys_words.data(), phys_words.size()), reader_options);
  const auto original = reader.getRoot<PhysicalNetlist::PhysNetlist>();
  const auto original_strings = original.getStrList();

  StringTable out_strings;
  std::vector<std::uint32_t> old_string_to_new(original_strings.size(), 0);
  for (std::uint32_t i = 0; i < original_strings.size(); ++i) {
    const std::string text(original_strings[i].cStr(), original_strings[i].size());
    old_string_to_new[i] = out_strings.intern(text);
  }
  for (const auto& [net_name, assignments] : assignments_by_net) {
    (void)net_name;
    collect_required_route_strings(out_strings, meta, assignments);
  }

  capnp::MallocMessageBuilder message;
  auto out = message.initRoot<PhysicalNetlist::PhysNetlist>();
  out.setPart(original.getPart());

  auto out_strings_list = out.initStrList(out_strings.strings.size());
  copy_text_list(out_strings_list, out_strings.strings);

  const auto in_placements = original.getPlacements();
  auto out_placements = out.initPlacements(in_placements.size());
  for (std::uint32_t i = 0; i < in_placements.size(); ++i) {
    copy_cell_placement(in_placements[i], out_placements[i], old_string_to_new);
  }

  const auto in_phys_nets = original.getPhysNets();
  auto out_phys_nets = out.initPhysNets(in_phys_nets.size());
  for (std::uint32_t i = 0; i < in_phys_nets.size(); ++i) {
    const auto in_net = in_phys_nets[i];
    const std::string net_name(original_strings[in_net.getName()].cStr(),
                               original_strings[in_net.getName()].size());
    const std::uint32_t remapped_name = old_string_to_new[in_net.getName()];

    const auto routed_it = assignments_by_net.find(net_name);
    if (routed_it == assignments_by_net.end()) {
      copy_phys_net(in_net, out_phys_nets[i], old_string_to_new);
      continue;
    }

    const NetRouteForest forest = build_route_forest(routed_it->second, meta);
    write_routed_net(in_net, out_phys_nets[i], out_strings, remapped_name, forest);
  }

  const auto in_phys_cells = original.getPhysCells();
  auto out_phys_cells = out.initPhysCells(in_phys_cells.size());
  for (std::uint32_t i = 0; i < in_phys_cells.size(); ++i) {
    copy_phys_cell(in_phys_cells[i], out_phys_cells[i], old_string_to_new);
  }

  const auto in_site_insts = original.getSiteInsts();
  auto out_site_insts = out.initSiteInsts(in_site_insts.size());
  for (std::uint32_t i = 0; i < in_site_insts.size(); ++i) {
    copy_site_instance(in_site_insts[i], out_site_insts[i], old_string_to_new);
  }

  const auto in_props = original.getProperties();
  auto out_props = out.initProperties(in_props.size());
  for (std::uint32_t i = 0; i < in_props.size(); ++i) {
    copy_property(in_props[i], out_props[i], old_string_to_new);
  }

  copy_phys_net(original.getNullNet(), out.initNullNet(), old_string_to_new);

  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  write_output_file(output_path, flat, gzip_output);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    const HostCsr csr = load_csr(options.csr_path);
    const Metadata meta = load_metadata(options.metadata_path);
    const std::vector<RoutedPath> routes = load_routes(options.routes_path);
    const auto request_by_name = build_request_map(meta);
    const auto edge_index_by_key = build_edge_index(csr, meta);
    const auto assignments_by_net =
        validate_routes(routes, meta, request_by_name, edge_index_by_key);

    rebuild_phys(csr, meta, assignments_by_net, options.output_phys_path,
                 options.gzip_output);

    std::cout << "csr: " << options.csr_path << "\n";
    std::cout << "metadata: " << options.metadata_path << "\n";
    std::cout << "routes: " << options.routes_path << "\n";
    std::cout << "wrote_phys: " << options.output_phys_path << "\n";
    std::cout << "routed_nets: " << assignments_by_net.size() << "\n";
    return 0;
  } catch (const std::exception& e) {
    print_usage(argv[0]);
    std::cerr << "\nerror: " << e.what() << "\n";
    return 1;
  }
}
