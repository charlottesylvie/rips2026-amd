#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t kExpectedCsrVersion = 1;
constexpr std::uint64_t kExpectedMetadataVersion = 2;
constexpr std::uint64_t kIncomingEdgeOrientation = 1;

using NodeId = std::int32_t;

struct Options {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path output_routes_path;
};

struct HostCsr {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::uint64_t declared_edges = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<float> values;
};

struct SitePinNode {
  NodeId node = -1;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = std::numeric_limits<std::uint64_t>::max();
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct Metadata {
  std::vector<std::string> strings;
  std::vector<RouteRequest> route_requests;
};

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program
            << " <graph.csrbin> <graph.ifmeta.bin> <output.routes.txt>\n";
}

Options parse_options(int argc, char** argv) {
  if (argc != 4) {
    throw std::runtime_error("expected three positional arguments");
  }

  Options options;
  options.csr_path = argv[1];
  options.metadata_path = argv[2];
  options.output_routes_path = argv[3];
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
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (!in) {
    throw std::runtime_error("failed while reading file: " + path.string());
  }
  return bytes;
}

std::string trim(const std::string& text) {
  std::size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }

  std::size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }

  return text.substr(begin, end - begin);
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& bytes,
                       std::size_t& at,
                       const char* field_name) {
  if (at + sizeof(std::uint64_t) > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             field_name);
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
                          const char* field_name) {
  const std::size_t total_bytes = count * sizeof(T);
  if (at + total_bytes > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             field_name);
  }

  std::vector<T> values(count);
  if (total_bytes != 0) {
    std::memcpy(values.data(), bytes.data() + at, total_bytes);
  }
  at += total_bytes;
  return values;
}

std::vector<std::uint8_t> read_bytes(const std::vector<std::uint8_t>& bytes,
                                     std::size_t& at,
                                     std::size_t count,
                                     const char* field_name) {
  if (at + count > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             field_name);
  }

  std::vector<std::uint8_t> out(count);
  if (count != 0) {
    std::memcpy(out.data(), bytes.data() + at, count);
  }
  at += count;
  return out;
}

std::string read_string(const std::vector<std::uint8_t>& bytes,
                        std::size_t& at,
                        const char* field_name) {
  const std::size_t length =
      static_cast<std::size_t>(read_u64(bytes, at, field_name));
  if (at + length > bytes.size()) {
    throw std::runtime_error(std::string("unexpected EOF while reading ") +
                             field_name);
  }

  const std::string text(reinterpret_cast<const char*>(bytes.data() + at),
                         length);
  at += length;
  return text;
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

  const std::uint64_t version = read_u64(bytes, at, "CSR version");
  const std::uint64_t orientation = read_u64(bytes, at, "CSR orientation");
  if (version != kExpectedCsrVersion) {
    throw std::runtime_error("unsupported CSR version");
  }
  if (orientation != kIncomingEdgeOrientation) {
    throw std::runtime_error("CSR orientation is not incoming-edge");
  }

  HostCsr csr;
  csr.rows = static_cast<std::int64_t>(read_u64(bytes, at, "CSR rows"));
  csr.cols = static_cast<std::int64_t>(read_u64(bytes, at, "CSR cols"));
  csr.declared_edges = read_u64(bytes, at, "CSR nnz");

  csr.rowptr = read_array<std::int64_t>(bytes,
                                        at,
                                        static_cast<std::size_t>(csr.rows + 1),
                                        "CSR rowptr");
  csr.colind = read_array<std::int32_t>(
      bytes, at, static_cast<std::size_t>(csr.declared_edges), "CSR colind");
  csr.values = read_array<float>(
      bytes, at, static_cast<std::size_t>(csr.declared_edges), "CSR values");

  if (csr.rowptr.size() != static_cast<std::size_t>(csr.rows + 1)) {
    throw std::runtime_error("CSR rowptr size mismatch");
  }
  if (csr.colind.size() != static_cast<std::size_t>(csr.declared_edges) ||
      csr.values.size() != static_cast<std::size_t>(csr.declared_edges)) {
    throw std::runtime_error("CSR nnz count mismatch");
  }
  if (at != bytes.size()) {
    throw std::runtime_error("CSR file contains trailing bytes");
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

  (void)read_u64(bytes, at, "device path string");
  (void)read_u64(bytes, at, "physical path string");
  (void)read_u64(bytes, at, "logical path string");
  (void)read_u64(bytes, at, "logical design name string");

  meta.strings.reserve(string_count);
  for (std::size_t i = 0; i < string_count; ++i) {
    meta.strings.push_back(read_string(bytes, at, "metadata string"));
  }

  (void)read_array<std::uint64_t>(bytes, at, node_count, "device node ids");

  for (std::size_t i = 0; i < edge_attr_count; ++i) {
    (void)read_u64(bytes, at, "edge tile string");
    (void)read_u64(bytes, at, "edge pip data index");
  }

  for (std::size_t i = 0; i < pip_data_count; ++i) {
    (void)read_u64(bytes, at, "pip wire0 string");
    (void)read_u64(bytes, at, "pip wire1 string");
    (void)read_u64(bytes, at, "pip forward flag");
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
  (void)read_bytes(bytes, at, physical_netlist_byte_count, "physical netlist bytes");
  (void)read_bytes(bytes, at, logical_netlist_byte_count, "logical netlist bytes");

  if (at != bytes.size()) {
    throw std::runtime_error("metadata file contains trailing bytes");
  }

  return meta;
}

std::string metadata_string(const Metadata& meta,
                            std::uint64_t index,
                            const char* what) {
  if (index >= meta.strings.size()) {
    throw std::runtime_error(std::string("metadata string index out of range for ") +
                             what);
  }
  return meta.strings[static_cast<std::size_t>(index)];
}

std::vector<std::vector<NodeId>> build_outgoing_adjacency(const HostCsr& csr) {
  std::vector<std::vector<NodeId>> outgoing(
      static_cast<std::size_t>(csr.rows));

  for (NodeId to = 0; to < csr.rows; ++to) {
    const std::size_t row = static_cast<std::size_t>(to);
    const std::int64_t begin = csr.rowptr[row];
    const std::int64_t end = csr.rowptr[row + 1];
    for (std::int64_t p = begin; p < end; ++p) {
      const NodeId from = csr.colind[static_cast<std::size_t>(p)];
      if (from < 0 || from >= csr.cols) {
        throw std::runtime_error("CSR colind contains out-of-range node id");
      }
      outgoing[static_cast<std::size_t>(from)].push_back(to);
    }
  }

  return outgoing;
}

std::vector<NodeId> bfs_predecessors(const std::vector<std::vector<NodeId>>& outgoing,
                                     NodeId source) {
  if (source < 0 || static_cast<std::size_t>(source) >= outgoing.size()) {
    throw std::runtime_error("source node is out of range: " +
                             std::to_string(source));
  }

  std::vector<NodeId> predecessor(outgoing.size(), -1);
  std::queue<NodeId> pending;
  pending.push(source);
  predecessor[static_cast<std::size_t>(source)] = source;

  while (!pending.empty()) {
    const NodeId from = pending.front();
    pending.pop();

    for (const NodeId to : outgoing[static_cast<std::size_t>(from)]) {
      NodeId& pred = predecessor[static_cast<std::size_t>(to)];
      if (pred != -1) {
        continue;
      }
      pred = from;
      pending.push(to);
    }
  }

  return predecessor;
}

std::vector<NodeId> reconstruct_path(const std::vector<NodeId>& predecessor,
                                     NodeId source,
                                     NodeId sink) {
  if (sink < 0 || static_cast<std::size_t>(sink) >= predecessor.size()) {
    throw std::runtime_error("sink node is out of range: " + std::to_string(sink));
  }
  if (predecessor[static_cast<std::size_t>(sink)] == -1) {
    throw std::runtime_error("sink node is unreachable from source " +
                             std::to_string(source) + ": " +
                             std::to_string(sink));
  }

  std::vector<NodeId> reversed;
  NodeId current = sink;
  while (current != source) {
    reversed.push_back(current);
    const NodeId pred = predecessor[static_cast<std::size_t>(current)];
    if (pred == -1 || pred == current) {
      throw std::runtime_error("failed to reconstruct path from source " +
                               std::to_string(source) + " to sink " +
                               std::to_string(sink));
    }
    current = pred;
  }
  reversed.push_back(source);
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

void write_routes(const Metadata& meta,
                  const std::vector<std::vector<NodeId>>& outgoing,
                  const std::filesystem::path& output_path) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("could not open routes output file: " +
                             output_path.string());
  }

  std::size_t routed_paths = 0;
  for (const RouteRequest& request : meta.route_requests) {
    if (request.sources.empty()) {
      throw std::runtime_error("route request has no legal source nodes");
    }

    const NodeId source = request.sources.front().node;
    const std::vector<NodeId> predecessor = bfs_predecessors(outgoing, source);
    const std::string net_name =
        trim(metadata_string(meta, request.net_string, "net name"));

    for (const SitePinNode& sink : request.sinks) {
      const std::vector<NodeId> path =
          reconstruct_path(predecessor, source, sink.node);
      out << "net=" << net_name << " source=" << source << " sink=" << sink.node
          << " path=";
      for (std::size_t i = 0; i < path.size(); ++i) {
        if (i != 0) {
          out << ",";
        }
        out << path[i];
      }
      out << "\n";
      if (!out) {
        throw std::runtime_error("failed while writing routes output: " +
                                 output_path.string());
      }
      ++routed_paths;
    }
  }

  std::cout << "wrote_routes: " << output_path << "\n";
  std::cout << "routed_paths: " << routed_paths << "\n";
  std::cout << "routed_nets: " << meta.route_requests.size() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const HostCsr csr = load_csr(options.csr_path);
    const Metadata meta = load_metadata(options.metadata_path);
    const std::vector<std::vector<NodeId>> outgoing =
        build_outgoing_adjacency(csr);
    write_routes(meta, outgoing, options.output_routes_path);
    return 0;
  } catch (const std::exception& e) {
    print_usage(argv[0]);
    std::cerr << "\nerror: " << e.what() << "\n";
    return 1;
  }
}
