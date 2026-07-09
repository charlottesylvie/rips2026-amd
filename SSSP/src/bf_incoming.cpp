// Runs HIP incoming-scan SSSP from route sources listed in a RIPS
// interchange metadata sidecar and writes validator-compatible shortest-path
// JSONL.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I HIP_kernel/minplus_mm/src \
//     SSSP/src/bf_incoming.cpp \
//     HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o bf_incoming
//
// Run:
//   ./bf_incoming data/logicnets_jscl.csrbin data/logicnets_jscl.csrbin.ifmeta.bin paths.jsonl

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"
#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR_device_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = kNoIndex;
  std::uint64_t pin_string = kNoIndex;
};

struct RouteRequest {
  std::uint64_t net_string = kNoIndex;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::vector<std::string> strings;
  std::vector<std::uint64_t> node_device_ids;
  std::uint64_t edge_attr_count = 0;
  std::vector<RouteRequest> route_requests;
};

struct PathEdge {
  int from = -1;
  int to = -1;
  Offset csr_edge = -1;
  float cost = 0.0f;
};

struct Query {
  std::size_t net_index = 0;
  std::string net_name;
  int source = -1;
  int target = -1;
};

struct SourceWork {
  int source = -1;
  std::vector<Query> queries;
};

struct Options {
  int max_iters = -1;
  std::size_t net_limit = 0;
  std::size_t source_limit = 0;
  std::size_t query_limit = 0;
  std::size_t source_progress_every = 100;
  double abs_tolerance = 1e-3;
  double rel_tolerance = 1e-5;
  bool include_unreached = true;
};

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

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }

  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

HostCsrF32 load_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(in, "CSR format version");
  const std::uint64_t orientation = read_u64(in, "CSR orientation");
  if (version != EXPECTED_CSR_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "CSR row count");
  const std::uint64_t cols = read_u64(in, "CSR column count");
  (void)read_u64(in, "declared edge count");
  (void)read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(in, "CSR colind count");
  const std::uint64_t values_count = read_u64(in, "CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(in, graph.colind, colind_count, "CSR colind");
  read_array(in, graph.values, values_count, "CSR values");
  validate_csr(graph);
  return graph;
}

RoutingMetadata load_interchange_metadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
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

  // Path strings and logical design name. They are not needed for SSSP output.
  (void)read_u64(in, "metadata device path string");
  (void)read_u64(in, "metadata physical path string");
  (void)read_u64(in, "metadata logical path string");
  (void)read_u64(in, "metadata logical design name");

  RoutingMetadata metadata;
  metadata.edge_attr_count = edge_attr_count;
  metadata.strings.reserve(static_cast<std::size_t>(string_count));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  read_array(in, metadata.node_device_ids, node_count, "metadata device node ids");

  // Edge attrs: tile string, pip data index.
  skip_bytes(in, edge_attr_count * 2 * sizeof(std::uint64_t), "metadata edge attrs");

  // PIP data: wire0 string, wire1 string, forward flag.
  skip_bytes(in, pip_data_count * 3 * sizeof(std::uint64_t), "metadata pip data");

  // Site pin attrs: node, site string, pin string.
  skip_bytes(in,
             site_pin_attr_count * 3 * sizeof(std::uint64_t),
             "metadata site pin attrs");

  metadata.route_requests.resize(static_cast<std::size_t>(route_request_count));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");

    const std::uint64_t source_count = read_u64(in, "metadata source count");
    request.sources.resize(static_cast<std::size_t>(source_count));
    for (SitePinNode& source : request.sources) {
      source.node = static_cast<int>(read_u64(in, "metadata source node"));
      source.site_string = read_u64(in, "metadata source site");
      source.pin_string = read_u64(in, "metadata source pin");
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    request.sinks.resize(static_cast<std::size_t>(sink_count));
    for (SitePinNode& sink : request.sinks) {
      sink.node = static_cast<int>(read_u64(in, "metadata sink node"));
      sink.site_string = read_u64(in, "metadata sink site");
      sink.pin_string = read_u64(in, "metadata sink pin");
    }
  }

  skip_bytes(in, logical_cell_count * 3 * sizeof(std::uint64_t), "metadata logical cells");
  skip_bytes(in, logical_net_count * 4 * sizeof(std::uint64_t), "metadata logical nets");
  skip_bytes(in,
             logical_port_instance_count * 7 * sizeof(std::uint64_t),
             "metadata logical port instances");
  skip_bytes(in, blocked_node_count * sizeof(std::uint64_t), "metadata blocked nodes");
  skip_bytes(in, sink_stop_node_count * sizeof(std::uint64_t), "metadata sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "metadata physical bytes");
  skip_bytes(in, logical_netlist_byte_count, "metadata logical bytes");

  return metadata;
}

bool valid_node(const HostCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) return "";
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

double tolerance_for(double lhs, double rhs, const Options& options) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return options.abs_tolerance + options.rel_tolerance * scale;
}

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dist,
                                                int source,
                                                int target,
                                                const Options& options) {
  if (!valid_node(graph, source) || !valid_node(graph, target)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance vector size does not match CSR rows");
  }
  if (source == target) {
    return {};
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (Offset guard = 0; guard < graph.rows && current != source; ++guard) {
    const std::size_t row = static_cast<std::size_t>(current);
    const float current_dist = dist[row];
    Offset best_edge = -1;
    int best_pred = -1;
    double best_error = std::numeric_limits<double>::infinity();

    for (Offset edge = graph.rowptr[row]; edge < graph.rowptr[row + 1]; ++edge) {
      const int pred = graph.colind[static_cast<std::size_t>(edge)];
      const float pred_dist = dist[static_cast<std::size_t>(pred)];
      if (!std::isfinite(pred_dist)) {
        continue;
      }

      const double candidate =
          static_cast<double>(pred_dist) + graph.values[static_cast<std::size_t>(edge)];
      const double error = std::fabs(candidate - current_dist);
      if (error <= tolerance_for(candidate, current_dist, options) &&
          error < best_error) {
        best_error = error;
        best_edge = edge;
        best_pred = pred;
      }
    }

    if (best_edge < 0) {
      throw std::runtime_error("could not reconstruct shortest path predecessor");
    }

    PathEdge path_edge;
    path_edge.from = best_pred;
    path_edge.to = current;
    path_edge.csr_edge = best_edge;
    path_edge.cost = graph.values[static_cast<std::size_t>(best_edge)];
    reversed.push_back(path_edge);
    current = best_pred;
  }

  if (current != source) {
    throw std::runtime_error("shortest path reconstruction did not reach source");
  }

  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::vector<int> nodes_from_edges(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

void write_metadata_record(std::ostream& out,
                           const HostCsrF32& graph,
                           const RoutingMetadata& metadata) {
  out << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\""
      << ",\"producer\":\"bf_incoming\""
      << ",\"node_count\":" << graph.rows
      << ",\"edge_count\":" << graph.nnz
      << ",\"route_request_count\":" << metadata.route_requests.size()
      << ",\"edge_orientation\":\"incoming\""
      << ",\"description\":\"row v, col u means directed edge u -> v\""
      << "}\n";
}

void write_path_record(std::ostream& out,
                       const Query& query,
                       bool reached,
                       float distance,
                       const std::vector<int>& nodes,
                       const std::vector<PathEdge>& edges) {
  out << "{\"type\":\"path\"";
  out << ",\"net\":";
  write_json_string(out, query.net_name);
  out << ",\"net_index\":" << query.net_index
      << ",\"source\":" << query.source
      << ",\"target\":" << query.target
      << ",\"reached\":" << (reached ? "true" : "false");

  if (reached) {
    out << ",\"distance\":" << std::setprecision(9) << distance;
  } else {
    out << ",\"distance\":null";
  }

  out << ",\"nodes\":[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) out << ',';
    out << nodes[i];
  }
  out << "]";

  out << ",\"csr_edges\":[";
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (i != 0) out << ',';
    out << edges[i].csr_edge;
  }
  out << "]}\n";
}

std::vector<SourceWork> build_source_work(const RoutingMetadata& metadata,
                                          const HostCsrF32& graph,
                                          const Options& options,
                                          std::size_t* total_queries) {
  std::vector<SourceWork> work;
  std::unordered_map<int, std::size_t> source_to_work;
  *total_queries = 0;

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());

  for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const std::string net_name = string_at(metadata, request.net_string);

    for (const SitePinNode& source_pin : request.sources) {
      if (options.query_limit != 0 && *total_queries >= options.query_limit) {
        return work;
      }
      if (!valid_node(graph, source_pin.node)) {
        std::cout << "[warning] skipping out-of-range source node "
                  << source_pin.node << " for net_index=" << net_index << '\n';
        continue;
      }

      std::size_t work_index = 0;
      const auto found = source_to_work.find(source_pin.node);
      if (found == source_to_work.end()) {
        work_index = work.size();
        source_to_work.emplace(source_pin.node, work_index);
        SourceWork source_work;
        source_work.source = source_pin.node;
        work.push_back(std::move(source_work));
      } else {
        work_index = found->second;
      }

      for (const SitePinNode& sink_pin : request.sinks) {
        if (options.query_limit != 0 && *total_queries >= options.query_limit) {
          return work;
        }
        if (!valid_node(graph, sink_pin.node)) {
          std::cout << "[warning] skipping out-of-range sink node "
                    << sink_pin.node << " for net_index=" << net_index << '\n';
          continue;
        }

        Query query;
        query.net_index = net_index;
        query.net_name = net_name;
        query.source = source_pin.node;
        query.target = sink_pin.node;
        work[work_index].queries.push_back(std::move(query));
        ++(*total_queries);
      }
    }
  }

  return work;
}

std::string progress_bar(std::size_t current, std::size_t total, int width) {
  if (width <= 0) return "";
  const double fraction =
      total <= 0 ? 1.0 : std::min(1.0, static_cast<double>(current) / total);
  const int filled = static_cast<int>(fraction * width);

  std::string bar;
  bar.reserve(static_cast<std::size_t>(width) + 2);
  bar.push_back('[');
  for (int i = 0; i < width; ++i) {
    bar.push_back(i < filled ? '#' : '.');
  }
  bar.push_back(']');
  return bar;
}

void print_source_progress(std::size_t completed, std::size_t total, bool final_line) {
  const double percent =
      total == 0 ? 100.0 : 100.0 * static_cast<double>(completed) /
                                static_cast<double>(total);
  std::ostringstream percent_text;
  percent_text << std::fixed
               << std::setprecision(percent > 0.0 && percent < 0.1 ? 4 : 1)
               << std::min(100.0, percent);

  std::cout << '\r'
            << "[bf_incoming] sources "
            << progress_bar(completed, total, 30)
            << ' ' << percent_text.str() << "%"
            << " " << completed << "/" << total
            << "        "
            << std::flush;
  if (final_line) {
    std::cout << '\n';
  }
}

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

std::size_t parse_size(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::size_t>(value);
}

int parse_int(const std::string& text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid integer for ") + name);
  }
  return static_cast<int>(value);
}

double parse_double(const std::string& text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid number for ") + name);
  }
  return value;
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
      << "  " << program << " <graph.csrbin> <metadata.ifmeta.bin> <output.paths.jsonl> [options]\n\n"
      << "Options:\n"
      << "  --max-iters <n>           Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --net-limit <n>           Use only the first n route requests.\n"
      << "  --source-limit <n>        Run only the first n unique source nodes.\n"
      << "  --query-limit <n>         Emit at most n source-sink path records.\n"
      << "  --source-progress-every <n>\n"
      << "                              Print source summary every n sources. Default: 100.\n"
      << "  --skip-unreached          Do not write JSONL records for unreachable sinks.\n"
      << "  --abs-tol <x>             Path reconstruction absolute tolerance. Default: 1e-3.\n"
      << "  --rel-tol <x>             Path reconstruction relative tolerance. Default: 1e-5.\n"
      << "  --help                    Print this message.\n";
}

Options parse_options(int argc,
                      char** argv,
                      std::filesystem::path* csr_path,
                      std::filesystem::path* metadata_path,
                      std::filesystem::path* output_path) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--max-iters") {
      options.max_iters = parse_int(require_value(&i, argc, argv, "--max-iters"), "--max-iters");
    } else if (arg == "--net-limit") {
      options.net_limit = parse_size(require_value(&i, argc, argv, "--net-limit"), "--net-limit");
    } else if (arg == "--source-limit") {
      options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (arg == "--query-limit") {
      options.query_limit =
          parse_size(require_value(&i, argc, argv, "--query-limit"), "--query-limit");
    } else if (arg == "--source-progress-every" || arg == "--bf-progress-every") {
      options.source_progress_every =
          parse_size(require_value(&i, argc, argv, arg.c_str()), arg.c_str());
    } else if (arg == "--skip-unreached") {
      options.include_unreached = false;
    } else if (arg == "--abs-tol") {
      options.abs_tolerance = parse_double(require_value(&i, argc, argv, "--abs-tol"),
                                           "--abs-tol");
    } else if (arg == "--rel-tol") {
      options.rel_tolerance = parse_double(require_value(&i, argc, argv, "--rel-tol"),
                                           "--rel-tol");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (csr_path->empty()) {
      *csr_path = arg;
    } else if (metadata_path->empty()) {
      *metadata_path = arg;
    } else if (output_path->empty()) {
      *output_path = arg;
    } else {
      throw std::runtime_error("too many positional arguments");
    }
  }

  if (csr_path->empty() || metadata_path->empty() || output_path->empty()) {
    throw std::runtime_error("expected <graph.csrbin> <metadata.ifmeta.bin> <output.paths.jsonl>");
  }
  if (options.max_iters < -1) {
    throw std::runtime_error("--max-iters must be -1 or nonnegative");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_begin = std::chrono::steady_clock::now();
    std::filesystem::path csr_path;
    std::filesystem::path metadata_path;
    std::filesystem::path output_path;
    const Options options = parse_options(argc, argv, &csr_path, &metadata_path, &output_path);

    std::cout << "[bf_incoming] loading CSR: " << csr_path.string() << '\n';
    HostCsrF32 graph = load_csrbin(csr_path);
    std::cout << "[bf_incoming] graph rows=" << graph.rows
              << " nnz=" << graph.nnz
              << " orientation=incoming(row v, col u means u -> v)\n";

    std::cout << "[bf_incoming] loading metadata: " << metadata_path.string() << '\n';
    RoutingMetadata metadata = load_interchange_metadata(metadata_path);
    if (metadata.node_device_ids.size() != static_cast<std::size_t>(graph.rows)) {
      throw std::runtime_error("metadata node count does not match CSR rows");
    }
    if (metadata.edge_attr_count != static_cast<std::uint64_t>(graph.nnz)) {
      throw std::runtime_error("metadata edge attribute count does not match CSR nnz");
    }
    std::cout << "[bf_incoming] route_requests="
              << metadata.route_requests.size() << '\n';

    std::size_t total_queries = 0;
    std::vector<SourceWork> work =
        build_source_work(metadata, graph, options, &total_queries);
    const std::size_t source_count =
        options.source_limit == 0
            ? work.size()
            : std::min(options.source_limit, work.size());
    std::cout << "[bf_incoming] unique_sources=" << work.size()
              << " sources_to_run=" << source_count
              << " source_sink_queries=" << total_queries << '\n';

    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path);
    if (!out) {
      throw std::runtime_error("could not open output JSONL file: " + output_path.string());
    }
    write_metadata_record(out, graph, metadata);

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    std::uint64_t paths_written = 0;
    std::uint64_t reached_count = 0;
    std::uint64_t unreached_count = 0;

    try {
      std::cout << "[bf_incoming] copying incoming CSR to GPU once\n";
      bf_csr_detail::DeviceCsrOwner d_graph =
          bf_csr_detail::copy_host_csr_to_device(graph, stream);

      print_source_progress(0, source_count, source_count == 0);
      for (std::size_t source_index = 0; source_index < source_count; ++source_index) {
        const SourceWork& source_work = work[source_index];
        if (source_work.queries.empty()) {
          print_source_progress(source_index + 1,
                                source_count,
                                source_index + 1 == source_count);
          continue;
        }
        const std::uint64_t paths_before = paths_written;

        BellmanFordCsrResult result =
            bellman_ford_minplus_hip_csr(d_graph.view,
                                         source_work.source,
                                         options.max_iters,
                                         stream);

        for (const Query& query : source_work.queries) {
          const float distance = result.dist[static_cast<std::size_t>(query.target)];
          const bool reached = std::isfinite(distance);
          if (!reached) {
            ++unreached_count;
            if (options.include_unreached) {
              write_path_record(out, query, false, distance, {}, {});
              ++paths_written;
            }
            continue;
          }

          std::vector<PathEdge> edges =
              reconstruct_shortest_path(graph,
                                        result.dist,
                                        query.source,
                                        query.target,
                                        options);
          std::vector<int> nodes = nodes_from_edges(query.source, edges);
          write_path_record(out, query, true, distance, nodes, edges);
          ++reached_count;
          ++paths_written;
        }

        const std::size_t completed = source_index + 1;
        const bool final_source = completed == source_count;
        print_source_progress(completed, source_count, final_source);

        const bool should_print_summary =
            options.source_progress_every > 0 &&
            (completed % options.source_progress_every == 0 || final_source);
        if (should_print_summary) {
          if (!final_source) {
            std::cout << '\n';
          }
          const auto now = std::chrono::steady_clock::now();
          const double total_seconds_so_far =
              std::chrono::duration<double>(now - total_begin).count();
          std::cout << "[bf_incoming] source " << completed
                    << "/" << source_count
                    << " node=" << source_work.source
                    << " queries=" << source_work.queries.size()
                    << " iterations_used=" << result.iterations_used
                    << " converged=" << (result.converged ? "yes" : "no")
                    << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
                    << total_seconds_so_far
                    << " source_paths_written="
                    << (paths_written - paths_before)
                    << " total_paths_written=" << paths_written << '\n'
                    << std::flush;
        }
      }
      check_hip(hipStreamSynchronize(stream), "sync before freeing incoming CSR graph");
    } catch (...) {
      const hipError_t destroy_status = hipStreamDestroy(stream);
      if (destroy_status != hipSuccess) {
        std::cerr << "[bf_incoming] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
      throw;
    }
    check_hip(hipStreamSynchronize(stream), "final stream sync");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");

    const auto total_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(total_end - total_begin).count();
    std::cout << "[bf_incoming] wrote " << paths_written
              << " path records to " << output_path.string()
              << " reached=" << reached_count
              << " unreached=" << unreached_count
              << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
              << total_seconds << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[bf_incoming] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
