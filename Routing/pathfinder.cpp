#include "pathfinder.hpp"

#include "../HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.hpp"

// Example build from the repository root, once ROCm/HIP is available:
//   hipcc -std=c++17 -O3 -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I HIP_kernel/delta_stepping/src \
//     Routing/pathfinder.cpp \
//     HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp \
//     -o pathfinder
//
// Run:
//   ./pathfinder design.csrbin design.csrbin.ifmeta.bin --net-limit 10

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace routing {
namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;

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
  if (values.empty()) {
    return;
  }
  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
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
  for (minplus_sparse::Offset row = 0; row < graph.rows; ++row) {
    const minplus_sparse::Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const minplus_sparse::Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<minplus_sparse::Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

void validate_options(const PathfinderOptions& options) {
  if (!(options.delta > 0.0f) || !std::isfinite(options.delta)) {
    throw std::invalid_argument("PathFinder delta must be finite and positive");
  }
  if (options.max_pathfinder_iterations <= 0) {
    throw std::invalid_argument("max_pathfinder_iterations must be positive");
  }
  if (options.capacity <= 0) {
    throw std::invalid_argument("capacity must be positive");
  }
  if (!(options.initial_present_factor >= 0.0f) ||
      !std::isfinite(options.initial_present_factor)) {
    throw std::invalid_argument("initial_present_factor must be finite and nonnegative");
  }
  if (!(options.present_factor_multiplier >= 1.0f) ||
      !std::isfinite(options.present_factor_multiplier)) {
    throw std::invalid_argument("present_factor_multiplier must be finite and >= 1");
  }
  if (!(options.history_factor >= 0.0f) || !std::isfinite(options.history_factor)) {
    throw std::invalid_argument("history_factor must be finite and nonnegative");
  }
}

bool valid_node(int node, minplus_sparse::Offset rows) {
  return node >= 0 && static_cast<minplus_sparse::Offset>(node) < rows;
}

void add_unique_node(std::vector<int>& nodes,
                     std::vector<std::uint8_t>& seen,
                     int node) {
  if (node < 0 || static_cast<std::size_t>(node) >= seen.size()) {
    return;
  }
  if (seen[static_cast<std::size_t>(node)] != 0) {
    return;
  }
  seen[static_cast<std::size_t>(node)] = 1;
  nodes.push_back(node);
}

std::vector<int> nodes_from_path(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

HostCsrF32 make_costed_graph(const HostCsrF32& base_graph,
                             const std::vector<int>& occupancy,
                             const std::vector<float>& history,
                             const PathfinderOptions& options,
                             float present_factor) {
  HostCsrF32 graph = base_graph;
  graph.values.resize(base_graph.values.size());

  for (minplus_sparse::Offset dst = 0; dst < base_graph.rows; ++dst) {
    const std::size_t dst_index = static_cast<std::size_t>(dst);
    const int overuse_if_taken =
        occupancy[dst_index] + 1 - options.capacity;
    const float present_cost =
        overuse_if_taken > 0
            ? 1.0f + present_factor * static_cast<float>(overuse_if_taken)
            : 1.0f;
    const float historical_cost = 1.0f + history[dst_index];
    const float node_cost = present_cost * historical_cost;

    for (minplus_sparse::Offset edge = base_graph.rowptr[dst_index];
         edge < base_graph.rowptr[dst_index + 1];
         ++edge) {
      graph.values[static_cast<std::size_t>(edge)] =
          base_graph.values[static_cast<std::size_t>(edge)] * node_cost;
    }
  }

  return graph;
}

RoutedSink route_sink_from_tree(const HostCsrF32& graph,
                               const std::vector<int>& source_candidates,
                               const std::vector<std::uint8_t>& tree_seen,
                               int target,
                               const PathfinderOptions& options,
                               hipStream_t stream) {
  RoutedSink best;
  best.target = target;
  best.distance = std::numeric_limits<float>::infinity();

  if (!valid_node(target, graph.rows)) {
    return best;
  }

  for (const int source : source_candidates) {
    if (!valid_node(source, graph.rows)) {
      continue;
    }

    RoutedSink candidate;
    candidate.source = source;
    candidate.target = target;

    if (source == target) {
      candidate.distance = 0.0f;
      candidate.reached = true;
      candidate.nodes.push_back(source);
    } else {
      DeltaSteppingCsrResult sssp =
          delta_stepping_minplus_hip_csr(graph,
                                         source,
                                         target,
                                         options.delta,
                                         options.max_sssp_iterations,
                                         stream,
                                         nullptr,
                                         nullptr);
      if (!sssp.target_reached ||
          !std::isfinite(sssp.dist[static_cast<std::size_t>(target)])) {
        continue;
      }

      candidate.distance = sssp.dist[static_cast<std::size_t>(target)];
      candidate.edges = reconstruct_shortest_path(graph, sssp.dist, source, target);
      candidate.nodes = nodes_from_path(source, candidate.edges);
      bool reenters_tree = false;
      for (std::size_t i = 1; i < candidate.nodes.size(); ++i) {
        const int node = candidate.nodes[i];
        if (node >= 0 &&
            static_cast<std::size_t>(node) < tree_seen.size() &&
            tree_seen[static_cast<std::size_t>(node)] != 0) {
          reenters_tree = true;
          break;
        }
      }
      if (reenters_tree) {
        continue;
      }
      candidate.reached = true;
    }

    if (!candidate.reached) {
      continue;
    }

    const bool better_distance = candidate.distance < best.distance;
    const bool equal_shorter_path =
        candidate.distance == best.distance &&
        candidate.nodes.size() < best.nodes.size();
    if (!best.reached || better_distance || equal_shorter_path) {
      best = std::move(candidate);
    }
  }

  return best;
}

RoutedNet route_net(const HostCsrF32& graph,
                    const RouteRequest& request,
                    const PathfinderOptions& options,
                    hipStream_t stream) {
  RoutedNet net;
  net.net_string = request.net_string;

  std::vector<int> source_candidates;
  std::vector<std::uint8_t> tree_seen(static_cast<std::size_t>(graph.rows), 0);
  for (const SitePinNode& source : request.sources) {
    add_unique_node(source_candidates, tree_seen, source.node);
  }

  if (source_candidates.empty()) {
    net.reached_all_sinks = false;
    return net;
  }

  bool reached_all = true;
  for (const SitePinNode& sink : request.sinks) {
    RoutedSink routed_sink =
        route_sink_from_tree(graph, source_candidates, tree_seen, sink.node, options, stream);
    if (!routed_sink.reached) {
      reached_all = false;
    } else {
      for (const int node : routed_sink.nodes) {
        add_unique_node(source_candidates, tree_seen, node);
      }
    }
    net.sinks.push_back(std::move(routed_sink));
  }

  net.reached_all_sinks = reached_all;
  net.unique_nodes = std::move(source_candidates);
  return net;
}

void commit_net_occupancy(const RoutedNet& net, std::vector<int>& occupancy) {
  if (!net.reached_all_sinks) {
    return;
  }
  for (const int node : net.unique_nodes) {
    if (node >= 0 && static_cast<std::size_t>(node) < occupancy.size()) {
      ++occupancy[static_cast<std::size_t>(node)];
    }
  }
}

void update_congestion_stats(const std::vector<int>& occupancy,
                             int capacity,
                             int* overused_nodes,
                             int* max_occupancy) {
  *overused_nodes = 0;
  *max_occupancy = 0;
  for (const int used : occupancy) {
    *max_occupancy = std::max(*max_occupancy, used);
    if (used > capacity) {
      ++(*overused_nodes);
    }
  }
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u";
          const char* hex = "0123456789abcdef";
          out << '0' << '0' << hex[(ch >> 4) & 0xf] << hex[ch & 0xf];
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

std::uint64_t edge_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

}  // namespace

std::filesystem::path default_metadata_path(const std::filesystem::path& csr_path) {
  std::filesystem::path path = csr_path;
  path += ".ifmeta.bin";
  return path;
}

int parse_int_arg(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

std::size_t parse_size_arg(const char* text, const char* name) {
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

float parse_float_arg(const char* text, const char* name) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <graph.csrbin> [metadata.ifmeta.bin] [options]\n\n"
      << "Options:\n"
      << "  --delta <float>                 Delta-stepping bucket width. Default: 4\n"
      << "  --max-pathfinder-iters <int>    PathFinder rip-up/reroute rounds. Default: 30\n"
      << "  --max-sssp-iters <int>          Delta-stepping bucket rounds, -1 for default.\n"
      << "  --capacity <int>                Routing-resource capacity. Default: 1\n"
      << "  --present-factor <float>        Initial present congestion factor. Default: 1\n"
      << "  --present-multiplier <float>    Per-iteration factor multiplier. Default: 2\n"
      << "  --history-factor <float>        Historical congestion increment. Default: 1\n"
      << "  --net-limit <count>             Route only the first count requests.\n"
      << "  --routes-out <path>             Write routed PIP tree data as JSONL.\n";
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
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<minplus_sparse::Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<minplus_sparse::Offset>(rows);
  graph.cols = static_cast<minplus_sparse::Offset>(cols);
  graph.nnz = static_cast<minplus_sparse::Offset>(nnz);
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

  RoutingMetadata metadata;
  metadata.device_path_string = read_u64(in, "metadata device path string");
  metadata.physical_path_string = read_u64(in, "metadata physical path string");
  metadata.logical_path_string = read_u64(in, "metadata logical path string");
  metadata.logical_design_name_string = read_u64(in, "metadata logical design name");

  metadata.strings.reserve(static_cast<std::size_t>(string_count));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  read_array(in, metadata.node_device_ids, node_count, "metadata device node ids");

  metadata.edge_attrs.resize(static_cast<std::size_t>(edge_attr_count));
  for (EdgeAttr& attr : metadata.edge_attrs) {
    attr.tile_string = read_u64(in, "metadata edge tile string");
    attr.pip_data_index = read_u64(in, "metadata edge pip data index");
  }

  metadata.pip_data.resize(static_cast<std::size_t>(pip_data_count));
  for (PipData& pip : metadata.pip_data) {
    pip.wire0_string = read_u64(in, "metadata pip wire0 string");
    pip.wire1_string = read_u64(in, "metadata pip wire1 string");
    pip.forward = read_u64(in, "metadata pip forward flag") != 0;
  }

  metadata.site_pin_attrs.resize(static_cast<std::size_t>(site_pin_attr_count));
  for (SitePinNode& attr : metadata.site_pin_attrs) {
    attr.node = static_cast<int>(read_u64(in, "metadata site pin node"));
    attr.site_string = read_u64(in, "metadata site pin site");
    attr.pin_string = read_u64(in, "metadata site pin pin");
  }

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

  for (std::uint64_t i = 0; i < logical_cell_count; ++i) {
    (void)read_u64(in, "metadata logical cell declaration");
    (void)read_u64(in, "metadata logical cell net begin");
    (void)read_u64(in, "metadata logical cell net count");
  }
  for (std::uint64_t i = 0; i < logical_net_count; ++i) {
    (void)read_u64(in, "metadata logical net name");
    (void)read_u64(in, "metadata logical net cell");
    (void)read_u64(in, "metadata logical net port begin");
    (void)read_u64(in, "metadata logical net port count");
  }
  for (std::uint64_t i = 0; i < logical_port_instance_count; ++i) {
    (void)read_u64(in, "metadata logical port name");
    (void)read_u64(in, "metadata logical instance name");
    (void)read_u64(in, "metadata logical port index");
    (void)read_u64(in, "metadata logical instance index");
    (void)read_u64(in, "metadata logical bus index");
    (void)read_u64(in, "metadata logical has bus");
    (void)read_u64(in, "metadata logical external port");
  }

  read_array(in, metadata.blocked_nodes, blocked_node_count, "metadata blocked nodes");
  read_array(in, metadata.sink_stop_nodes, sink_stop_node_count, "metadata sink stop nodes");

  std::vector<std::uint8_t> skipped_bytes;
  read_array(in, skipped_bytes, physical_netlist_byte_count, "metadata physical bytes");
  read_array(in, skipped_bytes, logical_netlist_byte_count, "metadata logical bytes");

  return metadata;
}

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dist,
                                                int source,
                                                int target) {
  validate_csr(graph);
  if (!valid_node(source, graph.rows) || !valid_node(target, graph.rows)) {
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
  for (minplus_sparse::Offset guard = 0; guard < graph.rows && current != source; ++guard) {
    const std::size_t row = static_cast<std::size_t>(current);
    const float current_dist = dist[row];
    minplus_sparse::Offset best_edge = -1;
    int best_pred = -1;
    float best_error = std::numeric_limits<float>::infinity();

    for (minplus_sparse::Offset edge = graph.rowptr[row];
         edge < graph.rowptr[row + 1];
         ++edge) {
      const int pred = graph.colind[static_cast<std::size_t>(edge)];
      if (pred == current) {
        continue;
      }
      const float pred_dist = dist[static_cast<std::size_t>(pred)];
      if (!std::isfinite(pred_dist)) {
        continue;
      }
      const float candidate =
          pred_dist + graph.values[static_cast<std::size_t>(edge)];
      const float error = std::fabs(candidate - current_dist);
      const float tolerance =
          1e-3f * std::max({1.0f, std::fabs(candidate), std::fabs(current_dist)});
      if (error <= tolerance && error < best_error) {
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

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream) {
  validate_csr(base_graph);
  validate_options(options);
  if (metadata.node_device_ids.size() != static_cast<std::size_t>(base_graph.rows)) {
    throw std::runtime_error("metadata node count does not match CSR row count");
  }
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(base_graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }

  PathfinderResult result;
  result.occupancy.assign(static_cast<std::size_t>(base_graph.rows), 0);
  result.history.assign(static_cast<std::size_t>(base_graph.rows), 0.0f);

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());
  float present_factor = options.initial_present_factor;

  for (int iteration = 0; iteration < options.max_pathfinder_iterations; ++iteration) {
    std::fill(result.occupancy.begin(), result.occupancy.end(), 0);
    result.nets.clear();
    result.nets.reserve(route_request_count);

    bool all_sinks_reached = true;
    for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
      HostCsrF32 graph =
          make_costed_graph(base_graph, result.occupancy, result.history, options, present_factor);
      RoutedNet net =
          route_net(graph, metadata.route_requests[net_index], options, stream);
      if (!net.reached_all_sinks) {
        all_sinks_reached = false;
      }
      commit_net_occupancy(net, result.occupancy);
      result.nets.push_back(std::move(net));
    }

    result.iterations_used = iteration + 1;
    result.all_sinks_reached = all_sinks_reached;
    update_congestion_stats(result.occupancy,
                            options.capacity,
                            &result.overused_nodes,
                            &result.max_occupancy);
    result.routed = all_sinks_reached && result.overused_nodes == 0;
    if (result.routed) {
      break;
    }

    for (std::size_t node = 0; node < result.occupancy.size(); ++node) {
      const int overuse = result.occupancy[node] - options.capacity;
      if (overuse > 0) {
        result.history[node] += options.history_factor * static_cast<float>(overuse);
      }
    }
    present_factor *= options.present_factor_multiplier;
  }

  return result;
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) {
    return {};
  }
  if (index >= metadata.strings.size()) {
    std::ostringstream out;
    out << "<bad-string-" << index << ">";
    return out.str();
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result) {
  validate_csr(graph);
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }
  if (result.nets.size() > metadata.route_requests.size()) {
    throw std::runtime_error("pathfinder result has more nets than metadata requests");
  }
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("could not open routes output file: " + path.string());
  }

  for (std::size_t net_index = 0; net_index < result.nets.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const RoutedNet& net = result.nets[net_index];

    out << "{\"net\":";
    write_json_string(out, string_at(metadata, request.net_string));
    out << ",\"routed\":" << (net.reached_all_sinks ? "true" : "false");

    out << ",\"sources\":[";
    for (std::size_t i = 0; i < request.sources.size(); ++i) {
      const SitePinNode& source = request.sources[i];
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << source.node << ",\"site\":";
      write_json_string(out, string_at(metadata, source.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, source.pin_string));
      out << '}';
    }
    out << ']';

    out << ",\"sinks\":[";
    for (std::size_t i = 0; i < request.sinks.size(); ++i) {
      const SitePinNode& sink_pin = request.sinks[i];
      const bool has_sink_result = i < net.sinks.size();
      const RoutedSink* sink = has_sink_result ? &net.sinks[i] : nullptr;
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << sink_pin.node << ",\"site\":";
      write_json_string(out, string_at(metadata, sink_pin.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, sink_pin.pin_string));
      out << ",\"reached\":" << (sink != nullptr && sink->reached ? "true" : "false");
      out << ",\"source\":" << (sink != nullptr ? sink->source : -1);
      out << '}';
    }
    out << ']';

    std::unordered_set<std::uint64_t> seen_edges;
    out << ",\"edges\":[";
    bool first_edge = true;
    for (const RoutedSink& sink : net.sinks) {
      if (!sink.reached) {
        continue;
      }
      for (const PathEdge& path_edge : sink.edges) {
        if (path_edge.csr_edge < 0 ||
            path_edge.csr_edge >= graph.nnz ||
            !valid_node(path_edge.from, graph.rows) ||
            !valid_node(path_edge.to, graph.rows)) {
          throw std::runtime_error("pathfinder result contains an invalid path edge");
        }
        if (!seen_edges.insert(edge_key(path_edge.from, path_edge.to)).second) {
          continue;
        }

        const EdgeAttr& attr =
            metadata.edge_attrs[static_cast<std::size_t>(path_edge.csr_edge)];
        if (attr.pip_data_index >= metadata.pip_data.size()) {
          throw std::runtime_error("route edge references invalid PIP data");
        }
        const PipData& pip =
            metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];

        if (!first_edge) {
          out << ',';
        }
        first_edge = false;
        out << "{\"from\":" << path_edge.from
            << ",\"to\":" << path_edge.to
            << ",\"csr_edge\":" << path_edge.csr_edge
            << ",\"tile\":";
        write_json_string(out, string_at(metadata, attr.tile_string));
        out << ",\"wire0\":";
        write_json_string(out, string_at(metadata, pip.wire0_string));
        out << ",\"wire1\":";
        write_json_string(out, string_at(metadata, pip.wire1_string));
        out << ",\"forward\":" << (pip.forward ? "true" : "false") << '}';
      }
    }
    out << "]}\n";
  }
}

}  // namespace routing

#ifndef ROUTING_PATHFINDER_NO_MAIN
int main(int argc, char** argv) {
  try {
    if (argc < 2 ||
        (argc == 2 && (std::string(argv[1]) == "-h" ||
                       std::string(argv[1]) == "--help"))) {
      routing::print_usage(argv[0]);
      return argc < 2 ? 1 : 0;
    }

    const std::filesystem::path csr_path = argv[1];
    std::filesystem::path metadata_path;
    std::filesystem::path routes_out_path;
    routing::PathfinderOptions options;

    int arg = 2;
    if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
      metadata_path = argv[arg++];
    } else {
      metadata_path = routing::default_metadata_path(csr_path);
    }

    while (arg < argc) {
      const std::string option = argv[arg++];
      auto require_value = [&](const char* name) -> const char* {
        if (arg >= argc) {
          throw std::runtime_error(std::string(name) + " requires a value");
        }
        return argv[arg++];
      };

      if (option == "-h" || option == "--help") {
        routing::print_usage(argv[0]);
        return 0;
      }
      if (option == "--delta") {
        options.delta = routing::parse_float_arg(require_value("--delta"), "delta");
      } else if (option == "--max-pathfinder-iters") {
        options.max_pathfinder_iterations =
            routing::parse_int_arg(require_value("--max-pathfinder-iters"),
                                   "max-pathfinder-iters");
      } else if (option == "--max-sssp-iters") {
        options.max_sssp_iterations =
            routing::parse_int_arg(require_value("--max-sssp-iters"), "max-sssp-iters");
      } else if (option == "--capacity") {
        options.capacity = routing::parse_int_arg(require_value("--capacity"), "capacity");
      } else if (option == "--present-factor") {
        options.initial_present_factor =
            routing::parse_float_arg(require_value("--present-factor"), "present-factor");
      } else if (option == "--present-multiplier") {
        options.present_factor_multiplier =
            routing::parse_float_arg(require_value("--present-multiplier"),
                                     "present-multiplier");
      } else if (option == "--history-factor") {
        options.history_factor =
            routing::parse_float_arg(require_value("--history-factor"), "history-factor");
      } else if (option == "--net-limit") {
        options.net_limit =
            routing::parse_size_arg(require_value("--net-limit"), "net-limit");
      } else if (option == "--routes-out") {
        routes_out_path = require_value("--routes-out");
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    HostCsrF32 graph = routing::load_csrbin(csr_path);
    routing::RoutingMetadata metadata = routing::load_interchange_metadata(metadata_path);
    routing::PathfinderResult result =
        routing::run_pathfinder(graph, metadata, options, nullptr);

    std::cout << "csr: " << csr_path << "\n";
    std::cout << "metadata: " << metadata_path << "\n";
    std::cout << "route_requests: " << metadata.route_requests.size() << "\n";
    if (options.net_limit != 0) {
      std::cout << "net_limit: " << options.net_limit << "\n";
    }
    std::cout << "iterations_used: " << result.iterations_used << "\n";
    std::cout << "all_sinks_reached: " << (result.all_sinks_reached ? "true" : "false") << "\n";
    std::cout << "overused_nodes: " << result.overused_nodes << "\n";
    std::cout << "max_occupancy: " << result.max_occupancy << "\n";
    std::cout << "routed: " << (result.routed ? "true" : "false") << "\n";

    if (!routes_out_path.empty()) {
      if (!result.routed) {
        std::cerr << "error: refusing to write routes for an unrouted or congested result\n";
        return 2;
      }
      routing::write_routes_jsonl(routes_out_path, graph, metadata, result);
      std::cout << "routes_out: " << routes_out_path << "\n";
    }

    const std::size_t printed_nets = std::min<std::size_t>(result.nets.size(), 10);
    for (std::size_t i = 0; i < printed_nets; ++i) {
      const routing::RoutedNet& net = result.nets[i];
      std::size_t reached_sinks = 0;
      for (const routing::RoutedSink& sink : net.sinks) {
        if (sink.reached) {
          ++reached_sinks;
        }
      }
      std::cout << "net[" << i << "]: "
                << routing::string_at(metadata, net.net_string)
                << " sinks=" << reached_sinks << "/" << net.sinks.size()
                << " nodes=" << net.unique_nodes.size()
                << " reached_all=" << (net.reached_all_sinks ? "true" : "false")
                << "\n";
    }
    if (printed_nets < result.nets.size()) {
      std::cout << "net_summary_truncated_after: " << printed_nets << "\n";
    }

    return result.routed ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    if (argc < 2) {
      routing::print_usage(argv[0]);
    }
    return 1;
  }
}
#endif
