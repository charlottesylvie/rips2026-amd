// Source proximity checker for RIPS CSR graphs.
//
// Reads a .csrbin graph and .csrbin.ifmeta.bin metadata file, extracts unique
// route sources, and reports hop-2 source-connector-source relationships. The
// .csrbin orientation is incoming-edge:
//
//   row v, col u means directed edge u -> v
//
// This tool always treats source-connector adjacency as undirected for
// discovery, then reports the directed orientation of each two-edge pattern.
//
// Build from rips2026-amd:
//   g++ -std=c++17 -O3 -I SSSP/src SSSP/src/check_sources.cpp -o check_sources
//
// Run:
//   ./check_sources graph.csrbin graph.csrbin.ifmeta.bin

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;

using Offset = std::int64_t;
using Index = int;

struct HostCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

struct OutgoingCsr {
  std::vector<Offset> rowptr;
  std::vector<Index> to;
  std::vector<Offset> csr_edge;
};

struct Options {
  std::size_t net_limit = 0;
  std::size_t source_limit = 0;
  std::size_t max_sets_print = 0;
  std::size_t progress_every = 25;
};

struct ConnectorSourceRelation {
  int connector = -1;
  int source_index = -1;
  int source = -1;
  bool source_to_connector = false;
  bool connector_to_source = false;
};

struct Hop2ConnectorStats {
  std::uint64_t source_connector_relations = 0;
  std::uint64_t connector_nodes_with_two_sources = 0;
  std::uint64_t connector_mediated_pairs = 0;
  std::uint64_t printed_sets = 0;
  std::uint64_t source1_to_connector_to_source2 = 0;
  std::uint64_t source1_from_connector_to_source2 = 0;
  std::uint64_t source1_to_connector_from_source2 = 0;
  std::uint64_t source1_from_connector_from_source2 = 0;
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

void skip_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  skip_bytes(in, size, "metadata string bytes");
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
    throw std::runtime_error("CSR graph is too large for this checker");
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

  return graph;
}

bool valid_node(const HostCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

void add_unique_source(const HostCsrF32& graph,
                       int node,
                       std::vector<int>* sources,
                       std::unordered_set<int>* seen) {
  if (!valid_node(graph, node)) {
    std::cout << "[check_sources] warning: skipping out-of-range source node "
              << node << '\n';
    return;
  }
  if (seen->insert(node).second) {
    sources->push_back(node);
  }
}

std::vector<int> load_unique_sources_from_metadata(const std::filesystem::path& path,
                                                   const HostCsrF32& graph,
                                                   const Options& options,
                                                   std::uint64_t* total_route_requests) {
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

  (void)blocked_node_count;
  (void)sink_stop_node_count;
  (void)logical_cell_count;
  (void)logical_net_count;
  (void)logical_port_instance_count;
  (void)physical_netlist_byte_count;
  (void)logical_netlist_byte_count;

  (void)read_u64(in, "metadata device path string");
  (void)read_u64(in, "metadata physical path string");
  (void)read_u64(in, "metadata logical path string");
  (void)read_u64(in, "metadata logical design name");

  if (total_route_requests) {
    *total_route_requests = route_request_count;
  }
  if (node_count != static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error("metadata node count does not match CSR rows");
  }
  if (edge_attr_count != static_cast<std::uint64_t>(graph.nnz)) {
    throw std::runtime_error("metadata edge attribute count does not match CSR nnz");
  }

  for (std::uint64_t i = 0; i < string_count; ++i) {
    skip_string(in);
  }

  skip_bytes(in, node_count * sizeof(std::uint64_t), "metadata device node ids");
  skip_bytes(in, edge_attr_count * 2 * sizeof(std::uint64_t), "metadata edge attrs");
  skip_bytes(in, pip_data_count * 3 * sizeof(std::uint64_t), "metadata pip data");
  skip_bytes(in,
             site_pin_attr_count * 3 * sizeof(std::uint64_t),
             "metadata site pin attrs");

  std::vector<int> sources;
  std::unordered_set<int> seen_sources;
  const std::uint64_t route_requests_to_read =
      options.net_limit == 0
          ? route_request_count
          : std::min<std::uint64_t>(route_request_count, options.net_limit);

  for (std::uint64_t net = 0; net < route_requests_to_read; ++net) {
    (void)read_u64(in, "metadata route request net");
    (void)read_u64(in, "metadata route logical net");

    const std::uint64_t source_count = read_u64(in, "metadata source count");
    for (std::uint64_t i = 0; i < source_count; ++i) {
      const int source_node = static_cast<int>(read_u64(in, "metadata source node"));
      (void)read_u64(in, "metadata source site");
      (void)read_u64(in, "metadata source pin");
      add_unique_source(graph, source_node, &sources, &seen_sources);
      if (options.source_limit != 0 && sources.size() >= options.source_limit) {
        return sources;
      }
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    skip_bytes(in,
               sink_count * 3 * sizeof(std::uint64_t),
               "metadata sink site pin nodes");
  }

  return sources;
}

OutgoingCsr build_outgoing_csr(const HostCsrF32& graph) {
  std::cout << "[check_sources] building outgoing adjacency from incoming CSR\n";
  const auto start = std::chrono::steady_clock::now();

  OutgoingCsr out;
  out.rowptr.assign(static_cast<std::size_t>(graph.rows + 1), 0);
  out.to.resize(static_cast<std::size_t>(graph.nnz));
  out.csr_edge.resize(static_cast<std::size_t>(graph.nnz));

  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const int from = graph.colind[static_cast<std::size_t>(edge)];
    ++out.rowptr[static_cast<std::size_t>(from + 1)];
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    out.rowptr[static_cast<std::size_t>(row + 1)] +=
        out.rowptr[static_cast<std::size_t>(row)];
  }

  std::vector<Offset> cursor = out.rowptr;
  for (Offset to_node = 0; to_node < graph.rows; ++to_node) {
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(to_node)];
         edge < graph.rowptr[static_cast<std::size_t>(to_node + 1)];
         ++edge) {
      const int from = graph.colind[static_cast<std::size_t>(edge)];
      const Offset out_pos = cursor[static_cast<std::size_t>(from)]++;
      out.to[static_cast<std::size_t>(out_pos)] = static_cast<Index>(to_node);
      out.csr_edge[static_cast<std::size_t>(out_pos)] = edge;
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << "[check_sources] outgoing adjacency built in "
            << std::fixed << std::setprecision(3) << seconds
            << " sec\n";
  return out;
}

void print_source_degree_stats(const HostCsrF32& graph,
                               const OutgoingCsr& outgoing,
                               const std::vector<int>& sources) {
  std::uint64_t zero_out = 0;
  std::uint64_t zero_in = 0;
  Offset max_out = 0;
  Offset max_in = 0;
  long double total_out = 0.0;
  long double total_in = 0.0;

  for (const int source : sources) {
    const Offset out_degree =
        outgoing.rowptr[static_cast<std::size_t>(source + 1)] -
        outgoing.rowptr[static_cast<std::size_t>(source)];
    const Offset in_degree =
        graph.rowptr[static_cast<std::size_t>(source + 1)] -
        graph.rowptr[static_cast<std::size_t>(source)];
    if (out_degree == 0) ++zero_out;
    if (in_degree == 0) ++zero_in;
    max_out = std::max(max_out, out_degree);
    max_in = std::max(max_in, in_degree);
    total_out += static_cast<long double>(out_degree);
    total_in += static_cast<long double>(in_degree);
  }

  const long double denom =
      sources.empty() ? 1.0L : static_cast<long double>(sources.size());
  std::cout << "[check_sources] source degree stats\n"
            << "[check_sources]   zero outgoing-degree sources: "
            << zero_out << "/" << sources.size() << '\n'
            << "[check_sources]   zero incoming-degree sources: "
            << zero_in << "/" << sources.size() << '\n'
            << "[check_sources]   avg outgoing degree: "
            << std::fixed << std::setprecision(3)
            << static_cast<double>(total_out / denom)
            << ", max outgoing degree: " << max_out << '\n'
            << "[check_sources]   avg incoming degree: "
            << static_cast<double>(total_in / denom)
            << ", max incoming degree: " << max_in << '\n';
}

void add_connector_source_relation(std::vector<ConnectorSourceRelation>* relations,
                                   int connector,
                                   int source_index,
                                   int source,
                                   bool source_to_connector,
                                   bool connector_to_source) {
  if (connector == source) return;
  ConnectorSourceRelation relation;
  relation.connector = connector;
  relation.source_index = source_index;
  relation.source = source;
  relation.source_to_connector = source_to_connector;
  relation.connector_to_source = connector_to_source;
  relations->push_back(relation);
}

Hop2ConnectorStats classify_hop2_connector_patterns(const HostCsrF32& graph,
                                                    const OutgoingCsr& outgoing,
                                                    const std::vector<int>& sources,
                                                    const Options& options) {
  std::cout << "[check_sources] enumerating hop-2 source-connector-source sets\n";
  const auto start = std::chrono::steady_clock::now();

  std::vector<ConnectorSourceRelation> relations;
  for (std::size_t source_i = 0; source_i < sources.size(); ++source_i) {
    const int source = sources[source_i];

    const Offset outgoing_begin = outgoing.rowptr[static_cast<std::size_t>(source)];
    const Offset outgoing_end = outgoing.rowptr[static_cast<std::size_t>(source + 1)];
    for (Offset p = outgoing_begin; p < outgoing_end; ++p) {
      const int connector = outgoing.to[static_cast<std::size_t>(p)];
      add_connector_source_relation(&relations,
                                    connector,
                                    static_cast<int>(source_i),
                                    source,
                                    true,
                                    false);
    }

    const Offset incoming_begin = graph.rowptr[static_cast<std::size_t>(source)];
    const Offset incoming_end = graph.rowptr[static_cast<std::size_t>(source + 1)];
    for (Offset edge = incoming_begin; edge < incoming_end; ++edge) {
      const int connector = graph.colind[static_cast<std::size_t>(edge)];
      add_connector_source_relation(&relations,
                                    connector,
                                    static_cast<int>(source_i),
                                    source,
                                    false,
                                    true);
    }

    if (options.progress_every != 0 &&
        ((source_i + 1) % options.progress_every == 0 ||
         source_i + 1 == sources.size())) {
      const auto now = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(now - start).count();
      std::cout << "[check_sources] relation scan progress: "
                << (source_i + 1) << "/" << sources.size()
                << " sources, source-connector adjacencies so far="
                << relations.size()
                << ", elapsed=" << std::fixed << std::setprecision(2)
                << seconds << " sec\n";
    }
  }

  std::sort(relations.begin(),
            relations.end(),
            [](const ConnectorSourceRelation& lhs,
               const ConnectorSourceRelation& rhs) {
              if (lhs.connector != rhs.connector) {
                return lhs.connector < rhs.connector;
              }
              return lhs.source_index < rhs.source_index;
            });

  std::vector<ConnectorSourceRelation> merged;
  merged.reserve(relations.size());
  for (const ConnectorSourceRelation& relation : relations) {
    if (merged.empty() ||
        merged.back().connector != relation.connector ||
        merged.back().source_index != relation.source_index) {
      merged.push_back(relation);
      continue;
    }
    merged.back().source_to_connector =
        merged.back().source_to_connector || relation.source_to_connector;
    merged.back().connector_to_source =
        merged.back().connector_to_source || relation.connector_to_source;
  }

  Hop2ConnectorStats stats;
  stats.source_connector_relations = merged.size();
  const bool unlimited_print = options.max_sets_print == 0;

  auto print_orientation = [&](const ConnectorSourceRelation& source1,
                               const ConnectorSourceRelation& source2,
                               const char* orientation) {
    if (!unlimited_print && stats.printed_sets >= options.max_sets_print) {
      return;
    }
    std::cout << "[check_sources] hop2_set"
              << " source1_idx=" << source1.source_index
              << " source1=" << source1.source
              << " connector=" << source1.connector
              << " source2_idx=" << source2.source_index
              << " source2=" << source2.source
              << " direction=\"" << orientation << "\"\n";
    ++stats.printed_sets;
  };

  for (std::size_t begin = 0; begin < merged.size();) {
    std::size_t end = begin + 1;
    while (end < merged.size() &&
           merged[end].connector == merged[begin].connector) {
      ++end;
    }

    const std::size_t adjacent_source_count = end - begin;
    if (adjacent_source_count >= 2) {
      ++stats.connector_nodes_with_two_sources;
      stats.connector_mediated_pairs +=
          (static_cast<std::uint64_t>(adjacent_source_count) *
           static_cast<std::uint64_t>(adjacent_source_count - 1)) /
          2;

      for (std::size_t i = begin; i < end; ++i) {
        for (std::size_t j = i + 1; j < end; ++j) {
          const ConnectorSourceRelation& source1 = merged[i];
          const ConnectorSourceRelation& source2 = merged[j];

          if (source1.source_to_connector && source2.connector_to_source) {
            ++stats.source1_to_connector_to_source2;
            print_orientation(source1, source2, "source1 -> connector -> source2");
          }
          if (source1.connector_to_source && source2.connector_to_source) {
            ++stats.source1_from_connector_to_source2;
            print_orientation(source1, source2, "source1 <- connector -> source2");
          }
          if (source1.source_to_connector && source2.source_to_connector) {
            ++stats.source1_to_connector_from_source2;
            print_orientation(source1, source2, "source1 -> connector <- source2");
          }
          if (source1.connector_to_source && source2.source_to_connector) {
            ++stats.source1_from_connector_from_source2;
            print_orientation(source1, source2, "source1 <- connector <- source2");
          }
        }
      }
    }

    begin = end;
  }

  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << "[check_sources] hop-2 enumeration finished in "
            << std::fixed << std::setprecision(3) << seconds << " sec\n";
  return stats;
}

void print_hop2_connector_stats(const Hop2ConnectorStats& stats) {
  const std::uint64_t orientation_total =
      stats.source1_to_connector_to_source2 +
      stats.source1_from_connector_to_source2 +
      stats.source1_to_connector_from_source2 +
      stats.source1_from_connector_from_source2;

  std::cout << "[check_sources] hop-2 source-connector-source summary\n"
            << "[check_sources]   source-connector adjacencies scanned: "
            << stats.source_connector_relations << '\n'
            << "[check_sources]   connector nodes adjacent to >=2 sources: "
            << stats.connector_nodes_with_two_sources << '\n'
            << "[check_sources]   undirected source1-connector-source2 triples: "
            << stats.connector_mediated_pairs << '\n'
            << "[check_sources]   orientation instances counted: "
            << orientation_total << '\n'
            << "[check_sources]   source1 -> connector -> source2: "
            << stats.source1_to_connector_to_source2 << '\n'
            << "[check_sources]   source1 <- connector -> source2: "
            << stats.source1_from_connector_to_source2 << '\n'
            << "[check_sources]   source1 -> connector <- source2: "
            << stats.source1_to_connector_from_source2 << '\n'
            << "[check_sources]   source1 <- connector <- source2: "
            << stats.source1_from_connector_from_source2 << '\n'
            << "[check_sources]   printed orientation lines: "
            << stats.printed_sets << '\n'
            << "[check_sources]   note: a bidirectional source-connector edge can make one triple support multiple orientations\n";
}

std::size_t parse_size(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
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
      << "  " << program << " <graph.csrbin> <metadata.ifmeta.bin> [options]\n\n"
      << "Options:\n"
      << "  --net-limit <n>              Read only the first n route requests.\n"
      << "  --source-limit <n>           Analyze only the first n unique sources.\n"
      << "  --max-sets-print <n>         Print at most n orientation lines; 0 = no cap. Default: 0.\n"
      << "  --max-pairs-print <n>        Alias for --max-sets-print.\n"
      << "  --progress-every <n>         Print progress every n sources. Default: 25.\n"
      << "  --help                       Print this message.\n";
}

Options parse_options(int argc,
                      char** argv,
                      std::filesystem::path* csr_path,
                      std::filesystem::path* metadata_path) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--net-limit") {
      options.net_limit = parse_size(require_value(&i, argc, argv, "--net-limit"), "--net-limit");
    } else if (arg == "--source-limit") {
      options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (arg == "--max-sets-print") {
      options.max_sets_print =
          parse_size(require_value(&i, argc, argv, "--max-sets-print"),
                     "--max-sets-print");
    } else if (arg == "--max-pairs-print") {
      options.max_sets_print =
          parse_size(require_value(&i, argc, argv, "--max-pairs-print"),
                     "--max-pairs-print");
    } else if (arg == "--progress-every") {
      options.progress_every =
          parse_size(require_value(&i, argc, argv, "--progress-every"), "--progress-every");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (csr_path->empty()) {
      *csr_path = arg;
    } else if (metadata_path->empty()) {
      *metadata_path = arg;
    } else {
      throw std::runtime_error("too many positional arguments");
    }
  }

  if (csr_path->empty() || metadata_path->empty()) {
    throw std::runtime_error("expected <graph.csrbin> <metadata.ifmeta.bin>");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path csr_path;
    std::filesystem::path metadata_path;
    const Options options = parse_options(argc, argv, &csr_path, &metadata_path);

    std::cout << "[check_sources] loading CSR: " << csr_path.string() << '\n';
    HostCsrF32 graph = load_csrbin(csr_path);
    std::cout << "[check_sources] graph rows=" << graph.rows
              << " nnz=" << graph.nnz
              << " orientation=incoming(row v, col u means u -> v)\n";

    std::cout << "[check_sources] loading metadata sources: "
              << metadata_path.string() << '\n';
    std::uint64_t total_route_requests = 0;
    std::vector<int> sources =
        load_unique_sources_from_metadata(metadata_path,
                                          graph,
                                          options,
                                          &total_route_requests);
    std::cout << "[check_sources] metadata route_requests="
              << total_route_requests
              << " unique_sources_to_scan=" << sources.size()
              << " hop_mode=exactly_2_undirected_source_connector_source\n";

    if (sources.empty()) {
      std::cout << "[check_sources] no sources found; nothing to scan\n";
      return 0;
    }

    const OutgoingCsr outgoing = build_outgoing_csr(graph);
    print_source_degree_stats(graph, outgoing, sources);
    const Hop2ConnectorStats hop2_stats =
        classify_hop2_connector_patterns(graph, outgoing, sources, options);
    print_hop2_connector_stats(hop2_stats);

    std::cout << "[check_sources] summary\n";
    std::cout << "[check_sources]   sources scanned: " << sources.size() << '\n';
    std::cout << "[check_sources]   exact hop-2 source-connector-source triples: "
              << hop2_stats.connector_mediated_pairs << '\n';
    if (hop2_stats.connector_mediated_pairs == 0) {
      std::cout << "[check_sources] no source-connector-source triples were found\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[check_sources] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
