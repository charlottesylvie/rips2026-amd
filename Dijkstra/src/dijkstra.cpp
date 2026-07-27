// CPU reference Dijkstra implementation for the outgoing RIPSCSR1 format.
//
// Build from rips2026-amd:
//   c++ -std=c++17 -O3 -Wall -Wextra -Wpedantic \
//     Dijkstra/src/dijkstra.cpp -o dijkstra
//
// Run:
//   ./dijkstra graph.csrbin 0 --stats-number 100
//   ./dijkstra graph.csrbin 0 --print
//   ./dijkstra graph.csrbin 0 --print paths.jsonl

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rips_dijkstra {
namespace {

using Clock = std::chrono::steady_clock;
using Offset = std::int64_t;
using Index = std::int32_t;

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t kLegacyCsrVersion = 1;
constexpr std::uint64_t kCurrentCsrVersion = 2;
constexpr std::uint64_t kOutgoingEdgeOrientation = 2;
constexpr std::size_t kMaximumPrintedPaths = 1000;

static_assert(sizeof(Offset) == 8, "RIPS CSR offsets must be 64-bit");
static_assert(sizeof(Index) == 4, "RIPS CSR indices must be 32-bit");
static_assert(sizeof(float) == 4, "RIPS CSR weights must be 32-bit floats");

struct CsrGraph {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

struct Options {
  std::filesystem::path csr_path;
  Index source = -1;
  bool stats_number_set = false;
  std::uint64_t stats_number = 0;
  bool print_paths = false;
  std::filesystem::path paths_output_path;
};

struct Statistics {
  std::uint64_t phases = 0;
  std::uint64_t vertices_settled = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxations_attempted = 0;
  std::uint64_t successful_distance_updates = 0;
  std::chrono::nanoseconds predicate_evaluation_time{0};
  std::chrono::nanoseconds relaxation_time{0};
  std::uint64_t gpu_kernel_launches = 0;
};

struct DijkstraResult {
  // Match CongestionFreeRouting's HostCsrF32 distance arithmetic exactly so
  // this executable can serve as a CPU correctness reference for its kernels.
  std::vector<float> distances;
  std::vector<Index> predecessor_nodes;
  std::vector<Offset> predecessor_edges;
  std::vector<std::uint8_t> settled;
};

struct ReconstructedPath {
  std::vector<Index> nodes;
  std::vector<Offset> csr_edges;
};

template <typename Rep, typename Period>
double milliseconds(std::chrono::duration<Rep, Period> duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

bool starts_with_dash(const std::string& value) {
  return !value.empty() && value.front() == '-';
}

std::uint64_t parse_positive_u64(const std::string& text,
                                 const char* option_name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string(option_name) +
                                " requires a positive integer");
  }

  std::size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option_name) +
                                " requires a positive integer");
  }
  if (parsed != text.size() || value == 0) {
    throw std::invalid_argument(std::string(option_name) +
                                " requires a positive integer");
  }
  return static_cast<std::uint64_t>(value);
}

Index parse_source(const std::string& text) {
  std::size_t parsed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &parsed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument("source node must be a nonnegative integer");
  }
  if (parsed != text.size() || value < 0 ||
      value > static_cast<long long>(std::numeric_limits<Index>::max())) {
    throw std::invalid_argument("source node must be a nonnegative 32-bit integer");
  }
  return static_cast<Index>(value);
}

std::filesystem::path default_paths_output(
    const std::filesystem::path& csr_path) {
  std::filesystem::path output = csr_path;
  output += ".dijkstra.paths.jsonl";
  return output;
}

void print_usage(std::ostream& out, const char* program) {
  out << "Usage:\n"
      << "  " << program
      << " <graph.csrbin> <source-node> [options]\n\n"
      << "Options:\n"
      << "  --stats-number <phase>  Print a cumulative statistics snapshot after "
         "that phase.\n"
      << "  --print [path]          Write at most 1000 source-to-target path "
         "records as JSONL.\n"
      << "                          Default path: "
         "<graph.csrbin>.dijkstra.paths.jsonl\n"
      << "  -h, --help              Show this help message.\n";
}

Options parse_args(int argc, char** argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "expected a RIPS CSR file and a source node; use --help for usage");
  }

  Options options;
  options.csr_path = argv[1];
  options.source = parse_source(argv[2]);

  for (int arg = 3; arg < argc; ++arg) {
    const std::string option = argv[arg];
    if (option == "--stats-number") {
      if (options.stats_number_set) {
        throw std::invalid_argument("--stats-number may be specified only once");
      }
      if (++arg >= argc) {
        throw std::invalid_argument("--stats-number requires a phase number");
      }
      options.stats_number =
          parse_positive_u64(argv[arg], "--stats-number");
      options.stats_number_set = true;
    } else if (option == "--print") {
      if (options.print_paths) {
        throw std::invalid_argument("--print may be specified only once");
      }
      options.print_paths = true;
      if (arg + 1 < argc &&
          !starts_with_dash(std::string(argv[arg + 1]))) {
        options.paths_output_path = argv[++arg];
      }
    } else if (option == "-h" || option == "--help") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }

  if (options.print_paths && options.paths_output_path.empty()) {
    options.paths_output_path = default_paths_output(options.csr_path);
  }
  return options;
}

std::uint64_t read_u64(std::ifstream& input, const char* field_name) {
  std::uint64_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error(std::string("failed while reading ") + field_name);
  }
  return value;
}

template <typename T>
void read_array(std::ifstream& input,
                std::vector<T>& values,
                std::uint64_t count,
                const char* field_name) {
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(field_name) +
                              " count is too large for this host");
  }
  const std::size_t host_count = static_cast<std::size_t>(count);
  if (host_count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(field_name) +
                              " byte count overflows");
  }
  const std::size_t byte_count = host_count * sizeof(T);
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(std::string(field_name) +
                              " byte count exceeds stream limits");
  }

  values.resize(host_count);
  if (values.empty()) {
    return;
  }
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(byte_count));
  if (!input) {
    throw std::runtime_error(std::string("failed while reading ") + field_name);
  }
}

void validate_csr_structure(const CsrGraph& graph) {
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
  for (Index destination : graph.colind) {
    if (destination < 0 ||
        static_cast<Offset>(destination) >= graph.cols) {
      throw std::runtime_error(
          "CSR colind contains an out-of-range vertex");
    }
  }
}

CsrGraph load_csr(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(kCsrMagic)] = {};
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, kCsrMagic, sizeof(kCsrMagic)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(input, "CSR format version");
  const std::uint64_t orientation = read_u64(input, "CSR orientation");
  if (version != kLegacyCsrVersion && version != kCurrentCsrVersion) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != kOutgoingEdgeOrientation) {
    throw std::runtime_error(
        "unsupported CSR orientation; expected outgoing orientation 2");
  }

  if (version == kCurrentCsrVersion) {
    const std::uint64_t pair_id_high =
        read_u64(input, "CSR artifact pair id high");
    const std::uint64_t pair_id_low =
        read_u64(input, "CSR artifact pair id low");
    if (pair_id_high == 0 && pair_id_low == 0) {
      throw std::runtime_error("CSR artifact pair id must not be zero");
    }
  }

  const std::uint64_t rows = read_u64(input, "CSR row count");
  const std::uint64_t cols = read_u64(input, "CSR column count");
  (void)read_u64(input, "declared edge count");
  (void)read_u64(input, "loaded edge count");
  const std::uint64_t nnz = read_u64(input, "CSR nnz");
  const std::uint64_t rowptr_count =
      read_u64(input, "CSR rowptr count");
  const std::uint64_t colind_count =
      read_u64(input, "CSR colind count");
  const std::uint64_t values_count =
      read_u64(input, "CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows >
          static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      rows >
          static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      nnz >
          static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this implementation");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz ||
      values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  CsrGraph graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(input, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(input, graph.colind, colind_count, "CSR colind");
  read_array(input, graph.values, values_count, "CSR values");

  // The caller guarantees finite, nonnegative weights. Structural validation
  // remains necessary because it protects all CSR array accesses.
  validate_csr_structure(graph);
  return graph;
}

void print_statistics(const Statistics& statistics,
                      const std::string& label,
                      Clock::duration preprocessing_time,
                      Clock::duration transfer_time,
                      Clock::duration end_to_end_time) {
  const double predicate_ms =
      milliseconds(statistics.predicate_evaluation_time);
  const double relaxation_ms = milliseconds(statistics.relaxation_time);

  std::cout << "\n=== Dijkstra statistics (" << label << ") ===\n"
            << "Number of phases: " << statistics.phases << '\n'
            << "Vertices settled per phase: 1"
            << " (one vertex in each completed phase)\n"
            << "Vertices settled total: " << statistics.vertices_settled
            << '\n'
            << "Edges examined: " << statistics.edges_examined << '\n'
            << "Relaxations attempted: "
            << statistics.relaxations_attempted << '\n'
            << "Successful distance updates: "
            << statistics.successful_distance_updates << '\n'
            << std::fixed << std::setprecision(6)
            << "Predicate-evaluation time (ms): " << predicate_ms << '\n'
            << "Relaxation time (ms): " << relaxation_ms
            << " (successful distance updates only)\n";
  if (predicate_ms > 0.0) {
    std::cout << "Relaxation/predicate time ratio: "
              << (relaxation_ms / predicate_ms) << '\n';
  } else {
    std::cout << "Relaxation/predicate time ratio: n/a\n";
  }
  std::cout << "GPU kernel-launch count: "
            << statistics.gpu_kernel_launches << '\n'
            << "Preprocessing time (ms): "
            << milliseconds(preprocessing_time) << '\n'
            << "Transfer time (ms): " << milliseconds(transfer_time)
            << " (CPU-only; no host/device transfer)\n"
            << "End-to-end time (ms): "
            << milliseconds(end_to_end_time) << '\n';
}

ReconstructedPath reconstruct_path(const CsrGraph& graph,
                                   const DijkstraResult& result,
                                   Index source,
                                   Index target) {
  ReconstructedPath path;
  if (result.settled[static_cast<std::size_t>(target)] == 0) {
    return path;
  }

  Index current = target;
  path.nodes.push_back(current);
  for (Offset hop = 0; hop < graph.rows && current != source; ++hop) {
    const std::size_t current_index = static_cast<std::size_t>(current);
    const Index predecessor = result.predecessor_nodes[current_index];
    const Offset predecessor_edge = result.predecessor_edges[current_index];
    if (predecessor < 0 || predecessor_edge < 0 ||
        static_cast<Offset>(predecessor) >= graph.rows ||
        predecessor_edge <
            graph.rowptr[static_cast<std::size_t>(predecessor)] ||
        predecessor_edge >=
            graph.rowptr[static_cast<std::size_t>(predecessor + 1)] ||
        graph.colind[static_cast<std::size_t>(predecessor_edge)] != current) {
      throw std::runtime_error(
          "could not reconstruct a valid predecessor path");
    }
    path.csr_edges.push_back(predecessor_edge);
    current = predecessor;
    path.nodes.push_back(current);
  }

  if (current != source) {
    throw std::runtime_error(
        "predecessor path exceeded the graph vertex count");
  }
  std::reverse(path.nodes.begin(), path.nodes.end());
  std::reverse(path.csr_edges.begin(), path.csr_edges.end());
  return path;
}

void write_path_record(std::ostream& output,
                       Index source,
                       Index target,
                       bool reached,
                       float distance,
                       const ReconstructedPath& path) {
  output << "{\"type\":\"path\""
         << ",\"net\":\"\""
         << ",\"net_index\":0"
         << ",\"logical_net_index\":null"
         << ",\"source\":" << source
         << ",\"target\":" << target
         << ",\"reached\":" << (reached ? "true" : "false");
  if (reached) {
    output << ",\"distance\":" << std::setprecision(9) << distance;
  } else {
    output << ",\"distance\":null";
  }

  output << ",\"nodes\":[";
  for (std::size_t index = 0; index < path.nodes.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << path.nodes[index];
  }
  output << "],\"csr_edges\":[";
  for (std::size_t index = 0; index < path.csr_edges.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << path.csr_edges[index];
  }
  output << "]}\n";
}

std::size_t write_paths_jsonl(const std::filesystem::path& output_path,
                              const CsrGraph& graph,
                              const DijkstraResult& result,
                              Index source) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("could not open paths output: " +
                             output_path.string());
  }

  const std::size_t possible_paths =
      static_cast<std::size_t>(graph.rows - 1);
  const std::size_t paths_to_write =
      std::min(kMaximumPrintedPaths, possible_paths);
  output << "{\"type\":\"metadata\""
         << ",\"format\":\"rips-sssp-paths-v1\""
         << ",\"producer\":\"dijkstra\""
         << ",\"node_count\":" << graph.rows
         << ",\"edge_count\":" << graph.nnz
         << ",\"route_request_count\":1"
         << ",\"selected_source_count\":1"
         << ",\"selected_query_count\":" << paths_to_write
         << ",\"edge_orientation\":\"outgoing\""
         << ",\"description\":\"row u stores directed edges u -> v\""
         << "}\n";

  std::size_t paths_written = 0;
  for (Offset raw_target = 0;
       raw_target < graph.rows && paths_written < paths_to_write;
       ++raw_target) {
    const Index target = static_cast<Index>(raw_target);
    if (target == source) {
      continue;
    }
    const bool reached =
        result.settled[static_cast<std::size_t>(target)] != 0;
    const ReconstructedPath path =
        reached ? reconstruct_path(graph, result, source, target)
                : ReconstructedPath{};
    write_path_record(output,
                      source,
                      target,
                      reached,
                      result.distances[static_cast<std::size_t>(target)],
                      path);
    ++paths_written;
  }

  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing paths output: " +
                             output_path.string());
  }
  return paths_written;
}

int run(const Options& options, Clock::time_point end_to_end_begin) {
  const auto preprocessing_begin = Clock::now();
  const CsrGraph graph = load_csr(options.csr_path);
  if (static_cast<Offset>(options.source) >= graph.rows) {
    throw std::out_of_range("source node is outside the CSR graph");
  }

  const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
  DijkstraResult result;
  result.distances.assign(vertex_count,
                          std::numeric_limits<float>::infinity());
  result.predecessor_nodes.assign(vertex_count, static_cast<Index>(-1));
  result.predecessor_edges.assign(vertex_count, static_cast<Offset>(-1));
  result.settled.assign(vertex_count, 0);

  using HeapEntry = std::pair<float, Index>;
  std::priority_queue<HeapEntry,
                      std::vector<HeapEntry>,
                      std::greater<HeapEntry>>
      heap;
  result.distances[static_cast<std::size_t>(options.source)] = 0.0f;
  heap.emplace(0.0f, options.source);
  const auto preprocessing_end = Clock::now();
  const Clock::duration preprocessing_time =
      preprocessing_end - preprocessing_begin;
  const Clock::duration transfer_time = Clock::duration::zero();

  Statistics statistics;
  bool requested_snapshot_printed = false;

  while (!heap.empty()) {
    const HeapEntry entry = heap.top();
    heap.pop();
    const float queued_distance = entry.first;
    const Index vertex = entry.second;
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);

    // Lazy duplicate entries remain in the binary heap after a better distance
    // is inserted. They are discarded without creating a Dijkstra phase.
    if (result.settled[vertex_index] != 0 ||
        queued_distance != result.distances[vertex_index]) {
      continue;
    }

    result.settled[vertex_index] = 1;
    ++statistics.phases;
    ++statistics.vertices_settled;

    const Offset edge_begin = graph.rowptr[vertex_index];
    const Offset edge_end = graph.rowptr[vertex_index + 1];
    for (Offset edge = edge_begin; edge < edge_end; ++edge) {
      ++statistics.edges_examined;
      ++statistics.relaxations_attempted;

      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const Index destination = graph.colind[edge_index];
      const std::size_t destination_index =
          static_cast<std::size_t>(destination);

      const auto predicate_begin = Clock::now();
      const float candidate_distance =
          queued_distance + graph.values[edge_index];
      const bool should_update =
          candidate_distance < result.distances[destination_index];
      const auto predicate_end = Clock::now();
      statistics.predicate_evaluation_time +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              predicate_end - predicate_begin);

      if (should_update) {
        const auto relaxation_begin = Clock::now();
        result.distances[destination_index] = candidate_distance;
        result.predecessor_nodes[destination_index] = vertex;
        result.predecessor_edges[destination_index] = edge;
        heap.emplace(candidate_distance, destination);
        ++statistics.successful_distance_updates;
        const auto relaxation_end = Clock::now();
        statistics.relaxation_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                relaxation_end - relaxation_begin);
      }
    }

    if (options.stats_number_set &&
        statistics.phases == options.stats_number) {
      print_statistics(statistics,
                       "phase " + std::to_string(options.stats_number) +
                           " snapshot",
                       preprocessing_time,
                       transfer_time,
                       Clock::now() - end_to_end_begin);
      requested_snapshot_printed = true;
    }
  }

  std::size_t paths_written = 0;
  if (options.print_paths) {
    paths_written = write_paths_jsonl(options.paths_output_path,
                                      graph,
                                      result,
                                      options.source);
  }
  const Clock::duration end_to_end_time =
      Clock::now() - end_to_end_begin;

  if (options.stats_number_set && !requested_snapshot_printed) {
    std::cout << "\nRequested statistics phase " << options.stats_number
              << " was not reached; the run completed after "
              << statistics.phases << " phase(s).\n";
  }
  if (options.print_paths) {
    std::cout << "\nWrote " << paths_written
              << " path record(s) to "
              << options.paths_output_path.string() << '\n';
  }
  print_statistics(statistics,
                   "final",
                   preprocessing_time,
                   transfer_time,
                   end_to_end_time);
  return 0;
}

}  // namespace
}  // namespace rips_dijkstra

int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" ||
       std::string(argv[1]) == "--help")) {
    rips_dijkstra::print_usage(std::cout, argv[0]);
    return 0;
  }

  const auto end_to_end_begin = rips_dijkstra::Clock::now();
  try {
    const rips_dijkstra::Options options =
        rips_dijkstra::parse_args(argc, argv);
    return rips_dijkstra::run(options, end_to_end_begin);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
