// CPU-only phased Dijkstra SSSP using the static and simple predicates from
// Kainer and Traff, "More Parallelism in Dijkstra's Single-Source Shortest
// Path Algorithm".
//
// Build from rips2026-amd:
//   g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror Dijkstra/preds_CPU.cpp -o preds_CPU
//
// Run:
//   ./preds_CPU graph.csrbin 0 --predicate IN_SIMPLE
//   ./preds_CPU graph.csrbin 0 --predicate IN_STATIC_OR_OUT_STATIC --print

#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rips_predicates {
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

enum class PredicateMode {
  kInSimple,
  kOutSimple,
  kInSimpleOrOutSimple,
  kInStatic,
  kOutStatic,
  kInStaticOrOutStatic,
};

enum class VertexState : std::uint8_t {
  kUnexplored,
  kFringe,
  kSettled,
};

struct CsrGraph {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

struct IncomingCsr {
  std::vector<Offset> rowptr;
  std::vector<Index> sources;
  std::vector<Offset> outgoing_edge_ids;
};

struct GraphAuxiliary {
  IncomingCsr incoming;
  std::vector<Index> edge_sources;
  std::vector<float> min_in_static;
  std::vector<float> min_out_static;
  std::vector<std::vector<Offset>> incoming_edges_descending;
  std::vector<std::vector<Offset>> outgoing_edges_descending;
};

struct Options {
  std::filesystem::path csr_path;
  Index source = -1;
  bool predicate_set = false;
  PredicateMode predicate = PredicateMode::kInSimple;
  bool stats_number_set = false;
  std::uint64_t stats_number = 0;
  bool print_paths = false;
  std::filesystem::path paths_output_path;
};

struct Statistics {
  std::uint64_t phases = 0;
  std::uint64_t vertices_settled = 0;
  std::uint64_t vertices_settled_current_phase = 0;
  std::vector<std::uint64_t> vertices_settled_per_phase;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxations_attempted = 0;
  std::uint64_t successful_distance_updates = 0;
  std::chrono::nanoseconds predicate_evaluation_time{0};
  std::chrono::nanoseconds relaxation_time{0};
  Clock::duration elapsed_time = Clock::duration::zero();
  std::uint64_t gpu_kernel_launches = 0;
};

struct DijkstraResult {
  std::vector<float> distances;
  std::vector<Index> predecessor_nodes;
  std::vector<Offset> predecessor_edges;
  std::vector<VertexState> states;
};

struct ReconstructedPath {
  std::vector<Index> nodes;
  std::vector<Offset> csr_edges;
};

struct HeapEntry {
  float key = 0.0f;
  Index vertex = -1;
};

struct HeapEntryGreater {
  bool operator()(const HeapEntry& left,
                  const HeapEntry& right) const noexcept {
    if (left.key != right.key) {
      return left.key > right.key;
    }
    return left.vertex > right.vertex;
  }
};

using FibonacciHeap =
    boost::heap::fibonacci_heap<
        HeapEntry,
        boost::heap::compare<HeapEntryGreater>>;

class MutableMinQueue {
 public:
  explicit MutableMinQueue(std::size_t vertex_count)
      : handles_(vertex_count) {}

  bool empty() const {
    return heap_.empty();
  }

  const HeapEntry& top() const {
    if (heap_.empty()) {
      throw std::logic_error("attempted to inspect an empty priority queue");
    }
    return heap_.top();
  }

  bool contains(Index vertex) const {
    return handles_.at(static_cast<std::size_t>(vertex)).has_value();
  }

  void insert(Index vertex, float key) {
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    if (handles_.at(vertex_index).has_value()) {
      throw std::logic_error("duplicate mutable-priority-queue insertion");
    }
    handles_[vertex_index] = heap_.push(HeapEntry{key, vertex});
  }

  void update(Index vertex, float key) {
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    std::optional<FibonacciHeap::handle_type>& handle =
        handles_.at(vertex_index);
    if (!handle.has_value()) {
      throw std::logic_error("mutable-priority-queue update without a handle");
    }
    // Boost's general update operation handles both increase-key and
    // decrease-key changes without introducing lazy duplicate entries.
    heap_.update(*handle, HeapEntry{key, vertex});
  }

  void erase(Index vertex) {
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    std::optional<FibonacciHeap::handle_type>& handle =
        handles_.at(vertex_index);
    if (!handle.has_value()) {
      return;
    }
    heap_.erase(*handle);
    handle.reset();
  }

  HeapEntry pop_min() {
    if (heap_.empty()) {
      throw std::logic_error("attempted to pop an empty priority queue");
    }
    const HeapEntry entry = heap_.top();
    heap_.pop();
    handles_.at(static_cast<std::size_t>(entry.vertex)).reset();
    return entry;
  }

 private:
  FibonacciHeap heap_;
  std::vector<std::optional<FibonacciHeap::handle_type>> handles_;
};

struct AlgorithmState {
  explicit AlgorithmState(std::size_t vertex_count)
      : distance_queue(vertex_count),
        in_queue(vertex_count),
        out_queue(vertex_count),
        selected(vertex_count, 0),
        affected(vertex_count, 0) {}

  DijkstraResult result;
  MutableMinQueue distance_queue;
  MutableMinQueue in_queue;
  MutableMinQueue out_queue;
  std::vector<float> min_in_simple;
  std::vector<float> min_out_simple;
  std::vector<std::uint8_t> selected;
  std::vector<std::uint8_t> affected;
};

template <typename Rep, typename Period>
double milliseconds(std::chrono::duration<Rep, Period> duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

const char* predicate_mode_name(PredicateMode mode) {
  switch (mode) {
    case PredicateMode::kInSimple:
      return "IN_SIMPLE";
    case PredicateMode::kOutSimple:
      return "OUT_SIMPLE";
    case PredicateMode::kInSimpleOrOutSimple:
      return "IN_SIMPLE_OR_OUT_SIMPLE";
    case PredicateMode::kInStatic:
      return "IN_STATIC";
    case PredicateMode::kOutStatic:
      return "OUT_STATIC";
    case PredicateMode::kInStaticOrOutStatic:
      return "IN_STATIC_OR_OUT_STATIC";
  }
  throw std::logic_error("unknown predicate mode");
}

PredicateMode parse_predicate_mode(const std::string& text) {
  if (text == "IN_SIMPLE") {
    return PredicateMode::kInSimple;
  }
  if (text == "OUT_SIMPLE") {
    return PredicateMode::kOutSimple;
  }
  if (text == "IN_SIMPLE_OR_OUT_SIMPLE") {
    return PredicateMode::kInSimpleOrOutSimple;
  }
  if (text == "IN_STATIC") {
    return PredicateMode::kInStatic;
  }
  if (text == "OUT_STATIC") {
    return PredicateMode::kOutStatic;
  }
  if (text == "IN_STATIC_OR_OUT_STATIC") {
    return PredicateMode::kInStaticOrOutStatic;
  }
  throw std::invalid_argument(
      "invalid predicate mode '" + text +
      "'; expected one of IN_SIMPLE, OUT_SIMPLE, "
      "IN_SIMPLE_OR_OUT_SIMPLE, IN_STATIC, OUT_STATIC, "
      "IN_STATIC_OR_OUT_STATIC");
}

bool uses_in_predicate(PredicateMode mode) {
  return mode == PredicateMode::kInSimple ||
         mode == PredicateMode::kInSimpleOrOutSimple ||
         mode == PredicateMode::kInStatic ||
         mode == PredicateMode::kInStaticOrOutStatic;
}

bool uses_out_predicate(PredicateMode mode) {
  return mode == PredicateMode::kOutSimple ||
         mode == PredicateMode::kInSimpleOrOutSimple ||
         mode == PredicateMode::kOutStatic ||
         mode == PredicateMode::kInStaticOrOutStatic;
}

bool uses_simple_in_predicate(PredicateMode mode) {
  return mode == PredicateMode::kInSimple ||
         mode == PredicateMode::kInSimpleOrOutSimple;
}

bool uses_simple_out_predicate(PredicateMode mode) {
  return mode == PredicateMode::kOutSimple ||
         mode == PredicateMode::kInSimpleOrOutSimple;
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
    throw std::invalid_argument(
        "source node must be a nonnegative 32-bit integer");
  }
  return static_cast<Index>(value);
}

std::filesystem::path default_paths_output(
    const std::filesystem::path& csr_path,
    PredicateMode mode) {
  std::filesystem::path output = csr_path;
  output += ".preds_CPU.";
  output += predicate_mode_name(mode);
  output += ".paths.jsonl";
  return output;
}

void print_usage(std::ostream& out, const char* program) {
  out << "Usage:\n"
      << "  " << program
      << " <graph.csrbin> <source-node> --predicate <MODE> [options]\n\n"
      << "Predicate modes:\n"
      << "  IN_SIMPLE\n"
      << "  OUT_SIMPLE\n"
      << "  IN_SIMPLE_OR_OUT_SIMPLE\n"
      << "  IN_STATIC\n"
      << "  OUT_STATIC\n"
      << "  IN_STATIC_OR_OUT_STATIC\n\n"
      << "Options:\n"
      << "  --predicate <MODE>      Select the required predicate mode.\n"
      << "  --stats-number <phase> Print a cumulative statistics snapshot "
         "after that phase.\n"
      << "  --print [path]         Write at most 1000 source-to-target path "
         "records as JSONL.\n"
      << "                         Default path: "
         "<graph.csrbin>.preds_CPU.<predicate-mode>.paths.jsonl\n"
      << "  -h, --help             Show this help message.\n";
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
    if (option == "--predicate") {
      if (options.predicate_set) {
        throw std::invalid_argument("--predicate may be specified only once");
      }
      if (++arg >= argc || starts_with_dash(std::string(argv[arg]))) {
        throw std::invalid_argument("--predicate requires a mode");
      }
      options.predicate = parse_predicate_mode(argv[arg]);
      options.predicate_set = true;
    } else if (option == "--stats-number") {
      if (options.stats_number_set) {
        throw std::invalid_argument(
            "--stats-number may be specified only once");
      }
      if (++arg >= argc) {
        throw std::invalid_argument(
            "--stats-number requires a phase number");
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

  if (!options.predicate_set) {
    throw std::invalid_argument("--predicate <MODE> is required");
  }
  if (options.print_paths && options.paths_output_path.empty()) {
    options.paths_output_path =
        default_paths_output(options.csr_path, options.predicate);
  }
  return options;
}

std::uint64_t read_u64(std::ifstream& input, const char* field_name) {
  std::uint64_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error(std::string("failed while reading ") +
                             field_name);
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
    throw std::runtime_error(std::string("failed while reading ") +
                             field_name);
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

  // Match the reference loader: callers guarantee finite, nonnegative
  // weights, while the executable validates all structure used for indexing.
  validate_csr_structure(graph);
  return graph;
}

GraphAuxiliary build_graph_auxiliary(const CsrGraph& graph,
                                     PredicateMode mode) {
  const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
  const std::size_t edge_count = static_cast<std::size_t>(graph.nnz);
  const float infinity = std::numeric_limits<float>::infinity();

  GraphAuxiliary auxiliary;
  auxiliary.incoming.rowptr.assign(vertex_count + 1, 0);
  auxiliary.incoming.sources.resize(edge_count);
  auxiliary.incoming.outgoing_edge_ids.resize(edge_count);
  auxiliary.edge_sources.resize(edge_count);
  auxiliary.min_in_static.assign(vertex_count, infinity);
  auxiliary.min_out_static.assign(vertex_count, infinity);

  for (Offset raw_source = 0; raw_source < graph.rows; ++raw_source) {
    const Index source = static_cast<Index>(raw_source);
    const std::size_t source_index = static_cast<std::size_t>(source);
    const Offset begin = graph.rowptr[source_index];
    const Offset end = graph.rowptr[source_index + 1];
    for (Offset edge = begin; edge < end; ++edge) {
      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const Index destination = graph.colind[edge_index];
      const std::size_t destination_index =
          static_cast<std::size_t>(destination);
      const float weight = graph.values[edge_index];
      auxiliary.edge_sources[edge_index] = source;
      ++auxiliary.incoming.rowptr[destination_index + 1];
      auxiliary.min_out_static[source_index] =
          std::min(auxiliary.min_out_static[source_index], weight);
      auxiliary.min_in_static[destination_index] =
          std::min(auxiliary.min_in_static[destination_index], weight);
    }
  }

  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    auxiliary.incoming.rowptr[vertex + 1] +=
        auxiliary.incoming.rowptr[vertex];
  }
  std::vector<Offset> incoming_cursor = auxiliary.incoming.rowptr;
  for (Offset raw_source = 0; raw_source < graph.rows; ++raw_source) {
    const Index source = static_cast<Index>(raw_source);
    const std::size_t source_index = static_cast<std::size_t>(source);
    const Offset begin = graph.rowptr[source_index];
    const Offset end = graph.rowptr[source_index + 1];
    for (Offset edge = begin; edge < end; ++edge) {
      const Index destination =
          graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t destination_index =
          static_cast<std::size_t>(destination);
      const Offset position = incoming_cursor[destination_index]++;
      const std::size_t position_index =
          static_cast<std::size_t>(position);
      auxiliary.incoming.sources[position_index] = source;
      auxiliary.incoming.outgoing_edge_ids[position_index] = edge;
    }
  }

  const auto edge_descending =
      [&graph](Offset left_edge, Offset right_edge) {
        const float left_weight =
            graph.values[static_cast<std::size_t>(left_edge)];
        const float right_weight =
            graph.values[static_cast<std::size_t>(right_edge)];
        if (left_weight != right_weight) {
          return left_weight > right_weight;
        }
        return left_edge > right_edge;
      };

  if (uses_simple_in_predicate(mode)) {
    auxiliary.incoming_edges_descending.resize(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      std::vector<Offset>& edges =
          auxiliary.incoming_edges_descending[vertex];
      const Offset begin = auxiliary.incoming.rowptr[vertex];
      const Offset end = auxiliary.incoming.rowptr[vertex + 1];
      edges.reserve(static_cast<std::size_t>(end - begin));
      for (Offset position = begin; position < end; ++position) {
        edges.push_back(
            auxiliary.incoming.outgoing_edge_ids[
                static_cast<std::size_t>(position)]);
      }
      std::sort(edges.begin(), edges.end(), edge_descending);
    }
  }

  if (uses_simple_out_predicate(mode)) {
    auxiliary.outgoing_edges_descending.resize(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      std::vector<Offset>& edges =
          auxiliary.outgoing_edges_descending[vertex];
      const Offset begin = graph.rowptr[vertex];
      const Offset end = graph.rowptr[vertex + 1];
      edges.reserve(static_cast<std::size_t>(end - begin));
      for (Offset edge = begin; edge < end; ++edge) {
        edges.push_back(edge);
      }
      std::sort(edges.begin(), edges.end(), edge_descending);
    }
  }

  return auxiliary;
}

float current_in_minimum(const AlgorithmState& state,
                         const GraphAuxiliary& auxiliary,
                         PredicateMode mode,
                         Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  if (uses_simple_in_predicate(mode)) {
    return state.min_in_simple[vertex_index];
  }
  return auxiliary.min_in_static[vertex_index];
}

float current_out_minimum(const AlgorithmState& state,
                          const GraphAuxiliary& auxiliary,
                          PredicateMode mode,
                          Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  if (uses_simple_out_predicate(mode)) {
    return state.min_out_simple[vertex_index];
  }
  return auxiliary.min_out_static[vertex_index];
}

float in_queue_key(const AlgorithmState& state,
                   const GraphAuxiliary& auxiliary,
                   PredicateMode mode,
                   Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  return state.result.distances[vertex_index] -
         current_in_minimum(state, auxiliary, mode, vertex);
}

float out_queue_key(const AlgorithmState& state,
                    const GraphAuxiliary& auxiliary,
                    PredicateMode mode,
                    Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  return state.result.distances[vertex_index] +
         current_out_minimum(state, auxiliary, mode, vertex);
}

void insert_fringe_vertex(AlgorithmState& state,
                          const GraphAuxiliary& auxiliary,
                          PredicateMode mode,
                          Index vertex) {
  const float distance =
      state.result.distances[static_cast<std::size_t>(vertex)];
  state.distance_queue.insert(vertex, distance);
  if (uses_in_predicate(mode)) {
    state.in_queue.insert(
        vertex, in_queue_key(state, auxiliary, mode, vertex));
  }
  if (uses_out_predicate(mode)) {
    state.out_queue.insert(
        vertex, out_queue_key(state, auxiliary, mode, vertex));
  }
}

void update_fringe_vertex(AlgorithmState& state,
                          const GraphAuxiliary& auxiliary,
                          PredicateMode mode,
                          Index vertex) {
  const float distance =
      state.result.distances[static_cast<std::size_t>(vertex)];
  state.distance_queue.update(vertex, distance);
  if (uses_in_predicate(mode)) {
    state.in_queue.update(
        vertex, in_queue_key(state, auxiliary, mode, vertex));
  }
  if (uses_out_predicate(mode)) {
    state.out_queue.update(
        vertex, out_queue_key(state, auxiliary, mode, vertex));
  }
}

void remove_from_active_queues(AlgorithmState& state,
                               PredicateMode mode,
                               Index vertex) {
  state.distance_queue.erase(vertex);
  if (uses_in_predicate(mode)) {
    state.in_queue.erase(vertex);
  }
  if (uses_out_predicate(mode)) {
    state.out_queue.erase(vertex);
  }
}

std::vector<Index> identify_phase_batch(AlgorithmState& state,
                                        PredicateMode mode) {
  if (state.distance_queue.empty()) {
    return {};
  }

  // Both thresholds are captured before any queue is changed. The subsequent
  // extractions therefore inspect one frozen pre-phase fringe view.
  const float minimum_fringe_distance = state.distance_queue.top().key;
  float out_threshold = 0.0f;
  if (uses_out_predicate(mode)) {
    if (state.out_queue.empty()) {
      throw std::logic_error(
          "out-criterion queue is empty while the fringe is nonempty");
    }
    out_threshold = state.out_queue.top().key;
  }

  std::vector<Index> batch;
  const auto add_to_batch = [&state, &batch](Index vertex) {
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    if (state.selected[vertex_index] == 0) {
      state.selected[vertex_index] = 1;
      batch.push_back(vertex);
    }
  };

  if (uses_in_predicate(mode)) {
    while (!state.in_queue.empty() &&
           state.in_queue.top().key <= minimum_fringe_distance) {
      add_to_batch(state.in_queue.pop_min().vertex);
    }
  }
  if (uses_out_predicate(mode)) {
    while (!state.distance_queue.empty() &&
           state.distance_queue.top().key <= out_threshold) {
      add_to_batch(state.distance_queue.pop_min().vertex);
    }
  }

  std::sort(batch.begin(), batch.end());
  for (Index vertex : batch) {
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    if (state.result.states[vertex_index] != VertexState::kFringe) {
      throw std::logic_error(
          "predicate selected a vertex outside the fringe");
    }
    remove_from_active_queues(state, mode, vertex);
    state.selected[vertex_index] = 0;
  }
  return batch;
}

bool advance_in_simple_minimum(const CsrGraph& graph,
                               GraphAuxiliary& auxiliary,
                               AlgorithmState& state,
                               Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  std::vector<Offset>& edges =
      auxiliary.incoming_edges_descending[vertex_index];
  const float old_minimum = state.min_in_simple[vertex_index];
  while (!edges.empty()) {
    const Offset edge = edges.back();
    const Index source =
        auxiliary.edge_sources[static_cast<std::size_t>(edge)];
    if (state.result.states[static_cast<std::size_t>(source)] !=
        VertexState::kSettled) {
      break;
    }
    edges.pop_back();
  }
  const float new_minimum =
      edges.empty()
          ? std::numeric_limits<float>::infinity()
          : graph.values[static_cast<std::size_t>(edges.back())];
  state.min_in_simple[vertex_index] = new_minimum;
  return old_minimum != new_minimum;
}

bool advance_out_simple_minimum(const CsrGraph& graph,
                                GraphAuxiliary& auxiliary,
                                AlgorithmState& state,
                                Index vertex) {
  const std::size_t vertex_index = static_cast<std::size_t>(vertex);
  std::vector<Offset>& edges =
      auxiliary.outgoing_edges_descending[vertex_index];
  const float old_minimum = state.min_out_simple[vertex_index];
  while (!edges.empty()) {
    const Offset edge = edges.back();
    const Index destination =
        graph.colind[static_cast<std::size_t>(edge)];
    if (state.result.states[static_cast<std::size_t>(destination)] !=
        VertexState::kSettled) {
      break;
    }
    edges.pop_back();
  }
  const float new_minimum =
      edges.empty()
          ? std::numeric_limits<float>::infinity()
          : graph.values[static_cast<std::size_t>(edges.back())];
  state.min_out_simple[vertex_index] = new_minimum;
  return old_minimum != new_minimum;
}

void maintain_dynamic_minima(const CsrGraph& graph,
                             GraphAuxiliary& auxiliary,
                             AlgorithmState& state,
                             PredicateMode mode,
                             const std::vector<Index>& batch) {
  std::vector<Index> affected_vertices;

  if (uses_simple_in_predicate(mode)) {
    for (Index settled_vertex : batch) {
      const std::size_t settled_index =
          static_cast<std::size_t>(settled_vertex);
      const Offset begin = graph.rowptr[settled_index];
      const Offset end = graph.rowptr[settled_index + 1];
      for (Offset edge = begin; edge < end; ++edge) {
        const Index destination =
            graph.colind[static_cast<std::size_t>(edge)];
        const std::size_t destination_index =
            static_cast<std::size_t>(destination);
        if (state.affected[destination_index] == 0) {
          state.affected[destination_index] = 1;
          affected_vertices.push_back(destination);
        }
      }
    }
    for (Index vertex : affected_vertices) {
      const std::size_t vertex_index = static_cast<std::size_t>(vertex);
      const bool changed =
          advance_in_simple_minimum(graph, auxiliary, state, vertex);
      if (changed &&
          state.result.states[vertex_index] == VertexState::kFringe) {
        state.in_queue.update(
            vertex, in_queue_key(state, auxiliary, mode, vertex));
      }
      state.affected[vertex_index] = 0;
    }
    affected_vertices.clear();
  }

  if (uses_simple_out_predicate(mode)) {
    for (Index settled_vertex : batch) {
      const std::size_t settled_index =
          static_cast<std::size_t>(settled_vertex);
      const Offset begin = auxiliary.incoming.rowptr[settled_index];
      const Offset end = auxiliary.incoming.rowptr[settled_index + 1];
      for (Offset position = begin; position < end; ++position) {
        const Index source =
            auxiliary.incoming.sources[
                static_cast<std::size_t>(position)];
        const std::size_t source_index = static_cast<std::size_t>(source);
        if (state.affected[source_index] == 0) {
          state.affected[source_index] = 1;
          affected_vertices.push_back(source);
        }
      }
    }
    for (Index vertex : affected_vertices) {
      const std::size_t vertex_index = static_cast<std::size_t>(vertex);
      const bool changed =
          advance_out_simple_minimum(graph, auxiliary, state, vertex);
      if (changed &&
          state.result.states[vertex_index] == VertexState::kFringe) {
        state.out_queue.update(
            vertex, out_queue_key(state, auxiliary, mode, vertex));
      }
      state.affected[vertex_index] = 0;
    }
  }
}

void relax_phase_batch(const CsrGraph& graph,
                       const GraphAuxiliary& auxiliary,
                       AlgorithmState& state,
                       PredicateMode mode,
                       const std::vector<Index>& batch,
                       Statistics& statistics) {
  for (Index source : batch) {
    const std::size_t source_index = static_cast<std::size_t>(source);
    const float source_distance = state.result.distances[source_index];
    const Offset begin = graph.rowptr[source_index];
    const Offset end = graph.rowptr[source_index + 1];
    for (Offset edge = begin; edge < end; ++edge) {
      ++statistics.edges_examined;
      ++statistics.relaxations_attempted;

      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const Index destination = graph.colind[edge_index];
      const std::size_t destination_index =
          static_cast<std::size_t>(destination);
      const float candidate_distance =
          source_distance + graph.values[edge_index];
      if (candidate_distance <
          state.result.distances[destination_index]) {
        if (state.result.states[destination_index] ==
            VertexState::kSettled) {
          throw std::runtime_error(
              "a strict relaxation improved an already settled vertex; "
              "the input must have finite nonnegative weights");
        }

        state.result.distances[destination_index] = candidate_distance;
        state.result.predecessor_nodes[destination_index] = source;
        state.result.predecessor_edges[destination_index] = edge;
        ++statistics.successful_distance_updates;

        if (state.result.states[destination_index] ==
            VertexState::kUnexplored) {
          state.result.states[destination_index] = VertexState::kFringe;
          insert_fringe_vertex(
              state, auxiliary, mode, destination);
        } else {
          update_fringe_vertex(
              state, auxiliary, mode, destination);
        }
      }
    }
  }
}

void print_statistics(const Statistics& statistics,
                      const std::string& label,
                      PredicateMode mode,
                      Clock::duration preprocessing_time,
                      Clock::duration transfer_time,
                      Clock::duration total_runtime,
                      Clock::duration end_to_end_time) {
  const double predicate_ms =
      milliseconds(statistics.predicate_evaluation_time);
  const double relaxation_ms =
      milliseconds(statistics.relaxation_time);

  std::cout << "\n=== preds_CPU statistics (" << label << ") ===\n"
            << "Predicate mode: " << predicate_mode_name(mode) << '\n'
            << "Number of completed phases: " << statistics.phases << '\n'
            << "Vertices settled in current phase: "
            << statistics.vertices_settled_current_phase << '\n'
            << "Vertices settled per phase (phases 1.."
            << statistics.phases << "): [";
  for (std::size_t index = 0;
       index < statistics.vertices_settled_per_phase.size();
       ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << statistics.vertices_settled_per_phase[index];
  }
  std::cout << "]\n"
            << "Total vertices settled: "
            << statistics.vertices_settled << '\n'
            << "Edges examined: " << statistics.edges_examined << '\n'
            << "Relaxations attempted: "
            << statistics.relaxations_attempted << '\n'
            << "Successful distance updates: "
            << statistics.successful_distance_updates << '\n'
            << std::fixed << std::setprecision(6)
            << "Predicate-evaluation time (ms): " << predicate_ms << '\n'
            << "Relaxation time (ms): " << relaxation_ms << '\n';
  if (statistics.predicate_evaluation_time.count() > 0) {
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
            << "Elapsed time (ms): "
            << milliseconds(statistics.elapsed_time) << '\n'
            << "Total runtime (ms): " << milliseconds(total_runtime) << '\n'
            << "End-to-end time (ms): "
            << milliseconds(end_to_end_time) << '\n';
}

ReconstructedPath reconstruct_path(const CsrGraph& graph,
                                   const DijkstraResult& result,
                                   Index source,
                                   Index target) {
  ReconstructedPath path;
  if (result.states[static_cast<std::size_t>(target)] !=
      VertexState::kSettled) {
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
        graph.colind[static_cast<std::size_t>(predecessor_edge)] !=
            current) {
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
                              Index source,
                              PredicateMode mode) {
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
         << ",\"producer\":\"preds_CPU\""
         << ",\"predicate_mode\":\"" << predicate_mode_name(mode) << "\""
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
        result.states[static_cast<std::size_t>(target)] ==
        VertexState::kSettled;
    const ReconstructedPath path =
        reached ? reconstruct_path(graph, result, source, target)
                : ReconstructedPath{};
    write_path_record(
        output,
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

  GraphAuxiliary auxiliary =
      build_graph_auxiliary(graph, options.predicate);
  const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
  AlgorithmState state(vertex_count);
  state.result.distances.assign(
      vertex_count, std::numeric_limits<float>::infinity());
  state.result.predecessor_nodes.assign(
      vertex_count, static_cast<Index>(-1));
  state.result.predecessor_edges.assign(
      vertex_count, static_cast<Offset>(-1));
  state.result.states.assign(
      vertex_count, VertexState::kUnexplored);
  state.min_in_simple = auxiliary.min_in_static;
  state.min_out_simple = auxiliary.min_out_static;

  const std::size_t source_index =
      static_cast<std::size_t>(options.source);
  state.result.distances[source_index] = 0.0f;
  state.result.states[source_index] = VertexState::kFringe;
  insert_fringe_vertex(
      state, auxiliary, options.predicate, options.source);

  const auto preprocessing_end = Clock::now();
  const Clock::duration preprocessing_time =
      preprocessing_end - preprocessing_begin;
  const Clock::duration transfer_time = Clock::duration::zero();

  Statistics statistics;
  bool requested_snapshot_printed = false;

  while (!state.distance_queue.empty()) {
    const auto phase_begin = Clock::now();

    const auto identification_begin = Clock::now();
    std::vector<Index> batch =
        identify_phase_batch(state, options.predicate);
    if (batch.empty()) {
      throw std::runtime_error(
          std::string("predicate ") +
          predicate_mode_name(options.predicate) +
          " selected no vertex while the fringe was nonempty at phase " +
          std::to_string(statistics.phases + 1));
    }

    // The entire frozen batch is settled before any edge in the batch is
    // relaxed. Newly reached vertices can therefore enter only a later phase.
    for (Index vertex : batch) {
      state.result.states[static_cast<std::size_t>(vertex)] =
          VertexState::kSettled;
    }
    const auto identification_end = Clock::now();
    statistics.predicate_evaluation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            identification_end - identification_begin);

    const auto relaxation_begin = Clock::now();
    relax_phase_batch(
        graph,
        auxiliary,
        state,
        options.predicate,
        batch,
        statistics);
    const auto relaxation_end = Clock::now();
    statistics.relaxation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            relaxation_end - relaxation_begin);

    const auto maintenance_begin = Clock::now();
    maintain_dynamic_minima(
        graph, auxiliary, state, options.predicate, batch);
    const auto maintenance_end = Clock::now();
    statistics.predicate_evaluation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            maintenance_end - maintenance_begin);

    ++statistics.phases;
    statistics.vertices_settled_current_phase =
        static_cast<std::uint64_t>(batch.size());
    statistics.vertices_settled +=
        statistics.vertices_settled_current_phase;
    statistics.vertices_settled_per_phase.push_back(
        statistics.vertices_settled_current_phase);

    const auto phase_end = Clock::now();
    statistics.elapsed_time += phase_end - phase_begin;

    if (options.stats_number_set &&
        statistics.phases == options.stats_number) {
      const Clock::duration snapshot_end_to_end =
          phase_end - end_to_end_begin;
      const Clock::duration snapshot_total_runtime =
          preprocessing_time + statistics.elapsed_time;
      print_statistics(
          statistics,
          "phase " + std::to_string(options.stats_number) + " snapshot",
          options.predicate,
          preprocessing_time,
          transfer_time,
          snapshot_total_runtime,
          snapshot_end_to_end);
      requested_snapshot_printed = true;
    }
  }

  const Clock::duration total_runtime =
      preprocessing_time + statistics.elapsed_time;
  std::size_t paths_written = 0;
  if (options.print_paths) {
    paths_written = write_paths_jsonl(
        options.paths_output_path,
        graph,
        state.result,
        options.source,
        options.predicate);
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
  print_statistics(
      statistics,
      "final",
      options.predicate,
      preprocessing_time,
      transfer_time,
      total_runtime,
      end_to_end_time);
  return 0;
}

}  // namespace
}  // namespace rips_predicates

int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" ||
       std::string(argv[1]) == "--help")) {
    rips_predicates::print_usage(std::cout, argv[0]);
    return 0;
  }

  const auto end_to_end_begin = rips_predicates::Clock::now();
  try {
    const rips_predicates::Options options =
        rips_predicates::parse_args(argc, argv);
    return rips_predicates::run(options, end_to_end_begin);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
