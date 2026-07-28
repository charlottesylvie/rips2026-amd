// Shared-memory, owner-partitioned predicate-based phased Dijkstra SSSP.
//
// Build from rips2026-amd:
//   g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror \
//     -pthread Dijkstra/preds_CPU_parallel.cpp -o preds_CPU_parallel

#include <algorithm>
#include <atomic>
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rips_predicates_parallel {
namespace {

using Clock = std::chrono::steady_clock;
using Offset = std::int64_t;
using Index = std::int32_t;

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t kLegacyCsrVersion = 1;
constexpr std::uint64_t kCurrentCsrVersion = 2;
constexpr std::uint64_t kOutgoingEdgeOrientation = 2;
constexpr std::size_t kMaximumPrintedPaths = 1000;
constexpr std::uint32_t kInfinityBits = 0x7f800000u;

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
  bool threads_set = false;
  std::uint64_t requested_threads = 0;
  bool stats_number_set = false;
  std::uint64_t stats_number = 0;
  bool print_paths = false;
  std::filesystem::path paths_output_path;
};

struct Statistics {
  std::uint64_t worker_threads = 1;
  std::uint64_t phases = 0;
  std::uint64_t vertices_settled = 0;
  std::uint64_t vertices_settled_current_phase = 0;
  std::vector<std::uint64_t> vertices_settled_per_phase;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxations_attempted = 0;
  std::uint64_t successful_distance_updates = 0;
  std::uint64_t remote_relaxations_buffered = 0;
  std::vector<std::uint64_t> worker_edges_examined;
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
  output += ".preds_CPU_parallel.";
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
      << "  --threads <count>       Set the positive CPU worker count.\n"
      << "                         Default: hardware concurrency, or 1.\n"
      << "  --stats-number <phase> Print a cumulative statistics snapshot "
         "after that phase.\n"
      << "  --print [path]         Write at most 1000 source-to-target path "
         "records as JSONL.\n"
      << "                         Default path: "
         "<graph.csrbin>.preds_CPU_parallel.<predicate-mode>.paths.jsonl\n"
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
    } else if (option == "--threads") {
      if (options.threads_set) {
        throw std::invalid_argument("--threads may be specified only once");
      }
      if (++arg >= argc) {
        throw std::invalid_argument("--threads requires a thread count");
      }
      options.requested_threads =
          parse_positive_u64(argv[arg], "--threads");
      options.threads_set = true;
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
    for (Offset edge = graph.rowptr[source_index];
         edge < graph.rowptr[source_index + 1];
         ++edge) {
      const Index destination =
          graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t destination_index =
          static_cast<std::size_t>(destination);
      const Offset position = incoming_cursor[destination_index]++;
      auxiliary.incoming.sources[static_cast<std::size_t>(position)] =
          source;
      auxiliary.incoming.outgoing_edge_ids[
          static_cast<std::size_t>(position)] = edge;
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

class VertexPartition {
 public:
  VertexPartition(std::size_t vertex_count, std::size_t worker_count)
      : vertex_count_(vertex_count),
        worker_count_(worker_count),
        base_(vertex_count / worker_count),
        remainder_(vertex_count % worker_count) {
    if (vertex_count == 0 || worker_count == 0 ||
        worker_count > vertex_count) {
      throw std::invalid_argument("invalid vertex partition");
    }
  }

  std::pair<Index, Index> range(std::size_t worker) const {
    if (worker >= worker_count_) {
      throw std::out_of_range("worker index is outside the partition");
    }
    const std::size_t begin =
        worker * base_ + std::min(worker, remainder_);
    const std::size_t length =
        base_ + (worker < remainder_ ? 1 : 0);
    return {
        static_cast<Index>(begin),
        static_cast<Index>(begin + length)};
  }

  std::size_t owner(Index vertex) const {
    if (vertex < 0 ||
        static_cast<std::size_t>(vertex) >= vertex_count_) {
      throw std::out_of_range("vertex is outside the partition");
    }
    const std::size_t value = static_cast<std::size_t>(vertex);
    const std::size_t longer_prefix = (base_ + 1) * remainder_;
    if (value < longer_prefix) {
      return value / (base_ + 1);
    }
    return remainder_ + (value - longer_prefix) / base_;
  }

  std::size_t worker_count() const {
    return worker_count_;
  }

 private:
  std::size_t vertex_count_ = 0;
  std::size_t worker_count_ = 0;
  std::size_t base_ = 0;
  std::size_t remainder_ = 0;
};

class CancellableBarrier {
 public:
  explicit CancellableBarrier(std::size_t participants)
      : participants_(participants),
        waiting_(participants) {
    if (participants == 0) {
      throw std::invalid_argument("barrier requires participants");
    }
  }

  bool arrive_and_wait() {
    if (cancelled_.load(std::memory_order_acquire)) {
      return false;
    }
    const std::uint64_t generation =
        generation_.load(std::memory_order_acquire);
    if (waiting_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      waiting_.store(participants_, std::memory_order_relaxed);
      // The last arriver has acquired the release sequence formed by all
      // participants. Publishing the next generation with release therefore
      // makes every pre-barrier owner write visible to acquire waiters.
      generation_.fetch_add(1, std::memory_order_release);
    } else {
      while (generation_.load(std::memory_order_acquire) == generation) {
        if (cancelled_.load(std::memory_order_acquire)) {
          return false;
        }
        std::this_thread::yield();
      }
    }
    return !cancelled_.load(std::memory_order_acquire);
  }

  void cancel() {
    cancelled_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_acq_rel);
  }

  bool cancelled() const {
    return cancelled_.load(std::memory_order_acquire);
  }

 private:
  const std::size_t participants_;
  std::atomic<std::size_t> waiting_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<bool> cancelled_{false};
};

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void atomic_min_nonnegative(std::atomic<std::uint32_t>& target,
                            float candidate) {
  const std::uint32_t candidate_bits = float_bits(candidate);
  std::uint32_t observed = target.load(std::memory_order_relaxed);
  while (candidate_bits < observed &&
         !target.compare_exchange_weak(
             observed,
             candidate_bits,
             std::memory_order_acq_rel,
             std::memory_order_relaxed)) {
  }
}

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

class OwnedMutableMinQueue {
 public:
  OwnedMutableMinQueue(Index begin, Index end)
      : begin_(begin),
        end_(end),
        handles_(static_cast<std::size_t>(end - begin)) {
    if (begin < 0 || end < begin) {
      throw std::invalid_argument("invalid owned queue range");
    }
  }

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
    return handles_.at(local_index(vertex)).has_value();
  }

  void insert(Index vertex, float key) {
    std::optional<FibonacciHeap::handle_type>& handle =
        handles_.at(local_index(vertex));
    if (handle.has_value()) {
      throw std::logic_error("duplicate mutable-priority-queue insertion");
    }
    handle = heap_.push(HeapEntry{key, vertex});
  }

  void update(Index vertex, float key) {
    std::optional<FibonacciHeap::handle_type>& handle =
        handles_.at(local_index(vertex));
    if (!handle.has_value()) {
      throw std::logic_error("mutable-priority-queue update without a handle");
    }
    heap_.update(*handle, HeapEntry{key, vertex});
  }

  void erase(Index vertex) {
    std::optional<FibonacciHeap::handle_type>& handle =
        handles_.at(local_index(vertex));
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
    handles_.at(local_index(entry.vertex)).reset();
    return entry;
  }

 private:
  std::size_t local_index(Index vertex) const {
    if (vertex < begin_ || vertex >= end_) {
      throw std::out_of_range(
          "priority-queue access violates vertex ownership");
    }
    return static_cast<std::size_t>(vertex - begin_);
  }

  Index begin_ = 0;
  Index end_ = 0;
  FibonacciHeap heap_;
  std::vector<std::optional<FibonacciHeap::handle_type>> handles_;
};

struct RelaxationRecord {
  Index destination = -1;
  float candidate_distance = std::numeric_limits<float>::infinity();
  Index predecessor_node = -1;
  Offset predecessor_edge = -1;
};

template <typename T>
class alignas(64) FixedInbox {
 public:
  explicit FixedInbox(std::size_t capacity)
      : records_(capacity) {}

  void push(const T& record) {
    const std::size_t slot =
        size_.fetch_add(1, std::memory_order_relaxed);
    if (slot >= records_.size()) {
      throw std::overflow_error("preallocated owner inbox overflow");
    }
    records_[slot] = record;
  }

  std::size_t size() const {
    return size_.load(std::memory_order_acquire);
  }

  const T& operator[](std::size_t index) const {
    return records_.at(index);
  }

  void clear() {
    size_.store(0, std::memory_order_release);
  }

 private:
  std::vector<T> records_;
  std::atomic<std::size_t> size_{0};
};

struct PhaseCounters {
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxations_attempted = 0;
  std::uint64_t successful_distance_updates = 0;
  std::uint64_t remote_relaxations_buffered = 0;

  void clear() {
    edges_examined = 0;
    relaxations_attempted = 0;
    successful_distance_updates = 0;
    remote_relaxations_buffered = 0;
  }
};

struct WorkerState {
  WorkerState(std::size_t worker_id, Index range_begin, Index range_end)
      : id(worker_id),
        begin(range_begin),
        end(range_end),
        distance_queue(range_begin, range_end),
        in_queue(range_begin, range_end),
        out_queue(range_begin, range_end),
        selected(static_cast<std::size_t>(range_end - range_begin), 0),
        affected_in(static_cast<std::size_t>(range_end - range_begin), 0),
        affected_out(static_cast<std::size_t>(range_end - range_begin), 0) {
    const std::size_t local_count =
        static_cast<std::size_t>(range_end - range_begin);
    batch.reserve(local_count);
    affected_vertices.reserve(local_count);
  }

  bool owns(Index vertex) const {
    return vertex >= begin && vertex < end;
  }

  std::size_t local_index(Index vertex) const {
    if (!owns(vertex)) {
      throw std::out_of_range("worker attempted a non-owned vertex update");
    }
    return static_cast<std::size_t>(vertex - begin);
  }

  std::size_t id = 0;
  Index begin = 0;
  Index end = 0;
  OwnedMutableMinQueue distance_queue;
  OwnedMutableMinQueue in_queue;
  OwnedMutableMinQueue out_queue;
  std::vector<std::uint8_t> selected;
  std::vector<std::uint8_t> affected_in;
  std::vector<std::uint8_t> affected_out;
  std::vector<Index> batch;
  std::vector<Index> affected_vertices;
  PhaseCounters phase;
};

class ParallelAlgorithm {
 public:
  ParallelAlgorithm(const CsrGraph& graph,
                    GraphAuxiliary& auxiliary,
                    const Options& options,
                    std::size_t worker_count)
      : graph_(graph),
        auxiliary_(auxiliary),
        options_(options),
        partition_(static_cast<std::size_t>(graph.rows), worker_count),
        barrier_(worker_count) {
    const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
    result_.distances.assign(
        vertex_count, std::numeric_limits<float>::infinity());
    result_.predecessor_nodes.assign(vertex_count, static_cast<Index>(-1));
    result_.predecessor_edges.assign(vertex_count, static_cast<Offset>(-1));
    result_.states.assign(vertex_count, VertexState::kUnexplored);
    if (uses_simple_in_predicate(options_.predicate)) {
      min_in_simple_ = auxiliary.min_in_static;
    }
    if (uses_simple_out_predicate(options_.predicate)) {
      min_out_simple_ = auxiliary.min_out_static;
    }

    workers_.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      const auto owned_range = partition_.range(worker);
      workers_.push_back(std::make_unique<WorkerState>(
          worker, owned_range.first, owned_range.second));
    }

    std::vector<std::size_t> remote_relaxation_capacity(worker_count, 0);
    std::vector<std::size_t> remote_in_notification_capacity(
        worker_count, 0);
    std::vector<std::size_t> remote_out_notification_capacity(
        worker_count, 0);
    for (Offset raw_source = 0; raw_source < graph.rows; ++raw_source) {
      const Index source = static_cast<Index>(raw_source);
      const std::size_t source_owner = partition_.owner(source);
      const std::size_t source_index = static_cast<std::size_t>(source);
      for (Offset edge = graph.rowptr[source_index];
           edge < graph.rowptr[source_index + 1];
           ++edge) {
        const Index destination =
            graph.colind[static_cast<std::size_t>(edge)];
        const std::size_t destination_owner =
            partition_.owner(destination);
        if (destination_owner != source_owner) {
          ++remote_relaxation_capacity[destination_owner];
          if (uses_simple_in_predicate(options_.predicate)) {
            ++remote_in_notification_capacity[destination_owner];
          }
          if (uses_simple_out_predicate(options_.predicate)) {
            ++remote_out_notification_capacity[source_owner];
          }
        }
      }
    }
    relaxation_inboxes_.reserve(worker_count);
    in_notification_inboxes_.reserve(worker_count);
    out_notification_inboxes_.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      relaxation_inboxes_.push_back(
          std::make_unique<FixedInbox<RelaxationRecord>>(
              remote_relaxation_capacity[worker]));
      in_notification_inboxes_.push_back(
          std::make_unique<FixedInbox<Index>>(
              remote_in_notification_capacity[worker]));
      out_notification_inboxes_.push_back(
          std::make_unique<FixedInbox<Index>>(
              remote_out_notification_capacity[worker]));
    }

    statistics_.worker_threads = worker_count;
    statistics_.worker_edges_examined.assign(worker_count, 0);
    statistics_.vertices_settled_per_phase.reserve(vertex_count);
  }

  void execute(Clock::time_point preprocessing_begin,
               Clock::time_point end_to_end_begin) {
    const std::size_t worker_count = partition_.worker_count();
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    try {
      for (std::size_t worker = 0; worker < worker_count; ++worker) {
        threads.emplace_back([this, worker] { worker_entry(worker); });
      }
    } catch (...) {
      barrier_.cancel();
      start_.store(true, std::memory_order_release);
      for (std::thread& thread : threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      throw;
    }

    while (ready_.load(std::memory_order_acquire) != worker_count) {
      std::this_thread::yield();
    }
    preprocessing_time_ = Clock::now() - preprocessing_begin;
    end_to_end_begin_ = end_to_end_begin;
    start_.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
      thread.join();
    }
    if (worker_exception_ != nullptr) {
      std::rethrow_exception(worker_exception_);
    }
  }

  const DijkstraResult& result() const {
    return result_;
  }

  const Statistics& statistics() const {
    return statistics_;
  }

  Clock::duration preprocessing_time() const {
    return preprocessing_time_;
  }

  bool requested_snapshot_printed() const {
    return requested_snapshot_printed_;
  }

 private:
  float current_in_minimum(Index vertex) const {
    const std::size_t index = static_cast<std::size_t>(vertex);
    return uses_simple_in_predicate(options_.predicate)
               ? min_in_simple_[index]
               : auxiliary_.min_in_static[index];
  }

  float current_out_minimum(Index vertex) const {
    const std::size_t index = static_cast<std::size_t>(vertex);
    return uses_simple_out_predicate(options_.predicate)
               ? min_out_simple_[index]
               : auxiliary_.min_out_static[index];
  }

  float in_queue_key(Index vertex) const {
    const std::size_t index = static_cast<std::size_t>(vertex);
    return result_.distances[index] - current_in_minimum(vertex);
  }

  float out_queue_key(Index vertex) const {
    const std::size_t index = static_cast<std::size_t>(vertex);
    return result_.distances[index] + current_out_minimum(vertex);
  }

  void insert_fringe_vertex(WorkerState& worker, Index vertex) {
    if (!worker.owns(vertex)) {
      throw std::logic_error("non-owner attempted a fringe insertion");
    }
    const float distance =
        result_.distances[static_cast<std::size_t>(vertex)];
    worker.distance_queue.insert(vertex, distance);
    if (uses_in_predicate(options_.predicate)) {
      worker.in_queue.insert(vertex, in_queue_key(vertex));
    }
    if (uses_out_predicate(options_.predicate)) {
      worker.out_queue.insert(vertex, out_queue_key(vertex));
    }
  }

  void update_fringe_vertex(WorkerState& worker, Index vertex) {
    if (!worker.owns(vertex)) {
      throw std::logic_error("non-owner attempted a fringe update");
    }
    const float distance =
        result_.distances[static_cast<std::size_t>(vertex)];
    worker.distance_queue.update(vertex, distance);
    if (uses_in_predicate(options_.predicate)) {
      worker.in_queue.update(vertex, in_queue_key(vertex));
    }
    if (uses_out_predicate(options_.predicate)) {
      worker.out_queue.update(vertex, out_queue_key(vertex));
    }
  }

  void remove_from_active_queues(WorkerState& worker, Index vertex) {
    worker.distance_queue.erase(vertex);
    if (uses_in_predicate(options_.predicate)) {
      worker.in_queue.erase(vertex);
    }
    if (uses_out_predicate(options_.predicate)) {
      worker.out_queue.erase(vertex);
    }
  }

  void initialize_source(WorkerState& worker) {
    if (!worker.owns(options_.source)) {
      return;
    }
    const std::size_t source_index =
        static_cast<std::size_t>(options_.source);
    result_.distances[source_index] = 0.0f;
    result_.states[source_index] = VertexState::kFringe;
    insert_fringe_vertex(worker, options_.source);
  }

  void identify_local_batch(WorkerState& worker,
                            float minimum_fringe_distance,
                            float out_threshold) {
    worker.batch.clear();
    const auto add_to_batch =
        [&worker](Index vertex) {
          const std::size_t local = worker.local_index(vertex);
          if (worker.selected[local] == 0) {
            worker.selected[local] = 1;
            worker.batch.push_back(vertex);
          }
        };

    if (uses_in_predicate(options_.predicate)) {
      while (!worker.in_queue.empty() &&
             worker.in_queue.top().key <= minimum_fringe_distance) {
        add_to_batch(worker.in_queue.pop_min().vertex);
      }
    }
    if (uses_out_predicate(options_.predicate)) {
      while (!worker.distance_queue.empty() &&
             worker.distance_queue.top().key <= out_threshold) {
        add_to_batch(worker.distance_queue.pop_min().vertex);
      }
    }
    std::sort(worker.batch.begin(), worker.batch.end());
    for (Index vertex : worker.batch) {
      const std::size_t index = static_cast<std::size_t>(vertex);
      if (result_.states[index] != VertexState::kFringe) {
        throw std::logic_error(
            "predicate selected a vertex outside the fringe");
      }
      remove_from_active_queues(worker, vertex);
      worker.selected[worker.local_index(vertex)] = 0;
    }
  }

  void apply_relaxation(WorkerState& owner,
                        const RelaxationRecord& record) {
    if (!owner.owns(record.destination)) {
      throw std::logic_error("relaxation was delivered to the wrong owner");
    }
    const std::size_t destination_index =
        static_cast<std::size_t>(record.destination);
    if (record.candidate_distance >=
        result_.distances[destination_index]) {
      return;
    }
    if (result_.states[destination_index] == VertexState::kSettled) {
      throw std::runtime_error(
          "a strict relaxation improved an already settled vertex; "
          "the input must have finite nonnegative weights");
    }
    result_.distances[destination_index] =
        record.candidate_distance;
    result_.predecessor_nodes[destination_index] =
        record.predecessor_node;
    result_.predecessor_edges[destination_index] =
        record.predecessor_edge;
    ++owner.phase.successful_distance_updates;
    if (result_.states[destination_index] == VertexState::kUnexplored) {
      result_.states[destination_index] = VertexState::kFringe;
      insert_fringe_vertex(owner, record.destination);
    } else {
      update_fringe_vertex(owner, record.destination);
    }
  }

  void produce_relaxations(WorkerState& worker) {
    for (Index source : worker.batch) {
      const std::size_t source_index = static_cast<std::size_t>(source);
      const float source_distance = result_.distances[source_index];
      for (Offset edge = graph_.rowptr[source_index];
           edge < graph_.rowptr[source_index + 1];
           ++edge) {
        ++worker.phase.edges_examined;
        ++worker.phase.relaxations_attempted;
        const std::size_t edge_index = static_cast<std::size_t>(edge);
        const Index destination = graph_.colind[edge_index];
        const RelaxationRecord record{
            destination,
            source_distance + graph_.values[edge_index],
            source,
            edge};
        const std::size_t destination_owner =
            partition_.owner(destination);
        if (destination_owner == worker.id) {
          apply_relaxation(worker, record);
        } else {
          relaxation_inboxes_[destination_owner]->push(record);
          ++worker.phase.remote_relaxations_buffered;
        }
      }
    }
  }

  void drain_remote_relaxations(WorkerState& worker) {
    FixedInbox<RelaxationRecord>& inbox =
        *relaxation_inboxes_[worker.id];
    const std::size_t count = inbox.size();
    for (std::size_t index = 0; index < count; ++index) {
      apply_relaxation(worker, inbox[index]);
    }
    inbox.clear();
  }

  void mark_affected(WorkerState& worker,
                     Index vertex,
                     std::vector<std::uint8_t>& flags) {
    const std::size_t local = worker.local_index(vertex);
    if (flags[local] == 0) {
      flags[local] = 1;
      worker.affected_vertices.push_back(vertex);
    }
  }

  bool advance_in_simple_minimum(WorkerState& worker, Index vertex) {
    if (!worker.owns(vertex)) {
      throw std::logic_error(
          "non-owner attempted to advance an IN_SIMPLE cursor");
    }
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    std::vector<Offset>& edges =
        auxiliary_.incoming_edges_descending[vertex_index];
    const float old_minimum = min_in_simple_[vertex_index];
    while (!edges.empty()) {
      const Offset edge = edges.back();
      const Index source =
          auxiliary_.edge_sources[static_cast<std::size_t>(edge)];
      if (result_.states[static_cast<std::size_t>(source)] !=
          VertexState::kSettled) {
        break;
      }
      edges.pop_back();
    }
    const float new_minimum =
        edges.empty()
            ? std::numeric_limits<float>::infinity()
            : graph_.values[static_cast<std::size_t>(edges.back())];
    min_in_simple_[vertex_index] = new_minimum;
    return old_minimum != new_minimum;
  }

  bool advance_out_simple_minimum(WorkerState& worker, Index vertex) {
    if (!worker.owns(vertex)) {
      throw std::logic_error(
          "non-owner attempted to advance an OUT_SIMPLE cursor");
    }
    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    std::vector<Offset>& edges =
        auxiliary_.outgoing_edges_descending[vertex_index];
    const float old_minimum = min_out_simple_[vertex_index];
    while (!edges.empty()) {
      const Offset edge = edges.back();
      const Index destination =
          graph_.colind[static_cast<std::size_t>(edge)];
      if (result_.states[static_cast<std::size_t>(destination)] !=
          VertexState::kSettled) {
        break;
      }
      edges.pop_back();
    }
    const float new_minimum =
        edges.empty()
            ? std::numeric_limits<float>::infinity()
            : graph_.values[static_cast<std::size_t>(edges.back())];
    min_out_simple_[vertex_index] = new_minimum;
    return old_minimum != new_minimum;
  }

  void produce_in_simple_notifications(WorkerState& worker) {
    for (Index settled_vertex : worker.batch) {
      const std::size_t settled_index =
          static_cast<std::size_t>(settled_vertex);
      for (Offset edge = graph_.rowptr[settled_index];
           edge < graph_.rowptr[settled_index + 1];
           ++edge) {
        const Index destination =
            graph_.colind[static_cast<std::size_t>(edge)];
        const std::size_t owner = partition_.owner(destination);
        if (owner == worker.id) {
          mark_affected(worker, destination, worker.affected_in);
        } else {
          in_notification_inboxes_[owner]->push(destination);
        }
      }
    }
  }

  void drain_in_simple_notifications(WorkerState& worker) {
    FixedInbox<Index>& inbox = *in_notification_inboxes_[worker.id];
    const std::size_t count = inbox.size();
    for (std::size_t index = 0; index < count; ++index) {
      mark_affected(worker, inbox[index], worker.affected_in);
    }
    inbox.clear();

    for (Index vertex : worker.affected_vertices) {
      const std::size_t vertex_index = static_cast<std::size_t>(vertex);
      const bool changed = advance_in_simple_minimum(worker, vertex);
      if (changed &&
          result_.states[vertex_index] == VertexState::kFringe) {
        worker.in_queue.update(vertex, in_queue_key(vertex));
      }
      worker.affected_in[worker.local_index(vertex)] = 0;
    }
    worker.affected_vertices.clear();
  }

  void produce_out_simple_notifications(WorkerState& worker) {
    for (Index settled_vertex : worker.batch) {
      const std::size_t settled_index =
          static_cast<std::size_t>(settled_vertex);
      const Offset begin = auxiliary_.incoming.rowptr[settled_index];
      const Offset end = auxiliary_.incoming.rowptr[settled_index + 1];
      for (Offset position = begin; position < end; ++position) {
        const Index source =
            auxiliary_.incoming.sources[
                static_cast<std::size_t>(position)];
        const std::size_t owner = partition_.owner(source);
        if (owner == worker.id) {
          mark_affected(worker, source, worker.affected_out);
        } else {
          out_notification_inboxes_[owner]->push(source);
        }
      }
    }
  }

  void drain_out_simple_notifications(WorkerState& worker) {
    FixedInbox<Index>& inbox = *out_notification_inboxes_[worker.id];
    const std::size_t count = inbox.size();
    for (std::size_t index = 0; index < count; ++index) {
      mark_affected(worker, inbox[index], worker.affected_out);
    }
    inbox.clear();

    for (Index vertex : worker.affected_vertices) {
      const std::size_t vertex_index = static_cast<std::size_t>(vertex);
      const bool changed = advance_out_simple_minimum(worker, vertex);
      if (changed &&
          result_.states[vertex_index] == VertexState::kFringe) {
        worker.out_queue.update(vertex, out_queue_key(vertex));
      }
      worker.affected_out[worker.local_index(vertex)] = 0;
    }
    worker.affected_vertices.clear();
  }

  void aggregate_completed_phase(Clock::time_point phase_begin,
                                 Clock::time_point predicate_identification_end,
                                 Clock::time_point relaxation_begin,
                                 Clock::time_point relaxation_end,
                                 Clock::time_point maintenance_begin,
                                 Clock::time_point phase_end) {
    std::uint64_t selected_count = 0;
    for (const std::unique_ptr<WorkerState>& worker : workers_) {
      selected_count += static_cast<std::uint64_t>(worker->batch.size());
      statistics_.edges_examined += worker->phase.edges_examined;
      statistics_.relaxations_attempted +=
          worker->phase.relaxations_attempted;
      statistics_.successful_distance_updates +=
          worker->phase.successful_distance_updates;
      statistics_.remote_relaxations_buffered +=
          worker->phase.remote_relaxations_buffered;
      statistics_.worker_edges_examined[worker->id] +=
          worker->phase.edges_examined;
    }
    if (selected_count == 0) {
      throw std::runtime_error(
          std::string("predicate ") +
          predicate_mode_name(options_.predicate) +
          " selected no vertex while the fringe was nonempty at phase " +
          std::to_string(statistics_.phases + 1));
    }
    ++statistics_.phases;
    statistics_.vertices_settled_current_phase = selected_count;
    statistics_.vertices_settled += selected_count;
    statistics_.vertices_settled_per_phase.push_back(selected_count);
    statistics_.predicate_evaluation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            predicate_identification_end - phase_begin);
    statistics_.predicate_evaluation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - maintenance_begin);
    statistics_.relaxation_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            relaxation_end - relaxation_begin);
    statistics_.elapsed_time += phase_end - phase_begin;
  }

  void worker_loop(WorkerState& worker) {
    while (true) {
      worker.phase.clear();
      Clock::time_point phase_begin{};
      if (worker.id == 0) {
        phase_begin = Clock::now();
        minimum_distance_bits_.store(
            kInfinityBits, std::memory_order_relaxed);
        out_threshold_bits_.store(
            kInfinityBits, std::memory_order_relaxed);
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      if (!worker.distance_queue.empty()) {
        atomic_min_nonnegative(
            minimum_distance_bits_, worker.distance_queue.top().key);
        if (uses_out_predicate(options_.predicate)) {
          if (worker.out_queue.empty()) {
            throw std::logic_error(
                "out-criterion queue is empty while the fringe is nonempty");
          }
          atomic_min_nonnegative(
              out_threshold_bits_, worker.out_queue.top().key);
        }
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      const float minimum_fringe_distance =
          bits_float(minimum_distance_bits_.load(std::memory_order_acquire));
      if (minimum_fringe_distance ==
          std::numeric_limits<float>::infinity()) {
        break;
      }
      const float out_threshold =
          bits_float(out_threshold_bits_.load(std::memory_order_acquire));
      identify_local_batch(
          worker, minimum_fringe_distance, out_threshold);
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      for (Index vertex : worker.batch) {
        const std::size_t index = static_cast<std::size_t>(vertex);
        if (result_.states[index] != VertexState::kFringe) {
          throw std::logic_error("invalid settle transition");
        }
        result_.states[index] = VertexState::kSettled;
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      Clock::time_point predicate_identification_end{};
      Clock::time_point relaxation_begin{};
      if (worker.id == 0) {
        predicate_identification_end = Clock::now();
        relaxation_begin = predicate_identification_end;
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      produce_relaxations(worker);
      if (!barrier_.arrive_and_wait()) {
        return;
      }
      drain_remote_relaxations(worker);
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      Clock::time_point relaxation_end{};
      if (worker.id == 0) {
        relaxation_end = Clock::now();
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      Clock::time_point maintenance_begin{};
      Clock::time_point phase_end{};
      if (worker.id == 0) {
        maintenance_begin = Clock::now();
      }
      if (!barrier_.arrive_and_wait()) {
        return;
      }

      // SIMPLE maintenance is staged IN then OUT to match preds_CPU.cpp.
      // Producers only append owner notifications; after the barrier, each
      // owner alone advances its cursors, minima, and mutable queue entries.
      if (uses_simple_in_predicate(options_.predicate)) {
        produce_in_simple_notifications(worker);
        if (!barrier_.arrive_and_wait()) {
          return;
        }
        drain_in_simple_notifications(worker);
        if (!barrier_.arrive_and_wait()) {
          return;
        }
      }
      if (uses_simple_out_predicate(options_.predicate)) {
        produce_out_simple_notifications(worker);
        if (!barrier_.arrive_and_wait()) {
          return;
        }
        drain_out_simple_notifications(worker);
        if (!barrier_.arrive_and_wait()) {
          return;
        }
      }

      if (worker.id == 0) {
        phase_end = Clock::now();
        aggregate_completed_phase(
            phase_begin,
            predicate_identification_end,
            relaxation_begin,
            relaxation_end,
            maintenance_begin,
            phase_end);
        if (options_.stats_number_set &&
            statistics_.phases == options_.stats_number) {
          print_statistics_snapshot(
              "phase " + std::to_string(options_.stats_number) +
                  " snapshot",
              phase_end - end_to_end_begin_);
          requested_snapshot_printed_ = true;
        }
      }
      // Other workers wait here while worker 0 aggregates and, if requested,
      // prints the completed-phase snapshot. Acquire/release also publishes
      // cleared inbox counters before the next phase's producers run.
      if (!barrier_.arrive_and_wait()) {
        return;
      }
    }
  }

  void worker_entry(std::size_t worker_id) noexcept {
    bool ready_announced = false;
    try {
      WorkerState& worker = *workers_.at(worker_id);
      initialize_source(worker);
      ready_.fetch_add(1, std::memory_order_release);
      ready_announced = true;
      while (!start_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (barrier_.cancelled()) {
        return;
      }
      worker_loop(worker);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (worker_exception_ == nullptr) {
          worker_exception_ = std::current_exception();
        }
      }
      if (!ready_announced) {
        ready_.fetch_add(1, std::memory_order_release);
      }
      barrier_.cancel();
    }
  }

  void print_statistics_snapshot(const std::string& label,
                                 Clock::duration end_to_end_time) const;

  const CsrGraph& graph_;
  GraphAuxiliary& auxiliary_;
  const Options& options_;
  VertexPartition partition_;
  CancellableBarrier barrier_;
  DijkstraResult result_;
  std::vector<float> min_in_simple_;
  std::vector<float> min_out_simple_;
  std::vector<std::unique_ptr<WorkerState>> workers_;
  std::vector<std::unique_ptr<FixedInbox<RelaxationRecord>>>
      relaxation_inboxes_;
  std::vector<std::unique_ptr<FixedInbox<Index>>>
      in_notification_inboxes_;
  std::vector<std::unique_ptr<FixedInbox<Index>>>
      out_notification_inboxes_;
  std::atomic<std::uint32_t> minimum_distance_bits_{kInfinityBits};
  std::atomic<std::uint32_t> out_threshold_bits_{kInfinityBits};
  std::atomic<std::size_t> ready_{0};
  std::atomic<bool> start_{false};
  std::mutex exception_mutex_;
  std::exception_ptr worker_exception_;
  Statistics statistics_;
  Clock::duration preprocessing_time_ = Clock::duration::zero();
  Clock::time_point end_to_end_begin_{};
  bool requested_snapshot_printed_ = false;
};

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
  const auto edge_minmax = std::minmax_element(
      statistics.worker_edges_examined.begin(),
      statistics.worker_edges_examined.end());
  const std::uint64_t minimum_worker_edges =
      edge_minmax.first == statistics.worker_edges_examined.end()
          ? 0
          : *edge_minmax.first;
  const std::uint64_t maximum_worker_edges =
      edge_minmax.second == statistics.worker_edges_examined.end()
          ? 0
          : *edge_minmax.second;

  std::cout << "\n=== preds_CPU_parallel statistics (" << label << ") ===\n"
            << "Predicate mode: " << predicate_mode_name(mode) << '\n'
            << "CPU worker-thread count: "
            << statistics.worker_threads << '\n'
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
            << "Remote relaxation messages: "
            << statistics.remote_relaxations_buffered << '\n';
  if (statistics.relaxations_attempted != 0) {
    const double remote_share =
        static_cast<double>(statistics.remote_relaxations_buffered) /
        static_cast<double>(statistics.relaxations_attempted);
    std::cout << std::fixed << std::setprecision(6)
              << "Remote relaxation share: " << remote_share << '\n';
  } else {
    std::cout << "Remote relaxation share: n/a\n";
  }
  std::cout << "Worker edges examined (min/max): "
            << minimum_worker_edges << '/' << maximum_worker_edges << '\n';
  if (minimum_worker_edges != 0) {
    std::cout << std::fixed << std::setprecision(6)
              << "Worker edge-load max/min ratio: "
              << (static_cast<double>(maximum_worker_edges) /
                  static_cast<double>(minimum_worker_edges))
              << '\n';
  } else {
    std::cout << "Worker edge-load max/min ratio: n/a\n";
  }
  std::cout << std::fixed << std::setprecision(6)
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

void ParallelAlgorithm::print_statistics_snapshot(
    const std::string& label,
    Clock::duration end_to_end_time) const {
  print_statistics(
      statistics_,
      label,
      options_.predicate,
      preprocessing_time_,
      Clock::duration::zero(),
      preprocessing_time_ + statistics_.elapsed_time,
      end_to_end_time);
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
         << ",\"producer\":\"preds_CPU_parallel\""
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

std::size_t effective_thread_count(const Options& options,
                                   Offset vertex_count) {
  std::uint64_t requested = options.requested_threads;
  if (!options.threads_set) {
    requested = std::thread::hardware_concurrency();
    if (requested == 0) {
      requested = 1;
    }
  }
  const std::uint64_t vertices =
      static_cast<std::uint64_t>(vertex_count);
  return static_cast<std::size_t>(std::min(requested, vertices));
}

int run(const Options& options, Clock::time_point end_to_end_begin) {
  const auto preprocessing_begin = Clock::now();
  const CsrGraph graph = load_csr(options.csr_path);
  if (static_cast<Offset>(options.source) >= graph.rows) {
    throw std::out_of_range("source node is outside the CSR graph");
  }

  GraphAuxiliary auxiliary =
      build_graph_auxiliary(graph, options.predicate);
  const std::size_t worker_count =
      effective_thread_count(options, graph.rows);
  ParallelAlgorithm algorithm(
      graph, auxiliary, options, worker_count);
  algorithm.execute(preprocessing_begin, end_to_end_begin);

  const Statistics& statistics = algorithm.statistics();
  const Clock::duration preprocessing_time =
      algorithm.preprocessing_time();
  const Clock::duration transfer_time = Clock::duration::zero();
  const Clock::duration total_runtime =
      preprocessing_time + statistics.elapsed_time;
  std::size_t paths_written = 0;
  if (options.print_paths) {
    paths_written = write_paths_jsonl(
        options.paths_output_path,
        graph,
        algorithm.result(),
        options.source,
        options.predicate);
  }
  const Clock::duration end_to_end_time =
      Clock::now() - end_to_end_begin;

  if (options.stats_number_set &&
      !algorithm.requested_snapshot_printed()) {
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
}  // namespace rips_predicates_parallel

int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" ||
       std::string(argv[1]) == "--help")) {
    rips_predicates_parallel::print_usage(std::cout, argv[0]);
    return 0;
  }

  const auto end_to_end_begin =
      rips_predicates_parallel::Clock::now();
  try {
    const rips_predicates_parallel::Options options =
        rips_predicates_parallel::parse_args(argc, argv);
    return rips_predicates_parallel::run(
        options, end_to_end_begin);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
