// HIP/ROCm GPU implementation of phased predicate-based Dijkstra SSSP.
//
// Build from rips2026-amd:
//   hipcc -x hip -std=c++17 -O3 -Wall -Wextra Dijkstra/preds_GPU.cpp -o preds_GPU
//
// Optional Strix Halo build:
//   hipcc -x hip -std=c++17 -O3 -Wall -Wextra --offload-arch=gfx1151 Dijkstra/preds_GPU.cpp -o preds_GPU

#include <hip/hip_runtime.h>

#include "preds_gpu_engine.hpp"

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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rips_predicates_gpu {
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

const char* termination_reason_name(TerminationReason reason) {
  switch (reason) {
    case TerminationReason::kFullConvergence:
      return "full_convergence";
    case TerminationReason::kAllTargetsSettled:
      return "all_targets_settled";
  }
  throw std::logic_error("unknown preds-gpu termination reason");
}

namespace {

using Clock = std::chrono::steady_clock;
using CompactOffset = std::uint32_t;

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t kLegacyCsrVersion = 1;
constexpr std::uint64_t kCurrentCsrVersion = 2;
constexpr std::uint64_t kOutgoingEdgeOrientation = 2;
constexpr std::size_t kMaximumPrintedPaths = 1000;
constexpr unsigned int kInfinityBits = 0x7f800000u;
constexpr unsigned int kNoPackedPredecessor = 0xffffffffu;
constexpr int kUnexplored = 0;
constexpr int kFringe = 1;
constexpr int kSettled = 2;
constexpr int kMaximumGridBlocks = 65535;

static_assert(sizeof(Offset) == 8, "RIPS CSR offsets must be 64-bit");
static_assert(sizeof(Index) == 4, "RIPS CSR indices must be 32-bit");
static_assert(sizeof(float) == 4, "RIPS CSR weights must be 32-bit floats");
static_assert(sizeof(CompactOffset) == 4,
              "compact device offsets must be 32-bit");

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

using Statistics = AlgorithmStatistics;

struct DeviceMetadata {
  int device = 0;
  hipDeviceProp_t properties{};
  int integrated = 0;
  int block_size = 0;
  std::size_t free_memory_bytes = 0;
  std::size_t total_memory_bytes = 0;
  std::size_t allocated_device_bytes = 0;
};

struct HostAuxiliary {
  std::vector<float> min_in_static;
  std::vector<float> min_out_static;
  std::vector<Offset> incoming_rowptr;
  std::vector<Index> incoming_sources;
  std::vector<float> incoming_weights;
  std::vector<Offset> outgoing_sorted_edge_ids;

  void release_upload_data() {
    std::vector<float>().swap(min_in_static);
    std::vector<float>().swap(min_out_static);
    std::vector<Offset>().swap(incoming_rowptr);
    std::vector<Index>().swap(incoming_sources);
    std::vector<float>().swap(incoming_weights);
    std::vector<Offset>().swap(outgoing_sorted_edge_ids);
  }
};

struct HostResult {
  std::vector<float> distances;
  std::vector<Offset> predecessor_edges;
};

struct ReconstructedPath {
  std::vector<Index> nodes;
  std::vector<Offset> csr_edges;
};

template <typename Rep, typename Period>
double milliseconds(std::chrono::duration<Rep, Period> duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
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
  output += ".preds_GPU.";
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
         "<graph.csrbin>.preds_GPU.<predicate-mode>.paths.jsonl\n"
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
  if (graph.rows >
      static_cast<Offset>(std::numeric_limits<Index>::max())) {
    throw std::runtime_error(
        "CSR graph has too many vertices for 32-bit node indices");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  if (static_cast<std::uint64_t>(graph.nnz) >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "CSR graph has too many edges for this host");
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
    const Index destination = graph.colind[edge];
    if (destination < 0 ||
        static_cast<Offset>(destination) >= graph.cols) {
      throw std::runtime_error(
          "CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error(
          "CSR values must be finite nonnegative weights");
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

struct IncomingRecord {
  Index source = -1;
  float weight = 0.0f;
  Offset edge = -1;
};

HostAuxiliary build_host_auxiliary(const CsrGraph& graph,
                                   PredicateMode mode) {
  const std::size_t vertex_count = static_cast<std::size_t>(graph.rows);
  const std::size_t edge_count = static_cast<std::size_t>(graph.nnz);
  const float infinity = std::numeric_limits<float>::infinity();
  HostAuxiliary auxiliary;

  if (mode == PredicateMode::kInStatic ||
      mode == PredicateMode::kInStaticOrOutStatic) {
    auxiliary.min_in_static.assign(vertex_count, infinity);
  }
  if (mode == PredicateMode::kOutStatic ||
      mode == PredicateMode::kInStaticOrOutStatic) {
    auxiliary.min_out_static.assign(vertex_count, infinity);
  }

  if (!auxiliary.min_in_static.empty() ||
      !auxiliary.min_out_static.empty()) {
    for (Offset raw_source = 0; raw_source < graph.rows; ++raw_source) {
      const std::size_t source = static_cast<std::size_t>(raw_source);
      for (Offset edge = graph.rowptr[source];
           edge < graph.rowptr[source + 1];
           ++edge) {
        const std::size_t edge_index = static_cast<std::size_t>(edge);
        const std::size_t destination =
            static_cast<std::size_t>(graph.colind[edge_index]);
        const float weight = graph.values[edge_index];
        if (!auxiliary.min_in_static.empty()) {
          auxiliary.min_in_static[destination] =
              std::min(auxiliary.min_in_static[destination], weight);
        }
        if (!auxiliary.min_out_static.empty()) {
          auxiliary.min_out_static[source] =
              std::min(auxiliary.min_out_static[source], weight);
        }
      }
    }
  }

  if (uses_simple_in_predicate(mode)) {
    auxiliary.incoming_rowptr.assign(vertex_count + 1, 0);
    for (Index destination : graph.colind) {
      ++auxiliary.incoming_rowptr[
          static_cast<std::size_t>(destination) + 1];
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      auxiliary.incoming_rowptr[vertex + 1] +=
          auxiliary.incoming_rowptr[vertex];
    }

    std::vector<Offset> cursor = auxiliary.incoming_rowptr;
    std::vector<IncomingRecord> records(edge_count);
    for (Offset raw_source = 0; raw_source < graph.rows; ++raw_source) {
      const Index source = static_cast<Index>(raw_source);
      const std::size_t source_index = static_cast<std::size_t>(source);
      for (Offset edge = graph.rowptr[source_index];
           edge < graph.rowptr[source_index + 1];
           ++edge) {
        const std::size_t edge_index = static_cast<std::size_t>(edge);
        const Index destination = graph.colind[edge_index];
        const Offset position =
            cursor[static_cast<std::size_t>(destination)]++;
        records[static_cast<std::size_t>(position)] =
            IncomingRecord{source, graph.values[edge_index], edge};
      }
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      const std::size_t begin = static_cast<std::size_t>(
          auxiliary.incoming_rowptr[vertex]);
      const std::size_t end = static_cast<std::size_t>(
          auxiliary.incoming_rowptr[vertex + 1]);
      std::sort(
          records.begin() + static_cast<std::ptrdiff_t>(begin),
          records.begin() + static_cast<std::ptrdiff_t>(end),
          [](const IncomingRecord& left, const IncomingRecord& right) {
            if (left.weight != right.weight) {
              return left.weight < right.weight;
            }
            return left.edge < right.edge;
          });
    }
    auxiliary.incoming_sources.resize(edge_count);
    auxiliary.incoming_weights.resize(edge_count);
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
      auxiliary.incoming_sources[edge] = records[edge].source;
      auxiliary.incoming_weights[edge] = records[edge].weight;
    }
  }

  if (uses_simple_out_predicate(mode)) {
    auxiliary.outgoing_sorted_edge_ids.resize(edge_count);
    for (Offset edge = 0; edge < graph.nnz; ++edge) {
      auxiliary.outgoing_sorted_edge_ids[static_cast<std::size_t>(edge)] =
          edge;
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      const std::size_t begin =
          static_cast<std::size_t>(graph.rowptr[vertex]);
      const std::size_t end =
          static_cast<std::size_t>(graph.rowptr[vertex + 1]);
      std::sort(
          auxiliary.outgoing_sorted_edge_ids.begin() +
              static_cast<std::ptrdiff_t>(begin),
          auxiliary.outgoing_sorted_edge_ids.begin() +
              static_cast<std::ptrdiff_t>(end),
          [&graph](Offset left, Offset right) {
            const float left_weight =
                graph.values[static_cast<std::size_t>(left)];
            const float right_weight =
                graph.values[static_cast<std::size_t>(right)];
            if (left_weight != right_weight) {
              return left_weight < right_weight;
            }
            return left < right;
          });
    }
  }
  return auxiliary;
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + hipGetErrorString(status));
  }
}

template <typename T>
std::size_t checked_bytes(std::size_t count, const char* what) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(what) + " byte count overflows");
  }
  return count * sizeof(T);
}

std::size_t checked_add_bytes(std::size_t left,
                              std::size_t right,
                              const char* what) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(what) + " byte count overflows");
  }
  return left + right;
}

void checked_accumulate(std::uint64_t& destination,
                        std::uint64_t value,
                        const char* what) {
  if (destination >
      std::numeric_limits<std::uint64_t>::max() - value) {
    throw std::overflow_error(
        std::string(what) + " counter overflows uint64");
  }
  destination += value;
}

template <typename T>
void add_allocation_bytes(std::size_t count,
                          std::size_t& total,
                          const char* what) {
  total = checked_add_bytes(total, checked_bytes<T>(count, what), what);
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    release();
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept {
    move_from(std::move(other));
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  void allocate(std::size_t count, const char* what) {
    if (count_ != 0 || pointer_ != nullptr) {
      throw std::logic_error("device buffer allocated more than once");
    }
    if (count == 0) {
      return;
    }
    check_hip(
        hipMalloc(reinterpret_cast<void**>(&pointer_),
                  checked_bytes<T>(count, what)),
        what);
    count_ = count;
  }

  T* get() const {
    return pointer_;
  }
  std::size_t size() const {
    return count_;
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(count_, other.count_);
  }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)hipFree(pointer_);
    }
    pointer_ = nullptr;
    count_ = 0;
  }
  void move_from(DeviceBuffer&& other) noexcept {
    pointer_ = other.pointer_;
    count_ = other.count_;
    other.pointer_ = nullptr;
    other.count_ = 0;
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() {
    if (pointer_ != nullptr) {
      (void)hipHostFree(pointer_);
    }
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void allocate(std::size_t count, const char* what) {
    if (pointer_ != nullptr || count_ != 0) {
      throw std::logic_error("pinned buffer allocated more than once");
    }
    if (count == 0) {
      return;
    }
    check_hip(
        hipHostMalloc(reinterpret_cast<void**>(&pointer_),
                      checked_bytes<T>(count, what),
                      hipHostMallocDefault),
        what);
    count_ = count;
  }

  T* get() const {
    return pointer_;
  }
  std::size_t size() const {
    return count_;
  }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

class HipEvent {
 public:
  explicit HipEvent(bool timing_enabled = true) {
    const unsigned flags =
        timing_enabled ? hipEventDefault : hipEventDisableTiming;
    check_hip(hipEventCreateWithFlags(&event_, flags),
              "hipEventCreateWithFlags");
  }
  ~HipEvent() {
    if (event_ != nullptr) {
      (void)hipEventDestroy(event_);
    }
  }
  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  hipEvent_t get() const {
    return event_;
  }

 private:
  hipEvent_t event_ = nullptr;
};

class HipStream {
 public:
  HipStream() {
    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
  }
  ~HipStream() {
    if (stream_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
      (void)hipStreamDestroy(stream_);
    }
  }
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  hipStream_t get() const {
    return stream_;
  }

 private:
  hipStream_t stream_ = nullptr;
};

float host_bits_float(unsigned int bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct PhaseStatus {
  unsigned int min_distance_bits = kInfinityBits;
  unsigned int out_threshold_bits = kInfinityBits;
  int selected_count = 0;
  int next_count = 0;
  int settled_target_count = 0;
  int error_status = 0;
  unsigned long long selected_edge_count = 0;
  unsigned long long successful_updates = 0;
  unsigned long long expanded_edge_count = 0;
};

enum DeviceError : int {
  kDeviceSuccess = 0,
  kDeviceBadFringeVertex = 1,
  kDeviceBadFringeState = 2,
  kDeviceSelectionOverflow = 3,
  kDeviceSettleTransition = 4,
  kDeviceBadDestination = 5,
  kDeviceNextFringeOverflow = 6,
  kDeviceExpandedEdgeOverflow = 7,
};

template <typename DeviceOffset>
struct DeviceGraphView {
  Offset rows = 0;
  Offset nnz = 0;
  const DeviceOffset* rowptr = nullptr;
  const Index* colind = nullptr;
  const float* values = nullptr;

  const float* min_in_static = nullptr;
  const float* min_out_static = nullptr;

  const DeviceOffset* incoming_rowptr = nullptr;
  const Index* incoming_sources = nullptr;
  const float* incoming_weights = nullptr;
  DeviceOffset* incoming_cursor = nullptr;

  const DeviceOffset* outgoing_sorted_edge_ids = nullptr;
  DeviceOffset* outgoing_cursor = nullptr;
};

template <typename DeviceOffset>
struct ExpandedEdge {
  DeviceOffset edge = 0;
  Index source = -1;
};

template <typename DeviceOffset, bool Packed>
struct DeviceWorkspaceView {
  unsigned long long* packed_state = nullptr;
  float* wide_distances = nullptr;
  Offset* wide_predecessor_edges = nullptr;
  int* wide_locks = nullptr;
  int* vertex_states = nullptr;
  Index* current_fringe = nullptr;
  Index* next_fringe = nullptr;
  Index* selected = nullptr;
  ExpandedEdge<DeviceOffset>* expanded_edges = nullptr;
  PhaseStatus* status = nullptr;
};

__device__ __forceinline__ bool device_mode_uses_in(int mode) {
  return mode == static_cast<int>(PredicateMode::kInSimple) ||
         mode == static_cast<int>(PredicateMode::kInSimpleOrOutSimple) ||
         mode == static_cast<int>(PredicateMode::kInStatic) ||
         mode == static_cast<int>(PredicateMode::kInStaticOrOutStatic);
}

__device__ __forceinline__ bool device_mode_uses_out(int mode) {
  return mode == static_cast<int>(PredicateMode::kOutSimple) ||
         mode == static_cast<int>(PredicateMode::kInSimpleOrOutSimple) ||
         mode == static_cast<int>(PredicateMode::kOutStatic) ||
         mode == static_cast<int>(PredicateMode::kInStaticOrOutStatic);
}

__device__ __forceinline__ bool device_mode_simple_in(int mode) {
  return mode == static_cast<int>(PredicateMode::kInSimple) ||
         mode == static_cast<int>(PredicateMode::kInSimpleOrOutSimple);
}

__device__ __forceinline__ bool device_mode_simple_out(int mode) {
  return mode == static_cast<int>(PredicateMode::kOutSimple) ||
         mode == static_cast<int>(PredicateMode::kInSimpleOrOutSimple);
}

__device__ __forceinline__ void publish_device_error(PhaseStatus* status,
                                                     int error) {
  (void)atomicCAS(&status->error_status, kDeviceSuccess, error);
}

__device__ __forceinline__ unsigned long long pack_state_bits(
    unsigned int distance_bits,
    unsigned int predecessor_edge) {
  return (static_cast<unsigned long long>(distance_bits) << 32) |
         static_cast<unsigned long long>(predecessor_edge);
}

__device__ __forceinline__ float unpack_state_distance(
    unsigned long long state) {
  return __uint_as_float(static_cast<unsigned int>(state >> 32));
}

template <bool Packed>
__device__ __forceinline__ float load_device_distance(
    const unsigned long long* packed_state,
    const float* wide_distances,
    Index vertex) {
  if constexpr (Packed) {
    return unpack_state_distance(
        packed_state[static_cast<Offset>(vertex)]);
  } else {
    return wide_distances[static_cast<Offset>(vertex)];
  }
}

template <bool Packed, typename DeviceOffset>
__device__ __forceinline__ bool strict_atomic_relax(
    unsigned long long* packed_state,
    float* wide_distances,
    Offset* wide_predecessor_edges,
    int* wide_locks,
    Index destination,
    float candidate,
    DeviceOffset predecessor_edge) {
  if constexpr (Packed) {
    const unsigned int candidate_bits = __float_as_uint(candidate);
    unsigned long long observed =
        packed_state[static_cast<Offset>(destination)];
    while (candidate_bits <
           static_cast<unsigned int>(observed >> 32)) {
      const unsigned long long desired =
          pack_state_bits(candidate_bits,
                          static_cast<unsigned int>(predecessor_edge));
      const unsigned long long prior =
          atomicCAS(&packed_state[static_cast<Offset>(destination)],
                    observed,
                    desired);
      if (prior == observed) {
        return true;
      }
      observed = prior;
    }
    return false;
  } else {
    int* const lock = &wide_locks[static_cast<Offset>(destination)];
    // Keep acquisition, the critical section, and release in the same loop
    // branch. A conventional "spin until acquired, then unlock after the
    // loop" can deadlock when contending lanes in one wave diverge and the
    // winning lane cannot reach the release.
    for (;;) {
      if (atomicCAS(lock, 0, 1) == 0) {
        bool improved = false;
        float& distance =
            wide_distances[static_cast<Offset>(destination)];
        if (candidate < distance) {
          distance = candidate;
          wide_predecessor_edges[static_cast<Offset>(destination)] =
              static_cast<Offset>(predecessor_edge);
          improved = true;
        }
        __threadfence();
        atomicExch(lock, 0);
        return improved;
      }
      // A lane that failed the try-lock immediately reconverges; whichever
      // lane acquired it has already released it before peers retry.
    }
  }
}

template <typename DeviceOffset>
__device__ __forceinline__ float device_in_minimum(
    const DeviceGraphView<DeviceOffset>& graph,
    int mode,
    Index vertex) {
  if (device_mode_simple_in(mode)) {
    const DeviceOffset cursor =
        graph.incoming_cursor[static_cast<Offset>(vertex)];
    const DeviceOffset end =
        graph.incoming_rowptr[static_cast<Offset>(vertex) + 1];
    return cursor < end
               ? graph.incoming_weights[static_cast<Offset>(cursor)]
               : __uint_as_float(kInfinityBits);
  }
  return graph.min_in_static[static_cast<Offset>(vertex)];
}

template <typename DeviceOffset>
__device__ __forceinline__ float device_out_minimum(
    const DeviceGraphView<DeviceOffset>& graph,
    int mode,
    Index vertex) {
  if (device_mode_simple_out(mode)) {
    const DeviceOffset cursor =
        graph.outgoing_cursor[static_cast<Offset>(vertex)];
    const DeviceOffset end =
        graph.rowptr[static_cast<Offset>(vertex) + 1];
    if (cursor >= end) {
      return __uint_as_float(kInfinityBits);
    }
    const DeviceOffset edge =
        graph.outgoing_sorted_edge_ids[static_cast<Offset>(cursor)];
    return graph.values[static_cast<Offset>(edge)];
  }
  return graph.min_out_static[static_cast<Offset>(vertex)];
}

template <typename DeviceOffset, bool Packed>
__global__ void initialize_sssp_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    Index source) {
  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  for (Offset vertex = global_thread; vertex < graph.rows; vertex += stride) {
    workspace.vertex_states[vertex] =
        vertex == static_cast<Offset>(source) ? kFringe : kUnexplored;
    if constexpr (Packed) {
      workspace.packed_state[vertex] =
          vertex == static_cast<Offset>(source)
              ? pack_state_bits(0u, kNoPackedPredecessor)
              : pack_state_bits(kInfinityBits, kNoPackedPredecessor);
    } else {
      workspace.wide_distances[vertex] =
          vertex == static_cast<Offset>(source)
              ? 0.0f
              : __uint_as_float(kInfinityBits);
      workspace.wide_predecessor_edges[vertex] = -1;
      workspace.wide_locks[vertex] = 0;
    }
    if (graph.incoming_cursor != nullptr) {
      graph.incoming_cursor[vertex] = graph.incoming_rowptr[vertex];
    }
    if (graph.outgoing_cursor != nullptr) {
      graph.outgoing_cursor[vertex] = graph.rowptr[vertex];
    }
  }
  if (global_thread == 0) {
    workspace.current_fringe[0] = source;
  }
}

__global__ void reset_phase_status_kernel(PhaseStatus* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->min_distance_bits = kInfinityBits;
    status->out_threshold_bits = kInfinityBits;
    status->selected_count = 0;
    status->next_count = 0;
    status->settled_target_count = 0;
    status->error_status = kDeviceSuccess;
    status->selected_edge_count = 0;
    status->successful_updates = 0;
    status->expanded_edge_count = 0;
  }
}

__global__ void count_settled_targets_kernel(
    const int* vertex_states,
    Offset rows,
    const Index* target_nodes,
    int target_count,
    PhaseStatus* status) {
  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  int local_count = 0;
  for (Offset position = global_thread;
       position < static_cast<Offset>(target_count);
       position += stride) {
    const Index target = target_nodes[position];
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      publish_device_error(status, kDeviceBadDestination);
      continue;
    }
    local_count +=
        vertex_states[static_cast<Offset>(target)] == kSettled ? 1 : 0;
  }
  if (local_count != 0) {
    atomicAdd(&status->settled_target_count, local_count);
  }
}

template <bool Packed>
__global__ void gather_target_distances_kernel(
    const unsigned long long* packed_state,
    const float* wide_distances,
    Offset rows,
    const Index* target_nodes,
    int target_count,
    float* target_distances,
    PhaseStatus* status) {
  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  for (Offset position = global_thread;
       position < static_cast<Offset>(target_count);
       position += stride) {
    const Index target = target_nodes[position];
    if (target < 0 || static_cast<Offset>(target) >= rows) {
      publish_device_error(status, kDeviceBadDestination);
      continue;
    }
    target_distances[position] = load_device_distance<Packed>(
        packed_state, wide_distances, target);
  }
}

template <typename DeviceOffset, bool Packed>
__global__ void threshold_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    int mode,
    int fringe_count) {
  extern __shared__ unsigned int shared_values[];
  unsigned int* const shared_distance = shared_values;
  unsigned int* const shared_out = shared_values + blockDim.x;
  const int thread = threadIdx.x;
  unsigned int local_distance = kInfinityBits;
  unsigned int local_out = kInfinityBits;

  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + thread;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  for (Offset position = global_thread;
       position < static_cast<Offset>(fringe_count);
       position += stride) {
    const Index vertex = workspace.current_fringe[position];
    if (vertex < 0 || static_cast<Offset>(vertex) >= graph.rows) {
      publish_device_error(workspace.status, kDeviceBadFringeVertex);
      continue;
    }
    if (workspace.vertex_states[static_cast<Offset>(vertex)] != kFringe) {
      publish_device_error(workspace.status, kDeviceBadFringeState);
      continue;
    }

    if (device_mode_simple_in(mode)) {
      DeviceOffset cursor =
          graph.incoming_cursor[static_cast<Offset>(vertex)];
      const DeviceOffset end =
          graph.incoming_rowptr[static_cast<Offset>(vertex) + 1];
      while (cursor < end) {
        const Index source =
            graph.incoming_sources[static_cast<Offset>(cursor)];
        if (workspace.vertex_states[static_cast<Offset>(source)] !=
            kSettled) {
          break;
        }
        ++cursor;
      }
      graph.incoming_cursor[static_cast<Offset>(vertex)] = cursor;
    }

    if (device_mode_simple_out(mode)) {
      DeviceOffset cursor =
          graph.outgoing_cursor[static_cast<Offset>(vertex)];
      const DeviceOffset end =
          graph.rowptr[static_cast<Offset>(vertex) + 1];
      while (cursor < end) {
        const DeviceOffset edge =
            graph.outgoing_sorted_edge_ids[static_cast<Offset>(cursor)];
        const Index destination =
            graph.colind[static_cast<Offset>(edge)];
        if (workspace.vertex_states[
                static_cast<Offset>(destination)] != kSettled) {
          break;
        }
        ++cursor;
      }
      graph.outgoing_cursor[static_cast<Offset>(vertex)] = cursor;
    }

    const float distance = load_device_distance<Packed>(
        workspace.packed_state,
        workspace.wide_distances,
        vertex);
    const unsigned int distance_bits = __float_as_uint(distance);
    if (distance_bits < local_distance) {
      local_distance = distance_bits;
    }
    if (device_mode_uses_out(mode)) {
      const float threshold_candidate =
          distance + device_out_minimum(graph, mode, vertex);
      const unsigned int threshold_bits =
          __float_as_uint(threshold_candidate);
      if (threshold_bits < local_out) {
        local_out = threshold_bits;
      }
    }
  }

  shared_distance[thread] = local_distance;
  shared_out[thread] = local_out;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (thread < width) {
      if (shared_distance[thread + width] < shared_distance[thread]) {
        shared_distance[thread] = shared_distance[thread + width];
      }
      if (shared_out[thread + width] < shared_out[thread]) {
        shared_out[thread] = shared_out[thread + width];
      }
    }
    __syncthreads();
  }
  if (thread == 0) {
    atomicMin(&workspace.status->min_distance_bits, shared_distance[0]);
    if (device_mode_uses_out(mode)) {
      atomicMin(&workspace.status->out_threshold_bits, shared_out[0]);
    }
  }
}

template <typename DeviceOffset, bool Packed>
__global__ void select_and_compact_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    int mode,
    int fringe_count) {
  extern __shared__ unsigned char shared_raw[];
  int* const selected_scan =
      reinterpret_cast<int*>(shared_raw);
  int* const unselected_scan = selected_scan + blockDim.x;
  auto* const edge_scan =
      reinterpret_cast<unsigned long long*>(
          unselected_scan + blockDim.x);
  __shared__ int selected_base;
  __shared__ int unselected_base;

  const int thread = threadIdx.x;
  const Offset block_span =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  for (Offset block_base =
           static_cast<Offset>(blockIdx.x) * blockDim.x;
       block_base < static_cast<Offset>(fringe_count);
       block_base += block_span) {
    const Offset position = block_base + thread;
    Index vertex = -1;
    int is_selected = 0;
    int is_unselected = 0;
    unsigned long long selected_edges = 0;

    if (position < static_cast<Offset>(fringe_count)) {
      vertex = workspace.current_fringe[position];
      if (vertex < 0 || static_cast<Offset>(vertex) >= graph.rows) {
        publish_device_error(workspace.status, kDeviceBadFringeVertex);
      } else if (workspace.vertex_states[
                     static_cast<Offset>(vertex)] != kFringe) {
        publish_device_error(workspace.status, kDeviceBadFringeState);
      } else {
        const float distance = load_device_distance<Packed>(
            workspace.packed_state,
            workspace.wide_distances,
            vertex);
        bool selected = false;
        if (device_mode_uses_in(mode)) {
          const float minimum =
              device_in_minimum(graph, mode, vertex);
          const float threshold =
              __uint_as_float(workspace.status->min_distance_bits);
          selected = distance - minimum <= threshold;
        }
        if (device_mode_uses_out(mode)) {
          const float threshold =
              __uint_as_float(workspace.status->out_threshold_bits);
          selected = selected || distance <= threshold;
        }
        is_selected = selected ? 1 : 0;
        is_unselected = selected ? 0 : 1;
        if (selected) {
          const DeviceOffset begin =
              graph.rowptr[static_cast<Offset>(vertex)];
          const DeviceOffset end =
              graph.rowptr[static_cast<Offset>(vertex) + 1];
          selected_edges =
              static_cast<unsigned long long>(end - begin);
        }
      }
    }

    selected_scan[thread] = is_selected;
    unselected_scan[thread] = is_unselected;
    edge_scan[thread] = selected_edges;
    __syncthreads();

    for (int offset = 1;
         offset < static_cast<int>(blockDim.x);
         offset *= 2) {
      const int selected_add =
          thread >= offset ? selected_scan[thread - offset] : 0;
      const int unselected_add =
          thread >= offset ? unselected_scan[thread - offset] : 0;
      const unsigned long long edge_add =
          thread >= offset ? edge_scan[thread - offset] : 0;
      __syncthreads();
      selected_scan[thread] += selected_add;
      unselected_scan[thread] += unselected_add;
      edge_scan[thread] += edge_add;
      __syncthreads();
    }

    const int selected_total = selected_scan[blockDim.x - 1];
    const int unselected_total = unselected_scan[blockDim.x - 1];
    if (thread == 0) {
      selected_base =
          atomicAdd(&workspace.status->selected_count, selected_total);
      unselected_base =
          atomicAdd(&workspace.status->next_count, unselected_total);
      if (edge_scan[blockDim.x - 1] != 0) {
        atomicAdd(&workspace.status->selected_edge_count,
                  edge_scan[blockDim.x - 1]);
      }
      if (selected_base < 0 || unselected_base < 0 ||
          static_cast<Offset>(selected_base) + selected_total >
              graph.rows ||
          static_cast<Offset>(unselected_base) + unselected_total >
              graph.rows) {
        publish_device_error(
            workspace.status, kDeviceSelectionOverflow);
      }
    }
    __syncthreads();

    if (is_selected != 0 &&
        static_cast<Offset>(selected_base) + selected_total <=
            graph.rows) {
      workspace.selected[
          selected_base + selected_scan[thread] - 1] = vertex;
    }
    if (is_unselected != 0 &&
        static_cast<Offset>(unselected_base) + unselected_total <=
            graph.rows) {
      workspace.next_fringe[
          unselected_base + unselected_scan[thread] - 1] = vertex;
    }
    __syncthreads();
  }
}

template <typename DeviceOffset, bool Packed>
__global__ void mark_selected_settled_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    int selected_count) {
  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  for (Offset position = global_thread;
       position < static_cast<Offset>(selected_count);
       position += stride) {
    const Index vertex = workspace.selected[position];
    if (vertex < 0 || static_cast<Offset>(vertex) >= graph.rows) {
      publish_device_error(workspace.status, kDeviceBadFringeVertex);
      continue;
    }
    const int prior = atomicCAS(
        &workspace.vertex_states[static_cast<Offset>(vertex)],
        kFringe,
        kSettled);
    if (prior != kFringe) {
      publish_device_error(
          workspace.status, kDeviceSettleTransition);
    }
  }
}

template <typename DeviceOffset, bool Packed>
__device__ __forceinline__ unsigned int relax_one_edge(
    const DeviceGraphView<DeviceOffset>& graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    Index source,
    DeviceOffset edge) {
  const Index destination =
      graph.colind[static_cast<Offset>(edge)];
  if (destination < 0 ||
      static_cast<Offset>(destination) >= graph.rows) {
    publish_device_error(workspace.status, kDeviceBadDestination);
    return 0;
  }
  if (workspace.vertex_states[
          static_cast<Offset>(destination)] == kSettled) {
    return 0;
  }
  const float source_distance = load_device_distance<Packed>(
      workspace.packed_state,
      workspace.wide_distances,
      source);
  const float candidate =
      source_distance + graph.values[static_cast<Offset>(edge)];
  const bool improved = strict_atomic_relax<Packed>(
      workspace.packed_state,
      workspace.wide_distances,
      workspace.wide_predecessor_edges,
      workspace.wide_locks,
      destination,
      candidate,
      edge);
  if (!improved) {
    return 0;
  }

  const int prior_state = atomicCAS(
      &workspace.vertex_states[static_cast<Offset>(destination)],
      kUnexplored,
      kFringe);
  if (prior_state == kUnexplored) {
    const int slot = atomicAdd(&workspace.status->next_count, 1);
    if (slot < 0 || static_cast<Offset>(slot) >= graph.rows) {
      publish_device_error(
          workspace.status, kDeviceNextFringeOverflow);
    } else {
      workspace.next_fringe[slot] = destination;
    }
  } else if (prior_state == kSettled) {
    publish_device_error(workspace.status, kDeviceSettleTransition);
  }
  return 1;
}

template <typename DeviceOffset, bool Packed>
__global__ void tiny_row_relax_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    int selected_count) {
  extern __shared__ unsigned long long success_counts[];
  const int thread = threadIdx.x;
  for (Offset selected_position = static_cast<Offset>(blockIdx.x);
       selected_position < static_cast<Offset>(selected_count);
       selected_position += static_cast<Offset>(gridDim.x)) {
    const Index source = workspace.selected[selected_position];
    const DeviceOffset begin =
        graph.rowptr[static_cast<Offset>(source)];
    const DeviceOffset end =
        graph.rowptr[static_cast<Offset>(source) + 1];
    unsigned long long local_success = 0;
    // Iterate in 64 bits even in compact mode: a row may legally end at
    // UINT32_MAX, where incrementing a uint32 edge iterator would wrap.
    for (Offset raw_edge =
             static_cast<Offset>(begin) + static_cast<Offset>(thread);
         raw_edge < static_cast<Offset>(end);
         raw_edge += static_cast<Offset>(blockDim.x)) {
      local_success += relax_one_edge(
          graph,
          workspace,
          source,
          static_cast<DeviceOffset>(raw_edge));
    }
    success_counts[thread] = local_success;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
      if (thread < width) {
        success_counts[thread] += success_counts[thread + width];
      }
      __syncthreads();
    }
    if (thread == 0 && success_counts[0] != 0) {
      atomicAdd(&workspace.status->successful_updates,
                success_counts[0]);
    }
    __syncthreads();
  }
}

template <typename DeviceOffset, bool Packed>
__global__ void expand_selected_edges_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    int selected_count) {
  __shared__ unsigned long long expanded_base;
  const int thread = threadIdx.x;
  for (Offset selected_position = static_cast<Offset>(blockIdx.x);
       selected_position < static_cast<Offset>(selected_count);
       selected_position += static_cast<Offset>(gridDim.x)) {
    const Index source = workspace.selected[selected_position];
    const DeviceOffset begin =
        graph.rowptr[static_cast<Offset>(source)];
    const DeviceOffset end =
        graph.rowptr[static_cast<Offset>(source) + 1];
    const unsigned long long degree =
        static_cast<unsigned long long>(end - begin);
    if (thread == 0) {
      expanded_base =
          atomicAdd(&workspace.status->expanded_edge_count, degree);
      const unsigned long long capacity =
          static_cast<unsigned long long>(graph.nnz);
      if (expanded_base > capacity ||
          degree > capacity - expanded_base) {
        publish_device_error(
            workspace.status, kDeviceExpandedEdgeOverflow);
      }
    }
    __syncthreads();
    const unsigned long long capacity =
        static_cast<unsigned long long>(graph.nnz);
    if (expanded_base <= capacity &&
        degree <= capacity - expanded_base) {
      for (unsigned long long local_edge =
               static_cast<unsigned long long>(thread);
           local_edge < degree;
           local_edge +=
               static_cast<unsigned long long>(blockDim.x)) {
        const DeviceOffset edge =
            begin + static_cast<DeviceOffset>(local_edge);
        workspace.expanded_edges[expanded_base + local_edge] =
            ExpandedEdge<DeviceOffset>{edge, source};
      }
    }
    __syncthreads();
  }
}

template <typename DeviceOffset, bool Packed>
__global__ void flat_edge_relax_kernel(
    DeviceGraphView<DeviceOffset> graph,
    DeviceWorkspaceView<DeviceOffset, Packed> workspace,
    Offset selected_edge_count) {
  extern __shared__ unsigned long long success_counts[];
  const Offset global_thread =
      static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Offset stride =
      static_cast<Offset>(gridDim.x) * blockDim.x;
  unsigned long long local_success = 0;
  for (Offset position = global_thread;
       position < selected_edge_count;
       position += stride) {
    const ExpandedEdge<DeviceOffset> item =
        workspace.expanded_edges[position];
    local_success += relax_one_edge(
        graph, workspace, item.source, item.edge);
  }
  success_counts[threadIdx.x] = local_success;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < static_cast<unsigned int>(width)) {
      success_counts[threadIdx.x] +=
          success_counts[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && success_counts[0] != 0) {
    atomicAdd(&workspace.status->successful_updates,
              success_counts[0]);
  }
}

template <typename DeviceOffset, bool Packed>
struct GpuResources {
  DeviceBuffer<DeviceOffset> rowptr;
  DeviceBuffer<Index> colind;
  DeviceBuffer<float> values;
  DeviceBuffer<float> min_in_static;
  DeviceBuffer<float> min_out_static;
  DeviceBuffer<DeviceOffset> incoming_rowptr;
  DeviceBuffer<Index> incoming_sources;
  DeviceBuffer<float> incoming_weights;
  DeviceBuffer<DeviceOffset> incoming_cursor;
  DeviceBuffer<DeviceOffset> outgoing_sorted_edge_ids;
  DeviceBuffer<DeviceOffset> outgoing_cursor;

  DeviceBuffer<unsigned long long> packed_state;
  DeviceBuffer<float> wide_distances;
  DeviceBuffer<Offset> wide_predecessor_edges;
  DeviceBuffer<int> wide_locks;
  DeviceBuffer<int> vertex_states;
  DeviceBuffer<Index> current_fringe;
  DeviceBuffer<Index> next_fringe;
  DeviceBuffer<Index> selected;
  DeviceBuffer<ExpandedEdge<DeviceOffset>> expanded_edges;
  DeviceBuffer<PhaseStatus> status;
  DeviceBuffer<Index> target_nodes;
  DeviceBuffer<float> target_distances;

  PinnedBuffer<PhaseStatus> host_status;
  PinnedBuffer<float> host_target_distances;
  PinnedBuffer<unsigned long long> host_packed_state;
  PinnedBuffer<float> host_wide_distances;
  PinnedBuffer<Offset> host_wide_predecessor_edges;
  HipEvent reset_start;
  HipEvent reset_end;
  HipEvent predicate_start;
  HipEvent predicate_end;
  HipEvent settlement_start;
  HipEvent settlement_end;
  HipEvent relaxation_start;
  HipEvent relaxation_end;
  HipEvent transfer_start;
  HipEvent transfer_end;
  HipEvent control_ready{false};

  // Declared last so it synchronizes and is destroyed before any event,
  // pinned buffer, or device allocation is released.
  HipStream stream;

  DeviceGraphView<DeviceOffset> graph_view(const CsrGraph& graph) {
    DeviceGraphView<DeviceOffset> view;
    view.rows = graph.rows;
    view.nnz = graph.nnz;
    view.rowptr = rowptr.get();
    view.colind = colind.get();
    view.values = values.get();
    view.min_in_static = min_in_static.get();
    view.min_out_static = min_out_static.get();
    view.incoming_rowptr = incoming_rowptr.get();
    view.incoming_sources = incoming_sources.get();
    view.incoming_weights = incoming_weights.get();
    view.incoming_cursor = incoming_cursor.get();
    view.outgoing_sorted_edge_ids =
        outgoing_sorted_edge_ids.get();
    view.outgoing_cursor = outgoing_cursor.get();
    return view;
  }

  DeviceWorkspaceView<DeviceOffset, Packed> workspace_view() {
    DeviceWorkspaceView<DeviceOffset, Packed> view;
    view.packed_state = packed_state.get();
    view.wide_distances = wide_distances.get();
    view.wide_predecessor_edges = wide_predecessor_edges.get();
    view.wide_locks = wide_locks.get();
    view.vertex_states = vertex_states.get();
    view.current_fringe = current_fringe.get();
    view.next_fringe = next_fringe.get();
    view.selected = selected.get();
    view.expanded_edges = expanded_edges.get();
    view.status = status.get();
    return view;
  }
};

struct LaunchPolicy {
  int block_size = 0;
  int initialization_blocks = 0;
  int threshold_block_cap = 0;
  int selection_block_cap = 0;
  int mark_block_cap = 0;
  int tiny_block_cap = 0;
  int expansion_block_cap = 0;
  int flat_block_cap = 0;
  std::uint64_t tiny_edge_limit = 0;
};

DeviceMetadata query_device_metadata() {
  DeviceMetadata metadata;
  check_hip(hipGetDevice(&metadata.device), "hipGetDevice");
  check_hip(
      hipGetDeviceProperties(&metadata.properties, metadata.device),
      "hipGetDeviceProperties");
  check_hip(
      hipDeviceGetAttribute(
          &metadata.integrated,
          hipDeviceAttributeIntegrated,
          metadata.device),
      "hipDeviceGetAttribute integrated");
  if (metadata.properties.warpSize <= 0 ||
      metadata.properties.multiProcessorCount <= 0 ||
      metadata.properties.maxThreadsPerBlock <= 0) {
    throw std::runtime_error(
        "HIP device reported invalid launch properties");
  }
  check_hip(
      hipMemGetInfo(
          &metadata.free_memory_bytes,
          &metadata.total_memory_bytes),
      "hipMemGetInfo");

  int target = metadata.properties.warpSize * 4;
  target = std::max(64, std::min(256, target));
  target = std::min(target, metadata.properties.maxThreadsPerBlock);
  int power_of_two = 1;
  while (power_of_two <= target / 2) {
    power_of_two *= 2;
  }
  metadata.block_size = power_of_two;
  return metadata;
}

template <typename Kernel>
int occupancy_blocks_per_compute_unit(Kernel kernel,
                                      int block_size,
                                      std::size_t shared_bytes,
                                      const char* what) {
  int active_blocks = 0;
  check_hip(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, kernel, block_size, shared_bytes),
      what);
  if (active_blocks <= 0) {
    throw std::runtime_error(
        std::string(what) + " returned nonpositive occupancy");
  }
  return active_blocks;
}

int capped_resident_grid(int active_per_compute_unit,
                         int compute_units,
                         int oversubscription) {
  const long long blocks =
      static_cast<long long>(active_per_compute_unit) *
      static_cast<long long>(compute_units) *
      static_cast<long long>(oversubscription);
  return static_cast<int>(std::max<long long>(
      1, std::min<long long>(kMaximumGridBlocks, blocks)));
}

template <typename DeviceOffset, bool Packed>
LaunchPolicy make_launch_policy(const DeviceMetadata& metadata) {
  LaunchPolicy policy;
  policy.block_size = metadata.block_size;
  const int compute_units = metadata.properties.multiProcessorCount;
  const std::size_t threshold_shared =
      static_cast<std::size_t>(policy.block_size) *
      2 * sizeof(unsigned int);
  const std::size_t selection_shared =
      static_cast<std::size_t>(policy.block_size) *
      (2 * sizeof(int) + sizeof(unsigned long long));
  const std::size_t relaxation_shared =
      static_cast<std::size_t>(policy.block_size) *
      sizeof(unsigned long long);

  const int threshold_occupancy =
      occupancy_blocks_per_compute_unit(
          threshold_kernel<DeviceOffset, Packed>,
          policy.block_size,
          threshold_shared,
          "threshold-kernel occupancy");
  const int selection_occupancy =
      occupancy_blocks_per_compute_unit(
          select_and_compact_kernel<DeviceOffset, Packed>,
          policy.block_size,
          selection_shared,
          "selection-kernel occupancy");
  const int mark_occupancy =
      occupancy_blocks_per_compute_unit(
          mark_selected_settled_kernel<DeviceOffset, Packed>,
          policy.block_size,
          0,
          "mark-kernel occupancy");
  const int tiny_occupancy =
      occupancy_blocks_per_compute_unit(
          tiny_row_relax_kernel<DeviceOffset, Packed>,
          policy.block_size,
          relaxation_shared,
          "tiny-relaxation-kernel occupancy");
  const int expansion_occupancy =
      occupancy_blocks_per_compute_unit(
          expand_selected_edges_kernel<DeviceOffset, Packed>,
          policy.block_size,
          0,
          "edge-expansion-kernel occupancy");
  const int flat_occupancy =
      occupancy_blocks_per_compute_unit(
          flat_edge_relax_kernel<DeviceOffset, Packed>,
          policy.block_size,
          relaxation_shared,
          "flat-relaxation-kernel occupancy");

  policy.threshold_block_cap =
      capped_resident_grid(threshold_occupancy, compute_units, 2);
  policy.selection_block_cap =
      capped_resident_grid(selection_occupancy, compute_units, 2);
  policy.mark_block_cap =
      capped_resident_grid(mark_occupancy, compute_units, 2);
  policy.tiny_block_cap =
      capped_resident_grid(tiny_occupancy, compute_units, 1);
  policy.expansion_block_cap =
      capped_resident_grid(expansion_occupancy, compute_units, 1);
  policy.flat_block_cap =
      capped_resident_grid(flat_occupancy, compute_units, 2);
  policy.initialization_blocks =
      std::min(kMaximumGridBlocks, std::max(1, compute_units * 8));

  // Edge work determines whether row-cooperative execution is sparse enough.
  // This lets a single very high-degree row switch to the flattened path,
  // while tiny_blocks still scales from the actual selected-vertex count.
  policy.tiny_edge_limit =
      static_cast<std::uint64_t>(policy.block_size) *
      static_cast<std::uint64_t>(compute_units) * 2;
  return policy;
}

int blocks_for_items(std::uint64_t items,
                     int block_size,
                     int block_cap) {
  if (items == 0) {
    return 1;
  }
  const std::uint64_t needed =
      (items + static_cast<std::uint64_t>(block_size) - 1) /
      static_cast<std::uint64_t>(block_size);
  return static_cast<int>(std::max<std::uint64_t>(
      1,
      std::min<std::uint64_t>(
          needed, static_cast<std::uint64_t>(block_cap))));
}

template <typename Launcher>
void launch_sssp_kernel(LaunchCounters& counters,
                        const char* what,
                        Launcher&& launcher) {
  launcher();
  check_hip(hipGetLastError(), what);
  ++counters.sssp;
}

float elapsed_event_ms(const HipEvent& start,
                       const HipEvent& end,
                       const char* what) {
  float elapsed = 0.0f;
  check_hip(
      hipEventElapsedTime(&elapsed, start.get(), end.get()), what);
  return elapsed;
}

void wait_for_control_event(hipStream_t stream,
                            HipEvent& control_ready,
                            const char* what) {
  check_hip(
      hipEventRecord(control_ready.get(), stream),
      "hipEventRecord control-ready");
  check_hip(hipEventSynchronize(control_ready.get()), what);
}

template <typename T>
void enqueue_h2d(T* destination,
                 const T* source,
                 std::size_t count,
                 hipStream_t stream,
                 TransferStatistics& transfers,
                 const char* what) {
  if (count == 0) {
    return;
  }
  const std::size_t bytes = checked_bytes<T>(count, what);
  check_hip(
      hipMemcpyAsync(
          destination,
          source,
          bytes,
          hipMemcpyHostToDevice,
          stream),
      what);
  transfers.h2d_bytes += static_cast<std::uint64_t>(bytes);
  ++transfers.h2d_operations;
}

template <typename T>
void enqueue_d2h(T* destination,
                 const T* source,
                 std::size_t count,
                 hipStream_t stream,
                 TransferStatistics& transfers,
                 const char* what) {
  if (count == 0) {
    return;
  }
  const std::size_t bytes = checked_bytes<T>(count, what);
  check_hip(
      hipMemcpyAsync(
          destination,
          source,
          bytes,
          hipMemcpyDeviceToHost,
          stream),
      what);
  transfers.d2h_bytes += static_cast<std::uint64_t>(bytes);
  ++transfers.d2h_operations;
}

template <typename DeviceOffset, bool Packed>
std::size_t required_device_bytes(const CsrGraph& graph,
                                  const HostAuxiliary& auxiliary,
                                  PredicateMode mode,
                                  std::size_t target_capacity) {
  const std::size_t vertices = static_cast<std::size_t>(graph.rows);
  const std::size_t edges = static_cast<std::size_t>(graph.nnz);
  std::size_t bytes = 0;
  add_allocation_bytes<DeviceOffset>(
      vertices + 1, bytes, "device rowptr");
  add_allocation_bytes<Index>(edges, bytes, "device colind");
  add_allocation_bytes<float>(edges, bytes, "device values");
  add_allocation_bytes<float>(
      auxiliary.min_in_static.size(), bytes, "static incoming minima");
  add_allocation_bytes<float>(
      auxiliary.min_out_static.size(), bytes, "static outgoing minima");
  add_allocation_bytes<DeviceOffset>(
      auxiliary.incoming_rowptr.size(),
      bytes,
      "incoming rowptr");
  add_allocation_bytes<Index>(
      auxiliary.incoming_sources.size(),
      bytes,
      "incoming sources");
  add_allocation_bytes<float>(
      auxiliary.incoming_weights.size(),
      bytes,
      "incoming weights");
  if (!auxiliary.incoming_rowptr.empty()) {
    add_allocation_bytes<DeviceOffset>(
        vertices, bytes, "incoming cursors");
  }
  add_allocation_bytes<DeviceOffset>(
      auxiliary.outgoing_sorted_edge_ids.size(),
      bytes,
      "sorted outgoing edge ids");
  if (uses_simple_out_predicate(mode)) {
    add_allocation_bytes<DeviceOffset>(
        vertices, bytes, "outgoing cursors");
  }

  if constexpr (Packed) {
    add_allocation_bytes<unsigned long long>(
        vertices, bytes, "packed state");
  } else {
    add_allocation_bytes<float>(
        vertices, bytes, "wide distances");
    add_allocation_bytes<Offset>(
        vertices, bytes, "wide predecessor edges");
    add_allocation_bytes<int>(
        vertices, bytes, "wide state locks");
  }
  add_allocation_bytes<int>(
      vertices, bytes, "vertex states");
  add_allocation_bytes<Index>(
      vertices, bytes, "current fringe");
  add_allocation_bytes<Index>(
      vertices, bytes, "next fringe");
  add_allocation_bytes<Index>(
      vertices, bytes, "selected batch");
  add_allocation_bytes<ExpandedEdge<DeviceOffset>>(
      edges, bytes, "expanded selected edges");
  add_allocation_bytes<PhaseStatus>(
      1, bytes, "phase status");
  add_allocation_bytes<Index>(
      target_capacity, bytes, "target nodes");
  add_allocation_bytes<float>(
      target_capacity, bytes, "target distances");
  return bytes;
}

template <typename DeviceOffset>
std::vector<DeviceOffset> convert_offsets(
    const std::vector<Offset>& source,
    const char* what) {
  std::vector<DeviceOffset> converted;
  converted.reserve(source.size());
  for (Offset value : source) {
    if (value < 0 ||
        static_cast<unsigned long long>(value) >
            static_cast<unsigned long long>(
                std::numeric_limits<DeviceOffset>::max())) {
      throw std::overflow_error(
          std::string(what) + " cannot be represented on the device");
    }
    converted.push_back(static_cast<DeviceOffset>(value));
  }
  return converted;
}

template <typename DeviceOffset, bool Packed>
void allocate_and_upload(
    const CsrGraph& graph,
    HostAuxiliary& auxiliary,
    PredicateMode mode,
    std::size_t target_capacity,
    bool reserve_path_scratch,
    GpuResources<DeviceOffset, Packed>& resources,
    DeviceMetadata& metadata,
    TransferStatistics& transfers) {
  const std::size_t vertices = static_cast<std::size_t>(graph.rows);
  const std::size_t edges = static_cast<std::size_t>(graph.nnz);
  const std::size_t required =
      required_device_bytes<DeviceOffset, Packed>(
          graph, auxiliary, mode, target_capacity);
  const std::size_t desired_reserve =
      std::max<std::size_t>(
          static_cast<std::size_t>(256) * 1024 * 1024,
          metadata.total_memory_bytes / 20);
  const std::size_t reserve =
      std::min(desired_reserve, metadata.free_memory_bytes / 4);
  if (required > metadata.free_memory_bytes - reserve) {
    std::ostringstream message;
    message << "preds_GPU needs " << required
            << " device bytes, but only "
            << metadata.free_memory_bytes
            << " are free and " << reserve
            << " bytes are reserved";
    throw std::runtime_error(message.str());
  }
  metadata.allocated_device_bytes = required;

  resources.rowptr.allocate(vertices + 1, "hipMalloc rowptr");
  resources.colind.allocate(edges, "hipMalloc colind");
  resources.values.allocate(edges, "hipMalloc values");
  resources.min_in_static.allocate(
      auxiliary.min_in_static.size(),
      "hipMalloc static incoming minima");
  resources.min_out_static.allocate(
      auxiliary.min_out_static.size(),
      "hipMalloc static outgoing minima");
  resources.incoming_rowptr.allocate(
      auxiliary.incoming_rowptr.size(),
      "hipMalloc incoming rowptr");
  resources.incoming_sources.allocate(
      auxiliary.incoming_sources.size(),
      "hipMalloc incoming sources");
  resources.incoming_weights.allocate(
      auxiliary.incoming_weights.size(),
      "hipMalloc incoming weights");
  if (!auxiliary.incoming_rowptr.empty()) {
    resources.incoming_cursor.allocate(
        vertices, "hipMalloc incoming cursors");
  }
  resources.outgoing_sorted_edge_ids.allocate(
      auxiliary.outgoing_sorted_edge_ids.size(),
      "hipMalloc sorted outgoing edge ids");
  if (uses_simple_out_predicate(mode)) {
    resources.outgoing_cursor.allocate(
        vertices, "hipMalloc outgoing cursors");
  }

  if constexpr (Packed) {
    resources.packed_state.allocate(
        vertices, "hipMalloc packed state");
  } else {
    resources.wide_distances.allocate(
        vertices, "hipMalloc wide distances");
    resources.wide_predecessor_edges.allocate(
        vertices, "hipMalloc wide predecessor edges");
    resources.wide_locks.allocate(
        vertices, "hipMalloc wide state locks");
  }
  resources.vertex_states.allocate(
      vertices, "hipMalloc vertex states");
  resources.current_fringe.allocate(
      vertices, "hipMalloc current fringe");
  resources.next_fringe.allocate(
      vertices, "hipMalloc next fringe");
  resources.selected.allocate(
      vertices, "hipMalloc selected batch");
  resources.expanded_edges.allocate(
      edges, "hipMalloc expanded selected edges");
  resources.status.allocate(1, "hipMalloc phase status");
  resources.target_nodes.allocate(
      target_capacity, "hipMalloc target nodes");
  resources.target_distances.allocate(
      target_capacity, "hipMalloc target distances");
  resources.host_status.allocate(1, "hipHostMalloc phase status");
  resources.host_target_distances.allocate(
      target_capacity, "hipHostMalloc target distances");
  if (reserve_path_scratch) {
    if constexpr (Packed) {
      resources.host_packed_state.allocate(
          vertices, "hipHostMalloc reusable packed path state");
    } else {
      resources.host_wide_distances.allocate(
          vertices, "hipHostMalloc reusable path distances");
      resources.host_wide_predecessor_edges.allocate(
          vertices, "hipHostMalloc reusable path predecessors");
    }
  }

  std::vector<DeviceOffset> rowptr =
      convert_offsets<DeviceOffset>(graph.rowptr, "CSR rowptr");
  std::vector<DeviceOffset> incoming_rowptr =
      convert_offsets<DeviceOffset>(
          auxiliary.incoming_rowptr, "incoming rowptr");
  std::vector<DeviceOffset> outgoing_sorted =
      convert_offsets<DeviceOffset>(
          auxiliary.outgoing_sorted_edge_ids,
          "sorted outgoing edge id");

  hipStream_t stream = resources.stream.get();
  check_hip(
      hipEventRecord(resources.transfer_start.get(), stream),
      "record H2D start");
  enqueue_h2d(
      resources.rowptr.get(),
      rowptr.data(),
      rowptr.size(),
      stream,
      transfers,
      "copy rowptr H2D");
  enqueue_h2d(
      resources.colind.get(),
      graph.colind.data(),
      graph.colind.size(),
      stream,
      transfers,
      "copy colind H2D");
  enqueue_h2d(
      resources.values.get(),
      graph.values.data(),
      graph.values.size(),
      stream,
      transfers,
      "copy values H2D");
  enqueue_h2d(
      resources.min_in_static.get(),
      auxiliary.min_in_static.data(),
      auxiliary.min_in_static.size(),
      stream,
      transfers,
      "copy static incoming minima H2D");
  enqueue_h2d(
      resources.min_out_static.get(),
      auxiliary.min_out_static.data(),
      auxiliary.min_out_static.size(),
      stream,
      transfers,
      "copy static outgoing minima H2D");
  enqueue_h2d(
      resources.incoming_rowptr.get(),
      incoming_rowptr.data(),
      incoming_rowptr.size(),
      stream,
      transfers,
      "copy incoming rowptr H2D");
  enqueue_h2d(
      resources.incoming_sources.get(),
      auxiliary.incoming_sources.data(),
      auxiliary.incoming_sources.size(),
      stream,
      transfers,
      "copy incoming sources H2D");
  enqueue_h2d(
      resources.incoming_weights.get(),
      auxiliary.incoming_weights.data(),
      auxiliary.incoming_weights.size(),
      stream,
      transfers,
      "copy incoming weights H2D");
  enqueue_h2d(
      resources.outgoing_sorted_edge_ids.get(),
      outgoing_sorted.data(),
      outgoing_sorted.size(),
      stream,
      transfers,
      "copy sorted outgoing edge ids H2D");
  check_hip(
      hipEventRecord(resources.transfer_end.get(), stream),
      "record H2D end");
  wait_for_control_event(
      stream, resources.control_ready, "wait for graph H2D");
  transfers.h2d_ms += elapsed_event_ms(
      resources.transfer_start,
      resources.transfer_end,
      "measure graph H2D");

  auxiliary.release_upload_data();
}

void check_device_phase_status(const PhaseStatus& status,
                               const char* stage) {
  if (status.error_status == kDeviceSuccess) {
    return;
  }
  const char* detail = "unknown device error";
  switch (status.error_status) {
    case kDeviceBadFringeVertex:
      detail = "fringe queue contains an out-of-range vertex";
      break;
    case kDeviceBadFringeState:
      detail = "fringe queue contains a non-fringe vertex";
      break;
    case kDeviceSelectionOverflow:
      detail = "selection compaction exceeded queue capacity";
      break;
    case kDeviceSettleTransition:
      detail = "invalid or duplicate vertex-state transition";
      break;
    case kDeviceBadDestination:
      detail = "relaxation saw an out-of-range destination";
      break;
    case kDeviceNextFringeOverflow:
      detail = "next-fringe queue exceeded vertex capacity";
      break;
    case kDeviceExpandedEdgeOverflow:
      detail = "expanded selected-edge queue exceeded edge capacity";
      break;
    default:
      break;
  }
  throw std::runtime_error(
      std::string(stage) + ": " + detail);
}

void print_run_metadata(const DeviceMetadata& metadata,
                        const LaunchPolicy& policy,
                        bool compact_offsets,
                        bool packed_state) {
  std::cout << "Selected HIP device: "
            << metadata.properties.name << '\n'
            << "Device architecture: "
            << metadata.properties.gcnArchName << '\n'
            << "Integrated device: "
            << (metadata.integrated != 0 ? "yes" : "no") << '\n'
            << "Wavefront size: "
            << metadata.properties.warpSize << '\n'
            << "Compute-unit count: "
            << metadata.properties.multiProcessorCount << '\n'
            << "Threads per block: " << policy.block_size << '\n'
            << "CSR device offsets: "
            << (compact_offsets ? "32-bit" : "64-bit") << '\n'
            << "Atomic predecessor state: "
            << (packed_state
                    ? "packed 64-bit distance/edge CAS"
                    : "wide locked distance/int64-edge pair")
            << '\n'
            << "Device allocation bytes: "
            << metadata.allocated_device_bytes << '\n'
            << "Tiny-phase heuristic: <= "
            << policy.tiny_edge_limit << " selected edges\n";
}

void print_statistics(const Statistics& statistics,
                      const std::string& label,
                      PredicateMode mode,
                      const TransferStatistics& transfers,
                      const LaunchCounters& launches,
                      double preprocessing_ms,
                      double total_runtime_ms,
                      double end_to_end_ms) {
  std::cout << "\n=== preds_GPU statistics (" << label << ") ===\n"
            << "Predicate mode: " << predicate_mode_name(mode) << '\n'
            << "Number of completed phases: " << statistics.phases << '\n'
            << "Vertices settled in current phase: "
            << statistics.vertices_settled_current_phase << '\n'
            << "Vertices settled per phase (phases 1.."
            << statistics.phases << "): [";
  for (std::size_t index = 0;
       index < std::min(
                   statistics.vertices_settled_per_phase.size(),
                   static_cast<std::size_t>(statistics.phases));
       ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << statistics.vertices_settled_per_phase[index];
  }
  if (statistics.vertices_settled_per_phase.size() <
      static_cast<std::size_t>(statistics.phases)) {
    std::cout << "not collected";
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
            << "Predicate-evaluation time (ms): "
            << statistics.predicate_evaluation_ms << '\n'
            << "Relaxation time (ms): "
            << statistics.relaxation_ms << '\n'
            << "Per-source reset time (ms): "
            << statistics.reset_ms << '\n'
            << "Per-source reset kernel-launch count: "
            << statistics.reset_kernel_launches << '\n';
  if (statistics.predicate_evaluation_ms > 0.0) {
    std::cout << "Relaxation/predicate time ratio: "
              << (statistics.relaxation_ms /
                  statistics.predicate_evaluation_ms)
              << '\n';
  } else {
    std::cout << "Relaxation/predicate time ratio: n/a\n";
  }
  std::cout << "GPU kernel-launch count: " << launches.sssp << '\n'
            << "Output reconstruction kernel-launch count: "
            << launches.output_reconstruction << '\n'
            << "Tiny relaxation phases: "
            << statistics.tiny_relaxation_phases << '\n'
            << "Large relaxation phases: "
            << statistics.large_relaxation_phases << '\n'
            << "Preprocessing time (ms): "
            << preprocessing_ms << '\n'
            << "Transfer time (ms): " << transfers.total_ms() << '\n'
            << "  H2D transfer time (ms): " << transfers.h2d_ms << '\n'
            << "  D2H transfer time (ms): " << transfers.d2h_ms << '\n'
            << "  H2D bytes: " << transfers.h2d_bytes << '\n'
            << "  D2H bytes: " << transfers.d2h_bytes << '\n'
            << "  H2D operations: " << transfers.h2d_operations << '\n'
            << "  D2H operations: " << transfers.d2h_operations << '\n'
            << "Elapsed time (ms): "
            << statistics.elapsed_ms << '\n'
            << "Total runtime (ms): " << total_runtime_ms
            << '\n'
            << "End-to-end time (ms): "
            << end_to_end_ms << '\n';
}

Offset predecessor_source_for_edge(const CsrGraph& graph, Offset edge) {
  if (edge < 0 || edge >= graph.nnz) {
    throw std::runtime_error(
        "predecessor edge is outside the outgoing CSR");
  }
  const auto upper = std::upper_bound(
      graph.rowptr.begin(), graph.rowptr.end(), edge);
  if (upper == graph.rowptr.begin() || upper == graph.rowptr.end()) {
    throw std::runtime_error(
        "could not locate predecessor edge in outgoing CSR");
  }
  return static_cast<Offset>(
      std::distance(graph.rowptr.begin(), upper) - 1);
}

ReconstructedPath reconstruct_path(const CsrGraph& graph,
                                   const HostResult& result,
                                   Index source,
                                   Index target) {
  ReconstructedPath path;
  const float distance =
      result.distances[static_cast<std::size_t>(target)];
  if (distance == std::numeric_limits<float>::infinity()) {
    return path;
  }

  Index current = target;
  path.nodes.push_back(current);
  for (Offset hop = 0; hop < graph.rows && current != source; ++hop) {
    const std::size_t current_index = static_cast<std::size_t>(current);
    const Offset predecessor_edge =
        result.predecessor_edges[current_index];
    const Offset raw_predecessor =
        predecessor_source_for_edge(graph, predecessor_edge);
    if (raw_predecessor < 0 || raw_predecessor >= graph.rows) {
      throw std::runtime_error(
          "predecessor vertex is outside the graph");
    }
    const Index predecessor = static_cast<Index>(raw_predecessor);
    if (predecessor_edge <
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

void validate_captured_target_path_against_graph(
    const CsrGraph& graph,
    Index source,
    const CapturedTargetPathView& path) {
  if (source < 0 || static_cast<Offset>(source) >= graph.rows) {
    throw std::out_of_range(
        "captured path source is outside the CSR graph");
  }
  if (path.target < 0 ||
      static_cast<Offset>(path.target) >= graph.rows) {
    throw std::out_of_range(
        "captured path target is outside the CSR graph");
  }
  if (!path.path_captured) {
    throw std::runtime_error(
        "captured target result is missing its path-captured marker");
  }
  if (!path.reached) {
    if (path.distance != std::numeric_limits<float>::infinity()) {
      throw std::runtime_error(
          "unreachable captured target must have positive-infinite distance");
    }
    if (path.node_count != 0 || path.csr_edge_count != 0) {
      throw std::runtime_error(
          "unreachable captured target contains path data");
    }
    return;
  }

  if (!std::isfinite(path.distance) || path.distance < 0.0f) {
    throw std::runtime_error(
        "reached captured target has an invalid distance");
  }
  if (path.node_count == 0 ||
      path.node_count > static_cast<std::size_t>(graph.rows)) {
    throw std::runtime_error(
        "captured target path has an invalid node count");
  }
  if (path.nodes == nullptr ||
      (path.csr_edge_count != 0 && path.csr_edges == nullptr)) {
    throw std::runtime_error(
        "captured target path has missing host storage");
  }
  if (path.csr_edge_count != path.node_count - 1 ||
      path.nodes[0] != source ||
      path.nodes[path.node_count - 1] != path.target) {
    throw std::runtime_error(
        "captured target path endpoints or lengths are inconsistent");
  }

  float reconstructed_distance = 0.0f;
  for (std::size_t hop = 0; hop < path.csr_edge_count; ++hop) {
    const Index from = path.nodes[hop];
    const Index to = path.nodes[hop + 1];
    if (from < 0 || static_cast<Offset>(from) >= graph.rows ||
        to < 0 || static_cast<Offset>(to) >= graph.rows) {
      throw std::runtime_error(
          "captured target path contains an out-of-range vertex");
    }
    const Offset edge = path.csr_edges[hop];
    if (edge < 0 || edge >= graph.nnz ||
        edge < graph.rowptr[static_cast<std::size_t>(from)] ||
        edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
        graph.colind[static_cast<std::size_t>(edge)] != to) {
      throw std::runtime_error(
          "captured target path contains an invalid CSR edge");
    }
    reconstructed_distance =
        reconstructed_distance + graph.values[static_cast<std::size_t>(edge)];
  }
  if (reconstructed_distance != path.distance) {
    throw std::runtime_error(
        "captured target path weight does not match its distance");
  }
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
                              Offset graph_rows,
                              Offset graph_nnz,
                              const std::vector<TargetResult>& targets,
                              Index source,
                              PredicateMode mode) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error(
        "could not open paths output: " + output_path.string());
  }
  const std::size_t paths_to_write = targets.size();
  output << "{\"type\":\"metadata\""
         << ",\"format\":\"rips-sssp-paths-v1\""
         << ",\"producer\":\"preds_GPU\""
         << ",\"predicate_mode\":\"" << predicate_mode_name(mode) << "\""
         << ",\"node_count\":" << graph_rows
         << ",\"edge_count\":" << graph_nnz
         << ",\"route_request_count\":1"
         << ",\"selected_source_count\":1"
         << ",\"selected_query_count\":" << paths_to_write
         << ",\"edge_orientation\":\"outgoing\""
         << ",\"description\":\"row u stores directed edges u -> v\""
         << "}\n";

  std::size_t paths_written = 0;
  for (const TargetResult& target_result : targets) {
    ReconstructedPath path;
    path.nodes = target_result.nodes;
    path.csr_edges = target_result.csr_edges;
    write_path_record(
        output,
        source,
        target_result.target,
        target_result.reached,
        target_result.distance,
        path);
    ++paths_written;
  }
  output.close();
  if (!output) {
    throw std::runtime_error(
        "failed while writing paths output: " + output_path.string());
  }
  return paths_written;
}

template <typename DeviceOffset, bool Packed>
void copy_final_state(
    const CsrGraph& graph,
    GpuResources<DeviceOffset, Packed>& resources,
    TransferStatistics& transfers,
    HostResult& result) {
  // Correctness-first optional-output fallback: copy the complete final
  // distance/predecessor state once for CPU path reconstruction. This is
  // never used for phase control and is skipped entirely without --print.
  const std::size_t vertices = static_cast<std::size_t>(graph.rows);
  if (result.distances.size() != vertices ||
      result.predecessor_edges.size() != vertices) {
    throw std::logic_error(
        "reusable host path state does not match graph size");
  }
  hipStream_t stream = resources.stream.get();

  if constexpr (Packed) {
    if (resources.host_packed_state.size() != vertices) {
      throw std::logic_error(
          "packed path capture requested without reserved scratch");
    }
    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record final packed D2H start");
    enqueue_d2h(
        resources.host_packed_state.get(),
        resources.packed_state.get(),
        vertices,
        stream,
        transfers,
        "copy final packed state D2H");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record final packed D2H end");
    wait_for_control_event(
        stream, resources.control_ready, "wait for final packed state");
    transfers.d2h_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure final packed D2H");
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
      const unsigned long long state =
          resources.host_packed_state.get()[vertex];
      const unsigned int distance_bits =
          static_cast<unsigned int>(state >> 32);
      result.distances[vertex] = host_bits_float(distance_bits);
      result.predecessor_edges[vertex] = -1;
      if (distance_bits != kInfinityBits) {
        const unsigned int edge =
            static_cast<unsigned int>(state);
        if (edge != kNoPackedPredecessor) {
          result.predecessor_edges[vertex] =
              static_cast<Offset>(edge);
        }
      }
    }
  } else {
    if (resources.host_wide_distances.size() != vertices ||
        resources.host_wide_predecessor_edges.size() != vertices) {
      throw std::logic_error(
          "wide path capture requested without reserved scratch");
    }
    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record final wide D2H start");
    enqueue_d2h(
        resources.host_wide_distances.get(),
        resources.wide_distances.get(),
        vertices,
        stream,
        transfers,
        "copy final wide distances D2H");
    enqueue_d2h(
        resources.host_wide_predecessor_edges.get(),
        resources.wide_predecessor_edges.get(),
        vertices,
        stream,
        transfers,
        "copy final wide predecessors D2H");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record final wide D2H end");
    wait_for_control_event(
        stream, resources.control_ready, "wait for final wide state");
    transfers.d2h_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure final wide D2H");
    std::copy(
        resources.host_wide_distances.get(),
        resources.host_wide_distances.get() + vertices,
        result.distances.begin());
    std::copy(
        resources.host_wide_predecessor_edges.get(),
        resources.host_wide_predecessor_edges.get() + vertices,
        result.predecessor_edges.begin());
  }
}

struct GpuRunResult {
  Statistics statistics;
  TransferStatistics transfers;
  LaunchCounters launches;
  std::vector<TargetResult> targets;
  TerminationReason termination = TerminationReason::kFullConvergence;
  bool fully_converged = false;
  bool all_targets_confirmed = false;
  double total_runtime_ms = 0.0;
  bool requested_snapshot_reached = false;
  Statistics requested_snapshot_statistics;
  TransferStatistics requested_snapshot_transfers;
  LaunchCounters requested_snapshot_launches;
  double requested_snapshot_runtime_ms = 0.0;
};

template <typename DeviceOffset, bool Packed>
GpuRunResult run_gpu_typed(
    const CsrGraph& graph,
    PredicateMode mode,
    const RunRequest& request,
    GpuResources<DeviceOffset, Packed>& resources,
    const LaunchPolicy& policy,
    HostResult* reusable_host_result) {
  const auto run_begin = Clock::now();
  GpuRunResult run_result;
  const std::size_t target_count = request.unique_targets.size();
  if (target_count > resources.target_nodes.size() ||
      target_count > resources.target_distances.size() ||
      target_count > resources.host_target_distances.size()) {
    throw std::invalid_argument(
        "preds-gpu target count exceeds reserved workload capacity");
  }
  if (target_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "preds-gpu target count exceeds the signed 32-bit device domain");
  }

  hipStream_t stream = resources.stream.get();
  if (target_count != 0) {
    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record target H2D start");
    enqueue_h2d(
        resources.target_nodes.get(),
        request.unique_targets.data(),
        target_count,
        stream,
        run_result.transfers,
        "copy target nodes H2D");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record target H2D end");
    wait_for_control_event(
        stream, resources.control_ready, "wait for target nodes H2D");
    run_result.transfers.h2d_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure target nodes H2D");
  }

  DeviceGraphView<DeviceOffset> device_graph =
      resources.graph_view(graph);
  DeviceWorkspaceView<DeviceOffset, Packed> initial_workspace =
      resources.workspace_view();
  const int initialization_blocks = blocks_for_items(
      static_cast<std::uint64_t>(graph.rows),
      policy.block_size,
      policy.initialization_blocks);
  check_hip(
      hipEventRecord(resources.reset_start.get(), stream),
      "record SSSP reset start");
  launch_sssp_kernel(
      run_result.launches,
      "launch SSSP initialization",
      [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                initialize_sssp_kernel<DeviceOffset, Packed>),
            dim3(static_cast<unsigned int>(initialization_blocks)),
            dim3(static_cast<unsigned int>(policy.block_size)),
            0,
            stream,
            device_graph,
            initial_workspace,
            request.source);
      });
  check_hip(
      hipEventRecord(resources.reset_end.get(), stream),
      "record SSSP reset end");
  wait_for_control_event(
      stream, resources.control_ready, "wait for SSSP initialization");
  run_result.statistics.reset_ms = elapsed_event_ms(
      resources.reset_start,
      resources.reset_end,
      "measure SSSP reset kernel");
  run_result.statistics.reset_kernel_launches = 1;

  if (request.capture_phase_histogram) {
    run_result.statistics.vertices_settled_per_phase.reserve(
        static_cast<std::size_t>(graph.rows));
  }
  int current_count = 1;
  while (current_count > 0) {
    const auto phase_begin = Clock::now();
    DeviceWorkspaceView<DeviceOffset, Packed> workspace =
        resources.workspace_view();

    check_hip(
        hipEventRecord(resources.predicate_start.get(), stream),
        "record predicate start");
    launch_sssp_kernel(
        run_result.launches,
        "launch phase-status reset",
        [&] {
          hipLaunchKernelGGL(
              reset_phase_status_kernel,
              dim3(1),
              dim3(1),
              0,
              stream,
              workspace.status);
        });

    const int threshold_blocks = blocks_for_items(
        static_cast<std::uint64_t>(current_count),
        policy.block_size,
        policy.threshold_block_cap);
    const std::size_t threshold_shared =
        static_cast<std::size_t>(policy.block_size) *
        2 * sizeof(unsigned int);
    launch_sssp_kernel(
        run_result.launches,
        "launch predicate-threshold reduction",
        [&] {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(
                  threshold_kernel<DeviceOffset, Packed>),
              dim3(static_cast<unsigned int>(threshold_blocks)),
              dim3(static_cast<unsigned int>(policy.block_size)),
              threshold_shared,
              stream,
              device_graph,
              workspace,
              static_cast<int>(mode),
              current_count);
        });

    const int selection_blocks = blocks_for_items(
        static_cast<std::uint64_t>(current_count),
        policy.block_size,
        policy.selection_block_cap);
    const std::size_t selection_shared =
        static_cast<std::size_t>(policy.block_size) *
        (2 * sizeof(int) + sizeof(unsigned long long));
    launch_sssp_kernel(
        run_result.launches,
        "launch predicate selection and compaction",
        [&] {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(
                  select_and_compact_kernel<DeviceOffset, Packed>),
              dim3(static_cast<unsigned int>(selection_blocks)),
              dim3(static_cast<unsigned int>(policy.block_size)),
              selection_shared,
              stream,
              device_graph,
              workspace,
              static_cast<int>(mode),
              current_count);
        });
    check_hip(
        hipEventRecord(resources.predicate_end.get(), stream),
        "record predicate end");

    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record predicate-status D2H start");
    enqueue_d2h(
        resources.host_status.get(),
        resources.status.get(),
        1,
        stream,
        run_result.transfers,
        "copy predicate phase status D2H");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record predicate-status D2H end");
    wait_for_control_event(
        stream,
        resources.control_ready,
        "wait for predicate phase status");
    run_result.statistics.predicate_evaluation_ms +=
        elapsed_event_ms(
            resources.predicate_start,
            resources.predicate_end,
            "measure predicate kernels");
    run_result.transfers.d2h_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure predicate-status D2H");

    const PhaseStatus predicate_status =
        resources.host_status.get()[0];
    check_device_phase_status(
        predicate_status, "predicate evaluation failed");
    if (predicate_status.selected_count <= 0) {
      throw std::runtime_error(
          std::string("predicate ") +
          predicate_mode_name(mode) +
          " selected no vertex while the fringe was nonempty at phase " +
          std::to_string(run_result.statistics.phases + 1));
    }
    if (predicate_status.selected_count > current_count ||
        predicate_status.next_count < 0 ||
        predicate_status.next_count >
            static_cast<int>(graph.rows) ||
        static_cast<long long>(predicate_status.selected_count) +
                static_cast<long long>(predicate_status.next_count) !=
            static_cast<long long>(current_count) ||
        predicate_status.selected_edge_count >
            static_cast<unsigned long long>(graph.nnz)) {
      throw std::runtime_error(
          "predicate compaction returned invalid phase counts");
    }

    const int mark_blocks = blocks_for_items(
        static_cast<std::uint64_t>(
            predicate_status.selected_count),
        policy.block_size,
        policy.mark_block_cap);
    check_hip(
        hipEventRecord(resources.settlement_start.get(), stream),
        "record settlement start");
    launch_sssp_kernel(
        run_result.launches,
        "launch selected-batch settlement",
        [&] {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(
                  mark_selected_settled_kernel<DeviceOffset, Packed>),
              dim3(static_cast<unsigned int>(mark_blocks)),
              dim3(static_cast<unsigned int>(policy.block_size)),
              0,
              stream,
              device_graph,
              workspace,
              predicate_status.selected_count);
        });
    check_hip(
        hipEventRecord(resources.settlement_end.get(), stream),
        "record settlement end");

    // The CPU reference charges complete-batch settlement to predicate
    // evaluation; outgoing-edge work starts only after this stream barrier.
    check_hip(
        hipEventRecord(resources.relaxation_start.get(), stream),
        "record relaxation start");
    const bool tiny_phase =
        predicate_status.selected_edge_count <=
        policy.tiny_edge_limit;
    const std::size_t relaxation_shared =
        static_cast<std::size_t>(policy.block_size) *
        sizeof(unsigned long long);
    if (tiny_phase) {
      const int tiny_blocks = std::max(
          1,
          std::min(
              predicate_status.selected_count,
              policy.tiny_block_cap));
      launch_sssp_kernel(
          run_result.launches,
          "launch tiny row-cooperative relaxation",
          [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    tiny_row_relax_kernel<DeviceOffset, Packed>),
                dim3(static_cast<unsigned int>(tiny_blocks)),
                dim3(static_cast<unsigned int>(policy.block_size)),
                relaxation_shared,
                stream,
                device_graph,
                workspace,
                predicate_status.selected_count);
          });
      ++run_result.statistics.tiny_relaxation_phases;
    } else {
      const int expansion_blocks = std::max(
          1,
          std::min(
              predicate_status.selected_count,
              policy.expansion_block_cap));
      launch_sssp_kernel(
          run_result.launches,
          "launch selected-edge expansion",
          [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    expand_selected_edges_kernel<
                        DeviceOffset,
                        Packed>),
                dim3(static_cast<unsigned int>(expansion_blocks)),
                dim3(static_cast<unsigned int>(policy.block_size)),
                0,
                stream,
                device_graph,
                workspace,
                predicate_status.selected_count);
          });
      const int flat_blocks = blocks_for_items(
          predicate_status.selected_edge_count,
          policy.block_size,
          policy.flat_block_cap);
      launch_sssp_kernel(
          run_result.launches,
          "launch flat selected-edge relaxation",
          [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    flat_edge_relax_kernel<DeviceOffset, Packed>),
                dim3(static_cast<unsigned int>(flat_blocks)),
                dim3(static_cast<unsigned int>(policy.block_size)),
                relaxation_shared,
                stream,
                device_graph,
                workspace,
                static_cast<Offset>(
                    predicate_status.selected_edge_count));
          });
      ++run_result.statistics.large_relaxation_phases;
    }
    const bool target_stop_enabled =
        request.early_stop && target_count != 0;
    if (target_stop_enabled) {
      const int target_blocks = blocks_for_items(
          static_cast<std::uint64_t>(target_count),
          policy.block_size,
          policy.mark_block_cap);
      launch_sssp_kernel(
          run_result.launches,
          "launch settled-target count",
          [&] {
            hipLaunchKernelGGL(
                count_settled_targets_kernel,
                dim3(static_cast<unsigned int>(target_blocks)),
                dim3(static_cast<unsigned int>(policy.block_size)),
                0,
                stream,
                workspace.vertex_states,
                graph.rows,
                resources.target_nodes.get(),
                static_cast<int>(target_count),
                workspace.status);
          });
    }
    check_hip(
        hipEventRecord(resources.relaxation_end.get(), stream),
        "record relaxation end");

    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record relaxation-status D2H start");
    enqueue_d2h(
        resources.host_status.get(),
        resources.status.get(),
        1,
        stream,
        run_result.transfers,
        "copy relaxation phase status D2H");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record relaxation-status D2H end");
    wait_for_control_event(
        stream,
        resources.control_ready,
        "wait for relaxation phase status");
    run_result.statistics.relaxation_ms += elapsed_event_ms(
        resources.relaxation_start,
        resources.relaxation_end,
        "measure relaxation kernels");
    run_result.statistics.predicate_evaluation_ms +=
        elapsed_event_ms(
            resources.settlement_start,
            resources.settlement_end,
            "measure settlement kernel");
    run_result.transfers.d2h_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure relaxation-status D2H");

    const PhaseStatus completed_status =
        resources.host_status.get()[0];
    check_device_phase_status(
        completed_status, "phase relaxation failed");
    if (completed_status.selected_count !=
            predicate_status.selected_count ||
        completed_status.selected_edge_count !=
            predicate_status.selected_edge_count ||
        completed_status.next_count < 0 ||
        completed_status.next_count >
            static_cast<int>(graph.rows)) {
      throw std::runtime_error(
          "relaxation returned invalid phase counts");
    }
    if (!tiny_phase &&
        completed_status.expanded_edge_count !=
            predicate_status.selected_edge_count) {
      throw std::runtime_error(
          "selected-edge expansion count is inconsistent");
    }
    if (completed_status.settled_target_count < 0 ||
        static_cast<std::size_t>(
            completed_status.settled_target_count) > target_count ||
        (!target_stop_enabled &&
         completed_status.settled_target_count != 0)) {
      throw std::runtime_error(
          "settled-target count is inconsistent");
    }

    ++run_result.statistics.phases;
    run_result.statistics.vertices_settled_current_phase =
        static_cast<std::uint64_t>(
            completed_status.selected_count);
    run_result.statistics.vertices_settled +=
        run_result.statistics.vertices_settled_current_phase;
    if (request.capture_phase_histogram) {
      run_result.statistics.vertices_settled_per_phase.push_back(
          static_cast<std::uint32_t>(
              run_result.statistics.vertices_settled_current_phase));
    }
    run_result.statistics.edges_examined +=
        completed_status.selected_edge_count;
    run_result.statistics.relaxations_attempted +=
        completed_status.selected_edge_count;
    run_result.statistics.successful_distance_updates +=
        completed_status.successful_updates;

    resources.current_fringe.swap(resources.next_fringe);
    current_count = completed_status.next_count;
    const auto phase_end = Clock::now();
    run_result.statistics.elapsed_ms +=
        milliseconds(phase_end - phase_begin);

    if (request.statistics_snapshot_phase != 0 &&
        run_result.statistics.phases ==
            request.statistics_snapshot_phase) {
      run_result.requested_snapshot_reached = true;
      run_result.requested_snapshot_statistics =
          run_result.statistics;
      run_result.requested_snapshot_transfers =
          run_result.transfers;
      run_result.requested_snapshot_launches =
          run_result.launches;
      run_result.requested_snapshot_runtime_ms =
          milliseconds(phase_end - run_begin);
    }

    if (target_stop_enabled && current_count > 0 &&
        static_cast<std::size_t>(
            completed_status.settled_target_count) == target_count) {
      run_result.termination =
          TerminationReason::kAllTargetsSettled;
      break;
    }
  }

  if (run_result.statistics.vertices_settled >
      static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error(
        "GPU settled more vertices than the graph contains");
  }
  run_result.fully_converged = current_count == 0;
  if (run_result.fully_converged) {
    run_result.termination =
        TerminationReason::kFullConvergence;
  }
  run_result.all_targets_confirmed =
      run_result.fully_converged ||
      run_result.termination ==
          TerminationReason::kAllTargetsSettled;

  if (target_count != 0) {
    DeviceWorkspaceView<DeviceOffset, Packed> workspace =
        resources.workspace_view();
    const int gather_blocks = blocks_for_items(
        static_cast<std::uint64_t>(target_count),
        policy.block_size,
        policy.mark_block_cap);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gather_target_distances_kernel<Packed>),
        dim3(static_cast<unsigned int>(gather_blocks)),
        dim3(static_cast<unsigned int>(policy.block_size)),
        0,
        stream,
        workspace.packed_state,
        workspace.wide_distances,
        graph.rows,
        resources.target_nodes.get(),
        static_cast<int>(target_count),
        resources.target_distances.get(),
        workspace.status);
    check_hip(
        hipGetLastError(), "launch compact target-distance gather");
    ++run_result.launches.output_reconstruction;

    check_hip(
        hipEventRecord(resources.transfer_start.get(), stream),
        "record compact target D2H start");
    enqueue_d2h(
        resources.host_target_distances.get(),
        resources.target_distances.get(),
        target_count,
        stream,
        run_result.transfers,
        "copy compact target distances D2H");
    check_hip(
        hipEventRecord(resources.transfer_end.get(), stream),
        "record compact target D2H end");
    wait_for_control_event(
        stream,
        resources.control_ready,
        "wait for compact target distances");
    run_result.transfers.d2h_ms += elapsed_event_ms(
        resources.transfer_start,
        resources.transfer_end,
        "measure compact target distances D2H");

    run_result.targets.reserve(target_count);
    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {
      TargetResult target_result;
      target_result.target =
          request.unique_targets[target_index];
      target_result.distance =
          resources.host_target_distances.get()[target_index];
      target_result.reached =
          std::isfinite(target_result.distance);
      run_result.targets.push_back(std::move(target_result));
    }
  }

  if (request.capture_paths && target_count != 0) {
    if (reusable_host_result == nullptr) {
      throw std::logic_error(
          "path capture requested without reserved host scratch");
    }
    copy_final_state(
        graph,
        resources,
        run_result.transfers,
        *reusable_host_result);
    for (TargetResult& target_result : run_result.targets) {
      target_result.path_captured = true;
      if (!target_result.reached) {
        continue;
      }
      ReconstructedPath path = reconstruct_path(
          graph,
          *reusable_host_result,
          request.source,
          target_result.target);
      target_result.nodes = std::move(path.nodes);
      target_result.csr_edges = std::move(path.csr_edges);
    }
  }

  run_result.total_runtime_ms =
      milliseconds(Clock::now() - run_begin);
  return run_result;
}

void add_transfer_statistics(TransferStatistics& destination,
                             const TransferStatistics& source) {
  destination.h2d_ms += source.h2d_ms;
  destination.d2h_ms += source.d2h_ms;
  checked_accumulate(
      destination.h2d_bytes,
      source.h2d_bytes,
      "cumulative H2D bytes");
  checked_accumulate(
      destination.d2h_bytes,
      source.d2h_bytes,
      "cumulative D2H bytes");
  checked_accumulate(
      destination.h2d_operations,
      source.h2d_operations,
      "cumulative H2D operations");
  checked_accumulate(
      destination.d2h_operations,
      source.d2h_operations,
      "cumulative D2H operations");
}

void add_launch_counters(LaunchCounters& destination,
                         const LaunchCounters& source) {
  checked_accumulate(
      destination.sssp,
      source.sssp,
      "cumulative SSSP kernel launches");
  checked_accumulate(
      destination.output_reconstruction,
      source.output_reconstruction,
      "cumulative output-reconstruction kernel launches");
}

void add_algorithm_statistics(Statistics& destination,
                              const Statistics& source) {
  checked_accumulate(
      destination.phases,
      source.phases,
      "cumulative SSSP phases");
  checked_accumulate(
      destination.vertices_settled,
      source.vertices_settled,
      "cumulative settled vertices");
  destination.vertices_settled_current_phase =
      source.vertices_settled_current_phase;
  checked_accumulate(
      destination.edges_examined,
      source.edges_examined,
      "cumulative examined edges");
  checked_accumulate(
      destination.relaxations_attempted,
      source.relaxations_attempted,
      "cumulative relaxation attempts");
  checked_accumulate(
      destination.successful_distance_updates,
      source.successful_distance_updates,
      "cumulative successful distance updates");
  destination.predicate_evaluation_ms +=
      source.predicate_evaluation_ms;
  destination.relaxation_ms += source.relaxation_ms;
  destination.reset_ms += source.reset_ms;
  destination.elapsed_ms += source.elapsed_ms;
  checked_accumulate(
      destination.reset_kernel_launches,
      source.reset_kernel_launches,
      "cumulative reset kernel launches");
  checked_accumulate(
      destination.tiny_relaxation_phases,
      source.tiny_relaxation_phases,
      "cumulative tiny-relaxation phases");
  checked_accumulate(
      destination.large_relaxation_phases,
      source.large_relaxation_phases,
      "cumulative large-relaxation phases");
}

RunResult public_run_result(GpuRunResult&& internal) {
  RunResult result;
  result.termination = internal.termination;
  result.fully_converged = internal.fully_converged;
  result.all_targets_confirmed =
      internal.all_targets_confirmed;
  result.phases_or_iterations = internal.statistics.phases;
  result.targets = std::move(internal.targets);
  result.telemetry.algorithm = std::move(internal.statistics);
  result.telemetry.transfers = internal.transfers;
  result.telemetry.launches = internal.launches;
  result.telemetry.runtime_ms = internal.total_runtime_ms;
  result.requested_snapshot_reached =
      internal.requested_snapshot_reached;
  if (internal.requested_snapshot_reached) {
    result.requested_snapshot.algorithm =
        std::move(internal.requested_snapshot_statistics);
    result.requested_snapshot.transfers =
        internal.requested_snapshot_transfers;
    result.requested_snapshot.launches =
        internal.requested_snapshot_launches;
    result.requested_snapshot.runtime_ms =
        internal.requested_snapshot_runtime_ms;
  }
  return result;
}

class PredsGpuTypedEngine {
 public:
  virtual ~PredsGpuTypedEngine() = default;
  virtual RunResult run(const RunRequest& request) = 0;
  virtual PredicateMode predicate_mode() const noexcept = 0;
  virtual Offset rows() const noexcept = 0;
  virtual Offset nnz() const noexcept = 0;
  virtual std::size_t target_capacity() const noexcept = 0;
  virtual bool path_scratch_reserved() const noexcept = 0;
  virtual void validate_captured_target_path(
      Index source,
      const CapturedTargetPathView& path) const = 0;
  virtual CumulativeStatistics cumulative_statistics() const = 0;
};

template <typename DeviceOffset, bool Packed>
class PredsGpuTypedEngineImpl final : public PredsGpuTypedEngine {
 public:
  PredsGpuTypedEngineImpl(CsrGraph graph,
                          PredicateMode mode,
                          std::size_t target_capacity,
                          bool reserve_path_scratch,
                          bool print_setup_metadata)
      : graph_(std::move(graph)),
        mode_(mode),
        target_capacity_(target_capacity),
        reserve_path_scratch_(reserve_path_scratch) {
    const auto setup_begin = Clock::now();
    if (target_capacity_ >
        static_cast<std::size_t>(graph_.rows)) {
      throw std::invalid_argument(
          "preds-gpu target capacity exceeds graph row count");
    }
    if (target_capacity_ >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "preds-gpu target capacity exceeds the signed 32-bit device domain");
    }

    HostAuxiliary auxiliary =
        build_host_auxiliary(graph_, mode_);
    preprocessing_ms_ =
        milliseconds(Clock::now() - setup_begin);
    metadata_ = query_device_metadata();
    allocate_and_upload(
        graph_,
        auxiliary,
        mode_,
        target_capacity_,
        reserve_path_scratch_,
        resources_,
        metadata_,
        setup_transfers_);
    policy_ =
        make_launch_policy<DeviceOffset, Packed>(metadata_);

    if (reserve_path_scratch_) {
      const std::size_t vertices =
          static_cast<std::size_t>(graph_.rows);
      reusable_host_result_.distances.resize(vertices);
      reusable_host_result_.predecessor_edges.resize(vertices);
    }
    normalized_request_.unique_targets.reserve(
        target_capacity_);
    if (target_capacity_ != 0) {
      host_target_epochs_.assign(
          static_cast<std::size_t>(graph_.rows), 0);
    }
    cumulative_.graph_setup_transfers = setup_transfers_;
    cumulative_.preprocessing_ms = preprocessing_ms_;
    cumulative_.allocated_device_bytes =
        metadata_.allocated_device_bytes;

    if (print_setup_metadata) {
      print_run_metadata(
          metadata_,
          policy_,
          sizeof(DeviceOffset) == 4,
          Packed);
      std::cout
          << "Successful-update telemetry counts successful strict GPU "
             "atomic decreases and can differ from sequential CPU update "
             "counts.\n";
    }
  }

  RunResult run(const RunRequest& raw_request) override {
    int active_device = -1;
    check_hip(
        hipGetDevice(&active_device),
        "hipGetDevice before reusable preds-gpu run");
    if (active_device != metadata_.device) {
      throw std::runtime_error(
          "preds-gpu context is device-affine; reactivate HIP device " +
          std::to_string(metadata_.device) + " before run()");
    }
    if (raw_request.source < 0 ||
        static_cast<Offset>(raw_request.source) >= graph_.rows) {
      throw std::out_of_range(
          "source node is outside the CSR graph");
    }
    if (raw_request.capture_paths &&
        !reserve_path_scratch_) {
      throw std::invalid_argument(
          "preds-gpu path capture was not reserved at context construction");
    }
    if (target_capacity_ == 0 &&
        !raw_request.unique_targets.empty()) {
      throw std::invalid_argument(
          "preds-gpu unique target count exceeds reserved workload "
          "capacity");
    }

    normalized_request_.source = raw_request.source;
    normalized_request_.early_stop = raw_request.early_stop;
    normalized_request_.capture_paths =
        raw_request.capture_paths;
    normalized_request_.capture_phase_histogram =
        raw_request.capture_phase_histogram;
    normalized_request_.statistics_snapshot_phase =
        raw_request.statistics_snapshot_phase;
    normalized_request_.unique_targets.clear();
    ++target_epoch_;
    if (target_epoch_ == 0) {
      std::fill(
          host_target_epochs_.begin(),
          host_target_epochs_.end(),
          0);
      target_epoch_ = 1;
    }
    for (Index target : raw_request.unique_targets) {
      if (target < 0 ||
          static_cast<Offset>(target) >= graph_.rows) {
        throw std::out_of_range(
            "target node is outside the CSR graph");
      }
      std::uint32_t& seen_epoch =
          host_target_epochs_[static_cast<std::size_t>(target)];
      if (seen_epoch != target_epoch_) {
        if (normalized_request_.unique_targets.size() ==
            target_capacity_) {
          throw std::invalid_argument(
              "preds-gpu unique target count exceeds reserved workload "
              "capacity");
        }
        seen_epoch = target_epoch_;
        normalized_request_.unique_targets.push_back(target);
      }
    }

    GpuRunResult internal = run_gpu_typed(
        graph_,
        mode_,
        normalized_request_,
        resources_,
        policy_,
        reserve_path_scratch_
            ? &reusable_host_result_
            : nullptr);

    checked_accumulate(
        cumulative_.runs, 1, "cumulative preds-gpu runs");
    if (internal.fully_converged) {
      checked_accumulate(
          cumulative_.fully_converged_runs,
          1,
          "cumulative fully converged preds-gpu runs");
    } else if (internal.termination ==
               TerminationReason::kAllTargetsSettled) {
      checked_accumulate(
          cumulative_.target_stopped_runs,
          1,
          "cumulative target-stopped preds-gpu runs");
    }
    add_algorithm_statistics(
        cumulative_.algorithm, internal.statistics);
    add_transfer_statistics(
        cumulative_.run_transfers, internal.transfers);
    add_launch_counters(
        cumulative_.launches, internal.launches);
    cumulative_.runtime_ms += internal.total_runtime_ms;
    return public_run_result(std::move(internal));
  }

  PredicateMode predicate_mode() const noexcept override {
    return mode_;
  }

  Offset rows() const noexcept override {
    return graph_.rows;
  }

  Offset nnz() const noexcept override {
    return graph_.nnz;
  }

  std::size_t target_capacity() const noexcept override {
    return target_capacity_;
  }

  bool path_scratch_reserved() const noexcept override {
    return reserve_path_scratch_;
  }

  void validate_captured_target_path(
      Index source,
      const CapturedTargetPathView& path) const override {
    validate_captured_target_path_against_graph(
        graph_, source, path);
  }

  CumulativeStatistics cumulative_statistics() const override {
    return cumulative_;
  }

 private:
  CsrGraph graph_;
  PredicateMode mode_;
  std::size_t target_capacity_ = 0;
  bool reserve_path_scratch_ = false;
  DeviceMetadata metadata_;
  LaunchPolicy policy_;
  GpuResources<DeviceOffset, Packed> resources_;
  HostResult reusable_host_result_;
  RunRequest normalized_request_;
  std::vector<std::uint32_t> host_target_epochs_;
  std::uint32_t target_epoch_ = 0;
  TransferStatistics setup_transfers_;
  double preprocessing_ms_ = 0.0;
  CumulativeStatistics cumulative_;
};

CsrGraph copy_csr_graph_view(CsrGraphView view) {
  if (view.rows <= 0 || view.rows != view.cols) {
    throw std::invalid_argument(
        "preds-gpu CSR view must be nonempty and square");
  }
  if (view.rows >
      static_cast<Offset>(std::numeric_limits<Index>::max()) ||
      view.nnz < 0) {
    throw std::invalid_argument(
        "preds-gpu CSR view dimensions are outside the supported domain");
  }
  if (view.rowptr == nullptr ||
      (view.nnz != 0 &&
       (view.colind == nullptr || view.values == nullptr))) {
    throw std::invalid_argument(
        "preds-gpu CSR view contains a null array");
  }
  if (static_cast<std::uint64_t>(view.nnz) >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(
        "preds-gpu CSR view edge count is too large for this host");
  }

  CsrGraph graph;
  graph.rows = view.rows;
  graph.cols = view.cols;
  graph.nnz = view.nnz;
  graph.rowptr.assign(
      view.rowptr,
      view.rowptr + static_cast<std::size_t>(view.rows) + 1);
  if (view.nnz != 0) {
    const std::size_t edges =
        static_cast<std::size_t>(view.nnz);
    graph.colind.assign(view.colind, view.colind + edges);
    graph.values.assign(view.values, view.values + edges);
  }
  return graph;
}

int run(const Options& options, Clock::time_point end_to_end_begin) {
  const auto total_begin = Clock::now();
  const double pre_run_end_to_end_ms =
      milliseconds(total_begin - end_to_end_begin);
  const auto graph_load_begin = Clock::now();
  CsrGraph graph = load_csr(options.csr_path);
  const double graph_load_ms =
      milliseconds(Clock::now() - graph_load_begin);
  if (static_cast<Offset>(options.source) >= graph.rows) {
    throw std::out_of_range("source node is outside the CSR graph");
  }
  const Offset graph_rows = graph.rows;
  const Offset graph_nnz = graph.nnz;

  std::vector<Index> output_targets;
  if (options.print_paths) {
    const std::size_t possible_paths =
        static_cast<std::size_t>(graph.rows - 1);
    output_targets.reserve(
        std::min(kMaximumPrintedPaths, possible_paths));
    for (Offset raw_target = 0;
         raw_target < graph.rows &&
         output_targets.size() < kMaximumPrintedPaths;
         ++raw_target) {
      const Index target = static_cast<Index>(raw_target);
      if (target != options.source) {
        output_targets.push_back(target);
      }
    }
  }

  PredsGpuContext context(
      std::move(graph),
      options.predicate,
      output_targets.size(),
      options.print_paths,
      true);
  RunRequest request;
  request.source = options.source;
  request.unique_targets = output_targets;
  request.early_stop = false;
  request.capture_paths = options.print_paths;
  request.capture_phase_histogram = true;
  request.statistics_snapshot_phase =
      options.stats_number_set ? options.stats_number : 0;
  const double setup_runtime_ms =
      milliseconds(Clock::now() - total_begin);
  RunResult result = context.run(request);
  const double total_runtime_ms =
      milliseconds(Clock::now() - total_begin);
  const CumulativeStatistics cumulative =
      context.cumulative_statistics();
  const double preprocessing_ms =
      graph_load_ms + cumulative.preprocessing_ms;

  TransferStatistics total_transfers =
      cumulative.graph_setup_transfers;
  add_transfer_statistics(
      total_transfers, result.telemetry.transfers);

  std::size_t paths_written = 0;
  if (options.stats_number_set &&
      result.requested_snapshot_reached) {
    TransferStatistics snapshot_transfers =
        cumulative.graph_setup_transfers;
    add_transfer_statistics(
        snapshot_transfers,
        result.requested_snapshot.transfers);
    print_statistics(
        result.requested_snapshot.algorithm,
        "phase " + std::to_string(options.stats_number) +
            " snapshot",
        options.predicate,
        snapshot_transfers,
        result.requested_snapshot.launches,
        preprocessing_ms,
        setup_runtime_ms +
            result.requested_snapshot.runtime_ms,
        pre_run_end_to_end_ms + setup_runtime_ms +
            result.requested_snapshot.runtime_ms);
  }
  if (options.print_paths) {
    paths_written = write_paths_jsonl(
        options.paths_output_path,
        graph_rows,
        graph_nnz,
        result.targets,
        options.source,
        options.predicate);
  }
  const double end_to_end_ms =
      milliseconds(Clock::now() - end_to_end_begin);

  if (options.stats_number_set &&
      !result.requested_snapshot_reached) {
    std::cout << "\nRequested statistics phase "
              << options.stats_number
              << " was not reached; the run completed after "
              << result.telemetry.algorithm.phases
              << " phase(s).\n";
  }
  if (options.print_paths) {
    std::cout << "\nWrote " << paths_written
              << " path record(s) to "
              << options.paths_output_path.string() << '\n';
  }
  print_statistics(
      result.telemetry.algorithm,
      "final",
      options.predicate,
      total_transfers,
      result.telemetry.launches,
      preprocessing_ms,
      total_runtime_ms,
      end_to_end_ms);
  return 0;
}

}  // namespace

struct PredsGpuContext::Impl {
  Impl(CsrGraph graph,
       PredicateMode mode,
       std::size_t max_unique_targets,
       bool reserve_path_scratch,
       bool print_setup_metadata) {
    (void)predicate_mode_name(mode);
    validate_csr_structure(graph);
    const bool compact =
        static_cast<unsigned long long>(graph.nnz) <=
        static_cast<unsigned long long>(
            std::numeric_limits<CompactOffset>::max());
    if (compact) {
      engine =
          std::make_unique<
              PredsGpuTypedEngineImpl<CompactOffset, true>>(
              std::move(graph),
              mode,
              max_unique_targets,
              reserve_path_scratch,
              print_setup_metadata);
    } else {
      engine =
          std::make_unique<
              PredsGpuTypedEngineImpl<Offset, false>>(
              std::move(graph),
              mode,
              max_unique_targets,
              reserve_path_scratch,
              print_setup_metadata);
    }
  }

  std::unique_ptr<PredsGpuTypedEngine> engine;
};

PredsGpuContext::PredsGpuContext(
    CsrGraph graph,
    PredicateMode mode,
    std::size_t max_unique_targets,
    bool reserve_path_scratch,
    bool print_setup_metadata)
    : impl_(std::make_unique<Impl>(
          std::move(graph),
          mode,
          max_unique_targets,
          reserve_path_scratch,
          print_setup_metadata)) {}

PredsGpuContext::PredsGpuContext(
    CsrGraphView graph,
    PredicateMode mode,
    std::size_t max_unique_targets,
    bool reserve_path_scratch,
    bool print_setup_metadata)
    : PredsGpuContext(
          copy_csr_graph_view(graph),
          mode,
          max_unique_targets,
          reserve_path_scratch,
          print_setup_metadata) {}

PredsGpuContext::~PredsGpuContext() = default;
PredsGpuContext::PredsGpuContext(PredsGpuContext&&) noexcept = default;
PredsGpuContext& PredsGpuContext::operator=(
    PredsGpuContext&&) noexcept = default;

PredicateMode PredsGpuContext::predicate_mode() const noexcept {
  return impl_ ? impl_->engine->predicate_mode()
               : PredicateMode::kInSimple;
}

Offset PredsGpuContext::rows() const noexcept {
  return impl_ ? impl_->engine->rows() : 0;
}

Offset PredsGpuContext::nnz() const noexcept {
  return impl_ ? impl_->engine->nnz() : 0;
}

std::size_t PredsGpuContext::target_capacity() const noexcept {
  return impl_ ? impl_->engine->target_capacity() : 0;
}

bool PredsGpuContext::path_scratch_reserved() const noexcept {
  return impl_ && impl_->engine->path_scratch_reserved();
}

RunResult PredsGpuContext::run(const RunRequest& request) {
  if (!impl_) {
    throw std::logic_error(
        "cannot run a moved-from preds-gpu context");
  }
  return impl_->engine->run(request);
}

void PredsGpuContext::validate_captured_target_path(
    Index source,
    const CapturedTargetPathView& path) const {
  if (!impl_) {
    throw std::logic_error(
        "cannot validate a path with a moved-from preds-gpu context");
  }
  impl_->engine->validate_captured_target_path(source, path);
}

CumulativeStatistics PredsGpuContext::cumulative_statistics() const {
  return impl_ ? impl_->engine->cumulative_statistics()
               : CumulativeStatistics{};
}

}  // namespace rips_predicates_gpu

// Keep the reusable translation unit entry-point-free by default.  The
// standalone executable opts in explicitly; PREDS_GPU_NO_MAIN remains an
// overriding compatibility guard for embedding build recipes.
#if defined(PREDS_GPU_STANDALONE_MAIN) && !defined(PREDS_GPU_NO_MAIN)
int main(int argc, char** argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" ||
       std::string(argv[1]) == "--help")) {
    rips_predicates_gpu::print_usage(std::cout, argv[0]);
    return 0;
  }
  const auto end_to_end_begin =
      rips_predicates_gpu::Clock::now();
  try {
    const rips_predicates_gpu::Options options =
        rips_predicates_gpu::parse_args(argc, argv);
    return rips_predicates_gpu::run(options, end_to_end_begin);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
#endif
