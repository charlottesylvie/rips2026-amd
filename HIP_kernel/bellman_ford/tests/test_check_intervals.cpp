// Benchmarks bf_var_checks.cpp across user-selected convergence-check intervals.
//
// Compile from the repository root with:
//
//   hipcc -std=c++17 -O3 -x hip \
//     HIP_kernel/bellman_ford/tests/test_check_intervals.cpp \
//     HIP_kernel/bellman_ford/src/bf_var_checks.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o test_check_intervals
//
// Run from the repository root with a .csrbin file from data/, a 1-based
// source node, max_iters, and one or more check intervals:
//
//   ./test_check_intervals USA-road-d.BAY.csrbin 1 50000 1 10 100 1000
//
// Use max_iters = -1 to run up to n - 1 iterations.

#include "../src/bf_var_checks.hpp"
#include "../src/bf_hip_CSR_device_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t EXPECTED_FORMAT_VERSION = 1;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;
const std::filesystem::path kDataDir =
    std::filesystem::path("HIP_kernel") / "bellman_ford" / "data";

static_assert(sizeof(minplus_sparse::Offset) == sizeof(std::int64_t),
              "CSR rowptr format expects 64-bit offsets");
static_assert(sizeof(minplus_sparse::Index) == sizeof(std::int32_t),
              "CSR colind format expects 32-bit indices");
static_assert(sizeof(float) == 4, "CSR values format expects 32-bit floats");

void check_hip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             hipGetErrorString(status));
  }
}

struct HipStreamOwner {
  hipStream_t stream = nullptr;

  HipStreamOwner() = default;

  HipStreamOwner(const HipStreamOwner&) = delete;
  HipStreamOwner& operator=(const HipStreamOwner&) = delete;

  ~HipStreamOwner() {
    if (stream) {
      (void)hipStreamDestroy(stream);
    }
  }

  void create() {
    check_hip(hipStreamCreate(&stream), "create HIP stream");
  }
};

struct RunSummary {
  int check_interval = 0;
  double copy_to_gpu_ms = 0.0;
  double external_runtime_ms = 0.0;
  BellmanFordCsrVarChecksResult result;
};

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <csrbin file-or-name> <source_node_1_based> <max_iters>"
               " <check_interval> [check_interval ...]\n\n";
  std::cerr << "Examples:\n";
  std::cerr << "  " << program
            << " USA-road-d.BAY.csrbin 1 50000 1 10 100 1000\n";
  std::cerr << "  " << program
            << " HIP_kernel/bellman_ford/data/USA-road-d.BAY.csrbin"
               " 1 -1 10 100\n\n";
  std::cerr << "Use max_iters = -1 to run up to n - 1 iterations.\n";

  if (std::filesystem::exists(kDataDir)) {
    std::cerr << "\nAvailable .csrbin files in " << kDataDir << ":\n";
    for (const auto& entry : std::filesystem::directory_iterator(kDataDir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".csrbin") {
        std::cerr << "  " << entry.path().filename().string() << "\n";
      }
    }
  }
}

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
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " is too large for host");
  }

  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) {
    return;
  }

  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

std::filesystem::path resolve_csrbin_path(const std::string& user_text) {
  const std::filesystem::path direct(user_text);
  if (std::filesystem::exists(direct)) {
    return direct;
  }

  std::filesystem::path in_data = kDataDir / direct;
  if (std::filesystem::exists(in_data)) {
    return in_data;
  }

  if (direct.extension() != ".csrbin") {
    in_data = kDataDir / (user_text + ".csrbin");
    if (std::filesystem::exists(in_data)) {
      return in_data;
    }
  }

  throw std::runtime_error("could not find CSR file: " + user_text);
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

  const std::uint64_t version = read_u64(in, "format version");
  const std::uint64_t orientation = read_u64(in, "orientation");
  const std::uint64_t rows = read_u64(in, "row count");
  const std::uint64_t cols = read_u64(in, "column count");
  const std::uint64_t declared_edges = read_u64(in, "declared edge count");
  const std::uint64_t loaded_edges = read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "nnz");
  const std::uint64_t rowptr_count = read_u64(in, "rowptr count");
  const std::uint64_t colind_count = read_u64(in, "colind count");
  const std::uint64_t values_count = read_u64(in, "values count");

  (void)declared_edges;
  (void)loaded_edges;

  if (version != EXPECTED_FORMAT_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }
  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Index>::max())) {
    throw std::runtime_error("CSR graph has too many nodes for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz ||
      values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<minplus_sparse::Offset>(rows);
  graph.cols = static_cast<minplus_sparse::Offset>(cols);
  graph.nnz = static_cast<minplus_sparse::Offset>(nnz);

  read_array(in, graph.rowptr, rowptr_count, "rowptr");
  read_array(in, graph.colind, colind_count, "colind");
  read_array(in, graph.values, values_count, "values");

  if (graph.rowptr.empty() || graph.rowptr.front() != 0 ||
      graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr does not start at 0 and end at nnz");
  }

  return graph;
}

int parse_source_1_based(const char* text, minplus_sparse::Offset rows) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 ||
      static_cast<minplus_sparse::Offset>(value) > rows) {
    throw std::runtime_error("source must be a valid positive 1-based node id");
  }
  return static_cast<int>(value - 1);
}

int parse_int_arg(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(name) + " must be an integer");
  }
  return static_cast<int>(value);
}

int parse_max_iters(const char* text) {
  const int value = parse_int_arg(text, "max_iters");
  if (value < -1) {
    throw std::runtime_error("max_iters must be -1 or a nonnegative integer");
  }
  return value;
}

int parse_check_interval(const char* text) {
  const int value = parse_int_arg(text, "check_interval");
  if (value <= 0) {
    throw std::runtime_error("check_interval must be positive");
  }
  return value;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_run_summary(const RunSummary& run) {
  const BellmanFordCsrVarChecksResult& r = run.result;
  const int unchecked_iterations = r.iterations_used - r.checks_performed;
  const bool check_estimate_available =
      r.checks_performed > 0 && unchecked_iterations > 0;
  const double check_ms =
      check_estimate_available && r.estimated_check_overhead_ms > 0.0
          ? r.estimated_check_overhead_ms
          : 0.0;
  const double bf_algorithm_ms =
      r.bellman_ford_loop_ms > check_ms ? r.bellman_ford_loop_ms - check_ms
                                        : 0.0;

  std::cout << "check_interval: " << run.check_interval << "\n";
  std::cout << "check_time_ms: ";
  if (check_estimate_available) {
    std::cout << check_ms << "\n";
  } else {
    std::cout << "not_estimable_no_unchecked_iterations\n";
  }
  std::cout << "bf_time_ms: " << bf_algorithm_ms << "\n";
  std::cout << "total_time_ms: " << run.external_runtime_ms << "\n";
  std::cout << "iterations_needed: ";
  if (r.converged) {
    std::cout << r.iterations_used << "\n";
  } else {
    std::cout << "did_not_converge_after_" << r.iterations_used << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::filesystem::path csr_path = resolve_csrbin_path(argv[1]);
    HostCsrF32 graph = load_csrbin(csr_path);
    const int source = parse_source_1_based(argv[2], graph.rows);
    const int max_iters = parse_max_iters(argv[3]);
    const int effective_max_iters =
        max_iters < 0 ? static_cast<int>(graph.rows) - 1 : max_iters;

    std::vector<int> check_intervals;
    check_intervals.reserve(static_cast<std::size_t>(argc - 4));
    for (int arg = 4; arg < argc; ++arg) {
      check_intervals.push_back(parse_check_interval(argv[arg]));
    }

    HipStreamOwner stream;
    stream.create();

    std::vector<RunSummary> summaries;
    summaries.reserve(check_intervals.size());

    for (int check_interval : check_intervals) {
      RunSummary summary;
      summary.check_interval = check_interval;

      const auto copy_start = std::chrono::steady_clock::now();
      bf_csr_detail::DeviceCsrOwner d_graph =
          bf_csr_detail::copy_host_csr_to_device(graph, stream.stream);
      const auto copy_end = std::chrono::steady_clock::now();
      summary.copy_to_gpu_ms = elapsed_ms(copy_start, copy_end);

      const auto run_start = std::chrono::steady_clock::now();
      summary.result =
          bellman_ford_minplus_hip_csr_var_checks_with_stats(d_graph.view,
                                                             source,
                                                             max_iters,
                                                             check_interval,
                                                             stream.stream);
      const auto run_end = std::chrono::steady_clock::now();
      summary.external_runtime_ms = elapsed_ms(run_start, run_end);

      d_graph.reset();

      summary.result.dist.clear();
      summary.result.dist.shrink_to_fit();
      summaries.push_back(std::move(summary));
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "csr_file: " << csr_path << "\n";
    std::cout << "nodes: " << graph.rows << "\n";
    std::cout << "nnz: " << graph.nnz << "\n";
    std::cout << "source_node_1_based: " << (source + 1) << "\n";
    std::cout << "max_iters: " << effective_max_iters << "\n";
    std::cout << "intervals_tested: " << summaries.size() << "\n\n";

    for (std::size_t i = 0; i < summaries.size(); ++i) {
      std::cout << "run_" << (i + 1) << "\n";
      print_run_summary(summaries[i]);
      if (i + 1 < summaries.size()) {
        std::cout << "\n";
      }
    }

    std::cout << "\nsummary_comparison\n";
    for (const RunSummary& summary : summaries) {
      std::cout << "check_interval: " << summary.check_interval
                << ", runtime_ms: " << summary.external_runtime_ms << "\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
