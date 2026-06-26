// Benchmark runner for the CSR Bellman-Ford implementations.
//
// Compile from the repository root with:
//
//   hipcc -std=c++17 -O3 -x hip \
//     HIP_kernel/bellman_ford/tests/big_test.cpp \
//     HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
//     HIP_kernel/bellman_ford/src/bf_hip_no_checks_CSR.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o big_test
//
// Run from the repository root with a .csrbin file from data/ and a 1-based
// source node:
//
//   ./big_test USA-road-d.BAY.csrbin 1 50
//
// The final argument is optional max_iters. If omitted, both implementations
// use n - 1 iterations.

#include "../src/bf_hip_CSR.hpp"
#include "../src/bf_hip_no_checks_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

struct DeviceCsrOwner {
  minplus_sparse::DeviceCsrF32 view{};
  minplus_sparse::Offset* rowptr = nullptr;
  minplus_sparse::Index* colind = nullptr;
  float* values = nullptr;

  DeviceCsrOwner() = default;

  DeviceCsrOwner(const DeviceCsrOwner&) = delete;
  DeviceCsrOwner& operator=(const DeviceCsrOwner&) = delete;

  DeviceCsrOwner(DeviceCsrOwner&& other) noexcept {
    *this = std::move(other);
  }

  DeviceCsrOwner& operator=(DeviceCsrOwner&& other) noexcept {
    if (this != &other) {
      reset();
      view = other.view;
      rowptr = other.rowptr;
      colind = other.colind;
      values = other.values;
      other.view = {};
      other.rowptr = nullptr;
      other.colind = nullptr;
      other.values = nullptr;
    }
    return *this;
  }

  ~DeviceCsrOwner() {
    reset();
  }

  void reset() {
    if (rowptr) {
      (void)hipFree(rowptr);
      rowptr = nullptr;
    }
    if (colind) {
      (void)hipFree(colind);
      colind = nullptr;
    }
    if (values) {
      (void)hipFree(values);
      values = nullptr;
    }
    view = {};
  }
};

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

  hipStream_t get() const {
    return stream;
  }
};

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <csrbin file-or-name> <source_node_1_based> [max_iters]\n\n";
  std::cerr << "Examples:\n";
  std::cerr << "  " << program
            << " USA-road-d.BAY.csrbin 1 50\n";
  std::cerr << "  " << program
            << " HIP_kernel/bellman_ford/data/USA-road-d.BAY.csrbin 1\n\n";
  std::cerr << "If max_iters is omitted, both implementations use n - 1.\n";

  if (std::filesystem::exists(kDataDir)) {
    std::cerr << "\nAvailable .csrbin files in " << kDataDir << ":\n";
    for (const auto& entry : std::filesystem::directory_iterator(kDataDir)) {
      if (entry.is_regular_file() &&
          entry.path().extension() == ".csrbin") {
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
  if (version != EXPECTED_FORMAT_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }

  const std::uint64_t orientation = read_u64(in, "orientation");
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }

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

int parse_max_iters(const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value > std::numeric_limits<int>::max() || value < -1) {
    throw std::runtime_error("max_iters must be -1 or a nonnegative integer");
  }
  return static_cast<int>(value);
}

DeviceCsrOwner copy_csr_to_device(const HostCsrF32& host,
                                  hipStream_t stream) {
  DeviceCsrOwner device;

  const std::size_t rowptr_bytes =
      host.rowptr.size() * sizeof(minplus_sparse::Offset);
  check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr), rowptr_bytes),
            "hipMalloc CSR rowptr");
  check_hip(hipMemcpyAsync(device.rowptr,
                           host.rowptr.data(),
                           rowptr_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy CSR rowptr to device");

  if (host.nnz > 0) {
    const std::size_t colind_bytes =
        host.colind.size() * sizeof(minplus_sparse::Index);
    const std::size_t values_bytes = host.values.size() * sizeof(float);

    check_hip(hipMalloc(reinterpret_cast<void**>(&device.colind),
                        colind_bytes),
              "hipMalloc CSR colind");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.values),
                        values_bytes),
              "hipMalloc CSR values");
    check_hip(hipMemcpyAsync(device.colind,
                             host.colind.data(),
                             colind_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy CSR colind to device");
    check_hip(hipMemcpyAsync(device.values,
                             host.values.data(),
                             values_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy CSR values to device");
  }

  check_hip(hipStreamSynchronize(stream), "synchronize CSR copy");

  device.view.rows = host.rows;
  device.view.cols = host.cols;
  device.view.nnz = host.nnz;
  device.view.rowptr = device.rowptr;
  device.view.colind = device.colind;
  device.view.values = device.values;
  return device;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int count_reachable(const std::vector<float>& dist) {
  int count = 0;
  for (float value : dist) {
    if (!std::isinf(value)) {
      ++count;
    }
  }
  return count;
}

bool close_enough(float a, float b) {
  if (std::isinf(a) || std::isinf(b)) {
    return std::isinf(a) && std::isinf(b);
  }

  const float diff = std::fabs(a - b);
  const float scale = 1.0f + std::fabs(a) + std::fabs(b);
  return diff <= 1e-4f * scale;
}

int count_distance_mismatches(const std::vector<float>& lhs,
                              const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return -1;
  }

  int mismatches = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!close_enough(lhs[i], rhs[i])) {
      ++mismatches;
    }
  }
  return mismatches;
}

std::string format_distance(float value) {
  if (std::isinf(value)) {
    return "INF";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

void print_distance_sample(const std::vector<float>& dist) {
  constexpr std::size_t max_printed = 12;

  std::cout << "distance_sample_1_based: [";
  const std::size_t printed = std::min(max_printed, dist.size());
  for (std::size_t i = 0; i < printed; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << (i + 1) << ":" << format_distance(dist[i]);
  }
  if (printed < dist.size()) {
    std::cout << ", ...";
  }
  std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::filesystem::path csr_path = resolve_csrbin_path(argv[1]);
    HostCsrF32 graph = load_csrbin(csr_path);
    const int source = parse_source_1_based(argv[2], graph.rows);
    const int max_iters = argc == 4 ? parse_max_iters(argv[3]) : -1;

    HipStreamOwner stream_owner;
    stream_owner.create();
    hipStream_t stream = stream_owner.get();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "csr_file: " << csr_path << "\n";
    std::cout << "nodes: " << graph.rows << "\n";
    std::cout << "nnz: " << graph.nnz << "\n";
    std::cout << "source_node_1_based: " << (source + 1) << "\n";
    std::cout << "max_iters: "
              << (max_iters < 0 ? graph.rows - 1 : max_iters) << "\n\n";

    const auto checked_copy_start = std::chrono::steady_clock::now();
    DeviceCsrOwner checked_graph = copy_csr_to_device(graph, stream);
    const auto checked_copy_end = std::chrono::steady_clock::now();

    const auto checked_start = std::chrono::steady_clock::now();
    const BellmanFordCsrResult checked_result =
        bellman_ford_minplus_hip_csr(checked_graph.view,
                                     source,
                                     max_iters,
                                     stream);
    const auto checked_end = std::chrono::steady_clock::now();

    checked_graph.reset();

    const auto no_checks_copy_start = std::chrono::steady_clock::now();
    DeviceCsrOwner no_checks_graph = copy_csr_to_device(graph, stream);
    const auto no_checks_copy_end = std::chrono::steady_clock::now();

    const auto no_checks_start = std::chrono::steady_clock::now();
    const BellmanFordCsrNoChecksResult no_checks_result =
        bellman_ford_minplus_hip_csr_no_checks(no_checks_graph.view,
                                               source,
                                               max_iters,
                                               stream);
    const auto no_checks_end = std::chrono::steady_clock::now();

    no_checks_graph.reset();

    const double checked_copy_ms =
        elapsed_ms(checked_copy_start, checked_copy_end);
    const double checked_ms = elapsed_ms(checked_start, checked_end);
    const double no_checks_copy_ms =
        elapsed_ms(no_checks_copy_start, no_checks_copy_end);
    const double no_checks_ms = elapsed_ms(no_checks_start, no_checks_end);
    const int mismatches =
        count_distance_mismatches(checked_result.dist, no_checks_result.dist);

    std::cout << "bf_hip_CSR\n";
    std::cout << "copy_to_gpu_ms: " << checked_copy_ms << "\n";
    std::cout << "runtime_ms: " << checked_ms << "\n";
    std::cout << "iterations_used: " << checked_result.iterations_used << "\n";
    std::cout << "converged: " << (checked_result.converged ? "true" : "false")
              << "\n";
    std::cout << "reachable_nodes: " << count_reachable(checked_result.dist)
              << "\n";
    print_distance_sample(checked_result.dist);

    std::cout << "\nbf_hip_no_checks_CSR\n";
    std::cout << "copy_to_gpu_ms: " << no_checks_copy_ms << "\n";
    std::cout << "runtime_ms: " << no_checks_ms << "\n";
    std::cout << "iterations_used: " << no_checks_result.iterations_used
              << "\n";
    std::cout << "reachable_nodes: " << count_reachable(no_checks_result.dist)
              << "\n";
    print_distance_sample(no_checks_result.dist);

    std::cout << "\ncomparison\n";
    std::cout << "no_checks_minus_checked_ms: "
              << (no_checks_ms - checked_ms) << "\n";
    if (checked_ms > 0.0) {
      std::cout << "no_checks_over_checked: "
                << (no_checks_ms / checked_ms) << "\n";
    }
    std::cout << "distance_mismatches: " << mismatches << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
