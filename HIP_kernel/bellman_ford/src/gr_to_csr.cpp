// Converts a DIMACS-style .gr file into the CSR binary format used by
// bf_hip_CSR.cpp and bf_hip_no_checks_CSR.cpp.
//
// Compile from the repository root:
//
//   g++ -std=c++17 -O3 \
//     HIP_kernel/bellman_ford/src/gr_to_csr.cpp \
//     -o gr_to_csr
//
// Run with an explicit output path:
//
//   ./gr_to_csr \
//     HIP_kernel/bellman_ford/data/gr_raw/USA-road-d.BAY.gr \
//     HIP_kernel/bellman_ford/data/USA-road-d.BAY.csrbin
//
// If the output path is omitted, the converter writes to
// HIP_kernel/bellman_ford/data/<graph-name>.csrbin.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Binary CSR format written by this converter:
//
//   char[8]  magic = "RIPSCSR1"
//   uint64   format_version = 1
//   uint64   orientation = 1, meaning incoming-edge orientation
//   uint64   rows
//   uint64   cols
//   uint64   declared_edges
//   uint64   loaded_edges
//   uint64   nnz after duplicate reduction
//   uint64   rowptr_count = rows + 1
//   uint64   colind_count = nnz
//   uint64   values_count = nnz
//   int64[]  rowptr
//   int32[]  colind
//   float[]  values
//
// This matches bf_hip_CSR's HostCsrF32 / minplus_sparse::DeviceCsrF32 layout:
//   rowptr uses int64 offsets
//   colind uses int indices
//   values uses float weights
//
// Matrix convention:
//   CSR row v, column u = weight of edge u -> v

static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t CSR_FORMAT_VERSION = 1;
constexpr std::uint64_t INCOMING_EDGE_ORIENTATION = 1;

struct CsrGraph {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<float> values;
};

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " graph.gr [output.csrbin]\n";
  std::cerr << "Raw .gr files usually live in "
               "HIP_kernel/bellman_ford/data/gr_raw/.\n";
  std::cerr << "Default output: HIP_kernel/bellman_ford/data/"
               "<graph-name>.csrbin\n";
}

std::filesystem::path default_output_path(const std::filesystem::path& input) {
  std::string name = input.filename().string();
  if (name.size() >= 3 && name.substr(name.size() - 3) == ".gr") {
    name.resize(name.size() - 3);
  }
  return std::filesystem::path("HIP_kernel") / "bellman_ford" / "data" /
         (name + ".csrbin");
}

void write_u64(std::ofstream& out, std::uint64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

template <typename T>
void write_array(std::ofstream& out,
                 const std::vector<T>& values,
                 const char* name) {
  if (values.empty()) {
    return;
  }

  const std::size_t bytes = values.size() * sizeof(T);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

std::uint64_t as_u64(std::int64_t value, const char* name) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " is negative");
  }
  return static_cast<std::uint64_t>(value);
}

CsrGraph load_dimacs_gr_as_incoming_csr(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("could not open .gr file: " + path.string());
  }

  CsrGraph graph;
  std::vector<std::vector<std::pair<std::int32_t, float>>> incoming_rows;

  std::uint64_t line_number = 0;
  std::string text;

  while (std::getline(file, text)) {
    ++line_number;

    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
      text.pop_back();
    }
    if (text.empty() || text[0] == 'c') {
      continue;
    }

    if (text[0] == 'p') {
      char problem[32] = {};
      long long n = 0;
      long long m = 0;
      if (std::sscanf(text.c_str(), "p %31s %lld %lld", problem, &n, &m) !=
          3) {
        throw std::runtime_error("invalid problem line at line " +
                                 std::to_string(line_number));
      }
      if (n <= 0 ||
          n > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("invalid node count in problem line");
      }
      if (m < 0) {
        throw std::runtime_error("invalid edge count in problem line");
      }

      graph.rows = static_cast<std::int64_t>(n);
      graph.cols = static_cast<std::int64_t>(n);
      graph.declared_edges = static_cast<std::uint64_t>(m);
      incoming_rows.assign(static_cast<std::size_t>(n), {});

      std::cout << "nodes: " << graph.rows << "\n";
      std::cout << "declared_edges: " << graph.declared_edges << "\n";
      continue;
    }

    if (text[0] == 'a') {
      if (graph.rows == 0) {
        throw std::runtime_error("edge line appeared before problem line");
      }

      long long from = 0;
      long long to = 0;
      double weight = 0.0;
      if (std::sscanf(text.c_str(), "a %lld %lld %lf", &from, &to, &weight) !=
          3) {
        throw std::runtime_error("invalid edge line at line " +
                                 std::to_string(line_number));
      }
      if (from <= 0 || from > graph.rows || to <= 0 || to > graph.rows) {
        throw std::runtime_error("edge endpoint outside graph at line " +
                                 std::to_string(line_number));
      }
      if (!std::isfinite(weight)) {
        throw std::runtime_error("edge weight is not finite at line " +
                                 std::to_string(line_number));
      }

      const std::int32_t u = static_cast<std::int32_t>(from - 1);
      const std::int32_t v = static_cast<std::int32_t>(to - 1);
      const float w = static_cast<float>(weight);

      incoming_rows[static_cast<std::size_t>(v)].push_back({u, w});
      ++graph.loaded_edges;
    }
  }

  if (graph.rows == 0) {
    throw std::runtime_error("missing DIMACS problem line");
  }

  graph.rowptr.resize(static_cast<std::size_t>(graph.rows) + 1);

  for (std::int64_t row = 0; row < graph.rows; ++row) {
    auto& entries = incoming_rows[static_cast<std::size_t>(row)];
    graph.rowptr[static_cast<std::size_t>(row)] =
        static_cast<std::int64_t>(graph.colind.size());

    std::sort(entries.begin(),
              entries.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
              });

    for (std::size_t i = 0; i < entries.size();) {
      const std::int32_t col = entries[i].first;
      float best_weight = entries[i].second;
      ++i;

      while (i < entries.size() && entries[i].first == col) {
        if (entries[i].second < best_weight) {
          best_weight = entries[i].second;
        }
        ++i;
      }

      graph.colind.push_back(col);
      graph.values.push_back(best_weight);
    }
  }

  graph.rowptr[static_cast<std::size_t>(graph.rows)] =
      static_cast<std::int64_t>(graph.colind.size());
  return graph;
}

void write_csr_graph(const CsrGraph& graph,
                     const std::filesystem::path& output_path) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open output file: " +
                             output_path.string());
  }

  out.write(CSR_MAGIC, sizeof(CSR_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing CSR magic");
  }

  const std::uint64_t nnz = static_cast<std::uint64_t>(graph.values.size());
  write_u64(out, CSR_FORMAT_VERSION, "format version");
  write_u64(out, INCOMING_EDGE_ORIENTATION, "orientation");
  write_u64(out, as_u64(graph.rows, "rows"), "row count");
  write_u64(out, as_u64(graph.cols, "cols"), "column count");
  write_u64(out, graph.declared_edges, "declared edge count");
  write_u64(out, graph.loaded_edges, "loaded edge count");
  write_u64(out, nnz, "nnz");
  write_u64(out, static_cast<std::uint64_t>(graph.rowptr.size()),
            "rowptr count");
  write_u64(out, static_cast<std::uint64_t>(graph.colind.size()),
            "colind count");
  write_u64(out, static_cast<std::uint64_t>(graph.values.size()),
            "values count");

  write_array(out, graph.rowptr, "rowptr");
  write_array(out, graph.colind, "colind");
  write_array(out, graph.values, "values");
}

double mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::filesystem::path input_path(argv[1]);
    const std::filesystem::path output_path =
        argc == 3 ? std::filesystem::path(argv[2])
                  : default_output_path(input_path);

    std::filesystem::create_directories("HIP_kernel/bellman_ford/data");

    std::cout << "input: " << input_path << "\n";
    std::cout << "output: " << output_path << "\n";

    CsrGraph graph = load_dimacs_gr_as_incoming_csr(input_path);
    write_csr_graph(graph, output_path);

    const std::uint64_t rowptr_bytes =
        static_cast<std::uint64_t>(graph.rowptr.size() * sizeof(std::int64_t));
    const std::uint64_t colind_bytes =
        static_cast<std::uint64_t>(graph.colind.size() * sizeof(std::int32_t));
    const std::uint64_t values_bytes =
        static_cast<std::uint64_t>(graph.values.size() * sizeof(float));

    std::cout << "loaded_edges: " << graph.loaded_edges << "\n";
    std::cout << "csr_nnz: " << graph.values.size() << "\n";
    std::cout << "duplicate_edges_removed: "
              << (graph.loaded_edges - graph.values.size()) << "\n";
    std::cout << "rowptr_bytes_mib: " << mib(rowptr_bytes) << "\n";
    std::cout << "colind_bytes_mib: " << mib(colind_bytes) << "\n";
    std::cout << "values_bytes_mib: " << mib(values_bytes) << "\n";
    std::cout << "wrote: " << output_path << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
