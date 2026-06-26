// Converts a DIMACS-style .gr file into a dense flattened adjacency binary.
//
// Compile from the repository root:
//
//   g++ -std=c++17 -O3 \
//     HIP_kernel/bellman_ford/src/gr_to_adjacency.cpp \
//     -o gr_to_adjacency
//
// Run with an explicit output path:
//
//   ./gr_to_adjacency \
//     HIP_kernel/bellman_ford/data/gr_raw/USA-road-d.BAY.gr \
//     HIP_kernel/bellman_ford/data/USA-road-d.BAY.adjbin
//
// If the output path is omitted, the converter writes to
// HIP_kernel/bellman_ford/data/<graph-name>.adjbin.

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr char MATRIX_MAGIC[8] = {'R', 'I', 'P', 'S', 'A', 'D', 'J', '1'};
constexpr std::uint64_t MATRIX_FORMAT_VERSION = 1;
constexpr std::uint64_t INCOMING_EDGE_ORIENTATION = 1;

struct GraphInput {
  int n = 0;
  std::uint64_t declared_edges = 0;
  std::uint64_t loaded_edges = 0;
  std::vector<float> adjacency;
};

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " graph.gr [output.adjbin]\n";
  std::cerr << "Raw .gr files usually live in "
               "HIP_kernel/bellman_ford/data/gr_raw/.\n";
  std::cerr << "Default output: HIP_kernel/bellman_ford/data/"
               "<graph-name>.adjbin\n";
}

std::filesystem::path default_output_path(const std::filesystem::path& input) {
  std::string name = input.filename().string();
  if (name.size() >= 3 && name.substr(name.size() - 3) == ".gr") {
    name.resize(name.size() - 3);
  }
  return std::filesystem::path("HIP_kernel") / "bellman_ford" / "data" /
         (name + ".adjbin");
}

void ensure_dense_matrix_fits(int n) {
  const std::size_t n_size = static_cast<std::size_t>(n);
  if (n_size > std::numeric_limits<std::size_t>::max() / n_size) {
    throw std::runtime_error("graph is too large for a dense n*n matrix");
  }

  const std::size_t entries = n_size * n_size;
  if (entries > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::runtime_error("graph is too large for host memory");
  }
}

double dense_matrix_gib(int n) {
  const double entries = static_cast<double>(n) * static_cast<double>(n);
  return entries * sizeof(float) / (1024.0 * 1024.0 * 1024.0);
}

void write_u64(std::ofstream& out, std::uint64_t value, const char* name) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
  }
}

GraphInput load_dimacs_gr(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("could not open .gr file: " + path.string());
  }

  GraphInput graph;
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
      if (std::sscanf(text.c_str(), "p %31s %lld %lld", problem, &n, &m) != 3) {
        throw std::runtime_error("invalid problem line at line " +
                                 std::to_string(line_number));
      }
      if (n <= 0 || n > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid node count in problem line");
      }
      if (m < 0) {
        throw std::runtime_error("invalid edge count in problem line");
      }

      graph.n = static_cast<int>(n);
      graph.declared_edges = static_cast<std::uint64_t>(m);
      ensure_dense_matrix_fits(graph.n);

      std::cout << "nodes: " << graph.n << "\n";
      std::cout << "declared_edges: " << graph.declared_edges << "\n";
      std::cout << "dense_matrix_size_gib: " << dense_matrix_gib(graph.n)
                << "\n";

      graph.adjacency.assign(static_cast<std::size_t>(graph.n) * graph.n,
                             INF);
      continue;
    }

    if (text[0] == 'a') {
      if (graph.n == 0) {
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
      if (from <= 0 || from > graph.n || to <= 0 || to > graph.n) {
        throw std::runtime_error("edge endpoint outside graph at line " +
                                 std::to_string(line_number));
      }
      if (!std::isfinite(weight)) {
        throw std::runtime_error("edge weight is not finite at line " +
                                 std::to_string(line_number));
      }

      const int u = static_cast<int>(from - 1);
      const int v = static_cast<int>(to - 1);

      // bellman_ford_hip expects incoming-edge orientation:
      // adjacency[v * n + u] = weight of edge u -> v.
      const std::size_t index =
          static_cast<std::size_t>(v) * graph.n + static_cast<std::size_t>(u);
      const float w = static_cast<float>(weight);
      if (w < graph.adjacency[index]) {
        graph.adjacency[index] = w;
      }
      ++graph.loaded_edges;
    }
  }

  if (graph.n == 0) {
    throw std::runtime_error("missing DIMACS problem line");
  }

  return graph;
}

void write_adjacency_matrix(const GraphInput& graph,
                            const std::filesystem::path& output_path) {
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open output file: " +
                             output_path.string());
  }

  out.write(MATRIX_MAGIC, sizeof(MATRIX_MAGIC));
  if (!out) {
    throw std::runtime_error("failed while writing matrix magic");
  }

  write_u64(out, MATRIX_FORMAT_VERSION, "format version");
  write_u64(out, INCOMING_EDGE_ORIENTATION, "orientation");
  write_u64(out, static_cast<std::uint64_t>(graph.n), "node count");
  write_u64(out, graph.declared_edges, "declared edge count");
  write_u64(out, graph.loaded_edges, "loaded edge count");

  const std::uint64_t entries =
      static_cast<std::uint64_t>(graph.adjacency.size());
  write_u64(out, entries, "matrix entry count");

  out.write(reinterpret_cast<const char*>(graph.adjacency.data()),
            static_cast<std::streamsize>(graph.adjacency.size() *
                                         sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed while writing adjacency matrix values");
  }
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

    GraphInput graph = load_dimacs_gr(input_path);
    write_adjacency_matrix(graph, output_path);

    std::cout << "loaded_edges: " << graph.loaded_edges << "\n";
    std::cout << "matrix_entries: " << graph.adjacency.size() << "\n";
    std::cout << "wrote: " << output_path << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
