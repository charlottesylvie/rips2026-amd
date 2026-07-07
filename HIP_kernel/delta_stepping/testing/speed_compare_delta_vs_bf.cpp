#include "delta_stepping_hip_CSR.hpp"
#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

struct DeviceCsrOwner {
  minplus_sparse::DeviceCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* colind = nullptr;
  float* values = nullptr;

  DeviceCsrOwner() = default;
  DeviceCsrOwner(const DeviceCsrOwner&) = delete;
  DeviceCsrOwner& operator=(const DeviceCsrOwner&) = delete;

  DeviceCsrOwner(DeviceCsrOwner&& other) noexcept { *this = std::move(other); }
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

  ~DeviceCsrOwner() { reset(); }

  void reset() {
    if (rowptr) (void)hipFree(rowptr);
    if (colind) (void)hipFree(colind);
    if (values) (void)hipFree(values);
    rowptr = nullptr;
    colind = nullptr;
    values = nullptr;
    view = {};
  }
};

void validate_host_csr(const HostCsrF32& g) {
  if (g.rows <= 0 || g.cols <= 0 || g.rows != g.cols) {
    throw std::invalid_argument("graph must be nonempty and square");
  }
  if (g.nnz < 0) {
    throw std::invalid_argument("graph nnz must be nonnegative");
  }
  if (g.rowptr.size() != static_cast<std::size_t>(g.rows + 1)) {
    throw std::invalid_argument("rowptr size must equal rows + 1");
  }
  if (g.colind.size() != static_cast<std::size_t>(g.nnz) ||
      g.values.size() != static_cast<std::size_t>(g.nnz)) {
    throw std::invalid_argument("colind and values size must equal nnz");
  }
  if (g.rowptr.front() != 0 || g.rowptr.back() != g.nnz) {
    throw std::invalid_argument("rowptr must start at 0 and end at nnz");
  }
  for (Offset r = 0; r < g.rows; ++r) {
    const Offset a = g.rowptr[static_cast<std::size_t>(r)];
    const Offset b = g.rowptr[static_cast<std::size_t>(r + 1)];
    if (a < 0 || b < a || b > g.nnz) {
      throw std::invalid_argument("invalid rowptr range");
    }
  }
  for (std::size_t i = 0; i < g.colind.size(); ++i) {
    if (g.colind[i] < 0 || static_cast<Offset>(g.colind[i]) >= g.cols) {
      throw std::invalid_argument("colind contains an out-of-range source vertex");
    }
    if (!(std::isfinite(g.values[i]) && g.values[i] >= 0.0f)) {
      throw std::invalid_argument("Delta-Stepping benchmark needs finite nonnegative weights");
    }
  }
}

DeviceCsrOwner copy_host_csr_to_device(const HostCsrF32& h, hipStream_t stream) {
  validate_host_csr(h);
  DeviceCsrOwner d;

  const std::size_t rowptr_bytes = h.rowptr.size() * sizeof(Offset);
  check_hip(hipMalloc(reinterpret_cast<void**>(&d.rowptr), rowptr_bytes),
            "hipMalloc rowptr");
  check_hip(hipMemcpyAsync(d.rowptr, h.rowptr.data(), rowptr_bytes,
                           hipMemcpyHostToDevice, stream),
            "copy rowptr");

  if (h.nnz > 0) {
    const std::size_t colind_bytes = h.colind.size() * sizeof(Index);
    const std::size_t values_bytes = h.values.size() * sizeof(float);
    check_hip(hipMalloc(reinterpret_cast<void**>(&d.colind), colind_bytes),
              "hipMalloc colind");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d.values), values_bytes),
              "hipMalloc values");
    check_hip(hipMemcpyAsync(d.colind, h.colind.data(), colind_bytes,
                             hipMemcpyHostToDevice, stream),
              "copy colind");
    check_hip(hipMemcpyAsync(d.values, h.values.data(), values_bytes,
                             hipMemcpyHostToDevice, stream),
              "copy values");
  }

  check_hip(hipStreamSynchronize(stream), "sync graph copy");
  d.view.rows = h.rows;
  d.view.cols = h.cols;
  d.view.nnz = h.nnz;
  d.view.rowptr = d.rowptr;
  d.view.colind = d.colind;
  d.view.values = d.values;
  return d;
}

HostCsrF32 make_random_incoming_csr(int n,
                                    int avg_out_degree,
                                    float min_weight,
                                    float max_weight,
                                    std::uint32_t seed) {
  if (n <= 1) throw std::invalid_argument("n must be > 1");
  if (avg_out_degree < 1) throw std::invalid_argument("avg_out_degree must be >= 1");
  if (!(min_weight >= 0.0f && max_weight >= min_weight)) {
    throw std::invalid_argument("weight range must be finite and nonnegative");
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> vertex_dist(0, n - 1);
  std::uniform_real_distribution<float> weight_dist(min_weight, max_weight);

  // The public CSR orientation is incoming-edge CSR:
  // row v stores edges u -> v.  Build incoming[dst] directly.
  std::vector<std::vector<std::pair<int, float>>> incoming(static_cast<std::size_t>(n));

  // Ensure the graph has at least one simple path from source 0 through all vertices.
  for (int u = 0; u + 1 < n; ++u) {
    incoming[static_cast<std::size_t>(u + 1)].push_back({u, 1.0f});
  }

  for (int u = 0; u < n; ++u) {
    for (int e = 0; e < avg_out_degree; ++e) {
      int v = vertex_dist(rng);
      if (v == u) v = (v + 1) % n;
      incoming[static_cast<std::size_t>(v)].push_back({u, weight_dist(rng)});
    }
  }

  HostCsrF32 g;
  g.rows = n;
  g.cols = n;
  g.rowptr.resize(static_cast<std::size_t>(n + 1), 0);
  for (int v = 0; v < n; ++v) {
    g.rowptr[static_cast<std::size_t>(v + 1)] =
        g.rowptr[static_cast<std::size_t>(v)] +
        static_cast<Offset>(incoming[static_cast<std::size_t>(v)].size());
  }
  g.nnz = g.rowptr.back();
  g.colind.resize(static_cast<std::size_t>(g.nnz));
  g.values.resize(static_cast<std::size_t>(g.nnz));

  for (int v = 0; v < n; ++v) {
    Offset pos = g.rowptr[static_cast<std::size_t>(v)];
    for (const auto& edge : incoming[static_cast<std::size_t>(v)]) {
      g.colind[static_cast<std::size_t>(pos)] = edge.first;
      g.values[static_cast<std::size_t>(pos)] = edge.second;
      ++pos;
    }
  }
  return g;
}

struct RunTiming {
  float gpu_ms = 0.0f;   // HIP-event time: queued GPU work on the stream.
  double wall_ms = 0.0;  // End-to-end host wall time for the call.
};

struct TimedResult {
  BellmanFordCsrResult result;
  RunTiming timing;
};

template <class Fn>
TimedResult time_call(hipStream_t stream, Fn&& fn) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check_hip(hipEventCreate(&start), "create start event");
  check_hip(hipEventCreate(&stop), "create stop event");

  check_hip(hipEventRecord(start, stream), "record start event");
  const auto t0 = Clock::now();
  BellmanFordCsrResult r = fn();
  const auto t1 = Clock::now();
  check_hip(hipEventRecord(stop, stream), "record stop event");
  check_hip(hipEventSynchronize(stop), "sync stop event");

  float gpu_ms = 0.0f;
  check_hip(hipEventElapsedTime(&gpu_ms, start, stop), "elapsed event time");
  check_hip(hipEventDestroy(start), "destroy start event");
  check_hip(hipEventDestroy(stop), "destroy stop event");

  TimedResult out;
  out.result = std::move(r);
  out.timing.gpu_ms = gpu_ms;
  out.timing.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return out;
}

struct Stats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
};

Stats stats_of(const std::vector<double>& xs) {
  if (xs.empty()) return {};
  Stats s;
  s.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
  s.min = *std::min_element(xs.begin(), xs.end());
  s.max = *std::max_element(xs.begin(), xs.end());
  return s;
}

struct CompareSummary {
  int mismatches = 0;
  float max_abs_diff = 0.0f;
};

CompareSummary compare_distances(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  float tolerance) {
  if (a.size() != b.size()) {
    throw std::runtime_error("distance vectors have different sizes");
  }
  CompareSummary summary;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool a_inf = std::isinf(a[i]);
    const bool b_inf = std::isinf(b[i]);
    if (a_inf || b_inf) {
      if (a_inf != b_inf) ++summary.mismatches;
      continue;
    }
    const float diff = std::fabs(a[i] - b[i]);
    summary.max_abs_diff = std::max(summary.max_abs_diff, diff);
    if (diff > tolerance) ++summary.mismatches;
  }
  return summary;
}

int arg_int(char** argv, int argc, int index, int fallback) {
  return index < argc ? std::atoi(argv[index]) : fallback;
}

float arg_float(char** argv, int argc, int index, float fallback) {
  return index < argc ? std::strtof(argv[index], nullptr) : fallback;
}

void print_usage(const char* exe) {
  std::cerr
      << "Usage: " << exe << " [n=2048] [avg_out_degree=8] [delta=4] "
      << "[warmup=1] [repeats=5] [seed=1] [source=0] [max_iters=-1]\n\n"
      << "Graph is generated as incoming-edge CSR, matching Bellman-Ford:\n"
      << "  row v, column u means edge u -> v.\n\n"
      << "Example:\n"
      << "  " << exe << " 10000 8 4 2 10 123 0 -1\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }

    const int n = arg_int(argv, argc, 1, 2048);
    const int avg_out_degree = arg_int(argv, argc, 2, 8);
    const float delta = arg_float(argv, argc, 3, 4.0f);
    const int warmup = arg_int(argv, argc, 4, 1);
    const int repeats = arg_int(argv, argc, 5, 5);
    const int seed = arg_int(argv, argc, 6, 1);
    const int source = arg_int(argv, argc, 7, 0);
    const int max_iters = arg_int(argv, argc, 8, -1);
    const int effective_max_iters = max_iters < 0 ? n - 1 : max_iters;

    if (source < 0 || source >= n) throw std::invalid_argument("source out of range");
    if (!(delta > 0.0f)) throw std::invalid_argument("delta must be > 0");
    if (repeats <= 0) throw std::invalid_argument("repeats must be > 0");
    if (warmup < 0) throw std::invalid_argument("warmup must be >= 0");

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "create stream");

    HostCsrF32 h_graph = make_random_incoming_csr(n, avg_out_degree, 1.0f, 16.0f,
                                                  static_cast<std::uint32_t>(seed));
    DeviceCsrOwner d_graph = copy_host_csr_to_device(h_graph, stream);

    std::cout << "Benchmark graph\n"
              << "  vertices:       " << h_graph.rows << '\n'
              << "  edges:          " << h_graph.nnz << '\n'
              << "  avg out degree: ~" << avg_out_degree << '\n'
              << "  source:         " << source << '\n'
              << "  delta:          " << delta << '\n'
              << "  warmup:         " << warmup << '\n'
              << "  repeats:        " << repeats << '\n'
              << "  max_iters:      " << effective_max_iters
              << " (" << max_iters << " requested; -1 means algorithm default)\n\n";

    BellmanFordCsrResult last_bf;
    BellmanFordCsrResult last_ds;

    for (int i = 0; i < warmup; ++i) {
      last_bf = bellman_ford_minplus_hip_csr(d_graph.view, source, max_iters, stream);
      last_ds = delta_stepping_minplus_hip_csr(d_graph.view, source, delta, max_iters, stream);
    }

    std::vector<double> bf_gpu_ms;
    std::vector<double> bf_wall_ms;
    std::vector<double> ds_gpu_ms;
    std::vector<double> ds_wall_ms;
    bf_gpu_ms.reserve(static_cast<std::size_t>(repeats));
    bf_wall_ms.reserve(static_cast<std::size_t>(repeats));
    ds_gpu_ms.reserve(static_cast<std::size_t>(repeats));
    ds_wall_ms.reserve(static_cast<std::size_t>(repeats));

    for (int i = 0; i < repeats; ++i) {
      TimedResult bf = time_call(stream, [&] {
        return bellman_ford_minplus_hip_csr(d_graph.view, source, max_iters, stream);
      });
      TimedResult ds = time_call(stream, [&] {
        return delta_stepping_minplus_hip_csr(d_graph.view, source, delta, max_iters, stream);
      });

      last_bf = std::move(bf.result);
      last_ds = std::move(ds.result);
      bf_gpu_ms.push_back(bf.timing.gpu_ms);
      bf_wall_ms.push_back(bf.timing.wall_ms);
      ds_gpu_ms.push_back(ds.timing.gpu_ms);
      ds_wall_ms.push_back(ds.timing.wall_ms);

      std::cout << "run " << std::setw(2) << i
                << " | BF gpu/wall " << std::setw(10) << bf_gpu_ms.back()
                << " / " << std::setw(10) << bf_wall_ms.back() << " ms"
                << " | DS gpu/wall " << std::setw(10) << ds_gpu_ms.back()
                << " / " << std::setw(10) << ds_wall_ms.back() << " ms\n";
    }

    const Stats bf_gpu = stats_of(bf_gpu_ms);
    const Stats bf_wall = stats_of(bf_wall_ms);
    const Stats ds_gpu = stats_of(ds_gpu_ms);
    const Stats ds_wall = stats_of(ds_wall_ms);
    const CompareSummary cmp = compare_distances(last_bf.dist, last_ds.dist, 1e-4f);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nSummary\n";
    std::cout << "  Bellman-Ford converged=" << last_bf.converged
              << " iterations=" << last_bf.iterations_used << '\n';
    std::cout << "  Delta-Stepping converged=" << last_ds.converged
              << " iterations=" << last_ds.iterations_used << '\n';
    std::cout << "  Distance mismatches > 1e-4: " << cmp.mismatches
              << " max_abs_diff=" << cmp.max_abs_diff << "\n\n";

    std::cout << "Timing table, milliseconds\n";
    std::cout << "  Algorithm        GPU mean   GPU min    GPU max    Wall mean  Wall min   Wall max\n";
    std::cout << "  Bellman-Ford  "
              << std::setw(10) << bf_gpu.mean << std::setw(10) << bf_gpu.min
              << std::setw(10) << bf_gpu.max << std::setw(11) << bf_wall.mean
              << std::setw(10) << bf_wall.min << std::setw(10) << bf_wall.max << '\n';
    std::cout << "  Delta-Step    "
              << std::setw(10) << ds_gpu.mean << std::setw(10) << ds_gpu.min
              << std::setw(10) << ds_gpu.max << std::setw(11) << ds_wall.mean
              << std::setw(10) << ds_wall.min << std::setw(10) << ds_wall.max << '\n';

    const double gpu_speedup = ds_gpu.mean > 0.0 ? bf_gpu.mean / ds_gpu.mean : 0.0;
    const double wall_speedup = ds_wall.mean > 0.0 ? bf_wall.mean / ds_wall.mean : 0.0;
    std::cout << "\nSpeedup, BF time / Delta-Stepping time\n"
              << "  GPU-event speedup: " << gpu_speedup << "x\n"
              << "  Wall-time speedup: " << wall_speedup << "x\n";

    check_hip(hipStreamDestroy(stream), "destroy stream");
    return cmp.mismatches == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
