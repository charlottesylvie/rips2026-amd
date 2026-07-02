#include "delta_stepping_hip_CSR.hpp"
#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kAbsTol = 1e-3f;
constexpr float kRelTol = 1e-5f;

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
      throw std::invalid_argument("Dijkstra/Delta-Stepping require finite nonnegative weights");
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
  if (!(std::isfinite(min_weight) && std::isfinite(max_weight) &&
        min_weight >= 0.0f && max_weight >= min_weight)) {
    throw std::invalid_argument("weight range must be finite and nonnegative");
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> vertex_dist(0, n - 1);
  std::uniform_real_distribution<float> weight_dist(min_weight, max_weight);

  // Public CSR orientation, matching Bellman-Ford and Delta-Stepping:
  // row v stores incoming edges u -> v.
  std::vector<std::vector<std::pair<int, float>>> incoming(static_cast<std::size_t>(n));

  // Ensure source 0 has at least one path to every vertex.  If a different
  // source is selected, random edges may still make all vertices reachable,
  // but unreachable vertices are also handled correctly by the verifier.
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

std::vector<float> dijkstra_reference_incoming_csr(const HostCsrF32& g, int source) {
  validate_host_csr(g);
  if (source < 0 || source >= g.rows) {
    throw std::invalid_argument("source out of range");
  }

  // Convert incoming CSR to CPU outgoing adjacency because Dijkstra naturally
  // relaxes outgoing edges.  Input row v, col u means edge u -> v.
  std::vector<std::vector<std::pair<int, float>>> outgoing(static_cast<std::size_t>(g.rows));
  for (int v = 0; v < g.rows; ++v) {
    for (Offset p = g.rowptr[static_cast<std::size_t>(v)];
         p < g.rowptr[static_cast<std::size_t>(v + 1)]; ++p) {
      const int u = g.colind[static_cast<std::size_t>(p)];
      const float w = g.values[static_cast<std::size_t>(p)];
      outgoing[static_cast<std::size_t>(u)].push_back({v, w});
    }
  }

  std::vector<float> dist(static_cast<std::size_t>(g.rows), kInf);
  using Item = std::pair<float, int>;  // distance, vertex
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

  dist[static_cast<std::size_t>(source)] = 0.0f;
  pq.push({0.0f, source});

  while (!pq.empty()) {
    const auto [du, u] = pq.top();
    pq.pop();
    if (du != dist[static_cast<std::size_t>(u)]) continue;

    for (const auto& [v, w] : outgoing[static_cast<std::size_t>(u)]) {
      const float nd = du + w;
      if (nd < dist[static_cast<std::size_t>(v)]) {
        dist[static_cast<std::size_t>(v)] = nd;
        pq.push({nd, v});
      }
    }
  }
  return dist;
}

bool close_enough(float expected, float actual) {
  const bool expected_inf = std::isinf(expected);
  const bool actual_inf = std::isinf(actual);
  if (expected_inf || actual_inf) return expected_inf == actual_inf;
  const float diff = std::fabs(expected - actual);
  const float scale = std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return diff <= kAbsTol || diff <= kRelTol * scale;
}

struct CompareSummary {
  int mismatches = 0;
  float max_abs_diff = 0.0f;
};

CompareSummary compare_to_reference(const std::vector<float>& ref,
                                     const std::vector<float>& got,
                                     const char* label,
                                     int max_print = 5) {
  if (ref.size() != got.size()) {
    throw std::runtime_error(std::string(label) + " distance vector has wrong size");
  }

  CompareSummary s;
  int printed = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const bool ref_inf = std::isinf(ref[i]);
    const bool got_inf = std::isinf(got[i]);
    float diff = 0.0f;
    if (!ref_inf && !got_inf) {
      diff = std::fabs(ref[i] - got[i]);
      s.max_abs_diff = std::max(s.max_abs_diff, diff);
    }

    if (!close_enough(ref[i], got[i])) {
      ++s.mismatches;
      if (printed < max_print) {
        std::cout << "    " << label << " mismatch at vertex " << i
                  << ": expected=" << ref[i]
                  << " got=" << got[i]
                  << " abs_diff=" << diff << '\n';
        ++printed;
      }
    }
  }
  return s;
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
      << "[warmup_ignored=0] [test_cases=10] [seed=1] [source=0] [max_iters=-1]\n\n"
      << "This accepts the same positional inputs as speed_compare_delta_vs_bf.cpp.\n"
      << "The old repeats argument is used here as test_cases; default is 10.\n\n"
      << "Each test builds a different random incoming-edge CSR graph using seed + test_id.\n"
      << "CSR convention: row v, column u means directed edge u -> v.\n\n"
      << "Example:\n"
      << "  " << exe << " 100000 8 20 0 10 123 0 -1\n";
}

}  // namespace

int main(int argc, char** argv) {
  hipStream_t stream = nullptr;
  try {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }

    const int n = arg_int(argv, argc, 1, 2048);
    const int avg_out_degree = arg_int(argv, argc, 2, 8);
    const float delta = arg_float(argv, argc, 3, 4.0f);
    (void)arg_int(argv, argc, 4, 0);  // kept for positional compatibility with warmup
    const int test_cases = arg_int(argv, argc, 5, 10);
    const int seed = arg_int(argv, argc, 6, 1);
    const int source = arg_int(argv, argc, 7, 0);
    const int max_iters = arg_int(argv, argc, 8, -1);

    if (n <= 1) throw std::invalid_argument("n must be > 1");
    if (avg_out_degree < 1) throw std::invalid_argument("avg_out_degree must be >= 1");
    if (!(delta > 0.0f && std::isfinite(delta))) throw std::invalid_argument("delta must be finite and > 0");
    if (test_cases <= 0) throw std::invalid_argument("test_cases must be > 0");
    if (source < 0 || source >= n) throw std::invalid_argument("source out of range");

    check_hip(hipStreamCreate(&stream), "create stream");

    std::cout << "Correctness verification against CPU Dijkstra\n"
              << "  vertices:       " << n << '\n'
              << "  avg out degree: ~" << avg_out_degree << '\n'
              << "  random weights: [1, 16]\n"
              << "  delta:          " << delta << '\n'
              << "  test cases:     " << test_cases << '\n'
              << "  base seed:      " << seed << '\n'
              << "  source:         " << source << '\n'
              << "  max_iters:      " << max_iters << " (-1 means algorithm default)\n\n";

    int failed_cases = 0;
    int total_bf_mismatches = 0;
    int total_ds_mismatches = 0;
    int total_target_mismatches = 0;
    int failed_target_cases = 0;
    float max_bf_abs_diff = 0.0f;
    float max_ds_abs_diff = 0.0f;
    float max_target_abs_diff = 0.0f;

    for (int t = 0; t < test_cases; ++t) {
      const std::uint32_t case_seed = static_cast<std::uint32_t>(seed + t);
      HostCsrF32 h_graph = make_random_incoming_csr(n, avg_out_degree, 1.0f, 16.0f, case_seed);
      DeviceCsrOwner d_graph = copy_host_csr_to_device(h_graph, stream);

      std::vector<float> expected = dijkstra_reference_incoming_csr(h_graph, source);

      BellmanFordCsrResult bf = bellman_ford_minplus_hip_csr(
          d_graph.view, source, max_iters, stream, nullptr, nullptr);
      DeltaSteppingCsrResult ds = delta_stepping_minplus_hip_csr(
          d_graph.view, source, delta, max_iters, stream, nullptr, nullptr);
      const int target = (source + n - 1) % n;
      DeltaSteppingCsrResult ds_target = delta_stepping_minplus_hip_csr(
          d_graph.view, source, target, delta, max_iters, stream, nullptr, nullptr);

      check_hip(hipStreamSynchronize(stream), "sync after algorithms");

      std::cout << "case " << std::setw(2) << t
                << " seed=" << case_seed
                << " edges=" << h_graph.nnz
                << " | BF converged=" << bf.converged
                << " iter=" << bf.iterations_used
                << " | DS converged=" << ds.converged
                << " iter=" << ds.iterations_used
                << " | target=" << target
                << " reached=" << ds_target.target_reached
                << " stopped=" << ds_target.stopped_on_target
                << " iter=" << ds_target.iterations_used << '\n';

      CompareSummary bf_cmp = compare_to_reference(expected, bf.dist, "Bellman-Ford");
      CompareSummary ds_cmp = compare_to_reference(expected, ds.dist, "Delta-Stepping");

      total_bf_mismatches += bf_cmp.mismatches;
      total_ds_mismatches += ds_cmp.mismatches;
      max_bf_abs_diff = std::max(max_bf_abs_diff, bf_cmp.max_abs_diff);
      max_ds_abs_diff = std::max(max_ds_abs_diff, ds_cmp.max_abs_diff);

      CompareSummary target_cmp;
      const float expected_target = expected[static_cast<std::size_t>(target)];
      const float got_target = ds_target.dist[static_cast<std::size_t>(target)];
      if (!close_enough(expected_target, got_target) ||
          (std::isfinite(expected_target) && !ds_target.target_reached)) {
        target_cmp.mismatches = 1;
        if (std::isfinite(expected_target) && std::isfinite(got_target)) {
          target_cmp.max_abs_diff = std::fabs(expected_target - got_target);
        }
        ++failed_target_cases;
      }
      total_target_mismatches += target_cmp.mismatches;
      max_target_abs_diff = std::max(max_target_abs_diff, target_cmp.max_abs_diff);

      const bool ok = bf_cmp.mismatches == 0 &&
                      ds_cmp.mismatches == 0 &&
                      target_cmp.mismatches == 0;
      if (!ok) ++failed_cases;

      std::cout << "    BF mismatches=" << bf_cmp.mismatches
                << " max_abs_diff=" << bf_cmp.max_abs_diff
                << " | DS mismatches=" << ds_cmp.mismatches
                << " max_abs_diff=" << ds_cmp.max_abs_diff
                << " | target mismatches=" << target_cmp.mismatches
                << " max_abs_diff=" << target_cmp.max_abs_diff
                << " | " << (ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\nSummary\n"
              << "  failed cases:          " << failed_cases << " / " << test_cases << '\n'
              << "  total BF mismatches:   " << total_bf_mismatches << '\n'
              << "  total DS mismatches:   " << total_ds_mismatches << '\n'
              << "  target failed cases:   " << failed_target_cases << '\n'
              << "  target mismatches:     " << total_target_mismatches << '\n'
              << "  max BF abs diff:       " << max_bf_abs_diff << '\n'
              << "  max DS abs diff:       " << max_ds_abs_diff << '\n'
              << "  max target abs diff:   " << max_target_abs_diff << '\n';

    check_hip(hipStreamDestroy(stream), "destroy stream");
    stream = nullptr;

    if (failed_cases == 0) {
      std::cout << "\nALL TESTS PASSED\n";
      return 0;
    }
    std::cout << "\nTESTS FAILED\n";
    return 2;
  } catch (const std::exception& e) {
    if (stream) (void)hipStreamDestroy(stream);
    std::cerr << "error: " << e.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
