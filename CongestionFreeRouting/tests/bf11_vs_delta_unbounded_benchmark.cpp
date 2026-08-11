// Unbounded production BF11 versus Delta-Stepping SSSP benchmark (AMD HIP GPU).
//
// Build from the repository root:
//   hipcc -std=c++17 -O3 -pthread -x hip -DBF11_NO_MAIN \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     CongestionFreeRouting/tests/bf11_vs_delta_unbounded_benchmark.cpp \
//     CongestionFreeRouting/bellman_ford/bf11.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     -o /tmp/bf11_vs_delta_unbounded_benchmark
//
// Example (vertices, degree, delta, warmups, repeats, queries, seed):
//   /tmp/bf11_vs_delta_unbounded_benchmark 100000 8 4 2 10 32 123

#include "../bellman_ford/bf11.hpp"
#include "../delta_stepping/delta_stepping_hip_CSR.hpp"

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

using Clock = std::chrono::steady_clock;
using Offset = minplus_sparse::Offset;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class HipStream {
 public:
  HipStream() {
    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "create stream");
  }
  ~HipStream() {
    if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
  }
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;
  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

class HipEvent {
 public:
  HipEvent() { check_hip(hipEventCreate(&event_), "create event"); }
  ~HipEvent() {
    if (event_ != nullptr) (void)hipEventDestroy(event_);
  }
  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;
  hipEvent_t get() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

struct Edge {
  int to = -1;
  float weight = 0.0f;
};

HostCsrF32 make_graph(int vertex_count,
                      int average_out_degree,
                      std::uint32_t seed) {
  if (vertex_count < 2) {
    throw std::invalid_argument("vertex count must be at least 2");
  }
  if (average_out_degree < 1) {
    throw std::invalid_argument("average out degree must be positive");
  }

  std::mt19937 random(seed);
  std::uniform_int_distribution<int> vertex_distribution(0, vertex_count - 1);
  std::uniform_real_distribution<float> weight_distribution(1.0f, 16.0f);
  std::vector<std::vector<Edge>> outgoing(
      static_cast<std::size_t>(vertex_count));

  // A directed ring makes every generated query reachable. The remaining
  // edges retain the requested out degree and give the algorithms useful
  // weighted alternatives.
  for (int from = 0; from < vertex_count; ++from) {
    outgoing[static_cast<std::size_t>(from)].push_back(
        Edge{(from + 1) % vertex_count, weight_distribution(random)});
    for (int edge_index = 1; edge_index < average_out_degree; ++edge_index) {
      int to = vertex_distribution(random);
      if (to == from) to = (to + 1) % vertex_count;
      outgoing[static_cast<std::size_t>(from)].push_back(
          Edge{to, weight_distribution(random)});
    }
  }

  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.rowptr.resize(static_cast<std::size_t>(vertex_count) + 1, 0);
  for (int from = 0; from < vertex_count; ++from) {
    graph.rowptr[static_cast<std::size_t>(from + 1)] =
        graph.rowptr[static_cast<std::size_t>(from)] +
        static_cast<Offset>(outgoing[static_cast<std::size_t>(from)].size());
  }
  graph.nnz = graph.rowptr.back();
  graph.colind.resize(static_cast<std::size_t>(graph.nnz));
  graph.values.resize(static_cast<std::size_t>(graph.nnz));
  for (int from = 0; from < vertex_count; ++from) {
    Offset position = graph.rowptr[static_cast<std::size_t>(from)];
    for (const Edge& edge : outgoing[static_cast<std::size_t>(from)]) {
      graph.colind[static_cast<std::size_t>(position)] = edge.to;
      graph.values[static_cast<std::size_t>(position)] = edge.weight;
      ++position;
    }
  }
  return graph;
}

BellmanFord11NodeSidecars make_unit_sidecars(int vertex_count) {
  BellmanFord11NodeSidecars sidecars;
  sidecars.route_end_x.resize(static_cast<std::size_t>(vertex_count));
  sidecars.route_end_y.assign(static_cast<std::size_t>(vertex_count), 0);
  sidecars.base_vertex_costs.assign(static_cast<std::size_t>(vertex_count),
                                    1.0f);
  for (int node = 0; node < vertex_count; ++node) {
    sidecars.route_end_x[static_cast<std::size_t>(node)] = node;
  }
  return sidecars;
}

struct Query {
  int source = -1;
  int target = -1;
};

std::vector<Query> make_queries(int count,
                                int vertex_count,
                                std::uint32_t seed) {
  if (count < 1) throw std::invalid_argument("query count must be positive");
  std::mt19937 random(seed);
  std::uniform_int_distribution<int> distribution(0, vertex_count - 1);
  std::vector<Query> queries;
  queries.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const int source = distribution(random);
    int target = distribution(random);
    if (target == source) target = (target + vertex_count / 2) % vertex_count;
    queries.push_back(Query{source, target});
  }
  return queries;
}

struct Timing {
  double gpu_ms = 0.0;
  double wall_ms = 0.0;
};

template <typename Function>
auto time_call(hipStream_t stream,
               HipEvent& start,
               HipEvent& stop,
               Function&& function) {
  check_hip(hipEventRecord(start.get(), stream), "record start event");
  const auto wall_start = Clock::now();
  auto result = function();
  check_hip(hipEventRecord(stop.get(), stream), "record stop event");
  check_hip(hipEventSynchronize(stop.get()), "synchronize stop event");
  const auto wall_stop = Clock::now();
  float gpu_ms = 0.0f;
  check_hip(hipEventElapsedTime(&gpu_ms, start.get(), stop.get()),
            "measure event time");
  return std::make_pair(
      std::move(result),
      Timing{static_cast<double>(gpu_ms),
             std::chrono::duration<double, std::milli>(wall_stop - wall_start)
                 .count()});
}

void require_matching_results(const BellmanFordCsrResult& bf11,
                              const DeltaSteppingCsrResult& delta,
                              int query_index) {
  if (bf11.target_distances.size() != 1 ||
      delta.target_distances.size() != 1) {
    throw std::runtime_error("query " + std::to_string(query_index) +
                             " did not return one compact target distance");
  }
  const float left = bf11.target_distances.front();
  const float right = delta.target_distances.front();
  if (bf11.target_reached != std::isfinite(left) ||
      delta.target_reached != std::isfinite(right)) {
    throw std::runtime_error("query " + std::to_string(query_index) +
                             " has an inconsistent reachability field");
  }
  if (std::isinf(left) || std::isinf(right)) {
    if (!(std::isinf(left) && std::isinf(right))) {
      throw std::runtime_error("query " + std::to_string(query_index) +
                               " has inconsistent reachability");
    }
    return;
  }
  const float scale = std::max({1.0f, std::fabs(left), std::fabs(right)});
  if (std::fabs(left - right) > 1e-5f * scale) {
    throw std::runtime_error("query " + std::to_string(query_index) +
                             " has mismatched distances: BF11=" +
                             std::to_string(left) + " Delta=" +
                             std::to_string(right));
  }
}

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
};

Stats summarize(std::vector<double> values) {
  if (values.empty()) return {};
  Stats stats;
  stats.mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  std::sort(values.begin(), values.end());
  stats.minimum = values.front();
  stats.maximum = values.back();
  stats.median = values[values.size() / 2];
  const std::size_t p95_index = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
  stats.p95 = values[std::min(p95_index, values.size() - 1)];
  return stats;
}

void print_stats(const char* engine,
                 const Stats& gpu,
                 const Stats& wall) {
  std::cout << std::left << std::setw(16) << engine << std::right
            << std::setw(11) << gpu.mean << std::setw(11) << gpu.median
            << std::setw(11) << gpu.p95 << std::setw(11) << gpu.minimum
            << std::setw(11) << gpu.maximum << std::setw(12) << wall.mean
            << std::setw(12) << wall.median << std::setw(12) << wall.p95
            << std::setw(12) << wall.minimum << std::setw(12) << wall.maximum
            << '\n';
}

int parse_int(char** argv, int argc, int index, int fallback) {
  return index < argc ? std::stoi(argv[index]) : fallback;
}

float parse_float(char** argv, int argc, int index, float fallback) {
  return index < argc ? std::stof(argv[index]) : fallback;
}

void print_usage(const char* executable) {
  std::cerr
      << "Usage: " << executable
      << " [vertices=100000] [degree=8] [delta=4] [warmups=2]"
         " [repeats=10] [queries=32] [seed=1]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && (std::string(argv[1]) == "-h" ||
                     std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }

    const int vertex_count = parse_int(argv, argc, 1, 100000);
    const int degree = parse_int(argv, argc, 2, 8);
    const float delta_value = parse_float(argv, argc, 3, 4.0f);
    const int warmups = parse_int(argv, argc, 4, 2);
    const int repeats = parse_int(argv, argc, 5, 10);
    const int query_count = parse_int(argv, argc, 6, 32);
    const int seed = parse_int(argv, argc, 7, 1);
    if (!(delta_value > 0.0f) || warmups < 0 || repeats < 1) {
      throw std::invalid_argument(
          "delta and repeats must be positive; warmups must be nonnegative");
    }

    HipStream stream;
    HipEvent start;
    HipEvent stop;
    HostCsrF32 graph =
        make_graph(vertex_count, degree, static_cast<std::uint32_t>(seed));
    BellmanFord11NodeSidecars sidecars = make_unit_sidecars(vertex_count);
    const std::vector<Query> queries = make_queries(
        query_count, vertex_count, static_cast<std::uint32_t>(seed + 1));

    BellmanFord11WorkspaceOptions bf11_options;
    bf11_options.auto_bounds = false;
    bf11_options.unbounded_fallback = false;
    bf11_options.telemetry = false;
    BellmanFord11CsrWorkspace bf11(graph, sidecars, stream.get(), bf11_options);
    DeltaSteppingCsrWorkspace delta(graph, stream.get());

    std::cout << "Unbounded production SSSP benchmark\n"
              << "  vertices: " << graph.rows << "\n"
              << "  edges:    " << graph.nnz << "\n"
              << "  degree:   " << degree << "\n"
              << "  delta:    " << delta_value << "\n"
              << "  queries:  " << query_count << "\n"
              << "  warmups:  " << warmups << " full query sets\n"
              << "  repeats:  " << repeats << " full query sets\n"
              << "  timed calls per engine: " << repeats * query_count << "\n\n";

    for (int warmup = 0; warmup < warmups; ++warmup) {
      for (const Query& query : queries) {
        const std::vector<int> sources{query.source};
        const std::vector<int> targets{query.target};
        (void)bf11.run(sources, targets, delta_value, -1, stream.get(),
                       nullptr, nullptr);
        (void)delta.run(sources, targets, delta_value, -1, stream.get(),
                        nullptr, nullptr);
      }
    }

    std::vector<double> bf11_gpu;
    std::vector<double> bf11_wall;
    std::vector<double> delta_gpu;
    std::vector<double> delta_wall;
    const std::size_t sample_count =
        static_cast<std::size_t>(repeats) * queries.size();
    bf11_gpu.reserve(sample_count);
    bf11_wall.reserve(sample_count);
    delta_gpu.reserve(sample_count);
    delta_wall.reserve(sample_count);

    for (int repetition = 0; repetition < repeats; ++repetition) {
      for (std::size_t query_index = 0; query_index < queries.size();
           ++query_index) {
        const Query query = queries[query_index];
        // Build the common one-source/one-target request outside timing. The
        // scalar overloads have intentionally different result shapes: Delta
        // copies the full distance vector while BF11 uses compact targets.
        const std::vector<int> sources{query.source};
        const std::vector<int> targets{query.target};
        BellmanFordCsrResult bf11_result;
        DeltaSteppingCsrResult delta_result;
        Timing bf11_timing;
        Timing delta_timing;

        // Alternate order to reduce systematic thermal/clock bias.
        if ((repetition + static_cast<int>(query_index)) % 2 == 0) {
          auto measured_bf11 = time_call(stream.get(), start, stop, [&] {
            return bf11.run(sources, targets, delta_value, -1, stream.get(),
                            nullptr, nullptr);
          });
          bf11_result = std::move(measured_bf11.first);
          bf11_timing = measured_bf11.second;
          auto measured_delta = time_call(stream.get(), start, stop, [&] {
            return delta.run(sources, targets, delta_value, -1, stream.get(),
                             nullptr, nullptr);
          });
          delta_result = std::move(measured_delta.first);
          delta_timing = measured_delta.second;
        } else {
          auto measured_delta = time_call(stream.get(), start, stop, [&] {
            return delta.run(sources, targets, delta_value, -1, stream.get(),
                             nullptr, nullptr);
          });
          delta_result = std::move(measured_delta.first);
          delta_timing = measured_delta.second;
          auto measured_bf11 = time_call(stream.get(), start, stop, [&] {
            return bf11.run(sources, targets, delta_value, -1, stream.get(),
                            nullptr, nullptr);
          });
          bf11_result = std::move(measured_bf11.first);
          bf11_timing = measured_bf11.second;
        }

        require_matching_results(bf11_result, delta_result,
                                 static_cast<int>(query_index));
        bf11_gpu.push_back(bf11_timing.gpu_ms);
        bf11_wall.push_back(bf11_timing.wall_ms);
        delta_gpu.push_back(delta_timing.gpu_ms);
        delta_wall.push_back(delta_timing.wall_ms);
      }
    }

    const Stats bf11_gpu_stats = summarize(bf11_gpu);
    const Stats bf11_wall_stats = summarize(bf11_wall);
    const Stats delta_gpu_stats = summarize(delta_gpu);
    const Stats delta_wall_stats = summarize(delta_wall);
    std::cout << std::fixed << std::setprecision(3)
              << "Milliseconds per SSSP query\n"
              << std::left << std::setw(16) << "Engine" << std::right
              << std::setw(11) << "GPU mean" << std::setw(11) << "GPU p50"
              << std::setw(11) << "GPU p95" << std::setw(11) << "GPU min"
              << std::setw(11) << "GPU max" << std::setw(12) << "Wall mean"
              << std::setw(12) << "Wall p50" << std::setw(12) << "Wall p95"
              << std::setw(12) << "Wall min" << std::setw(12) << "Wall max"
              << '\n';
    print_stats("BF11", bf11_gpu_stats, bf11_wall_stats);
    print_stats("Delta-Stepping", delta_gpu_stats, delta_wall_stats);

    std::cout << "\nSpeedup (BF11 / Delta; above 1 means Delta is faster)\n"
              << "  GPU mean:  "
              << bf11_gpu_stats.mean / delta_gpu_stats.mean << "x\n"
              << "  Wall mean: "
              << bf11_wall_stats.mean / delta_wall_stats.mean << "x\n"
              << "Correctness: all paired target distances matched\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 versus Delta benchmark failed: " << error.what()
              << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
