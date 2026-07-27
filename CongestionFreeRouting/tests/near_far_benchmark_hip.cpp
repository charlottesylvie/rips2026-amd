#include "../near_far/near_far.hpp"
#include "../unit_bfs/unit_bfs_hip_CSR.hpp"
#include "../delta_stepping/delta_stepping_hip_CSR.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O3 -DNDEBUG -pthread -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting/bellman_ford \
//     -I CongestionFreeRouting/delta_stepping \
//     -I CongestionFreeRouting/near_far \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/tests/near_far_benchmark_hip.cpp \
//     CongestionFreeRouting/near_far/near_far.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
//     -o /tmp/near_far_benchmark_hip
//   /tmp/near_far_benchmark_hip 32768 21

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Offset = minplus_sparse::Offset;

enum class WeightMode {
  kExactUnit,
  kMixed,
  kBroad,
};

enum class TopologyMode {
  kRegular,
  kSkewed,
};

struct Scenario {
  const char* name = nullptr;
  WeightMode weights = WeightMode::kExactUnit;
  bool full_targets = false;
  TopologyMode topology = TopologyMode::kRegular;
};

struct Measurement {
  double median_ms = 0.0;
  std::uint64_t device_allocations = 0;
  std::uint64_t pinned_allocations = 0;
  std::uint64_t status_copies = 0;
  std::uint64_t shard_count_copies = 0;
};

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + hipGetErrorString(status));
  }
}

float edge_weight(WeightMode mode, int vertex, int lane) {
  if (mode == WeightMode::kExactUnit) return 1.0f;
  if (mode == WeightMode::kMixed) {
    constexpr float weights[] = {0.0f, 0.25f, 1.0f, 3.0f, 7.0f};
    return weights[(vertex + lane) % 5];
  }
  constexpr float weights[] = {
      0.0f, 0.0001f, 0.125f, 1.0f, 32.0f, 4096.0f};
  return weights[(vertex * 3 + lane) % 6];
}

HostCsrF32 make_graph(int vertices,
                      WeightMode mode,
                      TopologyMode topology) {
  if (vertices < 64) {
    throw std::invalid_argument("benchmark vertex count must be at least 64");
  }
  if (topology == TopologyMode::kSkewed) {
    const int hub_degree = std::min(vertices - 1, 4096);
    HostCsrF32 graph;
    graph.rows = vertices;
    graph.cols = vertices;
    graph.rowptr.resize(static_cast<std::size_t>(vertices) + 1, 0);
    for (int vertex = 0; vertex < vertices; ++vertex) {
      const int degree = vertex == 0 ? hub_degree : 2;
      graph.rowptr[static_cast<std::size_t>(vertex + 1)] =
          graph.rowptr[static_cast<std::size_t>(vertex)] + degree;
    }
    graph.nnz = graph.rowptr.back();
    graph.colind.resize(static_cast<std::size_t>(graph.nnz));
    graph.values.resize(static_cast<std::size_t>(graph.nnz));
    for (int vertex = 0; vertex < vertices; ++vertex) {
      const Offset begin = graph.rowptr[static_cast<std::size_t>(vertex)];
      const int degree = vertex == 0 ? hub_degree : 2;
      for (int lane = 0; lane < degree; ++lane) {
        const std::size_t edge =
            static_cast<std::size_t>(begin + lane);
        graph.colind[edge] =
            vertex == 0
                ? lane + 1
                : (lane == 0 ? (vertex + 1) % vertices : 0);
        graph.values[edge] = edge_weight(mode, vertex, lane);
      }
    }
    return graph;
  }

  constexpr int degree = 4;
  HostCsrF32 graph;
  graph.rows = vertices;
  graph.cols = vertices;
  graph.nnz = static_cast<Offset>(vertices) * degree;
  graph.rowptr.resize(static_cast<std::size_t>(vertices) + 1);
  graph.colind.resize(static_cast<std::size_t>(graph.nnz));
  graph.values.resize(static_cast<std::size_t>(graph.nnz));
  constexpr int jumps[degree] = {1, 17, 257, 4093};
  for (int vertex = 0; vertex <= vertices; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex)] =
        static_cast<Offset>(vertex) * degree;
  }
  for (int vertex = 0; vertex < vertices; ++vertex) {
    for (int lane = 0; lane < degree; ++lane) {
      const std::size_t edge =
          static_cast<std::size_t>(vertex) * degree + lane;
      graph.colind[edge] = (vertex + jumps[lane]) % vertices;
      graph.values[edge] = edge_weight(mode, vertex, lane);
    }
  }
  return graph;
}

std::vector<int> make_targets(int vertices, bool full_targets) {
  if (full_targets) {
    std::vector<int> result(static_cast<std::size_t>(vertices));
    for (int vertex = 0; vertex < vertices; ++vertex) {
      result[static_cast<std::size_t>(vertex)] = vertex;
    }
    return result;
  }
  std::vector<int> result;
  for (int index = 0; index < 16; ++index) {
    result.push_back(
        static_cast<int>((static_cast<std::uint64_t>(index + 1) *
                          static_cast<std::uint64_t>(vertices - 1)) /
                         17U));
  }
  result.push_back(result[3]);
  result.push_back(0);
  return result;
}

std::vector<float> cpu_dijkstra(const HostCsrF32& graph,
                                const std::vector<int>& sources) {
  const float infinity = std::numeric_limits<float>::infinity();
  std::vector<float> distances(
      static_cast<std::size_t>(graph.rows), infinity);
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem,
                      std::vector<QueueItem>,
                      std::greater<QueueItem>>
      queue;
  for (const int source : sources) {
    float& distance = distances[static_cast<std::size_t>(source)];
    if (distance != 0.0f) {
      distance = 0.0f;
      queue.push({0.0f, source});
    }
  }
  while (!queue.empty()) {
    const auto [distance, vertex] = queue.top();
    queue.pop();
    if (distance != distances[static_cast<std::size_t>(vertex)]) continue;
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(vertex)];
         edge < graph.rowptr[static_cast<std::size_t>(vertex + 1)];
         ++edge) {
      const std::size_t edge_index = static_cast<std::size_t>(edge);
      const int destination = graph.colind[edge_index];
      const float candidate = distance + graph.values[edge_index];
      float& current = distances[static_cast<std::size_t>(destination)];
      if (candidate < current) {
        current = candidate;
        queue.push({candidate, destination});
      }
    }
  }
  return distances;
}

bool close_enough(float expected, float actual) {
  if (std::isinf(expected) || std::isinf(actual)) {
    return std::isinf(expected) && std::isinf(actual) &&
           std::signbit(expected) == std::signbit(actual);
  }
  if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <= 2e-3f + 2e-5f * scale;
}

void validate_target_distances(const char* engine,
                               const std::vector<float>& expected,
                               const std::vector<int>& targets,
                               const BellmanFordCsrResult& result) {
  if (result.target_distances.size() != targets.size()) {
    throw std::runtime_error(
        std::string(engine) + " result size mismatch");
  }
  bool all_reached = true;
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const float reference =
        expected[static_cast<std::size_t>(targets[index])];
    all_reached = all_reached && std::isfinite(reference);
    if (!close_enough(reference, result.target_distances[index])) {
      throw std::runtime_error(
          std::string(engine) + " target distance mismatch at index " +
          std::to_string(index));
    }
  }
  if (result.target_reached != all_reached) {
    throw std::runtime_error(
        std::string(engine) + " target_reached mismatch");
  }
}

template <typename Function>
Measurement measure(int repetitions,
                    hipStream_t stream,
                    bool near_far_counters,
                    Function&& function) {
  for (int warmup = 0; warmup < 3; ++warmup) {
    function();
    check_hip(hipStreamSynchronize(stream), "synchronize benchmark warmup");
  }
  if (near_far_counters) {
    near_far_internal_reset_optimization_counters();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    const auto start = Clock::now();
    function();
    check_hip(hipStreamSynchronize(stream), "synchronize benchmark sample");
    const auto stop = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(samples.begin(), samples.end());
  Measurement result;
  result.median_ms = samples[samples.size() / 2];
  if (near_far_counters) {
    result.device_allocations =
        near_far_internal_device_allocation_count();
    result.pinned_allocations =
        near_far_internal_pinned_allocation_count();
    result.status_copies = near_far_internal_status_copy_count();
    result.shard_count_copies =
        near_far_internal_shard_count_copy_count();
  }
  return result;
}

void print_measurement(const char* workload,
                       const char* engine,
                       const Measurement& measurement) {
  std::cout << workload << ',' << engine << ',' << std::fixed
            << std::setprecision(3) << measurement.median_ms << ','
            << measurement.device_allocations << ','
            << measurement.pinned_allocations << ','
            << measurement.status_copies << ','
            << measurement.shard_count_copies << '\n';
}

void run_scenario(const Scenario& scenario,
                  int vertices,
                  int repetitions,
                  hipStream_t stream) {
  const HostCsrF32 graph =
      make_graph(vertices, scenario.weights, scenario.topology);
  const std::vector<int> sources{0, vertices / 7};
  const std::vector<int> targets =
      make_targets(vertices, scenario.full_targets);
  const std::vector<float> expected = cpu_dijkstra(graph, sources);
  NearFarCsrWorkspace near_far(graph, stream);
  DeltaSteppingCsrWorkspace delta(graph, stream);

  // Validate each implementation outside the timed region. In particular,
  // this catches incoming/outgoing CSR mismatches in comparison engines
  // before their timings are reported.
  near_far_internal_force_generic(1);
  validate_target_distances(
      "Near-Far forced generic",
      expected,
      targets,
      near_far.run(sources, targets, 1.0f, -1, stream));
  near_far_internal_force_generic(0);
  validate_target_distances(
      "Near-Far automatic",
      expected,
      targets,
      near_far.run(sources, targets, 1.0f, -1, stream));
  validate_target_distances(
      "Delta-Stepping",
      expected,
      targets,
      delta.run(sources, targets, 1.0f, -1, stream));

  near_far_internal_force_generic(1);
  const Measurement generic = measure(
      repetitions, stream, true, [&] {
        const NearFarCsrResult result =
            near_far.run(sources, targets, 1.0f, -1, stream);
        if (result.target_distances.size() != targets.size()) {
          throw std::runtime_error("Near-Far generic result size mismatch");
        }
      });
  near_far_internal_force_generic(0);
  print_measurement(scenario.name, "near_far_forced_generic", generic);

  const Measurement automatic = measure(
      repetitions, stream, true, [&] {
        const NearFarCsrResult result =
            near_far.run(sources, targets, 1.0f, -1, stream);
        if (result.target_distances.size() != targets.size()) {
          throw std::runtime_error("Near-Far automatic result size mismatch");
        }
      });
  print_measurement(scenario.name, "near_far_auto", automatic);

  const Measurement delta_measurement = measure(
      repetitions, stream, false, [&] {
        const DeltaSteppingCsrResult result =
            delta.run(sources, targets, 1.0f, -1, stream);
        if (result.target_distances.size() != targets.size()) {
          throw std::runtime_error("Delta-Stepping result size mismatch");
        }
      });
  print_measurement(
      scenario.name, "delta_stepping", delta_measurement);

  if (scenario.weights == WeightMode::kExactUnit) {
    UnitBfsCsrWorkspace unit_bfs(graph, stream);
    validate_target_distances(
        "UnitBFS",
        expected,
        targets,
        unit_bfs.run(sources, targets, 1.0f, -1, stream));
    const Measurement unit_measurement = measure(
        repetitions, stream, false, [&] {
          const UnitBfsCsrResult result =
              unit_bfs.run(sources, targets, 1.0f, -1, stream);
          if (result.target_distances.size() != targets.size()) {
            throw std::runtime_error("UnitBFS result size mismatch");
          }
        });
    print_measurement(scenario.name, "unit_bfs", unit_measurement);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "query HIP device count");
    if (device_count == 0) {
      std::cout << "Near-Far benchmark skipped: no HIP device available\n";
      return 0;
    }
    const int vertices = argc > 1 ? std::atoi(argv[1]) : 32768;
    const int repetitions = argc > 2 ? std::atoi(argv[2]) : 21;
    if (vertices < 64 || repetitions < 3 || repetitions % 2 == 0) {
      throw std::invalid_argument(
          "usage: near_far_benchmark_hip [vertices>=64] "
          "[odd_repetitions>=3]");
    }

    hipStream_t stream = nullptr;
    check_hip(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        "create benchmark stream");
    const std::vector<Scenario> scenarios{
        {"exact_unit_short", WeightMode::kExactUnit, false},
        {"exact_unit_full", WeightMode::kExactUnit, true},
        {"mixed_weight_short", WeightMode::kMixed, false},
        {"broad_weight_short", WeightMode::kBroad, false},
        {"broad_weight_full", WeightMode::kBroad, true},
        {"mixed_weight_skewed",
         WeightMode::kMixed,
         false,
         TopologyMode::kSkewed},
        {"broad_weight_skewed",
         WeightMode::kBroad,
         false,
         TopologyMode::kSkewed},
    };
    std::cout << "workload,engine,median_ms,device_allocations,"
                 "pinned_allocations,status_copies,shard_count_copies\n";
    for (const Scenario& scenario : scenarios) {
      run_scenario(scenario, vertices, repetitions, stream);
    }
    check_hip(hipStreamDestroy(stream), "destroy benchmark stream");
    return 0;
  } catch (const std::exception& error) {
    near_far_internal_force_generic(0);
    std::cerr << "Near-Far benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
