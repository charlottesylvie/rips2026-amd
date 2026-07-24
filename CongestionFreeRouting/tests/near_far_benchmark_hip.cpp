#include "../near_far/near_far.hpp"
#include "../unit_bfs/unit_bfs_hip_CSR.hpp"
#include "../../HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.hpp"

// AMD build/run from the repository root:
//   hipcc -std=c++17 -O3 -DNDEBUG -pthread -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I HIP_kernel/delta_stepping/src \
//     -I CongestionFreeRouting/near_far \
//     -I CongestionFreeRouting/unit_bfs \
//     CongestionFreeRouting/tests/near_far_benchmark_hip.cpp \
//     CongestionFreeRouting/near_far/near_far.cpp \
//     CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
//     HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp \
//     -o /tmp/near_far_benchmark_hip
//   /tmp/near_far_benchmark_hip 32768 21

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct Scenario {
  const char* name = nullptr;
  WeightMode weights = WeightMode::kExactUnit;
  bool full_targets = false;
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

HostCsrF32 make_graph(int vertices, WeightMode mode) {
  if (vertices < 64) {
    throw std::invalid_argument("benchmark vertex count must be at least 64");
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
  const HostCsrF32 graph = make_graph(vertices, scenario.weights);
  const std::vector<int> sources{0, vertices / 7};
  const std::vector<int> targets =
      make_targets(vertices, scenario.full_targets);
  NearFarCsrWorkspace near_far(graph, stream);
  DeltaSteppingCsrWorkspace delta(graph, stream);

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

  const Measurement optimized = measure(
      repetitions, stream, true, [&] {
        const NearFarCsrResult result =
            near_far.run(sources, targets, 1.0f, -1, stream);
        if (result.target_distances.size() != targets.size()) {
          throw std::runtime_error("Near-Far optimized result size mismatch");
        }
      });
  print_measurement(scenario.name, "near_far_optimized", optimized);

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
