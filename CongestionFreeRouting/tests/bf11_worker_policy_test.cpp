#include "../bellman_ford/bf11_worker_policy.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_overflow(Function&& function, const std::string& message) {
  bool rejected = false;
  try {
    (void)function();
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, message);
}

bf11_worker_policy::Inputs ample_inputs(std::string_view architecture,
                                        std::size_t workspace_bytes) {
  return {/*route_request_count=*/100,
          /*cpu_hardware_concurrency=*/32,
          /*free_device_bytes=*/workspace_bytes * 8,
          /*workspace_device_bytes_estimate=*/workspace_bytes,
          architecture,
          /*compute_unit_count=*/40};
}

}  // namespace

int main() {
  try {
    using namespace bf11_worker_policy;

    static_assert(kPersistentBytesPerVertex == 29);
    static_assert(kFixedDeviceStatusBytes == 60);
    static_assert(kDeviceBytesPerSourceCapacity == 4);
    static_assert(kDeviceBytesPerTargetCapacity == 52);
    static_assert(kTelemetryDeviceBytes == 48);
    static_assert(kDeviceBytesPerCompactNode == 4);
    static_assert(kDeviceBytesPerCompactEdge == 8);
    static_assert(kMaxAutomaticWorkers == 4);
    static_assert(kGfx1151PreferredWorkers == 3);
    static_assert(persistent_device_workspace_bytes(0, 0, 0) == 60);
    static_assert(retained_query_capacity(100, 0) == 0);
    static_assert(retained_query_capacity(100, 1) == 1);
    static_assert(retained_query_capacity(100, 3) == 4);
    static_assert(retained_query_capacity(100, 4) == 4);
    static_assert(retained_query_capacity(5, 5) == 5);
    static_assert(retained_query_capacity(5, 100) == 5);
    static_assert(persistent_device_workspace_bytes(10, 2, 3) == 566);
    static_assert(persistent_device_workspace_bytes(10, 2, 3, true) ==
                  614);
    static_assert(compact_path_device_bytes_ceiling(10, 3) == 384);
    static_assert(automatic_worker_device_bytes_estimate(10, 2, 3) ==
                  1550);
    static_assert(automatic_worker_device_bytes_estimate(10, 2, 3, true) ==
                  1598);
    static_assert(is_measured_gfx1151("gfx1151"));
    static_assert(is_measured_gfx1151("gfx1151:sramecc+:xnack-"));
    static_assert(!is_measured_gfx1151("gfx11510"));
    static_assert(!is_measured_gfx1151("gfx1150"));

    require_overflow(
        [] {
          return persistent_device_workspace_bytes(
              std::numeric_limits<std::size_t>::max(), 0, 0);
        },
        "vertex-byte overflow was not rejected");
    require(persistent_device_workspace_bytes(
                1, std::numeric_limits<std::size_t>::max(),
                std::numeric_limits<std::size_t>::max()) == 145,
            "raw endpoint hints were not capped by the deduplicated V limit");

    const std::size_t workspace_bytes =
        automatic_worker_device_bytes_estimate(1'000, 8, 16);
    const Recommendation measured =
        recommend(ample_inputs("gfx1151", workspace_bytes));
    require(measured.worker_count == 3 &&
                measured.performance_preference == 3 &&
                measured.resource_limit == 4 &&
                measured.uses_measured_gfx1151_policy,
            "gfx1151 did not select the smallest measured plateau count");

    const Recommendation qualified_arch = recommend(
        ample_inputs("gfx1151:sramecc+:xnack-", workspace_bytes));
    require(qualified_arch.worker_count == 3 &&
                qualified_arch.uses_measured_gfx1151_policy,
            "qualified gfx1151 architecture name missed measured policy");

    Inputs route_limited = ample_inputs("gfx1151", workspace_bytes);
    route_limited.route_request_count = 2;
    require(recommend(route_limited).worker_count == 2,
            "route-request count did not bound workers");

    Inputs cpu_limited = ample_inputs("gfx1151", workspace_bytes);
    cpu_limited.cpu_hardware_concurrency = 2;
    require(recommend(cpu_limited).worker_count == 2,
            "CPU hardware concurrency did not bound workers");

    Inputs cu_limited = ample_inputs("gfx1151", workspace_bytes);
    cu_limited.compute_unit_count = 2;
    require(recommend(cu_limited).worker_count == 2,
            "compute-unit count did not bound workers");

    Inputs memory_limited = ample_inputs("gfx1151", workspace_bytes);
    memory_limited.free_device_bytes = workspace_bytes * 3;
    const Recommendation memory_result = recommend(memory_limited);
    require(memory_result.memory_budget_bytes ==
                memory_limited.free_device_bytes -
                    memory_limited.free_device_bytes / 4 &&
                memory_result.memory_limit == 2 &&
                memory_result.worker_count == 2,
            "available device memory did not bound workers");

    Inputs no_reported_resources = ample_inputs("gfx1151", workspace_bytes);
    no_reported_resources.route_request_count = 0;
    no_reported_resources.cpu_hardware_concurrency = 0;
    no_reported_resources.free_device_bytes = 0;
    no_reported_resources.compute_unit_count = 0;
    const Recommendation minimum = recommend(no_reported_resources);
    require(minimum.worker_count == 1 && minimum.resource_limit == 1 &&
                minimum.memory_limit == 0,
            "missing resource information did not fail closed to one worker");

    const Recommendation unmeasured =
        recommend(ample_inputs("gfx1201", workspace_bytes));
    require(unmeasured.worker_count == 1 &&
                unmeasured.performance_preference == 1 &&
                !unmeasured.uses_measured_gfx1151_policy,
            "an unmeasured architecture assumed concurrency scaling");

    Inputs zero_workspace = ample_inputs("gfx1151", workspace_bytes);
    zero_workspace.workspace_device_bytes_estimate = 0;
    require(recommend(zero_workspace).worker_count == 1,
            "an invalid zero-byte workspace estimate did not fail closed");

    std::cout << "BF11 worker policy test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 worker policy test failed: " << error.what() << '\n';
    return 1;
  }
}
