#include "../bellman_ford/bf12_worker_policy.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_contains(const std::string& value,
                      const std::string& expected,
                      const std::string& message) {
  require(value.find(expected) != std::string::npos,
          message + ": \"" + value + "\"");
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

BellmanFord12WorkerPolicyInputs ample_inputs(
    std::string_view architecture = "gfx1151") {
  BellmanFord12WorkerPolicyInputs inputs;
  inputs.vertex_count = 16'384;
  inputs.edge_count = 65'536;
  inputs.queries_waiting = 10;
  inputs.source_count = 40;
  inputs.target_count = 80;
  inputs.available_device_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  inputs.result_node_capacity = 32'768;
  inputs.result_edge_capacity = 32'768;
  inputs.gpu_architecture = architecture;
  inputs.compute_unit_count = 40;
  inputs.cooperative_launch_supported = true;
  inputs.memory_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
  return inputs;
}

}  // namespace

int main() {
  try {
    using namespace bf12_worker_policy;

    static_assert(kQueryStateBytesPerVertex == 25);
    static_assert(kAutomaticGfx1151BatchSize == 4);
    static_assert(kAutomaticDefaultBatchSize == 3);
    static_assert(kResultNodeBytes == 4);
    static_assert(kResultEdgeBytes == 8);
    static_assert(rounded_retained_capacity(0) == 0);
    static_assert(rounded_retained_capacity(1) == 1);
    static_assert(rounded_retained_capacity(3) == 4);
    static_assert(rounded_retained_capacity(4) == 4);
    static_assert(raw_query_vertex_state_bytes(1, 28'000'000) ==
                  700'000'000);
    static_assert(raw_query_vertex_state_bytes(4, 28'000'000) ==
                  2'800'000'000ULL);
    static_assert(shared_graph_device_bytes(256, 256) == 6'400);
    static_assert(shared_dynamic_cost_device_bytes(256) == 1'024);
    static_assert(is_gfx1151("gfx1151"));
    static_assert(is_gfx1151("gfx1151:sramecc+:xnack-"));
    static_assert(!is_gfx1151("gfx11510"));
    static_assert(!is_gfx1151("gfx1150"));
    static_assert(maximum_uint32_composite_batch_size(1) ==
                  std::numeric_limits<std::uint32_t>::max() - 1ULL);
    static_assert(maximum_uint32_composite_batch_size(
                      (std::numeric_limits<std::uint32_t>::max() - 1ULL) /
                      4) == 4);
    static_assert(maximum_uint32_composite_batch_size(
                      (std::numeric_limits<std::uint32_t>::max() - 1ULL) /
                              4 +
                          1) == 3);

    require_overflow(
        [] {
          return raw_query_vertex_state_bytes(
              2, std::numeric_limits<std::size_t>::max());
        },
        "query-state multiplication overflow was not rejected");
    require_overflow(
        [] {
          return result_device_bytes_estimate(
              1, 1, std::numeric_limits<std::size_t>::max(), 0);
        },
        "result-arena multiplication overflow was not rejected");
    require_overflow(
        [] {
          return aligned_allocation_bytes(
              std::numeric_limits<std::size_t>::max());
        },
        "allocation-alignment addition overflow was not rejected");

    const Inputs accounting = ample_inputs();
    require(shared_dynamic_cost_device_bytes(accounting.vertex_count) ==
                aligned_allocation_bytes(accounting.vertex_count * 4),
            "shared dynamic costs were not charged exactly once");
    const std::size_t one_state =
        query_vertex_state_device_bytes(1, accounting.vertex_count);
    const std::size_t three_states =
        query_vertex_state_device_bytes(3, accounting.vertex_count);
    const std::size_t four_states =
        query_vertex_state_device_bytes(4, accounting.vertex_count);
    require(one_state == accounting.vertex_count * 25 &&
                three_states == one_state * 3 &&
                four_states == one_state * 4,
            "segmented query state did not scale at 25 bytes/query/vertex");
    require(fixed_device_bytes_estimate(accounting, 4) -
                    fixed_device_bytes_estimate(accounting, 1) ==
                (four_states - one_state),
            "shared graph or dynamic costs were duplicated per query");

    Inputs exact_small;
    exact_small.vertex_count = 256;
    exact_small.edge_count = 0;
    exact_small.queries_waiting = 3;
    exact_small.source_count = 1;
    exact_small.target_count = 1;
    exact_small.graph_already_resident = true;
    exact_small.memory_reserve_bytes = 0;
    // Independent manual accounting for B=1:
    //   1024 shared costs + 6400 query state + 256 source states +
    //   256 target states + 512 query records + 768 global records.
    require(fixed_device_bytes_estimate(exact_small, 1) == 9'216,
            "small fixed-byte estimate differs from manual accounting");
    require(result_device_bytes_estimate(1, 1, 0, 0) == 1'536,
            "small result-byte estimate differs from manual accounting");
    require(result_device_bytes_estimate(3, 1, 0, 0) == 1'536,
            "result header still budgets returned query descriptors");
    // Graph-sized query state grows exactly to the submitted batch size.
    require(fixed_device_bytes_estimate(exact_small, 3) == 22'016,
            "batch three did not budget its exact query-state allocation");
    exact_small.requested_batch_size = 3;
    exact_small.available_device_bytes =
        total_device_bytes_estimate(exact_small, 3) - 1;
    const Decision rounded_capacity_limited = decide(exact_small);
    require(rounded_capacity_limited.selected_batch_size == 2,
            "policy admitted batch three without memory for its exact state");
    require_contains(rounded_capacity_limited.limiting_reason,
                     "available GPU memory",
                     "retained-capacity memory limit was not explained");
    Inputs auto_three_boundary = exact_small;
    auto_three_boundary.requested_batch_size = 0;
    const Decision auto_two = decide(auto_three_boundary);
    require(auto_two.selected_batch_size == 2 &&
                auto_two.automatic_preference == 3,
            "automatic batch three ignored its exact-state memory boundary");
    require_contains(auto_two.limiting_reason, "available GPU memory",
                     "automatic retained-capacity limit was not explained");

    Inputs telemetry_accounting = accounting;
    telemetry_accounting.telemetry_enabled = true;
    const std::size_t telemetry_bytes =
        aligned_allocation_bytes(kTelemetryCounterBytes) +
        aligned_allocation_bytes(kMaximumTelemetryRounds *
                                 sizeof(std::uint32_t)) +
        aligned_allocation_bytes(kMaximumTelemetryRounds *
                                 sizeof(std::uint64_t));
    require(fixed_device_bytes_estimate(telemetry_accounting, 4) -
                    fixed_device_bytes_estimate(accounting, 4) ==
                telemetry_bytes,
            "optional device telemetry storage was not budgeted");

    Inputs resident_graph = accounting;
    resident_graph.graph_already_resident = true;
    require(fixed_device_bytes_estimate(accounting, 4) -
                    fixed_device_bytes_estimate(resident_graph, 4) ==
                shared_graph_device_bytes(accounting.vertex_count,
                                          accounting.edge_count),
            "resident immutable graph was not separated from workspace bytes");

    const Decision gfx1151 = decide(ample_inputs("gfx1151"));
    require(gfx1151.selected_batch_size == 4 &&
                gfx1151.automatic_preference == 4 &&
                gfx1151.maximum_memory_batch_size == 10 &&
                gfx1151.used_gfx1151_default,
            "ample gfx1151 auto policy did not select conservative batch 4");
    require_contains(gfx1151.limiting_reason, "default (4)",
                     "gfx1151 decision did not explain its default");
    require(gfx1151.cooperative_allowed,
            "supported cooperative execution was incorrectly disabled");

    const Decision qualified =
        decide(ample_inputs("gfx1151:sramecc+:xnack-"));
    require(qualified.selected_batch_size == 4 &&
                qualified.used_gfx1151_default,
            "qualified gfx1151 architecture missed the target policy");

    const Decision unmeasured = decide(ample_inputs("gfx1201"));
    require(unmeasured.selected_batch_size == 3 &&
                unmeasured.retained_batch_capacity == 3 &&
                unmeasured.automatic_preference == 3 &&
                !unmeasured.used_gfx1151_default,
            "unmeasured architecture did not use conservative batch 3");
    require(unmeasured.selected_batch_size != 10,
            "automatic policy selected every waiting query");

    Inputs explicit_eight = ample_inputs();
    explicit_eight.requested_batch_size = 8;
    const Decision eight = decide(explicit_eight);
    require(eight.selected_batch_size == 8,
            "safe explicit batch size 8 was not honored");
    require_contains(eight.limiting_reason, "explicit requested",
                     "safe explicit request was not explained");

    Inputs baseline = ample_inputs();
    baseline.requested_batch_size = 1;
    const Decision one = decide(baseline);
    require(one.selected_batch_size == 1 &&
                one.maximum_memory_batch_size == 10,
            "explicit correctness baseline did not select one query");
    require_contains(one.limiting_reason, "correctness baseline",
                     "baseline decision was not clearly identified");

    Inputs queue_limited = ample_inputs();
    queue_limited.requested_batch_size = 8;
    queue_limited.queries_waiting = 3;
    const Decision partial = decide(queue_limited);
    require(partial.selected_batch_size == 3,
            "final partial batch was not bounded by waiting queries");
    require_contains(partial.limiting_reason, "waiting queries",
                     "partial batch did not explain its queue limit");

    Inputs cu_limited = ample_inputs();
    cu_limited.compute_unit_count = 2;
    const Decision two_cus = decide(cu_limited);
    require(two_cus.selected_batch_size == 2,
            "automatic policy ignored a small compute-unit count");
    require_contains(two_cus.limiting_reason, "compute-unit",
                     "compute-unit limit was not explained");

    Inputs explicit_ignores_cu = ample_inputs();
    explicit_ignores_cu.requested_batch_size = 4;
    explicit_ignores_cu.compute_unit_count = 2;
    require(decide(explicit_ignores_cu).selected_batch_size == 4,
            "a performance hint incorrectly invalidated an explicit safe batch");

    Inputs memory_limited = ample_inputs();
    memory_limited.requested_batch_size = 8;
    memory_limited.graph_already_resident = true;
    const std::size_t two_query_requirement =
        total_device_bytes_estimate(memory_limited, 2);
    memory_limited.available_device_bytes = checked_add(
        memory_limited.memory_reserve_bytes, two_query_requirement);
    const Decision memory_two = decide(memory_limited);
    require(memory_two.selected_batch_size == 2 &&
                memory_two.maximum_memory_batch_size == 2,
            "memory budget did not reduce an explicit batch to two");
    require_contains(memory_two.limiting_reason, "available GPU memory",
                     "memory-limited decision was not explained");
    require(checked_add(memory_two.estimated_fixed_bytes,
                        memory_two.estimated_result_bytes) <=
                memory_two.memory_budget_bytes,
            "selected batch exceeds its reported post-reserve budget");

    Inputs one_byte_short = memory_limited;
    const std::size_t baseline_requirement =
        total_device_bytes_estimate(one_byte_short, 1);
    one_byte_short.available_device_bytes = checked_add(
        one_byte_short.memory_reserve_bytes, baseline_requirement - 1);
    const Decision rejected = decide(one_byte_short);
    require(rejected.selected_batch_size == 0 &&
                rejected.estimated_fixed_bytes ==
                    fixed_device_bytes_estimate(one_byte_short, 1) &&
                rejected.estimated_result_bytes ==
                    result_device_bytes_estimate(
                        1, one_byte_short.target_count,
                        one_byte_short.result_node_capacity,
                        one_byte_short.result_edge_capacity),
            "unsafe baseline did not fail closed with diagnostic estimates");
    require_contains(rejected.limiting_reason, "batch size 1 exceeds",
                     "baseline memory rejection was not clear");

    Inputs reserve_consumes_all = ample_inputs();
    reserve_consumes_all.available_device_bytes =
        reserve_consumes_all.memory_reserve_bytes;
    const Decision no_budget = decide(reserve_consumes_all);
    require(no_budget.selected_batch_size == 0,
            "reserve-consuming memory report did not fail closed");
    require_contains(no_budget.limiting_reason, "reserve leaves no",
                     "exhausted reserve was not explained");

    Inputs no_memory_report = ample_inputs();
    no_memory_report.available_device_bytes = 0;
    const Decision unknown_memory = decide(no_memory_report);
    require(unknown_memory.selected_batch_size == 0,
            "missing memory information did not fail closed");
    require_contains(unknown_memory.limiting_reason, "not reported",
                     "missing memory information was not explained");

    Inputs unsupported = ample_inputs();
    unsupported.cooperative_launch_supported = false;
    const Decision host_only = decide(unsupported);
    require(host_only.selected_batch_size == 4 &&
                !host_only.cooperative_allowed,
            "lack of cooperative support incorrectly blocked host batching");
    require_contains(host_only.limiting_reason, "host-batch",
                     "host fallback requirement was not explained");

    Inputs index_limited = ample_inputs();
    index_limited.vertex_count =
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()) /
        2;
    index_limited.edge_count = 0;
    index_limited.source_count = 1;
    index_limited.target_count = 1;
    index_limited.result_node_capacity = 0;
    index_limited.result_edge_capacity = 0;
    index_limited.memory_reserve_bytes = 0;
    index_limited.available_device_bytes =
        std::numeric_limits<std::size_t>::max();
    index_limited.graph_already_resident = true;
    const Decision index_two = decide(index_limited);
    require(index_two.selected_batch_size == 2 &&
                index_two.maximum_index_batch_size == 2,
            "32-bit composite-state ceiling was not enforced");
    require_contains(index_two.limiting_reason, "32-bit",
                     "composite-index limit was not explained");

    Inputs no_queries = ample_inputs();
    no_queries.queries_waiting = 0;
    const Decision empty = decide(no_queries);
    require(empty.selected_batch_size == 0,
            "empty queue did not produce an empty decision");
    require_contains(empty.limiting_reason, "no independent",
                     "empty queue was not explained");

    Inputs no_vertices = ample_inputs();
    no_vertices.vertex_count = 0;
    const Decision empty_graph = decide(no_vertices);
    require(empty_graph.selected_batch_size == 0,
            "zero-vertex graph did not fail closed");
    require_contains(empty_graph.limiting_reason, "zero vertices",
                     "zero-vertex graph was not explained");

    std::cout << "BF12 worker policy test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF12 worker policy test failed: " << error.what() << '\n';
    return 1;
  }
}
