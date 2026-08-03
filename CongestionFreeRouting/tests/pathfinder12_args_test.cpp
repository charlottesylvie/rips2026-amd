// Build from the repository root:
//   g++ -std=c++17 -O2 -pthread -D__HIP_PLATFORM_AMD__=1 \
//     -I Routing/tests/fake_hip -I HIP_kernel/bellman_ford/src \
//     -I CongestionFreeRouting -I CongestionFreeRouting/bellman_ford \
//     CongestionFreeRouting/tests/pathfinder12_args_test.cpp \
//     CongestionFreeRouting/pathfinder12.cpp \
//     -DROUTING_PATHFINDER12_NO_MAIN \
//     -o /tmp/pathfinder12_args_test
//   /tmp/pathfinder12_args_test

#include "../pathfinder12.hpp"
#include "bf12_pathfinder_cpu_stub.inc"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

routing::Pathfinder12CommandLine parse(
    const std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (const std::string& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  return routing::parse_pathfinder12_args(
      static_cast<int>(argv.size()), argv.data());
}

template <typename Function>
void require_rejected(Function&& function, const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

}  // namespace

int main() {
  try {
    const auto defaults = parse({"pathfinder12", "design.csrbin"});
    require(defaults.csr_path == "design.csrbin", "CSR positional path changed");
    require(defaults.metadata_path == "design.csrbin.ifmeta.bin",
            "default metadata path changed");
    require(defaults.options.bf12_batch_size == 0,
            "automatic batch size is not the default");
    require(defaults.options.bf12_controller ==
                BellmanFord12ControllerMode::Auto,
            "automatic controller is not the default");
    require(defaults.options.bf12_enable_bounding_boxes,
            "bounding boxes unexpectedly default off");
    require(defaults.options.bf12_enable_unbounded_retry,
            "bounded fallback unexpectedly defaults off");

    const auto configured = parse(
        {"pathfinder12",
         "design.csrbin",
         "design.ifmeta.bin",
         "--sssp-engine",
         "bf12",
         "--bf12-batch-size",
         "8",
         "--bf12-controller",
         "host-batch",
         "--bf12-target-check-interval",
         "3",
         "--bf12-unbounded",
         "--bf12-bbox-margin-x",
         "7",
         "--bf12-bbox-margin-y",
         "9",
         "--bf12-no-unbounded-fallback",
         "--bf12-result-arena-nodes",
         "101",
         "--bf12-result-arena-edges",
         "202",
         "--bf12-memory-reserve-mib",
         "64",
         "--bf12-telemetry",
         "--max-sssp-iters",
         "42",
         "--capacity",
         "2",
         "--net-limit",
         "11",
         "--parallel-net-workers",
         "1",
         "--routes-out",
         "routes.jsonl",
         "--allow-unrouted"});
    require(configured.metadata_path == "design.ifmeta.bin",
            "explicit metadata path changed");
    require(configured.options.bf12_batch_size == 8,
            "BF12 batch size was not parsed");
    require(configured.options.bf12_controller ==
                BellmanFord12ControllerMode::HostBatch,
            "host controller was not parsed");
    require(configured.options.bf12_target_check_interval == 3,
            "target interval was not parsed");
    require(!configured.options.bf12_enable_bounding_boxes,
            "unbounded mode was not parsed");
    require(!configured.options.bf12_enable_unbounded_retry,
            "no-fallback mode was not parsed");
    require(configured.options.bf12_bbox_margin_x == 7 &&
                configured.options.bf12_bbox_margin_y == 9,
            "bounding margins were not parsed");
    require(configured.options.bf12_result_node_capacity == 101 &&
                configured.options.bf12_result_edge_capacity == 202,
            "result arenas were not parsed");
    require(configured.options.bf12_memory_reserve_bytes ==
                64ULL * 1024ULL * 1024ULL,
            "memory reserve was not converted to bytes");
    require(configured.options.bf12_enable_telemetry,
            "telemetry was not parsed");
    require(configured.options.max_sssp_iterations == 42 &&
                configured.options.capacity == 2 &&
                configured.options.net_limit == 11,
            "general one-shot controls were not parsed");
    require(configured.routes_out_path == "routes.jsonl" &&
                configured.allow_unrouted,
            "route-output controls were not parsed");

    require(routing::parse_bf12_controller("cooperative-batch") ==
                BellmanFord12ControllerMode::CooperativeBatch,
            "cooperative controller spelling changed");
    require_rejected(
        [] { (void)routing::parse_bf12_controller("cooperative"); },
        "invalid controller was accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--sssp-engine", "bf11"}); },
        "non-BF12 engine was accepted");
    const auto legacy_batch =
        parse({"pathfinder12", "g", "--route-batch-size", "4"});
    require(legacy_batch.options.bf12_batch_size == 0,
            "legacy route-batch flag silently controlled BF12");
    require_rejected(
        [] {
          (void)parse({"pathfinder12", "g",
                       "--bf12-target-check-interval", "0"});
        },
        "zero target-check interval was accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--bf12-bbox-margin-x", "-1"}); },
        "negative bounding margin was accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--max-sssp-iters", "-2"}); },
        "invalid maximum iteration policy was accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--parallel-net-workers", "2"}); },
        "multiple BF12 GPU workers were accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--bf12-batch-size"}); },
        "missing BF12 batch value was accepted");
    require_rejected(
        [] {
          (void)parse({"pathfinder12", "g", "--bf12-memory-reserve-mib",
                       "18446744073709551615"});
        },
        "memory-reserve byte overflow was accepted");
    require_rejected(
        [] { (void)parse({"pathfinder12", "g", "--unknown"}); },
        "unknown option was accepted");

    std::cout << "Pathfinder12 argument test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Pathfinder12 argument test failed: " << error.what() << '\n';
    return 1;
  }
}
