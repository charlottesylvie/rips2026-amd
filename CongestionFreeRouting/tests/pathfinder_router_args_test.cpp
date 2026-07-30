#define PATHFINDER_ROUTER_NO_MAIN

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../pathfinder_router.cpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Options parse(const std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (const std::string& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  return parse_args(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

int main() {
  try {
    const Options defaults =
        parse({"PathFinderFile", "design_unrouted.phys", "design_routed.phys"});
    require(defaults.pathfinder_args.empty(),
            "default router invocation unexpectedly forwarded PathFinder options");
    require(defaults.logical_netlist == "design.netlist",
            "default router invocation did not infer the logical netlist");
    require(defaults.allow_unrouted,
            "default router invocation unexpectedly enabled strict routing");

    const Options weighted = parse(
        {"PathFinderFile",
         "design_unrouted.phys",
         "design_routed.phys",
         "--sssp-engine",
         "delta-step",
         "--delta",
         "0.75",
         "--delta-force-generic",
         "--delta-telemetry",
         "--delta-telemetry",
         "--delta-force-legacy-parent",
         "--delta-controller",
         "reduced-round-trip",
         "--delta-controller-batch-size",
         "7",
         "--delta-query-batch-width",
         "3",
         "--delta-batch-blocks-per-cu",
         "2",
         "--delta-benchmark-weights",
         "mixed",
         "--delta-benchmark-weight-seed",
         "17"});
    require(weighted.pathfinder_args ==
                std::vector<std::string>({"--sssp-engine",
                                          "delta-step",
                                          "--delta",
                                          "0.75",
                                          "--delta-force-generic",
                                          "--delta-telemetry",
                                          "--delta-force-legacy-parent",
                                          "--delta-controller",
                                          "reduced-round-trip",
                                          "--delta-controller-batch-size",
                                          "7",
                                          "--delta-query-batch-width",
                                          "3",
                                          "--delta-batch-blocks-per-cu",
                                          "2",
                                          "--delta-benchmark-weights",
                                          "mixed",
                                          "--delta-benchmark-weight-seed",
                                          "17"}),
            "router did not preserve forced-generic weighted option forwarding");
    require(std::count(weighted.pathfinder_args.begin(),
                       weighted.pathfinder_args.end(),
                       "--delta-telemetry") == 1,
            "router did not forward delta telemetry exactly once");

    const Options fused = parse(
        {"PathFinderFile",
         "design_unrouted.phys",
         "design_routed.phys",
         "--sssp-engine",
         "delta-step",
         "--delta-controller",
         "fused-host-checked"});
    require(fused.pathfinder_args ==
                std::vector<std::string>({"--sssp-engine",
                                          "delta-step",
                                          "--delta-controller",
                                          "fused-host-checked"}),
            "router did not forward the fused host-checked controller");

    const Options shorthand = parse(
        {"PathFinderFile",
         "design_unrouted.phys",
         "design_routed.phys",
         "--use-delta-step",
         "--delta-force-generic"});
    require(shorthand.pathfinder_args ==
                std::vector<std::string>({"--use-delta-step",
                                          "--delta-force-generic"}),
            "router did not forward the delta shorthand and force-generic flag");

    bool missing_seed_rejected = false;
    try {
      (void)parse({"PathFinderFile",
                   "design_unrouted.phys",
                   "design_routed.phys",
                   "--delta-benchmark-weight-seed"});
    } catch (const std::runtime_error&) {
      missing_seed_rejected = true;
    }
    require(missing_seed_rejected,
            "router accepted a benchmark seed option without its value");

    bool missing_controller_batch_rejected = false;
    try {
      (void)parse({"PathFinderFile",
                   "design_unrouted.phys",
                   "design_routed.phys",
                   "--delta-controller-batch-size"});
    } catch (const std::runtime_error&) {
      missing_controller_batch_rejected = true;
    }
    require(missing_controller_batch_rejected,
            "router accepted a controller batch option without its value");

    for (const std::string& controller :
         std::vector<std::string>({"host-checked", "fused-host-checked"})) {
      bool incompatible_batch_rejected = false;
      try {
        (void)parse({"PathFinderFile",
                     "design_unrouted.phys",
                     "design_routed.phys",
                     "--delta-controller",
                     controller,
                     "--delta-controller-batch-size",
                     "4"});
      } catch (const std::runtime_error&) {
        incompatible_batch_rejected = true;
      }
      require(incompatible_batch_rejected,
              "router accepted a controller batch size with " + controller);
    }

    bool batch_without_controller_rejected = false;
    try {
      (void)parse({"PathFinderFile",
                   "design_unrouted.phys",
                   "design_routed.phys",
                   "--delta-controller-batch-size",
                   "4"});
    } catch (const std::runtime_error&) {
      batch_without_controller_rejected = true;
    }
    require(batch_without_controller_rejected,
            "router accepted a controller batch size without a controller");

    for (const std::string& batching_option :
         std::vector<std::string>({"--delta-query-batch-width",
                                   "--delta-batch-blocks-per-cu"})) {
      bool without_cooperative_controller_rejected = false;
      try {
        (void)parse({"PathFinderFile",
                     "design_unrouted.phys",
                     "design_routed.phys",
                     batching_option,
                     "2"});
      } catch (const std::runtime_error&) {
        without_cooperative_controller_rejected = true;
      }
      require(without_cooperative_controller_rejected,
              "router accepted " + batching_option +
                  " without an explicit cooperative controller");

      for (const std::string& invalid :
           std::vector<std::string>({"0", "9", "not-an-integer"})) {
        bool invalid_bound_rejected = false;
        try {
          (void)parse({"PathFinderFile",
                       "design_unrouted.phys",
                       "design_routed.phys",
                       "--delta-controller",
                       "fused-host-checked",
                       batching_option,
                       invalid});
        } catch (const std::runtime_error&) {
          invalid_bound_rejected = true;
        }
        require(invalid_bound_rejected,
                "router accepted out-of-range " + batching_option);
      }
    }

    bool invalid_controller_rejected = false;
    try {
      (void)parse({"PathFinderFile",
                   "design_unrouted.phys",
                   "design_routed.phys",
                   "--delta-controller",
                   "fast"});
    } catch (const std::runtime_error&) {
      invalid_controller_rejected = true;
    }
    require(invalid_controller_rejected,
            "router accepted an unknown Delta controller mode");

    for (const std::string& weight_family :
         std::vector<std::string>({"unit", "all-light", "all-heavy"})) {
      bool incompatible_seed_rejected = false;
      try {
        (void)parse({"PathFinderFile",
                     "design_unrouted.phys",
                     "design_routed.phys",
                     "--delta-benchmark-weight-seed",
                     "17",
                     "--delta-benchmark-weights",
                     weight_family});
      } catch (const std::runtime_error&) {
        incompatible_seed_rejected = true;
      }
      require(incompatible_seed_rejected,
              "router accepted a benchmark seed with " + weight_family +
                  " weights");
    }

    bool missing_family_rejected = false;
    try {
      (void)parse({"PathFinderFile",
                   "design_unrouted.phys",
                   "design_routed.phys",
                   "--delta-benchmark-weight-seed",
                   "17"});
    } catch (const std::runtime_error&) {
      missing_family_rejected = true;
    }
    require(missing_family_rejected,
            "router accepted a benchmark seed without mixed weights");

    std::cout << "PathFinder router argument test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PathFinder router argument test failed: " << error.what()
              << '\n';
    return 1;
  }
}
