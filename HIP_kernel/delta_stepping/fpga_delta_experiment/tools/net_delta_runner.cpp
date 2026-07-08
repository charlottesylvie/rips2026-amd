// Benchmark delta-stepping over route requests from a RIPSIFM1 metadata file.
//
// This intentionally does not run PathFinder's rip-up/reroute loop. It loads
// the same CSR graph and route requests, then times source-to-sink SSSP calls.

#define ROUTING_PATHFINDER_NO_MAIN
#include "pathfinder.hpp"
#include "delta_stepping_hip_CSR.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::high_resolution_clock;

struct Options {
  std::filesystem::path csr;
  std::filesystem::path metadata;
  float delta = 0.0f;
  int max_sssp_iters = -1;
  int repeat = 1;
  std::size_t net_limit = 0;
  std::size_t pair_limit = 0;
};

void usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --csr graph.csrbin --metadata graph.ifmeta.bin --delta D\n"
      << "       [--net-limit N] [--pair-limit N] [--repeat N]"
      << " [--max-sssp-iters N]\n";
}

int parse_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

std::size_t parse_size(const char* text, const char* name) {
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

float parse_float(const char* text, const char* name) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !(value > 0.0f) || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[++i];
    };

    if (arg == "--csr") {
      options.csr = need_value("--csr");
    } else if (arg == "--metadata") {
      options.metadata = need_value("--metadata");
    } else if (arg == "--delta") {
      options.delta = parse_float(need_value("--delta"), "delta");
    } else if (arg == "--max-sssp-iters") {
      options.max_sssp_iters = parse_int(need_value("--max-sssp-iters"), "max-sssp-iters");
    } else if (arg == "--repeat") {
      options.repeat = parse_int(need_value("--repeat"), "repeat");
    } else if (arg == "--net-limit") {
      options.net_limit = parse_size(need_value("--net-limit"), "net-limit");
    } else if (arg == "--pair-limit") {
      options.pair_limit = parse_size(need_value("--pair-limit"), "pair-limit");
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.csr.empty()) throw std::runtime_error("--csr is required");
  if (options.metadata.empty()) throw std::runtime_error("--metadata is required");
  if (!(options.delta > 0.0f)) throw std::runtime_error("--delta is required");
  if (options.repeat < 1) throw std::runtime_error("--repeat must be >= 1");
  return options;
}

void print_json_string(std::ostream& out, const std::string& text) {
  out << '"';
  for (const unsigned char ch : text) {
    if (ch == '"' || ch == '\\') {
      out << '\\' << static_cast<char>(ch);
    } else if (ch == '\n') {
      out << "\\n";
    } else if (ch == '\r') {
      out << "\\r";
    } else if (ch == '\t') {
      out << "\\t";
    } else if (ch < 0x20) {
      out << ' ';
    } else {
      out << static_cast<char>(ch);
    }
  }
  out << '"';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const HostCsrF32 graph = routing::load_csrbin(options.csr);
    const routing::RoutingMetadata metadata =
        routing::load_interchange_metadata(options.metadata);

    const std::size_t request_count =
        options.net_limit == 0
            ? metadata.route_requests.size()
            : std::min(options.net_limit, metadata.route_requests.size());

    double best_ms = std::numeric_limits<double>::infinity();
    double total_ms = 0.0;
    std::size_t total_pairs = 0;
    std::size_t reached_pairs = 0;
    std::size_t attempted_pairs = 0;
    double distance_checksum = 0.0;
    int max_iterations_used = 0;

    for (int repeat = 0; repeat < options.repeat; ++repeat) {
      std::size_t pairs_this_repeat = 0;
      const auto t0 = Clock::now();

      for (std::size_t net_index = 0; net_index < request_count; ++net_index) {
        const routing::RouteRequest& request = metadata.route_requests[net_index];
        for (const routing::SitePinNode& sink : request.sinks) {
          for (const routing::SitePinNode& source : request.sources) {
            if (options.pair_limit != 0 && pairs_this_repeat >= options.pair_limit) {
              goto repeat_done;
            }
            if (source.node < 0 || sink.node < 0) {
              continue;
            }

            DeltaSteppingCsrResult result =
                delta_stepping_minplus_hip_csr(graph,
                                               source.node,
                                               sink.node,
                                               options.delta,
                                               options.max_sssp_iters,
                                               nullptr,
                                               nullptr,
                                               nullptr);
            ++pairs_this_repeat;
            if (repeat == 0) {
              ++attempted_pairs;
            }
            max_iterations_used = std::max(max_iterations_used, result.iterations_used);
            if (result.target_reached && std::isfinite(result.target_distance)) {
              if (repeat == 0) {
                ++reached_pairs;
                distance_checksum += static_cast<double>(sink.node + 1) *
                                     static_cast<double>(result.target_distance);
              }
            }
          }
        }
      }

    repeat_done:
      const auto t1 = Clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      best_ms = std::min(best_ms, ms);
      total_ms += ms;
      total_pairs += pairs_this_repeat;
    }

    const double avg_ms = total_ms / static_cast<double>(options.repeat);
    const double pairs_per_second =
        total_ms > 0.0 ? 1000.0 * static_cast<double>(total_pairs) / total_ms : 0.0;

    std::cout << std::setprecision(10)
              << "{\"ok\":true"
              << ",\"route_requests\":" << metadata.route_requests.size()
              << ",\"net_count\":" << request_count
              << ",\"attempted_pairs\":" << attempted_pairs
              << ",\"reached\":" << reached_pairs
              << ",\"repeat\":" << options.repeat
              << ",\"total_pairs\":" << total_pairs
              << ",\"iterations\":" << max_iterations_used
              << ",\"checksum\":" << distance_checksum
              << ",\"runtime_ms_avg\":" << avg_ms
              << ",\"runtime_ms_best\":" << best_ms
              << ",\"pairs_per_second\":" << pairs_per_second
              << ",\"csr\":";
    print_json_string(std::cout, options.csr.string());
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    std::cout << "{\"ok\":false,\"error\":";
    print_json_string(std::cout, ex.what());
    std::cout << "}\n";
    return 1;
  }
}
