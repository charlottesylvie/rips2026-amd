#include "pathfinder.hpp"
#include "profiling/roctx_ranges.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::vector<routing::SsspEngine> engines;
  int warmups = 1;
  int repeats = 5;
  float delta = 4.0f;
  int max_sssp_iterations = -1;
  std::size_t net_limit = 0;
  // A comparison must use the same cross-net parallelism for every engine.
  // One worker measures the implementation rather than engine-specific
  // auto-scheduling policy; callers may explicitly choose another count.
  std::size_t parallel_net_workers = 1;
  bool verify = false;
  bool allow_inexact_unit_bfs = false;
  std::string baseline = "delta-step";
  std::filesystem::path json_out;
};

struct TimingStats {
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct SinkSignature {
  bool reached = false;
  float distance = std::numeric_limits<float>::infinity();
};

struct EngineSummary {
  routing::SsspEngine engine = routing::SsspEngine::kDeltaStep;
  bool exact_weighted_result = true;
  TimingStats wall;
  routing::PathfinderResult final_result;
  std::vector<SinkSignature> signature;
  std::size_t reached_sinks = 0;
  std::size_t total_sinks = 0;
  std::size_t route_edges = 0;
  double distance_checksum = 0.0;
  std::string verification_error;
};

std::filesystem::path default_metadata_path(const std::filesystem::path& csr_path) {
  std::filesystem::path metadata_path = csr_path;
  metadata_path += ".ifmeta.bin";
  return metadata_path;
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
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

routing::SsspEngine parse_engine(const std::string& value) {
  if (value == "unit-bfs" || value == "bfs") {
    return routing::SsspEngine::kUnitBfs;
  }
  if (value == "delta-step" || value == "delta-stepping" || value == "delta") {
    return routing::SsspEngine::kDeltaStep;
  }
  if (value == "bellman-ford" || value == "bellman_ford" || value == "bf8") {
    return routing::SsspEngine::kBellmanFord;
  }
  throw std::runtime_error("unsupported SSSP engine: " + value);
}

const char* engine_name(routing::SsspEngine engine) {
  switch (engine) {
    case routing::SsspEngine::kUnitBfs:
      return "unit-bfs";
    case routing::SsspEngine::kDeltaStep:
      return "delta-step";
    case routing::SsspEngine::kBellmanFord:
      return "bellman-ford";
  }
  return "unknown";
}

std::vector<routing::SsspEngine> parse_engines(const std::string& text) {
  std::vector<routing::SsspEngine> engines;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string::npos ? text.size() : comma;
    if (end == begin) {
      throw std::runtime_error("--engines contains an empty engine name");
    }
    const routing::SsspEngine engine =
        parse_engine(text.substr(begin, end - begin));
    if (std::find(engines.begin(), engines.end(), engine) != engines.end()) {
      throw std::runtime_error("--engines must not contain duplicates");
    }
    engines.push_back(engine);
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (engines.empty()) {
    throw std::runtime_error("--engines must not be empty");
  }
  return engines;
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <snapshot.csrbin> [metadata.ifmeta.bin] [options]\n\n"
      << "Routes all nets exactly once per engine; it never runs the congestion loop.\n\n"
      << "Options:\n"
      << "  --engines <delta-step,bellman-ford>  Engines to compare.\n"
      << "  --warmups <count>                    Untimed passes per engine. Default: 1\n"
      << "  --repeats <count>                    Timed passes per engine. Default: 5\n"
      << "  --delta <float>                      Delta width. Default: 4\n"
      << "  --max-sssp-iters <int>               Per-query iteration cap. Default: -1\n"
      << "  --net-limit <count>                  Route only the first count requests.\n"
      << "  --parallel-net-workers <count>       Per-pass net-worker setting. Default: 1\n"
      << "  --verify                             Compare reached states/distances to --baseline.\n"
      << "  --baseline <engine>                  Verification/speedup baseline. Default: delta-step\n"
      << "  --allow-inexact-unit-bfs             Permit unit-BFS on a weighted snapshot.\n"
      << "  --json-out <path>                    Write the comparison JSON to this path.\n";
}

Options parse_args(int argc, char** argv) {
  if (argc < 2 || (argc == 2 && std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    std::exit(argc < 2 ? 1 : 0);
  }

  Options options;
  options.csr_path = argv[1];
  int arg = 2;
  if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
    options.metadata_path = argv[arg++];
  } else {
    options.metadata_path = default_metadata_path(options.csr_path);
  }
  std::string engine_text = "delta-step,bellman-ford";
  for (; arg < argc; ++arg) {
    const std::string option = argv[arg];
    auto require_value = [&](const char* name) -> const char* {
      if (++arg >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[arg];
    };
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (option == "--engines") {
      engine_text = require_value("--engines");
    } else if (option == "--warmups") {
      options.warmups = parse_int(require_value("--warmups"), "warmups");
    } else if (option == "--repeats") {
      options.repeats = parse_int(require_value("--repeats"), "repeats");
    } else if (option == "--delta") {
      options.delta = parse_float(require_value("--delta"), "delta");
    } else if (option == "--max-sssp-iters") {
      options.max_sssp_iterations =
          parse_int(require_value("--max-sssp-iters"), "max-sssp-iters");
    } else if (option == "--net-limit") {
      options.net_limit = parse_size(require_value("--net-limit"), "net-limit");
    } else if (option == "--parallel-net-workers") {
      options.parallel_net_workers = parse_size(
          require_value("--parallel-net-workers"), "parallel-net-workers");
    } else if (option == "--verify") {
      options.verify = true;
    } else if (option == "--baseline") {
      options.baseline = require_value("--baseline");
    } else if (option == "--allow-inexact-unit-bfs") {
      options.allow_inexact_unit_bfs = true;
    } else if (option == "--json-out") {
      options.json_out = require_value("--json-out");
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (options.warmups < 0 || options.repeats <= 0) {
    throw std::runtime_error("--warmups must be nonnegative and --repeats positive");
  }
  if (options.parallel_net_workers == 0) {
    throw std::runtime_error(
        "--parallel-net-workers must be positive for a fair engine comparison");
  }
  if (!(options.delta > 0.0f)) {
    throw std::runtime_error("--delta must be positive");
  }
  options.engines = parse_engines(engine_text);
  // Accept the same aliases as --engines, but preserve one canonical spelling
  // for JSON matching and speedup computation.
  options.baseline = engine_name(parse_engine(options.baseline));
  if (std::none_of(options.engines.begin(), options.engines.end(),
                   [&](routing::SsspEngine engine) {
                     return options.baseline == engine_name(engine);
                   })) {
    throw std::runtime_error("--baseline is not included in --engines");
  }
  return options;
}

bool is_exact_unit_weight(const HostCsrF32& graph) {
  return std::all_of(graph.values.begin(), graph.values.end(),
                     [](float value) { return value == 1.0f; });
}

TimingStats calculate_timing(const std::vector<double>& samples) {
  if (samples.empty()) {
    throw std::logic_error("timing requires at least one sample");
  }
  TimingStats stats;
  stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) /
                  static_cast<double>(samples.size());
  const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
  stats.min_ms = *minimum;
  stats.max_ms = *maximum;
  return stats;
}

std::vector<SinkSignature> make_signature(const routing::PathfinderResult& result,
                                          std::size_t* reached_sinks,
                                          std::size_t* total_sinks,
                                          std::size_t* route_edges,
                                          double* distance_checksum) {
  std::vector<SinkSignature> signature;
  *reached_sinks = 0;
  *total_sinks = 0;
  *route_edges = 0;
  *distance_checksum = 0.0;
  for (const routing::RoutedNet& net : result.nets) {
    for (const routing::RoutedSink& sink : net.sinks) {
      ++(*total_sinks);
      signature.push_back({sink.reached, sink.distance});
      if (sink.reached) {
        ++(*reached_sinks);
        *route_edges += sink.edges.size();
        *distance_checksum += sink.distance;
      }
    }
  }
  return signature;
}

bool signatures_match(const std::vector<SinkSignature>& baseline,
                      const std::vector<SinkSignature>& candidate,
                      std::string* error) {
  if (baseline.size() != candidate.size()) {
    *error = "different sink counts";
    return false;
  }
  for (std::size_t index = 0; index < baseline.size(); ++index) {
    if (baseline[index].reached != candidate[index].reached) {
      *error = "reached-state mismatch at sink " + std::to_string(index);
      return false;
    }
    if (!baseline[index].reached) {
      continue;
    }
    const float lhs = baseline[index].distance;
    const float rhs = candidate[index].distance;
    const float tolerance = 1e-3f * std::max({1.0f, std::fabs(lhs), std::fabs(rhs)});
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || std::fabs(lhs - rhs) > tolerance) {
      *error = "distance mismatch at sink " + std::to_string(index);
      return false;
    }
  }
  return true;
}

EngineSummary run_engine(const HostCsrF32& graph,
                         const routing::RoutingMetadata& metadata,
                         const Options& options,
                         routing::SsspEngine engine) {
  EngineSummary summary;
  summary.engine = engine;
  summary.exact_weighted_result =
      engine != routing::SsspEngine::kUnitBfs || is_exact_unit_weight(graph);

  routing::PathfinderOptions pathfinder_options;
  pathfinder_options.sssp_engine = engine;
  pathfinder_options.delta = options.delta;
  pathfinder_options.max_sssp_iterations = options.max_sssp_iterations;
  pathfinder_options.net_limit = options.net_limit;
  pathfinder_options.parallel_net_workers = options.parallel_net_workers;
  pathfinder_options.report_progress = false;

  const std::string range_name =
      std::string("pathfinder.compare.") + engine_name(engine);
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    pathfinder_profile::ScopedRange range(range_name.c_str());
    (void)routing::run_pathfinder(graph, metadata, pathfinder_options, nullptr);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.repeats));
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    pathfinder_profile::ScopedRange range(range_name.c_str());
    const auto start = Clock::now();
    summary.final_result =
        routing::run_pathfinder(graph, metadata, pathfinder_options, nullptr);
    const auto finish = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(finish - start).count());
  }
  summary.wall = calculate_timing(samples);
  summary.signature = make_signature(summary.final_result,
                                     &summary.reached_sinks,
                                     &summary.total_sinks,
                                     &summary.route_edges,
                                     &summary.distance_checksum);
  return summary;
}

void write_json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << static_cast<char>(ch); break;
    }
  }
  out << '"';
}

void write_json(std::ostream& out,
                const Options& options,
                const std::vector<EngineSummary>& summaries) {
  const auto baseline = std::find_if(
      summaries.begin(), summaries.end(), [&](const EngineSummary& summary) {
        return options.baseline == engine_name(summary.engine);
      });
  const double baseline_ms = baseline == summaries.end() ? 0.0 : baseline->wall.mean_ms;
  out << std::fixed << std::setprecision(6);
  out << "{\n  \"snapshot\": ";
  write_json_string(out, options.csr_path.string());
  out << ",\n  \"metadata\": ";
  write_json_string(out, options.metadata_path.string());
  out << ",\n  \"warmups\": " << options.warmups
      << ",\n  \"repeats\": " << options.repeats
      << ",\n  \"parallel_net_workers\": " << options.parallel_net_workers
      << ",\n  \"baseline\": ";
  write_json_string(out, options.baseline);
  out << ",\n  \"engines\": [\n";
  for (std::size_t index = 0; index < summaries.size(); ++index) {
    const EngineSummary& summary = summaries[index];
    const double speedup = summary.wall.mean_ms > 0.0
        ? baseline_ms / summary.wall.mean_ms : 0.0;
    out << "    {\"engine\": ";
    write_json_string(out, engine_name(summary.engine));
    out << ", \"exact_weighted_result\": "
        << (summary.exact_weighted_result ? "true" : "false")
        << ", \"wall_ms\": {\"mean\": " << summary.wall.mean_ms
        << ", \"min\": " << summary.wall.min_ms
        << ", \"max\": " << summary.wall.max_ms << "}"
        << ", \"speedup_vs_baseline\": " << speedup
        << ", \"all_sinks_reached\": "
        << (summary.final_result.all_sinks_reached ? "true" : "false")
        << ", \"overused_nodes\": " << summary.final_result.overused_nodes
        << ", \"reached_sinks\": " << summary.reached_sinks
        << ", \"total_sinks\": " << summary.total_sinks
        << ", \"route_edges\": " << summary.route_edges
        << ", \"distance_checksum\": " << summary.distance_checksum;
    if (!summary.verification_error.empty()) {
      out << ", \"verification_error\": ";
      write_json_string(out, summary.verification_error);
    }
    out << '}';
    if (index + 1 != summaries.size()) {
      out << ',';
    }
    out << '\n';
  }
  out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    const HostCsrF32 graph = routing::load_csrbin(options.csr_path);
    const routing::RoutingMetadata metadata =
        routing::load_interchange_metadata(options.metadata_path);
    const bool unit_weight = is_exact_unit_weight(graph);

    if (!unit_weight && !options.allow_inexact_unit_bfs &&
        std::find(options.engines.begin(), options.engines.end(),
                  routing::SsspEngine::kUnitBfs) != options.engines.end()) {
      throw std::runtime_error(
          "unit-bfs ignores CSR weights and cannot correctly compare a "
          "congestion snapshot; remove it or pass --allow-inexact-unit-bfs");
    }
    std::vector<EngineSummary> summaries;
    summaries.reserve(options.engines.size());
    for (const routing::SsspEngine engine : options.engines) {
      std::cerr << "[pathfinder-compare] engine=" << engine_name(engine) << "\n";
      summaries.push_back(run_engine(graph, metadata, options, engine));
    }

    bool verified = true;
    if (options.verify) {
      const auto baseline = std::find_if(
          summaries.begin(), summaries.end(), [&](const EngineSummary& summary) {
            return options.baseline == engine_name(summary.engine);
          });
      for (EngineSummary& summary : summaries) {
        if (&summary == &*baseline) {
          continue;
        }
        if (!signatures_match(baseline->signature,
                             summary.signature,
                             &summary.verification_error)) {
          verified = false;
        }
      }
    }

    std::ostringstream json;
    write_json(json, options, summaries);
    std::cout << json.str();
    if (!options.json_out.empty()) {
      if (options.json_out.has_parent_path()) {
        std::filesystem::create_directories(options.json_out.parent_path());
      }
      std::ofstream out(options.json_out, std::ios::trunc);
      if (!out) {
        throw std::runtime_error("could not write JSON output: " +
                                 options.json_out.string());
      }
      out << json.str();
    }
    return verified ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
