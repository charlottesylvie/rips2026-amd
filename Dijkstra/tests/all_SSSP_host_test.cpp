#include "../preds_gpu_engine.hpp"

#include <stdexcept>
#include <string>

// The host-only test does not link the HIP translation unit. Supply the two
// parser/formatting helpers used by all_SSSP's pure host orchestration.
namespace rips_predicates_gpu {

const char* predicate_mode_name(PredicateMode mode) {
  switch (mode) {
    case PredicateMode::kInSimple:
      return "IN_SIMPLE";
    case PredicateMode::kOutSimple:
      return "OUT_SIMPLE";
    case PredicateMode::kInSimpleOrOutSimple:
      return "IN_SIMPLE_OR_OUT_SIMPLE";
    case PredicateMode::kInStatic:
      return "IN_STATIC";
    case PredicateMode::kOutStatic:
      return "OUT_STATIC";
    case PredicateMode::kInStaticOrOutStatic:
      return "IN_STATIC_OR_OUT_STATIC";
  }
  throw std::logic_error("unknown predicate mode");
}

PredicateMode parse_predicate_mode(const std::string& text) {
  if (text == "IN_SIMPLE") return PredicateMode::kInSimple;
  if (text == "OUT_SIMPLE") return PredicateMode::kOutSimple;
  if (text == "IN_SIMPLE_OR_OUT_SIMPLE") {
    return PredicateMode::kInSimpleOrOutSimple;
  }
  if (text == "IN_STATIC") return PredicateMode::kInStatic;
  if (text == "OUT_STATIC") return PredicateMode::kOutStatic;
  if (text == "IN_STATIC_OR_OUT_STATIC") {
    return PredicateMode::kInStaticOrOutStatic;
  }
  throw std::invalid_argument("invalid predicate mode");
}

}  // namespace rips_predicates_gpu

#define ALL_SSSP_HOST_TEST
#include "../all_SSSP.cpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("expectation failed: " + message);
  }
}

template <typename Callable>
void expect_throws(Callable&& callable, const std::string& message) {
  bool threw = false;
  try {
    callable();
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, message);
}

class ValidationEngine final : public all_sssp::GpuSsspEngine {
 public:
  const char* algorithm_name() const noexcept override {
    return "test-engine";
  }

  const char* predicate_name() const noexcept override {
    return "n/a";
  }

  all_sssp::SsspRunResult run(
      const all_sssp::SsspRequest&) override {
    throw std::logic_error("test engine run is not used");
  }

  void validate_captured_target_paths(
      all_sssp::Index source,
      const std::vector<all_sssp::TargetResult>& targets) const override {
    validation_called = true;
    validated_source = source;
    validated_target_count = targets.size();
    if (reject_paths) {
      throw std::runtime_error(
          "test engine rejected captured path");
    }
  }

  all_sssp::EngineStatistics cumulative_statistics() const override {
    return {};
  }

  mutable bool validation_called = false;
  mutable all_sssp::Index validated_source = -1;
  mutable std::size_t validated_target_count = 0;
  bool reject_paths = false;
};

all_sssp::SitePinNode pin(int node) {
  all_sssp::SitePinNode result;
  result.node = node;
  return result;
}

all_sssp::RouteRequest request(
    std::uint64_t net,
    std::uint64_t logical,
    std::initializer_list<int> sources,
    std::initializer_list<int> sinks) {
  all_sssp::RouteRequest result;
  result.net_string = net;
  result.logical_net_index = logical;
  for (int source : sources) result.sources.push_back(pin(source));
  for (int sink : sinks) result.sinks.push_back(pin(sink));
  return result;
}

void test_one_request() {
  all_sssp::RoutingMetadata metadata;
  metadata.strings = {"net0"};
  metadata.route_requests.push_back(request(0, 4, {1}, {2}));
  std::ostringstream warnings;
  const all_sssp::WorkBuildResult result =
      all_sssp::build_source_work(metadata, 4, warnings);
  expect(result.raw_source_occurrences == 1, "one raw source");
  expect(result.unique_valid_sources == 1, "one unique source");
  expect(result.raw_source_sink_queries == 1, "one raw query");
  expect(result.valid_source_sink_queries == 1, "one valid query");
  expect(result.work.size() == 1, "one work item");
  expect(result.work[0].source == 1, "source node preserved");
  expect(result.work[0].unique_targets ==
             std::vector<all_sssp::Index>{2},
         "target preserved");
  expect(result.work[0].queries[0].ordinal == 0,
         "first query ordinal");
  expect(warnings.str().empty(), "no warnings for valid request");
}

void test_grouping_matrix() {
  all_sssp::RoutingMetadata metadata;
  metadata.strings = {"net0", "net1", "net2"};
  metadata.route_requests = {
      request(0, 10, {0, 0}, {1, 1, 0}),
      request(1, 11, {0, 2}, {3}),
      request(2, 12, {4}, {}),
      request(1, 13, {9, 3}, {8, 3}),
  };

  std::ostringstream warnings;
  const all_sssp::WorkBuildResult result =
      all_sssp::build_source_work(metadata, 5, warnings);
  expect(result.raw_source_occurrences == 7,
         "all source occurrences counted");
  expect(result.invalid_source_occurrences == 1,
         "invalid source occurrence counted once");
  expect(result.raw_source_sink_queries == 12,
         "raw source-sink cross product");
  expect(result.valid_source_sink_queries == 9,
         "valid queries preserved");
  expect(result.invalid_sink_occurrences == 1,
         "invalid sink endpoint counted once");
  expect(result.invalid_source_queries == 2,
         "queries attached to invalid sources counted");
  expect(result.invalid_sink_queries == 2,
         "invalid sink queries count source multiplicity");
  expect(result.unique_valid_sources == 4,
         "unique source grouping includes empty source");
  expect(result.unique_target_nodes == 3,
         "global target nodes deduplicated");
  expect(result.unique_source_target_pairs == 5,
         "per-source target sets deduplicated");
  expect(result.work.size() == 4, "four stable source groups");
  expect(result.work[0].source == 0 && result.work[1].source == 2 &&
             result.work[2].source == 4 && result.work[3].source == 3,
         "source groups retain first-occurrence order");
  expect(result.work[0].queries.size() == 7,
         "duplicate sources and sinks remain queries");
  expect(result.work[0].unique_targets ==
             std::vector<all_sssp::Index>({1, 0, 3}),
         "targets stable-deduplicate in query order");
  expect(result.work[2].queries.empty() &&
             result.work[2].unique_targets.empty(),
         "valid source with no sinks remains work");
  expect(result.work[3].queries.size() == 1 &&
             result.work[3].queries[0].source ==
                 result.work[3].queries[0].target,
         "source equals sink is retained");
  expect(result.work[3].queries[0].ordinal == 11,
         "query ordinal preserves invalid gaps and metadata order");
  expect(warnings.str().find("source node 9") != std::string::npos,
         "invalid source warning");
  expect(warnings.str().find("sink node 8") != std::string::npos,
         "invalid sink warning");

  expect(all_sssp::count_selected_queries(result.work, 1) == 7,
         "source limit one query coverage");
  expect(all_sssp::count_selected_queries(result.work, 3) == 8,
         "empty source included in selected prefix");
  expect(all_sssp::max_unique_targets_per_source(
             result.work, 1, 5) == 3,
         "maximum target reservation for limit one");
  expect(all_sssp::max_unique_targets_per_source(
             result.work, result.work.size(), 5) == 3,
         "maximum target reservation for all");
  expect(all_sssp::selected_source_count(result.work.size(), 0) == 4,
         "source-limit zero selects all");
  expect(all_sssp::selected_source_count(result.work.size(), 1) == 1,
         "source-limit one");
  expect(all_sssp::selected_source_count(result.work.size(), 4) == 4,
         "source-limit equal to total");
  expect(all_sssp::selected_source_count(result.work.size(), 99) == 4,
         "source-limit above total clamps");
}

all_sssp::Options parse(std::vector<std::string> arguments) {
  std::vector<char*> raw;
  raw.reserve(arguments.size());
  for (std::string& argument : arguments) {
    raw.push_back(argument.data());
  }
  return all_sssp::parse_args(
      static_cast<int>(raw.size()), raw.data());
}

void test_cli() {
  const all_sssp::Options defaults = parse(
      {"all_SSSP", "g.csrbin", "g.ifmeta.bin",
       "--algorithm", "preds-gpu", "--predicate", "IN_STATIC"});
  expect(!defaults.early_stop, "full convergence is default");
  expect(defaults.stats_every == 100, "default statistics interval");
  expect(defaults.source_limit == 0, "default source limit means all");
  expect(!defaults.paths_output_set, "paths disabled by default");

  const all_sssp::Options configured = parse(
      {"all_SSSP", "g.csrbin", "g.ifmeta.bin",
       "--algorithm", "preds-gpu",
       "--predicate", "IN_STATIC_OR_OUT_STATIC",
       "--early-stop", "--stats-every", "0",
       "--source-limit", "3", "--paths-output", "paths.jsonl"});
  expect(configured.early_stop, "early stop parsed");
  expect(configured.stats_every == 0, "zero statistics interval parsed");
  expect(configured.source_limit == 3, "source limit parsed");
  expect(configured.paths_output_set, "path output parsed");

  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--algorithm", "preds-gpu", "--predicate", "IN_STATIC"});
      },
      "repeated algorithm rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC", "--predicate", "OUT_STATIC"});
      },
      "repeated predicate rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--early-stop", "--no-early-stop"});
      },
      "conflicting stop flags rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--stats-every", "1", "--stats-every", "3"});
      },
      "repeated stats interval rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--source-limit", "1", "--source-limit", "2"});
      },
      "repeated source limit rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--paths-output", "a", "--paths-output", "b"});
      },
      "repeated path output rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "bf10"});
      },
      "unavailable engine rejected");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "bf10",
               "--predicate", "IN_STATIC"});
      },
      "predicate rejected for predicate-free engine");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "delta-step"});
      },
      "recognized delta-step adapter gap reported");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "unit-bfs"});
      },
      "recognized unit-bfs adapter gap reported");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "not-an-engine"});
      },
      "unknown engine rejected separately");
  expect_throws(
      [] {
        parse({"x", "g", "m", "--algorithm", "preds-gpu"});
      },
      "missing predicate rejected");
  expect_throws(
      [] { parse({"x", "--help", "-h"}); },
      "repeated help rejected");
  expect_throws(
      [] {
        parse({"x", "graph.csrbin", "graph.ifmeta.bin",
               "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--paths-output", "graph.ifmeta.bin.generation"});
      },
      "generation sidecar cannot be overwritten");
  expect_throws(
      [] {
        parse({"x", "graph.csrbin", "graph.ifmeta.bin",
               "--algorithm", "preds-gpu",
               "--predicate", "IN_STATIC",
               "--paths-output", "graph.csrbin.publishing"});
      },
      "publication marker cannot be overwritten");

  std::ostringstream usage;
  all_sssp::print_usage(usage, "all_SSSP");
  expect(usage.str().find("preds-gpu") != std::string::npos,
         "implemented engine advertised");
  expect(usage.str().find("\n  bf10\n") == std::string::npos &&
             usage.str().find("\n  delta-step\n") == std::string::npos &&
             usage.str().find("\n  unit-bfs\n") == std::string::npos,
         "unimplemented engines omitted from help");
}

void test_nonterminal_progress() {
  const auto exercise = [](std::size_t interval,
                           std::size_t total) {
    std::ostringstream output;
    const all_sssp::Clock::time_point begin =
        all_sssp::Clock::now();
    {
      all_sssp::ProgressReporter progress(
          total, interval, begin, output, false);
      for (std::size_t completed = 0; completed <= total; ++completed) {
        progress.update(completed);
      }
    }
    expect(output.str().find('\r') == std::string::npos,
           "redirected progress has no carriage returns");
    expect(output.str().find(
               std::to_string(total) + "/" + std::to_string(total)) !=
               std::string::npos,
           "redirected progress always has final record");
    return output.str();
  };

  const std::string disabled = exercise(0, 5);
  const std::string every_one = exercise(1, 5);
  const std::string every_three = exercise(3, 5);
  const std::string above_total = exercise(9, 5);
  expect(!disabled.empty(), "stats-every zero fallback progress");
  expect(every_one.size() > every_three.size(),
         "interval one emits more records than interval three");
  expect(every_three.size() > above_total.size(),
         "interval three emits more records than interval above total");
}

void test_artifact_pair_validation() {
  const all_sssp::ArtifactPairId first{1, 2};
  const all_sssp::ArtifactPairId second{3, 4};
  routing::interchange::require_matching_interchange_pair_ids(
      first, first, first);
  routing::interchange::require_matching_interchange_pair_ids(
      std::nullopt, std::nullopt, std::nullopt);
  expect_throws(
      [&] {
        routing::interchange::require_matching_interchange_pair_ids(
            first, second, first);
      },
      "mismatched artifact ids rejected");
  expect_throws(
      [&] {
        routing::interchange::require_matching_interchange_pair_ids(
            first, first, std::nullopt);
      },
      "mixed legacy/tagged artifacts rejected");
}

void test_run_result_validation() {
  all_sssp::SourceWork work;
  work.source = 1;
  work.unique_targets = {1};

  all_sssp::Options options;
  ValidationEngine engine;
  all_sssp::SsspRunResult result;
  result.termination = all_sssp::TerminationReason::kFullConvergence;
  result.fully_converged = true;
  result.all_targets_confirmed = true;
  all_sssp::TargetResult target;
  target.target = 1;
  target.reached = false;
  target.distance = std::numeric_limits<float>::infinity();
  result.targets = {target};
  expect_throws(
      [&] {
        all_sssp::validate_run_result(work, options, result, engine);
      },
      "source-to-self cannot be unreachable");

  result.targets[0].reached = true;
  result.targets[0].distance = 0.0f;
  all_sssp::validate_run_result(work, options, result, engine);
  expect(!engine.validation_called,
         "path validator is skipped when paths are disabled");

  options.paths_output_set = true;
  result.targets[0].path_captured = true;
  result.targets[0].nodes = {1};
  all_sssp::validate_run_result(work, options, result, engine);
  expect(engine.validation_called && engine.validated_source == 1 &&
             engine.validated_target_count == 1,
         "generic validation invokes retained-graph engine contract");

  engine.reject_paths = true;
  expect_throws(
      [&] {
        all_sssp::validate_run_result(work, options, result, engine);
      },
      "engine path-validation failure propagates");
  engine.reject_paths = false;

  result.termination = all_sssp::TerminationReason::kMaxIterations;
  expect_throws(
      [&] {
        all_sssp::validate_run_result(work, options, result, engine);
      },
      "maximum-iteration result is a source failure");
  result.termination = all_sssp::TerminationReason::kError;
  expect_throws(
      [&] {
        all_sssp::validate_run_result(work, options, result, engine);
      },
      "explicit engine error is a source failure");
}

void test_jsonl_schema_and_query_order() {
  all_sssp::RoutingMetadata metadata;
  metadata.route_requests.resize(1);
  all_sssp::Options options;
  options.algorithm = "preds-gpu";
  options.early_stop = true;

  std::ostringstream output;
  all_sssp::write_paths_metadata(
      output, 3, 2, metadata, options, "IN_STATIC", 1, 1);
  all_sssp::Query query;
  query.net_name = "net";
  query.net_index = 7;
  query.logical_net_index = 9;
  query.source = 0;
  query.target = 2;
  all_sssp::TargetResult target;
  target.target = 2;
  target.reached = true;
  target.distance = 3.0f;
  target.path_captured = true;
  target.nodes = {0, 1, 2};
  target.csr_edges = {0, 1};
  all_sssp::write_path_record(output, query, target);
  const std::string jsonl = output.str();
  for (const std::string& field :
       {"\"type\":\"path\"",
        "\"net\":\"net\"",
        "\"net_index\":7",
        "\"logical_net_index\":9",
        "\"source\":0",
        "\"target\":2",
        "\"reached\":true",
        "\"distance\":3",
        "\"nodes\":[0,1,2]",
        "\"csr_edges\":[0,1]"}) {
    expect(jsonl.find(field) != std::string::npos,
           "JSONL path schema field " + field);
  }
  expect(jsonl.find("\"producer\":\"all_SSSP\"") !=
             std::string::npos &&
             jsonl.find("\"algorithm\":\"preds-gpu\"") !=
                 std::string::npos &&
             jsonl.find("\"predicate_mode\":\"IN_STATIC\"") !=
                 std::string::npos,
         "JSONL metadata identifies producer and engine");

  std::vector<all_sssp::SourceWork> work(2);
  for (std::uint64_t ordinal : {5ULL, 9ULL}) {
    all_sssp::Query item;
    item.ordinal = ordinal;
    work[0].queries.push_back(item);
  }
  for (std::uint64_t ordinal : {1ULL, 7ULL}) {
    all_sssp::Query item;
    item.ordinal = ordinal;
    work[1].queries.push_back(item);
  }
  const std::vector<const all_sssp::Query*> ordered =
      all_sssp::selected_queries_in_metadata_order(work, work.size());
  expect(ordered.size() == 4 && ordered[0]->ordinal == 1 &&
             ordered[1]->ordinal == 5 &&
             ordered[2]->ordinal == 7 &&
             ordered[3]->ordinal == 9,
         "grouped source results return to original metadata query order");
}

template <typename T>
void write_scalar(std::ofstream& output, const T& value) {
  output.write(
      reinterpret_cast<const char*>(&value),
      static_cast<std::streamsize>(sizeof(value)));
}

void write_zeros(std::ofstream& output, std::size_t count) {
  const std::vector<char> zeros(count, '\0');
  if (!zeros.empty()) {
    output.write(
        zeros.data(), static_cast<std::streamsize>(zeros.size()));
  }
}

void write_metadata_fixture(const std::filesystem::path& path,
                            std::uint64_t version,
                            all_sssp::ArtifactPairId pair) {
  std::ofstream output(path, std::ios::binary);
  expect(static_cast<bool>(output), "open metadata fixture");
  output.write(all_sssp::kMetadataMagic,
               static_cast<std::streamsize>(
                   sizeof(all_sssp::kMetadataMagic)));
  write_scalar(output, version);
  write_scalar(output, all_sssp::kOutgoingOrientation);
  if (version >= all_sssp::kArtifactPairMetadataVersion) {
    write_scalar(output, pair.high);
    write_scalar(output, pair.low);
  }

  constexpr std::uint64_t strings = 1;
  constexpr std::uint64_t nodes = 4;
  constexpr std::uint64_t edge_attrs = 0;
  constexpr std::uint64_t pip_data = 0;
  constexpr std::uint64_t site_pin_attrs = 0;
  constexpr std::uint64_t route_requests = 1;
  constexpr std::uint64_t zero = 0;
  write_scalar(output, strings);
  write_scalar(output, nodes);
  write_scalar(output, edge_attrs);
  write_scalar(output, pip_data);
  write_scalar(output, site_pin_attrs);
  write_scalar(output, route_requests);
  write_scalar(output, zero);  // blocked nodes
  write_scalar(output, zero);  // sink-stop nodes
  write_scalar(output, zero);  // logical cells
  write_scalar(output, zero);  // logical nets
  write_scalar(output, zero);  // logical port instances
  write_scalar(output, zero);  // physical bytes
  write_scalar(output, zero);  // logical bytes
  for (int field = 0; field < 4; ++field) {
    write_scalar(output, all_sssp::kNoIndex);
  }

  constexpr char net_name[] = "net";
  const std::uint64_t net_name_size = sizeof(net_name) - 1;
  write_scalar(output, net_name_size);
  output.write(net_name, static_cast<std::streamsize>(net_name_size));
  write_zeros(output, nodes * sizeof(std::uint64_t));
  if (version >= all_sssp::kNodePhysicalMetadataVersion) {
    write_zeros(output, nodes * 4 * sizeof(std::int32_t));
    write_zeros(output, nodes * 2 * sizeof(std::uint64_t));
  }

  const std::uint64_t net_string = 0;
  const std::uint64_t logical_net = 17;
  const std::uint64_t source_count = 2;
  write_scalar(output, net_string);
  write_scalar(output, logical_net);
  write_scalar(output, source_count);
  for (std::uint64_t source : {std::uint64_t{0}, std::uint64_t{1}}) {
    write_scalar(output, source);
    write_scalar(output, all_sssp::kNoIndex);
    write_scalar(output, all_sssp::kNoIndex);
  }
  const std::uint64_t sink_count = 2;
  write_scalar(output, sink_count);
  for (std::uint64_t sink : {std::uint64_t{2}, std::uint64_t{2}}) {
    write_scalar(output, sink);
    write_scalar(output, all_sssp::kNoIndex);
    write_scalar(output, all_sssp::kNoIndex);
  }
  output.close();
  expect(static_cast<bool>(output), "write metadata fixture");
}

void write_graph_fixture(const std::filesystem::path& path,
                         std::uint64_t version,
                         all_sssp::ArtifactPairId pair) {
  std::ofstream output(path, std::ios::binary);
  expect(static_cast<bool>(output), "open graph fixture");
  output.write(
      all_sssp::kCsrMagic,
      static_cast<std::streamsize>(sizeof(all_sssp::kCsrMagic)));
  write_scalar(output, version);
  write_scalar(output, all_sssp::kOutgoingOrientation);
  if (version >= all_sssp::kArtifactPairCsrVersion) {
    write_scalar(output, pair.high);
    write_scalar(output, pair.low);
  }
  constexpr std::uint64_t rows = 3;
  constexpr std::uint64_t edges = 2;
  write_scalar(output, rows);
  write_scalar(output, rows);
  write_scalar(output, edges);
  write_scalar(output, edges);
  write_scalar(output, edges);
  const std::uint64_t rowptr_count = rows + 1;
  write_scalar(output, rowptr_count);
  write_scalar(output, edges);
  write_scalar(output, edges);
  const all_sssp::Offset rowptr[] = {0, 1, 2, 2};
  const all_sssp::Index colind[] = {1, 2};
  const float values[] = {1.0f, 2.0f};
  output.write(
      reinterpret_cast<const char*>(rowptr),
      static_cast<std::streamsize>(sizeof(rowptr)));
  output.write(
      reinterpret_cast<const char*>(colind),
      static_cast<std::streamsize>(sizeof(colind)));
  output.write(
      reinterpret_cast<const char*>(values),
      static_cast<std::streamsize>(sizeof(values)));
  output.close();
  expect(static_cast<bool>(output), "write graph fixture");
}

void test_binary_loaders() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("all_SSSP_host_test_" +
       std::to_string(
           reinterpret_cast<std::uintptr_t>(&test_binary_loaders)));
  const all_sssp::ArtifactPairId pair{0x1234, 0x5678};
  std::vector<std::filesystem::path> paths;
  const auto cleanup = [&] {
    for (const std::filesystem::path& path : paths) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  };

  try {
    for (std::uint64_t version : {std::uint64_t{3},
                                  std::uint64_t{4},
                                  std::uint64_t{5}}) {
      const std::filesystem::path path =
          base.string() + ".metadata.v" + std::to_string(version);
      paths.push_back(path);
      write_metadata_fixture(path, version, pair);
      const all_sssp::RoutingMetadata metadata =
          all_sssp::load_metadata(path, 0);
      expect(metadata.version == version, "metadata version retained");
      expect(metadata.metadata_node_count == 4,
             "metadata node count parsed");
      expect(metadata.route_requests.size() == 1,
             "metadata request parsed");
      expect(metadata.route_requests[0].sources.size() == 2 &&
                 metadata.route_requests[0].sinks.size() == 2,
             "metadata source/sink vectors parsed");
      expect((version == 5) ==
                 metadata.artifact_pair_id.has_value(),
             "metadata pair-id version gate");
    }

    for (std::uint64_t version :
         {std::uint64_t{1}, std::uint64_t{2}}) {
      const std::filesystem::path path =
          base.string() + ".csr.v" + std::to_string(version);
      paths.push_back(path);
      write_graph_fixture(path, version, pair);
      const all_sssp::HostGraph graph = all_sssp::load_graph(path);
      expect(graph.rows == 3 && graph.nnz == 2,
             "CSR dimensions parsed");
      expect(graph.rowptr ==
                 std::vector<all_sssp::Offset>({0, 1, 2, 2}),
             "CSR row pointers parsed");
      expect((version == 2) == graph.artifact_pair_id.has_value(),
             "CSR pair-id version gate");
    }

    const std::filesystem::path truncated =
        base.string() + ".metadata.truncated";
    paths.push_back(truncated);
    write_metadata_fixture(truncated, 5, pair);
    const std::uintmax_t full_size =
        std::filesystem::file_size(truncated);
    std::filesystem::resize_file(truncated, full_size - 1);
    expect_throws(
        [&] { (void)all_sssp::load_metadata(truncated, 0); },
        "truncated skipped metadata payload rejected");

    const std::filesystem::path trailing =
        base.string() + ".csr.trailing";
    paths.push_back(trailing);
    write_graph_fixture(trailing, 2, pair);
    {
      std::ofstream output(trailing, std::ios::binary | std::ios::app);
      output.put('x');
    }
    expect_throws(
        [&] { (void)all_sssp::load_graph(trailing); },
        "unexpected CSR trailing data rejected");
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

}  // namespace

int main() {
  try {
    test_one_request();
    test_grouping_matrix();
    test_cli();
    test_nonterminal_progress();
    test_artifact_pair_validation();
    test_run_result_validation();
    test_jsonl_schema_and_query_order();
    test_binary_loaders();
    std::cout << "all_SSSP host tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "all_SSSP host test failure: " << error.what() << '\n';
    return 1;
  }
}
