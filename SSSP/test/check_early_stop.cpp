// Small end-to-end fixtures for bf_outgoing4 target-aware early stopping.
//
// Build from rips2026-amd:
//   g++ -std=c++17 -O2 SSSP/test/check_early_stop.cpp -o check_early_stop
//
// First build bf_outgoing4, then run:
//   ./check_early_stop --exe ./bf_outgoing4

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t CSR_VERSION = 1;
constexpr std::uint64_t METADATA_VERSION = 2;
constexpr std::uint64_t INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();

using Offset = std::int64_t;
using Index = int;

struct Edge {
  int from = -1;
  int to = -1;
  float cost = 1.0f;
};

struct ExpectedPath {
  int target = -1;
  bool reached = true;
  double distance = 0.0;
  std::vector<int> nodes;
};

enum class EarlyStopExpectation {
  No,
  Yes,
  Either,
};

struct TestCase {
  std::string name;
  int node_count = 0;
  int source = -1;
  std::vector<int> sinks;
  std::vector<Edge> edges;
  std::vector<ExpectedPath> expected_paths;
  int expected_iterations = 0;
  EarlyStopExpectation expected_early_stop = EarlyStopExpectation::No;
};

struct PathRecord {
  int net_index = -1;
  int source = -1;
  int target = -1;
  bool reached = false;
  bool distance_is_null = false;
  double distance = std::numeric_limits<double>::infinity();
  std::vector<int> nodes;
};

struct RunSummary {
  bool found = false;
  int iterations = -1;
  bool converged = false;
  bool target_early_stop = false;
  std::string line;
};

struct Options {
  std::filesystem::path executable;
  std::filesystem::path work_dir;
  bool keep_outputs = false;
};

void write_u64(std::ofstream& out, std::uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) {
    throw std::runtime_error("failed while writing uint64");
  }
}

template <typename T>
void write_array(std::ofstream& out, const std::vector<T>& values) {
  if (values.empty()) {
    return;
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) {
    throw std::runtime_error("failed while writing array");
  }
}

void write_string(std::ofstream& out, const std::string& value) {
  write_u64(out, static_cast<std::uint64_t>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out) {
    throw std::runtime_error("failed while writing string");
  }
}

std::string shell_quote(const std::filesystem::path& path) {
  std::string input = path.string();
  std::string quoted = "'";
  for (char ch : input) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

void write_incoming_csrbin(const std::filesystem::path& path,
                           int node_count,
                           const std::vector<Edge>& edges) {
  std::vector<Offset> rowptr(static_cast<std::size_t>(node_count) + 1, 0);
  for (const Edge& edge : edges) {
    if (edge.from < 0 || edge.from >= node_count ||
        edge.to < 0 || edge.to >= node_count) {
      throw std::runtime_error("test edge endpoint is out of range");
    }
    ++rowptr[static_cast<std::size_t>(edge.to) + 1];
  }

  for (int row = 0; row < node_count; ++row) {
    rowptr[static_cast<std::size_t>(row) + 1] += rowptr[static_cast<std::size_t>(row)];
  }

  std::vector<Offset> cursor = rowptr;
  std::vector<Index> colind(edges.size(), 0);
  std::vector<float> values(edges.size(), 0.0f);
  for (const Edge& edge : edges) {
    const Offset slot = cursor[static_cast<std::size_t>(edge.to)]++;
    colind[static_cast<std::size_t>(slot)] = edge.from;
    values[static_cast<std::size_t>(slot)] = edge.cost;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open CSR output: " + path.string());
  }

  out.write(CSR_MAGIC, sizeof(CSR_MAGIC));
  write_u64(out, CSR_VERSION);
  write_u64(out, INCOMING_EDGE_ORIENTATION);
  write_u64(out, static_cast<std::uint64_t>(node_count));
  write_u64(out, static_cast<std::uint64_t>(node_count));
  write_u64(out, static_cast<std::uint64_t>(edges.size()));
  write_u64(out, static_cast<std::uint64_t>(edges.size()));
  write_u64(out, static_cast<std::uint64_t>(edges.size()));
  write_u64(out, static_cast<std::uint64_t>(rowptr.size()));
  write_u64(out, static_cast<std::uint64_t>(colind.size()));
  write_u64(out, static_cast<std::uint64_t>(values.size()));
  write_array(out, rowptr);
  write_array(out, colind);
  write_array(out, values);
}

void write_metadata(const std::filesystem::path& path,
                    const TestCase& test,
                    int logical_net_index) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open metadata output: " + path.string());
  }

  const std::vector<std::string> strings = {test.name};

  out.write(METADATA_MAGIC, sizeof(METADATA_MAGIC));
  write_u64(out, METADATA_VERSION);
  write_u64(out, INCOMING_EDGE_ORIENTATION);
  write_u64(out, static_cast<std::uint64_t>(strings.size()));
  write_u64(out, static_cast<std::uint64_t>(test.node_count));
  write_u64(out, static_cast<std::uint64_t>(test.edges.size()));
  write_u64(out, 0);  // pip data count
  write_u64(out, 0);  // site pin attr count
  write_u64(out, 1);  // route request count
  write_u64(out, 0);  // blocked node count
  write_u64(out, 0);  // sink stop node count
  write_u64(out, 0);  // logical cell count
  write_u64(out, 0);  // logical net count
  write_u64(out, 0);  // logical port instance count
  write_u64(out, 0);  // physical netlist byte count
  write_u64(out, 0);  // logical netlist byte count
  write_u64(out, kNoIndex);
  write_u64(out, kNoIndex);
  write_u64(out, kNoIndex);
  write_u64(out, kNoIndex);

  for (const std::string& value : strings) {
    write_string(out, value);
  }

  std::vector<std::uint64_t> node_device_ids(static_cast<std::size_t>(test.node_count));
  for (int node = 0; node < test.node_count; ++node) {
    node_device_ids[static_cast<std::size_t>(node)] = static_cast<std::uint64_t>(node);
  }
  write_array(out, node_device_ids);

  for (std::size_t i = 0; i < test.edges.size(); ++i) {
    write_u64(out, kNoIndex);  // edge tile string
    write_u64(out, kNoIndex);  // pip data index
  }

  write_u64(out, 0);  // route net string index
  write_u64(out, static_cast<std::uint64_t>(logical_net_index));

  write_u64(out, 1);  // source count
  write_u64(out, static_cast<std::uint64_t>(test.source));
  write_u64(out, kNoIndex);
  write_u64(out, kNoIndex);

  write_u64(out, static_cast<std::uint64_t>(test.sinks.size()));
  for (int sink : test.sinks) {
    write_u64(out, static_cast<std::uint64_t>(sink));
    write_u64(out, kNoIndex);
    write_u64(out, kNoIndex);
  }
}

int parse_int_after(const std::string& text, const std::string& key) {
  const std::size_t pos = text.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing integer field: " + key);
  }
  std::size_t i = pos + key.size();
  bool negative = false;
  if (i < text.size() && text[i] == '-') {
    negative = true;
    ++i;
  }
  if (i >= text.size() || text[i] < '0' || text[i] > '9') {
    throw std::runtime_error("malformed integer field: " + key);
  }
  int value = 0;
  while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
    value = value * 10 + (text[i] - '0');
    ++i;
  }
  return negative ? -value : value;
}

double parse_double_after(const std::string& text, const std::string& key) {
  const std::size_t pos = text.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing double field: " + key);
  }
  const char* start = text.c_str() + pos + key.size();
  char* end = nullptr;
  const double value = std::strtod(start, &end);
  if (end == start) {
    throw std::runtime_error("malformed double field: " + key);
  }
  return value;
}

std::vector<int> parse_int_array_after(const std::string& text,
                                       const std::string& key) {
  const std::size_t pos = text.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing array field: " + key);
  }
  std::size_t i = pos + key.size();
  const std::size_t end = text.find(']', i);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated array field: " + key);
  }

  std::vector<int> values;
  while (i < end) {
    while (i < end && (text[i] == ' ' || text[i] == ',')) {
      ++i;
    }
    if (i >= end) {
      break;
    }
    bool negative = false;
    if (text[i] == '-') {
      negative = true;
      ++i;
    }
    if (i >= end || text[i] < '0' || text[i] > '9') {
      throw std::runtime_error("malformed integer array field: " + key);
    }
    int value = 0;
    while (i < end && text[i] >= '0' && text[i] <= '9') {
      value = value * 10 + (text[i] - '0');
      ++i;
    }
    values.push_back(negative ? -value : value);
  }
  return values;
}

bool parse_bool_after(const std::string& text, const std::string& key) {
  const std::size_t pos = text.find(key);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing bool field: " + key);
  }
  const std::size_t value_pos = pos + key.size();
  if (text.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (text.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  throw std::runtime_error("malformed bool field: " + key);
}

PathRecord parse_path_record(const std::string& line) {
  PathRecord record;
  record.net_index = parse_int_after(line, "\"net_index\":");
  record.source = parse_int_after(line, "\"source\":");
  record.target = parse_int_after(line, "\"target\":");
  record.reached = parse_bool_after(line, "\"reached\":");
  record.distance_is_null = line.find("\"distance\":null") != std::string::npos;
  if (!record.distance_is_null) {
    record.distance = parse_double_after(line, "\"distance\":");
  }
  record.nodes = parse_int_array_after(line, "\"nodes\":[");
  return record;
}

std::vector<PathRecord> read_path_records(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("could not open JSONL output: " + path.string());
  }

  std::vector<PathRecord> records;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("\"type\":\"path\"") == std::string::npos) {
      continue;
    }
    records.push_back(parse_path_record(line));
  }
  return records;
}

RunSummary read_run_summary(const std::filesystem::path& log_path) {
  std::ifstream in(log_path);
  if (!in) {
    throw std::runtime_error("could not open run log: " + log_path.string());
  }

  RunSummary summary;
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t start = line.rfind("[bf_outgoing4] source ");
    if (start == std::string::npos ||
        line.find("iterations_used=", start) == std::string::npos) {
      continue;
    }
    summary.found = true;
    summary.line = line.substr(start);
    summary.iterations = parse_int_after(summary.line, "iterations_used=");
    summary.converged =
        summary.line.find("converged=yes") != std::string::npos;
    summary.target_early_stop =
        summary.line.find("target_early_stop=yes") != std::string::npos;
  }
  return summary;
}

const PathRecord* find_record(const std::vector<PathRecord>& records,
                              int source,
                              int target) {
  for (const PathRecord& record : records) {
    if (record.source == source && record.target == target) {
      return &record;
    }
  }
  return nullptr;
}

std::string join_nodes(const std::vector<int>& nodes) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << nodes[i];
  }
  out << ']';
  return out.str();
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_expected_path(const TestCase& test,
                         const ExpectedPath& expected,
                         const std::vector<PathRecord>& records) {
  const PathRecord* record = find_record(records, test.source, expected.target);
  require(record != nullptr,
          test.name + ": missing path record for target " +
              std::to_string(expected.target));
  require(record->reached == expected.reached,
          test.name + ": reached mismatch for target " +
              std::to_string(expected.target));

  if (!expected.reached) {
    require(record->distance_is_null,
            test.name + ": unreachable target should have distance=null");
    require(record->nodes.empty(),
            test.name + ": unreachable target should have an empty node path");
    return;
  }

  require(!record->distance_is_null,
          test.name + ": reached target should have a numeric distance");
  require(std::fabs(record->distance - expected.distance) <= 1e-5,
          test.name + ": distance mismatch for target " +
              std::to_string(expected.target) + ", got " +
              std::to_string(record->distance) + ", expected " +
              std::to_string(expected.distance));
  require(record->nodes == expected.nodes,
          test.name + ": node path mismatch for target " +
              std::to_string(expected.target) + ", got " +
              join_nodes(record->nodes) + ", expected " +
              join_nodes(expected.nodes));
}

void check_early_stop_expectation(const TestCase& test,
                                  const RunSummary& summary) {
  if (test.expected_early_stop == EarlyStopExpectation::Either) {
    return;
  }
  const bool expected =
      test.expected_early_stop == EarlyStopExpectation::Yes;
  require(summary.target_early_stop == expected,
          test.name + ": target_early_stop mismatch in summary line: " +
              summary.line);
}

std::vector<TestCase> make_tests() {
  return {
      {
          "chat_example_relaxation",
          4,
          0,
          {2},
          {
              {0, 1, 1.0f},
              {0, 2, 4.0f},
              {1, 2, 1.0f},
              {1, 3, 5.0f},
              {2, 3, 1.0f},
          },
          {{2, true, 2.0, {0, 1, 2}}},
          3,
          EarlyStopExpectation::Either,
      },
      {
          "chat_example_target_early_stop",
          8,
          0,
          {3},
          {
              {0, 1, 1.0f},
              {1, 2, 1.0f},
              {2, 3, 1.0f},
              {0, 4, 100.0f},
              {4, 5, 1.0f},
              {5, 6, 1.0f},
              {6, 7, 1.0f},
          },
          {{3, true, 3.0, {0, 1, 2, 3}}},
          4,
          EarlyStopExpectation::Yes,
      },
      {
          "two_sinks_one_source",
          8,
          0,
          {2, 3},
          {
              {0, 1, 1.0f},
              {1, 2, 1.0f},
              {2, 3, 1.0f},
              {0, 4, 50.0f},
              {4, 5, 1.0f},
              {5, 6, 1.0f},
              {6, 7, 1.0f},
          },
          {
              {2, true, 2.0, {0, 1, 2}},
              {3, true, 3.0, {0, 1, 2, 3}},
          },
          4,
          EarlyStopExpectation::Yes,
      },
      {
          "unreachable_sink_full_convergence",
          5,
          0,
          {4},
          {
              {0, 1, 1.0f},
              {1, 2, 1.0f},
              {3, 4, 1.0f},
          },
          {{4, false, 0.0, {}}},
          3,
          EarlyStopExpectation::No,
      },
      {
          "near_sink_expensive_branch",
          5,
          0,
          {1},
          {
              {0, 1, 1.0f},
              {0, 2, 20.0f},
              {2, 3, 1.0f},
              {3, 4, 1.0f},
          },
          {{1, true, 1.0, {0, 1}}},
          2,
          EarlyStopExpectation::Yes,
      },
  };
}

Options parse_options(int argc, char** argv) {
  Options options;
  if (const char* env_exe = std::getenv("BF_OUTGOING4_EXE")) {
    options.executable = env_exe;
  } else {
    options.executable = "./bf_outgoing4";
  }

  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path default_work_dir =
      std::filesystem::temp_directory_path() /
      ("rips_check_early_stop_" + std::to_string(now));
  options.work_dir = default_work_dir;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--exe") {
      if (++i >= argc) {
        throw std::runtime_error("--exe requires a path");
      }
      options.executable = argv[i];
    } else if (arg == "--work-dir") {
      if (++i >= argc) {
        throw std::runtime_error("--work-dir requires a path");
      }
      options.work_dir = argv[i];
    } else if (arg == "--keep") {
      options.keep_outputs = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: check_early_stop [--exe ./bf_outgoing4] "
          << "[--work-dir DIR] [--keep]\n";
      std::exit(0);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      options.executable = arg;
    }
  }

  return options;
}

void run_one_test(const Options& options,
                  const TestCase& test,
                  int test_index) {
  const std::filesystem::path case_dir =
      options.work_dir / (std::to_string(test_index) + "_" + test.name);
  std::filesystem::create_directories(case_dir);

  const std::filesystem::path csr_path = case_dir / "graph.csrbin";
  const std::filesystem::path metadata_path = case_dir / "graph.csrbin.ifmeta.bin";
  const std::filesystem::path output_path = case_dir / "paths.jsonl";
  const std::filesystem::path log_path = case_dir / "bf_outgoing4.log";

  write_incoming_csrbin(csr_path, test.node_count, test.edges);
  write_metadata(metadata_path, test, test_index);

  const std::string command =
      shell_quote(options.executable) + " " +
      shell_quote(csr_path) + " " +
      shell_quote(metadata_path) + " " +
      shell_quote(output_path) +
      " --source-progress-every 1 > " +
      shell_quote(log_path) + " 2>&1";

  const int exit_code = std::system(command.c_str());
  require(exit_code == 0,
          test.name + ": bf_outgoing4 failed; see " + log_path.string());

  const std::vector<PathRecord> records = read_path_records(output_path);
  require(records.size() == test.expected_paths.size(),
          test.name + ": unexpected number of path records, got " +
              std::to_string(records.size()) + ", expected " +
              std::to_string(test.expected_paths.size()));

  for (const ExpectedPath& expected : test.expected_paths) {
    check_expected_path(test, expected, records);
  }

  const RunSummary summary = read_run_summary(log_path);
  require(summary.found, test.name + ": missing per-source summary in log");
  require(summary.iterations == test.expected_iterations,
          test.name + ": iterations_used mismatch, got " +
              std::to_string(summary.iterations) + ", expected " +
              std::to_string(test.expected_iterations) +
              "; summary line: " + summary.line);
  check_early_stop_expectation(test, summary);

  std::cout << "[PASS] " << test.name
            << " iterations=" << summary.iterations
            << " target_early_stop="
            << (summary.target_early_stop ? "yes" : "no")
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    if (!std::filesystem::exists(options.executable)) {
      throw std::runtime_error(
          "bf_outgoing4 executable not found: " + options.executable.string() +
          "\nBuild it first, or pass --exe /path/to/bf_outgoing4");
    }

    std::filesystem::create_directories(options.work_dir);
    const std::vector<TestCase> tests = make_tests();

    std::cout << "[check_early_stop] executable="
              << options.executable.string() << '\n'
              << "[check_early_stop] work_dir="
              << options.work_dir.string() << '\n';

    for (std::size_t i = 0; i < tests.size(); ++i) {
      run_one_test(options, tests[i], static_cast<int>(i));
    }

    if (!options.keep_outputs) {
      std::filesystem::remove_all(options.work_dir);
    } else {
      std::cout << "[check_early_stop] kept outputs in "
                << options.work_dir.string() << '\n';
    }

    std::cout << "[check_early_stop] all tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[check_early_stop] ERROR: " << error.what() << '\n';
    return 1;
  }
}
