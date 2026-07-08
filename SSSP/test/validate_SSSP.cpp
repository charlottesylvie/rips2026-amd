#include "../src/sssp_validation.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef SSSP_VALIDATOR_DEFAULT_MODES
#define SSSP_VALIDATOR_DEFAULT_MODES "1,2,3"
#endif

namespace {

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <graph.csrbin> <paths.jsonl> [options]\n"
      << "  " << program << " --format\n"
      << "\n"
      << "Validation modes:\n"
      << "  1  path-exists   Every concrete path edge exists in the CSR graph.\n"
      << "  2  path-cost     Sum CSR edge costs along each path and compare to\n"
      << "                   the path record's reported distance.\n"
      << "  3  path-shape    Check source/target endpoints and edge continuity.\n"
      << "\n"
      << "Options:\n"
      << "  --modes <list>          Comma list like 1,2,3 or all. Default: "
      << SSSP_VALIDATOR_DEFAULT_MODES << "\n"
      << "  --sample-rate <p>       Validate only a random fraction of path records.\n"
      << "  --max-paths <n>         Stop selecting path records after n paths.\n"
      << "  --seed <n>              Random sampling seed. Default: 1.\n"
      << "  --progress-every <n>    Print JSONL progress every n lines. Default: 10000.\n"
      << "  --max-errors <n>        Stop after n failures. 0 means no limit. Default: 20.\n"
      << "  --abs-tol <x>           Absolute tolerance. Default: 1e-3.\n"
      << "  --rel-tol <x>           Relative tolerance. Default: 1e-5.\n"
      << "  --strict                Treat suspicious-but-skippable records as failures.\n"
      << "  --format                Print the recommended JSONL schema.\n"
      << "  --help                  Print this message.\n"
      << "\n"
      << "Compile-time knobs:\n"
      << "  -DSSSP_VALIDATOR_DEFAULT_MODES=\"\\\"1,2,3\\\"\"\n";
}

std::uint64_t parse_u64(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::uint64_t>(value);
}

double parse_double(const std::string& text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid number for ") + name);
  }
  return value;
}

std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      print_usage(argv[0]);
      return 2;
    }

    rips_sssp::ValidatorOptions options;
    options.modes = rips_sssp::parse_mode_list(SSSP_VALIDATOR_DEFAULT_MODES);

    std::filesystem::path csr_path;
    std::filesystem::path jsonl_path;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        return 0;
      }
      if (arg == "--format") {
        rips_sssp::print_jsonl_format(std::cout);
        return 0;
      }
      if (arg == "--modes") {
        options.modes = rips_sssp::parse_mode_list(require_value(&i, argc, argv, "--modes"));
      } else if (arg == "--sample-rate") {
        options.sample_rate = parse_double(require_value(&i, argc, argv, "--sample-rate"),
                                           "--sample-rate");
      } else if (arg == "--max-paths") {
        options.max_paths = parse_u64(require_value(&i, argc, argv, "--max-paths"),
                                      "--max-paths");
      } else if (arg == "--seed") {
        options.sample_seed = parse_u64(require_value(&i, argc, argv, "--seed"), "--seed");
      } else if (arg == "--progress-every") {
        options.progress_every = parse_u64(require_value(&i, argc, argv, "--progress-every"),
                                           "--progress-every");
      } else if (arg == "--max-errors") {
        options.max_errors = parse_u64(require_value(&i, argc, argv, "--max-errors"),
                                       "--max-errors");
      } else if (arg == "--abs-tol") {
        options.abs_tolerance = parse_double(require_value(&i, argc, argv, "--abs-tol"),
                                             "--abs-tol");
      } else if (arg == "--rel-tol") {
        options.rel_tolerance = parse_double(require_value(&i, argc, argv, "--rel-tol"),
                                             "--rel-tol");
      } else if (arg == "--strict") {
        options.strict = true;
      } else if (!arg.empty() && arg[0] == '-') {
        throw std::runtime_error("unknown option: " + arg);
      } else if (csr_path.empty()) {
        csr_path = arg;
      } else if (jsonl_path.empty()) {
        jsonl_path = arg;
      } else {
        throw std::runtime_error("too many positional arguments");
      }
    }

    if (csr_path.empty() || jsonl_path.empty()) {
      throw std::runtime_error("expected <graph.csrbin> and <paths.jsonl>");
    }

    std::cout << "[validator] loading CSR graph: " << csr_path.string() << '\n';
    rips_sssp::HostCsrF32 graph = rips_sssp::load_csrbin(csr_path);
    std::cout << "[validator] loaded graph rows=" << graph.rows
              << " cols=" << graph.cols
              << " nnz=" << graph.nnz
              << " orientation=incoming(row v, col u means u -> v)\n";
    std::cout << "[validator] selected modes: "
              << rips_sssp::describe_modes(options.modes) << '\n';

    const rips_sssp::ValidationStats stats =
        rips_sssp::validate_paths_jsonl(graph, jsonl_path, options);

    std::cout << "[summary] jsonl_lines=" << stats.jsonl_lines << '\n';
    std::cout << "[summary] metadata_records=" << stats.metadata_records
              << " path_records=" << stats.path_records << '\n';
    std::cout << "[summary] paths_selected=" << stats.paths_selected
              << " paths_validated=" << stats.paths_validated
              << " paths_skipped_by_sampling=" << stats.paths_skipped_by_sampling << '\n';
    std::cout << "[summary] path_edge_checks=" << stats.path_edge_checks << '\n';
    std::cout << "[summary] warnings=" << stats.warnings
              << " failures=" << stats.failures << '\n';

    if (stats.failures != 0) {
      std::cout << "[validator] FAIL\n";
      return 1;
    }
    std::cout << "[validator] PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[validator] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
