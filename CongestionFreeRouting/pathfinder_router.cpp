// Benchmark-compatible C++ router wrapper for the RIPS PathFinder flow.
//
// This executable is the file the contest Makefile should time:
//   ./PathFinderFile <benchmark>_unrouted.phys <benchmark>_PathFinderFile.phys
//
// It keeps the benchmark-facing interface to exactly two positional arguments,
// then orchestrates the existing C++ tools:
//   interchange_to_csr -> pathfinder -> routes_to_phys
//
// Example compile command:
//   g++ -std=c++17 -O2 CongestionFreeRouting/pathfinder_router.cpp -o PathFinderFile

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::filesystem::path input_phys;
  std::filesystem::path output_phys;
  std::filesystem::path logical_netlist;
  std::filesystem::path device = "xcvu3p.device";
  std::filesystem::path work_dir;
  bool work_dir_was_provided = false;
  bool keep_work_dir = false;

  std::string interchange_to_csr = "./interchange_to_csr";
  std::string pathfinder = "./pathfinder";
  std::string routes_to_phys = "./routes_to_phys";

  std::vector<std::string> converter_args;
  std::vector<std::string> pathfinder_args;
  bool allow_unrouted = true;
  bool converter_bounds_set = false;
};

std::string env_or_default(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  if (value != nullptr && value[0] != '\0') {
    return value;
  }
  return fallback;
}

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::filesystem::path infer_logical_netlist(const std::filesystem::path& phys) {
  const std::string name = phys.filename().string();
  const std::string suffix = "_unrouted.phys";
  if (ends_with(name, suffix)) {
    return phys.parent_path() / (name.substr(0, name.size() - suffix.size()) + ".netlist");
  }
  if (phys.extension() == ".phys") {
    std::filesystem::path path = phys;
    path.replace_extension(".netlist");
    return path;
  }
  throw std::runtime_error("could not infer logical netlist; pass --logical-netlist");
}

std::string shell_quote(const std::string& text) {
  std::string out = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

std::string command_to_string(const std::vector<std::string>& argv) {
  std::ostringstream out;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    if (i != 0) out << ' ';
    out << shell_quote(argv[i]);
  }
  return out.str();
}

void print_progress(int completed, int total, const std::string& label) {
  constexpr int kWidth = 28;
  const int filled = total == 0 ? kWidth : (completed * kWidth) / total;
  std::cout << "[pathfinder-router] [";
  for (int i = 0; i < kWidth; ++i) {
    std::cout << (i < filled ? '#' : '-');
  }
  std::cout << "] " << completed << "/" << total << " " << label << "\n" << std::flush;
}

void run_command(const std::vector<std::string>& argv, const char* label) {
  const std::string command = command_to_string(argv);
  const int status = std::system(command.c_str());
  if (status != 0) {
    std::ostringstream out;
    out << label << " failed with status " << status
        << " while running " << command;
    throw std::runtime_error(out.str());
  }
}

std::filesystem::path make_work_dir(const Options& options) {
  if (!options.work_dir.empty()) {
    return options.work_dir;
  }

  std::filesystem::path base = options.output_phys;
  base += ".pathfinder-work";
  for (int attempt = 0; attempt < 10000; ++attempt) {
    std::filesystem::path candidate = base;
    if (attempt != 0) {
      candidate += "." + std::to_string(attempt);
    }
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("could not allocate a unique work directory");
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <input_unrouted.phys> <output_routed.phys> [options]\n\n"
      << "Options:\n"
      << "  --logical-netlist <path>       Override inferred .netlist path.\n"
      << "  --device <path>                DeviceResources input. Default: xcvu3p.device\n"
      << "  --work-dir <path>              Directory for temporary CSR/metadata/routes.\n"
      << "  --keep-work-dir                Do not remove temporary files.\n"
      << "  --interchange-to-csr <path>    Converter executable. Env: INTERCHANGE_TO_CSR\n"
      << "  --pathfinder <path>            PathFinder executable. Env: PATHFINDER_BIN\n"
      << "  --routes-to-phys <path>        Route reconstructor. Env: ROUTES_TO_PHYS\n"
      << "  --strict-routing               Fail instead of writing partial routes.\n"
      << "  --sssp-engine <unit-bfs|delta-step>\n"
      << "                                 Forwarded to pathfinder. Default: unit-bfs\n"
      << "  --use-delta-step               Forwarded to pathfinder for comparison.\n"
      << "  --delta <float>                Forwarded to pathfinder.\n"
      << "  --max-sssp-iters <int>         Forwarded to pathfinder.\n"
      << "  --net-limit <count>            Forwarded to pathfinder.\n"
      << "  --parallel-net-workers <count> Forwarded to pathfinder; 0 enables memory-aware auto-selection.\n"
      << "  --capacity <int>               Forwarded for overuse diagnostics.\n"
      << "  --max-pathfinder-iters <int>   Compatibility-only; accepted by pathfinder and ignored.\n"
      << "  --route-batch-size <count>     Compatibility-only; accepted by pathfinder and ignored.\n"
      << "  --present-factor <float>       Compatibility-only; accepted by pathfinder and ignored.\n"
      << "  --present-multiplier <float>   Compatibility-only; accepted by pathfinder and ignored.\n"
      << "  --history-factor <float>       Compatibility-only; accepted by pathfinder and ignored.\n"
      << "  --full-device                  Import the whole device instead of nxroute bounds.\n"
      << "  --nxroute-bounds               Import X36..X90, Y60..Y239. Default for fair nxroute-poc comparison.\n"
      << "  --bounds <minX> <maxX> <minY> <maxY>\n"
      << "                                 Forwarded to interchange_to_csr.\n"
      << "  --node-bounds-mode <mode>      Forwarded to interchange_to_csr.\n";
}

Options parse_args(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "-h" ||
                    std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    std::exit(0);
  }
  if (argc < 3) {
    print_usage(argv[0]);
    throw std::runtime_error("expected input and output .phys paths");
  }

  Options options;
  options.input_phys = argv[1];
  options.output_phys = argv[2];
  options.interchange_to_csr = env_or_default("INTERCHANGE_TO_CSR", "./interchange_to_csr");
  options.pathfinder = env_or_default("PATHFINDER_BIN", "./pathfinder");
  options.routes_to_phys = env_or_default("ROUTES_TO_PHYS", "./routes_to_phys");

  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[++i];
    };

    if (option == "--logical-netlist") {
      options.logical_netlist = require_value("--logical-netlist");
    } else if (option == "--device") {
      options.device = require_value("--device");
    } else if (option == "--work-dir") {
      options.work_dir = require_value("--work-dir");
      options.work_dir_was_provided = true;
    } else if (option == "--keep-work-dir") {
      options.keep_work_dir = true;
    } else if (option == "--interchange-to-csr") {
      options.interchange_to_csr = require_value("--interchange-to-csr");
    } else if (option == "--pathfinder") {
      options.pathfinder = require_value("--pathfinder");
    } else if (option == "--routes-to-phys") {
      options.routes_to_phys = require_value("--routes-to-phys");
    } else if (option == "--strict-routing") {
      options.allow_unrouted = false;
    } else if (option == "--use-delta-step") {
      options.pathfinder_args.push_back(option);
    } else if (option == "--sssp-engine" ||
               option == "--delta" ||
               option == "--max-pathfinder-iters" ||
               option == "--max-sssp-iters" ||
               option == "--net-limit" ||
               option == "--route-batch-size" ||
               option == "--parallel-net-workers" ||
               option == "--capacity" ||
               option == "--present-factor" ||
               option == "--present-multiplier" ||
               option == "--history-factor") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_value(option.c_str()));
    } else if (option == "--full-device") {
      options.converter_bounds_set = true;
      options.converter_args.push_back(option);
    } else if (option == "--nxroute-bounds") {
      options.converter_bounds_set = true;
      options.converter_args.push_back(option);
    } else if (option == "--bounds") {
      options.converter_bounds_set = true;
      options.converter_args.push_back(option);
      for (int j = 0; j < 4; ++j) {
        options.converter_args.push_back(require_value("--bounds"));
      }
    } else if (option == "--node-bounds-mode") {
      options.converter_args.push_back(option);
      options.converter_args.push_back(require_value("--node-bounds-mode"));
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }

  if (options.logical_netlist.empty()) {
    options.logical_netlist = infer_logical_netlist(options.input_phys);
  }
  return options;
}

void require_file(const std::filesystem::path& path, const char* label) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::string("missing ") + label + ": " + path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path work_dir;
  bool cleanup_work_dir = false;
  try {
    Options options = parse_args(argc, argv);
    require_file(options.input_phys, "input physical netlist");
    require_file(options.logical_netlist, "logical netlist");
    require_file(options.device, "device resources");

    work_dir = make_work_dir(options);
    std::filesystem::create_directories(work_dir);
    cleanup_work_dir = !options.keep_work_dir && !options.work_dir_was_provided;

    const std::filesystem::path csr_path =
        work_dir / (options.output_phys.stem().string() + ".csrbin");
    const std::filesystem::path metadata_path =
        work_dir / (options.output_phys.stem().string() + ".csrbin.ifmeta.bin");
    const std::filesystem::path routes_path =
        work_dir / (options.output_phys.stem().string() + ".routes.jsonl");

    print_progress(0, 3, "starting");

    std::vector<std::string> convert_cmd = {
        options.interchange_to_csr,
        options.input_phys.string(),
        options.logical_netlist.string(),
        csr_path.string(),
        "--device",
        options.device.string(),
        "--metadata",
        metadata_path.string(),
    };
    if (!options.converter_bounds_set) {
      convert_cmd.push_back("--nxroute-bounds");
    }
    convert_cmd.insert(convert_cmd.end(),
                       options.converter_args.begin(),
                       options.converter_args.end());
    run_command(convert_cmd, "convert FPGAIF to CSR");
    print_progress(1, 3, "CSR conversion complete");

    std::vector<std::string> pathfinder_cmd = {
        options.pathfinder,
        csr_path.string(),
        metadata_path.string(),
        "--routes-out",
        routes_path.string(),
    };
    if (options.allow_unrouted) {
      pathfinder_cmd.push_back("--allow-unrouted");
    }
    pathfinder_cmd.insert(pathfinder_cmd.end(),
                          options.pathfinder_args.begin(),
                          options.pathfinder_args.end());
    run_command(pathfinder_cmd, "run PathFinder");
    print_progress(2, 3, "PathFinder complete");

    std::vector<std::string> reconstruct_cmd = {
        options.routes_to_phys,
        options.input_phys.string(),
        metadata_path.string(),
        routes_path.string(),
        options.output_phys.string(),
    };
    if (options.allow_unrouted) {
      reconstruct_cmd.push_back("--allow-unrouted-stubs");
    }
    run_command(reconstruct_cmd, "reconstruct routed PhysicalNetlist");
    print_progress(3, 3, "routed PhysicalNetlist written");

    if (cleanup_work_dir) {
      std::filesystem::remove_all(work_dir);
    }
    return 0;
  } catch (const std::exception& ex) {
    if (cleanup_work_dir && !work_dir.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(work_dir, ignored);
    }
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
