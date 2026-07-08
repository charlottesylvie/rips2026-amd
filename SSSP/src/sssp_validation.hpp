#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace rips_sssp {

using Offset = std::int64_t;
using Index = int;

struct HostCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

enum ValidationModeBits : unsigned {
  kModePathExists = 1u << 0,  // CLI mode 1
  kModePathCost = 1u << 1,    // CLI mode 2
  kModePathShape = 1u << 2,   // CLI mode 3
};

struct ValidatorOptions {
  unsigned modes = kModePathExists | kModePathCost | kModePathShape;
  double abs_tolerance = 1e-3;
  double rel_tolerance = 1e-5;
  double sample_rate = 1.0;
  std::uint64_t sample_seed = 1;
  std::uint64_t max_paths = 0;
  std::uint64_t progress_every = 10000;
  std::uint64_t max_errors = 20;
  bool strict = false;
};

struct ValidationStats {
  std::uint64_t jsonl_lines = 0;
  std::uint64_t metadata_records = 0;
  std::uint64_t path_records = 0;
  std::uint64_t paths_selected = 0;
  std::uint64_t paths_validated = 0;
  std::uint64_t paths_skipped_by_sampling = 0;
  std::uint64_t path_edge_checks = 0;
  std::uint64_t failures = 0;
  std::uint64_t warnings = 0;
};

HostCsrF32 load_csrbin(const std::filesystem::path& path);

unsigned parse_mode_list(const std::string& text);
std::string describe_modes(unsigned modes);

ValidationStats validate_paths_jsonl(const HostCsrF32& graph,
                                     const std::filesystem::path& jsonl_path,
                                     const ValidatorOptions& options);

void print_jsonl_format(std::ostream& out);

}  // namespace rips_sssp
