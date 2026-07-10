#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Options {
  bool exact = false;
  bool compare_path_shape = true;
  double abs_tolerance = 1e-3;
  double rel_tolerance = 1e-5;
  std::uint64_t max_diffs = 20;
};

struct PathRecord {
  std::uint64_t line = 0;
  std::string raw_line;
  std::string key;
  std::string net;
  long long net_index = -1;
  int source = -1;
  int target = -1;
  bool reached = false;
  bool has_distance = false;
  double distance = 0.0;
  std::string nodes_raw;
  std::string csr_edges_raw;
};

struct JsonlIndex {
  std::uint64_t lines = 0;
  std::uint64_t metadata_records = 0;
  std::uint64_t path_records = 0;
  std::unordered_map<std::string, std::vector<PathRecord>> paths_by_key;
};

std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

double parse_double_arg(const std::string& text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid number for ") + name);
  }
  return value;
}

std::uint64_t parse_u64_arg(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::uint64_t>(value);
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <expected.paths.jsonl> <actual.paths.jsonl> [options]\n\n"
      << "Default mode compares path records semantically by key:\n"
      << "  (net_index, source, target), reached, distance, nodes, csr_edges.\n\n"
      << "Options:\n"
      << "  --exact              Compare files byte-for-byte by line, including metadata.\n"
      << "  --distance-only      In semantic mode, compare only reached/unreached and distance.\n"
      << "  --abs-tol <x>        Distance absolute tolerance. Default: 1e-3.\n"
      << "  --rel-tol <x>        Distance relative tolerance. Default: 1e-5.\n"
      << "  --max-diffs <n>      Stop printing after n differences. Default: 20.\n"
      << "  --help               Print this message.\n";
}

std::string trim(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::string truncate_for_print(const std::string& text, std::size_t limit = 220) {
  if (text.size() <= limit) return text;
  return text.substr(0, limit) + "...";
}

std::string parse_json_string_token(const std::string& text, std::size_t* pos) {
  if (*pos >= text.size() || text[*pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  std::string out;
  ++(*pos);
  while (*pos < text.size()) {
    const char c = text[(*pos)++];
    if (c == '"') return out;
    if (c == '\\') {
      if (*pos >= text.size()) throw std::runtime_error("unterminated JSON escape");
      const char esc = text[(*pos)++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          // Keys and fields we compare are ASCII. Preserve unicode escapes compactly.
          out += "\\u";
          for (int i = 0; i < 4 && *pos < text.size(); ++i) {
            out.push_back(text[(*pos)++]);
          }
          break;
        default:
          out.push_back(esc);
          break;
      }
    } else {
      out.push_back(c);
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

void skip_ws(const std::string& text, std::size_t* pos) {
  while (*pos < text.size()) {
    const char c = text[*pos];
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') return;
    ++(*pos);
  }
}

std::size_t end_of_json_value(const std::string& text, std::size_t pos) {
  skip_ws(text, &pos);
  if (pos >= text.size()) throw std::runtime_error("missing JSON value");

  if (text[pos] == '"') {
    ++pos;
    bool escape = false;
    for (; pos < text.size(); ++pos) {
      const char c = text[pos];
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        return pos + 1;
      }
    }
    throw std::runtime_error("unterminated JSON string value");
  }

  if (text[pos] == '[' || text[pos] == '{') {
    const char open = text[pos];
    const char close = open == '[' ? ']' : '}';
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (; pos < text.size(); ++pos) {
      const char c = text[pos];
      if (in_string) {
        if (escape) {
          escape = false;
        } else if (c == '\\') {
          escape = true;
        } else if (c == '"') {
          in_string = false;
        }
        continue;
      }
      if (c == '"') {
        in_string = true;
      } else if (c == open) {
        ++depth;
      } else if (c == close) {
        --depth;
        if (depth == 0) return pos + 1;
      }
    }
    throw std::runtime_error("unterminated JSON array/object value");
  }

  while (pos < text.size() && text[pos] != ',' && text[pos] != '}') {
    ++pos;
  }
  return pos;
}

std::map<std::string, std::string> parse_top_level_object(const std::string& line) {
  std::size_t pos = 0;
  skip_ws(line, &pos);
  if (pos >= line.size() || line[pos] != '{') {
    throw std::runtime_error("JSONL line is not an object");
  }
  ++pos;

  std::map<std::string, std::string> fields;
  while (true) {
    skip_ws(line, &pos);
    if (pos < line.size() && line[pos] == '}') {
      ++pos;
      skip_ws(line, &pos);
      if (pos != line.size()) throw std::runtime_error("trailing characters after object");
      return fields;
    }

    const std::string key = parse_json_string_token(line, &pos);
    skip_ws(line, &pos);
    if (pos >= line.size() || line[pos] != ':') {
      throw std::runtime_error("expected ':' after object key");
    }
    ++pos;
    const std::size_t value_begin = pos;
    const std::size_t value_end = end_of_json_value(line, pos);
    fields[key] = trim(line.substr(value_begin, value_end - value_begin));
    pos = value_end;

    skip_ws(line, &pos);
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < line.size() && line[pos] == '}') {
      continue;
    }
    throw std::runtime_error("expected ',' or '}' after object value");
  }
}

const std::string& required_field(const std::map<std::string, std::string>& fields,
                                  const char* name) {
  const auto it = fields.find(name);
  if (it == fields.end()) {
    throw std::runtime_error(std::string("missing required field: ") + name);
  }
  return it->second;
}

std::string parse_string_field(const std::map<std::string, std::string>& fields,
                               const char* name) {
  const std::string& raw = required_field(fields, name);
  std::size_t pos = 0;
  return parse_json_string_token(raw, &pos);
}

long long parse_integer_field(const std::map<std::string, std::string>& fields,
                              const char* name) {
  const std::string& raw = required_field(fields, name);
  char* end = nullptr;
  const long long value = std::strtoll(raw.c_str(), &end, 10);
  if (end == raw.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid integer field: ") + name);
  }
  return value;
}

bool parse_bool_field(const std::map<std::string, std::string>& fields,
                      const char* name) {
  const std::string& raw = required_field(fields, name);
  if (raw == "true") return true;
  if (raw == "false") return false;
  throw std::runtime_error(std::string("invalid bool field: ") + name);
}

double parse_nullable_distance(const std::map<std::string, std::string>& fields,
                               bool* has_distance) {
  const std::string& raw = required_field(fields, "distance");
  if (raw == "null") {
    *has_distance = false;
    return 0.0;
  }
  char* end = nullptr;
  const double value = std::strtod(raw.c_str(), &end);
  if (end == raw.c_str() || *end != '\0') {
    throw std::runtime_error("invalid distance field");
  }
  *has_distance = true;
  return value;
}

std::string make_key(long long net_index, int source, int target) {
  std::ostringstream out;
  out << net_index << ':' << source << ':' << target;
  return out.str();
}

PathRecord parse_path_record(const std::string& line, std::uint64_t line_no) {
  const auto fields = parse_top_level_object(line);
  PathRecord record;
  record.line = line_no;
  record.raw_line = line;
  record.net = parse_string_field(fields, "net");
  record.net_index = parse_integer_field(fields, "net_index");
  record.source = static_cast<int>(parse_integer_field(fields, "source"));
  record.target = static_cast<int>(parse_integer_field(fields, "target"));
  record.reached = parse_bool_field(fields, "reached");
  record.distance = parse_nullable_distance(fields, &record.has_distance);
  record.nodes_raw = required_field(fields, "nodes");
  record.csr_edges_raw = required_field(fields, "csr_edges");
  record.key = make_key(record.net_index, record.source, record.target);
  return record;
}

bool close_enough(double a, double b, const Options& options) {
  if (!std::isfinite(a) || !std::isfinite(b)) return a == b;
  const double diff = std::fabs(a - b);
  const double scale = std::max(std::fabs(a), std::fabs(b));
  return diff <= options.abs_tolerance ||
         diff <= options.rel_tolerance * std::max(1.0, scale);
}

JsonlIndex load_jsonl_index(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("could not open JSONL file: " + path.string());

  JsonlIndex index;
  std::string line;
  while (std::getline(in, line)) {
    ++index.lines;
    if (trim(line).empty()) continue;
    try {
      const auto fields = parse_top_level_object(line);
      const auto type_it = fields.find("type");
      if (type_it == fields.end()) continue;
      std::size_t type_pos = 0;
      const std::string type = parse_json_string_token(type_it->second, &type_pos);
      if (type == "metadata") {
        ++index.metadata_records;
      } else if (type == "path") {
        ++index.path_records;
        PathRecord record = parse_path_record(line, index.lines);
        index.paths_by_key[record.key].push_back(std::move(record));
      }
    } catch (const std::exception& ex) {
      std::ostringstream msg;
      msg << path.string() << ':' << index.lines << ": " << ex.what();
      throw std::runtime_error(msg.str());
    }
  }
  return index;
}

bool exact_compare(const std::filesystem::path& expected,
                   const std::filesystem::path& actual) {
  std::ifstream a(expected);
  std::ifstream b(actual);
  if (!a) throw std::runtime_error("could not open file: " + expected.string());
  if (!b) throw std::runtime_error("could not open file: " + actual.string());

  std::string line_a;
  std::string line_b;
  std::uint64_t line = 0;
  while (true) {
    const bool has_a = static_cast<bool>(std::getline(a, line_a));
    const bool has_b = static_cast<bool>(std::getline(b, line_b));
    if (!has_a && !has_b) {
      std::cout << "[compare] EXACT MATCH\n";
      return true;
    }
    ++line;
    if (has_a != has_b) {
      std::cout << "[compare] EXACT MISMATCH: file lengths differ at line "
                << line << '\n';
      if (has_a) std::cout << "  expected: " << truncate_for_print(line_a) << '\n';
      if (has_b) std::cout << "  actual:   " << truncate_for_print(line_b) << '\n';
      return false;
    }
    if (line_a != line_b) {
      std::cout << "[compare] EXACT MISMATCH at line " << line << '\n'
                << "  expected: " << truncate_for_print(line_a) << '\n'
                << "  actual:   " << truncate_for_print(line_b) << '\n';
      return false;
    }
  }
}

void print_record_brief(const char* label, const PathRecord& record) {
  std::cout << "  " << label << " line=" << record.line
            << " key=" << record.key
            << " reached=" << (record.reached ? "true" : "false");
  if (record.has_distance) {
    std::cout << " distance=" << std::setprecision(12) << record.distance;
  } else {
    std::cout << " distance=null";
  }
  std::cout << " nodes=" << truncate_for_print(record.nodes_raw, 80)
            << " csr_edges=" << truncate_for_print(record.csr_edges_raw, 80)
            << '\n';
}

bool semantic_compare(const std::filesystem::path& expected_path,
                      const std::filesystem::path& actual_path,
                      const Options& options) {
  const JsonlIndex expected = load_jsonl_index(expected_path);
  const JsonlIndex actual = load_jsonl_index(actual_path);

  std::cout << "[compare] expected lines=" << expected.lines
            << " metadata=" << expected.metadata_records
            << " paths=" << expected.path_records << '\n';
  std::cout << "[compare] actual   lines=" << actual.lines
            << " metadata=" << actual.metadata_records
            << " paths=" << actual.path_records << '\n';

  std::uint64_t differences = 0;
  auto report = [&](const std::string& message) {
    if (differences < options.max_diffs) {
      std::cout << "[diff] " << message << '\n';
    }
    ++differences;
  };

  for (const auto& [key, expected_records] : expected.paths_by_key) {
    const auto actual_it = actual.paths_by_key.find(key);
    if (actual_it == actual.paths_by_key.end()) {
      report("missing path key in actual: " + key);
      if (differences <= options.max_diffs) {
        print_record_brief("expected", expected_records.front());
      }
      continue;
    }

    const std::vector<PathRecord>& actual_records = actual_it->second;
    if (expected_records.size() != actual_records.size()) {
      std::ostringstream msg;
      msg << "record count mismatch for key " << key
          << ": expected " << expected_records.size()
          << ", actual " << actual_records.size();
      report(msg.str());
    }

    const std::size_t count = std::min(expected_records.size(), actual_records.size());
    for (std::size_t i = 0; i < count; ++i) {
      const PathRecord& expected_record = expected_records[i];
      const PathRecord& actual_record = actual_records[i];

      bool mismatch = false;
      std::ostringstream reason;
      if (expected_record.reached != actual_record.reached) {
        mismatch = true;
        reason << "reached mismatch";
      } else if (expected_record.has_distance != actual_record.has_distance) {
        mismatch = true;
        reason << "distance null/non-null mismatch";
      } else if (expected_record.has_distance &&
                 !close_enough(expected_record.distance,
                               actual_record.distance,
                               options)) {
        mismatch = true;
        reason << "distance mismatch: expected "
               << std::setprecision(12) << expected_record.distance
               << ", actual " << actual_record.distance;
      } else if (options.compare_path_shape &&
                 expected_record.nodes_raw != actual_record.nodes_raw) {
        mismatch = true;
        reason << "nodes mismatch";
      } else if (options.compare_path_shape &&
                 expected_record.csr_edges_raw != actual_record.csr_edges_raw) {
        mismatch = true;
        reason << "csr_edges mismatch";
      }

      if (mismatch) {
        report("path divergence for key " + key + " occurrence " +
               std::to_string(i) + ": " + reason.str());
        if (differences <= options.max_diffs) {
          print_record_brief("expected", expected_record);
          print_record_brief("actual", actual_record);
        }
      }
    }
  }

  for (const auto& [key, actual_records] : actual.paths_by_key) {
    if (expected.paths_by_key.find(key) == expected.paths_by_key.end()) {
      report("extra path key in actual: " + key);
      if (differences <= options.max_diffs) {
        print_record_brief("actual", actual_records.front());
      }
    }
  }

  if (differences == 0) {
    std::cout << "[compare] SEMANTIC MATCH\n";
    return true;
  }

  std::cout << "[compare] SEMANTIC MISMATCH differences=" << differences;
  if (differences > options.max_diffs) {
    std::cout << " printed=" << options.max_diffs;
  }
  std::cout << '\n';
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      print_usage(argv[0]);
      return 2;
    }

    std::filesystem::path expected_path;
    std::filesystem::path actual_path;
    Options options;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        return 0;
      }
      if (arg == "--exact") {
        options.exact = true;
      } else if (arg == "--distance-only") {
        options.compare_path_shape = false;
      } else if (arg == "--abs-tol") {
        options.abs_tolerance = parse_double_arg(require_value(&i, argc, argv, "--abs-tol"),
                                                 "--abs-tol");
      } else if (arg == "--rel-tol") {
        options.rel_tolerance = parse_double_arg(require_value(&i, argc, argv, "--rel-tol"),
                                                 "--rel-tol");
      } else if (arg == "--max-diffs") {
        options.max_diffs = parse_u64_arg(require_value(&i, argc, argv, "--max-diffs"),
                                          "--max-diffs");
      } else if (!arg.empty() && arg[0] == '-') {
        throw std::runtime_error("unknown option: " + arg);
      } else if (expected_path.empty()) {
        expected_path = arg;
      } else if (actual_path.empty()) {
        actual_path = arg;
      } else {
        throw std::runtime_error("too many positional arguments");
      }
    }

    if (expected_path.empty() || actual_path.empty()) {
      throw std::runtime_error("expected two paths.jsonl files");
    }

    const bool ok = options.exact
                        ? exact_compare(expected_path, actual_path)
                        : semantic_compare(expected_path, actual_path, options);
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "[compare] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
