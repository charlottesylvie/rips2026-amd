#include "sssp_validation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>

namespace rips_sssp {
namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;

struct EdgeRef {
  int from = -1;
  int to = -1;
  Offset csr_edge = -1;
  bool has_cost = false;
  double cost = 0.0;
};

struct ParsedPathRecord {
  int source = -1;
  int target = -1;
  bool reached = true;
  bool has_distance = false;
  double distance = 0.0;
  std::vector<int> nodes;
  std::vector<Offset> csr_edges;
  std::vector<EdgeRef> edges;
};

struct JsonValue {
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_number() const { return std::holds_alternative<double>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_array() const { return std::holds_alternative<Array>(value); }
  bool is_object() const { return std::holds_alternative<Object>(value); }

  bool as_bool(const char* name) const {
    if (const auto* v = std::get_if<bool>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not bool: ") + name);
  }

  double as_number(const char* name) const {
    if (const auto* v = std::get_if<double>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not number: ") + name);
  }

  const std::string& as_string(const char* name) const {
    if (const auto* v = std::get_if<std::string>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not string: ") + name);
  }

  const Array& as_array(const char* name) const {
    if (const auto* v = std::get_if<Array>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not array: ") + name);
  }

  const Object& as_object(const char* name) const {
    if (const auto* v = std::get_if<Object>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not object: ") + name);
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters after JSON value");
    }
    return value;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    return text_[pos_];
  }

  bool consume(char expected) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    const char ch = peek();
    if (ch == '{') return JsonValue{parse_object()};
    if (ch == '[') return JsonValue{parse_array()};
    if (ch == '"') return JsonValue{parse_string()};
    if (ch == 't') {
      expect_literal("true");
      return JsonValue{true};
    }
    if (ch == 'f') {
      expect_literal("false");
      return JsonValue{false};
    }
    if (ch == 'n') {
      expect_literal("null");
      return JsonValue{nullptr};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return JsonValue{parse_number()};
    }
    throw std::runtime_error("unexpected JSON value");
  }

  JsonValue::Object parse_object() {
    if (!consume('{')) throw std::runtime_error("expected JSON object");
    JsonValue::Object object;
    if (consume('}')) return object;
    while (true) {
      std::string key = parse_string();
      if (!consume(':')) throw std::runtime_error("expected ':' in JSON object");
      object.emplace(std::move(key), parse_value());
      if (consume('}')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON object");
    }
    return object;
  }

  JsonValue::Array parse_array() {
    if (!consume('[')) throw std::runtime_error("expected JSON array");
    JsonValue::Array array;
    if (consume(']')) return array;
    while (true) {
      array.push_back(parse_value());
      if (consume(']')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON array");
    }
    return array;
  }

  std::string parse_string() {
    if (!consume('"')) throw std::runtime_error("expected JSON string");
    std::string out;
    while (pos_ < text_.size()) {
      const char ch = text_[pos_++];
      if (ch == '"') return out;
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
          throw std::runtime_error("unicode JSON escapes are not supported");
        default:
          throw std::runtime_error("unknown JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  double parse_number() {
    const std::size_t begin = pos_;
    if (text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }

    const std::string token = text_.substr(begin, pos_ - begin);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
      throw std::runtime_error("invalid JSON number");
    }
    return value;
  }

  void expect_literal(std::string_view literal) {
    if (text_.compare(pos_, literal.size(), literal) != 0) {
      throw std::runtime_error("invalid JSON literal");
    }
    pos_ += literal.size();
  }

  std::string text_;
  std::size_t pos_ = 0;
};

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) return;

  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

void validate_csr_shape(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }

  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

bool valid_node(const HostCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

double tolerance_for(double lhs, double rhs, const ValidatorOptions& options) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return options.abs_tolerance + options.rel_tolerance * scale;
}

bool close_enough(double lhs, double rhs, const ValidatorOptions& options) {
  return std::fabs(lhs - rhs) <= tolerance_for(lhs, rhs, options);
}

const JsonValue* find_field(const JsonValue::Object& object, const char* key) {
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

std::int64_t require_i64(const JsonValue::Object& object, const char* key) {
  const JsonValue* value = find_field(object, key);
  if (value == nullptr) {
    throw std::runtime_error(std::string("missing required JSON field: ") + key);
  }
  const double number = value->as_number(key);
  if (!std::isfinite(number) ||
      number < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      number > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
      std::floor(number) != number) {
    throw std::runtime_error(std::string("JSON integer field is invalid: ") + key);
  }
  return static_cast<std::int64_t>(number);
}

int require_node_id(const JsonValue::Object& object, const char* key) {
  const std::int64_t value = require_i64(object, key);
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("node id is outside int range: ") + key);
  }
  return static_cast<int>(value);
}

bool optional_bool(const JsonValue::Object& object, const char* key, bool fallback) {
  const JsonValue* value = find_field(object, key);
  return value == nullptr || value->is_null() ? fallback : value->as_bool(key);
}

std::string optional_string(const JsonValue::Object& object,
                            const char* key,
                            std::string fallback) {
  const JsonValue* value = find_field(object, key);
  return value == nullptr || value->is_null() ? std::move(fallback) : value->as_string(key);
}

std::vector<int> parse_node_array(const JsonValue::Object& object, const char* key) {
  std::vector<int> out;
  const JsonValue* value = find_field(object, key);
  if (value == nullptr || value->is_null()) return out;

  const auto& array = value->as_array(key);
  out.reserve(array.size());
  for (const JsonValue& item : array) {
    const double number = item.as_number(key);
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<int>::min()) ||
        number > static_cast<double>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(std::string("invalid node id in array: ") + key);
    }
    out.push_back(static_cast<int>(number));
  }
  return out;
}

std::vector<Offset> parse_offset_array(const JsonValue::Object& object, const char* key) {
  std::vector<Offset> out;
  const JsonValue* value = find_field(object, key);
  if (value == nullptr || value->is_null()) return out;

  const auto& array = value->as_array(key);
  out.reserve(array.size());
  for (const JsonValue& item : array) {
    const double number = item.as_number(key);
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<Offset>::min()) ||
        number > static_cast<double>(std::numeric_limits<Offset>::max())) {
      throw std::runtime_error(std::string("invalid offset in array: ") + key);
    }
    out.push_back(static_cast<Offset>(number));
  }
  return out;
}

std::vector<EdgeRef> parse_edges_array(const JsonValue::Object& object) {
  std::vector<EdgeRef> out;
  const JsonValue* value = find_field(object, "edges");
  if (value == nullptr || value->is_null()) return out;

  const auto& array = value->as_array("edges");
  out.reserve(array.size());
  for (const JsonValue& item : array) {
    const auto& edge_object = item.as_object("edges[]");
    EdgeRef edge;
    edge.from = require_node_id(edge_object, "from");
    edge.to = require_node_id(edge_object, "to");
    if (const JsonValue* csr_edge = find_field(edge_object, "csr_edge")) {
      if (!csr_edge->is_null()) {
        const double number = csr_edge->as_number("csr_edge");
        if (!std::isfinite(number) || std::floor(number) != number ||
            number < static_cast<double>(std::numeric_limits<Offset>::min()) ||
            number > static_cast<double>(std::numeric_limits<Offset>::max())) {
          throw std::runtime_error("invalid csr_edge in edge object");
        }
        edge.csr_edge = static_cast<Offset>(number);
      }
    }
    if (const JsonValue* cost = find_field(edge_object, "cost")) {
      if (!cost->is_null()) {
        edge.has_cost = true;
        edge.cost = cost->as_number("cost");
      }
    }
    out.push_back(edge);
  }
  return out;
}

ParsedPathRecord parse_path_record(const JsonValue::Object& object) {
  ParsedPathRecord record;
  record.source = require_node_id(object, "source");
  record.target = require_node_id(object, "target");
  record.reached = optional_bool(object, "reached", true);

  if (const JsonValue* distance = find_field(object, "distance")) {
    if (!distance->is_null()) {
      record.has_distance = true;
      record.distance = distance->as_number("distance");
    }
  }

  record.nodes = parse_node_array(object, "nodes");
  record.csr_edges = parse_offset_array(object, "csr_edges");
  record.edges = parse_edges_array(object);
  return record;
}

std::string parse_record_type(const JsonValue::Object& object) {
  return optional_string(object, "type", "");
}

std::string mode_token_normalized(std::string token) {
  token.erase(std::remove_if(token.begin(),
                             token.end(),
                             [](unsigned char ch) { return std::isspace(ch) != 0; }),
              token.end());
  std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return token;
}

void require_node_in_range(const HostCsrF32& graph, int node, const char* label) {
  if (!valid_node(graph, node)) {
    std::ostringstream msg;
    msg << label << " node " << node << " is outside graph range [0, "
        << graph.rows << ')';
    throw std::runtime_error(msg.str());
  }
}

bool find_edge_by_endpoints(const HostCsrF32& graph,
                            int from,
                            int to,
                            Offset* found_edge,
                            float* found_cost) {
  require_node_in_range(graph, from, "from");
  require_node_in_range(graph, to, "to");

  const std::size_t row = static_cast<std::size_t>(to);
  bool found = false;
  Offset best_edge = -1;
  float best_cost = std::numeric_limits<float>::infinity();
  for (Offset edge = graph.rowptr[row]; edge < graph.rowptr[row + 1]; ++edge) {
    if (graph.colind[static_cast<std::size_t>(edge)] != from) continue;
    const float cost = graph.values[static_cast<std::size_t>(edge)];
    if (!found || cost < best_cost) {
      found = true;
      best_edge = edge;
      best_cost = cost;
    }
  }

  if (found) {
    if (found_edge != nullptr) *found_edge = best_edge;
    if (found_cost != nullptr) *found_cost = best_cost;
  }
  return found;
}

float validate_edge_ref(const HostCsrF32& graph, const EdgeRef& edge) {
  require_node_in_range(graph, edge.from, "from");
  require_node_in_range(graph, edge.to, "to");

  if (edge.csr_edge >= 0) {
    if (edge.csr_edge >= graph.nnz) {
      throw std::runtime_error("csr_edge is outside CSR edge range");
    }
    const std::size_t to_index = static_cast<std::size_t>(edge.to);
    if (edge.csr_edge < graph.rowptr[to_index] ||
        edge.csr_edge >= graph.rowptr[to_index + 1]) {
      throw std::runtime_error("csr_edge does not live in the incoming CSR row for edge.to");
    }
    if (graph.colind[static_cast<std::size_t>(edge.csr_edge)] != edge.from) {
      throw std::runtime_error("csr_edge column does not match edge.from");
    }

    const float graph_cost = graph.values[static_cast<std::size_t>(edge.csr_edge)];
    if (edge.has_cost && !std::isfinite(edge.cost)) {
      throw std::runtime_error("edge cost in JSON is not finite");
    }
    return graph_cost;
  }

  Offset found_edge = -1;
  float graph_cost = 0.0f;
  if (!find_edge_by_endpoints(graph, edge.from, edge.to, &found_edge, &graph_cost)) {
    std::ostringstream msg;
    msg << "edge " << edge.from << " -> " << edge.to
        << " does not exist in the CSR graph";
    throw std::runtime_error(msg.str());
  }
  (void)found_edge;
  return graph_cost;
}

std::vector<EdgeRef> edge_refs_from_path_record(const ParsedPathRecord& path) {
  if (!path.edges.empty()) {
    return path.edges;
  }

  std::vector<EdgeRef> refs;
  if (!path.csr_edges.empty()) {
    if (path.nodes.empty()) {
      throw std::runtime_error("csr_edges requires nodes so each edge's endpoints are explicit");
    }
    if (path.csr_edges.size() + 1 != path.nodes.size()) {
      throw std::runtime_error("csr_edges length must equal nodes length minus one");
    }
    refs.reserve(path.csr_edges.size());
    for (std::size_t i = 0; i < path.csr_edges.size(); ++i) {
      EdgeRef edge;
      edge.from = path.nodes[i];
      edge.to = path.nodes[i + 1];
      edge.csr_edge = path.csr_edges[i];
      refs.push_back(edge);
    }
    return refs;
  }

  if (!path.nodes.empty()) {
    refs.reserve(path.nodes.size() > 0 ? path.nodes.size() - 1 : 0);
    for (std::size_t i = 1; i < path.nodes.size(); ++i) {
      EdgeRef edge;
      edge.from = path.nodes[i - 1];
      edge.to = path.nodes[i];
      refs.push_back(edge);
    }
  }
  return refs;
}

void validate_path_record(const HostCsrF32& graph,
                          const ParsedPathRecord& path,
                          const ValidatorOptions& options,
                          ValidationStats* stats) {
  require_node_in_range(graph, path.source, "source");
  require_node_in_range(graph, path.target, "target");

  if (!path.reached) {
    if ((options.modes & kModePathShape) &&
        (!path.nodes.empty() || !path.edges.empty() || !path.csr_edges.empty()) &&
        options.strict) {
      throw std::runtime_error("unreached path record should not contain a concrete path");
    }
    return;
  }

  if (path.has_distance && (!std::isfinite(path.distance) || path.distance < 0.0)) {
    throw std::runtime_error("reached path has invalid reported distance");
  }

  if ((options.modes & kModePathShape) && !path.nodes.empty()) {
    if (path.nodes.front() != path.source) {
      throw std::runtime_error("nodes[0] does not match source");
    }
    if (path.nodes.back() != path.target) {
      throw std::runtime_error("nodes[-1] does not match target");
    }
    for (const int node : path.nodes) {
      require_node_in_range(graph, node, "path");
    }
  }

  std::vector<EdgeRef> refs = edge_refs_from_path_record(path);
  if (path.source == path.target && refs.empty()) {
    if ((options.modes & kModePathCost) && path.has_distance &&
        !close_enough(path.distance, 0.0, options)) {
      throw std::runtime_error("source==target path should have distance 0");
    }
    return;
  }
  if (refs.empty()) {
    throw std::runtime_error("reached path needs nodes, csr_edges+nodes, or edges");
  }

  if ((options.modes & kModePathShape) && refs.front().from != path.source) {
    throw std::runtime_error("first edge does not start at source");
  }
  if ((options.modes & kModePathShape) && refs.back().to != path.target) {
    throw std::runtime_error("last edge does not end at target");
  }

  double cost_sum = 0.0;
  for (std::size_t i = 0; i < refs.size(); ++i) {
    if ((options.modes & kModePathShape) && i > 0 && refs[i - 1].to != refs[i].from) {
      throw std::runtime_error("path edges are not continuous");
    }

    float graph_cost = 0.0f;
    if (options.modes & (kModePathExists | kModePathCost)) {
      graph_cost = validate_edge_ref(graph, refs[i]);
      ++stats->path_edge_checks;
    }
    if ((options.modes & kModePathCost) && refs[i].has_cost &&
        !close_enough(refs[i].cost, graph_cost, options)) {
      throw std::runtime_error("JSON edge cost does not match CSR edge cost");
    }
    cost_sum += graph_cost;
  }

  if ((options.modes & kModePathCost) && path.has_distance &&
      !close_enough(path.distance, cost_sum, options)) {
    std::ostringstream msg;
    msg << "path distance mismatch: reported " << path.distance
        << ", CSR path cost " << cost_sum;
    throw std::runtime_error(msg.str());
  }
}

std::string make_progress_bar(double fraction) {
  fraction = std::max(0.0, std::min(1.0, fraction));
  constexpr int kWidth = 30;
  const int filled = static_cast<int>(std::round(fraction * kWidth));
  std::string bar = "[";
  for (int i = 0; i < kWidth; ++i) {
    bar.push_back(i < filled ? '#' : '-');
  }
  bar.push_back(']');
  return bar;
}

void print_progress(const std::filesystem::path& jsonl_path,
                    std::uint64_t bytes_processed,
                    std::uint64_t file_size,
                    const ValidationStats& stats) {
  const double fraction =
      file_size == 0 ? 1.0 : static_cast<double>(bytes_processed) / file_size;
  std::cout << "\r[jsonl] " << make_progress_bar(fraction) << ' '
            << std::fixed << std::setprecision(1) << (fraction * 100.0)
            << "% lines=" << stats.jsonl_lines
            << " paths_validated=" << stats.paths_validated
            << " failures=" << stats.failures
            << " file=" << jsonl_path.filename().string() << std::flush;
}

bool should_sample_path(std::mt19937_64* rng, double sample_rate) {
  if (sample_rate >= 1.0) return true;
  if (sample_rate <= 0.0) return false;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(*rng) < sample_rate;
}

void report_warning(const std::string& message, ValidationStats* stats) {
  ++stats->warnings;
  std::cout << "\n[warning] " << message << '\n';
}

void report_failure(std::uint64_t line,
                    const std::string& message,
                    const ValidatorOptions& options,
                    ValidationStats* stats) {
  ++stats->failures;
  if (stats->failures <= options.max_errors) {
    std::cout << "\n[failure] line " << line << ": " << message << '\n';
  }
  if (options.max_errors != 0 && stats->failures >= options.max_errors) {
    throw std::runtime_error("stopping after reaching --max-errors");
  }
}

}  // namespace

HostCsrF32 load_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(in, "CSR format version");
  const std::uint64_t orientation = read_u64(in, "CSR orientation");
  if (version != EXPECTED_CSR_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "CSR row count");
  const std::uint64_t cols = read_u64(in, "CSR column count");
  (void)read_u64(in, "declared edge count");
  (void)read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(in, "CSR colind count");
  const std::uint64_t values_count = read_u64(in, "CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this validator API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(in, graph.colind, colind_count, "CSR colind");
  read_array(in, graph.values, values_count, "CSR values");
  validate_csr_shape(graph);
  return graph;
}

unsigned parse_mode_list(const std::string& text) {
  if (text.empty()) {
    throw std::runtime_error("mode list is empty");
  }

  unsigned modes = 0;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = mode_token_normalized(token);
    if (token.empty()) continue;
    if (token == "all") {
      modes |= kModePathExists | kModePathCost | kModePathShape;
    } else if (token == "1" || token == "path-exists" || token == "exists") {
      modes |= kModePathExists;
    } else if (token == "2" || token == "path-cost" || token == "cost") {
      modes |= kModePathCost;
    } else if (token == "3" || token == "path-shape" || token == "shape") {
      modes |= kModePathShape;
    } else {
      throw std::runtime_error("unknown validation mode: " + token);
    }
  }

  if (modes == 0) {
    throw std::runtime_error("mode list did not select any modes");
  }
  return modes;
}

std::string describe_modes(unsigned modes) {
  std::vector<std::string> names;
  if (modes & kModePathExists) names.push_back("1:path-exists");
  if (modes & kModePathCost) names.push_back("2:path-cost");
  if (modes & kModePathShape) names.push_back("3:path-shape");
  std::ostringstream out;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i != 0) out << ',';
    out << names[i];
  }
  return out.str();
}

ValidationStats validate_paths_jsonl(const HostCsrF32& graph,
                                     const std::filesystem::path& jsonl_path,
                                     const ValidatorOptions& options) {
  if (options.sample_rate < 0.0 || options.sample_rate > 1.0) {
    throw std::runtime_error("--sample-rate must be in [0, 1]");
  }

  std::ifstream in(jsonl_path);
  if (!in) {
    throw std::runtime_error("could not open JSONL file: " + jsonl_path.string());
  }

  std::uint64_t file_size = 0;
  if (std::filesystem::exists(jsonl_path)) {
    file_size = std::filesystem::file_size(jsonl_path);
  }

  ValidationStats stats;
  std::mt19937_64 rng(options.sample_seed);
  std::string line;
  std::uint64_t bytes_processed = 0;

  std::cout << "[jsonl] validating " << jsonl_path.string() << '\n';
  std::cout << "[jsonl] modes=" << describe_modes(options.modes)
            << " sample_rate=" << options.sample_rate
            << " max_paths=" << options.max_paths
            << " progress_every=" << options.progress_every << '\n';

  while (std::getline(in, line)) {
    ++stats.jsonl_lines;
    bytes_processed += static_cast<std::uint64_t>(line.size() + 1);
    if (line.empty()) {
      continue;
    }

    try {
      JsonValue root = JsonParser(line).parse();
      const auto& object = root.as_object("record");
      std::string type = parse_record_type(object);
      if (type.empty()) {
        if (find_field(object, "target") != nullptr) {
          type = "path";
        } else {
          type = "metadata";
        }
      }

      if (type == "metadata") {
        ++stats.metadata_records;
      } else if (type == "path") {
        ++stats.path_records;
        const bool selected =
            should_sample_path(&rng, options.sample_rate) &&
            (options.max_paths == 0 || stats.paths_selected < options.max_paths);
        if (!selected) {
          ++stats.paths_skipped_by_sampling;
        } else {
          ++stats.paths_selected;
          ParsedPathRecord path = parse_path_record(object);
          validate_path_record(graph, path, options, &stats);
          ++stats.paths_validated;
        }
      } else {
        report_warning("unknown record type '" + type + "' on line " +
                           std::to_string(stats.jsonl_lines) + "; skipping",
                       &stats);
      }
    } catch (const std::exception& ex) {
      report_failure(stats.jsonl_lines, ex.what(), options, &stats);
    }

    if (options.progress_every != 0 &&
        stats.jsonl_lines % options.progress_every == 0) {
      print_progress(jsonl_path, bytes_processed, file_size, stats);
    }
  }

  print_progress(jsonl_path, file_size == 0 ? bytes_processed : file_size, file_size, stats);
  std::cout << '\n';
  return stats;
}

void print_jsonl_format(std::ostream& out) {
  out
      << "Recommended RIPS SSSP JSONL format, version rips-sssp-paths-v1\n"
      << "\n"
      << "Each line is one JSON object. Edge weights come from the CSR file;\n"
      << "the JSONL records describe selected source-target paths.\n"
      << "\n"
      << "1) Optional metadata record:\n"
      << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\","
      << "\"node_count\":28000000,\"edge_orientation\":\"incoming\","
      << "\"description\":\"row v, col u means directed edge u -> v\"}\n"
      << "\n"
      << "2) Recommended compact path record:\n"
      << "{\"type\":\"path\",\"source\":12345,\"target\":67890,\"reached\":true,"
      << "\"distance\":17.0,\"nodes\":[12345,22222,67890],"
      << "\"csr_edges\":[901,902]}\n"
      << "\n"
      << "The path record distance is the algorithm's reported total path length.\n"
      << "Mode 2 recomputes that total from CSR edge weights.\n"
      << "\n"
      << "nodes must list the path from source to target. csr_edges is strongly\n"
      << "recommended because it disambiguates parallel edges and lets the validator\n"
      << "check the exact CSR edge. For edge e from u to v, e must satisfy:\n"
      << "rowptr[v] <= e < rowptr[v+1] and colind[e] == u.\n"
      << "\n"
      << "3) More verbose debug path record, also accepted:\n"
      << "{\"type\":\"path\",\"source\":12345,\"target\":67890,\"reached\":true,"
      << "\"distance\":17.0,\"edges\":[{\"from\":12345,\"to\":22222,"
      << "\"csr_edge\":901,\"cost\":8.0},{\"from\":22222,\"to\":67890,"
      << "\"csr_edge\":902,\"cost\":9.0}]}\n"
      << "\n"
      << "Unreached target record:\n"
      << "{\"type\":\"path\",\"source\":12345,\"target\":99999,\"reached\":false,"
      << "\"distance\":null,\"nodes\":[],\"csr_edges\":[]}\n";
}

}  // namespace rips_sssp
