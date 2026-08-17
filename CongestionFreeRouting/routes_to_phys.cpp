// Reconstructs a routed FPGA Interchange PhysicalNetlist from PathFinder's
// route JSONL output and the RIPS interchange metadata sidecar.
//
// Benchmark-facing use, after interchange_to_csr and pathfinder have run:
//   routes_to_phys <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys>
//
// Expected generated schema header:
//   PhysicalNetlist.capnp.h
//
// Example compile command:
//   g++ -std=c++17 -O3 \
//     -I<generated-schema-dir> \
//     CongestionFreeRouting/routes_to_phys.cpp \
//     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
//     -lcapnp -lkj -lz \
//     -o routes_to_phys

#include "PhysicalNetlist.capnp.h"
#include "interchange/gzip_io.hpp"
#include "interchange/import_policy.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t LEGACY_METADATA_VERSION = 4;
constexpr std::uint64_t CURRENT_METADATA_VERSION = 8;
constexpr std::uint64_t ARTIFACT_PAIR_METADATA_VERSION = 5;
constexpr std::uint64_t COMPACT_METADATA_VERSION = 6;
constexpr std::uint64_t ENDPOINT_PIP_METADATA_VERSION = 7;
constexpr std::uint64_t COMPACT_TABLE_METADATA_VERSION = 8;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kInvalidRouteNode =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoEndpointPip =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kNoRouteString =
    std::numeric_limits<std::uint32_t>::max();

constexpr bool metadata_has_artifact_pair(std::uint64_t version) {
  return version == ARTIFACT_PAIR_METADATA_VERSION ||
         version == COMPACT_METADATA_VERSION ||
         version == ENDPOINT_PIP_METADATA_VERSION ||
         version == COMPACT_TABLE_METADATA_VERSION;
}

constexpr bool metadata_has_endpoint_pips(std::uint64_t version) {
  return version == ENDPOINT_PIP_METADATA_VERSION ||
         version == COMPACT_TABLE_METADATA_VERSION;
}

constexpr bool metadata_has_legacy_node_arrays(std::uint64_t version) {
  return version == LEGACY_METADATA_VERSION ||
         version == ARTIFACT_PAIR_METADATA_VERSION;
}

constexpr bool metadata_has_legacy_logical_payloads(std::uint64_t version) {
  return version == LEGACY_METADATA_VERSION ||
         version == ARTIFACT_PAIR_METADATA_VERSION ||
         version == COMPACT_METADATA_VERSION ||
         version == ENDPOINT_PIP_METADATA_VERSION;
}

struct SitePinKey {
  std::string site;
  std::string pin;

  bool operator==(const SitePinKey& other) const {
    return site == other.site && pin == other.pin;
  }
};

struct SitePinKeyHash {
  std::size_t operator()(const SitePinKey& key) const noexcept {
    const std::size_t site_hash = std::hash<std::string>{}(key.site);
    const std::size_t pin_hash = std::hash<std::string>{}(key.pin);
    return site_hash ^ (pin_hash + 0x9e3779b9U + (site_hash << 6) +
                        (site_hash >> 2));
  }
};

struct RouteSitePin {
  int node = -1;
  std::string site;
  std::string pin;
  bool reached = true;
  std::uint64_t endpoint_pip_index = kNoEndpointPip;
};

struct RouteEdge {
  int from = -1;
  int to = -1;
  std::uint64_t csr_edge = 0;
  std::uint32_t tile = kNoRouteString;
  std::uint32_t wire0 = kNoRouteString;
  std::uint32_t wire1 = kNoRouteString;
  bool forward = true;
  bool attachment_field_present = false;
  std::optional<std::uint64_t> attachment;
  bool site_field_present = false;
  std::uint32_t site = kNoRouteString;
};

struct NetRoute {
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::string net;
  bool routed = false;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
  std::vector<RouteEdge> edges;
};

struct StoredStubBranch {
  SitePinKey key;
  std::uint32_t branch_index = 0;
  bool consumed = false;
};

struct StubBranchStore {
  std::vector<StoredStubBranch> stubs;
  std::unordered_map<SitePinKey, std::vector<std::size_t>, SitePinKeyHash>
      by_key;
  std::unordered_map<SitePinKey, std::size_t, SitePinKeyHash> cursor_by_key;
};

struct MetadataRouteRequest {
  std::string net;
  std::uint64_t logical_net_index = kNoEndpointPip;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
};

enum class MetadataEndpointPipRole : std::uint64_t {
  kSource = 0,
  kSink = 1,
};

struct MetadataEndpointPip {
  std::uint64_t csr_edge = 0;
  int from = -1;
  int to = -1;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
  std::uint64_t site_string = 0;
  int endpoint_node = -1;
  MetadataEndpointPipRole role = MetadataEndpointPipRole::kSource;
};

struct RoutingMetadataSummary {
  std::uint64_t version = 0;
  std::uint64_t node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::vector<std::string> strings;
  std::vector<MetadataEndpointPip> endpoint_pips;
  std::vector<MetadataRouteRequest> route_requests;
  std::vector<std::uint64_t> logical_net_name_strings;
};

class JsonCursor {
 public:
  explicit JsonCursor(std::string_view text) : text_(text) {}

  void require_end() {
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters after JSON value");
    }
  }

  bool consume(char expected) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char expected, const char* message) {
    if (!consume(expected)) throw std::runtime_error(message);
  }

  char peek() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    return text_[pos_];
  }

  std::string parse_string() {
    std::string out;
    parse_string_into(&out);
    return out;
  }

  void skip_string() { parse_string_into(nullptr); }

  bool parse_bool() {
    if (match_literal("true")) return true;
    if (match_literal("false")) return false;
    throw std::runtime_error("expected JSON boolean");
  }

  bool consume_null() {
    skip_ws();
    if (text_.substr(pos_, 4) != "null") return false;
    pos_ += 4;
    return true;
  }

  int parse_int(const char* field) {
    constexpr std::uint64_t kPositiveLimit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    constexpr std::uint64_t kNegativeLimit = kPositiveLimit + 1;
    const ParsedInteger value =
        parse_integer_token(kPositiveLimit, kNegativeLimit);
    if (value.negative) {
      if (value.magnitude == kNegativeLimit) {
        return std::numeric_limits<std::int32_t>::min();
      }
      return -static_cast<int>(value.magnitude);
    }
    if (value.magnitude > kPositiveLimit) {
      throw std::runtime_error(std::string("JSON field is not an in-range integer: ") +
                               field);
    }
    return static_cast<int>(value.magnitude);
  }

  std::uint64_t parse_u64(const char* field) {
    constexpr std::uint64_t kLargestExactJsonInteger =
        9007199254740991ULL;
    const ParsedInteger value =
        parse_integer_token(kLargestExactJsonInteger, 0);
    if (value.negative && value.magnitude != 0) {
      throw std::runtime_error(
          std::string("JSON field is not an exact nonnegative integer: ") +
          field);
    }
    return value.magnitude;
  }

  void skip_value(std::size_t depth = 0) {
    if (depth > 1024) {
      throw std::runtime_error("JSON nesting is too deep");
    }
    const char ch = peek();
    if (ch == '{') {
      consume('{');
      if (consume('}')) return;
      while (true) {
        skip_string();
        expect(':', "expected ':' in JSON object");
        skip_value(depth + 1);
        if (consume('}')) return;
        expect(',', "expected ',' in JSON object");
      }
    }
    if (ch == '[') {
      consume('[');
      if (consume(']')) return;
      while (true) {
        skip_value(depth + 1);
        if (consume(']')) return;
        expect(',', "expected ',' in JSON array");
      }
    }
    if (ch == '"') {
      skip_string();
      return;
    }
    if (ch == 't' || ch == 'f') {
      (void)parse_bool();
      return;
    }
    if (ch == 'n') {
      if (!consume_null()) throw std::runtime_error("expected JSON null");
      return;
    }
    skip_number();
  }

  void skip_number() {
    const std::string_view token = scan_number_token();
    // Most route numbers are short integers. Only pay for floating conversion
    // when a skipped extension value could overflow/underflow double or uses
    // a fractional/exponent spelling accepted by the legacy parser.
    if (token.size() > 32 ||
        token.find_first_of(".eE") != std::string_view::npos) {
      (void)parse_finite_number(token);
    }
  }

 private:
  struct ParsedInteger {
    bool negative = false;
    std::uint64_t magnitude = 0;
  };

  void parse_string_into(std::string* out) {
    expect('"', "expected JSON string");
    while (pos_ < text_.size()) {
      const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
      if (ch == '"') return;
      if (ch < 0x20) {
        throw std::runtime_error("unescaped control character in JSON string");
      }
      if (ch != '\\') {
        if (out != nullptr) out->push_back(static_cast<char>(ch));
        continue;
      }
      if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          if (out != nullptr) out->push_back(esc);
          break;
        case 'b': if (out != nullptr) out->push_back('\b'); break;
        case 'f': if (out != nullptr) out->push_back('\f'); break;
        case 'n': if (out != nullptr) out->push_back('\n'); break;
        case 'r': if (out != nullptr) out->push_back('\r'); break;
        case 't': if (out != nullptr) out->push_back('\t'); break;
        case 'u': {
          std::uint32_t codepoint = parse_hex_quad();
          if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            if (pos_ + 2 > text_.size() || text_[pos_] != '\\' ||
                text_[pos_ + 1] != 'u') {
              throw std::runtime_error("unpaired high surrogate in JSON string");
            }
            pos_ += 2;
            const std::uint32_t low = parse_hex_quad();
            if (low < 0xdc00 || low > 0xdfff) {
              throw std::runtime_error("invalid low surrogate in JSON string");
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                        (low - 0xdc00);
          } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
            throw std::runtime_error("unpaired low surrogate in JSON string");
          }
          if (out != nullptr) append_utf8(*out, codepoint);
          break;
        }
        default:
          throw std::runtime_error("bad JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }
  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  static double parse_finite_number(std::string_view scanned) {
    const std::string token(scanned);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size() || errno == ERANGE ||
        !std::isfinite(value)) {
      throw std::runtime_error("invalid finite JSON number");
    }
    return value;
  }

  ParsedInteger parse_integer_token(std::uint64_t positive_limit,
                                    std::uint64_t negative_limit) {
    const std::string_view token = scan_number_token();
    // Keep the legacy finite-number guarantee, but never use the rounded
    // double to decide integrality or range.  For example, strtod rounds
    // 1.0000000000000001 to 1 and 9007199254740991.4 to an integer.
    (void)parse_finite_number(token);

    std::size_t pos = 0;
    const bool negative = token[pos] == '-';
    if (negative) ++pos;
    const std::size_t integer_begin = pos;
    while (pos < token.size() &&
           std::isdigit(static_cast<unsigned char>(token[pos]))) {
      ++pos;
    }
    const std::size_t integer_end = pos;
    std::size_t fraction_begin = pos;
    std::size_t fraction_end = pos;
    if (pos < token.size() && token[pos] == '.') {
      fraction_begin = ++pos;
      while (pos < token.size() &&
             std::isdigit(static_cast<unsigned char>(token[pos]))) {
        ++pos;
      }
      fraction_end = pos;
    }

    bool exponent_negative = false;
    std::size_t exponent_magnitude = 0;
    if (pos < token.size() && (token[pos] == 'e' || token[pos] == 'E')) {
      ++pos;
      if (token[pos] == '+' || token[pos] == '-') {
        exponent_negative = token[pos] == '-';
        ++pos;
      }
      for (; pos < token.size(); ++pos) {
        const unsigned digit = static_cast<unsigned>(token[pos] - '0');
        if (exponent_magnitude >
            (std::numeric_limits<std::size_t>::max() - digit) / 10) {
          exponent_magnitude = std::numeric_limits<std::size_t>::max();
        } else {
          exponent_magnitude = exponent_magnitude * 10 + digit;
        }
      }
    }

    const std::size_t integer_digits = integer_end - integer_begin;
    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (integer_digits >
        std::numeric_limits<std::size_t>::max() - fraction_digits) {
      throw std::runtime_error("JSON integer digit count overflows size_t");
    }
    const std::size_t digit_count = integer_digits + fraction_digits;
    const auto digit_at = [&](std::size_t index) -> char {
      if (index < integer_digits) return token[integer_begin + index];
      return token[fraction_begin + index - integer_digits];
    };

    bool all_zero = true;
    for (std::size_t index = 0; index < digit_count; ++index) {
      if (digit_at(index) != '0') {
        all_zero = false;
        break;
      }
    }
    if (all_zero) return ParsedInteger{negative, 0};

    std::size_t kept_digits = digit_count;
    std::size_t appended_zeros = 0;
    if (exponent_negative) {
      if (exponent_magnitude >
          std::numeric_limits<std::size_t>::max() - fraction_digits) {
        throw std::runtime_error("JSON number is not an integer");
      }
      const std::size_t removed_digits =
          exponent_magnitude + fraction_digits;
      if (removed_digits > digit_count) {
        throw std::runtime_error("JSON number is not an integer");
      }
      kept_digits -= removed_digits;
    } else if (exponent_magnitude < fraction_digits) {
      kept_digits -= fraction_digits - exponent_magnitude;
    } else {
      appended_zeros = exponent_magnitude - fraction_digits;
    }

    for (std::size_t index = kept_digits; index < digit_count; ++index) {
      if (digit_at(index) != '0') {
        throw std::runtime_error("JSON number is not an integer");
      }
    }

    std::size_t first_significant = 0;
    while (first_significant < kept_digits &&
           digit_at(first_significant) == '0') {
      ++first_significant;
    }
    const std::uint64_t limit = negative ? negative_limit : positive_limit;
    if (first_significant == kept_digits) {
      return ParsedInteger{negative, 0};
    }
    // No accepted limit has more than 19 decimal digits. This also rejects
    // enormous positive exponents without looping over every implied zero.
    if (kept_digits - first_significant > 20 || appended_zeros > 20 ||
        kept_digits - first_significant + appended_zeros > 20) {
      throw std::runtime_error("JSON integer is out of range");
    }
    std::uint64_t magnitude = 0;
    const auto append_digit = [&](unsigned digit) {
      if (magnitude > limit / 10 ||
          (magnitude == limit / 10 && digit > limit % 10)) {
        throw std::runtime_error("JSON integer is out of range");
      }
      magnitude = magnitude * 10 + digit;
    };
    for (std::size_t index = first_significant; index < kept_digits;
         ++index) {
      append_digit(static_cast<unsigned>(digit_at(index) - '0'));
    }
    for (std::size_t index = 0; index < appended_zeros; ++index) {
      append_digit(0);
    }
    return ParsedInteger{negative, magnitude};
  }

  std::string_view scan_number_token() {
    skip_ws();
    const std::size_t begin = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      throw std::runtime_error("invalid JSON number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
      if (pos_ < text_.size() &&
          std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("JSON number has a leading zero");
      }
    } else {
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      const std::size_t fractional_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == fractional_begin) {
        throw std::runtime_error("JSON number has an empty fraction");
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      const std::size_t exponent_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == exponent_begin) {
        throw std::runtime_error("JSON number has an empty exponent");
      }
    }
    return text_.substr(begin, pos_ - begin);
  }

  bool match_literal(const char* literal) {
    skip_ws();
    const std::size_t n = std::strlen(literal);
    if (text_.substr(pos_, n) != literal) return false;
    pos_ += n;
    return true;
  }

  std::uint32_t parse_hex_quad() {
    if (pos_ + 4 > text_.size()) {
      throw std::runtime_error("truncated JSON unicode escape");
    }
    std::uint32_t code = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = text_[pos_++];
      code <<= 4;
      if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
      else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
      else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
      else throw std::runtime_error("bad JSON unicode escape");
    }
    return code;
  }

  static void append_utf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
      out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
      out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
  }

  std::string_view text_;
  std::size_t pos_ = 0;
};

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error(std::string("failed while reading ") + name);
  return value;
}

int read_route_node(std::ifstream& in, const char* name) {
  const std::uint64_t raw = read_u64(in, name);
  if (raw == kInvalidRouteNode) {
    return -1;
  }
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

int checked_nonnegative_int(std::uint64_t raw, const char* name) {
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

std::size_t checked_size_count(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds size_t range");
  }
  return static_cast<std::size_t>(count);
}

std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t bytes_per_item,
                                 const char* name) {
  if (bytes_per_item != 0 &&
      count > std::numeric_limits<std::uint64_t>::max() / bytes_per_item) {
    throw std::runtime_error(std::string(name) + " byte count overflow");
  }
  return count * bytes_per_item;
}

// Seek over unused bulk metadata without reading it through a temporary
// buffer. Checking the remaining file length first is required because a
// standard seek is otherwise allowed to move beyond end-of-file silently.
void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) {
    return;
  }
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) +
                             " byte count is too large to seek");
  }
  const std::streampos current = in.tellg();
  if (current == std::streampos(-1)) {
    throw std::runtime_error(std::string("failed while locating ") + name);
  }
  in.seekg(0, std::ios::end);
  const std::streampos end = in.tellg();
  if (!in || end == std::streampos(-1) || end < current ||
      static_cast<std::uint64_t>(end - current) < count) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
  in.seekg(current + static_cast<std::streamoff>(count));
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

std::string read_metadata_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large");
  }
  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) throw std::runtime_error("failed while reading metadata string");
  }
  return text;
}

const std::string& string_at(const RoutingMetadataSummary& metadata,
                             std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata references an invalid string index: " +
                             std::to_string(index));
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

RoutingMetadataSummary load_metadata_summary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open metadata file: " + path.string());

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version < LEGACY_METADATA_VERSION ||
      version > CURRENT_METADATA_VERSION) {
    throw std::runtime_error(
        "unsupported metadata version; regenerate it with "
        "interchange_to_csr");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  if (metadata_has_artifact_pair(version)) {
    routing::interchange::InterchangeArtifactPairId id;
    id.high = read_u64(in, "metadata artifact pair id high");
    id.low = read_u64(in, "metadata artifact pair id low");
    if (id.is_zero()) {
      throw std::runtime_error("metadata artifact pair id must not be zero");
    }
    artifact_pair_id = id;
  }

  const std::uint64_t string_count = read_u64(in, "string count");
  const std::uint64_t node_count = read_u64(in, "node count");
  const std::uint64_t edge_attr_count = read_u64(in, "edge attr count");
  const std::uint64_t pip_data_count = read_u64(in, "pip data count");
  const std::uint64_t endpoint_pip_count = metadata_has_endpoint_pips(version)
                                               ? read_u64(in,
                                                          "endpoint PIP count")
                                               : 0;
  const std::uint64_t site_pin_attr_count = read_u64(in, "site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "physical netlist byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "logical netlist byte count");
  if (version == COMPACT_TABLE_METADATA_VERSION) {
    if (string_count > std::numeric_limits<std::uint32_t>::max() ||
        pip_data_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error(
          "metadata v8 string/PIP counts exceed compact uint32 limits");
    }
    if (logical_cell_count != 0 || logical_port_instance_count != 0 ||
        physical_netlist_byte_count != 0 ||
        logical_netlist_byte_count != 0) {
      throw std::runtime_error(
          "metadata v8 omitted hierarchy/payload counts must be zero");
    }
  }

  (void)read_u64(in, "device path string");
  (void)read_u64(in, "physical path string");
  (void)read_u64(in, "logical path string");
  (void)read_u64(in, "logical design name string");

  RoutingMetadataSummary metadata;
  metadata.version = version;
  metadata.node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.artifact_pair_id = artifact_pair_id;
  metadata.strings.reserve(checked_size_count(string_count, "string count"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_metadata_string(in));
  }

  if (metadata_has_legacy_node_arrays(version)) {
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node ids"),
               "node ids");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node min x coordinates"),
               "node min x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node max x coordinates"),
               "node max x coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node min y coordinates"),
               "node min y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::int32_t),
                                  "node max y coordinates"),
               "node max y coordinates");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node tile type strings"),
               "node tile type strings");
    skip_bytes(in,
               checked_byte_count(node_count, sizeof(std::uint64_t),
                                  "node wire type strings"),
               "node wire type strings");
  }
  // These full-device tables can contain tens of millions of records.  Route
  // reconstruction only needs the sparse, self-contained EndpointPip table,
  // so retain the original streaming behavior here.
  const std::size_t edge_attr_record_bytes =
      version == COMPACT_TABLE_METADATA_VERSION
          ? 2 * sizeof(std::uint32_t)
          : 2 * sizeof(std::uint64_t);
  const std::size_t pip_record_bytes =
      version == COMPACT_TABLE_METADATA_VERSION
          ? 3 * sizeof(std::uint32_t)
          : 3 * sizeof(std::uint64_t);
  skip_bytes(in,
             checked_byte_count(edge_attr_count, edge_attr_record_bytes,
                                "edge attrs"),
             "edge attrs");
  skip_bytes(in,
             checked_byte_count(pip_data_count, pip_record_bytes,
                                "PIP data"),
             "PIP data");

  metadata.endpoint_pips.reserve(
      checked_size_count(endpoint_pip_count, "endpoint PIP count"));
  std::unordered_set<std::uint64_t> endpoint_pip_csr_edges;
  endpoint_pip_csr_edges.reserve(
      checked_size_count(endpoint_pip_count, "endpoint PIP count"));
  for (std::uint64_t i = 0; i < endpoint_pip_count; ++i) {
    MetadataEndpointPip endpoint;
    endpoint.csr_edge = read_u64(in, "endpoint PIP CSR edge");
    endpoint.from = checked_nonnegative_int(
        read_u64(in, "endpoint PIP source node"),
        "endpoint PIP source node");
    endpoint.to = checked_nonnegative_int(
        read_u64(in, "endpoint PIP destination node"),
        "endpoint PIP destination node");
    endpoint.tile_string = read_u64(in, "endpoint PIP tile string");
    endpoint.wire0_string = read_u64(in, "endpoint PIP wire0 string");
    endpoint.wire1_string = read_u64(in, "endpoint PIP wire1 string");
    const std::uint64_t forward = read_u64(in, "endpoint PIP forward flag");
    endpoint.site_string = read_u64(in, "endpoint PIP site string");
    endpoint.endpoint_node = checked_nonnegative_int(
        read_u64(in, "endpoint PIP endpoint node"),
        "endpoint PIP endpoint node");
    const std::uint64_t role = read_u64(in, "endpoint PIP role");

    if (forward > 1) {
      throw std::runtime_error(
          "metadata endpoint PIP has an invalid forward flag");
    }
    endpoint.forward = forward != 0;
    if (role == static_cast<std::uint64_t>(
                    MetadataEndpointPipRole::kSource)) {
      endpoint.role = MetadataEndpointPipRole::kSource;
    } else if (role == static_cast<std::uint64_t>(
                           MetadataEndpointPipRole::kSink)) {
      endpoint.role = MetadataEndpointPipRole::kSink;
    } else {
      throw std::runtime_error("metadata endpoint PIP has an invalid role");
    }
    if (endpoint.csr_edge >= edge_attr_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid CSR edge");
    }
    if (static_cast<std::uint64_t>(endpoint.from) >= node_count ||
        static_cast<std::uint64_t>(endpoint.to) >= node_count ||
        static_cast<std::uint64_t>(endpoint.endpoint_node) >= node_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid node");
    }
    if (endpoint.from == endpoint.to ||
        endpoint.endpoint_node == endpoint.from ||
        endpoint.endpoint_node == endpoint.to) {
      throw std::runtime_error(
          "metadata endpoint PIP has invalid endpoint alignment");
    }
    (void)string_at(metadata, endpoint.tile_string);
    (void)string_at(metadata, endpoint.wire0_string);
    (void)string_at(metadata, endpoint.wire1_string);
    if (string_at(metadata, endpoint.site_string).empty()) {
      throw std::runtime_error(
          "metadata endpoint PIP has an empty concrete site");
    }
    if (!endpoint_pip_csr_edges.insert(endpoint.csr_edge).second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }

    metadata.endpoint_pips.push_back(endpoint);
  }

  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t), "site pin attrs"),
             "site pin attrs");

  metadata.route_requests.reserve(
      checked_size_count(route_request_count, "route request count"));
  for (std::uint64_t i = 0; i < route_request_count; ++i) {
    MetadataRouteRequest request;
    request.net = string_at(metadata, read_u64(in, "route request net"));
    request.logical_net_index = read_u64(in, "route request logical net");

    const std::uint64_t source_count = read_u64(in, "source count");
    request.sources.reserve(checked_size_count(source_count, "source count"));
    for (std::uint64_t s = 0; s < source_count; ++s) {
      RouteSitePin source;
      source.node = read_route_node(in, "source node");
      source.site = string_at(metadata, read_u64(in, "source site"));
      source.pin = string_at(metadata, read_u64(in, "source pin"));
      source.endpoint_pip_index =
          metadata_has_endpoint_pips(version)
              ? read_u64(in, "source endpoint PIP index")
              : kNoEndpointPip;
      if (source.endpoint_pip_index != kNoEndpointPip) {
        if (source.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata source references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(source.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSource ||
            endpoint.endpoint_node != source.node) {
          throw std::runtime_error(
              "metadata source references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sources.push_back(std::move(source));
    }

    const std::uint64_t sink_count = read_u64(in, "sink count");
    request.sinks.reserve(checked_size_count(sink_count, "sink count"));
    for (std::uint64_t s = 0; s < sink_count; ++s) {
      RouteSitePin sink;
      sink.node = read_route_node(in, "sink node");
      sink.site = string_at(metadata, read_u64(in, "sink site"));
      sink.pin = string_at(metadata, read_u64(in, "sink pin"));
      sink.endpoint_pip_index =
          metadata_has_endpoint_pips(version)
              ? read_u64(in, "sink endpoint PIP index")
              : kNoEndpointPip;
      if (sink.endpoint_pip_index != kNoEndpointPip) {
        if (sink.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata sink references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(sink.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSink ||
            endpoint.endpoint_node != sink.node) {
          throw std::runtime_error(
              "metadata sink references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sinks.push_back(std::move(sink));
    }
    metadata.route_requests.push_back(std::move(request));
  }

  if (metadata_has_legacy_logical_payloads(version)) {
    skip_bytes(in,
               checked_byte_count(logical_cell_count,
                                  3 * sizeof(std::uint64_t),
                                  "logical cells"),
               "logical cells");
    skip_bytes(in,
               checked_byte_count(logical_net_count,
                                  4 * sizeof(std::uint64_t),
                                  "logical nets"),
               "logical nets");
    skip_bytes(in,
               checked_byte_count(logical_port_instance_count,
                                  7 * sizeof(std::uint64_t),
                                  "logical port instances"),
               "logical port instances");
  } else {
    metadata.logical_net_name_strings.reserve(
        checked_size_count(logical_net_count, "logical net count"));
    for (std::uint64_t i = 0; i < logical_net_count; ++i) {
      const std::uint64_t name_string =
          read_u64(in, "logical net name string");
      (void)string_at(metadata, name_string);
      metadata.logical_net_name_strings.push_back(name_string);
    }
    for (const MetadataRouteRequest& request : metadata.route_requests) {
      if (request.logical_net_index == kNoEndpointPip) {
        continue;
      }
      if (request.logical_net_index >=
          metadata.logical_net_name_strings.size()) {
        throw std::runtime_error(
            "metadata v8 route request references an invalid logical net");
      }
      const std::uint64_t name_string =
          metadata.logical_net_name_strings[static_cast<std::size_t>(
              request.logical_net_index)];
      if (string_at(metadata, name_string) != request.net) {
        throw std::runtime_error(
            "metadata v8 physical/logical net-name correlation mismatch");
      }
    }
  }
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t),
                                "blocked nodes"),
             "blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count, sizeof(std::uint64_t),
                                "sink stop nodes"),
             "sink stop nodes");
  if (metadata_has_legacy_logical_payloads(version)) {
    skip_bytes(in, physical_netlist_byte_count, "physical netlist bytes");
    skip_bytes(in, logical_netlist_byte_count, "logical netlist bytes");
  }

  char trailing = 0;
  in.read(&trailing, 1);
  if (in.gcount() != 0) {
    throw std::runtime_error("metadata has trailing bytes");
  }
  if (!in.eof()) {
    throw std::runtime_error("failed while checking metadata end of file");
  }

  return metadata;
}

bool take_first(bool& seen) {
  if (seen) return false;
  seen = true;
  return true;
}

std::uint32_t intern_phys_string(
    std::string text,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  const auto found = string_to_index.find(text);
  if (found != string_to_index.end()) return found->second;
  if (strings.size() >= std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("PhysicalNetlist strList exceeds uint32_t");
  }
  const std::uint32_t index = static_cast<std::uint32_t>(strings.size());
  strings.push_back(std::move(text));
  string_to_index.emplace(strings.back(), index);
  return index;
}

RouteSitePin parse_route_site_pin(JsonCursor& cursor) {
  RouteSitePin pin;
  bool have_node = false;
  bool have_site = false;
  bool have_pin = false;
  bool have_reached = false;
  cursor.expect('{', "expected site pin object");
  if (!cursor.consume('}')) {
    while (true) {
      const std::string key = cursor.parse_string();
      cursor.expect(':', "expected ':' in site pin object");
      if (key == "node" && take_first(have_node)) {
        pin.node = cursor.parse_int("node");
      } else if (key == "site" && take_first(have_site)) {
        pin.site = cursor.parse_string();
      } else if (key == "pin" && take_first(have_pin)) {
        pin.pin = cursor.parse_string();
      } else if (key == "reached" && take_first(have_reached)) {
        pin.reached = cursor.parse_bool();
      } else {
        cursor.skip_value();
      }
      if (cursor.consume('}')) break;
      cursor.expect(',', "expected ',' in site pin object");
    }
  }
  if (!have_node) throw std::runtime_error("missing JSON key: node");
  if (!have_site) throw std::runtime_error("missing JSON key: site");
  if (!have_pin) throw std::runtime_error("missing JSON key: pin");
  return pin;
}

RouteEdge parse_route_edge(
    JsonCursor& cursor,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  RouteEdge edge;
  bool have_from = false;
  bool have_to = false;
  bool have_csr_edge = false;
  bool have_tile = false;
  bool have_wire0 = false;
  bool have_wire1 = false;
  bool have_forward = false;
  cursor.expect('{', "expected route edge object");
  if (!cursor.consume('}')) {
    while (true) {
      const std::string key = cursor.parse_string();
      cursor.expect(':', "expected ':' in route edge object");
      if (key == "from" && take_first(have_from)) {
        edge.from = cursor.parse_int("from");
      } else if (key == "to" && take_first(have_to)) {
        edge.to = cursor.parse_int("to");
      } else if (key == "csr_edge" && take_first(have_csr_edge)) {
        edge.csr_edge = cursor.parse_u64("csr_edge");
      } else if (key == "tile" && take_first(have_tile)) {
        edge.tile = intern_phys_string(
            cursor.parse_string(), strings, string_to_index);
      } else if (key == "wire0" && take_first(have_wire0)) {
        edge.wire0 = intern_phys_string(
            cursor.parse_string(), strings, string_to_index);
      } else if (key == "wire1" && take_first(have_wire1)) {
        edge.wire1 = intern_phys_string(
            cursor.parse_string(), strings, string_to_index);
      } else if (key == "forward" && take_first(have_forward)) {
        edge.forward = cursor.parse_bool();
      } else if (key == "attachment" &&
                 take_first(edge.attachment_field_present)) {
        if (!cursor.consume_null()) {
          edge.attachment = cursor.parse_u64("attachment");
        }
      } else if (key == "site" && take_first(edge.site_field_present)) {
        if (!cursor.consume_null()) {
          edge.site = intern_phys_string(
              cursor.parse_string(), strings, string_to_index);
        }
      } else {
        cursor.skip_value();
      }
      if (cursor.consume('}')) break;
      cursor.expect(',', "expected ',' in route edge object");
    }
  }
  if (!have_from) throw std::runtime_error("missing JSON key: from");
  if (!have_to) throw std::runtime_error("missing JSON key: to");
  if (!have_csr_edge) throw std::runtime_error("missing JSON key: csr_edge");
  if (!have_tile) throw std::runtime_error("missing JSON key: tile");
  if (!have_wire0) throw std::runtime_error("missing JSON key: wire0");
  if (!have_wire1) throw std::runtime_error("missing JSON key: wire1");
  return edge;
}

template <typename ParseElement>
void parse_json_array(JsonCursor& cursor, ParseElement&& parse_element) {
  cursor.expect('[', "expected JSON array");
  if (cursor.consume(']')) return;
  while (true) {
    parse_element();
    if (cursor.consume(']')) return;
    cursor.expect(',', "expected ',' in JSON array");
  }
}

NetRoute parse_route_line(
    std::string_view line,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  JsonCursor cursor(line);
  NetRoute route;
  bool have_artifact_pair_id = false;
  bool have_net = false;
  bool have_routed = false;
  bool have_sources = false;
  bool have_sinks = false;
  bool have_edges = false;
  cursor.expect('{', "expected route object");
  if (!cursor.consume('}')) {
    while (true) {
      const std::string key = cursor.parse_string();
      cursor.expect(':', "expected ':' in route object");
      if (key == "artifact_pair_id" &&
          take_first(have_artifact_pair_id)) {
        route.artifact_pair_id =
            routing::interchange::parse_interchange_artifact_pair_id(
                cursor.parse_string());
      } else if (key == "net" && take_first(have_net)) {
        route.net = cursor.parse_string();
      } else if (key == "routed" && take_first(have_routed)) {
        route.routed = cursor.parse_bool();
      } else if (key == "sources" && take_first(have_sources)) {
        parse_json_array(cursor, [&] {
          route.sources.push_back(parse_route_site_pin(cursor));
        });
      } else if (key == "sinks" && take_first(have_sinks)) {
        parse_json_array(cursor, [&] {
          route.sinks.push_back(parse_route_site_pin(cursor));
        });
      } else if (key == "edges" && take_first(have_edges)) {
        parse_json_array(cursor, [&] {
          route.edges.push_back(
              parse_route_edge(cursor, strings, string_to_index));
        });
      } else {
        cursor.skip_value();
      }
      if (cursor.consume('}')) break;
      cursor.expect(',', "expected ',' in route object");
    }
  }
  cursor.require_end();
  if (!have_net) throw std::runtime_error("missing JSON key: net");
  if (!have_sources) throw std::runtime_error("missing JSON key: sources");
  if (!have_sinks) throw std::runtime_error("missing JSON key: sinks");
  if (!have_edges) throw std::runtime_error("missing JSON key: edges");
  return route;
}

struct RouteIndexHeader {
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::string net;
};

RouteIndexHeader parse_route_index_header(std::string_view line) {
  JsonCursor cursor(line);
  RouteIndexHeader header;
  bool have_artifact_pair_id = false;
  bool have_net = false;
  cursor.expect('{', "expected route object");
  if (!cursor.consume('}')) {
    while (true) {
      const std::string key = cursor.parse_string();
      cursor.expect(':', "expected ':' in route object");
      if (key == "artifact_pair_id" &&
          take_first(have_artifact_pair_id)) {
        header.artifact_pair_id =
            routing::interchange::parse_interchange_artifact_pair_id(
                cursor.parse_string());
      } else if (key == "net" && take_first(have_net)) {
        header.net = cursor.parse_string();
      } else {
        cursor.skip_value();
      }
      if (cursor.consume('}')) break;
      cursor.expect(',', "expected ',' in route object");
    }
  }
  cursor.require_end();
  if (!have_net) throw std::runtime_error("missing JSON key: net");
  return header;
}

struct RouteIndexEntry {
  std::streampos offset{};
  std::size_t line_number = 0;
  std::size_t line_size = 0;
  std::uint64_t line_hash = 0;
  bool present = false;
  bool reconstructed = false;
};

struct RouteIndex {
  std::vector<RouteIndexEntry> entries;
  std::unordered_map<std::string_view, std::size_t> request_by_net;
  std::ifstream routes_in;
};

std::uint64_t hash_route_line(std::string_view line) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : line) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

RouteIndex index_routes_jsonl(
    const std::filesystem::path& path,
    const RoutingMetadataSummary& metadata,
    const routing::interchange::InterchangePublicationSnapshot& snapshot) {
  RouteIndex index;
  index.entries.resize(metadata.route_requests.size());
  index.request_by_net.reserve(metadata.route_requests.size());
  for (std::size_t i = 0; i < metadata.route_requests.size(); ++i) {
    const std::string& net = metadata.route_requests[i].net;
    if (!index.request_by_net.emplace(std::string_view(net), i).second) {
      throw std::runtime_error("metadata contains duplicate route request: " +
                               net);
    }
  }

  index.routes_in.open(path, std::ios::binary);
  if (!index.routes_in) {
    throw std::runtime_error("could not open routes file: " + path.string());
  }
  std::size_t line_number = 0;
  std::size_t route_count = 0;
  std::string line;
  while (true) {
    const std::streampos offset = index.routes_in.tellg();
    if (!std::getline(index.routes_in, line)) break;
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    RouteIndexHeader header;
    try {
      header = parse_route_index_header(line);
    } catch (const std::exception& ex) {
      throw std::runtime_error("invalid route JSON on line " +
                               std::to_string(line_number) + ": " + ex.what());
    }
    const auto request = index.request_by_net.find(header.net);
    if (request == index.request_by_net.end()) {
      throw std::runtime_error(
          "route file contains net not present in metadata: " + header.net);
    }
    RouteIndexEntry& entry = index.entries[request->second];
    if (entry.present) {
      throw std::runtime_error("duplicate route entry for net: " + header.net);
    }
    routing::interchange::require_matching_interchange_pair_ids(
        header.artifact_pair_id, metadata.artifact_pair_id,
        snapshot.generation);
    entry.offset = offset;
    entry.line_number = line_number;
    entry.line_size = line.size();
    entry.line_hash = hash_route_line(line);
    entry.present = true;
    ++route_count;
  }
  if (!index.routes_in.eof()) {
    throw std::runtime_error("failed while reading routes file: " +
                             path.string());
  }
  if (route_count == 0) {
    throw std::runtime_error("routes file is empty: " + path.string());
  }
  for (std::size_t i = 0; i < index.entries.size(); ++i) {
    if (!index.entries[i].present) {
      throw std::runtime_error("metadata route is missing from route file: " +
                               metadata.route_requests[i].net);
    }
  }
  return index;
}

struct RouteValidationContext {
  std::unordered_map<std::uint64_t, std::size_t> endpoint_pip_by_csr_edge;
};

RouteValidationContext build_route_validation_context(
    const RoutingMetadataSummary& metadata) {
  RouteValidationContext context;
  context.endpoint_pip_by_csr_edge.reserve(metadata.endpoint_pips.size());
  for (std::size_t index = 0; index < metadata.endpoint_pips.size(); ++index) {
    const MetadataEndpointPip& endpoint = metadata.endpoint_pips[index];
    if (!context.endpoint_pip_by_csr_edge.emplace(endpoint.csr_edge, index)
             .second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }
  }
  return context;
}

void validate_route_against_metadata(
    const NetRoute& route,
    const MetadataRouteRequest& request,
    const RoutingMetadataSummary& metadata,
    const RouteValidationContext& context,
    const std::vector<std::string>& phys_strings) {
    const std::string& net = route.net;
    const auto route_string = [&](std::uint32_t index) -> const std::string& {
      if (index == kNoRouteString || index >= phys_strings.size()) {
        throw std::runtime_error(
            "route references an invalid PhysicalNetlist string index");
      }
      return phys_strings[index];
    };
    if (route.sources.size() != request.sources.size()) {
      std::ostringstream out;
      out << "route source count for " << net << " is "
          << route.sources.size() << " but metadata expects "
          << request.sources.size();
      throw std::runtime_error(out.str());
    }
    if (route.sinks.size() != request.sinks.size()) {
      std::ostringstream out;
      out << "route sink count for " << net << " is " << route.sinks.size()
          << " but metadata expects " << request.sinks.size();
      throw std::runtime_error(out.str());
    }
    const auto require_same_endpoint = [&](const RouteSitePin& actual,
                                           const RouteSitePin& expected,
                                           const char* role,
                                           std::size_t index) {
      if (actual.node != expected.node || actual.site != expected.site ||
          actual.pin != expected.pin) {
        throw std::runtime_error(
            "route " + std::string(role) + " " +
            std::to_string(index) + " does not match metadata for net " +
            net);
      }
    };
    for (std::size_t index = 0; index < route.sources.size(); ++index) {
      require_same_endpoint(route.sources[index], request.sources[index],
                            "source", index);
    }
    for (std::size_t index = 0; index < route.sinks.size(); ++index) {
      require_same_endpoint(route.sinks[index], request.sinks[index],
                            "sink", index);
    }

    std::unordered_set<std::uint64_t> authorized_source_attachments;
    std::unordered_set<std::uint64_t> authorized_reached_sink_attachments;
    std::unordered_map<int, std::uint64_t> source_attachment_by_node;
    authorized_source_attachments.reserve(request.sources.size());
    authorized_reached_sink_attachments.reserve(request.sinks.size());
    source_attachment_by_node.reserve(request.sources.size());
    for (std::size_t index = 0; index < request.sources.size(); ++index) {
      const RouteSitePin& source = request.sources[index];
      if (source.endpoint_pip_index == kNoEndpointPip) {
        continue;
      }
      authorized_source_attachments.insert(source.endpoint_pip_index);
      const auto inserted = source_attachment_by_node.emplace(
          source.node, source.endpoint_pip_index);
      if (!inserted.second &&
          inserted.first->second != source.endpoint_pip_index) {
        throw std::runtime_error(
            "metadata has ambiguous source attachments for one node in net " +
            net);
      }
    }
    for (std::size_t index = 0; index < request.sinks.size(); ++index) {
      if (route.sinks[index].reached &&
          request.sinks[index].endpoint_pip_index != kNoEndpointPip) {
        authorized_reached_sink_attachments.insert(
            request.sinks[index].endpoint_pip_index);
      }
    }

    std::unordered_map<int, const RouteEdge*> incoming_by_node;
    std::unordered_map<int, std::vector<const RouteEdge*>> outgoing_by_node;
    std::unordered_set<std::uint64_t> node_pairs;
    std::unordered_set<std::uint64_t> csr_edges;
    std::unordered_set<std::uint64_t> used_attachments;
    std::vector<std::uint64_t> used_attachment_order;
    incoming_by_node.reserve(route.edges.size());
    outgoing_by_node.reserve(route.edges.size());
    node_pairs.reserve(route.edges.size());
    csr_edges.reserve(route.edges.size());
    used_attachments.reserve(route.edges.size());
    used_attachment_order.reserve(
        std::min(route.edges.size(), metadata.endpoint_pips.size()));

    for (const RouteEdge& edge : route.edges) {
      if (edge.from < 0 || edge.to < 0 || edge.from == edge.to) {
        throw std::runtime_error("route contains an invalid edge for net " +
                                 net);
      }
      if (static_cast<std::uint64_t>(edge.from) >= metadata.node_count ||
          static_cast<std::uint64_t>(edge.to) >= metadata.node_count) {
        throw std::runtime_error(
            "route edge references an invalid node for net " + net);
      }
      if (edge.csr_edge >= metadata.edge_attr_count) {
        throw std::runtime_error(
            "route edge references an invalid CSR edge for net " + net);
      }
      const std::uint64_t node_pair =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge.from))
           << 32) |
          static_cast<std::uint32_t>(edge.to);
      if (!node_pairs.insert(node_pair).second ||
          !csr_edges.insert(edge.csr_edge).second) {
        throw std::runtime_error(
            "route contains a duplicate edge for net " + net);
      }
      const auto incoming = incoming_by_node.emplace(edge.to, &edge);
      if (!incoming.second && incoming.first->second->from != edge.from) {
        throw std::runtime_error(
            "route drives one node from multiple parents: " + net);
      }
      outgoing_by_node[edge.from].push_back(&edge);

      if (metadata_has_endpoint_pips(metadata.version) &&
          (!edge.attachment_field_present || !edge.site_field_present)) {
        throw std::runtime_error(
            "v7 route edge is missing attachment/site fields for net " + net);
      }
      const bool has_site = edge.site != kNoRouteString;
      if (edge.attachment.has_value() != has_site) {
        throw std::runtime_error(
            "route edge must pair attachment and site for net " + net);
      }

      const auto endpoint_for_edge =
          context.endpoint_pip_by_csr_edge.find(edge.csr_edge);
      if (endpoint_for_edge == context.endpoint_pip_by_csr_edge.end()) {
        if (edge.attachment.has_value() || has_site) {
          throw std::runtime_error(
              "conventional route edge must not carry attachment/site for net " +
              net);
        }
        continue;
      }

      const std::size_t expected_index = endpoint_for_edge->second;
      if (!edge.attachment.has_value() || !has_site) {
        throw std::runtime_error(
            "endpoint attachment edge is encoded as conventional for net " +
            net);
      }
      if (*edge.attachment != expected_index ||
          *edge.attachment >= metadata.endpoint_pips.size()) {
        throw std::runtime_error(
            "route attachment index does not match its CSR edge for net " +
            net);
      }
      if (!used_attachments.insert(*edge.attachment).second) {
        throw std::runtime_error(
            "route reuses one endpoint attachment in net " + net);
      }
      used_attachment_order.push_back(*edge.attachment);

      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[expected_index];
      if (edge.from != endpoint.from || edge.to != endpoint.to ||
          edge.csr_edge != endpoint.csr_edge ||
          route_string(edge.tile) != string_at(metadata, endpoint.tile_string) ||
          route_string(edge.wire0) != string_at(metadata, endpoint.wire0_string) ||
          route_string(edge.wire1) != string_at(metadata, endpoint.wire1_string) ||
          edge.forward != endpoint.forward ||
          route_string(edge.site) != string_at(metadata, endpoint.site_string)) {
        throw std::runtime_error(
            "route attachment does not exactly match sparse metadata for net " +
            net);
      }

      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        if (authorized_source_attachments.count(*edge.attachment) == 0) {
          throw std::runtime_error(
              "route source attachment belongs to a different endpoint for net " +
              net);
        }
      } else if (authorized_reached_sink_attachments.count(
                     *edge.attachment) == 0) {
        throw std::runtime_error(
            "route sink attachment belongs to a different or unreached "
            "endpoint for net " + net);
      }
    }

    for (std::uint64_t index : used_attachment_order) {
      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[static_cast<std::size_t>(index)];
      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        const auto corridor = incoming_by_node.find(endpoint.from);
        const auto attachment_children = outgoing_by_node.find(endpoint.from);
        const auto root_children = outgoing_by_node.find(endpoint.endpoint_node);
        const bool root_contains_corridor =
            corridor != incoming_by_node.end() &&
            root_children != outgoing_by_node.end() &&
            std::find(root_children->second.begin(),
                      root_children->second.end(), corridor->second) !=
                root_children->second.end();
        const bool source_contains_attachment =
            attachment_children != outgoing_by_node.end() &&
            std::any_of(
                attachment_children->second.begin(),
                attachment_children->second.end(),
                [&](const RouteEdge* edge) {
                  return edge->attachment == index;
                });
        // The filtered graph makes endpoint_node source-exclusive and gives
        // from exactly one incoming edge: this corridor. Extra outgoing edges
        // are same-net source fanout rather than attachment transit.
        if (corridor == incoming_by_node.end() ||
            corridor->second->from != endpoint.endpoint_node ||
            corridor->second->attachment.has_value() ||
            !root_contains_corridor || !source_contains_attachment ||
            incoming_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "source attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      } else {
        const auto corridor = outgoing_by_node.find(endpoint.to);
        if (corridor == outgoing_by_node.end() ||
            corridor->second.size() != 1 ||
            corridor->second.front()->to != endpoint.endpoint_node ||
            corridor->second.front()->attachment.has_value() ||
            outgoing_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "sink attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      }
    }

    // Endpoint metadata authorizes an audited pseudo-PIP when a route needs
    // that BITSLICE crossing.  A route that stays entirely on conventional CSR
    // edges is already represented exactly and need not consume the optional
    // attachment.  Used attachments remain subject to the strict corridor and
    // no-transit checks above.
}

struct WordAlignedPayload {
  kj::Array<capnp::word> words;
  std::size_t decoded_bytes = 0;

  std::size_t word_count() const {
    return decoded_bytes / sizeof(capnp::word);
  }
};

void append_word_aligned_bytes(WordAlignedPayload& payload,
                               const std::uint8_t* data,
                               std::size_t byte_count,
                               const std::filesystem::path& path) {
  if (byte_count == 0) return;
  constexpr std::size_t kWordBytes = sizeof(capnp::word);
  constexpr std::size_t kMaximumWordCount =
      std::numeric_limits<std::size_t>::max() / kWordBytes;
  if (byte_count >
      std::numeric_limits<std::size_t>::max() - payload.decoded_bytes) {
    throw std::runtime_error("decoded input is too large: " + path.string());
  }
  const std::size_t old_byte_count = payload.decoded_bytes;
  const std::size_t new_byte_count = old_byte_count + byte_count;
  if (new_byte_count >
      std::numeric_limits<std::size_t>::max() - (kWordBytes - 1)) {
    throw std::runtime_error(
        "decoded Cap'n Proto word count overflows size_t: " + path.string());
  }
  const std::size_t required_words =
      (new_byte_count + kWordBytes - 1) / kWordBytes;
  if (required_words > kMaximumWordCount) {
    throw std::runtime_error("decoded input is too large: " + path.string());
  }
  if (required_words > payload.words.size()) {
    std::size_t grown_words = payload.words.size();
    if (grown_words == 0) {
      grown_words = required_words;
    } else if (grown_words > kMaximumWordCount / 2) {
      grown_words = kMaximumWordCount;
    } else {
      grown_words *= 2;
    }
    grown_words = std::max(grown_words, required_words);
    kj::Array<capnp::word> grown = kj::heapArray<capnp::word>(grown_words);
    if (old_byte_count != 0) {
      // old_byte_count may end in a partial word; carry those bytes too.
      std::memcpy(grown.begin(), payload.words.begin(), old_byte_count);
    }
    payload.words = kj::mv(grown);
  }
  std::memcpy(reinterpret_cast<std::uint8_t*>(payload.words.begin()) +
                  old_byte_count,
              data, byte_count);
  payload.decoded_bytes = new_byte_count;
}

WordAlignedPayload read_gzip_or_plain_words(
    const std::filesystem::path& path) {
  WordAlignedPayload payload;
  routing::interchange::read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t byte_count) {
        append_word_aligned_bytes(payload, data, byte_count, path);
      });
  if (payload.decoded_bytes == 0) {
    throw std::runtime_error("input file is empty: " + path.string());
  }
  if (payload.decoded_bytes % sizeof(capnp::word) != 0) {
    throw std::runtime_error(
        "decoded physical netlist is not Cap'n Proto word-aligned");
  }
  return payload;
}

class GzipOutputStream final : public kj::OutputStream {
 public:
  using kj::OutputStream::write;

  explicit GzipOutputStream(const std::filesystem::path& path)
      : path_(path), file_(gzopen(path.string().c_str(), "wb6")) {
    if (file_ == nullptr) {
      throw std::runtime_error("could not open output file: " + path.string());
    }
  }

  ~GzipOutputStream() noexcept override {
    if (file_ != nullptr) (void)gzclose(file_);
  }

  void write(const void* buffer, std::size_t size) override {
    const auto* bytes = static_cast<const std::uint8_t*>(buffer);
    std::size_t offset = 0;
    // gzwrite() reports its byte count as int.  Bounded writes keep every
    // return value representable and do not coalesce Cap'n Proto segments.
    constexpr std::size_t kWriteChunkBytes = 64ULL * 1024ULL * 1024ULL;
    while (offset < size) {
      const unsigned int chunk_size = static_cast<unsigned int>(
          std::min<std::size_t>(size - offset, kWriteChunkBytes));
      const int written = gzwrite(file_, bytes + offset, chunk_size);
      if (written != static_cast<int>(chunk_size)) {
        throw write_error();
      }
      offset += static_cast<std::size_t>(written);
    }
  }

  void finish() {
    if (file_ == nullptr) return;
    const int close_status = gzclose(file_);
    file_ = nullptr;
    if (close_status != Z_OK) {
      const int saved_errno = errno;
      const char* zlib_message = zError(close_status);
      const std::string message =
          close_status == Z_ERRNO
              ? std::strerror(saved_errno)
              : (zlib_message == nullptr ? "zlib error" : zlib_message);
      throw std::runtime_error("failed while closing " + path_.string() +
                               ": " + message);
    }
  }

 private:
  std::runtime_error write_error() {
      int zlib_error = Z_OK;
      const char* raw_message = gzerror(file_, &zlib_error);
      // gzerror's pointer belongs to the gzFile and becomes invalid at close.
      std::string message = raw_message == nullptr ? std::string() : raw_message;
      const int saved_errno = errno;
      (void)gzclose(file_);
      file_ = nullptr;
      if (message.empty()) {
        if (zlib_error == Z_ERRNO) {
          message = std::strerror(saved_errno);
        } else if (const char* fallback = zError(zlib_error);
                   fallback != nullptr) {
          message = fallback;
        }
      }
      if (message.empty()) {
        message = "zlib write error " + std::to_string(zlib_error);
      }
      return std::runtime_error("failed while writing " + path_.string() +
                                ": " + message);
  }

  std::filesystem::path path_;
  gzFile file_ = nullptr;
};

std::vector<std::string> copy_string_list(
    capnp::List<capnp::Text>::Builder str_list,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  std::vector<std::string> strings;
  strings.reserve(str_list.size());
  for (std::uint32_t i = 0; i < str_list.size(); ++i) {
    capnp::Text::Builder text = str_list[i];
    strings.emplace_back(text.cStr(), text.size());
    string_to_index.emplace(strings.back(), i);
  }
  return strings;
}

std::string phys_string_at(const std::vector<std::string>& strings, std::uint32_t index) {
  if (index >= strings.size()) throw std::runtime_error("PhysicalNetlist string index out of range");
  return strings[static_cast<std::size_t>(index)];
}

void copy_route_branch(PhysicalNetlist::PhysNetlist::RouteBranch::Builder source,
                       PhysicalNetlist::PhysNetlist::RouteBranch::Builder destination) {
  auto source_segment = source.getRouteSegment();
  auto destination_segment = destination.initRouteSegment();
  if (source_segment.isBelPin()) {
    auto source_bel_pin = source_segment.getBelPin();
    auto destination_bel_pin = destination_segment.initBelPin();
    destination_bel_pin.setSite(source_bel_pin.getSite());
    destination_bel_pin.setBel(source_bel_pin.getBel());
    destination_bel_pin.setPin(source_bel_pin.getPin());
  } else if (source_segment.isSitePin()) {
    auto source_site_pin = source_segment.getSitePin();
    auto destination_site_pin = destination_segment.initSitePin();
    destination_site_pin.setSite(source_site_pin.getSite());
    destination_site_pin.setPin(source_site_pin.getPin());
  } else if (source_segment.isPip()) {
    auto source_pip = source_segment.getPip();
    auto destination_pip = destination_segment.initPip();
    destination_pip.setTile(source_pip.getTile());
    destination_pip.setWire0(source_pip.getWire0());
    destination_pip.setWire1(source_pip.getWire1());
    destination_pip.setForward(source_pip.getForward());
    destination_pip.setIsFixed(source_pip.getIsFixed());
    if (source_pip.isSite()) {
      destination_pip.setSite(source_pip.getSite());
    } else {
      destination_pip.setNoSite();
    }
  } else if (source_segment.isSitePIP()) {
    auto source_site_pip = source_segment.getSitePIP();
    auto destination_site_pip = destination_segment.initSitePIP();
    destination_site_pip.setSite(source_site_pip.getSite());
    destination_site_pip.setBel(source_site_pip.getBel());
    destination_site_pip.setPin(source_site_pip.getPin());
    destination_site_pip.setIsFixed(source_site_pip.getIsFixed());
    if (source_site_pip.isIsInverting()) {
      destination_site_pip.setIsInverting(source_site_pip.getIsInverting());
    } else {
      destination_site_pip.setInverts();
    }
  } else {
    throw std::runtime_error("unsupported PhysicalNetlist route segment");
  }

  auto source_children = source.getBranches();
  auto destination_children =
      destination.initBranches(static_cast<std::uint32_t>(source_children.size()));
  for (std::uint32_t i = 0; i < source_children.size(); ++i) {
    copy_route_branch(source_children[i], destination_children[i]);
  }
}

StubBranchStore snapshot_top_level_stubs(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder stubs,
    const std::vector<std::string>& strings,
    const std::string& net_name) {
  StubBranchStore store;
  store.stubs.reserve(stubs.size());
  store.by_key.reserve(stubs.size());
  store.cursor_by_key.reserve(stubs.size());
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    auto stub = stubs[i];
    auto segment = stub.getRouteSegment();
    if (!segment.isSitePin()) {
      throw std::runtime_error("non-sitePin stub found in net: " + net_name);
    }
    auto site_pin = segment.getSitePin();
    SitePinKey key{phys_string_at(strings, site_pin.getSite()),
                   phys_string_at(strings, site_pin.getPin())};
    StoredStubBranch stored;
    stored.key = key;
    stored.branch_index = i;
    const std::size_t index = store.stubs.size();
    store.stubs.push_back(std::move(stored));
    store.by_key[key].push_back(index);
  }
  return store;
}

std::uint32_t consume_stub_branch(StubBranchStore& store,
                                  const SitePinKey& key,
                                  const std::string& net_name) {
  const auto found = store.by_key.find(key);
  if (found == store.by_key.end()) {
    throw std::runtime_error("routed sink was not present as a stub in net: " + net_name);
  }

  std::size_t& cursor = store.cursor_by_key[key];
  if (cursor >= found->second.size()) {
    throw std::runtime_error("routed sink used more times than its stubs in net: " + net_name);
  }
  const std::size_t stub_index = found->second[cursor++];
  store.stubs[stub_index].consumed = true;
  return store.stubs[stub_index].branch_index;
}

void collect_site_pin_branches(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    const std::vector<std::string>& strings,
    std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>& out) {
  auto segment = branch.getRouteSegment();
  if (segment.isSitePin()) {
    auto site_pin = segment.getSitePin();
    out.push_back({{phys_string_at(strings, site_pin.getSite()),
                    phys_string_at(strings, site_pin.getPin())},
                   branch});
  }

  auto children = branch.getBranches();
  for (std::uint32_t i = 0; i < children.size(); ++i) {
    collect_site_pin_branches(children[i], strings, out);
  }
}

struct RouteTables {
  std::unordered_map<int, std::vector<const RouteEdge*>> children_by_node;
  std::unordered_map<int, std::vector<const RouteSitePin*>> sinks_by_node;
  std::unordered_map<SitePinKey, int, SitePinKeyHash> source_node_by_pin;
  std::size_t edge_count = 0;
};

RouteTables build_route_tables(const NetRoute& route) {
  RouteTables tables;
  tables.children_by_node.reserve(route.edges.size());
  tables.sinks_by_node.reserve(route.sinks.size());
  tables.source_node_by_pin.reserve(route.sources.size());
  std::unordered_map<int, int> parent_by_child;
  parent_by_child.reserve(route.edges.size());

  for (const RouteEdge& edge : route.edges) {
    const auto parent = parent_by_child.find(edge.to);
    if (parent != parent_by_child.end() && parent->second != edge.from) {
      throw std::runtime_error("route drives one node from multiple parents: " + route.net);
    }
    parent_by_child[edge.to] = edge.from;
    tables.children_by_node[edge.from].push_back(&edge);
    ++tables.edge_count;
  }
  for (const RouteSitePin& source : route.sources) {
    if (source.node < 0) continue;
    SitePinKey key{source.site, source.pin};
    const auto [found, inserted] =
        tables.source_node_by_pin.emplace(key, source.node);
    if (!inserted && found->second != source.node) {
      throw std::runtime_error(
          "one source site pin maps to multiple nodes in route: " +
          route.net);
    }
  }
  for (const RouteSitePin& sink : route.sinks) {
    if (!sink.reached || sink.node < 0) continue;
    tables.sinks_by_node[sink.node].push_back(&sink);
  }
  return tables;
}

bool has_reached_sink(const NetRoute& route) {
  for (const RouteSitePin& sink : route.sinks) {
    if (sink.reached) {
      return true;
    }
  }
  return false;
}

bool top_level_stubs_are_site_pins(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder stubs) {
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    if (!stubs[i].getRouteSegment().isSitePin()) {
      return false;
    }
  }
  return true;
}

std::size_t insert_route_tree(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    int node,
    const NetRoute& route,
    const RouteTables& tables,
    StubBranchStore& stub_store,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder old_stubs) {
  using RouteBranch = PhysicalNetlist::PhysNetlist::RouteBranch;
  using EdgeList = std::vector<const RouteEdge*>;
  using SinkList = std::vector<const RouteSitePin*>;
  struct DfsFrame {
    int node = -1;
    capnp::List<RouteBranch>::Builder output;
    const EdgeList* edges = nullptr;
    const SinkList* sinks = nullptr;
    std::size_t next_edge = 0;
    std::size_t next_sink = 0;
    std::uint32_t output_index = 0;
  };

  std::vector<DfsFrame> stack;
  std::unordered_set<int> active_path;
  const auto push_node = [&](RouteBranch::Builder output_branch,
                             int output_node) {
    if (!active_path.insert(output_node).second) {
      throw std::runtime_error("route tree has a cycle in net: " + route.net);
    }
    const auto children = tables.children_by_node.find(output_node);
    const auto sinks = tables.sinks_by_node.find(output_node);
    const EdgeList* edge_list =
        children == tables.children_by_node.end() ? nullptr
                                                  : &children->second;
    const SinkList* sink_list =
        sinks == tables.sinks_by_node.end() ? nullptr : &sinks->second;
    const std::size_t edge_count = edge_list == nullptr ? 0 : edge_list->size();
    const std::size_t sink_count = sink_list == nullptr ? 0 : sink_list->size();
    if (sink_count > std::numeric_limits<std::uint32_t>::max() ||
        edge_count >
            std::numeric_limits<std::uint32_t>::max() - sink_count) {
      throw std::runtime_error("route node has too many branches in net: " +
                               route.net);
    }
    const std::size_t branch_count = edge_count + sink_count;
    if (branch_count == 0) {
      active_path.erase(output_node);
      return;
    }
    if (output_branch.getBranches().size() != 0) {
      throw std::runtime_error(
          "source route branch already has children in net: " + route.net);
    }
    DfsFrame frame;
    frame.node = output_node;
    frame.output = output_branch.initBranches(
        static_cast<std::uint32_t>(branch_count));
    frame.edges = edge_list;
    frame.sinks = sink_list;
    stack.push_back(frame);
  };

  std::size_t emitted_edges = 0;
  push_node(branch, node);
  while (!stack.empty()) {
    DfsFrame& frame = stack.back();
    const std::size_t edge_count =
        frame.edges == nullptr ? 0 : frame.edges->size();
    if (frame.next_edge < edge_count) {
      const RouteEdge& edge = *(*frame.edges)[frame.next_edge++];
      auto child = frame.output[frame.output_index++];
      auto pip = child.initRouteSegment().initPip();
      pip.setTile(edge.tile);
      pip.setWire0(edge.wire0);
      pip.setWire1(edge.wire1);
      pip.setIsFixed(false);
      pip.setForward(edge.forward);
      if (edge.attachment.has_value()) {
        pip.setSite(edge.site);
      } else {
        pip.setNoSite();
      }
      ++emitted_edges;
      push_node(child, edge.to);
      continue;
    }

    const std::size_t sink_count =
        frame.sinks == nullptr ? 0 : frame.sinks->size();
    if (frame.next_sink < sink_count) {
      const RouteSitePin& sink = *(*frame.sinks)[frame.next_sink++];
      auto child = frame.output[frame.output_index++];
      copy_route_branch(
          old_stubs[consume_stub_branch(
              stub_store, SitePinKey{sink.site, sink.pin}, route.net)],
          child);
      continue;
    }

    active_path.erase(frame.node);
    stack.pop_back();
  }
  return emitted_edges;
}

void write_routed_phys(const std::filesystem::path& input_phys,
                       const std::filesystem::path& output_phys,
                       const RoutingMetadataSummary& metadata,
                       const routing::interchange::InterchangePublicationSnapshot&
                           publication_snapshot,
                       RouteIndex& route_index,
                       bool allow_unrouted_stubs) {
  capnp::MallocMessageBuilder builder;
  {
    WordAlignedPayload payload = read_gzip_or_plain_words(input_phys);
    capnp::ReaderOptions reader_options;
    reader_options.traversalLimitInWords =
        std::numeric_limits<std::uint64_t>::max();
    reader_options.nestingLimit = 1 << 20;
    capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(payload.words.begin(), payload.word_count()),
        reader_options);
    builder.setRoot(reader.getRoot<PhysicalNetlist::PhysNetlist>());
  }
  auto netlist = builder.getRoot<PhysicalNetlist::PhysNetlist>();

  std::unordered_map<std::string, std::uint32_t> string_to_index;
  std::vector<std::string> strings =
      copy_string_list(netlist.getStrList(), string_to_index);

  const RouteValidationContext validation_context =
      build_route_validation_context(metadata);
  std::ifstream& routes_in = route_index.routes_in;

  auto phys_nets = netlist.getPhysNets();
  for (std::uint32_t net_index = 0; net_index < phys_nets.size(); ++net_index) {
    auto net = phys_nets[net_index];
    const std::string net_name = phys_string_at(strings, net.getName());
    const auto request_entry = route_index.request_by_net.find(net_name);
    if (request_entry == route_index.request_by_net.end()) continue;
    const std::size_t request_index = request_entry->second;
    RouteIndexEntry& indexed_route = route_index.entries[request_index];
    routes_in.clear();
    routes_in.seekg(indexed_route.offset);
    if (!routes_in) {
      throw std::runtime_error("failed to seek to route JSON line " +
                               std::to_string(indexed_route.line_number));
    }
    std::string route_line;
    if (!std::getline(routes_in, route_line)) {
      throw std::runtime_error("failed to reread route JSON line " +
                               std::to_string(indexed_route.line_number));
    }
    if (route_line.size() != indexed_route.line_size ||
        hash_route_line(route_line) != indexed_route.line_hash) {
      throw std::runtime_error(
          "routes file changed after it was indexed at line " +
          std::to_string(indexed_route.line_number));
    }
    NetRoute route;
    try {
      route = parse_route_line(route_line, strings, string_to_index);
    } catch (const std::exception& ex) {
      throw std::runtime_error(
          "invalid route JSON on line " +
          std::to_string(indexed_route.line_number) + ": " + ex.what());
    }
    std::string().swap(route_line);
    if (route.net != net_name) {
      throw std::runtime_error(
          "route file changed after indexing net: " + net_name);
    }
    routing::interchange::require_matching_interchange_pair_ids(
        route.artifact_pair_id, metadata.artifact_pair_id,
        publication_snapshot.generation);
    validate_route_against_metadata(
        route, metadata.route_requests[request_index], metadata,
        validation_context, strings);

    const bool reached_any_sink = has_reached_sink(route);
    if (!reached_any_sink) {
      if (!route.edges.empty()) {
        throw std::runtime_error("route has PIP edges but no reached sinks: " +
                                 net_name);
      }
      if (!allow_unrouted_stubs) {
        throw std::runtime_error("unrouted route has no reached sinks: " +
                                 net_name);
      }
      indexed_route.reconstructed = true;
      continue;
    }
    if (!route.routed && allow_unrouted_stubs &&
        !top_level_stubs_are_site_pins(net.getStubs())) {
      indexed_route.reconstructed = true;
      continue;
    }

    RouteTables tables = build_route_tables(route);

    auto old_stubs_orphan = net.disownStubs();
    auto old_stubs = old_stubs_orphan.get();
    StubBranchStore stub_store =
        snapshot_top_level_stubs(old_stubs, strings, net_name);

    std::vector<std::pair<SitePinKey, PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
        source_branches;
    auto sources = net.getSources();
    for (std::uint32_t i = 0; i < sources.size(); ++i) {
      collect_site_pin_branches(sources[i], strings, source_branches);
    }

    std::size_t emitted_edges = 0;
    std::unordered_set<int> emitted_source_nodes;
    emitted_source_nodes.reserve(source_branches.size());
    for (const auto& [source_key, source_branch] : source_branches) {
      const auto source_node_it = tables.source_node_by_pin.find(source_key);
      if (source_node_it == tables.source_node_by_pin.end()) continue;
      const int source_node = source_node_it->second;
      if (tables.children_by_node.find(source_node) == tables.children_by_node.end() &&
          tables.sinks_by_node.find(source_node) == tables.sinks_by_node.end()) {
        continue;
      }
      // Alternate or duplicate source site pins can legally resolve to the
      // same routing node. Emit that node's tree from the first physical root
      // only; inserting it below every alias duplicates PIPs and consumes the
      // same sink stubs more than once.
      if (!emitted_source_nodes.insert(source_node).second) {
        continue;
      }
      emitted_edges += insert_route_tree(source_branch,
                                         source_node,
                                         route,
                                         tables,
                                         stub_store,
                                         old_stubs);
    }

    if (emitted_edges != tables.edge_count) {
      std::ostringstream out;
      out << "emitted " << emitted_edges << " PIPs for " << net_name
          << " but route contains " << tables.edge_count;
      throw std::runtime_error(out.str());
    }
    const std::size_t expected_reached_stubs =
        static_cast<std::size_t>(std::count_if(
            route.sinks.begin(), route.sinks.end(),
            [](const RouteSitePin& sink) {
              return sink.reached;
            }));
    const std::size_t consumed_reached_stubs =
        static_cast<std::size_t>(std::count_if(
            stub_store.stubs.begin(), stub_store.stubs.end(),
            [](const StoredStubBranch& stub) { return stub.consumed; }));
    if (consumed_reached_stubs != expected_reached_stubs) {
      throw std::runtime_error(
          "one or more reached sinks were not attached to a routed source in " +
          net_name);
    }
    std::size_t remaining_stub_count = 0;
    for (const StoredStubBranch& stub : stub_store.stubs) {
      if (!stub.consumed) {
        if (!allow_unrouted_stubs) {
          throw std::runtime_error("unrouted stub remains in routed net: " + net_name);
        }
        ++remaining_stub_count;
      }
    }

    auto new_stubs = net.initStubs(static_cast<std::uint32_t>(remaining_stub_count));
    std::uint32_t stub_index = 0;
    for (const StoredStubBranch& stub : stub_store.stubs) {
      if (stub.consumed) continue;
      copy_route_branch(old_stubs[stub.branch_index], new_stubs[stub_index++]);
    }
    indexed_route.reconstructed = true;
  }

  for (std::size_t request_index = 0;
       request_index < route_index.entries.size(); ++request_index) {
    if (!route_index.entries[request_index].reconstructed) {
      throw std::runtime_error(
          "route net was not found in PhysicalNetlist: " +
          metadata.route_requests[request_index].net);
    }
  }

  auto new_str_list = netlist.initStrList(static_cast<std::uint32_t>(strings.size()));
  for (std::uint32_t i = 0; i < strings.size(); ++i) {
    new_str_list.set(i, strings[i]);
  }

  if (output_phys.has_parent_path()) {
    std::filesystem::create_directories(output_phys.parent_path());
  }
  GzipOutputStream gzip_output(output_phys);
  capnp::writeMessage(gzip_output, builder);
  gzip_output.finish();
}

[[maybe_unused]] void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys> "
         "[--allow-unrouted-stubs]\n";
}

}  // namespace

#ifndef ROUTES_TO_PHYS_TESTING
int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "-h" ||
                      std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }
    bool allow_unrouted_stubs = false;
    if (argc == 6 && std::string(argv[5]) == "--allow-unrouted-stubs") {
      allow_unrouted_stubs = true;
    } else if (argc != 5) {
      print_usage(argv[0]);
      return 1;
    }

    const std::filesystem::path input_phys = argv[1];
    const std::filesystem::path metadata_path = argv[2];
    const std::filesystem::path routes_path = argv[3];
    const std::filesystem::path output_phys = argv[4];

    const routing::interchange::InterchangePublicationSnapshot
        publication_snapshot =
            routing::interchange::snapshot_interchange_publication(
                metadata_path);
    RoutingMetadataSummary metadata = load_metadata_summary(metadata_path);
    RouteIndex routes = index_routes_jsonl(
        routes_path, metadata, publication_snapshot);
    routing::interchange::verify_interchange_publication(
        metadata_path, publication_snapshot);
    write_routed_phys(input_phys, output_phys, metadata,
                      publication_snapshot, routes, allow_unrouted_stubs);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
#endif
