#include <zlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

// Force gzip_io.hpp to return deliberately awkward fragments.  zlib normally
// fills the caller's one-megabyte buffer, whose size happens to be word
// aligned, so ordinary fixtures would not exercise the partial-word carry
// path in routes_to_phys.cpp.  Including zlib.h before defining the macro
// leaves the real declaration available to this forwarding shim.
namespace {

bool fragment_gzip_reads = false;
std::size_t fragment_gzip_read_index = 0;

int routes_to_phys_test_gzread(gzFile file, voidp buffer,
                               unsigned int byte_count) {
  static constexpr unsigned int kFragmentSizes[] = {
      1, 7, 2, 13, 5, 3, 17, 4, 11,
  };
  if (fragment_gzip_reads && byte_count != 0) {
    byte_count = std::min(
        byte_count,
        kFragmentSizes[fragment_gzip_read_index++ %
                       (sizeof(kFragmentSizes) / sizeof(kFragmentSizes[0]))]);
  }
  return ::gzread(file, buffer, byte_count);
}

}  // namespace

#define gzread routes_to_phys_test_gzread
#define ROUTES_TO_PHYS_TESTING 1
#include "../routes_to_phys.cpp"
#undef gzread

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(
    std::is_same<decltype(RouteEdge::tile), std::uint32_t>::value,
    "route edges must retain interned string-table indices, not tile strings");
static_assert(
    std::is_same<decltype(RouteEdge::wire0), std::uint32_t>::value,
    "route edges must retain interned string-table indices, not wire strings");

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
std::string require_runtime_error(Function&& function,
                                  std::string_view expected_text = {}) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (!expected_text.empty()) {
      require(message.find(expected_text) != std::string::npos,
              "error did not contain '" + std::string(expected_text) +
                  "': " + message);
    }
    return message;
  }
  throw std::runtime_error("operation unexpectedly succeeded");
}

template <typename Function>
void require_exception(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("operation unexpectedly succeeded");
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("routes-to-phys-test-" + std::to_string(nonce));
    if (!std::filesystem::create_directory(path_)) {
      throw std::runtime_error("could not create temporary test directory");
    }
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  std::filesystem::path file(std::string_view name) const {
    return path_ / std::string(name);
  }

 private:
  std::filesystem::path path_;
};

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "could not create " + path.string());
  if (!bytes.empty()) {
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  output.close();
  require(static_cast<bool>(output), "could not finish " + path.string());
}

void write_text(const std::filesystem::path& path, std::string_view text) {
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "could not create " + path.string());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.close();
  require(static_cast<bool>(output), "could not finish " + path.string());
}

void write_gzip(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  gzFile output = gzopen(path.string().c_str(), "wb6");
  require(output != nullptr, "could not create " + path.string());
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const unsigned int chunk = static_cast<unsigned int>(
        std::min<std::size_t>(bytes.size() - offset, 32749));
    const int written = gzwrite(output, bytes.data() + offset, chunk);
    require(written == static_cast<int>(chunk),
            "could not write " + path.string());
    offset += chunk;
  }
  require(gzclose(output) == Z_OK, "could not finish " + path.string());
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "could not read " + path.string());
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>());
}

void require_payload_equals(const WordAlignedPayload& payload,
                            const std::vector<std::uint8_t>& expected,
                            std::string_view description) {
  require(payload.decoded_bytes == expected.size(),
          std::string(description) + " decoded byte count changed");
  require(payload.word_count() ==
              expected.size() / sizeof(capnp::word),
          std::string(description) + " decoded word count changed");
  require(std::memcmp(payload.words.begin(), expected.data(), expected.size()) ==
              0,
          std::string(description) + " payload changed across fragments");
}

void test_fragmented_word_accumulation(const TemporaryDirectory& temporary) {
  std::vector<std::uint8_t> expected(8192);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] =
        static_cast<std::uint8_t>((index * 131U + index / 7U) & 0xffU);
  }

  const std::filesystem::path plain = temporary.file("fragmented.plain");
  const std::filesystem::path gzip = temporary.file("fragmented.gz");
  write_bytes(plain, expected);
  write_gzip(gzip, expected);

  fragment_gzip_reads = true;
  fragment_gzip_read_index = 0;
  require_payload_equals(read_gzip_or_plain_words(plain), expected,
                         "plain input");
  fragment_gzip_read_index = 0;
  require_payload_equals(read_gzip_or_plain_words(gzip), expected,
                         "gzip input");
  fragment_gzip_reads = false;

  std::vector<std::uint8_t> unaligned(expected.begin(), expected.end() - 1);
  const std::filesystem::path unaligned_plain =
      temporary.file("unaligned.plain");
  const std::filesystem::path unaligned_gzip =
      temporary.file("unaligned.gz");
  write_bytes(unaligned_plain, unaligned);
  write_gzip(unaligned_gzip, unaligned);
  require_runtime_error(
      [&] { (void)read_gzip_or_plain_words(unaligned_plain); },
      "word-aligned");
  require_runtime_error(
      [&] { (void)read_gzip_or_plain_words(unaligned_gzip); },
      "word-aligned");

  const std::filesystem::path empty_plain = temporary.file("empty.plain");
  const std::filesystem::path empty_gzip = temporary.file("empty.gz");
  write_bytes(empty_plain, {});
  write_gzip(empty_gzip, {});
  require_runtime_error(
      [&] { (void)read_gzip_or_plain_words(empty_plain); }, "empty");
  require_runtime_error(
      [&] { (void)read_gzip_or_plain_words(empty_gzip); }, "empty");

  const std::vector<std::uint8_t> encoded = read_bytes(gzip);
  require(encoded.size() > 8, "gzip fixture was unexpectedly short");
  const std::filesystem::path truncated = temporary.file("truncated.gz");
  write_bytes(truncated, std::vector<std::uint8_t>(encoded.begin(),
                                                   encoded.end() - 8));
  require_runtime_error(
      [&] { (void)read_gzip_or_plain_words(truncated); },
      "ended prematurely");
}

NetRoute parse_test_route(
    std::string_view json,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  return parse_route_line(json, strings, string_to_index);
}

void require_bad_route(std::string_view json) {
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint32_t> string_to_index;
  require_runtime_error(
      [&] { (void)parse_test_route(json, strings, string_to_index); });
}

NetRoute test_typed_json_and_interning() {
  std::vector<std::string> strings = {"PREEXISTING", "TILE"};
  std::unordered_map<std::string, std::uint32_t> string_to_index = {
      {"PREEXISTING", 0},
      {"TILE", 1},
  };

  const std::string json = R"json(
    {
      "edges": [
        {
          "site": null,
          "wire1": "W\u0031",
          "attachment": null,
          "to": 1,
          "tile": "TILE",
          "from": 0,
          "forward": true,
          "csr_edge": 4,
          "wire0": "W0"
        },
        {
          "from": 1,
          "to": 2,
          "csr_edge": 5,
          "wire0": "W0",
          "wire1": "W1",
          "tile": "TILE",
          "forward": false,
          "attachment": 7,
          "site": "ATTACH_SITE"
        }
      ],
      "ignored": {"nested": [null, true, false, -1.25e+2]},
      "sinks": [
        {"reached": true, "pin": "P\u0049N", "site": "S\tITE", "node": 2}
      ],
      "routed": true,
      "sources": [
        {"pin": "OUT", "site": "SOURCE", "node": 0}
      ],
      "net": "n\u0065t"
    }
  )json";

  NetRoute route = parse_test_route(json, strings, string_to_index);
  require(route.net == "net", "escaped net name was not decoded");
  require(route.routed, "reordered routed field was not decoded");
  require(route.sources.size() == 1 && route.sources[0].node == 0,
          "typed source parsing failed");
  require(route.sinks.size() == 1 && route.sinks[0].pin == "PIN" &&
              route.sinks[0].site == "S\tITE" && route.sinks[0].reached,
          "typed sink parsing or string escaping failed");
  require(route.edges.size() == 2, "typed edge parsing failed");
  require(route.edges[0].tile == 1 && route.edges[1].tile == 1,
          "an existing PhysicalNetlist string was not reused");
  require(route.edges[0].wire0 == route.edges[1].wire0 &&
              route.edges[0].wire1 == route.edges[1].wire1,
          "repeated wire strings were not interned");
  require(route.edges[0].attachment_field_present &&
              !route.edges[0].attachment.has_value() &&
              route.edges[0].site_field_present &&
              route.edges[0].site == kNoRouteString,
          "null attachment/site fields were not retained exactly");
  require(route.edges[1].attachment == std::optional<std::uint64_t>(7) &&
              strings.at(route.edges[1].site) == "ATTACH_SITE",
          "non-null attachment/site fields were not retained exactly");
  require(strings.size() == 5,
          "the per-net interner retained duplicate edge strings");

  RouteTables tables = build_route_tables(route);
  require(tables.edge_count == route.edges.size(),
          "route table edge count changed");
  require(tables.children_by_node.at(0).at(0) == &route.edges[0] &&
              tables.children_by_node.at(1).at(0) == &route.edges[1],
          "route tables copied edge payloads instead of retaining pointers");
  require(tables.sinks_by_node.at(2).at(0) == &route.sinks[0],
          "route tables copied sink payloads instead of retaining pointers");
  require(tables.source_node_by_pin.at(SitePinKey{"SOURCE", "OUT"}) == 0,
          "source lookup table changed endpoint identity");

  NetRoute numeric_boundaries = parse_test_route(
      R"json({"net":"bounds","sources":[{"node":-2147483648,"site":"S0","pin":"P0"},{"node":2147483647,"site":"S1","pin":"P1"},{"node":1.0,"site":"S2","pin":"P2"},{"node":1e0,"site":"S3","pin":"P3"}],"sinks":[],"edges":[{"from":0,"to":1,"csr_edge":-0,"tile":"T","wire0":"A","wire1":"B"},{"from":1,"to":2,"csr_edge":1.0,"tile":"T","wire0":"A","wire1":"B"},{"from":2,"to":3,"csr_edge":1e0,"tile":"T","wire0":"A","wire1":"B"},{"from":3,"to":4,"csr_edge":9007199254740991,"tile":"T","wire0":"A","wire1":"B"}]})json",
      strings, string_to_index);
  require(numeric_boundaries.sources[0].node ==
              std::numeric_limits<int>::min() &&
              numeric_boundaries.sources[1].node ==
                  std::numeric_limits<int>::max() &&
              numeric_boundaries.sources[2].node == 1 &&
              numeric_boundaries.sources[3].node == 1,
          "signed integer boundaries or legacy spellings changed");
  require(numeric_boundaries.edges[0].csr_edge == 0 &&
              numeric_boundaries.edges[1].csr_edge == 1 &&
              numeric_boundaries.edges[2].csr_edge == 1 &&
              numeric_boundaries.edges[3].csr_edge == 9007199254740991ULL,
          "unsigned integer boundary or legacy spellings changed");

  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[]} trailing)json");
  require_bad_route(
      R"json({"net":"n","sources":null,"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","routed":1,"sources":[],"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[{"from":0,"to":1,"csr_edge":9007199254740992,"tile":"T","wire0":"A","wire1":"B"}]})json");
  require_bad_route(
      R"json({"net":"n","sources":[{"node":2147483648,"site":"S","pin":"P"}],"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","sources":[{"node":-2147483649,"site":"S","pin":"P"}],"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","sources":[{"node":1.5,"site":"S","pin":"P"}],"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[{"from":1.5,"to":2,"csr_edge":3,"tile":"T","wire0":"A","wire1":"B"}]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[{"from":1.0000000000000001,"to":2,"csr_edge":3,"tile":"T","wire0":"A","wire1":"B"}]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[{"from":1,"to":2,"csr_edge":42.5,"tile":"T","wire0":"A","wire1":"B"}]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[{"from":1,"to":2,"csr_edge":9007199254740991.4,"tile":"T","wire0":"A","wire1":"B"}]})json");
  require_bad_route(
      R"json({"net":"\uD800","sources":[],"sinks":[],"edges":[]})json");
  require_bad_route(
      R"json({"net":"n","sources":[],"sinks":[],"edges":[],})json");
  require_bad_route(R"json({"net":"n","sources":[],"sinks":[]})json");

  NetRoute duplicate_first_wins = parse_test_route(
      R"json({"net":"first","net":17,"sources":[],"sources":null,"sinks":[],"edges":[]})json",
      strings, string_to_index);
  require(duplicate_first_wins.net == "first",
          "duplicate JSON fields no longer preserve first-value semantics");

  NetRoute unicode_route = parse_test_route(
      R"json({"net":"\uD83D\uDE80","sources":[],"sinks":[],"edges":[]})json",
      strings, string_to_index);
  require(unicode_route.net == "\xF0\x9F\x9A\x80",
          "JSON surrogate-pair decoding changed");

  NetRoute alternate_integer_spellings = parse_test_route(
      R"json({"net":"numbers","sources":[],"sinks":[],"edges":[{"from":10e-1,"to":2.0,"csr_edge":3e0,"tile":"TILE","wire0":"W0","wire1":"W1"}]})json",
      strings, string_to_index);
  require(alternate_integer_spellings.edges[0].from == 1 &&
              alternate_integer_spellings.edges[0].to == 2 &&
              alternate_integer_spellings.edges[0].csr_edge == 3,
          "valid exponent/fraction integer spellings were rejected");
  return route;
}

RoutingMetadataSummary metadata_for_routes(
    std::initializer_list<std::string> names) {
  RoutingMetadataSummary metadata;
  for (const std::string& name : names) {
    MetadataRouteRequest request;
    request.net = name;
    metadata.route_requests.push_back(std::move(request));
  }
  return metadata;
}

std::string empty_route_record(std::string_view net) {
  return "{\"edges\":[],\"net\":\"" + std::string(net) +
         "\",\"sinks\":[],\"sources\":[]}";
}

void test_route_file_index(const TemporaryDirectory& temporary) {
  routing::interchange::InterchangePublicationSnapshot snapshot;

  RoutingMetadataSummary reordered_metadata = metadata_for_routes({"A", "B"});
  const std::filesystem::path reordered = temporary.file("reordered.jsonl");
  write_text(reordered,
             empty_route_record("B") + "\n" + empty_route_record("A") +
                 "\n");
  RouteIndex index =
      index_routes_jsonl(reordered, reordered_metadata, snapshot);
  require(index.entries.size() == 2 && index.entries[0].line_number == 2 &&
              index.entries[1].line_number == 1,
          "route index assumed metadata or PhysicalNetlist order");
  for (std::size_t request_index = 0;
       request_index < reordered_metadata.route_requests.size();
       ++request_index) {
    index.routes_in.clear();
    index.routes_in.seekg(index.entries[request_index].offset);
    std::string line;
    require(static_cast<bool>(std::getline(index.routes_in, line)),
            "could not reread indexed route");
    std::vector<std::string> strings;
    std::unordered_map<std::string, std::uint32_t> string_to_index;
    NetRoute route = parse_route_line(line, strings, string_to_index);
    require(route.net ==
                reordered_metadata.route_requests[request_index].net,
            "route offset selected the wrong record");
  }

  RoutingMetadataSummary one_route = metadata_for_routes({"A"});
  const std::filesystem::path duplicate = temporary.file("duplicate.jsonl");
  write_text(duplicate,
             empty_route_record("A") + "\n" + empty_route_record("A") +
                 "\n");
  require_runtime_error(
      [&] { (void)index_routes_jsonl(duplicate, one_route, snapshot); },
      "duplicate route entry");

  RoutingMetadataSummary two_routes = metadata_for_routes({"A", "B"});
  const std::filesystem::path missing = temporary.file("missing.jsonl");
  write_text(missing, empty_route_record("A") + "\n");
  require_runtime_error(
      [&] { (void)index_routes_jsonl(missing, two_routes, snapshot); },
      "missing from route file");

  const std::filesystem::path extra = temporary.file("extra.jsonl");
  write_text(extra, empty_route_record("B") + "\n");
  require_runtime_error(
      [&] { (void)index_routes_jsonl(extra, one_route, snapshot); },
      "not present in metadata");

  const std::filesystem::path malformed = temporary.file("malformed.jsonl");
  write_text(malformed, R"json({"net":"A","ignored":[1,]})json");
  require_runtime_error(
      [&] { (void)index_routes_jsonl(malformed, one_route, snapshot); },
      "invalid route JSON");

  const auto expected_id =
      routing::interchange::derive_interchange_artifact_pair_id("expected");
  const auto wrong_id =
      routing::interchange::derive_interchange_artifact_pair_id("wrong");
  RoutingMetadataSummary paired_metadata = metadata_for_routes({"A"});
  paired_metadata.artifact_pair_id = expected_id;
  routing::interchange::InterchangePublicationSnapshot paired_snapshot;
  paired_snapshot.generation = expected_id;
  const std::filesystem::path mismatched = temporary.file("mismatched.jsonl");
  write_text(
      mismatched,
      "{\"artifact_pair_id\":\"" +
          routing::interchange::interchange_artifact_pair_id_string(wrong_id) +
          "\",\"net\":\"A\",\"sources\":[],\"sinks\":[],\"edges\":[]}\n");
  require_runtime_error(
      [&] {
        (void)index_routes_jsonl(mismatched, paired_metadata,
                                 paired_snapshot);
      },
      "artifact");
}

void test_endpoint_attachment_validation() {
  RoutingMetadataSummary metadata;
  metadata.version = COMPACT_TABLE_METADATA_VERSION;
  metadata.node_count = 6;
  metadata.edge_attr_count = 5;
  metadata.strings = {
      "SOURCE_ATT_TILE", "SOURCE_ATT_W0", "SOURCE_ATT_W1",
      "SOURCE_ATT_SITE", "FAB_TILE", "FAB_W0", "FAB_W1",
      "SINK_ATT_TILE", "SINK_ATT_W0", "SINK_ATT_W1",
      "SINK_ATT_SITE",
  };

  MetadataEndpointPip source_attachment;
  source_attachment.csr_edge = 1;
  source_attachment.from = 1;
  source_attachment.to = 2;
  source_attachment.tile_string = 0;
  source_attachment.wire0_string = 1;
  source_attachment.wire1_string = 2;
  source_attachment.forward = true;
  source_attachment.site_string = 3;
  source_attachment.endpoint_node = 0;
  source_attachment.role = MetadataEndpointPipRole::kSource;
  metadata.endpoint_pips.push_back(source_attachment);

  MetadataEndpointPip sink_attachment;
  sink_attachment.csr_edge = 3;
  sink_attachment.from = 3;
  sink_attachment.to = 4;
  sink_attachment.tile_string = 7;
  sink_attachment.wire0_string = 8;
  sink_attachment.wire1_string = 9;
  sink_attachment.forward = false;
  sink_attachment.site_string = 10;
  sink_attachment.endpoint_node = 5;
  sink_attachment.role = MetadataEndpointPipRole::kSink;
  metadata.endpoint_pips.push_back(sink_attachment);

  MetadataRouteRequest request;
  request.net = "attachment-net";
  RouteSitePin source;
  source.node = 0;
  source.site = "SOURCE_SITE";
  source.pin = "OUT";
  source.endpoint_pip_index = 0;
  request.sources.push_back(source);
  RouteSitePin sink;
  sink.node = 5;
  sink.site = "SINK_SITE";
  sink.pin = "IN";
  sink.endpoint_pip_index = 1;
  request.sinks.push_back(sink);

  NetRoute route;
  route.net = request.net;
  route.routed = true;
  route.sources = request.sources;
  route.sinks = request.sinks;

  const auto conventional_edge = [](int from, int to,
                                    std::uint64_t csr_edge) {
    RouteEdge edge;
    edge.from = from;
    edge.to = to;
    edge.csr_edge = csr_edge;
    edge.tile = 4;
    edge.wire0 = 5;
    edge.wire1 = 6;
    edge.forward = true;
    edge.attachment_field_present = true;
    edge.site_field_present = true;
    return edge;
  };
  route.edges.push_back(conventional_edge(0, 1, 0));

  RouteEdge source_endpoint;
  source_endpoint.from = 1;
  source_endpoint.to = 2;
  source_endpoint.csr_edge = 1;
  source_endpoint.tile = 0;
  source_endpoint.wire0 = 1;
  source_endpoint.wire1 = 2;
  source_endpoint.forward = true;
  source_endpoint.attachment_field_present = true;
  source_endpoint.attachment = 0;
  source_endpoint.site_field_present = true;
  source_endpoint.site = 3;
  route.edges.push_back(source_endpoint);

  route.edges.push_back(conventional_edge(2, 3, 2));

  RouteEdge sink_endpoint;
  sink_endpoint.from = 3;
  sink_endpoint.to = 4;
  sink_endpoint.csr_edge = 3;
  sink_endpoint.tile = 7;
  sink_endpoint.wire0 = 8;
  sink_endpoint.wire1 = 9;
  sink_endpoint.forward = false;
  sink_endpoint.attachment_field_present = true;
  sink_endpoint.attachment = 1;
  sink_endpoint.site_field_present = true;
  sink_endpoint.site = 10;
  route.edges.push_back(sink_endpoint);
  route.edges.push_back(conventional_edge(4, 5, 4));

  const RouteValidationContext context =
      build_route_validation_context(metadata);
  validate_route_against_metadata(route, request, metadata, context,
                                  metadata.strings);

  NetRoute wrong_site = route;
  wrong_site.edges[1].site = 4;
  require_runtime_error(
      [&] {
        validate_route_against_metadata(wrong_site, request, metadata,
                                        context, metadata.strings);
      },
      "does not exactly match");

  NetRoute missing_null_fields = route;
  missing_null_fields.edges[0].attachment_field_present = false;
  require_runtime_error(
      [&] {
        validate_route_against_metadata(missing_null_fields, request,
                                        metadata, context,
                                        metadata.strings);
      },
      "missing attachment/site fields");

  NetRoute outside_corridor = route;
  outside_corridor.edges[0].from = 2;
  require_runtime_error(
      [&] {
        validate_route_against_metadata(outside_corridor, request, metadata,
                                        context, metadata.strings);
      },
      "corridor");

  NetRoute unreached_sink = route;
  unreached_sink.sinks[0].reached = false;
  require_runtime_error(
      [&] {
        validate_route_against_metadata(unreached_sink, request, metadata,
                                        context, metadata.strings);
      },
      "unreached");

  NetRoute missing_sink_corridor = route;
  missing_sink_corridor.edges.pop_back();
  require_runtime_error(
      [&] {
        validate_route_against_metadata(missing_sink_corridor, request,
                                        metadata, context,
                                        metadata.strings);
      },
      "corridor");
}

void test_iterative_deep_route_tree() {
  constexpr int kDepth = 50000;
  NetRoute route;
  route.net = "deep-chain";
  route.edges.reserve(kDepth);
  for (int node = 0; node < kDepth; ++node) {
    RouteEdge edge;
    edge.from = node;
    edge.to = node + 1;
    edge.csr_edge = static_cast<std::uint64_t>(node);
    edge.tile = 0;
    edge.wire0 = 0;
    edge.wire1 = 0;
    route.edges.push_back(edge);
  }
  RouteTables tables = build_route_tables(route);

  capnp::MallocMessageBuilder message;
  auto root = message.initRoot<PhysicalNetlist::PhysNetlist>();
  auto net = root.initPhysNets(1)[0];
  auto source = net.initSources(1)[0];
  auto old_stubs = net.initStubs(0);
  StubBranchStore stub_store;
  const std::size_t emitted = insert_route_tree(
      source, 0, route, tables, stub_store, old_stubs);
  require(emitted == route.edges.size(),
          "deep iterative route tree did not emit every edge");

  NetRoute cycle;
  cycle.net = "cycle";
  constexpr int kCycleDepth = 10000;
  cycle.edges.reserve(kCycleDepth + 1);
  for (int node = 0; node < kCycleDepth; ++node) {
    RouteEdge edge;
    edge.from = node;
    edge.to = node + 1;
    edge.tile = edge.wire0 = edge.wire1 = 0;
    cycle.edges.push_back(edge);
  }
  RouteEdge closing_edge;
  closing_edge.from = kCycleDepth;
  closing_edge.to = 0;
  closing_edge.tile = closing_edge.wire0 = closing_edge.wire1 = 0;
  cycle.edges.push_back(closing_edge);
  RouteTables cycle_tables = build_route_tables(cycle);

  capnp::MallocMessageBuilder cycle_message;
  auto cycle_root = cycle_message.initRoot<PhysicalNetlist::PhysNetlist>();
  auto cycle_net = cycle_root.initPhysNets(1)[0];
  auto cycle_source = cycle_net.initSources(1)[0];
  auto cycle_stubs = cycle_net.initStubs(0);
  StubBranchStore cycle_stub_store;
  require_runtime_error(
      [&] {
        (void)insert_route_tree(cycle_source, 0, cycle, cycle_tables,
                                cycle_stub_store, cycle_stubs);
      },
      "cycle");
}

void test_streamed_multisegment_output(
    const TemporaryDirectory& temporary) {
  capnp::MallocMessageBuilder message(
      8, capnp::AllocationStrategy::FIXED_SIZE);
  auto root = message.initRoot<PhysicalNetlist::PhysNetlist>();
  constexpr std::uint32_t kStringCount = 128;
  auto strings = root.initStrList(kStringCount);
  std::vector<std::string> expected;
  expected.reserve(kStringCount);
  for (std::uint32_t index = 0; index < kStringCount; ++index) {
    expected.push_back("string-" + std::to_string(index) + "-" +
                       std::string(80, static_cast<char>('a' + index % 26)));
    strings.set(index, expected.back());
  }
  require(message.getSegmentsForOutput().size() > 1,
          "multi-segment fixture unexpectedly fit in one segment");

  const std::filesystem::path output = temporary.file("multisegment.phys.gz");
  GzipOutputStream gzip_output(output);
  capnp::writeMessage(gzip_output, message);
  gzip_output.finish();

  const std::vector<std::uint8_t> encoded = read_bytes(output);
  require(encoded.size() >= 2 && encoded[0] == 0x1f && encoded[1] == 0x8b,
          "streamed output is not a gzip stream");

  fragment_gzip_reads = true;
  fragment_gzip_read_index = 0;
  WordAlignedPayload payload = read_gzip_or_plain_words(output);
  fragment_gzip_reads = false;
  const auto* raw =
      reinterpret_cast<const std::uint8_t*>(payload.words.begin());
  const std::uint32_t segment_count_minus_one =
      static_cast<std::uint32_t>(raw[0]) |
      (static_cast<std::uint32_t>(raw[1]) << 8) |
      (static_cast<std::uint32_t>(raw[2]) << 16) |
      (static_cast<std::uint32_t>(raw[3]) << 24);
  require(segment_count_minus_one + 1 > 1,
          "streamed output flattened a multi-segment message");
  capnp::ReaderOptions options;
  options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.begin(), payload.word_count()), options);
  auto reread = reader.getRoot<PhysicalNetlist::PhysNetlist>();
  auto reread_strings = reread.getStrList();
  require(reread_strings.size() == kStringCount,
          "streamed multi-segment output changed list size");
  for (std::uint32_t index : {0U, 1U, 63U, 127U}) {
    require(std::string(reread_strings[index].cStr()) == expected[index],
            "streamed multi-segment output changed message semantics");
  }
}

void write_u64(std::ofstream& output, std::uint64_t value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_zero_bytes(std::ofstream& output, std::size_t byte_count) {
  static constexpr char zeros[64] = {};
  while (byte_count != 0) {
    const std::size_t chunk = std::min(byte_count, sizeof(zeros));
    output.write(zeros, static_cast<std::streamsize>(chunk));
    byte_count -= chunk;
  }
}

void write_v4_metadata_fixture(const std::filesystem::path& path) {
  const std::vector<std::string> strings = {
      "A", "B", "C", "SRC_SITE", "OUT", "SINK_SITE", "IN",
  };
  constexpr std::uint64_t kNodeCount = 6;
  constexpr std::uint64_t kEdgeCount = 3;
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "could not create metadata fixture");
  output.write(METADATA_MAGIC, sizeof(METADATA_MAGIC));
  write_u64(output, LEGACY_METADATA_VERSION);
  write_u64(output, EXPECTED_OUTGOING_EDGE_ORIENTATION);
  write_u64(output, strings.size());
  write_u64(output, kNodeCount);
  write_u64(output, kEdgeCount);
  write_u64(output, 0);  // PIP records.
  write_u64(output, 0);  // Site-pin attributes.
  write_u64(output, 3);  // Route requests.
  write_u64(output, 0);  // Blocked nodes.
  write_u64(output, 0);  // Sink-stop nodes.
  write_u64(output, 0);  // Logical cells.
  write_u64(output, 0);  // Logical nets.
  write_u64(output, 0);  // Logical port instances.
  write_u64(output, 0);  // Embedded PhysicalNetlist bytes.
  write_u64(output, 0);  // Embedded LogicalNetlist bytes.
  for (int i = 0; i < 4; ++i) write_u64(output, 0);
  for (const std::string& string : strings) {
    write_u64(output, string.size());
    output.write(string.data(), static_cast<std::streamsize>(string.size()));
  }

  // Legacy node IDs, four coordinate arrays, and tile/wire type arrays.
  write_zero_bytes(output, kNodeCount *
                               (3 * sizeof(std::uint64_t) +
                                4 * sizeof(std::int32_t)));
  write_zero_bytes(output, kEdgeCount * 2 * sizeof(std::uint64_t));

  const auto write_request = [&](std::uint64_t net_string,
                                 std::uint64_t source_node,
                                 std::uint64_t sink_node) {
    write_u64(output, net_string);
    write_u64(output, kNoEndpointPip);
    write_u64(output, 1);
    write_u64(output, source_node);
    write_u64(output, 3);  // SRC_SITE.
    write_u64(output, 4);  // OUT.
    write_u64(output, 1);
    write_u64(output, sink_node);
    write_u64(output, 5);  // SINK_SITE.
    write_u64(output, 6);  // IN.
  };
  // Deliberately differ from both PhysicalNetlist order (A/B/C) and JSONL
  // order (C/A/B).
  write_request(1, 2, 3);
  write_request(2, 4, 5);
  write_request(0, 0, 1);
  output.close();
  require(static_cast<bool>(output), "could not finish metadata fixture");
}

std::vector<std::uint8_t> make_unrouted_phys_fixture() {
  const std::vector<std::string> strings = {
      "A",         "B",    "C",   "UNTOUCHED", "SRC_SITE", "OUT",
      "SINK_SITE", "IN",   "TILE", "W0",        "W1",
  };
  capnp::MallocMessageBuilder message;
  auto root = message.initRoot<PhysicalNetlist::PhysNetlist>();
  auto str_list = root.initStrList(static_cast<std::uint32_t>(strings.size()));
  for (std::uint32_t index = 0; index < strings.size(); ++index) {
    str_list.set(index, strings[index]);
  }
  root.setPart("semantic-equivalence-fixture");
  auto nets = root.initPhysNets(4);
  for (std::uint32_t index = 0; index < 3; ++index) {
    auto net = nets[index];
    net.setName(index);
    net.setType(PhysicalNetlist::PhysNetlist::NetType::SIGNAL);
    auto source = net.initSources(1)[0].initRouteSegment().initSitePin();
    source.setSite(4);
    source.setPin(5);
    auto sink = net.initStubs(1)[0].initRouteSegment().initSitePin();
    sink.setSite(6);
    sink.setPin(7);
  }
  nets[3].setName(3);
  nets[3].setType(PhysicalNetlist::PhysNetlist::NetType::GND);
  nets[3].initSources(0);
  nets[3].initStubs(0);

  kj::VectorOutputStream output;
  capnp::writeMessage(output, message);
  const auto bytes = output.getArray();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::string route_record(std::string_view net, int from, int to,
                         std::uint64_t csr_edge) {
  return "{\"net\":\"" + std::string(net) +
         "\",\"routed\":true,\"sources\":[{\"node\":" +
         std::to_string(from) +
         ",\"site\":\"SRC_SITE\",\"pin\":\"OUT\"}],\"sinks\":[{\"node\":" +
         std::to_string(to) +
         ",\"site\":\"SINK_SITE\",\"pin\":\"IN\",\"reached\":true}],"
         "\"edges\":[{\"from\":" +
         std::to_string(from) + ",\"to\":" + std::to_string(to) +
         ",\"csr_edge\":" + std::to_string(csr_edge) +
         ",\"tile\":\"TILE\",\"wire0\":\"W0\",\"wire1\":\"W1\","
         "\"forward\":true}]}";
}

std::string inspect_reconstructed_phys(const std::filesystem::path& path) {
  WordAlignedPayload payload = read_gzip_or_plain_words(path);
  capnp::ReaderOptions options;
  options.traversalLimitInWords =
      std::numeric_limits<std::uint64_t>::max();
  options.nestingLimit = 1 << 20;
  capnp::FlatArrayMessageReader reader(
      kj::arrayPtr(payload.words.begin(), payload.word_count()), options);
  auto root = reader.getRoot<PhysicalNetlist::PhysNetlist>();
  require(std::string(root.getPart().cStr()) ==
              "semantic-equivalence-fixture",
          "reconstruction changed unrelated PhysicalNetlist fields");
  const std::vector<std::string> expected_strings = {
      "A",         "B",    "C",   "UNTOUCHED", "SRC_SITE", "OUT",
      "SINK_SITE", "IN",   "TILE", "W0",        "W1",
  };
  auto strings = root.getStrList();
  require(strings.size() == expected_strings.size(),
          "semantic reconstruction changed strList size");
  for (std::uint32_t index = 0; index < strings.size(); ++index) {
    require(std::string(strings[index].cStr()) == expected_strings[index],
            "semantic reconstruction changed strList order or content");
  }

  auto nets = root.getPhysNets();
  require(nets.size() == 4, "semantic reconstruction changed net count");
  std::ostringstream projection;
  for (std::uint32_t index = 0; index < 3; ++index) {
    auto net = nets[index];
    require(net.getName() == index && net.getStubs().size() == 0,
            "routed net name/stub semantics changed");
    auto sources = net.getSources();
    require(sources.size() == 1, "routed net source count changed");
    auto source_segment = sources[0].getRouteSegment();
    require(source_segment.isSitePin(), "source site pin was replaced");
    auto source_pin = source_segment.getSitePin();
    require(source_pin.getSite() == 4 && source_pin.getPin() == 5,
            "source endpoint changed");
    auto source_children = sources[0].getBranches();
    require(source_children.size() == 1,
            "source did not receive exactly one PIP");
    auto pip_segment = source_children[0].getRouteSegment();
    require(pip_segment.isPip(), "routed child is not a PIP");
    auto pip = pip_segment.getPip();
    require(pip.getTile() == 8 && pip.getWire0() == 9 &&
                pip.getWire1() == 10 && pip.getForward() &&
                !pip.getIsFixed() && pip.isNoSite(),
            "emitted conventional PIP semantics changed");
    auto pip_children = source_children[0].getBranches();
    require(pip_children.size() == 1,
            "PIP did not receive exactly one routed sink");
    auto sink_segment = pip_children[0].getRouteSegment();
    require(sink_segment.isSitePin(), "routed sink is not a site pin");
    auto sink = sink_segment.getSitePin();
    require(sink.getSite() == 6 && sink.getPin() == 7,
            "routed sink endpoint changed");
    projection << net.getName() << ':' << pip.getTile() << ','
               << pip.getWire0() << ',' << pip.getWire1() << ';';
  }
  require(nets[3].getName() == 3 &&
              nets[3].getType() ==
                  PhysicalNetlist::PhysNetlist::NetType::GND &&
              nets[3].getSources().size() == 0 &&
              nets[3].getStubs().size() == 0,
          "unrequested PhysicalNetlist net was modified");
  return projection.str();
}

void test_end_to_end_semantic_equivalence(
    const TemporaryDirectory& temporary) {
  const std::filesystem::path plain_input = temporary.file("semantic.plain");
  const std::filesystem::path gzip_input = temporary.file("semantic.phys.gz");
  const std::filesystem::path metadata_path = temporary.file("semantic.ifmeta.bin");
  const std::filesystem::path routes_path = temporary.file("semantic.routes.jsonl");
  const std::vector<std::uint8_t> physical = make_unrouted_phys_fixture();
  write_bytes(plain_input, physical);
  write_gzip(gzip_input, physical);
  write_v4_metadata_fixture(metadata_path);
  write_text(routes_path,
             route_record("C", 4, 5, 2) + "\n" +
                 route_record("A", 0, 1, 0) + "\n" +
                 route_record("B", 2, 3, 1) + "\n");

  RoutingMetadataSummary metadata = load_metadata_summary(metadata_path);
  routing::interchange::InterchangePublicationSnapshot snapshot;
  std::string expected_projection;
  for (const auto& input_output :
       {std::pair{plain_input, temporary.file("from-plain.phys")},
        std::pair{gzip_input, temporary.file("from-gzip.phys")}}) {
    RouteIndex index = index_routes_jsonl(routes_path, metadata, snapshot);
    write_routed_phys(input_output.first, input_output.second, metadata,
                      snapshot, index, false);
    const std::vector<std::uint8_t> encoded = read_bytes(input_output.second);
    require(encoded.size() >= 2 && encoded[0] == 0x1f && encoded[1] == 0x8b,
            "end-to-end reconstruction output is not gzip");
    const std::string projection =
        inspect_reconstructed_phys(input_output.second);
    if (expected_projection.empty()) {
      expected_projection = projection;
    } else {
      require(projection == expected_projection,
              "plain and gzip inputs reconstructed differently");
    }
  }

  require(physical.size() > sizeof(capnp::word),
          "semantic input is too small for truncation test");
  const std::filesystem::path truncated_input =
      temporary.file("semantic-truncated.plain");
  write_bytes(truncated_input,
              std::vector<std::uint8_t>(physical.begin(),
                                        physical.end() - sizeof(capnp::word)));
  require_exception([&] {
    RouteIndex index = index_routes_jsonl(routes_path, metadata, snapshot);
    write_routed_phys(truncated_input,
                      temporary.file("from-truncated.phys"), metadata,
                      snapshot, index, false);
  });
}

}  // namespace

int main() {
  try {
    TemporaryDirectory temporary;
    test_fragmented_word_accumulation(temporary);
    (void)test_typed_json_and_interning();
    test_route_file_index(temporary);
    test_endpoint_attachment_validation();
    test_iterative_deep_route_tree();
    test_streamed_multisegment_output(temporary);
    test_end_to_end_semantic_equivalence(temporary);
    std::cout << "routes_to_phys regression test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "routes_to_phys regression test failed: " << error.what()
              << '\n';
    return 1;
  }
}
