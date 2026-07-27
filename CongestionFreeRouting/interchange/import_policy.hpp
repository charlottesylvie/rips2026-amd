#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace routing::interchange {

// PathFinder can safely extend only a completely unrouted ordinary signal
// net.  The contest wrapper makes the same distinction before invoking
// RWRoute: driverless OOC nets are not routing work, while nets that already
// contain inter-site routing are preserved rather than treating every site
// pin in their source tree as a new root.
enum class PhysicalNetDisposition {
  kRouteSignal,
  kPreserveCompleteOrLoadless,
  kExcludeDriverlessSignal,
  kPreserveUnsupportedPartialSignal,
  kPreserveUnsupportedSignalShape,
  kPreserveUnsupportedStatic,
};

struct PhysicalNetRoutingFacts {
  bool is_signal = true;
  std::size_t top_level_source_count = 0;
  std::size_t top_level_stub_count = 0;
  std::size_t source_site_pin_count = 0;
  bool source_site_pins_are_leaves = true;
  bool has_inter_site_pip = false;
  bool has_stub_nodes = false;
  bool top_level_stubs_are_site_pins = true;
};

constexpr PhysicalNetDisposition classify_physical_net(
    const PhysicalNetRoutingFacts& facts) {
  if (!facts.is_signal) {
    // Contest inputs arrive with global and static routing already present.
    // PhysicalNetlist's `stubs` field is part of that serialized routing
    // representation; it is not, by itself, proof that a GND/VCC net needs
    // PathFinder work.  Static nets are therefore always preservation-only:
    // the importer reserves every represented resource and reconstruction
    // leaves the net structurally unchanged.
    return PhysicalNetDisposition::kPreserveCompleteOrLoadless;
  }
  if (facts.top_level_stub_count == 0) {
    return PhysicalNetDisposition::kPreserveCompleteOrLoadless;
  }
  // Match the contest wrapper's source-less OOC rule exactly. A nonempty
  // source forest whose shape we cannot interpret is not driverless.
  if (facts.top_level_source_count == 0) {
    return PhysicalNetDisposition::kExcludeDriverlessSignal;
  }
  if (facts.source_site_pin_count == 0 ||
      !facts.source_site_pins_are_leaves) {
    return PhysicalNetDisposition::kPreserveUnsupportedSignalShape;
  }
  if (facts.has_inter_site_pip || facts.has_stub_nodes) {
    return PhysicalNetDisposition::kPreserveUnsupportedPartialSignal;
  }
  if (!facts.top_level_stubs_are_site_pins) {
    return PhysicalNetDisposition::kPreserveUnsupportedSignalShape;
  }
  return PhysicalNetDisposition::kRouteSignal;
}

constexpr bool include_pip_in_static_graph(bool conventional) {
  return conventional;
}

inline bool physical_part_matches_device(const std::string& device_name,
                                         const std::string& physical_part) {
  return !device_name.empty() && !physical_part.empty() &&
         (physical_part == device_name ||
         (physical_part.size() > device_name.size() &&
          physical_part.compare(0, device_name.size(), device_name) == 0 &&
          physical_part[device_name.size()] == '-'));
}

constexpr bool unresolved_resource_is_fatal(bool full_device_graph) {
  return full_device_graph;
}

constexpr bool sink_requires_terminal_row(bool is_source_of_same_net) {
  return !is_source_of_same_net;
}

constexpr bool fixed_site_pin_candidate_fallback_allowed(
    bool has_active_site_type) {
  return !has_active_site_type;
}

// RapidWright uses this exact signal-net sentinel to serialize site resources
// that are occupied without representing an ordinary logical connection. It
// is preservation-only routing state: keep its resources blocked, but never
// turn its apparent branches/stubs into a source-to-sink routing request.
constexpr bool is_reserved_used_resource_net(std::string_view net_name,
                                             bool is_signal) {
  return is_signal && net_name == "GLOBAL_USEDNET";
}

inline bool claim_unique_physical_net_name(
    std::unordered_map<std::string, std::size_t>& owners,
    const std::string& name,
    std::size_t owner) {
  return owners.emplace(name, owner).second;
}

// Logical net names are scoped by their owning cell, while PhysicalNetlist
// names are flat strings.  A name-only bridge is safe only when the logical
// name is globally unambiguous; once two different logical nets claim the same
// spelling, leave the physical request unassociated instead of silently
// selecting whichever cell happened to be serialized first.
inline void index_unambiguous_logical_net_name(
    std::unordered_map<std::string, std::uint64_t>& unique_names,
    std::unordered_set<std::string>& ambiguous_names,
    const std::string& name,
    std::uint64_t logical_net_index) {
  if (ambiguous_names.find(name) != ambiguous_names.end()) {
    return;
  }
  const auto [found, inserted] =
      unique_names.emplace(name, logical_net_index);
  if (!inserted && found->second != logical_net_index) {
    unique_names.erase(found);
    ambiguous_names.insert(name);
  }
}

// RWRoute reserves the paired LUT static-output site pin because UltraScale+
// [A-H]_O/[A-H]MUX share exclusive site routing.  Keep this model deliberately
// scoped to the audited xcvu3p device and an exact active SLICE type: Versal
// uses a different *_O/*Q pairing, while Q pins on xcvu3p are independent FF
// outputs and must not be blocked.
inline std::optional<std::string> paired_static_slice_output_pin(
    const std::string& device_name,
    const std::string& site_name,
    const std::optional<std::string>& active_site_type,
    const std::string& pin) {
  if (device_name != "xcvu3p" || pin.empty() || pin.front() < 'A' ||
      pin.front() > 'H') {
    return std::nullopt;
  }
  if (active_site_type.has_value()) {
    if (*active_site_type != "SLICEL" && *active_site_type != "SLICEM") {
      return std::nullopt;
    }
  } else if (site_name.compare(0, 7, "SLICE_X") != 0) {
    // Legacy netlists can omit siteInsts.  On the audited xcvu3p only, the
    // concrete SLICE_X naming convention is sufficient to select the same
    // UltraScale+ pairing while the caller blocks all typed node candidates.
    return std::nullopt;
  }
  const std::string letter(1, pin.front());
  if (pin.size() == 3 && pin[1] == '_' && pin[2] == 'O') {
    return letter + "MUX";
  }
  if (pin.size() == 4 && pin.compare(1, 3, "MUX") == 0) {
    return letter + "_O";
  }
  return std::nullopt;
}

inline std::filesystem::path normalized_interchange_path(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path normalized =
      std::filesystem::weakly_canonical(path, error);
  if (!error) {
    return normalized;
  }
  error.clear();
  normalized = std::filesystem::absolute(path, error);
  return error ? path.lexically_normal() : normalized.lexically_normal();
}

inline bool interchange_paths_alias(const std::filesystem::path& lhs,
                                    const std::filesystem::path& rhs) {
  std::error_code error;
  if (std::filesystem::exists(lhs, error) && !error) {
    error.clear();
    if (std::filesystem::exists(rhs, error) && !error) {
      error.clear();
      const bool equivalent = std::filesystem::equivalent(lhs, rhs, error);
      if (!error) {
        return equivalent;
      }
    }
  }
  return normalized_interchange_path(lhs) ==
         normalized_interchange_path(rhs);
}

// A conversion publishes two files that must always be consumed as one
// generation.  The 128-bit identifier is embedded in both binary headers and
// written to the publication sidecar.  It is an identity token, not a content
// checksum; its purpose is to make old/new or independently generated pairs
// fail closed.
struct InterchangeArtifactPairId {
  std::uint64_t high = 0;
  std::uint64_t low = 0;

  constexpr bool is_zero() const { return high == 0 && low == 0; }

  constexpr bool operator==(const InterchangeArtifactPairId& other) const {
    return high == other.high && low == other.low;
  }

  constexpr bool operator!=(const InterchangeArtifactPairId& other) const {
    return !(*this == other);
  }
};

inline std::string interchange_artifact_pair_id_string(
    const InterchangeArtifactPairId& id) {
  if (id.is_zero()) {
    throw std::runtime_error(
        "interchange artifact pair id must not be zero");
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(32, '0');
  const auto encode_word = [&](std::uint64_t word, std::size_t offset) {
    for (std::size_t digit = 0; digit < 16; ++digit) {
      const unsigned shift = static_cast<unsigned>((15 - digit) * 4);
      result[offset + digit] = kHex[(word >> shift) & 0xfU];
    }
  };
  encode_word(id.high, 0);
  encode_word(id.low, 16);
  return result;
}

inline InterchangeArtifactPairId parse_interchange_artifact_pair_id(
    std::string_view text) {
  if (text.size() != 32) {
    throw std::runtime_error(
        "interchange artifact pair id must contain exactly 32 hex digits");
  }
  const auto nibble = [](char value) -> std::uint64_t {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint64_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<std::uint64_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
      return static_cast<std::uint64_t>(value - 'A' + 10);
    }
    throw std::runtime_error(
        "interchange artifact pair id contains a non-hex digit");
  };
  const auto parse_word = [&](std::size_t offset) {
    std::uint64_t word = 0;
    for (std::size_t digit = 0; digit < 16; ++digit) {
      word = (word << 4) | nibble(text[offset + digit]);
    }
    return word;
  };
  InterchangeArtifactPairId result{parse_word(0), parse_word(16)};
  if (result.is_zero()) {
    throw std::runtime_error(
        "interchange artifact pair id must not be zero");
  }
  return result;
}

// Produce two independently mixed 64-bit words for host-side policy tests and
// callers that already own a suitably unique seed. The production converter
// reads its identifier directly from the operating system entropy source.
inline InterchangeArtifactPairId derive_interchange_artifact_pair_id(
    std::string_view seed) {
  std::uint64_t high = 1469598103934665603ULL;
  std::uint64_t low = 7809847782465536322ULL;
  for (const unsigned char byte : seed) {
    high ^= static_cast<std::uint64_t>(byte);
    high *= 1099511628211ULL;
    low ^= static_cast<std::uint64_t>(byte) + 0x9e3779b97f4a7c15ULL;
    low *= 0x100000001b3ULL;
    low ^= low >> 29;
  }
  high ^= low + 0x9e3779b97f4a7c15ULL + (high << 6) + (high >> 2);
  low ^= high + 0x517cc1b727220a95ULL + (low << 7) + (low >> 3);
  InterchangeArtifactPairId result{high, low};
  if (result.is_zero()) {
    result.low = 1;
  }
  return result;
}

inline std::filesystem::path interchange_publication_marker_path(
    const std::filesystem::path& artifact_path) {
  std::filesystem::path marker = artifact_path;
  marker += ".publishing";
  return marker;
}

inline std::filesystem::path interchange_publication_generation_path(
    const std::filesystem::path& metadata_path) {
  std::filesystem::path generation = metadata_path;
  generation += ".generation";
  return generation;
}

struct InterchangePublicationSnapshot {
  std::optional<InterchangeArtifactPairId> generation;

  bool operator==(const InterchangePublicationSnapshot& other) const {
    return generation == other.generation;
  }
};

inline void require_no_interchange_publication_markers(
    const std::vector<std::filesystem::path>& artifact_paths) {
  for (const std::filesystem::path& artifact_path : artifact_paths) {
    const std::filesystem::path marker =
        interchange_publication_marker_path(artifact_path);
    std::error_code error;
    const bool exists = std::filesystem::exists(marker, error);
    if (error) {
      throw std::runtime_error(
          "could not inspect interchange publication marker: " +
          error.message());
    }
    if (exists) {
      throw std::runtime_error(
          "interchange CSR/metadata publication is incomplete or active: " +
          marker.string());
    }
  }
}

// Sample the sidecar generation only while no converter owns either artifact
// marker. A reader takes one snapshot before loading CSR/metadata and verifies
// a second afterward. The embedded ids are checked separately against this
// snapshot, closing the stable-mismatched-pair hole that a seqlock alone cannot.
inline InterchangePublicationSnapshot snapshot_interchange_publication(
    const std::vector<std::filesystem::path>& artifact_paths,
    const std::filesystem::path& metadata_path) {
  const std::filesystem::path generation =
      interchange_publication_generation_path(metadata_path);
  require_no_interchange_publication_markers(artifact_paths);

  std::error_code error;
  const bool has_generation = std::filesystem::exists(generation, error);
  if (error) {
    throw std::runtime_error(
        "could not inspect interchange publication generation: " +
        error.message());
  }

  InterchangePublicationSnapshot snapshot;
  if (has_generation) {
    const std::uintmax_t byte_count =
        std::filesystem::file_size(generation, error);
    if (error || (byte_count != 32 && byte_count != 33)) {
      throw std::runtime_error(
          "interchange publication generation file is invalid: " +
          generation.string());
    }
    std::ifstream input(generation, std::ios::binary);
    if (!input) {
      throw std::runtime_error(
          "could not open interchange publication generation: " +
          generation.string());
    }
    std::string token(static_cast<std::size_t>(byte_count), '\0');
    input.read(token.data(), static_cast<std::streamsize>(token.size()));
    if (!input) {
      throw std::runtime_error(
          "could not read interchange publication generation: " +
          generation.string());
    }
    if (token.size() == 33) {
      if (token.back() != '\n') {
        throw std::runtime_error(
            "interchange publication generation has invalid trailing data: " +
            generation.string());
      }
      token.pop_back();
    }
    snapshot.generation = parse_interchange_artifact_pair_id(token);
  }
  require_no_interchange_publication_markers(artifact_paths);
  return snapshot;
}

inline InterchangePublicationSnapshot snapshot_interchange_publication(
    const std::filesystem::path& csr_path,
    const std::filesystem::path& metadata_path) {
  return snapshot_interchange_publication(
      std::vector<std::filesystem::path>{csr_path, metadata_path},
      metadata_path);
}

inline InterchangePublicationSnapshot snapshot_interchange_publication(
    const std::filesystem::path& metadata_path) {
  return snapshot_interchange_publication({metadata_path}, metadata_path);
}

inline void verify_interchange_publication(
    const std::vector<std::filesystem::path>& artifact_paths,
    const std::filesystem::path& metadata_path,
    const InterchangePublicationSnapshot& before) {
  if (!(snapshot_interchange_publication(artifact_paths, metadata_path) ==
        before)) {
    throw std::runtime_error(
        "interchange CSR/metadata generation changed while it was read");
  }
}

inline void verify_interchange_publication(
    const std::filesystem::path& csr_path,
    const std::filesystem::path& metadata_path,
    const InterchangePublicationSnapshot& before) {
  verify_interchange_publication(
      std::vector<std::filesystem::path>{csr_path, metadata_path},
      metadata_path, before);
}

inline void verify_interchange_publication(
    const std::filesystem::path& metadata_path,
    const InterchangePublicationSnapshot& before) {
  verify_interchange_publication({metadata_path}, metadata_path, before);
}

// Legacy pairs have no embedded or sidecar id. New pairs require all three
// identities. Mixed versions, a deleted/stale sidecar, or independently
// generated artifacts are rejected rather than guessed compatible.
inline void require_matching_interchange_pair_ids(
    const std::optional<InterchangeArtifactPairId>& csr_or_routes_id,
    const std::optional<InterchangeArtifactPairId>& metadata_id,
    const std::optional<InterchangeArtifactPairId>& publication_id) {
  const bool any = csr_or_routes_id.has_value() || metadata_id.has_value() ||
                   publication_id.has_value();
  if (!any) {
    return;
  }
  if (!csr_or_routes_id.has_value() || !metadata_id.has_value() ||
      !publication_id.has_value()) {
    throw std::runtime_error(
        "interchange artifact pair mixes legacy and generation-tagged files");
  }
  if (*csr_or_routes_id != *metadata_id ||
      *metadata_id != *publication_id) {
    throw std::runtime_error(
        "interchange CSR/metadata artifact pair ids do not match");
  }
}

inline void require_distinct_interchange_paths(
    const std::vector<std::filesystem::path>& paths) {
  for (const std::filesystem::path& path : paths) {
    if (path.empty()) {
      throw std::runtime_error("interchange path must not be empty");
    }
  }
  for (std::size_t lhs = 0; lhs < paths.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < paths.size(); ++rhs) {
      if (interchange_paths_alias(paths[lhs], paths[rhs])) {
        throw std::runtime_error(
            "interchange input/output paths must be distinct");
      }
    }
  }
}

inline void preserve_node(std::vector<std::uint8_t>& blocked,
                          std::int32_t node) {
  if (node < 0 || static_cast<std::size_t>(node) >= blocked.size()) {
    throw std::runtime_error("preserved node is outside the routing graph");
  }
  blocked[static_cast<std::size_t>(node)] = 1;
}

inline void preserve_pip_endpoints(std::vector<std::uint8_t>& blocked,
                                   std::int32_t node0,
                                   std::int32_t node1) {
  preserve_node(blocked, node0);
  preserve_node(blocked, node1);
}

inline void mark_sink_terminal(std::vector<std::uint8_t>& sink_stops,
                               std::int32_t node) {
  if (node < 0 || static_cast<std::size_t>(node) >= sink_stops.size()) {
    throw std::runtime_error("sink node is outside the routing graph");
  }
  sink_stops[static_cast<std::size_t>(node)] = 1;
}

inline void mark_source_exclusive(
    std::vector<std::uint8_t>& exclusive_sources,
    std::int32_t node) {
  if (node < 0 ||
      static_cast<std::size_t>(node) >= exclusive_sources.size()) {
    throw std::runtime_error("source node is outside the routing graph");
  }
  exclusive_sources[static_cast<std::size_t>(node)] = 1;
}

enum class EndpointClaim {
  kNew,
  kSameOwner,
  kDifferentOwner,
};

inline EndpointClaim claim_route_endpoint(
    std::unordered_map<std::int32_t, std::size_t>& owners,
    std::int32_t node,
    std::size_t owner) {
  if (node < 0) {
    throw std::runtime_error("route endpoint node is invalid");
  }
  const auto [found, inserted] = owners.emplace(node, owner);
  if (inserted) {
    return EndpointClaim::kNew;
  }
  return found->second == owner ? EndpointClaim::kSameOwner
                                : EndpointClaim::kDifferentOwner;
}

// Pure model for DeviceResources' primary/alternate site-pin mapping.  An
// alternate pin first maps to a primary-pin index and only then to the tile
// wire.  Keeping this arithmetic outside Cap'n Proto makes the invariants
// host-testable on systems without the generated FPGA Interchange headers.
struct SitePinTemplate {
  std::uint32_t site_type_string = 0;
  std::uint32_t pin_string = 0;
  std::uint32_t tile_wire_string = 0;
};

struct AlternateSitePinMap {
  std::uint32_t site_type_string = 0;
  std::vector<std::uint32_t> pin_strings;
  std::vector<std::uint32_t> primary_pin_indices;
};

inline std::vector<SitePinTemplate> build_site_pin_templates(
    std::uint32_t primary_site_type_string,
    const std::vector<std::uint32_t>& primary_pin_strings,
    const std::vector<std::uint32_t>& primary_pin_tile_wires,
    const std::vector<AlternateSitePinMap>& alternates) {
  if (primary_pin_strings.size() != primary_pin_tile_wires.size()) {
    throw std::runtime_error(
        "primary site-pin and tile-wire counts do not match");
  }

  std::size_t total = primary_pin_strings.size();
  for (const AlternateSitePinMap& alternate : alternates) {
    if (alternate.pin_strings.size() !=
        alternate.primary_pin_indices.size()) {
      throw std::runtime_error(
          "alternate site-pin and parent-index counts do not match");
    }
    if (alternate.pin_strings.size() >
        std::vector<SitePinTemplate>().max_size() - total) {
      throw std::overflow_error("site-pin template count overflows size_t");
    }
    total += alternate.pin_strings.size();
  }

  std::vector<SitePinTemplate> result;
  result.reserve(total);
  for (std::size_t pin = 0; pin < primary_pin_strings.size(); ++pin) {
    result.push_back({primary_site_type_string,
                      primary_pin_strings[pin],
                      primary_pin_tile_wires[pin]});
  }
  for (const AlternateSitePinMap& alternate : alternates) {
    for (std::size_t pin = 0; pin < alternate.pin_strings.size(); ++pin) {
      const std::uint32_t primary_pin =
          alternate.primary_pin_indices[pin];
      if (primary_pin >= primary_pin_tile_wires.size()) {
        throw std::runtime_error(
            "alternate site pin refers to an invalid primary pin");
      }
      result.push_back({alternate.site_type_string,
                        alternate.pin_strings[pin],
                        primary_pin_tile_wires[primary_pin]});
    }
  }
  return result;
}

}  // namespace routing::interchange
