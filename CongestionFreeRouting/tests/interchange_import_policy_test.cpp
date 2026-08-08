#include "../interchange/import_policy.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ri = routing::interchange;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callback>
void require_throws(Callback&& callback, const std::string& message) {
  bool threw = false;
  try {
    callback();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, message);
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("rips_interchange_policy_" + std::to_string(nonce));
    if (!std::filesystem::create_directory(path_)) {
      throw std::runtime_error("could not create temporary test directory");
    }
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

int main() {
  try {
    using D = ri::PhysicalNetDisposition;
    using F = ri::PhysicalNetRoutingFacts;

    require(ri::classify_physical_net(F{}) ==
                D::kPreserveCompleteOrLoadless,
            "complete signal net was not preserved");

    F driverless;
    driverless.top_level_stub_count = 210;
    require(ri::classify_physical_net(driverless) ==
                D::kExcludeDriverlessSignal,
            "driverless OOC signal was counted as routing work");

    F routable = driverless;
    routable.top_level_source_count = 1;
    routable.source_site_pin_count = 1;
    require(ri::classify_physical_net(routable) == D::kRouteSignal,
            "fully unrouted signal was not selected");

    F unsupported_source = routable;
    unsupported_source.source_site_pin_count = 0;
    require(ri::classify_physical_net(unsupported_source) ==
                D::kPreserveUnsupportedSignalShape,
            "nonempty unsupported source forest was mislabeled driverless");
    F occupied_source_site_pin = routable;
    occupied_source_site_pin.source_site_pins_are_leaves = false;
    require(ri::classify_physical_net(occupied_source_site_pin) ==
                D::kPreserveUnsupportedSignalShape,
            "source sitePin with existing children was accepted");

    F partial = routable;
    partial.has_inter_site_pip = true;
    require(ri::classify_physical_net(partial) ==
                D::kPreserveUnsupportedPartialSignal,
            "partially routed signal was treated as fresh routing work");
    partial.has_inter_site_pip = false;
    partial.has_stub_nodes = true;
    require(ri::classify_physical_net(partial) ==
                D::kPreserveUnsupportedPartialSignal,
            "signal with stubNodes was treated as fresh routing work");

    F nested = routable;
    nested.top_level_stubs_are_site_pins = false;
    require(ri::classify_physical_net(nested) ==
                D::kPreserveUnsupportedSignalShape,
            "non-sitePin top-level stub was accepted");

    F site_pin_with_children = routable;
    site_pin_with_children.top_level_stubs_are_site_pins = true;
    require(ri::classify_physical_net(site_pin_with_children) ==
                D::kRouteSignal,
            "top-level sitePin stub with children was rejected");

    F static_net = routable;
    static_net.is_signal = false;
    require(ri::classify_physical_net(static_net) ==
                D::kPreserveCompleteOrLoadless,
            "pre-routed static net was not preserved");
    static_net.source_site_pin_count = 0;
    static_net.source_site_pins_are_leaves = false;
    static_net.has_inter_site_pip = true;
    static_net.has_stub_nodes = true;
    static_net.top_level_stubs_are_site_pins = false;
    require(ri::classify_physical_net(static_net) ==
                D::kPreserveCompleteOrLoadless,
            "static routing shape affected preservation policy");
    static_net.top_level_stub_count = 0;
    require(ri::classify_physical_net(static_net) ==
                D::kPreserveCompleteOrLoadless,
            "loadless static net was not preserved");

    require(ri::include_pip_in_static_graph(true) &&
                !ri::include_pip_in_static_graph(false),
            "pseudo-PIP inclusion policy is unsafe");
    require(ri::attachment_traversed_site_type_is_compatible(false, false) &&
                ri::attachment_traversed_site_type_is_compatible(false, true) &&
                ri::attachment_traversed_site_type_is_compatible(true, true) &&
                !ri::attachment_traversed_site_type_is_compatible(true, false),
            "missing or incompatible traversed-site type policy is wrong");
    const auto iob_source = ri::classify_audited_iob_attachment_pip(
        "xcvu3p", "XIPHY_BYTE_L", "XIPHY_BITSLICE_TILE_67_RX_D_PIN",
        "XIPHY_BITSLICE_TILE_67_RX_Q5", false, true);
    const auto iob_sink = ri::classify_audited_iob_attachment_pip(
        "xcvu3p", "XIPHY_BYTE_L", "XIPHY_BITSLICE_TILE_22_TX_D0",
        "XIPHY_BITSLICE_TILE_22_TX_Q", false, true);
    require(iob_source.has_value() &&
                iob_source->role == ri::IobAttachmentRole::kSource &&
                iob_source->from_site_pin == "RX_D" &&
                iob_source->to_site_pin == "RX_Q5" &&
                iob_sink.has_value() &&
                iob_sink->role == ri::IobAttachmentRole::kSink &&
                iob_sink->from_site_pin == "TX_D0" &&
                iob_sink->to_site_pin == "TX_Q",
            "audited xcvu3p IOB attachment PIPs were not recognized");
    require(!ri::classify_audited_iob_attachment_pip(
                 "xcvu3p", "XIPHY_BYTE_L",
                 "XIPHY_BITSLICE_TILE_67_RX_Q5",
                 "XIPHY_BITSLICE_TILE_67_RX_D_PIN", false, true)
                 .has_value() &&
                !ri::classify_audited_iob_attachment_pip(
                     "xcvu3p", "XIPHY_BYTE_L",
                     "XIPHY_BITSLICE_TILE_67_RX_D_PIN",
                     "XIPHY_BITSLICE_TILE_68_RX_Q5", false, true)
                     .has_value() &&
                !ri::classify_audited_iob_attachment_pip(
                     "xcvu3p", "XIPHY_BYTE_L",
                     "XIPHY_BITSLICE_TILE_67_RX_D_PIN",
                     "XIPHY_BITSLICE_TILE_67_RX_Q5", true, true)
                     .has_value() &&
                !ri::classify_audited_iob_attachment_pip(
                     "xcvu3p", "XIPHY_BYTE_L",
                     "XIPHY_BITSLICE_TILE_67_RX_D_PIN",
                     "XIPHY_BITSLICE_TILE_67_RX_Q5", false, false)
                     .has_value() &&
                !ri::classify_audited_iob_attachment_pip(
                     "xcvu3p", "CLEL_R", "A", "B", false, true)
                     .has_value(),
            "unsupported, reversed, or malformed pseudo PIP was admitted");
    std::uint32_t source_resource_mask = 0;
    for (const auto& resource :
         std::vector<std::pair<std::string, std::string>>{
             {"RX_Q5", "RX_Q5"},
             {"RXTX_BITSLICE", "DATAIN"},
             {"RXTX_BITSLICE", "Q5"},
             {"RX_D", "RX_D"}}) {
      const auto bit = ri::audited_iob_pseudo_resource_bit(
          ri::IobAttachmentRole::kSource, resource.first, resource.second);
      require(bit.has_value(), "audited source pseudo resource was rejected");
      source_resource_mask |= 1U << *bit;
    }
    require(source_resource_mask == 0xfU &&
                !ri::audited_iob_pseudo_resource_bit(
                     ri::IobAttachmentRole::kSource, "RXTX_BITSLICE",
                     "D0")
                     .has_value(),
            "source pseudo-cell signature is not exact");
    require(ri::is_audited_iob_endpoint(
                ri::IobAttachmentRole::kSource, "IOB_X1Y41", "HPIOB_M",
                "I") &&
                ri::is_audited_iob_endpoint(
                    ri::IobAttachmentRole::kSink, "IOB_X2Y3",
                    "HPIOB_SNGL", "OP") &&
                !ri::is_audited_iob_endpoint(
                    ri::IobAttachmentRole::kSource, "IOB_X1Y41",
                    "HPIOB_M", "OP") &&
                !ri::is_audited_iob_endpoint(
                    ri::IobAttachmentRole::kSink, "IOB_X1Y41_suffix",
                    "HPIOB_M", "OP") &&
                !ri::is_audited_iob_endpoint(
                    ri::IobAttachmentRole::kSink, "IOB_X1Y41", "SLICEL",
                    "OP"),
            "typed IOB endpoint admission is too broad or has wrong role");
    require(ri::physical_part_matches_device(
                "xcvu3p", "xcvu3p-ffvc1517-2-e") &&
                ri::physical_part_matches_device("xcvu3p", "xcvu3p") &&
                !ri::physical_part_matches_device(
                    "xcvu3p", "xcvu9p-flgb2104-2-i") &&
                !ri::physical_part_matches_device("xcvu3p", "xcvu3plus") &&
                !ri::physical_part_matches_device("", "") &&
                !ri::physical_part_matches_device("", "-part") &&
                !ri::physical_part_matches_device("xcvu3p", ""),
            "physical part/device identity policy is wrong");
    require(ri::unresolved_resource_is_fatal(true) &&
                !ri::unresolved_resource_is_fatal(false),
            "full/bounded unresolved-resource policy is wrong");
    require(ri::sink_requires_terminal_row(false) &&
                !ri::sink_requires_terminal_row(true),
            "source/sink terminal-row policy is wrong");
    require(ri::fixed_site_pin_candidate_fallback_allowed(false) &&
                !ri::fixed_site_pin_candidate_fallback_allowed(true),
            "typed fixed-site fallback policy is wrong");
    require(ri::is_reserved_used_resource_net("GLOBAL_USEDNET", true) &&
                !ri::is_reserved_used_resource_net("GLOBAL_USEDNET", false) &&
                !ri::is_reserved_used_resource_net("global_usednet", true) &&
                !ri::is_reserved_used_resource_net("GLOBAL_USEDNET_suffix",
                                                    true),
            "RapidWright used-resource sentinel policy is wrong");
    std::unordered_map<std::string, std::size_t> physical_net_names;
    require(ri::claim_unique_physical_net_name(
                physical_net_names, "net_a", 0) &&
                !ri::claim_unique_physical_net_name(
                    physical_net_names, "net_a", 1),
            "duplicate physical net name was accepted");
    std::unordered_map<std::string, std::uint64_t> logical_net_names;
    std::unordered_set<std::string> ambiguous_logical_net_names;
    ri::index_unambiguous_logical_net_name(
        logical_net_names, ambiguous_logical_net_names, "unique", 4);
    ri::index_unambiguous_logical_net_name(
        logical_net_names, ambiguous_logical_net_names, "same", 6);
    ri::index_unambiguous_logical_net_name(
        logical_net_names, ambiguous_logical_net_names, "same", 7);
    ri::index_unambiguous_logical_net_name(
        logical_net_names, ambiguous_logical_net_names, "same", 8);
    require(logical_net_names.at("unique") == 4 &&
                logical_net_names.find("same") == logical_net_names.end() &&
                ambiguous_logical_net_names.count("same") == 1,
            "ambiguous scoped logical net name was not dropped");
    require(ri::paired_static_slice_output_pin(
                "xcvu3p", "SLICE_X1Y2", std::string("SLICEL"),
                "A_O") == "AMUX" &&
                ri::paired_static_slice_output_pin(
                    "xcvu3p", "SLICE_X1Y2", std::string("SLICEM"),
                    "HMUX") == "H_O" &&
                ri::paired_static_slice_output_pin(
                    "xcvu3p", "SLICE_X1Y2", std::nullopt, "B_O") ==
                    "BMUX" &&
                !ri::paired_static_slice_output_pin(
                     "xcvu3p", "SLICE_X1Y2", std::string("SLICEL"),
                     "AQ")
                     .has_value() &&
                !ri::paired_static_slice_output_pin(
                     "xcvu3p", "SLICE_X1Y2", std::string("SLICEL"),
                     "I_O")
                     .has_value() &&
                !ri::paired_static_slice_output_pin(
                     "xcvu3p", "SLICE_X1Y2", std::string("SLICEX"),
                     "A_O")
                     .has_value() &&
                !ri::paired_static_slice_output_pin(
                     "xcvu3p", "BITSLICE_X1Y2", std::nullopt, "A_O")
                     .has_value() &&
                !ri::paired_static_slice_output_pin(
                     "xcvp1002", "SLICE_X1Y2", std::string("SLICEL"),
                     "A_O")
                     .has_value(),
            "static SLICE output pairing policy is wrong");
    require(ri::interchange_paths_alias("fixture/../input.phys",
                                        "input.phys") &&
                !ri::interchange_paths_alias("input.phys",
                                             "output.csrbin"),
            "interchange path alias policy is wrong");
    require_throws(
        [] {
          ri::require_distinct_interchange_paths(
              {"input.phys", "fixture/../input.phys"});
        },
        "aliased interchange input/output paths were accepted");
    require_throws(
        [] { ri::require_distinct_interchange_paths({""}); },
        "empty interchange path was accepted");
    TemporaryDirectory temporary;
    const std::filesystem::path original = temporary.path() / "original";
    const std::filesystem::path hard_link = temporary.path() / "hard-link";
    const std::filesystem::path symbolic_link =
        temporary.path() / "symbolic-link";
    {
      std::ofstream output(original);
      output << "fixture";
    }
    std::filesystem::create_hard_link(original, hard_link);
    std::filesystem::create_symlink(original, symbolic_link);
    require(ri::interchange_paths_alias(original, hard_link) &&
                ri::interchange_paths_alias(original, symbolic_link),
            "filesystem aliases were not rejected");

    const ri::InterchangeArtifactPairId pair_one{
        0x0123456789abcdefULL, 0xfedcba9876543210ULL};
    const ri::InterchangeArtifactPairId pair_two{
        0x1111111111111111ULL, 0x2222222222222222ULL};
    const std::string pair_one_text =
        ri::interchange_artifact_pair_id_string(pair_one);
    require(pair_one_text == "0123456789abcdeffedcba9876543210" &&
                ri::parse_interchange_artifact_pair_id(pair_one_text) ==
                    pair_one &&
                ri::parse_interchange_artifact_pair_id(
                    "0123456789ABCDEFFEDCBA9876543210") == pair_one &&
                !ri::derive_interchange_artifact_pair_id("fixture").is_zero(),
            "artifact pair id encoding policy is wrong");
    require_throws(
        [] {
          (void)ri::interchange_artifact_pair_id_string(
              ri::InterchangeArtifactPairId{});
        },
        "zero artifact pair id was encoded");
    require_throws(
        [] {
          (void)ri::parse_interchange_artifact_pair_id(
              "00000000000000000000000000000000");
        },
        "zero artifact pair id was parsed");
    require_throws(
        [] { (void)ri::parse_interchange_artifact_pair_id("not-hex"); },
        "malformed artifact pair id was parsed");
    ri::require_matching_interchange_pair_ids(std::nullopt, std::nullopt,
                                               std::nullopt);
    ri::require_matching_interchange_pair_ids(pair_one, pair_one, pair_one);
    require_throws(
        [&] {
          ri::require_matching_interchange_pair_ids(pair_one, pair_two,
                                                     pair_one);
        },
        "mismatched CSR/metadata ids were accepted");
    require_throws(
        [&] {
          ri::require_matching_interchange_pair_ids(pair_one, pair_one,
                                                     std::nullopt);
        },
        "new artifact pair without a generation was accepted");
    require_throws(
        [&] {
          ri::require_matching_interchange_pair_ids(std::nullopt,
                                                     std::nullopt, pair_one);
        },
        "legacy artifact pair with a generation was accepted");

    const std::filesystem::path csr = temporary.path() / "graph.csrbin";
    const std::filesystem::path metadata =
        temporary.path() / "graph.ifmeta.bin";
    const ri::InterchangePublicationSnapshot no_generation =
        ri::snapshot_interchange_publication(csr, metadata);
    require(!no_generation.generation.has_value(),
            "missing publication generation was fabricated");
    const std::filesystem::path csr_marker =
        ri::interchange_publication_marker_path(csr);
    std::filesystem::create_directory(csr_marker);
    require_throws(
        [&] { (void)ri::snapshot_interchange_publication(csr, metadata); },
        "active/incomplete CSR publication was accepted");
    std::filesystem::remove(csr_marker);
    const std::filesystem::path metadata_marker =
        ri::interchange_publication_marker_path(metadata);
    std::filesystem::create_directory(metadata_marker);
    require_throws(
        [&] { (void)ri::snapshot_interchange_publication(csr, metadata); },
        "active/incomplete metadata publication was accepted");
    std::filesystem::remove(metadata_marker);
    const std::filesystem::path generation =
        ri::interchange_publication_generation_path(metadata);
    {
      std::ofstream output(generation, std::ios::binary);
      output << pair_one_text << '\n';
    }
    const ri::InterchangePublicationSnapshot generation_one =
        ri::snapshot_interchange_publication(csr, metadata);
    ri::verify_interchange_publication(csr, metadata, generation_one);
    {
      std::ofstream output(generation, std::ios::binary | std::ios::trunc);
      output << ri::interchange_artifact_pair_id_string(pair_two) << '\n';
    }
    require_throws(
        [&] {
          ri::verify_interchange_publication(csr, metadata, generation_one);
        },
        "changed CSR/metadata generation was accepted");
    std::vector<std::uint8_t> blocked(4, 0);
    ri::preserve_pip_endpoints(blocked, 1, 2);
    require(blocked == std::vector<std::uint8_t>({0, 1, 1, 0}),
            "fixed PIP did not reserve both endpoints");
    ri::preserve_node(blocked, 0);
    require(blocked[0] == 1,
            "fixed site pin/stub node was not reserved");
    require_throws([&] { ri::preserve_node(blocked, -1); },
                   "invalid fixed node was silently ignored");
    std::vector<std::uint8_t> sink_stops(4, 0);
    ri::mark_sink_terminal(sink_stops, 2);
    require(sink_stops == std::vector<std::uint8_t>({0, 0, 1, 0}),
            "routable sink was not made terminal");
    require_throws([&] { ri::mark_sink_terminal(sink_stops, 4); },
                   "invalid sink terminal was silently ignored");
    std::vector<std::uint8_t> exclusive_sources(4, 0);
    ri::mark_source_exclusive(exclusive_sources, 1);
    require(exclusive_sources ==
                std::vector<std::uint8_t>({0, 1, 0, 0}),
            "routable source was not made exclusive");
    require_throws(
        [&] { ri::mark_source_exclusive(exclusive_sources, -1); },
        "invalid exclusive source was silently ignored");
    std::unordered_map<std::int32_t, std::size_t> endpoint_owners;
    require(ri::claim_route_endpoint(endpoint_owners, 2, 10) ==
                ri::EndpointClaim::kNew &&
                ri::claim_route_endpoint(endpoint_owners, 2, 10) ==
                    ri::EndpointClaim::kSameOwner &&
                ri::claim_route_endpoint(endpoint_owners, 2, 11) ==
                    ri::EndpointClaim::kDifferentOwner,
            "route endpoint ownership transitions are wrong");
    require_throws(
        [&] { (void)ri::claim_route_endpoint(endpoint_owners, -1, 10); },
        "invalid route endpoint ownership claim was accepted");

    const std::vector<ri::SitePinTemplate> templates =
        ri::build_site_pin_templates(
            10, {100, 101, 102}, {200, 201, 202},
            {{11, {110, 111}, {2, 0}}, {12, {120}, {1}}});
    require(templates.size() == 6,
            "primary/alternate template count is wrong");
    require(templates[0].site_type_string == 10 &&
                templates[0].pin_string == 100 &&
                templates[0].tile_wire_string == 200,
            "primary site pin mapped incorrectly");
    require(templates[3].site_type_string == 11 &&
                templates[3].pin_string == 110 &&
                templates[3].tile_wire_string == 202 &&
                templates[4].tile_wire_string == 200 &&
                templates[5].tile_wire_string == 201,
            "alternate pin did not map through its primary index");

    require_throws(
        [] { (void)ri::build_site_pin_templates(1, {2}, {}, {}); },
        "mismatched primary mapping was accepted");
    require_throws(
        [] {
          (void)ri::build_site_pin_templates(
              1, {2}, {3}, {{4, {5, 6}, {0}}});
        },
        "mismatched alternate mapping was accepted");
    require_throws(
        [] {
          (void)ri::build_site_pin_templates(
              1, {2}, {3}, {{4, {5}, {1}}});
        },
        "out-of-range alternate parent pin was accepted");

    std::cout << "interchange import policy test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "interchange import policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
