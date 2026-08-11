#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(
    ROOT / "bellman_ford" / filename
    for filename in ("bf8.cpp", "bf9.cpp", "bf10.cpp")
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_between(text: str, begin: str, end: str) -> str:
    begin_offset = text.find(begin)
    require(begin_offset >= 0, f"missing source marker: {begin!r}")
    end_offset = text.find(end, begin_offset + len(begin))
    require(end_offset >= 0, f"missing source marker: {end!r}")
    return text[begin_offset:end_offset]


def check_reader(source_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    context = source_path.name

    require(
        "constexpr std::uint64_t ROUTING_SIDECAR_CSR_VERSION = 3;" in source
        and "constexpr std::uint64_t IMPLICIT_UNIT_CSR_VERSION = 4;" in source
        and "CURRENT_CSR_VERSION = IMPLICIT_UNIT_CSR_VERSION" in source,
        f"{context} does not define distinct v3/v4 CSR feature versions",
    )
    loader = source_between(
        source,
        "HostOutgoingCsrF32 load_outgoing_csrbin(",
        "DeviceOutgoingCsrOwner copy_outgoing_csr_to_device(",
    )

    values_header = loader.find(
        'read_u64(in, "outgoing CSR values count")'
    )
    sidecar_header = loader.find(
        'read_u64(in, "outgoing CSR route-end X count")'
    )
    core_payload = loader.find(
        'read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr")'
    )
    sidecar_payload = loader.find(
        '"outgoing CSR route-end X values"'
    )
    require(
        -1 not in (values_header, sidecar_header, core_payload, sidecar_payload)
        and values_header < sidecar_header < core_payload < sidecar_payload,
        f"{context} no longer parses the v3 header before the core payload "
        "and skips the v3 tail afterward",
    )

    for count_name in (
        "route_end_x_count != rows",
        "route_end_y_count != rows",
        "base_vertex_cost_count != rows",
        "spatial_edge_id_count != nnz",
        "spatial_offset_count != regular_shards + 2",
    ):
        require(
            count_name in loader,
            f"{context} lost v3 sidecar validation for {count_name}",
        )

    for feature_predicate in (
        "csr_has_artifact_pair(version)",
        "csr_has_routing_node_sidecars(version)",
        "csr_has_explicit_values(version)",
        "csr_has_spatial_edge_shards(version)",
        "csr_has_implicit_unit_values(version)",
    ):
        require(
            feature_predicate in loader,
            f"{context} does not branch on {feature_predicate}",
        )
    require(
        "version >= ROUTING_SIDECAR_CSR_VERSION" not in loader
        and "version >= CURRENT_CSR_VERSION" not in loader,
        f"{context} uses monotone version logic for incompatible v3/v4 payloads",
    )
    require(
        "values_count != 0" in loader
        and "spatial_width != 0" in loader
        and "spatial_height != 0" in loader
        and "spatial_offset_count != 0" in loader
        and "spatial_edge_id_count != 0" in loader,
        f"{context} does not reject nonzero v4 values/spatial fields",
    )
    require(
        "checked_size(nnz" in loader
        and "checked_byte_count(nnz, sizeof(float)" in loader
        and "graph.values.assign(implicit_value_count, 1.0f)" in loader,
        f"{context} does not synthesize checked implicit unit values",
    )

    for payload_type, payload_name in (
        ("std::int32_t", "route_end_x_count"),
        ("std::int32_t", "route_end_y_count"),
        ("float", "base_vertex_cost_count"),
        ("std::uint64_t", "spatial_offset_count"),
        ("std::uint32_t", "spatial_edge_id_count"),
    ):
        require(
            f"checked_byte_count({payload_name}" in loader
            and f"sizeof({payload_type})" in loader,
            f"{context} does not bounds-check the skipped {payload_name} payload",
        )

    require(
        "constexpr std::uint64_t COMPACT_EDGE_PIP_METADATA_VERSION = 8;"
        in source
        and "CURRENT_METADATA_VERSION =" in source,
        f"{context} does not define metadata v8 independently",
    )
    metadata_loader = source_between(
        source,
        "RoutingMetadata load_routing_metadata(",
        "bool valid_node(",
    )
    for feature_predicate in (
        "is_supported_metadata_version(version)",
        "metadata_has_artifact_pair(version)",
        "metadata_has_legacy_node_arrays(version)",
        "metadata_has_node_physical_arrays(version)",
        "metadata_has_endpoint_pips(version)",
        "metadata_has_compact_edge_pip_records(version)",
        "metadata_has_flat_logical_net_names(version)",
    ):
        require(
            feature_predicate in metadata_loader,
            f"{context} does not branch on {feature_predicate}",
        )
    require(
        "version >= ARTIFACT_PAIR_METADATA_VERSION" not in metadata_loader
        and "version >= NODE_PHYSICAL_METADATA_VERSION" not in metadata_loader
        and "version >= ENDPOINT_PIP_METADATA_VERSION" not in metadata_loader
        and "version >= CURRENT_METADATA_VERSION" not in metadata_loader,
        f"{context} uses monotone metadata-version logic across v3-v8",
    )
    require(
        "logical_cell_count != 0" in metadata_loader
        and "logical_port_instance_count != 0" in metadata_loader
        and "physical_netlist_byte_count != 0" in metadata_loader
        and "logical_netlist_byte_count != 0" in metadata_loader,
        f"{context} does not reject nonzero omitted metadata-v8 counts",
    )
    require(
        "string_count > std::numeric_limits<std::uint32_t>::max()"
        in metadata_loader
        and "pip_data_count > std::numeric_limits<std::uint32_t>::max()"
        in metadata_loader,
        f"{context} does not reject metadata-v8 compact counts beyond uint32",
    )
    require(
        "2 * sizeof(std::uint32_t)" in metadata_loader
        and "3 * sizeof(std::uint32_t)" in metadata_loader
        and "2 * sizeof(std::uint64_t)" in metadata_loader
        and "3 * sizeof(std::uint64_t)" in metadata_loader,
        f"{context} does not keep compact-v8 and legacy EdgeAttr/PIP widths",
    )
    require(
        "10 * sizeof(std::uint64_t)" in metadata_loader
        and "3 * sizeof(std::uint64_t)" in metadata_loader,
        f"{context} changed endpoint-PIP or site-pin record widths",
    )
    require(
        "std::vector<std::uint64_t> logical_net_name_strings" in metadata_loader
        and "checked_byte_count(logical_net_count, sizeof(std::uint64_t),"
        in metadata_loader
        and 'read_u64(in, "metadata logical net name string")'
        in metadata_loader
        and "physical/logical net-name correlation mismatch" in metadata_loader
        and "4 * sizeof(std::uint64_t)" in metadata_loader,
        f"{context} does not distinguish flat-v8 and legacy logical-net payloads",
    )
    require(
        'require_end_of_file(in, "metadata payload")' in metadata_loader,
        f"{context} does not reject trailing metadata bytes",
    )
    require(
        "regenerate metadata with interchange_to_csr" in metadata_loader,
        f"{context} stale metadata-version error omits regeneration guidance",
    )


def main() -> None:
    for source_path in SOURCES:
        check_reader(source_path)
    print(
        "Standalone Bellman-Ford CSR v1-v4/metadata v3-v8 reader source "
        "test passed"
    )


if __name__ == "__main__":
    main()
