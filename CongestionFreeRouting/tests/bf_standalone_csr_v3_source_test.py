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
        "constexpr std::uint64_t CURRENT_CSR_VERSION = 3;" in source
        and "constexpr std::uint64_t ROUTING_SIDECAR_CSR_VERSION = 3;"
        in source,
        f"{context} does not accept the routing-sidecar CSR version",
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


def main() -> None:
    for source_path in SOURCES:
        check_reader(source_path)
    print("Standalone Bellman-Ford CSR v3 reader source test passed")


if __name__ == "__main__":
    main()
