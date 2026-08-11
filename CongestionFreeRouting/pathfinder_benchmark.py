#!/usr/bin/env python3
"""Contest-style benchmark entry point for the CSR PathFinder prototype.

The FPGA24 contest Makefile times routers with a rule shaped like:

    %_<router>.phys: %_unrouted.phys
        (/usr/bin/time <router-command> $< $@) ...

This wrapper accepts that two-argument router interface, infers the matching
logical netlist, builds the CSR/metadata sidecar with interchange_to_csr, runs
the C++ PathFinder executable, and writes a routed FPGAIF PhysicalNetlist by
replacing routed stubs with pip routeSegments.

Example contest Makefile recipe:

    %_pathfinder.phys: %_unrouted.phys %.netlist xcvu3p.full-poc-base-wire.devicegraph
        (/usr/bin/time python3 CongestionFreeRouting/pathfinder_benchmark.py $< $@) \
          $(call log_and_or_display,$@.log)

Manual run:

    python3 CongestionFreeRouting/pathfinder_benchmark.py vtr_mcml_unrouted.phys \
      vtr_mcml_pathfinder.phys --net-limit 100

Forced-generic weighted benchmark run:

    python3 CongestionFreeRouting/pathfinder_benchmark.py vtr_mcml_unrouted.phys \
      vtr_mcml_pathfinder.phys --sssp-engine delta-step --delta 1 \
      --delta-force-generic --delta-benchmark-weights mixed \
      --delta-benchmark-weight-seed 123
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA_CACHE: dict[Path, Any] = {}
_SCHEMA_CACHE_BY_ID: dict[str, tuple[Path, str, Any]] = {}
_METADATA_MAGIC = b"RIPSIFM1"
_NO_ENDPOINT_PIP = 2**64 - 1
_ENDPOINT_PIP_METADATA_VERSION = 7
_COMPACT_TABLE_METADATA_VERSION = 8


def _metadata_has_endpoint_pips(version: int) -> bool:
    return version in (_ENDPOINT_PIP_METADATA_VERSION,
                       _COMPACT_TABLE_METADATA_VERSION)


@dataclass(frozen=True)
class MetadataEndpointPip:
    csr_edge: int
    from_node: int
    to_node: int
    tile: str
    wire0: str
    wire1: str
    forward: bool
    site: str
    endpoint_node: int
    role: int


@dataclass(frozen=True)
class MetadataSitePin:
    node: int
    site: str
    pin: str
    endpoint_pip_index: int = _NO_ENDPOINT_PIP


@dataclass(frozen=True)
class MetadataRouteRequest:
    net: str
    sources: tuple[MetadataSitePin, ...]
    sinks: tuple[MetadataSitePin, ...]


@dataclass(frozen=True)
class RoutingMetadataSummary:
    version: int
    artifact_pair_id: str | None
    node_count: int
    edge_attr_count: int
    endpoint_pips: tuple[MetadataEndpointPip, ...]
    route_requests: tuple[MetadataRouteRequest, ...]


def infer_logical_netlist(unrouted_phys: Path) -> Path:
    name = unrouted_phys.name
    suffix = "_unrouted.phys"
    if name.endswith(suffix):
        return unrouted_phys.with_name(name[: -len(suffix)] + ".netlist")
    if name.endswith(".phys"):
        return unrouted_phys.with_suffix(".netlist")
    raise ValueError(
        "could not infer logical netlist path; pass --logical-netlist explicitly"
    )


def default_work_dir(output_phys: Path) -> Path:
    return output_phys.with_suffix(output_phys.suffix + ".pathfinder-work")


def run_command(argv: list[str], label: str) -> None:
    print(f"[pathfinder-benchmark] {label}: {' '.join(argv)}", flush=True)
    subprocess.run(argv, check=True)


def executable_path(value: str, env_name: str) -> str:
    override = os.environ.get(env_name)
    return override if override else value


def delta_arg(value: str) -> str:
    if value == "auto":
        return value
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("delta must be positive or 'auto'") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise argparse.ArgumentTypeError("delta must be positive or 'auto'")
    return value


def positive_float_arg(value: str) -> float:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be finite and positive") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return numeric


def nonnegative_int_arg(value: str) -> int:
    try:
        numeric = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be an unsigned 64-bit integer"
        ) from exc
    if numeric < 0 or numeric > (1 << 64) - 1:
        raise argparse.ArgumentTypeError(
            "value must be an unsigned 64-bit integer"
        )
    return numeric


def positive_int_arg(value: str) -> int:
    try:
        numeric = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive 32-bit integer"
        ) from exc
    if numeric <= 0 or numeric > (1 << 31) - 1:
        raise argparse.ArgumentTypeError(
            "value must be a positive 32-bit integer"
        )
    return numeric


def nonnegative_i32_arg(value: str) -> int:
    try:
        numeric = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative 32-bit integer"
        ) from exc
    if numeric < 0 or numeric > (1 << 31) - 1:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative 32-bit integer"
        )
    return numeric


def bf11_segment_rounds_arg(value: str) -> int:
    try:
        numeric = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "BF11 segment rounds must be one of 1, 2, 4, 8, or 16"
        ) from exc
    if numeric not in (1, 2, 4, 8, 16):
        raise argparse.ArgumentTypeError(
            "BF11 segment rounds must be one of 1, 2, 4, 8, or 16"
        )
    return numeric


def bf11_reset_threshold_arg(value: str) -> float:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "BF11 adaptive reset threshold must be finite and in (0, 1]"
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
        raise argparse.ArgumentTypeError(
            "BF11 adaptive reset threshold must be finite and in (0, 1]"
        )
    return numeric


def default_schema_dir() -> Path | None:
    env_schema = os.environ.get("FPGA_INTERCHANGE_SCHEMA_DIR")
    if env_schema:
        return Path(env_schema)

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "fpga-interchange-schema" / "interchange",
        repo_root.parent / "fpga-interchange-schema" / "interchange",
        Path.cwd() / "fpga-interchange-schema" / "interchange",
        Path.cwd() / "interchange",
    )
    for candidate in candidates:
        if (candidate / "PhysicalNetlist.capnp").exists():
            return candidate
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CSR PathFinder prototype through the FPGA24 router interface."
    )
    parser.add_argument("input_phys", type=Path)
    parser.add_argument("output_phys", type=Path)
    parser.add_argument("--logical-netlist", type=Path)
    parser.add_argument(
        "--device-graph",
        type=Path,
        default=Path(
            executable_path(
                "xcvu3p.full-poc-base-wire.devicegraph", "DEVICE_ROUTING_GRAPH"
            )
        ),
        help="precomputed device routing graph, or set DEVICE_ROUTING_GRAPH",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=default_schema_dir(),
        help="directory containing PhysicalNetlist.capnp, or set FPGA_INTERCHANGE_SCHEMA_DIR",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--interchange-to-csr",
        default=executable_path("interchange_to_csr", "INTERCHANGE_TO_CSR"),
        help="converter executable path, or set INTERCHANGE_TO_CSR",
    )
    parser.add_argument(
        "--pathfinder",
        default=executable_path("pathfinder", "PATHFINDER_BIN"),
        help="PathFinder executable path, or set PATHFINDER_BIN",
    )
    engine = parser.add_mutually_exclusive_group()
    engine.add_argument(
        "--sssp-engine",
        choices=("unit-bfs", "delta-step", "bellman-ford", "bf11"),
        help="shortest-path backend forwarded to PathFinder",
    )
    engine.add_argument(
        "--use-delta-step",
        action="store_true",
        help="shorthand forwarded to select the delta-step backend",
    )
    parser.add_argument(
        "--delta", type=delta_arg, help="positive delta-step bucket width or 'auto'"
    )
    parser.add_argument(
        "--delta-multiplier",
        type=positive_float_arg,
        help="positive sweep multiplier used with --delta auto",
    )
    parser.add_argument(
        "--delta-force-generic",
        action="store_true",
        help="force generic delta-stepping even when exact-unit specialization is eligible",
    )
    parser.add_argument(
        "--delta-telemetry",
        action="store_true",
        help="emit opt-in delta-stepping runtime telemetry",
    )
    parser.add_argument(
        "--delta-force-legacy-parent",
        action="store_true",
        help="force legacy generic-delta predecessor recovery",
    )
    parser.add_argument(
        "--delta-controller",
        choices=("host-checked", "reduced-round-trip"),
        help="generic delta-stepping controller mode",
    )
    parser.add_argument(
        "--delta-controller-batch-size",
        type=positive_int_arg,
        help="positive batch size for the reduced-round-trip controller",
    )
    parser.add_argument(
        "--delta-benchmark-weights",
        choices=("unit", "all-light", "all-heavy", "mixed"),
        help="reproducible benchmark edge-weight family forwarded to PathFinder",
    )
    parser.add_argument(
        "--delta-benchmark-weight-seed",
        type=nonnegative_int_arg,
        help="uint64 seed for mixed reproducible benchmark edge weights",
    )
    parser.add_argument(
        "--max-pathfinder-iters",
        type=int,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--max-sssp-iters",
        type=int,
        help="SSSP round/depth cap",
    )
    parser.add_argument(
        "--bf11-unbounded",
        action="store_true",
        help="disable BF11 automatic endpoint bounding",
    )
    parser.add_argument(
        "--bf11-bbox-margin-x",
        type=nonnegative_i32_arg,
        help="nonnegative BF11 horizontal bounding margin",
    )
    parser.add_argument(
        "--bf11-bbox-margin-y",
        type=nonnegative_i32_arg,
        help="nonnegative BF11 vertical bounding margin",
    )
    parser.add_argument(
        "--bf11-target-check-interval",
        type=positive_int_arg,
        help="positive BF11 device target-check interval",
    )
    parser.add_argument(
        "--bf11-segment-rounds",
        type=bf11_segment_rounds_arg,
        help="BF11 explicit-stream segment size: 1, 2, 4, 8, or 16",
    )
    parser.add_argument(
        "--bf11-hip-graph",
        choices=("auto", "on", "off"),
        help="BF11 HIP Graph replay policy for multi-round segments",
    )
    parser.add_argument(
        "--bf11-adaptive-reset-threshold",
        type=bf11_reset_threshold_arg,
        help="BF11 dense-reset touched fraction in (0, 1]",
    )
    parser.add_argument(
        "--bf11-no-unbounded-fallback",
        action="store_true",
        help="do not retry an unreachable bounded BF11 query unbounded",
    )
    parser.add_argument(
        "--bf11-telemetry",
        action="store_true",
        help="emit opt-in aggregate BF11 phase, work, and memory telemetry",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        help="capacity used only for overuse diagnostics",
    )
    parser.add_argument(
        "--present-factor",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--present-multiplier",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--history-factor",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument("--net-limit", type=int)
    parser.add_argument("--parallel-net-workers", type=int)
    parser.add_argument(
        "--route-batch-size",
        type=int,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="keep generated .csrbin/.ifmeta.bin files after the run",
    )
    parser.add_argument(
        "--allow-unrouted-stubs",
        action="store_true",
        help="write any stubs not covered by PathFinder back to the output netlist",
    )
    args = parser.parse_args(argv)

    delta_selected = args.use_delta_step or args.sssp_engine == "delta-step"
    delta_specific_controls = (
        args.delta is not None
        or args.delta_multiplier is not None
        or args.delta_force_generic
        or args.delta_force_legacy_parent
        or args.delta_telemetry
        or args.delta_controller is not None
        or args.delta_controller_batch_size is not None
        or args.delta_benchmark_weights is not None
        or args.delta_benchmark_weight_seed is not None
    )
    if delta_specific_controls and not delta_selected:
        parser.error(
            "delta-specific options require --sssp-engine delta-step or "
            "--use-delta-step"
        )
    bf11_selected = args.sssp_engine == "bf11"
    bf11_specific_controls = (
        args.bf11_unbounded
        or args.bf11_bbox_margin_x is not None
        or args.bf11_bbox_margin_y is not None
        or args.bf11_target_check_interval is not None
        or args.bf11_segment_rounds is not None
        or args.bf11_hip_graph is not None
        or args.bf11_adaptive_reset_threshold is not None
        or args.bf11_no_unbounded_fallback
        or args.bf11_telemetry
    )
    if bf11_specific_controls and not bf11_selected:
        parser.error("BF11-specific options require --sssp-engine bf11")
    if args.delta_multiplier is not None and args.delta != "auto":
        parser.error("--delta-multiplier requires --delta auto")
    if args.delta_controller_batch_size is not None and (
        args.delta_controller != "reduced-round-trip"
    ):
        parser.error(
            "--delta-controller-batch-size requires "
            "--delta-controller reduced-round-trip"
        )
    if args.delta_benchmark_weights is not None and (
        args.delta is None or args.delta == "auto"
    ):
        parser.error(
            "--delta-benchmark-weights requires an explicit numeric --delta"
        )
    if (
        args.delta_benchmark_weight_seed is not None
        and args.delta_benchmark_weights != "mixed"
    ):
        parser.error(
            "--delta-benchmark-weight-seed requires "
            "--delta-benchmark-weights mixed"
        )
    return args


def pathfinder_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if args.use_delta_step:
        forwarded.append("--use-delta-step")
    if args.delta_force_generic:
        forwarded.append("--delta-force-generic")
    if args.delta_telemetry:
        forwarded.append("--delta-telemetry")
    if args.delta_force_legacy_parent:
        forwarded.append("--delta-force-legacy-parent")
    if args.bf11_unbounded:
        forwarded.append("--bf11-unbounded")
    if args.bf11_no_unbounded_fallback:
        forwarded.append("--bf11-no-unbounded-fallback")
    if args.bf11_telemetry:
        forwarded.append("--bf11-telemetry")
    for attr, option in (
        ("sssp_engine", "--sssp-engine"),
        ("delta", "--delta"),
        ("delta_multiplier", "--delta-multiplier"),
        ("delta_controller", "--delta-controller"),
        ("delta_controller_batch_size", "--delta-controller-batch-size"),
        ("delta_benchmark_weights", "--delta-benchmark-weights"),
        ("delta_benchmark_weight_seed", "--delta-benchmark-weight-seed"),
        ("bf11_bbox_margin_x", "--bf11-bbox-margin-x"),
        ("bf11_bbox_margin_y", "--bf11-bbox-margin-y"),
        ("bf11_target_check_interval", "--bf11-target-check-interval"),
        ("bf11_segment_rounds", "--bf11-segment-rounds"),
        ("bf11_hip_graph", "--bf11-hip-graph"),
        (
            "bf11_adaptive_reset_threshold",
            "--bf11-adaptive-reset-threshold",
        ),
        ("max_pathfinder_iters", "--max-pathfinder-iters"),
        ("max_sssp_iters", "--max-sssp-iters"),
        ("capacity", "--capacity"),
        ("present_factor", "--present-factor"),
        ("present_multiplier", "--present-multiplier"),
        ("history_factor", "--history-factor"),
        ("net_limit", "--net-limit"),
        ("parallel_net_workers", "--parallel-net-workers"),
        ("route_batch_size", "--route-batch-size"),
    ):
        value = getattr(args, attr)
        if value is not None:
            forwarded.extend([option, str(value)])
    return forwarded


def read_gzip_or_plain(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def schema_identity(schema_path: Path) -> tuple[str, str]:
    schema_text = schema_path.read_text(encoding="utf-8")
    match = re.search(r"@0x([0-9a-fA-F]+);", schema_text)
    schema_id = match.group(1).lower() if match else str(schema_path.resolve())
    digest = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
    return schema_id, digest


def load_physical_schema(schema_dir: Path):
    schema_path = (schema_dir / "PhysicalNetlist.capnp").resolve()
    cached = _SCHEMA_CACHE.get(schema_path)
    if cached is not None:
        return cached
    if not schema_path.exists():
        raise FileNotFoundError(f"missing PhysicalNetlist.capnp in {schema_dir}")

    schema_id, schema_digest = schema_identity(schema_path)
    cached_by_id = _SCHEMA_CACHE_BY_ID.get(schema_id)
    if cached_by_id is not None:
        cached_path, cached_digest, cached_schema = cached_by_id
        if cached_digest != schema_digest:
            raise RuntimeError(
                "a different PhysicalNetlist.capnp with the same Cap'n Proto "
                f"schema ID is already loaded: {cached_path}"
            )
        _SCHEMA_CACHE[schema_path] = cached_schema
        return cached_schema

    try:
        import capnp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pycapnp is required to write legal FPGAIF .phys files"
        ) from exc

    schema = capnp.load(str(schema_path), imports=[str(schema_dir.resolve())])
    _SCHEMA_CACHE[schema_path] = schema
    _SCHEMA_CACHE_BY_ID[schema_id] = (schema_path, schema_digest, schema)
    return schema


def read_metadata_artifact_pair_id(metadata_path: Path) -> str | None:
    marker = Path(str(metadata_path) + ".publishing")
    if marker.exists():
        raise RuntimeError(
            f"interchange metadata publication is incomplete or active: {marker}"
        )
    with metadata_path.open("rb") as metadata_file:
        prefix = metadata_file.read(24)
        if len(prefix) != 24 or prefix[:8] != _METADATA_MAGIC:
            raise ValueError(f"{metadata_path} is not RIPS interchange metadata")
        version = int.from_bytes(prefix[8:16], byteorder=sys.byteorder)
        orientation = int.from_bytes(prefix[16:24], byteorder=sys.byteorder)
        if orientation != 2:
            raise ValueError(f"{metadata_path} does not use outgoing CSR orientation")
        if version == 4:
            pair_id = None
        elif version in (5, 6, 7, 8):
            raw_id = metadata_file.read(16)
            if len(raw_id) != 16:
                raise ValueError(f"{metadata_path} has a truncated artifact pair id")
            high = int.from_bytes(raw_id[:8], byteorder=sys.byteorder)
            low = int.from_bytes(raw_id[8:], byteorder=sys.byteorder)
            if high == 0 and low == 0:
                raise ValueError(f"{metadata_path} has a zero artifact pair id")
            pair_id = f"{high:016x}{low:016x}"
        else:
            raise ValueError(
                f"{metadata_path} has unsupported metadata version {version}; "
                "regenerate it with interchange_to_csr"
            )

    generation_path = Path(str(metadata_path) + ".generation")
    if pair_id is None:
        if generation_path.exists():
            raise ValueError("legacy metadata unexpectedly has a generation sidecar")
    else:
        try:
            generation = generation_path.read_text(encoding="ascii")
        except FileNotFoundError as exc:
            raise ValueError(
                f"generation-tagged metadata is missing {generation_path}"
            ) from exc
        if generation not in (pair_id, pair_id + "\n"):
            raise ValueError("metadata and publication generation ids do not match")
    if marker.exists():
        raise RuntimeError(
            f"interchange metadata publication changed while it was read: {marker}"
        )
    return pair_id


def _read_metadata_exact(metadata_file, byte_count: int, label: str) -> bytes:
    if byte_count < 0:
        raise ValueError(f"negative byte count while reading {label}")
    data = metadata_file.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"metadata is truncated while reading {label}")
    return data


def _read_metadata_u64(metadata_file, label: str) -> int:
    return int.from_bytes(
        _read_metadata_exact(metadata_file, 8, label),
        byteorder=sys.byteorder,
    )


def _skip_metadata_bytes(metadata_file, byte_count: int, label: str) -> None:
    if byte_count < 0:
        raise ValueError(f"negative byte count while skipping {label}")
    current = metadata_file.tell()
    metadata_file.seek(0, os.SEEK_END)
    end = metadata_file.tell()
    if end - current < byte_count:
        raise ValueError(f"metadata is truncated while skipping {label}")
    metadata_file.seek(current + byte_count)


def _metadata_route_node(raw: int, label: str) -> int:
    if raw == _NO_ENDPOINT_PIP:
        return -1
    if raw > 2**31 - 1:
        raise ValueError(f"{label} exceeds the C++ node-id range")
    return raw


def read_metadata_summary(metadata_path: Path) -> RoutingMetadataSummary:
    """Read the sparse reconstruction subset of RIPS metadata v4-v8.

    EdgeAttr and PipData are deliberately seek-skipped: on a full device those
    tables dominate memory, while each v7 EndpointPip repeats the exact tuple
    needed to authenticate and reconstruct an endpoint attachment.
    """

    initial_pair_id = read_metadata_artifact_pair_id(metadata_path)
    with metadata_path.open("rb") as metadata_file:
        magic = _read_metadata_exact(metadata_file, 8, "metadata magic")
        if magic != _METADATA_MAGIC:
            raise ValueError(f"{metadata_path} is not RIPS interchange metadata")
        version = _read_metadata_u64(metadata_file, "metadata version")
        orientation = _read_metadata_u64(metadata_file, "metadata orientation")
        if version not in (4, 5, 6, 7, 8):
            raise ValueError(
                f"{metadata_path} has unsupported metadata version {version}; "
                "regenerate it with interchange_to_csr"
            )
        if orientation != 2:
            raise ValueError(f"{metadata_path} does not use outgoing CSR orientation")

        if version in (5, 6, 7, 8):
            high = _read_metadata_u64(metadata_file, "artifact pair id high")
            low = _read_metadata_u64(metadata_file, "artifact pair id low")
            if high == 0 and low == 0:
                raise ValueError(f"{metadata_path} has a zero artifact pair id")
            pair_id = f"{high:016x}{low:016x}"
        else:
            pair_id = None
        if pair_id != initial_pair_id:
            raise ValueError("metadata artifact pair id changed while it was read")

        string_count = _read_metadata_u64(metadata_file, "string count")
        node_count = _read_metadata_u64(metadata_file, "node count")
        edge_attr_count = _read_metadata_u64(metadata_file, "edge attr count")
        pip_data_count = _read_metadata_u64(metadata_file, "PIP data count")
        endpoint_pip_count = (
            _read_metadata_u64(metadata_file, "endpoint PIP count")
            if _metadata_has_endpoint_pips(version)
            else 0
        )
        site_pin_attr_count = _read_metadata_u64(
            metadata_file, "site-pin attr count"
        )
        route_request_count = _read_metadata_u64(
            metadata_file, "route request count"
        )
        blocked_node_count = _read_metadata_u64(
            metadata_file, "blocked node count"
        )
        sink_stop_node_count = _read_metadata_u64(
            metadata_file, "sink-stop node count"
        )
        logical_cell_count = _read_metadata_u64(
            metadata_file, "logical cell count"
        )
        logical_net_count = _read_metadata_u64(
            metadata_file, "logical net count"
        )
        logical_port_instance_count = _read_metadata_u64(
            metadata_file, "logical port-instance count"
        )
        physical_byte_count = _read_metadata_u64(
            metadata_file, "physical netlist byte count"
        )
        logical_byte_count = _read_metadata_u64(
            metadata_file, "logical netlist byte count"
        )
        if version == _COMPACT_TABLE_METADATA_VERSION:
            if string_count > 2**32 - 1 or pip_data_count > 2**32 - 1:
                raise ValueError(
                    "metadata v8 string/PIP counts exceed compact uint32 limits"
                )
            if (
                logical_cell_count != 0
                or logical_port_instance_count != 0
                or physical_byte_count != 0
                or logical_byte_count != 0
            ):
                raise ValueError(
                    "metadata v8 omitted hierarchy/payload counts must be zero"
                )
        for label in (
            "device path string",
            "physical path string",
            "logical path string",
            "logical design name string",
        ):
            _read_metadata_u64(metadata_file, label)

        strings: list[str] = []
        for _ in range(string_count):
            size = _read_metadata_u64(metadata_file, "metadata string length")
            raw = _read_metadata_exact(metadata_file, size, "metadata string")
            try:
                strings.append(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("metadata contains invalid UTF-8") from exc

        def metadata_string(index: int, label: str) -> str:
            if index >= len(strings):
                raise ValueError(f"{label} references an invalid string")
            return strings[index]

        if version in (4, 5):
            _skip_metadata_bytes(
                metadata_file, node_count * 40, "legacy node metadata"
            )
        edge_attr_bytes = 8 if version == _COMPACT_TABLE_METADATA_VERSION else 16
        pip_data_bytes = 12 if version == _COMPACT_TABLE_METADATA_VERSION else 24
        _skip_metadata_bytes(
            metadata_file, edge_attr_count * edge_attr_bytes, "edge attributes"
        )
        _skip_metadata_bytes(
            metadata_file, pip_data_count * pip_data_bytes, "PIP data"
        )

        endpoint_pips: list[MetadataEndpointPip] = []
        endpoint_edges: set[int] = set()
        for _ in range(endpoint_pip_count):
            raw = [
                _read_metadata_u64(metadata_file, "endpoint PIP field")
                for _ in range(10)
            ]
            (
                csr_edge,
                from_node,
                to_node,
                tile_string,
                wire0_string,
                wire1_string,
                forward,
                site_string,
                endpoint_node,
                role,
            ) = raw
            if csr_edge >= edge_attr_count:
                raise ValueError("endpoint PIP references an invalid CSR edge")
            if csr_edge in endpoint_edges:
                raise ValueError("duplicate endpoint PIPs reference one CSR edge")
            endpoint_edges.add(csr_edge)
            from_node = _metadata_route_node(from_node, "endpoint PIP from")
            to_node = _metadata_route_node(to_node, "endpoint PIP to")
            endpoint_node = _metadata_route_node(
                endpoint_node, "endpoint PIP endpoint node"
            )
            if (
                from_node < 0
                or to_node < 0
                or endpoint_node < 0
                or from_node >= node_count
                or to_node >= node_count
                or endpoint_node >= node_count
                or from_node == to_node
                or endpoint_node in (from_node, to_node)
            ):
                raise ValueError("endpoint PIP has invalid endpoint alignment")
            if forward not in (0, 1):
                raise ValueError("endpoint PIP has an invalid forward flag")
            if role not in (0, 1):
                raise ValueError("endpoint PIP has an invalid role")
            concrete_site = metadata_string(site_string, "endpoint PIP site")
            if not concrete_site:
                raise ValueError("endpoint PIP has an empty concrete site")
            endpoint_pips.append(
                MetadataEndpointPip(
                    csr_edge=csr_edge,
                    from_node=from_node,
                    to_node=to_node,
                    tile=metadata_string(tile_string, "endpoint PIP tile"),
                    wire0=metadata_string(wire0_string, "endpoint PIP wire0"),
                    wire1=metadata_string(wire1_string, "endpoint PIP wire1"),
                    forward=bool(forward),
                    site=concrete_site,
                    endpoint_node=endpoint_node,
                    role=role,
                )
            )

        _skip_metadata_bytes(
            metadata_file, site_pin_attr_count * 24, "site-pin attributes"
        )

        requests: list[MetadataRouteRequest] = []
        request_logical_indices: list[int] = []
        for _ in range(route_request_count):
            net = metadata_string(
                _read_metadata_u64(metadata_file, "route request net"),
                "route request net",
            )
            logical_net_index = _read_metadata_u64(
                metadata_file, "route request logical net"
            )

            def read_site_pins(role: int, label: str) -> tuple[MetadataSitePin, ...]:
                count = _read_metadata_u64(metadata_file, f"{label} count")
                pins: list[MetadataSitePin] = []
                for _ in range(count):
                    node = _metadata_route_node(
                        _read_metadata_u64(metadata_file, f"{label} node"),
                        f"metadata {label} node",
                    )
                    site = metadata_string(
                        _read_metadata_u64(metadata_file, f"{label} site"),
                        f"metadata {label} site",
                    )
                    pin = metadata_string(
                        _read_metadata_u64(metadata_file, f"{label} pin"),
                        f"metadata {label} pin",
                    )
                    endpoint_index = (
                        _read_metadata_u64(
                            metadata_file, f"{label} endpoint PIP index"
                        )
                        if _metadata_has_endpoint_pips(version)
                        else _NO_ENDPOINT_PIP
                    )
                    if endpoint_index != _NO_ENDPOINT_PIP:
                        if endpoint_index >= len(endpoint_pips):
                            raise ValueError(
                                f"metadata {label} references an invalid endpoint PIP"
                            )
                        endpoint = endpoint_pips[endpoint_index]
                        if (
                            endpoint.role != role
                            or endpoint.endpoint_node != node
                        ):
                            raise ValueError(
                                f"metadata {label} references an endpoint PIP "
                                "owned by a different endpoint or role"
                            )
                    pins.append(
                        MetadataSitePin(node, site, pin, endpoint_index)
                    )
                return tuple(pins)

            sources = read_site_pins(0, "source")
            sinks = read_site_pins(1, "sink")
            requests.append(MetadataRouteRequest(net, sources, sinks))
            request_logical_indices.append(logical_net_index)

        if version == _COMPACT_TABLE_METADATA_VERSION:
            logical_net_name_strings = [
                _read_metadata_u64(metadata_file, "logical net name string")
                for _ in range(logical_net_count)
            ]
            for name_string in logical_net_name_strings:
                metadata_string(name_string, "logical net name")
            for request, logical_net_index in zip(
                requests, request_logical_indices, strict=True
            ):
                if logical_net_index == _NO_ENDPOINT_PIP:
                    continue
                if logical_net_index >= len(logical_net_name_strings):
                    raise ValueError(
                        "metadata v8 route request references an invalid logical net"
                    )
                logical_name = metadata_string(
                    logical_net_name_strings[logical_net_index],
                    "logical net name",
                )
                if logical_name != request.net:
                    raise ValueError(
                        "metadata v8 physical/logical net-name correlation mismatch"
                    )
        else:
            _skip_metadata_bytes(
                metadata_file, logical_cell_count * 24, "logical cells"
            )
            _skip_metadata_bytes(
                metadata_file, logical_net_count * 32, "logical nets"
            )
            _skip_metadata_bytes(
                metadata_file,
                logical_port_instance_count * 56,
                "logical port instances",
            )
        _skip_metadata_bytes(
            metadata_file, blocked_node_count * 8, "blocked nodes"
        )
        _skip_metadata_bytes(
            metadata_file, sink_stop_node_count * 8, "sink-stop nodes"
        )
        if version != _COMPACT_TABLE_METADATA_VERSION:
            _skip_metadata_bytes(
                metadata_file, physical_byte_count, "physical netlist bytes"
            )
            _skip_metadata_bytes(
                metadata_file, logical_byte_count, "logical netlist bytes"
            )
        if metadata_file.read(1):
            raise ValueError("metadata has trailing bytes")

    if read_metadata_artifact_pair_id(metadata_path) != pair_id:
        raise ValueError("metadata publication changed while it was read")
    return RoutingMetadataSummary(
        version=version,
        artifact_pair_id=pair_id,
        node_count=node_count,
        edge_attr_count=edge_attr_count,
        endpoint_pips=tuple(endpoint_pips),
        route_requests=tuple(requests),
    )


def read_routes_jsonl(
    path: Path,
    expected_artifact_pair_id: str | None = None,
    metadata_summary: RoutingMetadataSummary | None = None,
) -> dict[str, dict[str, Any]]:
    if metadata_summary is not None:
        if (
            expected_artifact_pair_id is not None
            and expected_artifact_pair_id != metadata_summary.artifact_pair_id
        ):
            raise ValueError("metadata summary artifact pair id is inconsistent")
        expected_artifact_pair_id = metadata_summary.artifact_pair_id
    routes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as route_file:
        for line_no, line in enumerate(route_file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            route = json.loads(stripped)
            net_name = route.get("net")
            if not isinstance(net_name, str) or not net_name:
                raise ValueError(f"{path}:{line_no}: route entry has no net name")
            if route.get("routed") is not True:
                raise ValueError(f"{path}:{line_no}: net {net_name} is not fully routed")
            route_pair_id = route.get("artifact_pair_id")
            if route_pair_id != expected_artifact_pair_id:
                raise ValueError(
                    f"{path}:{line_no}: route artifact pair id does not match metadata"
                )
            if net_name in routes:
                raise ValueError(f"{path}:{line_no}: duplicate route for net {net_name}")
            routes[net_name] = route
    if not routes:
        raise ValueError(f"no routed nets were written to {path}")
    validate_routes_against_metadata(routes, metadata_summary)
    return routes


def string_at(str_list, index: int) -> str:
    return str_list[index]


def get_string_index(text: str, string_to_index: dict[str, int]) -> int:
    index = string_to_index.get(text)
    if index is None:
        index = len(string_to_index)
        string_to_index[text] = index
    return index


def site_pin_key(site: str, pin: str) -> tuple[str, str]:
    return (site, pin)


def route_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"route field {field} is not an integer")
    if value < -(2**31) or value > 2**31 - 1:
        raise ValueError(f"route field {field} exceeds the C++ node-id range")
    return value


def route_u64(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**64 - 1
    ):
        raise ValueError(f"route field {field} is not an unsigned 64-bit integer")
    return value


def _route_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"route field {field} is not a string")
    return value


def _route_attachment(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"route field {field} is not a nonnegative integer or null")
    if value > _NO_ENDPOINT_PIP:
        raise ValueError(f"route field {field} exceeds the metadata index range")
    return value


def validate_routes_against_metadata(
    routes: dict[str, dict[str, Any]],
    metadata: RoutingMetadataSummary | None,
) -> None:
    requests_by_net: dict[str, MetadataRouteRequest] = {}
    endpoint_by_csr_edge: dict[int, int] = {}
    if metadata is not None:
        for request in metadata.route_requests:
            if request.net in requests_by_net:
                raise ValueError(
                    f"metadata contains duplicate route request {request.net}"
                )
            requests_by_net[request.net] = request
        for index, endpoint in enumerate(metadata.endpoint_pips):
            if endpoint.csr_edge in endpoint_by_csr_edge:
                raise ValueError("metadata contains duplicate endpoint-PIP CSR edges")
            endpoint_by_csr_edge[endpoint.csr_edge] = index

    for net_name, route in routes.items():
        raw_sources = route.get("sources")
        raw_sinks = route.get("sinks")
        raw_edges = route.get("edges")
        if not isinstance(raw_sources, list) or not isinstance(raw_sinks, list):
            raise ValueError(f"net {net_name} has invalid source/sink arrays")
        if not isinstance(raw_edges, list):
            raise ValueError(f"net {net_name} has an invalid edge array")

        request = requests_by_net.get(net_name) if metadata is not None else None
        if metadata is not None and request is None:
            raise ValueError(f"route net {net_name} is not present in metadata")
        if request is not None:
            if len(raw_sources) != len(request.sources):
                raise ValueError(f"route source count does not match metadata for {net_name}")
            if len(raw_sinks) != len(request.sinks):
                raise ValueError(f"route sink count does not match metadata for {net_name}")

        sources: list[tuple[int, str, str]] = []
        for index, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                raise ValueError(f"net {net_name} source {index} is not an object")
            actual = (
                route_int(source.get("node"), "source.node"),
                _route_string(source.get("site"), "source.site"),
                _route_string(source.get("pin"), "source.pin"),
            )
            if request is not None:
                expected = request.sources[index]
                if actual != (expected.node, expected.site, expected.pin):
                    raise ValueError(
                        f"route source {index} does not match metadata for {net_name}"
                    )
            sources.append(actual)

        sinks: list[tuple[int, str, str, bool]] = []
        for index, sink in enumerate(raw_sinks):
            if not isinstance(sink, dict):
                raise ValueError(f"net {net_name} sink {index} is not an object")
            reached = sink.get("reached")
            if not isinstance(reached, bool):
                raise ValueError(f"net {net_name} sink {index} has invalid reached")
            actual = (
                route_int(sink.get("node"), "sink.node"),
                _route_string(sink.get("site"), "sink.site"),
                _route_string(sink.get("pin"), "sink.pin"),
                reached,
            )
            if request is not None:
                expected = request.sinks[index]
                if actual[:3] != (expected.node, expected.site, expected.pin):
                    raise ValueError(
                        f"route sink {index} does not match metadata for {net_name}"
                    )
            sinks.append(actual)

        authorized_sources: set[int] = set()
        authorized_reached_sinks: set[int] = set()
        source_attachment_by_node: dict[int, int] = {}
        if request is not None:
            for source in request.sources:
                endpoint_index = source.endpoint_pip_index
                if endpoint_index == _NO_ENDPOINT_PIP:
                    continue
                authorized_sources.add(endpoint_index)
                previous = source_attachment_by_node.setdefault(
                    source.node, endpoint_index
                )
                if previous != endpoint_index:
                    raise ValueError(
                        f"metadata has ambiguous source attachments for {net_name}"
                    )
            for index, sink in enumerate(request.sinks):
                if sinks[index][3] and sink.endpoint_pip_index != _NO_ENDPOINT_PIP:
                    authorized_reached_sinks.add(sink.endpoint_pip_index)

        incoming: dict[int, dict[str, Any]] = {}
        outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seen_pairs: set[tuple[int, int]] = set()
        seen_csr_edges: set[int] = set()
        used_attachments: dict[int, dict[str, Any]] = {}

        for edge_index, edge in enumerate(raw_edges):
            if not isinstance(edge, dict):
                raise ValueError(f"net {net_name} edge {edge_index} is not an object")
            parent = route_int(edge.get("from"), "edge.from")
            child = route_int(edge.get("to"), "edge.to")
            csr_edge = route_u64(edge.get("csr_edge"), "edge.csr_edge")
            if parent < 0 or child < 0 or parent == child:
                raise ValueError(f"net {net_name} contains an invalid route edge")
            if metadata is not None and (
                parent >= metadata.node_count or child >= metadata.node_count
            ):
                raise ValueError(f"net {net_name} references an invalid route node")
            if metadata is not None and csr_edge >= metadata.edge_attr_count:
                raise ValueError(f"net {net_name} references an invalid CSR edge")
            pair = (parent, child)
            if pair in seen_pairs or csr_edge in seen_csr_edges:
                raise ValueError(f"net {net_name} contains a duplicate route edge")
            seen_pairs.add(pair)
            seen_csr_edges.add(csr_edge)
            if child in incoming and route_int(
                incoming[child].get("from"), "edge.from"
            ) != parent:
                raise ValueError(
                    f"net {net_name} drives node {child} from multiple parents"
                )
            incoming[child] = edge
            outgoing[parent].append(edge)

            tile = _route_string(edge.get("tile"), "edge.tile")
            wire0 = _route_string(edge.get("wire0"), "edge.wire0")
            wire1 = _route_string(edge.get("wire1"), "edge.wire1")
            forward = edge.get("forward")
            if not isinstance(forward, bool):
                raise ValueError(f"net {net_name} edge.forward is not a bool")

            if (
                metadata is not None
                and _metadata_has_endpoint_pips(metadata.version)
                and ("attachment" not in edge or "site" not in edge)
            ):
                raise ValueError(
                    f"v7 route edge is missing attachment/site fields for {net_name}"
                )
            attachment = _route_attachment(
                edge.get("attachment"), "edge.attachment"
            )
            site_value = edge.get("site")
            if site_value is not None and not isinstance(site_value, str):
                raise ValueError(f"route field edge.site is not a string or null")
            if (attachment is None) != (site_value is None):
                raise ValueError(
                    f"net {net_name} edge must pair attachment and site"
                )

            expected_index = endpoint_by_csr_edge.get(csr_edge)
            if expected_index is None:
                if attachment is not None or site_value is not None:
                    raise ValueError(
                        f"conventional route edge carries attachment/site for {net_name}"
                    )
                continue
            if attachment is None or site_value is None:
                raise ValueError(
                    f"endpoint attachment is encoded as conventional for {net_name}"
                )
            if attachment != expected_index:
                raise ValueError(
                    f"attachment index does not match its CSR edge for {net_name}"
                )
            if attachment in used_attachments:
                raise ValueError(f"net {net_name} reuses an endpoint attachment")

            endpoint = metadata.endpoint_pips[expected_index]
            if (
                parent != endpoint.from_node
                or child != endpoint.to_node
                or tile != endpoint.tile
                or wire0 != endpoint.wire0
                or wire1 != endpoint.wire1
                or forward != endpoint.forward
                or site_value != endpoint.site
            ):
                raise ValueError(
                    f"route attachment does not exactly match sparse metadata for {net_name}"
                )
            if endpoint.role == 0:
                if attachment not in authorized_sources:
                    raise ValueError(
                        f"source attachment belongs to another endpoint for {net_name}"
                    )
            elif attachment not in authorized_reached_sinks:
                raise ValueError(
                    f"sink attachment belongs to another endpoint for {net_name}"
                )
            used_attachments[attachment] = edge

        if metadata is not None:
            for endpoint_index, attachment_edge in used_attachments.items():
                endpoint = metadata.endpoint_pips[endpoint_index]
                if endpoint.role == 0:
                    corridor = incoming.get(endpoint.from_node)
                    children = outgoing.get(endpoint.from_node, [])
                    root_children = outgoing.get(endpoint.endpoint_node, [])
                    if (
                        corridor is None
                        or route_int(corridor.get("from"), "edge.from")
                        != endpoint.endpoint_node
                        or corridor.get("attachment") is not None
                        or not any(child is corridor for child in root_children)
                        or not any(child is attachment_edge for child in children)
                        or endpoint.endpoint_node in incoming
                    ):
                        raise ValueError(
                            f"source attachment is outside its endpoint corridor "
                            f"or used for transit in {net_name}"
                        )
                else:
                    corridor_edges = outgoing.get(endpoint.to_node, [])
                    if (
                        len(corridor_edges) != 1
                        or route_int(corridor_edges[0].get("to"), "edge.to")
                        != endpoint.endpoint_node
                        or corridor_edges[0].get("attachment") is not None
                        or endpoint.endpoint_node in outgoing
                    ):
                        raise ValueError(
                            f"sink attachment is outside its endpoint corridor "
                            f"or used for transit in {net_name}"
                        )

def build_route_tables(route: dict[str, Any]):
    adjacency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    incoming_parent: dict[int, int] = {}
    seen_edges: set[tuple[int, int]] = set()

    for edge in route.get("edges", []):
        parent = route_int(edge["from"], "edge.from")
        child = route_int(edge["to"], "edge.to")
        key = (parent, child)
        if key in seen_edges:
            continue
        previous_parent = incoming_parent.get(child)
        if previous_parent is not None and previous_parent != parent:
            raise ValueError(
                f"net {route['net']} drives node {child} from both "
                f"{previous_parent} and {parent}"
            )
        incoming_parent[child] = parent
        seen_edges.add(key)
        adjacency[parent].append(edge)

    source_node_by_pin: dict[tuple[str, str], int] = {}
    for source in route.get("sources", []):
        key = site_pin_key(str(source["site"]), str(source["pin"]))
        node = route_int(source["node"], "source.node")
        previous = source_node_by_pin.get(key)
        if previous is not None and previous != node:
            raise ValueError(
                f"net {route['net']} maps source {key} to both "
                f"{previous} and {node}"
            )
        source_node_by_pin[key] = node

    sink_pins_by_node: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for sink in route.get("sinks", []):
        if not sink.get("reached", False):
            raise ValueError(f"net {route['net']} has unreached sink {sink}")
        key = site_pin_key(str(sink["site"]), str(sink["pin"]))
        sink_pins_by_node[route_int(sink["node"], "sink.node")].append(key)

    return adjacency, source_node_by_pin, sink_pins_by_node, seen_edges


def collect_site_pin_branches(branches, str_list):
    pins = []
    queue = list(branches)
    while queue:
        branch = queue.pop()
        route_segment = branch.routeSegment
        if route_segment.which() == "sitePin":
            site_pin = route_segment.sitePin
            pins.append(
                (
                    site_pin_key(
                        string_at(str_list, site_pin.site),
                        string_at(str_list, site_pin.pin),
                    ),
                    branch,
                )
            )
        queue.extend(branch.branches)
    return pins


def insert_route_tree(
    root_branch,
    root_node: int,
    net_name: str,
    adjacency: dict[int, list[dict[str, Any]]],
    sink_pins_by_node: dict[int, list[tuple[str, str]]],
    sink_pin_orphans: dict[tuple[str, str], list[Any]],
    string_to_index: dict[str, int],
) -> int:
    emitted_pips = 0
    stack: list[tuple[Any, int, tuple[int, ...]]] = [(root_branch, root_node, ())]

    while stack:
        branch, node, ancestors = stack.pop()
        if node in ancestors:
            raise ValueError(f"net {net_name} route tree has a cycle at node {node}")
        child_edges = adjacency.get(node, [])
        sink_keys = sink_pins_by_node.get(node, [])
        branch_count = len(child_edges) + len(sink_keys)
        if branch_count == 0:
            continue
        if len(branch.branches) != 0:
            raise ValueError(
                f"net {net_name} source/tree branch already has child branches"
            )

        new_branches = branch.init("branches", branch_count)
        branch_index = 0
        next_ancestors = ancestors + (node,)

        for edge in child_edges:
            next_branch = new_branches[branch_index]
            branch_index += 1
            pip = next_branch.routeSegment.init("pip")
            pip.tile = get_string_index(str(edge["tile"]), string_to_index)
            pip.wire0 = get_string_index(str(edge["wire0"]), string_to_index)
            pip.wire1 = get_string_index(str(edge["wire1"]), string_to_index)
            pip.isFixed = False
            pip.forward = bool(edge["forward"])
            attachment = edge.get("attachment")
            if attachment is None:
                # Explicitly select the conventional-PIP union arm.
                pip.noSite = None
            else:
                # Sparse metadata validation guarantees a concrete site here.
                pip.site = get_string_index(str(edge["site"]), string_to_index)
            stack.append(
                (next_branch, route_int(edge["to"], "edge.to"), next_ancestors)
            )
            emitted_pips += 1

        for sink_key in sink_keys:
            orphans = sink_pin_orphans.get(sink_key)
            if not orphans:
                raise ValueError(
                    f"net {net_name} routed sink {sink_key} was not present in stubs"
                )
            orphan = orphans.pop(0)
            if not orphans:
                del sink_pin_orphans[sink_key]
            new_branches[branch_index] = orphan.get()
            branch_index += 1

    return emitted_pips


def resize_string_list(netlist, string_to_index: dict[str, int]) -> int:
    old_str_list = netlist.strList
    old_count = len(old_str_list)
    orphan_str_list = [old_str_list.disown(i) for i in range(old_count)]
    new_str_list = netlist.init("strList", len(string_to_index))

    for text, index in string_to_index.items():
        if index < old_count:
            new_str_list.adopt(index, orphan_str_list[index])
        else:
            new_str_list[index] = text
    return len(string_to_index) - old_count


def write_routed_physical_netlist(
    input_phys: Path,
    output_phys: Path,
    schema_dir: Path,
    routes_path: Path,
    allow_unrouted_stubs: bool,
    expected_artifact_pair_id: str | None = None,
    metadata_summary: RoutingMetadataSummary | None = None,
) -> None:
    schema = load_physical_schema(schema_dir)
    routes_by_net = read_routes_jsonl(
        routes_path, expected_artifact_pair_id, metadata_summary
    )
    data = read_gzip_or_plain(input_phys)

    with schema.PhysNetlist.from_bytes(
        data,
        traversal_limit_in_words=sys.maxsize,
        nesting_limit=2**16,
    ) as reader:
        netlist = reader.as_builder()

    str_list = netlist.strList
    string_to_index = {string_at(str_list, i): i for i in range(len(str_list))}
    num_pips = 0
    routed_net_count = 0

    for net in netlist.physNets:
        net_name = string_at(str_list, net.name)
        route = routes_by_net.pop(net_name, None)
        if route is None:
            continue

        adjacency, source_node_by_pin, sink_pins_by_node, route_edges = build_route_tables(route)
        sink_pin_orphans: dict[tuple[str, str], list[Any]] = defaultdict(list)
        unknown_stub_orphans: list[Any] = []

        for index, stub in enumerate(net.stubs):
            route_segment = stub.routeSegment
            if route_segment.which() != "sitePin":
                orphan = net.stubs.disown(index)
                unknown_stub_orphans.append(orphan)
                continue
            site_pin = route_segment.sitePin
            key = site_pin_key(
                string_at(str_list, site_pin.site),
                string_at(str_list, site_pin.pin),
            )
            orphan = net.stubs.disown(index)
            sink_pin_orphans[key].append(orphan)

        original_site_stub_count = sum(
            len(orphans) for orphans in sink_pin_orphans.values()
        )
        expected_reached_stub_count = sum(
            len(sink_keys) for sink_keys in sink_pins_by_node.values()
        )

        net.disown("stubs")

        emitted_edges = 0
        emitted_source_nodes: set[int] = set()
        source_branches = collect_site_pin_branches(net.sources, str_list)
        for source_key, source_branch in source_branches:
            source_node = source_node_by_pin.get(source_key)
            if source_node is None:
                continue
            if source_node not in adjacency and source_node not in sink_pins_by_node:
                continue
            if source_node in emitted_source_nodes:
                continue
            emitted_source_nodes.add(source_node)
            emitted_edges += insert_route_tree(
                source_branch,
                source_node,
                net_name,
                adjacency,
                sink_pins_by_node,
                sink_pin_orphans,
                string_to_index,
            )

        if emitted_edges != len(route_edges):
            raise ValueError(
                f"net {net_name} emitted {emitted_edges} PIPs but route has "
                f"{len(route_edges)} PIPs"
            )

        remaining_site_stub_count = sum(
            len(orphans) for orphans in sink_pin_orphans.values()
        )
        if (
            original_site_stub_count - remaining_site_stub_count
            != expected_reached_stub_count
        ):
            raise ValueError(
                f"net {net_name} has a reached sink that was not attached "
                "to a routed source"
            )

        remaining_stubs = [
            orphan
            for orphans in sink_pin_orphans.values()
            for orphan in orphans
        ] + unknown_stub_orphans
        if remaining_stubs and not allow_unrouted_stubs:
            raise ValueError(
                f"net {net_name} still has {len(remaining_stubs)} unrouted stubs"
            )
        if remaining_stubs:
            new_stubs = net.init("stubs", len(remaining_stubs))
            for index, orphan in enumerate(remaining_stubs):
                new_stubs[index] = orphan.get()

        num_pips += emitted_edges
        routed_net_count += 1

    if routes_by_net:
        missing = ", ".join(sorted(routes_by_net)[:5])
        raise ValueError(f"routed nets were not found in input .phys: {missing}")

    added_strings = resize_string_list(netlist, string_to_index)
    output_phys.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_phys, "wb", compresslevel=6) as out_file:
        out_file.write(netlist.to_bytes())
    print(
        "[pathfinder-benchmark] wrote routed .phys with "
        f"{num_pips} PIPs across {routed_net_count} nets "
        f"({added_strings} new strings)",
        flush=True,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_phys = args.input_phys.resolve()
    output_phys = args.output_phys.resolve()
    logical_netlist = (
        args.logical_netlist.resolve()
        if args.logical_netlist is not None
        else infer_logical_netlist(input_phys).resolve()
    )
    device_graph = args.device_graph.resolve()
    schema_dir = args.schema_dir.resolve() if args.schema_dir is not None else None

    for path, label in (
        (input_phys, "input physical netlist"),
        (logical_netlist, "logical netlist"),
        (device_graph, "device routing graph"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if schema_dir is None:
        raise FileNotFoundError(
            "missing FPGA Interchange schema directory; pass --schema-dir or "
            "set FPGA_INTERCHANGE_SCHEMA_DIR"
        )
    if not schema_dir.exists():
        raise FileNotFoundError(f"missing FPGA Interchange schema directory: {schema_dir}")

    output_phys.parent.mkdir(parents=True, exist_ok=True)
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else default_work_dir(output_phys).resolve()
    )

    temporary_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary_owner = tempfile.TemporaryDirectory(
            prefix=work_dir.name + ".", dir=str(work_dir.parent)
        )
        work_dir = Path(temporary_owner.name)

    try:
        csr_path = work_dir / (output_phys.stem + ".csrbin")
        metadata_path = work_dir / (output_phys.stem + ".csrbin.ifmeta.bin")
        routes_path = work_dir / (output_phys.stem + ".routes.jsonl")

        run_command(
            [
                args.interchange_to_csr,
                str(device_graph),
                str(input_phys),
                str(logical_netlist),
                str(csr_path),
                "--metadata",
                str(metadata_path),
            ],
            "convert FPGAIF to CSR",
        )

        run_command(
            [
                args.pathfinder,
                str(csr_path),
                str(metadata_path),
                "--routes-out",
                str(routes_path),
                *pathfinder_args(args),
            ],
            "run CSR PathFinder",
        )

        metadata_summary = read_metadata_summary(metadata_path)
        write_routed_physical_netlist(
            input_phys,
            output_phys,
            schema_dir,
            routes_path,
            args.allow_unrouted_stubs,
            metadata_summary.artifact_pair_id,
            metadata_summary,
        )
        return 0
    finally:
        if temporary_owner is not None:
            temporary_owner.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"[pathfinder-benchmark] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
