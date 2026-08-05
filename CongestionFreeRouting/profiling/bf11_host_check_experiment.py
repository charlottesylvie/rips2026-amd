#!/usr/bin/env python3
"""Run and summarize the BF11 host-check K experiment on the gfx1151 host.

The default invocation performs one warm-up and five measured runs for
K=1,2,4,8, validates 1,000 completed queries, writes one route artifact per K,
requires equivalent valid route trees (while reporting byte identity), and
emits JSON/Markdown summaries. Pass --profile to collect and compare
rocprofv3 runtime traces for K=1 and the fastest non-1 K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

try:
    import resource
except ImportError:  # pragma: no cover - the experiment itself runs on Linux.
    resource = None  # type: ignore[assignment]


DEFAULT_GRAPH = Path(
    "d181-profile-work/mlcad_d181_lefttwo3rds_PathFinderFile.csrbin"
)
DEFAULT_KS = (1, 2, 4, 8)
NANOSECONDS_PER_SECOND = 1_000_000_000
PROFILE_SOURCES = (
    "CongestionFreeRouting/pathfinder.cpp",
    "CongestionFreeRouting/bellman_ford/bf10.cpp",
    "CongestionFreeRouting/bellman_ford/bf11.cpp",
    "CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp",
    "CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp",
)
PROFILE_INCLUDE_DIRS = (
    "HIP_kernel/bellman_ford/src",
    "CongestionFreeRouting/bellman_ford",
    "CongestionFreeRouting/delta_stepping",
    "CongestionFreeRouting/unit_bfs",
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def host_check_k(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 64:
        raise argparse.ArgumentTypeError("K must be between 1 and 64")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathfinder", type=Path, default=Path("./pathfinder"))
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="defaults to <graph>.ifmeta.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bf11-host-check-results"),
    )
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--ks", nargs="+", type=host_check_k, default=DEFAULT_KS)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="reuse measured logs already present in --output-dir",
    )
    parser.add_argument(
        "--no-route-check",
        action="store_true",
        help="skip the separate deterministic route-equivalence runs",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="trace K=1 and the fastest K>1 after unprofiled measurements",
    )
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--hipcc", default="hipcc")
    parser.add_argument(
        "--offload-arch",
        default="gfx1151",
        help="HIP target used by --build and the dedicated ROCTx profile build",
    )
    parser.add_argument(
        "--profile-pathfinder",
        type=Path,
        help=(
            "prebuilt PATHFINDER_ENABLE_ROCTX executable for trace runs; "
            "--build creates a dedicated one automatically"
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="run the repository's source-based pathfinder build first",
    )
    args = parser.parse_args(argv)
    if args.repetitions < 5:
        parser.error("--repetitions must be at least 5")
    if 1 not in args.ks:
        parser.error("--ks must include the K=1 control")
    if not any(k > 1 for k in args.ks):
        parser.error("--ks must include at least one experimental K")
    args.ks = tuple(dict.fromkeys(args.ks))
    if args.metadata is None:
        args.metadata = Path(str(args.graph) + ".ifmeta.bin")
    return args


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing or empty: {resolved}")
    return resolved


def common_command(
    args: argparse.Namespace, executable: Path | None = None
) -> list[str]:
    return [
        str(
            require_file(
                args.pathfinder if executable is None else executable,
                "pathfinder executable",
            )
        ),
        str(require_file(args.graph, "CSR graph")),
        str(require_file(args.metadata, "CSR metadata")),
        "--allow-unrouted",
        "--sssp-engine",
        "bf11",
        "--bf11-bbox-margin-x",
        "2",
        "--bf11-bbox-margin-y",
        "14",
        "--net-limit",
        "1000",
        "--parallel-net-workers",
        "4",
        "--bf11-telemetry",
    ]


def command_for_k(
    args: argparse.Namespace,
    k: int,
    routes_out: Path | None = None,
    executable: Path | None = None,
) -> list[str]:
    command = [
        *common_command(args, executable),
        "--bf11-iterations-per-host-check",
        str(k),
    ]
    if routes_out is not None:
        command.extend(["--routes-out", str(routes_out.resolve())])
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}; see {log_path}"
        )


def json_records(log_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as log:
        for line in log:
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            record_type = record.get("type")
            if isinstance(record_type, str):
                records[record_type] = record
    return records


def validate_records(
    log_path: Path, expected_k: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = json_records(log_path)
    try:
        runtime = records["bf11_runtime_stats"]
        telemetry = records["bf11_telemetry"]
        routing_time = records["bf11_routing_time"]
    except KeyError as exc:
        raise RuntimeError(
            f"{log_path} is missing required BF11 JSON record {exc}"
        ) from exc
    if telemetry.get("completed_queries") != 1000:
        raise RuntimeError(f"{log_path} did not complete 1,000 BF11 queries")
    if runtime.get("effective_workers") != 4 or telemetry.get(
        "effective_workers"
    ) != 4:
        raise RuntimeError(f"{log_path} did not exercise four BF11 workers")
    if runtime.get("iterations_per_host_check") != expected_k:
        raise RuntimeError(f"{log_path} reported the wrong K")
    control = telemetry.get("iteration_control", {})
    if control.get("iterations_per_host_check") != expected_k:
        raise RuntimeError(f"{log_path} telemetry reported the wrong K")
    if control.get("iteration_status_copies") != control.get("host_check_rounds"):
        raise RuntimeError(f"{log_path} status-copy count disagrees with host checks")
    if control.get("stream_synchronizations_for_iteration_control") != control.get(
        "host_check_rounds"
    ):
        raise RuntimeError(f"{log_path} control synchronization count is inconsistent")
    if control.get("device_iterations_enqueued") != control.get(
        "device_iterations_executed"
    ) + control.get("terminal_noop_iterations"):
        raise RuntimeError(f"{log_path} device-iteration accounting is inconsistent")
    return runtime, telemetry, routing_time


def measured_logs(output_dir: Path, k: int, repetitions: int) -> list[Path]:
    return [output_dir / f"k{k}-r{rep}.log" for rep in range(1, repetitions + 1)]


def run_benchmarks(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k in args.ks:
        run_logged(
            command_for_k(args, k),
            args.output_dir / f"k{k}-warmup.log",
        )
    for repetition in range(1, args.repetitions + 1):
        order = args.ks if repetition % 2 else tuple(reversed(args.ks))
        for k in order:
            run_logged(
                command_for_k(args, k),
                args.output_dir / f"k{k}-r{repetition}.log",
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_records(path: Path, expected_count: int = 1000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number} is not valid route JSON"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    if len(records) != expected_count:
        raise RuntimeError(
            f"{path} contains {len(records)} route records; expected {expected_count}"
        )
    return records


def require_integer_node(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} is not an integer node ID")
    return value


def validate_route_record(record: dict[str, Any], context: str) -> tuple[Any, ...]:
    routed = record.get("routed")
    sources = record.get("sources")
    sinks = record.get("sinks")
    edges = record.get("edges")
    if not isinstance(routed, bool):
        raise RuntimeError(f"{context} has a non-boolean routed flag")
    if not isinstance(sources, list) or not isinstance(sinks, list) or not isinstance(
        edges, list
    ):
        raise RuntimeError(f"{context} is missing source, sink, or edge arrays")

    source_nodes: list[int] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise RuntimeError(f"{context} source {index} is not an object")
        source_nodes.append(
            require_integer_node(source.get("node"), f"{context} source {index}")
        )
    if not source_nodes:
        raise RuntimeError(f"{context} has no declared sources")

    adjacency: dict[int, set[int]] = defaultdict(set)
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise RuntimeError(f"{context} edge {index} is not an object")
        source = require_integer_node(edge.get("from"), f"{context} edge {index} from")
        target = require_integer_node(edge.get("to"), f"{context} edge {index} to")
        adjacency[source].add(target)

    reachable = set(source_nodes)
    pending = deque(source_nodes)
    while pending:
        node = pending.popleft()
        for target in adjacency.get(node, ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)

    sink_signature: list[tuple[int, bool]] = []
    all_reached = True
    for index, sink in enumerate(sinks):
        if not isinstance(sink, dict):
            raise RuntimeError(f"{context} sink {index} is not an object")
        node = require_integer_node(sink.get("node"), f"{context} sink {index}")
        reached = sink.get("reached")
        if not isinstance(reached, bool):
            raise RuntimeError(f"{context} sink {index} has a non-boolean reached flag")
        sink_signature.append((node, reached))
        all_reached = all_reached and reached
        if reached:
            root = require_integer_node(
                sink.get("source"), f"{context} sink {index} source"
            )
            if root not in reachable:
                raise RuntimeError(
                    f"{context} reached sink {node} names disconnected tree source {root}"
                )
            root_reachable = {root}
            root_pending = deque([root])
            while root_pending:
                tree_node = root_pending.popleft()
                for target in adjacency.get(tree_node, ()):
                    if target not in root_reachable:
                        root_reachable.add(target)
                        root_pending.append(target)
            if node not in root_reachable:
                raise RuntimeError(
                    f"{context} reached sink {node} is disconnected from named tree source {root}"
                )
    if routed != all_reached:
        raise RuntimeError(f"{context} routed flag disagrees with its sink results")
    return (
        record.get("net"),
        routed,
        tuple(source_nodes),
        tuple(sink_signature),
    )


def validate_route_equivalence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    candidate_label: str,
) -> None:
    if len(candidate) != len(baseline):
        raise RuntimeError(
            f"{candidate_label} has {len(candidate)} nets; K=1 has {len(baseline)}"
        )
    for index, (baseline_record, candidate_record) in enumerate(
        zip(baseline, candidate)
    ):
        baseline_signature = validate_route_record(
            baseline_record, f"K=1 route record {index}"
        )
        candidate_signature = validate_route_record(
            candidate_record, f"{candidate_label} route record {index}"
        )
        if candidate_signature != baseline_signature:
            raise RuntimeError(
                f"{candidate_label} net {index} routed/sink signature differs from K=1"
            )


def validate_routes(args: argparse.Namespace) -> dict[int, str]:
    hashes: dict[int, str] = {}
    records: dict[int, list[dict[str, Any]]] = {}
    for k in args.ks:
        route_path = args.output_dir / f"routes-k{k}.jsonl"
        run_logged(
            command_for_k(args, k, route_path),
            args.output_dir / f"routes-k{k}.log",
        )
        validate_records(args.output_dir / f"routes-k{k}.log", k)
        require_file(route_path, f"K={k} routes")
        hashes[k] = sha256(route_path)
        records[k] = route_records(route_path)
    validate_route_equivalence(records[1], records[1], "K=1")
    for k in args.ks:
        if k != 1:
            validate_route_equivalence(records[1], records[k], f"K={k}")
    return hashes


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    median = statistics.median(samples)
    return {
        "median": median,
        "minimum": min(samples),
        "maximum": max(samples),
        "variation": 0.0 if median == 0 else (max(samples) - min(samples)) / median,
    }


def median_int(values: Iterable[int]) -> int:
    return int(statistics.median(list(values)))


def summarize(args: argparse.Namespace, route_hashes: dict[int, str]) -> dict[str, Any]:
    distinct_route_hashes = set(route_hashes.values())
    summary: dict[str, Any] = {
        "repetitions": args.repetitions,
        "route_sha256": {str(k): digest for k, digest in route_hashes.items()},
        "routes_topology_validated": bool(route_hashes),
        "routes_byte_identical": (
            len(distinct_route_hashes) == 1 if route_hashes else None
        ),
        "results": {},
    }
    for k in args.ks:
        runtimes: list[float] = []
        routing_times: list[float] = []
        runtime_records: list[dict[str, Any]] = []
        telemetry_records: list[dict[str, Any]] = []
        for log_path in measured_logs(args.output_dir, k, args.repetitions):
            runtime, telemetry, routing_time = validate_records(log_path, k)
            runtime_records.append(runtime)
            telemetry_records.append(telemetry)
            runtimes.append(float(runtime["routing_seconds"]))
            routing_times.append(float(routing_time["routing_seconds"]))
        controls = [record["iteration_control"] for record in telemetry_records]
        work = [record["work"] for record in telemetry_records]
        result = {
            "runtime_stats_routing_seconds": numeric_summary(runtimes),
            "bf11_routing_time_seconds": numeric_summary(routing_times),
            "completed_queries": 1000,
            "effective_workers": 4,
            "target_checks": median_int(r["target_checks"] for r in runtime_records),
            "auto_unbounded_retries": median_int(
                r["auto_unbounded_retries"] for r in runtime_records
            ),
        }
        for field in (
            "host_check_rounds",
            "iteration_status_copies",
            "stream_synchronizations_for_iteration_control",
            "device_iterations_enqueued",
            "device_iterations_executed",
            "terminal_noop_iterations",
            "speculative_iterations",
        ):
            result[field] = median_int(record[field] for record in controls)
        for field in (
            "iterations",
            "frontier_vertices_processed",
            "edges_examined",
            "successful_relaxations",
            "touched_vertices",
        ):
            result[field] = median_int(record[field] for record in work)
        summary["results"][str(k)] = result
    experimental = [k for k in args.ks if k > 1]
    best_k = min(
        experimental,
        key=lambda k: summary["results"][str(k)]["bf11_routing_time_seconds"][
            "median"
        ],
    )
    summary["best_experimental_k"] = best_k
    baseline = summary["results"]["1"]["bf11_routing_time_seconds"]["median"]
    winner = summary["results"][str(best_k)]["bf11_routing_time_seconds"][
        "median"
    ]
    summary["best_speedup_fraction"] = (baseline - winner) / baseline
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown_summary(args.output_dir / "summary.md", summary)
    return summary


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BF11 host-check experiment",
        "",
        "| K | Overall median (s) | Min | Max | Variation | Internal median (s) | Host checks | Status copies | Control syncs | Enqueued | Executed | Terminal no-ops | Edges | Relaxations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in summary["results"].items():
        overall = result["bf11_routing_time_seconds"]
        internal = result["runtime_stats_routing_seconds"]
        lines.append(
            f"| {key} | {overall['median']:.6f} | {overall['minimum']:.6f} | "
            f"{overall['maximum']:.6f} | {overall['variation']:.2%} | "
            f"{internal['median']:.6f} | {result['host_check_rounds']} | "
            f"{result['iteration_status_copies']} | "
            f"{result['stream_synchronizations_for_iteration_control']} | "
            f"{result['device_iterations_enqueued']} | "
            f"{result['device_iterations_executed']} | "
            f"{result['terminal_noop_iterations']} | {result['edges_examined']} | "
            f"{result['successful_relaxations']} |"
        )
    lines.extend(
        [
            "",
            f"Fastest experimental K: **{summary['best_experimental_k']}**",
            f"Overall median speedup vs K=1: **{summary['best_speedup_fraction']:.2%}**",
            (
                "Route outputs: **topologically equivalent and byte-identical**"
                if summary["routes_byte_identical"] is True
                else "Route outputs: **topologically equivalent; byte representations differ**"
                if summary["routes_topology_validated"]
                else "Route outputs: **not checked in this invocation**"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def seconds(nanoseconds: int | float) -> float:
    return float(nanoseconds) / NANOSECONDS_PER_SECOND


def percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(
        0,
        min(
            len(sorted_values) - 1,
            math.ceil(quantile * len(sorted_values)) - 1,
        ),
    )
    return float(sorted_values[index])


def duration_distribution(durations: Iterable[int]) -> dict[str, Any]:
    values = sorted(int(value) for value in durations)
    under_50 = sum(value < 50_000 for value in values)
    return {
        "calls": len(values),
        "aggregate_seconds": seconds(sum(values)),
        "minimum_microseconds": (values[0] / 1_000.0 if values else 0.0),
        "p50_microseconds": percentile(values, 0.50) / 1_000.0,
        "p90_microseconds": percentile(values, 0.90) / 1_000.0,
        "p95_microseconds": percentile(values, 0.95) / 1_000.0,
        "p99_microseconds": percentile(values, 0.99) / 1_000.0,
        "maximum_microseconds": (values[-1] / 1_000.0 if values else 0.0),
        "under_50_microseconds_calls": under_50,
        "under_50_microseconds_percent": (
            100.0 * under_50 / len(values) if values else 0.0
        ),
    }


def dispatch_union_and_gaps(
    intervals: Iterable[tuple[int, int]],
) -> tuple[int, list[int], int]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0, [], 0
    union_nanoseconds = 0
    gaps: list[int] = []
    islands = 1
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            union_nanoseconds += current_end - current_start
            gaps.append(start - current_end)
            islands += 1
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    union_nanoseconds += current_end - current_start
    return union_nanoseconds, gaps, islands


def trace_table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }


def marker_spans(
    connection: sqlite3.Connection,
) -> dict[str, tuple[int, int, int]]:
    spans: dict[str, tuple[int, int, int]] = {}
    for row in connection.execute(
        "SELECT start, end, extdata FROM regions "
        "WHERE category LIKE 'MARKER%RANGE%' ORDER BY start"
    ):
        try:
            payload = json.loads(row[2] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        marker = payload.get("message")
        if not isinstance(marker, str):
            continue
        start = int(row[0])
        end = int(row[1])
        if marker in spans:
            first, last, calls = spans[marker]
            spans[marker] = (min(first, start), max(last, end), calls + 1)
        else:
            spans[marker] = (start, end, 1)
    return spans


def analyze_bf11_rocpd(database: Path) -> dict[str, Any]:
    resolved = require_file(database, "RocPD database")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        required = {"regions", "kernels"}
        missing = sorted(required - trace_table_names(connection))
        if missing:
            raise RuntimeError(
                f"{resolved} is missing required RocPD views: {', '.join(missing)}"
            )
        spans = marker_spans(connection)
        if "pathfinder.run" not in spans:
            raise RuntimeError(
                f"{resolved} has no pathfinder.run marker; use the runner's "
                "ROCTx profile build or pass --profile-pathfinder"
            )
        if "bf11.run" not in spans:
            raise RuntimeError(
                f"{resolved} has no bf11.run markers; the trace cannot be "
                "scoped to BF11 routing"
            )
        scope_start, scope_end, pathfinder_markers = spans["pathfinder.run"]
        query_start, query_end, query_markers = spans["bf11.run"]
        if scope_end <= scope_start:
            raise RuntimeError(f"{resolved} has an empty BF11 routing span")

        kernel_rows = connection.execute(
            "SELECT name, start, end, duration FROM kernels "
            "WHERE end > ? AND start < ? ORDER BY start, end",
            (scope_start, scope_end),
        ).fetchall()
        if not kernel_rows:
            raise RuntimeError(f"{resolved} has no GPU dispatches in BF11 routing")
        kernel_intervals = [
            (max(scope_start, int(row[1])), min(scope_end, int(row[2])))
            for row in kernel_rows
        ]
        union_ns, gaps, islands = dispatch_union_and_gaps(kernel_intervals)
        aggregate_kernel_ns = sum(int(row[3]) for row in kernel_rows)
        relaxation_durations = [
            int(row[3])
            for row in kernel_rows
            if "frontier_relax_kernel" in str(row[0])
        ]
        runtime_copy_durations = [
            int(row[3])
            for row in kernel_rows
            if "__amd_rocclr_copyBuffer" in str(row[0])
        ]
        top_kernel_names = []
        by_kernel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in kernel_rows:
            values = by_kernel[str(row[0])]
            values[0] += 1
            values[1] += int(row[3])
        for name, (calls, duration) in sorted(
            by_kernel.items(), key=lambda item: item[1][1], reverse=True
        )[:20]:
            top_kernel_names.append(
                {
                    "name": name,
                    "calls": calls,
                    "aggregate_seconds": seconds(duration),
                }
            )

        api_by_name: dict[str, dict[str, float | int]] = {}
        for row in connection.execute(
            "SELECT name, COUNT(*) AS calls, SUM(duration) AS duration "
            "FROM regions WHERE category LIKE 'HIP_RUNTIME_API%' "
            "AND end > ? AND start < ? GROUP BY name",
            (scope_start, scope_end),
        ):
            api_by_name[str(row[0])] = {
                "calls": int(row[1]),
                "aggregate_seconds": seconds(int(row[2])),
            }

        scope_ns = scope_end - scope_start
        relaxation_ns = sum(relaxation_durations)
        runtime_copy_ns = sum(runtime_copy_durations)
        gap_distribution = duration_distribution(gaps)
        gap_distribution.pop("under_50_microseconds_calls")
        gap_distribution.pop("under_50_microseconds_percent")
        return {
            "source_database": str(resolved),
            "scope": {
                "marker": "pathfinder.run",
                "marker_calls": pathfinder_markers,
                "start_nanoseconds": scope_start,
                "end_nanoseconds": scope_end,
                "span_seconds": seconds(scope_ns),
            },
            "bf11_queries": {
                "marker": "bf11.run",
                "calls": query_markers,
                "first_start_nanoseconds": query_start,
                "last_end_nanoseconds": query_end,
                "span_seconds": seconds(query_end - query_start),
            },
            "kernel_dispatches": len(kernel_rows),
            "aggregate_kernel_seconds": seconds(aggregate_kernel_ns),
            "union_gpu_active_seconds": seconds(union_ns),
            "gpu_active_fraction": union_ns / scope_ns,
            "gpu_active_percent": 100.0 * union_ns / scope_ns,
            "dispatch_islands": islands,
            "hip_api": {
                "hipMemcpyAsync_calls": int(
                    api_by_name.get("hipMemcpyAsync", {}).get("calls", 0)
                ),
                "hipStreamSynchronize_calls": int(
                    api_by_name.get("hipStreamSynchronize", {}).get("calls", 0)
                ),
                "by_name": api_by_name,
            },
            "relaxation": {
                **duration_distribution(relaxation_durations),
                "aggregate_share_of_gpu_percent": (
                    100.0 * relaxation_ns / aggregate_kernel_ns
                    if aggregate_kernel_ns
                    else 0.0
                ),
            },
            "runtime_copy_kernels": {
                **duration_distribution(runtime_copy_durations),
                "aggregate_share_of_gpu_percent": (
                    100.0 * runtime_copy_ns / aggregate_kernel_ns
                    if aggregate_kernel_ns
                    else 0.0
                ),
            },
            "dispatch_gaps": gap_distribution,
            "top_kernels": top_kernel_names,
        }
    finally:
        connection.close()


def artifact_state(output: Path) -> dict[Path, tuple[int, int]]:
    if not output.exists():
        return {}
    return {
        path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }


def require_profile_artifacts(
    output: Path, before: dict[Path, tuple[int, int]] | None = None
) -> dict[str, Any]:
    candidates = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    ]
    if before is not None:
        candidates = [
            path
            for path in candidates
            if before.get(path.resolve())
            != (path.stat().st_size, path.stat().st_mtime_ns)
        ]
    csv_files = sorted(path for path in candidates if path.suffix.lower() == ".csv")
    pftrace_files = sorted(
        path for path in candidates if path.suffix.lower() == ".pftrace"
    )
    databases: list[Path] = []
    for path in candidates:
        try:
            with path.open("rb") as stream:
                if stream.read(16) == b"SQLite format 3\x00":
                    databases.append(path)
        except OSError:
            continue
    if not databases:
        raise RuntimeError(
            f"{output} is missing a new, nonempty RocPD SQLite database"
        )
    missing_formats = []
    if not csv_files:
        missing_formats.append("CSV")
    if not pftrace_files:
        missing_formats.append("PFTrace")
    return {
        "csv": [str(path.resolve()) for path in csv_files],
        "pftrace": [str(path.resolve()) for path in pftrace_files],
        "database": str(sorted(databases)[0].resolve()),
        "missing_requested_formats": missing_formats,
    }


def disable_core_dumps() -> None:
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def trace_metric_rows(
    baseline: dict[str, Any], winner: dict[str, Any]
) -> list[dict[str, Any]]:
    selectors = (
        ("Overall routing time (s)", lambda value: value["routing_seconds"]),
        (
            "Internal BF11 routing time (s)",
            lambda value: value["internal_routing_seconds"],
        ),
        ("Kernel dispatches", lambda value: value["kernel_dispatches"]),
        (
            "hipMemcpyAsync calls",
            lambda value: value["hip_api"]["hipMemcpyAsync_calls"],
        ),
        (
            "hipStreamSynchronize calls",
            lambda value: value["hip_api"]["hipStreamSynchronize_calls"],
        ),
        (
            "Aggregate GPU kernel time (s)",
            lambda value: value["aggregate_kernel_seconds"],
        ),
        (
            "Union GPU-active time (s)",
            lambda value: value["union_gpu_active_seconds"],
        ),
        ("GPU-active fraction", lambda value: value["gpu_active_fraction"]),
        (
            "Relaxation GPU time (s)",
            lambda value: value["relaxation"]["aggregate_seconds"],
        ),
        (
            "Relaxation share of GPU time (%)",
            lambda value: value["relaxation"][
                "aggregate_share_of_gpu_percent"
            ],
        ),
        (
            "Relaxation kernels under 50 us (%)",
            lambda value: value["relaxation"][
                "under_50_microseconds_percent"
            ],
        ),
        (
            "Runtime-copy kernel time (s)",
            lambda value: value["runtime_copy_kernels"]["aggregate_seconds"],
        ),
        (
            "Runtime-copy share of GPU time (%)",
            lambda value: value["runtime_copy_kernels"][
                "aggregate_share_of_gpu_percent"
            ],
        ),
        (
            "Inter-dispatch gap total (s)",
            lambda value: value["dispatch_gaps"]["aggregate_seconds"],
        ),
        (
            "Inter-dispatch gap p95 (us)",
            lambda value: value["dispatch_gaps"]["p95_microseconds"],
        ),
        (
            "Inter-dispatch gap maximum (us)",
            lambda value: value["dispatch_gaps"]["maximum_microseconds"],
        ),
    )
    rows = []
    for label, select in selectors:
        control = float(select(baseline))
        experimental = float(select(winner))
        rows.append(
            {
                "metric": label,
                "k1": control,
                "winner": experimental,
                "change": experimental - control,
                "change_percent": (
                    100.0 * (experimental - control) / control
                    if control != 0.0
                    else None
                ),
            }
        )
    return rows


def write_profile_comparison(
    path: Path, comparison: dict[str, Any], benchmark_summary: dict[str, Any]
) -> None:
    lines = [
        "# BF11 runtime-trace comparison",
        "",
        (
            "Scope: the `pathfinder.run` ROCTx range, with `bf11.run` query "
            "counts reported separately. GPU intervals are unioned before active "
            "fractions are computed; aggregate kernel time may overlap across streams."
        ),
        "",
        f"| Metric | K=1 | K={comparison['winner_k']} | Change vs K=1 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["metrics"]:
        change = (
            "n/a"
            if row["change_percent"] is None
            else f"{row['change_percent']:+.2f}%"
        )
        lines.append(
            f"| {row['metric']} | {row['k1']:.6g} | "
            f"{row['winner']:.6g} | {change} |"
        )

    speedup = float(benchmark_summary["best_speedup_fraction"])
    winner_key = str(comparison["winner_k"])
    baseline_work = benchmark_summary["results"]["1"]
    winner_work = benchmark_summary["results"][winner_key]
    copies_reduced = (
        winner_work["iteration_status_copies"]
        < baseline_work["iteration_status_copies"]
        and winner_work["stream_synchronizations_for_iteration_control"]
        < baseline_work["stream_synchronizations_for_iteration_control"]
    )
    baseline_overall = baseline_work["bf11_routing_time_seconds"]
    winner_overall = winner_work["bf11_routing_time_seconds"]
    repeatable = (
        winner_overall["maximum"] < baseline_overall["median"]
        and winner_overall["variation"] <= 0.10
    )
    hypothesis_validated = (
        benchmark_summary["routes_topology_validated"]
        and speedup >= 0.10
        and copies_reduced
        and repeatable
    )
    experimental_default = comparison["winner_k"] if hypothesis_validated else 1
    edge_growth = (
        winner_work["edges_examined"] / baseline_work["edges_examined"] - 1.0
        if baseline_work["edges_examined"]
        else 0.0
    )
    winner_trace = comparison["traces"][winner_key]
    scheduling_evidence = (
        hypothesis_validated
        and edge_growth <= 0.15
        and winner_trace["gpu_active_fraction"] < 0.70
        and winner_trace["dispatch_gaps"]["aggregate_seconds"] > 0.0
    )
    next_step = (
        "kernel fusion/persistent scheduling"
        if scheduling_evidence
        else (
            "evidence is insufficient to choose memory-layout optimization; "
            "collect the optional memory/L2 counters first"
        )
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "The host-check hypothesis was **validated**."
                if hypothesis_validated
                else "The host-check hypothesis was **not validated** by these runs."
            ),
            f"Experimental default recommendation: **K={experimental_default}**.",
            f"Recommended next step: **{next_step}**.",
            (
                "The production default remains K=1; changing it still requires "
                "review of these saved correctness, timing, and trace artifacts."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def profile(
    args: argparse.Namespace,
    best_k: int,
    benchmark_summary: dict[str, Any],
    profile_executable: Path,
) -> dict[str, Any]:
    traces: dict[int, dict[str, Any]] = {}
    for k in (1, best_k):
        output = args.output_dir / f"profile-k{k}"
        output.mkdir(parents=True, exist_ok=True)
        before = artifact_state(output)
        command = [
            args.rocprofv3,
            "--runtime-trace",
            "--stats",
            "--output-directory",
            str(output.resolve()),
            "--output-file",
            f"bf11-k{k}",
            "--output-format",
            "csv",
            "pftrace",
            "rocpd",
            "--",
            *command_for_k(args, k, executable=profile_executable),
        ]
        log_path = output / "pathfinder.log"
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                preexec_fn=disable_core_dumps if os.name == "posix" else None,
            )
        runtime, _telemetry, routing_time = validate_records(log_path, k)
        accepted_statuses = (0, 139, -signal.SIGSEGV)
        if completed.returncode not in accepted_statuses:
            raise RuntimeError(
                f"rocprofv3 K={k} failed with status {completed.returncode}; "
                f"see {log_path}"
            )
        artifacts = require_profile_artifacts(output, before)
        if artifacts["missing_requested_formats"]:
            print(
                f"warning: rocprofv3 K={k} completed without fresh "
                + "/".join(artifacts["missing_requested_formats"])
                + "; using the nonempty RocPD database",
                file=sys.stderr,
            )
        trace = analyze_bf11_rocpd(Path(artifacts["database"]))
        trace.update(
            {
                "iterations_per_host_check": k,
                "routing_seconds": float(routing_time["routing_seconds"]),
                "internal_routing_seconds": float(runtime["routing_seconds"]),
                "profiler_returncode": completed.returncode,
                "artifacts": artifacts,
            }
        )
        traces[k] = trace
        (output / "trace-summary.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    comparison = {
        "baseline_k": 1,
        "winner_k": best_k,
        "scope": "pathfinder.run ROCTx range",
        "traces": {str(k): value for k, value in traces.items()},
        "metrics": trace_metric_rows(traces[1], traces[best_k]),
    }
    (args.output_dir / "profile-comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_profile_comparison(
        args.output_dir / "profile-comparison.md", comparison, benchmark_summary
    )
    return comparison


def build_pathfinder(args: argparse.Namespace) -> None:
    flags = f"-std=c++17 -O3 -x hip --offload-arch={args.offload_arch}"
    subprocess.run(
        [
            "make",
            "-B",
            "PATHFINDER_BUILD_COMPONENTS=1",
            f"PATHFINDER_HIP_FLAGS={flags}",
            "./pathfinder",
        ],
        check=True,
    )


def build_profile_pathfinder(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (args.output_dir / "pathfinder-roctx").resolve()
    command = [
        args.hipcc,
        "-std=c++17",
        "-O3",
        "-x",
        "hip",
        f"--offload-arch={args.offload_arch}",
        "-DBF10_NO_MAIN",
        "-DBF11_NO_MAIN",
        "-DPATHFINDER_ENABLE_ROCTX",
    ]
    for include in PROFILE_INCLUDE_DIRS:
        command.extend(["-I", include])
    command.extend(PROFILE_SOURCES)
    command.extend(
        ["-pthread", "-lrocprofiler-sdk-roctx", "-o", str(output)]
    )
    subprocess.run(command, check=True)
    return require_file(output, "ROCTx profile pathfinder")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.build:
        build_pathfinder(args)
    if not args.summarize_only:
        run_benchmarks(args)
    route_hashes: dict[int, str] = {}
    if not args.no_route_check and not args.summarize_only:
        route_hashes = validate_routes(args)
    summary = summarize(args, route_hashes)
    if args.profile:
        if args.build:
            profile_executable = build_profile_pathfinder(args)
        elif args.profile_pathfinder is not None:
            profile_executable = require_file(
                args.profile_pathfinder, "ROCTx profile pathfinder"
            )
        else:
            profile_executable = require_file(
                args.pathfinder, "ROCTx-capable profile pathfinder"
            )
        profile(
            args,
            int(summary["best_experimental_k"]),
            summary,
            profile_executable,
        )
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
