#!/usr/bin/env python3
"""Summarize BF10 time, kernels, synchronization, and transfers in a RocPD trace."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NS_PER_SECOND = 1_000_000_000
BF10_MARKERS = (
    "pathfinder.run",
    "pathfinder.route_net",
    "pathfinder.sssp",
    "bf10.upload_graph",
    "bf10.run",
    "bf10.gpu_controller",
    "bf10.controller_fallback",
    "bf10.gather_targets",
    "bf10.gpu_reconstruct",
    "bf10.copy_compact_paths",
    "bf10.reconstruction_fallback",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="rocprofv3 RocPD SQLite database")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="report directory (default: <database-stem>_bf10_analysis)",
    )
    parser.add_argument("--top", type=int, default=15, help="rows in top-kernel/API tables")
    return parser.parse_args()


def seconds(value: int | float | None) -> float:
    return float(value or 0) / NS_PER_SECOND


def percentile(values: Iterable[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def kernel_category(name: str) -> str:
    if "outgoing_frontier_controller_kernel" in name:
        return "BF cooperative controller"
    if "outgoing_frontier_relax_kernel" in name:
        return "BF host-fallback relaxation"
    if "init_outgoing_sssp_kernel" in name:
        return "BF host-fallback initialization"
    if "reset_iteration_status_kernel" in name:
        return "BF host-fallback status reset"
    if "update_target_status_kernel" in name:
        return "BF host-fallback target check"
    if "gather_packed_states_kernel" in name:
        return "Target-state gather"
    if "count_reconstruction_paths_kernel" in name:
        return "Reconstruction count"
    if "scan_reconstruction_offsets_kernel" in name:
        return "Reconstruction offset scan"
    if "materialize_reconstruction_paths_kernel" in name:
        return "Reconstruction materialization"
    if name == "__amd_rocclr_copyBuffer":
        return "Runtime copy kernels"
    if name == "__amd_rocclr_fillBufferAligned":
        return "Runtime fill kernels"
    return "Other kernels"


def fetch_one(connection: sqlite3.Connection, query: str) -> sqlite3.Row:
    row = connection.execute(query).fetchone()
    if row is None:
        raise RuntimeError("RocPD query returned no rows")
    return row


def analyze(database: Path, top: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    required = {"processes", "regions", "region_args", "kernels", "memory_copies"}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"database is missing RocPD views: {', '.join(missing)}")

    process = fetch_one(
        connection,
        "SELECT pid, start, end, command FROM processes ORDER BY start LIMIT 1",
    )
    process_wall_ns = int(process["end"]) - int(process["start"])

    marker_placeholders = ",".join("?" for _ in BF10_MARKERS)
    marker_rows = connection.execute(
        f"""
        SELECT json_extract(extdata, '$.message') AS marker,
               COUNT(*) AS calls,
               SUM(duration) AS duration,
               AVG(duration) AS average_duration,
               MIN(duration) AS minimum_duration,
               MAX(duration) AS maximum_duration
        FROM regions
        WHERE category = 'MARKER_CORE_RANGE_API'
          AND json_extract(extdata, '$.message') IN ({marker_placeholders})
        GROUP BY marker
        ORDER BY duration DESC
        """,
        BF10_MARKERS,
    ).fetchall()
    markers = [
        {
            "name": str(row["marker"]),
            "calls": int(row["calls"]),
            "aggregate_seconds": seconds(row["duration"]),
            "average_milliseconds": float(row["average_duration"] or 0) / 1_000_000,
            "pseudoshare_of_process_wall_percent": (
                100.0 * int(row["duration"] or 0) / process_wall_ns
                if process_wall_ns
                else 0.0
            ),
            "minimum_milliseconds": float(row["minimum_duration"] or 0) / 1_000_000,
            "maximum_milliseconds": float(row["maximum_duration"] or 0) / 1_000_000,
        }
        for row in marker_rows
    ]
    marker_by_name = {item["name"]: item for item in markers}

    kernel_rows = connection.execute(
        """
        SELECT name, COUNT(*) AS calls, SUM(duration) AS duration,
               AVG(duration) AS average_duration, MAX(duration) AS maximum_duration
        FROM kernels
        GROUP BY name
        ORDER BY duration DESC
        """
    ).fetchall()
    aggregate_kernel_ns = sum(int(row["duration"] or 0) for row in kernel_rows)
    top_kernels = [
        {
            "name": str(row["name"]),
            "category": kernel_category(str(row["name"])),
            "calls": int(row["calls"]),
            "aggregate_seconds": seconds(row["duration"]),
            "aggregate_share_percent": (
                100.0 * int(row["duration"] or 0) / aggregate_kernel_ns
                if aggregate_kernel_ns
                else 0.0
            ),
            "average_microseconds": float(row["average_duration"] or 0) / 1_000,
            "maximum_milliseconds": float(row["maximum_duration"] or 0) / 1_000_000,
        }
        for row in kernel_rows[:top]
    ]

    category_values: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "duration": 0}
    )
    for row in kernel_rows:
        values = category_values[kernel_category(str(row["name"]))]
        values["calls"] += int(row["calls"])
        values["duration"] += int(row["duration"] or 0)
    kernel_categories = sorted(
        (
            {
                "name": name,
                "calls": values["calls"],
                "aggregate_seconds": seconds(values["duration"]),
                "aggregate_share_percent": (
                    100.0 * values["duration"] / aggregate_kernel_ns
                    if aggregate_kernel_ns
                    else 0.0
                ),
            }
            for name, values in category_values.items()
        ),
        key=lambda item: item["aggregate_seconds"],
        reverse=True,
    )

    api_rows = connection.execute(
        """
        SELECT name, COUNT(*) AS calls, SUM(duration) AS duration,
               AVG(duration) AS average_duration, MAX(duration) AS maximum_duration
        FROM regions
        WHERE category = 'HIP_RUNTIME_API_EXT'
        GROUP BY name
        ORDER BY duration DESC
        """
    ).fetchall()
    top_apis = [
        {
            "name": str(row["name"]),
            "calls": int(row["calls"]),
            "aggregate_seconds": seconds(row["duration"]),
            "average_microseconds": float(row["average_duration"] or 0) / 1_000,
            "maximum_milliseconds": float(row["maximum_duration"] or 0) / 1_000_000,
        }
        for row in api_rows[:top]
    ]
    api_by_name = {str(row["name"]): row for row in api_rows}

    explicit_copy_values: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "bytes": 0, "api_duration": 0}
    )
    for row in connection.execute(
        """
        WITH memcpy AS (
          SELECT r.id, r.duration,
                 MAX(CASE WHEN a.name = 'kind' THEN a.value END) AS kind,
                 CAST(MAX(CASE WHEN a.name = 'sizeBytes' THEN a.value END) AS INTEGER) AS bytes
          FROM regions AS r
          JOIN region_args AS a ON a.id = r.id
          WHERE r.name = 'hipMemcpyAsync'
          GROUP BY r.id, r.duration
        )
        SELECT kind, bytes, duration FROM memcpy
        """
    ):
        direction = str(row["kind"] or "unknown")
        values = explicit_copy_values[direction]
        values["calls"] += 1
        values["bytes"] += int(row["bytes"] or 0)
        values["api_duration"] += int(row["duration"] or 0)
    explicit_copies = sorted(
        (
            {
                "direction": direction,
                "calls": values["calls"],
                "bytes": values["bytes"],
                "api_seconds": seconds(values["api_duration"]),
                "average_bytes": (
                    values["bytes"] / values["calls"] if values["calls"] else 0.0
                ),
            }
            for direction, values in explicit_copy_values.items()
        ),
        key=lambda item: item["api_seconds"],
        reverse=True,
    )

    transfer_rows = connection.execute(
        """
        SELECT name, src_agent_type, dst_agent_type, COUNT(*) AS calls,
               SUM(size) AS bytes, SUM(duration) AS duration
        FROM memory_copies
        GROUP BY name, src_agent_type, dst_agent_type
        ORDER BY duration DESC
        """
    ).fetchall()
    transfers = [
        {
            "name": str(row["name"]),
            "source": str(row["src_agent_type"]),
            "destination": str(row["dst_agent_type"]),
            "calls": int(row["calls"]),
            "bytes": int(row["bytes"] or 0),
            "device_seconds": seconds(row["duration"]),
        }
        for row in transfer_rows
    ]

    bf_run_calls = int(marker_by_name.get("bf10.run", {}).get("calls", 0))
    cooperative_calls = sum(
        item["calls"]
        for item in kernel_categories
        if item["name"] == "BF cooperative controller"
    )
    fallback_relax_calls = sum(
        item["calls"]
        for item in kernel_categories
        if item["name"] == "BF host-fallback relaxation"
    )
    if cooperative_calls and fallback_relax_calls:
        controller_mode = "mixed cooperative and host fallback"
    elif cooperative_calls:
        controller_mode = "cooperative GPU controller"
    elif fallback_relax_calls:
        controller_mode = "host-controlled fallback"
    else:
        controller_mode = "not detected"

    sssp_durations = [
        int(row[0])
        for row in connection.execute(
            """
            SELECT duration
            FROM regions
            WHERE category = 'MARKER_CORE_RANGE_API'
              AND json_extract(extdata, '$.message') = 'bf10.run'
            """
        )
    ]
    operation_names = (
        "hipLaunchKernel",
        "hipLaunchCooperativeKernel",
        "hipMemcpyAsync",
        "hipStreamSynchronize",
        "hipEventSynchronize",
        "hipMalloc",
        "hipHostMalloc",
    )
    operations = {}
    for name in operation_names:
        row = api_by_name.get(name)
        calls = int(row["calls"]) if row is not None else 0
        operations[name] = {
            "calls": calls,
            "calls_per_bf10_run": calls / bf_run_calls if bf_run_calls else 0.0,
            "aggregate_seconds": seconds(row["duration"]) if row is not None else 0.0,
        }

    summary = {
        "source_database": str(database.resolve()),
        "process": {
            "pid": int(process["pid"]),
            "command": str(process["command"]),
            "wall_seconds": seconds(process_wall_ns),
        },
        "controller_mode": controller_mode,
        "markers": markers,
        "bf10_run_latency_milliseconds": {
            "calls": len(sssp_durations),
            "p50": percentile(sssp_durations, 0.50) / 1_000_000,
            "p90": percentile(sssp_durations, 0.90) / 1_000_000,
            "p95": percentile(sssp_durations, 0.95) / 1_000_000,
            "p99": percentile(sssp_durations, 0.99) / 1_000_000,
            "maximum": max(sssp_durations, default=0) / 1_000_000,
        },
        "kernel_aggregate_seconds": seconds(aggregate_kernel_ns),
        "kernel_categories": kernel_categories,
        "top_kernels": top_kernels,
        "top_hip_apis": top_apis,
        "operations": operations,
        "explicit_memcpy": explicit_copies,
        "transfer_engine": transfers,
        "notes": [
            "ROCTx ranges are nested and may overlap; their aggregate seconds must not be added together.",
            "Kernel aggregate seconds are device work, not wall time when streams overlap.",
            "Calls-per-BF10-run includes process-wide setup APIs and is most useful after the one-time upload is amortized.",
        ],
    }
    connection.close()
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    process = summary["process"]
    latency = summary["bf10_run_latency_milliseconds"]
    lines = [
        "# BF10 LogicNets profile",
        "",
        f"- Inner `pathfinder` wall time: **{process['wall_seconds']:.3f} s**",
        f"- Controller mode: **{summary['controller_mode']}**",
        f"- BF10 calls: **{latency['calls']:,}**",
        (
            "- BF10 latency: "
            f"p50 **{latency['p50']:.3f} ms**, "
            f"p95 **{latency['p95']:.3f} ms**, "
            f"p99 **{latency['p99']:.3f} ms**, "
            f"max **{latency['maximum']:.3f} ms**"
        ),
        "",
        "## ROCTx phase time",
        "",
        "Ranges are nested and can overlap, so use them for attribution rather than adding them.",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Range", "Calls", "Aggregate s", "Average ms", "Max ms"],
            [
                [
                    f"`{item['name']}`",
                    f"{item['calls']:,}",
                    f"{item['aggregate_seconds']:.6f}",
                    f"{item['average_milliseconds']:.3f}",
                    f"{item['maximum_milliseconds']:.3f}",
                ]
                for item in summary["markers"]
            ],
        )
    )
    lines.extend(["", "## GPU time by BF10 kernel category", ""])
    lines.extend(
        markdown_table(
            ["Category", "Calls", "Aggregate s", "GPU-work share"],
            [
                [
                    item["name"],
                    f"{item['calls']:,}",
                    f"{item['aggregate_seconds']:.6f}",
                    f"{item['aggregate_share_percent']:.2f}%",
                ]
                for item in summary["kernel_categories"]
            ],
        )
    )
    lines.extend(["", "## Top kernels", ""])
    lines.extend(
        markdown_table(
            ["Kernel", "Category", "Calls", "Aggregate s", "Average us", "Share"],
            [
                [
                    f"`{item['name']}`",
                    item["category"],
                    f"{item['calls']:,}",
                    f"{item['aggregate_seconds']:.6f}",
                    f"{item['average_microseconds']:.3f}",
                    f"{item['aggregate_share_percent']:.2f}%",
                ]
                for item in summary["top_kernels"]
            ],
        )
    )
    lines.extend(["", "## Host/API time", ""])
    lines.extend(
        markdown_table(
            ["HIP API", "Calls", "Aggregate s", "Average us", "Max ms"],
            [
                [
                    f"`{item['name']}`",
                    f"{item['calls']:,}",
                    f"{item['aggregate_seconds']:.6f}",
                    f"{item['average_microseconds']:.3f}",
                    f"{item['maximum_milliseconds']:.3f}",
                ]
                for item in summary["top_hip_apis"]
            ],
        )
    )
    lines.extend(["", "## Operation counts", ""])
    lines.extend(
        markdown_table(
            ["Operation", "Calls", "Calls per BF10 run", "Aggregate s"],
            [
                [
                    f"`{name}`",
                    f"{item['calls']:,}",
                    f"{item['calls_per_bf10_run']:.3f}",
                    f"{item['aggregate_seconds']:.6f}",
                ]
                for name, item in summary["operations"].items()
            ],
        )
    )
    lines.extend(["", "## Explicit HIP copies", ""])
    lines.extend(
        markdown_table(
            ["Direction", "Calls", "Bytes", "API s", "Average bytes"],
            [
                [
                    f"`{item['direction']}`",
                    f"{item['calls']:,}",
                    f"{item['bytes']:,}",
                    f"{item['api_seconds']:.6f}",
                    f"{item['average_bytes']:.1f}",
                ]
                for item in summary["explicit_memcpy"]
            ],
        )
    )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if not args.database.is_file():
        raise SystemExit(f"database does not exist: {args.database}")
    output_dir = args.output_dir or args.database.with_name(
        f"{args.database.stem}_bf10_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.database, args.top)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render_markdown(summary)
    markdown_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
