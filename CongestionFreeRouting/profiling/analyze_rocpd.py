#!/usr/bin/env python3
"""Summarize a rocprofv3 RocPD SQLite trace and render diagnostic plots.

The script deliberately distinguishes wall-clock intervals from cumulative API
and kernel durations.  Cumulative durations overlap when PathFinder uses
multiple streams and therefore must not be reported as wall-time shares.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NANOSECONDS_PER_SECOND = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="rocprofv3 RocPD SQLite database")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <database-stem>_analysis beside the database)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="write summary.json only; do not require matplotlib",
    )
    return parser.parse_args()


def seconds(nanoseconds: int | float | None) -> float:
    return float(nanoseconds or 0) / NANOSECONDS_PER_SECOND


def percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(quantile * len(sorted_values)) - 1))
    return float(sorted_values[index])


def interval_union(rows: Iterable[tuple[int, int]]) -> tuple[int, int]:
    total = 0
    islands = 0
    union_start: int | None = None
    union_end: int | None = None
    for start, end in rows:
        if end <= start:
            continue
        if union_start is None:
            union_start, union_end = start, end
            continue
        assert union_end is not None
        if start > union_end:
            total += union_end - union_start
            islands += 1
            union_start, union_end = start, end
        elif end > union_end:
            union_end = end
    if union_start is not None and union_end is not None:
        total += union_end - union_start
        islands += 1
    return total, islands


def concurrency_distribution(
    rows: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Return nanoseconds spent at each interval concurrency.

    Input must be ordered by start time.  A min-heap keeps memory proportional
    to maximum concurrency rather than to the number of trace records.
    """

    ends: list[int] = []
    active = 0
    cursor: int | None = None
    distribution: dict[int, int] = defaultdict(int)

    for start, end in rows:
        if end <= start:
            continue
        if cursor is None:
            cursor = start

        while ends and ends[0] <= start:
            next_end = ends[0]
            if next_end > cursor:
                distribution[active] += next_end - cursor
                cursor = next_end
            while ends and ends[0] == next_end:
                heapq.heappop(ends)
                active -= 1

        if start > cursor:
            distribution[active] += start - cursor
            cursor = start

        heapq.heappush(ends, end)
        active += 1

    while ends:
        next_end = ends[0]
        assert cursor is not None
        if next_end > cursor:
            distribution[active] += next_end - cursor
            cursor = next_end
        while ends and ends[0] == next_end:
            heapq.heappop(ends)
            active -= 1

    return dict(distribution)


def classify_kernel(name: str) -> str:
    if name == "__amd_rocclr_copyBuffer":
        return "Runtime status/copy kernels"
    if name == "__amd_rocclr_fillBufferAligned":
        return "Runtime memset/fill kernels"
    if "relax_light_edges_kernel" in name:
        return "Light-edge relaxation"
    if "relax_heavy_edges_kernel" in name:
        return "Heavy-edge relaxation"
    if "reset_touched_vertices_kernel" in name:
        return "Sparse state reset"
    if "materialize_predecessors_kernel" in name:
        return "Predecessor materialization"
    if "clear_flags_from_queue_kernel" in name:
        return "Queue flag clearing"
    if "compact_pending_to_current_bucket_kernel" in name or "reduce_min_pending_bucket_kernel" in name:
        return "Pending-bucket management"
    if "target_paths_kernel" in name:
        return "Path extraction"
    if "mark_settled_targets_kernel" in name:
        return "Target settlement"
    if "initialize_" in name or "validate_device_csr_kernel" in name:
        return "Initialization/validation"
    return "Other"


def add_aggregate(target: dict[str, float | int], calls: int, byte_count: int, duration: int) -> None:
    target["calls"] = int(target.get("calls", 0)) + calls
    target["bytes"] = int(target.get("bytes", 0)) + byte_count
    target["aggregate_api_nanoseconds"] = int(
        target.get("aggregate_api_nanoseconds", 0)
    ) + duration


def analyze(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    required = {"processes", "regions", "kernels", "memory_copies", "region_args"}
    present = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"database is missing required RocPD views: {', '.join(missing)}")

    process = connection.execute(
        "SELECT pid, start, end, command FROM processes ORDER BY start LIMIT 1"
    ).fetchone()
    if process is None:
        raise RuntimeError("database contains no process record")

    marker_rows = connection.execute(
        """
        SELECT json_extract(extdata, '$.message') AS marker,
               COUNT(*) AS calls,
               SUM(duration) AS aggregate_duration,
               AVG(duration) AS average_duration,
               MIN(duration) AS minimum_duration,
               MAX(duration) AS maximum_duration,
               MIN(start) AS first_start,
               MAX(end) AS last_end
        FROM regions
        WHERE category = 'MARKER_CORE_RANGE_API'
        GROUP BY marker
        ORDER BY aggregate_duration DESC
        """
    ).fetchall()
    markers = {
        str(row["marker"]): {
            "calls": int(row["calls"]),
            "aggregate_seconds": seconds(row["aggregate_duration"]),
            "average_seconds": seconds(row["average_duration"]),
            "minimum_seconds": seconds(row["minimum_duration"]),
            "maximum_seconds": seconds(row["maximum_duration"]),
            "first_start_nanoseconds": int(row["first_start"]),
            "last_end_nanoseconds": int(row["last_end"]),
        }
        for row in marker_rows
    }

    top_level_names = (
        "pathfinder.load_csr",
        "pathfinder.load_metadata",
        "pathfinder.run",
        "pathfinder.write_routes",
        "delta_step.upload_graph",
    )
    placeholders = ",".join("?" for _ in top_level_names)
    timeline_rows = connection.execute(
        f"""
        SELECT json_extract(extdata, '$.message') AS marker, start, end, duration
        FROM regions
        WHERE category = 'MARKER_CORE_RANGE_API'
          AND json_extract(extdata, '$.message') IN ({placeholders})
        ORDER BY start
        """,
        top_level_names,
    ).fetchall()
    timeline = [
        {
            "marker": str(row["marker"]),
            "start_nanoseconds": int(row["start"]),
            "end_nanoseconds": int(row["end"]),
            "duration_seconds": seconds(row["duration"]),
        }
        for row in timeline_rows
    ]

    kernel_record = connection.execute(
        """
        SELECT COUNT(*) AS calls, MIN(start) AS first_start, MAX(end) AS last_end,
               SUM(duration) AS aggregate_duration
        FROM kernels
        """
    ).fetchone()
    assert kernel_record is not None
    kernel_union, kernel_islands = interval_union(
        connection.execute("SELECT start, end FROM kernels ORDER BY start, end")
    )
    concurrency_ns = concurrency_distribution(
        connection.execute("SELECT start, end FROM kernels ORDER BY start, end")
    )
    kernel_span = int(kernel_record["last_end"]) - int(kernel_record["first_start"])
    max_concurrency = max(concurrency_ns, default=0)

    category_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "aggregate_nanoseconds": 0}
    )
    for row in connection.execute(
        "SELECT name, COUNT(*) AS calls, SUM(duration) AS duration FROM kernels GROUP BY name"
    ):
        category = classify_kernel(str(row["name"]))
        category_totals[category]["calls"] += int(row["calls"])
        category_totals[category]["aggregate_nanoseconds"] += int(row["duration"])
    aggregate_kernel_ns = int(kernel_record["aggregate_duration"])
    kernel_categories = []
    for category, values in category_totals.items():
        duration = values["aggregate_nanoseconds"]
        kernel_categories.append(
            {
                "category": category,
                "calls": values["calls"],
                "aggregate_seconds": seconds(duration),
                "aggregate_share_percent": 100.0 * duration / aggregate_kernel_ns,
            }
        )
    kernel_categories.sort(key=lambda item: item["aggregate_seconds"], reverse=True)

    copy_buffer_durations = [
        int(row[0])
        for row in connection.execute(
            "SELECT duration FROM kernels WHERE name = '__amd_rocclr_copyBuffer' ORDER BY duration"
        )
    ]
    copy_buffer_total = sum(copy_buffer_durations)
    copy_over_one_ms = sum(value for value in copy_buffer_durations if value >= 1_000_000)

    api_by_name = {}
    for row in connection.execute(
        """
        SELECT name, COUNT(*) AS calls, SUM(duration) AS duration,
               AVG(duration) AS average_duration, MAX(duration) AS maximum_duration
        FROM regions
        WHERE category = 'HIP_RUNTIME_API_EXT'
        GROUP BY name
        ORDER BY duration DESC
        """
    ):
        api_by_name[str(row["name"])] = {
            "calls": int(row["calls"]),
            "aggregate_seconds": seconds(row["duration"]),
            "average_microseconds": float(row["average_duration"]) / 1_000.0,
            "maximum_milliseconds": float(row["maximum_duration"]) / 1_000_000.0,
        }

    worker_sync = []
    for row in connection.execute(
        """
        SELECT tid, COUNT(*) AS calls, MIN(start) AS first_start, MAX(end) AS last_end,
               SUM(duration) AS duration
        FROM regions
        WHERE name = 'hipStreamSynchronize'
        GROUP BY tid
        HAVING COUNT(*) > 10
        ORDER BY tid
        """
    ):
        active_span = int(row["last_end"]) - int(row["first_start"])
        worker_sync.append(
            {
                "tid": int(row["tid"]),
                "calls": int(row["calls"]),
                "aggregate_seconds": seconds(row["duration"]),
                "active_span_seconds": seconds(active_span),
                "active_span_percent": 100.0 * int(row["duration"]) / active_span,
            }
        )

    copy_api: dict[str, Any] = {
        "total": {"calls": 0, "bytes": 0, "aggregate_api_nanoseconds": 0},
        "at_most_64_bytes": {"calls": 0, "bytes": 0, "aggregate_api_nanoseconds": 0},
        "by_direction": {},
        "by_direction_and_size": {},
    }
    memcpy_rows = connection.execute(
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
    )
    for row in memcpy_rows:
        direction = str(row["kind"])
        byte_count = int(row["bytes"])
        duration = int(row["duration"])
        if byte_count <= 64:
            bucket = "at_most_64_bytes"
        elif byte_count <= 4096:
            bucket = "65_bytes_to_4_kib"
        else:
            bucket = "over_4_kib"
        direction_values = copy_api["by_direction"].setdefault(
            direction, {"calls": 0, "bytes": 0, "aggregate_api_nanoseconds": 0}
        )
        bucket_values = copy_api["by_direction_and_size"].setdefault(
            f"{direction}:{bucket}",
            {"calls": 0, "bytes": 0, "aggregate_api_nanoseconds": 0},
        )
        add_aggregate(copy_api["total"], 1, byte_count, duration)
        add_aggregate(direction_values, 1, byte_count, duration)
        add_aggregate(bucket_values, 1, byte_count, duration)
        if byte_count <= 64:
            add_aggregate(copy_api["at_most_64_bytes"], 1, byte_count, duration)

    for section in (
        [copy_api["total"], copy_api["at_most_64_bytes"]]
        + list(copy_api["by_direction"].values())
        + list(copy_api["by_direction_and_size"].values())
    ):
        calls = int(section["calls"])
        duration = int(section.pop("aggregate_api_nanoseconds"))
        section["aggregate_api_seconds"] = seconds(duration)
        section["average_bytes_per_call"] = float(section["bytes"]) / calls if calls else 0.0
        section["average_api_microseconds"] = duration / calls / 1_000.0 if calls else 0.0

    transfer_engine = []
    for row in connection.execute(
        """
        SELECT name, src_agent_type, dst_agent_type, COUNT(*) AS calls,
               SUM(size) AS bytes, SUM(duration) AS duration
        FROM memory_copies
        GROUP BY name, src_agent_type, dst_agent_type
        ORDER BY duration DESC
        """
    ):
        duration = int(row["duration"])
        byte_count = int(row["bytes"])
        transfer_engine.append(
            {
                "name": str(row["name"]),
                "source_type": str(row["src_agent_type"]),
                "destination_type": str(row["dst_agent_type"]),
                "calls": int(row["calls"]),
                "bytes": byte_count,
                "duration_seconds": seconds(duration),
                "effective_gigabytes_per_second": byte_count / duration if duration else 0.0,
            }
        )

    allocation_rows = connection.execute(
        """
        SELECT type, agent_type, COUNT(*) AS calls, SUM(size) AS bytes,
               SUM(duration) AS duration
        FROM memory_allocations
        GROUP BY type, agent_type
        ORDER BY duration DESC
        """
    ).fetchall()
    allocations = [
        {
            "type": str(row["type"]),
            "agent_type": str(row["agent_type"] or "unknown"),
            "calls": int(row["calls"]),
            "bytes": int(row["bytes"]),
            "duration_seconds": seconds(row["duration"]),
        }
        for row in allocation_rows
    ]

    sssp_durations = [
        int(row[0])
        for row in connection.execute(
            """
            SELECT duration FROM regions
            WHERE category = 'MARKER_CORE_RANGE_API'
              AND json_extract(extdata, '$.message') = 'delta_step.generic'
            ORDER BY duration
            """
        )
    ]
    sssp_calls = len(sssp_durations)
    selected_operations = {
        "Kernel launches": api_by_name.get("hipLaunchKernel", {}).get("calls", 0),
        "HIP memcpy calls": api_by_name.get("hipMemcpyAsync", {}).get("calls", 0),
        "Runtime copy kernels": len(copy_buffer_durations),
        "Stream synchronizations": api_by_name.get("hipStreamSynchronize", {}).get("calls", 0),
        "Runtime fill kernels": next(
            (
                item["calls"]
                for item in kernel_categories
                if item["category"] == "Runtime memset/fill kernels"
            ),
            0,
        ),
    }

    agent_rows = connection.execute(
        "SELECT type, name, product_name, extdata FROM rocpd_info_agent ORDER BY id"
    ).fetchall()
    agents = []
    for row in agent_rows:
        try:
            details = json.loads(row["extdata"] or "{}")
        except json.JSONDecodeError:
            details = {}
        agents.append(
            {
                "type": str(row["type"]),
                "name": str(row["name"]),
                "product_name": str(row["product_name"]),
                "cu_count": int(details.get("cu_count", 0)),
                "wavefront_size": int(details.get("wave_front_size", 0)),
                "memory_banks": details.get("mem_banks", []),
                "io_links": details.get("io_links", []),
            }
        )

    counter_counts = {}
    for table in ("pmc_events", "samples", "counters_collection"):
        if table in present:
            counter_counts[table] = int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            )

    summary = {
        "source_database": str(database.resolve()),
        "scope": "inner pathfinder process only; converter, route serializer wrapper, checker, and wirelength are outside this database",
        "process": {
            "pid": int(process["pid"]),
            "command": str(process["command"]),
            "start_nanoseconds": int(process["start"]),
            "end_nanoseconds": int(process["end"]),
            "wall_seconds": seconds(int(process["end"]) - int(process["start"])),
            "timeline": timeline,
        },
        "markers": markers,
        "hardware_agents": agents,
        "device": {
            "kernel_calls": int(kernel_record["calls"]),
            "first_start_nanoseconds": int(kernel_record["first_start"]),
            "last_end_nanoseconds": int(kernel_record["last_end"]),
            "span_seconds": seconds(kernel_span),
            "dispatch_union_seconds": seconds(kernel_union),
            "at_least_one_dispatch_percent_of_span": 100.0 * kernel_union / kernel_span,
            "at_least_one_dispatch_percent_of_process_wall": 100.0
            * kernel_union
            / (int(process["end"]) - int(process["start"])),
            "aggregate_dispatch_seconds": seconds(aggregate_kernel_ns),
            "average_in_flight_concurrency_during_dispatch_union": aggregate_kernel_ns
            / kernel_union,
            "maximum_concurrency": max_concurrency,
            "percent_of_span_at_maximum_concurrency": 100.0
            * concurrency_ns.get(max_concurrency, 0)
            / kernel_span,
            "interval_union_islands": kernel_islands,
            "concurrency_seconds": {
                str(level): seconds(duration)
                for level, duration in sorted(concurrency_ns.items())
            },
            "categories": kernel_categories,
            "copy_buffer_dispatch_duration": {
                "calls": len(copy_buffer_durations),
                "p50_microseconds": percentile(copy_buffer_durations, 0.50) / 1_000.0,
                "p90_microseconds": percentile(copy_buffer_durations, 0.90) / 1_000.0,
                "p95_microseconds": percentile(copy_buffer_durations, 0.95) / 1_000.0,
                "p99_microseconds": percentile(copy_buffer_durations, 0.99) / 1_000.0,
                "at_least_1ms_calls": sum(value >= 1_000_000 for value in copy_buffer_durations),
                "at_least_1ms_share_of_aggregate_percent": 100.0
                * copy_over_one_ms
                / copy_buffer_total,
            },
        },
        "hip_api": {
            "by_name": api_by_name,
            "worker_stream_synchronize": worker_sync,
        },
        "copy_api": copy_api,
        "transfer_engine": transfer_engine,
        "allocations": allocations,
        "sssp": {
            "generic_calls": sssp_calls,
            "duration_milliseconds": {
                "p50": percentile(sssp_durations, 0.50) / 1_000_000.0,
                "p90": percentile(sssp_durations, 0.90) / 1_000_000.0,
                "p95": percentile(sssp_durations, 0.95) / 1_000_000.0,
                "p99": percentile(sssp_durations, 0.99) / 1_000_000.0,
                "maximum": max(sssp_durations, default=0) / 1_000_000.0,
            },
            "operations_per_call": {
                name: float(calls) / sssp_calls if sssp_calls else 0.0
                for name, calls in selected_operations.items()
            },
        },
        "hardware_counter_rows": counter_counts,
    }
    connection.close()
    return summary


def render_plots(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plots; use --no-plots for JSON only") from exc

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    process = summary["process"]
    process_start = process["start_nanoseconds"]

    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    timeline_ax, work_ax, copies_ax, operations_ax = axes.flat

    phase_labels = {
        "pathfinder.load_csr": "Load CSR",
        "pathfinder.load_metadata": "Load metadata",
        "pathfinder.run": "Route",
        "pathfinder.write_routes": "Write routes",
    }
    phase_colors = {
        "pathfinder.load_csr": colors[5],
        "pathfinder.load_metadata": colors[1],
        "pathfinder.run": colors[0],
        "pathfinder.write_routes": colors[2],
    }
    for item in process["timeline"]:
        marker = item["marker"]
        if marker not in phase_labels:
            continue
        start = (item["start_nanoseconds"] - process_start) / NANOSECONDS_PER_SECOND
        timeline_ax.barh(
            2,
            item["duration_seconds"],
            left=start,
            height=0.52,
            color=phase_colors[marker],
            label=f"{phase_labels[marker]} ({item['duration_seconds']:.3f}s)",
        )

    device = summary["device"]
    device_start = (device["first_start_nanoseconds"] - process_start) / NANOSECONDS_PER_SECOND
    timeline_ax.barh(
        1,
        device["span_seconds"],
        left=device_start,
        height=0.52,
        color=colors[3],
        alpha=0.72,
    )
    upload = next(
        (item for item in process["timeline"] if item["marker"] == "delta_step.upload_graph"),
        None,
    )
    if upload:
        upload_start = (upload["start_nanoseconds"] - process_start) / NANOSECONDS_PER_SECOND
        timeline_ax.barh(
            0,
            upload["duration_seconds"],
            left=upload_start,
            height=0.52,
            color=colors[4],
        )
    timeline_ax.set_yticks([0, 1, 2], ["Graph upload", "GPU dispatch span", "ROCTx phases"])
    timeline_ax.set_xlim(0, process["wall_seconds"])
    timeline_ax.set_xlabel("Seconds since inner pathfinder process start")
    timeline_ax.set_title("A. Inner-process timeline")
    timeline_ax.text(
        device_start + device["span_seconds"] / 2,
        1,
        f"≥1 dispatch in flight: {device['at_least_one_dispatch_percent_of_span']:.2f}%\n"
        f"4 concurrent: {device['percent_of_span_at_maximum_concurrency']:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
    )
    timeline_ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    timeline_ax.grid(axis="x", alpha=0.2)

    categories = summary["device"]["categories"]
    display_categories = [item for item in categories if item["aggregate_share_percent"] >= 0.08]
    display_categories.reverse()
    work_ax.barh(
        [item["category"] for item in display_categories],
        [item["aggregate_share_percent"] for item in display_categories],
        color=[colors[index % len(colors)] for index in range(len(display_categories))],
    )
    for index, item in enumerate(display_categories):
        work_ax.text(
            item["aggregate_share_percent"] + 0.35,
            index,
            f"{item['aggregate_share_percent']:.1f}%  ({item['calls']:,})",
            va="center",
            fontsize=8,
        )
    work_ax.set_xlabel("Share of aggregate GPU dispatch duration (%)")
    work_ax.set_title("B. Device work composition (overlapping stream totals)")
    work_ax.set_xlim(0, max(item["aggregate_share_percent"] for item in display_categories) * 1.24)
    work_ax.grid(axis="x", alpha=0.2)

    bucket_labels = {
        "DeviceToHost:at_most_64_bytes": "D2H ≤64 B",
        "DeviceToHost:65_bytes_to_4_kib": "D2H 65 B–4 KiB",
        "DeviceToHost:over_4_kib": "D2H >4 KiB",
        "HostToDevice:at_most_64_bytes": "H2D ≤64 B",
        "HostToDevice:65_bytes_to_4_kib": "H2D 65 B–4 KiB",
        "HostToDevice:over_4_kib": "H2D >4 KiB",
        "DeviceToDevice:at_most_64_bytes": "D2D ≤64 B",
    }
    bucket_data = []
    for key, label in bucket_labels.items():
        values = summary["copy_api"]["by_direction_and_size"].get(key)
        if values and values["calls"]:
            bucket_data.append((label, values))
    bucket_data.reverse()
    copies_ax.barh(
        [label for label, _ in bucket_data],
        [values["calls"] for _, values in bucket_data],
        color=[colors[index % len(colors)] for index in range(len(bucket_data))],
    )
    copies_ax.set_xscale("log")
    copies_ax.set_xlabel("HIP memcpy calls (log scale)")
    h2d = next(
        (item for item in summary["transfer_engine"] if item["name"] == "MEMORY_COPY_HOST_TO_DEVICE"),
        None,
    )
    copy_title = "C. CPU↔GPU copy-call pattern"
    if h2d:
        copy_title += (
            f"\nObserved bulk copy: {h2d['bytes']/1048576:.1f} MiB H2D / "
            f"{h2d['duration_seconds']*1000:.3f} ms / "
            f"{h2d['effective_gigabytes_per_second']:.1f} GB/s"
        )
    copies_ax.set_title(copy_title)
    for index, (_, values) in enumerate(bucket_data):
        copies_ax.text(
            values["calls"] * 1.08,
            index,
            f"{values['bytes']/1048576:.3g} MiB, API {values['average_api_microseconds']:.1f} µs/call",
            va="center",
            fontsize=8,
        )
    copies_ax.grid(axis="x", which="both", alpha=0.2)

    operations = summary["sssp"]["operations_per_call"]
    operation_items = list(operations.items())[::-1]
    operations_ax.barh(
        [name for name, _ in operation_items],
        [value for _, value in operation_items],
        color=[colors[index % len(colors)] for index in range(len(operation_items))],
    )
    for index, (_, value) in enumerate(operation_items):
        operations_ax.text(value + 0.35, index, f"{value:.1f}", va="center", fontsize=8)
    operations_ax.set_xlim(0, max(operations.values()) * 1.2)
    operations_ax.set_xlabel("Operations per generic SSSP")
    latency = summary["sssp"]["duration_milliseconds"]
    sync_values = [item["active_span_percent"] for item in summary["hip_api"]["worker_stream_synchronize"]]
    operations_ax.set_title(
        "D. Fine-grained control per SSSP\n"
        f"SSSP p50/90/99: {latency['p50']:.1f}/{latency['p90']:.1f}/{latency['p99']:.1f} ms · "
        f"worker sync: {min(sync_values):.1f}–{max(sync_values):.1f}%"
    )
    operations_ax.grid(axis="x", alpha=0.2)

    figure.suptitle(
        "logicnets_jscl · forced generic delta-step · rocprofv3 runtime trace",
        fontweight="bold",
    )
    png_path = output_dir / "profile-summary.png"
    svg_path = output_dir / "profile-summary.svg"
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    return png_path, svg_path


def main() -> int:
    args = parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else database.with_name(database.stem + "_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze(database)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    if not args.no_plots:
        for path in render_plots(summary, output_dir):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
