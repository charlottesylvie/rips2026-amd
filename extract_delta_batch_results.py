#!/usr/bin/env python3
"""Convert saved Delta light-round batch logs into analysis-ready CSV files.

The script only reads ``*.log`` files.  It recognizes the filenames used by
the documented sweep (``<benchmark>-batch<N>-run<N>.log``) and telemetry logs
(``telemetry-batch<N>.log``), extracts Make's end-to-end timings, and flattens
PathFinder's aggregate Delta telemetry JSON when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1"
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
WALL_CLOCK = re.compile(r"wall[- ]clock\s+time\s*\(sec\)\s*:\s*" + NUMBER, re.I)
USER_CPU = re.compile(r"user[- ]cpu\s+time\s*\(sec\)\s*:\s*" + NUMBER, re.I)
RUN_NAME = re.compile(r"^(?P<benchmark>.+)-batch(?P<batch>\d+)-run(?P<run>\d+)\.log$")
TELEMETRY_NAME = re.compile(r"^telemetry-batch(?P<batch>\d+)(?:-run(?P<run>\d+))?\.log$")

RESULT_COLUMNS = (
    "schema_version",
    "source_log",
    "record_type",
    "benchmark",
    "light_round_batch",
    "run",
    "wall_clock_seconds",
    "user_cpu_seconds",
    "telemetry_records",
    "queries",
    "completed_queries",
    "resolved_delta",
    "wavefront_size",
    "parallel_workers",
    "force_generic",
    "light_round_batch_reported",
    "execution_exact_unit",
    "execution_compact_generic",
    "execution_legacy_generic",
    "execution_generic_distances_only",
    "outer_buckets_processed",
    "light_relaxation_rounds",
    "heavy_edge_phases",
    "frontier_entries_processed",
    "light_edge_visits",
    "heavy_edge_visits",
    "successful_distance_relaxations",
    "reached_vertices",
    "controller_round_trips",
    "controller_round_trips_per_completed_query",
    "parse_notes",
)

SUMMARY_COLUMNS = (
    "benchmark",
    "light_round_batch",
    "record_type",
    "samples",
    "wall_clock_seconds_min",
    "wall_clock_seconds_median",
    "wall_clock_seconds_max",
    "user_cpu_seconds_median",
    "controller_round_trips_median",
    "controller_round_trips_per_completed_query_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_directory", type=Path, help="directory containing the saved batch-results logs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="detailed CSV path (default: <log_directory>/delta-light-batch-results.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="summary CSV path (default: <log_directory>/delta-light-batch-summary.csv)",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="benchmark name for telemetry-only filenames that do not encode one",
    )
    return parser.parse_args()


def last_number(pattern: re.Pattern[str], text: str) -> str:
    matches = list(pattern.finditer(text))
    return matches[-1].group(1) if matches else ""


def telemetry_records(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        marker = '{"type":"delta_stepping_telemetry"'
        if marker not in line:
            continue
        start = line.find(marker)
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError as exc:
            notes.append(f"invalid telemetry JSON on line {line_number}: {exc.msg}")
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, notes


def number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return format(value, ".12g")
    return ""


def boolean(value: Any) -> str:
    return "true" if value is True else "false" if value is False else ""


def record_value(record: dict[str, Any], key: str) -> Any:
    return record.get(key)


def nested_value(record: dict[str, Any], section: str, key: str) -> Any:
    nested = record.get(section)
    return nested.get(key) if isinstance(nested, dict) else None


def filename_metadata(path: Path, benchmark_override: str) -> tuple[str, str, str, str]:
    run_match = RUN_NAME.match(path.name)
    if run_match:
        return (
            "timing_run",
            run_match.group("benchmark"),
            run_match.group("batch"),
            run_match.group("run"),
        )
    telemetry_match = TELEMETRY_NAME.match(path.name)
    if telemetry_match:
        return (
            "telemetry",
            benchmark_override,
            telemetry_match.group("batch"),
            telemetry_match.group("run") or "",
        )
    return "", "", "", ""


def collect_row(log_path: Path, root: Path, benchmark_override: str) -> dict[str, str] | None:
    record_type, benchmark, batch, run = filename_metadata(log_path, benchmark_override)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records, notes = telemetry_records(text)
    if not record_type and not records:
        return None
    if not record_type:
        record_type = "telemetry"
        notes.append("batch and run were not encoded in the filename")
    record: dict[str, Any] = records[-1] if records else {}
    if not batch:
        batch = number(record_value(record, "light_round_batch"))
    if not benchmark:
        benchmark = benchmark_override
    if records and len(records) > 1:
        notes.append(f"found {len(records)} telemetry records; used the last")
    if records and batch and number(record_value(record, "light_round_batch")) not in ("", batch):
        notes.append("filename batch differs from telemetry batch")
    if not last_number(WALL_CLOCK, text):
        notes.append("wall-clock time not found")

    counters = record.get("counters") if isinstance(record.get("counters"), dict) else {}
    paths = record.get("execution_paths") if isinstance(record.get("execution_paths"), dict) else {}
    completed_queries = number(record_value(record, "completed_queries"))
    controller_round_trips = number(counters.get("controller_round_trips"))
    trips_per_query = ""
    if completed_queries and controller_round_trips and float(completed_queries) != 0.0:
        trips_per_query = format(
            float(controller_round_trips) / float(completed_queries), ".12g"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_log": str(log_path.relative_to(root)),
        "record_type": record_type,
        "benchmark": benchmark,
        "light_round_batch": batch,
        "run": run,
        "wall_clock_seconds": last_number(WALL_CLOCK, text),
        "user_cpu_seconds": last_number(USER_CPU, text),
        "telemetry_records": str(len(records)),
        "queries": number(record_value(record, "queries")),
        "completed_queries": completed_queries,
        "resolved_delta": number(record_value(record, "resolved_delta")),
        "wavefront_size": number(record_value(record, "wavefront_size")),
        "parallel_workers": number(record_value(record, "parallel_workers")),
        "force_generic": boolean(record_value(record, "force_generic")),
        "light_round_batch_reported": number(record_value(record, "light_round_batch")),
        "execution_exact_unit": number(paths.get("exact_unit")),
        "execution_compact_generic": number(paths.get("compact_generic")),
        "execution_legacy_generic": number(paths.get("legacy_generic")),
        "execution_generic_distances_only": number(paths.get("generic_distances_only")),
        "outer_buckets_processed": number(counters.get("outer_buckets_processed")),
        "light_relaxation_rounds": number(counters.get("light_relaxation_rounds")),
        "heavy_edge_phases": number(counters.get("heavy_edge_phases")),
        "frontier_entries_processed": number(counters.get("frontier_entries_processed")),
        "light_edge_visits": number(counters.get("light_edge_visits")),
        "heavy_edge_visits": number(counters.get("heavy_edge_visits")),
        "successful_distance_relaxations": number(counters.get("successful_distance_relaxations")),
        "reached_vertices": number(counters.get("reached_vertices")),
        "controller_round_trips": controller_round_trips,
        "controller_round_trips_per_completed_query": trips_per_query,
        "parse_notes": "; ".join(notes),
    }


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def median(values: list[float]) -> str:
    return format(statistics.median(values), ".12g") if values else ""


def summary_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["light_round_batch"], row["record_type"])].append(row)
    output: list[dict[str, str]] = []
    for (benchmark, batch, record_type), group in sorted(groups.items()):
        def values(column: str) -> list[float]:
            return [float(row[column]) for row in group if row[column]]

        wall = values("wall_clock_seconds")
        output.append(
            {
                "benchmark": benchmark,
                "light_round_batch": batch,
                "record_type": record_type,
                "samples": str(len(group)),
                "wall_clock_seconds_min": format(min(wall), ".12g") if wall else "",
                "wall_clock_seconds_median": median(wall),
                "wall_clock_seconds_max": format(max(wall), ".12g") if wall else "",
                "user_cpu_seconds_median": median(values("user_cpu_seconds")),
                "controller_round_trips_median": median(values("controller_round_trips")),
                "controller_round_trips_per_completed_query_median": median(
                    values("controller_round_trips_per_completed_query")
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    root = args.log_directory.resolve()
    if not root.is_dir():
        print(f"error: log directory does not exist: {root}", file=sys.stderr)
        return 2
    output = (args.output or root / "delta-light-batch-results.csv").resolve()
    summary = (args.summary or root / "delta-light-batch-summary.csv").resolve()
    rows = [
        row
        for log_path in sorted(root.rglob("*.log"))
        if (row := collect_row(log_path, root, args.benchmark)) is not None
    ]
    if not rows:
        print("error: no batch-named or Delta telemetry log files found", file=sys.stderr)
        return 2
    count = write_csv(output, RESULT_COLUMNS, rows)
    write_csv(summary, SUMMARY_COLUMNS, summary_rows(rows))
    print(f"Wrote {count} detailed row(s) to {output}")
    print(f"Wrote batch summary to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
