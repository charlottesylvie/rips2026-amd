#!/usr/bin/env python3
"""Run rocprofv3 and summarize SSSP-new runtime buckets.

The four-category summary is trace-based:
  compute             = GPU kernel dispatch time
  memory_access_ram   = explicit host-visible copy time
  memory_access_vram  = explicit device-to-device copy time
  networking          = RCCL trace time plus optional network-like ROCTx ranges

The CPU/GPU overlap summary is wall-time based:
  CPU-only            = runtime where no traced GPU command was active
  GPU-only            = traced GPU command time with no traced HIP API overlap
  simultaneous        = overlap between traced GPU commands and traced HIP APIs

Kernel time includes both arithmetic and in-kernel global-memory stalls. Use the
counter pass documented in docs/rocprof-runtime-split.md when you need to decide
whether a kernel is compute-bound or vRAM-bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


NS_PER_SEC = 1_000_000_000.0
NETWORK_MARKER_WORDS = ("network", "net/", "mpi", "rccl", "comm", "send", "recv")
GENERATED_CSV_NAMES = {
    "runtime_split.csv",
    "cpu_gpu_overlap.csv",
    "trace_audit.csv",
}


@dataclass
class Bucket:
    seconds: float = 0.0
    event_count: int = 0
    files: set[str] = field(default_factory=set)

    def add(self, interval: tuple[int, int], file_name: str) -> None:
        start, end = interval
        if end < start:
            return
        self.seconds += (end - start) / NS_PER_SEC
        self.event_count += 1
        self.files.add(file_name)


@dataclass
class FileAudit:
    file: str
    rows: int = 0
    parseable_intervals: int = 0
    recognized_events: int = 0
    domains: set[str] = field(default_factory=set)
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    warning: str = ""

    def note_interval(self, interval: tuple[int, int]) -> None:
        start, end = interval
        self.parseable_intervals += 1
        if self.first_timestamp_ns is None or start < self.first_timestamp_ns:
            self.first_timestamp_ns = start
        if self.last_timestamp_ns is None or end > self.last_timestamp_ns:
            self.last_timestamp_ns = end

    def to_row(self) -> dict[str, str]:
        return {
            "file": self.file,
            "rows": str(self.rows),
            "parseable_intervals": str(self.parseable_intervals),
            "recognized_events": str(self.recognized_events),
            "domains": ";".join(sorted(self.domains)),
            "first_timestamp_ns": "" if self.first_timestamp_ns is None else str(self.first_timestamp_ns),
            "last_timestamp_ns": "" if self.last_timestamp_ns is None else str(self.last_timestamp_ns),
            "warning": self.warning,
        }


@dataclass
class TraceSummary:
    buckets: dict[str, Bucket]
    gpu_intervals: list[tuple[int, int]] = field(default_factory=list)
    hip_api_intervals: list[tuple[int, int]] = field(default_factory=list)
    all_intervals: list[tuple[int, int]] = field(default_factory=list)
    audits: list[FileAudit] = field(default_factory=list)
    csv_files_seen: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rocprofv3 and write runtime_split/cpu_gpu_overlap summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="profiles/rocprof-runtime-split",
        help="Output directory for raw rocprof files and summaries.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run rocprofv3; summarize existing CSVs under --out/raw or --out.",
    )
    parser.add_argument(
        "--rocprof",
        default=os.environ.get("ROCPROF", "rocprofv3"),
        help="rocprofv3 executable.",
    )
    parser.add_argument(
        "--extra-rocprof-arg",
        action="append",
        default=[],
        help="Additional argument passed to rocprofv3. Repeat for multiple args.",
    )
    parser.add_argument(
        "program",
        nargs=argparse.REMAINDER,
        help="Program command after --, for example: -- ./bf3 ...",
    )
    args = parser.parse_args()
    if args.program and args.program[0] == "--":
        args.program = args.program[1:]
    if not args.summarize_only and not args.program:
        parser.error("provide a program command after --, or use --summarize-only")
    return args


def normalize_name(name: str | None) -> str:
    if name is None:
        return ""
    text = name.strip().lstrip("\ufeff").strip('"').strip("'").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized[normalize_name(key)] = "" if value is None else value.strip()
    return normalized


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(normalize_name(name))
        if value is not None:
            return value
    return ""


def parse_timestamp(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def interval_from_row(row: dict[str, str]) -> tuple[int, int] | None:
    start = parse_timestamp(row_value(row, "Start_Timestamp", "Start Timestamp", "Begin_Timestamp"))
    end = parse_timestamp(row_value(row, "End_Timestamp", "End Timestamp", "Stop_Timestamp"))
    if start is None or end is None or end < start:
        return None
    return start, end


def has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def classify_trace_row(file_name: str, row: dict[str, str]) -> str:
    name = file_name.lower()
    kind = row_value(row, "Kind").upper()
    domain = row_value(row, "Domain").upper()
    function = row_value(row, "Function").lower()
    operation = row_value(row, "Operation").upper()
    direction = row_value(row, "Direction").upper()
    row_text = " ".join(value.lower() for value in row.values())

    if "kernel_trace" in name or kind == "KERNEL_DISPATCH":
        return "kernel"
    if "memory_copy_trace" in name or kind == "MEMORY_COPY" or direction.startswith("MEMORY_COPY_"):
        return "memory_copy"
    if "hip_api_trace" in name or domain.startswith("HIP_") or function.startswith("hip"):
        return "hip_api"
    if "rccl" in name or "RCCL" in domain or "RCCL" in kind or function.startswith(("rccl", "nccl")):
        return "rccl"
    if "marker" in name or "MARKER" in domain or "MARKER" in kind:
        return "marker_network" if has_any(row_text, NETWORK_MARKER_WORDS) else "marker"
    if "scratch_memory_trace" in name or kind == "SCRATCH_MEMORY" or operation.startswith("SCRATCH_MEMORY"):
        return "scratch_memory"
    if "memory_allocation_trace" in name or kind == "MEMORY_ALLOCATION" or operation.startswith("MEMORY_ALLOCATION"):
        return "memory_allocation"
    return "unknown"


def classify_copy_direction(direction: str) -> str | None:
    text = direction.upper()
    if "HOST_TO_DEVICE" in text or "DEVICE_TO_HOST" in text or "HOST_TO_HOST" in text:
        return "memory_access_ram"
    if "DEVICE_TO_DEVICE" in text:
        return "memory_access_vram"
    return None


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_seconds(intervals: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals)) / NS_PER_SEC


def overlap_seconds(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    left = merge_intervals(a)
    right = merge_intervals(b)
    i = 0
    j = 0
    overlap_ns = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            overlap_ns += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return overlap_ns / NS_PER_SEC


def summarize_csvs(search_root: Path) -> TraceSummary:
    summary = TraceSummary(
        buckets={
            "compute": Bucket(),
            "memory_access_ram": Bucket(),
            "memory_access_vram": Bucket(),
            "networking": Bucket(),
        }
    )

    for path in sorted(search_root.rglob("*.csv")):
        if path.name in GENERATED_CSV_NAMES:
            continue
        summary.csv_files_seen.append(str(path))
        audit = FileAudit(file=str(path))
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    audit.warning = "empty CSV or missing header"
                    summary.audits.append(audit)
                    continue
                for raw_row in reader:
                    audit.rows += 1
                    row = normalize_row(raw_row)
                    interval = interval_from_row(row)
                    domain = classify_trace_row(path.name, row)
                    audit.domains.add(domain)

                    if interval is not None:
                        audit.note_interval(interval)
                        summary.all_intervals.append(interval)

                    if domain == "kernel" and interval is not None:
                        summary.buckets["compute"].add(interval, path.name)
                        summary.gpu_intervals.append(interval)
                        audit.recognized_events += 1
                    elif domain == "memory_copy" and interval is not None:
                        bucket_name = classify_copy_direction(row_value(row, "Direction"))
                        if bucket_name:
                            summary.buckets[bucket_name].add(interval, path.name)
                            audit.recognized_events += 1
                        summary.gpu_intervals.append(interval)
                    elif domain == "hip_api" and interval is not None:
                        summary.hip_api_intervals.append(interval)
                        audit.recognized_events += 1
                    elif domain == "rccl" and interval is not None:
                        summary.buckets["networking"].add(interval, path.name)
                        summary.hip_api_intervals.append(interval)
                        audit.recognized_events += 1
                    elif domain == "marker_network" and interval is not None:
                        summary.buckets["networking"].add(interval, path.name)
                        audit.recognized_events += 1

        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            audit.warning = f"could not read CSV: {exc}"
        if audit.rows > 0 and audit.parseable_intervals == 0:
            audit.warning = "rows were present, but no Start_Timestamp/End_Timestamp interval was parsed"
        summary.audits.append(audit)

    return summary


def trace_span_seconds(summary: TraceSummary) -> float:
    if not summary.all_intervals:
        return 0.0
    start = min(interval[0] for interval in summary.all_intervals)
    end = max(interval[1] for interval in summary.all_intervals)
    return (end - start) / NS_PER_SEC


def write_runtime_split(out_dir: Path,
                        summary: TraceSummary,
                        denominator: float,
                        residual: float) -> None:
    definitions = {
        "compute": "rocprof kernel dispatch duration; includes in-kernel memory stalls",
        "memory_access_ram": "HOST_TO_DEVICE, DEVICE_TO_HOST, and HOST_TO_HOST copy duration",
        "memory_access_vram": "DEVICE_TO_DEVICE copy duration only; kernel global-memory traffic is not split out by tracing",
        "networking": "RCCL trace duration plus ROCTx marker ranges with network-like names",
    }
    csv_path = out_dir / "runtime_split.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "seconds",
                "percent_of_runtime",
                "event_count",
                "definition",
            ],
        )
        writer.writeheader()
        for name in ("compute", "memory_access_ram", "memory_access_vram", "networking"):
            bucket = summary.buckets[name]
            percent = 100.0 * bucket.seconds / denominator if denominator else 0.0
            writer.writerow({
                "category": name,
                "seconds": f"{bucket.seconds:.9f}",
                "percent_of_runtime": f"{percent:.6f}",
                "event_count": bucket.event_count,
                "definition": definitions[name],
            })
        if residual > 0:
            writer.writerow({
                "category": "unattributed_host_or_idle",
                "seconds": f"{residual:.9f}",
                "percent_of_runtime": f"{100.0 * residual / denominator:.6f}" if denominator else "0.000000",
                "event_count": 0,
                "definition": "wall time not covered by the four requested rocprof trace buckets",
            })


def write_cpu_gpu_overlap(out_dir: Path,
                          summary: TraceSummary,
                          denominator: float,
                          wall_seconds: float | None) -> dict[str, float]:
    gpu_active_seconds = interval_seconds(summary.gpu_intervals)
    traced_hip_api_seconds = interval_seconds(summary.hip_api_intervals)
    simultaneous_seconds = overlap_seconds(summary.gpu_intervals, summary.hip_api_intervals)
    gpu_only_seconds = max(0.0, gpu_active_seconds - simultaneous_seconds)
    cpu_only_seconds = max(0.0, denominator - gpu_only_seconds - simultaneous_seconds)

    rows = [
        {
            "category": "cpu_only_or_no_traced_gpu_active",
            "seconds": cpu_only_seconds,
            "definition": "Wall time where no traced GPU kernel/memory-copy command was active; includes ordinary C++ CPU work and file I/O.",
        },
        {
            "category": "gpu_only",
            "seconds": gpu_only_seconds,
            "definition": "Union of traced GPU kernel/memory-copy command time that did not overlap a traced HIP API call.",
        },
        {
            "category": "simultaneous_cpu_gpu",
            "seconds": simultaneous_seconds,
            "definition": "Overlap between traced GPU kernel/memory-copy commands and traced HIP/RCCL API calls.",
        },
    ]
    with (out_dir / "cpu_gpu_overlap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "seconds", "percent_of_runtime", "definition"],
        )
        writer.writeheader()
        for row in rows:
            seconds = row["seconds"]
            writer.writerow({
                "category": row["category"],
                "seconds": f"{seconds:.9f}",
                "percent_of_runtime": f"{100.0 * seconds / denominator:.6f}" if denominator else "0.000000",
                "definition": row["definition"],
            })

    return {
        "cpu_only_or_no_traced_gpu_active_seconds": cpu_only_seconds,
        "gpu_only_seconds": gpu_only_seconds,
        "simultaneous_cpu_gpu_seconds": simultaneous_seconds,
        "gpu_active_union_seconds": gpu_active_seconds,
        "traced_hip_api_union_seconds": traced_hip_api_seconds,
        "wall_seconds": 0.0 if wall_seconds is None else wall_seconds,
    }


def write_trace_audit(out_dir: Path, summary: TraceSummary) -> None:
    with (out_dir / "trace_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "rows",
                "parseable_intervals",
                "recognized_events",
                "domains",
                "first_timestamp_ns",
                "last_timestamp_ns",
                "warning",
            ],
        )
        writer.writeheader()
        for audit in summary.audits:
            writer.writerow(audit.to_row())


def write_outputs(out_dir: Path,
                  summary: TraceSummary,
                  wall_seconds: float | None,
                  returncode: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    span_seconds = trace_span_seconds(summary)
    denominator = wall_seconds or span_seconds
    total_bucket_seconds = sum(bucket.seconds for bucket in summary.buckets.values())
    residual = max(0.0, denominator - total_bucket_seconds) if denominator else 0.0

    write_runtime_split(out_dir, summary, denominator, residual)
    cpu_gpu = write_cpu_gpu_overlap(out_dir, summary, denominator, wall_seconds)
    write_trace_audit(out_dir, summary)

    warnings: list[str] = []
    if not summary.csv_files_seen:
        warnings.append("No raw rocprof CSV files were found under the output directory.")
    if not summary.gpu_intervals:
        warnings.append("No kernel or memory-copy GPU command intervals were recognized.")
    if not summary.hip_api_intervals:
        warnings.append("No HIP/RCCL API intervals were recognized; CPU/GPU overlap will undercount simultaneous CPU/GPU time.")

    payload = {
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "denominator_seconds": denominator,
        "trace_span_seconds": span_seconds,
        "categories": {
            name: {
                "seconds": bucket.seconds,
                "percent_of_runtime": 100.0 * bucket.seconds / denominator if denominator else 0.0,
                "event_count": bucket.event_count,
                "source_files": sorted(bucket.files),
            }
            for name, bucket in summary.buckets.items()
        },
        "unattributed_host_or_idle_seconds": residual,
        "cpu_gpu_overlap": cpu_gpu,
        "csv_files_seen": summary.csv_files_seen,
        "audit": [audit.to_row() for audit in summary.audits],
        "warnings": warnings,
        "notes": [
            "Trace-based runtime_split percentages can exceed 100% if GPU streams overlap.",
            "Kernel dispatch time is compute bucket time, but rocprof tracing alone cannot split arithmetic cycles from in-kernel vRAM stalls.",
            "cpu_gpu_overlap.csv is a wall-time overlap report. CPU-only means no traced GPU command was active, not sampled CPU core utilization.",
            "SIGKILL and node failure can prevent rocprofv3 from flushing raw CSVs; normal exit, exceptions, SIGINT, and SIGTERM are the recoverable cases.",
        ],
    }
    (out_dir / "runtime_split.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_rocprof(args: argparse.Namespace, out_dir: Path) -> tuple[float, int]:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.rocprof,
        "--runtime-trace",
        "--rccl-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(raw_dir),
        "--output-file",
        "trace",
        *args.extra_rocprof_arg,
        "--",
        *args.program,
    ]

    start = time.monotonic()
    child = subprocess.Popen(command)

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, forward)
    old_term = signal.signal(signal.SIGTERM, forward)
    try:
        returncode = child.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    return time.monotonic() - start, returncode


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    wall_seconds: float | None = None
    returncode: int | None = None
    if not args.summarize_only:
        wall_seconds, returncode = run_rocprof(args, out_dir)

    search_root = out_dir / "raw"
    if not search_root.exists():
        search_root = out_dir
    summary = summarize_csvs(search_root)
    write_outputs(out_dir, summary, wall_seconds, returncode)

    print(f"runtime split: {out_dir / 'runtime_split.csv'}")
    print(f"CPU/GPU overlap: {out_dir / 'cpu_gpu_overlap.csv'}")
    print(f"trace audit: {out_dir / 'trace_audit.csv'}")
    print(f"runtime split JSON: {out_dir / 'runtime_split.json'}")
    if not summary.gpu_intervals:
        print("warning: no kernel or memory-copy GPU intervals were recognized", file=sys.stderr)
    if returncode is not None and returncode != 0:
        print(f"profiled program exited with status {returncode}", file=sys.stderr)
    return returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
