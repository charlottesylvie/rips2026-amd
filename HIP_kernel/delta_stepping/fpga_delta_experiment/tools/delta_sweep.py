#!/usr/bin/env python3
"""Run a delta sweep against an external delta-stepping/router command.

The command should exit 0 for a successful run. If it prints a JSON object on
its last stdout line, these optional fields are captured:

  ok, distance, checksum, relaxations, settled, edge_count, vertex_count

Example:
  python tools/delta_sweep.py \
    --manifest results/benchmark_manifest.csv \
    --deltas 1,2,4,8,16,32 \
    --repeats 3 \
    --command "python run_sssp.py --netlist {netlist} --phys {phys} --delta {delta}"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from string import Formatter


CAPTURE_FIELDS = (
    "ok",
    "distance",
    "checksum",
    "relaxations",
    "settled",
    "edge_count",
    "vertex_count",
    "iterations",
    "converged",
    "target_reached",
    "runtime_ms_avg",
    "runtime_ms_best",
    "route_requests",
    "net_count",
    "attempted_pairs",
    "total_pairs",
    "pairs_per_second",
)


def parse_deltas(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def command_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field, _, _ in Formatter().parse(template):
        if field:
            fields.add(field)
    return fields


def run_command(template: str, row: dict[str, str], delta: float) -> dict[str, object]:
    values = dict(row)
    values["delta"] = str(delta)
    command = template.format(**values)
    start = time.perf_counter()
    completed = subprocess.run(
        shlex.split(command, posix=(os.name != "nt")),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    parsed: dict[str, object] = {}
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            parsed = value
        break

    result: dict[str, object] = {
        "benchmark": row["benchmark"],
        "delta": delta,
        "runtime_seconds": elapsed,
        "exit_code": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-5:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-5:]),
    }
    for field in CAPTURE_FIELDS:
        result[field] = parsed.get(field, "")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/benchmark_manifest.csv", type=Path)
    parser.add_argument("--out", default="results/delta_sweep.csv", type=Path)
    parser.add_argument("--deltas", required=True, help="Comma-separated deltas, e.g. 1,2,4,8,16")
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--command", required=True, help="Command template using {netlist}, {phys}, {delta}")
    parser.add_argument("--limit", default=0, type=int, help="Only run the first N benchmarks")
    args = parser.parse_args()

    required_fields = command_fields(args.command)
    supported_fields = {
        "delta",
        "benchmark",
        "netlist",
        "phys",
        "csr",
        "metadata",
        "netlist_compressed_bytes",
        "phys_compressed_bytes",
        "netlist_uncompressed_bytes",
        "phys_uncompressed_bytes",
    }
    unsupported = required_fields - supported_fields
    if unsupported:
        raise SystemExit(f"unsupported command placeholders: {', '.join(sorted(unsupported))}")

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    deltas = parse_deltas(args.deltas)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "delta",
        "repeat",
        "runtime_seconds",
        "exit_code",
        *CAPTURE_FIELDS,
        "stdout_tail",
        "stderr_tail",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for delta in deltas:
                for repeat in range(1, args.repeats + 1):
                    print(f"{row['benchmark']} delta={delta:g} repeat={repeat}/{args.repeats}", flush=True)
                    result = run_command(args.command, row, delta)
                    result["repeat"] = repeat
                    writer.writerow(result)
                    stream.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
