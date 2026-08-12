#!/usr/bin/env python3
"""Sweep the BF11/Delta benchmark over graph sizes and write CSV results."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ENGINE_FIELDS = (
    "gpu_mean_ms",
    "gpu_p50_ms",
    "gpu_p95_ms",
    "gpu_min_ms",
    "gpu_max_ms",
    "wall_mean_ms",
    "wall_p50_ms",
    "wall_p95_ms",
    "wall_min_ms",
    "wall_max_ms",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_engine_rows(output: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^\s*(BF11|Delta-Stepping)\s+(.+?)\s*$")
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        values = match.group(2).split()
        if len(values) != len(ENGINE_FIELDS):
            raise ValueError(
                f"unexpected timing column count for {match.group(1)}: {line}"
            )
        key = "bf11" if match.group(1) == "BF11" else "delta"
        rows[key] = {
            field: float(value) for field, value in zip(ENGINE_FIELDS, values)
        }
    if set(rows) != {"bf11", "delta"}:
        raise ValueError("benchmark output did not contain both timing rows")
    return rows


def parse_scalar(output: str, label: str, cast: type[int] | type[float]):
    match = re.search(rf"^\s*{re.escape(label)}:\s+([^\s]+)", output, re.MULTILINE)
    if not match:
        raise ValueError(f"benchmark output did not contain {label!r}")
    return cast(match.group(1).removesuffix("x"))


def parse_result(output: str) -> dict[str, int | float]:
    engines = parse_engine_rows(output)
    result: dict[str, int | float] = {
        "edges": parse_scalar(output, "edges", int),
        "gpu_bf11_time_over_delta_time": parse_scalar(output, "GPU mean", float),
        "wall_bf11_time_over_delta_time": parse_scalar(output, "Wall mean", float),
    }
    for engine, values in engines.items():
        for field, value in values.items():
            result[f"{engine}_{field}"] = value
    return result


def fieldnames() -> list[str]:
    names = [
        "vertices",
        "edges",
        "degree",
        "delta_value",
        "warmups",
        "repeats",
        "queries",
        "seed",
        "mode",
        "timed_calls_per_engine",
        "sweep_elapsed_seconds",
        "binary",
    ]
    for engine in ("bf11", "delta"):
        names.extend(f"{engine}_{field}" for field in ENGINE_FIELDS)
    names.extend(
        ("gpu_bf11_time_over_delta_time", "wall_bf11_time_over_delta_time")
    )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production BF11/Delta SSSP benchmark at several vertex "
            "counts and save one timing row per graph to CSV."
        )
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/tmp/bf11_vs_delta_unbounded_benchmark"),
        help="compiled benchmark executable",
    )
    parser.add_argument(
        "--vertices",
        type=positive_int,
        nargs="+",
        required=True,
        help="vertex counts to benchmark, in execution order",
    )
    parser.add_argument("--degree", type=positive_int, default=4)
    parser.add_argument("--delta", type=float, default=4.0, dest="delta_value")
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--queries", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=("full", "target"), default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bf11_vs_delta_sssp_sweep.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append rows instead of replacing the output file",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="optional directory for complete stdout/stderr logs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {args.binary}")
    if not args.delta_value > 0:
        raise SystemExit("--delta must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    write_header = not args.append or not args.output.exists() or args.output.stat().st_size == 0
    file_mode = "a" if args.append else "w"

    with args.output.open(file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames())
        if write_header:
            writer.writeheader()

        for index, vertices in enumerate(args.vertices, start=1):
            command = [
                str(args.binary),
                str(vertices),
                str(args.degree),
                str(args.delta_value),
                str(args.warmups),
                str(args.repeats),
                str(args.queries),
                str(args.seed),
                args.mode,
            ]
            print(
                f"[{index}/{len(args.vertices)}] vertices={vertices}: "
                + " ".join(command),
                flush=True,
            )
            started = time.monotonic()
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.monotonic() - started
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")

            if args.log_dir:
                log_path = args.log_dir / (
                    f"sssp_{args.mode}_v{vertices}_d{args.degree}_seed{args.seed}.log"
                )
                log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise SystemExit(
                    f"benchmark failed for {vertices} vertices with exit code "
                    f"{completed.returncode}"
                )

            try:
                parsed = parse_result(completed.stdout)
            except ValueError as error:
                raise SystemExit(
                    f"could not parse benchmark result for {vertices} vertices: {error}"
                ) from error
            row: dict[str, object] = {
                "vertices": vertices,
                "degree": args.degree,
                "delta_value": args.delta_value,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "queries": args.queries,
                "seed": args.seed,
                "mode": args.mode,
                "timed_calls_per_engine": args.repeats * args.queries,
                "sweep_elapsed_seconds": round(elapsed, 6),
                "binary": str(args.binary.resolve()),
                **parsed,
            }
            writer.writerow(row)
            csv_file.flush()
            print(f"saved row to {args.output}", flush=True)

    print(f"completed {len(args.vertices)} graph sizes: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
