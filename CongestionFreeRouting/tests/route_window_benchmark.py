#!/usr/bin/env python3
"""Reproducible end-to-end RouteWindow A/B benchmark.

The benchmark alternates unbounded and windowed Delta-Stepping runs, collects
per-query window telemetry for both modes, and reports median wall-clock
speedup.  It deliberately measures the full PathFinder invocation rather than
only a bounded query, because global-cost verification is part of the Route
Window correctness contract.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


CONTROLLED_PATHFINDER_OPTIONS = {
    "--sssp-engine",
    "--use-delta-step",
    "--route-window",
    "--route-window-min-margin",
    "--route-window-max-margin",
    "--route-window-margin-scale",
    "--route-window-net-list",
    "--route-window-stats-out",
    "--routes-out",
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def finite_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare unbounded and adaptive route-window Delta-Stepping on "
            "one CSR/metadata pair.  Arguments after '--' are forwarded to "
            "both PathFinder invocations."
        )
    )
    parser.add_argument("--pathfinder", required=True, type=Path,
                        help="PathFinder executable")
    parser.add_argument("--graph", required=True, type=Path,
                        help="CSR graph (.csrbin)")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="routing metadata sidecar (.ifmeta.bin)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new or empty directory for logs, JSONL, and summary.json")
    parser.add_argument("--repetitions", type=positive_int, default=5,
                        help="measured repetitions per mode (default: 5)")
    parser.add_argument("--warmups", type=nonnegative_int, default=1,
                        help="unmeasured repetitions per mode (default: 1)")
    parser.add_argument("--window-min-margin", type=nonnegative_int,
                        help="window-only initial X/Y margin floor")
    parser.add_argument("--window-max-margin", type=nonnegative_int,
                        help="window-only initial X/Y margin cap")
    parser.add_argument("--window-margin-scale", type=finite_nonnegative_float,
                        help="window-only endpoint-span scale")
    parser.add_argument("--window-net-list", type=Path,
                        help="optional JSONL list of nets to window")
    parser.add_argument("forwarded_args", nargs=argparse.REMAINDER,
                        help="PathFinder arguments shared by both modes; prefix with '--'")
    return parser


def normalized_forwarded_args(values: list[str]) -> list[str]:
    if values[:1] == ["--"]:
        values = values[1:]
    for value in values:
        option = value.split("=", 1)[0]
        if option in CONTROLLED_PATHFINDER_OPTIONS:
            raise ValueError(
                f"{option} is controlled by route_window_benchmark.py; use its "
                "dedicated option instead"
            )
    return values


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"PathFinder did not produce telemetry: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}") from error
        if isinstance(record, dict):
            records.append(record)
    return records


def aggregate_telemetry(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    execution_paths: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for record in records:
        by_kind[str(record.get("kind", "unknown"))] += 1
        execution_paths[str(record.get("execution_path", "unknown"))] += 1
        for field in (
            "reached_target_count",
            "unreached_target_count",
            "touched_nodes",
            "edge_visits",
            "window_rejected_edges",
            "window_unknown_coordinate_nodes",
        ):
            value = record.get(field, 0)
            if isinstance(value, (int, float)):
                totals[field] += value
        if record.get("global_cost_verification_required") is True:
            totals["global_cost_verifications_required"] += 1
        if record.get("fallback_triggered") is True:
            totals["fallbacks_triggered"] += 1
    return {
        "record_count": len(records),
        "queries_by_kind": dict(sorted(by_kind.items())),
        "execution_paths": dict(sorted(execution_paths.items())),
        "totals": dict(sorted(totals.items())),
    }


def describe_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": samples,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "mean_seconds": sum(samples) / len(samples),
        "median_seconds": statistics.median(samples),
    }


def mode_command(args: argparse.Namespace,
                 forwarded_args: list[str],
                 mode: str,
                 stats_path: Path) -> list[str]:
    command = [
        str(args.pathfinder),
        str(args.graph),
        str(args.metadata),
        "--sssp-engine",
        "delta-step",
        *forwarded_args,
        "--route-window-stats-out",
        str(stats_path),
    ]
    if mode == "windowed":
        command.append("--route-window")
        if args.window_min_margin is not None:
            command.extend(["--route-window-min-margin", str(args.window_min_margin)])
        if args.window_max_margin is not None:
            command.extend(["--route-window-max-margin", str(args.window_max_margin)])
        if args.window_margin_scale is not None:
            command.extend(["--route-window-margin-scale", str(args.window_margin_scale)])
        if args.window_net_list is not None:
            command.extend(["--route-window-net-list", str(args.window_net_list)])
    return command


def run_sample(args: argparse.Namespace,
               forwarded_args: list[str],
               mode: str,
               ordinal: int,
               output_dir: Path) -> tuple[float, dict[str, Any]]:
    stem = f"{mode}-{ordinal:03d}"
    stats_path = output_dir / f"{stem}.jsonl"
    log_path = output_dir / f"{stem}.log"
    command = mode_command(args, forwarded_args, mode, stats_path)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"{mode} run {ordinal} failed with exit code {result.returncode}; "
            f"see {log_path}"
        )
    return elapsed, aggregate_telemetry(read_jsonl(stats_path))


def main() -> int:
    args = make_parser().parse_args()
    try:
        args.pathfinder = args.pathfinder.resolve()
        args.graph = args.graph.resolve()
        args.metadata = args.metadata.resolve()
        args.output_dir = args.output_dir.resolve()
        if args.window_net_list is not None:
            args.window_net_list = args.window_net_list.resolve()
        if not args.pathfinder.is_file():
            raise ValueError(f"PathFinder executable does not exist: {args.pathfinder}")
        if not args.graph.is_file():
            raise ValueError(f"CSR graph does not exist: {args.graph}")
        if not args.metadata.is_file():
            raise ValueError(f"metadata sidecar does not exist: {args.metadata}")
        if (args.window_min_margin is not None and
                args.window_max_margin is not None and
                args.window_min_margin > args.window_max_margin):
            raise ValueError("--window-min-margin must not exceed --window-max-margin")
        if args.window_net_list is not None and not args.window_net_list.is_file():
            raise ValueError(f"window net list does not exist: {args.window_net_list}")
        forwarded_args = normalized_forwarded_args(args.forwarded_args)
        prepare_output_dir(args.output_dir)
    except ValueError as error:
        make_parser().error(str(error))

    measured_samples: dict[str, list[float]] = {"unbounded": [], "windowed": []}
    measured_telemetry: dict[str, list[dict[str, Any]]] = {
        "unbounded": [], "windowed": []
    }
    total_rounds = args.warmups + args.repetitions
    for round_index in range(total_rounds):
        order = ("unbounded", "windowed")
        if round_index % 2:
            order = tuple(reversed(order))
        for mode in order:
            elapsed, telemetry = run_sample(
                args, forwarded_args, mode, round_index, args.output_dir)
            if round_index >= args.warmups:
                measured_samples[mode].append(elapsed)
                measured_telemetry[mode].append(telemetry)

    unbounded = describe_samples(measured_samples["unbounded"])
    windowed = describe_samples(measured_samples["windowed"])
    speedup = unbounded["median_seconds"] / windowed["median_seconds"]
    summary = {
        "schema_version": 1,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "pathfinder": str(args.pathfinder),
        "graph": str(args.graph),
        "metadata": str(args.metadata),
        "forwarded_args": forwarded_args,
        "unbounded": unbounded,
        "windowed": windowed,
        "median_speedup_unbounded_over_windowed": speedup,
        "telemetry_by_mode": measured_telemetry,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(f"unbounded median: {unbounded['median_seconds']:.3f} s")
    print(f"windowed median:  {windowed['median_seconds']:.3f} s")
    print(f"speedup:          {speedup:.3f}x (unbounded/windowed)")
    print(f"summary:          {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
