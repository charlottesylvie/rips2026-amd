#!/usr/bin/env python3
"""Summarize delta sweep results and choose global/per-benchmark winners."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def is_success(row: dict[str, str]) -> bool:
    if row.get("exit_code") not in ("0", 0):
        return False
    ok = str(row.get("ok", "")).strip().lower()
    return ok not in ("false", "0", "no")


def median_runtime(rows: list[dict[str, str]]) -> float:
    return statistics.median(float(row["runtime_seconds"]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="results/delta_sweep.csv", type=Path)
    parser.add_argument("--summary", default="results/delta_summary.csv", type=Path)
    args = parser.parse_args()

    rows = [row for row in load_rows(args.sweep) if is_success(row)]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["delta"])].append(row)

    by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (benchmark, delta), group in grouped.items():
        by_benchmark[benchmark].append(
            {
                "benchmark": benchmark,
                "delta": delta,
                "median_runtime_seconds": median_runtime(group),
                "runs": len(group),
            }
        )

    winners = {}
    for benchmark, items in by_benchmark.items():
        best = min(items, key=lambda item: float(item["median_runtime_seconds"]))
        winners[benchmark] = best

    delta_scores: dict[str, list[float]] = defaultdict(list)
    for benchmark, items in by_benchmark.items():
        best_runtime = float(winners[benchmark]["median_runtime_seconds"])
        for item in items:
            ratio = float(item["median_runtime_seconds"]) / best_runtime
            delta_scores[str(item["delta"])].append(ratio)

    global_rows = []
    for delta, ratios in sorted(delta_scores.items(), key=lambda item: float(item[0])):
        geometric_mean = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
        global_rows.append(
            {
                "kind": "global_delta",
                "benchmark": "",
                "delta": delta,
                "median_runtime_seconds": "",
                "runs": len(ratios),
                "slowdown_vs_best": geometric_mean,
            }
        )

    per_benchmark_rows = []
    for benchmark in sorted(winners):
        winner = winners[benchmark]
        per_benchmark_rows.append(
            {
                "kind": "best_for_benchmark",
                "benchmark": benchmark,
                "delta": winner["delta"],
                "median_runtime_seconds": winner["median_runtime_seconds"],
                "runs": winner["runs"],
                "slowdown_vs_best": 1.0,
            }
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kind", "benchmark", "delta", "median_runtime_seconds", "runs", "slowdown_vs_best"]
    with args.summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_benchmark_rows)
        writer.writerows(global_rows)

    if global_rows:
        best_global = min(global_rows, key=lambda row: float(row["slowdown_vs_best"]))
        print(
            "best global delta:",
            best_global["delta"],
            f"(geomean slowdown {float(best_global['slowdown_vs_best']):.3f}x)",
        )
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
