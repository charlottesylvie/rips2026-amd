#!/usr/bin/env python3
"""Compare HIP SSSP output against the CPU reference with float tolerance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from reference_sssp import dijkstra, load_graph


def load_output(path: Path, nodes: int) -> list[float]:
    result = [math.nan] * nodes
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two fields")
        node = int(fields[0])
        result[node] = math.inf if fields[1] == "INF" else float(fields[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", type=int, nargs="?", default=0)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    expected = dijkstra(graph, args.source)
    actual = load_output(args.output, len(graph))
    failures = 0
    for node, (want, got) in enumerate(zip(expected, actual)):
        equal = (math.isinf(want) and math.isinf(got)) or math.isclose(
            want, got, rel_tol=args.rtol, abs_tol=args.atol
        )
        if not equal:
            print(f"node {node}: expected {want}, got {got}")
            failures += 1
    if failures:
        raise SystemExit(f"{failures} mismatches")
    print(f"all {len(graph)} distances match")


if __name__ == "__main__":
    main()
