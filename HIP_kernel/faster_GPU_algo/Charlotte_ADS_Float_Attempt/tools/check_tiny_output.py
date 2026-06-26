#!/usr/bin/env python3
"""Validate distances for the graph produced by make_tiny_gr.py."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

EXPECTED = {0: 0.0, 1: 1.0, 2: 3.0, 3: 4.0, 4: math.inf}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    observed: dict[int, float] = {}
    for line in args.output.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise SystemExit(f"malformed output line: {line!r}")
        node = int(fields[0])
        observed[node] = math.inf if fields[1] == "INF" else float(fields[1])

    if set(observed) != set(EXPECTED):
        raise SystemExit(
            f"wrong vertex set: observed {sorted(observed)}, "
            f"expected {sorted(EXPECTED)}"
        )
    for node, expected in EXPECTED.items():
        actual = observed[node]
        if math.isinf(expected):
            if not math.isinf(actual):
                raise SystemExit(f"vertex {node}: expected INF, got {actual}")
        elif not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise SystemExit(
                f"vertex {node}: expected {expected}, got {actual}"
            )

    print("tiny graph distances are correct")


if __name__ == "__main__":
    main()
