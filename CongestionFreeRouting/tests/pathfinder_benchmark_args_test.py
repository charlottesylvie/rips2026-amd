#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pathfinder_benchmark as benchmark


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_arg_rejected(parser, value: str) -> None:
    try:
        parser(value)
    except argparse.ArgumentTypeError:
        return
    raise RuntimeError(f"argument parser unexpectedly accepted {value!r}")


def main() -> None:
    automatic = benchmark.parse_args(
        [
            "input.phys",
            "output.phys",
            "--sssp-engine",
            "delta-step",
            "--delta",
            "auto",
            "--delta-multiplier",
            "0.25",
        ]
    )
    require(automatic.delta == "auto", "benchmark wrapper must retain delta=auto")
    require(
        automatic.delta_multiplier == 0.25,
        "benchmark wrapper must parse the automatic-delta multiplier",
    )
    require(
        benchmark.pathfinder_args(automatic)[:6]
        == [
            "--sssp-engine",
            "delta-step",
            "--delta",
            "auto",
            "--delta-multiplier",
            "0.25",
        ],
        "benchmark wrapper must select delta-step and forward auto controls",
    )

    numeric = benchmark.parse_args(
        ["input.phys", "output.phys", "--delta", "2.5"]
    )
    require(
        benchmark.pathfinder_args(numeric)[:2] == ["--delta", "2.5"],
        "benchmark wrapper must preserve an explicit numeric delta",
    )

    for invalid in ("aut", "0", "-1", "nan", "inf"):
        require_arg_rejected(benchmark.delta_arg, invalid)
    for invalid in ("0", "-1", "nan", "inf"):
        require_arg_rejected(benchmark.positive_float_arg, invalid)

    print("PathFinder benchmark argument test passed")


if __name__ == "__main__":
    main()
