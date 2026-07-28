#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
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


def require_parse_rejected(argv: list[str], message: str) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            benchmark.parse_args(["input.phys", "output.phys", *argv])
    except SystemExit as exc:
        require(exc.code != 0, message + ": parser exited successfully")
        return
    raise RuntimeError(message)


def require_current_metadata_pair_versions() -> None:
    high = 0x123456789ABCDEF0
    low = 0x0FEDCBA987654321
    expected = f"{high:016x}{low:016x}"
    with tempfile.TemporaryDirectory() as directory:
        for version in (5, 6):
            metadata = Path(directory) / f"metadata-v{version}.ifmeta.bin"
            metadata.write_bytes(
                b"RIPSIFM1"
                + version.to_bytes(8, byteorder=sys.byteorder)
                + (2).to_bytes(8, byteorder=sys.byteorder)
                + high.to_bytes(8, byteorder=sys.byteorder)
                + low.to_bytes(8, byteorder=sys.byteorder)
            )
            Path(str(metadata) + ".generation").write_text(
                expected + "\n", encoding="ascii"
            )
            require(
                benchmark.read_metadata_artifact_pair_id(metadata) == expected,
                f"metadata v{version} pair ID was not accepted",
            )


def main() -> None:
    require_current_metadata_pair_versions()
    defaults = benchmark.parse_args(["input.phys", "output.phys"])
    require(
        not defaults.delta_force_generic,
        "force-generic benchmarking must remain disabled by default",
    )
    require(
        not defaults.delta_unit_weight_load_elision
        and not defaults.delta_host_value_frontier_count
        and not defaults.delta_wave_queue_reservations,
        "generic kernel optimization controls must remain disabled by default",
    )
    require(
        not defaults.delta_telemetry,
        "delta telemetry must remain disabled by default",
    )
    require(
        defaults.delta_benchmark_weights is None
        and defaults.delta_benchmark_weight_seed is None,
        "benchmark weights and seed must remain unset by default",
    )
    require(
        defaults.delta_controller is None
        and defaults.delta_controller_batch_size is None,
        "controller controls must remain unset by default",
    )
    require(
        benchmark.pathfinder_args(defaults) == [],
        "default wrapper arguments must not select or tune delta-stepping",
    )

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
        [
            "input.phys",
            "output.phys",
            "--sssp-engine",
            "delta-step",
            "--delta",
            "2.5",
        ]
    )
    require(
        benchmark.pathfinder_args(numeric)[:4]
        == ["--sssp-engine", "delta-step", "--delta", "2.5"],
        "benchmark wrapper must preserve an explicit numeric delta",
    )

    forced = benchmark.parse_args(
        [
            "input.phys",
            "output.phys",
            "--use-delta-step",
            "--delta",
            "0.75",
            "--delta-force-generic",
            "--delta-unit-weight-load-elision",
            "--delta-host-value-frontier-count",
            "--delta-wave-queue-reservations",
            "--delta-telemetry",
            "--delta-telemetry",
            "--delta-force-legacy-parent",
            "--delta-controller",
            "reduced-round-trip",
            "--delta-controller-batch-size",
            "7",
            "--delta-benchmark-weights",
            "mixed",
            "--delta-benchmark-weight-seed",
            "17",
        ]
    )
    require(
        benchmark.pathfinder_args(forced)
        == [
            "--use-delta-step",
            "--delta-force-generic",
            "--delta-unit-weight-load-elision",
            "--delta-host-value-frontier-count",
            "--delta-wave-queue-reservations",
            "--delta-telemetry",
            "--delta-force-legacy-parent",
            "--delta",
            "0.75",
            "--delta-controller",
            "reduced-round-trip",
            "--delta-controller-batch-size",
            "7",
            "--delta-benchmark-weights",
            "mixed",
            "--delta-benchmark-weight-seed",
            "17",
        ],
        "forced-generic weighted controls must be forwarded canonically",
    )
    require(
        benchmark.pathfinder_args(forced).count("--delta-telemetry") == 1,
        "enabled delta telemetry must be forwarded exactly once",
    )

    for weight_family in ("unit", "all-light", "all-heavy", "mixed"):
        weighted = benchmark.parse_args(
            [
                "input.phys",
                "output.phys",
                "--sssp-engine",
                "delta-step",
                "--delta",
                "1",
                "--delta-benchmark-weights",
                weight_family,
            ]
        )
        require(
            weighted.delta_benchmark_weights == weight_family,
            f"benchmark wrapper did not retain {weight_family!r} weights",
        )

    for invalid in ("aut", "0", "-1", "nan", "inf"):
        require_arg_rejected(benchmark.delta_arg, invalid)
    for invalid in ("0", "-1", "nan", "inf"):
        require_arg_rejected(benchmark.positive_float_arg, invalid)
    require(
        benchmark.nonnegative_int_arg("18446744073709551615")
        == (1 << 64) - 1,
        "benchmark seed parser rejected the maximum uint64 value",
    )
    for invalid in ("-1", "1.5", "seed", "18446744073709551616"):
        require_arg_rejected(benchmark.nonnegative_int_arg, invalid)
    require(
        benchmark.positive_int_arg("2147483647") == (1 << 31) - 1,
        "controller batch parser rejected the maximum 32-bit int",
    )
    for invalid in ("0", "-1", "1.5", "batch", "2147483648"):
        require_arg_rejected(benchmark.positive_int_arg, invalid)

    for delta_controls in (
        ["--delta", "1"],
        ["--delta", "auto", "--delta-multiplier", "2"],
        ["--delta-force-generic"],
        ["--delta-unit-weight-load-elision"],
        ["--delta-host-value-frontier-count"],
        ["--delta-wave-queue-reservations"],
        ["--delta-telemetry"],
        ["--delta-force-legacy-parent"],
        ["--delta-controller", "host-checked"],
        ["--delta-controller", "reduced-round-trip"],
        [
            "--delta-controller",
            "reduced-round-trip",
            "--delta-controller-batch-size",
            "4",
        ],
        ["--delta", "1", "--delta-benchmark-weights", "mixed"],
        [
            "--delta",
            "1",
            "--delta-benchmark-weights",
            "mixed",
            "--delta-benchmark-weight-seed",
            "3",
        ],
    ):
        require_parse_rejected(
            delta_controls,
            "delta-specific controls were accepted without explicit delta-step selection",
        )

    require_parse_rejected(
        ["--sssp-engine", "unit-bfs", "--delta-force-generic"],
        "force-generic was accepted with the unit-BFS engine",
    )
    for non_delta_engine in ("unit-bfs", "bellman-ford"):
        require_parse_rejected(
            ["--sssp-engine", non_delta_engine, "--delta-telemetry"],
            f"delta telemetry was accepted with {non_delta_engine!r}",
        )
    require_parse_rejected(
        ["--sssp-engine", "delta-step", "--delta-multiplier", "2"],
        "delta multiplier was accepted without --delta auto",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta-controller-batch-size",
            "4",
        ],
        "controller batch size was accepted without an explicit controller",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta-controller",
            "host-checked",
            "--delta-controller-batch-size",
            "4",
        ],
        "controller batch size was accepted with host-checked mode",
    )
    reduced_default_batch = benchmark.parse_args(
        [
            "input.phys",
            "output.phys",
            "--sssp-engine",
            "delta-step",
            "--delta-controller",
            "reduced-round-trip",
        ]
    )
    require(
        benchmark.pathfinder_args(reduced_default_batch)[-2:]
        == ["--delta-controller", "reduced-round-trip"],
        "reduced controller mode without a batch override was not forwarded",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta-controller",
            "fast",
        ],
        "unknown controller mode was accepted",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta",
            "2",
            "--delta-multiplier",
            "0.5",
        ],
        "delta multiplier was accepted with a numeric delta",
    )
    require_parse_rejected(
        ["--sssp-engine", "delta-step", "--delta-benchmark-weights", "mixed"],
        "benchmark weights were accepted without a numeric delta",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta",
            "auto",
            "--delta-benchmark-weights",
            "mixed",
        ],
        "benchmark weights were accepted with automatic delta",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta",
            "1",
            "--delta-benchmark-weight-seed",
            "3",
        ],
        "benchmark seed was accepted without a benchmark weight family",
    )
    for weight_family in ("unit", "all-light", "all-heavy"):
        require_parse_rejected(
            [
                "--sssp-engine",
                "delta-step",
                "--delta",
                "1",
                "--delta-benchmark-weights",
                weight_family,
                "--delta-benchmark-weight-seed",
                "3",
            ],
            f"benchmark seed was accepted with {weight_family!r} weights",
        )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--use-delta-step",
        ],
        "engine selector and --use-delta-step were not mutually exclusive",
    )
    require_parse_rejected(
        [
            "--sssp-engine",
            "delta-step",
            "--delta",
            "1",
            "--delta-benchmark-weights",
            "random",
        ],
        "unknown benchmark weight family was accepted",
    )

    print("PathFinder benchmark argument test passed")


if __name__ == "__main__":
    main()
