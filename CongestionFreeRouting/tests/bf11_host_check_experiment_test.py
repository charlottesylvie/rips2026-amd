#!/usr/bin/env python3
"""Host-only regression tests for the BF11 experiment runner."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "CongestionFreeRouting/profiling/bf11_host_check_experiment.py"
SPEC = importlib.util.spec_from_file_location("bf11_host_check_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_parse_rejected(arguments: list[str]) -> None:
    try:
        experiment.parse_args(arguments)
    except SystemExit:
        return
    raise AssertionError(f"argument list was not rejected: {arguments}")


def write_log(path: Path, k: int, seconds: float) -> None:
    executed = 100
    noops = 0 if k == 1 else 5
    enqueued = executed + noops
    checks = (enqueued + k - 1) // k
    runtime = {
        "type": "bf11_runtime_stats",
        "schema_version": 2,
        "effective_workers": 4,
        "routing_seconds": seconds - 0.05,
        "iterations_per_host_check": k,
        "target_checks": 25,
        "auto_unbounded_retries": 2,
    }
    control = {
        "iterations_per_host_check": k,
        "host_check_rounds": checks,
        "iteration_status_copies": checks,
        "stream_synchronizations_for_iteration_control": checks,
        "device_iterations_enqueued": enqueued,
        "device_iterations_executed": executed,
        "terminal_noop_iterations": noops,
        "speculative_iterations": noops,
    }
    telemetry = {
        "type": "bf11_telemetry",
        "schema_version": 1,
        "effective_workers": 4,
        "completed_queries": 1000,
        "iteration_control": control,
        "work": {
            "iterations": executed,
            "frontier_vertices_processed": 200,
            "edges_examined": 300,
            "successful_relaxations": 150,
            "touched_vertices": 120,
        },
    }
    routing_time = {
        "type": "bf11_routing_time",
        "schema_version": 1,
        "routing_seconds": seconds,
    }
    path.write_text(
        "noise before JSON\n"
        + json.dumps(runtime)
        + "\n"
        + json.dumps(telemetry)
        + "\n"
        + json.dumps(routing_time)
        + "\n",
        encoding="utf-8",
    )


def route_record(middle: int, named_source: int = 0) -> dict[str, object]:
    return {
        "net": "n0",
        "routed": True,
        "sources": [{"node": 0}],
        "sinks": [{"node": 3, "reached": True, "source": named_source}],
        "edges": [
            {"from": 0, "to": middle},
            {"from": middle, "to": 3},
        ],
    }


def write_trace_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE regions ("
        "name TEXT, category TEXT, start INTEGER, end INTEGER, "
        "duration INTEGER, extdata TEXT)"
    )
    connection.execute(
        "CREATE TABLE kernels ("
        "name TEXT, start INTEGER, end INTEGER, duration INTEGER)"
    )
    markers = [
        ("pathfinder.run", 0, 10_000_000),
        ("bf11.run", 1_000_000, 4_000_000),
        ("bf11.run", 5_000_000, 9_000_000),
    ]
    for marker, start, end in markers:
        connection.execute(
            "INSERT INTO regions VALUES (?, ?, ?, ?, ?, ?)",
            (
                "roctxRange",
                "MARKER_CORE_RANGE_API",
                start,
                end,
                end - start,
                json.dumps({"message": marker}),
            ),
        )
    for name, start, end in (
        ("hipMemcpyAsync", 1_050_000, 1_060_000),
        ("hipMemcpyAsync", 1_250_000, 1_260_000),
        ("hipStreamSynchronize", 1_500_000, 1_550_000),
    ):
        connection.execute(
            "INSERT INTO regions VALUES (?, ?, ?, ?, ?, ?)",
            (name, "HIP_RUNTIME_API_EXT", start, end, end - start, "{}"),
        )
    for kernel in (
        ("host_window_frontier_relax_kernel", 1_000_000, 1_100_000, 100_000),
        ("__amd_rocclr_copyBuffer", 1_200_000, 1_250_000, 50_000),
        ("frontier_relax_kernel", 1_300_000, 1_320_000, 20_000),
    ):
        connection.execute("INSERT INTO kernels VALUES (?, ?, ?, ?)", kernel)
    connection.commit()
    connection.close()


def main() -> None:
    require(experiment.host_check_k("1") == 1, "K=1 parser failed")
    require(experiment.host_check_k("64") == 64, "K=64 parser failed")
    require_parse_rejected(["--ks", "0", "2"])
    require_parse_rejected(["--ks", "1", "65"])
    require_parse_rejected(["--ks", "1", "2", "--repetitions", "4"])

    build_commands: list[list[str]] = []
    real_run = experiment.subprocess.run
    try:
        experiment.subprocess.run = lambda command, check: build_commands.append(
            command
        )
        experiment.build_pathfinder(SimpleNamespace(offload_arch="gfx1151"))
    finally:
        experiment.subprocess.run = real_run
    require(
        build_commands
        == [
            [
                "make",
                "-B",
                "PATHFINDER_BUILD_COMPONENTS=1",
                "PATHFINDER_HIP_FLAGS=-std=c++17 -O3 -x hip --offload-arch=gfx1151",
                "./pathfinder",
            ]
        ],
        "--build did not target the requested gfx1151 production binary",
    )

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        args = experiment.parse_args(
            [
                "--output-dir",
                str(output),
                "--summarize-only",
                "--no-route-check",
                "--ks",
                "1",
                "2",
                "4",
                "8",
            ]
        )
        medians = {1: 1.00, 2: 0.92, 4: 0.80, 8: 0.86}
        for k, median in medians.items():
            for repetition, delta in enumerate((-0.02, -0.01, 0.0, 0.01, 0.02), 1):
                write_log(output / f"k{k}-r{repetition}.log", k, median + delta)
        summary = experiment.summarize(args, {})
        require(summary["best_experimental_k"] == 4, "winner selection failed")
        require(
            abs(summary["best_speedup_fraction"] - 0.20) < 1e-12,
            "speedup calculation failed",
        )
        require(
            summary["results"]["4"]["host_check_rounds"] == 27,
            "control counters were not summarized",
        )
        require(
            (output / "summary.json").is_file()
            and (output / "summary.md").is_file(),
            "summary artifacts were not written",
        )

    baseline = [route_record(1)]
    alternate = [route_record(2)]
    experiment.validate_route_equivalence(baseline, alternate, "K=4")
    tree_root = [route_record(1, named_source=1)]
    experiment.validate_route_equivalence(baseline, tree_root, "tree-root")
    invalid = [route_record(2, named_source=2)]
    invalid[0]["edges"] = [{"from": 0, "to": 1}, {"from": 2, "to": 3}]
    try:
        experiment.validate_route_equivalence(baseline, invalid, "invalid")
    except RuntimeError:
        pass
    else:
        raise AssertionError("disconnected alternate route was accepted")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        database = output / "bf11.rocpd.db"
        write_trace_database(database)
        (output / "bf11.csv").write_text("header\n", encoding="utf-8")
        (output / "bf11.pftrace").write_bytes(b"trace")
        artifacts = experiment.require_profile_artifacts(output)
        require(
            artifacts["database"] == str(database.resolve())
            and not artifacts["missing_requested_formats"],
            "profile artifact discovery failed",
        )
        trace = experiment.analyze_bf11_rocpd(database)
        require(
            trace["scope"]["marker"] == "pathfinder.run"
            and trace["bf11_queries"]["calls"] == 2
            and trace["kernel_dispatches"] == 3
            and trace["hip_api"]["hipMemcpyAsync_calls"] == 2
            and trace["hip_api"]["hipStreamSynchronize_calls"] == 1,
            "RocPD control metrics were not summarized",
        )
        require(
            abs(trace["union_gpu_active_seconds"] - 0.00017) < 1e-12
            and abs(trace["gpu_active_fraction"] - 0.017) < 1e-12
            and trace["relaxation"]["calls"] == 2
            and abs(
                trace["relaxation"]["under_50_microseconds_percent"] - 50.0
            )
            < 1e-12
            and trace["runtime_copy_kernels"]["calls"] == 1
            and trace["dispatch_gaps"]["calls"] == 2,
            "RocPD GPU time, duration, or gap metrics were incorrect",
        )

    print("BF11 host-check experiment test passed")


if __name__ == "__main__":
    main()
