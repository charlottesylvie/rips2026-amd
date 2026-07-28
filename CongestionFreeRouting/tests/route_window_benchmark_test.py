#!/usr/bin/env python3
"""Smoke test for route_window_benchmark.py without a HIP installation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from unittest import mock
from pathlib import Path

import route_window_benchmark as benchmark


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        fake_pathfinder = root / "fake_pathfinder"
        fake_pathfinder.touch()
        graph = root / "design.csrbin"
        metadata = root / "design.csrbin.ifmeta.bin"
        graph.touch()
        metadata.touch()
        output_dir = root / "results"
        def fake_run(command, **_kwargs):
            stats = Path(command[command.index("--route-window-stats-out") + 1])
            if "--route-window" in command:
                records = [
                    {"kind": "window", "execution_path": "unit",
                     "window_rejected_edges": 4,
                     "global_cost_verification_required": True},
                    {"kind": "verification", "execution_path": "unit"},
                ]
            else:
                records = [{"kind": "unbounded_baseline",
                            "execution_path": "unit"}]
            stats.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        argv = [
                "route_window_benchmark.py",
                "--pathfinder", str(fake_pathfinder),
                "--graph", str(graph),
                "--metadata", str(metadata),
                "--output-dir", str(output_dir),
                "--repetitions", "2",
                "--warmups", "0",
                "--",
                "--delta", "1",
                "--parallel-net-workers", "1",
        ]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(benchmark.subprocess, "run", fake_run):
            require(benchmark.main() == 0, "benchmark driver failed")
        summary = json.loads((output_dir / "summary.json").read_text(
            encoding="utf-8"))
        require(len(summary["unbounded"]["samples_seconds"]) == 2,
                "benchmark did not record two unbounded samples")
        require(len(summary["windowed"]["samples_seconds"]) == 2,
                "benchmark did not record two windowed samples")
        telemetry = summary["telemetry_by_mode"]["windowed"]
        require(telemetry[0]["queries_by_kind"]["window"] == 1,
                "benchmark did not retain window telemetry")
        require(telemetry[0]["queries_by_kind"]["verification"] == 1,
                "benchmark did not retain verification telemetry")
        require(telemetry[0]["bounded_attempts"] == 1 and
                telemetry[0]["verification_sssp_queries"] == 1 and
                telemetry[0]["fallback_count"] == 0 and
                telemetry[0]["rejected_edges"] == 4 and
                telemetry[0]["execution_paths"] == {"unit": 2},
                "benchmark did not expose route-window performance counters")
    print("Route-window benchmark test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
