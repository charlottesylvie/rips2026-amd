#!/usr/bin/env python3
"""Lightweight tests for the Interchange-to-congestion-snapshot driver."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
import pathfinder_snapshot_benchmark as snapshots  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_iteration_parser() -> None:
    require(snapshots.parse_iteration_list("8,1,3") == [1, 3, 8],
            "iteration parser should sort requested snapshots")
    try:
        snapshots.parse_iteration_list("1,1")
    except Exception:
        pass
    else:
        raise AssertionError("iteration parser should reject duplicates")


def test_driver_commands_and_outputs() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        input_phys = root / "tiny_unrouted.phys"
        netlist = root / "tiny.netlist"
        device = root / "xcvu3p.device"
        for path in (input_phys, netlist, device):
            path.write_text("fixture", encoding="utf-8")

        output_root = root / "snapshots"
        benchmark = snapshots.Benchmark("tiny", input_phys, netlist, device)
        args = SimpleNamespace(
            out_dir=output_root,
            interchange_to_csr="fake-converter",
            pathfinder="fake-pathfinder",
            converter_arg=["--full-device"],
            warmup_sssp_engine="delta-step",
            delta=4.0,
            capacity=1,
            present_factor=1.0,
            present_multiplier=2.0,
            history_factor=1.0,
            route_batch_size=256,
            snapshot_iters=[1, 3],
            max_sssp_iters=None,
            net_limit=None,
            overwrite=False,
        )
        commands: list[list[str]] = []
        original_run_command = snapshots.run_command

        def fake_run_command(command: list[str], label: str) -> None:
            commands.append(command)
            if "convert" in label:
                base = Path(command[3])
                base.parent.mkdir(parents=True, exist_ok=True)
                base.write_bytes(b"base")
                Path(str(base) + ".ifmeta.bin").write_bytes(b"metadata")
                return
            snapshot_dir = Path(command[command.index("--snapshot-dir") + 1])
            for iteration in (1, 3):
                snapshot = snapshot_dir / f"iteration-{iteration}.csrbin"
                snapshot.write_bytes(b"snapshot")
                Path(str(snapshot) + ".ifmeta.bin").write_bytes(b"metadata")
                snapshot.with_suffix(".snapshot.json").write_text(
                    "{\"iteration\": "
                    + str(iteration)
                    + ", \"cost_phase\": \""
                    + snapshots.COST_PHASE
                    + "\"}",
                    encoding="utf-8",
                )

        snapshots.run_command = fake_run_command
        try:
            record = snapshots.run_benchmark(benchmark, args, max_iterations=3)
        finally:
            snapshots.run_command = original_run_command

        require(len(commands) == 2, "driver should run converter then PathFinder")
        pathfinder_command = commands[1]
        require("--snapshot-iters" in pathfinder_command and
                pathfinder_command[pathfinder_command.index("--snapshot-iters") + 1] == "1,3",
                "driver should forward requested snapshot iterations")
        require("--max-pathfinder-iters" in pathfinder_command and
                pathfinder_command[pathfinder_command.index("--max-pathfinder-iters") + 1] == "3",
                "driver should cap PathFinder at the final requested iteration")
        require(len(record["snapshots"]) == 2,
                "driver record should list independently usable snapshots")
        require((output_root / "tiny" / "benchmark.snapshot.json").is_file(),
                "driver should write a benchmark manifest")
        require(record["cost_phase"] == snapshots.COST_PHASE,
                "driver manifest should preserve the documented snapshot phase")


def test_safe_names_and_stale_outputs() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name in ("../escape", "nested/name", "nested\\name", "", "manifest.json",
                     "MANIFEST.JSON"):
            try:
                snapshots.validate_benchmark_name(name)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe benchmark name was accepted: {name!r}")

        output_dir = root / "snapshots" / "tiny"
        output_dir.mkdir(parents=True)
        stale_snapshot = output_dir / "iteration-1.csrbin"
        user_file = output_dir / "iteration-notes.csrbin"
        stale_snapshot.write_bytes(b"stale")
        user_file.write_bytes(b"user")
        try:
            snapshots.prepare_output_dir(output_dir, overwrite=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("stale snapshot artifacts should require --overwrite")
        snapshots.prepare_output_dir(output_dir, overwrite=True)
        require(not stale_snapshot.exists(),
                "--overwrite should remove stale generated snapshot artifacts")
        require(user_file.exists(),
                "--overwrite should not remove similarly named user files")

        root_manifest = root / "snapshots" / "manifest.json"
        root_manifest.write_text("old", encoding="utf-8")
        try:
            snapshots.prepare_root_output_dir(root / "snapshots", overwrite=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("root manifest should require --overwrite")
        snapshots.prepare_root_output_dir(root / "snapshots", overwrite=True)
        require(root_manifest.exists(),
                "--overwrite should retain the prior root manifest until an atomic replacement")


if __name__ == "__main__":
    test_iteration_parser()
    test_driver_commands_and_outputs()
    test_safe_names_and_stale_outputs()
    print("PathFinder snapshot benchmark test passed")
