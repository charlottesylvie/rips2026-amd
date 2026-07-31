#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_fixture_file(root: Path, relative_path: str, timestamp: int) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture\n", encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    makefile = repository_root / "Makefile"

    with tempfile.TemporaryDirectory() as directory:
        fixture_root = Path(directory)
        shutil.copy2(makefile, fixture_root / "Makefile")

        # Materialize only the files Make needs to resolve the benchmark and
        # in-tree PathFinder rules. The dry run must not need ROCm, generated
        # schemas, benchmark downloads, or executable fixture contents.
        old = 1_700_000_000
        current = old + 20
        for relative_path in (
            "logicnets_jscl_unrouted.phys",
            "logicnets_jscl.netlist",
            "xcvu3p.full-poc-base-wire.devicegraph",
            "CongestionFreeRouting/pathfinder_router.cpp",
            "CongestionFreeRouting/pathfinder.cpp",
            "CongestionFreeRouting/bellman_ford/bf10.cpp",
            "CongestionFreeRouting/bellman_ford/bf11.cpp",
            "CongestionFreeRouting/bellman_ford/bf11.hpp",
            "CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp",
            "CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp",
            "CongestionFreeRouting/pathfinder.hpp",
            "CongestionFreeRouting/interchange/import_policy.hpp",
            "CongestionFreeRouting/interchange/routing_csr_sidecars.hpp",
            "CongestionFreeRouting/profiling/roctx_ranges.hpp",
            "CongestionFreeRouting/sssp_query_capacity.hpp",
            "HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp",
            "HIP_kernel/minplus_mm/src/minplus_sparse_hip.hpp",
        ):
            write_fixture_file(fixture_root, relative_path, old)

        for relative_path in (
            "PathFinderFile",
            "interchange_to_csr",
            "pathfinder",
            "routes_to_phys",
        ):
            write_fixture_file(fixture_root, relative_path, current)

        # Exercise both implementation and cross-directory transitive headers.
        # Each case independently makes only that input newer than ./pathfinder.
        rebuild_inputs = (
            "CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp",
            "CongestionFreeRouting/bellman_ford/bf11.cpp",
            "CongestionFreeRouting/interchange/routing_csr_sidecars.hpp",
            "CongestionFreeRouting/interchange/import_policy.hpp",
            "CongestionFreeRouting/sssp_query_capacity.hpp",
            "HIP_kernel/minplus_mm/src/minplus_sparse_hip.hpp",
        )
        for changed_relative_path in rebuild_inputs:
            for relative_path in rebuild_inputs:
                path = fixture_root / relative_path
                os.utime(path, (old, old))
            changed_path = fixture_root / changed_relative_path
            os.utime(changed_path, (current + 20, current + 20))

            completed = subprocess.run(
                [
                    "make",
                    "--dry-run",
                    "PATHFINDER_HIPCC=pathfinder-hipcc-fixture",
                    "logicnets_jscl_PathFinderFile.phys",
                ],
                cwd=fixture_root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            context = f" while checking {changed_relative_path}"
            require(
                completed.returncode == 0,
                "isolated PathFinder make dry run failed"
                + context
                + ":\n"
                + completed.stdout,
            )

            build_position = completed.stdout.find("pathfinder-hipcc-fixture")
            route_position = completed.stdout.find(
                "./PathFinderFile logicnets_jscl_unrouted.phys"
            )
            require(
                build_position >= 0,
                "a newer PathFinder input did not schedule a rebuild" + context,
            )
            require(
                route_position >= 0,
                "the PathFinder benchmark recipe was not scheduled" + context,
            )
            require(
                build_position < route_position,
                "the stale ./pathfinder rebuild was not ordered before routing"
                + context,
            )
            require(
                "interchange_to_csr.cpp" not in completed.stdout
                and "routes_to_phys.cpp" not in completed.stdout,
                "the dependency check unexpectedly required schema-backed rebuilds"
                + context,
            )

    print("PathFinder Make dependency regression test passed")


if __name__ == "__main__":
    main()
