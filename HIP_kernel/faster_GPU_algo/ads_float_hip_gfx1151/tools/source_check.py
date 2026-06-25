#!/usr/bin/env python3
"""Fast static checks that do not require a ROCm installation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    root = parser.parse_args().root

    source_paths = [
        root / "common.h",
        root / "csr_graph.h",
        root / "csr_graph.cpp",
        root / "support.h",
        root / "support.cpp",
        root / "wl.h",
        root / "kernel.hip",
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing source files: " + ", ".join(missing))

    combined = "\n".join(path.read_text() for path in source_paths)
    forbidden = {
        r"\bcuda[A-Z_a-z0-9]*": "CUDA runtime symbol",
        r"\bcub::": "CUB symbol",
        r"\b__syncwarp\s*\(": "CUDA warp barrier",
        r"\.target\s+sm_": "PTX target directive",
        r"asm\s+volatile\s*\(\s*\"(?:bfe|bfi|popc|fns|bfind|ld\.global|st\.global|exit)":
            "PTX inline assembly",
    }
    failures: list[str] = []
    for pattern, description in forbidden.items():
        match = re.search(pattern, combined)
        if match:
            failures.append(f"{description}: {match.group(0)!r}")

    makefile = (root / "Makefile").read_text()
    if "gfx1151" not in makefile:
        failures.append("Makefile does not default to gfx1151")
    if makefile.count("-mno-wavefrontsize64") != 1:
        failures.append("Makefile must force wave32 exactly once")
    if "hipLaunchKernelGGL(wl_kernel" not in combined:
        failures.append("host launch for manager kernel is missing")
    if "hipLaunchKernelGGL(sssp_kernel" not in combined:
        failures.append("host launch for worker kernel is missing")
    if "hipExtStreamCreateWithCUMask" not in combined:
        failures.append("CU-mask persistent-stream partitioning is missing")
    if "edge_type_size == 0 ? 1.0f" not in combined:
        failures.append("unweighted .gr input must map to unit weights")
    if "wall_clock64()" not in combined:
        failures.append("gfx11-safe wall clock helper is missing")

    if failures:
        raise SystemExit("source check failed:\n  - " + "\n  - ".join(failures))
    print("source check passed")


if __name__ == "__main__":
    main()
