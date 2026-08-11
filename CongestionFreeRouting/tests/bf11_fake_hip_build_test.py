#!/usr/bin/env python3
"""Compile BF11's HIP translation unit with a host-only HIP surface.

These builds validate C++/macro expansion only. They do not emulate a GPU or
make any claim about kernel, atomic, stream, or HIP Graph runtime semantics.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CongestionFreeRouting/bellman_ford/bf11.cpp"
HIP_REGRESSION = (
    ROOT / "CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp"
)
FAKE_HIP = ROOT / "Routing/tests/fake_hip"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_checked(command: list[str], *, input_text: str | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}):\n"
            f"  {shlex.join(command)}\n{completed.stdout}"
        )


def select_compiler() -> str:
    configured = os.environ.get("BF11_FAKE_HIP_CXX")
    if configured:
        return configured
    compiler = shutil.which("clang++")
    require(
        compiler is not None,
        "BF11 fake-HIP normal-atomic coverage requires clang++; "
        "set BF11_FAKE_HIP_CXX to a compatible compiler",
    )
    return compiler


def require_atomic_load_builtin(compiler: str) -> None:
    # BF11's normal mode selects the intrinsic with this same feature test.
    # Refuse to silently compile its CAS fallback and label that normal mode.
    probe = r"""
#if !defined(__has_builtin)
#error compiler does not provide __has_builtin
#elif !__has_builtin(__hip_atomic_load)
#error compiler does not provide __hip_atomic_load
#endif
int main() { return 0; }
"""
    run_checked(
        [compiler, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
        input_text=probe,
    )


def main() -> None:
    compiler = select_compiler()
    require_atomic_load_builtin(compiler)

    source_text = SOURCE.read_text(encoding="utf-8")
    require(
        "BF11_ENABLE_HIP_GRAPHS" in source_text,
        "BF11 graph code is not guarded by BF11_ENABLE_HIP_GRAPHS",
    )
    require(
        "hipStreamBeginCapture" in source_text and "hipGraphLaunch" in source_text,
        "BF11 fake-HIP graph build would not exercise graph capture/replay code",
    )
    require(
        "hipStreamCaptureModeThreadLocal" in source_text
        and "hipStreamCaptureModeGlobal" not in source_text,
        "BF11 independent worker streams do not use thread-local capture",
    )

    common = [
        compiler,
        "-std=c++17",
        "-O0",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-c",
        "-DBF11_NO_MAIN",
        "-D__HIP_PLATFORM_AMD__=1",
        "-I",
        str(FAKE_HIP),
    ]
    cases = (
        ("atomic-load_graph-off", []),
        ("forced-cas_graph-off", ["-DBF11_FORCE_CAS_ATOMIC_LOAD"]),
        (
            "atomic-load_graph-fake",
            ["-DBF11_ENABLE_HIP_GRAPHS", "-DBF11_FAKE_HIP_ENABLE_GRAPHS"],
        ),
        (
            "forced-cas_graph-fake",
            [
                "-DBF11_FORCE_CAS_ATOMIC_LOAD",
                "-DBF11_ENABLE_HIP_GRAPHS",
                "-DBF11_FAKE_HIP_ENABLE_GRAPHS",
            ],
        ),
    )

    with tempfile.TemporaryDirectory(prefix="bf11-fake-hip-") as directory:
        output_root = Path(directory)
        for name, definitions in cases:
            output = output_root / f"{name}.o"
            run_checked(
                [*common, *definitions, str(SOURCE), "-o", str(output)]
            )
            require(output.is_file(), f"{name} did not produce an object file")
            print(f"BF11 fake-HIP build passed: {name}")

        # Also type-check and link the AMD regression in both macro modes. The
        # fake allocator and kernels remain compile-only, so these executables
        # are intentionally never run.
        integration_common = [
            compiler,
            "-std=c++17",
            "-O0",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-DBF11_NO_MAIN",
            "-D__HIP_PLATFORM_AMD__=1",
            "-I",
            str(FAKE_HIP),
            "-I",
            str(ROOT / "HIP_kernel/bellman_ford/src"),
            "-I",
            str(ROOT / "CongestionFreeRouting/bellman_ford"),
        ]
        integration_cases = (
            ("bounded-regression_graph-off", []),
            (
                "bounded-regression_graph-fake",
                ["-DBF11_ENABLE_HIP_GRAPHS", "-DBF11_FAKE_HIP_ENABLE_GRAPHS"],
            ),
        )
        for name, definitions in integration_cases:
            output = output_root / name
            run_checked(
                [
                    *integration_common,
                    *definitions,
                    str(HIP_REGRESSION),
                    str(SOURCE),
                    "-o",
                    str(output),
                ]
            )
            require(output.is_file(), f"{name} did not link")
            print(f"BF11 fake-HIP integration build passed: {name}")


if __name__ == "__main__":
    main()
