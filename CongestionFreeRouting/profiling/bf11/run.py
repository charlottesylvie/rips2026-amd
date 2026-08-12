#!/usr/bin/env python3
"""Reproducible, sudo-free BF11 profiling orchestrator for AMD gfx1151.

Each collection kind launches a fresh process. Commands are journaled before
execution, and the run refuses a dirty worktree unless --allow-dirty is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_BIN = HERE / "bin"
GROUPS_FILE = HERE / "counter_groups.json"
DIRECT_BUILD_SOURCES = (
    "CongestionFreeRouting/pathfinder.cpp",
    "CongestionFreeRouting/bellman_ford/bf10.cpp",
    "CongestionFreeRouting/bellman_ford/bf11.cpp",
    "CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp",
    "CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp",
)
DIRECT_BUILD_INCLUDES = (
    "HIP_kernel/bellman_ford/src",
    "CongestionFreeRouting/bellman_ford",
    "CongestionFreeRouting/delta_stepping",
    "CongestionFreeRouting/unit_bfs",
)

MATRIX = (
    ("w3-k1-graph-off", 3, 1, "off"),
    ("w3-k8-graph-off", 3, 8, "off"),
    ("w3-k8-graph-on", 3, 8, "on"),
    ("w1-k8-graph-off", 1, 8, "off"),
    ("w1-k1-graph-auto", 1, 1, "auto"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Run:
    def __init__(self, output: Path, dry_run: bool) -> None:
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.journal = self.output / "commands.jsonl"

    def command(
        self,
        argv: Iterable[os.PathLike[str] | str],
        *,
        cwd: Path = ROOT,
        capture: Path | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args = [os.fspath(item) for item in argv]
        record = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cwd": os.fspath(cwd.resolve()),
            "argv": args,
            "shell_command": shlex.join(args),
            "capture": os.fspath(capture.resolve()) if capture else None,
        }
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print("+", record["shell_command"], flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(args, 0, "", "")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = subprocess.run(
            args,
            cwd=cwd,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if capture:
            capture.parent.mkdir(parents=True, exist_ok=True)
            capture.write_text(result.stdout, encoding="utf-8")
        else:
            sys.stdout.write(result.stdout)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"command failed with exit {result.returncode}: {shlex.join(args)}"
            )
        return result


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip())
    return result.stdout


def require_clean(allow_dirty: bool, output: Path) -> None:
    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    if status and not allow_dirty:
        raise RuntimeError(
            "dirty worktree refused; commit/stash changes or pass --allow-dirty"
        )
    if not status:
        return
    archive = output / "provenance" / "dirty"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "status.txt").write_text(status, encoding="utf-8")
    (archive / "tracked.patch").write_text(
        git_text("diff", "--binary", "HEAD"), encoding="utf-8"
    )
    untracked = [
        ROOT / line
        for line in git_text("ls-files", "--others", "--exclude-standard").splitlines()
        if line.startswith("CongestionFreeRouting/profiling/")
    ]
    if untracked:
        with tarfile.open(archive / "untracked-bf11-profiling.tar.gz", "w:gz") as tar:
            for path in untracked:
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(ROOT))


def capture_provenance(run: Run, args: argparse.Namespace) -> None:
    directory = run.output / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    status = git_text("status", "--branch", "--untracked-files=all")
    (directory / "git-status.txt").write_text(status, encoding="utf-8")
    (directory / "dirty.patch").write_text(
        git_text("diff", "--binary", "HEAD"), encoding="utf-8"
    )
    profiling_settings = {
        "matrix": MATRIX,
        "counter_collection": {"workers": 1, "segment_rounds": 8,
                               "hip_graph": "off", "telemetry": False},
        "bounding_box": {"enabled": True, "margin_x": 2, "margin_y": 14,
                         "unbounded_fallback": True},
        "reset": {"adaptive_dense_threshold": 0.25},
        "telemetry_action": {"enabled": True, "per_query": True},
        "acceptance_timing": {"telemetry": False, "profiler": None},
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": git_text("rev-parse", "HEAD").strip(),
        "branch": git_text("branch", "--show-current").strip(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "configuration": vars(args),
        "profiling_settings": profiling_settings,
    }
    artifacts = {}
    for label in ("csr", "metadata", "manifest"):
        value = getattr(args, label, None)
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                raise RuntimeError(f"missing {label}: {path}")
            artifacts[label] = {"path": os.fspath(path), "sha256": sha256(path)}
    metadata["artifacts"] = artifacts
    metadata["configuration_fingerprint"] = fingerprint(
        {"artifacts": artifacts, "settings": profiling_settings,
         "device": args.device}
    )
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=os.fspath) + "\n",
        encoding="utf-8",
    )

    probes = (
        (["rocminfo"], "rocminfo.txt"),
        (["hipcc", "--version"], "hipcc-version.txt"),
        (["rocprofv3", "--version"], "rocprofv3-version.txt"),
        (["rocprofv3", "--help"], "rocprofv3-help.txt"),
        (["rocprofv3", "--list-avail"], "rocprofv3-metrics.txt"),
        (["rocprofv3-avail", "--help"], "rocprofv3-avail-help.txt"),
        (["rocprofv3-avail", "list", "--agent"], "available-agents.txt"),
        (["rocprofv3-avail", "list", "--pmc", "-d", str(args.device)],
         "available-counters.txt"),
        (["rocprofv3-avail", "info", "--pmc", "-d", str(args.device)],
         "counter-definitions.txt"),
        (["rocprof-compute", "--version"], "rocprof-compute-version.txt"),
        (["rocprof-compute", "profile", "--help"], "rocprof-compute-profile-help.txt"),
        (["rocprof-compute", "profile", "--list-available-metrics"],
         "rocprof-compute-available-metrics.txt"),
        (["rocprof-compute", "profile", "--list-blocks"],
         "rocprof-compute-block-catalog.txt"),
        (["rocprof-compute", "profile", "--list-sets"],
         "rocprof-compute-set-catalog.txt"),
        (["rocprof-compute", "analyze", "--help"], "rocprof-compute-analyze-help.txt"),
        (["rocprof-compute", "analyze", "--list-metrics"], "rocprof-compute-metric-catalog.txt"),
        (["pkg-config", "--modversion", "rocprofiler-sdk"], "rocprofiler-sdk-version.txt"),
        (["rocprofiler-sdk-avail", "--version"], "rocprofiler-sdk-avail-version.txt"),
        (["rocprof-trace-decoder", "--version"], "trace-decoder-version.txt"),
        (["rocm-smi", "--showclocks", "--showtemp", "--showmeminfo", "vram", "--showpids"], "gpu-state.txt"),
        (["ps", "-eo", "pid,user,comm,args"], "processes.txt"),
    )
    for command, name in probes:
        if shutil.which(command[0]):
            run.command(command, capture=directory / name, check=False)


def build(run: Run, args: argparse.Namespace) -> dict[str, Path]:
    if args.build_mode == "make":
        targets = [
            "bf11-profile-timing", "bf11-profile-trace",
            "bf11-profile-thread-trace",
        ]
        run.command(
            ["make", "-B", "-n", *targets],
            capture=run.output / "provenance" / "build-commands.txt",
        )
        run.command(["make", "-B", *targets])
        bins = {
            "timing": DEFAULT_BIN / "pathfinder-bf11-timing",
            "trace": DEFAULT_BIN / "pathfinder-bf11-roctx",
            "thread_trace": DEFAULT_BIN / "pathfinder-bf11-thread-trace",
        }
    else:
        bin_dir = run.output / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bins = {
            "timing": bin_dir / "pathfinder-bf11-timing",
            "trace": bin_dir / "pathfinder-bf11-roctx",
            "thread_trace": bin_dir / "pathfinder-bf11-thread-trace",
        }
        common = [
            "hipcc", "-std=c++17", "-O3", "-x", "hip",
            "-DBF10_NO_MAIN", "-DBF11_NO_MAIN", "-DBF11_ENABLE_HIP_GRAPHS",
        ]
        includes = [item for path in DIRECT_BUILD_INCLUDES for item in ("-I", path)]
        sources = list(DIRECT_BUILD_SOURCES)
        run.command(
            [*common, *includes, *sources, "-pthread", "-o", bins["timing"]],
            capture=run.output / "provenance" / "direct-build-timing.log",
        )
        diagnostics = [
            "-DPATHFINDER_ENABLE_BF11_DIAGNOSTICS",
            "-DPATHFINDER_ENABLE_ROCTX",
        ]
        run.command(
            [*common, *diagnostics, *includes, *sources, "-pthread", "-o",
             bins["trace"], "-lrocprofiler-sdk-roctx"],
            capture=run.output / "provenance" / "direct-build-trace.log",
        )
        run.command(
            [*common, "-gline-tables-only", *diagnostics, *includes, *sources,
             "-pthread", "-o", bins["thread_trace"],
             "-lrocprofiler-sdk-roctx"],
            capture=run.output / "provenance" / "direct-build-thread-trace.log",
        )
    hashes = {}
    for name, path in bins.items():
        if not run.dry_run and not path.is_file():
            raise RuntimeError(f"build did not produce {path}")
        if path.is_file():
            hashes[name] = {"path": os.fspath(path.resolve()), "sha256": sha256(path)}
    (run.output / "provenance" / "binary-hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bins


def app_command(
    args: argparse.Namespace,
    binary: Path,
    workers: int,
    rounds: int,
    graph: str,
    routes: Path,
    *,
    telemetry: bool = False,
) -> list[str]:
    command = [
        os.fspath(binary), os.fspath(Path(args.csr).resolve()),
        os.fspath(Path(args.metadata).resolve()), "--allow-unrouted",
        "--sssp-engine", "bf11", "--parallel-net-workers", str(workers),
        "--bf11-segment-rounds", str(rounds), "--bf11-hip-graph", graph,
        "--net-manifest", os.fspath(Path(args.manifest).resolve()),
        "--routes-out", os.fspath(routes.resolve()),
    ]
    if telemetry:
        command.append("--bf11-query-telemetry")
    return command


def run_matrix(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    for name, workers, rounds, graph in MATRIX:
        directory = run.output / "timing" / name
        directory.mkdir(parents=True, exist_ok=True)
        for repetition in range(args.repetitions):
            command = app_command(
                args, bins["timing"], workers, rounds, graph,
                directory / f"routes-{repetition}.jsonl"
            )
            run.command(command, capture=directory / f"run-{repetition}.log")


def run_correctness(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    directory = run.output / "correctness"
    directory.mkdir(parents=True, exist_ok=True)
    routes = []
    for name, rounds, graph in (("compat", 1, "auto"), ("segmented", 8, "off")):
        path = directory / f"routes-{name}.jsonl"
        routes.append(path)
        run.command(
            app_command(args, bins["timing"], 1, rounds, graph, path),
            capture=directory / f"{name}.log",
        )
    if not run.dry_run:
        hashes = [sha256(path) for path in routes]
        if hashes[0] != hashes[1]:
            raise RuntimeError(
                "BF11 route-equivalence failed between K=1 Graph-auto and K=8 Graph-off"
            )
        (directory / "equivalence.json").write_text(
            json.dumps({"sha256": hashes[0], "equivalent": True}, indent=2) + "\n",
            encoding="utf-8",
        )


def run_telemetry(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    directory = run.output / "telemetry"
    directory.mkdir(parents=True, exist_ok=True)
    command = app_command(
        args, bins["trace"], args.workers, args.rounds, args.graph,
        directory / "routes.jsonl", telemetry=True
    )
    run.command(command, capture=directory / "run.log")


def run_trace(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    directory = run.output / "runtime-trace"
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "rocprofv3", "--runtime-trace", "--marker-trace", "--stats",
        "--group-by-queue", "--output-format", "csv", "rocpd",
        "--output-directory", os.fspath(directory), "--",
        *app_command(
            args, bins["trace"], args.workers, args.rounds, args.graph,
            directory / "routes.jsonl"
        ),
    ]
    run.command(command, capture=directory / "profiler.log")


def compatible_groups(run: Run, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    plan = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    directory = run.output / "counter-compatibility"
    directory.mkdir(parents=True, exist_ok=True)
    proposed = dict(plan["groups"])
    catalog_path = run.output / "provenance" / "available-counters.txt"
    if catalog_path.is_file():
        catalog = catalog_path.read_text(encoding="utf-8", errors="replace")
        tokens = sorted(set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", catalog)))
        for category, patterns in plan["optional_name_patterns"].items():
            matches = [
                token for token in tokens
                if any(pattern.upper() in token.upper() for pattern in patterns)
            ][:16]
            for counter in matches:
                proposed[f"catalog_{category}_{counter}"] = [counter]
    accepted = []
    for name, counters in proposed.items():
        result = run.command(
            ["rocprofv3-avail", "pmc-check", "-d", str(args.device), *counters],
            capture=directory / f"{name}.txt", check=False
        )
        if result.returncode == 0:
            accepted.append((name, counters))
    (directory / "accepted.json").write_text(
        json.dumps(dict(accepted), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return accepted


def run_counters(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    groups = compatible_groups(run, args)
    if not groups and not run.dry_run:
        raise RuntimeError("no proposed counter group passed pmc-check")
    plan = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    for family, kernel_regex in plan["kernel_families"].items():
        for group, counters in groups:
            directory = run.output / "counters" / family / group
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                "rocprofv3", "--pmc", *counters, "--kernel-include-regex",
                f".*({kernel_regex}).*", "--output-format", "csv",
                "--output-directory", os.fspath(directory), "--",
                *app_command(
                    args, bins["timing"], 1, 8, "off",
                    directory / "routes.jsonl"
                ),
            ]
            run.command(command, capture=directory / "profiler.log")


def run_compute(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    # rocprof-compute report-block identifiers vary by installed release. Its
    # catalog/help is captured first and the default full collection is used;
    # the analyzer selects System SOL, Memory Chart, WGP/GL0/GL1/GL2/GCEA,
    # command-processor, and GRBM metrics by their captured names.
    for family, regex in json.loads(GROUPS_FILE.read_text())["kernel_families"].items():
        directory = run.output / "rocprof-compute" / family
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            "rocprof-compute", "profile", "--output-directory",
            os.fspath(directory), "-k", regex, "--",
            *app_command(
                args, bins["timing"], 1, 8, "off",
                directory / "routes.jsonl"
            ),
        ]
        run.command(command, capture=directory / "profiler.log")
        # Use the installed release's default report set. This deliberately
        # avoids fixed report/block identifiers, which changed between the
        # Omniperf and rocprof-compute releases used on gfx1151 systems.
        run.command(
            ["rocprof-compute", "analyze", "--path", os.fspath(directory)],
            capture=directory / "analysis.txt",
        )


def run_thread_trace(run: Run, args: argparse.Namespace, bins: dict[str, Path]) -> None:
    directory = run.output / "thread-trace"
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "rocprofv3", "--att", "--att-serialize-all", "false",
        "--att-consecutive-kernels", str(args.att_dispatches),
        "--kernel-include-regex", ".*segmented_frontier_relax_kernel.*",
        "--output-format", "csv", "--output-directory", os.fspath(directory),
        "--", *app_command(
            args, bins["thread_trace"], 1, 8, "off",
            directory / "routes.jsonl"
        ),
    ]
    run.command(command, capture=directory / "profiler.log")


def run_streaming(run: Run, args: argparse.Namespace) -> None:
    directory = run.output / "streaming-reference"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "bf11-streaming-reference"
    build_command = [
        "hipcc", "-std=c++17", "-O3", "-x", "hip",
        os.fspath(HERE / "streaming_memory.cpp"), "-o", os.fspath(binary)
    ]
    run.command(build_command, capture=directory / "build.log")
    reference = [
        os.fspath(binary), "--bytes", str(args.stream_bytes),
        "--repetitions", str(args.stream_repetitions)
    ]
    run.command(reference, capture=directory / "reference.jsonl")
    groups = compatible_groups(run, args)
    for name, counters in groups:
        output = directory / "counters" / name
        output.mkdir(parents=True, exist_ok=True)
        run.command(
            ["rocprofv3", "--pmc", *counters, "--kernel-include-regex",
             ".*streaming_copy.*", "--output-format", "csv",
             "--output-directory", os.fspath(output), "--", *reference],
            capture=output / "profiler.log"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "build", "correctness", "timing", "telemetry", "trace", "counters", "compute",
        "thread-trace", "streaming", "all"
    ))
    parser.add_argument("--csr", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--build-mode", choices=("make", "direct"), default="make",
        help="use direct hipcc commands on hosts without make",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--graph", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--att-dispatches", type=int, default=1)
    parser.add_argument("--stream-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--stream-repetitions", type=int, default=32)
    args = parser.parse_args()
    if args.device < 0 or args.workers <= 0 or args.repetitions <= 0:
        parser.error("device must be nonnegative; workers/repetitions must be positive")
    if args.action != "build" and not all((args.csr, args.metadata, args.manifest)):
        parser.error("collection actions require --csr, --metadata, and --manifest")
    return args


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    require_clean(args.allow_dirty, output)
    run = Run(output, args.dry_run)
    capture_provenance(run, args)
    bins = build(run, args)
    if args.action in ("correctness", "all"):
        run_correctness(run, args, bins)
    if args.action in ("timing", "all"):
        run_matrix(run, args, bins)
    if args.action in ("telemetry", "all"):
        run_telemetry(run, args, bins)
    if args.action in ("trace", "all"):
        run_trace(run, args, bins)
    if args.action in ("counters", "all"):
        run_counters(run, args, bins)
    if args.action in ("compute", "all"):
        run_compute(run, args, bins)
    if args.action in ("thread-trace", "all"):
        run_thread_trace(run, args, bins)
    if args.action in ("streaming", "all"):
        run_streaming(run, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # concise CLI boundary
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
