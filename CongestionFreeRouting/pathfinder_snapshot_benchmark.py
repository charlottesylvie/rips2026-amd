#!/usr/bin/env python3
"""Build real-congestion CSR snapshots from FPGA Interchange benchmarks.

This deliberately stops before writing a routed ``.phys`` file.  It converts
each benchmark once, runs the congestion-aware PathFinder loop, and leaves
normal RIPSCSR1/metadata pairs that can be consumed independently by the
one-pass SSSP comparison tool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COST_PHASE = "post_iteration_global_cost"
RESERVED_BENCHMARK_NAMES = {"manifest.json"}


@dataclass(frozen=True)
class Benchmark:
    name: str
    input_phys: Path
    logical_netlist: Path
    device: Path


def validate_benchmark_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or name in {".", ".."}
        or path.is_absolute()
        or path.name != name
        or "/" in name
        or "\\" in name
        or ":" in name
        or name.casefold() in RESERVED_BENCHMARK_NAMES
    ):
        raise ValueError(
            "benchmark name must be a single relative path component: "
            f"{name!r}"
        )


def benchmark_output_dir(out_dir: Path, name: str) -> Path:
    validate_benchmark_name(name)
    root = out_dir.resolve()
    output_dir = (root / name).resolve()
    if output_dir.parent != root:
        raise ValueError(f"benchmark output escapes --out-dir: {name!r}")
    return output_dir


def generated_artifacts(output_dir: Path) -> list[Path]:
    artifacts = [
        output_dir / "base.csrbin",
        output_dir / "base.csrbin.ifmeta.bin",
        output_dir / "benchmark.snapshot.json",
    ]
    iteration_artifact = re.compile(
        r"^iteration-[1-9][0-9]*\.(?:csrbin(?:\.ifmeta\.bin)?|snapshot\.json)$"
    )
    if output_dir.is_dir():
        artifacts.extend(
            path
            for path in output_dir.iterdir()
            if iteration_artifact.fullmatch(path.name) is not None
        )
    return [path for path in artifacts if path.exists() or path.is_symlink()]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = generated_artifacts(output_dir)
    if not artifacts:
        return
    if not overwrite:
        raise FileExistsError(
            f"snapshot artifacts already exist in {output_dir}; "
            "use --overwrite to replace this benchmark's generated files"
        )
    for artifact in artifacts:
        if not artifact.is_file() and not artifact.is_symlink():
            raise RuntimeError(
                f"refusing to replace non-file generated artifact: {artifact}"
            )
        artifact.unlink()


def prepare_root_output_dir(out_dir: Path, overwrite: bool) -> Path:
    root = out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if not manifest.exists() and not manifest.is_symlink():
        return root
    if not overwrite:
        raise FileExistsError(
            f"snapshot root manifest already exists at {manifest}; "
            "use --overwrite to replace this run's generated manifest"
        )
    if not manifest.is_file() and not manifest.is_symlink():
        raise RuntimeError(f"refusing to replace non-file root manifest: {manifest}")
    return root


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def executable_path(default: str, environment_name: str) -> str:
    return os.environ.get(environment_name, default)


def parse_iteration_list(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "snapshot iterations must be a comma-separated list of integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("snapshot iterations must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("snapshot iterations must not repeat")
    return sorted(values)


def resolve_path(value: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve()


def read_manifest(path: Path, default_device: Path | None) -> list[Benchmark]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    entries = data["benchmarks"] if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("benchmark manifest must be a list or {'benchmarks': [...]} object")

    benchmarks: list[Benchmark] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest benchmark {index} is not an object")
        try:
            name = str(entry["name"])
            input_phys = resolve_path(entry["input_phys"], path.parent)
            logical_netlist = resolve_path(entry["logical_netlist"], path.parent)
            device_value = entry.get("device")
            device = (
                resolve_path(device_value, path.parent)
                if device_value is not None
                else default_device
            )
        except KeyError as exc:
            raise ValueError(f"manifest benchmark {index} is missing {exc.args[0]}") from exc
        if device is None:
            raise ValueError(
                f"manifest benchmark {name!r} has no device; pass --device or add device"
            )
        benchmarks.append(Benchmark(name, input_phys, logical_netlist, device))
    return benchmarks


def discover_benchmarks(
    benchmarks_dir: Path, names: list[str], device: Path
) -> list[Benchmark]:
    benchmarks: list[Benchmark] = []
    for name in names:
        benchmarks.append(
            Benchmark(
                name=name,
                input_phys=benchmarks_dir / f"{name}_unrouted.phys",
                logical_netlist=benchmarks_dir / f"{name}.netlist",
                device=device,
            )
        )
    return benchmarks


def validate_benchmark(benchmark: Benchmark) -> None:
    validate_benchmark_name(benchmark.name)
    for path, label in (
        (benchmark.input_phys, "input physical netlist"),
        (benchmark.logical_netlist, "logical netlist"),
        (benchmark.device, "device resources"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{benchmark.name}: missing {label}: {path}")


def command_text(command: list[str]) -> str:
    # Commands contain ordinary file paths and are passed directly to
    # subprocess (never through a shell); this text is only a readable log.
    return " ".join(command)


def run_command(command: list[str], label: str) -> None:
    print(f"[pathfinder-snapshots] {label}: {command_text(command)}", flush=True)
    subprocess.run(command, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FPGA Interchange benchmarks and save PathFinder congestion CSR snapshots."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--benchmarks-dir",
        type=Path,
        help="directory containing <benchmark>_unrouted.phys and <benchmark>.netlist",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="JSON list of {name,input_phys,logical_netlist[,device]} benchmark objects",
    )
    source.add_argument("--input-phys", type=Path, help="single unrouted .phys input")
    parser.add_argument(
        "--benchmarks",
        help="comma-separated benchmark names; required with --benchmarks-dir",
    )
    parser.add_argument("--logical-netlist", type=Path, help="required with --input-phys")
    parser.add_argument("--name", help="output directory name for --input-phys")
    parser.add_argument("--device", type=Path, help="DeviceResources input")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot-iters", type=parse_iteration_list, required=True)
    parser.add_argument("--max-pathfinder-iters", type=int)
    parser.add_argument(
        "--warmup-sssp-engine",
        default="delta-step",
        choices=("delta-step",),
        help="weighted engine used to create congestion snapshots (default: delta-step)",
    )
    parser.add_argument("--interchange-to-csr", default=executable_path(
        "./interchange_to_csr", "INTERCHANGE_TO_CSR"))
    parser.add_argument("--pathfinder", default=executable_path("./pathfinder", "PATHFINDER_BIN"))
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--present-factor", type=float, default=1.0)
    parser.add_argument("--present-multiplier", type=float, default=2.0)
    parser.add_argument("--history-factor", type=float, default=1.0)
    parser.add_argument("--route-batch-size", type=int, default=256)
    parser.add_argument("--max-sssp-iters", type=int)
    parser.add_argument("--net-limit", type=int)
    parser.add_argument(
        "--converter-arg",
        action="append",
        default=[],
        help="extra argument forwarded to interchange_to_csr; may be repeated",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this benchmark's existing generated snapshot artifacts",
    )
    return parser.parse_args(argv)


def select_benchmarks(args: argparse.Namespace) -> list[Benchmark]:
    default_device = args.device.resolve() if args.device is not None else None
    if args.manifest is not None:
        return read_manifest(args.manifest.resolve(), default_device)
    if args.benchmarks_dir is not None:
        if default_device is None:
            raise ValueError("--benchmarks-dir requires --device")
        if not args.benchmarks:
            raise ValueError("--benchmarks-dir requires --benchmarks")
        names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
        if not names:
            raise ValueError("--benchmarks must contain at least one name")
        return discover_benchmarks(args.benchmarks_dir.resolve(), names, default_device)

    if args.logical_netlist is None or default_device is None:
        raise ValueError("--input-phys requires --logical-netlist and --device")
    name = args.name or args.input_phys.stem.removesuffix("_unrouted")
    return [
        Benchmark(
            name=name,
            input_phys=args.input_phys.resolve(),
            logical_netlist=args.logical_netlist.resolve(),
            device=default_device,
        )
    ]


def expected_snapshot_paths(output_dir: Path, iterations: list[int]) -> list[Path]:
    return [output_dir / f"iteration-{iteration}.csrbin" for iteration in iterations]


def validate_snapshot_artifacts(snapshot: Path, iteration: int) -> None:
    sidecar = Path(str(snapshot) + ".ifmeta.bin")
    manifest = snapshot.with_suffix(".snapshot.json")
    if not snapshot.is_file():
        raise RuntimeError(f"PathFinder did not write snapshot: {snapshot}")
    if not sidecar.is_file():
        raise RuntimeError(f"PathFinder did not write snapshot metadata: {sidecar}")
    if not manifest.is_file():
        raise RuntimeError(f"PathFinder did not write snapshot manifest: {manifest}")
    try:
        manifest_data: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid snapshot manifest {manifest}: {exc}") from exc
    if not isinstance(manifest_data, dict):
        raise RuntimeError(f"snapshot manifest is not an object: {manifest}")
    if (
        manifest_data.get("iteration") != iteration
        or manifest_data.get("cost_phase") != COST_PHASE
    ):
        raise RuntimeError(
            f"snapshot manifest has unexpected iteration/cost phase: {manifest}"
        )


def run_benchmark(
    benchmark: Benchmark, args: argparse.Namespace, max_iterations: int
) -> dict[str, Any]:
    validate_benchmark(benchmark)
    output_dir = benchmark_output_dir(args.out_dir, benchmark.name)
    prepare_output_dir(output_dir, args.overwrite)
    base_csr = output_dir / "base.csrbin"
    metadata = Path(str(base_csr) + ".ifmeta.bin")

    converter_command = [
        args.interchange_to_csr,
        str(benchmark.input_phys),
        str(benchmark.logical_netlist),
        str(base_csr),
        "--device",
        str(benchmark.device),
        "--metadata",
        str(metadata),
        *args.converter_arg,
    ]
    run_command(converter_command, f"{benchmark.name}: convert FPGAIF to CSR")

    pathfinder_command = [
        args.pathfinder,
        str(base_csr),
        str(metadata),
        "--sssp-engine",
        args.warmup_sssp_engine,
        "--delta",
        str(args.delta),
        "--capacity",
        str(args.capacity),
        "--present-factor",
        str(args.present_factor),
        "--present-multiplier",
        str(args.present_multiplier),
        "--history-factor",
        str(args.history_factor),
        "--route-batch-size",
        str(args.route_batch_size),
        "--max-pathfinder-iters",
        str(max_iterations),
        "--snapshot-iters",
        ",".join(str(iteration) for iteration in args.snapshot_iters),
        "--snapshot-dir",
        str(output_dir),
    ]
    if args.max_sssp_iters is not None:
        pathfinder_command.extend(["--max-sssp-iters", str(args.max_sssp_iters)])
    if args.net_limit is not None:
        pathfinder_command.extend(["--net-limit", str(args.net_limit)])
    run_command(pathfinder_command, f"{benchmark.name}: run congestion PathFinder")

    snapshots = expected_snapshot_paths(output_dir, args.snapshot_iters)
    missing = [path for path in snapshots if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{benchmark.name}: PathFinder stopped before requested snapshots: "
            + ", ".join(str(path) for path in missing)
        )
    for iteration, snapshot in zip(args.snapshot_iters, snapshots):
        try:
            validate_snapshot_artifacts(snapshot, iteration)
        except RuntimeError as exc:
            raise RuntimeError(f"{benchmark.name}: {exc}") from exc

    record: dict[str, Any] = {
        "name": benchmark.name,
        "input_phys": str(benchmark.input_phys),
        "logical_netlist": str(benchmark.logical_netlist),
        "device": str(benchmark.device),
        "base_csr": str(base_csr),
        "base_metadata": str(metadata),
        "snapshot_iterations": args.snapshot_iters,
        "snapshots": [str(path) for path in snapshots],
        "cost_phase": COST_PHASE,
        "pathfinder_command": pathfinder_command,
    }
    write_json_atomic(output_dir / "benchmark.snapshot.json", record)
    return record


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.max_pathfinder_iters is not None and args.max_pathfinder_iters <= 0:
        raise ValueError("--max-pathfinder-iters must be positive")
    max_iterations = args.max_pathfinder_iters or max(args.snapshot_iters)
    if max_iterations < max(args.snapshot_iters):
        raise ValueError("--max-pathfinder-iters is smaller than a requested snapshot")
    if args.route_batch_size <= 0:
        raise ValueError("--route-batch-size must be positive")

    benchmarks = select_benchmarks(args)
    if not benchmarks:
        raise ValueError("no benchmarks selected")
    names = [benchmark.name for benchmark in benchmarks]
    if len(set(names)) != len(names):
        raise ValueError("benchmark names must be unique")
    output_dirs = [benchmark_output_dir(args.out_dir, name) for name in names]
    if len({str(path).casefold() for path in output_dirs}) != len(output_dirs):
        raise ValueError("benchmark names resolve to duplicate output directories")
    # Do this before allowing --overwrite to replace the aggregate manifest,
    # so a misspelled benchmark input cannot disturb a prior successful run.
    for benchmark in benchmarks:
        validate_benchmark(benchmark)

    output_root = prepare_root_output_dir(args.out_dir, args.overwrite)
    records = [run_benchmark(benchmark, args, max_iterations) for benchmark in benchmarks]
    root_manifest = {
        "snapshot_iterations": args.snapshot_iters,
        "max_pathfinder_iterations": max_iterations,
        "cost_phase": COST_PHASE,
        "benchmarks": records,
    }
    write_json_atomic(output_root / "manifest.json", root_manifest)
    print(
        f"[pathfinder-snapshots] wrote {len(records)} benchmark snapshot set(s) to "
        f"{output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"[pathfinder-snapshots] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
