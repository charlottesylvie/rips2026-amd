#!/usr/bin/env python3
"""Build a manifest for the FPGA benchmark archive contents."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchmarkEntry:
    benchmark: str
    netlist: str
    phys: str
    netlist_compressed_bytes: int
    phys_compressed_bytes: int
    netlist_uncompressed_bytes: int
    phys_uncompressed_bytes: int


def gzip_size(path: Path) -> int:
    total = 0
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return total
            total += len(chunk)


def discover(benchmarks_dir: Path) -> list[BenchmarkEntry]:
    netlists = {path.stem: path for path in benchmarks_dir.glob("*.netlist")}
    phys_files = {}
    for path in benchmarks_dir.glob("*_unrouted.phys"):
        phys_files[path.name.removesuffix("_unrouted.phys")] = path

    entries: list[BenchmarkEntry] = []
    for benchmark, netlist in sorted(netlists.items()):
        phys = phys_files.get(benchmark)
        if phys is None:
            continue
        entries.append(
            BenchmarkEntry(
                benchmark=benchmark,
                netlist=str(netlist),
                phys=str(phys),
                netlist_compressed_bytes=netlist.stat().st_size,
                phys_compressed_bytes=phys.stat().st_size,
                netlist_uncompressed_bytes=gzip_size(netlist),
                phys_uncompressed_bytes=gzip_size(phys),
            )
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks-dir",
        default=os.environ.get("BENCHMARK_DIR", "benchmarks"),
        type=Path,
    )
    parser.add_argument("--csv", default="results/benchmark_manifest.csv", type=Path)
    parser.add_argument("--json", default="results/benchmark_manifest.json", type=Path)
    args = parser.parse_args()

    entries = discover(args.benchmarks_dir)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(entries[0]).keys()) if entries else [])
        if entries:
            writer.writeheader()
            writer.writerows(asdict(entry) for entry in entries)

    with args.json.open("w", encoding="utf-8") as stream:
        json.dump([asdict(entry) for entry in entries], stream, indent=2)
        stream.write("\n")

    print(f"wrote {len(entries)} benchmark entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
