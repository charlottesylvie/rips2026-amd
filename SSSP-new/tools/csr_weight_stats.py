#!/usr/bin/env python3
"""Report min, max, and mean edge weights from a RIPS CSR binary file.

Supports both CSR layouts used in this repo:
  - RIPSCSR1: incoming CSR from Routing/interchange_to_csr.cpp
  - RIPSOCS1: outgoing CSR from SSSP-new/src/csr_outgoing.cpp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


INCOMING_MAGIC = b"RIPSCSR1"
OUTGOING_MAGIC = b"RIPSOCS1"

INCOMING_ORIENTATION = 1
OUTGOING_ORIENTATION = 2


@dataclass
class CsrHeader:
    path: Path
    magic: bytes
    version: int
    orientation: int
    rows: int
    cols: int
    nnz: int
    rowptr_count: int
    values_count: int
    values_offset: int

    @property
    def format_name(self) -> str:
        if self.magic == INCOMING_MAGIC:
            return "RIPSCSR1"
        if self.magic == OUTGOING_MAGIC:
            return "RIPSOCS1"
        return self.magic.decode("ascii", errors="replace")

    @property
    def orientation_name(self) -> str:
        if self.orientation == INCOMING_ORIENTATION:
            return "incoming"
        if self.orientation == OUTGOING_ORIENTATION:
            return "outgoing"
        return f"unknown({self.orientation})"


@dataclass
class WeightStats:
    header: CsrHeader
    min_weight: float | None
    max_weight: float | None
    mean_weight: float | None
    nonfinite_count: int
    all_weights_equal_1: bool
    not_one_count: int


def read_exact(f: BinaryIO, size: int, name: str) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"unexpected EOF while reading {name}")
    return data


def read_u64(f: BinaryIO, name: str) -> int:
    return struct.unpack("<Q", read_exact(f, 8, name))[0]


def checked_skip(f: BinaryIO, byte_count: int, name: str) -> None:
    if byte_count < 0:
        raise ValueError(f"negative byte count while skipping {name}")
    f.seek(byte_count, os.SEEK_CUR)


def parse_header(path: Path) -> CsrHeader:
    with path.open("rb") as f:
        magic = read_exact(f, 8, "CSR magic")
        if magic not in (INCOMING_MAGIC, OUTGOING_MAGIC):
            raise ValueError(
                f"{path}: unrecognized CSR magic {magic!r}; expected RIPSCSR1 or RIPSOCS1"
            )

        version = read_u64(f, "CSR format version")
        orientation = read_u64(f, "CSR orientation")

        if magic == INCOMING_MAGIC:
            rows = read_u64(f, "incoming CSR row count")
            cols = read_u64(f, "incoming CSR column count")
            read_u64(f, "incoming CSR declared edge count")
            read_u64(f, "incoming CSR loaded edge count")
            nnz = read_u64(f, "incoming CSR nnz")
            rowptr_count = read_u64(f, "incoming CSR rowptr count")
            colind_count = read_u64(f, "incoming CSR colind count")
            values_count = read_u64(f, "incoming CSR values count")

            if version != 1:
                raise ValueError(f"{path}: unsupported incoming CSR version {version}")
            if orientation != INCOMING_ORIENTATION:
                raise ValueError(f"{path}: unsupported incoming CSR orientation {orientation}")
            if rowptr_count != rows + 1 or colind_count != nnz or values_count != nnz:
                raise ValueError(f"{path}: inconsistent incoming CSR counts")

            checked_skip(f, rowptr_count * 8, "incoming CSR rowptr")
            checked_skip(f, colind_count * 4, "incoming CSR colind")
            values_offset = f.tell()
        else:
            rows = read_u64(f, "outgoing CSR row count")
            cols = read_u64(f, "outgoing CSR column count")
            nnz = read_u64(f, "outgoing CSR nnz")
            rowptr_count = read_u64(f, "outgoing CSR rowptr count")
            degree_count = read_u64(f, "outgoing CSR degree count")
            to_count = read_u64(f, "outgoing CSR destination count")
            values_count = read_u64(f, "outgoing CSR values count")
            edge_id_count = read_u64(f, "outgoing CSR edge-id count")

            if version < 3:
                raise ValueError(f"{path}: unsupported outgoing CSR version {version}")
            if orientation != OUTGOING_ORIENTATION:
                raise ValueError(f"{path}: unsupported outgoing CSR orientation {orientation}")
            if (
                rowptr_count != rows + 1
                or degree_count != rows
                or to_count != nnz
                or values_count != nnz
                or edge_id_count != nnz
            ):
                raise ValueError(f"{path}: inconsistent outgoing CSR counts")

            checked_skip(f, rowptr_count * 8, "outgoing CSR rowptr")
            checked_skip(f, degree_count * 4, "outgoing CSR degree")
            checked_skip(f, to_count * 4, "outgoing CSR destinations")
            values_offset = f.tell()

    if rows == 0 or rows != cols:
        raise ValueError(f"{path}: CSR graph must be nonempty and square")

    file_size = path.stat().st_size
    values_end = values_offset + values_count * 4
    if values_end > file_size:
        raise ValueError(f"{path}: values array extends past end of file")

    return CsrHeader(
        path=path,
        magic=magic,
        version=version,
        orientation=orientation,
        rows=rows,
        cols=cols,
        nnz=nnz,
        rowptr_count=rowptr_count,
        values_count=values_count,
        values_offset=values_offset,
    )


def iter_float32_values(f: BinaryIO, count: int, chunk_values: int = 1 << 20) -> Iterable[float]:
    remaining = count
    while remaining:
        take = min(remaining, chunk_values)
        data = read_exact(f, take * 4, "CSR values")
        for (value,) in struct.iter_unpack("<f", data):
            yield value
        remaining -= take


def compute_weight_stats(path: Path, one_tol: float) -> WeightStats:
    header = parse_header(path)
    min_weight: float | None = None
    max_weight: float | None = None
    total = 0.0
    finite_count = 0
    nonfinite_count = 0
    not_one_count = 0

    with path.open("rb") as f:
        f.seek(header.values_offset)
        for value in iter_float32_values(f, header.values_count):
            if not math.isfinite(value):
                nonfinite_count += 1
                continue
            min_weight = value if min_weight is None else min(min_weight, value)
            max_weight = value if max_weight is None else max(max_weight, value)
            total += value
            finite_count += 1
            if abs(value - 1.0) > one_tol:
                not_one_count += 1

    mean_weight = total / finite_count if finite_count else None
    return WeightStats(
        header=header,
        min_weight=min_weight,
        max_weight=max_weight,
        mean_weight=mean_weight,
        nonfinite_count=nonfinite_count,
        all_weights_equal_1=(nonfinite_count == 0 and not_one_count == 0),
        not_one_count=not_one_count,
    )


def stats_to_dict(stats: WeightStats) -> dict[str, object]:
    h = stats.header
    return {
        "path": str(h.path),
        "format": h.format_name,
        "version": h.version,
        "orientation": h.orientation_name,
        "rows": h.rows,
        "cols": h.cols,
        "nnz": h.nnz,
        "edge_weight_count": h.values_count,
        "min_weight": stats.min_weight,
        "max_weight": stats.max_weight,
        "mean_weight": stats.mean_weight,
        "nonfinite_count": stats.nonfinite_count,
        "not_one_count": stats.not_one_count,
        "all_weights_equal_1": stats.all_weights_equal_1,
    }


def print_human(stats: WeightStats) -> None:
    data = stats_to_dict(stats)
    for key in (
        "path",
        "format",
        "version",
        "orientation",
        "rows",
        "cols",
        "nnz",
        "edge_weight_count",
        "min_weight",
        "max_weight",
        "mean_weight",
        "nonfinite_count",
        "not_one_count",
        "all_weights_equal_1",
    ):
        print(f"{key}: {data[key]}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print min, max, and mean edge weights from RIPS .csrbin files."
    )
    parser.add_argument("csrbin", nargs="+", type=Path, help="Input .csrbin file(s).")
    parser.add_argument(
        "--one-tol",
        type=float,
        default=0.0,
        help="Tolerance for all_weights_equal_1 / --expect-one. Default: 0.0.",
    )
    parser.add_argument(
        "--expect-one",
        action="store_true",
        help="Exit nonzero if any finite weight differs from 1.0 or any weight is non-finite.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.one_tol < 0.0:
        raise ValueError("--one-tol must be nonnegative")

    all_stats = [compute_weight_stats(path, args.one_tol) for path in args.csrbin]
    if args.json:
        print(json.dumps([stats_to_dict(stats) for stats in all_stats], indent=2))
    else:
        for index, stats in enumerate(all_stats):
            if index:
                print()
            print_human(stats)

    if args.expect_one and not all(stats.all_weights_equal_1 for stats in all_stats):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
