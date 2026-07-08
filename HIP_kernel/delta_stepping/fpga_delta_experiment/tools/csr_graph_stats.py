#!/usr/bin/env python3
"""Compute graph statistics and histograms from RIPSCSR1 .csrbin files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
from array import array
from collections import Counter
from pathlib import Path


CSR_MAGIC = b"RIPSCSR1"
CSR_VERSION = 1
INCOMING_EDGE_ORIENTATION = 1


def read_u64(stream, name: str) -> int:
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f"failed reading {name}")
    return struct.unpack("<Q", raw)[0]


def read_array(stream, typecode: str, count: int, name: str) -> array:
    values = array(typecode)
    if count == 0:
        return values
    byte_count = count * values.itemsize
    raw = stream.read(byte_count)
    if len(raw) != byte_count:
        raise ValueError(f"failed reading {name}")
    values.frombytes(raw)
    if values.itemsize > 1:
        values.byteswap() if is_big_endian_host() else None
    return values


def is_big_endian_host() -> bool:
    return struct.pack("=H", 1) == b"\x00\x01"


class CsrGraph:
    def __init__(self, rows: int, cols: int, nnz: int, rowptr: array, colind: array, values: array):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colind = colind
        self.values = values


class CsrHeader:
    def __init__(
        self,
        rows: int,
        cols: int,
        nnz: int,
        rowptr_count: int,
        colind_count: int,
        values_count: int,
    ):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr_count = rowptr_count
        self.colind_count = colind_count
        self.values_count = values_count


def read_csr_header(stream, path: Path) -> CsrHeader:
    magic = stream.read(8)
    if magic != CSR_MAGIC:
        raise ValueError(f"{path} is not a RIPSCSR1 file")

    version = read_u64(stream, "version")
    orientation = read_u64(stream, "orientation")
    rows = read_u64(stream, "rows")
    cols = read_u64(stream, "cols")
    read_u64(stream, "declared_edges")
    read_u64(stream, "loaded_edges")
    nnz = read_u64(stream, "nnz")
    rowptr_count = read_u64(stream, "rowptr_count")
    colind_count = read_u64(stream, "colind_count")
    values_count = read_u64(stream, "values_count")

    if version != CSR_VERSION:
        raise ValueError(f"{path} has unsupported CSR version {version}")
    if orientation != INCOMING_EDGE_ORIENTATION:
        raise ValueError(f"{path} is not incoming-edge CSR")
    if rowptr_count != rows + 1:
        raise ValueError(f"{path} rowptr_count does not equal rows + 1")
    if colind_count != nnz or values_count != nnz:
        raise ValueError(f"{path} CSR counts do not match nnz")

    return CsrHeader(rows, cols, nnz, rowptr_count, colind_count, values_count)


def load_csr(path: Path) -> CsrGraph:
    with path.open("rb") as stream:
        header = read_csr_header(stream, path)
        rowptr = read_array(stream, "q", header.rowptr_count, "rowptr")
        colind = read_array(stream, "i", header.colind_count, "colind")
        values = read_array(stream, "f", header.values_count, "values")

    return CsrGraph(header.rows, header.cols, header.nnz, rowptr, colind, values)


def quick_summarize_csr(benchmark: str, path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        header = read_csr_header(stream, path)
        stream.seek(header.rowptr_count * 8 + header.colind_count * 4, 1)

        total_weight = 0.0
        min_weight = math.inf
        max_weight = -math.inf
        remaining = header.values_count
        while remaining:
            count = min(remaining, 1_000_000)
            values = read_array(stream, "f", count, "values")
            total_weight += math.fsum(float(value) for value in values)
            if values:
                min_weight = min(min_weight, min(values))
                max_weight = max(max_weight, max(values))
            remaining -= count

    return {
        "benchmark": benchmark,
        "vertices": header.rows,
        "edges": header.nnz,
        "avg_vertex_degree": header.nnz / header.rows if header.rows else math.nan,
        "avg_in_degree": header.nnz / header.rows if header.rows else math.nan,
        "avg_out_degree": header.nnz / header.rows if header.rows else math.nan,
        "avg_edge_weight": total_weight / header.values_count
        if header.values_count
        else math.nan,
        "min_edge_weight": min_weight if header.values_count else math.nan,
        "max_edge_weight": max_weight if header.values_count else math.nan,
    }


def percentile(sorted_values: list[float] | array, pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return float(sorted_values[lo])
    weight = index - lo
    return float(sorted_values[lo]) * (1.0 - weight) + float(sorted_values[hi]) * weight


def histogram(values: list[float] | array, bins: int) -> list[tuple[float, float, int]]:
    if not values:
        return []
    min_value = float(min(values))
    max_value = float(max(values))
    if min_value == max_value:
        return [(min_value, max_value, len(values))]

    width = (max_value - min_value) / bins
    counts = [0] * bins
    for value in values:
        bucket = int((float(value) - min_value) / width)
        if bucket == bins:
            bucket -= 1
        counts[bucket] += 1

    return [
        (min_value + i * width, min_value + (i + 1) * width, count)
        for i, count in enumerate(counts)
    ]


def integer_histogram(values: list[int]) -> list[tuple[int, int]]:
    return sorted(Counter(values).items())


def incoming_degree_values(rowptr: array) -> list[int]:
    return [int(rowptr[i + 1] - rowptr[i]) for i in range(len(rowptr) - 1)]


def outgoing_degree_values(colind: array, vertex_count: int) -> list[int]:
    degrees = [0] * vertex_count
    for source in colind:
        if 0 <= source < vertex_count:
            degrees[int(source)] += 1
    return degrees


def summarize(benchmark: str, graph: CsrGraph) -> dict[str, object]:
    in_degrees = incoming_degree_values(graph.rowptr)
    out_degrees = outgoing_degree_values(graph.colind, graph.rows)
    total_degrees = [
        in_degree + out_degree for in_degree, out_degree in zip(in_degrees, out_degrees)
    ]
    degrees = in_degrees
    sorted_degrees = sorted(degrees)
    sorted_in_degrees = sorted(in_degrees)
    sorted_out_degrees = sorted(out_degrees)
    sorted_weights = sorted(float(value) for value in graph.values)
    nonzero_degrees = [degree for degree in degrees if degree != 0]

    return {
        "benchmark": benchmark,
        "vertices": graph.rows,
        "edges": graph.nnz,
        "avg_vertex_degree": graph.nnz / graph.rows if graph.rows else math.nan,
        "median_vertex_degree": statistics.median(sorted_degrees) if sorted_degrees else math.nan,
        "max_vertex_degree": max(degrees) if degrees else 0,
        "zero_degree_vertices": len(degrees) - len(nonzero_degrees),
        "avg_nonzero_vertex_degree": statistics.fmean(nonzero_degrees) if nonzero_degrees else math.nan,
        "median_nonzero_vertex_degree": statistics.median(nonzero_degrees) if nonzero_degrees else math.nan,
        "avg_in_degree": graph.nnz / graph.rows if graph.rows else math.nan,
        "median_in_degree": statistics.median(sorted_in_degrees) if sorted_in_degrees else math.nan,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "avg_out_degree": graph.nnz / graph.rows if graph.rows else math.nan,
        "median_out_degree": statistics.median(sorted_out_degrees) if sorted_out_degrees else math.nan,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "avg_total_degree": 2.0 * graph.nnz / graph.rows if graph.rows else math.nan,
        "median_total_degree": statistics.median(sorted(total_degrees)) if total_degrees else math.nan,
        "max_total_degree": max(total_degrees) if total_degrees else 0,
        "avg_edge_weight": statistics.fmean(sorted_weights) if sorted_weights else math.nan,
        "median_edge_weight": statistics.median(sorted_weights) if sorted_weights else math.nan,
        "min_edge_weight": sorted_weights[0] if sorted_weights else math.nan,
        "p25_edge_weight": percentile(sorted_weights, 25),
        "p75_edge_weight": percentile(sorted_weights, 75),
        "p90_edge_weight": percentile(sorted_weights, 90),
        "p95_edge_weight": percentile(sorted_weights, 95),
        "max_edge_weight": sorted_weights[-1] if sorted_weights else math.nan,
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["benchmark"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1000:
            return f"{value:,.3f}"
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        print("No rows to display.")
        return

    labels = {
        "benchmark": "benchmark",
        "vertices": "vertices",
        "edges": "edges",
        "avg_vertex_degree": "avg in degree",
        "median_vertex_degree": "median in degree",
        "max_vertex_degree": "max in degree",
        "zero_degree_vertices": "zero degree",
        "avg_nonzero_vertex_degree": "avg nz degree",
        "median_nonzero_vertex_degree": "median nz degree",
        "avg_in_degree": "avg in degree",
        "median_in_degree": "median in degree",
        "max_in_degree": "max in degree",
        "avg_out_degree": "avg out degree",
        "median_out_degree": "median out degree",
        "max_out_degree": "max out degree",
        "avg_total_degree": "avg total degree",
        "median_total_degree": "median total degree",
        "max_total_degree": "max total degree",
        "avg_edge_weight": "avg weight",
        "median_edge_weight": "median weight",
        "min_edge_weight": "min weight",
        "p25_edge_weight": "p25 weight",
        "p75_edge_weight": "p75 weight",
        "p90_edge_weight": "p90 weight",
        "p95_edge_weight": "p95 weight",
        "max_edge_weight": "max weight",
    }
    headers = [labels.get(column, column) for column in columns]
    formatted_rows = [[format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in formatted_rows))
        for i in range(len(columns))
    ]

    def render(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print()
    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        print(render(row))
    print()


def write_histogram(path: Path, benchmark: str, kind: str, rows: list[tuple[float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["benchmark", "kind", "bin_start", "bin_end", "count"],
        )
        writer.writeheader()
        for start, end, count in rows:
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "kind": kind,
                    "bin_start": start,
                    "bin_end": end,
                    "count": count,
                }
            )


def write_integer_histogram(path: Path, benchmark: str, kind: str, rows: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["benchmark", "kind", "degree", "count"],
        )
        writer.writeheader()
        for degree, count in rows:
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "kind": kind,
                    "degree": degree,
                    "count": count,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/csr_sweep_manifest.csv", type=Path)
    parser.add_argument("--out-dir", default="results/graph_stats", type=Path)
    parser.add_argument("--bins", default=50, type=int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only compute averages/min/max; skip medians, percentiles, and histograms.",
    )
    parser.add_argument(
        "--degree-kind",
        choices=["all", "incoming", "outgoing", "total"],
        default="all",
        help="Which degree histograms to write in full mode. Default: all.",
    )
    args = parser.parse_args()

    if args.bins < 1:
        raise SystemExit("--bins must be >= 1")

    summary_rows: list[dict[str, object]] = []
    for row in load_manifest(args.manifest):
        benchmark = row["benchmark"]
        csr_path = Path(row["csr"])
        print(f"analyzing {benchmark}: {csr_path}", flush=True)
        if args.quick:
            summary_rows.append(quick_summarize_csr(benchmark, csr_path))
            continue

        graph = load_csr(csr_path)
        summary_rows.append(summarize(benchmark, graph))

        in_degrees = incoming_degree_values(graph.rowptr)
        out_degrees = outgoing_degree_values(graph.colind, graph.rows)
        degrees = [in_degree + out_degree for in_degree, out_degree in zip(in_degrees, out_degrees)]
        if args.degree_kind in {"all", "total"}:
            write_integer_histogram(
                args.out_dir / f"{benchmark}_degree_histogram.csv",
                benchmark,
                "vertex_degree",
                integer_histogram(degrees),
            )
        if args.degree_kind in {"all", "incoming"}:
            write_integer_histogram(
                args.out_dir / f"{benchmark}_incoming_degree_histogram.csv",
                benchmark,
                "incoming_degree",
                integer_histogram(in_degrees),
            )
        if args.degree_kind in {"all", "outgoing"}:
            write_integer_histogram(
                args.out_dir / f"{benchmark}_outgoing_degree_histogram.csv",
                benchmark,
                "outgoing_degree",
                integer_histogram(out_degrees),
            )
        write_histogram(
            args.out_dir / f"{benchmark}_edge_weight_histogram.csv",
            benchmark,
            "edge_weight",
            histogram(graph.values, args.bins),
        )

    summary_name = "graph_summary_quick.csv" if args.quick else "graph_summary.csv"
    write_summary(summary_rows, args.out_dir / summary_name)
    table_columns = (
        [
            "benchmark",
            "vertices",
            "edges",
            "avg_vertex_degree",
            "avg_edge_weight",
            "min_edge_weight",
            "max_edge_weight",
        ]
        if args.quick
        else [
            "benchmark",
            "vertices",
            "edges",
            "avg_vertex_degree",
            "median_vertex_degree",
            "avg_edge_weight",
            "median_edge_weight",
            "p90_edge_weight",
        ]
    )
    print_table(summary_rows, table_columns)
    print(f"wrote {args.out_dir / summary_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
