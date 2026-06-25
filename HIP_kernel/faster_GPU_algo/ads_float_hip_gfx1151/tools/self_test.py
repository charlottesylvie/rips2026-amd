#!/usr/bin/env python3
"""CPU-only regression tests for the .gr utilities."""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path

from compare_output import load_output
from reference_sssp import dijkstra, load_graph


def write_gr(path: Path, rows: list[int], destinations: list[int],
             weights: list[float] | None) -> None:
    nodes = len(rows)
    edges = len(destinations)
    edge_size = 0 if weights is None else 4
    with path.open("wb") as stream:
        stream.write(struct.pack("<QQQQ", 1, edge_size, nodes, edges))
        stream.write(struct.pack(f"<{nodes}Q", *rows))
        stream.write(struct.pack(f"<{edges}I", *destinations))
        if edges & 1:
            stream.write(struct.pack("<I", 0))
        if weights is not None:
            stream.write(struct.pack(f"<{edges}f", *weights))


def assert_distances(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if math.isinf(want):
            assert math.isinf(got)
        else:
            assert math.isclose(got, want, rel_tol=1e-7, abs_tol=1e-7)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="adds_hip_self_test_") as tmp:
        root = Path(tmp)

        weighted = root / "weighted.gr"
        write_gr(
            weighted,
            rows=[2, 4, 5, 5, 5],
            destinations=[1, 2, 2, 3, 3],
            weights=[1.0, 4.0, 2.0, 5.0, 1.0],
        )
        weighted_distances = dijkstra(load_graph(weighted), 0)
        assert_distances(weighted_distances, [0.0, 1.0, 3.0, 4.0, math.inf])

        output = root / "weighted.out"
        output.write_text("0 0\n1 1\n2 3\n3 4\n4 INF\n")
        assert_distances(load_output(output, 5), weighted_distances)

        unweighted = root / "unweighted.gr"
        write_gr(
            unweighted,
            rows=[2, 3, 3],
            destinations=[1, 2, 2],
            weights=None,
        )
        # Missing edge payload denotes unit weights, so the direct 0->2 edge
        # has distance one rather than zero.
        assert_distances(dijkstra(load_graph(unweighted), 0), [0.0, 1.0, 1.0])

        malformed = root / "malformed.gr"
        write_gr(
            malformed,
            rows=[1, 1],
            destinations=[1, 0],
            weights=[1.0, 1.0],
        )
        try:
            load_graph(malformed)
        except ValueError as error:
            assert "final row end" in str(error)
        else:
            raise AssertionError("malformed final row offset was accepted")

    print("CPU graph/reference tests passed")


if __name__ == "__main__":
    main()
