#!/usr/bin/env python3
"""Create a small float-weighted Galois/Lonestar .gr graph."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # 0 -> 1 (1), 0 -> 2 (4)
    # 1 -> 2 (2), 1 -> 3 (5)
    # 2 -> 3 (1); vertices 3 and 4 have no outgoing edges.
    num_nodes = 5
    destinations = [1, 2, 2, 3, 3]
    weights = [1.0, 4.0, 2.0, 5.0, 1.0]
    row_ends = [2, 4, 5, 5, 5]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<QQQQ", 1, 4, num_nodes, len(destinations)))
        stream.write(struct.pack(f"<{num_nodes}Q", *row_ends))
        stream.write(struct.pack(f"<{len(destinations)}I", *destinations))
        if len(destinations) % 2:
            stream.write(struct.pack("<I", 0))  # 64-bit alignment
        stream.write(struct.pack(f"<{len(weights)}f", *weights))


if __name__ == "__main__":
    main()
