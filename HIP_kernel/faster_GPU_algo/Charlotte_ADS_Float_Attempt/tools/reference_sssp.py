#!/usr/bin/env python3
"""Parse a float .gr graph and compute reference SSSP with Dijkstra."""

from __future__ import annotations

import argparse
import heapq
import math
import struct
from pathlib import Path


def load_graph(path: Path) -> list[list[tuple[int, float]]]:
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError("file is too short")
    version, edge_size, nodes, edges = struct.unpack_from("<QQQQ", data, 0)
    if version != 1 or edge_size not in (0, 4):
        raise ValueError(f"unsupported header: version={version}, edge_size={edge_size}")

    offset = 32
    row_ends = struct.unpack_from(f"<{nodes}Q", data, offset)
    offset += nodes * 8
    destinations = struct.unpack_from(f"<{edges}I", data, offset)
    offset += edges * 4
    if edges & 1:
        offset += 4
    weights = ((1.0,) * edges if edge_size == 0 else
               struct.unpack_from(f"<{edges}f", data, offset))

    graph: list[list[tuple[int, float]]] = [[] for _ in range(nodes)]
    start = 0
    for node, end in enumerate(row_ends):
        if end < start or end > edges:
            raise ValueError(f"bad row end for node {node}: {end}")
        graph[node] = [(destinations[i], float(weights[i])) for i in range(start, end)]
        start = end
    if start != edges:
        raise ValueError(f"final row end is {start}, but header declares {edges} edges")
    return graph


def dijkstra(graph: list[list[tuple[int, float]]], source: int) -> list[float]:
    distances = [math.inf] * len(graph)
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for destination, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[destination]:
                distances[destination] = candidate
                heapq.heappush(queue, (candidate, destination))
    return distances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("source", type=int, nargs="?", default=0)
    args = parser.parse_args()
    graph = load_graph(args.graph)
    if not 0 <= args.source < len(graph):
        raise SystemExit("source is out of range")
    for node, distance in enumerate(dijkstra(graph, args.source)):
        print(node, "INF" if math.isinf(distance) else f"{distance:.9g}")


if __name__ == "__main__":
    main()
