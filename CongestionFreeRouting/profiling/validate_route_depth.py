#!/usr/bin/env python3
"""Validate routed JSONL topology and its maximum source-to-sink depth."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_MAX_DEPTH = 214


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routes", type=Path)
    parser.add_argument(
        "--expected-max-depth",
        type=int,
        default=DEFAULT_EXPECTED_MAX_DEPTH,
        help=f"required maximum hop depth (default: {DEFAULT_EXPECTED_MAX_DEPTH})",
    )
    parser.add_argument(
        "--expected-route-count",
        type=int,
        help="required number of nonempty JSONL records",
    )
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: expected an object, observed {value!r}")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: expected an array, observed {value!r}")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(
            f"{label}: expected a nonnegative integer, observed {value!r}"
        )
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label}: expected a string, observed {value!r}")
    return value


def _load_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid routes JSON: {error}"
                ) from error
            records.append(
                (line_number, _require_object(value, f"{path}:{line_number}"))
            )
    if not records:
        raise RuntimeError(f"{path}: routes file is empty")
    return records


def _is_ancestor(
    ancestor: int,
    node: int,
    source_nodes: set[int],
    parent: dict[int, int],
) -> bool:
    current = node
    while True:
        if current == ancestor:
            return True
        if current in source_nodes:
            return False
        current = parent[current]


def _validate_record(
    path: Path, line_number: int, record: dict[str, Any]
) -> dict[str, Any]:
    prefix = f"{path}:{line_number}"
    net = _require_string(record.get("net"), f"{prefix}: net")
    if not net:
        raise RuntimeError(f"{prefix}: net name must not be empty")
    context = f"{prefix}: net {net!r}"
    if record.get("routed") is not True:
        raise RuntimeError(f"{context}: routed must be true")

    sources = _require_list(record.get("sources"), f"{context}: sources")
    sinks = _require_list(record.get("sinks"), f"{context}: sinks")
    edges = _require_list(record.get("edges"), f"{context}: edges")
    if not sources:
        raise RuntimeError(f"{context}: sources must not be empty")
    if not sinks:
        raise RuntimeError(f"{context}: sinks must not be empty")

    source_nodes: set[int] = set()
    for index, value in enumerate(sources):
        label = f"{context}: sources[{index}]"
        source = _require_object(value, label)
        node = _require_nonnegative_int(source.get("node"), f"{label}.node")
        _require_string(source.get("site"), f"{label}.site")
        _require_string(source.get("pin"), f"{label}.pin")
        # Duplicate route sources are supported by the router and are
        # deduplicated before the SSSP query.  Count the JSON endpoints as
        # written, but treat equal node IDs as one topology root.
        source_nodes.add(node)

    sink_values: list[tuple[int, int]] = []
    sink_nodes: set[int] = set()
    for index, value in enumerate(sinks):
        label = f"{context}: sinks[{index}]"
        sink = _require_object(value, label)
        target = _require_nonnegative_int(sink.get("node"), f"{label}.node")
        _require_string(sink.get("site"), f"{label}.site")
        _require_string(sink.get("pin"), f"{label}.pin")
        if sink.get("reached") is not True:
            raise RuntimeError(f"{label}.reached: expected true")
        attachment = _require_nonnegative_int(
            sink.get("source"), f"{label}.source"
        )
        sink_values.append((target, attachment))
        sink_nodes.add(target)

    adjacency: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    graph_nodes = set(source_nodes)
    graph_nodes.update(sink_nodes)
    seen_edges: set[tuple[int, int]] = set()
    for index, value in enumerate(edges):
        label = f"{context}: edges[{index}]"
        edge = _require_object(value, label)
        source = _require_nonnegative_int(edge.get("from"), f"{label}.from")
        target = _require_nonnegative_int(edge.get("to"), f"{label}.to")
        if source == target:
            raise RuntimeError(f"{label}: self-loop at node {source}")
        edge_key = (source, target)
        if edge_key in seen_edges:
            raise RuntimeError(f"{label}: duplicate edge {source}->{target}")
        seen_edges.add(edge_key)
        if target in parent and parent[target] != source:
            raise RuntimeError(
                f"{label}: node {target} has multiple parents "
                f"({parent[target]} and {source})"
            )
        parent[target] = source
        adjacency.setdefault(source, []).append(target)
        graph_nodes.update((source, target))

    incoming_sources = sorted(source_nodes.intersection(parent))
    if incoming_sources:
        raise RuntimeError(
            f"{context}: route edges enter declared source nodes "
            f"{incoming_sources!r}"
        )

    missing_parents = sorted(
        node for node in graph_nodes if node not in source_nodes and node not in parent
    )
    if missing_parents:
        raise RuntimeError(
            f"{context}: non-source nodes have no parent: {missing_parents!r}"
        )

    indegree = {node: 0 for node in graph_nodes}
    for target in parent:
        indegree[target] += 1
    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for child in adjacency.get(node, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(graph_nodes):
        cyclic_nodes = sorted(node for node, degree in indegree.items() if degree)
        raise RuntimeError(
            f"{context}: route topology contains a cycle involving "
            f"{cyclic_nodes!r}"
        )

    depth = {source: 0 for source in source_nodes}
    pending = deque(sorted(source_nodes))
    while pending:
        node = pending.popleft()
        for child in adjacency.get(node, ()):
            depth[child] = depth[node] + 1
            pending.append(child)
    unreachable = sorted(graph_nodes.difference(depth))
    if unreachable:
        raise RuntimeError(
            f"{context}: nodes are unreachable from declared sources: "
            f"{unreachable!r}"
        )

    dangling_leaves = sorted(
        node
        for node in graph_nodes
        if node not in source_nodes
        and node not in adjacency
        and node not in sink_nodes
    )
    if dangling_leaves:
        raise RuntimeError(
            f"{context}: route has non-sink leaf nodes: {dangling_leaves!r}"
        )

    sink_depths: list[int] = []
    for index, (target, attachment) in enumerate(sink_values):
        if attachment not in depth:
            raise RuntimeError(
                f"{context}: sinks[{index}].source {attachment} is not in "
                "the route topology"
            )
        if not _is_ancestor(attachment, target, source_nodes, parent):
            raise RuntimeError(
                f"{context}: sinks[{index}].source {attachment} is not an "
                f"ancestor of target {target}"
            )
        sink_depths.append(depth[target])

    graph_maximum = max(depth.values())
    sink_maximum = max(sink_depths)
    if graph_maximum != sink_maximum:
        raise RuntimeError(
            f"{context}: deepest route node is not a sink "
            f"(node depth {graph_maximum}, sink depth {sink_maximum})"
        )

    return {
        "net": net,
        "sources": len(sources),
        "sinks": len(sinks),
        "edges": len(edges),
        "maximum_depth": sink_maximum,
        "sink_depths": sink_depths,
    }


def validate_routes(
    path: Path,
    expected_max_depth: int = DEFAULT_EXPECTED_MAX_DEPTH,
    expected_route_count: int | None = None,
) -> dict[str, Any]:
    if expected_max_depth < 0:
        raise ValueError("expected maximum depth must be nonnegative")
    if expected_route_count is not None and expected_route_count <= 0:
        raise ValueError("expected route count must be positive")

    records = _load_records(path)
    if expected_route_count is not None and len(records) != expected_route_count:
        raise RuntimeError(
            f"route count: expected {expected_route_count}, observed {len(records)}"
        )

    seen_nets: set[str] = set()
    total_sources = 0
    total_sinks = 0
    total_edges = 0
    maximum_depth = 0
    depth_histogram: Counter[int] = Counter()
    for line_number, record in records:
        result = _validate_record(path, line_number, record)
        net = result["net"]
        if net in seen_nets:
            raise RuntimeError(
                f"{path}:{line_number}: duplicate net record {net!r}"
            )
        seen_nets.add(net)
        total_sources += result["sources"]
        total_sinks += result["sinks"]
        total_edges += result["edges"]
        maximum_depth = max(maximum_depth, result["maximum_depth"])
        depth_histogram.update(result["sink_depths"])

    if maximum_depth != expected_max_depth:
        raise RuntimeError(
            f"maximum route depth: expected {expected_max_depth}, "
            f"observed {maximum_depth}"
        )

    return {
        "routes": str(path.resolve()),
        "route_requests": len(records),
        "routed": len(records),
        "unrouted": 0,
        "sources": total_sources,
        "sinks": total_sinks,
        "reached_sinks": total_sinks,
        "edges": total_edges,
        "expected_maximum_depth": expected_max_depth,
        "maximum_depth": maximum_depth,
        "sink_depth_histogram": {
            str(depth): count for depth, count in sorted(depth_histogram.items())
        },
    }


def main() -> int:
    args = parse_args()
    if not args.routes.is_file():
        raise FileNotFoundError(args.routes)
    summary = validate_routes(
        args.routes,
        expected_max_depth=args.expected_max_depth,
        expected_route_count=args.expected_route_count,
    )
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
