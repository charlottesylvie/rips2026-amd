#!/usr/bin/env python3
"""Summarize and checksum a PathFinder routes JSONL artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routes", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--require-all-routed", action="store_true")
    return parser.parse_args()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: expected an object, observed {value!r}")
    return value


def require_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label}: expected an array, observed {value!r}")
    return value


def require_boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{label}: expected a boolean, observed {value!r}")
    return value


def summarize(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    route_requests = 0
    routed = 0
    unrouted = 0
    sources = 0
    sinks = 0
    reached_sinks = 0
    edges = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid routes JSON: {error}"
                ) from error
            record = require_object(record, f"{path}:{line_number}")
            routed_value = require_boolean(
                record.get("routed"), f"{path}:{line_number}: routed"
            )
            record_sources = require_array(
                record.get("sources"), f"{path}:{line_number}: sources"
            )
            record_sinks = require_array(
                record.get("sinks"), f"{path}:{line_number}: sinks"
            )
            record_edges = require_array(
                record.get("edges"), f"{path}:{line_number}: edges"
            )
            route_requests += 1
            if routed_value:
                routed += 1
            else:
                unrouted += 1
            sources += len(record_sources)
            sinks += len(record_sinks)
            for sink_index, sink_value in enumerate(record_sinks):
                sink = require_object(
                    sink_value,
                    f"{path}:{line_number}: sinks[{sink_index}]",
                )
                if require_boolean(
                    sink.get("reached"),
                    f"{path}:{line_number}: sinks[{sink_index}].reached",
                ):
                    reached_sinks += 1
            edges += len(record_edges)
    if route_requests == 0:
        raise RuntimeError(f"{path}: routes file is empty")
    return {
        "routes": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "route_requests": route_requests,
        "routed": routed,
        "unrouted": unrouted,
        "sources": sources,
        "sinks": sinks,
        "reached_sinks": reached_sinks,
        "edges": edges,
    }


def require_all_routed(summary: dict[str, Any]) -> None:
    if summary["unrouted"] != 0 or summary["reached_sinks"] != summary["sinks"]:
        raise RuntimeError(
            "routes are incomplete: "
            f"unrouted={summary['unrouted']}, "
            f"reached_sinks={summary['reached_sinks']}/{summary['sinks']}"
        )


def main() -> int:
    args = parse_args()
    if not args.routes.is_file():
        raise FileNotFoundError(args.routes)
    summary = summarize(args.routes)
    if args.require_all_routed:
        require_all_routed(summary)
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
