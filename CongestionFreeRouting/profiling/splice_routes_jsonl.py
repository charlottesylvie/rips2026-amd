#!/usr/bin/env python3
"""Splice smoke-run routes into a full reference routes JSONL artifact.

Reference record order is preserved.  Each emitted record retains its original
JSON text from the selected input; only its line terminator is normalized to
LF.  The operation validates both inputs fully before exclusively creating the
output, so an existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple


class RouteLine(NamedTuple):
    line_number: int
    net: str
    text: str


class RoutesJsonl(NamedTuple):
    path: Path
    records: tuple[RouteLine, ...]
    artifact_pair_id: str
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="full reference routes JSONL")
    parser.add_argument(
        "replacements", type=Path, help="smoke-run replacement routes JSONL"
    )
    parser.add_argument("output", type=Path, help="new spliced routes JSONL")
    parser.add_argument("--expected-replacements", required=True, type=int)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def load_routes_jsonl(path: Path, label: str) -> RoutesJsonl:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{path}: {label} is not valid UTF-8: {error}") from error

    records: list[RouteLine] = []
    net_lines: dict[str, int] = {}
    artifact_pair_id: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{path}:{line_number}: invalid {label} routes JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                f"{path}:{line_number}: {label} route record must be an object"
            )
        net = value.get("net")
        if not isinstance(net, str) or not net.strip():
            raise RuntimeError(
                f"{path}:{line_number}: {label} route net must be a nonempty string"
            )
        if net in net_lines:
            raise RuntimeError(
                f"{path}:{line_number}: duplicate {label} net {net!r}; "
                f"first seen on line {net_lines[net]}"
            )
        net_lines[net] = line_number

        record_pair_id = value.get("artifact_pair_id")
        if not isinstance(record_pair_id, str) or not record_pair_id.strip():
            raise RuntimeError(
                f"{path}:{line_number}: {label} artifact_pair_id must be a "
                "nonempty string"
            )
        if artifact_pair_id is None:
            artifact_pair_id = record_pair_id
        elif record_pair_id != artifact_pair_id:
            raise RuntimeError(
                f"{path}:{line_number}: inconsistent {label} artifact_pair_id "
                f"{record_pair_id!r}; expected {artifact_pair_id!r}"
            )
        records.append(RouteLine(line_number=line_number, net=net, text=line))

    if not records:
        raise RuntimeError(f"{path}: {label} routes JSONL is empty")
    assert artifact_pair_id is not None
    return RoutesJsonl(
        path=path,
        records=tuple(records),
        artifact_pair_id=artifact_pair_id,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def splice_routes(
    reference_path: Path,
    replacements_path: Path,
    output_path: Path,
    expected_replacements: int,
) -> dict[str, Any]:
    if expected_replacements <= 0:
        raise ValueError("expected replacement count must be positive")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    reference = load_routes_jsonl(reference_path, "reference")
    replacements = load_routes_jsonl(replacements_path, "replacement")
    if len(replacements.records) != expected_replacements:
        raise RuntimeError(
            f"replacement count: expected {expected_replacements}, "
            f"observed {len(replacements.records)}"
        )
    if replacements.artifact_pair_id != reference.artifact_pair_id:
        raise RuntimeError(
            "replacement artifact_pair_id does not match reference: "
            f"replacement={replacements.artifact_pair_id!r}, "
            f"reference={reference.artifact_pair_id!r}"
        )

    reference_nets = {record.net for record in reference.records}
    replacement_by_net = {record.net: record for record in replacements.records}
    missing_nets = sorted(set(replacement_by_net).difference(reference_nets))
    if missing_nets:
        raise RuntimeError(
            "replacement nets are absent from the reference: "
            + ", ".join(repr(net) for net in missing_nets)
        )

    output_text = "".join(
        replacement_by_net.get(record.net, record).text + "\n"
        for record in reference.records
    )
    output_payload = output_text.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as output:
            output.write(output_payload)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_path}"
        ) from error

    return {
        "artifact_pair_id": reference.artifact_pair_id,
        "line_policy": "original_record_text_with_lf_terminator",
        "expected_replacements": expected_replacements,
        "reference": {
            "path": str(reference.path.resolve()),
            "records": len(reference.records),
            "sha256": reference.sha256,
        },
        "replacements": {
            "path": str(replacements.path.resolve()),
            "records": len(replacements.records),
            "sha256": replacements.sha256,
        },
        "output": {
            "path": str(output_path.resolve()),
            "records": len(reference.records),
            "sha256": hashlib.sha256(output_payload).hexdigest(),
        },
    }


def main() -> int:
    args = parse_args()
    if args.expected_replacements <= 0:
        raise ValueError("--expected-replacements must be positive")
    if args.summary_out is not None:
        protected_paths = {
            args.reference.resolve(),
            args.replacements.resolve(),
            args.output.resolve(),
        }
        if args.summary_out.resolve() in protected_paths:
            raise ValueError("--summary-out must differ from all route paths")
    summary = splice_routes(
        args.reference,
        args.replacements,
        args.output,
        args.expected_replacements,
    )
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary_json, encoding="utf-8")
    print(summary_json, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
