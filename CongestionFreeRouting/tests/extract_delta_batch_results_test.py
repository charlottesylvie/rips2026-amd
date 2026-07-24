#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "extract_delta_batch_results.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_log(path: Path, wall: str, user: str, batch: int | None = None) -> None:
    lines = [f"Wall-clock time (sec): {wall}", f"User-CPU time (sec): {user}"]
    if batch is not None:
        telemetry = {
            "type": "delta_stepping_telemetry",
            "queries": 10,
            "completed_queries": 8,
            "resolved_delta": 1,
            "wavefront_size": 64,
            "parallel_workers": 2,
            "force_generic": True,
            "light_round_batch": batch,
            "execution_paths": {"exact_unit": 0, "compact_generic": 8,
                                "legacy_generic": 0, "generic_distances_only": 0},
            "counters": {"controller_round_trips": 32,
                         "light_relaxation_rounds": 16,
                         "outer_buckets_processed": 4},
        }
        lines.append(json.dumps(telemetry, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_log(root / "logicnets_jscl-batch0-run2.log", "20.5", "9.25")
        write_log(root / "logicnets_jscl-batch4-run2.log", "18.0", "8.0")
        write_log(root / "telemetry-batch4.log", "1.5", "1.0", batch=4)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(root), "--benchmark", "logicnets_jscl"],
            capture_output=True,
            text=True,
            check=False,
        )
        require(result.returncode == 0, result.stderr)
        with (root / "delta-light-batch-results.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        require(len(rows) == 3, "extractor did not retain all recognized logs")
        timing = next(row for row in rows if row["source_log"] == "logicnets_jscl-batch4-run2.log")
        require(timing["wall_clock_seconds"] == "18.0", "timing was not extracted")
        telemetry = next(row for row in rows if row["record_type"] == "telemetry")
        require(telemetry["light_round_batch"] == "4", "batch was not extracted")
        require(telemetry["controller_round_trips"] == "32", "telemetry was not flattened")
        require(
            telemetry["controller_round_trips_per_completed_query"] == "4",
            "derived controller-round-trip rate is incorrect",
        )
        with (root / "delta-light-batch-summary.csv").open(newline="", encoding="utf-8") as file:
            summary = list(csv.DictReader(file))
        require(len(summary) == 3, "summary did not preserve timing and telemetry groups")


if __name__ == "__main__":
    main()
