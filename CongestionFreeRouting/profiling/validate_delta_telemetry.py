#!/usr/bin/env python3
"""Validate one aggregate Delta-Stepping telemetry record from a router log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_FALLBACK_REASONS = {
    "none",
    "exact_unit_specialization",
    "progress_callback_requires_host",
    "cooperative_launch_unavailable",
    "generation_budget_unavailable",
}
EXPECTED_EFFECTIVE_MODES = {"host_checked", "reduced_round_trip"}
EXPECTED_EXECUTION_PATHS = {
    "exact_unit",
    "compact_generic",
    "legacy_generic",
    "generic_distances_only",
}
CONTROLLER_ERROR_COUNTERS = (
    "controller_queue_overflow_events",
    "controller_invalid_state_events",
    "controller_stale_publication_events",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--expected-controller",
        required=True,
        choices=("host_checked", "reduced_round_trip"),
    )
    parser.add_argument("--expected-batch", required=True, type=int)
    parser.add_argument("--expected-workers", required=True, type=int)
    parser.add_argument(
        "--require-device-work",
        action="store_true",
        help="require at least one completed reduced-controller batch",
    )
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def load_record(log: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            if candidate.get("type") == "delta_stepping_telemetry":
                records.append(candidate)
    if len(records) != 1:
        raise RuntimeError(
            f"expected one Delta telemetry record in {log}, found {len(records)}"
        )
    return records[0]


def require_equal(
    failures: list[str], label: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, observed {actual!r}")


def require_exact_keys(
    failures: list[str], label: str, value: Any, expected: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(f"{label}: expected an object, observed {value!r}")
        return {}
    actual = set(value)
    if actual != expected:
        failures.append(
            f"{label}: expected keys {sorted(expected)!r}, observed {sorted(actual)!r}"
        )
    return value


def require_nonnegative_int(
    failures: list[str], label: str, value: Any
) -> int:
    if type(value) is not int:
        failures.append(
            f"{label}: expected a nonnegative integer, observed {value!r}"
        )
        return -1
    if value < 0:
        failures.append(
            f"{label}: expected a nonnegative integer, observed {value!r}"
        )
        return -1
    return value


def hip_error_count(log: Path) -> int:
    with log.open(encoding="utf-8", errors="replace") as stream:
        count = 0
        for line in stream:
            if "HIP error at " in line:
                count += 1
            elif re.search(r"\bhip[A-Z][A-Za-z0-9_]* failed:", line):
                count += 1
            elif "HIP device reported" in line or "HIP runtime device query" in line:
                count += 1
        return count


def validate(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    require_equal(failures, "type", record.get("type"), "delta_stepping_telemetry")
    require_equal(failures, "schema_version", record.get("schema_version"), 4)
    require_equal(failures, "scope", record.get("scope"), "pathfinder_run")

    queries = require_nonnegative_int(failures, "queries", record.get("queries"))
    completed = require_nonnegative_int(
        failures, "completed_queries", record.get("completed_queries")
    )
    if queries <= 0:
        failures.append(f"queries must be positive, observed {queries}")
    require_equal(failures, "completed_queries", completed, queries)
    require_equal(
        failures, "parallel_workers", record.get("parallel_workers"), args.expected_workers
    )

    require_equal(failures, "force_generic", record.get("force_generic"), True)
    require_equal(
        failures, "force_legacy_parent", record.get("force_legacy_parent"), False
    )

    requested_mode = record.get("requested_controller_mode")
    requested_batch = record.get("requested_controller_batch_size")
    require_equal(
        failures, "requested_controller_mode", requested_mode, args.expected_controller
    )
    require_equal(
        failures, "requested_controller_batch_size", requested_batch, args.expected_batch
    )
    # Schema 4 retains the old configured names for consumers of schema 3.
    # Require both spellings to agree so a collector cannot silently read a
    # stale field while the explicit requested field says something else.
    require_equal(
        failures, "controller_mode", record.get("controller_mode"), requested_mode
    )
    require_equal(
        failures,
        "controller_batch_size",
        record.get("controller_batch_size"),
        requested_batch,
    )

    effective_modes = require_exact_keys(
        failures,
        "effective_controller_modes",
        record.get("effective_controller_modes"),
        EXPECTED_EFFECTIVE_MODES,
    )
    effective_mode_counts = {
        mode: require_nonnegative_int(
            failures,
            f"effective_controller_modes.{mode}",
            effective_modes.get(mode),
        )
        for mode in EXPECTED_EFFECTIVE_MODES
    }
    expected_host = queries if args.expected_controller == "host_checked" else 0
    expected_reduced = (
        queries if args.expected_controller == "reduced_round_trip" else 0
    )
    require_equal(
        failures,
        "effective_controller_modes.host_checked",
        effective_mode_counts["host_checked"],
        expected_host,
    )
    require_equal(
        failures,
        "effective_controller_modes.reduced_round_trip",
        effective_mode_counts["reduced_round_trip"],
        expected_reduced,
    )
    require_equal(
        failures,
        "effective controller query accounting",
        sum(effective_mode_counts.values()),
        queries,
    )

    expected_effective_batch = (
        args.expected_batch
        if args.expected_controller == "reduced_round_trip"
        else 1
    )
    effective_batch = require_exact_keys(
        failures,
        "effective_controller_batch_size",
        record.get("effective_controller_batch_size"),
        {"min", "max"},
    )
    effective_batch_min = require_nonnegative_int(
        failures,
        "effective_controller_batch_size.min",
        effective_batch.get("min"),
    )
    effective_batch_max = require_nonnegative_int(
        failures,
        "effective_controller_batch_size.max",
        effective_batch.get("max"),
    )
    require_equal(
        failures,
        "effective_controller_batch_size.min",
        effective_batch_min,
        expected_effective_batch,
    )
    require_equal(
        failures,
        "effective_controller_batch_size.max",
        effective_batch_max,
        expected_effective_batch,
    )

    fallback_queries = require_nonnegative_int(
        failures,
        "controller_fallback_queries",
        record.get("controller_fallback_queries"),
    )
    require_equal(
        failures,
        "controller_fallback_queries",
        fallback_queries,
        0,
    )
    fallback_reasons = require_exact_keys(
        failures,
        "controller_fallback_reasons",
        record.get("controller_fallback_reasons"),
        EXPECTED_FALLBACK_REASONS,
    )
    fallback_reason_counts = {
        reason: require_nonnegative_int(
            failures,
            f"controller_fallback_reasons.{reason}",
            fallback_reasons.get(reason),
        )
        for reason in EXPECTED_FALLBACK_REASONS
    }
    for reason, count in fallback_reason_counts.items():
        require_equal(failures, f"controller_fallback_reasons.{reason}", count, 0)
    require_equal(
        failures,
        "fallback reason accounting",
        sum(fallback_reason_counts.values()),
        fallback_queries,
    )

    execution_paths = require_exact_keys(
        failures,
        "execution_paths",
        record.get("execution_paths"),
        EXPECTED_EXECUTION_PATHS,
    )
    execution_path_counts = {
        path: require_nonnegative_int(
            failures, f"execution_paths.{path}", execution_paths.get(path)
        )
        for path in EXPECTED_EXECUTION_PATHS
    }
    require_equal(
        failures, "execution_paths.compact_generic", execution_path_counts["compact_generic"], queries
    )
    for path in ("exact_unit", "legacy_generic", "generic_distances_only"):
        require_equal(failures, f"execution_paths.{path}", execution_path_counts[path], 0)
    require_equal(
        failures,
        "execution path query accounting",
        sum(execution_path_counts.values()),
        queries,
    )

    counters_value = record.get("counters")
    if not isinstance(counters_value, dict):
        failures.append(f"counters: expected an object, observed {counters_value!r}")
        counters: dict[str, Any] = {}
    else:
        counters = counters_value
    batches = require_nonnegative_int(
        failures, "device_controller_batches", counters.get("device_controller_batches")
    )
    status_readbacks = require_nonnegative_int(
        failures,
        "controller_status_readbacks",
        counters.get("controller_status_readbacks"),
    )
    device_iterations = require_nonnegative_int(
        failures,
        "device_controller_iterations",
        counters.get("device_controller_iterations"),
    )
    controller_round_trips = require_nonnegative_int(
        failures, "controller_round_trips", counters.get("controller_round_trips")
    )
    batched_readbacks = require_nonnegative_int(
        failures, "batched_status_readbacks", counters.get("batched_status_readbacks")
    )
    light_rounds = require_nonnegative_int(
        failures, "light_relaxation_rounds", counters.get("light_relaxation_rounds")
    )
    controller_errors = {
        counter: require_nonnegative_int(failures, counter, counters.get(counter))
        for counter in CONTROLLER_ERROR_COUNTERS
    }
    for counter, count in controller_errors.items():
        require_equal(failures, counter, count, 0)
    compact_parent_fallback_events = require_nonnegative_int(
        failures,
        "compact_parent_fallback_events",
        counters.get("compact_parent_fallback_events"),
    )
    require_equal(
        failures,
        "compact_parent_fallback_events",
        compact_parent_fallback_events,
        0,
    )

    maxima_value = record.get("maxima")
    if not isinstance(maxima_value, dict):
        failures.append(f"maxima: expected an object, observed {maxima_value!r}")
        maxima: dict[str, Any] = {}
    else:
        maxima = maxima_value
    maximum = require_nonnegative_int(
        failures,
        "maxima.device_iterations_in_batch",
        maxima.get("device_iterations_in_batch"),
    )
    require_equal(
        failures,
        "batched_status_readbacks",
        batched_readbacks,
        batches + queries,
    )
    if args.expected_controller == "reduced_round_trip":
        require_equal(
            failures, "controller_status_readbacks", status_readbacks, batches
        )
        require_equal(
            failures, "controller_round_trips", controller_round_trips, batches
        )
        require_equal(
            failures, "device_controller_iterations", device_iterations, light_rounds
        )
        if batches < 0 or device_iterations < batches:
            failures.append(
                "device controller iterations must cover every published batch: "
                f"batches={batches}, iterations={device_iterations}"
            )
        if device_iterations > batches * args.expected_batch:
            failures.append(
                "device controller iterations exceed the aggregate batch bound: "
                f"iterations={device_iterations}, batches={batches}, "
                f"batch_size={args.expected_batch}"
            )
        if batches > 0 and not (1 <= maximum <= args.expected_batch):
            failures.append(
                "max device iterations in a nonempty batch is outside the "
                "requested bound: "
                f"maximum={maximum}, requested={args.expected_batch}"
            )
        if args.require_device_work and batches <= 0:
            failures.append("reduced controller performed no device batches")
    else:
        for label, value in (
            ("device_controller_batches", batches),
            ("controller_status_readbacks", status_readbacks),
            ("device_controller_iterations", device_iterations),
            ("max_device_iterations_in_batch", maximum),
        ):
            require_equal(failures, label, value, 0)

    observed_hip_errors = hip_error_count(args.log)
    require_equal(failures, "hip_error_count", observed_hip_errors, 0)

    if failures:
        raise RuntimeError("Delta telemetry validation failed:\n- " + "\n- ".join(failures))

    return {
        "log": str(args.log.resolve()),
        "schema_version": record.get("schema_version"),
        "scope": record.get("scope"),
        "queries": queries,
        "completed_queries": completed,
        "requested_controller_mode": requested_mode,
        "requested_controller_batch_size": requested_batch,
        "effective_controller_modes": effective_mode_counts,
        "effective_controller_batch_size": {
            "min": effective_batch_min,
            "max": effective_batch_max,
        },
        "controller_fallback_queries": fallback_queries,
        "controller_fallback_reasons": fallback_reason_counts,
        "execution_paths": execution_path_counts,
        "device_controller_batches": batches,
        "controller_round_trips": controller_round_trips,
        "controller_status_readbacks": status_readbacks,
        "device_controller_iterations": device_iterations,
        "max_device_iterations_in_batch": maximum,
        "batched_status_readbacks": batched_readbacks,
        "compact_parent_fallback_events": compact_parent_fallback_events,
        "controller_error_counters": controller_errors,
        "hip_error_count": observed_hip_errors,
    }


def main() -> int:
    args = parse_args()
    if args.expected_batch <= 0:
        raise ValueError("--expected-batch must be positive")
    if args.expected_workers <= 0:
        raise ValueError("--expected-workers must be positive")
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    summary = validate(args, load_record(args.log))
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
