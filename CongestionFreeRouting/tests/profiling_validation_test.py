#!/usr/bin/env python3
"""Focused tests for Stage-1 profiling and route-artifact helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILING = REPOSITORY / "CongestionFreeRouting" / "profiling"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TELEMETRY = load_module(
    "validate_delta_telemetry",
    PROFILING / "validate_delta_telemetry.py",
)
ROUTES = load_module(
    "validate_route_depth",
    PROFILING / "validate_route_depth.py",
)
SUMMARIZER = load_module(
    "summarize_routes_jsonl",
    PROFILING / "summarize_routes_jsonl.py",
)
ANALYZER = load_module(
    "analyze_rocpd",
    PROFILING / "analyze_rocpd.py",
)
SPLICER = load_module(
    "splice_routes_jsonl",
    PROFILING / "splice_routes_jsonl.py",
)


def reduced_telemetry_record() -> dict[str, object]:
    return {
        "type": "delta_stepping_telemetry",
        "schema_version": 4,
        "scope": "pathfinder_run",
        "queries": 3,
        "completed_queries": 3,
        "parallel_workers": 4,
        "force_generic": True,
        "force_legacy_parent": False,
        "controller_mode": "reduced_round_trip",
        "controller_batch_size": 4,
        "requested_controller_mode": "reduced_round_trip",
        "requested_controller_batch_size": 4,
        "effective_controller_modes": {
            "host_checked": 0,
            "reduced_round_trip": 3,
        },
        "effective_controller_batch_size": {"min": 4, "max": 4},
        "controller_fallback_queries": 0,
        "controller_fallback_reasons": {
            "none": 0,
            "exact_unit_specialization": 0,
            "progress_callback_requires_host": 0,
            "cooperative_launch_unavailable": 0,
            "generation_budget_unavailable": 0,
        },
        "execution_paths": {
            "exact_unit": 0,
            "compact_generic": 3,
            "legacy_generic": 0,
            "generic_distances_only": 0,
        },
        "counters": {
            "light_relaxation_rounds": 7,
            "controller_round_trips": 3,
            "batched_status_readbacks": 6,
            "device_controller_batches": 3,
            "controller_status_readbacks": 3,
            "device_controller_iterations": 7,
            "controller_queue_overflow_events": 0,
            "controller_invalid_state_events": 0,
            "controller_stale_publication_events": 0,
            "compact_parent_fallback_events": 0,
        },
        "maxima": {"device_iterations_in_batch": 3},
    }


def source(node: int) -> dict[str, object]:
    return {"node": node, "site": f"SOURCE_{node}", "pin": "OUT"}


def sink(node: int, attachment: int) -> dict[str, object]:
    return {
        "node": node,
        "site": f"SINK_{node}",
        "pin": "IN",
        "reached": True,
        "source": attachment,
    }


def edge(source_node: int, target_node: int) -> dict[str, int]:
    return {"from": source_node, "to": target_node}


class TemporaryFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_jsonl(self, name: str, records: list[dict[str, object]]) -> Path:
        path = self.directory / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path


class DeltaTelemetryValidationTest(TemporaryFiles):
    def args(self, log: Path) -> argparse.Namespace:
        return argparse.Namespace(
            log=log,
            expected_controller="reduced_round_trip",
            expected_batch=4,
            expected_workers=4,
            require_device_work=True,
            summary_out=None,
        )

    def validate(self, record: dict[str, object], suffix: str = "") -> dict:
        log = self.directory / "router.log"
        log.write_text(json.dumps(record) + "\n" + suffix, encoding="utf-8")
        return TELEMETRY.validate(self.args(log), TELEMETRY.load_record(log))

    def test_accepts_complete_reduced_controller_telemetry(self) -> None:
        summary = self.validate(reduced_telemetry_record())
        self.assertEqual(summary["controller_fallback_queries"], 0)
        self.assertEqual(set(summary["controller_error_counters"].values()), {0})
        self.assertEqual(summary["hip_error_count"], 0)

    def test_accepts_host_checked_oracle_with_no_device_controller_work(self) -> None:
        record = reduced_telemetry_record()
        record["controller_mode"] = "host_checked"
        record["requested_controller_mode"] = "host_checked"
        record["effective_controller_modes"] = {
            "host_checked": 3,
            "reduced_round_trip": 0,
        }
        record["effective_controller_batch_size"] = {"min": 1, "max": 1}
        counters = record["counters"]
        assert isinstance(counters, dict)
        counters.update(
            {
                "controller_round_trips": 12,
                "batched_status_readbacks": 3,
                "device_controller_batches": 0,
                "controller_status_readbacks": 0,
                "device_controller_iterations": 0,
            }
        )
        maxima = record["maxima"]
        assert isinstance(maxima, dict)
        maxima["device_iterations_in_batch"] = 0
        log = self.directory / "host.log"
        log.write_text(json.dumps(record) + "\n", encoding="utf-8")
        args = self.args(log)
        args.expected_controller = "host_checked"
        args.require_device_work = False
        summary = TELEMETRY.validate(args, TELEMETRY.load_record(log))
        self.assertEqual(summary["effective_controller_modes"]["host_checked"], 3)

    def test_rejects_incomplete_fallback_reason_vocabulary(self) -> None:
        record = reduced_telemetry_record()
        reasons = record["controller_fallback_reasons"]
        assert isinstance(reasons, dict)
        del reasons["generation_budget_unavailable"]
        with self.assertRaisesRegex(RuntimeError, "expected keys"):
            self.validate(record)

    def test_rejects_compact_parent_fallback(self) -> None:
        record = reduced_telemetry_record()
        counters = record["counters"]
        assert isinstance(counters, dict)
        counters["compact_parent_fallback_events"] = 1
        with self.assertRaisesRegex(RuntimeError, "compact_parent_fallback_events"):
            self.validate(record)

    def test_rejects_stale_batched_readback_accounting(self) -> None:
        record = reduced_telemetry_record()
        counters = record["counters"]
        assert isinstance(counters, dict)
        counters["batched_status_readbacks"] = 9
        with self.assertRaisesRegex(RuntimeError, "batched_status_readbacks"):
            self.validate(record)

    def test_rejects_zero_or_overbound_nonempty_batch_maximum(self) -> None:
        for maximum in (0, 5):
            with self.subTest(maximum=maximum):
                record = reduced_telemetry_record()
                maxima = record["maxima"]
                assert isinstance(maxima, dict)
                maxima["device_iterations_in_batch"] = maximum
                with self.assertRaisesRegex(
                    RuntimeError, "max device iterations in a nonempty batch"
                ):
                    self.validate(record)

    def test_rejects_device_iterations_above_aggregate_batch_bound(self) -> None:
        record = reduced_telemetry_record()
        counters = record["counters"]
        assert isinstance(counters, dict)
        counters["device_controller_iterations"] = 13
        counters["light_relaxation_rounds"] = 13
        with self.assertRaisesRegex(RuntimeError, "aggregate batch bound"):
            self.validate(record)

    def test_rejects_hip_error_diagnostic(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "hip_error_count"):
            self.validate(
                reduced_telemetry_record(),
                "HIP error at delta_stepping_hip_CSR.cpp:99: invalid value\n",
            )

    def test_rejects_pathfinder_hip_failure_diagnostic(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "hip_error_count"):
            self.validate(
                reduced_telemetry_record(),
                "hipStreamCreateWithFlags failed: invalid resource handle\n",
            )


class RouteDepthValidationTest(TemporaryFiles):
    def valid_records(self) -> list[dict[str, object]]:
        return [
            {
                "net": "net_a",
                "routed": True,
                "sources": [source(0)],
                "sinks": [sink(2, 0), sink(3, 1)],
                "edges": [edge(0, 1), edge(1, 2), edge(1, 3)],
            },
            {
                "net": "net_b",
                "routed": True,
                "sources": [source(10)],
                "sinks": [sink(10, 10)],
                "edges": [],
            },
        ]

    def test_accepts_forest_and_counts_attachment_sources_as_ancestors(self) -> None:
        path = self.write_jsonl("routes.jsonl", self.valid_records())
        summary = ROUTES.validate_routes(
            path, expected_max_depth=2, expected_route_count=2
        )
        self.assertEqual(summary["maximum_depth"], 2)
        self.assertEqual(summary["sink_depth_histogram"], {"0": 1, "2": 2})

    def test_accepts_duplicate_source_nodes_supported_by_router(self) -> None:
        record = self.valid_records()[0]
        sources = record["sources"]
        assert isinstance(sources, list)
        sources.append(source(0))
        path = self.write_jsonl("duplicate-source.jsonl", [record])
        summary = ROUTES.validate_routes(path, expected_max_depth=2)
        self.assertEqual(summary["sources"], 2)

    def test_rejects_duplicate_edge(self) -> None:
        records = self.valid_records()[:1]
        edges = records[0]["edges"]
        assert isinstance(edges, list)
        edges.append(edge(1, 2))
        path = self.write_jsonl("duplicate.jsonl", records)
        with self.assertRaisesRegex(RuntimeError, "duplicate edge"):
            ROUTES.validate_routes(path, expected_max_depth=2)

    def test_rejects_missing_parent(self) -> None:
        record = {
            "net": "detached",
            "routed": True,
            "sources": [source(0)],
            "sinks": [sink(2, 1)],
            "edges": [edge(1, 2)],
        }
        path = self.write_jsonl("missing-parent.jsonl", [record])
        with self.assertRaisesRegex(RuntimeError, "no parent"):
            ROUTES.validate_routes(path, expected_max_depth=1)

    def test_rejects_disconnected_cycle(self) -> None:
        record = {
            "net": "cycle",
            "routed": True,
            "sources": [source(0)],
            "sinks": [sink(2, 1)],
            "edges": [edge(1, 2), edge(2, 1)],
        }
        path = self.write_jsonl("cycle.jsonl", [record])
        with self.assertRaisesRegex(RuntimeError, "contains a cycle"):
            ROUTES.validate_routes(path, expected_max_depth=1)

    def test_rejects_nonancestor_attachment_source(self) -> None:
        record = self.valid_records()[0]
        sinks = record["sinks"]
        assert isinstance(sinks, list)
        assert isinstance(sinks[0], dict)
        sinks[0]["source"] = 3
        path = self.write_jsonl("bad-attachment.jsonl", [record])
        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            ROUTES.validate_routes(path, expected_max_depth=2)

    def test_rejects_wrong_route_count(self) -> None:
        path = self.write_jsonl("count.jsonl", self.valid_records())
        with self.assertRaisesRegex(RuntimeError, "route count"):
            ROUTES.validate_routes(
                path, expected_max_depth=2, expected_route_count=27_960
            )


class RouteSummaryValidationTest(TemporaryFiles):
    def incomplete_records(self) -> list[dict[str, object]]:
        unreached_sink = sink(12, 10)
        unreached_sink["reached"] = False
        unreached_sink["source"] = -1
        return [
            {
                "net": "routed_net",
                "routed": True,
                "sources": [source(0)],
                "sinks": [sink(2, 0), sink(3, 1)],
                "edges": [edge(0, 1), edge(1, 2), edge(1, 3)],
            },
            {
                "net": "unrouted_net",
                "routed": False,
                "sources": [source(10), source(11)],
                "sinks": [unreached_sink],
                "edges": [edge(10, 11)],
            },
        ]

    def test_reports_exact_totals_and_raw_sha256(self) -> None:
        path = self.write_jsonl("summary.jsonl", self.incomplete_records())
        summary = SUMMARIZER.summarize(path)
        self.assertEqual(
            {
                key: summary[key]
                for key in (
                    "route_requests",
                    "routed",
                    "unrouted",
                    "sources",
                    "sinks",
                    "reached_sinks",
                    "edges",
                )
            },
            {
                "route_requests": 2,
                "routed": 1,
                "unrouted": 1,
                "sources": 3,
                "sinks": 3,
                "reached_sinks": 2,
                "edges": 4,
            },
        )
        self.assertEqual(
            summary["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def test_require_all_routed_cli_rejects_incomplete_routes(self) -> None:
        path = self.write_jsonl("incomplete.jsonl", self.incomplete_records())
        argv = ["summarize_routes_jsonl.py", str(path), "--require-all-routed"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(
                RuntimeError, r"unrouted=1, reached_sinks=2/3"
            ):
                SUMMARIZER.main()

    def test_rejects_non_boolean_completion_fields(self) -> None:
        record = self.incomplete_records()[0]
        record["routed"] = "true"
        path = self.write_jsonl("bad-routed-type.jsonl", [record])
        with self.assertRaisesRegex(RuntimeError, "expected a boolean"):
            SUMMARIZER.summarize(path)


class RocpdRouteSummaryValidationTest(TemporaryFiles):
    def complete_summary(self) -> dict[str, int]:
        return {
            "route_requests": 27_960,
            "routed": 27_960,
            "unrouted": 0,
            "sinks": 31_415,
            "reached_sinks": 31_415,
        }

    def write_summary(self, name: str, value: object) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def test_completed_routes_accepts_complete_strict_summary(self) -> None:
        path = self.write_summary("complete.json", self.complete_summary())
        self.assertEqual(ANALYZER.completed_routes_from_summary(path), 27_960)

    def test_completed_routes_rejects_incomplete_counts(self) -> None:
        mutations = {
            "routed": {"routed": 27_959},
            "unrouted": {"unrouted": 1},
            "sinks": {"reached_sinks": 31_414},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                summary = self.complete_summary()
                summary.update(mutation)
                path = self.write_summary(f"incomplete-{label}.json", summary)
                with self.assertRaisesRegex(RuntimeError, "summary is incomplete"):
                    ANALYZER.completed_routes_from_summary(path)

    def test_completed_routes_rejects_non_integer_and_boolean_counts(self) -> None:
        for index, invalid in enumerate((True, 27_960.0, "27960")):
            with self.subTest(value=invalid):
                summary: dict[str, object] = self.complete_summary()
                summary["route_requests"] = invalid
                path = self.write_summary(f"invalid-count-{index}.json", summary)
                with self.assertRaisesRegex(RuntimeError, "nonnegative integer"):
                    ANALYZER.completed_routes_from_summary(path)

    def test_completed_routes_rejects_malformed_json(self) -> None:
        path = self.directory / "malformed.json"
        path.write_text("{not JSON\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "invalid route-summary JSON"):
            ANALYZER.completed_routes_from_summary(path)

    def test_completed_routes_rejects_non_object_json(self) -> None:
        path = self.write_summary("array.json", [])
        with self.assertRaisesRegex(RuntimeError, "must be a JSON object"):
            ANALYZER.completed_routes_from_summary(path)

    def test_delta_marker_cardinality_allows_routes_without_sssp_queries(self) -> None:
        ANALYZER.validate_delta_marker_cardinality(
            {
                "pathfinder.route_net": {"calls": 500},
                "delta_step.generic": {"calls": 497},
                "delta_step.compact_edge_path_extraction": {"calls": 497},
            },
            500,
        )

    def test_delta_marker_cardinality_rejects_dropped_or_mismatched_ranges(self) -> None:
        cases = {
            "route-net": {
                "pathfinder.route_net": {"calls": 499},
                "delta_step.generic": {"calls": 497},
                "delta_step.compact_edge_path_extraction": {"calls": 497},
            },
            "delta": {
                "pathfinder.route_net": {"calls": 500},
                "delta_step.generic": {"calls": 497},
                "delta_step.compact_edge_path_extraction": {"calls": 496},
            },
        }
        for label, markers in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "cardinality is incomplete"):
                    ANALYZER.validate_delta_marker_cardinality(markers, 500)

    def test_analyzer_rejects_multi_process_trace_before_scope_queries(self) -> None:
        database = self.directory / "multi-process.db"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE processes "
                "(pid INTEGER, start INTEGER, end INTEGER, command TEXT)"
            )
            connection.executemany(
                "INSERT INTO processes VALUES (?, ?, ?, ?)",
                ((100, 0, 10, "first"), (200, 0, 10, "second")),
            )
            for table in ("regions", "kernels", "memory_copies", "region_args"):
                connection.execute(f"CREATE TABLE {table} (dummy INTEGER)")
        connection.close()
        opened_connections: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            connection = real_connect(*args, **kwargs)
            opened_connections.append(connection)
            return connection

        with mock.patch.object(
            ANALYZER.sqlite3, "connect", side_effect=tracked_connect
        ):
            try:
                with self.assertRaisesRegex(RuntimeError, "single-process trace"):
                    ANALYZER.analyze(database)
            finally:
                for connection in opened_connections:
                    connection.close()


class RouteSpliceValidationTest(TemporaryFiles):
    ARTIFACT_PAIR_ID = "0123456789abcdef0123456789abcdef"

    def route_record(
        self,
        net: str,
        marker: str,
        artifact_pair_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "artifact_pair_id": artifact_pair_id or self.ARTIFACT_PAIR_ID,
            "net": net,
            "marker": marker,
            "routed": True,
            "sources": [],
            "sinks": [],
            "edges": [],
        }

    def fixture_paths(self) -> tuple[Path, Path, Path]:
        reference = self.write_jsonl(
            "reference.jsonl",
            [
                self.route_record("net_a", "reference-a"),
                self.route_record("net_b", "reference-b"),
            ],
        )
        replacements = self.write_jsonl(
            "replacements.jsonl",
            [self.route_record("net_b", "replacement-b")],
        )
        return reference, replacements, self.directory / "spliced.jsonl"

    def test_splices_one_of_two_records_in_reference_order(self) -> None:
        reference, replacements, output = self.fixture_paths()
        summary = SPLICER.splice_routes(reference, replacements, output, 1)
        output_records = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["net"] for record in output_records], ["net_a", "net_b"])
        self.assertEqual(
            [record["marker"] for record in output_records],
            ["reference-a", "replacement-b"],
        )
        self.assertEqual(summary["reference"]["records"], 2)
        self.assertEqual(summary["replacements"]["records"], 1)
        self.assertEqual(summary["output"]["records"], 2)
        self.assertEqual(
            summary["reference"]["sha256"],
            hashlib.sha256(reference.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            summary["replacements"]["sha256"],
            hashlib.sha256(replacements.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            summary["output"]["sha256"],
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        self.assertEqual(summary["artifact_pair_id"], self.ARTIFACT_PAIR_ID)
        self.assertEqual(
            summary["line_policy"], "original_record_text_with_lf_terminator"
        )

    def test_rejects_duplicate_net(self) -> None:
        reference = self.write_jsonl(
            "duplicate-reference.jsonl",
            [
                self.route_record("net_a", "first"),
                self.route_record("net_a", "second"),
            ],
        )
        replacements = self.write_jsonl(
            "replacement.jsonl", [self.route_record("net_a", "replacement")]
        )
        output = self.directory / "duplicate-output.jsonl"
        with self.assertRaisesRegex(RuntimeError, "duplicate reference net"):
            SPLICER.splice_routes(reference, replacements, output, 1)
        self.assertFalse(output.exists())

    def test_rejects_replacement_net_missing_from_reference(self) -> None:
        reference, _, output = self.fixture_paths()
        replacements = self.write_jsonl(
            "missing-net.jsonl", [self.route_record("net_c", "replacement-c")]
        )
        with self.assertRaisesRegex(RuntimeError, "absent from the reference"):
            SPLICER.splice_routes(reference, replacements, output, 1)
        self.assertFalse(output.exists())

    def test_rejects_wrong_replacement_count(self) -> None:
        reference, replacements, output = self.fixture_paths()
        with self.assertRaisesRegex(RuntimeError, "replacement count"):
            SPLICER.splice_routes(reference, replacements, output, 2)
        self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        reference, replacements, output = self.fixture_paths()
        output.write_text("sentinel\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            SPLICER.splice_routes(reference, replacements, output, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_rejects_empty_input_and_empty_net(self) -> None:
        reference, replacements, output = self.fixture_paths()
        empty = self.directory / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "routes JSONL is empty"):
            SPLICER.splice_routes(empty, replacements, output, 1)

        bad_reference = self.write_jsonl(
            "empty-net.jsonl", [self.route_record("   ", "empty-net")]
        )
        with self.assertRaisesRegex(RuntimeError, "net must be a nonempty string"):
            SPLICER.splice_routes(bad_reference, replacements, output, 1)
        self.assertFalse(output.exists())

    def test_rejects_artifact_pair_mismatch(self) -> None:
        reference, _, output = self.fixture_paths()
        replacements = self.write_jsonl(
            "mismatched-pair.jsonl",
            [
                self.route_record(
                    "net_b",
                    "replacement-b",
                    artifact_pair_id="fedcba9876543210fedcba9876543210",
                )
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "does not match reference"):
            SPLICER.splice_routes(reference, replacements, output, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
