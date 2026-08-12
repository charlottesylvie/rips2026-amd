#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import analyze  # noqa: E402


class AnalyzeTest(unittest.TestCase):
    fixtures = HERE / "fixtures"

    def test_runtime_trace_union_concurrency_and_queries(self) -> None:
        trace = analyze.load_csv_trace(self.fixtures)
        summary = analyze.trace_summary(trace)
        self.assertEqual(summary["kernel_interval_union_ns"], 220)
        self.assertEqual(summary["additive_multi_queue_kernel_ns"], 320)
        self.assertEqual(summary["concurrency_ns"], {"0": 20, "1": 120, "2": 100})
        self.assertEqual(summary["per_query"]["7"]["kernels"], 2)
        self.assertEqual(summary["hip_stream_synchronize"]["count"], 2)
        self.assertEqual(summary["memory_copies"]["classification_counts"]["graph_upload"], 1)

    def test_kib_traffic_unit_is_not_misread_as_bytes(self) -> None:
        rows = analyze.load_counter_rows(self.fixtures)
        summary = analyze.counter_summary(self.fixtures, rows, wave_size=32)
        self.assertEqual(summary["traffic_units"]["FETCH_SIZE"]["unit"], "KiB")
        relaxation = summary["families"]["segmented_relaxation"]
        self.assertEqual(relaxation["read_request_bytes_per_second"], 40_960_000_000.0)
        self.assertEqual(relaxation["wave_size"], 32)

    def test_unknown_traffic_unit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unit is unresolved"):
            analyze.infer_traffic_unit("FETCH_SIZE", 7, 1_048_576, None)

    def test_negative_duration_is_rejected(self) -> None:
        row = {
            "Kernel_Name": "segmented_frontier_relax_kernel",
            "Start_Timestamp": "20",
            "End_Timestamp": "10",
        }
        with self.assertRaisesRegex(ValueError, "negative duration"):
            analyze.interval_from_row(row, ("Kernel_Name",))

    def test_impossible_stall_percentage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "impossible"):
            analyze.validated_percentage(101, 100, "stall")

    def test_compute_report_is_absent_without_profile(self) -> None:
        self.assertEqual(
            analyze.rocprof_compute_summary(self.fixtures),
            {"status": "not_collected"},
        )

    def test_captured_metric_formula_uses_only_supplied_counters(self) -> None:
        self.assertEqual(
            analyze.evaluate_metric_formula(
                "100 * SQ_INSTS_VALU / (SQ_BUSY_CYCLES * wave_size)",
                {"SQ_INSTS_VALU": 320, "SQ_BUSY_CYCLES": 20, "wave_size": 32},
            ),
            50.0,
        )
        with self.assertRaises(KeyError):
            analyze.evaluate_metric_formula("MISSING / 2", {})


if __name__ == "__main__":
    unittest.main()
