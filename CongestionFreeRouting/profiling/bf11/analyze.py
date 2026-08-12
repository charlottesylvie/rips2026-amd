#!/usr/bin/env python3
"""Strict BF11 analyzer for rocprofv3 CSV/RocPD, counters, and ATT output."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sqlite3
from statistics import mean
from typing import Any, Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trace_utils import (  # noqa: E402
    Interval,
    concurrency_distribution,
    first_column,
    interval_union,
    percentile,
    validated_percentage,
)


FAMILY_PATTERNS = (
    ("reset", re.compile(r"clear_(?:touched_)?state|initialize_state", re.I)),
    ("seed", re.compile(r"seed_sources", re.I)),
    ("segmented_relaxation", re.compile(r"segmented_frontier_relax", re.I)),
    ("cooperative_controller", re.compile(r"frontier_controller", re.I)),
    ("target_check", re.compile(r"update_target_status|finalize_segment_round", re.I)),
    ("prefix", re.compile(r"prefix_target_paths", re.I)),
    ("reconstruction", re.compile(r"summarize_target_paths|materialize_target_paths", re.I)),
    ("runtime_copy", re.compile(r"copyBuffer", re.I)),
    ("runtime_fill", re.compile(r"fillBuffer|fill_cost|sparse_cost_update", re.I)),
)


@dataclass
class Trace:
    kernels: list[Interval] = field(default_factory=list)
    apis: list[Interval] = field(default_factory=list)
    copies: list[Interval] = field(default_factory=list)
    markers: list[Interval] = field(default_factory=list)


def classify_kernel(name: str) -> str:
    for family, pattern in FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "other"


def parse_int(row: dict[str, str], names: Iterable[str], default: int | None = None) -> int:
    column = first_column(row, names)
    if column is None or row.get(column, "") == "":
        if default is None:
            raise ValueError(f"missing required column from {tuple(names)}")
        return default
    value = int(float(row[column]))
    if value < 0:
        raise ValueError(f"negative value in {column}: {value}")
    return value


def parse_name(row: dict[str, str], names: Iterable[str]) -> str:
    column = first_column(row, names)
    return row.get(column, "") if column else ""


def interval_from_row(row: dict[str, str], name_columns: Iterable[str]) -> Interval:
    start = parse_int(row, ("Start_Timestamp", "start", "start_ns"))
    end = parse_int(row, ("End_Timestamp", "end", "end_ns"))
    interval = Interval(
        start,
        end,
        parse_name(row, name_columns),
        parse_name(row, ("Queue_Id", "queue_id", "Stream_Id")),
        parse_name(row, ("Dispatch_Id", "dispatch_id")),
        parse_name(row, ("Agent_Id", "agent_id", "device")),
        row,
    )
    interval.validate()
    return interval


def load_csv_trace(root: Path) -> Trace:
    trace = Trace()
    for path in root.rglob("*.csv"):
        lower = path.name.lower()
        kind: str | None = None
        names: tuple[str, ...] = ()
        if "kernel_trace" in lower:
            kind, names = "kernels", ("Kernel_Name", "kernel_name", "name")
        elif "hip_api_trace" in lower:
            kind, names = "apis", ("Function", "name")
        elif "memory_copy_trace" in lower:
            kind, names = "copies", ("Direction", "name")
        elif "marker_api_trace" in lower:
            kind, names = "markers", ("Function", "message", "name")
        if kind is None:
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                getattr(trace, kind).append(interval_from_row(row, names))
    for values in (trace.kernels, trace.apis, trace.copies, trace.markers):
        values.sort(key=lambda item: (item.start_ns, item.end_ns))
    return trace


def load_rocpd(path: Path) -> Trace:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    present = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    required = {"kernels", "regions", "memory_copies"}
    if missing := required - present:
        raise ValueError(f"RocPD is missing views: {', '.join(sorted(missing))}")
    trace = Trace()
    kernel_columns = {row[1] for row in connection.execute("PRAGMA table_info(kernels)")}
    queue_column = next((name for name in ("queueId", "queue_id", "queue") if name in kernel_columns), None)
    dispatch_column = next((name for name in ("dispatchId", "dispatch_id", "id") if name in kernel_columns), None)
    queue_expr = queue_column or "''"
    dispatch_expr = dispatch_column or "''"
    for row in connection.execute(
        f"SELECT name,start,end,{queue_expr} AS queue,{dispatch_expr} AS dispatch "
        "FROM kernels ORDER BY start"
    ):
        trace.kernels.append(
            Interval(int(row["start"]), int(row["end"]), str(row["name"]),
                     str(row["queue"]), str(row["dispatch"]))
        )
    for row in connection.execute(
        "SELECT name,category,start,end,extdata FROM regions ORDER BY start"
    ):
        name = str(row["name"])
        category = str(row["category"])
        if category == "MARKER_CORE_RANGE_API":
            try:
                name = str(json.loads(row["extdata"] or "{}").get("message", name))
            except json.JSONDecodeError:
                pass
            trace.markers.append(Interval(int(row["start"]), int(row["end"]), name))
        elif category.startswith("HIP_"):
            trace.apis.append(Interval(int(row["start"]), int(row["end"]), name))
    columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_copies)")}
    name_column = next(
        (name for name in ("name", "direction", "copyDirection") if name in columns),
        None,
    )
    if name_column is None:
        raise ValueError("RocPD memory_copies view has no direction/name column")
    size_column = next(
        (name for name in ("bytes", "size", "sizeBytes", "size_bytes") if name in columns),
        None,
    )
    size_expr = f'"{size_column}"' if size_column else "NULL"
    for row in connection.execute(
        f'SELECT "{name_column}" AS name,start,end,{size_expr} AS bytes '
        "FROM memory_copies ORDER BY start"
    ):
        fields = {}
        if row["bytes"] is not None:
            fields["Bytes"] = str(row["bytes"])
        trace.copies.append(
            Interval(int(row["start"]), int(row["end"]), str(row["name"]),
                     fields=fields)
        )
    for values in (trace.kernels, trace.apis, trace.copies, trace.markers):
        for interval in values:
            interval.validate()
    connection.close()
    return trace


QUERY_RE = re.compile(
    r"bf11\.query net=(?P<net>\d+|unknown) worker=(?P<worker>\d+|unknown) "
    r"controller=(?P<controller>\w+) attempt=(?P<attempt>\w+)"
)


def query_markers(trace: Trace) -> list[tuple[Interval, dict[str, str]]]:
    result = []
    for marker in trace.markers:
        if match := QUERY_RE.fullmatch(marker.name):
            result.append((marker, match.groupdict()))
    return result


def containing_query(
    interval: Interval,
    queries: list[tuple[Interval, dict[str, str]]],
) -> dict[str, str] | None:
    candidates = [
        (marker.duration_ns, fields)
        for marker, fields in queries
        if marker.start_ns <= interval.start_ns and interval.end_ns <= marker.end_ns
    ]
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def trace_summary(trace: Trace) -> dict[str, Any]:
    if not trace.kernels:
        raise ValueError("trace contains no kernel dispatches")
    span = (min(item.start_ns for item in trace.kernels),
            max(item.end_ns for item in trace.kernels))
    union_ns, islands = interval_union((item.start_ns, item.end_ns) for item in trace.kernels)
    concurrency = concurrency_distribution(
        ((item.start_ns, item.end_ns) for item in trace.kernels), span=span
    )
    for level in range(0, max(concurrency, default=0) + 1):
        concurrency.setdefault(level, 0)
    concurrency = dict(sorted(concurrency.items()))
    additive = sum(item.duration_ns for item in trace.kernels)
    families: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "additive_ns": 0})
    for kernel in trace.kernels:
        family = classify_kernel(kernel.name)
        families[family]["count"] += 1
        families[family]["additive_ns"] += kernel.duration_ns

    api_counts = Counter(item.name for item in trace.apis)
    sync = sorted(item.duration_ns for item in trace.apis if item.name == "hipStreamSynchronize")
    copy_by_direction: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "engine_ns": 0, "payload_sizes": []}
    )
    copy_classes = Counter()
    queries = query_markers(trace)
    first_query_start = min((marker.start_ns for marker, _ in queries), default=None)
    for copy in trace.copies:
        record = copy_by_direction[copy.name or "unknown"]
        record["count"] += 1
        record["engine_ns"] += copy.duration_ns
        size_column = first_column(copy.fields, ("Bytes", "Size", "sizeBytes"))
        if size_column and copy.fields.get(size_column):
            record["payload_sizes"].append(int(copy.fields[size_column]))
        size = int(copy.fields[size_column]) if size_column and copy.fields.get(size_column) else None
        query = containing_query(copy, queries)
        direction = copy.name.upper()
        host_to_device = "HOST_TO_DEVICE" in direction or "H2D" in direction
        device_to_host = "DEVICE_TO_HOST" in direction or "D2H" in direction
        if query and (host_to_device or (size is not None and size <= 256)):
            copy_classes["per_query_control"] += 1
        elif query and device_to_host:
            copy_classes["extraction"] += 1
        elif (first_query_start is not None and copy.end_ns <= first_query_start
              and host_to_device):
            copy_classes["graph_upload"] += 1
        elif "PROF" in direction:
            copy_classes["profiler_only"] += 1
        else:
            copy_classes["other_or_unresolved"] += 1

    per_query: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kernels": 0, "kernel_ns": 0, "kernel_families": Counter(),
                 "hip_api": 0, "copies": 0,
                 "synchronizations": 0, "controllers": set(), "attempts": set()}
    )
    for domain, intervals in (("kernels", trace.kernels), ("apis", trace.apis), ("copies", trace.copies)):
        for interval in intervals:
            fields = containing_query(interval, queries)
            if not fields or fields["net"] == "unknown":
                continue
            record = per_query[fields["net"]]
            record["controllers"].add(fields["controller"])
            record["attempts"].add(fields["attempt"])
            if domain == "kernels":
                record["kernels"] += 1
                record["kernel_ns"] += interval.duration_ns
                record["kernel_families"][classify_kernel(interval.name)] += 1
            elif domain == "apis":
                record["hip_api"] += 1
                if interval.name == "hipStreamSynchronize":
                    record["synchronizations"] += 1
            else:
                record["copies"] += 1
    for record in per_query.values():
        record["controllers"] = sorted(record["controllers"])
        record["attempts"] = sorted(record["attempts"])
        record["kernel_families"] = dict(record["kernel_families"])

    marker_counts = Counter(marker.name.split()[0] for marker in trace.markers)
    graph_launches = api_counts.get("hipGraphLaunch", 0)
    graph_uploads = sum(api_counts[name] for name in api_counts if "GraphInstantiate" in name or "GraphUpload" in name)
    return {
        "span_ns": span[1] - span[0],
        "kernel_interval_union_ns": union_ns,
        "kernel_interval_union_islands": islands,
        "additive_multi_queue_kernel_ns": additive,
        "gpu_active_duty_cycle_percent": validated_percentage(
            union_ns, span[1] - span[0], "GPU active duty cycle"
        ),
        "concurrency_ns": {str(key): value for key, value in concurrency.items()},
        "kernel_families": dict(sorted(families.items())),
        "hip_api_counts": dict(api_counts),
        "hip_stream_synchronize": {
            "count": len(sync), "total_ns": sum(sync),
            "p50_ns": percentile(sync, 0.50), "p90_ns": percentile(sync, 0.90),
            "p95_ns": percentile(sync, 0.95), "p99_ns": percentile(sync, 0.99),
        },
        "memory_copies": {
            "by_direction": dict(copy_by_direction),
            "classification_counts": dict(copy_classes),
        },
        "per_query": dict(sorted(per_query.items(), key=lambda item: int(item[0]))),
        "hip_graph": {
            "launches": graph_launches, "uploads": graph_uploads,
            "direct_segments": marker_counts.get("bf11.segmented_controller", 0),
            "no_op_rounds": None,
            "fallbacks": None,
        },
    }


def read_agent_metadata(root: Path) -> dict[str, Any]:
    cache_sizes = []
    for rocminfo_path in root.rglob("rocminfo.txt"):
        text = rocminfo_path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(
            r"(?im)^\s*(?:Cache\s+Size|Size)\s*:\s*(\d+)\s*(KB|KiB|MB|MiB|B)\b",
            text,
        ):
            value = int(match.group(1))
            unit = match.group(2).lower()
            scale = 1 if unit == "b" else (1024 if unit in {"kb", "kib"} else 1024 * 1024)
            cache_sizes.append(value * scale)
        break
    current_state = []
    for path in root.rglob("gpu-state.txt"):
        current_state.extend(
            line.strip() for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if re.search(r"clock|temp|memory|vram", line, re.I)
        )
    gpu_records = []
    candidates = sorted(root.rglob("*agent_info.csv"))
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("Agent_Type", "").upper() != "GPU":
                    continue
                gpu_records.append({
                    "architecture": row.get("Name") or row.get("Gfx_Target_Version"),
                    "cu_count": int(row["Cu_Count"]),
                    "wave_size": int(row["Wave_Front_Size"]),
                    "max_waves_per_cu": int(row.get("Max_Waves_Per_Cu") or 0),
                    "lds_size_kib": int(row.get("Lds_Size_In_Kb") or 0),
                    "max_engine_clock_mhz": int(
                        row.get("Max_Engine_Clk_Fcompute") or 0
                    ),
                    "cache_sizes_bytes": cache_sizes,
                    "source": str(path),
                })
    if gpu_records:
        identities = {
            (record["architecture"], record["cu_count"], record["wave_size"],
             record["max_waves_per_cu"], record["max_engine_clock_mhz"])
            for record in gpu_records
        }
        if len(identities) != 1:
            raise ValueError(f"captured GPU agent metadata is inconsistent: {identities}")
        result = gpu_records[0]
        compute_sysinfo = []
        sysinfo_fields = (
            "gpu_arch", "gpu_model", "cu_per_gpu", "wave_size",
            "gpu_l1", "gpu_l2", "max_sclk", "max_mclk",
            "cur_sclk", "cur_mclk", "gpu_memory",
        )
        for path in sorted(root.rglob("sysinfo.csv")):
            with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                for row in csv.DictReader(stream):
                    captured = {name: row.get(name, "") for name in sysinfo_fields}
                    captured["source"] = str(path)
                    compute_sysinfo.append(captured)
        for captured in compute_sysinfo:
            if (captured["gpu_arch"] and
                    captured["gpu_arch"] != result["architecture"]):
                raise ValueError("agent and rocprof-compute architecture disagree")
            if (captured["cu_per_gpu"] and
                    int(captured["cu_per_gpu"]) != result["cu_count"]):
                raise ValueError("agent and rocprof-compute CU count disagree")
            if (captured["wave_size"] and
                    int(captured["wave_size"]) != result["wave_size"]):
                raise ValueError("agent and rocprof-compute wave size disagree")
        result["sources"] = [record["source"] for record in gpu_records]
        result["captured_clock_temperature_memory_lines"] = current_state
        result["rocprof_compute_sysinfo"] = compute_sysinfo
        return result
    raise ValueError("captured GPU agent metadata is required")


@dataclass(frozen=True)
class CounterRow:
    replay: str
    key: tuple[str, str, str, str, str]
    counter: str
    value: float
    start_ns: int
    end_ns: int
    fields: dict[str, str]


def load_counter_rows(root: Path) -> list[CounterRow]:
    rows = []
    for path in root.rglob("*counter_collection.csv"):
        replay = str(path.parent.relative_to(root))
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                counter = parse_name(raw, ("Counter_Name", "counter"))
                if not counter:
                    continue
                start = parse_int(raw, ("Start_Timestamp", "start"))
                end = parse_int(raw, ("End_Timestamp", "end"))
                if end <= start:
                    raise ValueError(f"nonpositive counter interval in {path}")
                kernel = parse_name(raw, ("Kernel_Name", "kernel"))
                key = (
                    parse_name(raw, ("Agent_Id", "device")),
                    parse_name(raw, ("Dispatch_Id", "dispatch_id")),
                    parse_name(raw, ("Queue_Id", "queue")), kernel, replay,
                )
                rows.append(CounterRow(
                    replay, key, counter,
                    float(parse_name(raw, ("Counter_Value", "value"))),
                    start, end, raw
                ))
    return rows


def counter_definitions(root: Path) -> str:
    candidates = set(root.rglob("counter-definitions.txt"))
    candidates.update(root.rglob("rocprof-compute-*metrics*.txt"))
    candidates.update(root.rglob("rocprof-compute-*catalog*.txt"))
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(candidates)
    )


def metric_unit(name: str, definitions: str) -> str | None:
    pattern = re.compile(
        rf"(?is)\b{re.escape(name)}\b.{{0,500}}?\bunit\s*[:=]\s*([A-Za-z0-9_/%.-]+)"
    )
    return match.group(1) if (match := pattern.search(definitions)) else None


def read_stream_reference(root: Path) -> dict[str, Any] | None:
    for path in root.rglob("reference.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "bf11_streaming_reference":
                return record
    return None


def read_json_records(root: Path, record_type: str) -> list[dict[str, Any]]:
    records = []
    for path in root.rglob("*.log"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == record_type:
                records.append(record)
    return records


def read_json_records_with_paths(
    root: Path, record_type: str
) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in root.rglob("*.log"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == record_type:
                records.append((path, record))
    return records


def infer_traffic_unit(
    counter: str,
    measured: float,
    expected_bytes: float,
    metadata_unit: str | None,
) -> tuple[str, float]:
    if measured <= 0 or expected_bytes <= 0:
        raise ValueError(f"cannot establish {counter} units from nonpositive data")
    normalized = (metadata_unit or "").lower()
    candidates = []
    if "kib" in normalized or normalized in {"kb", "kilobytes"}:
        candidates.append(("KiB", 1024.0))
    elif "byte" in normalized:
        candidates.append(("bytes", 1.0))
    elif "request" in normalized:
        candidates.append(("requests", math.nan))
    else:
        candidates.extend((("bytes", 1.0), ("KiB", 1024.0)))
    matches = []
    for unit, scale in candidates:
        if math.isnan(scale):
            continue
        ratio = measured * scale / expected_bytes
        if 0.75 <= ratio <= 1.25:
            matches.append((unit, scale))
    if len(matches) != 1:
        raise ValueError(
            f"{counter} unit is unresolved: value={measured}, expected={expected_bytes}, "
            f"metadata_unit={metadata_unit!r}"
        )
    return matches[0]


def evaluate_metric_formula(expression: str, values: dict[str, float]) -> float:
    """Evaluate captured arithmetic without executing profiler-supplied code."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return float(values[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.lower() in {"min", "max", "sum"}
                and not node.keywords):
            arguments = [visit(argument) for argument in node.args]
            if node.func.id.lower() == "min":
                return min(arguments)
            if node.func.id.lower() == "max":
                return max(arguments)
            return sum(arguments)
        raise ValueError(f"unsupported formula syntax: {ast.dump(node)}")

    result = visit(tree)
    if not math.isfinite(result):
        raise ValueError("metric formula produced a non-finite result")
    return result


def validate_replays(
    root: Path, rows: list[CounterRow], tolerance: float
) -> dict[str, Any]:
    by_replay: dict[str, list[CounterRow]] = defaultdict(list)
    for row in rows:
        by_replay[row.replay].append(row)
    fingerprints = {}
    family_counts: dict[str, list[int]] = defaultdict(list)
    family_durations = defaultdict(list)
    for replay, values in by_replay.items():
        dispatches = {(row.key[:4], row.start_ns, row.end_ns) for row in values}
        counts = Counter(classify_kernel(key[0][3]) for key in dispatches)
        durations = defaultdict(int)
        for key, start, end in dispatches:
            durations[classify_kernel(key[3])] += end - start
        fingerprints[replay] = {
            "dispatch_count": len(dispatches), "family_counts": dict(counts),
            "duration_ns": dict(durations),
        }
        for family, count in counts.items():
            family_counts[family].append(count)
        for family, duration in durations.items():
            family_durations[family].append(duration)
    for family, counts in family_counts.items():
        if len(set(counts)) != 1:
            raise ValueError(
                f"counter replays have inconsistent {family} dispatch counts: {counts}"
            )
    variance = {}
    for family, values in family_durations.items():
        if len(values) < 2 or mean(values) == 0:
            variance[family] = 0.0
            continue
        spread = (max(values) - min(values)) / mean(values)
        variance[family] = spread
        if spread > tolerance:
            raise ValueError(
                f"counter replay variance for {family} is {spread:.1%}, above {tolerance:.1%}"
            )
    query_counts = {}
    for replay in by_replay:
        routes = root / replay / "routes.jsonl"
        if routes.is_file():
            query_counts[replay] = sum(
                bool(line.strip())
                for line in routes.read_text(encoding="utf-8", errors="replace").splitlines()
            )
    if query_counts and len(query_counts) != len(by_replay):
        missing = sorted(set(by_replay) - set(query_counts))
        raise ValueError(f"counter replays are missing route/query counts: {missing}")
    if len(set(query_counts.values())) > 1:
        raise ValueError(f"counter replays have inconsistent query counts: {query_counts}")
    return {
        "replays": fingerprints,
        "relative_duration_spread": variance,
        "query_counts": query_counts,
    }


def counter_summary(root: Path, rows: list[CounterRow], wave_size: int) -> dict[str, Any]:
    if not rows:
        return {"status": "not_collected"}
    definitions = counter_definitions(root)
    application_rows = [row for row in rows if "streaming_copy" not in row.key[3]]
    if not application_rows:
        return {"status": "reference_only"}
    replays = validate_replays(root, application_rows, 0.15)
    grouped: dict[tuple[str, str, str, str, str], dict[str, CounterRow]] = defaultdict(dict)
    for row in application_rows:
        if row.counter in grouped[row.key]:
            raise ValueError(f"ambiguous duplicate counter {row.counter} for {row.key}")
        grouped[row.key][row.counter] = row
    expected_by_replay: dict[str, set[str]] = defaultdict(set)
    for key, counters in grouped.items():
        expected_by_replay[key[4]].update(counters)
    for key, counters in grouped.items():
        missing = expected_by_replay[key[4]] - set(counters)
        if missing:
            raise ValueError(
                f"incomplete counter join for {key}: missing {sorted(missing)}"
            )
    reference = read_stream_reference(root)
    family_replay_durations: dict[str, Counter[str]] = defaultdict(Counter)
    metric_replay_values: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    metric_replay_durations: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    observations: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    resources: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    zeros_invalid = {
        "VALUInsts", "SALUInsts", "SQ_INSTS_VALU", "SQ_WAVE_CYCLES",
        "Wavefronts", "SQ_WAVES",
    }
    for key, counters in grouped.items():
        interval_pairs = {(row.start_ns, row.end_ns) for row in counters.values()}
        if len(interval_pairs) != 1:
            raise ValueError(
                f"mismatched counter dispatch intervals for {key}: "
                f"{sorted(interval_pairs)}"
            )
        family = classify_kernel(key[3])
        duration = next(iter(counters.values())).end_ns - next(iter(counters.values())).start_ns
        family_replay_durations[family][key[4]] += duration
        for name, row in counters.items():
            if name in zeros_invalid and row.value == 0 and duration > 0:
                raise ValueError(f"zero-invalid {name} for working dispatch {key}")
            metric_replay_values[family][name][key[4]] += row.value
            metric_replay_durations[family][name][key[4]] += (
                row.end_ns - row.start_ns
            )
            observations[family][name].append((row.value, row.end_ns - row.start_ns))
            for sources, label in (
                (("Grid_Size",), "grid_size"),
                (("Workgroup_Size",), "workgroup_size"),
                (("VGPR_Count", "Arch_VGPR", "Accum_VGPR"), "vgprs"),
                (("SGPR_Count", "SGPR"), "sgprs"),
                (("LDS_Block_Size", "LDS_Per_Workgroup"), "lds_bytes"),
                (("Scratch_Size", "Scratch_Per_Workitem"), "scratch_bytes"),
                (("Wave_Size",), "wave_size"),
            ):
                column = first_column(row.fields, sources)
                if column and row.fields.get(column) != "":
                    resources[family][label].add(row.fields[column])
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    durations = Counter()
    for family, metrics in metric_replay_values.items():
        for name, replay_values in metrics.items():
            by_family[family][name] = mean(replay_values.values())
        durations[family] = mean(family_replay_durations[family].values())
    totals = Counter()
    for counters in by_family.values():
        totals.update(counters)
    traffic_units: dict[str, Any] = {}
    streaming_rates: dict[str, float] = {}
    if any(name in totals for name in ("FETCH_SIZE", "WRITE_SIZE")):
        if reference is None:
            raise ValueError("FETCH_SIZE/WRITE_SIZE require a captured streaming reference")
        # Reference counter rows are recognized by streaming_copy; each metric
        # is validated independently against its known read/write payload.
        for counter, expected_key in (("FETCH_SIZE", "unique_read_bytes"),
                                      ("WRITE_SIZE", "unique_write_bytes")):
            measured = sum(
                row.value for row in rows
                if row.counter == counter and "streaming_copy" in row.key[3]
            )
            if counter in totals and measured == 0:
                raise ValueError(f"no streaming-reference row found for {counter}")
            if measured:
                unit, scale = infer_traffic_unit(
                    counter, measured, float(reference[expected_key]),
                    metric_unit(counter, definitions)
                )
                traffic_units[counter] = {"unit": unit, "bytes_per_unit": scale}
                reference_rows = [
                    row for row in rows
                    if row.counter == counter and "streaming_copy" in row.key[3]
                ]
                reference_duration_ns = sum(
                    row.end_ns - row.start_ns for row in reference_rows
                )
                streaming_rates[counter] = (
                    measured * scale / (reference_duration_ns * 1e-9)
                )
    telemetry = read_json_records(root, "bf11_query_telemetry")
    telemetry_denominators = {
        "queries": len({record.get("net_index") for record in telemetry
                        if record.get("net_index") is not None}),
        "frontier_vertices": sum(record.get("frontier_vertices_processed", 0)
                                 for record in telemetry),
        "edges_examined": sum(record.get("edges_examined", 0) for record in telemetry),
        "successful_relaxations": sum(record.get("successful_relaxations", 0)
                                      for record in telemetry),
    }
    families = {}
    for family, counters in sorted(by_family.items()):
        duration = durations[family]
        metric_durations = {
            name: mean(replay_durations.values())
            for name, replay_durations in metric_replay_durations[family].items()
        }
        record: dict[str, Any] = {
            "additive_counter_replay_duration_ns": duration,
            "counter_interval_ns_by_metric": metric_durations,
            "counters": dict(counters),
        }
        record["resources"] = {
            label: sorted(values) for label, values in resources[family].items()
        }
        captured_wave_sizes = resources[family].get("wave_size", set())
        if len(captured_wave_sizes) > 1:
            raise ValueError(
                f"inconsistent wave sizes for {family}: {sorted(captured_wave_sizes)}"
            )
        kernel_wave_size = (
            int(next(iter(captured_wave_sizes))) if captured_wave_sizes else wave_size
        )
        if captured_wave_sizes and kernel_wave_size != wave_size:
            raise ValueError(
                f"kernel/agent wave-size mismatch for {family}: "
                f"{kernel_wave_size} versus {wave_size}"
            )
        record["wave_size"] = kernel_wave_size
        weighted = {}
        for name, values in observations[family].items():
            unit = (metric_unit(name, definitions) or "").lower()
            is_percentage = (
                any(token in name.lower()
                    for token in ("percent", "busy", "hitrate", "cachehit"))
                or unit in {"%", "percent", "percentage"}
            )
            if is_percentage:
                denominator = sum(item_duration for _, item_duration in values)
                value = sum(metric * item_duration for metric, item_duration in values) / denominator
                validated_percentage(value, 100.0, name)
                weighted[name] = value
        if weighted:
            record["duration_weighted_metrics"] = weighted
        read = counters.get("FETCH_SIZE")
        write = counters.get("WRITE_SIZE")
        if read is not None and "FETCH_SIZE" in traffic_units:
            read_bytes = read * traffic_units["FETCH_SIZE"]["bytes_per_unit"]
            record["read_request_bytes"] = read_bytes
            record["read_request_bytes_per_second"] = read_bytes / (
                metric_durations["FETCH_SIZE"] * 1e-9
            )
        if write is not None and "WRITE_SIZE" in traffic_units:
            write_bytes = write * traffic_units["WRITE_SIZE"]["bytes_per_unit"]
            record["write_request_bytes"] = write_bytes
            record["write_request_bytes_per_second"] = write_bytes / (
                metric_durations["WRITE_SIZE"] * 1e-9
            )
        if "read_request_bytes" in record or "write_request_bytes" in record:
            combined = record.get("read_request_bytes", 0.0) + record.get("write_request_bytes", 0.0)
            record["combined_request_bytes"] = combined
            record["combined_request_bytes_per_second"] = (
                record.get("read_request_bytes_per_second", 0.0) +
                record.get("write_request_bytes_per_second", 0.0)
            )
            reference_rate = sum(streaming_rates.values())
            if reference_rate > 0:
                record["percentage_of_streaming_request_reference"] = validated_percentage(
                    record["combined_request_bytes_per_second"], reference_rate,
                    f"{family} streaming-reference request rate",
                    allow_accumulation=True,
                )
            for label, denominator in telemetry_denominators.items():
                if denominator > 0:
                    record[f"request_bytes_per_{label}"] = combined / denominator
        raw_request_counters = {}
        for name, value in counters.items():
            if not re.search(r"(?:REQ(?:_|$)|REQUEST)", name, re.I):
                continue
            metric_duration = metric_durations.get(name, 0)
            request_record = {"total_requests": value}
            if metric_duration > 0:
                request_record["requests_per_second"] = value / (
                    metric_duration * 1e-9
                )
            for label, denominator in telemetry_denominators.items():
                if denominator > 0:
                    request_record[f"requests_per_{label}"] = value / denominator
            raw_request_counters[name] = request_record
        if raw_request_counters:
            record["raw_request_counters"] = raw_request_counters
        physical_metrics = {}
        for name, value in counters.items():
            if not re.search(r"GCEA|MEMORY_CONTROLLER", name, re.I):
                continue
            unit = (metric_unit(name, definitions) or "").lower()
            scale = 1024.0 if "kib" in unit else (1.0 if "byte" in unit else None)
            if scale is None:
                continue
            byte_value = value * scale
            physical_metrics[name] = {
                "bytes": byte_value,
                "bytes_per_second": byte_value / (metric_durations[name] * 1e-9),
                "basis": f"captured metric definition unit {unit}",
            }
        if physical_metrics:
            record["physical_memory_traffic"] = physical_metrics
        if "SQ_WAVE_CYCLES" in counters:
            record["wave_cycles"] = counters["SQ_WAVE_CYCLES"]
        waves = counters.get("Wavefronts") or counters.get("SQ_WAVES")
        if waves:
            record["launched_waves"] = waves
            for metric, label in (
                ("VALUInsts", "valu_instructions_per_wave"),
                ("SALUInsts", "salu_instructions_per_wave"),
                ("SFetchInsts", "scalar_memory_instructions_per_wave"),
                ("SQ_INSTS_VALU", "sq_valu_instructions_per_wave"),
                ("SQ_INSTS_SALU", "sq_salu_instructions_per_wave"),
                ("SQ_INSTS_SMEM", "sq_scalar_memory_instructions_per_wave"),
                ("SQ_INSTS_FLAT", "flat_instructions_per_wave"),
                ("SQ_INSTS_LDS", "lds_instructions_per_wave"),
                ("SQ_INSTS_TEX_LOAD", "vmem_load_instructions_per_wave"),
                ("SQ_INSTS_TEX_STORE", "vmem_store_instructions_per_wave"),
                ("SQ_INSTS_BRANCH", "branch_instructions_per_wave"),
                ("SQ_INSTS_CONTROL", "control_instructions_per_wave"),
            ):
                if metric in counters:
                    record[label] = counters[metric] / waves
        resident = {
            name: value for name, value in counters.items()
            if "RESIDENT" in name.upper() or "ACTIVE_WAVE" in name.upper()
        }
        if resident:
            record["active_or_resident_wave_metrics"] = resident
        if family in {"segmented_relaxation", "cooperative_controller"}:
            for name in ("MemUnitBusy", "WriteUnitStalled"):
                values = observations[family].get(name, [])
                if values and all(value == 0 for value, _ in values):
                    raise ValueError(
                        f"zero-invalid {name} cannot be interpreted as zero stalling"
                    )
        # The gfx1151 System-SOL VALU issue formula is intentionally not
        # reconstructed from names. It is only evaluated when a captured
        # metric definition explicitly exposes the required expression.
        formula_match = re.search(
            r"(?is)VALU.*issue.*(?:formula|expression)\s*[:=]\s*([^\n]+)",
            definitions,
        )
        if formula_match:
            formula = formula_match.group(1).strip()
            record["valu_issue_density_formula"] = formula
            formula_values = {name: float(value) for name, value in counters.items()}
            formula_values.update({"wave_size": float(kernel_wave_size),
                                   "WAVE_SIZE": float(kernel_wave_size)})
            try:
                issue_density = evaluate_metric_formula(formula, formula_values)
                record["valu_issue_density"] = validated_percentage(
                    issue_density, 100.0, "VALU issue density"
                )
                record["valu_issue_density_status"] = (
                    "evaluated from captured System-SOL formula and counters"
                )
            except (KeyError, SyntaxError, ValueError, ZeroDivisionError) as error:
                record["valu_issue_density_status"] = (
                    "captured formula could not be evaluated from this compatible pass: "
                    f"{error}"
                )
        else:
            record["valu_issue_density_status"] = "unavailable: captured System-SOL formula not present"
        families[family] = record
    aggregate_weighted = {}
    metric_names = {
        name for family_values in observations.values() for name in family_values
        if any(token in name.lower() for token in ("percent", "busy", "hitrate", "cachehit"))
    }
    for name in sorted(metric_names):
        values = [item for family_values in observations.values()
                  for item in family_values.get(name, [])]
        denominator = sum(duration for _, duration in values)
        if denominator:
            value = sum(metric * duration for metric, duration in values) / denominator
            validated_percentage(value, 100.0, f"aggregate {name}")
            aggregate_weighted[name] = value
    return {
        "status": "measured", "metric_definitions_captured": bool(definitions),
        "traffic_units": traffic_units, "replay_validation": replays,
        "streaming_reference_request_bytes_per_second": streaming_rates,
        "telemetry_denominators": telemetry_denominators,
        "duration_weighted_aggregate_metrics": aggregate_weighted,
        "families": families,
        "limitations": [
            "GL2 request traffic is cache-line/request traffic, not physical DRAM bandwidth.",
            "Physical memory traffic is reported only when a captured GCEA or memory-controller metric defines it.",
            "Occupancy and resident waves do not imply wave eligibility or issue efficiency.",
        ],
    }


COMPUTE_REPORT_PATTERNS = {
    "system_speed_of_light": re.compile(r"system.*(?:speed.?of.?light|\bsol\b)", re.I),
    "memory_chart": re.compile(r"memory\s+chart", re.I),
    "wgp": re.compile(r"\bWGP\b|\bSQ_|\bSPI_", re.I),
    "gl0": re.compile(r"\bGL0\b|\bTCP_", re.I),
    "gl1": re.compile(r"\bGL1\b|\bSQC_", re.I),
    "gl2": re.compile(r"\bGL2\b|\bTCC_", re.I),
    "gcea": re.compile(r"\bGCEA\b", re.I),
    "command_processor": re.compile(r"command\s+processor|\bCP[CFG]_", re.I),
    "grbm": re.compile(r"\bGRBM\b", re.I),
}


def rocprof_compute_summary(root: Path) -> dict[str, Any]:
    """Inventory captured rocprof-compute reports without assuming block IDs."""
    profile_dirs = sorted({
        path.parent for path in root.rglob("profiling_config.yaml")
        if "rocprof-compute" in path.parts
    })
    if not profile_dirs:
        return {"status": "not_collected"}
    profiles = []
    for directory in profile_dirs:
        columns: set[str] = set()
        for path in directory.glob("pmc_perf_*.csv"):
            with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                columns.update(csv.DictReader(stream).fieldnames or ())
        report_text = ""
        report_path = directory / "analysis.txt"
        if report_path.is_file():
            report_text = report_path.read_text(encoding="utf-8", errors="replace")
        evidence_lines = [line.strip() for line in report_text.splitlines() if line.strip()]
        sections = {}
        for name, pattern in COMPUTE_REPORT_PATTERNS.items():
            metric_names = sorted(column for column in columns if pattern.search(column))
            report_evidence = []
            for line in evidence_lines:
                if pattern.search(line) and line not in report_evidence:
                    report_evidence.append(line)
                if len(report_evidence) == 20:
                    break
            sections[name] = {
                "captured_in_report": bool(report_evidence),
                "report_evidence": report_evidence,
                "raw_metric_names": metric_names,
            }
        profiles.append({
            "path": str(directory.relative_to(root)),
            "raw_counter_files": len(list(directory.glob("pmc_perf_*.csv"))),
            "report_captured": report_path.is_file(),
            "sections": sections,
        })
    return {
        "status": "measured",
        "selection_policy": (
            "installed profiler default reports; sections matched by captured names, "
            "never fixed block identifiers"
        ),
        "profiles": profiles,
    }


def thread_trace_summary(root: Path) -> dict[str, Any]:
    categories = Counter()
    hotspots = Counter()
    states = Counter()
    files = 0
    for path in root.rglob("*stats*.csv"):
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not any(
                name.lower() in {"instruction", "opcode"} for name in reader.fieldnames
            ):
                continue
            files += 1
            for row in reader:
                instruction = parse_name(row, ("Instruction", "instruction", "Opcode"))
                execute = parse_int(row, ("Execute", "execute", "Latency"), 0)
                stall = parse_int(row, ("Stall", "stall"), 0)
                idle = parse_int(row, ("Idle", "idle"), 0)
                total = execute + stall + idle
                states["execution"] += execute
                states["stall_or_wait"] += stall
                states["idle"] += idle
                lower = instruction.lower()
                category = "other"
                for label, pattern in (
                    ("wait", r"waitcnt|sleep"), ("vmem", r"global_|buffer_|flat_|image_"),
                    ("flat", r"flat_"), ("lds", r"ds_|lds"), ("valu", r"^v_"),
                    ("salu", r"^s_(?!load|buffer)"), ("branch", r"branch|cbranch"),
                    ("barrier", r"barrier"), ("atomic", r"atomic|cmpswap|cas"),
                ):
                    if re.search(pattern, lower):
                        category = label
                        break
                categories[category] += total
                source = parse_name(row, ("Source", "source", "File", "file"))
                hotspots[(source or "unavailable", instruction or "unknown")] += total
    total = sum(categories.values())
    if not files:
        return {"status": "not_collected"}
    diagnostic_patterns = {
        "atomic_cas_or_load": re.compile(r"atomic|cmpswap|\bcas\b|atomic_load", re.I),
        "memory_wait": re.compile(r"waitcnt|sleep|memory.?wait", re.I),
        "csr_load": re.compile(r"csr|rowptr|colind|global_load|buffer_load", re.I),
        "global_queue_reservation": re.compile(r"queue|reservation|atomic_add", re.I),
        "reduction": re.compile(r"reduce|reduction|readlane|dpp", re.I),
        "barrier": re.compile(r"barrier", re.I),
    }
    diagnostic_hotspots = {}
    for label, pattern in diagnostic_patterns.items():
        matches = [
            {"source": key[0], "instruction": key[1], "cycles": cycles}
            for key, cycles in hotspots.most_common()
            if pattern.search(f"{key[0]} {key[1]}")
        ][:10]
        diagnostic_hotspots[label] = matches
    return {
        "status": "sampled_diagnostic", "decoded_files": files,
        "cycle_categories": {
            name: {"cycles": cycles, "percent": validated_percentage(cycles, total, f"ATT {name}")}
            for name, cycles in categories.most_common()
        },
        "wave_states": {
            name: {"cycles": cycles, "percent": validated_percentage(cycles, total, f"ATT {name}")}
            for name, cycles in states.most_common()
        },
        "high_cycle_instructions": [
            {"source": key[0], "instruction": key[1], "cycles": cycles}
            for key, cycles in hotspots.most_common(20)
        ],
        "diagnostic_hotspots": diagnostic_hotspots,
        "scope": "selected target CU/WGP and SIMD; not whole-GPU utilization",
    }


def bottleneck_table(
    trace: dict[str, Any], counters: dict[str, Any], att: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    families = trace["kernel_families"]
    counter_families = counters.get("families", {})
    for family, timeline in families.items():
        evidence = []
        classification_candidates = []
        duty = trace["gpu_active_duty_cycle_percent"]
        evidence.append(f"whole-run GPU-active duty cycle {duty:.1f}%")
        metrics = counter_families.get(family, {})
        weighted = metrics.get("duration_weighted_metrics", {})
        mem_busy = weighted.get("MemUnitBusy")
        occupancy = weighted.get("OccupancyPercent")
        reference_percent = metrics.get("percentage_of_streaming_request_reference")
        if duty < 50 and trace["hip_stream_synchronize"]["count"]:
            classification_candidates.append("launch/synchronization limited")
            evidence.append("low dispatch duty cycle with stream synchronization")
        if occupancy is not None and 0 < occupancy < 30:
            classification_candidates.append("occupancy limited")
            evidence.append(f"occupancy {occupancy:.1f}%")
        if reference_percent is not None and reference_percent >= 80:
            classification_candidates.append("memory-bandwidth pressured")
            evidence.append(
                f"request rate {reference_percent:.1f}% of streaming reference"
            )
        elif mem_busy is not None and mem_busy >= 80:
            evidence.append(
                f"MemUnitBusy {mem_busy:.1f}% without a near-reference validated request rate; "
                "bandwidth versus latency remains ambiguous"
            )
        valu_issue_metrics = {
            name: value for name, value in weighted.items()
            if "VALU" in name.upper()
            and any(token in name.upper() for token in ("ISSUE", "UTIL", "BUSY"))
        }
        if metrics.get("valu_issue_density") is not None:
            valu_issue_metrics["captured_system_sol_formula"] = metrics[
                "valu_issue_density"
            ]
        valu_issue = max(valu_issue_metrics.values(), default=None)
        if valu_issue is not None and valu_issue >= 80:
            classification_candidates.append("compute-throughput limited")
            evidence.append(f"captured VALU issue/utilization metric {valu_issue:.1f}%")
        if att.get("status") == "sampled_diagnostic":
            wait = att["cycle_categories"].get("wait", {}).get("percent", 0)
            if (family == "segmented_relaxation" and wait >= 30 and
                    (reference_percent is None or reference_percent < 80)):
                classification_candidates.append(
                    "irregular-memory latency/dependency limited"
                )
                evidence.append(f"sampled wait cycles {wait:.1f}%")
        unique_candidates = sorted(set(classification_candidates))
        if len(unique_candidates) == 1:
            classification = unique_candidates[0]
        elif len(unique_candidates) > 1:
            classification = "mixed"
            evidence.append("multiple limiting signals: " + ", ".join(unique_candidates))
        else:
            classification = "unresolved"
        rows.append({
            "kernel_family": family, "dispatches": timeline["count"],
            "additive_duration_ns": timeline["additive_ns"],
            "classification": classification, "evidence": evidence,
            "decision_factors": {
                "gpu_active_duty_cycle_percent": duty,
                "maximum_concurrency": max(int(level) for level in trace["concurrency_ns"]),
                "concurrency_distribution_ns": trace["concurrency_ns"],
                "occupancy_percent": occupancy,
                "resident_wave_metrics": metrics.get("active_or_resident_wave_metrics"),
                "launched_waves": metrics.get("launched_waves"),
                "wave_size": metrics.get("wave_size"),
                "valu_instruction_density_per_wave":
                    metrics.get("valu_instructions_per_wave") or
                    metrics.get("sq_valu_instructions_per_wave"),
                "valu_issue_density": metrics.get("valu_issue_density"),
                "captured_valu_issue_or_utilization_metrics": valu_issue_metrics,
                "instruction_density_per_wave": {
                    name: value for name, value in metrics.items()
                    if name.endswith("instructions_per_wave")
                },
                "cache_hit_metrics": {
                    name: value for name, value in weighted.items()
                    if any(token in name.upper() for token in ("GL0", "GL1", "L2", "CACHEHIT"))
                },
                "read_request_bytes_per_second": metrics.get("read_request_bytes_per_second"),
                "write_request_bytes_per_second": metrics.get("write_request_bytes_per_second"),
                "percentage_of_streaming_reference": reference_percent,
                "memory_unit_busy": mem_busy,
                "valid_wait_stall_indicators": {
                    name: value for name, value in weighted.items()
                    if "STALL" in name.upper() or "WAIT" in name.upper()
                },
                "synchronization_count": trace["hip_stream_synchronize"]["count"],
                "copy_count": sum(
                    item["count"] for item in trace["memory_copies"]["by_direction"].values()
                ),
                "classification_candidates": unique_candidates,
            },
            "counter_limitations": counters.get("limitations", []) if counters else [],
        })
    return rows


def validate_provenance(root: Path) -> dict[str, Any]:
    records = []
    for path in root.rglob("metadata.json"):
        if path.parent.name != "provenance":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        fingerprint_value = record.get("configuration_fingerprint")
        if not fingerprint_value:
            raise ValueError(f"missing configuration fingerprint in {path}")
        records.append((path, record))
    if not records:
        return {"status": "not_captured"}
    fingerprints = {record["configuration_fingerprint"] for _, record in records}
    if len(fingerprints) != 1:
        raise ValueError(
            "refusing to mix captured runs with different configuration fingerprints"
        )
    return {
        "status": "validated",
        "configuration_fingerprint": next(iter(fingerprints)),
        "metadata_files": [str(path.relative_to(root)) for path, _ in records],
    }


def analyze(root: Path, rocpd: Path | None = None) -> dict[str, Any]:
    provenance = validate_provenance(root)
    trace = load_rocpd(rocpd) if rocpd else load_csv_trace(root)
    timeline = trace_summary(trace)
    runtime_with_paths = read_json_records_with_paths(root, "bf11_runtime_stats")
    runtime_stats = [record for _, record in runtime_with_paths]
    runtime_configurations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, record in runtime_with_paths:
        relative = path.relative_to(root)
        if len(relative.parts) > 2 and relative.parts[0] == "timing":
            domain = "/".join(relative.parts[:2])
        elif len(relative.parts) > 1 and relative.parts[0] == "correctness":
            domain = f"correctness/{path.stem}"
        else:
            domain = relative.parts[0] if len(relative.parts) > 1 else "."
        runtime_configurations[domain].append(record)
    replay_fingerprints = {}
    for domain, records in runtime_configurations.items():
        configurations = {
            (record.get("effective_workers", record.get("workers")),
             record.get("segment_rounds"),
             record.get("hip_graph_mode", record.get("hip_graph")),
             record.get("adaptive_reset_threshold"))
            for record in records
        }
        query_counts = {record.get("telemetry_queries") for record in records
                        if record.get("telemetry_queries") is not None}
        if len(configurations) != 1 or len(query_counts) > 1:
            raise ValueError(
                f"refusing to mix mismatched application replays in {domain}"
            )
        replay_fingerprints[domain] = {
            "configuration": list(configurations)[0],
            "query_counts": sorted(query_counts),
            "replays": len(records),
        }
    if runtime_stats:
        # Trace/telemetry/counter processes are intentionally separate. Only
        # attach graph controller counts when exactly one matching record is
        # available; never add counts across application replays.
        record = next(
            (item for path, item in runtime_with_paths if "runtime-trace" in path.parts),
            runtime_stats[0],
        )
        timeline["hip_graph"].update({
            "direct_segments": record.get("direct_segments"),
            "launches_reported_by_bf11": record.get("hip_graph_segments"),
            "no_op_rounds": record.get("no_op_segment_rounds"),
            "fallbacks": record.get("graph_fallbacks"),
        })
    agent = read_agent_metadata(root)
    counters = counter_summary(root, load_counter_rows(root), agent["wave_size"])
    att = thread_trace_summary(root)
    query_telemetry = read_json_records(root, "bf11_query_telemetry")
    marker_query_ids = set(timeline["per_query"])
    telemetry_query_ids = {
        str(record["net_index"]) for record in query_telemetry
        if record.get("net_index") is not None
    }
    if marker_query_ids and telemetry_query_ids and marker_query_ids != telemetry_query_ids:
        raise ValueError(
            "runtime trace and telemetry contain inconsistent query identities: "
            f"{sorted(marker_query_ids)} versus {sorted(telemetry_query_ids)}"
        )
    result = {
        "schema_version": 1,
        "provenance_validation": provenance,
        "metadata": agent,
        "timeline": timeline,
        "counters": counters,
        "rocprof_compute": rocprof_compute_summary(root),
        "thread_trace": att,
        "query_telemetry": query_telemetry,
        "application_replay_fingerprints": replay_fingerprints,
    }
    result["bottleneck_decisions"] = bottleneck_table(timeline, counters, att)
    atomic_queue = {
        "mark_cas_attempts": sum(item.get("mark_cas_attempts", 0) for item in result["query_telemetry"]),
        "mark_cas_wins": sum(item.get("mark_cas_wins", 0) for item in result["query_telemetry"]),
        "queue_reservations": sum(item.get("queue_reservations", 0) for item in result["query_telemetry"]),
    }
    for row in result["bottleneck_decisions"]:
        row["decision_factors"]["atomic_and_queue_telemetry"] = atomic_queue
    return result


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    timeline = summary["timeline"]
    lines = [
        "# BF11 profiling analysis", "",
        f"Architecture: `{summary['metadata']['architecture']}`; "
        f"CUs: {summary['metadata']['cu_count']}; wave size: {summary['metadata']['wave_size']}.",
        "",
        f"GPU-active interval union: {timeline['kernel_interval_union_ns'] / 1e9:.6f} s "
        f"({timeline['gpu_active_duty_cycle_percent']:.1f}% of the dispatch span).",
        f"Additive multi-queue kernel duration: {timeline['additive_multi_queue_kernel_ns'] / 1e9:.6f} s. "
        "This is not wall time.", "", "## Bottleneck decisions", "",
        "| Kernel family | Dispatches | Classification | Evidence |", "|---|---:|---|---|",
    ]
    for row in summary["bottleneck_decisions"]:
        lines.append(
            f"| {row['kernel_family']} | {row['dispatches']} | {row['classification']} | "
            f"{' ; '.join(row['evidence'])} |"
        )
    lines.extend(("", "Counter and ATT limitations are retained in `summary.json`; "
                       "unavailable counters are not treated as zero."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="captured run directory")
    parser.add_argument("--rocpd", type=Path, help="RocPD database instead of CSV traces")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.input / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.input.resolve(), args.rocpd.resolve() if args.rocpd else None)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(summary, output / "README.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
