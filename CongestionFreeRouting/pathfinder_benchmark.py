#!/usr/bin/env python3
"""Contest-style benchmark entry point for the CSR PathFinder prototype.

The FPGA24 contest Makefile times routers with a rule shaped like:

    %_<router>.phys: %_unrouted.phys
        (/usr/bin/time <router-command> $< $@) ...

This wrapper accepts that two-argument router interface, infers the matching
logical netlist, builds the CSR/metadata sidecar with interchange_to_csr, runs
the C++ PathFinder executable, and writes a routed FPGAIF PhysicalNetlist by
replacing routed stubs with pip routeSegments.

Example contest Makefile recipe:

    %_pathfinder.phys: %_unrouted.phys %.netlist xcvu3p.device
        (/usr/bin/time python3 CongestionFreeRouting/pathfinder_benchmark.py $< $@) \
          $(call log_and_or_display,$@.log)

Manual run:

    python3 CongestionFreeRouting/pathfinder_benchmark.py vtr_mcml_unrouted.phys \
      vtr_mcml_pathfinder.phys --net-limit 100
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


_SCHEMA_CACHE: dict[Path, Any] = {}
_SCHEMA_CACHE_BY_ID: dict[str, tuple[Path, str, Any]] = {}


def infer_logical_netlist(unrouted_phys: Path) -> Path:
    name = unrouted_phys.name
    suffix = "_unrouted.phys"
    if name.endswith(suffix):
        return unrouted_phys.with_name(name[: -len(suffix)] + ".netlist")
    if name.endswith(".phys"):
        return unrouted_phys.with_suffix(".netlist")
    raise ValueError(
        "could not infer logical netlist path; pass --logical-netlist explicitly"
    )


def default_work_dir(output_phys: Path) -> Path:
    return output_phys.with_suffix(output_phys.suffix + ".pathfinder-work")


def run_command(argv: list[str], label: str) -> None:
    print(f"[pathfinder-benchmark] {label}: {' '.join(argv)}", flush=True)
    subprocess.run(argv, check=True)


def executable_path(value: str, env_name: str) -> str:
    override = os.environ.get(env_name)
    return override if override else value


def default_schema_dir() -> Path | None:
    env_schema = os.environ.get("FPGA_INTERCHANGE_SCHEMA_DIR")
    if env_schema:
        return Path(env_schema)

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "fpga-interchange-schema" / "interchange",
        repo_root.parent / "fpga-interchange-schema" / "interchange",
        Path.cwd() / "fpga-interchange-schema" / "interchange",
        Path.cwd() / "interchange",
    )
    for candidate in candidates:
        if (candidate / "PhysicalNetlist.capnp").exists():
            return candidate
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CSR PathFinder prototype through the FPGA24 router interface."
    )
    parser.add_argument("input_phys", type=Path)
    parser.add_argument("output_phys", type=Path)
    parser.add_argument("--logical-netlist", type=Path)
    parser.add_argument("--device", type=Path, default=Path("xcvu3p.device"))
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=default_schema_dir(),
        help="directory containing PhysicalNetlist.capnp, or set FPGA_INTERCHANGE_SCHEMA_DIR",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--interchange-to-csr",
        default=executable_path("interchange_to_csr", "INTERCHANGE_TO_CSR"),
        help="converter executable path, or set INTERCHANGE_TO_CSR",
    )
    parser.add_argument(
        "--pathfinder",
        default=executable_path("pathfinder", "PATHFINDER_BIN"),
        help="PathFinder executable path, or set PATHFINDER_BIN",
    )
    parser.add_argument("--delta", type=float, help="delta-step bucket width")
    parser.add_argument(
        "--max-pathfinder-iters",
        type=int,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--max-sssp-iters",
        type=int,
        help="delta-step rounds or unit-BFS depth cap",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        help="capacity used only for overuse diagnostics",
    )
    parser.add_argument(
        "--present-factor",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--present-multiplier",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--history-factor",
        type=float,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument("--net-limit", type=int)
    parser.add_argument(
        "--route-batch-size",
        type=int,
        help="compatibility-only for the one-shot router; forwarded and ignored",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="keep generated .csrbin/.ifmeta.bin files after the run",
    )
    parser.add_argument(
        "--allow-unrouted-stubs",
        action="store_true",
        help="write any stubs not covered by PathFinder back to the output netlist",
    )
    return parser.parse_args(argv)


def pathfinder_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for attr, option in (
        ("delta", "--delta"),
        ("max_pathfinder_iters", "--max-pathfinder-iters"),
        ("max_sssp_iters", "--max-sssp-iters"),
        ("capacity", "--capacity"),
        ("present_factor", "--present-factor"),
        ("present_multiplier", "--present-multiplier"),
        ("history_factor", "--history-factor"),
        ("net_limit", "--net-limit"),
        ("route_batch_size", "--route-batch-size"),
    ):
        value = getattr(args, attr)
        if value is not None:
            forwarded.extend([option, str(value)])
    return forwarded


def read_gzip_or_plain(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def schema_identity(schema_path: Path) -> tuple[str, str]:
    schema_text = schema_path.read_text(encoding="utf-8")
    match = re.search(r"@0x([0-9a-fA-F]+);", schema_text)
    schema_id = match.group(1).lower() if match else str(schema_path.resolve())
    digest = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
    return schema_id, digest


def load_physical_schema(schema_dir: Path):
    schema_path = (schema_dir / "PhysicalNetlist.capnp").resolve()
    cached = _SCHEMA_CACHE.get(schema_path)
    if cached is not None:
        return cached
    if not schema_path.exists():
        raise FileNotFoundError(f"missing PhysicalNetlist.capnp in {schema_dir}")

    schema_id, schema_digest = schema_identity(schema_path)
    cached_by_id = _SCHEMA_CACHE_BY_ID.get(schema_id)
    if cached_by_id is not None:
        cached_path, cached_digest, cached_schema = cached_by_id
        if cached_digest != schema_digest:
            raise RuntimeError(
                "a different PhysicalNetlist.capnp with the same Cap'n Proto "
                f"schema ID is already loaded: {cached_path}"
            )
        _SCHEMA_CACHE[schema_path] = cached_schema
        return cached_schema

    try:
        import capnp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pycapnp is required to write legal FPGAIF .phys files"
        ) from exc

    schema = capnp.load(str(schema_path), imports=[str(schema_dir.resolve())])
    _SCHEMA_CACHE[schema_path] = schema
    _SCHEMA_CACHE_BY_ID[schema_id] = (schema_path, schema_digest, schema)
    return schema


def read_routes_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as route_file:
        for line_no, line in enumerate(route_file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            route = json.loads(stripped)
            net_name = route.get("net")
            if not isinstance(net_name, str) or not net_name:
                raise ValueError(f"{path}:{line_no}: route entry has no net name")
            if not route.get("routed", False):
                raise ValueError(f"{path}:{line_no}: net {net_name} is not fully routed")
            if net_name in routes:
                raise ValueError(f"{path}:{line_no}: duplicate route for net {net_name}")
            routes[net_name] = route
    if not routes:
        raise ValueError(f"no routed nets were written to {path}")
    return routes


def string_at(str_list, index: int) -> str:
    return str_list[index]


def get_string_index(text: str, string_to_index: dict[str, int]) -> int:
    index = string_to_index.get(text)
    if index is None:
        index = len(string_to_index)
        string_to_index[text] = index
    return index


def site_pin_key(site: str, pin: str) -> tuple[str, str]:
    return (site, pin)


def build_route_tables(route: dict[str, Any]):
    adjacency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    incoming_parent: dict[int, int] = {}
    seen_edges: set[tuple[int, int]] = set()

    for edge in route.get("edges", []):
        parent = int(edge["from"])
        child = int(edge["to"])
        key = (parent, child)
        if key in seen_edges:
            continue
        previous_parent = incoming_parent.get(child)
        if previous_parent is not None and previous_parent != parent:
            raise ValueError(
                f"net {route['net']} drives node {child} from both "
                f"{previous_parent} and {parent}"
            )
        incoming_parent[child] = parent
        seen_edges.add(key)
        adjacency[parent].append(edge)

    source_node_by_pin: dict[tuple[str, str], int] = {}
    for source in route.get("sources", []):
        key = site_pin_key(str(source["site"]), str(source["pin"]))
        if key in source_node_by_pin:
            raise ValueError(f"net {route['net']} has duplicate source {key}")
        source_node_by_pin[key] = int(source["node"])

    sink_pins_by_node: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for sink in route.get("sinks", []):
        if not sink.get("reached", False):
            raise ValueError(f"net {route['net']} has unreached sink {sink}")
        key = site_pin_key(str(sink["site"]), str(sink["pin"]))
        sink_pins_by_node[int(sink["node"])].append(key)

    return adjacency, source_node_by_pin, sink_pins_by_node, seen_edges


def collect_site_pin_branches(branches, str_list):
    pins = []
    queue = list(branches)
    while queue:
        branch = queue.pop()
        route_segment = branch.routeSegment
        if route_segment.which() == "sitePin":
            site_pin = route_segment.sitePin
            pins.append(
                (
                    site_pin_key(
                        string_at(str_list, site_pin.site),
                        string_at(str_list, site_pin.pin),
                    ),
                    branch,
                )
            )
        queue.extend(branch.branches)
    return pins


def insert_route_tree(
    root_branch,
    root_node: int,
    net_name: str,
    adjacency: dict[int, list[dict[str, Any]]],
    sink_pins_by_node: dict[int, list[tuple[str, str]]],
    sink_pin_orphans: dict[tuple[str, str], Any],
    string_to_index: dict[str, int],
) -> int:
    emitted_pips = 0
    stack: list[tuple[Any, int, tuple[int, ...]]] = [(root_branch, root_node, ())]

    while stack:
        branch, node, ancestors = stack.pop()
        if node in ancestors:
            raise ValueError(f"net {net_name} route tree has a cycle at node {node}")
        child_edges = adjacency.get(node, [])
        sink_keys = sink_pins_by_node.get(node, [])
        branch_count = len(child_edges) + len(sink_keys)
        if branch_count == 0:
            continue
        if len(branch.branches) != 0:
            raise ValueError(
                f"net {net_name} source/tree branch already has child branches"
            )

        new_branches = branch.init("branches", branch_count)
        branch_index = 0
        next_ancestors = ancestors + (node,)

        for edge in child_edges:
            next_branch = new_branches[branch_index]
            branch_index += 1
            pip = next_branch.routeSegment.init("pip")
            pip.tile = get_string_index(str(edge["tile"]), string_to_index)
            pip.wire0 = get_string_index(str(edge["wire0"]), string_to_index)
            pip.wire1 = get_string_index(str(edge["wire1"]), string_to_index)
            pip.forward = bool(edge["forward"])
            stack.append((next_branch, int(edge["to"]), next_ancestors))
            emitted_pips += 1

        for sink_key in sink_keys:
            orphan = sink_pin_orphans.pop(sink_key, None)
            if orphan is None:
                raise ValueError(
                    f"net {net_name} routed sink {sink_key} was not present in stubs"
                )
            new_branches[branch_index] = orphan.get()
            branch_index += 1

    return emitted_pips


def resize_string_list(netlist, string_to_index: dict[str, int]) -> int:
    old_str_list = netlist.strList
    old_count = len(old_str_list)
    orphan_str_list = [old_str_list.disown(i) for i in range(old_count)]
    new_str_list = netlist.init("strList", len(string_to_index))

    for text, index in string_to_index.items():
        if index < old_count:
            new_str_list.adopt(index, orphan_str_list[index])
        else:
            new_str_list[index] = text
    return len(string_to_index) - old_count


def write_routed_physical_netlist(
    input_phys: Path,
    output_phys: Path,
    schema_dir: Path,
    routes_path: Path,
    allow_unrouted_stubs: bool,
) -> None:
    schema = load_physical_schema(schema_dir)
    routes_by_net = read_routes_jsonl(routes_path)
    data = read_gzip_or_plain(input_phys)

    with schema.PhysNetlist.from_bytes(
        data,
        traversal_limit_in_words=sys.maxsize,
        nesting_limit=2**16,
    ) as reader:
        netlist = reader.as_builder()

    str_list = netlist.strList
    string_to_index = {string_at(str_list, i): i for i in range(len(str_list))}
    num_pips = 0
    routed_net_count = 0

    for net in netlist.physNets:
        net_name = string_at(str_list, net.name)
        route = routes_by_net.pop(net_name, None)
        if route is None:
            continue

        adjacency, source_node_by_pin, sink_pins_by_node, route_edges = build_route_tables(route)
        sink_pin_orphans: dict[tuple[str, str], Any] = {}
        unknown_stub_orphans: list[Any] = []

        for index, stub in enumerate(net.stubs):
            route_segment = stub.routeSegment
            if route_segment.which() != "sitePin":
                orphan = net.stubs.disown(index)
                unknown_stub_orphans.append(orphan)
                continue
            site_pin = route_segment.sitePin
            key = site_pin_key(
                string_at(str_list, site_pin.site),
                string_at(str_list, site_pin.pin),
            )
            if key in sink_pin_orphans:
                raise ValueError(f"net {net_name} has duplicate stub {key}")
            orphan = net.stubs.disown(index)
            sink_pin_orphans[key] = orphan

        net.disown("stubs")

        emitted_edges = 0
        source_branches = collect_site_pin_branches(net.sources, str_list)
        for source_key, source_branch in source_branches:
            source_node = source_node_by_pin.get(source_key)
            if source_node is None:
                continue
            if source_node not in adjacency and source_node not in sink_pins_by_node:
                continue
            emitted_edges += insert_route_tree(
                source_branch,
                source_node,
                net_name,
                adjacency,
                sink_pins_by_node,
                sink_pin_orphans,
                string_to_index,
            )

        if emitted_edges != len(route_edges):
            raise ValueError(
                f"net {net_name} emitted {emitted_edges} PIPs but route has "
                f"{len(route_edges)} PIPs"
            )

        remaining_stubs = list(sink_pin_orphans.values()) + unknown_stub_orphans
        if remaining_stubs and not allow_unrouted_stubs:
            raise ValueError(
                f"net {net_name} still has {len(remaining_stubs)} unrouted stubs"
            )
        if remaining_stubs:
            new_stubs = net.init("stubs", len(remaining_stubs))
            for index, orphan in enumerate(remaining_stubs):
                new_stubs[index] = orphan.get()

        num_pips += emitted_edges
        routed_net_count += 1

    if routes_by_net:
        missing = ", ".join(sorted(routes_by_net)[:5])
        raise ValueError(f"routed nets were not found in input .phys: {missing}")

    added_strings = resize_string_list(netlist, string_to_index)
    output_phys.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_phys, "wb", compresslevel=6) as out_file:
        out_file.write(netlist.to_bytes())
    print(
        "[pathfinder-benchmark] wrote routed .phys with "
        f"{num_pips} PIPs across {routed_net_count} nets "
        f"({added_strings} new strings)",
        flush=True,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_phys = args.input_phys.resolve()
    output_phys = args.output_phys.resolve()
    logical_netlist = (
        args.logical_netlist.resolve()
        if args.logical_netlist is not None
        else infer_logical_netlist(input_phys).resolve()
    )
    device = args.device.resolve()
    schema_dir = args.schema_dir.resolve() if args.schema_dir is not None else None

    for path, label in (
        (input_phys, "input physical netlist"),
        (logical_netlist, "logical netlist"),
        (device, "device resources"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if schema_dir is None:
        raise FileNotFoundError(
            "missing FPGA Interchange schema directory; pass --schema-dir or "
            "set FPGA_INTERCHANGE_SCHEMA_DIR"
        )
    if not schema_dir.exists():
        raise FileNotFoundError(f"missing FPGA Interchange schema directory: {schema_dir}")

    output_phys.parent.mkdir(parents=True, exist_ok=True)
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else default_work_dir(output_phys).resolve()
    )

    temporary_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary_owner = tempfile.TemporaryDirectory(
            prefix=work_dir.name + ".", dir=str(work_dir.parent)
        )
        work_dir = Path(temporary_owner.name)

    try:
        csr_path = work_dir / (output_phys.stem + ".csrbin")
        metadata_path = work_dir / (output_phys.stem + ".csrbin.ifmeta.bin")
        routes_path = work_dir / (output_phys.stem + ".routes.jsonl")

        run_command(
            [
                args.interchange_to_csr,
                str(input_phys),
                str(logical_netlist),
                str(csr_path),
                "--device",
                str(device),
                "--metadata",
                str(metadata_path),
            ],
            "convert FPGAIF to CSR",
        )

        run_command(
            [
                args.pathfinder,
                str(csr_path),
                str(metadata_path),
                "--routes-out",
                str(routes_path),
                *pathfinder_args(args),
            ],
            "run CSR PathFinder",
        )

        write_routed_physical_netlist(
            input_phys,
            output_phys,
            schema_dir,
            routes_path,
            args.allow_unrouted_stubs,
        )
        return 0
    finally:
        if temporary_owner is not None:
            temporary_owner.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"[pathfinder-benchmark] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
