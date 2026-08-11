#!/usr/bin/env python3
from __future__ import annotations

import copy
import gzip
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "CongestionFreeRouting")
)

import pathfinder_benchmark as benchmark


SCHEMA = """@0xb6fdf5d32bc43210;

struct PhysNetlist {
  strList @0 :List(Text);
  physNets @1 :List(PhysNet);

  struct PhysNet {
    name @0 :UInt32;
    type @1 :NetType;
    sources @2 :List(RouteBranch);
    stubs @3 :List(RouteBranch);
  }

  enum NetType {
    signal @0;
    gnd @1;
    vcc @2;
  }

  struct RouteBranch {
    routeSegment @0 :RouteSegment;
    branches @1 :List(RouteBranch);
  }

  struct RouteSegment {
    union {
      sitePin @0 :PhysSitePin;
      pip @1 :PhysPIP;
    }
  }

  struct PhysSitePin {
    site @0 :UInt32;
    pin @1 :UInt32;
  }

  struct PhysPIP {
    tile @0 :UInt32;
    wire0 @1 :UInt32;
    wire1 @2 :UInt32;
    forward @3 :Bool;
    isFixed @4 :Bool;
    union {
      noSite @5 :Void;
      site @6 :UInt32;
    }
  }
}
"""


def make_input_phys(schema, path: Path) -> None:
    message = schema.PhysNetlist.new_message()
    strings = ["net0", "SRC_SITE", "SRC_PIN", "SINK_SITE", "SINK_PIN"]
    str_list = message.init("strList", len(strings))
    for index, text in enumerate(strings):
        str_list[index] = text

    phys_nets = message.init("physNets", 1)
    net = phys_nets[0]
    net.name = 0
    net.type = "signal"

    sources = net.init("sources", 1)
    source_site_pin = sources[0].routeSegment.init("sitePin")
    source_site_pin.site = 1
    source_site_pin.pin = 2

    stubs = net.init("stubs", 1)
    sink_site_pin = stubs[0].routeSegment.init("sitePin")
    sink_site_pin.site = 3
    sink_site_pin.pin = 4

    path.write_bytes(message.to_bytes())


def assert_routed_output(schema, output_phys: Path) -> None:
    with schema.PhysNetlist.from_bytes(
        gzip.decompress(output_phys.read_bytes()),
        traversal_limit_in_words=sys.maxsize,
        nesting_limit=2**16,
    ) as routed:
        strings = routed.strList
        net = routed.physNets[0]
        assert len(net.stubs) == 0
        assert len(net.sources[0].branches) == 1
        pip_branch = net.sources[0].branches[0]
        assert pip_branch.routeSegment.which() == "pip"
        pip = pip_branch.routeSegment.pip
        assert strings[pip.tile] == "TILE_A"
        assert strings[pip.wire0] == "WIRE_0"
        assert strings[pip.wire1] == "WIRE_1"
        assert pip.forward is True
        assert pip.which() == "noSite"
        assert len(pip_branch.branches) == 1
        sink_branch = pip_branch.branches[0]
        assert sink_branch.routeSegment.which() == "sitePin"
        sink = sink_branch.routeSegment.sitePin
        assert strings[sink.site] == "SINK_SITE"
        assert strings[sink.pin] == "SINK_PIN"


def make_duplicate_endpoint_phys(schema, path: Path) -> None:
    message = schema.PhysNetlist.new_message()
    strings = [
        "net0",
        "SRC_SITE",
        "SRC_PIN",
        "SRC_PIN_ALT",
        "SINK_SITE",
        "SINK_PIN",
    ]
    str_list = message.init("strList", len(strings))
    for index, text in enumerate(strings):
        str_list[index] = text

    net = message.init("physNets", 1)[0]
    net.name = 0
    net.type = "signal"
    sources = net.init("sources", 3)
    for index, pin_index in enumerate((2, 2, 3)):
        source = sources[index].routeSegment.init("sitePin")
        source.site = 1
        source.pin = pin_index
    stubs = net.init("stubs", 2)
    for stub in stubs:
        sink = stub.routeSegment.init("sitePin")
        sink.site = 4
        sink.pin = 5
    path.write_bytes(message.to_bytes())


def assert_duplicate_endpoint_output(schema, output_phys: Path) -> None:
    with schema.PhysNetlist.from_bytes(
        gzip.decompress(output_phys.read_bytes()),
        traversal_limit_in_words=sys.maxsize,
        nesting_limit=2**16,
    ) as routed:
        net = routed.physNets[0]
        assert len(net.stubs) == 0
        rooted = [source for source in net.sources if len(source.branches) != 0]
        assert len(rooted) == 1
        assert len(rooted[0].branches) == 1
        pip_branch = rooted[0].branches[0]
        assert pip_branch.routeSegment.which() == "pip"
        assert len(pip_branch.branches) == 2
        assert all(
            child.routeSegment.which() == "sitePin"
            for child in pip_branch.branches
        )


def make_legacy_metadata(path: Path) -> None:
    strings = [
        "net0",
        "SRC_SITE",
        "SRC_PIN",
        "SINK_SITE",
        "SINK_PIN",
        "TILE_A",
        "WIRE_0",
        "WIRE_1",
    ]
    words = [
        4,  # version
        2,  # outgoing CSR orientation
        len(strings),
        2,  # node count
        1,  # edge attr count
        1,  # PIP data count
        0,  # site-pin attr count
        1,  # route request count
        0,  # blocked nodes
        0,  # sink-stop nodes
        0,  # logical cells
        0,  # logical nets
        0,  # logical port instances
        0,  # physical bytes
        0,  # logical bytes
        0,
        0,
        0,
        0,
    ]
    payload = bytearray(b"RIPSIFM1")
    payload.extend(struct.pack(f"={len(words)}Q", *words))
    for text in strings:
        encoded = text.encode()
        payload.extend(struct.pack("=Q", len(encoded)))
        payload.extend(encoded)
    payload.extend(b"\0" * 80)  # seven legacy arrays, 40 bytes/node
    payload.extend(struct.pack("=2Q", 5, 0))
    payload.extend(struct.pack("=3Q", 6, 7, 1))
    payload.extend(struct.pack("=3Q", 0, 2**64 - 1, 1))
    payload.extend(struct.pack("=3Q", 0, 1, 2))
    payload.extend(struct.pack("=Q", 1))
    payload.extend(struct.pack("=3Q", 1, 3, 4))
    path.write_bytes(payload)


def make_v8_metadata(path: Path) -> str:
    strings = [
        "net0",
        "SRC_SITE",
        "SRC_PIN",
        "SINK_SITE",
        "SINK_PIN",
        "TILE_A",
        "WIRE_0",
        "WIRE_1",
    ]
    pair_high = 0x123456789ABCDEF0
    pair_low = 0x0FEDCBA987654321
    pair_id = f"{pair_high:016x}{pair_low:016x}"
    counts = [
        len(strings),
        2,  # node count
        1,  # compact edge attr count
        1,  # compact PIP count
        0,  # endpoint PIPs
        0,  # site-pin attrs
        1,  # route requests
        0,  # blocked nodes
        0,  # sink-stop nodes
        0,  # omitted logical cells
        1,  # flat logical-net names
        0,  # omitted logical port instances
        0,  # omitted physical payload
        0,  # omitted logical payload
    ]
    payload = bytearray(b"RIPSIFM1")
    payload.extend(struct.pack("=4Q", 8, 2, pair_high, pair_low))
    payload.extend(struct.pack(f"={len(counts)}Q", *counts))
    payload.extend(struct.pack("=4Q", 0, 0, 0, 0))
    for text in strings:
        encoded = text.encode()
        payload.extend(struct.pack("=Q", len(encoded)))
        payload.extend(encoded)
    payload.extend(struct.pack("=2I", 5, 0))
    payload.extend(struct.pack("=3I", 6, 7, 1))
    no_endpoint = 2**64 - 1
    payload.extend(struct.pack("=3Q", 0, 0, 1))
    payload.extend(struct.pack("=4Q", 0, 1, 2, no_endpoint))
    payload.extend(struct.pack("=Q", 1))
    payload.extend(struct.pack("=4Q", 1, 3, 4, no_endpoint))
    payload.extend(struct.pack("=Q", 0))  # logical net 0 -> "net0"
    path.write_bytes(payload)
    Path(str(path) + ".generation").write_text(pair_id + "\n", encoding="ascii")
    return pair_id


def make_attachment_case():
    pair_id = "00000000000000010000000000000002"
    endpoints = (
        benchmark.MetadataEndpointPip(
            1, 1, 2, "SRC_TILE", "SRC_W0", "SRC_W1", False,
            "TRAVERSED_SRC", 0, 0,
        ),
        benchmark.MetadataEndpointPip(
            3, 3, 4, "SINK_TILE", "SINK_W0", "SINK_W1", True,
            "TRAVERSED_SINK", 5, 1,
        ),
    )
    request = benchmark.MetadataRouteRequest(
        "net0",
        (benchmark.MetadataSitePin(0, "SRC_SITE", "SRC_PIN", 0),),
        (benchmark.MetadataSitePin(5, "SINK_SITE", "SINK_PIN", 1),),
    )
    metadata = benchmark.RoutingMetadataSummary(
        version=7,
        artifact_pair_id=pair_id,
        node_count=7,
        edge_attr_count=6,
        endpoint_pips=endpoints,
        route_requests=(request,),
    )
    route = {
        "artifact_pair_id": pair_id,
        "net": "net0",
        "routed": True,
        "sources": [{"node": 0, "site": "SRC_SITE", "pin": "SRC_PIN"}],
        "sinks": [{
            "node": 5,
            "site": "SINK_SITE",
            "pin": "SINK_PIN",
            "reached": True,
            "source": 0,
        }],
        "edges": [
            {"from": 0, "to": 1, "csr_edge": 0, "tile": "C0_TILE",
             "wire0": "C0_W0", "wire1": "C0_W1", "forward": True,
             "attachment": None, "site": None},
            {"from": 1, "to": 2, "csr_edge": 1, "tile": "SRC_TILE",
             "wire0": "SRC_W0", "wire1": "SRC_W1", "forward": False,
             "attachment": 0, "site": "TRAVERSED_SRC"},
            {"from": 2, "to": 3, "csr_edge": 2, "tile": "C2_TILE",
             "wire0": "C2_W0", "wire1": "C2_W1", "forward": True,
             "attachment": None, "site": None},
            {"from": 3, "to": 4, "csr_edge": 3, "tile": "SINK_TILE",
             "wire0": "SINK_W0", "wire1": "SINK_W1", "forward": True,
             "attachment": 1, "site": "TRAVERSED_SINK"},
            {"from": 4, "to": 5, "csr_edge": 4, "tile": "C4_TILE",
             "wire0": "C4_W0", "wire1": "C4_W1", "forward": False,
             "attachment": None, "site": None},
        ],
    }
    return metadata, route


def assert_attachment_output(schema, output_phys: Path) -> None:
    expected_sites = [None, "TRAVERSED_SRC", None, "TRAVERSED_SINK", None]
    expected_forward = [True, False, True, True, False]
    with schema.PhysNetlist.from_bytes(
        gzip.decompress(output_phys.read_bytes()),
        traversal_limit_in_words=sys.maxsize,
        nesting_limit=2**16,
    ) as routed:
        strings = routed.strList
        branch = routed.physNets[0].sources[0]
        for expected_site, forward in zip(expected_sites, expected_forward):
            assert len(branch.branches) == 1
            branch = branch.branches[0]
            pip = branch.routeSegment.pip
            assert pip.forward is forward
            if expected_site is None:
                assert pip.which() == "noSite"
            else:
                assert pip.which() == "site"
                assert strings[pip.site] == expected_site
        assert branch.branches[0].routeSegment.which() == "sitePin"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        schema_dir = tmp_path / "schema"
        schema_dir.mkdir()
        (schema_dir / "PhysicalNetlist.capnp").write_text(SCHEMA, encoding="utf-8")

        schema = benchmark.load_physical_schema(schema_dir)
        input_phys = tmp_path / "input.phys"
        output_phys = tmp_path / "output.phys"
        routes_path = tmp_path / "routes.jsonl"
        make_input_phys(schema, input_phys)

        route = {
            "net": "net0",
            "routed": True,
            "sources": [{"node": 0, "site": "SRC_SITE", "pin": "SRC_PIN"}],
            "sinks": [
                {
                    "node": 1,
                    "site": "SINK_SITE",
                    "pin": "SINK_PIN",
                    "reached": True,
                    "source": 0,
                }
            ],
            "edges": [
                {
                    "from": 0,
                    "to": 1,
                    "csr_edge": 0,
                    "tile": "TILE_A",
                    "wire0": "WIRE_0",
                    "wire1": "WIRE_1",
                    "forward": True,
                    "attachment": None,
                    "site": None,
                }
            ],
        }
        routes_path.write_text(json.dumps(route) + "\n", encoding="utf-8")

        benchmark.write_routed_physical_netlist(
            input_phys,
            output_phys,
            schema_dir,
            routes_path,
            allow_unrouted_stubs=False,
        )
        assert_routed_output(schema, output_phys)

        attachment_metadata, attachment_route = make_attachment_case()
        attachment_routes = tmp_path / "attachment_routes.jsonl"
        attachment_output = tmp_path / "attachment_output.phys"
        attachment_routes.write_text(
            json.dumps(attachment_route) + "\n", encoding="utf-8"
        )
        benchmark.write_routed_physical_netlist(
            input_phys,
            attachment_output,
            schema_dir,
            attachment_routes,
            allow_unrouted_stubs=False,
            metadata_summary=attachment_metadata,
        )
        assert_attachment_output(schema, attachment_output)

        conventional_attachment_route = copy.deepcopy(attachment_route)
        conventional_attachment_route["edges"] = [
            {"from": 0, "to": 6, "csr_edge": 0, "tile": "CONV_0",
             "wire0": "CW0", "wire1": "CW1", "forward": True,
             "attachment": None, "site": None},
            {"from": 6, "to": 5, "csr_edge": 2, "tile": "CONV_1",
             "wire0": "CW2", "wire1": "CW3", "forward": True,
             "attachment": None, "site": None},
        ]
        benchmark.validate_routes_against_metadata(
            {"net0": conventional_attachment_route}, attachment_metadata
        )

        malformed_cases = []
        conventional_site = copy.deepcopy(attachment_route)
        conventional_site["edges"][0]["attachment"] = 0
        conventional_site["edges"][0]["site"] = "TRAVERSED_SRC"
        malformed_cases.append((conventional_site, "conventional"))
        wrong_direction = copy.deepcopy(attachment_route)
        wrong_direction["edges"][1]["forward"] = True
        malformed_cases.append((wrong_direction, "exactly match"))
        wrong_index = copy.deepcopy(attachment_route)
        wrong_index["edges"][1]["attachment"] = 1
        malformed_cases.append((wrong_index, "index"))
        source_transit = copy.deepcopy(attachment_route)
        source_transit["edges"].append(
            {"from": 0, "to": 6, "csr_edge": 5, "tile": "TRANSIT",
             "wire0": "T0", "wire1": "T1", "forward": True,
             "attachment": None, "site": None}
        )
        malformed_cases.append((source_transit, "transit"))
        for malformed, expected_error in malformed_cases:
            try:
                benchmark.validate_routes_against_metadata(
                    {"net0": malformed}, attachment_metadata
                )
            except ValueError as exc:
                assert expected_error in str(exc)
            else:
                raise AssertionError(f"malformed attachment route accepted: {malformed}")

        detached_routes = tmp_path / "detached_routes.jsonl"
        detached_routes.write_text(
            json.dumps({**route, "edges": []}) + "\n", encoding="utf-8"
        )
        try:
            benchmark.write_routed_physical_netlist(
                input_phys,
                tmp_path / "detached_output.phys",
                schema_dir,
                detached_routes,
                allow_unrouted_stubs=True,
            )
        except ValueError as exc:
            assert "reached sink" in str(exc)
        else:
            raise AssertionError("detached reached sink was accepted")

        fractional_routes = tmp_path / "fractional_routes.jsonl"
        fractional_routes.write_text(
            json.dumps(
                {
                    **route,
                    "edges": [{**route["edges"][0], "from": 0.5}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            benchmark.write_routed_physical_netlist(
                input_phys,
                tmp_path / "fractional_output.phys",
                schema_dir,
                fractional_routes,
                allow_unrouted_stubs=False,
            )
        except ValueError as exc:
            assert "not an integer" in str(exc)
        else:
            raise AssertionError("fractional route node was accepted")

        duplicate_input = tmp_path / "duplicate_input.phys"
        duplicate_output = tmp_path / "duplicate_output.phys"
        duplicate_routes = tmp_path / "duplicate_routes.jsonl"
        make_duplicate_endpoint_phys(schema, duplicate_input)
        duplicate_route = {
            **route,
            "sources": [
                {"node": 0, "site": "SRC_SITE", "pin": "SRC_PIN"},
                {"node": 0, "site": "SRC_SITE", "pin": "SRC_PIN"},
                {"node": 0, "site": "SRC_SITE", "pin": "SRC_PIN_ALT"},
            ],
            "sinks": [route["sinks"][0], route["sinks"][0]],
        }
        duplicate_routes.write_text(
            json.dumps(duplicate_route) + "\n", encoding="utf-8"
        )
        benchmark.write_routed_physical_netlist(
            duplicate_input,
            duplicate_output,
            schema_dir,
            duplicate_routes,
            allow_unrouted_stubs=False,
        )
        assert_duplicate_endpoint_output(schema, duplicate_output)

        logical_netlist = tmp_path / "input.netlist"
        logical_netlist.write_bytes(b"logical")
        device = tmp_path / "xcvu3p.device"
        device.write_bytes(b"device")
        metadata_fixture = tmp_path / "legacy.ifmeta.bin"
        make_legacy_metadata(metadata_fixture)

        v8_metadata = tmp_path / "compact-v8.ifmeta.bin"
        v8_pair_id = make_v8_metadata(v8_metadata)
        assert benchmark.read_metadata_artifact_pair_id(v8_metadata) == v8_pair_id
        v8_summary = benchmark.read_metadata_summary(v8_metadata)
        assert v8_summary.version == 8
        assert v8_summary.artifact_pair_id == v8_pair_id
        assert v8_summary.node_count == 2
        assert v8_summary.edge_attr_count == 1
        assert len(v8_summary.route_requests) == 1
        assert v8_summary.route_requests[0].net == "net0"

        malformed_v8_count = tmp_path / "compact-v8-bad-count.ifmeta.bin"
        make_v8_metadata(malformed_v8_count)
        malformed_bytes = bytearray(malformed_v8_count.read_bytes())
        struct.pack_into("=Q", malformed_bytes, 40 + 9 * 8, 1)
        malformed_v8_count.write_bytes(malformed_bytes)
        try:
            benchmark.read_metadata_summary(malformed_v8_count)
        except ValueError as exc:
            assert "must be zero" in str(exc)
        else:
            raise AssertionError("metadata v8 accepted a nonzero omitted count")

        malformed_v8_correlation = (
            tmp_path / "compact-v8-bad-correlation.ifmeta.bin"
        )
        make_v8_metadata(malformed_v8_correlation)
        correlation_bytes = bytearray(malformed_v8_correlation.read_bytes())
        struct.pack_into("=Q", correlation_bytes, len(correlation_bytes) - 8, 1)
        malformed_v8_correlation.write_bytes(correlation_bytes)
        try:
            benchmark.read_metadata_summary(malformed_v8_correlation)
        except ValueError as exc:
            assert "correlation mismatch" in str(exc)
        else:
            raise AssertionError("metadata v8 accepted a mismatched logical net")

        trailing_v8 = tmp_path / "compact-v8-trailing.ifmeta.bin"
        make_v8_metadata(trailing_v8)
        trailing_v8.write_bytes(trailing_v8.read_bytes() + b"x")
        try:
            benchmark.read_metadata_summary(trailing_v8)
        except ValueError as exc:
            assert "trailing bytes" in str(exc)
        else:
            raise AssertionError("metadata v8 accepted trailing bytes")

        fake_converter = tmp_path / "fake_interchange_to_csr.py"
        fake_converter.write_text(
            f"""#!/usr/bin/env python3
from pathlib import Path
import sys
Path(sys.argv[4]).write_bytes(b"csr")
Path(sys.argv[sys.argv.index("--metadata") + 1]).write_bytes(Path({str(metadata_fixture)!r}).read_bytes())
""",
            encoding="utf-8",
        )
        fake_converter.chmod(0o755)

        fake_pathfinder = tmp_path / "fake_pathfinder.py"
        fake_pathfinder.write_text(
            f"""#!/usr/bin/env python3
from pathlib import Path
import sys
route = {route!r}
Path(sys.argv[sys.argv.index("--routes-out") + 1]).write_text(__import__("json").dumps(route) + "\\n")
""",
            encoding="utf-8",
        )
        fake_pathfinder.chmod(0o755)

        wrapper_output = tmp_path / "wrapper_output.phys"
        result = benchmark.main(
            [
                str(input_phys),
                str(wrapper_output),
                "--logical-netlist",
                str(logical_netlist),
                "--device",
                str(device),
                "--schema-dir",
                str(schema_dir),
                "--interchange-to-csr",
                str(fake_converter),
                "--pathfinder",
                str(fake_pathfinder),
            ]
        )
        assert result == 0
        assert_routed_output(schema, wrapper_output)

    print("PathFinder benchmark writer test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
