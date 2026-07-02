#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        assert len(pip_branch.branches) == 1
        sink_branch = pip_branch.branches[0]
        assert sink_branch.routeSegment.which() == "sitePin"
        sink = sink_branch.routeSegment.sitePin
        assert strings[sink.site] == "SINK_SITE"
        assert strings[sink.pin] == "SINK_PIN"


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

        logical_netlist = tmp_path / "input.netlist"
        logical_netlist.write_bytes(b"logical")
        device = tmp_path / "xcvu3p.device"
        device.write_bytes(b"device")

        fake_converter = tmp_path / "fake_interchange_to_csr.py"
        fake_converter.write_text(
            """#!/usr/bin/env python3
from pathlib import Path
import sys
Path(sys.argv[3]).write_bytes(b"csr")
Path(sys.argv[sys.argv.index("--metadata") + 1]).write_bytes(b"metadata")
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
