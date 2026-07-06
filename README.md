# RIPS AMD 2026

FPGA routing prototype for the FPGA24/RWRoute-style benchmarks, with a
PathFinder-style router, FPGA Interchange conversion utilities, and HIP kernels
for shortest-path experiments.

The current benchmark-facing flow is the C++ `PathFinderFile` wrapper. It
converts FPGA Interchange inputs into an internal CSR graph, routes nets with a
PathFinder-style rip-up/reroute loop, and writes a routed `.phys` file.

## Project Status

- Main router path: `Routing/pathfinder_router.cpp`.
- Main shortest-path engine: targeted Delta Stepping in
  `HIP_kernel/delta_stepping`.
- Main interchange flow:
  `interchange_to_csr -> pathfinder -> routes_to_phys`.
- Local CPU-only tests are available for routing logic, but full router builds
  require ROCm/HIP, Cap'n Proto C++ support, and FPGA Interchange schema files.

## Benchmark Interface

The benchmark Makefile expects a router executable that accepts an unrouted
physical netlist and writes a routed physical netlist:

```bash
./<router-name> <benchmark>_unrouted.phys <benchmark>_<router-name>.phys
```

`Routing/pathfinder_router.cpp` is the C++ wrapper intended to expose that
contest-style interface. The compiled executable name can match whatever router
name the benchmark Makefile uses, for example `PathFinderFile`.

The repository Makefile currently defaults to:

```make
ROUTER ?= PathFinderFile
PATHFINDER_ROUTER_BIN ?= ./PathFinderFile
```

Run a single benchmark through the contest infrastructure with:

```bash
make setup BENCHMARKS="boom_med_pb"
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1
```

Forward wrapper or PathFinder tuning options through `PATHFINDER_ARGS`:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_ARGS="--delta 2 --max-pathfinder-iters 20 --keep-work-dir"
```

The Makefile rule supplies the matching `.netlist` and `xcvu3p.device`:

```make
%_PathFinderFile.phys: %_unrouted.phys %.netlist xcvu3p.device
	(/usr/bin/time $(PATHFINDER_ROUTER_BIN) $< $@ \
	  --logical-netlist $*.netlist --device xcvu3p.device $(PATHFINDER_ARGS)) ...
```

## Routing Pipeline

The C++ benchmark wrapper orchestrates this flow:

```text
<benchmark>_unrouted.phys + <benchmark>.netlist + xcvu3p.device
  -> interchange_to_csr
     -> <benchmark>.csrbin
     -> <benchmark>.csrbin.ifmeta.bin
  -> pathfinder
     -> <benchmark>.routes.jsonl
  -> routes_to_phys
     -> <benchmark>_PathFinderFile.phys
```

The intermediate work directory is removed by default when the wrapper creates
it automatically. Pass `--keep-work-dir` or provide `--work-dir <path>` to keep
the generated `.csrbin`, metadata, and routes files for debugging.

## Repository Layout

### Routing

| Path | Purpose |
| --- | --- |
| `Routing/interchange_to_csr.cpp` | Converts FPGA Interchange `.phys`, `.netlist`, and `.device` inputs into the router CSR graph and metadata sidecar. |
| `Routing/pathfinder.cpp` / `Routing/pathfinder.hpp` | Implements the current PathFinder-style rip-up/reroute loop over CSR input. |
| `Routing/routes_to_phys.cpp` | Reconstructs a routed FPGA Interchange `PhysicalNetlist` from PathFinder JSONL output. |
| `Routing/pathfinder_router.cpp` | Benchmark-facing C++ wrapper that runs the full conversion, routing, and reconstruction pipeline. |
| `Routing/pathfinder_benchmark.py` | Older Python wrapper/reference implementation. Useful for reference and writer tests, but not the preferred benchmark-facing router. |
| `Routing/tests` | CPU-only routing tests and Python writer regression tests. |

### HIP Kernels

| Path | Purpose |
| --- | --- |
| `HIP_kernel/delta_stepping` | Target-aware Delta Stepping implementation used by PathFinder. |
| `HIP_kernel/bellman_ford` | Bellman-Ford experiments and correctness tests. Result metadata has target fields, but Bellman-Ford does not currently target-early-stop like Delta Stepping. |
| `HIP_kernel/minplus_mm` | Dense and sparse min-plus matrix multiplication experiments. |
| `HIP_kernel/faster_GPU_algo` | Additional GPU SSSP experiments and porting attempts. |

Real HIP compilation requires ROCm/HIP and `hipcc`. Local macOS builds will fail
on `hip/hip_runtime.h` unless they are using fake test headers.

## Prerequisites

The full benchmark flow expects:

- A C++17 compiler.
- ROCm/HIP with `hipcc` for `pathfinder` and the HIP kernels.
- Cap'n Proto C++ headers and libraries: `capnp`, `kj`.
- `zlib`.
- Generated FPGA Interchange C++ schema files, including
  `DeviceResources.capnp.c++`, `PhysicalNetlist.capnp.c++`,
  `LogicalNetlist.capnp.c++`, `References.capnp.c++`, and matching headers.
- RapidWright/Gradle contest infrastructure for benchmark setup, device-file
  generation, checking, and scoring.
- Python 3 for legacy wrappers and regression tests.

The benchmark files and `xcvu3p.device` are not stored in this repository.
`make setup` downloads or generates the contest assets when the surrounding
contest infrastructure is present.

## Build Notes

Assuming generated FPGA Interchange C++ schema files are available in
`fpga-interchange-schema/interchange`, the main components can be built with:

```bash
SCHEMA_DIR=fpga-interchange-schema/interchange

g++ -std=c++17 -O3 -I"$SCHEMA_DIR" \
  Routing/interchange_to_csr.cpp \
  "$SCHEMA_DIR"/DeviceResources.capnp.c++ \
  "$SCHEMA_DIR"/PhysicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/LogicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/References.capnp.c++ \
  -lcapnp -lkj -lz -o interchange_to_csr

hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/delta_stepping/src \
  Routing/pathfinder.cpp \
  HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp \
  -o pathfinder

g++ -std=c++17 -O3 -I"$SCHEMA_DIR" \
  Routing/routes_to_phys.cpp \
  "$SCHEMA_DIR"/PhysicalNetlist.capnp.c++ \
  -lcapnp -lkj -lz -o routes_to_phys

g++ -std=c++17 -O2 Routing/pathfinder_router.cpp -o PathFinderFile
```

If the helper binaries are not in the repository root, point the wrapper at
them with options or environment variables:

```bash
INTERCHANGE_TO_CSR=/path/to/interchange_to_csr \
PATHFINDER_BIN=/path/to/pathfinder \
ROUTES_TO_PHYS=/path/to/routes_to_phys \
./PathFinderFile input_unrouted.phys output_routed.phys \
  --logical-netlist input.netlist --device xcvu3p.device
```

## Running Components Directly

### `interchange_to_csr`

Converts FPGA Interchange inputs into the CSR graph and metadata sidecar:

```bash
./interchange_to_csr <unrouted.phys> <logical.netlist> <output.csrbin> \
  --device xcvu3p.device \
  --metadata <output.csrbin.ifmeta.bin>
```

It also accepts the device path as the first positional argument:

```bash
./interchange_to_csr <device.device> <unrouted.phys> <logical.netlist> \
  <output.csrbin> --metadata <output.csrbin.ifmeta.bin>
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--bounds <minX> <maxX> <minY> <maxY>` | Import a tile-coordinate subset. |
| `--node-bounds-mode <mode>` | Select `poc-base-wire`, `fully-contained`, or `intersects`. |
| `--full-device` | Import every tile that has XY coordinates. |

Default bounds match the older proof-of-concept flow: `X36..X90` and
`Y60..Y239`.

### `pathfinder`

Runs the CSR PathFinder prototype:

```bash
./pathfinder <graph.csrbin> [metadata.ifmeta.bin] \
  --routes-out <routes.jsonl>
```

Tuning options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--delta <float>` | `4` | Delta-stepping bucket width. |
| `--max-pathfinder-iters <int>` | `30` | Rip-up/reroute rounds. |
| `--max-sssp-iters <int>` | `-1` | Delta-stepping bucket rounds; `-1` uses the default. |
| `--capacity <int>` | `1` | Routing-resource capacity. |
| `--present-factor <float>` | `1` | Initial present congestion factor. |
| `--present-multiplier <float>` | `2` | Per-iteration present-factor multiplier. |
| `--history-factor <float>` | `1` | Historical congestion increment. |
| `--net-limit <count>` | unset | Route only the first `count` requests. |
| `--routes-out <path>` | unset | Write routed PIP tree data as JSONL. |

### `routes_to_phys`

Reconstructs a routed FPGA Interchange physical netlist:

```bash
./routes_to_phys <unrouted.phys> <metadata.ifmeta.bin> \
  <routes.jsonl> <output.phys>
```

The route reconstructor validates that route entries match metadata route
requests, inserts `pip` route branches, and adds sink `sitePin` leaves.

### `PathFinderFile`

Runs the full benchmark-facing C++ pipeline:

```bash
./PathFinderFile <input_unrouted.phys> <output_routed.phys> \
  --logical-netlist <input.netlist> \
  --device xcvu3p.device
```

Useful wrapper options:

| Option | Meaning |
| --- | --- |
| `--work-dir <path>` | Directory for temporary CSR, metadata, and routes files. Provided work directories are kept. |
| `--keep-work-dir` | Keep automatically allocated temporary files. |
| `--interchange-to-csr <path>` | Override converter executable. Env: `INTERCHANGE_TO_CSR`. |
| `--pathfinder <path>` | Override PathFinder executable. Env: `PATHFINDER_BIN`. |
| `--routes-to-phys <path>` | Override route reconstructor. Env: `ROUTES_TO_PHYS`. |
| `--delta`, `--max-pathfinder-iters`, `--max-sssp-iters`, `--capacity`, `--present-factor`, `--present-multiplier`, `--history-factor` | Forwarded to `pathfinder`. |
| `--full-device`, `--bounds`, `--node-bounds-mode` | Forwarded to `interchange_to_csr`. |

## File Formats And Artifacts

| Artifact | Producer | Description |
| --- | --- | --- |
| `<benchmark>_unrouted.phys` | Contest setup | Unrouted FPGA Interchange physical netlist. |
| `<benchmark>.netlist` | Contest setup | Matching logical netlist. |
| `xcvu3p.device` | RapidWright | FPGA Interchange device resources for the target part. |
| `.csrbin` | `interchange_to_csr` | Internal CSR routing graph. |
| `.csrbin.ifmeta.bin` | `interchange_to_csr` | Metadata sidecar with string table, PIP data, site pins, logical summaries, and route requests. |
| `.routes.jsonl` | `pathfinder` | One JSON object per net containing sources, sinks, and selected PIP edges. |
| `<benchmark>_PathFinderFile.phys` | `routes_to_phys` | Routed physical netlist. |
| `.check` / `.check.log` | Makefile checker | `PASS` or `FAIL` plus checker output. |
| `.wirelength` | Makefile wirelength analyzer | Routed design wirelength report. |

CSR orientation is incoming-edge: row `v`, column `u` represents directed edge
`u -> v`. This convention appears in both the converter and the shortest-path
kernels.

## Testing

CPU-only PathFinder logic test:

```bash
g++ -std=c++17 -O2 \
  -I Routing/tests/fake_hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/delta_stepping/src \
  Routing/tests/pathfinder_cpu_stub_test.cpp \
  -o /tmp/pathfinder_cpu_stub_test

/tmp/pathfinder_cpu_stub_test
```

Python route-writer regression test:

```bash
python3 Routing/tests/pathfinder_benchmark_writer_test.py
```

HIP Bellman-Ford correctness test:

```bash
hipcc -std=c++17 -O3 -x hip \
  HIP_kernel/bellman_ford/tests/test_correctness.cpp \
  HIP_kernel/bellman_ford/src/bellman_ford_hip.cpp \
  HIP_kernel/minplus_mm/src/minplus_hip.cpp \
  -o test_correctness

./test_correctness --quick
```

Delta Stepping versus Dijkstra verification:

```bash
hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/delta_stepping/src \
  HIP_kernel/delta_stepping/testing/verify_bf_delta_vs_dijkstra.cpp \
  HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp \
  HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
  -o verify_bf_delta_vs_dijkstra

./verify_bf_delta_vs_dijkstra
```

Full contest-style validation is done through the Makefile:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1
```

That produces a routed `.phys`, runs `CheckPhysNetlist`, writes wirelength
output, and computes the benchmark score.

## Development Notes

- `Routing/pathfinder.cpp` does not directly read or write FPGA Interchange
  `.phys` files; keep interchange concerns in `interchange_to_csr.cpp` and
  `routes_to_phys.cpp`.
- PathFinder routes against internal route requests from the metadata sidecar.
  Use `--routes-out` and `--keep-work-dir` when debugging route-tree output.
- The default converter bounds are intentionally smaller than the full device
  to keep early experiments tractable. Use `--full-device` only when the build
  and runtime environment can handle the larger graph.
- On macOS, prefer CPU-stub tests unless ROCm/HIP headers and runtime are
  available through a container or remote machine.
- Generated benchmark assets, routed outputs, logs, and temporary work
  directories should not be committed unless they are intentionally added as
  small fixtures.

## Caveats

- `xcvu3p.device` is not stored in this repository; the benchmark Makefile or
  RapidWright normally generates it.
- `routes_to_phys.cpp` and `interchange_to_csr.cpp` cannot compile without the
  generated FPGA Interchange Cap'n Proto C++ files.
- `pathfinder` and most HIP kernel tests require ROCm/HIP and `hipcc`; CPU-only
  tests use the fake HIP headers in `Routing/tests/fake_hip`.
- Temporary files and downloaded reference files should be removed after tests.
