# RIPS AMD 2026

FPGA routing prototype for the FPGA24/RWRoute-style benchmarks, with a
one-shot shortest-path router, FPGA Interchange conversion utilities, and HIP
kernels for shortest-path experiments.

The current benchmark-facing flow is the C++ `PathFinderFile` wrapper. It
converts FPGA Interchange inputs into an internal CSR graph, routes nets with a
one-shot shortest-path pass, and writes a routed `.phys` file. Overused nodes
are reported as diagnostics; they do not make the one-shot router fail.

## Project Status

- Main router path: `CongestionFreeRouting/pathfinder_router.cpp`.
- Main shortest-path engine: unit-weight HIP BFS in
  `CongestionFreeRouting/unit_bfs`; Delta Stepping remains available for
  comparison.
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

`CongestionFreeRouting/pathfinder_router.cpp` is the C++ wrapper intended to expose that
contest-style interface. The compiled executable name can match whatever router
name the benchmark Makefile uses, for example `PathFinderFile`.

Run the contest-provided Makefile path with:

```make
ROUTER=PathFinderFile
PATHFINDER_ROUTER_BIN ?= ./PathFinderFile
PATHFINDER_SSSP_ENGINE ?= unit-bfs
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

Compare the congestion-free unit-BFS backend against delta-stepping while
keeping the same benchmark, bounds, route writer, and wrapper path:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=unit-bfs
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step
```

For AMD GPU profiling through the same Makefile path, see
[`CongestionFreeRouting/GPU_PROFILING.md`](CongestionFreeRouting/GPU_PROFILING.md).
The integration can wrap only the inner GPU router with `rocprofv3`,
`rocprof-sys`, or `rocprof-compute` while preserving end-to-end Make timing.

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
| `CongestionFreeRouting/interchange_to_csr.cpp` | Converts FPGA Interchange `.phys`, `.netlist`, and `.device` inputs into the router CSR graph and metadata sidecar. |
| `CongestionFreeRouting/pathfinder.cpp` / `CongestionFreeRouting/pathfinder.hpp` | Implements the current one-shot source-to-sink shortest-path router over CSR input. |
| `CongestionFreeRouting/routes_to_phys.cpp` | Reconstructs a routed FPGA Interchange `PhysicalNetlist` from route JSONL output. |
| `CongestionFreeRouting/pathfinder_router.cpp` | Benchmark-facing C++ wrapper that runs the full conversion, routing, and reconstruction pipeline. |
| `CongestionFreeRouting/pathfinder_benchmark.py` | Older Python wrapper/reference implementation. Useful for reference and writer tests, but not the preferred benchmark-facing router. |
| `CongestionFreeRouting/tests` | CPU-only routing tests. |

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
  CongestionFreeRouting/interchange_to_csr.cpp \
  "$SCHEMA_DIR"/DeviceResources.capnp.c++ \
  "$SCHEMA_DIR"/PhysicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/LogicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/References.capnp.c++ \
  -lcapnp -lkj -lz -o interchange_to_csr

hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -pthread -o pathfinder

g++ -std=c++17 -O3 -I"$SCHEMA_DIR" \
  CongestionFreeRouting/routes_to_phys.cpp \
  "$SCHEMA_DIR"/PhysicalNetlist.capnp.c++ \
  -lcapnp -lkj -lz -o routes_to_phys

g++ -std=c++17 -O2 CongestionFreeRouting/pathfinder_router.cpp -o PathFinderFile
```

To include optional ROCTx ranges in profiler traces, add
`-DPATHFINDER_ENABLE_ROCTX -lrocprofiler-sdk-roctx` to the `hipcc` command.

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
| `--nxroute-bounds` | Import the older proof-of-concept subset: `X36..X90`, `Y60..Y239`. |
| `--node-bounds-mode <mode>` | Select `poc-base-wire`, `fully-contained`, or `intersects`. |
| `--full-device` | Import every tile that has XY coordinates. |

The converter and `PathFinderFile` wrapper import the full device by default.
Use `--nxroute-bounds` only for an explicit comparison with the older
proof-of-concept flow.

### `pathfinder`

Runs the CSR PathFinder prototype:

```bash
./pathfinder <graph.csrbin> [metadata.ifmeta.bin] \
  --routes-out <routes.jsonl>
```

Tuning options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--sssp-engine <unit-bfs\|delta-step>` | `unit-bfs` | Shortest-path backend. |
| `--use-delta-step` | unset | Shorthand for `--sssp-engine delta-step`. |
| `--delta <float>` | `4` | Delta-stepping bucket width; meaningful for delta-step. |
| `--max-sssp-iters <int>` | `-1` | Delta-step bucket rounds or unit-BFS depth cap; `-1` uses the default. |
| `--capacity <int>` | `1` | Capacity used only for overuse diagnostics. |
| `--net-limit <count>` | unset | Route only the first `count` requests. |
| `--parallel-net-workers <count>` | `0` | Independent net workers; `0` auto-selects up to 8 from available GPU memory. Both engines share one immutable CSR across worker-private search state. |
| `--routes-out <path>` | unset | Write routed PIP tree data as JSONL. |
| `--max-pathfinder-iters`, `--present-factor`, `--present-multiplier`, `--history-factor`, `--route-batch-size` | ignored | Compatibility-only options accepted by the one-shot router. |

The converter emits exact unit weights. For unlimited multi-target calls with
no vertex-cost override or progress callback, the delta backend automatically
uses its equivalent append-only unit-weight specialization; weighted and
limited-iteration calls retain generic delta-stepping.

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
| `--sssp-engine`, `--use-delta-step`, `--delta`, `--max-sssp-iters`, `--parallel-net-workers`, `--capacity` | Forwarded to `pathfinder`. |
| `--max-pathfinder-iters`, `--present-factor`, `--present-multiplier`, `--history-factor`, `--route-batch-size` | Compatibility-only; forwarded to `pathfinder` and ignored. |
| `--full-device`, `--nxroute-bounds`, `--bounds`, `--node-bounds-mode` | Forwarded to `interchange_to_csr`; full-device conversion is the default. |

## File Formats And Artifacts

| Artifact | Producer | Description |
| --- | --- | --- |
| `<benchmark>_unrouted.phys` | Contest setup | Unrouted FPGA Interchange physical netlist. |
| `<benchmark>.netlist` | Contest setup | Matching logical netlist. |
| `xcvu3p.device` | RapidWright | FPGA Interchange device resources for the target part. |
| `.csrbin` | `interchange_to_csr` | Internal CSR routing graph. |
| `.csrbin.ifmeta.bin` | `interchange_to_csr` | Metadata sidecar with string table, node coordinate ranges, tile/wire type IDs, PIP data, site pins, logical summaries, and route requests. |
| `.routes.jsonl` | `pathfinder` | One JSON object per net containing sources, sinks, and selected PIP edges. |
| `<benchmark>_PathFinderFile.phys` | `routes_to_phys` | Routed physical netlist. |
| `.check` / `.check.log` | Makefile checker | `PASS` or `FAIL` plus checker output. |
| `.wirelength` | Makefile wirelength analyzer | Routed design wirelength report. |

CSR orientation is outgoing-edge: row `u`, column `v` represents directed edge
`u -> v`. Edge weights are stored as a separate `float` array aligned with
`colind`; node coordinate ranges and tile/wire type metadata stay in the sidecar.

## Testing

CPU-only PathFinder logic test:

```bash
g++ -std=c++17 -O2 -pthread \
  -I Routing/tests/fake_hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/tests/pathfinder_cpu_stub_test.cpp \
  -o /tmp/congestion_free_pathfinder_cpu_stub_test

/tmp/congestion_free_pathfinder_cpu_stub_test
```

Unit-BFS compact-offset and batched-level correctness test (requires an AMD HIP
system):

```bash
hipcc -std=c++17 -O2 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/tests/unit_bfs_hip_test.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -o /tmp/unit_bfs_hip_test

/tmp/unit_bfs_hip_test
```

Production outgoing-CSR delta-stepping regression test (requires an AMD HIP
system):

```bash
hipcc -std=c++17 -O2 -pthread -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/delta_stepping \
  CongestionFreeRouting/tests/delta_stepping_hip_test.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  -o /tmp/delta_stepping_hip_test

/tmp/delta_stepping_hip_test
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

Legacy incoming-CSR Delta Stepping versus Dijkstra verification (this tests the
separate implementation under `HIP_kernel/delta_stepping`, not the production
outgoing-CSR router implementation):

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

- `CongestionFreeRouting/pathfinder.cpp` does not directly read or write FPGA Interchange
  `.phys` files; keep interchange concerns in `interchange_to_csr.cpp` and
  `routes_to_phys.cpp`.
- PathFinder routes against internal route requests from the metadata sidecar.
  Use `--routes-out` and `--keep-work-dir` when debugging route-tree output.
- Full-device conversion is the default. Use `--nxroute-bounds` or explicit
  `--bounds` only for intentionally bounded experiments, and validate that the
  selected region covers every source and sink being benchmarked.
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
