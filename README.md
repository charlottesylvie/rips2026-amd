# RIPS AMD 2026

FPGA routing prototype for the FPGA24/RWRoute-style benchmarks, with a
one-shot shortest-path router, FPGA Interchange conversion utilities, and HIP
kernels for shortest-path experiments.

The current benchmark-facing flow is the C++ `PathFinderFile` wrapper. A
one-time preprocessing step converts the FPGA Interchange `DeviceResources`
file into a reusable device-routing graph. The wrapper combines that graph
with each benchmark's physical and logical netlists, routes nets with a
one-shot shortest-path pass, and writes a routed `.phys` file. Overused nodes
are reported as diagnostics; they do not make the one-shot router fail.

## Project Status

- Main router path: `CongestionFreeRouting/pathfinder_router.cpp`.
- Main shortest-path engine: unit-weight HIP BFS in
  `CongestionFreeRouting/unit_bfs`; Delta Stepping remains available for
  comparison.
- Main interchange flow: `device_to_routing_graph` once per device/bounds
  policy, then `interchange_to_csr -> pathfinder -> routes_to_phys` per test
  case.
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
PATHFINDER_DEVICE_GRAPH ?= xcvu3p.full-poc-base-wire.devicegraph
```

Run a single benchmark through the contest infrastructure with:

```bash
make setup BENCHMARKS="boom_med_pb"
./device_to_routing_graph xcvu3p.device xcvu3p.full-poc-base-wire.devicegraph --full-device
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1
```

Device preprocessing is deliberately manual. Make assumes the selected
`PATHFINDER_DEVICE_GRAPH` already exists and validates it as a prerequisite; it
does not invoke `device_to_routing_graph` automatically.

Forward wrapper or PathFinder tuning options through `PATHFINDER_ARGS`:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --max-pathfinder-iters 20 --keep-work-dir"
```

Compare the congestion-free unit-BFS backend against delta-stepping while
keeping the same benchmark, preprocessed device graph, route writer, and
wrapper path:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=unit-bfs
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step
```

The delta backend also accepts a graph-aware bucket-width seed and a sweep
multiplier while retaining numeric widths as explicit overrides:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta auto --delta-multiplier 0.5"
```

Use the explicit force control for an exact-unit versus generic A/B run on the
same graph. It changes dispatch only; it does not rewrite weights or delta:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic"
```

Synthetic weighted runs are reproducible without rebuilding the CSR. This
example uses a fixed mixed-family seed and emits one aggregate telemetry JSON
line after all net workers join:

```bash
make ROUTER=PathFinderFile BENCHMARKS="boom_med_pb" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 17 \
    --delta-telemetry"
```

For AMD GPU profiling through the same Makefile path, see
[`CongestionFreeRouting/GPU_PROFILING.md`](CongestionFreeRouting/GPU_PROFILING.md).
The integration can wrap only the inner GPU router with `rocprofv3`,
`rocprof-sys`, or `rocprof-compute` while preserving end-to-end per-test Make
timing.

The Makefile validates the manually generated device graph before entering the
timed per-test recipe, then supplies that artifact and the matching `.netlist`
to the wrapper:

```make
%_PathFinderFile.phys: %_unrouted.phys %.netlist $(PATHFINDER_DEVICE_GRAPH)
	(time $(PATHFINDER_ROUTER_BIN) $< $@ \
	  --logical-netlist $*.netlist \
	  --device-graph $(PATHFINDER_DEVICE_GRAPH) $(PATHFINDER_ARGS)) ...
```

## Routing Pipeline

The complete manual-preprocessing and benchmark-wrapper workflow is:

```text
Manual one-time prerequisite (outside Make and per-test timing):
xcvu3p.device
  -> device_to_routing_graph
     -> xcvu3p.full-poc-base-wire.devicegraph

Per test case (inside the PathFinderFile timing):
<benchmark>_unrouted.phys + <benchmark>.netlist + the shared .devicegraph
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
| `CongestionFreeRouting/device_to_routing_graph.cpp` | Preprocesses invariant `DeviceResources` data into a reusable `.devicegraph` artifact. |
| `CongestionFreeRouting/interchange/device_routing_graph.cpp` / `.hpp` | Shared device-graph serialization, validation, lookup, and per-design filtering support. |
| `CongestionFreeRouting/interchange_to_csr.cpp` | Combines a preprocessed `.devicegraph` with one benchmark's `.phys` and `.netlist` inputs to produce the router CSR graph and metadata sidecar. |
| `CongestionFreeRouting/pathfinder.cpp` / `CongestionFreeRouting/pathfinder.hpp` | Implements the current one-shot source-to-sink shortest-path router over CSR input. |
| `CongestionFreeRouting/routes_to_phys.cpp` | Reconstructs a routed FPGA Interchange `PhysicalNetlist` from route JSONL output. |
| `CongestionFreeRouting/pathfinder_router.cpp` | Benchmark-facing C++ wrapper that runs the full conversion, routing, and reconstruction pipeline. |
| `CongestionFreeRouting/pathfinder_benchmark.py` | Older Python wrapper/reference implementation. Useful for reference and writer tests, but not the preferred benchmark-facing router. |
| `CongestionFreeRouting/tests` | CPU-only routing tests. |

### HIP Kernels

| Path | Purpose |
| --- | --- |
| `CongestionFreeRouting/delta_stepping` | Production outgoing-CSR Delta-Stepping implementation used by PathFinder. |
| `HIP_kernel/delta_stepping` | Legacy incoming-CSR Delta-Stepping experiment; it is not used by PathFinder. |
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
  CongestionFreeRouting/device_to_routing_graph.cpp \
  CongestionFreeRouting/interchange/device_routing_graph.cpp \
  "$SCHEMA_DIR"/DeviceResources.capnp.c++ \
  "$SCHEMA_DIR"/LogicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/References.capnp.c++ \
  -lcapnp -lkj -lz -o device_to_routing_graph

g++ -std=c++17 -O3 -I"$SCHEMA_DIR" \
  CongestionFreeRouting/interchange_to_csr.cpp \
  CongestionFreeRouting/interchange/device_routing_graph.cpp \
  "$SCHEMA_DIR"/PhysicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/LogicalNetlist.capnp.c++ \
  "$SCHEMA_DIR"/References.capnp.c++ \
  -lcapnp -lkj -lz -o interchange_to_csr

hipcc -std=c++17 -O3 -x hip -DBF10_NO_MAIN \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf10.cpp \
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
DEVICE_ROUTING_GRAPH=/path/to/xcvu3p.full-poc-base-wire.devicegraph \
INTERCHANGE_TO_CSR=/path/to/interchange_to_csr \
PATHFINDER_BIN=/path/to/pathfinder \
ROUTES_TO_PHYS=/path/to/routes_to_phys \
./PathFinderFile input_unrouted.phys output_routed.phys \
  --logical-netlist input.netlist
```

## Running Components Directly

### `device_to_routing_graph`

Preprocesses the device-wide routing graph once for every combination of raw
device, tile bounds, and node-bounds policy:

```bash
./device_to_routing_graph xcvu3p.device \
  xcvu3p.full-poc-base-wire.devicegraph --full-device
```

Useful preprocessing options:

| Option | Meaning |
| --- | --- |
| `--bounds <minX> <maxX> <minY> <maxY>` | Import a tile-coordinate subset. |
| `--nxroute-bounds` | Import the older proof-of-concept subset: `X36..X90`, `Y60..Y239`. |
| `--node-bounds-mode <mode>` | Select `poc-base-wire`, `fully-contained`, or `intersects`. |
| `--full-device` | Import every tile that has XY coordinates (default). |

Bounds and node-bounds mode are properties of the `.devicegraph`, not
per-benchmark converter or wrapper options. Use a different artifact name for
each policy, then select that prebuilt artifact with
`PATHFINDER_DEVICE_GRAPH`.

### `interchange_to_csr`

Combines a preprocessed device graph with one test case to produce its CSR
graph and metadata sidecar:

```bash
./interchange_to_csr <device.devicegraph> <unrouted.phys> \
  <logical.netlist> <output.csrbin> \
  --metadata <output.csrbin.ifmeta.bin>
```

The expensive parsing, coordinate extraction, PIP construction, and base-CSR
formatting have already happened in `device_to_routing_graph`. This stage
loads the two design netlists, extracts route requests and blockages, filters
the shared graph, and writes design-specific CSR and metadata outputs.

### `pathfinder`

Runs the CSR PathFinder prototype:

```bash
./pathfinder <graph.csrbin> [metadata.ifmeta.bin] \
  --routes-out <routes.jsonl>
```

Tuning options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--sssp-engine <unit-bfs\|delta-step\|bellman-ford\|bf10>` | `unit-bfs` | Shortest-path backend; `bf8`, `bf9`, and `bf10` are Bellman-Ford compatibility aliases. |
| `--use-delta-step` | unset | Shorthand for `--sssp-engine delta-step`. |
| `--delta <float\|auto>` | `1` | Explicit Delta-Stepping bucket width, or a graph-aware seed based on runtime wavefront size, average edge weight, and average out-degree. |
| `--delta-multiplier <float>` | `1` | Positive multiplier for sweeping around `--delta auto`; rejected with an explicit numeric width. |
| `--delta-force-generic` | unset | Bypass only the exact-unit Delta dispatch while preserving weights, delta, destination costs, and automatic compact-parent selection. |
| `--delta-force-legacy-parent` | unset | Select legacy predecessor recovery for generic vector-target Delta runs; combine with force-generic for a parent-policy A/B test. |
| `--delta-telemetry` | unset | Emit one aggregate Delta-Stepping telemetry JSON record after all net workers join. |
| `--delta-benchmark-weights <unit\|all-light\|all-heavy\|mixed>` | unset | Deterministically replace in-memory CSR weights for a benchmark; requires an explicit numeric delta. |
| `--delta-benchmark-weight-seed <uint>` | `0` | Seed the `mixed` family; rejected for every other family. |
| `--max-sssp-iters <int>` | `-1` | Delta buckets, unit-BFS depth, or Bellman-Ford rounds; `-1` uses the default. |
| `--capacity <int>` | `1` | Capacity used only for overuse diagnostics. |
| `--net-limit <count>` | unset | Route only the first `count` requests. |
| `--parallel-net-workers <count>` | `0` | Independent net workers; `0` enables engine-dependent auto-selection. Workers share one immutable CSR across worker-private search state. |
| `--diagnose-net <zero-based>` | unset | Replay the route-request prefix through this net and emit one UnitBFS diagnostic JSON record. Requires `--diagnose-sink`. |
| `--diagnose-sink <zero-based>` | unset | Compare the selected sink's routed per-sink result with fresh same-workspace CPU/GPU searches from both the original and exact expanded route trees. |
| `--routes-out <path>` | unset | Write routed PIP tree data as JSONL. |
| `--max-pathfinder-iters`, `--present-factor`, `--present-multiplier`, `--history-factor`, `--route-batch-size` | ignored | Compatibility-only options accepted by the one-shot router. |

Every Delta-specific control requires `--sssp-engine delta-step` or
`--use-delta-step`; other engines reject rather than ignore it. Force-generic
works with numeric or automatic delta. The multiplier requires automatic
delta, benchmark weight families require an explicit numeric delta, and the
seed is valid only with `mixed`. Force-generic and force-legacy-parent may be
combined: the first chooses generic execution and the second chooses its
parent representation. Force-legacy-parent by itself also makes the fixed
exact-unit parent path ineligible, so use force-generic with automatic parents
for a clean execution-path A/B comparison.

The converter emits exact unit weights. An automatic vector-target Delta
workspace uses its append-only exact-unit specialization only when the graph
has at most `2^24` rows, every edge weight is exactly `1`, destination costs
are absent, the iteration limit is negative, no progress callback is installed,
and execution and parent modes remain automatic. `--delta-force-generic`
bypasses only that dispatch; it does not rewrite weights, delta,
destination-cost semantics, or compact-versus-legacy parent policy. A finite
iteration limit or nonnull low-level callback also makes the call generic, but
those controls change execution semantics and should not be used as benchmark
forcing tricks.

Benchmark families are applied after loading and change only the in-memory
graph; the `.csrbin` file remains unchanged:

| Family | Replacement edge weights |
| --- | --- |
| `unit` | Every edge is `1`. |
| `all-light` | Every edge is `0.25 * delta`. |
| `all-heavy` | Every edge is `4 * delta`. |
| `mixed` | Each original CSR edge index makes a seeded deterministic choice from `{0, 0.25, 1, 4} * delta`. |

Automatic delta is resolved once before worker dispatch, so every worker uses
the same numeric width. Its graph-statistics scan is reported separately as
the optional `pathfinder.delta_auto_stats` ROCTX range. The low-level helper
also accepts destination vertex costs and computes the exact mean of
`edge_weight(u,v) * vertex_cost(v)`; the one-shot PathFinder currently has no
dynamic vertex-cost state and therefore resolves from immutable edge weights.
Low-level callers that update edge values or vertex costs must resolve a new
numeric width from the updated host data before the next run.

#### Delta-Stepping telemetry

Telemetry is disabled by default. Disabled dispatch uses separate
uninstrumented kernel instantiations and does not allocate, reset, or copy the
device counter buffer. `--delta-telemetry` selects instrumented kernels and is
intended for diagnosis, not clean wall-time measurement. After a successful
worker join, PathFinder writes one JSON line to standard output with
`type="delta_stepping_telemetry"` and `schema_version=1`; filter mixed logs on
that type. `queries` counts actual collected net searches, counter fields are
sums across searches, and queue fields under `maxima` are per-search maxima
combined with `max`, not sums. Execution-path counts distinguish exact-unit,
compact generic, legacy generic, and generic distances-only work.

The counters measure bucket/light/heavy rounds, frontier and edge visits,
distance atomic attempts/successes/CAS retries, logical queue insertions and
peaks, repeated pending-token examinations, unique reached vertices,
controller status round trips, and compact-parent fallbacks. In particular,
pending examinations may count one retained token in multiple scheduler scans,
edge visits are phase examinations rather than unique light/heavy edges, queue
peaks are entries rather than bytes, and controller round trips are counted
algorithmic status reads rather than all HIP calls. The precise field-by-field
definitions and limitations are in
[`CongestionFreeRouting/GPU_PROFILING.md`](CongestionFreeRouting/GPU_PROFILING.md#opt-in-algorithm-telemetry).

Low-level code requests a per-invocation record by passing
`DeltaSteppingCsrRunOptions{&telemetry}`. The workspace zero-resets the record
before dispatch, leaves `completed=false` on an exception, and requires
distinct records for concurrent workspaces.

#### Low-level distances-only API

Callers that need a complete distance vector without routes can omit all
parent work explicitly:

```cpp
auto graph = std::make_shared<DeltaSteppingCsrGraph>(
    adjacency, stream, DeltaSteppingCsrStorageMode::kDistancesOnly);
DeltaSteppingCsrWorkspace workspace(graph, stream);
DeltaSteppingCsrTelemetry telemetry;
auto result = workspace.run_distances(
    sources, delta, -1, DeltaSteppingCsrRunOptions{&telemetry}, stream);
```

`run_distances` is a compile-time no-parent generic specialization. Only
`result.dist` is populated; predecessor, target, and compact-path vectors stay
empty. Source distances remain zero, unreachable distances remain positive
infinity, and destination costs retain the effective-weight rule
`edge_weight(u,v) * vertex_cost(v)`. Forced legacy-parent mode is incompatible
and rejected.

The generic mutable core is approximately `40 B/V`, excluding small
scalar/source buffers, the immutable CSR, returned host output, telemetry, and
an optional `4 B/V` destination-cost array. This saves `8 B/V` against compact
generic parents (`48 B/V`) or `20 B/V` against legacy generic parents
(`60 B/V`) before variable target/path buffers. A strict `kDistancesOnly`
graph also omits the eligible compact-parent edge-to-source map, saving
`4 B/E` once per shared graph, and rejects every path-producing `run` overload.
Calling `run_distances` on an ordinary path-capable workspace is valid and
releases retained mutable path state, but that graph keeps its edge map so a
later path run can lazily rebuild the workspace buffers. `allocation_state()`
exposes the retained path-buffer state for tests and diagnostics.

PathFinder itself requires paths and does not use `run_distances`. Host tests
cover CLI forwarding and telemetry aggregation, and the HIP regression source
covers the new execution/allocation contracts. That HIP suite and performance
measurements on an AMD GPU remain unexecuted in this checkout.

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
  --device-graph xcvu3p.full-poc-base-wire.devicegraph
```

Useful wrapper options:

| Option | Meaning |
| --- | --- |
| `--device-graph <path>` | Reusable device-routing graph. Default: `xcvu3p.full-poc-base-wire.devicegraph`. Env: `DEVICE_ROUTING_GRAPH`. |
| `--work-dir <path>` | Directory for temporary CSR, metadata, and routes files. Provided work directories are kept. |
| `--keep-work-dir` | Keep automatically allocated temporary files. |
| `--interchange-to-csr <path>` | Override converter executable. Env: `INTERCHANGE_TO_CSR`. |
| `--pathfinder <path>` | Override PathFinder executable. Env: `PATHFINDER_BIN`. |
| `--routes-to-phys <path>` | Override route reconstructor. Env: `ROUTES_TO_PHYS`. |
| `--sssp-engine`, `--use-delta-step`, `--delta`, `--delta-multiplier`, `--delta-force-generic`, `--delta-force-legacy-parent`, `--delta-telemetry`, `--delta-benchmark-weights`, `--delta-benchmark-weight-seed`, `--max-sssp-iters`, `--net-limit`, `--parallel-net-workers`, `--capacity` | Forwarded to `pathfinder`. |
| `--max-pathfinder-iters`, `--present-factor`, `--present-multiplier`, `--history-factor`, `--route-batch-size` | Compatibility-only; forwarded to `pathfinder` and ignored. |

## File Formats And Artifacts

| Artifact | Producer | Description |
| --- | --- | --- |
| `<benchmark>_unrouted.phys` | Contest setup | Unrouted FPGA Interchange physical netlist. |
| `<benchmark>.netlist` | Contest setup | Matching logical netlist. |
| `xcvu3p.device` | RapidWright | FPGA Interchange device resources for the target part. |
| `.devicegraph` | `device_to_routing_graph` | Persistent device-wide CSR, node/PIP metadata, and lookup tables for one device/bounds policy. |
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

Device-graph serialization, lookup, masked filtering, and edge-attribute
alignment test:

```bash
g++ -std=c++17 -O2 \
  CongestionFreeRouting/tests/device_routing_graph_test.cpp \
  CongestionFreeRouting/interchange/device_routing_graph.cpp \
  -o /tmp/device_routing_graph_test

/tmp/device_routing_graph_test
```

Gzip/plain FPGAIF input integrity test (including truncated gzip streams):

```bash
g++ -std=c++17 -O2 \
  CongestionFreeRouting/tests/gzip_io_test.cpp \
  -lz -o /tmp/gzip_io_test

/tmp/gzip_io_test
```

CPU-only PathFinder logic test:

```bash
g++ -std=c++17 -O2 -pthread \
  -I Routing/tests/fake_hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/tests/pathfinder_bf10_cpu_stub_test.cpp \
  -o /tmp/pathfinder_bf10_cpu_stub_test

/tmp/pathfinder_bf10_cpu_stub_test
```

Targeted UnitBFS path diagnostic (requires the normal AMD HIP `pathfinder`
build and an existing CSR/metadata pair):

```bash
./pathfinder design.csrbin design.csrbin.ifmeta.bin \
  --sssp-engine unit-bfs \
  --parallel-net-workers 4 \
  --diagnose-net 27825 \
  --diagnose-sink 0
```

The diagnostic replays every request through the selected zero-based net so
the chosen worker has realistic workspace-reuse history, then exits without
writing routes. Its final JSON line compares the actual per-sink query from the
expanded route tree with independent CPU and GPU reference queries. Run the
same selection with one and four workers to compare the null-stream controller
with explicit worker streams. The JSON key `raw_batched` is retained for
compatibility, but now contains the routed per-sink query.

Delta benchmark argument parsing/forwarding test:

```bash
python3 CongestionFreeRouting/tests/pathfinder_benchmark_args_test.py
```

C++ router forwarding test:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  CongestionFreeRouting/tests/pathfinder_router_args_test.cpp \
  -o /tmp/pathfinder_router_args_test

/tmp/pathfinder_router_args_test
```

Unit-BFS compact-offset and batched-level correctness test (requires an AMD HIP
system):

```bash
hipcc -std=c++17 -O2 -pthread -x hip \
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
- Full-device preprocessing is the default. Pass `--nxroute-bounds` or
  explicit `--bounds` to `device_to_routing_graph` only for intentionally
  bounded experiments, give each policy a distinct `.devicegraph` name, and
  validate that the selected region covers every benchmark source and sink.
- On macOS, prefer CPU-stub tests unless ROCm/HIP headers and runtime are
  available through a container or remote machine.
- Generated benchmark assets, routed outputs, logs, and temporary work
  directories should not be committed unless they are intentionally added as
  small fixtures.

## Caveats

- `xcvu3p.device` is not stored in this repository; the benchmark Makefile or
  RapidWright normally generates it.
- The benchmark Makefile does not generate `.devicegraph` files. Run
  `device_to_routing_graph` manually before a `PathFinderFile` benchmark.
- `device_to_routing_graph.cpp`, `interchange_to_csr.cpp`, and
  `routes_to_phys.cpp` cannot compile without their generated FPGA Interchange
  Cap'n Proto C++ files.
- `pathfinder` and most HIP kernel tests require ROCm/HIP and `hipcc`; CPU-only
  tests use the fake HIP headers in `Routing/tests/fake_hip`.
- Temporary files and downloaded reference files should be removed after tests.
