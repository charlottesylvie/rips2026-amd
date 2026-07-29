# GPU profiling the PathFinder benchmark

## Evidence boundary (2026-07-27)

This file is a profiling procedure and telemetry reference, not a statement
that the current kernels have been measured. The retained `39155` Delta trace
used the generic all-light path with legacy predecessor materialization. It
predates automatic compact parents, distances-only execution, and the current
telemetry. A later forced-generic, unit-weight, `delta=1`, four-worker gfx1151
trace recorded 70,980 queries, 9,459,976 kernel dispatches, 4,614,765
`hipStreamSynchronize` calls, 4,011,171 `hipMemcpyAsync` calls, and 2,555,788
`hipMemsetAsync` calls. That is about 133 dispatches and 65 synchronizations
per query. Kernels were shorter than 50 microseconds 90.8% of the time, and no
kernel was active for 34.09 seconds (20.6% of the kernel span). Treat those
figures as the host-checked baseline, not as measurements of the opt-in
controller in this checkout. Reproduce correctness and profiler-free timing
before using archived percentages to rank current code.

The full profiling workflow has a manual one-time device stage followed by
three Make-driven per-test processes:

```text
xcvu3p.device
  -> device_to_routing_graph
     -> shared .devicegraph              (manual persistent prerequisite)

shared .devicegraph + test-case .phys/.netlist
  -> interchange_to_csr -> pathfinder (HIP) -> routes_to_phys
```

Profiling belongs around the inner `pathfinder` process. This keeps conversion
and route serialization out of GPU traces, while the Makefile's existing
`time` invocation still measures the complete per-test pipeline. The expensive
DeviceResources parsing and static graph formatting happen manually before
Make and are reused across tests. Make assumes the selected device graph exists
and validates it as a prerequisite; it does not invoke the preprocessor. The
Makefile passes a profiler prefix to `PathFinderFile` through
`PATHFINDER_PROFILE_COMMAND`; the wrapper applies it only when launching
`pathfinder`.

Generate the default persistent artifact manually with:

```bash
./device_to_routing_graph xcvu3p.device xcvu3p.full-poc-base-wire.devicegraph --full-device
```

Override `PATHFINDER_DEVICE_GRAPH` to use another manually preprocessed
artifact. Bounds and `--node-bounds-mode` are preprocessor options; they must
not be placed in `PATHFINDER_ARGS`. Give every policy a distinct artifact name
so profiling cannot accidentally consume a differently preprocessed graph.

## Build with ROCTx ranges

ROCTx ranges label graph upload, each net, each SSSP query, exact-unit versus
generic delta-stepping, input loading, and route output. They are optional so
normal and CPU-stub builds do not depend on ROCprofiler-SDK.

Add the following to the normal `pathfinder` build:

```bash
-DPATHFINDER_ENABLE_ROCTX -lrocprofiler-sdk-roctx
```

For example:

```bash
hipcc -std=c++17 -O3 -x hip -DBF10_NO_MAIN \
  -DPATHFINDER_ENABLE_ROCTX \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf10.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -pthread -lrocprofiler-sdk-roctx -o pathfinder
```

ROCm distributions package the header and library as
`rocprofiler-sdk-roctx`. Rebuild without the macro/library for final timing if
even the small marker overhead matters.

## First pass: controller A/B GPU timelines with rocprofv3

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller host-checked --parallel-net-workers 4" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-host-checked-w4

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 --parallel-net-workers 4" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-reduced-b4-w4
```

Four workers are the current measured classic Delta-Stepping timing control.
Keep the count explicit when comparing traces or profiler-free timings; it is
workload-specific and is not a universal default.

The host-checked controller is the default and the correctness reference. The
reduced-round-trip controller is opt-in. It capability-checks its device
controller and falls back before traversal when the selected configuration is
unsupported; callbacks also require host bucket boundaries. Always pair a
profile with telemetry from a separate run and confirm that
`effective_controller_modes.reduced_round_trip` equals the query count and
`controller_fallback_queries` is zero before calling it a reduced-controller
measurement. Batch size one is the state-machine equivalence control; batch
size four is the initial performance candidate, not a portable worker count.

Important: the current FPGA Interchange converter writes every CSR edge weight
as `1.0f`. A graph at or below the exact-unit specialization's `2^24`-row
limit can therefore select that path, while the documented 28,226,432-row full
graph remains generic because it exceeds the guard. In either case, use the
explicit force control to record the intent to profile generic execution on
the identical graph without changing weights, delta, destination-cost
semantics, or the iteration limit:

```bash
make ... PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller host-checked \
    --parallel-net-workers 4" \
  PATHFINDER_PROFILE=rocprofv3
```

For a reproducible weighted profile, `pathfinder` can replace the loaded CSR's
weights in memory. The family is derived from an explicit numeric delta: `unit`
uses `1`, `all-light` uses `delta/4`, `all-heavy` uses `4*delta`, and `mixed`
selects deterministically per CSR edge from `{0, delta/4, delta, 4*delta}`. For
example:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 17 \
    --parallel-net-workers 4" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-mixed-d2-seed17
```

`--delta-benchmark-weights` requires an explicitly supplied numeric `--delta`;
it is rejected with `--delta auto`. The seed is valid only for `mixed` and
defaults to zero. `--delta-force-generic` is independent of the family and is
the correct A/B control even for `unit`; it deliberately bypasses only the
exact-unit specialization. Custom converters and prebuilt weighted CSR inputs
remain useful when these four synthetic families do not model the workload.

The default integration runs:

```text
rocprofv3 --runtime-trace --stats --output-directory <run>/<benchmark> -- pathfinder ...
```

`--runtime-trace` captures HIP runtime calls, kernels, memory copies,
allocations, scratch use, and ROCTx markers. Current ROCm versions write RocPD
by default; use `rocpd` to generate CSV, summaries, or Perfetto traces. Add
profiler options without changing the Makefile, for example:

```bash
make ... PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_ARGS="--output-format pftrace"
```

If the program uses managed memory or page migration, add `--kfd-trace` on a
supported kernel. The current PathFinder primarily uses explicit HIP copies,
so memory-copy tracing is normally the more important CPU/GPU-transfer view.

Start by checking:

- total time and call count by kernel;
- gaps between kernels and time blocked in `hipStreamSynchronize` or copies;
- host-to-device and device-to-host byte counts, durations, and effective
  bandwidth;
- whether multiple worker streams overlap or serialize;
- allocation and scratch-memory activity inside repeated SSSP queries; and
- the `delta_step.generic` range rather than conversion or JSON output.

### UnitBFS production pass

Profile the production specialization separately; a Delta trace cannot
establish UnitBFS's launch, reset, or extraction costs:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=unit-bfs \
  PATHFINDER_ARGS="--parallel-net-workers 4" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=unit-bfs-w4
```

Repeat with 1 and 8 workers after a correctness stress run. Attribute the
one-time graph upload separately from per-query source/target setup, the
cooperative controller, final status copies, sparse level reset, target
measurement, compact path fill, and output copies. Current UnitBFS has no
algorithm-counter JSON analogous to Delta telemetry, so use ROCTx/runtime
traces or add diagnostic-only counters before making a structural claim.

## Opt-in algorithm telemetry

Hardware counters explain kernel behavior; Delta telemetry explains the work
the algorithm submitted. Enable it on a separate diagnostic run, not on the
timing baseline:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 17 \
    --delta-controller host-checked --parallel-net-workers 4 \
    --delta-telemetry" \
  PATHFINDER_PROFILE=none 2>&1 | tee delta-mixed-host-telemetry.log

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 17 \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 --parallel-net-workers 4 \
    --delta-telemetry" \
  PATHFINDER_PROFILE=none 2>&1 | tee delta-mixed-reduced-telemetry.log

grep '^{"type":"delta_stepping_telemetry"' \
  delta-mixed-host-telemetry.log delta-mixed-reduced-telemetry.log
```

After all workers join, `pathfinder` writes exactly one compact JSON line to
standard output with `type="delta_stepping_telemetry"`, `schema_version=3`,
and `scope="pathfinder_run"`. `queries` counts actual SSSP invocations, not net
slots; a net with no unresolved target leaves its slot uncollected.
`completed_queries` counts records whose cleanup and final synchronization
finished. `execution_paths` counts `exact_unit`, `compact_generic`,
`legacy_generic`, and `generic_distances_only`. The record also includes the
resolved numeric delta, runtime wavefront size, actual worker count,
auto-delta/multiplier values, force-mode flags, configured `controller_mode`
and `controller_batch_size`, per-query `effective_controller_modes` counts,
and `controller_fallback_queries`. The configured names use
`host_checked`/`reduced_round_trip` in JSON even though the CLI spells them
`host-checked`/`reduced-round-trip`. Counter fields are summed across queries;
the three queue high-water fields under `maxima` are maxima across queries,
not sums.

The counters are exact under these definitions:

| JSON field | Semantics |
| --- | --- |
| `outer_buckets_processed` | Generic outer bucket iterations. On the exact-unit path, the unit controller's bucket-round count: zero when every target is already a source, otherwise initialized for the source bucket and advanced when a discovered BFS depth crosses a Delta bucket. |
| `light_relaxation_rounds` | Generic same-bucket frontier-closure launches; on the exact-unit path, completed BFS depth expansions. |
| `heavy_edge_phases` | Generic heavy-phase passes executed after light closure. Zero when the heavy phase is skipped, including exact-unit and statically all-light runs. |
| `frontier_entries_processed` | Current-frontier queue tokens examined by the light or exact-unit expansion. Repeated tokens count repeatedly. |
| `active_vertices_processed` | Examined frontier tokens whose live distance belongs to the current bucket. Exact-unit frontier tokens are all active. |
| `stale_frontier_entries` | Generic frontier tokens rejected because their live distance is nonfinite or no longer belongs to the current bucket. |
| `light_edge_visits` | Outgoing edges examined during light closure. This includes edges later classified as heavy and every rescan. |
| `heavy_edge_visits` | Outgoing edges examined for vertices staged to the heavy phase, before per-edge heavy classification. This is not a unique-heavy-edge count. |
| `distance_atomic_attempts` | Generic atomic-min calls for eligible light/heavy candidates. Exact-unit traversal counts first-discovery CAS attempts made after observing infinity. Parent-key atomics are excluded. |
| `successful_distance_relaxations` | Atomic distance updates that strictly lowered a generic distance, or successful first discoveries on the exact-unit path. |
| `distance_cas_retries` | Failed distance-CAS contention retries. On exact-unit traversal, a competing first-discovery loss counts as one; parent-key CAS retries are excluded. |
| `current_queue_insertions` | Successful appends to a current/next frontier, including pending-to-current compaction. Initial source placement is excluded. |
| `pending_queue_insertions` | New future-bucket tokens appended after successful relaxations. Tokens retained while compacting an existing pending queue are not new insertions. |
| `heavy_queue_insertions` | Active vertices newly staged for a generic heavy phase. |
| `bucket_insertions` | Derived as `current_queue_insertions + pending_queue_insertions`; it excludes initial sources and heavy staging. |
| `pending_entry_examinations` | Pending tokens scanned by minimum-bucket reduction and compaction. A retained token can be examined and counted again in later scans. |
| `stale_pending_entry_examinations` | Those examinations whose token is inactive or no longer names a valid future bucket for that scan. It has the same repeated-examination behavior. |
| `reached_vertices` | Unique vertices whose distance became finite during the invocation, including deduplicated sources. |
| `controller_round_trips` | Explicitly counted host-visible status/count transfers used for control decisions. It is not a count of every HIP call or synchronization. |
| `scalar_d2h_readbacks` | Generic-controller scalar D2H transfers completed for host decisions or sparse-reset counts. |
| `explicit_stream_waits` | Completion waits issued by the generic controller and its cleanup, separate from scalar/status readback completion. Null-stream no-op helper calls, one-time workspace setup, and result/path/telemetry materialization waits are excluded. |
| `batched_status_readbacks` | Generic compact-path status tuples that include sparse-reset state, plus reduced-controller descriptor transfers that publish several fields with one D2H copy and wait. Non-generic paths report zero. |
| `controller_mode`, `controller_batch_size` | Run-level requested A/B configuration. These fields alone do not prove that the reduced controller executed. |
| `effective_controller_modes`, `controller_fallback_queries` | Query counts by controller actually used and the number that fell back to host checking. Reject a performance sample when either total does not match the intended A/B arm. |
| `compact_parent_fallback_events` | One when an automatic compact-parent vector-target query had to use legacy parents because its edge-to-source map was unavailable; otherwise zero. |
| `current_queue_high_water`, `pending_queue_high_water`, `heavy_queue_high_water` | Maximum observed queue entry counts within one invocation. The exact-unit current queue is append-only, so its peak is cumulative rather than one BFS layer's width. The run-level JSON reports the maximum per-query value; these are entries, not bytes. |

The exact-unit and generic definitions intentionally reflect their different
controllers, especially for atomic attempts, light rounds, and current-queue
peaks. Stratify comparisons by `execution_paths` rather than treating a mixed
aggregate as one homogeneous workload.

At the C++ level, pass a `DeltaSteppingCsrTelemetry` through
`DeltaSteppingCsrRunOptions` to obtain one per-invocation record. The workspace
resets it before dispatch; `completed` remains false if the invocation throws.
The record carries collected/completed state, execution path, resolved delta,
wavefront size, force flags, `has_vertex_costs`, and `all_edges_light` in
addition to the counters. Here `all_edges_light` means the implementation's
no-heavy-phase shortcut is active; it is conservatively false when destination
costs are installed. Parallel PathFinder workers write distinct net-indexed
records before the host aggregation above.

Telemetry-disabled dispatch uses a compile-time kernel specialization and does
not allocate, clear, copy, or pass the device counter buffer. When enabled,
hot-loop observations are reduced per block before global aggregation, but the
extra instructions, registers, shared state, atomics, and final copy can still
perturb performance. Compare instrumented runs by counter semantics and
selected execution path; measure speed with telemetry off.

## Low-level distances-only experiments

Callers that need the complete distance vector but no route or predecessor can
select the compile-time no-parent kernel explicitly:

```cpp
auto graph = std::make_shared<DeltaSteppingCsrGraph>(
    host_csr, stream, DeltaSteppingCsrStorageMode::kDistancesOnly);
DeltaSteppingCsrWorkspace workspace(graph, stream);
DeltaSteppingCsrResult result =
    workspace.run_distances(sources, numeric_delta, -1, stream);
```

`run_distances` populates only `result.dist`; predecessor, target, and compact
path vectors remain empty. Its kernels compile out parent writes, and entering
the mode releases mutable parent/path buffers left by an earlier path-capable
workspace run. Forced legacy-parent mode is rejected. A graph constructed with
`kDistancesOnly` rejects every path-producing `run` overload and also omits
the immutable `uint32_t` edge-to-source map, saving `4 B/E` when a path-capable
graph would otherwise allocate the eligible map. A normal path-capable graph
may still call `run_distances`, but retains that shared map.

The generic distances-only workspace is `40 B/V` of mutable device arrays,
excluding small scalar/source buffers, the immutable CSR, the returned host
distance vector, telemetry, and an optional `4 B/V` destination-cost array.
It avoids the compact mode's `8 B/V` parent key and the legacy mode's
additional predecessor storage; the corresponding generic mutable footprints
are `48 B/V` for compact parents and `60 B/V` for legacy parents before
target/path output buffers. The effective weight remains
`edge_weight(u,v) * vertex_cost(v)`. Resolve automatic delta from the current
host weights/costs with `delta_stepping_auto_delta` before calling this numeric
low-level API. Telemetry identifies this path as `generic_distances_only`, and
ROCTx identifies it as `delta_step.generic_distances_only`.

PathFinder itself needs paths and therefore does not select this API. Host
tests cover the new controls and telemetry aggregation, and the HIP regression
source covers device telemetry and distances-only allocation/transitions. That
HIP suite and performance work have not been run on a real AMD GPU here. Treat
all commands in this document as the outstanding hardware-validation
procedure, not as new measured results.

## Second pass: combined CPU/GPU timeline with rocprof-sys

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_PROFILE=rocprof-sys \
  PATHFINDER_PROFILE_RUN=delta-system
```

The Makefile configuration enables a Perfetto trace and profile, 100 Hz CPU
call-stack sampling, ROCm kernel/API/copy domains, ROCTx markers, and AMD SMI
GPU telemetry. This is the best view for answering whether GPU work is starved
by CPU orchestration, synchronization, or worker scheduling. Open the
generated `.proto` file in Perfetto.

For detailed host DRAM bandwidth, NUMA, and CPU memory-latency analysis, AMD
uProf can complement this trace. It is not wrapped into the benchmark recipe:
those are machine-wide CPU/platform measurements rather than PathFinder/HIP
events, and collection often needs system-specific permissions and event
selection.

`rocprof-sys` CLI options have evolved between releases. If the installed
version rejects the provided flags, keep the Make integration and supply the
locally supported command explicitly:

```bash
make ... PATHFINDER_PROFILE=custom \
  PATHFINDER_PROFILE_PREFIX="rocprof-sys-run <local-options> --"
```

There is no MPI or GPU-to-GPU communication in the current router. Here,
"networking" means PCIe or coherent CPU/GPU transfers. Use the HIP memory-copy
track for explicit transfers, KFD migration events for managed memory, and AMD
SMI/rocprof-sys telemetry for system-level utilization. xGMI/RCCL metrics only
become relevant after adding multi-GPU execution.

## Third pass: kernel counters with rocprof-compute

For AMD Instinct/CDNA GPUs, `rocprof-compute` is the most useful high-level
tool for VRAM/cache/compute diagnosis:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_PROFILE=rocprof-compute \
  PATHFINDER_PROFILE_RUN=delta-counters
```

The tool can rerun `pathfinder` in multiple passes to collect incompatible
counter groups. Therefore, the Makefile wall time from this mode is profiling
time, not a routing performance result. Start with a small representative
generic-weight CSR workload, identify the hot kernels in the timeline, then
use the installed version's kernel/dispatch filters.

Run `rocprof-compute profile --list-available-metrics` because report block
IDs and available counters depend on the GPU and tool version. Prioritize:

- System and hardware-block Speed-of-Light;
- Memory Chart and cache hit/miss behavior;
- HBM/VRAM read and write bandwidth;
- VALU, SALU, and VMEM utilization;
- wave occupancy, active waves, and issue/stall metrics;
- VGPR, SGPR, LDS, and scratch allocation;
- branch/divergence and instruction mix; and
- atomics and memory-pipeline stalls where supported.

DRAM latency is usually not exposed as one trustworthy scalar. Infer a
latency-bound kernel from low achieved bandwidth plus high memory-dependency
stalls, low eligible-wave count, cache misses, and insufficient occupancy.
A bandwidth-bound kernel instead approaches the memory Speed-of-Light while
compute units wait on VMEM. This distinction matters for delta-stepping's
irregular CSR reads and atomic relaxations.

Examples of narrower collection, subject to the installed tool's metric list:

```bash
make ... PATHFINDER_PROFILE=rocprof-compute \
  PATHFINDER_PROFILE_ARGS="--set compute_thruput_util --no-roof"

make ... PATHFINDER_PROFILE=rocprof-compute \
  PATHFINDER_PROFILE_ARGS="-b MEMORY_BLOCK_IDS --no-roof"
```

## Output and repeatability

Profile output is written to:

```text
pathfinder-profiles/<PATHFINDER_PROFILE_RUN>/<benchmark>/
```

Change `PATHFINDER_PROFILE_ROOT` to place large traces elsewhere. Any profile
mode other than `none` makes the routed `.phys` target run again, avoiding the
normal Make cache. Use an explicit `PATHFINDER_PROFILE_RUN` so related traces
have stable names and do not overwrite one another.

Profilers perturb launch timing, synchronization, clocks, and cache state. Use
profiles to explain behavior, then measure speed with `PATHFINDER_PROFILE=none`
over several fresh runs. Keep GPU clocks/power mode, worker count, numeric or
automatic delta, force mode, benchmark weight family and seed, telemetry
setting, benchmark input, device-graph artifact and preprocessing policy, and
thermal state fixed when comparing versions.

## Algorithm-specific interpretation

The most informative delta-stepping kernels and symptoms are:

| Observation | Likely next optimization |
| --- | --- |
| Pending reduction/compaction dominates | Circular buckets and a nonempty-bucket bitmap |
| Light/heavy relax kernels rescan many edges with few successful relaxations | Prepartition adjacency and reduce bucket-index divisions |
| Many short kernels separated by host gaps | HIP Graphs, device-resident orchestration, or a persistent kernel |
| Low occupancy or tiny grids | Batch independent nets/SSSP queries |
| High atomic stalls | Reduce duplicate relax attempts, aggregate within a wave/workgroup, or alter bucket width |
| High HBM traffic with low cache reuse | Compact offsets/state, improve frontier locality, and avoid repeated edge scans |
| Large copy/synchronization share | Keep status/path recovery on device longer and batch host transfers |

Hardware profiles alone cannot report algorithmic efficiency such as bucket
work, relax attempts per successful update, or repeated pending scans. Use the
opt-in JSON counters above for those ratios, but collect them separately from
the uninstrumented timing baseline because telemetry deliberately adds device
instrumentation.

## AMD validation checklist for the bounded controller

Nothing in this section was run on the host that implemented the bounded
controller pass: it has no HIP compiler, ROCm runtime, or AMD GPU. Run these
commands from the repository root on a gfx1151 AMD system and retain every
log. Keep both the reduced controller and generation-tagged membership opt-in
until their complete matrix and repeated explicit-stream stress pass.

Record the checkout and device first, then build the production router and the
two focused HIP regressions:

```bash
set -euo pipefail
mkdir -p amd-validation/bin amd-validation/logs \
  amd-validation/results amd-validation/work amd-validation/timing

git rev-parse HEAD | tee amd-validation/logs/git-head.txt
hipcc --version 2>&1 | tee amd-validation/logs/hipcc-version.txt
rocminfo > amd-validation/logs/rocminfo.txt

make ./PathFinderFile ./pathfinder
test -x ./interchange_to_csr
test -x ./routes_to_phys

hipcc -std=c++17 -O2 -pthread -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/tests/unit_bfs_hip_test.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -o amd-validation/bin/unit_bfs_hip_test

hipcc -std=c++17 -O2 -pthread -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/delta_stepping \
  CongestionFreeRouting/tests/delta_stepping_hip_test.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  -o amd-validation/bin/delta_stepping_hip_test
```

The UnitBFS binary runs automatic-compact and forced-wide rows through all
four extraction/visitation combinations: host offsets with sparse reset (the
default), device offsets with sparse reset, host offsets with generation
visitation, and device offsets with generation visitation. The Delta binary
must run the Cartesian controller matrix: host-checked and
reduced-round-trip (batch sizes 1 and 4), automatic-compact and forced-wide
row offsets, Boolean and generation membership, automatic compact and forced
legacy parents, and path-producing and distances-only runs. Fixtures cover
zero-weight SCCs, parallel edges, duplicate/multiple sources and targets,
unreachable targets, vertex costs, exclusive bounds, early targets, and
iteration limits. Randomized small graphs are compared with the CPU reference.
Run the ordinary matrices and then increase the explicit-stream reuse count:

```bash
./amd-validation/bin/unit_bfs_hip_test \
  2>&1 | tee amd-validation/logs/unit-bfs-matrix.log
UNIT_BFS_REUSE_STRESS_RUNS=1200 \
  ./amd-validation/bin/unit_bfs_hip_test \
  2>&1 | tee amd-validation/logs/unit-bfs-explicit-stream-stress.log

DELTA_REQUIRE_REDUCED_CONTROLLER=1 \
  ./amd-validation/bin/delta_stepping_hip_test \
  2>&1 | tee amd-validation/logs/delta-matrix.log
DELTA_REQUIRE_REDUCED_CONTROLLER=1 \
DELTA_MULTI_QUEUE_STRESS_RUNS=1200 \
  ./amd-validation/bin/delta_stepping_hip_test \
  2>&1 | tee amd-validation/logs/delta-explicit-stream-stress.log
```

`DELTA_REQUIRE_REDUCED_CONTROLLER=1` makes every instrumented reduced-mode
matrix/stress probe fail on a capability fallback; callback probes still
require their intentional host fallback. The Delta stress must keep at least
four nonblocking streams active, reuse each
workspace for thousands of queries, force generation rollover repeatedly, and
exercise a callback exception/abort followed by a successful query on the
same workspace. Confirm exact distance/path equivalence and outgoing-row edge
ownership after every run. Confirm telemetry reports reduced mode (not
fallback) for every intended reduced query, while callback queries report a
host-checked fallback. A passing low-level suite is required before
interpreting timing.

For `logicnets_jscl`, first create a small route-tree depth checker. It treats
the JSONL edge union as an outgoing graph, starts at every requested source,
requires every recorded sink to be reached, and fails unless the maximum
source-to-sink hop count is exactly 214:

```bash
cat > amd-validation/check_route_depth.py <<'PY'
import collections
import json
import pathlib
import sys

routes_path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
critical = 0
net_count = 0

with routes_path.open(encoding="utf-8") as routes:
    for line_number, line in enumerate(routes, 1):
        if not line.strip():
            continue
        net_count += 1
        net = json.loads(line)
        if not net.get("routed", False):
            raise SystemExit(f"line {line_number}: net is not fully routed")
        adjacency = collections.defaultdict(list)
        for edge in net["edges"]:
            adjacency[edge["from"]].append(edge["to"])
        distance = {source["node"]: 0 for source in net["sources"]}
        queue = collections.deque(distance)
        while queue:
            node = queue.popleft()
            for successor in adjacency[node]:
                if successor not in distance:
                    distance[successor] = distance[node] + 1
                    queue.append(successor)
        for sink in net["sinks"]:
            target = sink["node"]
            if not sink.get("reached", False) or target not in distance:
                raise SystemExit(
                    f"line {line_number}: sink {target} is not source-rooted"
                )
            critical = max(critical, distance[target])

if net_count == 0:
    raise SystemExit("route file is empty")
if critical != expected:
    raise SystemExit(f"critical path {critical}, expected {expected}")
print(f"critical path {critical}; {net_count} nets source-rooted and reached")
PY
```

Run the complete checker/analyzer pipeline first with the enabled UnitBFS
behavior and four workers. The low-level suite above, not this production
command, selects the still-opt-in device-offset and generation modes:

```bash
rm -f logicnets_jscl_PathFinderFile.phys \
  logicnets_jscl_PathFinderFile.phys.log \
  logicnets_jscl_PathFinderFile.check \
  logicnets_jscl_PathFinderFile.check.log \
  logicnets_jscl_PathFinderFile.wirelength

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=unit-bfs \
  PATHFINDER_ARGS="--parallel-net-workers 4 --strict-routing \
    --work-dir amd-validation/work/logicnets-unit-w4" \
  run-PathFinderFile \
  2>&1 | tee amd-validation/logs/logicnets-unit-w4.log

test "$(cat logicnets_jscl_PathFinderFile.check)" = PASS
python3 amd-validation/check_route_depth.py \
  amd-validation/work/logicnets-unit-w4/logicnets_jscl_PathFinderFile.routes.jsonl \
  214 | tee amd-validation/logs/logicnets-unit-w4-depth.log
cp logicnets_jscl_PathFinderFile.phys \
  amd-validation/results/logicnets-unit-w4.phys
cp logicnets_jscl_PathFinderFile.wirelength \
  amd-validation/results/logicnets-unit-w4.wirelength
```

Repeat the full routing validation with forced-generic classic Delta-Stepping,
`delta=1`, four workers, and each controller. Automatic compact row offsets
and Boolean membership are the production settings here; the wider storage,
membership, parent, and distances-only matrix remains in the low-level suite.
Run the host reference first:

```bash
rm -f logicnets_jscl_PathFinderFile.phys \
  logicnets_jscl_PathFinderFile.phys.log \
  logicnets_jscl_PathFinderFile.check \
  logicnets_jscl_PathFinderFile.check.log \
  logicnets_jscl_PathFinderFile.wirelength

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller host-checked \
    --parallel-net-workers 4 --strict-routing \
    --work-dir amd-validation/work/logicnets-delta-host-w4" \
  run-PathFinderFile \
  2>&1 | tee amd-validation/logs/logicnets-delta-host-w4.log

test "$(cat logicnets_jscl_PathFinderFile.check)" = PASS
python3 amd-validation/check_route_depth.py \
  amd-validation/work/logicnets-delta-host-w4/logicnets_jscl_PathFinderFile.routes.jsonl \
  214 | tee amd-validation/logs/logicnets-delta-host-w4-depth.log
cp logicnets_jscl_PathFinderFile.phys \
  amd-validation/results/logicnets-delta-host-w4.phys
cp logicnets_jscl_PathFinderFile.wirelength \
  amd-validation/results/logicnets-delta-host-w4.wirelength
```

Then remove the prior outputs and rerun the same checker with only these
controller arguments changed:

```bash
rm -f logicnets_jscl_PathFinderFile.phys \
  logicnets_jscl_PathFinderFile.phys.log \
  logicnets_jscl_PathFinderFile.check \
  logicnets_jscl_PathFinderFile.check.log \
  logicnets_jscl_PathFinderFile.wirelength

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 \
    --parallel-net-workers 4 --strict-routing \
    --work-dir amd-validation/work/logicnets-delta-reduced-b4-w4" \
  run-PathFinderFile \
  2>&1 | tee amd-validation/logs/logicnets-delta-reduced-b4-w4.log

test "$(cat logicnets_jscl_PathFinderFile.check)" = PASS
python3 amd-validation/check_route_depth.py \
  amd-validation/work/logicnets-delta-reduced-b4-w4/logicnets_jscl_PathFinderFile.routes.jsonl \
  214 | tee amd-validation/logs/logicnets-delta-reduced-b4-w4-depth.log
cp logicnets_jscl_PathFinderFile.phys \
  amd-validation/results/logicnets-delta-reduced-b4-w4.phys
cp logicnets_jscl_PathFinderFile.wirelength \
  amd-validation/results/logicnets-delta-reduced-b4-w4.wirelength
```

Collect telemetry in a separate, otherwise identical diagnostic run so its
kernel instrumentation cannot contaminate correctness timing:

```bash
rm -f logicnets_jscl_PathFinderFile.phys
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 --delta-telemetry \
    --parallel-net-workers 4 --strict-routing \
    --work-dir amd-validation/work/logicnets-delta-reduced-telemetry" \
  run-PathFinderFile \
  2>&1 | tee amd-validation/logs/logicnets-delta-reduced-telemetry.log

grep '^{"type":"delta_stepping_telemetry"' \
  amd-validation/logs/logicnets-delta-reduced-telemetry.log \
  > amd-validation/results/logicnets-delta-reduced-b4-w4-telemetry.json
```

Inspect the telemetry JSON and require `completed_queries == queries`,
`effective_controller_modes.reduced_round_trip == queries`, and
`controller_fallback_queries == 0`. Compare every route request and sink, not
only the final `.check` marker. Four workers are a controlled benchmark input,
not a default to embed in portable workspace behavior.

Finally collect profiler-free end-to-end samples and peak resident memory.
The commands below use GNU `time`, keep telemetry and profilers disabled,
perform one warm-up for each Delta controller, and then record five fresh
conversion, routing, and reconstruction runs per controller. Keep all inputs
except the controller flags identical:

```bash
env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
  logicnets_jscl_unrouted.phys \
  amd-validation/timing/delta-host-w4-warmup.phys \
  --logical-netlist logicnets_jscl.netlist \
  --device-graph xcvu3p.full-poc-base-wire.devicegraph \
  --work-dir amd-validation/work/timing-delta-host-w4-warmup \
  --sssp-engine delta-step --delta 1 --delta-force-generic \
  --delta-controller host-checked \
  --parallel-net-workers 4 --strict-routing

env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
  logicnets_jscl_unrouted.phys \
  amd-validation/timing/delta-reduced-b4-w4-warmup.phys \
  --logical-netlist logicnets_jscl.netlist \
  --device-graph xcvu3p.full-poc-base-wire.devicegraph \
  --work-dir amd-validation/work/timing-delta-reduced-b4-w4-warmup \
  --sssp-engine delta-step --delta 1 --delta-force-generic \
  --delta-controller reduced-round-trip \
  --delta-controller-batch-size 4 \
  --parallel-net-workers 4 --strict-routing

for validation_run in 1 2 3 4 5; do
  /usr/bin/time -f 'wall_seconds=%e\npeak_rss_kib=%M' \
    -o "amd-validation/logs/delta-host-w4-time-${validation_run}.txt" \
    env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
      logicnets_jscl_unrouted.phys \
      "amd-validation/timing/delta-host-w4-${validation_run}.phys" \
      --logical-netlist logicnets_jscl.netlist \
      --device-graph xcvu3p.full-poc-base-wire.devicegraph \
      --work-dir "amd-validation/work/timing-delta-host-w4-${validation_run}" \
      --sssp-engine delta-step --delta 1 --delta-force-generic \
      --delta-controller host-checked \
      --parallel-net-workers 4 --strict-routing
done

for validation_run in 1 2 3 4 5; do
  /usr/bin/time -f 'wall_seconds=%e\npeak_rss_kib=%M' \
    -o "amd-validation/logs/delta-reduced-b4-w4-time-${validation_run}.txt" \
    env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
      logicnets_jscl_unrouted.phys \
      "amd-validation/timing/delta-reduced-b4-w4-${validation_run}.phys" \
      --logical-netlist logicnets_jscl.netlist \
      --device-graph xcvu3p.full-poc-base-wire.devicegraph \
      --work-dir "amd-validation/work/timing-delta-reduced-b4-w4-${validation_run}" \
      --sssp-engine delta-step --delta 1 --delta-force-generic \
      --delta-controller reduced-round-trip \
      --delta-controller-batch-size 4 \
      --parallel-net-workers 4 --strict-routing
done

python3 - <<'PY'
import pathlib
import statistics

for label in ("delta-host-w4", "delta-reduced-b4-w4"):
    walls = []
    peaks = []
    for path in sorted(pathlib.Path("amd-validation/logs").glob(f"{label}-time-*.txt")):
        fields = dict(
            line.strip().split("=", 1)
            for line in path.read_text().splitlines()
            if "=" in line
        )
        walls.append(float(fields["wall_seconds"]))
        peaks.append(int(fields["peak_rss_kib"]))
    if len(walls) != 5:
        raise SystemExit(f"{label}: expected 5 samples, found {len(walls)}")
    print(
        label,
        f"median_wall_seconds={statistics.median(walls):.3f}",
        f"median_peak_rss_kib={statistics.median(peaks):.0f}",
        f"max_peak_rss_kib={max(peaks)}",
    )
PY
```

Compare exact route/checker outputs before comparing medians. Report all five
raw samples, dispersion, median peak RSS, and maximum peak RSS; do not infer a
kernel speedup from end-to-end samples alone. In separate timeline runs,
report dispatch, `hipStreamSynchronize`, `hipMemcpyAsync`, and
`hipMemsetAsync` counts per query; true kernel-union busy/idle time;
concurrency from zero through four active streams; top-kernel additive times;
controller/queue telemetry; and peak tracked GPU memory. Overlapping additive
kernel durations are not wall time.
