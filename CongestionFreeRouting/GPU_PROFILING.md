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

The superseded independent-worker cooperative path was exercised on gfx1151.
One retained four-worker run completed 27,960 queries with 170,929 logical
cooperative publications/round trips (6.11 per query), 2,225,246 grid barriers
(about 13.0 per publication), 5.944 billion frontier entries, and 25.831
billion light-edge visits. Each worker grid was restricted to five blocks by
dividing 20 CUs among four workers even though occupancy allowed eight blocks
per CU. The user also observed severe slowdown and a multiworker crash from
that design. These results motivate, but do not measure, the current
multi-query coordinator: ready private workspaces now share one physical
cooperative launch, and the full admitted grid visits slots sequentially with
uniform barriers. This pass keeps `host-checked` as the unchanged scalar
correctness reference, retains explicit `fused-host-checked` and
`reduced-round-trip` arms, and claims no speedup for the HIP-unvalidated
redesign.

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
hipcc -std=c++17 -O3 -x hip --offload-arch=gfx1151 -DBF10_NO_MAIN \
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

## First pass: three-arm controller GPU timelines with rocprofv3

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
    --delta-controller fused-host-checked --parallel-net-workers 4 \
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-fused-host-checked-w4

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 --parallel-net-workers 4 \
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-reduced-b4-w4
```

Four workers are the current measured classic Delta-Stepping timing control.
Keep the count explicit when comparing traces or profiler-free timings; it is
workload-specific and is not a universal default.

The host-checked controller is the default, trusted scalar correctness
reference; it never silently selects the cooperative backend.
`fused-host-checked` is an explicit experimental cooperative controller with a
fixed one-action budget. `reduced-round-trip` is the explicit multi-action
controller, with batch size four as the initial candidate and a hard maximum
of 64. With multiple workers, both modes rendezvous ready queries into one
coordinator-owned launch; all blocks process slot 0, then slot 1, and so on,
which keeps every `grid.sync()` uniform and makes the maximum number of
concurrent cooperative launches exactly one. Width four is the initial query
batch. The physical grid defaults to one block per CU, not the legal occupancy
maximum; the runtime clamps it further by graph size and occupancy. After the
three-arm correctness gate, sweep `--delta-batch-blocks-per-cu` over `1`, `2`,
`4`, and `8` because more barrier participants can be slower despite higher
occupancy. Both modes capability-check before traversal and retain their
ordinary unsupported/callback fallback behavior.

For one successful all-light vector-target query, let `L` be light actions,
`B` processed buckets, and `K` the reduced action budget. The scalar reference
retains `2L + 4B + 5` blocking boundaries. A query produces `L` logical fused
publications or `ceil(L / K)` logical reduced publications, plus its bounded
setup/extraction boundaries. In a multiworker run those logical publications
are rendezvoused: if `P` slot publications are simultaneously batchable at
width `W`, the ideal physical-completion term is `ceil(P / W)` rather than
`P`; skew and tail batches can raise it. Therefore schema-4
`query_batching.physical_completions` is the authoritative dynamic count. Do
not add per-query formulas or equate logical publications with physical
cooperative launches when interpreting a four-worker trace. Applied only as
an ideal packing estimate to the archived 170,929 publications, width four
would approach 42,733 physical completions; if a four-action reduced arm also
cut logical publications by four, the corresponding ideal is about 10,684.
Query skew, early terminals, and tail batches can make either count higher;
neither number is a measured result.

The cooperative mechanics can still make fewer host waits slower: a Boolean
all-light bucket-closing action that compacted and continued crossed 12 phase
barriers, while the Boolean target-settled terminal branch crossed six; other
branches have different counts, and every launch adds entry/final publication
barriers. Every thread also issued redundant fences and contended atomic
scalar reads around phase changes, queue reservation used a CAS loop, pending
minimum used one global atomic per thread, and each query could span a
one-block-per-CU grid. The repair uses grid-barrier ordering with
one final system fence, stable post-barrier loads, one `atomicAdd` reservation,
a block minimum followed by one global atomic per block. The current physical
batch uses one admitted grid for every sequential query slot instead of
partitioning CUs among workers; at the default one block per CU, the cited
20-CU device can use up to 20 blocks for a sufficiently large graph rather
than five. That is a design expectation, not a measurement. Treat physical
completions, logical slot publications, completed actions, grid size, and grid
barriers as acceptance data; fewer publications alone is insufficient.

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
    --delta-controller fused-host-checked --parallel-net-workers 4 \
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
    --delta-telemetry" \
  PATHFINDER_PROFILE=none 2>&1 | tee delta-mixed-fused-telemetry.log

make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 17 \
    --delta-controller reduced-round-trip \
    --delta-controller-batch-size 4 --parallel-net-workers 4 \
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
    --delta-telemetry" \
  PATHFINDER_PROFILE=none 2>&1 | tee delta-mixed-reduced-telemetry.log

grep '^{"type":"delta_stepping_telemetry"' \
  delta-mixed-host-telemetry.log delta-mixed-fused-telemetry.log \
  delta-mixed-reduced-telemetry.log
```

After all workers join, `pathfinder` writes exactly one compact JSON line to
standard output with `type="delta_stepping_telemetry"`, `schema_version=4`,
and `scope="pathfinder_run"`. `queries` counts actual SSSP invocations, not net
slots; a net with no unresolved target leaves its slot uncollected.
`completed_queries` counts records whose cleanup and final synchronization
finished. `execution_paths` counts `exact_unit`, `compact_generic`,
`legacy_generic`, and `generic_distances_only`. The record also includes the
resolved numeric delta, runtime wavefront size, actual worker count,
auto-delta/multiplier values, force-mode flags, configured `controller_mode`
and `controller_batch_size`, per-query `effective_controller_modes` counts,
`controller_fallback_queries`, backend/fallback-reason histograms, and a
`controller_diagnostics` object. Schema 4 adds a `query_batching` object that
separates physical batch launches/completions from logical slot publications,
records slot/action utilization and relaunches, and exposes the selected grid
and the maximum number of simultaneous cooperative launches. The configured
names use `host_checked`,
`fused_host_checked`, and `reduced_round_trip` in JSON even though the CLI
uses hyphens. Counter fields are summed across queries; controller grid minima
and maxima summarize selected cooperative queries, and the three queue
high-water fields under `maxima` are maxima across queries, not sums.

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
| `controller_mode`, `controller_batch_size` | Run-level requested A/B configuration. These fields alone do not prove that a cooperative controller executed. |
| `effective_controller_modes` | Query counts by effective mode: `host_checked`, `fused_host_checked`, and `reduced_round_trip`. Host must remain host; a fused or reduced sample must put every generic query in its requested bucket. |
| `controller_backends` | Query counts for `not_run`, `scalar_host`, `cooperative_grid`, and `exact_unit`. With `--delta-force-generic`, a valid fused/reduced sample requires `cooperative_grid == queries`; host requires `scalar_host == queries`. |
| `controller_fallback_reasons`, `controller_fallback_queries` | Counts for `none`, callback, exact-unit bypass, generation budget, unsupported cooperative launch, capability/occupancy query failure, and no resident grid. Requested scalar host execution reports `none`, not a fallback. Reject an intended cooperative sample if any query falls back. |
| `query_batching.enabled`, `configured_width`, `effective_width` | Physical multi-query coordinator selection and its bounded width. Four-worker fused/reduced acceptance requires enabled batching and effective width four; host-checked requires disabled batching. |
| `query_batching.physical_launch_attempts`, `physical_launches`, `physical_completions`, `max_concurrent_cooperative_launches` | Physical coordinator boundaries. Successful acceptance requires launches and completions to match, no launch failure, and maximum concurrency exactly one. Attempts can exceed launches only on a reported failure. |
| `query_batching.logical_queries_admitted/completed`, `slot_dispatches`, `slot_relaunches`, `active_slots_sum`, `unused_slots` | Logical query-to-slot accounting. `slot_dispatches` is the sum of active slots across physical attempts; it can greatly exceed physical launches and equals logical per-query publications on a successful forced-generic run. |
| `query_batching.action_slots_budgeted`, `actions_completed`, `unused_action_slots` | Physical-batch view of bounded query actions. Completed plus unused action slots equals budgeted slots. |
| `query_batching.requested_blocks_per_compute_unit`, selected/occupancy blocks per CU, and `grid_blocks_min/max` | The requested cap, runtime occupancy ceiling, selected cap, and actual physical grid. Default request is one block per CU even if occupancy legally permits up to eight. |
| `controller_diagnostics.cooperative_launches`, `controller_diagnostics.controller_publications` | Per-query diagnostics retained for direct/single-query compatibility. In a coordinated multi-query run, `controller_publications` and `controller_round_trips` count logical slot publications, while physical launches live only under `query_batching`; never require publications to equal physical launches. |
| `controller_diagnostics.controller_nonterminal_publications`, `controller_diagnostics.controller_terminal_publications` | Progress versus terminal descriptors; their sum must equal total publications. |
| `controller_diagnostics.controller_action_slots_budgeted`, `controller_diagnostics.controller_actions_completed`, `controller_diagnostics.controller_unused_action_slots` | Bounded-work accounting. Completed plus unused slots must equal budgeted slots. Fused budgets one action per logical publication; reduced budgets at most 64 and should publish fewer times than completed actions on the intended workload. |
| `controller_diagnostics.cooperative_grid_blocks_min/max`, active blocks per CU, and compute units | Per-query view of the admitted physical grid. In a coordinated run it must agree with `query_batching.grid_blocks_min/max`; the grid is no longer divided by worker count. |
| `controller_diagnostics.cooperative_grid_barriers` | Device-wide controller phase barriers actually crossed. Compare it with actions and wall time to detect work amplification; it is not a host synchronization count. |
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

## gfx1151 three-arm controller acceptance

Run this section on the target AMD host from the repository root. It supersedes
the historical two-arm controller examples later in this file. The input names
match the existing `logicnets_jscl` procedure; substitute absolute paths
without changing controller flags if the artifacts live elsewhere.

Build the normal timing binary explicitly for gfx1151, then run one strict,
telemetry-free correctness pass for each arm:

```bash
set -euo pipefail
validation_root=amd-validation/fused-controller
mkdir -p "$validation_root/results" "$validation_root/work" \
  "$validation_root/logs" "$validation_root/profiles" "$validation_root/timing"

make -B ./PathFinderFile
make -B ./pathfinder \
  PATHFINDER_HIP_FLAGS='-std=c++17 -O3 -x hip --offload-arch=gfx1151'

common_args=(
  --logical-netlist logicnets_jscl.netlist
  --device-graph xcvu3p.full-poc-base-wire.devicegraph
  --sssp-engine delta-step --delta 1 --delta-force-generic
  --parallel-net-workers 4 --strict-routing
)

run_correctness_arm() {
  local arm=$1
  shift
  env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
    logicnets_jscl_unrouted.phys \
    "$validation_root/results/${arm}.phys" \
    "${common_args[@]}" \
    --work-dir "$validation_root/work/${arm}" \
    "$@" \
    2>&1 | tee "$validation_root/logs/${arm}.log"
}

run_correctness_arm host \
  --delta-controller host-checked
run_correctness_arm fused \
  --delta-controller fused-host-checked \
  --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1
run_correctness_arm reduced-b4 \
  --delta-controller reduced-round-trip --delta-controller-batch-size 4 \
  --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1

cmp "$validation_root/results/host.phys" \
  "$validation_root/results/fused.phys"
cmp "$validation_root/results/host.phys" \
  "$validation_root/results/reduced-b4.phys"
cmp "$validation_root/work/host/logicnets_jscl_PathFinderFile.routes.jsonl" \
  "$validation_root/work/fused/logicnets_jscl_PathFinderFile.routes.jsonl"
cmp "$validation_root/work/host/logicnets_jscl_PathFinderFile.routes.jsonl" \
  "$validation_root/work/reduced-b4/logicnets_jscl_PathFinderFile.routes.jsonl"
```

Exact `cmp` equality is the strongest gate for this deterministic benchmark.
Also retain the strict router/checker result and the route-depth check described
below. If another workload permits multiple equally valid route encodings, use
the CPU-reference distance/path checker instead of weakening semantic
validation to a file-size or aggregate-cost comparison.

Collect schema-4 telemetry in separate runs. These are diagnostics, never
timing samples:

```bash
run_telemetry_arm() {
  local arm=$1
  shift
  env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
    logicnets_jscl_unrouted.phys \
    "$validation_root/results/${arm}-telemetry.phys" \
    "${common_args[@]}" --delta-telemetry \
    --work-dir "$validation_root/work/${arm}-telemetry" \
    "$@" \
    2>&1 | tee "$validation_root/logs/${arm}-telemetry.log"
  grep '^{"type":"delta_stepping_telemetry"' \
    "$validation_root/logs/${arm}-telemetry.log" \
    > "$validation_root/results/${arm}-telemetry.json"
}

run_telemetry_arm host \
  --delta-controller host-checked
run_telemetry_arm fused \
  --delta-controller fused-host-checked \
  --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1
run_telemetry_arm reduced-b4 \
  --delta-controller reduced-round-trip --delta-controller-batch-size 4 \
  --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1

python3 - <<'PY'
import json
import pathlib

root = pathlib.Path("amd-validation/fused-controller/results")
arms = {
    "host": ("host_checked", "scalar_host"),
    "fused": ("fused_host_checked", "cooperative_grid"),
    "reduced-b4": ("reduced_round_trip", "cooperative_grid"),
}

for arm, (mode, backend) in arms.items():
    record = json.loads((root / f"{arm}-telemetry.json").read_text())
    queries = record["queries"]
    assert record["schema_version"] == 4
    assert record["completed_queries"] == queries
    assert record["force_generic"] is True
    assert record["execution_paths"]["exact_unit"] == 0
    assert record["effective_controller_modes"][mode] == queries
    assert record["controller_backends"][backend] == queries
    assert record["controller_fallback_queries"] == 0
    assert record["controller_fallback_reasons"]["none"] == queries

    diagnostics = record["controller_diagnostics"]
    batching = record["query_batching"]
    if backend == "scalar_host":
        assert batching["enabled"] is False
        assert batching["physical_launches"] == 0
        assert diagnostics["cooperative_launches"] == 0
        assert diagnostics["controller_publications"] == 0
        continue

    publications = diagnostics["controller_publications"]
    physical_launches = batching["physical_launches"]
    actions = diagnostics["controller_actions_completed"]
    slots = diagnostics["controller_action_slots_budgeted"]
    unused = diagnostics["controller_unused_action_slots"]
    assert batching["enabled"] is True
    assert batching["configured_width"] == 4
    assert batching["effective_width"] == 4
    assert batching["requested_blocks_per_compute_unit"] == 1
    assert batching["max_concurrent_cooperative_launches"] == 1
    assert batching["physical_launch_attempts"] == physical_launches
    assert physical_launches == batching["physical_completions"]
    assert batching["cancellations"] == 0
    assert batching["launch_failures"] == 0
    assert batching["descriptor_failures"] == 0
    assert batching["logical_queries_admitted"] == queries
    assert batching["logical_queries_completed"] == queries
    assert batching["slot_dispatches"] == publications
    assert batching["slot_relaunches"] == publications - queries
    assert physical_launches <= publications
    assert publications == record["counters"]["controller_round_trips"]
    assert diagnostics["controller_nonterminal_publications"] + \
        diagnostics["controller_terminal_publications"] == publications
    assert actions + unused == slots
    assert batching["actions_completed"] == actions
    assert batching["action_slots_budgeted"] == slots
    assert batching["unused_action_slots"] == unused
    assert batching["active_slots_sum"] == publications
    assert batching["active_slots_sum"] + batching["unused_slots"] == \
        batching["physical_launch_attempts"] * batching["effective_width"]
    assert batching["grid_blocks_min"] > 0
    assert batching["grid_blocks_max"] >= batching["grid_blocks_min"]
    assert diagnostics["cooperative_grid_blocks_min"] > 0
    assert diagnostics["cooperative_grid_blocks_max"] >= \
        diagnostics["cooperative_grid_blocks_min"]
    assert diagnostics["cooperative_grid_barriers"] >= 2 * publications
    if arm == "fused":
        assert slots == publications
    else:
        assert slots <= 64 * publications
        assert publications < actions, "batch-four did not amortize publications"

    print(
        arm,
        f"queries={queries}",
        f"logical_publications={publications}",
        f"physical_completions={batching['physical_completions']}",
        f"actions={actions}",
        f"grid_barriers={diagnostics['cooperative_grid_barriers']}",
    )
PY
```

Then collect five uninstrumented samples per arm. The alternating order limits
clock/thermal drift; the warm-up is deliberately not timed:

```bash
for arm in host fused reduced-b4; do
  case "$arm" in
    host) controller_args=(--delta-controller host-checked) ;;
    fused) controller_args=(--delta-controller fused-host-checked \
      --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1) ;;
    reduced-b4) controller_args=(--delta-controller reduced-round-trip \
      --delta-controller-batch-size 4 --delta-query-batch-width 4 \
      --delta-batch-blocks-per-cu 1) ;;
  esac
  env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
    logicnets_jscl_unrouted.phys \
    "$validation_root/timing/${arm}-warmup.phys" \
    "${common_args[@]}" "${controller_args[@]}" \
    --work-dir "$validation_root/work/${arm}-warmup"
done

for sample in 1 2 3 4 5; do
  if (( sample % 2 )); then
    arms=(host fused reduced-b4)
  else
    arms=(reduced-b4 fused host)
  fi
  for arm in "${arms[@]}"; do
    case "$arm" in
      host) controller_args=(--delta-controller host-checked) ;;
      fused) controller_args=(--delta-controller fused-host-checked \
        --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1) ;;
      reduced-b4) controller_args=(--delta-controller reduced-round-trip \
        --delta-controller-batch-size 4 --delta-query-batch-width 4 \
        --delta-batch-blocks-per-cu 1) ;;
    esac
    /usr/bin/time -f 'wall_seconds=%e\npeak_rss_kib=%M' \
      -o "$validation_root/logs/${arm}-time-${sample}.txt" \
      env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
        logicnets_jscl_unrouted.phys \
        "$validation_root/timing/${arm}-${sample}.phys" \
        "${common_args[@]}" "${controller_args[@]}" \
        --work-dir "$validation_root/work/${arm}-${sample}"
  done
done
```

For a worker/concurrency regression, rerun the same telemetry-free timing loop
with `--parallel-net-workers` set to 1, 2, and 4 for every controller. The
following exact sweep records one warm-up and three samples per cell:

```bash
for workers in 1 2 4; do
  for arm in host fused reduced-b4; do
    case "$arm" in
      host) controller_args=(--delta-controller host-checked) ;;
      fused) controller_args=(--delta-controller fused-host-checked \
        --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1) ;;
      reduced-b4) controller_args=(--delta-controller reduced-round-trip \
        --delta-controller-batch-size 4 --delta-query-batch-width 4 \
        --delta-batch-blocks-per-cu 1) ;;
    esac
    for sample in warmup 1 2 3; do
      command=(env PATHFINDER_PROFILE_COMMAND= ./PathFinderFile \
        logicnets_jscl_unrouted.phys \
        "$validation_root/timing/${arm}-w${workers}-${sample}.phys" \
        --logical-netlist logicnets_jscl.netlist \
        --device-graph xcvu3p.full-poc-base-wire.devicegraph \
        --sssp-engine delta-step --delta 1 --delta-force-generic \
        --parallel-net-workers "$workers" --strict-routing \
        "${controller_args[@]}" \
        --work-dir "$validation_root/work/${arm}-w${workers}-${sample}")
      if [[ "$sample" == warmup ]]; then
        "${command[@]}"
      else
        /usr/bin/time -f 'wall_seconds=%e\npeak_rss_kib=%M' \
          -o "$validation_root/logs/${arm}-w${workers}-time-${sample}.txt" \
          "${command[@]}"
      fi
    done
  done
done
```

Finally rebuild with the ROCTx command above and capture one runtime trace per
arm with telemetry disabled. `PathFinderFile` applies the profile command only
to the GPU `pathfinder` child:

```bash
for arm in host fused reduced-b4; do
  case "$arm" in
    host) controller_args=(--delta-controller host-checked) ;;
    fused) controller_args=(--delta-controller fused-host-checked \
      --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1) ;;
    reduced-b4) controller_args=(--delta-controller reduced-round-trip \
      --delta-controller-batch-size 4 --delta-query-batch-width 4 \
      --delta-batch-blocks-per-cu 1) ;;
  esac
  env PATHFINDER_PROFILE_COMMAND="rocprofv3 --runtime-trace --stats \
    --output-directory $validation_root/profiles/${arm} --" \
    ./PathFinderFile \
      logicnets_jscl_unrouted.phys \
      "$validation_root/results/${arm}-rocprof.phys" \
      "${common_args[@]}" "${controller_args[@]}" \
      --work-dir "$validation_root/work/${arm}-rocprof" \
      2>&1 | tee "$validation_root/logs/${arm}-rocprof.log"
  cmp "$validation_root/results/host.phys" \
    "$validation_root/results/${arm}-rocprof.phys"
done
```

For each trace report dispatches, `hipStreamSynchronize` calls, descriptor D2H
copies, kernel-union idle time, concurrent active worker streams, and the
`delta_step.scalar_host_controller` versus
`delta_step.cooperative_controller` ROCTx ranges. The fused arm must materially
reduce host boundaries before it is considered a controller win; the reduced
arm must reduce publications beyond fused without increasing controller
barriers/device work enough to lose profiler-free wall time.

## Additional low-level AMD validation checklist

Nothing in this section was run on the host that implemented the bounded
controller pass: it has no HIP compiler, ROCm runtime, or AMD GPU. Run these
commands from the repository root on a gfx1151 AMD system and retain every
log. Keep both cooperative controllers and generation-tagged membership opt-in
until their complete matrix and repeated explicit-stream stress pass. Use the
three-arm procedure above for current PathFinder acceptance; this section adds
the lower-level storage, ownership, and reuse matrix.

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

hipcc -std=c++17 -O2 -pthread -x hip --offload-arch=gfx1151 \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/tests/unit_bfs_hip_test.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -o amd-validation/bin/unit_bfs_hip_test

hipcc -std=c++17 -O2 -pthread -x hip --offload-arch=gfx1151 \
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
must run the Cartesian controller matrix: host-checked, fused-host-checked,
and reduced-round-trip (batch sizes 1 and 4), automatic-compact and forced-wide
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

`DELTA_REQUIRE_REDUCED_CONTROLLER=1` makes every instrumented fused/reduced
matrix or stress probe fail on a capability fallback; callback probes still
require their intentional host fallback. The Delta stress must keep at least
four nonblocking streams active, reuse each
workspace for thousands of queries, force generation rollover repeatedly, and
exercise a callback exception/abort followed by a successful query on the
same workspace. Confirm exact distance/path equivalence and outgoing-row edge
ownership after every run. Confirm telemetry reports the requested cooperative
mode and backend (not fallback) for every intended fused/reduced query, while
callback queries report a host-checked fallback. A passing low-level suite is required before
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

The retained route-depth example below shows the older host/reduced workflow;
use the three-arm procedure above for current controller acceptance. It uses
forced-generic classic Delta-Stepping, `delta=1`, four workers, automatic
compact row offsets, and Boolean membership. The wider storage, membership,
parent, and distances-only matrix remains in the low-level suite. Run the host
reference first:

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
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
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
    --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
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

The historical commands below collect profiler-free end-to-end samples for
the host/reduced pair. The three-arm alternating-order procedure above
supersedes them for this pass. They use GNU `time`, keep telemetry and profilers
disabled, perform one warm-up per arm, and then record five fresh conversion,
routing, and reconstruction runs. Keep all inputs except the controller flags
identical:

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
  --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
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
      --delta-query-batch-width 4 --delta-batch-blocks-per-cu 1 \
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
