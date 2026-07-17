# GPU profiling the PathFinder benchmark

The benchmark wrapper performs three separate processes:

```text
interchange_to_csr -> pathfinder (HIP) -> routes_to_phys
```

Profiling belongs around the inner `pathfinder` process. This keeps conversion
and route serialization out of GPU traces, while the Makefile's existing
`time` invocation still measures the complete pipeline. The Makefile passes a
profiler prefix to `PathFinderFile` through `PATHFINDER_PROFILE_COMMAND`; the
wrapper applies it only when launching `pathfinder`.

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
hipcc -std=c++17 -O3 -x hip -DBF8_NO_MAIN \
  -DPATHFINDER_ENABLE_ROCTX \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf8.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -pthread -lrocprofiler-sdk-roctx -o pathfinder
```

ROCm distributions package the header and library as
`rocprofiler-sdk-roctx`. Rebuild without the macro/library for final timing if
even the small marker overhead matters.

## First pass: GPU timeline with rocprofv3

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-baseline
```

Important: the current FPGA Interchange converter writes every CSR edge weight
as `1.0f`. This command therefore profiles the exact-unit delta specialization,
not a representative weighted graph. A finite `--max-sssp-iters` forces the
generic implementation for control-flow testing, but its graph is still
unit-weight/all-light:

```bash
make ... PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--max-sssp-iters 2147483647" \
  PATHFINDER_PROFILE=rocprofv3
```

For a meaningful general-weight profile, generate a real-congestion snapshot
with the workflow below, then run the comparison executable on that frozen
CSR. The converter topology and metadata remain unchanged; only the saved CSR
edge values reflect PathFinder's accumulated present and historical costs.

## Real-congestion snapshot workflow

The snapshot generator runs the Interchange converter once per benchmark, then
runs weighted delta-stepping PathFinder iterations and saves normal,
independently usable `RIPSCSR1` files. For example:

```bash
python3 CongestionFreeRouting/pathfinder_snapshot_benchmark.py \
  --benchmarks-dir /path/to/benchmarks \
  --benchmarks logicnets_jscl,boom_med_pb \
  --device /path/to/xcvu3p.device \
  --out-dir results/pathfinder-snapshots \
  --snapshot-iters 1,3,5,8 \
  --delta 4 --capacity 1 \
  --present-factor 1 --present-multiplier 2 --history-factor 1 \
  --route-batch-size 256
```

For each benchmark this creates:

```text
results/pathfinder-snapshots/<benchmark>/
  base.csrbin
  base.csrbin.ifmeta.bin
  iteration-1.csrbin
  iteration-1.csrbin.ifmeta.bin
  iteration-1.snapshot.json
  ...
```

`iteration-N.csrbin` is a static outgoing CSR materialized from the completed
iteration `N`'s global occupancy/history state, with the present factor
advanced after a nonconverged iteration as PathFinder normally does, and
congestion charged to each edge's destination node. The manifest
records this as `"cost_phase": "post_iteration_global_cost"`. It is a stable
all-net workload for comparison, rather than a literal CSR for a later
PathFinder batch (which would first rip up selected congested nets).
If PathFinder converges before a requested checkpoint, it does not reroute
already legal nets just to manufacture more work; it emits each later requested
snapshot from that unchanged converged congestion state. The generator still
checks that every requested artifact was written. It also refuses existing
generated snapshot artifacts by default; pass `--overwrite` to replace that
benchmark's prior snapshot files.

## Compare SSSP implementations on one frozen workload

Build `pathfinder_sssp_compare` with the same sources as `pathfinder`, adding
`CongestionFreeRouting/pathfinder_sssp_compare.cpp` and defining `BF8_NO_MAIN`
when `bf8.cpp` is linked. For example:

```bash
hipcc -std=c++17 -O3 -x hip -DBF8_NO_MAIN -DROUTING_PATHFINDER_NO_MAIN \
  -DPATHFINDER_ENABLE_ROCTX \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs \
  CongestionFreeRouting/pathfinder_sssp_compare.cpp \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf8.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp \
  -pthread -lrocprofiler-sdk-roctx -o pathfinder_sssp_compare
```

Then each engine routes every net once without modifying the frozen CSR:

```bash
./pathfinder_sssp_compare \
  results/pathfinder-snapshots/logicnets_jscl/iteration-3.csrbin \
  --engines delta-step,bellman-ford \
  --warmups 1 --repeats 5 --delta 4 --parallel-net-workers 1 \
  --verify --baseline delta-step \
  --json-out results/comparisons/logicnets_jscl.iteration-3.json
```

The JSON reports full-pass wall-time statistics, reached sinks, route-edge
count, a distance checksum, speedup against the selected baseline, and
verification mismatches. It defaults to one cross-net worker for every engine
so the timing compares the SSSP implementations rather than their distinct
auto-scheduling policies. It intentionally rejects unit BFS on weighted
snapshots: BFS ignores CSR values and is not a correct weighted SSSP baseline.
Pass `--allow-inexact-unit-bfs` only when a topology-only comparison is wanted.
For exact delta-step versus Bellman-Ford verification, leave
`--max-sssp-iters` at its default unlimited value; a finite cap bounds
delta buckets and Bellman-Ford rounds differently.

Profile only the comparison phase by wrapping that executable directly:

```bash
rocprofv3 --runtime-trace --stats --output-directory traces/logicnets.iter3 -- \
  ./pathfinder_sssp_compare \
    results/pathfinder-snapshots/logicnets_jscl/iteration-3.csrbin \
    --engines delta-step,bellman-ford --warmups 0 --repeats 1 --verify
```

The trace contains outer `pathfinder.compare.<engine>` ROCTx ranges and the
existing per-net/per-SSSP ranges, so conversion and PathFinder warm-up are not
included in the comparison trace.

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
over several fresh runs. Keep GPU clocks/power mode, worker count, delta,
benchmark input, and thermal state fixed when comparing versions.

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

Hardware profiles still cannot report algorithmic efficiency such as bucket
occupancy, relax attempts per successful update, or pending vertices scanned
per retained vertex. Those should be added as optional device counters after
the first timeline identifies which phases matter; enabling them beforehand
would add atomics and risk distorting the baseline.
