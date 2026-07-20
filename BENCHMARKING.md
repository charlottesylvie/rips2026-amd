# PathFinder GPU Benchmarking

## Benchmark configuration

The archived profile discussed here was produced for `logicnets_jscl` before
the explicit generic-dispatch control existed and used the former forcing
workaround. For every new or reproduction run, use
`--delta-force-generic`:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic" \
  PATHFINDER_PROFILE=rocprofv3 \
  PATHFINDER_PROFILE_RUN=delta-generic
```

Before invoking the benchmark Make target, generate the persistent artifact
manually:

```bash
./device_to_routing_graph xcvu3p.device xcvu3p.full-poc-base-wire.devicegraph --full-device
```

Make assumes this artifact already exists and validates it as a prerequisite;
it does not run the preprocessor. DeviceResources parsing and static graph
formatting are therefore not part of the per-test result. Design-specific CSR
conversion, routing, and route reconstruction remain inside that timing.

The available full-system profile routed all 28,026 nets; it was not the later
100-net counter experiment. It took 144.435 seconds under `rocprof-sys`.
Profiler wall time must not be compared directly with an unprofiled benchmark
because tracing substantially perturbs this launch- and synchronization-heavy
workload.

## Reproducible weighted and telemetry runs

The current router can replace loaded CSR weights deterministically in memory.
For an exact-unit-versus-generic differential run, keep every edge at one and
select the implementation explicitly:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 1 --delta-force-generic \
    --delta-benchmark-weights unit" \
  PATHFINDER_PROFILE=none
```

For genuinely mixed effective weights with a reproducible seed:

```bash
make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
  PATHFINDER_SSSP_ENGINE=delta-step \
  PATHFINDER_ARGS="--delta 2 --delta-force-generic \
    --delta-benchmark-weights mixed --delta-benchmark-weight-seed 123" \
  PATHFINDER_PROFILE=none
```

The other families are `all-light = delta/4` and `all-heavy = 4*delta`;
`mixed` chooses per original CSR edge index from
`{0, delta/4, delta, 4*delta}`. A family requires an explicitly supplied
numeric delta, and the seed is accepted only for `mixed`. The transformation
does not rewrite the `.csrbin`.

Add `--delta-telemetry` only to a separate diagnostic run. It selects
instrumented kernels and emits one aggregate JSON line after every worker has
joined, not one record per worker or net. Counter values are sums over actual
SSSP queries, while queue high-water values are maxima. Keep telemetry disabled
for wall-time and PMC baselines. Precise counter definitions are in
[`CongestionFreeRouting/GPU_PROFILING.md`](CongestionFreeRouting/GPU_PROFILING.md#opt-in-algorithm-telemetry).

These new force, weighted-family, telemetry, and distances-only paths have not
yet been rerun in the production HIP regression suite or benchmarked on an AMD
GPU in this checkout. The commands here define the outstanding validation;
they are not new measurements.

## Worker-count follow-up

The following values are means of five unprofiled runs:

| Parallel net workers | Mean runtime (s) | Change from one worker |
| ---: | ---: | ---: |
| 1 | 68.771 | baseline |
| 2 | 66.281 | -3.62% |
| 4 | 66.471 | -3.34% |
| 8 | 66.304 | -3.59% |

The 2-, 4-, and 8-worker results are effectively flat. Without the individual
samples or confidence intervals, the roughly 0.2-second spread among them is
most plausibly ordinary run-to-run noise. Extra host workers are therefore not
a promising primary optimization for this workload. The small one-to-two
worker difference can be rechecked with interleaved runs if a precise estimate
is needed, but it does not alter the optimization order below.

## Full-system profile findings

### Host-side runtime allocation

| HIP API category | Time (s) | Share of 144.435 s | Calls | Calls/net |
| --- | ---: | ---: | ---: | ---: |
| `hipStreamSynchronize` | 89.699 | 62.1% | 312,496 | 11.15 |
| `hipMemcpyAsync` | 33.407 | 23.1% | 594,072 | 21.20 |
| Launch, memset, and error checks | 7.020 | 4.9% | — | — |
| Everything else | 14.309 | 9.9% | — | — |

The dominant end-to-end cost is fine-grained GPU orchestration: many kernel
batches, status transfers, and host waits per net. The retained device trace
attributes only about 1.2% of GPU execution time to copy/fill helper kernels.
Consequently, the large `hipMemcpyAsync` API share is not evidence of sustained
bulk graph transfer. It is principally the accumulated overhead and ordering
cost of many small operations, possibly including time that the runtime waits
for prior stream work.

### GPU execution

During the steady portion of the run, GPU busy time was about 96.3% and power
was about 82.6 W. This rules out long periods in which CPU route preparation
leaves the GPU idle. It does not mean the GPU kernels are compute-bound: a
kernel stalled on memory dependencies still counts as GPU-busy.

The retained kernel trace divided GPU execution approximately as follows:

| Kernel phase | GPU execution share |
| --- | ---: |
| Generic delta relaxation | 43.626% |
| Sparse touched-state reset | 28.625% |
| Predecessor materialization/path support | 23.897% |
| Other kernels and device helpers | 3.852% |

Reset and predecessor materialization together consume about 52.5% of traced
GPU execution, more than relaxation itself. Avoiding, fusing, or amortizing
that state-management work is therefore at least as important as tuning the
relaxation kernel.

This archive measured the legacy predecessor/materialization path. Current
automatic generic vector-target runs use a compact winning edge ID when the
shared graph has its edge-to-source map. Use
`--delta-force-legacy-parent` only to reproduce the old parent-policy side of
an A/B comparison. The percentages above are historical evidence, not the cost
of the current compact path; reprofile before transferring them to current
code.

### Memory and CPU/GPU communication

The input CSR contains 28,226,432 nodes and 125,423,075 edges and is about
1,172.25 MiB on disk. Observed GPU memory then rises to a nearly constant
3,276.7 MB (decimal). That shape is consistent with one graph upload plus
long-lived graph/workspace arrays. It is not evidence that all available GPU
memory has been consumed, a leak is occurring, or data is being paged for each
SSSP.

The target Radeon 8060S/Strix Halo system uses shared/coherent system memory,
so this is not the classic discrete-GPU model in which every CPU/GPU exchange
crosses a PCIe link to separate VRAM. Explicit HIP copies and synchronization
still incur software, cache-coherence, and ordering costs. The current evidence
shows that their *frequency* is important, not that bulk-link bandwidth is the
main limit.

This full-system trace does not contain usable hardware performance counters.
A subsequent 100-net System SOL run does, and its results are reported below.
The distinction matters: GPU utilization and allocated-memory telemetry alone
cannot classify the hot kernels, while the later counter data can rule out
arithmetic-throughput and low-residency limits.

### Low-level distances-only memory baseline

PathFinder needs reconstructed routes, so it cannot exercise the new
`DeltaSteppingCsrWorkspace::run_distances` specialization. A distance-only
comparison requires a dedicated low-level harness using identical sources,
numeric delta, edge weights, and destination costs. Constructing the shared
graph with `DeltaSteppingCsrStorageMode::kDistancesOnly` also removes the
compact-parent edge-to-source map; separate graph-upload measurements from
traversal time.

The code-layout estimate is `40 B/V` of generic mutable state. For this
28,226,432-vertex, 125,423,075-edge input, that saves about `215.35 MiB` per
workspace versus compact generic's extra `8 B/V`, or `538.38 MiB` versus
legacy generic's extra `20 B/V`. Strict distances-only graph storage saves an
additional `478.45 MiB` (`4 B/E`) once per shared graph, for approximately
`693.80 MiB` versus one compact workspace plus map or `1,016.83 MiB` versus
one legacy workspace plus map. These figures exclude target/path capacity,
small scalars, sources, optional vertex costs, and allocator overhead; they are
layout calculations that assume the eligible edge map is allocated, not
measured allocations. Real AMD HIP validation and measurement remain
outstanding.

### Trace limitations

The detailed trace buffer overflowed around 112 seconds into the 144-second
run and retained roughly 76–77% of events. There were also Perfetto patch
failures. Kernel percentages above should therefore be treated as representative
steady-state shares, not exact complete-run accounting. The aggregate HIP API
profile is complete enough to support the orchestration conclusion. PAPI CPU
counters were disabled, so this run also does not provide direct host DRAM or
CPU-cache metrics.

## Provisional optimization order

1. Use dedicated Unit-BFS for this production input rather than forcing
   generic Delta-Stepping. The full-device CSR has 28,226,432 rows, above the
   Delta exact-unit specialization's `2^24`-row guard. The generic path is
   therefore automatic on this graph, but the force flag records experimental
   intent and remains correct on smaller inputs.
2. Reprofile the landed compact winning-edge parent mode on AMD hardware. It
   removes the legacy predecessor row-recovery scan, but the historical 24.1%
   materialization result cannot quantify the new path without measurement.
3. Reduce sparse-reset writes, specializing state cleanup for the compiled
   mode before evaluating generation-stamped state. Reset is 28.9% of kernel
   time.
4. Reduce the roughly 11 synchronizations and 21 asynchronous-copy API calls
   per net. Device-resident batching, a persistent controller, or carefully
   captured HIP graphs are candidates; the objective is fewer host round trips,
   not merely faster bulk copies.
5. Use 32-bit internal row/edge offsets when the graph has fewer than
   `INT32_MAX` edges; the low GL1/GL2 hit rates make narrower state and row
   traffic more valuable than block-size tuning.
6. Treat additional parallel net workers as low priority for this workload;
   2, 4, and 8 workers produced no meaningful scaling.

## Hardware-counter follow-up status

### Completed System SOL profile

The Radeon 8060S/Strix Halo (`gfx1151`, 40 CUs, wave32) was profiled with
`rocprof-compute` System SOL (`-b 2 --no-roof --no-native-tool`) using one
worker and the first 100 `logicnets_jscl` nets. The archived invocation
predates `--delta-force-generic` and selected the generic all-edges-light path
with the former workaround; current reproductions must use the force flag. This
is not an exact-unit or genuinely mixed-weight profile. Moreover, this CSR has
28,226,432 rows, so its Delta exact-unit specialization cannot activate even
with automatic execution: the implementation guards that path at
`2^24 = 16,777,216` rows to preserve exact float depth values.

Four profiler replay passes were stable: aggregate kernel time ranged from
604.4 to 608.7 ms, with a 0.39% coefficient of variation. Their mean phase
allocation was:

| Phase | Kernel time (ms) | Share |
| --- | ---: | ---: |
| Light-edge relaxation | 259.6 | 42.8% |
| Sparse touched-state reset | 175.1 | 28.9% |
| Predecessor materialization | 145.9 | 24.1% |
| Other | 25.9 | 4.3% |

Reset and materialization together account for 52.9% of kernel time. Complete
removal would have a 2.12x kernel-only Amdahl ceiling, not a 2.12x end-to-end
prediction.

| Hot kernel | VALU proxy vs FP32 peak | GL1 hit | L2 hit | Resident waves/WGP | Wait-any cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relax light edges | 1.12% | 1.83% | 21.41% | 54.25 / 64 | 99.42% |
| Reset touched vertices | 0.09% | ~0.00% | 16.45% | 44.31 / 64 | 53.78% |
| Materialize predecessors | 0.39% | 4.04% | 19.74% | 52.26 / 64 | 99.80% |

The VALU value applies ROCm Compute Profiler's gfx115x System SOL formula to
the observed `SQ_INSTS_VALU_sum`, using the 14.848-TFLOP/s FP32 single-issue
FMA ceiling for 40 CUs at 2.9 GHz. The numerator includes integer and logic
VALU instructions, so it is a throughput proxy, not a literal floating-point
operation rate. It nevertheless rules out arithmetic saturation by a wide
margin. High residency, low cache hit rates, and high wait fractions instead
implicate irregular memory dependencies and scattered state traffic.
Across all recorded kernels, the time-weighted proxy is 0.63% of that peak.

The separate `compute_thruput_util` run is incomplete: the tool skipped metrics
11.2.5 and 11.2.6, and its CSV contains no VALU instruction counter. It cannot
support claims about VALU, VMEM, SALU, or branch utilization. System SOL is the
usable source for the table above.

The exposed GL2 request sizes produce the following interface-request rates:

| Hot kernel | GL2-to-fabric reads (GB/s) | GL2-to-fabric writes (GB/s) |
| --- | ---: | ---: |
| Relax light edges | 574.19 | 5.52 |
| Reset touched vertices | 1.98 | 9.47 |
| Materialize predecessors | 537.95 | 2.23 |

These estimate traffic at the cache/fabric interface, not physical DRAM
bandwidth. Replayed/coherent requests can exceed eventual DRAM transactions,
`max_mclk` is zero in the captured system data, and no MALL counter is exposed
on this gfx1151 catalog. Consequently, the archive cannot support a percentage
of physical peak memory bandwidth or a MALL hit rate.

The ten largest searches account for 72.4% of reset time and 69.0% of
materialization time. Launch size and duration correlate at 0.995 and 0.997,
respectively. The new aggregate telemetry supplies reached-vertex sums,
successful relaxations, frontier/queue work and queue maxima, light rounds,
CAS retries, and controller reads. Its CLI JSON intentionally has no net
identity or per-search distribution, so correlating the worst searches still
requires a dedicated per-invocation harness or trace annotation.

### Remaining targeted counter experiment

The corrected catalogs contain the device-specific counter set for gfx1151.
This consumer GPU exposes substantially fewer useful counters than an Instinct
GPU. In particular:

- L2 (`GL2C`) hits, misses, read sizes, write sizes, and write stalls are
  available;
- VALU, SALU, scalar-memory, wave, occupancy, and texture/memory instruction
  counters are available;
- texture-address-unit busy time is available as `MemUnitBusy`; but
- no MALL counter, eligible-wave counter, general VMEM dependency-stall
  counter, or direct DRAM-latency counter is exposed in this catalog.

The System SOL run is sufficient to classify the main limit and reprioritize
the roadmap. The following five small raw-counter passes remain useful for
cross-checking derived occupancy, memory-address-pipeline activity, and GL2
request volumes without relying on unavailable Instinct counters:

| Pass | Counters | Purpose |
| ---: | --- | --- |
| 1 | `VALUInsts SALUInsts SFetchInsts Wavefronts` | Compute/scalar instruction mix per wave and total launched waves |
| 2 | `MeanOccupancyPerActiveCU OccupancyPercent GPUBusy MemUnitBusy` | Occupancy and whether active dispatches stress the memory-address path |
| 3 | `L2CacheHit` | L2 hit percentage from `GL2C_HIT` and `GL2C_MISS` |
| 4 | `FETCH_SIZE` | Bytes fetched beyond L2, derived from 32/64/96/128-byte read requests |
| 5 | `WRITE_SIZE WriteUnitStalled` | Bytes written beyond L2 and GL2C write-stall severity |

These are the smallest useful groups that preserve each derived metric's
underlying raw counters. `FETCH_SIZE`, for example, already expands to four
hardware read-request counters, so combining every memory metric into one pass
is likely to exceed GL2C counter slots. `VALUInsts` is an instruction-density
proxy, not a true percentage-of-peak compute-utilization counter.

### Compatibility check

The catalog establishes availability but does not report whether a chosen set
fits the hardware counter slots simultaneously. Run the installed tool's
authoritative compatibility check before profiling:

```bash
cd /opt/workspace

{
  rocprofv3-avail -d 0 pmc-check VALUInsts SALUInsts SFetchInsts Wavefronts
  rocprofv3-avail -d 0 pmc-check MeanOccupancyPerActiveCU OccupancyPercent GPUBusy MemUnitBusy
  rocprofv3-avail -d 0 pmc-check L2CacheHit
  rocprofv3-avail -d 0 pmc-check FETCH_SIZE
  rocprofv3-avail -d 0 pmc-check WRITE_SIZE WriteUnitStalled
} 2>&1 | tee gfx1151-pmc-check.txt
```

All five groups were reported as accepted on the target gfx1151 GPU on
2026-07-16. No additional splitting is required.

### Generate the 100-net run inputs

The profiler consumes the router's binary CSR, not the original FPGA
Interchange files. First preprocess the raw device once, then generate the CSR
and metadata sidecar from the reusable device graph and the test case's two
netlists. Full-device preprocessing is explicit here; no nxroute or custom
bounds are applied:

```bash
cd /opt/workspace

test -x ./device_to_routing_graph
test -x ./interchange_to_csr
test -f logicnets_jscl_unrouted.phys
test -f logicnets_jscl.netlist
test -f xcvu3p.device

mkdir -p direct-profile-work

./device_to_routing_graph \
  xcvu3p.device \
  xcvu3p.full-poc-base-wire.devicegraph \
  --full-device \
  2>&1 | tee direct-profile-work/device-to-routing-graph.log

./interchange_to_csr \
  xcvu3p.full-poc-base-wire.devicegraph \
  logicnets_jscl_unrouted.phys \
  logicnets_jscl.netlist \
  direct-profile-work/logicnets_jscl.csrbin \
  --metadata direct-profile-work/logicnets_jscl.csrbin.ifmeta.bin \
  2>&1 | tee direct-profile-work/interchange-to-csr.log

ls -lh \
  direct-profile-work/logicnets_jscl.csrbin \
  direct-profile-work/logicnets_jscl.csrbin.ifmeta.bin
```

Keep `xcvu3p.full-poc-base-wire.devicegraph` for later test cases that use the
same device and preprocessing policy; do not rerun the preprocessing command
for each benchmark. If a `test -f` command fails, locate the corresponding
source asset with
`find /opt/workspace -type f -name '<filename>' -print` and substitute its
absolute path. These outputs cannot be reconstructed from the `pathfinder`
executable alone.

If all five sets are accepted, collect them in five repeatable application
passes using a text input file. This direct command assumes the CSR and metadata
files were just generated in `direct-profile-work`:

```bash
cd /opt/workspace

printf '%s\n' \
  'pmc: VALUInsts SALUInsts SFetchInsts Wavefronts' \
  'pmc: MeanOccupancyPerActiveCU OccupancyPercent GPUBusy MemUnitBusy' \
  'pmc: L2CacheHit' \
  'pmc: FETCH_SIZE' \
  'pmc: WRITE_SIZE WriteUnitStalled' \
  > gfx1151-pathfinder-pmcs.txt

mkdir -p pathfinder-profiles/delta-pmc-minimal

rocprofv3 \
  --input gfx1151-pathfinder-pmcs.txt \
  --output-directory "$PWD/pathfinder-profiles/delta-pmc-minimal" \
  --output-format csv \
  -- \
  ./pathfinder \
    direct-profile-work/logicnets_jscl.csrbin \
    direct-profile-work/logicnets_jscl.csrbin.ifmeta.bin \
    --routes-out direct-profile-work/routes-pmc.jsonl \
    --allow-unrouted \
    --sssp-engine delta-step \
    --delta-force-generic \
    --parallel-net-workers 1 \
    --net-limit 100 \
  > delta-pmc-minimal.log 2>&1
```

Each `pmc:` row reruns the 100-net application. The CSV output includes kernel
start/end timestamps, so per-dispatch read and write bandwidth can be computed
from `FETCH_SIZE`/`WRITE_SIZE` and duration. Aggregate raw bytes and time by
kernel class; do not average per-dispatch bandwidths without weighting.

After the run, provide one archive containing the compatibility output,
configuration, log, and profiler output:

```bash
tar -czf delta-pmc-minimal-results.tar.gz \
  gfx1151-pmc-check.txt \
  gfx1151-pathfinder-pmcs.txt \
  delta-pmc-minimal.log \
  pathfinder-profiles/delta-pmc-minimal
```

This test would sharpen the memory-pipeline diagnosis, but it is no longer a
prerequisite for ranking the largest changes: the measured 52.9%
state-management share and sub-1.2% VALU-throughput proxies already establish
the first implementation targets. A calibrated physical bandwidth percentage
still requires an on-machine memory-bandwidth baseline or valid memory-clock
data.

## Artifacts and cleanup record

Cleanup was performed on 2026-07-16; the counter archive was added on
2026-07-17:

- deleted `gfx1151-pmcs.txt` and `gfx1151-pmc-info.txt`, because they were
  87-byte error logs rather than usable catalogs;
- deleted `rocprofv3-avail-help.txt` after using it to establish the correct
  global-option ordering (`rocprofv3-avail -d 0 <subcommand> ...`);
- processed and deleted the corrected 3.3 KiB/24 KiB gfx1151 catalog files
  after recording the available metrics and selected five-pass counter plan;
- deleted `analysis/`, a 1.0 GiB derived extraction containing a duplicate
  Perfetto trace, temporary SQL queries, text extracts, and an empty failed
  SQLite export; and
- retained `39155_results.db`, `rocprof-sys-analysis-bundle.tar.gz`,
  `step2-rocprof-sys-results.tar.gz`, and
  `delta-sol-valu-nonative-results.tar.gz` as original/source benchmark
  evidence.

The source archives should only be deleted after deciding that the profiles no
longer need to be re-queried. The generated counter visualization is also
retained as a compact presentation of these results.
