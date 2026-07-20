# Delta-Stepping Optimization Roadmap

## Audit status (2026-07-19)

The original summary contained 22 equally weighted items (priorities `0` through
`21`). The audit found 11 fully implemented items, or **50.0%** (`11 / 22`).
Three additional items are partial; assigning each half credit gives **56.8%**
(`12.5 / 22`). Completed tasks have been removed from the active roadmap.

The three partial items are:

- the direct HIP regression is extensive, and the target gfx1151 run exposed
  an explicit-stream controller-state failure; full validation of the
  guarded path remains;
- query/path buffers now grow geometrically, but PathFinder does not pre-reserve
  metadata-derived capacities; and
- compile-time kernel modes exist, but architecture, block-size, batch-size,
  launch-bound, and compiler tuning have not been performed on the target GPU.

The code audit confirmed the completed exact-unit specialization and four-level
status batching on the default/null stream; shared immutable CSR with lazy
24-byte unit scratch and memory-aware workers; default-stream four-round generic
light-closure batching; device-counted heavy and pending scans; compile-time
relaxation modes; thread-per-short-row relaxation; per-successful-append atomic
queue reservations; removal of overflow and touched membership
synchronization/state; pinned scalar transfers with device-final pending
reduction; float-correct same-bucket closure and terminal saturation; and
race-safe parent publication with sparse exact-edge recovery.

Parallel explicit worker streams now host-check every exact-unit BFS level and
every generic light-closure round. Four-round device-controlled batching is
gated to `stream == nullptr`: on gfx1151, batched dependent kernels otherwise
produced state in which expansion counters advanced without matching frontier
bounds. Retain this correctness gate until the failure is resolved and the
affected runtime is validated.

A later correctness audit removed wave-coalesced reservation from irregular
adjacency loops because lanes can reach different dynamic collective calls.
The completion percentages above are historical; a safe buffered reservation
replacement remains future work.

## Target-GPU profiling update (2026-07-17)

The first hardware-counter profile now exists for the Radeon 8060S/Strix Halo
(`gfx1151`, 40 CUs, wave32). It used one worker and the first 100
`logicnets_jscl` nets. The finite `--max-sssp-iters 2147483647` argument forced
the generic all-edges-light Delta-Stepping path; these measurements do not
describe the exact-unit specialization or a genuinely mixed-weight graph. The
full-device CSR has 28,226,432 rows, above the specialization's
`2^24 = 16,777,216`-row exact-float guard, so removing the finite option alone
would still leave this particular graph on generic Delta-Stepping.

Four counter-replay passes were highly repeatable: summed kernel time was
604.4--608.7 ms (0.39% coefficient of variation). Mean kernel-time allocation
was:

| Phase | Kernel time (ms) | Share |
| --- | ---: | ---: |
| Light-edge relaxation | 259.6 | 42.8% |
| Sparse touched-state reset | 175.1 | 28.9% |
| Predecessor materialization | 145.9 | 24.1% |
| All other kernels and runtime helpers | 25.9 | 4.3% |

Reset plus predecessor materialization therefore consumes **52.9%** of kernel
time. Their combined kernel-only Amdahl ceiling is 2.12x if they were removed
entirely; this is not an end-to-end prediction because graph loading, host
round trips, and output remain.

| Hot kernel | Profiler VALU estimate vs FP32 single-issue peak | GL1 hit | GL2 hit | Resident waves per WGP | Wave cycles waiting for any reason |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relax light edges | 1.12% | 1.83% | 21.41% | 54.25 / 64 | 99.42% |
| Reset touched vertices | 0.09% | ~0.00% | 16.45% | 44.31 / 64 | 53.78% |
| Materialize predecessors | 0.39% | 4.04% | 19.74% | 52.26 / 64 | 99.80% |

The VALU percentage follows ROCm Compute Profiler's gfx115x System SOL formula:
wave32 aggregate VALU instructions divided by the 14.848-TFLOP/s FP32
single-issue FMA ceiling at 2.9 GHz. The numerator includes integer and logic
VALU instructions, so it is a conservative throughput proxy rather than a true
floating-point operation count. Even with that caveat, the result decisively
rules out arithmetic-throughput saturation. High wave residency, low cache hit
rates, and high wait fractions instead point to irregular memory dependency and
state-traffic costs.

The separate `compute_thruput_util` metric-set run was incomplete on this
ROCm/gfx1151 combination: metrics 11.2.5 and 11.2.6 were skipped and the output
contained no VALU instruction counter. Do not use that run to claim VALU, VMEM,
SALU, or branch utilization. The System SOL counters above are the usable data.
Likewise, `max_mclk` was reported as zero, so request-derived fabric rates are
not calibrated physical DRAM bandwidth percentages.

The work is strongly skewed across nets. The ten largest searches account for
72.4% of reset time and 69.0% of materialization time; launch size and duration
correlate at 0.995 and 0.997 respectively. Future telemetry should attach the
net identity and reached/touched count to each search so these outliers can be
reproduced directly.

The retained profile predates the explicit-stream safety gate, so reprofile
before using its launch or synchronization counts to predict current parallel
worker performance.

Except for the measured values above, percentages below remain engineering
estimates. They are not additive and must be validated separately for
exact-unit and genuinely weighted graphs.

## Remaining work

| Priority | Optimization or validation | State | Expected improvement | Risk | Invasiveness |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Run correctness and Unit-BFS performance validation on the target AMD GPU; test Delta exact-unit only on an eligible graph | Partial: counter run complete; explicit-stream failure reproduced and guarded-path validation pending | Establishes correctness and the relevant production baseline | Low | Low |
| 2 | Publish the winning compact edge ID and eliminate predecessor-row recovery | Parent publication complete; recovery is measured at 24.1% of kernel time | Up to 1.32x kernel-only if removed | Medium | Medium--High |
| 3 | Remove avoidable sparse-reset writes, then evaluate generation-stamped state | Reset is measured at 28.9% of kernel time | Up to 1.41x kernel-only if removed | Medium | Medium--High |
| 4 | Add adaptive 32-bit internal row and predecessor-edge offsets | Not started; strongly supported by low cache hit rates | 5--20%, plus lower worker memory | Medium | Medium |
| 5 | Reduce host round trips with safe device-resident batching and then HIP graph capture | Four-round batching is null-stream-only; explicit streams host-check each level/round | 5--25% end-to-end | Medium | Medium--High |
| 6 | Batch independent nets in one launch | Not started; worker-count scaling was flat | 1.5--5x when individual searches underfill the GPU | High | High |
| 7 | Pre-reserve query/path buffers from route metadata | Partial: geometric growth complete | 2--8% | Low | Low |
| 8 | Tune block size, launch bounds, architecture, and compiler | Partial: hot kernels already show high wave residency | 0--10% | Low | Low |
| 9 | Add a hybrid degree-aware scheduler | Still profile-gated; reached-row degree data is missing | 5--35% only if long rows dominate | Medium | Medium |
| 10 | Replace pending rescans with circular buckets and a nonempty bitmap | Deprioritized for this run: reduce plus compact are only 1.0% of kernel time | 1.2--4x only on many-bucket weighted graphs | High | High |
| 11 | Prepartition light and heavy adjacency | Not evaluated by the all-edges-light profile | 10--80% only on mixed-weight rows | Medium | High |
| 12 | Add persistent/asynchronous Delta-Stepping | Future research after simpler batching | 1.3--3x on suitable weighted graphs | High | High |

## Highest-priority changes

### 1. Validate the relevant production path on the target AMD GPU

Compile and repeatedly run `tests/delta_stepping_hip_test.cpp`. The suite already
covers exact-unit and arbitrary weighted graphs, float bucket boundaries,
terminal-bucket saturation, duplicate sources/targets, zero-weight SCCs,
limited iterations, callbacks, workspace reuse, concurrent streams, graph
sharing, updates, vertex costs, queue boundary cases, and CPU-Dijkstra result
comparison. Host-only or fake-HIP tests cannot validate real ballot, shuffle,
atomic, stream, or floating-point device behavior.

Exercise both execution modes explicitly: null-stream cases must cover
four-round device batching, while concurrent nonblocking streams must preserve
the host-visible boundary between every level/round. Do not infer
explicit-stream safety from a passing default-stream batch.

The counter run validates that the generic path executes. The finite
maximum-iteration argument disabled the exact-unit fast path, but the
28,226,432-row full-device CSR also exceeds that path's `2^24`-row guard.
Therefore, benchmark dedicated Unit-BFS against generic Delta-Stepping first;
Delta without the finite argument is still generic for this graph. Validate the
Delta exact-unit specialization separately on an eligible smaller/bounded CSR,
or redesign it around integer BFS depth before considering it a full-device
production option. Then sweep `1`, `2`, `4`, and `8` workers. For a genuinely
weighted generic path, record light-round counts, remaining synchronizations,
pending live/stale ratios, reached-row degrees, exact-edge recovery time, and
allocation stalls.

### 2. Eliminate predecessor-row recovery

The current `parent_key` atomically publishes distance bits plus predecessor
node, after which `materialize_predecessors_kernel` scans the predecessor's CSR
row to rediscover the exact edge. The materialization pass is now measured at
24.1% of all kernel time and has a 19.7% GL2 hit rate with 99.8% of wave cycles
in `SQ_WAIT_ANY`.

For graphs with fewer than `2^32` edges, replace the low 32 bits of the winning
key with the original edge ID. Prefer a single nonnegative-float-ordered
64-bit `{distance_bits, edge_id}` atomic state so distance and parent cannot
diverge. Build one shared 32-bit edge-to-source array, or an equivalent compact
mapping, and have target path extraction decode the state directly. This can
remove the predecessor materialization launch, the dependent row scan, and
potentially the separate `pred_node` and `pred_edge` arrays. Retain the current
wide fallback and test equal-distance parallel edges and deterministic
tie-breaking explicitly.

### 3. Reduce or eliminate sparse reset

`reset_touched_vertices_kernel` is 28.9% of kernel time and writes seven
scattered arrays per reached vertex. Start with a low-risk specialized reset:
do not clear `pred_node` or `pred_edge` when current-run distance/key validity
already gates their use, and do not touch `in_heavy` in the compiled
all-edges-light mode. Prove which membership flags are zero at normal and
target-early termination, clearing only the queues that can remain live.

Then A/B a generation-stamped state word that treats stale distance, parent,
and membership values as logically empty without scattering a reset over every
touched vertex. Account for the extra checks/atomics in relaxation; a faster
reset that slows the 42.8% relaxation kernel is not a win. The ten hard nets
that dominate reset time should be a dedicated regression subset.

### 4. Use compact internal offsets

Delta-Stepping still uses 64-bit `minplus_sparse::Offset` for device row offsets
and predecessor edges. When `nnz <= INT32_MAX`, convert row offsets to signed
32-bit, store compact predecessor-edge IDs, instantiate both offset widths, and
widen only while producing public output. Keep 64-bit on-disk and public edge
IDs and add a forced-wide same-graph A/B mode.

Prefer packing the winning 32-bit original edge ID into the existing 64-bit
distance/key state. A shared compact edge-source map can then recover the
predecessor without scanning the winning predecessor's row, completing the
remaining exact-edge-recovery task while preserving strict-decrease parent
publication.

### 5. Add degree-aware scheduling only if profiles justify it

Thread-per-vertex is appropriate for short FPGA rows. The measured relaxation
kernel already has about 54 resident waves per WGP and only 1.12% of the
profiler's arithmetic peak, so occupancy or VALU throughput alone does not
justify a cooperative-row rewrite. First profile the degree distribution of
vertices actually reached by the ten hard nets. If long or skewed rows dominate,
use one thread for short rows, a few lanes for medium rows, one wave for long
rows, and optionally one block for extreme outliers. A first version can divert
only long rows to a secondary cooperative queue.

### 6. Finish allocation and target tuning

The scratch allocator now grows sources, targets, and compact paths
geometrically. PathFinder should additionally reserve maximum source/sink
capacities from route metadata before workers start. Do not reserve worst-case
`targets * vertices` path storage. Move target prefix/offset work to the device
only if extraction remains material after compact parent work.

Measure four-round batching against other batch sizes on the default/null
stream. Benchmark explicit worker streams separately with their required
per-level/per-round host checks; do not enable speculative batching there until
the gfx1151 controller-state failure is resolved. Profile 128- and 256-thread
blocks per kernel. Current hot kernels use 24 VGPRs, 128 SGPRs, no scratch, and
no dynamic LDS, while measured wave residency is already high. Treat block-size
tuning as a secondary experiment. Inspect register pressure before adding
`__launch_bounds__`; use exact `--offload-arch`; test wave32/wave64 where
supported; add `__restrict__` only to proven non-aliasing pointers; and use
fast-math only if the float-boundary regressions remain correct.

## Conditional weighted-graph redesigns

The generic implementation maintains one unique pending set, scans it for the
minimum bucket, and scans again to select/compact. In the current all-edges-light
profile, reduction plus compaction is only 1.0% of kernel time, so a circular
bucket redesign is not justified. Reconsider it only when a genuinely weighted
profile shows many live buckets. The design must handle ring generations, stale
entries, concurrent bit clearing/appending, terminal float buckets, duplicate
capacity, and target settlement by absolute bucket.

Prepartition light/heavy adjacency only for workloads that repeatedly scan
mixed rows. Preserve a mapping to original CSR edge IDs. Destination vertex
cost updates change effective edge classes, so either rebuild the partition or
fall back to the current scan when costs are active.

Reset cost is now material, so compact/epoch-stamped state has moved into active
work. Add HIP graph capture only after batch tuning and only if launch gaps
remain. Attempt multi-net launches or persistent ADDS only if tuned independent
streams still underfill the GPU.

## Correctness invariants

Future changes must preserve:

1. outgoing CSR orientation and original 64-bit public edge identity;
2. finite nonnegative edge weights and destination-cost multipliers;
3. complete same-bucket light closure, including nominally heavy candidates
   that round into the current bucket and saturated terminal-bucket edges;
4. heavy relaxation from every vertex settled in the bucket using its final
   distance for that bucket;
5. strict monotone distance decreases and the queue uniqueness/capacity proof;
6. target settlement only after its bucket is fully closed;
7. duplicate source/target semantics and sparse workspace reset;
8. exact-unit gating: unit weights, no vertex costs, at most `2^24` rows,
   unlimited iterations, no callback, and the target-set overload;
9. float bucket calculation by nonnegative `distance / delta` truncation;
10. race-safe strict-winner parent publication, including zero-weight SCCs;
11. per-workspace stream ordering, including null-stream-only four-round
    batching and explicit-stream host checks, plus shared-graph lifetime/device
    rules.

## Verification and benchmark protocol

Run focused sizes around the hardware wave and block boundaries, including
`1`, `waveSize - 1`, `waveSize`, `waveSize + 1`, `255`, `256`, and `257`.
Exercise zero through extreme degrees; all/sparse/no-lane appends; exactly `V`
queue entries; values immediately around bucket boundaries; all-light,
all-heavy, and mixed rows; pending-to-current decreases; duplicate and source
targets; unreachable nodes; self-loops, parallel edges, unsorted rows;
iteration/batch boundaries; callbacks; updates; repeated calls; and concurrent
streams. Compare distances with CPU Dijkstra and validate paths independently.

Report separately:

1. graph validation/upload and workspace construction;
2. traversal through target settlement;
3. predecessor/path extraction and sparse reset; and
4. full PathFinder wall time across many nets.

Use HIP events for GPU intervals, wall time for end-to-end routing, a warm-up,
and repeated-run medians with dispersion. Keep graph, net order, target sets,
worker count, validation, clocks, compiler, architecture, and delta fixed for
comparisons. Maintain both production exact-unit and genuinely weighted suites.
