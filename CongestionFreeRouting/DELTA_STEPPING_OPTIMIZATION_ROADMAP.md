# Classic Delta-Stepping Optimization Roadmap

Updated 2026-07-28 for the host-checked relaxation pass. Compact row offsets,
generation-tagged current membership, capacity pre-reservation, the optional
reduced-round-trip controller, and three new relaxation A/B paths are
implemented and host-tested, but the changed HIP translation units have not
been compiled or run locally.

## Scope

This roadmap improves the existing classic Delta-Stepping implementation for
nonnegative weighted outgoing CSR graphs. It does not propose replacing it
with Near/Far or another SSSP algorithm. Shared primitives may also be reused
by UnitBFS.

The speed ranges below are engineering estimates, not measurements. They are
workload-dependent, overlap, and must not be added. The only retained hardware
profile predates compact parents, used unit weights on the generic all-light
path, and cannot rank genuinely mixed-weight behavior.

## Verified current implementation

- PathFinder can resolve `--delta auto` from runtime wave size, average
  effective edge weight, and average out-degree, or preserve an explicit
  numeric delta and multiplier sweep.
- Generic execution assigns one thread to each active row and scans that row
  serially.
- A mixed row is scanned once during light closure and again in the heavy
  phase. The all-light specialization skips the heavy phase.
- Future work occupies one flat pending set. Every bucket transition reduces
  the full pending set to find the next bucket and then scans it again to
  compact the selected bucket.
- The established/default controller copies frontier counts after light
  rounds, settled-target counts after buckets, next-bucket state, and compacted
  frontier counts to the host. Explicit worker streams retain extra completion
  boundaries for gfx1151 correctness. An opt-in capability-gated controller
  keeps those dependent phases in one grid-synchronized cooperative kernel for
  a bounded batch and publishes one compact descriptor; callbacks and
  unsupported kernels use the full host fallback.
- The legacy/default relaxation uses scalar global atomics for every competing
  distance update, queue claim, and queue reservation. A default-off wave32
  specialization now keeps distance/parent/membership atomics unchanged but
  combines each wave's successful touched/current/pending tail reservation.
- Compact vector-target runs use a 64-bit `{distance_bits,
  original_edge_id}` parent key. The old predecessor-row recovery is only a
  forced legacy or allocation fallback.
- Mutable generic scratch is about 48 B/V with compact parents, 60 B/V with
  legacy parents, and 40 B/V for the compile-time distances-only path. The
  graph also keeps a 4 B/E edge-to-source map when path-capable and eligible.
- Device row offsets automatically use `uint32_t` only when every CSR offset,
  including the terminal `nnz`, is representable. `kForce64Bit` retains the
  wide A/B path; public predecessor and compact-path edge IDs remain 64-bit.
- Six queues and three membership arrays are each sized to `V` in the generic
  path. Boolean `in_current` plus its clear kernel remains the default. An
  opt-in generation-tagged representation removes that clear and its dependent
  synchronization in either controller, with a full reset before token reuse.
- Deterministic weighted families, force-generic and force-legacy controls,
  opt-in telemetry, distances-only storage, and an exclusive distance bound
  are implemented and covered by test source.
- PathFinder supplies checked source/target capacity hints. Query and compact
  path buffers grow geometrically and retain high-water capacity; compact paths
  are never guessed from graph size. Compact-parent queries avoid legacy parent
  arrays, and strict distances-only storage ignores target/path hints.

### Controller boundary inventory

The two modes own the same classic-Delta queues and distance/parent state but
publish them at different boundaries. This is a structural inventory, not a
hardware result:

| Dependency boundary | Host-checked/default | Reduced-round-trip/opt-in |
| --- | --- | --- |
| Same-bucket light round | Boolean mode launches a membership clear, preserves its explicit-stream completion boundary, launches relaxation, copies the next-frontier count to host, and synchronizes. Generation mode removes only the clear. | Boolean clear and light relaxation are grid phases; generation uses a fresh reserved token. Queue payloads and counts cross a device fence plus cooperative grid barrier. |
| Target settlement | A vector target launches mark/count then copies the count; a scalar target copies its distance. Each host decision synchronizes. | Settlement follows completed light closure in the cooperative grid and becomes a sticky terminal descriptor status before any future-bucket heavy work. |
| Heavy phase | A separate relaxation launch runs after closure; vector-target consumers retain an explicit-stream dependency boundary. | Heavy relaxation is another grid phase in the same launch. |
| Pending minimum | The host initializes the minimum, synchronizes explicit streams, launches the full pending reduction, then copies/synchronizes the scalar minimum. | The leader initializes the minimum; the grid reduces it and crosses a grid barrier without a host scalar. |
| Pending compaction | The host resets two counters, synchronizes explicit streams, launches compaction, copies/synchronizes the current count, copies the pending count device-to-device, and preserves the next dependency boundary. | The grid compacts into ping-pong current/pending queues, validates bounded counts, advances queue parity, and resumes the successor bucket. |
| Bounded publication | Several independent scalar copies and synchronizations per bucket, plus one frontier-count decision per explicit-stream light round. | At most the configured number of light actions (including atomic bucket advancement) execute before one fixed-layout descriptor copy and host synchronization. |
| Failure/cleanup ownership | Allocation failures throw before traversal; predecessor/path validation throws after traversal; sparse cleanup and final reuse synchronization remain host-controlled. | The same pre/post boundaries apply. Queue capacity/invalid controller state are sticky during the batch; either forces a full-state reset before throwing. Allocation and invalid-predecessor failures therefore cannot be overwritten by a batched descriptor. |

```text
host: validate + allocate + initialize
                      |
                      v
device: [light closure -> settle -> heavy -> pending min -> compaction]
        [              repeat for at most B light actions              ]
                      |
                      v
host: one descriptor check -> stop, relaunch, extract, or full-reset error
```

The retained four-worker gfx1151 host baseline averaged about 133 dispatches
and 65 `hipStreamSynchronize` calls per query, with 34.09 seconds of zero
active-kernel time. The reduced-controller cells are deliberately marked “not
run” in that historical table. A later measured reduced-controller run was a
severe regression while host-checked and telemetry-enabled host-checked runs
were fast. The reduced controller therefore remains opt-in for A/B diagnosis
and is not a performance solution or optimization target.

The relaxation profile that now determines priority instantiated
`relax_light_edges_kernel<unsigned int, false, true, true, false, false, true,
false>`. Across 2,951 dispatches it accumulated 524.97 ms (177.9 us mean,
34.0 us median), with 97.28% wave dependency wait, 0.12 IPC, 20.32% reported
occupancy, 11.69% GL1 and 58.65% L2 hit rates, and approximately 23.3% of L2
requests atomic. “Wait for Counter” intensity was 112.13% and is not an
additive time fraction. This evidence prioritizes dependent-memory and queue
tail contention, not VALU, instruction-cache, or bulk-bandwidth tuning.

| Metric per query | Host-checked retained trace | Reduced controller |
| --- | ---: | ---: |
| Kernel dispatches | about 133 | not run |
| `hipStreamSynchronize` calls | about 65 | not run |
| True kernel-union idle | 34.09 s total (20.6% of span) | not run |

The current measured classic Delta-Stepping routing control uses
`--parallel-net-workers 4`. Keep that count explicit for comparable timing and
profiling runs; it is workload-specific and is not a portable algorithm
default.

The shared-capacity and Delta policy/model tests, both fake-HIP PathFinder
suites, and ASan+UBSan variants pass locally. The production Delta translation
unit and HIP regression were not compiled because this host has no `hipcc`,
ROCm, or AMD GPU. No speedup or GPU-memory saving is claimed.

## Optimization status and remaining ranking

The ranking is by expected speed benefit on the current routing-oriented
workload, with broader weighted-graph upside called out separately.

| Rank | Optimization | Status | Expected speed improvement | Difficulty | Why it ranks here |
| ---: | --- | --- | --- | --- | --- |
| 1 | Wave-aggregate touched/current/pending reservations in the exact profiled kernel | Implemented, opt-in, HIP-unvalidated | Unknown until profiler-free AMD A/B | High | One elected wave32 lane reserves each output queue for all winners. The edge-step loop is convergent; distance, deterministic parent, and first-membership atomics remain per edge. Wave64 and every unprofiled specialization fall back. |
| 2 | Elide verified-unit edge-value loads | Implemented, opt-in, HIP-unvalidated | Unknown until profiler-free AMD A/B | Low | An immutable proof tied to the uploaded value payload selects exact `1.0f`; delta/all-light state never implies unit weights, and force-generic remains classic Delta. |
| 3 | Pass authoritative host frontier counts by value | Implemented, opt-in, HIP-unvalidated | Unknown until profiler-free AMD A/B | Low | Eligible explicit-stream host-checked rounds already have a synchronized, range-checked count. Device-produced/default-stream rounds retain the pointer load and barrier, and the gfx1151 clear-to-relax boundary is unchanged. |
| — | Reduced-round-trip controller | Preserved opt-in; measured severe regression | Do not pursue as a performance solution | Very high | The measured regression is controller-specific, not telemetry, conversion, graph loading, or four-stream contention. Keep it only for existing A/B coverage. |
| 4 | Generation-tagged `in_current` | Implemented, opt-in, HIP-unvalidated | 10--30% end-to-end, 15--40% traversal when many vertices are touched | High | The new representation removes only the current-membership clear path. Sparse reset of distance, parent, pending, and heavy state remains necessary. Boolean/clear remains default until AMD validation. |
| 5 | Eligible 32-bit device row offsets with forced-wide A/B | Implemented, automatic, HIP-unvalidated | 5--15% traversal plus 4 B/V shared-graph savings | Medium | Complete-range eligibility is exact at `UINT32_MAX`; every row-reading kernel is typed, while the public CSR/path edge identity remains 64-bit. |
| 6 | Add a degree-aware outgoing-edge expander | Not implemented | 0--15% on the mostly short-row routing graph; 10--40% on skewed weighted graphs | High | Thread-per-row is appropriate for short rows but serializes long rows. Use lane groups, wave-per-row, and CTA/edge-balanced paths only above measured reached-degree thresholds. |
| 7 | Reduce candidates by destination before global atomics | Not implemented | 0--10% on low-collision routing frontiers; 10--30% when destinations collide heavily | High | The current kernel performs one distance atomic and queue decision per eligible edge. Wave/block aggregation is worthwhile only after convergent edge assignment and collision telemetry exist. |
| 8 | Batch independent searches in one launch | Not implemented | 5--30% throughput when individual frontiers underfill the GPU | Very high | It can amortize launches and fill small frontiers, but the retained worker sweep was nearly flat from 2 to 8 workers and the old trace already showed high overlap. Implement only after per-query state is smaller and current single-query control is measured. |
| 9 | Prepartition immutable adjacency into light and heavy edge ranges | Not implemented | 0% for all-light routing runs; 10--35% for truly mixed fixed weights | High | It avoids rescanning mixed rows, but a new delta or destination-cost update can invalidate the partition. Keep the existing direct path for all-light and mutable-cost workloads. |
| 10 | Replace flat pending scans with circular/windowed buckets plus a nonempty bitmap | Not implemented | About 0--5% on the retained all-light profile; 5--30% on broad weighted bucket spans | High | Pending management was only about 1% in the historical unit-weight trace, so this must be justified by new weighted telemetry before implementation. |
| 11 | Tune weighted automatic delta and refresh it after mutable value/cost updates | Partially implemented | 0--25% on weighted workloads | Low--Medium | Exact-unit effective weights now select the natural width `1 * multiplier`; weighted graphs retain the graph-aware seed. The remaining work is a target-GPU weighted multiplier sweep and an optional workspace-owned statistic refresh after updates. |
| 12 | Pre-reserve and geometrically retain query/path buffers | Implemented, enabled, HIP-unvalidated | 2--8% where allocation/free is visible | Low | Metadata-derived source/target reservations are active, all growth retains geometric high water, and compact paths remain demand-sized. |
| 13 | Tune block sizes, launch bounds, architecture flags, and compiler options | Not implemented | 0--10% | Low | Useful after structural kernels stabilize; it cannot remove the present controller or state traffic. |

## Recommended implementation sequence

### Gate 0: current weighted baseline

Before enabling or tuning the new kernel paths:

1. Compile and run `delta_stepping_hip_test` on the target AMD GPU.
2. Compare every test family with CPU Dijkstra, including zero-weight SCCs,
   repeated workspaces, explicit streams, compact/legacy parents, and
   distances-only storage.
3. Record profiler-free medians for all-light, all-heavy, seeded mixed, and a
   representative real weighted CSR in both path-producing and distances-only
   modes. Use four PathFinder workers for the current routing baseline; worker
   count is a benchmark control, not a portable algorithm default.
4. Capture telemetry with reached-degree histograms and
   candidate-to-unique-destination ratios added to a diagnostic build. Keep
   telemetry off for wall-time measurements.
5. Reprofile compact-parent execution; do not use the historical legacy-parent
   materialization share as a current result.

### Phase 1: compact offsets — implemented, HIP-unvalidated

Template or specialize graph row access on 32- versus 64-bit device offsets.
Keep public path edge IDs as 64-bit and retain a forced-wide mode. This is the
most bounded implementation change and gives a clean A/B before controller or
state semantics change.

Acceptance: exact result equivalence for both offset modes, no overflow at the
cutoff, lower shared graph memory, and a repeated-run traversal win or neutral
result on the target graph.

### Phase 2: state and reset reduction — partially implemented, opt-in

The bounded pass implements generation tags for `in_current` only. Distance,
parent, pending, and heavy state retain their existing sparse cleanup. AMD
validation must confirm the following before generation membership can become
the default:

- retain the existing distance, parent, pending, and heavy cleanup on every
  normal, early-stop, and exceptional exit;
- preserve atomic publication of the winning distance and parent edge;
- append each vertex at most once per generation while still permitting a
  processed vertex to re-enter after a later same-bucket decrease;
- survive token rollover through a full reset before token reuse;
- preserve current/pending/heavy queue semantics across early stop,
  callbacks, exceptions, and workspace reuse; and
- remove the Boolean clear kernel and its dependent synchronization only in
  the generation path.

Evaluate queue-buffer aliasing or bounded growth separately. The six queues
overlap only partially in lifetime, and touched vertices are needed until
cleanup, so unsafe buffer reuse is not acceptable.

### Phase 3: device-resident classic-Delta controller — implemented, opt-in,
HIP-unvalidated

The reduced-round-trip mode moves light closure, target settlement,
heavy-phase completion, next-bucket selection, and pending compaction into one
bounded cooperative controller. Its queue/count dependencies cross only
grid-wide barriers inside that kernel; release publication exposes one
fixed-layout descriptor at the host boundary. Runtime capability/occupancy
checks and callback policy select the unchanged host controller when the fused
path cannot be used. Queue overflow and invalid state are sticky terminal
statuses followed by full cleanup before workspace reuse.

The batch bound is configurable, and batch size one is the host-model
equivalence control. Boolean membership and the optional generation-tagged
representation are both retained; generation rollover clears tags before
token reuse. The implementation does not use HIP Graph capture or hard-code
the four-worker benchmark control into portable workspace behavior.

Acceptance remains outstanding: compile on ROCm, pass the full controller ×
offset × membership × parent/output matrix, survive repeated concurrent
explicit-stream stress on gfx1151, preserve callback/iteration/target cleanup,
and demonstrate fewer dispatches/synchronizations plus a profiler-free
end-to-end win with identical route results.

### Phase 4: shared edge expansion and contention control

The first bounded contention-control slice is implemented for the exact
profiled specialization. A wave owns up to 32 frontier rows, reduces the
maximum row length, executes one uniform masked edge-step loop, and performs
three independent ballot/prefix/reservation sequences. It is enabled only
with exact-unit load elision, wave32, compact rows, Boolean membership, compact
edge parents, no costs/heavy work/telemetry, and the host-checked controller.
All other combinations use the unchanged scalar row loop. Host tests cover
wave32 and wave64 prefix arithmetic (wave64 is model-only), empty/sparse/full
ballots, highest lanes, capacity boundaries, parent ties, and randomized
classic-Delta scheduling equivalence. HIP correctness and performance remain
outstanding.

After that bounded path is validated, build a convergent degree-aware edge
assignment for broader weighted workloads:

- thread-per-row for short rows;
- lane groups or packed waves for medium rows;
- wave-per-row for long rows; and
- CTA or edge-balanced processing for extreme rows.

Only then add wave-local queue reservation and destination grouping. Add a
direct-atomic bypass for small or low-collision frontiers. Consider block-local
aggregation before any global radix sort/reduce.

### Phase 5: weighted scheduler refinements

Use new mixed-weight telemetry to choose between static light/heavy
partitioning and circular/windowed buckets. These solve different measured
costs and should not be implemented together by default:

- partitioning targets duplicate edge scans within a processed bucket;
- circular/windowed buckets target repeated global pending scans between
  buckets.

Destination vertex costs and mutable edge weights require either rebuilding
partitions or retaining the current unpartitioned fallback.

## Correctness invariants

Every optimization must preserve:

1. outgoing CSR orientation and original public edge identity;
2. finite nonnegative weights and `edge_weight(u,v) * vertex_cost(v)`;
3. strict monotone distance decreases and deterministic valid parent ties;
4. zero-weight edges, parallel edges, self-loops, and zero-weight SCCs;
5. settlement only after no current or earlier bucket work can lower a target;
6. duplicate source/target semantics and paths rooted at a requested source;
7. exact iteration limits, exclusive distance bounds, and callback timing;
8. cleanup after convergence, target stop, bounds, callback exceptions, and
   extraction exceptions;
9. null-stream and concurrent explicit-stream ordering on gfx1151; and
10. a 64-bit offset fallback when the compact representation is ineligible.

## Measurement contract

Report graph upload, traversal, reset, controller, path extraction, and total
PathFinder time separately. Use warm-ups, repeated medians with dispersion,
identical graph/query order, fixed clocks when possible, fixed worker count,
and exact output validation. Record delta, weights, destination costs, compiler
flags, ROCm version, device, offset mode, parent mode, and telemetry state for
every comparison.
