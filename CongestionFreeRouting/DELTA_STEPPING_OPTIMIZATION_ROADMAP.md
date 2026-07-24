# Classic Delta-Stepping Optimization Roadmap

Updated 2026-07-24 for the bounded optimization pass. Compact row offsets,
generation-tagged current membership, and capacity pre-reservation are
implemented and host-tested, but not compiled or run with HIP.

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
- The controller copies frontier counts after light rounds, settled-target
  counts after buckets, next-bucket state, and compacted frontier counts to the
  host. Explicit worker streams retain extra completion boundaries for gfx1151
  correctness.
- Scalar global atomics publish every competing distance update, queue claim,
  and queue reservation.
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
  synchronization only in the new path, with a full reset before token reuse.
- Deterministic weighted families, force-generic and force-legacy controls,
  opt-in telemetry, distances-only storage, and an exclusive distance bound
  are implemented and covered by test source.
- PathFinder supplies checked source/target capacity hints. Query and compact
  path buffers grow geometrically and retain high-water capacity; compact paths
  are never guessed from graph size. Compact-parent queries avoid legacy parent
  arrays, and strict distances-only storage ignores target/path hints.

The shared-capacity and Delta policy/model tests, both fake-HIP PathFinder
suites, and ASan+UBSan variants pass locally. The production Delta translation
unit and HIP regression were not compiled because this host has no `hipcc`,
ROCm, or AMD GPU. No speedup or GPU-memory saving is claimed.

## Optimization status and remaining ranking

The ranking is by expected speed benefit on the current routing-oriented
workload, with broader weighted-graph upside called out separately.

| Rank | Optimization | Status | Expected speed improvement | Difficulty | Why it ranks here |
| ---: | --- | --- | --- | --- | --- |
| 1 | Keep classic-Delta bucket and light-closure control on the GPU | Not implemented | 20--50% end-to-end on control-bound searches; potentially larger for many shallow buckets | Very high | The retained trace attributed most host time to tiny copies/synchronizations and 41.7% of aggregate device-dispatch duration to runtime status/copy kernels. Current code still performs several scalar host decisions per bucket and checks every explicit-stream light round. |
| 2 | Generation-tagged `in_current` | Implemented, opt-in, HIP-unvalidated | 10--30% end-to-end, 15--40% traversal when many vertices are touched | High | The new representation removes only the current-membership clear path. Sparse reset of distance, parent, pending, and heavy state remains necessary. Boolean/clear remains default until AMD validation. |
| 3 | Eligible 32-bit device row offsets with forced-wide A/B | Implemented, automatic, HIP-unvalidated | 5--15% traversal plus 4 B/V shared-graph savings | Medium | Complete-range eligibility is exact at `UINT32_MAX`; every row-reading kernel is typed, while the public CSR/path edge identity remains 64-bit. |
| 4 | Add a degree-aware outgoing-edge expander | Not implemented | 0--15% on the mostly short-row routing graph; 10--40% on skewed weighted graphs | High | Thread-per-row is appropriate for short rows but serializes long rows. Use lane groups, wave-per-row, and CTA/edge-balanced paths only above measured reached-degree thresholds. |
| 5 | Reduce candidates by destination before global atomics | Not implemented | 0--10% on low-collision routing frontiers; 10--30% when destinations collide heavily | High | The current kernel performs one distance atomic and queue decision per eligible edge. Wave/block aggregation is worthwhile only after convergent edge assignment and collision telemetry exist. |
| 6 | Batch independent searches in one launch | Not implemented | 5--30% throughput when individual frontiers underfill the GPU | Very high | It can amortize launches and fill small frontiers, but the retained worker sweep was nearly flat from 2 to 8 workers and the old trace already showed high overlap. Implement only after per-query state is smaller and current single-query control is measured. |
| 7 | Prepartition immutable adjacency into light and heavy edge ranges | Not implemented | 0% for all-light routing runs; 10--35% for truly mixed fixed weights | High | It avoids rescanning mixed rows, but a new delta or destination-cost update can invalidate the partition. Keep the existing direct path for all-light and mutable-cost workloads. |
| 8 | Replace flat pending scans with circular/windowed buckets plus a nonempty bitmap | Not implemented | About 0--5% on the retained all-light profile; 5--30% on broad weighted bucket spans | High | Pending management was only about 1% in the historical unit-weight trace, so this must be justified by new weighted telemetry before implementation. |
| 9 | Tune automatic delta and refresh it after mutable value/cost updates | Not implemented | 0--25% on weighted workloads | Low--Medium | PathFinder seeding is implemented. The remaining work is a target-GPU multiplier sweep and an optional workspace-owned statistic refresh after updates. |
| 10 | Pre-reserve and geometrically retain query/path buffers | Implemented, enabled, HIP-unvalidated | 2--8% where allocation/free is visible | Low | Metadata-derived source/target reservations are active, all growth retains geometric high water, and compact paths remain demand-sized. |
| 11 | Tune block sizes, launch bounds, architecture flags, and compiler options | Not implemented | 0--10% | Low | Useful after structural kernels stabilize; it cannot remove the present controller or state traffic. |

## Recommended implementation sequence

### Gate 0: current weighted baseline

Before enabling or tuning the new kernel paths:

1. Compile and run `delta_stepping_hip_test` on the target AMD GPU.
2. Compare every test family with CPU Dijkstra, including zero-weight SCCs,
   repeated workspaces, explicit streams, compact/legacy parents, and
   distances-only storage.
3. Record profiler-free medians for all-light, all-heavy, seeded mixed, and a
   representative real weighted CSR in both path-producing and distances-only
   modes. Use two PathFinder workers for the future routing baseline; worker
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

### Phase 3: device-resident classic-Delta controller

Move light closure, heavy-phase completion, target settlement, next-bucket
selection, and pending compaction continuation into a resident or bounded
cooperative controller. Preserve an instrumented host fallback for unsupported
devices and progress callbacks.

Do not treat HIP Graph capture alone as the solution: the current launch
sequence depends on scalar device results. Capture becomes useful only after
those decisions no longer return to the host.

Acceptance: identical bucket/iteration semantics, correct concurrent explicit
streams on gfx1151, no scalar D2H dependency inside steady-state traversal,
and a measured end-to-end win with compact-parent reset included.

### Phase 4: shared edge expansion and contention control

Build a convergent degree-aware edge assignment:

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
