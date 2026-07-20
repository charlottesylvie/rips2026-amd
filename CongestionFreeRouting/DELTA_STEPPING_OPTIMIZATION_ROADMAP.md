# Generic Delta-Stepping Optimization Roadmap

## Project direction (2026-07-20)

The primary optimization target is now the generic nonnegative-weight SSSP
backend. UnitBFS remains a useful specialization, but project-wide scheduling,
memory-layout, and traversal work should first improve arbitrary weighted
graphs.

The highest-upside direction is not another incremental flat-bucket change. It
is a cuGraph-inspired adaptive Near/Far scheduler backed by reusable
degree-aware edge expansion and destination aggregation. The existing
Delta-Stepping implementation should remain available as a correctness and
performance reference while this backend is developed.

Impact estimates below are qualitative until they are measured on genuinely
weighted graphs. They are workload-dependent and are not additive.

## Current implementation audit

The generic path currently has these important properties:

- Low-level runs still take a numeric `delta`. PathFinder can now resolve
  `--delta auto` once from the active HIP wave size, average edge weight, and
  average out-degree, with an explicit multiplier and numeric override. A
  host helper also supports exact destination-cost-weighted statistics.
- One GPU thread owns each active CSR row and scans that row serially.
- Mixed rows are scanned during light closure and scanned again during the
  heavy pass.
- Future work is stored in one flat pending set. The implementation scans it to
  find the minimum bucket, then scans it again to select and compact that
  bucket.
- Frontier counts, target settlement, and the next bucket still create
  host-visible controller boundaries. Explicit worker streams host-check every
  generic light-closure round because dependent batched dispatches exposed a
  controller-state failure on gfx1151.
- Queue reservations and competing distance updates use scalar global atomics.
- Automatic compact-parent generic scratch is approximately 48 bytes per
  vertex before per-target buffers because legacy predecessor arrays are now
  lazy. At 28,226,432 vertices this is approximately 1.26 GiB per worker.
  Forced-legacy or compact-map allocation fallback still grows to about 60
  bytes per vertex (1.58 GiB).
- Explicit `run_distances()` calls now instantiate the same generic scheduler
  with parent publication compiled out and release mutable parent, target, and
  path buffers. Its core scratch is approximately 40 bytes per vertex. A
  `kDistancesOnly` graph/workspace also omits the 4-byte-per-edge edge-source
  map; path-output calls on that storage are rejected.
- PathFinder can deliberately select the generic scheduler with
  `--delta-force-generic`, synthesize deterministic all-light/all-heavy/seeded
  mixed weight families, and emit one aggregate telemetry JSON record with
  `--delta-telemetry`. These controls establish reproducible experiments, but
  no target-AMD timing baseline has been recorded in this repository.
- Device CSR row offsets and legacy predecessor edges are always 64-bit.

Two previously listed tasks are already complete for automatic vector-target
runs and must not remain active priorities:

- the winning 64-bit parent key already contains `{distance_bits,
  original_edge_id}` when the graph has an eligible compact edge map; and
- compact target-path extraction uses that edge ID directly, so predecessor-row
  recovery is now only a legacy or allocation-fallback behavior.

The retained 2026-07-17 gfx1151 profile predates the compact-parent path and
used generic **all-edges-light** Delta-Stepping. Its 24.1% predecessor
materialization measurement describes the old fallback, not current automatic
vector-target routing. Its pending-bucket result also cannot predict a
heterogeneous weighted workload.

## Ranked remaining changes

| Impact rank | Optimization | Expected impact | Invasiveness | Status and rationale |
| ---: | --- | --- | --- | --- |
| 1 | Add an adaptive two-level Near/Far backend | Very high on heterogeneous weighted graphs | Very high | Four queues, bounded urgent work, queue sharding, and lazy stale filtering replace the flat pending scan and classical light/heavy traversal structure. |
| 2 | Slim generic state and reset traffic | High | Medium--High | Add mode-specific allocation, compact or versioned membership, bounded queues, and eligible 32-bit row offsets. This directly targets the large per-worker footprint and irregular writes. |
| 3 | Add degree-aware, edge-balanced expansion | High on general or skewed graphs; modest on uniformly short rows | High | Retain thread-per-row for short rows, but add packed-wave, wave-per-row, and CTA/edge-balanced paths for longer rows. |
| 4 | Aggregate competing candidates by destination before global updates | High when frontier destinations collide | High | Start with wave-local aggregation after edge balancing; use block/global sort-reduce only when telemetry justifies it. |
| 5 | Remove the generic host-controlled inner loop | High for many small closure rounds or buckets | High | Keep scheduler state on the device. Evaluate persistent/cooperative execution or HIP Graph capture only after data dependencies no longer require host decisions. |
| 6 | Add a distances-only execution mode | High for competitive SSSP benchmarks; low for routing calls that require paths | Low--Medium | **Implemented; AMD validation pending.** `run_distances()` compiles out all parent access and releases mutable path state; strict storage also omits `edge_source`. |
| 7 | Validate and tune automatic delta selection | Medium--High and workload-dependent | Low | PathFinder resolution, numeric override, multiplier, runtime wave size, destination-cost helper, and numeric edge-case clamps are implemented. Weighted GPU sweeps and workspace-integrated refresh after value/cost updates remain. |
| 8 | Batch independent searches in one launch | High throughput upside when one search underfills the GPU | Very high | Pursue only after per-search state is smaller and the single-query scheduler is stable. |
| 9 | Prepartition static light/heavy adjacency | Medium--High only on fixed-weight mixed rows | High | Conditional fallback for classic Delta-Stepping. Destination vertex-cost updates can invalidate the partition. |
| 10 | Add circular buckets and a nonempty bitmap | Medium--High only when retaining classic Delta-Stepping | High | Near/Far largely supersedes this redesign. Implement it only if classic Delta remains the intended production scheduler. |
| 11 | Relabel vertices or reorder adjacency for locality | Medium | High | Requires complete original-ID and metadata remapping and must amortize preprocessing. |
| 12 | Tune block sizes, launch bounds, architecture flags, and compiler options | Low--Medium | Low | Repeat after structural changes; tuning alone will not remove current memory and synchronization costs. |

## Recommended implementation order

### Phase 0: establish the weighted baseline

Create a benchmark matrix that actually exercises the generic algorithm:

1. low-degree, high-diameter road-like graphs with broad positive weights;
2. skewed-degree RMAT or scale-free graphs;
3. routing graphs with non-unit edge weights; and
4. routing graphs with destination vertex costs enabled and updated.

For each search, the landed opt-in telemetry records attempted and successful
relaxations, light and heavy edge examinations, light-closure rounds, queue
peaks, stale pending/frontier entries, atomic retries, and logical controller
round trips. Reached-row degree histograms, duplicate-destination rate,
absolute bucket span, device/host synchronization time, and reset time remain
to be instrumented. Report
single-source distance-only latency separately from multi-source/multi-target
path-producing routing.

### Phase 1: low-risk weighted improvements

1. **Implemented for PathFinder:** add `delta=auto` behind an explicit mode
   while preserving numeric `delta`.
2. **Implemented for immutable PathFinder graphs and the host helper:** compute
   the initial value from the active device wave size, average
   out-degree, and average **effective** edge weight. When vertex costs are
   active, low-level callers must recompute from updated host data; automatic
   workspace refresh remains future work.
3. **Controls implemented; GPU sweep pending:** test multipliers such as
   `0.25`, `0.5`, `1`, `2`, and `4` around the seed.
4. **Implemented; AMD validation pending:** use the explicit distances-only
   result mode and optional distances-only graph storage to omit parent/path
   state and the compact edge-source map.
5. Instantiate eligible 32-bit device row offsets while retaining public
   64-bit edge identities and a forced-wide A/B path.
6. **Implemented:** stop allocating legacy `pred_node` and `pred_edge` arrays
   for automatic compact-parent searches; allocate them on the first unit or
   legacy/wide run instead.

### Phase 2: reusable traversal primitives

Build one degree-aware outgoing-edge expansion layer usable by both classic
Delta-Stepping and Near/Far:

- one thread per short row;
- several lanes or one packed wave for collections of short/medium rows;
- one wave per longer row; and
- one CTA or an edge-balanced representation for extreme rows.

Use AMD's runtime wave size rather than hard-coding CUDA warp assumptions.
Choose thresholds from reached-frontier telemetry. Once lanes process edges in
a convergence-safe pattern, add wave-local destination grouping and
wave-coalesced queue reservation. Do not place wave collectives directly in the
current divergent per-thread adjacency loops.

### Phase 3: Near/Far A/B backend

Implement Near/Far behind a selectable backend flag and retain classic Delta
for comparison. The first version should include:

- current and next urgent/near-near queues;
- a near-far queue and a far queue;
- lazy validation against the current distance so stale entries can be skipped
  without requiring unique membership in every queue;
- multiple queue subpartitions to reduce tail contention;
- a CU-scaled cap on urgent work so a very large near set spills safely to less
  urgent queues; and
- original edge IDs and optional predecessor output.

Tune the number of subpartitions and the urgent-work cap on AMD hardware rather
than copying cuGraph's CUDA constants unchanged. Use effective weights for all
threshold decisions when destination vertex costs are enabled.

### Phase 4: contention and state reduction

Measure the candidate-to-unique-destination ratio after degree-aware expansion.
Use wave-local reduction unconditionally only when it wins. Add block-level or
global radix sort/reduce by destination for large, collision-heavy frontiers;
small or low-collision frontiers should retain the direct atomic path.

Fold queue-state changes into the selected scheduler rather than optimizing
the old flat membership arrays twice. Evaluate packed membership, per-query
epochs, or queue entries tagged with the distance/generation that created them.
The goal is to remove scattered reset writes without adding more cost to the
relaxation hot path.

### Phase 5: device-resident control and query batching

After the queue design is stable, remove per-round and per-bucket host
decisions. Preserve the gfx1151 explicit-stream correctness gate until a new
controller has been stress-tested on concurrent streams. HIP Graph capture is
useful only after the captured sequence no longer depends on scalar D2H values.

Batch multiple searches inside one launch only after the single-query backend
is competitive. Give every query independent distance/parent epochs and queue
state, and schedule work by available edges rather than assigning one static
block group to each query.

## Deprioritized alternatives

- Do not implement circular buckets before Near/Far unless maintaining a
  textbook Delta-Stepping scheduler is itself a requirement.
- Do not prepartition light/heavy edges when destination costs can change their
  effective class. A fixed-weight specialization may still benefit.
- Do not copy cuGraph's global destination sort/reduce into every frontier.
  Low-degree routing graphs may spend more on sorting than they save in atomics.
- Do not treat HIP Graph capture as a substitute for eliminating host-dependent
  control.
- Direction-optimizing BFS, bidirectional search, and A* are separate routing
  algorithms, not generic full-SSSP optimizations.
- Multi-GPU work follows a competitive single-GPU implementation.

## Correctness invariants

All backends must preserve:

1. outgoing CSR orientation and original public 64-bit edge identity;
2. finite nonnegative edge weights and destination-cost multiplication;
3. monotone strict distance decreases and deterministic valid parent ties;
4. correct zero-weight edges, parallel edges, self-loops, and zero-weight SCCs;
5. target settlement only when no more urgent work can lower that target;
6. duplicate source and target semantics;
7. exact paths rooted at one of the requested sources;
8. safe workspace reuse after convergence, early target termination,
   iteration limits, callbacks, and exceptions;
9. null-stream and concurrent explicit-stream ordering on gfx1151; and
10. a wide-offset fallback when rows or edges do not fit the compact form.

## Acceptance gates

Every optimization must pass CPU-Dijkstra distance comparison and independent
path validation. Test frontier sizes around wave and block boundaries, zero to
extreme degree, all-light/all-heavy/mixed rows, weights immediately around
thresholds, unreachable targets, terminal distances, repeated workspaces, and
concurrent streams.

Report graph upload, traversal, path extraction, reset, and end-to-end routing
separately. Use warm-ups, repeated medians with dispersion, identical graph and
query order, fixed clocks when possible, and explicit compiler/architecture
records. Compare the classic and Near/Far backends in both distances-only and
path-producing modes.
