# UnitBFS Optimization Roadmap

Updated 2026-07-24 for the bounded optimization pass. The three changes marked
implemented below are host-tested but not compiled or run with HIP.

## Scope

UnitBFS is the production specialization for exact unit-weight routing graphs.
It remains a first-class optimization target alongside classic
Delta-Stepping. The speed ranges below are unmeasured engineering estimates,
are workload-dependent, and must not be added.

## Verified current implementation

- `UnitBfsCsrGraph` validates the entire CSR and rejects every edge weight that
  is not exactly `1.0f`. The earlier roadmap claim that only diagnostics
  enforced this contract was stale.
- The immutable outgoing CSR is shared across private stream-affine
  workspaces.
- Row offsets and predecessor edge IDs use signed 32-bit storage when
  `nnz <= INT32_MAX`; an explicit forced-64-bit mode exists for correctness and
  performance comparisons. Public path edge IDs remain 64-bit.
- Each workspace allocates level, predecessor node, predecessor edge, and one
  append-only frontier/visited queue. The visited prefix is also the sparse
  level-reset list.
- Source claims use an authoritative atomic CAS. A successful claim reserves
  one queue slot and stores the original outgoing edge ID.
- Cooperative-capable devices run up to 32 BFS levels in one resident
  grid-synchronized kernel for null and explicit streams. Grid size is capped
  for the expected 4--8 concurrent workers.
- Unsupported devices and progress callbacks retain proven fallbacks. The
  null-stream fallback batches up to four levels; the explicit-stream fallback
  checks every level on the host.
- Source, target, target-metadata, offset, and compact-path buffers grow
  geometrically and retain their high-water capacity. PathFinder supplies safe
  metadata-derived source/target hints; compact paths remain demand-sized and
  are never pre-reserved from graph size.
- Host-built compact offsets remain the default. An opt-in device two-pass path
  measures per-target lengths, deterministically scans offsets, publishes one
  totals/status descriptor, grows compact payload buffers, and fills validated
  paths without the host prefix sum or two H2D offset copies.
- Sparse level reset remains the default. An opt-in packed generation/level
  representation makes prior-query entries stale without resetting each
  visited level and performs a synchronized full reset before token reuse.

## Evidence boundary

The HIP regression source covers compact/wide rows, host/device extraction,
sparse/generation visitation, exact-unit rejection, depth limits, callback
abort/reuse, duplicate and unreachable targets, capacity high-water retention,
wide/deep frontiers, four- and eight-worker explicit streams, cooperative
control when supported, and device path validation.

That suite and the production UnitBFS translation unit have not been compiled
or executed on an AMD GPU in this checkout. `hipcc` is unavailable locally.
The shared-capacity and UnitBFS policy/model tests, both fake-HIP PathFinder
suites, and ASan+UBSan variants pass; these do not validate HIP syntax or
device behavior. The historical observation that 4 and 8 workers were fastest
predates later correctness guards and is not a reliable post-cooperative
baseline. No exact profiler-free samples for the current commit are stored in
the repository.

## Required gate before performance changes

1. Run the full HIP suite across host/device extraction and sparse/generation
   visitation in automatic-compact and forced-wide row modes.
2. Repeat the explicit-stream stress suite with the default and fully opt-in
   paths, increasing
   `UNIT_BFS_REUSE_STRESS_RUNS` enough to cover more reused queries than the
   production failure point.
3. Rerun `logicnets_jscl` and require critical path 214, valid routed output,
   identical complete route trees across worker counts, and no GPU faults.
4. Record profiler-free medians, dispersion, and peak memory with four workers
   as the controlled future UnitBFS performance baseline. Other worker counts
   remain diagnostic sweeps, not portable defaults.
5. Profile setup, cooperative traversal, status copies, sparse reset, target
   measurement, compact fill, and payload copies separately.

## Optimization status and remaining ranking

The ranking is by expected production routing speed benefit.

| Rank | Optimization | Status | Expected speed improvement | Difficulty | Rationale |
| ---: | --- | --- | --- | --- | --- |
| 1 | Batch several independent net searches inside one GPU submission | Not implemented | 10--40% throughput when per-net frontiers are small | Very high | The router issues thousands of short searches through separate host workers. A batched work queue can amortize launch/status overhead and schedule by available frontier edges, but every query needs isolated level, parent, target, and output state. |
| 2 | Keep compact-offset construction on the GPU longer | Implemented, opt-in, HIP-unvalidated | 5--20% end-to-end | High | The two-pass device path removes the host scan and two H2D copies while preserving the guarded totals/offset/metadata transfers required by gfx1151. Host-offset extraction remains default. |
| 3 | Tune cooperative level budget and residency share | Not implemented | 5--20% on launch-bound searches | Medium | The constants 32 levels, target four workers, and maximum eight workers are policy choices rather than measured optima. Sweep them against route depth and worker count while checking watchdog and occupancy behavior. |
| 4 | Reuse a degree-aware outgoing-edge expander | Not implemented | 0--15% on the routing graph; up to 25% if reached rows are skewed | Medium--High after the Delta primitive exists | Thread-per-row is efficient for short rows. Add lane/wave/CTA cooperation only for measured long-row buckets and retain the existing direct path. |
| 5 | Pre-reserve and geometrically retain query capacity | Implemented, enabled, HIP-unvalidated | 2--8% | Low | Metadata-derived source/target reservations and geometric high-water growth are active. Compact paths deliberately remain demand-sized rather than using unsafe graph-sized estimates. |
| 6 | Generation-stamped visitation | Implemented, opt-in, HIP-unvalidated | 3--10% when many vertices are reached | Medium | Packed generation/level claims preserve atomic first discovery and safely reset on rollover. Sparse reset remains default until AMD validation. |
| 7 | Reorder vertices or adjacency for locality while preserving original IDs | Not implemented | 0--20% | High | This may improve row and frontier locality, but conversion, metadata, edge identity, and route reconstruction all need stable remapping. Require cache evidence and preprocessing amortization first. |
| 8 | Tune block size, launch bounds, architecture flags, and compiler options | Not implemented | 0--10% | Low | Repeat after the controller and shared expansion paths stabilize. |

## Design notes for the top two items

### Multi-query batching

Do not merely assign a fixed block subset to each query. Nets have widely
different frontier widths and depths, so a device scheduler should pull ready
frontier ranges by edge volume. Each query still needs:

- a unique visitation generation or disjoint state slice;
- independent source/target counts and stopping state;
- deterministic original edge parents;
- bounded queue/output capacity with an explicit spill/error path; and
- exact association of compact results with the originating net.

Start with a small fixed batch and compare it with the best 4- or 8-workspace
baseline at the same memory budget.

### Device-side extraction control

The default path still copies target metadata, builds prefix offsets on the
host, and copies two offset arrays back. The implemented opt-in path measures
paths, performs a deterministic one-thread device scan, publishes one small
totals/status record, then fills and validates compact payloads. It preserves
query epochs, target order, duplicate and unreachable targets, predecessor
levels, original outgoing edges, and source roots. AMD validation must compare
its exact results and guarded control transfers with the default path before it
can be enabled.

## Shared classic-Delta work

These improvements can be shared without changing UnitBFS semantics:

- degree-aware and edge-balanced outgoing-edge expansion;
- convergent wave-local queue reservation;
- metadata-driven buffer capacity management;
- optional epoch infrastructure;
- 32/64-bit offset A/B conventions; and
- reached-degree, frontier-size, reset, controller, and extraction telemetry.

Weighted-only Delta features--bucket-width selection, light/heavy
classification, pending-bucket structures, and weighted target settlement--do
not belong in UnitBFS.

## Correctness and measurement contract

Test frontiers below, at, and above AMD wave and block sizes; growing and
exhausted cooperative launches; depth caps around 32-level boundaries;
duplicate/source/unreachable targets; callbacks; repeated workspaces;
concurrent explicit streams; compact/wide offsets; and capacity growth. Every
returned edge must belong to the predecessor's original outgoing CSR row.

Report converted UnitBFS/PathFinder time, traversal, controller/status, reset,
path extraction, full `PathFinderFile` wall time, and peak memory separately.
Use a warm-up, repeated medians with dispersion, fixed graphs and net order,
fixed clocks when possible, exact route-tree comparison, and checker/analyzer
validation.
