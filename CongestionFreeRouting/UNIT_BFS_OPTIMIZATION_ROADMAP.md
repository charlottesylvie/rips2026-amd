# UnitBFS Optimization Roadmap

## Scope and project priority (2026-07-20)

UnitBFS is the specialized fast path for exact unit-weight routing graphs. It is
no longer the primary research target: new project-wide traversal work should
first improve the generic weighted SSSP backend described in
`DELTA_STEPPING_OPTIMIZATION_ROADMAP.md` and `cuGraph-roadmap.md`.

UnitBFS should remain correct, stable, and fast for its eligible inputs. New
UnitBFS-only structural work is deferred unless profiling shows that it is
material to the production workload or it naturally reuses a primitive built
for generic weighted SSSP.

Percentages below are unmeasured engineering estimates. They are
workload-dependent and are not additive.

## Implemented changes

The two selected UnitBFS-only changes are implemented:

1. eligible device row offsets and predecessor edge IDs use a compact 32-bit
   representation with a wide fallback; and
2. the default/null stream no longer performs a host round trip after every
   BFS level. It checks the first expansion and then batches four
   device-controlled levels before polling status.

The gfx1151 direct-dispatch audit found inconsistent controller state when
dependent kernels were batched on parallel explicit streams. Those streams
therefore retain a host-visible boundary after every level. This is a
correctness gate, not an optimization opportunity to remove without first
reproducing and fixing the underlying behavior.

The current implementation also retains the useful baseline design: one shared
uploaded outgoing CSR, memory-aware worker creation, one append-only
frontier/visited queue, an inexpensive visited precheck before the authoritative
atomic claim, sparse reset, pinned status transfer, direct source
initialization, and compact target-path extraction with original outgoing edge
IDs.

## Measured worker result

On `logicnets_jscl`, the manual worker sweep found:

- 4 and 8 workers were best and very similar;
- 2 workers were worse; and
- 1 worker was significantly worse.

Use 4 workers as the conservative baseline and retain 8 as the first comparison
point. Do not infer the same optimum for generic weighted Delta-Stepping; its
per-worker state and controller behavior are different.

## Remaining UnitBFS work

| Priority | Optimization or validation | Expected impact | Invasiveness | Disposition |
| ---: | --- | --- | --- | --- |
| 1 | Validate compact/wide offsets, four-level null-stream batching, and explicit-stream host checks on the target AMD GPU | Establishes correctness and the production baseline | Low | Required maintenance |
| 2 | Reprofile 4 versus 8 workers on a fixed net set and record traversal, synchronization, reset, and path extraction separately | Establishes the production worker default | Low | Required measurement |
| 3 | Reuse the generic degree-aware edge-expansion primitive | Medium only if reached rows are skewed | Medium once the shared primitive exists | Wait for generic implementation |
| 4 | Pre-reserve source, target, and compact-path capacities from route metadata | 2--8% | Low | Low-risk follow-up |
| 5 | Add generation-stamped visitation only if reset becomes material | 3--10% | Medium | Profile-gated |
| 6 | Batch independent nets in one GPU launch | High throughput upside | High | Follow generic state/scheduler work |
| 7 | Add persistent/cooperative BFS or HIP Graph capture | Medium on launch-bound searches | High | Defer until traces justify it |
| 8 | Relabel vertices or reorder neighbors for locality | 0--30% | High | Defer; preserve original IDs |
| 9 | Tune block size, launch bounds, architecture, and compiler flags | 0--10% | Low | Repeat after shared primitive changes |

## Recommended order

1. Run the full UnitBFS HIP regression on the target GPU in automatic compact
   and forced-wide modes.
2. Benchmark 4 and 8 workers repeatedly with identical graph, nets, validation,
   and clocks. Use 4 as the default baseline unless 8 wins consistently by a
   meaningful margin.
3. Profile a representative 100-net run and record reached-row degree,
   per-level frontier size, launch gaps, host-poll time, reset, and extraction.
4. Finish the generic weighted degree-aware expansion primitive.
5. Reuse that primitive in UnitBFS only if UnitBFS telemetry predicts a win.
6. Pursue UnitBFS-specific batching, persistence, or locality work only after
   the generic weighted roadmap reaches a competitive baseline.

## Shared versus specialized work

The following generic-roadmap changes can be shared with UnitBFS:

- adaptive 32-bit CSR row offsets;
- degree-aware and edge-balanced outgoing-edge expansion;
- safe wave-local queue reservation after convergent work assignment;
- query-buffer capacity management;
- optional generation/epoch infrastructure; and
- benchmark and profiling instrumentation.

The following cuGraph-inspired weighted changes do not directly belong in
UnitBFS:

- automatic delta selection;
- Near/Far distance queues;
- light/heavy edge classification;
- weighted destination-candidate reduction; and
- weighted target-settlement thresholds.

Direction-optimizing BFS, bidirectional BFS, and A*-like routing may reduce the
number of visited vertices, but they change the algorithm or result scope. Keep
them outside the competitive generic full-SSSP roadmap.

## Verification requirements

Test frontier sizes below, at, and above the AMD wave and block sizes; growing
and exhausted four-level batches; targets discovered in every batch position;
depth caps around batch boundaries; duplicate/source/unreachable targets;
repeated workspace reuse; concurrent explicit streams; null-stream batching;
and both compact and wide offsets. Every returned edge must belong to the
predecessor's original outgoing CSR row.

Measure separately:

1. already-converted UnitBFS/PathFinder time;
2. traversal, status copies, reset, and compact extraction;
3. full `PathFinderFile` wall time; and
4. worker throughput for 4 and 8 workers.

Use a warm-up and report repeated-run medians and dispersion.
