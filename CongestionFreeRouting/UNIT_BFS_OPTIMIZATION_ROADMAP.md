# UnitBFS Optimization Roadmap

## Scope and project priority (2026-07-23)

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
2. cooperative-capable devices run up to 32 BFS levels in one persistent,
   grid-synchronized launch before polling status. This path applies to both
   the default stream and parallel explicit streams.

The gfx1151 failures came from controller state crossing dependent kernel and
copy-engine handoffs on parallel streams. The cooperative kernel keeps every
level transition inside one dispatch with a grid-wide barrier, while retaining
the proven host-controlled path for progress callbacks or unsupported devices.
Launches are bounded to 32 levels to avoid an unbounded persistent dispatch;
unexpected launch errors remain fatal rather than being mistaken for a safe
fallback condition.

The current implementation also retains the useful baseline design: one shared
uploaded outgoing CSR, memory-aware worker creation, one append-only
frontier/visited queue, an unconditional authoritative atomic claim, sparse
reset, pinned status transfer, direct source
initialization, and compact target-path extraction with original outgoing edge
IDs.

## Measured worker result

Before the later correctness guards exposed the synchronization regression, the
`logicnets_jscl` manual worker sweep found:

- 4 and 8 workers were best and very similar;
- 2 workers were worse; and
- 1 worker was significantly worse.

Use 4 workers as the conservative baseline and retain 8 as the first comparison
point. Do not infer the same optimum for generic weighted Delta-Stepping; its
per-worker state and controller behavior are different.

## Remaining UnitBFS work

| Priority | Optimization or validation | Expected impact | Invasiveness | Disposition |
| ---: | --- | --- | --- | --- |
| 1 | Validate compact/wide offsets and the bounded cooperative controller at 1, 4, and 8 workers on the target AMD GPU | Establishes correctness and the production baseline | Low | Required maintenance |
| 2 | Reprofile 4 versus 8 workers on a fixed net set and record cooperative launch, synchronization, reset, and path extraction separately | Confirms whether the 20-second baseline is restored | Low | Required measurement |
| 3 | Reuse the generic degree-aware edge-expansion primitive | Medium only if reached rows are skewed | Medium once the shared primitive exists | Wait for generic implementation |
| 4 | Pre-reserve source, target, and compact-path capacities from route metadata | 2--8% | Low | Low-risk follow-up |
| 5 | Add generation-stamped visitation only if reset becomes material | 3--10% | Medium | Profile-gated |
| 6 | Batch independent nets in one GPU launch | High throughput upside | High | Follow generic state/scheduler work |
| 7 | Tune the implemented cooperative level budget and residency share | Medium on launch-bound searches | Medium | Profile-gated follow-up |
| 8 | Relabel vertices or reorder neighbors for locality | 0--30% | High | Defer; preserve original IDs |
| 9 | Tune block size, launch bounds, architecture, and compiler flags | 0--10% | Low | Repeat after shared primitive changes |

## Recommended order

1. Run the full UnitBFS HIP regression on the target GPU in automatic compact
   and forced-wide modes.
2. Benchmark 4 and 8 workers repeatedly with identical graph, nets, validation,
   and clocks. Use 4 as the default baseline unless 8 wins consistently by a
   meaningful margin.
3. Profile a representative 100-net run and record reached-row degree,
   per-level frontier size, cooperative launch gaps, host-poll time, reset, and
   extraction.
4. Finish the generic weighted degree-aware expansion primitive.
5. Reuse that primitive in UnitBFS only if UnitBFS telemetry predicts a win.
6. Pursue further UnitBFS-specific persistence or locality work only after the
   generic weighted roadmap reaches a competitive baseline.

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
and exhausted cooperative runs; targets discovered at varied depths; depth caps
around controller-launch boundaries; duplicate/source/unreachable targets;
repeated workspace reuse; concurrent explicit streams; forced callback
fallback; and both compact and wide offsets. Every returned edge must belong to
the predecessor's original outgoing CSR row.

Measure separately:

1. already-converted UnitBFS/PathFinder time;
2. traversal, status copies, reset, and compact extraction;
3. full `PathFinderFile` wall time; and
4. worker throughput for 4 and 8 workers.

Use a warm-up and report repeated-run medians and dispersion.
