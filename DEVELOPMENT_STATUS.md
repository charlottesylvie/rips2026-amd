# GPU SSSP Development Status

Updated 2026-07-23 from the working tree after commit `c35b824`.

This is a concise snapshot of landed work, current confidence, and the next
development gates. Build instructions, public options, the routing pipeline,
and file formats belong in [README.md](README.md); measurements and detailed
design proposals are linked below rather than repeated here.

## Current objective

The primary research target is a competitive single-GPU HIP/ROCm SSSP backend
for arbitrary nonnegative weights. UnitBFS remains the production specialization
for exact-unit routing graphs, while classic Delta-Stepping is the generic
baseline from which the weighted implementation will evolve.

## Immediate correctness gate

The expanded-tree repair attempt was both about twice as slow and produced a
reported critical path of 309 instead of 214 on `logicnets_jscl`. That design
treated newly attached tree nodes as zero-distance sources and was reverted.

Current PathFinder instead performs one original-source, multi-target SSSP per
net, trims each returned source-rooted shortest path at its last existing-tree
intersection, and rejects conflicting child-parent assignments. The CPU-stub
suite now contains an explicit 214-versus-309 regression for UnitBFS and forced
generic Delta; it passes at the current commit. A fresh end-to-end AMD run on
`logicnets_jscl` has not yet confirmed the current code's analyzer result or
restored throughput. Until it does, the real multi-sink fix and its performance
remain provisional.

## Landed implementation

| Area | Current state |
| --- | --- |
| Routing adapter | One batched original-source search per net, source-attached compact paths, deterministic parent-conflict rejection, and a regression for critical-path inflation. |
| UnitBFS | Shared immutable graph; private stream-affine workspaces; compact 32-bit offsets and edge IDs when eligible with a wide fallback; append-only traversal; sparse reset; compact validated target paths. |
| UnitBFS control | Cooperative-capable devices run up to 32 levels per persistent grid-synchronized launch, including explicit worker streams. Unsupported devices retain the proven null-stream batching / explicit-stream host fallback, and progress callbacks still report every level. |
| Generic Delta | Multi-source/multi-target classic Delta-Stepping with compact original-edge parents, lazy legacy fallback, sparse reset, optional no-parent distances-only execution, and strict distances-only graph storage. |
| Delta experiments | Automatic delta seed and multiplier, explicit generic/legacy controls, deterministic synthetic weight families, opt-in telemetry, and a low-level exclusive distance bound are implemented. |
| Bellman--Ford | Retained as an active-frontier reference/fallback, not the current optimization target. |

The CPU-only PathFinder suite passes at this snapshot, including detached-path,
multi-worker adapter, telemetry, source-rooting, and critical-path-inflation
coverage. The production HIP UnitBFS and Delta suites cannot be executed on
this macOS host and still require a current AMD ROCm run.

## Known limitations

- Normal UnitBFS routing assumes exact-unit weights but does not currently
  reject a non-unit graph; only diagnostic mode checks that contract.
- The new UnitBFS cooperative controller needs a repeated AMD stress campaign
  at 1, 4, and 8 workers. Earlier explicit-stream implementations exposed
  nondeterministic validation failures and GPU faults, so one successful run is
  not sufficient evidence.
- Generic Delta still assigns one thread to each active row, processes long
  rows serially, scans mixed rows in both light and heavy phases, and scans and
  compacts one flat pending set to advance buckets.
- Generic Delta retains frequent host-visible count/controller boundaries;
  explicit streams check every light-closure round for correctness.
- Generic Delta device row offsets remain 64-bit and its per-worker
  per-vertex queues and membership arrays remain memory-intensive.
- The production graph is larger than the exact-unit Delta specialization's
  row limit, so selecting Delta there exercises the generic path.
- No current weighted AMD correctness or performance baseline exists. The
  retained profile used all-unit weights and legacy predecessor materialization
  and must be treated as historical evidence, not current performance.
- The low-level Delta distance bound remains implemented, but PathFinder no
  longer uses it after reverting expanded-tree repair.
- Interchange reconstruction limitations are tracked under
  [README caveats](README.md#known-interchange-limitations).

## Next milestones

1. Rerun `logicnets_jscl` at the current commit and require critical path 214,
   valid routed output, and comparable wall time before accepting further
   end-to-end performance changes.
2. Stress UnitBFS's cooperative path repeatedly with 1, 4, and 8 workers;
   compare complete route trees, analyzer output, and per-phase wall times, not
   just process exit status.
3. Enforce UnitBFS's exact-unit input contract at the normal dispatch boundary.
4. Establish weighted Delta correctness against CPU Dijkstra and record AMD
   baselines for both distances-only and path-producing modes.
5. Reduce generic state and reset traffic, including eligible 32-bit device row
   offsets, before increasing worker or query concurrency.
6. Build a degree-aware outgoing-edge expander and measure destination
   collision rates; add wave-local aggregation only where measurements justify
   it.
7. Prototype cuGraph-style Near/Far scheduling behind an A/B flag. Pursue
   broader destination reduction and device-resident control after the queue
   design is stable.

Multi-query launch batching, graph relabeling, and multi-GPU work remain
deferred until one weighted single-query backend is correct and competitive.

## Canonical references

- [README.md](README.md): build, CLI/API use, routing flow, tests, formats, and
  interchange caveats.
- [BENCHMARKING.md](BENCHMARKING.md): recorded measurements, reproduction
  commands, and historical profiling evidence.
- [GPU_PROFILING.md](CongestionFreeRouting/GPU_PROFILING.md): profiler workflow
  and telemetry-field definitions.
- [Generic Delta roadmap](CongestionFreeRouting/DELTA_STEPPING_OPTIMIZATION_ROADMAP.md):
  ranked weighted-SSSP implementation and acceptance plan.
- [cuGraph roadmap](CongestionFreeRouting/cuGraph-roadmap.md): source audit and
  design rationale for Near/Far, edge expansion, and candidate reduction.
- [UnitBFS roadmap](CongestionFreeRouting/UNIT_BFS_OPTIMIZATION_ROADMAP.md):
  specialization-specific maintenance and validation work.
