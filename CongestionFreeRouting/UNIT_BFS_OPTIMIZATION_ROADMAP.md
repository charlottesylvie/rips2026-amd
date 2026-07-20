# Unit-BFS Optimization Roadmap

## Audit status (2026-07-19)

The original roadmap contained 24 equally weighted optimization items. Two are
implemented: adaptive 32-bit device row offsets and predecessor-edge IDs, and
four-level device-controlled BFS batching after the first level on the
default/null stream. That is **8.3% implemented** (`2 / 24`). Both changes
retain their target-AMD validation work below; implementation status alone is
not a performance claim.

The code audit also confirmed the pre-roadmap baseline: outgoing CSR is uploaded
once and shared by memory-aware parallel workers; traversal uses one append-only
frontier/visited queue, per-successful-claim atomic reservations, a non-atomic
visited precheck before the authoritative CAS, pinned status transfers, direct
source initialization, sparse reset, and one predecessor-chain traversal for
compact path extraction. Frontier bounds, completed depth, target progress, and
the stopping condition now remain device-resident between batched status checks.

The gfx1151 direct-dispatch run exposed inconsistent controller state after
dependent kernels were batched on parallel explicit streams. Those streams now
host-check every level; four-level device batching remains enabled only when
`stream == nullptr`. This is a correctness gate, not a tuning choice to remove
without reproducing and resolving the runtime failure.

Percentages below are unmeasured engineering estimates for the unit-BFS routing
portion. They are workload-dependent and are not additive.

## Target-GPU profiling status (2026-07-17)

The new Radeon 8060S/Strix Halo hardware-counter archive profiles forced
generic, all-edges-light Delta-Stepping, not Unit-BFS. Its low cache hit rates
and state-management cost must not be copied directly into Unit-BFS estimates.
It does make the dedicated Unit-BFS baseline more important: the production
CSR has 28,226,432 rows, above Delta-Stepping's current
`2^24 = 16,777,216`-row exact-unit guard, so Unit-BFS is the only specialized
unit-weight path that can execute on the full-device graph today.

The next 100-net counter run should therefore use `PATHFINDER_SSSP_ENGINE=unit-bfs`
with one worker and the same System SOL blocks. Report frontier expansion,
sparse reset, path extraction, copies/fills, per-level synchronization, cache
hit rates, resident waves, and wait fractions separately. Follow it with
profiler-free worker sweeps; the flat worker scaling measured for generic
Delta-Stepping is not automatically transferable to Unit-BFS.

## Remaining work

| Priority | Optimization or validation | Expected improvement | Risk | Invasiveness |
| ---: | --- | ---: | ---: | ---: |
| 1 | Profile 100-net Unit-BFS, benchmark worker counts, and validate compact/wide offsets on the target AMD GPU | Establishes the real production baseline; possible 1.2--3x throughput from worker tuning | Low | Low |
| 2 | Validate default-stream four-level batching and the explicit-stream host-check gate; tune the batch/grid size | Confirms or revises the estimated 15--40% where batching is safe | Low | Low |
| 3 | Add hybrid degree-aware frontier expansion | 10--35% | Medium | High |
| 4 | Pre-reserve per-worker query buffers from route metadata | 2--8% | Low | Low |
| 5 | Batch independent nets in one GPU launch | 1.5--4x throughput | High | High |
| 6 | Add a persistent/cooperative BFS kernel | 1.3--2.5x | High | High |
| 7 | Use generation-stamped visitation | 3--10% | Medium | Medium |
| 8 | Improve small-target detection | 2--10% | Medium | Medium |
| 9 | Fuse more target path extraction | 3--12% | Medium | High |
| 10 | Capture repeated level batches in a HIP graph | 5--15% after batching | Medium | High |
| 11 | Relabel nodes or reorder CSR neighbors for locality | 0--30% | High | High |
| 12 | Add direction-optimizing BFS | 1.2--3x on wide searches | High | High |
| 13 | Add bidirectional BFS | 1.3--5x on suitable searches | High | High |
| 14 | Add a safe spatially bounded or A*-like mode | 1.5--10x on suitable nets | High | High |
| 15 | Partition independent nets across multiple GPUs | Up to near-linear throughput scaling | Medium | High |
| 16 | Tune block size, launch bounds, architecture, and compiler flags | 0--10% | Low | Low |
| 17 | Add safe per-thread or block-level buffered queue reservation | 0--10% | Medium | Medium |
| 18 | Remove duplicate CSR validation | 1--5% startup | Low | Low |
| 19 | Skip or lazily load unused node metadata | 2--15% full wrapper | Medium | Medium |
| 20 | Cache tile-coordinate parsing during conversion | 2--10% full wrapper | Low | Low |

The former separate tasks for a high-degree kernel and hybrid expansion are now
one degree-aware scheduling task. Likewise, neighbor ordering and full node
relabeling are grouped because both require the same edge-identity and metadata
remapping audit. This consolidation does not change the historical `1 / 24`
completion calculation.

## Highest-priority changes

### 1. Establish the AMD baseline

Compile and repeatedly run `tests/unit_bfs_hip_test.cpp` on the target GPU. Run
both automatic compact offsets and `UnitBfsCsrOffsetMode::kForce64Bit`, then
collect a one-worker 100-net System SOL profile and compare explicit worker
counts `1`, `2`, `4`, and `8` on the production graph. The current automatic
policy chooses at most eight workers from free memory, net count, and CPU
concurrency, but the throughput-optimal count is not known.

Record full routing time, traversal time, path extraction, launch gaps,
synchronization time, achieved occupancy, memory bandwidth, and atomic
contention. Do not claim a speedup until the target-GPU run passes and the same
graph, nets, worker count, and validation settings are compared.

### 2. Validate and tune batched BFS levels

On the default/null stream, dedicated Unit-BFS checks the first expansion
immediately, then queues four device-controlled expansions and frontier
advances before copying status. Later rounds become no-ops after an empty
frontier, complete target discovery, or `max_depth`. Progress callbacks and all
explicit streams use one host-checked level at a time. The explicit-stream gate
avoids the gfx1151 controller-state failure described above. The batched
launch grid is sized from both the visible frontier and the current device's
compute-unit count so growth inside a batch does not inherit a tiny
source-frontier grid.

The HIP regression suite covers exact `iterations_used`, `max_depth` around
batch boundaries, targets in every round, growing and exhausted frontiers,
callback equivalence, repeated workspaces, both offset widths, and original
outgoing CSR edge IDs. Run it on the target GPU, then trace a deep no-callback
query to verify traversal status polls fall from `D` to approximately
`1 + ceil((D - 1) / 4)` on the null stream and remain one per level on
parallel explicit worker streams.

### 3. Add degree-aware expansion

The current kernel assigns one thread to each frontier vertex. Profile the
degree distribution of vertices that actually reach frontiers, then use one
thread for short rows, four or eight lanes for medium rows, one wave for long
rows, and optionally one block for extreme outliers. A simple first version can
send only high-degree vertices to a secondary cooperative kernel.

### 4. Pre-reserve query storage

Source, target, and compact path buffers currently grow exactly on demand;
`hipMalloc`/`hipFree` may synchronize. PathFinder already knows maximum source
and sink counts from route metadata, so reserve those capacities before worker
threads start. Grow path buffers geometrically rather than allocating the
worst-case `targets * vertices` size.

## Conditional follow-ups

- Add HIP graph capture only if batched traces remain launch-bound.
- Attempt true multi-net batching only if explicit-stream workers, including
  their required host checks, still underfill the GPU.
- Consider a persistent kernel only after level batching is measured.
- Pursue direction-optimizing, bidirectional, or spatial search only when
  profiling shows that reducing explored vertices is necessary.
- Use generation stamps only if sparse-reset time is material.
- Change target detection or path extraction only if those stages are visible
  in profiles.

Node relabeling, neighbor reordering, and spatial pruning must preserve the
mapping to original outgoing 64-bit CSR edge IDs. Spatial bounds must have an
unrestricted fallback so reachable sinks cannot be incorrectly reported as
unreachable.

## End-to-end wrapper work

The converter still computes and writes six node-metadata arrays, and the
loader reads them regardless of whether the selected routing mode uses them.
Make that work optional before attempting more invasive metadata changes.
Coordinate parsing can also be cached per tile instead of repeatedly parsing
tile names. Finally, validation currently occurs at the file boundary,
PathFinder entry, and shared unit-BFS graph construction; a trusted internal
constructor can remove the final repeated `O(V+E)` scan while retaining
boundary validation.

## Verification requirements

Every traversal change must test frontier sizes below, at, and above the AMD
wave size; skewed degrees; duplicate sources and targets; source targets;
unreachable targets; depth limits around termination; repeated workspace calls;
concurrent explicit-stream workers with per-level host checks; null-stream
four-level batches; and that every returned edge belongs to the predecessor's
original outgoing CSR row.

Measure separately:

1. already-converted unit-BFS/PathFinder time;
2. full `PathFinderFile` wall time; and
3. kernel, copy, and synchronization time from ROCm profiling.

Use a warm-up and report the median and dispersion of repeated runs.
