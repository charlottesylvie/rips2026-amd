# Bellman-Ford routing memory

This file is the durable working memory for Bellman-Ford integration in
`CongestionFreeRouting`. Keep design decisions and integration notes above the
file inventory.

There is no background watcher for this directory. Update this file manually
whenever a source file is added or modified under
`CongestionFreeRouting/bellman_ford`. The inventory below intentionally omits
`MEMORY.md` itself to avoid recursive bookkeeping churn.

## Current branch context

- Reviewed branch: `parallel-router-optimization` at `7e44acd80b24` on
  2026-07-17.
- End-to-end entry point: `CongestionFreeRouting/pathfinder_router.cpp`, built
  as `PathFinderFile`.
- Pipeline: `interchange_to_csr` -> `pathfinder` -> `routes_to_phys`.
- The Makefile selects and invokes `PathFinderFile`; it does not build the four
  pipeline executables. `PATHFINDER_PROFILE_COMMAND` wraps only the inner
  `pathfinder` process.
- `interchange_to_csr.cpp` emits outgoing CSR: row `u` contains every edge
  `u -> v`. The graph and metadata orientation tag is `2`.
- The current routing graph has exact unit edge weights. Unit BFS remains the
  overall default SSSP engine; bf10 is selected for the Bellman-Ford engine.

## Bellman-Ford integration contract

- `bf10.cpp` is the active Pathfinder Bellman-Ford implementation. `bf9.cpp`
  and `bf9.hpp` remain byte-for-byte unchanged as the immediate A/B baseline.
- The reusable `bf10.hpp` interface fits the workspace call used by `route_net`:
  `run(sources, targets, delta, max_iters, stream, callback, user_data)`.
- `BellmanFord10CsrGraph` validates and uploads one immutable outgoing CSR. It
  is shared by independent stream-affine `BellmanFord10CsrWorkspace`
  instances; each workspace owns and reuses its mutable frontier and compact
  reconstruction allocations. Distinct BF10 public names and the
  `rips_sssp_bf10` namespace permit BF9/BF10 side-by-side test linkage.
- Multi-source routing stable-deduplicates sources and invokes the single-source
  bf10 implementation once per source. The `delta` argument is accepted for
  Pathfinder interface compatibility and does not affect Bellman-Ford.
- Results use Pathfinder's compact target arrays, preserving target order and
  duplicates. Equal-distance candidates prefer an identity path, then the
  lower source node id, then packed predecessor-edge ordering. A deterministic
  host CSR-order tight-edge fallback repairs zero-weight predecessor cycles.
- CSR positions are the physical edge IDs. They remain aligned with
  `rowptr`/`colind` so metadata can map every selected edge back to a PIP.
- `pathfinder` selects bf10 with `--sssp-engine bellman-ford` or `bf10`.
  Existing `bf8` and `bf9` spellings remain compatibility aliases for the
  current Bellman-Ford backend; direct A/B tests instantiate the distinct BF9
  and BF10 adapter classes.
- Defining `BF10_NO_MAIN` removes only the standalone bf10 entry point. Library
  operation does not parse files, print source progress, or write CSV/JSONL.
- Normal builds compile out BF10 ROCTx ranges unless
  `PATHFINDER_ENABLE_ROCTX` is defined.
- `bf6.cpp` exists on branch `sssp-comparisons`, not in this worktree. Its
  standalone `RIPSOCS1`/metadata-v3 and one-source interfaces are not drop-in
  compatible with the current `RIPSCSR1`/metadata-v4 Pathfinder flow.

## BF10 controller and reconstruction design

BF10 was copied from the final BF9 implementation before being symbol-isolated.
It does not change `pack_state_bits`, `pack_state_float`, `atomic_relax_state`,
`relax_outgoing_edge`, or the separate BF9 files.

### GPU-controlled Bellman-Ford

- On devices reporting cooperative-launch support, a resident 1-D cooperative
  grid owns the complete single-source iteration loop. All resident threads
  grid-stride over state initialization, frontier relaxation, and target
  status; thread zero performs the scalar reset and exact BF9 decision order
  between grid-wide barriers.
- The controller preserves active-frontier deduplication, packed atomic edge-ID
  ties, `mark_token = completed_iterations + 1`, empty-frontier convergence,
  the exact target-distance bound, and maximum-iteration status. It writes one
  `ControllerResult`, copied once to pinned host memory per source.
- Occupancy probing proves the selected grid is resident. BF10 currently caps
  the grid at one block per compute unit so explicit PathFinder worker streams
  retain occupancy headroom. Unsupported capability, zero legal occupancy, or
  requested frontier tracing selects a BF10-local copy of the host-controlled
  BF9 loop. Actual cooperative launch/runtime errors are reported rather than
  silently rerunning a partially mutated source.
- Compatibility-loop iterations increment the host-status-copy counter and use
  the reusable pinned status/event. Optimized cooperative runs leave that
  counter at zero. `bf10.gpu_controller` and `bf10.controller_fallback` mark
  the two paths.

### GPU-resident deterministic reconstruction

- The shared device graph adds `from_for_edge[edge]` (`4E` bytes), avoiding a
  row search during reconstruction. Pass 1 assigns one thread per improving
  unique target, follows packed predecessor edges, validates the sentinel,
  edge range, destination, and predecessor vertex, and records 64-bit node and
  edge counts. A `rows` transition guard detects a cyclic/nonterminating chain.
- A deterministic one-thread device exclusive scan assigns candidate-order
  compact node and edge slices. PathFinder sink batches are normally small;
  switch to a deterministic parallel scan if large target batches make this a
  measured bottleneck. One tiny totals copy lets the host grow compact buffers
  before pass 2.
- Pass 2 retraverses valid chains and writes backward into each fixed slice, so
  output is always `source -> ... -> target` and is independent of GPU
  scheduling. One event-scoped batch copies statuses, offsets, compact nodes,
  and 32-bit CSR edge IDs. Normal transfer/synchronization count is bounded per
  source and does not scale with predecessor depth.
- Invalid/cyclic packed chains enter `bf10.reconstruction_fallback`. BF10 makes
  at most one lazy `8R` full-state copy for that source and applies the existing
  deterministic CPU distance-tight repair only to affected candidates.
  `bf10.gpu_reconstruct` and `bf10.copy_compact_paths` mark normal work.
- Candidate and compact allocations grow geometrically, use checked arithmetic,
  query current free device memory, and preserve a safety reserve. Growth is
  persistent per stream-affine workspace; no allocation occurs per source once
  capacity is sufficient.

### Memory model

Ignoring small scalars/events, shared BF10 graph storage is approximately
`4R + 12E` bytes for weighted input or `4R + 8E` for exact-unit input: compact
row offsets, destinations, predecessor vertices, and optional weights. Each
worker retains the BF state/frontiers/marks (`20R` bytes), target buffers
(`12U` device plus `8U` pinned host), reconstruction metadata (approximately
`48C` device plus `32C` pinned host), and compact outputs on both device and
pinned host (`4N + 4M` bytes each for nodes and edges). `C` is candidate
capacity, `N` is total materialized candidate nodes, and `M` is total edges.
The path-growth counter counts metadata and compact-output growth separately.

## bf9 optimization audit (historical baseline)

The ranking is based on static hot-path analysis. No ROCm compiler, AMD GPU, or
ROCprofiler tool was available on the implementation host, so expected runtime
benefits below are hypotheses until measured on the target machine.

### Implemented optimizations

1. **Compact, specialization-aware CSR storage.** Device row offsets use
   32 bits because the packed predecessor format already requires
   `nnz < UINT32_MAX`. The redundant device `degree` and identity `edge_id`
   arrays are gone; kernels derive row ends from `rowptr[u + 1]` and use the CSR
   position as the edge ID. Generic device graph storage falls from roughly
   `12R + 16E` to `4R + 8E` bytes. On all-unit input, bf9 also omits the weight
   allocation, copy, and per-edge load, reducing it to roughly `4R + 4E` bytes.
   Host-only graph data still retains weights for reconstruction.
2. **Aggregate frontier-status atomics.** Each active-vertex thread reduces
   successful candidate distances locally and performs at most one global
   `atomicMin`, instead of one for every successful edge relaxation. Frontier
   deduplication retains the proven `atomicExch` sequence. The packed 64-bit
   relaxation CAS also remains intact to preserve deterministic
   distance/predecessor ordering.
3. **Targeted state transfer and batched path reconstruction.** The Pathfinder
   adapter no longer copies the `8R`-byte packed state field after every source.
   It keeps the stable-deduplicated early-stop target list resident, gathers
   `8U` bytes of target states for `U` unique targets, rejects candidates that
   cannot improve the current winner, and reconstructs each improved unique
   target once. The gathered target state is also consumed as reconstruction
   depth zero, avoiding an immediate duplicate transfer. At every remaining
   predecessor depth it gathers one packed state for all active candidate paths
   together. Original duplicate-target positions share that work and are fanned
   back out only when the compact result is assembled.
   Normal D2H volume therefore scales with unique targets plus selected path
   lengths rather than graph rows times source count.
4. **Lazy zero-cycle fallback.** Targeted reconstruction validates predecessor
   sentinels, edge ranges, edge destinations, predecessor sources, repeated
   nodes, and the row-count guard. A cycle or equivalent packed-predecessor
   failure performs one lazy `8R` full-state copy for that source, shared by all
   affected candidates, then invokes the existing deterministic CSR-order
   distance-tight fallback only for those candidates. ROCTx ranges identify
   `bf9.gather_targets`, `bf9.reconstruct_paths`, and
   `bf9.full_state_fallback` separately.
5. **Narrow iteration-status waits.** Each workspace owns one pinned host
   `IterationStatus` and one reusable disable-timing HIP event. Every required
   20-byte iteration status copy is followed immediately by an event record and
   `hipEventSynchronize`; the main relaxation loop no longer calls
   `hipStreamSynchronize`. This narrows the wait to work through the recorded
   copy but does not remove the host dependency: `next_count`, graph errors,
   and target-bound fields are still required before choosing the next launch
   and termination state.
6. **Remove redundant stream drains and target uploads.** bf9 does not
   synchronize after workspace allocation, target-buffer growth, or per-source
   state initialization. Same-stream ordering protects subsequent operations.
   A multi-source adapter call uploads its unchanged target list once instead
   of once per source. The one-time graph-upload synchronization remains because
   workspaces may subsequently use other streams.

Per workspace, the compact extraction scratch is proportional to the workload:
target device storage is `4U` bytes of node IDs plus `8U` bytes of packed
states, with `8U` pinned host output; path storage is `4P + 8P` bytes on both
device and pinned host for at most `P` simultaneously active candidate paths.
The pinned iteration status is 20 bytes plus one event. A normal predecessor
depth transfers `4P` bytes H2D and `8P` bytes D2H, plus the small validation
status; only the exceptional fallback transfers `8R` bytes D2H.

### Ranked candidates not selected

7. Batch several Bellman rounds with device-resident frontier counts. This can
   eliminate most 20-byte status copies and per-round host synchronizations,
   but exact early-stop, error, iteration-count, and alternating-frontier
   semantics need an AMD HIP validation environment before taking that risk.
8. Sparse-reset only vertices touched by the previous source. It can avoid the
   current `O(R)` state initialization after short searches, but costs a
   `4R`-byte queue and an extra atomic append per newly reached vertex; an
   adaptive dense fallback needs profiling to show it wins on routing graphs.
9. Move the predecessor walk itself onto the GPU. The current compact approach
   removes full-state traffic but still has one host/device dependency per path
   depth. A device-resident walk could remove those waits, but emitting variable
   compact paths and preserving deterministic cycle fallback would require a
   more invasive design.
10. Wave-level high-degree row processing, queue reservation aggregation, and
   additional Pathfinder workers. These can improve load balance or throughput,
   but need degree-distribution, atomic-stall, memory-capacity, and worker-sweep
   evidence. Automatic Bellman-Ford selection therefore remains one worker;
   explicit `--parallel-net-workers N` is still supported.

## Verification status

- CPU-only Pathfinder integration and alias coverage can run with the fake HIP
  runtime.
- The BF10 adapter regression covers weighted/unit graphs, duplicate and
  multiple endpoints, identity/unreachable targets, deterministic source and
  edge ties, iteration bounds, distance early stop, long batched paths,
  persistent growth/reuse, malformed input, explicit parallel workspaces,
  forced controller fallback, optimization counters, and the zero-weight-cycle
  fallback. It passes under the local sequential fake-HIP runtime with
  AddressSanitizer and UndefinedBehaviorSanitizer enabled.
- A side-by-side fake-HIP BF9/BF10 build links both isolated adapters and
  produces exact matching distances, source choices, compact offsets, nodes,
  edges, and status on the weighted regression graph. The dedicated production
  HIP comparison test also checks BF9/BF10 exact results and delta-stepping
  distances on weighted and unit graphs.
- The BF10-specific Pathfinder CPU-stub test passes for `bellman-ford`, `bf10`,
  and legacy `bf9` engine spellings. The BF10 standalone translation unit and
  `BF10_NO_MAIN` library mode both pass local host compilation; side-by-side
  BF9/BF10 linkage has no duplicate public or internal symbol.
- The bf9 HIP adapter test covers weighted and unit graphs, duplicate sources
  and targets, identity/unreachable paths, strict and tied source replacement,
  long and simultaneously reconstructed paths, zero-weight predecessor cycles,
  iteration limits, distance-bound stopping, target/path buffer growth and
  reuse, shared graphs, and concurrent stream-affine workspaces. Internal
  regression hooks assert zero full-state copies for normal cases and exactly
  one copy for a source whose affected targets require the cycle fallback.
- The bf9/delta comparison covers weighted and exact-unit graphs and validates
  every compact path against the CSR.
- Host-side fake-HIP syntax/link checking and the BF10 Pathfinder CPU-stub test
  pass on the implementation host. Production HIP builds, cooperative-kernel
  execution, GPU comparison tests, route conversion, four-worker profiler
  traces, and speedups remain to be measured on an AMD ROCm system; no `hipcc`,
  AMD GPU, `rocprofv3`, generated `xcvu3p.device`, or saved benchmark CSR is
  available locally.

## Manual file inventory

Last manual update: 2026-07-17T18:47:52Z

### Code inventory

| File | Bytes | Modified (UTC) | SHA-256 |
| --- | ---: | --- | --- |
| `bf8.cpp` | 107645 | 2026-07-16T21:12:11Z | `d9cf1b0b9d84f914354d3d53a777f9d225929d8a115a5f30e4a41fb5f37bcc74` |
| `bf8.hpp` | 3499 | 2026-07-16T21:10:49Z | `3951ec66391659487ec38f90fabb72d53ec7c49b6002422e0a315216b6f173b1` |
| `bf9.cpp` | 130948 | 2026-07-17T17:27:25Z | `a9c1ae67719dd8579846c5d0904055028851103ba09cbaadc18e80174f624a8a` |
| `bf9.hpp` | 3431 | 2026-07-17T16:12:12Z | `5d39191c44880058a3e6291ef7ebb9d100af891a96e8a9a4ccea46add1b8a862` |
| `bf10.cpp` | 182185 | 2026-07-17T18:46:51Z | `02e263b3bf85f2b10e3b346eead0c7ed81fbd7fd1221071327fd8b8d9a12989e` |
| `bf10.hpp` | 3554 | 2026-07-17T18:43:06Z | `3cca513e79f2a92135c6a162455db5bac8f33079b4aa162481cfb06eb45d4310` |

### Recent manual updates

- 2026-07-16T18:39:05Z: Removed `.memory_watch.py`; memory tracking is now
  manual only.
- 2026-07-16T18:39:05Z: Refreshed `bf8.cpp` inventory after adapting it to
  the `interchange_to_csr.cpp` `RIPSCSR1` CSR and metadata-v4 format.
- 2026-07-16T21:12:21Z: Added the reusable `bf8.hpp` graph/workspace interface,
  repeated-single-source Pathfinder adapter, compact target result construction,
  shared graph ownership, and zero-weight predecessor-cycle path fallback.
- 2026-07-16T21:12:21Z: Refreshed the manual inventory to match the directory;
  removed the stale `csr_outgoing.cpp` entry because that file is not present in
  this worktree.
- 2026-07-17T16:12:29Z: Added bf9 as the active Pathfinder implementation with
  compact/specialized graph storage, aggregated status atomics, redundant
  synchronization/target-upload removal, optional ROCTx ranges, and dedicated
  HIP regression tests.
- 2026-07-17T17:28:40Z: Replaced per-source full-state transfers in the
  Pathfinder adapter with compact target gathering, target-depth reuse, batched
  predecessor retrieval, and a one-copy lazy zero-cycle fallback. Replaced the
  per-round stream drain with a reusable pinned status buffer and disable-timing
  event, added transfer/growth regression counters, and expanded adapter tests.
- 2026-07-17T18:47:52Z: Forked BF10 from the unchanged BF9 implementation,
  isolated all public/internal symbols, added cooperative GPU iteration control
  with an instrumented host compatibility loop, uploaded immutable edge-source
  IDs, replaced predecessor-depth gathers with deterministic two-pass compact
  GPU reconstruction, retained the one-copy distance-tight fallback, and added
  persistent checked/reserved reconstruction storage plus BF10 regression
  instrumentation.
