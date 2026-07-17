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
  default SSSP engine; bf9 is selected as Bellman-Ford.

## Bellman-Ford integration contract

- `bf9.cpp` is the active Pathfinder Bellman-Ford implementation. `bf8.cpp`
  and `bf8.hpp` remain unchanged as the correctness/performance baseline.
- The reusable `bf9.hpp` interface fits the workspace call used by `route_net`:
  `run(sources, targets, delta, max_iters, stream, callback, user_data)`.
- `BellmanFordCsrGraph` validates and uploads one immutable outgoing CSR. It is
  shared by independent stream-affine `BellmanFordCsrWorkspace` instances;
  each workspace owns and reuses its mutable frontier allocations.
- Multi-source routing stable-deduplicates sources and invokes the single-source
  bf9 implementation once per source. The `delta` argument is accepted for
  Pathfinder interface compatibility and does not affect Bellman-Ford.
- Results use Pathfinder's compact target arrays, preserving target order and
  duplicates. Equal-distance candidates prefer an identity path, then the
  lower source node id, then packed predecessor-edge ordering. A deterministic
  host CSR-order tight-edge fallback repairs zero-weight predecessor cycles.
- CSR positions are the physical edge IDs. They remain aligned with
  `rowptr`/`colind` so metadata can map every selected edge back to a PIP.
- `pathfinder` selects bf9 with `--sssp-engine bellman-ford` or `bf9`. `bf8`
  remains a compatibility alias, but it selects the implementation linked into
  that executable; a real A/B test requires separate bf8- and bf9-linked
  binaries.
- Defining `BF9_NO_MAIN` removes only the standalone bf9 entry point. Library
  operation does not parse files, print source progress, or write CSV/JSONL.
- Normal builds compile out the `bf9.upload_graph` and `bf9.run` ROCTx ranges.
- `bf6.cpp` exists on branch `sssp-comparisons`, not in this worktree. Its
  standalone `RIPSOCS1`/metadata-v3 and one-source interfaces are not drop-in
  compatible with the current `RIPSCSR1`/metadata-v4 Pathfinder flow.

## bf9 optimization audit

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
- The bf9 HIP adapter test covers weighted and unit graphs, duplicate sources
  and targets, identity/unreachable paths, strict and tied source replacement,
  long and simultaneously reconstructed paths, zero-weight predecessor cycles,
  iteration limits, distance-bound stopping, target/path buffer growth and
  reuse, shared graphs, and concurrent stream-affine workspaces. Internal
  regression hooks assert zero full-state copies for normal cases and exactly
  one copy for a source whose affected targets require the cycle fallback.
- The bf9/delta comparison covers weighted and exact-unit graphs and validates
  every compact path against the CSR.
- Host-side fake-HIP syntax/link checking and both Pathfinder CPU-stub tests
  pass on the implementation host. Production HIP builds, GPU tests, profiler
  traces, and speedups remain to be measured on an AMD ROCm system; no `hipcc`,
  AMD GPU, or `rocprofv3` is installed locally.

## Manual file inventory

Last manual update: 2026-07-17T17:28:40Z

### Code inventory

| File | Bytes | Modified (UTC) | SHA-256 |
| --- | ---: | --- | --- |
| `bf8.cpp` | 107645 | 2026-07-16T21:12:11Z | `d9cf1b0b9d84f914354d3d53a777f9d225929d8a115a5f30e4a41fb5f37bcc74` |
| `bf8.hpp` | 3499 | 2026-07-16T21:10:49Z | `3951ec66391659487ec38f90fabb72d53ec7c49b6002422e0a315216b6f173b1` |
| `bf9.cpp` | 130948 | 2026-07-17T17:27:25Z | `a9c1ae67719dd8579846c5d0904055028851103ba09cbaadc18e80174f624a8a` |
| `bf9.hpp` | 3431 | 2026-07-17T16:12:12Z | `5d39191c44880058a3e6291ef7ebb9d100af891a96e8a9a4ccea46add1b8a862` |

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
