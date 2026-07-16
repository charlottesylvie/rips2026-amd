# Bellman-Ford routing memory

This file is the durable working memory for Bellman-Ford integration in
`CongestionFreeRouting`. Keep design decisions and integration notes above the
file inventory.

There is no background watcher for this directory. Update this file manually
whenever a source file is added or modified under
`CongestionFreeRouting/bellman_ford`. The inventory below intentionally omits
`MEMORY.md` itself to avoid recursive bookkeeping churn.

## Current branch context

- Reviewed branch: `parallel-router-optimization` at `172afc4b0e97` on
  2026-07-16.
- End-to-end entry point: `CongestionFreeRouting/pathfinder_router.cpp`, built
  as `PathFinderFile`.
- Pipeline: `interchange_to_csr` -> `pathfinder` -> `routes_to_phys`.
- The Makefile selects and invokes `PathFinderFile`; it does not implement the
  pipeline or build its four executables.
- `interchange_to_csr.cpp` emits outgoing CSR: row `u` contains every edge
  `u -> v`. The graph and metadata orientation tag is `2`.
- The current routing graph has unit edge weights. Unit BFS is the default SSSP
  engine; delta stepping is the comparison engine.

## Bellman-Ford integration contract

- The reusable `bf8.hpp` engine fits the workspace call used by `route_net`:
  `run(sources, targets, delta, max_iters, stream, callback, user_data)`.
- `BellmanFordCsrGraph` validates and uploads one immutable outgoing CSR. It is
  shared by independent stream-affine `BellmanFordCsrWorkspace` instances;
  each workspace owns and reuses its mutable frontier allocations.
- Multi-source routing stable-deduplicates sources and invokes the protected
  single-source `bf8` implementation once per source. The `delta` argument is
  accepted for PathFinder interface compatibility and does not affect the
  Bellman-Ford computation.
- Results use PathFinder's compact target arrays, preserving target order and
  duplicates. Equal-distance candidates prefer an identity path, then the
  lower source node id, then the packed predecessor-edge result. A host-only,
  deterministic CSR-order tight-edge fallback repairs the inherited case in
  which zero-weight ties make final packed predecessors cyclic.
- CSR edge IDs must remain aligned with `rowptr`/`colind` so metadata can map
  every selected edge back to a physical PIP.
- `pathfinder` selects this engine with `--sssp-engine bellman-ford` (or `bf8`),
  while its existing route construction and per-net route-tree JSONL writer
  remain authoritative. Standalone SSSP JSONL is not accepted by
  `routes_to_phys`.
- Defining `BF8_NO_MAIN` removes only the legacy standalone entry point; library
  operation does not parse files, print per-source progress, or write CSV/JSONL.
- `bf6.cpp` exists on branch `sssp-comparisons`, not in this worktree. Its
  outgoing frontier relaxation is directionally compatible, but its standalone
  `RIPSOCS1`/metadata-v3 loader and one-source result interface are not drop-in
  compatible with the current `RIPSCSR1`/metadata-v4 PathFinder flow.

## Manual file inventory

Last manual update: 2026-07-16T21:12:21Z

### Code inventory

| File | Bytes | Modified (UTC) | SHA-256 |
| --- | ---: | --- | --- |
| `bf8.cpp` | 107645 | 2026-07-16T21:12:11Z | `d9cf1b0b9d84f914354d3d53a777f9d225929d8a115a5f30e4a41fb5f37bcc74` |
| `bf8.hpp` | 3499 | 2026-07-16T21:10:49Z | `3951ec66391659487ec38f90fabb72d53ec7c49b6002422e0a315216b6f173b1` |

### Recent manual updates

- 2026-07-16T18:39:05Z: Removed `.memory_watch.py`; memory tracking is now
  manual only.
- 2026-07-16T18:39:05Z: Refreshed `bf8.cpp` inventory after adapting it to
  the `interchange_to_csr.cpp` `RIPSCSR1` CSR and metadata-v4 format.
- 2026-07-16T21:12:21Z: Added the reusable `bf8.hpp` graph/workspace interface,
  repeated-single-source PathFinder adapter, compact target result construction,
  shared graph ownership, and zero-weight predecessor-cycle path fallback.
- 2026-07-16T21:12:21Z: Refreshed the manual inventory to match the directory;
  removed the stale `csr_outgoing.cpp` entry because that file is not present in
  this worktree.
