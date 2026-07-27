# BF10 optimizations not yet fully integrated into Delta-Stepping

**Audit date:** 2026-07-24  
**Source snapshot:** `c35b824`  
**Scope:** `CongestionFreeRouting/bellman_ford/bf10.cpp`,
`CongestionFreeRouting/bellman_ford/MEMORY.md`,
`CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp`, and their
Pathfinder integration.

## Executive summary

The most promising BF10 ideas for Delta-Stepping are:

1. moving generic bucket and light-closure control onto the GPU;
2. using eligible 32-bit row offsets and internal edge identifiers;
3. omitting the device weight array for exact-unit graphs; and
4. adopting BF10's checked, transactional, headroom-aware buffer growth.

The first three can materially reduce controller overhead or memory traffic.
The fourth is primarily a robustness improvement, but can also avoid
synchronizing `hipFree` calls during buffer growth.

Several headline BF10 optimizations already have Delta-specific equivalents.
Delta already has target settlement, frontier/pending membership filtering,
sparse touched-vertex reset, compact edge-parent path extraction, persistent
shared graph/workspace storage, and true simultaneous multi-source traversal.
Those are not ranked as missing work.

The rankings below are qualitative. They should be validated on the target AMD
GPU with the current code because the retained 2026-07-17 profile predates
several Delta changes, including automatic compact edge-parent extraction.

## Rating definitions

- **Integration Difficulty** estimates the algorithmic complexity, GPU-runtime
  complexity, and validation effort needed to implement the change correctly.
- **Invasiveness** estimates how broadly the change affects data structures,
  kernel interfaces, control flow, public behavior, and the regression surface.
- The table is ordered by expected overall benefit, not by ease of
  implementation.

## Ranked improvements

| Improvement | Integration Difficulty | Invasiveness | Recommendations |
| --- | --- | --- | --- |
| **1. GPU-resident traversal controller**<br>Expected benefit: **High** | **High** — Generic Delta must coordinate light-edge closure, heavy-edge relaxation, target settlement, minimum-bucket selection, iteration limits, errors, callbacks, and reusable-workspace cleanup. The exact-unit path also has separate null-stream and explicit-stream behavior. | **Very High** — Changes the central execution loop, several kernel contracts, controller state, synchronization, fallback behavior, and multi-stream validation. | **Helpful:** Yes; this has the highest likely runtime upside when searches execute many small closure rounds or buckets. Delta still copies frontier counts and bucket decisions to the host, and explicit worker streams contain additional synchronization required by observed `gfx1151` behavior. Implement a device state machine or cooperative persistent kernel first for the exact-unit/default-stream path, then for generic Delta. Preserve a host-controlled fallback for callbacks, unsupported cooperative launch, and affected explicit-stream configurations. Cap cooperative residency so concurrent Pathfinder workers retain occupancy. Treat HIP Graph capture as a later option, not a substitute for removing host-dependent decisions. |
| **2. Eligible 32-bit device row offsets and internal edge IDs**<br>Expected benefit: **High** | **Medium–High** — Requires compact/wide graph variants or templated kernels, careful eligibility checks, conversion validation, and tests at 32-bit boundaries. | **High** — Touches device graph ownership, graph upload, nearly every CSR-consuming kernel, path buffers, memory estimates, and wide-offset fallback tests. Public `Offset` outputs can remain 64-bit. | **Helpful:** Yes, especially on the large routing graph. Delta always stores `rowptr` as 64-bit, whereas BF10 compacts it to 32-bit. Row-offset compaction alone saves approximately `4 * (R + 1)` bytes and reduces row-access bandwidth. Reuse the dual-width pattern already present in the Unit BFS implementation. Require the final CSR offset, and therefore `nnz`, to fit the chosen compact representation; keep a forced-wide A/B path. Store internal compact path edges as 32-bit when eligible, widening only while assembling the public result. |
| **3. Omit exact-unit device weights**<br>Expected benefit: **High for unit routing graphs; Low otherwise** | **Medium** — Graph construction can detect unit weights easily, but every eligible fallback path must support an implicit weight of `1.0f`. Vertex costs, forced-generic execution, finite iteration limits, callbacks, and value updates complicate the specialization boundary. | **Medium–High** — Changes graph ownership and validation plus generic, unit, reconstruction, and update dispatch. It can remain internal without changing the Pathfinder interface. | **Helpful:** Yes for the graph emitted by `interchange_to_csr`, whose edge weights are exact unit values. Delta's unit kernel avoids weight loads, but the shared graph still allocates and uploads `4E` bytes of values. Add a graph-level `unit_weights` flag and template or branch kernels on implicit weights. Do not omit values until every permitted fallback can consume implicit `1.0f`, or lazily materialize the values array on the first operation that truly requires it. Define how `update_values()` converts a unit-specialized graph to weighted storage. |
| **4. Checked, transactional, headroom-aware buffer growth and pre-reservation**<br>Expected benefit: **Medium–High for reliability; Low–Medium for steady-state speed** | **Medium** — The allocation policy is straightforward, but multi-buffer growth must provide a strong failure guarantee and account for device and pinned-host allocations together. | **Medium** — Mostly contained in buffer owners, workspace capacity helpers, graph construction, and Pathfinder worker sizing, with limited kernel impact. | **Helpful:** Yes. Delta grows buffers geometrically, but `DeviceBuffer::reset()` releases the old allocation before allocating the replacement. A failure can therefore discard reusable state, and several byte multiplications are not checked before `hipMalloc`. Allocate all replacement buffers first, validate checked additions/multiplications, query `hipMemGetInfo`, retain a reserve, and swap only after the complete growth transaction succeeds. Where Pathfinder metadata already provides source/sink limits, pre-reserve once before worker execution. This also reduces `hipFree` calls, which may impose implicit synchronization. Coordinate the local reserve with Pathfinder's global worker-memory reserve rather than counting the same headroom twice. |
| **5. Unified packed `{distance, parent_edge}` atomic state**<br>Expected benefit: **Medium but uncertain** | **High** — Delta currently relies on a 32-bit distance atomic plus separate parent publication and queue decisions based on strict decreases. Replacing that protocol with a packed 64-bit CAS must preserve monotonicity, queue insertion, equal-distance ties, zero-weight behavior, and compact/legacy modes. | **High** — Affects relaxation primitives, distance reads, parent extraction, reset logic, distance-only mode, telemetry, and most correctness tests. | **Potentially helpful:** It could remove the separate `4R` distance array and make distance/edge tie selection atomic as in BF10. However, 64-bit CAS traffic may be slower and more contended than Delta's current 32-bit distance update, particularly when many frontier edges share destinations. Prototype it behind an execution flag, retain the current representation for A/B testing, and benchmark both memory-limited and contention-heavy graphs. Do not adopt it solely because BF10 benefits from the packed representation; Delta's update pattern is different. |
| **6. Persistent pinned target and compact-path staging**<br>Expected benefit: **Medium–Low** | **Low–Medium** — The copy sequence is already batched; the main work is adding geometrically grown pinned buffers and copying from them into the public vectors. | **Medium** — Changes workspace buffer ownership, growth/error cleanup, extraction copies, and result assembly, but not shortest-path traversal. | **Helpful:** Probably, especially on discrete GPUs or when many small target batches make pageable-transfer setup visible. Delta currently copies target distances, lengths, status, sources, nodes, and edges directly into pageable `std::vector` storage and then synchronizes. Add persistent pinned staging for these arrays, grow it transactionally, enqueue the batch, wait once, and then populate public vectors. Benchmark on the target platform because pinned memory can be neutral or counterproductive on some unified-memory systems. |
| **7. Stable target deduplication with result fan-out**<br>Expected benefit: **Low–Medium and workload-dependent** | **Low–Medium** — Host-side stable deduplication and an original-to-unique index map are conventional, but target settlement counters and unit-target multiplicities must retain duplicate-result semantics. | **Medium** — Affects target upload, settlement state, path measurement/fill, result offsets, and duplicate-target tests. Traversal kernels need little or no change. | **Helpful:** Yes when nets commonly contain duplicate sinks; otherwise the gain is negligible. BF10 reconstructs each unique target once and fans the result back to duplicate positions. Delta presently measures, scans, fills, and copies duplicate target paths separately. Stable-deduplicate before upload, run settlement and extraction on unique targets, then fan distances, sources, and compact path slices back into original order. Preserve duplicate outputs exactly. BF10's per-source losing-candidate filtering should **not** be copied: Delta already performs one true multi-source search instead of one search per source. |
| **8. Device-side reconstruction prefix scan**<br>Expected benefit: **Low** | **Low–Medium** — BF10's deterministic one-thread device scan is easy to reproduce for ordinary small sink batches; a scalable parallel scan would require more work. | **Low–Medium** — Localized to reconstruction metadata, offsets, total-size transfer, and associated tests. | **Helpful:** Only if target batches are large enough for the current CPU scan and offset upload to matter. Delta already measures and fills paths on the GPU, but copies path lengths to the host, computes offsets on the CPU, and copies the offsets back. Start with a deterministic one-thread device exclusive scan, copy only total node/edge counts so buffers can grow, then launch the fill pass. Retain the CPU path for A/B testing. Do not prioritize a parallel scan until profiling shows the one-thread scan is a bottleneck. |
| **9. Event-scoped waits for remaining scalar decisions**<br>Expected benefit: **Low by itself** | **Low** — Add a reusable disable-timing event to each workspace and record it immediately after the required scalar/status copy. | **Low** — Mostly changes `copy_scalar_to_host()` and workspace lifetime management, although the explicit-stream `gfx1151` workarounds need separate validation. | **Helpful:** Slightly. BF10 waits on an event tied to its small status copy instead of draining the full stream. Delta uses pinned scalar staging but calls `hipStreamSynchronize`. Replace only true scalar-decision waits with event synchronization and leave intentionally required producer/consumer completion barriers intact until verified on `gfx1151`. The major gain will come from removing host decisions entirely; event waits are a small transitional improvement. |
| **10. Exact target-derived distance-bound early stopping**<br>Expected benefit: **Low for generic Delta** | **Medium** — A safe check must cover all remaining current, next, pending, and accumulated heavy-edge work and retain the strict `>` comparison for equal-distance tie behavior. | **Low–Medium at bucket boundaries; High inside light closure** — A boundary-only reduction is localized, while an intra-bucket check changes the hottest control path and synchronization. | **Usually not worth adding:** BF10 stops when all targets are finite and `min_next_frontier_dist > max_target_dist`. Delta's existing rule stops when all targets are settled in the completed current bucket; at that boundary, all future work lies in later buckets, so this is the bucket-granularity equivalent. Delta also now supports a caller-supplied exclusive distance limit, but Pathfinder does not dynamically derive BF10's bound from target distances. If profiling shows expensive work at the final target bucket, test an exact reduction only after light closure **and heavy relaxation**. Checking only the next light frontier is unsafe. The exact-unit path already stops on target discovery and needs no BF10-style bound. |
| **11. Lazy full-state/CSR-order repair for invalid predecessor chains**<br>Expected benefit: **None expected; defensive only** | **High** — Delta does not retain BF10's packed Bellman-Ford state, so an equivalent repair needs a different source of distances and a deterministic tight-edge search over the host or device CSR. | **Medium–High** — Adds exceptional result-recovery logic, graph/state availability requirements, more memory traffic, and a substantial edge-case test matrix. | **Do not implement proactively.** BF10 needs this fallback because equal-distance packed predecessor ties can form zero-weight cycles. Delta publishes parents only on strict distance decreases, which should prevent that failure mode, and its path kernels already validate chains. Add a lazy deterministic repair only if a reproducible valid-distance/invalid-parent case is observed. Until then, retaining a hard validation failure is simpler and avoids an unmeasured `O(R + E)` exceptional path. |

## Important optimizations that are already present or algorithmically covered

These BF10 ideas should not be entered as active Delta work:

- **Early target termination:** Generic Delta marks a target settled only after
  its bucket is closed; exact-unit traversal stops when all targets are
  discovered. The exact BF10 frontier-distance predicate is discussed
  separately in rank 10 because it is not implemented verbatim.
- **Frontier filtering and deduplication:** Delta uses `in_current`,
  `in_pending`, and `in_heavy` membership state with strict-decrease
  relaxation. BF10's frontier-status `atomicMin` aggregation does not transfer
  directly because Delta does not maintain the same next-frontier distance
  bound.
- **Sparse reset:** Delta resets only touched vertices between queries. This is
  already more specialized than BF10's dense per-source initialization.
- **GPU path measurement and fill:** Delta already performs predecessor-chain
  measurement and compact path filling on the GPU. Only the intervening prefix
  scan and pageable staging remain as partial gaps.
- **Targeted result transfer:** Pathfinder's vector-target Delta path does not
  copy the full distance/predecessor state merely to reconstruct requested
  targets.
- **Compact edge-parent mode:** Automatic eligible vector-target runs already
  store an original edge ID in the parent key and avoid predecessor-row
  materialization. The retained profile's predecessor-materialization cost
  describes an older or fallback path.
- **Shared immutable graph and reusable workspaces:** Both engines already
  support this.
- **Geometric capacity growth:** Both engines have it; the remaining Delta gap
  is checked, transactional, headroom-aware growth.
- **One target upload per query:** Delta's simultaneous multi-source run does
  not repeat target upload for each source.
- **Telemetry:** Delta now exposes opt-in execution-path, relaxation, queue,
  contention, and controller-round-trip counters. BF10-style counters are
  therefore no longer a missing feature, though additional timing telemetry
  may still help.

## Suggested implementation order

1. Establish a fresh AMD profile of current automatic compact-parent Delta.
2. Implement compact/wide row-offset variants and exact-unit weight omission,
   measuring memory capacity and throughput independently.
3. Harden all graph/workspace growth with checked transactional allocation and
   a coordinated device-memory reserve.
4. Design the GPU-resident controller behind a fallback flag and validate it
   first on one stream, then on concurrent Pathfinder worker streams.
5. Add pinned reconstruction staging and target deduplication only if current
   path-output profiles justify them.
6. Treat packed state, exact distance-bound stopping, and lazy reconstruction
   repair as experiments or defensive work, not default implementation tasks.

## Source pointers

- BF10 GPU controller and target-distance bound:
  [`bellman_ford/bf10.cpp`](bellman_ford/bf10.cpp#L713)
- BF10 compact graph and exact-unit storage:
  [`bellman_ford/bf10.cpp`](bellman_ford/bf10.cpp#L1973)
- BF10 checked, transactional reconstruction growth:
  [`bellman_ford/bf10.cpp`](bellman_ford/bf10.cpp#L3548)
- BF10 design and memory model:
  [`bellman_ford/MEMORY.md`](bellman_ford/MEMORY.md)
- Delta buffer ownership and growth:
  [`delta_stepping/delta_stepping_hip_CSR.cpp`](delta_stepping/delta_stepping_hip_CSR.cpp#L95)
- Delta shared CSR upload:
  [`delta_stepping/delta_stepping_hip_CSR.cpp`](delta_stepping/delta_stepping_hip_CSR.cpp#L2113)
- Delta compact GPU path extraction and host prefix scan:
  [`delta_stepping/delta_stepping_hip_CSR.cpp`](delta_stepping/delta_stepping_hip_CSR.cpp#L2446)
- Delta exact-unit controller:
  [`delta_stepping/delta_stepping_hip_CSR.cpp`](delta_stepping/delta_stepping_hip_CSR.cpp#L2632)
- Delta generic host-controlled loop and bucket settlement:
  [`delta_stepping/delta_stepping_hip_CSR.cpp`](delta_stepping/delta_stepping_hip_CSR.cpp#L3040)
- Current broader Delta optimization roadmap:
  [`DELTA_STEPPING_OPTIMIZATION_ROADMAP.md`](DELTA_STEPPING_OPTIMIZATION_ROADMAP.md)
