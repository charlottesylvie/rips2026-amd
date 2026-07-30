# GPU SSSP Development Status

Updated 2026-07-30 for the bounded UnitBFS/classic-Delta optimization pass,
the experimental multi-query fused-host-checked and reduced-round-trip Delta
controllers, and the FPGA Interchange import correctness audit.
All GPU changes below are implemented but HIP-unvalidated.

This file is the concise source of truth for landed work, confidence, and the
next development gates. Build instructions, public options, the routing
pipeline, and file formats belong in [README.md](README.md). Historical
measurements and reproduction commands belong in
[BENCHMARKING.md](BENCHMARKING.md), while profiler procedures and Delta
telemetry definitions belong in
[GPU_PROFILING.md](CongestionFreeRouting/GPU_PROFILING.md).

## Current objective and algorithm scope

The production graph has exact unit weights, so UnitBFS is the default and the
most relevant routing backend. Classic Delta-Stepping is the retained backend
for arbitrary nonnegative weights and for controlled comparisons on the same
CSR. Bellman--Ford/BF10 remains a reference and fallback.

The optimization scope is UnitBFS and classic Delta-Stepping. A separate
Near/Far scheduler is not planned. Reusable ideas such as degree-aware edge
expansion, destination aggregation, compact state, and device-resident control
remain in scope when they improve either of the two retained algorithms.

## Current routing correctness gate

PathFinder performs one original-source, multi-target SSSP per net, trims each
source-rooted result at its last existing-tree intersection, and rejects
conflicting child-parent assignments. The reverted expanded-tree repair was
about twice as slow and reported a critical path of 309 instead of 214 on
`logicnets_jscl` because newly attached tree nodes were incorrectly treated as
zero-distance sources.

The CPU-stub regression contains the 214-versus-309 case and passes at this
snapshot for UnitBFS and forced-generic Delta. A fresh end-to-end AMD run has
not yet confirmed critical path 214, routed-output validity, or restored
throughput, so the real multi-sink result remains provisional.

## Implementation verified in the working tree

| Area | Current implementation |
| --- | --- |
| Routing adapter | One batched original-source search per net, compact source-rooted paths, last-tree-intersection trimming, deterministic parent-conflict rejection, and a critical-path-inflation regression. |
| Interchange import | Truly source-less OOC signals are preserved but excluded from route completion. RapidWright's exact `GLOBAL_USEDNET` sentinel and GND/VCC physical nets are supported preservation-only state: their represented resources are blocked and their original physical-net records remain structurally unchanged, without creating UnitBFS/Delta work. Unsupported partial/shape signal work is preserved and reported, then rejected by default unless diagnostic-only opt-in is explicit. Fixed routes reserve pins, stub nodes, both PIP endpoints, and xcvu3p's paired static SLICEL/SLICEM outputs. Device-graph v3 uses active typed alternate-site mappings, validates the physical part, rejects ambiguous lookups, duplicate physical names, and cross-net endpoint ownership, excludes unserializable pseudo-PIPs, makes non-source sinks terminal, and removes incoming edges to exclusive sources while retaining their outgoing rows. Source/stub analysis retains scratch high-water capacity, and the blocked/exclusive destination union restores one destination-mask read per filtered edge. The one-time preprocessor avoids duplicate device-string/wire lookup storage and compacts row offsets in place. Logical name-only links are emitted only when globally unambiguous. New CSR v2/metadata v6 outputs embed one 128-bit pair ID, publish it in the generation sidecar, and propagate it to routes; guarded readers reject mixed or stale artifacts. Metadata v6 retains declared counts while omitting seven unused device-wide node arrays (40 bytes per node); v4/v5 remain readable. |
| UnitBFS graph | Shared immutable outgoing CSR. Graph construction validates every edge weight as exactly `1.0f`; normal dispatch therefore already rejects non-unit input. |
| Shared query capacity | PathFinder derives optional source/target high-water hints from exactly the routed metadata prefix, retaining duplicate endpoint counts. Checked count/byte arithmetic rejects overflow; low-level callers may omit hints. Compact paths are never reserved from graph size. |
| UnitBFS state | Private stream-affine workspaces, automatic 32-bit row/predecessor-edge offsets when `nnz <= INT32_MAX`, a forced-64-bit test mode, one append-only frontier/visited queue, geometrically retained source/target/metadata/offset/path buffers, and compact validated target paths. Sparse reset remains the default; packed generation-stamped visitation is opt-in and performs a safe full reset on rollover. |
| UnitBFS controller | Cooperative-capable devices run at most 32 levels per grid-synchronized launch on the null/default stream. Explicit worker streams always use the synchronized host-controlled level handoff required on gfx1151; unsupported devices and progress callbacks retain their fallbacks, and the null-stream noncooperative fallback can batch four levels. |
| UnitBFS extraction | Host-built offsets remain the default. An opt-in two-pass device path measures target lengths, scans deterministic offsets, publishes one totals/status descriptor, grows demand-sized compact buffers, validates paths, and copies the result without the host prefix sum or two H2D offset copies. |
| Generic Delta | Multi-source/multi-target classic Delta-Stepping with one thread per active row, separate light/heavy semantics, a flat pending set with minimum reduction and compaction, and sparse touched reset. Immutable CSR row offsets are automatically `uint32_t` only when the complete range fits; `kForce64Bit` retains the wide A/B path and public edge IDs stay 64-bit. |
| Delta controller | The established `host-checked` scalar controller remains the trusted default/reference. Experimental opt-in cooperative modes keep dependent light closure, target settlement, heavy work, pending minimum, and compaction state on-device: `fused-host-checked` publishes after one action, while `reduced-round-trip` executes a bounded configurable multi-action batch. Multiworker PathFinder runs rendezvous ready private workspaces into one bounded physical launch; its full grid processes query slots sequentially so every block crosses each grid barrier uniformly and concurrent cooperative launches are structurally impossible. Query width defaults to four, while the conservative grid cap defaults to one block per CU and is clamped to legal occupancy. Unsupported kernels and progress callbacks retain the complete scalar fallback; schema-4 telemetry separates physical batch launches from logical per-query publications. |
| Delta parents | Automatic vector-target runs use a compact 64-bit `{distance_bits, original_edge_id}` key and a shared 32-bit edge-to-source map when eligible. Legacy predecessor arrays are lazy fallback state. |
| Delta modes | Exact-unit specialization for eligible small graphs, compile-time no-parent `run_distances()`, strict distances-only graph storage, exclusive distance bounds, exact-unit automatic width `1 * multiplier`, weighted graph-aware seeding, deterministic weight families, force/controller controls, and opt-in telemetry are implemented. Boolean `in_current` plus its guarded clear remains the default; generation-tagged membership is opt-in and rollover-safe. |
| Delta capacity | Source/target hints pre-reserve only applicable state. Query and compact-path buffers retain geometric high-water capacity; compact-parent runs avoid legacy predecessor arrays, and strict distances-only storage ignores target/path hints. |
| Bellman--Ford | BF10 is wired as the Bellman--Ford engine and retained as a reference/fallback, not a current optimization target. |

## Verification completed in this audit

The following CPU/fake-HIP checks pass on the current macOS checkout with
`-Wall -Wextra -Wpedantic -Werror` where applicable:

- `pathfinder_bf10_cpu_stub_test`;
- `pathfinder_cpu_stub_test`;
- `pathfinder_router_args_test`; and
- `pathfinder_benchmark_args_test.py`;
- `sssp_query_capacity_test`;
- `unit_bfs_policy_test`;
- `delta_stepping_policy_test`;
- `device_routing_graph_test`; and
- `interchange_import_policy_test`; and
- `gzip_io_test`; and
- `pathfinder_benchmark_writer_test.py`.

The PathFinder suites cover the current adapter, engine dispatch,
automatic Delta controls, worker behavior, compact results, telemetry
aggregation, source rooting, and the critical-path regression through the fake
HIP runtime. The interchange policy suites cover exact driverless OOC
classification, unsupported-net preservation policy, both-endpoint fixed
occupancy, terminal-sink and endpoint-ownership invariants, safe pseudo-PIP
exclusion, typed alternate pin mapping, xcvu3p static-output pairing, part
matching, lookup/name-conflict rejection, path alias guards, artifact pair-ID
round trips/mismatch rejection, CSR filtering, binary round trips, duplicate
endpoint reconstruction, reached-sink attachment, integral route IDs, and
deterministic shared-node source roots without generated Cap'n Proto headers.
The device-graph suite also exhausts all 4,096 four-node combinations of
blocked, terminal-sink, and exclusive-source masks against the former
two-mask reference policy.

ASan+UBSan builds pass for both fake-HIP PathFinder suites and the host
policy/model suites with `ASAN_OPTIONS=detect_leaks=0`; this macOS ASan runtime
does not support leak detection. The Delta model covers bounded controller
publication, sticky failure precedence, target/iteration stops, batch-size-one
equivalence, callback-style abort/reuse, membership reuse, and rollover.
`git diff --check` also passes.

`hipcc`, ROCm, and an AMD GPU are unavailable on this host. Consequently the
production `unit_bfs_hip_CSR.cpp`, `delta_stepping_hip_CSR.cpp`, the linked
HIP `pathfinder` executable, the standalone `bf8.cpp`, `bf9.cpp`, and
`bf10.cpp` HIP programs, and every HIP regression translation unit that uses
those implementations remain uncompiled and unexecuted. The Bellman--Ford
files received fake-HIP preprocessing only. No production speedup or
GPU-memory reduction has been measured. The exact later-hardware
commands and acceptance checklist are in
[GPU_PROFILING.md](CongestionFreeRouting/GPU_PROFILING.md#amd-validation-checklist-for-the-bounded-controller).

The generated FPGA Interchange Cap'n Proto C++ headers/libraries and compiler
are also unavailable on this host. Therefore `device_to_routing_graph.cpp`,
`interchange_to_csr.cpp`, and `routes_to_phys.cpp` remain production-uncompiled;
their schema-facing changes received static review plus host policy/model and
Python reconstruction coverage, not a substitute compile claim.

## Remaining correctness and measurement gates

1. Rerun `logicnets_jscl` and require critical path 214, valid
   routed output, complete route-tree equivalence, and a profiler-free timing
   baseline.
2. Run UnitBFS's host/device extraction and sparse/generation visitation matrix
   in compact and wide row modes. Repeat its four-worker explicit-stream stress
   long enough to expose reuse failures rather than accepting one successful
   run. Use four workers for the future production timing baseline.
3. Establish weighted Delta correctness against CPU Dijkstra and record AMD
   baselines for host/fused/reduced controllers, compact/wide rows, and
   Boolean/generation membership in compact-parent, legacy-parent, and
   distances-only modes. Include all-light, all-heavy, mixed, zero-weight, and
   skewed-degree cases plus callback abort/reuse and explicit-stream stress.
   Use four workers for the current controlled timing baseline.
4. Reprofile current compact-parent Delta. The retained profile used all-unit
   weights and the legacy parent materialization path, so its percentages are
   historical evidence rather than a measurement of current code.
5. Collect reached-row degree histograms, per-frontier destination collision
   ratios, reset time, path-extraction time, and controller time before choosing
   collision- or degree-specific kernels.

## Known implementation limits

- Generic Delta allocates six `V`-sized queues and three `V`-sized membership
  arrays in path-producing mode. Its core mutable footprint is approximately
  48 B/V with compact parents, 60 B/V with legacy parents, and 40 B/V in
  distances-only mode.
- Generic Delta serially scans each active row, scans mixed rows in both light
  and heavy phases, scans the flat pending set to find the next bucket, and
  scans it again to compact that bucket.
- The default scalar `host-checked` explicit-stream generic Delta path checks every light-closure
  round on the host because dependent batched dispatches previously exposed
  controller-state failures on gfx1151. The experimental fused and reduced
  controllers avoid that cross-dispatch handoff only inside a cooperative,
  grid-synchronized kernel. Multiworker cooperative execution now uses one
  coordinator-owned stream and grid rather than independent worker launches;
  this redesign remains HIP-unvalidated and may select the scalar host
  fallback for ordinary capability/callback reasons.
- Default UnitBFS still has host-visible per-query setup, status,
  compact-offset, and extraction boundaries even when its inner level loop is
  cooperative. The device-offset alternative is opt-in pending AMD validation.
- UnitBFS and Delta capacity hints cover source/target-derived storage only.
  Compact paths remain demand-sized because graph-sized or otherwise
  speculative path reservation is intentionally prohibited.
- Generation-stamped UnitBFS visitation, generation-tagged Delta current
  membership, and the fused-host-checked/reduced-round-trip Delta controllers
  are opt-in. Sparse reset, Boolean/clear membership, and scalar host-checked
  control remain the enabled defaults until the AMD checklist passes.
- The production full-device graph has more than `2^24` rows, so Delta's
  exact-unit specialization is ineligible and Delta selection exercises the
  generic scheduler even though the converter emits unit weights.
- Automatic Delta is resolved once by PathFinder. Mutable low-level callers
  must recompute a numeric width after `update_values()` or
  `update_vertex_costs()`.
- Interchange reconstruction limitations remain under
  [README caveats](README.md#known-interchange-limitations).

## Optimization backlogs

The ranked, algorithm-specific backlogs and acceptance criteria are now kept
only in:

- [Classic Delta-Stepping roadmap](CongestionFreeRouting/DELTA_STEPPING_OPTIMIZATION_ROADMAP.md); and
- [UnitBFS roadmap](CongestionFreeRouting/UNIT_BFS_OPTIMIZATION_ROADMAP.md).

The former standalone cuGraph roadmap was removed because its primary proposal
was an out-of-scope Near/Far backend and its applicable traversal ideas are now
folded into the two retained algorithm roadmaps. The branch-specific
`bellman_ford/MEMORY.md` manual ledger was also removed; the durable BF10 status
is the implementation row above plus the public build and test documentation in
the README.
