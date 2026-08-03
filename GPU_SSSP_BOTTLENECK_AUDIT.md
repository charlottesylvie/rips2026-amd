# GPU Weighted-Routing Bottleneck Audit and Runtime Redesign Plan

**Date:** 2026-07-31

**Repository:** RIPS-AMD-2026
**Scope:** Production PathFinder data flow, general nonnegative routing costs, dynamic bounding boxes, Delta-stepping, BF8/BF9/BF10, older Bellman–Ford and ADDS prototypes, graph interchange/layout, profiling artifacts, tests, and benchmark infrastructure.

**Revision context:** The target is a general congestion-/timing-aware router, not an exact-unit benchmark engine. UnitBFS appears below only to interpret historical evidence; no new UnitBFS product work is recommended. Global shortest-path equivalence is not a requirement when the routing controller deliberately restricts each connection to an RWRoute-style bounding box; route legality, routability, timing/wirelength quality, congestion convergence, and end-to-end runtime are the governing outcomes.

**Implementation update:** The first bounded weighted experiment recommended by this audit is now implemented as the separate **BF11** backend. BF11 is a true-multi-source, active-frontier Bellman--Ford controller with immutable route-end/base-cost sidecars, workspace-local dynamic destination multipliers, inclusive automatic or explicit bounds, exact in-box target certification every configurable `N` rounds, and an optional one-time unbounded retry. Device-graph v4 and CSR v3 now persist representative route-end coordinates and base costs; CSR v3 also carries a destination-coordinate spatial edge permutation with a missing-coordinate spill shard. BF11 deliberately loads only the node columns today--the shard view is the prepared input for a future selected-edge/full-sweep arm. The remaining system gaps identified below (full-V reset, whole-net rather than progressive connection boxes, GPU-resident negotiated costs/commit, and real AMD timing) are still open.

## Executive decision

The fastest likely path is a **persistent, bounded, weighted routing service**, not UnitBFS and not a conventional full-device Bellman–Ford sweep:

1. Use BF10 as the shortest route to a real experiment. Extend its already persistent active-frontier controller with per-query bounding boxes, dynamic vertex/resource costs, sparse reset, route-tree sources, and a configurable “first feasible / bounded-quality / exact-in-box” stopping policy.
2. In parallel, turn the reduced-controller Delta implementation into a fully device-resident **bounded bucketed search**. The long-term engine should bucket by tentative routing cost, optionally including an RWRoute-like goal estimate. With `h=0` and strict settlement it is an exact nonnegative SSSP; with a goal estimate/coarse buckets/early acceptance it becomes the faster routing-heuristic mode.
3. Move occupancy, present/historical congestion, route ownership, and path commit/rip-up onto the GPU. Updating or copying a full V/E weight array from the CPU for every connection would recreate the existing bottleneck.
4. Route source-to-sink **connections** with tight per-connection boxes, rather than one device-wide multi-target search per net. Batch spatially disjoint or low-overlap connections so the GPU sees many useful frontiers while limiting speculative congestion conflicts.
5. Keep outgoing CSR for active-frontier expansion, but restore compact geometry and add static cost SoAs (base cost, resource length/type, delay) plus dynamic cost SoAs (occupancy/history). Add tile/cell indices and a long-wire spill structure for fast box admission.
6. Test full-edge persistent Bellman–Ford only after the box selects a genuinely compact edge set. Scanning all 125 million edges remains a hard reject; scanning a small preindexed box may be competitive and is now a legitimate experiment.

The central redesign is larger than the SSSP kernel. The current program explicitly ignores present/historical congestion and commits occupancy only after all nets have been routed. Competing with RWRoute requires an iterative rip-up/reroute controller and device-resident dynamic costs, otherwise a faster SSSP only accelerates a different problem.

### Ranked recommendation

| Priority | Action | Expected value | Confidence |
|---|---|---:|---:|
| P0 | Define and replay a general routing-cost model: congestion, history, wirelength, timing, sharing, and bbox | Prevents optimizing the wrong one-shot problem | Very high |
| P0 | Port negotiated-congestion semantics/tests to the current artifact and establish closed-loop legality/QoR baselines | Builds the router whose runtime actually matters | Very high |
| P0 | Keep occupancy/history/ownership resident and replace full-vector cost uploads with sparse GPU updates | Removes the next CPU/GPU bottleneck before creating it | Very high |
| P0 | Restore representative node coordinates and build per-connection RWRoute-style box descriptors | Enables the largest likely work reduction | Very high |
| P1 | Extend BF10 into a bounded dynamic-cost persistent prototype | Fastest causal test of minimal host/device control | High |
| P1 | Make generic Delta’s controller fully persistent and add goal-directed/coarse-bucket modes | Strongest likely long-term weighted engine | Medium-high |
| P2 | Batch spatially disjoint/low-overlap connections and schedule by selected edge volume | Adds useful parallelism with bounded routing conflicts | Medium-high |
| P2 | Add spatial cell indices, lazy/paged query state, and long-wire spill lists | Makes tight boxes reduce both work and workspace | Medium-high |
| P3 | Compare active-frontier BF, bounded bucketed search, and full-edge BF inside the same compact boxes | Lets measurements choose the per-box engine | High |

## What the production system is actually doing

### Workload and routing semantics

- The current artifact happens to contain exact unit weights, so UnitBFS is today’s default. That is an implementation snapshot, not the revised product requirement. Generic Delta already accepts nonnegative edge values and optional vertex multipliers; BF10 accepts nonnegative edge values (`DEVELOPMENT_STATUS.md:16-26`).
- The documented full graph has **28,226,432 vertices** and **125,423,075 edges**, average out-degree about **4.44**, and a disk CSR size of **1,172.25 MiB** (`BENCHMARKING.md:144-151`).
- PathFinder creates one multi-source, multi-target SSSP request for each nontrivial net, then trims each candidate path at its last intersection with the existing route tree (`CongestionFreeRouting/pathfinder.cpp:1027-1190`).
- Independent CPU workers own private full-vertex workspaces and nonblocking HIP streams (`CongestionFreeRouting/pathfinder.cpp:1497-1668`). The immutable graph is shared.
- This is a one-shot router. It does not run negotiated-congestion iterations; occupancy is accumulated after the route set has been selected (`CongestionFreeRouting/pathfinder.cpp:9-15,2743-2765`). The accepted `--present-factor`, `--present-multiplier`, `--history-factor`, and `--route-batch-size` compatibility options are parsed and ignored (`pathfinder.cpp:3100-3113`). Runtime results therefore do not yet measure a general router.

### What the RWRoute comparison changes

Current RWRoute is connection-oriented and deliberately heuristic:

- It decomposes a net into source-sink connections and computes a box from the connection source, sink, and net geometric center. Cross-SLR cases extend toward Laguna/SLL resources. Current defaults are horizontal extension 3 and vertical extension 15 INT tiles; optional enlargement increments are 1 and 2 respectively ([`Connection.java`](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/rwroute/Connection.java), [`RWRouteConfig.java`](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/rwroute/RWRouteConfig.java)).
- Expansion requires the child’s representative end coordinate to lie strictly inside that connection’s box. Bounding-box enlargement is configurable and defaults off in the current full-routing configuration; when enabled, unroutable or congested connections grow their boxes across routing iterations ([`RWRoute.java`](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/rwroute/RWRoute.java), [`RWRouteConfig.java`](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/rwroute/RWRouteConfig.java), [`RouteNode.java`](https://github.com/Xilinx/RapidWright/blob/master/src/com/xilinx/rapidwright/rwroute/RouteNode.java)).
- The priority key combines upstream congestion/base cost, wirelength, delay, and a geometric estimate to the sink. Present and historical congestion change across rip-up/reroute iterations, and connection criticality changes the coefficients.
- RWRoute marks a resource visited when it is pushed and explicitly discards a later cheaper path because its Java priority queue lacks efficient decrease-key. Therefore “globally exact SSSP” is not the competitive contract even before considering the box.

The RWRoute paper describes this connection box as a way to exclude resources that are less promising for the routing objective, not as a shortest-path certificate ([RWRoute paper](https://www.rapidwright.io/docs/_downloads/3858e01e500602e00f7d5dda065b19ec/RWRoute_final_submitted.pdf)). The correct comparison is consequently **time to a legal, congestion-free route at acceptable timing/wirelength**, not equality with an unrestricted shortest-path oracle.

### Current data flow

```text
static .devicegraph + design netlist
             |
             | CPU reads topology, skips physical node columns,
             | builds endpoint/blockage masks, compacts all retained edges
             v
       per-design .csrbin + metadata
             |
             | PathFinder loads outgoing CSR and route requests
             v
       shared GPU graph + N private V-sized workspaces
             |
             | one multi-source/multi-target SSSP per nontrivial net
             v
       GPU/host path extraction -> route-tree trimming -> JSONL -> physical netlist
```

This flow contains two different costs that should not be conflated:

1. **Offline/end-to-end graph preparation:** CPU O(E) filtering, a large per-design artifact write, then a large read/upload.
2. **Inner routing:** thousands of small SSSP searches whose controllers currently create many fine-grained runtime dependencies.

The archived inner-process trace contained 9,654 SSSP calls. Fixing only the graph upload will not fix inner traversal; fixing only traversal will not eliminate conversion and artifact duplication.

## Bottleneck diagnosis

### The evidence supports a control-granularity problem, not a bulk-transfer problem

The clearest retained trace is `CongestionFreeRouting/profiling/39155_analysis`. It profiled historical generic all-light Delta, selected through the former `--max-sssp-iters 2147483647` forcing workaround, with legacy parent handling—not current default UnitBFS and not BF10. It therefore diagnoses a mechanism rather than proving which current backend wins (`BENCHMARKING.md:5-8`; `39155_analysis/README.md:3-16`).

| Archived trace fact | Value | Interpretation |
|---|---:|---|
| SSSP calls | 9,654 | The control cost repeats thousands of times |
| Kernel launches | 278,869; 28.89/query | Very fine-grained dispatch |
| Async copies | 202,142; 20.94/query | Repeated scalar/control exchange |
| Stream synchronizations | 105,770; 10.96/query | Long dependency chain through the host |
| Copies no larger than 64 B | 195,777; 96.85% | Transaction count, not payload size, is the issue |
| D2H payload | 4.95 MiB over 144,385 calls | Extremely small transfers with disproportionate orchestration cost |
| Bulk graph H2D | 353.2 MiB in 4.632 ms | Graph upload is not the inner-loop bottleneck |
| All transfer-engine time | 0.022% of process lifetime | Not link-bandwidth saturation |
| Inner routing | 19.654 s of 20.895 s | SSSP dominates the measured process |
| Four concurrent dispatches | 88.16% of GPU span | The GPU was generally submitted work, but much of it was control/runtime work |

Sources: `CongestionFreeRouting/profiling/39155_analysis/README.md:32-107` and `summary.json:571-586`.

The target was a Radeon 8060S/gfx1151 integrated GPU with shared/coherent memory (`BENCHMARKING.md:144-158`). Calling the issue “CPU–GPU networking” is directionally useful, but the precise problem is **software submission, ordering, visibility, and synchronization around tiny operations**, not PCIe bulk bandwidth.

### The GPU is busy without using much arithmetic throughput

The retained counter replay reported, for the hot relaxation kernel:

- about 1.12% VALU-throughput proxy;
- 1.83% GL1 hit rate and 21.41% L2 hit rate;
- high wave residency;
- 99.42% wait-any cycles.

See `BENCHMARKING.md:221-290`.

That combination is consistent with sparse, irregular, dependency-heavy graph access. It does **not** mean there is a reservoir of arithmetic that can be filled profitably by scanning every edge repeatedly. Additional Bellman–Ford arithmetic would also add topology reads, distance reads, atomics/writes, and barriers—the exact resources on which graph traversal is already waiting.

### Reset and reconstruction are also material

Historical traced kernel time attributed roughly half of old execution to reset plus predecessor materialization (`BENCHMARKING.md:122-141`). In the later trace, reset was 9.00% and legacy predecessor materialization 8.65% of overlapping aggregate dispatch duration, while runtime status/copy kernels were 41.72% (`39155_analysis/README.md:92-107`).

The ten largest searches contributed about 72.4% of reset time and 69.0% of materialization time in the counter sample (`BENCHMARKING.md:284-290`). Optimizing only the median tiny search will miss much of the wall clock; telemetry and decisions must be stratified by query size/depth.

### The current working tree reinforces the ordering diagnosis

The uncommitted Delta change replaces an H2D-seeded global pending-bucket minimum with per-block device publications, a system fence, a D2H array copy, and host reduction (`CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp:2148-2195,4330-4363`). Its comment describes a stale scalar observed under concurrent nonblocking streams on gfx1151 despite stream synchronization.

That fix should be preserved and GPU-validated. Architecturally, it also shows why a long sequence of independently submitted producer/control kernels is fragile. A controller that initializes and advances its own state within one cooperative kernel has fewer visibility boundaries to get wrong.

## Algorithm audit

| Engine | Work model | Host control today | General-routing assessment |
|---|---|---|---|
| UnitBFS | Active frontier; first discovery final | Cooperative batches only on null stream; explicit streams check every level | Diagnostic control only; it cannot represent congestion/timing weights |
| Generic Delta | Cost buckets, light closure, heavy scan, pending reduction/compaction | Host-checked by default; reduced controller opt-in | Best existing foundation for exact or approximate nonnegative weighted search |
| BF10 | Active-frontier label-correcting BF | One cooperative traversal kernel plus final status when supported | Fastest path to a minimal-control bounded weighted prototype |
| Full-edge BF | Every selected edge every sweep | Naturally persistent | Reject over the full graph; credible only on a preindexed compact box |
| Bucketed A*/weighted search | Quantized `g + λh` work queues | Not implemented | Strongest likely long-term engine when route quality, not global optimality, is the contract |

### Generic Delta: promote it from comparator to weighted foundation

Generic Delta already supports nonnegative edge values and an optional V-sized vertex-cost multiplier. It exposes `update_values()` and `update_vertex_costs()` (`delta_stepping_hip_CSR.cpp:5595-5661`), although edge-value replacement is unavailable on a shared graph. These APIs prove the relaxation machinery can carry dynamic costs, but their usable update paths currently upload a complete host vector and synchronize. Calling them per connection or frequently per routing iteration would reproduce the control/transfer bottleneck.

The default controller synchronizes around light closure, target settlement, next-bucket selection, and compaction (`delta_stepping_hip_CSR.cpp:4846-5075`). The reduced-round-trip cooperative controller already keeps closure, target checks, heavy work, reduction, and compaction on-device (`CongestionFreeRouting/DELTA_STEPPING_OPTIMIZATION_ROADMAP.md:182-204`). It should become the baseline controller, then be extended from bounded batches to a full-query persistent loop.

The current generic workspace remains expensive—about 48 B/V with compact parents, 60 B/V in legacy parent mode, or 40 B/V without paths—and maintains several full-V queues/membership arrays. Bounding reduces relaxation work but not this footprint. The staged fix is:

1. generation-stamp or sparse-reset membership;
2. never copy a full dynamic-cost array from the host per query;
3. pass query coefficients and read resident static/dynamic cost SoAs during relaxation;
4. introduce cell-paged or sparse query state only after the bounded replay quantifies memory pressure.

For exact-in-box mode, retain ordinary distance buckets and the existing settlement rule. For routing mode, quantize an estimated total key `f(v)=g(v)+λh(v)` and allow coarse buckets/early target acceptance. With `λ=0` the same controller falls back to ordinary cost ordering; with a geometry/timing lookahead it behaves more like RWRoute’s best-first expansion.

### BF10: immediate bounded weighted experiment

BF10 is not a textbook whole-edge Bellman–Ford. It is an outgoing, active-frontier, label-correcting algorithm (`CongestionFreeRouting/bellman_ford/bf10.hpp:10-14`). Its cooperative kernel performs initialization, all rounds, target certification, and termination on-device (`bf10.cpp:709-867`), followed by one final controller-status transfer for traversal (`bf10.cpp:1427-1511`). Target-state gathering and path reconstruction still perform synchronized transfers afterward (`bf10.cpp:3972-3988,4083-4094,4140-4188`). A forced-host fallback checks every round and therefore provides an unusually clean negative control for pricing CPU/GPU handoff (`bf10.cpp:1196-1355`).

Its target certificate is:

```text
all targets have finite labels
AND
minimum distance in the next frontier > maximum current target distance
```

See `bf10.cpp:823-857`. For nonnegative weights, that proves no future frontier can improve a target inside the admitted graph. The strict inequality also protects equal-distance predecessor behavior when zero-weight edges are allowed.

For routing, expose three stopping policies:

- **first feasible:** stop at the first barrier after a target obtains a valid predecessor chain;
- **bounded quality:** continue until `best_target <= α × lower_bound`, or for a fixed number of cleanup rounds after first discovery;
- **exact in box:** use BF10’s current frontier certificate.

Only the third is an exact SSSP. All three can yield a legal route, and the first two should be judged by total negotiated-routing convergence rather than per-query path optimality.

BF10 currently reads immutable edge weights. Extend relaxation to compute a destination/resource cost from resident SoAs and per-query coefficients instead of rewriting E weights. It also reruns once per source and initializes roughly 12 B/V of state/marks every query (`bf10.hpp:51-61`; `bf10.cpp:709-738,4357-4447`). A routing version should seed the existing route-tree nodes together, use generation/sparse reset, and apply the connection box before reading dynamic costs.

PathFinder automatically selects one BF10 worker (`pathfinder.cpp:2717-2738`), while the cooperative grid leaves headroom for concurrent streams (`bf10.cpp:1360-1425`). Sweep 1/2/4 workers, but the more important next step is batching bounded connections with known-low spatial overlap.

### UnitBFS: historical diagnostic only

Existing UnitBFS measurements remain useful for interpreting the minimum control overhead of the current APIs. Its cooperative controller can process up to 32 levels on the null stream, whereas explicit worker streams use a host handoff per level (`unit_bfs_policy.hpp:15-23`; `unit_bfs_hip_CSR.cpp:842-919,1833-1889`).

Do not invest in or use UnitBFS to select the routing architecture. Its first-discovery semantics disappear as soon as congestion, wirelength, delay, or timing-criticality costs differ.

### BF8, BF9, old min-plus BF, and ADDS prototypes

These branches are useful historical evidence, not production candidates:

- BF8 is active-frontier BF with a host wait per round and copies a V-sized packed state to the CPU for reconstruction (`bellman_ford/bf8.cpp:578-655,2616-2669`).
- BF9 improves pinned status handling and target gathers but still waits every round; predecessor gathering adds H2D/D2H traffic per path depth (`bellman_ford/bf9.cpp:787-885,2995-3043`).
- The old `HIP_kernel/bellman_ford` code expects incoming CSR and performs sparse min-plus merge/allocation work each iteration (`HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp:51-64`; `bf_hip_CSR_device_utils.hpp:332-481`).
- The old Delta implementation is incoming-oriented, block-per-row, state-heavy, and host-controlled. Production Delta supersedes it.
- `Charlotte_oneshot` scans every edge and synchronizes every iteration; its ADDS-lite queue defaults to roughly 512 B/V before distances/counters.
- `jasper_oneshot_untested` is a genuine persistent manager/worker concept, but allocates a large page pool per call, uses many global atomics, and lacks targets, paths, and multi-source integration.
- `Charlotte_ADS_Float_Attempt` is a manager/worker persistent design but remains single-source/full-distance, wave32-specific, lacks target/path semantics, and has not been run on HIP hardware.
- The PyTorch Bellman–Ford experiments are incomplete or introduce Python `.item()`/`.any()` synchronization and do not represent the production path.

The production Makefile compiles BF10, Delta, and UnitBFS, not these earlier engines (`Makefile:181-211`). Effort should flow into the production adapters and controllers, not reviving superseded prototypes.

## Why full-edge Bellman–Ford is the wrong full-graph trade

A very optimistic full sweep must read at least an edge endpoint and a source label—roughly 8 bytes per edge before row offsets, destination labels, atomics, writes, or cache misses. On this graph:

```text
125,423,075 edges × 8 bytes ≈ 1.00 GB per sweep
1.00 GB × 214 levels        ≈ 214 GB per 214-hop query
214 GB × 9,654 queries      ≈ 2.07 PB over the archived query count
```

This is an illustrative level-synchronous scenario, not a claim that every query requires 214 BF sweeps. The 214 depth is one repository critical-path regression, not the mean, and an in-place asynchronous BF sweep can sometimes propagate multiple hops depending on edge order and scheduling. The calculation nevertheless establishes the traffic scale: even 20 full sweeps would read roughly 20 GB/query under an unrealistically low traffic model. Checking targets only every `N` sweeps does not reduce this edge traffic.

This aligns with established GPU graph results: work-efficient frontier methods avoid the repeated unnecessary work of traditional GPU Bellman–Ford. Davidson et al. report that their work-saving SSSP methods consistently outperform traditional GPU Bellman–Ford, while Merrill et al. and Gunrock emphasize frontier construction and load balancing for O(V+E)-style traversal ([Davidson et al.](https://escholarship.org/uc/item/8qr166v2), [Merrill et al.](https://research.nvidia.com/publication/2012-02_scalable-gpu-graph-traversal), [Gunrock](https://escholarship.org/uc/item/9gj6r1dj)).

A full-edge BF experiment becomes rational only if a spatial or topological filter makes `E_selected` a small fraction of `E_full`, frontiers are sufficiently dense that queue/claim overhead dominates, and subgraph construction is amortized. Until those conditions are measured, it is a bounded research arm with an early-stop rule, not a redesign candidate.

## Target checking and route acceptance

“Check every N rounds” is no longer primarily a correctness question because the routing contract permits bounded and approximate search. It is a **quality/runtime knob**, and the check must stay on-device.

For a single connection target (plus any legal alternate sinks), expose:

1. **First feasible (`N=0` cleanup):** stop at the first controller barrier after a target has a complete predecessor chain. This is the minimum-work legal-route mode.
2. **Cleanup rounds (`N=1,2,4,8,16`):** remember the first target cost, allow N more improvement rounds, then commit the best current chain.
3. **Relative bound:** stop when `best_target <= α × frontier_lower_bound`, with `α` such as 1.00, 1.05, 1.10, 1.25. This gives a more transferable quality control than a fixed round count when the lower bound is valid.
4. **Exact in box:** BF10 uses its current frontier certificate; Delta stops only when the target’s bucket is settled.

“The target has not changed for N rounds” alone is not an exact certificate. That is acceptable only in the explicitly heuristic modes. A frontier-free full-edge BF must either run to convergence or reduce a valid lower bound from vertices improved in the last sweep if it wants exact/bounded guarantees.

BF10 already performs target checks on-device, so checking less often does not remove CPU traffic. The historical Delta target-settlement kernel was only 0.30% of aggregate dispatch duration; expect box size, dynamic-cost reads, reset, and relaxation to matter more. Still sweep the intervals because the cost distribution of bounded BF10 may differ.

The outer router must decide whether a poorer local route is truly faster. A first-feasible path that causes extra congestion iterations can lose end-to-end even when its SSSP is much faster.

## Recommended compute redesign: persistent bounded weighted routing

### Stage A: bounded BF10 routing service

Extend the existing cooperative controller rather than writing another Bellman–Ford first. Each connection descriptor should contain:

- route-tree source list and legal sink/alternate-sink list;
- representative-coordinate bounding box and box generation;
- criticality, wirelength/timing weights, net-center/HPWL data, and acceptance policy;
- pointers/versions for occupancy, historical congestion, preserved-resource masks, and route ownership;
- output slots for cost, predecessor chain, reached status, and touched-resource list.

The kernel should initialize its own query epoch, expand only admitted nodes, compute dynamic destination cost from resident SoAs, apply one of the acceptance policies above, and publish one traversal descriptor. Compact extraction and commit can be follow-on GPU phases initially, but the CPU should not receive per-round state.

Keeping initialization and controller transitions inside one cooperative kernel minimizes the producer/consumer visibility boundaries that have been fragile on gfx1151. Cooperative occupancy limits and concurrent grids must be queried and stress-tested ([AMD cooperative-groups reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/cooperative_groups_reference.html)). HIP Graph capture is an overhead experiment for a fixed guarded epoch, not a presumed fix for cross-kernel visibility ([AMD HIP Graph documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)).

### Stage B: persistent bucketed goal-directed search

BF10 is the fastest prototype; a bounded bucketed engine is the stronger likely endpoint because it avoids repeatedly revisiting labels that are far from competitive.

Use a quantized priority:

```text
g(v) = accumulated nonnegative routing cost
h(v) = geometric/timing estimate from v to this connection's sink
key(v) = floor((g(v) + lambda * h(v)) / bucket_width)
```

- `lambda=0`, strict bucket settlement: Delta-like exact SSSP inside the box.
- admissible `h`, reopen-capable strict priority: exact A*-style search inside the box.
- `lambda>1`, coarse buckets, or first-feasible target: faster routing heuristic.

The controller owns bucket selection, closure, target acceptance, compaction, and termination. The host sees one query/batch result, not each bucket. Sweep bucket width relative to the observed dynamic-cost distribution; a fixed `delta=1` is meaningless once costs combine congestion, delay, and wirelength.

### Stage C: device-resident negotiated routing state

The SSSP service should consume a factorized cost rather than an uploaded edge-weight vector. A useful form is:

```text
cost_q(u -> v) =
    query_congestion_weight * base_cost[v] * history[v] * present(occupancy[v], q)
  + query_wire_weight       * length[v]
  + query_delay_weight      * delay[v]
  + query_bias_weight       * geometric_bias(v, q)
  + static_edge_cost[u -> v]
```

This mirrors the classes of terms RWRoute evaluates while allowing coefficients to vary by connection. Store static resource attributes once; keep occupancy/history on-device; pass only small query descriptors.

After path extraction, a GPU commit phase should atomically update occupancy/ownership and append touched nodes. Rip-up should traverse the prior compact path and reverse those updates. Historical congestion can be updated once per outer iteration from an overused-node list. The CPU may retain high-level iteration/timing orchestration initially, but not per-resource cost construction or per-round SSSP control.

Same-net sharing is the hardest semantic gap. Current PathFinder approximates route-tree reuse by seeding many sources; RWRoute also discounts resources already used by the same net. Start with route-tree nodes as zero-cost sources plus an owned-by-net test, then validate that this preserves the desired tree behavior before reproducing the full sharing formula.

### Stage D: conflict-aware query batching

One bounded connection can still expose too little work. Batch useful independent work rather than scanning the full graph:

- build an overlap score from connection boxes and existing route-tree cells;
- color or partition the connections so a batch is spatially disjoint/low-overlap where possible;
- preserve sequential order among connections of the same net;
- schedule `(query_id, frontier segment)` tasks by selected outgoing edge volume;
- use snapshot costs within a batch, atomically commit paths, and mark conflicts for the next repair wave.

This relaxes the serial negotiated-routing order, so quality and iteration count must be measured. A stricter first implementation can route one conflict color at a time. Later, an RPTT/spatial partitioner or optimistic batch with conflict repair can expose more parallelism.

### State, reset, and load balance

Generic Delta currently uses about 40–60 B/V per workspace; BF10 uses about 20 B/V plus its edge-source map and performs roughly 323 MiB of V-sized initialization writes per query at production scale. Tight boxes do not help if every query still clears all V state.

Priorities are:

1. generation-stamped or touched-list reset;
2. no full-V target-membership array for connection-sized target lists;
3. query state allocated lazily by spatial cell/page, or a sparse global-ID table, once multi-query memory becomes limiting;
4. append-only touched/frontier lists for reset, extraction, and telemetry;
5. hybrid row expansion: thread-per-row at ordinary degree, warp/block splitting only for measured long-row tails.

Average degree is 4.44, so outgoing CSR remains a good frontier format. Wave-aggregated queue reservation and edge-balanced frontiers should be driven by degree/frontier telemetry, not assumed wins.

## Graph format and spatial audit

### CSR did not destroy the geometry; BF11 now restores the compact runtime view

The upstream `.devicegraph` already retains spatial data:

- node IDs follow original `DeviceResources.nodes` order, not spatial order (`device_to_routing_graph.cpp:493`);
- each node has `min_x/max_x/min_y/max_y` computed from coordinate-bearing wires (`device_to_routing_graph.cpp:536`);
- exact tile/wire-to-node membership is retained (`device_to_routing_graph.cpp:553`);
- each PIP edge retains its exact tile (`device_to_routing_graph.cpp:664`);
- the schema contains physical node columns, lookup records, outgoing CSR, and edge attributes (`interchange/device_routing_graph.hpp:124-158`).

Before BF11, the production converter skipped all 40 B/V of physical node columns, compacted edges on the CPU, and wrote only topology and weights to `.csrbin`; metadata v6 intentionally omitted geometry. That was the audit finding, not a limitation inherent to CSR. The implemented device-graph v4 adds compact `route_end_x`, `route_end_y`, and `base_vertex_cost` columns. The routing projection can skip the legacy 40 B/V physical arrays while retaining these 12 B/V columns. CSR v3 preserves them after design filtering and adds stable compact-edge IDs grouped by destination-coordinate tile plus a final spill shard. PathFinder can load all shards for future selected-edge work or seek past them for BF11's current active-frontier traversal. CSR v1/v2 remain readable; bounded BF11 requires v3 unless explicitly run unbounded.

So the concern is correct for the current runtime representation, but the information is recoverable without reparsing the original device resource.

### Recommended artifact boundary

Split the data into a static device artifact and a small dynamic design artifact.

**Static, once per FPGA device and bounds policy:**

- `uint32 rowptr` and `uint32 colind`;
- immutable edge/base-resource cost terms;
- static PIP attributes and stable original edge IDs;
- a packed representative route coordinate per vertex for the fast RWRoute-like predicate;
- optional packed node extents and edge/PIP tile IDs for conservative or strict variants;
- resource type, length, delay, SLR/region, and any lookahead attributes;
- a spatial cell index and explicit long/global-wire spill lists;
- lookup data and a content/node-order identity.

**Dynamic, once per routed design:**

- route requests;
- disabled-row and disabled-destination bitsets;
- endpoint/source/sink metadata;
- initial preserved routes and resource ownership/reference counts;
- netlist provenance.

**Mutable and device-resident during routing:**

- occupancy/capacity;
- present and historical congestion terms;
- per-net ownership or same-net sharing state;
- compact committed paths and touched/overused worklists;
- one explicit cost-epoch/version counter.

The primary arm should compact static design exclusions once on the GPU, because one design graph is reused across thousands of searches. Direct mask checks are a simpler measured alternative. Compaction must emit a compact-edge-to-original-edge map; masking in place can preserve static positions. Either approach removes the current CPU O(E) compaction and almost-full `.csrbin` write/read. Filtering currently preserves every row, so it does not reduce V-sized query state (`interchange/device_routing_graph.cpp:1106-1126`).

Do not compact or upload effective routing weights per connection. Current Delta’s `update_vertex_costs()` copies about **107.7 MiB** for this V and synchronizes. The full-system workload contains 28,026 nets (`BENCHMARKING.md:30`); at the older negotiated router’s default batch size of 256, 110 full snapshots would expose about **11.6 GiB** of cost upload per outer iteration. Resident factorized costs and sparse commit/rip-up updates are mandatory.

### Layout costs at production scale

| Representation | Approximate size |
|---|---:|
| Current disk CSR: 8 B/V row offsets + 8 B/E destinations/values | 1,172.25 MiB |
| 32-bit topology: 4 B/V + 4 B/E | 586.13 MiB |
| Four `int32` node extents | 430.70 MiB |
| Four packed `uint16` node extents, if coordinate bounds permit | 215.35 MiB |
| One packed `uint16 x/y` representative coordinate | 107.68 MiB |
| One 32-bit V-sized dynamic cost/state array | 107.68 MiB |
| All seven existing physical node columns | 1,076.75 MiB |
| Existing 16-byte edge attributes | 1,913.80 MiB |
| Packed 4-byte edge tile/XY | 478.45 MiB |

Production engines already support 32-bit row offsets when E fits. Keep generic weighted data, but store it as independently mappable sections rather than forcing every consumer to materialize a single wide representation.

Packing coordinates to `uint16` is conditional: the current format permits signed `int32` coordinates and `-1`. Validate the device bounds and reserve an explicit missing-coordinate sentinel, otherwise retain a wider representation.

Use a section-directory/mmap-friendly artifact so consumers can map topology, geometry, or PIP data independently. Keep stable global IDs. This is lower risk than immediately renumbering every vertex and edge.

### Which adjacency formats are worth considering

- **Outgoing CSR:** keep as the primary frontier format. It matches expansion and is compact at degree 4.44.
- **Reverse CSR/CSC:** add only for a measured weighted bidirectional/A* arm. A second 32-bit topology is about 586 MiB. Efficient reconstruction of original forward edge IDs may also require a 4 B/E permutation (about 478.5 MiB), for roughly 1,064.6 MiB total unless IDs can be recovered another way.
- **Spatial cell index:** useful as a sidecar mapping tiles/cells to ID lists or slices in a separate index array; it need not replace CSR. Current IDs are not spatially ordered, so direct contiguous CSR ranges require renumbering first.
- **Destination-endpoint cell-partitioned COO/edge-ID lists:** the preferred auxiliary view for full-edge BF over selected boxes. Precompute macro-tile shards so a rectangle scans only selected edge ranges; an inline box predicate over the unsharded edge array would still scan all 125 million edges. Store explicit sources/destinations or selected stable edge IDs, and do not replace CSR for frontier engines.
- **ELL/HYB:** likely wasteful because routing-node degree is sparse and irregular; padding buys little.
- **Dense/min-plus matrix formats:** reject for this graph size and sparsity.
- **Morton/tile vertex ordering:** defer until measurements show spatial windows and cache locality are strong. Long/global wires should live in a spill partition, and inverse node/edge maps are mandatory.

## Bounding boxes and spatial expansion

### The repository already contains a stale route-window prototype

Git history contains a substantial Delta route-window implementation on `origin/bounding-boxes`:

- `0760494`: fixed margin 20;
- `1739f98`: fixed margin 50;
- `8838f42`: selected-net control plus JSONL telemetry.

At `8838f42`, PathFinder combined all source/sink extents for a net, uploaded one packed `uint64_t` bounds record per row, rejected candidate destinations whose extents did not intersect the window, conservatively retained missing-coordinate nodes, and reran unbounded only when a sink was unreachable. Telemetry distinguished `unbounded_baseline`, `window`, and `fallback` and recorded reached vertices, edge visits, rejected edges, atomic attempts/CAS retries, materialization, and reset (`git show 8838f42:CongestionFreeRouting/ROUTE_WINDOW_README.md`).

This should be ported conceptually, not cherry-picked: it diverges from current Delta/controller changes, depends on coordinate-bearing legacy metadata, uses a fixed whole-net box, and was never HIP-validated. Its packed sidecar, destination predicate, fallback accounting, and telemetry are nevertheless the best starting point in the repository.

### Lowest-risk current prototype

Keep the existing global outgoing CSR and attach a compact, ID-aligned spatial SoA:

- packed representative `{endX,endY}` for the aggressive RWRoute-like predicate;
- optional validated node extent `{minX,maxX,minY,maxY}`, with a missing-coordinate sentinel, for the conservative predicate;
- optionally packed edge/PIP tile ID for strict edge-location filtering;
- per-query endpoint rectangle plus halo.

Benchmark three predicates. Representative-end-coordinate inclusion is cheapest and matches RWRoute’s aggressive idea; extent intersection is more conservative; edge/PIP-tile inclusion is stricter about switch location. Existing node extents cannot reproduce RWRoute’s predicate exactly, which is why a distinct `route_end_x/route_end_y` sidecar is required. A packed PIP tile still does not prove that the entire long wire lies in the box, which is acceptable under the routing-heuristic contract. Exact emulation of bidirectional SLL/Laguna endpoint selection can also depend on the predecessor, so either retain edge-arrival coordinates or encode an explicit SLL exception.

The existing preprocessing policies clarify the choice (`device_to_routing_graph.cpp:504`):

- `poc-base-wire` can exclude a node that intersects elsewhere;
- `fully-contained` excludes boundary-crossing nodes and therefore disproportionately removes long nodes;
- `intersects` is the safest basis for query-time experiments.

### RWRoute-style connection policy

For each source-sink connection:

1. Form the minimum rectangle containing source, sink, and the net geometric center.
2. Apply anisotropic initial extensions; start the sweep around RWRoute’s current 3 horizontal / 15 vertical defaults rather than assuming a symmetric halo.
3. Add device-specific cross-SLR/region rules so the box admits Laguna/SLL access or analogous scarce resources.
4. If the search is unreachable, or its committed route remains congested in the next outer iteration, grow the persistent connection box (RWRoute currently exposes 1 horizontal / 2 vertical increments).
5. After a configurable number/area threshold, use a maximum box or unbounded fallback.
6. Restart a failed attempt unless deferred cut edges are explicitly recorded and replayed.

Always admit the connection source, target, and already-owned route-tree resources even when representative coordinates are missing or sit on an ambiguous boundary. Put unknown-coordinate/global resources behind an explicit resource-type or spill policy; silently admitting every unknown node could destroy box selectivity, while silently rejecting one can make a legal connection unreachable.

Let the device controller manage an attempt; do not synchronize the CPU after each shell. Do not commit occupancy for failed attempts.

Connection-level boxes avoid the huge union rectangle produced by all sinks of a high-fanout net. Connections of the same net remain ordered because each accepted branch changes the route tree and sharing state. Quantize boxes/cells so unrelated connections can reuse selected cell lists and local-state pages.

### Correctness contract

A successful attempt need only produce a legal predecessor chain using admitted original edges under one frozen cost epoch. It need not match the unrestricted shortest path. Exact mode means exact **inside the fixed box and cost snapshot**.

Costs must not mutate while a query is active; otherwise settlement certificates and priority order can become invalid. Snapshot/version the dynamic arrays for a batch, then commit afterward, or route only nonoverlapping boxes when immediate updates are required.

Unbounded fallback is a routability/quality policy, not a correctness requirement after every bounded success. Validate bounded routes with the physical checker and evaluate them by outer congestion convergence, wirelength, timing, and total runtime.

### Spatial measurements required before building local CSR

For every real query, record:

- endpoint-box width, height, and area;
- connection criticality, source/sink/net-center geometry, and route-tree size;
- nodes/edges admitted by representative-coordinate and extent predicates at multiple anisotropic extensions;
- retained outgoing edges and long/global-node count;
- bounded reachability and accepted route cost versus an exact-in-same-box oracle;
- expansion-on-failure, expansion-on-congestion, and fallback rates;
- outer iterations, overuse trajectory, and later reroutes caused by each box policy;
- time saved after charging coordinate reads and any remapping.

Only build dense local subgraphs if windows retain a small fraction of V/E and are reused. Constructing and uploading a unique CSR per net is likely to recreate the current orchestration bottleneck in a more expensive form.

## Additional algorithmic opportunities

### Adaptive per-box engine dispatch

No single weighted engine must win every box. A device dispatcher can choose from telemetry:

- active-frontier BF when boxes are small, cost spread is modest, and duplicate decreases are rare;
- hierarchical/circular buckets when cost spread or unordered re-relaxation is high;
- full-edge BF when `E_box` is tiny and prior frontier density indicates most selected edges will be visited anyway.

At 1% edge retention, the optimistic full-BF lower bound falls from about 1.00 GB to about 10 MB per sweep. That does not guarantee a win, but it makes full-edge BF a serious bounded arm rather than a strawman.

### Fixed-point costs and radix/hierarchical buckets

If the routing objective can be quantized without harming convergence/QoR, fixed-point nonnegative costs enable Dial/radix-style queues, deterministic comparisons, and cheaper atomics than packed floating labels. Sweep scale/bit width against the full routing result, not only SSSP distance error. Retain a float mode until timing and congestion ranges are characterized.

### Weighted bidirectional/A* search

A reverse graph can support bidirectional weighted search for single-source/single-sink connections, but it costs at least another 586 MiB of topology plus edge-ID mapping. Bounding and a forward geometric lookahead may capture most of the benefit. Implement only after forward bounded telemetry shows deep, narrow searches remain dominant.

The heuristic need not be admissible in routing mode. If exact-in-box mode is requested, however, raw Manhattan distance is not automatically safe because long wires cross many tiles cheaply; use a wire-aware lower bound or set `h=0`.

### Extraction

Traversal is not the only control path. Keep compact path extraction, commit, rip-up, and occupancy deltas on-device. Report them separately; otherwise a faster search can be hidden—or its worse route quality can be hidden in later rerouting.

## Existing negotiated-routing semantics worth porting

The current production directory is one-shot, but the older `Routing/` implementation contains a usable semantic skeleton:

- destination congestion cost construction in `Routing/pathfinder.cpp:306-380`;
- occupancy rip-up and selective reroute in `Routing/pathfinder.cpp:600-625`;
- iterative present/history updates and cost snapshots in `Routing/pathfinder.cpp:1009-1199`;
- congestion behavior tests in `Routing/tests/pathfinder_cpu_stub_test.cpp:443-502`.

It defaults to batches of 256 routes per cost snapshot and uploads a full vertex-cost vector before each batch. Port its behavior and tests to the current outgoing-CSR-v2/metadata-v6 path, but do not revive its incoming-CSR-v1 engine or full-vector update architecture wholesale.

The complete-router gap is now the primary repository finding. Missing production capabilities are:

- negotiated-congestion iterations and selective reroute;
- live route commit/rip-up and resident occupancy/history;
- connection-level rather than whole-net rerouting;
- same-net resource ownership/reference counts;
- persistent adaptive connection boxes;
- coordinate/resource/delay metadata in the current artifact;
- factorized per-connection timing/wirelength/congestion cost evaluation;
- conflict-aware GPU batching and deterministic commit policy;
- closed-loop legality, QoR, and convergence benchmarks.

## Concrete implementation roadmap

### Phase 0 — specify the routing contract

1. Preserve the current dirty working-tree changes and record the complete diff with every run.
2. Build all production HIP translation units and focused tests on the target AMD system using both validation flags and exact production `-O3` flags.
3. Define the cost equation, update epoch, same-net sharing rule, bbox predicate, expansion rule, acceptance policy, and batch consistency contract.
4. Port the older negotiated-routing CPU-stub semantics/tests to the current outgoing artifact.
5. Establish hard gates: checker pass, all required connections, zero illegal overuse, preserved-route validity, deterministic original edge IDs.

### Phase 1 — weighted bounded replay

1. Capture fixed cost snapshots with connection sources/targets, bbox, resource attributes, and expected original edge IDs.
2. Port the `8838f42` packed-bounds predicate and telemetry to current generic Delta.
3. Compare host-checked and reduced-controller Delta on identical weighted boxes.
4. Add a CPU Dijkstra oracle for exact-in-same-box validation.

### Phase 2 — bounded BF11 challenger (baseline implemented)

1. **Implemented:** separate BF11 API, resident base/dynamic destination costs, inclusive destination-coordinate predicate, protected route-tree sources, exact interval target certificate, cooperative controller, and complete host fallback.
2. **Implemented:** CSR v3/device-graph v4 route-end and base-cost columns plus destination spatial edge shards.
3. Add touched/generation reset so a tight box no longer pays a full-V initialization.
4. Add first-feasible, cleanup-round, and relative-bound routing modes beside the current exact-in-box policy.
5. Compare 1/2/4 workers and forced host fallback under identical weighted snapshots; then add selected-edge full-BF as the third arm for tiny/dense boxes.

### Phase 3 — persistent hierarchical bucket engine

1. Extend reduced Delta to own a full query on-device.
2. Replace flat global pending scans with circular/multilevel buckets over active box state.
3. Compare float-adaptive and fixed-point/radix policies.
4. Add optional `g + lambda*h` routing priority and strict `h=0` exact mode.

### Phase 4 — device-resident negotiated routing

1. Put occupancy, history, ownership, committed paths, and overused/touched lists on-device.
2. Add sparse GPU commit/rip-up and outer-epoch cost updates.
3. Keep one immutable cost version per active batch.
4. Port selective reroute and adaptive box enlargement.
5. Measure full legal-route convergence before choosing the search engine.

### Phase 5 — connection batching and local state

1. Partition/color connection boxes; preserve same-net dependencies.
2. Sweep snapshot batch sizes 1/8/32/128/256.
3. Add lazy cell-paged or sparse query state to increase concurrent lanes.
4. Add adaptive per-box engine dispatch.
5. Consider Morton reordering/local dense graphs only if sidecar filtering shows a remaining locality bottleneck.

## Benchmark and decision protocol

### Tier 1: fixed weighted-snapshot replay

Freeze topology, dynamic-cost epoch, sources/target, bbox, and cost coefficients so kernel work is comparable.

| Arm | Variants |
|---|---|
| Delta | host/reduced/full-persistent controller; adaptive widths; circular/hierarchical buckets; float versus fixed-point |
| Bounded bucketed search | `lambda=0/1/>1`; strict settlement versus first-feasible/relative-bound acceptance |
| BF10-derived frontier | cooperative/forced-host; workers 1/2/4; target intervals; acceptance policies; dynamic-cost accessor |
| Full-edge BF | selected box edge lists; first-feasible/bounded/exact; adaptive dispatch threshold |
| Spatial predicate | representative coordinate, extent intersection, PIP tile; RWRoute-like anisotropic extensions and growth |

Use CPU Dijkstra or strict unheuristic Delta as the **exact-in-same-box, same-cost-snapshot** oracle. Do not compare bounded results to unrestricted global shortest paths as an acceptance gate.

### Tier 2: closed-loop routing

Start from the same unrouted design and deterministic connection order. Sweep:

- search engine and route-acceptance policy;
- fixed whole-net windows, connection windows, adaptive RWRoute-style boxes, and unbounded routing;
- snapshot/commit batch sizes `1,8,32,128,256`;
- expansion-on-failure, expansion-on-congestion, maximum box area, and unbounded fallback;
- exact-in-box versus heuristic search.

A static replay cannot select the production router by itself because different routes create different occupancy, history, reroute sets, boxes, and future query distributions.

### Sampling

- Two warmups.
- At least ten profiler-free replay samples per low-level arm, interleaved/randomized by arm.
- At least five to ten fresh interleaved full-pipeline samples for the finalists.
- For run-level totals, report median, MAD, and bootstrap confidence intervals plus every raw sample. With only 5–10 full runs, do not present unstable p95/p99 totals.
- For the thousands of per-query latencies, report p10/p50/p90/p95/p99 separately for each run and summarize them across runs.
- Profile only after a profiler-free winner is established; tracing materially perturbs the existing measurements.

The untracked `run_pathfinder_benchmarks.sh` records useful provenance, but currently hardcodes historical-style generic host-checked Delta, one warmup plus one measured run, `--allow-unrouted`, and no physical-netlist checker. It is not yet an algorithm-decision harness.

### Metrics

**End-to-end:** conversion, artifact write/read, graph upload, all negotiated iterations, extraction, commit/rip-up, reconstruction, checker, and total wall time.

**Per connection/search:** criticality, source/target/tree sizes, cost spread, bbox and selected V/E, frontier density, relax attempts, successful decreases, duplicate requeues, bucket/pending scans, rejected-by-box edges, rounds/sweeps, target policy, reset, extraction, and accepted route cost.

**Routing convergence:** connections rerouted per iteration, overused resources, maximum occupancy, present/history ranges, unreachable attempts, box expansions/fallbacks, time to first legal route, and total iterations.

**Runtime/GPU:** cost-state update, bbox construction, search, target detection, extraction, commit/rip-up, launch/copy/sync counts, edge visits, supported cache/request counters, CAS retries, peak memory, power, and energy. Do not infer physical DRAM-bandwidth saturation or general VMEM dependency stalls from counters gfx1151 does not expose (`BENCHMARKING.md:270-304`).

**Hard correctness gates:** physical-netlist checker pass, every required connection, zero illegal overuse/conflicts at completion, preserved routes valid, original edge IDs valid, and same-net sharing/reference counts consistent.

**QoR:** PIP/wirelength and routed-resource count, critical path/delay when timing-driven, contest/benchmark score, and delta versus the agreed RWRoute configuration.

### Proposed decision gates

These are experiment gates, not measured claims:

- Promote BF10 only after the dynamic-cost bounded version beats reduced/persistent Delta on weighted snapshots and improves closed-loop wall time/QoR.
- Choose target-check `N>1`, cleanup rounds, or a relative bound only by closed-loop runtime and convergence; target scan savings alone are not decisive.
- Dispatch full-edge BF only where selected-edge scans beat frontier work after cell-list construction and state reset are charged.
- Treat edge-retention/fallback percentages as screening signals. Promote a box policy only when end-to-end legal-route time improves after expansions and later reroutes are included.
- Promote larger route batches only when stale cost snapshots do not increase total iterations, conflicts, or QoR enough to erase the parallel speedup.
- Require strict checker success for every production arm, but do not require unrestricted shortest-path equality.

## Validation completed during this audit

No HIP compiler/runtime is available in this environment, so no GPU timing or HIP correctness claim was made. The repository itself also states that current GPU translation units and recent changes remain HIP-unvalidated (`DEVELOPMENT_STATUS.md:99-107`).

The following C++ host checks were compiled with `-Wall -Wextra -Wpedantic -Werror` and passed:

- `device_routing_graph_test`
- `bf11_sidecar_policy_test`
- `interchange_import_policy_test`
- `gzip_io_test`
- `sssp_query_capacity_test`
- `unit_bfs_policy_test`
- `delta_stepping_policy_test`
- `pathfinder_router_args_test`
- `pathfinder_cpu_stub_test`
- `pathfinder_bf10_cpu_stub_test`

The following interpreted Python/source-policy checks also passed:

- `pathfinder_benchmark_args_test.py`
- `pathfinder_benchmark_writer_test.py`
- `delta_stepping_pending_reduction_source_test.py`
- `pathfinder_make_dependencies_test.py`
- `bf_standalone_csr_v3_source_test.py`

The BF11 production translation unit and its bounded/dynamic HIP regression also pass strict host-side macro-expanded syntax checking with fake HIP headers. This verifies C++ integration and launch signatures, not HIP code generation, cooperative-grid behavior, or GPU correctness.

These results were observed during this audit, but their command logs were not added as repository artifacts. They protect host policy, parsing, adapters, and source-level regressions; they do not substitute for the BF10/UnitBFS/Delta HIP suites, concurrent-stream stress, or full routing checker on AMD hardware.

## Evidence limitations

1. The best retained profile is historical generic all-light Delta selected with the old max-iteration forcing workaround and legacy predecessor handling. It demonstrates control overhead but not the behavior of dynamic weighted bounding-box routing.
2. The raw `39155_results.db` is absent; only derived JSON, documentation, and plots remain, so the profile cannot be independently regenerated from this checkout.
3. Historical worker timings lack raw samples/confidence intervals and show 2/4/8 workers effectively flat (`BENCHMARKING.md:78-94`).
4. There are no stored BF10 timings, no current profiler-free reduced-Delta samples, and no comparable weighted bounded snapshots.
5. The historical route-window branch has source/telemetry but no retained AMD validation or current-code performance result.
6. The only negotiated-routing loop lives in the older `Routing/` path and uses obsolete artifact/orientation assumptions.
7. Current Delta, test, Makefile, and untracked harness changes make old binaries and profiles stale relative to the working tree.
8. Recent roadmap speedup percentages are explicitly estimates, not measurements.

These gaps change the order of work: the first milestone is a controlled replay, not an architectural commitment based on the old trace alone.

## Final recommendation

The archived host-checked Delta path strongly supports the synchronization hypothesis, but the production objective is now broader: minimize time to a legal, congestion-free, acceptable-quality route under dynamic costs. UnitBFS does not address that objective and should not guide the redesign.

The immediate experiment is now **bounded dynamic-cost BF11**: its persistent controller most directly tests whether eliminating per-round host control breaks the bottleneck. The likely long-term winner remains **persistent bounded hierarchical-bucket/Delta-style search**, optionally ordered by an RWRoute-like goal estimate, because weighted priority reduces repeated label corrections as congestion costs spread. A selected-edge full-BF kernel should remain available for tiny dense boxes.

The highest-value system change is to put negotiated-routing state on the GPU: occupancy, history, route ownership, compact paths, commit/rip-up, adaptive connection boxes, and cost epochs. Without that, every routing iteration would rebuild or transfer huge cost vectors and the SSSP controller improvement would be lost.

CSR should remain the frontier backbone. Restore geometry beside it, add static resource/delay/base-cost SoAs and a spatial cell index, and use cell-partitioned edge lists only as an auxiliary full-BF view. A successful bounded route does not need global-shortest verification; it needs legality and acceptable closed-loop QoR.

If only one engineering sequence is funded, use this one:

```text
routing contract + ported negotiated-routing tests
    -> restore current-artifact connection geometry and route-window telemetry
    -> weighted bounded snapshot replay
    -> benchmark and harden the implemented persistent dynamic-cost BF11 challenger
    -> fully persistent hierarchical-bucket/Delta engine
    -> GPU-resident commit/rip-up, occupancy, and history
    -> adaptive connection boxes and conflict-aware batching
    -> per-box BF-frontier / bucket / full-edge dispatch
```

That sequence targets the measured orchestration problem while converging on the actual RWRoute-class workload rather than optimizing the current unit-weight one-shot surrogate.
