# cuGraph-Inspired Generic SSSP Roadmap

## Purpose

This document maps current open-source RAPIDS cuGraph SSSP ideas onto the
HIP/ROCm implementation in `CongestionFreeRouting`. The target is a competitive
single-GPU, nonnegative-weight SSSP backend that supports both conventional
distances-only output and PathFinder's multi-source/multi-target path output.

The source audit used RAPIDS cuGraph commit
[`89fa4920231b83ba1bbd70cc9e97a64709ba919b`](https://github.com/rapidsai/cugraph/tree/89fa4920231b83ba1bbd70cc9e97a64709ba919b).
The principal references are:

- [SSSP implementation and Near/Far control](https://github.com/rapidsai/cugraph/blob/89fa4920231b83ba1bbd70cc9e97a64709ba919b/cpp/src/traversal/sssp_impl.cuh#L189-L374);
- [outgoing-edge transform and destination reduction](https://github.com/rapidsai/cugraph/blob/89fa4920231b83ba1bbd70cc9e97a64709ba919b/cpp/include/cugraph/prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh#L431-L599);
- [degree-aware edge extraction/expansion](https://github.com/rapidsai/cugraph/blob/89fa4920231b83ba1bbd70cc9e97a64709ba919b/cpp/include/cugraph/prims/detail/extract_transform_if_v_frontier_e.cuh#L127-L296); and
- [graph degree thresholds](https://github.com/rapidsai/cugraph/blob/89fa4920231b83ba1bbd70cc9e97a64709ba919b/cpp/include/cugraph/graph_view.hpp#L244-L254).

cuGraph's current SSSP is best treated as an adaptive Near/Far label-correcting
algorithm rather than textbook Delta-Stepping. The local implementation should
therefore add it as an A/B backend, not silently replace the current algorithm
before correctness and performance are established.

## cuGraph techniques to adopt

### 1. Graph-aware distance threshold

cuGraph seeds its distance threshold using graph statistics:

```text
delta = warp_size * average_edge_weight / average_vertex_degree
```

For HIP/ROCm, use the active device's runtime wave size and average out-degree.
The local algorithm supports destination vertex costs, so the relevant
quantity is average **effective** weight:

```text
effective_weight(u, v) = edge_weight(u, v) * vertex_cost(v)
```

Treat the formula as a seed, not a universal optimum. Preserve an explicit
numeric override and expose a multiplier that can be swept around the seed.
Heavy-tailed distributions may need sampling, clamping, or a quantile-based
alternative to the global mean.

### 2. Two-level Near/Far scheduling

cuGraph organizes pending work into four logical queues:

1. current urgent/near-near work;
2. next urgent/near-near work;
3. near-far work; and
4. far work.

This design keeps enough near work available to occupy the GPU without forcing
every tentative distance into a strict circular bucket. Queue entries are
validated lazily against current distance state, so stale entries can be
dropped when consumed instead of maintaining a unique token in every queue.

The urgent queue is bounded according to graph size, GPU capacity, and average
degree. cuGraph also uses multiple subpartitions to reduce queue contention.
The local port should use compute-unit count and measured edges-per-CU rather
than copying an NVIDIA SM constant. The number of shards should be swept on the
target AMD GPU; 16 is a starting reference, not a requirement.

Expected local benefits:

- remove repeated minimum-bucket reduction and pending compaction;
- avoid maintaining one flat unique pending set;
- process all relevant outgoing edges in the selected frontier model rather
  than rescanning every settled row as separate light and heavy passes;
- trade some duplicate queue entries for fewer membership atomics and scattered
  resets; and
- expose enough urgent work for stable GPU occupancy across broad weights.

### 3. Degree-aware outgoing-edge expansion

cuGraph does not assign every CSR row to the same execution policy. Its graph
primitives distinguish low-, medium-, and high-degree work and use increasing
cooperation for longer rows.

The HIP version should provide:

- thread-per-row for the short FPGA rows already served well by the current
  kernel;
- packed-wave or lane-group processing for collections of short/medium rows;
- wave-per-row for long rows; and
- CTA or edge-balanced processing for extreme rows.

Thresholds must be chosen from the degree distribution of **reached frontier
vertices**, not the whole graph alone. Use AMD wave32/wave64 behavior and
measure register, LDS, and occupancy effects separately.

### 4. Reduce candidates by destination

The local relaxation kernels currently issue a global distance atomic for each
candidate, publish a parent candidate, claim queue membership, and reserve a
queue slot. Several frontier edges targeting the same vertex therefore create
contention and redundant traffic.

cuGraph's frontier primitive transforms outgoing edges, groups candidates by
destination, and reduces competing values before applying the winning update.
The local implementation should adopt this in tiers:

1. count candidate-to-unique-destination ratio and atomic retries;
2. after edge-balanced expansion, combine equal destinations within a wave;
3. add block-local aggregation for sufficiently large tiles; and
4. use global radix sort/reduce only for large, collision-heavy frontiers.

Do not insert wave collectives into the current divergent per-thread CSR loops.
Lanes can execute different numbers of loop iterations, so those call sites do
not provide a safe convergent collective.

### 5. Optional predecessor output

cuGraph can discard predecessor output. A competitive SSSP interface should
offer the same choice.

Distances-only mode should omit:

- the compact parent key;
- the shared edge-to-source map;
- legacy predecessor node/edge arrays;
- path-measurement and path-fill buffers; and
- predecessor/path extraction kernels.

PathFinder still needs exact original outgoing edge IDs. Its path-producing
mode should retain the existing compact `{distance_bits, original_edge_id}`
parent key and shared edge-source map, with a wide/legacy fallback when compact
edge IDs are unavailable.

## Local gap summary

| Local behavior | cuGraph-inspired replacement | Expected impact | Invasiveness |
| --- | --- | --- | --- |
| Numeric low-level delta; PathFinder auto seed available | Validate graph-aware `delta=auto` and multiplier sweeps on weighted GPUs; add workspace refresh for dynamic costs | Medium--High | Low |
| Flat pending set plus minimum scan and compaction | Four-queue adaptive Near/Far scheduler | Very high on heterogeneous weights | Very high |
| Three 32-bit membership arrays | Lazy stale entries or compact/versioned state | High | Medium--High |
| Six full-size per-worker vertex queues | Bounded, sharded scheduler queues with safe spill | High memory and cache benefit | High |
| One thread serially scans every active row | Degree-aware, edge-balanced expansion | High on skewed graphs | High |
| One global update sequence per candidate edge | Hierarchical reduction by destination | High when destinations collide | Medium--High to High |
| Host-visible count and bucket transitions | Device-resident scheduler control | High on deep/small-frontier searches | High |
| Parent state always available | Optional distances-only specialization | High for standard SSSP measurement | Low--Medium |
| Always-64-bit row offsets | Eligible 32-bit device CSR specialization | Medium memory/cache benefit | Medium |

## Implementation roadmap

### Stage A: measurement and cheap specializations

1. Add weighted benchmark families and CPU-Dijkstra validation.
2. Add queue, degree, collision, stale-entry, atomic, and synchronization
   telemetry.
3. Validate the implemented PathFinder `delta=auto` mode and multiplier sweep
   on weighted GPU families; add automatic low-level refresh after value/cost
   updates if needed.
4. Add distances-only output and finish lazy parent/path allocation. Legacy
   predecessor arrays are already lazy for eligible compact-parent searches.
5. Add eligible 32-bit device row offsets and a forced-wide A/B mode.

Exit criterion: reproducible weighted baselines for classic Delta-Stepping in
distance-only and path-producing modes, with enough telemetry to predict which
structural changes matter.

### Stage B: common edge-expansion primitive

1. Retain thread-per-row below a measured short-row threshold.
2. Add lane-group/wave and CTA paths for longer rows.
3. Make the primitive usable by classic Delta, Near/Far, and eventually
   UnitBFS.
4. Add safe wave-local queue reservation and destination grouping.

Exit criterion: the hybrid expander matches current performance on uniformly
short rows and improves skewed-degree weighted cases without correctness
regressions.

### Stage C: Near/Far prototype

1. Add a selectable backend flag.
2. Implement four distance queues with multiple shards.
3. Add lazy stale validation and safe queue overflow/spill behavior.
4. Tune urgent capacity from CU count, graph size, average degree, and measured
   edge throughput.
5. Preserve multi-source initialization, target settlement, iteration limits,
   callbacks, vertex costs, compact parents, and workspace reuse.

Exit criterion: Near/Far is correct on the complete weighted regression suite
and wins on at least one broad-weight and one skewed-degree family without a
material regression on short-row routing graphs.

### Stage D: destination aggregation

1. Enable wave-local aggregation where collision telemetry warrants it.
2. Add a block-local path and a direct-atomic bypass.
3. Prototype global sort/reduce using ROCm primitives only for large frontiers.
4. Select the path from frontier size and observed or predicted collision rate.

Exit criterion: fewer global distance/parent/queue atomics and lower traversal
time, including the cost of grouping or sorting.

### Stage E: state and controller redesign

1. Combine or version queue membership state.
2. Remove touched-state resets made obsolete by epochs or lazy entries.
3. Keep Near/Far queue capacity proportional to useful active work rather than
   allocating every queue at `V`.
4. Move scheduler continuation and queue promotion decisions to the device.
5. Stress concurrent explicit streams on gfx1151 before removing existing host
   safety boundaries.

Exit criterion: lower per-worker memory and reset traffic, no stale-state leak
across repeated queries, and no host synchronization inside steady-state
scheduler processing.

### Stage F: throughput scaling

1. Re-sweep independent worker counts after state reduction.
2. Batch multiple queries in one launch when individual queries underfill the
   GPU.
3. Evaluate persistent/cooperative execution or HIP Graph capture only after
   scalar D2H dependencies are gone.
4. Consider multi-GPU distribution only after the single-GPU backend is
   competitive.

## What not to copy blindly

- CUDA warp and SM constants: derive wave and CU behavior from HIP and target
  hardware.
- Global sort/reduce on every frontier: retain a direct path for small or
  low-collision frontiers.
- Queue shrinking after each search: repeated routing queries should retain and
  reuse useful capacity.
- Graph-wide degree thresholds without reached-frontier telemetry.
- BFS direction optimization: it requires different graph assumptions and is
  not a generic weighted SSSP optimization.
- NVIDIA-specific cooperative or memory-order assumptions: validate all
  dependent dispatch and atomic behavior on gfx1151.

## Benchmark and acceptance requirements

Benchmark at least:

- all-light, all-heavy, and mixed-weight rows;
- low-degree road-like and skewed-degree synthetic graphs;
- zero-weight edges and SCCs;
- weights immediately around scheduler thresholds;
- dynamic destination vertex costs;
- duplicate sources/targets, unreachable targets, and early target stopping;
- small, medium, and GPU-saturating frontiers;
- repeated workspace calls and concurrent explicit streams; and
- both distances-only and exact-path output.

Report graph preprocessing/upload separately from traversal. For traversal,
report elapsed time, examined edges, successful relaxations, unique updated
destinations, queue insertions, stale entries, queue peaks, atomic retries,
controller synchronization, reset, and path extraction. Use identical graphs,
source sets, target sets, compiler options, device clocks, and validation for
all A/B comparisons.
