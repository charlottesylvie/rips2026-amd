# Adaptive route windows

Route windows are an opt-in Delta-Stepping optimization.  They never change
PathFinder's shortest-path result: a bounded path is only an incumbent, and
every reached target set is checked by one unbounded multi-target query.  Its
exclusive distance limit is the largest bounded incumbent cost.  A target is
replaced only by a strictly cheaper global path; equal-cost and more-expensive
verification paths retain the bounded incumbent.

The single verification explores every path below that largest incumbent.  As
each target's incumbent is no larger than the limit, any path cheaper than that
target is included.  Thus batching is equivalent to checking every sink below
its own incumbent cost, while avoiding one SSSP launch and CPU/GPU result
synchronization per sink.

```text
pathfinder graph.csrbin graph.ifmeta.bin --sssp-engine delta-step \
  --route-window --route-window-stats-out route-window.jsonl
```

`--route-window-net-list selected.jsonl` limits the optimization to JSONL
entries such as `{"net_index":17}` or `{"net":"example_net"}`.  It requires
`--route-window`.

## Window schedule

PathFinder forms the bounding box of valid source and initially unresolved sink
node extents.  Initial margins are independent for X and Y:

```text
margin_axis = clamp(ceil(endpoint_span_axis * scale), min_margin, max_margin)
```

Defaults are `min_margin=16`, `max_margin=512`, and `scale=0.5`.  Configure
them with `--route-window-min-margin`, `--route-window-max-margin`, and
`--route-window-margin-scale`.

If any sink is missing, each margin doubles from the original endpoint box on
every retry.  Each resulting box is clamped to the coordinate extents of the
uploaded device node metadata.  When expansion cannot enlarge the box (which
includes reaching full-device bounds), PathFinder performs a full unbounded
fallback.  Endpoint nodes with unknown coordinates simply disable the window
for that net and run an explicitly reported unbounded query.

## Bounds and safety

The immutable shared Delta graph uploads one compact `int32_t` bounds record
per CSR row once.  It has four inclusive coordinates plus an explicit validity
bit; coordinates at or above 65535 are valid.  An all-unknown node remains
traversable conservatively, and telemetry reports the graph-wide count of such
nodes.  Partial unknown coordinates and inverted metadata ranges are rejected
before dispatch.  Enabling a route window without uploaded bounds fails fast.

Both generic Delta-Stepping light/heavy relaxation and the exact-unit
specialization apply the same destination interval-intersection predicate, so
unit-weight nets retain their specialization.  Bounds are shared immutable
graph data; no per-net device upload occurs.

## Telemetry

`--route-window-stats-out` writes one stable-order JSONL record for every
bounded attempt, unbounded baseline, verification, and fallback.  Records
include the attempt number, box, retry/verification reason, reached and
unreached target counts, rejected edges, unknown-coordinate count, and whether
global-cost verification was required, plus the selected execution path.  A
successful bounded multi-target attempt is always followed by exactly one
verification record.  On a verification record, `target_count` is the full
target set and `reached_target_count` counts only targets with strictly cheaper
replacement paths; `unreached_target_count` counts retained incumbents.

## Reproducible speed benchmark

Use the supplied A/B driver with a HIP-built `pathfinder` executable and one
existing CSR/metadata pair.  It alternates unbounded and windowed runs, warms
up the GPU, writes a log and telemetry JSONL for every run, and reports median
end-to-end wall-clock speedup.  Arguments after `--` are shared by both modes;
keep the worker count and delta configuration fixed there.

```bash
python3 CongestionFreeRouting/tests/route_window_benchmark.py \
  --pathfinder ./pathfinder \
  --graph design.csrbin \
  --metadata design.csrbin.ifmeta.bin \
  --output-dir /tmp/route-window-benchmark \
  --repetitions 7 --warmups 2 \
  -- --delta auto --parallel-net-workers 4
```

`summary.json` contains every measured wall-clock sample, median speedup
(`unbounded/windowed`), and the telemetry for each measured run.  A speedup
greater than one means the complete safe windowed flow was faster.  The result
includes global-cost verification and any fallback work, so it is not a
misleading bounded-query-only number.  Use `--window-net-list` to benchmark a
selected JSONL net subset, or the `--window-*-margin` options to evaluate a
specific margin schedule.  The telemetry-enabled runs are diagnostic: use a
matching run without `--route-window-stats-out` for production timing, keeping
the Delta configuration and execution path (exact-unit or generic) identical.
Each telemetry summary also exposes bounded attempts, verification SSSP queries,
fallback count, rejected edges, and execution-path counts alongside the median
end-to-end time.

The benchmark harness itself can be smoke-tested without HIP or routing data:

```bash
python3 CongestionFreeRouting/tests/route_window_benchmark_test.py
```
