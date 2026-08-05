# BF11 concurrency-saturation optimization

## Evidence and selected change

The corrected target sweep shows an early throughput plateau: several worker
counts around three through seven are similar, while eight is substantially
slower. It does not support monotonic scaling or a large multi-query batch.
Automatic BF11 selection therefore uses the smallest measured plateau point on
`gfx1151` (three) when route count, CPU threads, free GPU memory, the exact
graph/state bytes plus conservative geometrically rounded ceilings for query
capacity and successful compact paths, and CU count all permit it. The memory
estimate doubles replaceable query/path storage to cover old-plus-new allocation
overlap. The remaining 25% is explicit HIP runtime/allocator headroom, not an
assumption that unmeasured path storage will fit.
Automatic selection is capped at four; an explicit
`--parallel-net-workers N` remains an override.
Unmeasured architectures conservatively remain at one worker.

The retained ROCm trace in `profiling/39155_analysis` is useful evidence for
runtime saturation, but it measured generic Delta-Stepping rather than BF11 and
cannot establish BF11 phase dominance. BF11 static analysis does establish one
high-frequency relaxation cost: every admissible examined edge previously used
a 64-bit `atomicCAS(address, 0, 0)` read-modify-write merely to observe packed
state. The implemented optimization replaces that unconditional observation with
`__hip_atomic_load(..., __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)` when the
compiler supports it. `BF11_FORCE_CAS_ATOMIC_LOAD` and compilers without the
intrinsic retain the proven CAS load. The successful compare/exchange still
decides exact infinity-to-finite first discovery. Successful producers use a
relaxed mark load before CAS election, and each block aggregates its first
frontier/touched reservation with direct overflow fallback for additional
entries. Block reduction also limits the next-frontier minimum to at most one
global `atomicMin` per block.

Explicit streams now support exact device-controlled segments of 1, 2, 4, 8,
or 16 Bellman-Ford rounds between host status check-ins. Segment size one is the
compatibility/control path. HIP Graph replay is separately controlled by
`--bf11-hip-graph auto|on|off`; `auto` is conservative, never selects replay for
K=1, and both `auto` and `on` fall back to direct segmented enqueue after any
unsupported setup, capture, instantiation, or launch. Adaptive sparse/dense
state reset starts at a conservative touched fraction of 0.25 and can be swept
with `--bf11-adaptive-reset-threshold FRACTION`, where the fraction must be in
the interval (0, 1]. Independent overlapping cooperative grids remain disabled.
HIP Graph API code is compile-gated by `BF11_ENABLE_HIP_GRAPHS`; conservative
production builds may omit it, while the profiling build below enables it
explicitly. Runtime `on` still falls back to direct enqueue if the compiled or
runtime capability is unavailable.

## Telemetry

Pass `--bf11-telemetry` to emit one `bf11_telemetry` JSON record after all
workers join. It is disabled by default. Timing fields are aggregate
nanoseconds:

- `reset_seed_gpu`, `relaxation_gpu`, and `target_check_gpu` measure device
  phases;
- `iteration_status_copy_gpu` measures device-to-host controller/status copies;
- `stream_synchronize_cpu` measures blocking host synchronization;
- `target_summary_gpu`, `target_prefix_gpu`, `path_reconstruction_gpu`, and
  `output_transfer_gpu` cover compact result extraction;
- `total_query_cpu` is the sum of complete query durations.

Both the explicit-stream segmented controller and the null-stream cooperative
controller measure internal relaxation/target phases with `wall_clock64()` and
convert ticks using
`hipDeviceAttributeWallClockRate`; ROCm documents that `clock()`/`clock64()` do
not work correctly on RDNA3/GFX11, so they are deliberately not used here. See
the official [HIP timer-function documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.1/how-to/hip_cpp_language_extensions.html#timer-functions).

The `work` object reports iterations, frontier vertices, edges examined,
successful relaxations, touched vertices, and maximum touched density. The
`memory` object reports the sum and maximum of each worker's lifetime high-water
retained device allocations, plus free GPU memory before and immediately after
worker construction. Those byte counts exclude allocator metadata and transient
old-plus-new overlap during a growing-buffer replacement; the automatic policy's
separate `peak_workspace_device_bytes_estimate` includes a conservative bound
for that overlap and for compact paths. PathFinder explicitly selects the
identity-cost estimate because it never updates dynamic multipliers: retained
graph-sized state is 24 bytes per vertex. The memory record also reports
preallocated query bytes and the conservative retained/peak dynamic ceilings.
Leaving identity mode lazily adds a four-byte multiplier per vertex; a maximum
sparse update can additionally retain eight bytes per vertex of node/value
staging, and the peak estimate covers replacement overlap for that staging.
Pinned host staging is excluded from every device-memory estimate. Requested worker count uses zero
for automatic selection; effective count reflects route-count and
external-stream limits.

GPU and query times are sums across queries and can exceed routing wall time
when workers overlap. Compare phase shares against their aggregate, and use
`bf11_runtime_stats.routing_seconds` for end-to-end BF11 backend acceptance.
Compare `status_copies` and `stream_synchronizations` against actual rounds to
confirm the requested K-fold controller reduction. A high relaxation share with
many edges per successful relaxation supports further atomic or degree-balancing
work; a high reset share should be read together with maximum touched fraction;
a high summary/reconstruction share supports reconstruction-chain reduction.

The `bf11_runtime_stats` schema is version 3 and the opt-in `bf11_telemetry`
schema is version 2. They include the requested segment/Graph/reset
configuration; actual rounds, segments, no-op rounds, direct/Graph segments,
status copies, stream synchronizations, and Graph fallbacks; sparse/dense reset
and cost-mode counts; relaxation/mark/queue work; bounded fallback extraction
avoidance; summary/prefix/reconstruction/transfer timing; and current plus
high-water workspace bytes. Aggregate query and GPU phase times overlap across
workers and must not be added or interpreted as routing wall time.
With telemetry disabled, phase events, device counters, and process-wide
telemetry counter updates are bypassed. Compact device arenas retain their
allocation high-water marks, while the pinned transfer window follows recent
useful output; a larger result takes the documented extraction-only replay
instead of making every later small query copy the high-water arena.
Accordingly, the always-emitted runtime record keeps configuration and routing
wall time when telemetry is off, while its detailed work/copy/reset counters
remain zero; use a separate `--bf11-telemetry` run for those fields.

## Pre-profile controller commands

The exact profiling configuration requested for K=8 plus HIP Graph replay is:

```bash
/tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
  --sssp-engine bf11 \
  --parallel-net-workers 3 \
  --bf11-segment-rounds 8 \
  --bf11-hip-graph on \
  --bf11-adaptive-reset-threshold 0.25 \
  --allow-unrouted
```

Run the full controller matrix with telemetry omitted for one warm-up and five
wall-time samples. Worker four is a contention control, not an automatic-policy
candidate. Graph `on` at K=1 must remain a direct/fallback compatibility case:

```bash
GRAPH=/absolute/path/design.csrbin
META=/absolute/path/design.csrbin.ifmeta.bin
OUT=/tmp/bf11-preprofile
mkdir -p "$OUT"

for MODE in optimized cas-control; do
  BIN="/tmp/pathfinder-bf11-$MODE"
  for WORKERS in 1 3 4; do
    for K in 1 2 4 8 16; do
      for HIP_GRAPH in off on auto; do
        for REP in 0 1 2 3 4 5; do
          "$BIN" "$GRAPH" "$META" \
            --sssp-engine bf11 \
            --parallel-net-workers "$WORKERS" \
            --bf11-segment-rounds "$K" \
            --bf11-hip-graph "$HIP_GRAPH" \
            --bf11-adaptive-reset-threshold 0.25 \
            --allow-unrouted \
            >"$OUT/$MODE-w$WORKERS-k$K-g$HIP_GRAPH-r$REP.log" 2>&1
        done
      done
    done
  done
done
```

Collect telemetry in separate runs so event and counter overhead cannot affect
the five-run medians:

```bash
for WORKERS in 1 3 4; do
  for K in 1 2 4 8 16; do
    for HIP_GRAPH in off on auto; do
      /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
        --sssp-engine bf11 \
        --parallel-net-workers "$WORKERS" \
        --bf11-segment-rounds "$K" \
        --bf11-hip-graph "$HIP_GRAPH" \
        --bf11-adaptive-reset-threshold 0.25 \
        --bf11-telemetry \
        --allow-unrouted \
        >"$OUT/telemetry-w$WORKERS-k$K-g$HIP_GRAPH.log" 2>&1
    done
  done
done
```

## Exact target build and benchmark

Build the optimized intrinsic and the otherwise identical CAS control from the
same source tree. The control isolates this optimization without mixing in a
different graph, worker policy, or telemetry implementation:

```bash
COMMON_FLAGS=(-std=c++17 -O3 -x hip -DBF10_NO_MAIN -DBF11_NO_MAIN \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  -I CongestionFreeRouting/delta_stepping \
  -I CongestionFreeRouting/unit_bfs)
COMMON_SOURCES=(CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf10.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  CongestionFreeRouting/delta_stepping/delta_stepping_hip_CSR.cpp \
  CongestionFreeRouting/unit_bfs/unit_bfs_hip_CSR.cpp)

hipcc "${COMMON_FLAGS[@]}" -DBF11_ENABLE_HIP_GRAPHS \
  "${COMMON_SOURCES[@]}" -pthread \
  -o /tmp/pathfinder-bf11-optimized
hipcc "${COMMON_FLAGS[@]}" -DBF11_ENABLE_HIP_GRAPHS \
  -DBF11_FORCE_CAS_ATOMIC_LOAD \
  "${COMMON_SOURCES[@]}" -pthread -o /tmp/pathfinder-bf11-cas-control
g++ -std=c++17 -O2 CongestionFreeRouting/pathfinder_router.cpp \
  -o /tmp/PathFinderFile-bf11

# Build the pre-change three-worker BF11 baseline from the recorded branch
# base without switching or modifying the validation checkout.
BASE_COMMIT=399e176dc0e93a751dcee03a4481dc5cd34c9b87
BASE_TREE=/tmp/bf11-base-399e176
if [ ! -e "$BASE_TREE/.git" ]; then
  git worktree add --detach "$BASE_TREE" "$BASE_COMMIT"
fi
test "$(git -C "$BASE_TREE" rev-parse HEAD)" = "$BASE_COMMIT"
(
  cd "$BASE_TREE"
  hipcc "${COMMON_FLAGS[@]}" "${COMMON_SOURCES[@]}" -pthread \
    -o /tmp/pathfinder-bf11-baseline
)
```

Use an already converted CSR v3/metadata pair. Omitting `--routes-out` prevents
route-output metadata loading; conversion is not part of these runs. The
following K=1, Graph-off run isolates the relaxed atomic load from its CAS
control with one warm-up and five measured repetitions:

```bash
GRAPH=/absolute/path/design.csrbin
META=/absolute/path/design.csrbin.ifmeta.bin
OUT=/tmp/bf11-cas-control
mkdir -p "$OUT"

for MODE in optimized cas-control; do
  BIN="/tmp/pathfinder-bf11-$MODE"
  for WORKERS in 1 3 4; do
    for REP in 0 1 2 3 4 5; do
      "$BIN" "$GRAPH" "$META" \
        --sssp-engine bf11 \
        --parallel-net-workers "$WORKERS" \
        --bf11-segment-rounds 1 \
        --bf11-hip-graph off \
        --allow-unrouted \
        >"$OUT/$MODE-w$WORKERS-r$REP.log" 2>&1
    done
  done
done
```

Record the pre-change three-worker BF11 baseline with the same graph, one
warm-up, five measured repetitions, and telemetry disabled:

```bash
for REP in 0 1 2 3 4 5; do
  /tmp/pathfinder-bf11-baseline "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers 3 \
    --allow-unrouted \
    >"$OUT/baseline-w3-r$REP.log" 2>&1
  /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers 3 \
    --bf11-segment-rounds 8 \
    --bf11-hip-graph on \
    --bf11-adaptive-reset-threshold 0.25 \
    --allow-unrouted \
    >"$OUT/candidate-w3-k8-graph-on-r$REP.log" 2>&1
done
```

Extract medians from repetitions one through five:

```bash
python3 - "$OUT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
for mode in ("optimized", "cas-control"):
    for workers in (1, 3, 4):
        samples = []
        for rep in range(1, 6):
            path = root / f"{mode}-w{workers}-r{rep}.log"
            records = [json.loads(line) for line in path.read_text().splitlines()
                       if line.startswith("{")]
            record = next(r for r in records
                          if r.get("type") == "bf11_runtime_stats")
            samples.append(record["routing_seconds"])
        print(mode, workers, statistics.median(samples), samples)

baseline = []
candidate = []
for rep in range(1, 6):
    path = root / f"baseline-w3-r{rep}.log"
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.startswith("{")]
    record = next(r for r in records
                  if r.get("type") == "bf11_runtime_stats")
    baseline.append(record["routing_seconds"])
    path = root / f"candidate-w3-k8-graph-on-r{rep}.log"
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.startswith("{")]
    record = next(r for r in records
                  if r.get("type") == "bf11_runtime_stats")
    candidate.append(record["routing_seconds"])
baseline_median = statistics.median(baseline)
candidate_median = statistics.median(candidate)
print("pre-change-baseline", 3, baseline_median, baseline)
print("candidate", "w3-k8-graph-on", candidate_median, candidate)
print("candidate_speedup", baseline_median / candidate_median)
PY
```

Acceptance requires the optimized one-worker median to be no more than 1.03
times the CAS-control one-worker median. The best optimized small-worker median
must be below the better of the CAS-control three- and four-worker medians.
The selected candidate must also have a median no greater than one half of the
pre-change three-worker BF11 median. Worker four is the contention/control
point, not an automatic candidate.

Collect phase and memory telemetry separately so event overhead is not mixed
into the disabled-telemetry acceptance result:

```bash
for WORKERS in 1 3 4; do
  for REP in 1 2 3 4 5; do
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers "$WORKERS" \
      --bf11-segment-rounds 1 \
      --bf11-hip-graph off \
      --bf11-telemetry \
      --allow-unrouted \
      >"$OUT/telemetry-w$WORKERS-r$REP.log" 2>&1
  done
done
```

## Correctness controls

The HIP regression compares 1, 3, 4, and 8 explicit-stream workers against the
one-worker BF11 result and CPU Dijkstra across weighted diamonds, zero weights,
long and wide frontiers, multiple sources/targets, missing coordinates,
unreachable targets, `max_iters=0`, reuse, and exceptional recovery:

```bash
hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
  -DBF11_ENABLE_HIP_GRAPHS \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test
/tmp/bf11_bounded_dynamic_hip_test

hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
  -DBF11_ENABLE_HIP_GRAPHS \
  -DBF11_FORCE_CAS_ATOMIC_LOAD \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test_cas
/tmp/bf11_bounded_dynamic_hip_test_cas
```

For deterministic PathFinder output hashes, run the same weighted CSR through
every segment/Graph combination with `--routes-out`, then compare each file to
the K=1, Graph-off control. The HIP regression above is the independent CPU
Dijkstra oracle and includes constant-one, general-static, dynamic-cost,
bounded, unbounded, fallback, zero-weight, multi-source, and multi-target cases:

```bash
for K in 1 2 4 8 16; do
  for HIP_GRAPH in off on auto; do
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers 1 \
      --bf11-segment-rounds "$K" \
      --bf11-hip-graph "$HIP_GRAPH" \
      --routes-out "$OUT/routes-k$K-g$HIP_GRAPH.jsonl" \
      --allow-unrouted
  done
done
sha256sum "$OUT"/routes-k*-g*.jsonl
```

Collect HIP API counts for K=1 and K=8 with the same one-worker query prefix.
The K=8 trace must show status-copy and stream-synchronization counts reduced to
approximately `ceil(rounds / 8)` without changing the output hash:

```bash
for K in 1 8; do
  rocprofv3 --runtime-trace --stats \
    --output-directory "$OUT/rocprof-k$K" -- \
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers 1 \
      --bf11-segment-rounds "$K" \
      --bf11-hip-graph off \
      --net-limit 100 \
      --allow-unrouted
done
```

Run the contest wrapper/checker/wirelength/score pipeline once each for the
recorded pre-change baseline, the optimized K=1 Graph-off correctness control,
and the K=8 Graph candidate. `CheckPhysNetlist`
is the full route-legality and resource-use/congestion gate; `run-PathFinderFile`
also runs the wirelength analyzer and `compute-score.py`. The exact route and
wirelength comparisons below make the QoR gate stronger than score equality.
The `/usr/bin/time` records cover the complete conversion, routing, and route
reconstruction wrapper, while the five-run inner medians above remain the BF11
performance acceptance measurement:

```bash
set -euo pipefail
QOR_ROOT=amd-validation/bf11-qor
mkdir -p "$QOR_ROOT"
test -x /tmp/PathFinderFile-bf11
test -x /tmp/pathfinder-bf11-baseline
test -x /tmp/pathfinder-bf11-optimized
test -x ./interchange_to_csr
test -x ./routes_to_phys

for CONFIG in baseline control candidate; do
  WORK_DIR="$QOR_ROOT/work-$CONFIG"
  if [ "$CONFIG" = baseline ]; then
    BIN=/tmp/pathfinder-bf11-baseline
    BF11_ARGS="--parallel-net-workers 3 --strict-routing --work-dir $WORK_DIR"
  elif [ "$CONFIG" = control ]; then
    BIN=/tmp/pathfinder-bf11-optimized
    BF11_ARGS="--parallel-net-workers 3 --bf11-segment-rounds 1 \
--bf11-hip-graph off --bf11-adaptive-reset-threshold 0.25 \
--strict-routing --work-dir $WORK_DIR"
  else
    BIN=/tmp/pathfinder-bf11-optimized
    BF11_ARGS="--parallel-net-workers 3 --bf11-segment-rounds 8 \
--bf11-hip-graph on --bf11-adaptive-reset-threshold 0.25 \
--strict-routing --work-dir $WORK_DIR"
  fi

  rm -f logicnets_jscl_PathFinderFile.phys \
    logicnets_jscl_PathFinderFile.phys.log \
    logicnets_jscl_PathFinderFile.check \
    logicnets_jscl_PathFinderFile.check.log \
    logicnets_jscl_PathFinderFile.wirelength
  /usr/bin/time -f 'wall_seconds=%e\npeak_rss_kib=%M' \
    -o "$QOR_ROOT/$CONFIG.time" \
    make ROUTER=PathFinderFile BENCHMARKS="logicnets_jscl" VERBOSE=1 \
      PATHFINDER_ROUTER_BIN=/tmp/PathFinderFile-bf11 \
      PATHFINDER_BIN="$BIN" PATHFINDER_SSSP_ENGINE=bf11 \
      PATHFINDER_ARGS="$BF11_ARGS" run-PathFinderFile \
      2>&1 | tee "$QOR_ROOT/$CONFIG.make.log"

  test "$(cat logicnets_jscl_PathFinderFile.check)" = PASS
  cp logicnets_jscl_PathFinderFile.phys "$QOR_ROOT/$CONFIG.phys"
  cp logicnets_jscl_PathFinderFile.check "$QOR_ROOT/$CONFIG.check"
  cp logicnets_jscl_PathFinderFile.wirelength \
    "$QOR_ROOT/$CONFIG.wirelength"
  cp "$WORK_DIR/logicnets_jscl_PathFinderFile.routes.jsonl" \
    "$QOR_ROOT/$CONFIG.routes.jsonl"
done

sha256sum "$QOR_ROOT"/*.routes.jsonl
cmp "$QOR_ROOT/control.routes.jsonl" "$QOR_ROOT/candidate.routes.jsonl"
cmp "$QOR_ROOT/baseline.wirelength" "$QOR_ROOT/candidate.wirelength"
cat "$QOR_ROOT"/baseline.time "$QOR_ROOT"/control.time \
  "$QOR_ROOT"/candidate.time
```

Acceptance requires all of the following: exact output/path hashes against K=1
and CPU Dijkstra; no route-legality, congestion, wirelength, timing, or other
QoR regression; telemetry counts consistent with executed rounds and reset/cost
modes; clean direct fallback for unavailable HIP Graph support; and a median
end-to-end routing time at least 2x faster than the current three-worker BF11
baseline. Worker counts one and three are primary measurements; four is the
contention control. No speedup is inferred from host-only results.

The development host used for this change has no HIP compiler or AMD GPU, so
target runtime and acceptance medians must be recorded on the measured
`gfx1151` system rather than inferred locally.
