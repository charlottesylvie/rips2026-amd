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

### HIP Graph recovery

The high-confidence source-level explanation for the reported error is
cross-worker invalidation of `hipStreamCaptureModeGlobal`; confirmation on the
original ROCm target is still pending. AMD CLR registers Global captures in a
process-wide list, and its unsafe-call checks can invalidate captures from
other threads. ThreadLocal captures remain in the owning thread's capture list.
Each BF11 workspace begins and ends capture synchronously in the same worker,
so BF11 now uses `hipStreamCaptureModeThreadLocal`, retaining same-thread unsafe
call checking without cross-worker registration. See AMD's official
[`hip_graph.cpp`](https://github.com/ROCm/clr/blob/develop/hipamd/src/hip_graph.cpp),
[`hip_stream.cpp`](https://github.com/ROCm/clr/blob/develop/hipamd/src/hip_stream.cpp),
and [HIP Graph API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html).

Begin, enqueue, end, and instantiation failures terminate capture, release
every partially returned object, record one sticky fallback per workspace, and
then enqueue the same segment directly. A captured-kernel error is recoverable
only when its typed status is capture-specific; invalid kernel configuration,
invalid device function, and device errors are propagated. Cleanup failures
are also surfaced rather than losing resource ownership silently. A
pre-existing or different asynchronous error is never cleared as a Graph
fallback.

A real `hipGraphLaunch` error is treated more conservatively because AMD may
have submitted a prefix of graph nodes before returning it. BF11 quiesces the
stream, destroys the cached graph, resets and seeds the whole query again, and
then runs sticky-direct. If quiescence itself reports an asynchronous device
failure, the query is not replayed. For deterministic policy validation,
`begin`, `end`, `instantiate`, and `launch` in
`BF11_TEST_HIP_GRAPH_FAILURE_STAGE` are synthetic stage failures. `enqueue`
calls a capture-prohibited synchronous stream wait while capture is active,
forcing the runtime's real invalidation/error path; EndCapture must then return
the stream to non-capturing state before direct fallback.
`launch-after-submit` performs a real graph launch, then deliberately enters
the quiesce and whole-query restart path so target tests cover the
non-atomic-launch recovery integration.

### Packed-state and latency-hiding audit

The proposed three logical relaxation fields are already represented without a
third graph-sized array. `best_state[V]` atomically packs float distance bits
and predecessor edge into one aligned 64-bit word; infinity-to-finite ownership
derives discovery, and `(0, no-predecessor)` derives a source root. The old
source mask is absent. `next_marks[V]` remains separate because its generation
lifetime and write frequency differ from labels; combining it would require a
larger atomic word or pull an otherwise cold mark into every failed relaxation.
The implementation now has explicit predecessor unpacking and alignment
assertions, with no change to the 24-byte-per-vertex identity workspace.

The segmented relaxation hot path has the following logical global accesses;
physical transactions depend on wave coalescing and cache residency and must
not be counted one-for-one with source-level loads:

| Scope | Unconditional access | Conditional access and locality |
| --- | --- | --- |
| Frontier item | 4-byte contiguous `current[item]`; coherent 8-byte `best_state[from]`; two adjacent 4-byte `rowptr[from/from+1]` reads | Source indices make state/row lookups irregular across a wave, although repeated sources can hit cache. |
| Examined edge | 4-byte `to[edge]`; coherent 8-byte destination state | CSR edge ranges are sequential within a vertex but different lanes can occupy disjoint ranges; destination state is irregular. |
| Bounds/cost | Bounded mode reads two 4-byte SoA coordinates; static mode reads one sequential 4-byte edge cost; dynamic mode adds one irregular 4-byte vertex multiplier; constant-one reads neither cost array | Coordinates are always paired only for bounded queries. Packing them is plausible but still 8 bytes/vertex and needs target transaction evidence before changing layout. |
| Strict improvement | One or more 8-byte state CAS operations; 4-byte generation-mark load and possible CAS; queue/touched writes | First publications are block-compacted to one tail reservation per block batch; overflow publications use direct atomics. |

HIP allocations provide the base alignment required by the naturally aligned
8-byte packed state, while consecutive words preserve it. A 32-lane GFX11
wave therefore can coalesce contiguous queue/CSR traffic into aligned memory
transactions, but destination-indexed state and degree-skewed CSR ranges remain
irregular. SoA remains preferable: unbounded and constant-one queries avoid
coordinates and cost arrays entirely, and failed relaxations do not fetch the
cold generation mark. Folding `next_marks` into state would require a wider
atomic object (or an extra word fetched on every edge), increase contention,
and complicate generation rollover without reducing the 24-byte identity
workspace.

The segmented grid is at most 256 blocks of 256 threads: eight 32-lane waves
per block on GFX11 and up to 65,536 vertex lanes per query. The grid is sized
from `V`, while useful blocks in a round are bounded by the current frontier;
`F <= 256` therefore uses only one block, and `F > 65,536` grid-strides. One
lane serially walks a vertex's entire edge range, so a skewed high-degree
vertex can hold its block at the publication barriers. The visible LDS footprint
is small, but VGPR-limited occupancy is unknown without target compilation.
Three independent worker streams already expose separate query frontiers and
were the smallest measured plateau point. A wave-per-vertex mapping, batched
mutable query state, or persistent multi-query queue would be an unmeasured
scheduling, memory, exception-recovery, and determinism change. No such
redesign is enabled without gfx1151 occupancy, register, LDS, queue-overlap,
and memory-stall measurements.

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
  --bf11-adaptive-reset-threshold 0.25
```

Run the full controller matrix with telemetry omitted for one warm-up and five
wall-time samples. Worker four is a contention control, not an automatic-policy
candidate. Graph `on` at K=1 must remain a direct/fallback compatibility case:

```bash
set -euo pipefail
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
set -euo pipefail
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
set -euo pipefail
TARGET_ARCH=${TARGET_ARCH:-gfx1151}
git rev-parse HEAD
git diff --check
hipcc --version
rocminfo > /tmp/bf11-rocminfo.txt
grep -m1 -E 'Name:.*gfx' /tmp/bf11-rocminfo.txt
COMMON_FLAGS=(-std=c++17 -O3 -x hip -DBF10_NO_MAIN -DBF11_NO_MAIN \
  "--offload-arch=$TARGET_ARCH" \
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
test /tmp/pathfinder-bf11-optimized -nt \
  CongestionFreeRouting/bellman_ford/bf11.cpp
test /tmp/pathfinder-bf11-optimized -nt \
  CongestionFreeRouting/pathfinder.cpp
test /tmp/pathfinder-bf11-optimized -nt \
  CongestionFreeRouting/bellman_ford/bf11_graph_execution_policy.hpp
stat /tmp/pathfinder-bf11-optimized \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  CongestionFreeRouting/pathfinder.cpp \
  CongestionFreeRouting/bellman_ford/bf11_graph_execution_policy.hpp
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

Use an already converted CSR v3/v4 metadata pair. Omitting `--routes-out` prevents
route-output metadata loading; conversion is not part of these runs. The
following K=1, Graph-off run isolates the relaxed atomic load from its CAS
control with one warm-up and five measured repetitions:

```bash
set -euo pipefail
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
        >"$OUT/$MODE-w$WORKERS-r$REP.log" 2>&1
    done
  done
done
```

Record the pre-change three-worker BF11 baseline with the same graph, one
warm-up, five measured repetitions, and telemetry disabled:

```bash
set -euo pipefail
for REP in 0 1 2 3 4 5; do
  /tmp/pathfinder-bf11-baseline "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers 3 \
    >"$OUT/baseline-w3-r$REP.log" 2>&1
  /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers 3 \
    --bf11-segment-rounds 8 \
    --bf11-hip-graph on \
    --bf11-adaptive-reset-threshold 0.25 \
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
set -euo pipefail
for WORKERS in 1 3 4; do
  for REP in 1 2 3 4 5; do
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers "$WORKERS" \
      --bf11-segment-rounds 1 \
      --bf11-hip-graph off \
      --bf11-telemetry \
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
set -euo pipefail
c++ -std=c++17 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
  CongestionFreeRouting/tests/bf11_graph_execution_policy_test.cpp \
  -o /tmp/bf11_graph_execution_policy_test
/tmp/bf11_graph_execution_policy_test

hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
  -DBF11_ENABLE_HIP_GRAPHS \
  "--offload-arch=${TARGET_ARCH:-gfx1151}" \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test
/tmp/bf11_bounded_dynamic_hip_test

hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
  -DBF11_ENABLE_HIP_GRAPHS \
  -DBF11_FORCE_CAS_ATOMIC_LOAD \
  "--offload-arch=${TARGET_ARCH:-gfx1151}" \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test_cas
/tmp/bf11_bounded_dynamic_hip_test_cas
```

Run the requested strict PathFinder matrix below. Strict routing is the direct
binary's default: deliberately do **not** pass `--allow-unrouted`. Set
`RUN_W4=1` only when target memory permits. The HIP regression above is the
independent CPU Dijkstra oracle and includes constant-one, general-static,
dynamic-cost, bounded, unbounded, fallback, zero-weight, multi-source, and
multi-target cases.

```bash
set -euo pipefail
STRICT_OUT=${STRICT_OUT:-/tmp/bf11-strict-matrix}
RUN_W4=${RUN_W4:-0}
export RUN_W4
mkdir -p "$STRICT_OUT"
CONFIGS=(
  "control 1 1 off"
  "w1-k8-off 1 8 off"
  "w1-k8-on 1 8 on"
  "w3-k8-off 3 8 off"
  "w3-k8-auto 3 8 auto"
  "w3-k8-on 3 8 on"
)
if [ "${RUN_W4:-0}" = 1 ]; then
  CONFIGS+=("w4-k8-off 4 8 off" "w4-k8-auto 4 8 auto" \
            "w4-k8-on 4 8 on")
fi

for SPEC in "${CONFIGS[@]}"; do
  read -r LABEL WORKERS K HIP_GRAPH <<<"$SPEC"
  /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers "$WORKERS" \
    --bf11-segment-rounds "$K" \
    --bf11-hip-graph "$HIP_GRAPH" \
    --bf11-adaptive-reset-threshold 0.25 \
    --bf11-telemetry \
    --routes-out "$STRICT_OUT/$LABEL.routes.jsonl" \
    >"$STRICT_OUT/$LABEL.log" 2>&1
done

python3 - "$STRICT_OUT" <<'PY'
import hashlib, json, os, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
control = (root / "control.routes.jsonl").read_bytes()
labels = ["control", "w1-k8-off", "w1-k8-on", "w3-k8-off",
          "w3-k8-auto", "w3-k8-on"]
if os.environ.get("RUN_W4", "0") == "1":
    labels += ["w4-k8-off", "w4-k8-auto", "w4-k8-on"]
for label in labels:
    routes_path = root / f"{label}.routes.jsonl"
    assert routes_path.is_file(), routes_path
    rows = [json.loads(line) for line in routes_path.read_text().splitlines()]
    reached = sum(sink.get("reached") is True
                  for row in rows for sink in row["sinks"])
    total = sum(len(row["sinks"]) for row in rows)
    assert total > 0 and reached == total, (routes_path.name, reached, total)
    assert all(row.get("routed") is True for row in rows), routes_path.name
    data = routes_path.read_bytes()
    assert data == control, f"route mismatch: {routes_path.name}"
    records = [json.loads(line)
               for line in (root / f"{label}.log").read_text().splitlines()
               if line.startswith("{")]
    runtime = next(row for row in records
                   if row.get("type") == "bf11_runtime_stats")
    telemetry = next(row for row in records
                     if row.get("type") == "bf11_telemetry")
    expected = (1, 1, "off") if label == "control" else (
        lambda match: (int(match.group(1)), int(match.group(2)), match.group(3))
    )(re.fullmatch(r"w(\d+)-k(\d+)-(off|auto|on)", label))
    workers, rounds, graph_mode = expected
    assert runtime["requested_workers"] == workers, label
    assert runtime["effective_workers"] == workers, label
    assert runtime["segment_rounds"] == rounds, label
    assert runtime["hip_graph"] == graph_mode, label
    assert telemetry["requested_workers"] == workers, label
    assert telemetry["effective_workers"] == workers, label
    assert telemetry["configuration"]["segment_rounds"] == rounds, label
    assert telemetry["configuration"]["hip_graph"] == graph_mode, label
    assert telemetry["queries"] > 0, label
    assert telemetry["queries"] == telemetry["completed_queries"], label
    work = telemetry["work"]
    assert work["direct_segments"] + work["hip_graph_segments"] == work["segments"], label
    if runtime["hip_graph"] == "off":
        assert work["direct_segments"] == work["segments"], label
        assert work["hip_graph_segments"] == 0 and work["graph_fallbacks"] == 0, label
    elif runtime["segment_rounds"] > 1:
        assert work["hip_graph_segments"] > 0 or work["graph_fallbacks"] > 0, label
        if work["hip_graph_segments"] == 0:
            assert work["direct_segments"] == work["segments"], label
    print(label, hashlib.sha256(data).hexdigest(),
          f"reached={reached}/{total}", runtime,
          {"queries": telemetry["queries"], "work": work})
PY
```

Force active-capture invalidation, require a successful sticky-direct run, and
compare its route output with the K=1 Graph-off control. The concurrent K=8 HIP
regression separately tests actual cross-worker first capture and reuse:

```bash
set -euo pipefail
BF11_TEST_HIP_GRAPH_FAILURE_STAGE=enqueue \
  /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
    --sssp-engine bf11 \
    --parallel-net-workers 3 \
    --bf11-segment-rounds 8 \
    --bf11-hip-graph on \
    --bf11-telemetry \
    --routes-out "$STRICT_OUT/forced-enqueue.routes.jsonl" \
    >"$STRICT_OUT/forced-enqueue.log" 2>&1
cmp "$STRICT_OUT/control.routes.jsonl" \
  "$STRICT_OUT/forced-enqueue.routes.jsonl"
python3 - "$STRICT_OUT/forced-enqueue.log" <<'PY'
import json, pathlib, sys
rows = [json.loads(line)
        for line in pathlib.Path(sys.argv[1]).read_text().splitlines()
        if line.startswith("{")]
runtime = next(row for row in rows if row.get("type") == "bf11_runtime_stats")
telemetry = next(row for row in rows if row.get("type") == "bf11_telemetry")
work = telemetry["work"]
assert telemetry["queries"] > 0
assert telemetry["queries"] == telemetry["completed_queries"]
assert work["segments"] > 0 and work["direct_segments"] == work["segments"]
assert work["hip_graph_segments"] == 0 and work["graph_fallbacks"] >= 1
assert work["stream_synchronizations"] >= (
    work["segments"] + telemetry["queries"] + work["graph_fallbacks"])
print(runtime)
print(telemetry)
PY
```

Collect HIP API counts for K=1 and K=8 with the same one-worker query prefix.
The K=8 trace must show status-copy and stream-synchronization counts reduced to
approximately `ceil(rounds / 8)` without changing the output hash:

```bash
set -euo pipefail
for K in 1 8; do
  rocprofv3 --runtime-trace --stats \
    --output-directory "$OUT/rocprof-k$K" -- \
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers 1 \
      --bf11-segment-rounds "$K" \
      --bf11-hip-graph off \
      --net-limit 100 \
      --routes-out "$OUT/rocprof-k$K.routes.jsonl"
done
cmp "$OUT/rocprof-k1.routes.jsonl" "$OUT/rocprof-k8.routes.jsonl"
sha256sum "$OUT"/rocprof-k*.routes.jsonl
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
