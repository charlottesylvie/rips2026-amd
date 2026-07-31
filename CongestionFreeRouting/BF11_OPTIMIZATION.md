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
state. Touched/frontier reservations happen only after a successful relaxation,
so they are lower-frequency and do not yet justify wave aggregation. The
implemented optimization replaces that unconditional observation with
`__hip_atomic_load(..., __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)` when the
compiler supports it. `BF11_FORCE_CAS_ATOMIC_LOAD` and compilers without the
intrinsic retain the proven CAS load. The successful compare/exchange still
decides exact infinity-to-finite first discovery.

No segmented batch, dense-reset threshold, degree-hybrid kernel, or overlapping
cooperative grid is enabled without BF11 phase evidence. The telemetry below is
the decision gate for any follow-up.

## Telemetry

Pass `--bf11-telemetry` to emit one `bf11_telemetry` JSON record after all
workers join. It is disabled by default. Timing fields are aggregate
nanoseconds:

- `reset_seed_gpu`, `relaxation_gpu`, and `target_check_gpu` measure device
  phases;
- `iteration_status_copy_gpu` measures device-to-host controller/status copies;
- `stream_synchronize_cpu` measures blocking host synchronization;
- `target_summary_gpu` and `path_reconstruction_gpu` cover compact result
  extraction;
- `total_query_cpu` is the sum of complete query durations.

The explicit-stream host controller measures GPU phases with HIP events. The
null-stream cooperative controller measures its internal phases with
`wall_clock64()` and converts ticks using
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
for that overlap and for compact paths. Requested worker count uses zero for
automatic selection; effective count reflects route-count and external-stream
limits.

GPU and query times are sums across queries and can exceed routing wall time
when workers overlap. Compare phase shares against their aggregate, and use
`bf11_runtime_stats.routing_seconds` for end-to-end BF11 backend acceptance.
Interpret a high `stream_synchronize_cpu` or status-copy share as evidence for a
future three/four-query segmented controller; a high relaxation share with many
edges per successful relaxation supports further atomic or degree-balancing
work; a high reset share should be read together with maximum touched fraction;
a high summary/reconstruction share supports reconstruction-chain reduction.

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

hipcc "${COMMON_FLAGS[@]}" "${COMMON_SOURCES[@]}" -pthread \
  -o /tmp/pathfinder-bf11-optimized
hipcc "${COMMON_FLAGS[@]}" -DBF11_FORCE_CAS_ATOMIC_LOAD \
  "${COMMON_SOURCES[@]}" -pthread -o /tmp/pathfinder-bf11-cas-control
```

Use an already converted CSR v3/metadata pair. Omitting `--routes-out` prevents
route-output metadata loading; conversion is not part of these runs. Run one
warm-up plus five measured repetitions for every required worker count:

```bash
GRAPH=/absolute/path/design.csrbin
META=/absolute/path/design.csrbin.ifmeta.bin
OUT=/tmp/bf11-saturation
mkdir -p "$OUT"

for MODE in optimized cas-control; do
  BIN="/tmp/pathfinder-bf11-$MODE"
  for WORKERS in 1 3 4 8; do
    for REP in 0 1 2 3 4 5; do
      "$BIN" "$GRAPH" "$META" \
        --sssp-engine bf11 \
        --parallel-net-workers "$WORKERS" \
        --allow-unrouted \
        >"$OUT/$MODE-w$WORKERS-r$REP.log" 2>&1
    done
  done
done
```

Extract medians from repetitions one through five:

```bash
python3 - "$OUT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
for mode in ("optimized", "cas-control"):
    for workers in (1, 3, 4, 8):
        samples = []
        for rep in range(1, 6):
            path = root / f"{mode}-w{workers}-r{rep}.log"
            records = [json.loads(line) for line in path.read_text().splitlines()
                       if line.startswith("{")]
            record = next(r for r in records
                          if r.get("type") == "bf11_runtime_stats")
            samples.append(record["routing_seconds"])
        print(mode, workers, statistics.median(samples), samples)
PY
```

Acceptance requires the optimized one-worker median to be no more than 1.03
times the CAS-control one-worker median. The best optimized small-worker median
must be below the better of the CAS-control three- and four-worker medians.
Eight is a required contention/control point, not an automatic candidate.

Collect phase and memory telemetry separately so event overhead is not mixed
into the disabled-telemetry acceptance result:

```bash
for WORKERS in 1 3 4 8; do
  for REP in 1 2 3 4 5; do
    /tmp/pathfinder-bf11-optimized "$GRAPH" "$META" \
      --sssp-engine bf11 \
      --parallel-net-workers "$WORKERS" \
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
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test
/tmp/bf11_bounded_dynamic_hip_test

hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN \
  -DBF11_FORCE_CAS_ATOMIC_LOAD \
  -I HIP_kernel/bellman_ford/src \
  -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp \
  -o /tmp/bf11_bounded_dynamic_hip_test_cas
/tmp/bf11_bounded_dynamic_hip_test_cas
```

The development host used for this change has no HIP compiler or AMD GPU, so
target runtime and acceptance medians must be recorded on the measured
`gfx1151` system rather than inferred locally.
