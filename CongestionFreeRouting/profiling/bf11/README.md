# BF11 AMD profiling harness

This harness targets a sudo-free AMD `gfx1151` installation. It does not alter
BF11 routing semantics or production defaults. All kernel work telemetry is
runtime-gated, and the profiler-free timing/counter processes use the existing
`CollectTelemetry=false` specializations.

## Safety and provenance

`run.py` refuses a dirty worktree unless `--allow-dirty` is explicit. When
allowed, it archives the complete tracked binary diff and every untracked file
under the profiling tree, including shared analyzer utilities. Each run records the branch, commit, full status, dirty
patch, build command, binary hashes, graph/metadata/manifest hashes, ROCm tool
versions, full `rocminfo`, metric catalogs/definitions, GPU clocks,
temperature, memory, visible GPU PIDs, and every profiler/application command.
It never invokes `sudo`.

An exact query manifest contains comma- or whitespace-separated zero-based
route-request indices; `#` starts a comment. Ordering is retained. Empty,
negative, duplicate, malformed, and out-of-range selections are rejected.
The application also accepts the same syntax with `--net-indices`.

Example:

```text
# small, medium, and long-query strata
7, 103, 991
18 244 1502
```

## Builds

```bash
python3 CongestionFreeRouting/profiling/bf11/run.py build \
  --output bf11-run/build --allow-dirty
```

The build creates three `-O3` binaries:

- `pathfinder-bf11-timing`: HIP Graph code enabled, no profiling dependency;
- `pathfinder-bf11-roctx`: HIP Graphs and ROCTx/query diagnostics enabled, with
  `rocprofiler-sdk-roctx` last on the link line;
- `pathfinder-bf11-thread-trace`: the trace build plus line-table source
  correlation for ATT.

The ordinary `pathfinder` target remains free of profiler headers and link
dependencies.

## Required run separation

Correctness, telemetry, tracing, counters, ATT, streaming reference, and
acceptance timing are separate processes. Never use telemetry-enabled kernels
for acceptance timing or for hardware-counter comparison with production.
Counter collection can serialize dispatches; it is therefore restricted to
one worker, `K=8`, Graph-off. Production timeline runs use three workers.

The controlled matrix is:

| Case | Workers | K | HIP Graph |
|---|---:|---:|---|
| production compatibility | 3 | 1 | off |
| production segmented | 3 | 8 | off |
| production graph replay | 3 | 8 | on |
| counter/microarchitecture | 1 | 8 | off |
| cooperative/default control | 1 | 1 | auto |

Run route equivalence before collection:

```bash
python3 CongestionFreeRouting/profiling/bf11/run.py correctness \
  --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
  --manifest queries.txt --output bf11-run/correctness
```

Run the complete suite (real `gfx1151` machine only):

```bash
python3 CongestionFreeRouting/profiling/bf11/run.py all \
  --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
  --manifest queries.txt --output bf11-run/profile
```

Individual actions are `timing`, `telemetry`, `trace`, `counters`, `compute`,
`thread-trace`, and `streaming`. `timing` performs five profiler-free
repetitions of each matrix row by default. `telemetry` emits thread-safe
`bf11_query_telemetry` JSON records keyed by original net and worker indices.

## Counter and memory rules

The harness captures the installed counter catalog and runs
`rocprofv3-avail pmc-check -d DEVICE ...` for every proposed group before
collection. It selects fixed core groups plus available WGP, GL0, GL1, GL2,
GCEA, command-processor, GRBM, VMEM/wait/stall, wave, issue, and occupancy
metrics by catalog name. Unsupported groups are recorded, never substituted by
zero.

`rocprof-compute` report IDs vary across releases, so the harness captures the
installed metric/report catalog and uses the profiler's full default
collection. It also archives the profiler's default derived report for each
kernel family. Analysis inventories System Speed-of-Light, Memory Chart, WGP, GL0,
GL1, GL2, GCEA, command-processor, and GRBM sections by their captured names,
not hard-coded block numbers.

`FETCH_SIZE` and `WRITE_SIZE` units must be established from captured metric
metadata and the known-size `streaming_copy` reference. The analyzer tests
bytes and KiB explicitly and rejects ambiguous units. This prevents the
historical KiB-as-bytes error. Request/cache-line traffic is labeled as such:
GL2-to-fabric requests are never called physical DRAM bandwidth unless a
captured GCEA or memory-controller definition directly supports it. Unique
application payload, cache/request amplification, replay/coherence traffic,
and physical traffic remain separate concepts.

## Analysis

Analyze a complete harness capture (individual subdirectories also work):

```bash
python3 CongestionFreeRouting/profiling/bf11/analyze.py bf11-run/profile
```

The analyzer supports rocprofv3 CSV layouts and RocPD views (`--rocpd`). It
classifies BF11 reset, seed, segmented relaxation, cooperative controller,
target check, prefix, reconstruction, runtime copy/fill, and graph activity.
It reports interval unions and time at concurrency levels zero through N;
additive multi-queue duration is labeled separately from wall time. It also
reports per-query kernel/API/copy/synchronization counts, synchronize latency
percentiles, copy direction/size/engine duration and transfer classification,
graph launches/uploads/direct segments/no-op rounds/fallbacks, wave/resource
data, request rates, and sampled ATT wait/stall/instruction hotspots.

Strict checks reject negative intervals, impossible percentages, invalid
zero counters for working dispatches, ambiguous joins, missing traffic units,
inconsistent dispatch/query counts, materially variant replays, and mixed
configuration fingerprints. Eligible-wave metrics are reported only when the
installed catalog exposes them; otherwise ATT wave state is the narrower
diagnostic. Occupancy is never treated as issue efficiency, instruction
density is not percent-of-peak arithmetic utilization, and bandwidth pressure
is distinguished from irregular-memory latency/dependency stalls.

## Tests and remaining real-device commands

Local host/fake-HIP validation:

```bash
python3 -m unittest discover -s CongestionFreeRouting/profiling/bf11/tests -p 'test_*.py'
c++ -std=c++17 -O2 CongestionFreeRouting/profiling/bf11/tests/query_selection_test.cpp \
  -o /tmp/bf11-query-selection-test && /tmp/bf11-query-selection-test
python3 CongestionFreeRouting/tests/bf11_fake_hip_build_test.py
python3 CongestionFreeRouting/tests/bf11_sparse_reset_source_test.py
```

Commands that require the real target remain:

```bash
make bf11-profile-binaries
# Full HIP correctness regression:
hipcc -std=c++17 -O2 -pthread -x hip -DBF11_NO_MAIN -DBF11_ENABLE_HIP_GRAPHS \
  -I HIP_kernel/bellman_ford/src -I CongestionFreeRouting/bellman_ford \
  CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp \
  CongestionFreeRouting/bellman_ford/bf11.cpp -o /tmp/bf11-hip-test
/tmp/bf11-hip-test

# Route equivalence gate:
python3 CongestionFreeRouting/profiling/bf11/run.py correctness \
  --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
  --manifest queries.txt --output bf11-run/correctness

# Counter compatibility checks plus compatible raw passes:
python3 CongestionFreeRouting/profiling/bf11/run.py counters \
  --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
  --manifest queries.txt --output bf11-run/counters

# System SOL/Memory Chart, runtime trace, ATT, and streaming reference:
for action in compute trace thread-trace streaming; do
  python3 CongestionFreeRouting/profiling/bf11/run.py "$action" \
    --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
    --manifest queries.txt --output "bf11-run/$action"
done

# Profiler-free acceptance matrix (run only after correctness passes):
python3 CongestionFreeRouting/profiling/bf11/run.py timing \
  --csr design.csrbin --metadata design.csrbin.ifmeta.bin \
  --manifest queries.txt --output bf11-run/timing
```

The generated `commands.jsonl` is the authoritative exact command record for
the installed ROCm release and chosen artifacts.
