# Delta controller AMD-server validation

`profiling/validate_delta_controller_rocm.sh` is the acceptance harness for the
generic HIP Delta-Stepping controller change. It builds candidates outside the
repository root, runs the required smoke and full correctness gates, measures
unprofiled wall time, and analyzes full `rocprofv3` RocPD traces.

The harness is intentionally fail-closed. A route JSONL validator is
supplemental; the repository's normal `CheckPhysNetlist` Java/Vivado flow is the
primary physical correctness gate.

## Recommended invocation

Run from the repository root on the AMD server. The script refuses another
working directory.

```bash
cd /home/jovyan/fpga24_routing_contest/rips2026-amd

bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --phase 1
```

The first line of harness output identifies a unique run directory such as:

```text
/home/jovyan/fpga24_routing_contest/rips2026-amd-validation/runs/delta-controller-20260729T190000Z-12345
```

The default root is a sibling of the checkout, keeping the harness's own
artifacts out of the Git status captured during preflight.

Use that exact absolute directory to resume. Phase 2 cannot run until phase 1
has passed, and phase 3 cannot run until phase 2 has passed.

```bash
bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --resume /absolute/path/to/delta-controller-20260729T190000Z-12345 \
  --phase 2

# Stop unrelated GPU workloads and verify the server is idle first.
bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --resume /absolute/path/to/delta-controller-20260729T190000Z-12345 \
  --phase 3 \
  --confirm-gpu-idle
```

A one-shot run is supported, although staged operation makes the GPU-idle
assertion easier to audit:

```bash
bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --phase all \
  --confirm-gpu-idle
```

For large traces, placing the append-only run outside the checkout is useful:

```bash
bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --phase 1 \
  --run-root /scratch/$USER/delta-controller-validation
```

All paths and thresholds are saved in `state/config.sh`. They cannot be changed
while resuming; start a fresh run when a binary, input, source file, route-count
expectation, or timing policy changes.

## Prerequisites

The default configuration expects these existing repository artifacts:

- `logicnets_jscl_unrouted.phys`
- `logicnets_jscl.netlist`
- `xcvu3p.full-poc-base-wire.devicegraph`
- executable `interchange_to_csr` and `routes_to_phys` helpers
- a one-line, nonempty `java-classpath.txt` with explicit file/directory entries
  (empty components and the repository root itself are rejected)

The server needs a working AMD ROCm device (`/dev/kfd` and `rocminfo`), `hipcc`,
`g++`, Python 3, Java, GNU core tools, and the normal contest checker/Vivado
environment. Phase 3 additionally requires GNU `/usr/bin/time`, a working
`rocprofv3`, and at least 20 GiB free as a configured floor. The actual phase 3
disk gate is the larger of that floor and a projection based on the completed
phase 2 case size for all planned full cases plus a 40 GiB RocPD reserve. The
estimate is printed and saved before timing starts. The script does not download
benchmarks, compile Java, generate a device graph, or install dependencies.

Use fresh-run options to override paths or tools when the server layout differs:

```bash
bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh \
  --phase 1 \
  --input-phys /data/logicnets_jscl_unrouted.phys \
  --logical-netlist /data/logicnets_jscl.netlist \
  --device-graph /data/xcvu3p.full-poc-base-wire.devicegraph \
  --interchange-to-csr /opt/router-tools/interchange_to_csr \
  --routes-to-phys /opt/router-tools/routes_to_phys \
  --java-classpath /data/java-classpath.txt
```

Run `bash CongestionFreeRouting/profiling/validate_delta_controller_rocm.sh
--help` for every supported override. Commands such as `--hipcc` are one
executable name or path, not a shell fragment.

## What the phases do

### Phase 1: preflight, builds, low-level tests, and smoke matrix

Preflight records UTC time, hostname, Git HEAD/status/diffs/untracked files,
tool versions, CPU/GPU information, input hashes, and disk state. A
`rocprofv3 --version` probe is recorded when the profiler exists but is not a
phase 1/2 requirement. The harness fingerprints relevant source and validation
files; a changed fingerprint makes resume fail rather than mix results from two
source states. It also resolves every Java classpath file/directory and terminal
JAR wildcard from the repository root and fingerprints the actual checker
artifacts, so replacing a checker JAR between phases invalidates resume.

Before building, existing repository-root `PathFinderFile` and `pathfinder`
binaries are copied into the run with size, timestamps, and SHA-256 hashes. They
are never executed as candidates or overwritten. The following new binaries are
built under the unique run directory:

- normal `PathFinderFile-candidate`
- normal `pathfinder-candidate`
- host policy, argument forwarding, and CPU-stub tests
- UnitBFS HIP and Delta-Stepping HIP tests

The low-level sequence includes the Python forwarding test, ordinary HIP tests,
UnitBFS sequential workspace-reuse stress, and reduced-controller multi-queue
stress. The default stress count is 1,200 and is configurable with
`--stress-runs` on a fresh run. GNU `timeout` owns the noninteractive command
process group so wrapper-spawned children are also terminated; each low-level
test fails nonzero after 60 minutes by default, while smoke, full, and profile
defaults are 120, 720, and 1,440 minutes respectively. Fresh-run
`--test-timeout-minutes`, `--smoke-timeout-minutes`,
`--full-timeout-minutes`, and `--profile-timeout-minutes` overrides must all be
positive. Timeout exit status is retained in the attempt log.

The exact 500-route smoke matrix is:

| Case | Controller | Requested batch | Workers |
|---|---|---:|---:|
| `host-w1` | `host-checked` | default 4 | 1 |
| `host-w4` | `host-checked` | default 4 | 4 |
| `reduced-b2w1` | `reduced-round-trip` | 2 | 1 |
| `reduced-b4w1` | `reduced-round-trip` | 4 | 1 |
| `reduced-b2w4` | `reduced-round-trip` | 2 | 4 |
| `reduced-b4w4` | `reduced-round-trip` | 4 | 4 |
| `reduced-b8w4` | `reduced-round-trip` | 8 | 4 |
| `reduced-b4w8` | `reduced-round-trip` | 4 | 8 |

The host controller deliberately receives no
`--delta-controller-batch-size` argument: PathFinder rejects that combination.
Its requested/default batch is 4 while its effective batch is 1, and the
telemetry gate checks both semantics.

#### Why phase 1 bootstraps one full host reference

`CheckPhysNetlist` invokes Vivado route-status checking and requires a fully
routed physical netlist. A raw `--net-limit 500` output leaves the other nets
unrouted and therefore cannot honestly pass the normal checker.

Phase 1 resolves this without weakening the gate or adding hidden external
inputs:

1. `interchange_to_csr` runs once to create one immutable CSR/metadata pair.
2. A full host-checked, four-worker candidate run on that pair is summarized,
   depth-checked, reconstructed, and required to pass `CheckPhysNetlist`.
3. Every 500-route smoke PathFinder process reads the same immutable CSR and
   metadata, so every route record has the same `artifact_pair_id`.
4. `splice_routes_jsonl.py` replaces exactly 500 records in the full reference
   by unique net name, preserves the reference record order, rejects duplicate
   or missing nets, and rejects any artifact-pair mismatch.
5. The spliced artifact must contain all 27,960 expected routes, have exact
   maximum depth 214, reconstruct successfully, and pass `CheckPhysNetlist`.

The bootstrap costs one extra full host run before the smoke matrix. It is not a
candidate correctness oracle for the 500 replaced nets; it only supplies the
untouched nets needed to make a checkable physical artifact. Each selected
candidate record is retained verbatim, the splice provenance and input/output
SHA-256 values are saved, and the normal checker catches conflicts in the full
combined design. Pair IDs are never rewritten or bypassed.

Each smoke case also independently requires:

- exactly 500 routed route records and every sink reached;
- strict schema-v4 aggregate telemetry for the requested/effective controller;
- zero fallback queries, controller corruption/stale-publication counters, and
  HIP errors;
- actual bounded device work for reduced mode;
- the same request/source/sink/reached endpoint view as `host-w1`;
- fewer controller round trips than the matching one- or four-worker host case
  where that host case exists;
- no fatal HIP/HSA/GPU-fault diagnostics in logs.

The raw route-tree SHA is recorded but is not an equality gate. Equal-distance
parent ties may legitimately change edges or a sink's selected source. Endpoint
comparison therefore checks net, routed state, every source node/site/pin, and
every sink node/site/pin/reached value while deliberately ignoring route edges,
`sink.source`, and `artifact_pair_id`.

### Phase 2: full correctness only

Phase 2 runs exactly these full, telemetry-enabled cases through the complete
wrapper/conversion/reconstruction flow:

| Case | Controller | Requested batch | Workers |
|---|---|---:|---:|
| `host-w4` | `host-checked` | default 4 | 4 |
| `reduced-b4w4` | `reduced-round-trip` | 4 | 4 |

Both must have exactly 27,960 fully routed records, every sink reached, maximum
depth exactly 214, and a physical checker `PASS`. Reduced telemetry must prove
the requested mode was effective, bounded device work occurred, and no fallback
or controller error occurred. The reduced endpoint view must match the full
host output. Phase 2 contains no profiler and no acceptance timing.

### Phase 3: unprofiled timing and RocPD analysis

Phase 3 starts only after phase 2 passes and only with
`--confirm-gpu-idle`. That flag is the operator's assertion that unrelated GPU
jobs have been stopped; the harness also snapshots `rocm-smi` when available.

Unprofiled end-to-end timing uses the normal non-ROCTx candidate and GNU time.
Validation and checking occur after the timed interval. Each configuration gets
one full warmup and five full repetitions by default (never fewer than three):

- host-checked, four workers;
- reduced batch 2, four workers;
- reduced batch 4, four workers;
- reduced batch 8, four workers.

Every warmup and repetition still passes the physical checker, strict route
count, depth, and endpoint gates. Medians are calculated from unprofiled wall
times; profiler wall time is never presented as the performance result. The
reduced-b4 median may not regress more than 5% versus host by default. Change
that fresh-run policy with `--max-timing-regression-percent`.

The harness separately builds `pathfinder-roctx-candidate` with
`PATHFINDER_ENABLE_ROCTX` and profiles full host-w4 and reduced-b4w4 runs using:

```text
rocprofv3 --runtime-trace --stats --output-format rocpd --output-directory ... --
```

Profiling applies only to the inner PathFinder process. The resulting SQLite
database is selected only when it contains the required nonempty RocPD views.
`analyze_rocpd.py` receives the already-gated route summary through
`--routes-summary`; a caller-supplied integer cannot become the acceptance
denominator. The `pathfinder.route_net` marker count must equal the completed
route count. Generic and compact-extraction marker counts must be positive and
equal to each other; they may be lower because a valid net whose sinks are
already in its source/tree needs no SSSP query.

The synchronization report requires the disjoint generic, compact extraction,
and other scopes to sum to both the scoped total and the HIP API
`hipStreamSynchronize` total. Profile logs are also rejected on nonzero dropped
record/event/sample, trace data-loss/incomplete, or buffer overflow/overrun
diagnostics. Reduced-b4w4 passes only at:

- total synchronizations per route `<= 15.0`;
- generic-controller synchronizations per route `<= 10.0`.

Compact extraction and other synchronizations are always reported separately.
The host trace is analyzed with the same accounting checks but is not subject to
the reduced-controller numerical threshold.

## Evidence and resume behavior

The run directory is append-only at the attempt level:

```text
RUN_DIR/
  state/                 immutable config, fingerprints, done markers, events
  attempts/              preflight, preserved binaries, builds, low-level tests
  cases/                 every route/check/profile attempt and its work files
  profile-analysis/      selected database, analyzer output, synchronization gate
  reports/               phase summaries, timing medians, acceptance gates
```

Every retry allocates a new UTC/PID attempt directory. A `.done` marker is
written only after all gates for that step pass. Each referenced result also
contains a read-only success manifest; resume verifies every recorded file is
still present with its recorded size and SHA-256, rejects missing or unexpected
files, and recursively validates every build/test/case/profile dependency of a
completed phase before skipping it. Each parent pins the child `.done` marker's
SHA-256, so replacing a marker with a different valid attempt also invalidates
the phase. All failed attempts remain in place. There is no `rm`, `git clean`,
`git reset`, Make clean target, or repository-root candidate output.

For a failure, inspect the context printed at exit and then the newest attempt's
logs, `fatal-matches.txt`, checker marker/log, telemetry summary, route/depth
summaries, and endpoint or splice provenance JSON. Correct the external problem
without changing the recorded source/input state, then resume the same phase. If
source or input content must change, start a fresh run so evidence is not mixed.

At completion, the phase reports contain the case paths and raw route SHAs,
requested/effective telemetry, endpoint comparison policy, timing repetitions
and medians, profiler scope counts/rates, compact extraction count, acceptance
limits, and pass/fail decisions needed for the final engineering report.
