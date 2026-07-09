# RIPS SSSP Validator

This folder contains a standalone validator for shortest-path outputs generated
from the RIPS `.csrbin` graph.

## Build

CPU/local build:

```bash
g++ -std=c++17 -O2 \
  -I SSSP/src \
  SSSP/src/sssp_validation.cpp \
  SSSP/test/validate_SSSP.cpp \
  -o validate_SSSP
```

The default validation modes can be changed at compile time:

```bash
-DSSSP_VALIDATOR_DEFAULT_MODES="\"1,2,3\""
```

## Run

```bash
./validate_SSSP graph.csrbin paths.jsonl --modes 1,2,3
./validate_SSSP graph.csrbin paths.jsonl --modes 1,2,3 --sample-rate 0.01 --seed 7
```

Modes:

| Number | Name | Meaning |
| --- | --- | --- |
| 1 | `path-exists` | Every concrete path edge exists in the CSR graph. |
| 2 | `path-cost` | Sum CSR edge weights along each path and compare against the reported path distance. |
| 3 | `path-shape` | Check source/target endpoints and edge continuity. |

## JSONL Format

Each line is one JSON object. Edge weights come from the `.csrbin` file. The
JSONL records describe selected source-target paths.

The CSR orientation is incoming-edge:

```text
row v, column u means directed edge u -> v
```

### Metadata Record

Optional, useful for humans and pipeline sanity checks:

```json
{"type":"metadata","format":"rips-sssp-paths-v1","node_count":28000000,"edge_orientation":"incoming","description":"row v, col u means directed edge u -> v"}
```

### Compact Path Record

Recommended for production:

```json
{"type":"path","source":12345,"target":67890,"reached":true,"distance":17.0,"nodes":[12345,22222,67890],"csr_edges":[901,902]}
```

The `distance` field is the algorithm's reported total path length. Mode 2
recomputes that total from CSR edge weights and compares the two values.

Rules:

- `nodes[0]` must equal `source`.
- `nodes[-1]` must equal `target`.
- `csr_edges.length` must equal `nodes.length - 1`.
- For edge `csr_edges[i]` from `nodes[i]` to `nodes[i+1]`, the validator checks:

```text
rowptr[nodes[i+1]] <= csr_edges[i] < rowptr[nodes[i+1] + 1]
colind[csr_edges[i]] == nodes[i]
```

`csr_edges` is strongly recommended because it disambiguates parallel edges.

### Verbose Debug Path Record

Also accepted:

```json
{"type":"path","source":12345,"target":67890,"reached":true,"distance":17.0,"edges":[{"from":12345,"to":22222,"csr_edge":901,"cost":8.0},{"from":22222,"to":67890,"csr_edge":902,"cost":9.0}]}
```

### Unreached Target Record

```json
{"type":"path","source":12345,"target":99999,"reached":false,"distance":null,"nodes":[],"csr_edges":[]}
```

## Bellman-Ford JSONL Generator

[src/bf_for_validation.cpp](src/bf_for_validation.cpp) reads a `.csrbin` graph
and its `.csrbin.ifmeta.bin` metadata sidecar, runs HIP Bellman-Ford from the
route sources listed in metadata, reconstructs paths to the listed sinks, and
writes JSONL records accepted by `validate_SSSP`.

Build on an AMD ROCm/HIP machine from `rips2026-amd`:

```bash
hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/minplus_mm/src \
  SSSP/src/bf_for_validation.cpp \
  HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
  HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
  -o bf_for_validation
```

Run:

```bash
./bf_for_validation graph.csrbin graph.csrbin.ifmeta.bin paths.jsonl
```

Example with smaller limits for a smoke test:

```bash
./bf_for_validation graph.csrbin graph.csrbin.ifmeta.bin paths.jsonl \
  --net-limit 10 \
  --source-limit 5 \
  --bf-progress-every 100
```

Then validate the generated paths:

```bash
./validate_SSSP graph.csrbin paths.jsonl --modes 1,2,3
```

Batched generator:

[src/bf_batched.cpp](src/bf_batched.cpp) uses the same `.csrbin`,
`.csrbin.ifmeta.bin`, and JSONL path format as `bf_for_validation`, but runs
Bellman-Ford for a batch of sources at once. Internally, the distance state is
an `N x B` sparse matrix, where `N` is the number of graph nodes and `B` is
`--batch-size`.

Build on an AMD ROCm/HIP machine from `rips2026-amd`:

```bash
hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/minplus_mm/src \
  SSSP/src/bf_batched.cpp \
  HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
  -o bf_batched
```

Run:

```bash
./bf_batched graph.csrbin graph.csrbin.ifmeta.bin paths.jsonl --batch-size 4
```

Example smoke test:

```bash
./bf_batched graph.csrbin graph.csrbin.ifmeta.bin paths.jsonl \
  --batch-size 4 \
  --net-limit 10 \
  --source-limit 8 \
  --bf-progress-every 100
```

The batched generator copies the CSR graph to the GPU once, sends only the
source IDs for each batch, keeps the batched distance matrix on the GPU during
relaxation, and copies back one dense `N x B` distance block per batch for
CPU-side path reconstruction and JSONL writing. Start with `--batch-size 2`,
`4`, or `8`; the dense host copy costs roughly `N * B * 4` bytes per batch.

Warm-started single-source generator:

[src/bf_warm_init.cpp](src/bf_warm_init.cpp) is an experimental variant
that first finds hop-2 source groups that share a connector node. As a
preprocessing step, it chooses connector groups, then processes one connector
and each of its related sources before moving to the next connector. Related
sources are routed one at a time with their distance vector initialized by
`edge_weight(source -> connector) + connector_dist[]`.
Connector paths are not written to JSONL; only source-to-sink path records are
emitted. Remaining sources are routed cold, one source at a time.

Build on an AMD ROCm/HIP machine from `rips2026-amd`:

```bash
hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/minplus_mm/src \
  SSSP/src/bf_warm_init.cpp \
  HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
  -o bf_warm_init
```

Run:

```bash
./bf_warm_init data/logicnets_jscl.csrbin data/logicnets_jscl.csrbin.ifmeta.bin paths.jsonl
```

Only directed edges `source -> connector` are used for warm starts, because
those produce valid upper bounds. Only one connector distance vector is kept at
a time.

## Source Proximity Checker

[src/check_sources.cpp](src/check_sources.cpp) checks whether route sources are
close enough for connector warm-start ordering to be useful. It reads the same
`.csrbin` and `.csrbin.ifmeta.bin` files, builds outgoing adjacency from the
incoming-edge CSR, then enumerates exact hop-2
`source1 - connector - source2` triples. Discovery always treats the two
source-connector adjacencies as undirected, then classifies the directed
orientation of the two edges.

Build from `rips2026-amd`:

```bash
g++ -std=c++17 -O3 \
  -I SSSP/src \
  SSSP/src/check_sources.cpp \
  -o check_sources
```

Run the exact hop-2 scan:

```bash
./check_sources graph.csrbin graph.csrbin.ifmeta.bin
```

Smoke-test options for a large graph:

```bash
./check_sources graph.csrbin graph.csrbin.ifmeta.bin \
  --source-limit 100 \
  --max-sets-print 50
```

Limit per-set output while still computing full orientation counts:

```bash
./check_sources graph.csrbin graph.csrbin.ifmeta.bin \
  --max-sets-print 1
```

The final summary counts these four source-index-ordered orientations:

```text
source1 -> connector -> source2
source1 <- connector -> source2
source1 -> connector <- source2
source1 <- connector <- source2
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--source-limit <n>` | Analyze only the first `n` unique sources. |
| `--net-limit <n>` | Read only the first `n` route requests. |
| `--max-sets-print <n>` | Print at most `n` orientation lines; `0` means no cap. |
| `--max-pairs-print <n>` | Alias for `--max-sets-print`. |
| `--progress-every <n>` | Print relation-scan progress every `n` sources; default `25`. |

Generator options:

| Option | Meaning |
| --- | --- |
| `--batch-size <n>` | Batched generator only: sources per GPU Bellman-Ford batch; default `4`, max `64`. |
| `--no-connector-warm-start` | Warm-started generator only: disable hop-2 shared-connector warm starts. |
| `--min-connector-sources <n>` | Warm-started generator only: route a connector only if at least `n` sources can warm-start through `source -> connector`; default `2`. |
| `--max-connector-groups <n>` | Warm-started generator only: cap the number of connector groups; `0` means no cap. |
| `--max-iters <n>` | Bellman-Ford relaxation limit; default `-1` means `n - 1`. |
| `--net-limit <n>` | Use only the first `n` route requests from metadata. |
| `--source-limit <n>` | Run only the first `n` unique source nodes. |
| `--query-limit <n>` | Emit at most `n` source-sink path records. |
| `--bf-progress-every <n>` | Print Bellman-Ford progress every `n` iterations; generated drivers default to automatic updates at least every 100 iterations. |
| `--no-bf-progress` | Disable the BF progress bar. |
| `--skip-unreached` | Do not write JSONL records for unreachable sinks. |
| `--abs-tol <x>` / `--rel-tol <x>` | Path reconstruction tolerances. |

The single-source generator copies the CSR graph to the GPU once, reuses that
device graph for all unique metadata sources, and copies back one dense
distance vector per source for CPU-side path reconstruction and JSONL writing.
