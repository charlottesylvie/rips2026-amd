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
