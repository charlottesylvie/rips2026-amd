# FPGA Net Delta Tests

This folder is for testing delta-stepping performance on real FPGA net
source/sink pairs without running the full PathFinder routing loop.

## Folder Contents

```text
tools/benchmark_manifest.py                  Builds benchmark inventory
tools/delta_sweep.py                         Sweeps delta values
tools/analyze_delta_results.py               Summarizes sweep results
tools/csr_graph_stats.py                     Summarizes CSR degree/weight stats
tools/net_delta_runner.cpp                   Delta-only HIP/C++ runner
run_generate_csr_with_pathfinder_wrapper.sh  Generates CSR + metadata
run_net_delta_sweep.sh                       Runs the delta-only sweep
results/                                     Generated outputs
```

Benchmark and device files can live outside this folder. Point to them with
`BENCHMARK_DIR` and `DEVICE`.

## Server Setup

Run from this folder on the Linux/ROCm server:

```bash
cd /opt/workspace/fpga_delta_experiment

export SETUP_ROOT=/opt/workspace
export AMD_REPO="$SETUP_ROOT/rips2026-amd"
export BENCHMARK_DIR="$SETUP_ROOT/benchmarks"
export DEVICE="$SETUP_ROOT/devices/xcvu3p.device"
export INTERCHANGE_TO_CSR="$AMD_REPO/interchange_to_csr"
export PATHFINDER_BIN="$AMD_REPO/pathfinder"
export SCHEMA_DIR="$SETUP_ROOT/fpga24_routing_contest/fpga-interchange-schema/interchange"
```

## Build The Delta-Only Runner

```bash
mkdir -p build

hipcc -std=c++17 -O3 -DROUTING_PATHFINDER_NO_MAIN \
  -I "$AMD_REPO/Routing" \
  -I "$AMD_REPO/HIP_kernel/delta_stepping/src" \
  -I "$AMD_REPO/HIP_kernel/bellman_ford/src" \
  -I "$AMD_REPO/HIP_kernel/minplus_mm/src" \
  tools/net_delta_runner.cpp \
  "$AMD_REPO/Routing/pathfinder.cpp" \
  "$AMD_REPO/HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp" \
  "$AMD_REPO/HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp" \
  "$AMD_REPO/HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp" \
  -o build/net_delta_runner
```

## Generate CSR For One Benchmark

```bash
BENCHMARK_DIR="$SETUP_ROOT/benchmarks" \
DEVICE="$SETUP_ROOT/devices/xcvu3p.device" \
NET_LIMIT=20 \
./run_generate_csr_with_pathfinder_wrapper.sh vtr_mcml
```

This writes:

```text
results/csr_sweep_manifest.csv
results/pathfinder_csr/vtr_mcml/*.csrbin
results/pathfinder_csr/vtr_mcml/*.ifmeta.bin
```

## Sweep Delta On Real Net Pairs

```bash
DELTAS=1,2,4,8,16,32 \
NET_LIMIT=5 \
PAIR_LIMIT=20 \
REPEATS=3 \
./run_net_delta_sweep.sh
```

The summary is:

```text
results/net_delta_summary.csv
```

## Analyze CSR Graph Properties

After `results/csr_sweep_manifest.csv` exists, run:

```bash
python3 tools/csr_graph_stats.py \
  --manifest results/csr_sweep_manifest.csv \
  --out-dir results/graph_stats \
  --bins 50
```

This writes:

```text
results/graph_stats/graph_summary.csv
results/graph_stats/<benchmark>_degree_histogram.csv
results/graph_stats/<benchmark>_edge_weight_histogram.csv
```

For a much faster averages-only pass:

```bash
python3 tools/csr_graph_stats.py \
  --manifest results/csr_sweep_manifest.csv \
  --out-dir results/graph_stats \
  --quick
```

This writes:

```text
results/graph_stats/graph_summary_quick.csv
```
