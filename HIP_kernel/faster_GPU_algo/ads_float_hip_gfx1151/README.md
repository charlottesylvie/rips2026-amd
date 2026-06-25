# ADDS floating-point SSSP — HIP port for Radeon 8060S

This directory contains a HIP/ROCm port of the CUDA `ads_float` implementation.
It retains the original asynchronous dynamic delta-stepping design:

- one persistent manager workgroup owns the 32 priority bags;
- persistent worker workgroups relax CSR edges and generate new bag entries;
- the manager adapts the delta shift and number of concurrent bags at runtime;
- distances and edge weights remain IEEE-754 single-precision values.

The default build target is **AMD Radeon 8060S / Strix Halo**, whose LLVM target
is `gfx1151`. The code deliberately uses a 32-lane wave because the worklist
metadata maps one lane to each of 32 bags.

## Requirements

Recommended environment:

- Linux or WSL2 on x86-64;
- ROCm/HIP 7.2.1 or newer with development tools installed;
- a Radeon 8060S exposed by `rocminfo` as `gfx1151`;
- Python 3 for the smoke-test utilities.

Check the target before building:

```bash
/opt/rocm/bin/rocminfo | grep -m1 -A2 gfx1151
/opt/rocm/bin/hipcc --version
```

AMD references:

- <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>

## Build

Using Make:

```bash
make
```

The result is `sssp_hip`. Override the compiler or target when necessary:

```bash
make HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151
```

Using CMake:

```bash
cmake -S . -B build-cmake \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j
```

The build forces wave32 with `-mno-wavefrontsize64`. At startup, the executable
also checks that HIP reports `warpSize == 32`, concurrent-kernel support, and a
sufficient workgroup-size limit.

## Run

```bash
./sssp_hip -g 0 -s 0 -o distances.txt graph.gr 3>benchmark.txt
```

Arguments:

```text
-g N       HIP device number; default 0
-s N       source vertex; default 0
-o FILE    write "vertex distance" rows to FILE
-o -       write distances to stdout
-q         accepted for compatibility; currently has no effect
```

When file descriptor 3 is open, one benchmark row is appended:

```text
graph-name average-time-ms average-processed-vertex-count
```

Without descriptor 3, the same summary is written to stderr. The default is
eight measured runs, matching the CUDA program. For a short run:

```bash
ADDS_RUNS=1 ./sssp_hip -s 0 -o distances.txt graph.gr
```

## Input format

The reader accepts the Galois/Lonestar version-1 `.gr` CSR format used by the
original project:

1. four little-endian 64-bit values: version, edge-data size, nodes, edges;
2. one 64-bit cumulative row end per node;
3. one 32-bit destination per edge;
4. optional 32-bit padding when the edge count is odd;
5. one 32-bit float per edge when edge-data size is 4.

The graph must fit 32-bit node and edge indices. Edge weights must be finite and
nonnegative. As in the supplied CUDA reader, a `.gr` file with no edge payload
is interpreted as having unit-weight edges.

## Validation

Static port checks that do not require ROCm:

```bash
make check
```

On the target machine, compile and execute a five-node graph with known
shortest paths:

```bash
make smoke
```

For another graph, compare HIP output against the included CPU Dijkstra
reference:

```bash
ADDS_RUNS=1 ./sssp_hip -s 0 -o result.txt graph.gr
python3 tools/compare_output.py graph.gr result.txt 0
```

## Runtime tuning

The defaults are conservative for the 40-CU Radeon 8060S. These environment
variables can be used without recompiling:

```text
ADDS_RUNS=N             number of benchmark repetitions, 1..1000
ADDS_RESERVED_CUS=N     aggregate worker capacity left for the manager
ADDS_WORKER_BLOCKS=N    worker workgroups, capped by occupancy and MAX_TB
ADDS_WL_ENTRIES=N       requested worklist entries before power-of-two rounding
```

The worker count is derived from HIP occupancy and leaves one CU-equivalent of
aggregate capacity for the manager by default. When supported, the manager and workers are pinned to disjoint CU masks;
otherwise the manager is submitted first and worker occupancy is conservatively capped.

## Important porting detail

HIP on AMD does not support CUDA dynamic parallelism. The CUDA program's
`driver_kernel` launched the manager and workers from device code. In this port,
`run_manager_and_workers()` launches those same persistent kernels from the host
into two nonblocking HIP streams and joins them with events. See
`PORTING_NOTES.md` for the other substitutions.

## Validation status

The source has passed:

- a scan for remaining CUDA runtime calls, CUB calls, PTX, and `__syncwarp`;
- C++17 host-side syntax checking with a small HIP API mock;
- graph-reader and CPU-reference smoke tests.

This environment does not contain ROCm or a Radeon 8060S, so the final HIP
compilation and GPU execution must be performed on the target system. The
included `make smoke` target is intended to expose compiler, scheduling, or
runtime differences immediately.
