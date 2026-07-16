# Delta-Threading V2

This variant is intentionally linkable alongside the original
`delta_threading` implementation.  It changes only the SSSP engine:

- Low-degree active vertices (outdegree <= 16) use one GPU thread each.
- Higher-degree active vertices use one HIP wavefront each.
- A vertex is added to the heavy queue only if it has at least one edge with
  effective weight greater than `delta`.  All-light buckets therefore skip the
  heavy-edge kernel entirely.

The high-degree cutoff defaults to 16 and can be swept without source changes,
for example with `-DDS_V2_HIGH_DEGREE_THRESHOLD=8` in the compile command.

The existing random-graph comparison now reports Delta-Step, Delta-Thread V1,
and Delta-Thread V2, and checks each threaded result against the Delta-Step
distances.

From the repository root, build the comparison with:

```bash
hipcc -std=c++17 -O3 -x hip \
  -I HIP_kernel/bellman_ford/src \
  -I HIP_kernel/delta_stepping/src \
  -I HIP_kernel/delta_threading/src \
  -I HIP_kernel/delta_threading_v2/src \
  HIP_kernel/delta_stepping/testing/speed_compare_delta_vs_bf.cpp \
  HIP_kernel/delta_stepping/src/delta_stepping_hip_CSR.cpp \
  HIP_kernel/bellman_ford/src/bf_hip_CSR.cpp \
  HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
  HIP_kernel/delta_threading/src/delta_stepping_hip_CSR_threads.cpp \
  HIP_kernel/delta_threading_v2/src/delta_stepping_hip_CSR_threads_v2.cpp \
  -o build/speed_compare_delta_vs_bf
```

Example unit-weight-style run (all edges are light when `delta >= 1`):

```bash
build/speed_compare_delta_vs_bf 100000 7 4 2 10 123 0 -1 1
```

The comparison generator still uses weights in `[1, 16]`; use the production
CSR runner for a final FPGA-graph measurement.
