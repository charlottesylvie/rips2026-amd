# Route-window implementation notes

This file records the current implementation status of the opt-in per-net
PathFinder route-window heuristic. Update it whenever this feature changes.

## Current behavior

Enable the feature only for production Delta-Stepping runs:

```text
pathfinder <graph.csrbin> <metadata.ifmeta.bin> \
  --sssp-engine delta-step --route-window
```

For every net, PathFinder combines the extents of all source nodes and all
initially unresolved sink nodes, then expands that rectangle by **20 tiles on
every side**. The bounds are clamped to the packed-coordinate range.

The shared immutable Delta-Stepping graph uploads one packed `uint64_t` bounds
entry per CSR row. During generic light/heavy edge relaxation, a candidate
destination is skipped when its physical extent does not intersect the current
net window. Nodes without valid physical coordinates are kept conservatively.

If any requested sink is unreachable in the window, PathFinder reruns the same
batched query with no window before accepting the net. Consequently, the
current fixed-margin version preserves completeness relative to the unbounded
query, although a successful bounded route remains a heuristic: it need not be
the globally shortest route.

## Current scope and limitations

- The feature is disabled unless `--route-window` is supplied.
- It applies only to Delta-Stepping, and windowed queries deliberately bypass
  the exact-unit specialization in favor of generic Delta-Stepping.
- The margin is currently fixed at 20; adaptive growth, CLI-configurable
  margins, per-net telemetry, and metadata-sidecar serialization of packed
  bounds are follow-up work.
- Bounds are packed from the existing metadata coordinate arrays at PathFinder
  startup and uploaded once. This avoids a per-net upload, but a later format
  revision can serialize the packed sidecar array directly.

## Files changed for the first implementation

- `pathfinder.hpp`: route-window option and fixed 20-tile default.
- `pathfinder.cpp`: per-net window construction, packed-host bounds, fallback,
  CLI parsing, and shared graph creation.
- `delta_stepping/delta_stepping_hip_CSR.hpp/.cpp`: immutable device bounds,
  query window API, generic dispatch, and destination filtering.
- `pathfinder_router.cpp`: forwarding of `--route-window`.
- `tests/pathfinder_bf10_cpu_stub_test.cpp`: constructor stub matching the new
  shared Delta graph API.

## Verification status

`git diff --check` passes. The local environment currently has neither `g++`
nor `hipcc` on `PATH`, so the CPU-stub and HIP test suites have not yet run.
The CPU-stub test has been kept in sync with the added graph constructor and
the `route_net()` metadata parameter.
