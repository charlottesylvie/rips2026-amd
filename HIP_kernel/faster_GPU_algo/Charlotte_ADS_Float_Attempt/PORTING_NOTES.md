# CUDA-to-HIP porting notes

## Launch topology

The original implementation used CUDA dynamic parallelism: a device-side driver
kernel created two streams and launched `wl_kernel` plus `sssp_kernel`. HIP does
not support launching a `__global__` function from device code, so the driver is
now a host function. It records one start event, releases the manager and worker
streams together, and records completion only after both persistent kernels
finish.

On runtimes supporting `hipExtStreamCreateWithCUMask`, the manager is pinned
to `ADDS_RESERVED_CUS` compute units and workers to the remaining units. This
prevents persistent worker workgroups from starving the manager. If CU-mask
stream creation fails, the code falls back to ordinary nonblocking streams and
limits worker occupancy to the capacity of `CUs - ADDS_RESERVED_CUS`.

## Wavefront assumptions

The data structure has 32 bags and repeatedly maps one metadata item to one
lane. The port therefore targets wave32 rather than attempting to generalize the
algorithm to wave64. It:

- builds `gfx1151` with `-mno-wavefrontsize64`;
- verifies `hipDeviceProp_t::warpSize == 32` at runtime;
- keeps logical masks as 32-bit values;
- widens masks to HIP's required 64-bit type only at `_sync` intrinsics.

## PTX replacements

`common.h` replaces inline PTX with C++ or HIP functionality:

- `bfind` -> `__clz`-based highest-set-bit lookup;
- `popc` -> `__popc`;
- `bfe` / `bfi` -> masked shifts;
- `fns` -> a small nth-set-bit helper;
- `%laneid` -> `threadIdx.x & 31`;
- `%clock` -> `wall_clock64()`;
- `ld.global.cg` / `st.global.cg` -> volatile global loads/stores plus the
  algorithm's existing fences;
- PTX `exit` -> an explicit return path observed by the complete wave.

AMD waves execute in lockstep. The local `warp_sync()` helper uses the AMD wave
barrier when compiling device code and a workgroup-scoped memory fence for
shared-memory visibility.

## CUB removal

The port has no CUB or hipCUB dependency. It replaces:

- `cub::ThreadLoad` and `cub::ThreadStore` with typed volatile helpers;
- `cub::WarpReduce<unsigned>::Sum` with a shuffle reduction;
- `cub::WarpScan<int>::ExclusiveSum` with a shuffle prefix scan.

This avoids relying on CUB cache-modifier behavior that is specific to the CUDA
backend.

## Graph and host-code fixes

The port also corrects or hardens several host-side issues while preserving the
file format and algorithm:

- validates `.gr` bounds, row offsets, destinations, and edge weights;
- avoids mutating the input path while extracting its basename;
- fixes the original assignment-in-an-assert typo in CPU copy-back;
- frees all per-bag pointer arrays and worker-status storage;
- guards float-to-integer bucket quantization against overflow;
- handles all-zero average weights using a representable bucket quantum;
- initializes state that the original code relied on control flow to set;
- computes worker count from HIP occupancy instead of a fixed SM formula.

## Timing difference

The reported time covers the complete concurrent lifetime of the manager and
worker kernels. Distance initialization, worklist reset, and copying the
per-thread processed-work counters are outside the timed interval. The average
edge weight is computed exactly once on the host before the benchmark loop.

## Deliberately unchanged constraints

- single-source shortest paths only;
- finite, nonnegative `float` edge weights; unweighted input uses unit weights;
- 32-bit node and edge indices;
- 32 bags and at most 127 worker workgroups;
- Galois/Lonestar version-1 `.gr` input;
- asynchronous duplicate work entries resolved by atomic float minimum.
