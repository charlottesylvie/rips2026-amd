#include "minplus_sparse_hip.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <cstddef>
#include <limits>

namespace minplus_sparse {
namespace {

constexpr Index kEmptyKey = -1;

constexpr int kStatusOk = 0;
constexpr int kStatusHashOverflow = 1;
constexpr int kStatusInvalidIndex = 2;
constexpr int kStatusInvalidRowPtr = 3;

// Conservative 2D grid launcher.
// This supports up to roughly 4.29B output rows.
// In practice, memory will usually become the limiting factor first.
constexpr unsigned kGridX = 65535u;
constexpr Offset kMaxLaunchRows =
    static_cast<Offset>(kGridX) * static_cast<Offset>(kGridX);

inline bool make_row_grid(Offset rows, dim3* grid) {
  if (rows < 0 || rows > kMaxLaunchRows) {
    return false;
  }

  if (rows == 0) {
    *grid = dim3(1, 1);
    return true;
  }

  const unsigned gx =
      rows < static_cast<Offset>(kGridX)
          ? static_cast<unsigned>(rows)
          : kGridX;

  const unsigned gy =
      static_cast<unsigned>((rows + static_cast<Offset>(gx) - 1) /
                            static_cast<Offset>(gx));

  *grid = dim3(gx, gy);
  return true;
}

inline bool checked_bytes(Offset count, std::size_t elem_size, std::size_t* out) {
  if (count < 0) {
    return false;
  }

  const auto ucount = static_cast<std::size_t>(count);

  if (elem_size != 0 &&
      ucount > std::numeric_limits<std::size_t>::max() / elem_size) {
    return false;
  }

  *out = ucount * elem_size;
  return true;
}

inline hipError_t checked_malloc(void** ptr,
                                 Offset count,
                                 std::size_t elem_size) {
  std::size_t bytes = 0;

  if (!checked_bytes(count, elem_size, &bytes)) {
    return hipErrorInvalidValue;
  }

  return hipMalloc(ptr, bytes);
}

__device__ __forceinline__ Offset logical_block_id_1d() {
  return static_cast<Offset>(blockIdx.x) +
         static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x);
}

__device__ __forceinline__ unsigned hash_u32(unsigned x) {
  // Small integer hash suitable for open addressing.
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ bool store_candidate(float x, float INF) {
  // Skip NaN and +INF.
  //
  // Missing sparse entries represent +INF, so explicitly storing +INF would
  // only make the sparse matrix denser without changing min-plus semantics.
  return x == x && x != INF;
}

__device__ __forceinline__ void atomic_min_float(float* addr, float value) {
  // Generic CAS-based float min.
  // Works for positive and negative finite values, +INF, and -INF.
  // Assumes value is not NaN.
  unsigned int* addr_u = reinterpret_cast<unsigned int*>(addr);
  unsigned int old_bits = *addr_u;

  while (value < __uint_as_float(old_bits)) {
    const unsigned int assumed = old_bits;
    old_bits = atomicCAS(addr_u, assumed, __float_as_uint(value));

    if (old_bits == assumed) {
      break;
    }
  }
}

template <int HASH_CAP>
__device__ __forceinline__ void hash_insert_key(Index* keys,
                                                Index col,
                                                int* status) {
  static_assert((HASH_CAP & (HASH_CAP - 1)) == 0,
                "HASH_CAP must be a power of two");

  unsigned slot =
      hash_u32(static_cast<unsigned>(col)) & static_cast<unsigned>(HASH_CAP - 1);

#pragma unroll 1
  for (int probe = 0; probe < HASH_CAP; ++probe) {
    const Index old = atomicCAS(&keys[slot], kEmptyKey, col);

    if (old == kEmptyKey || old == col) {
      return;
    }

    slot = (slot + 1) & static_cast<unsigned>(HASH_CAP - 1);
  }

  atomicExch(status, kStatusHashOverflow);
}

template <int HASH_CAP>
__device__ __forceinline__ void hash_insert_min(Index* keys,
                                                float* vals,
                                                Index col,
                                                float value,
                                                int* status) {
  static_assert((HASH_CAP & (HASH_CAP - 1)) == 0,
                "HASH_CAP must be a power of two");

  unsigned slot =
      hash_u32(static_cast<unsigned>(col)) & static_cast<unsigned>(HASH_CAP - 1);

#pragma unroll 1
  for (int probe = 0; probe < HASH_CAP; ++probe) {
    const Index old = atomicCAS(&keys[slot], kEmptyKey, col);

    if (old == kEmptyKey || old == col) {
      atomic_min_float(&vals[slot], value);
      return;
    }

    slot = (slot + 1) & static_cast<unsigned>(HASH_CAP - 1);
  }

  atomicExch(status, kStatusHashOverflow);
}

template <int HASH_CAP>
__global__ void minplus_spgemm_symbolic_kernel(
    const Offset* __restrict__ A_rowptr,
    const Index* __restrict__ A_colind,
    const float* __restrict__ A_values,
    Offset A_nnz,
    const Offset* __restrict__ B_rowptr,
    const Index* __restrict__ B_colind,
    const float* __restrict__ B_values,
    Offset B_nnz,
    Offset M,
    Offset K,
    Offset N,
    Offset* __restrict__ row_counts,
    float INF,
    int* __restrict__ status) {
  extern __shared__ unsigned char smem[];
  Index* keys = reinterpret_cast<Index*>(smem);

  const Offset row = logical_block_id_1d();
  const int tid = static_cast<int>(threadIdx.x);

  if (row >= M) {
    return;
  }

  for (int p = tid; p < HASH_CAP; p += static_cast<int>(blockDim.x)) {
    keys[p] = kEmptyKey;
  }

  __syncthreads();

  const Offset a0 = A_rowptr[row];
  const Offset a1 = A_rowptr[row + 1];

  if (a0 < 0 || a1 < a0 || a1 > A_nnz) {
    if (tid == 0) {
      atomicExch(status, kStatusInvalidRowPtr);
      row_counts[row] = 0;
    }
    return;
  }

  for (Offset ap = a0; ap < a1; ++ap) {
    const Index k = A_colind[ap];

    if (k < 0 || static_cast<Offset>(k) >= K) {
      if (tid == 0) {
        atomicExch(status, kStatusInvalidIndex);
      }
      continue;
    }

    const float av = A_values[ap];

    if (!store_candidate(av, INF)) {
      continue;
    }

    const Offset b0 = B_rowptr[k];
    const Offset b1 = B_rowptr[static_cast<Offset>(k) + 1];

    if (b0 < 0 || b1 < b0 || b1 > B_nnz) {
      if (tid == 0) {
        atomicExch(status, kStatusInvalidRowPtr);
      }
      continue;
    }

    for (Offset bp = b0 + static_cast<Offset>(tid);
         bp < b1;
         bp += static_cast<Offset>(blockDim.x)) {
      const Index col = B_colind[bp];

      if (col < 0 || static_cast<Offset>(col) >= N) {
        atomicExch(status, kStatusInvalidIndex);
        continue;
      }

      const float bv = B_values[bp];
      const float candidate = av + bv;

      if (store_candidate(candidate, INF)) {
        hash_insert_key<HASH_CAP>(keys, col, status);
      }
    }
  }

  __syncthreads();

  __shared__ int count;

  if (tid == 0) {
    count = 0;
  }

  __syncthreads();

  int local_count = 0;

  for (int p = tid; p < HASH_CAP; p += static_cast<int>(blockDim.x)) {
    local_count += keys[p] != kEmptyKey;
  }

  atomicAdd(&count, local_count);

  __syncthreads();

  if (tid == 0) {
    row_counts[row] = static_cast<Offset>(count);
  }
}

template <int HASH_CAP>
__global__ void minplus_spgemm_numeric_kernel(
    const Offset* __restrict__ A_rowptr,
    const Index* __restrict__ A_colind,
    const float* __restrict__ A_values,
    Offset A_nnz,
    const Offset* __restrict__ B_rowptr,
    const Index* __restrict__ B_colind,
    const float* __restrict__ B_values,
    Offset B_nnz,
    const Offset* __restrict__ C_rowptr,
    Offset M,
    Offset K,
    Offset N,
    Index* __restrict__ C_colind,
    float* __restrict__ C_values,
    float INF,
    int* __restrict__ status) {
  extern __shared__ unsigned char smem[];

  Index* keys = reinterpret_cast<Index*>(smem);
  float* vals = reinterpret_cast<float*>(keys + HASH_CAP);

  const Offset row = logical_block_id_1d();
  const int tid = static_cast<int>(threadIdx.x);

  if (row >= M) {
    return;
  }

  for (int p = tid; p < HASH_CAP; p += static_cast<int>(blockDim.x)) {
    keys[p] = kEmptyKey;
    vals[p] = INF;
  }

  __syncthreads();

  const Offset a0 = A_rowptr[row];
  const Offset a1 = A_rowptr[row + 1];

  if (a0 < 0 || a1 < a0 || a1 > A_nnz) {
    if (tid == 0) {
      atomicExch(status, kStatusInvalidRowPtr);
    }
    return;
  }

  for (Offset ap = a0; ap < a1; ++ap) {
    const Index k = A_colind[ap];

    if (k < 0 || static_cast<Offset>(k) >= K) {
      if (tid == 0) {
        atomicExch(status, kStatusInvalidIndex);
      }
      continue;
    }

    const float av = A_values[ap];

    if (!store_candidate(av, INF)) {
      continue;
    }

    const Offset b0 = B_rowptr[k];
    const Offset b1 = B_rowptr[static_cast<Offset>(k) + 1];

    if (b0 < 0 || b1 < b0 || b1 > B_nnz) {
      if (tid == 0) {
        atomicExch(status, kStatusInvalidRowPtr);
      }
      continue;
    }

    for (Offset bp = b0 + static_cast<Offset>(tid);
         bp < b1;
         bp += static_cast<Offset>(blockDim.x)) {
      const Index col = B_colind[bp];

      if (col < 0 || static_cast<Offset>(col) >= N) {
        atomicExch(status, kStatusInvalidIndex);
        continue;
      }

      const float bv = B_values[bp];
      const float candidate = av + bv;

      if (store_candidate(candidate, INF)) {
        hash_insert_min<HASH_CAP>(keys, vals, col, candidate, status);
      }
    }
  }

  __syncthreads();

  __shared__ int write_count;

  if (tid == 0) {
    write_count = 0;
  }

  __syncthreads();

  const Offset out0 = C_rowptr[row];

  for (int p = tid; p < HASH_CAP; p += static_cast<int>(blockDim.x)) {
    const Index col = keys[p];

    if (col != kEmptyKey) {
      const int local_dst = atomicAdd(&write_count, 1);
      const Offset dst = out0 + static_cast<Offset>(local_dst);

      C_colind[dst] = col;
      C_values[dst] = vals[p];
    }
  }
}

__global__ void set_last_rowptr_kernel(const Offset* __restrict__ row_counts,
                                       Offset* __restrict__ C_rowptr,
                                       Offset M) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && M > 0) {
    C_rowptr[M] = C_rowptr[M - 1] + row_counts[M - 1];
  }
}

}  // namespace

template <int HASH_CAP, int BLOCK_SIZE>
hipError_t minplus_spgemm_csr_f32_tuned(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream) {
  static_assert(HASH_CAP > 0, "HASH_CAP must be positive");
  static_assert((HASH_CAP & (HASH_CAP - 1)) == 0,
                "HASH_CAP must be a power of two");
  static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");

  if (!d_C_rowptr || !d_C_colind || !d_C_values || !h_C_nnz) {
    return hipErrorInvalidValue;
  }

  *d_C_rowptr = nullptr;
  *d_C_colind = nullptr;
  *d_C_values = nullptr;
  *h_C_nnz = 0;

  if (A.rows < 0 || A.cols < 0 || A.nnz < 0 ||
      B.rows < 0 || B.cols < 0 || B.nnz < 0) {
    return hipErrorInvalidValue;
  }

  if (A.cols != B.rows) {
    return hipErrorInvalidValue;
  }

  if (A.cols > static_cast<Offset>(std::numeric_limits<Index>::max()) ||
      B.cols > static_cast<Offset>(std::numeric_limits<Index>::max())) {
    return hipErrorInvalidValue;
  }

  if (!A.rowptr || !B.rowptr) {
    return hipErrorInvalidValue;
  }

  if (A.nnz > 0 && (!A.colind || !A.values)) {
    return hipErrorInvalidValue;
  }

  if (B.nnz > 0 && (!B.colind || !B.values)) {
    return hipErrorInvalidValue;
  }

  if (A.rows > kMaxLaunchRows) {
    return hipErrorInvalidValue;
  }

  dim3 row_grid;
  if (!make_row_grid(A.rows, &row_grid)) {
    return hipErrorInvalidValue;
  }

  Offset* d_row_counts = nullptr;
  int* d_status = nullptr;
  void* d_scan_temp = nullptr;

  auto cleanup_on_error = [&]() {
    if (d_row_counts) {
      hipFree(d_row_counts);
      d_row_counts = nullptr;
    }

    if (d_status) {
      hipFree(d_status);
      d_status = nullptr;
    }

    if (d_scan_temp) {
      hipFree(d_scan_temp);
      d_scan_temp = nullptr;
    }

    if (*d_C_rowptr) {
      hipFree(*d_C_rowptr);
      *d_C_rowptr = nullptr;
    }

    if (*d_C_colind) {
      hipFree(*d_C_colind);
      *d_C_colind = nullptr;
    }

    if (*d_C_values) {
      hipFree(*d_C_values);
      *d_C_values = nullptr;
    }

    *h_C_nnz = 0;
  };

#define HIP_TRY_CLEAN(expr)                   \
  do {                                        \
    hipError_t _err = (expr);                 \
    if (_err != hipSuccess) {                 \
      cleanup_on_error();                     \
      return _err;                            \
    }                                         \
  } while (0)

  HIP_TRY_CLEAN(checked_malloc(
      reinterpret_cast<void**>(d_C_rowptr),
      A.rows + 1,
      sizeof(Offset)));

  // Empty output: M == 0, K == 0, A empty, or B empty.
  //
  // If A.nnz == 0 or B.nnz == 0, C has no finite entries.
  if (A.rows == 0 || A.cols == 0 || A.nnz == 0 || B.nnz == 0) {
    HIP_TRY_CLEAN(hipMemsetAsync(
        *d_C_rowptr,
        0,
        static_cast<std::size_t>(A.rows + 1) * sizeof(Offset),
        stream));

    HIP_TRY_CLEAN(hipStreamSynchronize(stream));
    *h_C_nnz = 0;

#undef HIP_TRY_CLEAN

    return hipSuccess;
  }

  std::size_t row_counts_bytes = 0;
  if (!checked_bytes(A.rows, sizeof(Offset), &row_counts_bytes)) {
    cleanup_on_error();
    return hipErrorInvalidValue;
  }

  HIP_TRY_CLEAN(hipMalloc(
      reinterpret_cast<void**>(&d_row_counts),
      row_counts_bytes));

  HIP_TRY_CLEAN(hipMalloc(
      reinterpret_cast<void**>(&d_status),
      sizeof(int)));

  HIP_TRY_CLEAN(hipMemsetAsync(d_row_counts, 0, row_counts_bytes, stream));
  HIP_TRY_CLEAN(hipMemsetAsync(d_status, 0, sizeof(int), stream));

  {
    const std::size_t shmem_bytes =
        static_cast<std::size_t>(HASH_CAP) * sizeof(Index);

    hipLaunchKernelGGL(
        (minplus_spgemm_symbolic_kernel<HASH_CAP>),
        row_grid,
        dim3(BLOCK_SIZE),
        shmem_bytes,
        stream,
        A.rowptr,
        A.colind,
        A.values,
        A.nnz,
        B.rowptr,
        B.colind,
        B.values,
        B.nnz,
        A.rows,
        A.cols,
        B.cols,
        d_row_counts,
        std::numeric_limits<float>::infinity(),
        d_status);

    HIP_TRY_CLEAN(hipGetLastError());
  }

  {
    std::size_t scan_temp_bytes = 0;

    HIP_TRY_CLEAN(rocprim::exclusive_scan(
        nullptr,
        scan_temp_bytes,
        d_row_counts,
        *d_C_rowptr,
        static_cast<Offset>(0),
        static_cast<std::size_t>(A.rows),
        rocprim::plus<Offset>(),
        stream));

    HIP_TRY_CLEAN(hipMalloc(&d_scan_temp, scan_temp_bytes));

    HIP_TRY_CLEAN(rocprim::exclusive_scan(
        d_scan_temp,
        scan_temp_bytes,
        d_row_counts,
        *d_C_rowptr,
        static_cast<Offset>(0),
        static_cast<std::size_t>(A.rows),
        rocprim::plus<Offset>(),
        stream));

    HIP_TRY_CLEAN(hipFree(d_scan_temp));
    d_scan_temp = nullptr;

    hipLaunchKernelGGL(
        set_last_rowptr_kernel,
        dim3(1),
        dim3(1),
        0,
        stream,
        d_row_counts,
        *d_C_rowptr,
        A.rows);

    HIP_TRY_CLEAN(hipGetLastError());
  }

  int h_status = kStatusOk;

  HIP_TRY_CLEAN(hipMemcpyAsync(
      h_C_nnz,
      *d_C_rowptr + A.rows,
      sizeof(Offset),
      hipMemcpyDeviceToHost,
      stream));

  HIP_TRY_CLEAN(hipMemcpyAsync(
      &h_status,
      d_status,
      sizeof(int),
      hipMemcpyDeviceToHost,
      stream));

  HIP_TRY_CLEAN(hipStreamSynchronize(stream));

  if (h_status != kStatusOk) {
    cleanup_on_error();
    return hipErrorInvalidValue;
  }

  HIP_TRY_CLEAN(hipFree(d_row_counts));
  d_row_counts = nullptr;

  if (*h_C_nnz == 0) {
    HIP_TRY_CLEAN(hipFree(d_status));
    d_status = nullptr;

#undef HIP_TRY_CLEAN

    return hipSuccess;
  }

  HIP_TRY_CLEAN(checked_malloc(
      reinterpret_cast<void**>(d_C_colind),
      *h_C_nnz,
      sizeof(Index)));

  HIP_TRY_CLEAN(checked_malloc(
      reinterpret_cast<void**>(d_C_values),
      *h_C_nnz,
      sizeof(float)));

  HIP_TRY_CLEAN(hipMemsetAsync(d_status, 0, sizeof(int), stream));

  {
    const std::size_t shmem_bytes =
        static_cast<std::size_t>(HASH_CAP) *
        (sizeof(Index) + sizeof(float));

    hipLaunchKernelGGL(
        (minplus_spgemm_numeric_kernel<HASH_CAP>),
        row_grid,
        dim3(BLOCK_SIZE),
        shmem_bytes,
        stream,
        A.rowptr,
        A.colind,
        A.values,
        A.nnz,
        B.rowptr,
        B.colind,
        B.values,
        B.nnz,
        *d_C_rowptr,
        A.rows,
        A.cols,
        B.cols,
        *d_C_colind,
        *d_C_values,
        std::numeric_limits<float>::infinity(),
        d_status);

    HIP_TRY_CLEAN(hipGetLastError());
  }

  HIP_TRY_CLEAN(hipMemcpyAsync(
      &h_status,
      d_status,
      sizeof(int),
      hipMemcpyDeviceToHost,
      stream));

  HIP_TRY_CLEAN(hipStreamSynchronize(stream));

  HIP_TRY_CLEAN(hipFree(d_status));
  d_status = nullptr;

  if (h_status != kStatusOk) {
    cleanup_on_error();
    return hipErrorInvalidValue;
  }

#undef HIP_TRY_CLEAN

  return hipSuccess;
}

hipError_t minplus_spgemm_csr_f32(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream) {
  return minplus_spgemm_csr_f32_tuned<kDefaultHashCapacity,
                                      kDefaultBlockSize>(
      A,
      B,
      d_C_rowptr,
      d_C_colind,
      d_C_values,
      h_C_nnz,
      stream);
}

hipError_t minplus_spgemm_csr_f32_large_rows(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream) {
  return minplus_spgemm_csr_f32_tuned<kLargeHashCapacity,
                                      kLargeBlockSize>(
      A,
      B,
      d_C_rowptr,
      d_C_colind,
      d_C_values,
      h_C_nnz,
      stream);
}

// Explicit instantiations available from the .hpp.
//
// For ultra-sparse graph rows, start with <256, 64> or <512, 64>.
// For more row expansion, use <1024, 128>, <2048, 128>, or <4096, 256>.
template hipError_t minplus_spgemm_csr_f32_tuned<256, 64>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

template hipError_t minplus_spgemm_csr_f32_tuned<512, 64>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

template hipError_t minplus_spgemm_csr_f32_tuned<1024, 128>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

template hipError_t minplus_spgemm_csr_f32_tuned<2048, 128>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

template hipError_t minplus_spgemm_csr_f32_tuned<4096, 256>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

}  // namespace minplus_sparse
