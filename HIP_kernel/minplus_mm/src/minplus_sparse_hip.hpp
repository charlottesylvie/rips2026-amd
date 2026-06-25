#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace minplus_sparse {

// 64-bit offsets let rowptr and nnz exceed 2^31.
// Column indices remain 32-bit, so cols must fit in int.
using Offset = std::int64_t;
using Index = int;

// Default tuning for very sparse graphs.
// Good when each output row has at most a few hundred unique columns.
inline constexpr int kDefaultHashCapacity = 512;
inline constexpr int kDefaultBlockSize = 64;

// Larger-row tuning.
// Use this if the default returns hipErrorInvalidValue due to hash overflow.
inline constexpr int kLargeHashCapacity = 4096;
inline constexpr int kLargeBlockSize = 256;

struct DeviceCsrF32 {
  Offset rows;
  Offset cols;
  Offset nnz;

  // rowptr length: rows + 1
  // colind length: nnz
  // values length: nnz
  const Offset* rowptr;
  const Index* colind;
  const float* values;
};

// Computes C = A min-plus B.
//
// Semiring:
//   add/multiply replacement: a + b
//   sum/reduce replacement:   min
//
// Therefore:
//   C(i,j) = min_k A(i,k) + B(k,j)
//
// CSR sparse semantics:
//   Missing entries are treated as +INF.
//   Explicit +INF input entries are ignored.
//   NaN input/candidate values are ignored.
//
// Output:
//   *d_C_rowptr  -> device allocation, length A.rows + 1
//   *d_C_colind  -> device allocation, length *h_C_nnz, or nullptr if empty
//   *d_C_values  -> device allocation, length *h_C_nnz, or nullptr if empty
//   *h_C_nnz     -> output nnz copied to host
//
// Caller owns output allocations and must hipFree() them.
//
// Notes:
//   - Output column indices are not sorted.
//   - Duplicate input columns are allowed; duplicates are reduced by min.
//   - Returns hipErrorInvalidValue if one output row exceeds HASH_CAP unique
//     columns, if dimensions/pointers are invalid, or if indices are invalid.
hipError_t minplus_spgemm_csr_f32(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream = 0);

// Same algorithm, but with a larger per-row shared-memory hash table.
// More robust for row expansion, but less efficient for ultra-sparse rows.
hipError_t minplus_spgemm_csr_f32_large_rows(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream = 0);

// Tuned entry point.
//
// This template is implemented in the .cpp file and explicitly instantiated
// there for the following pairs:
//
//   <256, 64>
//   <512, 64>
//   <1024, 128>
//   <2048, 128>
//   <4096, 256>
//
// To use another HASH_CAP/BLOCK_SIZE pair, add an explicit instantiation at
// the bottom of minplus_sparse_hip.cpp.
template <int HASH_CAP, int BLOCK_SIZE>
hipError_t minplus_spgemm_csr_f32_tuned(
    const DeviceCsrF32& A,
    const DeviceCsrF32& B,
    Offset** d_C_rowptr,
    Index** d_C_colind,
    float** d_C_values,
    Offset* h_C_nnz,
    hipStream_t stream = 0);

extern template hipError_t minplus_spgemm_csr_f32_tuned<256, 64>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

extern template hipError_t minplus_spgemm_csr_f32_tuned<512, 64>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

extern template hipError_t minplus_spgemm_csr_f32_tuned<1024, 128>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

extern template hipError_t minplus_spgemm_csr_f32_tuned<2048, 128>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

extern template hipError_t minplus_spgemm_csr_f32_tuned<4096, 256>(
    const DeviceCsrF32&,
    const DeviceCsrF32&,
    Offset**,
    Index**,
    float**,
    Offset*,
    hipStream_t);

}  // namespace minplus_sparse
