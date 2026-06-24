#pragma once

#include <hip/hip_runtime.h>

// Computes:
//
//   C[i,j] = min_k(A[i,k] + B[k,j])
//
// All matrices are row-major.
//
// A is M x K
// B is K x N
// C is M x N
//
// d_A, d_B, d_C must already be GPU/device pointers.
//
// For tightly packed row-major matrices:
//   lda = K
//   ldb = N
//   ldc = N
//
// The kernel launch is asynchronous with respect to the host.
// Call hipDeviceSynchronize() or synchronize the stream if you need completion.
hipError_t minplus_gemm_f32(const float* d_A,
                            const float* d_B,
                            float* d_C,
                            int M,
                            int N,
                            int K,
                            int lda,
                            int ldb,
                            int ldc,
                            hipStream_t stream = nullptr);
