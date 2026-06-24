#include "minplus_hip.hpp"

#include <limits>

namespace {

template <
    int BM = 64,
    int BN = 64,
    int BK = 32,
    int TM = 4,
    int TN = 4>
__global__ void minplus_kernel_f32(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M,
                                   int N,
                                   int K,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   float INF) {
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int tid = ty * blockDim.x + tx;
  const int nthreads = blockDim.x * blockDim.y;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  float acc[TM][TN];

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = INF;
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += nthreads) {
      const int r = idx / BK;
      const int k = idx - r * BK;

      const int global_r = block_row + r;
      const int global_k = k0 + k;

      As[r][k] =
          (global_r < M && global_k < K)
              ? A[global_r * lda + global_k]
              : INF;
    }

    for (int idx = tid; idx < BK * BN; idx += nthreads) {
      const int k = idx / BN;
      const int c = idx - k * BN;

      const int global_k = k0 + k;
      const int global_c = block_col + c;

      Bs[k][c] =
          (global_k < K && global_c < N)
              ? B[global_k * ldb + global_c]
              : INF;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = As[ty + i * TY][k];
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = Bs[k][tx + j * TX];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const float candidate = a_frag[i] + b_frag[j];
          acc[i][j] = candidate < acc[i][j] ? candidate : acc[i][j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + ty + i * TY;

#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = block_col + tx + j * TX;

      if (r < M && c < N) {
        C[r * ldc + c] = acc[i][j];
      }
    }
  }
}

}  // namespace

hipError_t minplus_gemm_f32(const float* d_A,
                            const float* d_B,
                            float* d_C,
                            int M,
                            int N,
                            int K,
                            int lda,
                            int ldb,
                            int ldc,
                            hipStream_t stream) {
  if (d_A == nullptr || d_B == nullptr || d_C == nullptr) {
    return hipErrorInvalidValue;
  }

  if (M <= 0 || N <= 0 || K <= 0) {
    return hipErrorInvalidValue;
  }

  if (lda < K || ldb < N || ldc < N) {
    return hipErrorInvalidValue;
  }

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  const float INF = std::numeric_limits<float>::infinity();

  hipLaunchKernelGGL(
      (minplus_kernel_f32<BM, BN, BK, TM, TN>),
      grid,
      block,
      0,
      stream,
      d_A,
      d_B,
      d_C,
      M,
      N,
      K,
      lda,
      ldb,
      ldc,
      INF);

  return hipGetLastError();
}
