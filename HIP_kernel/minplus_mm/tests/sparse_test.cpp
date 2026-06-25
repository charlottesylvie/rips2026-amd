#include "minplus_hip.hpp"
#include "minplus_sparse_hip.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#define HIP_CHECK(expr)                                                   \
  do {                                                                    \
    hipError_t err__ = (expr);                                            \
    if (err__ != hipSuccess) {                                            \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                << ": " << hipGetErrorString(err__) << std::endl;         \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;

struct HostCsr {
  Offset rows;
  Offset cols;
  Offset nnz;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

struct DeviceCsrOwner {
  minplus_sparse::DeviceCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* colind = nullptr;
  float* values = nullptr;

  ~DeviceCsrOwner() {
    if (rowptr) hipFree(rowptr);
    if (colind) hipFree(colind);
    if (values) hipFree(values);
  }
};

static HostCsr make_random_graph_csr(
    int n,
    int degree,
    unsigned seed) {
  HostCsr g;
  g.rows = n;
  g.cols = n;
  g.rowptr.resize(static_cast<size_t>(n) + 1);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> vertex_dist(0, n - 1);
  std::uniform_real_distribution<float> weight_dist(1.0f, 10.0f);

  std::vector<int> cols;
  cols.reserve(degree);

  for (int u = 0; u < n; ++u) {
    g.rowptr[u] = static_cast<Offset>(g.colind.size());
    cols.clear();

    while (static_cast<int>(cols.size()) < degree) {
      int v = vertex_dist(rng);
      if (v == u) continue;

      if (std::find(cols.begin(), cols.end(), v) == cols.end()) {
        cols.push_back(v);
      }
    }

    std::sort(cols.begin(), cols.end());

    for (int v : cols) {
      g.colind.push_back(v);
      g.values.push_back(weight_dist(rng));
    }
  }

  g.rowptr[n] = static_cast<Offset>(g.colind.size());
  g.nnz = static_cast<Offset>(g.colind.size());
  return g;
}

static DeviceCsrOwner copy_csr_to_device(const HostCsr& h) {
  DeviceCsrOwner d;

  HIP_CHECK(hipMalloc(&d.rowptr, sizeof(Offset) * h.rowptr.size()));
  HIP_CHECK(hipMemcpy(
      d.rowptr,
      h.rowptr.data(),
      sizeof(Offset) * h.rowptr.size(),
      hipMemcpyHostToDevice));

  if (h.nnz > 0) {
    HIP_CHECK(hipMalloc(&d.colind, sizeof(Index) * h.colind.size()));
    HIP_CHECK(hipMalloc(&d.values, sizeof(float) * h.values.size()));

    HIP_CHECK(hipMemcpy(
        d.colind,
        h.colind.data(),
        sizeof(Index) * h.colind.size(),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        d.values,
        h.values.data(),
        sizeof(float) * h.values.size(),
        hipMemcpyHostToDevice));
  }

  d.view.rows = h.rows;
  d.view.cols = h.cols;
  d.view.nnz = h.nnz;
  d.view.rowptr = d.rowptr;
  d.view.colind = d.colind;
  d.view.values = d.values;

  return d;
}

static std::vector<float> csr_to_dense(const HostCsr& csr) {
  const float INF = std::numeric_limits<float>::infinity();
  std::vector<float> dense(
      static_cast<size_t>(csr.rows) * static_cast<size_t>(csr.cols),
      INF);

  for (Offset r = 0; r < csr.rows; ++r) {
    for (Offset p = csr.rowptr[r]; p < csr.rowptr[r + 1]; ++p) {
      dense[static_cast<size_t>(r) * static_cast<size_t>(csr.cols) +
            static_cast<size_t>(csr.colind[p])] = csr.values[p];
    }
  }

  return dense;
}

static float time_dense_minplus(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int n,
    int warmup_iters,
    int timed_iters,
    hipStream_t stream) {
  for (int i = 0; i < warmup_iters; ++i) {
    HIP_CHECK(minplus_gemm_f32(
        d_A, d_B, d_C,
        n, n, n,
        n, n, n,
        stream));
  }

  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, stream));

  for (int i = 0; i < timed_iters; ++i) {
    HIP_CHECK(minplus_gemm_f32(
        d_A, d_B, d_C,
        n, n, n,
        n, n, n,
        stream));
  }

  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return ms / static_cast<float>(timed_iters);
}

static float time_sparse_minplus(
    const minplus_sparse::DeviceCsrF32& A,
    const minplus_sparse::DeviceCsrF32& B,
    Offset* out_nnz,
    int warmup_iters,
    int timed_iters,
    hipStream_t stream) {
  for (int i = 0; i < warmup_iters; ++i) {
    Offset* C_rowptr = nullptr;
    Index* C_colind = nullptr;
    float* C_values = nullptr;
    Offset C_nnz = 0;

    HIP_CHECK(minplus_sparse::minplus_spgemm_csr_f32(
        A, B,
        &C_rowptr,
        &C_colind,
        &C_values,
        &C_nnz,
        stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    hipFree(C_rowptr);
    hipFree(C_colind);
    hipFree(C_values);
  }

  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, stream));

  Offset last_nnz = 0;

  for (int i = 0; i < timed_iters; ++i) {
    Offset* C_rowptr = nullptr;
    Index* C_colind = nullptr;
    float* C_values = nullptr;
    Offset C_nnz = 0;

    HIP_CHECK(minplus_sparse::minplus_spgemm_csr_f32(
        A, B,
        &C_rowptr,
        &C_colind,
        &C_values,
        &C_nnz,
        stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    last_nnz = C_nnz;

    hipFree(C_rowptr);
    hipFree(C_colind);
    hipFree(C_values);
  }

  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  if (out_nnz) {
    *out_nnz = last_nnz;
  }

  return ms / static_cast<float>(timed_iters);
}

int main(int argc, char** argv) {
  int n = 100000;
  int degree = 10;
  int warmup_iters = 2;
  int timed_iters = 5;
  bool run_dense = false;

  if (argc > 1) n = std::atoi(argv[1]);
  if (argc > 2) degree = std::atoi(argv[2]);
  if (argc > 3) timed_iters = std::atoi(argv[3]);
  if (argc > 4) run_dense = std::atoi(argv[4]) != 0;

  std::cout << "n           = " << n << "\n";
  std::cout << "degree      = " << degree << "\n";
  std::cout << "timed iters = " << timed_iters << "\n";
  std::cout << "run dense   = " << (run_dense ? "yes" : "no") << "\n\n";

  if (n <= 0 || degree <= 0 || degree >= n) {
    std::cerr << "Invalid n/degree.\n";
    return 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::cout << "Generating random CSR graph...\n";
  HostCsr hA = make_random_graph_csr(n, degree, 1234);
  HostCsr hB = hA;

  std::cout << "Input nnz   = " << hA.nnz << "\n";
  std::cout << "Copying CSR to device...\n";

  DeviceCsrOwner dA = copy_csr_to_device(hA);
  DeviceCsrOwner dB = copy_csr_to_device(hB);

  Offset C_sparse_nnz = 0;

  std::cout << "\nBenchmarking sparse CSR min-plus SpGEMM...\n";

  float sparse_ms = time_sparse_minplus(
      dA.view,
      dB.view,
      &C_sparse_nnz,
      warmup_iters,
      timed_iters,
      stream);

  std::cout << "Sparse avg ms       = " << sparse_ms << "\n";
  std::cout << "Sparse output nnz   = " << C_sparse_nnz << "\n";

  if (!run_dense) {
    std::cout << "\nDense benchmark skipped.\n";
    std::cout << "Pass argv[4] = 1 to run dense.\n";
    std::cout << "Example:\n";
    std::cout << "  ./benchmark_minplus 4096 10 5 1\n";
    std::cout << "\nDo not run dense at n=100000; it would require ~40 GB per matrix.\n";

    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
  }

  const size_t dense_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
  const double dense_gb_per_matrix =
      static_cast<double>(dense_elems * sizeof(float)) / 1.0e9;

  std::cout << "\nDense matrix size per matrix = "
            << dense_gb_per_matrix << " GB\n";

  if (dense_gb_per_matrix > 8.0) {
    std::cerr << "Refusing dense benchmark: matrix too large.\n";
    std::cerr << "Use smaller n, e.g. 2048, 4096, or 8192 if memory allows.\n";
    HIP_CHECK(hipStreamDestroy(stream));
    return 1;
  }

  std::cout << "Converting CSR to dense with INF fill...\n";

  std::vector<float> hA_dense = csr_to_dense(hA);
  std::vector<float> hB_dense = csr_to_dense(hB);

  float* d_A_dense = nullptr;
  float* d_B_dense = nullptr;
  float* d_C_dense = nullptr;

  HIP_CHECK(hipMalloc(&d_A_dense, sizeof(float) * dense_elems));
  HIP_CHECK(hipMalloc(&d_B_dense, sizeof(float) * dense_elems));
  HIP_CHECK(hipMalloc(&d_C_dense, sizeof(float) * dense_elems));

  HIP_CHECK(hipMemcpyAsync(
      d_A_dense,
      hA_dense.data(),
      sizeof(float) * dense_elems,
      hipMemcpyHostToDevice,
      stream));

  HIP_CHECK(hipMemcpyAsync(
      d_B_dense,
      hB_dense.data(),
      sizeof(float) * dense_elems,
      hipMemcpyHostToDevice,
      stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "\nBenchmarking dense min-plus GEMM...\n";

  float dense_ms = time_dense_minplus(
      d_A_dense,
      d_B_dense,
      d_C_dense,
      n,
      warmup_iters,
      timed_iters,
      stream);

  std::cout << "Dense avg ms        = " << dense_ms << "\n";
  std::cout << "Speedup sparse/dense = " << dense_ms / sparse_ms << "x\n";

  hipFree(d_A_dense);
  hipFree(d_B_dense);
  hipFree(d_C_dense);

  HIP_CHECK(hipStreamDestroy(stream));

  return 0;
}
