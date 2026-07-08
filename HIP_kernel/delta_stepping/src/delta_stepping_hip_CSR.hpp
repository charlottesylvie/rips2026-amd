#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>

// Delta-Stepping SSSP for nonnegative edge weights over the same incoming-edge
// CSR convention used by bellman_ford_minplus_hip_csr:
//   adjacency row v, column u = weight of directed edge u -> v.
// The implementation internally builds an outgoing CSR transpose on the GPU.
using DeltaSteppingCsrProgress = BellmanFordCsrProgress;
using DeltaSteppingCsrProgressCallback = BellmanFordCsrProgressCallback;
using DeltaSteppingCsrResult = BellmanFordCsrResult;

class DeltaSteppingGraphContext {
 public:
  explicit DeltaSteppingGraphContext(const minplus_sparse::DeviceCsrF32& d_adjacency,
                                     hipStream_t stream = nullptr);
  explicit DeltaSteppingGraphContext(const HostCsrF32& adjacency,
                                     hipStream_t stream = nullptr);
  ~DeltaSteppingGraphContext();

  DeltaSteppingGraphContext(const DeltaSteppingGraphContext&) = delete;
  DeltaSteppingGraphContext& operator=(const DeltaSteppingGraphContext&) = delete;
  DeltaSteppingGraphContext(DeltaSteppingGraphContext&&) noexcept;
  DeltaSteppingGraphContext& operator=(DeltaSteppingGraphContext&&) noexcept;

  minplus_sparse::Offset rows() const;
  minplus_sparse::Offset cols() const;
  minplus_sparse::Offset nnz() const;
  std::size_t frontier_capacity() const;

  DeltaSteppingCsrResult run(
      int source,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr) const;

  DeltaSteppingCsrResult run(int source,
                             float delta,
                             hipStream_t stream = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const DeltaSteppingGraphContext& context,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const DeltaSteppingGraphContext& context,
    int source,
    float delta,
    hipStream_t stream = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream = nullptr);

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, source, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}
