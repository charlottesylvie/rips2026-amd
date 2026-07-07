#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

// Delta-Stepping SSSP for nonnegative edge weights over the same incoming-edge
// CSR convention used by bellman_ford_minplus_hip_csr:
//   adjacency row v, column u = weight of directed edge u -> v.
// The implementation internally builds an outgoing CSR transpose on the GPU.
using DeltaSteppingCsrProgress = BellmanFordCsrProgress;
using DeltaSteppingCsrProgressCallback = BellmanFordCsrProgressCallback;
using DeltaSteppingCsrResult = BellmanFordCsrResult;

class DeltaSteppingCsrWorkspace {
 public:
  struct Impl;

  explicit DeltaSteppingCsrWorkspace(const HostCsrF32& adjacency,
                                     hipStream_t stream = nullptr);
  ~DeltaSteppingCsrWorkspace();

  DeltaSteppingCsrWorkspace(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace& operator=(const DeltaSteppingCsrWorkspace&) = delete;
  DeltaSteppingCsrWorkspace(DeltaSteppingCsrWorkspace&&) noexcept;
  DeltaSteppingCsrWorkspace& operator=(DeltaSteppingCsrWorkspace&&) noexcept;

  void update_values(const std::vector<float>& values,
                     hipStream_t stream = nullptr);

  DeltaSteppingCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      DeltaSteppingCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

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
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingCsrResult delta_stepping_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

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
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(d_adjacency, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, source, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

inline DeltaSteppingCsrResult delta_stepping_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters,
    hipStream_t stream = nullptr,
    DeltaSteppingCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_minplus_hip_csr(adjacency, sources, target, delta, max_iters,
                                        stream, progress_callback, progress_user_data);
}

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
