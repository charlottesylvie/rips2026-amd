#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

// Thread-per-active-vertex Delta-Stepping SSSP for nonnegative edge weights.
//
// This is intended to coexist with the original delta_stepping_hip_CSR.cpp in
// the same executable.  The public names are intentionally different from the
// original implementation to avoid duplicate linker symbols.
//
// CSR convention matches bellman_ford_minplus_hip_csr and the original
// delta-stepping implementation:
//   adjacency row v, column u = weight of directed edge u -> v.
// The implementation internally builds an outgoing CSR transpose on the GPU.
using DeltaSteppingThreadsCsrProgress = BellmanFordCsrProgress;
using DeltaSteppingThreadsCsrProgressCallback = BellmanFordCsrProgressCallback;
using DeltaSteppingThreadsCsrResult = BellmanFordCsrResult;

class DeltaSteppingThreadsCsrWorkspace {
 public:
  explicit DeltaSteppingThreadsCsrWorkspace(const HostCsrF32& adjacency,
                                            hipStream_t stream = nullptr);
  ~DeltaSteppingThreadsCsrWorkspace();

  DeltaSteppingThreadsCsrWorkspace(const DeltaSteppingThreadsCsrWorkspace&) = delete;
  DeltaSteppingThreadsCsrWorkspace& operator=(const DeltaSteppingThreadsCsrWorkspace&) = delete;

  DeltaSteppingThreadsCsrWorkspace(DeltaSteppingThreadsCsrWorkspace&&) noexcept;
  DeltaSteppingThreadsCsrWorkspace& operator=(DeltaSteppingThreadsCsrWorkspace&&) noexcept;

  void update_values(const std::vector<float>& values,
                     hipStream_t stream = nullptr);

  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  DeltaSteppingThreadsCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingThreadsCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingThreadsCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsCsrResult delta_stepping_threads_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream);

inline DeltaSteppingThreadsCsrResult delta_stepping_threads_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_threads_minplus_hip_csr(d_adjacency, source, target, delta,
                                                max_iters, stream,
                                                progress_callback, progress_user_data);
}

inline DeltaSteppingThreadsCsrResult delta_stepping_threads_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsCsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr) {
  return delta_stepping_threads_minplus_hip_csr(adjacency, source, target, delta,
                                                max_iters, stream,
                                                progress_callback, progress_user_data);
}
