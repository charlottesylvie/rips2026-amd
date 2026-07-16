#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

// Hybrid Delta-Stepping SSSP for nonnegative edge weights.
//
// Version 2 keeps one GPU thread per low-degree active vertex and uses one
// wavefront per high-degree vertex.  It also queues a vertex for the heavy
// phase only when that vertex actually has an edge heavier than delta.
// CSR convention is outgoing: adjacency row u, column v is u -> v.
using DeltaSteppingThreadsV2CsrProgress = BellmanFordCsrProgress;
using DeltaSteppingThreadsV2CsrProgressCallback = BellmanFordCsrProgressCallback;
using DeltaSteppingThreadsV2CsrResult = BellmanFordCsrResult;

class DeltaSteppingThreadsV2CsrWorkspace {
 public:
  explicit DeltaSteppingThreadsV2CsrWorkspace(const HostCsrF32& adjacency,
                                              hipStream_t stream = nullptr);
  ~DeltaSteppingThreadsV2CsrWorkspace();

  DeltaSteppingThreadsV2CsrWorkspace(const DeltaSteppingThreadsV2CsrWorkspace&) = delete;
  DeltaSteppingThreadsV2CsrWorkspace& operator=(const DeltaSteppingThreadsV2CsrWorkspace&) = delete;

  DeltaSteppingThreadsV2CsrWorkspace(DeltaSteppingThreadsV2CsrWorkspace&&) noexcept;
  DeltaSteppingThreadsV2CsrWorkspace& operator=(DeltaSteppingThreadsV2CsrWorkspace&&) noexcept;

  void update_values(const std::vector<float>& values,
                     hipStream_t stream = nullptr);
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  DeltaSteppingThreadsV2CsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingThreadsV2CsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  DeltaSteppingThreadsV2CsrResult run(
      int source,
      int target,
      float delta,
      int max_iters = -1,
      hipStream_t stream = nullptr,
      DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const minplus_sparse::DeviceCsrF32& d_adjacency,
    int source,
    float delta,
    hipStream_t stream);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const HostCsrF32& adjacency,
    const std::vector<int>& sources,
    int target,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    int max_iters = -1,
    hipStream_t stream = nullptr,
    DeltaSteppingThreadsV2CsrProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

DeltaSteppingThreadsV2CsrResult delta_stepping_threads_v2_minplus_hip_csr(
    const HostCsrF32& adjacency,
    int source,
    float delta,
    hipStream_t stream);
