#pragma once

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

using UnitBfsCsrProgress = BellmanFordCsrProgress;
using UnitBfsCsrProgressCallback = BellmanFordCsrProgressCallback;
using UnitBfsCsrResult = BellmanFordCsrResult;

class UnitBfsCsrWorkspace {
 public:
  struct Impl;

  explicit UnitBfsCsrWorkspace(const HostCsrF32& adjacency,
                               hipStream_t stream = nullptr);
  ~UnitBfsCsrWorkspace();

  UnitBfsCsrWorkspace(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace& operator=(const UnitBfsCsrWorkspace&) = delete;
  UnitBfsCsrWorkspace(UnitBfsCsrWorkspace&&) noexcept;
  UnitBfsCsrWorkspace& operator=(UnitBfsCsrWorkspace&&) noexcept;

  UnitBfsCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  UnitBfsCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  UnitBfsCsrResult run(
      int source,
      int target,
      float delta,
      int max_depth,
      hipStream_t stream = nullptr,
      UnitBfsCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
