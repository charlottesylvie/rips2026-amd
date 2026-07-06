#pragma once

#include "bf_hip_CSR.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bf_csr_detail {

constexpr float INF = std::numeric_limits<float>::infinity();

using minplus_sparse::DeviceCsrF32;
using minplus_sparse::Index;
using minplus_sparse::Offset;

inline void check_hip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             hipGetErrorString(status));
  }
}

inline void validate_device_csr_adjacency(const DeviceCsrF32& adjacency,
                                          int source) {
  if (adjacency.rows <= 0 || adjacency.cols <= 0) {
    throw std::invalid_argument("CSR adjacency dimensions must be positive");
  }
  if (adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR adjacency must be square");
  }
  if (adjacency.rows >
      static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CSR adjacency has too many rows for int API");
  }
  if (source < 0 || static_cast<Offset>(source) >= adjacency.rows) {
    throw std::invalid_argument("source is outside graph");
  }
  if (!adjacency.rowptr) {
    throw std::invalid_argument("CSR adjacency rowptr must not be null");
  }
  if (adjacency.nnz < 0) {
    throw std::invalid_argument("CSR adjacency nnz must be nonnegative");
  }
  if (adjacency.nnz > 0 && (!adjacency.colind || !adjacency.values)) {
    throw std::invalid_argument(
        "CSR adjacency colind and values must not be null when nnz > 0");
  }
}

inline void validate_host_csr_adjacency(const HostCsrF32& adjacency,
                                        int source) {
  if (adjacency.rows <= 0 || adjacency.cols <= 0) {
    throw std::invalid_argument("CSR adjacency dimensions must be positive");
  }
  if (adjacency.rows != adjacency.cols) {
    throw std::invalid_argument("CSR adjacency must be square");
  }
  if (adjacency.rows >
      static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CSR adjacency has too many rows for int API");
  }
  if (source < 0 || static_cast<Offset>(source) >= adjacency.rows) {
    throw std::invalid_argument("source is outside graph");
  }
  if (adjacency.nnz < 0) {
    throw std::invalid_argument("CSR adjacency nnz must be nonnegative");
  }
  if (adjacency.rowptr.size() !=
      static_cast<std::size_t>(adjacency.rows + 1)) {
    throw std::invalid_argument("CSR rowptr must contain rows + 1 entries");
  }
  if (adjacency.colind.size() != static_cast<std::size_t>(adjacency.nnz) ||
      adjacency.values.size() != static_cast<std::size_t>(adjacency.nnz)) {
    throw std::invalid_argument(
        "CSR colind and values must each contain nnz entries");
  }
  if (adjacency.rowptr.empty() || adjacency.rowptr.front() != 0 ||
      adjacency.rowptr.back() != adjacency.nnz) {
    throw std::invalid_argument(
        "CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < adjacency.rows; ++row) {
    const Offset begin = adjacency.rowptr[static_cast<std::size_t>(row)];
    const Offset end = adjacency.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > adjacency.nnz) {
      throw std::invalid_argument("CSR rowptr entries are invalid");
    }
  }

  for (Index col : adjacency.colind) {
    if (col < 0 || static_cast<Offset>(col) >= adjacency.cols) {
      throw std::invalid_argument("CSR column index is outside graph");
    }
  }
}

struct DeviceCsrOwner {
  DeviceCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* colind = nullptr;
  float* values = nullptr;

  DeviceCsrOwner() = default;

  DeviceCsrOwner(const DeviceCsrOwner&) = delete;
  DeviceCsrOwner& operator=(const DeviceCsrOwner&) = delete;

  DeviceCsrOwner(DeviceCsrOwner&& other) noexcept {
    *this = std::move(other);
  }

  DeviceCsrOwner& operator=(DeviceCsrOwner&& other) noexcept {
    if (this != &other) {
      reset();
      view = other.view;
      rowptr = other.rowptr;
      colind = other.colind;
      values = other.values;
      other.view = {};
      other.rowptr = nullptr;
      other.colind = nullptr;
      other.values = nullptr;
    }
    return *this;
  }

  ~DeviceCsrOwner() {
    reset();
  }

  void reset() {
    if (rowptr) {
      (void)hipFree(rowptr);
      rowptr = nullptr;
    }
    if (colind) {
      (void)hipFree(colind);
      colind = nullptr;
    }
    if (values) {
      (void)hipFree(values);
      values = nullptr;
    }
    view = {};
  }
};

inline DeviceCsrOwner copy_host_csr_to_device(const HostCsrF32& host,
                                              hipStream_t stream) {
  DeviceCsrOwner device;

  const std::size_t rowptr_bytes = host.rowptr.size() * sizeof(Offset);
  check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr), rowptr_bytes),
            "hipMalloc CSR rowptr");
  check_hip(hipMemcpyAsync(device.rowptr,
                           host.rowptr.data(),
                           rowptr_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy CSR rowptr to device");

  if (host.nnz > 0) {
    const std::size_t colind_bytes = host.colind.size() * sizeof(Index);
    const std::size_t values_bytes = host.values.size() * sizeof(float);

    check_hip(hipMalloc(reinterpret_cast<void**>(&device.colind),
                        colind_bytes),
              "hipMalloc CSR colind");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.values),
                        values_bytes),
              "hipMalloc CSR values");
    check_hip(hipMemcpyAsync(device.colind,
                             host.colind.data(),
                             colind_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy CSR colind to device");
    check_hip(hipMemcpyAsync(device.values,
                             host.values.data(),
                             values_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy CSR values to device");
  }

  check_hip(hipStreamSynchronize(stream), "synchronize CSR copy");

  device.view.rows = host.rows;
  device.view.cols = host.cols;
  device.view.nnz = host.nnz;
  device.view.rowptr = device.rowptr;
  device.view.colind = device.colind;
  device.view.values = device.values;
  return device;
}

static __global__ void init_single_source_vector_kernel(Offset rows,
                                                        int source,
                                                        Offset* rowptr,
                                                        Index* colind,
                                                        float* values) {
  const Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i <= rows) {
    rowptr[i] = i <= static_cast<Offset>(source) ? 0 : 1;
  }
  if (i == 0) {
    colind[0] = 0;
    values[0] = 0.0f;
  }
}

inline DeviceCsrOwner make_device_single_source_vector(Offset rows,
                                                       int source,
                                                       hipStream_t stream) {
  DeviceCsrOwner vector;
  const std::size_t rowptr_bytes =
      static_cast<std::size_t>(rows + 1) * sizeof(Offset);
  const std::size_t entry_capacity =
      static_cast<std::size_t>(rows) * sizeof(Index);
  const std::size_t value_capacity =
      static_cast<std::size_t>(rows) * sizeof(float);

  check_hip(hipMalloc(reinterpret_cast<void**>(&vector.rowptr), rowptr_bytes),
            "hipMalloc source rowptr");
  check_hip(hipMalloc(reinterpret_cast<void**>(&vector.colind),
                      entry_capacity),
            "hipMalloc source colind");
  check_hip(hipMalloc(reinterpret_cast<void**>(&vector.values),
                      value_capacity),
            "hipMalloc source values");

  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows + 1 + threads - 1) / threads);
  hipLaunchKernelGGL(init_single_source_vector_kernel,
                     dim3(blocks),
                     dim3(threads),
                     0,
                     stream,
                     rows,
                     source,
                     vector.rowptr,
                     vector.colind,
                     vector.values);
  check_hip(hipGetLastError(), "launch source CSR initialization kernel");

  vector.view.rows = rows;
  vector.view.cols = 1;
  vector.view.nnz = rows;
  vector.view.rowptr = vector.rowptr;
  vector.view.colind = vector.colind;
  vector.view.values = vector.values;
  return vector;
}

static __device__ float row_min_col_zero(const Offset* rowptr,
                                         const Index* colind,
                                         const float* values,
                                         Offset row) {
  float best = INF;
  for (Offset p = rowptr[row]; p < rowptr[row + 1]; ++p) {
    if (colind[p] == 0 && values[p] < best) {
      best = values[p];
    }
  }
  return best;
}

static __global__ void compute_merge_counts_kernel(
    Offset rows,
    const Offset* old_rowptr,
    const Index* old_colind,
    const float* old_values,
    const Offset* relaxed_rowptr,
    const Index* relaxed_colind,
    const float* relaxed_values,
    Offset* row_counts,
    float* merged_values_by_row,
    int* changed) {
  const Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float old_value =
      row_min_col_zero(old_rowptr, old_colind, old_values, row);
  const float relaxed_value =
      row_min_col_zero(relaxed_rowptr, relaxed_colind, relaxed_values, row);
  const float merged = relaxed_value < old_value ? relaxed_value : old_value;

  merged_values_by_row[row] = merged;
  row_counts[row] = (merged == merged && merged != INF) ? 1 : 0;

  if (relaxed_value < old_value) {
    atomicExch(changed, 1);
  }
}

static __global__ void set_last_rowptr_kernel(const Offset* row_counts,
                                              Offset* rowptr,
                                              Offset rows) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    rowptr[rows] = rows == 0 ? 0 : rowptr[rows - 1] + row_counts[rows - 1];
  }
}

static __global__ void write_merged_vector_kernel(
    Offset rows,
    const Offset* row_counts,
    const Offset* rowptr,
    const float* merged_values_by_row,
    Index* colind,
    float* values) {
  const Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || row_counts[row] == 0) {
    return;
  }

  const Offset dst = rowptr[row];
  colind[dst] = 0;
  values[dst] = merged_values_by_row[row];
}

inline DeviceCsrOwner merge_distances_on_device(const DeviceCsrF32& old_dist,
                                                const DeviceCsrF32& relaxed,
                                                int* h_changed,
                                                hipStream_t stream) {
  const Offset rows = old_dist.rows;
  DeviceCsrOwner merged;

  const std::size_t rowptr_bytes =
      static_cast<std::size_t>(rows + 1) * sizeof(Offset);
  const std::size_t counts_bytes = static_cast<std::size_t>(rows) *
                                   sizeof(Offset);
  const std::size_t colind_capacity =
      static_cast<std::size_t>(rows) * sizeof(Index);
  const std::size_t values_capacity =
      static_cast<std::size_t>(rows) * sizeof(float);

  check_hip(hipMalloc(reinterpret_cast<void**>(&merged.rowptr), rowptr_bytes),
            "hipMalloc merged rowptr");
  check_hip(hipMalloc(reinterpret_cast<void**>(&merged.colind),
                      colind_capacity),
            "hipMalloc merged colind");
  check_hip(hipMalloc(reinterpret_cast<void**>(&merged.values),
                      values_capacity),
            "hipMalloc merged values");

  Offset* d_row_counts = nullptr;
  float* d_merged_by_row = nullptr;
  int* d_changed = nullptr;
  void* d_scan_temp = nullptr;

  auto cleanup_temp = [&]() {
    if (d_row_counts) {
      (void)hipFree(d_row_counts);
      d_row_counts = nullptr;
    }
    if (d_merged_by_row) {
      (void)hipFree(d_merged_by_row);
      d_merged_by_row = nullptr;
    }
    if (d_changed) {
      (void)hipFree(d_changed);
      d_changed = nullptr;
    }
    if (d_scan_temp) {
      (void)hipFree(d_scan_temp);
      d_scan_temp = nullptr;
    }
  };

  try {
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_row_counts),
                        counts_bytes),
              "hipMalloc merge row counts");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_merged_by_row),
                        values_capacity),
              "hipMalloc merge values by row");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_changed), sizeof(int)),
              "hipMalloc merge changed flag");
    check_hip(hipMemsetAsync(d_changed, 0, sizeof(int), stream),
              "reset merge changed flag");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    hipLaunchKernelGGL(compute_merge_counts_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       rows,
                       old_dist.rowptr,
                       old_dist.colind,
                       old_dist.values,
                       relaxed.rowptr,
                       relaxed.colind,
                       relaxed.values,
                       d_row_counts,
                       d_merged_by_row,
                       d_changed);
    check_hip(hipGetLastError(), "launch CSR vector merge count kernel");

    std::size_t scan_temp_bytes = 0;
    check_hip(rocprim::exclusive_scan(nullptr,
                                      scan_temp_bytes,
                                      d_row_counts,
                                      merged.rowptr,
                                      static_cast<Offset>(0),
                                      static_cast<std::size_t>(rows),
                                      rocprim::plus<Offset>(),
                                      stream),
              "size merge rowptr scan temp");
    check_hip(hipMalloc(&d_scan_temp, scan_temp_bytes),
              "hipMalloc merge scan temp");
    check_hip(rocprim::exclusive_scan(d_scan_temp,
                                      scan_temp_bytes,
                                      d_row_counts,
                                      merged.rowptr,
                                      static_cast<Offset>(0),
                                      static_cast<std::size_t>(rows),
                                      rocprim::plus<Offset>(),
                                      stream),
              "scan merge row counts");

    hipLaunchKernelGGL(set_last_rowptr_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       d_row_counts,
                       merged.rowptr,
                       rows);
    check_hip(hipGetLastError(), "launch set merged rowptr tail kernel");

    hipLaunchKernelGGL(write_merged_vector_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       rows,
                       d_row_counts,
                       merged.rowptr,
                       d_merged_by_row,
                       merged.colind,
                       merged.values);
    check_hip(hipGetLastError(), "launch CSR vector merge write kernel");

    if (h_changed) {
      check_hip(hipMemcpyAsync(h_changed,
                               d_changed,
                               sizeof(int),
                               hipMemcpyDeviceToHost,
                               stream),
                "copy merge changed flag to host");
      check_hip(hipStreamSynchronize(stream), "synchronize merge changed flag");
    } else {
      check_hip(hipStreamSynchronize(stream), "synchronize CSR vector merge");
    }

    cleanup_temp();
  } catch (...) {
    cleanup_temp();
    throw;
  }

  merged.view.rows = rows;
  merged.view.cols = 1;
  merged.view.nnz = rows;
  merged.view.rowptr = merged.rowptr;
  merged.view.colind = merged.colind;
  merged.view.values = merged.values;
  return merged;
}

inline DeviceCsrOwner sparse_minplus_relax(const DeviceCsrF32& adjacency,
                                           const DeviceCsrF32& dist,
                                           hipStream_t stream) {
  DeviceCsrOwner relaxed;
  Offset relaxed_nnz = 0;

  hipError_t status = minplus_sparse::minplus_spgemm_csr_f32(
      adjacency,
      dist,
      &relaxed.rowptr,
      &relaxed.colind,
      &relaxed.values,
      &relaxed_nnz,
      stream);

  if (status == hipErrorInvalidValue) {
    status = minplus_sparse::minplus_spgemm_csr_f32_large_rows(
        adjacency,
        dist,
        &relaxed.rowptr,
        &relaxed.colind,
        &relaxed.values,
        &relaxed_nnz,
        stream);
  }
  check_hip(status, "sparse min-plus relaxation");

  relaxed.view.rows = adjacency.rows;
  relaxed.view.cols = 1;
  relaxed.view.nnz = relaxed_nnz;
  relaxed.view.rowptr = relaxed.rowptr;
  relaxed.view.colind = relaxed.colind;
  relaxed.view.values = relaxed.values;
  return relaxed;
}

inline std::vector<float> copy_dist_vector_to_host(const DeviceCsrF32& dist,
                                                   hipStream_t stream) {
  std::vector<float> dense(static_cast<std::size_t>(dist.rows), INF);
  std::vector<Offset> rowptr(static_cast<std::size_t>(dist.rows + 1));

  check_hip(hipMemcpyAsync(rowptr.data(),
                           dist.rowptr,
                           rowptr.size() * sizeof(Offset),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy final CSR rowptr to host");

  std::vector<Index> colind(static_cast<std::size_t>(dist.nnz));
  std::vector<float> values(static_cast<std::size_t>(dist.nnz));
  if (dist.nnz > 0) {
    check_hip(hipMemcpyAsync(colind.data(),
                             dist.colind,
                             colind.size() * sizeof(Index),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy final CSR colind to host");
    check_hip(hipMemcpyAsync(values.data(),
                             dist.values,
                             values.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy final CSR values to host");
  }
  check_hip(hipStreamSynchronize(stream), "synchronize final CSR copy");

  for (Offset row = 0; row < dist.rows; ++row) {
    for (Offset p = rowptr[static_cast<std::size_t>(row)];
         p < rowptr[static_cast<std::size_t>(row + 1)];
         ++p) {
      if (colind[static_cast<std::size_t>(p)] == 0) {
        const float value = values[static_cast<std::size_t>(p)];
        float& current = dense[static_cast<std::size_t>(row)];
        if (value < current) {
          current = value;
        }
      }
    }
  }

  return dense;
}

}  // namespace bf_csr_detail
