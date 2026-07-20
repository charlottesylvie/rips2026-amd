#pragma once

#include "bf_hip_CSR.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace delta_stepping_auto_detail {

inline std::size_t checked_auto_delta_size(minplus_sparse::Offset value,
                                           const char* name) {
  if (value < 0 ||
      static_cast<unsigned long long>(value) >
          static_cast<unsigned long long>(
              std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(std::string(name) +
                                " does not fit in host address space");
  }
  return static_cast<std::size_t>(value);
}

inline float clamp_auto_delta(double seed, float multiplier) {
  const double scaled = seed * static_cast<double>(multiplier);
  const double min_delta =
      static_cast<double>(std::numeric_limits<float>::min());
  const double max_delta =
      static_cast<double>(std::numeric_limits<float>::max());
  if (scaled < min_delta) return std::numeric_limits<float>::min();
  if (!std::isfinite(scaled) || scaled > max_delta) {
    return std::numeric_limits<float>::max();
  }
  return static_cast<float>(scaled);
}

inline float compute_auto_delta(
    const HostCsrF32& adjacency,
    const std::vector<float>* vertex_costs,
    int wavefront_size,
    float multiplier) {
  if (wavefront_size <= 0) {
    throw std::invalid_argument("wavefront size must be positive");
  }
  if (!(multiplier > 0.0f) || !std::isfinite(multiplier)) {
    throw std::invalid_argument(
        "automatic delta multiplier must be finite and positive");
  }
  if (adjacency.rows <= 0 || adjacency.rows != adjacency.cols) {
    throw std::invalid_argument(
        "automatic delta expects a nonempty square CSR graph");
  }

  const std::size_t rows =
      checked_auto_delta_size(adjacency.rows, "CSR row count");
  const std::size_t nnz =
      checked_auto_delta_size(adjacency.nnz, "CSR edge count");
  if (adjacency.rowptr.size() != rows + 1 ||
      adjacency.colind.size() != nnz || adjacency.values.size() != nnz) {
    throw std::invalid_argument(
        "CSR array sizes do not match automatic delta graph dimensions");
  }
  if (adjacency.rowptr.front() != 0 ||
      adjacency.rowptr.back() != adjacency.nnz) {
    throw std::invalid_argument("CSR rowptr must start at zero and end at nnz");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const minplus_sparse::Offset begin = adjacency.rowptr[row];
    const minplus_sparse::Offset end = adjacency.rowptr[row + 1];
    if (begin < 0 || end < begin || end > adjacency.nnz) {
      throw std::invalid_argument(
          "CSR rowptr must be monotone and remain within [0, nnz]");
    }
  }

  if (vertex_costs != nullptr && vertex_costs->size() != rows) {
    throw std::invalid_argument(
        "vertex cost size does not match automatic delta graph rows");
  }
  if (vertex_costs != nullptr) {
    for (const float cost : *vertex_costs) {
      if (!std::isfinite(cost) || cost < 0.0f) {
        throw std::invalid_argument(
            "automatic delta vertex costs must be finite and nonnegative");
      }
    }
  }

  // Double precision is sufficient even for the largest addressable graph:
  // FLT_MAX^2 times a 64-bit edge count remains far below DBL_MAX. Compensated
  // summation keeps the final float seed stable without a slow long-double
  // scan on hosts where long double uses x87 instructions.
  double effective_weight_sum = 0.0;
  double compensation = 0.0;
  for (std::size_t edge = 0; edge < nnz; ++edge) {
    const minplus_sparse::Index destination = adjacency.colind[edge];
    const float weight = adjacency.values[edge];
    if (destination < 0 ||
        static_cast<minplus_sparse::Offset>(destination) >= adjacency.cols) {
      throw std::invalid_argument(
          "CSR colind contains an out-of-range destination vertex");
    }
    if (!std::isfinite(weight) || weight < 0.0f) {
      throw std::invalid_argument(
          "automatic delta edge weights must be finite and nonnegative");
    }
    const double destination_cost =
        vertex_costs == nullptr
            ? 1.0
            : static_cast<double>(
                  (*vertex_costs)[static_cast<std::size_t>(destination)]);
    const double effective_weight =
        static_cast<double>(weight) * destination_cost;
    const double adjusted = effective_weight - compensation;
    const double next = effective_weight_sum + adjusted;
    compensation = (next - effective_weight_sum) - adjusted;
    effective_weight_sum = next;
  }

  // With no edges or only zero effective weights, every relaxation is light
  // for any positive width. Use a stable unit seed and still honor the sweep
  // multiplier instead of returning the invalid formula value zero.
  if (nnz == 0 || effective_weight_sum == 0.0) {
    return clamp_auto_delta(1.0, multiplier);
  }

  const double edge_count = static_cast<double>(nnz);
  const double average_effective_weight =
      effective_weight_sum / edge_count;
  const double average_out_degree =
      edge_count / static_cast<double>(rows);
  const double seed =
      static_cast<double>(wavefront_size) * average_effective_weight /
      average_out_degree;
  return clamp_auto_delta(seed, multiplier);
}

}  // namespace delta_stepping_auto_detail

// cuGraph-inspired graph-aware bucket-width seed:
//   wavefront_size * average_effective_weight / average_out_degree.
// The multiplier supports an explicit sweep around that seed. The overload
// with vertex costs uses the algorithm's destination-cost convention exactly:
// effective_weight(u, v) = edge_weight(u, v) * vertex_cost(v).
// Callers that change weights or costs must invoke the helper again; it does
// not cache statistics in a mutable workspace.
inline float delta_stepping_auto_delta(const HostCsrF32& adjacency,
                                       int wavefront_size,
                                       float multiplier = 1.0f) {
  return delta_stepping_auto_detail::compute_auto_delta(
      adjacency, nullptr, wavefront_size, multiplier);
}

inline float delta_stepping_auto_delta(
    const HostCsrF32& adjacency,
    const std::vector<float>& vertex_costs,
    int wavefront_size,
    float multiplier = 1.0f) {
  return delta_stepping_auto_detail::compute_auto_delta(
      adjacency, &vertex_costs, wavefront_size, multiplier);
}
