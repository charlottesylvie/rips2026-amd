#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Host-only policy and state models shared by the Delta-Stepping production
// dispatch and its CPU tests.  This header deliberately has no HIP dependency.

enum class DeltaSteppingCsrOffsetMode {
  kAuto,
  kForce64Bit,
};

enum class DeltaSteppingCsrCurrentMembershipMode {
  // Preserve the established Boolean membership plus clear-kernel path.
  kBoolean,
  // Tag next-frontier membership with a distinct light-round generation.
  kGeneration,
};

enum class DeltaSteppingCsrDeviceRowOffsetWidth {
  k32Bit,
  k64Bit,
};

// rowptr includes the terminal nnz value.  UINT32_MAX edges are eligible, but
// 2^32 edges are not: the latter terminal offset does not fit in uint32_t.
constexpr bool delta_stepping_compact_row_offsets_eligible(
    std::int64_t nnz) noexcept {
  return nnz >= 0 &&
         static_cast<std::uint64_t>(nnz) <=
             static_cast<std::uint64_t>(
                 std::numeric_limits<std::uint32_t>::max());
}

constexpr DeltaSteppingCsrDeviceRowOffsetWidth
delta_stepping_device_row_offset_width(
    std::int64_t nnz,
    DeltaSteppingCsrOffsetMode mode) noexcept {
  return mode == DeltaSteppingCsrOffsetMode::kAuto &&
                 delta_stepping_compact_row_offsets_eligible(nnz)
             ? DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit
             : DeltaSteppingCsrDeviceRowOffsetWidth::k64Bit;
}

struct DeltaSteppingCsrRowOffsetPlan {
  DeltaSteppingCsrDeviceRowOffsetWidth width =
      DeltaSteppingCsrDeviceRowOffsetWidth::k64Bit;
  std::size_t entry_count = 0;
  std::size_t byte_count = 0;

  constexpr bool uses_32_bit_offsets() const noexcept {
    return width == DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit;
  }
};

inline DeltaSteppingCsrRowOffsetPlan delta_stepping_row_offset_plan(
    std::int64_t rows,
    std::int64_t nnz,
    DeltaSteppingCsrOffsetMode mode) {
  if (rows < 0) {
    throw std::invalid_argument("Delta-Stepping row count must be nonnegative");
  }
  if (nnz < 0) {
    throw std::invalid_argument("Delta-Stepping nnz must be nonnegative");
  }
  const std::uint64_t unsigned_rows = static_cast<std::uint64_t>(rows);
  if (unsigned_rows >=
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(
        "Delta-Stepping row-offset entry count overflows size_t");
  }
  const std::size_t entry_count = static_cast<std::size_t>(unsigned_rows) + 1;
  const DeltaSteppingCsrDeviceRowOffsetWidth width =
      delta_stepping_device_row_offset_width(nnz, mode);
  const std::size_t element_bytes =
      width == DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit
          ? sizeof(std::uint32_t)
          : sizeof(std::int64_t);
  if (entry_count >
      std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::overflow_error(
        "Delta-Stepping row-offset byte count overflows size_t");
  }
  return {width, entry_count, entry_count * element_bytes};
}

inline std::vector<std::uint32_t> delta_stepping_compact_row_offsets(
    const std::vector<std::int64_t>& row_offsets) {
  std::vector<std::uint32_t> compact;
  compact.reserve(row_offsets.size());
  for (const std::int64_t offset : row_offsets) {
    if (!delta_stepping_compact_row_offsets_eligible(offset)) {
      throw std::overflow_error(
          "Delta-Stepping CSR row offset does not fit in uint32_t");
    }
    compact.push_back(static_cast<std::uint32_t>(offset));
  }
  return compact;
}

struct DeltaSteppingGenerationAdvance {
  std::uint32_t token = 1;
  bool reset_required = false;
};

constexpr DeltaSteppingGenerationAdvance
delta_stepping_advance_generation(std::uint32_t previous) noexcept {
  return previous == std::numeric_limits<std::uint32_t>::max()
             ? DeltaSteppingGenerationAdvance{1, true}
             : DeltaSteppingGenerationAdvance{
                   static_cast<std::uint32_t>(previous + 1), false};
}

// Query counts and compact offsets are passed to kernels as int. Geometric
// growth must therefore stop at that representable ceiling even when size_t
// could describe a larger allocation.
constexpr std::size_t delta_stepping_device_geometric_capacity(
    std::size_t current,
    std::size_t required) {
  const std::size_t limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (current > limit || required > limit) {
    throw std::overflow_error(
        "Delta-Stepping capacity exceeds device int range");
  }
  if (current >= required) return current;
  if (current == 0) return required;
  const std::size_t grown = current + current / 2 + 1;
  return std::min(limit, std::max(required, grown));
}

// Sequential model of the optional in_current representation.  A new token is
// acquired for every produced frontier.  Stale tags never block a claim, and
// rollover clears all tags before token reuse.
class DeltaSteppingCurrentMembershipModel {
 public:
  explicit DeltaSteppingCurrentMembershipModel(std::size_t vertex_count)
      : tags_(vertex_count, 0), pending_(vertex_count, false) {}

  std::uint32_t begin_frontier() {
    const DeltaSteppingGenerationAdvance advance =
        delta_stepping_advance_generation(last_token_);
    if (advance.reset_required) {
      std::fill(tags_.begin(), tags_.end(), 0);
      ++rollovers_;
    }
    last_token_ = advance.token;
    active_token_ = advance.token;
    return active_token_;
  }

  bool claim_current(std::size_t vertex) {
    require_vertex(vertex);
    if (active_token_ == 0) {
      throw std::logic_error(
          "Delta-Stepping membership claim requires an active frontier");
    }
    if (tags_[vertex] == active_token_) return false;
    tags_[vertex] = active_token_;
    return true;
  }

  bool claim_current_after_relaxation(std::size_t vertex,
                                      bool strictly_decreased) {
    return strictly_decreased && claim_current(vertex);
  }

  bool claim_pending(std::size_t vertex) {
    require_vertex(vertex);
    if (pending_[vertex]) return false;
    pending_[vertex] = true;
    return true;
  }

  bool promote_pending(std::size_t vertex) {
    require_vertex(vertex);
    if (!pending_[vertex]) return false;
    pending_[vertex] = false;
    return claim_current(vertex);
  }

  bool is_current(std::size_t vertex) const {
    require_vertex(vertex);
    return active_token_ != 0 && tags_[vertex] == active_token_;
  }

  bool is_pending(std::size_t vertex) const {
    require_vertex(vertex);
    return pending_[vertex];
  }

  std::uint32_t active_token() const noexcept { return active_token_; }
  std::size_t rollover_count() const noexcept { return rollovers_; }

  // Early target stops and callback-like aborts leave tags untouched.  The
  // next produced frontier receives a distinct token, so old tags are stale.
  void finish_or_abort() noexcept {
    active_token_ = 0;
    std::fill(pending_.begin(), pending_.end(), false);
  }

  void force_last_token_for_test(std::uint32_t token) noexcept {
    last_token_ = token;
    active_token_ = 0;
  }

 private:
  void require_vertex(std::size_t vertex) const {
    if (vertex >= tags_.size()) {
      throw std::out_of_range(
          "Delta-Stepping membership vertex is outside the model");
    }
  }

  std::vector<std::uint32_t> tags_;
  std::vector<bool> pending_;
  std::uint32_t last_token_ = 0;
  std::uint32_t active_token_ = 0;
  std::size_t rollovers_ = 0;
};

enum class DeltaSteppingCsrPolicyQueryKind {
  kDistancesOnly,
  kCompactTargets,
  kLegacyParents,
};

struct DeltaSteppingCsrAllocationPlan {
  bool reserve_sources = false;
  bool reserve_targets = false;
  bool reserve_target_state = false;
  bool reserve_target_offsets = false;
  bool allocate_compact_paths = false;
  bool allocate_parent_key = false;
  bool allocate_legacy_parents = false;
};

constexpr DeltaSteppingCsrAllocationPlan delta_stepping_allocation_plan(
    bool path_capable,
    DeltaSteppingCsrPolicyQueryKind query_kind,
    std::size_t source_hint,
    std::size_t target_hint) noexcept {
  DeltaSteppingCsrAllocationPlan plan;
  plan.reserve_sources = source_hint != 0;
  if (!path_capable ||
      query_kind == DeltaSteppingCsrPolicyQueryKind::kDistancesOnly) {
    return plan;
  }
  plan.reserve_targets = target_hint != 0;
  plan.reserve_target_state = target_hint != 0;
  plan.reserve_target_offsets = target_hint != 0;
  plan.allocate_parent_key =
      query_kind == DeltaSteppingCsrPolicyQueryKind::kCompactTargets ||
      query_kind == DeltaSteppingCsrPolicyQueryKind::kLegacyParents;
  plan.allocate_legacy_parents =
      query_kind == DeltaSteppingCsrPolicyQueryKind::kLegacyParents;
  // Compact path totals are query results and are never inferred from hints.
  plan.allocate_compact_paths = false;
  return plan;
}

struct DeltaSteppingCsrResultShapeModel {
  bool distances = false;
  bool predecessor_nodes = false;
  bool predecessor_edges_64_bit = false;
  bool compact_target_paths = false;
  bool compact_path_edges_64_bit = false;
};

constexpr DeltaSteppingCsrResultShapeModel
delta_stepping_result_shape_model(
    DeltaSteppingCsrPolicyQueryKind query_kind,
    DeltaSteppingCsrDeviceRowOffsetWidth) noexcept {
  switch (query_kind) {
    case DeltaSteppingCsrPolicyQueryKind::kDistancesOnly:
      return {true, false, false, false, false};
    case DeltaSteppingCsrPolicyQueryKind::kCompactTargets:
      return {false, false, false, true, true};
    case DeltaSteppingCsrPolicyQueryKind::kLegacyParents:
      return {true, true, true, false, false};
  }
  return {};
}

constexpr bool operator==(const DeltaSteppingCsrResultShapeModel& lhs,
                          const DeltaSteppingCsrResultShapeModel& rhs) noexcept {
  return lhs.distances == rhs.distances &&
         lhs.predecessor_nodes == rhs.predecessor_nodes &&
         lhs.predecessor_edges_64_bit == rhs.predecessor_edges_64_bit &&
         lhs.compact_target_paths == rhs.compact_target_paths &&
         lhs.compact_path_edges_64_bit == rhs.compact_path_edges_64_bit;
}
