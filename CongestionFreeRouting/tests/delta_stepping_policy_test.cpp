#include "../delta_stepping/delta_stepping_policy.hpp"
#include "../sssp_query_capacity.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void test_row_offset_policy() {
  constexpr std::int64_t kU32Max =
      static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
  static_assert(!delta_stepping_compact_row_offsets_eligible(-1));
  static_assert(delta_stepping_compact_row_offsets_eligible(0));
  static_assert(delta_stepping_compact_row_offsets_eligible(kU32Max));
  static_assert(!delta_stepping_compact_row_offsets_eligible(kU32Max + 1));
  static_assert(delta_stepping_device_row_offset_width(
                    0, DeltaSteppingCsrOffsetMode::kAuto) ==
                DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit);
  static_assert(delta_stepping_device_row_offset_width(
                    0, DeltaSteppingCsrOffsetMode::kForce64Bit) ==
                DeltaSteppingCsrDeviceRowOffsetWidth::k64Bit);

  const DeltaSteppingCsrRowOffsetPlan edgeless =
      delta_stepping_row_offset_plan(
          3, 0, DeltaSteppingCsrOffsetMode::kAuto);
  require(edgeless.uses_32_bit_offsets() && edgeless.entry_count == 4 &&
              edgeless.byte_count == 4 * sizeof(std::uint32_t),
          "edgeless compact row-offset plan is incorrect");
  const DeltaSteppingCsrRowOffsetPlan forced =
      delta_stepping_row_offset_plan(
          3, 0, DeltaSteppingCsrOffsetMode::kForce64Bit);
  require(!forced.uses_32_bit_offsets() &&
              forced.byte_count == 4 * sizeof(std::int64_t),
          "forced-wide row-offset plan is incorrect");

  const std::vector<std::uint32_t> compact =
      delta_stepping_compact_row_offsets({0, 2, 2, kU32Max});
  require(compact == std::vector<std::uint32_t>(
                         {0, 2, 2,
                          std::numeric_limits<std::uint32_t>::max()}),
          "compact row-offset conversion changed values");
  require_throws<std::overflow_error>(
      [=] { (void)delta_stepping_compact_row_offsets({0, kU32Max + 1}); },
      "overflowing compact row offset was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)delta_stepping_row_offset_plan(
            -1, 0, DeltaSteppingCsrOffsetMode::kAuto);
      },
      "negative row count was accepted");
}

void test_result_shape_policy() {
  for (const DeltaSteppingCsrPolicyQueryKind kind : {
           DeltaSteppingCsrPolicyQueryKind::kDistancesOnly,
           DeltaSteppingCsrPolicyQueryKind::kCompactTargets,
           DeltaSteppingCsrPolicyQueryKind::kLegacyParents}) {
    require(delta_stepping_result_shape_model(
                kind, DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit) ==
                delta_stepping_result_shape_model(
                    kind, DeltaSteppingCsrDeviceRowOffsetWidth::k64Bit),
            "device row width changed the public result shape");
  }
  const auto compact = delta_stepping_result_shape_model(
      DeltaSteppingCsrPolicyQueryKind::kCompactTargets,
      DeltaSteppingCsrDeviceRowOffsetWidth::k32Bit);
  require(compact.compact_target_paths &&
              compact.compact_path_edges_64_bit &&
              !compact.predecessor_nodes,
          "compact result shape lost public 64-bit path edges");
}

void test_generation_membership_model() {
  DeltaSteppingCurrentMembershipModel model(5);
  const std::uint32_t sources = model.begin_frontier();
  require(sources == 1 && model.claim_current(0) &&
              !model.claim_current(0),
          "duplicate source/current claims were not suppressed");

  const std::uint32_t next = model.begin_frontier();
  require(next != sources && model.claim_current(0),
          "processed vertex could not re-enqueue after a same-bucket decrease");
  require(!model.claim_current(0),
          "same next generation admitted a duplicate candidate");

  require(model.claim_pending(2) && !model.claim_pending(2),
          "pending duplicate candidates were not suppressed");
  model.begin_frontier();
  require(model.promote_pending(2) && model.is_current(2) &&
              !model.is_pending(2),
          "pending-to-current transition failed");

  // A zero-weight cycle that does not strictly decrease distance publishes no
  // membership claim; the model therefore remains unchanged.
  require(!model.claim_current_after_relaxation(3, false) &&
              !model.is_current(3),
          "zero-weight no-decrease transition appended a vertex");

  require(model.claim_pending(4),
          "callback cleanup fixture could not create pending membership");
  model.finish_or_abort();
  require(!model.is_pending(4),
          "callback-like abort did not clean pending membership");
  model.begin_frontier();
  require(model.claim_current(2),
          "early-stop stale membership blocked workspace reuse");
  model.finish_or_abort();
  model.begin_frontier();
  require(model.claim_current(2),
          "callback-like abort stale membership blocked workspace reuse");

  model.force_last_token_for_test(
      std::numeric_limits<std::uint32_t>::max());
  require(model.begin_frontier() == 1 && model.rollover_count() == 1 &&
              model.claim_current(0),
          "generation rollover did not clear stale tags safely");
}

void test_allocation_policy() {
  constexpr std::size_t kDeviceLimit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  require(delta_stepping_device_geometric_capacity(0, 4) == 4 &&
              delta_stepping_device_geometric_capacity(4, 3) == 4 &&
              delta_stepping_device_geometric_capacity(4, 5) == 7 &&
              delta_stepping_device_geometric_capacity(kDeviceLimit - 1,
                                                       kDeviceLimit) ==
                  kDeviceLimit,
          "Delta geometric capacity did not retain or bound its high water");
  require_throws<std::overflow_error>(
      [=] {
        (void)delta_stepping_device_geometric_capacity(
            kDeviceLimit, kDeviceLimit + 1);
      },
      "Delta geometric capacity accepted an unrepresentable count");

  const auto compact = delta_stepping_allocation_plan(
      true, DeltaSteppingCsrPolicyQueryKind::kCompactTargets, 4, 7);
  require(compact.reserve_sources && compact.reserve_targets &&
              compact.reserve_target_state && compact.reserve_target_offsets &&
              compact.allocate_parent_key &&
              !compact.allocate_legacy_parents &&
              !compact.allocate_compact_paths,
          "compact-parent allocation plan is incorrect");

  const auto distances = delta_stepping_allocation_plan(
      false, DeltaSteppingCsrPolicyQueryKind::kDistancesOnly, 4, 7);
  require(distances.reserve_sources && !distances.reserve_targets &&
              !distances.reserve_target_state &&
              !distances.reserve_target_offsets &&
              !distances.allocate_parent_key &&
              !distances.allocate_legacy_parents &&
              !distances.allocate_compact_paths,
          "strict distances-only allocation plan retained path state");

  const auto legacy = delta_stepping_allocation_plan(
      true, DeltaSteppingCsrPolicyQueryKind::kLegacyParents, 0, 3);
  require(legacy.allocate_legacy_parents && legacy.allocate_parent_key,
          "legacy allocation plan omitted deterministic parent state");

  require_throws<std::overflow_error>(
      [] {
        SsspQueryCapacityHints hints;
        hints.max_targets =
            static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        sssp_capacity::validate_reservation(hints);
      },
      "unrepresentable Delta target reservation was accepted");
  require_throws<std::overflow_error>(
      [] {
        (void)sssp_capacity::geometric_capacity(
            std::numeric_limits<std::size_t>::max() - 1,
            std::numeric_limits<std::size_t>::max());
      },
      "overflowing Delta geometric growth was accepted");
}

}  // namespace

int main() {
  test_row_offset_policy();
  test_result_shape_policy();
  test_generation_membership_model();
  test_allocation_policy();
  std::cout << "Delta-Stepping policy test passed\n";
  return 0;
}
