#include "../delta_stepping/delta_stepping_policy.hpp"
#include "../sssp_query_capacity.hpp"

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
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

bool same_controller_semantics(
    const DeltaSteppingCsrControllerDescriptor& lhs,
    const DeltaSteppingCsrControllerDescriptor& rhs) {
  return lhs.version == rhs.version &&
         lhs.query_sequence == rhs.query_sequence &&
         lhs.status == rhs.status && lhs.phase == rhs.phase &&
         lhs.light_rounds == rhs.light_rounds &&
         lhs.current_count == rhs.current_count &&
         lhs.pending_count == rhs.pending_count &&
         lhs.current_bucket == rhs.current_bucket &&
         lhs.next_bucket == rhs.next_bucket &&
         lhs.iterations == rhs.iterations;
}

void require_valid_controller_descriptor(
    const DeltaSteppingCsrControllerDescriptor& descriptor,
    const char* message) {
  require(delta_stepping_controller_descriptor_is_valid(descriptor), message);
}

void test_controller_policy_and_descriptor() {
  static_assert(sizeof(DeltaSteppingCsrControllerMode) ==
                sizeof(std::uint32_t));
  static_assert(sizeof(DeltaSteppingCsrControllerStatus) ==
                sizeof(std::uint32_t));
  static_assert(sizeof(DeltaSteppingCsrControllerPhase) ==
                sizeof(std::uint32_t));
  static_assert(sizeof(DeltaSteppingCsrControllerAction) ==
                sizeof(std::uint32_t));

  const DeltaSteppingCsrControllerPolicy defaults;
  require(defaults.mode == DeltaSteppingCsrControllerMode::kHostChecked &&
              defaults.batch_size ==
                  kDeltaSteppingCsrRecommendedControllerBatchSize &&
              delta_stepping_effective_controller_batch_size(defaults) == 1,
          "default controller policy did not preserve host checking");

  const DeltaSteppingCsrControllerPolicy reduced{
      DeltaSteppingCsrControllerMode::kReducedRoundTrip, 7};
  require(delta_stepping_effective_controller_batch_size(reduced) == 7,
          "reduced controller did not retain its bounded batch size");
  require_throws<std::invalid_argument>(
      [] {
        delta_stepping_validate_controller_policy(
            {DeltaSteppingCsrControllerMode::kReducedRoundTrip, 0});
      },
      "zero controller batch size was accepted");
  require_throws<std::invalid_argument>(
      [] {
        delta_stepping_validate_controller_policy(
            {DeltaSteppingCsrControllerMode::kReducedRoundTrip,
             std::numeric_limits<std::uint32_t>::max()});
      },
      "unrepresentable controller token budget was accepted");
  require_throws<std::invalid_argument>(
      [] {
        delta_stepping_validate_controller_policy(
            {static_cast<DeltaSteppingCsrControllerMode>(99), 1});
      },
      "unknown controller mode was accepted");

  const DeltaSteppingCsrControllerStatus overflow_and_stop =
      DeltaSteppingCsrControllerStatus::kComplete |
      DeltaSteppingCsrControllerStatus::kTargetSettled |
      DeltaSteppingCsrControllerStatus::kQueueOverflow;
  require(delta_stepping_controller_terminal_action(overflow_and_stop) ==
              DeltaSteppingCsrControllerAction::kStopQueueOverflow,
          "ordinary stop status masked queue overflow");
  require(delta_stepping_controller_terminal_action(
              overflow_and_stop |
              DeltaSteppingCsrControllerStatus::kInvalidState) ==
              DeltaSteppingCsrControllerAction::kStopInvalidState,
          "queue overflow masked invalid controller state");
  require(delta_stepping_controller_has_status(
              delta_stepping_normalize_controller_status(
                  static_cast<DeltaSteppingCsrControllerStatus>(
                      std::uint32_t{1} << 31)),
              DeltaSteppingCsrControllerStatus::kInvalidState),
          "unknown status bits did not become invalid state");

  DeltaSteppingCsrControllerDescriptor descriptor;
  require_valid_controller_descriptor(
      descriptor, "default controller descriptor is invalid");
  descriptor.version = 0;
  require(!delta_stepping_controller_descriptor_is_valid(descriptor),
          "controller descriptor accepted the wrong ABI version");
}

void test_controller_zero_and_light_rounds() {
  const DeltaSteppingCsrControllerPolicy reduced{
      DeltaSteppingCsrControllerMode::kReducedRoundTrip, 4};

  DeltaSteppingCsrControllerSequentialModel zero_model(reduced);
  const auto zero = zero_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(0));
  require(zero.status == DeltaSteppingCsrControllerStatus::kComplete &&
              zero.phase == DeltaSteppingCsrControllerPhase::kFinished &&
              zero.action ==
                  DeltaSteppingCsrControllerAction::kStopComplete &&
              zero.light_rounds == 0 && zero.iterations == 0 &&
              zero.publication_sequence == 1,
          "zero-work controller query did not complete immediately");
  require_valid_controller_descriptor(
      zero, "zero-work controller descriptor is invalid");

  DeltaSteppingCsrControllerSequentialModel one_round_model(reduced);
  const auto one_round_states = one_round_model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0),
  });
  require(one_round_states.size() == 2 &&
              one_round_states[0].phase ==
                  DeltaSteppingCsrControllerPhase::kLightClosure &&
              one_round_states[0].action ==
                  DeltaSteppingCsrControllerAction::kContinueDevice &&
              one_round_states[1].action ==
                  DeltaSteppingCsrControllerAction::kStopComplete &&
              one_round_states[1].light_rounds == 1 &&
              one_round_states[1].iterations == 1,
          "single light round did not complete correctly");

  // Four rounds with remaining work reach the exact bounded publication
  // boundary. Completion in the immediately following round must publish a
  // second, terminal descriptor.
  DeltaSteppingCsrControllerSequentialModel after_check_model(reduced);
  const auto after_check = after_check_model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 1, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kComplete),
  });
  require(after_check[3].action ==
              DeltaSteppingCsrControllerAction::kContinueDevice &&
              after_check[3].rounds_since_host_check == 3 &&
              after_check[3].publication_sequence == 0,
          "controller published before its exact batch boundary");
  require(after_check[4].action ==
              DeltaSteppingCsrControllerAction::kPublishHostCheck &&
              after_check[4].rounds_since_host_check == 4 &&
              after_check[4].publication_sequence == 1,
          "controller missed its exact batch boundary");
  require(after_check[5].action ==
              DeltaSteppingCsrControllerAction::kStopComplete &&
              after_check[5].light_rounds == 5 &&
              after_check[5].iterations == 1 &&
              after_check[5].rounds_since_host_check == 1 &&
              after_check[5].publication_sequence == 2,
          "completion immediately after a host check was not terminal");
  for (const auto& state : after_check) {
    require_valid_controller_descriptor(
        state, "batched light-round descriptor is invalid");
  }

  // Completion one round before the scheduled check is published immediately
  // instead of padding the batch with an unnecessary dispatch.
  DeltaSteppingCsrControllerSequentialModel before_check_model(reduced);
  const auto before_check = before_check_model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0),
  });
  require(before_check.back().action ==
              DeltaSteppingCsrControllerAction::kStopComplete &&
              before_check.back().light_rounds == 3 &&
              before_check.back().rounds_since_host_check == 3 &&
              before_check.back().publication_sequence == 1,
          "completion immediately before a host check was delayed");

  // Completion on the round where a check would otherwise be published takes
  // precedence over a nonterminal host-check action.
  DeltaSteppingCsrControllerSequentialModel at_check_model(reduced);
  const auto at_check = at_check_model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0),
  });
  require(at_check.back().action ==
              DeltaSteppingCsrControllerAction::kStopComplete &&
              at_check.back().light_rounds == 4 &&
              at_check.back().iterations == 1 &&
              at_check.back().rounds_since_host_check == 4 &&
              at_check.back().publication_sequence == 1,
          "completion at a host-check boundary was delayed or overwritten");
}

void test_controller_multiple_buckets() {
  DeltaSteppingCsrControllerSequentialModel model(
      {DeltaSteppingCsrControllerMode::kReducedRoundTrip, 4});
  const auto states = model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0, 3, 3),
      DeltaSteppingCsrControllerTraceStep::begin_bucket(3, 2, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(3, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(3, 0, 1, 7),
      DeltaSteppingCsrControllerTraceStep::begin_bucket(7, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(7, 0),
  });
  require(states[1].phase ==
              DeltaSteppingCsrControllerPhase::kBucketBoundary &&
              states[1].action ==
                  DeltaSteppingCsrControllerAction::kAdvanceBucket &&
              states[2].current_bucket == 3 &&
              states[2].phase ==
                  DeltaSteppingCsrControllerPhase::kLightClosure,
          "first bucket advancement was not modeled correctly");
  require(states[4].action ==
              DeltaSteppingCsrControllerAction::kAdvanceBucket &&
              states[5].current_bucket == 7 &&
              states.back().action ==
                  DeltaSteppingCsrControllerAction::kStopComplete &&
              states.back().light_rounds == 4 &&
              states.back().iterations == 3 &&
              states.back().publication_sequence == 1,
          "multi-bucket trace did not finish through device advancement");
  for (const auto& state : states) {
    require_valid_controller_descriptor(
        state, "multi-bucket controller descriptor is invalid");
  }
}

void test_controller_terminal_statuses() {
  const DeltaSteppingCsrControllerPolicy reduced{
      DeltaSteppingCsrControllerMode::kReducedRoundTrip, 4};

  DeltaSteppingCsrControllerSequentialModel target_model(reduced);
  target_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  target_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1));
  const auto target = target_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 1, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kTargetSettled));
  require(target.action ==
              DeltaSteppingCsrControllerAction::kStopTargetSettled &&
              target.phase == DeltaSteppingCsrControllerPhase::kFinished &&
              target.current_count == 0 && target.pending_count == 1 &&
              target.light_rounds == 2 && target.iterations == 1 &&
              target.publication_sequence == 1,
          "target settlement inside a batch did not stop immediately");
  require_valid_controller_descriptor(
      target, "target-settled boundary descriptor is invalid");

  DeltaSteppingCsrControllerSequentialModel mid_light_target_model(reduced);
  mid_light_target_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto mid_light_target = mid_light_target_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 1, 0, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kTargetSettled));
  require(mid_light_target.action ==
              DeltaSteppingCsrControllerAction::kStopInvalidState,
          "target settlement before light closure was accepted");

  DeltaSteppingCsrControllerSequentialModel iteration_model(reduced);
  iteration_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  iteration_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0, 1, 4));
  const auto iteration = iteration_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_bucket(
          4, 1, 0, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kIterationLimit));
  require(iteration.action ==
              DeltaSteppingCsrControllerAction::kStopIterationLimit &&
              iteration.light_rounds == 1 && iteration.iterations == 1 &&
              iteration.publication_sequence == 1,
          "iteration limit inside a batch did not stop immediately");

  DeltaSteppingCsrControllerSequentialModel overflow_model(reduced);
  overflow_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto overflow = overflow_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 0, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kComplete |
              DeltaSteppingCsrControllerStatus::kTargetSettled |
              DeltaSteppingCsrControllerStatus::kQueueOverflow));
  require(overflow.action ==
              DeltaSteppingCsrControllerAction::kStopQueueOverflow &&
              delta_stepping_controller_has_status(
                  overflow.status,
                  DeltaSteppingCsrControllerStatus::kComplete) &&
              delta_stepping_controller_has_status(
                  overflow.status,
                  DeltaSteppingCsrControllerStatus::kTargetSettled) &&
              delta_stepping_controller_has_status(
                  overflow.status,
                  DeltaSteppingCsrControllerStatus::kQueueOverflow),
          "queue overflow was lost at a combined terminal boundary");

  DeltaSteppingCsrControllerSequentialModel overflow_with_pending_model(
      reduced);
  overflow_with_pending_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto overflow_with_pending = overflow_with_pending_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 1, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kQueueOverflow));
  require(overflow_with_pending.action ==
              DeltaSteppingCsrControllerAction::kStopQueueOverflow &&
              !delta_stepping_controller_has_status(
                  overflow_with_pending.status,
                  DeltaSteppingCsrControllerStatus::kInvalidState),
          "pre-reduction overflow with pending work became invalid state");

  DeltaSteppingCsrControllerSequentialModel stale_completion_model(reduced);
  stale_completion_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto stale_completion = stale_completion_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 1, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kComplete));
  require(stale_completion.action ==
              DeltaSteppingCsrControllerAction::kStopComplete &&
              stale_completion.pending_count == 1,
          "stale physical pending token blocked valid completion");

  DeltaSteppingCsrControllerSequentialModel lost_pending_model(reduced);
  lost_pending_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  lost_pending_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1));
  const auto lost_pending = lost_pending_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 0));
  require(lost_pending.action ==
              DeltaSteppingCsrControllerAction::kStopInvalidState,
          "pending work disappeared before compaction without invalid state");

  // A later illegal transition adds invalid-state status without erasing the
  // already-published overflow bit; invalid state then becomes dominant.
  const auto invalid_after_overflow = overflow_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0));
  require(invalid_after_overflow.action ==
              DeltaSteppingCsrControllerAction::kStopInvalidState &&
              delta_stepping_controller_has_status(
                  invalid_after_overflow.status,
                  DeltaSteppingCsrControllerStatus::kQueueOverflow) &&
              delta_stepping_controller_has_status(
                  invalid_after_overflow.status,
                  DeltaSteppingCsrControllerStatus::kInvalidState),
          "invalid-state precedence erased sticky queue overflow");

  DeltaSteppingCsrControllerSequentialModel malformed_model(reduced);
  malformed_model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto malformed = malformed_model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(
          9, 1, 0, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kQueueOverflow));
  require(malformed.action ==
              DeltaSteppingCsrControllerAction::kStopInvalidState &&
              delta_stepping_controller_has_status(
                  malformed.status,
                  DeltaSteppingCsrControllerStatus::kQueueOverflow) &&
              delta_stepping_controller_has_status(
                  malformed.status,
                  DeltaSteppingCsrControllerStatus::kInvalidState),
          "malformed trace did not propagate invalid state over overflow");
  require_valid_controller_descriptor(
      malformed, "terminal error controller descriptor is invalid");
}

void test_controller_callback_abort_and_reuse() {
  DeltaSteppingCsrControllerSequentialModel model(
      {DeltaSteppingCsrControllerMode::kHostChecked, 9});
  require(model.effective_batch_size() == 1,
          "host-checked controller did not force one-round boundaries");
  model.apply(DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto boundary = model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0, 1, 5));
  require(boundary.action ==
              DeltaSteppingCsrControllerAction::kPublishHostCheck &&
              boundary.phase ==
                  DeltaSteppingCsrControllerPhase::kBucketBoundary &&
              boundary.publication_sequence == 1,
          "host-checked controller did not publish its callback bucket");
  const auto aborted = model.apply(
      DeltaSteppingCsrControllerTraceStep::callback_abort());
  require(aborted.action ==
              DeltaSteppingCsrControllerAction::kStopCallbackAbort &&
              aborted.publication_sequence == 2,
          "callback-like abort did not publish terminal state");

  const auto reused = model.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(2, 5));
  require(reused.query_sequence == 2 && reused.publication_sequence == 0 &&
              reused.status == DeltaSteppingCsrControllerStatus::kNone &&
              reused.phase ==
                  DeltaSteppingCsrControllerPhase::kLightClosure &&
              reused.current_bucket == 5,
          "controller state was not reset for workspace reuse");
  const auto reused_done = model.apply(
      DeltaSteppingCsrControllerTraceStep::light_round(5, 0));
  require(reused_done.query_sequence == 2 &&
              reused_done.action ==
                  DeltaSteppingCsrControllerAction::kStopComplete,
          "reused controller query did not complete successfully");

  DeltaSteppingCsrControllerSequentialModel invalid_callback(
      {DeltaSteppingCsrControllerMode::kReducedRoundTrip, 4});
  invalid_callback.apply(
      DeltaSteppingCsrControllerTraceStep::begin_query(1));
  const auto invalid = invalid_callback.apply(
      DeltaSteppingCsrControllerTraceStep::callback_abort());
  require(invalid.action ==
              DeltaSteppingCsrControllerAction::kStopInvalidState,
          "callback abort outside a host boundary was accepted");
}

void test_controller_batch_one_equivalence() {
  const std::vector<DeltaSteppingCsrControllerTraceStep> trace = {
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1, 4),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 0, 1, 4),
      DeltaSteppingCsrControllerTraceStep::begin_bucket(4, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(4, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(4, 0),
  };
  DeltaSteppingCsrControllerSequentialModel host_checked(
      {DeltaSteppingCsrControllerMode::kHostChecked, 32});
  DeltaSteppingCsrControllerSequentialModel reduced_one(
      {DeltaSteppingCsrControllerMode::kReducedRoundTrip, 1});
  const auto host_states = host_checked.run(trace);
  const auto reduced_states = reduced_one.run(trace);
  require(host_states.size() == reduced_states.size(),
          "batch-one comparison produced different trace lengths");
  for (std::size_t i = 0; i < host_states.size(); ++i) {
    require(same_controller_semantics(host_states[i], reduced_states[i]),
            "reduced batch size one diverged from host controller semantics");
  }
  require(host_states[3].action ==
              DeltaSteppingCsrControllerAction::kContinueDevice &&
              reduced_states[3].action ==
                  DeltaSteppingCsrControllerAction::kPublishHostCheck,
          "batch-one model did not publish after atomic reduced compaction");
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

void test_unit_weight_proof() {
  const std::vector<float> empty;
  const DeltaSteppingCsrUnitWeightProof empty_proof =
      delta_stepping_make_unit_weight_proof(empty);
  require(empty_proof.verified &&
              delta_stepping_unit_weight_proof_matches(empty_proof, empty),
          "empty exact-unit payload did not retain its vacuous proof");

  const std::vector<float> unit(17, 1.0f);
  const DeltaSteppingCsrUnitWeightProof proof =
      delta_stepping_make_unit_weight_proof(unit);
  require(proof.verified && proof.value_count == unit.size() &&
              delta_stepping_unit_weight_proof_matches(proof, unit),
          "exact-unit proof did not match its uploaded payload");

  for (const float rejected : {
           0.0f,
           -0.0f,
           std::nextafter(1.0f, 0.0f),
           std::nextafter(1.0f, 2.0f),
           std::numeric_limits<float>::infinity(),
           -std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    std::vector<float> values = unit;
    values[8] = rejected;
    require(!delta_stepping_make_unit_weight_proof(values).verified &&
                !delta_stepping_unit_weight_proof_matches(proof, values),
            "non-unit or non-finite payload passed exact-unit proof checks");
  }

  std::vector<float> stale = unit;
  stale.push_back(1.0f);
  require(!delta_stepping_unit_weight_proof_matches(proof, stale),
          "stale exact-unit proof survived a payload-size change");
  DeltaSteppingCsrUnitWeightProof forged = proof;
  forged.fingerprint =
      delta_stepping_edge_value_fingerprint(std::vector<float>(17, 2.0f));
  require(!delta_stepping_unit_weight_proof_matches(forged, unit),
          "incorrect exact-unit fingerprint was accepted");
}

void test_host_count_and_wave_eligibility() {
  require(delta_stepping_host_value_frontier_count_eligible(
              DeltaSteppingCsrControllerMode::kHostChecked, true, true, 1),
          "authoritative explicit-stream host count was not eligible");
  require(!delta_stepping_host_value_frontier_count_eligible(
              DeltaSteppingCsrControllerMode::kReducedRoundTrip, true, true,
              1) &&
              !delta_stepping_host_value_frontier_count_eligible(
                  DeltaSteppingCsrControllerMode::kHostChecked, false, true,
                  1) &&
              !delta_stepping_host_value_frontier_count_eligible(
                  DeltaSteppingCsrControllerMode::kHostChecked, true, false,
                  1) &&
              !delta_stepping_host_value_frontier_count_eligible(
                  DeltaSteppingCsrControllerMode::kHostChecked, true, true,
                  4),
          "host frontier-count policy admitted a non-authoritative path");

  auto eligible = [](bool requested,
                     int wavefront,
                     bool compact_rows,
                     DeltaSteppingCsrCurrentMembershipMode membership,
                     bool parents,
                     bool edge_parents,
                     bool vertex_costs,
                     bool heavy,
                     bool all_light,
                     bool telemetry,
                     DeltaSteppingCsrControllerMode controller,
                     bool unit_proof) {
    return delta_stepping_wave32_relaxation_eligible(
        requested, wavefront, compact_rows, membership, parents,
        edge_parents, vertex_costs, heavy, all_light, telemetry, controller,
        unit_proof);
  };
  require(eligible(true, 32, true,
                   DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                   true, false, false, true, false,
                   DeltaSteppingCsrControllerMode::kHostChecked, true),
          "exact profiled wave32 specialization was not eligible");
  require(!eligible(false, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 64, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, false,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kGeneration, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, true, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, true, false, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, true,
                    DeltaSteppingCsrControllerMode::kHostChecked, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kReducedRoundTrip, true) &&
              !eligible(true, 32, true,
                    DeltaSteppingCsrCurrentMembershipMode::kBoolean, true,
                    true, false, false, true, false,
                    DeltaSteppingCsrControllerMode::kHostChecked, false),
          "wave32 policy admitted an unprofiled specialization");
}

void test_ballot_prefix_and_queue_reservations() {
  for (const unsigned int width : {32U, 64U}) {
    const std::uint64_t all = delta_stepping_wave_width_mask(width);
    require(delta_stepping_ballot_prefix_position(0, 0, width, 7) ==
                kDeltaSteppingCsrNoQueuePosition &&
                delta_stepping_ballot_prefix_position(all, 0, width, 7) ==
                    7 &&
                delta_stepping_ballot_prefix_position(
                    all, width - 1, width, 7) == 7 + width - 1,
            "zero/all ballot prefix failed at a wave boundary");
    const std::uint64_t sparse =
        UINT64_C(1) | (UINT64_C(1) << 3) |
        (UINT64_C(1) << (width - 1));
    require(delta_stepping_ballot_prefix_position(sparse, 3, width, 11) ==
                12 &&
                delta_stepping_ballot_prefix_position(
                    sparse, width - 1, width, 11) == 13,
            "sparse/highest-lane ballot prefix failed");
  }
  require_throws<std::invalid_argument>(
      [] {
        (void)delta_stepping_ballot_prefix_position(
            UINT64_C(1) << 32, 0, 32, 0);
      },
      "wave32 ballot accepted a bit above lane 31");
  require_throws<std::overflow_error>(
      [] {
        (void)delta_stepping_ballot_prefix_position(
            UINT64_C(3), 1, 32,
            std::numeric_limits<std::size_t>::max());
      },
      "ballot prefix arithmetic overflow was accepted");

  std::vector<DeltaSteppingCsrWaveQueueClaim> claims(32);
  claims[0] = {true, true, false, true};
  claims[3] = {true, true, true, false};
  claims[17] = {false, true, true, true};
  claims[31] = {true, false, true, true};
  const DeltaSteppingCsrWaveQueueReservationModel model =
      delta_stepping_model_wave_queue_reservations(
          claims, 32, 5, 7, 11, 64);
  require(model.touched_tail == 7 && model.current_tail == 9 &&
              model.pending_tail == 13 &&
              model.positions[0].touched == 5 &&
              model.positions[3].touched == 6 &&
              model.positions[3].current == 7 &&
              model.positions[31].current == 8 &&
              model.positions[0].pending == 11 &&
              model.positions[31].pending == 12 &&
              model.positions[17].touched ==
                  kDeltaSteppingCsrNoQueuePosition,
          "independent touched/current/pending reservations were incorrect");
  require(delta_stepping_checked_queue_reservation(31, 1, 32) == 32,
          "exact-capacity queue reservation was rejected");
  require_throws<std::overflow_error>(
      [&] {
        (void)delta_stepping_model_wave_queue_reservations(
            claims, 32, 31, 0, 0, 32);
      },
      "wave reservation exceeded queue capacity");
  require_throws<std::overflow_error>(
      [] { (void)delta_stepping_checked_queue_reservation(33, 0, 32); },
      "queue reservation accepted a base beyond capacity");

  DeltaSteppingCsrCompactParentCandidateModel parent;
  for (const std::uint32_t edge : {17U, 9U, 23U, 3U, 11U}) {
    parent = delta_stepping_select_compact_parent_candidate(
        parent, {4.0f, edge});
  }
  require(parent.distance == 4.0f && parent.edge == 3,
          "distance tie did not select the deterministic lowest edge id");
  parent = delta_stepping_select_compact_parent_candidate(
      parent, {3.0f, 31U});
  require(parent.distance == 3.0f && parent.edge == 31,
          "strict distance decrease did not replace the parent tie winner");
  require_throws<std::invalid_argument>(
      [&] {
        (void)delta_stepping_select_compact_parent_candidate(
            parent,
            {std::numeric_limits<float>::quiet_NaN(), 0});
      },
      "non-finite parent candidate was accepted");
}

std::vector<int> sequential_unit_distances(
    const std::vector<std::vector<int>>& adjacency,
    const std::vector<int>& sources) {
  const int infinity = std::numeric_limits<int>::max();
  std::vector<int> distance(adjacency.size(), infinity);
  std::vector<int> queue;
  for (const int source : sources) {
    if (distance[static_cast<std::size_t>(source)] == 0) continue;
    distance[static_cast<std::size_t>(source)] = 0;
    queue.push_back(source);
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    for (const int v : adjacency[static_cast<std::size_t>(u)]) {
      if (distance[static_cast<std::size_t>(v)] != infinity) continue;
      distance[static_cast<std::size_t>(v)] =
          distance[static_cast<std::size_t>(u)] + 1;
      queue.push_back(v);
    }
  }
  return distance;
}

std::vector<int> wave_scheduled_classic_delta_unit_distances(
    const std::vector<std::vector<int>>& adjacency,
    const std::vector<int>& sources,
    int delta,
    unsigned int wave_width) {
  const int infinity = std::numeric_limits<int>::max();
  const std::size_t capacity = adjacency.size();
  std::vector<int> distance(capacity, infinity);
  std::vector<bool> in_current(capacity, false);
  std::vector<bool> in_pending(capacity, false);
  std::vector<int> touched;
  std::vector<int> current;
  std::vector<int> pending;
  for (const int source : sources) {
    const std::size_t index = static_cast<std::size_t>(source);
    if (distance[index] == 0) continue;
    distance[index] = 0;
    in_current[index] = true;
    current.push_back(source);
    touched.push_back(source);
  }

  int current_bucket = 0;
  while (!current.empty() || !pending.empty()) {
    while (!current.empty()) {
      for (const int u : current) {
        in_current[static_cast<std::size_t>(u)] = false;
      }
      std::vector<int> next;
      std::vector<DeltaSteppingCsrWaveQueueClaim> claims;
      std::vector<int> claim_vertices;
      auto flush = [&] {
        if (claims.empty()) return;
        const auto reservations = delta_stepping_model_wave_queue_reservations(
            claims, wave_width, touched.size(), next.size(), pending.size(),
            capacity);
        touched.resize(reservations.touched_tail);
        next.resize(reservations.current_tail);
        pending.resize(reservations.pending_tail);
        for (std::size_t lane = 0; lane < claims.size(); ++lane) {
          const auto& positions = reservations.positions[lane];
          const int vertex = claim_vertices[lane];
          if (positions.touched != kDeltaSteppingCsrNoQueuePosition) {
            touched[positions.touched] = vertex;
          }
          if (positions.current != kDeltaSteppingCsrNoQueuePosition) {
            next[positions.current] = vertex;
          }
          if (positions.pending != kDeltaSteppingCsrNoQueuePosition) {
            pending[positions.pending] = vertex;
          }
        }
        claims.clear();
        claim_vertices.clear();
      };

      for (const int u : current) {
        for (const int v : adjacency[static_cast<std::size_t>(u)]) {
          const int candidate =
              distance[static_cast<std::size_t>(u)] + 1;
          const std::size_t v_index = static_cast<std::size_t>(v);
          const int old = distance[v_index];
          bool append_touched = false;
          bool append_current = false;
          bool append_pending = false;
          if (candidate < old) {
            distance[v_index] = candidate;
            append_touched = old == infinity;
            const int bucket = candidate / delta;
            if (bucket == current_bucket && !in_current[v_index]) {
              in_current[v_index] = true;
              append_current = true;
            } else if (bucket > current_bucket && !in_pending[v_index]) {
              in_pending[v_index] = true;
              append_pending = true;
            }
          }
          claims.push_back(
              {true, append_touched, append_current, append_pending});
          claim_vertices.push_back(v);
          if (claims.size() == wave_width) flush();
        }
      }
      flush();
      current = std::move(next);
    }

    int next_bucket = std::numeric_limits<int>::max();
    std::vector<int> future_pending;
    for (const int vertex : pending) {
      const std::size_t index = static_cast<std::size_t>(vertex);
      const int live_bucket = distance[index] / delta;
      if (live_bucket <= current_bucket) {
        // Match the production pending reduction: a token whose live distance
        // was pulled into the completed light closure is stale and is cleared.
        in_pending[index] = false;
        continue;
      }
      next_bucket = std::min(next_bucket, live_bucket);
      future_pending.push_back(vertex);
    }
    pending = std::move(future_pending);
    if (next_bucket == std::numeric_limits<int>::max()) break;
    current_bucket = next_bucket;
    std::vector<int> remaining;
    for (const int vertex : pending) {
      const std::size_t index = static_cast<std::size_t>(vertex);
      if (distance[index] / delta == current_bucket) {
        in_pending[index] = false;
        if (!in_current[index]) {
          in_current[index] = true;
          current.push_back(vertex);
        }
      } else {
        remaining.push_back(vertex);
      }
    }
    pending = std::move(remaining);
  }
  return distance;
}

void test_randomized_wave_scheduling_against_sequential() {
  // Deterministically force zero-degree rows, mixed row lengths, duplicate
  // edges, and more than one converged reservation batch before fuzzing.
  std::vector<std::vector<int>> batched_adjacency(96);
  std::vector<int> batched_sources;
  for (int u = 0; u < 48; ++u) {
    batched_sources.push_back(u);
    const int row_length = u % 7;
    for (int offset = 0; offset < row_length; ++offset) {
      const int v = 48 + (u * 3 + offset) % 40;
      batched_adjacency[static_cast<std::size_t>(u)].push_back(v);
      if (offset == 0) {
        batched_adjacency[static_cast<std::size_t>(u)].push_back(v);
      }
    }
  }
  batched_sources.push_back(0);
  const std::vector<int> batched_reference =
      sequential_unit_distances(batched_adjacency, batched_sources);
  require(wave_scheduled_classic_delta_unit_distances(
              batched_adjacency, batched_sources, 4, 32) ==
              batched_reference &&
              wave_scheduled_classic_delta_unit_distances(
                  batched_adjacency, batched_sources, 4, 64) ==
                  batched_reference,
          "multiple mixed-row wave batches disagreed with sequential SSSP");

  std::mt19937 rng(0x5eed1234U);
  for (int trial = 0; trial < 200; ++trial) {
    const int rows = 2 + static_cast<int>(rng() % 63U);
    std::vector<std::vector<int>> adjacency(
        static_cast<std::size_t>(rows));
    const int edges = static_cast<int>(rng() % (rows * 5 + 1));
    for (int edge = 0; edge < edges; ++edge) {
      const int u = static_cast<int>(rng() % static_cast<unsigned int>(rows));
      const int v = static_cast<int>(rng() % static_cast<unsigned int>(rows));
      adjacency[static_cast<std::size_t>(u)].push_back(v);
      if ((rng() & 7U) == 0) {
        adjacency[static_cast<std::size_t>(u)].push_back(v);
      }
    }
    std::vector<int> sources;
    const int source_count = 1 + static_cast<int>(rng() % 5U);
    for (int i = 0; i < source_count; ++i) {
      sources.push_back(
          static_cast<int>(rng() % static_cast<unsigned int>(rows)));
    }
    const std::vector<int> reference =
        sequential_unit_distances(adjacency, sources);
    for (const unsigned int wave_width : {32U, 64U}) {
      for (const int delta : {1, 2, 4}) {
        require(wave_scheduled_classic_delta_unit_distances(
                    adjacency, sources, delta, wave_width) == reference,
                "wave-scheduled classic Delta disagreed with sequential unit SSSP");
      }
    }
  }
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
  test_controller_policy_and_descriptor();
  test_controller_zero_and_light_rounds();
  test_controller_multiple_buckets();
  test_controller_terminal_statuses();
  test_controller_callback_abort_and_reuse();
  test_controller_batch_one_equivalence();
  test_row_offset_policy();
  test_result_shape_policy();
  test_generation_membership_model();
  test_unit_weight_proof();
  test_host_count_and_wave_eligibility();
  test_ballot_prefix_and_queue_reservations();
  test_randomized_wave_scheduling_against_sequential();
  test_allocation_policy();
  std::cout << "Delta-Stepping policy test passed\n";
  return 0;
}
