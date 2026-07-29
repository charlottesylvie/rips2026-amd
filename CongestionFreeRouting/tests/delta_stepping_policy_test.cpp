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

  require(static_cast<std::uint32_t>(
              DeltaSteppingCsrControllerMode::kFusedHostChecked) == 2U,
          "fused host-checked controller mode lost its stable value");
  const DeltaSteppingCsrControllerPolicy host_checked{
      DeltaSteppingCsrControllerMode::kHostChecked,
      kDeltaSteppingCsrMaxControllerBatchSize};
  const DeltaSteppingCsrControllerPolicy fused_host_checked{
      DeltaSteppingCsrControllerMode::kFusedHostChecked,
      kDeltaSteppingCsrMaxControllerBatchSize};
  require(delta_stepping_effective_controller_batch_size(host_checked) == 1 &&
              delta_stepping_effective_controller_batch_size(
                  fused_host_checked) == 1,
          "host and fused-host controllers must remain one-action modes");

  const DeltaSteppingCsrControllerPolicy reduced{
      DeltaSteppingCsrControllerMode::kReducedRoundTrip, 7};
  require(delta_stepping_effective_controller_batch_size(reduced) == 7,
          "reduced controller did not retain its bounded batch size");
  const DeltaSteppingCsrControllerPolicy maximum_reduced{
      DeltaSteppingCsrControllerMode::kReducedRoundTrip,
      kDeltaSteppingCsrMaxControllerBatchSize};
  require(delta_stepping_effective_controller_batch_size(maximum_reduced) ==
              kDeltaSteppingCsrMaxControllerBatchSize,
          "maximum bounded reduced-controller batch was rejected");
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
             kDeltaSteppingCsrMaxControllerBatchSize + 1U});
      },
      "reduced controller accepted a batch beyond its watchdog cap");
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
  DeltaSteppingCsrControllerSequentialModel fused_host_checked(
      {DeltaSteppingCsrControllerMode::kFusedHostChecked, 32});
  DeltaSteppingCsrControllerSequentialModel reduced_one(
      {DeltaSteppingCsrControllerMode::kReducedRoundTrip, 1});
  const auto host_states = host_checked.run(trace);
  const auto fused_states = fused_host_checked.run(trace);
  const auto reduced_states = reduced_one.run(trace);
  require(host_checked.effective_batch_size() == 1 &&
              fused_host_checked.effective_batch_size() == 1 &&
              reduced_one.effective_batch_size() == 1,
          "a batch-one controller exposed a wider action budget");
  require(host_states.size() == fused_states.size() &&
              host_states.size() == reduced_states.size(),
          "batch-one comparison produced different trace lengths");
  for (std::size_t i = 0; i < host_states.size(); ++i) {
    require(same_controller_semantics(host_states[i], fused_states[i]),
            "fused host-checked controller diverged from scalar semantics");
    require(same_controller_semantics(host_states[i], reduced_states[i]),
            "reduced batch size one diverged from host controller semantics");
  }
  require(host_states[3].action ==
              DeltaSteppingCsrControllerAction::kContinueDevice &&
              fused_states[3].action ==
                  DeltaSteppingCsrControllerAction::kPublishHostCheck &&
              reduced_states[3].action ==
                  DeltaSteppingCsrControllerAction::kPublishHostCheck,
          "batch-one mode-specific compaction boundary changed unexpectedly");
}

void test_controller_publication_accounting() {
  constexpr std::uint64_t kBatchSize = 4;
  DeltaSteppingCsrControllerSequentialModel model(
      {DeltaSteppingCsrControllerMode::kReducedRoundTrip,
       static_cast<std::uint32_t>(kBatchSize)});
  const auto states = model.run({
      DeltaSteppingCsrControllerTraceStep::begin_query(1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(0, 1, 1),
      DeltaSteppingCsrControllerTraceStep::light_round(
          0, 0, 1, kDeltaSteppingCsrNoControllerBucket,
          DeltaSteppingCsrControllerStatus::kComplete),
  });

  std::uint64_t publications = 0;
  std::uint64_t nonterminal_publications = 0;
  std::uint64_t terminal_publications = 0;
  std::uint64_t action_slots_budgeted = 0;
  std::uint64_t actions_completed = 0;
  std::uint64_t unused_action_slots = 0;
  std::uint32_t previous_publication = 0;
  std::uint32_t previous_light_rounds = 0;
  for (const auto& state : states) {
    if (state.publication_sequence == previous_publication) continue;
    require(state.publication_sequence == previous_publication + 1U,
            "controller publication sequence skipped a descriptor");
    const std::uint64_t completed =
        static_cast<std::uint64_t>(state.light_rounds) -
        static_cast<std::uint64_t>(previous_light_rounds);
    require(completed > 0 && completed <= kBatchSize,
            "published action count exceeded its bounded batch");
    if (state.status == DeltaSteppingCsrControllerStatus::kNone) {
      require(completed == kBatchSize &&
                  state.action ==
                      DeltaSteppingCsrControllerAction::kPublishHostCheck,
              "nonterminal publication did not consume its full batch");
      ++nonterminal_publications;
    } else {
      ++terminal_publications;
    }
    ++publications;
    action_slots_budgeted += kBatchSize;
    actions_completed += completed;
    unused_action_slots += kBatchSize - completed;
    previous_publication = state.publication_sequence;
    previous_light_rounds = state.light_rounds;
  }

  require(publications == 2 && nonterminal_publications == 1 &&
              terminal_publications == 1 &&
              action_slots_budgeted == 8 && actions_completed == 5 &&
              unused_action_slots == 3,
          "controller publication/action diagnostics accounting diverged");
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
  test_controller_policy_and_descriptor();
  test_controller_zero_and_light_rounds();
  test_controller_multiple_buckets();
  test_controller_terminal_statuses();
  test_controller_callback_abort_and_reuse();
  test_controller_batch_one_equivalence();
  test_controller_publication_accounting();
  test_row_offset_policy();
  test_result_shape_policy();
  test_generation_membership_model();
  test_allocation_policy();
  std::cout << "Delta-Stepping policy test passed\n";
  return 0;
}
