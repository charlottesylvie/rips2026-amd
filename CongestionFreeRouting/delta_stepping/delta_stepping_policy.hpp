#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
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

// The host-checked controller preserves the established synchronization
// behavior.  The reduced-round-trip controller is an explicit opt-in until it
// has passed repeated target-GPU validation.
enum class DeltaSteppingCsrControllerMode : std::uint32_t {
  kHostChecked = 0,
  kReducedRoundTrip = 1,
};

constexpr std::uint32_t
    kDeltaSteppingCsrRecommendedControllerBatchSize = 4;

struct DeltaSteppingCsrControllerPolicy {
  DeltaSteppingCsrControllerMode mode =
      DeltaSteppingCsrControllerMode::kHostChecked;
  std::uint32_t batch_size =
      kDeltaSteppingCsrRecommendedControllerBatchSize;
};

inline void delta_stepping_validate_controller_policy(
    const DeltaSteppingCsrControllerPolicy& policy) {
  switch (policy.mode) {
    case DeltaSteppingCsrControllerMode::kHostChecked:
    case DeltaSteppingCsrControllerMode::kReducedRoundTrip:
      break;
    default:
      throw std::invalid_argument(
          "Delta-Stepping controller mode is invalid");
  }
  if (policy.batch_size == 0) {
    throw std::invalid_argument(
        "Delta-Stepping controller batch size must be positive");
  }
  if (policy.mode == DeltaSteppingCsrControllerMode::kReducedRoundTrip &&
      std::uint64_t{2} * policy.batch_size >=
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "Delta-Stepping controller batch size exceeds its bounded token "
        "range");
  }
}

inline std::uint32_t delta_stepping_effective_controller_batch_size(
    const DeltaSteppingCsrControllerPolicy& policy) {
  delta_stepping_validate_controller_policy(policy);
  return policy.mode == DeltaSteppingCsrControllerMode::kHostChecked
             ? std::uint32_t{1}
             : policy.batch_size;
}

// These types deliberately use fixed-width representations: a device may
// publish one descriptor and the host can inspect it without reconstructing
// several independently copied scalar values. Status is a sticky bit set so a
// later completion or early-stop signal cannot erase overflow or invalid
// state observed at the same control boundary.
enum class DeltaSteppingCsrControllerStatus : std::uint32_t {
  kNone = 0,
  kComplete = std::uint32_t{1} << 0,
  kTargetSettled = std::uint32_t{1} << 1,
  kIterationLimit = std::uint32_t{1} << 2,
  kCallbackAbort = std::uint32_t{1} << 3,
  kQueueOverflow = std::uint32_t{1} << 4,
  kInvalidState = std::uint32_t{1} << 5,
};

constexpr DeltaSteppingCsrControllerStatus operator|(
    DeltaSteppingCsrControllerStatus lhs,
    DeltaSteppingCsrControllerStatus rhs) noexcept {
  return static_cast<DeltaSteppingCsrControllerStatus>(
      static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

constexpr DeltaSteppingCsrControllerStatus operator&(
    DeltaSteppingCsrControllerStatus lhs,
    DeltaSteppingCsrControllerStatus rhs) noexcept {
  return static_cast<DeltaSteppingCsrControllerStatus>(
      static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

inline DeltaSteppingCsrControllerStatus& operator|=(
    DeltaSteppingCsrControllerStatus& lhs,
    DeltaSteppingCsrControllerStatus rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

constexpr std::uint32_t kDeltaSteppingCsrControllerKnownStatusBits =
    static_cast<std::uint32_t>(DeltaSteppingCsrControllerStatus::kComplete) |
    static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kTargetSettled) |
    static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kIterationLimit) |
    static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kCallbackAbort) |
    static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kQueueOverflow) |
    static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kInvalidState);

constexpr bool delta_stepping_controller_has_status(
    DeltaSteppingCsrControllerStatus statuses,
    DeltaSteppingCsrControllerStatus status) noexcept {
  return (static_cast<std::uint32_t>(statuses) &
          static_cast<std::uint32_t>(status)) != 0;
}

constexpr DeltaSteppingCsrControllerStatus
delta_stepping_normalize_controller_status(
    DeltaSteppingCsrControllerStatus statuses) noexcept {
  const std::uint32_t raw = static_cast<std::uint32_t>(statuses);
  std::uint32_t normalized =
      raw & kDeltaSteppingCsrControllerKnownStatusBits;
  if ((raw & ~kDeltaSteppingCsrControllerKnownStatusBits) != 0) {
    normalized |= static_cast<std::uint32_t>(
        DeltaSteppingCsrControllerStatus::kInvalidState);
  }
  return static_cast<DeltaSteppingCsrControllerStatus>(normalized);
}

enum class DeltaSteppingCsrControllerPhase : std::uint32_t {
  kIdle = 0,
  kLightClosure = 1,
  kBucketBoundary = 2,
  kFinished = 3,
};

enum class DeltaSteppingCsrControllerAction : std::uint32_t {
  kContinueDevice = 0,
  kPublishHostCheck = 1,
  kAdvanceBucket = 2,
  kStopComplete = 3,
  kStopTargetSettled = 4,
  kStopIterationLimit = 5,
  kStopCallbackAbort = 6,
  kStopQueueOverflow = 7,
  kStopInvalidState = 8,
};

constexpr DeltaSteppingCsrControllerAction
delta_stepping_controller_terminal_action(
    DeltaSteppingCsrControllerStatus statuses) noexcept {
  const DeltaSteppingCsrControllerStatus normalized =
      delta_stepping_normalize_controller_status(statuses);
  // Invalid state dominates overflow, which dominates every ordinary stop.
  // The underlying status flags remain accumulated even when one action wins.
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kInvalidState)) {
    return DeltaSteppingCsrControllerAction::kStopInvalidState;
  }
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kQueueOverflow)) {
    return DeltaSteppingCsrControllerAction::kStopQueueOverflow;
  }
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kCallbackAbort)) {
    return DeltaSteppingCsrControllerAction::kStopCallbackAbort;
  }
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kIterationLimit)) {
    return DeltaSteppingCsrControllerAction::kStopIterationLimit;
  }
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kTargetSettled)) {
    return DeltaSteppingCsrControllerAction::kStopTargetSettled;
  }
  if (delta_stepping_controller_has_status(
          normalized, DeltaSteppingCsrControllerStatus::kComplete)) {
    return DeltaSteppingCsrControllerAction::kStopComplete;
  }
  return DeltaSteppingCsrControllerAction::kContinueDevice;
}

constexpr std::uint64_t kDeltaSteppingCsrNoControllerBucket =
    std::numeric_limits<std::uint64_t>::max();

struct DeltaSteppingCsrControllerDescriptor {
  static constexpr std::uint32_t kVersion = 1;

  std::uint32_t version = kVersion;
  std::uint32_t query_sequence = 0;
  std::uint32_t publication_sequence = 0;
  DeltaSteppingCsrControllerStatus status =
      DeltaSteppingCsrControllerStatus::kNone;
  DeltaSteppingCsrControllerPhase phase =
      DeltaSteppingCsrControllerPhase::kIdle;
  DeltaSteppingCsrControllerAction action =
      DeltaSteppingCsrControllerAction::kContinueDevice;
  std::uint32_t rounds_since_host_check = 0;
  std::uint32_t light_rounds = 0;
  std::uint32_t current_count = 0;
  std::uint32_t pending_count = 0;
  std::uint64_t current_bucket = 0;
  std::uint64_t next_bucket = kDeltaSteppingCsrNoControllerBucket;
  std::uint64_t iterations = 0;
};

static_assert(std::is_standard_layout<
                  DeltaSteppingCsrControllerDescriptor>::value,
              "Delta controller descriptor must have standard layout");
static_assert(std::is_trivially_copyable<
                  DeltaSteppingCsrControllerDescriptor>::value,
              "Delta controller descriptor must be trivially copyable");
static_assert(sizeof(DeltaSteppingCsrControllerDescriptor) == 64,
              "Delta controller descriptor layout unexpectedly changed");

constexpr bool delta_stepping_controller_phase_is_valid(
    DeltaSteppingCsrControllerPhase phase) noexcept {
  switch (phase) {
    case DeltaSteppingCsrControllerPhase::kIdle:
    case DeltaSteppingCsrControllerPhase::kLightClosure:
    case DeltaSteppingCsrControllerPhase::kBucketBoundary:
    case DeltaSteppingCsrControllerPhase::kFinished:
      return true;
  }
  return false;
}

constexpr bool delta_stepping_controller_action_is_valid(
    DeltaSteppingCsrControllerAction action) noexcept {
  switch (action) {
    case DeltaSteppingCsrControllerAction::kContinueDevice:
    case DeltaSteppingCsrControllerAction::kPublishHostCheck:
    case DeltaSteppingCsrControllerAction::kAdvanceBucket:
    case DeltaSteppingCsrControllerAction::kStopComplete:
    case DeltaSteppingCsrControllerAction::kStopTargetSettled:
    case DeltaSteppingCsrControllerAction::kStopIterationLimit:
    case DeltaSteppingCsrControllerAction::kStopCallbackAbort:
    case DeltaSteppingCsrControllerAction::kStopQueueOverflow:
    case DeltaSteppingCsrControllerAction::kStopInvalidState:
      return true;
  }
  return false;
}

constexpr bool delta_stepping_controller_pending_state_is_valid(
    std::uint32_t pending_count,
    std::uint64_t current_bucket,
    std::uint64_t next_bucket) noexcept {
  if (pending_count == 0) {
    return next_bucket == kDeltaSteppingCsrNoControllerBucket;
  }
  return next_bucket != kDeltaSteppingCsrNoControllerBucket &&
         next_bucket > current_bucket;
}

// While light closure is still active the pending minimum has deliberately
// not been reduced yet. A publication may therefore report an unknown next
// bucket even with pending entries. Bucket-boundary state is stricter.
constexpr bool delta_stepping_controller_light_pending_state_is_valid(
    std::uint32_t pending_count,
    std::uint64_t current_bucket,
    std::uint64_t next_bucket) noexcept {
  return next_bucket == kDeltaSteppingCsrNoControllerBucket ||
         delta_stepping_controller_pending_state_is_valid(
             pending_count, current_bucket, next_bucket);
}

constexpr bool delta_stepping_controller_descriptor_is_valid(
    const DeltaSteppingCsrControllerDescriptor& descriptor) noexcept {
  if (descriptor.version != DeltaSteppingCsrControllerDescriptor::kVersion ||
      !delta_stepping_controller_phase_is_valid(descriptor.phase) ||
      !delta_stepping_controller_action_is_valid(descriptor.action)) {
    return false;
  }
  const std::uint32_t raw_status =
      static_cast<std::uint32_t>(descriptor.status);
  if ((raw_status & ~kDeltaSteppingCsrControllerKnownStatusBits) != 0) {
    return false;
  }
  if (descriptor.status != DeltaSteppingCsrControllerStatus::kNone) {
    return descriptor.phase == DeltaSteppingCsrControllerPhase::kFinished &&
           descriptor.action ==
               delta_stepping_controller_terminal_action(descriptor.status);
  }
  switch (descriptor.phase) {
    case DeltaSteppingCsrControllerPhase::kIdle:
      return descriptor.action ==
                 DeltaSteppingCsrControllerAction::kContinueDevice &&
             descriptor.current_count == 0 && descriptor.pending_count == 0 &&
             descriptor.next_bucket == kDeltaSteppingCsrNoControllerBucket;
    case DeltaSteppingCsrControllerPhase::kLightClosure:
      return descriptor.current_count != 0 &&
             delta_stepping_controller_light_pending_state_is_valid(
                 descriptor.pending_count,
                 descriptor.current_bucket,
                 descriptor.next_bucket) &&
             (descriptor.action ==
                  DeltaSteppingCsrControllerAction::kContinueDevice ||
              descriptor.action ==
                  DeltaSteppingCsrControllerAction::kPublishHostCheck);
    case DeltaSteppingCsrControllerPhase::kBucketBoundary:
      return descriptor.current_count == 0 &&
             descriptor.pending_count != 0 &&
             delta_stepping_controller_pending_state_is_valid(
                 descriptor.pending_count,
                 descriptor.current_bucket,
                 descriptor.next_bucket) &&
             (descriptor.action ==
                  DeltaSteppingCsrControllerAction::kAdvanceBucket ||
              descriptor.action ==
                  DeltaSteppingCsrControllerAction::kPublishHostCheck);
    case DeltaSteppingCsrControllerPhase::kFinished:
      return false;
  }
  return false;
}

enum class DeltaSteppingCsrControllerTraceEvent : std::uint32_t {
  kBeginQuery = 0,
  kLightRound = 1,
  kBeginBucket = 2,
  kCallbackAbort = 3,
};

// Counts describe state after the event. For a light-round event,
// current_count is the next same-bucket frontier. The next pending minimum may
// remain unknown while light work is active and after compaction; a closing
// light round with pending work supplies the selected successor bucket.
struct DeltaSteppingCsrControllerTraceStep {
  DeltaSteppingCsrControllerTraceEvent event =
      DeltaSteppingCsrControllerTraceEvent::kBeginQuery;
  DeltaSteppingCsrControllerStatus status =
      DeltaSteppingCsrControllerStatus::kNone;
  std::uint32_t current_count = 0;
  std::uint32_t pending_count = 0;
  std::uint64_t bucket = 0;
  std::uint64_t next_bucket = kDeltaSteppingCsrNoControllerBucket;

  static constexpr DeltaSteppingCsrControllerTraceStep begin_query(
      std::uint32_t initial_count,
      std::uint64_t initial_bucket = 0) noexcept {
    return {DeltaSteppingCsrControllerTraceEvent::kBeginQuery,
            DeltaSteppingCsrControllerStatus::kNone,
            initial_count,
            0,
            initial_bucket,
            kDeltaSteppingCsrNoControllerBucket};
  }

  static constexpr DeltaSteppingCsrControllerTraceStep light_round(
      std::uint64_t bucket,
      std::uint32_t next_current_count,
      std::uint32_t pending_count = 0,
      std::uint64_t next_bucket = kDeltaSteppingCsrNoControllerBucket,
      DeltaSteppingCsrControllerStatus status =
          DeltaSteppingCsrControllerStatus::kNone) noexcept {
    return {DeltaSteppingCsrControllerTraceEvent::kLightRound,
            status,
            next_current_count,
            pending_count,
            bucket,
            next_bucket};
  }

  static constexpr DeltaSteppingCsrControllerTraceStep begin_bucket(
      std::uint64_t bucket,
      std::uint32_t current_count,
      std::uint32_t pending_count = 0,
      std::uint64_t next_bucket = kDeltaSteppingCsrNoControllerBucket,
      DeltaSteppingCsrControllerStatus status =
          DeltaSteppingCsrControllerStatus::kNone) noexcept {
    return {DeltaSteppingCsrControllerTraceEvent::kBeginBucket,
            status,
            current_count,
            pending_count,
            bucket,
            next_bucket};
  }

  static constexpr DeltaSteppingCsrControllerTraceStep callback_abort()
      noexcept {
    return {DeltaSteppingCsrControllerTraceEvent::kCallbackAbort,
            DeltaSteppingCsrControllerStatus::kNone,
            0,
            0,
            0,
            kDeltaSteppingCsrNoControllerBucket};
  }
};

static_assert(std::is_standard_layout<
                  DeltaSteppingCsrControllerTraceStep>::value,
              "Delta controller trace step must have standard layout");
static_assert(std::is_trivially_copyable<
                  DeltaSteppingCsrControllerTraceStep>::value,
              "Delta controller trace step must be trivially copyable");
static_assert(sizeof(DeltaSteppingCsrControllerTraceStep) == 32,
              "Delta controller trace-step layout unexpectedly changed");

// A deterministic host model of bounded controller publication. It consumes
// device-observation traces one event at a time. The model is intentionally
// sequential: it establishes transition and stop precedence, not HIP memory
// ordering, which must still be validated on the target GPU.
class DeltaSteppingCsrControllerSequentialModel {
 public:
  explicit DeltaSteppingCsrControllerSequentialModel(
      DeltaSteppingCsrControllerPolicy policy = {})
      : mode_(policy.mode),
        effective_batch_size_(
            delta_stepping_effective_controller_batch_size(policy)) {}

  const DeltaSteppingCsrControllerDescriptor& apply(
      const DeltaSteppingCsrControllerTraceStep& step) {
    switch (step.event) {
      case DeltaSteppingCsrControllerTraceEvent::kBeginQuery:
        begin_query(step);
        break;
      case DeltaSteppingCsrControllerTraceEvent::kLightRound:
        resume_after_host_check();
        apply_light_round(step);
        break;
      case DeltaSteppingCsrControllerTraceEvent::kBeginBucket:
        resume_after_host_check();
        begin_bucket(step);
        break;
      case DeltaSteppingCsrControllerTraceEvent::kCallbackAbort:
        callback_abort(step);
        break;
      default:
        finish_with(DeltaSteppingCsrControllerStatus::kInvalidState);
        break;
    }
    return descriptor_;
  }

  std::vector<DeltaSteppingCsrControllerDescriptor> run(
      const std::vector<DeltaSteppingCsrControllerTraceStep>& trace) {
    std::vector<DeltaSteppingCsrControllerDescriptor> states;
    states.reserve(trace.size());
    for (const DeltaSteppingCsrControllerTraceStep& step : trace) {
      states.push_back(apply(step));
    }
    return states;
  }

  const DeltaSteppingCsrControllerDescriptor& descriptor() const noexcept {
    return descriptor_;
  }

  std::uint32_t effective_batch_size() const noexcept {
    return effective_batch_size_;
  }

 private:
  void begin_query(const DeltaSteppingCsrControllerTraceStep& step) {
    if (descriptor_.phase != DeltaSteppingCsrControllerPhase::kIdle &&
        descriptor_.phase != DeltaSteppingCsrControllerPhase::kFinished) {
      finish_with(DeltaSteppingCsrControllerStatus::kInvalidState);
      return;
    }
    const std::uint32_t next_query_sequence =
        descriptor_.query_sequence ==
                std::numeric_limits<std::uint32_t>::max()
            ? std::uint32_t{1}
            : static_cast<std::uint32_t>(descriptor_.query_sequence + 1);
    descriptor_ = {};
    descriptor_.query_sequence = next_query_sequence;
    descriptor_.current_count = step.current_count;
    descriptor_.current_bucket = step.bucket;
    if (step.status != DeltaSteppingCsrControllerStatus::kNone ||
        step.pending_count != 0 ||
        step.next_bucket != kDeltaSteppingCsrNoControllerBucket ||
        step.bucket == kDeltaSteppingCsrNoControllerBucket) {
      finish_with(DeltaSteppingCsrControllerStatus::kInvalidState);
      return;
    }
    if (step.current_count == 0) {
      finish_with(DeltaSteppingCsrControllerStatus::kComplete);
      return;
    }
    descriptor_.phase = DeltaSteppingCsrControllerPhase::kLightClosure;
    descriptor_.action =
        DeltaSteppingCsrControllerAction::kContinueDevice;
  }

  void apply_light_round(const DeltaSteppingCsrControllerTraceStep& step) {
    if (descriptor_.phase !=
        DeltaSteppingCsrControllerPhase::kLightClosure) {
      finish_with(DeltaSteppingCsrControllerStatus::kInvalidState);
      return;
    }
    DeltaSteppingCsrControllerStatus observed =
        delta_stepping_normalize_controller_status(step.status);
    const bool bucket_closed = step.current_count == 0;
    const bool stops_before_pending_reduction =
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kTargetSettled) ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kQueueOverflow) ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kInvalidState);
    const bool completes_without_successor =
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kComplete);
    const bool pending_state_valid =
        bucket_closed && !stops_before_pending_reduction &&
                !completes_without_successor
            ? delta_stepping_controller_pending_state_is_valid(
                  step.pending_count,
                  descriptor_.current_bucket,
                  step.next_bucket)
            : delta_stepping_controller_light_pending_state_is_valid(
                  step.pending_count,
                  descriptor_.current_bucket,
                  step.next_bucket);
    if (step.bucket != descriptor_.current_bucket ||
        step.pending_count < descriptor_.pending_count ||
        !pending_state_valid ||
        (delta_stepping_controller_has_status(
             observed, DeltaSteppingCsrControllerStatus::kComplete) &&
         (!bucket_closed ||
          step.next_bucket != kDeltaSteppingCsrNoControllerBucket)) ||
        (delta_stepping_controller_has_status(
             observed, DeltaSteppingCsrControllerStatus::kTargetSettled) &&
         !bucket_closed) ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kCallbackAbort) ||
        delta_stepping_controller_has_status(
            observed,
            DeltaSteppingCsrControllerStatus::kIterationLimit)) {
      observed |= DeltaSteppingCsrControllerStatus::kInvalidState;
    }
    if (descriptor_.light_rounds ==
            std::numeric_limits<std::uint32_t>::max() ||
        descriptor_.rounds_since_host_check ==
            std::numeric_limits<std::uint32_t>::max() ||
        (bucket_closed &&
         descriptor_.iterations ==
             std::numeric_limits<std::uint64_t>::max())) {
      observed |= DeltaSteppingCsrControllerStatus::kInvalidState;
    } else {
      ++descriptor_.light_rounds;
      ++descriptor_.rounds_since_host_check;
      if (bucket_closed) ++descriptor_.iterations;
    }
    descriptor_.current_count = step.current_count;
    descriptor_.pending_count = step.pending_count;
    descriptor_.next_bucket = step.next_bucket;
    descriptor_.status |= observed;
    if (descriptor_.status != DeltaSteppingCsrControllerStatus::kNone) {
      finish_with(DeltaSteppingCsrControllerStatus::kNone);
      return;
    }
    if (step.current_count != 0) {
      descriptor_.phase = DeltaSteppingCsrControllerPhase::kLightClosure;
      if (descriptor_.rounds_since_host_check >= effective_batch_size_) {
        publish_host_check();
      } else {
        descriptor_.action =
            DeltaSteppingCsrControllerAction::kContinueDevice;
      }
      return;
    }
    if (step.pending_count != 0) {
      descriptor_.phase =
          DeltaSteppingCsrControllerPhase::kBucketBoundary;
      // The host controller exposes this boundary for its callback and scalar
      // decisions. The reduced controller keeps reduction and compaction in
      // the same device action, then publishes a resumable successor bucket.
      if (mode_ == DeltaSteppingCsrControllerMode::kHostChecked) {
        publish_host_check();
      } else {
        descriptor_.action =
            DeltaSteppingCsrControllerAction::kAdvanceBucket;
      }
      return;
    }
    finish_with(DeltaSteppingCsrControllerStatus::kComplete);
  }

  void begin_bucket(const DeltaSteppingCsrControllerTraceStep& step) {
    DeltaSteppingCsrControllerStatus observed =
        delta_stepping_normalize_controller_status(step.status);
    const std::uint64_t compacted_count =
        static_cast<std::uint64_t>(step.current_count) +
        static_cast<std::uint64_t>(step.pending_count);
    if (descriptor_.phase !=
            DeltaSteppingCsrControllerPhase::kBucketBoundary ||
        step.current_count == 0 || step.bucket != descriptor_.next_bucket ||
        compacted_count > descriptor_.pending_count ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kComplete) ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kTargetSettled) ||
        delta_stepping_controller_has_status(
            observed, DeltaSteppingCsrControllerStatus::kCallbackAbort) ||
        !delta_stepping_controller_light_pending_state_is_valid(
            step.pending_count, step.bucket, step.next_bucket)) {
      observed |= DeltaSteppingCsrControllerStatus::kInvalidState;
    }
    descriptor_.current_count = step.current_count;
    descriptor_.pending_count = step.pending_count;
    descriptor_.current_bucket = step.bucket;
    descriptor_.next_bucket = step.next_bucket;
    descriptor_.phase = DeltaSteppingCsrControllerPhase::kLightClosure;
    descriptor_.status |= observed;
    if (descriptor_.status != DeltaSteppingCsrControllerStatus::kNone) {
      finish_with(DeltaSteppingCsrControllerStatus::kNone);
      return;
    }
    if (mode_ == DeltaSteppingCsrControllerMode::kReducedRoundTrip &&
        descriptor_.rounds_since_host_check >= effective_batch_size_) {
      publish_host_check();
    } else {
      descriptor_.action =
          DeltaSteppingCsrControllerAction::kContinueDevice;
    }
  }

  void callback_abort(const DeltaSteppingCsrControllerTraceStep& step) {
    // Progress callbacks are outer-bucket events in the production host
    // controller. A light-round publication with a still-live current
    // frontier is a status boundary, not a callback boundary.
    if (descriptor_.phase !=
            DeltaSteppingCsrControllerPhase::kBucketBoundary ||
        descriptor_.action !=
            DeltaSteppingCsrControllerAction::kPublishHostCheck ||
        step.status != DeltaSteppingCsrControllerStatus::kNone ||
        step.current_count != 0 || step.pending_count != 0 ||
        step.bucket != 0 ||
        step.next_bucket != kDeltaSteppingCsrNoControllerBucket) {
      finish_with(DeltaSteppingCsrControllerStatus::kInvalidState);
      return;
    }
    finish_with(DeltaSteppingCsrControllerStatus::kCallbackAbort);
  }

  void publish_host_check() {
    descriptor_.action =
        DeltaSteppingCsrControllerAction::kPublishHostCheck;
    increment_publication_sequence();
  }

  void resume_after_host_check() {
    if (descriptor_.action !=
        DeltaSteppingCsrControllerAction::kPublishHostCheck) {
      return;
    }
    descriptor_.rounds_since_host_check = 0;
    descriptor_.action =
        descriptor_.phase == DeltaSteppingCsrControllerPhase::kBucketBoundary
            ? DeltaSteppingCsrControllerAction::kAdvanceBucket
            : DeltaSteppingCsrControllerAction::kContinueDevice;
  }

  void finish_with(DeltaSteppingCsrControllerStatus status) {
    descriptor_.status |=
        delta_stepping_normalize_controller_status(status);
    if (descriptor_.status == DeltaSteppingCsrControllerStatus::kNone) {
      descriptor_.status =
          DeltaSteppingCsrControllerStatus::kInvalidState;
    }
    descriptor_.phase = DeltaSteppingCsrControllerPhase::kFinished;
    descriptor_.action =
        delta_stepping_controller_terminal_action(descriptor_.status);
    increment_publication_sequence();
  }

  void increment_publication_sequence() {
    if (descriptor_.publication_sequence ==
        std::numeric_limits<std::uint32_t>::max()) {
      descriptor_.status |=
          DeltaSteppingCsrControllerStatus::kInvalidState;
      descriptor_.phase = DeltaSteppingCsrControllerPhase::kFinished;
      descriptor_.action =
          DeltaSteppingCsrControllerAction::kStopInvalidState;
      return;
    }
    ++descriptor_.publication_sequence;
  }

  DeltaSteppingCsrControllerMode mode_ =
      DeltaSteppingCsrControllerMode::kHostChecked;
  std::uint32_t effective_batch_size_ = 1;
  DeltaSteppingCsrControllerDescriptor descriptor_{};
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
