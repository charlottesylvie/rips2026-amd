#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Host-testable policies for BF11's segmented controller and retained
// workspace.  The fixed-width/POD functions are also suitable for use by HIP
// kernels.  This header deliberately does not include a HIP header.
namespace bf11_execution_policy {

#if defined(__HIPCC__) || defined(__HIP__)
#define BF11_EXECUTION_POLICY_HD __host__ __device__
#else
#define BF11_EXECUTION_POLICY_HD
#endif

inline constexpr std::uint32_t kDefaultSegmentRounds = 1;
inline constexpr std::uint32_t kDefaultResetThresholdNumerator = 1;
inline constexpr std::uint32_t kDefaultResetThresholdDenominator = 4;
inline constexpr std::uint32_t kPolicyInvalidControllerError = 6;
inline constexpr std::uint32_t kPositiveInfinityFloatBits = 0x7f800000u;
inline constexpr std::uint32_t kNoPredecessorEdge = 0xffffffffu;
inline constexpr std::uint32_t kNoFailureIndex =
    std::numeric_limits<std::uint32_t>::max();

BF11_EXECUTION_POLICY_HD constexpr bool is_supported_segment_rounds(
    std::uint32_t rounds) noexcept {
  return rounds == 1 || rounds == 2 || rounds == 4 || rounds == 8 ||
         rounds == 16;
}

inline void validate_segment_rounds(std::uint32_t rounds) {
  if (!is_supported_segment_rounds(rounds)) {
    throw std::invalid_argument(
        "BF11 segment rounds must be one of 1, 2, 4, 8, or 16");
  }
}

enum class ControllerTerminalReason : std::uint32_t {
  kRunning = 0,
  kConverged = 1,
  kTargetCertified = 2,
  kMaxIterations = 3,
  kError = 4,
  kInvalidConfiguration = 5,
};

struct SegmentedControllerConfig {
  std::uint32_t initial_frontier_count = 0;
  std::uint32_t target_count = 0;
  std::uint32_t max_iterations = 0;
  std::uint32_t target_check_interval = 1;
  // The caller reserves max_iterations consecutive nonzero tokens before
  // initializing the descriptor. Each executed round consumes one token.
  std::uint32_t first_mark_token = 0;
};

// Device-resident controller state. A segment enqueues fixed geometry for K
// rounds. Relaxation and target-check kernels inspect terminal_reason and are
// no-ops after termination; the finalize kernel calls finalize_controller_round
// once per slot so no-op accounting remains exact.
struct SegmentedControllerDescriptor {
  static constexpr std::uint32_t kVersion = 1;

  std::uint32_t version = kVersion;
  std::uint32_t frontier_count = 0;
  std::uint32_t current_frontier_index = 0;
  std::uint32_t iterations_used = 0;
  std::uint32_t max_iterations = 0;
  std::uint32_t target_check_interval = 1;
  std::uint32_t target_count = 0;
  std::uint32_t mark_token = 0;
  std::int32_t error_status = 0;
  ControllerTerminalReason terminal_reason =
      ControllerTerminalReason::kRunning;
  std::uint32_t target_checks = 0;
  std::uint32_t all_targets_reached = 0;
  std::uint32_t actual_rounds = 0;
  std::uint32_t no_op_rounds = 0;
};

static_assert(
    std::is_standard_layout<SegmentedControllerDescriptor>::value,
    "BF11 segmented controller must have standard layout");
static_assert(
    std::is_trivially_copyable<SegmentedControllerDescriptor>::value,
    "BF11 segmented controller must be trivially copyable");
static_assert(sizeof(SegmentedControllerDescriptor) == 56,
              "BF11 segmented controller layout unexpectedly changed");

// One relaxation round's completed device observations. Queue entries and
// packed labels must be published before this is consumed: production obtains
// that ordering from the preceding kernel boundary (or a cooperative grid
// barrier). The next round likewise cannot consume next_count/frontier entries
// until this finalize kernel completes.
struct SegmentedRoundObservation {
  std::uint32_t next_count = 0;
  std::int32_t error_status = 0;
  std::uint32_t reached_target_count = 0;
  std::uint32_t min_next_frontier_dist_bits =
      kPositiveInfinityFloatBits;
  std::uint32_t max_target_dist_bits = 0;
};

static_assert(std::is_standard_layout<SegmentedRoundObservation>::value,
              "BF11 round observation must have standard layout");
static_assert(std::is_trivially_copyable<SegmentedRoundObservation>::value,
              "BF11 round observation must be trivially copyable");
static_assert(sizeof(SegmentedRoundObservation) == 20,
              "BF11 round observation layout unexpectedly changed");

struct ControllerRoundAdvance {
  bool executed = false;
  // A reachability scan may also run at convergence/max_iters so bounded
  // fallback can be decided before extraction. Only certificate_evaluated is
  // allowed to trigger the interval-controlled exact early stop.
  bool checked_targets = false;
  bool certificate_evaluated = false;
  bool swapped_frontiers = false;
  ControllerTerminalReason terminal_reason =
      ControllerTerminalReason::kRunning;
};

BF11_EXECUTION_POLICY_HD constexpr bool controller_is_done(
    const SegmentedControllerDescriptor& descriptor) noexcept {
  return descriptor.terminal_reason != ControllerTerminalReason::kRunning;
}

BF11_EXECUTION_POLICY_HD constexpr bool target_certificate_due(
    std::uint32_t completed_iteration,
    std::uint32_t target_check_interval) noexcept {
  return completed_iteration != 0 && target_check_interval != 0 &&
         completed_iteration % target_check_interval == 0;
}

BF11_EXECUTION_POLICY_HD constexpr bool target_scan_due(
    std::uint32_t completed_iteration,
    std::uint32_t max_iterations,
    std::uint32_t target_check_interval,
    std::uint32_t next_count) noexcept {
  return target_certificate_due(completed_iteration,
                                target_check_interval) ||
         next_count == 0 || completed_iteration >= max_iterations;
}

BF11_EXECUTION_POLICY_HD constexpr bool exact_target_certificate_holds(
    bool all_targets_reached,
    std::uint32_t min_next_frontier_dist_bits,
    std::uint32_t max_target_dist_bits) noexcept {
  return all_targets_reached &&
         min_next_frontier_dist_bits >= max_target_dist_bits;
}

// Stateless stop policy for production descriptors that keep hot status
// fields on separate cache lines rather than storing the compact model POD.
BF11_EXECUTION_POLICY_HD constexpr ControllerTerminalReason
completed_round_terminal_reason(
    std::int32_t error_status,
    std::uint32_t next_count,
    bool certificate_due,
    bool all_targets_reached,
    std::uint32_t min_next_frontier_dist_bits,
    std::uint32_t max_target_dist_bits,
    std::uint32_t completed_iteration,
    std::uint32_t max_iterations) noexcept {
  if (error_status != 0) return ControllerTerminalReason::kError;
  if (next_count == 0) return ControllerTerminalReason::kConverged;
  if (certificate_due &&
      exact_target_certificate_holds(all_targets_reached,
                                     min_next_frontier_dist_bits,
                                     max_target_dist_bits)) {
    return ControllerTerminalReason::kTargetCertified;
  }
  return completed_iteration >= max_iterations
             ? ControllerTerminalReason::kMaxIterations
             : ControllerTerminalReason::kRunning;
}

BF11_EXECUTION_POLICY_HD constexpr bool controller_target_check_due(
    const SegmentedControllerDescriptor& descriptor) noexcept {
  return !controller_is_done(descriptor) &&
         target_certificate_due(descriptor.iterations_used + 1,
                                descriptor.target_check_interval);
}

BF11_EXECUTION_POLICY_HD constexpr bool controller_target_scan_due(
    const SegmentedControllerDescriptor& descriptor,
    std::uint32_t next_count) noexcept {
  return !controller_is_done(descriptor) &&
         target_scan_due(descriptor.iterations_used + 1,
                         descriptor.max_iterations,
                         descriptor.target_check_interval, next_count);
}

BF11_EXECUTION_POLICY_HD constexpr SegmentedControllerDescriptor
initialize_segmented_controller(
    const SegmentedControllerConfig& config) noexcept {
  SegmentedControllerDescriptor descriptor{};
  descriptor.frontier_count = config.initial_frontier_count;
  descriptor.max_iterations = config.max_iterations;
  descriptor.target_check_interval = config.target_check_interval;
  descriptor.target_count = config.target_count;
  descriptor.mark_token = config.first_mark_token;

  const bool token_range_invalid =
      config.max_iterations != 0 && config.initial_frontier_count != 0 &&
      (config.first_mark_token == 0 ||
       config.max_iterations - 1 >
           std::numeric_limits<std::uint32_t>::max() -
               config.first_mark_token);
  if (config.target_count == 0 || config.target_check_interval == 0 ||
      token_range_invalid) {
    descriptor.error_status =
        static_cast<std::int32_t>(kPolicyInvalidControllerError);
    descriptor.terminal_reason =
        ControllerTerminalReason::kInvalidConfiguration;
  } else if (config.max_iterations == 0) {
    // No relaxation is legal. In particular, an identity source/target query
    // may be recognized by the caller after seeding without changing this
    // exact max-iteration accounting.
    descriptor.terminal_reason = ControllerTerminalReason::kMaxIterations;
  } else if (config.initial_frontier_count == 0) {
    descriptor.terminal_reason = ControllerTerminalReason::kConverged;
  }
  return descriptor;
}

// Finalize exactly one scheduled segment slot. Stop precedence mirrors BF11's
// established controller: error, empty next frontier, exact target
// certificate, then max_iters. A terminal descriptor is sticky and only its
// no-op counter changes.
BF11_EXECUTION_POLICY_HD inline ControllerRoundAdvance
finalize_controller_round(
    SegmentedControllerDescriptor* descriptor,
    const SegmentedRoundObservation& observation) noexcept {
  ControllerRoundAdvance advance{};
  if (descriptor == nullptr) {
    advance.terminal_reason =
        ControllerTerminalReason::kInvalidConfiguration;
    return advance;
  }
  if (controller_is_done(*descriptor)) {
    if (descriptor->no_op_rounds !=
        std::numeric_limits<std::uint32_t>::max()) {
      ++descriptor->no_op_rounds;
    }
    advance.terminal_reason = descriptor->terminal_reason;
    return advance;
  }

  advance.executed = true;
  advance.certificate_evaluated =
      controller_target_check_due(*descriptor);
  advance.checked_targets =
      controller_target_scan_due(*descriptor, observation.next_count);
  ++descriptor->iterations_used;
  ++descriptor->actual_rounds;
  descriptor->frontier_count = observation.next_count;
  if (advance.checked_targets) {
    ++descriptor->target_checks;
    descriptor->all_targets_reached =
        observation.reached_target_count == descriptor->target_count ? 1u
                                                                      : 0u;
  }

  if (observation.error_status == 0 && advance.checked_targets &&
      observation.reached_target_count > descriptor->target_count) {
    descriptor->error_status =
        static_cast<std::int32_t>(kPolicyInvalidControllerError);
    descriptor->terminal_reason =
        ControllerTerminalReason::kInvalidConfiguration;
  } else {
    descriptor->error_status = observation.error_status;
    descriptor->terminal_reason = completed_round_terminal_reason(
        observation.error_status, observation.next_count,
        advance.certificate_evaluated,
        descriptor->all_targets_reached != 0,
        observation.min_next_frontier_dist_bits,
        observation.max_target_dist_bits, descriptor->iterations_used,
        descriptor->max_iterations);
  }
  if (descriptor->terminal_reason != ControllerTerminalReason::kRunning) {
    advance.terminal_reason = descriptor->terminal_reason;
    return advance;
  }
  if (descriptor->mark_token ==
             std::numeric_limits<std::uint32_t>::max()) {
    // A valid reservation makes this unreachable. Treat it as corruption
    // rather than reusing generation zero or wrapping onto a live token.
    descriptor->error_status =
        static_cast<std::int32_t>(kPolicyInvalidControllerError);
    descriptor->terminal_reason =
        ControllerTerminalReason::kInvalidConfiguration;
  } else {
    descriptor->current_frontier_index ^= 1u;
    ++descriptor->mark_token;
    advance.swapped_frontiers = true;
  }
  advance.terminal_reason = descriptor->terminal_reason;
  return advance;
}

BF11_EXECUTION_POLICY_HD constexpr bool controller_semantically_equivalent(
    const SegmentedControllerDescriptor& lhs,
    const SegmentedControllerDescriptor& rhs) noexcept {
  return lhs.version == rhs.version &&
         lhs.frontier_count == rhs.frontier_count &&
         lhs.current_frontier_index == rhs.current_frontier_index &&
         lhs.iterations_used == rhs.iterations_used &&
         lhs.max_iterations == rhs.max_iterations &&
         lhs.target_check_interval == rhs.target_check_interval &&
         lhs.target_count == rhs.target_count &&
         lhs.mark_token == rhs.mark_token &&
         lhs.error_status == rhs.error_status &&
         lhs.terminal_reason == rhs.terminal_reason &&
         lhs.target_checks == rhs.target_checks &&
         lhs.all_targets_reached == rhs.all_targets_reached &&
         lhs.actual_rounds == rhs.actual_rounds;
}

struct SegmentedControllerModelResult {
  SegmentedControllerDescriptor descriptor{};
  std::uint64_t segments = 0;
  std::uint64_t status_copies = 0;
  std::size_t observations_consumed = 0;
};

// Deterministic CPU model of fixed-K enqueue. It models controller state and
// publication frequency, not GPU scheduling or memory ordering.
class SegmentedControllerSequentialModel {
 public:
  SegmentedControllerSequentialModel(SegmentedControllerConfig config,
                                     std::uint32_t segment_rounds)
      : descriptor_(initialize_segmented_controller(config)),
        segment_rounds_(segment_rounds) {
    validate_segment_rounds(segment_rounds);
  }

  const SegmentedControllerDescriptor& descriptor() const noexcept {
    return descriptor_;
  }

  std::uint32_t segment_rounds() const noexcept { return segment_rounds_; }

  void run_one_segment(
      const std::vector<SegmentedRoundObservation>& observations,
      std::size_t* observation_cursor) {
    if (observation_cursor == nullptr) {
      throw std::invalid_argument("BF11 model requires an observation cursor");
    }
    ++segments_;
    for (std::uint32_t slot = 0; slot < segment_rounds_; ++slot) {
      if (controller_is_done(descriptor_)) {
        (void)finalize_controller_round(&descriptor_, {});
        continue;
      }
      if (*observation_cursor >= observations.size()) {
        throw std::invalid_argument(
            "BF11 controller trace ended before a terminal state");
      }
      (void)finalize_controller_round(
          &descriptor_, observations[(*observation_cursor)++]);
    }
    ++status_copies_;
  }

  SegmentedControllerModelResult run_to_completion(
      const std::vector<SegmentedRoundObservation>& observations) {
    std::size_t cursor = 0;
    while (!controller_is_done(descriptor_)) {
      run_one_segment(observations, &cursor);
    }
    return {descriptor_, segments_, status_copies_, cursor};
  }

 private:
  SegmentedControllerDescriptor descriptor_{};
  std::uint32_t segment_rounds_ = 1;
  std::uint64_t segments_ = 0;
  std::uint64_t status_copies_ = 0;
};

struct MarkTokenReservation {
  std::uint32_t first_token = 0;
  std::uint32_t last_token = 0;
  std::uint32_t token_count = 0;
  // Zero is an exhausted sentinel. The next nonempty reservation performs a
  // dense mark reset before restarting at token one.
  std::uint32_t next_token = 1;
  bool dense_reset_required = false;
  bool valid = true;
};

// Reserve a consecutive nonzero token range. max_token is normally UINT32_MAX
// and can be reduced by tests to force wrap without billions of queries.
BF11_EXECUTION_POLICY_HD constexpr MarkTokenReservation reserve_mark_tokens(
    std::uint32_t next_token,
    std::uint32_t token_count,
    std::uint32_t max_token =
        std::numeric_limits<std::uint32_t>::max()) noexcept {
  MarkTokenReservation result{};
  result.token_count = token_count;
  result.next_token = next_token;
  if (token_count == 0) return result;
  if (max_token == 0 || token_count > max_token) {
    result.valid = false;
    return result;
  }

  const bool must_reset =
      next_token == 0 || next_token > max_token ||
      token_count - 1 > max_token - next_token;
  result.dense_reset_required = must_reset;
  result.first_token = must_reset ? 1u : next_token;
  result.last_token = result.first_token + token_count - 1;
  result.next_token = result.last_token == max_token
                          ? 0u
                          : result.last_token + 1;
  return result;
}

BF11_EXECUTION_POLICY_HD constexpr std::uint32_t mark_token_for_round(
    const MarkTokenReservation& reservation,
    std::uint32_t zero_based_round) noexcept {
  return reservation.valid && zero_based_round < reservation.token_count
             ? reservation.first_token + zero_based_round
             : 0u;
}

class MarkTokenAllocatorModel {
 public:
  explicit MarkTokenAllocatorModel(
      std::uint32_t max_token =
          std::numeric_limits<std::uint32_t>::max(),
      std::uint32_t next_token = 1)
      : max_token_(max_token), next_token_(next_token) {
    if (max_token == 0) {
      throw std::invalid_argument("BF11 mark-token maximum must be positive");
    }
  }

  MarkTokenReservation reserve(std::uint32_t token_count) {
    const MarkTokenReservation reservation =
        reserve_mark_tokens(next_token_, token_count, max_token_);
    if (!reservation.valid) {
      throw std::overflow_error(
          "BF11 query requires more mark tokens than the generation range");
    }
    next_token_ = reservation.next_token;
    if (reservation.dense_reset_required) ++dense_reset_count_;
    return reservation;
  }

  std::uint32_t next_token() const noexcept { return next_token_; }
  std::uint64_t dense_reset_count() const noexcept {
    return dense_reset_count_;
  }

 private:
  std::uint32_t max_token_ = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t next_token_ = 1;
  std::uint64_t dense_reset_count_ = 0;
};

struct AdaptiveResetPolicy {
  std::uint32_t dense_threshold_numerator =
      kDefaultResetThresholdNumerator;
  std::uint32_t dense_threshold_denominator =
      kDefaultResetThresholdDenominator;
};

inline void validate_reset_policy(const AdaptiveResetPolicy& policy) {
  if (policy.dense_threshold_numerator == 0 ||
      policy.dense_threshold_denominator == 0 ||
      policy.dense_threshold_numerator >
          policy.dense_threshold_denominator) {
    throw std::invalid_argument(
        "BF11 dense-reset threshold must be in the interval (0, 1]");
  }
}

enum class ResetMode : std::uint32_t {
  kSparse = 0,
  kDense = 1,
};

enum class ResetReason : std::uint32_t {
  kEmptyTouchedSet = 0,
  kBelowDensityThreshold = 1,
  kAtOrAboveDensityThreshold = 2,
  kDefensiveFailureRecovery = 3,
  kMarkGenerationWrap = 4,
  kInvalidTouchedCount = 5,
  kInvalidPolicy = 6,
};

struct ResetSelectionInputs {
  std::uint64_t touched_count = 0;
  std::uint64_t vertex_count = 0;
  bool defensive_full_reset = false;
  bool mark_generation_wrap = false;
};

struct ResetDecision {
  ResetMode mode = ResetMode::kSparse;
  ResetReason reason = ResetReason::kEmptyTouchedSet;
  std::uint64_t dense_threshold_count = 0;
};

// ceil(value * numerator / denominator), avoiding overflow when
// 0 < numerator <= denominator.
BF11_EXECUTION_POLICY_HD constexpr std::uint64_t scaled_ceiling(
    std::uint64_t value,
    std::uint32_t numerator,
    std::uint32_t denominator) noexcept {
  if (numerator == 0 || denominator == 0 || numerator > denominator) return 0;
  const std::uint64_t quotient = value / denominator;
  const std::uint64_t remainder = value % denominator;
  return quotient * numerator +
         (remainder * numerator + denominator - 1) / denominator;
}

// This decision can run on-device after reading touched_count, avoiding a host
// status copy solely to choose sparse versus dense reset.
BF11_EXECUTION_POLICY_HD constexpr ResetDecision select_reset_mode(
    const ResetSelectionInputs& inputs,
    const AdaptiveResetPolicy& policy = {}) noexcept {
  if (policy.dense_threshold_numerator == 0 ||
      policy.dense_threshold_denominator == 0 ||
      policy.dense_threshold_numerator >
          policy.dense_threshold_denominator) {
    return {ResetMode::kDense, ResetReason::kInvalidPolicy, 0};
  }
  const std::uint64_t threshold = scaled_ceiling(
      inputs.vertex_count, policy.dense_threshold_numerator,
      policy.dense_threshold_denominator);
  if (inputs.defensive_full_reset) {
    return {ResetMode::kDense, ResetReason::kDefensiveFailureRecovery,
            threshold};
  }
  if (inputs.mark_generation_wrap) {
    return {ResetMode::kDense, ResetReason::kMarkGenerationWrap, threshold};
  }
  if (inputs.touched_count > inputs.vertex_count) {
    return {ResetMode::kDense, ResetReason::kInvalidTouchedCount, threshold};
  }
  if (inputs.touched_count == 0) {
    return {ResetMode::kSparse, ResetReason::kEmptyTouchedSet, threshold};
  }
  if (inputs.touched_count >= threshold) {
    return {ResetMode::kDense, ResetReason::kAtOrAboveDensityThreshold,
            threshold};
  }
  return {ResetMode::kSparse, ResetReason::kBelowDensityThreshold,
          threshold};
}

enum class CostMode : std::uint32_t {
  kConstantOne = 0,
  kStatic = 1,
  kDynamic = 2,
};

struct CostPolicyState {
  bool static_costs_constant_one = false;
  // Logical multiplier state used by traversal and reconstruction.
  bool identity_proven = true;
  // Allocation is retained after first use. Its contents can be stale while
  // identity_proven is true because identity mode does not read this array.
  bool dynamic_storage_allocated = false;
  bool dynamic_storage_contains_identity = false;
};

BF11_EXECUTION_POLICY_HD constexpr CostMode cost_mode(
    const CostPolicyState& state) noexcept {
  return !state.identity_proven
             ? CostMode::kDynamic
             : (state.static_costs_constant_one ? CostMode::kConstantOne
                                                : CostMode::kStatic);
}

BF11_EXECUTION_POLICY_HD constexpr bool cost_policy_state_is_valid(
    const CostPolicyState& state) noexcept {
  return (state.identity_proven || state.dynamic_storage_allocated) &&
         (!state.dynamic_storage_contains_identity ||
          (state.dynamic_storage_allocated && state.identity_proven));
}

enum class CostUpdateKind : std::uint32_t {
  kSparse = 0,
  kCompleteReplacement = 1,
};

struct CostUpdateSummary {
  CostUpdateKind kind = CostUpdateKind::kSparse;
  std::uint64_t value_count = 0;
  bool all_values_are_identity = true;
};

struct CostUpdatePlan {
  CostPolicyState before{};
  CostPolicyState after{};
  CostMode mode_before = CostMode::kStatic;
  CostMode mode_after = CostMode::kStatic;
  bool allocate_dynamic_storage = false;
  bool initialize_dynamic_storage_to_identity = false;
  bool apply_sparse_update = false;
  bool upload_complete_replacement = false;
  bool skip_device_write = false;
  bool valid = true;
};

// Decide lazy dynamic-array work without inspecting individual values on the
// device. Callers validate values and compute all_values_are_identity on the
// host. Sparse all-one updates cannot re-prove global identity once dynamic
// mode has been entered; only a complete replacement can do so.
BF11_EXECUTION_POLICY_HD constexpr CostUpdatePlan plan_cost_update(
    const CostPolicyState& state,
    const CostUpdateSummary& update) noexcept {
  CostUpdatePlan plan{};
  plan.before = state;
  plan.after = state;
  plan.mode_before = cost_mode(state);
  plan.mode_after = plan.mode_before;
  if (!cost_policy_state_is_valid(state) ||
      (update.kind != CostUpdateKind::kSparse &&
       update.kind != CostUpdateKind::kCompleteReplacement)) {
    plan.valid = false;
    return plan;
  }

  if (update.kind == CostUpdateKind::kCompleteReplacement) {
    if (update.all_values_are_identity) {
      plan.after.identity_proven = true;
      // Skipping the all-one upload is safe because static/constant traversal
      // and reconstruction do not read the retained dynamic array. If it is
      // stale, a later sparse departure from identity reinitializes it first.
      plan.skip_device_write = true;
    } else {
      if (!plan.after.dynamic_storage_allocated) {
        plan.after.dynamic_storage_allocated = true;
        plan.allocate_dynamic_storage = true;
      }
      plan.upload_complete_replacement = true;
      plan.after.identity_proven = false;
      plan.after.dynamic_storage_contains_identity = false;
    }
    plan.mode_after = cost_mode(plan.after);
    return plan;
  }

  if (update.value_count == 0) {
    plan.skip_device_write = true;
    return plan;
  }
  if (state.identity_proven && update.all_values_are_identity) {
    plan.skip_device_write = true;
    return plan;
  }

  if (state.identity_proven) {
    if (!plan.after.dynamic_storage_allocated) {
      plan.after.dynamic_storage_allocated = true;
      plan.allocate_dynamic_storage = true;
    }
    if (!plan.after.dynamic_storage_contains_identity) {
      plan.initialize_dynamic_storage_to_identity = true;
      plan.after.dynamic_storage_contains_identity = true;
    }
    plan.after.identity_proven = false;
  }
  plan.apply_sparse_update = true;
  // Any nonempty sparse write made while dynamic cannot prove the full array
  // is identity, even when every supplied value is one.
  plan.after.identity_proven = false;
  plan.after.dynamic_storage_contains_identity = false;
  plan.mode_after = CostMode::kDynamic;
  return plan;
}

enum class TargetCardinalityStatus : std::uint32_t {
  kUnreachable = 0,
  kValid = 1,
  kInvalid = 2,
};

struct TargetPathCardinality {
  std::uint64_t node_count = 0;
  std::uint64_t edge_count = 0;
  TargetCardinalityStatus status = TargetCardinalityStatus::kUnreachable;
};

enum class PrefixScanStatus : std::uint32_t {
  kReady = 0,
  kNeedsGrowth = 1,
  kInvalidSummary = 2,
  kCountOverflow = 3,
  kResultLimitExceeded = 4,
};

struct TargetPrefixOffsets {
  std::uint64_t node_offset = 0;
  std::uint64_t edge_offset = 0;
};

struct PrefixScanAccumulator {
  std::uint64_t total_nodes = 0;
  std::uint64_t total_edges = 0;
  std::uint32_t processed_targets = 0;
  std::uint32_t failure_index = kNoFailureIndex;
  PrefixScanStatus status = PrefixScanStatus::kReady;
};

struct PrefixScanHeader {
  PrefixScanStatus status = PrefixScanStatus::kReady;
  std::uint64_t total_nodes = 0;
  std::uint64_t total_edges = 0;
  std::uint64_t required_node_capacity = 0;
  std::uint64_t required_edge_capacity = 0;
  std::uint32_t processed_targets = 0;
  std::uint32_t failure_index = kNoFailureIndex;
};

// Sequential deterministic target-order scan primitive. A one-thread device
// kernel can call this while writing offsets, avoiding nondeterministic atomics
// and preserving duplicate-target order.
BF11_EXECUTION_POLICY_HD inline TargetPrefixOffsets append_target_prefix(
    PrefixScanAccumulator* accumulator,
    const TargetPathCardinality& cardinality,
    std::uint64_t result_item_limit =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max())) noexcept {
  if (accumulator == nullptr) return {};
  const TargetPrefixOffsets offsets{accumulator->total_nodes,
                                    accumulator->total_edges};
  if (accumulator->status != PrefixScanStatus::kReady) return offsets;

  const std::uint32_t item = accumulator->processed_targets++;
  if (cardinality.status == TargetCardinalityStatus::kInvalid ||
      (cardinality.status == TargetCardinalityStatus::kUnreachable &&
       (cardinality.node_count != 0 || cardinality.edge_count != 0)) ||
      (cardinality.status != TargetCardinalityStatus::kUnreachable &&
       cardinality.status != TargetCardinalityStatus::kValid)) {
    accumulator->status = PrefixScanStatus::kInvalidSummary;
    accumulator->failure_index = item;
    return offsets;
  }
  if (cardinality.status == TargetCardinalityStatus::kUnreachable) {
    return offsets;
  }
  if (cardinality.node_count == 0 ||
      cardinality.edge_count == std::numeric_limits<std::uint64_t>::max() ||
      cardinality.node_count != cardinality.edge_count + 1) {
    accumulator->status = PrefixScanStatus::kInvalidSummary;
    accumulator->failure_index = item;
    return offsets;
  }
  if (cardinality.node_count >
          std::numeric_limits<std::uint64_t>::max() -
              accumulator->total_nodes ||
      cardinality.edge_count >
          std::numeric_limits<std::uint64_t>::max() -
              accumulator->total_edges) {
    accumulator->status = PrefixScanStatus::kCountOverflow;
    accumulator->failure_index = item;
    return offsets;
  }
  accumulator->total_nodes += cardinality.node_count;
  accumulator->total_edges += cardinality.edge_count;
  if (accumulator->total_nodes > result_item_limit ||
      accumulator->total_edges > result_item_limit) {
    accumulator->status = PrefixScanStatus::kResultLimitExceeded;
    accumulator->failure_index = item;
  }
  return offsets;
}

BF11_EXECUTION_POLICY_HD constexpr PrefixScanHeader finalize_prefix_scan(
    const PrefixScanAccumulator& accumulator,
    std::uint64_t node_capacity,
    std::uint64_t edge_capacity) noexcept {
  PrefixScanHeader header{};
  header.status = accumulator.status;
  header.total_nodes = accumulator.total_nodes;
  header.total_edges = accumulator.total_edges;
  header.required_node_capacity = accumulator.total_nodes;
  header.required_edge_capacity = accumulator.total_edges;
  header.processed_targets = accumulator.processed_targets;
  header.failure_index = accumulator.failure_index;
  if (header.status == PrefixScanStatus::kReady &&
      (header.total_nodes > node_capacity ||
       header.total_edges > edge_capacity)) {
    header.status = PrefixScanStatus::kNeedsGrowth;
  }
  return header;
}

struct PrefixScanResult {
  PrefixScanHeader header{};
  std::vector<TargetPrefixOffsets> offsets;
};

inline PrefixScanResult scan_target_prefixes(
    const std::vector<TargetPathCardinality>& targets,
    std::uint64_t node_capacity,
    std::uint64_t edge_capacity,
    std::uint64_t result_item_limit =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max())) {
  PrefixScanAccumulator accumulator{};
  PrefixScanResult result;
  result.offsets.reserve(targets.size());
  for (const TargetPathCardinality& target : targets) {
    result.offsets.push_back(
        append_target_prefix(&accumulator, target, result_item_limit));
  }
  result.header =
      finalize_prefix_scan(accumulator, node_capacity, edge_capacity);
  return result;
}

enum class PredecessorChainStatus : std::uint32_t {
  kUnreachable = 0,
  kValid = 1,
  kInvalid = 2,
};

struct PackedPredecessorModelState {
  std::uint32_t distance_bits = kPositiveInfinityFloatBits;
  std::uint32_t predecessor_edge = kNoPredecessorEdge;
};

struct PredecessorModelEdge {
  std::int32_t source = -1;
  std::int32_t destination = -1;
};

struct PredecessorChainModelResult {
  PredecessorChainStatus status = PredecessorChainStatus::kInvalid;
  std::int32_t root = -1;
  std::uint64_t node_count = 0;
  std::uint64_t edge_count = 0;
};

// Host model of the production reconstruction guard. It intentionally checks
// edge destination ownership, source ranges, the zero/no-predecessor root
// predicate, and a V-edge cycle bound so malformed predecessor state can be
// exercised without a GPU mutation hook.
inline PredecessorChainModelResult validate_predecessor_chain_model(
    const std::vector<PackedPredecessorModelState>& states,
    const std::vector<PredecessorModelEdge>& edges,
    std::int32_t target) {
  PredecessorChainModelResult result{};
  if (target < 0 || static_cast<std::size_t>(target) >= states.size()) {
    return result;
  }
  if (states[static_cast<std::size_t>(target)].distance_bits ==
      kPositiveInfinityFloatBits) {
    result.status = PredecessorChainStatus::kUnreachable;
    return result;
  }

  std::int32_t current = target;
  for (std::size_t guard = 0; guard <= states.size(); ++guard) {
    const PackedPredecessorModelState state =
        states[static_cast<std::size_t>(current)];
    if (state.predecessor_edge == kNoPredecessorEdge) {
      if (state.distance_bits == 0u) {
        result.status = PredecessorChainStatus::kValid;
        result.root = current;
        result.node_count = result.edge_count + 1;
      }
      return result;
    }
    if (static_cast<std::size_t>(state.predecessor_edge) >= edges.size()) {
      return result;
    }
    const PredecessorModelEdge edge =
        edges[static_cast<std::size_t>(state.predecessor_edge)];
    if (edge.destination != current || edge.source < 0 ||
        static_cast<std::size_t>(edge.source) >= states.size()) {
      return result;
    }
    current = edge.source;
    ++result.edge_count;
  }
  return result;
}

struct CapacityGrowth {
  std::uint64_t capacity = 0;
  bool grew = false;
  bool valid = true;
};

// Mirror retained grow-on-demand behavior: double the current high-water mark
// until sufficient. If the next doubling would cross a legal maximum, use the
// exact required capacity instead.
BF11_EXECUTION_POLICY_HD constexpr CapacityGrowth grow_retained_capacity(
    std::uint64_t current,
    std::uint64_t required,
    std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max()) noexcept {
  if (current > maximum) return {current, false, false};
  if (required > maximum) return {current, false, false};
  if (current >= required) return {current, false, true};
  std::uint64_t grown = current == 0 ? 1 : current;
  while (grown < required) {
    if (grown > maximum / 2) {
      grown = required;
      break;
    }
    grown *= 2;
  }
  return {grown, true, grown >= required && grown <= maximum};
}

enum class TraversalPhase : std::uint32_t {
  kAwaitingBounded = 0,
  kAwaitingUnbounded = 1,
  kExtractionPending = 2,
  kFinished = 3,
  kFailed = 4,
};

enum class TraversalDecision : std::uint32_t {
  kRetryUnbounded = 0,
  kExtractAcceptedAttempt = 1,
  kPropagateError = 2,
  kInvalidTransition = 3,
};

struct TraversalAttemptObservation {
  bool bounded = false;
  bool completed = true;
  bool all_targets_reached = false;
  std::int32_t iterations_used = 0;
};

struct FallbackExtractionState {
  TraversalPhase phase = TraversalPhase::kAwaitingBounded;
  std::int32_t accumulated_iterations = 0;
  std::uint32_t traversal_attempts = 0;
  std::uint32_t extractions = 0;
  std::uint32_t bounded_fallbacks = 0;
  std::uint32_t avoided_failed_attempt_extractions = 0;
  bool accepted_attempt_was_bounded = false;
  bool accepted_attempt_reached_all_targets = false;
};

BF11_EXECUTION_POLICY_HD constexpr FallbackExtractionState
initialize_fallback_extraction(bool first_attempt_is_bounded) noexcept {
  FallbackExtractionState state{};
  state.phase = first_attempt_is_bounded
                    ? TraversalPhase::kAwaitingBounded
                    : TraversalPhase::kAwaitingUnbounded;
  return state;
}

BF11_EXECUTION_POLICY_HD constexpr std::int32_t saturating_iteration_add(
    std::int32_t left,
    std::int32_t right) noexcept {
  if (right <= 0) return left;
  return left > std::numeric_limits<std::int32_t>::max() - right
             ? std::numeric_limits<std::int32_t>::max()
             : left + right;
}

BF11_EXECUTION_POLICY_HD inline TraversalDecision record_traversal_attempt(
    FallbackExtractionState* state,
    const TraversalAttemptObservation& observation,
    bool unbounded_fallback_enabled) noexcept {
  if (state == nullptr || observation.iterations_used < 0) {
    return TraversalDecision::kInvalidTransition;
  }
  const bool expected_bounded =
      state->phase == TraversalPhase::kAwaitingBounded;
  const bool expected_unbounded =
      state->phase == TraversalPhase::kAwaitingUnbounded;
  if ((!expected_bounded && !expected_unbounded) ||
      observation.bounded != expected_bounded) {
    return TraversalDecision::kInvalidTransition;
  }
  ++state->traversal_attempts;
  state->accumulated_iterations = saturating_iteration_add(
      state->accumulated_iterations, observation.iterations_used);
  if (!observation.completed) {
    state->phase = TraversalPhase::kFailed;
    return TraversalDecision::kPropagateError;
  }
  if (observation.bounded && !observation.all_targets_reached &&
      unbounded_fallback_enabled) {
    ++state->bounded_fallbacks;
    ++state->avoided_failed_attempt_extractions;
    state->phase = TraversalPhase::kAwaitingUnbounded;
    return TraversalDecision::kRetryUnbounded;
  }
  state->accepted_attempt_was_bounded = observation.bounded;
  state->accepted_attempt_reached_all_targets =
      observation.all_targets_reached;
  state->phase = TraversalPhase::kExtractionPending;
  return TraversalDecision::kExtractAcceptedAttempt;
}

BF11_EXECUTION_POLICY_HD inline bool record_extraction(
    FallbackExtractionState* state) noexcept {
  if (state == nullptr || state->phase != TraversalPhase::kExtractionPending) {
    return false;
  }
  ++state->extractions;
  state->phase = TraversalPhase::kFinished;
  return true;
}

#undef BF11_EXECUTION_POLICY_HD

}  // namespace bf11_execution_policy
