#include "../bellman_ford/bf11_execution_policy.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace policy = bf11_execution_policy;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_invalid_argument(Function&& function,
                              const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, message);
}

template <typename Function>
void require_overflow(Function&& function, const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, message);
}

policy::SegmentedControllerConfig controller_config(
    std::uint32_t max_iterations = 20,
    std::uint32_t target_check_interval = 1) {
  return {/*initial_frontier_count=*/2,
          /*target_count=*/2,
          max_iterations,
          target_check_interval,
          /*first_mark_token=*/100};
}

policy::SegmentedRoundObservation continuing_round(
    std::uint32_t next_count = 2) {
  return {next_count,
          /*error_status=*/0,
          /*reached_target_count=*/0,
          /*min_next_frontier_dist_bits=*/100,
          /*max_target_dist_bits=*/0};
}

void test_segment_validation_and_equivalence() {
  static_assert(policy::is_supported_segment_rounds(1));
  static_assert(policy::is_supported_segment_rounds(2));
  static_assert(policy::is_supported_segment_rounds(4));
  static_assert(policy::is_supported_segment_rounds(8));
  static_assert(policy::is_supported_segment_rounds(16));
  static_assert(!policy::is_supported_segment_rounds(0));
  static_assert(!policy::is_supported_segment_rounds(3));
  static_assert(!policy::is_supported_segment_rounds(32));
  static_assert(policy::target_certificate_due(8, 4));
  static_assert(!policy::target_certificate_due(7, 4));
  static_assert(policy::target_scan_due(7, 7, 4, 2));
  static_assert(policy::completed_round_terminal_reason(
                    9, 0, true, true, 10, 10, 4, 4) ==
                policy::ControllerTerminalReason::kError);
  static_assert(policy::completed_round_terminal_reason(
                    0, 0, true, true, 10, 10, 4, 4) ==
                policy::ControllerTerminalReason::kConverged);
  static_assert(policy::completed_round_terminal_reason(
                    0, 1, true, true, 10, 10, 4, 4) ==
                policy::ControllerTerminalReason::kTargetCertified);
  static_assert(policy::completed_round_terminal_reason(
                    0, 1, false, true, 10, 10, 4, 4) ==
                policy::ControllerTerminalReason::kMaxIterations);

  for (const std::uint32_t invalid : {0u, 3u, 7u, 32u}) {
    require_invalid_argument(
        [invalid] { policy::validate_segment_rounds(invalid); },
        "unsupported BF11 segment length was accepted");
  }

  // The third round supplies the exact target certificate. The first two
  // observations deliberately fail it even though all targets are finite in
  // round two (min next label remains below the maximum target label).
  const std::vector<policy::SegmentedRoundObservation> trace = {
      {3, 0, 0, 4, 0},
      {2, 0, 2, 4, 5},
      {4, 0, 2, 7, 5},
  };
  policy::SegmentedControllerSequentialModel control(controller_config(), 1);
  const policy::SegmentedControllerModelResult baseline =
      control.run_to_completion(trace);
  require(baseline.descriptor.terminal_reason ==
              policy::ControllerTerminalReason::kTargetCertified &&
              baseline.descriptor.iterations_used == 3 &&
              baseline.descriptor.actual_rounds == 3 &&
              baseline.descriptor.target_checks == 3 &&
              baseline.segments == 3 && baseline.status_copies == 3,
          "K=1 controller did not establish the expected control result");

  for (const std::uint32_t rounds : {1u, 2u, 4u, 8u, 16u}) {
    policy::SegmentedControllerSequentialModel model(controller_config(),
                                                     rounds);
    const policy::SegmentedControllerModelResult result =
        model.run_to_completion(trace);
    const std::uint64_t expected_segments = (3 + rounds - 1) / rounds;
    const std::uint32_t expected_no_ops =
        static_cast<std::uint32_t>(expected_segments * rounds - 3);
    require(policy::controller_semantically_equivalent(
                baseline.descriptor, result.descriptor),
            "segmented controller diverged from K=1 semantics");
    require(result.segments == expected_segments &&
                result.status_copies == expected_segments &&
                result.observations_consumed == 3 &&
                result.descriptor.no_op_rounds == expected_no_ops,
            "segmented controller publication/no-op accounting is wrong");
  }
}

void test_early_completion_and_terminal_no_ops() {
  const std::vector<policy::SegmentedRoundObservation> trace = {
      continuing_round(3),
      {0, 0, 2, 10, 10},
      // This must not be consumed after convergence.
      {9, 77, 0, 0, 0},
  };
  policy::SegmentedControllerSequentialModel model(controller_config(), 8);
  policy::SegmentedControllerModelResult result =
      model.run_to_completion(trace);
  require(result.descriptor.terminal_reason ==
              policy::ControllerTerminalReason::kConverged &&
              result.descriptor.iterations_used == 2 &&
              result.descriptor.actual_rounds == 2 &&
              result.descriptor.no_op_rounds == 6 &&
              result.observations_consumed == 2 && result.segments == 1,
          "early convergence did not turn the rest of the segment into no-ops");

  const policy::SegmentedControllerDescriptor before = result.descriptor;
  const policy::ControllerRoundAdvance no_op =
      policy::finalize_controller_round(&result.descriptor, trace.back());
  require(!no_op.executed &&
              result.descriptor.terminal_reason == before.terminal_reason &&
              result.descriptor.error_status == before.error_status &&
              result.descriptor.iterations_used == before.iterations_used &&
              result.descriptor.no_op_rounds == before.no_op_rounds + 1,
          "a scheduled no-op mutated sticky terminal controller state");
}

void test_max_iteration_accounting() {
  policy::SegmentedControllerConfig zero_config = controller_config(0);
  zero_config.first_mark_token = 0;
  policy::SegmentedControllerSequentialModel zero_model(zero_config, 4);
  require(zero_model.descriptor().terminal_reason ==
              policy::ControllerTerminalReason::kMaxIterations &&
              zero_model.descriptor().iterations_used == 0 &&
              zero_model.descriptor().actual_rounds == 0,
          "max_iters=0 executed a relaxation round");
  std::size_t cursor = 0;
  zero_model.run_one_segment({}, &cursor);
  require(zero_model.descriptor().iterations_used == 0 &&
              zero_model.descriptor().actual_rounds == 0 &&
              zero_model.descriptor().no_op_rounds == 4 && cursor == 0,
          "a pre-finished max_iters=0 segment was not entirely no-op");

  const std::vector<policy::SegmentedRoundObservation> trace(
      8, continuing_round());
  for (const std::uint32_t limit : {4u, 5u, 8u}) {
    policy::SegmentedControllerConfig config = controller_config(limit, 2);
    config.first_mark_token = 10;
    policy::SegmentedControllerSequentialModel model(config, 4);
    const policy::SegmentedControllerModelResult result =
        model.run_to_completion(trace);
    const std::uint64_t expected_segments = (limit + 3) / 4;
    require(result.descriptor.terminal_reason ==
                policy::ControllerTerminalReason::kMaxIterations &&
                result.descriptor.iterations_used == limit &&
                result.descriptor.actual_rounds == limit &&
                result.descriptor.no_op_rounds ==
                    expected_segments * 4 - limit &&
                result.descriptor.target_checks == (limit + 1) / 2 &&
                result.segments == expected_segments &&
                result.status_copies == expected_segments,
            "max-iteration accounting changed at a segment boundary");
    require(result.descriptor.mark_token == 10 + limit - 1,
            "mark token did not advance exactly once per continuing round");
  }
}

void test_error_precedence_and_target_interval() {
  std::vector<policy::SegmentedRoundObservation> error_trace = {
      continuing_round(),
      continuing_round(),
      {/*next_count=*/0,
       /*error_status=*/77,
       // Also corrupt the target count: the explicit controller error must
       // retain stop precedence over every derived status check.
       /*reached_target_count=*/3,
       /*min_next_frontier_dist_bits=*/100,
       /*max_target_dist_bits=*/1},
  };
  policy::SegmentedControllerSequentialModel error_model(controller_config(),
                                                         8);
  const policy::SegmentedControllerModelResult error_result =
      error_model.run_to_completion(error_trace);
  require(error_result.descriptor.terminal_reason ==
              policy::ControllerTerminalReason::kError &&
              error_result.descriptor.error_status == 77 &&
              error_result.descriptor.iterations_used == 3 &&
              error_result.descriptor.actual_rounds == 3 &&
              error_result.descriptor.no_op_rounds == 5,
          "an in-segment controller error was hidden by "
          "convergence/certification");

  // Qualifying target evidence is present every round, but interval three
  // permits the exact certificate only after the third completed relaxation.
  const std::vector<policy::SegmentedRoundObservation> target_trace(3, {
      /*next_count=*/2,
      /*error_status=*/0,
      /*reached_target_count=*/2,
      /*min_next_frontier_dist_bits=*/50,
      /*max_target_dist_bits=*/50,
  });
  policy::SegmentedControllerSequentialModel interval_model(
      controller_config(10, 3), 4);
  const policy::SegmentedControllerModelResult interval_result =
      interval_model.run_to_completion(target_trace);
  require(interval_result.descriptor.terminal_reason ==
              policy::ControllerTerminalReason::kTargetCertified &&
              interval_result.descriptor.iterations_used == 3 &&
              interval_result.descriptor.target_checks == 1 &&
              interval_result.descriptor.no_op_rounds == 1,
          "target_check_interval did not delay only the target certificate");

  // A final reachability scan is needed to choose bounded fallback without
  // extracting the failed attempt. It must not become an out-of-interval
  // target certificate.
  policy::SegmentedControllerSequentialModel final_scan_model(
      controller_config(2, 3), 4);
  const policy::SegmentedControllerModelResult final_scan =
      final_scan_model.run_to_completion(
          std::vector<policy::SegmentedRoundObservation>(2, {
              /*next_count=*/2,
              /*error_status=*/0,
              /*reached_target_count=*/2,
              /*min_next_frontier_dist_bits=*/50,
              /*max_target_dist_bits=*/50,
          }));
  require(final_scan.descriptor.terminal_reason ==
              policy::ControllerTerminalReason::kMaxIterations &&
              final_scan.descriptor.target_checks == 1 &&
              final_scan.descriptor.all_targets_reached == 1 &&
              final_scan.descriptor.iterations_used == 2,
          "final reachability scan bypassed target_check_interval semantics");

  // Empty frontier wins over an otherwise simultaneous target certificate,
  // matching the established BF11 stop precedence.
  policy::SegmentedControllerDescriptor convergence =
      policy::initialize_segmented_controller(controller_config(1, 1));
  (void)policy::finalize_controller_round(
      &convergence, {/*next_count=*/0, 0, 2, 50, 50});
  require(convergence.terminal_reason ==
              policy::ControllerTerminalReason::kConverged,
          "target certification incorrectly dominated convergence");
}

void test_mark_token_reservation_and_forced_wrap() {
  static_assert(policy::mark_token_for_round(
                    policy::reserve_mark_tokens(7, 3, 15), 0) == 7);
  static_assert(policy::mark_token_for_round(
                    policy::reserve_mark_tokens(7, 3, 15), 2) == 9);
  static_assert(policy::mark_token_for_round(
                    policy::reserve_mark_tokens(7, 3, 15), 3) == 0);

  policy::MarkTokenAllocatorModel allocator(/*max_token=*/15);
  const policy::MarkTokenReservation first = allocator.reserve(4);
  const policy::MarkTokenReservation second = allocator.reserve(8);
  require(first.first_token == 1 && first.last_token == 4 &&
              first.next_token == 5 && !first.dense_reset_required &&
              second.first_token == 5 && second.last_token == 12 &&
              second.next_token == 13 && !second.dense_reset_required,
          "ordinary mark-token reservations were not monotonic/disjoint");

  const policy::MarkTokenReservation wrapped = allocator.reserve(4);
  require(wrapped.dense_reset_required && wrapped.first_token == 1 &&
              wrapped.last_token == 4 && wrapped.next_token == 5 &&
              allocator.dense_reset_count() == 1,
          "small test generation did not force a pre-wrap dense reset");

  policy::MarkTokenAllocatorModel exact_end(/*max_token=*/7,
                                            /*next_token=*/5);
  const policy::MarkTokenReservation final_range = exact_end.reserve(3);
  require(final_range.first_token == 5 && final_range.last_token == 7 &&
              final_range.next_token == 0 &&
              !final_range.dense_reset_required,
          "the final safe token range was discarded prematurely");
  const policy::MarkTokenReservation empty = exact_end.reserve(0);
  require(empty.token_count == 0 && empty.next_token == 0 &&
              !empty.dense_reset_required &&
              exact_end.dense_reset_count() == 0,
          "an empty max_iters=0 reservation reset an exhausted mark array");
  const policy::MarkTokenReservation restarted = exact_end.reserve(1);
  require(restarted.first_token == 1 && restarted.last_token == 1 &&
              restarted.dense_reset_required &&
              exact_end.dense_reset_count() == 1,
          "the reservation after token exhaustion wrapped without reset");
  require_overflow(
      [] {
        policy::MarkTokenAllocatorModel small(3);
        (void)small.reserve(4);
      },
      "a query larger than the full mark-token range was accepted");

  policy::SegmentedControllerConfig invalid = controller_config(2);
  invalid.first_mark_token = 0;
  require(policy::initialize_segmented_controller(invalid).terminal_reason ==
              policy::ControllerTerminalReason::kInvalidConfiguration,
          "controller accepted a zero live mark token");
}

void test_adaptive_reset_selection() {
  static_assert(policy::scaled_ceiling(100, 1, 4) == 25);
  static_assert(policy::scaled_ceiling(10, 1, 3) == 4);
  static_assert(policy::scaled_ceiling(
                    std::numeric_limits<std::uint64_t>::max(), 1, 4) ==
                4'611'686'018'427'387'904ULL);

  const policy::ResetDecision empty =
      policy::select_reset_mode({0, 100, false, false});
  const policy::ResetDecision sparse =
      policy::select_reset_mode({24, 100, false, false});
  const policy::ResetDecision dense =
      policy::select_reset_mode({25, 100, false, false});
  require(empty.mode == policy::ResetMode::kSparse &&
              empty.reason == policy::ResetReason::kEmptyTouchedSet &&
              sparse.mode == policy::ResetMode::kSparse &&
              sparse.reason == policy::ResetReason::kBelowDensityThreshold &&
              dense.mode == policy::ResetMode::kDense &&
              dense.reason ==
                  policy::ResetReason::kAtOrAboveDensityThreshold &&
              dense.dense_threshold_count == 25,
          "default one-quarter sparse/dense reset threshold is wrong");

  const policy::AdaptiveResetPolicy thirds{/*numerator=*/1,
                                            /*denominator=*/3};
  require(policy::select_reset_mode({3, 10, false, false}, thirds).mode ==
              policy::ResetMode::kSparse &&
              policy::select_reset_mode({4, 10, false, false}, thirds).mode ==
                  policy::ResetMode::kDense,
          "tunable reset threshold did not use a conservative ceiling");
  require(policy::select_reset_mode({1, 100, true, false}).reason ==
              policy::ResetReason::kDefensiveFailureRecovery &&
              policy::select_reset_mode({1, 100, false, true}).reason ==
                  policy::ResetReason::kMarkGenerationWrap &&
              policy::select_reset_mode({101, 100, false, false}).reason ==
                  policy::ResetReason::kInvalidTouchedCount,
          "mandatory dense reset reasons were not preserved");

  const policy::AdaptiveResetPolicy invalid{/*numerator=*/2,
                                             /*denominator=*/1};
  require(policy::select_reset_mode({1, 10, false, false}, invalid).reason ==
              policy::ResetReason::kInvalidPolicy,
          "invalid device-side reset policy did not fail dense");
  require_invalid_argument(
      [invalid] { policy::validate_reset_policy(invalid); },
      "invalid host reset threshold was accepted");
}

void test_cost_modes_and_lazy_dynamic_transitions() {
  policy::CostPolicyState state{/*static_costs_constant_one=*/true,
                                /*identity_proven=*/true,
                                /*dynamic_storage_allocated=*/false,
                                /*dynamic_storage_contains_identity=*/false};
  require(policy::cost_mode(state) == policy::CostMode::kConstantOne,
          "constant-one graph did not start in its specialized mode");

  policy::CostUpdatePlan plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kSparse, 3, true});
  require(plan.valid && plan.skip_device_write &&
              !plan.allocate_dynamic_storage &&
              policy::cost_mode(plan.after) == policy::CostMode::kConstantOne,
          "identity sparse update allocated a dynamic array");
  state = plan.after;

  plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kSparse, 1, false});
  require(plan.allocate_dynamic_storage &&
              plan.initialize_dynamic_storage_to_identity &&
              plan.apply_sparse_update &&
              plan.mode_after == policy::CostMode::kDynamic &&
              plan.after.dynamic_storage_allocated &&
              !plan.after.identity_proven,
          "first sparse nonidentity update was not lazily initialized");
  state = plan.after;

  plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kSparse, 1, true});
  require(!plan.allocate_dynamic_storage && plan.apply_sparse_update &&
              plan.mode_after == policy::CostMode::kDynamic &&
              !plan.after.identity_proven,
          "sparse all-one update incorrectly re-proved global identity");
  state = plan.after;

  plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kCompleteReplacement, 100, true});
  require(plan.skip_device_write && plan.after.identity_proven &&
              plan.after.dynamic_storage_allocated &&
              plan.mode_after == policy::CostMode::kConstantOne,
          "complete all-one replacement did not re-enter identity mode");
  state = plan.after;

  plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kSparse, 2, false});
  require(!plan.allocate_dynamic_storage &&
              plan.initialize_dynamic_storage_to_identity &&
              plan.apply_sparse_update &&
              plan.mode_after == policy::CostMode::kDynamic,
          "sparse departure from re-entered identity reused stale storage");
  state = plan.after;

  plan = policy::plan_cost_update(
      state, {policy::CostUpdateKind::kCompleteReplacement, 100, false});
  require(!plan.allocate_dynamic_storage &&
              plan.upload_complete_replacement &&
              !plan.initialize_dynamic_storage_to_identity &&
              plan.mode_after == policy::CostMode::kDynamic,
          "complete dynamic replacement performed needless identity fill");

  const policy::CostPolicyState general_static{
      /*static_costs_constant_one=*/false,
      /*identity_proven=*/true,
      /*dynamic_storage_allocated=*/false,
      /*dynamic_storage_contains_identity=*/false};
  require(policy::cost_mode(general_static) == policy::CostMode::kStatic,
          "general static graph selected the wrong cost specialization");
  const policy::CostUpdatePlan full_ones = policy::plan_cost_update(
      general_static,
      {policy::CostUpdateKind::kCompleteReplacement, 100, true});
  require(full_ones.skip_device_write &&
              full_ones.mode_after == policy::CostMode::kStatic &&
              !full_ones.after.dynamic_storage_allocated,
          "complete all-one update allocated storage for general static mode");

  const policy::CostPolicyState invalid{
      /*static_costs_constant_one=*/false,
      /*identity_proven=*/false,
      /*dynamic_storage_allocated=*/false,
      /*dynamic_storage_contains_identity=*/false};
  require(!policy::plan_cost_update(
               invalid, {policy::CostUpdateKind::kSparse, 1, false})
               .valid,
          "impossible dynamic-without-storage state was accepted");
  require(!policy::plan_cost_update(
               general_static,
               {static_cast<policy::CostUpdateKind>(99), 1, false})
               .valid,
          "unknown cost-update kind was accepted");
}

bool offsets_equal(const std::vector<policy::TargetPrefixOffsets>& lhs,
                   const std::vector<policy::TargetPrefixOffsets>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].node_offset != rhs[i].node_offset ||
        lhs[i].edge_offset != rhs[i].edge_offset) {
      return false;
    }
  }
  return true;
}

void test_prefix_scan_overflow_and_grow_retry() {
  const std::vector<policy::TargetPathCardinality> targets = {
      {3, 2, policy::TargetCardinalityStatus::kValid},
      {0, 0, policy::TargetCardinalityStatus::kUnreachable},
      {2, 1, policy::TargetCardinalityStatus::kValid},
      // Duplicate target summaries retain deterministic target order.
      {3, 2, policy::TargetCardinalityStatus::kValid},
  };
  const policy::PrefixScanResult overflow =
      policy::scan_target_prefixes(targets, 4, 2);
  require(overflow.header.status == policy::PrefixScanStatus::kNeedsGrowth &&
              overflow.header.total_nodes == 8 &&
              overflow.header.total_edges == 5 &&
              overflow.header.required_node_capacity == 8 &&
              overflow.header.required_edge_capacity == 5 &&
              overflow.offsets[0].node_offset == 0 &&
              overflow.offsets[1].node_offset == 3 &&
              overflow.offsets[2].node_offset == 3 &&
              overflow.offsets[3].node_offset == 5 &&
              overflow.offsets[3].edge_offset == 3,
          "deterministic prefix scan did not return exact required capacity");

  const policy::CapacityGrowth grown_nodes =
      policy::grow_retained_capacity(4, overflow.header.total_nodes);
  const policy::CapacityGrowth grown_edges =
      policy::grow_retained_capacity(2, overflow.header.total_edges);
  require(grown_nodes.valid && grown_nodes.grew &&
              grown_nodes.capacity == 8 && grown_edges.valid &&
              grown_edges.grew && grown_edges.capacity == 8,
          "compact arena did not grow geometrically");
  const policy::PrefixScanResult retry = policy::scan_target_prefixes(
      targets, grown_nodes.capacity, grown_edges.capacity);
  require(retry.header.status == policy::PrefixScanStatus::kReady &&
              offsets_equal(overflow.offsets, retry.offsets),
          "extraction-only grow/retry changed deterministic prefix offsets");

  const policy::CapacityGrowth exact_growth =
      policy::grow_retained_capacity(8, 9, 10);
  require(exact_growth.valid && exact_growth.capacity == 9 &&
              !policy::grow_retained_capacity(8, 11, 10).valid &&
              !policy::grow_retained_capacity(11, 9, 10).valid,
          "capacity growth did not respect the legal maximum");

  const policy::PrefixScanResult malformed = policy::scan_target_prefixes(
      {{2, 2, policy::TargetCardinalityStatus::kValid}}, 10, 10);
  require(malformed.header.status ==
              policy::PrefixScanStatus::kInvalidSummary &&
              malformed.header.failure_index == 0,
          "malformed node/edge cardinality was accepted");
  const policy::PrefixScanResult malformed_unreachable =
      policy::scan_target_prefixes(
          {{1, 0, policy::TargetCardinalityStatus::kUnreachable}}, 10, 10);
  require(malformed_unreachable.header.status ==
              policy::PrefixScanStatus::kInvalidSummary,
          "unreachable target with materialized nodes was accepted");

  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const policy::PrefixScanResult count_overflow =
      policy::scan_target_prefixes(
          {{maximum, maximum - 1,
            policy::TargetCardinalityStatus::kValid},
           {1, 0, policy::TargetCardinalityStatus::kValid}},
          maximum, maximum, maximum);
  require(count_overflow.header.status ==
              policy::PrefixScanStatus::kCountOverflow &&
              count_overflow.header.failure_index == 1,
          "uint64 compact prefix overflow was not detected");

  const policy::PrefixScanResult result_limit = policy::scan_target_prefixes(
      {{4, 3, policy::TargetCardinalityStatus::kValid},
       {2, 1, policy::TargetCardinalityStatus::kValid}},
      10, 10, /*result_item_limit=*/5);
  require(result_limit.header.status ==
              policy::PrefixScanStatus::kResultLimitExceeded &&
              result_limit.header.failure_index == 1,
          "host/result offset limit overflow was not detected");
}

void test_malformed_predecessor_chain_model() {
  using State = policy::PackedPredecessorModelState;
  using Edge = policy::PredecessorModelEdge;
  constexpr std::uint32_t one = 0x3f800000u;
  constexpr std::uint32_t two = 0x40000000u;

  const std::vector<State> valid_states = {
      {0u, policy::kNoPredecessorEdge}, {one, 0u}, {two, 1u}};
  const std::vector<Edge> valid_edges = {{0, 1}, {1, 2}};
  const policy::PredecessorChainModelResult valid =
      policy::validate_predecessor_chain_model(valid_states, valid_edges, 2);
  require(valid.status == policy::PredecessorChainStatus::kValid &&
              valid.root == 0 && valid.node_count == 3 &&
              valid.edge_count == 2,
          "valid predecessor chain failed the reconstruction model");

  std::vector<State> unreachable = valid_states;
  unreachable[2] = {};
  require(policy::validate_predecessor_chain_model(
              unreachable, valid_edges, 2).status ==
              policy::PredecessorChainStatus::kUnreachable,
          "infinite target was not reported unreachable");

  std::vector<State> nonzero_root = valid_states;
  nonzero_root[0].distance_bits = one;
  require(policy::validate_predecessor_chain_model(
              nonzero_root, valid_edges, 2).status ==
              policy::PredecessorChainStatus::kInvalid,
          "nonzero no-predecessor state was accepted as a source root");

  std::vector<Edge> wrong_destination = valid_edges;
  wrong_destination[1].destination = 1;
  require(policy::validate_predecessor_chain_model(
              valid_states, wrong_destination, 2).status ==
              policy::PredecessorChainStatus::kInvalid,
          "predecessor edge owned by another destination was accepted");

  std::vector<State> bad_edge = valid_states;
  bad_edge[2].predecessor_edge = 99;
  require(policy::validate_predecessor_chain_model(
              bad_edge, valid_edges, 2).status ==
              policy::PredecessorChainStatus::kInvalid,
          "out-of-range predecessor edge was accepted");

  const std::vector<State> cycle_states = {{one, 0u}, {two, 1u}};
  const std::vector<Edge> cycle_edges = {{1, 0}, {0, 1}};
  require(policy::validate_predecessor_chain_model(
              cycle_states, cycle_edges, 1).status ==
              policy::PredecessorChainStatus::kInvalid,
          "cyclic predecessor chain escaped the V-edge guard");
}

void test_bounded_fallback_extracts_only_accepted_attempt() {
  policy::FallbackExtractionState state =
      policy::initialize_fallback_extraction(true);
  policy::TraversalDecision decision = policy::record_traversal_attempt(
      &state,
      {/*bounded=*/true,
       /*completed=*/true,
       /*all_targets_reached=*/false,
       /*iterations_used=*/7},
      /*unbounded_fallback_enabled=*/true);
  require(decision == policy::TraversalDecision::kRetryUnbounded &&
              state.phase == policy::TraversalPhase::kAwaitingUnbounded &&
              state.traversal_attempts == 1 && state.extractions == 0 &&
              state.bounded_fallbacks == 1 &&
              state.avoided_failed_attempt_extractions == 1 &&
              !policy::record_extraction(&state),
          "bounded miss was extracted before its unbounded retry");

  decision = policy::record_traversal_attempt(
      &state,
      {/*bounded=*/false,
       /*completed=*/true,
       /*all_targets_reached=*/true,
       /*iterations_used=*/5},
      /*unbounded_fallback_enabled=*/true);
  require(decision == policy::TraversalDecision::kExtractAcceptedAttempt &&
              state.phase == policy::TraversalPhase::kExtractionPending &&
              state.accumulated_iterations == 12 &&
              state.traversal_attempts == 2 &&
              !state.accepted_attempt_was_bounded &&
              state.accepted_attempt_reached_all_targets,
          "unbounded retry was not selected as the accepted final attempt");
  require(policy::record_extraction(&state) &&
              !policy::record_extraction(&state) && state.extractions == 1 &&
              state.phase == policy::TraversalPhase::kFinished,
          "accepted traversal was extracted more than once");

  policy::FallbackExtractionState no_fallback{};
  decision = policy::record_traversal_attempt(
      &no_fallback, {true, true, false, 4},
      /*unbounded_fallback_enabled=*/false);
  require(decision == policy::TraversalDecision::kExtractAcceptedAttempt &&
              policy::record_extraction(&no_fallback) &&
              no_fallback.extractions == 1 &&
              no_fallback.bounded_fallbacks == 0,
          "disabled fallback did not extract the bounded unreachable result");

  policy::FallbackExtractionState failed{};
  decision = policy::record_traversal_attempt(
      &failed, {true, false, false, 2}, true);
  require(decision == policy::TraversalDecision::kPropagateError &&
              failed.phase == policy::TraversalPhase::kFailed &&
              failed.extractions == 0 && !policy::record_extraction(&failed),
          "failed traversal was materialized");
  require(policy::saturating_iteration_add(
              std::numeric_limits<std::int32_t>::max() - 2, 10) ==
              std::numeric_limits<std::int32_t>::max(),
          "fallback iteration accumulation did not saturate safely");

  policy::FallbackExtractionState initially_unbounded =
      policy::initialize_fallback_extraction(false);
  require(policy::record_traversal_attempt(
              &initially_unbounded, {false, true, true, 3}, true) ==
              policy::TraversalDecision::kExtractAcceptedAttempt &&
              initially_unbounded.traversal_attempts == 1 &&
              initially_unbounded.bounded_fallbacks == 0,
          "an initially unbounded query used the bounded-attempt state");
}

}  // namespace

int main() {
  try {
    test_segment_validation_and_equivalence();
    test_early_completion_and_terminal_no_ops();
    test_max_iteration_accounting();
    test_error_precedence_and_target_interval();
    test_mark_token_reservation_and_forced_wrap();
    test_adaptive_reset_selection();
    test_cost_modes_and_lazy_dynamic_transitions();
    test_prefix_scan_overflow_and_grow_retry();
    test_malformed_predecessor_chain_model();
    test_bounded_fallback_extracts_only_accepted_attempt();
    std::cout << "BF11 execution policy tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "BF11 execution policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
