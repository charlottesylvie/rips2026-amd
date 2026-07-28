#include "preds_gpu_engine.hpp"

#include <string>

int main() {
  using rips_predicates_gpu::PredicateMode;
  using rips_predicates_gpu::TerminationReason;

  const PredicateMode mode =
      rips_predicates_gpu::parse_predicate_mode("OUT_STATIC");
  if (mode != PredicateMode::kOutStatic) {
    return 1;
  }
  if (std::string(rips_predicates_gpu::predicate_mode_name(mode)) !=
      "OUT_STATIC") {
    return 2;
  }
  if (std::string(rips_predicates_gpu::termination_reason_name(
          TerminationReason::kAllTargetsSettled)) !=
      "all_targets_settled") {
    return 3;
  }
  return 0;
}
