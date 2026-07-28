#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rips_predicates_gpu {

using Offset = std::int64_t;
using Index = std::int32_t;

enum class PredicateMode : int {
  kInSimple = 0,
  kOutSimple = 1,
  kInSimpleOrOutSimple = 2,
  kInStatic = 3,
  kOutStatic = 4,
  kInStaticOrOutStatic = 5,
};

const char* predicate_mode_name(PredicateMode mode);
PredicateMode parse_predicate_mode(const std::string& text);

// Owning host representation of an outgoing RIPS CSR graph.  Moving this
// object into PredsGpuContext avoids a second host copy.
struct CsrGraph {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> colind;
  std::vector<float> values;
};

// Non-owning input accepted for callers whose CSR lives in another compatible
// host type.  The context copies this view once so its path-reconstruction
// storage cannot dangle.
struct CsrGraphView {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const Offset* rowptr = nullptr;
  const Index* colind = nullptr;
  const float* values = nullptr;
};

enum class TerminationReason {
  kFullConvergence,
  kAllTargetsSettled,
};

const char* termination_reason_name(TerminationReason reason);

struct AlgorithmStatistics {
  std::uint64_t phases = 0;
  std::uint64_t vertices_settled = 0;
  std::uint64_t vertices_settled_current_phase = 0;
  std::vector<std::uint32_t> vertices_settled_per_phase;
  std::uint64_t edges_examined = 0;
  std::uint64_t relaxations_attempted = 0;
  std::uint64_t successful_distance_updates = 0;
  double predicate_evaluation_ms = 0.0;
  double relaxation_ms = 0.0;
  double reset_ms = 0.0;
  double elapsed_ms = 0.0;
  std::uint64_t reset_kernel_launches = 0;
  std::uint64_t tiny_relaxation_phases = 0;
  std::uint64_t large_relaxation_phases = 0;
};

struct TransferStatistics {
  double h2d_ms = 0.0;
  double d2h_ms = 0.0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t h2d_operations = 0;
  std::uint64_t d2h_operations = 0;

  double total_ms() const {
    return h2d_ms + d2h_ms;
  }
};

struct LaunchCounters {
  std::uint64_t sssp = 0;
  std::uint64_t output_reconstruction = 0;
};

struct RunTelemetry {
  AlgorithmStatistics algorithm;
  TransferStatistics transfers;
  LaunchCounters launches;
  double runtime_ms = 0.0;
};

struct TargetResult {
  Index target = -1;
  float distance = 0.0f;
  bool reached = false;
  bool path_captured = false;
  std::vector<Index> nodes;
  std::vector<Offset> csr_edges;
};

// Non-owning view used to validate an adapter's captured result against the
// immutable CSR retained by PredsGpuContext.
struct CapturedTargetPathView {
  Index target = -1;
  float distance = 0.0f;
  bool reached = false;
  bool path_captured = false;
  const Index* nodes = nullptr;
  std::size_t node_count = 0;
  const Offset* csr_edges = nullptr;
  std::size_t csr_edge_count = 0;
};

struct RunRequest {
  Index source = -1;
  // The context stable-deduplicates this list defensively while preserving the
  // first occurrence.  all_SSSP normally supplies an already unique list.
  std::vector<Index> unique_targets;
  bool early_stop = false;
  bool capture_paths = false;
  // The standalone CLI prints this graph-sized histogram.  Reusable callers
  // leave it false so no graph-sized host allocation occurs per source.
  bool capture_phase_histogram = false;
  // Used by the legacy standalone CLI.  Zero disables the phase snapshot.
  std::uint64_t statistics_snapshot_phase = 0;
};

struct RunResult {
  TerminationReason termination = TerminationReason::kFullConvergence;
  bool fully_converged = false;
  bool all_targets_confirmed = false;
  std::uint64_t phases_or_iterations = 0;
  std::vector<TargetResult> targets;
  RunTelemetry telemetry;

  bool requested_snapshot_reached = false;
  RunTelemetry requested_snapshot;
};

struct CumulativeStatistics {
  std::uint64_t runs = 0;
  std::uint64_t fully_converged_runs = 0;
  std::uint64_t target_stopped_runs = 0;
  AlgorithmStatistics algorithm;
  TransferStatistics graph_setup_transfers;
  TransferStatistics run_transfers;
  LaunchCounters launches;
  double preprocessing_ms = 0.0;
  double runtime_ms = 0.0;
  std::size_t allocated_device_bytes = 0;
};

// One immutable graph upload plus one sequentially reused SSSP workspace.
// max_unique_targets reserves every compact target buffer before the first
// query.  reserve_path_scratch retains graph-sized pinned host scratch for the
// optional predecessor fallback; no graph-sized allocation occurs in run().
// The context is bound to the HIP device active during construction.  Keep
// that device active for run() and destruction; run() detects a mismatch.
class PredsGpuContext {
 public:
  PredsGpuContext(CsrGraph graph,
                  PredicateMode mode,
                  std::size_t max_unique_targets,
                  bool reserve_path_scratch,
                  bool print_setup_metadata = false);
  PredsGpuContext(CsrGraphView graph,
                  PredicateMode mode,
                  std::size_t max_unique_targets,
                  bool reserve_path_scratch,
                  bool print_setup_metadata = false);
  ~PredsGpuContext();

  PredsGpuContext(const PredsGpuContext&) = delete;
  PredsGpuContext& operator=(const PredsGpuContext&) = delete;
  PredsGpuContext(PredsGpuContext&&) noexcept;
  PredsGpuContext& operator=(PredsGpuContext&&) noexcept;

  PredicateMode predicate_mode() const noexcept;
  Offset rows() const noexcept;
  Offset nnz() const noexcept;
  std::size_t target_capacity() const noexcept;
  bool path_scratch_reserved() const noexcept;

  RunResult run(const RunRequest& request);
  void validate_captured_target_path(
      Index source,
      const CapturedTargetPathView& path) const;
  CumulativeStatistics cumulative_statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rips_predicates_gpu
