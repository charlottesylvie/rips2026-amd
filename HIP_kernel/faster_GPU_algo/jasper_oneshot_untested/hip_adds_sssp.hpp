#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace hip_adds {

// ADDS uses the paper's fixed 32-bucket circular priority queue and
// 64K-word bucket pages. SegmentWords is 64 to match AMD wavefront width.
static constexpr uint32_t kBucketCount = 32u;
static constexpr uint32_t kPageWords = 1u << 16;     // 64K 32-bit words.
static constexpr uint32_t kSegmentWords = 64u;       // One ROCm/AMD wavefront.
static constexpr uint32_t kThreadsPerBlock = 256u;   // Four wavefronts.

// CSR graph view. The arrays must already live in device memory.
// Edge weights must be finite and non-negative. Vertex IDs are uint32_t.
template <typename Weight>
struct DeviceCsrGraph {
  uint32_t num_vertices = 0;
  uint32_t num_edges = 0;
  const uint32_t* row_offsets = nullptr;  // length num_vertices + 1
  const uint32_t* col_indices = nullptr;  // length num_edges
  const Weight* weights = nullptr;        // length num_edges
};

struct Options {
  // 0 means auto-select a persistent grid that can reside concurrently.
  uint32_t worker_blocks = 0;

  // Consecutive work items assigned by the manager to one worker block.
  // Larger chunks reduce manager traffic; smaller chunks improve load balance.
  uint32_t assignment_chunk = 256;

  // The manager can allocate work from multiple high-priority buckets. It starts
  // here and changes this value at runtime when dynamic_active_buckets is true.
  uint32_t initial_active_buckets = 1;
  uint32_t max_active_buckets = kBucketCount;

  // If initial_delta <= 0, a profiling kernel estimates the Near-Far heuristic
  // delta = initial_delta_scale * average_weight / average_degree, then ADDS
  // adjusts delta dynamically during the run.
  double initial_delta = 0.0;
  double initial_delta_scale = 16.0;
  double min_delta = 1.0e-3;

  // Feedback-loop parameters for dynamic delta selection.
  double delta_growth = 1.5;
  double delta_shrink = 0.75;
  double clip_threshold = 0.65;      // Tail-bucket clipping proxy threshold.
  double utilization_low = 0.35;     // Fraction of worker_blocks*chunk.
  double utilization_high = 0.85;    // Fraction of worker_blocks*chunk.
  uint32_t adjust_period_head_switches = 8;

  bool dynamic_delta = true;
  bool dynamic_active_buckets = true;

  // Worklist pool size in 32-bit words. 0 means choose a conservative default
  // from graph size and currently free device memory.
  size_t pool_words = 0;
  double memory_fraction = 0.45;

  // If true, distances are initialized to infinity and source is set to zero.
  // If false, only source is set to zero; existing distances are otherwise kept.
  bool initialize_distances = true;
};

struct RunStats {
  double initial_delta = 0.0;
  double final_delta = 0.0;
  double final_bucket_base = 0.0;

  uint32_t worker_blocks = 0;
  uint32_t final_active_buckets = 0;
  uint32_t pool_pages = 0;
  uint32_t device_error = 0;

  uint64_t assigned_items = 0;
  uint64_t processed_items = 0;
  uint64_t enqueued_items = 0;
  uint64_t edge_relaxations = 0;
  uint64_t successful_relaxations = 0;
  uint64_t head_switches = 0;
  uint64_t delta_updates = 0;
  uint64_t max_inflight_items = 0;
};

enum class Status : int {
  Success = 0,
  InvalidArgument,
  HipError,
  OutOfMemory,
  PoolExhausted,
  BucketOverflow,
  DeviceError
};

const char* status_string(Status status);

Status sssp(const DeviceCsrGraph<float>& graph,
            uint32_t source,
            float* d_distances,
            RunStats* stats = nullptr,
            const Options& options = Options{},
            hipStream_t stream = nullptr);

Status sssp(const DeviceCsrGraph<uint32_t>& graph,
            uint32_t source,
            uint32_t* d_distances,
            RunStats* stats = nullptr,
            const Options& options = Options{},
            hipStream_t stream = nullptr);

}  // namespace hip_adds
