#include "hip_adds_sssp.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace hip_adds {
namespace detail {

static_assert(kBucketCount == 32u, "ADDS implementation assumes 32 buckets");
static_assert(kPageWords == (1u << 16), "ADDS page size must be 64K words");
static_assert((kSegmentWords & (kSegmentWords - 1u)) == 0u,
              "SegmentWords must be a power of two");
static_assert((kPageWords % kSegmentWords) == 0u,
              "PageWords must be a multiple of SegmentWords");

constexpr uint32_t kPageShift = 16u;
constexpr uint32_t kPageMask = kPageWords - 1u;
constexpr uint32_t kSegmentsPerPage = kPageWords / kSegmentWords;
constexpr uint32_t kInvalidPage = 0xffffffffu;
constexpr uint32_t kOpen = 0u;
constexpr uint32_t kRetiring = 1u;
constexpr uint32_t kAssignmentIdle = 0u;
constexpr uint32_t kAssignmentReady = 1u;
constexpr uint32_t kAssignmentStop = 2u;
constexpr uint32_t kCacheEntries = 128u;
constexpr uint32_t kHighVertexCapacity = 256u;
constexpr uint32_t kCoopDegreeThreshold = 1024u;

constexpr uint32_t kDeviceSuccess = 0u;
constexpr uint32_t kDevicePoolExhausted = 1u;
constexpr uint32_t kDeviceBucketOverflow = 2u;
constexpr uint32_t kDeviceInvalidVertex = 3u;

inline double clamp_double(double x, double lo, double hi) {
  return std::max(lo, std::min(x, hi));
}

inline bool hip_ok(hipError_t err) { return err == hipSuccess; }

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() { reset(); }

  Status allocate(size_t count) {
    reset();
    count_ = count;
    if (count == 0) return Status::Success;
    hipError_t err = hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
    if (err == hipErrorOutOfMemory) return Status::OutOfMemory;
    if (err != hipSuccess) return Status::HipError;
    return Status::Success;
  }

  void reset() {
    if (ptr_ != nullptr) {
      hipFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* get() const { return ptr_; }
  size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

struct alignas(32) AssignmentSlot {
  uint32_t state;
  uint32_t bucket;
  uint32_t generation;
  uint32_t start;
  uint32_t count;
  uint32_t logical_offset;
  uint32_t reserved0;
  uint32_t reserved1;
};

struct DeviceControl {
  uint32_t terminate;
  uint32_t error;

  uint32_t worker_blocks;
  uint32_t pool_pages;
  uint32_t max_pages_per_bucket;
  uint32_t assignment_chunk;

  uint32_t head;
  uint32_t epoch;
  uint32_t active_buckets;
  uint32_t max_active_buckets;
  uint32_t dynamic_delta;
  uint32_t dynamic_active_buckets;
  uint32_t adjust_period_head_switches;
  uint32_t head_switches_since_adjust;
  uint32_t quiet_passes;

  float bucket_base;
  float delta;
  float inv_delta;
  float min_delta;
  float delta_growth;
  float delta_shrink;
  float delta_lower_bound;
  float clip_threshold;
  float utilization_low;
  float utilization_high;

  uint64_t inflight_items;
  uint64_t max_inflight_items;
  uint64_t assigned_items_total;
  uint64_t processed_items_total;
  uint64_t enqueued_items_total;
  uint64_t edge_relaxations_total;
  uint64_t successful_relaxations_total;
  uint64_t head_switches_total;
  uint64_t delta_updates;

  uint64_t period_enqueued_total;
  uint64_t period_tail_enqueued;
  uint64_t period_util_accum;
  uint64_t period_util_samples;
};

struct DeviceState {
  uint32_t* pool;
  uint32_t* free_stack;
  uint32_t* free_top;

  uint32_t* page_table;       // kBucketCount * max_pages_per_bucket entries.
  uint32_t* page_count;       // kBucketCount.
  uint32_t* resv_ptr;         // kBucketCount.
  uint32_t* read_ptr;         // kBucketCount.
  uint32_t* completed_count;  // kBucketCount; CWC in the paper.
  uint32_t* bucket_state;     // Open/retiring guard.
  uint32_t* bucket_generation;
  uint32_t* writer_count;

  uint32_t* wcc;              // pool_pages * kSegmentsPerPage.
  AssignmentSlot* assignments;
  DeviceControl* control;
};

__device__ inline unsigned long long atomic_add_u64(uint64_t* ptr, uint64_t val) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(ptr),
                   static_cast<unsigned long long>(val));
}

__device__ inline unsigned long long atomic_max_u64(uint64_t* ptr, uint64_t val) {
  return atomicMax(reinterpret_cast<unsigned long long*>(ptr),
                   static_cast<unsigned long long>(val));
}

__device__ inline void device_set_error(DeviceControl* control, uint32_t code) {
  atomicCAS(&control->error, kDeviceSuccess, code);
  __threadfence();
  control->terminate = 1u;
}

template <typename Weight>
struct DistanceOps;

template <>
struct DistanceOps<float> {
  __device__ static float zero() { return 0.0f; }
  __device__ static float infinity() { return __uint_as_float(0x7f800000u); }
  __device__ static bool less(float a, float b) { return a < b; }
  __device__ static float add(float a, float b) { return a + b; }
  __device__ static float to_bucket_float(float x) { return x; }

  __device__ static float atomic_min(float* address, float value) {
    // ADDS assumes non-negative distances; this CAS loop matches the common
    // Gunrock-style software atomicMin used when hardware float atomicMin is
    // unavailable.
    uint32_t* address_as_u = reinterpret_cast<uint32_t*>(address);
    uint32_t old = __float_as_uint(*address);
    while (value < __uint_as_float(old)) {
      const uint32_t assumed = old;
      old = atomicCAS(address_as_u, assumed, __float_as_uint(value));
      if (old == assumed) break;
    }
    return __uint_as_float(old);
  }
};

template <>
struct DistanceOps<uint32_t> {
  __device__ static uint32_t zero() { return 0u; }
  __device__ static uint32_t infinity() { return 0xffffffffu; }
  __device__ static bool less(uint32_t a, uint32_t b) { return a < b; }
  __device__ static uint32_t add(uint32_t a, uint32_t b) {
    const uint32_t out = a + b;
    return (out < a) ? 0xffffffffu : out;
  }
  __device__ static float to_bucket_float(uint32_t x) { return static_cast<float>(x); }
  __device__ static uint32_t atomic_min(uint32_t* address, uint32_t value) {
    return atomicMin(address, value);
  }
};

__device__ inline uint32_t volatile_load_u32(const uint32_t* p) {
  return *reinterpret_cast<const volatile uint32_t*>(p);
}

__device__ inline uint64_t volatile_load_u64(const uint64_t* p) {
  return *reinterpret_cast<const volatile uint64_t*>(p);
}

__device__ inline uint32_t dev_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
__device__ inline uint32_t dev_max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }
__device__ inline uint64_t dev_max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }

__device__ inline uint32_t cache_slot(uint32_t generation, uint32_t key) {
  return (key ^ (generation * 2654435761u) ^ (key >> 7)) & (kCacheEntries - 1u);
}

__device__ inline uint32_t translate_page_cached(const DeviceState& state,
                                                 uint32_t bucket,
                                                 uint32_t generation,
                                                 uint32_t page_index,
                                                 uint32_t* cache_gen,
                                                 uint32_t* cache_key,
                                                 uint32_t* cache_page) {
  if (page_index >= state.control->max_pages_per_bucket) {
    device_set_error(state.control, kDeviceBucketOverflow);
    return kInvalidPage;
  }

  const uint32_t key = (bucket << 16) | (page_index & 0xffffu);
  const uint32_t slot = cache_slot(generation, key);
  if (cache_gen[slot] == generation && cache_key[slot] == key) {
    const uint32_t page = cache_page[slot];
    if (page != kInvalidPage) return page;
  }

  const uint64_t table_index = static_cast<uint64_t>(bucket) *
                               state.control->max_pages_per_bucket + page_index;
  uint32_t page = volatile_load_u32(&state.page_table[table_index]);
  while (page == kInvalidPage) {
    if (volatile_load_u32(&state.control->terminate) ||
        volatile_load_u32(&state.control->error)) {
      return kInvalidPage;
    }
    page = volatile_load_u32(&state.page_table[table_index]);
  }

  cache_page[slot] = page;
  __threadfence_block();
  cache_key[slot] = key;
  __threadfence_block();
  cache_gen[slot] = generation;
  return page;
}

__device__ inline uint32_t load_bucket_item(const DeviceState& state,
                                            uint32_t bucket,
                                            uint32_t generation,
                                            uint32_t index,
                                            uint32_t* cache_gen,
                                            uint32_t* cache_key,
                                            uint32_t* cache_page) {
  const uint32_t page_index = index >> kPageShift;
  const uint32_t offset = index & kPageMask;
  const uint32_t page = translate_page_cached(state, bucket, generation, page_index,
                                              cache_gen, cache_key, cache_page);
  if (page == kInvalidPage) return kInvalidPage;
  return state.pool[static_cast<uint64_t>(page) * kPageWords + offset];
}

template <typename Weight>
__device__ inline uint32_t bucket_offset_for_priority(const DeviceControl* control,
                                                      Weight priority) {
  const float d = DistanceOps<Weight>::to_bucket_float(priority);
  const float delta = control->delta;
  if (!(delta > 0.0f)) return 0u;
  const float scaled = (d - control->bucket_base) * control->inv_delta;
  int offset = scaled <= 0.0f ? 0 : static_cast<int>(scaled);
  if (offset < 0) offset = 0;
  if (offset >= static_cast<int>(kBucketCount)) offset = static_cast<int>(kBucketCount - 1u);
  return static_cast<uint32_t>(offset);
}

template <typename Weight>
__device__ bool enqueue_work_item(const DeviceState& state,
                                  uint32_t vertex,
                                  Weight priority,
                                  uint32_t* cache_gen,
                                  uint32_t* cache_key,
                                  uint32_t* cache_page) {
  DeviceControl* control = state.control;

  while (!volatile_load_u32(&control->terminate)) {
    const uint32_t head = volatile_load_u32(&control->head);
    const uint32_t offset = bucket_offset_for_priority(control, priority);
    const uint32_t bucket = (head + offset) & (kBucketCount - 1u);
    const uint32_t generation = volatile_load_u32(&state.bucket_generation[bucket]);

    if (volatile_load_u32(&state.bucket_state[bucket]) != kOpen) continue;

    atomicAdd(&state.writer_count[bucket], 1u);
    __threadfence();

    if (volatile_load_u32(&state.bucket_state[bucket]) == kOpen &&
        volatile_load_u32(&state.bucket_generation[bucket]) == generation) {
      const uint32_t index = atomicAdd(&state.resv_ptr[bucket], 1u);
      const uint32_t page_index = index >> kPageShift;
      if (page_index >= control->max_pages_per_bucket) {
        atomicSub(&state.writer_count[bucket], 1u);
        device_set_error(control, kDeviceBucketOverflow);
        return false;
      }

      const uint32_t page = translate_page_cached(state, bucket, generation, page_index,
                                                  cache_gen, cache_key, cache_page);
      if (page == kInvalidPage) {
        atomicSub(&state.writer_count[bucket], 1u);
        return false;
      }

      state.pool[static_cast<uint64_t>(page) * kPageWords + (index & kPageMask)] = vertex;
      __threadfence();
      const uint32_t seg_in_page = (index & kPageMask) / kSegmentWords;
      atomicAdd(&state.wcc[static_cast<uint64_t>(page) * kSegmentsPerPage + seg_in_page], 1u);

      atomic_add_u64(&control->enqueued_items_total, 1u);
      atomic_add_u64(&control->period_enqueued_total, 1u);
      if (offset == kBucketCount - 1u) {
        atomic_add_u64(&control->period_tail_enqueued, 1u);
      }

      atomicSub(&state.writer_count[bucket], 1u);
      return true;
    }

    atomicSub(&state.writer_count[bucket], 1u);
  }

  return false;
}

template <typename Weight>
__device__ inline void relax_edge(const DeviceState& state,
                                  const DeviceCsrGraph<Weight>& graph,
                                  Weight* distances,
                                  Weight source_distance,
                                  uint32_t edge,
                                  uint32_t* cache_gen,
                                  uint32_t* cache_key,
                                  uint32_t* cache_page) {
  const uint32_t dst = graph.col_indices[edge];
  const Weight nd = DistanceOps<Weight>::add(source_distance, graph.weights[edge]);
  const Weight old = DistanceOps<Weight>::atomic_min(&distances[dst], nd);
  if (DistanceOps<Weight>::less(nd, old)) {
    atomic_add_u64(&state.control->successful_relaxations_total, 1u);
    enqueue_work_item(state, dst, nd, cache_gen, cache_key, cache_page);
  }
}

template <typename Weight>
__device__ void process_low_degree_vertex(const DeviceState& state,
                                           const DeviceCsrGraph<Weight>& graph,
                                           Weight* distances,
                                           uint32_t v,
                                           uint32_t* cache_gen,
                                           uint32_t* cache_key,
                                           uint32_t* cache_page) {
  if (v >= graph.num_vertices) {
    device_set_error(state.control, kDeviceInvalidVertex);
    return;
  }

  const Weight dv = distances[v];
  const uint32_t begin = graph.row_offsets[v];
  const uint32_t end = graph.row_offsets[v + 1u];
  atomic_add_u64(&state.control->edge_relaxations_total,
                 static_cast<uint64_t>(end - begin));
  for (uint32_t e = begin; e < end; ++e) {
    relax_edge(state, graph, distances, dv, e, cache_gen, cache_key, cache_page);
  }
}

template <typename Weight>
__device__ void process_assignment(const DeviceState& state,
                                   const DeviceCsrGraph<Weight>& graph,
                                   Weight* distances,
                                   const AssignmentSlot& assignment,
                                   uint32_t* cache_gen,
                                   uint32_t* cache_key,
                                   uint32_t* cache_page,
                                   uint32_t* high_vertices,
                                   uint32_t* high_count) {
  if (threadIdx.x == 0) *high_count = 0u;
  __syncthreads();

  for (uint32_t local = threadIdx.x; local < assignment.count; local += blockDim.x) {
    const uint32_t item_index = assignment.start + local;
    const uint32_t v = load_bucket_item(state, assignment.bucket, assignment.generation,
                                        item_index, cache_gen, cache_key, cache_page);
    if (v == kInvalidPage || v >= graph.num_vertices) {
      device_set_error(state.control, kDeviceInvalidVertex);
      continue;
    }

    const uint32_t begin = graph.row_offsets[v];
    const uint32_t end = graph.row_offsets[v + 1u];
    const uint32_t degree = end - begin;
    if (degree >= kCoopDegreeThreshold) {
      const uint32_t pos = atomicAdd(high_count, 1u);
      if (pos < kHighVertexCapacity) {
        high_vertices[pos] = v;
      } else {
        // Rare overflow fallback: keep correctness by processing serially.
        process_low_degree_vertex(state, graph, distances, v, cache_gen, cache_key, cache_page);
      }
    } else {
      process_low_degree_vertex(state, graph, distances, v, cache_gen, cache_key, cache_page);
    }
  }

  __syncthreads();

  const uint32_t high_n = dev_min_u32(*high_count, kHighVertexCapacity);
  for (uint32_t i = 0; i < high_n; ++i) {
    const uint32_t v = high_vertices[i];
    const Weight dv = distances[v];
    const uint32_t begin = graph.row_offsets[v];
    const uint32_t end = graph.row_offsets[v + 1u];
    if (threadIdx.x == 0) {
      atomic_add_u64(&state.control->edge_relaxations_total,
                     static_cast<uint64_t>(end - begin));
    }
    for (uint32_t e = begin + threadIdx.x; e < end; e += blockDim.x) {
      relax_edge(state, graph, distances, dv, e, cache_gen, cache_key, cache_page);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&state.completed_count[assignment.bucket], assignment.count);
    atomic_add_u64(&state.control->processed_items_total, assignment.count);
    atomic_add_u64(&state.control->inflight_items,
                   static_cast<uint64_t>(0ull - static_cast<unsigned long long>(assignment.count)));
  }
}

template <typename Weight>
__device__ void worker_loop(const DeviceState& state,
                            const DeviceCsrGraph<Weight>& graph,
                            Weight* distances,
                            uint32_t worker_id) {
  __shared__ uint32_t cache_gen[kCacheEntries];
  __shared__ uint32_t cache_key[kCacheEntries];
  __shared__ uint32_t cache_page[kCacheEntries];
  __shared__ uint32_t high_vertices[kHighVertexCapacity];
  __shared__ uint32_t high_count;
  __shared__ uint32_t sh_state;
  __shared__ uint32_t sh_terminate;
  __shared__ AssignmentSlot sh_assignment;

  for (uint32_t i = threadIdx.x; i < kCacheEntries; i += blockDim.x) {
    cache_gen[i] = 0u;
    cache_key[i] = 0xffffffffu;
    cache_page[i] = kInvalidPage;
  }
  __syncthreads();

  AssignmentSlot* af = &state.assignments[worker_id];

  while (true) {
    if (threadIdx.x == 0) {
      sh_state = volatile_load_u32(&af->state);
      sh_terminate = volatile_load_u32(&state.control->terminate);
      if (sh_state == kAssignmentReady) {
        __threadfence();
        sh_assignment.bucket = af->bucket;
        sh_assignment.generation = af->generation;
        sh_assignment.start = af->start;
        sh_assignment.count = af->count;
        sh_assignment.logical_offset = af->logical_offset;
      }
    }
    __syncthreads();

    if (sh_state == kAssignmentReady) {
      process_assignment(state, graph, distances, sh_assignment, cache_gen, cache_key,
                         cache_page, high_vertices, &high_count);
      __syncthreads();

      if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(&af->state, kAssignmentIdle);
      }
      __syncthreads();
    } else if (sh_state == kAssignmentStop || sh_terminate) {
      break;
    }
  }
}


__device__ bool manager_bucket_complete_thread0(const DeviceState& state, uint32_t bucket) {
  __threadfence();
  const uint32_t resv = volatile_load_u32(&state.resv_ptr[bucket]);
  const uint32_t read = volatile_load_u32(&state.read_ptr[bucket]);
  const uint32_t done = volatile_load_u32(&state.completed_count[bucket]);
  return read == resv && done == resv;
}

__device__ void manager_ensure_capacity(const DeviceState& state, uint32_t bucket) {
  __shared__ uint32_t sh_need;
  __shared__ uint32_t sh_page;

  while (true) {
    if (threadIdx.x == 0) {
      sh_need = 0u;
      if (volatile_load_u32(&state.control->error)) {
        // Do not allocate after an error.
      } else {
        const uint32_t resv = volatile_load_u32(&state.resv_ptr[bucket]);
        const uint32_t needed = (resv + kPageWords - 1u) >> kPageShift;
        const uint32_t have = volatile_load_u32(&state.page_count[bucket]);
        if (needed > state.control->max_pages_per_bucket) {
          device_set_error(state.control, kDeviceBucketOverflow);
        } else if (have < needed) {
          const uint32_t top = volatile_load_u32(state.free_top);
          if (top == 0u) {
            device_set_error(state.control, kDevicePoolExhausted);
          } else {
            *state.free_top = top - 1u;
            sh_page = state.free_stack[top - 1u];
            sh_need = 1u;
          }
        }
      }
    }
    __syncthreads();

    if (sh_need == 0u) break;

    const uint32_t page = sh_page;
    for (uint32_t i = threadIdx.x; i < kSegmentsPerPage; i += blockDim.x) {
      state.wcc[static_cast<uint64_t>(page) * kSegmentsPerPage + i] = 0u;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const uint32_t have = state.page_count[bucket];
      state.page_table[static_cast<uint64_t>(bucket) * state.control->max_pages_per_bucket + have] = page;
      __threadfence();
      state.page_count[bucket] = have + 1u;
    }
    __syncthreads();
  }
}

__device__ uint32_t manager_committed_end_thread0(const DeviceState& state,
                                                  uint32_t bucket,
                                                  uint32_t max_count) {
  uint32_t end = state.read_ptr[bucket];
  const uint32_t resv0 = volatile_load_u32(&state.resv_ptr[bucket]);
  const uint32_t limit = dev_min_u32(resv0, end + max_count);

  while (end < limit) {
    const uint32_t page_index = end >> kPageShift;
    if (page_index >= volatile_load_u32(&state.page_count[bucket])) break;

    const uint32_t page = volatile_load_u32(&state.page_table[
        static_cast<uint64_t>(bucket) * state.control->max_pages_per_bucket + page_index]);
    if (page == kInvalidPage) break;

    const uint32_t offset = end & kPageMask;
    const uint32_t seg_in_page = offset / kSegmentWords;
    const uint32_t seg_abs_start = (end / kSegmentWords) * kSegmentWords;
    const uint32_t seg_abs_end = seg_abs_start + kSegmentWords;
    const uint32_t seg_end = dev_min_u32(seg_abs_end, limit);
    const uint32_t wcc = volatile_load_u32(&state.wcc[
        static_cast<uint64_t>(page) * kSegmentsPerPage + seg_in_page]);

    if (wcc >= kSegmentWords) {
      end = seg_end;
      continue;
    }

    __threadfence();
    const uint32_t resv = volatile_load_u32(&state.resv_ptr[bucket]);
    if (seg_abs_start + wcc == resv) {
      end = dev_min_u32(resv, limit);
    }
    break;
  }

  return end;
}

__device__ bool manager_assign_idle_workers_thread0(const DeviceState& state) {
  DeviceControl* control = state.control;
  bool assigned_any = false;
  const uint32_t active = dev_min_u32(control->active_buckets, control->max_active_buckets);
  const uint32_t chunk = dev_max_u32(1u, control->assignment_chunk);

  for (uint32_t worker = 0; worker < control->worker_blocks; ++worker) {
    AssignmentSlot* af = &state.assignments[worker];
    if (volatile_load_u32(&af->state) != kAssignmentIdle) continue;

    for (uint32_t logical = 0; logical < active; ++logical) {
      const uint32_t bucket = (control->head + logical) & (kBucketCount - 1u);
      if (volatile_load_u32(&state.bucket_state[bucket]) != kOpen) continue;

      const uint32_t read = volatile_load_u32(&state.read_ptr[bucket]);
      const uint32_t committed = manager_committed_end_thread0(state, bucket, chunk);
      if (committed <= read) continue;

      const uint32_t count = committed - read;
      state.read_ptr[bucket] = committed;

      af->bucket = bucket;
      af->generation = volatile_load_u32(&state.bucket_generation[bucket]);
      af->start = read;
      af->count = count;
      af->logical_offset = logical;
      __threadfence();
      atomicExch(&af->state, kAssignmentReady);

      control->assigned_items_total += count;
      const uint64_t inflight = atomic_add_u64(&control->inflight_items, count) + count;
      atomic_max_u64(&control->max_inflight_items, inflight);
      assigned_any = true;
      break;
    }
  }

  return assigned_any;
}

__device__ bool manager_all_quiescent_thread0(const DeviceState& state) {
  const DeviceControl* control = state.control;

  for (uint32_t worker = 0; worker < control->worker_blocks; ++worker) {
    if (volatile_load_u32(&state.assignments[worker].state) == kAssignmentReady) return false;
  }

  for (uint32_t b = 0; b < kBucketCount; ++b) {
    if (volatile_load_u32(&state.writer_count[b]) != 0u) return false;
    const uint32_t resv = volatile_load_u32(&state.resv_ptr[b]);
    if (volatile_load_u32(&state.read_ptr[b]) != resv) return false;
    if (volatile_load_u32(&state.completed_count[b]) != resv) return false;
  }

  return true;
}

__device__ void manager_update_active_buckets_thread0(const DeviceState& state) {
  DeviceControl* control = state.control;
  if (!control->dynamic_active_buckets) return;

  const float capacity = static_cast<float>(dev_max_u32(1u, control->worker_blocks) *
                                           dev_max_u32(1u, control->assignment_chunk));
  const float inflight = static_cast<float>(volatile_load_u64(&control->inflight_items));
  const float low = control->utilization_low * capacity;
  const float high = control->utilization_high * capacity;

  if (inflight < low && control->active_buckets < control->max_active_buckets) {
    ++control->active_buckets;
  } else if (inflight > high && control->active_buckets > 1u) {
    --control->active_buckets;
  }
}

__device__ void manager_adjust_delta_thread0(const DeviceState& state) {
  DeviceControl* control = state.control;
  if (!control->dynamic_delta) return;
  if (control->head_switches_since_adjust < control->adjust_period_head_switches) return;

  const uint64_t enq_total = control->period_enqueued_total;
  const float tail_ratio = enq_total == 0u
      ? 0.0f
      : static_cast<float>(control->period_tail_enqueued) / static_cast<float>(enq_total);

  const uint64_t samples = dev_max_u64(1ull, control->period_util_samples);
  const float avg_inflight = static_cast<float>(control->period_util_accum) /
                             static_cast<float>(samples);
  const float capacity = static_cast<float>(dev_max_u32(1u, control->worker_blocks) *
                                           dev_max_u32(1u, control->assignment_chunk));
  const float low = control->utilization_low * capacity;
  const float high = control->utilization_high * capacity;

  float new_delta = control->delta;

  if (tail_ratio >= control->clip_threshold) {
    new_delta = fmaxf(control->delta * control->delta_growth,
                      control->delta + control->min_delta);
    control->delta_lower_bound = fmaxf(control->delta_lower_bound, new_delta);
  } else if (avg_inflight < low) {
    new_delta = control->delta * control->delta_growth;
  } else if (avg_inflight > high && control->active_buckets <= 2u) {
    new_delta = fmaxf(control->delta * control->delta_shrink,
                      control->delta_lower_bound);
  }

  new_delta = fmaxf(new_delta, control->min_delta);
  if (new_delta != control->delta && new_delta > 0.0f) {
    control->delta = new_delta;
    control->inv_delta = 1.0f / new_delta;
    ++control->delta_updates;
    ++control->epoch;
  }

  control->period_enqueued_total = 0u;
  control->period_tail_enqueued = 0u;
  control->period_util_accum = 0u;
  control->period_util_samples = 0u;
  control->head_switches_since_adjust = 0u;
}

__device__ void manager_try_retire_head(const DeviceState& state) {
  __shared__ uint32_t sh_bucket;
  __shared__ uint32_t sh_try;
  __shared__ uint32_t sh_can_reset;

  if (threadIdx.x == 0) {
    sh_bucket = state.control->head;
    sh_try = manager_bucket_complete_thread0(state, sh_bucket) ? 1u : 0u;
    if (sh_try) {
      state.bucket_state[sh_bucket] = kRetiring;
      __threadfence();
    }
  }
  __syncthreads();

  if (!sh_try) return;

  while (true) {
    manager_ensure_capacity(state, sh_bucket);
    if (threadIdx.x == 0) {
      sh_can_reset = (volatile_load_u32(&state.writer_count[sh_bucket]) == 0u ||
                      volatile_load_u32(&state.control->error)) ? 1u : 0u;
    }
    __syncthreads();
    if (sh_can_reset) break;
  }

  if (threadIdx.x == 0) {
    if (volatile_load_u32(&state.control->error)) {
      sh_can_reset = 0u;
      state.bucket_state[sh_bucket] = kOpen;
    } else {
      __threadfence();
      sh_can_reset = (manager_bucket_complete_thread0(state, sh_bucket) &&
                      volatile_load_u32(&state.writer_count[sh_bucket]) == 0u) ? 1u : 0u;
      if (!sh_can_reset) state.bucket_state[sh_bucket] = kOpen;
    }
  }
  __syncthreads();

  if (!sh_can_reset) return;

  const uint32_t page_count = state.page_count[sh_bucket];
  for (uint32_t pi = threadIdx.x; pi < page_count; pi += blockDim.x) {
    const uint64_t table_index = static_cast<uint64_t>(sh_bucket) *
                                 state.control->max_pages_per_bucket + pi;
    const uint32_t page = state.page_table[table_index];
    if (page != kInvalidPage) {
      const uint32_t pos = atomicAdd(state.free_top, 1u);
      if (pos < state.control->pool_pages) {
        state.free_stack[pos] = page;
      } else {
        device_set_error(state.control, kDevicePoolExhausted);
      }
      state.page_table[table_index] = kInvalidPage;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    state.resv_ptr[sh_bucket] = 0u;
    state.read_ptr[sh_bucket] = 0u;
    state.completed_count[sh_bucket] = 0u;
    state.page_count[sh_bucket] = 0u;
    uint32_t next_gen = state.bucket_generation[sh_bucket] + 1u;
    if (next_gen == 0u) next_gen = 1u;
    state.bucket_generation[sh_bucket] = next_gen;
    state.bucket_state[sh_bucket] = kOpen;

    state.control->head = (state.control->head + 1u) & (kBucketCount - 1u);
    state.control->bucket_base += state.control->delta;
    ++state.control->epoch;
    ++state.control->head_switches_since_adjust;
    ++state.control->head_switches_total;
  }
  __syncthreads();
}

__device__ void manager_loop(const DeviceState& state) {
  while (true) {
    for (uint32_t b = 0; b < kBucketCount; ++b) {
      manager_ensure_capacity(state, b);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (volatile_load_u32(&state.control->error)) {
        state.control->terminate = 1u;
      }

      if (!state.control->terminate) {
        const bool assigned = manager_assign_idle_workers_thread0(state);
        manager_update_active_buckets_thread0(state);

        const uint64_t inflight = volatile_load_u64(&state.control->inflight_items);
        state.control->period_util_accum += inflight;
        ++state.control->period_util_samples;

        if (manager_all_quiescent_thread0(state) && !assigned) {
          ++state.control->quiet_passes;
          if (state.control->quiet_passes >= 2u) {
            state.control->terminate = 1u;
          }
        } else {
          state.control->quiet_passes = 0u;
        }
      }
    }
    __syncthreads();

    if (volatile_load_u32(&state.control->terminate)) break;

    manager_try_retire_head(state);
    __syncthreads();

    if (threadIdx.x == 0) {
      manager_adjust_delta_thread0(state);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (uint32_t worker = 0; worker < state.control->worker_blocks; ++worker) {
      atomicExch(&state.assignments[worker].state, kAssignmentStop);
    }
    __threadfence();
  }
}

template <typename Weight>
__global__ __launch_bounds__(kThreadsPerBlock, 1)
void adds_kernel(DeviceState state, DeviceCsrGraph<Weight> graph, Weight* distances) {
  if (blockIdx.x == 0) {
    manager_loop(state);
  } else {
    worker_loop(state, graph, distances, blockIdx.x - 1u);
  }
}

template <typename Weight>
__global__ void init_kernel(DeviceState state,
                            DeviceCsrGraph<Weight> graph,
                            uint32_t source,
                            Weight* distances,
                            bool initialize_distances) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

  if (initialize_distances) {
    for (uint64_t i = tid; i < graph.num_vertices; i += stride) {
      distances[i] = (i == source) ? DistanceOps<Weight>::zero()
                                   : DistanceOps<Weight>::infinity();
    }
  }

  for (uint64_t i = tid; i < state.control->pool_pages - 1ull; i += stride) {
    state.free_stack[i] = static_cast<uint32_t>(i);
  }

  for (uint64_t i = tid; i < kBucketCount; i += stride) {
    state.page_count[i] = 0u;
    state.resv_ptr[i] = 0u;
    state.read_ptr[i] = 0u;
    state.completed_count[i] = 0u;
    state.bucket_state[i] = kOpen;
    state.bucket_generation[i] = 1u;
    state.writer_count[i] = 0u;
  }

  for (uint64_t i = tid; i < state.control->worker_blocks; i += stride) {
    state.assignments[i].state = kAssignmentIdle;
    state.assignments[i].bucket = 0u;
    state.assignments[i].generation = 0u;
    state.assignments[i].start = 0u;
    state.assignments[i].count = 0u;
    state.assignments[i].logical_offset = 0u;
  }

  if (tid == 0u) {
    if (!initialize_distances) distances[source] = DistanceOps<Weight>::zero();

    const uint32_t seed_page = state.control->pool_pages - 1u;
    *state.free_top = seed_page;
    state.page_table[0] = seed_page;
    state.page_count[0] = 1u;
    state.pool[static_cast<uint64_t>(seed_page) * kPageWords] = source;
    state.resv_ptr[0] = 1u;
    state.wcc[static_cast<uint64_t>(seed_page) * kSegmentsPerPage] = 1u;
    state.control->enqueued_items_total = 1u;
  }
}

template <typename Weight>
__global__ void profile_kernel(const Weight* weights, uint32_t edge_count, double* partial) {
  extern __shared__ double sh[];
  const uint32_t tid = threadIdx.x;
  double sum = 0.0;
  for (uint64_t e = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
       e < edge_count;
       e += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    sum += static_cast<double>(weights[e]);
  }

  sh[tid] = sum;
  __syncthreads();

  for (uint32_t offset = blockDim.x / 2u; offset > 0u; offset >>= 1u) {
    if (tid < offset) sh[tid] += sh[tid + offset];
    __syncthreads();
  }

  if (tid == 0u) partial[blockIdx.x] = sh[0];
}

struct Storage {
  DeviceBuffer<uint32_t> pool;
  DeviceBuffer<uint32_t> free_stack;
  DeviceBuffer<uint32_t> free_top;
  DeviceBuffer<uint32_t> page_table;
  DeviceBuffer<uint32_t> page_count;
  DeviceBuffer<uint32_t> resv_ptr;
  DeviceBuffer<uint32_t> read_ptr;
  DeviceBuffer<uint32_t> completed_count;
  DeviceBuffer<uint32_t> bucket_state;
  DeviceBuffer<uint32_t> bucket_generation;
  DeviceBuffer<uint32_t> writer_count;
  DeviceBuffer<uint32_t> wcc;
  DeviceBuffer<AssignmentSlot> assignments;
  DeviceBuffer<DeviceControl> control;

  DeviceState state(uint32_t max_pages_per_bucket) const {
    DeviceState s{};
    s.pool = pool.get();
    s.free_stack = free_stack.get();
    s.free_top = free_top.get();
    s.page_table = page_table.get();
    s.page_count = page_count.get();
    s.resv_ptr = resv_ptr.get();
    s.read_ptr = read_ptr.get();
    s.completed_count = completed_count.get();
    s.bucket_state = bucket_state.get();
    s.bucket_generation = bucket_generation.get();
    s.writer_count = writer_count.get();
    s.wcc = wcc.get();
    s.assignments = assignments.get();
    s.control = control.get();
    (void)max_pages_per_bucket;
    return s;
  }
};

Status allocate_storage(Storage* storage,
                        uint32_t worker_blocks,
                        uint32_t pool_pages,
                        uint32_t max_pages_per_bucket) {
  Status st = Status::Success;
#define ALLOC_OR_RETURN(buffer, count)       \
  do {                                      \
    st = (buffer).allocate((count));        \
    if (st != Status::Success) return st;   \
  } while (0)

  ALLOC_OR_RETURN(storage->pool, static_cast<size_t>(pool_pages) * kPageWords);
  ALLOC_OR_RETURN(storage->free_stack, pool_pages);
  ALLOC_OR_RETURN(storage->free_top, 1);
  ALLOC_OR_RETURN(storage->page_table,
                  static_cast<size_t>(kBucketCount) * max_pages_per_bucket);
  ALLOC_OR_RETURN(storage->page_count, kBucketCount);
  ALLOC_OR_RETURN(storage->resv_ptr, kBucketCount);
  ALLOC_OR_RETURN(storage->read_ptr, kBucketCount);
  ALLOC_OR_RETURN(storage->completed_count, kBucketCount);
  ALLOC_OR_RETURN(storage->bucket_state, kBucketCount);
  ALLOC_OR_RETURN(storage->bucket_generation, kBucketCount);
  ALLOC_OR_RETURN(storage->writer_count, kBucketCount);
  ALLOC_OR_RETURN(storage->wcc, static_cast<size_t>(pool_pages) * kSegmentsPerPage);
  ALLOC_OR_RETURN(storage->assignments, worker_blocks);
  ALLOC_OR_RETURN(storage->control, 1);

#undef ALLOC_OR_RETURN
  return Status::Success;
}

template <typename Weight>
uint32_t choose_worker_blocks_for_kernel(hipError_t* out_err,
                                         uint32_t requested,
                                         uint32_t threads_per_block) {
  int device = 0;
  hipError_t err = hipGetDevice(&device);
  if (!hip_ok(err)) {
    *out_err = err;
    return 0;
  }

  hipDeviceProp_t prop{};
  err = hipGetDeviceProperties(&prop, device);
  if (!hip_ok(err)) {
    *out_err = err;
    return 0;
  }

  int active_per_cu = 0;
  err = hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_per_cu,
                                                     adds_kernel<Weight>,
                                                     static_cast<int>(threads_per_block),
                                                     0);
  if (!hip_ok(err) || active_per_cu <= 0) {
    active_per_cu = 2;  // Conservative fallback for common ROCm GPUs.
    err = hipSuccess;
  }

  const uint64_t resident = static_cast<uint64_t>(std::max(1, prop.multiProcessorCount)) *
                            static_cast<uint64_t>(active_per_cu);
  if (resident < 2u) {
    *out_err = hipErrorInvalidValue;
    return 0;
  }

  uint32_t max_workers = static_cast<uint32_t>(std::min<uint64_t>(resident - 1u,
                                                                  std::numeric_limits<uint32_t>::max()));
  uint32_t workers = requested == 0u ? max_workers : std::min(requested, max_workers);
  workers = std::max(1u, workers);
  *out_err = err;
  return workers;
}

size_t choose_pool_words(uint32_t n, uint32_t m, const Options& options) {
  if (options.pool_words != 0u) return options.pool_words;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;

  const double fraction = clamp_double(options.memory_fraction, 0.05, 0.90);
  size_t budget_words = static_cast<size_t>(1ull << 26);  // 256 MiB fallback.
  if (err == hipSuccess && free_bytes > 0u) {
    // WCC/free-stack/page-table overhead is modest but not zero; divide by a
    // safety factor so the work queue itself does not consume the full budget.
    budget_words = static_cast<size_t>((static_cast<long double>(free_bytes) * fraction) /
                                       (sizeof(uint32_t) * 1.10L));
  }

  const size_t by_vertices = std::max<size_t>(static_cast<size_t>(n) * 4u, kPageWords * 4u);
  const size_t by_edges = static_cast<size_t>(m) * 2u;
  size_t desired = std::max(by_vertices, by_edges);
  desired = std::max<size_t>(desired, static_cast<size_t>(kPageWords) * kBucketCount);
  desired = std::min(desired, budget_words);
  desired = std::max<size_t>(desired, kPageWords);
  return desired;
}

template <typename Weight>
Status estimate_initial_delta(const DeviceCsrGraph<Weight>& graph,
                              const Options& options,
                              hipStream_t stream,
                              double* delta_out) {
  if (options.initial_delta > 0.0) {
    *delta_out = options.initial_delta;
    return Status::Success;
  }

  double avg_weight = 1.0;
  if (graph.num_edges != 0u) {
    const uint32_t blocks = std::min<uint32_t>(1024u, (graph.num_edges + kThreadsPerBlock - 1u) /
                                                     kThreadsPerBlock);
    DeviceBuffer<double> d_partial;
    Status st = d_partial.allocate(blocks);
    if (st != Status::Success) return st;

    hipLaunchKernelGGL((profile_kernel<Weight>), dim3(blocks), dim3(kThreadsPerBlock),
                       kThreadsPerBlock * sizeof(double), stream,
                       graph.weights, graph.num_edges, d_partial.get());
    hipError_t err = hipGetLastError();
    if (!hip_ok(err)) return Status::HipError;

    std::vector<double> partial(blocks, 0.0);
    err = hipMemcpyAsync(partial.data(), d_partial.get(), blocks * sizeof(double),
                         hipMemcpyDeviceToHost, stream);
    if (!hip_ok(err)) return Status::HipError;
    err = hipStreamSynchronize(stream);
    if (!hip_ok(err)) return Status::HipError;

    double sum = 0.0;
    for (double v : partial) sum += v;
    avg_weight = sum / static_cast<double>(graph.num_edges);
    if (!(avg_weight > 0.0) || !std::isfinite(avg_weight)) avg_weight = 1.0;
  }

  const double avg_degree = graph.num_vertices == 0u
      ? 1.0
      : static_cast<double>(std::max<uint32_t>(1u, graph.num_edges)) /
            static_cast<double>(graph.num_vertices);
  double delta = options.initial_delta_scale * avg_weight / std::max(avg_degree, 1.0e-12);
  delta = std::max(delta, options.min_delta);
  if (std::is_integral<Weight>::value) delta = std::max(delta, 1.0);
  if (!std::isfinite(delta) || delta <= 0.0) delta = std::max(options.min_delta, 1.0e-3);
  *delta_out = delta;
  return Status::Success;
}

Status map_device_error(uint32_t code) {
  switch (code) {
    case kDeviceSuccess: return Status::Success;
    case kDevicePoolExhausted: return Status::PoolExhausted;
    case kDeviceBucketOverflow: return Status::BucketOverflow;
    default: return Status::DeviceError;
  }
}

template <typename Weight>
Status run_sssp(const DeviceCsrGraph<Weight>& graph,
                uint32_t source,
                Weight* d_distances,
                RunStats* stats,
                const Options& user_options,
                hipStream_t stream) {
  if (graph.num_vertices == 0u || source >= graph.num_vertices || d_distances == nullptr ||
      graph.row_offsets == nullptr || graph.col_indices == nullptr || graph.weights == nullptr) {
    return Status::InvalidArgument;
  }

  Options options = user_options;
  options.assignment_chunk = std::max(1u, options.assignment_chunk);
  options.assignment_chunk = std::min(options.assignment_chunk, 4096u);
  options.initial_active_buckets = std::max(1u, std::min(options.initial_active_buckets, kBucketCount));
  options.max_active_buckets = std::max(1u, std::min(options.max_active_buckets, kBucketCount));
  options.max_active_buckets = std::max(options.max_active_buckets, options.initial_active_buckets);
  options.adjust_period_head_switches = std::max(1u, options.adjust_period_head_switches);
  options.delta_growth = std::max(options.delta_growth, 1.0001);
  options.delta_shrink = clamp_double(options.delta_shrink, 0.05, 0.9999);
  options.clip_threshold = clamp_double(options.clip_threshold, 0.05, 0.99);
  options.utilization_low = clamp_double(options.utilization_low, 0.01, 0.95);
  options.utilization_high = clamp_double(options.utilization_high, options.utilization_low + 0.01, 0.99);
  options.min_delta = std::max(options.min_delta, 1.0e-12);

  double initial_delta = 0.0;
  Status st = estimate_initial_delta(graph, options, stream, &initial_delta);
  if (st != Status::Success) return st;
  if (!std::isfinite(initial_delta) || initial_delta <= 0.0) initial_delta = options.min_delta;
  initial_delta = std::max(initial_delta, options.min_delta);
  if (std::is_integral<Weight>::value) initial_delta = std::max(initial_delta, 1.0);
  initial_delta = std::min(initial_delta,
                           static_cast<double>(std::numeric_limits<float>::max()) * 0.5);

  hipError_t err = hipSuccess;
  const uint32_t workers = choose_worker_blocks_for_kernel<Weight>(
      &err, options.worker_blocks, kThreadsPerBlock);
  if (!hip_ok(err) || workers == 0u) return Status::HipError;

  const size_t requested_pool_words = choose_pool_words(graph.num_vertices, graph.num_edges, options);
  size_t pool_pages_sz = requested_pool_words / kPageWords;
  if (pool_pages_sz == 0u) pool_pages_sz = 1u;
  if (pool_pages_sz > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
  const uint32_t pool_pages = static_cast<uint32_t>(pool_pages_sz);
  const uint32_t max_pages_per_bucket = std::min<uint32_t>(pool_pages, 1u << 16);
  if (max_pages_per_bucket == 0u) return Status::InvalidArgument;

  Storage storage;
  st = allocate_storage(&storage, workers, pool_pages, max_pages_per_bucket);
  if (st != Status::Success) return st;

  DeviceState state = storage.state(max_pages_per_bucket);

  err = hipMemsetAsync(storage.page_table.get(), 0xff,
                       static_cast<size_t>(kBucketCount) * max_pages_per_bucket * sizeof(uint32_t),
                       stream);
  if (!hip_ok(err)) return Status::HipError;
  err = hipMemsetAsync(storage.wcc.get(), 0,
                       static_cast<size_t>(pool_pages) * kSegmentsPerPage * sizeof(uint32_t),
                       stream);
  if (!hip_ok(err)) return Status::HipError;

  DeviceControl h_control{};
  h_control.terminate = 0u;
  h_control.error = kDeviceSuccess;
  h_control.worker_blocks = workers;
  h_control.pool_pages = pool_pages;
  h_control.max_pages_per_bucket = max_pages_per_bucket;
  h_control.assignment_chunk = options.assignment_chunk;
  h_control.head = 0u;
  h_control.epoch = 1u;
  h_control.active_buckets = options.initial_active_buckets;
  h_control.max_active_buckets = options.max_active_buckets;
  h_control.dynamic_delta = options.dynamic_delta ? 1u : 0u;
  h_control.dynamic_active_buckets = options.dynamic_active_buckets ? 1u : 0u;
  h_control.adjust_period_head_switches = options.adjust_period_head_switches;
  h_control.bucket_base = 0.0f;
  h_control.delta = static_cast<float>(initial_delta);
  h_control.inv_delta = 1.0f / h_control.delta;
  h_control.min_delta = static_cast<float>(options.min_delta);
  h_control.delta_growth = static_cast<float>(options.delta_growth);
  h_control.delta_shrink = static_cast<float>(options.delta_shrink);
  h_control.delta_lower_bound = h_control.min_delta;
  h_control.clip_threshold = static_cast<float>(options.clip_threshold);
  h_control.utilization_low = static_cast<float>(options.utilization_low);
  h_control.utilization_high = static_cast<float>(options.utilization_high);

  err = hipMemcpyAsync(storage.control.get(), &h_control, sizeof(DeviceControl),
                       hipMemcpyHostToDevice, stream);
  if (!hip_ok(err)) return Status::HipError;

  const uint32_t init_blocks = std::min<uint32_t>(65535u,
      std::max<uint32_t>(1u, (std::max<uint32_t>(graph.num_vertices, workers) +
                              kThreadsPerBlock - 1u) / kThreadsPerBlock));
  hipLaunchKernelGGL((init_kernel<Weight>), dim3(init_blocks), dim3(kThreadsPerBlock), 0,
                     stream, state, graph, source, d_distances, options.initialize_distances);
  err = hipGetLastError();
  if (!hip_ok(err)) return Status::HipError;

  hipLaunchKernelGGL((adds_kernel<Weight>), dim3(workers + 1u), dim3(kThreadsPerBlock), 0,
                     stream, state, graph, d_distances);
  err = hipGetLastError();
  if (!hip_ok(err)) return Status::HipError;

  err = hipMemcpyAsync(&h_control, storage.control.get(), sizeof(DeviceControl),
                       hipMemcpyDeviceToHost, stream);
  if (!hip_ok(err)) return Status::HipError;
  err = hipStreamSynchronize(stream);
  if (!hip_ok(err)) return Status::HipError;

  if (stats != nullptr) {
    stats->initial_delta = initial_delta;
    stats->final_delta = h_control.delta;
    stats->final_bucket_base = h_control.bucket_base;
    stats->worker_blocks = workers;
    stats->final_active_buckets = h_control.active_buckets;
    stats->pool_pages = pool_pages;
    stats->device_error = h_control.error;
    stats->assigned_items = h_control.assigned_items_total;
    stats->processed_items = h_control.processed_items_total;
    stats->enqueued_items = h_control.enqueued_items_total;
    stats->edge_relaxations = h_control.edge_relaxations_total;
    stats->successful_relaxations = h_control.successful_relaxations_total;
    stats->head_switches = h_control.head_switches_total;
    stats->delta_updates = h_control.delta_updates;
    stats->max_inflight_items = h_control.max_inflight_items;
  }

  return map_device_error(h_control.error);
}

}  // namespace detail

const char* status_string(Status status) {
  switch (status) {
    case Status::Success: return "success";
    case Status::InvalidArgument: return "invalid argument";
    case Status::HipError: return "HIP runtime error";
    case Status::OutOfMemory: return "device out of memory";
    case Status::PoolExhausted: return "ADDS worklist pool exhausted";
    case Status::BucketOverflow: return "ADDS bucket exceeded 32-bit/64K-page addressing capacity";
    case Status::DeviceError: return "ADDS device-side error";
    default: return "unknown";
  }
}

Status sssp(const DeviceCsrGraph<float>& graph,
            uint32_t source,
            float* d_distances,
            RunStats* stats,
            const Options& options,
            hipStream_t stream) {
  return detail::run_sssp<float>(graph, source, d_distances, stats, options, stream);
}

Status sssp(const DeviceCsrGraph<uint32_t>& graph,
            uint32_t source,
            uint32_t* d_distances,
            RunStats* stats,
            const Options& options,
            hipStream_t stream) {
  return detail::run_sssp<uint32_t>(graph, source, d_distances, stats, options, stream);
}

}  // namespace hip_adds
