#include "sssp_hip.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr int THREADS = 256;
constexpr int MAX_ALLOWED_BUCKETS = 64;

void check_hip(hipError_t status, const char* message) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(status));
  }
}

void validate_graph(const CsrGraph& g, int source) {
  if (g.num_nodes <= 0) {
    throw std::invalid_argument("graph.num_nodes must be positive");
  }
  if (source < 0 || source >= g.num_nodes) {
    throw std::invalid_argument("source is outside graph");
  }
  if (static_cast<int>(g.row_offsets.size()) != g.num_nodes + 1) {
    throw std::invalid_argument("row_offsets must contain n + 1 entries");
  }
  if (g.row_offsets.front() != 0 || g.row_offsets.back() != g.num_edges) {
    throw std::invalid_argument("row_offsets must start at 0 and end at num_edges");
  }
  if (static_cast<int>(g.col_indices.size()) != g.num_edges ||
      static_cast<int>(g.weights.size()) != g.num_edges) {
    throw std::invalid_argument("col_indices and weights must contain num_edges entries");
  }
  for (int u = 0; u < g.num_nodes; ++u) {
    if (g.row_offsets[u] > g.row_offsets[u + 1]) {
      throw std::invalid_argument("row_offsets must be nondecreasing");
    }
  }
  for (int e = 0; e < g.num_edges; ++e) {
    if (g.col_indices[e] < 0 || g.col_indices[e] >= g.num_nodes) {
      throw std::invalid_argument("col_indices contains a vertex outside graph");
    }
    if (!(g.weights[e] >= 0.0f) || std::isnan(g.weights[e])) {
      throw std::invalid_argument("ADDS / delta-stepping requires non-negative, non-NaN weights");
    }
  }
}

float choose_initial_delta(const CsrGraph& g, const SSSPOptions& opts) {
  if (opts.initial_delta > 0.0f && std::isfinite(opts.initial_delta)) {
    return opts.initial_delta;
  }

  double sum = 0.0;
  int positive = 0;
  for (float w : g.weights) {
    if (w > 0.0f && std::isfinite(w)) {
      sum += static_cast<double>(w);
      ++positive;
    }
  }

  const double avg_weight = positive ? (sum / static_cast<double>(positive)) : 1.0;
  const double avg_degree = g.num_nodes ?
      (static_cast<double>(g.num_edges) / static_cast<double>(g.num_nodes)) : 1.0;

  // Same family as the Near-Far heuristic cited in the paper: Delta = C * W / D.
  // C=16 is intentionally moderate. Dynamic adjustment can move from here.
  const double delta = 16.0 * avg_weight / std::max(1.0, avg_degree);
  return static_cast<float>(std::max(1.0e-6, delta));
}

CsrGraph validate_and_materialize_dense(const std::vector<float>& adjacency, int n) {
  if (n <= 0) {
    throw std::invalid_argument("n must be positive");
  }
  const std::size_t expected = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (adjacency.size() != expected) {
    throw std::invalid_argument("adjacency must contain n*n row-major values");
  }
  return dense_to_csr_outgoing(adjacency, n);
}

struct DeviceCsrGraph {
  int num_nodes;
  int num_edges;
  const int* row_offsets;
  const int* col_indices;
  const float* weights;
};

struct DeviceGraphStorage {
  int* d_row_offsets = nullptr;
  int* d_col_indices = nullptr;
  int* d_edge_src = nullptr;
  float* d_weights = nullptr;

  void allocate_and_copy(const CsrGraph& g, hipStream_t stream) {
    const std::size_t rows_bytes = static_cast<std::size_t>(g.num_nodes + 1) * sizeof(int);
    const std::size_t edges_bytes = static_cast<std::size_t>(g.num_edges) * sizeof(int);
    const std::size_t weights_bytes = static_cast<std::size_t>(g.num_edges) * sizeof(float);

    check_hip(hipMalloc(reinterpret_cast<void**>(&d_row_offsets), rows_bytes), "hipMalloc row_offsets");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_col_indices), edges_bytes), "hipMalloc col_indices");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_weights), weights_bytes), "hipMalloc weights");

    check_hip(hipMemcpyAsync(d_row_offsets, g.row_offsets.data(), rows_bytes,
                             hipMemcpyHostToDevice, stream), "copy row_offsets");
    check_hip(hipMemcpyAsync(d_col_indices, g.col_indices.data(), edges_bytes,
                             hipMemcpyHostToDevice, stream), "copy col_indices");
    check_hip(hipMemcpyAsync(d_weights, g.weights.data(), weights_bytes,
                             hipMemcpyHostToDevice, stream), "copy weights");
  }

  void allocate_edge_src_and_copy(const CsrGraph& g, hipStream_t stream) {
    std::vector<int> edge_src(g.num_edges);
    for (int u = 0; u < g.num_nodes; ++u) {
      for (int e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
        edge_src[e] = u;
      }
    }
    const std::size_t edges_bytes = static_cast<std::size_t>(g.num_edges) * sizeof(int);
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_edge_src), edges_bytes), "hipMalloc edge_src");
    check_hip(hipMemcpyAsync(d_edge_src, edge_src.data(), edges_bytes,
                             hipMemcpyHostToDevice, stream), "copy edge_src");
  }

  DeviceCsrGraph view(const CsrGraph& g) const {
    return DeviceCsrGraph{g.num_nodes, g.num_edges, d_row_offsets, d_col_indices, d_weights};
  }

  void release() noexcept {
    if (d_row_offsets) hipFree(d_row_offsets);
    if (d_col_indices) hipFree(d_col_indices);
    if (d_edge_src) hipFree(d_edge_src);
    if (d_weights) hipFree(d_weights);
    d_row_offsets = nullptr;
    d_col_indices = nullptr;
    d_edge_src = nullptr;
    d_weights = nullptr;
  }

  ~DeviceGraphStorage() { release(); }
};

__device__ float atomic_min_float_nonnegative(float* address, float value) {
  // This CAS loop is safe for non-negative finite values and +inf.
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int;
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
  return __int_as_float(old);
}

__global__ void init_dist_kernel(float* dist, int n, int source) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dist[i] = (i == source) ? 0.0f : INF;
  }
}

__global__ void bf_relax_edges_kernel(const int* edge_src,
                                      const int* col_indices,
                                      const float* weights,
                                      int m,
                                      float* dist,
                                      int* changed,
                                      unsigned long long* work_count) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= m) {
    return;
  }

  const int u = edge_src[e];
  const int v = col_indices[e];
  const float du = dist[u];
  if (!isfinite(du)) {
    return;
  }

  atomicAdd(work_count, 1ULL);
  const float candidate = du + weights[e];
  const float old = atomic_min_float_nonnegative(&dist[v], candidate);
  if (candidate < old) {
    atomicExch(changed, 1);
  }
}

__global__ void bf_detect_negative_like_change_kernel(const int* edge_src,
                                                       const int* col_indices,
                                                       const float* weights,
                                                       int m,
                                                       const float* dist,
                                                       int* changed) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= m) {
    return;
  }
  const int u = edge_src[e];
  const int v = col_indices[e];
  const float du = dist[u];
  if (isfinite(du) && du + weights[e] < dist[v]) {
    atomicExch(changed, 1);
  }
}

__device__ void add_to_bucket(int vertex,
                              float distance,
                              float head_lower,
                              float delta,
                              int head_bucket,
                              int num_buckets,
                              int bucket_capacity,
                              int* bucket_items,
                              int* bucket_counts,
                              int* overflow) {
  float relative = (distance - head_lower) / delta;
  int offset = static_cast<int>(floorf(relative));
  if (offset < 0) {
    offset = 0;
  }
  if (offset >= num_buckets) {
    offset = num_buckets - 1;  // clipping, like the paper's limited bucket window
  }

  const int bucket = (head_bucket + offset) % num_buckets;
  const int pos = atomicAdd(&bucket_counts[bucket], 1);
  if (pos < bucket_capacity) {
    bucket_items[bucket * bucket_capacity + pos] = vertex;
  } else {
    atomicSub(&bucket_counts[bucket], 1);
    atomicExch(overflow, 1);
  }
}

__global__ void adds_process_bucket_kernel(DeviceCsrGraph g,
                                           float* dist,
                                           const int current_bucket,
                                           const int start_pos,
                                           const int end_pos,
                                           const float head_lower,
                                           const float delta,
                                           const int num_buckets,
                                           const int bucket_capacity,
                                           int* bucket_items,
                                           int* bucket_counts,
                                           int* overflow,
                                           unsigned long long* work_count) {
  const int local_pos = start_pos + blockIdx.x * blockDim.x + threadIdx.x;
  if (local_pos >= end_pos) {
    return;
  }

  const int u = bucket_items[current_bucket * bucket_capacity + local_pos];
  const float du = dist[u];
  if (!isfinite(du)) {
    return;
  }

  // If this vertex was clipped into the current bucket but is still outside the
  // current delta range, put it back into the future bucket window instead of
  // relaxing it too early. This preserves useful ordering without requiring
  // a huge number of buckets.
  if (du >= head_lower + delta) {
    add_to_bucket(u, du, head_lower, delta, current_bucket,
                  num_buckets, bucket_capacity, bucket_items,
                  bucket_counts, overflow);
    return;
  }

  atomicAdd(work_count, 1ULL);

  const int row_begin = g.row_offsets[u];
  const int row_end = g.row_offsets[u + 1];
  for (int e = row_begin; e < row_end; ++e) {
    const int v = g.col_indices[e];
    const float candidate = du + g.weights[e];
    const float old = atomic_min_float_nonnegative(&dist[v], candidate);
    if (candidate < old) {
      add_to_bucket(v, candidate, head_lower, delta, current_bucket,
                    num_buckets, bucket_capacity, bucket_items,
                    bucket_counts, overflow);
    }
  }
}

void start_event_pair(hipEvent_t* start, hipEvent_t* stop, hipStream_t stream) {
  check_hip(hipEventCreate(start), "hipEventCreate start");
  check_hip(hipEventCreate(stop), "hipEventCreate stop");
  check_hip(hipEventRecord(*start, stream), "hipEventRecord start");
}

float stop_event_pair(hipEvent_t start, hipEvent_t stop, hipStream_t stream) {
  check_hip(hipEventRecord(stop, stream), "hipEventRecord stop");
  check_hip(hipEventSynchronize(stop), "hipEventSynchronize stop");
  float ms = 0.0f;
  check_hip(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
  hipEventDestroy(start);
  hipEventDestroy(stop);
  return ms;
}

}  // namespace

CsrGraph dense_to_csr_outgoing(const std::vector<float>& adjacency, int n) {
  if (n <= 0) {
    throw std::invalid_argument("n must be positive");
  }
  const std::size_t expected = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (adjacency.size() != expected) {
    throw std::invalid_argument("adjacency must contain n*n row-major values");
  }

  CsrGraph g;
  g.num_nodes = n;
  g.row_offsets.assign(n + 1, 0);

  for (int u = 0; u < n; ++u) {
    for (int v = 0; v < n; ++v) {
      const float w = adjacency[static_cast<std::size_t>(u) * n + v];
      if (u != v && std::isfinite(w)) {
        if (w < 0.0f) {
          throw std::invalid_argument("negative edge in dense adjacency");
        }
        g.col_indices.push_back(v);
        g.weights.push_back(w);
      }
    }
    g.row_offsets[u + 1] = static_cast<int>(g.col_indices.size());
  }
  g.num_edges = static_cast<int>(g.col_indices.size());
  return g;
}

SSSPResult bellman_ford_sssp_hip(const CsrGraph& graph,
                                 int source,
                                 const SSSPOptions& options,
                                 hipStream_t stream) {
  validate_graph(graph, source);

  DeviceGraphStorage dev;
  dev.allocate_and_copy(graph, stream);
  dev.allocate_edge_src_and_copy(graph, stream);

  float* d_dist = nullptr;
  int* d_changed = nullptr;
  unsigned long long* d_work_count = nullptr;
  const std::size_t dist_bytes = static_cast<std::size_t>(graph.num_nodes) * sizeof(float);

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_dist), dist_bytes), "hipMalloc dist");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_changed), sizeof(int)), "hipMalloc changed");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_work_count), sizeof(unsigned long long)), "hipMalloc work_count");
  check_hip(hipMemsetAsync(d_work_count, 0, sizeof(unsigned long long), stream), "clear work_count");

  const int node_blocks = (graph.num_nodes + THREADS - 1) / THREADS;
  const int edge_blocks = (graph.num_edges + THREADS - 1) / THREADS;
  hipLaunchKernelGGL(init_dist_kernel, dim3(node_blocks), dim3(THREADS), 0, stream,
                     d_dist, graph.num_nodes, source);
  check_hip(hipGetLastError(), "launch init_dist_kernel");

  hipEvent_t start, stop;
  start_event_pair(&start, &stop, stream);

  const int max_iters = options.max_iters > 0 ? options.max_iters : graph.num_nodes - 1;
  SSSPResult result;
  for (int iter = 0; iter < max_iters; ++iter) {
    int changed = 0;
    check_hip(hipMemcpyAsync(d_changed, &changed, sizeof(int), hipMemcpyHostToDevice, stream),
              "reset changed");
    hipLaunchKernelGGL(bf_relax_edges_kernel, dim3(edge_blocks), dim3(THREADS), 0, stream,
                       dev.d_edge_src, dev.d_col_indices, dev.d_weights, graph.num_edges,
                       d_dist, d_changed, d_work_count);
    check_hip(hipGetLastError(), "launch bf_relax_edges_kernel");
    check_hip(hipMemcpyAsync(&changed, d_changed, sizeof(int), hipMemcpyDeviceToHost, stream),
              "copy changed");
    check_hip(hipStreamSynchronize(stream), "sync BF iteration");
    result.iterations_used = iter + 1;
    if (!changed) {
      result.converged = true;
      break;
    }
  }

  result.milliseconds = stop_event_pair(start, stop, stream);

  // One extra pass detects whether another relaxation is possible. For the
  // intended non-negative input graphs this should remain false.
  int changed = 0;
  check_hip(hipMemcpyAsync(d_changed, &changed, sizeof(int), hipMemcpyHostToDevice, stream),
            "reset negative-cycle changed");
  hipLaunchKernelGGL(bf_detect_negative_like_change_kernel, dim3(edge_blocks), dim3(THREADS), 0, stream,
                     dev.d_edge_src, dev.d_col_indices, dev.d_weights, graph.num_edges,
                     d_dist, d_changed);
  check_hip(hipGetLastError(), "launch bf_detect_negative_like_change_kernel");
  check_hip(hipMemcpyAsync(&changed, d_changed, sizeof(int), hipMemcpyDeviceToHost, stream),
            "copy negative-cycle flag");
  check_hip(hipStreamSynchronize(stream), "sync negative-cycle check");
  result.has_negative_cycle = changed != 0;

  result.dist.resize(graph.num_nodes);
  check_hip(hipMemcpyAsync(result.dist.data(), d_dist, dist_bytes, hipMemcpyDeviceToHost, stream),
            "copy BF result");
  check_hip(hipMemcpyAsync(&result.work_count, d_work_count, sizeof(unsigned long long),
                           hipMemcpyDeviceToHost, stream), "copy BF work_count");
  check_hip(hipStreamSynchronize(stream), "sync BF result copy");

  hipFree(d_dist);
  hipFree(d_changed);
  hipFree(d_work_count);
  return result;
}

SSSPResult adds_sssp_hip(const CsrGraph& graph,
                         int source,
                         const SSSPOptions& options,
                         hipStream_t stream) {
  validate_graph(graph, source);

  SSSPOptions opts = options;
  if (opts.num_buckets <= 1 || opts.num_buckets > MAX_ALLOWED_BUCKETS) {
    throw std::invalid_argument("num_buckets must be in [2, 64]");
  }
  if (opts.bucket_capacity_per_bucket <= 0) {
    const long long chosen = std::max<long long>(4096, 4LL * graph.num_nodes);
    if (chosen > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("default bucket capacity overflow; set bucket_capacity_per_bucket manually");
    }
    opts.bucket_capacity_per_bucket = static_cast<int>(chosen);
  }

  DeviceGraphStorage dev;
  dev.allocate_and_copy(graph, stream);
  const DeviceCsrGraph dg = dev.view(graph);

  float* d_dist = nullptr;
  int* d_bucket_items = nullptr;
  int* d_bucket_counts = nullptr;
  int* d_overflow = nullptr;
  unsigned long long* d_work_count = nullptr;

  const std::size_t dist_bytes = static_cast<std::size_t>(graph.num_nodes) * sizeof(float);
  const std::size_t counts_bytes = static_cast<std::size_t>(opts.num_buckets) * sizeof(int);
  const std::size_t total_bucket_items =
      static_cast<std::size_t>(opts.num_buckets) * static_cast<std::size_t>(opts.bucket_capacity_per_bucket);
  const std::size_t items_bytes = total_bucket_items * sizeof(int);

  check_hip(hipMalloc(reinterpret_cast<void**>(&d_dist), dist_bytes), "hipMalloc ADDS dist");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_bucket_items), items_bytes), "hipMalloc ADDS bucket_items");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_bucket_counts), counts_bytes), "hipMalloc ADDS bucket_counts");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_overflow), sizeof(int)), "hipMalloc ADDS overflow");
  check_hip(hipMalloc(reinterpret_cast<void**>(&d_work_count), sizeof(unsigned long long)), "hipMalloc ADDS work_count");

  check_hip(hipMemsetAsync(d_bucket_counts, 0, counts_bytes, stream), "clear bucket_counts");
  check_hip(hipMemsetAsync(d_overflow, 0, sizeof(int), stream), "clear overflow");
  check_hip(hipMemsetAsync(d_work_count, 0, sizeof(unsigned long long), stream), "clear ADDS work_count");

  const int node_blocks = (graph.num_nodes + THREADS - 1) / THREADS;
  hipLaunchKernelGGL(init_dist_kernel, dim3(node_blocks), dim3(THREADS), 0, stream,
                     d_dist, graph.num_nodes, source);
  check_hip(hipGetLastError(), "launch ADDS init_dist_kernel");

  const int first_count = 1;
  check_hip(hipMemcpyAsync(d_bucket_items, &source, sizeof(int), hipMemcpyHostToDevice, stream),
            "copy source to first bucket");
  check_hip(hipMemcpyAsync(d_bucket_counts, &first_count, sizeof(int), hipMemcpyHostToDevice, stream),
            "set first bucket count");
  check_hip(hipStreamSynchronize(stream), "sync ADDS initialization");

  std::vector<int> h_counts(opts.num_buckets, 0);
  float delta = choose_initial_delta(graph, opts);
  const float min_delta = std::max(1.0e-6f, delta / 1024.0f);
  const float max_delta = std::max(delta * 1024.0f, min_delta);
  float head_lower = 0.0f;
  int current_bucket = 0;
  int empty_advances = 0;
  const int max_bucket_advances = opts.max_iters > 0 ? opts.max_iters :
      std::max(4096, 64 * graph.num_nodes);

  SSSPResult result;
  hipEvent_t start, stop;
  start_event_pair(&start, &stop, stream);

  for (int step = 0; step < max_bucket_advances; ++step) {
    check_hip(hipMemcpyAsync(h_counts.data(), d_bucket_counts, counts_bytes,
                             hipMemcpyDeviceToHost, stream), "copy ADDS bucket_counts");
    check_hip(hipStreamSynchronize(stream), "sync ADDS bucket_counts");

    bool all_empty = true;
    for (int c : h_counts) {
      if (c > 0) {
        all_empty = false;
        break;
      }
    }
    if (all_empty) {
      result.converged = true;
      break;
    }

    int count = h_counts[current_bucket];
    if (count <= 0) {
      current_bucket = (current_bucket + 1) % opts.num_buckets;
      head_lower += delta;
      ++empty_advances;
      result.iterations_used = step + 1;
      continue;
    }

    empty_advances = 0;
    int processed = 0;
    int bucket_work_seen = 0;
    while (true) {
      check_hip(hipMemcpyAsync(&count, d_bucket_counts + current_bucket, sizeof(int),
                               hipMemcpyDeviceToHost, stream), "copy current bucket count");
      check_hip(hipStreamSynchronize(stream), "sync current bucket count");
      if (processed >= count) {
        break;
      }

      const int start_pos = processed;
      const int end_pos = count;
      const int work_items = end_pos - start_pos;
      bucket_work_seen += work_items;
      const int blocks = (work_items + THREADS - 1) / THREADS;
      hipLaunchKernelGGL(adds_process_bucket_kernel, dim3(blocks), dim3(THREADS), 0, stream,
                         dg, d_dist, current_bucket, start_pos, end_pos, head_lower, delta,
                         opts.num_buckets, opts.bucket_capacity_per_bucket, d_bucket_items,
                         d_bucket_counts, d_overflow, d_work_count);
      check_hip(hipGetLastError(), "launch adds_process_bucket_kernel");
      check_hip(hipStreamSynchronize(stream), "sync ADDS process bucket");
      processed = end_pos;

      int overflow = 0;
      check_hip(hipMemcpyAsync(&overflow, d_overflow, sizeof(int), hipMemcpyDeviceToHost, stream),
                "copy ADDS overflow");
      check_hip(hipStreamSynchronize(stream), "sync ADDS overflow");
      if (overflow) {
        throw std::runtime_error(
            "ADDS bucket overflow. Increase SSSPOptions::bucket_capacity_per_bucket "
            "or increase initial_delta so less work clips into the last bucket.");
      }
    }

    const int zero = 0;
    check_hip(hipMemcpyAsync(d_bucket_counts + current_bucket, &zero, sizeof(int),
                             hipMemcpyHostToDevice, stream), "reset completed bucket");
    check_hip(hipStreamSynchronize(stream), "sync reset completed bucket");

    if (opts.dynamic_delta) {
      if (bucket_work_seen < opts.target_low_work && delta < max_delta) {
        delta *= 2.0f;
      } else if (bucket_work_seen > opts.target_high_work && delta > min_delta) {
        delta *= 0.5f;
      }
      delta = std::min(max_delta, std::max(min_delta, delta));
    }

    current_bucket = (current_bucket + 1) % opts.num_buckets;
    head_lower += delta;
    result.iterations_used = step + 1;
    (void)empty_advances;
  }

  result.milliseconds = stop_event_pair(start, stop, stream);
  result.final_delta = delta;

  result.dist.resize(graph.num_nodes);
  check_hip(hipMemcpyAsync(result.dist.data(), d_dist, dist_bytes, hipMemcpyDeviceToHost, stream),
            "copy ADDS result");
  check_hip(hipMemcpyAsync(&result.work_count, d_work_count, sizeof(unsigned long long),
                           hipMemcpyDeviceToHost, stream), "copy ADDS work_count");
  check_hip(hipStreamSynchronize(stream), "sync ADDS result copy");

  hipFree(d_dist);
  hipFree(d_bucket_items);
  hipFree(d_bucket_counts);
  hipFree(d_overflow);
  hipFree(d_work_count);
  return result;
}

SSSPResult bellman_ford_sssp_hip(const std::vector<float>& adjacency,
                                 int n,
                                 int source,
                                 const SSSPOptions& options,
                                 hipStream_t stream) {
  CsrGraph g = validate_and_materialize_dense(adjacency, n);
  return bellman_ford_sssp_hip(g, source, options, stream);
}

SSSPResult adds_sssp_hip(const std::vector<float>& adjacency,
                         int n,
                         int source,
                         const SSSPOptions& options,
                         hipStream_t stream) {
  CsrGraph g = validate_and_materialize_dense(adjacency, n);
  return adds_sssp_hip(g, source, options, stream);
}
