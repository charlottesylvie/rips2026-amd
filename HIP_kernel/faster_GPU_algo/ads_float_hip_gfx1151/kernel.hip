/*
 * HIP port of the ADDS floating-point SSSP implementation.
 * Target: AMD Radeon 8060S / Strix Halo (gfx1151, wave32).
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "common.h"
#include "csr_graph.h"
#include "support.h"
#include "wl.h"

constexpr int TB_SIZE = 768;
constexpr int DEFAULT_RUNS = 8;

int HIP_DEVICE = 0;
int start_node = 0;
char* INPUT = nullptr;
char* OUTPUT = nullptr;

namespace {

unsigned read_env_unsigned(const char* name, unsigned fallback,
                           unsigned minimum, unsigned maximum) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return fallback;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        std::fprintf(stderr,
                     "Invalid %s='%s'; expected an integer in [%u, %u]\n",
                     name, text, minimum, maximum);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<unsigned>(parsed);
}

}  // namespace

__global__ void initialize_distances(CSRGraph graph, int source) {
    const unsigned tid = thread_id_x() + block_id_x() * block_dim_x();
    const unsigned stride = block_dim_x() * grid_dim_x();
    for (index_type node = tid; node < graph.nnodes; node += stride) {
        graph.node_data[node] =
            node == static_cast<index_type>(source) ? 0.0f : INF;
    }
}

#define HOR_EDGE 16

__global__ void __launch_bounds__(TB_SIZE, 1)
sssp_kernel(CSRGraph graph, worklist wl, unsigned* work_count) {
    wl.init_regular();

    const unsigned warp_id = thread_id_x() / ADS_WARP_SIZE;
    unsigned total_work = 0;

    // A low-degree wave can contain at most 32 * (HOR_EDGE - 1) edges.
    // The original implementation reserves 32*32 bytes per wave.
    __shared__ unsigned char
        leader_lane_tb_storage[TB_SIZE * ADS_WARP_SIZE];
    unsigned char* leader_lane_storage =
        &leader_lane_tb_storage[warp_id * ADS_WARP_SIZE * ADS_WARP_SIZE];

    const unsigned average_degree =
        graph.nnodes == 0 ? 0u : graph.nedges / graph.nnodes;
    const unsigned tb_coop_threshold = hd_max(
        ADS_WARP_SIZE * static_cast<unsigned>(TB_COOP_MUL),
        average_degree * static_cast<unsigned>(TB_COOP_MUL));

    while (true) {
        unsigned long long assignment = 0;
        const unsigned num = wl.get_assignment(
            graph, assignment, warp_id, work_count, total_work);
        if (assignment == EXIT) {
            return;
        }

        const unsigned ptr = agm_get_real_ptr(assignment);
        const unsigned source_bag_id = agm_get_bag_id(assignment);
        const unsigned worklist_offset = get_lane_id();

        int first_edge = 0;
        int degree = 0;
        int vertex = 0;
        if (worklist_offset < num) {
            vertex = static_cast<int>(
                wl.pop_work(ptr + worklist_offset));
            if (vertex >= 0 &&
                static_cast<unsigned>(vertex) < graph.nnodes) {
                first_edge = static_cast<int>(graph.row_start[vertex]);
                degree = static_cast<int>(graph.row_start[vertex + 1]) -
                         first_edge;
                ++total_work;
            }
        }

        // Split one extremely high-degree vertex among otherwise idle waves in
        // this block.  tb_coop_assign sets the leader's degree to zero if the
        // cooperative assignment succeeds.
        unsigned process_mask = warp_ballot_sync(
            FULL_MASK,
            degree >= static_cast<int>(tb_coop_threshold));
        const unsigned tb_coop_lane = find_ms_bit(process_mask);
        if (tb_coop_lane != NOT_FOUND) {
            wl.tb_coop_assign(vertex, first_edge, assignment, degree,
                              tb_coop_lane);
        }

        // Process each remaining high-degree vertex with an entire wave.
        process_mask = warp_ballot_sync(FULL_MASK, degree >= HOR_EDGE);
        while (process_mask != 0u) {
            const unsigned leader = find_ms_bit(process_mask);
            const int leader_first_edge =
                warp_shfl(FULL_MASK, first_edge, leader);
            const int leader_degree =
                warp_shfl(FULL_MASK, degree, leader);
            const int leader_vertex =
                warp_shfl(FULL_MASK, vertex, leader);

            for (int offset = static_cast<int>(get_lane_id());
                 offset < leader_degree;
                 offset += static_cast<int>(ADS_WARP_SIZE)) {
                const index_type edge = static_cast<index_type>(
                    leader_first_edge + offset);
                const index_type destination = graph.edge_dst[edge];
                const edge_data_type weight = graph.edge_data[edge];
                const node_data_type new_distance =
                    global_load(&graph.node_data[leader_vertex]) + weight;
                const node_data_type old_distance =
                    global_load(&graph.node_data[destination]);

                if (old_distance > new_distance &&
                    atomicMin_float(&graph.node_data[destination],
                                    new_distance) > new_distance) {
                    const unsigned destination_bag =
                        wl.dist_to_bag_id_int(source_bag_id, new_distance);
                    wl.push_work(destination_bag, destination);
                }
            }
            process_mask = set_bits(process_mask, 0u, leader, 1u);
        }
        if (degree >= HOR_EDGE) {
            degree = 0;
        }

        // Pack edges from low-degree vertices into contiguous wave-sized
        // batches.  This replaces CUB WarpScan with a native HIP shuffle scan.
        warp_sync();
        int warp_offset = 0;
        int warp_total = 0;
        warp_exclusive_sum(degree, warp_offset, warp_total);

        for (int write_index = warp_offset;
             write_index < warp_offset + degree; ++write_index) {
            leader_lane_storage[write_index] =
                static_cast<unsigned char>(get_lane_id());
        }
        warp_sync();

        for (int read_start = 0; read_start < warp_total;
             read_start += static_cast<int>(ADS_WARP_SIZE)) {
            const int read_index =
                read_start + static_cast<int>(get_lane_id());
            // Do not use an uninitialized byte as a shuffle source for lanes
            // beyond warp_total in the final partial batch.
            int leader = 0;
            if (read_index < warp_total) {
                leader = static_cast<int>(
                    leader_lane_storage[read_index]);
            }

            const int leader_warp_offset =
                warp_shfl(FULL_MASK, warp_offset,
                          static_cast<unsigned>(leader));
            const int leader_first_edge =
                warp_shfl(FULL_MASK, first_edge,
                          static_cast<unsigned>(leader));
            const int leader_vertex =
                warp_shfl(FULL_MASK, vertex,
                          static_cast<unsigned>(leader));

            if (read_index < warp_total) {
                const int edge_offset = read_index - leader_warp_offset;
                const index_type edge = static_cast<index_type>(
                    leader_first_edge + edge_offset);
                const index_type destination = graph.edge_dst[edge];
                const edge_data_type weight = graph.edge_data[edge];
                const node_data_type new_distance =
                    global_load(&graph.node_data[leader_vertex]) + weight;
                const node_data_type old_distance =
                    global_load(&graph.node_data[destination]);

                if (old_distance > new_distance &&
                    atomicMin_float(&graph.node_data[destination],
                                    new_distance) > new_distance) {
                    const unsigned destination_bag =
                        wl.dist_to_bag_id_int(source_bag_id, new_distance);
                    wl.push_work(destination_bag, destination);
                }
            }
        }
        warp_sync();
        wl.epilog(assignment, num, false);
    }
}

// HIP on AMD does not provide CUDA device-side kernel launches. The manager
// and workers are therefore launched by the host into independent streams.
// On AMD runtimes that expose CU-mask streams, the two persistent kernels are
// pinned to disjoint CU sets. The fallback still limits aggregate worker
// occupancy so at least one CU of capacity remains for the manager.
struct PersistentStreams {
    hipStream_t control = nullptr;
    hipStream_t manager = nullptr;
    hipStream_t worker = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t manager_done = nullptr;
    hipEvent_t worker_done = nullptr;
    hipEvent_t stop = nullptr;
    bool cu_partitioned = false;

    void create(int compute_units, unsigned reserved_cus) {
        HIP_CHECK(hipStreamCreateWithFlags(&control, hipStreamNonBlocking));

        const unsigned mask_words =
            (static_cast<unsigned>(compute_units) + 31u) / 32u;
        std::vector<std::uint32_t> manager_mask(mask_words, 0u);
        std::vector<std::uint32_t> worker_mask(mask_words, 0u);
        const unsigned worker_cus =
            static_cast<unsigned>(compute_units) - reserved_cus;
        for (unsigned cu = 0; cu < static_cast<unsigned>(compute_units); ++cu) {
            std::vector<std::uint32_t>& mask =
                cu < worker_cus ? worker_mask : manager_mask;
            mask[cu / 32u] |= (1u << (cu % 32u));
        }

        const hipError_t manager_status = hipExtStreamCreateWithCUMask(
            &manager, mask_words, manager_mask.data());
        hipError_t worker_status = manager_status;
        if (manager_status == hipSuccess) {
            worker_status = hipExtStreamCreateWithCUMask(
                &worker, mask_words, worker_mask.data());
        }

        if (manager_status == hipSuccess && worker_status == hipSuccess) {
            cu_partitioned = true;
        } else {
            if (manager != nullptr) {
                HIP_CHECK(hipStreamDestroy(manager));
                manager = nullptr;
            }
            if (worker != nullptr) {
                HIP_CHECK(hipStreamDestroy(worker));
                worker = nullptr;
            }
            // Clear any extension error retained by the runtime before using
            // ordinary nonblocking streams as a compatibility fallback.
            (void)hipGetLastError();
            HIP_CHECK(hipStreamCreateWithFlags(&manager,
                                               hipStreamNonBlocking));
            HIP_CHECK(hipStreamCreateWithFlags(&worker,
                                               hipStreamNonBlocking));
            std::fprintf(
                stderr,
                "warning: HIP CU-mask streams are unavailable; using "
                "occupancy-limited concurrent streams instead\n");
        }

        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreateWithFlags(&manager_done,
                                          hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&worker_done,
                                          hipEventDisableTiming));
        HIP_CHECK(hipEventCreate(&stop));
    }

    void destroy() {
        if (start != nullptr) HIP_CHECK(hipEventDestroy(start));
        if (manager_done != nullptr) HIP_CHECK(hipEventDestroy(manager_done));
        if (worker_done != nullptr) HIP_CHECK(hipEventDestroy(worker_done));
        if (stop != nullptr) HIP_CHECK(hipEventDestroy(stop));
        if (control != nullptr) HIP_CHECK(hipStreamDestroy(control));
        if (manager != nullptr) HIP_CHECK(hipStreamDestroy(manager));
        if (worker != nullptr) HIP_CHECK(hipStreamDestroy(worker));
        control = manager = worker = nullptr;
        start = manager_done = worker_done = stop = nullptr;
    }
};

float run_manager_and_workers(CSRGraph graph, unsigned source, int num_blocks,
                              worklist wl, unsigned* work_count,
                              PersistentStreams& streams) {
    // Release both persistent kernels from one start event, then join both on
    // the control stream. This measures the full manager/worker lifetime.
    HIP_CHECK(hipEventRecord(streams.start, streams.control));
    HIP_CHECK(hipStreamWaitEvent(streams.manager, streams.start, 0));
    HIP_CHECK(hipStreamWaitEvent(streams.worker, streams.start, 0));

    hipLaunchKernelGGL(wl_kernel, dim3(1), dim3(192), 0, streams.manager, wl,
                       static_cast<int>(source));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(streams.manager_done, streams.manager));

    hipLaunchKernelGGL(sssp_kernel, dim3(num_blocks), dim3(TB_SIZE), 0,
                       streams.worker, graph, wl, work_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(streams.worker_done, streams.worker));

    HIP_CHECK(hipStreamWaitEvent(streams.control, streams.manager_done, 0));
    HIP_CHECK(hipStreamWaitEvent(streams.control, streams.worker_done, 0));
    HIP_CHECK(hipEventRecord(streams.stop, streams.control));
    HIP_CHECK(hipEventSynchronize(streams.stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, streams.start, streams.stop));
    return elapsed_ms;
}

float compute_bucket_weight_scale(const CSRGraph& graph) {
    double sum = 0.0;
    for (index_type edge = 0; edge < graph.nedges; ++edge) {
        const float weight = graph.edge_data[edge];
        if (!std::isfinite(weight) || weight < 0.0f) {
            std::fprintf(stderr,
                         "ADDS requires finite nonnegative edge weights; "
                         "edge %u has weight %.9g\n",
                         edge, weight);
            std::exit(EXIT_FAILURE);
        }
        sum += static_cast<double>(weight);
    }

    const double mean = sum / static_cast<double>(graph.nedges);
    // quantum = mean / 2^QUAN_SHIFT must remain representable as float.  A
    // floor changes only bucket granularity, not relaxation correctness.
    const double minimum_mean =
        static_cast<double>(std::numeric_limits<float>::min()) *
        static_cast<double>(1u << QUAN_SHIFT);
    const double adjusted = std::max(mean, minimum_mean);
    if (adjusted != mean) {
        std::fprintf(stderr,
                     "warning: mean edge weight %.9g is too small for the "
                     "ADDS quantizer; using %.9g for bucket placement\n",
                     mean, adjusted);
    }
    return static_cast<float>(adjusted);
}

int benchmark_run_count() {
    const char* value = std::getenv("ADDS_RUNS");
    if (value == nullptr || *value == '\0') {
        return DEFAULT_RUNS;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 1000) {
        std::fprintf(stderr,
                     "warning: ignoring invalid ADDS_RUNS='%s'; using %d\n",
                     value, DEFAULT_RUNS);
        return DEFAULT_RUNS;
    }
    return static_cast<int>(parsed);
}

void append_benchmark_summary(const char* graph_name, double time_ms,
                              unsigned long long work) {
    FILE* fd3 = std::fopen("/dev/fd/3", "a");
    if (fd3 != nullptr) {
        std::fprintf(fd3, "%s %.3f %llu\n", graph_name, time_ms, work);
        std::fclose(fd3);
        return;
    }
    std::fprintf(stderr,
                 "warning: file descriptor 3 is not open; benchmark "
                 "summary: %s %.3f %llu\n",
                 graph_name, time_ms, work);
}

void initialize_only(CSRGraph& host_graph, CSRGraph& device_graph,
                     const hipDeviceProp_t& properties) {
    const int blocks = std::max(1, properties.multiProcessorCount * 2);
    hipLaunchKernelGGL(initialize_distances, dim3(blocks), dim3(256), 0, 0,
                       device_graph, start_node);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("%s has no edges; initialized source distance only\n",
                host_graph.file_name);
    append_benchmark_summary(host_graph.file_name, 0.0, 1u);
}

void gg_main(CSRGraph& host_graph, CSRGraph& device_graph) {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, HIP_DEVICE));

    if (properties.warpSize != static_cast<int>(ADS_WARP_SIZE)) {
        std::fprintf(stderr,
                     "This ADDS port requires wave32, but device %d reports "
                     "warpSize=%d\n",
                     HIP_DEVICE, properties.warpSize);
        std::exit(EXIT_FAILURE);
    }
    if (!properties.concurrentKernels) {
        std::fprintf(stderr,
                     "The selected device/runtime does not report concurrent "
                     "kernel support. ADDS requires its manager and workers "
                     "to run at the same time.\n");
        std::exit(EXIT_FAILURE);
    }
    if (properties.maxThreadsPerBlock < TB_SIZE) {
        std::fprintf(stderr,
                     "The selected device supports only %d threads per block; "
                     "ADDS requires %d.\n",
                     properties.maxThreadsPerBlock, TB_SIZE);
        std::exit(EXIT_FAILURE);
    }
    if (properties.multiProcessorCount < 2) {
        std::fprintf(stderr,
                     "ADDS needs at least two compute units so one unit of "
                     "capacity can be reserved for the persistent manager.\n");
        std::exit(EXIT_FAILURE);
    }
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7) != 0) {
        std::fprintf(stderr,
                     "warning: this build is tuned for gfx1151, while device "
                     "%d reports '%s'\n",
                     HIP_DEVICE, properties.gcnArchName);
    }

    std::printf("device: %s (%s), CUs=%d, warpSize=%d\n", properties.name,
                properties.gcnArchName, properties.multiProcessorCount,
                properties.warpSize);

    if (host_graph.nedges == 0) {
        initialize_only(host_graph, device_graph, properties);
        return;
    }

    int worker_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &worker_blocks_per_cu, sssp_kernel, TB_SIZE, 0));
    int manager_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &manager_blocks_per_cu, wl_kernel, 192, 0));
    if (worker_blocks_per_cu < 1 || manager_blocks_per_cu < 1) {
        std::fprintf(stderr,
                     "The compiled kernels have zero reported occupancy "
                     "(manager=%d, worker=%d).\n",
                     manager_blocks_per_cu, worker_blocks_per_cu);
        std::exit(EXIT_FAILURE);
    }

    const unsigned reserved_cus = read_env_unsigned(
        "ADDS_RESERVED_CUS", 1u, 1u,
        static_cast<unsigned>(properties.multiProcessorCount - 1));
    const unsigned safe_worker_blocks = static_cast<unsigned>(
        worker_blocks_per_cu *
        (properties.multiProcessorCount - static_cast<int>(reserved_cus)));
    const unsigned capped_worker_blocks =
        std::min(safe_worker_blocks, static_cast<unsigned>(MAX_TB - 1));
    const unsigned num_blocks = read_env_unsigned(
        "ADDS_WORKER_BLOCKS", capped_worker_blocks, 1u,
        capped_worker_blocks);
    const int waves_per_block = TB_SIZE / static_cast<int>(ADS_WARP_SIZE);

    std::printf(
        "occupancy: manager=%d block(s)/CU, worker=%d block(s)/CU; "
        "worker block=%d threads (%d waves), worker blocks=%u, "
        "reserved CUs=%u%s\n",
        manager_blocks_per_cu, worker_blocks_per_cu, TB_SIZE,
        waves_per_block, num_blocks, reserved_cus,
        capped_worker_blocks < safe_worker_blocks ? " (MAX_TB cap)" : "");

    PersistentStreams streams;
    streams.create(properties.multiProcessorCount, reserved_cus);
    std::printf("persistent streams: %s\n",
                streams.cu_partitioned
                    ? "manager/worker CU masks enabled"
                    : "occupancy-limited fallback");

    constexpr double worklist_size_multiplier = 1.5;
    constexpr std::uint64_t maximum_entries = 536870912ull;
    const std::uint64_t minimum_entries =
        static_cast<std::uint64_t>(NUM_BAG) * BLOCK_SIZE * 8u;
    const std::uint64_t requested_entries =
        static_cast<std::uint64_t>(
            std::ceil(static_cast<double>(host_graph.nedges) *
                      worklist_size_multiplier)) +
        minimum_entries;
    const unsigned default_worklist_entries = static_cast<unsigned>(
        std::min(requested_entries, maximum_entries));
    const unsigned suggested_size = read_env_unsigned(
        "ADDS_WL_ENTRIES", default_worklist_entries,
        static_cast<unsigned>(minimum_entries),
        static_cast<unsigned>(maximum_entries));

    const std::size_t work_count_entries =
        static_cast<std::size_t>(num_blocks) * TB_SIZE;
    unsigned* work_count = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&work_count),
                        work_count_entries * sizeof(unsigned)));

    worklist wl;
    wl.alloc(static_cast<int>(num_blocks), waves_per_block, suggested_size,
             host_graph.nnodes,
             host_graph.nedges);

    const float average_degree =
        static_cast<float>(host_graph.nedges) /
        static_cast<float>(host_graph.nnodes);
    const float bucket_weight_scale = compute_bucket_weight_scale(host_graph);
    wl.set_param(bucket_weight_scale, average_degree);

    const int runs = benchmark_run_count();
    unsigned long long aggregate_work = 0;
    double aggregate_time_ms = 0.0;
    std::vector<unsigned> work_count_host(work_count_entries);
    const int initialization_blocks =
        std::max(1, properties.multiProcessorCount * 2);

    for (int run = 0; run < runs; ++run) {
        hipLaunchKernelGGL(initialize_distances,
                           dim3(initialization_blocks), dim3(256), 0, 0,
                           device_graph, start_node);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemset(work_count, 0,
                            work_count_entries * sizeof(unsigned)));
        wl.reinit();
        HIP_CHECK(hipDeviceSynchronize());

        const float elapsed_ms = run_manager_and_workers(
            device_graph, static_cast<unsigned>(start_node),
            static_cast<int>(num_blocks), wl, work_count, streams);

        HIP_CHECK(hipMemcpy(work_count_host.data(), work_count,
                            work_count_entries * sizeof(unsigned),
                            hipMemcpyDeviceToHost));
        unsigned long long total_work = 0;
        for (const unsigned value : work_count_host) {
            total_work += value;
        }

        aggregate_time_ms += elapsed_ms;
        aggregate_work += total_work;
        std::printf("%s run %d/%d: %.3f ms, processed vertices=%llu\n",
                    host_graph.file_name, run + 1, runs, elapsed_ms,
                    total_work);
    }

    const double average_time_ms = aggregate_time_ms / runs;
    const unsigned long long average_work = aggregate_work / runs;
    append_benchmark_summary(host_graph.file_name, average_time_ms,
                             average_work);

    streams.destroy();
    wl.free();
    HIP_CHECK(hipFree(work_count));
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    parse_args(argc, argv);

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (HIP_DEVICE < 0 || HIP_DEVICE >= device_count) {
        std::fprintf(stderr,
                     "GPU device %d is unavailable; HIP reports %d device(s)\n",
                     HIP_DEVICE, device_count);
        return EXIT_FAILURE;
    }
    HIP_CHECK(hipSetDevice(HIP_DEVICE));

    CSRGraphTy graph;
    CSRGraphTy device_graph;
    if (graph.read(INPUT) != 0) {
        graph.dealloc();
        return EXIT_FAILURE;
    }
    if (start_node < 0 ||
        static_cast<unsigned>(start_node) >= graph.nnodes) {
        std::fprintf(stderr, "Source vertex %d is outside [0, %u)\n",
                     start_node, graph.nnodes);
        graph.dealloc();
        return EXIT_FAILURE;
    }

    graph.copy_to_gpu(device_graph);
    gg_main(graph, device_graph);
    device_graph.copy_to_cpu(graph);
    output(graph, OUTPUT);

    device_graph.dealloc();
    graph.dealloc();
    return EXIT_SUCCESS;
}
