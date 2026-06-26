#include "csr_graph.h"

#include <endian.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

const char* basename_ptr(const char* path) {
    const char* slash = std::strrchr(path, '/');
    return slash == nullptr ? path : slash + 1;
}

template <typename T>
bool range_fits(const std::byte* begin, const std::byte* end,
                const T* ptr, std::size_t count) {
    const auto* p = reinterpret_cast<const std::byte*>(ptr);
    const std::size_t bytes = count * sizeof(T);
    return p >= begin && p <= end && bytes <= static_cast<std::size_t>(end - p);
}

}  // namespace

CSRGraph::CSRGraph() {
    init();
}

unsigned CSRGraph::init() {
    row_start = nullptr;
    edge_dst = nullptr;
    edge_data = nullptr;
    node_data = nullptr;
    nnodes = 0;
    nedges = 0;
    device_graph = false;
    file_name[0] = '\0';
    return 0;
}

unsigned CSRGraph::allocOnHost() {
    assert(nnodes > 0);
    assert(!device_graph);

    if (row_start != nullptr) {
        return true;
    }

    row_start = static_cast<index_type*>(
        std::calloc(static_cast<std::size_t>(nnodes) + 1, sizeof(index_type)));
    edge_dst = static_cast<index_type*>(
        std::calloc(static_cast<std::size_t>(nedges), sizeof(index_type)));
    edge_data = static_cast<edge_data_type*>(
        std::calloc(static_cast<std::size_t>(nedges), sizeof(edge_data_type)));
    node_data = static_cast<node_data_type*>(
        std::calloc(static_cast<std::size_t>(nnodes), sizeof(node_data_type)));

    const std::size_t mem_usage =
        (static_cast<std::size_t>(nnodes) + 1 + nedges) * sizeof(index_type) +
        static_cast<std::size_t>(nedges) * sizeof(edge_data_type) +
        static_cast<std::size_t>(nnodes) * sizeof(node_data_type);
    std::printf("Host memory for graph: %.2f MiB\n",
                static_cast<double>(mem_usage) / (1024.0 * 1024.0));

    return row_start != nullptr && node_data != nullptr &&
           (nedges == 0 || (edge_dst != nullptr && edge_data != nullptr));
}

unsigned CSRGraph::allocOnDevice() {
    if (row_start != nullptr) {
        return true;
    }

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&row_start),
                        (static_cast<std::size_t>(nnodes) + 1) *
                            sizeof(index_type)));
    if (nedges > 0) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&edge_dst),
                            static_cast<std::size_t>(nedges) *
                                sizeof(index_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&edge_data),
                            static_cast<std::size_t>(nedges) *
                                sizeof(edge_data_type)));
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&node_data),
                        static_cast<std::size_t>(nnodes) *
                            sizeof(node_data_type)));
    device_graph = true;
    return true;
}

unsigned CSRGraph::deallocOnHost() {
    if (!device_graph) {
        std::free(row_start);
        std::free(edge_dst);
        std::free(edge_data);
        std::free(node_data);
        row_start = nullptr;
        edge_dst = nullptr;
        edge_data = nullptr;
        node_data = nullptr;
    }
    return 0;
}

unsigned CSRGraph::deallocOnDevice() {
    if (device_graph) {
        if (edge_dst != nullptr) HIP_CHECK(hipFree(edge_dst));
        if (edge_data != nullptr) HIP_CHECK(hipFree(edge_data));
        if (row_start != nullptr) HIP_CHECK(hipFree(row_start));
        if (node_data != nullptr) HIP_CHECK(hipFree(node_data));
        row_start = nullptr;
        edge_dst = nullptr;
        edge_data = nullptr;
        node_data = nullptr;
    }
    return 0;
}

void CSRGraph::dealloc() {
    if (device_graph) {
        deallocOnDevice();
    } else {
        deallocOnHost();
    }
}

void CSRGraph::progressPrint(unsigned maxii, unsigned ii) const {
    if (maxii == 0) {
        return;
    }
    constexpr unsigned steps = 10;
    const unsigned each = (maxii / steps) == 0 ? 1 : (maxii / steps);
    if ((ii % each) == 0) {
        const int progress =
            static_cast<int>((static_cast<std::size_t>(ii) * 100) / maxii + 1);
        std::printf("\t%3d%%\r", progress);
        std::fflush(stdout);
    }
}

unsigned CSRGraph::readFromGR(const char file[]) {
    const int fd = open(file, O_RDONLY);
    if (fd == -1) {
        std::fprintf(stderr, "Unable to open %s: %s\n", file,
                     std::strerror(errno));
        return 1;
    }

    struct stat st {};
    if (fstat(fd, &st) == -1) {
        std::fprintf(stderr, "Unable to stat %s: %s\n", file,
                     std::strerror(errno));
        close(fd);
        return 1;
    }

    const std::size_t length = static_cast<std::size_t>(st.st_size);
    void* mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed for %s: %s\n", file,
                     std::strerror(errno));
        close(fd);
        return 1;
    }

    const auto* begin = static_cast<const std::byte*>(mapping);
    const auto* end = begin + length;
    const auto* header = reinterpret_cast<const std::uint64_t*>(begin);
    if (!range_fits(begin, end, header, 4)) {
        std::fprintf(stderr, "%s is too small to be a .gr graph\n", file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    const std::uint64_t version = le64toh(header[0]);
    const std::uint64_t edge_type_size = le64toh(header[1]);
    const std::uint64_t num_nodes64 = le64toh(header[2]);
    const std::uint64_t num_edges64 = le64toh(header[3]);

    if (version != 1) {
        std::fprintf(stderr, "Unsupported .gr version %llu in %s\n",
                     static_cast<unsigned long long>(version), file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }
    if (edge_type_size != 0 && edge_type_size != sizeof(float)) {
        std::fprintf(stderr,
                     "Expected unweighted or float-weighted .gr input; "
                     "%s reports %llu-byte edge data\n",
                     file, static_cast<unsigned long long>(edge_type_size));
        munmap(mapping, length);
        close(fd);
        return 1;
    }
    if (num_nodes64 == 0 || num_nodes64 > INT_MAX ||
        num_edges64 > INT_MAX) {
        std::fprintf(stderr,
                     "%s exceeds this implementation's signed 32-bit kernel limits\n",
                     file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    const auto* out_idx = header + 4;
    if (!range_fits(begin, end, out_idx,
                    static_cast<std::size_t>(num_nodes64))) {
        std::fprintf(stderr, "Truncated row-offset table in %s\n", file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    const auto* outs = reinterpret_cast<const std::uint32_t*>(
        out_idx + static_cast<std::size_t>(num_nodes64));
    if (!range_fits(begin, end, outs,
                    static_cast<std::size_t>(num_edges64))) {
        std::fprintf(stderr, "Truncated destination table in %s\n", file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    const auto* after_outs = outs + static_cast<std::size_t>(num_edges64);
    if ((num_edges64 & 1ull) != 0) {
        ++after_outs;  // Galois aligns edge data to 64 bits.
    }
    const auto* weights = reinterpret_cast<const float*>(after_outs);
    if (edge_type_size != 0 &&
        !range_fits(begin, end, weights,
                    static_cast<std::size_t>(num_edges64))) {
        std::fprintf(stderr, "Truncated edge-weight table in %s\n", file);
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    nnodes = static_cast<index_type>(num_nodes64);
    nedges = static_cast<index_type>(num_edges64);
    std::printf("nnodes=%u, nedges=%u\n", nnodes, nedges);
    if (!allocOnHost()) {
        std::fprintf(stderr, "Host graph allocation failed\n");
        munmap(mapping, length);
        close(fd);
        return 1;
    }

    row_start[0] = 0;
    bool valid = true;
    for (unsigned node = 0; node < nnodes; ++node) {
        const std::uint64_t row_end64 = le64toh(out_idx[node]);
        if (row_end64 > nedges || row_end64 < row_start[node]) {
            std::fprintf(stderr, "Invalid row offset for node %u in %s\n",
                         node, file);
            valid = false;
            break;
        }
        row_start[node + 1] = static_cast<index_type>(row_end64);
        for (index_type edge = row_start[node]; edge < row_start[node + 1];
             ++edge) {
            const unsigned dst = le32toh(outs[edge]);
            if (dst >= nnodes) {
                std::fprintf(stderr,
                             "Invalid edge from %u to %u at edge %u\n", node,
                             dst, edge);
                valid = false;
                break;
            }
            edge_dst[edge] = dst;
            edge_data[edge] = edge_type_size == 0 ? 1.0f : weights[edge];
            if (!std::isfinite(edge_data[edge]) || edge_data[edge] < 0.0f) {
                std::fprintf(stderr,
                             "Invalid weight at edge %u; SSSP requires finite "
                             "nonnegative weights\n",
                             edge);
                valid = false;
                break;
            }
        }
        if (!valid) break;
        progressPrint(nnodes, node);
    }
    if (valid && row_start[nnodes] != nedges) {
        std::fprintf(stderr,
                     "Final row offset is %u but the header reports %u edges in %s\n",
                     row_start[nnodes], nedges, file);
        valid = false;
    }
    std::printf("\nread %zu bytes\n", length);

    std::snprintf(file_name, sizeof(file_name), "%s", basename_ptr(file));
    munmap(mapping, length);
    close(fd);
    return valid ? 0u : 1u;
}

unsigned CSRGraph::read(const char file[]) {
    return readFromGR(file);
}

void CSRGraph::copy_to_gpu(CSRGraph& copygraph) const {
    copygraph.nnodes = nnodes;
    copygraph.nedges = nedges;
    std::snprintf(copygraph.file_name, sizeof(copygraph.file_name), "%s",
                  file_name);
    assert(copygraph.allocOnDevice());

    if (nedges > 0) {
        HIP_CHECK(hipMemcpy(copygraph.edge_dst, edge_dst,
                            static_cast<std::size_t>(nedges) *
                                sizeof(index_type),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(copygraph.edge_data, edge_data,
                            static_cast<std::size_t>(nedges) *
                                sizeof(edge_data_type),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(copygraph.node_data, node_data,
                        static_cast<std::size_t>(nnodes) *
                            sizeof(node_data_type),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(copygraph.row_start, row_start,
                        (static_cast<std::size_t>(nnodes) + 1) *
                            sizeof(index_type),
                        hipMemcpyHostToDevice));
}

void CSRGraph::copy_to_cpu(CSRGraph& copygraph) const {
    assert(device_graph);
    assert(copygraph.nnodes == nnodes);
    assert(copygraph.nedges == nedges);

    if (nedges > 0) {
        HIP_CHECK(hipMemcpy(copygraph.edge_dst, edge_dst,
                            static_cast<std::size_t>(nedges) *
                                sizeof(index_type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(copygraph.edge_data, edge_data,
                            static_cast<std::size_t>(nedges) *
                                sizeof(edge_data_type),
                            hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipMemcpy(copygraph.node_data, node_data,
                        static_cast<std::size_t>(nnodes) *
                            sizeof(node_data_type),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(copygraph.row_start, row_start,
                        (static_cast<std::size_t>(nnodes) + 1) *
                            sizeof(index_type),
                        hipMemcpyDeviceToHost));
}
