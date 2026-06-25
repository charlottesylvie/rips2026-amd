/*
 * CSR graph interface derived from the supplied GGC/LonestarGPU source.
 * Original notice: Copyright (C) 2014--2016, The University of Texas at Austin.
 * See the upstream source tree's LICENSE.TXT before redistributing.
 * Original author: Sreepathi Pai.
 */

#ifndef ADS_FLOAT_HIP_CSR_GRAPH_H_
#define ADS_FLOAT_HIP_CSR_GRAPH_H_

#include "common.h"

#include <cfloat>
#include <cstdint>

using index_type = unsigned;
using edge_data_type = float;
using node_data_type = float;

constexpr node_data_type INF = FLT_MAX;

// ADDS accepts finite, non-negative weights. For those values, a CAS loop is
// sufficient and avoids depending on architecture-specific floating-point
// atomic-min instructions.
__device__ __forceinline__ float atomicMin_float(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = atomicCAS(address_as_int, 0, 0);  // atomic read
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
    return __int_as_float(old);
}

struct CSRGraph {
    CSRGraph();

    unsigned init();
    unsigned read(const char file[]);
    unsigned readFromGR(const char file[]);

    unsigned allocOnHost();
    unsigned allocOnDevice();
    unsigned deallocOnHost();
    unsigned deallocOnDevice();
    void dealloc();

    void copy_to_gpu(CSRGraph& copygraph) const;
    void copy_to_cpu(CSRGraph& copygraph) const;
    void progressPrint(unsigned maxii, unsigned ii) const;

    __device__ __host__ bool valid_node(index_type node) const {
        return node < nnodes;
    }
    __device__ __host__ bool valid_edge(index_type edge) const {
        return edge < nedges;
    }
    __device__ __host__ index_type getOutDegree(unsigned src) const {
        return row_start[src + 1] - row_start[src];
    }
    __device__ __host__ index_type getDestination(unsigned src,
                                                   unsigned edge) const {
        return edge_dst[row_start[src] + edge];
    }
    __device__ __host__ index_type getAbsDestination(unsigned edge) const {
        return edge_dst[edge];
    }
    __device__ __host__ index_type getFirstEdge(unsigned src) const {
        return row_start[src];
    }
    __device__ __host__ edge_data_type getWeight(unsigned src,
                                                  unsigned edge) const {
        return edge_data[row_start[src] + edge];
    }
    __device__ __host__ edge_data_type getAbsWeight(unsigned edge) const {
        return edge_data[edge];
    }

    index_type nnodes = 0;
    index_type nedges = 0;
    index_type* row_start = nullptr;
    index_type* edge_dst = nullptr;
    edge_data_type* edge_data = nullptr;
    node_data_type* node_data = nullptr;
    bool device_graph = false;
    char file_name[256]{};
};

using CSRGraphTy = CSRGraph;

#endif  // ADS_FLOAT_HIP_CSR_GRAPH_H_
