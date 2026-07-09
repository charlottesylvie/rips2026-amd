// Warm-started single-source HIP Bellman-Ford for validator JSONL generation.
//
// This driver reads a RIPS .csrbin graph and .csrbin.ifmeta.bin sidecar,
// groups metadata route queries by unique source node, discovers hop-2 shared
// connector nodes, decides a connector/source processing order, and runs
// Bellman-Ford one source at a time. Connector SSSPs are internal only: their
// paths are not written to JSONL. A source can warm-start from a connector only
// when there is a valid directed edge:
//
//   source -> connector
//
// Then the source column is initialized as:
//
//   dist_source[v] = edge_weight(source -> connector) + dist_connector[v]
//
// The distance object is an N x 1 sparse CSR matrix. Each relaxation reuses the
// same graph CSR:
//
//   dist_next = min(dist_old, adjacency min-plus dist_old)
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I HIP_kernel/minplus_mm/src \
//     SSSP/src/bf_warm_init.cpp \
//     HIP_kernel/minplus_mm/src/minplus_sparse_hip.cpp \
//     -o bf_warm_init
//
// Run:
//   ./bf_warm_init data/logicnets_jscl.csrbin data/logicnets_jscl.csrbin.ifmeta.bin paths.jsonl

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR_device_utils.hpp"
#include "../../HIP_kernel/minplus_mm/src/minplus_sparse_hip.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr int kDistanceColumns = 1;

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;
using DeviceCsrOwner = bf_csr_detail::DeviceCsrOwner;

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = kNoIndex;
  std::uint64_t pin_string = kNoIndex;
};

struct RouteRequest {
  std::uint64_t net_string = kNoIndex;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::vector<std::string> strings;
  std::vector<std::uint64_t> node_device_ids;
  std::uint64_t edge_attr_count = 0;
  std::vector<RouteRequest> route_requests;
};

struct PathEdge {
  int from = -1;
  int to = -1;
  Offset csr_edge = -1;
  float cost = 0.0f;
};

struct Query {
  std::size_t net_index = 0;
  std::string net_name;
  int source = -1;
  int target = -1;
};

struct SourceWork {
  int source = -1;
  std::vector<Query> queries;
};

struct Options {
  int max_iters = -1;
  std::size_t min_connector_sources = 2;
  std::size_t max_connector_groups = 0;
  std::size_t net_limit = 0;
  std::size_t source_limit = 0;
  std::size_t query_limit = 0;
  int bf_progress_every = 0;
  bool show_bf_progress = true;
  bool connector_warm_start = true;
  double abs_tolerance = 1e-3;
  double rel_tolerance = 1e-5;
  bool include_unreached = true;
};

struct SsspResult {
  std::vector<float> dense_dist;  // one distance per graph row
  int iterations_used = 0;
  bool converged = false;
  Offset initial_nnz = 0;
};

struct InitialColumn {
  int source = -1;
  int cached_source = -1;
  Offset connecting_edge = -1;
  float connecting_cost = std::numeric_limits<float>::infinity();
  bool from_connector = false;
  const std::vector<float>* cached_dist = nullptr;

  bool has_warm_start() const {
    return cached_dist != nullptr && std::isfinite(connecting_cost);
  }
};

struct OutgoingCsr {
  std::vector<Offset> rowptr;
  std::vector<Index> to;
  std::vector<Offset> csr_edge;
};

struct ConnectorRelation {
  int connector = -1;
  std::size_t work_index = 0;
  int source = -1;
  Offset source_to_connector_edge = -1;
  float source_to_connector_cost = std::numeric_limits<float>::infinity();
  Offset connector_to_source_edge = -1;
  float connector_to_source_cost = std::numeric_limits<float>::infinity();

  bool can_warm_start_from_connector() const {
    return source_to_connector_edge >= 0 &&
           std::isfinite(source_to_connector_cost);
  }
};

struct ConnectorCandidate {
  int connector = -1;
  std::vector<ConnectorRelation> relations;
  std::size_t warmable_count = 0;
};

struct ConnectorPlan {
  int connector = -1;
  std::vector<ConnectorRelation> relations;
};

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " count is too large for this host");
  }
  values.resize(static_cast<std::size_t>(count));
  if (values.empty()) return;

  const std::size_t bytes = values.size() * sizeof(T);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) return;
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) + " byte count is too large to seek");
  }
  in.seekg(static_cast<std::streamoff>(count), std::ios::cur);
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

std::string read_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("metadata string is too large for this host");
  }

  std::string text(static_cast<std::size_t>(size), '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed while reading metadata string bytes");
    }
  }
  return text;
}

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

constexpr unsigned kGridX = 65535u;

dim3 grid_for_items(Offset items, int block_size) {
  if (items <= 0) return dim3(1, 1);
  const Offset blocks_needed = (items + block_size - 1) / block_size;
  const unsigned gx =
      blocks_needed < static_cast<Offset>(kGridX)
          ? static_cast<unsigned>(blocks_needed)
          : kGridX;
  const unsigned gy =
      static_cast<unsigned>((blocks_needed + static_cast<Offset>(gx) - 1) /
                            static_cast<Offset>(gx));
  return dim3(gx, gy);
}

__device__ __forceinline__ Offset logical_thread_id_1d() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

__device__ __forceinline__ bool finite_distance(float value) {
  return value == value && value != INFINITY;
}

__device__ __forceinline__ int popcount_u64(unsigned long long value) {
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

__global__ void batched_merge_count_kernel(Offset rows,
                                           Offset cols,
                                           const Offset* old_rowptr,
                                           const Index* old_colind,
                                           const float* old_values,
                                           const Offset* relaxed_rowptr,
                                           const Index* relaxed_colind,
                                           const float* relaxed_values,
                                           Offset* row_counts,
                                           int* changed,
                                           int* status) {
  const Offset row = logical_thread_id_1d();
  if (row >= rows) return;

  unsigned long long mask = 0;
  const Offset old0 = old_rowptr[row];
  const Offset old1 = old_rowptr[row + 1];
  const Offset relaxed0 = relaxed_rowptr[row];
  const Offset relaxed1 = relaxed_rowptr[row + 1];

  for (Offset p = old0; p < old1; ++p) {
    const Index col = old_colind[p];
    if (col < 0 || static_cast<Offset>(col) >= cols) {
      atomicExch(status, 1);
      continue;
    }
    if (finite_distance(old_values[p])) {
      mask |= 1ULL << static_cast<unsigned>(col);
    }
  }

  for (Offset p = relaxed0; p < relaxed1; ++p) {
    const Index col = relaxed_colind[p];
    if (col < 0 || static_cast<Offset>(col) >= cols) {
      atomicExch(status, 1);
      continue;
    }

    const float relaxed_value = relaxed_values[p];
    if (!finite_distance(relaxed_value)) {
      continue;
    }

    mask |= 1ULL << static_cast<unsigned>(col);

    float old_best = INFINITY;
    for (Offset q = old0; q < old1; ++q) {
      if (old_colind[q] == col && old_values[q] < old_best) {
        old_best = old_values[q];
      }
    }
    if (relaxed_value < old_best) {
      atomicExch(changed, 1);
    }
  }

  row_counts[row] = static_cast<Offset>(popcount_u64(mask));
}

__global__ void set_last_rowptr_kernel(const Offset* row_counts,
                                       Offset* rowptr,
                                       Offset rows) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    rowptr[rows] = rows == 0 ? 0 : rowptr[rows - 1] + row_counts[rows - 1];
  }
}

__global__ void batched_merge_write_kernel(Offset rows,
                                           Offset cols,
                                           const Offset* old_rowptr,
                                           const Index* old_colind,
                                           const float* old_values,
                                           const Offset* relaxed_rowptr,
                                           const Index* relaxed_colind,
                                           const float* relaxed_values,
                                           const Offset* merged_rowptr,
                                           Index* merged_colind,
                                           float* merged_values) {
  const Offset row = logical_thread_id_1d();
  if (row >= rows) return;

  const Offset old0 = old_rowptr[row];
  const Offset old1 = old_rowptr[row + 1];
  const Offset relaxed0 = relaxed_rowptr[row];
  const Offset relaxed1 = relaxed_rowptr[row + 1];
  Offset out = merged_rowptr[row];

  for (Index col = 0; static_cast<Offset>(col) < cols; ++col) {
    float best = INFINITY;
    for (Offset p = old0; p < old1; ++p) {
      if (old_colind[p] == col && old_values[p] < best) {
        best = old_values[p];
      }
    }
    for (Offset p = relaxed0; p < relaxed1; ++p) {
      if (relaxed_colind[p] == col && relaxed_values[p] < best) {
        best = relaxed_values[p];
      }
    }

    if (finite_distance(best)) {
      merged_colind[out] = col;
      merged_values[out] = best;
      ++out;
    }
  }
}

void validate_csr(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }

  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

HostCsrF32 load_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(in, "CSR format version");
  const std::uint64_t orientation = read_u64(in, "CSR orientation");
  if (version != EXPECTED_CSR_VERSION) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "CSR row count");
  const std::uint64_t cols = read_u64(in, "CSR column count");
  (void)read_u64(in, "declared edge count");
  (void)read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(in, "CSR colind count");
  const std::uint64_t values_count = read_u64(in, "CSR values count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(in, graph.colind, colind_count, "CSR colind");
  read_array(in, graph.values, values_count, "CSR values");
  validate_csr(graph);
  return graph;
}

RoutingMetadata load_interchange_metadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_INCOMING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
  const std::uint64_t pip_data_count = read_u64(in, "metadata pip data count");
  const std::uint64_t site_pin_attr_count = read_u64(in, "metadata site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "metadata route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "metadata sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "metadata logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "metadata logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "metadata physical byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "metadata logical byte count");

  (void)read_u64(in, "metadata device path string");
  (void)read_u64(in, "metadata physical path string");
  (void)read_u64(in, "metadata logical path string");
  (void)read_u64(in, "metadata logical design name");

  RoutingMetadata metadata;
  metadata.edge_attr_count = edge_attr_count;
  metadata.strings.reserve(static_cast<std::size_t>(string_count));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }

  read_array(in, metadata.node_device_ids, node_count, "metadata device node ids");
  skip_bytes(in, edge_attr_count * 2 * sizeof(std::uint64_t), "metadata edge attrs");
  skip_bytes(in, pip_data_count * 3 * sizeof(std::uint64_t), "metadata pip data");
  skip_bytes(in,
             site_pin_attr_count * 3 * sizeof(std::uint64_t),
             "metadata site pin attrs");

  metadata.route_requests.resize(static_cast<std::size_t>(route_request_count));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");

    const std::uint64_t source_count = read_u64(in, "metadata source count");
    request.sources.resize(static_cast<std::size_t>(source_count));
    for (SitePinNode& source : request.sources) {
      source.node = static_cast<int>(read_u64(in, "metadata source node"));
      source.site_string = read_u64(in, "metadata source site");
      source.pin_string = read_u64(in, "metadata source pin");
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    request.sinks.resize(static_cast<std::size_t>(sink_count));
    for (SitePinNode& sink : request.sinks) {
      sink.node = static_cast<int>(read_u64(in, "metadata sink node"));
      sink.site_string = read_u64(in, "metadata sink site");
      sink.pin_string = read_u64(in, "metadata sink pin");
    }
  }

  skip_bytes(in, logical_cell_count * 3 * sizeof(std::uint64_t), "metadata logical cells");
  skip_bytes(in, logical_net_count * 4 * sizeof(std::uint64_t), "metadata logical nets");
  skip_bytes(in,
             logical_port_instance_count * 7 * sizeof(std::uint64_t),
             "metadata logical port instances");
  skip_bytes(in, blocked_node_count * sizeof(std::uint64_t), "metadata blocked nodes");
  skip_bytes(in, sink_stop_node_count * sizeof(std::uint64_t), "metadata sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "metadata physical bytes");
  skip_bytes(in, logical_netlist_byte_count, "metadata logical bytes");

  return metadata;
}

bool valid_node(const HostCsrF32& graph, int node) {
  return node >= 0 && static_cast<Offset>(node) < graph.rows;
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) return "";
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

double tolerance_for(double lhs, double rhs, const Options& options) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return options.abs_tolerance + options.rel_tolerance * scale;
}

OutgoingCsr build_outgoing_csr(const HostCsrF32& graph) {
  std::cout << "[bf_warm_init] building outgoing adjacency for connector discovery\n";
  const auto start = std::chrono::steady_clock::now();

  OutgoingCsr out;
  out.rowptr.assign(static_cast<std::size_t>(graph.rows + 1), 0);
  out.to.resize(static_cast<std::size_t>(graph.nnz));
  out.csr_edge.resize(static_cast<std::size_t>(graph.nnz));

  for (Offset edge = 0; edge < graph.nnz; ++edge) {
    const int from = graph.colind[static_cast<std::size_t>(edge)];
    ++out.rowptr[static_cast<std::size_t>(from + 1)];
  }
  for (Offset row = 0; row < graph.rows; ++row) {
    out.rowptr[static_cast<std::size_t>(row + 1)] +=
        out.rowptr[static_cast<std::size_t>(row)];
  }

  std::vector<Offset> cursor = out.rowptr;
  for (Offset to_node = 0; to_node < graph.rows; ++to_node) {
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(to_node)];
         edge < graph.rowptr[static_cast<std::size_t>(to_node + 1)];
         ++edge) {
      const int from = graph.colind[static_cast<std::size_t>(edge)];
      const Offset out_pos = cursor[static_cast<std::size_t>(from)]++;
      out.to[static_cast<std::size_t>(out_pos)] = static_cast<Index>(to_node);
      out.csr_edge[static_cast<std::size_t>(out_pos)] = edge;
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << "[bf_warm_init] outgoing adjacency built in "
            << std::fixed << std::setprecision(3) << seconds
            << " sec\n";
  return out;
}

void add_connector_relation(std::vector<ConnectorRelation>* relations,
                            int connector,
                            std::size_t work_index,
                            int source,
                            Offset source_to_connector_edge,
                            float source_to_connector_cost,
                            Offset connector_to_source_edge,
                            float connector_to_source_cost) {
  if (connector == source) return;
  ConnectorRelation relation;
  relation.connector = connector;
  relation.work_index = work_index;
  relation.source = source;
  relation.source_to_connector_edge = source_to_connector_edge;
  relation.source_to_connector_cost = source_to_connector_cost;
  relation.connector_to_source_edge = connector_to_source_edge;
  relation.connector_to_source_cost = connector_to_source_cost;
  relations->push_back(relation);
}

std::vector<ConnectorRelation> collect_connector_relations(const HostCsrF32& graph,
                                                           const OutgoingCsr& outgoing,
                                                           const std::vector<SourceWork>& work,
                                                           std::size_t source_count) {
  std::vector<ConnectorRelation> relations;
  for (std::size_t work_index = 0; work_index < source_count; ++work_index) {
    const int source = work[work_index].source;

    const Offset outgoing_begin = outgoing.rowptr[static_cast<std::size_t>(source)];
    const Offset outgoing_end = outgoing.rowptr[static_cast<std::size_t>(source + 1)];
    for (Offset p = outgoing_begin; p < outgoing_end; ++p) {
      const int connector = outgoing.to[static_cast<std::size_t>(p)];
      const Offset edge = outgoing.csr_edge[static_cast<std::size_t>(p)];
      add_connector_relation(&relations,
                             connector,
                             work_index,
                             source,
                             edge,
                             graph.values[static_cast<std::size_t>(edge)],
                             -1,
                             std::numeric_limits<float>::infinity());
    }

    const Offset incoming_begin = graph.rowptr[static_cast<std::size_t>(source)];
    const Offset incoming_end = graph.rowptr[static_cast<std::size_t>(source + 1)];
    for (Offset edge = incoming_begin; edge < incoming_end; ++edge) {
      const int connector = graph.colind[static_cast<std::size_t>(edge)];
      add_connector_relation(&relations,
                             connector,
                             work_index,
                             source,
                             -1,
                             std::numeric_limits<float>::infinity(),
                             edge,
                             graph.values[static_cast<std::size_t>(edge)]);
    }
  }

  std::sort(relations.begin(),
            relations.end(),
            [](const ConnectorRelation& lhs, const ConnectorRelation& rhs) {
              if (lhs.connector != rhs.connector) return lhs.connector < rhs.connector;
              return lhs.work_index < rhs.work_index;
            });

  std::vector<ConnectorRelation> merged;
  merged.reserve(relations.size());
  for (const ConnectorRelation& relation : relations) {
    if (merged.empty() ||
        merged.back().connector != relation.connector ||
        merged.back().work_index != relation.work_index) {
      merged.push_back(relation);
      continue;
    }

    ConnectorRelation& current = merged.back();
    if (relation.source_to_connector_edge >= 0 &&
        relation.source_to_connector_cost < current.source_to_connector_cost) {
      current.source_to_connector_edge = relation.source_to_connector_edge;
      current.source_to_connector_cost = relation.source_to_connector_cost;
    }
    if (relation.connector_to_source_edge >= 0 &&
        relation.connector_to_source_cost < current.connector_to_source_cost) {
      current.connector_to_source_edge = relation.connector_to_source_edge;
      current.connector_to_source_cost = relation.connector_to_source_cost;
    }
  }
  return merged;
}

std::vector<ConnectorCandidate> build_connector_candidates(
    const std::vector<ConnectorRelation>& relations) {
  std::vector<ConnectorCandidate> candidates;
  for (std::size_t begin = 0; begin < relations.size();) {
    std::size_t end = begin + 1;
    while (end < relations.size() &&
           relations[end].connector == relations[begin].connector) {
      ++end;
    }

    ConnectorCandidate candidate;
    candidate.connector = relations[begin].connector;
    candidate.relations.assign(relations.begin() + static_cast<std::ptrdiff_t>(begin),
                               relations.begin() + static_cast<std::ptrdiff_t>(end));
    for (const ConnectorRelation& relation : candidate.relations) {
      if (relation.can_warm_start_from_connector()) {
        ++candidate.warmable_count;
      }
    }
    if (candidate.relations.size() >= 2) {
      candidates.push_back(std::move(candidate));
    }
    begin = end;
  }

  std::sort(candidates.begin(),
            candidates.end(),
            [](const ConnectorCandidate& lhs, const ConnectorCandidate& rhs) {
              if (lhs.warmable_count != rhs.warmable_count) {
                return lhs.warmable_count > rhs.warmable_count;
              }
              if (lhs.relations.size() != rhs.relations.size()) {
                return lhs.relations.size() > rhs.relations.size();
              }
              return lhs.connector < rhs.connector;
            });
  return candidates;
}

std::vector<ConnectorPlan> select_connector_plans(const std::vector<ConnectorCandidate>& candidates,
                                                  std::size_t source_count,
                                                  const Options& options,
                                                  std::vector<char>* assigned_sources) {
  assigned_sources->assign(source_count, 0);
  std::vector<ConnectorPlan> plans;

  for (const ConnectorCandidate& candidate : candidates) {
    ConnectorPlan plan;
    plan.connector = candidate.connector;
    for (const ConnectorRelation& relation : candidate.relations) {
      if (!relation.can_warm_start_from_connector()) continue;
      if (relation.work_index >= assigned_sources->size()) continue;
      if ((*assigned_sources)[relation.work_index]) continue;
      plan.relations.push_back(relation);
    }

    if (plan.relations.size() < options.min_connector_sources) {
      continue;
    }

    for (const ConnectorRelation& relation : plan.relations) {
      (*assigned_sources)[relation.work_index] = 1;
    }
    plans.push_back(std::move(plan));
    if (options.max_connector_groups != 0 &&
        plans.size() >= options.max_connector_groups) {
      break;
    }
  }

  return plans;
}

std::vector<ConnectorPlan> discover_connector_plans(const HostCsrF32& graph,
                                                    const std::vector<SourceWork>& work,
                                                    std::size_t source_count,
                                                    const Options& options,
                                                    std::vector<char>* connector_assigned_sources) {
  if (!options.connector_warm_start || source_count == 0) {
    connector_assigned_sources->assign(source_count, 0);
    return {};
  }

  const OutgoingCsr outgoing = build_outgoing_csr(graph);
  const std::vector<ConnectorRelation> relations =
      collect_connector_relations(graph, outgoing, work, source_count);
  const std::vector<ConnectorCandidate> candidates =
      build_connector_candidates(relations);
  std::uint64_t hop2_undirected_pairs = 0;
  std::uint64_t warmable_relation_count = 0;
  for (const ConnectorCandidate& candidate : candidates) {
    const std::uint64_t adjacent = candidate.relations.size();
    hop2_undirected_pairs += adjacent * (adjacent - 1);
    warmable_relation_count += candidate.warmable_count;
  }

  std::cout << "[bf_warm_init] connector discovery summary\n"
            << "[bf_warm_init]   source-connector relations: "
            << relations.size() << '\n'
            << "[bf_warm_init]   common connector nodes with >=2 adjacent sources: "
            << candidates.size() << '\n'
            << "[bf_warm_init]   directed source->connector usable relations: "
            << warmable_relation_count << '\n'
            << "[bf_warm_init]   ordered hop-2 undirected pair opportunities: "
            << hop2_undirected_pairs << '\n';

  std::vector<ConnectorPlan> plans =
      select_connector_plans(candidates, source_count, options, connector_assigned_sources);

  std::size_t assigned_count = 0;
  for (const char assigned : *connector_assigned_sources) {
    if (assigned) ++assigned_count;
  }
  std::cout << "[bf_warm_init] planned " << plans.size()
            << " connector groups for warm starts, assigning "
            << assigned_count << "/" << source_count
            << " sources to connector-ordered routing\n";

  return plans;
}

float initial_column_value(Offset row, const InitialColumn& column) {
  float value = std::numeric_limits<float>::infinity();
  if (row == static_cast<Offset>(column.source)) {
    value = 0.0f;
  }
  if (column.has_warm_start()) {
    const float cached = (*column.cached_dist)[static_cast<std::size_t>(row)];
    if (std::isfinite(cached)) {
      const double shifted =
          static_cast<double>(column.connecting_cost) + static_cast<double>(cached);
      if (shifted <= static_cast<double>(std::numeric_limits<float>::max())) {
        value = std::min(value, static_cast<float>(shifted));
      }
    }
  }
  return value;
}

DeviceCsrOwner make_device_initialized_batched_matrix(Offset rows,
                                                      const std::vector<InitialColumn>& columns,
                                                      hipStream_t stream,
                                                      Offset* h_initial_nnz) {
  if (columns.size() != static_cast<std::size_t>(kDistanceColumns)) {
    throw std::runtime_error("single-source initialization expects exactly one column");
  }
  if (rows < 0 ||
      static_cast<std::uint64_t>(rows) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("row count is too large for host initialization");
  }

  const std::size_t host_rows = static_cast<std::size_t>(rows);
  const Offset batch_cols = static_cast<Offset>(columns.size());
  std::vector<Offset> rowptr(host_rows + 1, 0);

  for (Offset row = 0; row < rows; ++row) {
    Offset count = 0;
    for (const InitialColumn& column : columns) {
      if (std::isfinite(initial_column_value(row, column))) {
        ++count;
      }
    }
    rowptr[static_cast<std::size_t>(row + 1)] = count;
  }
  for (std::size_t i = 0; i < host_rows; ++i) {
    rowptr[i + 1] += rowptr[i];
  }

  const Offset nnz = rowptr.back();
  if (h_initial_nnz) {
    *h_initial_nnz = nnz;
  }
  if (nnz < batch_cols) {
    throw std::runtime_error("initialized matrix is missing source entries");
  }
  if (nnz < 0 ||
      static_cast<std::uint64_t>(nnz) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("initialized matrix is too large for this host");
  }

  std::vector<Index> colind(static_cast<std::size_t>(nnz));
  std::vector<float> values(static_cast<std::size_t>(nnz));
  for (Offset row = 0; row < rows; ++row) {
    Offset out = rowptr[static_cast<std::size_t>(row)];
    for (std::size_t col = 0; col < columns.size(); ++col) {
      const float value = initial_column_value(row, columns[col]);
      if (!std::isfinite(value)) {
        continue;
      }
      colind[static_cast<std::size_t>(out)] = static_cast<Index>(col);
      values[static_cast<std::size_t>(out)] = value;
      ++out;
    }
  }

  DeviceCsrOwner matrix;
  try {
    check_hip(hipMalloc(reinterpret_cast<void**>(&matrix.rowptr),
                        rowptr.size() * sizeof(Offset)),
              "hipMalloc initialized batched rowptr");
    check_hip(hipMemcpyAsync(matrix.rowptr,
                             rowptr.data(),
                             rowptr.size() * sizeof(Offset),
                             hipMemcpyHostToDevice,
                             stream),
              "copy initialized batched rowptr");
    check_hip(hipMalloc(reinterpret_cast<void**>(&matrix.colind),
                        colind.size() * sizeof(Index)),
              "hipMalloc initialized batched colind");
    check_hip(hipMemcpyAsync(matrix.colind,
                             colind.data(),
                             colind.size() * sizeof(Index),
                             hipMemcpyHostToDevice,
                             stream),
              "copy initialized batched colind");
    check_hip(hipMalloc(reinterpret_cast<void**>(&matrix.values),
                        values.size() * sizeof(float)),
              "hipMalloc initialized batched values");
    check_hip(hipMemcpyAsync(matrix.values,
                             values.data(),
                             values.size() * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream),
              "copy initialized batched values");
    check_hip(hipStreamSynchronize(stream), "synchronize initialized batch copy");

    matrix.view.rows = rows;
    matrix.view.cols = batch_cols;
    matrix.view.nnz = nnz;
    matrix.view.rowptr = matrix.rowptr;
    matrix.view.colind = matrix.colind;
    matrix.view.values = matrix.values;
    return matrix;
  } catch (...) {
    throw;
  }
}

DeviceCsrOwner sparse_minplus_relax_batched(const minplus_sparse::DeviceCsrF32& adjacency,
                                            const minplus_sparse::DeviceCsrF32& dist,
                                            hipStream_t stream) {
  DeviceCsrOwner relaxed;
  Offset relaxed_nnz = 0;

  hipError_t status = minplus_sparse::minplus_spgemm_csr_f32(
      adjacency,
      dist,
      &relaxed.rowptr,
      &relaxed.colind,
      &relaxed.values,
      &relaxed_nnz,
      stream);
  if (status == hipErrorInvalidValue) {
    status = minplus_sparse::minplus_spgemm_csr_f32_large_rows(
        adjacency,
        dist,
        &relaxed.rowptr,
        &relaxed.colind,
        &relaxed.values,
        &relaxed_nnz,
        stream);
  }
  check_hip(status, "batched sparse min-plus relaxation");

  relaxed.view.rows = adjacency.rows;
  relaxed.view.cols = dist.cols;
  relaxed.view.nnz = relaxed_nnz;
  relaxed.view.rowptr = relaxed.rowptr;
  relaxed.view.colind = relaxed.colind;
  relaxed.view.values = relaxed.values;
  return relaxed;
}

DeviceCsrOwner merge_batched_distances_on_device(const minplus_sparse::DeviceCsrF32& old_dist,
                                                 const minplus_sparse::DeviceCsrF32& relaxed,
                                                 int* h_changed,
                                                 hipStream_t stream) {
  if (old_dist.rows != relaxed.rows || old_dist.cols != relaxed.cols) {
    throw std::runtime_error("batched merge matrix dimensions do not match");
  }
  if (old_dist.cols != kDistanceColumns) {
    throw std::runtime_error("single-source merge expects exactly one distance column");
  }

  const Offset rows = old_dist.rows;
  const Offset cols = old_dist.cols;
  DeviceCsrOwner merged;
  Offset* d_row_counts = nullptr;
  int* d_changed = nullptr;
  int* d_status = nullptr;
  void* d_scan_temp = nullptr;

  auto cleanup_temp = [&]() {
    if (d_row_counts) (void)hipFree(d_row_counts);
    if (d_changed) (void)hipFree(d_changed);
    if (d_status) (void)hipFree(d_status);
    if (d_scan_temp) (void)hipFree(d_scan_temp);
    d_row_counts = nullptr;
    d_changed = nullptr;
    d_status = nullptr;
    d_scan_temp = nullptr;
  };

  try {
    check_hip(hipMalloc(reinterpret_cast<void**>(&merged.rowptr),
                        static_cast<std::size_t>(rows + 1) * sizeof(Offset)),
              "hipMalloc batched merged rowptr");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_row_counts),
                        static_cast<std::size_t>(rows) * sizeof(Offset)),
              "hipMalloc batched merge row counts");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_changed), sizeof(int)),
              "hipMalloc batched merge changed flag");
    check_hip(hipMalloc(reinterpret_cast<void**>(&d_status), sizeof(int)),
              "hipMalloc batched merge status");
    check_hip(hipMemsetAsync(d_changed, 0, sizeof(int), stream),
              "reset batched merge changed flag");
    check_hip(hipMemsetAsync(d_status, 0, sizeof(int), stream),
              "reset batched merge status");

    constexpr int threads = 256;
    hipLaunchKernelGGL(batched_merge_count_kernel,
                       grid_for_items(rows, threads),
                       dim3(threads),
                       0,
                       stream,
                       rows,
                       cols,
                       old_dist.rowptr,
                       old_dist.colind,
                       old_dist.values,
                       relaxed.rowptr,
                       relaxed.colind,
                       relaxed.values,
                       d_row_counts,
                       d_changed,
                       d_status);
    check_hip(hipGetLastError(), "launch batched merge count kernel");

    std::size_t scan_temp_bytes = 0;
    check_hip(rocprim::exclusive_scan(nullptr,
                                      scan_temp_bytes,
                                      d_row_counts,
                                      merged.rowptr,
                                      static_cast<Offset>(0),
                                      static_cast<std::size_t>(rows),
                                      rocprim::plus<Offset>(),
                                      stream),
              "size batched merge rowptr scan temp");
    check_hip(hipMalloc(&d_scan_temp, scan_temp_bytes),
              "hipMalloc batched merge scan temp");
    check_hip(rocprim::exclusive_scan(d_scan_temp,
                                      scan_temp_bytes,
                                      d_row_counts,
                                      merged.rowptr,
                                      static_cast<Offset>(0),
                                      static_cast<std::size_t>(rows),
                                      rocprim::plus<Offset>(),
                                      stream),
              "scan batched merge row counts");
    check_hip(hipFree(d_scan_temp), "free batched merge scan temp");
    d_scan_temp = nullptr;

    hipLaunchKernelGGL(set_last_rowptr_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       d_row_counts,
                       merged.rowptr,
                       rows);
    check_hip(hipGetLastError(), "launch set batched merged rowptr tail kernel");

    Offset h_nnz = 0;
    int h_status = 0;
    check_hip(hipMemcpyAsync(&h_nnz,
                             merged.rowptr + rows,
                             sizeof(Offset),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy batched merged nnz");
    check_hip(hipMemcpyAsync(&h_status,
                             d_status,
                             sizeof(int),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy batched merge status");
    if (h_changed) {
      check_hip(hipMemcpyAsync(h_changed,
                               d_changed,
                               sizeof(int),
                               hipMemcpyDeviceToHost,
                               stream),
                "copy batched merge changed flag");
    }
    check_hip(hipStreamSynchronize(stream), "synchronize batched merge metadata");
    if (h_status != 0) {
      throw std::runtime_error("batched merge saw invalid column index");
    }

    if (h_nnz > 0) {
      check_hip(hipMalloc(reinterpret_cast<void**>(&merged.colind),
                          static_cast<std::size_t>(h_nnz) * sizeof(Index)),
                "hipMalloc batched merged colind");
      check_hip(hipMalloc(reinterpret_cast<void**>(&merged.values),
                          static_cast<std::size_t>(h_nnz) * sizeof(float)),
                "hipMalloc batched merged values");
      hipLaunchKernelGGL(batched_merge_write_kernel,
                         grid_for_items(rows, threads),
                         dim3(threads),
                         0,
                         stream,
                         rows,
                         cols,
                         old_dist.rowptr,
                         old_dist.colind,
                         old_dist.values,
                         relaxed.rowptr,
                         relaxed.colind,
                         relaxed.values,
                         merged.rowptr,
                         merged.colind,
                         merged.values);
      check_hip(hipGetLastError(), "launch batched merge write kernel");
      check_hip(hipStreamSynchronize(stream), "synchronize batched merge write");
    }

    cleanup_temp();
    merged.view.rows = rows;
    merged.view.cols = cols;
    merged.view.nnz = h_nnz;
    merged.view.rowptr = merged.rowptr;
    merged.view.colind = merged.colind;
    merged.view.values = merged.values;
    return merged;
  } catch (...) {
    cleanup_temp();
    throw;
  }
}

std::vector<float> copy_dist_matrix_to_host_dense(const minplus_sparse::DeviceCsrF32& dist,
                                                  hipStream_t stream) {
  const std::size_t rows = static_cast<std::size_t>(dist.rows);
  const std::size_t cols = static_cast<std::size_t>(dist.cols);
  if (cols == 0 || rows > std::numeric_limits<std::size_t>::max() / cols) {
    throw std::runtime_error("dense distance matrix is too large for this host");
  }

  std::vector<float> dense(rows * cols, std::numeric_limits<float>::infinity());
  std::vector<Offset> rowptr(rows + 1);
  std::vector<Index> colind(static_cast<std::size_t>(dist.nnz));
  std::vector<float> values(static_cast<std::size_t>(dist.nnz));

  check_hip(hipMemcpyAsync(rowptr.data(),
                           dist.rowptr,
                           rowptr.size() * sizeof(Offset),
                           hipMemcpyDeviceToHost,
                           stream),
            "copy batched final rowptr");
  if (dist.nnz > 0) {
    check_hip(hipMemcpyAsync(colind.data(),
                             dist.colind,
                             colind.size() * sizeof(Index),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy batched final colind");
    check_hip(hipMemcpyAsync(values.data(),
                             dist.values,
                             values.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy batched final values");
  }
  check_hip(hipStreamSynchronize(stream), "synchronize batched final copy");

  for (Offset row = 0; row < dist.rows; ++row) {
    for (Offset p = rowptr[static_cast<std::size_t>(row)];
         p < rowptr[static_cast<std::size_t>(row + 1)];
         ++p) {
      const Index col = colind[static_cast<std::size_t>(p)];
      if (col < 0 || static_cast<Offset>(col) >= dist.cols) continue;
      float& current =
          dense[static_cast<std::size_t>(row) * cols + static_cast<std::size_t>(col)];
      const float value = values[static_cast<std::size_t>(p)];
      if (value < current) {
        current = value;
      }
    }
  }

  return dense;
}

int choose_progress_interval(int max_iters, int requested_every, bool show_progress) {
  if (!show_progress || max_iters <= 0) return 0;
  if (requested_every > 0) return requested_every;
  return std::max(1, std::min(100, max_iters / 100));
}

std::string progress_bar(int current, int total, int width) {
  if (width <= 0) return "";
  const double fraction =
      total <= 0 ? 1.0 : std::min(1.0, static_cast<double>(current) / total);
  const int filled = static_cast<int>(fraction * width);

  std::string bar;
  bar.reserve(static_cast<std::size_t>(width) + 2);
  bar.push_back('[');
  for (int i = 0; i < width; ++i) {
    bar.push_back(i < filled ? '#' : '.');
  }
  bar.push_back(']');
  return bar;
}

void print_bf_iteration_progress(std::size_t run_index,
                                 std::size_t run_count,
                                 int iteration,
                                 int max_iters,
                                 int changed,
                                 Offset old_nnz,
                                 Offset relaxed_nnz,
                                 Offset merged_nnz,
                                 double elapsed_sec,
                                 bool final_line) {
  const double percent =
      max_iters <= 0 ? 100.0 : 100.0 * static_cast<double>(iteration) / max_iters;
  const double iter_per_sec =
      elapsed_sec > 0.0 ? static_cast<double>(iteration) / elapsed_sec : 0.0;
  std::ostringstream percent_text;
  percent_text << std::fixed
               << std::setprecision(percent > 0.0 && percent < 0.1 ? 4 : 1)
               << std::min(100.0, percent);

  std::cout << '\r'
            << "[bf_warm_init] BF run " << run_index << "/" << run_count << ' '
            << progress_bar(iteration, max_iters, 30) << ' '
            << percent_text.str() << "% "
            << "iter=" << iteration << "/" << max_iters
            << " changed=" << (changed ? "yes" : "no")
            << " dist_nnz=" << old_nnz << "->" << merged_nnz
            << " relaxed_nnz=" << relaxed_nnz
            << " elapsed=" << std::setprecision(2) << elapsed_sec << "s"
            << " rate=" << std::setprecision(2) << iter_per_sec << " iter/s"
            << "        " << std::flush;
  if (final_line) {
    std::cout << '\n';
  }
}

SsspResult bellman_ford_single_source(const minplus_sparse::DeviceCsrF32& d_graph,
                                      Offset rows,
                                      const InitialColumn& initial_column,
                                      int max_iters,
                                      int progress_every,
                                      bool show_progress,
                                      std::size_t run_index,
                                      std::size_t run_count,
                                      hipStream_t stream) {
  if (max_iters < 0) {
    max_iters = static_cast<int>(rows) - 1;
  }

  SsspResult result;
  std::vector<InitialColumn> initial_columns{initial_column};
  DeviceCsrOwner d_dist =
      make_device_initialized_batched_matrix(rows,
                                             initial_columns,
                                             stream,
                                             &result.initial_nnz);
  const int progress_interval =
      choose_progress_interval(max_iters, progress_every, show_progress);
  const auto bf_start_time = std::chrono::steady_clock::now();

  for (int iter = 0; iter < max_iters; ++iter) {
    const Offset old_nnz = d_dist.view.nnz;
    DeviceCsrOwner d_relaxed = sparse_minplus_relax_batched(d_graph, d_dist.view, stream);
    const Offset relaxed_nnz = d_relaxed.view.nnz;
    int changed = 0;
    DeviceCsrOwner d_next =
        merge_batched_distances_on_device(d_dist.view, d_relaxed.view, &changed, stream);
    const Offset merged_nnz = d_next.view.nnz;
    d_dist = std::move(d_next);
    result.iterations_used = iter + 1;

    const bool should_print =
        progress_interval > 0 &&
        (iter == 0 || result.iterations_used % progress_interval == 0 ||
         changed == 0 || result.iterations_used == max_iters);
    if (should_print) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed_sec = std::chrono::duration<double>(now - bf_start_time).count();
      print_bf_iteration_progress(run_index,
                                  run_count,
                                  result.iterations_used,
                                  max_iters,
                                  changed,
                                  old_nnz,
                                  relaxed_nnz,
                                  merged_nnz,
                                  elapsed_sec,
                                  changed == 0 || result.iterations_used == max_iters);
    }

    if (!changed) {
      result.converged = true;
      break;
    }
  }

  result.dense_dist = copy_dist_matrix_to_host_dense(d_dist.view, stream);
  return result;
}

float dense_dist_at(const std::vector<float>& dense,
                    Offset rows,
                    int node) {
  (void)rows;
  return dense[static_cast<std::size_t>(node)];
}

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dense_dist,
                                                int source,
                                                int target,
                                                const Options& options) {
  if (!valid_node(graph, source) || !valid_node(graph, target)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (source == target) {
    return {};
  }
  if (!std::isfinite(dense_dist_at(dense_dist, graph.rows, target))) {
    return {};
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (Offset guard = 0; guard < graph.rows && current != source; ++guard) {
    const std::size_t row = static_cast<std::size_t>(current);
    const float current_dist =
        dense_dist_at(dense_dist, graph.rows, current);
    Offset best_edge = -1;
    int best_pred = -1;
    double best_error = std::numeric_limits<double>::infinity();

    for (Offset edge = graph.rowptr[row]; edge < graph.rowptr[row + 1]; ++edge) {
      const int pred = graph.colind[static_cast<std::size_t>(edge)];
      const float pred_dist =
          dense_dist_at(dense_dist, graph.rows, pred);
      if (!std::isfinite(pred_dist)) {
        continue;
      }

      const double candidate =
          static_cast<double>(pred_dist) + graph.values[static_cast<std::size_t>(edge)];
      const double error = std::fabs(candidate - current_dist);
      if (error <= tolerance_for(candidate, current_dist, options) &&
          error < best_error) {
        best_error = error;
        best_edge = edge;
        best_pred = pred;
      }
    }

    if (best_edge < 0) {
      throw std::runtime_error("could not reconstruct shortest path predecessor");
    }

    PathEdge path_edge;
    path_edge.from = best_pred;
    path_edge.to = current;
    path_edge.csr_edge = best_edge;
    path_edge.cost = graph.values[static_cast<std::size_t>(best_edge)];
    reversed.push_back(path_edge);
    current = best_pred;
  }

  if (current != source) {
    throw std::runtime_error("shortest path reconstruction did not reach source");
  }

  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::vector<int> nodes_from_edges(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

void write_metadata_record(std::ostream& out,
                           const HostCsrF32& graph,
                           const RoutingMetadata& metadata,
                           const Options& options) {
  out << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\""
      << ",\"producer\":\"bf_warm_init\""
      << ",\"processing_mode\":\"single_source_connector_order\""
      << ",\"connector_warm_start\":" << (options.connector_warm_start ? "true" : "false")
      << ",\"node_count\":" << graph.rows
      << ",\"edge_count\":" << graph.nnz
      << ",\"route_request_count\":" << metadata.route_requests.size()
      << ",\"edge_orientation\":\"incoming\""
      << ",\"description\":\"row v, col u means directed edge u -> v\""
      << "}\n";
}

void write_path_record(std::ostream& out,
                       const Query& query,
                       bool reached,
                       float distance,
                       const std::vector<int>& nodes,
                       const std::vector<PathEdge>& edges) {
  out << "{\"type\":\"path\"";
  out << ",\"net\":";
  write_json_string(out, query.net_name);
  out << ",\"net_index\":" << query.net_index
      << ",\"source\":" << query.source
      << ",\"target\":" << query.target
      << ",\"reached\":" << (reached ? "true" : "false");
  if (reached) {
    out << ",\"distance\":" << std::setprecision(9) << distance;
  } else {
    out << ",\"distance\":null";
  }

  out << ",\"nodes\":[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) out << ',';
    out << nodes[i];
  }
  out << "]";

  out << ",\"csr_edges\":[";
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (i != 0) out << ',';
    out << edges[i].csr_edge;
  }
  out << "]}\n";
}

std::vector<SourceWork> build_source_work(const RoutingMetadata& metadata,
                                          const HostCsrF32& graph,
                                          const Options& options,
                                          std::size_t* total_queries) {
  std::vector<SourceWork> work;
  std::unordered_map<int, std::size_t> source_to_work;
  *total_queries = 0;

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());

  for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const std::string net_name = string_at(metadata, request.net_string);

    for (const SitePinNode& source_pin : request.sources) {
      if (options.query_limit != 0 && *total_queries >= options.query_limit) {
        return work;
      }
      if (!valid_node(graph, source_pin.node)) {
        std::cout << "[warning] skipping out-of-range source node "
                  << source_pin.node << " for net_index=" << net_index << '\n';
        continue;
      }

      std::size_t work_index = 0;
      const auto found = source_to_work.find(source_pin.node);
      if (found == source_to_work.end()) {
        work_index = work.size();
        source_to_work.emplace(source_pin.node, work_index);
        SourceWork source_work;
        source_work.source = source_pin.node;
        work.push_back(std::move(source_work));
      } else {
        work_index = found->second;
      }

      for (const SitePinNode& sink_pin : request.sinks) {
        if (options.query_limit != 0 && *total_queries >= options.query_limit) {
          return work;
        }
        if (!valid_node(graph, sink_pin.node)) {
          std::cout << "[warning] skipping out-of-range sink node "
                    << sink_pin.node << " for net_index=" << net_index << '\n';
          continue;
        }

        Query query;
        query.net_index = net_index;
        query.net_name = net_name;
        query.source = source_pin.node;
        query.target = sink_pin.node;
        work[work_index].queries.push_back(std::move(query));
        ++(*total_queries);
      }
    }
  }

  return work;
}

std::size_t parse_size(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::size_t>(value);
}

int parse_int(const std::string& text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid integer for ") + name);
  }
  return static_cast<int>(value);
}

double parse_double(const std::string& text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid number for ") + name);
  }
  return value;
}

std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <graph.csrbin> <metadata.ifmeta.bin> <output.paths.jsonl> [options]\n\n"
      << "Options:\n"
      << "  --no-connector-warm-start Disable hop-2 shared-connector warm starts.\n"
      << "  --min-connector-sources <n> Min source->connector sources per connector. Default: 2.\n"
      << "  --max-connector-groups <n> Cap connector groups; 0 = no cap.\n"
      << "  --max-iters <n>          Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --net-limit <n>          Use only the first n route requests.\n"
      << "  --source-limit <n>       Run only the first n unique source nodes.\n"
      << "  --query-limit <n>        Emit at most n source-sink path records.\n"
      << "  --bf-progress-every <n>  Print BF progress every n iterations. Default: auto.\n"
      << "  --no-bf-progress         Disable the per-run BF progress bar.\n"
      << "  --skip-unreached         Do not write JSONL records for unreachable sinks.\n"
      << "  --abs-tol <x>            Path reconstruction absolute tolerance. Default: 1e-3.\n"
      << "  --rel-tol <x>            Path reconstruction relative tolerance. Default: 1e-5.\n"
      << "  --help                   Print this message.\n";
}

Options parse_options(int argc,
                      char** argv,
                      std::filesystem::path* csr_path,
                      std::filesystem::path* metadata_path,
                      std::filesystem::path* output_path) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--no-connector-warm-start") {
      options.connector_warm_start = false;
    } else if (arg == "--min-connector-sources") {
      options.min_connector_sources =
          parse_size(require_value(&i, argc, argv, "--min-connector-sources"),
                     "--min-connector-sources");
    } else if (arg == "--max-connector-groups") {
      options.max_connector_groups =
          parse_size(require_value(&i, argc, argv, "--max-connector-groups"),
                     "--max-connector-groups");
    } else if (arg == "--max-iters") {
      options.max_iters = parse_int(require_value(&i, argc, argv, "--max-iters"), "--max-iters");
    } else if (arg == "--net-limit") {
      options.net_limit = parse_size(require_value(&i, argc, argv, "--net-limit"), "--net-limit");
    } else if (arg == "--source-limit") {
      options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (arg == "--query-limit") {
      options.query_limit =
          parse_size(require_value(&i, argc, argv, "--query-limit"), "--query-limit");
    } else if (arg == "--bf-progress-every") {
      options.bf_progress_every =
          parse_int(require_value(&i, argc, argv, "--bf-progress-every"),
                    "--bf-progress-every");
    } else if (arg == "--no-bf-progress") {
      options.show_bf_progress = false;
    } else if (arg == "--skip-unreached") {
      options.include_unreached = false;
    } else if (arg == "--abs-tol") {
      options.abs_tolerance = parse_double(require_value(&i, argc, argv, "--abs-tol"),
                                           "--abs-tol");
    } else if (arg == "--rel-tol") {
      options.rel_tolerance = parse_double(require_value(&i, argc, argv, "--rel-tol"),
                                           "--rel-tol");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (csr_path->empty()) {
      *csr_path = arg;
    } else if (metadata_path->empty()) {
      *metadata_path = arg;
    } else if (output_path->empty()) {
      *output_path = arg;
    } else {
      throw std::runtime_error("too many positional arguments");
    }
  }

  if (csr_path->empty() || metadata_path->empty() || output_path->empty()) {
    throw std::runtime_error("expected <graph.csrbin> <metadata.ifmeta.bin> <output.paths.jsonl>");
  }
  if (options.max_iters < -1) {
    throw std::runtime_error("--max-iters must be -1 or nonnegative");
  }
  if (options.min_connector_sources == 0) {
    throw std::runtime_error("--min-connector-sources must be positive");
  }
  if (options.bf_progress_every < 0) {
    throw std::runtime_error("--bf-progress-every must be nonnegative");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path csr_path;
    std::filesystem::path metadata_path;
    std::filesystem::path output_path;
    const Options options = parse_options(argc, argv, &csr_path, &metadata_path, &output_path);

    std::cout << "[bf_warm_init] loading CSR: " << csr_path.string() << '\n';
    HostCsrF32 graph = load_csrbin(csr_path);
    std::cout << "[bf_warm_init] graph rows=" << graph.rows
              << " nnz=" << graph.nnz
              << " orientation=incoming(row v, col u means u -> v)\n";

    std::cout << "[bf_warm_init] loading metadata: " << metadata_path.string() << '\n';
    RoutingMetadata metadata = load_interchange_metadata(metadata_path);
    if (metadata.node_device_ids.size() != static_cast<std::size_t>(graph.rows)) {
      throw std::runtime_error("metadata node count does not match CSR rows");
    }
    if (metadata.edge_attr_count != static_cast<std::uint64_t>(graph.nnz)) {
      throw std::runtime_error("metadata edge attribute count does not match CSR nnz");
    }

    std::size_t total_queries = 0;
    std::vector<SourceWork> work =
        build_source_work(metadata, graph, options, &total_queries);
    const std::size_t source_count =
        options.source_limit == 0
            ? work.size()
            : std::min(options.source_limit, work.size());
    std::cout << "[bf_warm_init] route_requests=" << metadata.route_requests.size()
              << " unique_sources=" << work.size()
              << " sources_to_run=" << source_count
              << " source_sink_queries=" << total_queries
              << " processing=one_source_at_a_time\n";
    if (source_count != 0) {
      const double dense_mib =
          static_cast<double>(graph.rows) *
          static_cast<double>(sizeof(float)) / (1024.0 * 1024.0);
      std::cout << "[bf_warm_init] dense distance copy per source ~= "
                << std::fixed << std::setprecision(2) << dense_mib
                << " MiB (N * 4 bytes)\n";
    }

    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path);
    if (!out) {
      throw std::runtime_error("could not open output JSONL file: " + output_path.string());
    }
    write_metadata_record(out, graph, metadata, options);

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    std::uint64_t paths_written = 0;
    std::uint64_t reached_count = 0;
    std::uint64_t unreached_count = 0;
    std::uint64_t warm_started_sources = 0;
    std::uint64_t cold_started_sources = 0;

    try {
      std::cout << "[bf_warm_init] copying CSR to GPU once\n";
      DeviceCsrOwner d_graph = bf_csr_detail::copy_host_csr_to_device(graph, stream);

      std::vector<char> routed_sources(source_count, 0);
      std::vector<char> connector_assigned_sources;
      std::vector<ConnectorPlan> connector_plans =
          discover_connector_plans(graph,
                                   work,
                                   source_count,
                                   options,
                                   &connector_assigned_sources);
      std::size_t connector_assigned_count = 0;
      for (const char assigned : connector_assigned_sources) {
        if (assigned) ++connector_assigned_count;
      }
      const std::size_t fallback_source_count =
          source_count >= connector_assigned_count ? source_count - connector_assigned_count : 0;
      const std::size_t total_bf_runs = connector_plans.size() + source_count;
      std::cout << "[bf_warm_init] processing order decided: "
                << connector_plans.size() << " connector runs, "
                << connector_assigned_count << " connector-warmed sources, "
                << fallback_source_count << " cold fallback sources\n";

      std::size_t bf_run_index = 0;

      auto route_source = [&](std::size_t work_index, const InitialColumn& column) {
        if (work_index >= source_count || routed_sources[work_index]) return;

        const SourceWork& source_work = work[work_index];
        const bool used_warm_init = column.has_warm_start() && column.from_connector;
        ++bf_run_index;

        SsspResult result = bellman_ford_single_source(d_graph.view,
                                                       graph.rows,
                                                       column,
                                                       options.max_iters,
                                                       options.bf_progress_every,
                                                       options.show_bf_progress,
                                                       bf_run_index,
                                                       total_bf_runs,
                                                       stream);

        std::uint64_t source_paths_written = 0;
        std::uint64_t source_reached = 0;
        std::uint64_t source_unreached = 0;
        for (const Query& query : source_work.queries) {
          const float distance = dense_dist_at(result.dense_dist, graph.rows, query.target);
          const bool reached = std::isfinite(distance);
          if (!reached) {
            ++unreached_count;
            ++source_unreached;
            if (options.include_unreached) {
              write_path_record(out, query, false, distance, {}, {});
              ++paths_written;
              ++source_paths_written;
            }
            continue;
          }

          std::vector<PathEdge> edges =
              reconstruct_shortest_path(graph,
                                        result.dense_dist,
                                        query.source,
                                        query.target,
                                        options);
          std::vector<int> nodes = nodes_from_edges(query.source, edges);
          write_path_record(out, query, true, distance, nodes, edges);
          ++reached_count;
          ++source_reached;
          ++paths_written;
          ++source_paths_written;
        }

        routed_sources[work_index] = 1;
        if (used_warm_init) {
          ++warm_started_sources;
        } else {
          ++cold_started_sources;
        }

        std::cout << "[bf_warm_init] source=" << source_work.source
                  << " warm_init=" << (used_warm_init ? "yes" : "no")
                  << " connector=";
        if (used_warm_init) {
          std::cout << column.cached_source;
        } else {
          std::cout << "none";
        }
        std::cout << " iterations=" << result.iterations_used
                  << " converged=" << (result.converged ? "yes" : "no")
                  << " reached=" << source_reached
                  << " unreached=" << source_unreached
                  << " paths_written=" << source_paths_written << '\n';
      };

      for (std::size_t plan_index = 0; plan_index < connector_plans.size(); ++plan_index) {
        const ConnectorPlan& plan = connector_plans[plan_index];
        std::cout << "[bf_warm_init] processing connector "
                  << (plan_index + 1) << "/" << connector_plans.size()
                  << " connector=" << plan.connector
                  << " related_sources=" << plan.relations.size()
                  << " action=route_connector_then_sources\n";

        InitialColumn connector_column;
        connector_column.source = plan.connector;
        ++bf_run_index;
        SsspResult connector_result = bellman_ford_single_source(d_graph.view,
                                                                 graph.rows,
                                                                 connector_column,
                                                                 options.max_iters,
                                                                 options.bf_progress_every,
                                                                 options.show_bf_progress,
                                                                 bf_run_index,
                                                                 total_bf_runs,
                                                                 stream);
        std::cout << "[bf_warm_init] connector=" << plan.connector
                  << " routed internally, iterations_used="
                  << connector_result.iterations_used
                  << " converged="
                  << (connector_result.converged ? "yes" : "no")
                  << "; no connector JSONL paths emitted\n";

        std::size_t routed_for_connector = 0;
        for (const ConnectorRelation& relation : plan.relations) {
          if (routed_sources[relation.work_index]) continue;

          InitialColumn column;
          column.source = relation.source;
          column.cached_source = plan.connector;
          column.connecting_edge = relation.source_to_connector_edge;
          column.connecting_cost = relation.source_to_connector_cost;
          column.cached_dist = &connector_result.dense_dist;
          column.from_connector = true;
          route_source(relation.work_index, column);
          ++routed_for_connector;
        }

        const std::size_t released_entries = connector_result.dense_dist.size();
        connector_result.dense_dist.clear();
        connector_result.dense_dist.shrink_to_fit();
        std::cout << "[bf_warm_init] finished connector=" << plan.connector
                  << " routed_sources=" << routed_for_connector
                  << " released_connector_dist_entries=" << released_entries << '\n';
      }

      std::size_t fallback_count = 0;
      for (std::size_t work_index = 0; work_index < source_count; ++work_index) {
        if (!routed_sources[work_index]) ++fallback_count;
      }
      std::cout << "[bf_warm_init] fallback routing sources not assigned to usable "
                << "connector groups: " << fallback_count << '\n';

      for (std::size_t work_index = 0; work_index < source_count; ++work_index) {
        if (routed_sources[work_index]) continue;
        InitialColumn column;
        column.source = work[work_index].source;
        route_source(work_index, column);
      }
      check_hip(hipStreamSynchronize(stream), "sync before freeing device CSR graph");
    } catch (...) {
      const hipError_t destroy_status = hipStreamDestroy(stream);
      if (destroy_status != hipSuccess) {
        std::cerr << "[bf_warm_init] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
      throw;
    }

    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");

    std::cout << "[bf_warm_init] wrote " << paths_written
              << " path records to " << output_path.string()
              << " reached=" << reached_count
              << " unreached=" << unreached_count
              << " warm_started_sources=" << warm_started_sources
              << " cold_started_sources=" << cold_started_sources << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[bf_warm_init] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
