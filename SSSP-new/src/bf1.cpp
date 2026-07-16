// Bare-bones outgoing-CSR Bellman-Ford driver using mp_csr.cpp relaxation.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     SSSP-new/src/bf1.cpp \
//     SSSP-new/src/mp_csr.cpp \
//     -o bf1
//
// Run:
//   ./bf1 SSSP-new/data/logicnets_jscl.outgoing.csrbin \
//     SSSP-new/data/logicnets_jscl.outgoing.csrbin.ifmeta.bin \
//     bf1_summary.csv \
//     --source-progress-every 100

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rips_sssp_new {

using Offset = std::int64_t;
using Index = int;

struct DeviceOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const Offset* rowptr = nullptr;
  const Index* degree = nullptr;
  const Index* to = nullptr;
  const float* values = nullptr;
  const Offset* edge_id = nullptr;
};

struct DeviceSsspWorkspace {
  Offset rows = 0;
  Offset gather_capacity = 0;
  unsigned long long* best_state = nullptr;
  int* changed_flag = nullptr;
  int* status = nullptr;
  Index* gather_nodes = nullptr;
  unsigned long long* gather_states = nullptr;
};

struct OutgoingSsspResult {
  int iterations_used = 0;
  bool converged = false;
};

// Hybrid CPU/GPU function from mp_csr.cpp. Inputs: HIP status and label. Output: none, or throws.
// Purpose: expose mp_csr.cpp HIP error handling to this driver.
void check_hip(hipError_t status, const char* what);

// Hybrid CPU/GPU function from mp_csr.cpp. Inputs: rows, gather capacity, stream. Output: GPU workspace.
// Purpose: allocate reusable packed-state workspace for all source runs.
DeviceSsspWorkspace make_sssp_workspace(Offset rows,
                                        Offset gather_capacity,
                                        hipStream_t stream = nullptr);

// Hybrid CPU/GPU function from mp_csr.cpp. Inputs: workspace pointer. Output: freed workspace pointers.
// Purpose: release reusable packed-state GPU workspace.
void free_sssp_workspace(DeviceSsspWorkspace* workspace);

// Hybrid CPU/GPU function from mp_csr.cpp. Inputs: device graph, workspace, source, max iterations, stream.
// Output: iterations used and convergence flag.
// Purpose: run bare global dense outgoing Bellman-Ford for one source.
OutgoingSsspResult run_outgoing_dense_sssp(const DeviceOutgoingCsrF32& graph,
                                           DeviceSsspWorkspace& workspace,
                                           int source,
                                           int max_iters,
                                           hipStream_t stream = nullptr);

}  // namespace rips_sssp_new

namespace {

using Offset = rips_sssp_new::Offset;
using Index = rips_sssp_new::Index;

constexpr char OUTGOING_CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'O', 'C', 'S', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t MIN_OUTGOING_CSR_VERSION = 3;
constexpr std::uint64_t CURRENT_OUTGOING_CSR_VERSION = 3;
// SSSP-new metadata version 3 is outgoing-edge aligned: edge_attrs[k]
// describes outgoing CSR edge k, not Routing's incoming-oriented edge order.
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 3;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;

struct HostOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> degree;
  std::vector<Index> to;
  std::vector<float> values;
  std::vector<Offset> edge_id;
};

struct DeviceOutgoingCsrOwner {
  rips_sssp_new::DeviceOutgoingCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* degree = nullptr;
  Index* to = nullptr;
  float* values = nullptr;
  Offset* edge_id = nullptr;
};

struct RouteSources {
  std::uint64_t raw_source_count = 0;
  std::uint64_t invalid_source_count = 0;
  std::uint64_t metadata_node_count = 0;
  std::vector<int> unique_sources;
};

struct Options {
  int max_iters = -1;
  std::size_t source_progress_every = 100;
  std::size_t source_limit = 0;
};

struct ParsedArgs {
  std::filesystem::path csr_path;
  std::filesystem::path metadata_path;
  std::filesystem::path output_path;
  Options options;
};

// CPU helper. Inputs: binary stream and field name. Output: one little-endian u64.
// Purpose: read fixed-width counts and ids from RIPS binary files.
std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

// CPU helper. Inputs: binary stream, destination vector, element count, field name.
// Output: vector resized and filled from the stream.
// Purpose: read CSR arrays while checking host-size limits.
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

// CPU helper. Inputs: item count, bytes per item, field name. Output: total byte count.
// Purpose: detect overflow before skipping repeated binary records.
std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t bytes_per_item,
                                 const char* name) {
  if (bytes_per_item != 0 &&
      count > std::numeric_limits<std::uint64_t>::max() / bytes_per_item) {
    throw std::runtime_error(std::string(name) + " byte count overflow");
  }
  return count * bytes_per_item;
}

// CPU helper. Inputs: binary stream, byte count, field name. Output: stream advanced.
// Purpose: skip metadata sections bf1 does not need.
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

// CPU helper. Inputs: binary stream. Output: stream advanced over one length-prefixed string.
// Purpose: skip metadata strings without keeping them in memory.
void skip_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  skip_bytes(in, size, "metadata string bytes");
}

// CPU helper. Inputs: host outgoing CSR. Output: none, or throws.
// Purpose: validate outgoing graph shape and array consistency before GPU copy.
void validate_outgoing_csr(const HostOutgoingCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph.rows > static_cast<Offset>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("outgoing CSR has too many rows for int node ids");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.degree.size() != static_cast<std::size_t>(graph.rows) ||
      graph.to.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.edge_id.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("outgoing CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("outgoing CSR rowptr must start at 0 and end at nnz");
  }

  for (Offset row = 0; row < graph.rows; ++row) {
    const Offset begin = graph.rowptr[static_cast<std::size_t>(row)];
    const Offset end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("outgoing CSR rowptr is not monotone");
    }
    const Offset row_degree = end - begin;
    if (row_degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    if (graph.degree[static_cast<std::size_t>(row)] !=
        static_cast<Index>(row_degree)) {
      throw std::runtime_error("outgoing CSR degree array does not match rowptr");
    }
  }

  for (std::size_t edge = 0; edge < graph.to.size(); ++edge) {
    if (graph.to[edge] < 0 || static_cast<Offset>(graph.to[edge]) >= graph.cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (graph.edge_id[edge] < 0 ||
        graph.edge_id[edge] >= static_cast<Offset>(kPackedNoPredEdge)) {
      throw std::runtime_error("outgoing CSR edge id cannot be packed");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
  }
}

// CPU helper. Inputs: outgoing RIPSOCS1 CSR path. Output: host outgoing CSR arrays.
// Purpose: load the outgoing CSR format consumed by mp_csr.cpp.
HostOutgoingCsrF32 load_outgoing_csrbin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open outgoing CSR file: " + path.string());
  }

  char magic[sizeof(OUTGOING_CSR_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, OUTGOING_CSR_MAGIC, sizeof(OUTGOING_CSR_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS outgoing CSR file");
  }

  const std::uint64_t version = read_u64(in, "outgoing CSR format version");
  const std::uint64_t orientation = read_u64(in, "outgoing CSR orientation");
  if (version < MIN_OUTGOING_CSR_VERSION ||
      version > CURRENT_OUTGOING_CSR_VERSION) {
    throw std::runtime_error("unsupported outgoing CSR format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported outgoing CSR orientation");
  }

  const std::uint64_t rows = read_u64(in, "outgoing CSR row count");
  const std::uint64_t cols = read_u64(in, "outgoing CSR column count");
  const std::uint64_t nnz = read_u64(in, "outgoing CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "outgoing CSR rowptr count");
  const std::uint64_t degree_count = read_u64(in, "outgoing CSR degree count");
  const std::uint64_t to_count = read_u64(in, "outgoing CSR destination count");
  const std::uint64_t values_count = read_u64(in, "outgoing CSR values count");
  const std::uint64_t edge_id_count =
      read_u64(in, "outgoing CSR edge-id count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || degree_count != rows ||
      to_count != nnz || values_count != nnz || edge_id_count != nnz) {
    throw std::runtime_error("outgoing CSR header counts are inconsistent");
  }

  HostOutgoingCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  read_array(in, graph.degree, degree_count, "outgoing CSR degree");
  read_array(in, graph.to, to_count, "outgoing CSR destinations");
  read_array(in, graph.values, values_count, "outgoing CSR values");
  read_array(in, graph.edge_id, edge_id_count, "outgoing CSR edge-id map");
  validate_outgoing_csr(graph);
  return graph;
}

// Hybrid CPU/GPU helper. Inputs: host outgoing CSR and HIP stream.
// Output: device owner with graph arrays copied once to GPU memory.
// Purpose: avoid transferring the CSR inside Bellman-Ford iterations or per source.
DeviceOutgoingCsrOwner copy_outgoing_csr_to_device(const HostOutgoingCsrF32& host,
                                                   hipStream_t stream) {
  DeviceOutgoingCsrOwner device;
  try {
    const std::size_t rowptr_bytes = host.rowptr.size() * sizeof(Offset);
    const std::size_t degree_bytes = host.degree.size() * sizeof(Index);
    rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr),
                                       rowptr_bytes),
                             "hipMalloc outgoing rowptr");
    rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.degree),
                                       degree_bytes),
                             "hipMalloc outgoing degree");
    rips_sssp_new::check_hip(hipMemcpyAsync(device.rowptr,
                                            host.rowptr.data(),
                                            rowptr_bytes,
                                            hipMemcpyHostToDevice,
                                            stream),
                             "copy outgoing rowptr to device");
    rips_sssp_new::check_hip(hipMemcpyAsync(device.degree,
                                            host.degree.data(),
                                            degree_bytes,
                                            hipMemcpyHostToDevice,
                                            stream),
                             "copy outgoing degree to device");

    if (host.nnz > 0) {
      const std::size_t to_bytes = host.to.size() * sizeof(Index);
      const std::size_t values_bytes = host.values.size() * sizeof(float);
      const std::size_t edge_id_bytes = host.edge_id.size() * sizeof(Offset);
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.to),
                                         to_bytes),
                               "hipMalloc outgoing destinations");
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.values),
                                         values_bytes),
                               "hipMalloc outgoing values");
      rips_sssp_new::check_hip(hipMalloc(reinterpret_cast<void**>(&device.edge_id),
                                         edge_id_bytes),
                               "hipMalloc outgoing edge-id map");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.to,
                                              host.to.data(),
                                              to_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing destinations to device");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.values,
                                              host.values.data(),
                                              values_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing values to device");
      rips_sssp_new::check_hip(hipMemcpyAsync(device.edge_id,
                                              host.edge_id.data(),
                                              edge_id_bytes,
                                              hipMemcpyHostToDevice,
                                              stream),
                               "copy outgoing edge-id map to device");
    }

    rips_sssp_new::check_hip(hipStreamSynchronize(stream),
                             "synchronize outgoing CSR copy");
    device.view.rows = host.rows;
    device.view.cols = host.cols;
    device.view.nnz = host.nnz;
    device.view.rowptr = device.rowptr;
    device.view.degree = device.degree;
    device.view.to = device.to;
    device.view.values = device.values;
    device.view.edge_id = device.edge_id;
    return device;
  } catch (...) {
    if (device.rowptr) (void)hipFree(device.rowptr);
    if (device.degree) (void)hipFree(device.degree);
    if (device.to) (void)hipFree(device.to);
    if (device.values) (void)hipFree(device.values);
    if (device.edge_id) (void)hipFree(device.edge_id);
    throw;
  }
}

// Hybrid CPU/GPU helper. Inputs: device owner. Output: freed GPU graph pointers.
// Purpose: release the one-time outgoing CSR device copy.
void free_device_outgoing_csr(DeviceOutgoingCsrOwner* device) {
  if (!device) return;
  if (device->rowptr) {
    (void)hipFree(device->rowptr);
    device->rowptr = nullptr;
  }
  if (device->degree) {
    (void)hipFree(device->degree);
    device->degree = nullptr;
  }
  if (device->to) {
    (void)hipFree(device->to);
    device->to = nullptr;
  }
  if (device->values) {
    (void)hipFree(device->values);
    device->values = nullptr;
  }
  if (device->edge_id) {
    (void)hipFree(device->edge_id);
    device->edge_id = nullptr;
  }
  device->view = {};
}

// CPU helper. Inputs: metadata path and graph shape. Output: raw and unique valid sources.
// Purpose: parse outgoing SSSP-new metadata and collect source nodes in first-seen order.
RouteSources load_route_sources_from_metadata(const std::filesystem::path& path,
                                              Offset graph_rows,
                                              Offset graph_nnz) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS SSSP-new metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != EXPECTED_METADATA_VERSION) {
    throw std::runtime_error("unsupported metadata format version");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  const std::uint64_t string_count = read_u64(in, "metadata string count");
  const std::uint64_t node_count = read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count = read_u64(in, "metadata edge attribute count");
  if (edge_attr_count != static_cast<std::uint64_t>(graph_nnz)) {
    throw std::runtime_error(
        "metadata edge attribute count does not match outgoing CSR nnz");
  }
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

  for (std::uint64_t i = 0; i < string_count; ++i) {
    skip_string(in);
  }

  skip_bytes(in,
             checked_byte_count(node_count, sizeof(std::uint64_t), "metadata device node ids"),
             "metadata device node ids");
  skip_bytes(in,
             checked_byte_count(edge_attr_count, 2 * sizeof(std::uint64_t), "metadata edge attrs"),
             "metadata edge attrs");
  skip_bytes(in,
             checked_byte_count(pip_data_count, 3 * sizeof(std::uint64_t), "metadata pip data"),
             "metadata pip data");
  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t),
                                "metadata site pin attrs"),
             "metadata site pin attrs");

  RouteSources sources;
  sources.metadata_node_count = node_count;
  std::unordered_set<int> seen;
  const std::uint64_t reserve_hint =
      route_request_count > 500000 ? 1000000 : route_request_count * 2;
  seen.reserve(static_cast<std::size_t>(reserve_hint));

  for (std::uint64_t request_index = 0; request_index < route_request_count; ++request_index) {
    (void)read_u64(in, "metadata route request net");
    (void)read_u64(in, "metadata route logical net");

    const std::uint64_t source_count = read_u64(in, "metadata source count");
    for (std::uint64_t i = 0; i < source_count; ++i) {
      const std::uint64_t node_u64 = read_u64(in, "metadata source node");
      (void)read_u64(in, "metadata source site");
      (void)read_u64(in, "metadata source pin");
      ++sources.raw_source_count;

      if (node_u64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        ++sources.invalid_source_count;
        continue;
      }
      const int node = static_cast<int>(node_u64);
      if (node < 0 || static_cast<Offset>(node) >= graph_rows) {
        ++sources.invalid_source_count;
        continue;
      }
      if (seen.insert(node).second) {
        sources.unique_sources.push_back(node);
      }
    }

    const std::uint64_t sink_count = read_u64(in, "metadata sink count");
    skip_bytes(in,
               checked_byte_count(sink_count, 3 * sizeof(std::uint64_t), "metadata sinks"),
               "metadata sinks");
  }

  skip_bytes(in,
             checked_byte_count(logical_cell_count,
                                3 * sizeof(std::uint64_t),
                                "metadata logical cells"),
             "metadata logical cells");
  skip_bytes(in,
             checked_byte_count(logical_net_count,
                                4 * sizeof(std::uint64_t),
                                "metadata logical nets"),
             "metadata logical nets");
  skip_bytes(in,
             checked_byte_count(logical_port_instance_count,
                                7 * sizeof(std::uint64_t),
                                "metadata logical port instances"),
             "metadata logical port instances");
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t), "metadata blocked nodes"),
             "metadata blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count,
                                sizeof(std::uint64_t),
                                "metadata sink stop nodes"),
             "metadata sink stop nodes");
  skip_bytes(in, physical_netlist_byte_count, "metadata physical bytes");
  skip_bytes(in, logical_netlist_byte_count, "metadata logical bytes");
  return sources;
}

// CPU helper. Inputs: text and field name. Output: parsed size_t.
// Purpose: parse nonnegative integer CLI options.
std::size_t parse_size(const std::string& text, const char* name) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid unsigned integer for ") + name);
  }
  return static_cast<std::size_t>(value);
}

// CPU helper. Inputs: text and field name. Output: parsed int.
// Purpose: parse --max-iters, including -1 for the default n - 1 limit.
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

// CPU helper. Inputs: argv index and option name. Output: option value string.
// Purpose: consume the value following a CLI option.
std::string require_value(int* index, int argc, char** argv, const char* option) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value after ") + option);
  }
  ++(*index);
  return argv[*index];
}

// CPU helper. Inputs: program name. Output: usage text on stderr.
// Purpose: document bf1's bare-bones command line.
void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <outgoing.csrbin> <metadata.ifmeta.bin> [summary.csv] [options]\n\n"
      << "Options:\n"
      << "  --max-iters <n>             Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --source-progress-every <n> Print detailed progress every n sources. Default: 100; 0 disables detailed lines.\n"
      << "  --source-limit <n>          Run only the first n unique valid source nodes. Default: 0 = all.\n"
      << "  --help                      Print this message.\n";
}

// CPU helper. Inputs: argc and argv. Output: parsed paths and options.
// Purpose: accept required CSR/metadata paths, optional summary CSV, and simple controls.
ParsedArgs parse_args(int argc, char** argv) {
  ParsedArgs parsed;
  int positional_count = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--max-iters") {
      parsed.options.max_iters =
          parse_int(require_value(&i, argc, argv, "--max-iters"), "--max-iters");
    } else if (arg == "--source-progress-every") {
      parsed.options.source_progress_every =
          parse_size(require_value(&i, argc, argv, "--source-progress-every"),
                     "--source-progress-every");
    } else if (arg == "--source-limit") {
      parsed.options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      if (positional_count == 0) {
        parsed.csr_path = arg;
      } else if (positional_count == 1) {
        parsed.metadata_path = arg;
      } else if (positional_count == 2) {
        parsed.output_path = arg;
      } else {
        throw std::runtime_error("too many positional arguments");
      }
      ++positional_count;
    }
  }

  if (parsed.csr_path.empty() || parsed.metadata_path.empty()) {
    throw std::runtime_error("expected <outgoing.csrbin> <metadata.ifmeta.bin>");
  }
  if (parsed.options.max_iters < -1) {
    throw std::runtime_error("--max-iters must be -1 or nonnegative");
  }
  return parsed;
}

// CPU helper. Inputs: completed count, total count, visual width. Output: progress bar string.
// Purpose: create a compact progress bar for source-level status.
std::string progress_bar(std::size_t completed, std::size_t total, int width) {
  if (width <= 0) return "";
  const double fraction =
      total == 0 ? 1.0 : std::min(1.0, static_cast<double>(completed) /
                                           static_cast<double>(total));
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

// CPU helper. Inputs: completed count, total count, and whether to end the line.
// Output: progress bar printed to stdout.
// Purpose: show how many unique sources have completed SSSP.
void print_source_progress(std::size_t completed, std::size_t total, bool final_line) {
  const double percent =
      total == 0 ? 100.0 : 100.0 * static_cast<double>(completed) /
                                static_cast<double>(total);
  std::ostringstream percent_text;
  percent_text << std::fixed
               << std::setprecision(percent > 0.0 && percent < 0.1 ? 4 : 1)
               << std::min(100.0, percent);
  std::cout << '\r'
            << "[bf1] sources "
            << progress_bar(completed, total, 30)
            << ' ' << percent_text.str() << "%"
            << " " << completed << "/" << total
            << "        "
            << std::flush;
  if (final_line) {
    std::cout << '\n';
  }
}

// CPU helper. Inputs: optional output path. Output: opened CSV stream, or unopened stream.
// Purpose: provide minimal per-source summary output without copying distances.
std::ofstream open_summary_output(const std::filesystem::path& output_path) {
  std::ofstream out;
  if (output_path.empty()) {
    return out;
  }
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  out.open(output_path);
  if (!out) {
    throw std::runtime_error("could not open summary output: " + output_path.string());
  }
  out << "source,iterations_used,converged\n";
  return out;
}

// CPU helper. Inputs: summary stream, source node, result. Output: one CSV row if stream is open.
// Purpose: record minimal Bellman-Ford status per source.
void write_summary_record(std::ofstream& out,
                          int source,
                          const rips_sssp_new::OutgoingSsspResult& result) {
  if (!out) return;
  out << source << ','
      << result.iterations_used << ','
      << (result.converged ? "yes" : "no") << '\n';
}

// CPU function. Inputs: command-line arguments. Output: process exit code.
// Purpose: load inputs once, copy CSR once, run bare global Bellman-Ford for each unique source.
int main_impl(int argc, char** argv) {
  const auto total_begin = std::chrono::steady_clock::now();
  const ParsedArgs args = parse_args(argc, argv);

  std::cout << "[bf1] loading outgoing CSR: " << args.csr_path.string() << '\n';
  HostOutgoingCsrF32 graph = load_outgoing_csrbin(args.csr_path);
  std::cout << "[bf1] graph rows=" << graph.rows
            << " nnz=" << graph.nnz
            << " orientation=outgoing(row u stores edges u -> v)\n";

  std::cout << "[bf1] loading metadata: " << args.metadata_path.string() << '\n';
  RouteSources sources =
      load_route_sources_from_metadata(args.metadata_path, graph.rows, graph.nnz);
  if (sources.metadata_node_count != static_cast<std::uint64_t>(graph.rows)) {
    throw std::runtime_error("metadata node count does not match outgoing CSR rows");
  }

  const std::size_t source_count =
      args.options.source_limit == 0
          ? sources.unique_sources.size()
          : std::min(args.options.source_limit, sources.unique_sources.size());
  std::cout << "[bf1] raw_metadata_sources=" << sources.raw_source_count
            << " invalid_sources=" << sources.invalid_source_count
            << " unique_valid_sources=" << sources.unique_sources.size()
            << " sources_to_run=" << source_count << '\n';

  std::ofstream summary_out = open_summary_output(args.output_path);
  if (source_count == 0) {
    print_source_progress(0, 0, true);
    std::cout << "[bf1] no valid unique sources to run\n";
    return 0;
  }

  hipStream_t stream = nullptr;
  DeviceOutgoingCsrOwner d_graph;
  rips_sssp_new::DeviceSsspWorkspace workspace;
  bool stream_created = false;
  bool graph_copied = false;
  bool workspace_allocated = false;

  try {
    rips_sssp_new::check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    stream_created = true;

    std::cout << "[bf1] copying outgoing CSR to GPU once\n";
    d_graph = copy_outgoing_csr_to_device(graph, stream);
    graph_copied = true;

    std::cout << "[bf1] allocating reusable SSSP workspace\n";
    workspace = rips_sssp_new::make_sssp_workspace(graph.rows, 0, stream);
    workspace_allocated = true;

    print_source_progress(0, source_count, false);
    for (std::size_t source_index = 0; source_index < source_count; ++source_index) {
      const int source = sources.unique_sources[source_index];
      const rips_sssp_new::OutgoingSsspResult result =
          rips_sssp_new::run_outgoing_dense_sssp(d_graph.view,
                                                 workspace,
                                                 source,
                                                 args.options.max_iters,
                                                 stream);
      write_summary_record(summary_out, source, result);

      const std::size_t completed = source_index + 1;
      const bool final_source = completed == source_count;
      const bool detailed =
          args.options.source_progress_every > 0 &&
          (completed % args.options.source_progress_every == 0 || final_source);
      print_source_progress(completed, source_count, detailed || final_source);

      if (detailed) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - total_begin).count();
        const double sources_per_sec =
            elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
        std::cout << "[bf1] source " << completed << "/" << source_count
                  << " node=" << source
                  << " iterations_used=" << result.iterations_used
                  << " converged=" << (result.converged ? "yes" : "no")
                  << " elapsed_sec=" << std::fixed << std::setprecision(3)
                  << elapsed
                  << " avg_sources_per_sec=" << std::setprecision(3)
                  << sources_per_sec << '\n';
      }
    }

    rips_sssp_new::check_hip(hipStreamSynchronize(stream), "final stream sync");
    if (workspace_allocated) {
      rips_sssp_new::free_sssp_workspace(&workspace);
      workspace_allocated = false;
    }
    if (graph_copied) {
      free_device_outgoing_csr(&d_graph);
      graph_copied = false;
    }
    rips_sssp_new::check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    stream_created = false;
  } catch (...) {
    if (workspace_allocated) {
      rips_sssp_new::free_sssp_workspace(&workspace);
    }
    if (graph_copied) {
      free_device_outgoing_csr(&d_graph);
    }
    if (stream_created) {
      const hipError_t destroy_status = hipStreamDestroy(stream);
      if (destroy_status != hipSuccess) {
        std::cerr << "[bf1] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
    }
    throw;
  }

  const auto total_end = std::chrono::steady_clock::now();
  const double total_seconds =
      std::chrono::duration<double>(total_end - total_begin).count();
  std::cout << "[bf1] completed " << source_count
            << " source SSSP runs"
            << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
            << total_seconds << '\n';
  if (!args.output_path.empty()) {
    std::cout << "[bf1] wrote summary: " << args.output_path.string() << '\n';
  }
  return 0;
}

}  // namespace

// CPU function. Inputs: command-line arguments. Output: process exit code.
// Purpose: top-level exception boundary for the bf1 executable.
int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "[bf1] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
