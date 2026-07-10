// Runs HIP outgoing-frontier SSSP from route sources listed in a RIPS
// interchange metadata sidecar and writes validator-compatible shortest-path
// JSONL.
//
// Build from rips2026-amd on an AMD ROCm/HIP machine:
//   hipcc -std=c++17 -O3 -x hip \
//     -I HIP_kernel/bellman_ford/src \
//     -I HIP_kernel/minplus_mm/src \
//     SSSP/src/bf_outgoing4.cpp \
//     -o bf_outgoing4
//
// Run:
//   ./bf_outgoing4 data/logicnets_jscl.csrbin \
//     data/logicnets_jscl.csrbin.ifmeta.bin paths.jsonl

#include "../../HIP_kernel/bellman_ford/src/bf_hip_CSR.hpp"

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
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char OUTGOING_CSR_MAGIC[8] = {'R', 'I', 'P', 'S', 'O', 'C', 'S', '1'};
constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t EXPECTED_CSR_VERSION = 1;
constexpr std::uint64_t MIN_OUTGOING_CSR_VERSION = 1;
constexpr std::uint64_t CURRENT_OUTGOING_CSR_VERSION = 2;
constexpr std::uint64_t EXPECTED_METADATA_VERSION = 2;
constexpr std::uint64_t EXPECTED_INCOMING_EDGE_ORIENTATION = 1;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();
constexpr unsigned int kPackedNoPredEdge = 0xffffffffu;
constexpr unsigned int kPackedInfinityBits = 0x7f800000u;

using Offset = minplus_sparse::Offset;
using Index = minplus_sparse::Index;

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

struct OutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  std::vector<Offset> rowptr;
  std::vector<Index> degree;
  std::vector<Index> to;
  std::vector<float> values;
  std::vector<Offset> incoming_edge;
};

struct DeviceOutgoingCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const Offset* rowptr = nullptr;
  const Index* degree = nullptr;
  const Index* to = nullptr;
  const float* values = nullptr;
  const Offset* incoming_edge = nullptr;
};

struct DeviceOutgoingCsrOwner {
  DeviceOutgoingCsrF32 view{};
  Offset* rowptr = nullptr;
  Index* degree = nullptr;
  Index* to = nullptr;
  float* values = nullptr;
  Offset* incoming_edge = nullptr;

  DeviceOutgoingCsrOwner() = default;
  DeviceOutgoingCsrOwner(const DeviceOutgoingCsrOwner&) = delete;
  DeviceOutgoingCsrOwner& operator=(const DeviceOutgoingCsrOwner&) = delete;

  DeviceOutgoingCsrOwner(DeviceOutgoingCsrOwner&& other) noexcept {
    *this = std::move(other);
  }

  DeviceOutgoingCsrOwner& operator=(DeviceOutgoingCsrOwner&& other) noexcept {
    if (this != &other) {
      reset();
      view = other.view;
      rowptr = other.rowptr;
      degree = other.degree;
      to = other.to;
      values = other.values;
      incoming_edge = other.incoming_edge;
      other.view = {};
      other.rowptr = nullptr;
      other.degree = nullptr;
      other.to = nullptr;
      other.values = nullptr;
      other.incoming_edge = nullptr;
    }
    return *this;
  }

  ~DeviceOutgoingCsrOwner() {
    reset();
  }

  void reset() {
    if (rowptr) {
      (void)hipFree(rowptr);
      rowptr = nullptr;
    }
    if (degree) {
      (void)hipFree(degree);
      degree = nullptr;
    }
    if (to) {
      (void)hipFree(to);
      to = nullptr;
    }
    if (values) {
      (void)hipFree(values);
      values = nullptr;
    }
    if (incoming_edge) {
      (void)hipFree(incoming_edge);
      incoming_edge = nullptr;
    }
    view = {};
  }
};

struct Options {
  int max_iters = -1;
  std::size_t net_limit = 0;
  std::size_t source_limit = 0;
  std::size_t query_limit = 0;
  std::size_t source_progress_every = 100;
  double abs_tolerance = 1e-3;
  double rel_tolerance = 1e-5;
  bool include_unreached = true;
  bool profile_frontier_degrees = false;
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

template <typename T>
void write_array(std::ofstream& out,
                 const std::vector<T>& values,
                 const char* name) {
  if (values.empty()) return;

  const std::size_t bytes = values.size() * sizeof(T);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    throw std::runtime_error(std::string("failed while writing ") + name);
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

void validate_outgoing_csr(const OutgoingCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("outgoing CSR nnz must be nonnegative");
  }
  if (graph.rowptr.size() != static_cast<std::size_t>(graph.rows + 1) ||
      graph.degree.size() != static_cast<std::size_t>(graph.rows) ||
      graph.to.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.incoming_edge.size() != static_cast<std::size_t>(graph.nnz)) {
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
    const Offset degree = end - begin;
    if (degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    if (graph.degree[static_cast<std::size_t>(row)] !=
        static_cast<Index>(degree)) {
      throw std::runtime_error("outgoing CSR degree array does not match rowptr");
    }
  }

  for (std::size_t edge = 0; edge < graph.to.size(); ++edge) {
    if (graph.to[edge] < 0 ||
        static_cast<Offset>(graph.to[edge]) >= graph.cols) {
      throw std::runtime_error("outgoing CSR contains an out-of-range destination");
    }
    if (graph.incoming_edge[edge] < 0 || graph.incoming_edge[edge] >= graph.nnz) {
      throw std::runtime_error("outgoing CSR contains an out-of-range incoming edge id");
    }
    if (static_cast<std::uint64_t>(graph.incoming_edge[edge]) >=
        static_cast<std::uint64_t>(kPackedNoPredEdge)) {
      throw std::runtime_error("outgoing CSR incoming edge id is too large");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("outgoing CSR values must be finite nonnegative costs");
    }
  }
}

OutgoingCsrF32 build_outgoing_csr(const HostCsrF32& incoming) {
  OutgoingCsrF32 outgoing;
  outgoing.rows = incoming.rows;
  outgoing.cols = incoming.cols;
  outgoing.nnz = incoming.nnz;
  outgoing.rowptr.assign(static_cast<std::size_t>(incoming.rows + 1), 0);
  outgoing.degree.assign(static_cast<std::size_t>(incoming.rows), 0);
  outgoing.to.resize(static_cast<std::size_t>(incoming.nnz));
  outgoing.values.resize(static_cast<std::size_t>(incoming.nnz));
  outgoing.incoming_edge.resize(static_cast<std::size_t>(incoming.nnz));

  for (Offset edge = 0; edge < incoming.nnz; ++edge) {
    const Index from = incoming.colind[static_cast<std::size_t>(edge)];
    ++outgoing.rowptr[static_cast<std::size_t>(from + 1)];
  }

  for (Offset row = 0; row < incoming.rows; ++row) {
    outgoing.rowptr[static_cast<std::size_t>(row + 1)] +=
        outgoing.rowptr[static_cast<std::size_t>(row)];
  }

  for (Offset row = 0; row < incoming.rows; ++row) {
    const Offset degree = outgoing.rowptr[static_cast<std::size_t>(row + 1)] -
                          outgoing.rowptr[static_cast<std::size_t>(row)];
    if (degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
      throw std::runtime_error("outgoing CSR row degree is too large");
    }
    outgoing.degree[static_cast<std::size_t>(row)] =
        static_cast<Index>(degree);
  }

  std::vector<Offset> cursor = outgoing.rowptr;
  for (Offset to = 0; to < incoming.rows; ++to) {
    for (Offset edge = incoming.rowptr[static_cast<std::size_t>(to)];
         edge < incoming.rowptr[static_cast<std::size_t>(to + 1)];
         ++edge) {
      const Index from = incoming.colind[static_cast<std::size_t>(edge)];
      const Offset dst = cursor[static_cast<std::size_t>(from)]++;
      outgoing.to[static_cast<std::size_t>(dst)] = static_cast<Index>(to);
      outgoing.values[static_cast<std::size_t>(dst)] =
          incoming.values[static_cast<std::size_t>(edge)];
      outgoing.incoming_edge[static_cast<std::size_t>(dst)] = edge;
    }
  }

  validate_outgoing_csr(outgoing);
  return outgoing;
}

std::filesystem::path outgoing_csr_cache_path(const std::filesystem::path& csr_path) {
  std::filesystem::path filename = csr_path.filename();
  if (filename.extension() == ".csrbin") {
    filename = filename.stem();
  }
  const std::string out_name = filename.string() + ".outgoing.csrbin";
  return std::filesystem::path("SSSP") / "data" / out_name;
}

void write_outgoing_csrbin(const std::filesystem::path& path,
                           const OutgoingCsrF32& graph) {
  validate_outgoing_csr(graph);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("could not open outgoing CSR file for writing: " +
                             path.string());
  }

  out.write(OUTGOING_CSR_MAGIC, sizeof(OUTGOING_CSR_MAGIC));
  const std::uint64_t header[] = {
      CURRENT_OUTGOING_CSR_VERSION,
      EXPECTED_OUTGOING_EDGE_ORIENTATION,
      static_cast<std::uint64_t>(graph.rows),
      static_cast<std::uint64_t>(graph.cols),
      static_cast<std::uint64_t>(graph.nnz),
      static_cast<std::uint64_t>(graph.rowptr.size()),
      static_cast<std::uint64_t>(graph.degree.size()),
      static_cast<std::uint64_t>(graph.to.size()),
      static_cast<std::uint64_t>(graph.values.size()),
      static_cast<std::uint64_t>(graph.incoming_edge.size())};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  if (!out) {
    throw std::runtime_error("failed while writing outgoing CSR header");
  }

  write_array(out, graph.rowptr, "outgoing CSR rowptr");
  write_array(out, graph.degree, "outgoing CSR degree");
  write_array(out, graph.to, "outgoing CSR destinations");
  write_array(out, graph.values, "outgoing CSR values");
  write_array(out, graph.incoming_edge, "outgoing CSR incoming-edge map");
}

OutgoingCsrF32 load_outgoing_csrbin(const std::filesystem::path& path) {
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
  std::uint64_t degree_count = rows;
  if (version >= 2) {
    degree_count = read_u64(in, "outgoing CSR degree count");
  }
  const std::uint64_t to_count = read_u64(in, "outgoing CSR destination count");
  const std::uint64_t values_count = read_u64(in, "outgoing CSR values count");
  const std::uint64_t incoming_edge_count =
      read_u64(in, "outgoing CSR incoming-edge count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("outgoing CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      nnz > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("outgoing CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || to_count != nnz ||
      values_count != nnz || incoming_edge_count != nnz ||
      degree_count != rows) {
    throw std::runtime_error("outgoing CSR header counts are inconsistent");
  }

  OutgoingCsrF32 graph;
  graph.rows = static_cast<Offset>(rows);
  graph.cols = static_cast<Offset>(cols);
  graph.nnz = static_cast<Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "outgoing CSR rowptr");
  if (version >= 2) {
    read_array(in, graph.degree, degree_count, "outgoing CSR degree");
  } else {
    graph.degree.resize(static_cast<std::size_t>(rows));
    for (std::uint64_t row = 0; row < rows; ++row) {
      const Offset degree =
          graph.rowptr[static_cast<std::size_t>(row + 1)] -
          graph.rowptr[static_cast<std::size_t>(row)];
      if (degree < 0 ||
          degree > static_cast<Offset>(std::numeric_limits<Index>::max())) {
        throw std::runtime_error("legacy outgoing CSR row degree is invalid");
      }
      graph.degree[static_cast<std::size_t>(row)] =
          static_cast<Index>(degree);
    }
  }
  read_array(in, graph.to, to_count, "outgoing CSR destinations");
  read_array(in, graph.values, values_count, "outgoing CSR values");
  read_array(in,
             graph.incoming_edge,
             incoming_edge_count,
             "outgoing CSR incoming-edge map");
  validate_outgoing_csr(graph);
  return graph;
}

OutgoingCsrF32 load_or_create_outgoing_csrbin(const HostCsrF32& incoming,
                                              const std::filesystem::path& incoming_path) {
  const std::filesystem::path outgoing_path = outgoing_csr_cache_path(incoming_path);

  if (std::filesystem::exists(outgoing_path)) {
    try {
      OutgoingCsrF32 outgoing = load_outgoing_csrbin(outgoing_path);
      if (outgoing.rows == incoming.rows &&
          outgoing.cols == incoming.cols &&
          outgoing.nnz == incoming.nnz) {
        std::cout << "[bf_outgoing4] loaded outgoing CSR: "
                  << outgoing_path.string() << '\n';
        return outgoing;
      }
      throw std::runtime_error(
          "outgoing CSR cache shape does not match input CSR; remove " +
          outgoing_path.string() + " to regenerate it");
    } catch (const std::exception& ex) {
      throw std::runtime_error("could not use existing outgoing CSR cache " +
                               outgoing_path.string() + ": " + ex.what());
    }
  }

  std::cout << "[bf_outgoing4] building outgoing CSR: "
            << outgoing_path.string() << '\n';
  OutgoingCsrF32 outgoing = build_outgoing_csr(incoming);
  write_outgoing_csrbin(outgoing_path, outgoing);
  std::cout << "[bf_outgoing4] wrote outgoing CSR: "
            << outgoing_path.string() << '\n';
  return outgoing;
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

  // Path strings and logical design name. They are not needed for SSSP output.
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

  // Edge attrs: tile string, pip data index.
  skip_bytes(in, edge_attr_count * 2 * sizeof(std::uint64_t), "metadata edge attrs");

  // PIP data: wire0 string, wire1 string, forward flag.
  skip_bytes(in, pip_data_count * 3 * sizeof(std::uint64_t), "metadata pip data");

  // Site pin attrs: node, site string, pin string.
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

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dist,
                                                int source,
                                                int target,
                                                const Options& options) {
  if (!valid_node(graph, source) || !valid_node(graph, target)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance vector size does not match CSR rows");
  }
  if (source == target) {
    return {};
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (Offset guard = 0; guard < graph.rows && current != source; ++guard) {
    const std::size_t row = static_cast<std::size_t>(current);
    const float current_dist = dist[row];
    Offset best_edge = -1;
    int best_pred = -1;
    double best_error = std::numeric_limits<double>::infinity();

    for (Offset edge = graph.rowptr[row]; edge < graph.rowptr[row + 1]; ++edge) {
      const int pred = graph.colind[static_cast<std::size_t>(edge)];
      const float pred_dist = dist[static_cast<std::size_t>(pred)];
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

std::vector<PathEdge> reconstruct_shortest_path_from_sparse_states(
    const HostCsrF32& incoming,
    const std::unordered_map<int, unsigned long long>& best_states,
    int source,
    int target) {
  if (!valid_node(incoming, source) || !valid_node(incoming, target)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (source == target) {
    return {};
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (Offset guard = 0; guard < incoming.rows && current != source; ++guard) {
    const auto state_it = best_states.find(current);
    if (state_it == best_states.end()) {
      throw std::runtime_error("shortest path predecessor state was not copied");
    }
    const unsigned int edge_bits =
        static_cast<unsigned int>(state_it->second & kPackedNoPredEdge);
    const Offset edge = edge_bits == kPackedNoPredEdge
                            ? -1
                            : static_cast<Offset>(edge_bits);
    if (edge < 0 || edge >= incoming.nnz) {
      throw std::runtime_error("shortest path predecessor is missing");
    }
    if (edge < incoming.rowptr[static_cast<std::size_t>(current)] ||
        edge >= incoming.rowptr[static_cast<std::size_t>(current + 1)]) {
      throw std::runtime_error("shortest path predecessor edge does not enter current node");
    }

    const int pred = incoming.colind[static_cast<std::size_t>(edge)];
    PathEdge path_edge;
    path_edge.from = pred;
    path_edge.to = current;
    path_edge.csr_edge = edge;
    path_edge.cost = incoming.values[static_cast<std::size_t>(edge)];
    reversed.push_back(path_edge);
    current = pred;
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
                           const RoutingMetadata& metadata) {
  out << "{\"type\":\"metadata\",\"format\":\"rips-sssp-paths-v1\""
      << ",\"producer\":\"bf_outgoing4\""
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

Offset max_unique_targets_per_source(const std::vector<SourceWork>& work,
                                     std::size_t source_count,
                                     Offset rows) {
  std::size_t max_targets = 0;
  for (std::size_t i = 0; i < source_count; ++i) {
    std::unordered_set<int> targets;
    targets.reserve(work[i].queries.size());
    for (const Query& query : work[i].queries) {
      targets.insert(query.target);
    }
    max_targets = std::max(max_targets, targets.size());
  }
  if (max_targets > static_cast<std::size_t>(rows)) {
    throw std::runtime_error("unique sink count exceeds graph row count");
  }
  return static_cast<Offset>(max_targets);
}

std::vector<Index> unique_targets_for_queries(const std::vector<Query>& queries) {
  std::vector<Index> targets;
  targets.reserve(queries.size());
  std::unordered_set<int> seen;
  seen.reserve(queries.size() * 2 + 1);
  for (const Query& query : queries) {
    if (seen.insert(query.target).second) {
      targets.push_back(static_cast<Index>(query.target));
    }
  }
  return targets;
}

std::string progress_bar(std::size_t current, std::size_t total, int width) {
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

void print_source_progress(std::size_t completed, std::size_t total, bool final_line) {
  const double percent =
      total == 0 ? 100.0 : 100.0 * static_cast<double>(completed) /
                                static_cast<double>(total);
  std::ostringstream percent_text;
  percent_text << std::fixed
               << std::setprecision(percent > 0.0 && percent < 0.1 ? 4 : 1)
               << std::min(100.0, percent);

  std::cout << '\r'
            << "[bf_outgoing4] sources "
            << progress_bar(completed, total, 30)
            << ' ' << percent_text.str() << "%"
            << " " << completed << "/" << total
            << "        "
            << std::flush;
  if (final_line) {
    std::cout << '\n';
  }
}

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

DeviceOutgoingCsrOwner copy_outgoing_csr_to_device(const OutgoingCsrF32& host,
                                                   hipStream_t stream) {
  DeviceOutgoingCsrOwner device;

  const std::size_t rowptr_bytes = host.rowptr.size() * sizeof(Offset);
  check_hip(hipMalloc(reinterpret_cast<void**>(&device.rowptr), rowptr_bytes),
            "hipMalloc outgoing rowptr");
  check_hip(hipMemcpyAsync(device.rowptr,
                           host.rowptr.data(),
                           rowptr_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy outgoing rowptr to device");

  const std::size_t degree_bytes = host.degree.size() * sizeof(Index);
  check_hip(hipMalloc(reinterpret_cast<void**>(&device.degree), degree_bytes),
            "hipMalloc outgoing degree");
  check_hip(hipMemcpyAsync(device.degree,
                           host.degree.data(),
                           degree_bytes,
                           hipMemcpyHostToDevice,
                           stream),
            "copy outgoing degree to device");

  if (host.nnz > 0) {
    const std::size_t to_bytes = host.to.size() * sizeof(Index);
    const std::size_t values_bytes = host.values.size() * sizeof(float);
    const std::size_t incoming_edge_bytes =
        host.incoming_edge.size() * sizeof(Offset);

    check_hip(hipMalloc(reinterpret_cast<void**>(&device.to), to_bytes),
              "hipMalloc outgoing destinations");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.values), values_bytes),
              "hipMalloc outgoing values");
    check_hip(hipMalloc(reinterpret_cast<void**>(&device.incoming_edge),
                        incoming_edge_bytes),
              "hipMalloc outgoing incoming-edge map");

    check_hip(hipMemcpyAsync(device.to,
                             host.to.data(),
                             to_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy outgoing destinations to device");
    check_hip(hipMemcpyAsync(device.values,
                             host.values.data(),
                             values_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy outgoing values to device");
    check_hip(hipMemcpyAsync(device.incoming_edge,
                             host.incoming_edge.data(),
                             incoming_edge_bytes,
                             hipMemcpyHostToDevice,
                             stream),
              "copy outgoing incoming-edge map to device");
  }

  check_hip(hipStreamSynchronize(stream), "synchronize outgoing CSR copy");

  device.view.rows = host.rows;
  device.view.cols = host.cols;
  device.view.nnz = host.nnz;
  device.view.rowptr = device.rowptr;
  device.view.degree = device.degree;
  device.view.to = device.to;
  device.view.values = device.values;
  device.view.incoming_edge = device.incoming_edge;
  return device;
}

constexpr int kFrontierDegreeBucketCount = 13;

struct FrontierDegreeStats {
  unsigned long long active_vertices;
  unsigned long long total_degree;
  int max_degree;
  int reserved;
  unsigned long long degree_buckets[kFrontierDegreeBucketCount];
};

struct FrontierDegreeProfileSummary {
  std::uint64_t iterations = 0;
  std::uint64_t active_vertices = 0;
  std::uint64_t total_degree = 0;
  std::uint64_t max_frontier_count = 0;
  int max_degree = 0;
  std::uint64_t degree_buckets[kFrontierDegreeBucketCount] = {};

  void add_iteration(const FrontierDegreeStats& stats, int frontier_count) {
    ++iterations;
    active_vertices += stats.active_vertices;
    total_degree += stats.total_degree;
    max_frontier_count = std::max(max_frontier_count,
                                  static_cast<std::uint64_t>(frontier_count));
    max_degree = std::max(max_degree, stats.max_degree);
    for (int bucket = 0; bucket < kFrontierDegreeBucketCount; ++bucket) {
      degree_buckets[bucket] += stats.degree_buckets[bucket];
    }
  }

  void add_summary(const FrontierDegreeProfileSummary& other) {
    iterations += other.iterations;
    active_vertices += other.active_vertices;
    total_degree += other.total_degree;
    max_frontier_count = std::max(max_frontier_count, other.max_frontier_count);
    max_degree = std::max(max_degree, other.max_degree);
    for (int bucket = 0; bucket < kFrontierDegreeBucketCount; ++bucket) {
      degree_buckets[bucket] += other.degree_buckets[bucket];
    }
  }

  double average_degree() const {
    return active_vertices == 0
               ? 0.0
               : static_cast<double>(total_degree) /
                     static_cast<double>(active_vertices);
  }
};

const char* frontier_degree_bucket_label(int bucket) {
  static constexpr const char* labels[kFrontierDegreeBucketCount] = {
      "0",       "1",       "2",       "3-4",     "5-8",
      "9-16",    "17-32",   "33-64",   "65-128",  "129-256",
      "257-512", "513-1024", ">1024"};
  return bucket >= 0 && bucket < kFrontierDegreeBucketCount ? labels[bucket]
                                                            : "?";
}

void print_frontier_degree_profile(const FrontierDegreeProfileSummary& profile) {
  std::cout << "[bf_outgoing4] frontier_degree_profile"
            << " iterations=" << profile.iterations
            << " active_vertices=" << profile.active_vertices
            << " total_outgoing_edges=" << profile.total_degree
            << " avg_degree=" << std::fixed << std::setprecision(3)
            << profile.average_degree()
            << " max_frontier_count=" << profile.max_frontier_count
            << " max_degree=" << profile.max_degree << '\n';

  std::cout << "[bf_outgoing4] frontier_degree_buckets";
  for (int bucket = 0; bucket < kFrontierDegreeBucketCount; ++bucket) {
    const double percent =
        profile.active_vertices == 0
            ? 0.0
            : 100.0 * static_cast<double>(profile.degree_buckets[bucket]) /
                  static_cast<double>(profile.active_vertices);
    std::cout << ' ' << frontier_degree_bucket_label(bucket) << '='
              << profile.degree_buckets[bucket] << '(' << std::fixed
              << std::setprecision(2) << percent << "%)";
  }
  std::cout << '\n';
}

struct IterationStatus {
  int next_count;
  int error_status;
  int reached_target_count;
  unsigned int min_next_frontier_dist_bits;
  unsigned int max_target_dist_bits;
};

struct DeviceSsspWorkspace {
  Offset rows = 0;
  Offset gather_capacity = 0;
  unsigned long long* best_state = nullptr;
  Index* frontier = nullptr;
  Index* next_frontier = nullptr;
  int* next_marks = nullptr;
  IterationStatus* iteration_status = nullptr;
  FrontierDegreeStats* frontier_degree_stats = nullptr;
  Index* gather_nodes = nullptr;
  unsigned long long* gather_states = nullptr;

  DeviceSsspWorkspace() = default;
  DeviceSsspWorkspace(const DeviceSsspWorkspace&) = delete;
  DeviceSsspWorkspace& operator=(const DeviceSsspWorkspace&) = delete;

  DeviceSsspWorkspace(DeviceSsspWorkspace&& other) noexcept {
    *this = std::move(other);
  }

  DeviceSsspWorkspace& operator=(DeviceSsspWorkspace&& other) noexcept {
    if (this != &other) {
      reset();
      rows = other.rows;
      gather_capacity = other.gather_capacity;
      best_state = other.best_state;
      frontier = other.frontier;
      next_frontier = other.next_frontier;
      next_marks = other.next_marks;
      iteration_status = other.iteration_status;
      frontier_degree_stats = other.frontier_degree_stats;
      gather_nodes = other.gather_nodes;
      gather_states = other.gather_states;
      other.rows = 0;
      other.gather_capacity = 0;
      other.best_state = nullptr;
      other.frontier = nullptr;
      other.next_frontier = nullptr;
      other.next_marks = nullptr;
      other.iteration_status = nullptr;
      other.frontier_degree_stats = nullptr;
      other.gather_nodes = nullptr;
      other.gather_states = nullptr;
    }
    return *this;
  }

  ~DeviceSsspWorkspace() {
    reset();
  }

  void reset() {
    if (best_state) {
      (void)hipFree(best_state);
      best_state = nullptr;
    }
    if (frontier) {
      (void)hipFree(frontier);
      frontier = nullptr;
    }
    if (next_frontier) {
      (void)hipFree(next_frontier);
      next_frontier = nullptr;
    }
    if (next_marks) {
      (void)hipFree(next_marks);
      next_marks = nullptr;
    }
    if (iteration_status) {
      (void)hipFree(iteration_status);
      iteration_status = nullptr;
    }
    if (frontier_degree_stats) {
      (void)hipFree(frontier_degree_stats);
      frontier_degree_stats = nullptr;
    }
    if (gather_nodes) {
      (void)hipFree(gather_nodes);
      gather_nodes = nullptr;
    }
    if (gather_states) {
      (void)hipFree(gather_states);
      gather_states = nullptr;
    }
    rows = 0;
    gather_capacity = 0;
  }
};

DeviceSsspWorkspace make_sssp_workspace(Offset rows,
                                        Offset gather_capacity,
                                        bool enable_frontier_degree_profile,
                                        hipStream_t stream) {
  if (gather_capacity < 0 || gather_capacity > rows) {
    throw std::runtime_error("SSSP gather capacity is outside graph bounds");
  }
  DeviceSsspWorkspace workspace;
  workspace.rows = rows;
  workspace.gather_capacity = gather_capacity;
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.best_state),
                      static_cast<std::size_t>(rows) * sizeof(unsigned long long)),
            "hipMalloc SSSP best state");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.frontier),
                      static_cast<std::size_t>(rows) * sizeof(Index)),
            "hipMalloc SSSP frontier");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.next_frontier),
                      static_cast<std::size_t>(rows) * sizeof(Index)),
            "hipMalloc SSSP next frontier");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.next_marks),
                      static_cast<std::size_t>(rows) * sizeof(int)),
            "hipMalloc SSSP next marks");
  check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.iteration_status),
                      sizeof(IterationStatus)),
            "hipMalloc SSSP iteration status");
  if (enable_frontier_degree_profile) {
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.frontier_degree_stats),
                        sizeof(FrontierDegreeStats)),
              "hipMalloc SSSP frontier degree stats");
  }
  if (gather_capacity > 0) {
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_nodes),
                        static_cast<std::size_t>(gather_capacity) * sizeof(Index)),
              "hipMalloc SSSP gather nodes");
    check_hip(hipMalloc(reinterpret_cast<void**>(&workspace.gather_states),
                        static_cast<std::size_t>(gather_capacity) *
                            sizeof(unsigned long long)),
              "hipMalloc SSSP gather states");
  }
  check_hip(hipMemsetAsync(workspace.next_marks,
                           0,
                           static_cast<std::size_t>(rows) * sizeof(int),
                           stream),
            "reset SSSP next marks");
  check_hip(hipStreamSynchronize(stream), "synchronize SSSP workspace allocation");
  return workspace;
}

__device__ __forceinline__ Offset logical_thread_id_1d() {
  return (static_cast<Offset>(blockIdx.x) +
          static_cast<Offset>(blockIdx.y) * static_cast<Offset>(gridDim.x)) *
             static_cast<Offset>(blockDim.x) +
         static_cast<Offset>(threadIdx.x);
}

constexpr unsigned kGridX = 65535u;
constexpr int kFrontierBlockSize = 256;

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

float host_float_from_bits(unsigned int bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

__device__ __forceinline__ unsigned long long pack_state_bits(unsigned int dist_bits,
                                                              unsigned int edge) {
  return (static_cast<unsigned long long>(dist_bits) << 32) |
         static_cast<unsigned long long>(edge);
}

__device__ __forceinline__ unsigned long long pack_state_float(float dist,
                                                               unsigned int edge) {
  return pack_state_bits(__float_as_uint(dist), edge);
}

__device__ __forceinline__ float unpack_state_dist(unsigned long long state) {
  return __uint_as_float(static_cast<unsigned int>(state >> 32));
}

__device__ __forceinline__ bool atomic_relax_state(unsigned long long* addr,
                                                   float value,
                                                   unsigned int edge) {
  const unsigned long long desired = pack_state_float(value, edge);
  unsigned long long old_state = *addr;

  while (desired < old_state) {
    const unsigned long long assumed = old_state;
    old_state = atomicCAS(addr, assumed, desired);
    if (old_state == assumed) {
      return true;
    }
  }

  return false;
}

__device__ __forceinline__ bool finite_device_float(float value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

__device__ __forceinline__ void relax_outgoing_edge(
    Index dst,
    float weight,
    Offset original_edge,
    Offset rows,
    float from_dist,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    IterationStatus* __restrict__ iteration_status) {
  if (dst < 0 || static_cast<Offset>(dst) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  if (!finite_device_float(weight) || weight < 0.0f) {
    atomicExch(&iteration_status->error_status, 3);
    return;
  }

  const float candidate = from_dist + weight;
  if (!finite_device_float(candidate)) {
    return;
  }

  if (original_edge < 0 ||
      original_edge >= static_cast<Offset>(kPackedNoPredEdge)) {
    atomicExch(&iteration_status->error_status, 5);
    return;
  }

  if (atomic_relax_state(&best_state[static_cast<Offset>(dst)],
                         candidate,
                         static_cast<unsigned int>(original_edge))) {
    atomicMin(&iteration_status->min_next_frontier_dist_bits,
              __float_as_uint(candidate));
    const int old_mark = atomicExch(&next_marks[static_cast<Offset>(dst)],
                                    mark_token);
    if (old_mark != mark_token) {
      const int slot = atomicAdd(&iteration_status->next_count, 1);
      if (slot >= rows) {
        atomicExch(&iteration_status->error_status, 4);
      } else {
        next_frontier[slot] = dst;
      }
    }
  }
}

__global__ void init_outgoing_sssp_kernel(Offset rows,
                                          int source,
                                          unsigned long long* best_state,
                                          Index* frontier,
                                          int* next_marks,
                                          IterationStatus* iteration_status) {
  const Offset row = logical_thread_id_1d();
  if (row < rows) {
    best_state[row] =
        row == static_cast<Offset>(source)
            ? pack_state_bits(0u, 0u)
            : pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
    next_marks[row] = 0;
  }
  if (row == 0) {
    frontier[0] = source;
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
    iteration_status->reached_target_count = 0;
    iteration_status->min_next_frontier_dist_bits = kPackedInfinityBits;
    iteration_status->max_target_dist_bits = 0;
  }
}

__global__ void reset_iteration_status_kernel(IterationStatus* iteration_status) {
  if (logical_thread_id_1d() == 0) {
    iteration_status->next_count = 0;
    iteration_status->error_status = 0;
    iteration_status->reached_target_count = 0;
    iteration_status->min_next_frontier_dist_bits = kPackedInfinityBits;
    iteration_status->max_target_dist_bits = 0;
  }
}

// Read only requested rows from the dense packed state. The host uses this in
// batches while following the predecessor chains of the requested sinks.
__global__ void gather_best_states_kernel(
    const unsigned long long* __restrict__ best_state,
    Offset rows,
    const Index* __restrict__ nodes,
    Offset node_count,
    unsigned long long* __restrict__ states) {
  const Offset item = logical_thread_id_1d();
  if (item >= node_count) {
    return;
  }

  const Index node = nodes[item];
  states[item] =
      node >= 0 && static_cast<Offset>(node) < rows
          ? best_state[static_cast<Offset>(node)]
          : pack_state_bits(kPackedInfinityBits, kPackedNoPredEdge);
}

__device__ __forceinline__ int frontier_degree_bucket(int degree) {
  if (degree <= 0) return 0;
  if (degree == 1) return 1;
  if (degree == 2) return 2;
  if (degree <= 4) return 3;
  if (degree <= 8) return 4;
  if (degree <= 16) return 5;
  if (degree <= 32) return 6;
  if (degree <= 64) return 7;
  if (degree <= 128) return 8;
  if (degree <= 256) return 9;
  if (degree <= 512) return 10;
  if (degree <= 1024) return 11;
  return 12;
}

__global__ void profile_frontier_degrees_kernel(
    const Index* __restrict__ degree,
    Offset rows,
    const Index* __restrict__ frontier,
    int frontier_count,
    FrontierDegreeStats* __restrict__ stats,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= static_cast<Offset>(frontier_count)) {
    return;
  }

  const Index node = frontier[item];
  if (node < 0 || static_cast<Offset>(node) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  const int deg = degree[static_cast<Offset>(node)];
  if (deg < 0) {
    atomicExch(&iteration_status->error_status, 2);
    return;
  }

  atomicAdd(&stats->active_vertices, 1ull);
  atomicAdd(&stats->total_degree, static_cast<unsigned long long>(deg));
  atomicMax(&stats->max_degree, deg);
  atomicAdd(&stats->degree_buckets[frontier_degree_bucket(deg)], 1ull);
}

__global__ void update_target_status_kernel(
    const unsigned long long* __restrict__ best_state,
    Offset rows,
    const Index* __restrict__ target_nodes,
    Offset target_count,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= target_count) {
    return;
  }

  const Index target = target_nodes[item];
  if (target < 0 || static_cast<Offset>(target) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  const unsigned long long state = best_state[static_cast<Offset>(target)];
  const unsigned int dist_bits = static_cast<unsigned int>(state >> 32);
  if (dist_bits != kPackedInfinityBits) {
    atomicAdd(&iteration_status->reached_target_count, 1);
    atomicMax(&iteration_status->max_target_dist_bits, dist_bits);
  }
}

__global__ void outgoing_frontier_relax_kernel(
    const Offset* __restrict__ rowptr,
    const Index* __restrict__ degree,
    const Index* __restrict__ to,
    const float* __restrict__ values,
    const Offset* __restrict__ incoming_edge,
    Offset rows,
    Offset nnz,
    const Index* __restrict__ frontier,
    int frontier_count,
    int mark_token,
    unsigned long long* __restrict__ best_state,
    Index* __restrict__ next_frontier,
    int* __restrict__ next_marks,
    IterationStatus* __restrict__ iteration_status) {
  const Offset item = logical_thread_id_1d();
  if (item >= static_cast<Offset>(frontier_count)) {
    return;
  }

  const Index from = frontier[item];
  if (from < 0 || static_cast<Offset>(from) >= rows) {
    atomicExch(&iteration_status->error_status, 1);
    return;
  }

  const float from_dist = unpack_state_dist(best_state[static_cast<Offset>(from)]);
  if (!finite_device_float(from_dist)) {
    return;
  }

  const Offset begin = rowptr[static_cast<Offset>(from)];
  const int deg = degree[static_cast<Offset>(from)];
  const Offset end = begin + static_cast<Offset>(deg);
  if (begin < 0 || deg < 0 || end < begin || end > nnz) {
    atomicExch(&iteration_status->error_status, 2);
    return;
  }

  for (Offset edge = begin; edge < end; ++edge) {
    relax_outgoing_edge(to[edge],
                        values[edge],
                        incoming_edge[edge],
                        rows,
                        from_dist,
                        mark_token,
                        best_state,
                        next_frontier,
                        next_marks,
                        iteration_status);
  }
}

struct OutgoingSsspResult {
  int iterations_used = 0;
  bool converged = false;
  bool target_early_stopped = false;
};

OutgoingSsspResult run_outgoing_frontier_sssp(const DeviceOutgoingCsrF32& graph,
                                              DeviceSsspWorkspace& workspace,
                                              int source,
                                              const std::vector<Index>& target_nodes,
                                              int max_iters,
                                              hipStream_t stream,
                                              FrontierDegreeProfileSummary*
                                                  degree_profile) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("outgoing SSSP expects a nonempty square graph");
  }
  if (workspace.rows != graph.rows) {
    throw std::runtime_error("SSSP workspace row count does not match graph");
  }
  if (source < 0 || static_cast<Offset>(source) >= graph.rows) {
    throw std::runtime_error("source is outside outgoing graph");
  }
  if (target_nodes.size() > static_cast<std::size_t>(workspace.gather_capacity) ||
      (!target_nodes.empty() && !workspace.gather_nodes)) {
    throw std::runtime_error("SSSP target workspace capacity is too small");
  }
  if (max_iters < 0) {
    max_iters = static_cast<int>(graph.rows) - 1;
  }

  const Offset target_count = static_cast<Offset>(target_nodes.size());
  if (!target_nodes.empty()) {
    check_hip(hipMemcpyAsync(workspace.gather_nodes,
                             target_nodes.data(),
                             target_nodes.size() * sizeof(Index),
                             hipMemcpyHostToDevice,
                             stream),
              "copy outgoing SSSP target nodes");
  }

  constexpr int threads = kFrontierBlockSize;
  hipLaunchKernelGGL(init_outgoing_sssp_kernel,
                     grid_for_items(graph.rows, threads),
                     dim3(threads),
                     0,
                     stream,
                     graph.rows,
                     source,
                     workspace.best_state,
                     workspace.frontier,
                     workspace.next_marks,
                     workspace.iteration_status);
  check_hip(hipGetLastError(), "launch outgoing SSSP initialization");
  check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP initialization");

  int frontier_count = 1;
  IterationStatus h_iteration_status{};
  FrontierDegreeStats h_degree_stats{};
  OutgoingSsspResult result;

  for (int iter = 0; iter < max_iters && frontier_count > 0; ++iter) {
    const int current_frontier_count = frontier_count;
    hipLaunchKernelGGL(reset_iteration_status_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch outgoing SSSP status reset");

    if (degree_profile) {
      if (!workspace.frontier_degree_stats) {
        throw std::runtime_error("frontier degree profile workspace is not allocated");
      }
      check_hip(hipMemsetAsync(workspace.frontier_degree_stats,
                               0,
                               sizeof(FrontierDegreeStats),
                               stream),
                "reset outgoing SSSP frontier degree stats");
      hipLaunchKernelGGL(profile_frontier_degrees_kernel,
                         grid_for_items(frontier_count, threads),
                         dim3(threads),
                         0,
                         stream,
                         graph.degree,
                         graph.rows,
                         workspace.frontier,
                         frontier_count,
                         workspace.frontier_degree_stats,
                         workspace.iteration_status);
      check_hip(hipGetLastError(), "launch outgoing SSSP frontier degree profile");
    }

    hipLaunchKernelGGL(outgoing_frontier_relax_kernel,
                       grid_for_items(frontier_count, threads),
                       dim3(threads),
                       0,
                       stream,
                       graph.rowptr,
                       graph.degree,
                       graph.to,
                       graph.values,
                       graph.incoming_edge,
                       graph.rows,
                       graph.nnz,
                       workspace.frontier,
                       frontier_count,
                       iter + 1,
                       workspace.best_state,
                       workspace.next_frontier,
                       workspace.next_marks,
                       workspace.iteration_status);
    check_hip(hipGetLastError(), "launch outgoing SSSP relaxation");

    if (target_count > 0) {
      hipLaunchKernelGGL(update_target_status_kernel,
                         grid_for_items(target_count, threads),
                         dim3(threads),
                         0,
                         stream,
                         workspace.best_state,
                         workspace.rows,
                         workspace.gather_nodes,
                         target_count,
                         workspace.iteration_status);
      check_hip(hipGetLastError(), "launch outgoing SSSP target status");
    }

    if (degree_profile) {
      check_hip(hipMemcpyAsync(&h_degree_stats,
                               workspace.frontier_degree_stats,
                               sizeof(FrontierDegreeStats),
                               hipMemcpyDeviceToHost,
                               stream),
                "copy outgoing SSSP frontier degree stats");
    }
    check_hip(hipMemcpyAsync(&h_iteration_status,
                             workspace.iteration_status,
                             sizeof(IterationStatus),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy outgoing SSSP iteration status");
    check_hip(hipStreamSynchronize(stream), "synchronize outgoing SSSP iteration");
    if (h_iteration_status.error_status != 0) {
      throw std::runtime_error("outgoing SSSP relaxation saw invalid graph data");
    }
    if (degree_profile) {
      degree_profile->add_iteration(h_degree_stats, current_frontier_count);
    }

    frontier_count = h_iteration_status.next_count;
    if (frontier_count > 0 &&
        target_count > 0 &&
        h_iteration_status.reached_target_count == static_cast<int>(target_count) &&
        h_iteration_status.min_next_frontier_dist_bits >
            h_iteration_status.max_target_dist_bits) {
      result.target_early_stopped = true;
      result.iterations_used = iter + 1;
      break;
    }
    std::swap(workspace.frontier, workspace.next_frontier);
    result.iterations_used = iter + 1;
  }

  result.converged = frontier_count == 0;
  return result;
}

std::unordered_map<int, unsigned long long> gather_requested_path_states(
    const HostCsrF32& incoming,
    DeviceSsspWorkspace& workspace,
    int source,
    const std::vector<Query>& queries,
    hipStream_t stream) {
  if (workspace.rows != incoming.rows) {
    throw std::runtime_error("SSSP gather workspace row count does not match graph");
  }
  if (!valid_node(incoming, source)) {
    throw std::runtime_error("source is outside graph during sparse result gather");
  }

  std::vector<Index> pending;
  pending.reserve(queries.size());
  std::unordered_set<int> scheduled;
  scheduled.reserve(queries.size() * 2 + 1);
  for (const Query& query : queries) {
    if (!valid_node(incoming, query.target)) {
      throw std::runtime_error("target is outside graph during sparse result gather");
    }
    if (scheduled.insert(query.target).second) {
      pending.push_back(static_cast<Index>(query.target));
    }
  }

  if (pending.size() > static_cast<std::size_t>(workspace.gather_capacity) ||
      (!pending.empty() && (!workspace.gather_nodes || !workspace.gather_states))) {
    throw std::runtime_error("SSSP sparse result gather capacity is too small");
  }

  std::unordered_map<int, unsigned long long> states_by_node;
  states_by_node.reserve(pending.size() * 4 + 1);
  std::vector<unsigned long long> gathered_states;
  gathered_states.reserve(static_cast<std::size_t>(workspace.gather_capacity));
  std::vector<Index> next_pending;
  next_pending.reserve(pending.size());

  constexpr int threads = kFrontierBlockSize;
  for (Offset depth = 0; !pending.empty() && depth < incoming.rows; ++depth) {
    const Offset pending_count = static_cast<Offset>(pending.size());
    check_hip(hipMemcpyAsync(workspace.gather_nodes,
                             pending.data(),
                             pending.size() * sizeof(Index),
                             hipMemcpyHostToDevice,
                             stream),
              "copy sparse SSSP gather nodes");
    hipLaunchKernelGGL(gather_best_states_kernel,
                       grid_for_items(pending_count, threads),
                       dim3(threads),
                       0,
                       stream,
                       workspace.best_state,
                       workspace.rows,
                       workspace.gather_nodes,
                       pending_count,
                       workspace.gather_states);
    check_hip(hipGetLastError(), "launch sparse SSSP state gather");

    gathered_states.resize(pending.size());
    check_hip(hipMemcpyAsync(gathered_states.data(),
                             workspace.gather_states,
                             gathered_states.size() * sizeof(unsigned long long),
                             hipMemcpyDeviceToHost,
                             stream),
              "copy sparse SSSP gathered states");
    check_hip(hipStreamSynchronize(stream),
              "synchronize sparse SSSP state gather");

    next_pending.clear();
    for (std::size_t i = 0; i < pending.size(); ++i) {
      const int node = pending[i];
      const unsigned long long state = gathered_states[i];
      states_by_node.emplace(node, state);

      if (node == source) {
        continue;
      }
      const unsigned int dist_bits = static_cast<unsigned int>(state >> 32);
      const float distance = host_float_from_bits(dist_bits);
      if (!std::isfinite(distance)) {
        continue;
      }

      const unsigned int edge_bits =
          static_cast<unsigned int>(state & kPackedNoPredEdge);
      if (edge_bits == kPackedNoPredEdge) {
        throw std::runtime_error("reachable sparse SSSP state has no predecessor");
      }
      const Offset edge = static_cast<Offset>(edge_bits);
      if (edge < 0 || edge >= incoming.nnz ||
          edge < incoming.rowptr[static_cast<std::size_t>(node)] ||
          edge >= incoming.rowptr[static_cast<std::size_t>(node + 1)]) {
        throw std::runtime_error("sparse SSSP predecessor edge is outside its row");
      }

      const int pred = incoming.colind[static_cast<std::size_t>(edge)];
      if (!valid_node(incoming, pred)) {
        throw std::runtime_error("sparse SSSP predecessor node is outside graph");
      }
      if (scheduled.insert(pred).second) {
        next_pending.push_back(static_cast<Index>(pred));
      }
    }

    if (next_pending.size() >
        static_cast<std::size_t>(workspace.gather_capacity)) {
      throw std::runtime_error("SSSP sparse result gather exceeded its capacity");
    }
    pending.swap(next_pending);
  }

  if (!pending.empty()) {
    throw std::runtime_error("sparse SSSP predecessor chain exceeded graph rows");
  }
  return states_by_node;
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
      << "  --max-iters <n>           Bellman-Ford relaxation limit. Default: -1 = n - 1.\n"
      << "  --net-limit <n>           Use only the first n route requests.\n"
      << "  --source-limit <n>        Run only the first n unique source nodes.\n"
      << "  --query-limit <n>         Emit at most n source-sink path records.\n"
      << "  --source-progress-every <n>\n"
      << "                              Print source summary every n sources. Default: 100.\n"
      << "  --profile-frontier-degrees\n"
      << "                              Profile outgoing-degree distribution of active frontiers.\n"
      << "  --skip-unreached          Do not write JSONL records for unreachable sinks.\n"
      << "  --abs-tol <x>             Path reconstruction absolute tolerance. Default: 1e-3.\n"
      << "  --rel-tol <x>             Path reconstruction relative tolerance. Default: 1e-5.\n"
      << "  --help                    Print this message.\n";
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
    if (arg == "--max-iters") {
      options.max_iters = parse_int(require_value(&i, argc, argv, "--max-iters"), "--max-iters");
    } else if (arg == "--net-limit") {
      options.net_limit = parse_size(require_value(&i, argc, argv, "--net-limit"), "--net-limit");
    } else if (arg == "--source-limit") {
      options.source_limit =
          parse_size(require_value(&i, argc, argv, "--source-limit"), "--source-limit");
    } else if (arg == "--query-limit") {
      options.query_limit =
          parse_size(require_value(&i, argc, argv, "--query-limit"), "--query-limit");
    } else if (arg == "--source-progress-every" || arg == "--bf-progress-every") {
      options.source_progress_every =
          parse_size(require_value(&i, argc, argv, arg.c_str()), arg.c_str());
    } else if (arg == "--profile-frontier-degrees") {
      options.profile_frontier_degrees = true;
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
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto total_begin = std::chrono::steady_clock::now();
    std::filesystem::path csr_path;
    std::filesystem::path metadata_path;
    std::filesystem::path output_path;
    const Options options = parse_options(argc, argv, &csr_path, &metadata_path, &output_path);

    std::cout << "[bf_outgoing4] loading CSR: " << csr_path.string() << '\n';
    HostCsrF32 graph = load_csrbin(csr_path);
    std::cout << "[bf_outgoing4] graph rows=" << graph.rows
              << " nnz=" << graph.nnz
              << " orientation=incoming(row v, col u means u -> v)\n";

    OutgoingCsrF32 outgoing_graph = load_or_create_outgoing_csrbin(graph, csr_path);
    std::cout << "[bf_outgoing4] outgoing graph rows=" << outgoing_graph.rows
              << " nnz=" << outgoing_graph.nnz
              << " orientation=outgoing(row u stores edges u -> v)\n";

    std::cout << "[bf_outgoing4] loading metadata: " << metadata_path.string() << '\n';
    RoutingMetadata metadata = load_interchange_metadata(metadata_path);
    if (metadata.node_device_ids.size() != static_cast<std::size_t>(graph.rows)) {
      throw std::runtime_error("metadata node count does not match CSR rows");
    }
    if (metadata.edge_attr_count != static_cast<std::uint64_t>(graph.nnz)) {
      throw std::runtime_error("metadata edge attribute count does not match CSR nnz");
    }
    std::cout << "[bf_outgoing4] route_requests="
              << metadata.route_requests.size() << '\n';

    std::size_t total_queries = 0;
    std::vector<SourceWork> work =
        build_source_work(metadata, graph, options, &total_queries);
    const std::size_t source_count =
        options.source_limit == 0
            ? work.size()
            : std::min(options.source_limit, work.size());
    const Offset gather_capacity =
        max_unique_targets_per_source(work, source_count, graph.rows);
    std::cout << "[bf_outgoing4] unique_sources=" << work.size()
              << " sources_to_run=" << source_count
              << " source_sink_queries=" << total_queries
              << " max_unique_sinks_per_source=" << gather_capacity << '\n';

    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path);
    if (!out) {
      throw std::runtime_error("could not open output JSONL file: " + output_path.string());
    }
    write_metadata_record(out, graph, metadata);

    hipStream_t stream = nullptr;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    std::uint64_t paths_written = 0;
    std::uint64_t reached_count = 0;
    std::uint64_t unreached_count = 0;
    FrontierDegreeProfileSummary total_frontier_degree_profile;

    try {
      std::cout << "[bf_outgoing4] copying outgoing CSR to GPU once\n";
      DeviceOutgoingCsrOwner d_graph =
          copy_outgoing_csr_to_device(outgoing_graph, stream);
      DeviceSsspWorkspace sssp_workspace =
          make_sssp_workspace(outgoing_graph.rows,
                              gather_capacity,
                              options.profile_frontier_degrees,
                              stream);

      print_source_progress(0, source_count, source_count == 0);
      for (std::size_t source_index = 0; source_index < source_count; ++source_index) {
        const SourceWork& source_work = work[source_index];
        if (source_work.queries.empty()) {
          print_source_progress(source_index + 1,
                                source_count,
                                source_index + 1 == source_count);
          continue;
        }
        const std::uint64_t paths_before = paths_written;
        FrontierDegreeProfileSummary source_frontier_degree_profile;
        const std::vector<Index> target_nodes =
            unique_targets_for_queries(source_work.queries);

        OutgoingSsspResult result =
            run_outgoing_frontier_sssp(d_graph.view,
                                       sssp_workspace,
                                       source_work.source,
                                       target_nodes,
                                       options.max_iters,
                                       stream,
                                       options.profile_frontier_degrees
                                           ? &source_frontier_degree_profile
                                           : nullptr);
        if (options.profile_frontier_degrees) {
          total_frontier_degree_profile.add_summary(source_frontier_degree_profile);
        }
        const std::unordered_map<int, unsigned long long> sparse_states =
            gather_requested_path_states(graph,
                                         sssp_workspace,
                                         source_work.source,
                                         source_work.queries,
                                         stream);

        for (const Query& query : source_work.queries) {
          const auto state_it = sparse_states.find(query.target);
          if (state_it == sparse_states.end()) {
            throw std::runtime_error("requested sink state was not gathered");
          }
          const unsigned int dist_bits =
              static_cast<unsigned int>(state_it->second >> 32);
          const float distance = host_float_from_bits(dist_bits);
          const bool reached = std::isfinite(distance);
          if (!reached) {
            ++unreached_count;
            if (options.include_unreached) {
              write_path_record(out, query, false, distance, {}, {});
              ++paths_written;
            }
            continue;
          }

          std::vector<PathEdge> edges =
              reconstruct_shortest_path_from_sparse_states(graph,
                                                           sparse_states,
                                                           query.source,
                                                           query.target);
          std::vector<int> nodes = nodes_from_edges(query.source, edges);
          write_path_record(out, query, true, distance, nodes, edges);
          ++reached_count;
          ++paths_written;
        }

        const std::size_t completed = source_index + 1;
        const bool final_source = completed == source_count;
        print_source_progress(completed, source_count, final_source);

        const bool should_print_summary =
            options.source_progress_every > 0 &&
            (completed % options.source_progress_every == 0 || final_source);
        if (should_print_summary) {
          if (!final_source) {
            std::cout << '\n';
          }
          const auto now = std::chrono::steady_clock::now();
          const double total_seconds_so_far =
              std::chrono::duration<double>(now - total_begin).count();
          std::cout << "[bf_outgoing4] source " << completed
                    << "/" << source_count
                    << " node=" << source_work.source
                    << " queries=" << source_work.queries.size()
                    << " copied_states=" << sparse_states.size()
                    << " iterations_used=" << result.iterations_used
                    << " converged=" << (result.converged ? "yes" : "no");
          if (result.target_early_stopped) {
            std::cout << " target_early_stop=yes";
          }
          if (options.profile_frontier_degrees) {
            std::cout << " frontier_active_vertices="
                      << source_frontier_degree_profile.active_vertices
                      << " frontier_edges="
                      << source_frontier_degree_profile.total_degree
                      << " frontier_avg_degree=" << std::fixed
                      << std::setprecision(3)
                      << source_frontier_degree_profile.average_degree()
                      << " frontier_max_degree="
                      << source_frontier_degree_profile.max_degree
                      << " max_frontier_count="
                      << source_frontier_degree_profile.max_frontier_count;
          }
          std::cout << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
                    << total_seconds_so_far
                    << " source_paths_written="
                    << (paths_written - paths_before)
                    << " total_paths_written=" << paths_written << '\n'
                    << std::flush;
        }
      }
      check_hip(hipStreamSynchronize(stream), "sync before freeing device CSR graph");
    } catch (...) {
      const hipError_t destroy_status = hipStreamDestroy(stream);
      if (destroy_status != hipSuccess) {
        std::cerr << "[bf_outgoing4] warning: hipStreamDestroy during cleanup failed: "
                  << hipGetErrorString(destroy_status) << '\n';
      }
      throw;
    }
    check_hip(hipStreamSynchronize(stream), "final stream sync");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");

    const auto total_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(total_end - total_begin).count();
    std::cout << "[bf_outgoing4] wrote " << paths_written
              << " path records to " << output_path.string()
              << " reached=" << reached_count
              << " unreached=" << unreached_count
              << " total_elapsed_sec=" << std::fixed << std::setprecision(3)
              << total_seconds << '\n';
    if (options.profile_frontier_degrees) {
      print_frontier_degree_profile(total_frontier_degree_profile);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[bf_outgoing4] ERROR: " << ex.what() << '\n';
    return 2;
  }
}
