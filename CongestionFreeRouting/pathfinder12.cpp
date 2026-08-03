#include "pathfinder12.hpp"

#include "bellman_ford/bf12_worker_policy.hpp"
#include "interchange/import_policy.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace routing {
namespace {

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char kMetadataMagic[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t kLegacyCsrVersion = 1;
constexpr std::uint64_t kPairedCsrVersion = 2;
constexpr std::uint64_t kCurrentCsrVersion = 3;
constexpr std::uint64_t kLegacyMetadataVersion = 4;
constexpr std::uint64_t kFirstPairedMetadataVersion = 5;
constexpr std::uint64_t kCurrentMetadataVersion = 6;
constexpr std::uint64_t kOutgoingEdgeOrientation = 2;

struct PipDataDisk {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
};

struct SitePinNodeDisk {
  std::uint64_t node = 0;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

static_assert(sizeof(EdgeAttr) == 2 * sizeof(std::uint64_t),
              "EdgeAttr metadata layout changed");
static_assert(std::is_trivially_copyable<EdgeAttr>::value,
              "EdgeAttr must remain bulk-readable");
static_assert(sizeof(PipDataDisk) == 3 * sizeof(std::uint64_t),
              "PipData disk layout changed");
static_assert(sizeof(SitePinNodeDisk) == 3 * sizeof(std::uint64_t),
              "site-pin disk layout changed");

bool valid_node(int node, minplus_sparse::Offset rows) {
  return node >= 0 && static_cast<minplus_sparse::Offset>(node) < rows;
}

std::size_t checked_add_size(std::size_t left,
                             std::size_t right,
                             const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(std::string(name) + " overflows size_t");
  }
  return left + right;
}

std::size_t checked_multiply_size(std::size_t left,
                                  std::size_t right,
                                  const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string(name) + " overflows size_t");
  }
  return left * right;
}

template <typename T>
std::size_t checked_vector_count(std::uint64_t count, const char* name) {
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(name) + " count exceeds size_t");
  }
  const std::size_t host_count = static_cast<std::size_t>(count);
  (void)checked_multiply_size(host_count, sizeof(T), name);
  return host_count;
}

void validate_graph_shape(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::invalid_argument("BF12 PathFinder requires a nonempty square CSR graph");
  }
  if (graph.nnz < 0 ||
      graph.rowptr.size() != static_cast<std::size_t>(graph.rows) + 1 ||
      graph.colind.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.values.size() != static_cast<std::size_t>(graph.nnz) ||
      graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::invalid_argument("BF12 PathFinder CSR array sizes are inconsistent");
  }
}

void validate_graph(const HostCsrF32& graph) {
  validate_graph_shape(graph);
  for (minplus_sparse::Offset row = 0; row < graph.rows; ++row) {
    const auto begin = graph.rowptr[static_cast<std::size_t>(row)];
    const auto end = graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::invalid_argument("BF12 PathFinder CSR row offsets are invalid");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (!valid_node(graph.colind[edge], graph.rows) ||
        !std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::invalid_argument(
          "BF12 PathFinder CSR destinations and weights must be valid and nonnegative");
    }
  }
}

void validate_sidecars_for_bf12(
    const interchange::RoutingCsrSidecars& sidecars,
    const HostCsrF32& graph) {
  interchange::validate_routing_csr_sidecars(
      sidecars, static_cast<std::size_t>(graph.rows),
      static_cast<std::size_t>(graph.nnz), false);
}

int parse_int(const char* text, const char* name) {
  if (text == nullptr) {
    throw std::runtime_error(std::string(name) + " requires a value");
  }
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

std::size_t parse_size(const char* text, const char* name) {
  if (text == nullptr || text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " +
                             (text == nullptr ? "<missing>" : text));
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      value > static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

std::filesystem::path default_metadata_path(
    const std::filesystem::path& csr_path) {
  std::filesystem::path result = csr_path;
  result += ".ifmeta.bin";
  return result;
}

std::int32_t saturating_margin(std::int32_t value,
                               int margin,
                               bool subtract) {
  const std::int64_t widened = subtract
                                   ? static_cast<std::int64_t>(value) - margin
                                   : static_cast<std::int64_t>(value) + margin;
  return static_cast<std::int32_t>(std::max<std::int64_t>(
      std::numeric_limits<std::int32_t>::min(),
      std::min<std::int64_t>(std::numeric_limits<std::int32_t>::max(),
                             widened)));
}

bool make_query_bounds(const interchange::RoutingCsrSidecars& sidecars,
                       const std::vector<int>& sources,
                       const std::vector<int>& targets,
                       const Pathfinder12Options& options,
                       BellmanFord12BoundingBox* output) {
  if (output == nullptr) {
    throw std::invalid_argument("BF12 bounding-box output is null");
  }
  *output = {};
  if (!options.bf12_enable_bounding_boxes) return false;

  std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_x = std::numeric_limits<std::int32_t>::min();
  std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_y = std::numeric_limits<std::int32_t>::min();
  const auto include = [&](int node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const std::int32_t x = sidecars.route_end_x[index];
    const std::int32_t y = sidecars.route_end_y[index];
    if (!interchange::has_route_coordinate(x, y)) return false;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    return true;
  };
  for (const int source : sources) {
    if (!include(source)) return false;
  }
  for (const int target : targets) {
    if (!include(target)) return false;
  }
  output->enabled = true;
  output->min_x = saturating_margin(min_x, options.bf12_bbox_margin_x, true);
  output->max_x = saturating_margin(max_x, options.bf12_bbox_margin_x, false);
  output->min_y = saturating_margin(min_y, options.bf12_bbox_margin_y, true);
  output->max_y = saturating_margin(max_y, options.bf12_bbox_margin_y, false);
  return true;
}

std::vector<int> unique_valid_sources(const RouteRequest& request,
                                      minplus_sparse::Offset rows) {
  std::vector<int> result;
  std::unordered_set<int> seen;
  seen.reserve(request.sources.size());
  for (const SitePinNode& source : request.sources) {
    if (valid_node(source.node, rows) && seen.insert(source.node).second) {
      result.push_back(source.node);
    }
  }
  return result;
}

bool approximately_equal(float left, float right) {
  if (!std::isfinite(left) || !std::isfinite(right)) return left == right;
  const float scale = std::max({1.0f, std::fabs(left), std::fabs(right)});
  return std::fabs(left - right) <= 1e-3f * scale;
}

float path_cost(const std::vector<PathEdge>& edges) {
  float cost = 0.0f;
  for (const PathEdge& edge : edges) cost += edge.cost;
  return cost;
}

void clear_unreached_sink(RoutedSink* sink) {
  const int target = sink->target;
  *sink = {};
  sink->target = target;
  sink->distance = std::numeric_limits<float>::infinity();
}

RoutedSink extract_compact_sink(const HostCsrF32& graph,
                                const RouteRequest& request,
                                const BellmanFordCsrResult& result,
                                std::size_t target_position,
                                int expected_target) {
  const std::size_t target_count = result.target_distances.size();
  if (result.target_sources.size() != target_count ||
      result.target_path_offsets.size() != target_count + 1 ||
      result.target_edge_offsets.size() != target_count + 1 ||
      target_position >= target_count) {
    throw std::runtime_error("BF12 returned inconsistent compact target arrays");
  }
  if (!result.target_path_edge_costs.empty() &&
      result.target_path_edge_costs.size() != result.target_path_edges.size()) {
    throw std::runtime_error("BF12 returned inconsistent compact edge costs");
  }

  RoutedSink sink;
  sink.target = expected_target;
  sink.distance = std::numeric_limits<float>::infinity();
  const float reported_distance = result.target_distances[target_position];
  if (!std::isfinite(reported_distance)) return sink;
  if (reported_distance < 0.0f) {
    throw std::runtime_error("BF12 returned a negative target distance");
  }

  const int source = result.target_sources[target_position];
  bool source_member = false;
  for (const SitePinNode& candidate : request.sources) {
    source_member = source_member || candidate.node == source;
  }
  if (!valid_node(source, graph.rows) || !source_member) {
    throw std::runtime_error("BF12 compact path root is not a request source");
  }

  const int node_begin = result.target_path_offsets[target_position];
  const int node_end = result.target_path_offsets[target_position + 1];
  const int edge_begin = result.target_edge_offsets[target_position];
  const int edge_end = result.target_edge_offsets[target_position + 1];
  if (node_begin < 0 || node_end <= node_begin || edge_begin < 0 ||
      edge_end < edge_begin || edge_end - edge_begin + 1 != node_end - node_begin ||
      static_cast<std::size_t>(node_end) > result.target_path_nodes.size() ||
      static_cast<std::size_t>(edge_end) > result.target_path_edges.size() ||
      result.target_path_nodes[static_cast<std::size_t>(node_begin)] != source ||
      result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] !=
          expected_target) {
    throw std::runtime_error("BF12 returned a malformed compact target path");
  }

  sink.source = source;
  sink.nodes.assign(result.target_path_nodes.begin() + node_begin,
                    result.target_path_nodes.begin() + node_end);
  std::unordered_set<int> path_nodes;
  path_nodes.reserve(sink.nodes.size());
  for (const int node : sink.nodes) {
    if (!valid_node(node, graph.rows) || !path_nodes.insert(node).second) {
      throw std::runtime_error("BF12 compact path leaves the graph or contains a cycle");
    }
  }

  sink.edges.reserve(static_cast<std::size_t>(edge_end - edge_begin));
  for (int item = edge_begin; item < edge_end; ++item) {
    const std::size_t local = static_cast<std::size_t>(item - edge_begin);
    const int from = sink.nodes[local];
    const int to = sink.nodes[local + 1];
    const minplus_sparse::Offset csr_edge =
        result.target_path_edges[static_cast<std::size_t>(item)];
    if (csr_edge < graph.rowptr[static_cast<std::size_t>(from)] ||
        csr_edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
        graph.colind[static_cast<std::size_t>(csr_edge)] != to) {
      throw std::runtime_error("BF12 compact path contains an invalid predecessor edge");
    }
    const float cost = result.target_path_edge_costs.empty()
                           ? graph.values[static_cast<std::size_t>(csr_edge)]
                           : result.target_path_edge_costs[
                                 static_cast<std::size_t>(item)];
    if (!std::isfinite(cost) || cost < 0.0f) {
      throw std::runtime_error("BF12 compact path contains an invalid edge cost");
    }
    sink.edges.push_back({from, to, csr_edge, cost});
  }
  sink.distance = path_cost(sink.edges);
  if (!approximately_equal(sink.distance, reported_distance)) {
    throw std::runtime_error("BF12 compact path cost does not match its distance");
  }
  return sink;
}

using RouteTreeStamp = std::uint32_t;

#ifdef ROUTING_PATHFINDER12_TEST_STAMP_LIMIT
static_assert(ROUTING_PATHFINDER12_TEST_STAMP_LIMIT > 0,
              "the BF12 test stamp limit must be positive");
static_assert(
    static_cast<std::uint64_t>(ROUTING_PATHFINDER12_TEST_STAMP_LIMIT) <=
        std::numeric_limits<RouteTreeStamp>::max(),
    "the BF12 test stamp limit must fit the production stamp type");
constexpr RouteTreeStamp kRouteTreeStampLimit =
    static_cast<RouteTreeStamp>(ROUTING_PATHFINDER12_TEST_STAMP_LIMIT);
#else
constexpr RouteTreeStamp kRouteTreeStampLimit =
    std::numeric_limits<RouteTreeStamp>::max();
#endif

// One instance is retained for the complete Pathfinder12 run. Membership and
// parent state are valid only when their stamp matches the current query, so a
// new query normally costs one integer increment instead of two graph-sized
// clears. A rare stamp wrap clears only the membership stamps; stale parent
// values remain safely hidden behind tree_seen.
struct RouteTreeScratch {
  explicit RouteTreeScratch(std::size_t node_count)
      : tree_seen(node_count, 0),
        parent_by_child(node_count, -1) {}

  RouteTreeStamp begin_query() {
    if (current_stamp >= kRouteTreeStampLimit) {
      std::fill(tree_seen.begin(), tree_seen.end(), RouteTreeStamp{0});
      current_stamp = 0;
    }
    return ++current_stamp;
  }

  bool contains(int node, RouteTreeStamp stamp) const {
    return node >= 0 && static_cast<std::size_t>(node) < tree_seen.size() &&
           tree_seen[static_cast<std::size_t>(node)] == stamp;
  }

  std::vector<RouteTreeStamp> tree_seen;
  std::vector<int> parent_by_child;
  RouteTreeStamp current_stamp = 0;
};

void trim_sink_to_tree(RoutedSink* sink,
                       const RouteTreeScratch& scratch,
                       RouteTreeStamp tree_stamp) {
  if (sink->nodes.empty() || sink->edges.size() + 1 != sink->nodes.size()) {
    throw std::runtime_error("BF12 route candidate has an invalid shape");
  }
  std::size_t last_tree = sink->nodes.size();
  for (std::size_t item = 0; item < sink->nodes.size(); ++item) {
    const int node = sink->nodes[item];
    if (scratch.contains(node, tree_stamp)) {
      last_tree = item;
    }
  }
  if (last_tree == sink->nodes.size()) {
    throw std::runtime_error("BF12 route candidate is detached from its source tree");
  }
  if (last_tree != 0) {
    sink->nodes.erase(sink->nodes.begin(), sink->nodes.begin() + last_tree);
    sink->edges.erase(sink->edges.begin(), sink->edges.begin() + last_tree);
  }
  sink->source = sink->nodes.front();
  sink->distance = path_cost(sink->edges);
}

bool attach_single_parent(const RoutedSink& sink,
                          RouteTreeScratch* scratch,
                          RouteTreeStamp tree_stamp,
                          std::vector<int>* unique_nodes) {
  for (const PathEdge& edge : sink.edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    if (child >= scratch->parent_by_child.size()) return false;
    if (scratch->tree_seen[child] == tree_stamp &&
        scratch->parent_by_child[child] != edge.from) {
      return false;
    }
  }
  for (const PathEdge& edge : sink.edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    if (scratch->tree_seen[child] != tree_stamp) {
      scratch->tree_seen[child] = tree_stamp;
      scratch->parent_by_child[child] = edge.from;
      unique_nodes->push_back(edge.to);
    }
  }
  return true;
}

void add_tree_node(int node,
                   RouteTreeScratch* scratch,
                   RouteTreeStamp tree_stamp,
                   std::vector<int>* unique_nodes) {
  if (node < 0 || static_cast<std::size_t>(node) >= scratch->tree_seen.size()) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(node);
  if (scratch->tree_seen[index] != tree_stamp) {
    scratch->tree_seen[index] = tree_stamp;
    scratch->parent_by_child[index] = -1;
    unique_nodes->push_back(node);
  }
}

RoutedNet trivial_net_result(const HostCsrF32& graph,
                             const RouteRequest& request) {
  RoutedNet net;
  net.net_string = request.net_string;
  net.sinks.resize(request.sinks.size());
  net.unique_nodes = unique_valid_sources(request, graph.rows);
  bool reached_all = !net.unique_nodes.empty();
  for (std::size_t index = 0; index < request.sinks.size(); ++index) {
    RoutedSink& sink = net.sinks[index];
    sink.target = request.sinks[index].node;
    sink.distance = std::numeric_limits<float>::infinity();
    if (valid_node(sink.target, graph.rows) &&
        std::find(net.unique_nodes.begin(), net.unique_nodes.end(),
                  sink.target) != net.unique_nodes.end()) {
      sink.source = sink.target;
      sink.distance = 0.0f;
      sink.reached = true;
      sink.nodes.push_back(sink.target);
    } else {
      reached_all = false;
    }
  }
  net.reached_all_sinks = reached_all;
  return net;
}

std::size_t route_request_count(const RoutingMetadata& metadata,
                                std::size_t limit) {
  return limit == 0 ? metadata.route_requests.size()
                    : std::min(limit, metadata.route_requests.size());
}

struct EndpointWindowMaximum {
  std::size_t sources = 0;
  std::size_t targets = 0;
};

EndpointWindowMaximum maximum_endpoint_window(
    const RoutingMetadata& metadata,
    std::size_t net_count,
    std::size_t maximum_window_size) {
  if (net_count > metadata.route_requests.size()) {
    throw std::invalid_argument(
        "BF12 endpoint preflight exceeds the route-request count");
  }
  if (maximum_window_size == 0 || net_count == 0) return {};

  const std::size_t window_size = std::min(net_count, maximum_window_size);
  std::size_t window_sources = 0;
  std::size_t window_targets = 0;
  EndpointWindowMaximum maximum;
  for (std::size_t net = 0; net < net_count; ++net) {
    if (net >= window_size) {
      const RouteRequest& outgoing =
          metadata.route_requests[net - window_size];
      if (outgoing.sources.size() > window_sources ||
          outgoing.sinks.size() > window_targets) {
        throw std::logic_error(
            "BF12 endpoint preflight sliding-window invariant failed");
      }
      window_sources -= outgoing.sources.size();
      window_targets -= outgoing.sinks.size();
    }

    const RouteRequest& incoming = metadata.route_requests[net];
    window_sources = checked_add_size(
        window_sources, incoming.sources.size(),
        "BF12 policy source window");
    window_targets = checked_add_size(
        window_targets, incoming.sinks.size(),
        "BF12 policy target window");
    maximum.sources = std::max(maximum.sources, window_sources);
    maximum.targets = std::max(maximum.targets, window_targets);
  }
  return maximum;
}

std::size_t select_safe_batch_size(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata,
    std::size_t net_count,
    const Pathfinder12Options& options,
    const BellmanFord12CooperativeCapability& capability) {
  const std::size_t automatic_preference =
      capability.architecture.find("gfx1151") != std::string::npos ? 4 : 3;
  const std::size_t candidate =
      std::min(net_count, options.bf12_batch_size == 0
                              ? automatic_preference
                              : options.bf12_batch_size);
  if (candidate == 0) return 0;

  // The memory policy may reduce candidate to any K <= candidate.  A K-wide
  // schedule can then cross the original candidate-aligned group boundaries,
  // so charge the largest rolling candidate window rather than only aligned
  // groups.  Endpoint counts are nonnegative: every contiguous window of size
  // at most K is contained in a candidate-wide window, so this one O(N) bound
  // remains safe for every possible downselected schedule.  Invalid endpoints
  // are still counted here, making the estimate conservative.
  const EndpointWindowMaximum endpoint_maximum =
      maximum_endpoint_window(metadata, net_count, candidate);
  const std::size_t maximum_sources = endpoint_maximum.sources;
  const std::size_t maximum_targets = endpoint_maximum.targets;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const hipError_t memory_status = hipMemGetInfo(&free_bytes, &total_bytes);
  if (memory_status != hipSuccess) {
    throw std::runtime_error(
        std::string("hipMemGetInfo failed while scheduling BF12: ") +
        hipGetErrorString(memory_status));
  }
  (void)total_bytes;
  BellmanFord12WorkerPolicyInputs inputs;
  inputs.requested_batch_size = options.bf12_batch_size;
  inputs.vertex_count = static_cast<std::size_t>(graph.rows);
  inputs.edge_count = static_cast<std::size_t>(graph.nnz);
  inputs.queries_waiting = candidate;
  inputs.source_count = maximum_sources;
  inputs.target_count = maximum_targets;
  inputs.available_device_bytes = free_bytes;
  inputs.result_node_capacity =
      options.bf12_result_node_capacity != 0
          ? options.bf12_result_node_capacity
          : checked_multiply_size(maximum_targets, 64,
                                  "BF12 default result-node capacity");
  inputs.result_edge_capacity =
      options.bf12_result_edge_capacity != 0
          ? options.bf12_result_edge_capacity
          : checked_multiply_size(maximum_targets, 63,
                                  "BF12 default result-edge capacity");
  inputs.gpu_architecture = capability.architecture;
  inputs.compute_unit_count = capability.compute_unit_count;
  inputs.cooperative_launch_supported =
      capability.device_supports_cooperative_launch &&
      capability.legally_resident_blocks > 0;
  inputs.memory_reserve_bytes = options.bf12_memory_reserve_bytes != 0
                                    ? options.bf12_memory_reserve_bytes
                                    : free_bytes / 4;
  inputs.telemetry_enabled = options.bf12_enable_telemetry;
  inputs.graph_already_resident = true;
  const BellmanFord12WorkerDecision decision =
      bf12_worker_policy::decide(inputs);
  if (decision.selected_batch_size == 0) {
    throw std::runtime_error("BF12 cannot select a safe batch: " +
                             decision.limiting_reason);
  }
  return decision.selected_batch_size;
}

void update_result_summary(PathfinderResult* output,
                           int capacity,
                           Pathfinder12RoutingStatistics* statistics) {
  output->occupancy.assign(output->occupancy.size(), 0);
  output->routed = true;
  output->all_sinks_reached = true;
  output->iterations_used = output->nets.empty() ? 0 : 1;
  for (const RoutedNet& net : output->nets) {
    if (!net.reached_all_sinks) {
      output->routed = false;
      output->all_sinks_reached = false;
      continue;
    }
    if (statistics != nullptr) ++statistics->successful_net_count;
    for (const int node : net.unique_nodes) {
      if (valid_node(node,
                     static_cast<minplus_sparse::Offset>(output->occupancy.size()))) {
        ++output->occupancy[static_cast<std::size_t>(node)];
      }
    }
  }
  output->overused_nodes = 0;
  output->max_occupancy = 0;
  for (const int occupancy : output->occupancy) {
    output->max_occupancy = std::max(output->max_occupancy, occupancy);
    if (occupancy > capacity) ++output->overused_nodes;
  }
}

std::uint64_t edge_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
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
          constexpr char kHex[] = "0123456789abcdef";
          out << "\\u00" << kHex[(ch >> 4) & 0xf] << kHex[ch & 0xf];
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

std::string metadata_string(const RoutingMetadata& metadata,
                            std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    return "<bad-string-" + std::to_string(index) + ">";
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

}  // namespace

BellmanFord12ControllerMode parse_bf12_controller(const std::string& value) {
  if (value == "auto") return BellmanFord12ControllerMode::Auto;
  if (value == "host-batch") return BellmanFord12ControllerMode::HostBatch;
  if (value == "cooperative-batch") {
    return BellmanFord12ControllerMode::CooperativeBatch;
  }
  throw std::runtime_error(
      "invalid bf12-controller: " + value +
      " (expected auto, host-batch, or cooperative-batch)");
}

void validate_pathfinder12_options(const Pathfinder12Options& options) {
  switch (options.bf12_controller) {
    case BellmanFord12ControllerMode::Auto:
    case BellmanFord12ControllerMode::HostBatch:
    case BellmanFord12ControllerMode::CooperativeBatch:
      break;
    default:
      throw std::invalid_argument("invalid BF12 controller mode");
  }
  if (options.bf12_target_check_interval <= 0) {
    throw std::invalid_argument("BF12 target-check interval must be positive");
  }
  if (options.bf12_bbox_margin_x < 0 || options.bf12_bbox_margin_y < 0) {
    throw std::invalid_argument("BF12 bounding-box margins must be nonnegative");
  }
  if (options.max_sssp_iterations < -1) {
    throw std::invalid_argument("BF12 maximum SSSP iterations must be -1 or nonnegative");
  }
  if (options.capacity <= 0) {
    throw std::invalid_argument("PathFinder capacity must be positive");
  }
  if (options.parallel_net_workers > 1) {
    throw std::invalid_argument(
        "BF12 owns one GPU workspace/controller; --parallel-net-workers greater "
        "than 1 would permit unsafe simultaneous cooperative traversals");
  }
}

std::string pathfinder12_usage(const std::string& executable) {
  std::ostringstream out;
  out << "Usage: " << executable
      << " <graph.csrbin> [metadata.ifmeta.bin] [options]\n"
      << "  --sssp-engine bf12\n"
      << "  --bf12-batch-size N              0=auto, 1=baseline\n"
      << "  --bf12-controller MODE           auto|host-batch|cooperative-batch\n"
      << "  --bf12-target-check-interval N\n"
      << "  --bf12-unbounded\n"
      << "  --bf12-bbox-margin-x N\n"
      << "  --bf12-bbox-margin-y N\n"
      << "  --bf12-no-unbounded-fallback\n"
      << "  --bf12-result-arena-nodes N\n"
      << "  --bf12-result-arena-edges N\n"
      << "  --bf12-memory-reserve-mib N\n"
      << "  --bf12-telemetry\n"
      << "  --max-sssp-iters N\n"
      << "  --capacity N\n"
      << "  --net-limit N\n"
      << "  --routes-out PATH\n"
      << "  --allow-unrouted\n"
      << "  --route-batch-size N             accepted and ignored; use "
         "--bf12-batch-size\n";
  return out.str();
}

Pathfinder12CommandLine parse_pathfinder12_args(int argc, char** argv) {
  if (argc < 2 || argv == nullptr) {
    throw std::runtime_error("a CSR input path is required");
  }
  Pathfinder12CommandLine parsed;
  parsed.csr_path = argv[1];
  int arg = 2;
  if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
    parsed.metadata_path = argv[arg++];
  } else {
    parsed.metadata_path = default_metadata_path(parsed.csr_path);
  }
  bool engine_seen = false;
  while (arg < argc) {
    const std::string option = argv[arg++];
    const auto require_value = [&](const char* name) -> const char* {
      if (arg >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[arg++];
    };
    if (option == "--sssp-engine") {
      if (engine_seen) {
        throw std::runtime_error("--sssp-engine may be specified only once");
      }
      engine_seen = true;
      const std::string value = require_value("--sssp-engine");
      if (value != "bf12") {
        throw std::runtime_error(
            "pathfinder12 supports only --sssp-engine bf12");
      }
    } else if (option == "--bf12-batch-size") {
      parsed.options.bf12_batch_size =
          parse_size(require_value("--bf12-batch-size"), "bf12-batch-size");
    } else if (option == "--bf12-controller") {
      parsed.options.bf12_controller =
          parse_bf12_controller(require_value("--bf12-controller"));
    } else if (option == "--bf12-target-check-interval") {
      parsed.options.bf12_target_check_interval = parse_int(
          require_value("--bf12-target-check-interval"),
          "bf12-target-check-interval");
    } else if (option == "--bf12-unbounded") {
      parsed.options.bf12_enable_bounding_boxes = false;
    } else if (option == "--bf12-bbox-margin-x") {
      parsed.options.bf12_bbox_margin_x = parse_int(
          require_value("--bf12-bbox-margin-x"), "bf12-bbox-margin-x");
    } else if (option == "--bf12-bbox-margin-y") {
      parsed.options.bf12_bbox_margin_y = parse_int(
          require_value("--bf12-bbox-margin-y"), "bf12-bbox-margin-y");
    } else if (option == "--bf12-no-unbounded-fallback") {
      parsed.options.bf12_enable_unbounded_retry = false;
    } else if (option == "--bf12-result-arena-nodes") {
      parsed.options.bf12_result_node_capacity = parse_size(
          require_value("--bf12-result-arena-nodes"),
          "bf12-result-arena-nodes");
    } else if (option == "--bf12-result-arena-edges") {
      parsed.options.bf12_result_edge_capacity = parse_size(
          require_value("--bf12-result-arena-edges"),
          "bf12-result-arena-edges");
    } else if (option == "--bf12-memory-reserve-mib") {
      const std::size_t mib = parse_size(
          require_value("--bf12-memory-reserve-mib"),
          "bf12-memory-reserve-mib");
      parsed.options.bf12_memory_reserve_bytes = checked_multiply_size(
          mib, 1024ULL * 1024ULL, "bf12-memory-reserve-mib");
    } else if (option == "--bf12-telemetry") {
      parsed.options.bf12_enable_telemetry = true;
    } else if (option == "--max-sssp-iters") {
      parsed.options.max_sssp_iterations = parse_int(
          require_value("--max-sssp-iters"), "max-sssp-iters");
    } else if (option == "--capacity") {
      parsed.options.capacity =
          parse_int(require_value("--capacity"), "capacity");
    } else if (option == "--net-limit") {
      parsed.options.net_limit =
          parse_size(require_value("--net-limit"), "net-limit");
    } else if (option == "--parallel-net-workers") {
      parsed.options.parallel_net_workers = parse_size(
          require_value("--parallel-net-workers"), "parallel-net-workers");
    } else if (option == "--routes-out") {
      parsed.routes_out_path = require_value("--routes-out");
    } else if (option == "--allow-unrouted") {
      parsed.allow_unrouted = true;
    } else if (option == "--route-batch-size") {
      // Preserve the one-shot router's compatibility surface, but never let
      // this historical JSON/host-flow flag control independent GPU queries.
      (void)parse_size(require_value("--route-batch-size"),
                       "route-batch-size");
    } else if (option == "-h" || option == "--help") {
      parsed.show_help = true;
      break;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  validate_pathfinder12_options(parsed.options);
  return parsed;
}

namespace {

// These views can be constructed only inside this translation unit. Their
// callers establish the validation precondition once at the public boundary,
// then use the trusted helpers for every batch/query in that routing run.
struct ValidatedGraphContext {
  const HostCsrF32& graph;
};

struct ValidatedRoutingContext {
  ValidatedGraphContext graph;
  const interchange::RoutingCsrSidecars& sidecars;
};

Pathfinder12PreparedBatch prepare_bf12_batch_validated(
    const ValidatedRoutingContext& context,
    const RoutingMetadata& metadata,
    const std::vector<std::size_t>& net_indices,
    const Pathfinder12Options& options) {
  const HostCsrF32& graph = context.graph.graph;
  const interchange::RoutingCsrSidecars& sidecars = context.sidecars;
  Pathfinder12PreparedBatch batch;
  batch.queries.reserve(net_indices.size());
  batch.descriptors.reserve(net_indices.size());
  for (const std::size_t net_index : net_indices) {
    if (net_index >= metadata.route_requests.size()) {
      throw std::out_of_range("BF12 net index exceeds routing metadata");
    }
    const RouteRequest& request = metadata.route_requests[net_index];
    Pathfinder12PreparedQuery query;
    query.original_net_index = net_index;
    query.sources = unique_valid_sources(request, graph.rows);
    for (std::size_t sink = 0; sink < request.sinks.size(); ++sink) {
      if (valid_node(request.sinks[sink].node, graph.rows)) {
        query.targets.push_back(request.sinks[sink].node);
        query.target_sink_indices.push_back(sink);
      }
    }
    if (query.sources.empty() || query.targets.empty()) continue;
    if (batch.flattened_sources.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        query.sources.size() > std::numeric_limits<std::uint32_t>::max() ||
        batch.flattened_targets.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        query.targets.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("BF12 flattened endpoint range exceeds uint32");
    }
    query.descriptor.source_begin =
        static_cast<std::uint32_t>(batch.flattened_sources.size());
    query.descriptor.source_count =
        static_cast<std::uint32_t>(query.sources.size());
    query.descriptor.target_begin =
        static_cast<std::uint32_t>(batch.flattened_targets.size());
    query.descriptor.target_count =
        static_cast<std::uint32_t>(query.targets.size());
    query.descriptor.max_iterations = options.max_sssp_iterations;
    query.descriptor.target_check_interval =
        options.bf12_target_check_interval;
    if (net_index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("BF12 original net index exceeds uint32");
    }
    query.descriptor.original_query_index =
        static_cast<std::uint32_t>(net_index);
    const bool bounded = make_query_bounds(sidecars, query.sources,
                                           query.targets, options,
                                           &query.descriptor.bounds);
    if (options.bf12_enable_bounding_boxes && !bounded &&
        !options.bf12_enable_unbounded_retry) {
      throw std::invalid_argument(
          "BF12 cannot bound a request whose terminal lacks route coordinates; "
          "use --bf12-unbounded or allow unbounded fallback");
    }
    batch.flattened_sources.insert(batch.flattened_sources.end(),
                                   query.sources.begin(), query.sources.end());
    batch.flattened_targets.insert(batch.flattened_targets.end(),
                                   query.targets.begin(), query.targets.end());
    batch.descriptors.push_back(query.descriptor);
    batch.queries.push_back(std::move(query));
  }
  return batch;
}

RoutedNet finalize_bf12_routed_net_validated(
    const ValidatedGraphContext& context,
    const RouteRequest& request,
    const Pathfinder12PreparedQuery& prepared,
    const BellmanFordCsrResult& result,
    const BellmanFord12QueryStatus& status,
    RouteTreeScratch* scratch) {
  const HostCsrF32& graph = context.graph;
  if (prepared.targets.size() != prepared.target_sink_indices.size() ||
      prepared.descriptor.target_count != prepared.targets.size()) {
    throw std::invalid_argument("BF12 prepared target mapping is inconsistent");
  }
  if (result.target_distances.size() != prepared.targets.size() ||
      result.target_sources.size() != prepared.targets.size() ||
      result.target_path_offsets.size() != prepared.targets.size() + 1 ||
      result.target_edge_offsets.size() != prepared.targets.size() + 1) {
    throw std::runtime_error("BF12 result target count does not match its request");
  }
  if (scratch == nullptr ||
      scratch->tree_seen.size() != static_cast<std::size_t>(graph.rows) ||
      scratch->parent_by_child.size() !=
          static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument(
        "BF12 route-tree scratch does not match the validated graph");
  }
  const RouteTreeStamp tree_stamp = scratch->begin_query();

  RoutedNet net;
  net.net_string = request.net_string;
  net.sinks.resize(request.sinks.size());
  for (const int source : unique_valid_sources(request, graph.rows)) {
    add_tree_node(source, scratch, tree_stamp, &net.unique_nodes);
  }
  for (std::size_t sink = 0; sink < request.sinks.size(); ++sink) {
    net.sinks[sink].target = request.sinks[sink].node;
    net.sinks[sink].distance = std::numeric_limits<float>::infinity();
  }

  std::vector<RoutedSink> candidates(request.sinks.size());
  std::vector<bool> have_candidate(request.sinks.size(), false);
  if (status.paths_certified) {
    for (std::size_t target = 0; target < prepared.targets.size(); ++target) {
      const std::size_t sink_index = prepared.target_sink_indices[target];
      if (sink_index >= request.sinks.size() ||
          request.sinks[sink_index].node != prepared.targets[target]) {
        throw std::runtime_error("BF12 target-to-sink mapping changed before finalization");
      }
      RoutedSink candidate = extract_compact_sink(
          graph, request, result, target, prepared.targets[target]);
      if (std::isfinite(candidate.distance)) {
        candidates[sink_index] = std::move(candidate);
        have_candidate[sink_index] = true;
      }
    }
  }

  bool reached_all = !net.unique_nodes.empty();
  for (std::size_t sink_index = 0; sink_index < request.sinks.size();
       ++sink_index) {
    const int target = request.sinks[sink_index].node;
    RoutedSink& routed = net.sinks[sink_index];
    if (!valid_node(target, graph.rows)) {
      reached_all = false;
      continue;
    }
    if (scratch->contains(target, tree_stamp)) {
      routed.source = target;
      routed.distance = 0.0f;
      routed.reached = true;
      routed.nodes.push_back(target);
      continue;
    }
    if (!have_candidate[sink_index]) {
      reached_all = false;
      continue;
    }
    routed = std::move(candidates[sink_index]);
    trim_sink_to_tree(&routed, *scratch, tree_stamp);
    if (!attach_single_parent(routed, scratch, tree_stamp,
                              &net.unique_nodes)) {
      clear_unreached_sink(&routed);
      reached_all = false;
      continue;
    }
    routed.reached = true;
  }
  for (const RoutedSink& sink : net.sinks) {
    if (!sink.reached) reached_all = false;
  }
  net.reached_all_sinks = reached_all;
  return net;
}

}  // namespace

Pathfinder12PreparedBatch prepare_bf12_batch(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata,
    const interchange::RoutingCsrSidecars& sidecars,
    const std::vector<std::size_t>& net_indices,
    const Pathfinder12Options& options) {
  validate_pathfinder12_options(options);
  validate_graph(graph);
  validate_sidecars_for_bf12(sidecars, graph);
  return prepare_bf12_batch_validated(
      {{graph}, sidecars}, metadata, net_indices, options);
}

RoutedNet finalize_bf12_routed_net(
    const HostCsrF32& graph,
    const RouteRequest& request,
    const Pathfinder12PreparedQuery& prepared,
    const BellmanFordCsrResult& result,
    const BellmanFord12QueryStatus& status) {
  validate_graph(graph);
  // Preserve the checked compatibility API without returning to graph-sized
  // allocation/initialization on every direct call. Thread-local ownership
  // keeps independent finalizers race-free; a graph-size change starts a new
  // stamp domain and safely discards the old scratch.
  thread_local std::optional<RouteTreeScratch> thread_scratch;
  if (!thread_scratch ||
      thread_scratch->tree_seen.size() !=
          static_cast<std::size_t>(graph.rows)) {
    thread_scratch.emplace(static_cast<std::size_t>(graph.rows));
  }
  return finalize_bf12_routed_net_validated(
      {graph}, request, prepared, result, status, &*thread_scratch);
}

PathfinderResult route_all_nets_bf12(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata,
    const interchange::RoutingCsrSidecars* sidecars,
    const Pathfinder12Options& options,
    hipStream_t stream,
    Pathfinder12RoutingStatistics* statistics,
    Pathfinder12ProgressCallback progress) {
  validate_pathfinder12_options(options);
  // Establish only the constant-time shape invariants needed before backend
  // construction. A successful BellmanFord12CsrGraph construction below is
  // the single full graph/sidecar validation boundary for a nonempty run.
  validate_graph_shape(graph);
  if (options.bf12_controller ==
          BellmanFord12ControllerMode::CooperativeBatch &&
      stream != nullptr) {
    throw std::invalid_argument(
        "BF12 cooperative batching requires the null/default stream and one "
        "exclusive full-residency controller");
  }
  const std::size_t count = route_request_count(metadata, options.net_limit);
  Pathfinder12RoutingStatistics local_stats;
  local_stats.requested_batch_size = options.bf12_batch_size;
  local_stats.net_count = count;

  PathfinderResult output;
  output.occupancy.assign(static_cast<std::size_t>(graph.rows), 0);
  output.nets.resize(count);
  for (std::size_t net = 0; net < count; ++net) {
    output.nets[net] = trivial_net_result(graph, metadata.route_requests[net]);
  }
  if (count == 0) {
    // No backend graph is constructed for an empty request set, so retain the
    // public route API's full graph validation at this boundary.
    validate_graph(graph);
    output.routed = true;
    output.all_sinks_reached = true;
    if (statistics != nullptr) *statistics = local_stats;
    if (progress) progress(0, 0, 0);
    return output;
  }

  interchange::RoutingCsrSidecars legacy_sidecars;
  const interchange::RoutingCsrSidecars* effective_sidecars = sidecars;
  if (effective_sidecars == nullptr || effective_sidecars->route_end_x.empty()) {
    if (options.bf12_enable_bounding_boxes) {
      throw std::invalid_argument(
          "bounded BF12 requires CSR v3 route-end sidecars; use --bf12-unbounded "
          "for a legacy artifact");
    }
    legacy_sidecars.route_end_x.assign(
        static_cast<std::size_t>(graph.rows),
        interchange::kMissingRouteCoordinate);
    legacy_sidecars.route_end_y.assign(
        static_cast<std::size_t>(graph.rows),
        interchange::kMissingRouteCoordinate);
    legacy_sidecars.base_vertex_cost.assign(static_cast<std::size_t>(graph.rows),
                                             1.0f);
    effective_sidecars = &legacy_sidecars;
  }
  auto shared_graph = std::make_shared<BellmanFord12CsrGraph>(
      graph, *effective_sidecars, stream);
  const ValidatedGraphContext validated_graph{graph};
  const ValidatedRoutingContext validated_routing{
      validated_graph, *effective_sidecars};
  RouteTreeScratch route_tree_scratch(static_cast<std::size_t>(graph.rows));

  // One workspace owns the only controller.  The default stream requirement
  // is enforced again by the backend for explicit cooperative mode.
  BellmanFord12CsrWorkspace workspace(shared_graph, 0, stream);
  const BellmanFord12CooperativeCapability capability =
      workspace.cooperative_capability(
          options.bf12_controller != BellmanFord12ControllerMode::HostBatch);
  const std::size_t selected_batch_size = select_safe_batch_size(
      graph, metadata, count, options, capability);
  if (selected_batch_size > workspace.maximum_batch_size()) {
    throw std::invalid_argument(
        "requested BF12 batch size exceeds composite-index workspace capacity");
  }
  local_stats.selected_batch_size = selected_batch_size;

  // BF12 constructs every workspace with an all-one dynamic-cost epoch.
  // Pathfinder12 never changes that one-shot epoch, so all batches share the
  // initialized values without a redundant V-sized host vector, H2D copy, and
  // synchronization here.

  std::size_t next_net = 0;
  while (next_net < count) {
    const std::size_t end =
        std::min(count, checked_add_size(next_net, selected_batch_size,
                                         "BF12 batch end"));
    std::vector<std::size_t> indices;
    indices.reserve(end - next_net);
    for (std::size_t index = next_net; index < end; ++index) {
      indices.push_back(index);
    }
    Pathfinder12PreparedBatch batch = prepare_bf12_batch_validated(
        validated_routing, metadata, indices, options);
    if (!batch.descriptors.empty()) {
      BellmanFord12BatchOptions backend_options;
      backend_options.requested_batch_size = options.bf12_batch_size;
      backend_options.controller_mode = options.bf12_controller;
      backend_options.target_check_interval =
          options.bf12_target_check_interval;
      backend_options.enable_bounding_boxes =
          options.bf12_enable_bounding_boxes;
      backend_options.enable_unbounded_retry =
          options.bf12_enable_unbounded_retry;
      backend_options.enable_telemetry = options.bf12_enable_telemetry;
      backend_options.result_node_capacity =
          options.bf12_result_node_capacity;
      backend_options.result_edge_capacity =
          options.bf12_result_edge_capacity;
      backend_options.memory_safety_reserve_bytes =
          options.bf12_memory_reserve_bytes;
      BellmanFord12BatchResult result = workspace.run_batch(
          batch.flattened_sources, batch.flattened_targets,
          batch.descriptors, backend_options, stream);
      if (result.query_results.size() != batch.queries.size() ||
          result.query_statuses.size() != batch.queries.size() ||
          result.original_query_indices.size() != batch.queries.size()) {
        throw std::runtime_error("BF12 backend returned the wrong query count");
      }
      ++local_stats.batch_count;
      ++(batch.queries.size() == selected_batch_size
             ? local_stats.full_batch_count
             : local_stats.partial_batch_count);
      local_stats.submitted_query_count += batch.queries.size();
      for (std::size_t query = 0; query < batch.queries.size(); ++query) {
        const Pathfinder12PreparedQuery& prepared = batch.queries[query];
        if (result.original_query_indices[query] !=
                prepared.descriptor.original_query_index ||
            prepared.original_net_index >= output.nets.size()) {
          throw std::runtime_error("BF12 backend did not preserve query identity");
        }
        if (prepared.descriptor.bounds.enabled) {
          ++local_stats.bounded_descriptor_count;
        } else {
          ++local_stats.unbounded_descriptor_count;
        }
        output.nets[prepared.original_net_index] =
            finalize_bf12_routed_net_validated(
                validated_graph,
                metadata.route_requests[prepared.original_net_index],
                prepared, result.query_results[query],
                result.query_statuses[query], &route_tree_scratch);
      }
    }
    next_net = end;
    if (progress) progress(next_net, count, local_stats.batch_count);
  }

  if (options.bf12_enable_telemetry) {
    local_stats.backend_telemetry = workspace.telemetry();
    local_stats.backend_telemetry.selected_batch_size = selected_batch_size;
    const double available_query_slots =
        static_cast<double>(local_stats.batch_count) *
        static_cast<double>(selected_batch_size);
    local_stats.backend_telemetry.batch_fill_ratio =
        available_query_slots == 0.0
            ? 0.0
            : static_cast<double>(local_stats.submitted_query_count) /
                  available_query_slots;
  }
  update_result_summary(&output, options.capacity, &local_stats);
  if (statistics != nullptr) *statistics = local_stats;
  return output;
}

namespace {

std::uint64_t read_u64(std::ifstream& input, const char* name) {
  std::uint64_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error(std::string("failed while reading ") + name);
  return value;
}

std::int64_t read_i64(std::ifstream& input, const char* name) {
  std::int64_t value = 0;
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error(std::string("failed while reading ") + name);
  return value;
}

template <typename T>
void read_array(std::ifstream& input,
                std::vector<T>* values,
                std::uint64_t count,
                const char* name) {
  const std::size_t size = checked_vector_count<T>(count, name);
  const std::size_t bytes = checked_multiply_size(size, sizeof(T), name);
  if (bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(std::string(name) + " exceeds stream range");
  }
  values->resize(size);
  if (bytes != 0) {
    input.read(reinterpret_cast<char*>(values->data()),
               static_cast<std::streamsize>(bytes));
    if (!input) {
      throw std::runtime_error(std::string("failed while reading ") + name);
    }
  }
}

void skip_bytes(std::ifstream& input, std::size_t bytes, const char* name) {
  if (bytes == 0) return;
  if (bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::overflow_error(std::string(name) + " exceeds stream offset range");
  }
  input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
  if (!input) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

template <typename T>
void skip_array(std::ifstream& input,
                std::uint64_t count,
                const char* name) {
  const std::size_t size = checked_vector_count<T>(count, name);
  skip_bytes(input, checked_multiply_size(size, sizeof(T), name), name);
}

void require_position_in_file(std::ifstream& input, const char* name) {
  const auto position = input.tellg();
  if (position == std::ifstream::pos_type(-1)) {
    throw std::runtime_error(std::string("failed while checking ") + name);
  }
  input.seekg(0, std::ios::end);
  if (!input) {
    throw std::runtime_error(std::string("failed while checking ") + name);
  }
  const auto end = input.tellg();
  if (end == std::ifstream::pos_type(-1) || position > end) {
    throw std::runtime_error(std::string(name) + " is truncated");
  }
}

std::string read_string(std::ifstream& input) {
  const std::uint64_t count = read_u64(input, "metadata string length");
  const std::size_t size = checked_vector_count<char>(count, "metadata string");
  if (size > static_cast<std::size_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error("metadata string exceeds stream range");
  }
  std::string result(size, '\0');
  if (!result.empty()) {
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!input) throw std::runtime_error("failed while reading metadata string");
  }
  return result;
}

int route_node_from_disk(std::uint64_t raw, const char* name) {
  if (raw == kNoIndex) return -1;
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

int read_route_node(std::ifstream& input, const char* name) {
  return route_node_from_disk(read_u64(input, name), name);
}

}  // namespace

HostCsrF32 load_pathfinder12_csrbin(
    const std::filesystem::path& path,
    std::optional<interchange::InterchangeArtifactPairId>* artifact_pair_id,
    interchange::RoutingCsrSidecars* routing_sidecars) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }
  char magic[sizeof(kCsrMagic)] = {};
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, kCsrMagic, sizeof(kCsrMagic)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }
  const std::uint64_t version = read_u64(input, "CSR format version");
  const std::uint64_t orientation = read_u64(input, "CSR orientation");
  if (version < kLegacyCsrVersion || version > kCurrentCsrVersion) {
    throw std::runtime_error("unsupported CSR format version");
  }
  if (orientation != kOutgoingEdgeOrientation) {
    throw std::runtime_error("unsupported CSR orientation");
  }
  std::optional<interchange::InterchangeArtifactPairId> parsed_pair;
  if (version >= kPairedCsrVersion) {
    interchange::InterchangeArtifactPairId pair;
    pair.high = read_u64(input, "CSR artifact pair id high");
    pair.low = read_u64(input, "CSR artifact pair id low");
    if (pair.is_zero()) throw std::runtime_error("CSR artifact pair id is zero");
    parsed_pair = pair;
  }
  const std::uint64_t rows = read_u64(input, "CSR row count");
  const std::uint64_t cols = read_u64(input, "CSR column count");
  (void)read_u64(input, "CSR declared edge count");
  (void)read_u64(input, "CSR loaded edge count");
  const std::uint64_t nnz = read_u64(input, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(input, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(input, "CSR colind count");
  const std::uint64_t values_count = read_u64(input, "CSR values count");

  std::uint64_t route_x_count = 0;
  std::uint64_t route_y_count = 0;
  std::uint64_t base_cost_count = 0;
  std::int64_t spatial_min_x = 0;
  std::int64_t spatial_min_y = 0;
  std::uint64_t spatial_width = 0;
  std::uint64_t spatial_height = 0;
  std::uint64_t spatial_offset_count = 0;
  std::uint64_t spatial_edge_count = 0;
  if (version >= kCurrentCsrVersion) {
    route_x_count = read_u64(input, "CSR route-end x count");
    route_y_count = read_u64(input, "CSR route-end y count");
    base_cost_count = read_u64(input, "CSR base vertex cost count");
    spatial_min_x = read_i64(input, "CSR spatial minimum x");
    spatial_min_y = read_i64(input, "CSR spatial minimum y");
    spatial_width = read_u64(input, "CSR spatial width");
    spatial_height = read_u64(input, "CSR spatial height");
    spatial_offset_count = read_u64(input, "CSR spatial offset count");
    spatial_edge_count = read_u64(input, "CSR spatial edge-id count");
  }
  if (rows == 0 || rows != cols ||
      rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Index>::max()) ||
      nnz > static_cast<std::uint64_t>(
                std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rowptr_count != rows + 1 || colind_count != nnz || values_count != nnz) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }
  if (version >= kCurrentCsrVersion) {
    if (route_x_count != rows || route_y_count != rows ||
        base_cost_count != rows || spatial_edge_count != nnz ||
        spatial_min_x < std::numeric_limits<std::int32_t>::min() ||
        spatial_min_x > std::numeric_limits<std::int32_t>::max() ||
        spatial_min_y < std::numeric_limits<std::int32_t>::min() ||
        spatial_min_y > std::numeric_limits<std::int32_t>::max() ||
        (spatial_width == 0) != (spatial_height == 0) ||
        (spatial_width != 0 &&
         spatial_height > std::numeric_limits<std::uint64_t>::max() /
                              spatial_width)) {
      throw std::runtime_error("CSR routing sidecar header is inconsistent");
    }
    const std::uint64_t regular = spatial_width * spatial_height;
    if (regular > interchange::maximum_dense_spatial_cells(
                      static_cast<std::size_t>(rows)) ||
        regular > std::numeric_limits<std::uint64_t>::max() - 2 ||
        spatial_offset_count != regular + 2) {
      throw std::runtime_error("CSR spatial shard count is inconsistent");
    }
  }

  HostCsrF32 graph;
  graph.rows = static_cast<minplus_sparse::Offset>(rows);
  graph.cols = static_cast<minplus_sparse::Offset>(cols);
  graph.nnz = static_cast<minplus_sparse::Offset>(nnz);
  read_array(input, &graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(input, &graph.colind, colind_count, "CSR colind");
  read_array(input, &graph.values, values_count, "CSR values");

  interchange::RoutingCsrSidecars parsed_sidecars;
  if (version >= kCurrentCsrVersion) {
    if (routing_sidecars != nullptr) {
      read_array(input, &parsed_sidecars.route_end_x, route_x_count,
                 "CSR route-end x coordinates");
      read_array(input, &parsed_sidecars.route_end_y, route_y_count,
                 "CSR route-end y coordinates");
      read_array(input, &parsed_sidecars.base_vertex_cost, base_cost_count,
                 "CSR base vertex costs");
      // BF12 does not consume spatial edge shards.  Validate the compact node
      // columns while skipping the graph-sized shard arrays.
      skip_array<std::uint64_t>(input, spatial_offset_count,
                                "CSR spatial shard offsets");
      skip_array<std::uint32_t>(input, spatial_edge_count,
                                "CSR spatial shard edge IDs");
      interchange::validate_routing_csr_sidecars(
          parsed_sidecars, static_cast<std::size_t>(rows),
          static_cast<std::size_t>(nnz), false);
    } else {
      skip_array<std::int32_t>(input, route_x_count, "CSR route-end x");
      skip_array<std::int32_t>(input, route_y_count, "CSR route-end y");
      skip_array<float>(input, base_cost_count, "CSR base vertex costs");
      skip_array<std::uint64_t>(input, spatial_offset_count,
                                "CSR spatial shard offsets");
      skip_array<std::uint32_t>(input, spatial_edge_count,
                                "CSR spatial shard edge IDs");
    }
  }
  require_position_in_file(input, "CSR payload");
  validate_graph(graph);
  if (artifact_pair_id != nullptr) *artifact_pair_id = parsed_pair;
  if (routing_sidecars != nullptr) *routing_sidecars = std::move(parsed_sidecars);
  return graph;
}

RoutingMetadata load_pathfinder12_metadata(
    const std::filesystem::path& path,
    InterchangeMetadataLoadMode mode) {
  switch (mode) {
    case InterchangeMetadataLoadMode::kFull:
    case InterchangeMetadataLoadMode::kRoutingOnly:
    case InterchangeMetadataLoadMode::kRoutingWithRouteOutput:
      break;
    default:
      throw std::invalid_argument("unknown interchange metadata load mode");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }
  char magic[sizeof(kMetadataMagic)] = {};
  input.read(magic, sizeof(magic));
  if (!input ||
      std::memcmp(magic, kMetadataMagic, sizeof(kMetadataMagic)) != 0) {
    throw std::runtime_error(
        "input is not a recognized RIPS interchange metadata file");
  }
  const std::uint64_t version = read_u64(input, "metadata version");
  const std::uint64_t orientation = read_u64(input, "metadata orientation");
  if (version < kLegacyMetadataVersion || version > kCurrentMetadataVersion ||
      orientation != kOutgoingEdgeOrientation) {
    throw std::runtime_error("unsupported metadata version or orientation");
  }
  std::optional<interchange::InterchangeArtifactPairId> parsed_pair;
  if (version >= kFirstPairedMetadataVersion) {
    interchange::InterchangeArtifactPairId pair;
    pair.high = read_u64(input, "metadata pair id high");
    pair.low = read_u64(input, "metadata pair id low");
    if (pair.is_zero()) throw std::runtime_error("metadata pair id is zero");
    parsed_pair = pair;
  }
  const std::uint64_t string_count = read_u64(input, "metadata string count");
  const std::uint64_t node_count = read_u64(input, "metadata node count");
  const std::uint64_t edge_count = read_u64(input, "metadata edge count");
  const std::uint64_t pip_count = read_u64(input, "metadata PIP count");
  const std::uint64_t site_pin_count = read_u64(input, "metadata site-pin count");
  const std::uint64_t request_count = read_u64(input, "metadata request count");
  const std::uint64_t blocked_count = read_u64(input, "metadata blocked count");
  const std::uint64_t sink_stop_count = read_u64(input, "metadata sink-stop count");
  const std::uint64_t logical_cell_count = read_u64(input, "logical cell count");
  const std::uint64_t logical_net_count = read_u64(input, "logical net count");
  const std::uint64_t logical_port_count = read_u64(input, "logical port count");
  const std::uint64_t physical_bytes = read_u64(input, "physical byte count");
  const std::uint64_t logical_bytes = read_u64(input, "logical byte count");

  RoutingMetadata metadata;
  metadata.artifact_pair_id = parsed_pair;
  metadata.declared_node_count = node_count;
  metadata.declared_edge_attr_count = edge_count;
  metadata.device_path_string = read_u64(input, "device path string");
  metadata.physical_path_string = read_u64(input, "physical path string");
  metadata.logical_path_string = read_u64(input, "logical path string");
  metadata.logical_design_name_string = read_u64(input, "logical design string");
  metadata.strings.reserve(
      checked_vector_count<std::string>(string_count, "metadata strings"));
  for (std::uint64_t item = 0; item < string_count; ++item) {
    metadata.strings.push_back(read_string(input));
  }

  const bool has_node_arrays = version < kCurrentMetadataVersion;
  const bool load_node_arrays =
      has_node_arrays && mode == InterchangeMetadataLoadMode::kFull;
  if (has_node_arrays) {
    if (load_node_arrays) {
      read_array(input, &metadata.node_device_ids, node_count, "node ids");
      read_array(input, &metadata.node_min_x, node_count, "node min x");
      read_array(input, &metadata.node_max_x, node_count, "node max x");
      read_array(input, &metadata.node_min_y, node_count, "node min y");
      read_array(input, &metadata.node_max_y, node_count, "node max y");
      read_array(input, &metadata.node_tile_type_strings, node_count,
                 "node tile types");
      read_array(input, &metadata.node_wire_type_strings, node_count,
                 "node wire types");
    } else {
      skip_array<std::uint64_t>(input, node_count, "node ids");
      skip_array<std::int32_t>(input, node_count, "node min x");
      skip_array<std::int32_t>(input, node_count, "node max x");
      skip_array<std::int32_t>(input, node_count, "node min y");
      skip_array<std::int32_t>(input, node_count, "node max y");
      skip_array<std::uint64_t>(input, node_count, "node tile types");
      skip_array<std::uint64_t>(input, node_count, "node wire types");
    }
  }

  const bool load_output = mode != InterchangeMetadataLoadMode::kRoutingOnly;
  if (load_output) {
    read_array(input, &metadata.edge_attrs, edge_count, "edge attributes");
    std::vector<PipDataDisk> disk_pips;
    read_array(input, &disk_pips, pip_count, "PIP data");
    metadata.pip_data.reserve(disk_pips.size());
    for (const PipDataDisk& pip : disk_pips) {
      metadata.pip_data.push_back(
          {pip.wire0_string, pip.wire1_string, pip.forward != 0});
    }
  } else {
    skip_array<EdgeAttr>(input, edge_count, "edge attributes");
    skip_array<PipDataDisk>(input, pip_count, "PIP data");
  }

  if (mode == InterchangeMetadataLoadMode::kFull) {
    std::vector<SitePinNodeDisk> disk_site_pins;
    read_array(input, &disk_site_pins, site_pin_count, "site-pin data");
    metadata.site_pin_attrs.reserve(disk_site_pins.size());
    for (const SitePinNodeDisk& pin : disk_site_pins) {
      metadata.site_pin_attrs.push_back(
          {route_node_from_disk(pin.node, "site-pin node"),
           pin.site_string, pin.pin_string});
    }
  } else {
    skip_array<SitePinNodeDisk>(input, site_pin_count, "site-pin data");
  }

  metadata.route_requests.resize(
      checked_vector_count<RouteRequest>(request_count, "route requests"));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(input, "route request net");
    request.logical_net_index = read_u64(input, "route logical net");
    const std::uint64_t sources = read_u64(input, "route source count");
    request.sources.resize(
        checked_vector_count<SitePinNode>(sources, "route sources"));
    for (SitePinNode& source : request.sources) {
      source.node = read_route_node(input, "route source node");
      source.site_string = read_u64(input, "route source site");
      source.pin_string = read_u64(input, "route source pin");
    }
    const std::uint64_t sinks = read_u64(input, "route sink count");
    request.sinks.resize(
        checked_vector_count<SitePinNode>(sinks, "route sinks"));
    for (SitePinNode& sink : request.sinks) {
      sink.node = read_route_node(input, "route sink node");
      sink.site_string = read_u64(input, "route sink site");
      sink.pin_string = read_u64(input, "route sink pin");
    }
  }
  skip_array<std::array<std::uint64_t, 3>>(input, logical_cell_count,
                                            "logical cells");
  skip_array<std::array<std::uint64_t, 4>>(input, logical_net_count,
                                            "logical nets");
  skip_array<std::array<std::uint64_t, 7>>(input, logical_port_count,
                                            "logical ports");
  if (mode == InterchangeMetadataLoadMode::kFull) {
    read_array(input, &metadata.blocked_nodes, blocked_count, "blocked nodes");
    read_array(input, &metadata.sink_stop_nodes, sink_stop_count,
               "sink-stop nodes");
  } else {
    skip_array<std::uint64_t>(input, blocked_count, "blocked nodes");
    skip_array<std::uint64_t>(input, sink_stop_count, "sink-stop nodes");
  }
  skip_array<std::uint8_t>(input, physical_bytes, "physical bytes");
  skip_array<std::uint8_t>(input, logical_bytes, "logical bytes");
  require_position_in_file(input, "metadata payload");
  return metadata;
}

void write_pathfinder12_routes_jsonl(const std::filesystem::path& path,
                                     const HostCsrF32& graph,
                                     const RoutingMetadata& metadata,
                                     const PathfinderResult& result) {
  validate_graph(graph);
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }
  if (result.nets.size() > metadata.route_requests.size()) {
    throw std::runtime_error("BF12 result has more nets than route requests");
  }
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open routes output: " + path.string());
  }

  for (std::size_t net_index = 0; net_index < result.nets.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const RoutedNet& net = result.nets[net_index];
    if (net.net_string != request.net_string) {
      throw std::runtime_error("BF12 result net order or identity is unstable");
    }
    output << '{';
    if (metadata.artifact_pair_id.has_value()) {
      output << "\"artifact_pair_id\":";
      write_json_string(
          output, interchange::interchange_artifact_pair_id_string(
                      *metadata.artifact_pair_id));
      output << ',';
    }
    output << "\"net\":";
    write_json_string(output, metadata_string(metadata, request.net_string));
    output << ",\"routed\":"
           << (net.reached_all_sinks ? "true" : "false");

    output << ",\"sources\":[";
    for (std::size_t index = 0; index < request.sources.size(); ++index) {
      if (index != 0) output << ',';
      const SitePinNode& source = request.sources[index];
      output << "{\"node\":" << source.node << ",\"site\":";
      write_json_string(output, metadata_string(metadata, source.site_string));
      output << ",\"pin\":";
      write_json_string(output, metadata_string(metadata, source.pin_string));
      output << '}';
    }
    output << ']';

    output << ",\"sinks\":[";
    for (std::size_t index = 0; index < request.sinks.size(); ++index) {
      if (index != 0) output << ',';
      const SitePinNode& pin = request.sinks[index];
      const RoutedSink* sink =
          index < net.sinks.size() ? &net.sinks[index] : nullptr;
      output << "{\"node\":" << pin.node << ",\"site\":";
      write_json_string(output, metadata_string(metadata, pin.site_string));
      output << ",\"pin\":";
      write_json_string(output, metadata_string(metadata, pin.pin_string));
      output << ",\"reached\":"
             << (sink != nullptr && sink->reached ? "true" : "false")
             << ",\"source\":" << (sink == nullptr ? -1 : sink->source)
             << '}';
    }
    output << ']';

    std::unordered_set<std::uint64_t> seen_edges;
    output << ",\"edges\":[";
    bool first_edge = true;
    for (const RoutedSink& sink : net.sinks) {
      if (!sink.reached) continue;
      for (const PathEdge& edge : sink.edges) {
        if (!valid_node(edge.from, graph.rows) ||
            !valid_node(edge.to, graph.rows) || edge.csr_edge < 0 ||
            edge.csr_edge >= graph.nnz ||
            edge.csr_edge < graph.rowptr[static_cast<std::size_t>(edge.from)] ||
            edge.csr_edge >=
                graph.rowptr[static_cast<std::size_t>(edge.from + 1)] ||
            graph.colind[static_cast<std::size_t>(edge.csr_edge)] != edge.to) {
          throw std::runtime_error("BF12 JSONL route contains an invalid edge");
        }
        if (!seen_edges.insert(edge_key(edge.from, edge.to)).second) continue;
        const EdgeAttr& attr =
            metadata.edge_attrs[static_cast<std::size_t>(edge.csr_edge)];
        if (attr.pip_data_index >= metadata.pip_data.size()) {
          throw std::runtime_error("BF12 route edge references invalid PIP data");
        }
        const PipData& pip =
            metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];
        if (!first_edge) output << ',';
        first_edge = false;
        output << "{\"from\":" << edge.from << ",\"to\":" << edge.to
               << ",\"csr_edge\":" << edge.csr_edge << ",\"tile\":";
        write_json_string(output, metadata_string(metadata, attr.tile_string));
        output << ",\"wire0\":";
        write_json_string(output, metadata_string(metadata, pip.wire0_string));
        output << ",\"wire1\":";
        write_json_string(output, metadata_string(metadata, pip.wire1_string));
        output << ",\"forward\":" << (pip.forward ? "true" : "false")
               << '}';
      }
    }
    output << "]}\n";
  }
}

}  // namespace routing

#ifndef ROUTING_PATHFINDER12_NO_MAIN
int main(int argc, char** argv) {
  try {
    if (argc < 2 ||
        (argc == 2 &&
         (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
      std::cout << routing::pathfinder12_usage(argc > 0 ? argv[0] : "pathfinder12");
      return argc < 2 ? 1 : 0;
    }
    const routing::Pathfinder12CommandLine command =
        routing::parse_pathfinder12_args(argc, argv);
    if (command.show_help) {
      std::cout << routing::pathfinder12_usage(argv[0]);
      return 0;
    }
    std::cerr << "[pathfinder12] Loading graph and metadata...\n";
    const auto publication =
        routing::interchange::snapshot_interchange_publication(
            command.csr_path, command.metadata_path);
    std::optional<routing::interchange::InterchangeArtifactPairId> csr_pair;
    routing::interchange::RoutingCsrSidecars sidecars;
    HostCsrF32 graph = routing::load_pathfinder12_csrbin(
        command.csr_path, &csr_pair, &sidecars);
    routing::RoutingMetadata metadata = routing::load_pathfinder12_metadata(
        command.metadata_path,
        command.routes_out_path.empty()
            ? routing::InterchangeMetadataLoadMode::kRoutingOnly
            : routing::InterchangeMetadataLoadMode::kRoutingWithRouteOutput);
    routing::interchange::verify_interchange_publication(
        command.csr_path, command.metadata_path, publication);
    routing::interchange::require_matching_interchange_pair_ids(
        csr_pair, metadata.artifact_pair_id, publication.generation);

    const std::size_t planned_net_count =
        command.options.net_limit == 0
            ? metadata.route_requests.size()
            : std::min(command.options.net_limit,
                       metadata.route_requests.size());
    std::cerr << "[pathfinder12] Initializing GPU and routing "
              << planned_net_count << " nets...\n";
    std::cerr << "[pathfinder12] Routing [--------------------] 0%\n";
    std::size_t next_progress_percent = 10;
    const routing::Pathfinder12ProgressCallback progress =
        [&next_progress_percent](std::size_t completed,
                                 std::size_t total,
                                 std::size_t batches) {
          const std::size_t percent =
              total == 0
                  ? 100
                  : static_cast<std::size_t>(
                        100.0L * static_cast<long double>(completed) /
                        static_cast<long double>(total));
          if (completed != total && percent < next_progress_percent) return;
          constexpr std::size_t kProgressBarWidth = 20;
          const std::size_t filled =
              std::min(kProgressBarWidth,
                       percent * kProgressBarWidth / 100);
          std::cerr << "[pathfinder12] Routing ["
                    << std::string(filled, '#')
                    << std::string(kProgressBarWidth - filled, '-') << "] "
                    << percent << "% (" << completed << '/' << total
                    << " nets, " << batches << " GPU batches)\n";
          next_progress_percent =
              std::min<std::size_t>(101, (percent / 10 + 1) * 10);
        };

    routing::Pathfinder12RoutingStatistics statistics;
    routing::PathfinderResult result = routing::route_all_nets_bf12(
        graph, metadata, sidecars.route_end_x.empty() ? nullptr : &sidecars,
        command.options, nullptr, &statistics, progress);
    std::cerr << "[pathfinder12] Routing complete: "
              << statistics.successful_net_count << '/'
              << statistics.net_count << " nets successful.\n";
    if (command.options.bf12_enable_telemetry) {
      const BellmanFord12BatchTelemetry& telemetry =
          statistics.backend_telemetry;
      std::cout << "{\"type\":\"bf12_pathfinder_telemetry\""
                << ",\"schema_version\":1"
                << ",\"requested_batch_size\":"
                << statistics.requested_batch_size
                << ",\"selected_batch_size\":"
                << statistics.selected_batch_size
                << ",\"batch_count\":" << statistics.batch_count
                << ",\"submitted_queries\":"
                << statistics.submitted_query_count
                << ",\"bounded_queries\":"
                << statistics.bounded_descriptor_count
                << ",\"unbounded_queries\":"
                << statistics.unbounded_descriptor_count
                << ",\"controller\":\""
                << bellman_ford12_controller_mode_name(
                       telemetry.controller_mode_used)
                << "\""
                << ",\"cooperative_launches\":"
                << telemetry.cooperative_controller_launch_count
                << ",\"host_rounds\":"
                << telemetry.host_fallback_round_count
                << ",\"synchronizations\":"
                << telemetry.synchronization_count
                << ",\"global_rounds\":"
                << telemetry.global_traversal_rounds
                << ",\"batch_fill_ratio\":"
                << telemetry.batch_fill_ratio
                << ",\"edges_examined\":" << telemetry.edges_examined
                << ",\"successful_relaxations\":"
                << telemetry.successful_relaxations
                << ",\"maximum_touched_state_density\":"
                << telemetry.maximum_touched_state_density
                << ",\"target_summary_nanoseconds\":"
                << telemetry.target_summary_nanoseconds
                << ",\"reconstruction_nanoseconds\":"
                << telemetry.reconstruction_nanoseconds
                << ",\"device_to_host_bytes\":"
                << telemetry.device_to_host_bytes
                << ",\"result_node_capacity\":"
                << telemetry.result_node_capacity
                << ",\"result_edge_capacity\":"
                << telemetry.result_edge_capacity
                << ",\"result_nodes_used\":"
                << telemetry.result_nodes_used
                << ",\"result_edges_used\":"
                << telemetry.result_edges_used
                << ",\"bounded_retries\":"
                << telemetry.bounded_query_retry_count
                << ",\"arena_overflow_retries\":"
                << telemetry.arena_overflow_retry_count
                << ",\"total_batch_wall_nanoseconds\":"
                << telemetry.total_batch_wall_nanoseconds
                << ",\"active_queries_per_round\":[";
      for (std::size_t index = 0;
           index < telemetry.active_queries_per_round.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << telemetry.active_queries_per_round[index];
      }
      std::cout << "],\"combined_frontier_size_per_round\":[";
      for (std::size_t index = 0;
           index < telemetry.combined_frontier_size_per_round.size();
           ++index) {
        if (index != 0) std::cout << ',';
        std::cout << telemetry.combined_frontier_size_per_round[index];
      }
      std::cout << "],\"touched_nodes_per_query\":[";
      for (std::size_t index = 0;
           index < telemetry.touched_nodes_per_query.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << telemetry.touched_nodes_per_query[index];
      }
      std::cout << "]}\n";
    }
    if (!command.routes_out_path.empty()) {
      if (!result.routed && !command.allow_unrouted) {
        std::cerr << "error: refusing to write routes because not all sinks "
                     "were reached\n";
        return 2;
      }
      std::cerr << "[pathfinder12] Writing routes to "
                << command.routes_out_path << "...\n";
      routing::write_pathfinder12_routes_jsonl(
          command.routes_out_path, graph, metadata, result);
    }
    std::cerr << "[pathfinder12] Done.\n";
    return result.routed || command.allow_unrouted ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
#endif
