#!/usr/bin/env python3
"""Source-level guardrails for BF12's HIP-only runtime fast paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BF12 = ROOT / "CongestionFreeRouting/bellman_ford/bf12.cpp"
PATHFINDER12 = ROOT / "CongestionFreeRouting/pathfinder12.cpp"
HIP_TEST = ROOT / "CongestionFreeRouting/tests/bf12_batched_hip_test.cpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def between(source: str, begin: str, end: str) -> str:
    begin_offset = source.find(begin)
    require(begin_offset >= 0, f"missing source marker: {begin!r}")
    end_offset = source.find(end, begin_offset + len(begin))
    require(end_offset >= 0, f"missing source marker: {end!r}")
    return source[begin_offset:end_offset]


def main() -> None:
    bf12 = BF12.read_text(encoding="utf-8")
    pathfinder = PATHFINDER12.read_text(encoding="utf-8")
    hip_test = HIP_TEST.read_text(encoding="utf-8")

    sparse_grid = between(
        bf12,
        "dim3 sparse_reset_grid(",
        "__device__ __forceinline__ std::size_t logical_thread_id()",
    )
    require(
        "kMaximumSparseResetBlocks = 256" in sparse_grid,
        "BF12 sparse reset lost its bounded 256-block grid",
    )
    sparse_kernel = between(
        bf12,
        "__global__ void clear_sparse_state_kernel(",
        "__global__ void fill_float_kernel(",
    )
    require(
        "item < safe_count" in sparse_kernel
        and "item += logical_thread_count()" in sparse_kernel,
        "BF12 sparse reset no longer strides only over touched states",
    )
    host_initialization = between(
        bf12, "void initialize_host_batch(", "void run_host_controller("
    )
    require(
        "sparse_reset_grid(workspace.state_capacity)" in host_initialization
        and "clear_sparse_state_kernel" in host_initialization,
        "normal BF12 reuse no longer uses the bounded sparse reset",
    )

    host_controller = between(
        bf12, "void run_host_controller(", "void launch_cooperative_controller("
    )
    require(
        host_controller.count("hipMemcpyAsync(") == 1
        and "host_round_status" in host_controller
        and "host_query_controls" not in host_controller,
        "BF12 host rounds no longer use exactly one compact D2H status copy",
    )
    require(
        "if (target_check_due(batch, round + 1u))" in host_controller,
        "BF12 host rounds launch target checks when no interval is due",
    )

    workspace = between(
        bf12,
        "struct DeviceWorkspace {",
        "__host__ __device__ inline unsigned long long pack_state(",
    )
    require(
        "Index* sources =" not in workspace
        and "Index* targets =" not in workspace
        and "StateIndex* source_states" in workspace
        and "StateIndex* target_states" in workspace,
        "BF12 restored unused raw endpoint device buffers",
    )
    upload = between(
        bf12, "void upload_batch_inputs(", "void defensive_dense_reset("
    )
    require(
        upload.count("hipMemcpyAsync(") == 3
        and "copy BF12 source states" in upload
        and "copy BF12 target states" in upload
        and "copy BF12 query descriptors" in upload
        and "copy BF12 sources" not in upload
        and "copy BF12 targets" not in upload,
        "BF12 endpoint upload is not the three required composite/control copies",
    )

    transfer_layout = between(
        bf12,
        "struct TransferHeaderLayout {",
        "static_assert(sizeof(DeviceTargetSummary)",
    )
    require(
        "descriptors" not in transfer_layout,
        "BF12 transfer header still packs host-unused query descriptors",
    )
    result_copies = between(
        bf12, "std::uint64_t enqueue_result_copies(", "void unpack_result_header("
    )
    require(
        "pack_transfer_header_kernel" in result_copies
        and "hipMemcpyDeviceToDevice" not in result_copies,
        "BF12 compact result header regressed to repeated D2D packing calls",
    )

    policy = between(bf12, "BellmanFord12WorkerDecision policy_decision(",
                     "void merge_telemetry(")
    reuse = policy.find("if (all_policy_storage_reusable)")
    memory_query = policy.find("hipMemGetInfo", reuse)
    require(
        reuse >= 0 and memory_query > reuse,
        "BF12 lost the resident-capacity policy path that skips hipMemGetInfo",
    )
    run_once = between(bf12, "BellmanFord12BatchResult run_once(",
                       "BellmanFord12BatchResult run_with_retry(")
    require(
        "options.controller_mode != BellmanFord12ControllerMode::HostBatch"
        in run_once,
        "forced BF12 host mode still requests cooperative occupancy",
    )
    require(
        "dynamic_costs_are_unit" in bf12
        and "if (all_unit && dynamic_costs_are_unit) return;" in bf12,
        "BF12 lost redundant all-one dynamic-cost upload suppression",
    )
    sidecar_boundary = between(
        bf12,
        "HostSidecarView sidecar_view(\n    const routing::interchange::RoutingCsrSidecars&",
        "HostSidecarView sidecar_view(const BellmanFord12NodeSidecars&",
    )
    require(
        "validate_routing_csr_sidecars" in sidecar_boundary
        and "already_validated" in bf12,
        "BF12 no longer validates complete routing sidecars exactly once",
    )

    trivial = between(
        pathfinder, "RoutedNet trivial_net_result(",
        "std::size_t route_request_count("
    )
    require(
        "vector<bool>" not in trivial and "source_seen" not in trivial,
        "Pathfinder12 trivial results restored a per-net V-sized clear",
    )
    public_finalize = between(
        pathfinder, "RoutedNet finalize_bf12_routed_net(",
        "PathfinderResult route_all_nets_bf12("
    )
    require(
        "thread_local std::optional<RouteTreeScratch>" in public_finalize,
        "checked Pathfinder12 finalization no longer reuses stamped scratch",
    )
    route_all = between(
        pathfinder, "PathfinderResult route_all_nets_bf12(",
        "namespace {\n\nstd::uint64_t read_u64("
    )
    require(
        "prepare_bf12_batch_validated" in route_all
        and "finalize_bf12_routed_net_validated" in route_all
        and "RouteTreeScratch route_tree_scratch" in route_all
        and "workspace.update_vertex_costs" not in route_all,
        "Pathfinder12 hot routing loop lost its validated/stamped fast path",
    )
    require(
        "options.bf12_controller != BellmanFord12ControllerMode::HostBatch"
        in route_all,
        "Pathfinder12 forced host scheduling still requests occupancy",
    )

    require(
        "test_single_query_endpoint_semantics_and_exception_recovery" in hip_test
        and "overflowing effective edge weight" in hip_test
        and "identity_and_duplicates" in hip_test,
        "BF12 HIP coverage lost duplicate/identity or exception-reuse cases",
    )

    print("BF12 runtime structure test passed")


if __name__ == "__main__":
    main()
