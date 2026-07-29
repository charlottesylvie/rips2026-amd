#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


DELTA_DIRECTORY = Path(__file__).resolve().parents[1] / "delta_stepping"
SOURCE = DELTA_DIRECTORY / "delta_stepping_hip_CSR.cpp"
HEADER = DELTA_DIRECTORY / "delta_stepping_hip_CSR.hpp"
POLICY = DELTA_DIRECTORY / "delta_stepping_policy.hpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_between(text: str, begin: str, end: str) -> str:
    begin_offset = text.find(begin)
    require(begin_offset >= 0, f"missing source marker: {begin!r}")
    end_offset = text.find(end, begin_offset + len(begin))
    require(end_offset >= 0, f"missing source marker: {end!r}")
    return text[begin_offset:end_offset]


def test_controller_mode_policy(source: str, policy: str) -> None:
    require(
        "kFusedHostChecked = 2" in policy,
        "the explicit fused host-checked A/B mode disappeared",
    )
    require(
        "kDeltaSteppingCsrMaxControllerBatchSize = 64" in policy,
        "the cooperative watchdog batch cap is no longer 64",
    )
    effective_batch = source_between(
        policy,
        "inline std::uint32_t delta_stepping_effective_controller_batch_size(",
        "// These types deliberately use fixed-width representations",
    )
    require(
        "policy.mode == DeltaSteppingCsrControllerMode::kReducedRoundTrip"
        in effective_batch
        and "? policy.batch_size" in effective_batch
        and ": std::uint32_t{1}" in effective_batch,
        "host and fused modes must remain batch-one while reduced mode batches",
    )

    controller_request = source_between(
        source,
        "const bool cooperative_controller_requested =",
        "const std::uint32_t cooperative_action_budget =",
    )
    require(
        "DeltaSteppingCsrControllerMode::kFusedHostChecked"
        in controller_request
        and "DeltaSteppingCsrControllerMode::kReducedRoundTrip"
        in controller_request
        and "DeltaSteppingCsrControllerMode::kHostChecked"
        not in controller_request,
        "only the explicit fused/reduced modes may select the cooperative grid",
    )
    require(
        "if (!use_cooperative_controller) {\n"
        "    scratch.ensure_host_checked_reduction_storage();\n"
        "  }" in source,
        "the scalar host controller lost its stable lazy staging path",
    )
    action_budget = source_between(
        source,
        "const std::uint32_t cooperative_action_budget =",
        "CooperativeLaunchConfiguration cooperative_configuration;",
    )
    require(
        "delta_stepping_effective_controller_batch_size(controller_policy)"
        in action_budget,
        "production dispatch no longer derives fused/reduced action budgets "
        "from the validated controller policy",
    )


def test_scalar_host_reduction(source: str) -> None:
    reduction = source_between(
        source,
        "int find_min_pending_bucket(",
        "int mark_and_count_settled_targets(",
    )
    require(
        "hipMemcpyHostToDevice" not in reduction,
        "pending-min reduction reintroduced a blocking sentinel upload",
    )
    require(
        reduction.count("synchronize_explicit_stream(stream);") == 1
        and "if (!pending_updates_synchronized)" in reduction,
        "pending-min reduction must guard only an unsynchronized producer",
    )
    require(
        "hipMemcpyDeviceToHost" in reduction
        and "std::min_element" in reduction,
        "pending-min block results must be completed before the host reduction",
    )
    reduction_kernel = source_between(
        source,
        "__global__ void reduce_min_pending_bucket_kernel(",
        "__global__ void compact_pending_to_current_bucket_kernel(",
    )
    require(
        "atomicExch(block_mins + blockIdx.x, s_min[0]);" in reduction_kernel
        and "__threadfence_system();" in reduction_kernel,
        "each scalar-controller pending-min block must publish a host-visible result",
    )

    host_controller = source_between(
        source,
        "if (!use_cooperative_controller) {\n"
        '    PATHFINDER_PROFILE_RANGE("delta_step.scalar_host_controller");',
        "int touched_count_for_reset = -1;",
    )
    require(
        "std::swap(pending_queue, pending_scratch);" in host_controller
        and "std::swap(pending_count_device, pending_scratch_count_device);"
        in host_controller,
        "pending queue and count parity must advance together",
    )
    require(
        "hipMemcpyAsync(scratch.pending_count.get()," not in host_controller,
        "scalar host controller reintroduced the compacted pending-count D2D copy",
    )
    require(
        "pending_queue, pending_count_device" in host_controller,
        "relaxation and reduction must consume the active pending-count parity",
    )
    require(
        "scratch.host_pending_bucket_block_mins->get()" in host_controller
        and "pending_scratch," in host_controller,
        "pending-min results must use persistent pinned host storage and the "
        "inactive device queue",
    )
    require(
        "pending_updates_synchronized = false;" in host_controller
        and host_controller.count("pending_updates_synchronized = true;") >= 3,
        "heavy pending producers and target-observation completions must be "
        "tracked explicitly",
    )


def test_cooperative_memory_ordering(source: str) -> None:
    stable_loads = source_between(
        source,
        "controller_stable_load_u32(",
        "controller_set_status(",
    )
    require(
        "volatile unsigned int" in stable_loads
        and "volatile int" in stable_loads,
        "post-grid-barrier controller reads must use stable volatile loads",
    )

    bounded_append = source_between(
        source,
        "controller_bounded_append(",
        "template <bool CollectTelemetry, typename Grid>",
    )
    require(
        "atomicAdd(queue_tail, 1)" in bounded_append
        and "observed >= 0 && observed < capacity" in bounded_append
        and "atomicCAS(queue_tail" not in bounded_append,
        "bounded queue append must use one reservation plus an OOB guard",
    )

    grid_release = source_between(
        source,
        "controller_grid_release_sync(",
        "template <typename RowOffset,",
    )
    require(
        grid_release.count("grid.sync();") == 1
        and "if constexpr (CollectTelemetry)" in grid_release
        and "++state->grid_barriers" in grid_release,
        "each cooperative phase boundary must be one diagnosed grid barrier",
    )
    require(
        "__threadfence();" not in grid_release
        and "__threadfence_system();" not in grid_release,
        "per-phase grid barriers must not duplicate device/system fences",
    )

    block_minimum = source_between(
        source,
        "cooperative_reduce_min_pending(",
        "cooperative_compact_pending(",
    )
    require(
        "__shared__ int block_min[kBlockSize];" in block_minimum
        and "for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)"
        in block_minimum
        and block_minimum.count("atomicMin(args.min_pending_bucket") == 1
        and "atomicMin(args.min_pending_bucket, block_min[0]);"
        in block_minimum
        and "atomicMin(args.min_pending_bucket, local_min);"
        not in block_minimum,
        "pending minimum must reduce within each block before one global atomic",
    )

    controller_kernel = source_between(
        source,
        "__global__ void cooperative_delta_controller_kernel(",
        "query_cooperative_launch_configuration(",
    )
    require(
        "for (std::uint32_t action = 0; action < args.batch_size; ++action)"
        in controller_kernel,
        "the cooperative kernel no longer owns a bounded multi-action batch",
    )
    for stable_load in (
        "controller_stable_load_u32(&args.state->current_queue_parity)",
        "controller_stable_load_u32(&args.state->pending_queue_parity)",
        "controller_stable_load_u32(&args.state->generation_cursor)",
        "controller_stable_load_int(current_count_ptr)",
        "controller_stable_load_int(next_count_ptr)",
        "controller_stable_load_int(pending_count_ptr)",
        "controller_stable_load_int(args.heavy_count)",
        "controller_stable_load_int(new_pending_count_ptr)",
        "controller_stable_load_int(args.min_pending_bucket)",
    ):
        require(
            stable_load in controller_kernel,
            f"controller state lost stable load: {stable_load}",
        )
    require(
        "controller_atomic_load_u32(&args.state->current_queue_parity)"
        not in controller_kernel
        and "controller_atomic_load_u32(&args.state->pending_queue_parity)"
        not in controller_kernel
        and "controller_atomic_load_u32(&args.state->generation_cursor)"
        not in controller_kernel,
        "stable controller state reintroduced contended no-op atomic loads",
    )
    require(
        "controller_stable_load_u64(\n"
        "        &args.state->descriptor.current_bucket)" in controller_kernel
        and controller_kernel.count(
            "controller_phase_after_grid_sync(args.state)"
        ) == 4
        and "static_cast<int>(args.state->descriptor.current_bucket)"
        not in controller_kernel,
        "grid-uniform bucket/phase control must use stable post-barrier reads",
    )
    require(
        controller_kernel.count(
            "controller_grid_release_sync<CollectTelemetry>"
        ) == 16
        and controller_kernel.count("grid.sync();") == 1,
        "the cooperative kernel must retain diagnosed phase barriers plus "
        "exactly one final publication barrier",
    )
    require(
        controller_kernel.count("__threadfence_system();") == 1
        and "__threadfence();" not in controller_kernel,
        "only final host descriptor publication may retain a system fence",
    )
    final_publication = controller_kernel[controller_kernel.rfind("  if (leader) {") :]
    require(
        "++args.state->descriptor.publication_sequence;" in final_publication
        and "++args.state->grid_barriers;" in final_publication
        and final_publication.find("__threadfence_system();")
        < final_publication.find("grid.sync();"),
        "final descriptor publication must be sequenced and system-visible "
        "before kernel completion",
    )


def test_publication_diagnostics(source: str, header: str) -> None:
    descriptor_copy = source_between(
        source,
        "copy_controller_descriptor_to_host(",
        "template <bool TrackParents, bool UseEdgeParent>",
    )
    require(
        descriptor_copy.count("hipMemcpyAsync(") == 1
        and "hipMemcpyDeviceToHost" in descriptor_copy
        and descriptor_copy.count("hipStreamSynchronize(stream)") == 1,
        "each cooperative publication must use one descriptor copy/completion",
    )

    controller_loop = source_between(
        source,
        "if (use_cooperative_controller && !result.stopped_on_target",
        "} else if (use_cooperative_controller &&",
    )
    require(
        controller_loop.count("copy_controller_descriptor_to_host") == 1
        and "copy_scalar_to_host" not in controller_loop
        and "hipStreamSynchronize" not in controller_loop,
        "a cooperative batch reintroduced extra scalar or direct host waits",
    )
    for diagnostic in (
        "controller_action_slots_budgeted",
        "controller_actions_completed",
        "controller_unused_action_slots",
        "controller_publications",
        "controller_nonterminal_publications",
        "controller_terminal_publications",
    ):
        require(
            diagnostic in controller_loop and diagnostic in header,
            f"controller diagnostic {diagnostic} is not wired end to end",
        )
    require(
        "completed_actions > cooperative_action_budget" in controller_loop
        and "completed_actions != cooperative_action_budget" in controller_loop
        and "args.batch_size = cooperative_action_budget;" in controller_loop
        and "descriptor.publication_sequence != previous_publication + 1U"
        in controller_loop,
        "host publication validation no longer enforces bounded/full batches",
    )

    telemetry_copy = source_between(
        source,
        "void copy_device_telemetry_to_host(",
        "void initialize_legacy_predecessors_once(",
    )
    require(
        "offsetof(CooperativeDeltaControllerState, grid_barriers)"
        in telemetry_copy
        and "telemetry.cooperative_grid_barriers = controller_grid_barriers;"
        in telemetry_copy
        and "cooperative_grid_barriers" in header,
        "cooperative grid-barrier diagnostics are not copied to telemetry",
    )
    require(
        telemetry_copy.count("hipStreamSynchronize(stream)") == 1,
        "barrier telemetry must share the existing final telemetry completion",
    )


def test_initial_controller_publication_lifetime(source: str) -> None:
    controller_storage = source_between(
        source,
        "void ensure_controller_storage()",
        "void ensure_target_capacity(",
    )
    require(
        "PinnedHostBuffer<CooperativeDeltaControllerState>" in source
        and "host_controller_state" in controller_storage,
        "controller setup/publication staging must be pinned and workspace-owned",
    )

    source_publication = source_between(
        source,
        "const std::uint32_t source_generation =",
        "int* current_queue = scratch.current_queue.get();",
    )
    staged = "*scratch.host_controller_state->get() = initial_controller_state;"
    upload = "scratch.controller_state.get(), scratch.host_controller_state->get(),"
    launch = "initialize_delta_sources_kernel<TrackParents && !UseEdgeParent>"
    require(
        staged in source_publication
        and upload in source_publication
        and source_publication.find(staged) < source_publication.find(upload)
        < source_publication.find(launch)
        and source_publication.count("synchronize_explicit_stream(stream);") == 1,
        "initial controller H2D must use live pinned staging and share the "
        "existing post-source publication boundary",
    )
    require(
        "const std::exception_ptr setup_exception = std::current_exception();"
        in source_publication
        and "DS_DELTA_HIP_CHECK(hipStreamSynchronize(stream));"
        in source_publication
        and "std::rethrow_exception(setup_exception);" in source_publication,
        "a source-launch rejection must drain pinned controller staging before retry",
    )


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    policy = POLICY.read_text(encoding="utf-8")
    test_controller_mode_policy(source, policy)
    test_scalar_host_reduction(source)
    test_cooperative_memory_ordering(source)
    test_publication_diagnostics(source, header)
    test_initial_controller_publication_lifetime(source)
    print(
        "Delta-Stepping synchronization source policy test passed "
        "(cooperative waits = one descriptor completion per bounded batch; "
        "fused batch = 1, reduced batch <= 64)"
    )


if __name__ == "__main__":
    main()
