#!/usr/bin/env python3
"""Source guardrails for BF11 reset, mark, root, and controller invariants."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CongestionFreeRouting/bellman_ford/bf11.cpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def braced_region(source: str, signature: str) -> str:
    """Return one definition, including nested braces, from its signature."""
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth = 0
    for offset in range(opening, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[begin : offset + 1]
    raise AssertionError(f"unterminated source region: {signature}")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    require(
        '#include "bf11_execution_policy.hpp"' in source,
        "BF11 production code no longer includes its host-tested policy",
    )
    require(
        '#include "bf11_graph_execution_policy.hpp"' in source,
        "BF11 production code bypasses its host-tested graph lifecycle policy",
    )
    require(
        "workspace.stream == nullptr" in source
        and "retain_default_cooperative_path" in source
        and "workspace_options.segment_rounds == 1" in source,
        "explicit BF11 streams can still enter the full-residency controller",
    )
    graph_enqueue = braced_region(source, "enqueue_graph_segment(")
    graph_backend = braced_region(source, "class HipGraphSegmentBackend")
    require(
        "same_bounds(workspace.graph_bounds, options.bounds)" in graph_enqueue
        and "workspace_.graph_bounds = options_.bounds" in graph_backend,
        "BF11 HIP Graph replay can reuse kernel arguments from another box",
    )
    require(
        "hipStreamCaptureModeThreadLocal" in graph_backend
        and "hipStreamCaptureModeGlobal" not in graph_backend,
        "BF11 worker captures are not isolated to their owning host thread",
    )
    require(
        "hipPeekAtLastError" in graph_backend
        and "hipStreamIsCapturing" in graph_backend
        and "hipStreamCaptureStatusNone" in graph_backend,
        "BF11 graph fallback does not preserve prior errors or verify capture cleanup",
    )
    require(
        "try_enqueue_direct_segment" in graph_backend
        and "is_recoverable_capture_enqueue_error" in graph_backend
        and "hipErrorStreamCaptureUnsupported" in graph_backend
        and "hipErrorStreamCaptureWrongThread" in graph_backend
        and "record_consumed_unrecoverable_failure" in graph_backend,
        "captured enqueue errors no longer distinguish capture fallback from "
        "unrelated kernel/device failure",
    )
    require(
        "wait_for_bf11_graph_capture_barrier" in graph_backend
        and "hipStreamSynchronize(workspace_.stream)" in graph_backend,
        "BF11 lost deterministic concurrent-capture or real invalidation "
        "target-test hooks",
    )
    require(
        "launch_failure_requires_restart" in graph_backend
        and "prepare_launch_failure" in graph_backend
        and "hipStreamSynchronize" in graph_backend
        and "kLaunchAfterSubmit" in graph_backend
        and "restart_query_direct" in source,
        "a partial HIP Graph launch can be replayed directly without a full query restart",
    )
    require(
        "Index* touched_nodes" in source and "int* touched_count" in source,
        "BF11 workspace lost its sparse touched-state storage",
    )
    require(
        "source_mask" not in source,
        "BF11 reintroduced the graph-sized source mask or a hot source-mask read",
    )

    atomic_relax = braced_region(
        source,
        "__device__ __forceinline__ AtomicRelaxResult atomic_relax_strict(",
    )
    require(
        "unsigned long long old_state = coherent_atomic_load(address);"
        in atomic_relax,
        "BF11 first-discovery checks bypass the coherent atomic-load helper",
    )
    require(
        "candidate_bits < state_distance_bits(old_state)" in atomic_relax
        and "state_distance_bits(assumed) == kInfinityBits" in atomic_relax,
        "BF11 no longer uses a strict packed-state improvement with one "
        "infinity-to-finite owner",
    )
    require(
        "BF11_FORCE_CAS_ATOMIC_LOAD" in source,
        "BF11 lost the compile-time CAS compatibility override",
    )
    require(
        re.search(
            r"__hip_atomic_load\s*\(\s*address\s*,\s*__ATOMIC_RELAXED\s*,"
            r"\s*__HIP_MEMORY_SCOPE_AGENT\s*\)",
            source,
        )
        is not None,
        "BF11 primary state observation is not a relaxed agent-scope atomic load",
    )
    coherent_load = braced_region(
        source,
        "__device__ __forceinline__ unsigned long long coherent_atomic_load(",
    )
    require(
        "BF11_FORCE_CAS_ATOMIC_LOAD" in coherent_load
        and "__hip_atomic_load" in coherent_load
        and "atomicCAS(address, 0ULL, 0ULL)" in coherent_load
        and "#else" in coherent_load,
        "BF11 coherent load does not keep guarded HIP-load and CAS paths",
    )
    require(
        re.search(r"(?:return\s+|=\s*)\*\s*address\b", coherent_load) is None
        and re.search(r"\baddress\s*\[\s*0\s*\]", coherent_load) is None,
        "BF11 coherent state observation regressed to an ordinary cached load",
    )

    relax_vertex = braced_region(
        source,
        "__device__ __forceinline__ unsigned int relax_frontier_vertex(",
    )
    require(
        "publication.first_discovery" in relax_vertex
        and "pending_touched" in relax_vertex
        and "touched_nodes" in relax_vertex
        and "touched_count" in relax_vertex,
        "first finite BF11 labels are no longer published for sparse reset",
    )
    require(
        "coherent_atomic_load(&best_state[from])" in relax_vertex,
        "BF11 reads a concurrently CAS-updated frontier label non-atomically",
    )
    segmented_relax = braced_region(
        source, "__global__ void segmented_frontier_relax_kernel("
    )
    require(
        "block_reserve_one" in segmented_relax
        and "pending_touched" in segmented_relax
        and "pending_queue" in segmented_relax,
        "segmented BF11 lost aggregated frontier/touched-list reservations",
    )

    summarize = braced_region(
        source, "__global__ void summarize_target_paths_kernel("
    )
    require(
        "edge == kNoPredecessor" in summarize
        and "state_distance_bits(state) == 0u" in summarize,
        "BF11 reconstruction no longer recognizes roots by zero distance and "
        "no predecessor",
    )
    materialize = braced_region(
        source, "__global__ void materialize_target_paths_kernel("
    )
    require(
        "state_distance_bits(root_state) != 0u" in materialize
        and "kNoPredecessor" in materialize,
        "BF11 materialization lost its final root-state validation",
    )

    mark_reservation = braced_region(
        source, "unsigned int reserve_query_mark_tokens("
    )
    require(
        "reserve_mark_tokens(" in mark_reservation
        and "dense_reset_required" in mark_reservation,
        "production BF11 bypasses the forced-wrap-tested mark-token policy",
    )
    require(
        "next_mark_generation" in source
        and "clear_marks_kernel" in source
        and "g_bf11_mark_generation_limit" in source
        and "bf11_internal_set_mark_generation_limit" in source,
        "BF11 lost generation marks or its forced-wrap test control",
    )
    sparse_reset = braced_region(
        source, "__global__ void clear_touched_state_kernel("
    )
    require(
        "next_marks" not in sparse_reset,
        "normal BF11 sparse/adaptive reset unnecessarily clears frontier marks",
    )
    require(
        "dense_threshold" in sparse_reset
        and "touched_count" in sparse_reset
        and "controller->reset_mode" in sparse_reset
        and "for (Offset row" in sparse_reset
        and "touched_nodes[item]" in sparse_reset,
        "BF11 reset no longer selects sparse versus contiguous dense work on-device",
    )
    require(
        "hipMemcpy" not in sparse_reset
        and "hipStreamSynchronize" not in sparse_reset,
        "BF11 adaptive reset reintroduced a host decision rendezvous",
    )

    prepare_query = braced_region(source, "void prepare_query_controller(")
    require(
        "clear_touched_state_kernel" in prepare_query
        and "clear_marks_kernel" not in prepare_query
        and "next_marks" not in prepare_query,
        "normal query setup regressed to per-query mark clearing",
    )
    require(
        "hipMemcpy" not in prepare_query
        and "hipStreamSynchronize" not in prepare_query,
        "BF11 added a host round trip between reset, seed, and round one",
    )
    seed_sources = braced_region(source, "__global__ void seed_sources_kernel(")
    require(
        "pack_state(0u, kNoPredecessor)" in seed_sources
        and "touched_nodes[item] = source" in seed_sources
        and "next_marks" not in seed_sources,
        "source seeding lost its root/touched invariant or clears generation marks",
    )
    defensive_reset = braced_region(
        source, "void fully_reset_workspace_state("
    )
    require(
        "clear_state_kernel" in defensive_reset
        and "next_mark_generation = 1" in defensive_reset,
        "exception recovery lost its defensive state/mark reset",
    )

    for signature in (
        "__global__ void begin_segment_round_kernel(",
        "__global__ void segmented_frontier_relax_kernel(",
        "__global__ void finalize_segment_round_kernel(",
    ):
        require(signature in source, f"BF11 lost segmented kernel {signature}")

    telemetry_initialization = braced_region(
        source, "void initialize_workspace_telemetry("
    )
    require(
        "hipDeviceAttributeWallClockRate" in telemetry_initialization,
        "BF11 cooperative telemetry does not query the fixed wall-clock rate",
    )
    cooperative_controller = braced_region(
        source, "__global__ void frontier_controller_kernel("
    )
    require(
        "wall_clock64()" in cooperative_controller,
        "BF11 cooperative telemetry does not use the GFX11-safe wall clock",
    )
    require(
        re.search(r"(?<!wall_)clock64\s*\(", source) is None,
        "BF11 telemetry uses bare clock64, which is unreliable on GFX11",
    )
    require(
        re.search(r"\b\w+\s*\.\s*clockRate\b", source) is None,
        "BF11 telemetry converts timer ticks with multiprocessor clockRate",
    )

    require(
        "float* dynamic_vertex_cost = nullptr" in source
        and "dynamic_cost_identity = true" in source,
        "BF11 lost lazy identity-mode dynamic-cost storage",
    )
    require(
        "dynamic_cost_epoch_valid" in source
        and "dynamic costs require a complete replacement" in source
        and "after a failed update" in source,
        "BF11 can silently reuse a partially updated dynamic-cost epoch",
    )
    make_workspace = braced_region(source, "DeviceWorkspace make_workspace(")
    require(
        "device_allocate<float>" not in make_workspace,
        "BF11 eagerly allocates a graph-sized dynamic multiplier array",
    )
    require(
        "needs_full_state_reset = true" in source
        and "fully_reset_workspace_state(workspace)" in source,
        "BF11 lost defensive dense recovery after an exceptional query",
    )
    require(
        "compact_node_transfer_capacity" in source
        and "compact_edge_transfer_capacity" in source,
        "BF11 again transfers retained compact-arena high-water bytes per query",
    )

    print("BF11 sparse-reset/controller source policy test passed")


if __name__ == "__main__":
    main()
