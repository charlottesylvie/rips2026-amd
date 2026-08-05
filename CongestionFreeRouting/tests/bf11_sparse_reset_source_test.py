#!/usr/bin/env python3
"""Source-level guardrails for BF11's HIP-only reset/controller policy."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CongestionFreeRouting/bellman_ford/bf11.cpp"
HIP_TEST = ROOT / "CongestionFreeRouting/tests/bf11_bounded_dynamic_hip_test.cpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def function_body(source: str, signature: str, next_signature: str) -> str:
    begin = source.index(signature)
    end = source.index(next_signature, begin)
    return source[begin:end]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    hip_test = HIP_TEST.read_text(encoding="utf-8")

    require(
        "workspace.stream == nullptr ? cooperative_block_count(workspace) : 0"
        in source,
        "explicit BF11 streams can still enter the full-residency cooperative controller",
    )
    require(
        "Index* touched_nodes" in source and "int* touched_count" in source,
        "BF11 workspace lost its sparse touched-state storage",
    )
    require(
        "relaxation.first_discovery" in source
        and "touched_nodes[touched_slot] = dst" in source,
        "first finite BF11 labels are no longer recorded for sparse reset",
    )
    atomic_relax = function_body(
        source,
        "__device__ __forceinline__ AtomicRelaxResult atomic_relax_strict(",
        "__device__ __forceinline__ float effective_edge_weight(",
    )
    require(
        "unsigned long long old_state = coherent_atomic_load(address);"
        in atomic_relax,
        "BF11 first-discovery checks bypass the coherent atomic-load helper",
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
        "BF11 primary state observation is not a relaxed agent-scope HIP atomic load",
    )
    require(
        "atomicCAS(address, 0ULL, 0ULL)" in source,
        "BF11 lost the proven CAS compatibility load",
    )
    coherent_load = function_body(
        source,
        "__device__ __forceinline__ unsigned long long coherent_atomic_load(",
        "__device__ __forceinline__ AtomicRelaxResult atomic_relax_strict(",
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

    telemetry_initialization = function_body(
        source,
        "void initialize_workspace_telemetry(",
        "void begin_telemetry_event(",
    )
    require(
        "hipDeviceAttributeWallClockRate" in telemetry_initialization,
        "BF11 cooperative telemetry does not query the fixed wall-clock rate",
    )
    cooperative_controller = function_body(
        source,
        "__global__ void frontier_controller_kernel(",
        "__global__ void summarize_target_paths_kernel(",
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

    host_controller = function_body(
        source, "SsspStatus run_host_controller(", "SsspStatus run_sssp("
    )
    require(
        "clear_touched_state_kernel" in host_controller
        and "clear_state_kernel" not in host_controller,
        "the normal BF11 host controller regressed to a full graph clear",
    )
    host_initialization = host_controller.split("int frontier_count", 1)[0]
    require(
        "hipStreamSynchronize" not in host_initialization,
        "BF11 reintroduced a host round trip between reset, seed, and round 1",
    )
    require(
        "if (workspace.iterations_per_host_check == 1)" in host_controller
        and "Preserve the original one-iteration host loop exactly"
        in host_controller,
        "BF11 no longer preserves an exact K=1 host-controller branch",
    )
    batched_relax = function_body(
        source,
        "__global__ void host_window_frontier_relax_kernel(",
        "__global__ void update_host_window_target_status_kernel(",
    )
    require(
        "controller_result->done != 0" in batched_relax
        and "controller_result->current_frontier_index" in batched_relax
        and "controller_result->iterations_used + 1" in batched_relax
        and "item += thread_count" in batched_relax,
        "BF11 host windows lost terminal gating, absolute tokens, parity, or "
        "grid-stride frontier coverage",
    )
    advance = function_body(
        source,
        "__global__ void advance_host_window_controller_kernel(",
        "__global__ void sparse_cost_update_kernel(",
    )
    terminal = advance.find("controller_result->done = 1")
    reset = advance.find("iteration_status->next_count = 0", terminal)
    require(
        "controller_result->done != 0) return" in advance
        and terminal >= 0
        and reset > terminal
        and "controller_result->current_frontier_index ^= 1" in advance,
        "BF11 host-window finalization can erase terminal state or lost frontier parity",
    )
    require(
        "sizeof(ControllerResult), hipMemcpyDeviceToHost" in host_controller
        and "window_size" in host_controller
        and "terminal_noops" in host_controller,
        "BF11 host windows no longer transfer one persistent controller latch "
        "or account for terminal no-ops",
    )
    gpu_controller = function_body(
        source, "__global__ void frontier_controller_kernel(",
        "__global__ void summarize_target_paths_kernel(",
    )
    require(
        "prior_touched_count" in gpu_controller
        and "for (Offset row = thread; row < graph.rows" not in gpu_controller,
        "the persistent BF11 controller regressed to a per-query full graph clear",
    )
    require(
        "needs_full_state_reset = true" in source
        and "fully_reset_workspace_state(workspace)" in source,
        "BF11 lost its defensive dense reset after an exceptional query",
    )
    require(
        "test_parallel_explicit_stream_host_controller" in hip_test
        and "bf11_internal_gpu_controller_launch_count() == 0" in hip_test
        and "bf11_internal_controller_fallback_count() == 16" in hip_test,
        "BF11 no longer has a concurrent explicit-stream fallback regression",
    )
    require(
        "test_defensive_reset_after_controller_error" in hip_test
        and "bf11_internal_dense_state_reset_count() == 1" in hip_test,
        "BF11 no longer behaviorally tests exceptional-query recovery",
    )

    print("BF11 sparse-reset/controller source policy test passed")


if __name__ == "__main__":
    main()
