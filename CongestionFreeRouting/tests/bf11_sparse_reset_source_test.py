#!/usr/bin/env python3
"""Source-level guardrails for BF11's HIP-only reset/controller policy."""

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
    require(
        "old_state = atomicCAS(address, 0ULL, 0ULL)" in source,
        "BF11 first-discovery checks lost their coherent reused-state load",
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
