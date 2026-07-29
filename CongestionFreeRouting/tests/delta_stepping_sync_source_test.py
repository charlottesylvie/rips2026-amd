#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "delta_stepping"
    / "delta_stepping_hip_CSR.cpp"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_between(text: str, begin: str, end: str) -> str:
    begin_offset = text.find(begin)
    require(begin_offset >= 0, f"missing source marker: {begin!r}")
    end_offset = text.find(end, begin_offset + len(begin))
    require(end_offset >= 0, f"missing source marker: {end!r}")
    return text[begin_offset:end_offset]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    require(
        source.count("scratch.ensure_host_checked_reduction_storage();") == 1
        and "if (!use_reduced_controller) {\n"
        "    scratch.ensure_host_checked_reduction_storage();\n"
        "  }" in source,
        "pinned reduction staging must remain lazy for the host controller",
    )

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
        "each pending-min block must publish a system-visible result",
    )

    host_controller = source_between(
        source,
        "if (!use_reduced_controller) {\n    for (int iter = 0;",
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
        "host controller reintroduced the compacted pending-count D2D copy",
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
    print(
        "Delta-Stepping synchronization source policy test passed "
        "(two waits removed per successful host-checked PathFinder bucket "
        "transition)"
    )


if __name__ == "__main__":
    main()
