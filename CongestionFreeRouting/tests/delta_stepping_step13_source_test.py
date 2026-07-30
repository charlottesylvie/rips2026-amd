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

    light_launcher = source_between(
        source,
        "void launch_relax_light_edges(",
        "void launch_relax_heavy_edges(",
    )
    require(
        "int* pending_queue,\n    int* pending_count," in light_launcher
        and "pending_queue, pending_count, scratch.heavy_queue.get(),"
        in light_launcher,
        "light relaxation must append through the active queue/count pair",
    )

    heavy_launcher = source_between(
        source,
        "void launch_relax_heavy_edges(",
        "__global__ void clear_flags_from_queue_kernel(",
    )
    require(
        "int* pending_queue,\n    int* pending_count," in heavy_launcher
        and "pending_queue, pending_count, scratch.in_heavy.get(),"
        in heavy_launcher,
        "heavy relaxation must append through the active queue/count pair",
    )

    pending_pair_initialization = source_between(
        source,
        "int* pending_queue = scratch.pending_a.get();",
        "int current_bucket = 0;",
    )
    require(
        "int* pending_queue = scratch.pending_a.get();\n"
        "  int* pending_scratch = scratch.pending_b.get();\n"
        "  int* pending_count_device = scratch.pending_count.get();\n"
        "  int* pending_scratch_count_device = "
        "scratch.new_pending_count.get();"
        in pending_pair_initialization,
        "active and inactive pending queues must start with their own counts",
    )

    host_controller = source_between(
        source,
        "if (!use_reduced_controller) {\n    for (int iter = 0;",
        "int touched_count_for_reset = -1;",
    )
    require(
        host_controller.count(
            "next_current_generation, pending_queue, pending_count_device,"
        )
        == 4,
        "every light-relaxation variant must receive the active pending count",
    )
    require(
        host_controller.count(
            "exclusive_distance_limit, pending_queue,\n"
            "            pending_count_device, stream);"
        )
        == 2,
        "every heavy-relaxation variant must receive the active pending count",
    )
    require(
        "pending_queue, pending_count_device, device_count_blocks,"
        in host_controller,
        "pending reduction must consume the active queue/count pair",
    )

    step_9 = source_between(
        source,
        "int find_min_pending_bucket(",
        "int mark_and_count_settled_targets(",
    )
    require(
        "hipMemcpyHostToDevice" in step_9
        and "synchronize_explicit_stream(stream);" in step_9,
        "Step 9's sentinel upload wait must remain intact",
    )

    step_11_to_13 = host_controller[
        host_controller.find(
            "reset_int_zero_async(scratch.current_count.get(), stream);"
        ) :
    ]
    require(
        step_11_to_13.startswith(
            "reset_int_zero_async(scratch.current_count.get(), stream);"
        )
        and "reset_int_zero_async(pending_scratch_count_device, stream);"
        in step_11_to_13
        and step_11_to_13.count("synchronize_explicit_stream(stream);") == 1,
        "Step 11 must reset both compaction outputs and retain its wait",
    )
    require(
        "hipMemcpyAsync" not in step_11_to_13
        and "hipStreamSynchronize(stream)" not in step_11_to_13,
        "compaction added a pending publication or direct stream wait",
    )
    require(
        "pending_queue, pending_count_device, current_bucket, delta,"
        in step_11_to_13
        and "scratch.current_count.get(), pending_scratch,\n"
        "        pending_scratch_count_device," in step_11_to_13,
        "compaction input and output must use matching queue/count pairs",
    )

    step_12_to_13 = step_11_to_13[
        step_11_to_13.find("current_count = copy_scalar_to_host(") :
    ]
    require(
        step_12_to_13.startswith("current_count = copy_scalar_to_host(")
        and "scratch.current_count.get(), stream, scratch.host_scalar.get());"
        in step_12_to_13,
        "Step 12's compacted-frontier D2H observation must remain intact",
    )
    scalar_copy = source_between(
        source,
        "inline T copy_scalar_to_host(",
        "inline void reset_int_zero_async(",
    )
    require(
        "hipMemcpyDeviceToHost" in scalar_copy
        and "hipStreamSynchronize(stream)" in scalar_copy,
        "Step 12 must still wait for its D2H copy",
    )
    require(
        "hipMemcpyAsync" not in step_12_to_13
        and "synchronize_explicit_stream(stream);" not in step_12_to_13
        and "hipStreamSynchronize(stream)" not in step_12_to_13,
        "Step 13 reintroduced a pending-count publication or stream wait",
    )
    require(
        "std::swap(pending_queue, pending_scratch);\n"
        "    std::swap(pending_count_device, pending_scratch_count_device);"
        in step_12_to_13,
        "pending queue and count parity must advance together",
    )
    require(
        "scratch.pending_count.get()" not in host_controller
        and "scratch.new_pending_count.get()" not in host_controller,
        "host-controller work bypassed the active pending-count parity",
    )

    print("Delta-Stepping Step 13 source policy test passed")


if __name__ == "__main__":
    main()
