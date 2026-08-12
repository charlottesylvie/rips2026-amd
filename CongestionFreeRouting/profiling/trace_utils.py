"""Algorithm-neutral trace/counter validation utilities."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int
    name: str = ""
    queue: str = ""
    dispatch_id: str = ""
    device: str = ""
    fields: Mapping[str, str] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def validate(self) -> None:
        if self.start_ns < 0 or self.end_ns < 0:
            raise ValueError(f"negative timestamp for {self.name!r}")
        if self.end_ns < self.start_ns:
            raise ValueError(f"negative duration for {self.name!r}")


def percentile(sorted_values: list[int | float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be in [0, 1]")
    index = max(
        0,
        min(len(sorted_values) - 1, math.ceil(quantile * len(sorted_values)) - 1),
    )
    return float(sorted_values[index])


def interval_union(rows: Iterable[tuple[int, int]]) -> tuple[int, int]:
    merged: list[list[int]] = []
    for start, end in sorted(rows):
        if start < 0 or end < start:
            raise ValueError(f"invalid interval [{start}, {end}]")
        if end == start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged), len(merged)


def concurrency_distribution(
    rows: Iterable[tuple[int, int]],
    *,
    span: tuple[int, int] | None = None,
) -> dict[int, int]:
    ordered = sorted(rows)
    ends: list[int] = []
    active = 0
    cursor = span[0] if span else (ordered[0][0] if ordered else None)
    distribution: dict[int, int] = defaultdict(int)
    for start, end in ordered:
        if start < 0 or end < start:
            raise ValueError(f"invalid interval [{start}, {end}]")
        if end == start:
            continue
        if cursor is None:
            cursor = start
        while ends and ends[0] <= start:
            next_end = ends[0]
            if next_end > cursor:
                distribution[active] += next_end - cursor
                cursor = next_end
            while ends and ends[0] == next_end:
                heapq.heappop(ends)
                active -= 1
        if start > cursor:
            distribution[active] += start - cursor
            cursor = start
        heapq.heappush(ends, end)
        active += 1
    while ends:
        next_end = ends[0]
        assert cursor is not None
        if next_end > cursor:
            distribution[active] += next_end - cursor
            cursor = next_end
        while ends and ends[0] == next_end:
            heapq.heappop(ends)
            active -= 1
    if span and cursor is not None and cursor < span[1]:
        distribution[0] += span[1] - cursor
    return dict(sorted(distribution.items()))


def validated_percentage(
    numerator: float,
    denominator: float,
    name: str,
    *,
    allow_accumulation: bool = False,
) -> float:
    if denominator <= 0:
        raise ValueError(f"{name} denominator is not positive")
    value = 100.0 * numerator / denominator
    if value < 0.0 or (value > 100.0 and not allow_accumulation):
        raise ValueError(f"impossible {name}: {value}")
    return value


def first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {column.lower().replace(" ", "_"): column for column in columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower().replace(" ", "_"))
        if found is not None:
            return found
    return None
