from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SplitAssignment:
    record_id: str
    source_id: str
    split: str


def deterministic_source_split(
    source_ids: Iterable[str],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    if not fractions:
        raise ValueError("at least one split fraction is required")
    total = sum(float(value) for value in fractions.values())
    if total <= 0:
        raise ValueError("split fractions must have positive total")
    names = list(fractions)
    cumulative = []
    running = 0.0
    for name in names:
        fraction = float(fractions[name]) / total
        if fraction < 0:
            raise ValueError("split fractions must be non-negative")
        running += fraction
        cumulative.append((running, name))
    result = {}
    for source_id in sorted(set(source_ids)):
        digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        split = names[-1]
        for threshold, name in cumulative:
            if value < threshold:
                split = name
                break
        result[source_id] = split
    return result


def assign_records_by_source(
    records: Sequence[Mapping],
    source_assignment: Mapping[str, str],
    *,
    record_id_key: str = "record_id",
    source_id_key: str = "source_id",
) -> list[SplitAssignment]:
    assignments = []
    for record in records:
        record_id = str(record[record_id_key])
        source_id = str(record[source_id_key])
        if source_id not in source_assignment:
            raise KeyError(f"source ID missing from split assignment: {source_id}")
        assignments.append(SplitAssignment(record_id, source_id, source_assignment[source_id]))
    validate_source_isolation(assignments)
    return assignments


def validate_source_isolation(assignments: Sequence[SplitAssignment]) -> None:
    seen: dict[str, str] = {}
    duplicate_records: set[str] = set()
    record_ids: set[str] = set()
    for assignment in assignments:
        if assignment.record_id in record_ids:
            duplicate_records.add(assignment.record_id)
        record_ids.add(assignment.record_id)
        existing = seen.setdefault(assignment.source_id, assignment.split)
        if existing != assignment.split:
            raise ValueError(
                f"source {assignment.source_id!r} appears in both {existing!r} and {assignment.split!r}"
            )
    if duplicate_records:
        raise ValueError(f"duplicate record IDs: {sorted(duplicate_records)}")
