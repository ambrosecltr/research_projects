"""Lossless fixed-length packing that never joins forbidden source families."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schema import TokenSequence


@dataclass(frozen=True, slots=True)
class PackedSequence:
    pack_id: int
    boundary_key: str
    example_ids: tuple[str, ...]
    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    objective: str = "conditional_poetry"

    def __post_init__(self) -> None:
        if isinstance(self.pack_id, bool) or not isinstance(self.pack_id, int) or self.pack_id < 0:
            raise ValueError("pack_id must be a non-negative integer")
        if not isinstance(self.boundary_key, str) or not self.boundary_key:
            raise ValueError("packed sequences require a boundary key")
        if not isinstance(self.example_ids, tuple) or any(
            not isinstance(example_id, str) or not example_id for example_id in self.example_ids
        ):
            raise ValueError("packed example IDs must be non-empty strings")
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError("packed IDs and loss mask must align")
        if not self.example_ids or not self.input_ids or not any(self.loss_mask):
            raise ValueError("packed sequences require examples, tokens, and supervised targets")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.input_ids
        ):
            raise ValueError("packed token IDs must be non-negative integers")
        if any(not isinstance(enabled, bool) for enabled in self.loss_mask):
            raise ValueError("packed loss mask values must be booleans")
        if self.objective not in {"conditional_poetry", "auxiliary_prose_ntp", "poetry_ntp"}:
            raise ValueError("packed sequence has unsupported objective")


def chunk_sequence(
    sequence: TokenSequence, *, sequence_length: int, context_overlap: int = 1
) -> tuple[TokenSequence, ...]:
    """Split long records without losing supervised target tokens or source family.

    One preceding context token is carried into later chunks but marked non-loss,
    so every original target token contributes exactly once.
    """
    if sequence_length < 2 or not 0 <= context_overlap < sequence_length:
        raise ValueError("invalid chunk length or context overlap")
    target_positions = tuple(index for index, enabled in enumerate(sequence.loss_mask) if enabled)
    if not target_positions:
        raise ValueError(f"{sequence.example_id} has no supervised tokens to pack")
    if len(sequence.input_ids) <= sequence_length:
        return (sequence,)
    chunks: list[TokenSequence] = []
    target_cursor = 0
    chunk_index = 0
    while target_cursor < len(target_positions):
        first_target = target_positions[target_cursor]
        start = max(0, first_target - context_overlap)
        end = min(start + sequence_length, len(sequence.input_ids))
        ids = sequence.input_ids[start:end]
        next_target_cursor = target_cursor
        while (
            next_target_cursor < len(target_positions)
            and target_positions[next_target_cursor] < end
        ):
            next_target_cursor += 1
        if next_target_cursor == target_cursor:
            raise AssertionError("target-aware chunking failed to include its first target")
        mask_values = [False] * len(ids)
        for target_index in range(target_cursor, next_target_cursor):
            mask_values[target_positions[target_index] - start] = True
        chunks.append(
            TokenSequence(
                example_id=f"{sequence.example_id}:chunk:{chunk_index}",
                boundary_key=f"{sequence.boundary_key}:chunk:{chunk_index}",
                input_ids=ids,
                loss_mask=tuple(mask_values),
                objective=sequence.objective,
            )
        )
        target_cursor = next_target_cursor
        chunk_index += 1
    return tuple(chunks)


def pack_sequences(
    sequences: Iterable[TokenSequence], *, sequence_length: int
) -> tuple[PackedSequence, ...]:
    """First-fit pack without crossing poem families; long records are chunked safely."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    result: list[PackedSequence] = []
    current_ids: list[int] = []
    current_mask: list[bool] = []
    current_examples: list[str] = []
    current_key: str | None = None
    current_objective: str | None = None

    def flush() -> None:
        nonlocal current_ids, current_mask, current_examples, current_key, current_objective
        if current_ids:
            result.append(
                PackedSequence(
                    pack_id=len(result),
                    boundary_key=current_key or "",
                    example_ids=tuple(current_examples),
                    input_ids=tuple(current_ids),
                    loss_mask=tuple(current_mask),
                    objective=current_objective or "conditional_poetry",
                )
            )
        current_ids, current_mask, current_examples, current_key, current_objective = (
            [],
            [],
            [],
            None,
            None,
        )

    expanded = (
        chunk
        for sequence in sequences
        for chunk in chunk_sequence(sequence, sequence_length=sequence_length)
    )
    for sequence in expanded:
        if (
            current_key != sequence.boundary_key
            or current_objective != sequence.objective
            or len(current_ids) + len(sequence.input_ids) > sequence_length
        ):
            flush()
        current_key = sequence.boundary_key
        current_objective = sequence.objective
        current_ids.extend(sequence.input_ids)
        current_mask.extend(sequence.loss_mask)
        current_examples.append(sequence.example_id)
    flush()
    return tuple(result)
