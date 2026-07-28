from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from itertools import pairwise
from typing import ClassVar

import pytest

import poetry50m.data.packing as packing_module
from poetry50m.data.packing import chunk_sequence
from poetry50m.data.schema import TokenSequence


def _brute_chunk_sequence(
    sequence: TokenSequence,
    *,
    sequence_length: int,
    context_overlap: int,
) -> tuple[TokenSequence, ...]:
    target_positions = tuple(index for index, enabled in enumerate(sequence.loss_mask) if enabled)
    if len(sequence.input_ids) <= sequence_length:
        return (sequence,)
    chunks: list[TokenSequence] = []
    remaining_targets = list(target_positions)
    chunk_index = 0
    while remaining_targets:
        first_target = remaining_targets[0]
        start = max(0, first_target - context_overlap)
        end = min(start + sequence_length, len(sequence.input_ids))
        ids = sequence.input_ids[start:end]
        target_set = set(remaining_targets)
        mask = tuple(index + start in target_set for index in range(len(ids)))
        covered = [index for index in remaining_targets if index < end]
        chunks.append(
            TokenSequence(
                example_id=f"{sequence.example_id}:chunk:{chunk_index}",
                boundary_key=f"{sequence.boundary_key}:chunk:{chunk_index}",
                input_ids=ids,
                loss_mask=mask,
                objective=sequence.objective,
            )
        )
        remaining_targets = [index for index in remaining_targets if index not in covered]
        chunk_index += 1
    return tuple(chunks)


@pytest.mark.parametrize("target_density", (0.03, 0.3, 0.95))
def test_cursor_chunking_matches_brute_oracle_for_random_masks(
    target_density: float,
):
    generator = random.Random(20_260_726 + int(target_density * 100))
    for case_index in range(150):
        token_count = generator.randint(1, 160)
        sequence_length = generator.randint(2, 40)
        context_overlap = generator.randrange(sequence_length)
        loss_mask = [generator.random() < target_density for _ in range(token_count)]
        loss_mask[generator.randrange(token_count)] = True
        sequence = TokenSequence(
            example_id=f"example-{target_density}-{case_index}",
            boundary_key=f"family-{case_index % 7}",
            input_ids=tuple(range(token_count)),
            loss_mask=tuple(loss_mask),
            objective=("conditional_poetry" if case_index % 2 == 0 else "auxiliary_prose_ntp"),
        )

        assert chunk_sequence(
            sequence,
            sequence_length=sequence_length,
            context_overlap=context_overlap,
        ) == _brute_chunk_sequence(
            sequence,
            sequence_length=sequence_length,
            context_overlap=context_overlap,
        )


class _ComparisonBoundIndex(int):
    comparisons: ClassVar[int] = 0
    comparison_limit: ClassVar[int] = 10_000

    def __eq__(self, other: object) -> bool:
        type(self).comparisons += 1
        if type(self).comparisons > type(self).comparison_limit:
            raise AssertionError("dense chunking regressed to list membership")
        return isinstance(other, int) and int(self) == int(other)

    __hash__ = int.__hash__


def test_very_long_dense_targets_advance_without_list_membership(
    monkeypatch: pytest.MonkeyPatch,
):
    token_count = 100_000
    sequence = TokenSequence(
        "dense",
        "dense-family",
        tuple(range(token_count)),
        (True,) * token_count,
    )
    builtin_enumerate = enumerate

    def comparison_bound_enumerate(
        values: Iterable[bool],
    ) -> Iterator[tuple[int, bool]]:
        for index, value in builtin_enumerate(values):
            yield _ComparisonBoundIndex(index), value

    _ComparisonBoundIndex.comparisons = 0
    monkeypatch.setattr(
        packing_module,
        "enumerate",
        comparison_bound_enumerate,
        raising=False,
    )

    chunks = chunk_sequence(sequence, sequence_length=257, context_overlap=1)

    assert _ComparisonBoundIndex.comparisons == 0
    assert sum(sum(chunk.loss_mask) for chunk in chunks) == token_count
    assert (
        tuple(dict.fromkeys(token for chunk in chunks for token in chunk.input_ids))
        == sequence.input_ids
    )
    for chunk_index, chunk in enumerate(chunks):
        assert chunk.example_id == f"dense:chunk:{chunk_index}"
        assert chunk.boundary_key == f"dense-family:chunk:{chunk_index}"
    for previous, current in pairwise(chunks):
        assert previous.input_ids[-1] == current.input_ids[0]
        assert current.loss_mask[0] is False
