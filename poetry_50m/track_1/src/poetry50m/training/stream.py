"""Small checkpointable batch-stream adapter useful for deterministic experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from torch import Tensor


class Batch(TypedDict):
    input_ids: Tensor
    targets: Tensor
    loss_mask: NotRequired[Tensor]
    example_ids: NotRequired[Sequence[str] | Sequence[int]]
    data_token_count: NotRequired[int]


@dataclass(frozen=True, slots=True)
class SkippedBatchStats:
    """Exact stream movement performed by a virtual training transport."""

    batch_count: int
    data_token_count: int

    def __post_init__(self) -> None:
        if self.batch_count < 0 or self.data_token_count < 0:
            raise ValueError("skipped batch statistics cannot be negative")


class CyclingBatchStream:
    """Cycle a fixed non-empty batch sequence while retaining an exact cursor."""

    def __init__(self, batches: Sequence[Batch]) -> None:
        if not batches:
            raise ValueError("batches must not be empty")
        self._batches = batches
        self._index = 0

    def __iter__(self) -> CyclingBatchStream:
        return self

    def __next__(self) -> Batch:
        batch = self._batches[self._index]
        self._index = (self._index + 1) % len(self._batches)
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"index": self._index}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        index = state.get("index")
        if not isinstance(index, int) or not 0 <= index < len(self._batches):
            raise ValueError("invalid CyclingBatchStream cursor")
        self._index = index

    def skip_batches(self, count: int) -> SkippedBatchStats:
        if count < 0:
            raise ValueError("skip count cannot be negative")
        data_token_count = 0
        for _ in range(count):
            batch = next(self)
            data_token_count += _data_token_count(batch)
        return SkippedBatchStats(batch_count=count, data_token_count=data_token_count)


def _data_token_count(batch: Batch) -> int:
    supplied = batch.get("data_token_count")
    dense_count = int(batch["input_ids"].numel())
    if supplied is None:
        return dense_count
    if isinstance(supplied, bool) or not isinstance(supplied, int):
        raise TypeError("data_token_count must be an integer")
    if not 0 < supplied <= dense_count:
        raise ValueError("data_token_count must lie in [1, input_ids.numel()]")
    return supplied
