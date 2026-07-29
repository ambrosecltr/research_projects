from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TensorNode:
    index: int
    name: str
    role: str
    layer: int | None
    shape: tuple[int, ...]
    dtype: str
    tied_to: str | None = None

    @property
    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    @property
    def is_matrix(self) -> bool:
        return len(self.shape) == 2

    @property
    def is_vector(self) -> bool:
        return len(self.shape) == 1
