from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import torch

from ..types import GenomeBudget, GenomeProgram, TensorSpec


class GenomeCodec(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        base_state: Mapping[str, torch.Tensor],
        target_state: Mapping[str, torch.Tensor],
        tensor_specs: Sequence[TensorSpec],
        *,
        tied_groups: Sequence[Sequence[str]] = (),
        budget: GenomeBudget | None = None,
        manifest_metadata: Mapping | None = None,
    ) -> GenomeProgram:
        raise NotImplementedError
