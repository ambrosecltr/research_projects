from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..state import compute_delta
from ..tensor_inventory import tied_owner_map
from ..types import TensorSpec


@dataclass(frozen=True)
class SVDFactorization:
    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor


@dataclass(frozen=True)
class SVDWorkspace:
    """Reusable Delta-T and exact matrix factorizations for one frozen specimen.

    A rate-distortion sweep must not pay for the same full SVD at every rank. Build this
    workspace once, charge its elapsed time once, and pass it to every SVD-family codec.
    """

    delta: Mapping[str, torch.Tensor]
    factors: Mapping[str, SVDFactorization]
    factorization_seconds: float

    @classmethod
    def build(
        cls,
        base_state: Mapping[str, torch.Tensor],
        target_state: Mapping[str, torch.Tensor],
        tensor_specs: Sequence[TensorSpec],
        *,
        tied_groups: Sequence[Sequence[str]] = (),
    ) -> SVDWorkspace:
        delta = compute_delta(base_state, target_state, tensor_specs)
        aliases = tied_owner_map(tied_groups)
        factors: dict[str, SVDFactorization] = {}
        started = time.perf_counter()
        for spec in tensor_specs:
            if spec.name in aliases or len(spec.shape) != 2:
                continue
            matrix = delta[spec.name]
            u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
            factors[spec.name] = SVDFactorization(
                u=u.contiguous(),
                s=s.contiguous(),
                vh=vh.contiguous(),
            )
        elapsed = time.perf_counter() - started
        workspace = cls(delta=delta, factors=factors, factorization_seconds=elapsed)
        workspace.validate(tensor_specs, tied_groups=tied_groups)
        return workspace

    def validate(
        self,
        tensor_specs: Sequence[TensorSpec],
        *,
        tied_groups: Sequence[Sequence[str]] = (),
    ) -> None:
        expected_names = {spec.name for spec in tensor_specs}
        if set(self.delta) != expected_names:
            missing = sorted(expected_names - set(self.delta))
            extra = sorted(set(self.delta) - expected_names)
            raise ValueError(f"SVD workspace delta mismatch; missing={missing}, extra={extra}")
        for spec in tensor_specs:
            tensor = self.delta[spec.name]
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(f"SVD workspace delta shape mismatch for {spec.name}")
            if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise ValueError("SVD workspace deltas must be CPU float32 tensors")

        aliases = tied_owner_map(tied_groups)
        expected_matrices = {
            spec.name
            for spec in tensor_specs
            if spec.name not in aliases and len(spec.shape) == 2
        }
        if set(self.factors) != expected_matrices:
            missing = sorted(expected_matrices - set(self.factors))
            extra = sorted(set(self.factors) - expected_matrices)
            raise ValueError(f"SVD workspace factor mismatch; missing={missing}, extra={extra}")
        by_name = {spec.name: spec for spec in tensor_specs}
        for name, factor in self.factors.items():
            rows, columns = by_name[name].shape
            rank = min(rows, columns)
            if (
                tuple(factor.u.shape) != (rows, rank)
                or tuple(factor.s.shape) != (rank,)
                or tuple(factor.vh.shape) != (rank, columns)
            ):
                raise ValueError(f"SVD workspace factor shape mismatch for {name}")
            for tensor in (factor.u, factor.s, factor.vh):
                if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                    raise ValueError("SVD workspace factors must be CPU float32 tensors")
                if not bool(torch.isfinite(tensor).all().item()):
                    raise ValueError(f"SVD workspace factor contains NaN/Inf: {name}")

    @property
    def matrix_count(self) -> int:
        return len(self.factors)
