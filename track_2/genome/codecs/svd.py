from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..state import compute_delta, state_num_bytes
from ..types import GenomeBudget, GenomeComponent, GenomeProgram, TensorSpec
from ..mgp.opcodes import COPY_FROM_TIED, DENSE_DELTA, LOW_RANK, QUANTIZED_DELTA
from .base import GenomeCodec
from .common import make_manifest, make_records
from .quantized import pack_int4
from .workspace import SVDFactorization, SVDWorkspace


def _factorize(matrix: torch.Tensor) -> SVDFactorization:
    if matrix.ndim != 2:
        raise ValueError("SVD factorization expects a matrix")
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    return SVDFactorization(u=u, s=s, vh=vh)


class SVDCodec(GenomeCodec):
    name = "svd"

    def __init__(
        self,
        rank: int | None = 8,
        *,
        factor_dtype: torch.dtype = torch.float32,
        vector_bits: int = 8,
        candidate_id: str | None = None,
        workspace: SVDWorkspace | None = None,
    ) -> None:
        if rank is not None and rank < 0:
            raise ValueError("rank must be non-negative")
        if vector_bits not in {4, 8, 32}:
            raise ValueError("vector_bits must be 4, 8, or 32")
        self.rank = rank
        self.factor_dtype = factor_dtype
        self.vector_bits = vector_bits
        self.candidate_id = candidate_id or f"g0_svd_r{rank if rank is not None else 'budget'}"
        self.workspace = workspace

    def _rank_allocation(
        self,
        delta: Mapping[str, torch.Tensor],
        specs: Sequence[TensorSpec],
        budget: GenomeBudget | None,
    ) -> tuple[dict[str, int], dict[str, SVDFactorization]]:
        matrices = [spec for spec in specs if len(spec.shape) == 2]
        factors = (
            {spec.name: self.workspace.factors[spec.name] for spec in matrices}
            if self.workspace is not None
            else {spec.name: _factorize(delta[spec.name]) for spec in matrices}
        )
        if self.rank is not None:
            return {
                spec.name: min(self.rank, factors[spec.name].s.numel()) for spec in matrices
            }, factors

        raw_bytes = state_num_bytes(delta)
        budget_bytes = None if budget is None else budget.resolve(raw_bytes)
        if budget_bytes is None:
            raise ValueError("rank=None requires a byte budget")
        element_size = torch.empty((), dtype=self.factor_dtype).element_size()
        candidates: list[tuple[float, str, int, int]] = []
        for spec in matrices:
            factor = factors[spec.name]
            per_component = (spec.shape[0] + spec.shape[1] + 1) * element_size
            for index, sigma in enumerate(factor.s):
                score = float(sigma.square().item()) / max(per_component, 1)
                candidates.append((score, spec.name, index, per_component))
        candidates.sort(key=lambda item: item[0], reverse=True)
        ranks = {spec.name: 0 for spec in matrices}
        used = 0
        for _, name, index, cost in candidates:
            if used + cost > budget_bytes:
                continue
            # Singular values are sorted, so selecting index implies all earlier components have
            # at least the same value-per-byte and should already be selected.
            if index == ranks[name]:
                ranks[name] += 1
                used += cost
        return ranks, factors

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
        if self.workspace is not None:
            self.workspace.validate(tensor_specs, tied_groups=tied_groups)
            delta = self.workspace.delta
        else:
            delta = compute_delta(base_state, target_state, tensor_specs)
        records, aliases = make_records(tensor_specs, tied_groups)
        unique_specs = [spec for spec in tensor_specs if spec.name not in aliases]
        ranks, factors = self._rank_allocation(delta, unique_specs, budget)
        payload: dict[str, torch.Tensor] = {}

        for record in records:
            name = record.tensor_name
            if name in aliases:
                record.components.append(
                    GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[name]})
                )
                continue
            source = delta[name]
            if source.ndim == 2:
                rank = ranks.get(name, 0)
                if rank == 0:
                    continue
                factor = factors[name]
                prefix = f"t{record.canonical_index:05d}.svd"
                keys = [f"{prefix}.u", f"{prefix}.s", f"{prefix}.vh"]
                payload[keys[0]] = factor.u[:, :rank].to(self.factor_dtype).contiguous()
                payload[keys[1]] = factor.s[:rank].to(self.factor_dtype).contiguous()
                payload[keys[2]] = factor.vh[:rank, :].to(self.factor_dtype).contiguous()
                record.components.append(
                    GenomeComponent(
                        LOW_RANK,
                        payload_keys=keys,
                        arguments={
                            "rank": rank,
                            "factor_dtype": str(self.factor_dtype).replace("torch.", ""),
                        },
                    )
                )
            elif self.vector_bits == 32:
                key = f"t{record.canonical_index:05d}.dense_delta"
                payload[key] = source.to(torch.float32).contiguous()
                record.components.append(GenomeComponent(DENSE_DELTA, payload_keys=[key]))
            else:
                qmax = 127 if self.vector_bits == 8 else 7
                absmax = float(source.abs().max().item()) if source.numel() else 0.0
                scale = absmax / qmax if absmax > 0 else 1.0
                q = torch.round(source / scale).clamp(-qmax, qmax).to(torch.int8)
                q_key = f"t{record.canonical_index:05d}.q{self.vector_bits}"
                s_key = f"t{record.canonical_index:05d}.scale"
                payload[q_key] = q if self.vector_bits == 8 else pack_int4(q)
                payload[s_key] = torch.tensor(scale, dtype=torch.float32)
                record.components.append(
                    GenomeComponent(
                        QUANTIZED_DELTA,
                        payload_keys=[q_key, s_key],
                        arguments={"bits": self.vector_bits, "shape": list(record.shape)},
                    )
                )

        manifest = make_manifest(candidate_id=self.candidate_id, codec=self.name, metadata=manifest_metadata)
        manifest["codec_config"] = {
            "rank": self.rank,
            "factor_dtype": str(self.factor_dtype).replace("torch.", ""),
            "vector_bits": self.vector_bits,
            "allocated_ranks": ranks,
            "budget": None if budget is None else budget.__dict__,
        }
        return GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)
