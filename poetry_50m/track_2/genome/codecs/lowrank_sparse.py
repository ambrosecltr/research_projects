from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..state import compute_delta
from ..types import GenomeBudget, GenomeComponent, GenomeProgram, TensorSpec
from ..mgp.opcodes import COPY_FROM_TIED, DENSE_DELTA, LOW_RANK, SPARSE_PATCH
from .base import GenomeCodec
from .common import make_manifest, make_records
from .workspace import SVDWorkspace


class LowRankSparseCodec(GenomeCodec):
    name = "lowrank_sparse"

    def __init__(
        self,
        rank: int = 8,
        sparse_fraction: float = 0.001,
        *,
        factor_dtype: torch.dtype = torch.float32,
        patch_dtype: torch.dtype = torch.float32,
        candidate_id: str | None = None,
        workspace: SVDWorkspace | None = None,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if not 0.0 <= sparse_fraction <= 1.0:
            raise ValueError("sparse_fraction must be in [0, 1]")
        self.rank = rank
        self.sparse_fraction = sparse_fraction
        self.factor_dtype = factor_dtype
        self.patch_dtype = patch_dtype
        self.candidate_id = candidate_id or f"g0_svd_r{rank}_sp{int(sparse_fraction * 1e6)}ppm"
        self.workspace = workspace

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
        del budget
        if self.workspace is not None:
            self.workspace.validate(tensor_specs, tied_groups=tied_groups)
            delta = self.workspace.delta
        else:
            delta = compute_delta(base_state, target_state, tensor_specs)
        records, aliases = make_records(tensor_specs, tied_groups)
        payload: dict[str, torch.Tensor] = {}
        patch: dict[str, torch.Tensor] = {}
        for record in records:
            name = record.tensor_name
            if name in aliases:
                record.components.append(
                    GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[name]})
                )
                continue
            source = delta[name]
            approximation = torch.zeros_like(source)
            if source.ndim == 2 and self.rank > 0:
                if self.workspace is not None:
                    factor = self.workspace.factors[name]
                    u, s, vh = factor.u, factor.s, factor.vh
                else:
                    u, s, vh = torch.linalg.svd(source.to(torch.float32), full_matrices=False)
                rank = min(self.rank, s.numel())
                prefix = f"t{record.canonical_index:05d}.svd"
                keys = [f"{prefix}.u", f"{prefix}.s", f"{prefix}.vh"]
                payload[keys[0]] = u[:, :rank].to(self.factor_dtype).contiguous()
                payload[keys[1]] = s[:rank].to(self.factor_dtype).contiguous()
                payload[keys[2]] = vh[:rank].to(self.factor_dtype).contiguous()
                record.components.append(
                    GenomeComponent(LOW_RANK, payload_keys=keys, arguments={"rank": rank})
                )
                approximation = (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank]
            elif source.ndim != 2:
                # Vectors are cheap and often sensitive; keep them exact in the first hybrid.
                key = f"t{record.canonical_index:05d}.dense_delta"
                payload[key] = source.to(torch.float32).contiguous()
                record.components.append(GenomeComponent(DENSE_DELTA, payload_keys=[key]))
                approximation = source

            if source.ndim == 2 and self.sparse_fraction > 0:
                residual = (source - approximation).flatten()
                count = min(residual.numel(), max(1, int(round(residual.numel() * self.sparse_fraction))))
                if count > 0:
                    _, indices = torch.topk(residual.abs(), k=count, sorted=False)
                    indices = indices.sort().values
                    values = residual[indices]
                    index_dtype = torch.int32 if residual.numel() < 2**31 else torch.int64
                    i_key = f"t{record.canonical_index:05d}.sparse.indices"
                    v_key = f"t{record.canonical_index:05d}.sparse.values"
                    patch[i_key] = indices.to(index_dtype).contiguous()
                    patch[v_key] = values.to(self.patch_dtype).contiguous()
                    record.components.append(
                        GenomeComponent(
                            SPARSE_PATCH,
                            payload_keys=[i_key, v_key],
                            arguments={"nnz": count, "index_dtype": str(index_dtype).replace("torch.", "")},
                        )
                    )
        manifest = make_manifest(candidate_id=self.candidate_id, codec=self.name, metadata=manifest_metadata)
        manifest["codec_config"] = {
            "rank": self.rank,
            "sparse_fraction": self.sparse_fraction,
            "factor_dtype": str(self.factor_dtype).replace("torch.", ""),
            "patch_dtype": str(self.patch_dtype).replace("torch.", ""),
        }
        return GenomeProgram(
            manifest=manifest, records=records, payload_tensors=payload, patch_tensors=patch
        )
