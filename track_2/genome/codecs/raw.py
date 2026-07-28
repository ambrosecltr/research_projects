from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..hashing import sha256_tensor
from ..state import compute_delta
from ..types import GenomeBudget, GenomeComponent, GenomeProgram, TensorSpec
from ..mgp.opcodes import COPY_FROM_TIED, DENSE_DELTA
from .base import GenomeCodec
from .common import make_manifest, make_records


class DenseDeltaCodec(GenomeCodec):
    name = "dense_delta"

    def __init__(self, *, payload_dtype: torch.dtype = torch.float64, candidate_id: str = "g0_dense"):
        self.payload_dtype = payload_dtype
        self.candidate_id = candidate_id

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
        delta = compute_delta(base_state, target_state, tensor_specs, dtype=self.payload_dtype)
        records, aliases = make_records(tensor_specs, tied_groups)
        payload: dict[str, torch.Tensor] = {}
        for record in records:
            if record.tensor_name in aliases:
                record.components.append(
                    GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[record.tensor_name]})
                )
                continue
            key = f"t{record.canonical_index:05d}.dense_delta"
            payload[key] = delta[record.tensor_name].to(self.payload_dtype).contiguous()
            record.components.append(
                GenomeComponent(
                    DENSE_DELTA,
                    payload_keys=[key],
                    arguments={"payload_dtype": str(self.payload_dtype).replace("torch.", "")},
                )
            )
            # Exact for common floating target dtypes when delta is stored in fp32.
            if self.payload_dtype in {torch.float32, torch.float64}:
                reconstructed = (
                    base_state[record.tensor_name].to(self.payload_dtype) + payload[key].to(self.payload_dtype)
                ).to(target_state[record.tensor_name].dtype)
                if torch.equal(reconstructed, target_state[record.tensor_name]):
                    record.output_checksum = sha256_tensor(reconstructed)
        manifest = make_manifest(
            candidate_id=self.candidate_id,
            codec=self.name,
            metadata=manifest_metadata,
        )
        manifest["codec_config"] = {
            "payload_dtype": str(self.payload_dtype).replace("torch.", "")
        }
        return GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)
