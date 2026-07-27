from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..state import compute_delta
from ..types import GenomeBudget, GenomeComponent, GenomeProgram, TensorSpec
from ..mgp.opcodes import COPY_FROM_TIED, DENSE_DELTA, QUANTIZED_DELTA
from .base import GenomeCodec
from .common import make_manifest, make_records


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    flat = values.to(torch.int8).flatten()
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
    encoded = torch.bitwise_and(flat.to(torch.int16) + 8, 0x0F).to(torch.uint8)
    low = encoded[0::2]
    high = torch.bitwise_left_shift(encoded[1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


class QuantizedDeltaCodec(GenomeCodec):
    name = "quantized_delta"

    def __init__(self, bits: int = 8, *, candidate_id: str | None = None):
        if bits not in {4, 8}:
            raise ValueError("bits must be 4 or 8")
        self.bits = bits
        self.candidate_id = candidate_id or f"g0_int{bits}"

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
        delta = compute_delta(base_state, target_state, tensor_specs)
        records, aliases = make_records(tensor_specs, tied_groups)
        payload: dict[str, torch.Tensor] = {}
        qmax = 127 if self.bits == 8 else 7
        for record in records:
            if record.tensor_name in aliases:
                record.components.append(
                    GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[record.tensor_name]})
                )
                continue
            source = delta[record.tensor_name]
            if not source.is_floating_point():
                key = f"t{record.canonical_index:05d}.dense_delta"
                payload[key] = source.to(torch.float32)
                record.components.append(GenomeComponent(DENSE_DELTA, payload_keys=[key]))
                continue
            absmax = float(source.abs().max().item()) if source.numel() else 0.0
            scale_value = absmax / qmax if absmax > 0 else 1.0
            quantized = torch.round(source / scale_value).clamp(-qmax, qmax).to(torch.int8)
            q_key = f"t{record.canonical_index:05d}.q{self.bits}"
            scale_key = f"t{record.canonical_index:05d}.scale"
            payload[q_key] = quantized if self.bits == 8 else pack_int4(quantized)
            payload[scale_key] = torch.tensor(scale_value, dtype=torch.float32)
            record.components.append(
                GenomeComponent(
                    QUANTIZED_DELTA,
                    payload_keys=[q_key, scale_key],
                    arguments={
                        "bits": self.bits,
                        "shape": list(record.shape),
                        "scheme": "symmetric_per_tensor",
                        "qmax": qmax,
                    },
                )
            )
        manifest = make_manifest(
            candidate_id=self.candidate_id, codec=f"{self.name}_int{self.bits}", metadata=manifest_metadata
        )
        manifest["codec_config"] = {"bits": self.bits, "scheme": "symmetric_per_tensor"}
        return GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)
