from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .schema import ModelGenomeProgram


@dataclass(frozen=True)
class ProgramPolicy:
    primary_fraction: float = 0.10
    exploratory_fraction: float = 0.20
    max_vector_values: int = 4096
    max_sparse_fraction: float = 0.001
    minimum_development_progress: float = 0.80


@dataclass(frozen=True)
class ProgramAudit:
    accepted_structure: bool
    serialized: bool
    direct_fp16_delta_bytes: int
    target_specific_bytes: int | None
    byte_fraction: float | None
    primary_budget_pass: bool
    exploratory_budget_pass: bool
    reasons: tuple[str, ...]

    @property
    def eligible_for_function_gate(self) -> bool:
        return self.accepted_structure and self.serialized and self.exploratory_budget_pass


def audit_program(
    program: ModelGenomeProgram,
    payloads: Mapping[str, torch.Tensor],
    *,
    direct_fp16_delta_bytes: int,
    artifact_directory: str | Path | None = None,
    policy: ProgramPolicy = ProgramPolicy(),
) -> ProgramAudit:
    reasons: list[str] = []
    payload_use: set[str] = set()
    total_values = sum(int(torch.tensor(item.shape).prod().item()) for item in program.tensors)
    sparse_values = 0
    for tensor in program.tensors:
        for component in tensor.components:
            payload_use.update(component.payload.values())
            if component.primitive == "QUANTIZED_VECTOR":
                if len(tensor.shape) != 1:
                    reasons.append(f"dense_matrix_disguised_as_vector:{tensor.name}")
                if tensor.shape[0] > policy.max_vector_values:
                    reasons.append(f"vector_too_large:{tensor.name}")
            if component.primitive == "DIRECT_VECTOR":
                values = payloads.get(component.payload.get("values", ""))
                if len(tensor.shape) != 1:
                    reasons.append(f"direct_vector_on_non_vector:{tensor.name}")
                elif tensor.shape[0] > policy.max_vector_values:
                    reasons.append(f"vector_too_large:{tensor.name}")
                elif values is None:
                    reasons.append(f"missing_direct_vector_payload:{tensor.name}")
                elif (
                    values.dtype != torch.float16
                    or values.ndim != 1
                    or values.numel() != tensor.shape[0]
                ):
                    reasons.append(f"invalid_direct_vector_payload:{tensor.name}")
            if component.primitive == "SPARSE_PATCH":
                key = component.payload.get("indices")
                if key in payloads:
                    sparse_values += payloads[key].numel()
            if component.primitive == "LOW_RANK":
                left = payloads.get(component.payload.get("left", ""))
                right = payloads.get(component.payload.get("right", ""))
                if left is None or right is None:
                    reasons.append(f"missing_low_rank_payload:{tensor.name}")
                elif left.numel() + right.numel() >= tensor.shape[0] * tensor.shape[1]:
                    reasons.append(f"non_compact_low_rank:{tensor.name}")
            if component.primitive == "HADAMARD_SCALE":
                row = payloads.get(component.payload.get("row", ""))
                column = payloads.get(component.payload.get("column", ""))
                if len(tensor.shape) != 2:
                    reasons.append(f"matrix_scale_on_non_matrix:{tensor.name}")
                elif row is None or column is None:
                    reasons.append(f"missing_matrix_scale_payload:{tensor.name}")
                elif (
                    row.ndim != 1
                    or column.ndim != 1
                    or row.numel() != tensor.shape[0]
                    or column.numel() != tensor.shape[1]
                ):
                    reasons.append(f"invalid_matrix_scale_payload:{tensor.name}")
                elif row.numel() + column.numel() >= tensor.shape[0] * tensor.shape[1]:
                    reasons.append(f"non_compact_matrix_scale:{tensor.name}")
    unused = set(payloads) - payload_use
    if unused:
        reasons.append(f"unused_payloads:{','.join(sorted(unused))}")
    if total_values and sparse_values / total_values > policy.max_sparse_fraction:
        reasons.append("sparse_patch_budget_exceeded")
    serialized = artifact_directory is not None
    target_bytes: int | None = None
    fraction: float | None = None
    if serialized:
        root = Path(artifact_directory)
        required = (root / "manifest.json", root / "payload.safetensors")
        if not all(path.is_file() for path in required):
            reasons.append("missing_serialized_mgp")
        else:
            target_bytes = sum(path.stat().st_size for path in required)
            fraction = target_bytes / direct_fp16_delta_bytes
    primary = fraction is not None and fraction <= policy.primary_fraction
    exploratory = fraction is not None and fraction <= policy.exploratory_fraction
    return ProgramAudit(
        accepted_structure=not reasons,
        serialized=serialized and target_bytes is not None,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes,
        target_specific_bytes=target_bytes,
        byte_fraction=fraction,
        primary_budget_pass=primary,
        exploratory_budget_pass=exploratory,
        reasons=tuple(reasons),
    )
