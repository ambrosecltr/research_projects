from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..hashing import canonical_json_bytes
from ..types import GenomeProgram, TensorSpec
from .opcodes import (
    DENSE_DELTA,
    NEURAL_BLOCK_FIELD,
    QUANTIZED_DELTA,
    SPARSE_PATCH,
)
from .validation import validate_program


@dataclass(frozen=True)
class CompilerTargetPolicy:
    """Hard boundary between a model program and disguised endpoint storage."""

    max_target_fraction_of_fp16_delta: float = 0.10
    exploratory_max_fraction_of_fp16_delta: float = 0.25
    max_sparse_fraction: float = 0.01
    direct_tensor_numel_limit: int = 4096
    max_direct_fraction_of_total_numel: float = 0.01
    max_neural_code_values_per_weight: float = 0.05
    allow_exploratory_band: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_target_fraction_of_fp16_delta",
            "exploratory_max_fraction_of_fp16_delta",
            "max_sparse_fraction",
            "max_direct_fraction_of_total_numel",
            "max_neural_code_values_per_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_target_fraction_of_fp16_delta > self.exploratory_max_fraction_of_fp16_delta:
            raise ValueError("primary target fraction cannot exceed exploratory target fraction")
        if (
            isinstance(self.direct_tensor_numel_limit, bool)
            or not isinstance(self.direct_tensor_numel_limit, int)
            or self.direct_tensor_numel_limit < 0
        ):
            raise ValueError("direct_tensor_numel_limit must be a non-negative integer")

    @property
    def active_fraction_limit(self) -> float:
        return (
            self.exploratory_max_fraction_of_fp16_delta
            if self.allow_exploratory_band
            else self.max_target_fraction_of_fp16_delta
        )


@dataclass(frozen=True)
class CompilerTargetAudit:
    accepted: bool
    target_specific_bytes: int
    shared_bytes: int
    fp16_delta_bytes: int
    target_fraction: float
    direct_value_numel: int
    total_numel: int
    sparse_nnz: int
    neural_code_values: int
    failure_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_codes"] = list(self.failure_codes)
        data["warnings"] = list(self.warnings)
        return data


def _logical_bytes(program: GenomeProgram, keys: set[str]) -> int:
    all_tensors = {**program.payload_tensors, **program.patch_tensors}
    return sum(all_tensors[key].numel() * all_tensors[key].element_size() for key in keys)


def audit_compiler_target(
    program: GenomeProgram,
    inventory: Sequence[TensorSpec],
    *,
    fp16_delta_bytes: int,
    policy: CompilerTargetPolicy | None = None,
    actual_mgp_bytes: int | None = None,
) -> CompilerTargetAudit:
    """Reject target labels that merely hide WT in dense or per-weight residual payloads."""

    policy = policy or CompilerTargetPolicy()
    validate_program(program, inventory)
    if isinstance(fp16_delta_bytes, bool) or not isinstance(fp16_delta_bytes, int) or fp16_delta_bytes < 1:
        raise ValueError("fp16_delta_bytes must be a positive integer")
    if actual_mgp_bytes is not None and (
        isinstance(actual_mgp_bytes, bool) or not isinstance(actual_mgp_bytes, int) or actual_mgp_bytes < 1
    ):
        raise ValueError("actual_mgp_bytes must be a positive integer when supplied")

    shared_keys_raw = program.manifest.get("shared_payload_keys", [])
    if not isinstance(shared_keys_raw, list) or any(not isinstance(key, str) for key in shared_keys_raw):
        raise ValueError("shared_payload_keys must be an array of strings")
    shared_keys = set(shared_keys_raw)
    all_keys = set(program.payload_tensors) | set(program.patch_tensors)
    if not shared_keys.issubset(all_keys):
        raise ValueError("shared payload declaration references missing tensors")
    target_keys = all_keys - shared_keys

    manifest_bytes = len(
        canonical_json_bytes(
            {
                **program.manifest,
                "records": [record.to_dict() for record in program.records],
            }
        )
    )
    target_specific_bytes = (
        actual_mgp_bytes
        if actual_mgp_bytes is not None
        else manifest_bytes + _logical_bytes(program, target_keys)
    )
    shared_bytes = _logical_bytes(program, shared_keys)

    by_name = {spec.name: spec for spec in inventory}
    total_numel = sum(spec.numel for spec in inventory if not spec.is_buffer)
    direct_value_numel = 0
    sparse_nnz = 0
    neural_code_values = 0
    failures: list[str] = []
    warnings: list[str] = []

    decoder_config = program.manifest.get("codec_config", {})
    if isinstance(decoder_config, dict):
        nested_decoder = decoder_config.get("decoder", decoder_config)
        if isinstance(nested_decoder, dict) and nested_decoder.get("block_code_mode") == "residual":
            failures.append("full_residual_block_mode")

    if program.manifest.get("exact_residual") or program.manifest.get("contains_exact_residual"):
        failures.append("exact_residual_declared")

    for record in program.records:
        spec = by_name[record.tensor_name]
        for component in record.components:
            if component.opcode in {DENSE_DELTA, QUANTIZED_DELTA}:
                direct_value_numel += spec.numel
                if spec.numel > policy.direct_tensor_numel_limit:
                    failures.append(f"direct_large_tensor:{record.tensor_name}")
            elif component.opcode == SPARSE_PATCH:
                raw_nnz = component.arguments.get("nnz")
                if isinstance(raw_nnz, bool) or not isinstance(raw_nnz, int) or raw_nnz < 0:
                    failures.append(f"invalid_sparse_count:{record.tensor_name}")
                else:
                    sparse_nnz += raw_nnz
                    if raw_nnz > max(1, int(spec.numel * policy.max_sparse_fraction)):
                        failures.append(f"oversized_sparse_patch:{record.tensor_name}")
            elif component.opcode == NEURAL_BLOCK_FIELD:
                block_key = component.arguments.get("block_codes_key")
                if isinstance(block_key, str) and block_key in program.payload_tensors:
                    values = program.payload_tensors[block_key].numel()
                    neural_code_values += values
                    if values > max(1, int(spec.numel * policy.max_neural_code_values_per_weight)):
                        failures.append(f"per_weight_neural_code:{record.tensor_name}")

    direct_fraction = direct_value_numel / max(total_numel, 1)
    if direct_fraction > policy.max_direct_fraction_of_total_numel:
        failures.append("too_many_direct_values")
    if sparse_nnz > max(1, int(total_numel * policy.max_sparse_fraction)):
        failures.append("too_many_sparse_values")
    if neural_code_values > max(1, int(total_numel * policy.max_neural_code_values_per_weight)):
        failures.append("too_many_neural_code_values")

    target_fraction = target_specific_bytes / fp16_delta_bytes
    if target_fraction > policy.active_fraction_limit:
        failures.append("target_program_exceeds_byte_budget")
    elif target_fraction > policy.max_target_fraction_of_fp16_delta:
        warnings.append("exploratory_byte_band_only")

    if target_specific_bytes >= fp16_delta_bytes:
        failures.append("not_smaller_than_direct_fp16_delta")

    return CompilerTargetAudit(
        accepted=not failures,
        target_specific_bytes=target_specific_bytes,
        shared_bytes=shared_bytes,
        fp16_delta_bytes=fp16_delta_bytes,
        target_fraction=target_fraction,
        direct_value_numel=direct_value_numel,
        total_numel=total_numel,
        sparse_nnz=sparse_nnz,
        neural_code_values=neural_code_values,
        failure_codes=tuple(sorted(set(failures))),
        warnings=tuple(sorted(set(warnings))),
        policy=asdict(policy),
    )


def assert_compiler_target(
    program: GenomeProgram,
    inventory: Sequence[TensorSpec],
    *,
    fp16_delta_bytes: int,
    policy: CompilerTargetPolicy | None = None,
    actual_mgp_bytes: int | None = None,
) -> CompilerTargetAudit:
    audit = audit_compiler_target(
        program,
        inventory,
        fp16_delta_bytes=fp16_delta_bytes,
        policy=policy,
        actual_mgp_bytes=actual_mgp_bytes,
    )
    if not audit.accepted:
        raise ValueError(
            "MGP is not a valid compact compiler target: " + ", ".join(audit.failure_codes)
        )
    return audit
