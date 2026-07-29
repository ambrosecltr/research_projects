from __future__ import annotations

from pathlib import Path

import torch

from genome.compact_targets import (
    CompactTargetConfig,
    CompactTargetResult,
    canonicalize_svd,
    fit_compact_svd_target,
    serialize_and_audit_compiler_target,
)
from genome.hashing import sha256_directory, sha256_file
from genome.io import read_json
from genome.mgp.opcodes import (
    DENSE_DELTA,
    LOW_RANK,
    NEURAL_BLOCK_FIELD,
    QUANTIZED_DELTA,
    SPARSE_PATCH,
)
from genome.mgp.policy import CompilerTargetPolicy, audit_compiler_target
from genome.mgp.serializer import save_program
from genome.types import GenomeComponent, GenomeProgram, TensorGenomeRecord, TensorSpec


def matrix_spec(size: int) -> TensorSpec:
    return TensorSpec(
        canonical_index=0,
        name="weight",
        role="attention_q",
        layer_index=0,
        shape=(size, size),
        dtype="float32",
        numel=size * size,
        nbytes=size * size * 4,
    )


def test_canonical_svd_removes_sign_ambiguity() -> None:
    matrix = torch.Generator().manual_seed(7)
    value = torch.randn(16, 8, generator=matrix)
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    first = canonicalize_svd(u, s, vh)
    second = canonicalize_svd(-u, s, -vh)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_rank_one_target_is_compact_and_decodes_without_residual() -> None:
    size = 256
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(size, 1, generator=generator)
    right = torch.randn(1, size, generator=generator)
    base = {"weight": torch.randn(size, size, generator=generator) * 0.01}
    target = {"weight": base["weight"] + left @ right}
    result = fit_compact_svd_target(
        base,
        target,
        [matrix_spec(size)],
        config=CompactTargetConfig(
            target_fraction_of_fp16_delta=0.10,
            max_rank=1,
            direct_vector_numel_limit=0,
        ),
    )
    assert result.audit.accepted
    assert result.audit.target_fraction < 0.10
    assert result.allocated_ranks == {"weight": 1}
    assert result.program.manifest["created_unix"] == 0.0
    assert result.program.manifest["contains_exact_residual"] is False
    assert all(
        component.opcode != DENSE_DELTA
        for record in result.program.records
        for component in record.components
    )


def test_v4_style_per_weight_residual_is_rejected() -> None:
    spec = matrix_spec(16)
    record = TensorGenomeRecord(
        tensor_name="weight",
        canonical_index=0,
        role=spec.role,
        layer_index=0,
        shape=spec.shape,
        output_dtype="float32",
        components=[
            GenomeComponent(
                NEURAL_BLOCK_FIELD,
                payload_keys=["tensor_code", "block_codes"],
                arguments={"block_codes_key": "block_codes"},
            )
        ],
    )
    program = GenomeProgram(
        manifest={
            "format": "MGP",
            "version": "0.1.0",
            "candidate_id": "residual-disguise",
            "codec_config": {
                "decoder": {
                    "block_code_mode": "residual",
                    "block_rows": 16,
                    "block_cols": 16,
                    "block_code_dim": 256,
                }
            },
        },
        records=[record],
        payload_tensors={
            "tensor_code": torch.zeros(32),
            "block_codes": torch.zeros(1, 256, dtype=torch.float16),
        },
    )
    audit = audit_compiler_target(
        program,
        [spec],
        fp16_delta_bytes=spec.numel * 2,
        policy=CompilerTargetPolicy(
            max_target_fraction_of_fp16_delta=1.0,
            exploratory_max_fraction_of_fp16_delta=1.0,
        ),
    )
    assert not audit.accepted
    assert "full_residual_block_mode" in audit.failure_codes
    assert "per_weight_block_code_width" in audit.failure_codes
    assert any(code.startswith("per_weight_neural_code") for code in audit.failure_codes)


def test_full_dense_delta_is_not_a_compiler_target() -> None:
    spec = matrix_spec(128)
    program = GenomeProgram(
        manifest={"format": "MGP", "version": "0.1.0", "candidate_id": "dense"},
        records=[
            TensorGenomeRecord(
                tensor_name="weight",
                canonical_index=0,
                role=spec.role,
                layer_index=0,
                shape=spec.shape,
                output_dtype="float32",
                components=[GenomeComponent(DENSE_DELTA, payload_keys=["delta"])],
            )
        ],
        payload_tensors={"delta": torch.zeros(spec.shape)},
    )
    audit = audit_compiler_target(
        program,
        [spec],
        fp16_delta_bytes=spec.numel * 2,
        policy=CompilerTargetPolicy(
            max_target_fraction_of_fp16_delta=1.0,
            exploratory_max_fraction_of_fp16_delta=1.0,
        ),
    )
    assert not audit.accepted
    assert "direct_large_tensor:weight" in audit.failure_codes
    assert "too_many_direct_values" in audit.failure_codes


def rank_one_target(size: int = 256) -> tuple[CompactTargetResult, TensorSpec]:
    generator = torch.Generator().manual_seed(29)
    left = torch.randn(size, 1, generator=generator)
    right = torch.randn(1, size, generator=generator)
    base = {"weight": torch.randn(size, size, generator=generator) * 0.01}
    target = {"weight": base["weight"] + left @ right}
    spec = matrix_spec(size)
    result = fit_compact_svd_target(
        base,
        target,
        [spec],
        config=CompactTargetConfig(
            target_fraction_of_fp16_delta=0.10,
            max_rank=1,
            direct_vector_numel_limit=0,
        ),
        candidate_id="stable-rank-one",
    )
    return result, spec


def test_serialized_overhead_can_reject_a_logical_budget_pass(tmp_path: Path) -> None:
    result, spec = rank_one_target()
    serialized = serialize_and_audit_compiler_target(
        result,
        [spec],
        tmp_path / "candidate.mgp",
    )
    assert serialized.audit.serialized_policy_ready
    assert serialized.audit.byte_accounting == "serialized"
    assert serialized.audit.target_specific_bytes > result.audit.target_specific_bytes

    midpoint = (
        result.audit.target_specific_bytes + serialized.audit.target_specific_bytes
    ) / (2 * result.audit.fp16_delta_bytes)
    strict_policy = CompilerTargetPolicy(
        max_target_fraction_of_fp16_delta=midpoint,
        exploratory_max_fraction_of_fp16_delta=midpoint,
    )
    logical = audit_compiler_target(
        result.program,
        [spec],
        fp16_delta_bytes=result.audit.fp16_delta_bytes,
        policy=strict_policy,
    )
    actual = audit_compiler_target(
        result.program,
        [spec],
        fp16_delta_bytes=result.audit.fp16_delta_bytes,
        policy=strict_policy,
        actual_mgp_bytes=serialized.artifact_sizes["mgp_bytes"],
    )
    assert logical.accepted
    assert not logical.serialized_policy_ready
    assert not actual.accepted
    assert "target_program_exceeds_byte_budget" in actual.failure_codes


def test_repeated_target_fitting_has_stable_serialized_identity(tmp_path: Path) -> None:
    first, spec = rank_one_target()
    repeated, _ = rank_one_target()
    first_path = tmp_path / "first.mgp"
    repeated_path = tmp_path / "repeated.mgp"
    serialize_and_audit_compiler_target(first, [spec], first_path)
    serialize_and_audit_compiler_target(repeated, [spec], repeated_path)
    assert sha256_file(first_path / "manifest.json") == sha256_file(
        repeated_path / "manifest.json"
    )
    assert sha256_file(first_path / "genome.safetensors") == sha256_file(
        repeated_path / "genome.safetensors"
    )
    assert sha256_directory(first_path) == sha256_directory(repeated_path)


def test_serialized_accounting_includes_all_mgp_files_and_payload_classes(
    tmp_path: Path,
) -> None:
    matrix = matrix_spec(256)
    vector = TensorSpec(
        canonical_index=1,
        name="norm",
        role="norm_scale",
        layer_index=0,
        shape=(16,),
        dtype="float32",
        numel=16,
        nbytes=64,
    )
    records = [
        TensorGenomeRecord(
            tensor_name="weight",
            canonical_index=0,
            role=matrix.role,
            layer_index=0,
            shape=matrix.shape,
            output_dtype="float32",
            components=[
                GenomeComponent(
                    LOW_RANK,
                    payload_keys=["u", "s", "vh"],
                    arguments={"rank": 1},
                ),
                GenomeComponent(
                    SPARSE_PATCH,
                    payload_keys=["indices", "patch_values"],
                    arguments={"nnz": 1},
                ),
            ],
        ),
        TensorGenomeRecord(
            tensor_name="norm",
            canonical_index=1,
            role=vector.role,
            layer_index=0,
            shape=vector.shape,
            output_dtype="float32",
            components=[
                GenomeComponent(
                    QUANTIZED_DELTA,
                    payload_keys=["quantized", "scale"],
                    arguments={"bits": 8, "shape": [16], "small_vector": True},
                )
            ],
        ),
    ]
    program = GenomeProgram(
        manifest={
            "format": "MGP",
            "version": "0.1.0",
            "candidate_id": "complete-accounting",
            "shared_payload_keys": ["u"],
        },
        records=records,
        payload_tensors={
            "u": torch.zeros(256, 1, dtype=torch.float16),
            "s": torch.zeros(1, dtype=torch.float16),
            "vh": torch.zeros(1, 256, dtype=torch.float16),
            "quantized": torch.zeros(16, dtype=torch.int8),
            "scale": torch.ones((), dtype=torch.float32),
        },
        patch_tensors={
            "indices": torch.zeros(1, dtype=torch.int64),
            "patch_values": torch.zeros(1, dtype=torch.float16),
        },
    )
    path = tmp_path / "accounting.mgp"
    sizes = save_program(program, path)
    manifest = read_json(path / "manifest.json")
    assert sizes["mgp_bytes"] == sum(
        (path / name).stat().st_size
        for name in ("manifest.json", "genome.safetensors", "patch.safetensors")
    )
    assert manifest["logical_payload_bytes"] == sum(
        value.numel() * value.element_size()
        for value in [*program.payload_tensors.values(), *program.patch_tensors.values()]
    )
    audit = audit_compiler_target(
        program,
        [matrix, vector],
        fp16_delta_bytes=(matrix.numel + vector.numel) * 2,
        actual_mgp_bytes=sizes["mgp_bytes"],
    )
    assert audit.target_specific_bytes == sizes["mgp_bytes"]
    assert audit.shared_bytes == program.payload_tensors["u"].numel() * 2
