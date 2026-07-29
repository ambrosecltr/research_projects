from __future__ import annotations

import torch

from genome.compact_targets import CompactTargetConfig, canonicalize_svd, fit_compact_svd_target
from genome.mgp.opcodes import DENSE_DELTA, NEURAL_BLOCK_FIELD
from genome.mgp.policy import CompilerTargetPolicy, audit_compiler_target
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
