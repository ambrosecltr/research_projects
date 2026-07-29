from __future__ import annotations

import torch

from genome.mgp.interpreter import decode_program
from genome.mgp.opcodes import COPY_FROM_TIED, LOW_RANK, QUANTIZED_DELTA
from genome.program_compiler import CompilerConditioning, ProgramCompilerConfig, VariableProgramCompiler
from genome.program_grammar import generate_valid_program
from genome.types import TensorSpec


def compiler_config() -> ProgramCompilerConfig:
    return ProgramCompilerConfig(
        global_feature_dim=3,
        semantic_feature_dim=5,
        stage_feature_dim=4,
        tensor_feature_dim=6,
        stage_type_count=4,
        tensor_role_count=6,
        model_dim=32,
        feedforward_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        coefficient_chunk_dim=4,
        graph_message_layers=1,
        max_program_tokens=128,
    )


def conditioning(tensor_count: int) -> CompilerConditioning:
    generator = torch.Generator().manual_seed(41)
    adjacency = torch.eye(tensor_count, dtype=torch.bool).unsqueeze(0)
    if tensor_count > 1:
        adjacency[0, 0, 1] = True
        adjacency[0, 1, 0] = True
    return CompilerConditioning(
        global_features=torch.randn(1, 3, generator=generator),
        semantic_features=torch.randn(1, 5, generator=generator),
        stage_features=torch.randn(1, 2, 4, generator=generator),
        stage_type_ids=torch.tensor([[0, 1]]),
        stage_mask=torch.tensor([[True, True]]),
        tensor_features=torch.randn(1, tensor_count, 6, generator=generator),
        tensor_role_ids=torch.arange(tensor_count).unsqueeze(0),
        tensor_mask=torch.ones(1, tensor_count, dtype=torch.bool),
        tensor_adjacency=adjacency,
    )


def inventory() -> list[TensorSpec]:
    return [
        TensorSpec(
            canonical_index=0,
            name="matrix",
            role="attention_q",
            layer_index=0,
            shape=(4, 4),
            dtype="float32",
            numel=16,
            nbytes=64,
        ),
        TensorSpec(
            canonical_index=1,
            name="norm",
            role="norm_scale",
            layer_index=0,
            shape=(4,),
            dtype="float32",
            numel=4,
            nbytes=16,
        ),
    ]


def test_random_untrained_compiler_still_emits_parseable_grammar() -> None:
    torch.manual_seed(5)
    model = VariableProgramCompiler(compiler_config()).eval()
    generated = generate_valid_program(
        model,
        conditioning(2),
        inventory(),
        max_tokens=128,
        candidate_id="grammar-smoke",
    )
    allowed = {LOW_RANK, QUANTIZED_DELTA, COPY_FROM_TIED}
    assert all(
        component.opcode in allowed
        for record in generated.program.records
        for component in record.components
    )
    assert generated.program.manifest["created_unix"] == 0.0

    decoded = decode_program(
        generated.program,
        {"matrix": torch.zeros(4, 4), "norm": torch.ones(4)},
        inventory(),
    )
    assert set(decoded) == {"matrix", "norm"}
    assert all(torch.isfinite(value).all() for value in decoded.values())


def test_tied_alias_is_forced_to_base_copy() -> None:
    specs = [
        TensorSpec(
            canonical_index=0,
            name="embedding",
            role="embedding",
            layer_index=None,
            shape=(4, 4),
            dtype="float32",
            numel=16,
            nbytes=64,
            tied_group="embedding-tie",
        ),
        TensorSpec(
            canonical_index=1,
            name="lm_head",
            role="lm_head",
            layer_index=None,
            shape=(4, 4),
            dtype="float32",
            numel=16,
            nbytes=64,
            tied_group="embedding-tie",
        ),
    ]
    torch.manual_seed(9)
    generated = generate_valid_program(
        VariableProgramCompiler(compiler_config()).eval(),
        conditioning(2),
        specs,
        tied_groups=(("embedding", "lm_head"),),
        max_tokens=128,
    )
    alias = generated.program.records[1]
    assert alias.tied_owner == "embedding"
    assert [component.opcode for component in alias.components] == [COPY_FROM_TIED]
