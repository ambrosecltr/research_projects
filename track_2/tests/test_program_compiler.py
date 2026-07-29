from __future__ import annotations

import torch

from genome.program_compiler import (
    PROGRAM_TOKEN_TO_ID,
    CompilerConditioning,
    ProgramCompilerConfig,
    ProgramTeacherBatch,
    VariableProgramCompiler,
)


def config() -> ProgramCompilerConfig:
    return ProgramCompilerConfig(
        global_feature_dim=5,
        semantic_feature_dim=7,
        stage_feature_dim=6,
        tensor_feature_dim=9,
        stage_type_count=8,
        tensor_role_count=12,
        model_dim=32,
        feedforward_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        coefficient_chunk_dim=4,
        graph_message_layers=1,
        max_program_tokens=32,
    )


def conditioning() -> CompilerConditioning:
    generator = torch.Generator().manual_seed(7)
    stage_mask = torch.tensor([[True, True, False], [True, False, False]])
    tensor_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    adjacency = torch.zeros(2, 4, 4, dtype=torch.bool)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
    adjacency[0, 1, 2] = adjacency[0, 2, 1] = True
    adjacency[0, 2, 3] = adjacency[0, 3, 2] = True
    adjacency[1, 0, 1] = adjacency[1, 1, 0] = True
    return CompilerConditioning(
        global_features=torch.randn(2, 5, generator=generator),
        semantic_features=torch.randn(2, 7, generator=generator),
        stage_features=torch.randn(2, 3, 6, generator=generator),
        stage_type_ids=torch.tensor([[0, 1, 0], [2, 0, 0]]),
        stage_mask=stage_mask,
        tensor_features=torch.randn(2, 4, 9, generator=generator),
        tensor_role_ids=torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]]),
        tensor_mask=tensor_mask,
        tensor_adjacency=adjacency,
    )


def teacher_batch() -> ProgramTeacherBatch:
    token_ids = torch.tensor(
        [
            [
                PROGRAM_TOKEN_TO_ID["TENSOR_START"],
                PROGRAM_TOKEN_TO_ID["LOW_RANK"],
                PROGRAM_TOKEN_TO_ID["INTEGER"],
                PROGRAM_TOKEN_TO_ID["COEFFICIENT_CHUNK"],
                PROGRAM_TOKEN_TO_ID["TENSOR_END"],
                PROGRAM_TOKEN_TO_ID["EOS"],
            ],
            [
                PROGRAM_TOKEN_TO_ID["TENSOR_START"],
                PROGRAM_TOKEN_TO_ID["BASE_COPY"],
                PROGRAM_TOKEN_TO_ID["TENSOR_END"],
                PROGRAM_TOKEN_TO_ID["EOS"],
                PROGRAM_TOKEN_TO_ID["PAD"],
                PROGRAM_TOKEN_TO_ID["PAD"],
            ],
        ]
    )
    numeric = torch.zeros(2, 6, 4)
    numeric[0, 2, 0] = 1.0
    numeric[0, 3] = torch.tensor([0.5, -0.5, 1.0, 0.0])
    numeric_mask = torch.zeros(2, 6, dtype=torch.bool)
    numeric_mask[0, 2:4] = True
    token_mask = token_ids.ne(PROGRAM_TOKEN_TO_ID["PAD"])
    return ProgramTeacherBatch(
        token_ids=token_ids,
        numeric_values=numeric,
        numeric_mask=numeric_mask,
        token_mask=token_mask,
    )


def test_compiler_handles_variable_stage_and_tensor_counts() -> None:
    model = VariableProgramCompiler(config()).eval()
    condition = conditioning()
    memory, padding = model.encode_conditioning(condition)
    assert memory.shape == (2, 1 + 3 + 4, 32)
    assert padding.shape == (2, 8)
    assert int(padding[1].sum().item()) == 4
    assert torch.isfinite(memory).all()


def test_teacher_forcing_forward_and_loss_are_finite() -> None:
    model = VariableProgramCompiler(config())
    condition = conditioning()
    targets = teacher_batch()
    decoder_tokens = torch.full_like(targets.token_ids, PROGRAM_TOKEN_TO_ID["BOS"])
    output = model(condition, decoder_tokens, targets.numeric_values)
    assert output.token_logits.shape == (2, 6, len(PROGRAM_TOKEN_TO_ID))
    assert output.numeric_values.shape == (2, 6, 4)
    losses = model.loss(output, targets)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_greedy_generation_is_bounded_by_program_budget() -> None:
    model = VariableProgramCompiler(config()).eval()
    tokens, numeric = model.generate(conditioning(), max_tokens=7)
    assert tokens.shape[0] == 2
    assert tokens.shape[1] <= 7
    assert numeric.shape == (2, tokens.shape[1], 4)
