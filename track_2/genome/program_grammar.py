from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch.nn import functional as F

from .program_compiler import (
    PROGRAM_TOKEN_NAMES,
    PROGRAM_TOKEN_TO_ID,
    CompilerConditioning,
    VariableProgramCompiler,
)
from .program_tokens import ProgramSequence, ProgramTokenizationConfig, sequence_to_program
from .tensor_inventory import tied_owner_map
from .types import GenomeProgram, TensorSpec

Phase = Literal[
    "tensor_start",
    "opcode",
    "rank",
    "coefficients",
    "tensor_end",
    "done",
]


@dataclass
class _GrammarState:
    tensor_index: int = 0
    phase: Phase = "tensor_start"
    remaining_chunks: int = 0
    vector_value_chunks: int = 0
    coefficient_chunk_index: int = 0
    opcode: str | None = None


@dataclass(frozen=True)
class GeneratedProgram:
    sequence: ProgramSequence
    program: GenomeProgram


def _chunk_count(values: int, chunk_dim: int) -> int:
    return math.ceil(values / chunk_dim)


def _allowed_token_names(
    state: _GrammarState,
    inventory: Sequence[TensorSpec],
    aliases: dict[str, str],
) -> tuple[str, ...]:
    if state.phase == "tensor_start":
        return ("EOS",) if state.tensor_index == len(inventory) else ("TENSOR_START",)
    if state.phase == "done":
        return ("EOS",)
    if state.phase == "rank":
        return ("INTEGER",)
    if state.phase == "coefficients":
        return ("COEFFICIENT_CHUNK",)
    if state.phase == "tensor_end":
        return ("TENSOR_END",)
    if state.phase != "opcode":
        raise AssertionError(f"unsupported grammar phase: {state.phase}")

    spec = inventory[state.tensor_index]
    if spec.name in aliases or spec.is_buffer:
        return ("BASE_COPY",)
    if len(spec.shape) == 2:
        return ("BASE_COPY", "LOW_RANK")
    return ("BASE_COPY", "QUANTIZED_VECTOR")


def _masked_sample(
    logits: torch.Tensor,
    allowed_names: Sequence[str],
    *,
    temperature: float,
) -> torch.Tensor:
    allowed_ids = torch.tensor(
        [PROGRAM_TOKEN_TO_ID[name] for name in allowed_names],
        dtype=torch.long,
        device=logits.device,
    )
    selected = logits.index_select(0, allowed_ids)
    if temperature == 0.0:
        local_index = selected.argmax(dim=0)
    else:
        probabilities = torch.softmax(selected / temperature, dim=0)
        local_index = torch.multinomial(probabilities, num_samples=1).squeeze(0)
    return allowed_ids[local_index]


def _normalize_numeric_for_token(
    token_name: str,
    raw: torch.Tensor,
    state: _GrammarState,
    spec: TensorSpec | None,
    *,
    chunk_dim: int,
) -> torch.Tensor:
    value = raw.detach().to(torch.float32).clone()
    if not bool(torch.isfinite(value).all().item()):
        value.zero_()
    if token_name == "INTEGER":
        if spec is None or len(spec.shape) != 2:
            raise ValueError("rank token requires a matrix tensor")
        maximum = max(1, min(spec.shape))
        rank = max(1, min(maximum, int(round(float(value[0].item())))))
        value.zero_()
        value[0] = float(rank)
    elif token_name == "COEFFICIENT_CHUNK" and state.opcode == "QUANTIZED_VECTOR":
        if state.coefficient_chunk_index < state.vector_value_chunks:
            value = torch.round(value).clamp(-127, 127)
        else:
            positive = F.softplus(value[0]) + 1e-8
            value.zero_()
            value[0] = positive
    if value.numel() != chunk_dim:
        raise ValueError("compiler numeric head width differs from grammar chunk width")
    return value


def _advance(
    state: _GrammarState,
    token_name: str,
    numeric: torch.Tensor,
    inventory: Sequence[TensorSpec],
    *,
    chunk_dim: int,
) -> None:
    spec = inventory[state.tensor_index] if state.tensor_index < len(inventory) else None
    if state.phase == "tensor_start":
        if token_name == "EOS":
            state.phase = "done"
        else:
            state.phase = "opcode"
        return
    if state.phase == "opcode":
        state.opcode = token_name
        state.coefficient_chunk_index = 0
        if token_name == "BASE_COPY":
            state.phase = "tensor_end"
        elif token_name == "LOW_RANK":
            state.phase = "rank"
        elif token_name == "QUANTIZED_VECTOR":
            if spec is None or len(spec.shape) == 2:
                raise ValueError("QUANTIZED_VECTOR grammar state has an invalid tensor")
            state.vector_value_chunks = _chunk_count(spec.numel, chunk_dim)
            state.remaining_chunks = state.vector_value_chunks + 1
            state.phase = "coefficients"
        else:
            raise ValueError(f"unsupported active compiler opcode: {token_name}")
        return
    if state.phase == "rank":
        if spec is None or len(spec.shape) != 2:
            raise ValueError("rank grammar state has an invalid tensor")
        rank = int(numeric[0].item())
        rows, cols = spec.shape
        state.remaining_chunks = (
            _chunk_count(rows * rank, chunk_dim)
            + _chunk_count(rank, chunk_dim)
            + _chunk_count(rank * cols, chunk_dim)
        )
        state.coefficient_chunk_index = 0
        state.phase = "coefficients" if state.remaining_chunks else "tensor_end"
        return
    if state.phase == "coefficients":
        state.remaining_chunks -= 1
        state.coefficient_chunk_index += 1
        if state.remaining_chunks < 0:
            raise AssertionError("grammar consumed too many coefficient chunks")
        if state.remaining_chunks == 0:
            state.phase = "tensor_end"
        return
    if state.phase == "tensor_end":
        state.tensor_index += 1
        state.opcode = None
        state.phase = "done" if state.tensor_index == len(inventory) else "tensor_start"
        return
    if state.phase == "done" and token_name != "EOS":
        raise AssertionError("done grammar state accepted a non-EOS token")


@torch.no_grad()
def generate_valid_program_sequence(
    model: VariableProgramCompiler,
    conditioning: CompilerConditioning,
    inventory: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> ProgramSequence:
    """Generate one syntactically valid program for one model life.

    The grammar masks impossible opcodes, fixes tensor order, clamps predicted ranks to each tensor,
    rounds int8 vectors, and maps predicted scales to positive values. It never inserts a dense
    fallback or residual payload.
    """

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    conditioning.validate(model.config)
    if conditioning.global_features.shape[0] != 1:
        raise ValueError("grammar-constrained generation currently accepts one life at a time")
    if not inventory:
        raise ValueError("program generation requires a non-empty tensor inventory")
    limit = min(max_tokens or model.config.max_program_tokens, model.config.max_program_tokens)
    if limit < 2:
        raise ValueError("max_tokens is too small for an MGP")

    aliases = tied_owner_map(tied_groups)
    device = conditioning.global_features.device
    tokens = [PROGRAM_TOKEN_TO_ID["BOS"]]
    numeric_values = [torch.zeros(model.config.coefficient_chunk_dim, dtype=torch.float32)]
    numeric_mask = [False]
    state = _GrammarState()

    while len(tokens) < limit:
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        numeric_tensor = torch.stack(numeric_values).unsqueeze(0).to(device)
        output = model(conditioning, token_tensor, numeric_tensor)
        allowed = _allowed_token_names(state, inventory, aliases)
        next_token = _masked_sample(
            output.token_logits[0, -1],
            allowed,
            temperature=temperature,
        )
        token_name = PROGRAM_TOKEN_NAMES[int(next_token.item())]
        spec = inventory[state.tensor_index] if state.tensor_index < len(inventory) else None
        next_numeric = _normalize_numeric_for_token(
            token_name,
            output.numeric_values[0, -1],
            state,
            spec,
            chunk_dim=model.config.coefficient_chunk_dim,
        )
        tokens.append(int(next_token.item()))
        numeric_values.append(next_numeric.cpu())
        numeric_mask.append(token_name in {"INTEGER", "COEFFICIENT_CHUNK"})
        _advance(
            state,
            token_name,
            next_numeric,
            inventory,
            chunk_dim=model.config.coefficient_chunk_dim,
        )
        if token_name == "EOS":
            break
    else:
        raise ValueError("generated program exceeded max_tokens before reaching EOS")

    sequence = ProgramSequence(
        token_ids=torch.tensor(tokens, dtype=torch.long),
        numeric_values=torch.stack(numeric_values),
        numeric_mask=torch.tensor(numeric_mask, dtype=torch.bool),
        token_mask=torch.ones(len(tokens), dtype=torch.bool),
    )
    # Parsing here is intentional: generation is not accepted merely because the token state
    # machine terminated. The deterministic inverse must also accept the result.
    sequence_to_program(
        sequence,
        inventory,
        tied_groups=tied_groups,
        config=ProgramTokenizationConfig(
            coefficient_chunk_dim=model.config.coefficient_chunk_dim
        ),
        candidate_id="grammar-validation",
    )
    return sequence


@torch.no_grad()
def generate_valid_program(
    model: VariableProgramCompiler,
    conditioning: CompilerConditioning,
    inventory: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    max_tokens: int | None = None,
    temperature: float = 0.0,
    candidate_id: str = "compiled-program",
) -> GeneratedProgram:
    sequence = generate_valid_program_sequence(
        model,
        conditioning,
        inventory,
        tied_groups=tied_groups,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    program = sequence_to_program(
        sequence,
        inventory,
        tied_groups=tied_groups,
        config=ProgramTokenizationConfig(
            coefficient_chunk_dim=model.config.coefficient_chunk_dim
        ),
        candidate_id=candidate_id,
    )
    return GeneratedProgram(sequence=sequence, program=program)
