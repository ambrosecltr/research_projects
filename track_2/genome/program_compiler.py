from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

PROGRAM_TOKEN_NAMES = (
    "PAD",
    "BOS",
    "EOS",
    "TENSOR_START",
    "TENSOR_END",
    "BASE_COPY",
    "LOW_RANK",
    "KRONECKER",
    "SPECTRAL_DCT",
    "SHARED_BASIS",
    "CODEBOOK_BLOCKS",
    "LOW_RANK_PATCH",
    "SPARSE_PATCH",
    "INTEGER",
    "COEFFICIENT_CHUNK",
)
PROGRAM_TOKEN_TO_ID = {name: index for index, name in enumerate(PROGRAM_TOKEN_NAMES)}


@dataclass(frozen=True)
class ProgramCompilerConfig:
    global_feature_dim: int
    semantic_feature_dim: int
    stage_feature_dim: int
    tensor_feature_dim: int
    stage_type_count: int
    tensor_role_count: int
    model_dim: int = 384
    feedforward_dim: int = 1024
    encoder_layers: int = 6
    decoder_layers: int = 6
    attention_heads: int = 8
    coefficient_chunk_dim: int = 16
    dropout: float = 0.0
    graph_message_layers: int = 2
    max_program_tokens: int = 4096

    def __post_init__(self) -> None:
        positive = (
            "global_feature_dim",
            "semantic_feature_dim",
            "stage_feature_dim",
            "tensor_feature_dim",
            "stage_type_count",
            "tensor_role_count",
            "model_dim",
            "feedforward_dim",
            "encoder_layers",
            "decoder_layers",
            "attention_heads",
            "coefficient_chunk_dim",
            "max_program_tokens",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.graph_message_layers, bool)
            or not isinstance(self.graph_message_layers, int)
            or self.graph_message_layers < 0
        ):
            raise ValueError("graph_message_layers must be a non-negative integer")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class CompilerConditioning:
    """Variable-sized architecture and training-problem description."""

    global_features: torch.Tensor
    semantic_features: torch.Tensor
    stage_features: torch.Tensor
    stage_type_ids: torch.Tensor
    stage_mask: torch.Tensor
    tensor_features: torch.Tensor
    tensor_role_ids: torch.Tensor
    tensor_mask: torch.Tensor
    tensor_adjacency: torch.Tensor

    def validate(self, config: ProgramCompilerConfig) -> None:
        if self.global_features.ndim != 2 or self.global_features.shape[1] != config.global_feature_dim:
            raise ValueError("global feature shape differs from compiler configuration")
        batch = self.global_features.shape[0]
        if self.semantic_features.shape != (batch, config.semantic_feature_dim):
            raise ValueError("semantic feature shape differs from compiler configuration")
        if self.stage_features.ndim != 3 or self.stage_features.shape[0] != batch:
            raise ValueError("stage features must have shape [batch, stages, features]")
        if self.stage_features.shape[2] != config.stage_feature_dim:
            raise ValueError("stage feature width differs from compiler configuration")
        if self.stage_type_ids.shape != self.stage_features.shape[:2]:
            raise ValueError("stage type IDs must align with stage features")
        if self.stage_mask.shape != self.stage_features.shape[:2] or self.stage_mask.dtype != torch.bool:
            raise ValueError("stage mask must be boolean and align with stage features")
        if self.tensor_features.ndim != 3 or self.tensor_features.shape[0] != batch:
            raise ValueError("tensor features must have shape [batch, tensors, features]")
        if self.tensor_features.shape[2] != config.tensor_feature_dim:
            raise ValueError("tensor feature width differs from compiler configuration")
        if self.tensor_role_ids.shape != self.tensor_features.shape[:2]:
            raise ValueError("tensor role IDs must align with tensor features")
        if self.tensor_mask.shape != self.tensor_features.shape[:2] or self.tensor_mask.dtype != torch.bool:
            raise ValueError("tensor mask must be boolean and align with tensor features")
        expected_adjacency = (
            batch,
            self.tensor_features.shape[1],
            self.tensor_features.shape[1],
        )
        if self.tensor_adjacency.shape != expected_adjacency:
            raise ValueError("tensor adjacency must have shape [batch, tensors, tensors]")
        if self.tensor_adjacency.dtype != torch.bool:
            raise ValueError("tensor adjacency must be boolean")
        if self.stage_type_ids.numel() and (
            self.stage_type_ids.min() < 0
            or self.stage_type_ids.max() >= config.stage_type_count
        ):
            raise ValueError("stage type ID is outside the configured vocabulary")
        if self.tensor_role_ids.numel() and (
            self.tensor_role_ids.min() < 0
            or self.tensor_role_ids.max() >= config.tensor_role_count
        ):
            raise ValueError("tensor role ID is outside the configured vocabulary")


@dataclass(frozen=True)
class ProgramTeacherBatch:
    token_ids: torch.Tensor
    numeric_values: torch.Tensor
    numeric_mask: torch.Tensor
    token_mask: torch.Tensor

    def validate(self, config: ProgramCompilerConfig) -> None:
        if self.token_ids.ndim != 2:
            raise ValueError("program token IDs must have shape [batch, length]")
        if self.numeric_values.shape != (
            self.token_ids.shape[0],
            self.token_ids.shape[1],
            config.coefficient_chunk_dim,
        ):
            raise ValueError("program numeric values do not align with token IDs")
        if self.numeric_mask.shape != self.token_ids.shape or self.numeric_mask.dtype != torch.bool:
            raise ValueError("numeric mask must be boolean and align with token IDs")
        if self.token_mask.shape != self.token_ids.shape or self.token_mask.dtype != torch.bool:
            raise ValueError("token mask must be boolean and align with token IDs")
        if self.token_ids.numel() and (
            self.token_ids.min() < 0 or self.token_ids.max() >= len(PROGRAM_TOKEN_NAMES)
        ):
            raise ValueError("program token ID is outside the vocabulary")
        if self.token_ids.shape[1] > config.max_program_tokens:
            raise ValueError("program sequence exceeds max_program_tokens")


class ProgramCompilerOutput(NamedTuple):
    token_logits: torch.Tensor
    numeric_values: torch.Tensor
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor


class ProgramCompilerLoss(NamedTuple):
    total: torch.Tensor
    token: torch.Tensor
    numeric: torch.Tensor
    rate: torch.Tensor


class GraphMessageBlock(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(model_dim, model_dim)
        self.update = nn.Sequential(
            nn.Linear(model_dim * 2, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        tensors: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_edges = adjacency & mask.unsqueeze(1) & mask.unsqueeze(2)
        edge_weights = valid_edges.to(tensors.dtype)
        degree = edge_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        messages = torch.bmm(edge_weights, self.message(tensors)) / degree
        updated = self.update(torch.cat([tensors, messages], dim=-1))
        return self.norm(tensors + updated) * mask.unsqueeze(-1)


class SinusoidalPositions(nn.Module):
    def __init__(self, model_dim: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dim)
        )
        table = torch.zeros(max_length, model_dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        if length > self.table.shape[0]:
            raise ValueError("position request exceeds configured maximum")
        return self.table[:length]


class VariableProgramCompiler(nn.Module):
    """One learned compiler that emits a compact MGP token stream.

    The output length scales with formulas and coefficients, not with the child model's parameter
    count. The deterministic MGP Runtime remains the only component that expands formulas into
    complete tensors.
    """

    def __init__(self, config: ProgramCompilerConfig) -> None:
        super().__init__()
        self.config = config
        self.global_encoder = nn.Linear(
            config.global_feature_dim + config.semantic_feature_dim,
            config.model_dim,
        )
        self.stage_encoder = nn.Linear(config.stage_feature_dim, config.model_dim)
        self.tensor_encoder = nn.Linear(config.tensor_feature_dim, config.model_dim)
        self.stage_type_embedding = nn.Embedding(config.stage_type_count, config.model_dim)
        self.tensor_role_embedding = nn.Embedding(config.tensor_role_count, config.model_dim)
        self.condition_type_embedding = nn.Embedding(3, config.model_dim)
        self.graph_blocks = nn.ModuleList(
            GraphMessageBlock(config.model_dim, config.feedforward_dim, config.dropout)
            for _ in range(config.graph_message_layers)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.condition_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )

        self.program_token_embedding = nn.Embedding(len(PROGRAM_TOKEN_NAMES), config.model_dim)
        self.program_numeric_projection = nn.Linear(config.coefficient_chunk_dim, config.model_dim)
        self.program_positions = SinusoidalPositions(config.model_dim, config.max_program_tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.program_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.token_head = nn.Linear(config.model_dim, len(PROGRAM_TOKEN_NAMES))
        self.numeric_head = nn.Linear(config.model_dim, config.coefficient_chunk_dim)

    def encode_conditioning(
        self,
        conditioning: CompilerConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning.validate(self.config)
        batch = conditioning.global_features.shape[0]
        global_token = self.global_encoder(
            torch.cat([conditioning.global_features, conditioning.semantic_features], dim=-1)
        ).unsqueeze(1)
        global_token = global_token + self.condition_type_embedding.weight[0].view(1, 1, -1)

        stage_tokens = self.stage_encoder(conditioning.stage_features)
        stage_tokens = stage_tokens + self.stage_type_embedding(conditioning.stage_type_ids)
        stage_tokens = stage_tokens + self.condition_type_embedding.weight[1].view(1, 1, -1)
        stage_tokens = stage_tokens * conditioning.stage_mask.unsqueeze(-1)

        tensor_tokens = self.tensor_encoder(conditioning.tensor_features)
        tensor_tokens = tensor_tokens + self.tensor_role_embedding(conditioning.tensor_role_ids)
        tensor_tokens = tensor_tokens + self.condition_type_embedding.weight[2].view(1, 1, -1)
        tensor_tokens = tensor_tokens * conditioning.tensor_mask.unsqueeze(-1)
        for block in self.graph_blocks:
            tensor_tokens = block(
                tensor_tokens,
                conditioning.tensor_adjacency,
                conditioning.tensor_mask,
            )

        memory = torch.cat([global_token, stage_tokens, tensor_tokens], dim=1)
        global_mask = torch.ones(batch, 1, dtype=torch.bool, device=memory.device)
        valid_mask = torch.cat(
            [global_mask, conditioning.stage_mask, conditioning.tensor_mask], dim=1
        )
        padding_mask = ~valid_mask
        memory = self.condition_encoder(memory, src_key_padding_mask=padding_mask)
        return memory, padding_mask

    def forward(
        self,
        conditioning: CompilerConditioning,
        decoder_token_ids: torch.Tensor,
        decoder_numeric_values: torch.Tensor | None = None,
    ) -> ProgramCompilerOutput:
        memory, memory_padding_mask = self.encode_conditioning(conditioning)
        if decoder_token_ids.ndim != 2 or decoder_token_ids.shape[0] != memory.shape[0]:
            raise ValueError("decoder token IDs must have shape [batch, length]")
        if decoder_token_ids.shape[1] > self.config.max_program_tokens:
            raise ValueError("decoder sequence exceeds max_program_tokens")
        if decoder_token_ids.numel() and (
            decoder_token_ids.min() < 0
            or decoder_token_ids.max() >= len(PROGRAM_TOKEN_NAMES)
        ):
            raise ValueError("decoder token ID is outside the program vocabulary")
        if decoder_numeric_values is None:
            decoder_numeric_values = torch.zeros(
                decoder_token_ids.shape[0],
                decoder_token_ids.shape[1],
                self.config.coefficient_chunk_dim,
                dtype=memory.dtype,
                device=memory.device,
            )
        if decoder_numeric_values.shape != (
            decoder_token_ids.shape[0],
            decoder_token_ids.shape[1],
            self.config.coefficient_chunk_dim,
        ):
            raise ValueError("decoder numeric values do not align with decoder tokens")

        target = self.program_token_embedding(decoder_token_ids)
        target = target + self.program_numeric_projection(decoder_numeric_values.to(target.dtype))
        target = target + self.program_positions(target.shape[1]).to(target.device, target.dtype)
        causal_mask = torch.triu(
            torch.ones(
                target.shape[1],
                target.shape[1],
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        target_padding_mask = decoder_token_ids.eq(PROGRAM_TOKEN_TO_ID["PAD"])
        decoded = self.program_decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return ProgramCompilerOutput(
            token_logits=self.token_head(decoded),
            numeric_values=self.numeric_head(decoded),
            memory=memory,
            memory_padding_mask=memory_padding_mask,
        )

    def loss(
        self,
        output: ProgramCompilerOutput,
        targets: ProgramTeacherBatch,
        *,
        numeric_weight: float = 1.0,
        rate_weight: float = 1e-4,
    ) -> ProgramCompilerLoss:
        targets.validate(self.config)
        if output.token_logits.shape[:2] != targets.token_ids.shape:
            raise ValueError("compiler output and target sequence shapes differ")
        if numeric_weight < 0.0 or rate_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        token_loss = F.cross_entropy(
            output.token_logits.reshape(-1, output.token_logits.shape[-1]),
            targets.token_ids.reshape(-1),
            ignore_index=PROGRAM_TOKEN_TO_ID["PAD"],
        )
        if bool(targets.numeric_mask.any().item()):
            numeric_error = (output.numeric_values - targets.numeric_values).square().mean(dim=-1)
            numeric_loss = numeric_error[targets.numeric_mask].mean()
        else:
            numeric_loss = output.numeric_values.sum() * 0.0
        rate = targets.token_mask.to(torch.float32).sum(dim=1).mean() / self.config.max_program_tokens
        total = token_loss + numeric_weight * numeric_loss + rate_weight * rate
        return ProgramCompilerLoss(total=total, token=token_loss, numeric=numeric_loss, rate=rate)

    @torch.no_grad()
    def generate(
        self,
        conditioning: CompilerConditioning,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        conditioning.validate(self.config)
        limit = min(max_tokens or self.config.max_program_tokens, self.config.max_program_tokens)
        batch = conditioning.global_features.shape[0]
        device = conditioning.global_features.device
        tokens = torch.full(
            (batch, 1),
            PROGRAM_TOKEN_TO_ID["BOS"],
            dtype=torch.long,
            device=device,
        )
        numeric = torch.zeros(
            batch,
            1,
            self.config.coefficient_chunk_dim,
            dtype=conditioning.global_features.dtype,
            device=device,
        )
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(limit - 1):
            output = self(conditioning, tokens, numeric)
            logits = output.token_logits[:, -1]
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1)
            else:
                next_token = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1),
                    num_samples=1,
                ).squeeze(1)
            next_numeric = output.numeric_values[:, -1]
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PROGRAM_TOKEN_TO_ID["PAD"]),
                next_token,
            )
            next_numeric = torch.where(
                finished.unsqueeze(1),
                torch.zeros_like(next_numeric),
                next_numeric,
            )
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            numeric = torch.cat([numeric, next_numeric.unsqueeze(1)], dim=1)
            finished |= next_token.eq(PROGRAM_TOKEN_TO_ID["EOS"])
            if bool(finished.all().item()):
                break
        return tokens, numeric
