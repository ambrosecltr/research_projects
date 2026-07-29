from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .block_decoder import BlockDecoderConfig


def _mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int,
    depth: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dimension = input_dim
    for _ in range(max(depth, 1)):
        layers.extend(
            [
                nn.Linear(dimension, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ]
        )
        dimension = hidden_dim
    layers.append(nn.Linear(dimension, output_dim))
    return nn.Sequential(*layers)


class BlockwiseGenomeCompiler(nn.Module):
    """Predict hierarchical genome codes without a model-sized flat output head."""

    def __init__(
        self,
        *,
        architecture_dim: int,
        dataset_fingerprint_dim: int,
        conditioning_dim: int,
        layer_count: int,
        tensor_count: int,
        role_count: int,
        decoder_config: BlockDecoderConfig,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("architecture_dim", architecture_dim),
            ("dataset_fingerprint_dim", dataset_fingerprint_dim),
            ("conditioning_dim", conditioning_dim),
            ("layer_count", layer_count),
            ("tensor_count", tensor_count),
            ("role_count", role_count),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if decoder_config.block_code_dim < 1:
            raise ValueError("blockwise compiler requires decoder block codes")

        self.decoder_config = decoder_config
        self.layer_count = layer_count
        self.tensor_count = tensor_count
        self.role_count = role_count
        self.hidden_dim = hidden_dim
        self.depth = depth
        index_dim = decoder_config.role_embedding_dim
        evidence_dim = architecture_dim + dataset_fingerprint_dim + conditioning_dim

        self.evidence_encoder = _mlp(
            evidence_dim,
            hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.layer_embedding = nn.Embedding(layer_count, index_dim)
        self.tensor_embedding = nn.Embedding(tensor_count, index_dim)
        self.role_embedding = nn.Embedding(role_count, index_dim)
        self.global_head = nn.Linear(hidden_dim, decoder_config.global_code_dim)
        self.layer_head = _mlp(
            hidden_dim + index_dim,
            decoder_config.layer_code_dim,
            hidden_dim=hidden_dim,
            depth=1,
        )
        self.tensor_head = _mlp(
            hidden_dim + index_dim,
            decoder_config.tensor_code_dim,
            hidden_dim=hidden_dim,
            depth=1,
        )
        self.block_head = _mlp(
            hidden_dim + 3 * index_dim + decoder_config.feature_dim,
            decoder_config.block_code_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def encode_evidence(
        self,
        architecture: torch.Tensor,
        dataset: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        return self.evidence_encoder(torch.cat([architecture, dataset, conditioning], dim=-1))

    def shared_codes(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        life_count = context.shape[0]
        layer_ids = torch.arange(self.layer_count, device=context.device)
        tensor_ids = torch.arange(self.tensor_count, device=context.device)
        layer_context = context[:, None, :].expand(-1, self.layer_count, -1)
        tensor_context = context[:, None, :].expand(-1, self.tensor_count, -1)
        layer_features = self.layer_embedding(layer_ids)[None, :, :].expand(life_count, -1, -1)
        tensor_features = self.tensor_embedding(tensor_ids)[None, :, :].expand(life_count, -1, -1)
        return {
            "global_code": self.global_head(context),
            "layer_codes": self.layer_head(torch.cat([layer_context, layer_features], dim=-1)),
            "tensor_codes": self.tensor_head(torch.cat([tensor_context, tensor_features], dim=-1)),
        }

    def decoder_inputs(
        self,
        context: torch.Tensor,
        *,
        life_indices: torch.Tensor,
        layer_slots: torch.Tensor,
        tensor_indices: torch.Tensor,
        role_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shared = self.shared_codes(context)
        selected_context = context[life_indices]
        layer_features = self.layer_embedding(layer_slots)
        tensor_features = self.tensor_embedding(tensor_indices)
        role_features = self.role_embedding(role_ids)
        block_codes = self.block_head(
            torch.cat(
                [
                    selected_context,
                    layer_features,
                    tensor_features,
                    role_features,
                    features,
                ],
                dim=-1,
            )
        )
        return {
            "global_codes": shared["global_code"][life_indices],
            "layer_codes": shared["layer_codes"][life_indices, layer_slots],
            "tensor_codes": shared["tensor_codes"][life_indices, tensor_indices],
            "block_codes": block_codes,
            "role_ids": role_ids,
            "features": features,
        }

    @staticmethod
    def rate_proxy(codes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = [
            value.square().mean()
            for name, value in codes.items()
            if name.endswith(("code", "codes"))
        ]
        if not values:
            raise ValueError("compiler code mapping is empty")
        return torch.stack(values).mean()
