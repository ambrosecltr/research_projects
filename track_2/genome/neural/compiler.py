"""Archived fixed-layout V1 compiler for historical reproduction only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class GenomeCodeLayout:
    global_code_dim: int
    n_layers: int
    layer_code_dim: int
    n_tensors: int
    tensor_code_dim: int

    def __post_init__(self) -> None:
        for name in (
            "global_code_dim",
            "n_layers",
            "layer_code_dim",
            "n_tensors",
            "tensor_code_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def total_dim(self) -> int:
        return (
            self.global_code_dim
            + self.n_layers * self.layer_code_dim
            + self.n_tensors * self.tensor_code_dim
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def flatten(
        self, global_code: torch.Tensor, layer_codes: torch.Tensor, tensor_codes: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([global_code.flatten(), layer_codes.flatten(), tensor_codes.flatten()])

    def unflatten(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.shape[-1] != self.total_dim:
            raise ValueError(f"code vector has {vector.shape[-1]} values; expected {self.total_dim}")
        start = 0
        global_code = vector[..., start : start + self.global_code_dim]
        start += self.global_code_dim
        layer_count = self.n_layers * self.layer_code_dim
        layer_codes = vector[..., start : start + layer_count].reshape(
            *vector.shape[:-1], self.n_layers, self.layer_code_dim
        )
        start += layer_count
        tensor_codes = vector[..., start:].reshape(
            *vector.shape[:-1], self.n_tensors, self.tensor_code_dim
        )
        return {
            "global_code": global_code,
            "layer_codes": layer_codes,
            "tensor_codes": tensor_codes,
        }


@dataclass
class CompilerDistribution:
    mean: torch.Tensor
    log_scale: torch.Tensor
    layout: GenomeCodeLayout

    def rsample(self) -> dict[str, torch.Tensor]:
        noise = torch.randn_like(self.mean)
        sample = self.mean + noise * self.log_scale.exp()
        return self.layout.unflatten(sample)

    def mode(self) -> dict[str, torch.Tensor]:
        return self.layout.unflatten(self.mean)

    def rate_proxy(self) -> torch.Tensor:
        # Differential-Gaussian proxy. Final claims must use quantized serialized bytes.
        return 0.5 * torch.sum(self.mean.square() + (2 * self.log_scale).exp() - 2 * self.log_scale)


class GenomeCompiler(nn.Module):
    """Baseline G1 compiler from fixed-size model-native evidence to genome codes.

    This intentionally does not pretend to solve architecture-general compilation. Architecture,
    dataset, and trajectory encoders can be swapped while preserving the output contract.
    """

    def __init__(
        self,
        *,
        architecture_dim: int,
        dataset_fingerprint_dim: int,
        trajectory_fingerprint_dim: int,
        layout: GenomeCodeLayout,
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (
            ("architecture_dim", architecture_dim),
            ("dataset_fingerprint_dim", dataset_fingerprint_dim),
            ("trajectory_fingerprint_dim", trajectory_fingerprint_dim),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        self.layout = layout
        input_dim = architecture_dim + dataset_fingerprint_dim + trajectory_fingerprint_dim
        layers: list[nn.Module] = []
        dimension = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dimension, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
            dimension = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dimension, layout.total_dim)
        self.log_scale_head = nn.Linear(dimension, layout.total_dim)

    def forward(
        self,
        architecture_features: torch.Tensor,
        dataset_fingerprint: torch.Tensor,
        trajectory_fingerprint: torch.Tensor,
    ) -> CompilerDistribution:
        value = torch.cat(
            [architecture_features, dataset_fingerprint, trajectory_fingerprint], dim=-1
        )
        hidden = self.trunk(value)
        mean = self.mean_head(hidden)
        log_scale = self.log_scale_head(hidden).clamp(-8.0, 2.0)
        return CompilerDistribution(mean=mean, log_scale=log_scale, layout=self.layout)


def compiler_loss(
    distribution: CompilerDistribution,
    target_flat_codes: torch.Tensor,
    *,
    rate_weight: float = 1e-5,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    code_loss = torch.nn.functional.mse_loss(distribution.mean, target_flat_codes)
    rate = distribution.rate_proxy() / target_flat_codes.numel()
    total = code_loss + rate_weight * rate
    return total, {
        "total": float(total.detach().item()),
        "code_mse": float(code_loss.detach().item()),
        "rate_proxy": float(rate.detach().item()),
    }
