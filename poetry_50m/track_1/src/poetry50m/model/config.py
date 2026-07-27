"""Typed configuration for Track 1 decoder-only models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

Architecture = Literal["gpt", "ngpt"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration shared by the conventional GPT and normalized nGPT variants.

    ``ffn_dim`` is explicit rather than inferred so parameter budgets remain auditable.
    The production configuration in ``configs/model/track1_8m.yaml`` has 8,335,008
    trainable parameters with these assumptions (no linear biases).
    """

    architecture: Architecture
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rope_fraction: float = 1.0
    norm_epsilon: float = 1e-6
    linear_bias: bool = False
    tie_embeddings: bool = True
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.architecture not in {"gpt", "ngpt"}:
            raise ValueError(f"architecture must be 'gpt' or 'ngpt', got {self.architecture!r}")
        for field_name in (
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
            "ffn_dim",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        for field_name in ("dropout", "rope_base", "rope_fraction", "norm_epsilon"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not isinstance(self.linear_bias, bool) or not isinstance(self.tie_embeddings, bool):
            raise TypeError("linear_bias and tie_embeddings must be booleans")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must divide evenly across n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even so rotary pairs can be formed")
        if self.ffn_dim < self.d_model:
            raise ValueError("ffn_dim must be at least d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError("rope_fraction must lie in (0, 1]")
        if self.rope_base <= 1.0:
            raise ValueError("rope_base must be greater than 1")
        if self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def rope_dim(self) -> int:
        requested = int(self.head_dim * self.rope_fraction)
        return max(2, requested - (requested % 2))

    @property
    def base_scale(self) -> float:
        """nGPT's width-dependent parameterization scale, 1 / sqrt(d_model)."""
        return float(self.d_model**-0.5)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ModelConfig:
        """Construct a validated config from a decoded YAML or JSON mapping."""
        parameters = cast(dict[str, Any], dict(values))
        return cls(**parameters)
