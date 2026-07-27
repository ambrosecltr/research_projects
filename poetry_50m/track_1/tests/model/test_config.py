from __future__ import annotations

import math
from dataclasses import replace

import pytest

from poetry50m.model import ModelConfig


def valid_config() -> ModelConfig:
    return ModelConfig(
        architecture="gpt",
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("d_model", True),
        ("vocab_size", False),
        ("dropout", math.nan),
        ("rope_base", math.inf),
        ("rope_fraction", math.nan),
        ("norm_epsilon", True),
        ("linear_bias", 1),
    ),
)
def test_model_config_rejects_boolean_and_nonfinite_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(valid_config(), **{field: value})
