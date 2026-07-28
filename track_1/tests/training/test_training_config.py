from __future__ import annotations

import math
from dataclasses import replace

import pytest

from poetry50m.training import TrainConfig


def valid_config() -> TrainConfig:
    return TrainConfig(max_steps=4, learning_rate=1e-3, device="cpu", precision="none")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_steps", True),
        ("warmup_steps", False),
        ("learning_rate", math.nan),
        ("weight_decay", math.inf),
        ("beta1", math.nan),
        ("epsilon", math.nan),
        ("min_learning_rate_ratio", math.nan),
        ("max_grad_norm", True),
        ("deterministic", 1),
    ),
)
def test_train_config_rejects_boolean_and_nonfinite_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(valid_config(), **{field: value})
