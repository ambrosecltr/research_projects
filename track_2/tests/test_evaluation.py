from __future__ import annotations

import pytest
import torch

from genome.evaluation import hidden_result_tier, parameter_distortion


def test_parameter_distortion_is_mean_relative_delta_error() -> None:
    base = {
        "a": torch.tensor([0.0, 0.0]),
        "b": torch.tensor([3.0]),
    }
    endpoint = {
        "a": torch.tensor([2.0, 0.0]),
        "b": torch.tensor([5.0]),
    }
    candidate = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([5.0]),
    }

    assert parameter_distortion(base, candidate, endpoint) == pytest.approx(0.125)


def test_parameter_distortion_rejects_different_states() -> None:
    with pytest.raises(ValueError, match="state dictionaries differ"):
        parameter_distortion(
            {"a": torch.tensor([0.0])},
            {"b": torch.tensor([0.0])},
            {"a": torch.tensor([1.0])},
        )


@pytest.mark.parametrize(
    ("progress", "tier"),
    [
        (-0.1, "no_signal"),
        (0.0, "no_signal"),
        (0.1, "weak_signal"),
        (0.249, "weak_signal"),
        (0.25, "partial_result"),
        (0.799, "partial_result"),
        (0.8, "strong_result"),
        (1.0, "strong_result"),
    ],
)
def test_hidden_result_tiers_are_predeclared(progress: float, tier: str) -> None:
    assert hidden_result_tier(progress) == tier
