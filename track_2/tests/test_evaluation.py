from __future__ import annotations

import pytest
import torch

from genome.evaluation import parameter_distortion


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
