from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poetry50m.trajectory.config import TrajectoryConfig
from poetry50m.trajectory.forecast import (
    LinearForecastConfig,
    LowRankForecastConfig,
    linear_finite_difference,
    low_rank_temporal_forecast,
)

from .conftest import make_snapshot


def test_linear_finite_difference_exactly_recovers_irregular_linear_trajectory() -> None:
    initial = torch.tensor([1.0, -2.0])
    slope = torch.tensor([0.5, 3.0])
    previous = make_snapshot(
        checkpoint_id="s2",
        step=2,
        state_dict={"weight": initial + 2 * slope, "counter": torch.tensor(7)},
    )
    latest = make_snapshot(
        checkpoint_id="s5",
        step=5,
        state_dict={"weight": initial + 5 * slope, "counter": torch.tensor(8)},
    )

    forecast = linear_finite_difference(
        previous, latest, target_step=11, config=LinearForecastConfig(max_extrapolation_ratio=2.0)
    )

    torch.testing.assert_close(forecast.state_dict["weight"], initial + 11 * slope)
    assert torch.equal(forecast.state_dict["counter"], torch.tensor(8))
    assert forecast.diagnostics["counter"].unchanged


def test_low_rank_temporal_gram_recovers_curved_rank_two_trajectory() -> None:
    first = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    second = torch.tensor([[2.0, 1.0], [-1.0, 0.25]])

    def state(step: int) -> torch.Tensor:
        return first * step + second * (step**2)

    snapshots = tuple(
        make_snapshot(checkpoint_id=f"s{step}", step=step, state_dict={"weight": state(step)})
        for step in (0, 1, 2, 3)
    )
    forecast = low_rank_temporal_forecast(
        snapshots,
        target_step=4,
        config=LowRankForecastConfig(max_rank=2, energy_threshold=1.0, polynomial_degree=2),
    )

    torch.testing.assert_close(forecast.state_dict["weight"], state(4), atol=1e-5, rtol=1e-5)
    diagnostics = forecast.diagnostics["weight"]
    assert diagnostics.selected_rank == 2
    assert diagnostics.temporal_rank == 2
    assert diagnostics.retained_energy == 1.0
    assert diagnostics.coefficient_condition is not None


def test_low_rank_rejects_too_distant_horizon() -> None:
    snapshots = tuple(
        make_snapshot(
            checkpoint_id=f"s{step}", step=step, state_dict={"weight": torch.tensor([float(step)])}
        )
        for step in (0, 2, 5)
    )
    config = LowRankForecastConfig(max_extrapolation_ratio=0.5)

    try:
        low_rank_temporal_forecast(snapshots, target_step=9, config=config)
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("expected an extrapolation-horizon failure")


def test_linear_rejects_too_distant_horizon() -> None:
    previous = make_snapshot(checkpoint_id="s0", step=0, state_dict={"weight": torch.zeros(1)})
    latest = make_snapshot(checkpoint_id="s2", step=2, state_dict={"weight": torch.ones(1)})
    with pytest.raises(ValueError, match="horizon"):
        linear_finite_difference(
            previous,
            latest,
            target_step=4,
            config=LinearForecastConfig(max_extrapolation_ratio=0.5),
        )


def test_first_branch_config_declares_validated_linear_horizon() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "trajectory" / "first_branch.json"
    config = TrajectoryConfig.load(config_path)
    assert config.linear.max_extrapolation_ratio == 2.0


@pytest.mark.parametrize(
    "factory",
    (
        lambda: LinearForecastConfig(max_extrapolation_ratio=True),
        lambda: LowRankForecastConfig(max_rank=True),
        lambda: LowRankForecastConfig(energy_threshold=float("nan")),
        lambda: LowRankForecastConfig(polynomial_degree=True),
    ),
)
def test_forecast_configs_reject_boolean_and_nonfinite_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    ("section", "field", "value", "exception"),
    (
        ("low_rank", "max_rank", True, TypeError),
        ("safety", "near_zero_norm", "small", TypeError),
        ("gates", "unexpected", 1.0, ValueError),
    ),
)
def test_trajectory_config_rejects_malformed_nested_sections(
    section: str, field: str, value: object, exception: type[Exception]
) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "trajectory" / "first_branch.json"
    mapping = json.loads(config_path.read_text())
    mapping[section][field] = value
    with pytest.raises(exception):
        TrajectoryConfig.from_mapping(mapping)
