"""Finite-difference and temporal-Gram low-rank weight forecasting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from poetry50m.trajectory.snapshots import assert_single_run_trajectory
from poetry50m.trajectory.types import WeightSnapshot


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class LowRankForecastConfig:
    """Numerically bounded low-rank extrapolation settings."""

    max_rank: int = 4
    energy_threshold: float = 0.98
    polynomial_degree: int = 1
    max_extrapolation_ratio: float = 2.0
    eigenvalue_relative_floor: float = 1e-10

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_rank, bool)
            or not isinstance(self.max_rank, int)
            or self.max_rank < 1
        ):
            raise ValueError("max_rank must be positive")
        energy_threshold = _finite_number(self.energy_threshold, name="energy_threshold")
        max_extrapolation_ratio = _finite_number(
            self.max_extrapolation_ratio, name="max_extrapolation_ratio"
        )
        eigenvalue_relative_floor = _finite_number(
            self.eigenvalue_relative_floor, name="eigenvalue_relative_floor"
        )
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must lie in (0, 1]")
        if (
            isinstance(self.polynomial_degree, bool)
            or not isinstance(self.polynomial_degree, int)
            or self.polynomial_degree < 1
        ):
            raise ValueError("polynomial_degree must be at least one")
        if max_extrapolation_ratio <= 0.0:
            raise ValueError("max_extrapolation_ratio must be positive")
        if eigenvalue_relative_floor <= 0.0:
            raise ValueError("eigenvalue_relative_floor must be positive")


@dataclass(frozen=True, slots=True)
class LinearForecastConfig:
    """Bound a direct finite-difference leap by observed trajectory duration."""

    max_extrapolation_ratio: float = 2.0

    def __post_init__(self) -> None:
        if _finite_number(self.max_extrapolation_ratio, name="max_extrapolation_ratio") <= 0.0:
            raise ValueError("max_extrapolation_ratio must be positive")


@dataclass(frozen=True, slots=True)
class TensorTrajectoryDiagnostics:
    name: str
    floating_point: bool
    selected_rank: int
    temporal_rank: int
    retained_energy: float
    singular_values: tuple[float, ...]
    coefficient_condition: float | None
    unchanged: bool = False


@dataclass(frozen=True, slots=True)
class ForecastResult:
    method: str
    source_checkpoint_ids: tuple[str, ...]
    source_steps: tuple[int, ...]
    target_step: int
    state_dict: Mapping[str, Tensor]
    diagnostics: Mapping[str, TensorTrajectoryDiagnostics]

    def diagnostics_mapping(self) -> dict[str, object]:
        return {name: asdict(value) for name, value in self.diagnostics.items()}


def _require_target_step(snapshots: tuple[WeightSnapshot, ...], target_step: int) -> None:
    if target_step <= snapshots[-1].metadata.step:
        raise ValueError("forecast target_step must be later than the latest checkpoint")


def linear_finite_difference(
    previous: WeightSnapshot,
    latest: WeightSnapshot,
    *,
    target_step: int,
    config: LinearForecastConfig,
) -> ForecastResult:
    """Extrapolate each floating tensor through the latest irregular step interval."""

    snapshots = (previous, latest)
    assert_single_run_trajectory(snapshots)
    _require_target_step(snapshots, target_step)
    spacing = latest.metadata.step - previous.metadata.step
    if target_step - latest.metadata.step > spacing * config.max_extrapolation_ratio:
        raise ValueError("target step exceeds configured extrapolation horizon")
    scale = (target_step - latest.metadata.step) / spacing
    result: dict[str, Tensor] = {}
    diagnostics: dict[str, TensorTrajectoryDiagnostics] = {}
    for name, latest_tensor in latest.state_dict.items():
        prior_tensor = previous.state_dict[name]
        if not latest_tensor.is_floating_point():
            result[name] = latest_tensor.clone()
            diagnostics[name] = TensorTrajectoryDiagnostics(
                name, False, 0, 0, 1.0, (), None, unchanged=True
            )
            continue
        delta = latest_tensor - prior_tensor
        result[name] = latest_tensor + delta * scale
        diagnostics[name] = TensorTrajectoryDiagnostics(
            name, True, 1, 1, 1.0, (float(delta.norm().item()),), 1.0
        )
    return ForecastResult(
        method="linear_finite_difference",
        source_checkpoint_ids=(previous.metadata.checkpoint_id, latest.metadata.checkpoint_id),
        source_steps=(previous.metadata.step, latest.metadata.step),
        target_step=target_step,
        state_dict=result,
        diagnostics=diagnostics,
    )


def _temporal_gram(deltas: tuple[Tensor, ...]) -> Tensor:
    """Construct only a time-by-time Gram matrix, never a feature-by-time matrix."""

    count = len(deltas)
    gram = torch.empty((count, count), dtype=torch.float64)
    for row in range(count):
        for column in range(row, count):
            value = torch.sum(deltas[row].double() * deltas[column].double()).item()
            gram[row, column] = value
            gram[column, row] = value
    return gram


def _select_rank(
    eigenvalues: Tensor, config: LowRankForecastConfig
) -> tuple[int, int, float, Tensor]:
    positive = eigenvalues.clamp_min(0.0)
    total = positive.sum()
    if total <= 0.0:
        return 0, 0, 1.0, positive
    descending = torch.flip(positive, dims=(0,))
    threshold = descending[0] * config.eigenvalue_relative_floor
    temporal_rank = int((descending > threshold).sum().item())
    capped = min(config.max_rank, temporal_rank)
    cumulative = torch.cumsum(descending[:capped], dim=0) / total
    selected = (
        int(
            torch.searchsorted(
                cumulative, torch.tensor(config.energy_threshold, dtype=torch.float64)
            ).item()
        )
        + 1
    )
    selected = min(selected, capped)
    retained = float(cumulative[selected - 1].item())
    return selected, temporal_rank, retained, descending


def _polynomial_coefficients(
    steps: tuple[int, ...], values: Tensor, target_step: int, degree: int
) -> tuple[Tensor, float]:
    """Fit each coefficient on centered/scaled time to avoid an ill-conditioned Vandermonde."""

    if degree >= len(steps):
        raise ValueError("polynomial_degree must be smaller than the number of snapshots")
    times = torch.tensor(steps, dtype=torch.float64)
    center = times.mean()
    span = times[-1] - times[0]
    if span <= 0.0:
        raise ValueError("trajectory steps must span a positive interval")
    normalized = (times - center) / span
    target = torch.as_tensor((target_step - center) / span, dtype=torch.float64)
    design = torch.stack([normalized.pow(power) for power in range(degree + 1)], dim=1)
    solution = torch.linalg.lstsq(design, values.double()).solution
    target_row = torch.stack([target.pow(power) for power in range(degree + 1)])
    predicted = target_row @ solution
    condition = float(torch.linalg.cond(design).item())
    if not math.isfinite(condition):
        raise ValueError("coefficient fit is numerically singular")
    return predicted, condition


def low_rank_temporal_forecast(
    snapshots: tuple[WeightSnapshot, ...], *, target_step: int, config: LowRankForecastConfig
) -> ForecastResult:
    r"""Forecast every floating tensor through a per-tensor temporal Gram decomposition.

    For deltas \(D=[d_1,\ldots,d_n]\), this computes only \(D^T D\), an
    ``n x n`` matrix. Reconstruction is streamed as a weighted sum of existing
    tensor deltas, avoiding a 50M-by-n allocation or a giant flattened SVD.
    """

    assert_single_run_trajectory(snapshots)
    _require_target_step(snapshots, target_step)
    steps = tuple(snapshot.metadata.step for snapshot in snapshots)
    observed_span = steps[-1] - steps[0]
    future_span = target_step - steps[-1]
    if future_span > observed_span * config.max_extrapolation_ratio:
        raise ValueError("target step exceeds configured extrapolation horizon")
    result: dict[str, Tensor] = {}
    diagnostics: dict[str, TensorTrajectoryDiagnostics] = {}
    base_snapshot, latest_snapshot = snapshots[0], snapshots[-1]
    for name, base_tensor in base_snapshot.state_dict.items():
        latest_tensor = latest_snapshot.state_dict[name]
        if not base_tensor.is_floating_point():
            result[name] = latest_tensor.clone()
            diagnostics[name] = TensorTrajectoryDiagnostics(
                name, False, 0, 0, 1.0, (), None, unchanged=True
            )
            continue
        deltas = tuple(
            snapshot.state_dict[name].double() - base_tensor.double() for snapshot in snapshots
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(_temporal_gram(deltas))
        selected_rank, temporal_rank, retained_energy, descending = _select_rank(
            eigenvalues, config
        )
        singular_values = tuple(float(value.sqrt().item()) for value in descending[:temporal_rank])
        if selected_rank == 0:
            result[name] = latest_tensor.clone()
            diagnostics[name] = TensorTrajectoryDiagnostics(
                name, True, 0, 0, 1.0, singular_values, None, unchanged=True
            )
            continue
        selected_values = torch.flip(eigenvalues, dims=(0,))[:selected_rank].clamp_min(0.0)
        selected_vectors = torch.flip(eigenvectors, dims=(1,))[:, :selected_rank]
        coefficients = selected_vectors * selected_values.sqrt().unsqueeze(0)
        future_coefficients, condition = _polynomial_coefficients(
            steps, coefficients, target_step, config.polynomial_degree
        )
        weights = selected_vectors @ (future_coefficients / selected_values.sqrt())
        predicted_delta = torch.zeros_like(base_tensor, dtype=torch.float64)
        for delta, weight in zip(deltas, weights, strict=True):
            predicted_delta.add_(delta, alpha=float(weight.item()))
        result[name] = (base_tensor.double() + predicted_delta).to(dtype=latest_tensor.dtype)
        diagnostics[name] = TensorTrajectoryDiagnostics(
            name=name,
            floating_point=True,
            selected_rank=selected_rank,
            temporal_rank=temporal_rank,
            retained_energy=retained_energy,
            singular_values=singular_values,
            coefficient_condition=condition,
        )
    return ForecastResult(
        method="low_rank_temporal_gram",
        source_checkpoint_ids=tuple(snapshot.metadata.checkpoint_id for snapshot in snapshots),
        source_steps=steps,
        target_step=target_step,
        state_dict=result,
        diagnostics=diagnostics,
    )
