"""Numerical endpoint geometry calculations over validated weight snapshots."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from poetry50m.trajectory.endpoint_schema import (
    EVIDENCE_LABEL,
    FORMULA_DEFINITIONS,
    METHOD,
    SCHEMA_VERSION,
    AngularGeometrySummary,
    AngularTensorMetrics,
    ConsecutiveDeltaGeometry,
    EndpointGeometryReport,
    GeometryMetrics,
    SnapshotProvenance,
    TurningSummary,
)
from poetry50m.trajectory.preparation import declared_normalization_axes
from poetry50m.trajectory.snapshots import assert_single_run_trajectory, load_weight_snapshot
from poetry50m.trajectory.types import WeightSnapshot

_CHUNK_ELEMENTS = 1_000_000


@dataclass(slots=True)
class _EndpointSums:
    early_squared_norm: float = 0.0
    remaining_squared_norm: float = 0.0
    early_remaining_dot: float = 0.0

    def add(self, other: _EndpointSums) -> None:
        self.early_squared_norm += other.early_squared_norm
        self.remaining_squared_norm += other.remaining_squared_norm
        self.early_remaining_dot += other.early_remaining_dot


@dataclass(slots=True)
class _TurningSums:
    first_velocity_squared_norm: float = 0.0
    second_velocity_squared_norm: float = 0.0
    velocity_dot: float = 0.0
    acceleration_squared_norm: float = 0.0

    def add(self, other: _TurningSums) -> None:
        self.first_velocity_squared_norm += other.first_velocity_squared_norm
        self.second_velocity_squared_norm += other.second_velocity_squared_norm
        self.velocity_dot += other.velocity_dot
        self.acceleration_squared_norm += other.acceleration_squared_norm


@dataclass(slots=True)
class _MetricAccumulator:
    tensor_count: int = 0
    parameter_count: int = 0
    endpoint: _EndpointSums | None = None
    turning: list[_TurningSums] | None = None

    def add(
        self, parameter_count: int, endpoint: _EndpointSums, turning: list[_TurningSums]
    ) -> None:
        self.tensor_count += 1
        self.parameter_count += parameter_count
        if self.endpoint is None:
            self.endpoint = _EndpointSums()
        self.endpoint.add(endpoint)
        if self.turning is None:
            self.turning = [_TurningSums() for _ in turning]
        if len(self.turning) != len(turning):
            raise RuntimeError("inconsistent trajectory turning intervals")
        for target, source in zip(self.turning, turning, strict=True):
            target.add(source)


def _validate_tensor(tensor: Tensor, *, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(
            f"floating tensor {name} must be on CPU; load a weights-only snapshot first"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"floating tensor {name} must be contiguous for bounded streaming analysis"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"floating tensor {name} contains non-finite values")


def _tensor_endpoint_sums(initial: Tensor, early: Tensor, endpoint: Tensor) -> _EndpointSums:
    result = _EndpointSums()
    initial_flat, early_flat, endpoint_flat = initial.view(-1), early.view(-1), endpoint.view(-1)
    for offset in range(0, initial_flat.numel(), _CHUNK_ELEMENTS):
        count = min(_CHUNK_ELEMENTS, initial_flat.numel() - offset)
        start = initial_flat.narrow(0, offset, count).to(dtype=torch.float64)
        middle = early_flat.narrow(0, offset, count).to(dtype=torch.float64)
        finish = endpoint_flat.narrow(0, offset, count).to(dtype=torch.float64)
        early_delta = middle - start
        remaining_delta = finish - middle
        result.early_squared_norm += float(torch.sum(early_delta.square()).item())
        result.remaining_squared_norm += float(torch.sum(remaining_delta.square()).item())
        result.early_remaining_dot += float(torch.sum(early_delta * remaining_delta).item())
    return result


def _tensor_turning_sums(snapshots: tuple[WeightSnapshot, ...], name: str) -> list[_TurningSums]:
    if len(snapshots) < 4:
        return []
    values = tuple(snapshot.state_dict[name].view(-1) for snapshot in snapshots)
    steps = tuple(snapshot.metadata.step for snapshot in snapshots)
    results = [_TurningSums() for _ in range(len(snapshots) - 2)]
    for offset in range(0, values[0].numel(), _CHUNK_ELEMENTS):
        count = min(_CHUNK_ELEMENTS, values[0].numel() - offset)
        for index, result in enumerate(results):
            first_interval = steps[index + 1] - steps[index]
            second_interval = steps[index + 2] - steps[index + 1]
            first = values[index].narrow(0, offset, count).to(dtype=torch.float64)
            middle = values[index + 1].narrow(0, offset, count).to(dtype=torch.float64)
            finish = values[index + 2].narrow(0, offset, count).to(dtype=torch.float64)
            first_velocity = (middle - first) / first_interval
            second_velocity = (finish - middle) / second_interval
            acceleration = (second_velocity - first_velocity) / (
                (first_interval + second_interval) / 2.0
            )
            result.first_velocity_squared_norm += float(torch.sum(first_velocity.square()).item())
            result.second_velocity_squared_norm += float(torch.sum(second_velocity.square()).item())
            result.velocity_dot += float(torch.sum(first_velocity * second_velocity).item())
            result.acceleration_squared_norm += float(torch.sum(acceleration.square()).item())
    return results


def _sqrt_nonnegative(value: float, *, name: str, scale: float) -> float:
    if value < 0.0:
        if value >= -1e-12 * max(1.0, scale):
            value = 0.0
        else:
            raise ValueError(f"{name} became negative; endpoint geometry is numerically invalid")
    result = math.sqrt(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _cosine(dot: float, left_squared_norm: float, right_squared_norm: float) -> float | None:
    if left_squared_norm == 0.0 or right_squared_norm == 0.0:
        return None
    value = dot / (math.sqrt(left_squared_norm) * math.sqrt(right_squared_norm))
    if not math.isfinite(value):
        raise ValueError("endpoint geometry cosine is not finite")
    return max(-1.0, min(1.0, value))


def _turning_summary(sums: list[_TurningSums], steps: tuple[int, ...]) -> TurningSummary | None:
    if not sums:
        return None
    intervals: list[ConsecutiveDeltaGeometry] = []
    for index, value in enumerate(sums):
        cosine = _cosine(
            value.velocity_dot,
            value.first_velocity_squared_norm,
            value.second_velocity_squared_norm,
        )
        acceleration_norm = _sqrt_nonnegative(
            value.acceleration_squared_norm,
            name="acceleration squared norm",
            scale=value.first_velocity_squared_norm + value.second_velocity_squared_norm,
        )
        intervals.append(
            ConsecutiveDeltaGeometry(
                start_step=steps[index],
                middle_step=steps[index + 1],
                end_step=steps[index + 2],
                velocity_cosine=cosine,
                acceleration_norm=acceleration_norm,
            )
        )
    cosines = [
        interval.velocity_cosine for interval in intervals if interval.velocity_cosine is not None
    ]
    angles = [math.acos(value) for value in cosines]
    return TurningSummary(
        intervals=tuple(intervals),
        mean_velocity_cosine=sum(cosines) / len(cosines) if cosines else None,
        mean_turning_angle_radians=sum(angles) / len(angles) if angles else None,
        mean_acceleration_norm=sum(interval.acceleration_norm for interval in intervals)
        / len(intervals),
    )


def _geometry_metrics(
    *, name: str, scope: str, accumulator: _MetricAccumulator, steps: tuple[int, ...]
) -> GeometryMetrics:
    endpoint = accumulator.endpoint
    if endpoint is None:
        raise RuntimeError("geometry accumulator has no endpoint sums")
    early_squared_norm = endpoint.early_squared_norm
    remaining_squared_norm = endpoint.remaining_squared_norm
    full_squared_norm = (
        early_squared_norm + remaining_squared_norm + 2.0 * endpoint.early_remaining_dot
    )
    early_norm = _sqrt_nonnegative(
        early_squared_norm, name="early squared norm", scale=early_squared_norm
    )
    remaining_norm = _sqrt_nonnegative(
        remaining_squared_norm, name="remaining squared norm", scale=remaining_squared_norm
    )
    full_norm = _sqrt_nonnegative(
        full_squared_norm,
        name="endpoint squared norm",
        scale=early_squared_norm + remaining_squared_norm,
    )
    early_full_dot = early_squared_norm + endpoint.early_remaining_dot
    if early_squared_norm == 0.0:
        endpoint_scale = remaining_scale = residual_norm = residual_ratio = None
    else:
        endpoint_scale = early_full_dot / early_squared_norm
        remaining_scale = endpoint.early_remaining_dot / early_squared_norm
        residual_squared_norm = (
            full_squared_norm - early_full_dot * early_full_dot / early_squared_norm
        )
        residual_norm = _sqrt_nonnegative(
            residual_squared_norm,
            name="endpoint prediction residual squared norm",
            scale=full_squared_norm,
        )
        residual_ratio = residual_norm / full_norm if full_norm else None
    progress = (
        early_full_dot / full_squared_norm if early_squared_norm and full_squared_norm else None
    )
    return GeometryMetrics(
        name=name,
        scope=scope,
        tensor_count=accumulator.tensor_count,
        parameter_count=accumulator.parameter_count,
        initial_to_early_norm=early_norm,
        early_to_endpoint_norm=remaining_norm,
        initial_to_endpoint_norm=full_norm,
        cosine_early_remaining=_cosine(
            endpoint.early_remaining_dot, early_squared_norm, remaining_squared_norm
        ),
        cosine_early_endpoint=_cosine(early_full_dot, early_squared_norm, full_squared_norm),
        endpoint_scale_from_initial=endpoint_scale,
        remaining_scale_from_early=remaining_scale,
        endpoint_prediction_residual_norm=residual_norm,
        endpoint_prediction_residual_ratio=residual_ratio,
        endpoint_projection_progress_fraction=progress,
        turning=_turning_summary(accumulator.turning or [], steps),
    )


def _module_prefixes(name: str) -> tuple[str, ...]:
    parts = name.split(".")
    return tuple(".".join(parts[:index]) for index in range(1, len(parts)))


def _mean_angle(left: Tensor, right: Tensor) -> tuple[float | None, int]:
    left_norm, right_norm = (
        torch.linalg.vector_norm(left, dim=1),
        torch.linalg.vector_norm(right, dim=1),
    )
    valid = (left_norm > 0.0) & (right_norm > 0.0)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return None, 0
    cosine = (left[valid] * right[valid]).sum(dim=1) / (left_norm[valid] * right_norm[valid])
    return float(torch.acos(cosine.clamp(-1.0, 1.0)).sum().item()), valid_count


def _angular_tensor_metrics(
    initial: Tensor, early: Tensor, endpoint: Tensor, *, name: str, axis: int
) -> AngularTensorMetrics:
    normalized_axis = axis if axis >= 0 else initial.ndim + axis
    if normalized_axis < 0 or normalized_axis >= initial.ndim:
        raise ValueError(f"declared normalization axis {axis} is invalid for {name}")
    vector_width = initial.shape[normalized_axis]
    if vector_width == 0:
        raise ValueError(f"declared normalization tensor {name} has an empty vector axis")
    initial_vectors = initial.movedim(normalized_axis, -1).reshape(-1, vector_width)
    early_vectors = early.movedim(normalized_axis, -1).reshape(-1, vector_width)
    endpoint_vectors = endpoint.movedim(normalized_axis, -1).reshape(-1, vector_width)
    sums, counts = [0.0, 0.0, 0.0], [0, 0, 0]
    chunk_vectors = max(1, _CHUNK_ELEMENTS // vector_width)
    for offset in range(0, initial_vectors.shape[0], chunk_vectors):
        count = min(chunk_vectors, initial_vectors.shape[0] - offset)
        vectors = tuple(
            vector.narrow(0, offset, count).to(dtype=torch.float64)
            for vector in (initial_vectors, early_vectors, endpoint_vectors)
        )
        for index, (left, right) in enumerate(
            ((vectors[0], vectors[1]), (vectors[1], vectors[2]), (vectors[0], vectors[2]))
        ):
            total, valid_count = _mean_angle(left, right)
            if total is not None:
                sums[index] += total
                counts[index] += valid_count
    return AngularTensorMetrics(
        name=name,
        normalization_axis=axis,
        vector_count=initial_vectors.shape[0],
        initial_to_early_mean_angle_radians=sums[0] / counts[0] if counts[0] else None,
        early_to_endpoint_mean_angle_radians=sums[1] / counts[1] if counts[1] else None,
        initial_to_endpoint_mean_angle_radians=sums[2] / counts[2] if counts[2] else None,
    )


def _angular_geometry(
    snapshots: tuple[WeightSnapshot, ...], model: nn.Module | None
) -> AngularGeometrySummary:
    if model is None:
        return AngularGeometrySummary(
            False,
            "no matching model instance supplied; normalization axes were not inferred",
            (),
            None,
            None,
            None,
        )
    model_state, initial = model.state_dict(), snapshots[0]
    if tuple(model_state) != tuple(initial.state_dict):
        raise ValueError("model state_dict does not match endpoint-analysis snapshot coordinates")
    axes = declared_normalization_axes(model)
    if not axes:
        return AngularGeometrySummary(
            False, "matching model declares no nGPT normalization axes", (), None, None, None
        )
    results: list[AngularTensorMetrics] = []
    for name, axis in sorted(axes.items()):
        model_tensor, snapshot_tensor = model_state.get(name), initial.state_dict.get(name)
        if model_tensor is None or snapshot_tensor is None:
            raise ValueError(
                f"declared normalization tensor {name} is absent from snapshot coordinates"
            )
        if (
            model_tensor.shape != snapshot_tensor.shape
            or model_tensor.dtype != snapshot_tensor.dtype
        ):
            raise ValueError(
                f"declared normalization tensor {name} does not match snapshot coordinates"
            )
        results.append(
            _angular_tensor_metrics(
                initial.state_dict[name],
                snapshots[1].state_dict[name],
                snapshots[-1].state_dict[name],
                name=name,
                axis=axis,
            )
        )

    def weighted(values: list[tuple[float | None, int]]) -> float | None:
        present = [(value, count) for value, count in values if value is not None]
        if not present:
            return None
        return sum(value * count for value, count in present) / sum(count for _, count in present)

    return AngularGeometrySummary(
        True,
        None,
        tuple(results),
        weighted(
            [
                (result.initial_to_early_mean_angle_radians, result.vector_count)
                for result in results
            ]
        ),
        weighted(
            [
                (result.early_to_endpoint_mean_angle_radians, result.vector_count)
                for result in results
            ]
        ),
        weighted(
            [
                (result.initial_to_endpoint_mean_angle_radians, result.vector_count)
                for result in results
            ]
        ),
    )


def analyze_endpoint_geometry(
    snapshots: Sequence[WeightSnapshot], *, model: nn.Module | None = None
) -> EndpointGeometryReport:
    """Compare initial, early, and observed endpoint weights as offline teacher evidence."""

    values = tuple(snapshots)
    if len(values) < 3:
        raise ValueError("endpoint geometry requires initial, early, and endpoint snapshots")
    assert_single_run_trajectory(values)
    if values[0].metadata.step != 0:
        raise ValueError(
            "endpoint geometry requires the first snapshot to be the step-zero initialization"
        )
    floating_names = tuple(
        name for name, tensor in values[0].state_dict.items() if tensor.is_floating_point()
    )
    if not floating_names:
        raise ValueError("endpoint geometry requires at least one floating tensor")
    for snapshot in values:
        for name in floating_names:
            _validate_tensor(snapshot.state_dict[name], name=name)
    global_accumulator = _MetricAccumulator()
    module_accumulators: dict[str, _MetricAccumulator] = {}
    tensor_accumulators: dict[str, _MetricAccumulator] = {}
    for name in sorted(floating_names):
        endpoint_sums = _tensor_endpoint_sums(
            values[0].state_dict[name], values[1].state_dict[name], values[-1].state_dict[name]
        )
        turning_sums = _tensor_turning_sums(values, name)
        parameter_count = values[0].state_dict[name].numel()
        global_accumulator.add(parameter_count, endpoint_sums, turning_sums)
        tensor_accumulators[name] = _MetricAccumulator()
        tensor_accumulators[name].add(parameter_count, endpoint_sums, turning_sums)
        for prefix in _module_prefixes(name):
            module_accumulators.setdefault(prefix, _MetricAccumulator()).add(
                parameter_count, endpoint_sums, turning_sums
            )
    steps = tuple(snapshot.metadata.step for snapshot in values)
    metrics = [
        _geometry_metrics(
            name="global", scope="global", accumulator=global_accumulator, steps=steps
        )
    ]
    metrics.extend(
        _geometry_metrics(
            name=name, scope="module_prefix", accumulator=module_accumulators[name], steps=steps
        )
        for name in sorted(module_accumulators)
    )
    metrics.extend(
        _geometry_metrics(
            name=name, scope="tensor", accumulator=tensor_accumulators[name], steps=steps
        )
        for name in sorted(tensor_accumulators)
    )
    metrics[1:] = sorted(metrics[1:], key=lambda metric: (metric.scope, metric.name))
    return EndpointGeometryReport(
        schema_version=SCHEMA_VERSION,
        method=METHOD,
        endpoint_informed=True,
        evidence_label=EVIDENCE_LABEL,
        snapshots=tuple(SnapshotProvenance.from_snapshot(snapshot) for snapshot in values),
        formula_definitions=FORMULA_DEFINITIONS,
        excluded_non_floating_tensor_names=tuple(
            sorted(
                name
                for name, tensor in values[0].state_dict.items()
                if not tensor.is_floating_point()
            )
        ),
        metrics=tuple(metrics),
        angular_geometry=_angular_geometry(values, model),
    )


def analyze_endpoint_geometry_paths(
    paths: Sequence[Path], *, model: nn.Module | None = None
) -> EndpointGeometryReport:
    """Safely load weights-only artifacts before endpoint analysis."""

    if len(paths) < 3:
        raise ValueError("endpoint geometry requires at least three snapshot paths")
    first = load_weight_snapshot(paths[0])
    snapshots = (first,) + tuple(load_weight_snapshot(path, expected=first) for path in paths[1:])
    return analyze_endpoint_geometry(snapshots, model=model)
