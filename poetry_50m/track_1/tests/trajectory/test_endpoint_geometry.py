from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from poetry50m.model.transformer import UnitLinear
from poetry50m.trajectory.endpoint_geometry import (
    EVIDENCE_LABEL,
    EndpointGeometryReport,
    analyze_endpoint_geometry,
    analyze_endpoint_geometry_paths,
)
from poetry50m.trajectory.snapshots import save_weight_snapshot

from .conftest import make_snapshot


def _snapshots(*states: torch.Tensor, steps: tuple[int, ...] | None = None):
    resolved_steps = steps if steps is not None else tuple(range(len(states)))
    return tuple(
        make_snapshot(checkpoint_id=f"s{step}", step=step, state_dict={"blocks.0.weight": state})
        for step, state in zip(resolved_steps, states, strict=True)
    )


def _metric(report: EndpointGeometryReport, *, scope: str, name: str):
    return next(
        metric for metric in report.metrics if metric.scope == scope and metric.name == name
    )


def test_endpoint_geometry_recovers_collinear_endpoint_extrapolation() -> None:
    report = analyze_endpoint_geometry(
        _snapshots(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0]))
    )

    global_metrics = _metric(report, scope="global", name="global")
    assert global_metrics.initial_to_early_norm == pytest.approx(1.0)
    assert global_metrics.early_to_endpoint_norm == pytest.approx(2.0)
    assert global_metrics.initial_to_endpoint_norm == pytest.approx(3.0)
    assert global_metrics.cosine_early_remaining == pytest.approx(1.0)
    assert global_metrics.cosine_early_endpoint == pytest.approx(1.0)
    assert global_metrics.endpoint_scale_from_initial == pytest.approx(3.0)
    assert global_metrics.remaining_scale_from_early == pytest.approx(2.0)
    assert global_metrics.endpoint_prediction_residual_norm == pytest.approx(0.0)
    assert global_metrics.endpoint_prediction_residual_ratio == pytest.approx(0.0)
    assert global_metrics.endpoint_projection_progress_fraction == pytest.approx(1.0 / 3.0)
    assert _metric(report, scope="module_prefix", name="blocks.0").tensor_count == 1
    assert _metric(
        report, scope="tensor", name="blocks.0.weight"
    ).initial_to_endpoint_norm == pytest.approx(global_metrics.initial_to_endpoint_norm)
    assert report.endpoint_informed is True
    assert report.evidence_label == EVIDENCE_LABEL
    assert report.angular_geometry.available is False
    assert report.angular_geometry.absence_reason is not None


def test_endpoint_geometry_exposes_orthogonal_residual_and_zero_direction_without_nans() -> None:
    orthogonal = analyze_endpoint_geometry(
        _snapshots(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 2.0]))
    )
    metrics = _metric(orthogonal, scope="global", name="global")
    assert metrics.cosine_early_remaining == pytest.approx(0.0)
    assert metrics.endpoint_scale_from_initial == pytest.approx(1.0)
    assert metrics.remaining_scale_from_early == pytest.approx(0.0)
    assert metrics.endpoint_prediction_residual_norm == pytest.approx(2.0)
    assert metrics.endpoint_prediction_residual_ratio == pytest.approx(2.0 / math.sqrt(5.0))
    assert metrics.endpoint_projection_progress_fraction == pytest.approx(0.2)

    zero_early = analyze_endpoint_geometry(
        _snapshots(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([2.0]))
    )
    zero_metrics = _metric(zero_early, scope="global", name="global")
    assert zero_metrics.cosine_early_remaining is None
    assert zero_metrics.cosine_early_endpoint is None
    assert zero_metrics.endpoint_scale_from_initial is None
    assert zero_metrics.remaining_scale_from_early is None
    assert zero_metrics.endpoint_prediction_residual_norm is None
    assert zero_metrics.endpoint_prediction_residual_ratio is None
    assert zero_metrics.endpoint_projection_progress_fraction is None
    assert "NaN" not in zero_early.to_json()
    assert "Infinity" not in zero_early.to_json()


def test_endpoint_geometry_reports_irregular_interval_turning() -> None:
    report = analyze_endpoint_geometry(
        _snapshots(
            torch.tensor([0.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([5.0, 0.0]),
            torch.tensor([5.0, 4.0]),
            steps=(0, 2, 5, 9),
        )
    )

    turning = _metric(report, scope="global", name="global").turning
    assert turning is not None
    assert len(turning.intervals) == 2
    assert turning.intervals[0].velocity_cosine == pytest.approx(1.0)
    assert turning.intervals[0].acceleration_norm == pytest.approx(0.0)
    assert turning.intervals[1].velocity_cosine == pytest.approx(0.0)
    assert turning.intervals[1].acceleration_norm == pytest.approx(math.sqrt(2.0) / 3.5)
    assert turning.mean_turning_angle_radians == pytest.approx(math.pi / 4.0)


def test_endpoint_geometry_uses_declared_ngpt_axes_without_inference() -> None:
    model = UnitLinear(2, 2, epsilon=1e-6, normalization_axis=1)
    report = analyze_endpoint_geometry(
        tuple(
            make_snapshot(
                checkpoint_id=f"s{step}",
                step=step,
                state_dict={"weight": value},
            )
            for step, value in (
                (0, torch.tensor([[1.0, 0.0], [1.0, 0.0]])),
                (1, torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
                (2, torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])),
            )
        ),
        model=model,
    )

    angular = report.angular_geometry
    assert angular.available is True
    assert angular.absence_reason is None
    assert angular.tensors[0].normalization_axis == 1
    assert angular.initial_to_early_mean_angle_radians == pytest.approx(math.pi / 2.0)
    assert angular.initial_to_endpoint_mean_angle_radians == pytest.approx(math.pi)


def test_endpoint_geometry_handles_ngpt_column_normalization_axis() -> None:
    model = UnitLinear(2, 2, epsilon=1e-6, normalization_axis=0)
    report = analyze_endpoint_geometry(
        tuple(
            make_snapshot(
                checkpoint_id=f"s{step}",
                step=step,
                state_dict={"weight": value},
            )
            for step, value in (
                (0, torch.tensor([[1.0, 1.0], [0.0, 0.0]])),
                (1, torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
                (2, torch.tensor([[-1.0, -1.0], [0.0, 0.0]])),
            )
        ),
        model=model,
    )

    angular = report.angular_geometry
    assert angular.available is True
    assert angular.tensors[0].normalization_axis == 0
    assert angular.tensors[0].vector_count == 2
    assert angular.initial_to_early_mean_angle_radians == pytest.approx(math.pi / 2.0)
    assert angular.initial_to_endpoint_mean_angle_radians == pytest.approx(math.pi)


def test_endpoint_geometry_safe_paths_round_trip_and_reject_mismatch(tmp_path: Path) -> None:
    snapshots = _snapshots(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]))
    paths = tuple(tmp_path / f"{snapshot.metadata.checkpoint_id}.pt" for snapshot in snapshots)
    for path, snapshot in zip(paths, snapshots, strict=True):
        save_weight_snapshot(path, snapshot)

    report = analyze_endpoint_geometry_paths(paths)
    assert EndpointGeometryReport.from_json(report.to_json()) == report
    assert report.to_json() == report.to_json()
    assert (
        json.loads(report.to_json())["snapshots"][0]["state_dict_hash"]
        == report.snapshots[0].state_dict_hash
    )

    mismatched = make_snapshot(
        checkpoint_id="s3", step=3, state_dict={"blocks.0.weight": torch.zeros(2)}
    )
    mismatch_path = tmp_path / "mismatch.pt"
    save_weight_snapshot(mismatch_path, mismatched)
    with pytest.raises(ValueError, match="shape"):
        analyze_endpoint_geometry_paths((paths[0], paths[1], mismatch_path))


def test_endpoint_geometry_report_rejects_nonfinite_and_bool_provenance() -> None:
    report = analyze_endpoint_geometry(
        _snapshots(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]))
    )
    invalid_step = report.to_mapping()
    snapshots = invalid_step["snapshots"]
    assert isinstance(snapshots, list)
    snapshots[0]["step"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        EndpointGeometryReport.from_mapping(invalid_step)

    invalid_number = report.to_mapping()
    metrics = invalid_number["metrics"]
    assert isinstance(metrics, list)
    metrics[0]["initial_to_early_norm"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        EndpointGeometryReport.from_mapping(invalid_number)
    with pytest.raises(ValueError, match="invalid numeric constant"):
        EndpointGeometryReport.from_json(report.to_json().replace("1.0", "NaN", 1))


@pytest.mark.parametrize(
    "snapshots, message",
    (
        (
            _snapshots(
                torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]), steps=(1, 2, 3)
            ),
            "step-zero",
        ),
        (
            _snapshots(
                torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]), steps=(0, 1, 1)
            ),
            "strictly increasing",
        ),
    ),
)
def test_endpoint_geometry_rejects_invalid_trajectory(snapshots: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_endpoint_geometry(snapshots)
