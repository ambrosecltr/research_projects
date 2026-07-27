from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .types import TensorSpec


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    state: Mapping[str, torch.Tensor]
    loss: float | None = None
    learning_rate: float | None = None


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if denominator <= 1e-30:
        return 0.0
    return float(torch.dot(a.flatten(), b.flatten()).item() / denominator)


def extract_trajectory_features(
    points: Sequence[TrajectoryPoint],
    tensor_specs: Sequence[TensorSpec],
) -> dict[str, torch.Tensor | list[str]]:
    if len(points) < 2:
        raise ValueError("trajectory features require at least two checkpoints")
    ordered = sorted(points, key=lambda point: point.step)
    names = [spec.name for spec in tensor_specs]
    rows = []
    for point_index, point in enumerate(ordered):
        previous = ordered[max(point_index - 1, 0)]
        initial = ordered[0]
        features = []
        for spec in tensor_specs:
            current_tensor = point.state[spec.name].detach().to(torch.float32).cpu()
            initial_delta = current_tensor - initial.state[spec.name].detach().to(torch.float32).cpu()
            step_delta = current_tensor - previous.state[spec.name].detach().to(torch.float32).cpu()
            previous_delta = (
                previous.state[spec.name].detach().to(torch.float32).cpu()
                - ordered[max(point_index - 2, 0)].state[spec.name].detach().to(torch.float32).cpu()
            )
            features.append(
                [
                    float(torch.linalg.vector_norm(current_tensor).item()),
                    float(torch.linalg.vector_norm(initial_delta).item()),
                    float(torch.linalg.vector_norm(step_delta).item()),
                    _cosine(step_delta, previous_delta) if point_index >= 2 else 0.0,
                    float(current_tensor.mean().item()),
                    float(current_tensor.std(unbiased=False).item()) if current_tensor.numel() > 1 else 0.0,
                ]
            )
        rows.append(features)
    tensor_features = torch.tensor(rows, dtype=torch.float32)
    global_features = torch.tensor(
        [
            [
                float(point.step),
                float(point.loss) if point.loss is not None else float("nan"),
                float(point.learning_rate) if point.learning_rate is not None else float("nan"),
            ]
            for point in ordered
        ],
        dtype=torch.float32,
    )
    return {
        "tensor_names": names,
        "tensor_features": tensor_features,
        "global_features": global_features,
        "steps": torch.tensor([point.step for point in ordered], dtype=torch.int64),
    }


def extrapolate_endpoint_in_code_space(
    codes: torch.Tensor,
    steps: torch.Tensor,
    target_step: int,
    *,
    fit_points: int = 3,
) -> torch.Tensor:
    """Transparent linear baseline required before a temporal transformer."""
    if codes.ndim < 2:
        raise ValueError("codes must have shape [time, ...]")
    if codes.shape[0] != steps.numel():
        raise ValueError("codes and steps have different time dimensions")
    count = min(fit_points, steps.numel())
    x = steps[-count:].to(torch.float64)
    y = codes[-count:].to(torch.float64).reshape(count, -1)
    x_centered = x - x.mean()
    denominator = torch.sum(x_centered.square()).clamp_min(1e-12)
    slope = torch.sum(x_centered[:, None] * (y - y.mean(dim=0)), dim=0) / denominator
    intercept_at_mean = y.mean(dim=0)
    prediction = intercept_at_mean + (float(target_step) - float(x.mean().item())) * slope
    return prediction.reshape(codes.shape[1:]).to(codes.dtype)
