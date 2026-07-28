"""Model-aware preparation of forecasts before they are evaluated or applied."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from poetry50m.model.transformer import UnitEmbedding, UnitLinear
from poetry50m.trajectory.forecast import ForecastResult


@dataclass(frozen=True, slots=True)
class PreparedForecast:
    """The sole state dict eligible for safety checks, verification, and application."""

    forecast: ForecastResult
    state_dict: Mapping[str, Tensor]
    retracted_axes: Mapping[str, int]
    state_hash: str


def declared_normalization_axes(module: nn.Module) -> dict[str, int]:
    """Return every weight coordinate represented on nGPT's product of spheres."""

    axes: dict[str, int] = {}
    for module_name, child in module.named_modules():
        if isinstance(child, UnitEmbedding):
            key = f"{module_name}.weight" if module_name else "weight"
            axes[key] = -1
        elif isinstance(child, UnitLinear):
            key = f"{module_name}.weight" if module_name else "weight"
            axes[key] = child.normalization_axis
    return axes


def state_dict_hash(state_dict: Mapping[str, Tensor]) -> str:
    """Return a stable content hash for an ordered model state dict."""

    digest = sha256()
    for name, value in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(value.shape)).encode("ascii"))
        raw_bytes = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
    return digest.hexdigest()


def prepare_forecast(module: nn.Module, forecast: ForecastResult) -> PreparedForecast:
    """Retract nGPT's normalized vectors before *any* candidate measurement."""

    current = module.state_dict()
    if tuple(current) != tuple(forecast.state_dict):
        raise ValueError("forecast state_dict names or order do not match the live module")
    axes = declared_normalization_axes(module)
    prepared: dict[str, Tensor] = {}
    for name, current_tensor in current.items():
        predicted = forecast.state_dict[name]
        if current_tensor.shape != predicted.shape or current_tensor.dtype != predicted.dtype:
            raise ValueError(f"forecast tensor {name} does not match the live model coordinate")
        value = predicted.detach().to(device=current_tensor.device, dtype=current_tensor.dtype)
        if name in axes:
            owner_name = name.rsplit(".", 1)[0]
            owner = module.get_submodule(owner_name) if owner_name else module
            epsilon = getattr(owner, "epsilon", None)
            if not isinstance(epsilon, float) or epsilon <= 0.0:
                raise ValueError(f"normalized module for {name} has no positive epsilon")
            value = F.normalize(value, p=2.0, dim=axes[name], eps=epsilon)
        prepared[name] = value
    return PreparedForecast(
        forecast=forecast,
        state_dict=prepared,
        retracted_axes=axes,
        state_hash=state_dict_hash(prepared),
    )
