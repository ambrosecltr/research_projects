"""Validated configuration for reproducible training runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
PrecisionPreference = Literal["auto", "none", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    max_steps: int
    learning_rate: float
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    warmup_steps: int = 0
    min_learning_rate_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    device: DevicePreference = "auto"
    precision: PrecisionPreference = "auto"
    seed: int = 1337
    deterministic: bool = True
    log_every_steps: int = 1
    checkpoint_every_steps: int = 0
    trajectory_every_steps: int = 0
    checkpoint_steps: tuple[int, ...] = ()
    trajectory_capture_steps: tuple[int, ...] = ()
    analysis_every_steps: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "warmup_steps",
            "gradient_accumulation_steps",
            "seed",
            "log_every_steps",
            "checkpoint_every_steps",
            "trajectory_every_steps",
            "analysis_every_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        for field_name in (
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "min_learning_rate_ratio",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.max_grad_norm is not None:
            if isinstance(self.max_grad_norm, bool) or not isinstance(
                self.max_grad_norm, (int, float)
            ):
                raise TypeError("max_grad_norm must be a finite number when set")
            if not math.isfinite(self.max_grad_norm):
                raise ValueError("max_grad_norm must be finite when set")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if self.precision not in {"auto", "none", "float16", "bfloat16"}:
            raise ValueError("precision must be auto, none, float16, or bfloat16")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must lie in [0, max_steps)")
        if not 0 <= self.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must lie in [0, 1]")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be at least 1")
        if (
            min(self.checkpoint_every_steps, self.trajectory_every_steps, self.analysis_every_steps)
            < 0
        ):
            raise ValueError("checkpoint, trajectory, and analysis intervals cannot be negative")
        self._validate_capture_steps("checkpoint_steps", self.checkpoint_steps)
        self._validate_capture_steps("trajectory_capture_steps", self.trajectory_capture_steps)

    def _validate_capture_steps(self, name: str, steps: tuple[int, ...]) -> None:
        if not isinstance(steps, tuple):
            raise TypeError(f"{name} must be a tuple of integer steps")
        if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
            raise TypeError(f"{name} must contain only integer steps")
        if tuple(sorted(set(steps))) != steps:
            raise ValueError(f"{name} must be unique and sorted")
        if any(not 1 <= step <= self.max_steps for step in steps):
            raise ValueError(f"{name} steps must lie in [1, max_steps]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TrainConfig:
        parameters = cast(dict[str, Any], dict(values))
        for name in ("checkpoint_steps", "trajectory_capture_steps"):
            if name in parameters and isinstance(parameters[name], list):
                parameters[name] = tuple(parameters[name])
        return cls(**parameters)
