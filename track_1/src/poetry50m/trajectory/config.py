"""JSON configuration for the first verified trajectory branch."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from poetry50m.trajectory._persistence import load_json_object
from poetry50m.trajectory.forecast import LinearForecastConfig, LowRankForecastConfig
from poetry50m.trajectory.gates import CandidateAcceptanceGates, ForecastSafetyConfig


def _section(value: object, *, name: str, expected: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"trajectory config {name} must be a mapping with string keys")
    if set(value) != expected:
        raise ValueError(f"trajectory config {name} must contain exactly {sorted(expected)}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"trajectory config {name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"trajectory config {name} must be finite")
    return number


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"trajectory config {name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    linear: LinearForecastConfig
    low_rank: LowRankForecastConfig
    safety: ForecastSafetyConfig
    gates: CandidateAcceptanceGates

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TrajectoryConfig:
        expected = {"linear", "low_rank", "safety", "gates"}
        if set(value) != expected:
            raise ValueError(f"trajectory config must contain exactly {sorted(expected)}")
        linear = _section(value["linear"], name="linear", expected={"max_extrapolation_ratio"})
        low_rank = _section(
            value["low_rank"],
            name="low_rank",
            expected={
                "max_rank",
                "energy_threshold",
                "polynomial_degree",
                "max_extrapolation_ratio",
                "eigenvalue_relative_floor",
            },
        )
        safety = _section(
            value["safety"],
            name="safety",
            expected={"max_relative_tensor_norm", "min_relative_tensor_norm", "near_zero_norm"},
        )
        gates = _section(
            value["gates"],
            name="gates",
            expected={
                "max_verification_loss_increase",
                "max_post_leap_loss_increase",
                "max_anchor_mse",
                "max_anchor_cosine_distance",
                "max_anchor_symmetric_kl",
            },
        )
        return cls(
            linear=LinearForecastConfig(
                max_extrapolation_ratio=_number(
                    linear["max_extrapolation_ratio"], name="linear.max_extrapolation_ratio"
                )
            ),
            low_rank=LowRankForecastConfig(
                max_rank=_integer(low_rank["max_rank"], name="low_rank.max_rank"),
                energy_threshold=_number(
                    low_rank["energy_threshold"], name="low_rank.energy_threshold"
                ),
                polynomial_degree=_integer(
                    low_rank["polynomial_degree"], name="low_rank.polynomial_degree"
                ),
                max_extrapolation_ratio=_number(
                    low_rank["max_extrapolation_ratio"],
                    name="low_rank.max_extrapolation_ratio",
                ),
                eigenvalue_relative_floor=_number(
                    low_rank["eigenvalue_relative_floor"],
                    name="low_rank.eigenvalue_relative_floor",
                ),
            ),
            safety=ForecastSafetyConfig(
                max_relative_tensor_norm=_number(
                    safety["max_relative_tensor_norm"], name="safety.max_relative_tensor_norm"
                ),
                min_relative_tensor_norm=_number(
                    safety["min_relative_tensor_norm"], name="safety.min_relative_tensor_norm"
                ),
                near_zero_norm=_number(safety["near_zero_norm"], name="safety.near_zero_norm"),
            ),
            gates=CandidateAcceptanceGates(
                max_verification_loss_increase=_number(
                    gates["max_verification_loss_increase"],
                    name="gates.max_verification_loss_increase",
                ),
                max_post_leap_loss_increase=_number(
                    gates["max_post_leap_loss_increase"],
                    name="gates.max_post_leap_loss_increase",
                ),
                max_anchor_mse=_number(gates["max_anchor_mse"], name="gates.max_anchor_mse"),
                max_anchor_cosine_distance=_number(
                    gates["max_anchor_cosine_distance"],
                    name="gates.max_anchor_cosine_distance",
                ),
                max_anchor_symmetric_kl=_number(
                    gates["max_anchor_symmetric_kl"], name="gates.max_anchor_symmetric_kl"
                ),
            ),
        )

    @classmethod
    def load(cls, path: Path) -> TrajectoryConfig:
        return cls.from_mapping(load_json_object(path, name="trajectory config"))
