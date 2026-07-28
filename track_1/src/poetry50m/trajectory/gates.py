"""Safety, fixed-anchor function drift, and predeclared acceptance gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class ForecastSafetyConfig:
    max_relative_tensor_norm: float = 4.0
    min_relative_tensor_norm: float = 0.25
    near_zero_norm: float = 1e-9

    def __post_init__(self) -> None:
        max_relative_tensor_norm, min_relative_tensor_norm, near_zero_norm = tuple(
            _finite_number(value, name=name)
            for name, value in (
                ("max_relative_tensor_norm", self.max_relative_tensor_norm),
                ("min_relative_tensor_norm", self.min_relative_tensor_norm),
                ("near_zero_norm", self.near_zero_norm),
            )
        )
        if (
            max_relative_tensor_norm <= 0.0
            or min_relative_tensor_norm < 0.0
            or near_zero_norm <= 0.0
        ):
            raise ValueError("safety limits must be positive")
        if min_relative_tensor_norm > max_relative_tensor_norm:
            raise ValueError("minimum relative norm cannot exceed maximum relative norm")


@dataclass(frozen=True, slots=True)
class TensorSafetyResult:
    name: str
    finite: bool
    current_norm: float | None
    candidate_norm: float | None
    relative_norm: float | None
    near_zero_reference: bool
    accepted: bool


@dataclass(frozen=True, slots=True)
class ForecastSafetyReport:
    accepted: bool
    tensors: Mapping[str, TensorSafetyResult]

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "tensors": {name: asdict(value) for name, value in self.tensors.items()},
        }


def forecast_safety_report(
    current: Mapping[str, Tensor], candidate: Mapping[str, Tensor], config: ForecastSafetyConfig
) -> ForecastSafetyReport:
    if tuple(current) != tuple(candidate):
        raise ValueError("current and candidate state_dicts must have identical named coordinates")
    outcomes: dict[str, TensorSafetyResult] = {}
    for name, current_tensor in current.items():
        candidate_tensor = candidate[name]
        if (
            current_tensor.shape != candidate_tensor.shape
            or current_tensor.dtype != candidate_tensor.dtype
        ):
            raise ValueError(f"candidate tensor {name} has a different shape or dtype")
        if not candidate_tensor.is_floating_point():
            accepted = torch.equal(current_tensor, candidate_tensor)
            outcomes[name] = TensorSafetyResult(name, True, None, None, None, False, accepted)
            continue
        finite = bool(torch.isfinite(candidate_tensor).all().item())
        current_norm = float(torch.linalg.vector_norm(current_tensor.float()).item())
        candidate_norm = float(torch.linalg.vector_norm(candidate_tensor.float()).item())
        near_zero = current_norm <= config.near_zero_norm
        relative = candidate_norm / max(current_norm, config.near_zero_norm)
        if near_zero:
            accepted = (
                finite
                and math.isfinite(candidate_norm)
                and candidate_norm <= config.max_relative_tensor_norm * config.near_zero_norm
            )
        else:
            accepted = (
                finite
                and math.isfinite(candidate_norm)
                and config.min_relative_tensor_norm <= relative <= config.max_relative_tensor_norm
            )
        outcomes[name] = TensorSafetyResult(
            name, finite, current_norm, candidate_norm, relative, near_zero, accepted
        )
    return ForecastSafetyReport(all(value.accepted for value in outcomes.values()), outcomes)


@dataclass(frozen=True, slots=True)
class FunctionDriftMetrics:
    mean_squared_error: float
    cosine_distance: float
    symmetric_kl: float
    max_absolute_difference: float
    element_count: int
    distribution_count: int

    def __post_init__(self) -> None:
        values = (
            self.mean_squared_error,
            self.cosine_distance,
            self.symmetric_kl,
            self.max_absolute_difference,
        )
        if any(_finite_number(value, name="function drift metric") < 0.0 for value in values):
            raise ValueError("function drift metrics must be finite and non-negative")
        if (
            isinstance(self.element_count, bool)
            or not isinstance(self.element_count, int)
            or isinstance(self.distribution_count, bool)
            or not isinstance(self.distribution_count, int)
            or self.element_count < 1
            or self.distribution_count < 0
        ):
            raise ValueError("function drift counts are invalid")


@dataclass(frozen=True, slots=True)
class AnchorOutputs:
    """Fixed-anchor outputs with explicit distribution versus representation semantics."""

    logits: Mapping[str, Tensor]
    representations: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not self.logits and not self.representations:
            raise ValueError("anchor outputs cannot be empty")
        overlap = set(self.logits) & set(self.representations)
        if overlap:
            raise ValueError(f"anchor output names cannot overlap: {sorted(overlap)}")


def fixed_anchor_function_drift(
    reference: AnchorOutputs, candidate: AnchorOutputs, *, temperature: float = 1.0
) -> FunctionDriftMetrics:
    """Aggregate behaviour drift over a fixed, predeclared anchor-output mapping."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if tuple(reference.logits) != tuple(candidate.logits):
        raise ValueError("anchor logits must have identical ordered names")
    if tuple(reference.representations) != tuple(candidate.representations):
        raise ValueError("anchor representations must have identical ordered names")
    squared_error = 0.0
    maximum = 0.0
    dot = 0.0
    reference_norm_sq = 0.0
    candidate_norm_sq = 0.0
    kl_sum = 0.0
    count = 0
    distribution_count = 0

    def accumulate(
        name: str, reference_tensor: Tensor, candidate_tensor: Tensor, is_distribution: bool
    ) -> None:
        nonlocal squared_error, maximum, dot, reference_norm_sq, candidate_norm_sq, kl_sum
        nonlocal count, distribution_count
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"anchor output shape differs for {name}")
        if not reference_tensor.is_floating_point() or not candidate_tensor.is_floating_point():
            raise TypeError("anchor outputs must be floating tensors")
        ref = reference_tensor.detach().double()
        proposed = candidate_tensor.detach().double()
        if not bool(torch.isfinite(ref).all().item()) or not bool(
            torch.isfinite(proposed).all().item()
        ):
            raise ValueError("anchor outputs must be finite")
        difference = proposed - ref
        squared_error += float(difference.square().sum().item())
        maximum = max(maximum, float(difference.abs().max().item()))
        dot += float(torch.sum(ref * proposed).item())
        reference_norm_sq += float(torch.sum(ref.square()).item())
        candidate_norm_sq += float(torch.sum(proposed.square()).item())
        if is_distribution:
            if ref.shape[-1] < 2:
                raise ValueError(f"logit anchor {name} requires at least two vocabulary values")
            log_p = F.log_softmax(ref / temperature, dim=-1)
            log_q = F.log_softmax(proposed / temperature, dim=-1)
            p, q = log_p.exp(), log_q.exp()
            kl_sum += float(
                (0.5 * ((p * (log_p - log_q)).sum(dim=-1) + (q * (log_q - log_p)).sum(dim=-1)))
                .sum()
                .item()
            )
            distribution_count += ref.numel() // ref.shape[-1]
        count += ref.numel()

    for name, reference_tensor in reference.logits.items():
        accumulate(name, reference_tensor, candidate.logits[name], True)
    for name, reference_tensor in reference.representations.items():
        accumulate(name, reference_tensor, candidate.representations[name], False)
    if reference_norm_sq <= 1e-24 and candidate_norm_sq <= 1e-24:
        cosine = 1.0
    elif reference_norm_sq <= 1e-24 or candidate_norm_sq <= 1e-24:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(reference_norm_sq * candidate_norm_sq)
    return FunctionDriftMetrics(
        mean_squared_error=squared_error / max(count, 1),
        cosine_distance=1.0 - max(-1.0, min(1.0, cosine)),
        symmetric_kl=kl_sum / max(distribution_count, 1),
        max_absolute_difference=maximum,
        element_count=count,
        distribution_count=distribution_count,
    )


def future_target_distillation_loss(
    current_logits: Tensor,
    future_logits: Tensor,
    *,
    temperature: float = 1.0,
    mse_weight: float = 0.0,
) -> Tensor:
    """Differentiable function-space backup target; it does not generate weights."""

    if current_logits.shape != future_logits.shape or current_logits.ndim < 2:
        raise ValueError("current and future logits must have equal shape with a vocabulary axis")
    if temperature <= 0.0 or mse_weight < 0.0:
        raise ValueError("temperature must be positive and mse_weight non-negative")
    target = future_logits.detach()
    per_distribution = F.kl_div(
        F.log_softmax(current_logits / temperature, dim=-1),
        F.softmax(target / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    kl = per_distribution.mean() * (temperature**2)
    return kl + mse_weight * F.mse_loss(current_logits, target)


@dataclass(frozen=True, slots=True)
class CandidateAcceptanceGates:
    max_verification_loss_increase: float
    max_post_leap_loss_increase: float
    max_anchor_mse: float
    max_anchor_cosine_distance: float
    max_anchor_symmetric_kl: float

    def __post_init__(self) -> None:
        values = (
            self.max_verification_loss_increase,
            self.max_post_leap_loss_increase,
            self.max_anchor_mse,
            self.max_anchor_cosine_distance,
            self.max_anchor_symmetric_kl,
        )
        validated = tuple(_finite_number(value, name="acceptance gate") for value in values)
        if min(validated) < 0.0:
            raise ValueError("acceptance gates must be non-negative")


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    verification_loss_increase: float
    post_leap_loss_increase: float
    drift: FunctionDriftMetrics
    safety: ForecastSafetyReport
    candidate_state_hash: str | None = None

    def __post_init__(self) -> None:
        _finite_number(self.verification_loss_increase, name="verification_loss_increase")
        _finite_number(self.post_leap_loss_increase, name="post_leap_loss_increase")
        if not isinstance(self.accepted, bool):
            raise TypeError("candidate decision accepted must be boolean")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.reasons
        ):
            raise TypeError("candidate decision reasons must be non-empty strings")
        if not isinstance(self.drift, FunctionDriftMetrics) or not isinstance(
            self.safety, ForecastSafetyReport
        ):
            raise TypeError("candidate decision requires typed drift and safety reports")
        if self.candidate_state_hash is not None and (
            not isinstance(self.candidate_state_hash, str) or not self.candidate_state_hash
        ):
            raise ValueError("candidate_state_hash cannot be empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "verification_loss_increase": self.verification_loss_increase,
            "post_leap_loss_increase": self.post_leap_loss_increase,
            "drift": asdict(self.drift),
            "safety": self.safety.to_mapping(),
            "candidate_state_hash": self.candidate_state_hash,
        }


def decide_candidate(
    *,
    baseline_verification_loss: float,
    candidate_verification_loss: float,
    baseline_post_leap_loss: float,
    candidate_post_leap_loss: float,
    drift: FunctionDriftMetrics,
    safety: ForecastSafetyReport,
    gates: CandidateAcceptanceGates,
    candidate_state_hash: str | None = None,
) -> CandidateDecision:
    values = (
        baseline_verification_loss,
        candidate_verification_loss,
        baseline_post_leap_loss,
        candidate_post_leap_loss,
    )
    if any(_finite_number(value, name="loss value") < 0.0 for value in values):
        raise ValueError("loss values must be finite and non-negative")
    verification_increase = candidate_verification_loss - baseline_verification_loss
    post_leap_increase = candidate_post_leap_loss - baseline_post_leap_loss
    reasons: list[str] = []
    if not safety.accepted:
        reasons.append("forecast safety checks failed")
    if verification_increase > gates.max_verification_loss_increase:
        reasons.append("verification loss increase exceeded gate")
    if post_leap_increase > gates.max_post_leap_loss_increase:
        reasons.append("post-leap loss increase exceeded gate")
    if drift.mean_squared_error > gates.max_anchor_mse:
        reasons.append("anchor mean squared error exceeded gate")
    if drift.cosine_distance > gates.max_anchor_cosine_distance:
        reasons.append("anchor cosine distance exceeded gate")
    if drift.symmetric_kl > gates.max_anchor_symmetric_kl:
        reasons.append("anchor symmetric KL exceeded gate")
    return CandidateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        verification_loss_increase=verification_increase,
        post_leap_loss_increase=post_leap_increase,
        drift=drift,
        safety=safety,
        candidate_state_hash=candidate_state_hash,
    )
