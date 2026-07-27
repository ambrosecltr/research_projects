"""Typed fixed-batch candidate verification before any forecast is applied."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from poetry50m.model.transformer import DecoderOnlyTransformer
from poetry50m.trajectory.forecast import ForecastResult
from poetry50m.trajectory.gates import (
    AnchorOutputs,
    CandidateAcceptanceGates,
    CandidateDecision,
    ForecastSafetyConfig,
    decide_candidate,
    fixed_anchor_function_drift,
    forecast_safety_report,
)
from poetry50m.trajectory.preparation import PreparedForecast, prepare_forecast


@dataclass(frozen=True, slots=True)
class VerificationBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.targets.shape != self.input_ids.shape:
            raise ValueError("verification input_ids and targets must be matching rank-two tensors")
        if self.loss_mask is not None and self.loss_mask.shape != self.input_ids.shape:
            raise ValueError("verification loss_mask must match input_ids")


LossEvaluator = Callable[[nn.Module, Sequence[VerificationBatch]], float]
AnchorEvaluator = Callable[[nn.Module], AnchorOutputs]
PostLeapEvaluator = Callable[[nn.Module], float]


def decoder_loss_evaluator(module: nn.Module, batches: Sequence[VerificationBatch]) -> float:
    """Evaluate fixed NTP batches for the shared decoder model without gradient updates."""

    if not isinstance(module, DecoderOnlyTransformer):
        raise TypeError("decoder_loss_evaluator requires DecoderOnlyTransformer")
    if not batches:
        raise ValueError("verification batches cannot be empty")
    was_training = module.training
    module.eval()
    try:
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in batches:
                output = module(batch.input_ids, targets=batch.targets, loss_mask=batch.loss_mask)
                if output.loss is None or output.token_count < 1:
                    raise ValueError("verification batch produced no language-model loss")
                total_loss += float(output.loss.item()) * output.token_count
                total_tokens += output.token_count
    finally:
        module.train(was_training)
    return total_loss / total_tokens


@contextmanager
def _temporary_state(module: nn.Module, state_dict: Mapping[str, Tensor]) -> Iterator[None]:
    original = {name: value.detach().clone() for name, value in module.state_dict().items()}
    was_training = module.training
    module.load_state_dict(dict(state_dict), strict=True)
    module.eval()
    try:
        yield
    finally:
        module.load_state_dict(original, strict=True)
        module.train(was_training)


def _validate_continued_baseline_state(
    current_state: Mapping[str, Tensor], continued_baseline_state: Mapping[str, Tensor]
) -> None:
    if tuple(current_state) != tuple(continued_baseline_state):
        raise ValueError("continued baseline state names or order do not match the live module")
    for name, current_tensor in current_state.items():
        candidate_tensor = continued_baseline_state[name]
        if not isinstance(candidate_tensor, Tensor):
            raise TypeError(f"continued baseline tensor {name} is not a Tensor")
        if candidate_tensor.shape != current_tensor.shape:
            raise ValueError(
                f"continued baseline tensor {name} shape does not match the live model"
            )
        if candidate_tensor.dtype != current_tensor.dtype:
            raise ValueError(
                f"continued baseline tensor {name} dtype does not match the live model"
            )
        if not torch.isfinite(candidate_tensor).all():
            raise ValueError(f"continued baseline tensor {name} must be finite")


@dataclass(frozen=True, slots=True)
class VerificationTiming:
    baseline_seconds: float
    candidate_seconds: float
    post_leap_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    prepared: PreparedForecast
    decision: CandidateDecision
    timing: VerificationTiming
    baseline_comparator: Literal["current_checkpoint", "continued_baseline"]


def verify_candidate(
    *,
    module: nn.Module,
    forecast: ForecastResult,
    batches: Sequence[VerificationBatch],
    anchor_evaluator: AnchorEvaluator,
    gates: CandidateAcceptanceGates,
    safety_config: ForecastSafetyConfig,
    loss_evaluator: LossEvaluator = decoder_loss_evaluator,
    post_leap_evaluator: PostLeapEvaluator | None = None,
    continued_baseline_state: Mapping[str, Tensor] | None = None,
) -> CandidateVerification:
    """Score against W_t or an explicit continued-baseline state without live mutation."""

    prepared = prepare_forecast(module, forecast)
    current = {name: value.detach().clone() for name, value in module.state_dict().items()}
    if continued_baseline_state is not None:
        _validate_continued_baseline_state(current, continued_baseline_state)
    safety = forecast_safety_report(current, prepared.state_dict, safety_config)
    baseline_state = current if continued_baseline_state is None else continued_baseline_state
    baseline_comparator: Literal["current_checkpoint", "continued_baseline"] = (
        "current_checkpoint" if continued_baseline_state is None else "continued_baseline"
    )
    baseline_start = time.perf_counter()
    with _temporary_state(module, baseline_state):
        baseline_loss = loss_evaluator(module, batches)
        baseline_anchors = anchor_evaluator(module)
        baseline_post_loss = (
            baseline_loss if post_leap_evaluator is None else post_leap_evaluator(module)
        )
    baseline_seconds = time.perf_counter() - baseline_start
    candidate_start = time.perf_counter()
    with _temporary_state(module, prepared.state_dict):
        candidate_loss = loss_evaluator(module, batches)
        candidate_anchors = anchor_evaluator(module)
        if post_leap_evaluator is None:
            candidate_post_loss = candidate_loss
            post_leap_seconds = 0.0
        else:
            post_start = time.perf_counter()
            candidate_post_loss = post_leap_evaluator(module)
            post_leap_seconds = time.perf_counter() - post_start
    candidate_seconds = time.perf_counter() - candidate_start
    drift = fixed_anchor_function_drift(baseline_anchors, candidate_anchors)
    decision = decide_candidate(
        baseline_verification_loss=baseline_loss,
        candidate_verification_loss=candidate_loss,
        baseline_post_leap_loss=baseline_post_loss,
        candidate_post_leap_loss=candidate_post_loss,
        drift=drift,
        safety=safety,
        gates=gates,
        candidate_state_hash=prepared.state_hash,
    )
    return CandidateVerification(
        prepared=prepared,
        decision=decision,
        timing=VerificationTiming(baseline_seconds, candidate_seconds, post_leap_seconds),
        baseline_comparator=baseline_comparator,
    )
