from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from .mgp.runtime import execute_program
from .mgp.schema import ModelGenomeProgram


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float
    tokens: int
    finite: bool


@dataclass(frozen=True)
class ComparisonResult:
    w0: EvaluationResult
    candidate: EvaluationResult
    endpoint: EvaluationResult | None
    endpoint_progress: float | None
    candidate_beats_w0: bool
    logit_kl_to_endpoint: float | None
    top1_agreement: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_causal_lm(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> EvaluationResult:
    model = model.to(device).eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for index, batch in enumerate(batches):
            if max_batches is not None and index >= max_batches:
                break
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                return EvaluationResult(loss=float("inf"), perplexity=float("inf"), tokens=total_tokens, finite=False)
            labels = inputs.get("labels")
            tokens = int((labels != -100).sum()) if labels is not None else int(inputs["input_ids"].numel())
            total_loss += float(loss) * tokens
            total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("evaluation received no supervised tokens")
    loss_value = total_loss / total_tokens
    return EvaluationResult(
        loss=loss_value,
        perplexity=float(torch.exp(torch.tensor(min(loss_value, 80.0)))),
        tokens=total_tokens,
        finite=True,
    )


def endpoint_progress(w0_loss: float, candidate_loss: float, endpoint_loss: float) -> float:
    denominator = w0_loss - endpoint_loss
    if denominator <= 0:
        raise ValueError("endpoint must improve on W0 to define progress")
    return (w0_loss - candidate_loss) / denominator


def compare_logits(
    candidate: torch.nn.Module,
    endpoint: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    max_batches: int = 4,
) -> tuple[float, float]:
    candidate = candidate.to(device).eval()
    endpoint = endpoint.to(device).eval()
    total_kl = 0.0
    total_agreement = 0.0
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
            candidate_logits = candidate(**inputs, use_cache=False).logits.float()
            endpoint_logits = endpoint(**inputs, use_cache=False).logits.float()
            kl = F.kl_div(
                F.log_softmax(candidate_logits, dim=-1),
                F.softmax(endpoint_logits, dim=-1),
                reduction="batchmean",
            )
            agreement = (candidate_logits.argmax(dim=-1) == endpoint_logits.argmax(dim=-1)).float().mean()
            total_kl += float(kl)
            total_agreement += float(agreement)
            count += 1
    if count == 0:
        raise ValueError("logit comparison received no batches")
    return total_kl / count, total_agreement / count


def load_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
    missing, unexpected = model.load_state_dict(dict(state), strict=True)
    if missing or unexpected:  # pragma: no cover - strict load already raises in modern torch
        raise ValueError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def evaluate_program(
    *,
    model_factory,
    base_state: Mapping[str, torch.Tensor],
    program: ModelGenomeProgram,
    payloads: Mapping[str, torch.Tensor],
    batches: Iterable[Mapping[str, torch.Tensor]],
    endpoint_state: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> ComparisonResult:
    cached = list(batches)
    if not cached:
        raise ValueError("Genome Gate requires evaluation batches")
    candidate_state = execute_program(base_state, program, payloads)
    w0_model = load_state(model_factory(), base_state)
    candidate_model = load_state(model_factory(), candidate_state)
    w0_result = evaluate_causal_lm(w0_model, cached, device=device, max_batches=max_batches)
    candidate_result = evaluate_causal_lm(candidate_model, cached, device=device, max_batches=max_batches)
    if endpoint_state is None:
        return ComparisonResult(
            w0=w0_result,
            candidate=candidate_result,
            endpoint=None,
            endpoint_progress=None,
            candidate_beats_w0=candidate_result.loss < w0_result.loss,
            logit_kl_to_endpoint=None,
            top1_agreement=None,
        )
    endpoint_model = load_state(model_factory(), endpoint_state)
    endpoint_result = evaluate_causal_lm(endpoint_model, cached, device=device, max_batches=max_batches)
    progress = endpoint_progress(w0_result.loss, candidate_result.loss, endpoint_result.loss)
    kl, agreement = compare_logits(candidate_model, endpoint_model, cached, device=device)
    return ComparisonResult(
        w0=w0_result,
        candidate=candidate_result,
        endpoint=endpoint_result,
        endpoint_progress=progress,
        candidate_beats_w0=candidate_result.loss < w0_result.loss,
        logit_kl_to_endpoint=kl,
        top1_agreement=agreement,
    )


@dataclass(frozen=True)
class FunctionalGate:
    maximum_target_fraction: float = 0.10
    minimum_development_progress: float = 0.80
    minimum_hidden_progress: float = 0.0

    def accept_development(self, comparison: ComparisonResult, target_fraction: float) -> bool:
        return bool(
            comparison.endpoint_progress is not None
            and comparison.candidate.finite
            and target_fraction <= self.maximum_target_fraction
            and comparison.endpoint_progress >= self.minimum_development_progress
        )

    def accept_hidden(self, comparison: ComparisonResult, target_fraction: float) -> bool:
        return bool(
            comparison.endpoint_progress is not None
            and comparison.candidate.finite
            and target_fraction <= self.maximum_target_fraction
            and comparison.endpoint_progress > self.minimum_hidden_progress
        )
