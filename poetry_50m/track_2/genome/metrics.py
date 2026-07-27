from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from .types import TensorSpec


def parameter_metrics(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
) -> dict[str, Any]:
    total_sq = 0.0
    target_sq = 0.0
    total_abs = 0.0
    total_numel = 0
    max_abs = 0.0
    by_role_acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_tensor: list[dict[str, Any]] = []

    for spec in inventory:
        a = candidate[spec.name].detach().to(torch.float64).cpu()
        b = target[spec.name].detach().to(torch.float64).cpu()
        error = a - b
        sq = float(error.square().sum().item())
        ref_sq = float(b.square().sum().item())
        abs_sum = float(error.abs().sum().item())
        local_max = float(error.abs().max().item()) if error.numel() else 0.0
        total_sq += sq
        target_sq += ref_sq
        total_abs += abs_sum
        total_numel += error.numel()
        max_abs = max(max_abs, local_max)
        role = by_role_acc[spec.role]
        role["squared_error"] += sq
        role["target_squared_norm"] += ref_sq
        role["absolute_error"] += abs_sum
        role["numel"] += error.numel()
        by_tensor.append(
            {
                "name": spec.name,
                "role": spec.role,
                "layer_index": spec.layer_index,
                "mse": sq / max(error.numel(), 1),
                "relative_l2": math.sqrt(sq / max(ref_sq, 1e-30)),
                "max_abs": local_max,
            }
        )

    by_role = {}
    for role, values in by_role_acc.items():
        by_role[role] = {
            "mse": values["squared_error"] / max(values["numel"], 1.0),
            "mae": values["absolute_error"] / max(values["numel"], 1.0),
            "relative_l2": math.sqrt(
                values["squared_error"] / max(values["target_squared_norm"], 1e-30)
            ),
            "numel": int(values["numel"]),
        }

    return {
        "mse": total_sq / max(total_numel, 1),
        "mae": total_abs / max(total_numel, 1),
        "relative_l2": math.sqrt(total_sq / max(target_sq, 1e-30)),
        "max_abs": max_abs,
        "numel": total_numel,
        "by_role": by_role,
        "by_tensor": by_tensor,
    }


def logits_kl(candidate_logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    if candidate_logits.shape != reference_logits.shape:
        raise ValueError(
            f"logit shape mismatch: {tuple(candidate_logits.shape)} != {tuple(reference_logits.shape)}"
        )
    candidate_log_probs = torch.log_softmax(candidate_logits.to(torch.float64), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.to(torch.float64), dim=-1)
    reference_probs = reference_log_probs.exp()
    value = torch.sum(reference_probs * (reference_log_probs - candidate_log_probs), dim=-1).mean()
    return float(value.item())


def topk_agreement(candidate_logits: torch.Tensor, reference_logits: torch.Tensor, k: int = 1) -> float:
    if candidate_logits.shape != reference_logits.shape:
        raise ValueError("logit shapes differ")
    candidate = candidate_logits.topk(k, dim=-1).indices
    reference = reference_logits.topk(k, dim=-1).indices
    if k == 1:
        return float(candidate.eq(reference).to(torch.float32).mean().item())
    overlaps = []
    for c, r in zip(candidate.reshape(-1, k), reference.reshape(-1, k), strict=True):
        overlaps.append(len(set(c.tolist()).intersection(r.tolist())) / k)
    return float(sum(overlaps) / max(len(overlaps), 1))


def perplexity_from_mean_loss(mean_loss: float) -> float:
    return float(math.exp(min(mean_loss, 80.0)))


def terminal_noise_normalized_gap(
    candidate_loss: float,
    reference_loss: float,
    terminal_loss_std: float | None,
) -> float | None:
    if terminal_loss_std is None or terminal_loss_std <= 0:
        return None
    return (candidate_loss - reference_loss) / terminal_loss_std
