from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .state import clone_state, compute_delta
from .tensor_inventory import restore_tied_values
from .types import TensorSpec


def singular_summaries(
    delta: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    top_k: int = 32,
    max_exact_elements: int = 4_000_000,
) -> list[dict]:
    rows = []
    for spec in tensor_specs:
        tensor = delta[spec.name]
        if tensor.ndim != 2:
            continue
        matrix = tensor.to(torch.float32)
        truncated = matrix.numel() > max_exact_elements
        if truncated:
            q = min(max(top_k + 8, top_k), min(matrix.shape))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1701 + spec.canonical_index)
                _, singular, _ = torch.svd_lowrank(matrix, q=q, niter=2)
            singular = singular.sort(descending=True).values
        else:
            singular = torch.linalg.svdvals(matrix)
        energy = singular.square()
        measured_total = float(energy.sum().item())
        probabilities = singular / singular.sum().clamp_min(1e-30)
        effective_rank = float(
            torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()).item()
        )
        if truncated:
            rank_90 = None
            rank_99 = None
            total = float(matrix.square().sum().item())
        else:
            total = measured_total
            cumulative = energy.cumsum(0) / max(total, 1e-30)
            rank_90 = int(torch.searchsorted(cumulative, torch.tensor(0.90)).item() + 1)
            rank_99 = int(torch.searchsorted(cumulative, torch.tensor(0.99)).item() + 1)
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "layer_index": spec.layer_index,
                "shape": list(spec.shape),
                "top_singular_values": singular[:top_k].tolist(),
                "rank_90": rank_90,
                "rank_99": rank_99,
                "effective_rank_of_measured_spectrum": effective_rank,
                "frobenius_energy": total,
                "spectrum_truncated": truncated,
                "measured_rank": int(singular.numel()),
            }
        )
    return rows


def compose_role_ablation(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    roles_from_target: set[str],
    tied_groups: Sequence[Sequence[str]] = (),
) -> dict[str, torch.Tensor]:
    output = clone_state(base_state)
    for spec in tensor_specs:
        if spec.role in roles_from_target:
            output[spec.name] = target_state[spec.name].detach().clone()
    restore_tied_values(output, tied_groups)
    return output


def add_calibrated_noise(
    state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    relative_std: float,
    seed: int,
    roles: set[str] | None = None,
    tied_groups: Sequence[Sequence[str]] = (),
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = clone_state(state)
    for spec in tensor_specs:
        if roles is not None and spec.role not in roles:
            continue
        tensor = output[spec.name]
        if not tensor.is_floating_point():
            continue
        scale = float(tensor.to(torch.float32).std(unbiased=False).item()) * relative_std
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32) * scale
        output[spec.name] = (tensor.to(torch.float32) + noise).to(tensor.dtype)
    restore_tied_values(output, tied_groups)
    return output


def delta_energy_by_role(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
) -> dict[str, dict[str, float]]:
    delta = compute_delta(base_state, target_state, tensor_specs)
    result: dict[str, dict[str, float]] = defaultdict(lambda: {"energy": 0.0, "numel": 0.0})
    for spec in tensor_specs:
        result[spec.role]["energy"] += float(delta[spec.name].square().sum().item())
        result[spec.role]["numel"] += spec.numel
    return dict(result)
