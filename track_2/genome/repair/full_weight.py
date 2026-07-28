from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Mapping

import torch

from ..adapters.base import Track1Adapter
from ..tensor_inventory import canonicalize_state_dict


@dataclass(frozen=True)
class FullWeightRepairConfig:
    steps: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    split: str = "probe"
    max_batches: int | None = None
    device: str = "cpu"


def repair_full_weights(
    adapter: Track1Adapter,
    initial_state: Mapping[str, torch.Tensor],
    *,
    config: FullWeightRepairConfig | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    config = config or FullWeightRepairConfig()
    device = torch.device(config.device)
    model = adapter.build_model().to(device)
    missing, unexpected = model.load_state_dict(dict(initial_state), strict=False)
    if missing or unexpected:
        raise ValueError(f"repair state mismatch; missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batches = list(adapter.evaluation_batches(config.split, max_batches=config.max_batches))
    if not batches:
        raise ValueError("repair split produced no batches")
    iterator = cycle(batches)
    metrics = []
    model.train()
    for step in range(1, config.steps + 1):
        batch = adapter.move_batch(next(iterator), device)
        loss_sum, count = adapter.batch_loss(model, batch)
        loss = loss_sum / max(count, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, config.steps // 20) == 0 or step == config.steps:
            metrics.append({"step": float(step), "probe_loss": float(loss.detach().item())})
    return canonicalize_state_dict(model.state_dict()), metrics
