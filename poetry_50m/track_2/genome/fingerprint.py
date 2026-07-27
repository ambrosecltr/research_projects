from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .adapters.base import Track1Adapter
from .types import TensorSpec


@dataclass(frozen=True)
class GradientFingerprintConfig:
    sketch_dim_per_role: int = 128
    max_batches: int = 32
    seed: int = 1701
    normalize_each_tensor: bool = True
    split: str = "fingerprint"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_sketch(
    tensor: torch.Tensor,
    output_dim: int,
    *,
    seed: int,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    """Deterministic CountSketch without materializing a dense projection matrix."""
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    flat = tensor.detach().to(torch.float32).flatten().cpu()
    output = torch.zeros(output_dim, dtype=torch.float32)
    # Fixed odd constants. Arithmetic is modulo 2^63 through bit masking, then modulo k.
    a = 6364136223846793005
    b = (1442695040888963407 + int(seed) * 2 + 1) & ((1 << 63) - 1)
    c = 2862933555777941757
    d = (3037000493 + int(seed) * 4 + 1) & ((1 << 63) - 1)
    mask = (1 << 63) - 1
    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        indices = torch.arange(start, end, dtype=torch.int64)
        buckets = torch.bitwise_and(indices * a + b, mask).remainder(output_dim)
        signs = torch.bitwise_and(indices * c + d, mask)
        signs = torch.where(signs.bitwise_and(1).eq(0), 1.0, -1.0).to(torch.float32)
        output.scatter_add_(0, buckets, flat[start:end] * signs)
    return output


def _parameter_role_map(specs: Sequence[TensorSpec]) -> dict[str, str]:
    return {spec.name: spec.role for spec in specs if not spec.is_buffer}


def build_gradient_fingerprint(
    adapter: Track1Adapter,
    base_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    config: GradientFingerprintConfig | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    config = config or GradientFingerprintConfig()
    device_obj = torch.device(device)
    model = adapter.build_model().to(device_obj)
    missing, unexpected = model.load_state_dict(dict(base_state), strict=False)
    if missing or unexpected:
        raise ValueError(f"base-state mismatch; missing={missing}, unexpected={unexpected}")
    role_by_name = _parameter_role_map(tensor_specs)
    roles = sorted(set(role_by_name.values()))
    batch_vectors: list[torch.Tensor] = []
    losses: list[float] = []
    counts: list[int] = []

    model.train(False)
    for batch_index, raw_batch in enumerate(
        adapter.evaluation_batches(config.split, max_batches=config.max_batches)
    ):
        if batch_index >= config.max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = adapter.move_batch(raw_batch, device_obj)
        loss_sum, count = adapter.batch_loss(model, batch)
        loss = loss_sum / max(count, 1)
        loss.backward()
        role_vectors = {role: torch.zeros(config.sketch_dim_per_role) for role in roles}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or name not in role_by_name:
                continue
            gradient = parameter.grad.detach()
            if config.normalize_each_tensor:
                gradient = gradient / max(float(torch.linalg.vector_norm(gradient).item()), 1e-12)
            tensor_seed = config.seed + sum((index + 1) * ord(char) for index, char in enumerate(name))
            role_vectors[role_by_name[name]] += count_sketch(
                gradient, config.sketch_dim_per_role, seed=tensor_seed
            )
        batch_vectors.append(torch.cat([role_vectors[role] for role in roles]))
        losses.append(float(loss.item()))
        counts.append(int(count))

    if not batch_vectors:
        raise ValueError("fingerprint split produced no batches")
    stacked = torch.stack(batch_vectors)
    return {
        "roles": roles,
        "config": config.to_dict(),
        "batch_count": len(batch_vectors),
        "token_or_item_counts": counts,
        "loss_mean": float(sum(losses) / len(losses)),
        "fingerprint_mean": stacked.mean(dim=0),
        "fingerprint_std": stacked.std(dim=0, unbiased=False),
        "fingerprint_min": stacked.min(dim=0).values,
        "fingerprint_max": stacked.max(dim=0).values,
    }
