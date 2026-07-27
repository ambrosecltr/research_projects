from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .tensor_inventory import assert_tied_equal, restore_tied_values
from .types import TensorSpec


def clone_state(state: Mapping[str, torch.Tensor], *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to(device).clone() for name, tensor in state.items()}


def validate_compatible_states(
    base: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], inventory: Sequence[TensorSpec]
) -> None:
    expected = {spec.name for spec in inventory}
    if set(base) != expected or set(target) != expected:
        raise ValueError("base/target keys do not match frozen tensor inventory")
    for spec in inventory:
        if tuple(base[spec.name].shape) != spec.shape or tuple(target[spec.name].shape) != spec.shape:
            raise ValueError(f"shape mismatch for {spec.name}")
        if not torch.isfinite(base[spec.name]).all() or not torch.isfinite(target[spec.name]).all():
            raise ValueError(f"non-finite values in {spec.name}")


def compute_delta(
    base: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    validate_compatible_states(base, target, inventory)
    return {
        spec.name: target[spec.name].detach().to(dtype=dtype, device="cpu")
        - base[spec.name].detach().to(dtype=dtype, device="cpu")
        for spec in inventory
    }


def apply_delta(
    base: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    *,
    output_dtypes: Mapping[str, torch.dtype] | None = None,
    tied_groups: Sequence[Sequence[str]] = (),
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for spec in inventory:
        value = base[spec.name].to(torch.float32) + delta[spec.name].to(torch.float32)
        dtype = output_dtypes[spec.name] if output_dtypes is not None else base[spec.name].dtype
        output[spec.name] = value.to(dtype).contiguous()
    restore_tied_values(output, tied_groups)
    assert_tied_equal(output, tied_groups)
    return output


def state_num_bytes(state: Mapping[str, torch.Tensor], *, unique_storage: bool = False) -> int:
    if not unique_storage:
        return sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    seen: set[tuple[int, int, int]] = set()
    total = 0
    for tensor in state.values():
        try:
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.numel())
        except RuntimeError:
            key = (tensor.data_ptr(), tensor.storage_offset(), tensor.numel())
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def delta_statistics(
    base: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
) -> list[dict]:
    delta = compute_delta(base, target, inventory)
    rows: list[dict] = []
    for spec in inventory:
        d = delta[spec.name]
        b = base[spec.name].to(torch.float32)
        t = target[spec.name].to(torch.float32)
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "layer_index": spec.layer_index,
                "shape": list(spec.shape),
                "numel": spec.numel,
                "raw_bytes": spec.nbytes,
                "base_l2": float(torch.linalg.vector_norm(b).item()),
                "target_l2": float(torch.linalg.vector_norm(t).item()),
                "delta_l2": float(torch.linalg.vector_norm(d).item()),
                "delta_mean": float(d.mean().item()),
                "delta_std": float(d.std(unbiased=False).item()) if d.numel() > 1 else 0.0,
                "delta_abs_max": float(d.abs().max().item()),
                "relative_delta_l2": float(
                    torch.linalg.vector_norm(d).item() / max(torch.linalg.vector_norm(b).item(), 1e-12)
                ),
            }
        )
    return rows


def aggregate_statistics_by_role(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        role = str(row["role"])
        counts[role] += 1
        for key in ("numel", "raw_bytes", "base_l2", "target_l2", "delta_l2"):
            grouped[role][key] += float(row[key])
    result = []
    for role in sorted(grouped):
        result.append({"role": role, "tensor_count": counts[role], **dict(grouped[role])})
    return result
