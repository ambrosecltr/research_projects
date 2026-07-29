from __future__ import annotations

from collections.abc import Mapping

import torch

from .schema import ModelGenomeProgram


def _payload(payloads: Mapping[str, torch.Tensor], key: str, *, name: str) -> torch.Tensor:
    if key not in payloads:
        raise KeyError(f"missing payload {key!r} for {name}")
    return payloads[key]


def _execution_device(
    base: torch.Tensor, payloads: Mapping[str, torch.Tensor], keys: list[str]
) -> torch.device:
    for key in keys:
        value = payloads.get(key)
        if value is not None and (value.requires_grad or value.device.type != "cpu"):
            return value.device
    return base.device


def execute_program(
    base_state: Mapping[str, torch.Tensor],
    program: ModelGenomeProgram,
    payloads: Mapping[str, torch.Tensor],
    *,
    output_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Execute an MGP with deterministic tensor operations.

    The Runtime is code, not a learned network. Floating payload gradients are preserved so compact
    program coefficients can be fitted through the real model while W0 remains fixed.
    """

    program_names = {item.name for item in program.tensors}
    if program_names != set(base_state):
        missing = set(base_state) - program_names
        extra = program_names - set(base_state)
        raise ValueError(
            f"program/base tensor mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result: dict[str, torch.Tensor] = {}
    tied: list[tuple[str, str]] = []
    for tensor_program in program.tensors:
        base = base_state[tensor_program.name]
        if tuple(base.shape) != tensor_program.shape:
            raise ValueError(f"shape mismatch for {tensor_program.name}")
        if tensor_program.tied_to is not None:
            tied.append((tensor_program.name, tensor_program.tied_to))
            continue
        payload_keys = [
            key for component in tensor_program.components for key in component.payload.values()
        ]
        device = _execution_device(base, payloads, payload_keys)
        base_value = base.to(device=device, dtype=torch.float32)
        delta = torch.zeros_like(base_value)
        for component in tensor_program.components:
            primitive = component.primitive
            if primitive == "BASE_COPY":
                continue
            if primitive == "LOW_RANK":
                left = (
                    _payload(payloads, component.payload["left"], name=tensor_program.name)
                    .to(device)
                    .float()
                )
                right = (
                    _payload(payloads, component.payload["right"], name=tensor_program.name)
                    .to(device)
                    .float()
                )
                if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
                    raise ValueError(f"invalid low-rank factors for {tensor_program.name}")
                update = left @ right.transpose(0, 1)
                if tuple(update.shape) != tensor_program.shape:
                    raise ValueError(f"low-rank shape mismatch for {tensor_program.name}")
                delta = delta + update
            elif primitive == "HADAMARD_SCALE":
                if len(tensor_program.shape) != 2:
                    raise ValueError("HADAMARD_SCALE is permitted only for matrices")
                row = _payload(
                    payloads,
                    component.payload["row"],
                    name=tensor_program.name,
                ).to(device)
                column = _payload(
                    payloads,
                    component.payload["column"],
                    name=tensor_program.name,
                ).to(device)
                if (
                    row.ndim != 1
                    or column.ndim != 1
                    or row.numel() != tensor_program.shape[0]
                    or column.numel() != tensor_program.shape[1]
                ):
                    raise ValueError(f"invalid Hadamard scaling payload for {tensor_program.name}")
                scale = row.float().unsqueeze(1) + column.float().unsqueeze(0)
                delta = delta + base_value * scale
            elif primitive == "DIRECT_VECTOR":
                if len(tensor_program.shape) != 1:
                    raise ValueError("DIRECT_VECTOR is permitted only for vectors")
                values = _payload(
                    payloads,
                    component.payload["values"],
                    name=tensor_program.name,
                ).to(device)
                if (
                    not values.is_floating_point()
                    or values.ndim != 1
                    or values.numel() != base.numel()
                ):
                    raise ValueError(f"invalid direct vector payload for {tensor_program.name}")
                delta = delta + values.float()
            elif primitive == "QUANTIZED_VECTOR":
                if len(tensor_program.shape) != 1:
                    raise ValueError("QUANTIZED_VECTOR is permitted only for vectors")
                values = _payload(
                    payloads, component.payload["values"], name=tensor_program.name
                ).to(device)
                scale = _payload(payloads, component.payload["scale"], name=tensor_program.name).to(
                    device
                )
                if (
                    values.dtype != torch.int8
                    or values.numel() != base.numel()
                    or scale.numel() != 1
                ):
                    raise ValueError(f"invalid quantized vector payload for {tensor_program.name}")
                delta = delta + values.float().reshape_as(base_value) * scale.float().reshape(())
            elif primitive == "SPARSE_PATCH":
                indices = _payload(
                    payloads, component.payload["indices"], name=tensor_program.name
                ).to(device)
                values = _payload(
                    payloads, component.payload["values"], name=tensor_program.name
                ).to(device)
                if indices.dtype != torch.int64 or indices.numel() != values.numel():
                    raise ValueError(f"invalid sparse patch for {tensor_program.name}")
                flat_size = delta.numel()
                if indices.numel() and (indices.min() < 0 or indices.max() >= flat_size):
                    raise ValueError(f"sparse patch index outside {tensor_program.name}")
                flat = torch.zeros(flat_size, device=device, dtype=torch.float32).scatter_add(
                    0, indices.reshape(-1), values.float().reshape(-1)
                )
                delta = delta + flat.reshape_as(delta)
            elif primitive == "COPY_FROM_TIED":
                raise ValueError("COPY_FROM_TIED must be declared through tied_to")
            else:  # pragma: no cover - schema rejects this
                raise ValueError(f"unsupported primitive {primitive}")
        candidate = base_value + delta
        if not torch.isfinite(candidate).all():
            raise ValueError(f"program produced non-finite tensor {tensor_program.name}")
        result[tensor_program.name] = candidate.to(output_dtype or base.dtype)
    for alias, owner in tied:
        if owner not in result:
            raise ValueError(f"tied owner {owner!r} must be decoded before alias {alias!r}")
        result[alias] = result[owner]
    return result
