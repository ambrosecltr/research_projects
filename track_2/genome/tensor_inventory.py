from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import torch
from torch import nn

from .types import TensorSpec

ROLE_ORDER = {
    "embedding": 0,
    "attention_norm": 10,
    "q_proj": 20,
    "k_proj": 21,
    "v_proj": 22,
    "qkv_proj": 23,
    "o_proj": 24,
    "attention_scale": 25,
    "residual_rate": 26,
    "mlp_norm": 30,
    "gate_proj": 40,
    "up_proj": 41,
    "gate_up_proj": 42,
    "down_proj": 43,
    "mlp_scale": 44,
    "mlp_other": 45,
    "final_norm": 50,
    "lm_head": 60,
    "logit_scale": 61,
    "position": 70,
    "bias": 80,
    "buffer": 90,
    "other": 100,
}

_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:layers|blocks|h|transformer_blocks|block)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(\d+)(?:\.|$)"),
]


def infer_layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def infer_role(name: str, tensor: torch.Tensor, *, is_buffer: bool = False) -> str:
    """Infer a stable semantic role, with exact Track 1 names handled first.

    Ordering is important: ``blocks.N.attention.output.weight`` is an attention output
    projection, not an LM head, and ``blocks.N.mlp.in_projection.weight`` is Track 1's fused
    SwiGLU gate/value input projection, not a QKV projection.
    """
    lower = name.lower()
    if is_buffer:
        return "buffer"

    # Exact/specific poetry50m names.
    if lower == "token_embedding.weight":
        return "embedding"
    if lower.startswith("blocks."):
        if ".attention_norm." in lower:
            return "attention_norm"
        if ".mlp_norm." in lower:
            return "mlp_norm"
        if ".attention.query." in lower:
            return "q_proj"
        if ".attention.key." in lower:
            return "k_proj"
        if ".attention.value." in lower:
            return "v_proj"
        if ".attention.qkv." in lower:
            return "qkv_proj"
        if ".attention.output." in lower:
            return "o_proj"
        if lower.endswith(".attention.qk_scale"):
            return "attention_scale"
        if lower.endswith((".attention_rate", ".mlp_rate")):
            return "residual_rate"
        if ".mlp.in_projection." in lower:
            return "gate_up_proj"
        if ".mlp.out_projection." in lower:
            return "down_proj"
        if lower.endswith(".mlp.uv_scale"):
            return "mlp_scale"
    if lower.startswith("final_norm."):
        return "final_norm"
    if lower.startswith("output_projection."):
        return "lm_head"
    if lower == "logit_scale":
        return "logit_scale"

    # Architecture-generic fallbacks.
    if "lm_head" in lower or lower in {"output.weight", "output.bias"}:
        return "lm_head"
    if any(
        token in lower for token in ("tok_emb", "token_emb", "embed_tokens", "wte", "embedding")
    ):
        return "embedding"
    if any(token in lower for token in ("position", "pos_emb", "wpe", "rotary")):
        return "position"
    if "norm" in lower or "ln_" in lower or ".ln" in lower:
        if infer_layer_index(name) is None:
            return "final_norm"
        if any(token in lower for token in ("post_attention", "ffn_norm", "mlp_norm", "ln_2")):
            return "mlp_norm"
        return "attention_norm"
    if any(token in lower for token in ("qkv", "c_attn")):
        return "qkv_proj"
    if any(token in lower for token in ("q_proj", "query")):
        return "q_proj"
    if any(token in lower for token in ("k_proj", "key")):
        return "k_proj"
    if any(token in lower for token in ("v_proj", "value")):
        return "v_proj"
    if any(token in lower for token in ("o_proj", "out_proj", "c_proj")) and any(
        token in lower for token in ("attn", "attention")
    ):
        return "o_proj"
    if any(token in lower for token in ("gate_proj", "w1")):
        return "gate_proj"
    if any(token in lower for token in ("up_proj", "fc1", "c_fc", "w3")):
        return "up_proj"
    if any(token in lower for token in ("down_proj", "fc2", "w2")):
        return "down_proj"
    if any(token in lower for token in ("mlp", "ffn", "feed_forward")):
        return "mlp_other"
    if tensor.ndim == 1 and lower.endswith("bias"):
        return "bias"
    return "other"


def _tensor_storage_identity(
    tensor: torch.Tensor,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except RuntimeError:
        pointer = tensor.data_ptr()
    return pointer, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride())


def discover_tied_groups(model: nn.Module) -> list[list[str]]:
    groups: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups[_tensor_storage_identity(parameter)].append(name)
    return [sorted(names) for names in groups.values() if len(names) > 1]


def canonicalize_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for original_name, tensor in state.items():
        name = original_name
        while name.startswith("module."):
            name = name[len("module.") :]
        if name in result:
            raise ValueError(f"canonical state key collision: {original_name!r} -> {name!r}")
        result[name] = tensor.detach().contiguous().cpu()
    return result


def build_tensor_inventory(
    model: nn.Module,
    state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[list[TensorSpec], list[list[str]]]:
    canonical_state = canonicalize_state_dict(state if state is not None else model.state_dict())
    tied_groups = discover_tied_groups(model)
    canonical_tied_groups: list[list[str]] = []
    for names in tied_groups:
        canonical_names = []
        for name in names:
            canonical_name = name.removeprefix("module.")
            canonical_names.append(canonical_name)
        canonical_tied_groups.append(sorted(canonical_names))

    parameter_names = {
        name.removeprefix("module.") for name, _ in model.named_parameters(remove_duplicate=False)
    }
    return build_tensor_inventory_from_state(
        canonical_state,
        tied_groups=canonical_tied_groups,
        parameter_names=parameter_names,
    )


def build_tensor_inventory_from_state(
    state: Mapping[str, torch.Tensor],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    parameter_names: Iterable[str] | None = None,
) -> tuple[list[TensorSpec], list[list[str]]]:
    canonical_state = canonicalize_state_dict(state)
    canonical_tied_groups = [sorted(str(name) for name in group) for group in tied_groups]
    tied_by_name: dict[str, str] = {}
    for index, names in enumerate(canonical_tied_groups):
        if len(names) < 2:
            raise ValueError("tied groups must contain at least two tensor names")
        for name in names:
            if name not in canonical_state:
                raise ValueError(f"tied group references an unknown tensor: {name}")
            if name in tied_by_name:
                raise ValueError(f"tensor appears in more than one tied group: {name}")
            tied_by_name[name] = f"tie_{index:04d}"
    parameter_name_set = (
        set(canonical_state) if parameter_names is None else {str(name) for name in parameter_names}
    )
    unknown_parameters = parameter_name_set - set(canonical_state)
    if unknown_parameters:
        raise ValueError(f"parameter names are absent from state: {sorted(unknown_parameters)}")

    provisional: list[TensorSpec] = []
    for name, tensor in canonical_state.items():
        is_buffer = name not in parameter_name_set
        role = infer_role(name, tensor, is_buffer=is_buffer)
        provisional.append(
            TensorSpec(
                canonical_index=-1,
                name=name,
                role=role,
                layer_index=infer_layer_index(name),
                shape=tuple(int(x) for x in tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", ""),
                numel=tensor.numel(),
                nbytes=tensor.numel() * tensor.element_size(),
                tied_group=tied_by_name.get(name),
                initialization={},
                is_buffer=is_buffer,
            )
        )

    provisional.sort(
        key=lambda item: (
            item.layer_index if item.layer_index is not None else -1,
            ROLE_ORDER.get(item.role, ROLE_ORDER["other"]),
            item.name,
        )
    )
    inventory = [replace(item, canonical_index=index) for index, item in enumerate(provisional)]
    validate_inventory(inventory, canonical_state)
    return inventory, canonical_tied_groups


def validate_inventory(inventory: Sequence[TensorSpec], state: Mapping[str, torch.Tensor]) -> None:
    if len(inventory) != len(state):
        raise ValueError(f"inventory has {len(inventory)} entries but state has {len(state)}")
    names = [spec.name for spec in inventory]
    if len(set(names)) != len(names):
        raise ValueError("inventory contains duplicate names")
    if set(names) != set(state):
        missing = sorted(set(state) - set(names))
        extra = sorted(set(names) - set(state))
        raise ValueError(f"inventory/state mismatch; missing={missing}, extra={extra}")
    for expected_index, spec in enumerate(inventory):
        if spec.canonical_index != expected_index:
            raise ValueError("inventory canonical indices are not contiguous")
        tensor = state[spec.name]
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"shape mismatch for {spec.name}: {tuple(tensor.shape)} != {spec.shape}"
            )
        if tensor.numel() != spec.numel:
            raise ValueError(f"numel mismatch for {spec.name}")


def inventory_to_dict(
    inventory: Sequence[TensorSpec], tied_groups: Sequence[Sequence[str]]
) -> dict:
    return {
        "version": "1.0",
        "role_order": ROLE_ORDER,
        "tensors": [spec.to_dict() for spec in inventory],
        "tied_groups": [list(group) for group in tied_groups],
    }


def inventory_from_dict(value: Mapping) -> tuple[list[TensorSpec], list[list[str]]]:
    inventory = [TensorSpec.from_dict(item) for item in value["tensors"]]
    tied_groups = [list(group) for group in value.get("tied_groups", [])]
    return inventory, tied_groups


def tied_owner_map(tied_groups: Iterable[Sequence[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in tied_groups:
        if not group:
            continue
        owner = group[0]
        for name in group[1:]:
            result[name] = owner
    return result


def assert_tied_equal(
    state: Mapping[str, torch.Tensor], tied_groups: Iterable[Sequence[str]]
) -> None:
    for group in tied_groups:
        if len(group) < 2:
            continue
        reference = state[group[0]]
        for name in group[1:]:
            if not torch.equal(reference, state[name]):
                raise ValueError(f"tied tensors differ: {group[0]} and {name}")


def restore_tied_values(
    state: dict[str, torch.Tensor], tied_groups: Iterable[Sequence[str]], *, clone: bool = False
) -> None:
    for group in tied_groups:
        if not group:
            continue
        owner = group[0]
        if owner not in state:
            raise KeyError(f"missing tied owner: {owner}")
        for alias in group[1:]:
            state[alias] = state[owner].clone() if clone else state[owner]
