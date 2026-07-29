from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..io import read_json

_LAYER_PATTERN = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
_CANONICAL_LAYER_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")

_NATIVE_LAYER_PREFIXES = (
    ("input_layernorm.", "attention_norm."),
    ("post_attention_layernorm.", "mlp_norm."),
    ("attention.query_key_value.", "attention.qkv_proj."),
    ("attention.dense.", "attention.o_proj."),
    ("attention.rotary_emb.", "attention.rotary."),
    ("mlp.dense_h_to_4h.", "mlp.up_proj."),
    ("mlp.dense_4h_to_h.", "mlp.down_proj."),
)
_CANONICAL_LAYER_PREFIXES = tuple(
    (canonical, native) for native, canonical in _NATIVE_LAYER_PREFIXES
)
_OUTPUT_HEAD_PREFIXES = ("embed_out.", "lm_head.")


def _strip_wrappers(name: str) -> str:
    result = name
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if result.startswith(prefix):
                result = result[len(prefix) :]
                changed = True
    return result


def native_to_canonical_key(name: str) -> str:
    value = _strip_wrappers(name)
    direct = (
        ("gpt_neox.embed_in.", "token_embedding."),
        ("gpt_neox.final_layer_norm.", "final_norm."),
        ("embed_out.", "lm_head."),
        ("lm_head.", "lm_head."),
    )
    for native, canonical in direct:
        if value.startswith(native):
            return canonical + value[len(native) :]
    match = _LAYER_PATTERN.match(value)
    if match is None:
        raise ValueError(f"unsupported GPT-NeoX state key: {name}")
    layer, suffix = match.groups()
    for native, canonical in _NATIVE_LAYER_PREFIXES:
        if suffix.startswith(native):
            return f"layers.{layer}.{canonical}{suffix[len(native) :]}"
    raise ValueError(f"unsupported GPT-NeoX layer state key: {name}")


def canonical_to_native_key(
    name: str,
    *,
    output_head_prefix: str = "embed_out.",
) -> str:
    if output_head_prefix not in _OUTPUT_HEAD_PREFIXES:
        raise ValueError(f"unsupported GPT-NeoX output head prefix: {output_head_prefix}")
    direct = (
        ("token_embedding.", "gpt_neox.embed_in."),
        ("final_norm.", "gpt_neox.final_layer_norm."),
        ("lm_head.", output_head_prefix),
    )
    for canonical, native in direct:
        if name.startswith(canonical):
            return native + name[len(canonical) :]
    match = _CANONICAL_LAYER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported canonical GPT-NeoX state key: {name}")
    layer, suffix = match.groups()
    for canonical, native in _CANONICAL_LAYER_PREFIXES:
        if suffix.startswith(canonical):
            return f"gpt_neox.layers.{layer}.{native}{suffix[len(canonical) :]}"
    raise ValueError(f"unsupported canonical GPT-NeoX layer state key: {name}")


def canonicalize_gpt_neox_state(
    state: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    for native_name, tensor in state.items():
        if not isinstance(native_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("GPT-NeoX state must map string names to tensors")
        canonical_name = native_to_canonical_key(native_name)
        if canonical_name in canonical:
            raise ValueError(f"canonical GPT-NeoX key collision: {canonical_name}")
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype)
        canonical[canonical_name] = value.contiguous()
    if not canonical:
        raise ValueError("GPT-NeoX state is empty")
    return canonical


def nativeize_gpt_neox_state(
    state: Mapping[str, torch.Tensor],
    *,
    output_head_prefix: str = "embed_out.",
) -> dict[str, torch.Tensor]:
    native: dict[str, torch.Tensor] = {}
    for canonical_name, tensor in state.items():
        if not isinstance(canonical_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("canonical GPT-NeoX state must map string names to tensors")
        native_name = canonical_to_native_key(
            canonical_name,
            output_head_prefix=output_head_prefix,
        )
        if native_name in native:
            raise ValueError(f"native GPT-NeoX key collision: {native_name}")
        native[native_name] = tensor.detach().cpu().contiguous()
    if not native:
        raise ValueError("canonical GPT-NeoX state is empty")
    return native


def load_huggingface_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = Path(path).expanduser().resolve(strict=True)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw: object = load_file(str(checkpoint), device="cpu")
    elif checkpoint.suffix in {".bin", ".pt", ".pth"}:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unsupported Hugging Face checkpoint type: {checkpoint.name}")
    if isinstance(raw, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            nested = raw.get(key)
            if isinstance(nested, Mapping):
                raw = nested
                break
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint is not a state mapping: {type(raw)!r}")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("checkpoint must map string names to tensors")
        result[name] = tensor
    return result


def load_canonical_gpt_neox_state(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    return canonicalize_gpt_neox_state(load_huggingface_state(path), dtype=dtype)


def validate_gpt_neox_config(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("GPT-NeoX config must be an object with string keys")
    if value.get("model_type") != "gpt_neox":
        raise ValueError(f"expected a gpt_neox config, got {value.get('model_type')!r}")
    required_positive = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    for key in required_positive:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"GPT-NeoX config field {key} must be a positive integer")
    return value


def build_gpt_neox_model(config_path: str | Path) -> nn.Module:
    validate_gpt_neox_config(config_path)
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig.from_json_file(str(config_path))
    return GPTNeoXForCausalLM(config)


def load_canonical_state_into_model(
    model: nn.Module,
    canonical_state: Mapping[str, torch.Tensor],
) -> None:
    expected_names = set(model.state_dict())
    if "lm_head.weight" in expected_names:
        output_head_prefix = "lm_head."
    elif "embed_out.weight" in expected_names:
        output_head_prefix = "embed_out."
    else:
        raise ValueError("GPT-NeoX model has no supported output head")
    native = nativeize_gpt_neox_state(
        canonical_state,
        output_head_prefix=output_head_prefix,
    )
    try:
        model.load_state_dict(native, strict=True)
    except RuntimeError as error:
        raise ValueError(f"GPT-NeoX canonical-to-native mismatch: {error}") from error


def model_from_canonical_state(
    config_path: str | Path,
    canonical_state: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = build_gpt_neox_model(config_path)
    load_canonical_state_into_model(model, canonical_state)
    return model.to(device).eval()


def assert_native_canonical_roundtrip(state: Mapping[str, torch.Tensor]) -> None:
    stripped = {_strip_wrappers(name): tensor.detach().cpu() for name, tensor in state.items()}
    if any(name.startswith("lm_head.") for name in stripped):
        output_head_prefix = "lm_head."
    else:
        output_head_prefix = "embed_out."
    canonical = canonicalize_gpt_neox_state(state)
    roundtrip = nativeize_gpt_neox_state(
        canonical,
        output_head_prefix=output_head_prefix,
    )
    if set(roundtrip) != set(stripped):
        raise ValueError("GPT-NeoX native/canonical key round-trip changed the state inventory")
    for name in sorted(stripped):
        expected = stripped[name]
        actual = roundtrip[name]
        if expected.is_floating_point():
            expected = expected.to(torch.float32)
        if not torch.equal(actual, expected):
            raise ValueError(f"GPT-NeoX native/canonical round-trip changed tensor {name}")
