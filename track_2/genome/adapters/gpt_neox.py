from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..architecture import ArchitectureGraph, graph_from_state


@dataclass(frozen=True)
class GPTNeoXAdapter:
    """Strict adapter for the Hugging Face GPT-NeoX/Pythia state layout.

    The active v1 language keeps fused QKV tensors fused. This is reversible and avoids inventing
    a cross-architecture alignment before the same-family Pythia experiment works.
    """

    family: str = "gpt_neox"

    @staticmethod
    def canonical_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        canonical = {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
        GPTNeoXAdapter.validate_state(canonical)
        return canonical

    @staticmethod
    def native_state(canonical: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        native = {name: tensor.detach().cpu().contiguous() for name, tensor in canonical.items()}
        GPTNeoXAdapter.validate_state(native)
        return native

    @staticmethod
    def validate_state(state: Mapping[str, torch.Tensor]) -> None:
        required_fragments = (
            "gpt_neox.embed_in.weight",
            "gpt_neox.layers.0.attention.query_key_value.weight",
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            "gpt_neox.final_layer_norm.weight",
            "embed_out.weight",
        )
        missing = [name for name in required_fragments if name not in state]
        if missing:
            raise ValueError(f"not a supported GPT-NeoX state; missing {missing}")
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"state entry {name!r} is not a tensor")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"state entry {name!r} contains non-finite values")

    @staticmethod
    def graph(state: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> ArchitectureGraph:
        return graph_from_state(state, family="gpt_neox", config=config)

    @staticmethod
    def load_hf(
        repo_id: str,
        *,
        revision: str,
        cache_dir: str | Path,
        local_files_only: bool = False,
        torch_dtype: torch.dtype = torch.float32,
    ) -> tuple[Any, Any]:
        try:
            from transformers import AutoTokenizer, GPTNeoXForCausalLM
        except ImportError as error:  # pragma: no cover - dependency error is explicit
            raise RuntimeError("transformers is required to load Pythia") from error
        model = GPTNeoXForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
        return model, tokenizer

    @staticmethod
    def roundtrip_equal(state: Mapping[str, torch.Tensor]) -> bool:
        native = GPTNeoXAdapter.native_state(GPTNeoXAdapter.canonical_state(state))
        return set(native) == set(state) and all(torch.equal(native[k], state[k].cpu()) for k in state)
