from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .hashing import sha256_json
from .types import TensorNode

_LAYER = re.compile(r"(?:layers|blocks|h)\.(\d+)")


def infer_layer(name: str) -> int | None:
    match = _LAYER.search(name)
    return None if match is None else int(match.group(1))


def infer_role(name: str, tensor: torch.Tensor) -> str:
    lower = name.lower()
    if "embed_in" in lower or "embed_tokens" in lower or "token_embedding" in lower:
        return "embedding"
    if "embed_out" in lower or "lm_head" in lower or "output_projection" in lower:
        return "lm_head"
    if "final_layer_norm" in lower or "final_norm" in lower:
        return "final_norm"
    if "input_layernorm" in lower or "attention_norm" in lower:
        return "attention_norm"
    if "post_attention_layernorm" in lower or "mlp_norm" in lower:
        return "mlp_norm"
    if "query_key_value" in lower or "qkv" in lower:
        return "qkv"
    if ".attention.dense" in lower or "attention.output" in lower or "o_proj" in lower:
        return "attention_output"
    if "dense_h_to_4h" in lower or "gate_up" in lower or "up_proj" in lower:
        return "mlp_up"
    if "dense_4h_to_h" in lower or "down_proj" in lower:
        return "mlp_down"
    if tensor.ndim == 1 and name.endswith("bias"):
        return "bias"
    return "other"


@dataclass(frozen=True)
class ArchitectureGraph:
    family: str
    config: dict[str, Any]
    tensors: tuple[TensorNode, ...]
    edges: tuple[tuple[int, int], ...]
    format: str = "GENOME_ARCHITECTURE_GRAPH"
    version: str = "1.0.0"

    @property
    def graph_id(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "family": self.family,
            "config": self.config,
            "tensors": [asdict(item) for item in self.tensors],
            "edges": [list(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArchitectureGraph":
        return cls(
            family=str(value["family"]),
            config=dict(value["config"]),
            tensors=tuple(TensorNode(**item) for item in value["tensors"]),
            edges=tuple(tuple(int(v) for v in edge) for edge in value["edges"]),
            format=str(value.get("format", "GENOME_ARCHITECTURE_GRAPH")),
            version=str(value.get("version", "1.0.0")),
        )


def graph_from_state(
    state: Mapping[str, torch.Tensor],
    *,
    family: str,
    config: Mapping[str, Any],
    ties: Mapping[str, str] | None = None,
) -> ArchitectureGraph:
    ties = dict(ties or {})
    tensors: list[TensorNode] = []
    for index, (name, tensor) in enumerate(sorted(state.items())):
        tensors.append(
            TensorNode(
                index=index,
                name=name,
                role=infer_role(name, tensor),
                layer=infer_layer(name),
                shape=tuple(int(v) for v in tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", ""),
                tied_to=ties.get(name),
            )
        )
    edges: set[tuple[int, int]] = set()
    by_layer: dict[int, list[int]] = {}
    globals_: list[int] = []
    for node in tensors:
        if node.layer is None:
            globals_.append(node.index)
        else:
            by_layer.setdefault(node.layer, []).append(node.index)
    for nodes in by_layer.values():
        for left, right in zip(nodes, nodes[1:]):
            edges.add((left, right))
            edges.add((right, left))
    ordered_layers = sorted(by_layer)
    for first, second in zip(ordered_layers, ordered_layers[1:]):
        for left in by_layer[first]:
            for right in by_layer[second]:
                edges.add((left, right))
                edges.add((right, left))
    for global_index in globals_:
        for node in tensors:
            if node.index != global_index:
                edges.add((global_index, node.index))
                edges.add((node.index, global_index))
    return ArchitectureGraph(
        family=family,
        config=dict(config),
        tensors=tuple(tensors),
        edges=tuple(sorted(edges)),
    )
