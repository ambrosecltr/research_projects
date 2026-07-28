from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .tensor_inventory import ROLE_ORDER
from .types import TensorSpec


@dataclass(frozen=True)
class ArchitectureNode:
    index: int
    tensor_name: str
    role: str
    layer_index: int | None
    shape: tuple[int, ...]
    numel: int
    tied_group: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class ArchitectureEdge:
    source: int
    target: int
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureGraph:
    nodes: list[ArchitectureNode]
    edges: list[ArchitectureEdge]
    role_to_id: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "0.1.0",
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "role_to_id": self.role_to_id,
        }

    def as_tensors(self, *, max_shape_dims: int = 4) -> dict[str, torch.Tensor]:
        max_layer = max((node.layer_index for node in self.nodes if node.layer_index is not None), default=0)
        features = []
        for node in self.nodes:
            shape = list(node.shape[:max_shape_dims]) + [1] * max(0, max_shape_dims - len(node.shape))
            features.append(
                [
                    float(self.role_to_id[node.role]),
                    -1.0 if node.layer_index is None else node.layer_index / max(max_layer, 1),
                    float(len(node.shape)),
                    math.log1p(node.numel) / 24.0,
                    *[math.log1p(dimension) / 16.0 for dimension in shape],
                    1.0 if node.tied_group is not None else 0.0,
                ]
            )
        edge_index = torch.tensor(
            [[edge.source for edge in self.edges], [edge.target for edge in self.edges]],
            dtype=torch.long,
        )
        relation_names = sorted({edge.relation for edge in self.edges})
        relation_to_id = {name: index for index, name in enumerate(relation_names)}
        edge_type = torch.tensor([relation_to_id[edge.relation] for edge in self.edges], dtype=torch.long)
        return {
            "node_features": torch.tensor(features, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_type": edge_type,
        }


def build_architecture_graph(
    specs: Sequence[TensorSpec], tied_groups: Sequence[Sequence[str]] = ()
) -> ArchitectureGraph:
    nodes = [
        ArchitectureNode(
            index=spec.canonical_index,
            tensor_name=spec.name,
            role=spec.role,
            layer_index=spec.layer_index,
            shape=spec.shape,
            numel=spec.numel,
            tied_group=spec.tied_group,
        )
        for spec in specs
    ]
    by_name = {node.tensor_name: node.index for node in nodes}
    edges: list[ArchitectureEdge] = []
    # Canonical adjacency is a weak but universal structural prior.
    for left, right in zip(nodes, nodes[1:]):
        edges.append(ArchitectureEdge(left.index, right.index, "canonical_next"))
        edges.append(ArchitectureEdge(right.index, left.index, "canonical_previous"))

    by_layer: dict[int | None, list[ArchitectureNode]] = {}
    for node in nodes:
        by_layer.setdefault(node.layer_index, []).append(node)
    for layer_nodes in by_layer.values():
        ordered = sorted(layer_nodes, key=lambda node: (ROLE_ORDER.get(node.role, 100), node.tensor_name))
        for left, right in zip(ordered, ordered[1:]):
            edges.append(ArchitectureEdge(left.index, right.index, "within_layer_forward"))
            edges.append(ArchitectureEdge(right.index, left.index, "within_layer_backward"))

    role_layers: dict[str, list[ArchitectureNode]] = {}
    for node in nodes:
        if node.layer_index is not None:
            role_layers.setdefault(node.role, []).append(node)
    for role_nodes in role_layers.values():
        ordered = sorted(role_nodes, key=lambda node: node.layer_index or 0)
        for left, right in zip(ordered, ordered[1:]):
            edges.append(ArchitectureEdge(left.index, right.index, "same_role_next_layer"))
            edges.append(ArchitectureEdge(right.index, left.index, "same_role_previous_layer"))

    for group in tied_groups:
        valid = [by_name[name] for name in group if name in by_name]
        for source in valid:
            for target in valid:
                if source != target:
                    edges.append(ArchitectureEdge(source, target, "tied"))

    # De-duplicate deterministically.
    unique = {(edge.source, edge.target, edge.relation): edge for edge in edges}
    edges = [unique[key] for key in sorted(unique)]
    roles = sorted({node.role for node in nodes}, key=lambda role: (ROLE_ORDER.get(role, 100), role))
    return ArchitectureGraph(nodes=nodes, edges=edges, role_to_id={role: i for i, role in enumerate(roles)})
