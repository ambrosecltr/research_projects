from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..fingerprint import count_sketch
from ..hashing import canonical_json_bytes, sha256_json
from ..types import TensorSpec
from .hub import DatasetOrderPlan, HubFile


@dataclass(frozen=True)
class EvidenceConfig:
    initialization_sketch_dim_per_role: int = 128
    digest_vector_dim: int = 32
    seed: int = 1701

    def __post_init__(self) -> None:
        for field in ("initialization_sketch_dim_per_role", "digest_vector_dim"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def _digest_bytes(value: Any) -> bytes:
    return hashlib.sha256(canonical_json_bytes(value)).digest()


def digest_vector(value: Any, dimension: int) -> torch.Tensor:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("digest vector dimension must be a positive integer")
    output = bytearray()
    counter = 0
    seed = canonical_json_bytes(value)
    while len(output) < dimension:
        output.extend(hashlib.sha256(seed + counter.to_bytes(8, "little")).digest())
        counter += 1
    return torch.tensor(
        [(byte / 127.5) - 1.0 for byte in output[:dimension]],
        dtype=torch.float32,
    )


def _tensor_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def initialization_fingerprint(
    state: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    *,
    sketch_dim_per_role: int,
    seed: int,
) -> tuple[torch.Tensor, list[str]]:
    roles = sorted({spec.role for spec in inventory if not spec.is_buffer})
    role_sketches = {role: torch.zeros(sketch_dim_per_role, dtype=torch.float32) for role in roles}
    role_stats: dict[str, list[float]] = {role: [0.0, 0.0, 0.0, 0.0] for role in roles}
    role_counts = {role: 0 for role in roles}
    for spec in inventory:
        if spec.is_buffer:
            continue
        tensor = state[spec.name].detach().to(torch.float32)
        norm = max(float(torch.linalg.vector_norm(tensor).item()), 1e-12)
        role_sketches[spec.role] += count_sketch(
            tensor / norm,
            sketch_dim_per_role,
            seed=_tensor_seed(spec.name, seed),
        )
        stats = role_stats[spec.role]
        stats[0] += math.log1p(tensor.numel()) / 24.0
        stats[1] += float(tensor.mean().item())
        stats[2] += float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
        stats[3] += math.log1p(norm) / 16.0
        role_counts[spec.role] += 1
    parts = []
    for role in roles:
        count = max(role_counts[role], 1)
        parts.append(role_sketches[role] / math.sqrt(count))
        parts.append(torch.tensor([value / count for value in role_stats[role]]))
    return torch.cat(parts), roles


def architecture_features(
    architecture: Mapping[str, Any],
    inventory: Sequence[TensorSpec],
) -> torch.Tensor:
    numeric_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "max_position_embeddings",
        "rotary_pct",
        "layer_norm_eps",
        "initializer_range",
    )
    values = []
    for key in numeric_fields:
        raw = architecture.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"architecture field {key} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"architecture field {key} must be finite")
        values.append(math.copysign(math.log1p(abs(number)), number) / 16.0)
    roles = sorted({spec.role for spec in inventory})
    state_numel = max(sum(spec.numel for spec in inventory), 1)
    for role in roles:
        role_numel = sum(spec.numel for spec in inventory if spec.role == role)
        role_tensors = sum(1 for spec in inventory if spec.role == role)
        values.extend([role_numel / state_numel, math.log1p(role_tensors) / 8.0])
    values.extend(
        [
            1.0 if architecture.get("use_parallel_residual") else 0.0,
            1.0 if architecture.get("tie_word_embeddings") else 0.0,
            math.log1p(len(inventory)) / 8.0,
        ]
    )
    return torch.tensor(values, dtype=torch.float32)


def _file_identity(file: HubFile) -> dict[str, Any]:
    return {
        "name": file.name,
        "size": file.size,
        "sha256": file.sha256,
        "git_blob_id": file.git_blob_id,
    }


def dataset_order_fingerprint(
    plan: DatasetOrderPlan,
    *,
    data_order_seed: int,
    dimension: int,
) -> torch.Tensor:
    seed_key = str(data_order_seed)
    if seed_key not in plan.seed_files:
        raise KeyError(f"dataset-order plan has no seed {data_order_seed}")
    identity = {
        "repository": plan.repository,
        "commit": plan.commit,
        "data_order_seed": data_order_seed,
        "files": [_file_identity(file) for file in plan.seed_files[seed_key]],
    }
    digest = digest_vector(identity, dimension)
    sizes = torch.tensor(
        [math.log1p(file.size) / 32.0 for file in plan.seed_files[seed_key]],
        dtype=torch.float32,
    )
    return torch.cat([digest, sizes, torch.tensor([data_order_seed / 9.0])])


def build_compiler_evidence(
    *,
    base_state: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    architecture: Mapping[str, Any],
    dataset_order: DatasetOrderPlan,
    data_order_seed: int,
    tokenizer_identity: Mapping[str, Any],
    training_recipe: Mapping[str, Any],
    config: EvidenceConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    initialization, roles = initialization_fingerprint(
        base_state,
        inventory,
        sketch_dim_per_role=config.initialization_sketch_dim_per_role,
        seed=config.seed,
    )
    tensors = {
        "architecture_features": architecture_features(architecture, inventory),
        "initialization_fingerprint": initialization,
        "dataset_fingerprint": dataset_order_fingerprint(
            dataset_order,
            data_order_seed=data_order_seed,
            dimension=config.digest_vector_dim,
        ),
        "tokenizer_fingerprint": digest_vector(
            tokenizer_identity,
            config.digest_vector_dim,
        ),
        "training_recipe_fingerprint": digest_vector(
            training_recipe,
            config.digest_vector_dim,
        ),
    }
    manifest = {
        "format": "GENOME_COMPILER_EVIDENCE",
        "version": "0.1.0",
        "conditioning": [
            "canonical_W0_fingerprint",
            "architecture",
            "tokenizer_fingerprint",
            "dataset_and_order_fingerprint",
            "complete_training_recipe",
        ],
        "forbidden_endpoint_inputs": [
            "WT_values",
            "endpoint_hashes",
            "endpoint_fitted_codes",
            "intermediate_or_early_training_weights",
        ],
        "initialization_roles": roles,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "config": {
            "initialization_sketch_dim_per_role": config.initialization_sketch_dim_per_role,
            "digest_vector_dim": config.digest_vector_dim,
            "seed": config.seed,
        },
        "architecture_sha256": sha256_json(dict(architecture)),
        "dataset_order_identity_sha256": sha256_json(
            {
                "repository": dataset_order.repository,
                "commit": dataset_order.commit,
                "seed": data_order_seed,
                "files": [
                    _file_identity(file) for file in dataset_order.seed_files[str(data_order_seed)]
                ],
            }
        ),
        "tokenizer_identity_sha256": sha256_json(dict(tokenizer_identity)),
        "training_recipe_sha256": sha256_json(dict(training_recipe)),
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return tensors, manifest
