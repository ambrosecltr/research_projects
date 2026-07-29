from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .fingerprint import count_sketch
from .hashing import sha256_json


@dataclass(frozen=True)
class SemanticFingerprintConfig:
    token_sketch_dim: int = 512
    bigram_sketch_dim: int = 512
    gradient_sketch_dim_per_role: int = 128
    length_bin_edges: tuple[int, ...] = (
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    )
    seed: int = 1701

    def __post_init__(self) -> None:
        for name in (
            "token_sketch_dim",
            "bigram_sketch_dim",
            "gradient_sketch_dim_per_role",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not self.length_bin_edges
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.length_bin_edges
            )
            or tuple(sorted(set(self.length_bin_edges))) != self.length_bin_edges
        ):
            raise ValueError("length_bin_edges must be unique, positive, and increasing")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["length_bin_edges"] = list(self.length_bin_edges)
        return data


@dataclass(frozen=True)
class SemanticFingerprint:
    """Model inputs derived from corpus content and W0 responses.

    Source revisions and SHA-256 values belong in the accompanying manifest for provenance, but
    never in these tensors. Cryptographic digests intentionally destroy semantic neighbourhoods.
    """

    tensors: dict[str, torch.Tensor]
    manifest: dict[str, Any]

    def flattened(self) -> torch.Tensor:
        order = self.manifest.get("semantic_tensor_order")
        if not isinstance(order, list) or any(not isinstance(name, str) for name in order):
            raise ValueError("semantic fingerprint lacks a valid tensor order")
        if set(order) != set(self.tensors):
            raise ValueError("semantic fingerprint tensor order differs from stored tensors")
        return torch.cat([self.tensors[name].to(torch.float32).flatten() for name in order])


class CorpusFingerprintBuilder:
    """Streaming, deterministic corpus statistics suitable for compiler conditioning."""

    def __init__(
        self,
        *,
        vocab_size: int,
        config: SemanticFingerprintConfig | None = None,
    ) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        self.vocab_size = vocab_size
        self.config = config or SemanticFingerprintConfig()
        self.token_sketch = torch.zeros(self.config.token_sketch_dim, dtype=torch.float64)
        self.bigram_sketch = torch.zeros(self.config.bigram_sketch_dim, dtype=torch.float64)
        self.byte_counts = torch.zeros(256, dtype=torch.float64)
        self.length_counts = torch.zeros(len(self.config.length_bin_edges) + 1, dtype=torch.float64)
        self.sequence_count = 0
        self.token_count = 0
        self.supervised_token_count = 0
        self.length_sum = 0.0
        self.length_squared_sum = 0.0
        self.minimum_length: int | None = None
        self.maximum_length = 0

    @staticmethod
    def _mix64(value: int, seed: int) -> int:
        mask = (1 << 64) - 1
        result = (value + 0x9E3779B97F4A7C15 + seed) & mask
        result = ((result ^ (result >> 30)) * 0xBF58476D1CE4E5B9) & mask
        result = ((result ^ (result >> 27)) * 0x94D049BB133111EB) & mask
        return (result ^ (result >> 31)) & mask

    @classmethod
    def _sketch_add(cls, output: torch.Tensor, key: int, value: float, *, seed: int) -> None:
        mixed = cls._mix64(key, seed)
        bucket = mixed % output.numel()
        sign = 1.0 if ((mixed >> 63) & 1) == 0 else -1.0
        output[bucket] += sign * value

    def update_tokens(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        supervised_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> None:
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.detach().to(torch.int64).flatten().cpu().tolist()
        else:
            values = list(token_ids)
        if not values:
            raise ValueError("fingerprint sequences must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("token IDs must be integers")
        if any(value < 0 or value >= self.vocab_size for value in values):
            raise ValueError("token ID is outside the configured vocabulary")

        if supervised_mask is None:
            supervised = len(values)
        else:
            if isinstance(supervised_mask, torch.Tensor):
                mask = supervised_mask.detach().to(torch.bool).flatten().cpu().tolist()
            else:
                mask = list(supervised_mask)
            if len(mask) != len(values) or any(not isinstance(item, bool) for item in mask):
                raise ValueError("supervised mask must be a boolean sequence aligned with token IDs")
            supervised = sum(mask)

        for token in values:
            self._sketch_add(
                self.token_sketch,
                token,
                1.0,
                seed=self.config.seed,
            )
        for left, right in zip(values, values[1:], strict=False):
            key = (left << 32) ^ right
            self._sketch_add(
                self.bigram_sketch,
                key,
                1.0,
                seed=self.config.seed + 1,
            )

        length = len(values)
        bin_index = 0
        while (
            bin_index < len(self.config.length_bin_edges)
            and length > self.config.length_bin_edges[bin_index]
        ):
            bin_index += 1
        self.length_counts[bin_index] += 1.0
        self.sequence_count += 1
        self.token_count += length
        self.supervised_token_count += supervised
        self.length_sum += length
        self.length_squared_sum += length * length
        self.minimum_length = length if self.minimum_length is None else min(self.minimum_length, length)
        self.maximum_length = max(self.maximum_length, length)

    def update_bytes(self, value: bytes | bytearray | memoryview | str) -> None:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        if not raw:
            return
        counts = torch.bincount(torch.tensor(list(raw), dtype=torch.int64), minlength=256)
        self.byte_counts += counts.to(torch.float64)

    def update_many(
        self,
        sequences: Iterable[Sequence[int] | torch.Tensor],
    ) -> None:
        for sequence in sequences:
            self.update_tokens(sequence)

    @staticmethod
    def _unit_mass(value: torch.Tensor) -> torch.Tensor:
        denominator = float(value.abs().sum().item())
        return value.to(torch.float32) / max(denominator, 1.0)

    def finalize(
        self,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> SemanticFingerprint:
        if self.sequence_count < 1 or self.token_count < 1:
            raise ValueError("cannot finalize an empty corpus fingerprint")
        mean_length = self.length_sum / self.sequence_count
        variance = max(self.length_squared_sum / self.sequence_count - mean_length**2, 0.0)
        scalars = torch.tensor(
            [
                math.log1p(self.sequence_count),
                math.log1p(self.token_count),
                mean_length,
                variance**0.5,
                float(self.minimum_length or 0),
                float(self.maximum_length),
                self.supervised_token_count / self.token_count,
                math.log1p(float(self.byte_counts.sum().item())),
            ],
            dtype=torch.float32,
        )
        tensors = {
            "corpus.token_unigram_countsketch": self._unit_mass(self.token_sketch),
            "corpus.token_bigram_countsketch": self._unit_mass(self.bigram_sketch),
            "corpus.byte_frequency": self._unit_mass(self.byte_counts),
            "corpus.sequence_length_histogram": self._unit_mass(self.length_counts),
            "corpus.scalar_statistics": scalars,
        }
        order = sorted(tensors)
        manifest: dict[str, Any] = {
            "format": "GENOME_SEMANTIC_FINGERPRINT",
            "version": "0.1.0",
            "contains_endpoint_data": False,
            "semantic_tensor_order": order,
            "tensor_shapes": {name: list(tensors[name].shape) for name in order},
            "config": self.config.to_dict(),
            "vocab_size": self.vocab_size,
            "statistics": {
                "sequence_count": self.sequence_count,
                "token_count": self.token_count,
                "supervised_token_count": self.supervised_token_count,
            },
            "provenance": dict(provenance or {}),
            "provenance_is_model_input": False,
        }
        manifest["content_sha256"] = sha256_json(manifest)
        return SemanticFingerprint(tensors=tensors, manifest=manifest)


def gradient_probe_fingerprint(
    named_gradients: Mapping[str, torch.Tensor],
    role_by_name: Mapping[str, str],
    *,
    config: SemanticFingerprintConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Summarize actual W0 gradient response by tensor role.

    The result is invariant to tensor length and bounded in size. It is semantic evidence because
    it is calculated from model/data interaction, unlike repository or file hashes.
    """

    config = config or SemanticFingerprintConfig()
    if not named_gradients:
        raise ValueError("gradient probe requires at least one gradient")
    missing_roles = set(named_gradients) - set(role_by_name)
    if missing_roles:
        raise ValueError(f"gradient roles are missing for tensors: {sorted(missing_roles)}")
    roles = sorted({role_by_name[name] for name in named_gradients})
    sketches = {
        role: torch.zeros(config.gradient_sketch_dim_per_role, dtype=torch.float32)
        for role in roles
    }
    stats = {role: torch.zeros(5, dtype=torch.float64) for role in roles}
    counts = {role: 0 for role in roles}

    for tensor_index, name in enumerate(sorted(named_gradients)):
        gradient = named_gradients[name]
        if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point():
            raise TypeError(f"gradient {name!r} must be a floating-point tensor")
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError(f"gradient {name!r} contains NaN or Inf")
        role = role_by_name[name]
        flat = gradient.detach().to(torch.float32).flatten().cpu()
        norm = max(float(torch.linalg.vector_norm(flat).item()), 1e-12)
        sketches[role] += count_sketch(
            flat / norm,
            config.gradient_sketch_dim_per_role,
            seed=config.seed + 10_000 + tensor_index,
        )
        stats[role] += torch.tensor(
            [
                math.log1p(flat.numel()),
                norm,
                float(flat.abs().max().item()),
                float(flat.mean().item()),
                float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            ],
            dtype=torch.float64,
        )
        counts[role] += 1

    tensors: dict[str, torch.Tensor] = {}
    for role in roles:
        count = max(counts[role], 1)
        tensors[f"w0_gradient.{role}.countsketch"] = sketches[role] / math.sqrt(count)
        tensors[f"w0_gradient.{role}.statistics"] = (stats[role] / count).to(torch.float32)
    manifest = {
        "format": "GENOME_W0_GRADIENT_FINGERPRINT",
        "version": "0.1.0",
        "contains_endpoint_data": False,
        "roles": roles,
        "tensor_order": sorted(tensors),
        "tensor_shapes": {name: list(tensors[name].shape) for name in sorted(tensors)},
        "config": config.to_dict(),
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return tensors, manifest


def activation_probe_fingerprint(
    named_activations: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not named_activations:
        raise ValueError("activation probe requires at least one activation tensor")
    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(named_activations):
        value = named_activations[name]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"activation {name!r} must be a floating-point tensor")
        flat = value.detach().to(torch.float32).flatten().cpu()
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"activation {name!r} contains NaN or Inf")
        quantiles = torch.quantile(flat, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
        tensors[f"w0_activation.{name}.summary"] = torch.cat(
            [
                torch.tensor(
                    [
                        float(flat.mean().item()),
                        float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
                        float(torch.linalg.vector_norm(flat).item()) / math.sqrt(max(flat.numel(), 1)),
                        math.log1p(flat.numel()),
                    ]
                ),
                quantiles,
            ]
        ).to(torch.float32)
    manifest = {
        "format": "GENOME_W0_ACTIVATION_FINGERPRINT",
        "version": "0.1.0",
        "contains_endpoint_data": False,
        "tensor_order": sorted(tensors),
        "tensor_shapes": {name: list(tensors[name].shape) for name in sorted(tensors)},
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return tensors, manifest
