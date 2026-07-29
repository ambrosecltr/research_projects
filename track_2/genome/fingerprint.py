from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .hashing import sha256_json, sha256_tensor, stable_u64
from .io import atomic_write_json, load_json


@dataclass(frozen=True)
class FingerprintConfig:
    unigram_dim: int = 256
    bigram_dim: int = 512
    byte_dim: int = 256
    length_bins: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
    gradient_dim: int = 128
    max_probe_batches: int = 16
    seed: int = 1729


@dataclass(frozen=True)
class FingerprintBundle:
    metadata: dict[str, object]
    tensors: dict[str, torch.Tensor]

    @property
    def fingerprint_id(self) -> str:
        summary = {
            "metadata": self.metadata,
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes_sha256": sha256_tensor(tensor),
                }
                for name, tensor in sorted(self.tensors.items())
            },
        }
        return sha256_json(summary)

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payload = {name: tensor.detach().cpu().contiguous() for name, tensor in self.tensors.items()}
        save_file(payload, str(root / "fingerprint.safetensors"))
        atomic_write_json(
            root / "fingerprint.json",
            {
                "format": "GENOME_SEMANTIC_FINGERPRINT",
                "version": "1.0.0",
                "fingerprint_id": self.fingerprint_id,
                "metadata": self.metadata,
                "tensor_keys": sorted(payload),
            },
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FingerprintBundle":
        root = Path(directory)
        manifest = load_json(root / "fingerprint.json")
        tensors = load_file(str(root / "fingerprint.safetensors"), device="cpu")
        bundle = cls(metadata=dict(manifest["metadata"]), tensors=dict(tensors))
        if bundle.fingerprint_id != manifest["fingerprint_id"]:
            raise ValueError("fingerprint integrity check failed")
        return bundle



def _hashed_indices_signs(values: torch.Tensor, dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    integers = values.to(torch.int64)
    prime = 2_147_483_647
    indices = ((integers * 1_103_515_245 + seed * 12_345) % prime) % dim
    signs = ((integers * 214_013 + seed * 2_531_011) % prime) & 1
    return indices, signs.to(torch.float32).mul_(2.0).sub_(1.0)


def _sketch_integer_values(values: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    output = torch.zeros(dim, dtype=torch.float64)
    if values.numel() == 0:
        return output
    indices, signs = _hashed_indices_signs(values.reshape(-1).cpu(), dim, seed)
    output.scatter_add_(0, indices, signs.to(torch.float64))
    return output


def corpus_fingerprint(
    token_sequences: Iterable[Sequence[int]],
    *,
    raw_texts: Iterable[str] | None = None,
    config: FingerprintConfig = FingerprintConfig(),
) -> FingerprintBundle:
    unigram = torch.zeros(config.unigram_dim, dtype=torch.float64)
    bigram = torch.zeros(config.bigram_dim, dtype=torch.float64)
    lengths = torch.zeros(len(config.length_bins) + 1, dtype=torch.float64)
    sequence_count = 0
    token_count = 0
    bigram_count = 0
    for sequence in token_sequences:
        values = torch.tensor(list(sequence), dtype=torch.int64)
        sequence_count += 1
        token_count += values.numel()
        bucket = next(
            (i for i, upper in enumerate(config.length_bins) if values.numel() <= upper),
            len(config.length_bins),
        )
        lengths[bucket] += 1
        unigram += _sketch_integer_values(values, config.unigram_dim, config.seed)
        if values.numel() > 1:
            pairs = values[:-1] * 2_147_483_647 + values[1:]
            bigram += _sketch_integer_values(pairs, config.bigram_dim, config.seed + 1)
            bigram_count += pairs.numel()
    byte_counts = torch.zeros(config.byte_dim, dtype=torch.float64)
    byte_total = 0
    if raw_texts is not None:
        for text in raw_texts:
            data = torch.tensor(list(text.encode("utf-8", errors="replace")), dtype=torch.int64)
            byte_total += data.numel()
            if data.numel():
                byte_counts += torch.bincount(data, minlength=config.byte_dim).to(torch.float64)
    tensors = {
        "corpus.unigram": (unigram / max(1.0, math.sqrt(token_count))).float(),
        "corpus.bigram": (bigram / max(1.0, math.sqrt(bigram_count))).float(),
        "corpus.length_histogram": (lengths / max(1, sequence_count)).float(),
        "corpus.byte_frequency": (byte_counts / max(1, byte_total)).float(),
        "corpus.scalars": torch.tensor(
            [sequence_count, token_count, byte_total], dtype=torch.float64
        ).log1p().float(),
    }
    return FingerprintBundle(
        metadata={"kind": "corpus", "config": asdict(config)},
        tensors=tensors,
    )


def _count_sketch(vector: torch.Tensor, dim: int, seed: int, *, chunk_size: int = 1_000_000) -> torch.Tensor:
    flat = vector.detach().float().reshape(-1).cpu()
    output = torch.zeros(dim, dtype=torch.float32)
    for start in range(0, flat.numel(), chunk_size):
        end = min(flat.numel(), start + chunk_size)
        offsets = torch.arange(start, end, dtype=torch.int64)
        indices, signs = _hashed_indices_signs(offsets, dim, seed)
        output.scatter_add_(0, indices, flat[start:end] * signs)
    return output / max(1.0, math.sqrt(flat.numel()))

def w0_response_fingerprint(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    role_by_parameter: Mapping[str, str] | None = None,
    config: FingerprintConfig = FingerprintConfig(),
    device: str | torch.device = "cpu",
) -> FingerprintBundle:
    """Compute endpoint-free loss, gradient and hidden-state evidence at W0."""
    role_by_parameter = dict(role_by_parameter or {})
    model = model.to(device)
    model.train()
    loss_values: list[float] = []
    role_gradients: dict[str, torch.Tensor] = {}
    hidden_moments: list[torch.Tensor] = []
    seen = 0
    for batch in batches:
        if seen >= config.max_probe_batches:
            break
        seen += 1
        model.zero_grad(set_to_none=True)
        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise ValueError("W0 probe produced a non-finite loss")
        loss.backward()
        loss_values.append(float(loss.detach().cpu()))
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            role = role_by_parameter.get(name, "other")
            sketch = _count_sketch(
                parameter.grad,
                config.gradient_dim,
                config.seed + int(stable_u64(role) % 100_000),
            )
            role_gradients[role] = role_gradients.get(role, torch.zeros_like(sketch)) + sketch
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            per_layer = []
            for hidden in hidden_states:
                values = hidden.detach().float().cpu()
                per_layer.append(
                    torch.tensor(
                        [
                            values.mean(),
                            values.std(unbiased=False),
                            values.quantile(0.1),
                            values.quantile(0.5),
                            values.quantile(0.9),
                        ]
                    )
                )
            hidden_moments.append(torch.stack(per_layer))
    if not loss_values:
        raise ValueError("at least one W0 probe batch is required")
    losses = torch.tensor(loss_values, dtype=torch.float32)
    tensors: dict[str, torch.Tensor] = {
        "w0.loss_stats": torch.tensor(
            [losses.mean(), losses.std(unbiased=False), losses.min(), losses.max()]
        )
    }
    for role, sketch in sorted(role_gradients.items()):
        tensors[f"w0.gradient.{role}"] = sketch / seen
    if hidden_moments:
        stacked = torch.stack(hidden_moments)
        tensors["w0.hidden_moments"] = stacked.mean(dim=0)
    return FingerprintBundle(
        metadata={"kind": "w0_response", "config": asdict(config), "probe_batches": seen},
        tensors=tensors,
    )


def merge_fingerprints(*bundles: FingerprintBundle) -> FingerprintBundle:
    tensors: dict[str, torch.Tensor] = {}
    metadata = {"kind": "merged", "parts": []}
    for index, bundle in enumerate(bundles):
        metadata["parts"].append(bundle.metadata)
        for name, tensor in bundle.tensors.items():
            key = name if name not in tensors else f"part{index}.{name}"
            tensors[key] = tensor
    return FingerprintBundle(metadata=metadata, tensors=tensors)
