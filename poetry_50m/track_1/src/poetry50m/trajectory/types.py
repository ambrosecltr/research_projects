"""Typed, local-only checkpoint artifacts for trajectory analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor

SNAPSHOT_FORMAT = "poetry50m.weights.v1"


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Identity required to know whether two weight coordinates are comparable."""

    run_id: str
    checkpoint_id: str
    step: int
    initialization_id: str
    data_order_id: str
    architecture_signature: str
    corpus_signature: str
    model_config_hash: str
    tokenizer_hash: str
    code_signature: str
    training_config_hash: str
    wall_seconds: float = 0.0
    tokens_seen: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "checkpoint_id",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "model_config_hash",
            "tokenizer_hash",
            "code_signature",
            "training_config_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("step", "tokens_seen"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0.0
        ):
            raise ValueError("wall_seconds must be a finite non-negative number")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SnapshotMetadata:
        expected = {
            "run_id",
            "checkpoint_id",
            "step",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "model_config_hash",
            "tokenizer_hash",
            "code_signature",
            "training_config_hash",
            "wall_seconds",
            "tokens_seen",
        }
        if set(value) != expected:
            raise ValueError(f"snapshot metadata must contain exactly {sorted(expected)}")
        step = value["step"]
        tokens_seen = value["tokens_seen"]
        wall_seconds = value["wall_seconds"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int):
            raise TypeError("tokens_seen must be an integer")
        if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)):
            raise TypeError("wall_seconds must be a number")
        return cls(
            run_id=_string(value["run_id"], name="run_id"),
            checkpoint_id=_string(value["checkpoint_id"], name="checkpoint_id"),
            step=step,
            initialization_id=_string(value["initialization_id"], name="initialization_id"),
            data_order_id=_string(value["data_order_id"], name="data_order_id"),
            architecture_signature=_string(
                value["architecture_signature"], name="architecture_signature"
            ),
            corpus_signature=_string(value["corpus_signature"], name="corpus_signature"),
            model_config_hash=_string(value["model_config_hash"], name="model_config_hash"),
            tokenizer_hash=_string(value["tokenizer_hash"], name="tokenizer_hash"),
            code_signature=_string(value["code_signature"], name="code_signature"),
            training_config_hash=_string(
                value["training_config_hash"], name="training_config_hash"
            ),
            wall_seconds=float(wall_seconds),
            tokens_seen=tokens_seen,
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """A named state-dict coordinate, including dtype and exact shape."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    floating_point: bool

    @classmethod
    def from_tensor(cls, name: str, tensor: Tensor) -> TensorSpec:
        return cls(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            floating_point=tensor.is_floating_point(),
        )


@dataclass(frozen=True, slots=True)
class WeightSnapshot:
    """Validated weights-only checkpoint held in a named coordinate system."""

    metadata: SnapshotMetadata
    state_dict: Mapping[str, Tensor]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.state_dict:
            raise ValueError("weight snapshots require a non-empty state_dict")
        names = tuple(self.state_dict)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("state_dict names must be non-empty strings")
        for name, value in self.state_dict.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"state_dict[{name!r}] must be a tensor")
            if value.layout != torch.strided:
                raise TypeError(f"state_dict[{name!r}] uses unsupported layout {value.layout}")

    @property
    def tensor_specs(self) -> tuple[TensorSpec, ...]:
        return tuple(
            TensorSpec.from_tensor(name, tensor) for name, tensor in self.state_dict.items()
        )

    @property
    def coordinate_signature(self) -> str:
        payload = [asdict(spec) for spec in self.tensor_specs]
        return sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def cpu_clone(self) -> WeightSnapshot:
        return WeightSnapshot(
            metadata=self.metadata,
            state_dict={
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict.items()
            },
            source_path=self.source_path,
        )
