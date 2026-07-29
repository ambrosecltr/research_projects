from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..hashing import sha256_json

Primitive = Literal[
    "BASE_COPY",
    "LOW_RANK",
    "HADAMARD_SCALE",
    "QUANTIZED_VECTOR",
    "SPARSE_PATCH",
    "COPY_FROM_TIED",
]
_ALLOWED = {
    "BASE_COPY",
    "LOW_RANK",
    "HADAMARD_SCALE",
    "QUANTIZED_VECTOR",
    "SPARSE_PATCH",
    "COPY_FROM_TIED",
}


@dataclass(frozen=True)
class Component:
    primitive: Primitive
    payload: dict[str, str] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in _ALLOWED:
            raise ValueError(f"unsupported MGP primitive: {self.primitive}")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.payload.items()
        ):
            raise TypeError("payload references must be string-to-string mappings")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Component:
        return cls(
            primitive=str(value["primitive"]),  # type: ignore[arg-type]
            payload=dict(value.get("payload", {})),
            arguments=dict(value.get("arguments", {})),
        )


@dataclass(frozen=True)
class TensorProgram:
    name: str
    shape: tuple[int, ...]
    components: tuple[Component, ...]
    tied_to: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor program name must not be empty")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError(f"invalid shape for {self.name}: {self.shape}")
        if self.tied_to is not None:
            if len(self.components) != 1 or self.components[0].primitive != "COPY_FROM_TIED":
                raise ValueError("a tied tensor must use exactly one COPY_FROM_TIED component")
        elif not self.components:
            raise ValueError("a tensor program must contain at least one component")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorProgram:
        return cls(
            name=str(value["name"]),
            shape=tuple(int(item) for item in value["shape"]),
            components=tuple(Component.from_dict(item) for item in value["components"]),
            tied_to=None if value.get("tied_to") is None else str(value["tied_to"]),
        )


@dataclass(frozen=True)
class ModelGenomeProgram:
    architecture_id: str
    base_state_id: str
    tensors: tuple[TensorProgram, ...]
    shared_assets: tuple[str, ...] = ()
    format: str = "MODEL_GENOME_PROGRAM"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.format != "MODEL_GENOME_PROGRAM" or self.version != "1.0.0":
            raise ValueError("unsupported MGP format")
        if not self.architecture_id or not self.base_state_id:
            raise ValueError("architecture_id and base_state_id are required")
        names = [item.name for item in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor program names must be unique")
        name_set = set(names)
        for item in self.tensors:
            if item.tied_to is not None and item.tied_to not in name_set:
                raise ValueError(f"unknown tied owner {item.tied_to!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "architecture_id": self.architecture_id,
            "base_state_id": self.base_state_id,
            "shared_assets": list(self.shared_assets),
            "tensors": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "tied_to": item.tied_to,
                    "components": [asdict(component) for component in item.components],
                }
                for item in self.tensors
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelGenomeProgram:
        raw_tensors = value.get("tensors")
        if not isinstance(raw_tensors, Sequence) or isinstance(raw_tensors, (str, bytes)):
            raise TypeError("program.tensors must be an array")
        return cls(
            architecture_id=str(value["architecture_id"]),
            base_state_id=str(value["base_state_id"]),
            tensors=tuple(TensorProgram.from_dict(item) for item in raw_tensors),
            shared_assets=tuple(str(item) for item in value.get("shared_assets", [])),
            format=str(value.get("format", "MODEL_GENOME_PROGRAM")),
            version=str(value.get("version", "1.0.0")),
        )

    @property
    def structural_id(self) -> str:
        return sha256_json(self.to_dict())
