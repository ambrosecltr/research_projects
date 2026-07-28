from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TensorSpec:
    canonical_index: int
    name: str
    role: str
    layer_index: int | None
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int
    tied_group: str | None = None
    initialization: dict[str, Any] = field(default_factory=dict)
    is_buffer: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorSpec":
        data = dict(value)
        data["shape"] = tuple(int(x) for x in data["shape"])
        return cls(**data)


@dataclass(frozen=True)
class GenomeBudget:
    target_bytes: int | None = None
    fraction_of_raw: float | None = None
    max_rank: int | None = None
    sparse_fraction: float = 0.0

    def resolve(self, raw_bytes: int) -> int | None:
        if self.target_bytes is not None:
            if self.target_bytes < 0:
                raise ValueError("target_bytes must be non-negative")
            return self.target_bytes
        if self.fraction_of_raw is not None:
            if not 0.0 <= self.fraction_of_raw <= 1.0:
                raise ValueError("fraction_of_raw must be in [0, 1]")
            return int(raw_bytes * self.fraction_of_raw)
        return None


@dataclass
class GenomeComponent:
    opcode: str
    payload_keys: list[str] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "payload_keys": list(self.payload_keys),
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenomeComponent":
        return cls(
            opcode=str(value["opcode"]),
            payload_keys=list(value.get("payload_keys", [])),
            arguments=dict(value.get("arguments", {})),
        )


@dataclass
class TensorGenomeRecord:
    tensor_name: str
    canonical_index: int
    role: str
    layer_index: int | None
    shape: tuple[int, ...]
    output_dtype: str
    base_source: str = "W0"
    components: list[GenomeComponent] = field(default_factory=list)
    tied_owner: str | None = None
    output_checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "canonical_index": self.canonical_index,
            "role": self.role,
            "layer_index": self.layer_index,
            "shape": list(self.shape),
            "output_dtype": self.output_dtype,
            "base_source": self.base_source,
            "components": [item.to_dict() for item in self.components],
            "tied_owner": self.tied_owner,
            "output_checksum": self.output_checksum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorGenomeRecord":
        return cls(
            tensor_name=str(value["tensor_name"]),
            canonical_index=int(value["canonical_index"]),
            role=str(value["role"]),
            layer_index=(None if value.get("layer_index") is None else int(value["layer_index"])),
            shape=tuple(int(x) for x in value["shape"]),
            output_dtype=str(value["output_dtype"]),
            base_source=str(value.get("base_source", "W0")),
            components=[GenomeComponent.from_dict(x) for x in value.get("components", [])],
            tied_owner=value.get("tied_owner"),
            output_checksum=value.get("output_checksum"),
        )


@dataclass
class GenomeProgram:
    manifest: dict[str, Any]
    records: list[TensorGenomeRecord]
    payload_tensors: dict[str, torch.Tensor]
    patch_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    source_path: Path | None = None

    def clone_without_payload_aliases(self) -> "GenomeProgram":
        return GenomeProgram(
            manifest=dict(self.manifest),
            records=[TensorGenomeRecord.from_dict(x.to_dict()) for x in self.records],
            payload_tensors={k: v.detach().clone() for k, v in self.payload_tensors.items()},
            patch_tensors={k: v.detach().clone() for k, v in self.patch_tensors.items()},
            source_path=self.source_path,
        )


@dataclass(frozen=True)
class BitBreakdown:
    manifest: int = 0
    codes: int = 0
    factors: int = 0
    quantized_values: int = 0
    scales: int = 0
    indices: int = 0
    patch: int = 0
    exact_residual: int = 0
    interpreter: int = 0
    dictionaries: int = 0
    base: int = 0
    container_overhead: int = 0

    @property
    def target_specific(self) -> int:
        return (
            self.manifest
            + self.codes
            + self.factors
            + self.quantized_values
            + self.scales
            + self.indices
            + self.patch
            + self.exact_residual
            + self.container_overhead
        )

    @property
    def single_model_total(self) -> int:
        return self.target_specific + self.interpreter + self.dictionaries + self.base

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data["target_specific"] = self.target_specific
        data["single_model_total"] = self.single_model_total
        return data


@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    mgp_sha256: str | None
    validity: dict[str, Any]
    bytes: dict[str, int]
    compute: dict[str, float]
    parameter_metrics: dict[str, Any]
    functional_metrics: dict[str, Any]
    generation_metrics: dict[str, Any]
    decision: str
    failure_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_codes"] = list(self.failure_codes)
        return data
