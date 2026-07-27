from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..types import GenomeProgram, TensorSpec
from .opcodes import SUPPORTED_OPCODES


def dtype_from_name(name: str) -> torch.dtype:
    clean = name.replace("torch.", "")
    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if clean not in mapping:
        raise ValueError(f"unsupported dtype name: {name!r}")
    return mapping[clean]


def validate_program(
    program: GenomeProgram,
    inventory: Sequence[TensorSpec] | None = None,
    *,
    contract: Mapping[str, str] | None = None,
) -> None:
    manifest = program.manifest
    if manifest.get("format") != "MGP":
        raise ValueError("manifest format is not MGP")
    version = str(manifest.get("version", ""))
    if version not in {"0.1.0"}:
        raise ValueError(f"unsupported MGP version: {version!r}")
    if not manifest.get("candidate_id"):
        raise ValueError("manifest lacks candidate_id")
    if contract:
        for key, expected in contract.items():
            actual = manifest.get(key)
            if actual != expected:
                raise ValueError(f"MGP contract mismatch for {key}: {actual!r} != {expected!r}")

    names = [record.tensor_name for record in program.records]
    if len(names) != len(set(names)):
        raise ValueError("MGP contains duplicate tensor records")
    indices = [record.canonical_index for record in program.records]
    if indices != sorted(indices):
        raise ValueError("MGP records are not in canonical order")

    all_payloads = {**program.payload_tensors, **program.patch_tensors}
    referenced: set[str] = set()
    raw_shared_payloads = manifest.get("shared_payload_keys", [])
    if (
        not isinstance(raw_shared_payloads, list)
        or any(not isinstance(key, str) or not key for key in raw_shared_payloads)
        or len(raw_shared_payloads) != len(set(raw_shared_payloads))
    ):
        raise ValueError("shared_payload_keys must be a unique array of non-empty strings")
    shared_payloads = set(raw_shared_payloads)
    for record in program.records:
        if any(dimension < 0 for dimension in record.shape):
            raise ValueError(f"invalid shape for {record.tensor_name}")
        dtype_from_name(record.output_dtype)
        for component in record.components:
            if component.opcode not in SUPPORTED_OPCODES:
                raise ValueError(f"unsupported opcode: {component.opcode}")
            for key in component.payload_keys:
                if key not in all_payloads:
                    raise ValueError(f"record {record.tensor_name} references missing payload {key!r}")
                if key in referenced and key not in shared_payloads:
                    raise ValueError(f"payload {key!r} is referenced more than once")
                referenced.add(key)
                tensor = all_payloads[key]
                if not torch.isfinite(tensor).all() and tensor.is_floating_point():
                    raise ValueError(f"payload {key!r} contains NaN or Inf")
    missing_shared = shared_payloads - set(all_payloads)
    if missing_shared:
        raise ValueError(f"manifest declares missing shared payloads: {sorted(missing_shared)}")
    referenced.update(shared_payloads)
    unreferenced = set(all_payloads) - referenced
    if unreferenced:
        raise ValueError(f"MGP contains unreferenced payload tensors: {sorted(unreferenced)}")

    if inventory is not None:
        expected_names = [spec.name for spec in inventory]
        if names != expected_names:
            missing = sorted(set(expected_names) - set(names))
            extra = sorted(set(names) - set(expected_names))
            raise ValueError(f"MGP inventory mismatch; missing={missing}, extra={extra}")
        by_name = {spec.name: spec for spec in inventory}
        for record in program.records:
            spec = by_name[record.tensor_name]
            if record.canonical_index != spec.canonical_index:
                raise ValueError(f"canonical index mismatch for {record.tensor_name}")
            if tuple(record.shape) != tuple(spec.shape):
                raise ValueError(f"shape mismatch for {record.tensor_name}")
