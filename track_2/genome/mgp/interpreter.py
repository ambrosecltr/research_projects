from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..hashing import sha256_tensor
from ..tensor_inventory import assert_tied_equal, restore_tied_values, tied_owner_map
from ..types import GenomeComponent, GenomeProgram, TensorSpec
from .opcodes import (
    BASE_COPY,
    COPY_FROM_TIED,
    DENSE_DELTA,
    LOW_RANK,
    LOW_RANK_PATCH,
    NEURAL_BLOCK_FIELD,
    QUANTIZED_DELTA,
    SPARSE_PATCH,
)
from .validation import dtype_from_name, validate_program


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).flatten()
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    values = torch.empty(packed.numel() * 2, dtype=torch.int8)
    values[0::2] = low.to(torch.int8) - 8
    values[1::2] = high.to(torch.int8) - 8
    return values[:numel]


def _payload(program: GenomeProgram, key: str) -> torch.Tensor:
    if key in program.payload_tensors:
        return program.payload_tensors[key]
    if key in program.patch_tensors:
        return program.patch_tensors[key]
    raise KeyError(key)


def _apply_component(
    value: torch.Tensor,
    component: GenomeComponent,
    program: GenomeProgram,
    *,
    record_name: str,
    interpreter: Any | None,
) -> torch.Tensor:
    opcode = component.opcode
    args = component.arguments
    keys = component.payload_keys
    if opcode == BASE_COPY:
        if keys:
            raise ValueError("BASE_COPY does not accept payloads")
        return value
    if opcode == DENSE_DELTA:
        if len(keys) != 1:
            raise ValueError("DENSE_DELTA requires one payload")
        payload = _payload(program, keys[0])
        work_dtype = payload.dtype if payload.is_floating_point() else torch.float64
        return value.to(work_dtype) + payload.to(device=value.device, dtype=work_dtype).reshape(value.shape)
    if opcode == QUANTIZED_DELTA:
        bits = int(args["bits"])
        shape = tuple(int(x) for x in args["shape"])
        if len(keys) != 2:
            raise ValueError("QUANTIZED_DELTA requires values and scale payloads")
        quantized = _payload(program, keys[0])
        scale = _payload(program, keys[1]).to(torch.float32).reshape(()).item()
        if bits == 8:
            decoded = quantized.to(device=value.device, dtype=torch.float32).reshape(shape) * scale
        elif bits == 4:
            decoded = unpack_int4(quantized, int(torch.tensor(shape).prod().item())).to(value.device)
            decoded = decoded.to(torch.float32).reshape(shape) * scale
        else:
            raise ValueError(f"unsupported quantization bits: {bits}")
        return value + decoded
    if opcode in {LOW_RANK, LOW_RANK_PATCH}:
        if len(keys) != 3:
            raise ValueError(f"{opcode} requires U, S, Vh payloads")
        u = _payload(program, keys[0]).to(device=value.device, dtype=torch.float32)
        s = _payload(program, keys[1]).to(device=value.device, dtype=torch.float32)
        vh = _payload(program, keys[2]).to(device=value.device, dtype=torch.float32)
        decoded = (u * s.unsqueeze(0)) @ vh
        return value + decoded.reshape_as(value)
    if opcode == SPARSE_PATCH:
        if len(keys) != 2:
            raise ValueError("SPARSE_PATCH requires indices and values")
        indices = _payload(program, keys[0]).to(device=value.device, dtype=torch.int64).flatten()
        patch_values = _payload(program, keys[1]).to(device=value.device, dtype=torch.float32).flatten()
        if indices.numel() != patch_values.numel():
            raise ValueError("sparse patch index/value counts differ")
        flat = value.flatten().clone()
        if indices.numel() and (indices.min() < 0 or indices.max() >= flat.numel()):
            raise ValueError("sparse patch index out of range")
        flat.index_add_(0, indices, patch_values)
        return flat.reshape_as(value)
    if opcode == NEURAL_BLOCK_FIELD:
        if interpreter is None:
            raise ValueError("NEURAL_BLOCK_FIELD requires a loaded neural interpreter")
        decoded = interpreter.decode_tensor_from_component(
            record_name, component, program, base_tensor=value
        )
        return value + decoded.to(torch.float32).reshape_as(value)
    if opcode == COPY_FROM_TIED:
        return value
    raise ValueError(f"unsupported opcode during decode: {opcode}")


def decode_program(
    program: GenomeProgram,
    base_state: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    interpreter: Any | None = None,
    contract: Mapping[str, str] | None = None,
    verify_checksums: bool = True,
) -> dict[str, torch.Tensor]:
    validate_program(program, inventory, contract=contract)
    owner_by_alias = tied_owner_map(tied_groups)
    output: dict[str, torch.Tensor] = {}
    for record in program.records:
        if record.tensor_name not in base_state:
            raise KeyError(f"base state missing {record.tensor_name}")
        if record.tied_owner is not None or record.tensor_name in owner_by_alias:
            continue
        if record.base_source != "W0":
            raise ValueError(f"unsupported base source {record.base_source!r}")
        value = base_state[record.tensor_name].detach().to(torch.float32).clone()
        if tuple(value.shape) != tuple(record.shape):
            raise ValueError(f"base shape mismatch for {record.tensor_name}")
        for component in record.components:
            value = _apply_component(
                value,
                component,
                program,
                record_name=record.tensor_name,
                interpreter=interpreter,
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"decoded tensor contains NaN/Inf: {record.tensor_name}")
        value = value.to(dtype_from_name(record.output_dtype)).contiguous()
        if record.output_checksum and verify_checksums:
            actual = sha256_tensor(value)
            if actual != record.output_checksum:
                raise ValueError(f"decoded checksum mismatch for {record.tensor_name}")
        output[record.tensor_name] = value

    restore_tied_values(output, tied_groups)
    missing = [spec.name for spec in inventory if spec.name not in output]
    if missing:
        raise ValueError(f"decoder did not produce tensors: {missing}")
    assert_tied_equal(output, tied_groups)
    return output
