from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ..hashing import sha256_tensor
from ..tensor_inventory import assert_tied_equal, restore_tied_values, tied_owner_map
from ..types import GenomeComponent, GenomeProgram, TensorSpec
from .opcodes import (
    BASE_COPY,
    CODEBOOK_BLOCKS,
    COPY_FROM_TIED,
    DENSE_DELTA,
    KRONECKER,
    LOW_RANK,
    LOW_RANK_PATCH,
    NEURAL_BLOCK_FIELD,
    QUANTIZED_DELTA,
    SHARED_BASIS,
    SPARSE_PATCH,
    SPECTRAL_DCT,
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


def _matrix_shape(args: Mapping[str, Any], value: torch.Tensor) -> tuple[int, int]:
    raw = args.get("matrix_shape")
    if raw is None:
        return (1, value.numel()) if value.ndim < 2 else (value.shape[0], value.numel() // value.shape[0])
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in raw)
    ):
        raise ValueError("matrix_shape must contain two positive integers")
    rows, cols = int(raw[0]), int(raw[1])
    if rows * cols != value.numel():
        raise ValueError("matrix_shape does not match the target tensor size")
    return rows, cols


def _decode_kronecker(
    component: GenomeComponent,
    program: GenomeProgram,
    value: torch.Tensor,
) -> torch.Tensor:
    keys = component.payload_keys
    terms = component.arguments.get("terms")
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 1:
        raise ValueError("KRONECKER requires a positive term count")
    if len(keys) != terms * 3:
        raise ValueError("KRONECKER payloads must be A, B, coefficient triples")
    rows, cols = _matrix_shape(component.arguments, value)
    decoded = torch.zeros(rows, cols, dtype=torch.float32, device=value.device)
    for index in range(terms):
        left = _payload(program, keys[index * 3]).to(device=value.device, dtype=torch.float32)
        right = _payload(program, keys[index * 3 + 1]).to(
            device=value.device, dtype=torch.float32
        )
        coefficient = _payload(program, keys[index * 3 + 2]).to(torch.float32).reshape(())
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("KRONECKER factors must be matrices")
        term = torch.kron(left, right)
        if term.shape != (rows, cols):
            raise ValueError(
                f"KRONECKER term has shape {tuple(term.shape)}, expected {(rows, cols)}"
            )
        decoded += coefficient.to(value.device) * term
    return decoded.reshape_as(value)


def _decode_spectral_dct(
    component: GenomeComponent,
    program: GenomeProgram,
    value: torch.Tensor,
) -> torch.Tensor:
    if len(component.payload_keys) != 2:
        raise ValueError("SPECTRAL_DCT requires frequency indices and coefficients")
    rows, cols = _matrix_shape(component.arguments, value)
    indices = _payload(program, component.payload_keys[0]).to(torch.int64).cpu()
    coefficients = _payload(program, component.payload_keys[1]).to(
        device=value.device, dtype=torch.float32
    )
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("SPECTRAL_DCT indices must have shape [count, 2]")
    if coefficients.ndim != 1 or coefficients.numel() != indices.shape[0]:
        raise ValueError("SPECTRAL_DCT coefficient count differs from index count")
    if indices.numel() and (
        bool(torch.any(indices[:, 0] < 0))
        or bool(torch.any(indices[:, 0] >= rows))
        or bool(torch.any(indices[:, 1] < 0))
        or bool(torch.any(indices[:, 1] >= cols))
    ):
        raise ValueError("SPECTRAL_DCT frequency index is out of range")
    if not indices.shape[0]:
        return torch.zeros_like(value, dtype=torch.float32)

    frequencies_u = indices[:, 0].to(device=value.device, dtype=torch.float32)
    frequencies_v = indices[:, 1].to(device=value.device, dtype=torch.float32)
    positions_i = torch.arange(rows, device=value.device, dtype=torch.float32).unsqueeze(1)
    positions_j = torch.arange(cols, device=value.device, dtype=torch.float32).unsqueeze(1)
    row_scale = torch.where(
        frequencies_u.eq(0),
        torch.full_like(frequencies_u, math.sqrt(1.0 / rows)),
        torch.full_like(frequencies_u, math.sqrt(2.0 / rows)),
    )
    col_scale = torch.where(
        frequencies_v.eq(0),
        torch.full_like(frequencies_v, math.sqrt(1.0 / cols)),
        torch.full_like(frequencies_v, math.sqrt(2.0 / cols)),
    )
    row_basis = row_scale.unsqueeze(0) * torch.cos(
        math.pi * (2.0 * positions_i + 1.0) * frequencies_u.unsqueeze(0) / (2.0 * rows)
    )
    col_basis = col_scale.unsqueeze(0) * torch.cos(
        math.pi * (2.0 * positions_j + 1.0) * frequencies_v.unsqueeze(0) / (2.0 * cols)
    )
    decoded = (row_basis * coefficients.unsqueeze(0)) @ col_basis.transpose(0, 1)
    return decoded.reshape_as(value)


def _decode_shared_basis(
    component: GenomeComponent,
    program: GenomeProgram,
    value: torch.Tensor,
) -> torch.Tensor:
    if len(component.payload_keys) != 2:
        raise ValueError("SHARED_BASIS requires basis and coefficient payloads")
    basis = _payload(program, component.payload_keys[0]).to(
        device=value.device, dtype=torch.float32
    )
    coefficients = _payload(program, component.payload_keys[1]).to(
        device=value.device, dtype=torch.float32
    )
    if basis.ndim < 2:
        raise ValueError("SHARED_BASIS basis must have a leading component dimension")
    basis_flat = basis.reshape(basis.shape[0], -1)
    coefficients = coefficients.flatten()
    if basis_flat.shape[0] != coefficients.numel() or basis_flat.shape[1] != value.numel():
        raise ValueError("SHARED_BASIS dimensions do not match coefficients or target tensor")
    return (coefficients @ basis_flat).reshape_as(value)


def _decode_codebook_blocks(
    component: GenomeComponent,
    program: GenomeProgram,
    value: torch.Tensor,
) -> torch.Tensor:
    if len(component.payload_keys) not in {2, 3}:
        raise ValueError("CODEBOOK_BLOCKS requires codebook, indices, and optional scales")
    rows, cols = _matrix_shape(component.arguments, value)
    block_rows = component.arguments.get("block_rows")
    block_cols = component.arguments.get("block_cols")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or block_rows < 1
        or isinstance(block_cols, bool)
        or not isinstance(block_cols, int)
        or block_cols < 1
    ):
        raise ValueError("CODEBOOK_BLOCKS requires positive block dimensions")
    codebook = _payload(program, component.payload_keys[0]).to(
        device=value.device, dtype=torch.float32
    )
    indices = _payload(program, component.payload_keys[1]).to(
        device=value.device, dtype=torch.int64
    ).flatten()
    if codebook.ndim != 3 or codebook.shape[1:] != (block_rows, block_cols):
        raise ValueError("CODEBOOK_BLOCKS codebook has the wrong shape")
    row_blocks = math.ceil(rows / block_rows)
    col_blocks = math.ceil(cols / block_cols)
    if indices.numel() != row_blocks * col_blocks:
        raise ValueError("CODEBOOK_BLOCKS index count differs from the tensor block count")
    if indices.numel() and (indices.min() < 0 or indices.max() >= codebook.shape[0]):
        raise ValueError("CODEBOOK_BLOCKS index is outside the codebook")
    if len(component.payload_keys) == 3:
        scales = _payload(program, component.payload_keys[2]).to(
            device=value.device, dtype=torch.float32
        ).flatten()
        if scales.numel() not in {1, indices.numel()}:
            raise ValueError("CODEBOOK_BLOCKS scales must be scalar or one per block")
    else:
        scales = torch.ones(1, device=value.device)

    decoded = torch.zeros(rows, cols, dtype=torch.float32, device=value.device)
    cursor = 0
    for row_block in range(row_blocks):
        for col_block in range(col_blocks):
            row_start = row_block * block_rows
            col_start = col_block * block_cols
            row_end = min(row_start + block_rows, rows)
            col_end = min(col_start + block_cols, cols)
            block = codebook[indices[cursor]]
            scale = scales[0] if scales.numel() == 1 else scales[cursor]
            decoded[row_start:row_end, col_start:col_end] = (
                block[: row_end - row_start, : col_end - col_start] * scale
            )
            cursor += 1
    return decoded.reshape_as(value)


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
        return value.to(work_dtype) + payload.to(device=value.device, dtype=work_dtype).reshape(
            value.shape
        )
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
            decoded = unpack_int4(quantized, int(torch.tensor(shape).prod().item())).to(
                value.device
            )
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
    if opcode == KRONECKER:
        return value + _decode_kronecker(component, program, value)
    if opcode == SPECTRAL_DCT:
        return value + _decode_spectral_dct(component, program, value)
    if opcode == SHARED_BASIS:
        return value + _decode_shared_basis(component, program, value)
    if opcode == CODEBOOK_BLOCKS:
        return value + _decode_codebook_blocks(component, program, value)
    if opcode == SPARSE_PATCH:
        if len(keys) != 2:
            raise ValueError("SPARSE_PATCH requires indices and values")
        indices = _payload(program, keys[0]).to(device=value.device, dtype=torch.int64).flatten()
        patch_values = (
            _payload(program, keys[1]).to(device=value.device, dtype=torch.float32).flatten()
        )
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
        return value + decoded.to(device=value.device, dtype=torch.float32).reshape_as(value)
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
