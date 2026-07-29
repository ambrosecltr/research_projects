from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .codecs.common import make_manifest, make_records
from .mgp.opcodes import COPY_FROM_TIED, LOW_RANK
from .program_compiler import PROGRAM_TOKEN_TO_ID, ProgramTeacherBatch
from .types import GenomeComponent, GenomeProgram, TensorSpec


@dataclass(frozen=True)
class ProgramTokenizationConfig:
    coefficient_chunk_dim: int = 16
    factor_dtype: str = "float16"

    def __post_init__(self) -> None:
        if (
            isinstance(self.coefficient_chunk_dim, bool)
            or not isinstance(self.coefficient_chunk_dim, int)
            or self.coefficient_chunk_dim < 1
        ):
            raise ValueError("coefficient_chunk_dim must be a positive integer")
        if self.factor_dtype not in {"float16", "float32"}:
            raise ValueError("factor_dtype must be float16 or float32")

    @property
    def torch_factor_dtype(self) -> torch.dtype:
        return torch.float16 if self.factor_dtype == "float16" else torch.float32


@dataclass(frozen=True)
class ProgramSequence:
    token_ids: torch.Tensor
    numeric_values: torch.Tensor
    numeric_mask: torch.Tensor
    token_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1:
            raise ValueError("program sequence token IDs must be one-dimensional")
        if self.numeric_values.ndim != 2 or self.numeric_values.shape[0] != self.token_ids.numel():
            raise ValueError("program numeric values must align with token IDs")
        if self.numeric_mask.shape != self.token_ids.shape or self.numeric_mask.dtype != torch.bool:
            raise ValueError("program numeric mask must align with token IDs")
        if self.token_mask.shape != self.token_ids.shape or self.token_mask.dtype != torch.bool:
            raise ValueError("program token mask must align with token IDs")


class _SequenceWriter:
    def __init__(self, chunk_dim: int) -> None:
        self.chunk_dim = chunk_dim
        self.tokens: list[int] = []
        self.numeric: list[torch.Tensor] = []
        self.numeric_mask: list[bool] = []

    def token(self, name: str) -> None:
        self.tokens.append(PROGRAM_TOKEN_TO_ID[name])
        self.numeric.append(torch.zeros(self.chunk_dim, dtype=torch.float32))
        self.numeric_mask.append(False)

    def integer(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("program integer values must be non-negative integers")
        numeric = torch.zeros(self.chunk_dim, dtype=torch.float32)
        numeric[0] = float(value)
        self.tokens.append(PROGRAM_TOKEN_TO_ID["INTEGER"])
        self.numeric.append(numeric)
        self.numeric_mask.append(True)

    def coefficients(self, value: torch.Tensor) -> None:
        flat = value.detach().to(torch.float32).flatten().cpu()
        for start in range(0, flat.numel(), self.chunk_dim):
            chunk = torch.zeros(self.chunk_dim, dtype=torch.float32)
            source = flat[start : start + self.chunk_dim]
            chunk[: source.numel()] = source
            self.tokens.append(PROGRAM_TOKEN_TO_ID["COEFFICIENT_CHUNK"])
            self.numeric.append(chunk)
            self.numeric_mask.append(True)

    def finish(self) -> ProgramSequence:
        token_ids = torch.tensor(self.tokens, dtype=torch.long)
        numeric_values = torch.stack(self.numeric)
        return ProgramSequence(
            token_ids=token_ids,
            numeric_values=numeric_values,
            numeric_mask=torch.tensor(self.numeric_mask, dtype=torch.bool),
            token_mask=torch.ones(token_ids.shape, dtype=torch.bool),
        )


class _SequenceReader:
    def __init__(self, sequence: ProgramSequence, chunk_dim: int) -> None:
        self.sequence = sequence
        self.chunk_dim = chunk_dim
        self.cursor = 0

    def _name(self) -> str:
        token_id = int(self.sequence.token_ids[self.cursor].item())
        for name, candidate in PROGRAM_TOKEN_TO_ID.items():
            if candidate == token_id:
                return name
        raise ValueError(f"unknown program token ID: {token_id}")

    def expect(self, name: str) -> None:
        if self.cursor >= self.sequence.token_ids.numel() or self._name() != name:
            actual = "<end>" if self.cursor >= self.sequence.token_ids.numel() else self._name()
            raise ValueError(f"expected program token {name}, got {actual}")
        self.cursor += 1

    def integer(self) -> int:
        self.expect("INTEGER")
        raw = float(self.sequence.numeric_values[self.cursor - 1, 0].item())
        value = int(round(raw))
        if value < 0 or abs(raw - value) > 1e-4:
            raise ValueError("program integer token is not a non-negative integer")
        return value

    def coefficients(self, count: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("coefficient count must be a non-negative integer")
        chunks = math.ceil(count / self.chunk_dim)
        values = []
        for _ in range(chunks):
            self.expect("COEFFICIENT_CHUNK")
            values.append(self.sequence.numeric_values[self.cursor - 1].to(torch.float32))
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(values)[:count]

    def finished(self) -> bool:
        return self.cursor == self.sequence.token_ids.numel()


def program_to_sequence(
    program: GenomeProgram,
    inventory: Sequence[TensorSpec],
    *,
    config: ProgramTokenizationConfig | None = None,
) -> ProgramSequence:
    """Encode canonical low-rank target programs for compiler teacher forcing.

    This deliberately rejects dense, quantized-per-weight, neural-residual, and exact-residual
    components. Expanding the compiler vocabulary requires a deterministic inverse implementation,
    not an opaque fallback payload.
    """

    config = config or ProgramTokenizationConfig()
    by_name = {record.tensor_name: record for record in program.records}
    if [record.tensor_name for record in program.records] != [spec.name for spec in inventory]:
        raise ValueError("program and inventory order differ")
    writer = _SequenceWriter(config.coefficient_chunk_dim)
    writer.token("BOS")
    for spec in inventory:
        record = by_name[spec.name]
        writer.token("TENSOR_START")
        meaningful = [
            component for component in record.components if component.opcode != COPY_FROM_TIED
        ]
        if not meaningful:
            writer.token("BASE_COPY")
        else:
            if len(meaningful) != 1 or meaningful[0].opcode != LOW_RANK:
                raise ValueError(
                    f"compiler tokenizer supports only canonical LOW_RANK targets; "
                    f"{spec.name} uses {[item.opcode for item in meaningful]}"
                )
            component = meaningful[0]
            if len(component.payload_keys) != 3:
                raise ValueError("LOW_RANK target requires U, S, and Vh payloads")
            u = program.payload_tensors[component.payload_keys[0]]
            s = program.payload_tensors[component.payload_keys[1]]
            vh = program.payload_tensors[component.payload_keys[2]]
            rank = int(component.arguments.get("rank", s.numel()))
            if u.shape != (spec.shape[0], rank) or s.shape != (rank,) or vh.shape != (
                rank,
                spec.shape[1],
            ):
                raise ValueError(f"LOW_RANK payload dimensions differ for {spec.name}")
            writer.token("LOW_RANK")
            writer.integer(rank)
            writer.coefficients(u)
            writer.coefficients(s)
            writer.coefficients(vh)
        writer.token("TENSOR_END")
    writer.token("EOS")
    return writer.finish()


def sequence_to_program(
    sequence: ProgramSequence,
    inventory: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    config: ProgramTokenizationConfig | None = None,
    candidate_id: str = "compiled-program",
) -> GenomeProgram:
    config = config or ProgramTokenizationConfig(
        coefficient_chunk_dim=sequence.numeric_values.shape[1]
    )
    if sequence.numeric_values.shape[1] != config.coefficient_chunk_dim:
        raise ValueError("sequence coefficient width differs from tokenizer configuration")
    reader = _SequenceReader(sequence, config.coefficient_chunk_dim)
    reader.expect("BOS")
    records, aliases = make_records(inventory, tied_groups)
    payload: dict[str, torch.Tensor] = {}
    for spec, record in zip(inventory, records, strict=True):
        reader.expect("TENSOR_START")
        if spec.name in aliases:
            reader.expect("BASE_COPY")
            record.components.append(
                GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[spec.name]})
            )
        else:
            if reader._name() == "BASE_COPY":
                reader.expect("BASE_COPY")
            elif reader._name() == "LOW_RANK":
                if len(spec.shape) != 2:
                    raise ValueError(f"LOW_RANK token cannot target non-matrix tensor {spec.name}")
                reader.expect("LOW_RANK")
                rank = reader.integer()
                maximum_rank = min(spec.shape)
                if rank < 1 or rank > maximum_rank:
                    raise ValueError(f"LOW_RANK rank is invalid for {spec.name}")
                u_count = spec.shape[0] * rank
                s_count = rank
                vh_count = rank * spec.shape[1]
                u = reader.coefficients(u_count).reshape(spec.shape[0], rank)
                s = reader.coefficients(s_count)
                vh = reader.coefficients(vh_count).reshape(rank, spec.shape[1])
                prefix = f"t{record.canonical_index:05d}.low_rank"
                keys = [f"{prefix}.u", f"{prefix}.s", f"{prefix}.vh"]
                payload[keys[0]] = u.to(config.torch_factor_dtype).contiguous()
                payload[keys[1]] = s.to(config.torch_factor_dtype).contiguous()
                payload[keys[2]] = vh.to(config.torch_factor_dtype).contiguous()
                record.components.append(
                    GenomeComponent(
                        LOW_RANK,
                        payload_keys=keys,
                        arguments={
                            "rank": rank,
                            "factor_dtype": config.factor_dtype,
                            "canonical_sign": "max_abs_u_pivot_positive",
                        },
                    )
                )
            else:
                raise ValueError(f"unsupported tensor program token: {reader._name()}")
        reader.expect("TENSOR_END")
    reader.expect("EOS")
    if not reader.finished():
        raise ValueError("program sequence contains trailing tokens")
    manifest = make_manifest(
        candidate_id=candidate_id,
        codec="compiled_canonical_program_v2",
        metadata={
            "compiler_target": True,
            "contains_exact_residual": False,
            "target_language": "canonical_low_rank",
        },
    )
    manifest["tokenization"] = {
        "coefficient_chunk_dim": config.coefficient_chunk_dim,
        "factor_dtype": config.factor_dtype,
    }
    return GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)


def collate_program_sequences(
    sequences: Sequence[ProgramSequence],
) -> ProgramTeacherBatch:
    if not sequences:
        raise ValueError("cannot collate an empty program batch")
    chunk_dim = sequences[0].numeric_values.shape[1]
    if any(sequence.numeric_values.shape[1] != chunk_dim for sequence in sequences):
        raise ValueError("program sequences use different coefficient chunk widths")
    maximum = max(sequence.token_ids.numel() for sequence in sequences)
    batch = len(sequences)
    pad_id = PROGRAM_TOKEN_TO_ID["PAD"]
    token_ids = torch.full((batch, maximum), pad_id, dtype=torch.long)
    numeric_values = torch.zeros(batch, maximum, chunk_dim, dtype=torch.float32)
    numeric_mask = torch.zeros(batch, maximum, dtype=torch.bool)
    token_mask = torch.zeros(batch, maximum, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = sequence.token_ids.numel()
        token_ids[index, :length] = sequence.token_ids
        numeric_values[index, :length] = sequence.numeric_values
        numeric_mask[index, :length] = sequence.numeric_mask
        token_mask[index, :length] = sequence.token_mask
    return ProgramTeacherBatch(
        token_ids=token_ids,
        numeric_values=numeric_values,
        numeric_mask=numeric_mask,
        token_mask=token_mask,
    )


def teacher_forcing_shift(
    batch: ProgramTeacherBatch,
) -> tuple[torch.Tensor, torch.Tensor, ProgramTeacherBatch]:
    if batch.token_ids.shape[1] < 2:
        raise ValueError("teacher-forcing sequence must contain at least two tokens")
    decoder_tokens = batch.token_ids[:, :-1]
    decoder_numeric = batch.numeric_values[:, :-1]
    targets = ProgramTeacherBatch(
        token_ids=batch.token_ids[:, 1:],
        numeric_values=batch.numeric_values[:, 1:],
        numeric_mask=batch.numeric_mask[:, 1:],
        token_mask=batch.token_mask[:, 1:],
    )
    return decoder_tokens, decoder_numeric, targets
