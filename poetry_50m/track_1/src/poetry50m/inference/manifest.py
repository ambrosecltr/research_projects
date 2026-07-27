"""Canonical JSONL execution records for evaluation generation manifests."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from tokenizers import Tokenizer

from poetry50m.evaluation.schema import GenerationRequest
from poetry50m.inference.generation import GenerationConfig, generate
from poetry50m.model import DecoderOnlyTransformer

StopReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One canonical result for a sealed evaluation generation request."""

    request_id: str
    case_id: str
    checkpoint_id: str
    seed: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: StopReason
    generated_token_count: int
    wall_seconds: float

    def __post_init__(self) -> None:
        for name in ("request_id", "case_id", "checkpoint_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (
            ("seed", self.seed),
            ("generated_token_count", self.generated_token_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float)):
            raise TypeError("wall_seconds must be a finite number")
        if not math.isfinite(self.wall_seconds):
            raise ValueError("wall_seconds must be finite")
        if self.seed < 0 or self.generated_token_count < 0 or self.wall_seconds < 0.0:
            raise ValueError("invalid generation count, seed, or wall_seconds")
        if self.generated_token_count != len(self.generated_token_ids):
            raise ValueError("generated_token_count must equal generated_token_ids length")
        if not isinstance(self.generated_text, str):
            raise TypeError("generated_text must be a string")
        if not isinstance(self.generated_token_ids, tuple):
            raise TypeError("generated_token_ids must be a tuple")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in self.generated_token_ids
        ):
            raise ValueError("generated_token_ids must contain non-negative integers")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("invalid stop reason")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GenerationRecord:
        data = dict(value)
        expected = {
            "request_id",
            "case_id",
            "checkpoint_id",
            "seed",
            "generated_token_ids",
            "generated_text",
            "stop_reason",
            "generated_token_count",
            "wall_seconds",
        }
        if set(data) != expected:
            raise ValueError(f"generation record must contain exactly {sorted(expected)}")
        ids = data.get("generated_token_ids")
        if not isinstance(ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in ids
        ):
            raise ValueError("generated_token_ids must be a JSON integer list")
        required_strings = (
            "request_id",
            "case_id",
            "checkpoint_id",
            "generated_text",
            "stop_reason",
        )
        if any(not isinstance(data.get(name), str) for name in required_strings):
            raise ValueError("generation record string fields are malformed")
        required_integers = ("seed", "generated_token_count")
        if any(
            isinstance(data.get(name), bool) or not isinstance(data.get(name), int)
            for name in required_integers
        ):
            raise ValueError("generation record integer fields are malformed")
        wall_seconds = data.get("wall_seconds")
        if not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool):
            raise ValueError("generation record wall_seconds is malformed")
        if not math.isfinite(wall_seconds):
            raise ValueError("generation record wall_seconds must be finite")
        return cls(
            request_id=cast(str, data["request_id"]),
            case_id=cast(str, data["case_id"]),
            checkpoint_id=cast(str, data["checkpoint_id"]),
            seed=cast(int, data["seed"]),
            generated_token_ids=tuple(ids),
            generated_text=cast(str, data["generated_text"]),
            stop_reason=cast(StopReason, data["stop_reason"]),
            generated_token_count=cast(int, data["generated_token_count"]),
            wall_seconds=float(wall_seconds),
        )

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["generated_token_ids"] = list(self.generated_token_ids)
        return value


def run_generation_manifest(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    requests: Iterable[GenerationRequest],
    output_path: Path,
    *,
    thoughts_by_case: Mapping[str, str] | None = None,
) -> tuple[GenerationRecord, ...]:
    """Execute a fixed manifest and atomically persist canonical JSONL records."""
    requests_tuple = tuple(requests)
    request_ids = [request.request_id for request in requests_tuple]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("generation manifest request IDs must be unique")
    thoughts_by_case = thoughts_by_case or {}
    unknown_thought_cases = set(thoughts_by_case).difference(
        request.case_id for request in requests_tuple
    )
    if unknown_thought_cases:
        raise ValueError(f"thoughts supplied for unknown cases: {sorted(unknown_thought_cases)!r}")
    records: list[GenerationRecord] = []
    for request in requests_tuple:
        result = generate(
            model,
            tokenizer,
            request.prompt,
            GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            ),
            thought=thoughts_by_case.get(request.case_id),
        )
        records.append(
            GenerationRecord(
                request_id=request.request_id,
                case_id=request.case_id,
                checkpoint_id=request.checkpoint_id,
                seed=request.seed,
                generated_token_ids=result.generated_token_ids,
                generated_text=result.generated_text,
                stop_reason=result.stop_reason,
                generated_token_count=result.generated_token_count,
                wall_seconds=result.wall_seconds,
            )
        )
    ordered = tuple(sorted(records, key=lambda record: record.request_id))
    save_generation_records(output_path, ordered)
    return ordered


def save_generation_records(path: Path, records: Iterable[GenerationRecord]) -> None:
    """Atomically write canonical JSONL sorted by request ID."""
    ordered = tuple(sorted(records, key=lambda record: record.request_id))
    if len({record.request_id for record in ordered}) != len(ordered):
        raise ValueError("generation records must have unique request IDs")
    payload = "".join(
        json.dumps(record.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in ordered
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_generation_records(path: Path) -> tuple[GenerationRecord, ...]:
    """Load a canonical JSONL output file and reject malformed or duplicate rows."""
    records: list[GenerationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in generation records at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"generation record at line {line_number} must be a JSON object")
            records.append(GenerationRecord.from_mapping(value))
    if tuple(record.request_id for record in records) != tuple(
        sorted(record.request_id for record in records)
    ):
        raise ValueError("generation records must be ordered by request_id")
    if len({record.request_id for record in records}) != len(records):
        raise ValueError("generation records must have unique request IDs")
    return tuple(records)
