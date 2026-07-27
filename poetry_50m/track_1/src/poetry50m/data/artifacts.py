"""Canonical JSONL artifacts used between corpus preparation and training."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .packing import PackedSequence
from .schema import (
    ConditionalExample,
    CrossDocumentPairing,
    PromptRecord,
    ProseNTPExample,
    ThoughtRecord,
)

T = TypeVar("T")


def _canonical_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_records(
    path: Path,
    records: Iterable[T],
    identifier: Callable[[T], str],
    serializer: Callable[[T], Mapping[str, Any]],
) -> None:
    ordered = tuple(sorted(records, key=identifier))
    ids = [identifier(record) for record in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate artifact IDs in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(_canonical_line(serializer(record)))


def _iter_records(path: Path, decoder: Callable[[Mapping[str, Any]], T]) -> Iterator[T]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank artifact record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise ValueError(f"{path}:{line_number} must be a JSON object with string keys")
            try:
                yield decoder(value)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid artifact record at {path}:{line_number}: {error}"
                ) from error


def _read_unique(
    path: Path, decoder: Callable[[Mapping[str, Any]], T], identifier: Callable[[T], str]
) -> tuple[T, ...]:
    records = tuple(_iter_records(path, decoder))
    ids = [identifier(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate artifact IDs in {path}")
    return records


def write_prompt_records(path: Path, records: Iterable[PromptRecord]) -> None:
    _write_records(path, records, lambda item: item.prompt_id, asdict)


def read_prompt_records(path: Path) -> tuple[PromptRecord, ...]:
    return _read_unique(path, lambda value: PromptRecord(**value), lambda item: item.prompt_id)


def write_thought_records(path: Path, records: Iterable[ThoughtRecord]) -> None:
    _write_records(path, records, lambda item: item.thought_id, asdict)


def read_thought_records(path: Path) -> tuple[ThoughtRecord, ...]:
    return _read_unique(path, lambda value: ThoughtRecord(**value), lambda item: item.thought_id)


def write_pairings(path: Path, records: Iterable[CrossDocumentPairing]) -> None:
    _write_records(path, records, lambda item: item.pairing_id, asdict)


def read_pairings(path: Path) -> tuple[CrossDocumentPairing, ...]:
    def decode(value: Mapping[str, Any]) -> CrossDocumentPairing:
        data = dict(value)
        lineage = data.get("transformation_lineage", ())
        if not isinstance(lineage, list) or any(
            not isinstance(item, str) or not item for item in lineage
        ):
            raise ValueError("transformation_lineage must be a JSON list of non-empty strings")
        data["transformation_lineage"] = tuple(lineage)
        return CrossDocumentPairing(**data)

    return _read_unique(path, decode, lambda item: item.pairing_id)


def write_conditional_examples(path: Path, records: Iterable[ConditionalExample]) -> None:
    _write_records(path, records, lambda item: item.example_id, asdict)


def read_conditional_examples(path: Path) -> tuple[ConditionalExample, ...]:
    def decode(value: Mapping[str, Any]) -> ConditionalExample:
        data = dict(value)
        lineage = data.get("transformation_lineage", ())
        if not isinstance(lineage, list) or any(
            not isinstance(item, str) or not item for item in lineage
        ):
            raise ValueError("transformation_lineage must be a JSON list of non-empty strings")
        data["transformation_lineage"] = tuple(lineage)
        return ConditionalExample(**data)

    return _read_unique(path, decode, lambda item: item.example_id)


def write_prose_examples(path: Path, records: Iterable[ProseNTPExample]) -> None:
    _write_records(path, records, lambda item: item.example_id, asdict)


def read_prose_examples(path: Path) -> tuple[ProseNTPExample, ...]:
    return _read_unique(path, lambda value: ProseNTPExample(**value), lambda item: item.example_id)


def write_packed_sequences(path: Path, records: Iterable[PackedSequence]) -> None:
    _write_records(path, records, lambda item: str(item.pack_id), asdict)


def read_packed_sequences(path: Path) -> tuple[PackedSequence, ...]:
    def decode(value: Mapping[str, Any]) -> PackedSequence:
        data = dict(value)
        for name in ("example_ids", "input_ids", "loss_mask"):
            field = data.get(name)
            if not isinstance(field, list):
                raise ValueError(f"{name} must be a JSON list")
            data[name] = tuple(field)
        if not all(isinstance(item, str) and item for item in data["example_ids"]):
            raise ValueError("example_ids must be non-empty strings")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in data["input_ids"]
        ):
            raise ValueError("input_ids must be non-negative integers")
        if not all(isinstance(item, bool) for item in data["loss_mask"]):
            raise ValueError("loss_mask must be booleans")
        return PackedSequence(**data)

    return _read_unique(path, decode, lambda item: str(item.pack_id))
