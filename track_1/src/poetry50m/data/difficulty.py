"""Persistent first-pass difficulty evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DifficultyRecord:
    example_id: str
    token_count: int
    negative_log_likelihood: float
    pass_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.example_id, str)
            or not self.example_id
            or isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 1
            or isinstance(self.negative_log_likelihood, bool)
            or not isinstance(self.negative_log_likelihood, (int, float))
            or not math.isfinite(self.negative_log_likelihood)
            or self.negative_log_likelihood < 0
            or isinstance(self.pass_index, bool)
            or not isinstance(self.pass_index, int)
            or self.pass_index < 0
        ):
            raise ValueError("invalid difficulty record")

    @property
    def mean_loss(self) -> float:
        return self.negative_log_likelihood / self.token_count


class DifficultyLedger:
    """Append-only first-pass observations, with one record per example/pass."""

    def __init__(self, records: Iterable[DifficultyRecord] = ()) -> None:
        self._records: dict[tuple[str, int], DifficultyRecord] = {}
        for record in records:
            self.record(record)

    def record(self, record: DifficultyRecord) -> None:
        key = (record.example_id, record.pass_index)
        if key in self._records:
            raise ValueError(f"difficulty already recorded for {key}")
        self._records[key] = record

    def for_pass(self, pass_index: int = 0) -> dict[str, DifficultyRecord]:
        return {
            example_id: record
            for (example_id, index), record in self._records.items()
            if index == pass_index
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(
            self._records.values(), key=lambda item: (item.pass_index, item.example_id)
        )
        payload = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> DifficultyLedger:
        with path.open(encoding="utf-8") as handle:
            records: list[DifficultyRecord] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank difficulty record at {path}:{line_number}")
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("difficulty record must be a JSON object")
                    records.append(DifficultyRecord(**value))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid difficulty record at {path}:{line_number}: {error}"
                    ) from error
            return cls(records)
