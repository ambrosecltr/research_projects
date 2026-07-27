from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .hashing import sha256_file, sha256_json
from .io import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class ModelLifeRecord:
    run_id: str
    architecture_manifest: str
    base_state: str
    target_state: str
    dataset_fingerprint: str | None
    trajectory_features: str | None
    fitted_genome: str | None
    endpoint_hidden: bool
    seed: int | None = None
    data_order_id: str | None = None
    corpus_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelLifeIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "model_lives.jsonl"

    def append(self, record: ModelLifeRecord) -> None:
        ensure_dir(self.root)
        existing = {item.run_id for item in self}
        if record.run_id in existing:
            raise ValueError(f"model life already exists: {record.run_id}")
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    def __iter__(self) -> Iterator[ModelLifeRecord]:
        if not self.index_path.exists():
            return iter(())
        records = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(ModelLifeRecord(**json.loads(line)))
        return iter(records)

    def validate_paths(self) -> list[str]:
        failures = []
        for record in self:
            for field in (
                "architecture_manifest",
                "base_state",
                "target_state",
                "dataset_fingerprint",
                "trajectory_features",
                "fitted_genome",
            ):
                value = getattr(record, field)
                if value and not (self.root / value).exists():
                    failures.append(f"{record.run_id}:{field}:{value}")
        return failures

    def split(self, *, hidden_run_ids: set[str]) -> tuple[list[ModelLifeRecord], list[ModelLifeRecord]]:
        train = []
        hidden = []
        for record in self:
            if record.run_id in hidden_run_ids or record.endpoint_hidden:
                hidden.append(record)
            else:
                train.append(record)
        return train, hidden
