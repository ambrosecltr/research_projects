"""Append-only structured training telemetry."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


class JSONLTelemetry:
    """Synchronous JSONL sink whose flushed records are immediately inspectable."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: Mapping[str, object]) -> None:
        try:
            payload = json.dumps(dict(event), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("telemetry event is not finite JSON-serializable data") from error
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
