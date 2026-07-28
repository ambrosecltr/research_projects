"""Small, durable persistence primitives for trajectory artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write and durably replace ``path`` without exposing a partial artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
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


def load_json_object(path: Path, *, name: str) -> dict[str, object]:
    """Load a JSON object and reject non-standard numeric constants."""

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains invalid JSON numeric constant {value!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value
