from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def ensure_output_dir(path: str | Path, *, force: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination
