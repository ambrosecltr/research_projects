from __future__ import annotations

from pathlib import Path

from .io import atomic_write_json

DIRECTORIES = (
    "repo",
    "source/hf",
    "source/receipts",
    "canonical/lives",
    "evidence",
    "programs/candidates",
    "programs/accepted",
    "compiler/configs",
    "compiler/checkpoints",
    "runs",
    "logs",
    "cache/huggingface",
    "control",
)


def initialize_workspace(root: str | Path) -> dict[str, object]:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    created = []
    for relative in DIRECTORIES:
        directory = path / relative
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))
    manifest = {
        "format": "GENOME_WORKSPACE",
        "version": "1.0.0",
        "root": str(path.resolve()),
        "directories": list(DIRECTORIES),
        "rules": {
            "source_is_immutable_after_receipt": True,
            "hidden_wt_forbidden_before_prediction_seal": True,
            "fresh_isolated_volume": True,
        },
    }
    atomic_write_json(path / "control" / "workspace.json", manifest)
    return manifest
