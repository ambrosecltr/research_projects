from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .hashing import sha256_json
from .io import ensure_dir, write_json, write_yaml


@dataclass(frozen=True)
class RunContext:
    run_id: str
    root: Path
    config_hash: str

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    def log_metric(self, value: Mapping[str, Any]) -> None:
        record = {"time_unix": time.time(), **dict(value)}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _git_state(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "status": command("status", "--porcelain"),
        "diff_sha256": sha256_json(command("diff") or ""),
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "device_count": torch.cuda.device_count(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA", "PYTORCH", "OMP", "MKL", "TOKENIZERS"))
        },
    }


def create_run(
    runs_root: str | Path,
    *,
    config: Mapping[str, Any],
    prefix: str,
    project_root: str | Path | None = None,
    parent_artifacts: Mapping[str, Any] | None = None,
) -> RunContext:
    config_hash = sha256_json(config)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{prefix}_{timestamp}_{config_hash[:10]}"
    root = Path(runs_root) / run_id
    if root.exists():
        raise FileExistsError(root)
    ensure_dir(root / "checkpoints")
    ensure_dir(root / "candidates")
    write_yaml(root / "resolved_config.yaml", dict(config))
    write_json(root / "environment.json", environment_snapshot())
    write_json(root / "git_state.json", _git_state(Path(project_root or ".").resolve()))
    write_json(root / "parent_artifacts.json", dict(parent_artifacts or {}))
    return RunContext(run_id=run_id, root=root, config_hash=config_hash)
