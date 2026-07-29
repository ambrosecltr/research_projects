"""Archived fixed-layout V1 compiler training for historical reproduction only."""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from ..hashing import sha256_file, sha256_json
from ..io import (
    read_json,
    replace_directory_atomic,
    resolve_artifact_member,
    temporary_directory,
    write_json,
)
from .compiler import GenomeCodeLayout, GenomeCompiler, compiler_loss


@dataclass(frozen=True)
class CompilerTrainingConfig:
    seed: int = 1701
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    rate_weight: float = 1e-5
    hidden_dim: int = 512
    depth: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("seed", "epochs", "batch_size", "hidden_dim"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        for name in ("learning_rate", "weight_decay", "rate_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            minimum_ok = float(value) > 0 if name == "learning_rate" else float(value) >= 0
            if not math.isfinite(float(value)) or not minimum_ok:
                qualifier = "positive" if name == "learning_rate" else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


class ModelLifeTensorDataset(torch.utils.data.Dataset):
    """A run-level dataset; one safetensors file is one independent model life.

    Required keys: architecture_features, dataset_fingerprint, trajectory_fingerprint,
    target_flat_codes. Do not create thousands of pseudo-independent examples from one run.
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("compiler dataset is empty")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = load_file(str(self.paths[index]), device="cpu")
        required = {
            "architecture_features",
            "dataset_fingerprint",
            "trajectory_fingerprint",
            "target_flat_codes",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"model-life tensor file lacks {sorted(missing)}: {self.paths[index]}")
        return {key: value[key].to(torch.float32) for key in required}


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key].flatten() for item in items]) for key in items[0]}


def train_compiler(
    train_paths: Sequence[str | Path],
    *,
    layout: GenomeCodeLayout,
    output_path: str | Path,
    config: CompilerTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or CompilerTrainingConfig()
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dataset = ModelLifeTensorDataset(train_paths)
    sample = dataset[0]
    model = GenomeCompiler(
        architecture_dim=sample["architecture_features"].numel(),
        dataset_fingerprint_dim=sample["dataset_fingerprint"].numel(),
        trajectory_fingerprint_dim=sample["trajectory_fingerprint"].numel(),
        layout=layout,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    metrics = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        total = 0.0
        count = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            distribution = model(
                batch["architecture_features"],
                batch["dataset_fingerprint"],
                batch["trajectory_fingerprint"],
            )
            loss, values = compiler_loss(
                distribution, batch["target_flat_codes"], rate_weight=config.rate_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += values["total"]
            count += 1
        if epoch == 1 or epoch % max(1, config.epochs // 20) == 0 or epoch == config.epochs:
            metrics.append({"epoch": float(epoch), "loss": total / max(count, 1)})

    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        weights = temp / "compiler.safetensors"
        save_file({name: value.detach().cpu() for name, value in model.state_dict().items()}, str(weights))
        manifest = {
            "format": "GENOME_COMPILER",
            "version": "0.1.0",
            "layout": layout.to_dict(),
            "architecture_dim": sample["architecture_features"].numel(),
            "dataset_fingerprint_dim": sample["dataset_fingerprint"].numel(),
            "trajectory_fingerprint_dim": sample["trajectory_fingerprint"].numel(),
            "training_config": asdict(config),
            "training_run_count": len(dataset),
            "weights_file": "compiler.safetensors",
            "weights_sha256": sha256_file(weights),
            "metrics": metrics,
        }
        manifest["manifest_content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"path": str(destination), "metrics": metrics, "manifest": manifest}


def _lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_compiler(path: str | Path, *, device: str | torch.device = "cpu") -> GenomeCompiler:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"compiler artifact path is not a directory: {root}")
    manifest_path = resolve_artifact_member(root, "manifest.json", field="manifest_file")
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict) or any(
        not isinstance(key, str) for key in raw_manifest
    ):
        raise TypeError("compiler manifest must be an object with string keys")
    manifest: dict[str, Any] = raw_manifest
    if manifest.get("format") != "GENOME_COMPILER":
        raise ValueError("not a GENOME compiler artifact")
    if manifest.get("version") != "0.1.0":
        raise ValueError(f"unsupported GENOME compiler version: {manifest.get('version')!r}")

    weights = resolve_artifact_member(
        root, manifest.get("weights_file"), field="weights_file"
    )
    if sha256_file(weights) != _lower_sha256(
        manifest.get("weights_sha256"), field="weights_sha256"
    ):
        raise ValueError("compiler weight hash mismatch")
    declared_content_hash = _lower_sha256(
        manifest.get("manifest_content_sha256"), field="manifest_content_sha256"
    )
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    if sha256_json(content) != declared_content_hash:
        raise ValueError("compiler manifest content hash mismatch")

    raw_layout = manifest.get("layout")
    if not isinstance(raw_layout, dict):
        raise TypeError("compiler layout must be an object")
    expected_layout_keys = set(GenomeCodeLayout.__dataclass_fields__)
    if set(raw_layout) != expected_layout_keys:
        raise ValueError("compiler layout fields do not match GenomeCodeLayout")
    layout = GenomeCodeLayout(**raw_layout)
    raw_config = manifest.get("training_config")
    if not isinstance(raw_config, dict):
        raise TypeError("compiler training_config must be an object")
    expected_config_keys = set(CompilerTrainingConfig.__dataclass_fields__)
    if set(raw_config) != expected_config_keys:
        raise ValueError("compiler training_config fields do not match CompilerTrainingConfig")
    config = CompilerTrainingConfig(**raw_config)
    architecture_dim = _positive_int(manifest.get("architecture_dim"), field="architecture_dim")
    dataset_dim = _positive_int(
        manifest.get("dataset_fingerprint_dim"), field="dataset_fingerprint_dim"
    )
    trajectory_dim = _positive_int(
        manifest.get("trajectory_fingerprint_dim"), field="trajectory_fingerprint_dim"
    )
    model = GenomeCompiler(
        architecture_dim=architecture_dim,
        dataset_fingerprint_dim=dataset_dim,
        trajectory_fingerprint_dim=trajectory_dim,
        layout=layout,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    )
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    return model.to(device).eval()
