from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..hashing import sha256_file, sha256_json
from ..io import load_tensor_file, read_json, resolve_artifact_member
from ..tensor_inventory import inventory_from_dict
from ..types import TensorSpec


@dataclass(frozen=True)
class CanonicalModelLife:
    root: Path
    manifest: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    def load_base(self) -> dict[str, torch.Tensor]:
        path = resolve_artifact_member(self.root, "W0.safetensors", field="W0")
        if sha256_file(path) != self.manifest["W0"]["canonical_file_sha256"]:
            raise ValueError(f"W0 file hash mismatch for {self.run_id}")
        return load_tensor_file(path)

    def load_target(self) -> dict[str, torch.Tensor]:
        target = self.manifest.get("WT")
        if not isinstance(target, Mapping) or target.get("canonical_file") is None:
            raise ValueError(f"WT is hidden for {self.run_id}")
        path = resolve_artifact_member(
            self.root,
            target["canonical_file"],
            field="WT.canonical_file",
        )
        if sha256_file(path) != target["canonical_file_sha256"]:
            raise ValueError(f"WT file hash mismatch for {self.run_id}")
        return load_tensor_file(path)

    def load_evidence(self) -> dict[str, torch.Tensor]:
        evidence = self.manifest.get("compiler_evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError(f"compiler evidence declaration is invalid for {self.run_id}")
        path = resolve_artifact_member(
            self.root,
            evidence["tensor_file"],
            field="compiler_evidence.tensor_file",
        )
        if sha256_file(path) != evidence["tensor_file_sha256"]:
            raise ValueError(f"compiler evidence hash mismatch for {self.run_id}")
        tensors = dict(load_file(str(path), device="cpu"))
        expected = {
            "architecture_features",
            "initialization_fingerprint",
            "dataset_fingerprint",
            "tokenizer_fingerprint",
            "training_recipe_fingerprint",
        }
        if set(tensors) != expected:
            raise ValueError(
                f"compiler evidence keys differ for {self.run_id}: "
                f"missing={sorted(expected - set(tensors))}, "
                f"extra={sorted(set(tensors) - expected)}"
            )
        return {name: tensor.to(torch.float32) for name, tensor in tensors.items()}


@dataclass(frozen=True)
class CanonicalLifeCorpus:
    root: Path
    manifest: dict[str, Any]
    inventory: list[TensorSpec]
    tied_groups: list[list[str]]
    lives: tuple[CanonicalModelLife, ...]

    def for_split(self, split: str) -> tuple[CanonicalModelLife, ...]:
        return tuple(life for life in self.lives if life.split == split)

    def by_id(self, run_id: str) -> CanonicalModelLife:
        matches = [life for life in self.lives if life.run_id == run_id]
        if len(matches) != 1:
            raise KeyError(f"expected one canonical life named {run_id}, found {len(matches)}")
        return matches[0]


def _validated_life(path: Path) -> CanonicalModelLife:
    if path.is_symlink():
        raise ValueError(f"model-life directory must not be a symbolic link: {path}")
    path = path.resolve(strict=True)
    manifest_path = resolve_artifact_member(path, "life.json", field="life.manifest")
    raw = read_json(manifest_path)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("model-life manifest must be an object with string keys")
    if raw.get("format") != "GENOME_MODEL_LIFE" or raw.get("version") != "0.2.0":
        raise ValueError(f"unsupported model-life artifact: {path}")
    content = dict(raw)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError(f"model-life manifest hash mismatch: {path}")
    if path.name != raw.get("run_id"):
        raise ValueError(f"model-life directory and run ID differ: {path}")
    evidence = raw.get("compiler_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("contains_endpoint_data") is not False:
        raise ValueError(f"compiler evidence endpoint policy is invalid: {path}")
    return CanonicalModelLife(root=path, manifest=raw)


def load_canonical_life_corpus(path: str | Path) -> CanonicalLifeCorpus:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"canonical life corpus is not a directory: {root}")
    raw = read_json(resolve_artifact_member(root, "manifest.json", field="manifest"))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("life-corpus manifest must be an object with string keys")
    if raw.get("format") != "GENOME_MODEL_LIFE_CORPUS" or raw.get("version") != "0.1.0":
        raise ValueError("unsupported canonical life corpus")
    content = dict(raw)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("canonical life-corpus manifest hash mismatch")
    files = raw.get("files")
    hashes = raw.get("file_sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise TypeError("canonical life-corpus files are invalid")
    for filename in files.values():
        if not isinstance(filename, str):
            raise TypeError("canonical life-corpus filenames must be strings")
        file_path = resolve_artifact_member(root, filename, field=f"files.{filename}")
        if sha256_file(file_path) != hashes.get(filename):
            raise ValueError(f"canonical life-corpus file hash mismatch: {filename}")
    inventory_value = read_json(
        resolve_artifact_member(
            root,
            files["tensor_inventory"],
            field="files.tensor_inventory",
        )
    )
    inventory, tied_groups = inventory_from_dict(inventory_value)
    unresolved_lives_root = root / "lives"
    if unresolved_lives_root.is_symlink():
        raise ValueError("canonical life corpus lives directory must not be a symbolic link")
    lives_root = unresolved_lives_root.resolve(strict=True)
    if not lives_root.is_dir() or not lives_root.is_relative_to(root):
        raise ValueError("canonical life corpus lacks a valid lives directory")
    lives = tuple(_validated_life(item) for item in sorted(lives_root.iterdir()) if item.is_dir())
    if len(lives) != int(raw["life_count"]):
        raise ValueError("canonical life count differs from the manifest")
    if len({life.run_id for life in lives}) != len(lives):
        raise ValueError("canonical life corpus contains duplicate run IDs")
    return CanonicalLifeCorpus(
        root=root,
        manifest=raw,
        inventory=inventory,
        tied_groups=tied_groups,
        lives=lives,
    )


def require_targets(lives: Iterable[CanonicalModelLife]) -> None:
    for life in lives:
        target = life.manifest.get("WT")
        if not isinstance(target, Mapping) or target.get("canonical_file") is None:
            raise ValueError(f"target endpoint is unavailable for {life.run_id}")
