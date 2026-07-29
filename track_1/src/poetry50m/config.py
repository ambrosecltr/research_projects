"""Strict, auditable configuration and public artifact identities.

The CLI owns decoding configuration.  Core modules deliberately accept typed
objects so they cannot accidentally acquire YAML parsing or filesystem policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


def canonical_json(value: object) -> str:
    """Encode JSON-compatible values in the one representation used for IDs."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    """Hash an exact regular file without resolving a user-provided path silently."""
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    """Hash the explicit Python source contract, excluding metadata and caches."""
    if not root.is_dir():
        raise ValueError(f"expected a source directory: {root}")
    files = sorted(
        path for path in root.rglob("*.py") if path.is_file() and _is_public_source_path(root, path)
    )
    return _source_files_hash(root, files)


def coordinate_source_hash(root: Path) -> str:
    """Hash only code capable of changing initialized weights or optimizer coordinates."""

    if not root.is_dir():
        raise ValueError(f"expected a source directory: {root}")
    files = {
        path
        for path in (root / "model").rglob("*.py")
        if path.is_file() and _is_public_source_path(root, path)
    }
    files.update(
        path
        for relative in (
            "training/config.py",
            "training/engine.py",
            "training/stream.py",
            "data/batch_stream.py",
            "data/binary_stream.py",
            "data/sft_training.py",
            "data/general_sft.py",
            "workflows/sft.py",
            "workflows/training.py",
        )
        if (path := root / relative).is_file()
    )
    if not files:
        raise ValueError("coordinate source contract did not resolve any Python files")
    return _source_files_hash(root, sorted(files))


def _source_files_hash(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def _is_public_source_path(root: Path, path: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return "__pycache__" not in relative_parts and not any(
        part.startswith(".") for part in relative_parts
    )


def lineage_hash(*parts: str) -> str:
    """Domain-separate a stable lineage identity from its named components."""
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("lineage hash requires non-empty string components")
    return hashlib.sha256(b"poetry50m-lineage-v1\0" + "\0".join(parts).encode()).hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    """Load only JSON or safe YAML mappings, rejecting duplicate YAML keys."""
    if path.suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError("configuration must use .json, .yaml, or .yml")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
    else:

        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def construct_mapping(
            loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
        ) -> object:
            mapping: dict[object, object] = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("YAML configuration keys must be strings")
                if key in mapping:
                    raise ValueError(f"duplicate YAML configuration key: {key}")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
        )
        value = yaml.load(text, Loader=UniqueKeyLoader)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("configuration must be an object with string keys")
    return value


def config_hash(value: object) -> str:
    """Hash a typed config in its public JSON form."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


OptimizerStatePolicy = Literal["retain", "reset"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Pre-training commitment to the exact acceptance protocol."""

    fixed_heldout_batches: int
    anchor_positions_per_batch: int
    fixed_probe_batches: int
    probe_steps: int
    optimizer_policy: OptimizerStatePolicy

    def __post_init__(self) -> None:
        for name in (
            "fixed_heldout_batches",
            "anchor_positions_per_batch",
            "fixed_probe_batches",
            "probe_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"verification.{name} must be a positive integer")
        if self.optimizer_policy not in {"retain", "reset"}:
            raise ValueError("verification.optimizer_policy must be retain or reset")

    @classmethod
    def from_mapping(cls, value: object) -> VerificationPolicy:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError("run policy verification must be an object")
        expected = {
            "fixed_heldout_batches",
            "anchor_positions_per_batch",
            "fixed_probe_batches",
            "probe_steps",
            "optimizer_policy",
        }
        if set(value) != expected:
            raise ValueError(f"run policy verification must contain exactly {sorted(expected)}")
        optimizer_policy = value["optimizer_policy"]
        if optimizer_policy not in {"retain", "reset"}:
            raise ValueError("verification.optimizer_policy must be retain or reset")
        return cls(
            fixed_heldout_batches=value["fixed_heldout_batches"],
            anchor_positions_per_batch=value["anchor_positions_per_batch"],
            fixed_probe_batches=value["fixed_probe_batches"],
            probe_steps=value["probe_steps"],
            optimizer_policy=optimizer_policy,
        )


@dataclass(frozen=True, slots=True)
class RunPolicy:
    """Strict, hash-bound policy shared by training and trajectory acceptance."""

    format_version: int
    trajectory_config_sha256: str
    verification: VerificationPolicy

    def __post_init__(self) -> None:
        if isinstance(self.format_version, bool) or not isinstance(self.format_version, int):
            raise TypeError("run policy format_version must be an integer")
        if self.format_version != 1:
            raise ValueError("run policy format_version must be 1")
        if not isinstance(self.trajectory_config_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.trajectory_config_sha256
        ):
            raise ValueError("run policy trajectory_config_sha256 must be lowercase SHA-256")

    @classmethod
    def load(cls, path: Path) -> RunPolicy:
        value = load_mapping(path)
        expected = {"format_version", "trajectory_config_sha256", "verification"}
        if set(value) != expected:
            raise ValueError(f"run policy must contain exactly {sorted(expected)}")
        return cls(
            format_version=value["format_version"],
            trajectory_config_sha256=value["trajectory_config_sha256"],
            verification=VerificationPolicy.from_mapping(value["verification"]),
        )
