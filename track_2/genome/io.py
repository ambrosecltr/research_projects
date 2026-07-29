from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from .hashing import canonical_json_bytes


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def write_json(path: str | Path, value: Any, *, canonical: bool = False) -> None:
    if canonical:
        data = canonical_json_bytes(value)
    else:
        data = json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    atomic_write_bytes(path, data + (b"\n" if not data.endswith(b"\n") else b""))


def read_json(path: str | Path) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object)


def resolve_artifact_member(root: str | Path, value: object, *, field: str) -> Path:
    """Resolve one declared artifact file without permitting path escape or symlinks."""

    artifact_root = Path(root).expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty filename")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must name a direct artifact child, got {value!r}")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
        raise ValueError(f"{field} must name a direct artifact child, got {value!r}")
    candidate = artifact_root / value
    if candidate.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link: {value!r}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != artifact_root or not resolved.is_file():
        raise ValueError(f"{field} does not resolve to a regular artifact file: {value!r}")
    return resolved


def resolve_artifact_directory(
    root: str | Path,
    value: object,
    *,
    field: str,
) -> Path:
    """Resolve one declared direct-child directory without path escape or symlinks."""

    artifact_root = Path(root).expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty directory name")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must name a direct artifact child, got {value!r}")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
        raise ValueError(f"{field} must name a direct artifact child, got {value!r}")
    candidate = artifact_root / value
    if candidate.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link: {value!r}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != artifact_root or not resolved.is_dir():
        raise ValueError(f"{field} does not resolve to an artifact directory: {value!r}")
    return resolved


def resolve_artifact_relative_file(
    root: str | Path,
    value: object,
    *,
    field: str,
) -> Path:
    """Resolve a declared nested file while rejecting traversal and symlink components."""

    artifact_root = Path(root).expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")
    if not isinstance(value, str) or not value or "\\" in value:
        raise TypeError(f"{field} must be a non-empty POSIX relative file path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field} is not a safe artifact-relative path: {value!r}")
    candidate = artifact_root / relative
    current = candidate
    while current != artifact_root:
        if current.is_symlink():
            raise ValueError(f"{field} must not pass through a symbolic link: {value!r}")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(artifact_root) or not resolved.is_file():
        raise ValueError(f"{field} does not resolve to an artifact file: {value!r}")
    return resolved


def resolve_artifact_relative_directory(
    root: str | Path,
    value: object,
    *,
    field: str,
) -> Path:
    """Resolve a declared nested directory while rejecting traversal and symlinks."""

    artifact_root = Path(root).expanduser().resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")
    if not isinstance(value, str) or not value or "\\" in value:
        raise TypeError(f"{field} must be a non-empty POSIX relative directory path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field} is not a safe artifact-relative path: {value!r}")
    candidate = artifact_root / relative
    current = candidate
    while current != artifact_root:
        if current.is_symlink():
            raise ValueError(f"{field} must not pass through a symbolic link: {value!r}")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(artifact_root) or not resolved.is_dir():
        raise ValueError(f"{field} does not resolve to an artifact directory: {value!r}")
    return resolved


def write_yaml(path: str | Path, value: Any) -> None:
    data = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    atomic_write_bytes(path, data)


def read_yaml(path: str | Path) -> Any:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
    ) -> object:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=UniqueKeyLoader)


def save_tensor_file(path: str | Path, tensors: Mapping[str, torch.Tensor]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    clean: dict[str, torch.Tensor] = {}
    seen_storages: set[int] = set()
    for name, tensor in tensors.items():
        value = tensor.detach().contiguous().cpu()
        storage_pointer = value.untyped_storage().data_ptr()
        # safetensors deliberately rejects shared storage. Clone only repeated storages rather
        # than doubling the memory of an entire 50M-parameter state during every write.
        if storage_pointer in seen_storages:
            value = value.clone()
        seen_storages.add(storage_pointer)
        clean[name] = value
    save_file(clean, str(temporary))
    os.replace(temporary, destination)


def load_tensor_file(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    return dict(load_file(str(path), device=str(device)))


def replace_directory_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    ensure_dir(destination.parent)
    os.replace(source, destination)


def temporary_directory(parent: str | Path, prefix: str) -> Path:
    ensure_dir(parent)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=Path(parent)))


def directory_size(path: str | Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def copytree_verified(source: str | Path, destination: str | Path) -> None:
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    shutil.copytree(source, destination_path)
