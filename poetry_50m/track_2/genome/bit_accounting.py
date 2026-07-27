from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .io import directory_size, read_json, resolve_artifact_member


def _declared_sha256(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"MGP {field} must be a lowercase SHA-256 digest")
    return value


def _path_bytes(path: str | Path | None) -> int:
    if path is None:
        return 0
    resolved = Path(path).expanduser().resolve(strict=True)
    return directory_size(resolved) if resolved.is_dir() else resolved.stat().st_size


def account_mgp(
    mgp_path: str | Path,
    *,
    interpreter_path: str | Path | None = None,
    base_path: str | Path | None = None,
    amortization_count: int = 1,
) -> dict[str, Any]:
    if (
        isinstance(amortization_count, bool)
        or not isinstance(amortization_count, int)
        or amortization_count <= 0
    ):
        raise ValueError("amortization_count must be a positive integer")
    root = Path(mgp_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"MGP path is not a directory: {root}")
    manifest_path = resolve_artifact_member(root, "manifest.json", field="manifest_file")
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict) or any(
        not isinstance(key, str) for key in raw_manifest
    ):
        raise TypeError("MGP manifest must be an object with string keys")
    manifest: dict[str, Any] = raw_manifest

    payload_file = resolve_artifact_member(
        root, manifest.get("payload_file"), field="payload_file"
    )
    if sha256_file(payload_file) != _declared_sha256(manifest, "payload_sha256"):
        raise ValueError("MGP payload hash mismatch")
    patch_file: Path | None = None
    if manifest.get("patch_file") is not None:
        patch_file = resolve_artifact_member(
            root, manifest.get("patch_file"), field="patch_file"
        )
        if sha256_file(patch_file) != _declared_sha256(manifest, "patch_sha256"):
            raise ValueError("MGP patch hash mismatch")
    elif manifest.get("patch_sha256") is not None:
        raise ValueError("MGP patch_sha256 must be null when patch_file is null")

    manifest_bytes = manifest_path.stat().st_size
    payload_bytes = payload_file.stat().st_size
    patch_bytes = patch_file.stat().st_size if patch_file else 0
    mgp_bytes = manifest_bytes + payload_bytes + patch_bytes
    artifact_directory_bytes = directory_size(root)
    interpreter_bytes = _path_bytes(interpreter_path)
    base_bytes = _path_bytes(base_path)
    target_specific = mgp_bytes
    single_model = target_specific + interpreter_bytes + base_bytes
    amortized = (
        target_specific
        + interpreter_bytes / amortization_count
        + base_bytes / amortization_count
    )
    logical_payload_bytes = manifest.get("logical_payload_bytes", 0)
    if (
        isinstance(logical_payload_bytes, bool)
        or not isinstance(logical_payload_bytes, int)
        or logical_payload_bytes < 0
    ):
        raise ValueError("MGP logical_payload_bytes must be a non-negative integer")
    return {
        "manifest_bytes": manifest_bytes,
        "payload_file_bytes": payload_bytes,
        "patch_file_bytes": patch_bytes,
        "mgp_bytes": mgp_bytes,
        "artifact_bytes": mgp_bytes,
        "artifact_directory_bytes": artifact_directory_bytes,
        "logical_payload_bytes": logical_payload_bytes,
        "interpreter_bytes": interpreter_bytes,
        "base_bytes": base_bytes,
        "target_specific_bytes": target_specific,
        "single_model_total_bytes": single_model,
        "amortized_total_bytes": amortized,
        "amortization_count": amortization_count,
    }
