from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from ..io import (
    directory_size,
    load_tensor_file,
    read_json,
    replace_directory_atomic,
    resolve_artifact_member,
    save_tensor_file,
    temporary_directory,
    write_json,
)
from ..types import GenomeProgram, TensorGenomeRecord
from .validation import validate_program


def _tensor_payload_metadata(tensors: dict) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "logical_bytes": value.numel() * value.element_size(),
        }
        for key, value in sorted(tensors.items())
    }


def save_program(program: GenomeProgram, path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite MGP: {destination}")
    validate_program(program)
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        payload_name = "genome.safetensors"
        patch_name = "patch.safetensors" if program.patch_tensors else None
        save_tensor_file(temp / payload_name, program.payload_tensors)
        if patch_name:
            save_tensor_file(temp / patch_name, program.patch_tensors)

        manifest = dict(program.manifest)
        manifest.setdefault("format", "MGP")
        manifest.setdefault("version", "0.1.0")
        manifest["records"] = [record.to_dict() for record in program.records]
        manifest["tensor_order"] = [record.tensor_name for record in program.records]
        manifest["payload_file"] = payload_name
        manifest["patch_file"] = patch_name
        manifest["payload_tensors"] = _tensor_payload_metadata(program.payload_tensors)
        manifest["patch_tensors"] = _tensor_payload_metadata(program.patch_tensors)
        manifest["payload_sha256"] = sha256_file(temp / payload_name)
        manifest["patch_sha256"] = sha256_file(temp / patch_name) if patch_name else None
        manifest["logical_payload_bytes"] = sum(
            value.numel() * value.element_size()
            for value in [*program.payload_tensors.values(), *program.patch_tensors.values()]
        )
        write_json(temp / "manifest.json", manifest, canonical=True)

        sizes = {
            "manifest_bytes": (temp / "manifest.json").stat().st_size,
            "genome_payload_file_bytes": (temp / payload_name).stat().st_size,
            "patch_payload_file_bytes": (temp / patch_name).stat().st_size if patch_name else 0,
        }
        sizes["mgp_bytes"] = (
            sizes["manifest_bytes"]
            + sizes["genome_payload_file_bytes"]
            + sizes["patch_payload_file_bytes"]
        )
        # This report is auxiliary and is deliberately excluded from MGP byte claims.
        write_json(temp / "artifact_sizes.json", sizes)
        sizes["artifact_directory_bytes"] = directory_size(temp)

        replace_directory_atomic(temp, destination)
        return {
            **sizes,
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "payload_sha256": manifest["payload_sha256"],
            "patch_sha256": manifest["patch_sha256"],
        }
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _declared_sha256(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"MGP {field} must be a lowercase SHA-256 digest")
    return value


def load_program(path: str | Path) -> GenomeProgram:
    root = Path(path).expanduser().resolve(strict=True)
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
    payload = load_tensor_file(payload_file)

    patch: dict = {}
    patch_name = manifest.get("patch_file")
    patch_hash = manifest.get("patch_sha256")
    if patch_name is None:
        if patch_hash is not None:
            raise ValueError("MGP patch_sha256 must be null when patch_file is null")
    else:
        patch_file = resolve_artifact_member(root, patch_name, field="patch_file")
        if sha256_file(patch_file) != _declared_sha256(manifest, "patch_sha256"):
            raise ValueError("MGP patch hash mismatch")
        patch = load_tensor_file(patch_file)

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or any(not isinstance(value, dict) for value in raw_records):
        raise TypeError("MGP records must be an array of objects")
    records = [TensorGenomeRecord.from_dict(value) for value in raw_records]
    clean_manifest = dict(manifest)
    clean_manifest.pop("records", None)
    program = GenomeProgram(
        manifest=clean_manifest,
        records=records,
        payload_tensors=payload,
        patch_tensors=patch,
        source_path=root,
    )
    validate_program(program)
    return program
