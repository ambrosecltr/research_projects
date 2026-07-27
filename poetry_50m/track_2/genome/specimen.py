from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .adapters.base import Track1Adapter
from .hashing import sha256_file, sha256_json, sha256_state_dict
from .io import (
    ensure_dir,
    load_tensor_file,
    read_json,
    replace_directory_atomic,
    resolve_artifact_member,
    save_tensor_file,
    temporary_directory,
    write_json,
)
from .state import validate_compatible_states
from .tensor_inventory import (
    assert_tied_equal,
    build_tensor_inventory,
    canonicalize_state_dict,
    inventory_from_dict,
    inventory_to_dict,
)
from .types import TensorSpec


@dataclass(frozen=True)
class FrozenSpecimen:
    root: Path
    manifest: dict[str, Any]
    inventory: list[TensorSpec]
    tied_groups: list[list[str]]

    @property
    def specimen_id(self) -> str:
        return str(self.manifest["specimen_id"])

    @property
    def base_path(self) -> Path:
        return resolve_artifact_member(
            self.root, self.manifest["files"]["base_state"], field="files.base_state"
        )

    @property
    def target_path(self) -> Path:
        return resolve_artifact_member(
            self.root, self.manifest["files"]["target_state"], field="files.target_state"
        )

    def load_base(self, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        return load_tensor_file(self.base_path, device=device)

    def load_target(self, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        return load_tensor_file(self.target_path, device=device)


def _environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }


def freeze_specimen(
    adapter: Track1Adapter,
    *,
    output_dir: str | Path,
    specimen_id: str,
    final_checkpoint: str | Path,
    base_checkpoint: str | Path | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> FrozenSpecimen:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"specimen already exists: {destination}")
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        endpoint_validation = adapter.validate_endpoint_checkpoint(final_checkpoint)
        if base_checkpoint is not None:
            base_validation = adapter.validate_base_checkpoint(
                base_checkpoint, endpoint_checkpoint=final_checkpoint
            )
            base_model = adapter.build_model().cpu()
            adapter.load_checkpoint(base_model, base_checkpoint)
            base_state = canonicalize_state_dict(base_model.state_dict())
            del base_model
            base_mode = "checkpoint"
            base_reproducibility = None
        else:
            base_validation = {"valid_base": True, "mode": "reconstructed"}
            first_initial = canonicalize_state_dict(adapter.initial_state())
            second_initial = canonicalize_state_dict(adapter.initial_state())
            first_hash = sha256_state_dict(first_initial)
            second_hash = sha256_state_dict(second_initial)
            if first_hash != second_hash:
                raise ValueError(
                    "adapter.initial_state() is not reproducible; provide an exact W0 checkpoint"
                )
            del second_initial
            base_state = first_initial
            base_mode = "adapter_initial_state"
            base_reproducibility = {
                "verified": True,
                "first_sha256": first_hash,
                "second_sha256": second_hash,
            }

        target_model = adapter.build_model().cpu()
        adapter.load_checkpoint(target_model, final_checkpoint)
        target_state = canonicalize_state_dict(target_model.state_dict())
        inventory, tied_groups = build_tensor_inventory(target_model, target_state)
        validate_compatible_states(base_state, target_state, inventory)
        assert_tied_equal(target_state, tied_groups)
        assert_tied_equal(base_state, tied_groups)

        save_tensor_file(temp / "W0.safetensors", base_state)
        save_tensor_file(temp / "WT.safetensors", target_state)

        architecture = adapter.architecture_manifest(target_model)
        architecture.update(
            {
                "state_tensor_count": len(target_state),
                "state_numel": sum(tensor.numel() for tensor in target_state.values()),
                "state_logical_bytes": sum(
                    tensor.numel() * tensor.element_size() for tensor in target_state.values()
                ),
            }
        )
        files_and_values = {
            "architecture.json": architecture,
            "tensor_inventory.json": inventory_to_dict(inventory, tied_groups),
            "tokenizer.json": adapter.tokenizer_manifest(),
            "corpus.json": adapter.corpus_manifest(),
            "training_recipe.json": adapter.training_recipe(),
            "splits.json": adapter.split_manifest(),
            "environment.json": _environment_manifest(),
        }
        for filename, value in files_and_values.items():
            write_json(temp / filename, value)

        manifest = {
            "format": "GENOME_SPECIMEN",
            "version": "0.2.0",
            "specimen_id": specimen_id,
            "created_unix": time.time(),
            "adapter_id": adapter.adapter_id,
            "base_mode": base_mode,
            "base_reproducibility": base_reproducibility,
            "base_validation": base_validation,
            "endpoint_validation": endpoint_validation,
            "source": {
                "final_checkpoint": str(Path(final_checkpoint).resolve()),
                "base_checkpoint": (
                    None if base_checkpoint is None else str(Path(base_checkpoint).resolve())
                ),
                **dict(source_metadata or {}),
            },
            "files": {
                "base_state": "W0.safetensors",
                "target_state": "WT.safetensors",
                "architecture": "architecture.json",
                "tensor_inventory": "tensor_inventory.json",
                "tokenizer": "tokenizer.json",
                "corpus": "corpus.json",
                "training_recipe": "training_recipe.json",
                "splits": "splits.json",
                "environment": "environment.json",
                "hashes": "hashes.json",
            },
            "state_hashes": {
                "W0": sha256_state_dict(base_state),
                "WT": sha256_state_dict(target_state),
            },
            "contract_hashes": {
                "architecture": sha256_json(architecture),
                "tensor_inventory": sha256_json(files_and_values["tensor_inventory.json"]),
                "tokenizer": sha256_json(files_and_values["tokenizer.json"]),
                "corpus": sha256_json(files_and_values["corpus.json"]),
                "training_recipe": sha256_json(files_and_values["training_recipe.json"]),
                "splits": sha256_json(files_and_values["splits.json"]),
            },
        }
        write_json(temp / "manifest.json", manifest)

        hashes = {
            path.name: sha256_file(path)
            for path in sorted(temp.iterdir())
            if path.is_file() and path.name != "hashes.json"
        }
        write_json(temp / "hashes.json", hashes)
        replace_directory_atomic(temp, destination)
        return load_specimen(destination)
    except BaseException:
        if temp.exists():
            import shutil

            shutil.rmtree(temp, ignore_errors=True)
        raise


_SPECIMEN_VERSIONS = {"0.1.0", "0.2.0"}
_SPECIMEN_FILE_KEYS = {
    "base_state",
    "target_state",
    "architecture",
    "tensor_inventory",
    "tokenizer",
    "corpus",
    "training_recipe",
    "splits",
    "environment",
    "hashes",
}


def _specimen_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("files")
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("specimen files must be an object with string keys")
    if set(raw) != _SPECIMEN_FILE_KEYS:
        raise ValueError(f"specimen files must contain exactly {sorted(_SPECIMEN_FILE_KEYS)}")
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise TypeError("specimen file declarations must be non-empty strings")
    if len(set(raw.values())) != len(raw):
        raise ValueError("specimen file declarations must be unique")
    return dict(raw)


def _sha256_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def load_specimen(path: str | Path) -> FrozenSpecimen:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"specimen path is not a directory: {root}")
    manifest_path = resolve_artifact_member(root, "manifest.json", field="manifest_file")
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict) or any(
        not isinstance(key, str) for key in raw_manifest
    ):
        raise TypeError("specimen manifest must be an object with string keys")
    manifest: dict[str, Any] = raw_manifest
    if manifest.get("format") != "GENOME_SPECIMEN":
        raise ValueError("not a GENOME specimen")
    version = str(manifest.get("version", ""))
    if version not in _SPECIMEN_VERSIONS:
        raise ValueError(f"unsupported GENOME specimen version: {version!r}")
    files = _specimen_files(manifest)
    for field, filename in files.items():
        resolve_artifact_member(root, filename, field=f"files.{field}")
    inventory_path = resolve_artifact_member(
        root, files["tensor_inventory"], field="files.tensor_inventory"
    )
    inventory_value = read_json(inventory_path)
    inventory, tied_groups = inventory_from_dict(inventory_value)
    specimen = FrozenSpecimen(
        root=root,
        manifest=manifest,
        inventory=inventory,
        tied_groups=tied_groups,
    )
    verify_specimen_files(specimen)
    return specimen


def verify_specimen_files(specimen: FrozenSpecimen) -> dict[str, Any]:
    files = _specimen_files(specimen.manifest)
    hashes_path = resolve_artifact_member(specimen.root, files["hashes"], field="files.hashes")
    raw_hashes = read_json(hashes_path)
    if not isinstance(raw_hashes, dict) or any(not isinstance(key, str) for key in raw_hashes):
        raise TypeError("specimen hash manifest must be an object with string keys")
    hashes: dict[str, str] = {
        filename: _sha256_digest(expected, field=f"hashes[{filename!r}]")
        for filename, expected in raw_hashes.items()
    }
    expected_files = {filename for key, filename in files.items() if key != "hashes"}
    expected_files.add("manifest.json")
    if set(hashes) != expected_files:
        missing = sorted(expected_files - set(hashes))
        extra = sorted(set(hashes) - expected_files)
        raise ValueError(f"specimen hash coverage mismatch; missing={missing}, extra={extra}")

    failures = []
    for filename, expected in hashes.items():
        path = resolve_artifact_member(specimen.root, filename, field=f"hashes[{filename!r}]")
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"hash:{filename}")
    if failures:
        raise ValueError(f"specimen integrity failure: {failures}")

    base = specimen.load_base()
    target = specimen.load_target()
    validate_compatible_states(base, target, specimen.inventory)
    state_hashes = specimen.manifest.get("state_hashes")
    if not isinstance(state_hashes, dict):
        raise TypeError("specimen state_hashes must be an object")
    if sha256_state_dict(base) != _sha256_digest(state_hashes.get("W0"), field="state_hashes.W0"):
        raise ValueError("W0 state hash mismatch")
    if sha256_state_dict(target) != _sha256_digest(
        state_hashes.get("WT"), field="state_hashes.WT"
    ):
        raise ValueError("WT state hash mismatch")
    assert_tied_equal(base, specimen.tied_groups)
    assert_tied_equal(target, specimen.tied_groups)
    return {"valid": True, "file_count": len(hashes)}
