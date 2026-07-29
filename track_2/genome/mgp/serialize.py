from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

from ..hashing import sha256_file, sha256_json
from ..io import atomic_write_json, load_json
from .schema import ModelGenomeProgram


def save_program(
    directory: str | Path,
    program: ModelGenomeProgram,
    payloads: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / "payload.safetensors"
    safe_payloads = {name: tensor.detach().cpu().contiguous() for name, tensor in payloads.items()}
    save_file(safe_payloads, str(payload_path))
    executable = {
        "program": program.to_dict(),
        "payload_sha256": sha256_file(payload_path),
        "payload_keys": sorted(safe_payloads),
    }
    executable["program_id"] = sha256_json(executable)
    atomic_write_json(root / "manifest.json", executable)
    return {
        "program_id": executable["program_id"],
        "manifest_bytes": (root / "manifest.json").stat().st_size,
        "payload_bytes": payload_path.stat().st_size,
        "total_bytes": (root / "manifest.json").stat().st_size + payload_path.stat().st_size,
    }


def load_program(directory: str | Path) -> tuple[ModelGenomeProgram, dict[str, torch.Tensor], dict[str, object]]:
    root = Path(directory)
    manifest = load_json(root / "manifest.json")
    payload_path = root / "payload.safetensors"
    if sha256_file(payload_path) != manifest["payload_sha256"]:
        raise ValueError("MGP payload integrity check failed")
    check = dict(manifest)
    program_id = check.pop("program_id")
    if sha256_json(check) != program_id:
        raise ValueError("MGP manifest integrity check failed")
    payloads = dict(load_file(str(payload_path), device="cpu"))
    if sorted(payloads) != sorted(manifest["payload_keys"]):
        raise ValueError("MGP payload-key mismatch")
    return ModelGenomeProgram.from_dict(manifest["program"]), payloads, manifest
