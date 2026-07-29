from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from ..adapters.gpt_neox import (
    assert_native_canonical_roundtrip,
    load_canonical_gpt_neox_state,
    load_huggingface_state,
    validate_gpt_neox_config,
)
from ..hashing import sha256_file, sha256_json, sha256_state_dict
from ..io import (
    read_json,
    replace_directory_atomic,
    save_tensor_file,
    temporary_directory,
    write_json,
)
from ..state import validate_compatible_states
from ..tensor_inventory import (
    build_tensor_inventory_from_state,
    inventory_to_dict,
)
from ..types import TensorSpec
from .evidence import EvidenceConfig, build_compiler_evidence
from .hub import LifeSourcePlan, RoundOneSourcePlan


def _validated_receipt(path: str | Path, plan: RoundOneSourcePlan) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("download receipt must be an object with string keys")
    if value.get("format") != "GENOME_POLYPYTHIA_DOWNLOAD_RECEIPT":
        raise ValueError("not a PolyPythia download receipt")
    if value.get("version") != "0.1.0":
        raise ValueError(f"unsupported download receipt version: {value.get('version')!r}")
    if value.get("source_plan_content_sha256") != plan.to_dict()["content_sha256"]:
        raise ValueError("download receipt belongs to a different source plan")
    declared = value.get("content_sha256")
    content = dict(value)
    content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("download receipt content hash mismatch")
    return value


def _receipt_records(
    receipt: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    raw_records = receipt.get("checkpoints")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise TypeError("download receipt checkpoints must be a sequence")
    records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise TypeError("download receipt checkpoint must be a mapping")
        key = (str(record["run_id"]), int(record["step"]))
        if key in records:
            raise ValueError(f"duplicate download receipt checkpoint: {key}")
        records[key] = record
    return records


def _resolve_download(
    download_root: Path,
    record: Mapping[str, Any],
) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise TypeError("download receipt path must be a non-empty string")
    resolved_root = download_root.resolve(strict=True)
    unresolved = download_root / relative
    if unresolved.is_symlink():
        raise ValueError(f"download receipt path must not be a symlink: {relative}")
    candidate = unresolved.resolve(strict=True)
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
        raise ValueError(f"download receipt path escapes the download root: {relative}")
    if candidate.stat().st_size != int(record["bytes"]):
        raise ValueError(f"downloaded checkpoint size changed: {relative}")
    if sha256_file(candidate) != record["sha256"]:
        raise ValueError(f"downloaded checkpoint hash changed: {relative}")
    return candidate


def _life_by_id(plan: RoundOneSourcePlan) -> dict[str, LifeSourcePlan]:
    result = {life.run_id: life for life in plan.lives}
    if len(result) != len(plan.lives):
        raise ValueError("source plan contains duplicate life IDs")
    return result


def _checkpoint_source(life: LifeSourcePlan, step: int) -> dict[str, Any]:
    checkpoint = next((item for item in life.checkpoints if item.step == step), None)
    if checkpoint is None:
        raise ValueError(f"{life.run_id} has no planned checkpoint at step {step}")
    return {
        "step": checkpoint.step,
        "branch": checkpoint.branch,
        "commit": checkpoint.commit,
        "weight": asdict(checkpoint.weight),
    }


def _inventory_signature(inventory: Sequence[TensorSpec]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_index": spec.canonical_index,
            "name": spec.name,
            "role": spec.role,
            "layer_index": spec.layer_index,
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "is_buffer": spec.is_buffer,
        }
        for spec in inventory
    ]


def _tokenizer_identity(plan: RoundOneSourcePlan) -> dict[str, Any]:
    return {
        "repository": plan.tokenizer.repository,
        "commit": plan.tokenizer.commit,
        "files": [asdict(file) for file in plan.tokenizer.files],
    }


def prepare_canonical_lives(
    plan: RoundOneSourcePlan,
    *,
    receipt_path: str | Path,
    download_root: str | Path,
    output_root: str | Path,
    evidence_config: EvidenceConfig | None = None,
) -> dict[str, Any]:
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(destination)
    download_root_path = Path(download_root).expanduser().resolve(strict=True)
    receipt = _validated_receipt(receipt_path, plan)
    records = _receipt_records(receipt)
    catalog = plan.catalog
    initial_step = int(catalog["initial_step"])
    final_step = int(catalog["final_step"])
    architecture_path = download_root_path / "tokenizer" / "config.json"
    architecture = validate_gpt_neox_config(architecture_path)
    if architecture.get("tie_word_embeddings") is not False:
        raise ValueError("PolyPythia 14M must use separate input and output embeddings")
    evidence_config = evidence_config or EvidenceConfig()
    lives_by_id = _life_by_id(plan)
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        common_inventory: list[TensorSpec] | None = None
        common_inventory_signature: list[dict[str, Any]] | None = None
        life_manifests = []
        for life in plan.lives:
            base_record = records.get((life.run_id, initial_step))
            if base_record is None:
                raise ValueError(f"download receipt lacks W0 for {life.run_id}")
            base_path = _resolve_download(download_root_path, base_record)
            native_base = load_huggingface_state(base_path)
            assert_native_canonical_roundtrip(native_base)
            base_state = load_canonical_gpt_neox_state(base_path)
            inventory, tied_groups = build_tensor_inventory_from_state(base_state)
            if tied_groups:
                raise ValueError(f"unexpected tied tensors in {life.run_id}: {tied_groups}")
            signature = _inventory_signature(inventory)
            if common_inventory is None:
                common_inventory = inventory
                common_inventory_signature = signature
            elif signature != common_inventory_signature:
                raise ValueError(f"canonical tensor inventory differs for {life.run_id}")

            target_record = records.get((life.run_id, final_step))
            target_state: dict[str, torch.Tensor] | None = None
            target_path: Path | None = None
            if target_record is not None:
                if life.split == "hidden" and not receipt.get("reveal_hidden"):
                    raise ValueError("sealed receipt must not contain the hidden endpoint")
                target_path = _resolve_download(download_root_path, target_record)
                native_target = load_huggingface_state(target_path)
                assert_native_canonical_roundtrip(native_target)
                target_state = load_canonical_gpt_neox_state(target_path)
                validate_compatible_states(base_state, target_state, inventory)

            life_root = temp / "lives" / life.run_id
            life_root.mkdir(parents=True)
            save_tensor_file(life_root / "W0.safetensors", base_state)
            if target_state is not None:
                save_tensor_file(life_root / "WT.safetensors", target_state)
            evidence_tensors, evidence_manifest = build_compiler_evidence(
                base_state=base_state,
                inventory=inventory,
                architecture=architecture,
                dataset_order=plan.dataset_order,
                data_order_seed=life.data_order_seed,
                tokenizer_identity=_tokenizer_identity(plan),
                training_recipe=dict(catalog["training_recipe"]),
                config=evidence_config,
            )
            save_file(evidence_tensors, str(life_root / "compiler_evidence.safetensors"))
            write_json(
                life_root / "compiler_evidence.json",
                evidence_manifest,
                canonical=True,
            )

            planned_life = lives_by_id[life.run_id]
            downloaded_steps = sorted(step for run_id, step in records if run_id == life.run_id)
            life_manifest = {
                "format": "GENOME_MODEL_LIFE",
                "version": "0.2.0",
                "run_id": life.run_id,
                "family": catalog["family"],
                "architecture": catalog["architecture"],
                "model_size": catalog["model_size"],
                "split": life.split,
                "seed": life.seed,
                "data_order_seed": life.data_order_seed,
                "repository": life.repository,
                "main_commit": life.main_commit,
                "checkpoint_policy": {
                    "planned_steps": [item.step for item in planned_life.checkpoints],
                    "downloaded_steps": downloaded_steps,
                    "endpoint_pair_materialized": {
                        initial_step,
                        final_step,
                    }.issubset(downloaded_steps),
                    "all_trajectory_checkpoints_materialized": len(downloaded_steps)
                    == len(planned_life.checkpoints),
                },
                "W0": {
                    "source": _checkpoint_source(life, initial_step),
                    "canonical_file": "W0.safetensors",
                    "canonical_file_sha256": sha256_file(life_root / "W0.safetensors"),
                    "canonical_state_sha256": sha256_state_dict(base_state),
                },
                "WT": (
                    {
                        "access": "hidden",
                        "canonical_file": None,
                    }
                    if target_state is None
                    else {
                        "access": "available",
                        "source": _checkpoint_source(life, final_step),
                        "canonical_file": "WT.safetensors",
                        "canonical_file_sha256": sha256_file(life_root / "WT.safetensors"),
                        "canonical_state_sha256": sha256_state_dict(target_state),
                    }
                ),
                "compiler_evidence": {
                    "tensor_file": "compiler_evidence.safetensors",
                    "tensor_file_sha256": sha256_file(life_root / "compiler_evidence.safetensors"),
                    "manifest_file": "compiler_evidence.json",
                    "manifest_file_sha256": sha256_file(life_root / "compiler_evidence.json"),
                    "contains_endpoint_data": False,
                },
                "inventory_sha256": sha256_json(signature),
                "architecture_sha256": sha256_json(architecture),
                "source_plan_content_sha256": plan.to_dict()["content_sha256"],
                "download_receipt_sha256": sha256_file(receipt_path),
            }
            life_manifest["content_sha256"] = sha256_json(life_manifest)
            write_json(life_root / "life.json", life_manifest, canonical=True)
            life_manifests.append(life_manifest)

        if common_inventory is None:
            raise ValueError("no PolyPythia lives were prepared")
        write_json(temp / "architecture.json", architecture, canonical=True)
        write_json(
            temp / "tensor_inventory.json",
            inventory_to_dict(common_inventory, ()),
            canonical=True,
        )
        write_json(
            temp / "training_recipe.json",
            dict(catalog["training_recipe"]),
            canonical=True,
        )
        corpus_manifest = {
            "corpus": catalog["corpus_id"],
            "variant": catalog["corpus_variant"],
            "order_repository": plan.dataset_order.repository,
            "order_repository_commit": plan.dataset_order.commit,
            "seed_files": {
                seed: [asdict(file) for file in files]
                for seed, files in plan.dataset_order.seed_files.items()
            },
        }
        write_json(temp / "corpus.json", corpus_manifest, canonical=True)
        manifest = {
            "format": "GENOME_MODEL_LIFE_CORPUS",
            "version": "0.1.0",
            "experiment_id": catalog["experiment_id"],
            "source_plan_content_sha256": plan.to_dict()["content_sha256"],
            "download_receipt_sha256": sha256_file(receipt_path),
            "revealed_hidden_endpoint": bool(receipt.get("reveal_hidden")),
            "life_count": len(life_manifests),
            "splits": {
                split: [life["run_id"] for life in life_manifests if life["split"] == split]
                for split in ("training", "development", "hidden")
            },
            "files": {
                "architecture": "architecture.json",
                "tensor_inventory": "tensor_inventory.json",
                "training_recipe": "training_recipe.json",
                "corpus": "corpus.json",
            },
            "file_sha256": {
                filename: sha256_file(temp / filename)
                for filename in (
                    "architecture.json",
                    "tensor_inventory.json",
                    "training_recipe.json",
                    "corpus.json",
                )
            },
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
