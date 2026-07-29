"""Archived failed PolyPythia V4 predictive compiler for historical reproduction only."""

from __future__ import annotations

import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ..hashing import sha256_directory, sha256_file, sha256_json
from ..io import (
    read_json,
    replace_directory_atomic,
    resolve_artifact_directory,
    resolve_artifact_member,
    temporary_directory,
    write_json,
)
from ..mgp.serializer import save_program
from ..polypythia.lives import CanonicalModelLife
from ..types import TensorSpec
from .block_decoder import BlockDecoderConfig
from .blockwise_compiler import BlockwiseGenomeCompiler
from .compiler import GenomeCodeLayout, GenomeCompiler
from .multilife_decoder import (
    BlockBatch,
    MultiLifeBlockSampler,
    decoder_tensor_scales,
    genome_program_from_codes,
    interpreter_artifact_info,
    load_shared_decoder,
    masked_block_mse,
)


@dataclass(frozen=True)
class PredictiveCompilerTrainingConfig:
    seed: int = 2718
    updates: int = 50_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    rate_weight: float = 1e-5
    hidden_dim: int = 1024
    depth: int = 4
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    log_every: int = 500
    development_batches: int = 8

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "updates",
            "batch_size",
            "hidden_dim",
            "log_every",
            "development_batches",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "rate_weight",
            "grad_clip_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            positive = name in {"learning_rate", "grad_clip_norm"}
            if not math.isfinite(float(value)) or (
                float(value) <= 0 if positive else float(value) < 0
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not self.device:
            raise ValueError("device cannot be empty")


@dataclass(frozen=True)
class EvidenceBatch:
    architecture: torch.Tensor
    dataset: torch.Tensor
    conditioning: torch.Tensor
    dimensions: dict[str, int]


def _stack_evidence(
    lives: Sequence[CanonicalModelLife],
    *,
    device: torch.device,
) -> EvidenceBatch:
    if not lives:
        raise ValueError("compiler evidence requires model lives")
    loaded = [life.load_evidence() for life in lives]
    keys = set(loaded[0])
    if any(set(item) != keys for item in loaded):
        raise ValueError("compiler evidence keys differ across model lives")
    for key in keys:
        shape = loaded[0][key].shape
        if any(item[key].shape != shape for item in loaded):
            raise ValueError(f"compiler evidence shape differs for {key}")
    architecture = torch.stack(
        [
            torch.cat(
                [
                    item["architecture_features"].flatten(),
                    item["initialization_fingerprint"].flatten(),
                ]
            )
            for item in loaded
        ]
    ).to(device)
    dataset = torch.stack([item["dataset_fingerprint"].flatten() for item in loaded]).to(device)
    conditioning = torch.stack(
        [
            torch.cat(
                [
                    item["tokenizer_fingerprint"].flatten(),
                    item["training_recipe_fingerprint"].flatten(),
                ]
            )
            for item in loaded
        ]
    ).to(device)
    return EvidenceBatch(
        architecture=architecture,
        dataset=dataset,
        conditioning=conditioning,
        dimensions={
            "architecture": architecture.shape[1],
            "dataset": dataset.shape[1],
            "conditioning": conditioning.shape[1],
        },
    )


def _predicted_decoder_inputs(
    codes: Mapping[str, torch.Tensor],
    batch: BlockBatch,
) -> dict[str, torch.Tensor]:
    return {
        "global_codes": codes["global_code"][batch.life_indices],
        "layer_codes": codes["layer_codes"][
            batch.life_indices,
            batch.layer_slots,
        ],
        "tensor_codes": codes["tensor_codes"][
            batch.life_indices,
            batch.tensor_indices,
        ],
        "role_ids": batch.role_ids,
        "features": batch.features,
    }


def _blockwise_decoder_inputs(
    model: BlockwiseGenomeCompiler,
    context: torch.Tensor,
    batch: BlockBatch,
) -> dict[str, torch.Tensor]:
    return model.decoder_inputs(
        context,
        life_indices=batch.life_indices,
        layer_slots=batch.layer_slots,
        tensor_indices=batch.tensor_indices,
        role_ids=batch.role_ids,
        features=batch.features,
    )


@torch.no_grad()
def _development_loss(
    *,
    model: GenomeCompiler | BlockwiseGenomeCompiler,
    evidence: EvidenceBatch,
    decoder: torch.nn.Module,
    sampler: MultiLifeBlockSampler,
    config: PredictiveCompilerTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    if isinstance(model, BlockwiseGenomeCompiler):
        context = model.encode_evidence(
            evidence.architecture,
            evidence.dataset,
            evidence.conditioning,
        )
        codes = None
    else:
        distribution = model(
            evidence.architecture,
            evidence.dataset,
            evidence.conditioning,
        )
        codes = distribution.mode()
    losses = []
    for _ in range(config.development_batches):
        batch = sampler.make_batch(
            batch_size=config.batch_size,
            generator=generator,
            device=device,
        )
        decoder_inputs = (
            _blockwise_decoder_inputs(model, context, batch)
            if isinstance(model, BlockwiseGenomeCompiler)
            else _predicted_decoder_inputs(codes, batch)
        )
        prediction = decoder(**decoder_inputs)
        losses.append(float(masked_block_mse(prediction, batch.targets, batch.valid_masks).item()))
    model.train()
    return sum(losses) / len(losses)


def train_predictive_compiler(
    training_lives: Sequence[CanonicalModelLife],
    development_life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    output_path: str | Path,
    config: PredictiveCompilerTrainingConfig,
) -> dict[str, Any]:
    if not training_lives or any(life.split != "training" for life in training_lives):
        raise ValueError("predictive compiler training accepts only training-split lives")
    if development_life.split != "development":
        raise ValueError("predictive compiler development life must use the development split")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    interpreter, decoder_manifest, layer_to_slot = load_shared_decoder(
        shared_decoder_path,
        device=device,
    )
    decoder = interpreter.decoder
    decoder.requires_grad_(False)
    decoder.eval()
    tensor_scales = decoder_tensor_scales(
        decoder_manifest,
        tensor_specs,
        tied_groups,
        interpreter.role_scales,
    )
    layout = GenomeCodeLayout(
        global_code_dim=interpreter.config.global_code_dim,
        n_layers=len(layer_to_slot),
        layer_code_dim=interpreter.config.layer_code_dim,
        n_tensors=len(tensor_specs),
        tensor_code_dim=interpreter.config.tensor_code_dim,
    )
    train_evidence = _stack_evidence(training_lives, device=device)
    development_evidence = _stack_evidence([development_life], device=device)
    if development_evidence.dimensions != train_evidence.dimensions:
        raise ValueError("development evidence dimensions differ from training evidence")
    train_sampler = MultiLifeBlockSampler(
        base_states=[life.load_base() for life in training_lives],
        target_states=[life.load_target() for life in training_lives],
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=interpreter.config,
        role_to_id=interpreter.role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=interpreter.role_scales,
        tensor_scales=tensor_scales,
    )
    development_sampler = MultiLifeBlockSampler(
        base_states=[development_life.load_base()],
        target_states=[development_life.load_target()],
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=interpreter.config,
        role_to_id=interpreter.role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=interpreter.role_scales,
        tensor_scales=tensor_scales,
    )
    if interpreter.config.block_code_dim:
        model: GenomeCompiler | BlockwiseGenomeCompiler = BlockwiseGenomeCompiler(
            architecture_dim=train_evidence.dimensions["architecture"],
            dataset_fingerprint_dim=train_evidence.dimensions["dataset"],
            conditioning_dim=train_evidence.dimensions["conditioning"],
            layer_count=len(layer_to_slot),
            tensor_count=len(tensor_specs),
            role_count=len(interpreter.role_to_id),
            decoder_config=interpreter.config,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        ).to(device)
    else:
        model = GenomeCompiler(
            architecture_dim=train_evidence.dimensions["architecture"],
            dataset_fingerprint_dim=train_evidence.dimensions["dataset"],
            trajectory_fingerprint_dim=train_evidence.dimensions["conditioning"],
            layout=layout,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    development_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    metrics = []
    model.train()
    for update in range(1, config.updates + 1):
        batch = train_sampler.make_batch(
            batch_size=config.batch_size,
            generator=train_generator,
            device=device,
        )
        if isinstance(model, BlockwiseGenomeCompiler):
            context = model.encode_evidence(
                train_evidence.architecture,
                train_evidence.dataset,
                train_evidence.conditioning,
            )
            decoder_inputs = _blockwise_decoder_inputs(model, context, batch)
            rate = model.rate_proxy(decoder_inputs)
        else:
            distribution = model(
                train_evidence.architecture,
                train_evidence.dataset,
                train_evidence.conditioning,
            )
            codes = distribution.mode()
            decoder_inputs = _predicted_decoder_inputs(codes, batch)
            rate = distribution.rate_proxy() / (len(training_lives) * layout.total_dim)
        prediction = decoder(**decoder_inputs)
        endpoint_loss = masked_block_mse(
            prediction,
            batch.targets,
            batch.valid_masks,
        )
        loss = endpoint_loss + config.rate_weight * rate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        if update == 1 or update % config.log_every == 0 or update == config.updates:
            development_loss = _development_loss(
                model=model,
                evidence=development_evidence,
                decoder=decoder,
                sampler=development_sampler,
                config=config,
                generator=development_generator,
                device=device,
            )
            metrics.append(
                {
                    "update": update,
                    "training_normalized_block_mse": float(endpoint_loss.detach().item()),
                    "development_normalized_block_mse": development_loss,
                    "rate_proxy": float(rate.detach().item()),
                    "total_loss": float(loss.detach().item()),
                }
            )

    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        weights_path = temp / "compiler.safetensors"
        save_file(
            {
                name: tensor.detach().contiguous().cpu()
                for name, tensor in model.state_dict().items()
            },
            str(weights_path),
        )
        write_json(temp / "training_metrics.json", metrics, canonical=True)
        decoder_root = Path(shared_decoder_path).expanduser().resolve(strict=True)
        manifest = {
            "format": "GENOME_PREDICTIVE_COMPILER",
            "version": "0.1.0",
            "training_method": "frozen_decoder_endpoint_loss",
            "compiler_kind": (
                "blockwise" if isinstance(model, BlockwiseGenomeCompiler) else "flat"
            ),
            "arbitrary_latent_code_labels_used": False,
            "early_training_prefix_used": False,
            "repair_or_polishing_used": False,
            "training_run_ids": [life.run_id for life in training_lives],
            "development_run_id": development_life.run_id,
            "training_config": asdict(config),
            "layout": layout.to_dict(),
            "model_config": {
                "layer_count": len(layer_to_slot),
                "tensor_count": len(tensor_specs),
                "role_count": len(interpreter.role_to_id),
                "decoder_config": interpreter.config.to_dict(),
            },
            "evidence_dimensions": train_evidence.dimensions,
            "weights_file": "compiler.safetensors",
            "weights_sha256": sha256_file(weights_path),
            "shared_decoder_manifest_sha256": sha256_file(decoder_root / "manifest.json"),
            "shared_decoder_interpreter_sha256": decoder_manifest["interpreter"]["manifest_sha256"],
            "metrics_file": "training_metrics.json",
            "metrics_sha256": sha256_file(temp / "training_metrics.json"),
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_predictive_compiler(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GenomeCompiler | BlockwiseGenomeCompiler, dict[str, Any]]:
    root = Path(path).expanduser().resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("predictive compiler manifest must be an object")
    if manifest.get("format") != "GENOME_PREDICTIVE_COMPILER" or manifest.get("version") != "0.1.0":
        raise ValueError("unsupported predictive compiler")
    content = dict(manifest)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("predictive compiler manifest hash mismatch")
    weights_path = resolve_artifact_member(
        root,
        manifest["weights_file"],
        field="weights_file",
    )
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise ValueError("predictive compiler weight hash mismatch")
    dimensions = manifest["evidence_dimensions"]
    config = PredictiveCompilerTrainingConfig(**manifest["training_config"])
    compiler_kind = manifest.get("compiler_kind", "flat")
    if compiler_kind == "blockwise":
        model_config = manifest["model_config"]
        model: GenomeCompiler | BlockwiseGenomeCompiler = BlockwiseGenomeCompiler(
            architecture_dim=int(dimensions["architecture"]),
            dataset_fingerprint_dim=int(dimensions["dataset"]),
            conditioning_dim=int(dimensions["conditioning"]),
            layer_count=int(model_config["layer_count"]),
            tensor_count=int(model_config["tensor_count"]),
            role_count=int(model_config["role_count"]),
            decoder_config=BlockDecoderConfig.from_dict(model_config["decoder_config"]),
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
    elif compiler_kind == "flat":
        layout = GenomeCodeLayout(**manifest["layout"])
        model = GenomeCompiler(
            architecture_dim=int(dimensions["architecture"]),
            dataset_fingerprint_dim=int(dimensions["dataset"]),
            trajectory_fingerprint_dim=int(dimensions["conditioning"]),
            layout=layout,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
    else:
        raise ValueError(f"unsupported predictive compiler kind: {compiler_kind!r}")
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    return model.to(device).eval(), manifest


def predict_hidden_genome(
    hidden_life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    compiler_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if hidden_life.split != "hidden":
        raise ValueError("one-shot prediction requires a hidden-split life")
    target = hidden_life.manifest.get("WT")
    if not isinstance(target, Mapping) or target.get("canonical_file") is not None:
        raise ValueError("hidden WT must remain unavailable while predicting the genome")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    device_object = torch.device(device)
    compiler, compiler_manifest = load_predictive_compiler(
        compiler_path,
        device=device_object,
    )
    interpreter, decoder_manifest, layer_to_slot = load_shared_decoder(
        shared_decoder_path,
        device=device_object,
    )
    decoder_root = Path(shared_decoder_path).expanduser().resolve(strict=True)
    if (
        sha256_file(decoder_root / "manifest.json")
        != compiler_manifest["shared_decoder_manifest_sha256"]
    ):
        raise ValueError("compiler and shared decoder artifacts do not match")
    evidence = _stack_evidence([hidden_life], device=device_object)
    if evidence.dimensions != compiler_manifest["evidence_dimensions"]:
        raise ValueError("hidden evidence dimensions differ from compiler training")
    tensor_scales = decoder_tensor_scales(
        decoder_manifest,
        tensor_specs,
        tied_groups,
        interpreter.role_scales,
    )
    with torch.no_grad():
        if isinstance(compiler, BlockwiseGenomeCompiler):
            context = compiler.encode_evidence(
                evidence.architecture,
                evidence.dataset,
                evidence.conditioning,
            )
            shared = compiler.shared_codes(context)
            base_state = hidden_life.load_base()
            hidden_sampler = MultiLifeBlockSampler(
                base_states=[base_state],
                target_states=[base_state],
                tensor_specs=tensor_specs,
                tied_groups=tied_groups,
                decoder_config=interpreter.config,
                role_to_id=interpreter.role_to_id,
                layer_to_slot=layer_to_slot,
                role_scales=interpreter.role_scales,
                tensor_scales=tensor_scales,
            )
            expected_blocks = int(decoder_manifest["block_layout"]["block_count"])
            if len(hidden_sampler.references) != expected_blocks:
                raise ValueError("hidden block layout differs from shared decoder training")
            block_batches = []
            compile_batch_size = int(compiler_manifest["training_config"]["batch_size"])
            for start in range(0, expected_blocks, compile_batch_size):
                stop = min(start + compile_batch_size, expected_blocks)
                batch = hidden_sampler.make_indexed_batch(
                    life_indices=torch.zeros(stop - start, dtype=torch.long),
                    reference_indices=torch.arange(start, stop, dtype=torch.long),
                    device=device_object,
                )
                inputs = _blockwise_decoder_inputs(compiler, context, batch)
                block_batches.append(inputs["block_codes"].to(torch.float32).cpu())
            predicted = {
                "global_code": shared["global_code"][0].to(torch.float32).cpu(),
                "layer_codes": shared["layer_codes"][0].to(torch.float32).cpu(),
                "tensor_codes": shared["tensor_codes"][0].to(torch.float32).cpu(),
                "block_codes": torch.cat(block_batches).to(
                    torch.float16
                    if interpreter.config.block_code_storage_dtype == "float16"
                    else torch.float32
                ),
            }
        else:
            distribution = compiler(
                evidence.architecture,
                evidence.dataset,
                evidence.conditioning,
            )
            predicted = {
                name: value[0].detach().to(torch.float32).cpu()
                for name, value in distribution.mode().items()
            }
    interpreter_path = resolve_artifact_directory(
        decoder_root,
        decoder_manifest["interpreter"]["path"],
        field="interpreter.path",
    )
    info = interpreter_artifact_info(interpreter_path)
    compiler_root = Path(compiler_path).expanduser().resolve(strict=True)
    program = genome_program_from_codes(
        codes=predicted,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=interpreter.config,
        role_to_id=interpreter.role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=interpreter.role_scales,
        tensor_scales=tensor_scales,
        interpreter_info=info,
        candidate_id=f"{hidden_life.run_id}-one-shot-predicted-genome",
        manifest_metadata={
            "research_level": "G2",
            "run_id": hidden_life.run_id,
            "base_state_sha256": hidden_life.manifest["W0"]["canonical_state_sha256"],
            "target_endpoint_seen_during_compile": False,
            "early_training_prefix_used": False,
            "repair_or_polishing_used": False,
        },
    )
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        code_path = temp / "predicted_genome_code.safetensors"
        save_file(predicted, str(code_path))
        program_path = temp / "predicted_genome.mgp"
        save_program(program, program_path)
        manifest = {
            "format": "GENOME_HIDDEN_PREDICTION",
            "version": "0.1.0",
            "research_level": "G2",
            "hidden_run_id": hidden_life.run_id,
            "one_shot": True,
            "target_endpoint_seen": False,
            "early_training_prefix_used": False,
            "repair_or_polishing_used": False,
            "compiler_manifest_sha256": sha256_file(compiler_root / "manifest.json"),
            "shared_decoder_manifest_sha256": sha256_file(decoder_root / "manifest.json"),
            "base_state_sha256": hidden_life.manifest["W0"]["canonical_state_sha256"],
            "source_plan_content_sha256": hidden_life.manifest["source_plan_content_sha256"],
            "compiler_evidence_sha256": hidden_life.manifest["compiler_evidence"][
                "tensor_file_sha256"
            ],
            "code_file": "predicted_genome_code.safetensors",
            "code_sha256": sha256_file(code_path),
            "mgp_path": "predicted_genome.mgp",
            "predicted_mgp_sha256": sha256_directory(program_path),
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        seal = {
            "format": "GENOME_HIDDEN_PREDICTION_SEAL",
            "version": "0.1.0",
            "hidden_run_id": hidden_life.run_id,
            "created_unix": time.time(),
            "prediction_manifest_sha256": sha256_file(temp / "manifest.json"),
            "predicted_mgp_sha256": manifest["predicted_mgp_sha256"],
            "compiler_manifest_sha256": manifest["compiler_manifest_sha256"],
            "shared_decoder_manifest_sha256": manifest["shared_decoder_manifest_sha256"],
            "base_state_sha256": manifest["base_state_sha256"],
            "source_plan_content_sha256": manifest["source_plan_content_sha256"],
            "target_endpoint_seen": False,
        }
        seal["content_sha256"] = sha256_json(seal)
        write_json(temp / "prediction_seal.json", seal, canonical=True)
        replace_directory_atomic(temp, destination)
        return {"prediction": manifest, "seal": seal}
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
