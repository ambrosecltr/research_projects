"""Fresh-optimizer supervised fine-tuning from a pinned pretrained checkpoint."""

from __future__ import annotations

import argparse
import pickle
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from tokenizers import Tokenizer

from poetry50m.config import (
    config_hash,
    coordinate_source_hash,
    file_hash,
    lineage_hash,
    load_mapping,
)
from poetry50m.data.sft_training import (
    SftArtifact,
    load_sft_training_artifact,
    sft_batch_stream,
)
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import TrainConfig, Trainer
from poetry50m.training.engine import seed_everything
from poetry50m.trajectory.manifest import RunManifest

JsonWriter = Callable[[Path, object], None]


def _model_config(path: Path) -> ModelConfig:
    return ModelConfig.from_mapping(load_mapping(path))


def _train_config(path: Path) -> TrainConfig:
    return TrainConfig.from_mapping(load_mapping(path))


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("base checkpoint must be a restricted checkpoint payload") from error
    if not isinstance(value, Mapping) or value.get("format_version") != 2:
        raise ValueError("unsupported or malformed base checkpoint")
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("base checkpoint lacks model weights")
    return cast(Mapping[str, Any], value)


def _base_lineage(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    model: ModelConfig,
    tokenizer_hash: str,
) -> tuple[Mapping[str, Any], RunManifest, dict[str, Any]]:
    checkpoint_hash = file_hash(checkpoint_path)
    manifest = RunManifest.load(manifest_path)
    receipt = load_mapping(receipt_path)
    if (
        receipt.get("checkpoint_sha256") != checkpoint_hash
        or receipt.get("run_id") != manifest.run_id
        or manifest.tokenizer_hash != tokenizer_hash
    ):
        raise ValueError("base checkpoint, manifest, and receipt do not describe one lineage")
    payload = _checkpoint_payload(checkpoint_path)
    expected_model_hash = config_hash(asdict(model))
    if (
        payload.get("model_config_hash") != expected_model_hash
        or manifest.model_config_hash != expected_model_hash
    ):
        raise ValueError("base checkpoint model configuration does not match SFT")
    return payload, manifest, receipt


def _stream_hash(stream: object) -> str:
    state_dict = getattr(stream, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("SFT stream is not checkpointable")
    value = state_dict().get("stream_hash")
    if not isinstance(value, str) or not value:
        raise ValueError("SFT stream did not expose an identity")
    return value


def _initialization_contract(
    *,
    artifact: SftArtifact,
    checkpoint_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    model: ModelConfig,
    train: TrainConfig,
    stream_hash: str,
    batch_size: int,
    data_seed: int,
) -> dict[str, object]:
    mixture_receipt_path = artifact.root / "receipt.json"
    return {
        "format_version": 1,
        "mode": "supervised_fine_tuning",
        "optimizer_initialization": "fresh",
        "base_checkpoint_sha256": file_hash(checkpoint_path),
        "base_manifest_sha256": file_hash(manifest_path),
        "base_receipt_sha256": file_hash(receipt_path),
        "mixture_receipt_sha256": file_hash(mixture_receipt_path),
        "data_sha256": artifact.data_sha256,
        "tokenizer_sha256": artifact.receipt["tokenizer_sha256"],
        "model_config_sha256": config_hash(model),
        "train_config_sha256": config_hash(train),
        "stream_sha256": stream_hash,
        "batch_size": batch_size,
        "data_seed": data_seed,
        "one_epoch_steps": (artifact.pack_count + batch_size - 1) // batch_size,
        "pack_count": artifact.pack_count,
        "formatted_tokens": artifact.receipt["actual_formatted_tokens"],
        "effective_input_tokens_per_epoch": artifact.effective_input_tokens,
        "supervised_tokens_per_epoch": artifact.receipt["supervised_tokens"],
    }


def sft_validate_command(args: argparse.Namespace, *, write_json: JsonWriter) -> int:
    tokenizer_path = Path(args.tokenizer)
    model = _model_config(Path(args.model_config))
    train = _train_config(Path(args.train_config))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError("model vocab_size does not match the SFT tokenizer")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("SFT tokenizer lacks <|pad|>")
    artifact = load_sft_training_artifact(
        Path(args.mixture),
        tokenizer_path=tokenizer_path,
    )
    data_seed = args.data_seed if args.data_seed is not None else train.seed
    stream = sft_batch_stream(
        artifact,
        batch_size=args.batch_size,
        pad_token_id=pad_id,
        seed=data_seed,
    )
    _base_lineage(
        checkpoint_path=Path(args.base_checkpoint),
        manifest_path=Path(args.base_manifest),
        receipt_path=Path(args.base_receipt),
        model=model,
        tokenizer_hash=file_hash(tokenizer_path),
    )
    contract = _initialization_contract(
        artifact=artifact,
        checkpoint_path=Path(args.base_checkpoint),
        manifest_path=Path(args.base_manifest),
        receipt_path=Path(args.base_receipt),
        model=model,
        train=train,
        stream_hash=_stream_hash(stream),
        batch_size=args.batch_size,
        data_seed=data_seed,
    )
    one_epoch_steps = cast(int, contract["one_epoch_steps"])
    if train.max_steps % one_epoch_steps != 0:
        raise ValueError("SFT max_steps must be a whole number of mixture passes")
    write_json(
        Path(args.output),
        {
            **contract,
            "max_steps": train.max_steps,
            "planned_epochs": train.max_steps // one_epoch_steps,
            "review_step": one_epoch_steps,
        },
    )
    return 0


def sft_train_command(args: argparse.Namespace, *, write_json: JsonWriter) -> int:
    started = time.perf_counter()
    mixture = Path(args.mixture)
    tokenizer_path = Path(args.tokenizer)
    model = _model_config(Path(args.model_config))
    train = _train_config(Path(args.train_config))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError("model vocab_size does not match the SFT tokenizer")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("SFT tokenizer lacks <|pad|>")
    artifact = load_sft_training_artifact(mixture, tokenizer_path=tokenizer_path)
    data_seed = args.data_seed if args.data_seed is not None else train.seed
    stream = sft_batch_stream(
        artifact,
        batch_size=args.batch_size,
        pad_token_id=pad_id,
        seed=data_seed,
    )
    stream_hash = _stream_hash(stream)
    run_directory = Path(args.run_dir)
    resume_path = Path(args.resume) if args.resume else None
    init_path = run_directory / "sft.init.json"

    base_checkpoint = Path(args.base_checkpoint)
    base_manifest_path = Path(args.base_manifest)
    base_receipt_path = Path(args.base_receipt)
    base_payload, base_manifest, _ = _base_lineage(
        checkpoint_path=base_checkpoint,
        manifest_path=base_manifest_path,
        receipt_path=base_receipt_path,
        model=model,
        tokenizer_hash=file_hash(tokenizer_path),
    )
    contract = _initialization_contract(
        artifact=artifact,
        checkpoint_path=base_checkpoint,
        manifest_path=base_manifest_path,
        receipt_path=base_receipt_path,
        model=model,
        train=train,
        stream_hash=stream_hash,
        batch_size=args.batch_size,
        data_seed=data_seed,
    )
    if resume_path is None:
        if run_directory.exists() and (
            not run_directory.is_dir() or any(run_directory.iterdir())
        ):
            raise ValueError("fresh SFT requires an absent or empty --run-dir")
    else:
        if not init_path.is_file() or load_mapping(init_path) != contract:
            raise ValueError("SFT resume does not match its initialization contract")
        try:
            resume_path.resolve(strict=True).relative_to(run_directory.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise ValueError("SFT resume checkpoint must be inside its run directory") from error

    run_id = lineage_hash(
        "sft-run",
        cast(str, contract["base_checkpoint_sha256"]),
        cast(str, contract["mixture_receipt_sha256"]),
        config_hash(model),
        config_hash(train),
        stream_hash,
    )
    run_metadata = {
        "run_id": run_id,
        "mode": "supervised_fine_tuning",
        "base_run_id": base_manifest.run_id,
        "base_checkpoint_sha256": cast(str, contract["base_checkpoint_sha256"]),
        "mixture_receipt_sha256": cast(str, contract["mixture_receipt_sha256"]),
        "data_seed": str(data_seed),
    }
    seed_everything(train.seed, train.deterministic)
    initialized_model = DecoderOnlyTransformer(model)
    if resume_path is None:
        initialized_model.load_state_dict(dict(base_payload["model"]), strict=True)
    trainer = Trainer(initialized_model, train, run_directory, run_metadata=run_metadata)
    if trainer.model_config_hash != base_manifest.model_config_hash:
        raise ValueError("live SFT model does not match the pretrained manifest")
    if resume_path is not None:
        trainer.load_checkpoint(resume_path, stream)
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
        write_json(init_path, contract)
        RunManifest(
            run_id=run_id,
            initialization_id=lineage_hash(
                "sft-initialization", cast(str, contract["base_checkpoint_sha256"])
            ),
            data_order_id=lineage_hash("sft-stream", stream_hash),
            architecture_signature=model.architecture,
            corpus_signature=lineage_hash(
                "sft-mixture", cast(str, contract["mixture_receipt_sha256"])
            ),
            tokenizer_hash=file_hash(tokenizer_path),
            code_signature=coordinate_source_hash(Path(__file__).resolve().parents[1]),
            model_config_hash=trainer.model_config_hash,
            training_config_hash=trainer.train_config_hash,
            endpoint_sealed=False,
        ).save(run_directory / "run.manifest.json")

    until_step = args.until_step if args.until_step is not None else train.max_steps
    before_step = trainer.state.global_step
    before_data_tokens = trainer.state.data_tokens_seen
    before_supervised_tokens = trainer.state.supervised_tokens_seen
    trainer.fit(stream, until_step=until_step)
    checkpoint = trainer.save_checkpoint(run_directory / "checkpoints" / "final.pt")
    write_json(
        run_directory / "train.receipt.json",
        {
            "format_version": 1,
            "mode": "supervised_fine_tuning",
            "run_id": run_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "global_step": trainer.state.global_step,
            "optimizer_steps_this_command": trainer.state.global_step - before_step,
            "data_tokens_processed": trainer.state.data_tokens_seen,
            "data_tokens_processed_this_command": (
                trainer.state.data_tokens_seen - before_data_tokens
            ),
            "supervised_tokens_processed": trainer.state.supervised_tokens_seen,
            "supervised_tokens_processed_this_command": (
                trainer.state.supervised_tokens_seen - before_supervised_tokens
            ),
            "one_epoch_steps": contract["one_epoch_steps"],
            "command_wall_seconds": time.perf_counter() - started,
            "device": trainer.device.type,
            "precision": trainer.config.precision,
        },
    )
    return 0
