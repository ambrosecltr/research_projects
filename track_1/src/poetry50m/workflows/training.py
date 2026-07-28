"""Coordinate-affecting training workflow and immutable run lineage."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from poetry50m.config import (
    RunPolicy,
    config_hash,
    coordinate_source_hash,
    file_hash,
    lineage_hash,
    load_mapping,
)
from poetry50m.data import PreparedBatchStream, load_prepared_data
from poetry50m.data.artifacts import read_packed_sequences
from poetry50m.data.difficulty import DifficultyLedger
from poetry50m.data.schema import ObjectiveMix
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import TrainConfig, Trainer
from poetry50m.training.engine import seed_everything
from poetry50m.trajectory import SnapshotMetadata, load_weight_snapshot
from poetry50m.trajectory.manifest import RunManifest
from poetry50m.trajectory.preparation import state_dict_hash
from poetry50m.workflows.trajectory import heldout_anchor_commitment

JsonWriter = Callable[[Path, object], None]


def _model_config(path: Path) -> ModelConfig:
    return ModelConfig.from_mapping(load_mapping(path))


def _train_config(path: Path) -> TrainConfig:
    return TrainConfig.from_mapping(load_mapping(path))


def prepared_stream(
    prepared: Path,
    batch_size: int,
    seed: int,
    *,
    curriculum: str = "shuffled",
    difficulty_path: Path | None = None,
) -> PreparedBatchStream:
    tokenizer = Tokenizer.from_file(str(prepared / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("prepared tokenizer lacks <|pad|>")
    metadata = load_prepared_data(prepared).metadata
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("prepared metadata lacks its data configuration")
    objectives = config.get("objective_mix")
    if not isinstance(objectives, dict):
        raise ValueError("prepared metadata lacks objective_mix")
    packs_path = prepared / "train.packed.jsonl"
    difficulty: dict[str, float] | None = None
    if curriculum != "shuffled":
        if difficulty_path is None:
            raise ValueError("non-shuffled curriculum requires --difficulty from score")
        ledger = DifficultyLedger.load(difficulty_path)
        records = ledger.for_pass(0)
        expected = {
            f"{pack.objective}:pack:{pack.pack_id}" for pack in read_packed_sequences(packs_path)
        }
        if set(records) != expected:
            raise ValueError("difficulty ledger must cover exactly every prepared training pack")
        difficulty = {name: record.mean_loss for name, record in records.items()}
    return PreparedBatchStream.from_artifact(
        str(packs_path),
        batch_size=batch_size,
        pad_token_id=pad_id,
        objective_mix=ObjectiveMix(**objectives),
        seed=seed,
        curriculum=curriculum,
        difficulty=difficulty,
    )


def _snapshot_metadata(
    prepared: Path,
    model: ModelConfig,
    train: TrainConfig,
    run_id: str,
    *,
    initialization_state_hash: str,
    stream_hash: str,
) -> SnapshotMetadata:
    artifact = load_prepared_data(prepared)
    source_root = Path(__file__).resolve().parents[1]
    return SnapshotMetadata(
        run_id=run_id,
        checkpoint_id="initial",
        step=0,
        initialization_id=lineage_hash("initialization-state", initialization_state_hash),
        data_order_id=lineage_hash("prepared-stream", stream_hash),
        architecture_signature=model.architecture,
        corpus_signature=lineage_hash(
            "prepared-artifact",
            file_hash(prepared / "metadata.json"),
        ),
        model_config_hash=config_hash(model),
        tokenizer_hash=str(artifact.metadata["tokenizer_hash"]),
        code_signature=coordinate_source_hash(source_root),
        training_config_hash=config_hash(train),
    )


def _policy(args: argparse.Namespace) -> tuple[RunPolicy | None, str]:
    raw_path = getattr(args, "run_policy", None)
    if raw_path is None:
        return None, "none"
    path = Path(raw_path)
    return RunPolicy.load(path), file_hash(path)


def trainer(
    args: argparse.Namespace,
    *,
    resume: Path | None = None,
    read_only: bool = False,
) -> tuple[Trainer, PreparedBatchStream]:
    prepared = Path(args.prepared)
    model, train = _model_config(Path(args.model_config)), _train_config(Path(args.train_config))
    policy, policy_hash = _policy(args)
    tokenizer = Tokenizer.from_file(str(prepared / "tokenizer.json"))
    if tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError("model vocab_size must exactly match the prepared tokenizer vocabulary")
    data_seed = args.data_seed if args.data_seed is not None else train.seed
    stream = prepared_stream(
        prepared,
        args.batch_size,
        data_seed,
        curriculum=getattr(args, "curriculum", "shuffled"),
        difficulty_path=Path(args.difficulty) if getattr(args, "difficulty", None) else None,
    )
    stream_hash = stream.state_dict().get("stream_hash")
    if not isinstance(stream_hash, str) or not stream_hash:
        raise ValueError("prepared stream did not expose a validated stream identity")
    run_id = lineage_hash(
        "run",
        str(prepared.resolve()),
        file_hash(prepared / "metadata.json"),
        config_hash(model),
        config_hash(train),
        policy_hash,
        stream_hash,
        str(Path(args.run_dir).resolve()),
    )
    seed_everything(train.seed, train.deterministic)
    initial_model = DecoderOnlyTransformer(model)
    initial_state_hash = state_dict_hash(initial_model.state_dict())
    if read_only and not Path(args.run_dir).is_dir():
        raise ValueError("read-only checkpoint loading requires its existing run directory")
    result = Trainer(
        initial_model,
        train,
        Path(args.run_dir),
        run_metadata={
            "prepared_metadata_hash": file_hash(prepared / "metadata.json"),
            "run_id": run_id,
            "data_seed": str(data_seed),
            "run_policy_hash": policy_hash,
            "trajectory_config_sha256": (
                policy.trajectory_config_sha256 if policy is not None else "none"
            ),
        },
        trajectory_metadata=_snapshot_metadata(
            prepared,
            model,
            train,
            run_id,
            initialization_state_hash=initial_state_hash,
            stream_hash=stream_hash,
        ),
    )
    if resume is not None:
        result.load_checkpoint(resume, stream, record_resume_event=not read_only)
    return result, stream


def _run_manifest(
    metadata: SnapshotMetadata,
    *,
    endpoint_sealed: bool,
    fit_checkpoint_ids: tuple[str, ...] = (),
) -> RunManifest:
    return RunManifest(
        run_id=metadata.run_id,
        initialization_id=metadata.initialization_id,
        data_order_id=metadata.data_order_id,
        architecture_signature=metadata.architecture_signature,
        corpus_signature=metadata.corpus_signature,
        tokenizer_hash=metadata.tokenizer_hash,
        code_signature=metadata.code_signature,
        model_config_hash=metadata.model_config_hash,
        training_config_hash=metadata.training_config_hash,
        endpoint_sealed=endpoint_sealed,
        fit_checkpoint_ids=fit_checkpoint_ids,
    )


def _target_policy_commitment(
    *,
    run_id: str,
    prepared: Path,
    policy: RunPolicy,
    policy_hash: str,
) -> dict[str, object]:
    anchors = heldout_anchor_commitment(prepared, policy)
    return {
        "format_version": 1,
        "run_id": run_id,
        "run_policy_sha256": policy_hash,
        "trajectory_config_sha256": policy.trajectory_config_sha256,
        "verification_policy": asdict(policy.verification),
        "heldout_anchor_selection": anchors,
        "heldout_anchor_selection_sha256": config_hash(anchors),
    }


def _peak_memory(trainer: Trainer) -> tuple[int | None, int | None, str]:
    if trainer.device.type == "cuda":
        return (
            int(torch.cuda.max_memory_allocated(trainer.device)),
            int(torch.cuda.memory_allocated(trainer.device)),
            "cuda_max_memory_allocated_since_command_reset",
        )
    if trainer.device.type == "mps":
        current = int(torch.mps.current_allocated_memory())
        return None, current, "mps_peak_unavailable_current_allocated_reported_separately"
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return (
        int(maximum_rss * multiplier),
        None,
        "process_lifetime_maximum_resident_set_size",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_command(args: argparse.Namespace, *, write_json: JsonWriter) -> int:
    command_started = time.perf_counter()
    process_started = time.process_time()
    resume = Path(args.resume) if args.resume else None
    run_directory = Path(args.run_dir)
    run_manifest_path = run_directory / "run.manifest.json"
    policy_commitment_path = run_directory / "run.policy.commitment.json"
    existing_manifest: RunManifest | None = None
    if resume is None:
        if run_directory.exists() and (not run_directory.is_dir() or any(run_directory.iterdir())):
            raise ValueError("fresh training requires an absent or empty --run-dir")
    else:
        if not run_directory.is_dir() or not run_manifest_path.is_file():
            raise ValueError("resume requires its exact existing run directory and manifest")
        existing_manifest = RunManifest.load(run_manifest_path)
        if existing_manifest.endpoint_sealed is not args.seal_endpoint:
            raise ValueError("resume cannot change the run's endpoint-sealed classification")
        if existing_manifest.endpoint_sealed and not policy_commitment_path.is_file():
            raise ValueError("sealed resume requires its pre-existing target policy commitment")
        try:
            resume.resolve(strict=True).relative_to(run_directory.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                "resume checkpoint must exist inside its exact run directory"
            ) from error
    policy, policy_hash = _policy(args)
    if args.seal_endpoint and policy is None:
        raise ValueError("sealed target training requires --run-policy before training begins")
    if existing_manifest is not None and existing_manifest.endpoint_sealed:
        assert policy is not None
        expected_commitment = _target_policy_commitment(
            run_id=existing_manifest.run_id,
            prepared=Path(args.prepared),
            policy=policy,
            policy_hash=policy_hash,
        )
        if load_mapping(policy_commitment_path) != expected_commitment:
            raise ValueError("sealed target run policy commitment does not match")
    current, stream = trainer(args, resume=resume)
    if current.trajectory_metadata is None:
        raise RuntimeError("training workflow requires trajectory metadata")
    sealed_manifest = _run_manifest(current.trajectory_metadata, endpoint_sealed=True)
    if args.seal_endpoint:
        assert policy is not None
        target_commitment = _target_policy_commitment(
            run_id=current.trajectory_metadata.run_id,
            prepared=Path(args.prepared),
            policy=policy,
            policy_hash=policy_hash,
        )
        if run_manifest_path.exists():
            if RunManifest.load(run_manifest_path) != sealed_manifest:
                raise ValueError("sealed target run manifest does not match its commitment")
            if (
                not policy_commitment_path.is_file()
                or load_mapping(policy_commitment_path) != target_commitment
            ):
                raise ValueError("sealed target run policy commitment does not match")
        elif resume is not None:
            raise ValueError("sealed resume requires its pre-existing target commitments")
        else:
            sealed_manifest.save(run_manifest_path)
            write_json(policy_commitment_path, target_commitment)
    state_before = replace(current.state)
    artifact_write_seconds = 0.0
    if resume is None:
        initial_write_started = time.perf_counter()
        current.save_trajectory_snapshot(run_directory / "trajectory" / "initial.pt")
        artifact_write_seconds += time.perf_counter() - initial_write_started
    until = args.until_step if args.until_step is not None else current.config.max_steps
    if current.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(current.device)
        accelerator_start = torch.cuda.Event(  # type: ignore[no-untyped-call]
            enable_timing=True
        )
        accelerator_end = torch.cuda.Event(  # type: ignore[no-untyped-call]
            enable_timing=True
        )
    else:
        accelerator_start = None
        accelerator_end = None
    _synchronize(current.device)
    device_active_started = time.perf_counter() if current.device.type in {"cuda", "mps"} else None
    if accelerator_start is not None:
        accelerator_start.record()
    current.fit(stream, until_step=until)
    if accelerator_end is not None:
        accelerator_end.record()
    _synchronize(current.device)
    device_active_wall_seconds = (
        None if device_active_started is None else time.perf_counter() - device_active_started
    )
    accelerator_seconds = (
        accelerator_start.elapsed_time(accelerator_end) / 1000.0
        if accelerator_start is not None and accelerator_end is not None
        else None
    )
    final_write_started = time.perf_counter()
    checkpoint = current.save_checkpoint(run_directory / "checkpoints" / "final.pt")
    snapshot = current.save_trajectory_snapshot(run_directory / "trajectory" / "final.pt")
    artifact_write_seconds += time.perf_counter() - final_write_started
    final_snapshot = load_weight_snapshot(snapshot)
    metadata = final_snapshot.metadata
    fit_checkpoint_ids = tuple(
        sorted(
            {
                load_weight_snapshot(path).metadata.checkpoint_id
                for path in (run_directory / "trajectory").glob("*.pt")
            }
        )
    )
    run_manifest = (
        sealed_manifest
        if args.seal_endpoint
        else _run_manifest(
            metadata,
            endpoint_sealed=False,
            fit_checkpoint_ids=fit_checkpoint_ids,
        )
    )
    if not args.seal_endpoint:
        run_manifest.save(run_manifest_path)
    checkpoint_hash = file_hash(checkpoint)
    checkpoint_bytes = checkpoint.stat().st_size
    snapshot_hash = file_hash(snapshot)
    snapshot_bytes = snapshot.stat().st_size
    run_manifest_hash = file_hash(run_manifest_path)
    policy_commitment_hash = file_hash(policy_commitment_path) if args.seal_endpoint else None
    peak_memory, current_memory, peak_semantics = _peak_memory(current)
    command_wall_seconds = time.perf_counter() - command_started
    process_cpu_seconds = time.process_time() - process_started
    receipt = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_hash,
        "snapshot_bytes": snapshot_bytes,
        "global_step": current.state.global_step,
        "optimizer_steps_executed": current.state.optimizer_steps_executed,
        "virtual_steps_skipped": current.state.virtual_steps_skipped,
        "micro_batches_skipped": current.state.micro_batches_skipped,
        "data_tokens_processed": current.state.data_tokens_seen,
        "data_tokens_skipped": current.state.data_tokens_skipped,
        "supervised_tokens_processed": current.state.supervised_tokens_seen,
        "optimizer_steps_executed_this_command": current.state.optimizer_steps_executed
        - state_before.optimizer_steps_executed,
        "virtual_steps_skipped_this_command": current.state.virtual_steps_skipped
        - state_before.virtual_steps_skipped,
        "data_tokens_processed_this_command": current.state.data_tokens_seen
        - state_before.data_tokens_seen,
        "data_tokens_skipped_this_command": current.state.data_tokens_skipped
        - state_before.data_tokens_skipped,
        "supervised_tokens_processed_this_command": current.state.supervised_tokens_seen
        - state_before.supervised_tokens_seen,
        "run_id": metadata.run_id,
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": run_manifest_hash,
        "run_policy_sha256": None if policy is None else policy_hash,
        "run_policy_commitment": (str(policy_commitment_path) if args.seal_endpoint else None),
        "run_policy_commitment_sha256": policy_commitment_hash,
        "data_seed": args.data_seed if args.data_seed is not None else current.config.seed,
        "endpoint_sealed": args.seal_endpoint,
        "command_wall_seconds": command_wall_seconds,
        "trainer_step_wall_seconds_this_command": current.state.elapsed_seconds
        - state_before.elapsed_seconds,
        "artifact_write_wall_seconds": artifact_write_seconds,
        "process_cpu_seconds": process_cpu_seconds,
        "accelerator_seconds": accelerator_seconds,
        "device_active_wall_seconds": device_active_wall_seconds,
        "device": current.device.type,
        "precision": current.config.precision,
        "actual_peak_working_memory_bytes": peak_memory,
        "current_working_memory_bytes": current_memory,
        "peak_memory_semantics": peak_semantics,
        "checkpoint_io_wall_seconds": artifact_write_seconds,
        "snapshot_bytes_read": 0,
        "snapshot_bytes_written": snapshot_bytes,
        "command_timing_scope": "through_final_artifact_hashing_before_receipt_write",
    }
    write_json(run_directory / "train.receipt.json", receipt)
    return 0
