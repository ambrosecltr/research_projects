"""Trajectory fitting, precommitted verification, and optional application."""

from __future__ import annotations

import argparse
import copy
import hashlib
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import torch
from tokenizers import Tokenizer
from torch import Tensor

from poetry50m.config import (
    RunPolicy,
    canonical_json,
    config_hash,
    file_hash,
    load_mapping,
    tree_hash,
)
from poetry50m.data import PreparedBatchStream
from poetry50m.data.artifacts import read_packed_sequences
from poetry50m.model import DecoderOnlyTransformer
from poetry50m.training import Trainer
from poetry50m.training.engine import capture_rng_state, restore_rng_state
from poetry50m.training.stream import Batch
from poetry50m.trajectory import (
    AnchorOutputs,
    TrajectoryConfig,
    VerificationBatch,
    linear_finite_difference,
    load_weight_snapshot,
    low_rank_temporal_forecast,
    verify_candidate,
)
from poetry50m.trajectory.ledgers import (
    AnalysisLedger,
    DecisionLedger,
    write_json_ledger,
)
from poetry50m.trajectory.manifest import (
    OperationScope,
    RunManifest,
    SuccessLevel,
    TrajectoryExperimentManifest,
)
from poetry50m.trajectory.preparation import state_dict_hash
from poetry50m.trajectory.types import WeightSnapshot
from poetry50m.trajectory.verification import decoder_loss_evaluator

JsonWriter = Callable[[Path, object], None]


class TrainerFactory(Protocol):
    def __call__(
        self,
        args: argparse.Namespace,
        *,
        resume: Path | None = None,
        read_only: bool = False,
    ) -> tuple[Trainer, PreparedBatchStream]: ...


@dataclass(frozen=True, slots=True)
class FixedAnchorSet:
    batches: tuple[VerificationBatch, ...]
    row_ids: tuple[str, ...]
    positions: tuple[tuple[int, ...], ...]
    commitment: Mapping[str, object]


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _working_memory(device: torch.device) -> tuple[int | None, int | None, str]:
    if device.type == "cuda":
        return (
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.memory_allocated(device)),
            "cuda_max_memory_allocated_during_forecast_and_verification",
        )
    if device.type == "mps":
        return (
            None,
            int(torch.mps.current_allocated_memory()),
            "mps_peak_unavailable_current_allocated_reported_separately",
        )
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return (
        int(maximum_rss * multiplier),
        None,
        "process_lifetime_maximum_resident_set_size",
    )


def _selected_positions(mask: Sequence[bool], count: int, *, row_id: str) -> tuple[int, ...]:
    supervised = tuple(index for index, selected in enumerate(mask[1:]) if selected)
    if len(supervised) < count:
        raise ValueError(
            f"held-out pack {row_id} has {len(supervised)} supervised positions; "
            f"the run policy requires {count}"
        )
    if count == 1:
        return (supervised[len(supervised) // 2],)
    return tuple(supervised[index * (len(supervised) - 1) // (count - 1)] for index in range(count))


def heldout_anchor_commitment(prepared: Path, policy: RunPolicy) -> dict[str, object]:
    """Resolve the exact validation rows and token positions promised by a target policy."""

    batch_count = policy.verification.fixed_heldout_batches
    positions_per_batch = policy.verification.anchor_positions_per_batch
    packs = read_packed_sequences(prepared / "validation.packed.jsonl")
    if len(packs) < batch_count:
        raise ValueError(
            f"prepared artifact has {len(packs)} validation packs; "
            f"the run policy requires exactly {batch_count}"
        )
    selected = packs[:batch_count]
    rows: list[dict[str, object]] = []
    for pack in selected:
        row_id = f"{pack.objective}:pack:{pack.pack_id}"
        selected_positions = _selected_positions(
            pack.loss_mask,
            positions_per_batch,
            row_id=row_id,
        )
        rows.append(
            {
                "row_id": row_id,
                "token_positions": list(selected_positions),
                "pack_content_sha256": _sha256(asdict(pack)),
            }
        )
    return {
        "split": "validation",
        "selection": "canonical_first_n_packs",
        "batch_count": batch_count,
        "positions_per_batch": positions_per_batch,
        "anchor_semantics": {
            "logits": "next-token distributions at selected supervised target positions",
            "representations": "final normalized residuals at the same selected positions",
        },
        "rows": rows,
        "selected_content_sha256": _sha256([asdict(pack) for pack in selected]),
    }


def _fixed_anchor_set(
    prepared: Path,
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    policy: RunPolicy,
) -> FixedAnchorSet:
    if tokenizer.token_to_id("<|pad|>") is None:
        raise ValueError("prepared tokenizer lacks <|pad|>")
    batch_count = policy.verification.fixed_heldout_batches
    positions_per_batch = policy.verification.anchor_positions_per_batch
    commitment = heldout_anchor_commitment(prepared, policy)
    packs = read_packed_sequences(prepared / "validation.packed.jsonl")
    selected = packs[:batch_count]
    batches: list[VerificationBatch] = []
    row_ids: list[str] = []
    positions: list[tuple[int, ...]] = []
    for pack in selected:
        row_id = f"{pack.objective}:pack:{pack.pack_id}"
        selected_positions = _selected_positions(
            pack.loss_mask,
            positions_per_batch,
            row_id=row_id,
        )
        batches.append(
            VerificationBatch(
                torch.tensor([pack.input_ids[:-1]], dtype=torch.long, device=device),
                torch.tensor([pack.input_ids[1:]], dtype=torch.long, device=device),
                torch.tensor(
                    [pack.loss_mask[1:]],
                    dtype=torch.bool,
                    device=device,
                ),
            )
        )
        row_ids.append(row_id)
        positions.append(selected_positions)
    return FixedAnchorSet(
        tuple(batches),
        tuple(row_ids),
        tuple(positions),
        commitment,
    )


def _batch_payload(batch: Batch) -> dict[str, object]:
    input_ids, targets = batch["input_ids"], batch["targets"]
    loss_mask = batch.get("loss_mask")
    example_ids = batch.get("example_ids")
    token_count = batch.get("data_token_count")
    if (
        not isinstance(input_ids, Tensor)
        or not isinstance(targets, Tensor)
        or not isinstance(loss_mask, Tensor)
        or isinstance(example_ids, (str, bytes))
        or not isinstance(example_ids, Sequence)
        or any(not isinstance(value, (str, int)) for value in example_ids)
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
    ):
        raise TypeError("prepared probe stream emitted a malformed batch")
    return {
        "input_ids": input_ids.tolist(),
        "targets": targets.tolist(),
        "loss_mask": loss_mask.to(dtype=torch.bool).tolist(),
        "example_ids": list(example_ids),
        "data_token_count": token_count,
    }


def _probe_commitment(
    stream: PreparedBatchStream,
    *,
    source_step: int,
    target_step: int,
    gradient_accumulation_steps: int,
    committed_batch_count: int,
    probe_steps: int,
) -> dict[str, object]:
    required_batches = probe_steps * gradient_accumulation_steps
    if committed_batch_count != required_batches:
        raise ValueError(
            "run policy fixed_probe_batches must exactly equal "
            "probe_steps * gradient_accumulation_steps"
        )
    clone = copy.deepcopy(stream)
    source_state = dict(stream.state_dict())
    clone.load_state_dict(source_state)
    skipped_count = (target_step - source_step) * gradient_accumulation_steps
    skipped = clone.skip_batches(skipped_count)
    target_state = dict(clone.state_dict())
    batches = [_batch_payload(next(clone)) for _ in range(committed_batch_count)]
    return {
        "split": "train",
        "selection": "live_stream_cursor_after_virtual_leap",
        "source_step": source_step,
        "target_step": target_step,
        "source_stream_state_sha256": _sha256(source_state),
        "target_stream_state_sha256": _sha256(target_state),
        "skipped_batch_count": skipped.batch_count,
        "skipped_data_token_count": skipped.data_token_count,
        "committed_batch_count": committed_batch_count,
        "probe_steps": probe_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batches": [
            {
                "index": index,
                "example_ids": batch["example_ids"],
                "content_sha256": _sha256(batch),
            }
            for index, batch in enumerate(batches)
        ],
        "selected_content_sha256": _sha256(batches),
    }


def _probe_stream(
    stream: PreparedBatchStream,
    *,
    source_step: int,
    target_step: int,
    gradient_accumulation_steps: int,
) -> PreparedBatchStream:
    clone = copy.deepcopy(stream)
    clone.load_state_dict(dict(stream.state_dict()))
    expected = (target_step - source_step) * gradient_accumulation_steps
    skipped = clone.skip_batches(expected)
    if skipped.batch_count != expected:
        raise RuntimeError("probe stream did not reproduce the committed virtual leap")
    return clone


def _lineage_matches(snapshot: WeightSnapshot, expected: WeightSnapshot) -> bool:
    fields = (
        "run_id",
        "initialization_id",
        "data_order_id",
        "architecture_signature",
        "corpus_signature",
        "model_config_hash",
        "tokenizer_hash",
        "code_signature",
        "training_config_hash",
    )
    return all(
        getattr(snapshot.metadata, field) == getattr(expected.metadata, field) for field in fields
    )


def analyze_command(
    args: argparse.Namespace,
    *,
    trainer_factory: TrainerFactory,
    write_json: JsonWriter,
) -> int:
    command_started = time.perf_counter()
    process_started = time.process_time()
    trainer_load_started = time.perf_counter()
    trainer_load_cpu_started = time.process_time()
    trainer, stream = trainer_factory(
        args,
        resume=Path(args.checkpoint),
        read_only=True,
    )
    trainer_load_seconds = time.perf_counter() - trainer_load_started
    trainer_load_cpu_seconds = time.process_time() - trainer_load_cpu_started
    if len(args.snapshots) < 2:
        raise ValueError("trajectory analysis requires at least two reference snapshots")
    checkpoint_io_started = time.perf_counter()
    checkpoint_io_cpu_started = time.process_time()
    snapshots = tuple(load_weight_snapshot(Path(path)) for path in args.snapshots)
    checkpoint_io_seconds = time.perf_counter() - checkpoint_io_started
    checkpoint_io_cpu_seconds = time.process_time() - checkpoint_io_cpu_started
    expected_metadata = trainer.trajectory_metadata
    if expected_metadata is None:
        raise RuntimeError("analysis requires a trajectory-aware trainer")
    latest = snapshots[-1]
    current_metadata = replace(
        expected_metadata,
        checkpoint_id=f"step-{trainer.state.global_step:08d}",
        step=trainer.state.global_step,
        wall_seconds=trainer.state.elapsed_seconds,
        tokens_seen=trainer.state.data_tokens_seen,
    )
    current_snapshot = WeightSnapshot(
        metadata=current_metadata,
        state_dict=trainer.model.state_dict(),
    )
    experiment: TrajectoryExperimentManifest | None = None
    if args.scope == "online":
        if (
            not _lineage_matches(latest, current_snapshot)
            or latest.metadata.step != trainer.state.global_step
            or state_dict_hash(latest.state_dict) != state_dict_hash(trainer.model.state_dict())
        ):
            raise ValueError("online analysis requires the exact live-run W_t snapshot")
    else:
        if args.reference_manifest is None or args.target_manifest is None:
            raise ValueError("Level 1/2 analysis requires reference and target run manifests")
        reference_manifest = RunManifest.load(Path(args.reference_manifest))
        target_manifest = RunManifest.load(Path(args.target_manifest))
        level = SuccessLevel.SAME_RUN if args.scope == "level1" else SuccessLevel.TRANSFER
        experiment = TrajectoryExperimentManifest(
            level,
            reference_manifest,
            target_manifest,
            OperationScope.RAW_WEIGHT_TRANSPORT,
        )
        snapshots = experiment.validate_fit_sources(snapshots)
        experiment.validate_target_snapshot(current_snapshot, latest)
        if latest.metadata.step != trainer.state.global_step:
            raise ValueError("reference and replay checkpoints must use the same W_t step")
        if args.scope == "level1" and state_dict_hash(latest.state_dict) != state_dict_hash(
            trainer.model.state_dict()
        ):
            raise ValueError("Level 1 replay W_t must exactly equal the reference W_t")
        if args.continued_baseline_snapshot is not None:
            raise ValueError("continued endpoints cannot influence honest Level 1/2 gates")

    run_policy_path = Path(args.run_policy)
    policy = RunPolicy.load(run_policy_path)
    policy_hash = file_hash(run_policy_path)
    trajectory_config_path = Path(args.trajectory_config)
    if file_hash(trajectory_config_path) != policy.trajectory_config_sha256:
        raise ValueError("trajectory config does not match the pre-training run policy commitment")
    if trainer.run_metadata.get("run_policy_hash") != policy_hash:
        raise ValueError("analysis run policy does not match the checkpoint-bound policy")
    trajectory = TrajectoryConfig.load(trajectory_config_path)
    if not trainer.state.global_step < args.target_step <= trainer.config.max_steps:
        raise ValueError("target_step must be later than W_t and within configured max_steps")
    if args.target_step + policy.verification.probe_steps > trainer.config.max_steps:
        raise ValueError("post-leap probe would exceed configured max_steps")

    tokenizer = Tokenizer.from_file(str(Path(args.prepared) / "tokenizer.json"))
    fixed = _fixed_anchor_set(
        Path(args.prepared),
        tokenizer,
        trainer.device,
        policy=policy,
    )
    target_policy_commitment = {
        "format_version": 1,
        "run_id": current_metadata.run_id,
        "run_policy_sha256": policy_hash,
        "trajectory_config_sha256": policy.trajectory_config_sha256,
        "verification_policy": asdict(policy.verification),
        "heldout_anchor_selection": dict(fixed.commitment),
        "heldout_anchor_selection_sha256": config_hash(dict(fixed.commitment)),
    }
    policy_commitment_path = Path(args.run_dir) / "run.policy.commitment.json"
    if (
        not policy_commitment_path.is_file()
        or load_mapping(policy_commitment_path) != target_policy_commitment
    ):
        raise ValueError(
            "analysis requires the exact target policy and held-out anchor commitment "
            "sealed before target training"
        )
    probe = _probe_commitment(
        stream,
        source_step=trainer.state.global_step,
        target_step=args.target_step,
        gradient_accumulation_steps=trainer.config.gradient_accumulation_steps,
        committed_batch_count=policy.verification.fixed_probe_batches,
        probe_steps=policy.verification.probe_steps,
    )
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("analysis output directory must be absent or empty")
    input_checkpoint_hash = file_hash(Path(args.checkpoint))
    commitment: dict[str, object] = {
        "format_version": 1,
        "run_id": current_metadata.run_id,
        "checkpoint_sha256": input_checkpoint_hash,
        "source_checkpoint_id": current_metadata.checkpoint_id,
        "scope": args.scope,
        "method": args.method,
        "target_step": args.target_step,
        "run_policy_sha256": policy_hash,
        "trajectory_config_sha256": policy.trajectory_config_sha256,
        "gates": asdict(trajectory.gates),
        "safety": asdict(trajectory.safety),
        "optimizer_policy": policy.verification.optimizer_policy,
        "heldout_anchors": dict(fixed.commitment),
        "post_leap_probe": probe,
    }
    commitment_hash = _sha256(commitment)
    commitment["commitment_sha256"] = commitment_hash
    write_json(output / "verification.commitment.json", commitment)

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    forecast_started = time.perf_counter()
    forecast_cpu_started = time.process_time()
    forecast = (
        linear_finite_difference(
            snapshots[-2],
            snapshots[-1],
            target_step=args.target_step,
            config=trajectory.linear,
        )
        if args.method == "linear"
        else low_rank_temporal_forecast(
            snapshots,
            target_step=args.target_step,
            config=trajectory.low_rank,
        )
    )
    forecast_wall_seconds = time.perf_counter() - forecast_started
    forecast_cpu_seconds = time.process_time() - forecast_cpu_started
    forecast_accelerator_seconds = 0.0 if trainer.device.type == "cuda" else None

    def anchors(module: torch.nn.Module) -> AnchorOutputs:
        if not isinstance(module, DecoderOnlyTransformer):
            raise TypeError("anchor evaluation requires the Track 1 decoder model")
        was_training = module.training
        module.eval()
        logits: dict[str, Tensor] = {}
        representations: dict[str, Tensor] = {}
        try:
            with torch.inference_mode():
                for batch, row_id, positions in zip(
                    fixed.batches,
                    fixed.row_ids,
                    fixed.positions,
                    strict=True,
                ):
                    selected_logits, selected_residuals = module.anchor_features(
                        batch.input_ids,
                        torch.tensor(
                            positions,
                            dtype=torch.long,
                            device=batch.input_ids.device,
                        ),
                    )
                    logits[f"logits:{row_id}"] = selected_logits.detach().cpu()
                    representations[f"residual:{row_id}"] = selected_residuals.detach().cpu()
        finally:
            module.train(was_training)
        return AnchorOutputs(logits=logits, representations=representations)

    continued: Mapping[str, Tensor] | None = None
    if args.continued_baseline_snapshot:
        continued_io_started = time.perf_counter()
        continued_io_cpu_started = time.process_time()
        continued_snapshot = load_weight_snapshot(Path(args.continued_baseline_snapshot))
        checkpoint_io_seconds += time.perf_counter() - continued_io_started
        checkpoint_io_cpu_seconds += time.process_time() - continued_io_cpu_started
        if (
            not _lineage_matches(continued_snapshot, current_snapshot)
            or continued_snapshot.metadata.step != args.target_step
        ):
            raise ValueError("continued baseline must share lineage and be saved at target_step")
        continued = continued_snapshot.state_dict

    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())
    scaler_state = copy.deepcopy(trainer.scaler.state_dict())
    training_state = replace(trainer.state)
    verification_rng = capture_rng_state()

    def post_leap_probe(module: torch.nn.Module) -> float:
        caller_rng = capture_rng_state()
        try:
            if not isinstance(module, DecoderOnlyTransformer):
                raise TypeError("post-leap probe requires the Track 1 decoder model")
            with tempfile.TemporaryDirectory(prefix="poetry50m-probe-") as directory:
                probe_trainer = Trainer(
                    copy.deepcopy(module),
                    trainer.config,
                    Path(directory),
                    run_metadata=trainer.run_metadata,
                    trajectory_metadata=trainer.trajectory_metadata,
                )
                probe_trainer.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                if policy.verification.optimizer_policy == "reset":
                    probe_trainer.optimizer.state.clear()
                probe_trainer.scheduler.load_state_dict(copy.deepcopy(scheduler_state))
                probe_trainer.scaler.load_state_dict(copy.deepcopy(scaler_state))
                probe_trainer.state = replace(training_state)
                probe_stream = _probe_stream(
                    stream,
                    source_step=training_state.global_step,
                    target_step=args.target_step,
                    gradient_accumulation_steps=trainer.config.gradient_accumulation_steps,
                )
                skipped_steps = args.target_step - training_state.global_step
                skipped_batches = skipped_steps * trainer.config.gradient_accumulation_steps
                skipped_state = copy.deepcopy(stream)
                skipped_state.load_state_dict(dict(stream.state_dict()))
                skipped_stats = skipped_state.skip_batches(skipped_batches)
                for _ in range(skipped_steps):
                    probe_trainer.scheduler.step()
                probe_trainer.state.global_step = args.target_step
                probe_trainer.state.virtual_steps_skipped += skipped_steps
                probe_trainer.state.micro_batches_skipped += skipped_stats.batch_count
                probe_trainer.state.data_tokens_skipped += skipped_stats.data_token_count
                restore_rng_state(verification_rng)
                probe_trainer.fit(
                    probe_stream,
                    until_step=args.target_step + policy.verification.probe_steps,
                )
                return decoder_loss_evaluator(probe_trainer.model, fixed.batches)
        finally:
            restore_rng_state(caller_rng)

    _synchronize(trainer.device)
    verification_cuda_start = (
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        if trainer.device.type == "cuda"
        else None
    )
    verification_cuda_end = (
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        if trainer.device.type == "cuda"
        else None
    )
    if verification_cuda_start is not None:
        verification_cuda_start.record()
    verification_started = time.perf_counter()
    verification_cpu_started = time.process_time()
    verification = verify_candidate(
        module=trainer.model,
        forecast=forecast,
        batches=fixed.batches,
        anchor_evaluator=anchors,
        gates=trajectory.gates,
        safety_config=trajectory.safety,
        post_leap_evaluator=post_leap_probe,
        continued_baseline_state=continued,
    )
    if verification_cuda_end is not None:
        verification_cuda_end.record()
    _synchronize(trainer.device)
    restore_rng_state(verification_rng)
    verification_wall_seconds = time.perf_counter() - verification_started
    verification_cpu_seconds = time.process_time() - verification_cpu_started
    verification_accelerator_seconds = (
        verification_cuda_start.elapsed_time(verification_cuda_end) / 1000.0
        if verification_cuda_start is not None and verification_cuda_end is not None
        else None
    )
    peak_memory, current_memory, peak_memory_semantics = _working_memory(trainer.device)

    snapshot_bytes_written = 0
    application_checkpoint_seconds = 0.0
    application_checkpoint_cpu_seconds = 0.0
    checkpoint: Path | None = None
    snapshot: Path | None = None
    if verification.decision.accepted and args.apply:
        trainer.apply_verified_transport(
            verification,
            stream,
            optimizer_state_policy=policy.verification.optimizer_policy,
        )
        checkpoint_write_started = time.perf_counter()
        checkpoint_write_cpu_started = time.process_time()
        checkpoint = trainer.save_checkpoint(output / "post_transport_checkpoint.pt")
        snapshot = trainer.save_trajectory_snapshot(output / "post_transport_snapshot.pt")
        application_checkpoint_seconds = time.perf_counter() - checkpoint_write_started
        application_checkpoint_cpu_seconds = time.process_time() - checkpoint_write_cpu_started
        snapshot_bytes_written = snapshot.stat().st_size
        write_json(
            output / "application.json",
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_hash(checkpoint),
                "snapshot": str(snapshot),
                "snapshot_sha256": file_hash(snapshot),
                "optimizer_policy": policy.verification.optimizer_policy,
                "continued_baseline": bool(continued),
                "checkpoint_write_seconds": application_checkpoint_seconds,
                "verification_commitment_sha256": commitment_hash,
            },
        )
    write_json_ledger(
        output / "decision.json",
        DecisionLedger(
            current_metadata.run_id,
            current_snapshot.metadata.checkpoint_id,
            verification.decision,
        ),
    )
    write_json_ledger(
        output / "analysis.json",
        AnalysisLedger(
            current_metadata.run_id,
            forecast,
            forecast_accelerator_seconds,
            checkpoint_io_seconds + application_checkpoint_seconds,
            verification.timing.baseline_seconds + verification.timing.candidate_seconds,
            peak_memory,
            sum(path.stat().st_size for path in map(Path, args.snapshots)),
            snapshot_bytes_written,
        ),
    )
    write_json(
        output / "verification.json",
        {
            "baseline_comparator": verification.baseline_comparator,
            "scope": args.scope,
            "verification_commitment_sha256": commitment_hash,
            "forecast_wall_seconds": forecast_wall_seconds,
            "forecast_accelerator_seconds": forecast_accelerator_seconds,
            "verification_wall_seconds": verification_wall_seconds,
            "verification_accelerator_seconds": verification_accelerator_seconds,
            "reference_fit_checkpoint_ids": [
                snapshot.metadata.checkpoint_id for snapshot in snapshots
            ],
            "target_checkpoint_id": current_snapshot.metadata.checkpoint_id,
            "timing": asdict(verification.timing),
            "decision": verification.decision.to_mapping(),
            "experiment_manifest": None
            if experiment is None
            else {
                "level": experiment.level.value,
                "operation_scope": experiment.operation_scope.value,
                "reference": asdict(experiment.reference),
                "target": asdict(experiment.target),
            },
        },
    )
    if experiment is not None:
        write_json(
            output / "experiment.manifest.json",
            {
                "level": experiment.level.value,
                "operation_scope": experiment.operation_scope.value,
                "reference": asdict(experiment.reference),
                "target": asdict(experiment.target),
                "fit_checkpoint_ids": [snapshot.metadata.checkpoint_id for snapshot in snapshots],
                "application_checkpoint_id": current_snapshot.metadata.checkpoint_id,
                "verification_commitment_sha256": commitment_hash,
            },
        )
    snapshot_bytes_read = sum(path.stat().st_size for path in map(Path, args.snapshots))
    analysis_source_hash = tree_hash(Path(__file__).resolve().parents[1])
    command_wall_seconds = time.perf_counter() - command_started
    process_cpu_seconds = time.process_time() - process_started
    checkpoint_io_component_wall_seconds = checkpoint_io_seconds + application_checkpoint_seconds
    checkpoint_io_component_cpu_seconds = (
        checkpoint_io_cpu_seconds + application_checkpoint_cpu_seconds
    )
    analysis_component_wall_seconds = (
        command_wall_seconds - checkpoint_io_component_wall_seconds - verification_wall_seconds
    )
    analysis_component_cpu_seconds = (
        process_cpu_seconds - checkpoint_io_component_cpu_seconds - verification_cpu_seconds
    )
    if analysis_component_wall_seconds < 0.0 or analysis_component_cpu_seconds < 0.0:
        raise RuntimeError("analysis cost component timings overlap")
    if trainer.device.type == "cuda":
        if verification_accelerator_seconds is None:
            raise RuntimeError("CUDA analysis timing events did not produce durations")
        accelerator_seconds = verification_accelerator_seconds
        checkpoint_io_accelerator_seconds: float | None = 0.0
    else:
        accelerator_seconds = None
        checkpoint_io_accelerator_seconds = None
    if trainer.device.type in {"cuda", "mps"}:
        forecast_device_active_wall_seconds: float | None = 0.0
        verification_device_active_wall_seconds: float | None = verification_wall_seconds
        checkpoint_io_device_active_wall_seconds: float | None = 0.0
        device_active_wall_seconds: float | None = verification_wall_seconds
    else:
        forecast_device_active_wall_seconds = None
        verification_device_active_wall_seconds = None
        checkpoint_io_device_active_wall_seconds = None
        device_active_wall_seconds = None
    write_json(
        output / "analysis.receipt.json",
        {
            "format_version": 1,
            "run_id": current_metadata.run_id,
            "checkpoint_id": current_snapshot.metadata.checkpoint_id,
            "checkpoint_sha256": input_checkpoint_hash,
            "steps": 0,
            "tokens": 0,
            "wall_seconds": command_wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "accelerator_seconds": accelerator_seconds,
            "device_active_wall_seconds": device_active_wall_seconds,
            "device": trainer.device.type,
            "actual_peak_working_memory_bytes": peak_memory,
            "current_working_memory_bytes": current_memory,
            "peak_memory_semantics": peak_memory_semantics,
            "trainer_and_checkpoint_load_wall_seconds": trainer_load_seconds,
            "trainer_and_checkpoint_load_cpu_seconds": trainer_load_cpu_seconds,
            "forecast_wall_seconds": forecast_wall_seconds,
            "forecast_cpu_seconds": forecast_cpu_seconds,
            "verification_wall_seconds": verification_wall_seconds,
            "verification_cpu_seconds": verification_cpu_seconds,
            "snapshot_io_wall_seconds": checkpoint_io_component_wall_seconds,
            "snapshot_io_cpu_seconds": checkpoint_io_component_cpu_seconds,
            "checkpoint_io_wall_seconds": checkpoint_io_component_wall_seconds,
            "snapshot_bytes_read": snapshot_bytes_read,
            "snapshot_bytes_written": snapshot_bytes_written,
            "cost_components": {
                "analysis": {
                    "steps": 0,
                    "tokens": 0,
                    "wall_seconds": analysis_component_wall_seconds,
                    "process_cpu_seconds": analysis_component_cpu_seconds,
                    "accelerator_seconds": forecast_accelerator_seconds,
                    "device_active_wall_seconds": (forecast_device_active_wall_seconds),
                    "timing_scope": ("full_command_remainder_after_checkpoint_io_and_verification"),
                },
                "checkpoint_io": {
                    "steps": 0,
                    "tokens": 0,
                    "wall_seconds": checkpoint_io_component_wall_seconds,
                    "process_cpu_seconds": checkpoint_io_component_cpu_seconds,
                    "accelerator_seconds": checkpoint_io_accelerator_seconds,
                    "device_active_wall_seconds": (checkpoint_io_device_active_wall_seconds),
                    "timing_scope": ("snapshot_loads_and_accepted_application_artifact_writes"),
                },
                "verification_per_replay": {
                    "steps": 0,
                    "tokens": 0,
                    "wall_seconds": verification_wall_seconds,
                    "process_cpu_seconds": verification_cpu_seconds,
                    "accelerator_seconds": verification_accelerator_seconds,
                    "device_active_wall_seconds": (verification_device_active_wall_seconds),
                    "timing_scope": ("synchronized_candidate_verification_and_post_leap_probe"),
                },
            },
            "verification_commitment_sha256": commitment_hash,
            "analysis_source_sha256": analysis_source_hash,
            "command_timing_scope": (
                "through_final_evidence_hashing_before_receipt_write; "
                "cost component wall and CPU times partition this interval"
            ),
            "device_active_timing_scope": (
                "synchronized_accelerator_exercising_verification_section_including_"
                "host_scheduling; CPU-only forecast and checkpoint IO are excluded"
            ),
        },
    )
    return 0
