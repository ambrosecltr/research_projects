from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from ..data import causal_batches_from_jsonl
from ..evaluation import evaluate_program
from ..io import atomic_write_json, ensure_output_dir, load_json
from ..mgp.policy import audit_program
from ..mgp.schema import ModelGenomeProgram
from ..mgp.serialize import save_program
from ..state import direct_fp16_delta_bytes, load_state
from .data import CompilerCorpus, CompilerRecord, load_record
from .model import CompilerConfig, GenomeCompiler, compiler_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20260729
    device: str = "cuda"
    checkpoint_every: int = 25
    development_every: int = 25
    rate_weight: float = 1e-6
    functional_weight: float = 0.05
    functional_every: int = 4
    functional_batches: int = 1
    development_evaluation_batches: int = 128

    def __post_init__(self) -> None:
        if self.development_evaluation_batches < 128:
            raise ValueError("free-running development evaluation requires at least 128 batches")


@dataclass(frozen=True)
class TargetLabels:
    primitives: torch.Tensor
    ranks: torch.Tensor


def labels_from_program(program: ModelGenomeProgram) -> TargetLabels:
    primitive_ids: list[int] = []
    ranks: list[int] = []
    for tensor in program.tensors:
        if tensor.tied_to is not None:
            primitive_ids.append(0)
            ranks.append(0)
            continue
        low_rank = next((item for item in tensor.components if item.primitive == "LOW_RANK"), None)
        vector = next(
            (
                item
                for item in tensor.components
                if item.primitive in {"DIRECT_VECTOR", "QUANTIZED_VECTOR"}
            ),
            None,
        )
        if low_rank is not None:
            primitive_ids.append(1)
            ranks.append(int(low_rank.arguments.get("rank", 0)))
        elif vector is not None:
            primitive_ids.append(2)
            ranks.append(0)
        else:
            primitive_ids.append(0)
            ranks.append(0)
    return TargetLabels(
        primitives=torch.tensor(primitive_ids, dtype=torch.long),
        ranks=torch.tensor(ranks, dtype=torch.long),
    )


def _save_checkpoint(
    output: Path,
    compiler: GenomeCompiler,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    epoch: int,
    record_index: int,
    best_generated_progress: float,
    training_config: TrainingConfig,
    compiler_config: CompilerConfig,
    final: bool = False,
) -> Path:
    name = f"final-step-{step:08d}" if final else f"step-{step:08d}"
    checkpoint = output / "checkpoints" / name
    checkpoint.mkdir(parents=True, exist_ok=False)
    save_file(
        {name: tensor.detach().cpu() for name, tensor in compiler.state_dict().items()},
        str(checkpoint / "compiler.safetensors"),
    )
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(
        {
            "python_random": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint / "rng.pt",
    )
    atomic_write_json(
        checkpoint / "state.json",
        {
            "step": step,
            "epoch": epoch,
            "record_index": record_index,
            "best_generated_progress": best_generated_progress,
            "training_config": asdict(training_config),
            "compiler_config": asdict(compiler_config),
        },
    )
    return checkpoint


def _load_checkpoint(
    checkpoint: Path,
    compiler: GenomeCompiler,
    optimizer: torch.optim.Optimizer,
    *,
    training_config: TrainingConfig,
    compiler_config: CompilerConfig,
) -> tuple[int, int, int, float]:
    from safetensors.torch import load_file

    state = load_json(checkpoint / "state.json")
    if state["compiler_config"] != asdict(compiler_config):
        raise ValueError("resume compiler configuration differs from checkpoint")
    previous_training = dict(state["training_config"])
    current_training = asdict(training_config)
    previous_training.pop("epochs", None)
    current_training.pop("epochs", None)
    if previous_training != current_training:
        raise ValueError("resume training configuration differs from checkpoint")
    compiler.load_state_dict(
        load_file(str(checkpoint / "compiler.safetensors"), device="cpu"), strict=True
    )
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    )
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python_random"])
    torch.set_rng_state(rng["torch_rng"])
    if torch.cuda.is_available() and rng.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
    return (
        int(state["step"]),
        int(state["epoch"]),
        int(state.get("record_index", 0)),
        float(state["best_generated_progress"]),
    )


def _evaluate_records(
    compiler: GenomeCompiler,
    records: Sequence[CompilerRecord],
    *,
    config: CompilerConfig,
    rate_weight: float,
) -> float:
    compiler.eval()
    total = 0.0
    with torch.no_grad():
        for record in records:
            example, w0, target_deltas, program, _ = load_record(
                record,
                global_feature_dim=config.global_feature_dim,
                tensor_feature_dim=config.tensor_feature_dim,
            )
            labels = labels_from_program(program)
            loss, _ = compiler_loss(
                compiler,
                example,
                target_primitives=labels.primitives,
                target_ranks=labels.ranks.clamp_max(config.max_rank),
                target_deltas=target_deltas,
                w0_state=w0,
                rate_weight=rate_weight,
            )
            total += float(loss)
    return total / max(1, len(records))


def _free_running_development(
    compiler: GenomeCompiler,
    records: Sequence[CompilerRecord],
    *,
    config: CompilerConfig,
    output: Path,
    step: int,
    device: torch.device,
    evaluation_batches: int,
) -> dict[str, Any]:
    if evaluation_batches < 128:
        raise ValueError("free-running development evaluation requires at least 128 batches")
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("transformers is required for free-running development evaluation") from error
    step_root = output / "free-running-development" / f"step-{step:08d}"
    step_root.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    compiler.eval()
    for record in records:
        if record.model_config_path is None:
            raise ValueError(f"record {record.run_id} lacks model_config_path")
        example, w0, _, _, _ = load_record(
            record,
            global_feature_dim=config.global_feature_dim,
            tensor_feature_dim=config.tensor_feature_dim,
        )
        program, payloads = compiler.generate_program(
            example,
            direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
        )
        program_root = step_root / record.run_id
        accounting = save_program(program_root, program, payloads)
        audit = audit_program(
            program,
            payloads,
            direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0),
            artifact_directory=program_root,
        )
        if not (audit.accepted_structure and audit.serialized and audit.primary_budget_pass):
            raise ValueError(
                f"generated development program for {record.run_id} failed the byte audit"
            )
        model_config = GPTNeoXConfig.from_dict(load_json(record.model_config_path))
        comparison = evaluate_program(
            model_factory=lambda model_config=model_config: GPTNeoXForCausalLM(model_config),
            base_state=w0,
            program=program,
            payloads=payloads,
            batches=causal_batches_from_jsonl(record.evaluation_jsonl),
            endpoint_state=load_state(record.wt_path),
            device=device,
            max_batches=evaluation_batches,
        )
        report = {
            "run_id": record.run_id,
            "split": "development",
            "program_path": str(program_root),
            "accounting": accounting,
            "audit": asdict(audit),
            "evaluation_jsonl": record.evaluation_jsonl,
            "evaluation_batches": evaluation_batches,
            "comparison": comparison.to_dict(),
        }
        atomic_write_json(program_root / "free_running_evaluation.json", report)
        reports.append(report)
    progresses = [
        float(report["comparison"]["endpoint_progress"])
        for report in reports
        if report["comparison"]["endpoint_progress"] is not None
    ]
    if len(progresses) != len(records):
        raise ValueError("free-running development evaluation lacks endpoint progress")
    summary = {
        "format": "GENOME_FREE_RUNNING_DEVELOPMENT",
        "version": "1.0.0",
        "step": step,
        "selection_metric": "mean_endpoint_progress",
        "mean_endpoint_progress": sum(progresses) / len(progresses),
        "reports": reports,
    }
    atomic_write_json(step_root / "summary.json", summary)
    return summary


def _functional_context(record: CompilerRecord, w0: Mapping[str, torch.Tensor], limit: int):
    if record.model_config_path is None or record.probe_jsonl is None:
        raise ValueError(
            f"record {record.run_id} lacks model_config_path/probe_jsonl required for functional training"
        )
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("transformers is required for functional compiler training") from error
    config = GPTNeoXConfig.from_dict(load_json(record.model_config_path))
    model = GPTNeoXForCausalLM(config)
    batches = []
    for index, batch in enumerate(causal_batches_from_jsonl(record.probe_jsonl)):
        if index >= limit:
            break
        batches.append(batch)
    if not batches:
        raise ValueError(f"record {record.run_id} has no functional probe batches")
    return model, batches


def train_compiler(
    corpus: CompilerCorpus,
    *,
    output: str | Path,
    compiler_config: CompilerConfig = CompilerConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
    resume_from: str | Path | None = None,
    free_running_evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if resume_from is None:
        output_path = ensure_output_dir(output, force=overwrite)
    else:
        output_path = Path(output)
        if not output_path.is_dir():
            raise FileNotFoundError(output_path)
    device = torch.device(
        training_config.device
        if training_config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
    compiler = GenomeCompiler(compiler_config).to(device)
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    training = [item for item in corpus.records if item.split == "training"]
    development = [item for item in corpus.records if item.split == "development"]
    metrics_path = output_path / "metrics.jsonl"
    step = 0
    start_epoch = 0
    start_record_index = 0
    best_generated_progress = float("-inf")
    if resume_from is not None:
        step, start_epoch, start_record_index, best_generated_progress = _load_checkpoint(
            Path(resume_from),
            compiler,
            optimizer,
            training_config=training_config,
            compiler_config=compiler_config,
        )
        compiler.to(device)
        for value in optimizer.state.values():
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    value[key] = tensor.to(device)
    started = time.time()
    for epoch in range(start_epoch, training_config.epochs):
        epoch_records = list(training)
        random.Random(training_config.seed + epoch).shuffle(epoch_records)
        offset = start_record_index if epoch == start_epoch else 0
        for record_position, record in enumerate(epoch_records[offset:], start=offset):
            compiler.train()
            example, w0, target_deltas, program, _ = load_record(
                record,
                global_feature_dim=compiler_config.global_feature_dim,
                tensor_feature_dim=compiler_config.tensor_feature_dim,
            )
            labels = labels_from_program(program)
            use_functional = (
                training_config.functional_weight > 0
                and training_config.functional_every > 0
                and (step + 1) % training_config.functional_every == 0
            )
            functional_model = None
            functional_batches = ()
            if use_functional:
                functional_model, functional_batches = _functional_context(
                    record, w0, training_config.functional_batches
                )
            optimizer.zero_grad(set_to_none=True)
            loss, row = compiler_loss(
                compiler,
                example,
                target_primitives=labels.primitives,
                target_ranks=labels.ranks.clamp_max(compiler_config.max_rank),
                target_deltas=target_deltas,
                w0_state=w0,
                rate_weight=training_config.rate_weight,
                functional_model=functional_model,
                functional_batches=functional_batches,
                functional_weight=training_config.functional_weight if use_functional else 0.0,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite compiler loss for {record.run_id}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                compiler.parameters(), training_config.gradient_clip
            )
            optimizer.step()
            step += 1
            row.update(
                {
                    "step": step,
                    "epoch": epoch,
                    "run_id": record.run_id,
                    "split": "training",
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step % training_config.development_every == 0:
                teacher_forced_loss = _evaluate_records(
                    compiler,
                    development,
                    config=compiler_config,
                    rate_weight=training_config.rate_weight,
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "split": "development",
                                "teacher_forced_surrogate_loss": teacher_forced_loss,
                                "elapsed_seconds": time.time() - started,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                evaluator = free_running_evaluator or _free_running_development
                generated = evaluator(
                    compiler,
                    development,
                    config=compiler_config,
                    output=output_path,
                    step=step,
                    device=device,
                    evaluation_batches=training_config.development_evaluation_batches,
                )
                generated_progress = float(generated["mean_endpoint_progress"])
                if generated_progress > best_generated_progress:
                    best_generated_progress = generated_progress
                    save_file(
                        {
                            name: tensor.detach().cpu()
                            for name, tensor in compiler.state_dict().items()
                        },
                        str(output_path / "best-compiler.safetensors"),
                    )
                    atomic_write_json(
                        output_path / "best.json",
                        {
                            "step": step,
                            "selection_metric": "mean_endpoint_progress",
                            "mean_endpoint_progress": best_generated_progress,
                            "teacher_forced_surrogate_loss": teacher_forced_loss,
                        },
                    )
            if step % training_config.checkpoint_every == 0:
                next_position = record_position + 1
                next_epoch = epoch
                if next_position >= len(epoch_records):
                    next_epoch = epoch + 1
                    next_position = 0
                _save_checkpoint(
                    output_path,
                    compiler,
                    optimizer,
                    step=step,
                    epoch=next_epoch,
                    record_index=next_position,
                    best_generated_progress=best_generated_progress,
                    training_config=training_config,
                    compiler_config=compiler_config,
                )
        start_record_index = 0
    if not math.isfinite(best_generated_progress):
        teacher_forced_loss = _evaluate_records(
            compiler,
            development,
            config=compiler_config,
            rate_weight=training_config.rate_weight,
        )
        evaluator = free_running_evaluator or _free_running_development
        generated = evaluator(
            compiler,
            development,
            config=compiler_config,
            output=output_path,
            step=step,
            device=device,
            evaluation_batches=training_config.development_evaluation_batches,
        )
        best_generated_progress = float(generated["mean_endpoint_progress"])
        save_file(
            {name: tensor.detach().cpu() for name, tensor in compiler.state_dict().items()},
            str(output_path / "best-compiler.safetensors"),
        )
        atomic_write_json(
            output_path / "best.json",
            {
                "step": step,
                "selection_metric": "mean_endpoint_progress",
                "mean_endpoint_progress": best_generated_progress,
                "teacher_forced_surrogate_loss": teacher_forced_loss,
            },
        )
    _save_checkpoint(
        output_path,
        compiler,
        optimizer,
        step=step,
        epoch=training_config.epochs,
        record_index=0,
        best_generated_progress=best_generated_progress,
        training_config=training_config,
        compiler_config=compiler_config,
        final=True,
    )
    summary = {
        "format": "GENOME_COMPILER_TRAINING_SUMMARY",
        "version": "1.0.0",
        "steps": step,
        "best_generated_endpoint_progress": best_generated_progress,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "compiler_parameters": sum(item.numel() for item in compiler.parameters()),
        "compiler_config": asdict(compiler_config),
        "training_config": asdict(training_config),
    }
    atomic_write_json(output_path / "summary.json", summary)
    return summary
