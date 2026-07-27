"""A small, exact trainer with checkpoints designed for trajectory research."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import pickle
import random
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from poetry50m.model import DecoderOnlyTransformer, ModelConfig, ModelOutput
from poetry50m.training.config import TrainConfig
from poetry50m.training.stream import Batch, SkippedBatchStats
from poetry50m.training.telemetry import JSONLTelemetry
from poetry50m.trajectory.preparation import state_dict_hash
from poetry50m.trajectory.snapshots import save_weight_snapshot
from poetry50m.trajectory.types import SnapshotMetadata, WeightSnapshot
from poetry50m.trajectory.verification import CandidateVerification


class CheckpointableBatchStream(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...

    def __next__(self) -> Batch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def skip_batches(self, count: int) -> SkippedBatchStats: ...


PerExampleLossHook = Callable[[int, Sequence[str] | Sequence[int] | None, Tensor], None]


@dataclass(slots=True)
class TrainingState:
    global_step: int = 0
    micro_step: int = 0
    optimizer_steps_executed: int = 0
    virtual_steps_skipped: int = 0
    micro_batches_skipped: int = 0
    data_tokens_seen: int = 0
    data_tokens_skipped: int = 0
    supervised_tokens_seen: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "global_step",
            "micro_step",
            "optimizer_steps_executed",
            "virtual_steps_skipped",
            "micro_batches_skipped",
            "data_tokens_seen",
            "data_tokens_skipped",
            "supervised_tokens_seen",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.elapsed_seconds, float) or not math.isfinite(self.elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite")
        if self.global_step != self.optimizer_steps_executed + self.virtual_steps_skipped:
            raise ValueError("global_step must equal executed plus virtual optimizer steps")


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    device: torch.device
    autocast_dtype: torch.dtype | None
    use_scaler: bool


def select_device(preference: str) -> torch.device:
    """Choose an explicitly requested accelerator or the best available local device."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if preference == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if available[candidate]:
                return torch.device(candidate)
    if preference not in available:
        raise ValueError(f"unknown device preference {preference!r}")
    if not available[preference]:
        raise RuntimeError(f"requested device {preference!r} is unavailable")
    return torch.device(preference)


def select_precision(device: torch.device, preference: str) -> PrecisionPolicy:
    """Pick a conservative autocast policy for the selected backend."""
    if preference not in {"auto", "none", "float16", "bfloat16"}:
        raise ValueError(f"unknown precision preference {preference!r}")
    if preference == "none":
        return PrecisionPolicy(device, None, False)
    if preference == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            return PrecisionPolicy(device, None, False)
        elif device.type == "cpu":
            dtype = torch.bfloat16
        else:
            raise RuntimeError(f"unsupported device type {device.type!r}")
    else:
        dtype = torch.float16 if preference == "float16" else torch.bfloat16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU float16 autocast is unsupported; use bfloat16 or none")
    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("MPS bfloat16 autocast is unsupported; use float16 or none")
    return PrecisionPolicy(device, dtype, device.type == "cuda" and dtype == torch.float16)


def seed_everything(seed: int, deterministic: bool) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def mapping_hash(values: Mapping[str, Any]) -> str:
    """Return the canonical configuration identity used by checkpoints and snapshots."""
    encoded = json.dumps(dict(values), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    python_version, python_state, python_gauss = random.getstate()
    numpy_generator, numpy_state, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    if not isinstance(numpy_state, np.ndarray):
        raise RuntimeError("legacy NumPy RNG state must use an array representation")
    state: dict[str, Any] = {
        "python": {
            "version": python_version,
            "state": list(python_state),
            "gauss_next": python_gauss,
        },
        "numpy": {
            "generator": numpy_generator,
            "state": torch.from_numpy(numpy_state.astype(np.int64)),
            "position": numpy_position,
            "has_gauss": numpy_has_gauss,
            "cached_gaussian": numpy_cached,
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing {sorted(missing)}")
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_state = state["torch"]
    if not isinstance(python_state, Mapping) or not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint RNG state is malformed")
    python_version = python_state.get("version")
    python_words = python_state.get("state")
    python_gauss = python_state.get("gauss_next")
    if (
        isinstance(python_version, bool)
        or not isinstance(python_version, int)
        or not isinstance(python_words, list)
        or any(isinstance(word, bool) or not isinstance(word, int) for word in python_words)
        or (python_gauss is not None and not isinstance(python_gauss, float))
    ):
        raise ValueError("checkpoint Python RNG state is malformed")
    generator = numpy_state.get("generator")
    words = numpy_state.get("state")
    position = numpy_state.get("position")
    has_gauss = numpy_state.get("has_gauss")
    cached_gaussian = numpy_state.get("cached_gaussian")
    if (
        not isinstance(generator, str)
        or not isinstance(words, Tensor)
        or words.dtype not in {torch.int64, torch.uint32}
        or isinstance(position, bool)
        or not isinstance(position, int)
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, int)
        or not isinstance(cached_gaussian, float)
    ):
        raise ValueError("checkpoint NumPy RNG state is malformed")
    words_cpu = words.detach().cpu()
    if words_cpu.dtype == torch.int64 and (
        bool(torch.any(words_cpu < 0))
        or bool(torch.any(words_cpu > np.iinfo(np.uint32).max))
    ):
        raise ValueError("checkpoint NumPy RNG state is malformed")
    restored_numpy_words = words_cpu.numpy().astype(np.uint32, copy=False)
    if not isinstance(torch_state, Tensor) or torch_state.dtype != torch.uint8:
        raise ValueError("checkpoint torch RNG state is malformed")
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not isinstance(cuda_state, list) or any(
            not isinstance(value, Tensor) or value.dtype != torch.uint8 for value in cuda_state
        ):
            raise ValueError("checkpoint CUDA RNG state is malformed")
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint includes CUDA RNG state but CUDA is unavailable")
    random.setstate((python_version, tuple(python_words), python_gauss))
    np.random.set_state(
        (generator, restored_numpy_words, position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cast(list[Tensor], cuda_state))


def _durable_torch_save(payload: object, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class Trainer:
    """AdamW trainer with gradient accumulation, resume, telemetry, and snapshots.

    Direct construction accepts an already-initialized model. Use ``create``
    when the training seed must determine model initialization as well.
    """

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        config: TrainConfig,
        run_directory: Path,
        *,
        per_example_loss_hook: PerExampleLossHook | None = None,
        run_metadata: Mapping[str, str] | None = None,
        trajectory_metadata: SnapshotMetadata | None = None,
    ) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.precision = select_precision(self.device, config.precision)
        seed_everything(config.seed, config.deterministic)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision.use_scaler)
        self.state = TrainingState()
        self.run_directory = run_directory
        self.checkpoint_directory = run_directory / "checkpoints"
        self.trajectory_directory = run_directory / "trajectory"
        self.telemetry = JSONLTelemetry(run_directory / "telemetry.jsonl")
        self.per_example_loss_hook = per_example_loss_hook
        self.run_metadata = dict(run_metadata or {})
        self.trajectory_metadata = trajectory_metadata
        self.model_config_hash = mapping_hash(asdict(model.config))
        self.train_config_hash = mapping_hash(asdict(config))
        if trajectory_metadata is not None and (
            trajectory_metadata.model_config_hash != self.model_config_hash
            or trajectory_metadata.training_config_hash != self.train_config_hash
        ):
            raise ValueError(
                "trajectory metadata configuration hashes do not match the live trainer"
            )
        self.trajectory_template_hash = (
            mapping_hash(asdict(trajectory_metadata)) if trajectory_metadata is not None else None
        )
        self.run_identity = mapping_hash(
            {
                "model": self.model_config_hash,
                "training": self.train_config_hash,
                "metadata": self.run_metadata,
                "trajectory_template": self.trajectory_template_hash,
            }
        )
        self._analysis_reference = (
            self._capture_analysis_reference() if config.analysis_every_steps else None
        )
        self._active_stream: CheckpointableBatchStream | None = None

    @classmethod
    def create(
        cls,
        model_config: ModelConfig,
        train_config: TrainConfig,
        run_directory: Path,
        *,
        per_example_loss_hook: PerExampleLossHook | None = None,
        run_metadata: Mapping[str, str] | None = None,
        trajectory_metadata: SnapshotMetadata | None = None,
    ) -> Trainer:
        """Seed first, then construct a model and its trainer reproducibly."""
        seed_everything(train_config.seed, train_config.deterministic)
        return cls(
            DecoderOnlyTransformer(model_config),
            train_config,
            run_directory,
            per_example_loss_hook=per_example_loss_hook,
            run_metadata=run_metadata,
            trajectory_metadata=trajectory_metadata,
        )

    def _build_optimizer(self) -> Optimizer:
        if self.model.config.architecture == "ngpt" and self.config.weight_decay != 0.0:
            raise ValueError(
                "nGPT requires weight_decay=0.0 because normalized matrices are retracted"
            )
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and not any(
                token in name for token in ("norm", "scale", "rate", "alpha")
            ):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        if not decay and not no_decay:
            raise ValueError("model has no trainable parameters")
        return AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> LambdaLR:
        def scale(step: int) -> float:
            if step < self.config.warmup_steps:
                return (step + 1) / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return (
                self.config.min_learning_rate_ratio
                + (1.0 - self.config.min_learning_rate_ratio) * cosine
            )

        return LambdaLR(self.optimizer, lr_lambda=scale)

    def _autocast(self) -> contextlib.AbstractContextManager[None]:
        if self.precision.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.precision.autocast_dtype)

    def _to_device(self, batch: Batch) -> Batch:
        required = {"input_ids", "targets"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"batch is missing required fields: {sorted(missing)}")
        converted: dict[str, object] = {}
        for key, value in batch.items():
            converted[key] = (
                value.to(self.device, non_blocking=self.device.type == "cuda")
                if isinstance(value, Tensor)
                else value
            )
        if not isinstance(converted["input_ids"], Tensor) or not isinstance(
            converted["targets"], Tensor
        ):
            raise TypeError("input_ids and targets must be tensors")
        return cast(Batch, converted)

    def fit(
        self, stream: CheckpointableBatchStream, *, until_step: int | None = None
    ) -> TrainingState:
        if until_step is None:
            until_step = self.config.max_steps
        if not self.state.global_step <= until_step <= self.config.max_steps:
            raise ValueError("until_step must lie between the current and configured max step")
        self._active_stream = stream
        self.model.train()
        while self.state.global_step < until_step:
            step_started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            loss_weight_sum = 0.0
            aggregate_data_tokens = 0
            aggregate_supervised_tokens = 0
            last_output: ModelOutput | None = None
            microbatches = [
                self._to_device(next(stream))
                for _ in range(self.config.gradient_accumulation_steps)
            ]
            loss_weights = [self._loss_weight(batch) for batch in microbatches]
            total_loss_weight = sum(loss_weights)
            if total_loss_weight <= 0.0:
                raise ValueError("gradient accumulation contains no supervised token weight")
            for batch, loss_weight in zip(microbatches, loss_weights, strict=True):
                input_ids = batch["input_ids"]
                targets = batch["targets"]
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None and not isinstance(loss_mask, Tensor):
                    raise TypeError("loss_mask must be a tensor")
                with self._autocast():
                    output = self.model(
                        input_ids,
                        targets,
                        loss_mask,
                        active_layers=self.model.config.n_layers,
                    )
                    if output.loss is None:
                        raise RuntimeError("model did not return a training loss")
                    scaled_loss = output.loss * (loss_weight / total_loss_weight)
                self.scaler.scale(scaled_loss).backward()
                weighted_loss_sum += float(output.loss.detach().item()) * loss_weight
                loss_weight_sum += loss_weight
                aggregate_data_tokens += self._data_token_count(batch)
                aggregate_supervised_tokens += output.token_count
                last_output = output
                self.state.micro_step += 1
                if self.per_example_loss_hook is not None and output.per_example_loss is not None:
                    example_ids = batch.get("example_ids")
                    if example_ids is not None and (
                        isinstance(example_ids, (str, bytes))
                        or not isinstance(example_ids, Sequence)
                    ):
                        raise TypeError(
                            "example_ids must be a sequence of string or integer identifiers"
                        )
                    if example_ids is not None and len(example_ids) != input_ids.shape[0]:
                        raise ValueError("example_ids length must match the batch row count")
                    self.per_example_loss_hook(
                        self.state.global_step,
                        example_ids,
                        output.per_example_loss.detach().cpu(),
                    )
            if last_output is None:
                raise RuntimeError("no microbatches were consumed")
            if self.precision.use_scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._gradient_norm()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.retract_normalized_parameters_()
            self.scheduler.step()
            self.state.global_step += 1
            self.state.optimizer_steps_executed += 1
            self.state.data_tokens_seen += aggregate_data_tokens
            self.state.supervised_tokens_seen += aggregate_supervised_tokens
            duration = time.perf_counter() - step_started
            self.state.elapsed_seconds += duration
            if self.state.global_step % self.config.log_every_steps == 0:
                self.telemetry.write(
                    {
                        "event": "train_step",
                        "step": self.state.global_step,
                        "micro_step": self.state.micro_step,
                        "optimizer_steps_executed": self.state.optimizer_steps_executed,
                        "virtual_steps_skipped": self.state.virtual_steps_skipped,
                        "loss": weighted_loss_sum / loss_weight_sum,
                        "grad_norm": grad_norm,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "active_layers": self.model.config.n_layers,
                        "data_tokens_this_step": aggregate_data_tokens,
                        "supervised_tokens_this_step": aggregate_supervised_tokens,
                        "data_tokens_seen": self.state.data_tokens_seen,
                        "supervised_tokens_seen": self.state.supervised_tokens_seen,
                        "seconds_this_step": duration,
                        "elapsed_seconds": self.state.elapsed_seconds,
                        "data_tokens_per_second": aggregate_data_tokens / max(duration, 1e-12),
                    }
                )
            if (
                self.config.analysis_every_steps
                and self.state.global_step % self.config.analysis_every_steps == 0
            ):
                self._write_analysis_event()
            if self._is_capture_step(
                self.config.trajectory_every_steps, self.config.trajectory_capture_steps
            ):
                self.save_trajectory_snapshot()
            if self._is_capture_step(
                self.config.checkpoint_every_steps, self.config.checkpoint_steps
            ):
                self.save_checkpoint()
        return self.state

    def apply_verified_transport(
        self,
        verification: CandidateVerification,
        stream: CheckpointableBatchStream,
        *,
        optimizer_state_policy: Literal["retain", "reset"] = "retain",
    ) -> SkippedBatchStats:
        """Apply one accepted, already-retracted forecast as an atomic virtual leap.

        The candidate has already passed verification.  This method never
        predicts optimizer moments: it either retains the existing state or
        explicitly clears it, then advances the data/schedule position without
        claiming skipped corpus tokens were processed by gradient descent.
        """
        if optimizer_state_policy not in {"retain", "reset"}:
            raise ValueError("optimizer_state_policy must be 'retain' or 'reset'")
        if not verification.decision.accepted:
            raise ValueError("refusing to apply a rejected transport candidate")
        if verification.decision.candidate_state_hash != verification.prepared.state_hash:
            raise ValueError("candidate decision hash does not match the prepared forecast")
        forecast = verification.prepared.forecast
        if state_dict_hash(verification.prepared.state_dict) != verification.prepared.state_hash:
            raise ValueError("prepared forecast state does not match its verified hash")
        if not forecast.source_steps or forecast.source_steps[-1] != self.state.global_step:
            raise ValueError("transport forecast must be anchored at the current global_step")
        if not self.state.global_step < forecast.target_step <= self.config.max_steps:
            raise ValueError(
                "transport target_step must be later than current and within max_steps"
            )
        current_state = self.model.state_dict()
        candidate_state = verification.prepared.state_dict
        if tuple(current_state) != tuple(candidate_state):
            raise ValueError("prepared transport coordinates do not match the live model")
        for name, current_tensor in current_state.items():
            candidate_tensor = candidate_state[name]
            if (
                current_tensor.shape != candidate_tensor.shape
                or current_tensor.dtype != candidate_tensor.dtype
            ):
                raise ValueError(f"prepared transport tensor {name} does not match the live model")
            if candidate_tensor.is_floating_point() and not bool(
                torch.isfinite(candidate_tensor).all()
            ):
                raise ValueError(f"prepared transport tensor {name} is non-finite")
        jump_norms = self._layer_delta_norms(current_state, candidate_state)
        skipped_optimizer_steps = forecast.target_step - self.state.global_step
        skipped_microbatches = skipped_optimizer_steps * self.config.gradient_accumulation_steps
        stream_state = dict(stream.state_dict())
        model_state = {
            name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()
        }
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        scaler_state = copy.deepcopy(self.scaler.state_dict())
        training_state = replace(self.state)
        analysis_reference = self._clone_analysis_reference(self._analysis_reference)
        active_stream = self._active_stream
        try:
            skipped = stream.skip_batches(skipped_microbatches)
            if not isinstance(skipped, SkippedBatchStats):
                raise TypeError("stream skip_batches must return SkippedBatchStats")
            if skipped.batch_count != skipped_microbatches:
                raise ValueError("stream skip_batches returned an incorrect batch count")
            self.model.load_state_dict(dict(candidate_state), strict=True)
            if optimizer_state_policy == "reset":
                self.optimizer.state.clear()
            for _ in range(skipped_optimizer_steps):
                self.scheduler.step()
            self._active_stream = stream
            self.state.global_step = forecast.target_step
            self.state.virtual_steps_skipped += skipped_optimizer_steps
            self.state.micro_batches_skipped += skipped.batch_count
            self.state.data_tokens_skipped += skipped.data_token_count
            if self.config.analysis_every_steps:
                self._analysis_reference = self._capture_analysis_reference()
            self.telemetry.write(
                {
                    "event": "verified_transport",
                    "source_step": forecast.source_steps[-1],
                    "target_step": forecast.target_step,
                    "candidate_state_hash": verification.prepared.state_hash,
                    "jump_parameter_norms": jump_norms,
                    "optimizer_state_policy": optimizer_state_policy,
                    "skipped_optimizer_steps": skipped_optimizer_steps,
                    "skipped_micro_batches": skipped.batch_count,
                    "skipped_data_tokens": skipped.data_token_count,
                    "data_tokens_seen": self.state.data_tokens_seen,
                    "data_tokens_skipped": self.state.data_tokens_skipped,
                    "optimizer_steps_executed": self.state.optimizer_steps_executed,
                    "virtual_steps_skipped": self.state.virtual_steps_skipped,
                }
            )
        except Exception:
            self.model.load_state_dict(model_state, strict=True)
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler.load_state_dict(scheduler_state)
            self.scaler.load_state_dict(scaler_state)
            self.state = training_state
            self._analysis_reference = analysis_reference
            self._active_stream = active_stream
            stream.load_state_dict(stream_state)
            raise
        return skipped

    def _is_capture_step(self, interval: int, scheduled_steps: tuple[int, ...]) -> bool:
        return self.state.global_step in scheduled_steps or (
            interval > 0 and self.state.global_step % interval == 0
        )

    @staticmethod
    def _data_token_count(batch: Batch) -> int:
        supplied = batch.get("data_token_count")
        dense_count = int(batch["input_ids"].numel())
        if supplied is None:
            return dense_count
        if isinstance(supplied, bool) or not isinstance(supplied, int):
            raise TypeError("data_token_count must be an integer")
        if not 0 < supplied <= dense_count:
            raise ValueError("data_token_count must lie in [1, input_ids.numel()]")
        return supplied

    def _loss_weight(self, batch: Batch) -> float:
        targets = batch["targets"]
        valid = targets.ne(self.model.config.ignore_index)
        loss_mask = batch.get("loss_mask")
        if loss_mask is None:
            return float(valid.sum().item())
        if not isinstance(loss_mask, Tensor):
            raise TypeError("loss_mask must be a tensor")
        if loss_mask.shape != targets.shape:
            raise ValueError("loss_mask must have the same shape as targets")
        if loss_mask.dtype == torch.bool:
            return float((valid & loss_mask).sum().item())
        if not loss_mask.dtype.is_floating_point:
            raise TypeError("loss_mask must be floating point or bool")
        if not torch.isfinite(loss_mask).all() or torch.any(loss_mask < 0):
            raise ValueError("loss_mask must contain only finite, non-negative weights")
        return float((loss_mask * valid).sum().item())

    def _write_analysis_event(self) -> None:
        current = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }
        per_layer: dict[str, dict[str, float]] = {}
        for name, _parameter in self.model.named_parameters():
            layer = (
                ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
            )
            entry = per_layer.setdefault(
                layer, {"parameter_squared_norm": 0.0, "update_squared_norm": 0.0, "dot": 0.0}
            )
            now = current[name]
            entry["parameter_squared_norm"] += float(now.square().sum())
            if self._analysis_reference is not None:
                update = now - self._analysis_reference[name]
                entry["update_squared_norm"] += float(update.square().sum())
                entry["dot"] += float((now * update).sum())
        metrics = {
            name: {
                "parameter_norm": values["parameter_squared_norm"] ** 0.5,
                "update_norm": values["update_squared_norm"] ** 0.5,
                "update_cosine": values["dot"]
                / max(
                    1e-12, (values["parameter_squared_norm"] * values["update_squared_norm"]) ** 0.5
                ),
            }
            for name, values in per_layer.items()
        }
        peak_allocated_memory = None
        if self.device.type == "cuda":
            peak_allocated_memory = torch.cuda.max_memory_allocated(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            peak_allocated_memory = torch.mps.current_allocated_memory()
        self.telemetry.write(
            {
                "event": "layer_analysis",
                "step": self.state.global_step,
                "per_layer": metrics,
                "peak_allocated_bytes_cuda": peak_allocated_memory
                if self.device.type == "cuda"
                else None,
                "current_allocated_bytes_mps": peak_allocated_memory
                if self.device.type == "mps"
                else None,
            }
        )
        self._analysis_reference = current

    @staticmethod
    def _layer_delta_norms(
        current: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
    ) -> dict[str, float]:
        squared_norms: dict[str, float] = {}
        for name, current_tensor in current.items():
            layer = (
                ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
            )
            delta = candidate[name].detach().float() - current_tensor.detach().float()
            squared_norms[layer] = squared_norms.get(layer, 0.0) + float(delta.square().sum())
        return {name: value**0.5 for name, value in squared_norms.items()}

    def _capture_analysis_reference(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }

    @staticmethod
    def _clone_analysis_reference(
        reference: Mapping[str, Tensor] | None,
    ) -> dict[str, Tensor] | None:
        if reference is None:
            return None
        return {name: tensor.detach().clone() for name, tensor in reference.items()}

    def _gradient_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().square().sum()
        return float(squared_norm.sqrt().item())

    def _stream_state(self) -> Mapping[str, Any]:
        if self._active_stream is None:
            raise RuntimeError("no active batch stream is available for checkpointing")
        return dict(self._active_stream.state_dict())

    def save_checkpoint(self, path: Path | None = None) -> Path:
        if path is None:
            path = self.checkpoint_directory / f"step_{self.state.global_step:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 2,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": capture_rng_state(),
            "training_state": asdict(self.state),
            "data_cursor": self._stream_state(),
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "model_config_hash": self.model_config_hash,
            "train_config_hash": self.train_config_hash,
            "run_identity": self.run_identity,
            "run_metadata": self.run_metadata,
            "analysis_reference": self._analysis_reference,
            "trajectory_template_hash": self.trajectory_template_hash,
        }
        _durable_torch_save(checkpoint, path)
        self.telemetry.write(
            {"event": "checkpoint", "step": self.state.global_step, "path": str(path)}
        )
        return path

    def load_checkpoint(
        self,
        path: Path,
        stream: CheckpointableBatchStream,
        *,
        record_resume_event: bool = True,
    ) -> TrainingState:
        if not isinstance(record_resume_event, bool):
            raise TypeError("record_resume_event must be a boolean")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise ValueError(
                "checkpoint must be a restricted tensor-and-primitive payload"
            ) from error
        if not isinstance(checkpoint, Mapping) or checkpoint.get("format_version") != 2:
            raise ValueError("unsupported or malformed checkpoint")
        required = {
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "rng",
            "training_state",
            "data_cursor",
            "model_config_hash",
            "train_config_hash",
            "run_identity",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
        if checkpoint["model_config_hash"] != self.model_config_hash:
            raise ValueError("checkpoint model configuration does not match this trainer")
        if checkpoint["train_config_hash"] != self.train_config_hash:
            raise ValueError("checkpoint training configuration does not match this trainer")
        if checkpoint["run_identity"] != self.run_identity:
            raise ValueError("checkpoint run identity does not match this trainer")
        if checkpoint.get("trajectory_template_hash") != self.trajectory_template_hash:
            raise ValueError("checkpoint trajectory metadata contract does not match this trainer")
        model_payload = checkpoint["model"]
        optimizer_payload = checkpoint["optimizer"]
        scheduler_payload = checkpoint["scheduler"]
        scaler_payload = checkpoint["scaler"]
        rng_payload = checkpoint["rng"]
        if not all(
            isinstance(payload, Mapping)
            for payload in (
                model_payload,
                optimizer_payload,
                scheduler_payload,
                scaler_payload,
                rng_payload,
            )
        ):
            raise ValueError(
                "checkpoint optimizer, model, scheduler, scaler, or RNG state is malformed"
            )
        state = checkpoint["training_state"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint training_state is malformed")
        try:
            loaded_state = TrainingState(**dict(state))
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint training_state is malformed") from error
        data_cursor = checkpoint["data_cursor"]
        if not isinstance(data_cursor, Mapping):
            raise ValueError("checkpoint data_cursor is malformed")
        analysis_reference = checkpoint.get("analysis_reference")
        loaded_analysis_reference: dict[str, Tensor] | None = None
        if self.config.analysis_every_steps:
            if analysis_reference is not None and not isinstance(analysis_reference, Mapping):
                raise ValueError("checkpoint analysis_reference is malformed")
            if analysis_reference is not None:
                loaded_analysis_reference = {
                    name: tensor.detach().float().cpu().clone()
                    for name, tensor in analysis_reference.items()
                    if isinstance(name, str) and isinstance(tensor, Tensor)
                }
                expected_names = {name for name, _ in self.model.named_parameters()}
                if set(loaded_analysis_reference) != expected_names:
                    raise ValueError(
                        "checkpoint analysis reference does not match model parameters"
                    )
        previous_model_state = {
            name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()
        }
        previous_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        previous_scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        previous_scaler_state = copy.deepcopy(self.scaler.state_dict())
        previous_training_state = replace(self.state)
        previous_analysis_reference = self._clone_analysis_reference(self._analysis_reference)
        previous_active_stream = self._active_stream
        previous_stream_state = dict(stream.state_dict())
        previous_rng_state = capture_rng_state()
        try:
            self.model.load_state_dict(cast(Mapping[str, Tensor], model_payload), strict=True)
            self.optimizer.load_state_dict(dict(cast(Mapping[str, Any], optimizer_payload)))
            self.scheduler.load_state_dict(cast(dict[str, Any], scheduler_payload))
            self.scaler.load_state_dict(cast(dict[str, Any], scaler_payload))
            stream.load_state_dict(cast(Mapping[str, Any], data_cursor))
            self.state = loaded_state
            self._active_stream = stream
            self._analysis_reference = (
                self._capture_analysis_reference()
                if self.config.analysis_every_steps and loaded_analysis_reference is None
                else loaded_analysis_reference
            )
            restore_rng_state(cast(Mapping[str, Any], rng_payload))
        except Exception:
            self.model.load_state_dict(previous_model_state, strict=True)
            self.optimizer.load_state_dict(previous_optimizer_state)
            self.scheduler.load_state_dict(previous_scheduler_state)
            self.scaler.load_state_dict(previous_scaler_state)
            self.state = previous_training_state
            self._analysis_reference = previous_analysis_reference
            self._active_stream = previous_active_stream
            stream.load_state_dict(previous_stream_state)
            restore_rng_state(previous_rng_state)
            raise
        if record_resume_event:
            self.telemetry.write(
                {"event": "resume", "step": self.state.global_step, "path": str(path)}
            )
        return self.state

    def save_trajectory_snapshot(self, path: Path | None = None) -> Path:
        """Save a strict, atomic trajectory snapshot using the shared contract."""
        if self.trajectory_metadata is None:
            raise RuntimeError("trajectory_metadata is required to save a trajectory snapshot")
        if path is None:
            path = self.trajectory_directory / f"step_{self.state.global_step:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = replace(
            self.trajectory_metadata,
            checkpoint_id=f"step-{self.state.global_step:08d}",
            step=self.state.global_step,
            wall_seconds=self.state.elapsed_seconds,
            tokens_seen=self.state.data_tokens_seen,
        )
        snapshot = WeightSnapshot(metadata=metadata, state_dict=self.model.state_dict())
        save_weight_snapshot(path, snapshot)
        self.telemetry.write(
            {"event": "trajectory_snapshot", "step": self.state.global_step, "path": str(path)}
        )
        return path
