"""Read-only planning for a fixed data-token pretraining exposure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from poetry50m.config import config_hash, file_hash
from poetry50m.data import PreparedBatchStream, load_prepared_data
from poetry50m.model import DecoderOnlyTransformer, ModelConfig, count_parameters
from poetry50m.training import TrainConfig


@dataclass(frozen=True, slots=True)
class ExposurePlan:
    """The exact stream prefix needed to meet a data-token exposure target."""

    parameter_count: int
    tokens_per_parameter_per_pass: int
    passes: int
    target_data_tokens: int
    planned_data_tokens: int
    planned_steps: int
    data_tokens_by_objective: Mapping[str, int]
    stream_hash: str
    order_digest: str

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.parameter_count,
                self.tokens_per_parameter_per_pass,
                self.passes,
                self.target_data_tokens,
                self.planned_data_tokens,
                self.planned_steps,
            )
        ):
            raise ValueError("exposure plan counts must be positive integers")
        if self.planned_data_tokens < self.target_data_tokens:
            raise ValueError("exposure plan must meet its data-token target")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in self.data_tokens_by_objective.items()
        ):
            raise ValueError("objective data-token counts are invalid")
        if sum(self.data_tokens_by_objective.values()) != self.planned_data_tokens:
            raise ValueError("objective data-token counts do not match the plan")
        if not self.stream_hash or not self.order_digest:
            raise ValueError("exposure plan requires stream identities")

    @property
    def data_tokens_per_parameter(self) -> float:
        return self.planned_data_tokens / self.parameter_count


def exact_trainable_parameter_count(model: ModelConfig) -> int:
    """Build only the declared model shape and return its de-duplicated parameters."""
    return count_parameters(DecoderOnlyTransformer(model), trainable_only=True)


def plan_exposure(
    stream: PreparedBatchStream,
    *,
    parameter_count: int,
    tokens_per_parameter_per_pass: int,
    passes: int,
) -> ExposurePlan:
    """Consume no training data; deterministically count the stream prefix only."""
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (parameter_count, tokens_per_parameter_per_pass, passes)
    ):
        raise ValueError("exposure inputs must be positive integers")
    target_data_tokens = parameter_count * tokens_per_parameter_per_pass * passes
    planned_data_tokens = 0
    planned_steps = 0
    while planned_data_tokens < target_data_tokens:
        skipped = stream.skip_batches(1)
        planned_data_tokens += skipped.data_token_count
        planned_steps += 1
    state = stream.state_dict()
    stream_hash = state.get("stream_hash")
    if not isinstance(stream_hash, str) or not stream_hash:
        raise ValueError("prepared stream did not expose a stable stream hash")
    return ExposurePlan(
        parameter_count=parameter_count,
        tokens_per_parameter_per_pass=tokens_per_parameter_per_pass,
        passes=passes,
        target_data_tokens=target_data_tokens,
        planned_data_tokens=planned_data_tokens,
        planned_steps=planned_steps,
        data_tokens_by_objective=stream.data_tokens_by_objective,
        stream_hash=stream_hash,
        order_digest=stream.order_digest,
    )


def derived_train_config(base: TrainConfig, *, planned_steps: int) -> TrainConfig:
    """Scale step-indexed controls to an exact, frozen stream horizon."""
    if isinstance(planned_steps, bool) or not isinstance(planned_steps, int) or planned_steps < 1:
        raise ValueError("planned_steps must be a positive integer")

    def scaled_step(step: int) -> int:
        return max(1, round(step * planned_steps / base.max_steps))

    def scaled_capture_steps(steps: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted({min(planned_steps, scaled_step(step)) for step in steps}))

    warmup_steps = (
        min(planned_steps - 1, scaled_step(base.warmup_steps)) if base.warmup_steps else 0
    )
    return replace(
        base,
        max_steps=planned_steps,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=(
            scaled_step(base.checkpoint_every_steps) if base.checkpoint_every_steps else 0
        ),
        trajectory_every_steps=(
            scaled_step(base.trajectory_every_steps) if base.trajectory_every_steps else 0
        ),
        analysis_every_steps=(
            scaled_step(base.analysis_every_steps) if base.analysis_every_steps else 0
        ),
        checkpoint_steps=scaled_capture_steps(base.checkpoint_steps),
        trajectory_capture_steps=scaled_capture_steps(base.trajectory_capture_steps),
    )


def exposure_receipt(
    *,
    prepared: Path,
    model_config_path: Path,
    train_config_path: Path,
    batch_size: int,
    data_seed: int,
    plan: ExposurePlan,
    derived_config: TrainConfig,
) -> dict[str, Any]:
    """Create an immutable-review payload without changing any training state."""
    artifact = load_prepared_data(prepared)
    objective_stats = artifact.metadata.get("train_objective_stats")
    if not isinstance(objective_stats, dict):
        raise ValueError("prepared metadata lacks per-objective training token counts")
    objective_exposure: dict[str, dict[str, float | int]] = {}
    for objective, planned_tokens in plan.data_tokens_by_objective.items():
        stats = objective_stats.get(objective)
        if not isinstance(stats, dict):
            raise ValueError(f"prepared metadata lacks {objective!r} token counts")
        unique_data_tokens = stats.get("data_token_count")
        if (
            isinstance(unique_data_tokens, bool)
            or not isinstance(unique_data_tokens, int)
            or unique_data_tokens < 1
        ):
            raise ValueError(f"prepared metadata has invalid {objective!r} data token count")
        objective_exposure[objective] = {
            "planned_data_tokens": planned_tokens,
            "actual_data_token_ratio": planned_tokens / plan.planned_data_tokens,
            "unique_train_data_tokens": unique_data_tokens,
            "unique_pool_passes": planned_tokens / unique_data_tokens,
        }
    return {
        "format_version": 1,
        "mode": "read_only_exposure_plan",
        "prepared_metadata_hash": file_hash(prepared / "metadata.json"),
        "prepared_tokenizer_hash": artifact.metadata["tokenizer_hash"],
        "model_config_hash": file_hash(model_config_path),
        "train_config_source_hash": file_hash(train_config_path),
        "batch_size": batch_size,
        "data_seed": data_seed,
        "parameter_count": plan.parameter_count,
        "tokens_per_parameter_per_pass": plan.tokens_per_parameter_per_pass,
        "passes": plan.passes,
        "target_data_tokens": plan.target_data_tokens,
        "planned_data_tokens": plan.planned_data_tokens,
        "target_overshoot_data_tokens": plan.planned_data_tokens - plan.target_data_tokens,
        "planned_steps": plan.planned_steps,
        "data_tokens_per_parameter": plan.data_tokens_per_parameter,
        "data_tokens_by_objective": dict(plan.data_tokens_by_objective),
        "objective_exposure": objective_exposure,
        "stream_hash": plan.stream_hash,
        "terminal_order_digest": plan.order_digest,
        "derived_train_config": asdict(derived_config),
        "derived_train_config_hash": config_hash(derived_config),
    }
