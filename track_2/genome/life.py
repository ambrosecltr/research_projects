from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .hashing import sha256_json
from .io import atomic_write_json, load_json

Split = Literal["training", "development", "hidden"]
Completeness = Literal["complete", "partial", "endpoint_only"]
Access = Literal["available", "sealed", "missing"]
StageKind = Literal[
    "pretraining",
    "continued_pretraining",
    "sft",
    "dpo",
    "rl",
    "rlvr",
    "distillation",
    "other",
]


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    bytes: int
    revision: str | None = None
    licence: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.uri, "artifact.uri")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("artifact.sha256 must be a lowercase SHA-256 digest")
        _nonnegative(self.bytes, "artifact.bytes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(**dict(value))


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    step: int
    tokens_seen: int
    access: Access
    artifact: ArtifactRef | None = None

    def __post_init__(self) -> None:
        _required_text(self.checkpoint_id, "checkpoint_id")
        _nonnegative(self.step, f"{self.checkpoint_id}.step")
        _nonnegative(self.tokens_seen, f"{self.checkpoint_id}.tokens_seen")
        if self.access == "available" and self.artifact is None:
            raise ValueError(f"available checkpoint {self.checkpoint_id} requires an artifact")
        if self.access != "available" and self.artifact is not None:
            raise ValueError(f"{self.access} checkpoint {self.checkpoint_id} cannot expose bytes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointRef":
        data = dict(value)
        if data.get("artifact") is not None:
            data["artifact"] = ArtifactRef.from_dict(data["artifact"])
        return cls(**data)


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    repository: str
    revision: str
    split: str
    licence: str
    order_id: str | None = None
    mixture_weight: float = 1.0
    semantic_evidence: ArtifactRef | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "repository", "revision", "split", "licence"):
            _required_text(getattr(self, name), name)
        if self.mixture_weight <= 0:
            raise ValueError("mixture_weight must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetRef":
        data = dict(value)
        if data.get("semantic_evidence") is not None:
            data["semantic_evidence"] = ArtifactRef.from_dict(data["semantic_evidence"])
        return cls(**data)


@dataclass(frozen=True)
class TrainingStage:
    stage_id: str
    kind: StageKind
    objective: str
    dataset_ids: tuple[str, ...]
    start_checkpoint_id: str
    end_checkpoint_id: str
    steps: int
    tokens: int
    context_length: int
    global_batch_tokens: int
    precision: str
    optimizer: dict[str, Any]
    schedule: dict[str, Any]
    data_order_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("stage_id", "kind", "objective", "start_checkpoint_id", "end_checkpoint_id", "precision"):
            _required_text(str(getattr(self, name)), name)
        if self.start_checkpoint_id == self.end_checkpoint_id:
            raise ValueError("a stage cannot start and end at the same checkpoint")
        if not self.dataset_ids:
            raise ValueError("dataset_ids must not be empty")
        for name in ("steps", "tokens", "context_length", "global_batch_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.optimizer or not self.schedule:
            raise ValueError("optimizer and schedule must be recorded")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingStage":
        data = dict(value)
        data["dataset_ids"] = tuple(data["dataset_ids"])
        return cls(**data)


@dataclass(frozen=True)
class TokenizerRef:
    repository: str
    revision: str
    tokenizer_class: str
    vocab_size: int
    special_tokens: dict[str, int]
    files: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        for name in ("repository", "revision", "tokenizer_class"):
            _required_text(getattr(self, name), name)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not self.files:
            raise ValueError("tokenizer files must be recorded")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenizerRef":
        data = dict(value)
        data["files"] = tuple(ArtifactRef.from_dict(item) for item in data["files"])
        return cls(**data)


@dataclass(frozen=True)
class ModelLife:
    run_id: str
    lineage_id: str
    split: Split
    completeness: Completeness
    architecture_family: str
    architecture: dict[str, Any]
    tensor_inventory: ArtifactRef
    tokenizer: TokenizerRef
    initialization: CheckpointRef | None
    datasets: tuple[DatasetRef, ...]
    stages: tuple[TrainingStage, ...]
    trajectory: tuple[CheckpointRef, ...]
    endpoint: CheckpointRef | None
    compiler_evidence: ArtifactRef | None = None
    accepted_program: ArtifactRef | None = None
    source: dict[str, Any] = field(default_factory=dict)
    format: str = "GENOME_MODEL_LIFE"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        for name in ("run_id", "lineage_id", "architecture_family"):
            _required_text(getattr(self, name), name)
        if self.split not in {"training", "development", "hidden"}:
            raise ValueError(f"unsupported split {self.split!r}")
        if self.completeness not in {"complete", "partial", "endpoint_only"}:
            raise ValueError(f"unsupported completeness {self.completeness!r}")
        if self.format != "GENOME_MODEL_LIFE" or self.version != "1.0.0":
            raise ValueError("unsupported model-life format")
        self._validate_chain()
        self._validate_access()

    def _validate_chain(self) -> None:
        checkpoints: list[CheckpointRef] = []
        if self.initialization is not None:
            checkpoints.append(self.initialization)
        checkpoints.extend(self.trajectory)
        if self.endpoint is not None:
            checkpoints.append(self.endpoint)
        ids = [item.checkpoint_id for item in checkpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("checkpoint IDs must be unique")
        declared = set(ids)
        ordered = sorted(checkpoints, key=lambda item: (item.step, item.tokens_seen))
        if checkpoints != ordered:
            raise ValueError("checkpoints must be monotonic by step and tokens_seen")
        dataset_ids = {item.dataset_id for item in self.datasets}
        for stage in self.stages:
            if stage.start_checkpoint_id not in declared or stage.end_checkpoint_id not in declared:
                raise ValueError(f"stage {stage.stage_id} references an undeclared checkpoint")
            missing = set(stage.dataset_ids) - dataset_ids
            if missing:
                raise ValueError(f"stage {stage.stage_id} references unknown datasets: {sorted(missing)}")
        if self.stages:
            if self.initialization is None:
                raise ValueError("a staged life requires initialization")
            if self.stages[0].start_checkpoint_id != self.initialization.checkpoint_id:
                raise ValueError("the first stage must start at W0")
            for previous, current in zip(self.stages, self.stages[1:]):
                if previous.end_checkpoint_id != current.start_checkpoint_id:
                    raise ValueError("training stages must form one continuous checkpoint chain")
            if self.endpoint is not None and self.stages[-1].end_checkpoint_id != self.endpoint.checkpoint_id:
                raise ValueError("the final stage must end at WT")
        if self.completeness == "complete":
            if self.initialization is None or self.endpoint is None or not self.stages:
                raise ValueError("a complete life requires W0, WT, and at least one stage")

    def _validate_access(self) -> None:
        if self.split == "hidden":
            if self.endpoint is None or self.endpoint.access != "sealed":
                raise ValueError("a hidden life must expose only a sealed WT reference")
            if self.accepted_program is not None:
                raise ValueError("a hidden life cannot contain a fitted endpoint program")
            if any(item.access == "available" for item in self.trajectory):
                raise ValueError("a hidden life cannot expose trajectory checkpoints")
        elif self.endpoint is not None and self.completeness == "complete":
            if self.endpoint.access != "available":
                raise ValueError("training/development complete lives require available WT")

    def compiler_view(self) -> dict[str, Any]:
        """Return endpoint-free metadata; tensor/evidence bytes are loaded separately."""
        return {
            "format": "GENOME_COMPILER_VIEW",
            "version": "1.0.0",
            "split": self.split,
            "architecture_family": self.architecture_family,
            "architecture": self.architecture,
            "datasets": [
                {
                    "dataset_id": item.dataset_id,
                    "repository": item.repository,
                    "revision": item.revision,
                    "split": item.split,
                    "mixture_weight": item.mixture_weight,
                    "order_known": item.order_id is not None,
                }
                for item in self.datasets
            ],
            "stages": [
                {
                    "kind": item.kind,
                    "objective": item.objective,
                    "dataset_ids": list(item.dataset_ids),
                    "steps": item.steps,
                    "tokens": item.tokens,
                    "context_length": item.context_length,
                    "global_batch_tokens": item.global_batch_tokens,
                    "precision": item.precision,
                    "optimizer": item.optimizer,
                    "schedule": item.schedule,
                }
                for item in self.stages
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelLife":
        data = dict(value)
        data["tensor_inventory"] = ArtifactRef.from_dict(data["tensor_inventory"])
        data["tokenizer"] = TokenizerRef.from_dict(data["tokenizer"])
        data["initialization"] = (
            None if data.get("initialization") is None else CheckpointRef.from_dict(data["initialization"])
        )
        data["datasets"] = tuple(DatasetRef.from_dict(item) for item in data.get("datasets", []))
        data["stages"] = tuple(TrainingStage.from_dict(item) for item in data.get("stages", []))
        data["trajectory"] = tuple(CheckpointRef.from_dict(item) for item in data.get("trajectory", []))
        data["endpoint"] = None if data.get("endpoint") is None else CheckpointRef.from_dict(data["endpoint"])
        for key in ("compiler_evidence", "accepted_program"):
            data[key] = None if data.get(key) is None else ArtifactRef.from_dict(data[key])
        return cls(**data)

    @property
    def manifest_id(self) -> str:
        return sha256_json(self.to_dict())

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ModelLife":
        return cls.from_dict(load_json(path))


@dataclass(frozen=True)
class LifeSplits:
    training: tuple[str, ...]
    development: tuple[str, ...]
    hidden: tuple[str, ...]
    format: str = "GENOME_LIFE_SPLITS"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        groups = [self.training, self.development, self.hidden]
        all_ids = [item for group in groups for item in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("a complete model life may appear in only one split")
        if not self.training or not self.development or not self.hidden:
            raise ValueError("training, development and hidden splits must all be non-empty")

    @property
    def split_id(self) -> str:
        return sha256_json(asdict(self))

    def assignment(self, run_id: str) -> Split:
        for split in ("training", "development", "hidden"):
            if run_id in getattr(self, split):
                return split  # type: ignore[return-value]
        raise KeyError(run_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LifeSplits":
        data = dict(value)
        for key in ("training", "development", "hidden"):
            data[key] = tuple(data.get(key, []))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self))
