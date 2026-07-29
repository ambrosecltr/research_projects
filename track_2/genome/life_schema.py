from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, cast

from .hashing import sha256_json

Split = Literal["training", "development", "hidden"]
Completeness = Literal["complete", "partial", "endpoint_only"]
CheckpointAccess = Literal["available", "sealed", "missing"]
StageType = Literal[
    "pretraining",
    "continued_pretraining",
    "sft",
    "dpo",
    "rl",
    "rlvr",
    "distillation",
    "other",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_SPLITS = {"training", "development", "hidden"}
_ALLOWED_COMPLETENESS = {"complete", "partial", "endpoint_only"}
_ALLOWED_ACCESS = {"available", "sealed", "missing"}
_ALLOWED_STAGE_TYPES = {
    "pretraining",
    "continued_pretraining",
    "sft",
    "dpo",
    "rl",
    "rlvr",
    "distillation",
    "other",
}


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_nonempty(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name=name)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _non_negative_int(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return dict(cast(Mapping[str, Any], value))


def _strings(value: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    result = tuple(_nonempty(item, name=f"{name} item") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to source or derived bytes.

    Artifact hashes are provenance. They must never be converted into semantic compiler
    features. A hidden endpoint uses ``CheckpointRef(access="sealed")`` and therefore has no
    ArtifactRef available to training code.
    """

    uri: str
    sha256: str
    bytes: int
    revision: str | None = None
    licence: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.uri, name="artifact.uri")
        _sha256(self.sha256, name="artifact.sha256")
        _non_negative_int(self.bytes, name="artifact.bytes")
        _optional_nonempty(self.revision, name="artifact.revision")
        _optional_nonempty(self.licence, name="artifact.licence")
        _optional_nonempty(self.media_type, name="artifact.media_type")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        return cls(**_mapping(value, name="artifact"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    step: int
    tokens_seen: int
    access: CheckpointAccess
    artifact: ArtifactRef | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint_id, name="checkpoint_id")
        _non_negative_int(self.step, name=f"checkpoint {self.checkpoint_id}.step")
        _non_negative_int(self.tokens_seen, name=f"checkpoint {self.checkpoint_id}.tokens_seen")
        if self.access not in _ALLOWED_ACCESS:
            raise ValueError(f"unsupported checkpoint access: {self.access!r}")
        if self.access == "available" and self.artifact is None:
            raise ValueError(f"available checkpoint {self.checkpoint_id} requires an artifact")
        if self.access != "available" and self.artifact is not None:
            raise ValueError(
                f"{self.access} checkpoint {self.checkpoint_id} must not expose endpoint bytes"
            )
        _mapping(self.evaluation, name=f"checkpoint {self.checkpoint_id}.evaluation")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointRef:
        data = _mapping(value, name="checkpoint")
        raw_artifact = data.get("artifact")
        data["artifact"] = None if raw_artifact is None else ArtifactRef.from_dict(raw_artifact)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact"] = None if self.artifact is None else self.artifact.to_dict()
        return data


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    repository: str
    revision: str
    configuration: str | None
    split: str
    licence: str
    order_id: str | None
    mixture_weight: float
    semantic_fingerprint: ArtifactRef | None
    source_files: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.dataset_id, name="dataset_id")
        _nonempty(self.repository, name=f"dataset {self.dataset_id}.repository")
        _nonempty(self.revision, name=f"dataset {self.dataset_id}.revision")
        _optional_nonempty(self.configuration, name=f"dataset {self.dataset_id}.configuration")
        _nonempty(self.split, name=f"dataset {self.dataset_id}.split")
        _nonempty(self.licence, name=f"dataset {self.dataset_id}.licence")
        _optional_nonempty(self.order_id, name=f"dataset {self.dataset_id}.order_id")
        if isinstance(self.mixture_weight, bool) or not isinstance(
            self.mixture_weight, (int, float)
        ):
            raise TypeError(f"dataset {self.dataset_id}.mixture_weight must be numeric")
        if not 0.0 < float(self.mixture_weight):
            raise ValueError(f"dataset {self.dataset_id}.mixture_weight must be positive")
        if not isinstance(self.source_files, tuple):
            raise TypeError("dataset source_files must be a tuple")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetRef:
        data = _mapping(value, name="dataset")
        raw_fingerprint = data.get("semantic_fingerprint")
        data["semantic_fingerprint"] = (
            None if raw_fingerprint is None else ArtifactRef.from_dict(raw_fingerprint)
        )
        raw_files = data.get("source_files", [])
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise TypeError("dataset.source_files must be an array")
        data["source_files"] = tuple(ArtifactRef.from_dict(item) for item in raw_files)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["semantic_fingerprint"] = (
            None if self.semantic_fingerprint is None else self.semantic_fingerprint.to_dict()
        )
        data["source_files"] = [item.to_dict() for item in self.source_files]
        return data


@dataclass(frozen=True)
class TrainingStage:
    stage_id: str
    stage_type: StageType
    objective: str
    dataset_ids: tuple[str, ...]
    start_checkpoint_id: str
    end_checkpoint_id: str
    steps: int
    tokens: int
    context_length: int
    global_batch_tokens: int
    data_order_id: str | None
    precision: str
    optimizer: dict[str, Any]
    schedule: dict[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.stage_id, name="stage_id")
        if self.stage_type not in _ALLOWED_STAGE_TYPES:
            raise ValueError(f"unsupported stage type: {self.stage_type!r}")
        _nonempty(self.objective, name=f"stage {self.stage_id}.objective")
        _strings(self.dataset_ids, name=f"stage {self.stage_id}.dataset_ids")
        _nonempty(self.start_checkpoint_id, name=f"stage {self.stage_id}.start_checkpoint_id")
        _nonempty(self.end_checkpoint_id, name=f"stage {self.stage_id}.end_checkpoint_id")
        if self.start_checkpoint_id == self.end_checkpoint_id:
            raise ValueError(f"stage {self.stage_id} cannot start and end at the same checkpoint")
        _positive_int(self.steps, name=f"stage {self.stage_id}.steps")
        _positive_int(self.tokens, name=f"stage {self.stage_id}.tokens")
        _positive_int(self.context_length, name=f"stage {self.stage_id}.context_length")
        _positive_int(self.global_batch_tokens, name=f"stage {self.stage_id}.global_batch_tokens")
        _optional_nonempty(self.data_order_id, name=f"stage {self.stage_id}.data_order_id")
        _nonempty(self.precision, name=f"stage {self.stage_id}.precision")
        if not _mapping(self.optimizer, name=f"stage {self.stage_id}.optimizer"):
            raise ValueError(f"stage {self.stage_id}.optimizer must not be empty")
        if not _mapping(self.schedule, name=f"stage {self.stage_id}.schedule"):
            raise ValueError(f"stage {self.stage_id}.schedule must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingStage:
        data = _mapping(value, name="training stage")
        data["dataset_ids"] = _strings(
            data.get("dataset_ids"), name=f"stage {data.get('stage_id')}.dataset_ids"
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_ids"] = list(self.dataset_ids)
        return data


@dataclass(frozen=True)
class TokenizerRef:
    repository: str
    revision: str
    tokenizer_class: str
    vocab_size: int
    special_tokens: dict[str, int]
    files: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        _nonempty(self.repository, name="tokenizer.repository")
        _nonempty(self.revision, name="tokenizer.revision")
        _nonempty(self.tokenizer_class, name="tokenizer.tokenizer_class")
        _positive_int(self.vocab_size, name="tokenizer.vocab_size")
        raw_tokens = _mapping(self.special_tokens, name="tokenizer.special_tokens")
        for name, token_id in raw_tokens.items():
            _nonempty(name, name="special token name")
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not (
                0 <= token_id < self.vocab_size
            ):
                raise ValueError(f"special token {name!r} is outside the vocabulary")
        if not self.files:
            raise ValueError("tokenizer.files must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TokenizerRef:
        data = _mapping(value, name="tokenizer")
        raw_files = data.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise TypeError("tokenizer.files must be an array")
        data["files"] = tuple(ArtifactRef.from_dict(item) for item in raw_files)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = [item.to_dict() for item in self.files]
        return data


@dataclass(frozen=True)
class ModelLifeManifest:
    """One complete training organism, not one checkpoint row.

    The manifest intentionally keeps provenance-rich source records separate from the semantic
    compiler view. Cryptographic hashes identify bytes; they are never model features.
    """

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
    compiler_evidence: ArtifactRef | None
    fitted_program: ArtifactRef | None = None
    evaluations: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    format: str = "GENOME_MODEL_LIFE"
    version: str = "0.3.0"

    def __post_init__(self) -> None:
        if self.format != "GENOME_MODEL_LIFE" or self.version != "0.3.0":
            raise ValueError("unsupported model-life format")
        _nonempty(self.run_id, name="run_id")
        _nonempty(self.lineage_id, name="lineage_id")
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.completeness not in _ALLOWED_COMPLETENESS:
            raise ValueError(f"unsupported completeness: {self.completeness!r}")
        _nonempty(self.architecture_family, name="architecture_family")
        if not _mapping(self.architecture, name="architecture"):
            raise ValueError("architecture must not be empty")
        _mapping(self.evaluations, name="evaluations")
        _mapping(self.source, name="source")

        dataset_ids = [item.dataset_id for item in self.datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("model life contains duplicate dataset IDs")
        known_datasets = set(dataset_ids)
        stage_ids = [item.stage_id for item in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("model life contains duplicate stage IDs")
        for stage in self.stages:
            missing = set(stage.dataset_ids) - known_datasets
            if missing:
                raise ValueError(f"stage {stage.stage_id} references unknown datasets: {sorted(missing)}")

        checkpoint_ids: set[str] = set()
        for checkpoint in (
            *((self.initialization,) if self.initialization is not None else ()),
            *self.trajectory,
            *((self.endpoint,) if self.endpoint is not None else ()),
        ):
            if checkpoint.checkpoint_id in checkpoint_ids:
                raise ValueError(f"duplicate checkpoint ID: {checkpoint.checkpoint_id}")
            checkpoint_ids.add(checkpoint.checkpoint_id)

        if self.completeness == "complete":
            self._validate_complete_life()
        elif self.completeness == "endpoint_only":
            if self.endpoint is None:
                raise ValueError("endpoint-only life requires an endpoint")
            if self.initialization is not None or self.stages:
                raise ValueError("endpoint-only life cannot claim initialization or stages")

        if self.split == "hidden":
            if self.endpoint is not None and self.endpoint.access == "available":
                raise ValueError("hidden model life must not expose WT bytes")
            if self.fitted_program is not None:
                raise ValueError("hidden model life must not expose a fitted endpoint program")
        elif self.completeness == "complete":
            if self.endpoint is None or self.endpoint.access != "available":
                raise ValueError("non-hidden complete life requires an available endpoint")

    def _validate_complete_life(self) -> None:
        if self.initialization is None or self.initialization.access != "available":
            raise ValueError("complete life requires an available true initialization")
        if not self.stages:
            raise ValueError("complete life requires at least one training stage")
        if self.endpoint is None:
            raise ValueError("complete life requires an endpoint declaration")
        if self.split == "hidden" and self.endpoint.access != "sealed":
            raise ValueError("hidden complete life must not expose WT; endpoint must be sealed")
        if self.stages[0].start_checkpoint_id != self.initialization.checkpoint_id:
            raise ValueError("first stage must begin at the true initialization checkpoint")
        for previous, current in zip(self.stages, self.stages[1:], strict=False):
            if previous.end_checkpoint_id != current.start_checkpoint_id:
                raise ValueError(
                    f"training stages are discontinuous: {previous.stage_id} -> {current.stage_id}"
                )
        if self.stages[-1].end_checkpoint_id != self.endpoint.checkpoint_id:
            raise ValueError("last stage must end at the declared final endpoint")
        if any(dataset.semantic_fingerprint is None for dataset in self.datasets):
            raise ValueError("complete compiler life requires semantic evidence for every dataset")
        if self.compiler_evidence is None:
            raise ValueError("complete compiler life requires endpoint-free compiler evidence")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelLifeManifest:
        data = _mapping(value, name="model life")
        expected = {
            "format",
            "version",
            "run_id",
            "lineage_id",
            "split",
            "completeness",
            "architecture_family",
            "architecture",
            "tensor_inventory",
            "tokenizer",
            "initialization",
            "datasets",
            "stages",
            "trajectory",
            "endpoint",
            "compiler_evidence",
            "fitted_program",
            "evaluations",
            "source",
        }
        if set(data) != expected:
            raise ValueError(
                f"model-life fields differ; missing={sorted(expected - set(data))}, "
                f"extra={sorted(set(data) - expected)}"
            )
        data["tensor_inventory"] = ArtifactRef.from_dict(data["tensor_inventory"])
        data["tokenizer"] = TokenizerRef.from_dict(data["tokenizer"])
        data["initialization"] = (
            None if data["initialization"] is None else CheckpointRef.from_dict(data["initialization"])
        )
        data["endpoint"] = (
            None if data["endpoint"] is None else CheckpointRef.from_dict(data["endpoint"])
        )
        for name, decoder in (
            ("compiler_evidence", ArtifactRef.from_dict),
            ("fitted_program", ArtifactRef.from_dict),
        ):
            data[name] = None if data[name] is None else decoder(data[name])
        data["datasets"] = tuple(DatasetRef.from_dict(item) for item in data["datasets"])
        data["stages"] = tuple(TrainingStage.from_dict(item) for item in data["stages"])
        data["trajectory"] = tuple(CheckpointRef.from_dict(item) for item in data["trajectory"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "split": self.split,
            "completeness": self.completeness,
            "architecture_family": self.architecture_family,
            "architecture": dict(self.architecture),
            "tensor_inventory": self.tensor_inventory.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "initialization": (
                None if self.initialization is None else self.initialization.to_dict()
            ),
            "datasets": [item.to_dict() for item in self.datasets],
            "stages": [item.to_dict() for item in self.stages],
            "trajectory": [item.to_dict() for item in self.trajectory],
            "endpoint": None if self.endpoint is None else self.endpoint.to_dict(),
            "compiler_evidence": (
                None if self.compiler_evidence is None else self.compiler_evidence.to_dict()
            ),
            "fitted_program": (
                None if self.fitted_program is None else self.fitted_program.to_dict()
            ),
            "evaluations": dict(self.evaluations),
            "source": dict(self.source),
        }

    @property
    def content_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def compiler_view(self) -> dict[str, Any]:
        """Return the only manifest fields a one-shot compiler may consume.

        WT, trajectory checkpoints, evaluations, fitted programs, source hashes and byte paths are
        deliberately absent. The W0 tensor and semantic evidence tensors are loaded separately from
        their validated artifacts.
        """

        return {
            "format": "GENOME_COMPILER_VIEW",
            "version": "0.1.0",
            "architecture_family": self.architecture_family,
            "architecture": dict(self.architecture),
            "tokenizer": {
                "tokenizer_class": self.tokenizer.tokenizer_class,
                "vocab_size": self.tokenizer.vocab_size,
                "special_tokens": dict(self.tokenizer.special_tokens),
            },
            "datasets": [
                {
                    "dataset_id": item.dataset_id,
                    "configuration": item.configuration,
                    "split": item.split,
                    "mixture_weight": float(item.mixture_weight),
                    "order_present": item.order_id is not None,
                }
                for item in self.datasets
            ],
            "stages": [
                {
                    "stage_type": item.stage_type,
                    "objective": item.objective,
                    "dataset_ids": list(item.dataset_ids),
                    "steps": item.steps,
                    "tokens": item.tokens,
                    "context_length": item.context_length,
                    "global_batch_tokens": item.global_batch_tokens,
                    "data_order_present": item.data_order_id is not None,
                    "precision": item.precision,
                    "optimizer": dict(item.optimizer),
                    "schedule": dict(item.schedule),
                }
                for item in self.stages
            ],
        }


def validate_life_splits(manifests: Sequence[ModelLifeManifest]) -> dict[str, str]:
    """Freeze whole-life splits and reject lineage leakage."""

    by_run: dict[str, str] = {}
    by_lineage: dict[str, str] = {}
    for manifest in manifests:
        if manifest.run_id in by_run:
            raise ValueError(f"duplicate model life: {manifest.run_id}")
        previous = by_lineage.get(manifest.lineage_id)
        if previous is not None and previous != manifest.split:
            raise ValueError(
                f"lineage {manifest.lineage_id!r} spans {previous!r} and {manifest.split!r}"
            )
        by_run[manifest.run_id] = manifest.split
        by_lineage[manifest.lineage_id] = manifest.split
    return by_run


def split_commitment(manifests: Sequence[ModelLifeManifest]) -> dict[str, Any]:
    split_by_run = validate_life_splits(manifests)
    payload = {
        "format": "GENOME_LIFE_SPLIT_COMMITMENT",
        "version": "0.1.0",
        "splits": {
            split: sorted(run_id for run_id, value in split_by_run.items() if value == split)
            for split in ("training", "development", "hidden")
        },
        "life_manifest_sha256": {
            manifest.run_id: manifest.content_sha256 for manifest in sorted(manifests, key=lambda x: x.run_id)
        },
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload
