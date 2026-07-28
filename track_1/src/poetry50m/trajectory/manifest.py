"""Run lineage and sealed-endpoint rules for honest Level 1/2/3 claims."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from poetry50m.trajectory._persistence import atomic_write, load_json_object
from poetry50m.trajectory.snapshots import assert_identical_coordinates
from poetry50m.trajectory.types import WeightSnapshot


class SuccessLevel(StrEnum):
    SAME_RUN = "level_1_same_run"
    TRANSFER = "level_2_transfer"
    GENERAL = "level_3_general"


class OperationScope(StrEnum):
    """The permissible object of an experiment; raw coordinates are deliberately narrow."""

    RAW_WEIGHT_TRANSPORT = "raw_weight_transport"
    FUNCTION_SPACE_DIAGNOSTIC = "function_space_diagnostic"


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"run manifest {name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable identity and endpoint-sealing policy for one run."""

    run_id: str
    initialization_id: str
    data_order_id: str
    architecture_signature: str
    corpus_signature: str
    tokenizer_hash: str
    code_signature: str
    model_config_hash: str
    training_config_hash: str
    endpoint_sealed: bool
    fit_checkpoint_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "tokenizer_hash",
            "code_signature",
            "model_config_hash",
            "training_config_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.endpoint_sealed, bool):
            raise TypeError("endpoint_sealed must be a boolean")
        if not isinstance(self.fit_checkpoint_ids, tuple) or any(
            not isinstance(checkpoint_id, str) or not checkpoint_id
            for checkpoint_id in self.fit_checkpoint_ids
        ):
            raise TypeError("fit_checkpoint_ids must be a tuple of non-empty strings")
        if len(set(self.fit_checkpoint_ids)) != len(self.fit_checkpoint_ids):
            raise ValueError("fit checkpoint IDs must be unique")
        if self.endpoint_sealed and self.fit_checkpoint_ids:
            raise ValueError("a sealed run cannot nominate checkpoints for fitting")

    def save(self, path: Path) -> None:
        payload = json.dumps(asdict(self), indent=2, sort_keys=True).encode("utf-8") + b"\n"

        def write(handle: BinaryIO) -> None:
            handle.write(payload)

        atomic_write(path, write)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        value = load_json_object(path, name="run manifest")
        expected = {
            "run_id",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "tokenizer_hash",
            "code_signature",
            "model_config_hash",
            "training_config_hash",
            "endpoint_sealed",
            "fit_checkpoint_ids",
        }
        if set(value) != expected:
            raise ValueError(f"run manifest must contain exactly {sorted(expected)}")
        endpoint_sealed = value["endpoint_sealed"]
        if not isinstance(endpoint_sealed, bool):
            raise TypeError("run manifest endpoint_sealed must be a boolean")
        checkpoint_ids = value["fit_checkpoint_ids"]
        if not isinstance(checkpoint_ids, list) or any(
            not isinstance(checkpoint_id, str) for checkpoint_id in checkpoint_ids
        ):
            raise TypeError("run manifest fit_checkpoint_ids must be an array of strings")
        return cls(
            run_id=_string(value["run_id"], name="run_id"),
            initialization_id=_string(value["initialization_id"], name="initialization_id"),
            data_order_id=_string(value["data_order_id"], name="data_order_id"),
            architecture_signature=_string(
                value["architecture_signature"], name="architecture_signature"
            ),
            corpus_signature=_string(value["corpus_signature"], name="corpus_signature"),
            tokenizer_hash=_string(value["tokenizer_hash"], name="tokenizer_hash"),
            code_signature=_string(value["code_signature"], name="code_signature"),
            model_config_hash=_string(value["model_config_hash"], name="model_config_hash"),
            training_config_hash=_string(
                value["training_config_hash"], name="training_config_hash"
            ),
            endpoint_sealed=endpoint_sealed,
            fit_checkpoint_ids=tuple(checkpoint_ids),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryExperimentManifest:
    """Reference/target policy; fitting has no path to a sealed target endpoint."""

    level: SuccessLevel
    reference: RunManifest
    target: RunManifest
    operation_scope: OperationScope = OperationScope.RAW_WEIGHT_TRANSPORT

    def __post_init__(self) -> None:
        if self.reference.endpoint_sealed:
            raise ValueError(
                "the reference trajectory must explicitly list its permitted fit checkpoints"
            )
        if not self.reference.fit_checkpoint_ids:
            raise ValueError("the reference manifest requires permitted fit checkpoints")
        if self.target.run_id == self.reference.run_id:
            raise ValueError("reference and target runs must be distinct")
        if not self.target.endpoint_sealed:
            raise ValueError("target endpoint must remain sealed while fitting")
        if self.level is SuccessLevel.SAME_RUN:
            self._require_equal("initialization_id")
            self._require_equal("data_order_id")
            self._require_raw_coordinate_lineage()
        elif self.level is SuccessLevel.TRANSFER:
            if self.operation_scope is OperationScope.RAW_WEIGHT_TRANSPORT:
                self._require_raw_coordinate_lineage()
                self._require_equal("initialization_id")
                if self.reference.data_order_id == self.target.data_order_id:
                    raise ValueError("raw Level 2 transport requires an unseen data order")
            elif (
                self.reference.initialization_id == self.target.initialization_id
                and self.reference.data_order_id == self.target.data_order_id
            ):
                raise ValueError(
                    "Level 2 function diagnostics require an unseen initialization or data order"
                )
        elif self.level is SuccessLevel.GENERAL:
            if self.operation_scope is OperationScope.RAW_WEIGHT_TRANSPORT:
                raise ValueError("raw weight transport is not a Level 3 cross-structure method")

    def _require_equal(self, name: str) -> None:
        if getattr(self.reference, name) != getattr(self.target, name):
            raise ValueError(f"{self.level.value} requires matching {name}")

    def _require_raw_coordinate_lineage(self) -> None:
        for name in (
            "architecture_signature",
            "corpus_signature",
            "tokenizer_hash",
            "code_signature",
            "model_config_hash",
            "training_config_hash",
        ):
            self._require_equal(name)

    def validate_fit_sources(
        self, snapshots: Iterable[WeightSnapshot]
    ) -> tuple[WeightSnapshot, ...]:
        values = tuple(snapshots)
        if not values:
            raise ValueError("trajectory fit requires snapshots")
        allowed = set(self.reference.fit_checkpoint_ids)
        for snapshot in values:
            metadata = snapshot.metadata
            if metadata.run_id != self.reference.run_id:
                raise ValueError("fit attempted to use a non-reference run")
            self._validate_snapshot_metadata(snapshot, self.reference)
            if metadata.checkpoint_id not in allowed:
                raise ValueError(
                    "fit attempted to use a checkpoint not declared in the reference manifest"
                )
            if (
                metadata.run_id == self.target.run_id
                or metadata.checkpoint_id in self.target.fit_checkpoint_ids
            ):
                raise ValueError("sealed target checkpoint leakage detected")
        return values

    def validate_target_snapshot(
        self, snapshot: WeightSnapshot, reference_snapshot: WeightSnapshot
    ) -> None:
        if self.operation_scope is not OperationScope.RAW_WEIGHT_TRANSPORT:
            raise ValueError("function-space diagnostics cannot apply raw forecast weights")
        if snapshot.metadata.run_id != self.target.run_id:
            raise ValueError("application snapshot does not belong to the target run")
        if snapshot.metadata.initialization_id != self.reference.initialization_id:
            raise ValueError("raw transport rejects cross-seed target coordinates")
        self._validate_snapshot_metadata(reference_snapshot, self.reference)
        self._validate_snapshot_metadata(snapshot, self.target)
        assert_identical_coordinates(reference_snapshot, snapshot)
        self._require_raw_coordinate_lineage()
        if (
            self.level is SuccessLevel.SAME_RUN
            and snapshot.metadata.data_order_id != self.reference.data_order_id
        ):
            raise ValueError("Level 1 raw transport requires the reference data order")
        if (
            self.level is SuccessLevel.TRANSFER
            and snapshot.metadata.data_order_id == self.reference.data_order_id
        ):
            raise ValueError("Level 2 raw transport requires an unseen data order")

    @staticmethod
    def _validate_snapshot_metadata(snapshot: WeightSnapshot, manifest: RunManifest) -> None:
        metadata = snapshot.metadata
        for name in (
            "run_id",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "tokenizer_hash",
            "code_signature",
            "model_config_hash",
            "training_config_hash",
        ):
            if getattr(metadata, name) != getattr(manifest, name):
                raise ValueError(f"snapshot metadata does not match manifest {name}")
