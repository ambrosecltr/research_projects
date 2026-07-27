"""Strict, JSON-safe schema for endpoint-informed geometry evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from poetry50m.trajectory.preparation import state_dict_hash
from poetry50m.trajectory.types import WeightSnapshot

SCHEMA_VERSION = "poetry50m.endpoint_geometry.v1"
METHOD = "tensorwise_streaming_float64_endpoint_geometry"
EVIDENCE_LABEL = (
    "offline_teacher_analysis_only_endpoint_knowledge_forbidden_for_sealed_r2_verification"
)


def finite_number(value: object, *, name: str) -> float:
    """Validate a finite JSON number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def finite_or_none(value: object, *, name: str) -> float | None:
    return None if value is None else finite_number(value, name=name)


def nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def optional_string(value: object, *, name: str) -> str | None:
    return None if value is None else required_string(value, name=name)


def mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def exact_fields(value: Mapping[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")


def _array(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


@dataclass(frozen=True, slots=True)
class FormulaDefinition:
    """One fixed formula represented by every geometry report."""

    name: str
    expression: str

    def __post_init__(self) -> None:
        required_string(self.name, name="formula name")
        required_string(self.expression, name="formula expression")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FormulaDefinition:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="formula definition")
        return cls(
            name=required_string(value["name"], name="formula name"),
            expression=required_string(value["expression"], name="formula expression"),
        )


FORMULA_DEFINITIONS = (
    FormulaDefinition("early_direction", "e = W_early - W_initial"),
    FormulaDefinition("remaining_direction", "r = W_endpoint - W_early"),
    FormulaDefinition("endpoint_direction", "f = W_endpoint - W_initial = e + r"),
    FormulaDefinition("endpoint_scale_from_initial", "argmin_a ||f - a e||_2 = <e,f> / ||e||_2^2"),
    FormulaDefinition("remaining_scale_from_early", "argmin_b ||r - b e||_2 = <e,r> / ||e||_2^2"),
    FormulaDefinition("endpoint_projection_progress_fraction", "<e,f> / ||f||_2^2"),
    FormulaDefinition("interval_velocity", "v_i = (W_{i+1} - W_i) / (step_{i+1} - step_i)"),
    FormulaDefinition("interval_acceleration", "a_i = (v_{i+1} - v_i) / ((dt_i + dt_{i+1}) / 2)"),
)


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    """Exact identity of a snapshot used by endpoint analysis."""

    checkpoint_id: str
    step: int
    run_id: str
    initialization_id: str
    data_order_id: str
    architecture_signature: str
    corpus_signature: str
    model_config_hash: str
    tokenizer_hash: str
    code_signature: str
    training_config_hash: str
    state_dict_hash: str
    coordinate_signature: str
    source_path: str | None

    def __post_init__(self) -> None:
        for field in (
            "checkpoint_id",
            "run_id",
            "initialization_id",
            "data_order_id",
            "architecture_signature",
            "corpus_signature",
            "model_config_hash",
            "tokenizer_hash",
            "code_signature",
            "training_config_hash",
            "state_dict_hash",
            "coordinate_signature",
        ):
            required_string(getattr(self, field), name=field)
        nonnegative_integer(self.step, name="step")
        optional_string(self.source_path, name="source_path")

    @classmethod
    def from_snapshot(cls, snapshot: WeightSnapshot) -> SnapshotProvenance:
        metadata = snapshot.metadata
        return cls(
            checkpoint_id=metadata.checkpoint_id,
            step=metadata.step,
            run_id=metadata.run_id,
            initialization_id=metadata.initialization_id,
            data_order_id=metadata.data_order_id,
            architecture_signature=metadata.architecture_signature,
            corpus_signature=metadata.corpus_signature,
            model_config_hash=metadata.model_config_hash,
            tokenizer_hash=metadata.tokenizer_hash,
            code_signature=metadata.code_signature,
            training_config_hash=metadata.training_config_hash,
            state_dict_hash=state_dict_hash(snapshot.state_dict),
            coordinate_signature=snapshot.coordinate_signature,
            source_path=str(snapshot.source_path) if snapshot.source_path is not None else None,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SnapshotProvenance:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="snapshot provenance")
        return cls(
            checkpoint_id=required_string(value["checkpoint_id"], name="checkpoint_id"),
            step=nonnegative_integer(value["step"], name="step"),
            run_id=required_string(value["run_id"], name="run_id"),
            initialization_id=required_string(value["initialization_id"], name="initialization_id"),
            data_order_id=required_string(value["data_order_id"], name="data_order_id"),
            architecture_signature=required_string(
                value["architecture_signature"], name="architecture_signature"
            ),
            corpus_signature=required_string(value["corpus_signature"], name="corpus_signature"),
            model_config_hash=required_string(value["model_config_hash"], name="model_config_hash"),
            tokenizer_hash=required_string(value["tokenizer_hash"], name="tokenizer_hash"),
            code_signature=required_string(value["code_signature"], name="code_signature"),
            training_config_hash=required_string(
                value["training_config_hash"], name="training_config_hash"
            ),
            state_dict_hash=required_string(value["state_dict_hash"], name="state_dict_hash"),
            coordinate_signature=required_string(
                value["coordinate_signature"], name="coordinate_signature"
            ),
            source_path=optional_string(value["source_path"], name="source_path"),
        )


@dataclass(frozen=True, slots=True)
class ConsecutiveDeltaGeometry:
    start_step: int
    middle_step: int
    end_step: int
    velocity_cosine: float | None
    acceleration_norm: float

    def __post_init__(self) -> None:
        if not self.start_step < self.middle_step < self.end_step:
            raise ValueError("consecutive delta steps must be strictly increasing")
        cosine = finite_or_none(self.velocity_cosine, name="velocity_cosine")
        if cosine is not None and not -1.0 <= cosine <= 1.0:
            raise ValueError("velocity_cosine must lie in [-1, 1]")
        if finite_number(self.acceleration_norm, name="acceleration_norm") < 0.0:
            raise ValueError("acceleration_norm must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ConsecutiveDeltaGeometry:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="consecutive delta geometry")
        return cls(
            start_step=nonnegative_integer(value["start_step"], name="start_step"),
            middle_step=nonnegative_integer(value["middle_step"], name="middle_step"),
            end_step=nonnegative_integer(value["end_step"], name="end_step"),
            velocity_cosine=finite_or_none(value["velocity_cosine"], name="velocity_cosine"),
            acceleration_norm=finite_number(value["acceleration_norm"], name="acceleration_norm"),
        )


@dataclass(frozen=True, slots=True)
class TurningSummary:
    intervals: tuple[ConsecutiveDeltaGeometry, ...]
    mean_velocity_cosine: float | None
    mean_turning_angle_radians: float | None
    mean_acceleration_norm: float

    def __post_init__(self) -> None:
        if not self.intervals:
            raise ValueError("turning summary requires at least one interval pair")
        cosine = finite_or_none(self.mean_velocity_cosine, name="mean_velocity_cosine")
        angle = finite_or_none(self.mean_turning_angle_radians, name="mean_turning_angle_radians")
        if cosine is not None and not -1.0 <= cosine <= 1.0:
            raise ValueError("mean_velocity_cosine must lie in [-1, 1]")
        if angle is not None and not 0.0 <= angle <= math.pi:
            raise ValueError("mean_turning_angle_radians must lie in [0, pi]")
        if finite_number(self.mean_acceleration_norm, name="mean_acceleration_norm") < 0.0:
            raise ValueError("mean_acceleration_norm must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TurningSummary:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="turning summary")
        return cls(
            intervals=tuple(
                ConsecutiveDeltaGeometry.from_mapping(mapping(item, name="turning interval"))
                for item in _array(value["intervals"], name="turning intervals")
            ),
            mean_velocity_cosine=finite_or_none(
                value["mean_velocity_cosine"], name="mean_velocity_cosine"
            ),
            mean_turning_angle_radians=finite_or_none(
                value["mean_turning_angle_radians"], name="mean_turning_angle_radians"
            ),
            mean_acceleration_norm=finite_number(
                value["mean_acceleration_norm"], name="mean_acceleration_norm"
            ),
        )


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    name: str
    scope: str
    tensor_count: int
    parameter_count: int
    initial_to_early_norm: float
    early_to_endpoint_norm: float
    initial_to_endpoint_norm: float
    cosine_early_remaining: float | None
    cosine_early_endpoint: float | None
    endpoint_scale_from_initial: float | None
    remaining_scale_from_early: float | None
    endpoint_prediction_residual_norm: float | None
    endpoint_prediction_residual_ratio: float | None
    endpoint_projection_progress_fraction: float | None
    turning: TurningSummary | None

    def __post_init__(self) -> None:
        required_string(self.name, name="metric name")
        if self.scope not in {"global", "module_prefix", "tensor"}:
            raise ValueError("metric scope must be global, module_prefix, or tensor")
        if nonnegative_integer(self.tensor_count, name="tensor_count") < 1:
            raise ValueError("metrics must cover at least one floating parameter")
        if nonnegative_integer(self.parameter_count, name="parameter_count") < 1:
            raise ValueError("metrics must cover at least one floating parameter")
        for field in (
            "initial_to_early_norm",
            "early_to_endpoint_norm",
            "initial_to_endpoint_norm",
        ):
            if finite_number(getattr(self, field), name=field) < 0.0:
                raise ValueError(f"{field} must be non-negative")
        for field in (
            "cosine_early_remaining",
            "cosine_early_endpoint",
            "endpoint_scale_from_initial",
            "remaining_scale_from_early",
            "endpoint_prediction_residual_norm",
            "endpoint_prediction_residual_ratio",
            "endpoint_projection_progress_fraction",
        ):
            value = finite_or_none(getattr(self, field), name=field)
            if field.startswith("cosine") and value is not None and not -1.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [-1, 1]")
            if (
                field.endswith(("residual_norm", "residual_ratio"))
                and value is not None
                and value < 0.0
            ):
                raise ValueError(f"{field} must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GeometryMetrics:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="geometry metrics")
        turning = value["turning"]
        return cls(
            name=required_string(value["name"], name="metric name"),
            scope=required_string(value["scope"], name="metric scope"),
            tensor_count=nonnegative_integer(value["tensor_count"], name="tensor_count"),
            parameter_count=nonnegative_integer(value["parameter_count"], name="parameter_count"),
            initial_to_early_norm=finite_number(
                value["initial_to_early_norm"], name="initial_to_early_norm"
            ),
            early_to_endpoint_norm=finite_number(
                value["early_to_endpoint_norm"], name="early_to_endpoint_norm"
            ),
            initial_to_endpoint_norm=finite_number(
                value["initial_to_endpoint_norm"], name="initial_to_endpoint_norm"
            ),
            cosine_early_remaining=finite_or_none(
                value["cosine_early_remaining"], name="cosine_early_remaining"
            ),
            cosine_early_endpoint=finite_or_none(
                value["cosine_early_endpoint"], name="cosine_early_endpoint"
            ),
            endpoint_scale_from_initial=finite_or_none(
                value["endpoint_scale_from_initial"], name="endpoint_scale_from_initial"
            ),
            remaining_scale_from_early=finite_or_none(
                value["remaining_scale_from_early"], name="remaining_scale_from_early"
            ),
            endpoint_prediction_residual_norm=finite_or_none(
                value["endpoint_prediction_residual_norm"], name="endpoint_prediction_residual_norm"
            ),
            endpoint_prediction_residual_ratio=finite_or_none(
                value["endpoint_prediction_residual_ratio"],
                name="endpoint_prediction_residual_ratio",
            ),
            endpoint_projection_progress_fraction=finite_or_none(
                value["endpoint_projection_progress_fraction"],
                name="endpoint_projection_progress_fraction",
            ),
            turning=None
            if turning is None
            else TurningSummary.from_mapping(mapping(turning, name="turning summary")),
        )


@dataclass(frozen=True, slots=True)
class AngularTensorMetrics:
    name: str
    normalization_axis: int
    vector_count: int
    initial_to_early_mean_angle_radians: float | None
    early_to_endpoint_mean_angle_radians: float | None
    initial_to_endpoint_mean_angle_radians: float | None

    def __post_init__(self) -> None:
        required_string(self.name, name="angular metric name")
        if isinstance(self.normalization_axis, bool) or not isinstance(
            self.normalization_axis, int
        ):
            raise TypeError("normalization_axis must be an integer")
        if nonnegative_integer(self.vector_count, name="angular vector_count") < 1:
            raise ValueError("angular metrics require at least one vector")
        for field in (
            "initial_to_early_mean_angle_radians",
            "early_to_endpoint_mean_angle_radians",
            "initial_to_endpoint_mean_angle_radians",
        ):
            value = finite_or_none(getattr(self, field), name=field)
            if value is not None and not 0.0 <= value <= math.pi:
                raise ValueError(f"{field} must lie in [0, pi]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AngularTensorMetrics:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="angular tensor metrics")
        return cls(
            name=required_string(value["name"], name="angular metric name"),
            normalization_axis=integer(value["normalization_axis"], name="normalization_axis"),
            vector_count=nonnegative_integer(value["vector_count"], name="angular vector_count"),
            initial_to_early_mean_angle_radians=finite_or_none(
                value["initial_to_early_mean_angle_radians"],
                name="initial_to_early_mean_angle_radians",
            ),
            early_to_endpoint_mean_angle_radians=finite_or_none(
                value["early_to_endpoint_mean_angle_radians"],
                name="early_to_endpoint_mean_angle_radians",
            ),
            initial_to_endpoint_mean_angle_radians=finite_or_none(
                value["initial_to_endpoint_mean_angle_radians"],
                name="initial_to_endpoint_mean_angle_radians",
            ),
        )


@dataclass(frozen=True, slots=True)
class AngularGeometrySummary:
    available: bool
    absence_reason: str | None
    tensors: tuple[AngularTensorMetrics, ...]
    initial_to_early_mean_angle_radians: float | None
    early_to_endpoint_mean_angle_radians: float | None
    initial_to_endpoint_mean_angle_radians: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("angular availability must be boolean")
        if self.available == (self.absence_reason is not None):
            raise ValueError("angular availability and absence_reason disagree")
        if self.available != bool(self.tensors):
            raise ValueError(
                "angular tensors must be present exactly when angular geometry is available"
            )
        optional_string(self.absence_reason, name="angular absence_reason")
        for field in (
            "initial_to_early_mean_angle_radians",
            "early_to_endpoint_mean_angle_radians",
            "initial_to_endpoint_mean_angle_radians",
        ):
            value = finite_or_none(getattr(self, field), name=field)
            if value is not None and not 0.0 <= value <= math.pi:
                raise ValueError(f"{field} must lie in [0, pi]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AngularGeometrySummary:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="angular geometry")
        available = value["available"]
        if not isinstance(available, bool):
            raise TypeError("angular availability must be boolean")
        return cls(
            available=available,
            absence_reason=optional_string(value["absence_reason"], name="angular absence_reason"),
            tensors=tuple(
                AngularTensorMetrics.from_mapping(mapping(item, name="angular tensor metric"))
                for item in _array(value["tensors"], name="angular tensors")
            ),
            initial_to_early_mean_angle_radians=finite_or_none(
                value["initial_to_early_mean_angle_radians"],
                name="initial_to_early_mean_angle_radians",
            ),
            early_to_endpoint_mean_angle_radians=finite_or_none(
                value["early_to_endpoint_mean_angle_radians"],
                name="early_to_endpoint_mean_angle_radians",
            ),
            initial_to_endpoint_mean_angle_radians=finite_or_none(
                value["initial_to_endpoint_mean_angle_radians"],
                name="initial_to_endpoint_mean_angle_radians",
            ),
        )


@dataclass(frozen=True, slots=True)
class EndpointGeometryReport:
    """Strict endpoint-informed, teacher-only analysis evidence."""

    schema_version: str
    method: str
    endpoint_informed: bool
    evidence_label: str
    snapshots: tuple[SnapshotProvenance, ...]
    formula_definitions: tuple[FormulaDefinition, ...]
    excluded_non_floating_tensor_names: tuple[str, ...]
    metrics: tuple[GeometryMetrics, ...]
    angular_geometry: AngularGeometrySummary

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported endpoint geometry schema {self.schema_version!r}")
        if self.method != METHOD:
            raise ValueError(f"unsupported endpoint geometry method {self.method!r}")
        if self.endpoint_informed is not True:
            raise ValueError("endpoint geometry reports must be explicitly endpoint-informed")
        if self.evidence_label != EVIDENCE_LABEL:
            raise ValueError("endpoint geometry reports require the sealed-R2 evidence warning")
        if len(self.snapshots) < 3:
            raise ValueError("endpoint geometry requires initial, early, and endpoint snapshots")
        steps = tuple(snapshot.step for snapshot in self.snapshots)
        if any(left >= right for left, right in zip(steps[:-1], steps[1:], strict=True)):
            raise ValueError("report snapshot steps must be strictly increasing")
        if self.formula_definitions != FORMULA_DEFINITIONS:
            raise ValueError("endpoint geometry report formula definitions are fixed by the schema")
        excluded = self.excluded_non_floating_tensor_names
        if tuple(sorted(excluded)) != excluded or len(set(excluded)) != len(excluded):
            raise ValueError("excluded tensor names must be sorted and unique")
        if not self.metrics or (self.metrics[0].scope, self.metrics[0].name) != (
            "global",
            "global",
        ):
            raise ValueError("the first metric must be the global aggregate")
        sorted_metrics = tuple(sorted((metric.scope, metric.name) for metric in self.metrics[1:]))
        if sorted_metrics != tuple((metric.scope, metric.name) for metric in self.metrics[1:]):
            raise ValueError("non-global metrics must be deterministically sorted")

    def to_mapping(self) -> dict[str, object]:
        return dict(mapping(json.loads(self.to_json()), name="endpoint geometry report"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> EndpointGeometryReport:
        def reject_constant(value: str) -> object:
            raise ValueError(f"endpoint geometry JSON contains invalid numeric constant {value!r}")

        return cls.from_mapping(
            mapping(json.loads(payload, parse_constant=reject_constant), name="report")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EndpointGeometryReport:
        exact_fields(value, frozenset(cls.__dataclass_fields__), name="endpoint geometry report")
        endpoint_informed = value["endpoint_informed"]
        if not isinstance(endpoint_informed, bool):
            raise TypeError("endpoint_informed must be boolean")
        excluded = _array(value["excluded_non_floating_tensor_names"], name="excluded tensor names")
        if any(not isinstance(name, str) for name in excluded):
            raise TypeError("excluded_non_floating_tensor_names must be an array of strings")
        return cls(
            schema_version=required_string(value["schema_version"], name="schema_version"),
            method=required_string(value["method"], name="method"),
            endpoint_informed=endpoint_informed,
            evidence_label=required_string(value["evidence_label"], name="evidence_label"),
            snapshots=tuple(
                SnapshotProvenance.from_mapping(mapping(item, name="snapshot provenance"))
                for item in _array(value["snapshots"], name="report snapshots")
            ),
            formula_definitions=tuple(
                FormulaDefinition.from_mapping(mapping(item, name="formula definition"))
                for item in _array(value["formula_definitions"], name="formula definitions")
            ),
            excluded_non_floating_tensor_names=tuple(
                name for name in excluded if isinstance(name, str)
            ),
            metrics=tuple(
                GeometryMetrics.from_mapping(mapping(item, name="geometry metrics"))
                for item in _array(value["metrics"], name="report metrics")
            ),
            angular_geometry=AngularGeometrySummary.from_mapping(
                mapping(value["angular_geometry"], name="angular geometry")
            ),
        )
