from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence, cast

from .hashing import sha256_json

Completeness = Literal["complete", "partial", "endpoint_only", "pending_verification"]
Decision = Literal["approved", "deferred", "rejected", "evaluation_only"]
W0Status = Literal["verified_step0", "reconstructable", "missing", "unknown"]

_COMPLETENESS = {"complete", "partial", "endpoint_only", "pending_verification"}
_DECISIONS = {"approved", "deferred", "rejected", "evaluation_only"}
_W0_STATUS = {"verified_step0", "reconstructable", "missing", "unknown"}
_REVEALED_HIDDEN_LIVES = {"pythia-14m-seed9"}


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _string_tuple(value: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    result = tuple(_nonempty(item, name=f"{name} item") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class SourceSize:
    label: str
    parameter_count: int
    life_ids: tuple[str, ...]
    checkpoint_count_per_life: int
    source_dtype_bytes: int = 4

    def __post_init__(self) -> None:
        _nonempty(self.label, name="source size label")
        _positive_int(self.parameter_count, name=f"{self.label}.parameter_count")
        _string_tuple(self.life_ids, name=f"{self.label}.life_ids")
        _positive_int(
            self.checkpoint_count_per_life,
            name=f"{self.label}.checkpoint_count_per_life",
        )
        if self.source_dtype_bytes not in {1, 2, 4, 8}:
            raise ValueError("source_dtype_bytes must be a standard scalar width")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceSize:
        data = dict(value)
        data["life_ids"] = _string_tuple(data.get("life_ids"), name="life_ids")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["life_ids"] = list(self.life_ids)
        return data

    @property
    def life_count(self) -> int:
        return len(self.life_ids)

    def estimated_endpoint_pair_bytes(self) -> int:
        return self.life_count * 2 * self.parameter_count * self.source_dtype_bytes

    def estimated_all_checkpoint_bytes(self) -> int:
        return (
            self.life_count
            * self.checkpoint_count_per_life
            * self.parameter_count
            * self.source_dtype_bytes
        )


@dataclass(frozen=True)
class SourceCandidate:
    source_id: str
    organization: str
    repository_pattern: str
    architecture_family: str
    licence: str
    completeness: Completeness
    decision: Decision
    priority: int
    w0_status: W0Status
    final_endpoint_available: bool
    dataset_content_available: bool
    exact_data_order_available: bool
    tokenizer_available: bool
    complete_recipe_available: bool
    provenance_available: bool
    intermediate_checkpoints_available: bool
    sizes: tuple[SourceSize, ...]
    approved_materialization: tuple[str, ...]
    blocked_by: tuple[str, ...]
    source_urls: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.source_id, name="source_id")
        _nonempty(self.organization, name=f"{self.source_id}.organization")
        _nonempty(self.repository_pattern, name=f"{self.source_id}.repository_pattern")
        _nonempty(self.architecture_family, name=f"{self.source_id}.architecture_family")
        _nonempty(self.licence, name=f"{self.source_id}.licence")
        if self.completeness not in _COMPLETENESS:
            raise ValueError(f"unsupported completeness: {self.completeness!r}")
        if self.decision not in _DECISIONS:
            raise ValueError(f"unsupported source decision: {self.decision!r}")
        if self.w0_status not in _W0_STATUS:
            raise ValueError(f"unsupported W0 status: {self.w0_status!r}")
        _non_negative_int(self.priority, name=f"{self.source_id}.priority")
        if not self.sizes:
            raise ValueError(f"{self.source_id} must declare at least one size")
        _string_tuple(
            self.approved_materialization,
            name=f"{self.source_id}.approved_materialization",
            allow_empty=self.decision != "approved",
        )
        _string_tuple(
            self.blocked_by,
            name=f"{self.source_id}.blocked_by",
            allow_empty=True,
        )
        _string_tuple(self.source_urls, name=f"{self.source_id}.source_urls")
        _string_tuple(self.notes, name=f"{self.source_id}.notes", allow_empty=True)

        if self.completeness == "complete":
            if self.w0_status not in {"verified_step0", "reconstructable"}:
                raise ValueError(f"complete source {self.source_id} lacks a valid W0")
            for field_name in (
                "final_endpoint_available",
                "dataset_content_available",
                "exact_data_order_available",
                "tokenizer_available",
                "complete_recipe_available",
                "provenance_available",
            ):
                if not getattr(self, field_name):
                    raise ValueError(f"complete source {self.source_id} lacks {field_name}")
        if self.decision == "approved":
            if self.completeness != "complete":
                raise ValueError("only complete sources may be approved for compiler targets")
            if self.blocked_by:
                raise ValueError("approved source cannot retain blockers")
        if self.decision in {"deferred", "rejected"} and not self.blocked_by:
            raise ValueError(f"{self.decision} source must explain its blockers")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceCandidate:
        data = dict(value)
        raw_sizes = data.get("sizes")
        if not isinstance(raw_sizes, Sequence) or isinstance(raw_sizes, (str, bytes)):
            raise TypeError("source sizes must be an array")
        data["sizes"] = tuple(SourceSize.from_dict(cast(Mapping[str, Any], item)) for item in raw_sizes)
        for name in (
            "approved_materialization",
            "blocked_by",
            "source_urls",
            "notes",
        ):
            data[name] = _string_tuple(
                data.get(name, []),
                name=f"source.{name}",
                allow_empty=name in {"blocked_by", "notes"} or data.get("decision") != "approved",
            )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sizes"] = [size.to_dict() for size in self.sizes]
        for name in (
            "approved_materialization",
            "blocked_by",
            "source_urls",
            "notes",
        ):
            data[name] = list(getattr(self, name))
        return data

    @property
    def life_count(self) -> int:
        return sum(size.life_count for size in self.sizes)

    def estimated_endpoint_pair_bytes(self) -> int:
        return sum(size.estimated_endpoint_pair_bytes() for size in self.sizes)

    def estimated_all_checkpoint_bytes(self) -> int:
        return sum(size.estimated_all_checkpoint_bytes() for size in self.sizes)


@dataclass(frozen=True)
class SplitPlan:
    training: tuple[str, ...]
    development: tuple[str, ...]
    hidden: tuple[str, ...]
    quarantined: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("training", "development", "hidden", "quarantined"):
            _string_tuple(getattr(self, name), name=f"split.{name}", allow_empty=True)
        all_ids = [*self.training, *self.development, *self.hidden, *self.quarantined]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("a model life appears in more than one split")
        active = {*self.training, *self.development, *self.hidden}
        revealed = active & _REVEALED_HIDDEN_LIVES
        if revealed:
            raise ValueError(
                f"revealed hidden lives must remain quarantined: {sorted(revealed)}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitPlan:
        return cls(
            training=_string_tuple(value.get("training", []), name="training", allow_empty=True),
            development=_string_tuple(
                value.get("development", []), name="development", allow_empty=True
            ),
            hidden=_string_tuple(value.get("hidden", []), name="hidden", allow_empty=True),
            quarantined=_string_tuple(
                value.get("quarantined", []), name="quarantined", allow_empty=True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": list(self.training),
            "development": list(self.development),
            "hidden": list(self.hidden),
            "quarantined": list(self.quarantined),
        }


@dataclass(frozen=True)
class SourceAuditManifest:
    candidates: tuple[SourceCandidate, ...]
    split_plan: SplitPlan
    assumptions: dict[str, Any]
    format: str = "GENOME_SOURCE_AUDIT"
    version: str = "0.1.0"

    def __post_init__(self) -> None:
        if self.format != "GENOME_SOURCE_AUDIT" or self.version != "0.1.0":
            raise ValueError("unsupported source-audit format")
        if not self.candidates:
            raise ValueError("source audit requires candidates")
        ids = [candidate.source_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("source audit contains duplicate source IDs")
        if not isinstance(self.assumptions, dict):
            raise TypeError("source-audit assumptions must be an object")

        known_lives = {
            life_id
            for candidate in self.candidates
            for size in candidate.sizes
            for life_id in size.life_ids
        }
        split_lives = {
            *self.split_plan.training,
            *self.split_plan.development,
            *self.split_plan.hidden,
            *self.split_plan.quarantined,
        }
        unknown = split_lives - known_lives
        if unknown:
            raise ValueError(f"split plan references unknown lives: {sorted(unknown)}")
        revealed_lives = known_lives & _REVEALED_HIDDEN_LIVES
        missing_quarantine = revealed_lives - set(self.split_plan.quarantined)
        if missing_quarantine:
            raise ValueError(
                "revealed hidden lives must be listed in the quarantine split: "
                f"{sorted(missing_quarantine)}"
            )
        approved_lives = {
            life_id
            for candidate in self.candidates
            if candidate.decision == "approved"
            for size in candidate.sizes
            for life_id in size.life_ids
        }
        active = {
            *self.split_plan.training,
            *self.split_plan.development,
            *self.split_plan.hidden,
        }
        if not active.issubset(approved_lives):
            raise ValueError("active split contains a life from a non-approved source")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceAuditManifest:
        expected = {"format", "version", "candidates", "split_plan", "assumptions"}
        if set(value) != expected:
            raise ValueError(
                f"source-audit fields differ; missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise TypeError("source-audit candidates must be an array")
        return cls(
            format=str(value["format"]),
            version=str(value["version"]),
            candidates=tuple(
                SourceCandidate.from_dict(cast(Mapping[str, Any], item))
                for item in raw_candidates
            ),
            split_plan=SplitPlan.from_dict(cast(Mapping[str, Any], value["split_plan"])),
            assumptions=dict(cast(Mapping[str, Any], value["assumptions"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "split_plan": self.split_plan.to_dict(),
            "assumptions": dict(self.assumptions),
        }

    @property
    def content_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def estimated_approved_endpoint_pair_bytes(self) -> int:
        """Estimate W0/WT storage from parameter counts and declared scalar widths."""

        active_ids = {
            *self.split_plan.training,
            *self.split_plan.development,
            *self.split_plan.hidden,
        }
        total = 0
        for candidate in self.candidates:
            if candidate.decision != "approved":
                continue
            for size in candidate.sizes:
                selected_count = sum(life_id in active_ids for life_id in size.life_ids)
                total += selected_count * 2 * size.parameter_count * size.source_dtype_bytes
        return total

    def estimated_maximum_catalog_bytes(self) -> int:
        """Estimate full-catalog storage; download receipts must report actual LFS bytes."""

        return sum(candidate.estimated_all_checkpoint_bytes() for candidate in self.candidates)
