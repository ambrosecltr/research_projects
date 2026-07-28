"""Validated, provenance-preserving corpus records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal

SplitName = Literal["train", "validation", "test"]
BlockKind = Literal["document", "verse_document", "poem", "stanza", "paragraph"]
PromptMethod = Literal[
    "title", "author_style", "generic", "theme", "imagery", "paraphrase", "passage"
]
RightsStatus = Literal[
    "public_domain",
    "licensed",
    "permission",
    "user_supplied_personal_copy",
    "synthetic",
    "unknown",
    "denied",
]


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalise_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _optional_text(name: str, value: str | None) -> None:
    if value is not None:
        _required_text(name, value)


def _string_mapping(name: str, value: Mapping[str, str]) -> None:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"{name} must be a mapping of strings to strings")


def _lineage(value: tuple[str, ...], *, name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{name} must be a tuple of non-empty strings")


def _exact_mapping(value: Mapping[str, Any], *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    actual = set(value)
    if actual != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return dict(value)


def _mapping_with_optional_keys(
    value: Mapping[str, Any], *, name: str, required: set[str], optional: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    actual = set(value)
    unknown = actual.difference(required | optional)
    missing = required.difference(actual)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append(f"unknown {sorted(unknown)}")
        if missing:
            detail.append(f"missing {sorted(missing)}")
        raise ValueError(f"{name} has {' and '.join(detail)} keys")
    return dict(value)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Rights and source information retained with every corpus unit."""

    work: str
    author: str
    licence: str
    source: str
    edition: str = ""
    translator: str = ""
    source_locator: str = ""
    page_or_section: str = ""
    retrieved_at: str = ""
    rights_status: RightsStatus = "unknown"
    rights_evidence: str = ""
    rights_notes: str = ""

    def __post_init__(self) -> None:
        for name in ("work", "author", "licence", "source"):
            _required_text(name, getattr(self, name))
        for name in (
            "edition",
            "translator",
            "source_locator",
            "page_or_section",
            "retrieved_at",
            "rights_evidence",
            "rights_notes",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if self.rights_status not in {
            "public_domain",
            "licensed",
            "permission",
            "user_supplied_personal_copy",
            "synthetic",
            "unknown",
            "denied",
        }:
            raise ValueError(f"unsupported rights status: {self.rights_status}")
        if self.rights_status in {"public_domain", "licensed", "permission"}:
            _required_text("rights_evidence", self.rights_evidence)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Provenance:
        return cls(
            **_mapping_with_optional_keys(
                value,
                name="provenance",
                required={"work", "author", "licence", "source"},
                optional={
                    "edition",
                    "translator",
                    "source_locator",
                    "page_or_section",
                    "retrieved_at",
                    "rights_status",
                    "rights_evidence",
                    "rights_notes",
                },
            )
        )


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """A text unit whose parent boundaries remain explicit."""

    block_id: str
    kind: BlockKind
    text: str
    poem_id: str | None = None
    stanza_index: int | None = None
    paragraph_index: int | None = None
    title: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text("block_id", self.block_id)
        _required_text("text", self.text)
        if self.kind not in {"document", "verse_document", "poem", "stanza", "paragraph"}:
            raise ValueError(f"unsupported block kind: {self.kind}")
        if self.kind in {"poem", "stanza"} and not self.poem_id:
            raise ValueError(f"{self.kind} blocks require poem_id")
        if self.kind == "stanza" and self.stanza_index is None:
            raise ValueError("stanza blocks require stanza_index")
        if self.kind == "paragraph" and self.paragraph_index is None:
            raise ValueError("paragraph blocks require paragraph_index")
        for name in ("poem_id", "title"):
            _optional_text(name, getattr(self, name))
        for name in ("stanza_index", "paragraph_index", "start_char", "end_char"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an integer when provided")
        if self.stanza_index is not None and self.stanza_index < 0:
            raise ValueError("stanza_index must be non-negative")
        if self.paragraph_index is not None and self.paragraph_index < 0:
            raise ValueError("paragraph_index must be non-negative")
        _string_mapping("metadata", self.metadata)
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("block character spans require both start_char and end_char")
        if self.start_char is not None and (
            self.start_char < 0 or self.end_char is None or self.end_char <= self.start_char
        ):
            raise ValueError("block character span is invalid")
        object.__setattr__(self, "text", _normalise_text(self.text))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContentBlock:
        return cls(
            **_mapping_with_optional_keys(
                value,
                name="content block",
                required={"block_id", "kind", "text"},
                optional={
                    "poem_id",
                    "stanza_index",
                    "paragraph_index",
                    "title",
                    "start_char",
                    "end_char",
                    "metadata",
                },
            )
        )


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One source document and its lossless structural decomposition."""

    document_id: str
    provenance: Provenance
    text: str
    blocks: tuple[ContentBlock, ...]
    source_path: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    raw_text: str = ""
    transformation_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text("document_id", self.document_id)
        _required_text("text", self.text)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance record")
        if not isinstance(self.blocks, tuple) or any(
            not isinstance(block, ContentBlock) for block in self.blocks
        ):
            raise TypeError("blocks must be a tuple of ContentBlock records")
        if not isinstance(self.source_path, str):
            raise TypeError("source_path must be a string")
        _string_mapping("metadata", self.metadata)
        _lineage(self.transformation_lineage, name="transformation_lineage")
        if not self.blocks:
            raise ValueError("documents require at least one content block")
        ids = [block.block_id for block in self.blocks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate block IDs in {self.document_id}")
        if not isinstance(self.raw_text, str):
            raise TypeError("raw_text must be a string")
        raw_text = self.raw_text or self.text
        _required_text("raw_text", raw_text)
        cleaned_text = _normalise_text(self.text)
        for block in self.blocks:
            if block.start_char is not None:
                assert block.end_char is not None
                if (
                    block.end_char > len(cleaned_text)
                    or cleaned_text[block.start_char : block.end_char] != block.text
                ):
                    raise ValueError(
                        "block span does not exactly identify its text in "
                        f"{self.document_id}: {block.block_id}"
                    )
        object.__setattr__(self, "raw_text", raw_text)
        object.__setattr__(self, "text", cleaned_text)

    @property
    def content_hash(self) -> str:
        payload = f"{self.document_id}\0{self.provenance.work}\0{self.text}".encode()
        return sha256(payload).hexdigest()

    @property
    def raw_content_hash(self) -> str:
        return sha256(self.raw_text.encode()).hexdigest()

    @property
    def cleaned_content_hash(self) -> str:
        return sha256(self.text.encode()).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceDocument:
        data = _exact_mapping(
            value,
            name="source document",
            keys={
                "document_id",
                "provenance",
                "text",
                "blocks",
                "source_path",
                "metadata",
                "raw_text",
                "transformation_lineage",
            },
        )
        if not isinstance(data["provenance"], Mapping):
            raise TypeError("source document provenance must be an object")
        if not isinstance(data["blocks"], list) or any(
            not isinstance(item, Mapping) for item in data["blocks"]
        ):
            raise TypeError("source document blocks must be an array of objects")
        if not isinstance(data["transformation_lineage"], list) or any(
            not isinstance(item, str) or not item for item in data["transformation_lineage"]
        ):
            raise TypeError("source document transformation_lineage must be a string array")
        data["provenance"] = Provenance.from_mapping(data["provenance"])
        data["blocks"] = tuple(ContentBlock.from_mapping(item) for item in data["blocks"])
        data["transformation_lineage"] = tuple(data["transformation_lineage"])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """A supplied conditioning record with an auditable origin; never auto-invented."""

    prompt_id: str
    document_id: str
    prompt: str
    method: PromptMethod
    source_attribution: str
    poem_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_id", "document_id", "prompt", "source_attribution"):
            _required_text(name, getattr(self, name))
        _optional_text("poem_id", self.poem_id)
        if self.method not in {
            "title",
            "author_style",
            "generic",
            "theme",
            "imagery",
            "paraphrase",
            "passage",
        }:
            raise ValueError(f"unsupported prompt method: {self.method}")


@dataclass(frozen=True, slots=True)
class ThoughtRecord:
    thought_id: str
    document_id: str
    text: str
    method: Literal["passage", "paraphrase", "editorial"]
    source_attribution: str

    def __post_init__(self) -> None:
        for name in ("thought_id", "document_id", "text", "source_attribution"):
            _required_text(name, getattr(self, name))
        if self.method not in {"passage", "paraphrase", "editorial"}:
            raise ValueError(f"unsupported thought method: {self.method}")


@dataclass(frozen=True, slots=True)
class CrossDocumentPairing:
    """An auditable philosophy/prompt-to-poetry target relationship."""

    pairing_id: str
    target_document_id: str
    target_block_id: str
    prompt_id: str
    thought_id: str | None
    transformation_lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("pairing_id", "target_document_id", "target_block_id", "prompt_id"):
            _required_text(name, getattr(self, name))
        _optional_text("thought_id", self.thought_id)
        _lineage(self.transformation_lineage, name="transformation_lineage")
        if not self.transformation_lineage:
            raise ValueError("cross-document pairings require transformation lineage")


@dataclass(frozen=True, slots=True)
class ProseNTPExample:
    """Explicit auxiliary raw next-token objective for attributed philosophy prose."""

    example_id: str
    document_id: str
    block_id: str
    text: str
    objective: Literal["auxiliary_prose_ntp"] = "auxiliary_prose_ntp"

    def __post_init__(self) -> None:
        for name in ("example_id", "document_id", "block_id", "text"):
            _required_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PoetryNTPExample:
    """Unconditional next-token verse from a source book, kept distinct from prompts."""

    example_id: str
    document_id: str
    block_id: str
    text: str
    objective: Literal["poetry_ntp"] = "poetry_ntp"

    def __post_init__(self) -> None:
        for name in ("example_id", "document_id", "block_id", "text"):
            _required_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ObjectiveMix:
    """Validated data-token exposure weights for the explicit training objectives."""

    conditional_poetry: float = 1.0
    auxiliary_prose_ntp: float = 0.0
    poetry_ntp: float = 0.0

    def __post_init__(self) -> None:
        values = (self.conditional_poetry, self.auxiliary_prose_ntp, self.poetry_ntp)
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            )
            or sum(values) <= 0
        ):
            raise ValueError("objective weights must be non-negative with a positive total")


@dataclass(frozen=True, slots=True)
class ConditionalExample:
    """A real conditional next-token example, not a prose convention."""

    example_id: str
    document_id: str
    poem_id: str | None
    prompt: str
    poem_target: str
    thought: str | None = None
    prompt_id: str | None = None
    thought_id: str | None = None
    prompt_document_id: str | None = None
    thought_document_id: str | None = None
    pairing_id: str | None = None
    transformation_lineage: tuple[str, ...] = ()
    loss_on_poem_only: bool = True
    split: SplitName | None = None

    def __post_init__(self) -> None:
        for name in ("example_id", "document_id", "prompt", "poem_target"):
            _required_text(name, getattr(self, name))
        _optional_text("poem_id", self.poem_id)
        if self.split is not None and self.split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {self.split}")
        object.__setattr__(self, "prompt", _normalise_text(self.prompt))
        object.__setattr__(self, "poem_target", _normalise_text(self.poem_target))
        if self.thought is not None:
            _required_text("thought", self.thought)
            object.__setattr__(self, "thought", _normalise_text(self.thought))
        if self.prompt_id is not None:
            _required_text("prompt_id", self.prompt_id)
            if self.prompt_document_id is None:
                raise ValueError("prompt_id requires prompt_document_id")
        if self.thought_id is not None:
            _required_text("thought_id", self.thought_id)
        if self.thought_id is not None and self.thought is None:
            raise ValueError("thought_id requires thought text")
        if self.prompt_document_id is not None:
            _required_text("prompt_document_id", self.prompt_document_id)
        if self.thought_document_id is not None:
            _required_text("thought_document_id", self.thought_document_id)
        if self.thought_id is not None and self.thought_document_id is None:
            raise ValueError("thought_id requires thought_document_id")
        if self.pairing_id is not None:
            _required_text("pairing_id", self.pairing_id)
        _lineage(self.transformation_lineage, name="transformation_lineage")
        if not isinstance(self.loss_on_poem_only, bool):
            raise TypeError("loss_on_poem_only must be boolean")

    @property
    def leakage_key(self) -> str:
        return self.poem_id or self.document_id

    @property
    def source_document_ids(self) -> frozenset[str]:
        return frozenset(
            value
            for value in (self.document_id, self.prompt_document_id, self.thought_document_id)
            if value is not None
        )


@dataclass(frozen=True, slots=True)
class TokenSequence:
    """Token IDs plus the exact positions that contribute to language loss."""

    example_id: str
    boundary_key: str
    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    objective: Literal["conditional_poetry", "auxiliary_prose_ntp", "poetry_ntp"] = (
        "conditional_poetry"
    )

    def __post_init__(self) -> None:
        _required_text("example_id", self.example_id)
        _required_text("boundary_key", self.boundary_key)
        if not self.input_ids:
            raise ValueError("token sequences cannot be empty")
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError("input_ids and loss_mask must have equal length")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.input_ids
        ):
            raise ValueError("token IDs must be non-negative")
        if self.objective not in {"conditional_poetry", "auxiliary_prose_ntp", "poetry_ntp"}:
            raise ValueError("token sequence has an unsupported objective")
