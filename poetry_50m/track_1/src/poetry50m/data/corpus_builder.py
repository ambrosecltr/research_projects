"""Build the pinned Track 1 corpus from a verified local acquisition."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import cast

import pyarrow.parquet as parquet  # type: ignore[import-untyped]

from poetry50m.config import canonical_json, file_hash, load_mapping

from .artifacts import write_pairings, write_prompt_records, write_thought_records
from .hf_sources import AcquisitionReceipt
from .loaders import write_manifest
from .schema import ContentBlock, PromptMethod, PromptRecord, Provenance, SourceDocument

_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_PROMPT_TITLE_CHARACTERS = 120
_REMOVABLE_CONTROL_CHARACTERS = (frozenset(range(32)) | frozenset(range(127, 160))).difference(
    {9, 10}
)
_APPROVED_ACQUISITION_CONFIG_SHA256 = (
    "e555f0b1054e91fffe08d85e8a764c663959af0a71107bbb74884aef461dcb20"
)
_EDITORIAL_NOTE_PREFIXES = (
    "editor's note",
    "editor’s note",
    "from the handwriting",
    "transcriber's note",
    "transcriber’s note",
)
_PURE_NOTES_PREFIX = re.compile(r"\A\s*notes?\s*:", re.IGNORECASE)
_TITLE_LABELLED_PROSE = re.compile(r"\bprose(?:\s+fable)?\b", re.IGNORECASE)
_NUMERIC_OR_ROMAN_TITLE = re.compile(r"[\s\dIVXLCDM.,:;()\-—_]+", re.IGNORECASE)
_EDITORIAL_TITLE = re.compile(
    r"(?:\A|[:.;—-]\s*)(?:advertisement|editor(?:'s|’s)? note|introduction|"
    r"notes?|preface|transcriber(?:'s|’s)? note)\b",
    re.IGNORECASE,
)
_UNUSABLE_AUTHORS = frozenset(
    {
        "anon",
        "anon.",
        "anonymous",
        "editor",
        "n/a",
        "nobody",
        "unknown",
        "unknown author",
        "various",
    }
)
_LEADING_MARKDOWN_INTRODUCTION = re.compile(
    r"\A##(?!#)[ \t]+Introduction[ \t]*(?:\n|\Z)",
    re.IGNORECASE,
)
_MARKDOWN_LEVEL_TWO_HEADING = re.compile(r"^##(?!#)[ \t]+", re.MULTILINE)

_ExpectedArtifact = tuple[str, str, int]
_ExpectedSource = tuple[str, str, str, Mapping[str, _ExpectedArtifact]]

_EXPECTED_SOURCES: Mapping[str, _ExpectedSource] = {
    "public_domain_poetry": (
        "DanFosing/public-domain-poetry",
        "84a87909d09ff0c3ae040c4e0af25a6344d96531",
        "poem_records_json",
        {
            "public_domain_poetry/poems.json": (
                "poems.json",
                "172cd2c5d953c7023390a8d1f337d023d7fbb2b925df0a66d0221f30c6adc308",
                94_209_335,
            )
        },
    ),
    "poetry_greats_public_domain": (
        "yoonholee/poetry-greats-public-domain",
        "3201e250462905a0c8f6134e124382ac96586dc9",
        "poem_records_parquet",
        {
            "poetry_greats_public_domain/data/train-00000-of-00001.parquet": (
                "data/train-00000-of-00001.parquet",
                "9920f9c5919dbb3a68c258f2597cbb7dfaa1ff9549d992c410cc5910cffc8725",
                4_362_856,
            )
        },
    ),
    "standardebooks": (
        "Nelathan/standardebooks",
        "a2bafeeff73d3ff553e29dffc54f07772472b409",
        "ebook_records_parquet",
        {
            "standardebooks/data/train-00000-of-00002.parquet": (
                "data/train-00000-of-00002.parquet",
                "09a748932912092d00dffae50f49c91f8d9c122cebb2c8df3b32a63a69c44cf6",
                219_906_476,
            ),
            "standardebooks/data/train-00001-of-00002.parquet": (
                "data/train-00001-of-00002.parquet",
                "2fdc4af4cf4b885523adb952c5df5ae03cc133296009f3f46fb57004bf16f1ee",
                210_331_958,
            ),
        },
    ),
}


@dataclass(frozen=True, slots=True)
class _LocatedArtifact:
    source_path: str
    local_path: str
    sha256: str
    size_bytes: int
    path: Path


@dataclass(frozen=True, slots=True)
class _LocatedSource:
    source_id: str
    repository: str
    revision: str
    artifact_kind: str
    artifacts: tuple[_LocatedArtifact, ...]


@dataclass(frozen=True, slots=True, order=True)
class SelectedWork:
    title: str
    author: str


@dataclass(frozen=True, slots=True)
class StandardEbooksSelection:
    works: tuple[SelectedWork, ...]


@dataclass(frozen=True, slots=True)
class CorpusBuildArtifact:
    root: Path
    report: Mapping[str, object]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Origin:
    source_id: str
    repository: str
    revision: str
    artifact_source_path: str
    artifact_local_path: str
    artifact_sha256: str
    row_index: int
    title: str | None
    author: str
    source_locator: str
    edition: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PoemCandidate:
    raw_text: str
    text: str
    origin: _Origin
    removed_control_characters: int


@dataclass(frozen=True, slots=True)
class _CleanedText:
    text: str
    removed_control_characters: int


@dataclass(frozen=True, slots=True)
class _Rejection:
    source_id: str
    artifact_local_path: str
    row_index: int
    reason: str


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact_object(
    value: object, *, name: str, expected: frozenset[str] | set[str]
) -> dict[str, object]:
    result = _object(value, name=name)
    if set(result) != set(expected):
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    return result


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _normalise_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def load_standard_ebooks_selection(path: Path) -> StandardEbooksSelection:
    """Load the exact title-and-author allowlist used for prose inclusion."""
    value = _exact_object(
        load_mapping(path),
        name="Standard Ebooks selection",
        expected={"format_version", "standard_ebooks"},
    )
    if value["format_version"] != 1 or isinstance(value["format_version"], bool):
        raise ValueError("Standard Ebooks selection format_version must be 1")
    standard_ebooks = _exact_object(
        value["standard_ebooks"],
        name="standard_ebooks",
        expected={"works"},
    )
    raw_works = standard_ebooks["works"]
    if not isinstance(raw_works, list) or not raw_works:
        raise ValueError("standard_ebooks.works must be a non-empty array")
    works: list[SelectedWork] = []
    for index, raw_work in enumerate(cast(list[object], raw_works)):
        work = _exact_object(
            raw_work,
            name=f"standard_ebooks.works[{index}]",
            expected={"title", "author"},
        )
        title = _normalise_identity(
            _non_empty_string(work["title"], name=f"standard_ebooks.works[{index}].title")
        )
        author = _normalise_identity(
            _non_empty_string(work["author"], name=f"standard_ebooks.works[{index}].author")
        )
        works.append(SelectedWork(title, author))
    if len(set(works)) != len(works):
        raise ValueError("standard_ebooks.works contains a duplicate title-and-author pair")
    return StandardEbooksSelection(tuple(works))


def _catalog_sources(
    acquisition_directory: Path, catalog_path: Path
) -> tuple[str, tuple[_LocatedSource, ...]]:
    catalog_text = catalog_path.read_text(encoding="utf-8")
    receipt = AcquisitionReceipt.from_mapping(json.loads(catalog_text))
    if catalog_text != canonical_json(receipt.to_mapping()) + "\n":
        raise ValueError("acquisition catalog must be the canonical acquisition receipt")
    if receipt.config_sha256 != _APPROVED_ACQUISITION_CONFIG_SHA256:
        raise ValueError("acquisition catalog is not bound to the approved source configuration")
    acquisition_root = acquisition_directory.resolve()
    sources: list[_LocatedSource] = []
    seen_source_ids: set[str] = set()
    seen_local_paths: set[str] = set()
    for source in receipt.sources:
        source_id = source.source_id
        repository = source.repository
        revision = source.revision
        artifact_kind = source.artifact_kind
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate acquisition source_id {source_id!r}")
        if source_id not in _EXPECTED_SOURCES:
            raise ValueError(f"unexpected acquisition source_id {source_id!r}")
        expected_repository, expected_revision, expected_kind, expected_artifacts = (
            _EXPECTED_SOURCES[source_id]
        )
        if (repository, artifact_kind) != (expected_repository, expected_kind):
            raise ValueError(f"acquisition source contract mismatch for {source_id}")
        if (
            revision != source.resolved_revision
            or _SHA40.fullmatch(revision) is None
            or revision != expected_revision
        ):
            raise ValueError(f"{source_id} does not use its approved resolved revision")
        artifacts: list[_LocatedArtifact] = []
        for artifact in source.artifacts:
            source_path = artifact.source_path
            local_path = artifact.local_path
            artifact_sha256 = artifact.sha256
            size_bytes = artifact.size_bytes
            expected_artifact = expected_artifacts.get(local_path)
            if expected_artifact != (source_path, artifact_sha256, size_bytes):
                raise ValueError(f"acquisition artifact identity is not approved: {local_path}")
            relative = PurePosixPath(local_path)
            if relative.is_absolute() or ".." in relative.parts or local_path in seen_local_paths:
                raise ValueError(f"unsafe or duplicate acquisition local_path {local_path!r}")
            path = acquisition_directory.joinpath(*relative.parts)
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(acquisition_root):
                raise ValueError(f"acquisition artifact escapes its root: {local_path}")
            if (
                _SHA256.fullmatch(artifact_sha256) is None
                or not path.is_file()
                or path.stat().st_size != size_bytes
                or file_hash(path) != artifact_sha256
            ):
                raise ValueError(f"acquisition artifact does not match its receipt: {local_path}")
            seen_local_paths.add(local_path)
            artifacts.append(
                _LocatedArtifact(
                    source_path,
                    local_path,
                    artifact_sha256,
                    size_bytes,
                    path,
                )
            )
        if {item.local_path for item in artifacts} != set(expected_artifacts):
            raise ValueError(f"acquisition artifact set mismatch for {source_id}")
        seen_source_ids.add(source_id)
        sources.append(
            _LocatedSource(
                source_id,
                repository,
                revision,
                artifact_kind,
                tuple(artifacts),
            )
        )
    if seen_source_ids != set(_EXPECTED_SOURCES):
        raise ValueError(
            f"acquisition catalog source set mismatch: "
            f"{sorted(set(_EXPECTED_SOURCES).symmetric_difference(seen_source_ids))}"
        )
    return receipt.config_sha256, tuple(sources)


def _clean_text(value: str) -> _CleanedText:
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    removed_control_characters = sum(
        ord(character) in _REMOVABLE_CONTROL_CHARACTERS for character in text
    )
    if removed_control_characters:
        text = "".join(
            character for character in text if ord(character) not in _REMOVABLE_CONTROL_CHARACTERS
        )
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            cleaned_lines.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
    return _CleanedText("\n".join(cleaned_lines), removed_control_characters)


def _content_rejection_reason(text: str) -> str | None:
    if not text:
        return "empty_text"
    if not any(character.isalpha() for character in text):
        return "no_alphabetic_content"
    return None


def _poetry_rejection_reason(text: str, *, title: str | None) -> str | None:
    if title is not None and _TITLE_LABELLED_PROSE.search(title):
        return "title_labelled_prose"
    reason = _content_rejection_reason(text)
    if reason is not None:
        return reason
    if _PURE_NOTES_PREFIX.match(text):
        return "editorial_notes"
    prefix = text.lstrip("[]*_ \t").casefold()
    if any(prefix.startswith(marker) for marker in _EDITORIAL_NOTE_PREFIXES):
        return "editorial_note"
    return None


def _strip_leading_markdown_introduction(text: str) -> tuple[str, bool]:
    introduction = _LEADING_MARKDOWN_INTRODUCTION.match(text)
    if introduction is None:
        return text, False
    next_heading = _MARKDOWN_LEVEL_TWO_HEADING.search(text, introduction.end())
    if next_heading is None:
        return text, False
    return text[next_heading.start() :], True


def _parquet_rows(
    artifact: _LocatedArtifact, *, expected_columns: frozenset[str]
) -> Iterator[dict[str, object]]:
    parquet_file = parquet.ParquetFile(artifact.path)
    columns = cast(list[str], parquet_file.schema_arrow.names)
    if set(columns) != set(expected_columns):
        raise ValueError(
            f"{artifact.local_path} columns must be exactly {sorted(expected_columns)}"
        )
    for batch in parquet_file.iter_batches(batch_size=64):
        rows = cast(list[object], batch.to_pylist())
        for row in rows:
            yield _object(row, name=f"Parquet row in {artifact.local_path}")


def _json_rows(artifact: _LocatedArtifact) -> Iterator[dict[str, object]]:
    raw_value: object = json.loads(artifact.path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, list):
        raise ValueError(f"{artifact.local_path} must contain one JSON array")
    for index, row in enumerate(cast(list[object], raw_value)):
        yield _object(row, name=f"{artifact.local_path} row {index}")


def _source_by_id(sources: Iterable[_LocatedSource]) -> dict[str, _LocatedSource]:
    return {source.source_id: source for source in sources}


def _origin(
    source: _LocatedSource,
    artifact: _LocatedArtifact,
    *,
    row_index: int,
    title: str | None,
    author: str,
    source_locator: str,
    edition: str,
) -> _Origin:
    return _Origin(
        source.source_id,
        source.repository,
        source.revision,
        artifact.source_path,
        artifact.local_path,
        artifact.sha256,
        row_index,
        title,
        author,
        source_locator,
        edition,
    )


def _poetry_candidates(
    sources: Mapping[str, _LocatedSource],
) -> tuple[
    dict[str, list[_PoemCandidate]],
    list[_Rejection],
    Counter[str],
    Counter[str],
]:
    candidates: dict[str, list[_PoemCandidate]] = defaultdict(list)
    rejections: list[_Rejection] = []
    input_counts: Counter[str] = Counter()
    sanitized_control_rows: Counter[str] = Counter()
    dan = sources["public_domain_poetry"]
    dan_artifact = dan.artifacts[0]
    for row_index, row in enumerate(_json_rows(dan_artifact)):
        input_counts[dan.source_id] += 1
        data = _exact_object(
            row,
            name=f"{dan_artifact.local_path} row {row_index}",
            expected={"Author", "Title", "text"},
        )
        author = _normalise_identity(
            _non_empty_string(data["Author"], name=f"DanFosing row {row_index} Author")
        )
        raw_title = data["Title"]
        if not isinstance(raw_title, str):
            raise ValueError(f"DanFosing row {row_index} Title must be a string")
        title = _normalise_identity(raw_title) if raw_title.strip() else None
        raw_text = data["text"]
        if not isinstance(raw_text, str):
            raise ValueError(f"DanFosing row {row_index} text must be a string")
        cleaned = _clean_text(raw_text)
        text = cleaned.text
        if cleaned.removed_control_characters:
            sanitized_control_rows[dan.source_id] += 1
        reason = _poetry_rejection_reason(text, title=title)
        if reason is not None:
            rejections.append(_Rejection(dan.source_id, dan_artifact.local_path, row_index, reason))
            continue
        text_hash = sha256(text.encode()).hexdigest()
        candidates[text_hash].append(
            _PoemCandidate(
                raw_text,
                text,
                _origin(
                    dan,
                    dan_artifact,
                    row_index=row_index,
                    title=title,
                    author=author,
                    source_locator=(
                        f"hf://datasets/{dan.repository}@{dan.revision}/"
                        f"{dan_artifact.source_path}#row={row_index}"
                    ),
                    edition="",
                ),
                cleaned.removed_control_characters,
            )
        )

    yoon = sources["poetry_greats_public_domain"]
    yoon_artifact = yoon.artifacts[0]
    yoon_columns = frozenset(
        {
            "author",
            "book_title",
            "gutenberg_id",
            "poem_title",
            "poem_text",
            "line_count",
            "word_count",
        }
    )
    for row_index, row in enumerate(_parquet_rows(yoon_artifact, expected_columns=yoon_columns)):
        input_counts[yoon.source_id] += 1
        data = _exact_object(
            row,
            name=f"{yoon_artifact.local_path} row {row_index}",
            expected={
                "author",
                "book_title",
                "gutenberg_id",
                "poem_title",
                "poem_text",
                "line_count",
                "word_count",
            },
        )
        author = _normalise_identity(
            _non_empty_string(data["author"], name=f"yoon row {row_index} author")
        )
        book_title = _normalise_identity(
            _non_empty_string(data["book_title"], name=f"yoon row {row_index} book_title")
        )
        gutenberg_id = _integer(
            data["gutenberg_id"], name=f"yoon row {row_index} gutenberg_id", minimum=1
        )
        _integer(data["line_count"], name=f"yoon row {row_index} line_count")
        _integer(data["word_count"], name=f"yoon row {row_index} word_count")
        raw_title = data["poem_title"]
        if raw_title is not None and not isinstance(raw_title, str):
            raise ValueError(f"yoon row {row_index} poem_title must be string or null")
        title = (
            _normalise_identity(raw_title)
            if isinstance(raw_title, str) and raw_title.strip()
            else None
        )
        raw_text = data["poem_text"]
        if not isinstance(raw_text, str):
            raise ValueError(f"yoon row {row_index} poem_text must be a string")
        cleaned = _clean_text(raw_text)
        text = cleaned.text
        if cleaned.removed_control_characters:
            sanitized_control_rows[yoon.source_id] += 1
        reason = _poetry_rejection_reason(text, title=title)
        if reason is not None:
            rejections.append(
                _Rejection(yoon.source_id, yoon_artifact.local_path, row_index, reason)
            )
            continue
        text_hash = sha256(text.encode()).hexdigest()
        candidates[text_hash].append(
            _PoemCandidate(
                raw_text,
                text,
                _origin(
                    yoon,
                    yoon_artifact,
                    row_index=row_index,
                    title=title,
                    author=author,
                    source_locator=f"https://www.gutenberg.org/ebooks/{gutenberg_id}",
                    edition=book_title,
                ),
                cleaned.removed_control_characters,
            )
        )
    return candidates, rejections, input_counts, sanitized_control_rows


def _poem_blocks(
    document_id: str, poem_id: str, text: str, title: str | None
) -> tuple[ContentBlock, ...]:
    blocks: list[ContentBlock] = [
        ContentBlock(
            f"{document_id}:poem",
            "poem",
            text,
            poem_id=poem_id,
            title=title,
            start_char=0,
            end_char=len(text),
        )
    ]
    for stanza_index, match in enumerate(
        re.finditer(r"[^\n](?:.*?[^\n])?(?=\n{2,}|\Z)", text, re.DOTALL)
    ):
        blocks.append(
            ContentBlock(
                f"{document_id}:stanza:{stanza_index}",
                "stanza",
                match.group(0),
                poem_id=poem_id,
                stanza_index=stanza_index,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return tuple(blocks)


def _is_usable_prompt_title(title: str | None) -> bool:
    return (
        title is not None
        and len(title) <= _MAX_PROMPT_TITLE_CHARACTERS
        and "[" not in title
        and "]" not in title
        and _NUMERIC_OR_ROMAN_TITLE.fullmatch(title) is None
        and _EDITORIAL_TITLE.search(title) is None
    )


def _is_usable_prompt_author(author: str) -> bool:
    return author.casefold() not in _UNUSABLE_AUTHORS


def _prompt_from_origin(origin: _Origin) -> tuple[str, PromptMethod, str]:
    if _is_usable_prompt_title(origin.title):
        assert origin.title is not None
        return (
            f"Write a poem titled: {origin.title}",
            "title",
            (f"Deterministic source-title prompt from {origin.source_id} row {origin.row_index}."),
        )
    if _is_usable_prompt_author(origin.author):
        return (
            f"Write a poem in the style of {origin.author}.",
            "author_style",
            (
                f"Deterministic source-author prompt from "
                f"{origin.source_id} row {origin.row_index}; "
                "the source title was missing or unsuitable for conditioning."
            ),
        )
    return (
        "Write a poem.",
        "generic",
        (
            f"Generic prompt for {origin.source_id} row {origin.row_index}; "
            "neither its source title nor author was suitable for conditioning."
        ),
    )


def _poetry_documents_and_prompts(
    candidates: Mapping[str, list[_PoemCandidate]],
) -> tuple[tuple[SourceDocument, ...], tuple[PromptRecord, ...], int, Counter[str]]:
    documents: list[SourceDocument] = []
    prompts: list[PromptRecord] = []
    duplicate_rows = 0
    prompt_strategy_counts: Counter[str] = Counter()
    source_priority = {"poetry_greats_public_domain": 0, "public_domain_poetry": 1}
    for text_hash, group in sorted(candidates.items()):
        ordered = sorted(
            group,
            key=lambda item: (
                source_priority[item.origin.source_id],
                item.origin.artifact_local_path,
                item.origin.row_index,
            ),
        )
        primary = ordered[0]
        origins = tuple(item.origin for item in ordered)
        duplicate_rows += len(origins) - 1
        document_id = f"poem-{text_hash}"
        poem_id = document_id
        title = primary.origin.title
        work = title or "Untitled poem (source title absent)"
        is_yoon = primary.origin.source_id == "poetry_greats_public_domain"
        provenance = Provenance(
            work=work,
            author=primary.origin.author,
            licence=(
                "CC0-1.0; source text asserted public domain in the United States"
                if is_yoon
                else "CC0-1.0 dataset tag; underlying source edition unspecified"
            ),
            source=primary.origin.repository,
            edition=primary.origin.edition,
            source_locator=primary.origin.source_locator,
            page_or_section=(
                f"{primary.origin.artifact_source_path}#row={primary.origin.row_index}"
            ),
            rights_evidence=(
                f"https://huggingface.co/datasets/{primary.origin.repository}/tree/"
                f"{primary.origin.revision}"
                if is_yoon
                else ""
            ),
            rights_notes=(
                (
                    "The dataset card asserts public-domain status in the United States; "
                    "that does not establish public-domain status in every jurisdiction."
                )
                if is_yoon
                else "The dataset does not identify each poem's source edition."
            ),
            rights_status="unknown",
        )
        metadata = {
            "raw_content_sha256": sha256(primary.raw_text.encode()).hexdigest(),
            "cleaned_content_sha256": text_hash,
            "origin_count": str(len(origins)),
            "origins_json": canonical_json([origin.to_mapping() for origin in origins]),
            "primary_source_id": primary.origin.source_id,
            "source_repository": primary.origin.repository,
            "source_revision": primary.origin.revision,
            "artifact_sha256": primary.origin.artifact_sha256,
            "removed_control_character_count": str(primary.removed_control_characters),
        }
        lineage = (
            "source_record_parse",
            "unicode_nfc",
            "newline_normalize",
            *(
                ("strip_removable_control_characters",)
                if primary.removed_control_characters
                else ()
            ),
            "trim_trailing_whitespace",
            "trim_outer_blank_lines",
            "collapse_blank_line_runs",
            "exact_normalized_text_dedupe",
            "stanza_segment",
        )
        documents.append(
            SourceDocument(
                document_id,
                provenance,
                primary.text,
                _poem_blocks(document_id, poem_id, primary.text, title),
                source_path=(
                    f"{primary.origin.artifact_local_path}#row={primary.origin.row_index}"
                ),
                metadata=metadata,
                raw_text=primary.raw_text,
                transformation_lineage=lineage,
            )
        )
        prompt, prompt_strategy, source_attribution = _prompt_from_origin(primary.origin)
        prompt_strategy_counts[prompt_strategy] += 1
        prompts.append(
            PromptRecord(
                prompt_id=f"prompt-{text_hash}",
                document_id=document_id,
                prompt=prompt,
                method=prompt_strategy,
                source_attribution=source_attribution,
                poem_id=poem_id,
            )
        )
    return tuple(documents), tuple(prompts), duplicate_rows, prompt_strategy_counts


def _paragraph_blocks(document_id: str, text: str) -> tuple[ContentBlock, ...]:
    matches = tuple(re.finditer(r"[^\n](?:.*?[^\n])?(?=\n{2,}|\Z)", text, re.DOTALL))
    if not matches:
        raise ValueError(f"selected prose document {document_id} has no paragraphs")
    return tuple(
        ContentBlock(
            f"{document_id}:paragraph:{paragraph_index}",
            "paragraph",
            match.group(0),
            paragraph_index=paragraph_index,
            start_char=match.start(),
            end_char=match.end(),
        )
        for paragraph_index, match in enumerate(matches)
    )


def _standard_ebooks_documents(
    source: _LocatedSource,
    selection: StandardEbooksSelection,
) -> tuple[tuple[SourceDocument, ...], list[_Rejection], int, int, int]:
    selected = set(selection.works)
    matched: dict[SelectedWork, list[tuple[_LocatedArtifact, int, dict[str, object]]]] = (
        defaultdict(list)
    )
    input_count = 0
    standard_columns = frozenset({"link", "title", "author", "text", "language"})
    for artifact in sorted(source.artifacts, key=lambda item: item.local_path):
        for row_index, row in enumerate(_parquet_rows(artifact, expected_columns=standard_columns)):
            input_count += 1
            data = _exact_object(
                row,
                name=f"{artifact.local_path} row {row_index}",
                expected={"link", "title", "author", "text", "language"},
            )
            title = _normalise_identity(
                _non_empty_string(data["title"], name=f"Standard Ebooks row {row_index} title")
            )
            author = _normalise_identity(
                _non_empty_string(data["author"], name=f"Standard Ebooks row {row_index} author")
            )
            _non_empty_string(data["link"], name=f"Standard Ebooks row {row_index} link")
            _non_empty_string(data["language"], name=f"Standard Ebooks row {row_index} language")
            if not isinstance(data["text"], str):
                raise ValueError(f"Standard Ebooks row {row_index} text must be a string")
            work = SelectedWork(title, author)
            if work in selected:
                matched[work].append((artifact, row_index, data))
    missing = sorted(selected.difference(matched))
    ambiguous = sorted(work for work, rows in matched.items() if len(rows) != 1)
    if missing or ambiguous:
        details = []
        if missing:
            details.append(f"missing {[(item.title, item.author) for item in missing]}")
        if ambiguous:
            details.append(f"ambiguous {[(item.title, item.author) for item in ambiguous]}")
        raise ValueError(f"Standard Ebooks selection did not resolve exactly: {'; '.join(details)}")
    documents: list[SourceDocument] = []
    rejections: list[_Rejection] = []
    introduction_removal_count = 0
    sanitized_control_row_count = 0
    for work in sorted(selected):
        artifact, row_index, data = matched[work][0]
        raw_text = cast(str, data["text"])
        cleaned = _clean_text(raw_text)
        text, introduction_removed = _strip_leading_markdown_introduction(cleaned.text)
        if cleaned.removed_control_characters:
            sanitized_control_row_count += 1
        if introduction_removed:
            introduction_removal_count += 1
        reason = _content_rejection_reason(text)
        if reason is not None:
            rejections.append(_Rejection(source.source_id, artifact.local_path, row_index, reason))
            continue
        link = cast(str, data["link"])
        language = cast(str, data["language"])
        identity_hash = sha256(f"{work.title}\0{work.author}\0{link}\0{text}".encode()).hexdigest()
        document_id = f"prose-{identity_hash}"
        documents.append(
            SourceDocument(
                document_id,
                Provenance(
                    work=work.title,
                    author=work.author,
                    licence=(
                        "Dataset marked public-domain/CC0; underlying edition and "
                        "translator are not supplied per row"
                    ),
                    source=source.repository,
                    source_locator=link,
                    page_or_section=f"{artifact.source_path}#row={row_index}",
                    rights_status="unknown",
                    rights_notes=(
                        "Dataset-level terms do not prove the rights status of this "
                        "specific edition or translation."
                    ),
                ),
                text,
                _paragraph_blocks(document_id, text),
                source_path=f"{artifact.local_path}#row={row_index}",
                metadata={
                    "language": language,
                    "source_id": source.source_id,
                    "source_repository": source.repository,
                    "source_revision": source.revision,
                    "artifact_sha256": artifact.sha256,
                    "source_row": str(row_index),
                    "source_link": link,
                    "raw_content_sha256": sha256(raw_text.encode()).hexdigest(),
                    "cleaned_content_sha256": sha256(text.encode()).hexdigest(),
                    "removed_control_character_count": str(cleaned.removed_control_characters),
                    "leading_markdown_introduction_removed": str(introduction_removed).lower(),
                },
                raw_text=raw_text,
                transformation_lineage=(
                    "source_record_parse",
                    "unicode_nfc",
                    "newline_normalize",
                    *(
                        ("strip_removable_control_characters",)
                        if cleaned.removed_control_characters
                        else ()
                    ),
                    "trim_trailing_whitespace",
                    "trim_outer_blank_lines",
                    "collapse_blank_line_runs",
                    *(("strip_leading_markdown_introduction",) if introduction_removed else ()),
                    "paragraph_segment",
                ),
            )
        )
    if rejections:
        rejected = [(item.artifact_local_path, item.row_index, item.reason) for item in rejections]
        raise ValueError(f"selected Standard Ebooks work failed content filters: {rejected}")
    return (
        tuple(documents),
        rejections,
        input_count,
        introduction_removal_count,
        sanitized_control_row_count,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_in_directory(
    *,
    acquisition_directory: Path,
    catalog_path: Path,
    selection_config_path: Path,
    output_directory: Path,
) -> CorpusBuildArtifact:
    acquisition_config_sha256, catalog_sources = _catalog_sources(
        acquisition_directory, catalog_path
    )
    sources = _source_by_id(catalog_sources)
    selection = load_standard_ebooks_selection(selection_config_path)
    (
        poetry_candidates,
        rejections,
        input_counts,
        sanitized_control_rows,
    ) = _poetry_candidates(sources)
    (
        poetry_documents,
        prompts,
        duplicate_rows,
        prompt_strategy_counts,
    ) = _poetry_documents_and_prompts(poetry_candidates)
    (
        prose_documents,
        prose_rejections,
        prose_input_count,
        introduction_removal_count,
        prose_sanitized_control_row_count,
    ) = _standard_ebooks_documents(sources["standardebooks"], selection)
    rejections.extend(prose_rejections)
    input_counts["standardebooks"] = prose_input_count
    sanitized_control_rows["standardebooks"] = prose_sanitized_control_row_count
    sanitized_rows_by_source = {
        source_id: sanitized_control_rows[source_id] for source_id in sorted(input_counts)
    }
    documents = tuple(
        sorted((*poetry_documents, *prose_documents), key=lambda item: item.document_id)
    )
    write_manifest(output_directory / "manifest.jsonl", documents)
    write_prompt_records(output_directory / "prompts.jsonl", prompts)
    write_thought_records(output_directory / "thoughts.jsonl", ())
    write_pairings(output_directory / "pairings.jsonl", ())
    rejection_counts = Counter(item.reason for item in rejections)
    rejection_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for rejection in rejections:
        rejection_counts_by_source[rejection.source_id][rejection.reason] += 1
    filters = {
        "common_rejections": ["empty_text", "no_alphabetic_content"],
        "poetry_rejections": [
            "editorial_note",
            "editorial_notes",
            "title_labelled_prose",
        ],
        "poetry_editorial_note_prefixes": list(_EDITORIAL_NOTE_PREFIXES),
        "poetry_pure_notes_prefix": _PURE_NOTES_PREFIX.pattern,
        "poetry_title_labelled_prose_pattern": _TITLE_LABELLED_PROSE.pattern,
    }
    transformations = {
        "removable_control_characters": (
            "C0 and C1 control characters other than tab and newline are removed "
            "before validation and deduplication"
        ),
        "standard_ebooks_leading_markdown_introduction": (
            "A leading level-two Introduction section is removed through the next level-two heading"
        ),
    }
    report: dict[str, object] = {
        "format_version": 1,
        "input_row_counts": dict(sorted(input_counts.items())),
        "document_counts": {
            "poetry": len(poetry_documents),
            "prose": len(prose_documents),
            "total": len(documents),
        },
        "prompt_count": len(prompts),
        "prompt_strategy_counts": dict(sorted(prompt_strategy_counts.items())),
        "thought_count": 0,
        "pairing_count": 0,
        "exact_duplicate_poetry_rows_removed": duplicate_rows,
        "poetry_origin_count": sum(len(group) for group in poetry_candidates.values()),
        "standard_ebooks_selected_works": [asdict(work) for work in selection.works],
        "filters": filters,
        "transformations": transformations,
        "transformation_counts": {
            "control_character_rows_sanitized": sum(sanitized_rows_by_source.values()),
            "control_character_rows_sanitized_by_source": sanitized_rows_by_source,
            "standard_ebooks_leading_introductions_removed": introduction_removal_count,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejection_counts_by_source": {
            source_id: dict(sorted(counts.items()))
            for source_id, counts in sorted(rejection_counts_by_source.items())
        },
        "rejections": [
            asdict(item)
            for item in sorted(
                rejections,
                key=lambda item: (
                    item.source_id,
                    item.artifact_local_path,
                    item.row_index,
                    item.reason,
                ),
            )
        ],
    }
    _write_json(output_directory / "corpus.report.json", report)
    output_names = (
        "manifest.jsonl",
        "prompts.jsonl",
        "thoughts.jsonl",
        "pairings.jsonl",
        "corpus.report.json",
    )
    receipt: dict[str, object] = {
        "format_version": 1,
        "acquisition_directory": str(acquisition_directory.resolve()),
        "acquisition_catalog": str(catalog_path.resolve()),
        "acquisition_catalog_sha256": file_hash(catalog_path),
        "acquisition_config_sha256": acquisition_config_sha256,
        "selection_config": str(selection_config_path.resolve()),
        "selection_config_sha256": file_hash(selection_config_path),
        "sources": [
            {
                "source_id": source.source_id,
                "repository": source.repository,
                "revision": source.revision,
                "artifact_kind": source.artifact_kind,
                "artifacts": [
                    {
                        "source_path": artifact.source_path,
                        "local_path": artifact.local_path,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                    for artifact in source.artifacts
                ],
            }
            for source in catalog_sources
        ],
        "counts": {
            "documents": len(documents),
            "prompts": len(prompts),
            "thoughts": 0,
            "pairings": 0,
        },
        "outputs": {name: {"sha256": file_hash(output_directory / name)} for name in output_names},
    }
    _write_json(output_directory / "corpus.receipt.json", receipt)
    return CorpusBuildArtifact(output_directory, report, receipt)


def build_corpus(
    *,
    acquisition_directory: Path,
    catalog_path: Path,
    selection_config_path: Path,
    output_directory: Path,
) -> CorpusBuildArtifact:
    """Build and atomically publish the canonical Track 1 corpus artifacts."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"corpus output is not empty: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.building-", dir=output_directory.parent)
    )
    expected = {
        "manifest.jsonl",
        "prompts.jsonl",
        "thoughts.jsonl",
        "pairings.jsonl",
        "corpus.report.json",
        "corpus.receipt.json",
    }
    try:
        _build_in_directory(
            acquisition_directory=acquisition_directory,
            catalog_path=catalog_path,
            selection_config_path=selection_config_path,
            output_directory=temporary,
        )
        actual = {path.name for path in temporary.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(f"corpus artifact file set mismatch: {sorted(actual ^ expected)}")
        if output_directory.exists():
            output_directory.rmdir()
        os.replace(temporary, output_directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    report = _object(
        json.loads((output_directory / "corpus.report.json").read_text(encoding="utf-8")),
        name="corpus report",
    )
    receipt = _object(
        json.loads((output_directory / "corpus.receipt.json").read_text(encoding="utf-8")),
        name="corpus receipt",
    )
    return CorpusBuildArtifact(output_directory, report, receipt)
