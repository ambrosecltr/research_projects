"""Build the three-source, provenance-preserving Track 1 corpus.

The sources intentionally have different roles.  Ultra-FineWeb is auxiliary
next-token prose, Gutenberg is contiguous book-family verse, and Poetry
Greats is the only source from which this module creates prompt-to-poem
relations.  Keeping those distinctions here prevents a line-mined Gutenberg
book from being quietly presented as a collection of poem-level targets.
"""

from __future__ import annotations

import importlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from heapq import heappush, heapreplace
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, cast

from poetry50m.config import canonical_json, file_hash
from poetry50m.trajectory._persistence import atomic_write

from .artifacts import write_pairings, write_prompt_records, write_thought_records
from .hf_sources import AcquiredArtifact, AcquiredSource, AcquisitionReceipt, verify_acquisition
from .schema import ContentBlock, PromptRecord, Provenance, SourceDocument

_REMOVABLE_CONTROLS = (frozenset(range(32)) | frozenset(range(127, 160))).difference({9, 10})
_CONTROL_TRANSLATION = {codepoint: None for codepoint in _REMOVABLE_CONTROLS}
_EDITORIAL_MARKER = re.compile(
    r"(?:\*{3}\s*(?:start|end)\s+of|project\s+gutenberg|"
    r"transcriber(?:'s|s)?\s+note|editor(?:'s|s)?\s+note|"
    r"end\s+of\s+(?:the\s+)?project\s+gutenberg\s+ebook)",
    re.IGNORECASE,
)
_GREATs_EDITORIAL_OPENING = re.compile(
    r"^(?:\[?\s*)?(?:editor(?:'s|s)?\s+note|transcrib(?:er|er's)\s+note|"
    r"from\s+the\s+handwriting|contents)\b",
    re.IGNORECASE,
)
_DOCUMENT_COMPONENT = re.compile(r"[^a-z0-9-]+")


class _RecordBatch(Protocol):
    def to_pylist(self) -> list[dict[str, object]]: ...


class _ParquetFile(Protocol):
    def iter_batches(self) -> Iterator[_RecordBatch]: ...


class _ParquetFileFactory(Protocol):
    def __call__(self, path: Path) -> _ParquetFile: ...


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    """The fixed identity and semantic role of one approved source."""

    source_id: str
    repository: str
    artifact_kind: str
    artifact_path: str
    role: str


@dataclass(frozen=True, slots=True)
class UltraFineWebSelection:
    """The reproducible, bounded UFW sample policy committed with a corpus build."""

    max_documents: int
    method: Literal["sha256_uid_priority_v1"]
    seed: str

    def __post_init__(self) -> None:
        if isinstance(self.max_documents, bool) or not isinstance(self.max_documents, int):
            raise TypeError("ultrafineweb_l3.max_documents must be an integer")
        if self.max_documents < 1:
            raise ValueError("ultrafineweb_l3.max_documents must be positive")
        if self.method != "sha256_uid_priority_v1":
            raise ValueError("unsupported ultrafineweb_l3 selection method")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("ultrafineweb_l3.seed must be a non-empty string")


@dataclass(frozen=True, slots=True)
class KnowledgeCorpusSelection:
    """Validated external selection policy for sources too large to ingest wholesale."""

    format_version: int
    ultrafineweb_l3: UltraFineWebSelection

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("knowledge corpus selection format_version must be 1")

    @classmethod
    def from_mapping(cls, value: object) -> KnowledgeCorpusSelection:
        if not isinstance(value, dict) or set(value) != {"format_version", "ultrafineweb_l3"}:
            raise ValueError(
                "knowledge corpus selection must contain exactly format_version and ultrafineweb_l3"
            )
        format_version = value["format_version"]
        selection = value["ultrafineweb_l3"]
        if isinstance(format_version, bool) or not isinstance(format_version, int):
            raise TypeError("knowledge corpus selection format_version must be an integer")
        if not isinstance(selection, dict) or set(selection) != {"max_documents", "method", "seed"}:
            raise ValueError(
                "ultrafineweb_l3 selection must contain exactly max_documents, method, and seed"
            )
        max_documents = selection["max_documents"]
        method = selection["method"]
        seed = selection["seed"]
        if not isinstance(method, str) or not isinstance(seed, str):
            raise TypeError("ultrafineweb_l3 selection method and seed must be strings")
        if method != "sha256_uid_priority_v1":
            raise ValueError("unsupported ultrafineweb_l3 selection method")
        return cls(
            format_version=format_version,
            ultrafineweb_l3=UltraFineWebSelection(
                max_documents=max_documents,
                method="sha256_uid_priority_v1",
                seed=seed,
            ),
        )


def load_knowledge_corpus_selection(path: Path) -> KnowledgeCorpusSelection:
    """Load the committed UFW selection policy without accepting loose JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid knowledge corpus selection JSON: {path}") from error
    return KnowledgeCorpusSelection.from_mapping(value)


@dataclass(frozen=True, slots=True)
class _MaxPriorityCandidate:
    """Reverse heap ordering so its root is the least desirable selected row."""

    key: tuple[bytes, str, int]
    row_index: int

    def __lt__(self, other: _MaxPriorityCandidate) -> bool:
        return self.key > other.key


_SOURCES = (
    # Poetry Greats is first deliberately: if a future exact document-level
    # collision appears, retain the prompt-bearing record rather than silently
    # discarding our only conditional target.
    KnowledgeSource(
        source_id="poetry_greats",
        repository="yoonholee/poetry-greats-public-domain",
        artifact_kind="poetry_greats_parquet",
        artifact_path="data/train-00000-of-00001.parquet",
        role="conditional_poetry",
    ),
    KnowledgeSource(
        source_id="gutenberg_poetry",
        repository="biglam/gutenberg-poetry-corpus",
        artifact_kind="gutenberg_poetry_parquet",
        artifact_path="data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet",
        role="unconditional_book_verse_ntp",
    ),
    KnowledgeSource(
        source_id="ultrafineweb_l3",
        repository="openbmb/Ultra-FineWeb-L3",
        artifact_kind="ultrafineweb_multistyle_parquet",
        artifact_path=(
            "data/ultrafineweb_en_l3/multi_style/"
            "part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet"
        ),
        role="auxiliary_prose_ntp",
    ),
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exact_record(
    record: Mapping[str, object], *, source_id: str, row_index: int, keys: set[str]
) -> None:
    actual = set(record)
    if actual != keys:
        raise ValueError(
            f"{source_id} row {row_index} must contain exactly {sorted(keys)}; got {sorted(actual)}"
        )


def _clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return normalized.translate(_CONTROL_TRANSLATION)


def _normalized_exact_text(text: str) -> str:
    """A conservative, format-insensitive key used only for exact duplicate removal."""
    return " ".join(_clean_text(text).casefold().split())


def _content_key(text: str) -> str:
    return sha256(_normalized_exact_text(text).encode("utf-8")).hexdigest()


def _parquet_records(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield Parquet rows without materialising a source in memory.

    ``pyarrow`` is deliberately imported here: corpus inspection and the rest
    of the package remain usable when the optional raw-corpus reader is absent.
    The pinned training environment installs it as a project dependency.
    """
    try:
        parquet_module = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "building the acquired Parquet corpus requires pyarrow; install the project "
            "training dependencies before running this command"
        ) from error
    factory = cast(_ParquetFileFactory, parquet_module.ParquetFile)
    parquet = factory(path)
    row_index = 0
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
                raise ValueError(f"Parquet row {row_index} in {path} must be a string-keyed object")
            yield row_index, row
            row_index += 1


def _acquired_sources(receipt: AcquisitionReceipt) -> dict[str, AcquiredSource]:
    sources = {source.source_id: source for source in receipt.sources}
    expected = {source.source_id for source in _SOURCES}
    if set(sources) != expected:
        raise ValueError(
            f"acquisition source IDs differ: expected={sorted(expected)}, got={sorted(sources)}"
        )
    return sources


def _artifact_path(
    acquisition_directory: Path, source: KnowledgeSource, acquired: AcquiredSource
) -> Path:
    if acquired.repository != source.repository or acquired.artifact_kind != source.artifact_kind:
        raise ValueError(f"unexpected acquisition identity for {source.source_id}")
    artifacts = [
        artifact for artifact in acquired.artifacts if artifact.source_path == source.artifact_path
    ]
    if len(artifacts) != 1:
        raise ValueError(f"expected one approved Parquet artifact for {source.source_id}")
    artifact = artifacts[0]
    if not isinstance(artifact, AcquiredArtifact):
        raise TypeError(f"invalid acquired artifact for {source.source_id}")
    path = acquisition_directory / artifact.local_path
    if path.suffix != ".parquet":
        raise ValueError(f"{source.source_id} artifact must remain Parquet")
    return path


def _locator(source: KnowledgeSource, acquired: AcquiredSource) -> str:
    return f"{acquired.repository}@{acquired.resolved_revision}:{source.artifact_path}"


def _ultrafineweb_provenance(
    source: KnowledgeSource, acquired: AcquiredSource, *, uid: str
) -> Provenance:
    return Provenance(
        work=f"Ultra-FineWeb-L3 Multi-Style record {uid}",
        author="OpenBMB; generated using MiniCPM4 and Qwen3",
        licence="Apache-2.0 dataset licence; upstream web-source rights are not row-level retained",
        source=source.repository,
        source_locator=_locator(source, acquired),
        rights_status="unknown",
        rights_notes=(
            "This is a synthetic multi-style rewrite. The dataset card declares Apache-2.0, "
            "but the source row does not preserve the underlying web document's rights."
        ),
    )


def _gutenberg_provenance(
    source: KnowledgeSource, acquired: AcquiredSource, *, gutenberg_id: int
) -> Provenance:
    return Provenance(
        work=f"Project Gutenberg book {gutenberg_id}",
        author="Not supplied by the line corpus",
        licence="CC0-1.0 corpus release; individual work rights require catalog verification",
        source=source.repository,
        source_locator=_locator(source, acquired),
        page_or_section=f"gutenberg_id={gutenberg_id}",
        rights_status="unknown",
        rights_notes=(
            "The line corpus has no per-book author, title, or rights record. This document "
            "must not be claimed public domain until its Gutenberg catalog record is joined."
        ),
    )


def _greats_provenance(
    source: KnowledgeSource,
    acquired: AcquiredSource,
    *,
    author: str,
    book_title: str,
    gutenberg_id: int,
    poem_title: str | None,
) -> Provenance:
    work = poem_title or f"Untitled poem in {book_title}"
    return Provenance(
        work=work,
        author=author,
        licence="CC0-1.0 dataset release; public-domain source collection",
        source=source.repository,
        edition=book_title,
        source_locator=_locator(source, acquired),
        page_or_section=f"gutenberg_id={gutenberg_id}",
        rights_status="public_domain",
        rights_evidence="https://huggingface.co/datasets/yoonholee/poetry-greats-public-domain",
        rights_notes="Dataset card identifies this as a public-domain poetry collection.",
    )


def _document_component(value: str) -> str:
    component = _DOCUMENT_COMPONENT.sub("-", value.casefold()).strip("-")
    return component or sha256(value.encode("utf-8")).hexdigest()[:16]


def _gutenberg_document(
    source: KnowledgeSource,
    acquired: AcquiredSource,
    *,
    gutenberg_id: int,
    lines: list[str],
    source_row_start: int,
    source_row_end: int,
) -> SourceDocument | None:
    text = "\n".join(lines)
    if not text.strip():
        return None
    document_id = f"gutenberg-book-{gutenberg_id}"
    return SourceDocument(
        document_id=document_id,
        provenance=_gutenberg_provenance(source, acquired, gutenberg_id=gutenberg_id),
        text=text,
        blocks=(
            ContentBlock(
                block_id=f"{document_id}:book:0",
                kind="verse_document",
                text=text,
                start_char=0,
                end_char=len(text),
                metadata={"source_unit": "ordered_line_corpus"},
            ),
        ),
        source_path=f"{source.source_id}/{source.artifact_path}",
        raw_text=text,
        metadata={
            "gutenberg_id": str(gutenberg_id),
            "leakage_family_id": f"gutenberg:{gutenberg_id}",
            "source_row_start": str(source_row_start),
            "source_row_end": str(source_row_end),
            "training_role": source.role,
        },
        transformation_lineage=(
            "parquet_row_decode",
            "ordered_gutenberg_id_grouping",
            "unicode_nfc",
            "control_character_filter",
        ),
    )


def _greats_document(
    source: KnowledgeSource,
    acquired: AcquiredSource,
    *,
    row_index: int,
    record: Mapping[str, object],
) -> tuple[SourceDocument | None, PromptRecord | None, str | None]:
    _exact_record(
        record,
        source_id=source.source_id,
        row_index=row_index,
        keys={
            "author",
            "book_title",
            "gutenberg_id",
            "poem_title",
            "poem_text",
            "line_count",
            "word_count",
        },
    )
    author = _clean_text(_required_text(record["author"], name="poetry_greats author")).strip()
    book_title = _clean_text(
        _required_text(record["book_title"], name="poetry_greats book_title")
    ).strip()
    gutenberg_id = _required_integer(record["gutenberg_id"], name="poetry_greats gutenberg_id")
    _required_integer(record["line_count"], name="poetry_greats line_count")
    _required_integer(record["word_count"], name="poetry_greats word_count")
    raw_text = _required_text(record["poem_text"], name="poetry_greats poem_text")
    text = _clean_text(raw_text).strip()
    title_value = record["poem_title"]
    if title_value is not None and not isinstance(title_value, str):
        raise ValueError("poetry_greats poem_title must be a string or null")
    poem_title = _clean_text(title_value).strip() if isinstance(title_value, str) else None
    if poem_title == "":
        poem_title = None
    if not text:
        return None, None, "empty"
    if _GREATs_EDITORIAL_OPENING.search(text):
        return None, None, "editorial_residue"
    document_id = f"poetry-greats-{row_index:08d}"
    poem_id = f"poetry-greats:{gutenberg_id}:{row_index:08d}"
    block_id = f"{document_id}:poem:0"
    document = SourceDocument(
        document_id=document_id,
        provenance=_greats_provenance(
            source,
            acquired,
            author=author,
            book_title=book_title,
            gutenberg_id=gutenberg_id,
            poem_title=poem_title,
        ),
        text=text,
        blocks=(
            ContentBlock(
                block_id=block_id,
                kind="poem",
                text=text,
                poem_id=poem_id,
                title=poem_title,
                start_char=0,
                end_char=len(text),
            ),
        ),
        source_path=f"{source.source_id}/{source.artifact_path}",
        raw_text=raw_text,
        metadata={
            "author": author,
            "book_title": book_title,
            "gutenberg_id": str(gutenberg_id),
            "leakage_family_id": f"gutenberg:{gutenberg_id}",
            "source_row": str(row_index),
            "training_role": source.role,
        },
        transformation_lineage=("parquet_row_decode", "unicode_nfc", "control_character_filter"),
    )
    if poem_title is not None:
        prompt = f'Write a poem titled "{poem_title}".'
        method: Literal["title", "author_style"] = "title"
    else:
        prompt = f"Write a poem in the style of {author}."
        method = "author_style"
    prompt_record = PromptRecord(
        prompt_id=f"{document_id}:prompt:0",
        document_id=document_id,
        prompt=prompt,
        method=method,
        source_attribution=(
            "Derived only from the Poetry Greats poem title"
            if poem_title is not None
            else "Author-name fallback because Poetry Greats supplies no poem title"
        ),
        poem_id=poem_id,
    )
    return document, prompt_record, None


def _ultrafineweb_document(
    source: KnowledgeSource,
    acquired: AcquiredSource,
    *,
    row_index: int,
    record: Mapping[str, object],
) -> tuple[SourceDocument | None, str | None]:
    _exact_record(
        record,
        source_id=source.source_id,
        row_index=row_index,
        keys={"uid", "content", "style"},
    )
    uid = _clean_text(_required_text(record["uid"], name="ultrafineweb_l3 uid")).strip()
    raw_content = _required_text(record["content"], name="ultrafineweb_l3 content")
    content = _clean_text(raw_content).strip()
    style = _clean_text(_required_text(record["style"], name="ultrafineweb_l3 style")).strip()
    if style != "multi_style":
        raise ValueError(f"ultrafineweb_l3 row {row_index} has unexpected style {style!r}")
    if not content:
        return None, "empty"
    if _EDITORIAL_MARKER.search(content):
        return None, "editorial_residue"
    document_id = f"ultrafineweb-l3-{_document_component(uid)}"
    return (
        SourceDocument(
            document_id=document_id,
            provenance=_ultrafineweb_provenance(source, acquired, uid=uid),
            text=content,
            blocks=(
                ContentBlock(
                    block_id=f"{document_id}:paragraph:0",
                    kind="paragraph",
                    text=content,
                    paragraph_index=0,
                    start_char=0,
                    end_char=len(content),
                ),
            ),
            source_path=f"{source.source_id}/{source.artifact_path}",
            raw_text=raw_content,
            metadata={
                "source_row": str(row_index),
                "source_uid": uid,
                "style": style,
                "training_role": source.role,
            },
            transformation_lineage=(
                "parquet_row_decode",
                "unicode_nfc",
                "control_character_filter",
            ),
        ),
        None,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    def write(handle: BinaryIO) -> None:
        handle.write(f"{canonical_json(value)}\n".encode())

    atomic_write(path, write)


def _ultrafineweb_priority(uid: str, *, seed: str) -> bytes:
    return sha256(f"{seed}\0{uid}".encode()).digest()


def _select_ultrafineweb_rows(
    *,
    source: KnowledgeSource,
    acquired: AcquiredSource,
    acquisition_directory: Path,
    selection: UltraFineWebSelection,
    rejection_counts: Counter[str],
) -> tuple[frozenset[int], dict[str, object]]:
    """Select the smallest UID priorities in one bounded-memory scan.

    The selected row indices are revisited in a second pass for manifest
    materialisation.  This avoids retaining up to 200,000 source texts in
    memory while ensuring a first-N artifact order never decides the sample.
    """
    heap: list[_MaxPriorityCandidate] = []
    eligible_count = 0
    path = _artifact_path(acquisition_directory, source, acquired)
    for row_index, record in _parquet_records(path):
        document, rejection = _ultrafineweb_document(
            source, acquired, row_index=row_index, record=record
        )
        if document is None:
            assert rejection is not None
            rejection_counts[rejection] += 1
            continue
        eligible_count += 1
        uid = document.metadata["source_uid"]
        candidate = _MaxPriorityCandidate(
            key=(_ultrafineweb_priority(uid, seed=selection.seed), uid, row_index),
            row_index=row_index,
        )
        if len(heap) < selection.max_documents:
            heappush(heap, candidate)
        elif candidate.key < heap[0].key:
            heapreplace(heap, candidate)
    selected = tuple(sorted(heap, key=lambda candidate: candidate.key))
    selected_rows = frozenset(candidate.row_index for candidate in selected)
    selection_hash = sha256(
        "\n".join(
            f"{candidate.key[0].hex()}\t{candidate.key[1]}\t{candidate.row_index}"
            for candidate in selected
        ).encode("utf-8")
    ).hexdigest()
    return selected_rows, {
        "method": selection.method,
        "seed": selection.seed,
        "max_documents": selection.max_documents,
        "eligible_count": eligible_count,
        "selected_count": len(selected_rows),
        "selected_priority_sha256": selection_hash,
    }


def build_knowledge_corpus(
    *,
    acquisition_directory: Path,
    sources_config: Path,
    selection_config: Path,
    output_directory: Path,
) -> Path:
    """Write a deterministic, auditable corpus from the verified three-source acquisition."""
    receipt = verify_acquisition(sources_config, acquisition_directory)
    selection = load_knowledge_corpus_selection(selection_config)
    acquired_by_id = _acquired_sources(receipt)
    input_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    rejection_counts: dict[str, Counter[str]] = {source.source_id: Counter() for source in _SOURCES}
    exact_duplicate_counts: Counter[str] = Counter()
    source_document_ids: dict[str, list[str]] = {source.source_id: [] for source in _SOURCES}
    leakage_families: dict[str, list[str]] = {}
    seen_document_ids: set[str] = set()
    seen_content_keys: set[str] = set()
    prompts: list[PromptRecord] = []
    ultrafineweb_selection_report: dict[str, object] = {}
    manifest_path = output_directory / "manifest.jsonl"

    def keep(document: SourceDocument, *, source: KnowledgeSource, handle: BinaryIO) -> bool:
        if document.document_id in seen_document_ids:
            raise ValueError(
                f"duplicate document ID after source conversion: {document.document_id}"
            )
        content_key = _content_key(document.text)
        if content_key in seen_content_keys:
            exact_duplicate_counts[source.source_id] += 1
            return False
        seen_document_ids.add(document.document_id)
        seen_content_keys.add(content_key)
        document_counts[source.source_id] += 1
        source_document_ids[source.source_id].append(document.document_id)
        family = document.metadata.get("leakage_family_id")
        if family is not None:
            leakage_families.setdefault(family, []).append(document.document_id)
        handle.write(f"{canonical_json(document.to_mapping())}\n".encode())
        return True

    def write_manifest_stream(handle: BinaryIO) -> None:
        greats = next(source for source in _SOURCES if source.source_id == "poetry_greats")
        acquired = acquired_by_id[greats.source_id]
        path = _artifact_path(acquisition_directory, greats, acquired)
        for row_index, record in _parquet_records(path):
            input_counts[greats.source_id] += 1
            document, prompt, rejection = _greats_document(
                greats, acquired, row_index=row_index, record=record
            )
            if document is None:
                assert rejection is not None
                rejection_counts[greats.source_id][rejection] += 1
            elif keep(document, source=greats, handle=handle):
                assert prompt is not None
                prompts.append(prompt)

        gutenberg = next(source for source in _SOURCES if source.source_id == "gutenberg_poetry")
        acquired = acquired_by_id[gutenberg.source_id]
        active_id: int | None = None
        active_lines: list[str] = []
        active_start = 0
        completed_ids: set[int] = set()
        path = _artifact_path(acquisition_directory, gutenberg, acquired)
        for row_index, record in _parquet_records(path):
            input_counts[gutenberg.source_id] += 1
            _exact_record(
                record,
                source_id=gutenberg.source_id,
                row_index=row_index,
                keys={"line", "gutenberg_id"},
            )
            gutenberg_id = _required_integer(record["gutenberg_id"], name="gutenberg_id")
            raw_line = record["line"]
            if not isinstance(raw_line, str):
                raise ValueError("gutenberg line must be a string")
            # A blank line is a meaningful stanza boundary in this line corpus.
            line = _clean_text(raw_line)
            if "\n" in line:
                raise ValueError(
                    f"gutenberg_poetry row {row_index} line must not contain a newline"
                )
            if active_id is None:
                active_id, active_start = gutenberg_id, row_index
            elif gutenberg_id != active_id:
                document = _gutenberg_document(
                    gutenberg,
                    acquired,
                    gutenberg_id=active_id,
                    lines=active_lines,
                    source_row_start=active_start,
                    source_row_end=row_index - 1,
                )
                if document is None:
                    rejection_counts[gutenberg.source_id]["empty_book"] += 1
                else:
                    keep(document, source=gutenberg, handle=handle)
                completed_ids.add(active_id)
                if gutenberg_id in completed_ids:
                    raise ValueError(
                        "gutenberg_id rows must be contiguous; reordering would destroy "
                        "source order"
                    )
                active_id, active_lines, active_start = gutenberg_id, [], row_index
            if _EDITORIAL_MARKER.search(line):
                rejection_counts[gutenberg.source_id]["editorial_line"] += 1
            else:
                active_lines.append(line)
        if active_id is not None:
            document = _gutenberg_document(
                gutenberg,
                acquired,
                gutenberg_id=active_id,
                lines=active_lines,
                source_row_start=active_start,
                source_row_end=input_counts[gutenberg.source_id] - 1,
            )
            if document is None:
                rejection_counts[gutenberg.source_id]["empty_book"] += 1
            else:
                keep(document, source=gutenberg, handle=handle)

        ultrafineweb = next(source for source in _SOURCES if source.source_id == "ultrafineweb_l3")
        acquired = acquired_by_id[ultrafineweb.source_id]
        selected_rows, ultrafineweb_selection = _select_ultrafineweb_rows(
            source=ultrafineweb,
            acquired=acquired,
            acquisition_directory=acquisition_directory,
            selection=selection.ultrafineweb_l3,
            rejection_counts=rejection_counts[ultrafineweb.source_id],
        )
        ultrafineweb_selection_report.update(ultrafineweb_selection)
        for row_index, record in _parquet_records(
            _artifact_path(acquisition_directory, ultrafineweb, acquired)
        ):
            input_counts[ultrafineweb.source_id] += 1
            if row_index not in selected_rows:
                continue
            document, rejection = _ultrafineweb_document(
                ultrafineweb, acquired, row_index=row_index, record=record
            )
            if document is None:
                raise RuntimeError(
                    "ultrafineweb_l3 row "
                    f"{row_index} changed between deterministic selection passes"
                )
            else:
                keep(document, source=ultrafineweb, handle=handle)

    atomic_write(manifest_path, write_manifest_stream)
    prompts_path = output_directory / "prompts.jsonl"
    thoughts_path = output_directory / "thoughts.jsonl"
    pairings_path = output_directory / "pairings.jsonl"
    write_prompt_records(prompts_path, prompts)
    write_thought_records(thoughts_path, ())
    write_pairings(pairings_path, ())

    report_path = output_directory / "knowledge.report.json"
    report: dict[str, object] = {
        "format_version": 2,
        "source_roles": {source.source_id: source.role for source in _SOURCES},
        "input_counts": dict(sorted(input_counts.items())),
        "document_counts": dict(sorted(document_counts.items())),
        "rejection_counts": {
            source_id: dict(sorted(counts.items()))
            for source_id, counts in sorted(rejection_counts.items())
        },
        "exact_normalized_duplicates_removed": dict(sorted(exact_duplicate_counts.items())),
        "conditional_prompt_count": len(prompts),
        "ultrafineweb_selection": ultrafineweb_selection_report,
        "selection_config_sha256": file_hash(selection_config),
        "source_document_ids": dict(sorted(source_document_ids.items())),
        "leakage_families": {
            family: document_ids
            for family, document_ids in sorted(leakage_families.items())
            if len(document_ids) > 1
        },
        "rights": {
            "ultrafineweb_l3": "dataset Apache-2.0; source-row web rights unknown",
            "gutenberg_poetry": (
                "CC0 corpus release; individual-book rights unverified without catalog join"
            ),
            "poetry_greats": "public-domain collection asserted by dataset card",
        },
        "limitations": [
            (
                "Gutenberg line rows are reconstructed only as contiguous books, never as "
                "poem-level targets."
            ),
            (
                "Gutenberg catalog metadata is not present in this acquisition; title, author, "
                "and per-book rights remain unverified."
            ),
            (
                "Ultra-FineWeb selection is the externally configured hash-priority sample "
                "over the pinned Parquet shard."
            ),
        ],
    }
    _write_json(report_path, report)
    receipt_path = output_directory / "knowledge.receipt.json"
    output_hashes = {
        path.name: file_hash(path)
        for path in (manifest_path, prompts_path, thoughts_path, pairings_path, report_path)
    }
    _write_json(
        receipt_path,
        {
            "format_version": 2,
            "sources_config_sha256": file_hash(sources_config),
            "selection_config_sha256": file_hash(selection_config),
            "acquisition_receipt_sha256": file_hash(
                acquisition_directory / "acquisition_receipt.json"
            ),
            "outputs": output_hashes,
        },
    )
    return receipt_path
